from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from llmops_demo.mlflow_models import AdapterArtifactModel
from llmops_demo.settings import ensure_dirs, settings


def format_messages(example: dict, tokenizer: AutoTokenizer) -> str:
    return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)


def train_adapter(adapter: str, cfg) -> None:
    dataset_path = cfg.dataset_dir / f"{adapter}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset {dataset_path}. Run `make datasets` first.")

    adapter_output = cfg.adapter_dir / adapter
    work_output = cfg.output_dir / "training" / adapter
    ensure_dirs(adapter_output, work_output)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    quantization_config = None
    if cfg.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    dataset = dataset.map(lambda row: {"text": format_messages(row, tokenizer)}, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=str(work_output),
        num_train_epochs=cfg.train_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=5,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
    )

    with mlflow.start_run(run_name=f"train-{adapter}") as run:
        mlflow.set_tags({"adapter_name": adapter, "base_model": cfg.base_model, "artifact_type": "peft_lora"})
        mlflow.log_params(
            {
                "lora_r": cfg.lora_r,
                "lora_alpha": cfg.lora_alpha,
                "lora_dropout": cfg.lora_dropout,
                "learning_rate": cfg.learning_rate,
                "epochs": cfg.train_epochs,
                "max_seq_length": cfg.max_seq_length,
            }
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=args,
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
        )
        trainer.train()
        trainer.model.save_pretrained(adapter_output)
        tokenizer.save_pretrained(adapter_output)
        mlflow.log_artifacts(str(adapter_output), artifact_path="adapter")
        mlflow.pyfunc.log_model(
            artifact_path="registered_adapter",
            python_model=AdapterArtifactModel(),
            artifacts={"adapter": str(adapter_output)},
            registered_model_name=f"{cfg.mlflow_registered_model_prefix}_{adapter}",
        )
        print(f"Saved adapter {adapter} to {adapter_output}")
        print(f"MLflow run_id={run.info.run_id}")


def main() -> None:
    cfg = settings()
    parser = argparse.ArgumentParser(description="Train standalone PEFT LoRA adapters.")
    parser.add_argument("--adapter", choices=list(cfg.adapters), help="Train one adapter. Defaults to all.")
    args = parser.parse_args()

    ensure_dirs(cfg.adapter_dir, cfg.output_dir)
    adapters = [args.adapter] if args.adapter else list(cfg.adapters)
    for adapter in adapters:
        train_adapter(adapter, cfg)


if __name__ == "__main__":
    main()
