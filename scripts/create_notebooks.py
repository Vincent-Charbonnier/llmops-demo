from __future__ import annotations

import json
import textwrap
from pathlib import Path


NOTEBOOK_DIR = Path("notebooks")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_SETUP = """
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "PROJECT_SPEC.md").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))

from llmops_demo.settings import settings, ensure_dirs

cfg = settings()
print(f"Project root: {PROJECT_ROOT}")
print(f"Base model: {cfg.base_model}")
print(f"Adapters: {cfg.adapters}")
"""


def write(name: str, cells: list[dict]) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_DIR / name
    path.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    write(
        "01_generate_datasets.ipynb",
        [
            md(
                """
                # 01 Generate Synthetic Datasets

                This notebook creates small local supervised fine-tuning datasets for the finance, legal, and healthcare adapters.

                ```mermaid
                flowchart LR
                    A[Domain templates] --> B[Synthetic chat records]
                    B --> C[datasets/generated/finance.jsonl]
                    B --> D[datasets/generated/legal.jsonl]
                    B --> E[datasets/generated/healthcare.jsonl]
                ```

                The records use chat-style `system`, `user`, and `assistant` messages so the same files can be consumed by the Qwen chat template during LoRA training.
                """
            ),
            code(COMMON_SETUP),
            md("## Generate datasets\n\nExpected output: three JSONL files under `datasets/generated/`, one per adapter."),
            code(
                """
                import importlib.util

                module_path = PROJECT_ROOT / "datasets" / "generate_synthetic.py"
                spec = importlib.util.spec_from_file_location("generate_synthetic", module_path)
                generator = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(generator)

                records_per_domain = int(os.getenv("SYNTHETIC_RECORDS_PER_DOMAIN", "60"))
                ensure_dirs(cfg.dataset_dir)

                for adapter in cfg.adapters:
                    rows = generator.generate_domain(adapter, records_per_domain)
                    output_path = cfg.dataset_dir / f"{adapter}.jsonl"
                    generator.write_jsonl(output_path, rows)
                    print(f"{adapter}: wrote {len(rows)} records to {output_path}")
                """
            ),
            md("## Inspect a sample\n\nThe sample should show the domain-specific system prompt and a concise assistant answer."),
            code(
                """
                import json

                for adapter in cfg.adapters:
                    path = cfg.dataset_dir / f"{adapter}.jsonl"
                    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
                    print(f"\\n[{adapter}] {first['id']}")
                    for message in first["messages"]:
                        print(f"{message['role']}: {message['content'][:160]}")
                """
            ),
        ],
    )

    train_template = """
# 0{number} Train {Title} LoRA Adapter

This notebook trains the `{adapter}` standalone PEFT LoRA adapter and writes it to `adapters/{adapter}/`.

```mermaid
flowchart LR
    A[datasets/generated/{adapter}.jsonl] --> B[Qwen chat template]
    B --> C[PEFT LoRA training]
    C --> D[adapters/{adapter}/]
    C --> E[MLflow run]
```

The adapter is not merged into the base model. vLLM will load it later as a runtime LoRA adapter.
"""

    for number, adapter, title in [
        ("2", "finance", "Finance"),
        ("3", "legal", "Legal"),
        ("4", "healthcare", "Healthcare"),
    ]:
        write(
            f"0{number}_train_{adapter}_lora.ipynb",
            [
                md(train_template.format(number=number, adapter=adapter, Title=title)),
                code(COMMON_SETUP),
                md(
                    f"""
                    ## Preflight

                    Expected inputs:

                    - `datasets/generated/{adapter}.jsonl`
                    - access to `{cfg_placeholder()}`
                    - CUDA GPU recommended for practical runtime
                    """
                ),
                code(
                    f"""
                    dataset_path = cfg.dataset_dir / "{adapter}.jsonl"
                    adapter_path = cfg.adapter_dir / "{adapter}"
                    print(f"Dataset exists: {{dataset_path.exists()}} - {{dataset_path}}")
                    print(f"Adapter output: {{adapter_path}}")
                    """
                ),
                md("## Train\n\nThis calls the shared production training entry point, logs the run to MLflow, and saves the standalone PEFT adapter."),
                code(
                    f"""
                    from training.train_lora import train_adapter

                    train_adapter("{adapter}", cfg)
                    """
                ),
                md(
                    f"""
                    ## Verify adapter files

                    Expected output: `adapter_config.json`, adapter weights, tokenizer files, and related PEFT metadata under `adapters/{adapter}/`.
                    """
                ),
                code(
                    f"""
                    for path in sorted((cfg.adapter_dir / "{adapter}").glob("*")):
                        print(path)
                    """
                ),
            ],
        )

    write(
        "05_mlflow_tracking.ipynb",
        [
            md(
                """
                # 05 MLflow Tracking and Registry

                This notebook reviews training runs, registers existing local adapters, and compares adapter experiment metadata.

                ```mermaid
                flowchart LR
                    A[Training notebooks] --> B[MLflow experiment]
                    B --> C[Run params and metrics]
                    B --> D[Adapter artifacts]
                    D --> E[MLflow Model Registry]
                ```
                """
            ),
            code(COMMON_SETUP),
            md("## Configure MLflow\n\nFor local notebooks, `MLFLOW_TRACKING_URI=file:./mlruns` works without any service. With `make up`, use `http://localhost:5000`."),
            code(
                """
                import mlflow
                from mlflow.tracking import MlflowClient

                mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
                mlflow.set_experiment(cfg.mlflow_experiment_name)
                client = MlflowClient()
                experiment = client.get_experiment_by_name(cfg.mlflow_experiment_name)
                print("Tracking URI:", mlflow.get_tracking_uri())
                print("Experiment:", experiment.name if experiment else "not created yet")
                """
            ),
            md("## Register local adapters\n\nExpected output: one registered model per adapter, named with `MLFLOW_REGISTERED_MODEL_PREFIX`."),
            code(
                """
                from scripts.register_mlflow import register_local_adapter

                for adapter in cfg.adapters:
                    adapter_path = cfg.adapter_dir / adapter
                    if adapter_path.exists():
                        register_local_adapter(adapter, adapter_path, cfg)
                    else:
                        print(f"Skipping {adapter}: {adapter_path} does not exist yet")
                """
            ),
            md("## Compare experiment runs\n\nThe table shows recent runs, adapter tags, and common LoRA parameters."),
            code(
                """
                import pandas as pd

                experiment = client.get_experiment_by_name(cfg.mlflow_experiment_name)
                if experiment:
                    runs = mlflow.search_runs([experiment.experiment_id])
                    cols = [
                        "run_id",
                        "status",
                        "tags.adapter_name",
                        "params.lora_r",
                        "params.learning_rate",
                        "metrics.mean_keyword_score",
                    ]
                    display(runs[[c for c in cols if c in runs.columns]].head(20))
                else:
                    print("No experiment found yet.")
                """
            ),
        ],
    )

    write(
        "06_start_vllm.ipynb",
        [
            md(
                """
                # 06 Start vLLM with LoRA Enabled

                This notebook documents the local vLLM launch path. vLLM should run as an OpenAI-compatible server with LoRA enabled.

                ```mermaid
                flowchart LR
                    A[Qwen2.5-7B-Instruct] --> B[vLLM OpenAI server]
                    C[adapters/*] --> B
                    B --> D[/v1/chat/completions]
                ```

                Dynamic adapter loading requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`.
                """
            ),
            code(COMMON_SETUP),
            md("## Compose command\n\nRun this in a terminal from the project root. Expected output: vLLM logs ending with an OpenAI API server listening on port 8000."),
            code('print("make serve")'),
            md("## Equivalent vLLM command\n\nUse this on an MLIS host or any Linux CUDA machine with vLLM installed."),
            code(
                """
                command = f'''
                VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \\
                python -m vllm.entrypoints.openai.api_server \\
                  --host ${'{'}VLLM_HOST:-0.0.0.0{'}'} \\
                  --port ${'{'}VLLM_PORT:-8000{'}'} \\
                  --model "{cfg.base_model}" \\
                  --served-model-name base \\
                  --enable-lora \\
                  --max-model-len {cfg.vllm_max_model_len if hasattr(cfg, "vllm_max_model_len") else 4096}
                '''
                print(command)
                """
            ),
            md("## Health check\n\nExpected output: JSON model list containing `base` before adapters are loaded."),
            code(
                """
                import requests

                response = requests.get(f"{cfg.vllm_base_url}/v1/models", headers={"Authorization": f"Bearer {cfg.vllm_api_key}"}, timeout=10)
                print(response.status_code)
                print(response.text[:1000])
                """
            ),
        ],
    )

    write(
        "07_load_adapters.ipynb",
        [
            md(
                """
                # 07 Dynamically Load Adapters into vLLM

                This notebook loads the local PEFT adapters into a running vLLM server.

                ```mermaid
                sequenceDiagram
                    participant N as Notebook
                    participant V as vLLM
                    N->>V: POST /v1/load_lora_adapter
                    N->>V: GET /v1/models
                    V-->>N: base plus LoRA model names
                ```
                """
            ),
            code(COMMON_SETUP),
            md("## Load adapters\n\nExpected output: one success line per adapter."),
            code(
                """
                from scripts.load_adapters import load_adapter

                for adapter in cfg.adapters:
                    load_adapter(cfg.vllm_base_url, cfg.vllm_api_key, adapter, cfg.adapter_dir / adapter)
                """
            ),
            md("## Verify vLLM model registration\n\nExpected output: `finance`, `legal`, and `healthcare` appear in `/v1/models`."),
            code(
                """
                import requests

                response = requests.get(f"{cfg.vllm_base_url}/v1/models", headers={"Authorization": f"Bearer {cfg.vllm_api_key}"}, timeout=10)
                response.raise_for_status()
                print(response.json())
                """
            ),
        ],
    )

    write(
        "08_fastapi_gateway.ipynb",
        [
            md(
                """
                # 08 FastAPI Gateway

                This notebook explains and starts the adapter routing gateway.

                ```mermaid
                flowchart LR
                    A[Client request] --> B[FastAPI gateway]
                    B --> C{Adapter}
                    C -->|finance| D[vLLM model finance]
                    C -->|legal| E[vLLM model legal]
                    C -->|healthcare| F[vLLM model healthcare]
                    C -->|fallback| G[vLLM model base]
                ```
                """
            ),
            code(COMMON_SETUP),
            md("## Routing rules\n\nThe gateway accepts an explicit `adapter` or `domain`. If none is provided, it uses simple domain keywords."),
            code(
                """
                from serving.gateway import infer_adapter, ChatMessage

                examples = [
                    "Explain revenue concentration risk.",
                    "Summarize this limitation of liability clause.",
                    "What does prior authorization mean?",
                    "Tell me a neutral productivity tip.",
                ]
                for prompt in examples:
                    adapter = infer_adapter([ChatMessage(role="user", content=prompt)])
                    print(f"{adapter}: {prompt}")
                """
            ),
            md("## Start gateway\n\nRun this in a terminal. Expected output: Uvicorn serving on `http://localhost:8080`."),
            code('print("make api")'),
            md("## Test gateway request\n\nExpected output: an OpenAI-compatible chat completion plus `routed_adapter`."),
            code(
                """
                import requests

                payload = {
                    "adapter": "finance",
                    "messages": [{"role": "user", "content": "Explain revenue concentration risk."}],
                    "temperature": 0.2,
                    "max_tokens": 160,
                }
                response = requests.post(f"{cfg.api_base_url}/chat", json=payload, timeout=60)
                print(response.status_code)
                print(response.text[:1200])
                """
            ),
        ],
    )

    write(
        "09_test_inference.ipynb",
        [
            md(
                """
                # 09 Test Inference and Adapter Specialization

                This notebook sends the same style of request to each adapter and compares outputs.

                ```mermaid
                flowchart TB
                    A[Prompt set] --> B[finance adapter]
                    A --> C[legal adapter]
                    A --> D[healthcare adapter]
                    B --> E[Compare tone and terminology]
                    C --> E
                    D --> E
                ```
                """
            ),
            code(COMMON_SETUP),
            md("## Example prompts and expected behavior\n\n- Finance should mention financial risk, margin, cash, revenue, or accounting context.\n- Legal should mention contracts, clauses, liability, or counsel.\n- Healthcare should mention coverage, care plans, discharge, or patient administration."),
            code(
                """
                from openai import OpenAI

                client = OpenAI(base_url=f"{cfg.vllm_base_url}/v1", api_key=cfg.vllm_api_key)
                prompts = {
                    "finance": "Explain revenue concentration risk in two sentences.",
                    "legal": "Explain why a limitation of liability clause matters.",
                    "healthcare": "Explain what prior authorization means.",
                }

                outputs = {}
                for adapter, prompt in prompts.items():
                    response = client.chat.completions.create(
                        model=adapter,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=180,
                    )
                    outputs[adapter] = response.choices[0].message.content
                    print(f"\\n## {adapter}\\nPrompt: {prompt}\\nOutput: {outputs[adapter]}")
                """
            ),
            md("## Lightweight evaluation\n\nExpected output: keyword scores logged to MLflow."),
            code(
                """
                import subprocess
                import sys

                subprocess.run([sys.executable, "evaluation/evaluate.py"], check=True)
                """
            ),
        ],
    )

    write(
        "10_end_to_end_demo.ipynb",
        [
            md(
                """
                # 10 End-to-End Demo

                This notebook ties together the complete local workflow and shows dynamic adapter switching.

                ```mermaid
                flowchart LR
                    A[Generate datasets] --> B[Train LoRA adapters]
                    B --> C[Register in MLflow]
                    C --> D[Start vLLM]
                    D --> E[Load adapters]
                    E --> F[FastAPI gateway]
                    F --> G[Domain-specific responses]
                ```
                """
            ),
            code(COMMON_SETUP),
            md("## Local workflow checklist\n\nRun notebooks `01` through `09` in order for the full live demo. The cells below provide a compact command checklist."),
            code(
                """
                commands = [
                    "make up",
                    "make notebooks",
                    "python datasets/generate_synthetic.py",
                    "python training/train_lora.py --adapter finance",
                    "python training/train_lora.py --adapter legal",
                    "python training/train_lora.py --adapter healthcare",
                    "python scripts/register_mlflow.py",
                    "make serve",
                    "python scripts/load_adapters.py",
                    "make api",
                    "python scripts/test_inference.py --adapter finance",
                ]
                print("\\n".join(commands))
                """
            ),
            md("## Dynamic adapter switching through the gateway\n\nExpected output: each request returns `routed_adapter` matching the domain."),
            code(
                """
                import requests

                demo_prompts = [
                    "Explain a revenue forecast risk.",
                    "Explain a contract indemnity clause.",
                    "Explain patient discharge instructions.",
                ]

                for prompt in demo_prompts:
                    payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 140}
                    response = requests.post(f"{cfg.api_base_url}/chat", json=payload, timeout=60)
                    print("\\nPROMPT:", prompt)
                    print(response.status_code)
                    print(response.text[:1000])
                """
            ),
            md(
                """
                ## MLIS transfer summary

                Copy `adapters/`, `.env`, `scripts/load_adapters.py`, and the serving configuration to the MLIS host. Set `VLLM_BASE_URL` to the remote vLLM endpoint and run the same adapter loading script. The trained adapters remain standalone PEFT artifacts.
                """
            ),
        ],
    )


def cfg_placeholder() -> str:
    return "BASE_MODEL from .env"


if __name__ == "__main__":
    main()
