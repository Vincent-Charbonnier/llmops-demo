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

CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR if (CURRENT_DIR / "PROJECT_SPEC.md").exists() else CURRENT_DIR.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
sys.path.append(str(PROJECT_ROOT))

from training.config import DEFAULT_CONFIG

cfg = DEFAULT_CONFIG.copy()
cfg["data_dir"] = str((NOTEBOOK_DIR / cfg["data_dir"]).resolve())
cfg["output_dir"] = str((NOTEBOOK_DIR / cfg["output_dir"]).resolve())
os.environ["DATA_DIR"] = cfg["data_dir"]
os.environ["ADAPTER_DIR"] = cfg["output_dir"]
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", cfg["experiment_name"])

from llmops_demo.settings import ensure_dirs, settings

settings_cfg = settings()
ensure_dirs(Path(cfg["data_dir"]), Path(cfg["output_dir"]))

print(f"Project root: {PROJECT_ROOT}")
print(f"Data dir: {cfg['data_dir']}")
print(f"Adapter output dir: {cfg['output_dir']}")
print(f"Base model: {settings_cfg.base_model}")
print(f"Adapters: {settings_cfg.adapters}")
"""


def write(name: str, cells: list[dict]) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_DIR / name
    path.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def training_notebook(number: str, adapter: str, title: str) -> tuple[str, list[dict]]:
    return (
        f"0{number}_train_{adapter}_lora.ipynb",
        [
            md(
                f"""
                # 0{number} Train {title} LoRA Adapter

                This notebook trains the `{adapter}` standalone PEFT LoRA adapter and writes it to `adapters/{adapter}/`.

                ```mermaid
                flowchart LR
                    A[training_data/{adapter}.json] --> B[Qwen chat template]
                    B --> C[PEFT LoRA training]
                    C --> D[adapters/{adapter}/]
                    C --> E[MLflow run]
                ```

                The adapter is not merged into the base model. vLLM loads it later as a runtime LoRA adapter.
                """
            ),
            code(COMMON_SETUP),
            md(
                f"""
                ## Preflight

                Expected inputs:

                - `training_data/{adapter}.json`
                - access to the base model configured by `BASE_MODEL`
                - CUDA GPU recommended for practical runtime
                """
            ),
            code(
                f"""
                dataset_path = Path(cfg["data_dir"]) / "{adapter}.json"
                adapter_path = Path(cfg["output_dir"]) / "{adapter}"
                print(f"Dataset exists: {{dataset_path.exists()}} - {{dataset_path}}")
                print(f"Adapter output: {{adapter_path}}")
                """
            ),
            md("## Train\n\nThis calls the shared training entry point, logs to MLflow, and saves the standalone PEFT adapter."),
            code(
                f"""
                from training.train_lora import train_adapter

                train_adapter("{adapter}", settings_cfg)
                """
            ),
            md(f"## Verify adapter files\n\nExpected output: PEFT adapter files under `adapters/{adapter}/`."),
            code(
                f"""
                for path in sorted((Path(cfg["output_dir"]) / "{adapter}").glob("*")):
                    print(path)
                """
            ),
        ],
    )


def main() -> None:
    write(
        "01_generate_datasets.ipynb",
        [
            md(
                """
                # 01 Generate Synthetic Training Data

                This notebook creates local supervised fine-tuning data for the finance, legal, and healthcare adapters.

                ```mermaid
                flowchart LR
                    A[Domain templates] --> B[Synthetic chat records]
                    B --> C[training_data/finance.json]
                    B --> D[training_data/legal.json]
                    B --> E[training_data/healthcare.json]
                ```

                The data files are intentionally stored in `training_data/` to avoid shadowing Hugging Face's `datasets` Python package.
                """
            ),
            code(COMMON_SETUP),
            md("## Generate training data\n\nExpected output: three JSON files under `training_data/`, one per adapter."),
            code(
                """
                from training.generate_synthetic import generate_domain, write_json

                records_per_domain = int(os.getenv("SYNTHETIC_RECORDS_PER_DOMAIN", "60"))
                data_dir = Path(cfg["data_dir"])
                ensure_dirs(data_dir)

                for adapter in settings_cfg.adapters:
                    rows = generate_domain(adapter, records_per_domain)
                    output_path = data_dir / f"{adapter}.json"
                    write_json(output_path, rows)
                    print(f"{adapter}: wrote {len(rows)} records to {output_path}")
                """
            ),
            md("## Inspect a sample\n\nThe sample should show the domain-specific system prompt and a concise assistant answer."),
            code(
                """
                import json

                data_dir = Path(cfg["data_dir"])
                for adapter in settings_cfg.adapters:
                    path = data_dir / f"{adapter}.json"
                    first = json.loads(path.read_text(encoding="utf-8"))[0]
                    print(f"\\n[{adapter}] {first['id']}")
                    for message in first["messages"]:
                        print(f"{message['role']}: {message['content'][:160]}")
                """
            ),
        ],
    )

    for filename, cells in [
        training_notebook("2", "finance", "Finance"),
        training_notebook("3", "legal", "Legal"),
        training_notebook("4", "healthcare", "Healthcare"),
    ]:
        write(filename, cells)

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
            md("## Configure MLflow\n\nFor local notebooks, file-backed MLflow works without a service. With `make up`, set `MLFLOW_TRACKING_URI=http://localhost:5000`."),
            code(
                """
                import mlflow
                from mlflow.tracking import MlflowClient

                mlflow.set_tracking_uri(settings_cfg.mlflow_tracking_uri)
                mlflow.set_experiment(settings_cfg.mlflow_experiment_name)
                client = MlflowClient()
                experiment = client.get_experiment_by_name(settings_cfg.mlflow_experiment_name)
                print("Tracking URI:", mlflow.get_tracking_uri())
                print("Experiment:", experiment.name if experiment else "not created yet")
                """
            ),
            md("## Register local adapters\n\nExpected output: one registered model per adapter."),
            code(
                """
                from training.register_mlflow import register_local_adapter

                for adapter in settings_cfg.adapters:
                    adapter_path = Path(cfg["output_dir"]) / adapter
                    if adapter_path.exists():
                        register_local_adapter(adapter, adapter_path, settings_cfg)
                    else:
                        print(f"Skipping {adapter}: {adapter_path} does not exist yet")
                """
            ),
            md("## Compare experiment runs\n\nThe table shows recent runs, adapter tags, and common LoRA parameters."),
            code(
                """
                import pandas as pd

                experiment = client.get_experiment_by_name(settings_cfg.mlflow_experiment_name)
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

                This notebook documents the local vLLM launch path.

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
            md("## Compose command\n\nRun this in a terminal from the project root. Expected output: vLLM logs with an OpenAI API server on port 8000."),
            code('print("make serve")'),
            md("## Equivalent vLLM command\n\nUse this on an MLIS host or Linux CUDA machine with vLLM installed."),
            code(
                """
                command = f'''
                VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \\
                python -m vllm.entrypoints.openai.api_server \\
                  --host ${'{'}VLLM_HOST:-0.0.0.0{'}'} \\
                  --port ${'{'}VLLM_PORT:-8000{'}'} \\
                  --model "{settings_cfg.base_model}" \\
                  --served-model-name base \\
                  --enable-lora \\
                  --max-model-len 4096
                '''
                print(command)
                """
            ),
            md("## Health check\n\nExpected output: JSON model list containing `base` before adapters are loaded."),
            code(
                """
                import requests

                response = requests.get(
                    f"{settings_cfg.vllm_base_url}/v1/models",
                    headers={"Authorization": f"Bearer {settings_cfg.vllm_api_key}"},
                    timeout=10,
                )
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

                This notebook loads local PEFT adapters into a running vLLM server.

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

                for adapter in settings_cfg.adapters:
                    load_adapter(
                        settings_cfg.vllm_base_url,
                        settings_cfg.vllm_api_key,
                        adapter,
                        Path(cfg["output_dir"]) / adapter,
                    )
                """
            ),
            md("## Verify vLLM model registration\n\nExpected output: `finance`, `legal`, and `healthcare` appear in `/v1/models`."),
            code(
                """
                import requests

                response = requests.get(
                    f"{settings_cfg.vllm_base_url}/v1/models",
                    headers={"Authorization": f"Bearer {settings_cfg.vllm_api_key}"},
                    timeout=10,
                )
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
            md("## Routing rules\n\nThe gateway accepts explicit `adapter` or `domain` fields. If none is provided, it uses simple domain keywords."),
            code(
                """
                from serving.gateway import ChatMessage, infer_adapter

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
            md("## Start gateway\n\nRun this in a terminal. Expected output: Uvicorn on `http://localhost:8080`."),
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
                response = requests.post(f"{settings_cfg.api_base_url}/chat", json=payload, timeout=60)
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

                This notebook sends representative requests to each adapter and compares outputs.

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

                client = OpenAI(base_url=f"{settings_cfg.vllm_base_url}/v1", api_key=settings_cfg.vllm_api_key)
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

                subprocess.run([sys.executable, str(PROJECT_ROOT / "evaluation" / "evaluate.py")], check=True)
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
                    A[Generate training data] --> B[Train LoRA adapters]
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
                    "python training/generate_synthetic.py",
                    "python training/train_lora.py --adapter finance",
                    "python training/train_lora.py --adapter legal",
                    "python training/train_lora.py --adapter healthcare",
                    "python training/register_mlflow.py",
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
                    response = requests.post(f"{settings_cfg.api_base_url}/chat", json=payload, timeout=60)
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


if __name__ == "__main__":
    main()
