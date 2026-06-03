from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import mlflow

from llmops_demo.mlflow_models import AdapterArtifactModel
from llmops_demo.settings import settings


def register_local_adapter(adapter: str, adapter_path: Path, cfg) -> None:
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter directory {adapter_path}. Run the training notebook first.")

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    model_name = f"{cfg.mlflow_registered_model_prefix}_{adapter}"

    with mlflow.start_run(run_name=f"register-{adapter}") as run:
        mlflow.set_tags({"adapter_name": adapter, "base_model": cfg.base_model, "artifact_type": "peft_lora"})
        mlflow.log_artifacts(str(adapter_path), artifact_path="adapter")
        mlflow.pyfunc.log_model(
            artifact_path="registered_adapter",
            python_model=AdapterArtifactModel(),
            artifacts={"adapter": str(adapter_path)},
            registered_model_name=model_name,
        )
        print(f"Registered {adapter_path} as {model_name} from run {run.info.run_id}")


def main() -> None:
    cfg = settings()
    parser = argparse.ArgumentParser(description="Register local PEFT adapters in MLflow.")
    parser.add_argument("--adapter", choices=list(cfg.adapters), help="Register one adapter. Defaults to all.")
    args = parser.parse_args()

    adapters = [args.adapter] if args.adapter else list(cfg.adapters)
    for adapter in adapters:
        register_local_adapter(adapter, cfg.adapter_dir / adapter, cfg)


if __name__ == "__main__":
    main()

