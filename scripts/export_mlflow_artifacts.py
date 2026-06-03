from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow.tracking import MlflowClient

from llmops_demo.settings import ensure_dirs, settings


def latest_run_for_adapter(client: MlflowClient, experiment_id: str, adapter: str):
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.adapter_name = '{adapter}'",
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No MLflow run found for adapter '{adapter}'")
    return runs[0]


def export_adapter(adapter: str, output_dir: Path, cfg) -> None:
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg.mlflow_experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{cfg.mlflow_experiment_name}' does not exist")

    run = latest_run_for_adapter(client, experiment.experiment_id, adapter)
    destination = output_dir / adapter
    staging_dir = output_dir / ".mlflow-downloads" / adapter
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    ensure_dirs(staging_dir)
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id,
        artifact_path="adapter",
        dst_path=str(staging_dir),
    )
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(downloaded, destination)
    print(f"Downloaded {adapter} artifacts from run {run.info.run_id} to {destination}")


def main() -> None:
    cfg = settings()
    parser = argparse.ArgumentParser(description="Download adapter artifacts from MLflow runs.")
    parser.add_argument("--adapter", choices=list(cfg.adapters), help="Export one adapter. Defaults to all.")
    parser.add_argument("--output-dir", type=Path, default=cfg.adapter_dir)
    args = parser.parse_args()

    ensure_dirs(args.output_dir)
    adapters = [args.adapter] if args.adapter else list(cfg.adapters)
    for adapter in adapters:
        export_adapter(adapter, args.output_dir, cfg)


if __name__ == "__main__":
    main()
