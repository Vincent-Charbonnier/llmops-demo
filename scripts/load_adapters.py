from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import requests

from llmops_demo.settings import settings


def load_adapter(base_url: str, api_key: str, name: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {path}")

    endpoint = f"{base_url.rstrip('/')}/v1/load_lora_adapter"
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"lora_name": name, "lora_path": str(path)},
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Failed to load {name}: {response.status_code} {response.text}")
    print(f"Loaded adapter '{name}' from {path}")


def main() -> None:
    cfg = settings()
    parser = argparse.ArgumentParser(description="Dynamically load local LoRA adapters into vLLM.")
    parser.add_argument("--adapter", choices=list(cfg.adapters), help="Load one adapter. Defaults to all.")
    parser.add_argument("--base-url", default=cfg.vllm_base_url)
    args = parser.parse_args()

    adapters = [args.adapter] if args.adapter else list(cfg.adapters)
    for adapter in adapters:
        load_adapter(args.base_url, cfg.vllm_api_key, adapter, cfg.adapter_dir / adapter)


if __name__ == "__main__":
    main()
