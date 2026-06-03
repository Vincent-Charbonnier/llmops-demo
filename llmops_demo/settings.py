from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None


load_dotenv()


def str_to_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def csv_env(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    base_model: str = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    dataset_dir: Path = Path(os.getenv("DATASET_DIR", "datasets/generated"))
    adapter_dir: Path = Path(os.getenv("ADAPTER_DIR", "adapters"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    adapters: tuple[str, ...] = tuple(csv_env("ADAPTERS", "finance,legal,healthcare"))

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "llmops-lora-demo")
    mlflow_registered_model_prefix: str = os.getenv(
        "MLFLOW_REGISTERED_MODEL_PREFIX", "qwen2_5_7b_lora"
    )

    train_epochs: int = int(os.getenv("TRAIN_EPOCHS", "1"))
    train_batch_size: int = int(os.getenv("TRAIN_BATCH_SIZE", "1"))
    gradient_accumulation_steps: int = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "0.0002"))
    max_seq_length: int = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
    lora_r: int = int(os.getenv("LORA_R", "16"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "32"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))
    load_in_4bit: bool = str_to_bool(os.getenv("LOAD_IN_4BIT"), True)

    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "local-dev")
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8080").rstrip("/")


def settings() -> Settings:
    return Settings()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
