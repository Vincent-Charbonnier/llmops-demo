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
    dataset_dir: Path = Path(os.getenv("DATA_DIR", os.getenv("TRAINING_DATA_DIR", "training_data")))
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

    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "https://qwen257b.project-public.serving.hpepcai3.demo.local").rstrip("/")
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "eyJhbGciOiJSUzI1NiIsImtpZCI6IjlobTctX21Wcm9wSkVwMTduR0dRWHdNWjZBWHhPX0dQZzFadWx0Zkh0aTQifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxODEyMDA4NjY1LCJpYXQiOjE3ODA0NzI2NjUsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiMWJmZjU3OGEtNjhiNy00ZGM4LThjMDQtYWIzZWIyMDcxMDBiIiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3ODA0NzI2NjUxNjIiLCJ1aWQiOiIxNWMwNjI1My0zN2Y1LTQyMjAtOGUxNy1jZmYyYWNmOGZjYzIifX0sIm5iZiI6MTc4MDQ3MjY2NSwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc4MDQ3MjY2NTE2MiJ9.TLAdJJxRZMgwNyilWzUy63yjmEVaVQUl6RRTJ3u7_d4aU4euhO9XWS9jgyhaDOgB5OxHtVJPUIouXgMnApPVhic3hRiGnuQoMc1v_-S0e3zU8VI4Yi4rCJQSrLDBqDVaOtCNpnIsonZtncrQxjjh0cLoIVy2Lb63oqBDExzLsZu9Bhb7yLO-bb_sHmJVZge1_vfLD3VsjVYhmwiVyc0b_m80euMxz_yc2F52nRoNe9WD_PlLVkq3iO5jXb8YdVtPo7gG3Gz8dpA_N2WBXMtOYWm_4DYUOozgwAmwXgzj7sHI8ExLsa9GVVVgo0fOrLTDGBQc9SXTvoKGtVY_cp8vkQ")
    api_base_url: str = os.getenv("API_BASE_URL", "https://qwen257b.project-public.serving.hpepcai3.demo.local").rstrip("/")


def settings() -> Settings:
    return Settings()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
