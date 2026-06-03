from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llmops_demo.settings import settings


def main() -> None:
    cfg = settings()
    for path in [cfg.dataset_dir, cfg.output_dir]:
        if path.exists():
            shutil.rmtree(path)
            print(f"Removed {path}")


if __name__ == "__main__":
    main()
