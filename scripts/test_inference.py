from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from llmops_demo.settings import settings


def main() -> None:
    cfg = settings()
    parser = argparse.ArgumentParser(description="Smoke test OpenAI-compatible inference through vLLM.")
    parser.add_argument("--adapter", default="finance", choices=["base", *cfg.adapters])
    parser.add_argument("--prompt", default="Explain one key risk to monitor in this domain.")
    parser.add_argument("--base-url", default=f"{cfg.vllm_base_url}/v1")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=cfg.vllm_api_key)
    response = client.chat.completions.create(
        model=args.adapter,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=0.2,
        max_tokens=160,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
