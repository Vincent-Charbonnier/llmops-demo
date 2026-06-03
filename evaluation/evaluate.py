from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import mlflow
from openai import OpenAI

from llmops_demo.settings import settings


@dataclass(frozen=True)
class EvalCase:
    adapter: str
    prompt: str
    required_terms: tuple[str, ...]


CASES = [
    EvalCase("finance", "Explain revenue concentration risk in two sentences.", ("revenue", "risk")),
    EvalCase("legal", "Explain why a limitation of liability clause matters.", ("liability", "contract")),
    EvalCase("healthcare", "Explain what prior authorization means.", ("authorization", "coverage")),
]


def score_response(text: str, required_terms: tuple[str, ...]) -> float:
    lowered = text.lower()
    hits = sum(1 for term in required_terms if term in lowered)
    return hits / max(len(required_terms), 1)


def main() -> None:
    cfg = settings()
    parser = argparse.ArgumentParser(description="Run a small adapter routing evaluation.")
    parser.add_argument("--base-url", default=f"{cfg.vllm_base_url}/v1")
    args = parser.parse_args()

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)
    client = OpenAI(base_url=args.base_url, api_key=cfg.vllm_api_key)

    with mlflow.start_run(run_name="adapter-evaluation"):
        scores = []
        for case in CASES:
            response = client.chat.completions.create(
                model=case.adapter,
                messages=[{"role": "user", "content": case.prompt}],
                temperature=0.1,
                max_tokens=160,
            )
            text = response.choices[0].message.content or ""
            score = score_response(text, case.required_terms)
            scores.append(score)
            mlflow.log_metric(f"{case.adapter}_keyword_score", score)
            print(f"[{case.adapter}] score={score:.2f} response={text}")
        mlflow.log_metric("mean_keyword_score", sum(scores) / len(scores))


if __name__ == "__main__":
    main()
