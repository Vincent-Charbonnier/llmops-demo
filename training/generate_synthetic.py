from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llmops_demo.settings import ensure_dirs, settings


DOMAIN_EXAMPLES = {
    "finance": [
        ("Explain gross margin to a new analyst.", "Gross margin is revenue minus cost of goods sold, divided by revenue. It shows how efficiently a company turns sales into profit before operating expenses."),
        ("What should I check before comparing two balance sheets?", "Compare cash, debt, working capital, retained earnings, and unusual one-time items. Confirm both statements use the same accounting period and standards."),
        ("Summarize the risk of high customer concentration.", "High customer concentration means revenue depends on a small number of buyers. Losing one account can materially reduce sales and cash flow."),
    ],
    "legal": [
        ("What is a limitation of liability clause?", "It caps or excludes certain damages between contracting parties. The exact enforceability depends on jurisdiction, contract context, and public policy."),
        ("Draft a plain-English NDA explanation.", "An NDA requires parties to protect confidential information, use it only for approved purposes, and return or destroy it when the relationship ends."),
        ("What should be reviewed in a vendor agreement?", "Review scope, payment terms, data handling, termination rights, warranties, indemnities, liability limits, and dispute resolution."),
    ],
    "healthcare": [
        ("Explain prior authorization.", "Prior authorization is insurer approval requested before a service, medication, or procedure. It confirms coverage criteria but is not clinical advice."),
        ("What is a care plan?", "A care plan documents goals, interventions, medications, follow-up steps, and responsible care team members for a patient's condition."),
        ("Summarize patient discharge instructions.", "Discharge instructions explain medications, warning signs, activity limits, follow-up appointments, and who to contact if symptoms worsen."),
    ],
}

SYSTEM_PROMPTS = {
    "finance": "You are a finance operations assistant. Give concise, practical analysis without inventing numbers.",
    "legal": "You are a legal operations assistant. Provide general information and say when legal counsel is needed.",
    "healthcare": "You are a healthcare operations assistant. Provide general administrative guidance and avoid diagnosis.",
}

DOMAIN_SEEDS = {"finance": 1101, "legal": 2202, "healthcare": 3303}


def make_record(domain: str, instruction: str, response: str, idx: int) -> dict[str, object]:
    return {
        "id": f"{domain}-{idx:04d}",
        "domain": domain,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[domain]},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
    }


def generate_domain(domain: str, count: int) -> list[dict[str, object]]:
    rng = random.Random(DOMAIN_SEEDS[domain])
    base = DOMAIN_EXAMPLES[domain]
    records = []
    for idx in range(count):
        instruction, response = rng.choice(base)
        suffix = "" if idx < len(base) else f" Include one practical checklist item for scenario {idx}."
        records.append(make_record(domain, instruction + suffix, response, idx))
    return records


def write_json(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dirs(path.parent)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    cfg = settings()
    parser = argparse.ArgumentParser(description="Generate local synthetic SFT training data.")
    parser.add_argument("--output-dir", type=Path, default=cfg.dataset_dir)
    parser.add_argument("--records-per-domain", type=int, default=60)
    args = parser.parse_args()

    for adapter in cfg.adapters:
        if adapter not in DOMAIN_EXAMPLES:
            raise ValueError(f"No synthetic data template for adapter '{adapter}'")
        rows = generate_domain(adapter, args.records_per_domain)
        output_path = args.output_dir / f"{adapter}.json"
        write_json(output_path, rows)
        print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

