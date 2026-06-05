import json
import os

import requests

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000").rstrip("/")
VLLM_API_KEY = os.getenv("VLLM_API_KEY")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

ROUTER_PROMPT = """
You are an AI routing classifier.

Classify the user request into EXACTLY one domain:

- finance
- legal
- healthcare
- general

Return ONLY valid JSON.

Example:
{"domain":"finance","confidence":0.94}

No explanation.
"""

ALLOWED_DOMAINS = [
    "finance",
    "legal",
    "healthcare",
    "general",
]


def vllm_headers() -> dict[str, str]:
    if not VLLM_API_KEY:
        return {}
    return {"Authorization": f"Bearer {VLLM_API_KEY}"}


def _normalize_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def classify_message(user_message: str) -> tuple[str, float]:
    response = requests.post(
        f"{VLLM_URL}/v1/chat/completions",
        headers=vllm_headers(),
        json={
            "model": BASE_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": ROUTER_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            "temperature": 0,
            "max_tokens": 20,
        },
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"].strip()

    try:
        parsed = json.loads(content)
        domain = parsed["domain"].lower()
        confidence = _normalize_confidence(parsed.get("confidence"))
    except Exception:
        domain = "general"
        confidence = 0.0

    if domain not in ALLOWED_DOMAINS:
        domain = "general"

    return domain, confidence
