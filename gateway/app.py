import os

import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    from .models import ChatCompletionRequest
    from .router import BASE_MODEL, VLLM_URL, classify_message, vllm_headers
except ImportError:
    from models import ChatCompletionRequest
    from router import BASE_MODEL, VLLM_URL, classify_message, vllm_headers

app = FastAPI()

CHAT_TIMEOUT_SECONDS = float(os.getenv("VLLM_CHAT_TIMEOUT_SECONDS", "120"))

ADAPTER_MAP = {
    "finance": "finance",
    "legal": "legal",
    "healthcare": "healthcare",
    "general": "base",
}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat(request: ChatCompletionRequest):
    messages = [
        message.model_dump()
        for message in request.messages
    ]

    user_message = messages[-1]["content"]
    domain, confidence = classify_message(user_message)
    adapter_name = ADAPTER_MAP.get(domain)
    
    print("\n========== ROUTING DEBUG ==========") 
    print(f"USER MESSAGE: {user_message}") 
    print(f"DOMAIN: {domain}") 
    print(f"ADAPTER: {adapter_name}") 
    print("===================================\n")

    payload = {
        "model": BASE_MODEL,
        "messages": messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }

    if adapter_name and adapter_name != "base":
        payload["extra_body"] = {
            "adapter_name": adapter_name,
        }

    response = requests.post(
        f"{VLLM_URL}/v1/chat/completions",
        headers=vllm_headers(),
        json=payload,
        timeout=CHAT_TIMEOUT_SECONDS,
    )

    response.raise_for_status()
    result = response.json()
    result["adapter"] = adapter_name
    result["confidence"] = confidence
    result["routed_domain"] = domain

    return JSONResponse(result)
