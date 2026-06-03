from __future__ import annotations

from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llmops_demo.settings import settings


cfg = settings()
app = FastAPI(title="Local-first LLMOps Adapter Gateway", version="0.1.0")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class RoutedChatRequest(BaseModel):
    messages: list[ChatMessage]
    adapter: str | None = Field(default=None, description="Explicit adapter name.")
    domain: str | None = Field(default=None, description="Alias for adapter.")
    temperature: float = 0.2
    max_tokens: int = 512
    stream: bool = False


class OpenAIChatRequest(BaseModel):
    model: str = "base"
    messages: list[dict[str, Any]]
    temperature: float | None = 0.2
    max_tokens: int | None = 512
    stream: bool | None = False


def infer_adapter(messages: list[ChatMessage]) -> str:
    text = " ".join(message.content.lower() for message in messages)
    keyword_map = {
        "finance": ("revenue", "margin", "cash", "balance sheet", "forecast", "investment"),
        "legal": ("contract", "liability", "clause", "nda", "indemnity", "jurisdiction"),
        "healthcare": ("patient", "clinical", "authorization", "discharge", "coverage", "care plan"),
    }
    for adapter, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            return adapter
    return "base"


def resolve_adapter(request: RoutedChatRequest) -> str:
    requested = request.adapter or request.domain
    if requested:
        if requested == "base" or requested in cfg.adapters:
            return requested
        raise HTTPException(status_code=400, detail=f"Unknown adapter '{requested}'")
    return infer_adapter(request.messages)


async def post_to_vllm(payload: dict[str, Any]) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {cfg.vllm_api_key}"}
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(f"{cfg.vllm_base_url}/v1/chat/completions", headers=headers, json=payload)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "adapters": ["base", *cfg.adapters], "vllm_base_url": cfg.vllm_base_url}


@app.post("/chat")
async def routed_chat(request: RoutedChatRequest) -> dict[str, Any]:
    adapter = resolve_adapter(request)
    payload = {
        "model": adapter,
        "messages": [message.model_dump() for message in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": request.stream,
    }
    response = await post_to_vllm(payload)
    response["routed_adapter"] = adapter
    return response


@app.post("/v1/chat/completions")
async def openai_compatible_chat(request: OpenAIChatRequest) -> dict[str, Any]:
    model = request.model
    if model not in {"base", *cfg.adapters}:
        raise HTTPException(status_code=400, detail=f"Unknown model or adapter '{model}'")
    return await post_to_vllm(request.model_dump(exclude_none=True))

