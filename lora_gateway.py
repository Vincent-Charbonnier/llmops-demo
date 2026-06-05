from typing import List, Union
from pydantic import BaseModel, Field
import requests


class Pipeline:
    """
    OpenWebUI pipeline for the vLLM LoRA Gateway.

    Flow:
    OpenWebUI -> Gateway -> Qwen classifier -> Selected LoRA adapter -> Response

    Features:
    - OpenAI-compatible
    - Non-streaming
    - Optional adapter override
    - Gateway routing metadata support
    """

    class Valves(BaseModel):
        GATEWAY_ENDPOINT: str = Field(
            default="http://llm-gateway.project-user-vinchar.svc.cluster.local:9000/v1",
            description="LLM Gateway endpoint",
        )

        DEFAULT_MODEL: str = Field(
            default="Qwen/Qwen2.5-7B-Instruct",
            description="Base model name",
        )

        ENABLE_ROUTING_DEBUG: bool = Field(
            default=True,
            description="Return routing metadata",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[dict, str]:

        # Optional manual adapter override
        adapter_override = body.get("adapter")

        payload = {
            "model": self.valves.DEFAULT_MODEL,
            "messages": messages,
            "stream": False,
            "temperature": body.get("temperature", 0.7),
            "max_tokens": body.get("max_tokens", 512),
        }

        # Optional testing override
        if adapter_override:
            payload["adapter"] = adapter_override

        try:
            response = requests.post(
                url=f"{self.valves.GATEWAY_ENDPOINT}/chat/completions",
                json=payload,
                timeout=120,
            )

            response.raise_for_status()
            data = response.json()

            routing = ""

            if data.get("adapter"):
                routing = (
                    f"\n\n---\n"
                    f"🔀 Routed to: {data['adapter']} "
                    f"({data.get('confidence', 0):.0%})"
                )

                data["choices"][0]["message"]["content"] += routing

            # result = {
                # "choices": data["choices"],
                # "object": data.get(
                    # "object",
                    # "chat.completion",
                # ),
            # }

            # if self.valves.ENABLE_ROUTING_DEBUG:
                # result["routing"] = {
                    # "domain": data.get("routed_domain"),
                    # "adapter": data.get("adapter"),
                    # "confidence": data.get("confidence"),
                # }

            # return result
            
            content = (
                data["choices"][0]
                .get("message", {})
                .get("content", "")
            )

            return content

        except Exception as e:
            return {
                "error": "Gateway pipeline error",
                "details": str(e),
            }