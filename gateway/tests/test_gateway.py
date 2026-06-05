from typing import Any

from fastapi.testclient import TestClient

import gateway.app as gateway_app
import gateway.router as gateway_router


class MockResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        return None


def test_gateway_sends_vllm_api_key(monkeypatch) -> None:
    observed_headers: list[dict[str, str]] = []
    observed_payloads: list[dict[str, Any]] = []

    monkeypatch.setattr(gateway_router, "VLLM_API_KEY", "test-token")

    def mock_post(*args: Any, **kwargs: Any) -> MockResponse:
        observed_headers.append(kwargs["headers"])
        observed_payloads.append(kwargs["json"])
        if len(observed_headers) == 1:
            return MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"domain":"finance","confidence":0.94}',
                            },
                        },
                    ],
                },
            )
        return MockResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "answer",
                        },
                    },
                ],
            },
        )

    monkeypatch.setattr(gateway_router.requests, "post", mock_post)
    monkeypatch.setattr(gateway_app.requests, "post", mock_post)

    client = TestClient(gateway_app.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Explain portfolio risk.",
                },
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["adapter"] == "finance"
    assert response.json()["confidence"] == 0.94
    assert response.json()["routed_domain"] == "finance"
    assert observed_headers == [
        {"Authorization": "Bearer test-token"},
        {"Authorization": "Bearer test-token"},
    ]
    assert observed_payloads[1]["extra_body"] == {"adapter_name": "finance"}


def test_gateway_uses_base_model_for_general(monkeypatch) -> None:
    observed_payloads: list[dict[str, Any]] = []

    def mock_post(*args: Any, **kwargs: Any) -> MockResponse:
        observed_payloads.append(kwargs["json"])
        if len(observed_payloads) == 1:
            return MockResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"domain":"general","confidence":0.61}',
                            },
                        },
                    ],
                },
            )
        return MockResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "answer",
                        },
                    },
                ],
            },
        )

    monkeypatch.setattr(gateway_router.requests, "post", mock_post)
    monkeypatch.setattr(gateway_app.requests, "post", mock_post)

    client = TestClient(gateway_app.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a neutral productivity tip.",
                },
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["adapter"] == "base"
    assert response.json()["confidence"] == 0.61
    assert response.json()["routed_domain"] == "general"
    assert "extra_body" not in observed_payloads[1]
