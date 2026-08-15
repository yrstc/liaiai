import json

from fastapi.testclient import TestClient

import app as proxy
from app import ChatMessage, _content_to_text, _extract_text, messages_to_agent_input


def test_messages_to_agent_input() -> None:
    result = messages_to_agent_input(
        [
            ChatMessage(role="system", content="Stay in character."),
            ChatMessage(role="user", name="Alice", content="Hello"),
            ChatMessage(role="assistant", content="Hi!"),
        ]
    )
    raw = result.split("<openai_chat_request>\n", 1)[1].split(
        "\n</openai_chat_request>", 1
    )[0]
    envelope = json.loads(raw)
    assert envelope["instruction_messages"][0]["role"] == "system"
    assert envelope["conversation"][0] == {
        "index": 1,
        "role": "user",
        "name": "Alice",
        "content": [{"type": "text", "text": "Hello"}],
    }
    assert envelope["conversation"][1]["role"] == "assistant"


def test_multimodal_text_is_flattened() -> None:
    assert (
        _content_to_text(
            [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "https://example/image.png"}},
            ]
        )
        == "Describe this\n[image: https://example/image.png]"
    )


def test_tool_message_metadata_is_preserved() -> None:
    result = messages_to_agent_input(
        [
            ChatMessage.model_validate(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                }
            ),
            ChatMessage.model_validate(
                {"role": "tool", "tool_call_id": "call_1", "content": "result"}
            ),
        ]
    )
    raw = result.split("<openai_chat_request>\n", 1)[1].split(
        "\n</openai_chat_request>", 1
    )[0]
    messages = json.loads(raw)["conversation"]
    assert messages[0]["tool_calls"][0]["id"] == "call_1"
    assert messages[1]["tool_call_id"] == "call_1"


def test_extract_text_from_done_payload() -> None:
    payload = {"run": {"status": "completed", "final_output": {"content": "answer"}}}
    assert _extract_text(payload) == "answer"


def test_model_metadata_exposes_proxy_limits() -> None:
    response = TestClient(proxy.app).get("/v1/models")
    assert response.status_code == 200
    model = response.json()["data"][0]
    if proxy.MAX_CONTEXT_TOKENS > 0:
        assert model["context_length"] == proxy.MAX_CONTEXT_TOKENS
    else:
        assert "context_length" not in model
    if proxy.MAX_INPUT_TOKENS > 0:
        assert model["max_input_tokens"] == proxy.MAX_INPUT_TOKENS
    else:
        assert "max_input_tokens" not in model
    assert model["max_output_tokens"] == proxy.MAX_OUTPUT_TOKENS


def test_rejects_oversized_output_request() -> None:
    response = TestClient(proxy.app).post(
        "/v1/chat/completions",
        json={
            "model": proxy.MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": proxy.MAX_OUTPUT_TOKENS + 1,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "max_tokens_exceeded"


def test_rejects_oversized_input(monkeypatch) -> None:
    monkeypatch.setattr(proxy, "MAX_INPUT_TOKENS", 5)
    response = TestClient(proxy.app).post(
        "/v1/chat/completions",
        json={
            "model": proxy.MODEL_ID,
            "messages": [{"role": "user", "content": "long input " * 30}],
            "max_tokens": 1,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"


def test_non_streaming_completion_and_stop(monkeypatch) -> None:
    async def fake_events(_prompt, _model):
        yield "stream_token", {"token": "hello STOP ignored"}
        yield "done", {}

    monkeypatch.setattr(proxy, "_upstream_events", fake_events)
    monkeypatch.setattr(proxy, "_require_upstream_config", lambda: None)
    response = TestClient(proxy.app).post(
        "/v1/chat/completions",
        json={
            "model": proxy.MODEL_ID,
            "messages": [
                {"role": "system", "content": "answer briefly"},
                {"role": "user", "content": "say hello"},
            ],
            "stop": " STOP",
            "max_tokens": 20,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] > 0


def test_streaming_completion_has_openai_sse_shape(monkeypatch) -> None:
    async def fake_events(_prompt, _model):
        yield "stream_token", {"token": "stream-ok"}
        yield "done", {}

    monkeypatch.setattr(proxy, "_upstream_events", fake_events)
    monkeypatch.setattr(proxy, "_require_upstream_config", lambda: None)
    with TestClient(proxy.app).stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": proxy.MODEL_ID,
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        text = response.read().decode()
    assert response.status_code == 200
    assert '"content": "stream-ok"' in text
    assert '"choices": []' in text
    assert text.rstrip().endswith("data: [DONE]")
