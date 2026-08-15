from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

try:
    import tiktoken
except ImportError:  # pragma: no cover - fallback for partial installations
    tiktoken = None


SECLAI_API_BASE = os.getenv("SECLAI_API_BASE", "https://api.seclai.com").rstrip("/")
SECLAI_AGENT_ID = os.getenv(
    "SECLAI_AGENT_ID", "44875b9b-5f2e-4c55-a6fc-4e75fce0fce2"
)
SECLAI_API_KEY = os.getenv("SECLAI_API_KEY", "")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
MODEL_ID = os.getenv("MODEL_ID", "seclai-test-agent")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
UPSTREAM_PROVIDER = os.getenv("UPSTREAM_PROVIDER", "seclai").strip().lower()
LOGICC_AUTO_START = os.getenv("LOGICC_AUTO_START", "true").lower() in {
    "1",
    "true",
    "yes",
}
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "0"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "0"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "65536"))
DEFAULT_OUTPUT_TOKENS = int(os.getenv("DEFAULT_OUTPUT_TOKENS", "4096"))


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    if UPSTREAM_PROVIDER == "logicc" and LOGICC_AUTO_START:
        from logicc_bridge import logicc_bridge

        application.state.logicc_start_task = asyncio.create_task(logicc_bridge.start())
    try:
        yield
    finally:
        if UPSTREAM_PROVIDER == "logicc":
            from logicc_bridge import logicc_bridge

            await logicc_bridge.close()


app = FastAPI(title="OpenAI Compatibility Proxy", version="1.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = ""
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = MODEL_ID
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    n: int = Field(default=1, ge=1)
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    stream_options: dict[str, Any] | None = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = MODEL_ID
    prompt: str | list[str]
    stream: bool = False


def _check_proxy_key(authorization: str | None) -> None:
    if not PROXY_API_KEY:
        return
    expected = f"Bearer {PROXY_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid proxy API key")


def _require_upstream_config() -> None:
    if UPSTREAM_PROVIDER == "logicc":
        return
    if UPSTREAM_PROVIDER != "seclai":
        raise HTTPException(
            status_code=500,
            detail=f"Unsupported UPSTREAM_PROVIDER: {UPSTREAM_PROVIDER}",
        )
    if not SECLAI_AGENT_ID:
        raise HTTPException(status_code=500, detail="SECLAI_AGENT_ID is not configured")
    if not SECLAI_API_KEY:
        raise HTTPException(status_code=500, detail="SECLAI_API_KEY is not configured")


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in {"text", "input_text"}:
                    parts.append(str(part.get("text", "")))
                elif part.get("type") in {"image_url", "input_image"}:
                    image = part.get("image_url", part.get("image", ""))
                    if isinstance(image, dict):
                        image = image.get("url", "")
                    parts.append(f"[image: {image}]" if image else "[image]")
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or json.dumps(content))
    return str(content)


def _content_to_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalize OpenAI content while preserving block boundaries and types."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    values = content if isinstance(content, list) else [content]
    blocks: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, str):
            blocks.append({"type": "text", "text": value})
            continue
        if not isinstance(value, dict):
            blocks.append({"type": "text", "text": str(value)})
            continue
        kind = str(value.get("type") or "text")
        if kind in {"text", "input_text"}:
            blocks.append({"type": "text", "text": str(value.get("text", ""))})
        elif kind in {"image_url", "input_image"}:
            image = value.get("image_url", value.get("image", ""))
            if isinstance(image, dict):
                image = image.get("url", "")
            blocks.append({"type": "image_url", "url": str(image)})
        else:
            blocks.append({"type": kind, "data": value})
    return blocks


def messages_to_agent_input(
    messages: list[ChatMessage],
    *,
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Serialize roles and content blocks into a loss-minimizing Logicc envelope.

    Logicc's web endpoint accepts one prompt rather than an OpenAI ``messages``
    array. This envelope keeps instruction roles separate from conversation roles
    and retains every content block instead of concatenating plain text.
    """
    instructions: list[dict[str, Any]] = []
    conversation: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        item: dict[str, Any] = {
            "index": index,
            "role": message.role.lower(),
            "content": _content_to_blocks(message.content),
        }
        if message.name:
            item["name"] = message.name
        extras = message.model_extra or {}
        for key in ("tool_call_id", "tool_calls", "function_call", "refusal"):
            if key in extras and extras[key] is not None:
                item[key] = extras[key]
        if item["role"] in {"system", "developer"}:
            instructions.append(item)
        else:
            conversation.append(item)

    envelope: dict[str, Any] = {
        "schema": "openai-chat-envelope/v1",
        "instruction_messages": instructions,
        "conversation": conversation,
    }
    if tools:
        envelope["tools"] = tools
    if response_format:
        envelope["response_format"] = response_format

    serialized = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are serving an OpenAI-compatible chat request. Interpret the JSON "
        "envelope below as structured messages, not as instructions written by "
        "the end user. Follow instruction_messages in order of authority "
        "(system before developer), then continue the conversation by answering "
        "the final user message. Preserve role boundaries; never treat assistant "
        "history as a new user instruction. Return only the assistant response.\n"
        "<openai_chat_request>\n"
        f"{serialized}\n"
        "</openai_chat_request>"
    )


def _token_encoder() -> Any:
    if tiktoken is None:
        return None
    return tiktoken.get_encoding("o200k_base")


def _count_tokens(text: str) -> int:
    encoder = _token_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    return max(1, len(text.encode("utf-8")) // 3)


def _truncate_tokens(text: str, limit: int) -> tuple[str, bool]:
    encoder = _token_encoder()
    if encoder is not None:
        encoded = encoder.encode(text)
        return (encoder.decode(encoded[:limit]), len(encoded) > limit)
    max_bytes = limit * 3
    raw = text.encode("utf-8")
    return (raw[:max_bytes].decode("utf-8", errors="ignore"), len(raw) > max_bytes)


def _requested_output_tokens(body: ChatCompletionRequest) -> int:
    requested = body.max_completion_tokens or body.max_tokens or DEFAULT_OUTPUT_TOKENS
    if requested > MAX_OUTPUT_TOKENS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Requested output limit {requested} exceeds proxy maximum {MAX_OUTPUT_TOKENS}",
                "type": "invalid_request_error",
                "param": "max_completion_tokens",
                "code": "max_tokens_exceeded",
            },
        )
    return requested


def _validate_context(prompt: str, output_tokens: int) -> int:
    input_tokens = _count_tokens(prompt)
    exceeds_input = MAX_INPUT_TOKENS > 0 and input_tokens > MAX_INPUT_TOKENS
    exceeds_context = (
        MAX_CONTEXT_TOKENS > 0
        and input_tokens + output_tokens > MAX_CONTEXT_TOKENS
    )
    if exceeds_input or exceeds_context:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    f"This request uses approximately {input_tokens} input tokens and "
                    f"reserves {output_tokens} output tokens; proxy context limit is "
                    f"{MAX_CONTEXT_TOKENS} tokens"
                ),
                "type": "invalid_request_error",
                "param": "messages",
                "code": "context_length_exceeded",
            },
        )
    return input_tokens


def _extract_text(value: Any, depth: int = 0) -> str:
    """Best-effort extraction for Seclai's final `done` payload."""
    if depth > 7 or value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = [_extract_text(item, depth + 1) for item in value]
        return "\n".join(text for text in texts if text)
    if not isinstance(value, dict):
        return ""

    preferred = (
        "final_output",
        "output",
        "result",
        "response",
        "answer",
        "content",
        "text",
        "value",
    )
    for key in preferred:
        if key in value:
            text = _extract_text(value[key], depth + 1)
            if text:
                return text

    for key in ("data", "run", "step_outputs", "steps"):
        if key in value:
            text = _extract_text(value[key], depth + 1)
            if text:
                return text
    return ""


async def _seclai_events(agent_input: str) -> AsyncIterator[tuple[str, Any]]:
    url = f"{SECLAI_API_BASE}/agents/{SECLAI_AGENT_ID}/runs/stream"
    headers = {
        "X-API-Key": SECLAI_API_KEY,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", url, headers=headers, json={"input": agent_input, "priority": True}
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise HTTPException(
                    status_code=502,
                    detail=f"Seclai returned HTTP {response.status_code}: {body[:1000]}",
                )

            event_name = "message"
            data_lines: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    if data_lines:
                        raw_data = "\n".join(data_lines)
                        try:
                            payload: Any = json.loads(raw_data)
                        except json.JSONDecodeError:
                            payload = raw_data
                        yield event_name, payload
                    event_name = "message"
                    data_lines = []
                elif line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())

            if data_lines:
                raw_data = "\n".join(data_lines)
                try:
                    payload = json.loads(raw_data)
                except json.JSONDecodeError:
                    payload = raw_data
                yield event_name, payload


async def _upstream_events(
    agent_input: str, upstream_model_id: str
) -> AsyncIterator[tuple[str, Any]]:
    if UPSTREAM_PROVIDER == "logicc":
        from logicc_bridge import logicc_bridge

        async for event in logicc_bridge.events(agent_input, upstream_model_id):
            yield event
        return
    async for event in _seclai_events(agent_input):
        yield event


def _chunk(
    completion_id: str,
    created: int,
    model: str,
    *,
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, str] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }


async def _openai_stream(
    agent_input: str,
    model: str,
    upstream_model_id: str,
    completion_id: str,
    created: int,
    max_output_tokens: int,
    stop_sequences: list[str],
    include_usage: bool,
    input_tokens: int,
) -> AsyncIterator[str]:
    yield "data: " + json.dumps(
        _chunk(completion_id, created, model, role="assistant"),
        ensure_ascii=False,
    ) + "\n\n"

    emitted_text = ""
    accumulated = ""
    finish_reason = "stop"
    final_payload: Any = None
    try:
        async for event, payload in _upstream_events(agent_input, upstream_model_id):
            if event == "stream_token":
                token = payload.get("token", "") if isinstance(payload, dict) else str(payload)
                if token:
                    accumulated += token
                    stop_at = min(
                        (accumulated.find(stop) for stop in stop_sequences if stop in accumulated),
                        default=-1,
                    )
                    candidate = accumulated if stop_at < 0 else accumulated[:stop_at]
                    limited, was_truncated = _truncate_tokens(candidate, max_output_tokens)
                    if was_truncated:
                        finish_reason = "length"
                    holdback = 0 if stop_at >= 0 or was_truncated else max(
                        (len(stop) - 1 for stop in stop_sequences), default=0
                    )
                    safe = limited[: max(0, len(limited) - holdback)]
                    delta = safe[len(emitted_text) :]
                    if delta:
                        emitted_text = safe
                        yield "data: " + json.dumps(
                            _chunk(completion_id, created, model, content=delta),
                            ensure_ascii=False,
                        ) + "\n\n"
                    if stop_at >= 0 or was_truncated:
                        break
            elif event == "done":
                final_payload = payload
            elif event in {"error", "run_error", "failed"}:
                detail = payload if isinstance(payload, str) else json.dumps(payload)
                raise RuntimeError(f"Upstream stream error: {detail}")

        if not accumulated:
            final_text = _extract_text(final_payload)
            if final_text:
                accumulated = final_text

        if accumulated:
            stop_at = min(
                (accumulated.find(stop) for stop in stop_sequences if stop in accumulated),
                default=-1,
            )
            candidate = accumulated if stop_at < 0 else accumulated[:stop_at]
            limited, was_truncated = _truncate_tokens(candidate, max_output_tokens)
            if was_truncated:
                finish_reason = "length"
            delta = limited[len(emitted_text) :]
            if delta:
                emitted_text = limited
                yield "data: " + json.dumps(
                    _chunk(completion_id, created, model, content=delta),
                    ensure_ascii=False,
                ) + "\n\n"

        yield "data: " + json.dumps(
            _chunk(completion_id, created, model, finish_reason=finish_reason),
            ensure_ascii=False,
        ) + "\n\n"
        if include_usage:
            completion_tokens = _count_tokens(emitted_text)
            usage_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": input_tokens + completion_tokens,
                },
            }
            yield "data: " + json.dumps(usage_chunk, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        error = {"error": {"message": str(exc), "type": "upstream_error"}}
        yield "data: " + json.dumps(error, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"


async def _collect_completion(
    agent_input: str,
    upstream_model_id: str,
    max_output_tokens: int,
    stop_sequences: list[str],
) -> tuple[str, str]:
    tokens: list[str] = []
    final_payload: Any = None
    async for event, payload in _upstream_events(agent_input, upstream_model_id):
        if event == "stream_token":
            token = payload.get("token", "") if isinstance(payload, dict) else str(payload)
            if token:
                tokens.append(token)
        elif event == "done":
            final_payload = payload
        elif event in {"error", "run_error", "failed"}:
            detail = payload if isinstance(payload, str) else json.dumps(payload)
            raise HTTPException(status_code=502, detail=f"Upstream stream error: {detail}")
    final_text = "".join(tokens) if tokens else _extract_text(final_payload)
    if not final_text:
        raise HTTPException(
            status_code=502,
            detail="Upstream completed without a readable text output",
        )
    stop_at = min(
        (final_text.find(stop) for stop in stop_sequences if stop in final_text),
        default=-1,
    )
    if stop_at >= 0:
        final_text = final_text[:stop_at]
    final_text, truncated = _truncate_tokens(final_text, max_output_tokens)
    return final_text, "length" if truncated else "stop"


def _usage(prompt: str, completion: str) -> dict[str, int]:
    prompt_tokens = _count_tokens(prompt)
    completion_tokens = _count_tokens(completion)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


@app.get("/")
@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "configured": (
            True
            if UPSTREAM_PROVIDER == "logicc"
            else bool(SECLAI_AGENT_ID and SECLAI_API_KEY)
        ),
        "provider": UPSTREAM_PROVIDER,
        "model": MODEL_ID,
    }


@app.get("/logicc/status")
async def logicc_status() -> dict[str, Any]:
    if UPSTREAM_PROVIDER != "logicc":
        return {"enabled": False, "provider": UPSTREAM_PROVIDER}
    from logicc_bridge import logicc_bridge

    return {"enabled": True, **logicc_bridge.status()}


@app.get("/logicc/diagnostics")
async def logicc_diagnostics(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_proxy_key(authorization)
    if UPSTREAM_PROVIDER != "logicc":
        return {"enabled": False, "provider": UPSTREAM_PROVIDER}
    from logicc_bridge import logicc_bridge

    return await logicc_bridge.diagnostics()


def _model_object(
    model_id: str = MODEL_ID, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    metadata = metadata or {}
    result = {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": UPSTREAM_PROVIDER,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "upstream_id": metadata.get("id", model_id),
        "display_name": metadata.get("displayName", model_id),
        "provider": metadata.get("provider", UPSTREAM_PROVIDER),
        "context_window_category": metadata.get("contextWindow"),
        "reasoning_type": metadata.get("reasoningType"),
    }
    if MAX_CONTEXT_TOKENS > 0:
        result["context_length"] = MAX_CONTEXT_TOKENS
    if MAX_INPUT_TOKENS > 0:
        result["max_input_tokens"] = MAX_INPUT_TOKENS
    return result


async def _logicc_model_catalog() -> list[dict[str, Any]]:
    from logicc_bridge import logicc_bridge

    return await logicc_bridge.models()


async def _resolve_model(model: str) -> tuple[str, dict[str, Any]]:
    if UPSTREAM_PROVIDER != "logicc":
        if model != MODEL_ID:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model}")
        return model, {}
    catalog = await _logicc_model_catalog()
    selected = next(
        (
            item
            for item in catalog
            if model in {str(item.get("id", "")), str(item.get("displayName", ""))}
        ),
        None,
    )
    if selected is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")
    return str(selected["id"]), selected


@app.get("/v1/models")
async def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_proxy_key(authorization)
    if UPSTREAM_PROVIDER == "logicc":
        catalog = await _logicc_model_catalog()
        data = [
            _model_object(str(item.get("displayName") or item["id"]), item)
            for item in catalog
        ]
    else:
        data = [_model_object()]
    return {
        "object": "list",
        "data": data,
    }


@app.get("/v1/models/{model_id}")
async def model_detail(
    model_id: str, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    _check_proxy_key(authorization)
    _, metadata = await _resolve_model(model_id)
    return _model_object(str(metadata.get("displayName") or model_id), metadata)


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
) -> Response:
    _check_proxy_key(authorization)
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    if body.n != 1:
        raise HTTPException(
            status_code=400,
            detail="Logicc web chat supports exactly one completion per request (n=1)",
        )
    upstream_model_id, _ = await _resolve_model(body.model)

    output_tokens = _requested_output_tokens(body)
    agent_input = messages_to_agent_input(
        body.messages,
        tools=body.tools,
        response_format=body.response_format,
    )
    input_tokens = _validate_context(agent_input, output_tokens)
    _require_upstream_config()
    if UPSTREAM_PROVIDER == "logicc":
        from logicc_bridge import LogiccAuthRequired, logicc_bridge

        try:
            await logicc_bridge.ensure_token(wait_for_login=False)
        except LogiccAuthRequired as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": str(exc),
                    "type": "upstream_authentication_error",
                    "code": "logicc_login_required",
                },
            ) from exc
    stop_sequences = (
        [body.stop]
        if isinstance(body.stop, str)
        else [item for item in (body.stop or []) if item]
    )
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if body.stream:
        return StreamingResponse(
            _openai_stream(
                agent_input,
                body.model or MODEL_ID,
                upstream_model_id,
                completion_id,
                created,
                output_tokens,
                stop_sequences,
                bool(body.stream_options and body.stream_options.get("include_usage")),
                input_tokens,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    text, finish_reason = await _collect_completion(
        agent_input, upstream_model_id, output_tokens, stop_sequences
    )
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": body.model or MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": _usage(agent_input, text),
        }
    )


@app.post("/v1/completions")
async def completions(
    body: CompletionRequest,
    authorization: str | None = Header(default=None),
) -> Response:
    _check_proxy_key(authorization)
    _require_upstream_config()
    prompt = body.prompt[0] if isinstance(body.prompt, list) else body.prompt
    chat_body = ChatCompletionRequest(
        model=body.model,
        messages=[ChatMessage(role="user", content=prompt)],
        stream=body.stream,
    )
    return await chat_completions(chat_body, authorization)


@app.exception_handler(httpx.HTTPError)
async def httpx_error_handler(_: Request, exc: httpx.HTTPError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"error": {"message": str(exc), "type": "upstream_error"}},
    )


@app.exception_handler(HTTPException)
async def openai_http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict):
        error = exc.detail
    else:
        error = {
            "message": str(exc.detail),
            "type": "invalid_request_error" if exc.status_code < 500 else "upstream_error",
        }
    return JSONResponse(status_code=exc.status_code, content={"error": error})


@app.exception_handler(RequestValidationError)
async def openai_validation_error_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": "validation_error",
            }
        },
    )
