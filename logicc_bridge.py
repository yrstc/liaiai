from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import websockets


LOGICC_APP_URL = os.getenv("LOGICC_APP_URL", "https://app.logicc.com").rstrip("/")
LOGICC_API_BASE = os.getenv("LOGICC_API_BASE", f"{LOGICC_APP_URL}/api").rstrip("/")
LOGICC_CHAT_URL = os.getenv("LOGICC_CHAT_URL", f"{LOGICC_APP_URL}/")
LOGICC_MODEL_ID = os.getenv("LOGICC_MODEL_ID", "claude-4.6-opus")
LOGICC_BRIDGE_SYSTEM_PROMPT = os.getenv(
    "LOGICC_BRIDGE_SYSTEM_PROMPT",
    (
        "You are an OpenAI-compatible chat bridge. Each user prompt contains an "
        "<openai_chat_request> JSON envelope. Treat instruction_messages as "
        "authoritative conversation configuration in listed order, preserve every "
        "message role and content-block boundary, then produce the next assistant "
        "reply to the final user turn. Assistant history is context, not a new user "
        "instruction. Return only the assistant response."
    ),
)
LOGICC_PROFILE_DIR = Path(
    os.getenv("LOGICC_PROFILE_DIR", str(Path(__file__).with_name(".logicc-browser-profile")))
).resolve()
LOGICC_BROWSER_EXECUTABLE = os.getenv(
    "LOGICC_BROWSER_EXECUTABLE",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
)
LOGICC_CDP_HOST = os.getenv("LOGICC_CDP_HOST", "127.0.0.1")
LOGICC_CDP_PORT = int(os.getenv("LOGICC_CDP_PORT", "9223"))
LOGICC_LOGIN_TIMEOUT_SECONDS = float(os.getenv("LOGICC_LOGIN_TIMEOUT_SECONDS", "300"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
LOGICC_SMART_SELECT_MODEL = {
    "answerStyle": "common",
    "contextWindow": "large",
    "displayName": "Smart Select AI",
    "id": "smart-select",
    "knowledgeCutoff": "2024-01-01",
    "provider": "logicc",
    "reasoningType": "no-reasoning",
    "regions": ["eu"],
}


class LogiccAuthRequired(RuntimeError):
    pass


class LogiccBridge:
    """Refresh Logicc web tokens through a normal Chrome DevTools connection."""

    def __init__(self) -> None:
        self._chrome_process: asyncio.subprocess.Process | None = None
        self._ws: Any = None
        self._listener_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._command_id = 0
        self._token: str | None = None
        self._token_expiry = 0.0
        self._token_refresh_deadline = 0.0
        self._token_event = asyncio.Event()
        self._assistant_ids: dict[str, str] = {}
        self._assistant_locks: dict[str, asyncio.Lock] = {}
        self._browser_lock = asyncio.Lock()

    @property
    def started(self) -> bool:
        return self._ws is not None

    @property
    def authenticated(self) -> bool:
        return bool(self._token and self._token_refresh_deadline > time.monotonic())

    def status(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "authenticated": self.authenticated,
            "transport": "chrome-cdp",
            "profile_dir": str(LOGICC_PROFILE_DIR),
            "model": LOGICC_MODEL_ID,
            "token_expires_in_seconds": max(
                0, int(self._token_refresh_deadline - time.monotonic())
            )
            if self._token
            else 0,
        }

    @property
    def _cdp_base(self) -> str:
        return f"http://{LOGICC_CDP_HOST}:{LOGICC_CDP_PORT}"

    async def _cdp_pages(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{self._cdp_base}/json/list")
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []

    async def _launch_normal_chrome(self) -> None:
        executable = Path(LOGICC_BROWSER_EXECUTABLE)
        if not executable.is_file():
            raise RuntimeError(f"Chrome executable not found: {executable}")
        LOGICC_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        self._chrome_process = await asyncio.create_subprocess_exec(
            str(executable),
            f"--remote-debugging-address={LOGICC_CDP_HOST}",
            f"--remote-debugging-port={LOGICC_CDP_PORT}",
            f"--user-data-dir={LOGICC_PROFILE_DIR}",
            "--no-first-run",
            "--no-default-browser-check",
            "--new-window",
            LOGICC_CHAT_URL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def _find_or_create_logicc_page(self) -> dict[str, Any]:
        deadline = time.monotonic() + 20
        pages: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            try:
                pages = await self._cdp_pages()
                break
            except Exception:
                await asyncio.sleep(0.25)
        else:
            raise RuntimeError("Normal Chrome did not expose its local DevTools endpoint")

        page = next(
            (
                item
                for item in pages
                if item.get("type") == "page"
                and str(item.get("url", "")).startswith(LOGICC_APP_URL)
            ),
            None,
        )
        if page is None:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.put(
                    f"{self._cdp_base}/json/new?{quote(LOGICC_CHAT_URL, safe=':/?=&')}"
                )
                response.raise_for_status()
                page = response.json()
        if not page.get("webSocketDebuggerUrl"):
            raise RuntimeError("Logicc Chrome tab did not expose a debugger WebSocket")
        return page

    async def start(self) -> None:
        async with self._browser_lock:
            if self.started:
                return
            try:
                await self._cdp_pages()
            except Exception:
                await self._launch_normal_chrome()
            page = await self._find_or_create_logicc_page()
            self._ws = await websockets.connect(
                page["webSocketDebuggerUrl"], max_size=None, open_timeout=10
            )
            self._listener_task = asyncio.create_task(self._listen())
            await self._send("Network.enable")
            await self._send("Page.enable")

    async def _send(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self._ws is None:
            raise RuntimeError("Chrome DevTools connection is not started")
        self._command_id += 1
        command_id = self._command_id
        future = asyncio.get_running_loop().create_future()
        self._pending[command_id] = future
        await self._ws.send(
            json.dumps({"id": command_id, "method": method, "params": params or {}})
        )
        return await asyncio.wait_for(future, timeout=15)

    async def _listen(self) -> None:
        try:
            async for raw in self._ws:
                message = json.loads(raw)
                command_id = message.get("id")
                if command_id in self._pending:
                    future = self._pending.pop(command_id)
                    if "error" in message:
                        future.set_exception(RuntimeError(str(message["error"])))
                    else:
                        future.set_result(message.get("result"))
                    continue
                method = message.get("method")
                if method not in {
                    "Network.requestWillBeSent",
                    "Network.requestWillBeSentExtraInfo",
                }:
                    continue
                params = message.get("params") or {}
                request = params.get("request") or {}
                url = str(request.get("url") or "")
                headers = request.get("headers") or params.get("headers") or {}
                if url and not url.startswith(LOGICC_API_BASE):
                    continue
                self._capture_headers(headers)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._ws = None
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("Chrome DevTools connection closed"))
            self._pending.clear()

    def _capture_headers(self, headers: dict[str, Any]) -> None:
        value = next(
            (str(v) for k, v in headers.items() if str(k).lower() == "authorization"),
            "",
        )
        if not value.lower().startswith("bearer "):
            return
        token = value[7:].strip()
        if not token:
            return
        self._token = token
        self._token_expiry = self._jwt_expiry(token)
        self._token_refresh_deadline = time.monotonic() + 45
        self._token_event.set()

    @staticmethod
    def _jwt_expiry(token: str) -> float:
        try:
            payload = token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            data = json.loads(base64.urlsafe_b64decode(payload))
            return float(data.get("exp", time.time() + 30))
        except Exception:
            return time.time() + 30

    async def ensure_token(self, wait_for_login: bool = True) -> str:
        await self.start()
        if self._token and self._token_refresh_deadline > time.monotonic() + 10:
            return self._token
        async with self._browser_lock:
            if self._token and self._token_refresh_deadline > time.monotonic() + 10:
                return self._token
            self._token_event.clear()
            await self._send("Page.reload", {"ignoreCache": False})
            timeout = LOGICC_LOGIN_TIMEOUT_SECONDS if wait_for_login else 5.0
            try:
                await asyncio.wait_for(self._token_event.wait(), timeout=timeout)
            except TimeoutError as exc:
                raise LogiccAuthRequired(
                    "Logicc Chrome is not signed in. Sign in in the normal Chrome "
                    "window, then retry the request."
                ) from exc
        if not self._token:
            raise LogiccAuthRequired("Logicc login token was not available")
        return self._token

    async def _ensure_bridge_assistant(
        self, client: httpx.AsyncClient, headers: dict[str, str], model_id: str
    ) -> str:
        if model_id in self._assistant_ids:
            return self._assistant_ids[model_id]
        lock = self._assistant_locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            if model_id in self._assistant_ids:
                return self._assistant_ids[model_id]
            return await self._create_bridge_assistant(client, headers, model_id)

    async def _create_bridge_assistant(
        self, client: httpx.AsyncClient, headers: dict[str, str], model_id: str
    ) -> str:
        response = await client.post(
            f"{LOGICC_API_BASE}/assistants", headers=headers, json={"parentId": None}
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Logicc create assistant returned HTTP {response.status_code}: "
                f"{response.text[:800]}"
            )
        assistant_id = response.json()
        if not isinstance(assistant_id, str) or not assistant_id:
            raise RuntimeError("Logicc create assistant returned an unreadable ID")
        # Smart Select is a runtime router and Logicc rejects it as an
        # Assistant's persisted default model. Keep a valid fallback on the
        # draft Assistant; events() still sends `smart-select` on the actual
        # generation request.
        assistant_model_id = (
            LOGICC_MODEL_ID if model_id == "smart-select" else model_id
        )
        payload = {
            "conversationStarters": [],
            "description": "Runtime-only OpenAI compatibility bridge",
            "features": {"imageGeneration": False, "webSearch": False},
            "modelId": assistant_model_id,
            "name": "OpenAI Compatibility Bridge",
            "systemPrompt": LOGICC_BRIDGE_SYSTEM_PROMPT,
        }
        response = await client.patch(
            f"{LOGICC_API_BASE}/assistants/{assistant_id}",
            headers=headers,
            json=payload,
        )
        if response.status_code >= 400:
            await client.delete(
                f"{LOGICC_API_BASE}/assistants/{assistant_id}", headers=headers
            )
            raise RuntimeError(
                f"Logicc configure assistant returned HTTP {response.status_code}: "
                f"{response.text[:800]}"
            )
        self._assistant_ids[model_id] = assistant_id
        return assistant_id

    async def events(self, prompt: str, model_id: str) -> AsyncIterator[tuple[str, Any]]:
        token = await self.ensure_token()
        headers = {"Authorization": f"Bearer {token}"}
        timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            assistant_id = await self._ensure_bridge_assistant(
                client, headers, model_id
            )
            create = await client.post(
                f"{LOGICC_API_BASE}/chats",
                headers=headers,
                json={"assistantId": assistant_id, "isPreview": True},
            )
            if create.status_code == 401:
                self._token_expiry = 0
                token = await self.ensure_token()
                headers["Authorization"] = f"Bearer {token}"
                create = await client.post(
                    f"{LOGICC_API_BASE}/chats",
                    headers=headers,
                    json={"assistantId": assistant_id, "isPreview": True},
                )
            if create.status_code >= 400:
                raise RuntimeError(
                    f"Logicc create chat returned HTTP {create.status_code}: "
                    f"{create.text[:800]}"
                )
            chat_id = create.json()
            if isinstance(chat_id, dict):
                chat_id = chat_id.get("id") or chat_id.get("chatId")
            if not isinstance(chat_id, str) or not chat_id:
                raise RuntimeError("Logicc create chat returned an unreadable chat ID")

            stream_headers = {
                **headers,
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "Stream-Response-Protocol": "ai-sdk-v6",
            }
            async with client.stream(
                "POST",
                f"{LOGICC_API_BASE}/chats/{chat_id}",
                headers=stream_headers,
                json={"isVoicePrompt": False, "modelId": model_id, "prompt": prompt},
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Logicc chat returned HTTP {response.status_code}: {body[:1000]}"
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].lstrip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    kind = item.get("type") if isinstance(item, dict) else None
                    if kind == "text-delta" and item.get("delta"):
                        yield "stream_token", {"token": item["delta"]}
                    elif kind == "error":
                        raise RuntimeError(item.get("errorText") or "Logicc stream error")
                    elif kind in {"finish", "finish-step"}:
                        yield "done", item

    async def _get_json(self, path: str) -> Any:
        token = await self.ensure_token()
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{LOGICC_API_BASE}/{path.lstrip('/')}", headers=headers)
            if response.status_code == 401:
                self._token_expiry = 0
                token = await self.ensure_token()
                headers["Authorization"] = f"Bearer {token}"
                response = await client.get(
                    f"{LOGICC_API_BASE}/{path.lstrip('/')}", headers=headers
                )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Logicc {path} returned HTTP {response.status_code}: {response.text[:800]}"
                )
            return response.json()

    async def models(self) -> list[dict[str, Any]]:
        data = await self._get_json("models?language=en")
        if not isinstance(data, list):
            raise RuntimeError("Logicc model catalog returned an unreadable response")
        models = [item for item in data if isinstance(item, dict) and item.get("id")]
        # Logicc injects Smart Select in its frontend instead of returning it from
        # /api/models. Its real chat model ID is `smart-select`.
        return [LOGICC_SMART_SELECT_MODEL, *models]

    async def diagnostics(self) -> dict[str, Any]:
        models, usage = await asyncio.gather(
            self._get_json("models?language=en"), self._get_json("organizations/usage")
        )
        selected = next(
            (item for item in models if isinstance(item, dict) and item.get("id") == LOGICC_MODEL_ID),
            None,
        ) if isinstance(models, list) else None
        return {
            **self.status(),
            "selected_model_metadata": selected,
            "organization_usage": usage,
            "notes": {
                "context_window": (
                    f"Logicc exposes a size category for {LOGICC_MODEL_ID}, "
                    "not a fixed numeric token count."
                ),
                "five_hour_window": (
                    "The reset countdown is supplied by secondsUntilFiveHourReset; "
                    "the client also honors RateLimit-Reset on HTTP 429."
                ),
            },
        }

    async def close(self) -> None:
        if self._assistant_ids and self._token:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await asyncio.gather(
                        *(
                            client.delete(
                                f"{LOGICC_API_BASE}/assistants/{assistant_id}",
                                headers={"Authorization": f"Bearer {self._token}"},
                            )
                            for assistant_id in self._assistant_ids.values()
                        ),
                        return_exceptions=True,
                    )
            except Exception:
                pass
            self._assistant_ids.clear()
        if self._listener_task is not None:
            self._listener_task.cancel()
            await asyncio.gather(self._listener_task, return_exceptions=True)
        if self._ws is not None:
            await self._ws.close()
        self._ws = None
        self._listener_task = None
        if self._chrome_process is not None and self._chrome_process.returncode is None:
            self._chrome_process.terminate()
            try:
                await asyncio.wait_for(self._chrome_process.wait(), timeout=5)
            except TimeoutError:
                self._chrome_process.kill()
        self._chrome_process = None


logicc_bridge = LogiccBridge()
