# Seclai / Logicc -> OpenAI compatible proxy

This local FastAPI service wraps either a Seclai Agent or the signed-in Logicc
web chat as an OpenAI-compatible API for SillyTavern and similar clients.

Implemented endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- Streaming and non-streaming responses

## Logicc web-chat mode

Set `UPSTREAM_PROVIDER=logicc` in `.env` and start the service. A normal Chrome
window opens using `.logicc-browser-profile`, with a DevTools endpoint bound only
to `127.0.0.1`. Sign in to Logicc once in that window and leave it open. There is
no Playwright automation: the proxy only observes Logicc API authorization
headers through local Chrome DevTools so it can keep the short-lived web session
current, then calls the same chat endpoints as the site.

`GET /v1/models` mirrors Logicc's current model catalog and also includes the
frontend-only `Smart Select AI` router (`smart-select`). Requests may use either
the displayed model name or Logicc's internal ID. The proxy maintains an unsaved
runtime Assistant per selected model whose native Logicc system prompt defines
the message-envelope semantics. Every OpenAI request creates an Assistant
preview chat, so it does not reuse unrelated chats from the sidebar. Runtime
Assistant drafts are deleted during clean shutdown.

Generation requests are concurrent. Only the first runtime-Assistant creation
for a given model is locked; after that, each request uses its own preview chat
and can stream independently.

## Start

```powershell
.\start.ps1
```

The server listens at `http://127.0.0.1:5100`. Check it with:

```powershell
Invoke-RestMethod http://127.0.0.1:5100/health
Invoke-RestMethod http://127.0.0.1:5100/logicc/status
```

## SillyTavern

- API type: OpenAI-compatible / Custom OpenAI
- Base URL: `http://127.0.0.1:5100/v1`
- API key: the value of `PROXY_API_KEY`; if empty, any non-empty placeholder
- Model: choose any model returned by `/v1/models`, including `Smart Select AI`
  or a fixed model such as `Claude 4.6 Opus`
- Streaming: supported

Logicc's chat endpoint accepts one `prompt`, not an OpenAI `messages` array. The
proxy serializes the request as a structured JSON envelope instead of
concatenating text. `system`/`developer` messages stay in a separate instruction
array; conversation roles, names, order, multimodal block boundaries, and tool
metadata are preserved. A runtime Logicc Assistant supplies the native system
instruction that authorizes and explains this envelope.

Preserving an OpenAI image or tool block in the envelope is not the same as
native support. This bridge does not currently upload images to Logicc's file
API and does not translate Logicc stream parts into OpenAI `tool_calls`; clients
should treat image understanding and arbitrary function calling as unsupported
until those translations are implemented and tested end to end.

The proxy leaves input and total-context limits to the client and Logicc:

- Context: no proxy-side token limit (`MAX_CONTEXT_TOKENS=0`)
- Input: no proxy-side token limit (`MAX_INPUT_TOKENS=0`)
- Output: 65,536 tokens (default request: 4,096; SillyTavern's 32,800 is accepted)

Logicc itself currently rejects a serialized prompt longer than 120,000
characters. The proxy does not truncate such requests.

Logicc describes Claude 4.6 Opus's context category as `gigantic`, but its web
metadata does not publish fixed numeric input/output limits. Change
`MAX_CONTEXT_TOKENS`, `MAX_INPUT_TOKENS`,
`MAX_OUTPUT_TOKENS`, and `DEFAULT_OUTPUT_TOKENS` in `.env` if verified upstream
limits become available.

`GET /logicc/diagnostics` reads the account's current model metadata and
organization usage after login. Logicc's five-hour protection allows at most
20% of the organization's monthly usage budget in a five-hour window. The
backend exposes the remaining reset time as `secondsUntilFiveHourReset`; HTTP
429 responses may also carry `RateLimit-Reset`.

## Seclai mode

Set `UPSTREAM_PROVIDER=seclai`, then configure `SECLAI_AGENT_ID`,
`SECLAI_API_KEY`, and `MODEL_ID`. Seclai mode continues to use the agent stream
endpoint and does not open Chrome.

Keep the service bound to localhost unless `PROXY_API_KEY` is set and network
access is intentionally secured.
