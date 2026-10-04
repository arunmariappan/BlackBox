# Phase 3: Recording proxy

**Goal:** every model, search and embedding call PaperPilot makes during an agentic run passes through BlackBox's
proxy, which stores the exact request and response, joins each call to the span that made it, and builds the run's
steps from these exchanges. The agent notices nothing except a few milliseconds.

**Size:** M · **Depends on:** phase 2

## Design

### Listeners
- One listener per configured upstream (README §5), each a small Starlette app served by its own `uvicorn.Server` in
  the shared event loop. A catch-all route forwards every method and path.
- Forwarding uses one `httpx.AsyncClient` per upstream (`trust_env=False`, the upstream's timeout, connection pooling)
  and `send(..., stream=True)`.
- Hop-by-hop headers (`connection`, `keep-alive`, `transfer-encoding`, `te`, `trailer`, `upgrade`, `proxy-*`) are
  dropped; `host` is set to the target's. Everything else, including `traceparent` and `authorization`, is forwarded
  unchanged.
- Responses are streamed back chunk by chunk as raw bytes (`aiter_raw()`), so compressed or streamed responses (Ollama's
  NDJSON, OpenAI-style SSE) pass through untouched and in real time.

### Recording
- Only paths matching the upstream's `record_paths` are recorded; others (health checks, `/api/tags`) pass through.
- The request body is read in full before forwarding (model and search requests are small). Response chunks are copied
  to a buffer as they are sent, with the byte offset and time of each chunk, which becomes `chunk_times`.
- When the response ends, the exchange is written: method, path, query, headers with `redact_headers` removed (D15),
  request and response blobs, status, timings (start, first byte, end), and `request_key`, a SHA-256 of the canonical
  request (method, path, sorted query, JSON body with sorted keys).
- If the client disconnects mid-stream, the exchange is stored with `error = "client_aborted"` and the partial body. If
  the upstream fails (refused, timeout), the proxy answers 502 with a JSON error naming the upstream, and stores the
  exchange with the error.

### Attribution
- The `traceparent` header gives the trace id and the parent span id (the HTTP client span in .NET and in the SDK's
  httpx instrumentation).
- `seq` is a per-trace counter assigned when the request starts, so steps are ordered by request start.
- The proxy tells the run assembler when a call for a trace starts and ends, so a run isn't completed while a call is
  in flight.
- A call without `traceparent` (an uninstrumented agent) is stored without a trace and listed on an **Unattributed
  calls** page. If spike S1 forced the fallback, an `X-BlackBox-Session` header attributes it instead.

### Steps from exchanges
When a run with exchanges completes, its steps are rebuilt from them, one step per exchange in `seq` order:

| Upstream and path | Step kind | View |
|---|---|---|
| Ollama `/api/chat`, `/api/generate`; `/v1/chat/completions` | `llm` | model, messages, tools, `format`, options; reply message, tool calls, token counts and Ollama's timings; streamed chunks joined into one reply |
| Ollama `/api/embed`, Jina `/v1/embeddings` | `embedding` | model, input texts, dimensions (vectors not shown) |
| OpenSearch `/*/_search` | `tool` | the query, then hits with paper id, title, score and chunk text |
| OpsDesk env (phase 6) | `tool` | tool name, arguments, result |

The node comes from the span chain: the exchange's parent span (HTTP client) → its ancestors → the nearest node span.
The step also links the `chat` GenAI span above it, so the UI can show span timing and exact bytes together. A run
whose spans never arrive still gets its steps, with node `unknown`. Runs with exchanges are `replayable = true`.

### PaperPilot: proxy routing (second half of the switch)
In the PaperPilot repo, extending phase 2's `BlackBox:Enabled`:
- [ ] When enabled, the AppHost gives the **API** (not the worker; ingestion isn't recorded):
      `ConnectionStrings__ollama=Endpoint=http://127.0.0.1:8210`, `OpenSearch__Host=http://127.0.0.1:8211`,
      `Jina__BaseUrl=http://127.0.0.1:8212/v1/` (the default is `https://api.jina.ai/v1/`). Check that these values
      win over the ones `WithReference` and the existing `WithEnvironment` calls set.
- [ ] The API waits for BlackBox: an external resource for `http://127.0.0.1:8200` with a health check on `/health`,
      if Aspire 13's external-service resource supports it; otherwise `CLAUDE.md` says to start BlackBox first.
- [ ] The switch assumes the host Ollama; with `Ollama:UseContainer=true` it is ignored, with a warning in the log.
- [ ] Docs: README section "Recording runs with BlackBox", `CLAUDE.md` gotcha, `docs/decisions.md` entry updated.

## Tasks

### 3.1 Proxy core
- [x] `blackbox/proxy/listener.py`: app factory per upstream, forwarding, header handling, streaming, error mapping.
- [x] `blackbox/proxy/recorder.py`: buffering with chunk times, redaction, `request_key`, exchange write, in-flight
      notifications.
- [x] `blackbox serve` starts every configured listener; `/health` reports each listener and its upstream's
      reachability.

### 3.2 Views
- [x] `blackbox/proxy/views/`: parsers for Ollama chat and generate (streamed and not), OpenAI chat, Ollama and Jina
      embeddings, OpenSearch search. Each tested on a recorded body.

### 3.3 Steps and UI
- [x] Step builder from exchanges with node lookup through the span chain.
- [x] Run page: each step shows the exact request and response (pretty-printed JSON, a raw tab, streamed chunks with
      their times), the "recorded" badge, and the span timing next to it.
- [x] Unattributed calls page.

### 3.4 PaperPilot routing
- [ ] The PaperPilot changes above, committed in the PaperPilot repo.

## Tests
- [x] Against a fake upstream: a JSON call, an NDJSON stream of 50 chunks and an SSE stream all arrive at the client
      byte-identical and the stored response equals what was sent; the first streamed chunk reaches the client before
      the upstream finishes.
- [x] `authorization` and `x-api-key` are forwarded but not stored.
- [x] Upstream refused → 502 with the upstream's name; client abort → stored with `client_aborted`.
- [x] Unrecorded paths pass through without a row.
- [x] A run isn't completed while a call is in flight, even after the root span ended.
- [x] Step builder on a fixture (exchanges + spans of one PaperPilot run): kinds, order and nodes as expected.
- [x] Added latency per call under 5 ms at the median on a local fake upstream (measured, printed by the test).

## Done when
- [ ] `blackbox run paperpilot "What are transformer architectures?"` produces a run whose steps are, in order:
      guardrail `llm`, Jina `embedding`, OpenSearch `tool`, grading `llm`, generation `llm` (more with a rewrite), each
      with its node, exact bytes and timings.
- [ ] A script that reads the Jina key from PaperPilot's user secrets (without printing it) finds it nowhere in
      `data/blackbox.db`.
- [ ] PaperPilot's classic `/stream` still streams token by token through the proxy.
- [ ] One real run's bundle is exported to `tests/fixtures/bundles/paperpilot-answered/` for phase 4's tests.

## What was built (2026-10-04)

- `proxy/listener.py` is a raw ASGI app per upstream (not Starlette responses), so it can watch for the client
  hanging up while it waits for the next upstream chunk; `proxy/core.py` holds forwarding, recording and attribution;
  `proxy/views.py` the parsers; `proxy/manager.py` starts the listeners from `blackbox serve` and adds each
  upstream's reachability to `/health`. The recorder lives in `proxy/core.py` (`Recording`, `write_exchange`).
- **Latency:** a call never waits for the database. The run row for a trace first seen at the proxy is written in
  the background (`RunAssembler.open_call`), `seq` comes from an in-memory counter (open runs are preloaded after a
  restart), and the exchange is written by its own task after the response has gone back, so a kept-alive connection
  takes the agent's next call at once. The test measures about 3 ms median added per call on a local fake upstream,
  each call being a new trace (the slowest case).
- **Found on the way:** listening sockets need `TCP_NODELAY`. Without it every response on a kept-alive connection
  waited ~40 ms for the client's delayed ACK (Nagle), on every proxied call.
- Redaction drops the configured headers and any header whose name looks like a key, token, secret or password, in
  both directions (`set-cookie` too). Compressed responses are stored raw and decoded only for views.
- Errors: refused or timed-out upstream → 502 `{"error": "blackbox_upstream_error", "upstream": ...}`, stored with
  `error = "upstream_error: ..."`; client hang-up → `client_aborted` with the partial body; an upstream that breaks
  mid-stream → `upstream_aborted: ...`.
- Risk R1's fallback is in place: without `traceparent`, a 32-hex `X-BlackBox-Trace-Id` header attributes the call.
- **Not done here (needs PaperPilot and its stack):** 3.4 PaperPilot routing, and the four "Done when" items.
  `tests/fixtures/paperpilot.py` builds a synthetic run with exchanges (`insert_run`) in place of the real bundle
  `tests/fixtures/bundles/paperpilot-answered/`; export a real one with `blackbox runs export` once a run is recorded.
