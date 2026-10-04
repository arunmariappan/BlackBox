# Phase 2: Live capture

**Goal:** agents export their spans to BlackBox, and BlackBox turns each trace into a run with steps, an output and an
ending. A real PaperPilot `/ask-agentic` run shows up in the UI with its node tree and every model call's messages.
Python agents get a small SDK that emits the same structure.

**Size:** M · **Depends on:** phase 1

## Design

### OTLP receiver
- `POST /v1/traces` on port 8200, accepting `application/x-protobuf` (decoded with `opentelemetry-proto`'s
  `ExportTraceServiceRequest`) and `application/json` (protobuf's JSON mapping), with optional gzip. Bodies over 16 MB
  are refused with 413.
- The receiver decodes, hands the spans to the writer and the run assembler, and answers with an empty
  `ExportTraceServiceResponse`. It doesn't wait for scoring.

### GenAI normalisation
`blackbox/otlp/genai.py` turns any span into a `SpanView` with optional parts. Each adapter reads one dialect and is
tested on a recorded fixture:

| Dialect | Read from |
|---|---|
| Current GenAI conventions (version pinned in this phase, from spike S2) | `gen_ai.operation.name`, `gen_ai.provider.name` (or older `gen_ai.system`), `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.definitions`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`, `gen_ai.response.finish_reasons`, `gen_ai.agent.name` |
| Older per-message events | `gen_ai.system.message`, `gen_ai.user.message`, `gen_ai.assistant.message`, `gen_ai.tool.message`, `gen_ai.choice` events |
| Microsoft.Extensions.AI (PaperPilot) | whatever S2 finds beyond the above |
| Langfuse attributes (PaperPilot's node spans) | `langfuse.observation.input`, `langfuse.observation.output`, `langfuse.observation.level`, `langfuse.observation.status_message`, `langfuse.trace.input`, `langfuse.trace.output`, `langfuse.*.metadata.*` |
| HTTP client spans | `http.request.method`, `url.full`, `server.address`, `server.port`, `http.response.status_code` |

Attributes that no adapter reads stay in `spans.attributes`, so a new dialect can be added later and re-run over old
spans.

### Run assembly
- A trace becomes a run when its first span arrives (`runs.status = open`). If BlackBox started the run, the row
  already exists with its entry request.
- The `RunAssembler` keeps per open trace: the time of the last span, whether a root span has ended, and (from phase 3)
  how many proxy calls are still in flight. A one-second tick completes a run when its root span has ended, nothing has
  arrived for `runs.quiet_seconds` (5), and nothing is in flight.
- **Root span:** the span with no parent, or whose parent isn't in the trace (a run started by BlackBox has a remote
  parent, the made-up span id in the `traceparent` it sent).
- **Profile match:** on completion, each profile's `matches(spans)` decides which agent the run belongs to.
  PaperPilot's profile matches a trace containing `agentic_rag_request`. Traces no profile matches (classic `/ask`,
  health checks) are deleted, unless `runs.keep_unmatched = true`.
- **Restart safety:** on startup, runs still `open` and older than the quiet period are queued for completion.
- **Completion** writes the steps, output and ending, and queues a `run_completed` job (phase 9 builds on it).

### Steps from spans
Until phase 3 adds exchanges, steps come from spans: every GenAI span (`chat`, `execute_tool`, `embeddings`) becomes a
step, ordered by start time. Its **node** is the nearest ancestor whose name is one of the profile's node spans, so a
`chat qwen3.5:4b` span under `document_grading` becomes step *n*, node `document_grading`. Runs built only from spans
are marked `replayable = false`.

### Profiles (first version)
`blackbox/profiles/base.py` defines `Profile` with `name`, `matches(spans)`, `node_spans`, `start(input, traceparent)`
(how to start a run), `read_output(run)`, `ending(run)` and, from later phases, upstreams, metrics and judges.

PaperPilot's profile (`profiles/paperpilot.py`):

| Item | Value |
|---|---|
| Start a run | `POST http://127.0.0.1:8100/api/v1/ask-agentic` with `{query, top_k, use_hybrid, model: "qwen3.5:4b", categories}` |
| Matches | a span named `agentic_rag_request` |
| Node spans | `guardrail_validation`, `document_retrieval_initiation`, `document_grading`, `query_rewriting`, `answer_generation` |
| Output | the HTTP response when BlackBox started the run; otherwise `langfuse.trace.output` on the root span |
| Ending | from the last reasoning step: `Generated answer from context` → `answered`, `Responded as out of scope` → `out_of_scope`, `Stopped after {n} retrieval attempts` → `max_attempts`, `Search was unavailable` → `search_unavailable`. Without a response, from which node spans ran. |

`blackbox run paperpilot "What are transformer architectures?"` starts a run with a `traceparent` BlackBox chose,
stores the request and response, and prints the run's URL in the UI.

### Python SDK (first version)
`blackbox.sdk` gives Python agents the same structure without hand-written OpenTelemetry code:

```python
from blackbox import sdk

sdk.init(service_name="opsdesk-agent", endpoint="http://127.0.0.1:8200")   # OTLP exporter, httpx instrumentation

with sdk.agent_run("opsdesk", input=request, headers=incoming_headers) as run:   # continues an incoming traceparent
    response = sdk.traced_chat(ollama_client, model="qwen3.5:4b", messages=messages, tools=tools)
    result = get_logs(service="checkout-api")     # @sdk.tool-decorated: an execute_tool span with arguments and result
    run.set_output(final_answer)
```

- `agent_run` makes the root span `invoke_agent opsdesk` with `gen_ai.operation.name = invoke_agent`.
- `traced_chat` makes a `chat {model}` span with the request model, input and output messages, tool definitions and
  token usage, in the current conventions.
- `@sdk.tool` makes an `execute_tool {name}` span.
- The exporter's batch delay is 500 ms, so runs complete quickly.

### PaperPilot: trace exporter (first half of the switch)
In the PaperPilot repo, with its own conventions and tests:
- [ ] `ServiceDefaults/BlackBoxExporter.cs`: when `BlackBox:OtlpEndpoint` is set, a tracer provider of its own (R8),
      like `LangfuseExporter`, that listens to `PaperPilot.*`, `Microsoft.Agents.AI*`, `Microsoft.AspNetCore` (server
      spans) and `System.Net.Http` (client spans, which phase 3 needs to join proxy calls to spans), and exports over
      OTLP/HTTP protobuf with a one-second batch delay.
- [ ] `AppHost.cs`: `BlackBox:Enabled` (default `false`) passes `BlackBox__OtlpEndpoint=http://127.0.0.1:8200/v1/traces`
      to the API only.
- [ ] A test that the exporter is registered only when the endpoint is set; a `CLAUDE.md` gotcha and a
      `docs/decisions.md` entry describing the switch.

## Tasks

### 2.1 Receiver
- [ ] `blackbox/otlp/receiver.py`: route, content types, gzip, size limit, response.
- [ ] `blackbox serve`: uvicorn with the FastAPI app; startup runs migrations and starts the writer and assembler.

### 2.2 Normalisation
- [ ] `SpanView` and the adapters in the table, each with a fixture test: S2's PaperPilot trace, a trace from the
      SDK, and a hand-written trace in the older event dialect.

### 2.3 Run assembly and steps
- [ ] `RunAssembler` with completion rules, profile matching, deletion of unmatched traces and restart recovery.
- [ ] Step builder from spans, node lookup, token totals per run.

### 2.4 Profiles
- [ ] `Profile` base class, PaperPilot profile, and `blackbox run <profile> <input>`.

### 2.5 SDK
- [ ] `sdk.init`, `agent_run`, `traced_chat`, `@tool`, with a test that runs a fake agent against an in-process
      BlackBox and checks the stored run.

### 2.6 UI and API
- [ ] Base layout (Jinja2, HTMX and its SSE extension served from `static/`, light and dark themes).
- [ ] **Runs** page: time, profile, source, status, ending, steps, tokens, duration; filters by profile, ending and
      source; new and completed runs appear live over SSE.
- [ ] **Run** page: input, output and ending at the top; the step list, each step expandable to its messages (system,
      user, assistant, tool) and tool calls, long text collapsed; a span waterfall drawn server-side with CSS bars; a
      raw spans tab.
- [ ] REST: `GET /api/runs`, `/api/runs/{id}`, `/api/runs/{id}/steps`, `/api/runs/{id}/spans`; SSE at `/api/events`.
- [ ] CLI: `blackbox runs list`, `blackbox runs show <id>` (a Rich tree of nodes and steps).

## Tests
- [ ] Protobuf and JSON bodies of the same trace produce identical stored spans; gzip works; 413 over the limit.
- [ ] Spans of one run arriving in three batches, out of order, give one run with the right root and steps.
- [ ] A run isn't completed while its root span is still open, and is completed 5 s after the last span.
- [ ] An unmatched trace is deleted on completion.
- [ ] PaperPilot fixture: five node spans, LLM steps with their node, ending `answered`.
- [ ] After a restart, open runs are completed.

## Done when
- [ ] With the PaperPilot switch on, `blackbox run paperpilot "What are transformer architectures?"` produces a run
      whose page shows the node tree, each `chat` step with its prompt and answer, token counts and the ending.
- [ ] A question asked in PaperPilot's own web UI also appears as a run (output read from the root span).
- [ ] The SDK's fake agent shows up the same way.
