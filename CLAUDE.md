# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

BlackBox is a flight recorder for AI agents, written in Python: it records agent runs (spans plus the exact bytes of
every model and tool response), replays or forks them, scores them with metrics and LLM judges, clusters failures,
alerts on quality drops, and runs regression suites. **The repo holds only the plan so far.** Read
[docs/plan/README.md](docs/plan/README.md) and the current phase file before working; each phase file has tasks,
tests and "done when" items to tick, and ends up recording what was actually built.

## Working rules

- Commit on `main`, one small conventional commit per task (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`),
  pushed (plan D2). No attribution trailer in commit messages. The repo-local identity is
  `Arun Mariappan Karunanithi <2525449+arunmariappan@users.noreply.github.com>`.
- Python only (3.14, uv); the UI is Jinja2 templates plus HTMX, with no hand-written JavaScript.
- Every model call (agents under test, judges, descriptions) uses `qwen3.5:4b` in the host Ollama with `think` off.
  BlackBox's own calls use a JSON schema in Ollama's `format`, Pydantic validation, one retry, then an explicit
  `invalid` result.
- Ask the user before any GPU run longer than a few minutes (baselines, traffic, live suites, labelling runs): this PC
  has shut down during long GPU runs. Develop and test on replay.
- Changes to PaperPilot (the `BlackBox:Enabled` switch, phases 2–3) are committed in `D:\ai_workspace\PaperPilot`,
  following that repo's own CLAUDE.md.
- When a decision changes, update the plan (decisions D1–D17, risks R1–R11 in docs/plan/README.md) in the same
  commit.

## Commands (later phases add the ones marked with a phase)

```bash
uv sync                                   # install (Python 3.14, uv.lock)
uv run ruff format --check && uv run ruff check
uv run mypy                               # strict on src/
uv run pytest                             # unit + integration; never needs Ollama or a GPU
uv run pytest tests/unit/test_matching.py::test_name   # a single test
uv run blackbox serve                     # UI/API/OTLP on :8200, proxy listeners :8210-8213, worker
uv run blackbox db upgrade                # Alembic migrations (serve also runs them)
uv run blackbox db prune --older-than 30d # keeps baseline and labelled runs
uv run blackbox runs export <run> --out <dir>   # run bundle; `runs import <dir>` is idempotent
uv run blackbox run paperpilot "What are transformer architectures?"     # start a recorded run (phase 2)
uv run blackbox replay <run> [--from-step N | --auto-fork] [--model M] [--patch FILE]   # phase 4
uv run opsdesk env    # :8221      uv run opsdesk agent    # :8220        (phase 6)
uv run blackbox regress opsdesk-core --mode replay --spawn   # what CI runs (phase 10)
```

## Architecture

- **One process** (`blackbox serve`) runs on one asyncio loop: the FastAPI app (UI, REST API, OTLP/HTTP receiver at
  `/v1/traces` on 8200), one uvicorn listener per proxy upstream (8210 Ollama, 8211 PaperPilot OpenSearch, 8212 Jina,
  8213 OpsDesk env), and the background worker. Everything binds to `127.0.0.1`; upstream URLs use `127.0.0.1`, never
  `localhost` (Windows tries `::1` first and a refused IPv6 connection costs ~2 s).
- **Two capture channels joined by trace id.** The proxy stores each exchange with the `traceparent` it carried; the
  OTLP receiver stores spans. An exchange's parent span id is the agent's HTTP client span, whose ancestors give the
  agent node (e.g. PaperPilot's `guardrail_validation`). A run completes when its root span has ended, nothing has
  arrived for `quiet_seconds`, and no proxy call is in flight; then steps (one per exchange) are built and jobs queued.
- **Runs BlackBox starts carry a `traceparent` it chose**, so it knows the trace id up front. Replay relies on this:
  a session is keyed by its new trace id, and the proxy answers that trace's calls from the source run's tape.
- **Replay modes:** `exact` (409 on divergence), `fork` (tape before step N, live after), `auto_fork` (tape until the
  first request that differs, live after). Matching uses a hash of the canonical request after the profile's
  normalisers; normalisers affect matching only, never what is sent or served. Patches edit requests before matching.
  Stateful upstreams (OpsDesk env) add sync-forwarding and identifier aliasing; Python agents also get SDK shims for
  the clock and randomness.
- **Profiles** (`src/blackbox/profiles/`, one Python module per agent) hold everything agent-specific: how to start a
  run, which spans are nodes, how to read output and ending, normalisers, metrics, judges. A new agent needs only a
  profile.
- **Storage:** one SQLite file (`data/blackbox.db`, WAL, gitignored). All writes go through the single `StoreWriter`
  task; never open a second write connection. Bodies are zstd blobs keyed by SHA-256. Run bundles (`run.json` +
  `blobs/`) are the export format for `baselines/` and test fixtures.
- **OpsDesk** (`src/opsdesk/`) is a test agent that must look like an outside agent: it may import only
  `blackbox.sdk` from BlackBox (a test enforces this) and talks to BlackBox only through the proxy and OTLP.
- **Trust:** only judges that passed the agreement gate (κ ≥ 0.6 on ≥ 20 held-out labels) may decide failures, alerts
  or regression verdicts.
- **Secrets:** the proxy strips `authorization`, API-key headers and cookies before storing; a test scans committed
  bundles for anything key-like.

## Gotchas

- Git here has `core.autocrlf=true`; keep files LF (`.gitattributes` enforces `eol=lf`).
- Integration tests start a whole BlackBox in-process (`tests/harness.py`: real sockets on free ports, a copy of a
  database migrated once per session). The SDK's exporter posts from a background thread, so tests call
  `await asyncio.to_thread(sdk.flush)`, never `sdk.flush()` on the event loop (it would deadlock the in-process server).
- Listening sockets set `TCP_NODELAY` (`server.bind`); without it every proxied call on a kept-alive connection
  waits ~40 ms for a delayed ACK. Keep database writes off the proxy's request path (see `RunAssembler.open_call`).
- BlackBox's own outgoing HTTP calls use `net.make_client` (an untraced transport): in a process where the SDK
  instrumented httpx, instrumentation would otherwise overwrite the `traceparent` the proxy forwards or the runner
  sends on purpose.
- `ruff format` also formats Python code blocks inside Markdown; `*.md` is excluded in `pyproject.toml` so the plan's
  aligned comments stay as written.
- The wheel is built with hatchling (`packages = ["src/blackbox", "src/opsdesk"]`): uv's own build backend takes one
  top-level module.

## Spike findings (phase 0)

- **S4, Python 3.14 wheels:** every package in plan §4 installs and imports on CPython 3.14.8 under Linux
  (`onnxruntime` 1.30, `scikit-learn` 1.9, `scipy` 1.18, `numpy` 2.5, `fastembed` 0.8); `compression.zstd` is in the
  standard library. D1 stays at 3.14. Still to confirm on Windows: `uv sync` and
  `uv run python -c "import fastembed, sklearn, scipy, onnxruntime"`.
- **S1 (trace propagation), S2 (PaperPilot span attributes), S3 (Ollama from Python): not run yet.** Phase 0 to 9
  were built in a cloud container that cannot reach Ollama, PaperPilot or HuggingFace. Run them on this PC as written
  in `docs/plan/phase-0-bootstrap.md` and record the answers here. Until then the code assumes the expected answers
  (the incoming `traceparent` sets PaperPilot's trace id and outgoing calls carry it, with the HttpClient span as
  parent) and keeps the R1 fallback cheap: the proxy also accepts an `X-BlackBox-Session` header.
- `tests/fixtures/otlp/paperpilot-agentic.json` is **hand-written** from the plan's description of PaperPilot's spans
  (`python -m tests.fixtures.paperpilot` regenerates it). Replace it with S2's real export and fix the adapters in
  `otlp/genai.py` if PaperPilot's attributes differ.
- The `ollama` Python client (0.6) takes `think`, `format` (a JSON schema dict) and `tools` on `chat`, and is built on
  `httpx`, so the SDK's httpx instrumentation carries `traceparent` to the proxy.
