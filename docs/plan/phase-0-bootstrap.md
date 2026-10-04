# Phase 0: Bootstrap and spikes

**Goal:** a Python project that installs, lints, type-checks and tests in CI, and four short spikes that settle the
assumptions the rest of the plan depends on. The spikes' answers go into `CLAUDE.md` and, where they change the
design, into this plan.

**Size:** S · **Depends on:** nothing

## Prerequisites

- Python 3.14 (`py -0` lists 3.14.7) and uv 0.12 (`uv --version`). Both are installed.
- Ollama on the Windows host with `qwen3.5:4b` pulled. If Ollama isn't running, start the tray app detached, not from
  a Claude Code shell (an Ollama started from the shell dies with the session).
- PaperPilot's stack for spikes S1 and S2: Docker Desktop started, then `aspire start --apphost
  D:\ai_workspace\PaperPilot\src\PaperPilot.AppHost` from PowerShell. Keep Langfuse off.

## Tasks

### 0.1 Repository
- [x] The repo is cloned at `D:\ai_workspace\BlackBox` with the repo-local identity `Arun Mariappan Karunanithi
      <2525449+arunmariappan@users.noreply.github.com>` (done 2026-10-05). The plan is in `docs/plan/`.
- [x] `.gitignore` (Python, `data/`, `.env`, `.venv/`), `.gitattributes` (`* text=auto eol=lf`), `.editorconfig`.
- [x] `README.md`: one paragraph on what BlackBox is, a "status: in development" line, a link to the plan.
- [x] `CLAUDE.md`: commands, layout and gotchas. Start small; each phase adds to it.

### 0.2 Project and tooling
- [x] `uv init --package` layout with `src/blackbox` and `src/opsdesk` in one project; `.python-version` = `3.14`.
- [x] `pyproject.toml`: console scripts `blackbox = "blackbox.cli:app"` and `opsdesk = "opsdesk.cli:app"`; ruff
      (line length 120, rules `E,F,I,UP,B,SIM,ASYNC,RUF`), mypy strict on `src/`, pytest with `asyncio_mode = "auto"`.
- [x] `uv add` every runtime package from README §4 now, even if unused yet, so spike S4 proves they all install on
      3.14.
- [x] `blackbox --version` and `blackbox --help` work (Typer app with no real commands yet).
- [x] `.github/workflows/ci.yml`: on push and pull request, Ubuntu, `astral-sh/setup-uv`, `uv sync --locked`,
      `ruff format --check`, `ruff check`, `mypy`, `pytest`. One trivial test so the pipeline is green.

### 0.3 Spike S1: trace propagation through PaperPilot (risk R1)
A throwaway script in the scratchpad, not committed: a 40-line logging proxy (httpx + uvicorn) on port 8299 that
forwards to Ollama and prints each request's headers, path and body size.

- [ ] Start PaperPilot with its Ollama connection string pointed at the logging proxy, for this test only. The
      AppHost's `appsettings.json` has `Endpoint=http://localhost:11434`; a user secret overrides it:
      `dotnet user-secrets set ConnectionStrings:ollama "Endpoint=http://127.0.0.1:8299" --project src/PaperPilot.AppHost`
      and `dotnet user-secrets remove ConnectionStrings:ollama --project src/PaperPilot.AppHost` afterwards.
- [ ] Send `POST /api/v1/ask-agentic` with `{"query": "What are transformer architectures?", "model": "qwen3.5:4b"}` and
      a header `traceparent: 00-<32 hex chosen by you>-<16 hex>-01`.
- [ ] Record the answers in `CLAUDE.md`:
  - Is the response's `trace_id` the trace id you sent? (Expected: yes. ASP.NET Core continues an incoming trace, and
    `agentic_rag_request` nests under the server span.)
  - Does each Ollama request carry `traceparent` with that trace id? Which span is its parent id? (Expected: the
    HttpClient client span.)
  - Does OllamaSharp send `stream: true` or `false` for structured calls? What does the body look like (`format`,
    `options`, `think`)?
  - Do two identical questions produce byte-identical Ollama request bodies? If not, which fields differ? Those become
    PaperPilot's request normalisers in phase 4.
- [ ] If the trace id is **not** continued, write the fallback into phase 3 before going on: the runner sends an
      `X-BlackBox-Session` header and the proxy groups calls by the response's `trace_id` (R1).

### 0.4 Spike S2: PaperPilot's span attributes (risk R2)
- [ ] For the S1 run, export its trace with `aspire otel traces api --apphost src/PaperPilot.AppHost --trace-id <id>
      --format Json` and keep the file as `tests/fixtures/otlp/paperpilot-agentic-aspire.json` (after checking it
      holds no secret).
- [ ] List in `CLAUDE.md`: the span names and parentage (server span → `agentic_rag_request` → node spans → `chat
      qwen3.5:4b` → HTTP client span); which GenAI attributes or events Microsoft.Extensions.AI sets (for example
      `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.input.messages` or the
      older per-message events); and the Langfuse attributes on the node spans (`langfuse.observation.input`,
      `output`, `level`).

### 0.5 Spike S3: Ollama from Python
- [ ] A scratch script calls `qwen3.5:4b` through the `ollama` client with `think=False`: one structured call with a
      JSON schema in `format` (a judge-shaped `{verdict, rationale}`), one tool-calling call with two tools.
- [ ] Note in `CLAUDE.md`: does the structured output always validate? Does the tool call come back in `tool_calls`
      with valid JSON arguments? How long does each call take on this machine?

### 0.6 Spike S4: Python 3.14 wheels (risk R11)
- [ ] `uv sync` on Windows succeeds with every package from §4, and `python -c "import fastembed, sklearn, scipy,
      onnxruntime"` works. CI's `uv sync --locked` on Ubuntu succeeds too.
- [ ] If anything fails on 3.14, switch `.python-version` to 3.13, change D1, and note why.

## Tests
- [ ] CI is green on the first push: format, lint, types and one test.

## Done when
- [ ] `uv run blackbox --help` works on Windows, and CI is green on GitHub.
- [ ] `CLAUDE.md` records the answers to S1–S4, and any change they force is written into the affected phase files.
- [ ] The temporary `ConnectionStrings:ollama` user secret is removed from PaperPilot's AppHost.

## What was built (2026-10-04)

- Repo files, the uv project (`src/blackbox`, `src/opsdesk`, hatchling build because uv's backend takes one
  top-level module), every package from README §4 locked in `uv.lock`, the Typer CLIs `blackbox` and `opsdesk`, and
  the CI workflow (`uv sync --locked`, format, lint, mypy strict, pytest).
- The work ran in a Linux cloud container with no access to Ollama, PaperPilot, HuggingFace or the Windows PC, so:
  - **S4** was checked on Linux only: every package installs and imports on CPython 3.14.8 (`onnxruntime` 1.30 has
    3.14 wheels). Windows still needs one `uv sync`.
  - **S1, S2 and S3 are still open.** They need the running PaperPilot stack and Ollama. The code built in phases 1–9
    assumes their expected answers and keeps the R1 fallback (`X-BlackBox-Session`) available; `CLAUDE.md` lists
    what to check.
  - No PaperPilot user secret was set, so there is nothing to remove.
