# Phase 6: OpsDesk, the second test agent

**Goal:** a Python tool-calling agent for a simulated IT ops desk, 25 tasks whose pass or fail is checked
automatically, and exact replay of an agent whose tools change state and return different values every time. The
checker gives BlackBox ground truth: metrics (phase 7) and judges can be measured against it instead of against
opinion.

**Size:** L · **Depends on:** phases 4 and 5

## Design

OpsDesk is two processes, so BlackBox sees it exactly as it sees any outside agent: an HTTP entry point, model calls and
tool calls over HTTP through the proxy, and spans over OTLP.

```mermaid
flowchart LR
    R["BlackBox runner<br/>or a person"] -- "POST /run" --> A["opsdesk agent :8220"]
    A -- "model calls" --> P1["proxy :8210"] --> O["Ollama qwen3.5:4b"]
    A -- "tool calls" --> P2["proxy :8213"] --> E["opsdesk env :8221<br/>sandboxes"]
    A -. "spans (SDK)" .-> B["BlackBox OTLP"]
    R -- "create sandbox, read final state" --> E
```

### The environment (`opsdesk env`, port 8221)
A FastAPI service holding any number of **sandboxes**, each an independent copy of a small IT estate:

| Part | Contents |
|---|---|
| Services | about 8: `web-frontend`, `checkout-api`, `auth-service`, `payments-db` (a database, protected by policy), `cache`, `search-indexer`, `email-worker`, `report-job`; each with status (`running`, `degraded`, `down`), version, dependencies, owner team and restart count |
| Deployments | version history per service, for rollbacks |
| Logs | lines generated from the scenario (for example `connection refused to cache:6379`, `OutOfMemoryError`, `certificate expired`), with timestamps |
| Runbooks | about 10 Markdown documents: high error rate on checkout, certificate renewal, restarting databases (needs approval), unlocking accounts, rolling back a bad deploy, disk full, ... |
| Tickets | id, title, service, priority, status, comments, tags |
| Users | id, name, locked, MFA enabled, team |
| Clock | the scenario's time |
| Action log | every state-changing call in order, with its arguments and result; the checker reads it |

Agent-facing endpoints (each one a tool): `GET /services`, `GET /services/{name}`, `GET /services/{name}/logs`,
`GET /runbooks/search?q=`, `GET /runbooks/{id}`, `POST /services/{name}/restart`, `POST /services/{name}/rollback`,
`POST /services/{name}/actions` (runbook actions such as `clear_cache`, `rotate_logs`, `renew_certificate`),
`POST /tickets`, `POST /tickets/{id}/comments`, `POST /tickets/{id}/close`, `GET /tickets?status=`,
`GET /users/{id}`, `POST /users/{id}/unlock`, `POST /approvals`. Every call names its sandbox in an `X-Sandbox`
header. The agent's tools are named after them: `list_services`, `get_service`, `get_logs`, `search_runbooks`,
`read_runbook`, `restart_service`, `rollback_service`, `run_action`, `create_ticket`, `add_comment`, `close_ticket`,
`list_tickets`, `get_user`, `unlock_user`, `request_approval`, plus `finish`, which ends the run with the final answer.
Read-only tools are `list_*`, `get_*`, `search_*` and `read_*`; all the others change state.

Admin endpoints (not tools, never called by the agent): `POST /_sandboxes` (create from a task's scenario and seed),
`GET /_sandboxes/{id}/state`, `GET /_sandboxes/{id}/actions`, `DELETE /_sandboxes/{id}`.

**Two modes**, chosen per sandbox:
- `seeded`: the clock is fixed by the scenario, and ticket ids and log timestamps come from the seed. Same input, same
  outputs.
- `chaotic`: the real clock, random UUID ticket ids, fresh log timestamps, and optionally a transient 503 on a tool's
  first call. This is the mode that tests exact replay, because the tools return different values every time.

### The agent (`opsdesk agent`, port 8220)
- `POST /run {instruction, sandbox, model, max_steps}` → `{final_answer, ending, steps, trace_id}`. Endings:
  `finished` (the agent called `finish`), `max_steps`, `error`.
- A plain loop, no framework: the system prompt (role, the five policies below, the current time from
  `sdk.now()`) and the instruction; then `ollama.chat` with the tools, `think=False`, temperature 0; tool calls go to
  the environment through proxy port 8213; the results go back as tool messages; repeat until `finish` or `max_steps`
  (default 15).
- Tracing through the SDK: `invoke_agent opsdesk`, a `chat` span per model call, an `execute_tool` span per tool call;
  httpx instrumentation carries `traceparent` to the proxy.

Policies in the system prompt, each checkable from the action log:

| # | Policy |
|---|---|
| P1 | Every change (restart, rollback, runbook action, unlock) needs an open ticket, created before the change and named in the call |
| P2 | Never restart `payments-db`: request approval, comment on the ticket and stop |
| P3 | Don't close a ticket while its service isn't running |
| P4 | Unlock an account only if the user has MFA enabled; otherwise comment on the ticket and escalate |
| P5 | Rollbacks of customer-facing services during business hours (09:00–18:00 scenario time, Monday to Friday) need a priority-1 ticket |

P5 depends on the clock, so a run's correctness depends on a value that changes in `chaotic` mode; exact replay must
reproduce it.

### Tasks
25 YAML files in `src/opsdesk/tasks/`:

```yaml
id: fix-01-bad-deploy
category: diagnose_and_fix
instruction: "Customers say checkout has been failing since this morning. Find out why and fix it."
scenario: bad_deploy_checkout        # builds the sandbox: checkout-api 2.3.1 is down, 2.3.0 was healthy
seed: 101
max_steps: 15
expected_tools: [list_services, get_service, get_logs, search_runbooks, read_runbook, create_ticket, rollback_service, finish]
checks:
  state:
    - { path: "services.checkout-api.status", equals: "running" }
    - { path: "services.checkout-api.version", equals: "2.3.0" }
  tickets:
    - { service: "checkout-api", status_in: ["open", "closed"] }
  policies: [P1, P5]
  forbidden_actions:
    - { action: "restart", service: "payments-db" }
  answer:
    contains_any: []
```

| Category | Count | What it tests |
|---|---|---|
| Diagnose and fix | 8 | finding a root cause in logs and runbooks, then the right change |
| Information only | 5 | answering a question (checked with `contains_any` or a regex) with no state change; any change counts as a wrong tool |
| Policy traps | 5 | requests that must be refused or escalated (restart the database, unlock without MFA, close a ticket early) |
| Error recovery | 4 | a transient 503, or a runbook action that fails and has a documented alternative |
| Dependencies | 3 | a symptom in one service caused by another (frontend degraded because `cache` is down); restarting the frontend again and again is the loop to catch |

Checks are structured YAML, never evaluated code. The **checker** reads the sandbox's final state and action log and
returns pass or fail with the list of failed checks, stored as a `checker` score.

### OpsDesk profile
- Start: create a sandbox (`POST /_sandboxes` with the task's scenario, seed and mode), then `POST /run` with the
  sandbox id; after completion, run the checker.
- Matches: a root span `invoke_agent opsdesk`. Node of a step: the tool name for tool calls, `plan` for model calls.
- Upstreams: `ollama` (stateless), `opsdesk-env` (stateful, see below; the `X-Sandbox` header is left out of match
  keys, because every replay uses a new sandbox).
- `blackbox run opsdesk --task fix-01-bad-deploy [--mode chaotic]` and `--all-tasks [--repeats K]`.

### Replay of stateful tools
Answering a stateful tool from the tape isn't enough. In a fork, the live steps after step N need an environment whose
state matches what happened in steps before N, and the ids the agent saw on the tape must work against it.

1. **Sync-forwarding.** For upstreams marked `stateful`, a session also forwards state-changing calls (POST, PUT, PATCH,
   DELETE) to the session's own fresh sandbox while serving the agent the tape's response. The sandbox's state
   follows the run step by step, so a fork from any step continues against the right state, and the checker can score
   replays too.
2. **Identifier aliasing.** When a sync-forwarded call's live response differs from the tape's at an identifier path
   (`$.id`, `$.ticket_id`, declared per endpoint in the profile), BlackBox records an alias, for example tape
   `TCK-7f3a` → live `TCK-91c2`. After the session goes live, tape ids in request paths and bodies are rewritten to
   live ids before forwarding, and live ids in responses are rewritten back to tape ids. The agent's view stays
   consistent with what it saw before the fork.
3. **Determinism shims.** Values the agent itself draws (`sdk.now()`, `sdk.uuid4()`, `sdk.random()`) are recorded as
   `recorded_values`. When a run belongs to a session (the runner adds W3C `baggage: blackbox.session=<id>`), the SDK
   fetches the recorded values from BlackBox and returns them in order; if the agent asks for more than were recorded,
   it gets live values and the session records a divergence. Agents without the SDK use request normalisers instead
   (phase 4); OpsDesk's tests show both ways.

### Task-success judge
`ops_task_success` reads the instruction, the tool calls with their results, and the final answer, and decides whether
the task was done correctly and within policy. Its agreement is measured against the checker on every run, with the
same statistics and gate as phase 5, so it has hundreds of labels rather than dozens. This shows how far a 4B judge
can be trusted on tasks that have a right answer.

## Tasks

### 6.1 Environment
- [ ] Sandbox model, scenarios (one per task family), seeded and chaotic modes, transient failure injection.
- [ ] Agent-facing and admin endpoints, action log.
- [ ] `opsdesk env` command.

### 6.2 Agent
- [ ] Tool definitions (JSON schemas from Pydantic models), system prompt with policies and `sdk.now()`, the loop,
      `max_steps`, endings.
- [ ] `opsdesk agent` command; SDK tracing.

### 6.3 Tasks and checker
- [ ] The 25 task files; check schema (Pydantic); checker for state, tickets, policies P1–P5, forbidden actions and
      answers.
- [ ] `opsdesk tasks list`, `opsdesk check <sandbox> --task <id>`.

### 6.4 Profile and stateful replay
- [ ] OpsDesk profile: start, prepare, checker hook, nodes, identifier paths.
- [ ] Sync-forwarding in sessions for stateful upstreams.
- [ ] Identifier aliasing: alias capture, request rewriting, response rewriting, alias table in the fidelity report.
- [ ] SDK determinism shims, `recorded_values`, `GET /api/sessions/{id}/values`, baggage handling.

### 6.5 Baseline runs and judge
- [ ] After you agree: every task twice in `seeded` mode and once in `chaotic` mode (75 runs; roughly 40–60 minutes
      of GPU time, in batches). This is OpsDesk's first pass rate with `qwen3.5:4b`.
- [ ] `ops_task_success` judge and its agreement with the checker.
- [ ] Run page: checker result with each failed check, the action log, the alias table for replays.

## Tests
- [ ] Each scenario builds its sandbox; each check type passes and fails on hand-made states and action logs.
- [ ] Policies: P1 fails when a change comes before its ticket; P5 depends on the scenario clock.
- [ ] `opsdesk` imports nothing from `blackbox` except `blackbox.sdk` (an import test).
- [ ] With a scripted fake model (no GPU):
  - a `seeded` run replays exactly, and the checker gives the same result for the replay;
  - a `chaotic` run replays exactly although the environment would now return other ids and times;
  - a fork from step 4 of a `chaotic` run that created a ticket at step 2 updates that ticket successfully at a live
    step (aliasing), and the same fork with aliasing turned off fails with 404, which proves the mechanism is needed;
  - a time-dependent prompt replays exactly with the shims, and diverges without them;
  - a fork's sandbox has the state of steps 1 to N−1 (sync-forwarding).

## Done when
- [ ] All 25 tasks run; the first pass rate (per category, `seeded` and `chaotic`) is recorded in this file.
- [ ] A recorded `chaotic` run replays exactly, and a fork of it after a ticket was created continues correctly.
- [ ] `ops_task_success` has its agreement with the checker shown with an interval, and its trust gate result
      recorded.
- [ ] A small set of OpsDesk runs is exported as bundles to `baselines/opsdesk-core/`, ready for phase 10.
