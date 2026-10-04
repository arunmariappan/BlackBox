# Phase 7: Run metrics

**Goal:** deterministic, cheap metrics on every completed run: wrong tool chosen, loops, recovery after errors, wasted
steps, fallbacks, cost and latency. They're checked against OpsDesk runs where the right answer is known, so a metric
that flags something is known to be right.

**Size:** M · **Depends on:** phase 6

## Design

Metrics are plain Python functions over a run's steps, spans, output and (for OpsDesk) the task file. They run on
every completed run, take milliseconds, need no model, and are stored as `metric` scores with a details object that
names the steps involved, so the UI can point at them. Each metric module has a version constant;
`blackbox metrics recompute [--profile P]` recomputes stored runs after a change.

A **step signature** is the tool name plus its arguments after normalisation (sorted keys, ids from aliasing mapped
back to tape ids), or, for a model call, the match key of its request. Several metrics compare signatures.

### Generic metrics (every profile)

| Metric | Definition |
|---|---|
| `steps`, `llm_calls`, `tool_calls` | counts of steps by kind |
| `input_tokens`, `output_tokens` | sums over model steps |
| `duration_ms`, `llm_ms`, `tool_ms` | run wall time; time spent in model and tool steps |
| `tool_errors` | tool steps with HTTP status ≥ 400, a transport error, or an error status on their `execute_tool` span |
| `recovered_errors`, `unrecovered_errors` | an error is recovered if a later step calls the same tool (or another tool the profile puts in the same group) successfully before the run ends; otherwise unrecovered |
| `repeated_calls` | read-only tool calls whose signature equals an earlier one with no state-changing call in between |
| `invalid_tool_calls` | calls to tools that don't exist, or with arguments the tool rejected as invalid (422) |
| `loop` | true when (a) one signature occurs 3 or more times, (b) a cycle of 2 or 3 signatures repeats twice in a row (for example `get_service → restart_service → get_service → restart_service`), or (c) the agent sends an identical model request twice. The details name the pattern and its first step. |
| `wasted_steps`, `wasted_ratio` | `repeated_calls + invalid_tool_calls`, and that number divided by `steps` |
| `fallbacks` | spans the agent marked as degraded (PaperPilot's `langfuse.observation.level` = `WARNING` or `ERROR`) and model replies that failed the agent's own output schema |
| `ending` | the profile's ending, stored as a label |

### OpsDesk metrics (from the task file and the checker)

| Metric | Definition |
|---|---|
| `wrong_tool` | a state-changing tool not in the task's `expected_tools`; any state-changing call in an information-only task; or a call on a target the task lists as `wrong_targets` (restarting `web-frontend` when `cache` is the cause) |
| `extra_read_tools` | read-only tools outside `expected_tools`; a softer signal than `wrong_tool` |
| `policy_violations` | the policies the checker found broken (P1–P5) |
| `steps_over_optimal` | `steps` minus the task's `optimal_steps` (a new field in each task file) |
| `checker_pass` | the checker's verdict, also stored as a `checker` score |

### PaperPilot metrics (from steps and output)

| Metric | Definition |
|---|---|
| `retrieval_attempts`, `rewrites`, `empty_retrievals` | from the node steps and search hits |
| `guardrail_score` | the score in the guardrail step's reply |
| `relevant_gradings` | gradings that answered `yes` |
| `citations_present` | the answer contains at least one `[arXiv:...]` citation (answered runs) |
| `citations_valid` | every cited id is among the run's sources |
| `wasted_rewrite` | a rewrite after a `yes` grading (PaperPilot's B5 says this never happens; the metric checks it) |
| `scope_mismatch` | for runs from the question set, the ending disagrees with the expected ending |

### Checking the metrics
- **Planted runs:** scripted fake-model transcripts run against the real OpsDesk environment (no GPU), each with known
  behaviour: a clean run, a three-times restart loop, a two-step cycle, a state change in an information-only task, a
  503 followed by a successful retry, a 503 followed by giving up, duplicate reads, a call with invalid arguments.
  Every metric must flag exactly what was planted and nothing on the clean run.
- **Real runs:** you review 20 OpsDesk runs from phase 6 on the run page and mark disagreements with the metrics
  (a label question `metrics_correct` with a note). The result is recorded in this file.

### Aggregates and UI
- **Profile overview** page: per profile and time window, rates (loop, wrong tool, unrecovered error, fallback,
  checker pass) and distributions (steps, tokens, latency p50 and p95), as Plotly charts over time.
- Run page: a metrics panel; flagged metrics link to the steps in their details.
- Runs list: filters on flags (`loop`, `wrong_tool`, `unrecovered_errors > 0`).
- Fidelity reports (phase 4) gain metric changes between the source run and the replay or fork.

## Tasks

### 7.1 Framework
- [x] `blackbox/metrics/`: metric registry, step signatures, versioning, storage as scores, `metrics recompute`.
- [x] Metrics computed by the worker on `run_completed`.

### 7.2 Metrics
- [x] Generic metrics.
- [x] OpsDesk metrics; `optimal_steps` and `wrong_targets` added to the task files.
- [x] PaperPilot metrics.

### 7.3 Checking
- [x] Planted-run scripts and the fake model that plays them; the expected flags for each.
- [ ] Review of 20 real OpsDesk runs, results recorded here.

### 7.4 UI
- [x] Profile overview, run metrics panel, flag filters, metric changes in fidelity reports.

## Tests
- [x] Each metric on small hand-built step lists, including edge cases: a cycle broken by a different call, a repeat
      after a state change (not a repeat), an error on the last step (unrecovered), aliased ids (same signature).
- [x] Planted runs: exactly the expected flags.
- [x] PaperPilot fixture bundles: attempts, rewrites, citation metrics as expected.
- [x] Recompute after a version change replaces the old scores.

## Done when
- [ ] Every completed run has its metrics within a second of completion.
- [ ] All planted runs are flagged correctly with no false flags on the clean run, and the real-run review is
      recorded.
- [ ] The profile overview shows OpsDesk's and PaperPilot's rates from the runs so far.

## What was built (2026-10-04)

- `metrics/framework.py` (modules with versions, `MetricInput`, step signatures, storage as `metric` scores whose
  version is `<module>-<version>`), `metrics/generic.py`, `metrics/opsdesk.py`, `metrics/paperpilot.py`,
  `metrics/hooks.py` (the worker's completion handler, metric changes in fidelity reports, flag badges), the
  `/overview` page (Plotly figures built in Python, Plotly's JavaScript served from the installed package), the
  run page's metrics panel, a flag filter on the runs list, and `blackbox metrics recompute`.
- Decisions on details the plan left open:
  - Metrics run after the checker in the same `run_completed` job, so OpsDesk metrics can read the task spec the
    checker stored with its score (BlackBox doesn't read OpsDesk's task files).
  - A metric value can carry a **flag** (`loop`, `wrong_tool`, `unrecovered_errors`, `policy_violations`, a failed
    checker, invalid or missing citations, `wasted_rewrite`, `scope_mismatch`); flags drive badges, the runs filter
    and, in phase 8, failure detection.
  - Read-only versus state-changing comes from the profile (`Profile.read_only`, OpsDesk's tool views,
    PaperPilot's searches), falling back to the HTTP method. Tools that stand in for each other when recovering
    from an error are the profile's `tool_groups` (OpsDesk: `run_action`, `restart_service`, `rollback_service`).
  - `invalid_tool_calls` counts model replies that call a tool the request didn't define, and tool calls the tool
    rejected with 400 or 422. To make the second visible, the OpsDesk agent no longer validates arguments itself:
    the environment does, and the rejected call is recorded.
  - `fallbacks` counts spans at Langfuse level `WARNING`/`ERROR` and model replies that should have been JSON
    (a schema in `format`) but weren't.
- Planted runs (`tests/integration/test_planted_metrics.py`): a clean run, a three-times restart loop, a two-step
  cycle, a change in an information-only task, a 503 then a successful retry, a 503 then giving up, duplicate reads
  and invalid arguments. Each is flagged exactly as planted, and the clean run not at all.
- **Not done here:** the review of 20 real OpsDesk runs (needs the phase 6 baseline runs) and the overview with real
  data. Label disagreements with the `metrics_correct` question on the label page once real runs exist.
