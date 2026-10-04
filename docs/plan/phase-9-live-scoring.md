# Phase 9: Live scoring and alerts

**Goal:** live runs are scored as they complete, a sample of them by judges, and BlackBox raises an alert within
minutes when quality drops, naming the signal that dropped and the failure cluster behind it. A traffic generator
supplies live runs on demand.

**Size:** M · **Depends on:** phases 7 and 8

## Design

### Durable job queue
- Work lives in the `jobs` table, so a restart loses nothing. `run_completed` fans out into `metrics`, `checker`
  (OpsDesk), `judge:<name>` (if sampled), `describe_failure` and `assign_cluster` (if failed), then `detect`.
- Two lanes:
  - the **model lane** runs one job at a time (judges, descriptions, cluster names), because the GPU serves one model
    call at a time anyway;
  - the **CPU lane** runs metrics, the checker, embeddings and detectors.
- The model lane waits while any agent run has a call in flight at the proxy, so judging never slows the agent being
  measured. After 10 minutes of waiting it runs anyway, so it can't starve.
- Failed jobs retry with backoff up to 3 times, then stay `failed` and are listed in the UI.

### Sampling
Per profile in `blackbox.toml`:

```toml
[live.sampling.paperpilot]
judge_rate = 0.3                  # share of ordinary runs that trusted judges score
always_judge_flagged = true       # runs with a metric flag or a failure ending are always judged
max_judge_calls_per_hour = 60
```

Sampling is decided by a hash of the trace id, so it can be reproduced. Metrics and the checker run on every run.

### Signals and detectors
Signals per profile: pass rate (checker, or trusted judges), each trusted judge's fail rate, metric flag rates, the
share of each ending, latency p95, tokens per run, steps per run.

| Detector | For | Rule |
|---|---|---|
| Rate drop | pass rates, fail rates, flag rates, ending shares | Baseline: the last 100 scored runs before the current window (fewer while history is short, but at least 30), skipping periods with an open alert. Current: the last 20 scored runs, at least 8. With a Beta(1, 1) prior on the current rate, alert when the probability that it is worse than the baseline by more than δ (default 15 percentage points) exceeds 0.95. |
| Shift | latency, tokens, steps | One-sided CUSUM on values standardised against the baseline (k = 0.5, h = 5) |
| New failure mode | clusters | a new cluster, or one that gained 3 or more members in 30 minutes |

Detectors run after every new score. With about one run per minute and a sharp drop, the rate-drop detector has its 8
current samples, and enough evidence, within 10 minutes.

### Alerts
- Lifecycle: `open` → shown at once as a banner on every page (SSE) and sent to Telegram if configured → `resolved`
  automatically after two clear evaluations in a row → closed. One open alert per profile and rule; after closing, the
  same rule waits 30 minutes before it can open again.
- Content: the signal, baseline and current values with intervals, the window, the failure clusters that grew in the
  window with example runs, and any **marker** in the window.
- **Markers** note changes on the timeline: `blackbox mark paperpilot "new guardrail prompt"` or
  `POST /api/markers`. They appear on charts and in alerts ("the drop started 3 runs after marker ..."). A migration
  adds the `markers` table.
- Telegram: `BLACKBOX__ALERTS__TELEGRAM_BOT_TOKEN` and `..._CHAT_ID` in `.env`, a separate bot from PaperPilot's, sent
  with a plain `httpx` POST. Never logged.

### Traffic generator
- `blackbox traffic <profile> --rate 1/min --max-runs 60 --max-minutes 60`:
  - PaperPilot: questions drawn from the question set, weighted towards the answerable ones;
  - OpsDesk: tasks in `seeded` mode with varied seeds.
- Off unless started. More than 30 runs or 30 minutes asks for confirmation (`--yes` skips it), because the PC has
  shut down during long GPU runs.
- Stops on Ctrl+C, after 3 failed runs in a row, or when free memory drops below 1.5 GB (`psutil`).
- Runs it starts have `source = traffic`.

### Live patches
- `blackbox live-patch add <patch.yaml> [--minutes 30]` applies `content_regex` patches (phase 4) to live traffic at
  the proxy, with no session. This is the deliberate break used to test alerts, and the live version of the challenge.
- While a live patch is active, every UI page says so in its header, and every patched exchange is marked `patched`.
  Patches expire after their minutes, so one can't be forgotten. `blackbox live-patch list` and `remove`.

### Live page
Per profile: runs per hour, the main pass rate with its interval and a sparkline, judge queue depth and lag, open alerts,
active live patches, and charts of each signal with markers and alert periods shaded.

## Tasks

### 9.1 Queue and worker
- [ ] Job kinds and fan-out, two lanes, in-flight wait with its time limit, retries, failed-job list.

### 9.2 Sampling and detectors
- [ ] Sampling configuration and hashing, hourly judge budget.
- [ ] Rate-drop (Beta-binomial), CUSUM and new-failure-mode detectors; baseline and window selection.

### 9.3 Alerts and markers
- [ ] Alert lifecycle, deduplication, cooldown; banner over SSE; Telegram notifier; markers table, command and API.

### 9.4 Traffic and live patches
- [ ] `blackbox traffic` with its caps and stop rules.
- [ ] Live patches at the proxy, header warning, expiry, commands.

### 9.5 UI
- [ ] Live page, alerts page (history with details), failed-jobs list.

## Tests
- [ ] Detectors on simulated streams: a 30-point drop in pass rate is caught within 10 runs; a 5-point wobble isn't
      caught in 500 runs; CUSUM catches a doubled latency.
- [ ] Alert lifecycle: opens once, resolves after two clear evaluations, respects the cooldown.
- [ ] The model lane waits while a proxy call is in flight and runs after the time limit.
- [ ] Jobs survive a restart; a job that fails three times ends up `failed`.
- [ ] Sampling is reproducible for a given trace id and respects the hourly budget.
- [ ] A live patch changes only matching requests and stops after expiry.

## Done when
After you agree on the GPU time (about an hour in total):
- [ ] 30 minutes of PaperPilot traffic at one run per minute raise no alert.
- [ ] A live patch that makes the guardrail reject in-scope questions raises an alert within 10 minutes, naming the
      `out_of_scope` share or the pass rate, and a new cluster such as "in-scope questions refused".
- [ ] Removing the patch resolves the alert.
- [ ] If Telegram is configured, the alert arrives on your phone.
