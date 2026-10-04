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
| Rate drop | pass rates, fail rates, flag rates, ending shares | Baseline: the last 100 scored runs before the current window (fewer while history is short, but at least 30), skipping periods with an open alert. Current: the last 20 scored runs, at least 8. **Changed in the build:** alert when a Bernoulli CUSUM over the current window, tuned to a 30-point drop from the baseline, passes h = 6 nats; resolve when it is back under h/2. The plan's Beta-binomial probability (P(current worse than baseline by more than δ = 15 points), Beta(1, 1) prior) is still computed and shown with every alert. Why: see "What was built". |
| Shift | latency, tokens, steps | One-sided CUSUM on log values standardised against the baseline's median and MAD (k = 0.5, h = 5); each run's z is capped at 4, so one outlier can't alert alone |
| New failure mode | clusters | a new cluster, or one that gained 3 or more members in 30 minutes |

Detectors run after every new score. With about one run per minute and a sharp drop (a broken guardrail), the
rate-drop detector has enough evidence within 10 minutes; a moderate 30-point drop takes about 16 runs (median).

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
- [x] Job kinds and fan-out, two lanes, in-flight wait with its time limit, retries, failed-job list.

### 9.2 Sampling and detectors
- [x] Sampling configuration and hashing, hourly judge budget.
- [x] Rate-drop (Bernoulli CUSUM, Beta-binomial reported), CUSUM and new-failure-mode detectors; baseline and
      window selection.

### 9.3 Alerts and markers
- [x] Alert lifecycle, deduplication, cooldown; banner over SSE; Telegram notifier; markers table, command and API.

### 9.4 Traffic and live patches
- [x] `blackbox traffic` with its caps and stop rules.
- [x] Live patches at the proxy, header warning, expiry, commands.

### 9.5 UI
- [x] Live page, alerts page (history with details), failed-jobs list.

## Tests
- [x] Detectors on simulated streams: a 30-point drop in pass rate is caught within 10 runs; a 5-point wobble isn't
      caught in 500 runs; CUSUM catches a doubled latency. *(As built: on noise-free streams a 5-point wobble is never
      caught in 500 runs, a collapse is caught within 10 runs and a 30-point drop within 20; on seeded noisy streams
      the median delay, the share within 20 runs and the false-alarm share are asserted, because no detector meets
      both of the original targets on noisy streams.)*
- [x] Alert lifecycle: opens once, resolves after two clear evaluations, respects the cooldown.
- [x] The model lane waits while a proxy call is in flight and runs after the time limit.
- [x] Jobs survive a restart; a job that fails three times ends up `failed`.
- [x] Sampling is reproducible for a given trace id and respects the hourly budget.
- [x] A live patch changes only matching requests and stops after expiry.

## Done when
After you agree on the GPU time (about an hour in total):
- [ ] 30 minutes of PaperPilot traffic at one run per minute raise no alert.
- [ ] A live patch that makes the guardrail reject in-scope questions raises an alert within 10 minutes, naming the
      `out_of_scope` share or the pass rate, and a new cluster such as "in-scope questions refused".
- [ ] Removing the patch resolves the alert.
- [ ] If Telegram is configured, the alert arrives on your phone.

## What was built (2026-10-04)

- `live/worker.py` (lane gates, give-up handlers, configurable backoff), `live/pipeline.py` (the fan-out and the
  model-lane gate), `live/sampling.py`, `live/signals.py`, `live/detectors.py`, `live/alerts.py`, `live/telegram.py`,
  `live/patches.py`, `live/service.py`, `live/traffic.py`; the `/live`, `/alerts`, `/alerts/{id}` and `/jobs` pages;
  `POST/GET /api/markers`, `GET /api/alerts`, `POST /api/alerts/{id}/resolve`, `GET/POST/DELETE /api/live-patches`,
  `GET /api/jobs`, `POST /api/jobs/{id}/retry`; `blackbox traffic`, `blackbox live-patch add|list|remove` and
  `blackbox mark`. Migration `0004` adds `markers` and `live_patches`. `[live]` and `[alerts]` settings are in
  `blackbox.toml` and `.env`. Markers are drawn on the live page's and the overview's charts, with alert periods
  shaded on the live page.
- **The rate-drop rule changed.** Simulated against the plan's own tests, the Beta-binomial window rule caught a
  30-point drop within 10 runs in only 5–10% of streams: after 10 bad runs its 20-run window still holds 10 good ones.
  Shorter windows caught more drops but alarmed on most 5-point wobbles. No rule can meet both targets on noisy
  streams. A 0.9 → 0.6 drop gives about 3 nats of evidence in 10 runs, while staying silent for 500 runs of noise
  needs a threshold near 9 nats. The detector is now a Bernoulli CUSUM over the current window, tuned to a 30-point
  drop (`design_drop`), against the plan's lagged baseline (`threshold` h = 6). Measured on simulated streams with a
  0.9 baseline:

  | h | 30-point drop: median delay | within 10 / 20 runs | collapse to 5%: median (max) | ±5-point wobble: runs per false alarm |
  |---|---|---|---|---|
  | 5 | 13 runs | 37% / 78% | 4 (7) | ~490 |
  | **6 (default)** | 16 runs | 24% / 68% | 5 (8) | ~1300 |
  | 7 | 19 runs | 15% / 55% | 6 (9) | ~2100 |

  Each watched signal (pass rate, every trusted judge's fail rate, flags, ending shares) adds its own false alarms,
  so h stays at 6. The Beta-binomial probability is still reported with each alert. The tests assert the noise-free
  version of the plan's targets, plus measured rates on seeded noise.
- Decisions on details the plan left open:
  - One `judge` job per sampled run runs all its trusted judges in turn (the model lane runs one job at a time
    anyway), instead of one `judge:<name>` job per judge. A retry costs nothing for verdicts already in the judge
    cache. After its last failed attempt the run goes on to failure detection without judge scores.
  - Failure detection (`describe_failure`, which also assigns the cluster) runs after the judges, because trusted
    judges decide failure. `describe_failure` and `recluster` end with a `detect` request for the new-failure-mode
    rule; a queued `detect` for a profile is never doubled.
  - The model lane's time limit counts per job: a job that has waited `model_lane_max_wait_seconds` (600) runs even
    with agent calls in flight, so the lane can't starve during continuous traffic.
  - The hourly budget counts the model calls planned by judge jobs queued in the last hour (a judge with
    `samples: 3` counts 3). Flagged runs are always judged, but they count against the budget like any other run.
  - Detectors watch `live.sources` (live and traffic). Replays and suites are checked for failure but never judged
    live and never move a detector.
  - Ending shares are watched for rises of each ending. A pass-rate alert reads as "dropped", the others as "rose",
    with values shown as the bad share.
  - Detectors are stateless: each evaluation recomputes from the stored history, so a restart changes nothing. An
    alert resolves after two evaluations in a row with the statistic at or under h/2. "Hold" (between h/2 and h)
    keeps it open and resets the count. Resolving by hand is on the alert page and in the API.
  - The new-failure-mode rule counts a cluster as new only if a member arrived in the window. Otherwise the first
    clustering of old failures would alert.
  - The Telegram token sits in the request URL, so a log filter on `httpx` masks `bot<id>:<secret>`, and errors are
    logged without the URL. A test checks the token never reaches the log.
  - Live patches take only `content_regex` patches (`tape_node` and `step` are refused with a 400), never touch
    replay sessions, and expire on their own. Each patch file entry becomes one live patch that can be removed by id
    or name.
  - `blackbox traffic` counts a run as failed when the agent's request fails, not when the run scores badly, so a
    deliberate live patch doesn't stop the traffic meant to reveal it. PaperPilot traffic draws answerable questions
    three times as often as the others (drafts included: only the text matters). OpsDesk traffic picks tasks in
    `seeded` mode with a random seed each. `--seed` makes the draw repeatable.
- **Not done here (needs PaperPilot, Ollama and the GPU, about an hour, agreed with you first):** all four "Done when"
  items. Run `blackbox traffic paperpilot --rate 1/min --max-runs 30 --max-minutes 30` and check for no alert. Then
  apply a live patch that makes the guardrail reject in-scope questions (`blackbox live-patch add <file> --minutes
  30`) and check that an alert names the `out_of_scope` share or the pass rate within 10 minutes. Then remove the
  patch and check that the alert resolves, and with Telegram configured, that the alert reached your phone. Live
  judging needs trusted judges, which need your labels (phase 5); until then the pass rate comes from OpsDesk's
  checker, and PaperPilot is watched through its ending shares and flags.
