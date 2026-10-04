# Phase 10: Regression suites and CI

**Goal:** a change to an agent (code, prompt or model) is compared against a recorded baseline on a fixed suite, and
BlackBox gives a statistical verdict with hints at the cause. In CI the suite runs on replay alone, in seconds and
without a model; on this PC it runs in fork or live mode.

**Size:** M · **Depends on:** phases 6 and 7

## Design

### Suites
```yaml
name: paperpilot-core
profile: paperpilot
cases:
  from: datasets/paperpilot/questions.yaml      # or an inline list of inputs
repeats: 1
primary:                                         # signals that decide the verdict
  - ending_matches_expected                      # deterministic
  - judge: pp_faithfulness                       # used only while its current version is trusted
  - metric: citations_valid
secondary: [steps, input_tokens, duration_ms]    # reported, never decide
min_effect: 0.10
```

| Suite | Cases | Notes |
|---|---|---|
| `opsdesk-core` | all 25 tasks, `seeded`, 2 repeats | primary: `checker_pass`, `policy_violations`, `loop` |
| `paperpilot-core` | the 50 questions | primary as above |
| `paperpilot-smoke` | 10 of the 50 | small enough to commit its baseline |

### Baselines
- `blackbox baseline record <suite> --name <name>` runs the suite live (after you agree on the GPU time), scores it,
  and exports every run as a bundle to `baselines/<suite>/<name>/` with `baseline.json`: the date, the model, the agent's
  version (git commit when known), and the scores.
- `opsdesk-core` and `paperpilot-smoke` baselines are committed; `paperpilot-core` stays local by default.
- A test scans every committed bundle for anything that looks like a key or token (D15).

### Modes
| Mode | What runs | Needs the model | Used for |
|---|---|---|---|
| `replay` | each baseline run replayed `exact` against the current agent | no | CI: has the agent's behaviour changed at all? |
| `fork` | each baseline run `auto_fork`ed: tape until the first divergence, live after it | only from the divergence | code or prompt changes, cheaply |
| `live` | every case run fresh | yes, every step | model changes, or measuring variance |

A candidate is the agent as it is now (restarted with the change), optionally with `--patch <file>` (a prompt patch at
the proxy, no rebuild) or `--model <name>`.

In `replay` mode the result per case is `exact` or `diverged at step k` with the diff, and the suite fails if any case
diverged, with a message: the agent's behaviour changed at step k of case X; run `blackbox regress <suite> --mode fork`
locally, and update the baseline if the change was intended. That's snapshot testing for agents.

### Comparison statistics
- **Pass/fail signals**, paired by case: counts of fixed, broken, both pass and both fail, and McNemar's exact test (a
  binomial test on the broken and fixed counts).
- **Numeric signals:** paired differences per case, a 95% bootstrap interval for the mean difference (10,000 resamples)
  and a Wilcoxon signed-rank p-value.
- **Repeats** are averaged per case first, so cases, not repeats, are the unit of comparison.
- **Pairwise judge** (`pairwise_preference`): for cases where both answered, the judge sees both answers in both
  orders; only a preference that holds in both orders counts, otherwise it's a tie. A win rate with a Wilson interval.
  It is reported but doesn't decide until it passes the trust gate against your own pairwise labels.

### Verdict
| Verdict | Rule |
|---|---|
| `regression` | a primary pass/fail signal has more broken than fixed cases with p < 0.05, or a primary numeric signal's interval lies entirely below −`min_effect` |
| `improvement` | the mirror image |
| `no_change` | every primary signal's interval lies within ±`min_effect` |
| `inconclusive` | anything else; the report states the smallest change this suite could have detected |

### Cause hints
These feed the challenge (phase 11):
- where cases first diverged, grouped by node ("18 of 20 cases first diverged at `guardrail_validation`") and the text
  diff of the request there;
- primary and secondary signals that changed significantly;
- failure clusters that are new or grew among the candidate's failures.

### Report
Markdown and HTML, stored in `suite_runs` and written to `reports/regress/<suite>-<timestamp>.md`: suite, baseline,
candidate, mode and model; the verdict; a signal table with intervals; a per-case table (baseline and candidate
outcome, first divergence, links to both runs); cause hints; and GPU time used. `blackbox regress` exits with 1 on
`regression` (or on any divergence in `replay` mode), so CI can block on it.

### CI
- `ci.yml` gains a `regress-replay` job: `uv run blackbox regress opsdesk-core --mode replay --spawn`. `--spawn` starts
  BlackBox, the OpsDesk environment and agent in the same process tree on free ports; no Ollama is needed because every
  response comes from the tape.
- The secret scan over `baselines/` runs in the test job.

### Using it on PaperPilot
Before changing a PaperPilot prompt: restart PaperPilot with the change, then `blackbox regress paperpilot-core --mode
fork`. A short section in PaperPilot's README points to this. PaperPilot's own CI doesn't run it, because that would
need BlackBox and a model in its pipeline.

## Tasks

### 10.1 Suites and baselines
- [ ] Suite schema (Pydantic), the three suites, `baseline record`, manifest, `--update-baseline`.

### 10.2 Runner and statistics
- [ ] `blackbox regress <suite> --mode replay|fork|live [--patch] [--model] [--baseline] [--spawn]`.
- [ ] `blackbox/regress/stats.py`: McNemar exact test, paired bootstrap, Wilcoxon, Wilson interval, minimum
      detectable effect.
- [ ] Pairwise judge with both orders.
- [ ] Verdict rules and cause hints.

### 10.3 Reports and UI
- [ ] Markdown and HTML reports; **Regressions** page listing suite runs with their verdicts.

### 10.4 CI
- [ ] `regress-replay` job, `--spawn`, secret scan.
- [ ] Record and commit the `opsdesk-core` and `paperpilot-smoke` baselines (after you agree on the GPU time).

## Tests
- [ ] Statistics against known values (scipy, hand-worked small cases); minimum detectable effect is monotonic in n.
- [ ] Verdict rules on synthetic paired results: clear regression, clear improvement, no change, underpowered.
- [ ] Pairwise judge: a preference that flips with the order counts as a tie (fake judge).
- [ ] `replay` mode with an unchanged OpsDesk agent: every case `exact`, exit code 0; with one prompt line changed:
      the diverging case and step are named, exit code 1.
- [ ] The secret scan finds a planted fake key in a test bundle.

## Done when
- [ ] CI runs `opsdesk-core` in `replay` mode in under 2 minutes on every push; a branch that changes an OpsDesk prompt
      line fails CI with the diverging step named.
- [ ] Locally, `blackbox regress opsdesk-core --mode fork` gives `no_change` or `inconclusive` for a harmless wording
      change and `regression` for removing policy P1 from the prompt.
- [ ] `blackbox regress paperpilot-smoke --mode fork --patch <file>` produces a full report.
