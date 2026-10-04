# Phase 5: Judges and labels

**Goal:** LLM judges score runs, you label runs blind, and BlackBox measures how often each judge agrees with you.
Only judges that pass an agreement gate are trusted. At the end of this phase the MVP is complete: PaperPilot runs are
recorded, one replays exactly, and runs are scored by a judge.

**Size:** M · **Depends on:** phase 4

## Design

### PaperPilot question set
`datasets/paperpilot/questions.yaml`, 50 questions, each with an id, the question, the expected ending and whether the
answer must cite papers:

| Group | Count | Expected ending | How it's made |
|---|---|---|---|
| Answerable from the corpus | 30 | `answered`, with citations | `blackbox dataset draft paperpilot` samples 30 papers from PaperPilot's OpenSearch index and has `qwen3.5:4b` draft one question per paper; you review and edit them |
| Out of scope | 10 | `out_of_scope` | written by hand (cooking, arithmetic, sport, ...) |
| In scope but not in the corpus | 10 | `max_attempts` | AI topics no indexed paper covers, checked against the index |

Each answerable question is also run with `top_k: 1`. With less context the model makes more unsupported claims, which
gives judges and labels some failures to find. That makes 80 runs, about an hour of GPU time, done in batches after
you agree.

### Judges
A judge is a prompt file plus an input builder and an output schema:

```markdown
---
name: pp_faithfulness
profile: paperpilot
applies_when: "ending == 'answered'"
output: FaithfulnessVerdict          # Pydantic model; its JSON schema goes to Ollama's `format`
---
You check whether an answer is supported by the retrieved paper excerpts.
... rubric, definitions, two short worked examples ...
Question: {question}
Excerpts:
{context}
Answer:
{answer}
```

- **Output fields put reasoning first** (`unsupported_claims`, `rationale`, then `verdict`), because the model writes
  fields in schema order and should reason before it decides.
- **Input builders** read the run, not the agent's own logs. `pp_faithfulness` takes the excerpts from the last
  OpenSearch step's recorded hits, so the judge sees exactly what the agent saw.
- **Version** = the first 12 hex characters of a SHA-256 over the prompt text, model, options, output schema and input
  builder version. Any change makes a new version; scores always name their version.
- **Calls** go straight to Ollama (not through the proxy; they aren't agent runs) with `qwen3.5:4b`, temperature 0,
  `think` off and the schema in `format`. Every call is stored in `judge_calls`; the same input hash and version is
  never judged twice. An invalid reply is retried once, then stored as `invalid`.

PaperPilot's judges:

| Judge | Applies to | Verdict |
|---|---|---|
| `pp_faithfulness` (the MVP judge) | `answered` runs | pass if every factual claim in the answer is supported by the excerpts; lists unsupported claims |
| `pp_relevance` | `answered` runs | pass if the answer addresses the question that was asked |
| `pp_scope` | every run | whether an AI-research assistant should answer the question; compared with the run's ending, it gives "scope decision correct" |

Citation checks (an `[arXiv:id]` present, every cited id among the sources) are deterministic, so they are metrics in
phase 7, not judges.

### Blind labelling
- Label questions mirror the judges: `faithful`, `relevant` (pass, fail or unsure) and `scope_correct`.
- The **Label** page shows one run at a time: the question, the numbered excerpts (collapsed), the answer. Keys:
  `p` pass, `f` fail, `u` unsure, `n` note, `j`/`k` next and previous.
- The judge's verdict stays hidden until your label is saved, then appears, with disagreements highlighted. Seeing it
  first would pull your label towards it.
- Queue order: runs where judge samples disagreed, then a spread across endings, then the rest at random.
- Labelled runs are exported as bundles with their labels to `datasets/paperpilot/labelled/`, so the labels survive a
  database reset and calibration can be reproduced.

### Agreement and trust
- For each judge version and label question, over runs that have both (unsure labels left out and counted): n, percent
  agreement, Cohen's κ (quadratic-weighted κ for ordinal scales) with a 95% bootstrap interval (2,000 resamples), the
  confusion matrix, and the judge's precision and recall on `fail`, the class that matters.
- **Held-out labels:** 40% of labels (chosen by a hash of the run id) are kept out while you tune a judge's prompt.
  The trust gate uses only those, so tuning can't overfit the judge to the labels it is scored on.
- **Trust gate:** at least 20 held-out labels, κ ≥ 0.6 and the interval's lower bound ≥ 0.3. With 20 labels the
  interval is wide, so passing needs a κ of roughly 0.75 or more. Trusted versions are marked in `judges.trusted`.
  Untrusted judges' scores are shown greyed out and never drive alerts or regression verdicts. OpsDesk's checker
  (phase 6) supplies hundreds of labels for free, so its judge gets a much tighter interval.
- `blackbox judge calibrate <judge>` runs the current version on every labelled run (cached calls make unchanged
  versions free) and prints a table of versions with their agreement, so a prompt change can be judged by its κ.

### Stability and sensitivity
- `blackbox judge stability <judge> --runs 20 --repeats 3` repeats the same inputs at temperature 0 and at 0.7 and
  reports how often the verdict flips. Above 5% at temperature 0, that judge switches to a majority vote of three
  samples at temperature 0.3, which is recorded in its options and therefore its version.
- **Sensitivity check:** 10 answers that passed get one made-up claim appended; `pp_faithfulness` must fail at least 8
  of them. A judge that passes everything is caught here.

### Replay integration
Fidelity reports (phase 4) gain judge results for both the source run and the replay or fork, so a fork shows whether
the answer got better or worse, not only whether it changed.

## Tasks

### 5.1 Dataset
- [ ] `blackbox dataset draft paperpilot`: sample papers from OpenSearch, draft questions, write YAML for review.
- [ ] The 50-question file, reviewed by you and committed.
- [ ] `blackbox run paperpilot --dataset datasets/paperpilot/questions.yaml [--top-k 1] [--limit N]` runs them in
      sequence (batches, after you agree).

### 5.2 Judge framework
- [ ] Prompt files with front matter, input builders, Pydantic output schemas, versioning.
- [ ] `blackbox/llm/`: a small Ollama client for BlackBox's own calls: JSON-schema output, validation, one retry,
      timing, `invalid` results.
- [ ] Judge runner with the `judge_calls` cache; `blackbox judge run <judge> [--runs ... | --profile P --last N]`.
- [ ] The three PaperPilot judges.

### 5.3 Labels and agreement
- [ ] Label page and its keyboard shortcuts, queue order, blind reveal, notes.
- [ ] `blackbox/judges/agreement.py`: κ (plain and weighted), bootstrap intervals, confusion matrix, precision and
      recall, held-out split, trust gate.
- [ ] Judges page: versions, agreement with intervals, confusion matrix, trusted badge, disagreement list linking to
      runs.
- [ ] `blackbox judge calibrate`, `blackbox judge stability`, `blackbox labels export`.

### 5.4 UI
- [ ] Runs list: a column per trusted judge. Run page: judge verdicts with rationale and unsupported claims.

## Tests
- [ ] κ, weighted κ and the bootstrap interval match hand-computed values on small tables (and scikit-learn's κ).
- [ ] The held-out split is stable across calls and never changes when labels are added.
- [ ] Changing one character of a prompt changes the version; an unchanged version hits the cache with no model call.
- [ ] An invalid model reply is retried once and then stored as `invalid` (fake Ollama).
- [ ] Input builder: `pp_faithfulness` input contains exactly the excerpts from the run's last search step.
- [ ] Label page: saving a label reveals the verdict; the verdict isn't in the page before that.

## Done when
- [ ] The 50 questions are committed and their 80 runs are recorded.
- [ ] You have labelled at least 50 answered runs for `faithful` (so at least 20 are held out) and every run for
      `scope_correct`.
- [ ] `pp_faithfulness` has its agreement shown with an interval, and its sensitivity check passes. Whether or not it
      passes the trust gate is recorded; if it doesn't after two prompt revisions, the reasons go into this file and
      the work continues.
- [ ] **MVP complete:** PaperPilot agentic runs are recorded, one replays exactly, and runs are scored by
      `pp_faithfulness`, all visible in the UI.
