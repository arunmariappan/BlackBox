# Phase 11: The challenge, and finishing touches

**Goal:** break each agent on purpose with small prompt changes and show that BlackBox catches every harmful change,
names its cause automatically, and leaves harmless changes alone. Then finish the docs and harden the project.

**Size:** M · **Depends on:** all earlier phases

## Design

### Mutations
`challenge/mutations/*.yaml`, each a small change with what it should break. Most are prompt patches at the proxy
(phase 4), so nothing has to be rebuilt; one is a real code change, to show the path a developer would take.

| # | Agent | Change | Expected effect | Expected node |
|---|---|---|---|---|
| M1 | PaperPilot | Guardrail prompt: "Only questions about transformer internals are in scope." | in-scope questions refused (`refused_in_scope`) | `guardrail_validation` |
| M2 | PaperPilot | Grading prompt: "Answer yes only if the documents fully answer every part of the question." | more rewrites, more `max_attempts` (`retrieval_miss`) | `document_grading` |
| M3 | PaperPilot | Generation prompt: the instruction to cite papers removed | `citations_present` drops | `answer_generation` |
| M4 | PaperPilot | M1's change made for real: PaperPilot's `guardrail.txt` edited on a local, unpushed branch, PaperPilot rebuilt and restarted, then `auto_fork` | as M1 | `guardrail_validation` |
| M5 | OpsDesk | Policy P1 removed from the system prompt | `policy_violations` (P1) | `plan` |
| M6 | OpsDesk | `restart_service`'s tool description says it is "the safe way to check on a service" | `wrong_tool`, `loop` | `plan` |
| M7 | OpsDesk | "Finish as quickly as you can." added to the system prompt | `premature_finish`, `checker_pass` drops | `plan` |

Harmless controls, which must **not** be called regressions:

| # | Agent | Change |
|---|---|---|
| H1 | PaperPilot | Whitespace and line-break changes in the guardrail prompt |
| H2 | PaperPilot | One sentence of the generation prompt reworded with the same meaning |
| H3 | OpsDesk | Two policy lines swapped |

### Procedure
For each mutation and control: `blackbox regress <suite> --mode fork --patch challenge/mutations/<file>` against the
committed baseline (`paperpilot-core` or `opsdesk-core`), followed by `blackbox explain <suite-run>`. Fork mode keeps
the GPU time down: only the steps after the change run live.

### Cause report (`blackbox explain`)
1. **Deterministic evidence**, from phase 10's cause hints: the node where cases first diverged and the request diff
   there (the prompt change itself), the signals that changed significantly, and the failure clusters that are new or
   grew among the candidate's failures.
2. **An explanation** written by `qwen3.5:4b` from that evidence only, through a JSON schema: `{summary,
   changed_component, effect, evidence: [{run_id, step}]}`, at most 120 words. `changed_component` must be a node where
   divergences happened, and each piece of evidence must exist; anything else is rejected and retried once. The
   deterministic part names the cause on its own; the model adds a readable account.

### Scoring
| Measure | Target |
|---|---|
| Detection: harmful mutations called `regression` | at least 6 of 7 |
| False alarms: controls called `regression` | 0 of 3 (`no_change` or `inconclusive` are fine) |
| Cause, automatic: `changed_component` is the expected node, and the top new cluster or changed signal matches the expected effect | at least 6 of 7 |
| Cause, by you: each explanation rated 1–5 for being correct and useful | average of 4 or more |
| Cost: GPU minutes per mutation, fork mode compared with live mode | reported |

**Live variant:** M1 applied as a live patch during PaperPilot traffic (phase 9). Measured: time from the patch to the
alert, and whether the alert names the guardrail and the refusal cluster.

### Results
`reports/challenge-<date>.md`: the table of mutations with verdicts, intervals, named causes, ratings and GPU time; the
live variant's timing; what failed and why. The README gets a short results section linking to it.

### Finishing touches
- **README** for developers: what BlackBox does and why, the architecture diagram, a quickstart (`uv sync`,
  `blackbox serve`, PaperPilot's switch, starting OpsDesk), the main commands, and screenshots of the main pages.
- **`docs/profiles.md`:** how to add a profile for a new agent in any language: what the agent must do (propagate
  `traceparent`, export OTLP, send its model and tool calls through the proxy) and what the profile must define. It's
  written with the next agents in mind (Mayday in .NET, AirMarshal and Jetway in Rust).
- **`CLAUDE.md`** complete: commands, layout, gotchas collected along the way.
- **Fresh clone check:** clone into an empty folder, `uv sync`, `uv run pytest`, and
  `uv run blackbox regress opsdesk-core --mode replay --spawn` all succeed.
- **Retention:** a weekly prune job (keeps labelled runs and baselines), and the database size shown on the live page.
- **Performance check:** with 2,000 runs stored, the runs page renders in under 300 ms and proxy overhead is still under
  5 ms at the median.

## Tasks
- [ ] The ten mutation and control files, and M4's branch in PaperPilot (local only, deleted afterwards).
- [ ] `blackbox explain` with its validation.
- [ ] Run every mutation and control (after you agree on the GPU time), score them, and write the report.
- [ ] The live variant.
- [ ] README, `docs/profiles.md`, `CLAUDE.md`, fresh clone check, retention, performance check.

## Tests
- [ ] `explain` rejects a `changed_component` that isn't a divergence node, and evidence that doesn't exist (fake
      model).
- [ ] Each mutation file applies cleanly to the current prompts (the patch's `find` text exists), so a prompt edit in
      PaperPilot or OpsDesk can't silently make a mutation do nothing.

## Done when
- [ ] The challenge report meets the targets above, or explains each miss.
- [ ] Every item of the definition of done in [README.md](README.md#9-definition-of-done-whole-project) is ticked.
