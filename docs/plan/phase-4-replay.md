# Phase 4: Exact replay and forks

**Goal:** any recorded run can be replayed exactly, with every model and tool response served from its tape, and can
be re-run from step N with another model or a patched prompt. When an agent's code or prompt changes, auto-fork serves
the tape until the agent's requests first differ, then goes live. This is the second half of the MVP.

**Size:** L · **Depends on:** phase 3

## Design

### Sessions
A session is one replay or fork. `POST /api/sessions` (and `blackbox replay`) creates it:

```json
{
  "source_run_id": "01JD...",
  "mode": "exact | fork | auto_fork",
  "fork_step": 4,
  "model": "qwen3.5:9b",
  "patches": [],
  "speed": 0,
  "lenient": false
}
```

The proxy keeps active sessions in a table keyed by the session's new trace id; a session expires after 30 minutes or
when its run completes. A call carrying an expired session's trace id gets 410, never a silent live call.

### Replay runner
`blackbox replay <run> [--from-step N | --auto-fork] [--model M] [--patch FILE] [--speed X] [--lenient] [--times K]`:

1. Load the source run. It must be `replayable`, and have an entry request, or a profile that can rebuild one from the
   root span (PaperPilot: `langfuse.trace.input` plus the trace metadata `top_k`, `use_hybrid`, `model`). A rebuilt
   input is marked as such in the report.
2. Create the session with a fresh trace id, and let the profile prepare (nothing for PaperPilot; a sandbox for
   OpsDesk in phase 6).
3. Start the run through the profile with `traceparent: 00-<new trace id>-<random span id>-01`.
4. Wait for the new run to complete, then write the fidelity report to the session and print it with the run's URL.

`--times K` repeats the replay K times; with `--from-step 1` it measures how much the live agent and model vary on
the same input.

### Matching
For each recorded-path request in a session, on upstream `u`:

1. Compute its **match key**: a SHA-256 of the canonical request after the profile's **normalisers** for `u`.
   Normalisers only affect matching, never what is sent or served:
   - `MaskRegex(pattern, replacement)`, for example an ISO timestamp → `<ts>`;
   - `DropJsonPath(path)`, for example a per-request id field.
   PaperPilot gets whichever normalisers spike S1 showed it needs (perhaps none). The default is none: exact means
   exact unless a profile says otherwise.
2. **Candidates** are the tape's unused exchanges on `u` with the same match key. The lowest step wins and is marked
   used. Matching by key rather than strictly by position tolerates concurrent calls that arrive in a different order.
3. **No candidate** is a divergence. It is stored with the next unused tape step on `u` and a diff: a text diff per
   chat message for model calls, a JSON diff otherwise. What happens next depends on the mode.

| Mode | Matched request | First divergence | After going live |
|---|---|---|---|
| `exact` | served from the tape | 409 with `{"error": "blackbox_divergence", "step": n}`; with `--lenient`, the next unused tape exchange on the same path is served anyway and flagged | (never goes live) |
| `fork` (from step N) | served from the tape while the matched step is before N; the request that would use step N or later goes live | goes live | every later request is live, with overrides |
| `auto_fork` | served from the tape | goes live | every later request is live, with overrides |

Once a session goes live it stays live; the tape's later steps can no longer be trusted to fit.

### Serving from the tape
- The recorded status, headers (without hop-by-hop headers; content length recomputed) and body are sent back.
- Streamed responses are re-sent chunk by chunk. `speed = 0` (the default) sends them at once; `speed = 1` keeps the
  recorded timing, which is useful in demos.
- The replay run records its own exchanges too, with `served_from = tape:<exchange id>`, so a replay is a complete run
  that can itself be replayed, scored and compared.

### Overrides for live steps
- **Model:** the `model` field of model-call bodies (Ollama and OpenAI shapes) is replaced on live steps only. The
  agent's original body and the body actually sent are both kept (new column `exchanges.sent_request_blob`, filled only
  when they differ).
- **Patches** edit a request before it is matched. A patched request therefore no longer matches the tape and goes live,
  which is exactly what a prompt change does. A patch file:

```yaml
patches:
  - name: strict-guardrail
    match:
      upstream: ollama
      tape_node: guardrail_validation       # replay only: the node of the tape step this request lines up with
      # step: 1                             # replay only: a step number
      # content_regex: "relevance of the question"   # any mode, also live traffic (phase 9)
    edit:
      path: "$.messages[?(@.role == 'system')].content"
      replace: { find: "Score from 0 to 100", with: "Be very strict. Score from 0 to 100" }
      # set: "a whole new text"
```

`tape_node` and `step` work because, during a replay, BlackBox can line a request up with its tape step (by match key)
before applying the patch, and the tape step's node is already known. In a live run the node isn't known when the
request is made (its span hasn't ended or been exported yet), so live patches match on content.

`exact` with patches is refused; the CLI suggests `--auto-fork`.

### Fidelity report
- Per step: number, node, how it was served (`tape`, `live`, `patched`), and for live steps whether the response
  differs from the tape's response at the same step.
- The first divergence, with its diff.
- The output comparison from the profile's `compare_outputs`. PaperPilot compares `answer`, `sources`,
  `reasoning_steps`, `retrieval_attempts`, `chunks_used` and `search_mode`, and ignores `trace_id`.
- **Exact replay passes** when every step was served from the tape, nothing diverged, and the outputs are equal.
- Phase 5 adds judge score changes, and phase 7 metric changes.

### UI
- Run page: **Replay exactly**, and **Fork from here** on each step. The fork form offers a model (from Ollama's
  `/api/tags`) and an editor pre-filled with that step's messages. Editing a message turns into a patch for that step.
- Session page: progress over SSE while it runs, then the fidelity report, the source and replay steps side by side,
  and an output diff.
- Runs list: replays are linked to their source run.

## Tasks

### 4.1 Sessions and matching
- [ ] `blackbox/proxy/sessions.py`: session table, expiry, mode state (`on_tape`, `live`), used-step tracking.
- [ ] `blackbox/proxy/matching.py`: canonical requests, normalisers, match keys (tape keys cached per session),
      candidate choice, divergence diffs.
- [ ] Tape serving with chunk timing and `speed`.
- [ ] Migration: `exchanges.sent_request_blob`.

### 4.2 Overrides
- [ ] Model override for live model calls.
- [ ] Patches: YAML schema (Pydantic), matching by `tape_node`, `step` or `content_regex`, JSONPath edit with `replace`
      or `set`, patched exchanges marked.

### 4.3 Runner and report
- [ ] `blackbox/replay/runner.py` and `blackbox replay` with every option above.
- [ ] Profile hooks: `rebuild_input(run)`, `prepare(session)`, `compare_outputs(a, b)`, `normalisers` per upstream;
      PaperPilot's versions.
- [ ] Fidelity report in the session row, in the CLI (Rich table) and on the session page.

### 4.4 UI
- [ ] Replay and fork actions, fork form with model choice and message editor, session page, links between runs.

## Tests
All without a GPU: a fake agent in Python makes model calls through the proxy to a fake model upstream that counts its
calls.
- [ ] Record, then replay exactly: identical output, zero calls reach the fake upstream.
- [ ] Change the fake agent's prompt at its third call: `exact` reports a divergence at step 3 with the right message
      diff; `auto_fork` serves steps 1–2 from the tape and sends 3 onwards live (the upstream sees exactly those).
- [ ] `fork` from step 2 with a model override: the upstream sees the new model only from step 2.
- [ ] Patches matched by `tape_node`, by `step` and by `content_regex` each change only the intended request.
- [ ] A timestamp in the prompt: diverges without the normaliser, matches with it.
- [ ] Streamed tape at `speed = 1`: total time within 20% of the recording; at `speed = 0`: under 100 ms.
- [ ] Two sessions replaying at once don't mix up their tapes.
- [ ] A call with an expired session's trace id gets 410.
- [ ] PaperPilot bundle (from phase 3): a test client re-sends the bundle's recorded requests in order with a session's
      trace id, and every one is served from the tape. This checks matching against real PaperPilot bodies without
      running .NET.

## Done when
- [ ] Five recorded PaperPilot runs (answered on the first retrieval, answered after a rewrite, out of scope, stopped
      after the maximum retrieval attempts, and one asked from PaperPilot's own UI, with a rebuilt input) each replay
      exactly: all steps from the tape, identical output, and no call reaches Ollama, OpenSearch or Jina.
- [ ] A fork of an answered run at `answer_generation` with `--model qwen3.5:9b` runs only that step live and shows
      the answer diff.
- [ ] A fork with a patched guardrail prompt shows the new score, and the new ending when it crosses the threshold.
- [ ] **MVP, part 2:** "replay one exactly" is shown in the UI with its fidelity report.
