# Phase 8: Failure clusters and likely causes

**Goal:** failed runs are grouped automatically with similar ones, and each group gets a title and a likely cause backed
by evidence from its runs. New failures join an existing group as they arrive, or start a new one.

**Size:** M · **Depends on:** phases 5 and 7

## Design

### Which runs count as failed
A run has failed if any of these hold: the OpsDesk checker failed; a **trusted** judge gave `fail`; the ending is one the
profile marks as a failure (`max_attempts`, `search_unavailable`, `max_steps`, `error`); or a metric flag is set
(`loop`, `wrong_tool`, `policy_violations > 0`, `unrecovered_errors > 0`). Untrusted judges never decide failure.

### Failure signature
For each failed run, BlackBox builds:

1. **Facts** (deterministic): ending, failed checks and judges, metric flags, the node and step where the first problem
   shows (first error, first flagged step, the step a judge's unsupported claim came from), tool error statuses, and for
   replays the first divergence.
2. **A description**: `qwen3.5:4b` reads a compact digest of the run (instruction or question, steps as one line each,
   the facts) and returns, through a JSON schema, `{what_went_wrong, where_step, category}`. The category comes from a
   fixed list: `wrong_tool`, `bad_arguments`, `loop`, `gave_up_after_error`, `policy_violation`, `unsupported_claim`,
   `premature_finish`, `refused_in_scope`, `retrieval_miss`, `format_error`, `infrastructure`, `other`. A description
   whose `where_step` doesn't exist in the run is retried once, then stored without it.

### Clustering
- The description is embedded with `fastembed` (`BAAI/bge-small-en-v1.5`, on the CPU, in a thread pool), and the facts
  become one-hot features (ending, category, flags, node of the first problem). The vector is the normalised embedding
  next to the weighted one-hot part; the weight is tuned on the planted failures below.
- **HDBSCAN** (`sklearn.cluster.HDBSCAN`, `min_cluster_size = 3`) groups the vectors. It doesn't need the number of
  groups in advance and leaves odd failures unclustered rather than forcing them into a group.
- Clustering is per profile. Failures from different agents never share a cluster.

### Names and likely causes
For each cluster, `qwen3.5:4b` sees the 3–5 members closest to its centre, plus statistics over all members (for example
"90% end `out_of_scope`; guardrail scores 40–55; all since 14:20"). It returns, through a JSON schema, `{title,
likely_cause, evidence: [{run_id, step, observation}], suggested_fix}`. Every piece of evidence must name a run and step
that exist in the cluster; invalid evidence is dropped, and a cause with no valid evidence is marked `unsupported`.

### Live assignment and stable ids
- A new failure joins the nearest cluster if its distance to the centre is within that cluster's radius (the 95th
  percentile of its members' distances); otherwise it stays unclustered.
- A full re-clustering runs on demand (`blackbox cluster --profile P`), nightly, and when 10 failures are waiting
  unclustered.
- After re-clustering, a new cluster takes the id of the old cluster it shares the most members with (Jaccard ≥ 0.5),
  so a cluster keeps its id, history and name as it grows. Its name and cause are regenerated only when its members
  change by more than 30%.

### Checking it on known failures
OpsDesk can fail in known ways on purpose: scripted runs (as in phase 7) and real runs under fault settings (a misleading
runbook, a tool that keeps failing, a missing policy line in the prompt). Each has a known failure type, so clustering
quality can be measured: adjusted Rand index against the known types, and whether each cluster's cause names its type.
The weights and `min_cluster_size` are tuned here, and the result is recorded in this file.

### UI
- **Clusters** page per profile: title, size, a sparkline of new members per day, likely cause, suggested fix, status
  (`new`, `known`, `fixed`, which you set), and the unclustered failures.
- Cluster page: cause with evidence links straight to run steps, member list, member statistics.
- Run page: the run's failure description and cluster.

## Tasks

### 8.1 Signatures
- [ ] Failure rules per profile; facts; run digests; descriptions with the fixed categories and validation.

### 8.2 Clustering
- [ ] Embeddings (`fastembed`, model downloaded once and cached), feature vectors, HDBSCAN, centres and radii.
- [ ] Cluster naming with evidence validation.
- [ ] Live assignment on `run_completed`, re-clustering triggers, id matching across re-clusterings.
- [ ] `blackbox cluster` command.

### 8.3 Checking
- [ ] OpsDesk fault settings and a set of about 40 known failures (scripted plus real, the real ones after you agree on
      the GPU time); adjusted Rand index and cause check; tuning; results recorded here.

### 8.4 UI
- [ ] Clusters page, cluster page, run page additions.

## Tests
- [ ] Synthetic vectors with three clear groups and noise: three clusters, noise left unclustered.
- [ ] Id matching: adding members keeps ids; splitting a cluster keeps the id on the larger part.
- [ ] Evidence naming a run outside the cluster, or a step that doesn't exist, is dropped.
- [ ] Assignment within the radius joins; outside stays unclustered; 10 unclustered trigger a re-clustering.
- [ ] Untrusted judges' `fail` doesn't make a run count as failed.

## Done when
- [ ] Known OpsDesk failures cluster with an adjusted Rand index of at least 0.6 against their types, and at least 3 of
      4 clusters have a cause that names the right type.
- [ ] PaperPilot's failures from phase 5 (unsupported claims, wrong scope decisions, maximum attempts) appear as
      named clusters with evidence.
- [ ] A new failure joins its cluster within a minute of the run completing.
