# Phase 1: Store

**Goal:** the data model and a SQLite store that the receiver, proxy and worker can all write to safely; content-
addressed blobs for bodies; and run bundles, the export format used for baselines and test fixtures.

**Size:** S · **Depends on:** phase 0

## Design

- **One database file**, `data/blackbox.db`, in WAL mode with `synchronous=NORMAL` and `foreign_keys=ON`.
- **One writer.** A `StoreWriter` asyncio task owns the only write connection. Other components call
  `await writer.submit(op)`, which queues the operation and waits for its result. The writer runs queued operations in
  small transactions (up to 50 operations or 50 ms per batch). Reads use a separate pool of read-only connections; WAL
  lets them run during writes.
- **Blobs.** Every request and response body, prompt and large payload is stored once in `blobs`, keyed by the
  SHA-256 of its uncompressed bytes and compressed with zstd (level 3, `compression.zstd`). Rows point at blobs by
  hash. A system prompt sent 1,000 times is stored once.
- **Migrations.** Alembic, with `render_as_batch=True` (SQLite can't alter most columns in place).
  `blackbox serve` upgrades the database on start; `blackbox db upgrade` does it explicitly.

## Tables

| Table | Key columns | Notes |
|---|---|---|
| `runs` | `id` (ULID), `trace_id` (unique), `profile`, `status` (`open`, `complete`, `failed_assembly`), `source` (`live`, `traffic`, `suite`, `replay`), `started_ms`, `ended_ms`, `entry_request_blob`, `output_blob`, `ending`, `model`, `replayable`, `session_id`, `replay_of`, `tags` (JSON) | One row per run. `entry_request_blob` is the exact request that started the run, when BlackBox started it. |
| `spans` | `(trace_id, span_id)`, `parent_span_id`, `name`, `kind`, `service`, `start_ns`, `end_ns`, `status_code`, `status_message`, `attributes` (JSON), `events` (JSON), `resource` (JSON) | Raw spans as received. Large attribute values (over 4 KB) move to blobs. |
| `exchanges` | `id`, `trace_id`, `parent_span_id`, `session_id`, `upstream`, `seq`, `method`, `path`, `query`, `request_headers` (JSON, redacted), `request_blob`, `request_key`, `status`, `response_headers` (JSON), `response_blob`, `stream`, `chunk_times` (JSON), `started_ms`, `first_byte_ms`, `ended_ms`, `error`, `served_from`, `divergence` (JSON) | One row per recorded HTTP call. `served_from` is `live`, `tape:<exchange id>` or `patched`. `chunk_times` lists `[byte_offset, ms_since_start]` for streamed responses. |
| `steps` | `(run_id, idx)`, `kind` (`llm`, `tool`, `embedding`, `other`), `node`, `exchange_id`, `span_id`, `model`, `tool_name`, `input_tokens`, `output_tokens`, `latency_ms`, `status`, `view` (JSON) | Built when a run completes. `view` is the parsed form shown in the UI: messages, tool calls, search hits. |
| `sessions` | `id`, `trace_id`, `source_run_id`, `mode` (`exact`, `fork`, `auto_fork`), `fork_step`, `overrides` (JSON: model, patches, speed), `status`, `created_ms`, `result` (JSON fidelity report) | Replays and forks. |
| `recorded_values` | `(trace_id, seq)`, `kind` (`now`, `uuid`, `random`), `value` | Values an SDK agent drew from its clock or random generator (phase 6). |
| `scores` | `id`, `run_id`, `kind` (`metric`, `judge`, `checker`), `name`, `version`, `value` (float), `label` (text), `rationale`, `details` (JSON), `created_ms` | Every number or verdict about a run. |
| `judge_calls` | `id`, `judge_version`, `input_hash`, `prompt_blob`, `response_blob`, `parsed` (JSON), `valid`, `latency_ms`, `created_ms` | Cache and audit trail of judge calls; the same input is never judged twice by the same version. |
| `judges` | `(name, version)`, `prompt_hash`, `model`, `options` (JSON), `created_ms`, `trusted`, `agreement` (JSON) | |
| `labels` | `id`, `run_id`, `question`, `value`, `note`, `labeler`, `created_ms` | Your verdicts. |
| `failures` | `run_id`, `signature` (JSON), `description`, `category`, `embedding` (float32 bytes), `cluster_id` | |
| `clusters` | `id`, `profile`, `title`, `likely_cause`, `evidence` (JSON), `suggested_fix`, `centroid`, `radius`, `size`, `status`, `created_ms`, `updated_ms` | |
| `alerts` | `id`, `profile`, `rule`, `severity`, `opened_ms`, `closed_ms`, `details` (JSON), `notified` | |
| `jobs` | `id`, `kind`, `run_id`, `payload` (JSON), `status`, `attempts`, `available_ms`, `last_error` | Durable work queue (phase 9 uses it fully; phase 2 already queues run completion). |
| `suite_runs`, `suite_cases` | suite name, baseline, candidate label, mode, verdict, report blob; per case: input, baseline run, candidate run, outcome | Regression results (phase 10). |
| `blobs` | `sha256`, `size`, `content_type`, `data` | zstd-compressed bytes. |

Indexes: `runs(profile, started_ms)`, `runs(status)`, `spans(trace_id)`, `exchanges(trace_id, seq)`,
`exchanges(session_id)`, `scores(run_id)`, `scores(name, version)`, `jobs(status, available_ms)`.

Columns that later phases fill (failures, clusters, alerts, suites) are created now, so their migrations don't have to
rewrite earlier tables.

## Run bundles

A run bundle is a run exported to plain files, so it can be committed (baselines in `baselines/`, fixtures in
`tests/fixtures/`), diffed and imported into another database:

```
<bundle>/
├── run.json        # the runs row, its spans, exchanges, steps, recorded values and scores
└── blobs/
    └── <sha256>.zst
```

`blackbox runs export <run> --out <dir>` and `blackbox runs import <dir>` move bundles in and out. Importing is
idempotent: same trace id, same blobs, no duplicates. A suite baseline is a folder of bundles plus a `baseline.json`
manifest.

## Tasks

### 1.1 Models and migrations
- [ ] SQLAlchemy 2 typed models (`Mapped[...]`) for every table above, in `blackbox/store/models.py`.
- [ ] Alembic setup in `blackbox/store/migrations`, first migration creating everything, `blackbox db upgrade`.

### 1.2 Writer and readers
- [ ] `StoreWriter`: queue, batching, one transaction per batch, errors returned to the caller that submitted the
      failing operation without affecting the others in the batch (retry the rest one by one).
- [ ] `StoreReader`: async read sessions (aiosqlite), helper queries for runs, steps and exchanges.
- [ ] Startup pragmas: WAL, `busy_timeout=5000`, `foreign_keys=ON`.

### 1.3 Blobs
- [ ] `BlobStore.put(bytes, content_type) -> sha256` (insert if missing) and `get(sha256) -> bytes`.
- [ ] `blackbox db prune --older-than 30d`: deletes runs older than the cutoff (except those in a baseline or with
      labels) and then any blob nothing references.

### 1.4 Bundles
- [ ] `export_run(run_id, dir)` and `import_bundle(dir)`, plus the CLI commands. JSON is written with sorted keys and
      two-space indent, so committed bundles diff cleanly.

## Tests
- [ ] 1,000 writes submitted at once from 20 concurrent tasks all land, with no "database is locked" error.
- [ ] A failing operation in a batch fails only its own caller.
- [ ] The same blob put twice is stored once; get returns identical bytes.
- [ ] Export, then import into an empty database, gives identical rows (round-trip test).
- [ ] `prune` keeps labelled and baseline runs and removes unreferenced blobs.

## Done when
- [ ] `blackbox db upgrade` creates `data/blackbox.db`, and the tests above pass in CI.
