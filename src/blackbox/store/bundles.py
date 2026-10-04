"""Run bundles: a run exported to plain files, so it can be committed, diffed and imported elsewhere.

```
<bundle>/
├── run.json        # the runs row, its spans, exchanges, steps, recorded values, scores and labels
└── blobs/
    └── <sha256>.zst
```

JSON is written with sorted keys and two-space indent. Importing is idempotent: rows that already exist are skipped.
"""

import json
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.store import Store
from blackbox.store.blobs import decompress
from blackbox.store.models import Base, Blob, Exchange, Label, RecordedValue, Run, Score, Session, Span, Step
from blackbox.util import pretty_json, sha256_hex

BUNDLE_FORMAT = 1
_TABLES: list[tuple[str, type[Base]]] = [
    ("spans", Span),
    ("exchanges", Exchange),
    ("steps", Step),
    ("recorded_values", RecordedValue),
    ("scores", Score),
    ("labels", Label),
    ("sessions", Session),
]


class BundleError(Exception):
    pass


def row_to_dict(obj: Base) -> dict[str, Any]:
    return {attr.key: getattr(obj, attr.key) for attr in inspect(obj).mapper.column_attrs}


def _blob_refs(data: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    run = data["run"]
    refs.update(v for v in (run.get("entry_request_blob"), run.get("output_blob")) if v)
    for span in data["spans"]:
        refs.update(span.get("blob_refs") or [])
    for exchange in data["exchanges"]:
        refs.update(v for k, v in exchange.items() if k.endswith("_blob") and isinstance(v, str) and v)
    return refs


async def run_bundle_data(store: Store, run_id: str) -> dict[str, Any]:
    async with store.read() as s:
        run = await s.get(Run, run_id)
        if run is None:
            raise BundleError(f"no run {run_id}")
        data: dict[str, Any] = {"format": BUNDLE_FORMAT, "run": row_to_dict(run)}
        data["spans"] = [
            row_to_dict(r)
            for r in (
                await s.execute(select(Span).where(Span.trace_id == run.trace_id).order_by(Span.start_ns, Span.span_id))
            ).scalars()
        ]
        data["exchanges"] = [
            row_to_dict(r)
            for r in (
                await s.execute(
                    select(Exchange).where(Exchange.trace_id == run.trace_id).order_by(Exchange.seq, Exchange.id)
                )
            ).scalars()
        ]
        data["steps"] = [
            row_to_dict(r)
            for r in (await s.execute(select(Step).where(Step.run_id == run_id).order_by(Step.idx))).scalars()
        ]
        data["recorded_values"] = [
            row_to_dict(r)
            for r in (
                await s.execute(
                    select(RecordedValue).where(RecordedValue.trace_id == run.trace_id).order_by(RecordedValue.seq)
                )
            ).scalars()
        ]
        data["scores"] = [
            row_to_dict(r)
            for r in (
                await s.execute(select(Score).where(Score.run_id == run_id).order_by(Score.kind, Score.name, Score.id))
            ).scalars()
        ]
        data["labels"] = [
            row_to_dict(r)
            for r in (await s.execute(select(Label).where(Label.run_id == run_id).order_by(Label.id))).scalars()
        ]
        data["sessions"] = []
        if run.session_id:
            session = await s.get(Session, run.session_id)
            if session is not None:
                data["sessions"].append(row_to_dict(session))
        refs = sorted(_blob_refs(data))
        content_types: dict[str, str | None] = {}
        for sha in refs:
            content_types[sha] = (
                await s.execute(select(Blob.content_type).where(Blob.sha256 == sha))
            ).scalar_one_or_none()
        data["blobs"] = content_types
    return data


async def export_run(store: Store, run_id: str, out_dir: Path) -> Path:
    """Write the run's bundle to `out_dir` (created if needed) and return the path of its `run.json`."""
    data = await run_bundle_data(store, run_id)
    blob_dir = out_dir / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(data["blobs"])
    for stale in blob_dir.glob("*.zst"):
        if stale.stem not in wanted:
            stale.unlink()
    for sha in sorted(wanted):
        target = blob_dir / f"{sha}.zst"
        if not target.exists():
            compressed, _ = await store.blobs.get_compressed(sha)
            target.write_bytes(compressed)
    run_json = out_dir / "run.json"
    run_json.write_text(pretty_json(data), encoding="utf-8", newline="\n")
    return run_json


def read_bundle(bundle_dir: Path) -> dict[str, Any]:
    run_json = bundle_dir / "run.json"
    if not run_json.exists():
        raise BundleError(f"{bundle_dir} has no run.json")
    data: dict[str, Any] = json.loads(run_json.read_text(encoding="utf-8"))
    if data.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"{run_json}: unsupported bundle format {data.get('format')!r}")
    return data


async def import_bundle(store: Store, bundle_dir: Path, *, tags: dict[str, Any] | None = None) -> str:
    """Import a bundle; returns the run id. Existing rows (same keys) are left as they are."""
    data = read_bundle(bundle_dir)
    blobs: list[dict[str, Any]] = []
    for sha, content_type in data.get("blobs", {}).items():
        path = bundle_dir / "blobs" / f"{sha}.zst"
        if not path.exists():
            raise BundleError(f"{bundle_dir}: missing blob {sha}")
        compressed = path.read_bytes()
        raw = decompress(compressed)
        if sha256_hex(raw) != sha:
            raise BundleError(f"{bundle_dir}: blob {sha} does not match its hash")
        blobs.append({"sha256": sha, "size": len(raw), "content_type": content_type, "data": compressed})
    run = dict(data["run"])
    if tags:
        run["tags"] = {**(run.get("tags") or {}), **tags}

    async def op(session: AsyncSession) -> str:
        if blobs:
            await session.execute(insert(Blob).on_conflict_do_nothing(), blobs)
        await session.execute(insert(Run).on_conflict_do_nothing(), [_known_columns(Run, run)])
        existing = (await session.execute(select(Run.id).where(Run.trace_id == run["trace_id"]))).scalar_one()
        for key, model in _TABLES:
            rows = [_known_columns(model, row) for row in data.get(key, [])]
            if key in ("steps", "scores", "labels"):
                rows = [{**row, "run_id": existing} for row in rows]
            if rows:
                await session.execute(insert(model).on_conflict_do_nothing(), rows)
        return str(existing)

    return await store.write(op)


def _known_columns(model: type[Base], row: dict[str, Any]) -> dict[str, Any]:
    """Keep the columns this database knows, so bundles from newer or older versions still import."""
    columns = {attr.key for attr in inspect(model).column_attrs}
    return {k: v for k, v in row.items() if k in columns}
