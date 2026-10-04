"""Formatting helpers for templates and the CLI."""

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import inspect

from blackbox.runs.context import RunContext


def fmt_time(ms: int | None) -> str:
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def fmt_duration(ms: float | int | None) -> str:
    if ms is None:
        return ""
    if ms < 1000:
        return f"{ms:.0f} ms"
    if ms < 60_000:
        return f"{ms / 1000:.1f} s"
    return f"{ms / 60_000:.1f} min"


def fmt_json(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            value = json.loads(value)
        except ValueError:
            return str(value.decode("utf-8", errors="replace"))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return str(value)
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False)


def fmt_num(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3g}"
    return f"{int(value):,}"


def row_dict(obj: Any) -> dict[str, Any]:
    return {attr.key: getattr(obj, attr.key) for attr in inspect(obj).mapper.column_attrs}


def ending_class(ending: str | None) -> str:
    if ending in ("answered", "finished"):
        return "ok"
    if ending in (None, ""):
        return ""
    if ending in ("out_of_scope",):
        return "warn"
    return "bad"


def waterfall(ctx: RunContext, node_spans: frozenset[str]) -> list[dict[str, Any]]:
    """Rows for the span waterfall, in tree order, with bar offsets as percentages of the trace's duration."""
    if not ctx.spans:
        return []
    t0 = min(span.start_ns for span in ctx.spans)
    t1 = max(span.end_ns for span in ctx.spans)
    total = max(t1 - t0, 1)
    rows: list[dict[str, Any]] = []

    def visit(parent: str | None, depth: int) -> None:
        for span in ctx.children.get(parent, []):
            view = ctx.view(span)
            css = "error" if span.status_code == "error" else ""
            if not css:
                css = "node" if span.name in node_spans else ("genai" if view.step_kind else "")
            rows.append(
                {
                    "span": span,
                    "depth": depth,
                    "left": (span.start_ns - t0) / total * 100,
                    "width": max((span.end_ns - span.start_ns) / total * 100, 0.2),
                    "css": css,
                    "duration": span.duration_ms,
                }
            )
            visit(span.span_id, depth + 1)

    visit(None, 0)
    return rows
