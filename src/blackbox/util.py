"""Small helpers shared by every part of BlackBox: ids, clocks, hashing and canonical JSON."""

import hashlib
import json
import time
from typing import Any

from ulid import ULID


def new_id() -> str:
    """A new ULID as a 26-character string; BlackBox's own rows use these."""
    return str(ULID())


def now_ms() -> int:
    """The current UTC time in epoch milliseconds."""
    return time.time_ns() // 1_000_000


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> str:
    """JSON with sorted keys and no whitespace, so equal values always give equal text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty_json(value: Any) -> str:
    """JSON with sorted keys and two-space indent, as committed files are written."""
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def parse_duration_ms(text: str) -> int:
    """Parse `30d`, `12h`, `15m`, `90s` or `500ms` into milliseconds."""
    units = {"ms": 1, "s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
    text = text.strip().lower()
    for suffix in ("ms", "s", "m", "h", "d"):
        if text.endswith(suffix) and text[: -len(suffix)].replace(".", "", 1).isdigit():
            return int(float(text[: -len(suffix)]) * units[suffix])
    raise ValueError(f"not a duration: {text!r} (use e.g. 30d, 12h, 15m, 90s)")
