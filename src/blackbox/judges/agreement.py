"""How often a judge agrees with your labels (or OpsDesk's checker), and whether it may be trusted.

- Cohen's κ (quadratic-weighted for ordinal scales) with a 95% bootstrap interval (2,000 resamples), the confusion
  matrix, percent agreement, and the judge's precision and recall on `fail`, the class that matters.
- **Held-out labels:** 40% of runs, chosen by a hash of the run id, are kept out while you tune a prompt. The trust
  gate uses only those, so tuning can't overfit to the labels it is scored on.
- **Trust gate:** at least 20 held-out labels, κ ≥ 0.6 and the interval's lower bound ≥ 0.3.
"""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

HELD_OUT_FRACTION = 0.4
GATE_MIN_LABELS = 20
GATE_MIN_KAPPA = 0.6
GATE_MIN_LOWER = 0.3
BINARY = ("pass", "fail")


def held_out(run_id: str, fraction: float = HELD_OUT_FRACTION) -> bool:
    """Stable for a run id: adding labels never moves a run between the tuning and held-out sets."""
    digest = hashlib.sha256(f"heldout:{run_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def _matrix(a: Sequence[str], b: Sequence[str], labels: Sequence[str]) -> np.ndarray[Any, np.dtype[np.float64]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b, strict=True):
        matrix[index[x], index[y]] += 1
    return matrix


def cohen_kappa(
    a: Sequence[str], b: Sequence[str], labels: Sequence[str] | None = None, weights: str | None = None
) -> float:
    """Cohen's κ between two raters. `weights="quadratic"` (or `"linear"`) for ordinal labels, in `labels` order.
    NaN when κ is undefined (both raters always gave the same single label)."""
    if len(a) != len(b):
        raise ValueError("both raters must rate the same items")
    if not a:
        return math.nan
    order = list(labels) if labels is not None else sorted(set(a) | set(b))
    k = len(order)
    observed = _matrix(a, b, order) / len(a)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.indices((k, k))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / max(k - 1, 1)
    elif weights == "quadratic":
        w = ((i - j) / max(k - 1, 1)) ** 2
    else:
        raise ValueError(f"unknown weights {weights!r}")
    denominator = float((w * expected).sum())
    if denominator == 0:
        return math.nan
    return 1.0 - float((w * observed).sum()) / denominator


def bootstrap_interval(
    a: Sequence[str],
    b: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    weights: str | None = None,
    resamples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """A percentile bootstrap interval for κ; resamples where κ is undefined are dropped."""
    n = len(a)
    if n == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    order = list(labels) if labels is not None else sorted(set(a) | set(b))
    values = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        value = cohen_kappa(list(a_arr[idx]), list(b_arr[idx]), order, weights)
        if not math.isnan(value):
            values.append(value)
    if not values:
        return (math.nan, math.nan)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return (float(low), float(high))


def precision_recall(judge: Sequence[str], truth: Sequence[str], positive: str = "fail") -> tuple[float, float]:
    tp = sum(1 for j, t in zip(judge, truth, strict=True) if j == positive and t == positive)
    flagged = sum(1 for j in judge if j == positive)
    actual = sum(1 for t in truth if t == positive)
    return (tp / flagged if flagged else math.nan, tp / actual if actual else math.nan)


@dataclass
class Pair:
    run_id: str
    judge: str  # the judge's label (pass / fail)
    truth: str  # your label, or the checker's (pass / fail / unsure)


def agreement(pairs: Sequence[Pair], *, labels: Sequence[str] = BINARY, weights: str | None = None) -> dict[str, Any]:
    """Agreement statistics over `pairs`; `unsure` truths are left out and counted."""
    usable = [p for p in pairs if p.truth in labels and p.judge in labels]
    unsure = sum(1 for p in pairs if p.truth not in labels)
    judge = [p.judge for p in usable]
    truth = [p.truth for p in usable]
    n = len(usable)
    kappa = cohen_kappa(judge, truth, labels, weights) if n else math.nan
    low, high = bootstrap_interval(judge, truth, labels=labels, weights=weights) if n else (math.nan, math.nan)
    precision, recall = precision_recall(judge, truth)
    matrix = _matrix(judge, truth, labels) if n else np.zeros((len(labels), len(labels)))
    return {
        "n": n,
        "unsure": unsure,
        "percent": (sum(1 for j, t in zip(judge, truth, strict=True) if j == t) / n * 100) if n else math.nan,
        "kappa": kappa,
        "ci": [low, high],
        "precision_fail": precision,
        "recall_fail": recall,
        "labels": list(labels),
        # rows: judge's label, columns: the truth
        "confusion": [[int(x) for x in row] for row in matrix],
        "disagreements": [p.run_id for p in usable if p.judge != p.truth],
    }


def trust_gate(held: dict[str, Any]) -> tuple[bool, list[str]]:
    """Whether a judge passes on its held-out statistics, and why not when it doesn't."""
    reasons = []
    if held["n"] < GATE_MIN_LABELS:
        reasons.append(f"only {held['n']} held-out labels (needs {GATE_MIN_LABELS})")
    kappa = held["kappa"]
    if math.isnan(kappa) or kappa < GATE_MIN_KAPPA:
        reasons.append(f"κ {kappa:.2f} below {GATE_MIN_KAPPA}" if not math.isnan(kappa) else "κ undefined")
    low = held["ci"][0]
    if math.isnan(low) or low < GATE_MIN_LOWER:
        reasons.append(
            f"interval lower bound {low:.2f} below {GATE_MIN_LOWER}" if not math.isnan(low) else "no interval"
        )
    return (not reasons, reasons)


def report(pairs: Sequence[Pair], *, labels: Sequence[str] = BINARY, weights: str | None = None) -> dict[str, Any]:
    """Statistics on all pairs and on the held-out ones, and the trust gate's decision."""
    held = [p for p in pairs if held_out(p.run_id)]
    tuning = [p for p in pairs if not held_out(p.run_id)]
    held_stats = agreement(held, labels=labels, weights=weights)
    trusted, reasons = trust_gate(held_stats)
    return {
        "all": agreement(pairs, labels=labels, weights=weights),
        "held_out": held_stats,
        "tuning": agreement(tuning, labels=labels, weights=weights),
        "trusted": trusted,
        "gate_reasons": reasons,
    }


def json_safe(value: Any) -> Any:
    """NaN → None, so reports can be stored as JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value
