"""Detectors over a profile's recent runs: rate drops, shifts in latency, tokens or steps, and new failure modes.

All of them are stateless: each evaluation recomputes its statistic from the stored history, so a restart changes
nothing and a test can replay any stream.

**Rate drop.** The plan's rule (alert when P(current rate < baseline - 15 points) > 0.95 over the last 20 runs) can't
catch a 30-point drop within 10 runs: after 10 bad runs its window still holds 10 good ones. No rule can meet both of
the plan's targets at once either. A 0.9 → 0.6 drop gives about 3 nats of evidence in 10 runs, while staying silent
for 500 runs of noise needs a threshold near 9 nats. So the alarm is a Bernoulli CUSUM tuned to a `design_drop` drop,
summed over the current window against the plan's lagged baseline, with `threshold` (h) trading speed against
quietness. Measured with the defaults (design drop 0.3, h = 6) on simulated streams with a 0.9 baseline: a 30-point
drop is caught after a median of 16 runs (24% within 10, 68% within 20); a collapse to 5%, like a broken guardrail,
after a median of 5 runs and at most 8; a ±5-point wobble raises one false alarm per ~1300 runs. h = 5 is 3 runs
faster with 2.5 times the false alarms. The plan's Beta-binomial probability is still computed and reported with every
result, so an alert reads in the plan's terms.

**Shift** (latency, tokens, steps): a one-sided CUSUM on log values standardised against the baseline's median and
MAD (k = 0.5, h = 5). Logs, because these values are skewed: on raw values a steady log-normal latency alarms by
itself. Each run's z is capped at `z_cap`, so one outlier alone can't cross h.

**New failure mode:** a cluster created in the last 30 minutes, or one that gained `growth` members in that time.
"""

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from scipy.stats import beta

from blackbox.config import NewModeConfig, RateDropConfig, ShiftConfig

type State = Literal["alarm", "hold", "clear", "insufficient"]


@dataclass(frozen=True)
class Windows[T]:
    baseline: list[T]
    current: list[T]


def split_windows[T](
    items: Sequence[T],
    *,
    window: int,
    min_current: int,
    baseline_runs: int,
    min_baseline: int,
    skip: Callable[[T], bool] = lambda _: False,
) -> Windows[T] | None:
    """The current window (the last `window` items) and the baseline before it (the last `baseline_runs` items not
    skipped). While history is short the current window shrinks, down to `min_current`, so the baseline keeps at least
    `min_baseline` items; `None` when even that isn't possible."""
    for size in range(min(window, len(items)), min_current - 1, -1):
        before = items[: len(items) - size]
        baseline = [item for item in before if not skip(item)][-baseline_runs:]
        if len(baseline) >= min_baseline:
            return Windows(baseline, list(items[len(items) - size :]))
    return None


def beta_interval(good: int, n: int, level: float = 0.9) -> tuple[float, float]:
    """Equal-tailed interval of a rate under a Beta(1, 1) prior."""
    tail = (1 - level) / 2
    a, b = 1 + good, 1 + n - good
    return float(beta.ppf(tail, a, b)), float(beta.ppf(1 - tail, a, b))


@dataclass(frozen=True)
class RateDrop:
    state: State
    statistic: float = 0.0
    threshold: float = 0.0
    baseline: float | None = None  # rate of good outcomes, smoothed
    baseline_n: int = 0
    baseline_interval: tuple[float, float] | None = None
    current: float | None = None
    current_n: int = 0
    current_interval: tuple[float, float] | None = None
    p_worse: float | None = None  # P(current < baseline - delta), the plan's Beta-binomial reading
    segment_start: int | None = None  # index in the current window where the drop began

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "statistic": round(self.statistic, 3),
            "threshold": self.threshold,
            "baseline": self.baseline,
            "baseline_n": self.baseline_n,
            "baseline_interval": self.baseline_interval,
            "current": self.current,
            "current_n": self.current_n,
            "current_interval": self.current_interval,
            "p_worse": self.p_worse,
            "segment_start": self.segment_start,
        }


def rate_drop(baseline: Sequence[bool], current: Sequence[bool], cfg: RateDropConfig) -> RateDrop:
    """Has the rate of good outcomes dropped? `True` is a good outcome (a pass, a run without the flag, ...)."""
    if len(baseline) < cfg.min_baseline or len(current) < cfg.min_current:
        return RateDrop("insufficient", threshold=cfg.threshold, baseline_n=len(baseline), current_n=len(current))
    good_b, n_b = sum(baseline), len(baseline)
    b = (good_b + 1) / (n_b + 2)
    p1 = max(b - cfg.design_drop, b / 4)
    ok_llr, bad_llr = math.log(p1 / b), math.log((1 - p1) / (1 - b))
    s, start = 0.0, None
    for i, ok in enumerate(current):
        nxt = max(0.0, s + (ok_llr if ok else bad_llr))
        if s == 0.0 and nxt > 0.0:
            start = i
        elif nxt == 0.0:
            start = None
        s = nxt
    good_c, n_c = sum(current), len(current)
    p_worse = float(beta.cdf(max(b - cfg.delta, 0.0), 1 + good_c, 1 + n_c - good_c))
    state: State = "alarm" if s > cfg.threshold else ("clear" if s <= cfg.threshold / 2 else "hold")
    return RateDrop(
        state,
        s,
        cfg.threshold,
        round(b, 4),
        n_b,
        _rounded(beta_interval(good_b, n_b)),
        round(good_c / n_c, 4),
        n_c,
        _rounded(beta_interval(good_c, n_c)),
        round(p_worse, 4),
        start,
    )


@dataclass(frozen=True)
class Shift:
    state: State
    statistic: float = 0.0
    threshold: float = 0.0
    baseline_median: float | None = None
    baseline_scale: float | None = None
    baseline_n: int = 0
    current_median: float | None = None
    current_n: int = 0
    segment_start: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "statistic": round(self.statistic, 3),
            "threshold": self.threshold,
            "baseline": self.baseline_median,
            "baseline_scale": self.baseline_scale,
            "baseline_n": self.baseline_n,
            "current": self.current_median,
            "current_n": self.current_n,
            "segment_start": self.segment_start,
        }


def shift_up(baseline: Sequence[float], current: Sequence[float], cfg: ShiftConfig, *, min_baseline: int = 30) -> Shift:
    """Has the value (latency, tokens, steps) shifted up? One-sided CUSUM on robust z-scores of log values."""
    if len(baseline) < min_baseline or not current:
        return Shift("insufficient", threshold=cfg.h, baseline_n=len(baseline), current_n=len(current))
    logs = [math.log1p(max(x, 0.0)) for x in baseline]
    median = statistics.median(logs)
    mad = statistics.median(abs(x - median) for x in logs)
    scale = max(1.4826 * mad, 0.02, 1e-9)  # at least 2% on the raw scale
    s, start = 0.0, None
    for i, x in enumerate(current):
        z = min((math.log1p(max(x, 0.0)) - median) / scale, cfg.z_cap)
        nxt = max(0.0, s + z - cfg.k)
        if s == 0.0 and nxt > 0.0:
            start = i
        elif nxt == 0.0:
            start = None
        s = nxt
    state: State = "alarm" if s > cfg.h else ("clear" if s <= cfg.h / 2 else "hold")
    return Shift(
        state,
        s,
        cfg.h,
        round(statistics.median(baseline), 3),
        round(scale, 3),
        len(baseline),
        round(statistics.median(current), 3),
        len(current),
        start,
    )


@dataclass(frozen=True)
class ClusterActivity:
    cluster_id: str
    title: str | None
    created_ms: int
    new_members: list[str] = field(default_factory=list)  # run ids that joined within the window


@dataclass(frozen=True)
class NewMode:
    state: State
    clusters: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"state": self.state, "clusters": self.clusters}


def new_failure_mode(activity: Sequence[ClusterActivity], now_ms: int, cfg: NewModeConfig) -> NewMode:
    """A cluster new within the window (with a member that arrived in it), or one that grew by `growth` members."""
    since = now_ms - int(cfg.minutes * 60_000)
    found: list[dict[str, object]] = []
    for cluster in activity:
        is_new = cluster.created_ms >= since and bool(cluster.new_members)
        grew = len(cluster.new_members) >= cfg.growth
        if is_new or grew:
            found.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "title": cluster.title,
                    "new": is_new,
                    "new_members": len(cluster.new_members),
                    "examples": cluster.new_members[:3],
                }
            )
    return NewMode("alarm" if found else "clear", found)


def _rounded(interval: tuple[float, float]) -> tuple[float, float]:
    return round(interval[0], 4), round(interval[1], 4)
