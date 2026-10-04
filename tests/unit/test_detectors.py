"""Detectors on simulated streams.

The plan's targets ("a 30-point drop caught within 10 runs; a 5-point wobble never caught in 500 runs") can't both
hold on noisy streams (see `blackbox.live.detectors`). They are tested here in the form that can hold: on noise-free
streams exactly as written, except that a moderate 30-point drop is caught within 20 runs; and on seeded random
streams as rates, so a change to the detector shows up as a change in these numbers.
"""

import math
import random

import pytest

from blackbox.config import NewModeConfig, RateDropConfig, ShiftConfig
from blackbox.live.detectors import (
    ClusterActivity,
    new_failure_mode,
    rate_drop,
    shift_up,
    split_windows,
)

CFG = RateDropConfig()


def state_of(history: list[bool], cfg: RateDropConfig = CFG) -> str:
    windows = split_windows(
        history,
        window=cfg.window,
        min_current=cfg.min_current,
        baseline_runs=cfg.baseline_runs,
        min_baseline=cfg.min_baseline,
    )
    if windows is None:
        return "insufficient"
    return rate_drop(windows.baseline, windows.current, cfg).state


def even(rate_of: list[float]) -> list[bool]:
    """A noise-free stream: run t fails exactly when the expected failure count passes a whole number."""
    out, due = [], 0.0
    for p in rate_of:
        before = due
        due += 1 - p
        out.append(math.floor(due) == math.floor(before))
    return out


def random_stream(rng: random.Random, rate_of: list[float]) -> list[bool]:
    return [rng.random() < p for p in rate_of]


def first_alarm(history: list[bool], start: int) -> int | None:
    """Runs after `start` until the first alarm (evaluating after every run, as the worker does)."""
    for t in range(start + 1, len(history) + 1):
        if state_of(history[:t]) == "alarm":
            return t - start
    return None


def test_windows_follow_the_plan() -> None:
    items = list(range(150))
    w = split_windows(items, window=20, min_current=8, baseline_runs=100, min_baseline=30)
    assert w is not None and w.current == items[-20:] and w.baseline == items[30:130]
    short = split_windows(items[:45], window=20, min_current=8, baseline_runs=100, min_baseline=30)
    assert short is not None and len(short.baseline) == 30 and len(short.current) == 15
    assert split_windows(items[:37], window=20, min_current=8, baseline_runs=100, min_baseline=30) is None
    skipped = split_windows(
        items, window=20, min_current=8, baseline_runs=100, min_baseline=30, skip=lambda i: i >= 100
    )
    assert skipped is not None and skipped.baseline == items[:100]  # runs in an alert period stay out of the baseline


def test_noise_free_streams() -> None:
    steady = even([0.9] * 120)
    assert first_alarm(steady + even([0.9] * 500), 120) is None
    wobble = [0.9 + 0.05 * math.sin(2 * math.pi * t / 60) for t in range(500)]
    assert first_alarm(steady + even(wobble), 120) is None  # a 5-point wobble isn't caught in 500 runs
    moderate = first_alarm(steady + even([0.6] * 40), 120)
    assert moderate is not None and moderate <= 20  # a 30-point drop
    collapse = first_alarm(steady + even([0.05] * 20), 120)
    assert collapse is not None and collapse <= 10  # a broken agent


def test_seeded_streams() -> None:
    """Measured rates on seeded noise; the docstring of `blackbox.live.detectors` quotes the full simulation."""
    rng = random.Random(7)
    delays = []
    for _ in range(200):
        history = random_stream(rng, [0.9] * 120 + [0.6] * 40)
        delays.append(first_alarm(history, 120) or 99)
    delays.sort()
    assert delays[len(delays) // 2] <= 18  # median delay for a 30-point drop
    assert sum(d <= 20 for d in delays) / len(delays) >= 0.6
    collapses = [first_alarm(random_stream(rng, [0.9] * 120 + [0.05] * 20), 120) for _ in range(30)]
    assert all(c is not None and c <= 10 for c in collapses)  # a broken agent is caught within 10 runs
    wobble = [0.9 + 0.05 * math.sin(2 * math.pi * t / 60) for t in range(500)]
    false_alarms = sum(first_alarm(random_stream(rng, [0.9] * 120 + wobble), 120) is not None for _ in range(20))
    assert false_alarms / 20 <= 0.5  # about a third of 500-run streams see one false alarm


def test_reading_in_the_plans_terms() -> None:
    reading = rate_drop([True] * 90 + [False] * 10, [True] * 6 + [False] * 14, CFG)
    assert reading.state == "alarm"
    assert reading.baseline == pytest.approx(91 / 102, abs=1e-3)
    assert reading.current == 0.3 and reading.p_worse is not None and reading.p_worse > 0.95
    assert reading.segment_start is not None and reading.current_interval is not None
    lo, hi = reading.current_interval
    assert lo < 0.3 < hi
    assert rate_drop([True] * 20, [False] * 20, CFG).state == "insufficient"  # baseline under min_baseline


def test_cusum_catches_doubled_latency() -> None:
    rng = random.Random(3)
    baseline = [rng.gauss(1000, 100) for _ in range(100)]
    steady = [rng.gauss(1000, 100) for _ in range(20)]
    assert shift_up(baseline, steady, ShiftConfig()).state != "alarm"
    doubled = [rng.gauss(2000, 200) for _ in range(20)]
    for n in range(1, 21):
        if shift_up(baseline, doubled[:n], ShiftConfig()).state == "alarm":
            break
    assert n <= 3
    one_outlier = [*steady[:10], 50_000.0, *steady[10:19]]
    assert shift_up(baseline, one_outlier, ShiftConfig()).state != "alarm"  # z is capped


def test_cusum_is_quiet_on_a_steady_stream() -> None:
    rng = random.Random(11)
    values = [rng.lognormvariate(7, 0.25) for _ in range(620)]
    alarms = sum(
        shift_up(values[t - 120 : t - 20], values[t - 20 : t], ShiftConfig()).state == "alarm" for t in range(120, 620)
    )
    assert alarms == 0


def test_new_failure_mode() -> None:
    now = 10 * 3_600_000
    cfg = NewModeConfig()
    old = now - 5 * 3_600_000
    assert new_failure_mode([ClusterActivity("C1", "loops", old, ["r1", "r2"])], now, cfg).state == "clear"
    grew = new_failure_mode([ClusterActivity("C1", "loops", old, ["r1", "r2", "r3"])], now, cfg)
    assert grew.state == "alarm" and grew.clusters[0]["new_members"] == 3
    fresh = new_failure_mode([ClusterActivity("C2", "refusals", now - 60_000, ["r4"])], now, cfg)
    assert fresh.state == "alarm" and fresh.clusters[0]["new"] is True
    assert new_failure_mode([ClusterActivity("C3", None, now - 60_000, [])], now, cfg).state == "clear"
