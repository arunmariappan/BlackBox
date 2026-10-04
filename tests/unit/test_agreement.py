import math
import random

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from blackbox.judges.agreement import Pair, agreement, bootstrap_interval, cohen_kappa, held_out, report, trust_gate


def test_kappa_hand_computed() -> None:
    # po = 0.75; judge 50/50, truth 25/75 → pe = 0.5·0.25 + 0.5·0.75 = 0.5 → κ = 0.25 / 0.5 = 0.5
    assert cohen_kappa(["pass", "pass", "fail", "fail"], ["pass", "fail", "fail", "fail"]) == pytest.approx(0.5)
    assert cohen_kappa(["pass", "fail"], ["pass", "fail"]) == pytest.approx(1.0)
    assert math.isnan(cohen_kappa(["pass", "pass"], ["pass", "pass"]))


def test_kappa_matches_scikit_learn() -> None:
    rng = random.Random(7)
    for _ in range(20):
        n = rng.randint(5, 60)
        a = [rng.choice(["pass", "fail"]) for _ in range(n)]
        b = [x if rng.random() < 0.7 else rng.choice(["pass", "fail"]) for x in a]
        if len(set(a) | set(b)) < 2:
            continue
        assert cohen_kappa(a, b, ["pass", "fail"]) == pytest.approx(cohen_kappa_score(a, b, labels=["pass", "fail"]))


def test_weighted_kappa_matches_scikit_learn() -> None:
    rng = random.Random(3)
    scale = ["1", "2", "3", "4", "5"]
    a = [rng.choice(scale) for _ in range(80)]
    b = [str(min(5, max(1, int(x) + rng.choice([-1, 0, 0, 1])))) for x in a]
    for weights in ("linear", "quadratic"):
        ours = cohen_kappa(a, b, scale, weights)
        theirs = cohen_kappa_score(a, b, labels=scale, weights=weights)
        assert ours == pytest.approx(theirs)


def test_bootstrap_interval() -> None:
    a = ["pass"] * 10 + ["fail"] * 10
    assert bootstrap_interval(a, a, resamples=200) == (1.0, 1.0)
    b = ["pass"] * 8 + ["fail"] * 2 + ["fail"] * 8 + ["pass"] * 2
    low, high = bootstrap_interval(a, b, resamples=2000, seed=1)
    kappa = cohen_kappa(a, b)
    assert low < kappa < high and low >= -1 and high <= 1
    assert bootstrap_interval(a, b, resamples=2000, seed=1) == (low, high)  # reproducible
    # The same resamples computed independently with scikit-learn give the same interval.
    rng = np.random.default_rng(1)
    values = []
    for _ in range(2000):
        idx = rng.integers(0, 20, 20)
        x, y = [a[i] for i in idx], [b[i] for i in idx]
        if len(set(x) | set(y)) > 1 and not (len(set(x)) == 1 and set(x) == set(y)):
            value = cohen_kappa_score(x, y, labels=["fail", "pass"])
            if not math.isnan(value):
                values.append(value)
    assert (low, high) == pytest.approx(tuple(np.percentile(values, [2.5, 97.5])))


def test_held_out_split_is_stable() -> None:
    ids = [f"01RUN{i:05d}" for i in range(2000)]
    first = [held_out(i) for i in ids]
    assert first == [held_out(i) for i in ids]
    assert 0.36 < sum(first) / len(ids) < 0.44
    more = ids + [f"01NEW{i:05d}" for i in range(500)]  # adding labels never moves an existing run
    assert [held_out(i) for i in more[:2000]] == first


def test_agreement_and_gate() -> None:
    pairs = [Pair(f"r{i}", "fail" if i % 3 == 0 else "pass", "fail" if i % 3 == 0 else "pass") for i in range(200)]
    pairs.append(Pair("unsure-one", "pass", "unsure"))
    stats = agreement(pairs)
    assert stats["n"] == 200 and stats["unsure"] == 1 and stats["kappa"] == pytest.approx(1.0)
    assert stats["confusion"] == [
        [sum(1 for i in range(200) if i % 3), 0],
        [0, sum(1 for i in range(200) if i % 3 == 0)],
    ]
    assert (stats["precision_fail"], stats["recall_fail"]) == (1.0, 1.0)
    full = report(pairs)
    assert full["held_out"]["n"] >= 20 and full["trusted"] is True
    few = report(pairs[:15])
    assert few["trusted"] is False and any("held-out labels" in reason for reason in few["gate_reasons"])
    noisy = [Pair(f"n{i}", "pass" if i % 2 else "fail", "pass" if i % 3 else "fail") for i in range(120)]
    passed, reasons = trust_gate(agreement([p for p in noisy if held_out(p.run_id)]))
    assert passed is False and any("κ" in r for r in reasons)
