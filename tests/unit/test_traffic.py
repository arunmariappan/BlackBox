"""`blackbox traffic`: pacing, caps and stop rules, against a fake API and a fake clock."""

import json
import random
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from blackbox.cli import app
from blackbox.live.traffic import Traffic, needs_confirmation, parse_rate
from blackbox.profiles.base import TrafficCase
from blackbox.profiles.paperpilot import PaperPilotProfile


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def api(statuses: list[int | None]) -> tuple[httpx.AsyncClient, list[dict[str, Any]]]:
    """A fake `POST /api/runs`; each call answers with the next agent status (None: the request failed)."""
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        status = statuses[min(len(bodies) - 1, len(statuses) - 1)]
        error = None if status is not None and status < 400 else "agent unreachable"
        return httpx.Response(200, json={"run_id": f"R{len(bodies)}", "status": status, "error": error})

    return httpx.AsyncClient(base_url="http://bb", transport=httpx.MockTransport(handler)), bodies


async def case(rng: random.Random) -> TrafficCase:
    return TrafficCase(f"question {rng.randrange(1000)}", {"top_k": 3}, {"case": "q"})


def traffic(client: httpx.AsyncClient, fake: FakeTime, **kwargs: Any) -> Traffic:
    options: dict[str, Any] = {"rate_per_second": 1 / 60, "max_runs": 5, "max_minutes": 60, "seed": 1}
    options.update(kwargs)
    return Traffic(client, "paperpilot", case, sleep=fake.sleep, clock=fake.clock, report=lambda _: None, **options)


def test_rates_and_confirmation() -> None:
    assert parse_rate("1/min") == pytest.approx(1 / 60)
    assert parse_rate("30/h") == pytest.approx(1 / 120)
    assert parse_rate("0.5/min") == pytest.approx(1 / 120)
    with pytest.raises(ValueError):
        parse_rate("fast")
    assert not needs_confirmation(30, 30) and needs_confirmation(31, 10) and needs_confirmation(10, 31)


async def test_paced_runs_with_the_traffic_source() -> None:
    client, bodies = api([200])
    fake = FakeTime()
    summary = await traffic(client, fake).run()
    assert (summary.started, summary.ok, summary.stopped) == (5, 5, "5 runs done")
    assert fake.now == pytest.approx(240)  # one run a minute
    assert all(b["source"] == "traffic" and b["wait"] is True and b["tags"]["traffic"] is True for b in bodies)
    again, repeated = api([200])
    await traffic(again, FakeTime()).run()
    assert [b["input"] for b in repeated] == [b["input"] for b in bodies]  # the same seed draws the same inputs


async def test_time_limit() -> None:
    client, _ = api([200])
    summary = await traffic(client, FakeTime(), max_runs=60, max_minutes=2).run()
    assert summary.started == 2 and summary.stopped.startswith("time limit")


async def test_stops_after_three_failed_runs_in_a_row() -> None:
    client, _ = api([200, 502, 200, 502, None, 500, 200])
    summary = await traffic(client, FakeTime(), max_runs=20).run()
    assert (summary.started, summary.ok, summary.failed) == (6, 2, 4)
    assert summary.stopped == "3 failed runs in a row"


async def test_stops_when_memory_runs_low() -> None:
    client, _ = api([200])
    free = iter([4 * 1024**3, 3 * 1024**3, 1 * 1024**3])
    summary = await traffic(client, FakeTime(), memory=lambda: next(free)).run()
    assert summary.started == 2 and summary.stopped.startswith("free memory 1.0 GB")


async def test_paperpilot_draws_answerable_questions_more_often(tmp_path: Path) -> None:
    questions = tmp_path / "questions.yaml"
    questions.write_text(
        "name: q\nprofile: paperpilot\nquestions:\n"
        "  - {id: a1, group: answerable, question: 'What is attention?', expected_ending: answered}\n"
        "  - {id: o1, group: out_of_scope, question: 'Boil an egg?', expected_ending: out_of_scope}\n",
        encoding="utf-8",
    )
    profile = PaperPilotProfile({"questions": str(questions)})
    rng = random.Random(5)
    drawn = [await profile.traffic_case(rng) for _ in range(400)]
    answerable = sum(1 for d in drawn if d is not None and d.tags["case"] == "a1") / len(drawn)
    assert 0.68 < answerable < 0.82  # three to one
    assert await PaperPilotProfile({"questions": str(tmp_path / "missing.yaml")}).traffic_case(rng) is None


def test_long_traffic_asks_first() -> None:
    result = CliRunner().invoke(app, ["traffic", "paperpilot", "--max-runs", "100"], input="n\n")
    assert result.exit_code != 0 and "shut down during long GPU runs" in result.output
    help_text = CliRunner().invoke(app, ["live-patch", "--help"]).output
    assert "add" in help_text and "remove" in help_text
    assert "timeline" in CliRunner().invoke(app, ["mark", "--help"]).output
