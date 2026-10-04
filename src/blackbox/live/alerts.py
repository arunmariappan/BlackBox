"""Alerts: run a profile's detectors and move alerts through their lifecycle.

`open` → shown at once as a banner on every page (SSE `alert`) and sent to Telegram if configured → `resolved`
automatically after `clear_evaluations` clear evaluations in a row (or by hand). One open alert per profile and rule;
after one resolves, the same rule waits `cooldown_minutes` before it can open again. An alert carries the signal, its
baseline and current values with intervals, the window, the failure clusters of the window's runs with examples, and
any marker near the window ("the drop started 3 runs after marker ...").
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.live.detectors import State, new_failure_mode, rate_drop, shift_up, split_windows
from blackbox.live.signals import (
    RunPoint,
    Signal,
    alert_periods,
    cluster_activity,
    failures_in,
    in_periods,
    load_points,
    markers_between,
    signals,
)
from blackbox.live.telegram import Telegram
from blackbox.store.models import Alert
from blackbox.util import new_id, now_ms

if TYPE_CHECKING:
    from blackbox.services import Services

log = logging.getLogger(__name__)

NEW_FAILURE_MODE = "new_failure_mode"


@dataclass
class Evaluation:
    rule: str
    state: State
    reading: dict[str, Any] = field(default_factory=dict)
    signal: Signal | None = None
    current: list[tuple[RunPoint, float]] = field(default_factory=list)


@dataclass
class Transition:
    alert_id: str
    rule: str
    change: str  # opened, updated, resolved, cleared_once


def _flip(value: Any) -> Any:
    """A good-outcome rate as the rate of the bad outcome the signal names (a fail rate, a flag rate, a share)."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [round(1 - value[1], 4), round(1 - value[0], 4)]
    return round(1 - float(value), 4)


def shown(signal: Signal, reading: dict[str, Any]) -> dict[str, Any]:
    """Baseline and current values as the signal names them (pass rate as is, the others as the bad share)."""
    keys = ("baseline", "baseline_interval", "current", "current_interval")
    if signal.kind == "rate" and signal.key != "pass_rate":
        return {k: _flip(reading.get(k)) for k in keys}
    return {k: reading.get(k) for k in keys}


def fmt_value(signal_key: str, value: Any) -> str:
    if value is None:
        return "-"
    if signal_key in ("latency", "tokens", "steps"):
        return f"{float(value):.0f}"
    return f"{100 * float(value):.0f}%"


class AlertEngine:
    def __init__(
        self, services: Services, *, clock: Callable[[], int] = now_ms, telegram: Telegram | None = None
    ) -> None:
        self.services = services
        self.clock = clock
        self.telegram = telegram or Telegram(services.settings.alerts)
        self.open_alerts: list[dict[str, Any]] = []  # for the header banner, refreshed after each evaluation

    # Evaluation ------------------------------------------------------------------------------------------------------

    async def evaluate(self, profile: str) -> list[Transition]:
        live = self.services.settings.live
        now = self.clock()
        points = await load_points(self.services, profile)
        periods = await alert_periods(self.services, profile, now)
        evaluations = [self._evaluate_signal(signal, periods) for signal in signals(points)]
        activity = await cluster_activity(self.services, profile, now - int(live.new_mode.minutes * 60_000))
        mode = new_failure_mode(activity, now, live.new_mode)
        evaluations.append(Evaluation(NEW_FAILURE_MODE, mode.state, mode.as_dict()))
        transitions = []
        for evaluation in evaluations:
            transition = await self.apply(profile, evaluation, now)
            if transition is not None:
                transitions.append(transition)
        await self.refresh_banners()
        return transitions

    def _evaluate_signal(self, signal: Signal, periods: Sequence[tuple[int, int]]) -> Evaluation:
        live = self.services.settings.live
        rd = live.rate_drop
        windows = split_windows(
            signal.points,
            window=rd.window,
            min_current=rd.min_current,
            baseline_runs=rd.baseline_runs,
            min_baseline=rd.min_baseline,
            skip=lambda item: in_periods(item[0].t_ms, periods),
        )
        if signal.kind == "rate":
            rule = f"rate_drop:{signal.key}"
            if windows is None:
                return Evaluation(rule, "insufficient", signal=signal)
            result = rate_drop(
                [v >= 0.5 for _, v in windows.baseline], [v >= 0.5 for _, v in windows.current], rd
            ).as_dict()
        else:
            rule = f"shift:{signal.key}"
            if windows is None:
                return Evaluation(rule, "insufficient", signal=signal)
            result = shift_up(
                [v for _, v in windows.baseline],
                [v for _, v in windows.current],
                live.shift,
                min_baseline=rd.min_baseline,
            ).as_dict()
        state = result["state"]
        assert isinstance(state, str)
        reading: dict[str, Any] = {"signal": signal.key, "label": signal.label, "kind": signal.kind, **result}
        reading["shown"] = shown(signal, reading)
        if windows.current:
            first, last = windows.current[0][0], windows.current[-1][0]
            reading["window"] = {
                "first_run": first.run_id,
                "last_run": last.run_id,
                "start_ms": first.t_ms,
                "end_ms": last.t_ms,
                "runs": len(windows.current),
            }
            start = result.get("segment_start")
            if isinstance(start, int):
                reading["window"]["drop_started_run"] = windows.current[start][0].run_id
                reading["window"]["drop_started_ms"] = windows.current[start][0].t_ms
        return Evaluation(rule, state, reading, signal, windows.current)  # type: ignore[arg-type]

    # Lifecycle -------------------------------------------------------------------------------------------------------

    async def open_alert(self, profile: str, rule: str) -> Alert | None:
        async with self.services.store.read() as s:
            return (
                await s.execute(
                    select(Alert).where(Alert.profile == profile, Alert.rule == rule, Alert.status == "open").limit(1)
                )
            ).scalar_one_or_none()

    async def cooling(self, profile: str, rule: str, now: int) -> bool:
        since = now - int(self.services.settings.live.cooldown_minutes * 60_000)
        async with self.services.store.read() as s:
            recent = (
                await s.execute(
                    select(Alert.id)
                    .where(
                        Alert.profile == profile,
                        Alert.rule == rule,
                        Alert.status == "resolved",
                        Alert.closed_ms >= since,
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
        return recent is not None

    async def apply(self, profile: str, evaluation: Evaluation, now: int) -> Transition | None:
        rule, state = evaluation.rule, evaluation.state
        current = await self.open_alert(profile, rule)
        if state == "alarm":
            if current is not None:
                await self._update(current.id, latest=evaluation.reading, clear_count=0, now=now)
                return Transition(current.id, rule, "updated")
            if await self.cooling(profile, rule, now):
                return None
            return await self._open(profile, evaluation, now)
        if current is None or state == "insufficient":
            return None
        if state == "hold":
            await self._update(current.id, latest=evaluation.reading, clear_count=0, now=now)
            return None
        clears = current.clear_count + 1
        if clears >= self.services.settings.live.clear_evaluations:
            await self.resolve(current.id, now=now, reason="clear")
            return Transition(current.id, rule, "resolved")
        await self._update(current.id, latest=evaluation.reading, clear_count=clears, now=now)
        return Transition(current.id, rule, "cleared_once")

    async def _open(self, profile: str, evaluation: Evaluation, now: int) -> Transition:
        details = {**evaluation.reading, **await self._context(profile, evaluation, now)}
        details["evaluated_ms"] = now
        alert = Alert(
            id=new_id(),
            profile=profile,
            rule=evaluation.rule,
            severity="warning",
            status="open",
            opened_ms=now,
            closed_ms=None,
            clear_count=0,
            details=details,
            notified=False,
        )

        async def op(session: AsyncSession) -> None:
            session.add(alert)

        await self.services.store.write(op)
        log.warning("alert opened: %s %s", profile, evaluation.rule)
        self.services.bus.publish("alert.opened", alert_id=alert.id, profile=profile, rule=evaluation.rule)
        if self.telegram.enabled:
            self.services.spawn(self._notify(alert.id, summary(alert)), name=f"telegram-{alert.id}")
        return Transition(alert.id, evaluation.rule, "opened")

    async def _context(self, profile: str, evaluation: Evaluation, now: int) -> dict[str, Any]:
        """Failure clusters of the window's runs, and markers near the window."""
        if evaluation.rule == NEW_FAILURE_MODE:
            return {}
        run_ids = [point.run_id for point, _ in evaluation.current]
        context: dict[str, Any] = {"clusters": await failures_in(self.services, run_ids)}
        if not evaluation.current or evaluation.signal is None:
            return context
        window = evaluation.reading.get("window", {})
        start_ms = evaluation.current[0][0].t_ms
        span = max(window.get("end_ms", now) - start_ms, 60_000)
        marks = await markers_between(self.services, profile, start_ms - span, now)
        times = [point.t_ms for point, _ in evaluation.signal.points]
        drop_ms = window.get("drop_started_ms")
        notes = []
        for marker in marks:
            item: dict[str, Any] = {"id": marker.id, "text": marker.text, "created_ms": marker.created_ms}
            if isinstance(drop_ms, int):
                if marker.created_ms <= drop_ms:
                    n = sum(1 for t in times if marker.created_ms <= t < drop_ms)
                    item["note"] = f"the drop started {n} runs after marker “{marker.text}”"
                else:
                    n = sum(1 for t in times if drop_ms <= t < marker.created_ms)
                    item["note"] = f"marker “{marker.text}” came {n} runs after the drop started"
            notes.append(item)
        context["markers"] = notes
        return context

    async def _update(self, alert_id: str, *, latest: dict[str, Any], clear_count: int, now: int) -> None:
        async def op(session: AsyncSession) -> None:
            alert = await session.get(Alert, alert_id)
            if alert is None:
                return
            alert.clear_count = clear_count
            alert.details = {**alert.details, "latest": latest, "evaluated_ms": now}

        await self.services.store.write(op)

    async def resolve(self, alert_id: str, *, now: int | None = None, reason: str = "clear") -> Alert | None:
        when = now if now is not None else self.clock()

        async def op(session: AsyncSession) -> Alert | None:
            alert = await session.get(Alert, alert_id)
            if alert is None or alert.status != "open":
                return None
            alert.status, alert.closed_ms = "resolved", when
            alert.details = {**alert.details, "resolved_by": reason}
            return alert

        alert = await self.services.store.write(op)
        if alert is None:
            return None
        log.warning("alert resolved: %s %s (%s)", alert.profile, alert.rule, reason)
        self.services.bus.publish("alert.resolved", alert_id=alert.id, profile=alert.profile, rule=alert.rule)
        if self.telegram.enabled and alert.notified:
            text = f"BlackBox: resolved: {alert.profile} {alert.details.get('label') or alert.rule}"
            self.services.spawn(self.telegram.send(text), name=f"telegram-resolve-{alert.id}")
        await self.refresh_banners()
        return alert

    async def _notify(self, alert_id: str, text: str) -> None:
        url = f"{self.services.settings.server.base_url}/alerts/{alert_id}"
        if await self.telegram.send(f"{text}\n{url}"):

            async def op(session: AsyncSession) -> None:
                alert = await session.get(Alert, alert_id)
                if alert is not None:
                    alert.notified = True

            await self.services.store.write(op)

    async def refresh_banners(self) -> None:
        async with self.services.store.read() as s:
            rows = (
                (await s.execute(select(Alert).where(Alert.status == "open").order_by(Alert.opened_ms))).scalars().all()
            )
        self.open_alerts = [{"id": a.id, "profile": a.profile, "rule": a.rule, "summary": summary(a)} for a in rows]


def summary(alert: Alert) -> str:
    details = alert.details
    if alert.rule == NEW_FAILURE_MODE:
        clusters = details.get("clusters") or []
        names = ", ".join(str(c.get("title") or c.get("cluster_id")) for c in clusters[:3])
        return f"{alert.profile}: new failure mode ({names})"
    key = str(details.get("signal", ""))
    values = details.get("shown") or {}
    direction = "dropped" if key == "pass_rate" else "rose"
    runs = (details.get("window") or {}).get("runs")
    return (
        f"{alert.profile}: {details.get('label', alert.rule)} {direction} from "
        f"{fmt_value(key, values.get('baseline'))} to {fmt_value(key, values.get('current'))}"
        + (f" over the last {runs} runs" if runs else "")
    )
