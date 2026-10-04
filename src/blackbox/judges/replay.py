"""Judge results in fidelity reports: did a fork make the answer better or worse, not only different?"""

from typing import TYPE_CHECKING, Any

from blackbox.llm.client import LLMUnavailable
from blackbox.runs.context import RunContext

if TYPE_CHECKING:
    from blackbox.proxy.sessions import ReplaySession
    from blackbox.services import Services


async def judge_report(
    services: Services, session: ReplaySession, source: RunContext, replay: RunContext
) -> dict[str, Any]:
    """Judge the source and the replay with each of the profile's judges. An exact replay gives the same inputs, so
    its verdicts come from the cache without a model call."""
    if not services.settings.judges.judge_replays:
        return {}
    results: dict[str, Any] = {}
    for judge in services.judges.values():
        if judge.profile != source.run.profile:
            continue
        entry: dict[str, Any] = {}
        for which, ctx in (("source", source), ("replay", replay)):
            try:
                outcome = await services.judge_runner.judge_run(judge, ctx.run)
            except LLMUnavailable as exc:
                entry[which] = {"status": "unavailable", "error": str(exc)[:200]}
                continue
            entry[which] = {"status": outcome.status, "label": outcome.label, "rationale": outcome.rationale}
        source_label = entry.get("source", {}).get("label")
        replay_label = entry.get("replay", {}).get("label")
        if source_label and replay_label:
            entry["change"] = (
                "same" if source_label == replay_label else "better" if replay_label == "pass" else "worse"
            )
        results[judge.name] = entry
    return {"judges": results} if results else {}
