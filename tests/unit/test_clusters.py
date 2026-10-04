import json
from typing import Any

import httpx
import numpy as np

from blackbox.clusters.failures import facts
from blackbox.clusters.naming import Evidence, Member, name_cluster, validate_evidence
from blackbox.clusters.space import Space, assign, centre_and_radius, cluster_labels, match_ids, tokens_of
from blackbox.config import OllamaConfig
from blackbox.llm.client import OllamaJSON
from blackbox.profiles.opsdesk import OpsDeskProfile
from blackbox.runs.context import RunContext, StepDraft
from blackbox.store.models import Run, Score


def test_three_groups_and_noise() -> None:
    rng = np.random.default_rng(4)
    centres = [np.eye(8)[i] * 3 for i in range(3)]
    groups = [centre + rng.normal(0, 0.05, (10, 8)) for centre in centres]
    noise = np.stack([np.eye(8)[3 + i] * 20 * (1 + i) + rng.normal(0, 1, 8) for i in range(5)])  # far from everything
    vectors = np.vstack([*groups, noise]).astype(np.float32)
    labels = cluster_labels(vectors, min_cluster_size=3)
    group_labels = [set(labels[i * 10 : (i + 1) * 10]) for i in range(3)]
    assert all(len(g) == 1 and -1 not in g for g in group_labels)
    assert len({next(iter(g)) for g in group_labels}) == 3
    assert labels[30:] == [-1] * 5
    assert cluster_labels(vectors[:2]) == [-1, -1]  # fewer than min_cluster_size


def test_ids_survive_growth_and_splits() -> None:
    old = {"C1": {"a", "b", "c", "d"}, "C2": {"x", "y", "z"}}
    assert match_ids(old, [{"a", "b", "c", "d", "e"}, {"x", "y", "z"}]) == ["C1", "C2"]
    split = {"C1": set("abcdefghij")}
    assert match_ids(split, [set("hij"), set("abcdefg")]) == [None, "C1"]  # the larger part keeps the id
    assert match_ids(old, [{"p", "q", "r"}]) == [None]


def test_assignment_within_the_radius() -> None:
    members = np.array([[1.0, 0.0], [1.1, 0.0], [0.9, 0.0], [1.0, 0.1]], dtype=np.float32)
    centre, radius = centre_and_radius(members)
    centres = [("C1", centre, radius)]
    assert assign(np.array([1.02, 0.01], dtype=np.float32), centres) == "C1"
    assert assign(np.array([3.0, 3.0], dtype=np.float32), centres) is None


def test_space_vectors() -> None:
    signature = {"ending": "finished", "flags": ["loop"], "first_problem_node": "restart_service"}
    tokens = tokens_of(signature, "loop")
    space = Space.build([tokens, ["ending:max_steps"]], weight=0.5)
    vector = space.vector(np.ones(4, dtype=np.float32) / 2, tokens)
    assert vector.shape == (4 + len(space.vocabulary),)
    assert abs(float(np.linalg.norm(vector[4:])) - 0.5) < 1e-6


MEMBERS = [Member("01RUNA", "R1", "loop", "", steps=5), Member("01RUNB", "R2", "loop", "", steps=3)]


def test_invalid_evidence_is_dropped() -> None:
    evidence = [
        Evidence(run="R1", step=4, observation="restarted again"),
        Evidence(run="R7", step=1, observation="not in the cluster"),
        Evidence(run="R2", step=9, observation="no such step"),
        Evidence(run="01RUNB", step=None, observation="full ids work too"),
    ]
    valid, dropped = validate_evidence(evidence, MEMBERS)
    assert [(v["run_id"], v["step"]) for v in valid] == [("01RUNA", 4), ("01RUNB", None)]
    assert [d["reason"] for d in dropped] == ["run not in the cluster", "run has no step 9"]


def fake_llm(content: dict[str, Any]) -> OllamaJSON:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": json.dumps(content)}, "done": True}
        )

    return OllamaJSON(OllamaConfig(), httpx.AsyncClient(base_url="http://x", transport=httpx.MockTransport(handler)))


async def test_a_cause_with_no_valid_evidence_is_unsupported() -> None:
    bad = {
        "title": "Loops",
        "likely_cause": "c",
        "suggested_fix": "f",
        "evidence": [{"run": "R9", "step": 1, "observation": "o"}],
    }
    naming = await name_cluster(fake_llm(bad), MEMBERS, "stats", 2)
    assert naming.status == "unsupported" and naming.evidence == [] and len(naming.dropped) == 1
    good = {**bad, "evidence": [{"run": "R1", "step": 2, "observation": "o"}]}
    assert (await name_cluster(fake_llm(good), MEMBERS, "stats", 2)).status == "supported"


def test_untrusted_judges_do_not_decide_failure() -> None:
    run = Run(id="R", trace_id="t" * 32, profile="opsdesk", status="complete", ending="finished", updated_ms=0)
    ctx = RunContext(run=run, spans=[], steps=[StepDraft(idx=1, kind="llm")])
    judge = Score(
        id="S", run_id="R", kind="judge", name="ops_task_success", version="v1", label="fail", details={}, created_ms=0
    )
    assert facts(ctx, OpsDeskProfile(), [judge], trusted={}).failed is False
    trusted = facts(ctx, OpsDeskProfile(), [judge], trusted={"ops_task_success": {"v1"}})
    assert trusted.failed is True and trusted.failed_judges == ["ops_task_success"]
    ending = facts(replace_ending(ctx, "max_steps"), OpsDeskProfile(), [], trusted={})
    assert ending.failed is True and ending.reasons == ["ending max_steps"]


def replace_ending(ctx: RunContext, ending: str) -> RunContext:
    run = Run(id="R", trace_id="t" * 32, profile="opsdesk", status="complete", ending=ending, updated_ms=0)
    return RunContext(run=run, spans=[], steps=ctx.steps)
