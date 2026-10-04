"""The failure pipeline end to end with a fake model: describe failures, re-cluster after 10 wait, name clusters
with validated evidence, assign new failures live, keep ids across re-clustering, and show it all in the UI."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx
from sklearn.metrics import adjusted_rand_score
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.clusters.jobs import describe_failure
from blackbox.config import ClustersConfig, OllamaConfig
from blackbox.live.worker import JobView
from blackbox.runs.scores import write_score
from blackbox.store.models import Cluster, Failure, Job, Run, Step
from blackbox.util import new_id, now_ms
from tests.factories import trace_id
from tests.fakes import FakeUpstream
from tests.harness import running_blackbox

TYPES = {
    "loop": ("loop", "kept restarting the same service again and again without progress"),
    "wrong_tool": ("wrong_tool", "restarted {svc}, a symptom, instead of fixing the cache that caused it"),
    "gave_up": ("gave_up_after_error", "stopped after {svc} returned 503 instead of retrying"),
    "policy": ("policy_violation", "changed {svc} without opening a ticket first"),
}
SERVICES = [
    "web-frontend",
    "checkout-api",
    "auth-service",
    "search-indexer",
    "email-worker",
    "report-job",
    "cache",
    "payments-db",
]


def fake_model(body: dict[str, Any]) -> dict[str, Any]:
    prompt = body["messages"][-1]["content"]
    if "Categories:" in prompt:  # a failure description
        kind = re.search(r"Input: TYPE:(\w+) (\S+)", prompt)
        assert kind is not None
        category, template = TYPES[kind.group(1)]
        content = {
            "what_went_wrong": "The agent " + template.format(svc=kind.group(2)),
            "where_step": 2,
            "category": category,
        }
    else:  # a cluster's name
        categories = re.findall(r"categories: \d+% (\w+)", prompt)
        title = categories[0] if categories else "unknown"
        content = {
            "title": f"Runs with {title.replace('_', ' ')}",
            "likely_cause": f"The agent shows {title.replace('_', ' ')} in every run",
            "evidence": [
                {"run": "R1", "step": 2, "observation": "the problem shows here"},
                {"run": "R9", "step": 1, "observation": "a run that isn't in the cluster"},
                {"run": "R2", "step": 999, "observation": "a step that doesn't exist"},
            ],
            "suggested_fix": "Add a rule to the prompt",
        }
    return {"role": "assistant", "content": json.dumps(content)}


async def failed_run(bb: Any, kind: str, index: int) -> Run:
    service = SERVICES[index % len(SERVICES)]
    run = Run(
        id=new_id(),
        trace_id=trace_id(),
        profile="opsdesk",
        status="complete",
        source="live",
        ending="finished",
        started_ms=now_ms(),
        updated_ms=now_ms(),
        input_text=f"TYPE:{kind} {service} please fix",
        step_count=3,
        tags={"known_failure": kind},
    )

    async def op(session: AsyncSession) -> None:
        session.add(run)
        for idx in (1, 2, 3):
            session.add(
                Step(
                    run_id=run.id,
                    idx=idx,
                    kind="tool" if idx == 2 else "llm",
                    tool_name="restart_service" if idx == 2 else None,
                    node="plan",
                    status="ok",
                    view={},
                )
            )

    await bb.services.store.write(op)
    await write_score(
        bb.services.store,
        run.id,
        kind="checker",
        name="checker",
        version="1",
        label="fail",
        value=0.0,
        details={"failed": [f"planted {kind}"]},
    )
    return run


async def test_failures_cluster_by_type_with_stable_ids(tmp_path: Path, migrated_db: Path) -> None:
    model = await FakeUpstream(reply=fake_model).start()
    model.release.set()
    try:
        async with running_blackbox(
            tmp_path,
            migrated_db,
            ollama=OllamaConfig(base_url=model.url),
            clusters=ClustersConfig(embedder="hashing", recluster_after=10),
        ) as bb:
            services = bb.services
            assert services.clusters is not None
            runs = [await failed_run(bb, kind, i) for kind in TYPES for i in range(6)]
            for n, run in enumerate(runs, start=1):
                await describe_failure(services, JobView(new_id(), "describe_failure", "model", run.id, {}, 1))
                if n == 9:
                    async with services.store.read() as s:
                        assert not (await s.execute(select(Job).where(Job.kind == "recluster"))).scalars().all()
            async with services.store.read() as s:
                queued = (await s.execute(select(Job).where(Job.kind == "recluster"))).scalars().all()
            assert queued  # ten failures waiting unclustered queued a re-clustering (none before the tenth)
            async with asyncio.timeout(20):  # let the worker finish the re-clusterings it was given
                while True:
                    async with services.store.read() as s:
                        busy = (
                            (
                                await s.execute(
                                    select(Job).where(Job.kind == "recluster", Job.status.in_(("queued", "running")))
                                )
                            )
                            .scalars()
                            .all()
                        )
                    if not busy:
                        break
                    await asyncio.sleep(0.1)
            first = await services.clusters.recluster("opsdesk")
            assert first["clusters"] == 4, first
            summary = await services.clusters.recluster("opsdesk")  # nothing changed: ids and names hold
            assert summary["clusters"] == 4 and summary["kept_ids"] == 4 and summary["named"] == 0
            async with services.store.read() as s:
                failures = {f.run_id: f for f in (await s.execute(select(Failure))).scalars()}
                clusters = {
                    c.id: c for c in (await s.execute(select(Cluster).where(Cluster.status != "retired"))).scalars()
                }
            truth = [run.tags["known_failure"] for run in runs]
            predicted = [failures[run.id].cluster_id or f"noise-{run.id}" for run in runs]
            assert adjusted_rand_score(truth, predicted) >= 0.6
            for cluster in clusters.values():
                assert cluster.cause_status == "supported"
                assert [e["observation"] for e in cluster.evidence] == [
                    "the problem shows here"
                ]  # bad evidence dropped
                kinds = {r.tags["known_failure"] for r in runs if failures[r.id].cluster_id == cluster.id}
                assert len(kinds) == 1 and TYPES[kinds.pop()][0].replace("_", " ") in cluster.likely_cause

            loop_cluster = failures[runs[0].id].cluster_id
            names_before = {c.id: c.title for c in clusters.values()}
            extra = [await failed_run(bb, "loop", 20 + i) for i in range(2)]
            for run in extra:
                recorded = await services.clusters.record(run)
                assert recorded.cluster_id == loop_cluster  # joined live, within the radius
            again = await services.clusters.recluster("opsdesk")
            assert again["kept_ids"] == 4 and again["named"] == 0  # 2 of 8 members changed: under 30%, no rename
            async with services.store.read() as s:
                after = {
                    c.id: c for c in (await s.execute(select(Cluster).where(Cluster.status != "retired"))).scalars()
                }
            assert {cid: c.title for cid, c in after.items()} == names_before
            assert after[loop_cluster].size == 8

            async with httpx.AsyncClient(base_url=bb.base_url, follow_redirects=False) as client:
                page = await client.get("/clusters?profile=opsdesk")
                assert page.status_code == 200 and "Runs with loop" in page.text
                detail = await client.get(f"/clusters/{loop_cluster}")
                assert "the problem shows here" in detail.text and "#step-2" in detail.text
                run_page = await client.get(f"/runs/{runs[0].id}")
                assert (
                    "Failure" in run_page.text
                    and "kept restarting" in run_page.text
                    and "Runs with loop" in run_page.text
                )
                status = await client.post(f"/clusters/{loop_cluster}/status", data={"status": "known"})
                assert status.status_code == 303
                api = (await client.get("/api/clusters?profile=opsdesk")).json()
                assert {c["id"]: c["status"] for c in api}[loop_cluster] == "known"
    finally:
        await model.stop()
