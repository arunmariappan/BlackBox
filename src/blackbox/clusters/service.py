"""Failures and clusters in the store: recording a failure (facts, description, embedding), assigning it to a cluster
as it arrives, and re-clustering a profile with stable ids."""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.clusters.describe import describe
from blackbox.clusters.embedding import Embedder, from_bytes, make_embedder, to_bytes
from blackbox.clusters.failures import FailureFacts, digest, facts
from blackbox.clusters.naming import Member, Naming, name_cluster
from blackbox.clusters.space import Space, assign, centre_and_radius, cluster_labels, jaccard, match_ids, tokens_of
from blackbox.judges.calibrate import trusted_versions
from blackbox.runs.context import RunContext, load_run_context
from blackbox.store.models import Cluster, ClusterSpace, Failure, Run, Score
from blackbox.util import new_id, now_ms

if TYPE_CHECKING:
    from blackbox.services import Services

log = logging.getLogger(__name__)


@dataclass
class Recorded:
    run_id: str
    failed: bool
    category: str | None = None
    cluster_id: str | None = None


class ClusterService:
    def __init__(self, services: Services, embedder: Embedder | None = None) -> None:
        self.services = services
        self.store = services.store
        self.config = services.settings.clusters
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            config = self.config
            self._embedder = make_embedder(config.embedder, config.model, config.cache_dir)
        return self._embedder

    # Failures -----------------------------------------------------------------------------------------------------

    async def facts_of(self, run: Run) -> tuple[FailureFacts, RunContext]:
        ctx = await load_run_context(self.store, run)
        async with self.store.read() as s:
            scores = list((await s.execute(select(Score).where(Score.run_id == run.id))).scalars())
        trusted = await trusted_versions(self.store)
        return facts(ctx, self.services.profiles.find(run.profile), scores, trusted), ctx

    async def is_failed(self, run: Run) -> bool:
        found, _ = await self.facts_of(run)
        return found.failed

    async def record(self, run: Run) -> Recorded:
        """Record (or clear) the run's failure: describe it with the model, embed it, and assign it to a cluster."""
        found, ctx = await self.facts_of(run)
        if not found.failed:
            await self._delete_failure(run.id)
            return Recorded(run.id, False)
        text = digest(ctx, found)
        description, error = await describe(self.services.llm, text, len(ctx.steps))
        if description is not None:
            embed_text = f"{description.category}: {description.what_went_wrong}"
        else:
            embed_text = "; ".join(found.reasons + found.failed_checks[:3])
        vector = (await self.embedder.embed([embed_text]))[0]
        signature = {**found.as_dict(), "digest": text, "embedder": self.embedder.name, "description_error": error}
        category = description.category if description is not None else None
        where = description.where_step if description is not None else None
        values = {
            "run_id": run.id,
            "profile": run.profile or "",
            "signature": signature,
            "description": description.what_went_wrong if description is not None else None,
            "category": category,
            "where_step": where,
            "embedding": to_bytes(vector),
            "cluster_id": None,
            "created_ms": now_ms(),
        }

        async def op(session: AsyncSession) -> None:
            await session.execute(
                insert(Failure).values(**values).on_conflict_do_update(index_elements=["run_id"], set_=values)
            )

        await self.store.write(op)
        cluster_id = await self.assign(run.profile or "", run.id, vector, tokens_of(signature, category))
        self.services.bus.publish("cluster.failure", run_id=run.id, cluster_id=cluster_id, profile=run.profile)
        return Recorded(run.id, True, category, cluster_id)

    async def _delete_failure(self, run_id: str) -> None:
        async def op(session: AsyncSession) -> None:
            failure = await session.get(Failure, run_id)
            if failure is not None:
                if failure.cluster_id:
                    await session.execute(
                        update(Cluster).where(Cluster.id == failure.cluster_id).values(size=Cluster.size - 1)
                    )
                await session.delete(failure)

        await self.store.write(op)

    # Live assignment ------------------------------------------------------------------------------------------------

    async def space(self, profile: str) -> ClusterSpace | None:
        async with self.store.read() as s:
            return await s.get(ClusterSpace, profile)

    async def assign(self, profile: str, run_id: str, embedding: np.ndarray[Any, Any], tokens: list[str]) -> str | None:
        """Join the nearest cluster if within its radius (the 95th percentile of its members' distances)."""
        space_row = await self.space(profile)
        if space_row is None or space_row.embedder != self.embedder.name:
            return None
        vector = Space(list(space_row.vocabulary), space_row.weight).vector(embedding, tokens)
        async with self.store.read() as s:
            clusters = list(
                (
                    await s.execute(
                        select(Cluster).where(
                            Cluster.profile == profile, Cluster.status != "retired", Cluster.centroid.is_not(None)
                        )
                    )
                ).scalars()
            )
        centres = [(c.id, from_bytes(c.centroid), c.radius or 0.0) for c in clusters if c.centroid]
        cluster_id = assign(vector, centres)
        if cluster_id is None:
            return None

        async def op(session: AsyncSession) -> None:
            await session.execute(update(Failure).where(Failure.run_id == run_id).values(cluster_id=cluster_id))
            await session.execute(
                update(Cluster).where(Cluster.id == cluster_id).values(size=Cluster.size + 1, updated_ms=now_ms())
            )

        await self.store.write(op)
        self.services.bus.publish("cluster.updated", cluster_id=cluster_id, profile=profile)
        return cluster_id

    async def unclustered(self, profile: str) -> int:
        async with self.store.read() as s:
            rows = (
                await s.execute(select(Failure.run_id).where(Failure.profile == profile, Failure.cluster_id.is_(None)))
            ).all()
        return len(rows)

    # Re-clustering --------------------------------------------------------------------------------------------------

    async def recluster(self, profile: str, *, name: bool = True) -> dict[str, Any]:
        async with self.store.read() as s:
            failures = list(
                (
                    await s.execute(select(Failure).where(Failure.profile == profile, Failure.embedding.is_not(None)))
                ).scalars()
            )
            old_clusters = {
                c.id: c
                for c in (
                    await s.execute(select(Cluster).where(Cluster.profile == profile, Cluster.status != "retired"))
                ).scalars()
            }
        failures = [f for f in failures if (f.signature or {}).get("embedder") == self.embedder.name]
        if not failures:
            return {"profile": profile, "failures": 0, "clusters": 0}
        token_lists = [tokens_of(f.signature, f.category) for f in failures]
        space = Space.build(token_lists, self.config.onehot_weight)
        vectors = np.stack(
            [
                space.vector(from_bytes(f.embedding or b""), tokens)
                for f, tokens in zip(failures, token_lists, strict=True)
            ]
        )
        labels = cluster_labels(vectors, self.config.min_cluster_size)
        groups: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            if label >= 0:
                groups.setdefault(label, []).append(index)
        new_groups = [sorted(groups[label]) for label in sorted(groups)]
        new_members = [{failures[i].run_id for i in group} for group in new_groups]
        old_members = {cid: {f.run_id for f in failures if f.cluster_id == cid} for cid in old_clusters}
        inherited = match_ids(old_members, new_members)
        now = now_ms()
        rows: list[dict[str, Any]] = []
        assignments: dict[str, str | None] = {f.run_id: None for f in failures}
        named = 0
        for group, members, old_id in zip(new_groups, new_members, inherited, strict=True):
            centre, radius = centre_and_radius(vectors[group])
            cluster_id = old_id or new_id()
            previous = old_clusters.get(old_id) if old_id else None
            needs_name = (
                previous is None
                or previous.title is None
                or 1 - jaccard(members, set(previous.named_members or [])) > self.config.rename_change
            )
            naming: Naming | None = None
            if name and needs_name:
                naming = await self._name(profile, [failures[i] for i in group], vectors[group], centre)
                named += 1
            for run_id in members:
                assignments[run_id] = cluster_id
            row: dict[str, Any] = {
                "id": cluster_id,
                "profile": profile,
                "centroid": to_bytes(centre),
                "radius": radius,
                "size": len(members),
                "updated_ms": now,
                "status": previous.status if previous is not None else "new",
                "created_ms": previous.created_ms if previous is not None else now,
                "title": previous.title if previous is not None else None,
                "likely_cause": previous.likely_cause if previous is not None else None,
                "suggested_fix": previous.suggested_fix if previous is not None else None,
                "evidence": previous.evidence if previous is not None else [],
                "cause_status": previous.cause_status if previous is not None else None,
                "named_members": previous.named_members if previous is not None else [],
            }
            if naming is not None:
                row.update(
                    title=naming.title,
                    likely_cause=naming.likely_cause,
                    suggested_fix=naming.suggested_fix,
                    evidence=naming.evidence,
                    cause_status=naming.status,
                    named_members=sorted(members),
                )
            rows.append(row)
        kept = {row["id"] for row in rows}

        async def op(session: AsyncSession) -> None:
            for row in rows:
                await session.execute(
                    insert(Cluster).values(**row).on_conflict_do_update(index_elements=["id"], set_=row)
                )
            for cid in old_clusters:
                if cid not in kept:
                    await session.execute(
                        update(Cluster).where(Cluster.id == cid).values(status="retired", size=0, updated_ms=now)
                    )
            for run_id, cluster_id in assignments.items():
                await session.execute(update(Failure).where(Failure.run_id == run_id).values(cluster_id=cluster_id))
            space_values = {
                "profile": profile,
                "vocabulary": space.vocabulary,
                "weight": space.weight,
                "embedder": self.embedder.name,
                "dimensions": int(vectors.shape[1]),
                "clustered_ms": now,
                "stats": {"failures": len(failures), "clusters": len(rows), "unclustered": labels.count(-1)},
            }
            await session.execute(
                insert(ClusterSpace)
                .values(**space_values)
                .on_conflict_do_update(index_elements=["profile"], set_=space_values)
            )

        await self.store.write(op)
        self.services.bus.publish("cluster.reclustered", profile=profile)
        return {
            "profile": profile,
            "failures": len(failures),
            "clusters": len(rows),
            "unclustered": labels.count(-1),
            "named": named,
            "kept_ids": sum(1 for i in inherited if i is not None),
        }

    async def _name(
        self, profile: str, members: list[Failure], vectors: np.ndarray[Any, Any], centre: np.ndarray[Any, Any]
    ) -> Naming:
        from blackbox.llm.client import LLMUnavailable

        distances = np.linalg.norm(vectors - centre, axis=1)
        closest = [members[i] for i in np.argsort(distances)[:5]]
        runs = {r.id: r for r in await self._runs([m.run_id for m in members])}
        shown = [
            Member(
                run_id=f.run_id,
                label=f"R{i}",
                description=f.description or "; ".join(f.signature.get("reasons") or []),
                digest=str(f.signature.get("digest") or "")[:2500],
                steps=runs[f.run_id].step_count if f.run_id in runs else 0,
            )
            for i, f in enumerate(closest, start=1)
        ]
        stats = await self.stats(members, runs)
        try:
            return await name_cluster(self.services.llm, shown, stats["text"], len(members))
        except LLMUnavailable as exc:
            log.warning("naming a %s cluster failed: %s", profile, exc)
            return Naming(None, None, None, status="invalid")

    async def _runs(self, run_ids: list[str]) -> list[Run]:
        async with self.store.read() as s:
            return list((await s.execute(select(Run).where(Run.id.in_(run_ids)))).scalars())

    async def stats(self, members: list[Failure], runs: dict[str, Run]) -> dict[str, Any]:
        n = len(members)

        def shares(values: list[str]) -> str:
            counts = Counter(values)
            return ", ".join(f"{share * 100 / n:.0f}% {value}" for value, share in counts.most_common(5))

        endings = [str(runs[m.run_id].ending if m.run_id in runs else "none") for m in members]
        categories = [m.category or "undescribed" for m in members]
        flags = [flag for m in members for flag in (m.signature.get("flags") or [])]
        nodes = [str(m.signature.get("first_problem_node")) for m in members if m.signature.get("first_problem_node")]
        started = [r.started_ms for r in runs.values() if r.started_ms]
        lines = [f"endings: {shares(endings)}", f"categories: {shares(categories)}"]
        if flags:
            lines.append(f"metric flags: {', '.join(f'{c} runs {f}' for f, c in Counter(flags).most_common(5))}")
        if nodes:
            lines.append(
                f"first problem at node: {', '.join(f'{c} x {node}' for node, c in Counter(nodes).most_common(3))}"
            )
        async with self.store.read() as s:
            scores = list(
                (
                    await s.execute(
                        select(Score).where(
                            Score.run_id.in_([m.run_id for m in members]),
                            Score.kind == "metric",
                            Score.name.in_(("guardrail_score", "steps")),
                        )
                    )
                ).scalars()
            )
        for metric in ("guardrail_score", "steps"):
            values = [sc.value for sc in scores if sc.name == metric and sc.value is not None]
            if values:
                lines.append(f"{metric}: {min(values):g} to {max(values):g}")
        if started:
            from blackbox.web.format import fmt_time

            lines.append(f"runs from {fmt_time(min(started))} to {fmt_time(max(started))} UTC")
        return {"text": "\n".join(lines), "endings": Counter(endings), "categories": Counter(categories)}


async def failure_and_cluster(services: Services, run_id: str) -> tuple[Failure | None, Cluster | None]:
    async with services.store.read() as s:
        failure = await s.get(Failure, run_id)
        cluster = await s.get(Cluster, failure.cluster_id) if failure is not None and failure.cluster_id else None
    return failure, cluster


async def clear_profile(services: Services, profile: str) -> None:
    """Forget a profile's clusters (tests and `blackbox cluster --reset`)."""

    async def op(session: AsyncSession) -> None:
        await session.execute(update(Failure).where(Failure.profile == profile).values(cluster_id=None))
        await session.execute(delete(Cluster).where(Cluster.profile == profile))
        await session.execute(delete(ClusterSpace).where(ClusterSpace.profile == profile))

    await services.store.write(op)
