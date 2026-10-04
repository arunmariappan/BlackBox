"""The blind label page and the judges page."""

from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from blackbox.store.models import Label, Score
from blackbox.util import new_id, now_ms
from tests.fixtures import paperpilot
from tests.harness import running_blackbox


async def test_label_reveals_the_verdict_only_after_saving(tmp_path: Path, migrated_db: Path) -> None:
    async with running_blackbox(tmp_path, migrated_db) as bb:
        store = bb.services.store
        run_id = await paperpilot.insert_run(store)
        judge = bb.services.judges["pp_faithfulness"]

        async def op(session: AsyncSession) -> None:
            session.add(
                Score(
                    id=new_id(),
                    run_id=run_id,
                    kind="judge",
                    name=judge.name,
                    version=judge.version,
                    value=1.0,
                    label="pass",
                    rationale="UNIQUE-RATIONALE-123",
                    details={"parsed": {"unsupported_claims": [], "verdict": "pass"}},
                    created_ms=now_ms(),
                )
            )

        await store.write(op)
        async with httpx.AsyncClient(base_url=bb.base_url) as client:
            page = await client.get("/label?question=faithful")
            assert page.status_code == 200
            assert run_id in page.text and paperpilot.ANSWER[:40] in page.text
            assert "UNIQUE-RATIONALE-123" not in page.text  # blind until saved
            assert "Attention Is All You Need" in page.text  # the excerpts
            saved = await client.post(
                "/label",
                data={"run_id": run_id, "question": "faithful", "value": "fail"},
                headers={"hx-request": "true"},
            )
            assert saved.status_code == 200
            assert "UNIQUE-RATIONALE-123" in saved.text and "disagrees with you" in saved.text
            labels = await store.reader.labels(run_id)
            assert [(lb.question, lb.value) for lb in labels] == [("faithful", "fail")]
            await client.post(
                "/label", data={"run_id": run_id, "question": "faithful", "value": "pass", "note": "close call"}
            )
            labels = await store.reader.labels(run_id)
            assert [(lb.value, lb.note) for lb in labels] == [("pass", "close call")]  # replaced, not added
            done = await client.get("/label?question=faithful")
            assert "Nothing left to label" in done.text
            note = await client.get(f"/label/note?run={run_id}&question=faithful")
            assert "autofocus" in note.text and "close call" in note.text
            assert (await client.get("/label?question=nope")).status_code == 404


async def test_judges_page_and_calibration_without_a_model(tmp_path: Path, migrated_db: Path) -> None:
    async with running_blackbox(tmp_path, migrated_db) as bb:
        store = bb.services.store
        run_id = await paperpilot.insert_run(store)
        judge = bb.services.judges["pp_relevance"]
        await bb.services.judge_runner.register(judge)

        async def op(session: AsyncSession) -> None:
            session.add(
                Score(
                    id=new_id(),
                    run_id=run_id,
                    kind="judge",
                    name=judge.name,
                    version=judge.version,
                    value=1.0,
                    label="pass",
                    details={},
                    created_ms=now_ms(),
                )
            )
            session.add(Label(id=new_id(), run_id=run_id, question="relevant", value="pass", created_ms=now_ms()))

        await store.write(op)
        async with httpx.AsyncClient(base_url=bb.base_url, follow_redirects=False) as client:
            assert (await client.get("/judges")).status_code == 200
            # Ollama isn't reachable in tests: calibration still refreshes agreement from the scores it has.
            response = await client.post(f"/judges/{judge.name}/calibrate")
            assert response.status_code == 303
            page = await client.get(f"/judges/{judge.name}")
            assert page.status_code == 200 and judge.version in page.text and "untrusted" in page.text
            api = (await client.get("/api/judges")).json()
            relevance = next(j for j in api if j["name"] == "pp_relevance")
            [version] = relevance["versions"]
            assert version["agreement"]["all"]["n"] == 1 and version["trusted"] is False
            runs_page = await client.get("/runs")
            assert "pp_relevance: pass" in runs_page.text and "dim" in runs_page.text
            run_page = await client.get(f"/runs/{run_id}")
            assert "Verdicts" in run_page.text and "untrusted" in run_page.text
