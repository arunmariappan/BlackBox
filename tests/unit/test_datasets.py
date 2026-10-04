from pathlib import Path

import pytest

from blackbox.datasets import load_dataset

ROOT = Path(__file__).parent.parent.parent


def test_paperpilot_question_set_is_valid() -> None:
    dataset = load_dataset(ROOT / "datasets" / "paperpilot" / "questions.yaml")
    assert dataset.profile == "paperpilot"
    groups = {q.group for q in dataset.questions}
    assert groups <= {"answerable", "out_of_scope", "not_in_corpus"}
    out_of_scope = [q for q in dataset.questions if q.group == "out_of_scope"]
    assert len(out_of_scope) == 10 and all(q.expected_ending == "out_of_scope" for q in out_of_scope)
    assert all(
        q.status == "draft" for q in dataset.questions if q.group == "not_in_corpus"
    )  # unchecked against the index
    assert len(dataset.reviewed()) == 10


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text(
        "name: x\nprofile: p\nquestions:\n"
        "  - {id: a, group: out_of_scope, question: q, expected_ending: out_of_scope}\n"
        "  - {id: a, group: out_of_scope, question: r, expected_ending: out_of_scope}\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)
