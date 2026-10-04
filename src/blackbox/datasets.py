"""Question sets for agents (PaperPilot's lives in `datasets/paperpilot/questions.yaml`)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DatasetQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    group: Literal["answerable", "out_of_scope", "not_in_corpus"]
    question: str
    expected_ending: Literal["answered", "out_of_scope", "max_attempts"]
    must_cite: bool = False
    paper_id: str | None = None  # answerable questions: the paper the question was drafted from
    status: Literal["draft", "reviewed"] = "reviewed"
    note: str | None = None


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    profile: str
    description: str = ""
    questions: list[DatasetQuestion] = Field(default_factory=list)

    def reviewed(self) -> list[DatasetQuestion]:
        return [q for q in self.questions if q.status == "reviewed"]


def load_dataset(path: Path) -> Dataset:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    dataset = Dataset.model_validate(data)
    ids = [q.id for q in dataset.questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"{path}: duplicate question ids {sorted(duplicates)}")
    return dataset


class DraftQuestion(BaseModel):
    rationale: str = Field(description="What the excerpt is about, in one sentence.")
    question: str = Field(description="A question a researcher might ask that this excerpt answers.")
    answerable_from_excerpt: bool
