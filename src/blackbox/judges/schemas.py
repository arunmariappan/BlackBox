"""Judge output schemas. Reasoning fields come first: the model writes fields in schema order, so it reasons before
it decides."""

from typing import Literal

from pydantic import BaseModel, Field


class FaithfulnessVerdict(BaseModel):
    unsupported_claims: list[str] = Field(
        description="Each factual claim in the answer that the excerpts do not support, quoted or closely paraphrased."
    )
    rationale: str = Field(description="One or two sentences explaining the verdict.")
    verdict: Literal["pass", "fail"] = Field(description="pass if every factual claim is supported by the excerpts.")


class RelevanceVerdict(BaseModel):
    rationale: str = Field(description="One or two sentences explaining the verdict.")
    verdict: Literal["pass", "fail"] = Field(description="pass if the answer addresses the question that was asked.")


class ScopeVerdict(BaseModel):
    rationale: str = Field(description="One sentence on what the question is about.")
    should_answer: Literal["yes", "no"] = Field(
        description="yes if an assistant for AI research papers should answer this question."
    )


class TaskSuccessVerdict(BaseModel):
    problems: list[str] = Field(description="Each thing the agent did wrong, missed or did against a policy.")
    rationale: str = Field(description="One or two sentences explaining the verdict.")
    verdict: Literal["pass", "fail"] = Field(
        description="pass if the task was done correctly and every policy was followed."
    )


class PairwisePreference(BaseModel):
    rationale: str
    preferred: Literal["A", "B", "tie"]


SCHEMAS: dict[str, type[BaseModel]] = {
    cls.__name__: cls
    for cls in (FaithfulnessVerdict, RelevanceVerdict, ScopeVerdict, TaskSuccessVerdict, PairwisePreference)
}
