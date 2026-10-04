"""Failure descriptions: `qwen3.5:4b` reads a run's digest and says what went wrong, where, and in which category."""

from typing import Literal

from pydantic import BaseModel, Field

from blackbox.llm.client import OllamaJSON

CATEGORIES = (
    "wrong_tool",
    "bad_arguments",
    "loop",
    "gave_up_after_error",
    "policy_violation",
    "unsupported_claim",
    "premature_finish",
    "refused_in_scope",
    "retrieval_miss",
    "format_error",
    "infrastructure",
    "other",
)

Category = Literal[
    "wrong_tool",
    "bad_arguments",
    "loop",
    "gave_up_after_error",
    "policy_violation",
    "unsupported_claim",
    "premature_finish",
    "refused_in_scope",
    "retrieval_miss",
    "format_error",
    "infrastructure",
    "other",
]


class FailureDescription(BaseModel):
    what_went_wrong: str = Field(description="One or two sentences: what the agent did wrong or what failed.")
    where_step: int | None = Field(description="The step number (#n) where the problem first shows, or null.")
    category: Category


PROMPT = """You analyse a failed run of an AI agent. Read the run below and say what went wrong.

Categories:
- wrong_tool: called a tool that doesn't fit, or changed the wrong thing (a symptom instead of the cause)
- bad_arguments: called the right tool with wrong or invalid arguments
- loop: repeated the same calls without making progress
- gave_up_after_error: a tool failed and the agent stopped instead of retrying or using an alternative
- policy_violation: broke a rule it was given (tickets, approvals, MFA, priorities)
- unsupported_claim: the answer states things the sources don't support
- premature_finish: finished before the task was done
- refused_in_scope: refused or declined a request it should have handled
- retrieval_miss: search found nothing useful, so the answer couldn't be given
- format_error: produced output in the wrong format
- infrastructure: a service, network or timeout problem outside the agent
- other: none of these

Be specific: name the tool, service, step or claim. where_step must be one of the step numbers shown (#n).

{digest}
"""


async def describe(llm: OllamaJSON, digest_text: str, steps: int) -> tuple[FailureDescription | None, str | None]:
    """A validated description, or (None, error). A `where_step` that isn't a step of the run is retried once,
    then the description is kept without it."""
    messages = [{"role": "user", "content": PROMPT.replace("{digest}", digest_text)}]
    result = await llm.call(messages, FailureDescription)
    if not result.valid or result.parsed is None:
        return None, result.error or "invalid reply"
    parsed = result.parsed
    if parsed.where_step is not None and not 1 <= parsed.where_step <= steps:
        retry_messages = [
            *messages,
            {"role": "assistant", "content": result.raw},
            {
                "role": "user",
                "content": f"Step {parsed.where_step} doesn't exist; the steps are #1 to #{steps}. Answer again.",
            },
        ]
        second = await llm.call(retry_messages, FailureDescription, retries=0)
        if (
            second.valid
            and second.parsed is not None
            and (second.parsed.where_step is None or 1 <= second.parsed.where_step <= steps)
        ):
            return second.parsed, None
        return parsed.model_copy(update={"where_step": None}), None
    return parsed, None
