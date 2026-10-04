"""Profiles for the fake agents tests use."""

from collections.abc import Sequence
from typing import Any

from blackbox.otlp.decode import SpanData
from blackbox.profiles.base import Profile, StartRequest


class FakeAgentProfile(Profile):
    name = "fake-agent"
    description = "the fake agent tests run in-process"
    node_spans = frozenset({"plan", "act", "answer"})

    def __init__(self, base_url: str = "http://127.0.0.1:1", **options: Any) -> None:
        super().__init__({"base_url": base_url, **options})

    def matches(self, spans: Sequence[SpanData]) -> bool:
        return any(span.name == "invoke_agent fake-agent" for span in spans)

    def build_request(self, run_input: dict[str, Any]) -> StartRequest:
        return StartRequest("POST", f"{self.options['base_url']}/run", body=run_input, timeout_seconds=30)
