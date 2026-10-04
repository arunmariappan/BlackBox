"""The `Profile` base class: everything BlackBox knows about one agent."""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from blackbox.otlp.decode import SpanData
from blackbox.runs.context import ExchangeData, RunContext, StepDraft

if TYPE_CHECKING:
    from blackbox.proxy.matching import Normaliser
    from blackbox.proxy.sessions import ReplaySession
    from blackbox.proxy.views import ExchangeView
    from blackbox.services import Services


@dataclass
class StartRequest:
    """The HTTP request that starts a run of the agent."""

    method: str
    url: str
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 600

    def envelope(self) -> dict[str, Any]:
        """What is stored as the run's entry request (headers that carry ids are left out)."""
        return {"method": self.method, "url": self.url, "headers": self.headers, "body": self.body}

    @classmethod
    def from_envelope(cls, envelope: dict[str, Any], timeout_seconds: float = 600) -> StartRequest:
        return cls(
            method=envelope["method"],
            url=envelope["url"],
            body=envelope.get("body"),
            headers=dict(envelope.get("headers") or {}),
            timeout_seconds=timeout_seconds,
        )


class Profile:
    """Subclasses describe one agent. Every hook has a safe default, so a profile only overrides what it knows."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    node_spans: ClassVar[frozenset[str]] = frozenset()
    failure_endings: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options: dict[str, Any] = dict(options or {})

    # Matching and structure -------------------------------------------------------------------------------------

    def matches(self, spans: Sequence[SpanData]) -> bool:
        """Whether a trace belongs to this agent (used for runs BlackBox didn't start)."""
        return False

    def node_for(self, step: StepDraft, chain: Iterable[SpanData], ctx: RunContext) -> str | None:
        """The agent node a step belongs to: by default the nearest ancestor span named in `node_spans`."""
        for span in chain:
            if span.name in self.node_spans:
                return span.name
        return None

    def exchange_view(self, exchange: ExchangeData, ctx: RunContext) -> ExchangeView | None:
        """A view of an exchange this profile understands better than the generic parsers (None: use them)."""
        return None

    # Starting runs --------------------------------------------------------------------------------------------------

    def parse_input(self, text: str, **options: Any) -> dict[str, Any]:
        """Turn command-line text into the input this agent takes."""
        return {"input": text, **options}

    def build_request(self, run_input: dict[str, Any]) -> StartRequest:
        raise NotImplementedError(f"profile {self.name!r} cannot start runs")

    # Reading runs ---------------------------------------------------------------------------------------------------

    def input_text(self, ctx: RunContext) -> str | None:
        body = ctx.entry_body
        if isinstance(body, dict):
            for key in ("query", "instruction", "input", "question"):
                if isinstance(body.get(key), str):
                    return str(body[key])
        root = ctx.root
        if root is not None:
            value = root.attributes.get("blackbox.run.input")
            if value is not None:
                return value if isinstance(value, str) else json.dumps(value)
        return None

    def read_output(self, ctx: RunContext) -> Any:
        """The run's output: the HTTP response when BlackBox started it, otherwise whatever the spans say."""
        if ctx.output_body is not None:
            parsed = ctx.output_json
            return parsed if parsed is not None else ctx.output_body.decode("utf-8", errors="replace")
        root = ctx.root
        if root is not None and "blackbox.run.output" in root.attributes:
            value = root.attributes["blackbox.run.output"]
            try:
                return json.loads(value) if isinstance(value, str) else value
            except ValueError:
                return value
        return None

    def output_text(self, output: Any) -> str | None:
        if output is None:
            return None
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            for key in ("answer", "final_answer", "output", "text"):
                if isinstance(output.get(key), str):
                    return str(output[key])
        return json.dumps(output, ensure_ascii=False)

    def ending(self, ctx: RunContext, output: Any) -> str | None:
        root = ctx.root
        if root is not None and isinstance(root.attributes.get("blackbox.run.ending"), str):
            return str(root.attributes["blackbox.run.ending"])
        if root is not None and root.status_code == "error":
            return "error"
        return None

    # Replay ---------------------------------------------------------------------------------------------------------

    def normalisers(self, upstream: str) -> list[Normaliser]:
        """Request normalisers for matching on `upstream` (none by default: exact means exact).

        Profiles may also take them from options: `normalisers = {ollama = [{mask = "<regex>", with = "<ts>"},
        {drop = "$.request_id"}]}`.
        """
        from blackbox.proxy.matching import DropJsonPath, MaskRegex

        out: list[Normaliser] = []
        for item in (self.options.get("normalisers") or {}).get(upstream, []):
            if "mask" in item:
                out.append(MaskRegex(item["mask"], item.get("with", "<masked>")))
            elif "drop" in item:
                out.append(DropJsonPath(item["drop"]))
        return out

    def rebuild_input(self, ctx: RunContext) -> StartRequest | None:
        """The entry request of a run BlackBox didn't start, rebuilt from its spans (None: can't)."""
        return None

    async def prepare(self, session: ReplaySession, services: Services) -> None:
        """Get ready for a replay or fork (OpsDesk creates the session's own sandbox here)."""
        return None

    def replay_request(self, request: StartRequest, session: ReplaySession) -> StartRequest:
        """The entry request a replay sends (OpsDesk points it at the session's sandbox)."""
        return request

    def compare_outputs(self, source: Any, replay: Any) -> dict[str, Any]:
        """Compare a source run's output with its replay's. Ignores `trace_id`, which differs by design."""

        def strip(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: strip(v) for k, v in value.items() if k != "trace_id"}
            if isinstance(value, list):
                return [strip(v) for v in value]
            return value

        a, b = strip(source), strip(replay)
        fields: dict[str, Any] = {}
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                fields[key] = {"equal": a.get(key) == b.get(key), "source": a.get(key), "replay": b.get(key)}
        return {"equal": a == b, "fields": fields}
