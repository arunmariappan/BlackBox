"""Judges: a prompt file with front matter, an input builder and an output schema.

```markdown
---
name: pp_faithfulness
profile: paperpilot
applies_when: "ending == 'answered'"
output: FaithfulnessVerdict          # Pydantic model; its JSON schema goes to Ollama's `format`
input: pp_faithfulness               # input builder (its version is part of the judge's version)
label_question: faithful             # the label question its verdict is compared with
---
You check whether an answer is supported by ... {question} ... {context} ... {answer}
```

A judge's **version** is the first 12 hex characters of a SHA-256 over the prompt text, model, options, output schema
and input builder version: any change makes a new version, and every score names its version.
"""

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from blackbox.judges.inputs import BUILDERS, InputBuilder, Inputs
from blackbox.judges.schemas import SCHEMAS
from blackbox.store.models import Run
from blackbox.util import canonical_json, sha256_hex

PROMPTS_DIR = Path(__file__).parent / "prompts"
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


class JudgeError(Exception):
    pass


@dataclass
class Judge:
    name: str
    profile: str
    template: str
    output: type[BaseModel]
    input_builder: InputBuilder
    model: str
    applies_when: str | None = None
    label_question: str | None = None
    label_rule: str = "verdict"  # verdict | scope_decision
    agreement_source: str = "labels"  # labels | checker
    options: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def samples(self) -> int:
        return int(self.options.get("samples", 1))

    @property
    def llm_options(self) -> dict[str, Any]:
        return {k: v for k, v in self.options.items() if k not in ("samples",)}

    @property
    def prompt_hash(self) -> str:
        return sha256_hex(self.template.encode())[:12]

    @property
    def version(self) -> str:
        material = canonical_json(
            {
                "prompt": self.template,
                "model": self.model,
                "options": self.options,
                "schema": self.output.model_json_schema(),
                "input": self.input_builder.ref,
            }
        )
        return sha256_hex(material.encode())[:12]

    def applies(self, run: Run) -> bool:
        if run.profile != self.profile:
            return False
        if not self.applies_when:
            return True
        names = {"ending": run.ending, "profile": run.profile, "source": run.source, "status": run.status}
        return bool(safe_eval(self.applies_when, names))

    def render(self, inputs: Inputs) -> str:
        return _PLACEHOLDER.sub(lambda m: inputs.get(m.group(1), m.group(0)), self.template)

    def messages(self, inputs: Inputs) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.render(inputs)}]

    def to_label(self, parsed: BaseModel, run: Run) -> str:
        """The judge's answer to its label question: `pass` or `fail`."""
        if self.label_rule == "scope_decision":
            should_answer = getattr(parsed, "should_answer", None) == "yes"
            refused = run.ending == "out_of_scope"
            return "pass" if should_answer != refused else "fail"
        verdict = getattr(parsed, "verdict", None)
        if verdict not in ("pass", "fail"):
            raise JudgeError(f"{self.name}: output has no pass/fail verdict")
        return str(verdict)


_ALLOWED = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Tuple,
    ast.List,
)


def safe_eval(expression: str, names: dict[str, Any]) -> Any:
    """Evaluate a small condition (`ending == 'answered'`, `ending in ('a', 'b') and source != 'replay'`)."""
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise JudgeError(f"applies_when may only compare run fields; {type(node).__name__} isn't allowed")
        if isinstance(node, ast.Name) and node.id not in names:
            raise JudgeError(f"applies_when: unknown name {node.id!r} (use {', '.join(sorted(names))})")
    return eval(compile(tree, "<applies_when>", "eval"), {"__builtins__": {}}, dict(names))


def parse_prompt_file(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        raise JudgeError("a judge file starts with front matter between --- lines")
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front) or {}
    if not isinstance(meta, dict):
        raise JudgeError("front matter must be a mapping")
    return meta, body.strip() + "\n"


def judge_from_file(path: Path, model: str) -> Judge:
    meta, template = parse_prompt_file(path.read_text(encoding="utf-8"))
    try:
        output = SCHEMAS[str(meta["output"])]
    except KeyError:
        raise JudgeError(f"{path.name}: unknown output schema {meta.get('output')!r}") from None
    builder_name = str(meta.get("input", meta["name"]))
    if builder_name not in BUILDERS:
        raise JudgeError(f"{path.name}: unknown input builder {builder_name!r}")
    return Judge(
        name=str(meta["name"]),
        profile=str(meta["profile"]),
        template=template,
        output=output,
        input_builder=BUILDERS[builder_name],
        model=str(meta.get("model", model)),
        applies_when=meta.get("applies_when"),
        label_question=meta.get("label_question"),
        label_rule=str(meta.get("label_rule", "verdict")),
        agreement_source=str(meta.get("agreement_source", "labels")),
        options=dict(meta.get("options") or {}),
        path=path,
    )


def load_judges(model: str, directory: Path = PROMPTS_DIR) -> dict[str, Judge]:
    judges = {}
    for path in sorted(directory.glob("*.md")):
        judge = judge_from_file(path, model)
        judges[judge.name] = judge
    return judges


def input_hash(inputs: Inputs) -> str:
    return sha256_hex(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode())
