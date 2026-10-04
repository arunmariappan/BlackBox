"""Request patches: edit a request before it is matched, so a changed prompt goes live, as a code change would.

```yaml
patches:
  - name: strict-guardrail
    match:
      upstream: ollama
      tape_node: guardrail_validation       # replay only: the node of the tape step this request lines up with
      # step: 1                             # replay only: a step number
      # content_regex: "relevance of the question"   # any mode, also live traffic
    edit:
      path: "$.messages[?(@.role == 'system')].content"
      replace: { find: "Score from 0 to 100", with: "Be very strict. Score from 0 to 100" }
      # set: "a whole new text"
```

Every condition given in `match` must hold. `tape_node` and `step` need a replay, where a request can be lined up with
its tape step; live traffic can only match on content.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonpath_ng.ext import parse as parse_jsonpath
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PatchMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upstream: str | None = None
    tape_node: str | None = None
    step: int | None = Field(default=None, ge=1)
    content_regex: str | None = None

    @field_validator("content_regex")
    @classmethod
    def _valid_regex(cls, value: str | None) -> str | None:
        if value is not None:
            re.compile(value)
        return value


class Replace(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    find: str
    with_: str = Field(alias="with")


class PatchEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    replace: Replace | None = None
    set: Any = None

    @field_validator("path")
    @classmethod
    def _valid_path(cls, value: str) -> str:
        parse_jsonpath(value)
        return value

    @model_validator(mode="after")
    def _one_edit(self) -> PatchEdit:
        has_set = "set" in self.model_fields_set
        if (self.replace is None) == (not has_set):
            raise ValueError("an edit needs exactly one of `replace` or `set`")
        return self


class Patch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    match: PatchMatch = Field(default_factory=PatchMatch)
    edit: PatchEdit

    @property
    def needs_tape(self) -> bool:
        return self.match.tape_node is not None or self.match.step is not None


class PatchFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patches: list[Patch]


def load_patches(source: str | Path | dict[str, Any] | list[Any]) -> list[Patch]:
    """Patches from a YAML file path, YAML text, or already-parsed data."""
    if isinstance(source, Path):
        source = source.read_text(encoding="utf-8")
    data = yaml.safe_load(source) if isinstance(source, str) else source
    if isinstance(data, list):
        data = {"patches": data}
    return PatchFile.model_validate(data).patches


@dataclass(frozen=True)
class LineUp:
    """The tape step a replayed request lines up with."""

    step: int
    node: str | None


def matches(patch: Patch, upstream: str, body_text: str, lineup: LineUp | None) -> bool:
    condition = patch.match
    if condition.upstream is not None and condition.upstream != upstream:
        return False
    if condition.step is not None and (lineup is None or lineup.step != condition.step):
        return False
    if condition.tape_node is not None and (lineup is None or lineup.node != condition.tape_node):
        return False
    return not (condition.content_regex is not None and re.search(condition.content_regex, body_text) is None)


def apply_edit(body: Any, edit: PatchEdit) -> bool:
    """Apply one edit in place; True if anything changed. A `set` on a path that doesn't exist yet creates it."""
    changed = False
    expression = parse_jsonpath(edit.path)
    found_any = expression.find(body)
    if not found_any and edit.replace is None:
        before = json.dumps(body, sort_keys=True)
        expression.update_or_create(body, edit.set)
        return json.dumps(body, sort_keys=True) != before
    for found in found_any:
        old = found.value
        if edit.replace is not None:
            new = old.replace(edit.replace.find, edit.replace.with_) if isinstance(old, str) else old
        else:
            new = edit.set
        if new != old:
            found.full_path.update(body, new)
            changed = True
    return changed


def apply_patches(
    patches: list[Patch], upstream: str, body: bytes, lineup: LineUp | None = None
) -> tuple[bytes, list[str]]:
    """The patched body and the names of the patches that changed it (the original bytes when none did)."""
    if not patches or not body:
        return body, []
    text = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except ValueError:
        return body, []
    applied = []
    for patch in patches:
        if matches(patch, upstream, text, lineup) and apply_edit(parsed, patch.edit):
            applied.append(patch.name)
    if not applied:
        return body, []
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), applied
