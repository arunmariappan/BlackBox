import json

import pytest
from pydantic import ValidationError

from blackbox.proxy.matching import ISO_TIMESTAMP, DropJsonPath, MaskRegex, json_diff, match_key, request_diff
from blackbox.proxy.sessions import split_chunks, tape_headers
from blackbox.replay.patches import LineUp, apply_patches, load_patches


def body(value: object) -> bytes:
    return json.dumps(value).encode()


def test_key_ignores_key_order_and_query_order() -> None:
    a = match_key("post", "/api/chat", "b=2&a=1", b'{"model":"m","messages":[1,2]}')
    b = match_key("POST", "/api/chat", "a=1&b=2", b'{"messages":[1,2],"model":"m"}')
    assert a == b
    assert a != match_key("POST", "/api/chat", "a=1&b=2", b'{"messages":[2,1],"model":"m"}')
    assert a != match_key("POST", "/api/generate", "a=1&b=2", b'{"messages":[1,2],"model":"m"}')


def test_mask_regex_normaliser() -> None:
    first = body({"messages": [{"role": "system", "content": "Now: 2026-10-04T10:00:00.123+00:00"}]})
    second = body({"messages": [{"role": "system", "content": "Now: 2026-10-05T11:30:59Z"}]})
    assert match_key("POST", "/x", "", first) != match_key("POST", "/x", "", second)
    assert match_key("POST", "/x", "", first, [ISO_TIMESTAMP]) == match_key("POST", "/x", "", second, [ISO_TIMESTAMP])
    custom = MaskRegex(r"req-\d+", "<id>")
    assert match_key("GET", "/x", "", b'"req-1"', [custom]) == match_key("GET", "/x", "", b'"req-2"', [custom])


def test_drop_json_path_normaliser() -> None:
    drop = [DropJsonPath("$.request_id")]
    a = body({"q": "x", "request_id": "a1"})
    b = body({"q": "x", "request_id": "b2"})
    assert match_key("POST", "/s", "", a, drop) == match_key("POST", "/s", "", b, drop)
    assert match_key("POST", "/s", "", a, drop) != match_key(
        "POST", "/s", "", body({"q": "y", "request_id": "a1"}), drop
    )


def test_message_diff() -> None:
    tape = body({"model": "m", "messages": [{"role": "system", "content": "A\nB"}, {"role": "user", "content": "q"}]})
    actual = body({"model": "m", "messages": [{"role": "system", "content": "A\nC"}, {"role": "user", "content": "q"}]})
    diff = request_diff(tape, actual)
    assert diff["kind"] == "messages"
    [change] = diff["messages"]
    assert change["index"] == 0 and change["role"] == "system" and change["change"] == "changed"
    assert "-B" in change["diff"] and "+C" in change["diff"]
    added = request_diff(
        tape, body({"model": "n", "messages": [*json.loads(tape)["messages"], {"role": "user", "content": "more"}]})
    )
    assert [m["change"] for m in added["messages"]] == ["added", "fields"]


def test_json_diff() -> None:
    assert json_diff({"a": 1, "b": [1, 2]}, {"a": 2, "b": [1, 3], "c": True}) == [
        {"path": "$.a", "tape": 1, "request": 2, "missing": None},
        {"path": "$.b[1]", "tape": 2, "request": 3, "missing": None},
        {"path": "$.c", "tape": None, "request": True, "missing": "tape"},
    ]
    assert request_diff(b'{"size": 3}', b'{"size": 5}')["kind"] == "json"


PATCHES = """
patches:
  - name: strict-guardrail
    match:
      upstream: ollama
      tape_node: guardrail_validation
    edit:
      path: "$.messages[?(@.role == 'system')].content"
      replace: { find: "Score from 0 to 100", with: "Be very strict. Score from 0 to 100" }
  - name: second-step
    match: { step: 2 }
    edit: { path: "$.messages[1].content", set: "replaced" }
  - name: by-content
    match: { content_regex: "relevance of the question" }
    edit: { path: "$.options.temperature", set: 0.5 }
"""


def test_patches() -> None:
    patches = load_patches(PATCHES)
    request = body(
        {
            "messages": [{"role": "system", "content": "Score from 0 to 100"}, {"role": "user", "content": "q"}],
            "options": {"temperature": 0},
        }
    )
    patched, names = apply_patches(patches, "ollama", request, LineUp(1, "guardrail_validation"))
    assert names == ["strict-guardrail"]
    assert json.loads(patched)["messages"][0]["content"] == "Be very strict. Score from 0 to 100"
    _, names = apply_patches(patches, "jina", request, LineUp(1, "guardrail_validation"))
    assert names == []
    patched, names = apply_patches(patches, "ollama", request, LineUp(2, "document_grading"))
    assert names == ["second-step"] and json.loads(patched)["messages"][1]["content"] == "replaced"
    patched, names = apply_patches(patches, "ollama", body({"prompt": "the relevance of the question", "options": {}}))
    assert names == ["by-content"] and json.loads(patched)["options"] == {"temperature": 0.5}
    untouched, names = apply_patches(patches, "ollama", request, None)
    assert untouched == request and names == []


def test_patch_schema_rejects_bad_edits() -> None:
    with pytest.raises(ValidationError):
        load_patches([{"name": "x", "edit": {"path": "$.a"}}])  # neither replace nor set
    with pytest.raises(ValidationError):
        load_patches([{"name": "x", "edit": {"path": "$.a", "set": 1, "replace": {"find": "a", "with": "b"}}}])
    with pytest.raises(ValidationError):
        load_patches([{"name": "x", "match": {"step": 0}, "edit": {"path": "$.a", "set": 1}}])
    with pytest.raises(ValidationError):
        load_patches([{"name": "x", "match": {"node": "typo"}, "edit": {"path": "$.a", "set": 1}}])
    assert load_patches([{"name": "x", "edit": {"path": "$.a", "set": None}}])[0].edit.set is None


def test_tape_chunks_and_headers() -> None:
    assert split_chunks(b"aaabbc", [[0, 0.0], [3, 10.5], [5, 20.0]]) == [(b"aaa", 0.0), (b"bb", 10.5), (b"c", 20.0)]
    assert split_chunks(b"xyz", []) == [(b"xyz", 0.0)]
    headers = tape_headers({"content-type": "application/json", "content-length": "99", "x-many": ["1", "2"]}, 3)
    assert headers == [("content-type", "application/json"), ("x-many", "1"), ("x-many", "2"), ("content-length", "3")]
