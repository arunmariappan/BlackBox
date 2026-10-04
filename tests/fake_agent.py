"""A fake agent for replay tests: a few model calls through the proxy, traced with the SDK, as an HTTP service."""

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request

from blackbox import sdk

NODES = ["plan", "act", "act", "answer"]


def deterministic_reply(body: dict[str, Any]) -> dict[str, Any]:
    """What the fake model answers: a function of the model and the messages, so the same request gets the same
    reply and a changed one gets another."""
    digest = hashlib.sha256(json.dumps([body.get("model"), body.get("messages")], sort_keys=True).encode()).hexdigest()
    return {"role": "assistant", "content": f"{body.get('model')}:{digest[:10]}"}


@dataclass
class AgentConfig:
    model_url: str
    model: str = "qwen3.5:4b"
    prompts: list[str] = field(default_factory=lambda: [f"You are step {i + 1}." for i in range(4)])
    clock: str = "none"  # none, wall (datetime.now in the prompt), sdk (sdk.now in the prompt)
    stream: bool = False


class HttpChat:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = httpx.Client(base_url=config.model_url, timeout=10)

    def chat(self, *, model: str, messages: list[Any], tools: Any = None, **kwargs: Any) -> dict[str, Any]:
        body = {"model": model, "messages": messages, "stream": self.config.stream}
        response = self.client.post("/api/chat", json=body)
        if response.status_code != 200:
            raise RuntimeError(f"model call failed with {response.status_code}: {response.text}")
        if not self.config.stream:
            return response.json()  # type: ignore[no-any-return]
        content, final = "", {}
        for line in response.text.splitlines():
            chunk = json.loads(line)
            content += chunk.get("message", {}).get("content", "")
            if chunk.get("done"):
                final = chunk
        return {**final, "message": {"role": "assistant", "content": content}}


def make_agent_app(config: AgentConfig) -> FastAPI:
    app = FastAPI()
    chat = HttpChat(config)

    @app.post("/run")
    def run(request_body: dict[str, Any], request: Request) -> dict[str, Any]:
        with sdk.agent_run("fake-agent", input=request_body, headers=dict(request.headers)) as run:
            replies: list[str] = []
            for i, node_name in enumerate(NODES):
                with sdk.node(node_name):
                    system = config.prompts[i]
                    if config.clock == "wall":
                        system += f" The time is {dt.datetime.now(dt.UTC).isoformat()}."
                    elif config.clock == "sdk":
                        system += f" The time is {sdk.now().isoformat()}."
                    user = request_body["question"] + "".join(f"\n{r}" for r in replies)
                    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                    reply = sdk.traced_chat(chat, model=config.model, messages=messages)
                    replies.append(reply["message"]["content"])
            output = {"answer": replies[-1], "steps": replies}
            run.set_output(output)
            run.set_ending("finished")
        return output

    return app
