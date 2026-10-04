"""The OpsDesk agent service (`opsdesk agent`, port 8220): `POST /run {instruction, sandbox, model, max_steps}`.

Model calls go to Ollama through BlackBox's proxy (8210), tool calls to the environment through the proxy (8213),
and spans to BlackBox over OTLP, all through `blackbox.sdk`, the only part of BlackBox OpsDesk may import.
"""

from typing import Any

import httpx
from fastapi import FastAPI, Request

from blackbox import sdk
from opsdesk.agent.loop import RunRequest, run_agent


def make_agent_app(model_url: str, env_url: str, chat: Any | None = None) -> FastAPI:
    app = FastAPI(title="OpsDesk agent")
    if chat is None:
        import ollama

        chat = ollama.Client(host=model_url, timeout=600)
    env = httpx.Client(base_url=env_url, timeout=60, trust_env=False)

    @app.post("/run")
    def run(request: RunRequest, http_request: Request) -> dict[str, Any]:
        with sdk.agent_run(
            "opsdesk",
            input=request.model_dump(),
            headers=dict(http_request.headers),
            attributes={"opsdesk.task": request.task or ""},
        ) as handle:
            try:
                result = run_agent(request, chat, env)
            except Exception as exc:
                handle.set_ending("error")
                failed: dict[str, Any] = {
                    "final_answer": None,
                    "ending": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace_id": handle.trace_id,
                }
                handle.set_output(failed)
                return failed
            output: dict[str, Any] = {
                "final_answer": result.final_answer,
                "ending": result.ending,
                "steps": result.steps,
                "tool_calls": result.tool_calls,
                "trace_id": handle.trace_id,
            }
            handle.set_output(output)
            handle.set_ending(result.ending)
            return output

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
