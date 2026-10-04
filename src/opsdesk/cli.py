"""The `opsdesk` command line: the environment, the agent, and the tasks."""

from typing import Annotated

import typer

app = typer.Typer(name="opsdesk", help="OpsDesk, a test agent for a simulated IT ops desk.", no_args_is_help=True)
tasks_app = typer.Typer(help="The 25 tasks.", no_args_is_help=True)
app.add_typer(tasks_app, name="tasks")


@app.callback()
def main() -> None:
    """OpsDesk, a test agent for a simulated IT ops desk."""


@app.command()
def env(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8221,
) -> None:
    """Run the simulated IT ops desk (sandboxes, tools, admin API)."""
    import uvicorn

    from opsdesk.env.app import make_env_app

    uvicorn.run(make_env_app(), host=host, port=port, log_level="warning")


@app.command()
def agent(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8220,
    model_url: Annotated[
        str, typer.Option("--model-url", help="Ollama, through BlackBox's proxy.")
    ] = "http://127.0.0.1:8210",
    env_url: Annotated[
        str, typer.Option("--env-url", help="The environment, through BlackBox's proxy.")
    ] = "http://127.0.0.1:8213",
    blackbox: Annotated[str, typer.Option("--blackbox", help="BlackBox's OTLP endpoint.")] = "http://127.0.0.1:8200",
) -> None:
    """Run the agent service (POST /run)."""
    import uvicorn

    from blackbox import sdk
    from opsdesk.agent.app import make_agent_app

    sdk.init(service_name="opsdesk-agent", endpoint=blackbox)
    try:
        uvicorn.run(make_agent_app(model_url, env_url), host=host, port=port, log_level="warning")
    finally:
        sdk.shutdown()


@tasks_app.command("list")
def tasks_list() -> None:
    """List the tasks by category."""
    from opsdesk.tasks import load_tasks

    for task in sorted(load_tasks().values(), key=lambda t: (t.category, t.id)):
        typer.echo(f"{task.category:17} {task.id:28} {task.scenario:28} {task.instruction}")


@app.command()
def check(
    sandbox: Annotated[str, typer.Argument(help="Sandbox id.")],
    task: Annotated[str | None, typer.Option("--task", help="Task id (default: the sandbox's task).")] = None,
    answer: Annotated[str | None, typer.Option("--answer", help="The agent's final answer.")] = None,
    env_url: Annotated[str, typer.Option("--env-url")] = "http://127.0.0.1:8221",
) -> None:
    """Check a sandbox's final state and action log against a task."""
    import json

    import httpx

    response = httpx.post(
        f"{env_url}/_sandboxes/{sandbox}/check", json={"task_id": task, "final_answer": answer}, trust_env=False
    )
    if response.status_code >= 400:
        typer.echo(f"check failed ({response.status_code}): {response.text}", err=True)
        raise typer.Exit(1)
    result = response.json()
    typer.echo("PASS" if result["passed"] else "FAIL")
    for failure in result["failed"]:
        typer.echo(f"  ✗ {failure}")
    if not result["passed"]:
        typer.echo(json.dumps(result["violations"], indent=2))
        raise typer.Exit(1)
