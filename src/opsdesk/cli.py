"""The `opsdesk` command line."""

import typer

app = typer.Typer(name="opsdesk", help="OpsDesk, a test agent for a simulated IT ops desk.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """OpsDesk, a test agent for a simulated IT ops desk."""
