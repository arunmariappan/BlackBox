"""Replay of stateful upstreams (the OpsDesk environment): sync-forwarding and identifier aliasing (phase 6)."""

from typing import TYPE_CHECKING

from blackbox.proxy.core import IncomingRequest, Outcome, Proxy, Recording

if TYPE_CHECKING:
    from blackbox.proxy.sessions import ReplaySession, TapeEntry


async def sync_forward(session: ReplaySession, proxy: Proxy, req: IncomingRequest, entry: TapeEntry) -> None:
    """Forward a state-changing call served from the tape to the session's own sandbox (phase 6)."""
    return None


async def forward_live_stateful(
    session: ReplaySession, proxy: Proxy, req: IncomingRequest, rec: Recording, body: bytes
) -> Outcome:
    """Send a live call to a stateful upstream (phase 6 adds identifier aliasing)."""
    return await proxy.record_live(req, rec.attribution, rec, hooks=False)
