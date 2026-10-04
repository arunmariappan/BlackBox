"""Telegram alerts: a plain `httpx` POST to the Bot API, when a bot token and chat id are configured.

The token sits in the request URL, so it must never reach a log: errors are logged without the URL, and a filter on
the `httpx` logger masks it in httpx's own request lines.
"""

import logging
import re

import httpx

from blackbox.config import AlertsConfig
from blackbox.net import make_client

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"bot\d+:[A-Za-z0-9_-]+")


class RedactBotToken(logging.Filter):
    """Masks `bot<id>:<secret>` in a log record's message and arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _TOKEN.sub("bot***", record.msg)
        if record.args:
            args = record.args if isinstance(record.args, tuple) else (record.args,)
            record.args = tuple(_TOKEN.sub("bot***", str(a)) if _TOKEN.search(str(a)) else a for a in args)
        return True


def install_redaction() -> None:
    for name in ("httpx", "httpcore"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactBotToken) for f in logger.filters):
            logger.addFilter(RedactBotToken())


class Telegram:
    def __init__(self, config: AlertsConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._client = client
        install_redaction()

    @property
    def enabled(self) -> bool:
        return bool(self.config.telegram_bot_token and self.config.telegram_chat_id)

    async def send(self, text: str) -> bool:
        if not self.enabled or self.config.telegram_bot_token is None:
            return False
        url = f"{self.config.telegram_api}/bot{self.config.telegram_bot_token.get_secret_value()}/sendMessage"
        body = {"chat_id": self.config.telegram_chat_id, "text": text[:4000], "disable_web_page_preview": True}
        try:
            if self._client is not None:
                response = await self._client.post(url, json=body, timeout=10)
            else:
                async with make_client(timeout=10) as client:
                    response = await client.post(url, json=body)
        except httpx.HTTPError as exc:
            log.warning("telegram: sending the alert failed (%s)", type(exc).__name__)
            return False
        if response.status_code >= 400:
            log.warning("telegram: sending the alert failed with HTTP %d", response.status_code)
            return False
        return True
