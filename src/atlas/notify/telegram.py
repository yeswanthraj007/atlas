"""Send messages to your own Telegram chat through a bot you create.

Setup (once): in Telegram, message @BotFather -> /newbot -> copy the token. Then send any
message to your new bot and open https://api.telegram.org/bot<TOKEN>/getUpdates to find
your chat id. Put both in .env as TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
"""

from __future__ import annotations

import os

import httpx

API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotConfiguredError(RuntimeError):
    pass


def send(
    text: str,
    token: str | None = None,
    chat_id: str | None = None,
    client: httpx.Client | None = None,
) -> None:
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise TelegramNotConfiguredError("set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
    http = client or httpx.Client(timeout=30)
    resp = http.post(
        API.format(token=token),
        json={
            "chat_id": chat_id,
            "text": text[:4000],
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
    )
    resp.raise_for_status()
    if not resp.json().get("ok"):
        raise RuntimeError(f"telegram error: {resp.text[:200]}")
