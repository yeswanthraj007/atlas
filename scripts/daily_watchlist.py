"""Daily watchlist job: refresh data, build the ranked report, notify.

    uv run python scripts/daily_watchlist.py --session morning
    uv run python scripts/daily_watchlist.py --session evening --no-refresh

Writes data/reports/watchlist/<date>_<session>.md and .json, and sends a short Telegram
summary if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set in .env.
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from atlas.lock import single_instance
from atlas.notify.telegram import TelegramNotConfiguredError, send
from atlas.recommend.daily import build_daily, telegram_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", choices=["morning", "evening"], required=True)
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--no-notify", action="store_true")
    args = ap.parse_args()
    load_dotenv(".env")
    with single_instance("daily_watchlist"):
        res = build_daily(args.session, refresh=not args.no_refresh)
        print(f"wrote {res.markdown_path} and {res.json_path}")
        if not args.no_notify:
            try:
                send(telegram_summary(res, os.getenv("ATLAS_DASHBOARD_URL")))
                print("telegram: sent")
            except TelegramNotConfiguredError as exc:
                print(f"telegram: skipped ({exc})")


if __name__ == "__main__":
    main()
