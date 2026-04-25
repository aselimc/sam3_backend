"""Seed dev data + key utilities.

`--print-key` echoes the current local API key from .env (per upgrade/03
§Authentication). Job rows are created by submission, not seeded — this
script exists so a new dev can grab the key without grepping .env.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from packages.core.config import get_settings
from packages.db.models import Base
from packages.db.session import get_engine, reset_engine


async def _create_all() -> None:
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("ok schema created")
    finally:
        await reset_engine()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Seed local dev data.")
    p.add_argument("--print-key", action="store_true", help="Echo LOCAL_API_KEY and exit.")
    p.add_argument(
        "--create-schema",
        action="store_true",
        help="Run Base.metadata.create_all against DATABASE_URL (tests only).",
    )
    args = p.parse_args(argv)

    if args.print_key:
        print(get_settings().local_api_key)
        return 0

    if args.create_schema:
        asyncio.run(_create_all())
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
