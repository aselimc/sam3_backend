"""Sphinx hook: build openapi.json from the FastAPI app at docs build time.

Runs in ``builder-inited`` so the spec lands on disk before any ``.. openapi::``
directive tries to read it. Imports only the router + schemas (pydantic-only),
never the heavy SAM3 service.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.router import router


def _build_spec() -> dict:
    app = FastAPI(title="SAM3.1 Backend", version="0.1.0")
    app.include_router(router)
    return get_openapi(
        title=app.title,
        version=app.version,
        description="Text-prompted segmentation over SAM3.",
        routes=app.routes,
    )


def write_openapi(app) -> None:
    out = Path(app.srcdir) / "openapi.json"
    out.write_text(json.dumps(_build_spec(), indent=2))


def setup(app):
    app.connect("builder-inited", lambda a: write_openapi(a))
    return {"version": "0.1", "parallel_read_safe": True}
