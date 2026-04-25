#!/usr/bin/env bash
# Bootstrap dev env on Linux / WSL2.
# Phase 0 skeleton: prereq probe + .env scaffold + uv sync.
# Phase 7 will extend with GPU detection, image build, alembic, init_minio.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "── prereqs ──"
uv run --no-project --python 3.12 python scripts/_env_check.py

if [[ ! -f .env ]]; then
    echo "── .env missing → copying from .env.example ──"
    cp .env.example .env
    # TODO(phase 7): randomize LOCAL_API_KEY + WEBHOOK_SECRET in-place
    echo "TODO: randomize LOCAL_API_KEY and WEBHOOK_SECRET in .env"
fi

echo "── uv sync ──"
uv sync --group api --group worker --group dev

echo "── done ──"
echo "next:"
echo "  COMPOSE_PROFILES=cpu docker compose -f infra/compose/docker-compose.yml up -d redis postgres minio"
