# Bootstrap dev env on Windows (PowerShell 7+).
# Phase 0 skeleton: prereq probe + .env scaffold + uv sync.
# Phase 7 will extend with GPU detection, image build, alembic, init_minio.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "-- prereqs --"
uv run --no-project --python 3.12 python scripts/_env_check.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path ".env")) {
    Write-Host "-- .env missing -> copying from .env.example --"
    Copy-Item ".env.example" ".env"
    # TODO(phase 7): randomize LOCAL_API_KEY + WEBHOOK_SECRET in-place
    Write-Host "TODO: randomize LOCAL_API_KEY and WEBHOOK_SECRET in .env"
}

Write-Host "-- uv sync --"
uv sync --group api --group worker --group dev
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "-- done --"
Write-Host "next:"
Write-Host "  `$env:COMPOSE_PROFILES='cpu'; docker compose -f infra/compose/docker-compose.yml up -d redis postgres minio"
