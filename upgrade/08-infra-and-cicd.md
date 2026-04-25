# 08 — Infra and CI/CD (local profile)

This document covers local development, container builds, and the CI required to keep the local profile honest. Kubernetes / Helm / KEDA / supply-chain controls (cosign, Trivy gating, External Secrets, image signing admission policy, multi-region) live in:

- [`enterprise/02-kubernetes-and-helm.md`](../enterprise/02-kubernetes-and-helm.md)
- [`enterprise/03-supply-chain-and-secrets.md`](../enterprise/03-supply-chain-and-secrets.md)
- [`enterprise/04-observability-at-scale.md`](../enterprise/04-observability-at-scale.md)
- [`enterprise/05-multi-region-and-ha.md`](../enterprise/05-multi-region-and-ha.md)

The local profile must work on:

- **Windows 10/11** with Docker Desktop and WSL2 (NVIDIA driver via the Microsoft CUDA-on-WSL channel).
- **Ubuntu 22.04+** with Docker Engine and the NVIDIA Container Toolkit.

Single-host, 1–2 GPUs.

## Local dev

### Bootstrap

```powershell
# Windows
.\scripts\bootstrap_dev.ps1
```

```bash
# Ubuntu
./scripts/bootstrap_dev.sh
```

The bootstrap script:

1. Verifies prerequisites (`docker`, `nvidia-smi`, `nvidia-container-runtime` on Linux; WSL2 + `nvidia-smi.exe` on Windows).
2. Generates `.env` from `.env.example` if missing, with random `LOCAL_API_KEY` and `WEBHOOK_SECRET`.
3. Detects GPU count and writes `COMPOSE_PROFILES` accordingly: `gpu1` (one GPU) or `gpu2` (two GPUs).
4. `uv sync --group api --group worker --group dev`.
5. Pulls / builds images.
6. Runs `alembic upgrade head` against the Compose Postgres.
7. Prints next-step commands.

### Compose stack

`infra/compose/docker-compose.yml` defines profiles:

```yaml
services:
  api:           { profiles: [cpu, gpu1, gpu2] }
  worker-gpu-0:  { profiles: [gpu1, gpu2] }    # binds CUDA_VISIBLE_DEVICES=0
  worker-gpu-1:  { profiles: [gpu2] }          # binds CUDA_VISIBLE_DEVICES=1
  worker-cpu:    { profiles: [cpu] }           # API integration tests; no real inference
  beat:          { profiles: [cpu, gpu1, gpu2] }
  redis:         { profiles: [cpu, gpu1, gpu2] }
  postgres:      { profiles: [cpu, gpu1, gpu2] }
  minio:         { profiles: [cpu, gpu1, gpu2] }
  prometheus:    { profiles: [cpu, gpu1, gpu2] }
  grafana:       { profiles: [cpu, gpu1, gpu2] }
  otelcol:       { profiles: [cpu, gpu1, gpu2] }
```

```bash
# Linux, 1 GPU
COMPOSE_PROFILES=gpu1 docker compose -f infra/compose/docker-compose.yml up -d

# Linux, 2 GPUs
COMPOSE_PROFILES=gpu2 docker compose -f infra/compose/docker-compose.yml up -d

# CPU-only (API integration tests only)
COMPOSE_PROFILES=cpu  docker compose -f infra/compose/docker-compose.yml up -d
```

Windows callout: Docker Desktop on Windows can only see the GPU through WSL2 + the Microsoft NVIDIA driver. The bootstrap script enforces this before activating a GPU profile. If WSL2 + NVIDIA is unavailable, the user is steered to the `cpu` profile.

### Volumes and data persistence

| Service | Volume | Notes |
|---|---|---|
| `postgres` | `pg_data` | survives `compose down` |
| `minio` | `minio_data` | survives `compose down` |
| `redis` | `redis_data` (AOF) | survives `compose down` |
| `worker-gpu-*` | bind: `~/.cache/huggingface` | shared HF cache across hosts and reruns; saves repeated multi-GB downloads |

`compose down -v` is the one-shot reset.

## Dockerfiles

### `Dockerfile.api`

Slim base, no CUDA. Multi-stage with `uv` for dependency install.

```dockerfile
FROM python:3.12-slim AS base
RUN useradd -m -u 1000 app
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --group api
COPY services/api /app/services/api
COPY packages    /app/packages
USER app
EXPOSE 8000 9100
ENTRYPOINT ["uv","run","python","-m","services.api.main"]
```

### `Dockerfile.worker`

CUDA base; submodules baked in; model weights optionally baked at build.

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS base
RUN apt-get update && apt-get install -y python3.12 python3.12-venv git ca-certificates libgl1
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --group worker
COPY packages       /app/packages
COPY services/worker /app/services/worker
COPY third_party    /app/third_party

ARG BAKE_MODELS=""
RUN if [ -n "$BAKE_MODELS" ]; then \
    python -m services.worker.bake_weights --models "$BAKE_MODELS"; \
    fi

USER 1000:1000
ENTRYPOINT ["uv","run","celery","-A","services.worker.main","worker"]
```

`BAKE_MODELS=sam3,depth_anything_v3` produces a `worker-baked` image with both weights pre-downloaded — useful for offline laptops and for CI runners that should not hit HuggingFace.

Image tags:

- `api:{git_sha}`, `api:{semver}` on release.
- `worker:{git_sha}` (no weights), `worker-baked:{git_sha}-{models_hash}` (weights baked).

## Migrations

```bash
uv run alembic upgrade head
```

Run against the Compose Postgres. The bootstrap script invokes this once. Subsequent migrations are applied manually or via `scripts/migrate.py`.

The enterprise overlay wires this into a Helm `pre-install`/`pre-upgrade` Job — see [`enterprise/02-kubernetes-and-helm.md`](../enterprise/02-kubernetes-and-helm.md).

## Buckets bootstrap

`scripts/init_minio.py` creates the `sam3-uploads` and `sam3-artifacts` buckets, applies versioning, and sets the local CORS rule. Idempotent. Invoked by the bootstrap script on first run.

## CI

GitHub Actions workflows. The local-profile CI is intentionally minimal — what matters for portfolio readers is "this code is clean, tested, and runs end-to-end". Full release / signing / SBOM pipelines are an enterprise concern and are documented in [`enterprise/03-supply-chain-and-secrets.md`](../enterprise/03-supply-chain-and-secrets.md).

### `ci.yml` — runs on every PR

```yaml
jobs:
  lint-test:
    steps:
      - checkout (with submodules)
      - uv sync --group api --group worker --group dev
      - uv run ruff check .
      - uv run ruff format --check .
      - uv run mypy packages services
      - uv run pytest -m "unit" --cov --cov-report=xml
  integration:
    services: { redis, postgres, minio }
    steps:
      - uv run alembic upgrade head
      - uv run pytest -m "integration"
  openapi-check:
    steps:
      - python -m services.api.openapi --out openapi.json
      - openapi-diff vs main; fail on breaking change
  docs-build:
    steps:
      - uv run sphinx-build docs docs/_build
      - upload artifact
```

The `integration` job uses Compose-style services in GitHub Actions. There is no GPU in CI; e2e tests against the GPU profile are run locally before tagging a release.

### `release.yml` — runs on `v*` tags

Local-profile release does:

```yaml
jobs:
  build-images:
    matrix: [api, worker]
    steps:
      - docker buildx build --push -t ghcr.io/<org>/sam3-<svc>:${tag}
  publish-sdks:
    steps:
      - python -m services.api.openapi --out openapi.json
      - openapi-generator-cli generate -g python            -i openapi.json -o sdk/py
      - openapi-generator-cli generate -g typescript-axios  -i openapi.json -o sdk/ts
      - publish to PyPI / npm under semver matching the tag
  publish-docs:
    steps:
      - uv run sphinx-build docs docs/_build
      - publish to GitHub Pages
```

Image signing (cosign), SBOM emission (Syft), vulnerability scanning that gates the release (Trivy), and registry admission policy are **enterprise** concerns. See [`enterprise/03-supply-chain-and-secrets.md`](../enterprise/03-supply-chain-and-secrets.md). The local release publishes images and SDKs to public registries without those guards; downstream adopters add them as part of the enterprise overlay.

## Pinned weights

HF revisions pinned in `packages/models/<name>/weights.py`; sha256 verified at load. Updating a pin is a code change reviewed in PR — never silently floats.

## Observability stack

Prometheus + Grafana + OTel Collector are wired into Compose. The local Grafana ships pre-loaded dashboards under `infra/grafana/dashboards/`:

- API health
- Queue depth
- Inference latency per `(model_id, gpu_class)`
- GPU memory & evictions
- Local request log (Loki not included — Grafana points at Promtail-less local files only as a stretch goal)

Loki + Tempo are enterprise; see [`enterprise/04-observability-at-scale.md`](../enterprise/04-observability-at-scale.md).

## Cost knobs (local)

Mostly N/A for local. Two notable knobs:

- `BAKE_MODELS=` at build avoids re-downloading multi-GB weights every time the worker container is rebuilt.
- `MODELS_ENABLED=` controls which adapters the worker even attempts to load — defaults to both `sam3,depth_anything_v3`. Setting it to a single model halves cold-start time and warm-pool footprint when only one is needed.
