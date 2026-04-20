# v2 Re-architecture: Enterprise-Grade Multi-Model Inference Backend

## Context

Current `master` is a single-process FastAPI service that holds a `SAM3Service` singleton in app state, drives concurrency with an `asyncio.Semaphore`, stores jobs in an in-memory dict (lost on restart), writes mask outputs to local disk, and has no authentication, rate limiting, or tenancy. It works for a single GPU box but cannot scale horizontally, survive crashes, support multiple users safely, or host additional model families (depth, pose, 3D reconstruction).

This plan re-architects the repo on a new `v2` branch as a decoupled, four-tier system — **API / Broker / Worker / Storage** — with a pluggable model-adapter layer so SAM3 becomes the first of many models rather than the whole app. Deployment targets: Docker Compose for local Windows+GPU dev parity, Helm chart + raw K8s manifests for prod.

## Decisions (confirmed with user)

| Axis | Choice |
|---|---|
| Task queue | **Celery** (Redis broker + Redis result backend) |
| Auth | **API Key + JWT** (argon2-hashed keys, python-jose JWT) |
| Metadata store | **Postgres** via SQLAlchemy 2.x async + Alembic |
| Deploy | **Docker Compose (local dev)** + **Helm/K8s manifests (prod)** |
| Object store | **S3 API** — boto3 client, MinIO for dev, AWS S3 / any S3-compat for prod |
| Rate limit | Redis token bucket, keyed by `owner_id` and endpoint class |

## Target Architecture

```
                ┌─────────────────────────────────────────────────────┐
                │                 Client (Browser / SDK)              │
                └───────────────────────┬─────────────────────────────┘
                                        │ HTTPS  (JWT or API Key)
                                        ▼
┌────────────────────────────── API TIER (stateless, N replicas) ──────────────────────────────┐
│  FastAPI  ─  deps: auth (JWT/APIKey) ─ rate-limit ─ storage ─ broker.submit                   │
│  • POST  /v1/auth/*        (login, refresh, api-keys CRUD)                                    │
│  • POST  /v1/uploads       → presigned PUT URL  (direct-to-S3, no proxy)                      │
│  • POST  /v1/tasks/{type}  → enqueue Celery job, return job_id                                │
│  • GET   /v1/jobs/{id}     → status (scoped to owner_id)                                      │
│  • GET   /v1/jobs/{id}/artifacts  → presigned GET URLs (expiring)                             │
└───────────────────────────────────┬─────────────────────────────┬─────────────────────────────┘
                                    │                             │
                        enqueue     │                             │ read/write metadata
                                    ▼                             ▼
                     ┌──────────── BROKER TIER ─────────┐   ┌──── METADATA TIER ────┐
                     │  Redis (jobs, rate-limit, locks) │   │  Postgres             │
                     │  • Celery queues (per task-type  │   │  • users, api_keys    │
                     │    or per GPU-class)             │   │  • jobs (audit/hist)  │
                     │  • Token-bucket rate limiters    │   │  • tenants            │
                     │  • Pub/Sub for SSE status push   │   └───────────────────────┘
                     └────────────────┬─────────────────┘
                                      │ consume
                                      ▼
┌────────────────────────────── COMPUTE TIER (GPU pods) ───────────────────────────────────────┐
│  Celery workers, one process per GPU. Each subscribes to queues it advertises capability for. │
│  ┌──────────────────────────── Worker process ────────────────────────────┐                   │
│  │  ModelRegistry (loaded-on-demand, LRU-evict on GPU mem pressure)       │                   │
│  │   ├─ sam3 adapter                                                      │                   │
│  │   ├─ sam2 adapter                                                      │                   │
│  │   ├─ depth_anything_v2 adapter                                         │                   │
│  │   └─ …                                                                 │                   │
│  │  TaskRunner  — input fetch → preflight (OOM gate) → infer → upload     │                   │
│  │  Prometheus exporter, signal-aware graceful shutdown                   │                   │
│  └────────────────────────────────────────────────────────────────────────┘                   │
└───────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                    │ S3 API (boto3)
                                    ▼
                     ┌──────────── STORAGE TIER ────────┐
                     │  MinIO (dev) / S3 (prod)         │
                     │  Bucket layout:                  │
                     │   uploads/{tenant}/{uuid}        │
                     │   artifacts/{tenant}/{job_id}/*  │
                     └──────────────────────────────────┘
```

### Why this split

- **Crash recovery**: API replicas are stateless → k8s reschedules at will. Jobs live in Redis + Postgres, not process memory. Worker crash → Celery re-delivers the message to another worker (`acks_late=True`, `task_reject_on_worker_lost=True`).
- **Back-pressure**: Workers pull from Redis only when idle → GPU saturation without losing requests.
- **Horizontal scale**: GPU box count = worker replica count. API tier scales independently on CPU nodes.
- **OOM isolation**: A worker OOM kills one job; API + other workers unaffected.
- **Security**: Auth/rate-limit runs *before* enqueue. Storage URLs are presigned and scoped. Workers never expose HTTP.

## v2 Directory Layout

```
/  (branch: v2, will become new master)
├── pyproject.toml                 # single workspace, groups per tier
├── README.md
├── CHANGELOG.md
├── .env.example
├── docker-compose.yml             # api + worker + redis + postgres + minio + prometheus + grafana
├── Dockerfile.api
├── Dockerfile.worker              # CUDA base, inherits models via submodules
│
├── packages/
│   ├── core/                      # shared kernel — NO framework deps
│   │   ├── types.py               # TaskType, Modality, JobState, Tenant, OwnerId
│   │   ├── schemas.py             # TaskRequest/Result base Pydantic models
│   │   ├── errors.py              # AppError hierarchy → HTTP mapping
│   │   ├── config.py              # pydantic-settings, env-driven
│   │   ├── logging.py             # loguru + JSON sink + request_id ctx
│   │   └── telemetry.py           # prometheus registries, OTel tracer
│   │
│   ├── storage/                   # S3 abstraction
│   │   ├── base.py                # StorageBackend ABC (put, get, presign_put, presign_get, delete, stat)
│   │   ├── s3.py                  # boto3 impl (works for AWS + MinIO)
│   │   ├── local.py               # filesystem impl — tests only
│   │   └── keys.py                # key layout helpers (tenant/job scoping)
│   │
│   ├── broker/                    # Celery app + queue helpers
│   │   ├── celery_app.py          # Celery() factory, routes per TaskType
│   │   ├── ratelimit.py           # Redis token-bucket (Lua script, atomic)
│   │   ├── locks.py               # Redis distributed lock (setnx + TTL)
│   │   └── pubsub.py              # job-status events for SSE
│   │
│   ├── db/                        # Postgres access
│   │   ├── models.py              # SQLAlchemy: User, ApiKey, Tenant, JobRecord
│   │   ├── session.py             # async engine + session factory
│   │   ├── repositories/          # one file per aggregate
│   │   └── migrations/            # Alembic
│   │
│   ├── security/
│   │   ├── jwt.py                 # python-jose encode/decode, RS256 rotation-ready
│   │   ├── apikey.py              # argon2 hash + verify
│   │   ├── passwords.py           # argon2
│   │   └── presign.py             # thin wrapper over storage.presign_*
│   │
│   ├── models/                    # ── ADAPTER LAYER (the pluggable part) ──
│   │   ├── base.py                # ModelAdapter Protocol, ModelCapabilities
│   │   ├── registry.py            # global registry + @register decorator
│   │   ├── device.py              # GPU mem probe, fits() check, LRU eviction
│   │   ├── sam3/
│   │   │   ├── adapter.py         # wraps third_party/sam3
│   │   │   └── weights.py         # HF download, cache dir resolution
│   │   ├── sam2/
│   │   │   └── adapter.py
│   │   ├── depth_anything_v2/
│   │   │   └── adapter.py         # stub, scaffolded but not wired unless enabled
│   │   └── README.md              # "how to add a new model" guide
│   │
│   └── tasks/                     # ── TASK LAYER (what the API exposes) ──
│       ├── base.py                # TaskSpec: input schema, output schema, required capability
│       ├── registry.py
│       ├── segmentation/
│       │   ├── text_prompt.py     # uses any adapter with SEGMENTATION_TEXT capability
│       │   ├── point_prompt.py
│       │   └── post/regularize.py # migrated from current app/regularization.py
│       ├── depth/
│       │   └── monocular.py
│       ├── pose/                  # scaffolded
│       └── reconstruction/        # scaffolded
│
├── services/
│   ├── api/                       # FastAPI app
│   │   ├── main.py                # app factory + lifespan
│   │   ├── deps.py                # get_current_user, get_api_key, enforce_rate_limit, get_storage
│   │   ├── middleware/
│   │   │   ├── request_id.py
│   │   │   ├── logging.py
│   │   │   └── error_handler.py   # AppError → HTTP mapping
│   │   └── routers/
│   │       ├── auth.py            # login, refresh, /api-keys
│   │       ├── uploads.py         # presigned PUT issuance
│   │       ├── tasks.py           # generic /v1/tasks/{task_type}
│   │       ├── jobs.py            # status + artifacts
│   │       └── health.py          # /live, /ready (checks redis+pg+s3)
│   │
│   └── worker/
│       ├── main.py                # celery -A services.worker.main worker
│       ├── bootstrap.py           # preload declared-enabled models
│       ├── runner.py              # wraps TaskSpec.run with preflight + OOM guard + retry
│       ├── signals.py             # graceful shutdown, warm-pool reset on SIGTERM
│       └── oom_guard.py           # context mgr: check torch.cuda.mem_get_info before run
│
├── third_party/                   # git submodules — one per model family
│   ├── sam3/      (existing)
│   ├── sam2/      (new)
│   └── depth_anything_v2/  (new)
│
├── infra/
│   ├── k8s/
│   │   ├── helm/sam3-backend/     # Chart.yaml, values.yaml, templates/*
│   │   └── manifests/             # raw kustomize for non-helm envs
│   ├── prometheus/                # scrape configs
│   └── grafana/                   # dashboards: latency, gpu mem, queue depth
│
├── tests/
│   ├── unit/                      # per-package, no external deps
│   ├── integration/               # docker-compose.test.yml, real redis+pg+minio
│   └── e2e/                       # end-to-end: upload → enqueue → poll → download
│
├── scripts/
│   ├── bootstrap_dev.ps1          # Windows dev setup
│   ├── seed_dev_data.py
│   └── migrate.py                 # alembic wrapper
│
└── docs/                          # Sphinx, same pipeline as current
```

## Core Abstractions

### `core/types.py`

```python
class Modality(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    POINTCLOUD = "pointcloud"
    TEXT = "text"
    DEPTH = "depth"
    MASK = "mask"
    MESH = "mesh"

class TaskType(StrEnum):
    SEGMENTATION_TEXT   = "segmentation.text"
    SEGMENTATION_POINT  = "segmentation.point"
    SEGMENTATION_BOX    = "segmentation.box"
    DEPTH_MONOCULAR     = "depth.monocular"
    POSE_HUMAN          = "pose.human"
    RECONSTRUCTION_MV   = "reconstruction.multiview"

class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    RETRYING = "retrying"

class Capability(BaseModel):
    task: TaskType
    inputs: list[Modality]
    outputs: list[Modality]
```

### `models/base.py`

```python
class ModelCapabilities(BaseModel):
    model_id: str                 # "sam3", "sam2_hiera_large", "depth_anything_v2_vitl"
    capabilities: list[Capability]
    min_gpu_mem_mb: int           # load budget
    per_request_gpu_mem_mb: int   # empirical working-set estimate
    max_input_pixels: int
    supports_fp16: bool
    supports_batching: bool

class ModelAdapter(Protocol):
    caps: ClassVar[ModelCapabilities]
    def load(self, device: str) -> None: ...
    def unload(self) -> None: ...
    def infer(self, req: "TaskRequest") -> "TaskResult": ...
```

### Adding a new model (documented in `packages/models/README.md`)

1. `git submodule add <url> third_party/<name>`
2. Create `packages/models/<name>/adapter.py` implementing `ModelAdapter`
3. Decorate class with `@register_model`
4. Declare `ModelCapabilities` (task list, GPU budget)
5. Add to `WORKER_ENABLED_MODELS` env var in the worker deployment
6. API exposes the task automatically via `TaskSpec` capability match

No API code changes needed per new model — the generic `/v1/tasks/{task_type}` router dispatches on `TaskType` + capability.

## Fault Tolerance & Safety

| Concern | Mechanism |
|---|---|
| **OOM prevention** | `oom_guard.py` checks `torch.cuda.mem_get_info()` vs `per_request_gpu_mem_mb` before invoking model. Reject → requeue with backoff. Input image pixel cap enforced at API validation. |
| **Crash recovery** | Celery `acks_late=True` + `task_reject_on_worker_lost=True` → messages redelivered on crash. Postgres `JobRecord` is source of truth; reconciler job marks orphaned `RUNNING` jobs as `FAILED` after heartbeat timeout. |
| **Graceful shutdown** | `SIGTERM` → worker stops prefetch, finishes current task, flushes GPU cache, exits. K8s preStop hook + `terminationGracePeriodSeconds: 300`. |
| **Model load coordination** | Redis distributed lock per `model_id` to prevent thundering-herd HF downloads when N workers warm simultaneously. |
| **Retry policy** | Exponential backoff, max 3 retries. Dead-letter queue for poison messages, surfaced in `/v1/jobs/{id}`. |
| **Health checks** | `/live` (process alive), `/ready` (redis + pg + s3 reachable, at least one worker heartbeat in last 30s). |
| **Circuit breakers** | `pybreaker` around S3 and HF download calls. |
| **Input safety** | Max file size (env-configured), MIME sniff, dimension cap, image-bomb detection. |
| **Resource quotas** | Per-tenant daily job count + GPU-second budget tracked in Postgres. |

## Security & Authentication

- **The Bouncer (FastAPI Depends)**: `get_principal` resolves either `Authorization: Bearer <JWT>` or `X-API-Key: <key>`. Unauth → 401. Returns `Principal(owner_id, tenant_id, scopes)`.
- **Data isolation**: every `JobRecord` has `owner_id` + `tenant_id`. All queries `WHERE owner_id = :principal.owner_id`. Celery task payload carries `owner_id`; worker re-verifies on storage key prefix.
- **Rate limiting**: `RateLimiter` dep takes `(principal, bucket)` and runs Lua token-bucket in Redis. Buckets per endpoint class (`upload`, `enqueue.gpu`, `read`). Configurable per-tenant overrides.
- **Secure downloads**: result masks land in `artifacts/{tenant}/{job_id}/…`. API issues presigned GET URLs with TTL=10 min; never serves bytes itself. Uploads are same pattern (presigned PUT) — image bytes never traverse FastAPI.
- **API key lifecycle**: argon2-hashed at rest, shown once on creation, revocable, per-key scopes (`tasks:submit`, `tasks:read`, `admin`).
- **JWT**: RS256 with key rotation (JWKS endpoint). Short-lived access (15 min), refresh in httpOnly cookie.
- **Secrets**: all via env + K8s Secret / `.env` (gitignored). No secrets in code or images.
- **CORS** and **security headers** middleware (HSTS, X-Content-Type-Options, CSP for docs).
- **Audit log**: every auth event + job submit in Postgres `audit_events` table.

## Migration / Execution Plan

Phased so each phase is independently verifiable. Each phase is a PR into `v2`.

### Phase 0 — Branch & scaffolding
- `git checkout -b v2`
- Create new top-level dirs (`packages/`, `services/`, `infra/`)
- Add `.env.example`, `docker-compose.yml` (stub: redis, pg, minio only)
- Leave current `app/` intact for reference; don't delete yet
- CI: add `v2` branch to GitHub Actions matrix

### Phase 1 — Core kernel
- Implement `packages/core/` (types, schemas, errors, config, logging, telemetry)
- Implement `packages/storage/` with `S3Backend` + `LocalBackend`
- Implement `packages/db/` (models, session, first Alembic migration: users, api_keys, tenants, job_records, audit_events)
- Implement `packages/security/` (jwt, apikey, passwords)
- Unit tests for each

### Phase 2 — Broker + Celery app
- `packages/broker/celery_app.py` with routes per `TaskType`
- Redis token-bucket rate limiter + tests (fakeredis)
- Distributed lock + pubsub

### Phase 3 — Model + task layers
- `packages/models/base.py`, `registry.py`, `device.py`
- Port current SAM3 service into `packages/models/sam3/adapter.py` (reuse logic from `app/sam3_service.py`)
- Port `app/regularization.py` → `packages/tasks/segmentation/post/regularize.py`
- Implement `TaskSpec` for `SEGMENTATION_TEXT`, `SEGMENTATION_POINT`, `SEGMENTATION_BOX`
- Scaffold `depth_anything_v2` adapter as empty class to prove the plugin story compiles

### Phase 4 — API service
- `services/api/main.py` + lifespan (no model loading — stateless)
- Auth router (login, refresh, api-key CRUD)
- Uploads router (presigned PUT)
- Generic tasks router: `POST /v1/tasks/{task_type}` → validate → rate-limit → enqueue → return `job_id`
- Jobs router: status + presigned artifact URLs
- Health router
- Middlewares: request_id, logging, error mapper
- Integration tests against compose stack

### Phase 5 — Worker service
- `services/worker/main.py` Celery entrypoint
- `bootstrap.py` preloads models listed in `WORKER_ENABLED_MODELS`
- `runner.py` with OOM guard, retry, heartbeat
- `oom_guard.py`, `signals.py`
- E2E test: submit via API → worker picks up → artifact lands in MinIO → presigned URL downloads it

### Phase 6 — Infra
- Finalize `docker-compose.yml` (api, worker, redis, pg, minio, prometheus, grafana)
- `Dockerfile.api` (slim, CPU) and `Dockerfile.worker` (CUDA base, submodules baked in)
- Helm chart + raw manifests under `infra/k8s/`
- Grafana dashboards for queue depth, inference latency, GPU mem

### Phase 7 — Cutover
- Delete legacy `app/`, `main.py`, `router.py`, `job_router.py`, `jobs.py`
- Update `README.md`, docs
- Make `v2` the new default branch on GitHub
- Tag `v2.0.0`

## Files reused from current repo (do not re-invent)

- `app/sam3_service.py` → logic ports into `packages/models/sam3/adapter.py` (model load, autocast inference, GPU mem metrics)
- `app/regularization.py` → verbatim to `packages/tasks/segmentation/post/regularize.py`
- `app/metrics.py` → expand, move to `packages/core/telemetry.py`
- `app/schemas.py` request fields → split between `packages/tasks/segmentation/*.py` (task I/O) and `packages/core/schemas.py` (base types)
- `tests/conftest.py` pre-import mocking pattern → keep for unit tests of model adapters
- CI workflow from `.github/workflows/ci.yml` → extend with `docker-compose up` integration job

## Verification

End-to-end smoke, all from `v2` branch:

1. `docker compose up -d` (api + worker + redis + pg + minio + prometheus)
2. `alembic upgrade head`
3. `scripts/seed_dev_data.py` creates a dev tenant + user + API key, prints key once
4. **Upload**: `POST /v1/uploads` with `X-API-Key` → returns presigned PUT → `curl -T image.jpg "<url>"`
5. **Enqueue**: `POST /v1/tasks/segmentation.text` `{image_ref, text_prompts:["cat"]}` → returns `job_id`
6. **Poll**: `GET /v1/jobs/{job_id}` until `succeeded`
7. **Download**: `GET /v1/jobs/{job_id}/artifacts` → presigned GET URLs → download masks
8. **Auth negative tests**: no key → 401; wrong tenant's `job_id` → 404 (not 403, to avoid enumeration)
9. **Rate-limit test**: exceed bucket → 429 with `Retry-After`
10. **Crash test**: `docker kill` the worker mid-job → verify Celery redelivers and job completes on restart
11. **OOM test**: submit oversized image → 422 at API (pre-enqueue) or safe failure at worker (post-preflight), job marked failed with reason, GPU not wedged
12. **Pluggability**: wire the scaffolded depth adapter, set `WORKER_ENABLED_MODELS=sam3,depth_anything_v2`, verify `POST /v1/tasks/depth.monocular` works
13. `pytest -m "unit or integration"` green; `pytest -m e2e` green against compose
14. Helm `helm template infra/k8s/helm/sam3-backend` renders cleanly; `helm lint` passes

## Non-goals (explicit)

- No multi-GPU-per-worker scheduling (one process = one GPU; scale by replicas)
- No streaming/video endpoints in Phase 1–7 (add later once task layer stable)
- No admin UI (CLI + API only)
- No on-the-fly model fine-tuning

## Open items to revisit mid-build

- Which specific SAM2 variant to ship first (`hiera_small` vs `hiera_large` — depends on GPU budget)
- Whether to add gRPC between API and workers later, or stay Celery-only
- Artifact retention policy (bucket lifecycle rules)
