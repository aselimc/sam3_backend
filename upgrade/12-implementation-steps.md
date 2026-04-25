# 12 — Implementation Steps

Granular, small-step companion to [09-phases.md](./09-phases.md). Each step is half-day to one-day of work, independently reviewable, with explicit files, exit check, and dependencies. Use as the daily TODO list.

Conventions:

- **ID**: `<phase>.<step>` so they sort.
- **Files**: paths the step creates or edits. `+` = new file, `~` = edit existing.
- **Exit**: one-line check the step is done.
- **Depends**: step IDs that must be complete first. `—` = none beyond the previous phase.

A step is not done until: code lands, tests pass, doc references updated.

---

## Phase 0 — Scaffolding - Done

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 0.1 | Create directory tree | `+ packages/{core,storage,broker,db,security,io,models,tasks}/__init__.py`, `+ services/{api,worker}/__init__.py`, `+ infra/compose/`, `+ tests/{unit,integration,e2e}/`, `+ scripts/` | `tree -L 3` matches [PLAN.md §Directory Layout](../PLAN.md) | — |
| 0.2 | Workspace `pyproject.toml` | `~ pyproject.toml` (groups: `api`, `worker`, `dev`, `docs`) | `uv sync --group api --group worker --group dev` resolves | 0.1 |
| 0.3 | `.env.example` + `.gitignore` adds | `+ .env.example`, `~ .gitignore` (`+ .env`, `+ data/`, `+ docs/_build/`) | `cp .env.example .env` produces a valid env | 0.2 |
| 0.4 | Compose stub (no app yet) | `+ infra/compose/docker-compose.yml` (services: redis, postgres, minio) | `docker compose -f infra/compose/docker-compose.yml up -d` brings 3 healthy | 0.3 |
| 0.5 | Submodule placeholders | `+ third_party/.gitkeep`, `~ .gitmodules` | folder exists; submodule add deferred to 4.13/4.16 | 0.1 |
| 0.6 | Bootstrap script skeletons | `+ scripts/bootstrap_dev.sh`, `+ scripts/bootstrap_dev.ps1`, `+ scripts/_env_check.py` | scripts exit 0 on a healthy host; print TODOs otherwise | 0.4 |
| 0.7 | CI matrix entry for `v2` | `~ .github/workflows/ci.yml` | PR to `v2` triggers the workflow | 0.2 |
| 0.8 | Legacy tests still green | n/a | `uv run pytest -q` passes against current `app/` | 0.2 |

---

## Phase 1 — Core kernel - Done

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 1.1 | `core.types` enums | `+ packages/core/types.py` (`TaskType`, `JobState`, `GpuClass`) | importable; `TaskType.SEGMENTATION_TEXT` exists | 0.* |
| 1.2 | `core.schemas` base | `+ packages/core/schemas.py` (`VersionedModel`) | `VersionedModel(version="1")` validates | 1.1 |
| 1.3 | `core.errors` hierarchy | `+ packages/core/errors.py` (`AppError`, mappings to HTTP) | every code in [03 §Errors](./03-api-spec.md#errors) has a class | 1.1 |
| 1.4 | `core.config` settings | `+ packages/core/config.py` (`Settings(BaseSettings)`) | `Settings()` reads `.env`; `MAX_UPLOAD_BYTES` defaulted | 1.1 |
| 1.5 | `core.logging` JSON sink | `+ packages/core/logging.py` (`configure()`, ContextVars for request_id/job_id/…) | `logger.info("x")` emits one line matching the schema in [07 §Log schema](./07-observability.md#log-schema) | 1.4 |
| 1.6 | `core.telemetry` Prom + OTel | `+ packages/core/telemetry.py` (registry + tracer factory) | `tracer.start_as_current_span("t")` works; `/metrics` registry returns text | 1.4 |
| 1.7 | `core.imageguard` global cap | `+ packages/core/imageguard.py` (`PIL.Image.MAX_IMAGE_PIXELS = …`) | importing the package sets the limit globally | — |
| 1.8 | `core.cancel` cooperative cancel | `+ packages/core/cancel.py` (`CancelCheck`, `CancelRequested`) | trip → next call raises | 1.3 |
| 1.9 | Unit tests | `+ tests/unit/core/test_*.py` | `pytest tests/unit/core` 100% on `errors` + `imageguard` | 1.* |

---

## Phase 2 — Broker - Done

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 2.1 | Celery factory | `+ packages/broker/celery_app.py` (conf from [05 §Celery config](./05-worker-runtime.md#celery-configuration)) | `celery_app.send_task("noop")` enqueues to Redis | 1.* |
| 2.2 | Distributed lock | `+ packages/broker/locks.py` (`redis_lock(key, ttl)`) | TTL release tested via fakeredis | 2.1 |
| 2.3 | Lua bucket script | `+ packages/broker/ratelimit.lua` | EVAL returns `[allowed, remaining, burst]` | 2.1 |
| 2.4 | Bucket Python wrapper | `+ packages/broker/ratelimit.py` (`RateLimiter`) | concurrent acquires across N tasks return correct allowed-count | 2.3 |
| 2.5 | Pub/Sub helpers | `+ packages/broker/pubsub.py` (`publish_event`, `subscribe`) | publish→subscribe round-trip in fakeredis | 2.1 |
| 2.6 | W3C trace propagation | `+ packages/broker/trace.py` (`submit_with_trace`, `extract_from_payload`) | `traceparent` survives Celery payload | 2.1, 1.6 |
| 2.7 | Unit tests | `+ tests/unit/broker/test_*.py` | fakeredis suite green | 2.* |
| 2.8 | Integration test | `+ tests/integration/broker/test_roundtrip.py` (real Redis from compose) | submit → consume → state-of-the-world checked | 0.4, 2.* |

---

## Phase 3 — Storage + DB - Done

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 3.1 | Storage ABC | `+ packages/storage/base.py` (`StorageBackend`, `IORef`) | abstract methods declared | 1.* |
| 3.2 | Key layout helpers | `+ packages/storage/keys.py` (`upload_key`, `artifact_key`) | unit-tested templates | 3.1 |
| 3.3 | S3 backend | `+ packages/storage/s3.py` (boto3, presign PUT/GET) | upload + download round-trip against MinIO | 3.1, 0.4 |
| 3.4 | Local FS backend (tests) | `+ packages/storage/local.py` | round-trip tests pass | 3.1 |
| 3.5 | DB models | `+ packages/db/models.py` (`Job`, `JobEvent`, `Artifact`, `WebhookDelivery`) per [02 §ER diagram](./02-data-model.md#er-diagram-local-profile) | `Base.metadata.create_all` on test SQLite passes | 1.1 |
| 3.6 | Async session factory | `+ packages/db/session.py` (`async_session_maker`) | `async with session_factory() as s: …` works | 3.5 |
| 3.7 | Jobs repo + state guards | `+ packages/db/repositories/jobs.py` (`transition`, `claim`, `insert_queued`) | `UPDATE … RETURNING` race test: only one of two concurrent claims wins | 3.6 |
| 3.8 | Artifacts + webhook repos | `+ packages/db/repositories/artifacts.py`, `+ packages/db/repositories/webhook.py` | basic CRUD covered | 3.6 |
| 3.9 | First Alembic migration | `+ packages/db/migrations/env.py`, `+ packages/db/migrations/versions/0001_local_schema.py` | `alembic upgrade head` against compose Postgres | 3.5, 0.4 |
| 3.10 | Static API key validator | `+ packages/security/apikey.py` (`verify_local_key`, constant-time) | wrong key raises `Unauthorized` | 1.3 |
| 3.11 | Presign wrapper | `+ packages/security/presign.py` | thin call to storage; tested with fake clock | 3.3 |
| 3.12 | Seed scripts | `+ scripts/seed_dev_data.py`, `+ scripts/init_minio.py` | `python scripts/init_minio.py` creates buckets idempotently | 3.3, 3.9 |
| 3.13 | Unit tests | `+ tests/unit/{db,storage,security}/` | repo state-guard test covers loser path | 3.* |
| 3.14 | Integration tests | `+ tests/integration/{db,storage}/` | green vs compose | 3.*, 0.4 |

---

## Phase 4 — I/O + Models + Tasks

### 4a — Typed I/O classes

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 4.1 | I/O base | `+ packages/io/base.py` (`InputBase`, `OutputBase`, `IORef`, `ArtifactSpec`) | abstract `validate_with_caps` + `serialize_artifacts` defined | 1.2 |
| 4.2 | Image inputs | `+ packages/io/inputs/image.py` (`ImageInput`, `ImageTextInput`, `ImagePointInput`, `ImageBoxInput`, `TextQuery`, `Point2DLabel`, `BBox2D`) | Pydantic round-trip tested | 4.1 |
| 4.3 | Multi-view inputs | `+ packages/io/inputs/multiview.py` (`MultiViewImageInput`, `ImageView`, `CameraHints`) | `views` length bound 2..16 enforced | 4.1 |
| 4.4 | Reserved inputs | `+ packages/io/inputs/{video,imu,pointcloud,action}.py` | classes exist but unreferenced | 4.1 |
| 4.5 | Mask outputs | `+ packages/io/outputs/mask.py` (`MaskLabelOutput`, `SegmentationMapOutput`) | `serialize_artifacts` returns expected `ArtifactSpec` list | 4.1 |
| 4.6 | Depth output | `+ packages/io/outputs/depth.py` (`DepthMapOutput`) | meta JSON sidecar emitted | 4.1 |
| 4.7 | Camera + pointcloud + composite | `+ packages/io/outputs/{camera,pointcloud,composite}.py` (`CameraParametersOutput`, `PointCloudOutput`, `MultiViewDepthOutput`) | composite serializes per-view + cameras + optional pointcloud | 4.5, 4.6 |
| 4.8 | Reserved outputs | `+ packages/io/outputs/{bbox,classification,text,pose}.py` | classes exist but unreferenced | 4.1 |
| 4.9 | Registry filter | `+ packages/io/registry.py` (`IORegistry.visible_inputs/outputs`) | given only SAM3 caps, returns 4 inputs + 1 output | 4.2..4.8 |
| 4.10 | I/O unit tests | `+ tests/unit/io/test_visibility.py`, `+ test_validators.py`, `+ test_artifacts.py` | green | 4.9 |

### 4b — Adapter contract + warm-pool

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 4.11 | Adapter contract | `+ packages/models/base.py` (`TypedCapability`, `ModelCapabilities`, `ModelAdapter` Protocol) | mypy passes on a fake adapter | 1.1, 4.1 |
| 4.12 | Registry | `+ packages/models/registry.py` (`@register_model`) | duplicate `model_id` raises | 4.11 |
| 4.13 | WarmPool + OOM guard | `+ packages/models/device.py` (`WarmPool`, `oom_guard` ctx mgr) | LRU eviction test with fake adapter; preflight + runtime OOM raise | 4.11, 2.2 |

### 4c — SAM3

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 4.14 | Submodule | `~ .gitmodules` add `third_party/sam3` (facebookresearch/sam3) | `git submodule update --init` clean | 0.5 |
| 4.15 | Weights pin | `+ packages/models/sam3/weights.py` (HF revision + sha256) | `weights.fetch()` downloads + verifies | 4.14 |
| 4.16 | Adapter | `+ packages/models/sam3/adapter.py` (port from `app/sam3_service.py`) | mocked-torch unit test passes; declares 3 `TypedCapability` rows | 4.11, 4.15 |

### 4d — Depth Anything 3

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 4.17 | Submodule | `~ .gitmodules` add `third_party/depth_anything_v3` (ByteDance-Seed/depth-anything-3) | clean clone | 0.5 |
| 4.18 | Weights pin | `+ packages/models/depth_anything_v3/weights.py` | sha256 verified | 4.17 |
| 4.19 | Monocular path | `+ packages/models/depth_anything_v3/adapter.py` (monocular only) | `infer([ImageInput])` returns `[DepthMapOutput]` | 4.11, 4.18, 4.6 |
| 4.20 | Multi-view path | `~ packages/models/depth_anything_v3/adapter.py` (add multi-view) | `infer([MultiViewImageInput])` returns `[MultiViewDepthOutput]` | 4.19, 4.7 |

### 4e — Tasks

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 4.21 | TaskSpec base | `+ packages/tasks/base.py`, `+ packages/tasks/registry.py` | `TaskSpec` + register decorator | 4.11 |
| 4.22 | Segmentation tasks | `+ packages/tasks/segmentation/{text,point,box}.py` | three `TaskSpec` subclasses; resolve to SAM3 caps | 4.16, 4.21 |
| 4.23 | Regularize port | `+ packages/tasks/segmentation/post/regularize.py` (verbatim from `app/regularization.py`) | unit-test parity vs legacy | 4.22 |
| 4.24 | Depth tasks | `+ packages/tasks/depth/{monocular,multiview}.py` | two `TaskSpec` subclasses; resolve to DA3 caps | 4.19, 4.20, 4.21 |
| 4.25 | Micro-batcher | `+ packages/tasks/batching.py` | N concurrent submits coalesce into ≤ ⌈N/max_batch⌉ adapter calls | 4.21 |
| 4.26 | Phase 4 unit tests | `+ tests/unit/{models,tasks}/` | green; capability search returns expected adapter | 4.* |

---

## Phase 5 — Worker

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 5.1 | Worker entry | `+ services/worker/main.py` (`celery -A services.worker.main worker`) | `celery -A services.worker.main inspect ping` returns | 4.* |
| 5.2 | Eligible queues CLI | `+ services/worker/eligible_queues.py` | prints expected queues for `MODELS_ENABLED=sam3,depth_anything_v3` on an A100 | 5.1 |
| 5.3 | Bootstrap | `+ services/worker/bootstrap.py` (preload models when `WORKER_PRELOAD=true`) | readiness gates until both adapters loaded | 5.1, 4.13 |
| 5.4 | Heartbeat thread | `+ services/worker/heartbeat.py` | `heartbeat_at` advances every 10 s during a task | 5.1, 3.7 |
| 5.5 | Runner | `+ services/worker/runner.py` (state guards, OOM, retry, cancel) | mocked-adapter run mutates DB row through full state path | 5.4, 4.* |
| 5.6 | Signals | `+ services/worker/signals.py` (SIGTERM drain, SIGUSR1 cancel handler) | unit test trips `CancelCheck` | 5.5, 1.8 |
| 5.7 | OOM guard re-export | `+ services/worker/oom_guard.py` | re-exports from `packages.models.device` | 5.5 |
| 5.8 | Webhook dispatcher | `+ services/worker/webhooks/dispatcher.py` (Celery task on non-GPU queue) | HMAC + retry tested | 5.1, 1.3 |
| 5.9 | Beat schedule + reconciler | `+ services/worker/beat/schedule.py`, `+ services/worker/beat/reconciler.py` | reconciler marks orphaned `RUNNING > 90s` as `FAILED` | 5.4, 3.7 |
| 5.10 | Bake-weights script | `+ services/worker/bake_weights.py` | `python -m services.worker.bake_weights --models sam3` populates HF cache | 4.15, 4.18 |
| 5.11 | `Dockerfile.worker` | `+ Dockerfile.worker` (CUDA 12.8.1) | `docker build -f Dockerfile.worker .` succeeds | 5.1, 5.10 |
| 5.12 | Acks-late kill test | `+ tests/integration/worker/test_acks_late.py` | kill mid-job → redelivery → exactly-one finalizer | 5.5, 0.4 |
| 5.13 | Cancel running test | `+ tests/integration/worker/test_cancel_running.py` | SIGUSR1 → `CANCELED`, worker still healthy | 5.6 |
| 5.14 | OOM bump test | `+ tests/integration/worker/test_oom_bump.py` | RuntimeOOM → retry with bumped `gpu_class` | 5.5, 4.13 |
| 5.15 | E2E worker loop | `+ tests/e2e/test_worker_loop.py` (SAM3 + DA3 mono + DA3 multi-view) | three jobs end in `SUCCEEDED` with correct artifacts | 5.11, 4.16, 4.20 |

---

## Phase 6 — API

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 6.1 | App factory + lifespan | `+ services/api/main.py` | `uvicorn services.api.main:app` boots | 1.*, 3.* |
| 6.2 | Deps | `+ services/api/deps.py` (`get_principal`, `get_storage`, `enforce_rate_limit`) | static `X-API-Key` validates | 3.10, 2.4 |
| 6.3 | Middleware | `+ services/api/middleware/{request_id,logging,error_handler,security_headers}.py` | request_id round-trips in headers and logs | 6.1, 1.5 |
| 6.4 | Health router | `+ services/api/routers/health.py` (`/v1/health/{live,ready,version}`) | `/v1/health/ready` returns 503 if Redis down | 6.1 |
| 6.5 | I/O types router | `+ services/api/routers/io_types.py` (`GET /v1/io/types`) | returns exactly the 5 declared inputs + 4 declared outputs | 4.9, 6.1 |
| 6.6 | Models router | `+ services/api/routers/models.py` (`GET /v1/models`) | reads readiness via Redis SETEX heartbeats | 6.1 |
| 6.7 | Uploads router | `+ services/api/routers/uploads.py` (presigned PUT + multipart complete) | round-trip with MinIO works | 6.1, 3.11 |
| 6.8 | Tasks router | `+ services/api/routers/tasks.py` (`POST /v1/tasks/{task_type}`, idempotency) | idempotency replay returns same `job_id` with header | 6.1, 4.21, 3.7 |
| 6.9 | Jobs router | `+ services/api/routers/jobs.py` (list, single, artifacts, SSE, cancel) | SSE stream emits state events from Redis pub/sub | 6.1, 5.9, 2.5 |
| 6.10 | OpenAPI emit CLI | `+ services/api/openapi.py` (`--out openapi.json`) | file produced; valid OpenAPI 3.1 | 6.4..6.9 |
| 6.11 | `Dockerfile.api` | `+ Dockerfile.api` | image builds; CMD boots | 6.1 |
| 6.12 | E2E happy paths | `+ tests/e2e/test_segmentation_text.py`, `+ test_depth_monocular.py`, `+ test_depth_multiview.py` | all three submit→poll→download green | 6.7, 6.8, 6.9, 5.15 |
| 6.13 | E2E negative paths | `+ tests/e2e/test_negative.py` (auth, ratelimit, idempotency conflict, no-overshoot) | 401 / 429 / 409 / filtered `/v1/io/types` all asserted | 6.5, 6.8 |

---

## Phase 7 — Infra (compose, ci, sphinx, sdk)

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 7.1 | Final `docker-compose.yml` | `~ infra/compose/docker-compose.yml` (api + worker-gpu-{0,1} + worker-cpu + beat + obs stack; profiles `cpu`/`gpu1`/`gpu2`) | `COMPOSE_PROFILES=gpu1 docker compose up -d` brings everything healthy | 5.11, 6.11 |
| 7.2 | Bootstrap (Linux) | `~ scripts/bootstrap_dev.sh` | fresh clone → e2e green on Ubuntu | 7.1 |
| 7.3 | Bootstrap (Windows) | `~ scripts/bootstrap_dev.ps1` | fresh clone → e2e green on Win11 + WSL2 | 7.1 |
| 7.4 | Prometheus scrape | `+ infra/prometheus/scrape.yaml` | scrapes API + worker + celery-exporter | 7.1 |
| 7.5 | Grafana dashboards | `+ infra/grafana/dashboards/{api_health,queue,inference,gpu,local}.json` | dashboards load with non-zero data after a job | 7.4 |
| 7.6 | OTel collector cfg | `+ infra/otel/collector-config.yaml` | OTLP from API + worker arrives at Prometheus | 7.1 |
| 7.7 | CI: lint + unit + integration | `~ .github/workflows/ci.yml` | green on `v2` PR | 6.* |
| 7.8 | CI: openapi-diff gate | `~ .github/workflows/ci.yml` (job) | breaking diff fails PR | 6.10 |
| 7.9 | Release: images + SDKs | `+ .github/workflows/release.yml` (build images, openapi-generator, publish PyPI + npm) | dry-run on test tag publishes both SDKs | 7.7 |
| 7.10 | Docs: Sphinx config | `+ docs/conf.py`, `+ docs/index.rst` (autodoc `packages.io`, route inventory from openapi.json) | `sphinx-build docs docs/_build` succeeds | 4.10, 6.10 |
| 7.11 | Docs: publish workflow | `+ .github/workflows/docs.yml` (GH Pages) | tag publishes site | 7.10 |
| 7.12 | README polish | `~ README.md` (architecture diagram, demo gif, install, quickstart, link to `upgrade/` and `enterprise/`) | reader can clone → run in 5 commands | 7.2, 7.3 |

---

## Phase 8 — Hardening

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 8.1 | Smoke load profile | `+ tests/load/smoke.js` (k6) | runs against compose; baseline P95 captured | 7.1 |
| 8.2 | Sustained load profile | `+ tests/load/sustained.js` | 30-min run; dashboards show meaningful data | 8.1, 7.5 |
| 8.3 | Chaos recipe | `+ scripts/chaos/pumba_runner.sh` | random worker kill under load → zero lost jobs | 5.12 |
| 8.4 | Coverage backfill | `~ tests/unit/**`, `~ pyproject.toml` (`fail_under=80`) | `pytest --cov` ≥ 80% | 6.* |
| 8.5 | SDK round-trip tests | `+ tests/sdk/{py,ts}/test_roundtrip.*` | both SDKs run a full job | 7.9 |
| 8.6 | Alert rules (local-baseline) | `+ infra/prometheus/rules/{api,queue,worker}.yaml` | every alert has a `runbook_url` annotation; lint step passes | 7.4 |
| 8.7 | Runbook stubs | `+ infra/runbooks/{api-availability,queue-stuck,worker-no-heartbeats}.md` | each runbook has the 6 required sections | 8.6 |

---

## Phase 9 — Cutover

| ID | Step | Files | Exit | Depends |
|---|---|---|---|---|
| 9.1 | Compatibility shim | `+ services/api/routers/legacy.py` (`/segment-from-path`, `/segment-from-upload`; map to new pipeline) | legacy curl still works against v2 | 6.8 |
| 9.2 | Sunset/Deprecation headers | `~ services/api/routers/legacy.py` | every legacy response carries the headers per RFC 8594 | 9.1 |
| 9.3 | Legacy-route telemetry | `~ packages/core/telemetry.py` (`legacy_route_requests_total` counter) | dashboard panel shows traffic by `User-Agent` | 9.1, 7.4 |
| 9.4 | Wait window (≥ 1 minor release / 60 d) | n/a | counter reaches 0 for 7 d | 9.3 |
| 9.5 | Delete legacy code | `- app/`, `- main.py`, `- app/router.py`, `- app/job_router.py`, `- app/jobs.py`, `- app/sam3_service.py`, `- app/regularization.py`, `- app/schemas.py` | `rg "from app"` returns nothing | 9.4 |
| 9.6 | Tag v2.0.0 | `~ README.md`, `~ CHANGELOG.md`, git tag | `gh release create v2.0.0` published | 9.5, 7.* |

---

## Estimated effort (solo, full-time-equivalent)

| Phase | Steps | Days (rough) |
|---|---|---|
| 0 | 8 | 1–2 |
| 1 | 9 | 2–3 |
| 2 | 8 | 2–3 |
| 3 | 14 | 4–6 |
| 4 | 26 | 8–12 |
| 5 | 15 | 5–8 |
| 6 | 13 | 5–8 |
| 7 | 12 | 4–6 |
| 8 | 7 | 3–5 |
| 9 | 6 | 1–2 |
| **Total** | **118** | **≈ 35–55 days** |

For weekend / part-time work, multiply by 3–4×. The **portfolio-minimum** path (skip 8.4–8.7, skip 9.*, defer SDK to TS later) compresses to ≈ 25–35 full-time-equivalent days.

## How to use this list

1. Pick the next undone step where all `Depends` are green.
2. Open a PR titled `phase X.Y — <step name>`.
3. Land it with: code + test + doc update if reality diverged from spec.
4. Tick it off. Re-evaluate dependencies.
5. Never skip a step's exit check — that is what protects later phases.

## Enterprise overlay steps (out of band)

Each [`enterprise/`](../enterprise/) document carries its own internal step list when implemented. Treat them as additive PRs against a green local profile; do **not** interleave them with the local phases above.
