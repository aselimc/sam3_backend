# 09 — Phase-by-Phase Integration Plan

This document is the executable plan for the **local profile** of v2. Each phase is a single PR (or a small chain) into `v2`, has clear scope, exit criteria, verification commands, and a rollback paragraph. Phases are sized to be reviewable in a single sitting (≤ 1500 LoC of net change excluding lockfiles).

A phase **must not** start until all prior phases are merged and green on `v2`.

Enterprise-track work (multi-tenancy, JWT/OIDC, Helm/K8s, supply-chain hardening, multi-region) lives in the [`enterprise/`](../enterprise/) folder. It is intentionally not interleaved with the phases below — each enterprise document is a self-contained overlay an adopter can apply on top of a green local profile, in any order.

## Phase ordering rationale

```
0  scaffolding        — no functional change, no risk
1  core kernel        — leaves of the dependency graph
2  broker + redis     — depends on core
3  storage + db       — depends on core; introduces persistence
4  io types + adapters + tasks   — depends on core; ports SAM3, scaffolds DA3
5  worker             — wires 2+3+4
6  api                — wires 1+2+3+4 + routers
7  infra (compose, ci, sphinx + sdk gen)
8  hardening          — observability, perf, chaos
9  cutover            — deprecation of legacy app/, v2.0.0
```

Phases 5 and 6 can run in parallel once 4 lands; everything else is serial.

---

## Phase 0 — Branch & scaffolding

**Goal**: stand up the new directory layout without touching legacy code.

### Scope

- `git checkout -b v2` (already done).
- Create directory skeleton: `packages/{core,storage,broker,db,security,models,tasks,io}/`, `services/{api,worker}/`, `infra/compose/`, `tests/{unit,integration,e2e}/`, `scripts/`.
- Add `third_party/` as the submodule root with placeholders for `sam3` and `depth_anything_v3`.
- Create `pyproject.toml` workspaces with `[project.optional-dependencies]` groups: `api`, `worker`, `dev`, `docs`.
- Add `.env.example`, `infra/compose/docker-compose.yml` (stub: redis, postgres, minio only — no app yet).
- Add CI matrix entry for `v2`.
- Leave `app/`, `main.py`, current `tests/` untouched.

### Exit criteria

- `uv sync --group api --group worker --group dev` resolves cleanly.
- `COMPOSE_PROFILES=cpu docker compose -f infra/compose/docker-compose.yml up -d redis postgres minio` works on Linux and Windows.
- Existing master tests still pass.

### Verification

```bash
uv sync --group api --group worker --group dev
uv run pytest -q                                       # legacy tests stay green
COMPOSE_PROFILES=cpu docker compose -f infra/compose/docker-compose.yml up -d redis postgres minio
docker compose -f infra/compose/docker-compose.yml ps  # all healthy
```

### Rollback

Delete the branch. No production impact.

---

## Phase 1 — Core kernel

**Goal**: framework-free shared code that everything else imports.

### Scope

- `packages/core/types.py` — `TaskType`, `JobState`, `GpuClass`. (`Modality` enum is **not** added; replaced by typed I/O classes in Phase 4.)
- `packages/core/schemas.py` — `VersionedModel`.
- `packages/core/errors.py` — `AppError` hierarchy with HTTP mapping.
- `packages/core/config.py` — `pydantic-settings`.
- `packages/core/logging.py` — JSON log sink with the schema from `07-observability.md`.
- `packages/core/telemetry.py` — Prometheus registry + OTel tracer factory.
- `packages/core/imageguard.py` — global `MAX_IMAGE_PIXELS` set at import.

### Exit criteria

- 100% unit test coverage on `errors.py` and `imageguard.py`.
- `python -c "from packages.core import logging; logging.configure(); …"` emits one valid JSON line.

### Verification

```bash
uv run pytest tests/unit/core -q
```

### Rollback

Revert PR. Nothing imports it yet.

---

## Phase 2 — Broker + Celery app

**Goal**: Redis-backed Celery, rate limiter, locks, pubsub.

### Scope

- `packages/broker/celery_app.py` — Celery factory, conf from `05-worker-runtime.md`.
- `packages/broker/ratelimit.py` + `ratelimit.lua` — Lua token bucket, fakeredis-backed tests.
- `packages/broker/locks.py` — `with redis_lock("key", ttl=…)`.
- `packages/broker/pubsub.py` — publish/subscribe helpers for job events.
- `packages/broker/trace.py` — submit-with-trace + worker extract.

### Exit criteria

- Unit tests against `fakeredis` for ratelimit (concurrent acquires) and locks (TTL release).
- Submitting a no-op task to a real Redis (compose) and consuming it from a smoke worker round-trips.

### Verification

```bash
docker compose -f infra/compose/docker-compose.yml up -d redis
uv run pytest tests/unit/broker -q
uv run pytest tests/integration/broker -q
```

### Rollback

Revert PR.

---

## Phase 3 — Storage + DB

**Goal**: persistent state and object store abstractions.

### Scope

- `packages/db/models.py` — SQLAlchemy 2.x models matching `02-data-model.md` **local schema** (jobs, job_events, artifacts, webhook_deliveries; no tenants/users/api_keys/audit).
- `packages/db/session.py` — async engine + session factory.
- `packages/db/repositories/{jobs,artifacts,webhook}.py`.
- `packages/db/migrations/0001_local_schema.py` — first Alembic migration.
- `packages/storage/base.py` + `s3.py` + `local.py` + `keys.py`.
- `packages/security/{apikey,presign}.py` — static `X-API-Key` validator + presign wrapper. JWT/OIDC/argon2 deferred to enterprise.
- `scripts/seed_dev_data.py` and `scripts/init_minio.py`.

### Exit criteria

- `alembic upgrade head` succeeds against compose Postgres.
- Repository tests demonstrate: state-machine guard works (loser sees zero rows), idempotency unique index rejects duplicates.
- S3 backend integration tests pass against MinIO.

### Verification

```bash
docker compose -f infra/compose/docker-compose.yml up -d postgres minio
uv run alembic upgrade head
uv run pytest tests/unit/{db,storage,security} -q
uv run pytest tests/integration/{db,storage} -q
```

### Rollback

`alembic downgrade base`. Drop schema. Revert PR.

---

## Phase 4 — I/O types + Models + Tasks

**Goal**: typed I/O hierarchy, adapter contract, registry, GPU device probe, port SAM3, scaffold DA3.

### Scope

- `packages/io/` — `base.py`, `inputs/{image,multiview,video,imu,pointcloud,action}.py`, `outputs/{mask,bbox,classification,text,pose,depth,camera,pointcloud,composite}.py`, `registry.py`. See [04a-io-types.md](./04a-io-types.md). At v2.0 the **declared** classes are: `ImageInput`, `ImageTextInput`, `ImagePointInput`, `ImageBoxInput`, `MultiViewImageInput`; `MaskLabelOutput`, `DepthMapOutput`, `MultiViewDepthOutput`, `CameraParametersOutput`, `PointCloudOutput`. The remaining classes (Video/Imu/etc.) are scaffolded in code but unreferenced by any adapter and therefore filtered out by the registry.
- `packages/models/base.py`, `registry.py`, `device.py` (warm-pool, OOM guard).
- `packages/models/sam3/adapter.py` — port `app/sam3_service.py` to the new `infer(batch: list[InputBase]) -> list[OutputBase]` shape.
- `packages/models/sam3/weights.py` — pinned HF revision, sha256 verify.
- `packages/models/depth_anything_v3/adapter.py` — full implementation (monocular + multi-view), wrapped over the upstream submodule.
- `packages/models/depth_anything_v3/weights.py`.
- `third_party/sam3/` and `third_party/depth_anything_v3/` submodules added.
- `packages/tasks/base.py`, `registry.py`.
- `packages/tasks/segmentation/{text,point,box}.py`.
- `packages/tasks/segmentation/post/regularize.py` — verbatim port of `app/regularization.py`.
- `packages/tasks/depth/{monocular,multiview}.py`.
- Per-queue micro-batcher: `packages/tasks/batching.py`.

### Exit criteria

- Adapter unit tests pass with mocked torch/sam3 (mirror legacy `tests/conftest.py` pattern).
- A capability search by `(task_type)` returns the expected adapter (SAM3 for segmentation.*, DA3 for depth.*).
- I/O registry visibility test: with only SAM3 enabled, `MultiViewImageInput` and `DepthMapOutput` are filtered out.
- Batching unit test: N concurrent submits coalesce into ≤ ⌈N / max_batch_size⌉ adapter calls.

### Verification

```bash
uv run pytest tests/unit/{io,models,tasks} -q
```

### Rollback

Revert PR. The new packages are not yet imported by any service.

---

## Phase 5 — Worker service

**Goal**: end-to-end runner. First time real GPU work flows through Celery.

### Scope

- `services/worker/main.py`, `bootstrap.py`, `runner.py`, `signals.py`, `oom_guard.py`, `eligible_queues.py`, `bake_weights.py`, `heartbeat.py`.
- Reconciler beat schedule under `services/worker/beat/`.
- Cancellation handler (SIGUSR1).
- Webhook dispatcher task on a non-GPU queue.
- Per-GPU device pinning via `CUDA_VISIBLE_DEVICES`; the bootstrap script writes this from detected GPU index.

### Exit criteria

- Submitting a SAM3 task via the broker (no API yet — a tiny test client) runs the adapter, writes mask artifacts to MinIO, transitions the DB row.
- Submitting a DA3 monocular task end-to-end produces a `DepthMapOutput` with both `depth.png` and `depth_meta.json`.
- Submitting a DA3 multi-view task with 3 views produces a `MultiViewDepthOutput` with per-view depth + camera params.
- Kill-mid-job test: redelivery happens, exactly-one-finalizer holds.
- Cancel-running test: SIGUSR1 path leaves the worker healthy for the next job.
- Two-GPU test (Linux only, optional): two worker processes serve disjoint `CUDA_VISIBLE_DEVICES` and consume the same queue without interfering.

### Verification

```bash
COMPOSE_PROFILES=gpu1 docker compose -f infra/compose/docker-compose.yml up -d
uv run pytest tests/integration/worker -q
uv run pytest tests/e2e/test_worker_loop.py -q
```

### Rollback

Stop the worker deployment; the API is not yet wired to Celery.

---

## Phase 6 — API service

**Goal**: the public surface defined in `03-api-spec.md`.

### Scope

- `services/api/main.py`, `deps.py`, middleware (request_id, logging, error_handler, security_headers).
- Routers: `uploads`, `tasks`, `jobs`, `models`, `io_types`, `health`. (No `auth` router in local profile — `X-API-Key` validator is a `Depends`.)
- OpenAPI emit + `openapi-diff` CI gate.
- SSE endpoint sourced from Redis pub/sub.
- Webhook signature validation example client (in `scripts/webhook_receiver.py`).
- Multipart upload completion endpoint.

### Exit criteria

- Full e2e test passes against compose:
  1. `POST /v1/uploads` returns presigned PUT.
  2. `PUT` to MinIO succeeds.
  3. `POST /v1/tasks/segmentation.text` → poll → `SUCCEEDED` → download masks.
  4. `POST /v1/tasks/depth.monocular` → poll → `SUCCEEDED` → download depth.
  5. `POST /v1/tasks/depth.multiview` (3 views) → poll → `SUCCEEDED` → download per-view + cameras.
- Auth negative tests pass (missing/wrong `X-API-Key` → 401).
- Rate-limit test: exceeding the bucket returns `429` with `Retry-After`.
- Idempotency replay test passes.
- `GET /v1/io/types` returns exactly the 5 declared inputs and 4 declared outputs (no overshooting).

### Verification

```bash
COMPOSE_PROFILES=gpu1 docker compose -f infra/compose/docker-compose.yml up -d
uv run pytest tests/e2e -q
```

### Rollback

Revert PR. Workers continue without the API.

---

## Phase 7 — Infra (compose, ci, sphinx, sdk)

**Goal**: the artifacts that let someone else clone the repo and run it.

### Scope

- Finalize `infra/compose/docker-compose.yml` (all profiles: `cpu`, `gpu1`, `gpu2`).
- `Dockerfile.api`, `Dockerfile.worker` per `08-infra-and-cicd.md`.
- `scripts/bootstrap_dev.{ps1,sh}` — cross-platform bootstrap.
- Prometheus scrape config + Grafana dashboards (5 of them).
- `release.yml`: build images, generate Python + TS SDKs, publish.
- `docs.yml`: build Sphinx, publish to GitHub Pages.

### Exit criteria

- Fresh clone → `bootstrap_dev` → `compose up` → e2e green on Windows + Ubuntu.
- `release.yml` produces published `sam3-client-py` and `@org/sam3-client` on a test tag.
- `docs.yml` produces a Sphinx site that includes both route docs and typed I/O class docs.

### Verification

```bash
# Fresh clone test (manual)
git clone …; cd …; ./scripts/bootstrap_dev.sh
COMPOSE_PROFILES=gpu1 docker compose -f infra/compose/docker-compose.yml up -d
uv run pytest tests/e2e -q

# Release dry run
gh workflow run release.yml --ref test-tag
```

### Rollback

Revert PR. Nothing is yet promoted to a public registry.

---

## Phase 8 — Hardening

**Goal**: close the gap between "it works" and "I trust it".

### Scope

- Full observability rollout (5 dashboards, alert rules, runbook stubs).
- Chaos test recipe (`pumba` killing workers under load).
- `k6` smoke + sustained load profiles committed; perf budget recorded.
- SDK round-trip tests against the running stack.
- Backfill remaining tests to coverage target (`fail_under=80`).
- Sphinx docs reviewed for completeness.

### Exit criteria

- Dashboards show meaningful data after a 30-min synthetic load.
- Chaos test passes: random kill of any worker pod yields zero lost jobs.
- SDK round-trip test passes for both Python and TypeScript clients.
- README contains a working demo (animated GIF of submit→result for both SAM3 and DA3).

### Verification

```bash
k6 run tests/load/sustained.js                      # 30-min profile
scripts/chaos/pumba_runner.sh                       # 10-min chaos run
uv run pytest tests/sdk -q
```

### Rollback

Per-feature; each subitem can land as its own small PR.

---

## Phase 9 — Cutover

**Goal**: legacy paths retired; v2 becomes the default.

### Scope

- Compatibility shim: `services/api/routers/legacy.py` exposes `/segment-from-path` and `/segment-from-upload` paths, but they enqueue via the new pipeline. `Sunset` and `Deprecation` headers are returned with every response.
- After one minor release (or a calendar window agreed in advance — default 60 d), legacy routes return `410 Gone`.
- Delete `app/`, `main.py`, `app/schemas.py`, `app/jobs.py`, `app/router.py`, `app/job_router.py` in the cutover commit.
- Update `README.md`. Tag `v2.0.0`. Set `v2` as the default branch.

### Exit criteria

- All clients identified by `User-Agent`/IP analysis on legacy routes have migrated (operator dashboard panel: `legacy_route_requests_total`).
- Fresh-clone bootstrap continues to work without the legacy shim mounted.

### Verification

```bash
# Confirm zero traffic to legacy routes for 7 d
sum(increase(http_requests_total{route=~"/segment-.*"}[7d])) == 0
```

### Rollback

Revert the deletion commit; re-enable the shim. Issue a `v2.0.1` patch announcement.

---

## Cross-phase rules

- Every phase ships its own integration tests; no "we'll add tests later".
- Every phase updates the relevant document in `upgrade/` if reality diverged from the design. The document is the spec; if code disagrees, either the doc is wrong or the code is wrong — pick one in the PR.
- Migrations are forward-compatible-only. Add columns first, deploy, then in a follow-up phase make them NOT NULL.
- Feature flags (env-driven) are used to dark-launch risky features (batching, multi-class routing). Default off until measured.
- The I/O class registry is the gate for "no overshooting". Adding a class without an adapter that uses it is fine in code but invisible at the API surface — that is intentional, not a bug.

## Enterprise overlay (out of scope here)

The enterprise track is **not** sequenced into the phases above. An adopter starts from a green local profile and applies one or more of:

- [`enterprise/01-multi-tenancy-and-auth.md`](../enterprise/01-multi-tenancy-and-auth.md) — `tenants`, `users`, `api_keys`, `audit_events`, `tenant_quotas`; JWT + OIDC; per-key scopes.
- [`enterprise/02-kubernetes-and-helm.md`](../enterprise/02-kubernetes-and-helm.md) — Helm chart, KEDA, HPA, NetworkPolicy.
- [`enterprise/03-supply-chain-and-secrets.md`](../enterprise/03-supply-chain-and-secrets.md) — cosign, Trivy gating, Syft SBOM, External Secrets, KMS.
- [`enterprise/04-observability-at-scale.md`](../enterprise/04-observability-at-scale.md) — Loki + Tempo + sampling tiers + SLO burn alerts.
- [`enterprise/05-multi-region-and-ha.md`](../enterprise/05-multi-region-and-ha.md) — Postgres HA, Redis Sentinel/cluster, multi-region S3, DR runbook.

Each enterprise document specifies its own additive Alembic migrations and Helm value overrides. None of them is required to ship v2.0 of the local profile.
