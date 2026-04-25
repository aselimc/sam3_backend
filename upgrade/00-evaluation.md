# 00 — Evaluation of `PLAN.md`

`PLAN.md` is a strong vision document. It picks the right axes (Celery + Redis + Postgres + S3 + Helm), the right tier split (API / Broker / Worker / Storage), and the right pluggability boundary (model adapter + task layer). Building it as written would already be a major step up from `master`.

This document records what `PLAN.md` does *not* answer, so the rest of this folder can answer it. Each weakness is followed by the fix and a pointer to the document that owns the detail.

## Categories

- **A — Correctness gaps**: things that would compile but be wrong or unsafe in production.
- **B — Missing capabilities**: features the architecture implies but does not specify.
- **C — Operational gaps**: nothing breaks at first deploy, but on-call cannot do their job.
- **D — Process gaps**: phasing, rollback, deprecation, SDK story.

## A — Correctness gaps

### A1. Two writers to `JobRecord` with no concurrency control

`PLAN.md` says the API writes `JobRecord` on submit and the worker mutates state on `RUNNING → SUCCEEDED|FAILED`. With `acks_late=True`, a single message can be redelivered: two workers may both think they own the job. Without optimistic concurrency, the loser overwrites the winner.

**Fix.** Single-writer rule: only the worker mutates state after enqueue. The API writes the initial `QUEUED` row and never updates it. Updates use `UPDATE … WHERE id=:id AND state IN (allowed_predecessors) RETURNING …`. See [02-data-model.md §State machine](./02-data-model.md#state-machine).

### A2. Idempotency on `POST /v1/tasks/{type}` is undefined

A client retry after a network blip enqueues the job twice. There is no `Idempotency-Key`.

**Fix.** Required `Idempotency-Key` header (UUID, scoped per principal, 24h TTL). API stores `(principal_id, key) → job_id` in Redis with NX-PX, returns the original `job_id` on replay. See [03-api-spec.md §Idempotency](./03-api-spec.md#idempotency).

### A3. Trace context does not propagate API → broker → worker

`PLAN.md` mentions OTel in `core/telemetry.py` but Celery does not propagate W3C trace headers automatically. Without explicit injection, distributed traces break at the queue boundary — exactly where they are most useful.

**Fix.** Inject `traceparent` and `tracestate` into Celery message headers on submit; extract on worker `task_prerun`. See [05-worker-runtime.md §Tracing](./05-worker-runtime.md#tracing).

### A4. Image-bomb defense is named, not specified

`PIL.Image.MAX_IMAGE_PIXELS` defaults are silently lifted by some libraries. The number must be set explicitly *before* `PIL` decodes anything — at process import, not per-request.

**Fix.** Hard limit set in `packages/core/imageguard.py`, imported eagerly. MIME sniffed via magic bytes (not `Content-Type`). HEIC/AVIF gated behind opt-in flag. See [03-api-spec.md §Input safety](./03-api-spec.md#input-safety).

### A5. `Image.open(...).convert("RGB")` on user upload happens in API

Decoding bytes in API memory means a malicious upload can OOM the API tier. The S3 presigned-PUT pattern is correct, but the legacy upload endpoints in `app/router.py` and `app/job_router.py` decode in-process. v2 must not retain that pattern.

**Fix.** API never decodes images. Workers fetch from S3 and decode under the OOM guard. See [03-api-spec.md §Uploads](./03-api-spec.md#uploads).

## B — Missing capabilities

### B1. No cancellation

There is no `DELETE /v1/jobs/{id}`. Long jobs cannot be aborted, queued jobs cannot be revoked.

**Fix.** Cancel endpoint with two semantics:
- *queued* → `app.control.revoke(task_id)` removes from queue, transitions to `CANCELED`.
- *running* → `revoke(task_id, terminate=True, signal='SIGUSR1')` (custom signal handler does GPU cleanup before exit).

See [03-api-spec.md §Cancel](./03-api-spec.md#cancel) and [05-worker-runtime.md §Cancellation](./05-worker-runtime.md#cancellation).

### B2. No webhooks; SSE alone forces clients to hold connections

`PLAN.md` mentions SSE via Redis pub/sub but no HTTP callback option. Many enterprise integrations cannot keep a long-lived connection.

**Fix.** Per-job optional `callback_url` with HMAC signing (`X-SAM3-Signature`), retried with exponential backoff, dead-lettered after N attempts. See [03-api-spec.md §Webhooks](./03-api-spec.md#webhooks).

### B3. No `GET /v1/jobs` list endpoint, no pagination

Clients cannot enumerate their jobs. Required for any UI.

**Fix.** Cursor-paginated list with filters (`state`, `task_type`, `created_after`). See [03-api-spec.md §Pagination](./03-api-spec.md#pagination).

### B4. No batching, despite `supports_batching` flag

`ModelCapabilities` has the field; nothing reads it. Batching is the dominant perf win for SAM3 / depth models.

**Fix.** Per-queue micro-batcher in the worker, configurable max-batch and max-wait, transparent to the task layer. See [04-model-and-tasks.md §Batching](./04-model-and-tasks.md#batching).

### B5. No GPU-class queue routing

`routes per TaskType` means every worker subscribes to every relevant task. A T4 cannot run a 24 GB model; submitting it to a mixed pool causes OOM, retry, OOM.

**Fix.** Queue naming `task.<type>.<gpu_class>` (e.g. `task.segmentation.text.a100_40g`). API resolves the right queue from `(model_id, requested_class)`; workers subscribe only to queues they can serve. See [04-model-and-tasks.md §Queue routing](./04-model-and-tasks.md#queue-routing).

### B6. No model swap policy on shared GPU

`device.py` mentions LRU eviction. Eviction policy must be deterministic: a slow stream of job-A interleaved with job-B will thrash. Cold-start latency (HF download minutes; load 30 s) becomes the dominant cost.

**Fix.** Models are pinned by `WORKER_ENABLED_MODELS`. LRU only between *eligible* models. Cold-start protected by the Redis lock per `model_id` plus a warm-pool sized via env. See [04-model-and-tasks.md §Warm-pool and eviction](./04-model-and-tasks.md#warm-pool-and-eviction).

### B7. Reconciler has no scheduler home

`PLAN.md` says "reconciler job marks orphaned RUNNING jobs as FAILED" but does not say where it runs.

**Fix.** Celery beat schedule, every 60 s. Idempotent. See [05-worker-runtime.md §Reconciler](./05-worker-runtime.md#reconciler).

### B8. Per-tenant overrides have no model

Rate-limit overrides are mentioned. Quota model is mentioned. There is no table.

**Fix.** `tenants.config_jsonb` for soft per-tenant config; explicit `tenant_quotas` table for hard counters. See [02-data-model.md §Tenants and quotas](./02-data-model.md#tenants-and-quotas).

### B9. Schema versioning

`/v1` is a path. Inner request/response schema evolution is undefined. Adding a field is fine; renaming or removing is not.

**Fix.** All public Pydantic models inherit `VersionedModel(version: Literal["1"])`. Breaking changes bump path *and* model. See [03-api-spec.md §Versioning](./03-api-spec.md#versioning).

### B10. Auth: only API key + JWT

Enterprise deployments need OIDC SSO. Service-to-service needs mTLS or workload identity (IRSA on EKS, Workload Identity on GKE).

**Fix.** OIDC token exchange endpoint scaffolded but optional. Workers use IAM roles, not static AWS keys, in cloud envs. See [06-storage-and-security.md §Identity](./06-storage-and-security.md#identity).

### B11. No SDK

Clients will paste curl into shells. We emit OpenAPI; we should emit a typed client.

**Fix.** OpenAPI 3.1 spec gated in CI; Python and TS clients generated by `openapi-generator-cli`, published as `sam3-client-py` and `@org/sam3-client`. See [10-use-cases.md §SDK examples](./10-use-cases.md#sdk-examples).

## C — Operational gaps

### C1. No SLOs

There is nothing for on-call to alert against.

**Fix.** Three SLOs: API availability, job success rate, P95 inference latency per `(model_id, gpu_class)`. See [07-observability.md §SLOs](./07-observability.md#slos).

### C2. Logs are not correlated across tiers

API logs `request_id`; worker logs `job_id`. They never meet.

**Fix.** Both log `request_id`, `job_id`, `principal_id`, `tenant_id`, `trace_id`. JSON sink. See [07-observability.md §Log schema](./07-observability.md#log-schema).

### C3. Bucket lifecycle is "open item"

S3 cost grows linearly with retention. PLAN punts.

**Fix.** Defaults written down: uploads expire 24 h after PUT if not referenced; artifacts default 30 d; per-tenant override allowed. See [06-storage-and-security.md §Lifecycle](./06-storage-and-security.md#lifecycle).

### C4. CI pipeline silent on supply chain

No SBOM, no image signing, no vuln scan, no registry policy.

**Fix.** Trivy scan blocks `HIGH+` on PR; `cosign sign` on release; Syft SBOM attached; pinned base images. See [08-infra-and-cicd.md §Supply chain](./08-infra-and-cicd.md#supply-chain).

### C5. Secrets distribution

`.env` and "K8s Secret" are listed. No rotation story, no Vault/External Secrets.

**Fix.** External Secrets Operator pulling from cloud secret manager; rotation runbook. See [06-storage-and-security.md §Secrets](./06-storage-and-security.md#secrets).

### C6. Windows + GPU dev parity

Docker Compose with NVIDIA on Windows requires WSL2 + CUDA toolkit. Plain Compose on Windows host does not see the GPU.

**Fix.** `docker-compose.yml` ships a CPU-only worker profile by default; GPU profile documented under WSL2. See [08-infra-and-cicd.md §Local dev](./08-infra-and-cicd.md#local-dev).

### C7. No load/chaos testing plan

E2E tests prove correctness, not throughput.

**Fix.** k6 load profile committed; chaos via `pumba` recipe (kill worker mid-job). See [09-phases.md §Phase 6 verification](./09-phases.md#phase-6).

## D — Process gaps

### D1. Phase 7 deletes legacy `app/` with no deprecation window

External callers will break.

**Fix.** v0 (legacy paths) kept on a thin compatibility shim for one minor version; `Sunset` header set; `410 Gone` after the window. See [09-phases.md §Phase 7](./09-phases.md#phase-7).

### D2. No phase-level rollback plan

If Phase 4 lands and is wrong, what undoes it?

**Fix.** Each phase entry in `09-phases.md` has a `Rollback` paragraph.

### D3. ADR-style decision log absent

`PLAN.md` lists "decisions confirmed with user" as a table but not the *reasons*. Future readers will not know why Celery beat RQ, why Postgres beat Mongo.

**Fix.** [11-risks-and-decisions.md](./11-risks-and-decisions.md) captures each decision in an ADR template.

## E — Scope and audience gaps (added in revision)

These were not weaknesses in `PLAN.md` per se — they are mismatches between the original document's audience (an enterprise team) and the actual primary user (a single developer building a portfolio + research repo). Listed here because they restructure the rest of this folder.

### E1. `Modality` string enum is too coarse for SDK generation

`Modality.IMAGE` cannot tell apart "a bare image", "an image + text prompt", "an image + IMU pair", "an image + paired pointcloud". Each of those is a different validator, a different storage layout, and a different SDK shape. Auto-generated clients built against a `Modality`-flat capability list end up with weakly-typed `dict` bags instead of typed inputs.

**Fix.** Replace `Modality` with a typed Pydantic class hierarchy under `packages/io/`. Adapters declare `TypedCapability(task, input_class, output_class)`; the public surface is filtered by an `IORegistry` to only the classes that at least one loaded adapter claims ("no overshooting"). See [04a-io-types.md](./04a-io-types.md) and [ADR-014](./11-risks-and-decisions.md#adr-014--typed-io-class-hierarchy-supersedes-modality-enum).

### E2. Adapter scaffolds for SAM2/depth-V2/pose/recon age without a maintainer

Four scaffolded adapters were proposed in `PLAN.md` to demonstrate pluggability. For a single-maintainer repo, scaffolds without a working backend become rot. Pluggability is better demonstrated by *two real adapters of different families* than by four stubs.

**Fix.** Ship two real adapters at v2.0:
- **SAM3** — segmentation (text/point/box).
- **Depth Anything 3** — monocular and multi-view depth.
SAM2 and the other scaffolds are removed from scope. See [04-model-and-tasks.md §v2.0 task catalogue](./04-model-and-tasks.md#v20-task-catalogue) and [ADR-015](./11-risks-and-decisions.md#adr-015--two-adapters-at-v20-sam3--depth-anything-3).

### E3. JWT + OIDC + Helm + KEDA + cosign on day one is too much for a single host

The original plan required the full enterprise auth and deployment surface from Phase 1. For a single-host, single-user, 1–2 GPU local profile, this is overhead without payoff. It also obscures the parts of the design that are interesting to a portfolio reader (the adapter contract, the typed I/O system, the GPU memory management, the crash-recovery state machine).

**Fix.** Split this folder into:
- `upgrade/` — the **local profile**: single host, Docker Compose, single static API key, no multi-tenancy.
- `enterprise/` — additive overlays for each enterprise concern (multi-tenancy + JWT/OIDC; K8s/Helm; supply-chain + secrets; observability-at-scale; multi-region/HA).

The local profile must remain runnable with zero enterprise overlays applied. See [ADR-016](./11-risks-and-decisions.md#adr-016--local-first-architecture-enterprise-as-additive-overlay) and [ADR-017](./11-risks-and-decisions.md#adr-017--single-static-api-key-in-the-local-profile).

### E4. Cross-platform local dev was implicit, not designed

`PLAN.md` mentioned "Windows + GPU dev parity" as an open item. For the actual primary user (Windows-first, with Ubuntu in CI), this needs to be a designed contract, not a footnote.

**Fix.** `scripts/bootstrap_dev.{ps1,sh}` enforces parity: WSL2 + Microsoft NVIDIA driver on Windows; NVIDIA Container Toolkit on Ubuntu. Compose profiles `cpu`, `gpu1`, `gpu2` cover the realistic single-host hardware shapes. See [08-infra-and-cicd.md §Local dev](./08-infra-and-cicd.md#local-dev) and [ADR-018](./11-risks-and-decisions.md#adr-018--cross-platform-local-dev-windows-wsl2--ubuntu-first-class).

### E5. Auto-generated SDK docs were under-specified

`PLAN.md` mentioned OpenAPI generation in passing but did not commit to which client surfaces ship. For a portfolio repo, the SDKs and the Sphinx reference are part of the deliverable.

**Fix.** Two SDKs (`sam3-client-py` on PyPI, `@org/sam3-client` on npm) generated from `openapi.json` per release. Sphinx site covers route docs **and** the typed I/O classes from `packages/io/`. See [03-api-spec.md §OpenAPI generation and SDKs](./03-api-spec.md#openapi-generation-and-sdks).

## What `PLAN.md` already nails

For balance, the following are correct and load-bearing; do not relitigate:

- Four-tier split, with workers never exposing HTTP.
- Celery `acks_late=True` + `task_reject_on_worker_lost=True`.
- Direct-to-S3 with presigned URLs (image bytes never traverse FastAPI).
- Adapter pattern: `ModelAdapter` Protocol + `ModelCapabilities` + registry.
- API stateless; metadata in Postgres; queue and locks in Redis.
- argon2 for API keys at rest.

These are inherited verbatim into the rest of this folder.
