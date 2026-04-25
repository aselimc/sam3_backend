# 11 — Risk Register and Decision Log

Two artifacts in one document because they are read together: a decision sets a direction; a risk is what could undermine it. Both age. Update during phases as new information arrives.

## Decision log (ADR-style)

Format borrowed from Michael Nygard's ADRs. One block per decision. Status values: `Proposed`, `Accepted`, `Superseded by …`.

### ADR-001 — Celery on Redis as the task queue

- **Status**: Accepted (`PLAN.md`).
- **Context**: We need durable, retryable, distributable task delivery between API and GPU workers. Options considered: Celery+Redis, Celery+RabbitMQ, RQ, Dramatiq, Arq, custom on Postgres `LISTEN/NOTIFY`.
- **Decision**: Celery with Redis broker and Redis result backend.
- **Consequences**:
  - + Mature, well-documented, large ecosystem, Pythonic.
  - + Redis already needed for rate-limit + locks + idempotency; one fewer system.
  - − Visibility timeouts and acks_late require care to avoid duplicate execution; we mitigate with the SQL state guard.
  - − Less observable than RabbitMQ for dead-letter inspection; we add our own DLQ table for webhooks.

### ADR-002 — Postgres for metadata

- **Status**: Accepted.
- **Context**: We need ACID transactions for the job state machine. Options considered: Postgres, MongoDB, MySQL, DynamoDB, SQLite.
- **Decision**: Postgres via SQLAlchemy 2.x async + Alembic.
- **Consequences**:
  - + ACID, mature, fits the row-shaped data we have.
  - + JSONB for flexible request payloads avoids per-task table sprawl.
  - − One more stateful service to run; mitigated by managed offerings in cloud and by Compose locally.

### ADR-003 — S3-compatible object store, MinIO in dev

- **Status**: Accepted.
- **Context**: Need to keep image bytes off the API and provide presigned URLs.
- **Decision**: S3 API as the contract; boto3 client; MinIO for local. `LocalBackend` for tests.
- **Consequences**:
  - + Single client across dev/prod.
  - + Lifecycle, versioning, KMS all standard.
  - − boto3 quirks (e.g. signature v4 + Content-Length on presign).

### ADR-004 — One process per GPU

- **Status**: Accepted.
- **Context**: Multi-tenant GPU sharing is hard. Memory leaks compound. CUDA context safety with forking is fragile.
- **Decision**: One Celery worker process per GPU; `--pool=solo`. Scale by replicas. Local profile pins via `CUDA_VISIBLE_DEVICES` per Compose service.
- **Consequences**:
  - + Simple resource accounting.
  - + Easy to reason about OOM blast radius.
  - − Higher per-job overhead (no intra-process concurrency); recovered by the micro-batcher.

### ADR-005 — Adapter pattern with capability advertisement

- **Status**: Accepted.
- **Context**: The system is supposed to host SAM3 today and additional models later without API churn.
- **Decision**: `ModelAdapter` Protocol + `ModelCapabilities`; tasks resolve adapters by capability; queues named per `(task, gpu_class)`.
- **Consequences**:
  - + Adding a model is a worker-side change; API is generic.
  - + Same task served by multiple adapters supports gradual migrations.
  - − Indirection. New contributors need to read `04-model-and-tasks.md`.

### ADR-006 — Single state writer post-enqueue

- **Status**: Accepted.
- **Context**: With `acks_late=True`, redelivery is possible. Two writers race.
- **Decision**: API writes only the initial `QUEUED` row. Worker mutates everything else through `UPDATE … RETURNING` with a state predicate. Loser exits without retry.
- **Consequences**: see [02-data-model.md §State machine](./02-data-model.md#state-machine).

### ADR-007 — Idempotency-Key required, 24 h scope

- **Status**: Accepted.
- **Context**: Network retries on submit double-spend GPU.
- **Decision**: Header required on mutating POSTs; Redis NX-PX cache; durable unique index as backstop.
- **Consequences**: clients that don't supply the header get `400`. SDKs default to a random UUID per call; users opt in to stable keys for true exactly-once semantics.

### ADR-008 — Trace context propagated through Celery payload

- **Status**: Accepted.
- **Context**: Celery does not propagate W3C headers automatically.
- **Decision**: Inject `traceparent`/`tracestate` into the payload at submit; extract at task entry.
- **Consequences**: distributed traces survive the queue boundary at the cost of two extra string fields.

### ADR-009 — Webhook signing via HMAC-SHA256 with rotating secrets

- **Status**: Accepted.
- **Context**: Webhooks need authenticity guarantees that survive rotation.
- **Decision**: `t=<unix>,v1=<hex>` signature header; per-tenant secret with `kid` for rotation in enterprise, single shared `WEBHOOK_SECRET` in local.
- **Consequences**: receiver code must verify; SDK provides a verifier.

### ADR-010 — Compatibility shim for `/segment-from-*` paths

- **Status**: Accepted (supersedes the implicit hard cutover in `PLAN.md` Phase 7).
- **Context**: Existing clients on `master` cannot all migrate atomically.
- **Decision**: Keep legacy paths for ≥1 minor release with `Sunset` and `Deprecation` headers.
- **Consequences**: extra route handler that maps to the new pipeline. Removed in v2.1.

### ADR-011 — Feature flags for risky new capabilities

- **Status**: Accepted.
- **Context**: Batching, multi-class routing interact with traffic in ways that need measurement.
- **Decision**: Each gated by an env-driven flag, default off until a phase explicitly enables it under SLO observation.
- **Consequences**: extra branches in code. Worth the safety. Flags removed two releases after they default on.

### ADR-012 — Promote images by re-tag, not rebuild (enterprise)

- **Status**: Accepted, scoped to enterprise.
- **Context**: Rebuild-per-environment causes drift between staging and prod.
- **Decision**: Build once per release, sign once, re-tag for promotion. Local profile does not promote; this ADR exists so the enterprise overlay can adopt it cleanly.
- **Consequences**: `release.yml` does build + sign; `promote.yml` (enterprise) only re-tags.

### ADR-013 — Sphinx docs stay; Mermaid diagrams added

- **Status**: Accepted.
- **Context**: We have a working Sphinx pipeline; rewriting in MkDocs is unnecessary.
- **Decision**: Keep Sphinx for the API reference and `packages/io/` class reference; embed Mermaid diagrams in design docs (this folder) which GitHub renders natively.
- **Consequences**: two doc surfaces. The split is intentional: reference vs. design.

### ADR-014 — Typed I/O class hierarchy supersedes `Modality` enum

- **Status**: Accepted (this folder).
- **Context**: A string `Modality` enum cannot distinguish "image", "image + text prompt", "image + IMU pair", and "image + paired pointcloud" — each is a different validation, storage, and SDK shape. The original `Capability(inputs=[Modality], outputs=[Modality])` design is too coarse for adapter discovery and SDK generation.
- **Decision**: Define typed Pydantic input/output classes under `packages/io/` per [04a-io-types.md](./04a-io-types.md). Adapters declare `TypedCapability(task, input_class, output_class)`. The API surface is gated by an `IORegistry` filter that hides any class not claimed by a loaded adapter ("no overshooting").
- **Consequences**:
  - + Self-describing schemas in OpenAPI; SDKs are tight.
  - + Public surface scales with adapters, not with hopes.
  - − Adapter authors must pick from existing classes or contribute new ones; small friction.

### ADR-015 — Two adapters at v2.0: SAM3 + Depth Anything 3

- **Status**: Accepted (this folder; supersedes the SAM2/depth-V2/pose/recon scaffolds in `PLAN.md`).
- **Context**: The original plan scaffolded SAM2, Depth Anything V2, pose, and reconstruction adapters as proof-of-pluggability. For a portfolio-shaped scope this is an unbounded liability — each scaffold ages without a maintainer.
- **Decision**: Ship two real adapters at v2.0:
  - **SAM3** ([facebookresearch/sam3](https://github.com/facebookresearch/sam3)) — text/point/box segmentation.
  - **Depth Anything 3** ([ByteDance-Seed/depth-anything-3](https://github.com/ByteDance-Seed/depth-anything-3)) — monocular and multi-view depth.
  Other model families are removed from scope; they remain trivially addable via the adapter pattern.
- **Consequences**:
  - + The pluggability claim is demonstrated by a *second real adapter* (DA3), not a stub.
  - + DA3 multi-view exercises the `MultiViewImageInput` / `MultiViewDepthOutput` classes, which would otherwise be theoretical.
  - − No public proof of cross-task pluggability for segmentation (a SAM2 alternative would be one). Documented as a future-work invitation in the README.

### ADR-016 — Local-first architecture; enterprise as additive overlay

- **Status**: Accepted (this folder).
- **Context**: Two distinct audiences read this repo: the primary user (single host, 1–2 GPUs, portfolio + research) and downstream adopters (Kubernetes, multi-tenant). Authoring one document for both leads to ceremony that confuses the first audience and detail-shortage that fails the second.
- **Decision**: The `upgrade/` folder describes the local profile. A separate top-level `enterprise/` folder contains additive overlays (multi-tenancy, K8s, supply-chain, observability-at-scale, multi-region/HA). Each overlay specifies its own additive Alembic migrations, code paths, and runbook. The local profile must remain runnable with zero enterprise overlays applied.
- **Consequences**:
  - + Local profile stays small and runnable by one person.
  - + Enterprise adopters get a recipe per concern, not a tangle.
  - − Two surfaces to maintain; mitigated by the rule that enterprise content is additive only (no edits to the local schema once shipped).

### ADR-017 — Single static API key in the local profile

- **Status**: Accepted (this folder).
- **Context**: The original plan required JWT + API-key CRUD + OIDC token exchange from day one. For a single-host single-user profile, that is overhead without payoff.
- **Decision**: Local profile uses a single `LOCAL_API_KEY` from `.env`, validated in constant time, resolving to `Principal(owner_id="local", scopes=["*"])`. Full auth (JWT issuance, refresh, OIDC, per-key scopes, revocation, JWKS rotation) is the enterprise overlay and reuses the same `Principal` shape so routers do not change.
- **Consequences**:
  - + Local bring-up is `LOCAL_API_KEY=…` in `.env` and nothing else.
  - + Enterprise overlay only swaps the auth dependency; routers untouched.
  - − A leaked local key is a full breach locally; acceptable for the audience.

### ADR-018 — Cross-platform local dev: Windows (WSL2) + Ubuntu first-class

- **Status**: Accepted (this folder).
- **Context**: The repo's primary user works on Windows; CI and most cloud environments are Linux. Both must work without one being a second-class citizen.
- **Decision**: `scripts/bootstrap_dev.{ps1,sh}` enforces parity: Windows requires WSL2 + the Microsoft NVIDIA driver; Ubuntu requires the NVIDIA Container Toolkit. Compose profiles `cpu`, `gpu1`, `gpu2` cover the realistic local hardware shapes. Tests must pass on both OSes in CI.
- **Consequences**:
  - + The bootstrap script is the single source of truth for "did this machine pass the prereqs?".
  - − Two install paths to keep in sync; mitigated by a shared validation helper (`scripts/_env_check.py`) called by both scripts.

### ADR-019 — Webhooks dispatcher shares the worker pool in local

- **Status**: Accepted (this folder).
- **Context**: The original plan had a dedicated webhook-dispatcher deployment. Locally that is one extra container for nothing.
- **Decision**: Local profile runs the webhook dispatcher as a non-GPU Celery queue inside the same worker container, with `--queues task.*,webhooks.*`. Enterprise overlay splits it back into its own deployment for HPA on outbound queue depth.
- **Consequences**:
  - + One fewer container locally.
  - − A misbehaving webhook receiver could starve the worker pool's non-GPU thread budget; mitigated by per-receiver timeout + cap.

## Risk register

Severity is `Low / Med / High`. Likelihood is `Low / Med / High`. Status: `Open`, `Mitigated`, `Accepted`, `Closed`.

### R-1 — Cold-start dominates first-request latency for new model

- **Severity**: Med. **Likelihood**: High. **Status**: Mitigated.
- **Detail**: HF download (minutes) + load (~30 s for SAM3, ~20 s for DA3) on first run.
- **Mitigation**: Bake weights into worker image (build-arg). Redis lock per `model_id` to coordinate. Pod readiness gate when `WORKER_PRELOAD=true`. Documented in `04-model-and-tasks.md`.
- **Residual**: First image pull on a fresh node still costs minutes. HF cache mounted as a Compose volume amortizes this across container rebuilds locally.

### R-2 — Queue stuck due to redelivery loop on a poison message

- **Severity**: High. **Likelihood**: Low. **Status**: Mitigated.
- **Mitigation**: `max_retries=3`; on the 4th attempt, the message is moved to a DLQ table and surfaced via `GET /v1/jobs/{id}` with `error_code=poison`. Reconciler force-fails any `RUNNING` row past heartbeat timeout.

### R-3 — GPU memory fragmentation from frequent model swaps

- **Severity**: Med. **Likelihood**: Med. **Status**: Open.
- **Mitigation**: `MODELS_ENABLED` is a strict allowlist; warm-pool sized to fit both SAM3 + DA3 on a 24 GB GPU. Operator runbook says "if eviction count grows, restart the worker container".
- **Open item**: experiment with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Phase 8.

### R-4 — Token-bucket Lua incorrect under clock skew

- **Severity**: Low (single host) / Med (enterprise). **Likelihood**: Low. **Status**: Mitigated.
- **Mitigation**: Use Redis server time (`TIME` command) inside the Lua script for `ts`; client-supplied time is a fallback only.

### R-5 — Presigned URL leakage

- **Severity**: High. **Likelihood**: Low. **Status**: Mitigated.
- **Mitigation**: Short TTL (10 min). PUT URLs bind `Content-Type` and `Content-Length`. Audit every presign issuance.
- **Residual**: A genuinely leaked URL within the TTL is exploitable. Documented as accepted residual.

### R-6 — DB primary loss

- **Severity**: High in enterprise; **Med locally**. **Likelihood**: Low. **Status**: Local: Accepted (no HA). Enterprise: Open — see [`enterprise/05-multi-region-and-ha.md`](../enterprise/05-multi-region-and-ha.md).

### R-7 — Redis outage

- **Severity**: High. **Likelihood**: Low. **Status**: Local: Accepted. Enterprise: Open — see enterprise overlay.

### R-8 — Image-bomb / decompression bomb

- **Severity**: Med. **Likelihood**: Med. **Status**: Mitigated.
- **Mitigation**: `MAX_IMAGE_PIXELS` set globally; dimension cap at API; magic-byte sniff at worker; no decoding in API; OOM guard in worker.

### R-9 — SDK lag behind API

- **Severity**: Low. **Likelihood**: High. **Status**: Mitigated.
- **Mitigation**: SDKs auto-generated from OpenAPI in `release.yml`; published synchronously with the API release. CI gates on OpenAPI breaking-diff. Sphinx docs published from the same source.

### R-10 — Webhook receivers misbehave (slow, 5xx, infinite redirect)

- **Severity**: Low. **Likelihood**: High. **Status**: Mitigated.
- **Mitigation**: Per-delivery 10 s timeout; 5 retries with exponential backoff; DLQ surfaced via admin endpoint. In local, dispatcher shares worker pool — a per-receiver in-flight cap keeps the pool from being starved.

### R-11 — Cross-tenant data exposure (enterprise only)

- **Severity**: High. **Likelihood**: Low. **Status**: N/A locally. See enterprise overlay.

### R-12 — Long migration window during cutover

- **Severity**: Med. **Likelihood**: Med. **Status**: Open.
- **Mitigation**: Telemetry on legacy routes; window configurable.

### R-13 — DA3 multi-view variance under sparse, hand-held captures

- **Severity**: Med. **Likelihood**: Med. **Status**: Open.
- **Detail**: Multi-view depth quality degrades sharply with very short baselines or with camera-hint mistakes. A user submitting two near-identical photos can produce noisy depth without an obvious error.
- **Mitigation**: API-side pre-check on view dissimilarity (perceptual hash distance); warning in `result_summary` when the heuristic flags low parallax. Documented in use case 8.

### R-14 — Typed I/O class proliferation

- **Severity**: Low. **Likelihood**: Med. **Status**: Mitigated.
- **Detail**: With many adapters, `packages/io/` could balloon into a sprawl of one-off classes.
- **Mitigation**: Class additions require an adapter that uses them (registry filter). Reviewer rule: a new I/O class needs a one-paragraph rationale in the PR description.

### R-15 — WSL2 / NVIDIA driver drift on Windows

- **Severity**: Med. **Likelihood**: Med. **Status**: Mitigated.
- **Detail**: Microsoft / NVIDIA driver updates occasionally break GPU passthrough.
- **Mitigation**: `bootstrap_dev.ps1` runs a CUDA smoke test (`nvidia-smi.exe` + small `torch.cuda.is_available()` probe) and prints a diagnostic if it fails. README links to known-good driver versions.

## Open items (with owners)

| ID | Item | Earliest phase to decide | Provisional default |
|---|---|---|---|
| O-1 | Whether to expose `pointcloud` as a *first-class* DA3 multi-view output by default (vs opt-in) | Phase 5 | Opt-in via input flag; off by default to keep latency lower |
| O-2 | Trace sampling rate in local | Phase 8 | 100% in local; 10% in enterprise prod |
| O-3 | Whether to mount `~/.cache/huggingface` as a Compose volume by default | Phase 7 | Yes; saves multi-GB re-downloads |
| O-4 | Whether to enable HEIC/AVIF by default | Phase 6 | Off; opt-in via env flag |
| O-5 | Whether webhook DLQ items expire automatically | Phase 8 | No; must be acknowledged manually |
| O-6 | Multi-region story | Enterprise | Single region; revisit when an adopter requires it |

## Updating this document

Each PR that lands a phase must:

1. Mark resolved open items as ADRs with status `Accepted`.
2. Update risk statuses (`Mitigated`, `Closed`) as mitigations land.
3. Add new risks discovered during the phase.

The risk register is a living artifact; an empty register means nobody is looking, not that the system is risk-free.
