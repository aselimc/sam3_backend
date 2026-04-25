# v2 Upgrade — Design & Integration Documents (Local Profile)

This folder is the working specification for the `v2` re-architecture in the **local profile**: single host, Docker Compose, 1–2 GPUs, cross-platform (Windows + WSL2, Ubuntu), single static API key, two adapters (SAM3 + Depth Anything 3).

It refines and extends `PLAN.md` (which remains the high-level vision) by closing the design gaps that block a confident, multi-phase build, and by re-scoping to the local profile's actual audience.

Enterprise add-ons (multi-tenancy + JWT/OIDC, Kubernetes/Helm, supply-chain hardening, observability-at-scale, multi-region/HA) live in a sibling top-level folder: [`../enterprise/`](../enterprise/). Each enterprise document is an additive overlay on the local profile; none is required to ship v2.0 locally.

The intent is that anyone joining the project mid-build can read this folder top-to-bottom and:

1. understand what the system is, why it is shaped this way, and where the boundaries are;
2. see what is still open versus decided;
3. pick up an in-flight phase and finish it without breaking guarantees made in earlier phases.

## How to read this folder

Read in order on the first pass; jump by topic afterwards.

| # | Document | Purpose |
|---|---|---|
| 00 | [evaluation.md](./00-evaluation.md) | Critique of `PLAN.md`. Lists weaknesses, gaps, scope mismatches, and unstated assumptions, then states how each is resolved in the rest of this folder. |
| 01 | [architecture.md](./01-architecture.md) | Refined target architecture (local profile). Component diagram, data flows, request lifecycles, and the deltas vs `PLAN.md`. |
| 02 | [data-model.md](./02-data-model.md) | Postgres schema (local profile), ER diagram, job state machine, idempotency keys, Redis layout. |
| 03 | [api-spec.md](./03-api-spec.md) | Concrete endpoint inventory, request/response shapes (with DA3), error envelope, headers, pagination, schema versioning, single-API-key auth. |
| 04 | [model-and-tasks.md](./04-model-and-tasks.md) | Model-adapter contract, task contract, GPU scheduling, queue routing by GPU class, batching, model warm-pool. SAM3 + DA3 declarations. |
| 04a | [io-types.md](./04a-io-types.md) | Typed Pydantic I/O class hierarchy (`packages/io/`). Replaces the loose `Modality` enum. The "no overshooting" guarantee. |
| 05 | [worker-runtime.md](./05-worker-runtime.md) | Celery patterns (acks_late, prefetch, late-bind), OOM guard, cancellation, trace-context propagation, heartbeat, reconciler. |
| 06 | [storage-and-security.md](./06-storage-and-security.md) | Bucket layout, lifecycle, presigning, multipart, CORS. Single-static-key auth, rate-limit algorithm, secret distribution (local). |
| 07 | [observability.md](./07-observability.md) | Structured log schema, metric inventory, OTel spans, baseline SLOs, dashboards, alerts. |
| 08 | [infra-and-cicd.md](./08-infra-and-cicd.md) | Compose profiles (`cpu` / `gpu1` / `gpu2`), Dockerfiles, bootstrap scripts, CI, SDK & Sphinx publishing. |
| 09 | [phases.md](./09-phases.md) | Phase-by-phase integration plan. Each phase has scope, exit criteria, rollback, and verification commands. |
| 10 | [use-cases.md](./10-use-cases.md) | End-to-end user journeys and SDK-shaped code samples for each canonical flow (SAM3 + DA3 monocular + DA3 multi-view). |
| 11 | [risks-and-decisions.md](./11-risks-and-decisions.md) | Risk register, ADR-style decision log, open items with owners. |

## Enterprise overlay (separate folder)

The [`enterprise/`](../enterprise/) folder layers production-grade concerns on top of the local profile:

| File | Purpose |
|---|---|
| [enterprise/README.md](../enterprise/README.md) | How to read and apply the enterprise overlays |
| [enterprise/01-multi-tenancy-and-auth.md](../enterprise/01-multi-tenancy-and-auth.md) | `tenants`, `users`, `api_keys`, `audit_events`, `tenant_quotas`; JWT + OIDC; per-key scopes |
| [enterprise/02-kubernetes-and-helm.md](../enterprise/02-kubernetes-and-helm.md) | Helm chart, KEDA autoscaling, HPA, NetworkPolicy, pod hygiene |
| [enterprise/03-supply-chain-and-secrets.md](../enterprise/03-supply-chain-and-secrets.md) | cosign signing, Trivy gating, Syft SBOM, External Secrets, KMS, IRSA |
| [enterprise/04-observability-at-scale.md](../enterprise/04-observability-at-scale.md) | Loki + Tempo, sampling tiers, SLO burn alerts, on-call runbooks |
| [enterprise/05-multi-region-and-ha.md](../enterprise/05-multi-region-and-ha.md) | Postgres HA, Redis Sentinel/cluster, multi-region S3, DR runbook |

Each enterprise document specifies its own additive Alembic migrations and Helm value overrides. The local schema is never modified by an enterprise overlay — overlays are purely additive.

## Status

These documents describe the intended design. No code in this repo has been re-architected yet — `master` is still the single-process FastAPI service. The first PR into `v2` will land Phase 0 (scaffolding). See `09-phases.md` for the executable plan.

## Conventions used in this folder

- Mermaid diagrams render natively on GitHub. Plain ASCII is preferred for low-noise structural sketches that do not benefit from layout.
- Code samples are illustrative pseudocode unless explicitly tagged with a language and a file path.
- "Must / should / may" follow RFC-2119 senses.
- Cross-references use relative links (e.g. `[02-data-model.md](./02-data-model.md)` within this folder; `[../enterprise/01-multi-tenancy-and-auth.md](../enterprise/01-multi-tenancy-and-auth.md)` to the sibling folder).
