# Enterprise Overlay

This folder is the set of additive overlays that turn the **local profile** of v2 (single host, Docker Compose, single static API key, two adapters) into a production deployment fit for an organization with multi-tenant traffic, SLOs, on-call rotations, regulated environments, and multi-region requirements.

The local profile lives in [`../upgrade/`](../upgrade/). Read that first. Each document here assumes the local profile is in place and green.

## Reading order

There is no enforced order. Each overlay is self-contained. A typical adopter applies them in the order below because earlier overlays make later ones easier, but any subset is supported.

| # | Document | Adds |
|---|---|---|
| 01 | [multi-tenancy-and-auth.md](./01-multi-tenancy-and-auth.md) | `tenants`, `users`, `api_keys`, `audit_events`, `tenant_quotas`; JWT + OIDC; per-key scopes; `Principal` resolution change-out |
| 02 | [kubernetes-and-helm.md](./02-kubernetes-and-helm.md) | Helm chart, KEDA queue-driven autoscaling, HPA, NetworkPolicy, pod hygiene, ServiceMonitor |
| 03 | [supply-chain-and-secrets.md](./03-supply-chain-and-secrets.md) | cosign image signing, Trivy gating, Syft SBOM, External Secrets Operator, KMS, IRSA / Workload Identity, secret rotation runbook |
| 04 | [observability-at-scale.md](./04-observability-at-scale.md) | Loki + Tempo, OTel sampling tiers, SLO burn-rate alerts, dashboards, runbook URLs in alert rules |
| 05 | [multi-region-and-ha.md](./05-multi-region-and-ha.md) | Postgres HA + standby, Redis Sentinel / cluster, S3 cross-region replication, DR runbook, RTO/RPO targets |

## Contract with the local profile

Every overlay obeys these rules:

1. **Additive only.** No overlay edits a column added by `upgrade/02-data-model.md` (the local schema). New tables and new nullable columns are fine; renames and drops are forbidden.
2. **Identical request shape.** No overlay changes the public OpenAPI shape of an existing route. New routes (e.g. `/v1/auth/login`) are additive. Header semantics added in an overlay (e.g. `Authorization: Bearer …`) coexist with the local-profile auth path (`X-API-Key`).
3. **Identical adapter contract.** Adapters and the I/O class hierarchy (`packages/io/`) are unchanged across profiles. An adapter built for the local profile runs in enterprise without modification.
4. **Identical `Principal` shape.** Whether the principal comes from a static `X-API-Key` (local) or a JWT claim (enterprise), the routers see `Principal(owner_id, tenant_id?, scopes[])` and filter the data layer by `owner_id` (and `tenant_id` when set). Routers do not branch on profile.
5. **Tested in CI.** Each overlay ships its own integration tests against a Compose-shaped substitute (e.g. `kind` for K8s; `localstack` for KMS; `postgres-ha-test` for replication). The local-profile test suite continues to pass with the overlay disabled.

If a proposed change to the local profile would break any of the above, it lands in an overlay first and migrates back only when both profiles agree.

## What is *not* here

- Per-customer onboarding runbooks — those are deployment-specific, not architecture.
- Pricing / billing tiers — out of scope for this codebase.
- Customer-success dashboards — out of scope.
- Mobile or browser SDKs beyond the auto-generated TypeScript client — out of scope.

## Status

All five documents describe an intended design. None of the enterprise code has been written yet. The local profile (Phases 0–9 in [`../upgrade/09-phases.md`](../upgrade/09-phases.md)) ships first.
