# Enterprise 05 — Multi-Region and HA

This overlay covers high availability within a region, then multi-region active-passive (and a sketch of active-active). It is the last enterprise overlay an adopter typically needs and is also the most opinionated, because the right answer depends heavily on the cloud provider and the recovery-time / recovery-point targets the business will fund.

## Targets

| Metric | Target | Rationale |
|---|---|---|
| **RTO** (recovery time) | ≤ 60 s for the API tier; ≤ 5 min for in-flight jobs | API is stateless; jobs can be redelivered from Redis |
| **RPO** (recovery point) | ≤ 5 s for `jobs` rows; 0 for already-uploaded artifacts | Postgres synchronous replica + S3 cross-region replication |
| **Single-region failure** | All in-flight jobs succeed or are explicitly retryable | DB primary failover + Redis Sentinel |
| **Region failure** | Service degrades to the active region; queued jobs lost; all `SUCCEEDED` jobs durable | S3 CRR; Postgres logical replica per region (passive) |

## In-region HA

### Postgres

- Managed Postgres with synchronous replication to a hot standby in a different AZ. RDS Multi-AZ / Cloud SQL HA / Patroni — pick one based on cloud.
- `pgbouncer` in front of the primary endpoint to avoid connection storms during failover.
- Read traffic (`GET /v1/jobs/*`) can use the read replica when latency budget allows; write traffic always goes to the primary.

### Redis

- Sentinel cluster (3 nodes) or managed Redis cluster (ElastiCache / Memorystore). AOF persistence every-second.
- Celery clients use the Sentinel-aware connection URL so failover is transparent.
- A failover during a job's lifetime: `acks_late=True` + the SQL state guard means at most one retry. The reconciler closes any orphan within 90 s.

### S3

- Versioning ON. Lifecycle as per [`../upgrade/06-storage-and-security.md`](../upgrade/06-storage-and-security.md).
- Cross-region replication enabled to the passive region for both `sam3-uploads` and `sam3-artifacts`.

### API + workers

- Multi-AZ pod scheduling with `topologySpreadConstraints` so that no AZ can host > 50% of either workload.
- PodDisruptionBudgets ensure rolling cluster operations cannot drain capacity below `replicas-1`.

## Region failure (active-passive)

The simpler shape; recommended starting point.

```
Region A (active)              Region B (passive)
  API (N replicas)               API (0 replicas, scaled to 0)
  Workers (M GPUs)               Workers (0)
  Postgres primary               Postgres logical replica
  Redis primary                  Redis cold (or absent)
  S3 (master)                    S3 (CRR target, read-only mirror)
```

### Failover steps

1. DNS flip (Route 53 / Cloud DNS) — `api.sam3.example.com` weighted record moves from A → B over the TTL window (60 s).
2. Promote Postgres logical replica in B to primary.
3. Scale up workers in B to the GPU node-pool count.
4. Scale up API in B.
5. Validate: smoke the canonical use cases (segmentation.text, depth.monocular, depth.multiview).
6. Notify customers via status page.

In-flight jobs in region A whose state is `RUNNING` at failure time are lost in the queue (Redis was not replicated). Submitters with `Idempotency-Key` see no doubled-spend on retry. The reconciler in B does not see the orphans (different DB primary), so they live as `RUNNING` rows in the A snapshot — handled by a one-shot recovery script that updates them to `FAILED` with `error_code=region_failover`.

### Failback

Inverse of the above, with one extra step: drain the new primary in B, snapshot, restore in A as the new primary, then DNS-flip back.

## Active-active (advanced)

Only justified when:

- A single customer's data sovereignty rules require regional pinning, **or**
- Aggregate latency from non-active regions is unacceptable.

The shape:

- Independent cells per region. Each cell has its own Postgres primary, Redis, S3, workers.
- A tenant is pinned to a home region (`tenants.config.home_region`).
- The API tier is global; it forwards a request to the tenant's home cell's API.
- Cross-region failover requires a tenant-by-tenant migration — there is no shared queue.

This is documented as a future option, not as a v2.0 deliverable. Adopters that need it should engage early; the schema and adapter layers do not change, but the deployment topology and the routing logic do.

## DR runbook

Stored in `infra/runbooks/dr.md`. Required sections:

1. **Inventory** — what is in each region, with current sizes.
2. **Pre-conditions** — which alerts are firing, who declared the incident.
3. **Failover commands** — copy-pasteable `kubectl` and `aws` (or `gcloud` / `az`) commands.
4. **Validation** — the smoke test checklist.
5. **Communications** — status page template, customer email template.
6. **Failback procedure** — separate from failover; do not rely on memory.
7. **Postmortem trigger** — file within 5 business days.

## DR exercise cadence

- Tabletop: quarterly.
- Live failover in staging: semi-annually.
- Live failover in prod: annually, scheduled, customer-notified.

## Costs

Multi-region is expensive. Cost knobs:

- Passive region runs Postgres replica + S3 CRR only; no API or worker pods.
- Cross-region S3 replication storage is roughly 1× the source storage; egress is the dominant variable cost — large adopters negotiate.
- Active-active doubles compute cost.

## Tests

- `tests/enterprise/dr/test_failover_scripts.py` — dry-run the failover script against a `localstack` + `kind` shadow.
- A scheduled GitHub Action runs the DR exercise checklist against staging on the first Monday of each quarter.

## What is *not* covered

- Disaster scenarios beyond region loss (entire cloud provider failure, regional cloud outage cascading to S3): out of scope; mitigated by `CRR` to a different account or — for very large adopters — to a different cloud.
- Compliance certifications (SOC2, ISO 27001, HIPAA): the controls listed in the enterprise overlays are necessary but not sufficient; an adopter must run the audit themselves.
