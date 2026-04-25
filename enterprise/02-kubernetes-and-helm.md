# Enterprise 02 — Kubernetes and Helm

This overlay packages the local Compose stack into a Helm chart suitable for clusters with horizontal scale, queue-driven worker autoscaling, and pod-hygiene policies. It assumes the multi-tenancy overlay (or the local profile) is already in place.

## Layout

```
infra/k8s/
├── helm/sam3-backend/
│   ├── Chart.yaml
│   ├── values.yaml                  # defaults
│   ├── values-dev.yaml
│   ├── values-staging.yaml
│   ├── values-prod.yaml
│   └── templates/
│       ├── api-deploy.yaml
│       ├── api-svc.yaml
│       ├── api-hpa.yaml
│       ├── worker-{a100,t4,l4}-deploy.yaml
│       ├── worker-keda-scaledobject.yaml
│       ├── webhook-dispatcher-deploy.yaml
│       ├── celery-beat-deploy.yaml
│       ├── db-migrate-job.yaml      # pre-install/pre-upgrade hook
│       ├── bucket-init-job.yaml     # post-install
│       ├── bucket-lifecycle.yaml    # S3 native lifecycle JSON
│       ├── servicemonitor.yaml      # for Prometheus Operator
│       ├── networkpolicy.yaml
│       └── poddisruptionbudget.yaml
└── manifests/                       # raw kustomize for non-helm clusters
```

## Workload resources (prod profile)

| Workload | Replicas | Requests / Limits | Notes |
|---|---|---|---|
| `api` | 2 (HPA min 2, max 20) | 500m CPU / 1 Gi RAM | HPA on RPS via custom metric |
| `worker-a100-40g` | =GPUs | 1 GPU + 8 CPU + 32 Gi | one process per GPU |
| `worker-l4-24g` | =GPUs | 1 GPU + 4 CPU + 16 Gi | |
| `worker-t4-16g` | =GPUs | 1 GPU + 4 CPU + 16 Gi | |
| `webhook-dispatcher` | 1 (HPA on outbound queue length) | 100m / 256 Mi | non-GPU |
| `celery-beat` | 1 (leader-elected) | 100m / 256 Mi | |
| `db-migrate` | Job, runs on upgrade | small | hooks `pre-upgrade` + `pre-install` |
| `bucket-init` | Job, runs post-install | small | applies bucket policy + lifecycle |

## API HPA

Scales on RPS (custom metric exported from `http_requests_total`) and CPU.

```yaml
metrics:
  - type: Pods
    pods:
      metric: { name: http_requests_per_second }
      target: { type: AverageValue, averageValue: "200" }
  - type: Resource
    resource:
      name: cpu
      target: { type: Utilization, averageUtilization: 70 }
```

## Worker autoscaling — KEDA

Worker autoscaling is **queue-driven**, not CPU-driven. KEDA `redis-list-length` trigger:

```yaml
triggers:
  - type: redis
    metadata:
      address: redis:6379
      listName: task.segmentation.text.a100_40g
      listLength: "4"     # 1 worker per 4 queued messages, capped by maxReplicas
```

`maxReplicaCount` per worker class is the GPU node-pool size; cluster autoscaler handles node provisioning. Scale-to-zero is enabled outside business hours via `keda.sh/cooldown` + a scheduled scaling rule for the node pool.

DA3 multi-view queues use a smaller `listLength` (2) because each request is more expensive.

## Pod hygiene

- Non-root user (`1000:1000`).
- `readOnlyRootFilesystem: true`; writable `emptyDir` for `/tmp` and `/var/log`.
- `automountServiceAccountToken: false` for API/worker; explicit ServiceAccount with IRSA / Workload Identity (see [`03-supply-chain-and-secrets.md`](./03-supply-chain-and-secrets.md)).
- `seccompProfile: RuntimeDefault`.
- Resource requests = limits for predictability.
- `terminationGracePeriodSeconds: 600` on workers (> longest expected job).
- `preStop` hook on workers: `kill -TERM $(pidof -s celery)`.
- `livenessProbe`: hits a tiny TCP server on the worker that returns OK if the heartbeat thread is alive.
- `readinessProbe`: returns OK only if the warm pool has loaded all `MODELS_ENABLED` (when `WORKER_PRELOAD=true`).

## Networking

- Internal-only Service for `worker → api` is unnecessary — workers do not call the API.
- NetworkPolicy: API ingress from the LB only; worker egress to S3 + Redis + Postgres + OTel; deny all else.
- Cluster operators deploy the policies; chart provides templates.

## Pod Disruption Budget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: { name: api-pdb }
spec:
  minAvailable: 1
  selector: { matchLabels: { app: api } }
```

Workers have `minAvailable: max(1, replicas-1)` to tolerate node drains during cluster upgrades while keeping at least one GPU online.

## Migrations as Helm hooks

`db-migrate` Job runs as `pre-install` and `pre-upgrade`. Idempotent and resumable. Failure rolls the release back. Forward-compatible migrations are required: a release that adds a column is paired with the next release that uses it, never both at once.

## Bucket bootstrap

`bucket-init` Job (post-install) creates buckets, applies policy and lifecycle JSON, and verifies the API has IRSA credentials sufficient to presign. Fails the release if credentials are missing.

## Release flow

```
PR (CI green) → merge to v2 → tag vX.Y.Z → release.yml builds + signs → ArgoCD/Flux syncs to staging
→ smoke (k6) → manual approve → prod sync
```

Promotion is **image tag promotion** (re-tag the signed staging image as prod), not rebuild. See [ADR-012 in upgrade/11](../upgrade/11-risks-and-decisions.md#adr-012--promote-images-by-re-tag-not-rebuild-enterprise).

## Multi-cluster (cell) story

For very large adopters, deploy one cell per region. Each cell has its own Postgres + Redis + S3. Routing across cells is a separate concern (DNS-level or service-mesh-level). See [`05-multi-region-and-ha.md`](./05-multi-region-and-ha.md).

## CI additions

- `helm-lint` and `helm template … | kubeval` gate every PR.
- `kind`-based integration test brings up the chart end-to-end in CI before tagging a release.

## Rollback

- App-level: `helm rollback <release> <previous>` → ArgoCD/Flux reapplies.
- Schema-level: forward-compatible migrations let app rollback without DB rollback. If a migration must be reverted, the playbook is in `infra/k8s/helm/sam3-backend/SECRETS.md` (yes, also covers schema rollback runbooks; rename pending).
