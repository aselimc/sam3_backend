# Enterprise 03 — Supply Chain and Secrets

This overlay adds the supply-chain controls and secret-management patterns that are required for regulated environments and recommended for any production deployment.

## What this overlay adds

- **Image signing** with cosign (keyless OIDC against Fulcio) and admission policy that requires verified signatures in prod.
- **Vulnerability scanning** with Trivy gating CI on `HIGH` / `CRITICAL` (with explicit waivers in `trivy.yaml`).
- **SBOM** emission with Syft per image, attached to GitHub releases.
- **Pinned base images** by digest, refreshed by Dependabot or Renovate.
- **Pinned model weights** with sha256 verification at load.
- **External Secrets Operator** pulling from a cloud secret manager (AWS SM / GCP SM / Vault) into K8s `Secret`s.
- **KMS-backed S3 encryption** with tenant-scoped CMKs.
- **IRSA / Workload Identity** for cloud workers — no static AWS keys baked into images.
- **JWT signing-key rotation** runbook.
- **Tenant webhook secret rotation** with `kid` segments.

## Image signing

CI on `v*` tag:

```yaml
jobs:
  build-images:
    matrix: [api, worker]
    steps:
      - docker buildx build --push -t ghcr.io/<org>/sam3-<svc>:${tag}
      - syft ghcr.io/<org>/sam3-<svc>:${tag} -o spdx-json > sbom-<svc>.json
      - trivy image --exit-code 1 --severity HIGH,CRITICAL --ignorefile trivy.yaml \
          ghcr.io/<org>/sam3-<svc>:${tag}
      - cosign sign --yes ghcr.io/<org>/sam3-<svc>:${tag}
      - gh release upload ${tag} sbom-<svc>.json
```

Admission policy in prod (Sigstore policy-controller or Kyverno):

```yaml
apiVersion: policy.sigstore.dev/v1beta1
kind: ClusterImagePolicy
metadata: { name: signed-by-org }
spec:
  images:
    - glob: ghcr.io/<org>/sam3-*
  authorities:
    - keyless:
        identities:
          - issuer: https://token.actions.githubusercontent.com
            subject: https://github.com/<org>/sam3-backend/.github/workflows/release.yml@refs/tags/v*
```

Unsigned or wrong-issuer images cannot be admitted.

## Vulnerability gating

`trivy.yaml` is the single source of truth for waivers. Every waiver carries:

- The CVE ID.
- The reason (false positive, fix unavailable, mitigated by network policy, …).
- An expiry date.
- An owner.

Expired waivers fail CI; this is intentional. The runbook for handling a fresh CVE is in `infra/runbooks/cve-response.md`.

## Pinned weights

`packages/models/<name>/weights.py` declares the HF revision and sha256:

```python
SAM3_WEIGHTS = HFModel(
    repo_id="facebook/sam3",
    revision="2c5d8a3f…",                 # pinned commit sha, never `main`
    filename="sam3_hiera_large.pt",
    sha256="9d8b2a4f3e…",
)
```

`packages/models/<name>/adapter.py` calls `weights.fetch(SAM3_WEIGHTS)` which downloads to the HF cache, verifies sha256, and raises if mismatched. A bumped weight is a code change reviewed in PR — no silent floats.

## Secrets distribution

Local profile uses `.env`. Enterprise replaces this with the External Secrets Operator:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata: { name: sam3-jwt-signing }
spec:
  refreshInterval: 1h
  secretStoreRef: { name: aws-sm, kind: ClusterSecretStore }
  target: { name: sam3-jwt-signing }
  data:
    - secretKey: private_key_pem
      remoteRef: { key: /sam3/jwt/private_key_pem }
    - secretKey: public_key_pem
      remoteRef: { key: /sam3/jwt/public_key_pem }
```

Pods mount the resulting `Secret` as a projected volume. ESO refreshes hourly; rotation is a write to the cloud secret manager and a forced rollout (`kubectl rollout restart deploy/api`).

### What lives in the secret manager

| Secret | Owner | Rotation |
|---|---|---|
| JWT signing private key | platform | quarterly; old key in JWKS for 7 d |
| Webhook secrets per tenant | platform | per-tenant; admin endpoint |
| MinIO / S3 root credentials | platform | quarterly |
| Database password | platform | as required by managed Postgres |
| OIDC client secret per tenant | platform | per-tenant |
| Trivy waiver overrides | security | none — config |

## KMS

- SSE-KMS with a tenant-scoped CMK if the cluster supports it; otherwise SSE-S3.
- CMK alias pattern: `alias/sam3/{tenant_slug}`.
- The `bucket-init` Helm Job (see [`02-kubernetes-and-helm.md`](./02-kubernetes-and-helm.md)) provisions one CMK per tenant on tenant creation and writes the alias into `tenants.config.kms_alias`.
- Bucket policy denies `s3:PutObject` without `x-amz-server-side-encryption-aws-kms-key-id` matching the tenant alias.

## IRSA / Workload Identity

In cloud, workers do **not** receive static AWS keys.

| Cluster | Mechanism |
|---|---|
| EKS | IRSA — the worker ServiceAccount is bound to an IAM role with the minimum S3 permissions for the worker's tenant prefixes |
| GKE | Workload Identity — same shape, different binding |
| AKS | Azure AD Workload Identity |

The Helm chart parameterizes the role ARN per environment and binds it to the worker SA:

```yaml
serviceAccount:
  create: true
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::<account>:role/sam3-worker
```

## CORS bucket policy (enterprise)

Uploads bucket allows `PUT` from approved web origins only (allowlist per environment). Artifacts bucket allows `GET` from anywhere (presigned URLs are themselves the auth) but blocks `OPTIONS` preflight from anonymous origins to deter scraping.

## Rotation runbook

`infra/k8s/helm/sam3-backend/SECRETS.md` covers:

- JWT signing key (quarterly): generate → write to SM → ESO refresh → restart API → wait 7 d → delete old.
- Tenant webhook secret (per-tenant on demand): admin endpoint mints `webhook_secret_current`; the previous value moves to `webhook_secret_previous` for 7 d.
- MinIO / S3 root: only run during a scheduled maintenance window; document the procedure for refreshing all dependent IAM roles.
- DB password: depends on the managed Postgres provider's supported flow (RDS rotation, etc.).

## Tests

- `tests/enterprise/supply_chain/test_signed_images.py`: pulls the latest `vX.Y.Z` tag and runs `cosign verify`.
- `tests/enterprise/supply_chain/test_sbom_present.py`: asserts every released image has a published SBOM.
- `tests/enterprise/secrets/test_eso_sync.py`: deploys ESO + a fake secret store via `kind`, asserts secrets land as expected.
- `tests/enterprise/security/test_no_static_aws_keys.py`: greps the rendered Helm output for `AWS_ACCESS_KEY_ID`; fails if found.

## Threat-model additions over the local profile

| Threat | Mitigation |
|---|---|
| Stolen image registry credential publishes a malicious image | Cosign keyless signing tied to a specific GitHub Actions identity; admission denies anything else |
| Long-lived AWS key leak | None used — IRSA / Workload Identity replaces static keys |
| KMS key compromise | Per-tenant CMKs limit blast radius; rotation runbook |
| JWT signing key compromise | Quarterly rotation; old key kept in JWKS for 7 d |
| Cross-tenant access via S3 directly | Bucket policy denies puts without correct CMK; per-tenant prefix scoping; worker double-check |
| SBOM tampering | SBOM signed alongside the image with cosign |
