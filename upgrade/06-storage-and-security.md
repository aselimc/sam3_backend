# 06 — Storage and Security (local profile)

This document covers the **local profile**. Multi-tenant identity (JWT + OIDC + per-key scopes), KMS-backed encryption, IRSA / Workload Identity for cloud workers, External Secrets Operator, per-tenant webhook secret rotation, and pen-test / threat-model expansion live in:

- [`enterprise/01-multi-tenancy-and-auth.md`](../enterprise/01-multi-tenancy-and-auth.md)
- [`enterprise/03-supply-chain-and-secrets.md`](../enterprise/03-supply-chain-and-secrets.md)

What follows is the minimum that runs on a single host with Docker Compose and a single static API key.

## Storage

### Buckets and key layout

Two logical buckets (or two prefixes within a single bucket — equivalent on S3-compatible stores; MinIO in local).

```
sam3-uploads
  uploads/local/{yyyy}/{mm}/{dd}/{uuid}                     # raw user input

sam3-artifacts
  artifacts/local/{yyyy}/{mm}/{dd}/{job_id}/{name}.{ext}    # job output
  artifacts/local/{yyyy}/{mm}/{dd}/{job_id}/_meta.json      # adapter version, params, hash
```

In the enterprise overlay the `local` segment is replaced by `{tenant_slug}`, enabling per-tenant lifecycle and IAM scoping. The path shape is identical otherwise.

Reasoning:
- Owner segment as the first prefix supports per-tenant lifecycle rules later without a key rewrite.
- Date prefixes amortize S3 partitioning and are convenient for lifecycle scoping.
- `_meta.json` next to artifacts removes the need to read Postgres for a downloaded result to be self-describing.

### Versioning

Both buckets have versioning **on**.

- Uploads bucket: a re-upload with the same key creates a new version. Workers always read the version pinned by the worker on submit (`image_ref` includes `?versionId=...`).
- Artifacts bucket: workers never overwrite; if a retry produces new outputs, they go to a sibling `attempt=2/` prefix. This prevents partial-overwrite races.

### Lifecycle

| Prefix | Default rule (local) |
|---|---|
| `uploads/*` | Expire 24 h after object creation |
| `artifacts/*` | Expire 30 d after creation |
| `*/...` versions | Permanently delete noncurrent after 7 d |

In local, lifecycle is enforced by a `bucket_sweeper` Celery beat job (since MinIO lifecycle support is partial across versions). Enterprise uses S3 native lifecycle policies.

### Presigning

- Presigner: `boto3` `generate_presigned_url`.
- TTL defaults: PUT 15 min, GET 10 min. Server-side overridable per request, capped at 1 h.
- PUT presign **must** require `Content-Type` and `Content-Length` headers; the bucket policy denies PUTs without them. This blocks request-smuggling tricks where the client uploads a different size/type than declared.
- GET presigns include `response-content-disposition` so browsers download with a sane filename.

### Multipart

Threshold: `byte_length > 16 MB` triggers multipart on `POST /v1/uploads`. Server returns:

```json
{
  "image_ref": "s3://sam3-uploads/uploads/local/.../{uuid}",
  "multipart": {
    "upload_id": "...",
    "part_size": 16777216,
    "parts": [
      { "part_number": 1, "url": "https://..." },
      { "part_number": 2, "url": "https://..." }
    ],
    "complete_url": "https://api/v1/uploads/{uuid}/complete"
  },
  "expires_at": "..."
}
```

`POST /v1/uploads/{uuid}/complete` body is the list of `{ part_number, etag }`; server calls `CompleteMultipartUpload`.

### Encryption

- **Local**: SSE-S3 (server-managed) on MinIO. TLS only between client and MinIO. The MinIO root key is in `.env`.
- **Enterprise**: SSE-KMS with tenant-scoped CMK; `alias/sam3/{tenant_slug}`. See [`enterprise/03-supply-chain-and-secrets.md`](../enterprise/03-supply-chain-and-secrets.md).

### CORS

Uploads bucket allows `PUT` from `http://localhost:*` in local. Artifacts bucket allows `GET` from anywhere (presigned URLs are themselves the auth).

### Backend abstraction

`packages/storage/base.py` defines `StorageBackend` ABC. Concrete impls: `S3Backend` (boto3, used for both MinIO and AWS S3) and `LocalBackend` (filesystem, tests only). The local backend implements presigning by issuing short-lived signed URLs against the API itself — keeps test code identical to prod code at the call site.

## Security (local profile)

### Identity

Single static API key, configured via `LOCAL_API_KEY` env var.

```
X-API-Key: <LOCAL_API_KEY>
```

- Compared in constant time.
- Resolves to `Principal(owner_id="local", scopes=["*"])`.
- Health, model discovery (`/v1/models`), and I/O discovery (`/v1/io/types`) are unauthenticated so SDKs can introspect without configuration.

The full identity surface (JWT issuance, refresh tokens, OIDC token exchange, JWKS rotation, per-key scopes, per-key revocation, IP allowlist, last-used tracking) is the enterprise overlay. Adding it does not break the local `X-API-Key` path — both stop at the same `Principal` shape.

### Authorization

In local profile there is one principal with all scopes. The scope strings still exist in code and in the OpenAPI document so that the enterprise overlay can wire them to JWT claims without re-touching the routers:

| Scope | Grants |
|---|---|
| `tasks:submit` | POST `/v1/uploads`, `/v1/tasks/*` |
| `tasks:read` | GET `/v1/jobs*`, `/v1/io/types`, `/v1/models` |
| `tasks:cancel` | DELETE `/v1/jobs/{id}` |

Resource visibility is filtered at the data layer by `WHERE owner_id = :principal.owner_id`. This is the **single** isolation rule and is enforced in `packages/db/repositories/*` — routers do not write `WHERE` clauses directly. With the constant `owner_id="local"` this is a no-op locally, but the call site is identical to the enterprise multi-owner case.

### Storage scoping (belt-and-braces)

- Storage keys carry the owner segment; presigned URLs scope by exact key.
- Worker runtime re-validates `key.startswith(f"uploads/{principal.owner_id}/")` before fetching.
- Database queries always filter by `owner_id`. A repository test fixture asserts that no `SELECT * FROM jobs` slips through without the filter.

The check is identity-shaped, so the only thing the enterprise overlay changes is what `principal.owner_id` resolves to (a tenant-scoped UUID instead of `"local"`).

### Rate limiting

Token-bucket per `(owner_id, bucket)` in Redis with a Lua script for atomicity:

```lua
-- packages/broker/ratelimit.lua  (illustrative)
local key      = KEYS[1]
local now_ms   = tonumber(ARGV[1])
local rps      = tonumber(ARGV[2])
local burst    = tonumber(ARGV[3])
local cost     = tonumber(ARGV[4])

local data   = redis.call("HMGET", key, "tokens", "ts")
local tokens = tonumber(data[1]) or burst
local ts     = tonumber(data[2]) or now_ms
local elapsed = math.max(0, now_ms - ts) / 1000.0
tokens = math.min(burst, tokens + elapsed * rps)
local allowed = tokens >= cost
if allowed then tokens = tokens - cost end
redis.call("HMSET", key, "tokens", tokens, "ts", now_ms)
redis.call("PEXPIRE", key, 600000)
return { allowed and 1 or 0, math.floor(tokens), burst }
```

Returned tokens populate `X-RateLimit-Remaining` / `X-RateLimit-Limit`.

In local the buckets are shared across all callers (one principal). The limiter is exercised in tests primarily; useful as a regression guard.

### Secrets

- **Local**: `.env` (gitignored). Bootstrap script `scripts/bootstrap_dev.{ps1,sh}` generates a random `LOCAL_API_KEY` and `WEBHOOK_SECRET` on first run; the user can override.
- **Enterprise**: External Secrets Operator pulling from cloud secret manager (AWS SM / GCP SM / Vault) — see [`enterprise/03-supply-chain-and-secrets.md`](../enterprise/03-supply-chain-and-secrets.md).

### Webhook signing

HMAC-SHA256 over `"{t}.{raw_body}"` with `WEBHOOK_SECRET` (env). Signature header:

```
X-SAM3-Signature: t=<unix>,v1=<hex>
```

Receivers must reject `t` older than 5 min. Per-tenant secret rotation with `kid` is enterprise.

### CORS, security headers, and TLS

- **Local**: TLS optional (Compose can bring up a self-signed cert via `caddy` if desired; default is plain HTTP on `localhost`). Security headers (`X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`) are still set by the API for parity.
- **Enterprise**: TLS terminated at ingress with HSTS preload; full security header set; CSP for the docs site.

### Audit

The `audit_events` table is **enterprise-only**. In local, structured logs in `packages/core/logging.py` cover the equivalent surface (every auth event and every state-mutating action emits a log line with `event_type` and `actor`).

### Threat model summary (local profile)

| Threat | Mitigation |
|---|---|
| Stolen API key | Single key, rotated by editing `.env` and restarting; suitable for trusted local dev |
| Cross-tenant data access | N/A locally (single owner). Code path enforced via repository filter. |
| Malicious upload (image bomb) | Magic-byte sniff in worker, MAX_IMAGE_PIXELS, dimension cap, byte-length cap, allowlisted MIME |
| OOM crash | Preflight + RuntimeOOM bump in worker; one job per process; replicas continue |
| Replay attacks | Idempotency keys; SSE auth via static API key in query for browser scenarios (5 min token), or in header for SDKs |
| Webhook spoofing | HMAC signature with timestamp; replay window 5 min |
| Stuck jobs | Heartbeat + reconciler |
| Supply-chain on weights | Pinned model revisions; HF download verifies sha256 against pinned manifest |
| Privilege escalation | Containers run as non-root; read-only root fs in production-like compose profile |

Enterprise expansion of the threat model — supply-chain on images (cosign + Trivy gating), pen-test scope, network policies, multi-tenant data exposure tests — lives in the enterprise folder.
