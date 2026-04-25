# Enterprise 01 — Multi-tenancy and Auth

This overlay turns the single-owner local profile (`Principal(owner_id="local", scopes=["*"])`) into a multi-tenant system with real identity. It is purely additive: the local `X-API-Key` path keeps working, and the routers do not change.

## What this overlay adds

- A `tenants` table and the concept of tenant isolation.
- A `users` table backed by argon2 password hashes.
- An `api_keys` table with per-key scopes, expiry, revocation, and last-used tracking.
- A `tenant_quotas` table for hard limits (concurrent jobs, daily job count, daily GPU-seconds).
- An append-only `audit_events` table.
- New auth routes: `/v1/auth/login`, `/v1/auth/refresh`, `/v1/auth/me`, `/v1/auth/api-keys/*`, `/v1/auth/oidc/exchange`, `/v1/auth/jwks.json`.
- A new auth dependency that resolves any of (`X-API-Key`, `Bearer JWT`, OIDC-exchanged token) to the same `Principal` shape — now carrying `tenant_id` and a real per-key `scopes` list.

## Schema delta (additive Alembic migration `0002_enterprise_users.py`)

```sql
CREATE TABLE tenants (
    id          uuid PRIMARY KEY,
    slug        text UNIQUE NOT NULL,
    config      jsonb NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE users (
    id              uuid PRIMARY KEY,
    tenant_id       uuid NOT NULL REFERENCES tenants(id),
    email           text UNIQUE NOT NULL,
    password_hash   text NOT NULL,
    scopes          text[] NOT NULL DEFAULT '{}',
    created_at      timestamptz NOT NULL DEFAULT now(),
    disabled_at     timestamptz
);

CREATE TABLE api_keys (
    id            uuid PRIMARY KEY,
    tenant_id     uuid NOT NULL REFERENCES tenants(id),
    user_id       uuid NOT NULL REFERENCES users(id),
    key_hash      text NOT NULL,                -- argon2id
    label         text NOT NULL,
    scopes        text[] NOT NULL DEFAULT '{}',
    expires_at    timestamptz,
    revoked_at    timestamptz,
    last_used_at  timestamptz
);

CREATE TABLE tenant_quotas (
    tenant_id              uuid PRIMARY KEY REFERENCES tenants(id),
    max_concurrent_jobs    int  NOT NULL DEFAULT 16,
    daily_job_limit        int  NOT NULL DEFAULT 5000,
    daily_gpu_seconds      int  NOT NULL DEFAULT 28800,
    rate_limit_overrides   jsonb NOT NULL DEFAULT '{}'
);

CREATE TABLE audit_events (
    id          bigserial PRIMARY KEY,
    actor_id    uuid,
    tenant_id   uuid,
    action      text NOT NULL,
    context     jsonb NOT NULL DEFAULT '{}',
    at          timestamptz NOT NULL DEFAULT now()
) PARTITION BY RANGE (at);

ALTER TABLE jobs
    ADD COLUMN tenant_id uuid REFERENCES tenants(id),
    ADD COLUMN owner_id_uuid uuid;

CREATE INDEX idx_jobs_tenant_created ON jobs (tenant_id, created_at DESC);
```

After backfill (`UPDATE jobs SET tenant_id = …, owner_id_uuid = …`) a follow-up migration makes `tenant_id` and `owner_id_uuid` `NOT NULL` and drops the legacy text `owner_id`. Forward-compatible per the cross-phase rules in [`../upgrade/09-phases.md`](../upgrade/09-phases.md).

## Identity sources

Three sources, all resolving to the same `Principal`:

| Source | Header | Validator |
|---|---|---|
| API key | `X-API-Key: <key>` | argon2id verify against `api_keys.key_hash`; reject if expired or revoked; update `last_used_at` |
| Local JWT | `Authorization: Bearer <jwt>` | RS256 verify against the active signing key; check `exp`, `aud`, `iss` |
| OIDC exchange | `POST /v1/auth/oidc/exchange` body `{ id_token }` | verify against the configured IdP's JWKS; provision/lookup user by email; mint our own JWT |

`packages/security/auth.py::get_principal` is the single dependency. The routers do not know which source produced the principal.

```python
class Principal(BaseModel):
    owner_id: UUID
    tenant_id: UUID
    scopes: list[str]
```

## JWT setup

- RS256 (asymmetric). Private signing key in cluster Secret; public keys exposed at `/v1/auth/jwks.json` so other services can verify.
- Access token TTL: 15 min. Refresh token TTL: 7 d, stored hashed in Postgres for revocation.
- Quarterly rotation. The JWKS endpoint serves both old and new keys for 7 d after a rotation so in-flight tokens stay valid.

## OIDC

Off by default; enable per-tenant via `tenants.config.oidc = { issuer, audience, jwks_url, claim_to_email_map }`. On `/v1/auth/oidc/exchange`:

1. Verify the incoming ID token against the configured JWKS.
2. Look up `users` by the mapped email.
3. If no user exists and `tenants.config.oidc.auto_provision == true`, create one with the default scope set.
4. Issue our own access + refresh JWTs.

This keeps downstream services unaware of the IdP.

## Per-key scopes

| Scope | Grants |
|---|---|
| `tasks:submit` | `POST /v1/uploads`, `POST /v1/tasks/*` |
| `tasks:read` | `GET /v1/jobs*`, `GET /v1/io/types`, `GET /v1/models` |
| `tasks:cancel` | `DELETE /v1/jobs/{id}` |
| `apikeys:read` | list keys |
| `apikeys:write` | create / revoke |
| `admin` | tenant config and quotas |

A missing scope returns `403 forbidden`; the local profile's wildcard scope (`["*"]`) keeps test code unchanged.

## Tenant isolation

The repository layer (`packages/db/repositories/`) already filters every query by `owner_id`. Enterprise adds `tenant_id` to the same filter:

```python
return await session.execute(
    select(Job).where(
        Job.tenant_id == principal.tenant_id,
        Job.owner_id_uuid == principal.owner_id,
    )
)
```

A repository test fixture asserts that no `SELECT` against `jobs`, `artifacts`, or `webhook_deliveries` slips through without both filters present.

## Storage scoping

The local profile's storage keys carry `local/` as the owner segment. Enterprise replaces this with `{tenant_slug}/`. The migration is purely a path rename — no data move; legacy `local/` rows are migrated into the first tenant on cutover.

The worker double-check (`key.startswith(f"uploads/{principal.tenant_slug}/")`) becomes mandatory.

## Audit log

`audit_events` is append-only. Inserted by:

- Auth events: `auth.login.ok`, `auth.login.fail`, `auth.refresh`, `auth.oidc.exchange`, `apikey.created`, `apikey.revoked`.
- Mutation events: `job.submitted`, `job.canceled`, `tenant.config.updated`, `quota.updated`.

Partitioned by month; retention 365 d; old partitions dropped by a Celery beat sweep.

## Rate-limit overrides

`tenant_quotas.rate_limit_overrides`:

```json
{
  "buckets": {
    "upload":      { "rps": 50,  "burst": 100 },
    "enqueue.gpu": { "rpm": 600, "burst": 60  },
    "read":        { "rps": 200, "burst": 400 }
  }
}
```

Defaults from `core/config.py` are merged with overrides at request time. Hard quotas (concurrent jobs, daily counts, daily GPU-seconds) are enforced in `packages/db/repositories/jobs.py` before the API even reaches the broker.

## Webhook secret rotation

Each tenant has `webhook_secret_current` and optional `webhook_secret_previous`. Outbound `X-SAM3-Signature` includes a `kid=` segment so receivers can verify against either during a rotation window. After 7 d the previous secret is deleted.

## CI additions

- A new test fixture `enterprise_principal_factory` produces principals across all three identity sources.
- A pen-test target `tests/enterprise/test_cross_tenant.py` attempts cross-tenant access via every endpoint and asserts `404` (not `403`, to avoid enumeration).
- `openapi-diff` checks that no enterprise-only field leaks back into the local profile's spec.

## Rollback

Disable the enterprise auth dependency and re-mount the local-profile `X-API-Key` validator. The DB tables remain (additive); future enterprise applications start clean.
