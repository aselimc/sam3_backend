# 03 — API Specification

This document defines the public HTTP surface of v2 in the **local profile**. The OpenAPI 3.1 file lives at `services/api/openapi.json` (generated from the FastAPI app on boot) and is the machine-readable counterpart. Auto-generated SDKs (Python + TS) are emitted from it; the Sphinx API reference includes both the route catalogue and the typed I/O schemas from [04a-io-types.md](./04a-io-types.md).

The full enterprise auth surface — `/v1/auth/login`, `/v1/auth/refresh`, `/v1/auth/oidc/exchange`, `/v1/auth/api-keys/*`, JWKS — lives in [`enterprise/01-multi-tenancy-and-auth.md`](../enterprise/01-multi-tenancy-and-auth.md). It is layered on top of this spec without breaking existing routes.

## Conventions

- Base path: `/v1`.
- Content type: `application/json` unless noted.
- All timestamps are RFC 3339 UTC with millisecond precision.
- IDs are UUID v4.
- Pagination uses opaque cursors.
- Errors share a single envelope (see [§Errors](#errors)).
- Versioning rules in [§Versioning](#versioning).

## Authentication (local profile)

Single static API key, configured via env var `LOCAL_API_KEY` and supplied by clients in the `X-API-Key` header.

```
X-API-Key: <value of LOCAL_API_KEY>
```

- Compared in constant time. On mismatch → `401 unauthorized`.
- The resolved `Principal` is the constant `Principal(owner_id="local", scopes=["*"])`.
- All routes accept this auth source.
- Health routes (`/v1/health/*`) and `/v1/io/types`, `/v1/models` are **unauthenticated** so that local SDK discovery and dashboards work without configuring a key.

The key is read from `.env` (gitignored). `scripts/seed_dev_data.py --print-key` prints the current value once for convenience.

Enterprise overlay replaces this with JWT + OIDC + per-key scopes; the existing `X-API-Key` path remains for compatibility.

## Idempotency

Required on every mutating POST. Header: `Idempotency-Key: <uuid>` (max 64 chars).

- Replay with identical body hash → original response, `X-Idempotent-Replay: true`.
- Replay with different body hash → `409 idempotency_conflict`.
- Missing header → `400 idempotency_required`.

Idempotency window: 24 hours per `(principal, key)`.

## Headers

### Request

| Header | Required on | Notes |
|---|---|---|
| `X-API-Key` | all `/v1/*` except `/v1/health/*`, `/v1/io/types`, `/v1/models` | local profile auth |
| `Idempotency-Key` | all mutating POST/DELETE | UUID |
| `X-Request-Id` | optional | echoed back; otherwise generated |
| `traceparent`, `tracestate` | optional | W3C trace context; propagated to workers |

### Response

| Header | When | Notes |
|---|---|---|
| `X-Request-Id` | always | always set |
| `X-RateLimit-Limit` | rate-limited routes | bucket capacity |
| `X-RateLimit-Remaining` | rate-limited routes | tokens left |
| `X-RateLimit-Reset` | rate-limited routes | unix epoch ms |
| `Retry-After` | `429`, `503` | seconds |
| `Sunset` | deprecated routes | RFC 8594 |
| `Deprecation` | deprecated routes | RFC draft |
| `X-Idempotent-Replay` | replay | `true` only |

## Errors

Single envelope; never leak stack traces.

```json
{
  "error": {
    "code": "rate_limited",
    "message": "Too many requests for bucket enqueue.gpu",
    "request_id": "8c8e5f8e-…",
    "trace_id": "7f3c…",
    "details": { "retry_after_s": 12 }
  }
}
```

Stable error codes:

| Code | HTTP | When |
|---|---|---|
| `unauthorized` | 401 | missing/invalid `X-API-Key` |
| `forbidden` | 403 | scope missing (enterprise) |
| `not_found` | 404 | resource does not exist *or* not visible to caller |
| `validation_error` | 422 | request body invalid |
| `idempotency_required` | 400 | header missing |
| `idempotency_conflict` | 409 | replay with different body |
| `state_conflict` | 409 | e.g. cancel on terminal job |
| `rate_limited` | 429 | bucket exhausted |
| `payload_too_large` | 413 | exceeds local max |
| `unsupported_media_type` | 415 | unknown image format |
| `model_unavailable` | 503 | no worker advertises capability |
| `internal` | 500 | catch-all, alert-worthy |

## Versioning

- The path `/v1/...` is the major version.
- Every public Pydantic model carries `version: Literal["1"]`.
- Adding a field is additive (no version bump).
- Removing or changing semantics requires a new path (`/v2/...`) plus a `Sunset` and `Deprecation` header on the old path for ≥1 minor release.
- Generated OpenAPI is gated in CI: a breaking diff on existing operations fails the PR.

## Endpoints

### Discovery

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/v1/health/live` | none | process responsive |
| GET | `/v1/health/ready` | none | redis + pg + s3 reachable; ≥1 worker heartbeat in 30 s |
| GET | `/v1/health/version` | none | git sha, version, model registry |
| GET | `/v1/models` | none | currently-ready adapters and their capabilities |
| GET | `/v1/io/types` | none | filtered Pydantic schemas for live I/O classes (see [04a](./04a-io-types.md)) |

### Uploads

`POST /v1/uploads`

- Body: `{ filename, content_type, byte_length }`.
- Response: `{ image_ref, upload_url, expires_at, headers }`.
- `image_ref` is `s3://bucket/uploads/local/{uuid}` and is what the client embeds in an `IORef` field of any input class.
- Server validates `content_type` against allowlist (`image/jpeg`, `image/png`, `image/webp`; `image/heic` opt-in via env).
- Server validates `byte_length ≤ MAX_UPLOAD_BYTES` (default 50 MB; env-configurable).
- Multipart: if `byte_length > 16 MB`, response includes `multipart: { upload_id, part_size, parts: [presigned_part_urls] }` and `complete_url`.

### Tasks (generic)

`POST /v1/tasks/{task_type}` where `task_type` is one of the v2.0 task catalogue values from [04-model-and-tasks.md §v2.0 task catalogue](./04-model-and-tasks.md#v20-task-catalogue):

- `segmentation.text`
- `segmentation.point`
- `segmentation.box`
- `depth.monocular`
- `depth.multiview`

Body: a typed `InputBase` discriminated by `input_type`, plus optional submission knobs.

Response: `202 { job_id, state: "QUEUED", created_at, links: { self, sse } }`.

#### `segmentation.text` (SAM3, `ImageTextInput`)

```jsonc
{
  "version": "1",
  "input_type": "image_text",
  "image": {
    "storage_key": "s3://.../uploads/local/9c3f...",
    "content_type": "image/jpeg",
    "byte_length": 482113
  },
  "queries": [
    { "text": "cat",    "confidence_threshold": 0.5, "regularize": false },
    { "text": "person", "confidence_threshold": 0.6, "regularize": true  }
  ],
  "model_id": "sam3",                 // optional; resolved by capability
  "gpu_class": "a100_40g",            // optional
  "callback_url": "https://hooks.local/jobs",
  "max_attempts": 3
}
```

#### `segmentation.point` (SAM3, `ImagePointInput`)

```jsonc
{
  "version": "1",
  "input_type": "image_point",
  "image": { "storage_key": "s3://...", "content_type": "image/png" },
  "points": [ { "xy": [120, 200], "label": 1 } ]
}
```

#### `depth.monocular` (DA3, `ImageInput`)

```jsonc
{
  "version": "1",
  "input_type": "image",
  "image": { "storage_key": "s3://.../uploads/local/abc...", "content_type": "image/jpeg" }
}
```

#### `depth.multiview` (DA3, `MultiViewImageInput`)

```jsonc
{
  "version": "1",
  "input_type": "multiview_image",
  "views": [
    { "image": { "storage_key": "s3://.../v0.jpg", "content_type": "image/jpeg" } },
    { "image": { "storage_key": "s3://.../v1.jpg", "content_type": "image/jpeg" } },
    { "image": { "storage_key": "s3://.../v2.jpg", "content_type": "image/jpeg" } }
  ],
  "camera_hints": { "focal_length_mm": 28.0, "sensor_width_mm": 36.0 }
}
```

### Jobs

| Method | Path | Description |
|---|---|---|
| GET | `/v1/jobs` | List with filters, cursor pagination |
| GET | `/v1/jobs/{id}` | Single job, includes `result_summary` and `output_type` |
| GET | `/v1/jobs/{id}/events` | SSE stream of state transitions |
| GET | `/v1/jobs/{id}/artifacts` | Presigned GET URLs (TTL 10 min) and per-artifact `role` |
| DELETE | `/v1/jobs/{id}` | Cancel — see [§Cancel](#cancel) |

#### Job DTO

```json
{
  "version": "1",
  "id": "0e6d…",
  "task_type": "depth.multiview",
  "model_id": "depth_anything_v3",
  "gpu_class": "a100_40g",
  "state": "SUCCEEDED",
  "attempt": 1,
  "max_attempts": 3,
  "created_at":  "2026-04-25T10:00:00.000Z",
  "queued_at":   "2026-04-25T10:00:00.012Z",
  "started_at":  "2026-04-25T10:00:01.300Z",
  "finished_at": "2026-04-25T10:00:31.900Z",
  "input_type":  "multiview_image",
  "output_type": "multiview_depth",
  "request_summary": { "n_views": 3 },
  "result_summary":  { "n_artifacts": 8, "min_depth": 0.41, "max_depth": 18.7, "units": "meters" },
  "error": null,
  "links": {
    "self":      "/v1/jobs/0e6d…",
    "events":    "/v1/jobs/0e6d…/events",
    "artifacts": "/v1/jobs/0e6d…/artifacts",
    "cancel":    "/v1/jobs/0e6d…"
  }
}
```

`output_type` lets SDK clients pick the right deserializer (`MultiViewDepthOutput`, `DepthMapOutput`, `MaskLabelOutput[]`, …) without hard-coding by `task_type`.

#### Pagination

`GET /v1/jobs?state=SUCCEEDED&task_type=depth.monocular&limit=50&cursor=…`

- `cursor` opaque, base64 of `(created_at, id)`.
- Response: `{ items: [...], next_cursor: "…", has_more: true }`.

#### Cancel

`DELETE /v1/jobs/{id}` returns:

| Job state | Effect | HTTP |
|---|---|---|
| `QUEUED` | revoke from queue, mark `CANCELED` | `200` |
| `RUNNING` | `revoke(terminate=True, signal=SIGUSR1)`, mark `CANCELING` | `202` |
| `CANCELING` | no-op | `202` |
| terminal | none | `409 state_conflict` |

Worker SIGUSR1 handler frees GPU memory then transitions to `CANCELED`. If the worker is wedged, the reconciler force-finishes after heartbeat timeout.

### Webhooks

If `callback_url` is provided on submit, the dispatcher POSTs:

```http
POST <callback_url>
Content-Type: application/json
X-SAM3-Event: job.state_changed
X-SAM3-Signature: t=1714000000,v1=hex(hmac_sha256(secret, t + "." + body))
X-SAM3-Delivery-Id: 1c…
```

Body is the Job DTO at the moment of the event.

- Retries: 1, 5, 25, 125 s; max 5 attempts; then dead-lettered.
- Verification: client computes `hmac_sha256(secret, "{t}.{raw_body}")` and compares to `v1`. Reject if `t` is older than 5 min.

In the local profile the secret is `WEBHOOK_SECRET` from `.env`. Per-tenant rotation lives in the enterprise overlay.

### SSE

`GET /v1/jobs/{id}/events` (text/event-stream)

```
event: state
data: {"state":"RUNNING","at":"2026-04-25T10:00:01.300Z"}

event: state
data: {"state":"SUCCEEDED","at":"2026-04-25T10:00:31.900Z","output_type":"multiview_depth","artifacts":[...]}
```

Heartbeat comment line every 15 s. Connection closes after terminal state.

## Input safety

Enforced *before* enqueue, on the API tier:

- `Content-Length` ≤ `MAX_UPLOAD_BYTES` (default 50 MB).
- `Content-Type` allowlisted; magic-byte sniff at worker.
- `byte_length`, `width × height` capped (default 8192 × 8192).
- HEIC / AVIF off by default.
- `PIL.Image.MAX_IMAGE_PIXELS` set in `packages/core/imageguard.py`, eagerly imported.
- `Image.open` is **never** called in the API process.

`MultiViewImageInput.views` is bounded `2 ≤ N ≤ 16` at the schema level; the per-view byte cap is the same as single-image upload.

## Uploads

Image bytes never traverse FastAPI. The API only issues presigned PUT URLs and validates metadata. Worker fetches by `IORef.storage_key`. This is the single biggest security and OOM win over `master`, which decodes uploads in-process (`app/router.py`, `app/job_router.py`).

## Rate limiting

Token bucket per `(owner_id, bucket)`. Buckets:

| Bucket | Default | Notes |
|---|---|---|
| `upload` | 50 rps, burst 100 | presign issuance |
| `enqueue.gpu` | 30 rpm, burst 30 | new GPU jobs |
| `read` | 200 rps, burst 400 | GET status |

Implementation: Redis Lua atomic refill. On exhaust → `429 rate_limited` with `Retry-After`.

In the local profile the single `owner_id="local"` shares the buckets across all callers; the rate limiter mostly serves as a regression test for the algorithm.

## OpenAPI generation and SDKs

CI step: `python -m services.api.openapi --out openapi.json`. The diff against the previous tag is checked for breaking changes via `openapi-diff`. Breaking diff blocks the PR.

Two SDK surfaces are generated and published per release:

- **Python**: `openapi-generator-cli generate -g python` → `sdk/py/`, packaged as `sam3-client-py`.
- **TypeScript**: `openapi-generator-cli generate -g typescript-axios` → `sdk/ts/`, packaged as `@org/sam3-client`.

Plus the Sphinx API reference under `docs/` covers route docs **and** the typed I/O classes from `packages/io/`. The `docs.yml` workflow publishes both on tag.
