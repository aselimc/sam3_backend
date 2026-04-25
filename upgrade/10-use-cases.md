# 10 — Use Cases and SDK Examples

This document walks the canonical user journeys end-to-end for the **local profile**, then provides shaped SDK examples in Python and TypeScript. The goal is that an integrator can read this file and ship working code without reading the OpenAPI spec.

The v2.0 surface is the closed task set:

- `segmentation.text`, `segmentation.point`, `segmentation.box` — backed by SAM3.
- `depth.monocular`, `depth.multiview` — backed by Depth Anything 3.

Every input and output below comes from the typed I/O catalogue in [04a-io-types.md](./04a-io-types.md). Tenant-admin / multi-key flows (`/v1/auth/api-keys`) are an enterprise concern documented in [`enterprise/01-multi-tenancy-and-auth.md`](../enterprise/01-multi-tenancy-and-auth.md).

## Personas

- **Web-app integrator** — embeds segmentation in a SaaS dashboard. Cares about latency, UI updates, error handling.
- **Pipeline operator** — runs nightly batch jobs against thousands of images. Cares about throughput, reliability, cost.
- **Researcher** — explores models interactively from a notebook. Cares about ergonomics and quick iteration.

## Use case 1 — Single-image text segmentation, polled

The simplest happy path. Used by the web-app integrator with SAM3.

```text
1. Create an upload   → POST /v1/uploads
2. PUT bytes to S3    → PUT  <upload_url>
3. Submit a task      → POST /v1/tasks/segmentation.text  (ImageTextInput)
4. Poll until done    → GET  /v1/jobs/{id}
5. Download artifacts → GET  /v1/jobs/{id}/artifacts → presigned GETs
```

Non-functional notes:

- Presigned PUT TTL is 15 min. Client must complete the upload in that window.
- Polling cadence: 1 s for first 10 s, then 5 s. Use SSE (use case 2) for tighter UI updates.
- `Idempotency-Key` on submit is required. The SDK generates one per call by default.

## Use case 2 — Streaming progress with SSE

```text
1–3. As above.
4. SSE stream  →  GET /v1/jobs/{id}/events  (text/event-stream)
   - Events: state=RUNNING, state=SUCCEEDED with embedded artifact list and output_type
   - Heartbeat comments every 15 s
5. Stream closes after terminal state.
```

The browser case wants SSE because it avoids the cost of recurring polls, and the connection survives across the typical 10–30 s of an inference. Mobile clients should still poll (SSE on flaky networks is worse than polling).

## Use case 3 — Webhook-driven batch pipeline

Used by the pipeline operator. Avoids holding many HTTP connections open.

```text
1. For each image:
   a. POST /v1/uploads
   b. PUT bytes
   c. POST /v1/tasks/<task_type> with callback_url and Idempotency-Key
2. Server queues, runs, then POSTs the Job DTO to callback_url with X-SAM3-Signature.
3. Receiver verifies signature, fetches artifacts via the embedded presigned URLs, marks job done in own pipeline.
```

The webhook secret is `WEBHOOK_SECRET` from `.env` in the local profile. Per-tenant secret rotation with `kid=` is enterprise.

## Use case 4 — Cancel a long-running job

```text
1. Submit (any task that takes more than a few seconds — DA3 multi-view fits).
2. Client decides to cancel.
3. DELETE /v1/jobs/{id}
   - 200 if QUEUED → CANCELED instantly
   - 202 if RUNNING → state=CANCELING, transitions to CANCELED within seconds
4. Subsequent GET shows CANCELED.
```

The system guarantees that cancellation frees GPU memory before the worker accepts the next task; a misbehaved adapter that does not honor `CancelCheck` will be hard-killed by the reconciler if the cancel takes longer than 5 min.

## Use case 5 — Multi-prompt with selective post-processing

The legacy `regularize` flag survives in v2 unchanged, exposed per-query inside `ImageTextInput`. Useful for remote-sensing workflows where building footprints want orthogonal polygons but vehicles do not.

```jsonc
POST /v1/tasks/segmentation.text
{
  "version": "1",
  "input_type": "image_text",
  "image": { "storage_key": "s3://.../uploads/local/...", "content_type": "image/jpeg" },
  "queries": [
    { "text": "building", "regularize": true,  "confidence_threshold": 0.6 },
    { "text": "car",      "regularize": false, "confidence_threshold": 0.5 },
    { "text": "tree",     "regularize": false }
  ]
}
```

Result: one or more `MaskLabelOutput` per query, with masks regularized when the flag is on.

## Use case 6 — Researcher in a Jupyter notebook

```python
from sam3_client import Client
import io
from PIL import Image

c = Client(api_key=os.environ["LOCAL_API_KEY"], base_url="http://localhost:8000")

img = Image.open("garden.jpg")
job = c.tasks.segmentation.text.submit(
    image=img,
    queries=[{"text": "cat"}, {"text": "person", "confidence_threshold": 0.6}],
)
job.wait()                                  # polls with sane backoff
for mask in job.result:                     # list[MaskLabelOutput]
    Image.open(io.BytesIO(mask.mask_bytes())).show()
    print(mask.label, mask.score)
```

The SDK transparently:

- Issues the presign and PUT.
- Sets `Idempotency-Key` from a UUID.
- Polls or subscribes to SSE based on the environment (notebook → SSE; CLI → poll).
- Decodes mask bytes from the presigned GET on demand.

## Use case 7 — Monocular depth from a single photo (DA3)

```python
from sam3_client import Client

c = Client(api_key=os.environ["LOCAL_API_KEY"])

job = c.tasks.depth.monocular.submit(image_path="kitchen.jpg")
job.wait()

depth = job.result                          # DepthMapOutput
print(depth.units, depth.min_depth, depth.max_depth)

with open("out/depth.png", "wb") as f:
    f.write(depth.depth_bytes())            # SDK fetches presigned GET on demand
```

`DepthMapOutput.units` is `"relative"` by default. If your DA3 install supports metric depth, the adapter sets `"meters"` and populates `min_depth` / `max_depth` accordingly.

## Use case 8 — Multi-view depth from a sparse capture (DA3)

The interesting DA3 path. Three handheld photos around an object → per-view depth, camera intrinsics/extrinsics, and an optional fused pointcloud.

```python
from sam3_client import Client

c = Client(api_key=os.environ["LOCAL_API_KEY"])

job = c.tasks.depth.multiview.submit(
    images=["v0.jpg", "v1.jpg", "v2.jpg"],
    camera_hints={"focal_length_mm": 28.0, "sensor_width_mm": 36.0},
)
job.wait()

mv = job.result                             # MultiViewDepthOutput

for i, dm in enumerate(mv.per_view):
    with open(f"out/depth_{i}.png", "wb") as f:
        f.write(dm.depth_bytes())

K = mv.cameras.intrinsics                   # 3x3
poses = mv.cameras.extrinsics               # list of 4x4

if mv.pointcloud is not None:
    with open("out/cloud.ply", "wb") as f:
        f.write(mv.pointcloud.bytes())
```

Notes:
- `views` is bounded `2 ≤ N ≤ 16`. Beyond 16, split the capture and fuse client-side.
- DA3 multi-view is **not** batched across requests on the worker (each request is already an internal batch over views), so per-request latency is dominated by N.

## Use case 9 — Failure: rate limited

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 12
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1714000020000
Content-Type: application/json

{"error":{"code":"rate_limited","message":"Too many requests for bucket enqueue.gpu","details":{"retry_after_s":12}}}
```

Client logic: respect `Retry-After`. SDK does this automatically with a single retry; for sustained pressure the client raises `RateLimitedError` so the caller can decide to back off.

## Use case 10 — Failure: oversized input

```http
HTTP/1.1 413 Payload Too Large
{"error":{"code":"payload_too_large","message":"Image exceeds local MAX_UPLOAD_BYTES of 50 MB"}}
```

This rejection happens at `POST /v1/uploads` — before any bytes traverse the system. A misbehaved client that PUTs an oversized object directly will be rejected by the bucket policy (`Content-Length` enforced).

## Use case 11 — Adding a new model family (operator)

End-to-end of how a new model becomes a callable task:

1. Add the upstream repo as a submodule under `third_party/<name>`.
2. Implement `ModelAdapter` in `packages/models/<name>/adapter.py`.
3. Declare `ModelCapabilities` (tasks satisfied, GPU budget, batch size). Reference existing I/O classes from `packages/io/`; if you need a new class, add it under `packages/io/inputs/` or `outputs/` and re-export.
4. Pin a HF revision in `weights.py`.
5. Bake into the worker image via `BAKE_MODELS=<name>` (optional).
6. Add `<name>` to `MODELS_ENABLED` in `.env`.
7. Restart the worker. The new task type appears in `GET /v1/models` and any new I/O classes appear in `GET /v1/io/types` — automatically.

No API code changes. No new routers. No DB migration unless the task introduces fields the generic `request_payload` JSONB cannot represent — which it usually can.

## SDK examples

### Python — `sam3-client-py`

```python
from sam3_client import Client, RateLimitedError, JobFailed

c = Client(api_key=os.environ["LOCAL_API_KEY"], base_url="http://localhost:8000")

# Segmentation
job = c.tasks.segmentation.text.submit(
    image_path="garden.jpg",
    queries=[{"text": "cat"}, {"text": "person", "confidence_threshold": 0.5, "regularize": True}],
)
try:
    job.wait(timeout=120)
except JobFailed as e:
    print("Failed:", e.error_code, e.error_detail)
else:
    for mask in job.result:
        with open(f"out/{mask.label}_{mask.score:.2f}.png", "wb") as f:
            f.write(mask.mask_bytes())

# Depth (monocular)
job = c.tasks.depth.monocular.submit(image_path="kitchen.jpg")
job.wait()
print(job.result.min_depth, job.result.max_depth, job.result.units)

# Stream events
for evt in c.jobs.events(job.id):
    print(evt.state, evt.at)
```

### TypeScript — `@org/sam3-client`

```ts
import { Sam3Client } from "@org/sam3-client";

const c = new Sam3Client({
  apiKey: process.env.LOCAL_API_KEY!,
  baseUrl: "http://localhost:8000",
});

// Segmentation
const file = await fetch("/garden.jpg").then(r => r.blob());
const job = await c.tasks.segmentation.text.submit({
  imageBlob: file,
  queries: [
    { text: "cat" },
    { text: "person", confidenceThreshold: 0.5, regularize: true },
  ],
});

const events = c.jobs.events(job.id);
for await (const e of events) {
  if (e.state === "SUCCEEDED") {
    for (const mask of e.output) {
      const blob = await fetch(mask.url).then(r => r.blob());
      renderMask(blob, mask.label);
    }
    break;
  } else if (e.state === "FAILED") {
    throw new Error(e.error.code);
  }
}

// Depth (multi-view)
const dv = await c.tasks.depth.multiview.submit({
  imageBlobs: [v0, v1, v2],
  cameraHints: { focalLengthMm: 28, sensorWidthMm: 36 },
});
const result = await dv.wait();        // MultiViewDepthOutput
console.log(result.cameras.intrinsics);
```

### curl — minimal smoke

```bash
KEY=$(grep '^LOCAL_API_KEY=' .env | cut -d= -f2)
BASE=http://localhost:8000

# 1. Get presigned PUT
UP=$(curl -s -X POST $BASE/v1/uploads \
        -H "X-API-Key: $KEY" -H "Idempotency-Key: $(uuidgen)" \
        -d '{"filename":"g.jpg","content_type":"image/jpeg","byte_length":'$(stat -c%s g.jpg)'}')
URL=$(echo "$UP" | jq -r .upload_url)
REF=$(echo "$UP" | jq -r .image_ref)

# 2. PUT bytes
curl -s -X PUT -T g.jpg -H "Content-Type: image/jpeg" "$URL"

# 3. Submit (segmentation.text)
JOB=$(curl -s -X POST $BASE/v1/tasks/segmentation.text \
        -H "X-API-Key: $KEY" -H "Idempotency-Key: $(uuidgen)" \
        -d "{\"version\":\"1\",\"input_type\":\"image_text\",\"image\":{\"storage_key\":\"$REF\",\"content_type\":\"image/jpeg\"},\"queries\":[{\"text\":\"cat\"}]}" \
        | jq -r .job_id)

# 4. Poll
while :; do
  S=$(curl -s -H "X-API-Key: $KEY" $BASE/v1/jobs/$JOB | jq -r .state)
  echo $S; [[ $S == SUCCEEDED || $S == FAILED || $S == CANCELED ]] && break
  sleep 1
done

# 5. Download
curl -s -H "X-API-Key: $KEY" $BASE/v1/jobs/$JOB/artifacts | jq
```

### Webhook receiver — Python (signature verification)

```python
import hmac, hashlib, time, os
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()
SECRET = os.environ["WEBHOOK_SECRET"].encode()

@app.post("/sam3-webhook")
async def receive(req: Request):
    sig = req.headers["x-sam3-signature"]              # "t=…,v1=…"
    parts = dict(p.split("=") for p in sig.split(","))
    t = int(parts["t"])
    if abs(time.time() - t) > 300:
        raise HTTPException(400, "stale")
    body = await req.body()
    expected = hmac.new(SECRET, f"{t}.".encode() + body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, parts["v1"]):
        raise HTTPException(401, "bad signature")
    payload = await req.json()
    print("job", payload["id"], payload["task_type"], "→", payload["state"], "output:", payload.get("output_type"))
    return {"ok": True}
```
