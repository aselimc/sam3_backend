# 05 — Worker Runtime

This document describes the Celery worker process: configuration, lifecycle, error handling, cancellation, tracing, and the Celery beat schedule.

It applies to both the **local profile** (one worker process per detected GPU on a single host, 1–2 GPUs) and the enterprise overlay (one process per GPU across many pods). Where the two diverge, it is called out inline.

The runner constructs typed `InputBase` instances per [04a-io-types.md](./04a-io-types.md) before invoking `adapter.infer(batch)`; the same typed contract holds in both profiles.

## Process model

- One worker process per GPU. `--concurrency=1` for GPU queues.
- Pool: `solo` (no fork). Forking after `torch` initialization breaks CUDA contexts.
- `--prefetch-multiplier=1` so a worker holds at most one extra unacked message.
- Separate pool for **non-GPU** queues (webhooks, beat targets) — `--concurrency=N`, gevent or threads. In the local profile this pool runs inside the same container as the GPU worker (sharing the Celery app); enterprise splits it into a dedicated `webhook-dispatcher` deployment.
- Local two-GPU layout: two worker containers, each with `CUDA_VISIBLE_DEVICES=<i>` and the `--queues` arg derived from `eligible_queues.py`. They share the broker and the warm-pool lock keys.

```bash
celery -A services.worker.main worker \
    --pool=solo \
    --concurrency=1 \
    --prefetch-multiplier=1 \
    --queues="$(python -m services.worker.eligible_queues)"
```

## Celery configuration

```python
# packages/broker/celery_app.py
celery.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    task_send_sent_event=True,
    worker_send_task_events=True,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    broker_transport_options={
        "visibility_timeout": 3600,           # 60 min (longest expected job)
        "global_keyprefix": "sam3:",
    },
    result_backend="redis://...",             # short-lived; we trust Postgres for history
    result_expires=3600,
    task_default_retry_delay=2,
    task_max_retries=3,
)
```

`acks_late=True` means a message is only ACKed after success. If the worker dies mid-job, Redis re-delivers. Combined with the SQL guard `UPDATE ... WHERE state='QUEUED' AND celery_task_id=:tid`, only one runner can claim the row; the loser ACKs and exits.

## Task entry

```python
# services/worker/main.py
@celery.task(bind=True, name="run_task", max_retries=3, autoretry_for=(TransientError,),
             retry_backoff=True, retry_backoff_max=120, retry_jitter=True)
def run_task(self, payload: dict):
    ctx = TaskContext.from_payload(payload, celery_task_id=self.request.id)
    runner.run(ctx)
```

`runner.run` is the place where everything wraps:

```python
# services/worker/runner.py
def run(ctx: TaskContext) -> None:
    with otel_extract(ctx.traceparent), bind_log(ctx):
        try:
            db.transition(ctx.job_id, from_=QUEUED, to=RUNNING, celery_task_id=ctx.celery_task_id)
            heartbeat.start(ctx.job_id)
            spec = task_registry.get(ctx.task_type)
            req  = spec.request_model.model_validate(ctx.request_payload)   # typed InputBase
            req.validate_with_caps(spec.required_capability_for(ctx.model_id).model_caps)
            spec.preflight(req, ctx.principal)

            adapter = warm_pool.ensure_loaded(ctx.model_id)
            with oom_guard(adapter.caps.per_request_gpu_mem_mb):
                raw = adapter.infer([spec.adapt(req, ctx)])[0]              # typed OutputBase

            result = spec.postprocess(raw, ctx)
            artifacts.upload(ctx, result)
            db.transition(ctx.job_id, from_=RUNNING, to=SUCCEEDED, result=result.summary())
            events.publish(ctx.job_id, "SUCCEEDED")
            webhooks.dispatch(ctx)
        except PreflightOOM as e:
            raise self.retry(exc=e, countdown=backoff(ctx.attempt))
        except CancelRequested:
            adapter_or_none = warm_pool.peek(ctx.model_id)
            adapter_or_none and adapter_or_none.healthcheck()
            db.transition(ctx.job_id, from_=CANCELING, to=CANCELED)
        except RuntimeOOM as e:
            ctx.bump_gpu_class()
            raise self.retry(exc=e, countdown=backoff(ctx.attempt))
        except Exception as e:
            db.transition(ctx.job_id, from_=RUNNING, to=FAILED, error=fmt_error(e))
            events.publish(ctx.job_id, "FAILED")
            webhooks.dispatch(ctx)
            raise        # let Celery NACK
        finally:
            heartbeat.stop(ctx.job_id)
```

## Cancellation

Two flavors driven by job state at the moment `DELETE /v1/jobs/{id}` arrives.

### Queued

API calls `app.control.revoke(celery_task_id, terminate=False)`. Celery removes the message from Redis; the worker never sees it. API transitions `QUEUED → CANCELED`.

### Running

API calls `app.control.revoke(celery_task_id, terminate=True, signal='SIGUSR1')`.

`SIGUSR1` is **not** the default `SIGTERM`. Reason: the worker installs a custom handler that:

1. sets a thread-local `cancel_requested = True` so the next `CancelCheck` raises `CancelRequested`;
2. is *not* installed by Python's default signal machinery for our pool;
3. avoids killing the process — we want to free the GPU cleanly and leave the worker available for the next job.

`CancelCheck` is sprinkled inside long loops (per-query within a multi-query SAM3 call, between batch chunks). The `_run_queries` loop in the legacy `app/sam3_service.py` is the obvious port location.

```python
# packages/core/cancel.py
class CancelCheck:
    def __init__(self): self._flag = False
    def trip(self): self._flag = True
    def __call__(self):
        if self._flag: raise CancelRequested()
```

If the worker is wedged (e.g. CUDA hang), the reconciler will fail the job after the heartbeat timeout; the pod will be killed by the K8s liveness probe. Documented operational expectation.

## Heartbeat

The runner spawns a background thread that updates `jobs.heartbeat_at = now()` every 10 s while the task runs.

```python
class Heartbeat:
    def start(self, job_id):
        self._stop = Event()
        Thread(target=self._loop, args=(job_id,), daemon=True).start()
    def _loop(self, job_id):
        while not self._stop.wait(10):
            db.execute("UPDATE jobs SET heartbeat_at=now() WHERE id=:id", id=job_id)
```

## Reconciler

A Celery beat schedule runs every 60 s on the leader. The leader is elected by `SETNX lock:beat … PX 30000`.

Idempotent SQL:

```sql
UPDATE jobs
SET state = 'FAILED',
    error_code = 'orphan',
    error_detail = 'no heartbeat for 90s',
    finished_at = now()
WHERE state IN ('RUNNING', 'CANCELING')
  AND heartbeat_at < now() - interval '90 seconds';
```

Plus a sweep for stuck `CANCELING` (`> 5 min`) → forced `CANCELED`. And a `webhook_deliveries` retry sweep that pushes due rows back onto the dispatcher queue.

## Lifecycle and graceful shutdown

```python
# services/worker/signals.py
def on_sigterm(*_):
    state.draining = True            # stop prefetching new
    # current task continues to completion; runner sees draining flag
    # at the next batch boundary or CancelCheck

def on_term(sender, **_):
    warm_pool.unload_all()
    torch.cuda.empty_cache()
```

Kubernetes setup:

- `terminationGracePeriodSeconds: 600` (≥ longest expected job).
- `preStop`: `sh -c "kill -TERM $(pidof -s celery)"`.
- `livenessProbe`: hits a tiny TCP server on the worker that returns OK if the heartbeat thread is alive.
- `readinessProbe`: returns OK only if the warm pool has loaded all `WORKER_ENABLED_MODELS` (when `WORKER_PRELOAD=true`).

## Tracing

W3C trace context propagated across the queue manually:

```python
# packages/broker/trace.py
def submit_with_trace(celery_app, task_name, payload, queue):
    headers = {}
    inject(headers)                          # opentelemetry.propagate
    payload["traceparent"] = headers.get("traceparent")
    payload["tracestate"]  = headers.get("tracestate")
    celery_app.send_task(task_name, args=[payload], queue=queue)

# worker side
def otel_extract(payload):
    ctx = extract({"traceparent": payload.get("traceparent",""),
                   "tracestate":  payload.get("tracestate","")})
    return tracer.start_as_current_span("worker.run_task", context=ctx)
```

Span attributes set by the worker: `job.id`, `tenant.id`, `model.id`, `gpu.class`, `attempt`, `batch.size`, `gpu.mem_used_mb`.

## Error taxonomy

| Type | Class | Retry? | DB error_code |
|---|---|---|---|
| Validation | `ValidationError` | no | `validation_error` |
| Preflight OOM | `PreflightOOM` | yes (no attempt cost) | `preflight_oom` |
| Runtime OOM | `RuntimeOOM` | yes, bump `gpu_class` | `runtime_oom` |
| HF/S3 transient | `TransientError` | yes | `transient_io` |
| Cancellation | `CancelRequested` | no | `canceled` |
| Adapter bug | `AdapterError` | no | `adapter_error` |
| Unknown | `Exception` | no | `internal` |

`PreflightOOM` retry does **not** consume `attempt` because the task never started; the requeue is a routing fix, not a failure.

## Webhook dispatcher

A separate Celery task `dispatch_webhook(job_id, attempt)` runs on a non-GPU queue.

- Computes `X-SAM3-Signature` with the tenant-specific secret (rotated by an admin endpoint; a key-id prefix in the signature header allows rolling rotation).
- Body is the Job DTO at the moment the task fires (snapshot, not live read).
- Retries: `delay = min(125, 5**attempt)`; max 5 attempts; logs row to `webhook_deliveries`.

## Testing the runtime

- `tests/integration/test_acks_late.py`: kill the worker mid-task with `os.kill(pid, SIGKILL)`; assert message redelivers and final state is one of `SUCCEEDED`/`FAILED` (never lost).
- `tests/integration/test_cancel_running.py`: submit a synthetic slow task that polls `CancelCheck`; cancel; assert `CANCELED`, GPU free, worker still serving.
- `tests/integration/test_oom_bump.py`: submit a job whose `per_request_gpu_mem_mb` is intentionally too high for `t4_16g`; assert it gets re-routed to `a100_40g` after one retry.
