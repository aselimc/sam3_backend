# 04 — Model Adapter and Task Layer

This document defines the two pluggability surfaces of v2: the **model adapter** (how a model family is integrated) and the **task** (what the API exposes). They are intentionally separate so a single adapter can satisfy multiple tasks (SAM3 → text/point/box segmentation) and a single task can be satisfied by multiple adapters in the future.

The set of supported I/O classes is defined in [04a-io-types.md](./04a-io-types.md). This document focuses on adapter mechanics, queue routing, batching, and warm-pool semantics.

At v2.0, exactly two adapters ship:

- **SAM3** ([facebookresearch/sam3](https://github.com/facebookresearch/sam3)) — segmentation.
- **Depth Anything 3** ([ByteDance-Seed/depth-anything-3](https://github.com/ByteDance-Seed/depth-anything-3)) — monocular and multi-view depth.

Both are vendored as git submodules under `third_party/`.

## Layering

```
TaskRequest  ──►  TaskSpec  ──► resolves to ──►  ModelAdapter (concrete)
                  (task)                          (model)
                  declares: input/output class    declares: capabilities, GPU budget
                  pre/post-processing             load / unload / infer
```

## Model adapter

### Contract

```python
# packages/models/base.py

class GpuClass(StrEnum):
    CPU       = "cpu"            # tests, tiny adapters
    T4_16G    = "t4_16g"
    L4_24G    = "l4_24g"
    A10_24G   = "a10_24g"
    A100_40G  = "a100_40g"
    A100_80G  = "a100_80g"
    H100_80G  = "h100_80g"

class TypedCapability(BaseModel):
    task: TaskType
    input_class:  type[InputBase]          # from packages.io.inputs
    output_class: type[OutputBase] | type[list]  # see 04a §How adapters declare I/O

class ModelCapabilities(BaseModel):
    model_id: str
    capabilities: list[TypedCapability]
    min_gpu_classes: list[GpuClass]        # cheapest → fattest hosts
    load_gpu_mem_mb: int                   # static load footprint
    per_request_gpu_mem_mb: int            # working-set per inference
    max_input_pixels: int
    supports_fp16: bool
    supports_batching: bool
    max_batch_size: int = 1
    cold_start_seconds_p95: float

class ModelAdapter(Protocol):
    caps: ClassVar[ModelCapabilities]
    def load(self, device: str) -> None: ...
    def unload(self) -> None: ...
    def infer(self, batch: list[InputBase]) -> list[OutputBase]: ...
    def healthcheck(self) -> None: ...
```

`infer` takes a list of typed inputs and returns a list of typed outputs of the same length, in order. The TaskSpec layer is responsible for constructing those typed inputs from the validated request and for any post-processing (e.g. polygon regularization).

### Initial adapters

#### SAM3 (`packages/models/sam3/adapter.py`)

```python
@register_model
class Sam3Adapter:
    caps = ModelCapabilities(
        model_id="sam3",
        capabilities=[
            TypedCapability(TaskType.SEGMENTATION_TEXT,  ImageTextInput,  list[MaskLabelOutput]),
            TypedCapability(TaskType.SEGMENTATION_POINT, ImagePointInput, list[MaskLabelOutput]),
            TypedCapability(TaskType.SEGMENTATION_BOX,   ImageBoxInput,   list[MaskLabelOutput]),
        ],
        min_gpu_classes=[GpuClass.A10_24G, GpuClass.A100_40G, GpuClass.H100_80G],
        load_gpu_mem_mb=4500,
        per_request_gpu_mem_mb=2200,
        max_input_pixels=4096*4096,
        supports_fp16=True,
        supports_batching=True,
        max_batch_size=4,
        cold_start_seconds_p95=35.0,
    )
```

The legacy `app/sam3_service.py` is the port source. The legacy `app/regularization.py` is reused verbatim under `packages/tasks/segmentation/post/regularize.py` and called from the SAM3 task spec, not the adapter — keeps the adapter pure.

#### Depth Anything 3 (`packages/models/depth_anything_v3/adapter.py`)

```python
@register_model
class DepthAnythingV3Adapter:
    caps = ModelCapabilities(
        model_id="depth_anything_v3",
        capabilities=[
            TypedCapability(TaskType.DEPTH_MONOCULAR, ImageInput,          DepthMapOutput),
            TypedCapability(TaskType.DEPTH_MULTIVIEW, MultiViewImageInput, MultiViewDepthOutput),
        ],
        min_gpu_classes=[GpuClass.L4_24G, GpuClass.A10_24G, GpuClass.A100_40G],
        load_gpu_mem_mb=3200,
        per_request_gpu_mem_mb=1800,
        max_input_pixels=2048*2048,
        supports_fp16=True,
        supports_batching=True,
        max_batch_size=2,
        cold_start_seconds_p95=20.0,
    )
```

Multi-view is the load-heavier branch: the adapter expects 2–16 views per request and returns per-view depth + camera params + an optional fused pointcloud.

### Adding a new model

1. `git submodule add <url> third_party/<name>`.
2. `packages/models/<name>/adapter.py` implementing `ModelAdapter`.
3. Decorate with `@register_model`. Declare `ModelCapabilities`.
4. Add `<model_id>` to `MODELS_ENABLED` env on the worker.
5. Add `<model_id>` weights to the bake step if you want zero-cold-start (`BAKE_MODELS=…` build-arg).

No API code change. The capability advertisement exposes the task automatically. The I/O classes the adapter references must already exist in `packages/io/`; if not, see [04a-io-types.md §Adding a new I/O class](./04a-io-types.md#adding-a-new-io-class).

## Task layer

### Contract

```python
# packages/tasks/base.py

class TaskSpec(Generic[Req, Res]):
    task_type: TaskType
    request_model: type[Req]                    # subclass of InputBase
    result_model:  type[Res]                    # subclass of OutputBase or list[OutputBase]
    required_capability: TypedCapability

    def preflight(self, req: Req, principal: Principal) -> None: ...
    def adapt(self, req: Req, ctx: TaskContext) -> InputBase: ...
    def postprocess(self, raw: OutputBase | list[OutputBase], ctx: TaskContext) -> Res: ...
```

`TaskSpec` is a thin orchestrator. Heavy lifting is in the adapter; reusable image-ops live in `packages/tasks/<family>/post/`.

### v2.0 task catalogue

| `task_type` | Adapter | Input class | Output class |
|---|---|---|---|
| `segmentation.text`  | `sam3` | `ImageTextInput`  | `list[MaskLabelOutput]` |
| `segmentation.point` | `sam3` | `ImagePointInput` | `list[MaskLabelOutput]` |
| `segmentation.box`   | `sam3` | `ImageBoxInput`   | `list[MaskLabelOutput]` |
| `depth.monocular`    | `depth_anything_v3` | `ImageInput`          | `DepthMapOutput` |
| `depth.multiview`    | `depth_anything_v3` | `MultiViewImageInput` | `MultiViewDepthOutput` |

This is the **closed set** at v2.0. Adding a new row requires:
1. an adapter that declares the task in its `TypedCapability` list, and
2. a `TaskSpec` subclass under `packages/tasks/<family>/<verb>.py`.

Until both exist, the task is invisible on the public API.

## Capability resolution

When `POST /v1/tasks/{task_type}` arrives:

1. Validate body against `TaskSpec.request_model`.
2. Determine candidate adapters: `registry.find(task=task_type)` filtered by `model_id` if specified.
3. Determine `gpu_class`: explicit → used as-is; else cheapest of `caps.min_gpu_classes` for which a worker is currently advertising readiness.
4. Resolve queue name `task.<task_type>.<gpu_class>`.
5. If no eligible queue has a worker → `503 model_unavailable`.

## Queue routing

Queues are named `task.<task_type>.<gpu_class>`.

Workers compute eligible queues at boot from `MODELS_ENABLED` × the GPU they detect, then subscribe only to those.

```python
# services/worker/bootstrap.py
def eligible_queues(gpu: GpuClass, enabled_models: list[str]) -> list[str]:
    queues: set[str] = set()
    for mid in enabled_models:
        adapter = registry.get(mid)
        if gpu not in adapter.caps.min_gpu_classes:
            continue
        for cap in adapter.caps.capabilities:
            queues.add(f"task.{cap.task}.{gpu}")
    return sorted(queues)
```

Two priority queues per `task_type × gpu_class`: `…hi` and `…default`. Premium tenants (configured in `tenants.config.priority="hi"`, see enterprise track) route to `…hi`. In the default local profile only `…default` exists.

### Multi-GPU on one host

Local dev targets one or two GPUs. The worker process pins to a specific device via `CUDA_VISIBLE_DEVICES`; a host with two GPUs runs two worker processes (one per GPU) with disjoint `CUDA_VISIBLE_DEVICES`. `docker-compose.yml` declares two `worker-gpu-{0,1}` services that activate when the second GPU is detected; otherwise only `worker-gpu-0` runs. Detection lives in `scripts/bootstrap_dev.{ps1,sh}`.

## Batching

Within a worker process, a per-queue micro-batcher accumulates compatible requests (same `model_id`, same `gpu_class`, identical hyperparameters) and flushes on:

- `max_batch_size` reached, **or**
- `max_wait_ms` elapsed since first request (default 25 ms), **or**
- estimated GPU memory of next add would exceed budget.

The batcher is transparent to the task layer: it sits between Celery's `task_prerun` and `TaskSpec.adapt`. Only adapters with `supports_batching=True` are eligible; others bypass.

```
Celery task   ─► batcher.submit(req)  ─► returns Future
                                ▲
   batcher tick (max_wait_ms or full)
                                │
                                ▼
                     adapter.infer(batch)
                                │
                                ▼
                       per-result Future.set()
```

Batching is opt-in per queue via `WORKER_BATCHING={ "task.segmentation.text.a100_40g": {"max_batch_size":4,"max_wait_ms":25} }`.

DA3 multi-view requests are not batched across requests (each request is already an internal batch over views); the flag is `False` for that queue.

## Warm-pool and eviction

A worker can host more than one model only if all of `caps.load_gpu_mem_mb` sums fit on the GPU.

```python
# packages/models/device.py

class WarmPool:
    def __init__(self, total_mem_mb: int, headroom_mb: int): ...
    def ensure_loaded(self, model_id: str) -> ModelAdapter:
        if model_id in self.loaded: return self.loaded[model_id]
        with redis_lock(f"model:{model_id}", ttl=600):
            while not self.fits(model_id):
                victim = self.lru_pop()
                victim.unload(); torch.cuda.empty_cache()
            adapter = registry.get(model_id)
            adapter.load(self.device)
            self.loaded[model_id] = adapter
            return adapter
```

- LRU only between models declared in `MODELS_ENABLED`.
- The Redis lock per `model_id` prevents thundering-herd HF downloads when N workers boot simultaneously.
- Eviction logged with reason and freed mem; metric `worker_model_evictions_total{model_id,reason}` emitted.

On a 24 GB GPU, both SAM3 (4.5 GB load) and DA3 (3.2 GB load) fit warm with headroom for one inference each. The default local profile preloads both when a single GPU is present.

## OOM guard

```python
# packages/models/device.py

@contextmanager
def oom_guard(per_request_mb: int, headroom_mb: int = 512):
    free, total = torch.cuda.mem_get_info()
    free_mb = free // (1024*1024)
    if per_request_mb + headroom_mb > free_mb:
        raise PreflightOOM(need=per_request_mb, free=free_mb)
    try:
        yield
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        raise RuntimeOOM(str(e)) from e
```

`PreflightOOM` → reject task without consuming retry attempt; broker re-queues. `RuntimeOOM` → consume one retry attempt, requeue with `gpu_class` bumped one tier if available, else `FAILED`.

## Cold-start cost and how we hide it

- HF download on first run is bracketed by the Redis lock so only one worker downloads per cluster.
- `Dockerfile.worker` optionally bakes weights via `huggingface-cli download` at build time, gated by build-arg `BAKE_MODELS=sam3,depth_anything_v3`.
- A `WORKER_PRELOAD=true` env causes `services/worker/bootstrap.py` to eagerly call `WarmPool.ensure_loaded` for every enabled model during readiness gating; the pod stays `NotReady` until all are loaded. Trade-off documented; recommended in enterprise prod, off in tests, default off in local.

## Capabilities surfaced through the API

`GET /v1/models` returns the capabilities advertised by at least one currently-ready worker. Useful for SDKs to discover what is callable without hard-coding model IDs.

```json
{
  "models": [
    {
      "model_id": "sam3",
      "ready": true,
      "available_gpu_classes": ["a100_40g"],
      "capabilities": [
        { "task": "segmentation.text",  "input_class": "ImageTextInput",  "output_class": "list[MaskLabelOutput]" },
        { "task": "segmentation.point", "input_class": "ImagePointInput", "output_class": "list[MaskLabelOutput]" },
        { "task": "segmentation.box",   "input_class": "ImageBoxInput",   "output_class": "list[MaskLabelOutput]" }
      ]
    },
    {
      "model_id": "depth_anything_v3",
      "ready": true,
      "available_gpu_classes": ["a100_40g"],
      "capabilities": [
        { "task": "depth.monocular", "input_class": "ImageInput",          "output_class": "DepthMapOutput" },
        { "task": "depth.multiview", "input_class": "MultiViewImageInput", "output_class": "MultiViewDepthOutput" }
      ]
    }
  ]
}
```

`GET /v1/io/types` returns the I/O Pydantic schemas — see [04a-io-types.md](./04a-io-types.md).

Readiness signal is supplied by workers via Redis SETEX heartbeat: `worker:{worker_id}:ready` with the capability JSON, TTL 30 s, refreshed every 10 s.
