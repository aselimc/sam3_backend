# 04a — Typed I/O Class Hierarchy

This document defines the typed input and output class system for inference. It is the single source of truth for `packages/io/`. Adapters declare which I/O classes they support; the API surfaces only the classes that at least one loaded adapter claims (**no overshooting**).

It exists because the previous `Modality` string enum (in [04-model-and-tasks.md](./04-model-and-tasks.md)) is too coarse. `Modality.IMAGE` cannot tell the difference between a bare image, an image + text prompt, an image + IMU pair, or an image + paired pointcloud — yet those are different input shapes with different validators, different storage layouts, and different SDK ergonomics. Typed I/O classes fix that.

## Goals

- **Logical, not physical.** A class names a *combination of signals* the model expects, not a file format.
- **Closed set per release.** Only classes claimed by at least one adapter exist on the public surface.
- **Pydantic-first.** Every class is a `BaseModel` so we get free OpenAPI emission and SDK generation.
- **Forward-compatible.** Adding a class is additive; renaming requires a `version` bump on the model.

## Layout

```
packages/io/
├── __init__.py            # public re-exports
├── base.py                # InputBase, OutputBase, IORef, ContentDescriptor
├── refs.py                # S3Ref, InlineBytes, FileRef helpers
├── inputs/
│   ├── image.py           # ImageInput, ImageTextInput, ImagePointInput, ImageBoxInput
│   ├── video.py           # VideoInput
│   ├── imu.py             # ImageImuPair
│   ├── pointcloud.py      # PointCloudInput, ImagePointCloudInput
│   ├── multiview.py       # MultiViewImageInput
│   └── action.py          # ImageActionPair
├── outputs/
│   ├── mask.py            # MaskLabelOutput, SegmentationMapOutput
│   ├── bbox.py            # BBoxOutput
│   ├── classification.py  # ClassOutput
│   ├── text.py            # TextOutput
│   ├── pose.py            # PoseOutput
│   ├── depth.py           # DepthMapOutput
│   ├── camera.py          # CameraParametersOutput
│   ├── pointcloud.py      # PointCloudOutput
│   └── composite.py       # MultiViewDepthOutput, et al.
└── registry.py            # IORegistry — discoverable from adapter caps
```

## Base contract

```python
# packages/io/base.py

class IORef(BaseModel):
    """Reference to a payload stored in object storage. Bytes are never
    embedded directly in input/output models — refs are the contract."""
    storage_key: str                            # s3://bucket/.../uuid
    content_type: str                           # MIME
    byte_length: int | None = None
    checksum_sha256: str | None = None

class InputBase(VersionedModel):
    """All inputs share: a stable type tag and the same validation hook."""
    input_type: ClassVar[str]                   # e.g. "image_text"
    def validate_with_caps(self, caps: ModelCapabilities) -> None: ...

class OutputBase(VersionedModel):
    output_type: ClassVar[str]
    def serialize_artifacts(self) -> list[ArtifactSpec]: ...
```

`VersionedModel` (defined in `packages/core/schemas.py`) carries `version: Literal["1"]` so additive changes are safe and breaking changes force a path bump.

## Input classes

| Class | `input_type` | Fields | Used by |
|---|---|---|---|
| `ImageInput` | `image` | `image: IORef` | DA3 (monocular) |
| `ImageTextInput` | `image_text` | `image: IORef`, `queries: list[TextQuery]` | SAM3 (text) |
| `ImagePointInput` | `image_point` | `image: IORef`, `points: list[Point2DLabel]` | SAM3 (point) |
| `ImageBoxInput` | `image_box` | `image: IORef`, `boxes: list[BBox2D]` | SAM3 (box) |
| `MultiViewImageInput` | `multiview_image` | `views: list[ImageView]`, `camera_hints: CameraHints?` | DA3 (multi-view) |
| `VideoInput` | `video` | `video: IORef`, `fps: float?`, `time_range_s: tuple?` | reserved (no adapter at v2.0) |
| `ImageImuPair` | `image_imu` | `image: IORef`, `imu_samples: list[ImuSample]` | reserved |
| `PointCloudInput` | `pointcloud` | `pointcloud: IORef`, `point_format: PointFormat` | reserved |
| `ImagePointCloudInput` | `image_pointcloud` | `image: IORef`, `pointcloud: IORef`, `extrinsic: Pose6D?` | reserved |
| `ImageActionPair` | `image_action` | `image: IORef`, `action: list[float]`, `action_space: ActionSpace` | reserved |

`reserved` = the class exists in code (so contributors can add an adapter without growing the type system) but is **not exposed on the public API surface** until an adapter declares it. The `IORegistry` filter is the gate.

### Field detail (the non-trivial ones)

```python
# packages/io/inputs/image.py

class TextQuery(BaseModel):
    text: str
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    regularize: bool = False              # SAM3 polygon regularization

class Point2DLabel(BaseModel):
    xy: tuple[float, float]
    label: Literal[0, 1]                  # 0 = background, 1 = foreground

class BBox2D(BaseModel):
    xyxy: tuple[float, float, float, float]
    label: str | None = None

class ImageInput(InputBase):
    input_type = "image"
    image: IORef

class ImageTextInput(InputBase):
    input_type = "image_text"
    image: IORef
    queries: list[TextQuery]

class ImagePointInput(InputBase):
    input_type = "image_point"
    image: IORef
    points: list[Point2DLabel]

class ImageBoxInput(InputBase):
    input_type = "image_box"
    image: IORef
    boxes: list[BBox2D]
```

```python
# packages/io/inputs/multiview.py

class ImageView(BaseModel):
    image: IORef
    intrinsic: list[list[float]] | None = None    # 3x3 K
    extrinsic: list[list[float]] | None = None    # 4x4 [R|t]

class CameraHints(BaseModel):
    """Optional priors that help DA3 with sparse-view depth."""
    focal_length_mm: float | None = None
    sensor_width_mm: float | None = None

class MultiViewImageInput(InputBase):
    input_type = "multiview_image"
    views: list[ImageView] = Field(min_length=2, max_length=16)
    camera_hints: CameraHints | None = None
```

## Output classes

| Class | `output_type` | Fields | Produced by |
|---|---|---|---|
| `MaskLabelOutput` | `mask_label` | `mask: IORef`, `label: str?`, `score: float`, `bbox: BBox2D?` | SAM3 |
| `SegmentationMapOutput` | `segmap` | `map: IORef`, `palette: list[RgbColor]?`, `class_names: list[str]?` | reserved |
| `BBoxOutput` | `bbox` | `boxes: list[ScoredBBox]` | reserved |
| `ClassOutput` | `class` | `top_k: list[ClassPrediction]` | reserved |
| `TextOutput` | `text` | `text: str` | reserved |
| `PoseOutput` | `pose` | `keypoints: list[Keypoint]`, `skeleton: list[tuple[int,int]]` | reserved |
| `DepthMapOutput` | `depth_map` | `depth: IORef`, `units: Literal["relative","meters"]`, `min_depth: float`, `max_depth: float` | DA3 (monocular) |
| `CameraParametersOutput` | `camera_params` | `intrinsics: list[list[float]]`, `extrinsics: list[list[float]]` | DA3 (multi-view) |
| `PointCloudOutput` | `pointcloud` | `pointcloud: IORef`, `n_points: int`, `point_format: PointFormat` | DA3 (multi-view, optional) |
| `MultiViewDepthOutput` | `multiview_depth` | `per_view: list[DepthMapOutput]`, `cameras: CameraParametersOutput`, `pointcloud: PointCloudOutput?` | DA3 (multi-view) |

### Output detail

```python
# packages/io/outputs/mask.py

class MaskLabelOutput(OutputBase):
    output_type = "mask_label"
    mask: IORef                          # PNG, single-channel
    label: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    bbox: BBox2D | None = None

# packages/io/outputs/depth.py

class DepthMapOutput(OutputBase):
    output_type = "depth_map"
    depth: IORef                         # PNG-16 or EXR; sidecar JSON for scale
    units: Literal["relative", "meters"]
    min_depth: float
    max_depth: float

# packages/io/outputs/composite.py

class MultiViewDepthOutput(OutputBase):
    output_type = "multiview_depth"
    per_view: list[DepthMapOutput]
    cameras: CameraParametersOutput
    pointcloud: PointCloudOutput | None = None
```

## How adapters declare I/O

Replaces the loose `Capability(inputs=[Modality], outputs=[Modality])` from [04-model-and-tasks.md](./04-model-and-tasks.md).

```python
# packages/models/base.py (revised)

class TypedCapability(BaseModel):
    task: TaskType
    input_class:  type[InputBase]
    output_class: type[OutputBase]        # for "list of X" outputs use list[X]; see note

class ModelCapabilities(BaseModel):
    model_id: str
    capabilities: list[TypedCapability]
    min_gpu_classes: list[GpuClass]
    load_gpu_mem_mb: int
    per_request_gpu_mem_mb: int
    max_input_pixels: int
    supports_fp16: bool
    supports_batching: bool
    max_batch_size: int = 1
    cold_start_seconds_p95: float
```

`output_class` accepts `type[OutputBase]` for single-output tasks and `list[T]` (via `typing.get_origin`) for fan-out tasks like SAM3 (one mask per matched object). The registry normalizes both.

### SAM3 declaration

```python
# packages/models/sam3/adapter.py

@register_model
class Sam3Adapter:
    caps = ModelCapabilities(
        model_id="sam3",
        capabilities=[
            TypedCapability(TaskType.SEGMENTATION_TEXT,
                            input_class=ImageTextInput,
                            output_class=list[MaskLabelOutput]),
            TypedCapability(TaskType.SEGMENTATION_POINT,
                            input_class=ImagePointInput,
                            output_class=list[MaskLabelOutput]),
            TypedCapability(TaskType.SEGMENTATION_BOX,
                            input_class=ImageBoxInput,
                            output_class=list[MaskLabelOutput]),
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

### Depth Anything 3 declaration

```python
# packages/models/depth_anything_v3/adapter.py

@register_model
class DepthAnythingV3Adapter:
    caps = ModelCapabilities(
        model_id="depth_anything_v3",
        capabilities=[
            TypedCapability(TaskType.DEPTH_MONOCULAR,
                            input_class=ImageInput,
                            output_class=DepthMapOutput),
            TypedCapability(TaskType.DEPTH_MULTIVIEW,
                            input_class=MultiViewImageInput,
                            output_class=MultiViewDepthOutput),
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

## Registry filter — "no overshooting"

```python
# packages/io/registry.py

class IORegistry:
    @classmethod
    def visible_inputs(cls, loaded: list[ModelCapabilities]) -> set[type[InputBase]]:
        return {cap.input_class for m in loaded for cap in m.capabilities}

    @classmethod
    def visible_outputs(cls, loaded: list[ModelCapabilities]) -> set[type[OutputBase]]:
        out: set[type[OutputBase]] = set()
        for m in loaded:
            for cap in m.capabilities:
                t = cap.output_class
                origin = typing.get_origin(t)
                out.add(typing.get_args(t)[0] if origin is list else t)
        return out
```

The API only emits OpenAPI schemas for the union of `visible_inputs ∪ visible_outputs`. SDKs generated downstream are correspondingly tight — a Python client built against a server with only SAM3 loaded has no `MultiViewImageInput` symbol at all.

`GET /v1/io/types` returns the live, filtered catalogue:

```json
{
  "inputs":  [
    { "name": "ImageInput",         "input_type": "image",          "schema": { ... } },
    { "name": "ImageTextInput",     "input_type": "image_text",     "schema": { ... } },
    { "name": "ImagePointInput",    "input_type": "image_point",    "schema": { ... } },
    { "name": "ImageBoxInput",      "input_type": "image_box",      "schema": { ... } },
    { "name": "MultiViewImageInput","input_type": "multiview_image","schema": { ... } }
  ],
  "outputs": [
    { "name": "MaskLabelOutput",      "output_type": "mask_label",      "schema": { ... } },
    { "name": "DepthMapOutput",       "output_type": "depth_map",       "schema": { ... } },
    { "name": "MultiViewDepthOutput", "output_type": "multiview_depth", "schema": { ... } }
  ]
}
```

`schema` is the JSON Schema fragment (Pydantic `.model_json_schema()`).

## Validator hook

Each input class implements `validate_with_caps`. The TaskSpec calls it after Pydantic parsing, before enqueue:

```python
# packages/io/inputs/image.py
class ImageInput(InputBase):
    image: IORef
    def validate_with_caps(self, caps: ModelCapabilities) -> None:
        if self.image.byte_length and self.image.byte_length > MAX_UPLOAD_BYTES:
            raise PayloadTooLarge()
        # pixel cap is enforced at worker after MIME sniff; can't trust client dims
```

```python
# packages/io/inputs/multiview.py
class MultiViewImageInput(InputBase):
    views: list[ImageView] = Field(min_length=2, max_length=16)
    def validate_with_caps(self, caps: ModelCapabilities) -> None:
        if len(self.views) * caps.per_request_gpu_mem_mb > caps.load_gpu_mem_mb * 2:
            raise PayloadTooLarge(detail="too many views for this model's budget")
```

## Artifact serialization

Every `OutputBase` knows how to lay itself out under `artifacts/{tenant}/{job_id}/`:

```python
# packages/io/outputs/depth.py
class DepthMapOutput(OutputBase):
    def serialize_artifacts(self) -> list[ArtifactSpec]:
        return [
            ArtifactSpec(name="depth.png",      ref=self.depth, content_type="image/png"),
            ArtifactSpec(name="depth_meta.json", ref=self._meta_ref(), content_type="application/json"),
        ]
```

The worker's `artifacts.upload(...)` consumes `ArtifactSpec` lists; no per-task code needed.

## Adding a new I/O class

1. Add the Pydantic model under `packages/io/inputs/` or `outputs/`.
2. Re-export from `packages/io/__init__.py`.
3. Reference it from at least one adapter's `TypedCapability`. The class is invisible to the API until then — that is the "no overshooting" guarantee.
4. (For inputs) implement `validate_with_caps`.
5. (For outputs) implement `serialize_artifacts`.

No router changes. No DB migration (request payloads are JSONB).

## Mapping from old `Modality` enum

The string enum is removed. Existing references migrate as follows:

| Old `Modality` | New shape |
|---|---|
| `IMAGE` | `ImageInput` (or part of richer input class) |
| `TEXT` | promoted to `TextQuery` field inside `ImageTextInput` |
| `VIDEO` | `VideoInput` (reserved) |
| `POINTCLOUD` | `PointCloudInput` (reserved) |
| `DEPTH` | `DepthMapOutput` |
| `MASK` | `MaskLabelOutput` or `SegmentationMapOutput` |
| `MESH` | `MeshOutput` (reserved; not in v2.0) |

The enum file is deleted in Phase 4. ADR-014 in [11-risks-and-decisions.md](./11-risks-and-decisions.md) records the swap.

## Test surface

- `tests/unit/io/test_visibility.py` — assert that with only SAM3 loaded, the registry hides `DepthMapOutput`; with both, it does not.
- `tests/unit/io/test_validators.py` — `validate_with_caps` enforces budget rules.
- `tests/unit/io/test_artifacts.py` — round-trip every output → `ArtifactSpec` list → fake uploader.
- `tests/contract/test_openapi_io_types.py` — `GET /v1/io/types` schema matches Pydantic's emitted JSON Schema exactly.
