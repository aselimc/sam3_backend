from pydantic import BaseModel


# --- Path-based endpoint schemas ---


class PredictRequest(BaseModel):
    image_path: str
    queries: list[str]
    output_dir: str
    confidence_threshold: float = 0.5


class SegmentationObject(BaseModel):
    mask_path: str
    box: list[float]  # [x1, y1, x2, y2] in pixels
    score: float


class QueryResult(BaseModel):
    query: str
    objects: list[SegmentationObject]


class PredictResponse(BaseModel):
    image_path: str
    results: list[QueryResult]


# --- Upload-based endpoint schemas ---


class SegmentationObjectB64(BaseModel):
    mask_b64: str  # base64-encoded PNG mask
    box: list[float]  # [x1, y1, x2, y2] in pixels
    score: float


class QueryResultB64(BaseModel):
    query: str
    objects: list[SegmentationObjectB64]


class PredictUploadResponse(BaseModel):
    filename: str
    results: list[QueryResultB64]
