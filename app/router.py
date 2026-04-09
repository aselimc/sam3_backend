import io
import os

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from app.schemas import PredictRequest, PredictResponse, PredictUploadResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest):
    if not os.path.isfile(body.image_path):
        raise HTTPException(status_code=400, detail=f"Image not found: {body.image_path}")

    service = request.app.state.sam3_service
    results = service.predict(
        image_path=body.image_path,
        queries=body.queries,
        output_dir=body.output_dir,
        confidence_threshold=body.confidence_threshold,
    )

    return PredictResponse(image_path=body.image_path, results=results)


@router.post("/predict/upload", response_model=PredictUploadResponse)
async def predict_upload(
    request: Request,
    image: UploadFile = File(...),
    queries: list[str] = Form(...),
    confidence_threshold: float = Form(0.5),
):
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    service = request.app.state.sam3_service
    results = service.predict_b64(
        image=pil_image,
        queries=queries,
        confidence_threshold=confidence_threshold,
    )

    return PredictUploadResponse(filename=image.filename or "upload", results=results)
