import io
import os

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from PIL import Image

from app.schemas import PredictRequest, PredictResponse, PredictUploadResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest):
    if not os.path.isfile(body.image_path):
        raise HTTPException(status_code=400, detail=f"Image not found: {body.image_path}")

    service = request.app.state.sam3_service
    semaphore = request.app.state.inference_semaphore

    async with semaphore:
        logger.info(
            "predict start path={} n_queries={}", body.image_path, len(body.queries)
        )
        results = await run_in_threadpool(
            service.predict,
            image_path=body.image_path,
            queries=body.queries,
            output_dir=body.output_dir,
            confidence_threshold=body.confidence_threshold,
            regularize=body.regularize,
        )

    return PredictResponse(image_path=body.image_path, results=results)


@router.post("/predict/upload", response_model=PredictUploadResponse)
async def predict_upload(
    request: Request,
    image: UploadFile = File(...),
    queries: list[str] = Form(...),
    confidence_threshold: float = Form(0.5),
    regularize: list[bool] | None = Form(None),
):
    if regularize is not None and len(regularize) != len(queries):
        raise HTTPException(
            status_code=400,
            detail=(
                "regularize must have the same length as queries "
                f"(got {len(regularize)} vs {len(queries)})"
            ),
        )

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    service = request.app.state.sam3_service
    semaphore = request.app.state.inference_semaphore

    async with semaphore:
        logger.info(
            "predict_upload start filename={} size={} n_queries={}",
            image.filename,
            pil_image.size,
            len(queries),
        )
        results = await run_in_threadpool(
            service.predict_b64,
            image=pil_image,
            queries=queries,
            confidence_threshold=confidence_threshold,
            regularize=regularize,
        )

    return PredictUploadResponse(filename=image.filename or "upload", results=results)
