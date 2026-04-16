"""Async job endpoints — submit inference work and poll for results.

These mirror the synchronous /segment-from-path and /segment-from-upload
endpoints but return immediately with a job_id.
"""

import asyncio
import io
import os
import traceback

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from PIL import Image

from app.job_schemas import JobStatusResponse, JobSubmitResponse
from app.jobs import JobStatus, job_store
from app.schemas import PredictRequest

router = APIRouter(prefix="/jobs", tags=["jobs"])


async def _run_inference_task(
    request: Request,
    job_id: str,
    coro_factory,
) -> None:
    """Background wrapper that acquires the semaphore, runs inference, and updates the job."""
    semaphore = request.app.state.inference_semaphore
    try:
        async with semaphore:
            await job_store.update(job_id, status=JobStatus.RUNNING)
            result = await coro_factory()
            await job_store.update(job_id, status=JobStatus.COMPLETED, result=result)
    except Exception as exc:
        logger.exception("job {} failed", job_id)
        await job_store.update(
            job_id, status=JobStatus.FAILED, error=traceback.format_exception_only(exc)[-1].strip()
        )


@router.post("/segment-from-path", response_model=JobSubmitResponse)
async def submit_predict(request: Request, body: PredictRequest):
    if not os.path.isfile(body.image_path):
        raise HTTPException(status_code=400, detail=f"Image not found: {body.image_path}")

    service = request.app.state.sam3_service
    job = await job_store.create()

    async def factory():
        return await run_in_threadpool(
            service.predict,
            image_path=body.image_path,
            queries=body.queries,
            output_dir=body.output_dir,
            confidence_threshold=body.confidence_threshold,
            regularize=body.regularize,
        )

    asyncio.get_event_loop().create_task(_run_inference_task(request, job.id, factory))
    logger.info("job submitted job_id={} endpoint=segment-from-path", job.id)
    return JobSubmitResponse(job_id=job.id, status=job.status)


@router.post("/segment-from-upload", response_model=JobSubmitResponse)
async def submit_predict_upload(
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
    job = await job_store.create()

    async def factory():
        return await run_in_threadpool(
            service.predict_b64,
            image=pil_image,
            queries=queries,
            confidence_threshold=confidence_threshold,
            regularize=regularize,
        )

    asyncio.get_event_loop().create_task(_run_inference_task(request, job.id, factory))
    logger.info("job submitted job_id={} endpoint=segment-from-upload", job.id)
    return JobSubmitResponse(job_id=job.id, status=job.status)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = await job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        result=job.result,
        error=job.error,
    )
