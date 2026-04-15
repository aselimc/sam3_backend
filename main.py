import asyncio
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.router import router
from app.sam3_service import SAM3Service


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        serialize=False,
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    logger.info(
        "starting sam3-backend (max_concurrent={}, metrics={})",
        settings.max_concurrent_inferences,
        settings.enable_metrics,
    )
    app.state.sam3_service = SAM3Service()
    app.state.inference_semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)
    logger.info("sam3 model ready")
    yield
    logger.info("shutting down sam3-backend")


app = FastAPI(title="SAM3.1 Backend", lifespan=lifespan)
app.include_router(router)

if settings.enable_metrics:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
