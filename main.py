import argparse
import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.router import router
from app.sam3_service import SAM3Service


DEFAULT_LOG_PATH = Path("logs") / "sam3_backend.log"


def _configure_logging(enable_file_logging: bool = False) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        serialize=False,
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )
    if enable_file_logging:
        DEFAULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(DEFAULT_LOG_PATH),
            level=settings.log_level,
            serialize=False,
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    enable_file_logging = getattr(app.state, "enable_file_logging", False)
    _configure_logging(enable_file_logging=enable_file_logging)
    logger.info(
        "starting sam3-backend (max_concurrent={}, metrics={})",
        settings.max_concurrent_inferences,
        settings.enable_metrics,
    )
    if enable_file_logging:
        logger.info("file logging enabled at {}", DEFAULT_LOG_PATH)
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
    parser = argparse.ArgumentParser(description="Run SAM3 backend")
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Also write logs to logs/sam3_backend.log",
    )
    args = parser.parse_args()

    app.state.enable_file_logging = args.logs
    uvicorn.run(app, host="0.0.0.0", port=8000)
