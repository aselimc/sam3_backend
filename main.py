from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.router import router
from app.sam3_service import SAM3Service


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.sam3_service = SAM3Service()
    yield


app = FastAPI(title="SAM3.1 Backend", lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
