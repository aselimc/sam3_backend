"""Pydantic schemas for the async job endpoints."""

from pydantic import BaseModel

from app.jobs import JobStatus


class JobSubmitResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: dict | list | None = None
    error: str | None = None
