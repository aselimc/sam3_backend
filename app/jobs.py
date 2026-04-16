"""In-memory async job queue for long-running SAM3 inference.

Clients POST to the /jobs/* endpoints to submit work and receive a job_id.
They then poll GET /jobs/{job_id} until the status is "completed" or "failed".
The inference semaphore is respected so GPU concurrency stays bounded.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: str | None = None


class JobStore:
    """Thread-safe, in-memory job store."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex)
        async with self._lock:
            self._jobs[job.id] = job
        return job

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job_id: str, **kwargs: Any) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)


# Module-level singleton — attached to app.state in main.py
job_store = JobStore()
