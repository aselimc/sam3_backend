"""Core enums shared by API, broker, worker, db, and SDK clients.

These three enums are the closed set at v2.0. Extending requires migration
+ adapter changes; see upgrade/02-data-model.md and upgrade/04-model-and-tasks.md.
"""

from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    SEGMENTATION_TEXT = "segmentation.text"
    SEGMENTATION_POINT = "segmentation.point"
    SEGMENTATION_BOX = "segmentation.box"
    DEPTH_MONOCULAR = "depth.monocular"
    DEPTH_MULTIVIEW = "depth.multiview"


class JobState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELING = "CANCELING"
    CANCELED = "CANCELED"


TERMINAL_STATES: frozenset[JobState] = frozenset(
    {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED}
)


class GpuClass(StrEnum):
    CPU = "cpu"
    T4_16G = "t4_16g"
    L4_24G = "l4_24g"
    A10_24G = "a10_24g"
    A100_40G = "a100_40g"
    A100_80G = "a100_80g"
    H100_80G = "h100_80g"
