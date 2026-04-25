"""Versioned Pydantic base for every public-shape model.

Every external request/response/I/O class subclasses VersionedModel and
locks its `version` to a Literal. Adding a field is additive; changing
semantics requires a new version (see 03-api-spec.md §Versioning).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class VersionedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False, populate_by_name=True)

    version: Literal["1"] = "1"
