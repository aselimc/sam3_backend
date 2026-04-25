"""Repository layer.

Routers and the worker runner do not write SQL directly. Each repository
encapsulates the `WHERE owner_id=:principal.owner_id` filter (the single
isolation rule) and the SQL state guards from upgrade/02-data-model.md.
"""

from .artifacts import ArtifactsRepo
from .jobs import JobsRepo, TransitionConflict
from .webhook import WebhookRepo

__all__ = ["JobsRepo", "TransitionConflict", "ArtifactsRepo", "WebhookRepo"]
