"""Cooperative cancellation primitive.

The worker SIGUSR1 handler trips the active CancelCheck. Long loops in
adapters / task specs sprinkle calls to the check; the next call after
trip raises CancelRequested, which the runner catches to transition
RUNNING → CANCELING → CANCELED.

See upgrade/05-worker-runtime.md §Cancellation.
"""

from __future__ import annotations

from .errors import CancelRequested

__all__ = ["CancelCheck", "CancelRequested"]


class CancelCheck:
    """A trippable flag; when tripped, the next call raises CancelRequested."""

    __slots__ = ("_flag",)

    def __init__(self) -> None:
        self._flag = False

    def trip(self) -> None:
        self._flag = True

    @property
    def tripped(self) -> bool:
        return self._flag

    def __call__(self) -> None:
        if self._flag:
            raise CancelRequested()
