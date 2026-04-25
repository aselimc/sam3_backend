"""Shared kernel. Importing this package eagerly sets the PIL pixel cap."""

from . import imageguard as _imageguard  # noqa: F401  (side effect: sets MAX_IMAGE_PIXELS)
