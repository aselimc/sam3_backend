"""Set PIL.Image.MAX_IMAGE_PIXELS at import.

Some libraries silently raise this limit on their own import; if PIL has
already decoded a frame before we set the cap, the cap is moot. So this
module is imported eagerly from packages/core/__init__.py on every
process boot — well before any image bytes are seen.

See upgrade/00-evaluation.md for the prior incident.
"""

from __future__ import annotations

from PIL import Image

from .config import get_settings

# Snap to the configured value. Settings has a sane default; an operator
# can lower it via env without touching code.
Image.MAX_IMAGE_PIXELS = get_settings().max_image_pixels
