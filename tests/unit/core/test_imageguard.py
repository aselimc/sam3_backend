"""100% coverage on imageguard.py."""

import importlib

from PIL import Image

import packages.core.imageguard as guard
from packages.core.config import get_settings


def test_max_image_pixels_set_to_settings():
    expected = get_settings().max_image_pixels
    assert Image.MAX_IMAGE_PIXELS == expected


def test_reimport_reapplies_cap(monkeypatch):
    Image.MAX_IMAGE_PIXELS = 1
    importlib.reload(guard)
    assert Image.MAX_IMAGE_PIXELS == get_settings().max_image_pixels
