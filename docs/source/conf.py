"""Sphinx configuration for sam3-backend documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "sam3-backend"
author = "sam3-backend contributors"
copyright = "2026, sam3-backend contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "sam3",
    "cv2",
    "pycocotools",
    "triton",
    "einops",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
}
