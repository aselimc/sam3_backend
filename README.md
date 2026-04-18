# SAM3 Backend

[![CI](https://github.com/aselimc/sam3_backend/actions/workflows/ci.yml/badge.svg)](https://github.com/aselimc/sam3_backend/actions/workflows/ci.yml)
[![Docs](https://github.com/aselimc/sam3_backend/actions/workflows/docs.yml/badge.svg)](https://github.com/aselimc/sam3_backend/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/aselimc/sam3_backend/branch/master/graph/badge.svg)](https://codecov.io/gh/aselimc/sam3_backend)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/aselimc/sam3_backend)](https://github.com/aselimc/sam3_backend/blob/master/LICENSE)

FastAPI service for text-prompted segmentation using SAM3.

📚 **Documentation:** https://aselimc.github.io/sam3_backend/

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- CUDA-capable GPU

## Install

```bash
uv sync
```

## Run

```bash
uv run python main.py
```

Server: http://localhost:8000

## How model weights are loaded

- Before first run, authenticate with Hugging Face:

```bash
hf auth login
```

- The app creates `SAM3Service`, which calls `build_sam3_image_model()`.
- No local checkpoint is passed.
- SAM3 then downloads weights from Hugging Face (`facebook/sam3`, file `sam3.pt`) using `hf_hub_download`.
- Downloaded files are reused from the local Hugging Face cache on later runs.

## Test

```bash
uv run pytest tests/ -v
```

With coverage report:

```bash
uv run pytest tests/ -v --cov --cov-report=term
```

## API

- `POST /segment-from-path`: image path input, saves masks to disk.
- `POST /segment-from-upload`: image upload input, returns masks as base64.

## Docs

API reference is auto-generated with Sphinx and published to GitHub Pages on every push to `master` via `.github/workflows/docs.yml`.

Build locally:

```bash
uv run --group docs sphinx-build -b html docs/source docs/build/html
```

Then open `docs/build/html/index.html`.
