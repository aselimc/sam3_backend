# SAM3 Backend

FastAPI service for text-prompted segmentation using SAM3.

📚 **Documentation:** https://aselimc.github.io/sam3_backend/

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- CUDA-capable GPU (recommended)

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
