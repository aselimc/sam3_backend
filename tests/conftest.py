"""Shared fixtures for the SAM3 backend test suite.

The SAM3 model is heavy and requires a GPU, so we mock the service layer
for API-level tests.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture()
def mock_sam3_service():
    """Return a MagicMock that behaves like SAM3Service."""
    service = MagicMock()

    def _fake_predict(*, image_path, queries, output_dir, confidence_threshold=0.5, regularize=None):
        return [
            {
                "query": q,
                "objects": [
                    {"mask_path": f"{output_dir}/mask_{i}.png", "box": [0, 0, 100, 100], "score": 0.95}
                ],
            }
            for i, q in enumerate(queries)
        ]

    def _fake_predict_b64(*, image, queries, confidence_threshold=0.5, regularize=None):
        return [
            {
                "query": q,
                "objects": [
                    {"mask_b64": "iVBORw0KGgo=", "box": [0, 0, 100, 100], "score": 0.95}
                ],
            }
            for q in queries
        ]

    service.predict = _fake_predict
    service.predict_b64 = _fake_predict_b64
    return service


@pytest.fixture()
async def client(mock_sam3_service):
    """Async test client with mocked SAM3 model."""
    with patch("main.SAM3Service", return_value=mock_sam3_service):
        from main import app

        app.state.sam3_service = mock_sam3_service
        app.state.inference_semaphore = asyncio.Semaphore(1)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
