"""Tests for API endpoints."""

import io

import pytest
from PIL import Image


class TestSegmentFromPath:
    @pytest.mark.asyncio
    async def test_missing_image(self, client):
        resp = await client.post(
            "/segment-from-path",
            json={
                "image_path": "/nonexistent/image.jpg",
                "queries": ["building"],
                "output_dir": "/tmp/out",
            },
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_successful_predict(self, client, tmp_path):
        img_path = str(tmp_path / "test.jpg")
        Image.new("RGB", (64, 64)).save(img_path)

        resp = await client.post(
            "/segment-from-path",
            json={
                "image_path": img_path,
                "queries": ["building"],
                "output_dir": str(tmp_path / "out"),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["image_path"] == img_path
        assert len(data["results"]) == 1
        assert data["results"][0]["query"] == "building"


class TestSegmentFromUpload:
    @pytest.mark.asyncio
    async def test_invalid_image(self, client):
        resp = await client.post(
            "/segment-from-upload",
            files={"image": ("bad.jpg", b"not-an-image", "image/jpeg")},
            data={"queries": ["building"]},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_successful_upload(self, client):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, format="PNG")
        buf.seek(0)

        resp = await client.post(
            "/segment-from-upload",
            files={"image": ("test.png", buf, "image/png")},
            data={"queries": ["building"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "test.png"
        assert len(data["results"]) == 1

    @pytest.mark.asyncio
    async def test_regularize_length_mismatch(self, client):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, format="PNG")
        buf.seek(0)

        resp = await client.post(
            "/segment-from-upload",
            files={"image": ("test.png", buf, "image/png")},
            data={"queries": ["building"], "regularize": [True, False]},
        )
        assert resp.status_code == 400
