"""Tests for async job endpoints."""

import asyncio
import io

import pytest
from PIL import Image


class TestJobSegmentFromPath:
    @pytest.mark.asyncio
    async def test_submit_missing_image(self, client):
        resp = await client.post(
            "/jobs/segment-from-path",
            json={
                "image_path": "/nonexistent/image.jpg",
                "queries": ["building"],
                "output_dir": "/tmp/out",
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_and_poll(self, client, tmp_path):
        img_path = str(tmp_path / "test.jpg")
        Image.new("RGB", (64, 64)).save(img_path)

        resp = await client.post(
            "/jobs/segment-from-path",
            json={
                "image_path": img_path,
                "queries": ["building"],
                "output_dir": str(tmp_path / "out"),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

        # Give the background task time to complete
        await asyncio.sleep(0.1)

        poll = await client.get(f"/jobs/{data['job_id']}")
        assert poll.status_code == 200
        poll_data = poll.json()
        assert poll_data["status"] in ("completed", "running")


class TestJobSegmentFromUpload:
    @pytest.mark.asyncio
    async def test_submit_and_poll(self, client):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, format="PNG")
        buf.seek(0)

        resp = await client.post(
            "/jobs/segment-from-upload",
            files={"image": ("test.png", buf, "image/png")},
            data={"queries": ["building"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data

        await asyncio.sleep(0.1)

        poll = await client.get(f"/jobs/{data['job_id']}")
        assert poll.status_code == 200


class TestJobNotFound:
    @pytest.mark.asyncio
    async def test_missing_job(self, client):
        resp = await client.get("/jobs/nonexistent-id")
        assert resp.status_code == 404
