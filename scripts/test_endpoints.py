"""Smoke-test the SAM3 backend endpoints against local assets/.

Usage:
    uv run python main.py          # in one shell
    uv run python scripts/test_endpoints.py   # in another

Tests:
    1. GET  /metrics              — Prometheus exposition
    2. POST /segment-from-path    — path-based, writes masks to disk
    3. POST /segment-from-upload  — multipart upload, returns base64 masks
    4. Concurrency: fire 3 parallel /segment-from-upload and confirm no 5xx
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUTPUT_DIR = ROOT / "data" / "test_outputs"

IMAGES = {
    "bee": ASSETS / "bee.jpg",
    "scene1": ASSETS / "scene1.jpg",
    "scene2": ASSETS / "scene2.jpg",
}
QUERIES = {
    "bee": ["bee", "flower"],
    "scene1": ["building", "tree"],
    "scene2": ["car", "road"],
}


def _http_get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.read()


def _http_post_json(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _http_post_multipart(
    url: str, image_path: Path, queries: list[str]
) -> tuple[int, dict]:
    boundary = "----sam3test" + str(int(time.time() * 1000))
    lines: list[bytes] = []
    for q in queries:
        lines += [
            f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="queries"',
            b"",
            q.encode(),
        ]
    lines += [
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="confidence_threshold"',
        b"",
        b"0.4",
    ]
    lines += [
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"'.encode(),
        b"Content-Type: image/jpeg",
        b"",
        image_path.read_bytes(),
    ]
    lines += [f"--{boundary}--".encode(), b""]
    body = b"\r\n".join(lines)
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_metrics(base: str) -> None:
    print("[1/4] GET /metrics")
    status, body = _http_get(f"{base}/metrics")
    assert status == 200, status
    text = body.decode(errors="replace")
    assert "sam3_inference_seconds" in text or "http_request" in text
    print("     ok (Prometheus text, %d bytes)" % len(body))


def test_predict(base: str) -> None:
    print("[2/4] POST /segment-from-path (path-based)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": str(IMAGES["scene1"]),
        "queries": QUERIES["scene1"],
        "output_dir": str(OUTPUT_DIR),
        "confidence_threshold": 0.4,
        "regularize": [False, True],
    }
    t0 = time.perf_counter()
    status, body = _http_post_json(f"{base}/segment-from-path", payload)
    dur = time.perf_counter() - t0
    assert status == 200, (status, body)
    n_objs = sum(len(r["objects"]) for r in body["results"])
    print(f"     ok ({dur:.2f}s, {len(body['results'])} queries, {n_objs} objects)")


def test_predict_upload(base: str) -> None:
    print("[3/4] POST /segment-from-upload (multipart)")
    t0 = time.perf_counter()
    status, body = _http_post_multipart(
        f"{base}/segment-from-upload", IMAGES["bee"], QUERIES["bee"]
    )
    dur = time.perf_counter() - t0
    assert status == 200, (status, body)
    n_objs = sum(len(r["objects"]) for r in body["results"])
    for r in body["results"]:
        for o in r["objects"]:
            assert isinstance(o["mask_b64"], str)
            base64.b64decode(o["mask_b64"])  # decodable
    print(f"     ok ({dur:.2f}s, {len(body['results'])} queries, {n_objs} objects)")


def test_concurrency(base: str) -> None:
    print("[4/4] 3x parallel /segment-from-upload (semaphore should serialize)")
    jobs = [
        (IMAGES["bee"], QUERIES["bee"]),
        (IMAGES["scene1"], QUERIES["scene1"]),
        (IMAGES["scene2"], QUERIES["scene2"]),
    ]
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [
            pool.submit(_http_post_multipart, f"{base}/segment-from-upload", img, q)
            for img, q in jobs
        ]
        results = [f.result() for f in futures]
    dur = time.perf_counter() - t0
    for status, body in results:
        assert status == 200, (status, body)
    print(f"     ok ({dur:.2f}s total wall-clock for {len(jobs)} requests)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8000")
    args = p.parse_args()

    for name, path in IMAGES.items():
        if not path.is_file():
            print(f"missing asset: {path}", file=sys.stderr)
            return 2

    try:
        test_metrics(args.base)
        test_predict(args.base)
        test_predict_upload(args.base)
        test_concurrency(args.base)
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"cannot reach server at {args.base}: {e}", file=sys.stderr)
        return 2

    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
