import io
import json

from packages.core import logging as log


def test_configure_emits_json_with_ctx(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr("sys.stderr", buf)
    log.configure(level="DEBUG")

    with log.bind(request_id="req-1", job_id="job-1"):
        log.logger.bind(model_id="sam3").info("inference done", duration_ms=14123)

    line = buf.getvalue().strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["msg"] == "inference done"
    assert rec["level"] == "INFO"
    assert rec["request_id"] == "req-1"
    assert rec["job_id"] == "job-1"
    # logger.bind extras flatten in
    assert rec["model_id"] == "sam3"
    assert rec["duration_ms"] == 14123
    assert rec["ts"].endswith("Z")


def test_bind_resets_after_exit(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr("sys.stderr", buf)
    log.configure(level="DEBUG")

    with log.bind(request_id="outer"):
        with log.bind(request_id="inner"):
            log.logger.info("hello")
        log.logger.info("middle")
    log.logger.info("after")

    lines = [json.loads(l) for l in buf.getvalue().strip().splitlines()]
    assert lines[-3]["request_id"] == "inner"
    assert lines[-2]["request_id"] == "outer"
    assert "request_id" not in lines[-1]
