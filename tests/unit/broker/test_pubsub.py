"""publish_event reaches both per-job + global channels; subscribe parses JSON."""

from __future__ import annotations

import threading
import time

from packages.broker.pubsub import GLOBAL_CHANNEL, channel_for, publish_event, subscribe


def test_channel_naming() -> None:
    assert channel_for("abc") == "pubsub:job.abc"
    assert GLOBAL_CHANNEL == "pubsub:job.events"


def test_publish_round_trip_per_job(redis) -> None:
    received: list[dict] = []
    started = threading.Event()

    def consume() -> None:
        gen = subscribe(redis, job_id="j1", timeout=0.05)
        started.set()
        for ev in gen:
            received.append(ev)
            if len(received) >= 1:
                gen.close()
                return

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    started.wait(timeout=1.0)
    time.sleep(0.05)  # let subscribe register

    publish_event(redis, job_id="j1", state="RUNNING", attempt=1)
    t.join(timeout=2.0)

    assert received and received[0]["job_id"] == "j1"
    assert received[0]["state"] == "RUNNING"
    assert received[0]["attempt"] == 1
    assert "ts" in received[0]


def test_global_channel_sees_all(redis) -> None:
    received: list[dict] = []
    started = threading.Event()

    def consume() -> None:
        gen = subscribe(redis, timeout=0.05)
        started.set()
        for ev in gen:
            received.append(ev)
            if len(received) >= 2:
                gen.close()
                return

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    started.wait(timeout=1.0)
    time.sleep(0.05)

    publish_event(redis, "j1", "RUNNING")
    publish_event(redis, "j2", "SUCCEEDED")
    t.join(timeout=2.0)

    states = {ev["state"] for ev in received}
    assert {"RUNNING", "SUCCEEDED"} <= states
