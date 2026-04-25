"""Pub/Sub helpers used by the SSE endpoint.

Channels:

- `pubsub:job.events` — global feed (every job state transition).
- `pubsub:job.{id}` — per-job feed for SSE subscribers.

`publish_event` writes to both so a subscriber can pick its scope. Payload
is JSON: `{"job_id": ..., "state": ..., "ts": ...}` plus any caller extras.
The subscribe iterator yields parsed dicts and skips control messages.

See upgrade/02-data-model.md §Redis layout, upgrade/03-api-spec.md §SSE.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from redis import Redis

GLOBAL_CHANNEL = "pubsub:job.events"


def channel_for(job_id: str) -> str:
    return f"pubsub:job.{job_id}"


def publish_event(
    redis: Redis,
    job_id: str,
    state: str,
    **extra: Any,
) -> int:
    payload = {"job_id": job_id, "state": state, "ts": time.time(), **extra}
    body = json.dumps(payload, default=str)
    n_per = redis.publish(channel_for(job_id), body)
    n_global = redis.publish(GLOBAL_CHANNEL, body)
    return int(n_per) + int(n_global)


def subscribe(
    redis: Redis,
    *,
    job_id: str | None = None,
    timeout: float | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield dict events for a single job (if `job_id`) or the global feed.

    `timeout` is per-message; `None` blocks. Caller closes the generator
    to release the underlying pubsub.
    """
    pubsub = redis.pubsub(ignore_subscribe_messages=True)
    channel = channel_for(job_id) if job_id else GLOBAL_CHANNEL
    pubsub.subscribe(channel)
    try:
        while True:
            msg = pubsub.get_message(timeout=timeout)
            if msg is None:
                continue
            data = msg.get("data")
            if not isinstance(data, (bytes, str)):
                continue
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue
    finally:
        try:
            pubsub.unsubscribe(channel)
            pubsub.close()
        except Exception:
            pass
