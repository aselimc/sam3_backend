"""Webhook deliveries repository.

The dispatcher (Phase 5) reads `due()` to find rows whose `next_retry_at
<= now() AND delivered_at IS NULL`. On success it stamps `delivered_at`;
on failure it records the status_code and reschedules with backoff
(1, 5, 25, 125 s — see upgrade/03 §Webhooks).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from packages.db.models import LOCAL_OWNER, Job, WebhookDelivery


def _utcnow() -> datetime:
    return datetime.now(UTC)


class WebhookRepo:
    BACKOFF_SCHEDULE_S = (1, 5, 25, 125)
    MAX_ATTEMPTS = 5

    def __init__(self, session: AsyncSession, *, owner_id: str = LOCAL_OWNER) -> None:
        self._s = session
        self._owner = owner_id

    async def enqueue(self, *, job_id: uuid.UUID, url: str) -> WebhookDelivery:
        d = WebhookDelivery(
            job_id=job_id,
            url=url,
            attempt=0,
            next_retry_at=_utcnow(),
        )
        self._s.add(d)
        await self._s.flush()
        return d

    async def due(self, *, limit: int = 100) -> list[WebhookDelivery]:
        now = _utcnow()
        stmt = (
            select(WebhookDelivery)
            .join(Job, Job.id == WebhookDelivery.job_id)
            .where(
                and_(
                    Job.owner_id == self._owner,
                    WebhookDelivery.delivered_at.is_(None),
                    WebhookDelivery.next_retry_at <= now,
                )
            )
            .order_by(WebhookDelivery.next_retry_at.asc())
            .limit(limit)
        )
        return list((await self._s.execute(stmt)).scalars().all())

    async def mark_delivered(self, delivery_id: uuid.UUID, *, status_code: int) -> None:
        await self._s.execute(
            update(WebhookDelivery)
            .where(WebhookDelivery.id == delivery_id)
            .values(
                status_code=status_code,
                delivered_at=_utcnow(),
                next_retry_at=None,
            )
        )

    async def schedule_retry(
        self,
        delivery_id: uuid.UUID,
        *,
        status_code: int | None,
        response_body: str | None,
    ) -> None:
        d = await self._s.get(WebhookDelivery, delivery_id)
        if d is None:
            return
        attempt = d.attempt + 1
        if attempt > self.MAX_ATTEMPTS:
            d.attempt = attempt
            d.status_code = status_code
            d.response_body = response_body
            d.next_retry_at = None  # dead-lettered
            await self._s.flush()
            return
        delay = self.BACKOFF_SCHEDULE_S[min(attempt - 1, len(self.BACKOFF_SCHEDULE_S) - 1)]
        d.attempt = attempt
        d.status_code = status_code
        d.response_body = response_body
        d.next_retry_at = _utcnow() + timedelta(seconds=delay)
        await self._s.flush()
