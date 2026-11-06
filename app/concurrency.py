from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import HTTPException

from app.schemas import ConcurrencySnapshot


class ConcurrencyLimiter:
    def __init__(self, max_concurrency: int, max_queue_size: int, queue_timeout: float) -> None:
        self._max = max_concurrency
        self._max_queue = max_queue_size
        self._timeout = queue_timeout
        self._sem = asyncio.Semaphore(max_concurrency)
        self._queued = 0
        self._active = 0

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        if not self._sem.locked():
            await self._sem.acquire()
        else:
            if self._max_queue and self._queued >= self._max_queue:
                raise HTTPException(503, "engine busy: queue full")
            self._queued += 1
            try:
                if self._timeout:
                    await asyncio.wait_for(self._sem.acquire(), self._timeout)
                else:
                    await self._sem.acquire()
            except asyncio.TimeoutError as exc:
                raise HTTPException(503, "engine busy: queue timeout") from exc
            finally:
                self._queued -= 1
        self._active += 1
        try:
            yield
        finally:
            self._active -= 1
            self._sem.release()

    def snapshot(self) -> ConcurrencySnapshot:
        return ConcurrencySnapshot(max=self._max, active=self._active, queued=self._queued)
