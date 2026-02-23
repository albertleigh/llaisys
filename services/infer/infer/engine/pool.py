"""Thread-safe request pool with async notification."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque

from .request import InferRequest, RequestStatus

logger = logging.getLogger(__name__)


class RequestPool:
    """A FIFO pool of pending inference requests.

    * **Thread-safe** — ``add()`` / ``pop_batch()`` / ``return_requests()``
      may be called from any thread.
    * **Async-aware** — ``wait_for_requests()`` is an ``async`` method that
      blocks until the pool is non-empty, so the inference loop can sleep
      without busy-waiting.

    Parameters
    ----------
    max_size:
        Maximum number of requests allowed in the pool.  ``add()`` returns
        *False* when the limit is reached.
    """

    def __init__(self, max_size: int = 128) -> None:
        self._max_size = max_size
        self._queue: deque[InferRequest] = deque()
        self._lock = threading.Lock()

        # Async event — signalled whenever the queue becomes non-empty.
        # Created lazily so the pool can be instantiated before an event
        # loop is running.
        self._event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Counters
        self._active_count = 0

    # ── Async event helpers ──────────────────────────────────────────

    def _ensure_event(self) -> asyncio.Event:
        """Lazily create the asyncio.Event in the current running loop."""
        loop = asyncio.get_running_loop()
        if self._event is None or self._loop is not loop:
            self._event = asyncio.Event()
            self._loop = loop
            if self._queue:
                self._event.set()
        return self._event

    def _signal(self) -> None:
        """Set the async event, safe to call from any thread."""
        if self._event is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._event.set)

    def _clear_if_empty(self) -> None:
        """Clear the event if the queue is empty (must hold _lock)."""
        if not self._queue and self._event is not None:
            # Safe because we hold the lock and only clear when truly empty
            try:
                self._loop.call_soon_threadsafe(self._event.clear)  # type: ignore[union-attr]
            except RuntimeError:
                pass  # loop closed

    # ── Public API ───────────────────────────────────────────────────

    @property
    def pending_count(self) -> int:
        """Number of requests waiting in the pool."""
        with self._lock:
            return len(self._queue)

    @property
    def active_count(self) -> int:
        """Number of requests currently being processed."""
        return self._active_count

    def add(self, request: InferRequest) -> bool:
        """Enqueue a request.  Returns *False* if the pool is full."""
        with self._lock:
            if len(self._queue) >= self._max_size:
                logger.warning("Request pool full (%d); rejecting %s", self._max_size, request.id)
                return False
            request.status = RequestStatus.QUEUED
            self._queue.append(request)
            logger.info("Request %s added to pool (pending=%d)", request.id, len(self._queue))
        self._signal()
        return True

    def pop_batch(self, max_batch_size: int = 1) -> list[InferRequest]:
        """Pop up to *max_batch_size* non-cancelled requests.

        Cancelled requests are silently discarded.  Returns an empty list
        when no eligible requests are available.
        """
        batch: list[InferRequest] = []
        with self._lock:
            while self._queue and len(batch) < max_batch_size:
                req = self._queue.popleft()
                if req.cancelled:
                    req.status = RequestStatus.CANCELLED
                    logger.debug("Skipping cancelled request %s", req.id)
                    continue
                req.status = RequestStatus.ACTIVE
                batch.append(req)

            self._active_count += len(batch)
            self._clear_if_empty()
        return batch

    def return_requests(self, requests: list[InferRequest]) -> None:
        """Return unfinished requests to the **front** of the pool.

        This is the "continuous batching" path: after one iteration step,
        requests that still have tokens to generate are returned so they
        can be re-scheduled in the next batch.
        """
        with self._lock:
            for req in reversed(requests):
                if req.cancelled:
                    req.status = RequestStatus.CANCELLED
                    self._active_count -= 1
                    continue
                req.status = RequestStatus.QUEUED
                self._queue.appendleft(req)
                self._active_count -= 1
        self._signal()

    def mark_completed(self, request: InferRequest) -> None:
        """Mark a request as done and decrement the active counter."""
        request.status = RequestStatus.COMPLETED
        with self._lock:
            self._active_count -= 1

    async def wait_for_requests(self) -> None:
        """Async-wait until the pool is non-empty."""
        event = self._ensure_event()
        await event.wait()

    # ── Diagnostics ──────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a snapshot of pool statistics."""
        with self._lock:
            return {
                "pending": len(self._queue),
                "active": self._active_count,
                "max_size": self._max_size,
            }
