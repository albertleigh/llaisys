"""Batch scheduler — selects requests from the pool to form inference batches."""

from __future__ import annotations

import logging
from .pool import RequestPool
from .request import InferRequest

logger = logging.getLogger(__name__)


class BatchScheduler:
    """Decide *which* and *how many* requests to process together.

    Current strategy
    ----------------
    Simple FIFO — pop up to ``max_batch_size`` requests from the pool.

    Future extensions
    -----------------
    * **Prefix-grouping** — group requests that share a common KV-cache
      prefix so they can reuse a single prefill.
    * **Token-budget** — limit the total number of tokens across all
      requests in a batch to stay within memory / latency targets.
    * **Priority scheduling** — give higher priority to interactive
      (streaming) requests over background / batch jobs.

    Parameters
    ----------
    pool:
        The :class:`RequestPool` to draw requests from.
    max_batch_size:
        Upper bound on the number of requests in a single batch.
        With the current single-KV-cache C backend this defaults to 1.
        Increase once the backend supports per-request KV caches.
    max_tokens_in_batch:
        Soft cap on the *total input-token count* of the batch
        (prompt + generated-so-far).  0 means no limit.
    """

    def __init__(
        self,
        pool: RequestPool,
        max_batch_size: int = 1,
        max_tokens_in_batch: int = 0,
    ) -> None:
        self.pool = pool
        self.max_batch_size = max_batch_size
        self.max_tokens_in_batch = max_tokens_in_batch

    # ── Core scheduling ──────────────────────────────────────────────

    def form_batch(self) -> list[InferRequest]:
        """Pop a batch from the pool using the current scheduling policy.

        Returns an empty list when no requests are available.
        """
        batch = self.pool.pop_batch(self.max_batch_size)
        if batch:
            logger.info(
                "Formed batch of %d request(s): [%s]",
                len(batch),
                ", ".join(r.id for r in batch),
            )
        return batch

    async def wait_and_form_batch(self) -> list[InferRequest]:
        """Block until at least one request is available, then form a batch.

        Handles spurious wake-ups by retrying if ``form_batch()`` returns
        an empty list (e.g. all popped requests were cancelled).
        """
        while True:
            await self.pool.wait_for_requests()
            batch = self.form_batch()
            if batch:
                return batch
            # All candidates were cancelled — loop and wait again
