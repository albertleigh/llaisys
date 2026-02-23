"""KV-cache pool with prefix matching for multi-request serving.

Each active request binds to a *slot* in the pool.  A slot holds a
saved KV-cache snapshot (an opaque handle backed by CPU memory) that
can be swapped in and out of the single shared device KV cache.

Only **one** KV cache lives in device memory at any time.  When a
request is suspended, its KV data is copied to host (CPU) memory via
``model.save_kv_state()``.  When a request resumes, the snapshot is
copied back via ``model.restore_kv_state()``.

Prefix matching
---------------
When a new request arrives, the pool searches for a slot whose token
prefix matches the new request's prompt.  If found, the slot is reused
and inference resumes from the matched prefix length instead of
re-processing the entire prompt from scratch.  This is especially
beneficial for multi-turn conversations where the prompt grows by
appending new user/assistant turns.

This module is thread-safe — all public methods acquire a lock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


@dataclass
class KVSlot:
    """A single KV-cache slot in the pool.

    Attributes
    ----------
    slot_id : int
        Unique identifier within the pool.
    token_ids : list[int]
        The token sequence that has been processed into this slot's
        KV cache.  Used for prefix matching.
    kv_snapshot : object | None
        Opaque snapshot handle returned by ``model.save_kv_state()``.
        Backed by CPU memory.  Must be freed via the pool's
        ``_free_fn`` when discarded.
    pos : int
        Number of tokens represented by the snapshot.
    request_id : str | None
        The ID of the request currently bound to this slot,
        or None if the slot is free.
    last_used : float
        Timestamp of the last time this slot was actively used.
    """

    slot_id: int
    token_ids: list[int] = field(default_factory=list)
    kv_snapshot: Any = None     # opaque c_void_p snapshot handle
    pos: int = 0
    request_id: Optional[str] = None
    conversation_id: Optional[str] = None
    last_used: float = field(default_factory=time.time)

    @property
    def is_free(self) -> bool:
        return self.request_id is None


def _common_prefix_length(a: list[int], b: list[int]) -> int:
    """Return the length of the common prefix between two token lists."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class KVCachePool:
    """Pool of KV-cache slots with prefix matching.

    Parameters
    ----------
    max_slots : int
        Maximum number of concurrent KV-cache slots.  Each slot
        corresponds to one active request or one cached prefix.
    free_fn : callable | None
        Called with a snapshot handle when that snapshot is evicted or
        cleared.  Should call ``model.free_kv_snapshot(snapshot)``.
    """

    def __init__(self, max_slots: int = 8, free_fn: Callable | None = None) -> None:
        self._max_slots = max_slots
        self._slots: list[KVSlot] = [
            KVSlot(slot_id=i) for i in range(max_slots)
        ]
        self._lock = threading.Lock()
        self._free_fn = free_fn

    # ── Slot acquisition ─────────────────────────────────────────────

    def acquire_slot(
        self,
        request_id: str,
        input_ids: list[int],
    ) -> tuple[KVSlot, int]:
        """Find the best slot for a request and bind it.

        Search order:
        1. Exact prefix match — reuse a free slot whose token_ids are
           a prefix of ``input_ids``.  Prefer the longest prefix.
        2. Empty slot — a slot that has never been used.
        3. LRU eviction — evict the least-recently-used *free* slot.

        Returns
        -------
        (slot, prefix_len) : tuple[KVSlot, int]
            The acquired slot and the number of tokens that can be
            skipped (already in the KV cache).

        Raises
        ------
        RuntimeError
            If no slot is available (all bound to active requests).
        """
        with self._lock:
            # 1. Find best prefix match among free slots
            best_slot: Optional[KVSlot] = None
            best_prefix_len = 0

            for slot in self._slots:
                if not slot.is_free:
                    continue

                if slot.kv_snapshot is not None and len(slot.token_ids) > 0:
                    plen = _common_prefix_length(slot.token_ids, input_ids)
                    if plen > best_prefix_len:
                        best_prefix_len = plen
                        best_slot = slot

            if best_slot is not None and best_prefix_len > 0:
                logger.info(
                    "Prefix match: slot %d matches %d/%d tokens for request %s",
                    best_slot.slot_id,
                    best_prefix_len,
                    len(input_ids),
                    request_id,
                )
                best_slot.request_id = request_id
                best_slot.conversation_id = request_id
                best_slot.last_used = time.time()
                return best_slot, best_prefix_len

            # 2. Find an empty (never-used) free slot
            for slot in self._slots:
                if slot.is_free and slot.kv_snapshot is None:
                    slot.request_id = request_id
                    slot.conversation_id = request_id
                    slot.last_used = time.time()
                    logger.info(
                        "Empty slot %d acquired for request %s",
                        slot.slot_id,
                        request_id,
                    )
                    return slot, 0

            # 3. LRU eviction of the oldest free slot
            free_slots = [s for s in self._slots if s.is_free]
            if free_slots:
                lru_slot = min(free_slots, key=lambda s: s.last_used)
                logger.info(
                    "LRU eviction: slot %d (last_used=%.1f) for request %s",
                    lru_slot.slot_id,
                    lru_slot.last_used,
                    request_id,
                )
                lru_slot.request_id = request_id
                lru_slot.conversation_id = request_id
                lru_slot.token_ids = []
                # Free the old snapshot's CPU memory
                if lru_slot.kv_snapshot is not None and self._free_fn:
                    self._free_fn(lru_slot.kv_snapshot)
                lru_slot.kv_snapshot = None
                lru_slot.pos = 0
                lru_slot.last_used = time.time()
                return lru_slot, 0

            raise RuntimeError(
                f"KV cache pool exhausted ({self._max_slots} slots); "
                "all slots are bound to active requests."
            )

    def release_slot(self, slot: KVSlot) -> None:
        """Release a slot back to the pool (mark as free).

        The KV state is preserved so future requests can reuse
        the prefix via prefix matching.
        """
        with self._lock:
            slot.request_id = None
            slot.last_used = time.time()
            logger.debug(
                "Slot %d released (tokens=%d, pos=%d)",
                slot.slot_id,
                len(slot.token_ids),
                slot.pos,
            )

    def update_slot(
        self,
        slot: KVSlot,
        token_ids: list[int],
        kv_snapshot,
        pos: int = 0,
    ) -> None:
        """Update a slot's token sequence and KV snapshot.

        Called after generation to keep the slot in sync.
        Any previously stored snapshot is freed first.
        """
        with self._lock:
            # Free old snapshot if being replaced
            if slot.kv_snapshot is not None and slot.kv_snapshot is not kv_snapshot:
                if self._free_fn:
                    self._free_fn(slot.kv_snapshot)
            slot.token_ids = list(token_ids)
            slot.kv_snapshot = kv_snapshot
            slot.pos = pos
            slot.last_used = time.time()

    def clear_slot(self, slot: KVSlot) -> None:
        """Clear a slot's state completely, freeing its snapshot."""
        with self._lock:
            if slot.kv_snapshot is not None and self._free_fn:
                self._free_fn(slot.kv_snapshot)
            slot.token_ids = []
            slot.kv_snapshot = None
            slot.pos = 0
            slot.request_id = None
            slot.conversation_id = None
            slot.last_used = time.time()

    def evict_conversation(self, conversation_id: str) -> int:
        """Evict all slots associated with a conversation, freeing snapshots.

        Only free (released) slots are evicted.  Active slots (currently
        being used by an in-flight request) are skipped.

        Returns the number of slots evicted.
        """
        evicted = 0
        with self._lock:
            for slot in self._slots:
                if slot.conversation_id == conversation_id and slot.is_free:
                    if slot.kv_snapshot is not None and self._free_fn:
                        self._free_fn(slot.kv_snapshot)
                    slot.token_ids = []
                    slot.kv_snapshot = None
                    slot.pos = 0
                    slot.request_id = None
                    slot.conversation_id = None
                    slot.last_used = time.time()
                    evicted += 1
        if evicted:
            logger.info(
                "Evicted %d slot(s) for conversation %s",
                evicted, conversation_id,
            )
        return evicted

    # ── Diagnostics ──────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return pool statistics."""
        with self._lock:
            free = sum(1 for s in self._slots if s.is_free)
            active = self._max_slots - free
            cached = sum(
                1 for s in self._slots if s.is_free and s.kv_snapshot is not None
            )
            return {
                "kv_pool_max_slots": self._max_slots,
                "kv_pool_active": active,
                "kv_pool_free": free,
                "kv_pool_cached_prefixes": cached,
            }

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"KVCachePool(max={stats['kv_pool_max_slots']}, "
            f"active={stats['kv_pool_active']}, "
            f"free={stats['kv_pool_free']}, "
            f"cached={stats['kv_pool_cached_prefixes']})"
        )
