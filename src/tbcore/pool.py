"""Resumable worker pool over a SQLite-backed work store."""

from __future__ import annotations

import asyncio
import inspect
import json
import random
import sqlite3
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from tbcore.client import Completion, RateLimitError, RetryableError, TransientError

TRANSIENT_EXCEPTIONS = (RateLimitError, TransientError, ConnectionError, TimeoutError)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS work (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_work_status ON work(status, id);
"""

_UPDATED_AT = "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


@dataclass(frozen=True)
class WorkItem:
    id: int
    payload: dict[str, Any]
    attempts: int


class WorkStore:
    """SQLite table of work rows with statuses: pending/running/done/failed."""

    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)

    def enqueue(self, payloads: Iterable[Mapping[str, Any]]) -> int:
        rows = [(json.dumps(dict(p), ensure_ascii=False),) for p in payloads]
        with self._lock:
            self.conn.executemany("INSERT INTO work(payload) VALUES (?)", rows)
        return len(rows)

    def claim(self, max_attempts: int) -> WorkItem | None:
        """Atomically claim the oldest claimable row, marking it running.

        A row is claimable when it is ``pending``, or ``failed`` with attempts
        still under ``max_attempts``.
        """
        cur = self.conn.cursor()
        with self._lock:
            cur.execute("BEGIN IMMEDIATE")
            try:
                row = cur.execute(
                    """
                    SELECT id, payload, attempts FROM work
                    WHERE status = 'pending' OR (status = 'failed' AND attempts < ?)
                    ORDER BY id
                    LIMIT 1
                    """,
                    (max_attempts,),
                ).fetchone()
                if row is None:
                    cur.execute("COMMIT")
                    return None
                cur.execute(
                    f"UPDATE work SET status='running', {_UPDATED_AT} WHERE id=?",
                    (row[0],),
                )
                cur.execute("COMMIT")
            except BaseException:
                cur.execute("ROLLBACK")
                raise
        return WorkItem(row[0], json.loads(row[1]), row[2])

    def mark_done(self, item_id: int, result: Mapping[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                f"UPDATE work SET status='done', result=?, error=NULL, {_UPDATED_AT} WHERE id=?",
                (json.dumps(dict(result), ensure_ascii=False), item_id),
            )

    def record_retry(self, item_id: int, error: str) -> None:
        """Count a failed attempt without releasing the running row to another worker."""
        with self._lock:
            self.conn.execute(
                f"UPDATE work SET error=?, attempts=attempts+1, {_UPDATED_AT} WHERE id=?",
                (error[:2000], item_id),
            )

    def mark_failed(self, item_id: int, error: str, *, exhaust_at: int | None = None) -> None:
        """Record a failure and bump attempts.

        With ``exhaust_at`` the attempt counter jumps to at least that value so
        the row is never re-claimed; used for non-transient errors that
        retrying cannot fix.
        """
        with self._lock:
            if exhaust_at is not None:
                self.conn.execute(
                    "UPDATE work SET status='failed', error=?, attempts=MAX(attempts+1, ?), "
                    f"{_UPDATED_AT} WHERE id=?",
                    (error[:2000], exhaust_at, item_id),
                )
            else:
                self.conn.execute(
                    f"UPDATE work SET status='failed', error=?, attempts=attempts+1, "
                    f"{_UPDATED_AT} WHERE id=?",
                    (error[:2000], item_id),
                )

    def reclaim_running(self) -> int:
        """Reset rows stuck in ``running`` (e.g. after a killed process)."""
        with self._lock:
            cur = self.conn.execute(
                f"UPDATE work SET status='pending', {_UPDATED_AT} WHERE status='running'"
            )
        return cur.rowcount

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self.conn.execute("SELECT status, COUNT(*) FROM work GROUP BY status")
            counts = {status: count for status, count in rows}
        return counts

    def close(self) -> None:
        self.conn.close()


def _result_row(completion: Completion) -> dict[str, Any]:
    usage = completion.usage
    return {
        "text": completion.text,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
        },
        "latency_s": round(completion.latency_s, 6),
    }


def enqueue(store: WorkStore, payloads: Iterable[Mapping[str, Any]]) -> int:
    """Insert pending rows for each payload."""
    return store.enqueue(payloads)


class ExecutionPool:
    """Drains a :class:`WorkStore` through a client with bounded concurrency.

    Rate-limit and transient errors are retried in-place with exponential
    backoff until ``max_attempts``; exhausted rows stay ``failed``. Calling
    :meth:`run` again resumes: only ``pending`` rows and ``failed`` rows under
    the retry cap are processed, so completed work is never redone.

    A server ``Retry-After`` hint raises the backoff floor. When any worker is
    rate limited, the whole pool pauses until that worker's backoff elapses, so
    sibling workers do not hammer the limit. ``sleeper`` and ``clock`` are
    injectable for tests (``clock`` must be monotonic seconds).
    """

    def __init__(
        self,
        client: Any,
        store: WorkStore,
        concurrency: int = 4,
        *,
        max_attempts: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 60.0,
        jitter: float = 0.1,
        sleeper: Callable[[float], Awaitable[object]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.client = client
        self.store = store
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.sleeper: Callable[[float], Awaitable[object]] = sleeper or asyncio.sleep
        self._clock: Callable[[], float] = clock or time.monotonic
        # Pool-wide "do not call before" time on the injected clock.
        self._resume_at = float("-inf")

    async def run(self) -> dict[str, int]:
        """Drain all claimable work; returns per-status row counts."""
        self.store.reclaim_running()
        self._resume_at = float("-inf")
        await asyncio.gather(*(self._worker() for _ in range(self.concurrency)))
        return self.store.counts()

    async def _worker(self) -> None:
        while True:
            item = self.store.claim(self.max_attempts)
            if item is None:
                return
            await self._process(item)

    async def _process(self, item: WorkItem) -> None:
        payload = item.payload
        if not payload.get("messages"):
            self.store.mark_failed(
                item.id, "payload missing 'messages'", exhaust_at=self.max_attempts
            )
            return
        kwargs: dict[str, Any] = {
            "system": payload.get("system"),
            "messages": payload["messages"],
            "cache_prefix": payload.get("cache_prefix"),
        }
        # Forwarded only when present so clients without these kwargs keep working.
        for key in ("max_tokens", "temperature", "tools"):
            if payload.get(key) is not None:
                kwargs[key] = payload[key]
        attempts = item.attempts
        own_pause: float | None = None
        while True:
            await self._wait_for_pause(skip=own_pause)
            try:
                completion = await self._invoke(**kwargs)
            except TRANSIENT_EXCEPTIONS as exc:
                attempts += 1
                error = f"{type(exc).__name__}: {exc}"
                hint = exc.retry_after if isinstance(exc, RetryableError) else None
                delay = max(self._delay(attempts), hint or 0.0)
                if isinstance(exc, RateLimitError):
                    own_pause = self._pause_for(delay)
                if attempts >= self.max_attempts:
                    self.store.mark_failed(item.id, error)
                    return
                self.store.record_retry(item.id, error)
                await self.sleeper(delay)
            except Exception as exc:
                self.store.mark_failed(
                    item.id, f"{type(exc).__name__}: {exc}", exhaust_at=self.max_attempts
                )
                return
            else:
                self.store.mark_done(item.id, _result_row(completion))
                return

    def _pause_for(self, delay: float) -> float:
        """Extend the pool-wide pause to ``now + delay``; returns the new resume time."""
        self._resume_at = max(self._resume_at, self._clock() + delay)
        return self._resume_at

    async def _wait_for_pause(self, skip: float | None = None) -> None:
        """Wait out the pool-wide pause, unless this worker already slept through it."""
        while True:
            target = self._resume_at
            remaining = target - self._clock()
            if remaining <= 0 or target == skip:
                return
            await self.sleeper(remaining)
            if self._resume_at == target:
                return

    async def _invoke(self, **kwargs: Any) -> Completion:
        complete = self.client.complete
        completion: Completion
        if inspect.iscoroutinefunction(complete):
            completion = await complete(**kwargs)
        else:
            completion = await asyncio.to_thread(complete, **kwargs)
        return completion

    def _delay(self, attempt: int) -> float:
        delay = min(self.base_delay * 2.0 ** (attempt - 1), self.max_delay)
        if self.jitter:
            delay *= 1.0 + random.uniform(0.0, self.jitter)
        return delay
