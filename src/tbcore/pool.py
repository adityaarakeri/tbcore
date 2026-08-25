"""Resumable worker pool over a SQLite-backed work store."""

from __future__ import annotations

import asyncio
import inspect
import json
import random
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from tbcore.client import RateLimitError, TransientError

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


def enqueue(store: WorkStore, payloads: Iterable[Mapping[str, Any]]) -> int:
    """Insert pending rows for each payload."""
    return store.enqueue(payloads)


class ExecutionPool:
    """Drains a :class:`WorkStore` through a client with bounded concurrency.

    Rate-limit and transient errors are retried in-place with exponential
    backoff until ``max_attempts``; exhausted rows stay ``failed``. Calling
    :meth:`run` again resumes: only ``pending`` rows and ``failed`` rows under
    the retry cap are processed, so completed work is never redone.
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
        sleeper=None,
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
        self.sleeper = sleeper or asyncio.sleep

    async def run(self) -> dict[str, int]:
        """Drain all claimable work; returns per-status row counts."""
        self.store.reclaim_running()
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
        kwargs = {
            "system": payload.get("system"),
            "messages": payload["messages"],
            "cache_prefix": payload.get("cache_prefix"),
        }
        attempts = item.attempts
        while True:
            try:
                completion = await self._invoke(**kwargs)
            except TRANSIENT_EXCEPTIONS as exc:
                attempts += 1
                self.store.mark_failed(item.id, f"{type(exc).__name__}: {exc}")
                if attempts >= self.max_attempts:
                    return
                await self.sleeper(self._delay(attempts))
            except Exception as exc:
                self.store.mark_failed(
                    item.id, f"{type(exc).__name__}: {exc}", exhaust_at=self.max_attempts
                )
                return
            else:
                self.store.mark_done(
                    item.id,
                    {
                        "text": completion.text,
                        "usage": {
                            "input_tokens": completion.usage.input_tokens,
                            "output_tokens": completion.usage.output_tokens,
                        },
                        "latency_s": round(completion.latency_s, 6),
                    },
                )
                return

    async def _invoke(self, **kwargs):
        complete = self.client.complete
        if inspect.iscoroutinefunction(complete):
            return await complete(**kwargs)
        return await asyncio.to_thread(complete, **kwargs)

    def _delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        if self.jitter:
            delay *= 1.0 + random.uniform(0.0, self.jitter)
        return delay
