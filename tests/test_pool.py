import asyncio
import json
import threading
from contextlib import suppress

import pytest

from tbcore.client import Completion, RateLimitError, Usage
from tbcore.pool import ExecutionPool, WorkStore, enqueue


class FakeClient:
    """Async fake that sleeps per call and records what it processed."""

    def __init__(self, sleep_s=0.01, fail_times_for_tag=0):
        self.sleep_s = sleep_s
        self.fail_times_for_tag = fail_times_for_tag
        self.calls = []
        self.processed = []
        self.in_flight = 0
        self.peak_in_flight = 0

    async def complete(self, system=None, messages=None, cache_prefix=None, **kwargs):
        tag = messages[0]["content"]
        self.calls.append(tag)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.sleep_s)
        finally:
            self.in_flight -= 1
        if self.calls.count(tag) <= self.fail_times_for_tag:
            raise RateLimitError(f"simulated 429 for {tag}")
        self.processed.append(tag)
        return Completion(
            text=f"done:{tag}", usage=Usage(input_tokens=1, output_tokens=1), latency_s=self.sleep_s
        )


def make_store(tmp_path, name="work.db"):
    return WorkStore(str(tmp_path / name))


def payloads_for(n):
    return [{"messages": [{"role": "user", "content": str(i)}]} for i in range(n)]


async def test_pool_respects_concurrency_bound(tmp_path):
    client = FakeClient(sleep_s=0.05)
    store = make_store(tmp_path)
    enqueue(store, payloads_for(12))

    pool = ExecutionPool(client, store, concurrency=4, base_delay=0.01)
    counts = await pool.run()

    assert counts["done"] == 12
    assert client.peak_in_flight <= 4
    assert client.peak_in_flight > 1


async def test_retries_with_growing_backoff(tmp_path):
    client = FakeClient(sleep_s=0.001, fail_times_for_tag=2)
    store = make_store(tmp_path)
    enqueue(store, payloads_for(1))

    delays = []

    async def record_sleep(seconds):
        delays.append(seconds)

    pool = ExecutionPool(
        client,
        store,
        concurrency=1,
        max_attempts=5,
        base_delay=0.125,
        jitter=0.0,
        sleeper=record_sleep,
    )
    counts = await pool.run()

    assert counts["done"] == 1
    assert len(delays) == 2
    assert delays[0] == pytest.approx(0.125)
    assert delays[1] == pytest.approx(0.25)
    assert delays[1] > delays[0]
    assert len(client.calls) == 3


async def test_pool_killed_and_restarted_processes_every_item_once(tmp_path):
    client = FakeClient(sleep_s=0.005)
    store = make_store(tmp_path)
    enqueue(store, payloads_for(100))

    first_pool = ExecutionPool(client, store, concurrency=8, base_delay=0.01)
    task = asyncio.create_task(first_pool.run())
    while store.counts().get("done", 0) < 50:
        await asyncio.sleep(0.002)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    partial_done = store.counts().get("done", 0)
    assert 40 <= partial_done < 100

    second_pool = ExecutionPool(client, store, concurrency=8, base_delay=0.01)
    counts = await second_pool.run()

    assert counts["done"] == 100
    assert counts.get("failed", 0) == 0
    assert len(client.processed) == 100
    assert len(set(client.processed)) == 100


async def test_rows_exhausting_retry_cap_stay_failed_across_runs(tmp_path):
    class AlwaysRateLimited:
        def __init__(self):
            self.calls = 0

        async def complete(self, **kwargs):
            self.calls += 1
            raise RateLimitError("429 forever")

    client = AlwaysRateLimited()
    store = make_store(tmp_path)
    enqueue(store, payloads_for(1))

    pool = ExecutionPool(client, store, concurrency=1, max_attempts=2, base_delay=0.001)
    counts = await pool.run()
    assert counts["failed"] == 1
    assert client.calls == 2

    counts_after_rerun = await ExecutionPool(
        client, store, concurrency=1, max_attempts=2, base_delay=0.001
    ).run()
    assert counts_after_rerun["failed"] == 1
    assert client.calls == 2


async def test_non_transient_error_fails_without_retry(tmp_path):
    class Broken:
        def __init__(self):
            self.calls = 0

        async def complete(self, **kwargs):
            self.calls += 1
            raise ValueError("bad payload shape")

    client = Broken()
    store = make_store(tmp_path)
    enqueue(store, payloads_for(1))

    counts = await ExecutionPool(client, store, concurrency=1).run()

    assert counts["failed"] == 1
    assert client.calls == 1


async def test_results_are_written_back_to_the_store(tmp_path):
    client = FakeClient(sleep_s=0.001)
    store = make_store(tmp_path)
    enqueue(store, payloads_for(3))

    await ExecutionPool(client, store, concurrency=2).run()

    rows = store.conn.execute("SELECT id, status, result FROM work ORDER BY id").fetchall()
    assert all(row[1] == "done" for row in rows)
    first_result = json.loads(rows[0][2])
    assert first_result["text"] == "done:0"
    assert first_result["usage"]["input_tokens"] == 1


def test_claim_is_atomic_under_contention(tmp_path):
    store = make_store(tmp_path)
    enqueue(store, payloads_for(50))
    claimed = []
    lock = threading.Lock()

    def worker():
        while True:
            item = store.claim(max_attempts=1)
            if item is None:
                return
            with lock:
                claimed.append(item.id)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(claimed) == list(range(1, 51))
