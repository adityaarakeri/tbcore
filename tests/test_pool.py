import asyncio
import json
import threading
from contextlib import suppress

import pytest

from tbcore.client import Completion, RateLimitError, TransientError, Usage
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


@pytest.mark.parametrize("error_type", [TransientError, RateLimitError])
async def test_retry_keeps_job_owned_during_backoff(tmp_path, error_type):
    second_started = asyncio.Event()
    owner_sleeping = asyncio.Event()
    second_done = asyncio.Event()
    release_backoff = asyncio.Event()

    class CoordinatedClient:
        def __init__(self):
            self.calls = []

        async def complete(self, *, messages, **kwargs):
            tag = messages[0]["content"]
            self.calls.append(tag)
            if tag == "0" and self.calls.count("0") == 1:
                await second_started.wait()
                raise error_type("retry me")
            if tag == "1":
                second_started.set()
                await owner_sleeping.wait()
                second_done.set()
            return Completion(text=tag, usage=Usage(input_tokens=1, output_tokens=1), latency_s=0.0)

    async def sleeper(_seconds):
        owner_sleeping.set()
        await release_backoff.wait()

    client = CoordinatedClient()
    store = make_store(tmp_path)
    enqueue(store, payloads_for(2))
    pool = ExecutionPool(client, store, concurrency=2, base_delay=0.01, jitter=0.0, sleeper=sleeper)
    running = asyncio.create_task(pool.run())
    try:
        await asyncio.wait_for(second_done.wait(), timeout=2)
        await asyncio.sleep(0)
        release_backoff.set()
        counts = await asyncio.wait_for(running, timeout=2)
    finally:
        release_backoff.set()
        if not running.done():
            running.cancel()
            with suppress(asyncio.CancelledError):
                await running

    assert counts["done"] == 2
    assert client.calls.count("0") == 2
    assert client.calls.count("1") == 1


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


async def test_cancel_during_backoff_resumes_with_attempt_count(tmp_path):
    sleeping = asyncio.Event()
    release = asyncio.Event()

    class AlwaysTransient:
        def __init__(self):
            self.calls = 0

        async def complete(self, **kwargs):
            self.calls += 1
            raise TransientError("retry me")

    async def blocked_sleep(_seconds):
        sleeping.set()
        await release.wait()

    client = AlwaysTransient()
    store = make_store(tmp_path)
    enqueue(store, payloads_for(1))
    running = asyncio.create_task(
        ExecutionPool(
            client,
            store,
            concurrency=1,
            max_attempts=2,
            base_delay=0.01,
            sleeper=blocked_sleep,
        ).run()
    )
    await asyncio.wait_for(sleeping.wait(), timeout=2)
    running.cancel()
    with suppress(asyncio.CancelledError):
        await running

    assert store.conn.execute("SELECT status, attempts FROM work").fetchone() == ("running", 1)
    counts = await ExecutionPool(
        client, store, concurrency=1, max_attempts=2, base_delay=0.01
    ).run()
    assert counts["failed"] == 1
    assert store.conn.execute("SELECT attempts FROM work").fetchone()[0] == 2
    await ExecutionPool(client, store, concurrency=1, max_attempts=2).run()
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


class RecordingClient:
    """Sync client that records kwargs; optionally raises scripted errors first."""

    def __init__(self, errors=()):
        self.errors = list(errors)
        self.kwargs = []

    def complete(self, system=None, messages=None, cache_prefix=None, **kwargs):
        self.kwargs.append({"system": system, "cache_prefix": cache_prefix, **kwargs})
        if self.errors:
            raise self.errors.pop(0)
        return Completion(
            text="ok",
            usage=Usage(
                input_tokens=3,
                output_tokens=2,
                cache_read_input_tokens=5,
                cache_creation_input_tokens=7,
            ),
            latency_s=0.01,
        )


async def test_pool_forwards_optional_call_params_only_when_present(tmp_path):
    client = RecordingClient()
    store = make_store(tmp_path)
    tools = [{"name": "t", "input_schema": {"type": "object"}}]
    enqueue(
        store,
        [
            {"messages": [{"role": "user", "content": "a"}]},
            {
                "messages": [{"role": "user", "content": "b"}],
                "max_tokens": 50,
                "temperature": 0.0,
                "tools": tools,
            },
        ],
    )

    await ExecutionPool(client, store, concurrency=1).run()

    plain, full = client.kwargs
    assert "max_tokens" not in plain
    assert "temperature" not in plain
    assert "tools" not in plain
    assert full["max_tokens"] == 50
    assert full["temperature"] == 0.0
    assert full["tools"] == tools


async def test_pool_stores_cache_token_usage(tmp_path):
    store = make_store(tmp_path)
    enqueue(store, payloads_for(1))

    await ExecutionPool(RecordingClient(), store, concurrency=1).run()

    usage = json.loads(store.conn.execute("SELECT result FROM work").fetchone()[0])["usage"]
    assert usage == {
        "input_tokens": 3,
        "output_tokens": 2,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 7,
    }


async def test_retry_after_raises_the_backoff_floor(tmp_path):
    client = RecordingClient(errors=[RateLimitError("429", retry_after=9.0)])
    store = make_store(tmp_path)
    enqueue(store, payloads_for(1))
    delays = []

    async def record_sleep(seconds):
        delays.append(seconds)

    pool = ExecutionPool(
        client, store, concurrency=1, base_delay=0.5, jitter=0.0, sleeper=record_sleep
    )
    counts = await pool.run()

    assert counts["done"] == 1
    assert delays == [9.0]


async def test_small_retry_after_does_not_shrink_backoff(tmp_path):
    client = RecordingClient(errors=[RateLimitError("429", retry_after=0.1)])
    store = make_store(tmp_path)
    enqueue(store, payloads_for(1))
    delays = []

    async def record_sleep(seconds):
        delays.append(seconds)

    await ExecutionPool(
        client, store, concurrency=1, base_delay=2.0, jitter=0.0, sleeper=record_sleep
    ).run()

    assert delays == [2.0]


async def test_rate_limit_pauses_other_workers(tmp_path):
    """One worker's 429 makes the other worker wait out the same window before its next call."""
    now = [0.0]
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    class Scripted:
        def __init__(self):
            self.calls = []

        def complete(self, system=None, messages=None, cache_prefix=None, **kwargs):
            tag = messages[0]["content"]
            self.calls.append((tag, now[0]))
            if tag == "0" and len([c for c in self.calls if c[0] == "0"]) == 1:
                raise RateLimitError("429", retry_after=30.0)
            return Completion(text=tag, usage=Usage(input_tokens=1, output_tokens=1), latency_s=0.0)

    client = Scripted()
    store = make_store(tmp_path)
    enqueue(store, payloads_for(4))
    pool = ExecutionPool(
        client,
        store,
        concurrency=2,
        base_delay=1.0,
        jitter=0.0,
        sleeper=fake_sleep,
        clock=lambda: now[0],
    )

    counts = await pool.run()

    assert counts["done"] == 4
    # Worker A slept the full 30s itself; worker B waited out the remainder before later calls.
    assert sleeps.count(30.0) == 1
    later = [at for tag, at in client.calls if tag in ("2", "3")]
    assert later
    assert all(at >= 30.0 for at in later)
