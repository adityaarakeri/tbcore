import json
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from tbcore.evidence import CapturePolicy
from tbcore.experiment import ExperimentSpec, RunRecord
from tbcore.experiment_store import ExperimentStore


def _spec():
    return ExperimentSpec("exp-1", "task-1", {"model": "local"})


def _run(status="pending", artifact_ids=()):
    return RunRecord(
        experiment_id="exp-1",
        run_id="run-1",
        trial_id="trial-1",
        attempt_id="attempt-1",
        status=status,
        created_at="2026-10-14T12:00:00Z",
        started_at="2026-10-14T12:00:01Z" if status != "pending" else None,
        finished_at="2026-10-14T12:00:03Z" if status == "completed" else None,
        artifact_ids=artifact_ids,
    )


def test_reopen_run_with_ordered_idempotent_events(tmp_path):
    root = tmp_path / "store"
    store = ExperimentStore(root)
    store.add_spec(_spec())
    store.save_run(_run())
    assert store.append_event("run-1", "event-1", {"event_type": "start"}) == 1
    assert store.append_event("run-1", "event-1", {"event_type": "start"}) == 1
    assert store.append_event("run-1", "event-2", {"event_type": "step"}) == 2
    with pytest.raises(ValueError, match="duplicate"):
        store.append_event("run-1", "event-1", {"event_type": "changed"})
    store.save_run(_run().transition("running", at="2026-10-14T12:00:01Z"))
    store.close()

    reopened = ExperimentStore(root)
    assert reopened.get_spec("exp-1") == _spec()
    assert reopened.get_run("run-1").status == "running"
    assert [event["event_id"] for event in reopened.events("run-1")] == ["event-1", "event-2"]
    assert [event["sequence"] for event in reopened.events("run-1")] == [1, 2]
    reopened.close()


def test_concurrent_writers_have_unique_ordered_sequences(tmp_path):
    store = ExperimentStore(tmp_path / "store")
    store.add_spec(_spec())
    store.save_run(_run())
    with ThreadPoolExecutor(max_workers=6) as executor:
        sequences = list(
            executor.map(
                lambda index: store.append_event(
                    "run-1", f"event-{index}", {"event_type": "step", "exit_code": index}
                ),
                range(30),
            )
        )
    assert sorted(sequences) == list(range(1, 31))
    assert [event["sequence"] for event in store.events("run-1")] == list(range(1, 31))
    store.close()


def test_interrupted_artifact_write_is_recoverable(tmp_path, monkeypatch):
    root = tmp_path / "store"
    store = ExperimentStore(root, CapturePolicy(content_enabled=True))
    store.add_spec(_spec())
    store.save_run(_run())

    def interrupted(*args):
        raise RuntimeError("metadata write interrupted")

    monkeypatch.setattr(store, "_commit_artifact_metadata", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        store.add_artifact("run-1", "result.txt", b"output")
    assert (root / "artifacts" / "run-1" / "result.txt").exists()
    store.close()

    reopened = ExperimentStore(root, CapturePolicy(content_enabled=True))
    assert not (root / "artifacts" / "run-1" / "result.txt").exists()
    reopened.save_run(_run("running"))
    with pytest.raises(ValueError, match="artifact"):
        reopened.save_run(_run("completed", ("missing.txt",)))
    reopened.add_artifact("run-1", "result.txt", b"output")
    reopened.save_run(_run("completed", ("result.txt",)))
    assert reopened.get_run("run-1").status == "completed"
    reopened.close()


def test_export_import_checks_artifact_hash_and_links(tmp_path):
    source = ExperimentStore(
        tmp_path / "source", CapturePolicy(content_enabled=True, secret_values=("private",))
    )
    source.add_spec(_spec())
    source.save_run(_run())
    source.append_event("run-1", "event-1", {"event_type": "step", "stdout": "private"})
    digest = source.add_artifact("run-1", "result.txt", b"output")
    source.save_run(_run("running"))
    source.save_run(_run("completed", ("result.txt",)))
    bundle = source.export_run("run-1")
    assert bundle["artifacts"][0]["sha256"] == digest
    assert "private" not in json.dumps(bundle)

    imported = ExperimentStore(tmp_path / "imported", CapturePolicy(content_enabled=True))
    imported.import_bundle(bundle)
    assert imported.get_run("run-1") == source.get_run("run-1")
    assert imported.events("run-1") == source.events("run-1")
    bad = json.loads(json.dumps(bundle))
    bad["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        ExperimentStore(tmp_path / "bad", CapturePolicy(content_enabled=True)).import_bundle(bad)
    source.close()
    imported.close()


def test_schema_migration_copy_and_unsupported_downgrade(tmp_path):
    original = tmp_path / "original"
    store = ExperimentStore(original)
    store.add_spec(_spec())
    store.save_run(_run())
    store.close()
    migrated = tmp_path / "migrated"
    shutil.copytree(original, migrated)
    with sqlite3.connect(migrated / "experiments.sqlite3") as db:
        db.execute("PRAGMA user_version=0")
    upgraded = ExperimentStore(migrated)
    assert upgraded.get_run("run-1") == _run()
    upgraded.close()
    with sqlite3.connect(original / "experiments.sqlite3") as db:
        db.execute("PRAGMA user_version=99")
    before = (original / "experiments.sqlite3").read_bytes()
    with pytest.raises(ValueError, match="schema version"):
        ExperimentStore(original)
    assert (original / "experiments.sqlite3").read_bytes() == before


@pytest.mark.parametrize("status", ["failed", "cancelled", "timed_out", "interrupted"])
def test_export_import_preserves_terminal_run_without_observations(tmp_path, status):
    run = RunRecord(
        "exp-1",
        "run-1",
        "trial-1",
        "attempt-1",
        status,
        "2026-10-14T12:00:00Z",
        finished_at="2026-10-14T12:00:03Z",
    )
    source = ExperimentStore(tmp_path / f"source-{status}")
    source.add_spec(_spec())
    source.save_run(_run())
    source.save_run(run)
    target = ExperimentStore(tmp_path / f"target-{status}")
    target.import_bundle(source.export_run("run-1"))
    assert target.get_run("run-1") == run
    source.close()
    target.close()


def test_completed_run_surfaces_missing_artifact_after_open(tmp_path):
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    store.add_spec(_spec())
    store.save_run(_run())
    store.add_artifact("run-1", "result.txt", b"output")
    store.save_run(_run("running"))
    store.save_run(_run("completed", ("result.txt",)))
    (tmp_path / "store" / "artifacts" / "run-1" / "result.txt").unlink()
    with pytest.raises(ValueError, match="artifact"):
        store.get_run("run-1")
    store.close()


class _GuardedCursor:
    def __init__(self, cursor, lock):
        self._cursor = cursor
        self._lock = lock

    def _check(self):
        assert self._lock._is_owned(), "store cursor read without holding the store lock"

    def fetchone(self):
        self._check()
        return self._cursor.fetchone()

    def fetchall(self):
        self._check()
        return self._cursor.fetchall()

    def __iter__(self):
        self._check()
        return iter(self._cursor.fetchall())

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _GuardedConnection:
    """Fails any query or fetch made without the store's lock held by this thread."""

    def __init__(self, conn, lock):
        self._conn = conn
        self._lock = lock

    def execute(self, *args):
        assert self._lock._is_owned(), "store connection used without holding the store lock"
        return _GuardedCursor(self._conn.execute(*args), self._lock)

    def close(self):
        assert self._lock._is_owned(), "store connection closed without holding the store lock"
        self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_every_connection_use_holds_the_store_lock(tmp_path):
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    store.conn = _GuardedConnection(store.conn, store._lock)

    store.add_spec(_spec())
    store.save_run(_run())
    store.save_run(_run().transition("running", at="2026-10-14T12:00:01Z"))
    store.add_artifact("run-1", "out.txt", b"done")
    store.append_event("run-1", "event-1", {"event_type": "step"})
    assert store.claim_trial("trial-1", "owner-1")
    store.register_trial_container("trial-1", "owner-1", "tbcore-sess-abc123")
    assert store.trial_containers("trial-1") == ["tbcore-sess-abc123"]
    store.forget_trial_container("trial-1", "tbcore-sess-abc123")
    store.release_trial("trial-1", "owner-1")
    store.save_run(_run(status="completed", artifact_ids=("out.txt",)))
    assert store.get_spec("exp-1") == _spec()
    assert store.get_run("run-1").status == "completed"
    assert [run.run_id for run in store.all_runs()] == ["run-1"]
    assert [run.run_id for run in store.runs_for_trial("trial-1")] == ["run-1"]
    assert [event["event_id"] for event in store.events("run-1")] == ["event-1"]
    assert store.export_run("run-1")["artifacts"][0]["name"] == "out.txt"
    store.close()


def test_concurrent_reads_and_writes_on_one_store_do_not_corrupt_results(tmp_path):
    store = ExperimentStore(tmp_path / "store")
    store.add_spec(_spec())
    store.save_run(_run())

    def worker(index):
        for step in range(150):
            if index % 2:
                store.append_event("run-1", f"event-{index}-{step}", {"event_type": "step"})
            else:
                assert store.get_run("run-1").run_id == "run-1"
                events = store.events("run-1")
                assert all(event["fields"]["event_type"] == "step" for event in events)
                assert [run.run_id for run in store.all_runs()] == ["run-1"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        for future in [executor.submit(worker, index) for index in range(8)]:
            future.result()

    assert len(store.events("run-1")) == 4 * 150
    store.close()
