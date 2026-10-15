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
