"""Portable completed-run bundles retain replayable captured inputs."""

from pathlib import Path

import pytest

from tbcore.evidence import CapturePolicy
from tbcore.experiment import ExperimentSpec, RunRecord
from tbcore.experiment_store import ExperimentStore


def test_bundle_reopens_workspace_and_evaluation_in_new_store(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "task.txt").write_text("frozen", encoding="utf-8")
    original = ExperimentStore(tmp_path / "original", CapturePolicy(content_enabled=True))
    snapshot = original.capture_workspace(source)
    original.add_spec(ExperimentSpec("exp", "task", {}, snapshot.snapshot_id))
    run = RunRecord(
        "exp",
        "run",
        "trial",
        "attempt",
        "pending",
        "2026-10-14T12:00:00Z",
        workspace_ref=snapshot.snapshot_id,
    )
    original.save_run(run)
    original.save_run(run.transition("running", at="2026-10-14T12:00:01Z"))
    original.save_run(original.get_run("run").transition("completed", at="2026-10-14T12:00:02Z"))
    original.append_event("run", "evaluation", {"event_type": "evaluation", "verdict": "pass"})
    bundle = original.export_run("run")
    imported = ExperimentStore(tmp_path / "imported", CapturePolicy(content_enabled=True))
    imported.import_bundle(bundle)
    imported.restore_workspace(snapshot.snapshot_id, tmp_path / "restored")
    assert (tmp_path / "restored" / "task.txt").read_text(encoding="utf-8") == "frozen"
    assert imported.events("run")[0]["fields"]["verdict"] == "pass"
    original.close()
    imported.close()


def test_bundle_rejects_tampered_workspace_blob(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "task.txt").write_text("frozen", encoding="utf-8")
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    snapshot = store.capture_workspace(source)
    store.add_spec(ExperimentSpec("exp", "task", {}, snapshot.snapshot_id))
    run = RunRecord(
        "exp",
        "run",
        "trial",
        "attempt",
        "pending",
        "2026-10-14T12:00:00Z",
        workspace_ref=snapshot.snapshot_id,
    )
    store.save_run(run)
    bundle = store.export_run("run")
    bundle["workspaces"][snapshot.snapshot_id]["blobs"][0]["content_base64"] = "dGFtcGVyZWQ="
    with pytest.raises(ValueError, match="workspace blob"):
        ExperimentStore(tmp_path / "imported", CapturePolicy(content_enabled=True)).import_bundle(
            bundle
        )
    store.close()
