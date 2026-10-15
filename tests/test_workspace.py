import json
import os

import pytest

from tbcore.evidence import CapturePolicy
from tbcore.experiment import ExperimentSpec, RunRecord
from tbcore.experiment_store import ExperimentStore
from tbcore.sandbox import build_codex_image
from tbcore.workspace import EnvironmentProvenance, WorkspaceStore


def test_snapshot_reconstructs_dirty_untracked_and_binary_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "tracked.txt").write_text("modified", encoding="utf-8")
    (source / "untracked.txt").write_text("new", encoding="utf-8")
    (source / "binary.bin").write_bytes(b"\x00\xff\x01")
    (source / "ignored.txt").write_text("exclude", encoding="utf-8")
    store = WorkspaceStore(
        tmp_path / "snapshots", CapturePolicy(content_enabled=True), allow_binary=True
    )

    manifest = store.capture(source, exclude=("ignored.txt",))
    (source / "tracked.txt").write_text("later", encoding="utf-8")
    (source / "untracked.txt").unlink()
    restored = tmp_path / "restored"
    store.restore(manifest.snapshot_id, restored)

    assert manifest.complete
    assert (restored / "tracked.txt").read_text(encoding="utf-8") == "modified"
    assert (restored / "untracked.txt").read_text(encoding="utf-8") == "new"
    assert (restored / "binary.bin").read_bytes() == b"\x00\xff\x01"
    assert not (restored / "ignored.txt").exists()
    assert store.get_manifest(manifest.snapshot_id) == manifest


def test_metadata_only_reports_incomplete_reconstruction(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "secret-name.txt").write_text("content", encoding="utf-8")
    store = WorkspaceStore(tmp_path / "snapshots", CapturePolicy())
    manifest = store.capture(source)
    assert not manifest.complete
    assert manifest.file_count == 1
    assert manifest.entries == ()
    assert "secret-name" not in json.dumps(manifest.to_dict())
    with pytest.raises(ValueError, match="incomplete"):
        store.restore(manifest.snapshot_id, tmp_path / "restored")


def test_symlinks_binary_without_opt_in_and_size_limits_fail_explicitly(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    binary = source / "binary.bin"
    binary.write_bytes(b"\x00\xff")
    store = WorkspaceStore(tmp_path / "snapshots", CapturePolicy(content_enabled=True))
    with pytest.raises(ValueError, match="binary"):
        store.capture(source)
    binary.unlink()
    (source / "large.txt").write_text("123456", encoding="utf-8")
    limited = WorkspaceStore(
        tmp_path / "limited", CapturePolicy(content_enabled=True), max_file_bytes=5
    )
    with pytest.raises(ValueError, match="size"):
        limited.capture(source)
    (source / "large.txt").unlink()
    (source / "safe.txt").write_text("safe", encoding="utf-8")
    try:
        (source / "link.txt").symlink_to(tmp_path / "outside.txt")
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="symlink"):
        store.capture(source)


def test_missing_or_corrupt_blob_is_detected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "file.txt").write_text("original", encoding="utf-8")
    store = WorkspaceStore(tmp_path / "snapshots", CapturePolicy(content_enabled=True))
    manifest = store.capture(source)
    blob = tmp_path / "snapshots" / "blobs" / manifest.entries[0].sha256
    blob.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        store.restore(manifest.snapshot_id, tmp_path / "restored")
    blob.unlink()
    with pytest.raises(ValueError, match="missing"):
        store.restore(manifest.snapshot_id, tmp_path / "restored")


def test_total_limit_and_configured_secrets_are_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.txt").write_text("abcd", encoding="utf-8")
    (source / "two.txt").write_text("efgh", encoding="utf-8")
    limited = WorkspaceStore(
        tmp_path / "limited", CapturePolicy(content_enabled=True), max_total_bytes=7
    )
    with pytest.raises(ValueError, match="total size"):
        limited.capture(source)
    (source / "two.txt").write_text("sk-secret", encoding="utf-8")
    guarded = WorkspaceStore(
        tmp_path / "guarded",
        CapturePolicy(content_enabled=True, secret_values=("sk-secret",)),
    )
    with pytest.raises(ValueError, match="configured secret"):
        guarded.capture(source)


def test_environment_provenance_round_trip_and_unavailable_image(tmp_path):
    provenance = EnvironmentProvenance(
        image_id="sha256:image",
        registry_digest="repo@sha256:digest",
        tbcore_version="0.1.0",
        agent_cli_version="codex 1.0",
        model="local-model",
        provider="local",
        resource_limits={"memory_mb": 512, "cpus": 1},
        runtime="runsc",
        policy_config={"network": "none"},
    )
    store = WorkspaceStore(tmp_path / "snapshots", CapturePolicy())
    ref = store.record_environment(provenance)
    assert store.get_environment(ref) == provenance
    provenance.verify_image({"sha256:image"})
    with pytest.raises(ValueError, match="image unavailable"):
        provenance.verify_image(set())
    assert os.path.isfile(tmp_path / "snapshots" / "environments" / f"{ref}.json")


def test_run_links_to_reconstructable_inputs_and_environment(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("print('before')", encoding="utf-8")
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    snapshot = store.capture_workspace(source)
    environment = EnvironmentProvenance(
        image_id="sha256:image",
        registry_digest=None,
        tbcore_version="0.1.0",
        agent_cli_version="codex 1.0",
        model="local-model",
        provider="local",
        resource_limits={"memory_mb": 512},
        runtime="runc",
        policy_config={"network": "none"},
    )
    environment_ref = store.record_environment(environment)
    store.add_spec(ExperimentSpec("exp-1", "task-1", {}, snapshot.snapshot_id, environment_ref))
    store.save_run(
        RunRecord(
            "exp-1",
            "run-1",
            "trial-1",
            "attempt-1",
            "pending",
            "2026-10-14T12:00:00Z",
            workspace_ref=snapshot.snapshot_id,
            environment_ref=environment_ref,
        )
    )
    store.close()
    (source / "main.py").write_text("print('after')", encoding="utf-8")
    reopened = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    reopened.restore_workspace(snapshot.snapshot_id, tmp_path / "restored")
    assert (tmp_path / "restored" / "main.py").read_text(encoding="utf-8") == "print('before')"
    assert reopened.get_run("run-1").workspace_ref == snapshot.snapshot_id
    assert reopened.workspaces.get_environment(environment_ref) == environment
    reopened.close()


def test_controlled_codex_build_requires_exact_version(monkeypatch):
    calls = []

    def docker(docker_bin, args, **kwargs):
        calls.append((docker_bin, args))

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr("tbcore.sandbox._docker", docker)
    with pytest.raises(ValueError, match="exact"):
        build_codex_image("latest")
    assert not calls
    tag = build_codex_image("1.2.3")
    assert tag == "tbcore/codex:1.2.3"
    assert "CODEX_VERSION=1.2.3" in calls[0][1]
