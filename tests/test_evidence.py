import json
import os
import stat

import pytest

from tbcore.evidence import CapturePolicy, EvidenceArtifacts


def test_metadata_only_capture_omits_content_and_credentials():
    policy = CapturePolicy(secret_values=("sk-live-secret",))
    captured = policy.capture(
        {
            "event_id": "event-1",
            "status": "failed",
            "exit_code": 1,
            "headers": {"Authorization": "Bearer sk-live-secret"},
            "environment": {"API_KEY": "sk-live-secret"},
            "command": "agent --key sk-live-secret",
            "stdout": "sk-live-secret",
            "stderr": "sk-live-secret",
            "prompt": "sk-live-secret",
            "tool_result": {"text": "sk-live-secret"},
            "other_content": "sk-live-secret",
        }
    )
    exported = json.dumps(captured)
    assert captured["event_id"] == "event-1"
    assert captured["exit_code"] == 1
    assert "sk-live-secret" not in exported
    assert "Bearer" not in exported
    assert "agent --key" not in exported


def test_enabled_content_redacts_nested_fields_and_known_secrets(tmp_path):
    policy = CapturePolicy(content_enabled=True, secret_values=("sk-live-secret",))
    captured = policy.capture(
        {
            "command": "agent --key sk-live-secret",
            "stdout": "token: sk-live-secret",
            "prompt": "use sk-live-secret",
            "tool_result": {"message": "sk-live-secret", "password": "another-secret"},
            "headers": {"Authorization": "Bearer another-secret"},
            "environment": {"SAFE": "also-secret"},
        }
    )
    path = tmp_path / "export.json"
    path.write_text(json.dumps(captured), encoding="utf-8")
    exported = path.read_text(encoding="utf-8")
    assert captured["command"] == "agent --key [REDACTED]"
    assert "sk-live-secret" not in exported
    assert "another-secret" not in exported
    assert "also-secret" not in exported
    assert captured["tool_result"]["password"] == "[REDACTED]"


def test_binary_and_oversized_output_is_not_embedded():
    policy = CapturePolicy(content_enabled=True, max_text_bytes=8)
    captured = policy.capture({"stdout": b"\x00\xffsecret", "stderr": "ninebytes"})
    assert captured["stdout"] == {"omitted": "binary", "bytes": 8}
    assert captured["stderr"] == {"omitted": "oversized", "bytes": 9}


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "C:\\escape", "\\\\server\\share", "sub/file"]
)
def test_artifact_paths_cannot_escape_or_nest(tmp_path, name):
    store = EvidenceArtifacts(tmp_path / "store", CapturePolicy(content_enabled=True))
    with pytest.raises(ValueError, match="artifact"):
        store.write("run-1", name, b"safe")
    assert not (tmp_path / "escape").exists()


def test_artifact_capture_is_explicit_bounded_redacted_and_restrictive(tmp_path):
    disabled = EvidenceArtifacts(tmp_path / "disabled", CapturePolicy())
    with pytest.raises(ValueError, match="content capture"):
        disabled.write("run-1", "output.txt", b"value")

    store = EvidenceArtifacts(
        tmp_path / "store",
        CapturePolicy(
            content_enabled=True, secret_values=("sk-live-secret",), max_artifact_bytes=32
        ),
    )
    digest = store.write("run-1", "output.txt", b"key=sk-live-secret")
    output = tmp_path / "store" / "run-1" / "output.txt"
    assert output.read_text(encoding="utf-8") == "key=[REDACTED]"
    assert len(digest) == 64
    with pytest.raises(ValueError, match="size"):
        store.write("run-1", "large.txt", b"x" * 33)
    with pytest.raises(ValueError, match="binary"):
        store.write("run-1", "binary.bin", b"\x00\xff")
    expanding = EvidenceArtifacts(
        tmp_path / "expanding",
        CapturePolicy(content_enabled=True, secret_values=("x",), max_artifact_bytes=32),
    )
    with pytest.raises(ValueError, match="size"):
        expanding.write("run-1", "expanded.txt", b"xxxx")
    if os.name != "nt":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
        assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700


def test_retention_cleanup_keeps_referenced_artifacts(tmp_path):
    store = EvidenceArtifacts(tmp_path / "store", CapturePolicy(content_enabled=True))
    store.write("run-1", "keep.txt", b"keep")
    store.write("run-1", "remove.txt", b"remove")
    store.write("run-2", "old.txt", b"old")

    store.cleanup({"run-1": {"keep.txt"}})

    assert (tmp_path / "store" / "run-1" / "keep.txt").exists()
    assert not (tmp_path / "store" / "run-1" / "remove.txt").exists()
    assert not (tmp_path / "store" / "run-2").exists()
