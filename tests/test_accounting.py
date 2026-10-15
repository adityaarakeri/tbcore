from pathlib import Path

from tbcore.accounting import DEFAULT_LOG_PATH, CallLog


def test_default_path_is_unchanged(monkeypatch):
    monkeypatch.delenv("TBCORE_LOG_PATH", raising=False)

    assert CallLog().path == DEFAULT_LOG_PATH == Path(".tbcore") / "calls.jsonl"


def test_env_var_overrides_default(monkeypatch, tmp_path):
    target = tmp_path / "env.jsonl"
    monkeypatch.setenv("TBCORE_LOG_PATH", str(target))

    log = CallLog()
    log.append({"a": 1})

    assert log.path == target
    assert log.entries() == [{"a": 1}]


def test_empty_env_var_is_ignored(monkeypatch):
    monkeypatch.setenv("TBCORE_LOG_PATH", "")

    assert CallLog().path == DEFAULT_LOG_PATH


def test_explicit_path_beats_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("TBCORE_LOG_PATH", str(tmp_path / "env.jsonl"))
    explicit = tmp_path / "explicit.jsonl"

    assert CallLog(explicit).path == explicit
    assert CallLog(DEFAULT_LOG_PATH).path == DEFAULT_LOG_PATH


def test_explicit_none_stays_in_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("TBCORE_LOG_PATH", str(tmp_path / "env.jsonl"))

    log = CallLog(None)
    other = CallLog(None)
    record = {"a": {"value": 1}}
    log.append(record)
    record["a"]["value"] = 2

    assert log.path is None
    assert log.entries() == [{"a": {"value": 1}}]
    assert other.entries() == []
    entries = log.entries()
    entries[0]["a"]["value"] = 3
    assert log.entries() == [{"a": {"value": 1}}]
    assert not (tmp_path / "env.jsonl").exists()
