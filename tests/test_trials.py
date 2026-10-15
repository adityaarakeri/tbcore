from __future__ import annotations

import threading
from pathlib import Path

import pytest

from tbcore.evidence import CapturePolicy
from tbcore.experiment_store import ExperimentStore
from tbcore.trials import SandboxTrialExecutor, TrialCase, TrialRunner


def test_matrix_uses_fresh_snapshots_and_resume_skips_completed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer.txt").write_text("initial", encoding="utf-8")
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    snapshot = store.capture_workspace(source)
    cases = [
        TrialCase("task", "base", snapshot.snapshot_id, {}, 0),
        TrialCase("task", "candidate", snapshot.snapshot_id, {}, 1),
    ]
    seen: list[str] = []

    def execute(case: TrialCase, workspace: Path, attempt_id: str, context) -> None:
        assert (workspace / "answer.txt").read_text(encoding="utf-8") == "initial"
        seen.append(case.configuration_id)
        (workspace / "answer.txt").write_text(case.configuration_id, encoding="utf-8")

    runner = TrialRunner(store, "exp", execute, output_paths=("answer.txt",), max_workers=2)
    first = runner.run(cases)
    assert len(first) == 2
    assert {run.status for run in first} == {"completed"}
    assert {store._verified_artifact(run.run_id, "answer.txt")[0] for run in first} == {
        b"base",
        b"candidate",
    }
    store.close()
    reopened = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    resumed = TrialRunner(reopened, "exp", execute, output_paths=("answer.txt",)).run(cases)
    assert {run.attempt_id for run in resumed} == {run.attempt_id for run in first}
    assert len(seen) == 2
    reopened.close()


def test_failed_attempt_is_visible_and_retry_reuses_trial(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    snapshot = store.capture_workspace(source)
    case = TrialCase("task", "base", snapshot.snapshot_id, {}, 123)
    attempts: list[str] = []

    def execute(case: TrialCase, workspace: Path, attempt_id: str, context) -> None:
        attempts.append(attempt_id)
        if len(attempts) == 1:
            raise RuntimeError("agent failed")

    runner = TrialRunner(store, "exp", execute, max_attempts=2)
    result = runner.run([case])[0]
    history = store.runs_for_trial(result.trial_id)
    assert [run.status for run in history] == ["failed", "completed"]
    assert history[1].retry_of_attempt_id == history[0].attempt_id
    assert history[1].attempt_id != history[0].attempt_id
    assert len({run.trial_id for run in history}) == 1
    store.close()


def test_simultaneous_trial_ownership_is_rejected(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    other = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    assert store.claim_trial("trial", "owner-a")
    assert not other.claim_trial("trial", "owner-b")
    with pytest.raises(ValueError, match="owner"):
        other.release_trial("trial", "owner-b")
    store.release_trial("trial", "owner-a")
    assert other.claim_trial("trial", "owner-b")
    other.release_trial("trial", "owner-b")
    store.close()
    other.close()


def test_bounded_concurrency(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    snapshot = store.capture_workspace(source)
    cases = [TrialCase(f"task-{i}", "base", snapshot.snapshot_id, {}) for i in range(5)]
    active = 0
    peak = 0
    started = 0
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def execute(case: TrialCase, workspace: Path, attempt_id: str, context) -> None:
        nonlocal active, peak, started
        with lock:
            active += 1
            started += 1
            peak = max(peak, active)
            wait_for_peer = started <= 2
        if wait_for_peer:
            barrier.wait(timeout=5)
        with lock:
            active -= 1

    TrialRunner(store, "exp", execute, max_workers=2).run(cases)
    assert peak == 2
    store.close()


def test_parallel_repetitions_share_one_spec(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    snapshot = store.capture_workspace(source)
    cases = [TrialCase("task", "base", snapshot.snapshot_id, {}, seed=i) for i in range(4)]
    results = TrialRunner(store, "exp", lambda *_: None, max_workers=4).run(cases)
    assert len({run.trial_id for run in results}) == 4
    assert len({run.experiment_id for run in results}) == 1
    store.close()


def test_stale_owner_recovery_marks_interrupted_attempt(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    snapshot = store.capture_workspace(source)
    case = TrialCase("task", "base", snapshot.snapshot_id, {})
    runner = TrialRunner(store, "exp", lambda *_: None, max_attempts=2)
    spec_id, trial_id = runner._ids(case)
    from tbcore.experiment import ExperimentSpec, RunRecord

    store.add_spec(
        ExperimentSpec(
            spec_id,
            "task",
            {
                "experiment_group": "exp",
                "configuration_id": "base",
                "dataset_version": "default",
                "configuration": {},
            },
            snapshot.snapshot_id,
        )
    )
    pending = RunRecord(
        spec_id,
        "old-run",
        trial_id,
        "old-attempt",
        "pending",
        "2026-10-14T12:00:00Z",
        workspace_ref=snapshot.snapshot_id,
    )
    store.save_run(pending)
    store.save_run(pending.transition("running", at="2026-10-14T12:00:01Z"))
    store.claim_trial(trial_id, "dead-owner")
    store.register_trial_container(trial_id, "dead-owner", "tbcore-sess-orphan")
    store.close()
    monkeypatch.setattr("tbcore.experiment_store._pid_alive", lambda _pid: False)
    reopened = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    cleaned = []
    result = TrialRunner(
        reopened, "exp", lambda *_: None, max_attempts=2, cleanup_container=cleaned.append
    ).run([case])[0]
    history = reopened.runs_for_trial(trial_id)
    assert [attempt.status for attempt in history] == ["interrupted", "completed"]
    assert result.retry_of_attempt_id == "old-attempt"
    assert cleaned == ["tbcore-sess-orphan"]
    assert reopened.trial_containers(trial_id) == []
    reopened.close()


def test_cancellation_during_execution_keeps_attempt_visible(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    snapshot = store.capture_workspace(source)
    cancelled = threading.Event()

    def execute(case: TrialCase, workspace: Path, attempt_id: str, context) -> None:
        cancelled.set()

    result = TrialRunner(store, "exp", execute, cancelled=cancelled).run(
        [TrialCase("task", "base", snapshot.snapshot_id, {})]
    )[0]
    assert result.status == "cancelled"
    assert len(store.runs_for_trial(result.trial_id)) == 1
    store.close()


def test_cancellation_during_evidence_capture_keeps_partial_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    snapshot = store.capture_workspace(source)
    cancelled = threading.Event()
    original_add = store.add_artifact

    def add_then_cancel(run_id: str, name: str, content: bytes) -> str:
        digest = original_add(run_id, name, content)
        cancelled.set()
        return digest

    monkeypatch.setattr(store, "add_artifact", add_then_cancel)

    def execute(case: TrialCase, workspace: Path, attempt_id: str, context) -> None:
        (workspace / "one.txt").write_text("one", encoding="utf-8")
        (workspace / "two.txt").write_text("two", encoding="utf-8")

    result = TrialRunner(
        store, "exp", execute, output_paths=("one.txt", "two.txt"), cancelled=cancelled
    ).run([TrialCase("task", "base", snapshot.snapshot_id, {})])[0]
    assert result.status == "cancelled"
    assert result.artifact_ids == ("one.txt",)
    store.close()


def test_sandbox_executor_copies_final_output_before_cleanup(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer.txt").write_text("before", encoding="utf-8")
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    snapshot = store.capture_workspace(source)
    actions = []

    class Session:
        container = "tbcore-sess-fixture"

        def put_files(self, files):
            assert files == {"answer.txt": b"before"}

        def exec(self, command):
            assert command == ("agent",)
            return type(
                "Result",
                (),
                {
                    "exit_code": 0,
                    "timed_out": False,
                    "command": command,
                    "stdout": "",
                    "stderr": "",
                    "duration_s": 0.1,
                },
            )()

        def copy_out(self, path, destination):
            actions.append("capture")
            destination.mkdir()
            (destination / "answer.txt").write_text("after", encoding="utf-8")

        def close(self):
            actions.append("close")

    class Box:
        def __init__(self, image, **kwargs):
            assert kwargs["allow_hosts"] == ()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def session(self):
            return Session()

    monkeypatch.setattr("tbcore.trials.Sandbox", Box)
    executor = SandboxTrialExecutor(store, "agent-image", ("agent",))
    result = TrialRunner(store, "exp", executor, output_paths=("answer.txt",)).run(
        [TrialCase("task", "base", snapshot.snapshot_id, {})]
    )[0]
    assert result.status == "completed"
    assert store._verified_artifact(result.run_id, "answer.txt")[0] == b"after"
    assert actions == ["capture", "close"]
    store.close()
