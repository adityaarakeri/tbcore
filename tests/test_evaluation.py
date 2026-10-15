from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tbcore.evaluation import EvaluationSpec, Evaluator
from tbcore.evidence import CapturePolicy
from tbcore.experiment import ExperimentSpec, RunRecord
from tbcore.experiment_store import ExperimentStore
from tbcore.sandbox import SandboxError


def stored_run(tmp_path: Path, content: bytes = b"correct") -> tuple[ExperimentStore, str]:
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    store.add_spec(ExperimentSpec("exp", "task", {}))
    run = RunRecord("exp", "run", "trial", "attempt", "pending", "2026-10-14T12:00:00Z")
    store.save_run(run)
    store.add_artifact("run", "answer.txt", content)
    store.save_run(run.transition("running", at="2026-10-14T12:00:01Z"))
    store.save_run(
        replace(
            store.get_run("run").transition("completed", at="2026-10-14T12:00:02Z"),
            artifact_ids=("answer.txt",),
        )
    )
    return store, "run"


def test_exact_verdict_is_versioned_and_re_evaluation_does_not_rerun_agent(tmp_path: Path) -> None:
    store, run_id = stored_run(tmp_path)
    evaluator = Evaluator(store)
    spec = EvaluationSpec("answer", "v1", expected_text={"answer.txt": "correct"})
    first = evaluator.evaluate(run_id, spec)
    assert first.verdict == "pass"
    assert first.input_hashes["answer.txt"] == store._verified_artifact(run_id, "answer.txt")[1]
    second = evaluator.evaluate(
        run_id, EvaluationSpec("answer", "v2", expected_text={"answer.txt": "wrong"})
    )
    assert second.verdict == "fail"
    assert len(store.get_run(run_id).evaluation_ids) == 2
    assert store.get_run(run_id).status == "completed"
    assert {event["fields"]["version"] for event in store.events(run_id)} == {"v1", "v2"}
    store.close()


def test_missing_output_and_malicious_paths(tmp_path: Path) -> None:
    store, run_id = stored_run(tmp_path)
    result = Evaluator(store).evaluate(
        run_id, EvaluationSpec("answer", "v1", expected_text={"missing.txt": "x"})
    )
    assert result.verdict == "fail"
    assert result.components["missing.txt"] is False
    with pytest.raises(ValueError, match="artifact path"):
        EvaluationSpec("answer", "v1", expected_text={"../escape": "x"})
    store.close()


def test_agent_artifact_cannot_change_host_grading_rules(tmp_path: Path) -> None:
    store, run_id = stored_run(tmp_path, b'{"expected_text": {"answer.txt": "wrong"}}')
    result = Evaluator(store).evaluate(
        run_id, EvaluationSpec("answer", "v1", expected_text={"answer.txt": "correct"})
    )
    assert result.verdict == "fail"
    store.close()


def test_command_grader_uses_keyless_isolated_sandbox_and_records_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    store, run_id = stored_run(tmp_path)
    calls = []

    class FakeSandbox:
        def __init__(self, image, **kwargs):
            calls.append((image, kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def run(self, command, files):
            assert files == {"answer.txt": b"correct"}
            assert command == ("python", "grade.py")
            return type(
                "Result", (), {"exit_code": None, "timed_out": True, "stderr": "", "stdout": ""}
            )()

    monkeypatch.setattr("tbcore.evaluation.Sandbox", FakeSandbox)
    spec = EvaluationSpec("tests", "v1", command=("python", "grade.py"), image="sha256:" + "a" * 64)
    result = Evaluator(store).evaluate(run_id, spec)
    assert result.verdict == "error"
    assert result.evidence["command"] == "timed out"
    assert calls[0][1]["allow_hosts"] == ()
    assert calls[0][1]["inject_headers"] is None
    assert calls[0][1]["read_only"] is True
    assert store.get_run(run_id).status == "completed"
    store.close()


def test_grader_crash_is_evaluation_error_not_execution_failure(
    tmp_path: Path, monkeypatch
) -> None:
    store, run_id = stored_run(tmp_path)

    class CrashedSandbox:
        def __init__(self, image, **kwargs):
            pass

        def __enter__(self):
            raise SandboxError("grader unavailable")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("tbcore.evaluation.Sandbox", CrashedSandbox)
    result = Evaluator(store).evaluate(
        run_id, EvaluationSpec("tests", "v1", command=("grade",), image="sha256:" + "b" * 64)
    )
    assert result.verdict == "error"
    assert store.get_run(run_id).status == "completed"
    store.close()
