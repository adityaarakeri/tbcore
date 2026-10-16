import shutil
from pathlib import Path

import pytest
from examples import experiment_workflow

from tbcore import CapturePolicy, ExperimentStore, SandboxError


def test_documented_example_and_disposable_backup_recovery(tmp_path: Path, monkeypatch):
    class FakeExecutor:
        def __init__(self, store, image, command):
            pass

        def __call__(self, case, workspace, attempt_id, context):
            (workspace / "answer.txt").write_text(case.configuration["answer"], encoding="utf-8")

    monkeypatch.setattr(experiment_workflow, "SandboxTrialExecutor", FakeExecutor)
    monkeypatch.setattr(
        experiment_workflow.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"stdout": "sha256:" + "a" * 64})(),
    )
    root = tmp_path / "store"
    result = experiment_workflow.run(root, "fixture-image")
    assert result["candidate_success_rate"] == 1.0
    assert result["candidate_cost_denominator"] == 0
    assert Path(result["comparison"]).is_file()
    assert Path(result["bundle"]).is_file()
    backup = tmp_path / "backup"
    shutil.copytree(root, backup)
    recovered = ExperimentStore(backup, CapturePolicy(content_enabled=True))
    run = recovered.get_run(result["candidate_run"])
    recovered.restore_workspace(run.workspace_ref, tmp_path / "restored")
    assert (tmp_path / "restored" / "task.txt").read_text(
        encoding="utf-8"
    ) == "Write the target answer."
    recovered.close()


def test_main_exits_cleanly_when_docker_is_unavailable(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(experiment_workflow, "docker_available", lambda: False)
    monkeypatch.setattr("sys.argv", ["experiment_workflow.py", "--store", str(tmp_path)])
    with pytest.raises(SystemExit) as exited:
        experiment_workflow.main()
    assert exited.value.code == 1
    assert "Docker is not available" in capsys.readouterr().err


def test_main_reports_sandbox_errors_without_traceback(tmp_path: Path, monkeypatch, capsys):
    def fail(store_root, image):
        raise SandboxError("docker build failed for tbcore/sandbox:latest")

    monkeypatch.setattr(experiment_workflow, "docker_available", lambda: True)
    monkeypatch.setattr(experiment_workflow, "run", fail)
    monkeypatch.setattr("sys.argv", ["experiment_workflow.py", "--store", str(tmp_path)])
    with pytest.raises(SystemExit) as exited:
        experiment_workflow.main()
    assert exited.value.code == 1
    assert "error: docker build failed" in capsys.readouterr().err
