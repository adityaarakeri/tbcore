import json

from tbcore.evidence import CapturePolicy
from tbcore.experiment import ExperimentSpec, RunRecord
from tbcore.experiment_store import ExperimentStore
from tbcore.sandbox import RunResult
from tbcore.timeline import ExecutionTimeline


def _store(tmp_path, *, content=False):
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=content))
    store.add_spec(ExperimentSpec("exp-1", "task-1", {"model": "local"}))
    store.save_run(
        RunRecord(
            experiment_id="exp-1",
            run_id="run-1",
            trial_id="trial-1",
            attempt_id="attempt-1",
            status="pending",
            created_at="2026-10-14T12:00:00Z",
        )
    )
    return store


def _result(stdout="", *, timed_out=False):
    return RunResult(
        ("codex", "exec", "--json", "task"),
        None if timed_out else 0,
        stdout,
        "",
        timed_out,
        1.25,
    )


def test_outer_command_and_two_codex_commands_share_timeline(tmp_path):
    store = _store(tmp_path, content=True)
    lines = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {
            "type": "item.started",
            "item": {"id": "a", "type": "command_execution", "command": "pwd"},
        },
        {
            "type": "item.completed",
            "item": {
                "id": "a",
                "type": "command_execution",
                "command": "pwd",
                "exit_code": 0,
                "aggregated_output": "/work\n",
            },
        },
        {"type": "item.started", "item": {"id": "b", "type": "command_execution", "command": "ls"}},
        {
            "type": "item.completed",
            "item": {
                "id": "b",
                "type": "command_execution",
                "command": "ls",
                "exit_code": 2,
                "aggregated_output": "missing\n",
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 10}},
    ]
    result = _result("\n".join(json.dumps(line) for line in lines) + "\n")
    timeline = ExecutionTimeline(store, "run-1")
    coverage = timeline.record_codex_exec("outer-1", result, cli_version="0.1.2")
    events = store.events("run-1")
    assert coverage.complete is True
    assert coverage.tool_calls == 2
    assert [event["event_id"] for event in events] == [
        "outer-1",
        "outer-1.a",
        "outer-1.b",
        "outer-1.coverage",
    ]
    assert events[0]["fields"]["duration_ms"] == 1250
    assert events[0]["fields"]["command"] == ["codex", "exec", "--json", "task"]
    assert events[1]["fields"]["parent_event_id"] == "outer-1"
    assert events[1]["fields"]["exit_code"] == 0
    assert events[2]["fields"]["exit_code"] == 2
    assert events[2]["fields"]["agent_version"] == "0.1.2"
    assert events[1]["fields"]["output_ref"] == "outer-1.a.output.txt"
    assert (store.artifacts.root / "run-1" / "outer-1.a.output.txt").read_bytes() == b"/work\n"


def test_coverage_gaps_are_explicit_and_duplicate_delivery_is_idempotent(tmp_path):
    store = _store(tmp_path)
    lines = [
        {
            "type": "item.started",
            "item": {"id": "a", "type": "command_execution", "command": "pwd"},
        },
        {
            "type": "item.completed",
            "item": {"id": "a", "type": "command_execution", "command": "pwd", "exit_code": 0},
        },
        {
            "type": "item.completed",
            "item": {"id": "a", "type": "command_execution", "command": "pwd", "exit_code": 0},
        },
        {"type": "future.event"},
    ]
    stdout = "\n".join(json.dumps(line) for line in lines) + '\n{bad\n{"type":"item.started"'
    coverage = ExecutionTimeline(store, "run-1").record_codex_exec(
        "outer-1", _result(stdout), cli_version="0.1.2"
    )
    assert coverage.complete is False
    assert coverage.tool_calls == 1
    assert set(coverage.gaps) >= {"unknown-event", "malformed-json", "truncated-json"}
    assert len(store.events("run-1")) == 3
    assert store.events("run-1")[1]["fields"]["command"] == {"omitted": "capture-disabled"}
    assert store.events("run-1")[-1]["fields"]["coverage_complete"] is False


def test_incomplete_tool_and_timeout_are_not_reported_complete(tmp_path):
    store = _store(tmp_path)
    stdout = (
        json.dumps(
            {
                "type": "item.started",
                "item": {"id": "pending", "type": "command_execution", "command": "sleep 30"},
            }
        )
        + "\n"
    )
    coverage = ExecutionTimeline(store, "run-1").record_codex_exec(
        "outer-1", _result(stdout, timed_out=True), cli_version="0.1.2", cancelled=True
    )
    assert coverage.complete is False
    assert "unfinished-tool" in coverage.gaps
    outer = store.events("run-1")[0]["fields"]
    assert outer["timed_out"] is True
    assert outer["cancelled"] is True
    assert outer["exit_code"] is None


def test_session_command_parent_and_bounded_output(tmp_path):
    store = ExperimentStore(
        tmp_path / "store",
        CapturePolicy(content_enabled=True, max_artifact_bytes=8),
    )
    store.add_spec(ExperimentSpec("exp-1", "task-1", {"model": "local"}))
    store.save_run(
        RunRecord(
            experiment_id="exp-1",
            run_id="run-1",
            trial_id="trial-1",
            attempt_id="attempt-1",
            status="pending",
            created_at="2026-10-14T12:00:00Z",
        )
    )
    result = RunResult(("sh", "-c", "echo hello"), 0, "output too long", "", False, 0.5)
    ExecutionTimeline(store, "run-1").record_command(
        "session-1", result, parent_event_id="attempt-1"
    )
    fields = store.events("run-1")[0]["fields"]
    assert fields["parent_event_id"] == "attempt-1"
    assert fields["duration_ms"] == 500
    assert fields["output_ref"] is None
    assert not list((store.artifacts.root / "run-1").glob("*.txt"))
