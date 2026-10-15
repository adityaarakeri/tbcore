import json

import pytest

from tbcore.experiment import ExperimentSpec, RunRecord


def test_spec_and_run_round_trip_with_linked_evidence():
    spec = ExperimentSpec(
        experiment_id="exp-1",
        task_id="repair-1",
        configuration={"model": "sonnet", "limits": {"seconds": 60}},
        workspace_ref="sha256:input",
        environment_ref="sha256:image",
    )
    run = RunRecord(
        experiment_id="exp-1",
        run_id="run-1",
        trial_id="trial-1",
        attempt_id="attempt-2",
        parent_attempt_id="attempt-1",
        retry_of_attempt_id="attempt-1",
        status="failed",
        created_at="2026-10-14T12:00:00Z",
        started_at="2026-10-14T12:00:01Z",
        finished_at="2026-10-14T12:00:03Z",
        configuration={"model": "sonnet"},
        workspace_ref="sha256:input",
        environment_ref="sha256:image",
        event_ids=("event-1", "event-2"),
        artifact_ids=("artifact-1",),
        model_call_ids=("call-1",),
        evaluation_ids=("eval-1",),
        usage_coverage="partial",
        cost_coverage="unknown",
    )

    assert ExperimentSpec.from_json(spec.to_json()) == spec
    assert RunRecord.from_json(run.to_json()) == run
    assert json.loads(run.to_json())["schema_version"] == 1
    assert run.trial_id != run.attempt_id


def test_run_preserves_missing_observations_and_explicit_terminal_statuses():
    for status in ("failed", "cancelled", "timed_out", "interrupted"):
        run = RunRecord(
            experiment_id="exp-1",
            run_id="run-1",
            trial_id="trial-1",
            attempt_id="attempt-1",
            status=status,
            created_at="2026-10-14T12:00:00Z",
            finished_at="2026-10-14T12:00:03Z",
        )
        restored = RunRecord.from_json(run.to_json())
        assert restored.status == status
        assert restored.model_call_ids == ()
        assert restored.usage_coverage == "unknown"
        assert restored.cost_coverage == "unknown"


def test_run_transitions_require_valid_path_and_timestamps():
    pending = RunRecord(
        experiment_id="exp-1",
        run_id="run-1",
        trial_id="trial-1",
        attempt_id="attempt-1",
        status="pending",
        created_at="2026-10-14T12:00:00Z",
    )
    running = pending.transition("running", at="2026-10-14T12:00:01Z")
    completed = running.transition("completed", at="2026-10-14T12:00:03Z")
    assert completed.started_at == "2026-10-14T12:00:01Z"
    assert completed.finished_at == "2026-10-14T12:00:03Z"
    with pytest.raises(ValueError, match="transition"):
        pending.transition("completed", at="2026-10-14T12:00:03Z")
    with pytest.raises(ValueError, match="transition"):
        completed.transition("running", at="2026-10-14T12:00:04Z")
    with pytest.raises(ValueError, match="timestamp"):
        pending.transition("running", at="2026-10-14T11:59:59Z")


@pytest.mark.parametrize("record_type", [ExperimentSpec, RunRecord])
def test_unsupported_schema_version_is_rejected(record_type):
    payload = {"schema_version": 99}
    with pytest.raises(ValueError, match="schema version"):
        record_type.from_json(json.dumps(payload))
