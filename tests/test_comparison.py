from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tbcore.comparison import compare_experiments
from tbcore.evidence import CapturePolicy
from tbcore.experiment import ExperimentSpec, RunRecord
from tbcore.experiment_store import ExperimentStore


def add_trial(store, config, task, verdict, *, seed=1, status="completed", usage=True):
    spec_id = f"{config}-{task}-{seed}"
    run_id = f"run-{spec_id}"
    store.add_spec(
        ExperimentSpec(
            spec_id,
            task,
            {
                "experiment_group": "group",
                "configuration_id": config,
                "dataset_version": "dataset-v1",
            },
        )
    )
    run = RunRecord(
        spec_id,
        run_id,
        f"trial-{spec_id}",
        f"attempt-{spec_id}",
        "pending",
        "2026-10-14T12:00:00Z",
        configuration={"seed": seed, "configuration_id": config},
        usage_coverage="complete" if usage else "partial",
        cost_coverage="complete" if usage else "partial",
    )
    store.save_run(run)
    store.save_run(run.transition("running", at="2026-10-14T12:00:01Z"))
    store.save_run(
        run.transition(status, at="2026-10-14T12:00:03Z")
        if status != "completed"
        else run.transition("running", at="2026-10-14T12:00:01Z").transition(
            "completed", at="2026-10-14T12:00:03Z"
        )
    )
    if status == "completed":
        store.append_event(
            run_id,
            f"eval-{run_id}",
            {
                "event_type": "evaluation",
                "evaluator_id": "grade",
                "version": "v1",
                "verdict": verdict,
            },
        )
        store.save_run(replace(store.get_run(run_id), evaluation_ids=(f"eval-{run_id}",)))
    if usage:
        store.append_event(
            run_id,
            f"call-{run_id}",
            {
                "event_type": "model_call",
                "attempt_id": run.attempt_id,
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        )
        store.append_event(
            run_id,
            f"cost:call-{run_id}",
            {
                "event_type": "cost_estimate",
                "priced_event_id": f"call-{run_id}",
                "attempt_id": run.attempt_id,
                "estimated_cost": "0.25",
            },
        )
    return run_id


def test_comparison_counts_trials_not_retries_and_exports_provenance(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    baseline = add_trial(store, "base", "a", "pass")
    failed_retry = RunRecord(
        "base-a-1",
        "failed-retry",
        "trial-base-a-1",
        "failed-attempt",
        "pending",
        "2026-10-14T12:00:04Z",
    )
    store.save_run(failed_retry)
    store.save_run(failed_retry.transition("failed", at="2026-10-14T12:00:05Z"))
    add_trial(store, "base", "b", "fail")
    add_trial(store, "candidate", "a", "pass")
    add_trial(store, "candidate", "b", "pass", usage=False)
    report = compare_experiments(store, "group", "base", "candidate", "dataset-v1", "grade", "v1")
    assert report.baseline.success_rate == 0.5
    assert report.baseline.success_ci_95 is not None
    assert report.baseline.success_ci_95[0] < 0.5 < report.baseline.success_ci_95[1]
    assert report.candidate.success_rate == 1.0
    assert report.baseline.denominators["quality"] == 2
    assert len(report.baseline.trials[0].attempt_ids) == 2
    assert report.candidate.denominators["tokens"] == 1
    assert report.paired_count == 2
    assert baseline in report.to_json()
    assert "Incomplete telemetry" in report.to_markdown()
    store.close()


def test_missing_pair_and_failed_attempt_are_explicit(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    add_trial(store, "base", "a", "pass")
    add_trial(store, "base", "b", "pass")
    add_trial(store, "candidate", "a", "pass", status="failed")
    report = compare_experiments(store, "group", "base", "candidate", "dataset-v1", "grade", "v1")
    assert report.paired_count == 1
    assert report.candidate.denominators["quality"] == 0
    assert report.candidate.exclusions["execution_failure"] == 1
    assert report.missing_candidate == ("b:1",)
    store.close()


def test_incompatible_dataset_or_evaluator_is_rejected(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    add_trial(store, "base", "a", "pass")
    add_trial(store, "candidate", "a", "pass")
    with pytest.raises(ValueError, match="dataset"):
        compare_experiments(store, "group", "base", "candidate", "other", "grade", "v1")
    with pytest.raises(ValueError, match="evaluator"):
        compare_experiments(store, "group", "base", "candidate", "dataset-v1", "grade", "v2")
    store.close()
