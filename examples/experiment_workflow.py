"""Offline Docker experiment: baseline/candidate edits, grading, comparison, replay."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import tbcore
from tbcore import (
    CapturePolicy,
    EnvironmentProvenance,
    EvaluationSpec,
    Evaluator,
    ExperimentStore,
    SandboxTrialExecutor,
    TrialCase,
    TrialRunner,
    build_default_images,
    compare_experiments,
)

AGENT = """from pathlib import Path
import json
configuration = json.loads(Path('config.json').read_text())
Path('answer.txt').write_text(configuration['answer'])
"""


def run(store_root: Path, image: str | None = None) -> dict[str, object]:
    if image is None:
        image = build_default_images()["sandbox"]
    inspected = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=True,
    )
    store = ExperimentStore(store_root, CapturePolicy(content_enabled=True))
    with tempfile.TemporaryDirectory(prefix="tbcore-example-") as directory:
        source = Path(directory)
        (source / "agent.py").write_text(AGENT, encoding="utf-8")
        (source / "task.txt").write_text("Write the target answer.", encoding="utf-8")
        snapshot = store.capture_workspace(source)
    environment = store.record_environment(
        EnvironmentProvenance(
            image_id=inspected.stdout.strip(),
            registry_digest=None,
            tbcore_version=tbcore.__version__,
            agent_cli_version="fixture-agent-1",
            model="none",
            provider="none",
            resource_limits={"memory": "512m", "cpus": "1.0"},
            runtime="runc",
            policy_config={"network": "none"},
        )
    )
    cases = [
        TrialCase(
            "edit-task",
            "baseline",
            snapshot.snapshot_id,
            {"answer": "before"},
            seed=1,
            environment_ref=environment,
            dataset_version="fixture-v1",
        ),
        TrialCase(
            "edit-task",
            "candidate",
            snapshot.snapshot_id,
            {"answer": "after"},
            seed=1,
            environment_ref=environment,
            dataset_version="fixture-v1",
        ),
    ]
    sandbox_executor = SandboxTrialExecutor(store, image, ("python", "agent.py"))

    def execute(case, workspace, attempt_id, context):
        (workspace / "config.json").write_text(json.dumps(case.configuration), encoding="utf-8")
        sandbox_executor(case, workspace, attempt_id, context)

    runs = TrialRunner(store, "example", execute, output_paths=("answer.txt",), max_workers=2).run(
        cases
    )
    grader = Evaluator(store)
    evaluation = EvaluationSpec("answer", "v1", expected_text={"answer.txt": "after"})
    for trial in runs:
        grader.evaluate(trial.run_id, evaluation)
    report = compare_experiments(
        store, "example", "baseline", "candidate", "fixture-v1", "answer", "v1"
    )
    candidate = runs[1]
    bundle = store.export_run(candidate.run_id)
    bundle_path = store_root / "candidate-bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    report_path = store_root / "comparison.md"
    report_path.write_text(report.to_markdown(), encoding="utf-8")
    store.close()

    replay_store = ExperimentStore(store_root / "replay", CapturePolicy(content_enabled=True))
    replay_store.import_bundle(bundle)
    replay_case = TrialCase(
        "edit-task",
        "candidate",
        snapshot.snapshot_id,
        {"answer": "after"},
        seed=2,
        environment_ref=environment,
        dataset_version="fixture-v1",
    )
    replay_executor = SandboxTrialExecutor(replay_store, image, ("python", "agent.py"))

    def replay_execute(case, workspace, attempt_id, context):
        (workspace / "config.json").write_text(json.dumps(case.configuration), encoding="utf-8")
        replay_executor(case, workspace, attempt_id, context)

    replay = TrialRunner(replay_store, "example", replay_execute, output_paths=("answer.txt",)).run(
        [replay_case]
    )[0]
    replay_store.close()
    return {
        "baseline_run": runs[0].run_id,
        "candidate_run": candidate.run_id,
        "replay_run": replay.run_id,
        "comparison": str(report_path),
        "bundle": str(bundle_path),
        "candidate_success_rate": report.candidate.success_rate,
        "candidate_cost_denominator": report.candidate.denominators["cost"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--image", help="already built sandbox image; default builds bundled images"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.store, args.image), indent=2))


if __name__ == "__main__":
    main()
