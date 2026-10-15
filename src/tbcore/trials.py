"""Durable, bounded execution of repeatable workspace trials."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from tbcore.experiment import ExperimentSpec, RunRecord
from tbcore.experiment_store import ExperimentStore
from tbcore.sandbox import Sandbox
from tbcore.timeline import ExecutionTimeline


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _identifier(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrialCase:
    task_id: str
    configuration_id: str
    workspace_ref: str
    configuration: dict[str, object]
    seed: int | None = None
    environment_ref: str | None = None
    dataset_version: str = "default"

    def __post_init__(self) -> None:
        if (
            not self.task_id
            or not self.configuration_id
            or not self.workspace_ref
            or not self.dataset_version
        ):
            raise ValueError("trial task, configuration, and workspace IDs are required")
        _identifier(self.configuration)


class TrialOwnedError(RuntimeError):
    """Another live process owns a trial."""


@dataclass(frozen=True)
class TrialContext:
    store: ExperimentStore
    trial_id: str
    owner_id: str
    cancelled: threading.Event

    def register_container(self, name: str) -> None:
        self.store.register_trial_container(self.trial_id, self.owner_id, name)

    def forget_container(self, name: str) -> None:
        self.store.forget_trial_container(self.trial_id, name)


class SandboxTrialExecutor:
    """Run a captured workspace in one owned sandbox session per attempt."""

    def __init__(
        self,
        store: ExperimentStore,
        image: str,
        command: Sequence[str],
        *,
        inject_headers: Mapping[str, Mapping[str, str]] | None = None,
        proxy_budget: Mapping[str, object] | None = None,
        cli_version: str | None = None,
    ) -> None:
        if not image or not command:
            raise ValueError("sandbox image and command are required")
        if "codex" in command[0] and "exec" in command and "--json" in command and not cli_version:
            raise ValueError("Codex JSON execution requires a CLI version")
        self.store = store
        self.image = image
        self.command = tuple(command)
        self.inject_headers = inject_headers
        self.proxy_budget = proxy_budget
        self.cli_version = cli_version

    def __call__(
        self, case: TrialCase, workspace: Path, attempt_id: str, context: TrialContext
    ) -> None:
        files: dict[str, bytes] = {}
        for path in workspace.rglob("*"):
            if path.is_symlink():
                raise ValueError("workspace contains a symlink")
            if path.is_file():
                files[path.relative_to(workspace).as_posix()] = path.read_bytes()
        with Sandbox(
            self.image,
            allow_hosts=(),
            inject_headers=self.inject_headers,
            telemetry_run_id=attempt_id if self.inject_headers else None,
            telemetry_attempt_id=attempt_id if self.inject_headers else None,
            proxy_budget=self.proxy_budget,
        ) as sandbox:
            session = sandbox.session()
            context.register_container(session.container)
            try:
                session.put_files(files)
                result = session.exec(self.command)
                timeline = ExecutionTimeline(self.store, attempt_id)
                if (
                    self.cli_version is not None
                    and "exec" in self.command
                    and "--json" in self.command
                ):
                    timeline.record_codex_exec(
                        f"command:{attempt_id}",
                        result,
                        cli_version=self.cli_version,
                        cancelled=context.cancelled.is_set(),
                    )
                else:
                    timeline.record_command(
                        f"command:{attempt_id}", result, cancelled=context.cancelled.is_set()
                    )
                shutil.rmtree(workspace)
                session.copy_out(".", workspace)
            finally:
                session.close()
                context.forget_container(session.container)
            if self.inject_headers:
                for observation in sandbox.proxy_usage():
                    self.store.ingest_proxy_usage(observation)
            if result.timed_out:
                raise TimeoutError("sandbox agent command timed out")
            if result.exit_code != 0:
                raise RuntimeError(f"sandbox agent command exited {result.exit_code}")


def _remove_container(name: str) -> None:
    result = subprocess.run(
        ["docker", "rm", "-f", "-v", name], capture_output=True, text=True, check=False
    )
    if result.returncode != 0 and "No such container" not in result.stderr:
        raise RuntimeError(f"cannot remove trial container {name}: {result.stderr}")


class TrialRunner:
    """Restore each trial from immutable inputs and retain every attempt."""

    def __init__(
        self,
        store: ExperimentStore,
        experiment_id: str,
        execute: Callable[[TrialCase, Path, str, TrialContext], None],
        *,
        output_paths: Sequence[str] = (),
        max_workers: int = 1,
        max_attempts: int = 1,
        cancelled: threading.Event | None = None,
        cleanup_container: Callable[[str], None] = _remove_container,
    ) -> None:
        if not experiment_id:
            raise ValueError("experiment_id must be nonempty")
        if max_workers < 1 or max_attempts < 1:
            raise ValueError("worker and attempt limits must be positive")
        for path in output_paths:
            relative = PurePosixPath(path)
            if (
                not path
                or relative.is_absolute()
                or ".." in relative.parts
                or "\\" in path
                or relative.as_posix() in {".", ""}
            ):
                raise ValueError(f"unsafe output path: {path!r}")
        self.store = store
        self.experiment_id = experiment_id
        self.execute = execute
        self.output_paths = tuple(output_paths)
        self.max_workers = max_workers
        self.max_attempts = max_attempts
        self.cancelled = cancelled or threading.Event()
        self.cleanup_container = cleanup_container

    def _ids(self, case: TrialCase) -> tuple[str, str]:
        base = (
            self.experiment_id,
            case.task_id,
            case.configuration_id,
            case.workspace_ref,
            case.environment_ref,
            case.dataset_version,
            case.configuration,
        )
        return _identifier(base), _identifier((base, case.seed))

    def run(self, cases: Sequence[TrialCase]) -> list[RunRecord]:
        ids = [self._ids(case)[1] for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate trial in matrix; provide distinct seeds")
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self._run_case, case) for case in cases]
            return [future.result() for future in futures]

    def _run_case(self, case: TrialCase) -> RunRecord:
        spec_id, trial_id = self._ids(case)
        owner = uuid.uuid4().hex
        if not self.store.claim_trial(trial_id, owner):
            raise TrialOwnedError(f"trial {trial_id} is owned by another process")
        try:
            self._cleanup_containers(trial_id)
            context = TrialContext(self.store, trial_id, owner, self.cancelled)
            history = self.store.runs_for_trial(trial_id)
            for run in history:
                if run.status == "running":
                    self.store.save_run(run.transition("interrupted", at=_now()))
            history = self.store.runs_for_trial(trial_id)
            completed = next((run for run in history if run.status == "completed"), None)
            if completed is not None:
                return completed
            spec = ExperimentSpec(
                spec_id,
                case.task_id,
                {
                    "experiment_group": self.experiment_id,
                    "configuration_id": case.configuration_id,
                    "dataset_version": case.dataset_version,
                    "configuration": case.configuration,
                },
                case.workspace_ref,
                case.environment_ref,
            )
            self.store.ensure_spec(spec)
            previous = history[-1] if history else None
            for _ in range(max(0, self.max_attempts - len(history))):
                attempt_id = uuid.uuid4().hex
                run = RunRecord(
                    spec_id,
                    attempt_id,
                    trial_id,
                    attempt_id,
                    "pending",
                    _now(),
                    retry_of_attempt_id=previous.attempt_id if previous else None,
                    configuration={
                        "configuration_id": case.configuration_id,
                        "seed": case.seed,
                        **case.configuration,
                    },
                    workspace_ref=case.workspace_ref,
                    environment_ref=case.environment_ref,
                )
                self.store.save_run(run)
                if self.cancelled.is_set():
                    result = run.transition("cancelled", at=_now())
                    self.store.save_run(result)
                    return result
                result = self._execute_attempt(case, run, context)
                self._cleanup_containers(trial_id)
                if result.status in {"completed", "cancelled"}:
                    return result
                previous = result
            if previous is None:
                raise RuntimeError("no trial attempt was created")
            return previous
        finally:
            self.store.release_trial(trial_id, owner)

    def _cleanup_containers(self, trial_id: str) -> None:
        for name in self.store.trial_containers(trial_id):
            self.cleanup_container(name)
            self.store.forget_trial_container(trial_id, name)

    def _execute_attempt(self, case: TrialCase, run: RunRecord, context: TrialContext) -> RunRecord:
        running = run.transition("running", at=_now())
        self.store.save_run(running)
        with tempfile.TemporaryDirectory(prefix="tbcore-trial-") as directory:
            workspace = Path(directory) / "work"
            try:
                self.store.restore_workspace(case.workspace_ref, workspace)
                self.execute(case, workspace, run.attempt_id, context)
                if self.cancelled.is_set():
                    result = running.transition("cancelled", at=_now())
                else:
                    artifacts: list[str] = []
                    for name in self.output_paths:
                        if self.cancelled.is_set():
                            break
                        path = workspace.joinpath(*PurePosixPath(name).parts)
                        if (
                            path.is_symlink()
                            or not path.is_file()
                            or not path.resolve().is_relative_to(workspace.resolve())
                        ):
                            raise ValueError(f"missing or unsafe output: {name}")
                        self.store.add_artifact(run.run_id, name, path.read_bytes())
                        artifacts.append(name)
                    status = "cancelled" if self.cancelled.is_set() else "completed"
                    result = replace(
                        running.transition(status, at=_now()), artifact_ids=tuple(artifacts)
                    )
            except Exception as exc:
                self.store.append_event(
                    run.run_id,
                    f"failure-{uuid.uuid4().hex}",
                    {
                        "event_type": "attempt_failure",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                status = "cancelled" if self.cancelled.is_set() else "failed"
                result = running.transition(status, at=_now())
        self.store.save_run(result)
        return result
