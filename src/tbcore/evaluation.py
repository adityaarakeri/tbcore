"""Host-owned, versioned checks over immutable run artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath

from tbcore.experiment_store import ExperimentStore
from tbcore.sandbox import Sandbox, SandboxError


def _artifact_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.name != name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"invalid artifact path: {name!r}")


@dataclass(frozen=True)
class EvaluationSpec:
    evaluator_id: str
    version: str
    expected_text: dict[str, str] = field(default_factory=dict)
    command: tuple[str, ...] = ()
    image: str | None = None

    def __post_init__(self) -> None:
        if not self.evaluator_id or not self.version:
            raise ValueError("evaluator identity and version are required")
        for name, expected in self.expected_text.items():
            _artifact_name(name)
            if not isinstance(expected, str):
                raise ValueError("expected artifact content must be text")
        if bool(self.command) != bool(self.image):
            raise ValueError("command grader requires an image and command")
        if not self.expected_text and not self.command:
            raise ValueError("evaluator needs at least one check")
        if self.image is not None and not re.search(r"(?:^|@)sha256:[0-9a-f]{64}$", self.image):
            raise ValueError("command grader image must use an immutable digest")


@dataclass(frozen=True)
class EvaluationResult:
    evaluation_id: str
    evaluator_id: str
    version: str
    spec_hash: str
    run_id: str
    verdict: str
    components: dict[str, bool]
    input_hashes: dict[str, str]
    duration_s: float
    evidence: dict[str, str]


class Evaluator:
    """Compare stored artifacts against grading rules supplied by the host."""

    def __init__(self, store: ExperimentStore) -> None:
        self.store = store

    def evaluate(self, run_id: str, spec: EvaluationSpec) -> EvaluationResult:
        started = time.perf_counter()
        run = self.store.get_run(run_id)
        spec_payload = json.dumps(
            {
                "id": spec.evaluator_id,
                "version": spec.version,
                "expected_text": spec.expected_text,
                "command": spec.command,
                "image": spec.image,
            },
            sort_keys=True,
        )
        spec_hash = hashlib.sha256(spec_payload.encode("utf-8")).hexdigest()
        components: dict[str, bool] = {}
        hashes = {name: self.store._verified_artifact(run_id, name)[1] for name in run.artifact_ids}
        evidence: dict[str, str] = {}
        for name, expected in spec.expected_text.items():
            if name not in run.artifact_ids:
                components[name] = False
                evidence[name] = "missing artifact"
                continue
            content, digest = self.store._verified_artifact(run_id, name)
            hashes[name] = digest
            components[name] = content == expected.encode("utf-8")
            evidence[name] = "matched" if components[name] else "content mismatch"
        verdict = "pass" if all(components.values()) else "fail"
        if spec.command and spec.image is not None:
            files = {
                name: self.store._verified_artifact(run_id, name)[0] for name in run.artifact_ids
            }
            try:
                with Sandbox(
                    spec.image, allow_hosts=(), inject_headers=None, read_only=True
                ) as sandbox:
                    command_result = sandbox.run(spec.command, files=files)
            except SandboxError as exc:
                verdict = "error"
                evidence["command"] = f"sandbox error: {exc}"
            else:
                if command_result.timed_out:
                    verdict = "error"
                    evidence["command"] = "timed out"
                elif command_result.exit_code != 0:
                    verdict = "fail"
                    evidence["command"] = f"exit code {command_result.exit_code}"
                else:
                    components["command"] = True
                    evidence["command"] = "passed"
        result = EvaluationResult(
            uuid.uuid4().hex,
            spec.evaluator_id,
            spec.version,
            spec_hash,
            run_id,
            verdict,
            components,
            hashes,
            time.perf_counter() - started,
            evidence,
        )
        self.store.append_event(
            run_id,
            result.evaluation_id,
            {
                "event_type": "evaluation",
                "evaluator_id": result.evaluator_id,
                "version": result.version,
                "verdict": result.verdict,
                "spec_hash": result.spec_hash,
                "grader_image": spec.image,
                "components": result.components,
                "input_hashes": result.input_hashes,
                "duration_s": result.duration_s,
                "evidence": result.evidence,
            },
        )
        self.store.save_run(
            replace(run, evaluation_ids=(*run.evaluation_ids, result.evaluation_id))
        )
        return result
