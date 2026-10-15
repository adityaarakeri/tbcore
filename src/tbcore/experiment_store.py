"""SQLite run metadata with ordered events and checked local artifacts."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import sqlite3
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from tbcore.evidence import CapturePolicy, EvidenceArtifacts
from tbcore.experiment import ExperimentSpec, RunRecord
from tbcore.workspace import EnvironmentProvenance, WorkspaceManifest, WorkspaceStore

STORE_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS specifications (
    experiment_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES specifications(experiment_id),
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    sequence INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, sequence),
    UNIQUE (run_id, event_id)
);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    byte_count INTEGER NOT NULL,
    PRIMARY KEY (run_id, name)
);
CREATE TABLE IF NOT EXISTS trial_claims (
    trial_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    owner_pid INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS trial_containers (
    trial_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    container_name TEXT NOT NULL,
    PRIMARY KEY (trial_id, container_name)
);
"""


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    # A sys.platform if/else (not os.name) lets mypy skip whichever branch does not apply to
    # the platform it is checking, instead of failing on ctypes.windll or unreachable code.
    if sys.platform == "win32":
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise OSError("cannot inspect trial owner process")
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


class ExperimentStore:
    """Operator-local store for one or more linked experiment runs.

    One instance may be shared by threads in a process: every use of its single SQLite
    connection, reads included, happens under ``_lock``. Use one writing process per root.
    """

    def __init__(
        self,
        root: str | Path,
        policy: CapturePolicy | None = None,
        *,
        allow_binary_workspaces: bool = False,
    ) -> None:
        self.root = Path(root)
        self.policy = policy or CapturePolicy()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = self.root / "experiments.sqlite3"
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        try:
            version_row = self.conn.execute("PRAGMA user_version").fetchone()
            if version_row is None:
                raise RuntimeError("cannot read store schema version")
            version = version_row[0]
            if version not in (0, STORE_SCHEMA_VERSION):
                raise ValueError(f"unsupported store schema version: {version}")
            self.artifacts = EvidenceArtifacts(self.root / "artifacts", self.policy)
            self.workspaces = WorkspaceStore(
                self.root / "workspaces", self.policy, allow_binary=allow_binary_workspaces
            )
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(_SCHEMA)
            if version == 0:
                self.conn.execute(f"PRAGMA user_version={STORE_SCHEMA_VERSION}")
            if os.name != "nt":
                self.path.chmod(0o600)
            self._reconcile_artifacts()
        except BaseException:
            self.conn.close()
            raise

    def _fetchone(self, sql: str, params: Sequence[object] = ()) -> Any:
        """Query and fetch under the lock; the connection is shared across threads."""
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def _fetchall(self, sql: str, params: Sequence[object] = ()) -> list[Any]:
        """Query and fetch every row under the lock; never hand out a live cursor."""
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def add_spec(self, spec: ExperimentSpec) -> None:
        self._check_provenance(spec.workspace_ref, spec.environment_ref)
        payload = spec.to_json()
        with self._lock:
            self.conn.execute(
                "INSERT INTO specifications(experiment_id, payload) VALUES (?, ?)",
                (spec.experiment_id, payload),
            )

    def ensure_spec(self, spec: ExperimentSpec) -> None:
        self._check_provenance(spec.workspace_ref, spec.environment_ref)
        payload = spec.to_json()
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO specifications(experiment_id, payload) VALUES (?, ?)",
                (spec.experiment_id, payload),
            )
            row = self.conn.execute(
                "SELECT payload FROM specifications WHERE experiment_id=?", (spec.experiment_id,)
            ).fetchone()
            if row is None or row[0] != payload:
                raise ValueError("trial specification changed")

    def get_spec(self, experiment_id: str) -> ExperimentSpec:
        row = self._fetchone(
            "SELECT payload FROM specifications WHERE experiment_id=?", (experiment_id,)
        )
        if row is None:
            raise KeyError(experiment_id)
        return ExperimentSpec.from_json(row[0])

    def save_run(self, run: RunRecord) -> None:
        self._check_provenance(run.workspace_ref, run.environment_ref)
        with self._lock:
            if (
                self.conn.execute(
                    "SELECT 1 FROM specifications WHERE experiment_id=?", (run.experiment_id,)
                ).fetchone()
                is None
            ):
                raise ValueError("run refers to an unknown experiment")
            previous = self.conn.execute(
                "SELECT payload FROM runs WHERE run_id=?", (run.run_id,)
            ).fetchone()
            if previous is not None:
                old = RunRecord.from_json(previous[0])
                if (
                    old.experiment_id != run.experiment_id
                    or old.trial_id != run.trial_id
                    or old.attempt_id != run.attempt_id
                ):
                    raise ValueError("run identity cannot change")
                if old.status != run.status:
                    at = run.started_at if run.status == "running" else run.finished_at
                    old.transition(run.status, at=at or "")
            for name in run.artifact_ids:
                self._verified_artifact(run.run_id, name)
            self.conn.execute(
                "INSERT INTO runs(run_id, experiment_id, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET payload=excluded.payload",
                (run.run_id, run.experiment_id, run.to_json()),
            )

    def _check_provenance(self, workspace_ref: str | None, environment_ref: str | None) -> None:
        if workspace_ref is not None:
            self.workspaces.get_manifest(workspace_ref)
        if environment_ref is not None:
            self.workspaces.get_environment(environment_ref)

    def capture_workspace(
        self, source: str | Path, *, exclude: tuple[str, ...] = ()
    ) -> WorkspaceManifest:
        return self.workspaces.capture(source, exclude=exclude)

    def restore_workspace(self, snapshot_id: str, destination: str | Path) -> None:
        self.workspaces.restore(snapshot_id, destination)

    def record_environment(self, provenance: EnvironmentProvenance) -> str:
        return self.workspaces.record_environment(provenance)

    def get_run(self, run_id: str) -> RunRecord:
        row = self._fetchone("SELECT payload FROM runs WHERE run_id=?", (run_id,))
        if row is None:
            raise KeyError(run_id)
        run = RunRecord.from_json(row[0])
        if run.status == "completed":
            for name in run.artifact_ids:
                self._verified_artifact(run_id, name)
        return run

    def runs_for_trial(self, trial_id: str) -> list[RunRecord]:
        rows = self._fetchall("SELECT payload FROM runs ORDER BY rowid")
        return [
            run for (payload,) in rows if (run := RunRecord.from_json(payload)).trial_id == trial_id
        ]

    def all_runs(self) -> list[RunRecord]:
        rows = self._fetchall("SELECT payload FROM runs ORDER BY rowid")
        return [RunRecord.from_json(payload) for (payload,) in rows]

    def claim_trial(self, trial_id: str, owner_id: str) -> bool:
        if not trial_id or not owner_id:
            raise ValueError("trial and owner IDs must be nonempty")
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = self.conn.execute(
                    "SELECT owner_id, owner_pid FROM trial_claims WHERE trial_id=?", (trial_id,)
                ).fetchone()
                if row is not None:
                    if not _pid_alive(int(row[1])):
                        self.conn.execute("DELETE FROM trial_claims WHERE trial_id=?", (trial_id,))
                    else:
                        self.conn.execute("COMMIT")
                        return False
                self.conn.execute(
                    "INSERT INTO trial_claims(trial_id, owner_id, owner_pid) VALUES (?, ?, ?)",
                    (trial_id, owner_id, os.getpid()),
                )
                self.conn.execute("COMMIT")
                return True
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    def release_trial(self, trial_id: str, owner_id: str) -> None:
        with self._lock:
            cursor = self.conn.execute(
                "DELETE FROM trial_claims WHERE trial_id=? AND owner_id=?", (trial_id, owner_id)
            )
            if cursor.rowcount != 1:
                raise ValueError("trial owner does not match")

    def register_trial_container(self, trial_id: str, owner_id: str, name: str) -> None:
        if not name.startswith(("tbcore-sess-", "tbcore-run-")) or not all(
            char.isalnum() or char == "-" for char in name
        ):
            raise ValueError("container name is not a tbcore workload")
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM trial_claims WHERE trial_id=? AND owner_id=?", (trial_id, owner_id)
            ).fetchone()
            if row is None:
                raise ValueError("trial owner does not match")
            self.conn.execute(
                "INSERT INTO trial_containers(trial_id, owner_id, container_name) VALUES (?, ?, ?)",
                (trial_id, owner_id, name),
            )

    def trial_containers(self, trial_id: str) -> list[str]:
        rows = self._fetchall(
            "SELECT container_name FROM trial_containers WHERE trial_id=?", (trial_id,)
        )
        return [str(row[0]) for row in rows]

    def forget_trial_container(self, trial_id: str, name: str) -> None:
        with self._lock:
            self.conn.execute(
                "DELETE FROM trial_containers WHERE trial_id=? AND container_name=?",
                (trial_id, name),
            )

    def append_event(self, run_id: str, event_id: str, fields: Mapping[str, object]) -> int:
        if not event_id or not isinstance(event_id, str):
            raise ValueError("event_id must be nonempty")
        payload = json.dumps(self.policy.capture(fields), sort_keys=True, allow_nan=False)
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                if (
                    self.conn.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
                    is None
                ):
                    raise KeyError(run_id)
                prior = self.conn.execute(
                    "SELECT sequence, payload FROM events WHERE run_id=? AND event_id=?",
                    (run_id, event_id),
                ).fetchone()
                if prior is not None:
                    if prior[1] != payload:
                        raise ValueError("duplicate event ID has different content")
                    self.conn.execute("COMMIT")
                    return int(prior[0])
                next_row = self.conn.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM events WHERE run_id=?", (run_id,)
                ).fetchone()
                if next_row is None:
                    raise RuntimeError("cannot allocate event sequence")
                sequence = next_row[0]
                self.conn.execute(
                    "INSERT INTO events(run_id, sequence, event_id, payload) VALUES (?, ?, ?, ?)",
                    (run_id, sequence, event_id, payload),
                )
                self.conn.execute("COMMIT")
                return int(sequence)
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    def ingest_proxy_usage(self, observation: Mapping[str, object]) -> int:
        """Accept host-collected proxy metadata; provider IDs deduplicate redelivery."""
        run_id = observation.get("run_id")
        attempt_id = observation.get("attempt_id")
        call_id = observation.get("call_id")
        if not all(isinstance(value, str) and value for value in (run_id, attempt_id, call_id)):
            raise ValueError("proxy observation requires run, attempt, and call IDs")
        assert isinstance(run_id, str) and isinstance(attempt_id, str) and isinstance(call_id, str)
        run = self.get_run(run_id)
        if run.attempt_id != attempt_id:
            raise ValueError("proxy observation attempt does not match the run")
        provider_id = observation.get("provider_request_id")
        provider = observation.get("provider")
        if provider_id is not None and not isinstance(provider_id, str):
            raise ValueError("provider_request_id must be a string or null")
        if provider is not None and provider not in ("openai", "anthropic"):
            raise ValueError("unsupported proxy provider")
        fields: dict[str, object] = {
            "event_type": "model_call",
            "call_source": "credential_proxy",
            "attempt_id": attempt_id,
            "provider": provider,
            "model": observation.get("model"),
            "provider_request_id": provider_id,
            "status": observation.get("status"),
            "usage_complete": observation.get("usage_complete"),
            "telemetry_reason": observation.get("telemetry_reason"),
        }
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            value = observation.get(name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"invalid {name} in proxy observation")
            fields[name] = value
        if provider_id:
            for event in self.events(run_id):
                prior = event["fields"]
                if (
                    prior.get("event_type") == "model_call"
                    and prior.get("provider") == provider
                    and prior.get("provider_request_id") == provider_id
                ):
                    for name in (
                        "input_tokens",
                        "output_tokens",
                        "cache_read_input_tokens",
                        "cache_creation_input_tokens",
                    ):
                        if (
                            prior.get(name) is not None
                            and fields[name] is not None
                            and prior.get(name) != fields[name]
                        ):
                            raise ValueError("duplicate provider request has conflicting usage")
                    if all(
                        prior.get(name) is not None
                        for name in (
                            "input_tokens",
                            "output_tokens",
                            "cache_read_input_tokens",
                            "cache_creation_input_tokens",
                        )
                    ):
                        return int(event["sequence"])
        return self.append_event(run_id, f"proxy:{call_id}", fields)

    def events(self, run_id: str) -> list[dict[str, Any]]:
        self.get_run(run_id)
        rows = self._fetchall(
            "SELECT sequence, event_id, payload FROM events WHERE run_id=? ORDER BY sequence",
            (run_id,),
        )
        return [
            {"sequence": row[0], "event_id": row[1], "fields": json.loads(row[2])} for row in rows
        ]

    def add_artifact(self, run_id: str, name: str, content: bytes) -> str:
        self.get_run(run_id)
        digest = self.artifacts.write(run_id, name, content)
        byte_count = (self.artifacts.root / run_id / name).stat().st_size
        self._commit_artifact_metadata(run_id, name, digest, byte_count)
        return digest

    def _commit_artifact_metadata(
        self, run_id: str, name: str, digest: str, byte_count: int
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO artifacts(run_id, name, sha256, byte_count) VALUES (?, ?, ?, ?)",
                (run_id, name, digest, byte_count),
            )

    def _verified_artifact(self, run_id: str, name: str) -> tuple[bytes, str]:
        row = self._fetchone(
            "SELECT sha256, byte_count FROM artifacts WHERE run_id=? AND name=?", (run_id, name)
        )
        if row is None:
            raise ValueError(f"artifact {name!r} is not recorded")
        path = self.artifacts.root / run_id / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact {name!r} is missing or unsafe")
        content = path.read_bytes()
        if len(content) != row[1] or hashlib.sha256(content).hexdigest() != row[0]:
            raise ValueError(f"artifact {name!r} has an integrity hash mismatch")
        return content, str(row[0])

    def _reconcile_artifacts(self) -> None:
        retained: dict[str, set[str]] = {}
        for run_id, name in self._fetchall("SELECT run_id, name FROM artifacts"):
            self._verified_artifact(run_id, name)
            retained.setdefault(run_id, set()).add(name)
        self.artifacts.cleanup(retained)

    def export_run(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        spec = self.get_spec(run.experiment_id)
        artifacts = []
        for name in run.artifact_ids:
            content, digest = self._verified_artifact(run_id, name)
            artifacts.append(
                {
                    "name": name,
                    "sha256": digest,
                    "content_base64": base64.b64encode(content).decode(),
                }
            )
        return {
            "schema_version": STORE_SCHEMA_VERSION,
            "specification": json.loads(spec.to_json()),
            "run": json.loads(run.to_json()),
            "events": self.events(run_id),
            "artifacts": artifacts,
            "workspaces": {
                ref: self.workspaces.export_snapshot(ref)
                for ref in {spec.workspace_ref, run.workspace_ref}
                if ref is not None
            },
            "environments": {
                ref: self.workspaces.export_environment(ref)
                for ref in {spec.environment_ref, run.environment_ref}
                if ref is not None
            },
        }

    def import_bundle(self, bundle: Mapping[str, Any]) -> None:
        if bundle.get("schema_version") != STORE_SCHEMA_VERSION:
            raise ValueError("unsupported store schema version")
        spec = ExperimentSpec.from_json(json.dumps(bundle["specification"]))
        run = RunRecord.from_json(json.dumps(bundle["run"]))
        if run.experiment_id != spec.experiment_id:
            raise ValueError("run and specification identities differ")
        artifacts = bundle["artifacts"]
        checked: list[tuple[str, bytes]] = []
        for entry in artifacts:
            name = entry["name"]
            content = base64.b64decode(entry["content_base64"], validate=True)
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError("artifact hash mismatch in import")
            checked.append((name, content))
        if set(run.artifact_ids) != {name for name, _ in checked}:
            raise ValueError("artifact links do not match imported contents")
        for ref, snapshot in bundle.get("workspaces", {}).items():
            if self.workspaces.import_snapshot(snapshot) != ref:
                raise ValueError("workspace reference mismatch")
        for ref, raw in bundle.get("environments", {}).items():
            self.workspaces.import_environment(ref, raw)
        self.add_spec(spec)
        pending = replace(run, status="pending", started_at=None, finished_at=None, artifact_ids=())
        self.save_run(pending)
        for name, content in checked:
            self.add_artifact(run.run_id, name, content)
        for event in bundle["events"]:
            sequence = self.append_event(run.run_id, event["event_id"], event["fields"])
            if sequence != event["sequence"]:
                raise ValueError("event sequence mismatch in import")
        if run.status in {"running", "completed", "failed", "timed_out"} and run.started_at:
            self.save_run(replace(run, status="running", finished_at=None, artifact_ids=()))
        self.save_run(run)

    def close(self) -> None:
        with self._lock:
            self.conn.close()
