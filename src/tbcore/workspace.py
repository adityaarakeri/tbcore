"""Immutable workspace inputs and environment provenance for local experiments."""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from tbcore.evidence import CapturePolicy

WORKSPACE_SCHEMA_VERSION = 1
_DEFAULT_EXCLUDES = (".git", ".git/**", ".venv", ".venv/**", ".agent", ".agent/**")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _relative(path: str) -> PurePosixPath:
    value = PurePosixPath(path)
    if (
        not path
        or value.is_absolute()
        or "\\" in path
        or ":" in path
        or any(part in ("", ".", "..") for part in path.split("/"))
    ):
        raise ValueError("unsafe workspace path")
    return value


def _excluded(path: str, patterns: tuple[str, ...]) -> bool:
    return any(path == pattern or fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise ValueError("stored content integrity mismatch")
        return
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(content)
        if os.name != "nt":
            temp.chmod(0o600)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


@dataclass(frozen=True)
class WorkspaceFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class WorkspaceManifest:
    snapshot_id: str
    entries: tuple[WorkspaceFile, ...]
    file_count: int
    total_bytes: int
    source_fingerprint: str
    complete: bool
    schema_version: int = WORKSPACE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkspaceManifest:
        if value.get("schema_version") != WORKSPACE_SCHEMA_VERSION:
            raise ValueError("unsupported workspace schema version")
        entries = tuple(WorkspaceFile(**item) for item in value["entries"])
        return cls(
            snapshot_id=value["snapshot_id"],
            entries=entries,
            file_count=value["file_count"],
            total_bytes=value["total_bytes"],
            source_fingerprint=value["source_fingerprint"],
            complete=value["complete"],
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class EnvironmentProvenance:
    image_id: str
    registry_digest: str | None
    tbcore_version: str
    agent_cli_version: str
    model: str
    provider: str
    resource_limits: dict[str, Any]
    runtime: str
    policy_config: dict[str, Any]
    schema_version: int = WORKSPACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WORKSPACE_SCHEMA_VERSION:
            raise ValueError("unsupported environment schema version")
        for name in (
            "image_id",
            "tbcore_version",
            "agent_cli_version",
            "model",
            "provider",
            "runtime",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if self.registry_digest is not None and not self.registry_digest:
            raise ValueError("registry_digest must be nonempty when provided")
        if not isinstance(self.resource_limits, dict) or not isinstance(self.policy_config, dict):
            raise ValueError("resource_limits and policy_config must be objects")
        _json(asdict(self))

    def verify_image(self, available_ids: set[str]) -> None:
        if self.image_id not in available_ids and self.registry_digest not in available_ids:
            raise ValueError(f"image unavailable: {self.image_id}")

    def to_json(self) -> str:
        return _json(asdict(self)).decode("utf-8")

    @classmethod
    def from_json(cls, raw: str) -> EnvironmentProvenance:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("environment provenance must be an object")
        return cls(**value)


class WorkspaceStore:
    """Content-addressed snapshots with explicit content and binary opt-ins."""

    def __init__(
        self,
        root: str | Path,
        policy: CapturePolicy,
        *,
        allow_binary: bool = False,
        max_file_bytes: int = 1024 * 1024,
        max_total_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        if type(max_file_bytes) is not int or max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be positive")
        if type(max_total_bytes) is not int or max_total_bytes <= 0:
            raise ValueError("max_total_bytes must be positive")
        self.root = Path(root)
        self.policy = policy
        self.allow_binary = allow_binary
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        for path in (
            self.root,
            self.root / "blobs",
            self.root / "manifests",
            self.root / "environments",
        ):
            if path.is_symlink():
                raise ValueError("workspace store path must not be a symlink")
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            if os.name != "nt":
                path.chmod(0o700)

    def capture(self, source: str | Path, *, exclude: tuple[str, ...] = ()) -> WorkspaceManifest:
        source_path = Path(source)
        if source_path.is_symlink() or not source_path.is_dir():
            raise ValueError("workspace source must be a real directory")
        if self.root.resolve().is_relative_to(source_path.resolve()):
            raise ValueError("workspace store must be outside the captured source")
        patterns = _DEFAULT_EXCLUDES + exclude
        pending: list[tuple[WorkspaceFile, bytes]] = []
        fingerprints: list[tuple[str, str]] = []
        total = 0
        for directory, dirs, files in os.walk(source_path, topdown=True, followlinks=False):
            current = Path(directory)
            selected_dirs = []
            for name in sorted(dirs):
                path = current / name
                relative = path.relative_to(source_path).as_posix()
                if _excluded(relative, patterns):
                    continue
                if path.is_symlink():
                    raise ValueError(f"workspace symlink is not supported: {relative}")
                selected_dirs.append(name)
            dirs[:] = selected_dirs
            for name in sorted(files):
                path = current / name
                relative = path.relative_to(source_path).as_posix()
                if _excluded(relative, patterns):
                    continue
                if path.is_symlink():
                    raise ValueError(f"workspace symlink is not supported: {relative}")
                _relative(relative)
                before = path.stat()
                if before.st_size > self.max_file_bytes:
                    raise ValueError(f"workspace file size exceeds limit: {relative}")
                content = path.read_bytes()
                if len(content) > self.max_file_bytes:
                    raise ValueError(f"workspace file size exceeds limit: {relative}")
                after = path.stat()
                if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                    raise ValueError(f"workspace file changed during capture: {relative}")
                total += len(content)
                if total > self.max_total_bytes:
                    raise ValueError("workspace total size exceeds limit")
                digest = _digest(content)
                fingerprints.append((relative, digest))
                if self.policy.content_enabled:
                    for secret in self.policy.secret_values:
                        if secret.encode("utf-8") in content:
                            raise ValueError(f"workspace contains configured secret: {relative}")
                    try:
                        decoded = content.decode("utf-8")
                        binary = "\x00" in decoded
                    except UnicodeDecodeError:
                        binary = True
                    if binary and not self.allow_binary:
                        raise ValueError(f"binary workspace file requires opt-in: {relative}")
                    pending.append((WorkspaceFile(relative, digest, len(content)), content))
        fingerprint = _digest(_json(fingerprints))
        entries = tuple(entry for entry, _ in pending)
        document = {
            "entries": [asdict(entry) for entry in entries],
            "file_count": len(fingerprints),
            "total_bytes": total,
            "source_fingerprint": fingerprint,
            "complete": self.policy.content_enabled,
            "schema_version": WORKSPACE_SCHEMA_VERSION,
        }
        snapshot_id = _digest(_json(document))
        manifest = WorkspaceManifest(
            snapshot_id=snapshot_id,
            entries=entries,
            file_count=len(fingerprints),
            total_bytes=total,
            source_fingerprint=fingerprint,
            complete=self.policy.content_enabled,
        )
        for entry, content in pending:
            _write_immutable(self.root / "blobs" / entry.sha256, content)
        _write_immutable(self.root / "manifests" / f"{snapshot_id}.json", _json(manifest.to_dict()))
        return manifest

    def get_manifest(self, snapshot_id: str) -> WorkspaceManifest:
        if len(snapshot_id) != 64 or any(char not in "0123456789abcdef" for char in snapshot_id):
            raise ValueError("invalid snapshot ID")
        path = self.root / "manifests" / f"{snapshot_id}.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("snapshot manifest missing")
        manifest = WorkspaceManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        document = manifest.to_dict()
        document.pop("snapshot_id")
        if _digest(_json(document)) != snapshot_id:
            raise ValueError("snapshot manifest integrity mismatch")
        return manifest

    def export_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        manifest = self.get_manifest(snapshot_id)
        blobs = []
        for digest in sorted({entry.sha256 for entry in manifest.entries}):
            path = self.root / "blobs" / digest
            if path.is_symlink() or not path.is_file():
                raise ValueError("workspace blob missing")
            content = path.read_bytes()
            if _digest(content) != digest:
                raise ValueError("workspace blob integrity mismatch")
            blobs.append(
                {"sha256": digest, "content_base64": base64.b64encode(content).decode("ascii")}
            )
        return {"manifest": manifest.to_dict(), "blobs": blobs}

    def import_snapshot(self, bundle: dict[str, Any]) -> str:
        manifest = WorkspaceManifest.from_dict(bundle["manifest"])
        document = manifest.to_dict()
        document.pop("snapshot_id")
        if _digest(_json(document)) != manifest.snapshot_id:
            raise ValueError("workspace manifest integrity mismatch")
        expected = {entry.sha256 for entry in manifest.entries}
        decoded: dict[str, bytes] = {}
        for blob in bundle["blobs"]:
            digest = blob["sha256"]
            content = base64.b64decode(blob["content_base64"], validate=True)
            if digest in decoded or _digest(content) != digest:
                raise ValueError("workspace blob integrity mismatch")
            decoded[digest] = content
        if set(decoded) != expected:
            raise ValueError("workspace blob set does not match manifest")
        for entry in manifest.entries:
            _relative(entry.path)
            if len(decoded[entry.sha256]) != entry.size:
                raise ValueError("workspace blob size mismatch")
        for digest, content in decoded.items():
            _write_immutable(self.root / "blobs" / digest, content)
        _write_immutable(
            self.root / "manifests" / f"{manifest.snapshot_id}.json", _json(manifest.to_dict())
        )
        return manifest.snapshot_id

    def restore(self, snapshot_id: str, destination: str | Path) -> None:
        manifest = self.get_manifest(snapshot_id)
        if not manifest.complete:
            raise ValueError("workspace reconstruction is incomplete: content capture was disabled")
        contents: list[tuple[PurePosixPath, bytes]] = []
        for entry in manifest.entries:
            relative = _relative(entry.path)
            path = self.root / "blobs" / entry.sha256
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"workspace blob missing: {entry.sha256}")
            content = path.read_bytes()
            if len(content) != entry.size or _digest(content) != entry.sha256:
                raise ValueError(f"workspace blob integrity mismatch: {entry.sha256}")
            contents.append((relative, content))
        target = Path(destination)
        if target.is_symlink() or (target.exists() and any(target.iterdir())):
            raise ValueError("restore destination must be an empty real directory")
        target.mkdir(mode=0o700, parents=True, exist_ok=True)
        for relative, content in contents:
            path = target.joinpath(*relative.parts)
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(content)

    def record_environment(self, provenance: EnvironmentProvenance) -> str:
        raw = provenance.to_json()
        capture = CapturePolicy(
            content_enabled=True,
            secret_values=self.policy.secret_values,
            max_text_bytes=self.policy.max_text_bytes,
        )
        if capture.capture(json.loads(raw)) != json.loads(raw):
            raise ValueError("environment provenance contains sensitive or oversized content")
        reference = _digest(raw.encode("utf-8"))
        _write_immutable(self.root / "environments" / f"{reference}.json", raw.encode("utf-8"))
        return reference

    def get_environment(self, reference: str) -> EnvironmentProvenance:
        if len(reference) != 64 or any(char not in "0123456789abcdef" for char in reference):
            raise ValueError("invalid environment reference")
        path = self.root / "environments" / f"{reference}.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("environment provenance missing")
        raw = path.read_bytes()
        if _digest(raw) != reference:
            raise ValueError("environment provenance integrity mismatch")
        return EnvironmentProvenance.from_json(raw.decode("utf-8"))

    def export_environment(self, reference: str) -> str:
        return self.get_environment(reference).to_json()

    def import_environment(self, reference: str, raw: str) -> None:
        provenance = EnvironmentProvenance.from_json(raw)
        if _digest(provenance.to_json().encode("utf-8")) != reference:
            raise ValueError("environment reference mismatch")
        self.record_environment(provenance)
