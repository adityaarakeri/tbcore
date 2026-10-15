"""Capture policy and bounded local evidence artifacts for experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REDACTED = "[REDACTED]"
_METADATA_KEYS = frozenset(
    {
        "event_id",
        "run_id",
        "attempt_id",
        "status",
        "event_type",
        "parent_event_id",
        "exit_code",
        "duration_ms",
        "timed_out",
        "cancelled",
        "command_sha256",
        "output_ref",
        "agent_version",
        "event_schema",
        "coverage_complete",
        "coverage_gaps",
        "tool_calls",
        "timestamp",
        "created_at",
        "provider",
        "provider_request_id",
        "model",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "call_source",
        "usage_complete",
        "telemetry_reason",
        "priced_event_id",
        "estimated_cost",
        "rate_source",
        "rate_version",
        "effective_date",
        "currency",
        "rate_units",
        "estimation_method",
        "input_rate_per_mtok",
        "output_rate_per_mtok",
        "cache_read_rate_per_mtok",
        "cache_write_rate_per_mtok",
        "budget_reason",
        "budget_cost_usd",
        "in_flight",
        "budget_in_flight",
        "budget_inflight_can_exceed_limit",
    }
)
_SENSITIVE_KEYS = frozenset(
    {
        "headers",
        "inject_headers",
        "environment",
        "env",
        "authorization",
        "proxy_authorization",
        "x_api_key",
        "api_key",
        "api_keys",
        "password",
        "secret",
        "secrets",
        "token",
        "credential",
        "credentials",
        "cookie",
        "set_cookie",
    }
)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _safe_name(value: str, kind: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"invalid {kind} name")
    return value


def _sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        ("_api_key", "_password", "_secret", "_token", "_credential")
    )


@dataclass(frozen=True)
class CapturePolicy:
    """Metadata-only by default; content capture requires an explicit opt-in."""

    content_enabled: bool = False
    secret_values: tuple[str, ...] = ()
    max_text_bytes: int = 64 * 1024
    max_artifact_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        if not isinstance(self.content_enabled, bool):
            raise ValueError("content_enabled must be a boolean")
        if any(not isinstance(value, str) or not value for value in self.secret_values):
            raise ValueError("secret_values must contain nonempty strings")
        if type(self.max_text_bytes) is not int or self.max_text_bytes <= 0:
            raise ValueError("max_text_bytes must be positive")
        if type(self.max_artifact_bytes) is not int or self.max_artifact_bytes <= 0:
            raise ValueError("max_artifact_bytes must be positive")

    def capture(self, fields: Mapping[str, object]) -> dict[str, Any]:
        """Return only safe metadata or redacted, bounded opted-in content."""
        if any(not isinstance(key, str) for key in fields):
            raise ValueError("evidence field names must be strings")
        return {key: self._capture_value(key, value) for key, value in fields.items()}

    def _capture_value(self, key: str, value: object) -> Any:
        if _sensitive_key(key):
            return _REDACTED
        if not self.content_enabled and key not in _METADATA_KEYS:
            return {"omitted": "capture-disabled"}
        if isinstance(value, Mapping):
            if any(not isinstance(child, str) for child in value):
                raise ValueError("evidence field names must be strings")
            return {child: self._capture_value(child, item) for child, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._capture_value(key, item) for item in value]
        if isinstance(value, bytes):
            try:
                text = value.decode("utf-8")
            except UnicodeDecodeError:
                return {"omitted": "binary", "bytes": len(value)}
            if "\x00" in text:
                return {"omitted": "binary", "bytes": len(value)}
            return self._capture_text(text)
        if isinstance(value, str):
            return self._capture_text(value)
        if value is None or isinstance(value, (bool, int, float)):
            json.dumps(value, allow_nan=False)
            return value
        raise ValueError(f"unsupported evidence value for {key!r}")

    def _capture_text(self, value: str) -> str | dict[str, int | str]:
        size = len(value.encode("utf-8"))
        if size > self.max_text_bytes:
            return {"omitted": "oversized", "bytes": size}
        return self._redact_text(value)

    def _redact_text(self, value: str) -> str:
        for secret in sorted(self.secret_values, key=len, reverse=True):
            value = value.replace(secret, _REDACTED)
        return value


class EvidenceArtifacts:
    """Small, flat artifact area with explicit content and retention policy."""

    def __init__(self, root: str | Path, policy: CapturePolicy) -> None:
        self.root = Path(root)
        self.policy = policy
        if self.root.is_symlink():
            raise ValueError("artifact root must not be a symlink")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt":
            self.root.chmod(0o700)

    def write(self, run_id: str, name: str, content: bytes) -> str:
        """Store a redacted UTF-8 artifact and return its SHA-256 digest."""
        _safe_name(run_id, "run")
        _safe_name(name, "artifact")
        if not self.policy.content_enabled:
            raise ValueError("content capture is disabled")
        if not isinstance(content, bytes):
            raise ValueError("artifact content must be bytes")
        if len(content) > self.policy.max_artifact_bytes:
            raise ValueError("artifact size exceeds policy limit")
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("binary artifacts are not supported by this policy") from exc
        if "\x00" in decoded:
            raise ValueError("binary artifacts are not supported by this policy")
        output = self.policy._redact_text(decoded).encode("utf-8")
        if len(output) > self.policy.max_artifact_bytes:
            raise ValueError("artifact size exceeds policy limit after redaction")
        run_dir = self.root / run_id
        if run_dir.is_symlink():
            raise ValueError("artifact run directory must not be a symlink")
        run_dir.mkdir(mode=0o700, exist_ok=True)
        if os.name != "nt":
            run_dir.chmod(0o700)
        target = run_dir / name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(target, flags, 0o600)
        with os.fdopen(fd, "wb") as artifact:
            artifact.write(output)
        return hashlib.sha256(output).hexdigest()

    def cleanup(self, retained: Mapping[str, set[str]]) -> None:
        """Remove only artifacts absent from the retained run-to-name map."""
        for run_id, names in retained.items():
            _safe_name(run_id, "run")
            for name in names:
                _safe_name(name, "artifact")
        for run_dir in self.root.iterdir():
            if run_dir.is_symlink() or not run_dir.is_dir():
                raise ValueError("unexpected artifact entry in store")
            _safe_name(run_dir.name, "run")
            keep = retained.get(run_dir.name, set())
            for target in run_dir.iterdir():
                if target.is_symlink() or not target.is_file():
                    raise ValueError("unexpected artifact entry in run directory")
                _safe_name(target.name, "artifact")
                if target.name not in keep:
                    target.unlink()
            if not any(run_dir.iterdir()):
                run_dir.rmdir()
