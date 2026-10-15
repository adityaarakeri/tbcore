"""Append-only JSONL call log for request accounting."""

from __future__ import annotations

import enum
import json
import os
from collections.abc import Mapping
from pathlib import Path

DEFAULT_LOG_PATH = Path(".tbcore") / "calls.jsonl"
LOG_PATH_ENV = "TBCORE_LOG_PATH"


class _Unset(enum.Enum):
    """Sentinel: no path passed, so the env override / default applies."""

    TOKEN = enum.auto()


_UNSET = _Unset.TOKEN


class CallLog:
    """One JSON object per line, appended per call.

    ``path=None`` keeps the log in memory only (records are still accepted and
    counted via :meth:`entries`), which is useful for tests and dry runs.

    When no path is passed, ``TBCORE_LOG_PATH`` is used if set and non-empty,
    otherwise :data:`DEFAULT_LOG_PATH` (relative to the working directory). An
    explicit path (including ``None``) always wins over the environment.
    """

    def __init__(self, path: str | os.PathLike[str] | None | _Unset = _UNSET) -> None:
        if isinstance(path, _Unset):
            path = os.environ.get(LOG_PATH_ENV) or DEFAULT_LOG_PATH
        self.path = Path(path) if path is not None else None
        self._memory_entries: list[str] = []

    def append(self, record: Mapping[str, object]) -> None:
        serialized = json.dumps(dict(record), ensure_ascii=False, sort_keys=True)
        if self.path is None:
            self._memory_entries.append(serialized)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(serialized + "\n")

    def entries(self) -> list[dict[str, object]]:
        if self.path is None:
            return [json.loads(entry) for entry in self._memory_entries]
        if not self.path.exists():
            return []
        entries = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    entries.append(json.loads(stripped))
        return entries
