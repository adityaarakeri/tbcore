"""Durable sandbox command timeline and Codex exec JSONL tool-event adapter."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from tbcore.experiment_store import ExperimentStore
from tbcore.sandbox import RunResult

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SCHEMA = "codex-exec-jsonl-v1"
_KNOWN_EVENTS = frozenset(
    {
        "thread.started",
        "turn.started",
        "turn.completed",
        "turn.failed",
        "item.started",
        "item.updated",
        "item.completed",
    }
)


@dataclass(frozen=True)
class TelemetryCoverage:
    complete: bool
    tool_calls: int
    gaps: tuple[str, ...]


class ExecutionTimeline:
    """Attach outer sandbox results and supported structured agent events to one run."""

    def __init__(self, store: ExperimentStore, run_id: str) -> None:
        self.store = store
        self.run_id = run_id

    def _output_ref(self, event_id: str, output: str) -> str | None:
        policy = self.store.policy
        if not output or not policy.content_enabled:
            return None
        if len(output.encode("utf-8")) > policy.max_artifact_bytes:
            return None
        name = (
            f"{event_id}.output.txt"
            if len(event_id) <= 100 and _SAFE_ID.fullmatch(event_id)
            else f"{hashlib.sha256(event_id.encode()).hexdigest()}.output.txt"
        )
        self.store.add_artifact(self.run_id, name, output.encode("utf-8"))
        return name

    def record_command(
        self,
        event_id: str,
        result: RunResult,
        *,
        parent_event_id: str | None = None,
        cancelled: bool = False,
    ) -> None:
        """Record a Sandbox.run or SandboxSession.exec result as an outer command."""
        command_json = json.dumps(result.command, separators=(",", ":"))
        self.store.append_event(
            self.run_id,
            event_id,
            {
                "event_type": "sandbox_command",
                "parent_event_id": parent_event_id,
                "command": list(result.command),
                "command_sha256": hashlib.sha256(command_json.encode()).hexdigest(),
                "duration_ms": round(result.duration_s * 1000),
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "cancelled": cancelled,
                "output_ref": self._output_ref(event_id, result.stdout + result.stderr),
            },
        )

    def record_codex_exec(
        self,
        event_id: str,
        result: RunResult,
        *,
        cli_version: str,
        cancelled: bool = False,
    ) -> TelemetryCoverage:
        """Parse only `codex exec --json` stdout, never prose or API response chunks."""
        if not cli_version:
            raise ValueError("cli_version is required for structured event provenance")
        if "--json" not in result.command or "exec" not in result.command:
            raise ValueError("Codex telemetry requires a codex exec --json result")
        self.record_command(event_id, result, cancelled=cancelled)
        gaps: list[str] = []
        started: dict[str, dict[str, object]] = {}
        completed: dict[str, dict[str, object]] = {}
        turn_complete = False
        lines = result.stdout.splitlines()
        for index, line in enumerate(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                gaps.append(
                    "truncated-json"
                    if index == len(lines) - 1 and not result.stdout.endswith("\n")
                    else "malformed-json"
                )
                continue
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                gaps.append("malformed-event")
                continue
            kind = event["type"]
            if kind not in _KNOWN_EVENTS:
                gaps.append("unknown-event")
                continue
            if kind == "turn.completed":
                turn_complete = True
            if kind == "turn.failed":
                gaps.append("turn-failed")
            if not kind.startswith("item."):
                continue
            item = event.get("item")
            if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                gaps.append("malformed-item")
                continue
            if item["type"] != "command_execution":
                if item["type"] not in {"agent_message", "reasoning"}:
                    gaps.append("unsupported-item-type")
                continue
            item_id = item.get("id")
            command = item.get("command")
            if not isinstance(item_id, str) or not item_id or not isinstance(command, str):
                gaps.append("malformed-command")
                continue
            if kind == "item.started":
                if item_id in started and started[item_id] != item:
                    gaps.append("conflicting-duplicate")
                started[item_id] = item
            elif kind == "item.completed":
                if item_id in completed:
                    if completed[item_id] != item:
                        gaps.append("conflicting-duplicate")
                    continue
                completed[item_id] = item
                if item_id not in started:
                    gaps.append("missing-start")
                exit_code = item.get("exit_code")
                if exit_code is not None and type(exit_code) is not int:
                    gaps.append("malformed-exit-code")
                    exit_code = None
                output = item.get("aggregated_output")
                if output is not None and not isinstance(output, str):
                    gaps.append("malformed-output")
                    output = None
                tool_event_id = f"{event_id}.{item_id}"
                self.store.append_event(
                    self.run_id,
                    tool_event_id,
                    {
                        "event_type": "agent_command",
                        "parent_event_id": event_id,
                        "command": command,
                        "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                        "exit_code": exit_code,
                        "duration_ms": (
                            item.get("duration_ms")
                            if type(item.get("duration_ms")) is int
                            else None
                        ),
                        "timed_out": (
                            item.get("timed_out") if type(item.get("timed_out")) is bool else None
                        ),
                        "cancelled": (
                            item.get("cancelled") if type(item.get("cancelled")) is bool else None
                        ),
                        "agent_version": cli_version,
                        "event_schema": _SCHEMA,
                        "output_ref": self._output_ref(tool_event_id, output or ""),
                    },
                )
        if not turn_complete:
            gaps.append("missing-turn-completed")
        if result.timed_out or cancelled or result.exit_code != 0:
            gaps.append("outer-command-incomplete")
        if started.keys() - completed.keys():
            gaps.append("unfinished-tool")
        coverage = TelemetryCoverage(not gaps, len(completed), tuple(dict.fromkeys(gaps)))
        self.store.append_event(
            self.run_id,
            f"{event_id}.coverage",
            {
                "event_type": "agent_coverage",
                "parent_event_id": event_id,
                "agent_version": cli_version,
                "event_schema": _SCHEMA,
                "coverage_complete": coverage.complete,
                "coverage_gaps": list(coverage.gaps),
                "tool_calls": coverage.tool_calls,
            },
        )
        return coverage
