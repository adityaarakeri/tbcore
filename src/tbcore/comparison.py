"""Compare trial populations with explicit denominators and exclusions."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from statistics import mean

from tbcore.costs import CostLedger
from tbcore.experiment import RunRecord
from tbcore.experiment_store import ExperimentStore


@dataclass(frozen=True)
class TrialEvidence:
    task_id: str
    seed: int | None
    trial_id: str
    run_id: str
    attempt_ids: tuple[str, ...]
    workspace_ref: str | None
    environment_ref: str | None
    event_ids: tuple[str, ...]
    tool_event_ids: tuple[str, ...]
    model_call_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    status: str
    verdict: str | None
    latency_s: float | None
    tokens: int | None
    cost_usd: str | None


@dataclass(frozen=True)
class PopulationSummary:
    success_rate: float | None
    success_ci_95: tuple[float, float] | None
    mean_latency_s: float | None
    mean_tokens: float | None
    mean_cost_usd: str | None
    denominators: dict[str, int]
    exclusions: dict[str, int]
    trials: tuple[TrialEvidence, ...]


@dataclass(frozen=True)
class ComparisonReport:
    experiment_group: str
    dataset_version: str
    evaluator_id: str
    evaluator_version: str
    baseline_id: str
    candidate_id: str
    baseline: PopulationSummary
    candidate: PopulationSummary
    paired_count: int
    missing_baseline: tuple[str, ...]
    missing_candidate: tuple[str, ...]
    paired_success_difference: float | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    def to_markdown(self) -> str:
        lines = [
            f"# Comparison: {self.baseline_id} vs {self.candidate_id}",
            "",
            f"Dataset: {self.dataset_version}; "
            f"evaluator: {self.evaluator_id}@{self.evaluator_version}.",
            f"Paired trials: {self.paired_count}. "
            f"Missing baseline: {', '.join(self.missing_baseline) or 'none'}. "
            f"Missing candidate: {', '.join(self.missing_candidate) or 'none'}.",
            "",
            "| Population | Success (95% CI) | Quality n | Mean latency s (n) | "
            "Mean tokens (n) | Mean cost USD (n) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, population in (
            (self.baseline_id, self.baseline),
            (self.candidate_id, self.candidate),
        ):
            d = population.denominators
            lines.append(
                f"| {name} | {population.success_rate} ({population.success_ci_95}) | "
                f"{d['quality']} | {population.mean_latency_s} ({d['latency']}) | "
                f"{population.mean_tokens} ({d['tokens']}) | "
                f"{population.mean_cost_usd} ({d['cost']}) |"
            )
        lines.append("")
        for name, population in (
            (self.baseline_id, self.baseline),
            (self.candidate_id, self.candidate),
        ):
            lines.append(
                f"Exclusions for {name}: "
                + ", ".join(f"{reason}={count}" for reason, count in population.exclusions.items())
                + "."
            )
        lines.extend(
            [
                "",
                "Incomplete telemetry and execution/evaluation failures are "
                "excluded from their affected metrics.",
                "",
                "## Trial evidence",
                "",
            ]
        )
        for population in (self.baseline, self.candidate):
            for trial in population.trials:
                lines.append(
                    f"- {trial.task_id}:{trial.seed} {trial.run_id}: {trial.status}, "
                    f"verdict={trial.verdict}, workspace={trial.workspace_ref}, "
                    f"tools={len(trial.tool_event_ids)}, "
                    f"model calls={len(trial.model_call_ids)}, cost={trial.cost_usd}"
                )
        return "\n".join(lines) + "\n"


def _key(task_id: str, seed: int | None) -> str:
    return f"{task_id}:{seed}"


def _wilson_95(successes: int, count: int) -> tuple[float, float] | None:
    if count == 0:
        return None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _population(
    store: ExperimentStore, runs: list[RunRecord], evaluator_id: str, evaluator_version: str
) -> PopulationSummary:
    by_trial: dict[str, list[RunRecord]] = {}
    for run in runs:
        by_trial.setdefault(run.trial_id, []).append(run)
    evidence: list[TrialEvidence] = []
    success: list[int] = []
    latencies: list[float] = []
    tokens: list[int] = []
    costs: list[Decimal] = []
    exclusions = {
        "execution_failure": 0,
        "evaluation_missing": 0,
        "incomplete_telemetry": 0,
        "unknown_cost": 0,
    }
    for history in by_trial.values():
        run = next((item for item in reversed(history) if item.status == "completed"), history[-1])
        spec = store.get_spec(run.experiment_id)
        seed = run.configuration.get("seed") if run.configuration else None
        if seed is not None and type(seed) is not int:
            raise ValueError("trial seed must be an integer or null")
        events = store.events(run.run_id)
        evaluations = [
            event["fields"]
            for event in events
            if event["fields"].get("event_type") == "evaluation"
            and event["fields"].get("evaluator_id") == evaluator_id
            and event["fields"].get("version") == evaluator_version
        ]
        verdict = evaluations[-1].get("verdict") if evaluations else None
        if run.status != "completed":
            exclusions["execution_failure"] += 1
        elif verdict not in {"pass", "fail"}:
            exclusions["evaluation_missing"] += 1
        else:
            success.append(int(verdict == "pass"))
        latency = None
        if run.started_at and run.finished_at:
            latency = (
                datetime.fromisoformat(run.finished_at.replace("Z", "+00:00"))
                - datetime.fromisoformat(run.started_at.replace("Z", "+00:00"))
            ).total_seconds()
            latencies.append(latency)
        calls = [
            event["fields"] for event in events if event["fields"].get("event_type") == "model_call"
        ]
        total_tokens = None
        cost = None
        if run.usage_coverage != "complete" or any(
            type(call.get(field)) is not int
            for call in calls
            for field in ("input_tokens", "output_tokens")
        ):
            exclusions["incomplete_telemetry"] += 1
        else:
            total_tokens = sum(call["input_tokens"] + call["output_tokens"] for call in calls)
            tokens.append(total_tokens)
        if run.cost_coverage == "complete":
            try:
                total = CostLedger(store).run_totals(run.run_id).total
            except ValueError:
                total = None
            if total is not None:
                cost = str(total)
                costs.append(total)
            else:
                exclusions["unknown_cost"] += 1
        else:
            exclusions["unknown_cost"] += 1
        evidence.append(
            TrialEvidence(
                spec.task_id,
                seed,
                run.trial_id,
                run.run_id,
                tuple(item.attempt_id for item in history),
                run.workspace_ref,
                run.environment_ref,
                tuple(event["event_id"] for event in events),
                tuple(
                    event["event_id"]
                    for event in events
                    if event["fields"].get("event_type") in {"tool_call", "command"}
                ),
                tuple(
                    event["event_id"]
                    for event in events
                    if event["fields"].get("event_type") == "model_call"
                ),
                run.artifact_ids,
                run.status,
                verdict if isinstance(verdict, str) else None,
                latency,
                total_tokens,
                cost,
            )
        )
    return PopulationSummary(
        mean(success) if success else None,
        _wilson_95(sum(success), len(success)),
        mean(latencies) if latencies else None,
        mean(tokens) if tokens else None,
        str(sum(costs) / len(costs)) if costs else None,
        {
            "trials": len(evidence),
            "quality": len(success),
            "latency": len(latencies),
            "tokens": len(tokens),
            "cost": len(costs),
        },
        exclusions,
        tuple(evidence),
    )


def compare_experiments(
    store: ExperimentStore,
    experiment_group: str,
    baseline_id: str,
    candidate_id: str,
    dataset_version: str,
    evaluator_id: str,
    evaluator_version: str,
) -> ComparisonReport:
    populations: dict[str, list[RunRecord]] = {baseline_id: [], candidate_id: []}
    for run in store.all_runs():
        spec = store.get_spec(run.experiment_id)
        if spec.configuration.get("experiment_group") != experiment_group:
            continue
        config_id = spec.configuration.get("configuration_id")
        if config_id not in populations:
            continue
        if spec.configuration.get("dataset_version") != dataset_version:
            raise ValueError("incompatible dataset version")
        populations[config_id].append(run)
    baseline = _population(store, populations[baseline_id], evaluator_id, evaluator_version)
    candidate = _population(store, populations[candidate_id], evaluator_id, evaluator_version)
    if any(
        trial.status == "completed" and trial.verdict is None
        for pop in (baseline, candidate)
        for trial in pop.trials
    ):
        raise ValueError("incompatible or missing evaluator version")
    base_keys = {_key(item.task_id, item.seed): item for item in baseline.trials}
    cand_keys = {_key(item.task_id, item.seed): item for item in candidate.trials}
    if len(base_keys) != len(baseline.trials) or len(cand_keys) != len(candidate.trials):
        raise ValueError("duplicate task and seed in comparison population")
    paired = sorted(base_keys.keys() & cand_keys.keys())
    differences = [
        int(cand_keys[key].verdict == "pass") - int(base_keys[key].verdict == "pass")
        for key in paired
        if base_keys[key].verdict in {"pass", "fail"} and cand_keys[key].verdict in {"pass", "fail"}
    ]
    return ComparisonReport(
        experiment_group,
        dataset_version,
        evaluator_id,
        evaluator_version,
        baseline_id,
        candidate_id,
        baseline,
        candidate,
        len(paired),
        tuple(sorted(cand_keys.keys() - base_keys.keys())),
        tuple(sorted(base_keys.keys() - cand_keys.keys())),
        mean(differences) if differences else None,
    )
