"""tbcore: token-build core plumbing."""

from tbcore.accounting import DEFAULT_LOG_PATH, CallLog
from tbcore.budget import BudgetExceeded, BudgetLimits, BudgetSnapshot, RunBudget
from tbcore.client import (
    APIStatusError,
    Client,
    Completion,
    HttpTransport,
    OpenAICompatTransport,
    RateLimitError,
    RetryableError,
    TbError,
    TransientError,
    Transport,
    TransportRequest,
    Usage,
)
from tbcore.comparison import (
    ComparisonReport,
    PopulationSummary,
    TrialEvidence,
    compare_experiments,
)
from tbcore.config import ConfigError, ModelConfig, load_model_config
from tbcore.costs import CostLedger, CostTotals, PriceCard, RunCostTotals, TokenRates
from tbcore.evaluation import EvaluationResult, EvaluationSpec, Evaluator
from tbcore.evidence import CapturePolicy, EvidenceArtifacts
from tbcore.experiment import ExperimentSpec, RunRecord
from tbcore.experiment_store import ExperimentStore
from tbcore.experiment_usage import CallContext
from tbcore.log import configure_logging
from tbcore.models import REGISTRY, ModelSpec, make_model_spec, resolve_model
from tbcore.pool import ExecutionPool, WorkItem, WorkStore, enqueue
from tbcore.sandbox import (
    DEFAULT_ALLOW_HOSTS,
    RunResult,
    Sandbox,
    SandboxError,
    SandboxSession,
    build_codex_image,
    build_default_images,
    build_image,
    image_context,
    pull_default_images,
)
from tbcore.stats import RegressionResult, Summary, bootstrap_ci, cohens_d, regressed, summary
from tbcore.timeline import ExecutionTimeline, TelemetryCoverage
from tbcore.trials import (
    SandboxTrialExecutor,
    TrialCase,
    TrialContext,
    TrialOwnedError,
    TrialRunner,
)
from tbcore.workspace import EnvironmentProvenance, WorkspaceManifest, WorkspaceStore

__version__ = "0.2.0"

__all__ = [
    "APIStatusError",
    "BudgetExceeded",
    "BudgetLimits",
    "BudgetSnapshot",
    "CapturePolicy",
    "DEFAULT_ALLOW_HOSTS",
    "DEFAULT_LOG_PATH",
    "EvidenceArtifacts",
    "EnvironmentProvenance",
    "ExecutionPool",
    "ExecutionTimeline",
    "ExperimentSpec",
    "ExperimentStore",
    "REGISTRY",
    "RegressionResult",
    "RunResult",
    "RunRecord",
    "Sandbox",
    "SandboxError",
    "SandboxSession",
    "Summary",
    "TelemetryCoverage",
    "CallLog",
    "CallContext",
    "CostLedger",
    "CostTotals",
    "Client",
    "Completion",
    "HttpTransport",
    "ModelSpec",
    "OpenAICompatTransport",
    "PriceCard",
    "RateLimitError",
    "RetryableError",
    "RunCostTotals",
    "RunBudget",
    "Transport",
    "TransportRequest",
    "TokenRates",
    "TransientError",
    "TbError",
    "Usage",
    "WorkItem",
    "WorkStore",
    "WorkspaceManifest",
    "WorkspaceStore",
    "__version__",
    "bootstrap_ci",
    "build_default_images",
    "build_codex_image",
    "build_image",
    "cohens_d",
    "enqueue",
    "image_context",
    "ConfigError",
    "ModelConfig",
    "configure_logging",
    "load_model_config",
    "make_model_spec",
    "pull_default_images",
    "regressed",
    "resolve_model",
    "summary",
    "ComparisonReport",
    "PopulationSummary",
    "TrialEvidence",
    "compare_experiments",
    "EvaluationResult",
    "EvaluationSpec",
    "Evaluator",
    "SandboxTrialExecutor",
    "TrialCase",
    "TrialContext",
    "TrialOwnedError",
    "TrialRunner",
]
