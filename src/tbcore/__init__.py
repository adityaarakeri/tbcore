"""tbcore: token-build core plumbing."""

from tbcore.accounting import DEFAULT_LOG_PATH, CallLog
from tbcore.client import (
    APIStatusError,
    Client,
    Completion,
    HttpTransport,
    OpenAICompatTransport,
    RateLimitError,
    TbError,
    TransientError,
    Transport,
    TransportRequest,
    Usage,
)
from tbcore.models import REGISTRY, ModelSpec, make_model_spec, resolve_model
from tbcore.pool import ExecutionPool, WorkItem, WorkStore, enqueue
from tbcore.sandbox import DEFAULT_ALLOW_HOSTS, RunResult, Sandbox, SandboxError, build_image
from tbcore.stats import RegressionResult, Summary, regressed, summary

__version__ = "0.1.0"

__all__ = [
    "APIStatusError",
    "DEFAULT_ALLOW_HOSTS",
    "DEFAULT_LOG_PATH",
    "ExecutionPool",
    "REGISTRY",
    "RegressionResult",
    "RunResult",
    "Sandbox",
    "SandboxError",
    "Summary",
    "CallLog",
    "Client",
    "Completion",
    "HttpTransport",
    "ModelSpec",
    "OpenAICompatTransport",
    "RateLimitError",
    "Transport",
    "TransportRequest",
    "TransientError",
    "TbError",
    "Usage",
    "WorkItem",
    "WorkStore",
    "__version__",
    "build_image",
    "enqueue",
    "make_model_spec",
    "regressed",
    "resolve_model",
    "summary",
]
