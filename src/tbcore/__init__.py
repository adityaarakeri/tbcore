"""tbcore: token-build core plumbing."""

from tbcore.accounting import DEFAULT_LOG_PATH, CallLog
from tbcore.client import (
    APIStatusError,
    Client,
    Completion,
    HttpTransport,
    RateLimitError,
    TbError,
    TransientError,
    Transport,
    TransportRequest,
    Usage,
)
from tbcore.models import REGISTRY, ModelSpec, resolve_model
from tbcore.pool import ExecutionPool, WorkItem, WorkStore, enqueue

__version__ = "0.1.0"

__all__ = [
    "APIStatusError",
    "DEFAULT_LOG_PATH",
    "ExecutionPool",
    "REGISTRY",
    "CallLog",
    "Client",
    "Completion",
    "HttpTransport",
    "ModelSpec",
    "RateLimitError",
    "Transport",
    "TransportRequest",
    "TransientError",
    "TbError",
    "Usage",
    "WorkItem",
    "WorkStore",
    "__version__",
    "enqueue",
    "resolve_model",
]
