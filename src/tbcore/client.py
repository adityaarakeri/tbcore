"""Request client with usage accounting and prompt-prefix caching."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from tbcore.accounting import CallLog
from tbcore.budget import RunBudget
from tbcore.config import ModelConfig
from tbcore.experiment_store import ExperimentStore
from tbcore.experiment_usage import CallContext
from tbcore.models import ModelSpec, resolve_model

logger = logging.getLogger(__name__)


class TbError(Exception):
    """Base error for tbcore clients."""


class RetryableError(TbError):
    """Base for retryable failures; ``retry_after`` is the server hint in seconds, if any."""

    def __init__(self, *args: object, retry_after: float | None = None) -> None:
        super().__init__(*args)
        self.retry_after = retry_after


class RateLimitError(RetryableError):
    """The provider rejected the call with a rate limit (HTTP 429)."""


class TransientError(RetryableError):
    """A retryable transport or server-side failure (5xx, network)."""


class APIStatusError(TbError):
    """A non-retryable API-level failure."""


@dataclass(frozen=True)
class TransportRequest:
    payload: Mapping[str, Any]
    headers: Mapping[str, str]


class Transport(Protocol):
    def send(self, request: TransportRequest) -> dict[str, Any]: ...


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP date) into seconds.

    Returns None when the header is absent or unparseable; past dates give 0.0.
    """
    if value is None or not value.strip():
        return None
    text = value.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


def _post_json(url: str, request: TransportRequest, timeout_s: float) -> dict[str, Any]:
    """POST the request payload as JSON and map HTTP/network failures to tbcore errors."""
    http_req = urllib.request.Request(
        url,
        data=json.dumps(request.payload).encode("utf-8"),
        headers=dict(request.headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_req, timeout=timeout_s) as resp:
            result: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return result
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        retry_after = parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
        if exc.code == 429:
            raise RateLimitError(f"rate limited (429): {detail}", retry_after=retry_after) from exc
        if exc.code >= 500:
            raise TransientError(
                f"server error ({exc.code}): {detail}", retry_after=retry_after
            ) from exc
        raise APIStatusError(f"API error ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TransientError(f"transport failure: {exc}") from exc


class HttpTransport:
    """Anthropic wire: JSON POST against the Messages API."""

    def __init__(
        self, base_url: str = "https://api.anthropic.com", timeout_s: float = 60.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def send(self, request: TransportRequest) -> dict[str, Any]:
        return _post_json(f"{self.base_url}/v1/messages", request, self.timeout_s)


class OpenAICompatTransport:
    """OpenAI-compatible wire: POST ``/v1/chat/completions``.

    Speaks to Ollama, LM Studio, vLLM, OpenAI itself, and any other server
    exposing the chat-completions shape. A base URL ending in ``/v1`` is used
    verbatim; otherwise ``/v1`` is appended. An API key is optional and sent
    as a Bearer header when present.
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def send(self, request: TransportRequest) -> dict[str, Any]:
        suffix = "/chat/completions" if self.base_url.endswith("/v1") else "/v1/chat/completions"
        return _post_json(f"{self.base_url}{suffix}", request, self.timeout_s)


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage
    latency_s: float
    model: str = ""
    stop_reason: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)
    usage_known: bool = True


def _message(message: Mapping[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    if role not in ("user", "assistant"):
        raise TbError(f"message role must be 'user' or 'assistant', got {role!r}")
    if "content" not in message:
        raise TbError("message missing 'content'")
    return {"role": role, "content": message["content"]}


_Parser = Callable[[Mapping[str, Any], float, str], Completion]


def _parse_response(raw: Mapping[str, Any], latency_s: float, model_id: str) -> Completion:
    text = "".join(
        block.get("text", "")
        for block in raw.get("content", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    usage_raw = raw.get("usage") or {}
    usage_known = isinstance(usage_raw, Mapping) and all(
        field in usage_raw for field in ("input_tokens", "output_tokens")
    )
    usage = Usage(
        input_tokens=int(usage_raw.get("input_tokens", 0)),
        output_tokens=int(usage_raw.get("output_tokens", 0)),
        cache_read_input_tokens=int(usage_raw.get("cache_read_input_tokens", 0)),
        cache_creation_input_tokens=int(usage_raw.get("cache_creation_input_tokens", 0)),
    )
    return Completion(
        text=text,
        usage=usage,
        latency_s=latency_s,
        model=str(raw.get("model", model_id)),
        stop_reason=raw.get("stop_reason"),
        raw=dict(raw),
        usage_known=usage_known,
    )


def _parse_openai_response(raw: Mapping[str, Any], latency_s: float, model_id: str) -> Completion:
    choices = raw.get("choices") or []
    message = (choices[0] or {}).get("message", {}) if choices else {}
    usage_raw = raw.get("usage") or {}
    usage_known = isinstance(usage_raw, Mapping) and all(
        field in usage_raw for field in ("prompt_tokens", "completion_tokens")
    )
    prompt_details = usage_raw.get("prompt_tokens_details") or {}
    cached = int(prompt_details.get("cached_tokens", 0))
    # prompt_tokens includes cached tokens; split them out to match the Anthropic wire,
    # where input_tokens excludes cache reads.
    usage = Usage(
        input_tokens=max(0, int(usage_raw.get("prompt_tokens", 0)) - cached),
        output_tokens=int(usage_raw.get("completion_tokens", 0)),
        cache_read_input_tokens=cached,
    )
    return Completion(
        text=str(message.get("content") or ""),
        usage=usage,
        latency_s=latency_s,
        model=str(raw.get("model", model_id)),
        stop_reason=(choices[0] or {}).get("finish_reason") if choices else None,
        raw=dict(raw),
        usage_known=usage_known,
    )


class Client:
    """Single entry point for model calls: sends requests and logs every one.

    ``model`` may be a short registry name (``sonnet``), a raw model id, or
    an already-built :class:`~tbcore.models.ModelSpec` (use
    :func:`tbcore.models.make_model_spec` for non-Anthropic providers such as
    Ollama). When ``cache_prefix`` is passed to :meth:`complete`, the prefix
    is sent as a cacheable system block so repeated calls with the same
    prefix hit the provider's prompt cache (Anthropic wire only).

    ``config`` (from :func:`tbcore.config.load_model_config`) resolves a string ``model``
    against the file first, then the built-in registry, and raises on an unknown name
    instead of treating it as a raw model id. A ``ModelSpec`` passed as ``model`` always
    wins over the config.
    """

    def __init__(
        self,
        model: str | ModelSpec,
        api_key: str | None = None,
        *,
        transport: Transport | None = None,
        log: CallLog | None = None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        max_tokens: int = 1024,
        experiment_store: ExperimentStore | None = None,
        budget: RunBudget | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        if isinstance(model, ModelSpec):
            self.spec: ModelSpec = model
        elif config is not None:
            self.spec = config.resolve(model)
        else:
            self.spec = resolve_model(model)
        if api_key is None and self.spec.api_key_env:
            api_key = os.environ.get(self.spec.api_key_env)
        if not api_key and self.spec.api_key_required:
            raise TbError(f"no API key: pass api_key= or set {self.spec.api_key_env}")
        self.api_key = api_key
        self.max_tokens_default = max_tokens
        if transport is not None:
            self.transport = transport
        elif self.spec.wire == "openai-compat":
            self.transport = OpenAICompatTransport(
                base_url=base_url or self.spec.base_url or "http://localhost:11434",
                timeout_s=timeout_s,
            )
        else:
            self.transport = HttpTransport(
                base_url=base_url or self.spec.base_url or "https://api.anthropic.com",
                timeout_s=timeout_s,
            )
        self.log = log if log is not None else CallLog()
        self.experiment_store = experiment_store
        self.budget = budget

    def complete(
        self,
        system: str | None,
        messages: Sequence[Mapping[str, Any]],
        cache_prefix: str | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
        call_context: CallContext | None = None,
    ) -> Completion:
        if call_context is not None:
            if self.experiment_store is None:
                raise ValueError("call_context requires experiment_store")
            run = self.experiment_store.get_run(call_context.run_id)
            if run.attempt_id != call_context.attempt_id:
                raise ValueError("call_context attempt does not match the run")
        if self.budget is not None and (
            call_context is None or call_context.run_id != self.budget.run_id
        ):
            raise ValueError("budget requires matching call_context")
        if self.spec.wire == "openai-compat":
            payload, headers, parse = self._build_openai_request(
                system, messages, cache_prefix, max_tokens, tools, temperature
            )
        else:
            payload, headers, parse = self._build_anthropic_request(
                system, messages, cache_prefix, max_tokens, tools, temperature
            )

        if self.budget is not None and call_context is not None:
            self.budget.begin_model_call(call_context.call_id, priced=self.spec.priced)
        start = time.perf_counter()
        error: str | None = None
        completion: Completion | None = None
        try:
            raw = self.transport.send(TransportRequest(payload=payload, headers=headers))
            latency_s = time.perf_counter() - start
            completion = parse(raw, latency_s, self.spec.model_id)
            self._log_success(completion)
            return completion
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("call failed model=%s error=%s", self.spec.model_id, type(exc).__name__)
            raise
        except asyncio.CancelledError:
            error = "CancelledError: call cancelled"
            raise
        finally:
            latency_s = time.perf_counter() - start
            if self.budget is not None and call_context is not None:
                measured = (
                    completion.usage if completion is not None and completion.usage_known else None
                )
                self.budget.finish_model_call(
                    call_context.call_id,
                    input_tokens=measured.input_tokens if measured else None,
                    output_tokens=measured.output_tokens if measured else None,
                    cost_usd=(
                        Decimal(str(self.spec.cost(measured)))
                        if measured is not None and self.spec.priced
                        else None
                    ),
                    cache_read_input_tokens=measured.cache_read_input_tokens if measured else 0,
                    cache_creation_input_tokens=(
                        measured.cache_creation_input_tokens if measured else 0
                    ),
                )
            self._record(payload, completion, latency_s, bool(cache_prefix), error, call_context)

    def _log_success(self, completion: Completion) -> None:
        if not completion.usage_known:
            logger.info(
                "call ok model=%s tokens=unknown latency=%.2fs",
                self.spec.model_id,
                completion.latency_s,
            )
            return
        logger.info(
            "call ok model=%s in=%d out=%d latency=%.2fs",
            self.spec.model_id,
            completion.usage.input_tokens,
            completion.usage.output_tokens,
            completion.latency_s,
        )

    def _build_anthropic_request(
        self,
        system: str | None,
        messages: Sequence[Mapping[str, Any]],
        cache_prefix: str | None,
        max_tokens: int | None,
        tools: Sequence[Mapping[str, Any]] | None,
        temperature: float | None,
    ) -> tuple[dict[str, Any], dict[str, str], _Parser]:
        payload: dict[str, Any] = {
            "model": self.spec.model_id,
            "max_tokens": max_tokens or self.max_tokens_default,
            "messages": [_message(m) for m in messages],
        }
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]
        if temperature is not None:
            payload["temperature"] = temperature
        system_blocks = []
        if cache_prefix:
            system_blocks.append(
                {
                    "type": "text",
                    "text": cache_prefix,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        if system:
            system_blocks.append({"type": "text", "text": system})
        if system_blocks:
            payload["system"] = system_blocks
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
        }
        return payload, headers, _parse_response

    def _build_openai_request(
        self,
        system: str | None,
        messages: Sequence[Mapping[str, Any]],
        cache_prefix: str | None,
        max_tokens: int | None,
        tools: Sequence[Mapping[str, Any]] | None,
        temperature: float | None,
    ) -> tuple[dict[str, Any], dict[str, str], _Parser]:
        if cache_prefix:
            raise TbError("prompt-prefix caching is only supported on the anthropic wire")
        if tools:
            raise TbError("tools are not yet supported on the openai-compat wire")
        chat_messages: list[dict[str, Any]] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(_message(m) for m in messages)
        payload: dict[str, Any] = {
            "model": self.spec.model_id,
            "max_tokens": max_tokens or self.max_tokens_default,
            "messages": chat_messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return payload, headers, _parse_openai_response

    def _record(
        self,
        payload: Mapping[str, Any],
        completion: Completion | None,
        latency_s: float,
        cached_prefix: bool,
        error: str | None,
        call_context: CallContext | None,
    ) -> None:
        usage = completion.usage if completion is not None and completion.usage_known else None
        # Unpriced specs (local models, unknown ids) log null, never a misleading $0.
        cost_equiv = self.spec.cost(usage) if usage is not None and self.spec.priced else None
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "event": "complete",
            "provider": self.spec.provider,
            "model": self.spec.model_id,
            "prompt_tokens": usage.input_tokens if usage else None,
            "completion_tokens": usage.output_tokens if usage else None,
            "cache_read_input_tokens": usage.cache_read_input_tokens if usage else None,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens if usage else None,
            "latency_s": round(latency_s, 6),
            "cost_equiv": round(cost_equiv, 8) if cost_equiv is not None else None,
            "cached_prefix": cached_prefix,
            "ok": completion is not None,
        }
        if error is not None:
            record["error"] = error
        self.log.append(record)
        if call_context is not None:
            assert self.experiment_store is not None
            raw_id = completion.raw.get("id") if completion is not None else None
            self.experiment_store.append_event(
                call_context.run_id,
                call_context.call_id,
                {
                    "event_type": "model_call",
                    "call_source": "direct_client",
                    "attempt_id": call_context.attempt_id,
                    "status": (
                        "completed"
                        if completion is not None
                        else "cancelled"
                        if error and error.startswith("CancelledError")
                        else "failed"
                    ),
                    "provider": self.spec.provider,
                    "model": self.spec.model_id,
                    "provider_request_id": raw_id if isinstance(raw_id, str) else None,
                    "input_tokens": usage.input_tokens if usage else None,
                    "output_tokens": usage.output_tokens if usage else None,
                    "cache_read_input_tokens": usage.cache_read_input_tokens if usage else None,
                    "cache_creation_input_tokens": (
                        usage.cache_creation_input_tokens if usage else None
                    ),
                    "duration_ms": round(latency_s * 1000),
                },
            )
