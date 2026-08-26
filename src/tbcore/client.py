"""Request client with usage accounting and prompt-prefix caching."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from tbcore.accounting import CallLog
from tbcore.models import ModelSpec, resolve_model


class TbError(Exception):
    """Base error for tbcore clients."""


class RateLimitError(TbError):
    """The provider rejected the call with a rate limit (HTTP 429)."""


class TransientError(TbError):
    """A retryable transport or server-side failure (5xx, network)."""


class APIStatusError(TbError):
    """A non-retryable API-level failure."""


@dataclass(frozen=True)
class TransportRequest:
    payload: Mapping[str, Any]
    headers: Mapping[str, str]


class Transport(Protocol):
    def send(self, request: TransportRequest) -> dict[str, Any]: ...


class HttpTransport:
    """Anthropic wire: JSON POST against the Messages API."""

    def __init__(
        self, base_url: str = "https://api.anthropic.com", timeout_s: float = 60.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def send(self, request: TransportRequest) -> dict[str, Any]:
        body = json.dumps(request.payload).encode("utf-8")
        http_req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=body,
            headers=dict(request.headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                raise RateLimitError(f"rate limited (429): {detail}") from exc
            if exc.code >= 500:
                raise TransientError(f"server error ({exc.code}): {detail}") from exc
            raise APIStatusError(f"API error ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise TransientError(f"transport failure: {exc}") from exc


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
        body = json.dumps(request.payload).encode("utf-8")
        http_req = urllib.request.Request(
            f"{self.base_url}{suffix}",
            data=body,
            headers=dict(request.headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                raise RateLimitError(f"rate limited (429): {detail}") from exc
            if exc.code >= 500:
                raise TransientError(f"server error ({exc.code}): {detail}") from exc
            raise APIStatusError(f"API error ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise TransientError(f"transport failure: {exc}") from exc


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


def _message(message: Mapping[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    if role not in ("user", "assistant"):
        raise TbError(f"message role must be 'user' or 'assistant', got {role!r}")
    if "content" not in message:
        raise TbError("message missing 'content'")
    return {"role": role, "content": message["content"]}


def _parse_response(raw: Mapping[str, Any], latency_s: float, model_id: str) -> Completion:
    text = "".join(
        block.get("text", "")
        for block in raw.get("content", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    usage_raw = raw.get("usage", {})
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
    )


def _parse_openai_response(raw: Mapping[str, Any], latency_s: float, model_id: str) -> Completion:
    choices = raw.get("choices") or []
    message = (choices[0] or {}).get("message", {}) if choices else {}
    usage_raw = raw.get("usage", {})
    prompt_details = usage_raw.get("prompt_tokens_details") or {}
    usage = Usage(
        input_tokens=int(usage_raw.get("prompt_tokens", 0)),
        output_tokens=int(usage_raw.get("completion_tokens", 0)),
        cache_read_input_tokens=int(prompt_details.get("cached_tokens", 0)),
    )
    return Completion(
        text=str(message.get("content") or ""),
        usage=usage,
        latency_s=latency_s,
        model=str(raw.get("model", model_id)),
        stop_reason=(choices[0] or {}).get("finish_reason") if choices else None,
        raw=dict(raw),
    )


class Client:
    """Single entry point for model calls: sends requests and logs every one.

    ``model`` may be a short registry name (``sonnet``), a raw model id, or
    an already-built :class:`~tbcore.models.ModelSpec` (use
    :func:`tbcore.models.make_model_spec` for non-Anthropic providers such as
    Ollama). When ``cache_prefix`` is passed to :meth:`complete`, the prefix
    is sent as a cacheable system block so repeated calls with the same
    prefix hit the provider's prompt cache (Anthropic wire only).
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
    ) -> None:
        self.spec: ModelSpec = model if isinstance(model, ModelSpec) else resolve_model(model)
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
                base_url=self.spec.base_url or "http://localhost:11434",
                timeout_s=timeout_s,
            )
        else:
            self.transport = HttpTransport(
                base_url=base_url or self.spec.base_url or "https://api.anthropic.com",
                timeout_s=timeout_s,
            )
        self.log = log if log is not None else CallLog()

    def complete(
        self,
        system: str | None,
        messages: Sequence[Mapping[str, Any]],
        cache_prefix: str | None = None,
        max_tokens: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> Completion:
        if self.spec.wire == "openai-compat":
            payload, headers, parse = self._build_openai_request(
                system, messages, cache_prefix, max_tokens, tools, temperature
            )
        else:
            payload, headers, parse = self._build_anthropic_request(
                system, messages, cache_prefix, max_tokens, tools, temperature
            )

        start = time.perf_counter()
        error: str | None = None
        completion: Completion | None = None
        try:
            raw = self.transport.send(TransportRequest(payload=payload, headers=headers))
            latency_s = time.perf_counter() - start
            completion = parse(raw, latency_s, self.spec.model_id)
            return completion
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            latency_s = time.perf_counter() - start
            self._record(payload, completion, latency_s, bool(cache_prefix), error)

    def _build_anthropic_request(
        self,
        system: str | None,
        messages: Sequence[Mapping[str, Any]],
        cache_prefix: str | None,
        max_tokens: int | None,
        tools: Sequence[Mapping[str, Any]] | None,
        temperature: float | None,
    ) -> tuple[dict[str, Any], dict[str, str], Any]:
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
    ) -> tuple[dict[str, Any], dict[str, str], Any]:
        if cache_prefix:
            raise TbError("prompt-prefix caching is only supported on the anthropic wire")
        if tools:
            raise TbError("tools are not yet supported on the openai-compat wire")
        chat_messages: list[dict[str, Any]] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(_message(m) for m in messages)
        payload = {
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
    ) -> None:
        usage = completion.usage if completion is not None else None
        cost_equiv = self.spec.cost(usage) if usage is not None else None
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
