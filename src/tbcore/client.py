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
    """Default transport: JSON POST against the Anthropic Messages API."""

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


class Client:
    """Single entry point for model calls: sends requests and logs every one.

    ``model`` may be a short registry name (``sonnet``) or a raw model id.
    When ``cache_prefix`` is passed to :meth:`complete`, the prefix is sent as a
    cacheable system block so repeated calls with the same prefix hit the
    provider's prompt cache.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        transport: Transport | None = None,
        log: CallLog | None = None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        max_tokens: int = 1024,
    ) -> None:
        self.spec: ModelSpec = resolve_model(model)
        if api_key is None:
            api_key = os.environ.get(self.spec.api_key_env)
        if not api_key:
            raise TbError(f"no API key: pass api_key= or set {self.spec.api_key_env}")
        self.api_key = api_key
        self.max_tokens_default = max_tokens
        if transport is not None:
            self.transport = transport
        else:
            self.transport = HttpTransport(
                base_url=base_url or "https://api.anthropic.com", timeout_s=timeout_s
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
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        start = time.perf_counter()
        error: str | None = None
        completion: Completion | None = None
        try:
            raw = self.transport.send(TransportRequest(payload=payload, headers=headers))
            latency_s = time.perf_counter() - start
            completion = _parse_response(raw, latency_s, self.spec.model_id)
            return completion
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            latency_s = time.perf_counter() - start
            self._record(payload, completion, latency_s, bool(cache_prefix), error)

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
