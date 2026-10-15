"""Allowlisting egress proxy used for sandbox networking.

The sandbox network is Docker-internal (no direct outbound routing). Egress is
possible only through this proxy, which has two listeners:

- ``LISTEN_PORT`` (3128): an HTTP/CONNECT forward proxy for the hosts listed in
  ``TBCORE_ALLOW_HOSTS``. Hosts that also appear in ``TBCORE_INJECT`` are refused
  here, so sandboxed code cannot reach them with credentials of its own.
- ``REVERSE_PORT`` (8080, only when ``TBCORE_INJECT`` is set): a plain-HTTP reverse
  proxy. ``http://proxy:8080/<host>/<path>`` is forwarded over verified HTTPS to
  ``https://<host>/<path>`` with the client's auth headers dropped and the configured
  headers for ``<host>`` set. The real credentials never enter the sandbox.

``TBCORE_INJECT`` is JSON: ``{"host": {"Header-Name": "value"}}``.
"""

import http.client
import json
import os
import re
import select
import socket
import ssl
import sys
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import StreamRequestHandler, ThreadingTCPServer
from urllib.parse import urlsplit

LISTEN_HOST = "0.0.0.0"
# Both ports are unprivileged so the proxy can run as a non-root user.
LISTEN_PORT = 3128
REVERSE_PORT = 8080
IDLE_TIMEOUT_S = 120
UPSTREAM_CONNECT_TIMEOUT_S = 10
UPSTREAM_READ_TIMEOUT_S = 300
MAX_REQUEST_BODY = 64 * 1024 * 1024
STREAM_CHUNK = 65536
MAX_TELEMETRY_BYTES = 256 * 1024
TELEMETRY_PREFIX = "TBCORE_USAGE "
_TELEMETRY_LOCK = threading.Lock()
FAIL_REASONS = {
    400: "Bad Request",
    403: "Forbidden",
    405: "Method Not Allowed",
    411: "Length Required",
    413: "Payload Too Large",
    429: "Too Many Requests",
    502: "Bad Gateway",
}
ALLOWED_METHODS = {"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"}
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
# Never forwarded from the client: the proxy supplies Host and Content-Length itself and
# overwrites authentication.
CLIENT_AUTH_HEADERS = frozenset({"authorization", "x-api-key", "host", "content-length"})
_REVERSE_TARGET = re.compile(r"^/([^/?#]+)([/?].*)?$")

ConnectionFactory = Callable[[str], http.client.HTTPConnection]


class ProxyBudgetExceeded(RuntimeError):
    """The proxy cannot admit another model request."""


class ProxyBudget:
    """Per-proxy best-effort admission; delayed usage can overshoot a limit."""

    def __init__(
        self,
        *,
        max_model_calls: int | None = None,
        max_total_tokens: int | None = None,
        max_cost_usd: Decimal | None = None,
        rates: Mapping[str, tuple[str, str, str, str]] | None = None,
    ) -> None:
        if max_model_calls is not None and (
            type(max_model_calls) is not int or max_model_calls <= 0
        ):
            raise ValueError("max_model_calls must be positive")
        if max_total_tokens is not None and (
            type(max_total_tokens) is not int or max_total_tokens <= 0
        ):
            raise ValueError("max_total_tokens must be positive")
        if max_cost_usd is not None and (not max_cost_usd.is_finite() or max_cost_usd <= 0):
            raise ValueError("max_cost_usd must be positive and finite")
        self.max_model_calls = max_model_calls
        self.max_total_tokens = max_total_tokens
        self.max_cost_usd = max_cost_usd
        self.rates = rates or {}
        self._lock = threading.RLock()
        self._calls: set[str] = set()
        self._in_flight: set[str] = set()
        self._total_tokens = 0
        self._cost_usd = Decimal(0)
        self._uncertain = False

    def begin(self, call_id: str, model: str) -> None:
        with self._lock:
            if self._uncertain:
                raise ProxyBudgetExceeded("unknown usage under proxy limit")
            if self.max_cost_usd is not None and model not in self.rates:
                raise ProxyBudgetExceeded("unpriced model under spend limit")
            if self.max_model_calls is not None and len(self._calls) >= self.max_model_calls:
                raise ProxyBudgetExceeded("proxy model-call limit reached")
            if self.max_total_tokens is not None and self._total_tokens >= self.max_total_tokens:
                raise ProxyBudgetExceeded("proxy token limit reached")
            if self.max_cost_usd is not None and self._cost_usd >= self.max_cost_usd:
                raise ProxyBudgetExceeded("proxy cost limit reached")
            self._calls.add(call_id)
            self._in_flight.add(call_id)

    def finish(self, call_id: str, observation: Mapping[str, object]) -> None:
        with self._lock:
            if call_id not in self._in_flight:
                raise ValueError("proxy call is not in flight")
            self._in_flight.remove(call_id)
            fields = (
                observation.get("input_tokens"),
                observation.get("output_tokens"),
                observation.get("cache_read_input_tokens"),
                observation.get("cache_creation_input_tokens"),
            )
            tokens: list[int] = []
            for value in fields:
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    break
                tokens.append(value)
            if not observation.get("usage_complete") or len(tokens) != len(fields):
                if self.max_total_tokens is not None or self.max_cost_usd is not None:
                    self._uncertain = True
                return
            self._total_tokens += sum(tokens)
            model = observation.get("model")
            if self.max_cost_usd is not None:
                rates = self.rates.get(str(model))
                if rates is None:
                    self._uncertain = True
                    return
                self._cost_usd += sum(
                    Decimal(token) * Decimal(rate)
                    for token, rate in zip(tokens, rates, strict=True)
                ) / Decimal(1_000_000)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "model_calls": len(self._calls),
                "in_flight": len(self._in_flight),
                "total_tokens": self._total_tokens,
                "cost_usd": self._cost_usd,
                "usage_uncertain": self._uncertain,
                "inflight_can_exceed_limit": bool(self._in_flight)
                and (self.max_total_tokens is not None or self.max_cost_usd is not None),
            }


class UsageTap:
    """Inspect bounded response chunks without delaying or altering relay bytes."""

    def __init__(
        self, provider: str, model: str, *, stream: bool, max_bytes: int = MAX_TELEMETRY_BYTES
    ) -> None:
        self.provider = provider
        self.model = model
        self.stream = stream
        self.max_bytes = max_bytes
        self.buffer = bytearray()
        self.usage: dict[str, int] | None = None
        self.request_id: str | None = None
        self.terminal = False
        self.overflow = False
        self.malformed = False

    def feed(self, data: bytes) -> None:
        if self.overflow or self.malformed:
            return
        if len(self.buffer) + len(data) > self.max_bytes:
            self.buffer.clear()
            self.overflow = True
            return
        self.buffer.extend(data)
        if self.stream:
            self._consume_frames()

    def _consume_frames(self) -> None:
        while True:
            original = bytes(self.buffer)
            lf = original.find(b"\n\n")
            crlf = original.find(b"\r\n\r\n")
            endings = [(lf, 2), (crlf, 4)]
            endings = [(index, width) for index, width in endings if index >= 0]
            if not endings:
                return
            end, width = min(endings)
            frame = original[:end].replace(b"\r\n", b"\n")
            del self.buffer[: end + width]
            data = b"\n".join(
                line[5:].lstrip() for line in frame.split(b"\n") if line.startswith(b"data:")
            )
            if not data or data == b"[DONE]":
                continue
            try:
                event = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                self.malformed = True
                return
            if isinstance(event, dict):
                self._observe(event)

    def _observe(self, event: Mapping[str, object]) -> None:
        kind = event.get("type")
        if self.provider == "openai" and kind == "response.completed":
            response = event.get("response")
            if isinstance(response, dict):
                self._take_response(response)
            self.terminal = True
        elif self.provider == "anthropic":
            if kind == "message_start":
                message = event.get("message")
                if isinstance(message, dict):
                    self._take_response(message)
            elif kind == "message_delta":
                usage = event.get("usage")
                if isinstance(usage, dict):
                    parsed = self._parse_usage(usage, partial=True)
                    if parsed is not None:
                        self.usage = {**(self.usage or {}), **parsed}
            elif kind == "message_stop":
                self.terminal = True

    @staticmethod
    def _nonnegative(value: object) -> int | None:
        return value if type(value) is int and value >= 0 else None

    def _parse_usage(
        self, raw: Mapping[str, object], *, partial: bool = False
    ) -> dict[str, int] | None:
        if self.provider == "openai":
            total = self._nonnegative(raw.get("input_tokens"))
            output = self._nonnegative(raw.get("output_tokens"))
            details = raw.get("input_tokens_details")
            cached = (
                self._nonnegative(details.get("cached_tokens")) if isinstance(details, dict) else 0
            )
            if total is None or output is None or cached is None or cached > total:
                return None
            return {
                "input_tokens": total - cached,
                "output_tokens": output,
                "cache_read_input_tokens": cached,
                "cache_creation_input_tokens": 0,
            }
        if not partial and ("input_tokens" not in raw or "output_tokens" not in raw):
            return None
        result: dict[str, int] = {}
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            value = raw.get(name)
            if value is None and partial:
                continue
            parsed = self._nonnegative(value) if value is not None else 0
            if parsed is None:
                return None
            result[name] = parsed
        return result

    def _take_response(self, response: Mapping[str, object]) -> None:
        request_id = response.get("id")
        model = response.get("model")
        if isinstance(request_id, str):
            self.request_id = request_id
        if isinstance(model, str):
            self.model = model
        raw = response.get("usage")
        if isinstance(raw, dict):
            self.usage = self._parse_usage(raw)

    def finish(self, status_code: int, *, complete: bool) -> dict[str, object]:
        if not self.stream and not self.overflow and complete and status_code < 400:
            try:
                response = json.loads(self.buffer)
            except (ValueError, UnicodeDecodeError):
                self.malformed = True
            else:
                if isinstance(response, dict):
                    self._take_response(response)
                    self.terminal = True
        known = (
            status_code < 400
            and complete
            and self.terminal
            and not self.overflow
            and not self.malformed
            and self.usage is not None
            and "input_tokens" in self.usage
            and "output_tokens" in self.usage
        )
        usage = self.usage or {}
        return {
            "provider": self.provider,
            "model": self.model,
            "provider_request_id": self.request_id,
            "status": "completed" if known else "failed" if status_code >= 400 else "incomplete",
            "usage_complete": known,
            "input_tokens": usage.get("input_tokens") if known else None,
            "output_tokens": usage.get("output_tokens") if known else None,
            "cache_read_input_tokens": usage.get("cache_read_input_tokens") if known else None,
            "cache_creation_input_tokens": (
                usage.get("cache_creation_input_tokens") if known else None
            ),
            "telemetry_reason": (
                "complete"
                if known
                else "overflow"
                if self.overflow
                else "malformed"
                if self.malformed
                else "http-error"
                if status_code >= 400
                else "missing-terminal-or-usage"
            ),
        }


def _https_factory() -> ConnectionFactory:
    context = ssl.create_default_context()

    def connect(host: str) -> http.client.HTTPConnection:
        return http.client.HTTPSConnection(
            host, 443, context=context, timeout=UPSTREAM_CONNECT_TIMEOUT_S
        )

    return connect


@dataclass
class ProxyConfig:
    allowed_hosts: frozenset[str] = frozenset()
    inject: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    # Tests replace this to point the reverse proxy at a local fake upstream.
    connect: ConnectionFactory = field(default_factory=_https_factory)
    # The factory's timeout only bounds connecting; once connected, reads may wait this
    # long (non-streaming model calls routinely take well over the connect timeout).
    read_timeout_s: float = UPSTREAM_READ_TIMEOUT_S
    run_id: str | None = None
    attempt_id: str | None = None
    emit: Callable[[dict[str, object]], None] | None = None
    budget: ProxyBudget | None = None

    def __post_init__(self) -> None:
        if (self.run_id is None) != (self.attempt_id is None):
            raise ValueError("proxy telemetry requires both run_id and attempt_id")
        if self.budget is not None and self.run_id is None:
            raise ValueError("proxy budget requires host-bound telemetry identity")

    def forward_allowed(self, authority: str) -> bool:
        """Forward/CONNECT path: allowlisted hosts, minus anything credential-injected."""
        host = authority.rsplit("@", 1)[-1]
        host = host.split(":", 1)[0].strip("[]").lower()
        return host in self.allowed_hosts and host not in self.inject


def parse_hosts(raw: str) -> frozenset[str]:
    return frozenset(host.strip().lower() for host in raw.split(",") if host.strip())


def parse_inject(raw: str) -> dict[str, dict[str, str]]:
    """Parse and validate ``TBCORE_INJECT``; raises ValueError on a malformed value."""
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("TBCORE_INJECT must be a JSON object")
    result: dict[str, dict[str, str]] = {}
    for host, headers in data.items():
        if not isinstance(host, str) or not isinstance(headers, dict):
            raise ValueError("TBCORE_INJECT must map host names to header objects")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
            raise ValueError("TBCORE_INJECT headers must be strings")
        result[host.lower()] = dict(headers)
    return result


def parse_upstream_overrides(raw: str) -> ConnectionFactory:
    """Host-only offline routes to single-label containers on the internal network."""
    if not raw:
        return _https_factory()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("upstream overrides must be a mapping")
    routes: dict[str, tuple[str, int]] = {}
    for host, target in data.items():
        if host not in {"api.openai.com", "api.anthropic.com"} or not isinstance(target, str):
            raise ValueError("invalid upstream host override")
        url = urlsplit(target)
        try:
            port = url.port
        except ValueError as exc:
            raise ValueError("invalid upstream port") from exc
        alias = url.hostname or ""
        if (
            url.scheme != "http"
            or not re.fullmatch(r"[a-z][a-z0-9-]*", alias)
            or port is None
            or not 1 <= port <= 65535
            or url.username is not None
            or url.password is not None
            or url.path
            or url.query
            or url.fragment
        ):
            raise ValueError("invalid upstream destination")
        routes[host] = alias, port
    default = _https_factory()

    def connect(host: str) -> http.client.HTTPConnection:
        route = routes.get(host)
        if route is None:
            return default(host)
        return http.client.HTTPConnection(*route, timeout=UPSTREAM_CONNECT_TIMEOUT_S)

    return connect


def parse_budget(raw: str) -> ProxyBudget | None:
    """Parse the host-owned proxy budget; reject unknown or invalid settings."""
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) - {
        "max_model_calls",
        "max_total_tokens",
        "max_cost_usd",
        "rates",
    }:
        raise ValueError("invalid proxy budget configuration")
    for name in ("max_model_calls", "max_total_tokens"):
        value = data.get(name)
        if value is not None and type(value) is not int:
            raise ValueError(f"{name} must be an integer")
    raw_rates = data.get("rates", {})
    if not isinstance(raw_rates, dict):
        raise ValueError("proxy rates must be a mapping")
    rates: dict[str, tuple[str, str, str, str]] = {}
    for model, values in raw_rates.items():
        if not isinstance(model, str) or not isinstance(values, list) or len(values) != 4:
            raise ValueError("proxy rate entry is invalid")
        if not all(isinstance(value, str) for value in values):
            raise ValueError("proxy rates must be decimal strings")
        if any(not Decimal(value).is_finite() or Decimal(value) < 0 for value in values):
            raise ValueError("proxy rates must be finite and nonnegative")
        rates[model] = (values[0], values[1], values[2], values[3])
    raw_cost = data.get("max_cost_usd")
    if raw_cost is not None and not isinstance(raw_cost, str):
        raise ValueError("max_cost_usd must be a decimal string")
    return ProxyBudget(
        max_model_calls=data.get("max_model_calls"),
        max_total_tokens=data.get("max_total_tokens"),
        max_cost_usd=Decimal(raw_cost) if raw_cost is not None else None,
        rates=rates,
    )


def _write_status(handler: StreamRequestHandler, code: int) -> None:
    reason = FAIL_REASONS.get(code, "Error")
    body = (f"HTTP/1.1 {code} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").encode(
        "latin-1"
    )
    try:
        handler.wfile.write(body)
        handler.wfile.flush()
    except OSError:
        pass


class ForwardServer(ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: ProxyConfig) -> None:
        self.config = config
        super().__init__(address, ForwardHandler)


class ReverseServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: ProxyConfig) -> None:
        self.config = config
        super().__init__(address, ReverseHandler)


class ForwardHandler(StreamRequestHandler):
    server: ForwardServer

    def handle(self) -> None:
        try:
            request_line = self.rfile.readline(65536).decode("latin-1").strip()
        except OSError:
            return
        parts = request_line.split()
        if len(parts) < 3:
            self._fail(400)
            return
        method, target, version = parts[0].upper(), parts[1], parts[2]
        if method == "CONNECT":
            self._tunnel(target)
        elif method in ALLOWED_METHODS:
            self._forward_http(method, target, version)
        else:
            self._fail(405)

    def _tunnel(self, authority: str) -> None:
        if not self.server.config.forward_allowed(authority):
            self._fail(403)
            return
        host, _, port = authority.partition(":")
        try:
            upstream = socket.create_connection((host, int(port or 443)), timeout=10)
        except (OSError, ValueError):
            self._fail(502)
            return
        try:
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.wfile.flush()
            self._relay(upstream)
        finally:
            upstream.close()

    def _forward_http(self, method: str, target: str, version: str) -> None:
        headers = {}
        while True:
            line = self.rfile.readline(65536)
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode("latin-1").partition(":")
            headers[name.strip().lower()] = value.strip()
        split = urlsplit(target)
        host = (split.hostname or "").lower()
        if not host or not self.server.config.forward_allowed(host):
            self._fail(403)
            return
        path = split.path or "/"
        if split.query:
            path += "?" + split.query
        request_lines = [f"{method} {path} {version}", f"Host: {split.netloc}"]
        request_lines += [
            f"{name.title()}: {value}"
            for name, value in headers.items()
            if name not in ("host", "proxy-authorization", "proxy-connection")
        ]
        payload = ("\r\n".join(request_lines) + "\r\n\r\n").encode("latin-1")
        try:
            upstream = socket.create_connection((host, split.port or 80), timeout=10)
        except OSError:
            self._fail(502)
            return
        try:
            upstream.sendall(payload)
            self._relay(upstream)
        finally:
            upstream.close()

    def _relay(self, upstream: socket.socket) -> None:
        peers = {self.connection: upstream, upstream: self.connection}
        try:
            while True:
                readable, _, _ = select.select(list(peers), [], [], IDLE_TIMEOUT_S)
                if not readable:
                    return
                for sock in readable:
                    data = sock.recv(65536)
                    if not data:
                        return
                    peers[sock].sendall(data)
        except OSError:
            return

    def _fail(self, code: int) -> None:
        _write_status(self, code)


def _connection_tokens(value: str | None) -> set[str]:
    return {token.strip().lower() for token in (value or "").split(",") if token.strip()}


class ReverseHandler(BaseHTTPRequestHandler):
    server: ReverseServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        # Paths and queries can carry data; keep the proxy log quiet.
        return

    def _fail(self, code: int) -> None:
        self.close_connection = True
        _write_status(self, code)

    def _handle(self) -> None:
        self.close_connection = True
        config = self.server.config
        match = _REVERSE_TARGET.match(self.path)
        if match is None:
            self._fail(400)
            return
        host = match.group(1).lower()
        injected = config.inject.get(host)
        if injected is None:
            self._fail(403)
            return
        upstream_path = match.group(2) or "/"
        if upstream_path.startswith("?"):
            upstream_path = "/" + upstream_path
        supported = {
            "api.openai.com": "/v1/responses",
            "api.anthropic.com": "/v1/messages",
        }
        if host in supported and (self.command != "POST" or upstream_path != supported[host]):
            self._fail(403)
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self._fail(411)
            return
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) > 1 or (lengths and not lengths[0].strip().isdigit()):
            self._fail(400)
            return
        length = int(lengths[0]) if lengths else 0
        if length > MAX_REQUEST_BODY:
            self._fail(413)
            return
        body = self.rfile.read(length) if length else b""
        if len(body) != length:
            self._fail(400)
            return

        dropped = (
            HOP_BY_HOP
            | CLIENT_AUTH_HEADERS
            | {"x-tbcore-run-id", "x-tbcore-attempt-id"}
            | _connection_tokens(self.headers.get("Connection"))
            | {name.lower() for name in injected}
        )
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in dropped and not name.lower().startswith("proxy-")
        }
        headers["Host"] = host
        headers.update(injected)
        self._forward(host, upstream_path, headers, body or None)

    def _telemetry(self, host: str, path: str, body: bytes | None) -> dict[str, object] | None:
        config = self.server.config
        if config.run_id is None or config.attempt_id is None:
            return None
        provider = (
            "openai"
            if host == "api.openai.com" and path == "/v1/responses"
            else "anthropic"
            if host == "api.anthropic.com" and path == "/v1/messages"
            else None
        )
        model = None
        if body is not None and len(body) <= MAX_TELEMETRY_BYTES:
            try:
                request = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                request = None
            if isinstance(request, dict) and isinstance(request.get("model"), str):
                model = request["model"]
        return {
            "run_id": config.run_id,
            "attempt_id": config.attempt_id,
            "call_id": uuid.uuid4().hex,
            "provider": provider,
            "model": model,
        }

    def _emit(self, observation: dict[str, object]) -> None:
        sink = self.server.config.emit
        if sink is not None:
            sink(observation)
        else:
            with _TELEMETRY_LOCK:
                print(TELEMETRY_PREFIX + json.dumps(observation, sort_keys=True), flush=True)

    def _forward(
        self, host: str, path: str, headers: Mapping[str, str], body: bytes | None
    ) -> None:
        config = self.server.config
        observation = self._telemetry(host, path, body)
        if (
            observation is not None
            and observation["provider"] is not None
            and config.budget is not None
        ):
            try:
                config.budget.begin(str(observation["call_id"]), str(observation["model"] or ""))
            except ProxyBudgetExceeded as exc:
                self._emit(
                    {
                        **observation,
                        "status": "denied",
                        "usage_complete": False,
                        "telemetry_reason": str(exc),
                    }
                )
                self._fail(429)
                return
            snapshot = config.budget.snapshot()
            observation["budget_in_flight"] = snapshot["in_flight"]
            observation["budget_inflight_can_exceed_limit"] = snapshot["inflight_can_exceed_limit"]
        upstream = config.connect(host)
        try:
            try:
                upstream.connect()
                if upstream.sock is not None:
                    upstream.sock.settimeout(config.read_timeout_s)
                upstream.request(self.command, path, body=body, headers=dict(headers))
                response = upstream.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                print(f"upstream error host={host}: {type(exc).__name__}", file=sys.stderr)
                if observation is not None:
                    failed = {
                        **observation,
                        "status": "failed",
                        "usage_complete": False,
                        "telemetry_reason": "upstream-error",
                    }
                    if config.budget is not None and observation["provider"] is not None:
                        config.budget.finish(str(observation["call_id"]), failed)
                    self._emit(failed)
                self._fail(502)
                return
            self._relay_response(response, observation)
        finally:
            upstream.close()

    def _relay_response(
        self, response: http.client.HTTPResponse, observation: dict[str, object] | None = None
    ) -> None:
        dropped = HOP_BY_HOP | _connection_tokens(response.getheader("Connection"))
        status = response.status
        has_body = self.command != "HEAD" and status >= 200 and status not in (204, 304)
        length = response.getheader("Content-Length")
        tap = None
        if observation is not None and observation["provider"] is not None:
            tap = UsageTap(
                str(observation["provider"]),
                str(observation["model"] or ""),
                stream="text/event-stream" in (response.getheader("Content-Type") or "").lower(),
            )
        chunked = has_body and length is None and self.request_version == "HTTP/1.1"
        self.send_response_only(status, response.reason)
        for name, value in response.getheaders():
            if name.lower() not in dropped:
                self.send_header(name, value)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.flush()
        if not has_body:
            if observation is not None:
                finished = {
                    **observation,
                    "status": "unmetered",
                    "usage_complete": False,
                    "telemetry_reason": "no-body",
                }
                budget = self.server.config.budget
                if budget is not None and observation["provider"] is not None:
                    budget.finish(str(observation["call_id"]), finished)
                self._emit(finished)
            return
        complete = False
        try:
            while True:
                data = response.read1(STREAM_CHUNK)
                if not data:
                    break
                if tap is not None:
                    tap.feed(data)
                if chunked:
                    self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
                else:
                    self.wfile.write(data)
                self.wfile.flush()
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            complete = True
        except (OSError, http.client.HTTPException) as exc:
            # Headers are already sent; dropping the connection signals truncation.
            print(f"stream aborted: {type(exc).__name__}", file=sys.stderr)
        finally:
            if observation is not None:
                if tap is None:
                    finished = {
                        **observation,
                        "status": "unmetered",
                        "usage_complete": False,
                        "telemetry_reason": "unsupported-api",
                    }
                else:
                    finished = {**observation, **tap.finish(status, complete=complete)}
                budget = self.server.config.budget
                if budget is not None and observation["provider"] is not None:
                    budget.finish(str(observation["call_id"]), finished)
                self._emit(finished)

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _handle


def start_servers(
    config: ProxyConfig,
    *,
    host: str = LISTEN_HOST,
    forward_port: int = LISTEN_PORT,
    reverse_port: int = REVERSE_PORT,
) -> tuple[ForwardServer, ReverseServer | None]:
    """Bind both listeners (reverse only when hosts are injected) and serve in threads."""
    forward = ForwardServer((host, forward_port), config)
    reverse = ReverseServer((host, reverse_port), config) if config.inject else None
    for server in (forward, reverse):
        if server is not None:
            threading.Thread(target=server.serve_forever, daemon=True).start()
    return forward, reverse


def main() -> None:
    config = ProxyConfig(
        allowed_hosts=parse_hosts(os.environ.get("TBCORE_ALLOW_HOSTS", "")),
        inject=parse_inject(os.environ.get("TBCORE_INJECT", "")),
        run_id=os.environ.get("TBCORE_RUN_ID") or None,
        attempt_id=os.environ.get("TBCORE_ATTEMPT_ID") or None,
        budget=parse_budget(os.environ.get("TBCORE_PROXY_BUDGET", "")),
        connect=parse_upstream_overrides(os.environ.get("TBCORE_UPSTREAM_OVERRIDES", "")),
    )
    forward, reverse = start_servers(config)
    # Hosts only; injected header values must never be logged.
    print(
        f"TBCORE_PROXY_READY allow={sorted(config.allowed_hosts - set(config.inject))} "
        f"inject={sorted(config.inject)}",
        flush=True,
    )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        forward.server_close()
        if reverse is not None:
            reverse.server_close()


if __name__ == "__main__":
    main()
