"""In-process tests for the bundled proxy against a local fake HTTP upstream."""

import http.client
import importlib.util
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tbcore.sandbox import image_context

_spec = importlib.util.spec_from_file_location(
    "tbcore_proxy_under_test", image_context("proxy") / "proxy.py"
)
assert _spec is not None and _spec.loader is not None
proxy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proxy)

HOST = "api.example.test"
INJECTED = {"x-api-key": "REAL-KEY", "anthropic-version": "2023-06-01"}


class Upstream:
    """Fake upstream that records requests and can stream a gated response."""

    def __init__(self):
        self.requests = []
        self.gate = threading.Event()
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):
                pass

            def _record(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                upstream.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": [(k.lower(), v) for k, v in self.headers.items()],
                        "body": body,
                    }
                )

            def do_GET(self):
                self._record()
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    self.wfile.write(b"5\r\nfirst\r\n")
                    self.wfile.flush()
                    upstream.gate.wait(timeout=10)
                    self.wfile.write(b"6\r\nsecond\r\n0\r\n\r\n")
                    self.wfile.flush()
                    return
                if self.path.startswith("/slow"):
                    time.sleep(1.0)
                payload = b"hello"
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Upstream", "yes")
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                self._record()
                if self.path == "/v1/responses":
                    request = json.loads(upstream.requests[-1]["body"])
                    if not request.get("stream"):
                        payload = json.dumps(
                            {
                                "id": "resp-local",
                                "model": request["model"],
                                "usage": {
                                    "input_tokens": 12,
                                    "output_tokens": 3,
                                    "input_tokens_details": {"cached_tokens": 2},
                                },
                            }
                        ).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    first = b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
                    second = (
                        b'data: {"type":"response.completed","response":'
                        b'{"id":"resp-stream","model":"gpt-6.1-sol",'
                        b'"usage":{"input_tokens":15,"output_tokens":2,'
                        b'"input_tokens_details":{"cached_tokens":5}}}}\n\n'
                    )
                    self.wfile.write(f"{len(first):X}\r\n".encode() + first + b"\r\n")
                    self.wfile.flush()
                    upstream.gate.wait(timeout=10)
                    self.wfile.write(f"{len(second):X}\r\n".encode() + second + b"\r\n0\r\n\r\n")
                    self.wfile.flush()
                    return
                payload = b"posted"
                self.send_response(201)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.gate.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def upstream():
    fake = Upstream()
    yield fake
    fake.close()


@pytest.fixture
def servers(upstream):
    config = proxy.ProxyConfig(
        allowed_hosts=frozenset({"plain.example.test", "127.0.0.1", HOST}),
        inject={HOST: INJECTED},
        connect=lambda host: http.client.HTTPConnection("127.0.0.1", upstream.port, timeout=10),
    )
    forward, reverse = proxy.start_servers(config, host="127.0.0.1", forward_port=0, reverse_port=0)
    assert reverse is not None
    yield forward, reverse
    for server in (forward, reverse):
        server.shutdown()
        server.server_close()


def reverse_request(servers, method, path, headers=None, body=None):
    _, reverse = servers
    conn = http.client.HTTPConnection("127.0.0.1", reverse.server_address[1], timeout=10)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def raw_exchange(port, payload):
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(payload)
        return sock.recv(4096)


def test_injected_headers_are_set_on_the_upstream_request(servers, upstream):
    status, headers, body = reverse_request(servers, "GET", f"/{HOST}/v1/models?limit=2")

    assert (status, body) == (200, b"hello")
    assert headers["X-Upstream"] == "yes"
    sent = upstream.requests[0]
    assert sent["path"] == "/v1/models?limit=2"
    sent_headers = dict(sent["headers"])
    assert sent_headers["x-api-key"] == "REAL-KEY"
    assert sent_headers["anthropic-version"] == "2023-06-01"
    assert sent_headers["host"] == HOST


def test_client_credentials_and_hop_headers_are_stripped(servers, upstream):
    reverse_request(
        servers,
        "GET",
        f"/{HOST}/x",
        headers={
            "x-api-key": "ATTACKER-KEY",
            "Authorization": "Bearer attacker",
            "Proxy-Authorization": "Basic abc",
            "Proxy-Foo": "1",
            "Connection": "close, X-Secret-Hop",
            "X-Secret-Hop": "1",
            "Anthropic-Version": "1999-01-01",
            "X-Custom": "keep",
        },
    )

    sent = upstream.requests[0]["headers"]
    names = [name for name, _ in sent]
    assert [v for n, v in sent if n == "x-api-key"] == ["REAL-KEY"]
    assert [v for n, v in sent if n == "anthropic-version"] == ["2023-06-01"]
    assert "authorization" not in names
    assert not any(n.startswith("proxy-") for n in names)
    assert "x-secret-hop" not in names
    assert dict(sent)["x-custom"] == "keep"
    assert "ATTACKER-KEY" not in repr(sent)


def test_post_body_is_forwarded(servers, upstream):
    status, _, body = reverse_request(servers, "POST", f"/{HOST}/v1/messages", body=b'{"a": 1}')

    assert (status, body) == (201, b"posted")
    assert upstream.requests[0]["body"] == b'{"a": 1}'


def test_non_injected_host_on_reverse_port_is_forbidden(servers, upstream):
    status, _, _ = reverse_request(servers, "GET", "/plain.example.test/x")

    assert status == 403
    assert upstream.requests == []


def test_malformed_reverse_target_is_rejected(servers):
    assert reverse_request(servers, "GET", "/")[0] == 400


def test_connect_to_injected_host_is_forbidden(servers):
    forward, _ = servers

    reply = raw_exchange(
        forward.server_address[1], f"CONNECT {HOST}:443 HTTP/1.1\r\nHost: {HOST}\r\n\r\n".encode()
    )

    assert reply.startswith(b"HTTP/1.1 403")


def test_plain_http_forward_to_injected_host_is_forbidden(servers, upstream):
    forward, _ = servers

    reply = raw_exchange(
        forward.server_address[1],
        f"GET http://{HOST}/v1/models HTTP/1.1\r\nHost: {HOST}\r\n\r\n".encode(),
    )

    assert reply.startswith(b"HTTP/1.1 403")
    assert upstream.requests == []


def test_connect_to_other_allowed_host_still_works(servers, upstream):
    forward, _ = servers

    reply = raw_exchange(
        forward.server_address[1],
        f"CONNECT 127.0.0.1:{upstream.port} HTTP/1.1\r\n\r\n".encode(),
    )

    assert reply.startswith(b"HTTP/1.1 200")


def test_connect_to_unlisted_host_is_forbidden(servers):
    forward, _ = servers

    reply = raw_exchange(forward.server_address[1], b"CONNECT evil.test:443 HTTP/1.1\r\n\r\n")

    assert reply.startswith(b"HTTP/1.1 403")


def test_forward_allowed_excludes_injected_hosts_even_if_listed():
    config = proxy.ProxyConfig(allowed_hosts=frozenset({HOST, "ok.test"}), inject={HOST: {}})

    assert not config.forward_allowed(f"{HOST}:443")
    assert config.forward_allowed("ok.test:443")


def test_streamed_response_passes_through_incrementally(servers, upstream):
    _, reverse = servers
    conn = http.client.HTTPConnection("127.0.0.1", reverse.server_address[1], timeout=10)
    try:
        conn.request("GET", f"/{HOST}/stream")
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream"
        # The upstream is blocked on the gate, so this proves the first chunk was relayed
        # before the upstream response finished.
        assert response.read1(5) == b"first"
        assert not upstream.gate.is_set()
        upstream.gate.set()
        assert response.read() == b"second"
    finally:
        conn.close()


def test_codex_responses_sse_passes_through_with_injected_bearer(upstream):
    openai_host = "api.openai.com"
    observed = []
    config = proxy.ProxyConfig(
        inject={openai_host: {"Authorization": "Bearer REAL-KEY"}},
        connect=lambda host: http.client.HTTPConnection("127.0.0.1", upstream.port, timeout=10),
        run_id="run-stream",
        attempt_id="attempt-stream",
        emit=observed.append,
    )
    forward, reverse = proxy.start_servers(config, host="127.0.0.1", forward_port=0, reverse_port=0)
    conn = http.client.HTTPConnection("127.0.0.1", reverse.server_address[1], timeout=10)
    request = json.dumps({"model": "gpt-6.1-sol", "input": "hi", "stream": True}).encode()
    first = b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
    second = (
        b'data: {"type":"response.completed","response":'
        b'{"id":"resp-stream","model":"gpt-6.1-sol",'
        b'"usage":{"input_tokens":15,"output_tokens":2,'
        b'"input_tokens_details":{"cached_tokens":5}}}}\n\n'
    )
    try:
        conn.request(
            "POST",
            f"/{openai_host}/v1/responses",
            body=request,
            headers={
                "Authorization": "Bearer sandbox-placeholder",
                "Content-Type": "application/json",
            },
        )
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream"
        assert response.read1(len(first)) == first
        assert not upstream.gate.is_set()
        upstream.gate.set()
        assert response.read() == second
    finally:
        upstream.gate.set()
        conn.close()
        forward.shutdown()
        forward.server_close()
        reverse.shutdown()
        reverse.server_close()

    sent = upstream.requests[0]
    assert sent["method"] == "POST"
    assert sent["path"] == "/v1/responses"
    assert sent["body"] == request
    assert dict(sent["headers"])["authorization"] == "Bearer REAL-KEY"
    assert observed[0]["run_id"] == "run-stream"
    assert observed[0]["input_tokens"] == 10
    assert observed[0]["usage_complete"] is True


def test_proxy_usage_identity_comes_from_host_config_not_sandbox_header(upstream):
    observed = []
    config = proxy.ProxyConfig(
        inject={"api.openai.com": {"Authorization": "Bearer REAL-KEY"}},
        connect=lambda host: http.client.HTTPConnection("127.0.0.1", upstream.port, timeout=10),
        run_id="host-run",
        attempt_id="host-attempt",
        emit=observed.append,
    )
    forward, reverse = proxy.start_servers(config, host="127.0.0.1", forward_port=0, reverse_port=0)
    try:
        request = json.dumps({"model": "gpt-test", "input": "hi"}).encode()
        status, _, body = reverse_request(
            (forward, reverse),
            "POST",
            "/api.openai.com/v1/responses",
            headers={"Content-Type": "application/json", "X-Tbcore-Run-Id": "attacker-run"},
            body=request,
        )
    finally:
        forward.shutdown()
        forward.server_close()
        reverse.shutdown()
        reverse.server_close()
    assert status == 200
    assert json.loads(body)["id"] == "resp-local"
    assert len(observed) == 1
    assert observed[0]["run_id"] == "host-run"
    assert observed[0]["attempt_id"] == "host-attempt"
    assert observed[0]["input_tokens"] == 10
    assert "x-tbcore-run-id" not in dict(upstream.requests[-1]["headers"])


def test_injected_model_hosts_reject_unsupported_endpoint(upstream):
    config = proxy.ProxyConfig(
        inject={"api.openai.com": {"Authorization": "Bearer REAL-KEY"}},
        connect=lambda host: http.client.HTTPConnection("127.0.0.1", upstream.port, timeout=10),
    )
    forward, reverse = proxy.start_servers(config, host="127.0.0.1", forward_port=0, reverse_port=0)
    try:
        status, _, _ = reverse_request((forward, reverse), "GET", "/api.openai.com/v1/models")
    finally:
        forward.shutdown()
        forward.server_close()
        reverse.shutdown()
        reverse.server_close()
    assert status == 403
    assert upstream.requests == []


def test_proxy_budget_denies_second_agent_call_before_upstream(upstream):
    budget = proxy.ProxyBudget(max_model_calls=1)
    config = proxy.ProxyConfig(
        inject={"api.openai.com": {"Authorization": "Bearer REAL-KEY"}},
        connect=lambda host: http.client.HTTPConnection("127.0.0.1", upstream.port, timeout=10),
        run_id="run-budget",
        attempt_id="attempt-budget",
        emit=lambda event: None,
        budget=budget,
    )
    forward, reverse = proxy.start_servers(config, host="127.0.0.1", forward_port=0, reverse_port=0)
    request = json.dumps({"model": "gpt-test", "input": "hi"}).encode()
    try:
        first = reverse_request(
            (forward, reverse), "POST", "/api.openai.com/v1/responses", body=request
        )
        second = reverse_request(
            (forward, reverse), "POST", "/api.openai.com/v1/responses", body=request
        )
    finally:
        forward.shutdown()
        forward.server_close()
        reverse.shutdown()
        reverse.server_close()
    assert first[0] == 200
    assert second[0] == 429
    assert len(upstream.requests) == 1


def test_chunked_request_body_is_rejected_with_411(servers, upstream):
    _, reverse = servers

    reply = raw_exchange(
        reverse.server_address[1],
        (
            f"POST /{HOST}/v1/messages HTTP/1.1\r\nHost: proxy\r\n"
            "Transfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n"
        ).encode(),
    )

    assert reply.startswith(b"HTTP/1.1 411")
    assert upstream.requests == []


def test_upstream_connection_failure_is_502(upstream):
    dead = socket.socket()
    dead.bind(("127.0.0.1", 0))
    port = dead.getsockname()[1]
    dead.close()
    config = proxy.ProxyConfig(
        inject={HOST: INJECTED},
        connect=lambda host: http.client.HTTPConnection("127.0.0.1", port, timeout=2),
    )
    forward, reverse = proxy.start_servers(config, host="127.0.0.1", forward_port=0, reverse_port=0)
    try:
        status, _, _ = reverse_request((forward, reverse), "GET", f"/{HOST}/x")
    finally:
        forward.shutdown()
        forward.server_close()
        reverse.shutdown()
        reverse.server_close()

    assert status == 502


def test_slow_upstream_is_bounded_by_read_timeout_not_connect_timeout(upstream):
    # Connect timeout (0.3s) is shorter than the upstream's 1s think time; the read
    # timeout must govern once connected, as with a long non-streaming model call.
    config = proxy.ProxyConfig(
        inject={HOST: INJECTED},
        connect=lambda host: http.client.HTTPConnection("127.0.0.1", upstream.port, timeout=0.3),
        read_timeout_s=5,
    )
    forward, reverse = proxy.start_servers(config, host="127.0.0.1", forward_port=0, reverse_port=0)
    try:
        status, _, body = reverse_request((forward, reverse), "GET", f"/{HOST}/slow")
    finally:
        forward.shutdown()
        forward.server_close()
        reverse.shutdown()
        reverse.server_close()

    assert (status, body) == (200, b"hello")


def test_parse_inject_validates_shape():
    assert proxy.parse_inject("") == {}
    assert proxy.parse_inject('{"A.Test": {"k": "v"}}') == {"a.test": {"k": "v"}}
    for bad in ("[]", '{"a": []}', '{"a": {"k": 1}}'):
        with pytest.raises(ValueError):
            proxy.parse_inject(bad)
