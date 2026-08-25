"""Minimal allowlisting HTTP/CONNECT forward proxy used for sandbox egress.

The sandbox network is Docker-internal (no direct outbound routing). Egress is
possible only through this proxy, which permits connections to hosts listed in
the TBCORE_ALLOW_HOSTS environment variable and rejects everything else.
"""

import os
import select
import socket
from socketserver import StreamRequestHandler, ThreadingTCPServer
from urllib.parse import urlsplit

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 3128
IDLE_TIMEOUT_S = 120
FAIL_REASONS = {400: "Bad Request", 403: "Forbidden", 405: "Method Not Allowed", 502: "Bad Gateway"}
ALLOWED_METHODS = {"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"}

ALLOWED_HOSTS = {
    host.strip().lower()
    for host in os.environ.get("TBCORE_ALLOW_HOSTS", "").split(",")
    if host.strip()
}


def host_allowed(authority: str) -> bool:
    host = authority.rsplit("@", 1)[-1]
    host = host.split(":", 1)[0].strip("[]").lower()
    return host in ALLOWED_HOSTS


class ProxyHandler(StreamRequestHandler):
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
        if not host_allowed(authority):
            self._fail(403)
            return
        host, _, port = authority.partition(":")
        try:
            upstream = socket.create_connection((host, int(port or 443)), timeout=10)
        except OSError:
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
        if not host or not host_allowed(host):
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
        reason = FAIL_REASONS.get(code, "Error")
        body = (
            f"HTTP/1.1 {code} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        ).encode("latin-1")
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except OSError:
            pass


class ProxyServer(ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    server = ProxyServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler)
    print(f"TBCORE_PROXY_READY allow={sorted(ALLOWED_HOSTS)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
