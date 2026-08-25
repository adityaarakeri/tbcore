"""Container sandbox for untrusted content.

Each :meth:`Sandbox.run` executes a command in an ephemeral container with:

- a fresh ephemeral workdir that is destroyed with the container,
- no host filesystem or docker socket access,
- CPU, memory, and PID limits,
- no network by default; when ``allow_hosts`` is non-empty, egress is only
  possible through an allowlisting proxy on an internal Docker network.

The proxy container is attached to the default bridge network so it can reach
the allowed hosts (including DNS), while sandbox containers stay isolated.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PROXY_PORT = 3128
DEFAULT_ALLOW_HOSTS = ("api.anthropic.com",)
_PROXY_READY_MARKER = "TBCORE_PROXY_READY"


class SandboxError(RuntimeError):
    """Raised when the sandbox itself fails to set up or run a command."""


@dataclass(frozen=True)
class RunResult:
    command: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float


def _docker(docker_bin: str, args: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([docker_bin, *args], encoding="utf-8", errors="replace", **kwargs)


def build_image(context_dir: str | Path, tag: str, docker_bin: str = "docker") -> str:
    result = _docker(
        docker_bin,
        ["build", "-t", tag, str(Path(context_dir))],
        capture_output=True,
    )
    if result.returncode != 0:
        raise SandboxError(f"docker build failed for {tag}:\n{result.stderr[-4000:]}")
    return tag


class Sandbox:
    """Runs commands inside ephemeral containers with no host access."""

    def __init__(
        self,
        image: str,
        timeout_s: float = 60.0,
        allow_hosts: Sequence[str] = DEFAULT_ALLOW_HOSTS,
        *,
        proxy_image: str = "tbcore/proxy:latest",
        work_root: str | Path | None = None,
        docker_bin: str = "docker",
        memory: str = "512m",
        cpus: str = "1.0",
    ) -> None:
        self.image = image
        self.timeout_s = timeout_s
        self.allow_hosts = tuple(host.lower() for host in allow_hosts)
        self.proxy_image = proxy_image
        self.work_root = Path(work_root) if work_root is not None else Path(tempfile.gettempdir())
        self.docker_bin = docker_bin
        self.memory = memory
        self.cpus = cpus
        self._network: str | None = None
        self._proxy_container: str | None = None

    @property
    def network_name(self) -> str | None:
        return self._network

    def prepare(self) -> None:
        """Eagerly create the internal network and proxy (no-op if offline)."""
        if self.allow_hosts:
            self._ensure_network()
            self._ensure_proxy()

    def close(self) -> None:
        if self._proxy_container is not None:
            _docker(
                self.docker_bin,
                ["rm", "-f", self._proxy_container],
                capture_output=True,
            )
            self._proxy_container = None
        if self._network is not None:
            _docker(self.docker_bin, ["network", "rm", self._network], capture_output=True)
            self._network = None

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def run(
        self,
        command: Sequence[str],
        files: Mapping[str, str | bytes] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RunResult:
        """Execute ``command`` in a fresh container and capture its output."""
        if not command:
            raise ValueError("command must be a non-empty sequence of argv strings")
        self.prepare()
        container = f"tbcore-run-{uuid.uuid4().hex[:12]}"
        staging = Path(tempfile.mkdtemp(prefix="tbcore-stage-", dir=self.work_root))
        started = time.perf_counter()
        try:
            create_args = [
                "create",
                "--name",
                container,
                "-w",
                "/work",
                "--cpus",
                self.cpus,
                "--memory",
                self.memory,
                "--pids-limit",
                "128",
                "--security-opt",
                "no-new-privileges",
            ]
            if self._network is not None:
                create_args += [
                    "--network",
                    self._network,
                    "-e",
                    f"HTTP_PROXY=http://proxy:{PROXY_PORT}",
                    "-e",
                    f"http_proxy=http://proxy:{PROXY_PORT}",
                    "-e",
                    f"HTTPS_PROXY=http://proxy:{PROXY_PORT}",
                    "-e",
                    f"https_proxy=http://proxy:{PROXY_PORT}",
                ]
            else:
                create_args += ["--network", "none"]
            for key, value in (env or {}).items():
                create_args += ["-e", f"{key}={value}"]
            create_args += [self.image, *command]

            created = _docker(self.docker_bin, create_args, capture_output=True)
            if created.returncode != 0:
                raise SandboxError(f"docker create failed:\n{created.stderr}")
            if files:
                self._stage_files(container, staging, files)

            try:
                proc = _docker(
                    self.docker_bin,
                    ["start", "-a", container],
                    capture_output=True,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired:
                _docker(self.docker_bin, ["kill", container], capture_output=True)
                logs = _docker(self.docker_bin, ["logs", container], capture_output=True)
                return RunResult(
                    command=tuple(command),
                    exit_code=None,
                    stdout=logs.stdout or "",
                    stderr=logs.stderr or "",
                    timed_out=True,
                    duration_s=time.perf_counter() - started,
                )
            return RunResult(
                command=tuple(command),
                exit_code=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                timed_out=False,
                duration_s=time.perf_counter() - started,
            )
        finally:
            _docker(self.docker_bin, ["rm", "-f", container], capture_output=True)
            shutil.rmtree(staging, ignore_errors=True)

    def _stage_files(
        self,
        container: str,
        staging: Path,
        files: Mapping[str, str | bytes],
    ) -> None:
        for relative, content in files.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")
        copied = _docker(
            self.docker_bin,
            ["cp", f"{staging.as_posix()}/.", f"{container}:/work"],
            capture_output=True,
        )
        if copied.returncode != 0:
            raise SandboxError(f"docker cp failed:\n{copied.stderr}")

    def _ensure_network(self) -> None:
        if self._network is not None:
            return
        network = f"tbcore-net-{uuid.uuid4().hex[:10]}"
        created = _docker(
            self.docker_bin,
            ["network", "create", "--internal", "--label", "owner=tbcore", network],
            capture_output=True,
        )
        if created.returncode != 0 and "already exists" not in (created.stderr or ""):
            raise SandboxError(f"docker network create failed:\n{created.stderr}")
        self._network = network

    def _ensure_proxy(self) -> None:
        if self._proxy_container is not None:
            return
        self._ensure_network()
        name = f"tbcore-proxy-{uuid.uuid4().hex[:10]}"
        started = _docker(
            self.docker_bin,
            [
                "run",
                "-d",
                "--name",
                name,
                "--network",
                self._network,
                "--network-alias",
                "proxy",
                "-e",
                f"TBCORE_ALLOW_HOSTS={','.join(self.allow_hosts)}",
                self.proxy_image,
            ],
            capture_output=True,
        )
        if started.returncode != 0:
            raise SandboxError(
                f"failed to start proxy container {self.proxy_image!r}; "
                f"build it with tbcore.sandbox.build_image:\n{started.stderr}"
            )
        self._proxy_container = name
        bridged = _docker(
            self.docker_bin, ["network", "connect", "bridge", name], capture_output=True
        )
        if bridged.returncode != 0:
            self.close()
            raise SandboxError(f"could not attach proxy to bridge network:\n{bridged.stderr}")
        self._wait_proxy_ready(name)

    def _wait_proxy_ready(self, name: str, timeout_s: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            logs = _docker(self.docker_bin, ["logs", name], capture_output=True)
            if _PROXY_READY_MARKER in (logs.stdout or ""):
                return
            state = _docker(
                self.docker_bin,
                ["inspect", "-f", "{{.State.Status}}", name],
                capture_output=True,
            )
            if state.returncode == 0 and (state.stdout or "").strip() not in (
                "created",
                "running",
            ):
                raise SandboxError(f"proxy container died:\n{logs.stderr}")
            time.sleep(0.2)
        raise SandboxError("proxy did not become ready in time")
