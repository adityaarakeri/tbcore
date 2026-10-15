"""Container sandbox for untrusted content.

Each :meth:`Sandbox.run` executes a command in an ephemeral container with:

- a fresh ephemeral workdir that is destroyed with the container,
- no host filesystem or docker socket access,
- CPU, memory, and PID limits,
- all Linux capabilities dropped, ``no-new-privileges``, and a non-root user,
- environment variables passed via a short-lived ``--env-file`` (not argv),
- no network by default; when ``allow_hosts`` is non-empty, egress is only
  possible through an allowlisting proxy on an internal Docker network,
- optionally (``inject_headers``) API credentials that live only in the proxy,
  which sets them on requests to configured hosts so the key never enters the
  sandbox,
- optionally a gVisor (or other) ``runtime`` and a ``read_only`` root filesystem.

:meth:`Sandbox.session` starts one long-lived container with the same hardening for
agent loops that run many commands against one workspace.

The proxy container is attached to the default bridge network so it can reach
the allowed hosts (including DNS), while sandbox containers stay isolated.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import TracebackType
from typing import IO

from tbcore.images.proxy.proxy import parse_upstream_overrides

PROXY_PORT = 3128
REVERSE_PROXY_PORT = 8080
# Value for SDKs that insist on a non-empty key; the proxy overwrites it on the wire.
PLACEHOLDER_API_KEY = "sandbox-placeholder"
ANTHROPIC_HOST = "api.anthropic.com"
OPENAI_HOST = "api.openai.com"
READ_ONLY_TMPFS = "/tmp:rw,size=64m"
DEFAULT_COPY_MAX_BYTES = 100 * 1024 * 1024
_READ_TEXT_MAX_BYTES = 10 * 1024 * 1024
# Grace period between SIGTERM and SIGKILL for `timeout` inside a session, and slack for the
# host-side backstop timeout on top of that.
_EXEC_KILL_AFTER_S = 5
_EXEC_BACKSTOP_SLACK_S = 10
_HOST_RE = re.compile(r"^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$")
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
IMAGE_NAMES = ("sandbox", "proxy", "codex")
DEFAULT_IMAGE_NAMES = ("sandbox", "proxy")
# Proxy container limits: it only relays bytes, so keep it small.
PROXY_MEMORY = "128m"
PROXY_CPUS = "0.5"
PROXY_PIDS_LIMIT = "256"
DEFAULT_ALLOW_HOSTS = ("api.anthropic.com",)
_PROXY_READY_MARKER = "TBCORE_PROXY_READY"
_PROXY_USAGE_PREFIX = "TBCORE_USAGE "
_TELEMETRY_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_FORBIDDEN_ENV_CHARS = frozenset(chr(c) for c in (0, 10, 13))


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


def _docker(
    docker_bin: str,
    args: Sequence[str],
    *,
    capture_output: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [docker_bin, *args],
        encoding="utf-8",
        errors="replace",
        capture_output=capture_output,
        timeout=timeout,
    )


def _docker_bytes(
    docker_bin: str,
    args: Sequence[str],
    *,
    input: bytes | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Binary-safe docker call (stdin/stdout are raw bytes, e.g. tar streams)."""
    return subprocess.run(
        [docker_bin, *args], input=input, capture_output=True, timeout=timeout, check=False
    )


def _docker_stream_out(
    docker_bin: str,
    args: Sequence[str],
    out: IO[bytes],
    *,
    max_bytes: int,
    timeout: float,
) -> tuple[int, str, bool]:
    """Copy docker's stdout into ``out``, stopping at ``max_bytes``.

    Returns ``(returncode, stderr, exceeded)``. Output is bounded so a hostile
    container cannot exhaust host memory or disk.
    """
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen([docker_bin, *args], stdout=subprocess.PIPE, stderr=err)
        if proc.stdout is None:
            raise SandboxError("docker produced no stdout pipe")
        timed_out = threading.Event()

        def expire() -> None:
            timed_out.set()
            proc.kill()

        timer = threading.Timer(timeout, expire)
        timer.start()
        total = 0
        exceeded = False
        try:
            while chunk := proc.stdout.read(65536):
                total += len(chunk)
                if total > max_bytes:
                    exceeded = True
                    proc.kill()
                    break
                out.write(chunk)
            returncode = proc.wait()
        finally:
            timer.cancel()
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if timed_out.is_set():
            raise SandboxError(f"docker {args[0]} timed out after {timeout:g}s")
        err.seek(0)
        stderr = err.read(4000).decode("utf-8", errors="replace")
    return returncode, stderr, exceeded


def build_image(
    context_dir: str | Path,
    tag: str,
    docker_bin: str = "docker",
    *,
    build_args: Mapping[str, str] | None = None,
) -> str:
    args = ["build"]
    for name, value in sorted((build_args or {}).items()):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or _FORBIDDEN_ENV_CHARS & set(value):
            raise ValueError(f"invalid Docker build argument: {name!r}")
        args.extend(("--build-arg", f"{name}={value}"))
    args.extend(("-t", tag, str(Path(context_dir))))
    result = _docker(
        docker_bin,
        args,
        capture_output=True,
    )
    if result.returncode != 0:
        raise SandboxError(f"docker build failed for {tag}:\n{result.stderr[-4000:]}")
    return tag


def build_codex_image(version: str, *, tag: str | None = None, docker_bin: str = "docker") -> str:
    """Build the optional agent image from one exact Codex CLI package version."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError("Codex CLI version must be an exact version, such as 1.2.3")
    image_tag = tag or f"tbcore/codex:{version}"
    return build_image(
        image_context("codex"),
        image_tag,
        docker_bin,
        build_args={"CODEX_VERSION": version},
    )


def image_context(name: str) -> Path:
    """Path to a bundled image build context: ``"sandbox"``, ``"proxy"``, or ``"codex"``.

    The contexts ship inside the package (``tbcore/images/<name>``); this
    requires an on-disk install (wheel or editable), not a zipped import.
    """
    if name not in IMAGE_NAMES:
        raise ValueError(f"unknown image {name!r}; expected one of {', '.join(IMAGE_NAMES)}")
    context = Path(str(resources.files("tbcore") / "images" / name))
    if not (context / "Dockerfile").is_file():
        raise SandboxError(f"bundled image context missing: {context}")
    return context


def build_default_images(tag_prefix: str = "tbcore", docker_bin: str = "docker") -> dict[str, str]:
    """Build the bundled sandbox and proxy images; returns ``{name: tag}``.

    With the default prefix the tags are ``tbcore/sandbox:latest`` and
    ``tbcore/proxy:latest``, which is what :class:`Sandbox` expects for its
    proxy image.
    """
    return {
        name: build_image(image_context(name), f"{tag_prefix}/{name}:latest", docker_bin)
        for name in DEFAULT_IMAGE_NAMES
    }


def _write_env_file(directory: Path, env: Mapping[str, str]) -> Path:
    """Write ``KEY=VALUE`` lines to a private (0600) temp file for ``docker --env-file``."""
    lines = []
    for key, value in env.items():
        if not key or "=" in key or key.startswith("#") or _FORBIDDEN_ENV_CHARS & set(key):
            raise ValueError(f"invalid environment variable name: {key!r}")
        if _FORBIDDEN_ENV_CHARS & set(value):
            raise ValueError(f"environment variable {key!r} contains a newline or NUL")
        lines.append(f"{key}={value}\n")
    fd, name = tempfile.mkstemp(prefix="tbcore-env-", dir=directory)  # mkstemp is 0600
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.writelines(lines)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def _safe_relative(path: str, *, allow_root: bool = False) -> str:
    """Validate a path inside ``/work``: relative, forward slashes, no ``..``."""
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if "\x00" in path or "\\" in path:
        raise ValueError(f"invalid path (NUL or backslash): {path!r}")
    windows = PureWindowsPath(path)
    if PurePosixPath(path).is_absolute() or windows.drive or windows.root:
        raise ValueError(f"path must be relative to /work: {path!r}")
    if ".." in PurePosixPath(path).parts:
        raise ValueError(f"path must not contain '..': {path!r}")
    normalized = PurePosixPath(path).as_posix()
    if normalized == "." and not allow_root:
        raise ValueError(f"path must name a file or directory below /work: {path!r}")
    return normalized


def _build_tar(files: Mapping[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    now = int(time.time())
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, content in files.items():
            data = content if isinstance(content, bytes) else content.encode("utf-8")
            info = tarfile.TarInfo(_safe_relative(name))
            info.size = len(data)
            info.mode = 0o644
            info.mtime = now
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _normalize_inject(
    inject_headers: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for host, headers in (inject_headers or {}).items():
        name = host.lower()
        if not _HOST_RE.match(name):
            raise ValueError(f"invalid host in inject_headers: {host!r}")
        checked: dict[str, str] = {}
        for header, value in headers.items():
            if not _HEADER_NAME_RE.match(header):
                raise ValueError(f"invalid header name for {host!r}: {header!r}")
            if _FORBIDDEN_ENV_CHARS & set(value):
                raise ValueError(f"header {header!r} for {host!r} contains a newline or NUL")
            checked[header] = value
        result[name] = checked
    return result


class Sandbox:
    """Runs commands inside ephemeral containers with no host access.

    ``inject_headers`` maps a host to headers the proxy sets on every request to it,
    e.g. ``{"api.anthropic.com": {"x-api-key": key}}``. Those hosts are added to the
    allowlist but are served only by a reverse proxy at :meth:`base_url`, which discards
    the sandbox's own ``Authorization``/``x-api-key`` headers; they are not reachable
    through CONNECT. The credential exists only in the proxy container.

    ``runtime`` selects a docker runtime for sandbox containers (e.g. ``"runsc"`` for
    gVisor). ``read_only`` mounts the root filesystem read-only, with a 64 MB ``/tmp``
    tmpfs and an anonymous volume at ``/work`` that is removed with the container.
    """

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
        user: str | None = None,
        inject_headers: Mapping[str, Mapping[str, str]] | None = None,
        runtime: str | None = None,
        read_only: bool = False,
        telemetry_run_id: str | None = None,
        telemetry_attempt_id: str | None = None,
        proxy_budget: Mapping[str, object] | None = None,
        upstream_overrides: Mapping[str, str] | None = None,
    ) -> None:
        if (telemetry_run_id is None) != (telemetry_attempt_id is None):
            raise ValueError("proxy telemetry requires both run and attempt IDs")
        for value in (telemetry_run_id, telemetry_attempt_id):
            if value is not None and not _TELEMETRY_ID_RE.fullmatch(value):
                raise ValueError("invalid proxy telemetry identity")
        if proxy_budget is not None and telemetry_run_id is None:
            raise ValueError("proxy budget requires host-bound run and attempt IDs")
        self.image = image
        self.timeout_s = timeout_s
        self._inject = _normalize_inject(inject_headers)
        self.upstream_overrides = dict(upstream_overrides or {})
        if set(self.upstream_overrides) - set(self._inject):
            raise ValueError("upstream overrides require a credential-injected model host")
        if self.upstream_overrides:
            parse_upstream_overrides(json.dumps(self.upstream_overrides))
        hosts = [host.lower() for host in allow_hosts]
        # Injected hosts are implicitly allowed (through the reverse proxy only).
        self.allow_hosts = tuple(dict.fromkeys([*hosts, *self._inject]))
        self.proxy_image = proxy_image
        self.work_root = Path(work_root) if work_root is not None else Path(tempfile.gettempdir())
        self.docker_bin = docker_bin
        self.memory = memory
        self.cpus = cpus
        self.user = user
        self.runtime = runtime
        self.read_only = read_only
        self.telemetry_run_id = telemetry_run_id
        self.telemetry_attempt_id = telemetry_attempt_id
        self.proxy_budget = dict(proxy_budget) if proxy_budget is not None else None
        self._network: str | None = None
        self._proxy_container: str | None = None
        self._sessions: set[SandboxSession] = set()

    @property
    def network_name(self) -> str | None:
        return self._network

    @property
    def injected_hosts(self) -> tuple[str, ...]:
        """Hosts whose credentials are injected by the proxy."""
        return tuple(self._inject)

    def base_url(self, host: str) -> str:
        """Base URL sandboxed code should use for an injected ``host``.

        For example ``http://proxy:8080/api.anthropic.com``; append the API path.
        """
        name = host.lower()
        if name not in self._inject:
            raise ValueError(f"{host!r} is not in inject_headers")
        return f"http://proxy:{REVERSE_PROXY_PORT}/{name}"

    def prepare(self) -> None:
        """Eagerly create the internal network and proxy (no-op if offline)."""
        if self.allow_hosts:
            self._ensure_network()
            self._ensure_proxy()

    def close(self) -> None:
        for session in list(self._sessions):
            session.close()
        if self._proxy_container is not None:
            self._remove_container(self._proxy_container)
            self._proxy_container = None
        if self._network is not None:
            _docker(self.docker_bin, ["network", "rm", self._network], capture_output=True)
            self._network = None

    def proxy_usage(self) -> list[dict[str, object]]:
        """Read host-bound, metadata-only usage observations before closing the proxy."""
        if self._proxy_container is None:
            raise SandboxError("proxy container is not running")
        logs = _docker(self.docker_bin, ["logs", self._proxy_container], capture_output=True)
        if logs.returncode != 0:
            raise SandboxError(f"cannot read proxy telemetry: {logs.stderr}")
        events: list[dict[str, object]] = []
        for line in logs.stdout.splitlines():
            if not line.startswith(_PROXY_USAGE_PREFIX):
                continue
            try:
                event = json.loads(line[len(_PROXY_USAGE_PREFIX) :])
            except json.JSONDecodeError as exc:
                raise SandboxError("malformed proxy telemetry") from exc
            if not isinstance(event, dict) or (
                event.get("run_id") != self.telemetry_run_id
                or event.get("attempt_id") != self.telemetry_attempt_id
            ):
                raise SandboxError("proxy telemetry identity mismatch")
            events.append(event)
        return events

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _remove_container(self, name: str) -> subprocess.CompletedProcess[str]:
        # -v also removes anonymous volumes (the /work volume of read_only containers).
        return _docker(self.docker_bin, ["rm", "-f", "-v", name], capture_output=True)

    def _container_env(self, env: Mapping[str, str] | None) -> dict[str, str]:
        """Environment for a sandbox container: injection defaults, then caller overrides."""
        merged: dict[str, str] = {}
        if self._inject:
            # httpx/requests honor HTTP_PROXY; the reverse proxy must be reached directly.
            merged["NO_PROXY"] = merged["no_proxy"] = "proxy"
            if ANTHROPIC_HOST in self._inject:
                merged["ANTHROPIC_BASE_URL"] = self.base_url(ANTHROPIC_HOST)
                merged["ANTHROPIC_API_KEY"] = PLACEHOLDER_API_KEY
            if OPENAI_HOST in self._inject:
                merged["OPENAI_BASE_URL"] = f"{self.base_url(OPENAI_HOST)}/v1"
                merged["OPENAI_API_KEY"] = PLACEHOLDER_API_KEY
        merged.update(env or {})
        return merged

    def _create_args(
        self, name: str, env_file: Path | None, extra: Sequence[str] = ()
    ) -> list[str]:
        """``docker create`` args up to (not including) the image; shared by run and sessions."""
        args = [
            "create",
            "--name",
            name,
            *extra,
            "-w",
            "/work",
            "--cpus",
            self.cpus,
            "--memory",
            self.memory,
            "--pids-limit",
            "128",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
        ]
        # None keeps the image's own USER (the bundled sandbox image uses uid 1000).
        if self.user is not None:
            args += ["--user", self.user]
        if self.runtime is not None:
            args += ["--runtime", self.runtime]
        if self.read_only:
            args += ["--read-only", "--tmpfs", READ_ONLY_TMPFS, "-v", "/work"]
        if self._network is not None:
            args += [
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
            args += ["--network", "none"]
        if env_file is not None:
            args += ["--env-file", str(env_file)]
        return args

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
        env_file: Path | None = None
        started = time.perf_counter()
        try:
            merged_env = self._container_env(env)
            if merged_env:
                # Outside staging so it is never copied into /work; keeps secrets off argv.
                env_file = _write_env_file(self.work_root, merged_env)
            create_args = self._create_args(container, env_file)
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
            self._remove_container(container)
            shutil.rmtree(staging, ignore_errors=True)
            if env_file is not None:
                env_file.unlink(missing_ok=True)

    def session(self, *, env: Mapping[str, str] | None = None) -> SandboxSession:
        """Start a long-lived container for many :meth:`SandboxSession.exec` calls.

        The container has the same hardening, network, limits and user as :meth:`run`.
        It is kept alive with ``--entrypoint sleep``, so the image's own ENTRYPOINT/CMD
        is bypassed. ``env`` is set for every exec in the session. Close the session (or
        use it as a context manager) to remove the container; closing it does not touch
        the shared proxy or network, which :meth:`close` owns.
        """
        self.prepare()
        container = f"tbcore-sess-{uuid.uuid4().hex[:12]}"
        env_file: Path | None = None
        try:
            merged_env = self._container_env(env)
            if merged_env:
                env_file = _write_env_file(self.work_root, merged_env)
            create_args = self._create_args(container, env_file, ["--label", "owner=tbcore"])
            create_args += ["--entrypoint", "sleep", self.image, "infinity"]
            created = _docker(self.docker_bin, create_args, capture_output=True)
            if created.returncode != 0:
                raise SandboxError(f"docker create failed:\n{created.stderr}")
        finally:
            if env_file is not None:
                env_file.unlink(missing_ok=True)
        started = _docker(self.docker_bin, ["start", container], capture_output=True)
        if started.returncode != 0:
            self._remove_container(container)
            raise SandboxError(f"docker start failed:\n{started.stderr}")
        session = SandboxSession(self, container)
        self._sessions.add(session)
        return session

    def _stage_files(
        self,
        container: str,
        staging: Path,
        files: Mapping[str, str | bytes],
    ) -> None:
        validated = [(_safe_relative(relative), content) for relative, content in files.items()]
        for relative, content in validated:
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
        network = self._network
        if network is None:
            raise SandboxError("internal network was not created")
        name = f"tbcore-proxy-{uuid.uuid4().hex[:10]}"
        # Injected hosts are served only by the reverse listener, never by CONNECT.
        forward_hosts = [host for host in self.allow_hosts if host not in self._inject]
        run_args = [
            "run",
            "-d",
            # The proxy holds injected credentials: never let a missing local image be
            # resolved by an implicit registry pull.
            "--pull=never",
            "--name",
            name,
            "--network",
            network,
            "--network-alias",
            "proxy",
            "--memory",
            PROXY_MEMORY,
            "--cpus",
            PROXY_CPUS,
            "--pids-limit",
            PROXY_PIDS_LIMIT,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "-e",
            f"TBCORE_ALLOW_HOSTS={','.join(forward_hosts)}",
        ]
        if self.telemetry_run_id is not None and self.telemetry_attempt_id is not None:
            run_args.extend(("-e", f"TBCORE_RUN_ID={self.telemetry_run_id}"))
            run_args.extend(("-e", f"TBCORE_ATTEMPT_ID={self.telemetry_attempt_id}"))
        inject_file: Path | None = None
        try:
            proxy_env: dict[str, str] = {}
            if self._inject:
                proxy_env["TBCORE_INJECT"] = json.dumps(self._inject)
            if self.proxy_budget is not None:
                proxy_env["TBCORE_PROXY_BUDGET"] = json.dumps(self.proxy_budget)
            if self.upstream_overrides:
                proxy_env["TBCORE_UPSTREAM_OVERRIDES"] = json.dumps(self.upstream_overrides)
            if proxy_env:
                # Credentials and budget policy are host-owned and never enter workload argv.
                inject_file = _write_env_file(self.work_root, proxy_env)
                run_args += ["--env-file", str(inject_file)]
            started = _docker(self.docker_bin, [*run_args, self.proxy_image], capture_output=True)
        finally:
            # docker has read the file by the time `run -d` returns.
            if inject_file is not None:
                inject_file.unlink(missing_ok=True)
        if started.returncode != 0:
            raise SandboxError(
                f"failed to start proxy container {self.proxy_image!r}; "
                f"build it with tbcore.build_default_images():\n{started.stderr}"
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


class SandboxSession:
    """A long-lived sandbox container; create with :meth:`Sandbox.session`.

    State (files, processes started by earlier calls that survive) persists across
    :meth:`exec` calls. Use :meth:`put_files` and :meth:`copy_out` to move files, and
    :meth:`close` (or ``with``) to remove the container and its anonymous volumes.
    """

    def __init__(self, sandbox: Sandbox, container: str) -> None:
        self._sandbox = sandbox
        self.container = container
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise SandboxError("session is closed")

    def exec(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> RunResult:
        """Run ``command`` in the session container (cwd ``/work``).

        The command runs under coreutils ``timeout`` inside the container, so it is
        really killed at the deadline (``timeout_s`` defaults to the Sandbox's). The
        image must provide ``timeout``. Processes that detach into their own session
        can outlive it. A command that exits 124 or 137 only at the deadline is
        reported as ``timed_out`` with ``exit_code=None``, as :meth:`Sandbox.run` does.
        """
        if not command:
            raise ValueError("command must be a non-empty sequence of argv strings")
        self._require_open()
        sandbox = self._sandbox
        limit = sandbox.timeout_s if timeout_s is None else timeout_s
        if limit <= 0:
            raise ValueError("timeout_s must be positive")
        env_file: Path | None = None
        started = time.perf_counter()
        try:
            args = ["exec"]
            if env:
                env_file = _write_env_file(sandbox.work_root, env)
                args += ["--env-file", str(env_file)]
            args += [
                self.container,
                "timeout",
                f"--kill-after={_EXEC_KILL_AFTER_S}",
                f"{limit:.3f}",
                *command,
            ]
            try:
                proc = _docker(
                    sandbox.docker_bin,
                    args,
                    capture_output=True,
                    timeout=limit + _EXEC_KILL_AFTER_S + _EXEC_BACKSTOP_SLACK_S,
                )
            except subprocess.TimeoutExpired:
                # Backstop: the in-container timeout should have fired long before this.
                return RunResult(
                    command=tuple(command),
                    exit_code=None,
                    stdout="",
                    stderr="",
                    timed_out=True,
                    duration_s=time.perf_counter() - started,
                )
            duration = time.perf_counter() - started
            # `timeout` exits 124 on expiry and 137 when it had to SIGKILL.
            expired = proc.returncode in (124, 137) and duration >= limit - 0.5
            return RunResult(
                command=tuple(command),
                exit_code=None if expired else proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                timed_out=expired,
                duration_s=duration,
            )
        finally:
            if env_file is not None:
                env_file.unlink(missing_ok=True)

    def put_files(self, files: Mapping[str, str | bytes]) -> None:
        """Write ``files`` (relative paths) into ``/work``, owned by the container user."""
        self._require_open()
        payload = _build_tar(files)
        sandbox = self._sandbox
        proc = _docker_bytes(
            sandbox.docker_bin,
            ["exec", "-i", self.container, "tar", "--no-same-owner", "-xf", "-", "-C", "/work"],
            input=payload,
            timeout=sandbox.timeout_s,
        )
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="replace")[-4000:]
            raise SandboxError(f"put_files failed:\n{detail}")

    def _fetch_tar(self, relative: str, out: IO[bytes], max_bytes: int) -> None:
        sandbox = self._sandbox
        returncode, stderr, exceeded = _docker_stream_out(
            sandbox.docker_bin,
            ["exec", self.container, "tar", "-cf", "-", "-C", "/work", "--", relative],
            out,
            max_bytes=max_bytes,
            timeout=sandbox.timeout_s,
        )
        if exceeded:
            raise SandboxError(f"{relative!r} is larger than max_bytes={max_bytes}")
        if returncode != 0:
            raise SandboxError(f"could not read {relative!r} from the sandbox:\n{stderr}")

    def copy_out(
        self, path: str, dest: str | Path, *, max_bytes: int = DEFAULT_COPY_MAX_BYTES
    ) -> Path:
        """Copy ``/work/<path>`` (file or directory) into the host directory ``dest``.

        The result is ``dest/<path>`` (returned). The archive is size-capped at
        ``max_bytes`` and extracted with ``tarfile``'s ``data`` filter, which rejects
        absolute paths, ``..`` and links pointing outside ``dest``.
        """
        self._require_open()
        if not hasattr(tarfile, "data_filter"):
            raise SandboxError("copy_out needs Python >= 3.11.4 (tarfile extraction filters)")
        relative = _safe_relative(path, allow_root=True)
        target = Path(dest)
        with tempfile.TemporaryFile(dir=self._sandbox.work_root) as spool:
            self._fetch_tar(relative, spool, max_bytes)
            spool.seek(0)
            target.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(fileobj=spool, mode="r:") as tar:
                    tar.extractall(target, filter="data")
            except tarfile.TarError as exc:
                raise SandboxError(
                    f"refusing unsafe or invalid archive for {path!r}: {exc}"
                ) from exc
        return target if relative == "." else target / relative

    def read_text(
        self, path: str, *, encoding: str = "utf-8", max_bytes: int = _READ_TEXT_MAX_BYTES
    ) -> str:
        """Read a regular file from ``/work`` as text (no host filesystem writes)."""
        self._require_open()
        relative = _safe_relative(path)
        buffer = io.BytesIO()
        self._fetch_tar(relative, buffer, max_bytes)
        buffer.seek(0)
        try:
            with tarfile.open(fileobj=buffer, mode="r:") as tar:
                members = tar.getmembers()
                if len(members) != 1 or not members[0].isfile():
                    raise SandboxError(f"{path!r} is not a regular file")
                handle = tar.extractfile(members[0])
                if handle is None:
                    raise SandboxError(f"{path!r} is not a regular file")
                return handle.read().decode(encoding)
        except tarfile.TarError as exc:
            raise SandboxError(f"invalid archive for {path!r}: {exc}") from exc

    def close(self) -> None:
        """Force-remove the container and its anonymous volumes (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._sandbox._sessions.discard(self)
        removed = self._sandbox._remove_container(self.container)
        if removed.returncode != 0 and "No such container" not in (removed.stderr or ""):
            raise SandboxError(f"could not remove {self.container}:\n{removed.stderr}")

    def __enter__(self) -> SandboxSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
