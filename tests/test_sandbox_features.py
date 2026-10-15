"""Docker-free tests for credential injection, runtime/read_only, and sessions."""

import io
import json
import os
import subprocess
import tarfile

import pytest

from tbcore import sandbox as sandbox_mod
from tbcore.sandbox import Sandbox, SandboxError
from test_sandbox_unit import FakeDocker

BASE_CREATE = [
    "create",
    "--name",
    "NAME",
    "-w",
    "/work",
    "--cpus",
    "1.0",
    "--memory",
    "512m",
    "--pids-limit",
    "128",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
]


@pytest.fixture
def fake_docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(sandbox_mod, "_docker", fake)
    return fake


def normalized_create(fake):
    create = list(fake.first("create"))
    create[create.index("--name") + 1] = "NAME"
    return create


# --- argv stability, runtime, read_only -------------------------------------------------


def test_run_argv_is_unchanged_offline(fake_docker, tmp_path):
    Sandbox("img:test", allow_hosts=(), work_root=tmp_path).run(["echo", "hi"])

    assert normalized_create(fake_docker) == [
        *BASE_CREATE,
        "--network",
        "none",
        "img:test",
        "echo",
        "hi",
    ]


def test_run_argv_is_unchanged_with_allowlist(fake_docker, tmp_path):
    box = Sandbox("img:test", allow_hosts=("example.com",), work_root=tmp_path)

    box.run(["true"])

    proxy_url = "http://proxy:3128"
    assert normalized_create(fake_docker) == [
        *BASE_CREATE,
        "--network",
        box.network_name,
        "-e",
        f"HTTP_PROXY={proxy_url}",
        "-e",
        f"http_proxy={proxy_url}",
        "-e",
        f"HTTPS_PROXY={proxy_url}",
        "-e",
        f"https_proxy={proxy_url}",
        "img:test",
        "true",
    ]


def test_removal_uses_force_and_removes_volumes(fake_docker, tmp_path):
    Sandbox("img:test", allow_hosts=(), work_root=tmp_path).run(["true"])

    assert fake_docker.first("rm")[:3] == ["rm", "-f", "-v"]


def test_runtime_flag_is_passed_to_sandbox_but_not_proxy(fake_docker, tmp_path):
    box = Sandbox("img:test", allow_hosts=("example.com",), work_root=tmp_path, runtime="runsc")

    box.run(["true"])

    create = fake_docker.first("create")
    assert create[create.index("--runtime") + 1] == "runsc"
    assert "--runtime" not in fake_docker.first("run")


def test_no_runtime_or_read_only_flags_by_default(fake_docker, tmp_path):
    Sandbox("img:test", allow_hosts=(), work_root=tmp_path).run(["true"])

    create = fake_docker.first("create")
    for flag in ("--runtime", "--read-only", "--tmpfs", "-v"):
        assert flag not in create


def test_read_only_adds_tmpfs_and_work_volume(fake_docker, tmp_path):
    Sandbox("img:test", allow_hosts=(), work_root=tmp_path, read_only=True).run(["true"])

    create = fake_docker.first("create")
    assert "--read-only" in create
    assert create[create.index("--tmpfs") + 1] == "/tmp:rw,size=64m"
    assert create[create.index("-v") + 1] == "/work"


# --- credential injection ---------------------------------------------------------------


def make_injecting(tmp_path, **kwargs):
    return Sandbox(
        "img:test",
        allow_hosts=("plain.test",),
        work_root=tmp_path,
        inject_headers={"API.Anthropic.com": {"x-api-key": "sk-REAL"}},
        **kwargs,
    )


def test_injected_hosts_join_allowlist_but_leave_the_connect_path(fake_docker, tmp_path):
    box = make_injecting(tmp_path)

    box.prepare()

    assert box.allow_hosts == ("plain.test", "api.anthropic.com")
    assert box.injected_hosts == ("api.anthropic.com",)
    assert "TBCORE_ALLOW_HOSTS=plain.test" in fake_docker.first("run")


def test_inject_map_reaches_proxy_via_private_env_file_not_argv(monkeypatch, tmp_path):
    fake = FakeDocker()
    seen = {}

    def spy(docker_bin, args, **kwargs):
        args = list(args)
        if args[0] == "run" and "--env-file" in args:
            path = args[args.index("--env-file") + 1]
            with open(path, encoding="utf-8") as fh:
                seen["contents"] = fh.read()
            seen["path"] = path
            seen["mode"] = os.stat(path).st_mode & 0o777
        return fake(docker_bin, args, **kwargs)

    monkeypatch.setattr(sandbox_mod, "_docker", spy)

    make_injecting(tmp_path).prepare()

    assert not any("sk-REAL" in arg for arg in fake.first("run"))
    payload = seen["contents"].removeprefix("TBCORE_INJECT=")
    assert json.loads(payload) == {"api.anthropic.com": {"x-api-key": "sk-REAL"}}
    if os.name == "posix":
        assert seen["mode"] == 0o600
    assert not os.path.exists(seen["path"])
    assert list(tmp_path.iterdir()) == []


def test_no_proxy_env_file_without_injection(fake_docker, tmp_path):
    Sandbox("img:test", allow_hosts=("example.com",), work_root=tmp_path).prepare()

    assert "--env-file" not in fake_docker.first("run")


def test_injection_sets_sdk_env_without_the_key(fake_docker, tmp_path):
    make_injecting(tmp_path).run(["true"])

    contents = fake_docker.env_file_contents
    assert contents is not None
    lines = contents.splitlines()
    assert "ANTHROPIC_BASE_URL=http://proxy:8080/api.anthropic.com" in lines
    assert "ANTHROPIC_API_KEY=sandbox-placeholder" in lines
    assert "NO_PROXY=proxy" in lines
    assert "sk-REAL" not in contents
    assert not any("sk-REAL" in arg for arg in fake_docker.first("create"))


def test_openai_injection_sets_sdk_env_without_the_key(fake_docker, tmp_path):
    box = Sandbox(
        "img:test",
        allow_hosts=(),
        work_root=tmp_path,
        inject_headers={"API.OpenAI.com": {"Authorization": "Bearer sk-REAL"}},
    )

    box.run(["true"])

    contents = fake_docker.env_file_contents
    assert contents is not None
    lines = contents.splitlines()
    assert "OPENAI_BASE_URL=http://proxy:8080/api.openai.com/v1" in lines
    assert "OPENAI_API_KEY=sandbox-placeholder" in lines
    assert "NO_PROXY=proxy" in lines
    assert "sk-REAL" not in contents
    assert not any("sk-REAL" in arg for arg in fake_docker.first("create"))


def test_injection_preserves_both_sdk_defaults_and_caller_overrides(fake_docker, tmp_path):
    box = Sandbox(
        "img:test",
        allow_hosts=(),
        work_root=tmp_path,
        inject_headers={
            "api.anthropic.com": {"x-api-key": "anthropic-real"},
            "api.openai.com": {"Authorization": "Bearer openai-real"},
        },
    )

    box.run(
        ["true"],
        env={"OPENAI_BASE_URL": "http://custom/v1", "OPENAI_API_KEY": "caller-key"},
    )

    lines = fake_docker.env_file_contents.splitlines()
    assert "ANTHROPIC_BASE_URL=http://proxy:8080/api.anthropic.com" in lines
    assert "ANTHROPIC_API_KEY=sandbox-placeholder" in lines
    assert "OPENAI_BASE_URL=http://custom/v1" in lines
    assert "OPENAI_API_KEY=caller-key" in lines
    assert "OPENAI_BASE_URL=http://proxy:8080/api.openai.com/v1" not in lines
    assert "OPENAI_API_KEY=sandbox-placeholder" not in lines
    assert "anthropic-real" not in fake_docker.env_file_contents
    assert "openai-real" not in fake_docker.env_file_contents


def test_caller_env_overrides_injection_defaults(fake_docker, tmp_path):
    make_injecting(tmp_path).run(["true"], env={"ANTHROPIC_API_KEY": "mine"})

    lines = fake_docker.env_file_contents.splitlines()
    assert "ANTHROPIC_API_KEY=mine" in lines
    assert "ANTHROPIC_API_KEY=sandbox-placeholder" not in lines


def test_base_url_for_injected_host(tmp_path):
    box = make_injecting(tmp_path)

    assert box.base_url("api.anthropic.com") == "http://proxy:8080/api.anthropic.com"
    with pytest.raises(ValueError, match="not in inject_headers"):
        box.base_url("plain.test")


@pytest.mark.parametrize(
    "inject",
    [
        {"bad host": {"k": "v"}},
        {"a.test": {"bad name": "v"}},
        {"a.test": {"k": "v\r\nX-Evil: 1"}},
        {"a/b": {"k": "v"}},
    ],
)
def test_invalid_inject_headers_are_rejected(tmp_path, inject):
    with pytest.raises(ValueError):
        Sandbox("img:test", work_root=tmp_path, inject_headers=inject)


# --- sessions ---------------------------------------------------------------------------


class SessionFake(FakeDocker):
    """FakeDocker plus the exec and binary helpers sessions use."""

    def __init__(self):
        super().__init__()
        self.byte_calls = []
        self.tar_stdin = None
        self.stream_payload = b""
        self.stream_result = (0, "", False)
        self.exec_result = (0, "ok\n", "")
        self.exec_timeout = None
        self.exec_env_contents = None

    def __call__(self, docker_bin, args, *, capture_output=False, timeout=None):
        args = list(args)
        if args[0] != "exec":
            return super().__call__(
                docker_bin, args, capture_output=capture_output, timeout=timeout
            )
        self.calls.append(args)
        self.exec_timeout = timeout
        if "--env-file" in args:
            with open(args[args.index("--env-file") + 1], encoding="utf-8") as fh:
                self.exec_env_contents = fh.read()
        code, out, err = self.exec_result
        return subprocess.CompletedProcess(args, code, stdout=out, stderr=err)

    def docker_bytes(self, docker_bin, args, *, input=None, timeout=None):
        self.byte_calls.append(list(args))
        self.tar_stdin = input
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    def stream_out(self, docker_bin, args, out, *, max_bytes, timeout):
        self.byte_calls.append(list(args))
        out.write(self.stream_payload)
        return self.stream_result


@pytest.fixture
def session_fake(monkeypatch):
    fake = SessionFake()
    monkeypatch.setattr(sandbox_mod, "_docker", fake)
    monkeypatch.setattr(sandbox_mod, "_docker_bytes", fake.docker_bytes)
    monkeypatch.setattr(sandbox_mod, "_docker_stream_out", fake.stream_out)
    return fake


def tar_of(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def exec_calls(fake):
    return [call for call in fake.calls if call[0] == "exec"]


def test_session_creates_hardened_sleeping_container(session_fake, tmp_path):
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path, read_only=True, runtime="runsc")

    with box.session() as session:
        create = session_fake.first("create")
        assert create[create.index("--cap-drop") + 1] == "ALL"
        assert "--read-only" in create
        assert create[create.index("--runtime") + 1] == "runsc"
        assert create[-4:] == ["--entrypoint", "sleep", "img:test", "infinity"]
        assert create[create.index("--name") + 1] == session.container
        assert ["start", session.container] in session_fake.calls

    assert session_fake.first("rm") == ["rm", "-f", "-v", session.container]


def test_session_inherits_openai_sdk_env_without_the_key(session_fake, tmp_path):
    box = Sandbox(
        "img:test",
        allow_hosts=(),
        work_root=tmp_path,
        inject_headers={"api.openai.com": {"Authorization": "Bearer sk-REAL"}},
    )

    with box.session():
        contents = session_fake.env_file_contents
        assert contents is not None
        assert "OPENAI_BASE_URL=http://proxy:8080/api.openai.com/v1" in contents.splitlines()
        assert "OPENAI_API_KEY=sandbox-placeholder" in contents.splitlines()
        assert "sk-REAL" not in contents
        assert not any("sk-REAL" in arg for arg in session_fake.first("create"))


def test_session_close_is_idempotent_and_leaves_proxy_and_network(session_fake, tmp_path):
    box = Sandbox("img:test", allow_hosts=("example.com",), work_root=tmp_path)
    session = box.session()
    proxy_name = box._proxy_container

    session.close()
    session.close()

    assert [c[-1] for c in session_fake.calls if c[0] == "rm"] == [session.container]
    assert box.network_name is not None
    assert box._proxy_container == proxy_name
    with pytest.raises(SandboxError, match="closed"):
        session.exec(["true"])
    box.close()
    assert [c[-1] for c in session_fake.calls if c[0] == "rm"][-1] == proxy_name


def test_sandbox_close_closes_open_sessions(session_fake, tmp_path):
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    session = box.session()

    box.close()

    assert session.closed
    assert session_fake.first("rm")[-1] == session.container


def test_exec_wraps_in_timeout_and_passes_env_by_file(session_fake, tmp_path):
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path, timeout_s=30)
    with box.session() as session:
        result = session.exec(["echo", "hi"], env={"SECRET": "s3"}, timeout_s=7)

    call = exec_calls(session_fake)[0]
    assert call[-6:] == [session.container, "timeout", "--kill-after=5", "7.000", "echo", "hi"]
    assert not any("s3" in arg for arg in call)
    assert session_fake.exec_env_contents == "SECRET=s3\n"
    assert session_fake.exec_timeout > 7
    assert (result.exit_code, result.stdout, result.timed_out) == (0, "ok\n", False)
    assert list(tmp_path.iterdir()) == []


def test_exec_defaults_to_sandbox_timeout(session_fake, tmp_path):
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path, timeout_s=12)
    with box.session() as session:
        session.exec(["true"])

    assert "12.000" in exec_calls(session_fake)[0]


def test_exec_reports_timeout_for_exit_124_at_the_deadline(session_fake, tmp_path, monkeypatch):
    session_fake.exec_result = (124, "", "")
    ticks = iter([0.0, 5.0])
    monkeypatch.setattr(sandbox_mod.time, "perf_counter", lambda: next(ticks, 5.0))
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    with box.session() as session:
        result = session.exec(["sleep", "99"], timeout_s=2)

    assert result.timed_out is True
    assert result.exit_code is None


def test_exit_124_before_the_deadline_is_a_plain_exit_code(session_fake, tmp_path):
    session_fake.exec_result = (124, "", "")
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    with box.session() as session:
        result = session.exec(["sh", "-c", "exit 124"], timeout_s=60)

    assert result.timed_out is False
    assert result.exit_code == 124


def test_put_files_streams_a_tar_with_relative_paths(session_fake, tmp_path):
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    with box.session() as session:
        session.put_files({"a/b.txt": "hello", "c.bin": b"\x00\x01"})

    args = session_fake.byte_calls[0]
    assert args[:3] == ["exec", "-i", session.container]
    assert "--no-same-owner" in args
    assert args[-2:] == ["-C", "/work"]
    with tarfile.open(fileobj=io.BytesIO(session_fake.tar_stdin)) as tar:
        assert tar.getnames() == ["a/b.txt", "c.bin"]
        assert tar.extractfile("a/b.txt").read() == b"hello"


@pytest.mark.parametrize(
    "bad", ["/etc/passwd", "../x", "a/../../x", "C:\\x", "C:/x", "", ".", "a\\b"]
)
def test_put_files_rejects_unsafe_paths(session_fake, tmp_path, bad):
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    with box.session() as session, pytest.raises(ValueError):
        session.put_files({bad: "x"})

    assert session_fake.byte_calls == []


@pytest.mark.parametrize("bad", ["/etc/passwd", "../x", "C:\\x", ""])
def test_copy_out_and_read_text_reject_unsafe_paths(session_fake, tmp_path, bad):
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    with box.session() as session:
        with pytest.raises(ValueError):
            session.copy_out(bad, tmp_path / "out")
        with pytest.raises(ValueError):
            session.read_text(bad)

    assert session_fake.byte_calls == []


def test_copy_out_extracts_under_dest(session_fake, tmp_path):
    session_fake.stream_payload = tar_of({"out/result.txt": b"42"})
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    with box.session() as session:
        path = session.copy_out("out/result.txt", tmp_path / "dest")

    assert path == tmp_path / "dest" / "out" / "result.txt"
    assert path.read_bytes() == b"42"
    assert session_fake.byte_calls[0][-4:] == ["-C", "/work", "--", "out/result.txt"]


def test_copy_out_rejects_path_traversal_in_archive(session_fake, tmp_path):
    session_fake.stream_payload = tar_of({"../escape.txt": b"x"})
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    with box.session() as session, pytest.raises(SandboxError, match="unsafe"):
        session.copy_out("x", tmp_path / "dest")

    assert not (tmp_path / "escape.txt").exists()


def test_copy_out_rejects_symlink_pointing_outside(session_fake, tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
    session_fake.stream_payload = buffer.getvalue()
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    with box.session() as session, pytest.raises(SandboxError, match="unsafe"):
        session.copy_out("link", tmp_path / "dest")

    assert not (tmp_path / "dest" / "link").exists()


def test_copy_out_size_cap_raises(session_fake, tmp_path):
    session_fake.stream_result = (0, "", True)
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    with box.session() as session, pytest.raises(SandboxError, match="max_bytes"):
        session.copy_out("big", tmp_path / "dest", max_bytes=10)


def test_copy_out_failure_raises(session_fake, tmp_path):
    session_fake.stream_result = (2, "tar: nope: No such file", False)
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    with box.session() as session, pytest.raises(SandboxError, match="No such file"):
        session.copy_out("nope", tmp_path / "dest")


def test_read_text_returns_file_contents(session_fake, tmp_path):
    session_fake.stream_payload = tar_of({"notes.txt": "caf\u00e9".encode()})
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    with box.session() as session:
        assert session.read_text("notes.txt") == "caf\u00e9"


def test_read_text_rejects_non_regular_files(session_fake, tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        link = tarfile.TarInfo("notes.txt")
        link.type = tarfile.SYMTYPE
        link.linkname = "other"
        tar.addfile(link)
    session_fake.stream_payload = buffer.getvalue()
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    with box.session() as session, pytest.raises(SandboxError, match="regular file"):
        session.read_text("notes.txt")


def test_session_start_failure_removes_container(monkeypatch, tmp_path):
    calls = []

    def failing(docker_bin, args, *, capture_output=False, timeout=None):
        calls.append(list(args))
        code = 1 if args[0] == "start" else 0
        return subprocess.CompletedProcess(args, code, stdout="", stderr="no")

    monkeypatch.setattr(sandbox_mod, "_docker", failing)

    with pytest.raises(SandboxError, match="start failed"):
        Sandbox("img:test", allow_hosts=(), work_root=tmp_path).session()

    assert calls[-1][:3] == ["rm", "-f", "-v"]
