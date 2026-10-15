"""Docker-free sandbox tests: assert on the docker argv the sandbox builds."""

import json
import os
import subprocess

import pytest

from tbcore import sandbox as sandbox_mod
from tbcore.sandbox import (
    PROXY_PIDS_LIMIT,
    Sandbox,
    SandboxError,
    build_default_images,
    image_context,
)


class FakeDocker:
    """Records docker invocations and answers with canned CompletedProcess results."""

    def __init__(self):
        self.calls = []
        self.env_file_contents = None
        self.env_file_mode = None
        self.env_file_path = None

    def __call__(self, docker_bin, args, *, capture_output=False, timeout=None):
        args = list(args)
        self.calls.append(args)
        stdout = ""
        if args[0] in ("create", "run") and "--env-file" in args:
            path = args[args.index("--env-file") + 1]
            self.env_file_path = path
            with open(path, encoding="utf-8", newline="") as fh:
                self.env_file_contents = fh.read()
            self.env_file_mode = os.stat(path).st_mode & 0o777
        if args[0] == "start":
            stdout = "out\n"
        if args[0] == "logs":
            stdout = f"{sandbox_mod._PROXY_READY_MARKER}\n"
        return subprocess.CompletedProcess([docker_bin, *args], 0, stdout=stdout, stderr="")

    def first(self, verb):
        return next(call for call in self.calls if call[0] == verb)


@pytest.fixture
def fake_docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(sandbox_mod, "_docker", fake)
    return fake


def test_create_args_are_hardened(fake_docker, tmp_path):
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)

    result = box.run(["echo", "hi"])

    create = fake_docker.first("create")
    assert result.exit_code == 0
    assert create[create.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in create
    # Default keeps the image's own USER rather than forcing one.
    assert "--user" not in create
    assert create[create.index("--network") + 1] == "none"
    assert create[-3:] == ["img:test", "echo", "hi"]


@pytest.mark.parametrize("content", ["unsafe text", b"unsafe bytes"])
@pytest.mark.parametrize(
    "bad_path",
    [
        "../outside.txt",
        "../new.txt",
        "a/../../outside.txt",
        "/outside.txt",
        "C:/outside.txt",
        "C:\\outside.txt",
        "\\\\server\\share\\outside.txt",
        "a\\outside.txt",
    ],
)
def test_run_rejects_unsafe_file_paths_before_host_write(
    fake_docker, monkeypatch, tmp_path, bad_path, content
):
    outside = tmp_path / "outside.txt"
    outside.write_text("keep me", encoding="utf-8")
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    writes = []

    def record_text(target, data, **kwargs):
        writes.append(target)
        return len(data)

    def record_bytes(target, data):
        writes.append(target)
        return len(data)

    monkeypatch.setattr(type(outside), "write_text", record_text)
    monkeypatch.setattr(type(outside), "write_bytes", record_bytes)

    with pytest.raises(ValueError):
        box.run(["true"], files={"nested/safe.txt": "safe", bad_path: content})

    assert writes == []
    assert outside.read_text(encoding="utf-8") == "keep me"
    assert not (tmp_path / "new.txt").exists()
    assert not any(call[0] in ("cp", "start") for call in fake_docker.calls)


def test_run_stages_nested_relative_text_and_bytes(fake_docker, monkeypatch, tmp_path):
    def inspect_copy(docker_bin, args, **kwargs):
        if args[0] == "cp":
            staging = tmp_path / args[1].split("/")[-2]
            assert (staging / "nested/hello.txt").read_text(encoding="utf-8") == "hello"
            assert (staging / "nested/data.bin").read_bytes() == b"\x00\x01"
        return fake_docker(docker_bin, args, **kwargs)

    monkeypatch.setattr(sandbox_mod, "_docker", inspect_copy)
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)
    result = box.run(["true"], files={"nested/hello.txt": "hello", "nested/data.bin": b"\x00\x01"})

    assert result.exit_code == 0
    assert any(call[0] == "cp" for call in fake_docker.calls)


def test_user_is_configurable(fake_docker, tmp_path):
    Sandbox("img:test", allow_hosts=(), work_root=tmp_path, user="2000:2000").run(["true"])

    create = fake_docker.first("create")
    assert create[create.index("--user") + 1] == "2000:2000"


def test_env_goes_through_env_file_not_argv(fake_docker, tmp_path):
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)

    box.run(["true"], env={"API_KEY": "hunter2", "MODE": "x=y"})

    create = fake_docker.first("create")
    assert not any("hunter2" in arg for arg in create)
    assert "-e" not in create
    assert fake_docker.env_file_contents == "API_KEY=hunter2\nMODE=x=y\n"
    if os.name == "posix":
        assert fake_docker.env_file_mode == 0o600


def test_env_file_is_deleted_after_run(fake_docker, tmp_path):
    Sandbox("img:test", allow_hosts=(), work_root=tmp_path).run(["true"], env={"A": "1"})

    assert fake_docker.env_file_path is not None
    assert not os.path.exists(fake_docker.env_file_path)
    assert list(tmp_path.iterdir()) == []


def test_env_file_is_deleted_when_create_fails(monkeypatch, tmp_path):
    seen = {}

    def failing(docker_bin, args, *, capture_output=False, timeout=None):
        if args[0] == "create":
            seen["path"] = args[list(args).index("--env-file") + 1]
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="boom")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(sandbox_mod, "_docker", failing)

    with pytest.raises(SandboxError):
        Sandbox("img:test", allow_hosts=(), work_root=tmp_path).run(["true"], env={"A": "1"})

    assert not os.path.exists(seen["path"])


def test_no_env_file_without_env(fake_docker, tmp_path):
    Sandbox("img:test", allow_hosts=(), work_root=tmp_path).run(["true"])

    assert "--env-file" not in fake_docker.first("create")


@pytest.mark.parametrize(
    ("key", "value"), [("A=B", "x"), ("", "x"), ("#C", "x"), ("A", "line1\nline2"), ("A", "a\x00b")]
)
def test_env_file_rejects_unsafe_entries(fake_docker, tmp_path, key, value):
    box = Sandbox("img:test", allow_hosts=(), work_root=tmp_path)

    with pytest.raises(ValueError):
        box.run(["true"], env={key: value})

    assert not any(call[0] == "create" for call in fake_docker.calls)
    assert list(tmp_path.iterdir()) == []


def test_proxy_container_has_resource_limits_and_dropped_caps(fake_docker, tmp_path):
    box = Sandbox("img:test", allow_hosts=("example.com",), work_root=tmp_path)

    box.prepare()

    run = fake_docker.first("run")
    assert run[run.index("--memory") + 1] == "128m"
    assert run[run.index("--pids-limit") + 1] == PROXY_PIDS_LIMIT
    assert "--cpus" in run
    assert run[run.index("--cap-drop") + 1] == "ALL"
    assert "TBCORE_ALLOW_HOSTS=example.com" in run


def test_proxy_telemetry_identity_is_bound_by_host(fake_docker, tmp_path):
    box = Sandbox(
        "img:test",
        allow_hosts=("api.openai.com",),
        work_root=tmp_path,
        telemetry_run_id="run-a",
        telemetry_attempt_id="attempt-a",
    )
    box.prepare()
    run = fake_docker.first("run")
    assert "TBCORE_RUN_ID=run-a" in run
    assert "TBCORE_ATTEMPT_ID=attempt-a" in run
    assert "TBCORE_RUN_ID=run-a" not in box._create_args("work", None)

    prior = sandbox_mod._docker

    def with_usage(docker_bin, args, **kwargs):
        result = prior(docker_bin, args, **kwargs)
        if args[0] == "logs":
            usage = {"run_id": "run-a", "attempt_id": "attempt-a", "call_id": "one"}
            return subprocess.CompletedProcess(
                args, 0, stdout="TBCORE_USAGE " + json.dumps(usage), stderr=""
            )
        return result

    sandbox_mod._docker = with_usage
    try:
        assert box.proxy_usage() == [
            {"run_id": "run-a", "attempt_id": "attempt-a", "call_id": "one"}
        ]
    finally:
        sandbox_mod._docker = prior


def test_proxy_budget_config_is_host_owned(fake_docker, tmp_path):
    box = Sandbox(
        "img:test",
        allow_hosts=("api.openai.com",),
        work_root=tmp_path,
        telemetry_run_id="run-a",
        telemetry_attempt_id="attempt-a",
        proxy_budget={"max_model_calls": 1},
    )
    box.prepare()
    assert 'TBCORE_PROXY_BUDGET={"max_model_calls": 1}' in fake_docker.env_file_contents
    assert "TBCORE_PROXY_BUDGET" not in " ".join(fake_docker.first("run"))


def test_network_mode_with_allowlist_uses_proxy_env(fake_docker, tmp_path):
    box = Sandbox("img:test", allow_hosts=("example.com",), work_root=tmp_path)

    box.run(["true"])

    create = fake_docker.first("create")
    assert "none" not in create
    assert any(arg.startswith("HTTPS_PROXY=http://proxy:") for arg in create)


@pytest.mark.parametrize("name", ["sandbox", "proxy", "codex"])
def test_bundled_image_contexts_exist(name):
    context = image_context(name)

    dockerfile = (context / "Dockerfile").read_text(encoding="utf-8")
    assert "@sha256:" in dockerfile


def test_proxy_context_includes_proxy_script():
    assert (image_context("proxy") / "proxy.py").is_file()


def test_image_context_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown image"):
        image_context("nope")


def test_build_default_images_builds_both_with_prefix(fake_docker):
    tags = build_default_images(tag_prefix="acme")

    assert tags == {"sandbox": "acme/sandbox:latest", "proxy": "acme/proxy:latest"}
    builds = [call for call in fake_docker.calls if call[0] == "build"]
    assert [call[2] for call in builds] == ["acme/sandbox:latest", "acme/proxy:latest"]
    assert builds[0][3].endswith("sandbox")
    assert builds[1][3].endswith("proxy")


def test_default_prefix_matches_sandbox_proxy_image(fake_docker):
    tags = build_default_images()

    assert tags["proxy"] == Sandbox("x").proxy_image
