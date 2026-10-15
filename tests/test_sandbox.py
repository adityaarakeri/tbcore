import subprocess
import time
import uuid

import pytest

from tbcore.sandbox import Sandbox

pytestmark = pytest.mark.docker

DENY_SCRIPT = (
    "import sys, urllib.request\n"
    "try:\n"
    "    urllib.request.urlopen('https://example.com/', timeout=8)\n"
    "except Exception:\n"
    "    sys.exit(7)\n"
    "sys.exit(0)\n"
)

ALLOW_SCRIPT = (
    "import urllib.request\n"
    "print(urllib.request.urlopen('http://upstream.test:8000/hello.txt', "
    "timeout=8).read().decode())\n"
)


@pytest.fixture
def sandbox(sandbox_image, proxy_image):
    box = Sandbox(
        image=sandbox_image,
        proxy_image=proxy_image,
        timeout_s=15.0,
        allow_hosts=("upstream.test",),
    )
    yield box
    box.close()


def test_run_captures_stdout_stderr_and_exit_code(sandbox):
    result = sandbox.run(["bash", "-c", "echo out-123; echo err-456 >&2"])

    assert result.exit_code == 0
    assert not result.timed_out
    assert "out-123" in result.stdout
    assert "err-456" in result.stderr


def test_nonzero_exit_code_is_propagated(sandbox):
    result = sandbox.run(["bash", "-c", "exit 3"])

    assert result.exit_code == 3


def test_files_are_available_in_the_workdir(sandbox):
    result = sandbox.run(
        ["bash", "-c", "cat input/hello.txt"], files={"input/hello.txt": "hi there"}
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "hi there"


def test_runs_as_non_root_with_no_capabilities(sandbox):
    result = sandbox.run(["bash", "-c", "id -u; grep CapEff /proc/self/status"])

    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines()[0] == "1000"
    assert result.stdout.split("CapEff:")[1].strip() == "0000000000000000"


def test_env_vars_reach_the_command(sandbox):
    result = sandbox.run(
        ["bash", "-c", 'printf %s "$TBCORE_TEST_SECRET"'], env={"TBCORE_TEST_SECRET": "s3cret"}
    )

    assert result.stdout == "s3cret"


def test_filesystem_is_fresh_per_run(sandbox):
    first = sandbox.run(["bash", "-c", "touch marker-from-run-a"])
    second = sandbox.run(["bash", "-c", "test -e marker-from-run-a"])

    assert first.exit_code == 0
    assert second.exit_code != 0


def test_run_exceeding_wall_clock_limit_is_killed(require_docker, sandbox_image):
    box = Sandbox(image=sandbox_image, timeout_s=3.0, allow_hosts=())
    try:
        started = time.monotonic()
        result = box.run(["bash", "-c", "while true; do :; done"])
        elapsed = time.monotonic() - started
    finally:
        box.close()

    assert result.timed_out is True
    assert result.exit_code is None
    assert elapsed < 30.0


def test_outbound_request_to_non_allowed_host_fails(sandbox):
    result = sandbox.run(["python", "-c", DENY_SCRIPT])

    assert result.exit_code == 7


def test_allowed_host_is_reachable_through_the_proxy(sandbox, sandbox_image):
    sandbox.prepare()
    upstream = f"tbcore-upstream-{uuid.uuid4().hex[:8]}"
    proc = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            upstream,
            "--network",
            sandbox.network_name,
            "--network-alias",
            "upstream.test",
            "-w",
            "/tmp",
            sandbox_image,
            "bash",
            "-c",
            "printf 'tbcore-ok' > hello.txt && exec python -m http.server 8000",
        ],
        capture_output=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    try:
        result = sandbox.run(["python", "-c", ALLOW_SCRIPT])
    finally:
        subprocess.run(["docker", "rm", "-f", upstream], capture_output=True)

    assert result.exit_code == 0, (result.stdout, result.stderr)
    assert "tbcore-ok" in result.stdout


def test_sandbox_requires_nonempty_command(require_docker, sandbox_image):
    box = Sandbox(image=sandbox_image)
    with pytest.raises(ValueError):
        box.run([])
