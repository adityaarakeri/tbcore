"""Docker integration tests: sessions, read-only rootfs, credential-injecting proxy."""

import socket
import subprocess
import time

import pytest

from tbcore.sandbox import Sandbox, SandboxError

pytestmark = pytest.mark.docker

DENY_SCRIPT = (
    "import sys, urllib.request\n"
    "try:\n"
    "    urllib.request.urlopen('https://example.com/', timeout=8)\n"
    "except Exception:\n"
    "    sys.exit(7)\n"
    "sys.exit(0)\n"
)


def docker_ok(*args):
    return subprocess.run(["docker", *args], capture_output=True, encoding="utf-8")


def container_exists(name):
    return docker_ok("inspect", name).returncode == 0


def volume_names():
    return set(docker_ok("volume", "ls", "-q").stdout.split())


# --- sessions ---------------------------------------------------------------------------


@pytest.fixture
def offline_box(sandbox_image):
    box = Sandbox(image=sandbox_image, timeout_s=20.0, allow_hosts=())
    yield box
    box.close()


def test_session_state_persists_across_execs(offline_box):
    with offline_box.session() as session:
        first = session.exec(["bash", "-c", "echo kept > state.txt"])
        second = session.exec(["bash", "-c", "cat state.txt"])

    assert first.exit_code == 0
    assert second.stdout.strip() == "kept"


def test_session_exec_env_is_per_call_and_exit_code_propagates(offline_box):
    with offline_box.session() as session:
        result = session.exec(
            ["bash", "-c", 'printf %s "$TBCORE_S"; exit 4'], env={"TBCORE_S": "per-call"}
        )
        after = session.exec(["bash", "-c", 'printf %s "${TBCORE_S:-gone}"'])

    assert (result.exit_code, result.stdout) == (4, "per-call")
    assert after.stdout == "gone"


def test_session_put_files_belong_to_the_container_user(offline_box):
    with offline_box.session() as session:
        session.put_files({"src/app.py": "print('v1')\n", "data.bin": b"\x00\xff"})
        edit = session.exec(["bash", "-c", "echo '# edited' >> src/app.py"])
        owner = session.exec(["bash", "-c", "stat -c '%u' src/app.py data.bin src; id -u"])
        text = session.read_text("src/app.py")

    assert edit.exit_code == 0, edit.stderr
    assert owner.stdout.split() == ["1000"] * 4
    assert text == "print('v1')\n# edited\n"


def test_session_copy_out_file_and_directory(offline_box, tmp_path):
    with offline_box.session() as session:
        session.exec(["bash", "-c", "mkdir -p out/sub && echo a > out/a.txt && echo b > out/sub/b"])
        one = session.copy_out("out/a.txt", tmp_path / "f")
        tree = session.copy_out("out", tmp_path / "d")

    assert one.read_text() == "a\n"
    assert (tree / "a.txt").read_text() == "a\n"
    assert (tree / "sub" / "b").read_text() == "b\n"


def test_session_copy_out_rejects_symlink_pointing_outside(offline_box, tmp_path):
    with offline_box.session() as session:
        session.exec(["bash", "-c", "mkdir d && ln -s /etc/passwd d/link && ln -s ../../x d/up"])
        with pytest.raises(SandboxError, match="unsafe"):
            session.copy_out("d", tmp_path / "dest")
        with pytest.raises(SandboxError):
            session.read_text("d/link")

    assert not (tmp_path / "dest" / "d" / "link").is_symlink()


def test_session_copy_out_size_cap(offline_box, tmp_path):
    with offline_box.session() as session:
        session.exec(["bash", "-c", "head -c 200000 /dev/zero > big"])
        with pytest.raises(SandboxError, match="max_bytes"):
            session.copy_out("big", tmp_path / "dest", max_bytes=50_000)


def test_session_timeout_really_kills_the_process(offline_box):
    with offline_box.session() as session:
        started = time.monotonic()
        result = session.exec(["bash", "-c", "echo $$ > pid; sleep 300"], timeout_s=2)
        elapsed = time.monotonic() - started
        alive = session.exec(["bash", "-c", "kill -0 $(cat pid)"])
        usable = session.exec(["true"])

    assert result.timed_out is True
    assert result.exit_code is None
    assert elapsed < 15
    assert alive.exit_code != 0
    assert usable.exit_code == 0


def test_session_close_removes_container(offline_box):
    session = offline_box.session()
    name = session.container
    assert container_exists(name)

    session.close()
    session.close()

    assert not container_exists(name)


def test_session_is_hardened_like_run(offline_box):
    with offline_box.session() as session:
        result = session.exec(["bash", "-c", "id -u; grep CapEff /proc/self/status"])
        net = session.exec(["python", "-c", DENY_SCRIPT])

    assert result.stdout.splitlines()[0] == "1000"
    assert result.stdout.split("CapEff:")[1].strip() == "0000000000000000"
    assert net.exit_code == 7


# --- read-only root filesystem ----------------------------------------------------------


@pytest.fixture
def read_only_box(sandbox_image):
    box = Sandbox(image=sandbox_image, timeout_s=20.0, allow_hosts=(), read_only=True)
    yield box
    box.close()


def test_read_only_run_with_files_and_work_writes(read_only_box):
    before = volume_names()
    script = (
        "cat in/x.txt; echo w > made.txt; cat made.txt; echo t > /tmp/t && cat /tmp/t; "
        "if echo no > /rootfile 2>/dev/null; then echo ROOT-WRITABLE; else echo ro-ok; fi; "
        "if touch /usr/x 2>/dev/null; then echo USR-WRITABLE; else echo usr-ro; fi"
    )

    result = read_only_box.run(["bash", "-c", script], files={"in/x.txt": "staged\n"})

    assert result.exit_code == 0, result.stderr
    assert result.stdout.split() == ["staged", "w", "t", "ro-ok", "usr-ro"]
    assert volume_names() == before


def test_read_only_session_put_files_and_cleanup(read_only_box):
    before = volume_names()
    with read_only_box.session() as session:
        session.put_files({"a.txt": "hello\n"})
        wrote = session.exec(["bash", "-c", "echo more >> a.txt && cat a.txt"])
        denied = session.exec(["bash", "-c", "touch /nope"])
        assert volume_names() - before

    assert wrote.stdout.split() == ["hello", "more"]
    assert denied.exit_code != 0
    assert volume_names() == before


# --- credential-injecting proxy ---------------------------------------------------------

HTTP_STATUS_SCRIPT = (
    "import sys, urllib.error, urllib.request\n"
    "req = urllib.request.Request(sys.argv[1], headers={'x-api-key': 'attacker-key'})\n"
    "try:\n"
    "    print(urllib.request.urlopen(req, timeout=20).status)\n"
    "except urllib.error.HTTPError as exc:\n"
    "    print(exc.code)\n"
)
ANTHROPIC_POST_STATUS_SCRIPT = (
    "import json, urllib.error, urllib.request\n"
    "body = json.dumps({'model': 'fixture', 'max_tokens': 1, 'messages': []}).encode()\n"
    "req = urllib.request.Request(\n"
    "    'http://proxy:8080/api.anthropic.com/v1/messages',\n"
    "    data=body,\n"
    "    headers={'x-api-key': 'attacker-key', 'anthropic-version': '2023-06-01',\n"
    "             'content-type': 'application/json'}, method='POST')\n"
    "try:\n"
    "    print(urllib.request.urlopen(req, timeout=20).status)\n"
    "except urllib.error.HTTPError as exc:\n"
    "    print(exc.code)\n"
)
SECRET = "sk-TBCORE-SECRET-1234"


@pytest.fixture
def injecting_box(sandbox_image, proxy_image):
    box = Sandbox(
        image=sandbox_image,
        proxy_image=proxy_image,
        timeout_s=40.0,
        allow_hosts=("upstream.test",),
        inject_headers={
            "api.anthropic.com": {"x-api-key": SECRET},
            "injected.invalid": {"authorization": f"Bearer {SECRET}"},
        },
    )
    yield box
    box.close()


def proxy_container_names():
    out = docker_ok("ps", "--filter", "name=tbcore-proxy-", "--format", "{{.Names}}").stdout
    return out.split()


def test_injected_credentials_stay_out_of_the_sandbox(injecting_box):
    result = injecting_box.run(["bash", "-c", "env; tr '\\0' '\\n' < /proc/self/environ"])

    assert result.exit_code == 0, result.stderr
    assert SECRET not in result.stdout
    assert "ANTHROPIC_BASE_URL=http://proxy:8080/api.anthropic.com" in result.stdout
    assert "ANTHROPIC_API_KEY=sandbox-placeholder" in result.stdout


def test_proxy_gets_injection_without_leaking_it_to_argv_or_logs(injecting_box):
    injecting_box.prepare()
    (name,) = proxy_container_names()

    inspect = docker_ok("inspect", "-f", "{{json .Config.Cmd}} {{json .Args}}", name).stdout
    logs = docker_ok("logs", name)
    env = docker_ok("exec", name, "sh", "-c", "echo $TBCORE_INJECT").stdout
    user = docker_ok("exec", name, "id", "-u").stdout.strip()
    caps = docker_ok("exec", name, "grep", "CapEff", "/proc/self/status").stdout

    assert SECRET not in inspect
    assert SECRET not in logs.stdout + logs.stderr
    assert "inject=['api.anthropic.com', 'injected.invalid']" in logs.stdout
    assert SECRET in env  # the proxy, and only the proxy, has it
    assert user != "0"
    assert caps.split()[1] == "0000000000000000"


def test_connect_to_injected_host_is_refused(injecting_box):
    script = (
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('https://api.anthropic.com/', timeout=15)\n"
        "except Exception as exc:\n"
        "    print(exc)\n"
    )

    result = injecting_box.run(["python", "-c", script])

    assert "403" in result.stdout, (result.stdout, result.stderr)


def test_reverse_port_rejects_hosts_not_in_the_inject_map(injecting_box):
    result = injecting_box.run(
        ["python", "-c", HTTP_STATUS_SCRIPT, "http://proxy:8080/upstream.test/x"]
    )

    assert result.stdout.strip() == "403", (result.stdout, result.stderr)


def test_reverse_port_accepts_injected_host_and_attempts_upstream(injecting_box):
    # injected.invalid never resolves, so the proxy accepted the host and failed upstream.
    result = injecting_box.run(
        ["python", "-c", HTTP_STATUS_SCRIPT, "http://proxy:8080/injected.invalid/v1/x"]
    )

    assert result.stdout.strip() == "502", (result.stdout, result.stderr)


def anthropic_reachable():
    try:
        socket.create_connection(("api.anthropic.com", 443), timeout=5).close()
    except OSError:
        return False
    return True


def test_reverse_path_verifies_tls_and_overwrites_client_key(injecting_box):
    if not anthropic_reachable():
        pytest.skip("no outbound network from the test host")
    # A bogus injected key makes Anthropic answer 401: TLS verified against the proxy
    # image's CA bundle and the request reached the API with the injected header.
    result = injecting_box.run(["python", "-c", ANTHROPIC_POST_STATUS_SCRIPT])

    assert result.stdout.strip() == "401", (result.stdout, result.stderr)
