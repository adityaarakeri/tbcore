"""Codex provider configuration and optional image checks."""

import os
import tomllib

import pytest
from examples.codex_sandbox import codex_config

from tbcore.sandbox import Sandbox, build_codex_image


def test_codex_config_targets_http_responses_without_real_key():
    config_text = codex_config("http://proxy:8080/api.openai.com/v1", "gpt-6.1-sol")

    config = tomllib.loads(config_text)
    provider = config["model_providers"]["tbcore_proxy"]
    assert config["model_provider"] == "tbcore_proxy"
    assert config["model"] == "gpt-6.1-sol"
    assert config["approval_policy"] == "never"
    assert config["sandbox_mode"] == "danger-full-access"
    assert provider["base_url"] == "http://proxy:8080/api.openai.com/v1"
    assert provider["env_key"] == "OPENAI_API_KEY"
    assert provider["wire_api"] == "responses"
    assert provider["supports_websockets"] is False
    assert provider["requires_openai_auth"] is False
    assert "sk-" not in config_text


@pytest.mark.docker
def test_codex_image_supports_session_edits_and_timeout(require_docker):
    version = os.environ.get("TBCORE_CODEX_VERSION")
    if not version:
        pytest.skip("set TBCORE_CODEX_VERSION to an exact CLI version for the optional image test")
    image = build_codex_image(version, tag=f"tbcore-dev/codex:{version}")

    with (
        Sandbox(image, allow_hosts=(), read_only=True, timeout_s=15) as box,
        box.session() as session,
    ):
        session.put_files({"hello.txt": "before\n"})
        version = session.exec(["codex", "--version"])
        edit = session.exec(["sh", "-c", "printf 'after\\n' > hello.txt"])
        result = session.read_text("hello.txt")
        timeout = session.exec(["sleep", "30"], timeout_s=1)

    assert version.exit_code == 0, version.stderr
    assert "codex" in version.stdout.lower()
    assert edit.exit_code == 0, edit.stderr
    assert result == "after\n"
    assert timeout.timed_out is True
    assert timeout.exit_code is None
