"""Explicitly opt-in provider smoke; never runs in default CI."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from examples.codex_sandbox import codex_config

import tbcore
from tbcore import Sandbox, build_codex_image, build_image, image_context

pytestmark = [pytest.mark.docker, pytest.mark.live]


def test_live_codex_provider_smoke(require_docker) -> None:
    if os.environ.get("TBCORE_RUN_LIVE_CODEX") != "1":
        pytest.skip("set TBCORE_RUN_LIVE_CODEX=1 for paid provider smoke")
    key = os.environ["OPENAI_API_KEY"]
    version = os.environ["TBCORE_CODEX_VERSION"]
    report_path = Path(os.environ["TBCORE_LIVE_REPORT"])
    model = os.environ.get("TBCORE_CODEX_MODEL", "gpt-6.1-sol")
    image = build_codex_image(version)
    image_id = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    proxy_image = build_image(image_context("proxy"), "tbcore/proxy:live-smoke")
    with Sandbox(
        image,
        proxy_image=proxy_image,
        allow_hosts=(),
        read_only=True,
        inject_headers={"api.openai.com": {"Authorization": f"Bearer {key}"}},
        proxy_budget={"max_model_calls": 4, "max_total_tokens": 2000},
        timeout_s=120,
    ) as box:
        config = codex_config(f"{box.base_url('api.openai.com')}/v1", model)
        with box.session() as session:
            session.put_files({".codex/config.toml": config, "hello.txt": "before\n"})
            cli = session.exec(("codex", "--version"), timeout_s=15)
            assert cli.exit_code == 0, cli.stderr
            result = session.exec(
                (
                    "codex",
                    "exec",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "Change hello.txt to contain exactly 'after' followed by a newline.",
                ),
                timeout_s=90,
            )
            assert result.exit_code == 0 and not result.timed_out, result.stderr
            assert session.read_text("hello.txt") == "after\n"
        usage = box.proxy_usage()
    assert usage and all(item["usage_complete"] for item in usage)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "tbcore_version": tbcore.__version__,
                "cli_version": cli.stdout.strip(),
                "image_id": image_id,
                "model": model,
                "model_calls": len(usage),
                "input_tokens": sum(item["input_tokens"] for item in usage),
                "output_tokens": sum(item["output_tokens"] for item in usage),
                "proxy_budget": {"max_model_calls": 4, "max_total_tokens": 2000},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
