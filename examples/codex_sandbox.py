"""Run Codex CLI against a disposable workspace using proxy-held API credentials.

Set OPENAI_API_KEY and an exact TBCORE_CODEX_VERSION on the host, then run
``uv run python examples/codex_sandbox.py``.
Requires Docker and an OpenAI API key with access to the selected model.
"""

import json
import os

from tbcore import Sandbox, build_codex_image, build_image, image_context


def codex_config(base_url: str, model: str) -> str:
    """Configure Codex's HTTP Responses transport for the injected proxy host."""
    return f"""model = {json.dumps(model)}
model_provider = "tbcore_proxy"
approval_policy = "never"
sandbox_mode = "danger-full-access"

[model_providers.tbcore_proxy]
name = "tbcore proxy"
base_url = {json.dumps(base_url)}
env_key = "OPENAI_API_KEY"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
"""


def main() -> None:
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("TBCORE_CODEX_MODEL", "gpt-6.1-sol")
    codex_image = build_codex_image(os.environ["TBCORE_CODEX_VERSION"])
    proxy_image = build_image(image_context("proxy"), "tbcore/proxy:latest")

    with Sandbox(
        codex_image,
        proxy_image=proxy_image,
        allow_hosts=(),
        inject_headers={"api.openai.com": {"Authorization": f"Bearer {api_key}"}},
        timeout_s=120,
        read_only=True,
    ) as box:
        # Codex uses Responses over HTTP at the proxy's /v1 endpoint. The environment
        # contains only a placeholder key; the real key stays in the proxy container.
        config = codex_config(f"{box.base_url('api.openai.com')}/v1", model)
        with box.session() as session:
            session.put_files(
                {
                    ".codex/config.toml": config,
                    "hello.txt": "before\n",
                }
            )
            result = session.exec(
                [
                    "codex",
                    "exec",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "Change hello.txt to contain exactly 'after' followed by a newline.",
                ],
                timeout_s=90,
            )
            if result.timed_out:
                raise TimeoutError(f"Codex exceeded 90 seconds:\n{result.stderr}")
            if result.exit_code != 0:
                raise RuntimeError(f"Codex failed ({result.exit_code}):\n{result.stderr}")
            edited = session.read_text("hello.txt")
            if edited != "after\n":
                raise RuntimeError(f"Codex did not make the requested edit: {edited!r}")
            print(edited, end="")


if __name__ == "__main__":
    main()
