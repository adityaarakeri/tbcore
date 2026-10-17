# Contributing

## Setup

```bash
uv sync --extra dev
```

This needs [uv](https://docs.astral.sh/uv/). It creates `.venv`, installs tbcore in
editable mode with the dev tools, and downloads Python if needed. Python 3.11+ is
required. Docker is only needed for the sandbox integration tests.

For examples and opt-in tests that read API keys or `TBCORE_*` settings, copy
`.env.example` to `.env` (git-ignored), uncomment what you need, and pass it to uv.
tbcore does not load `.env` itself:

```bash
uv run --env-file .env python examples/codex_sandbox.py
```

## Checks

All of these must pass before a pull request. CI runs the same script:

```bash
uv run python scripts/check.py          # all steps, stopping at the first failure
uv run python scripts/check.py tests    # one or more of: lint, format, types, tests
uv run python scripts/check.py --fix    # apply ruff fixes and formatting, then check
```

The steps are `ruff check .`, `ruff format --check .`, `mypy` (strict; configured in
`pyproject.toml`), and `pytest -m "not installed_wheel and not live"` with coverage.

Style: `from __future__ import annotations`, dataclasses, terse docstrings, 100-column
lines, ASCII only. No blanket `type: ignore` or `Any` casts, no broad `except`, no silent
fallbacks. Public API changes must stay backward compatible: add keyword-only parameters
with defaults that preserve current behavior. Other projects import tbcore.

## Tests

- Tests live in `tests/`; async tests run automatically (`asyncio_mode = auto`).
- Marker `docker`: integration tests that need a working Docker daemon. They skip when
  `tbcore.testing.docker_available()` is false. From an editable install, run them with
  `uv run pytest -m "docker and not installed_wheel and not live"`, or exclude them with
  `uv run pytest -m "not docker"`.
- Marker `installed_wheel`: the complete fake-provider experiment requires an installed
  wheel and runs in the separate Docker CI job; editable-install checks exclude it.
- Marker `live`: paid-provider smoke is opt-in and excluded from default checks.
- Sandbox behavior that does not need a container is unit-tested by mocking
  `tbcore.sandbox._docker` and asserting on the argv (see `tests/test_sandbox_unit.py`).
- Add or update tests for every behavior change; do not call real model APIs.

## Sandbox images

The build contexts are package data in `src/tbcore/images/{sandbox,proxy}`. Base images
are pinned by digest; refresh the digest deliberately (Docker Hub
`library/python` tag `3.11-slim`) and note it in the changelog.

Bump `PROTOCOL_VERSION` in `src/tbcore/images/proxy/proxy.py` whenever the host/proxy
contract changes: a `TBCORE_*` environment variable, the ready line, or the
`TBCORE_USAGE` telemetry format. `Sandbox` refuses a proxy reporting a different value.

Pushing a `v*` tag publishes `ghcr.io/<owner>/tbcore-{sandbox,proxy}:<version>` from
`release.yml` before the PyPI upload. The first release creates the GHCR packages; set
their visibility to public in the package settings so `pull_default_images()` works
without credentials.

## Changelog

Every user-visible change gets an entry under `## [Unreleased]` in `CHANGELOG.md`
(Added / Changed / Fixed / Security, Keep a Changelog format). Call out behavior changes
explicitly. The version lives only in `tbcore.__version__` (`src/tbcore/__init__.py`).

## Security

Do not open public issues for vulnerabilities; see [SECURITY.md](SECURITY.md).
