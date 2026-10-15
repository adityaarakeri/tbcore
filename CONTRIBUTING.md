# Contributing

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"      # Windows: .venv\Scripts\python
```

Python 3.11+ is required. Docker is only needed for the sandbox integration tests.

## Checks

All of these must pass before a pull request (CI runs them too):

```bash
ruff check .
ruff format --check .      # `ruff format .` to fix
mypy                       # strict; configured in pyproject.toml
pytest -m "not installed_wheel and not live" --cov=tbcore --cov-report=term-missing
```

Style: `from __future__ import annotations`, dataclasses, terse docstrings, 100-column
lines, ASCII only. No blanket `type: ignore` or `Any` casts, no broad `except`, no silent
fallbacks. Public API changes must stay backward compatible: add keyword-only parameters
with defaults that preserve current behavior. Other projects import tbcore.

## Tests

- Tests live in `tests/`; async tests run automatically (`asyncio_mode = auto`).
- Marker `docker`: integration tests that need a working Docker daemon. They skip when
  `tbcore.testing.docker_available()` is false. From an editable install, run them with
  `pytest -m "docker and not installed_wheel and not live"`, or exclude them with
  `pytest -m "not docker"`.
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
