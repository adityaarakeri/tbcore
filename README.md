```
 ▀█▀ █▄▄ █▀▀ █▀█ █▀█ █▀▀
  █  █▄█ █▄▄ █▄█ █▀▄ ██▄
```

> Run repeatable agent experiments in Docker: capture inputs, track tool calls, token usage
> and cost, grade outputs, and compare baseline vs candidate with evidence for every result.

<!-- badges -->
[![CI](https://github.com/adityaarakeri/tbcore/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adityaarakeri/tbcore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tbcore)](https://pypi.org/project/tbcore/)
[![Python](https://img.shields.io/pypi/pyversions/tbcore)](https://pypi.org/project/tbcore/)
[![Docker 20.10+](https://img.shields.io/badge/docker-20.10%2B-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/engine/install/)
[![Images on ghcr.io](https://img.shields.io/badge/ghcr.io-amd64%20%7C%20arm64-2496ED?logo=github)](https://github.com/adityaarakeri?tab=packages&repo_name=tbcore)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/adityaarakeri/tbcore/blob/main/LICENSE)

tbcore (Token-Build Core) captures a task and its environment, runs baseline and candidate workloads,
grades their outputs, and saves a comparison you can inspect or replay. The same
toolkit includes model-call accounting, resumable batches, and statistical checks.
Start with a complete local experiment: no API key or paid model required.

> **Status:** alpha; `pip install tbcore` (Python 3.11+). The Docker sandbox
> is defense in depth for a trusted local operator, not a hardened multi-tenant
> isolation boundary. Read [SECURITY.md](https://github.com/adityaarakeri/tbcore/blob/main/SECURITY.md) before running hostile code.

## Install

```bash
pip install tbcore          # or: uv pip install tbcore
```

The model client, pool, and stats work without Docker. The sandbox and experiments need
Docker; on a released version, pull the matching published images instead of building them:

```bash
python -c "import tbcore; print(tbcore.pull_default_images())"
```

| | Supported | Checked in CI |
| --- | --- | --- |
| Python | 3.11, 3.12, 3.13 | 3.11 to 3.13 on Linux; 3.12 on macOS and Windows |
| Docker Engine | 20.10+ running Linux containers | Docker tests on GitHub's Ubuntu runner, from the installed wheel |
| Images | `linux/amd64`, `linux/arm64` | Built, pushed, and attested by each release |
| OS | Linux, macOS, Windows | Unit tests on all three; Docker tests on Linux |

Docker Desktop 29.1.5 on Windows has also been tested locally. macOS and Windows CI
runners have no Linux-container Docker, so Docker-backed tests are skipped there.

## Try the experiment

You need Python 3.11+, Docker with a running daemon, and a clone of this repository.
From the repository root:

```bash
python -m pip install -e .
python examples/experiment_workflow.py --store .tbcore/example
```

The example builds the bundled images, runs the same fixture task with baseline
and candidate settings, grades the outputs, and replays the candidate from an
exported bundle. It prints run IDs and writes:

- `.tbcore/example/comparison.md` - the baseline/candidate report.
- `.tbcore/example/candidate-bundle.json` - a portable run bundle.

See the [experiment walkthrough](https://github.com/adityaarakeri/tbcore/blob/main/docs/experiment-workflow.md) for how evidence
is captured, how to resume work, and what the comparison can establish.

On Windows, these commands work in PowerShell when `python` is on `PATH`.
For an isolated install, create a venv with `python -m venv .venv` and use
`.venv\Scripts\python.exe` in place of `python` (on macOS/Linux, `.venv/bin/python`).

## What is included

| Component | What it does |
| --- | --- |
| Experiments | Captures workspaces and environment provenance; links trials, attempts, events, artifacts, evaluations, and baseline/candidate comparisons. |
| Sandbox | Runs commands in disposable Docker containers with resource limits, timeouts, and a network allowlist. Sessions keep a workspace across commands. |
| Client and Pool | Logs model usage, latency, and price estimates; runs rate-limit-aware batches from a resumable SQLite queue. |
| Stats | Computes summaries, bootstrap confidence intervals, and two-sample regression checks. |

The experiment store captures metadata by default. Workspace contents, prompts,
outputs, and other text evidence require an explicit content-capture policy;
review exports before sharing them. See [experiment storage](https://github.com/adityaarakeri/tbcore/blob/main/docs/experiment-store.md)
and [security](https://github.com/adityaarakeri/tbcore/blob/main/SECURITY.md).

Python 3.11+ is required. Docker is needed for sandbox and experiment runs;
the model client, pool, and stats can be used without it. If you only need to
call an API, the provider's official SDK is a better fit. tbcore is for work
that needs the run, its accounting, and its comparison together.

## Model-call quickstart

This example calls Anthropic and requires `ANTHROPIC_API_KEY` on the host. It can
incur provider charges. For a no-key first run, use [the experiment](#try-the-experiment).

```python
from tbcore import Client

client = Client("sonnet")  # API key read from ANTHROPIC_API_KEY
completion = client.complete(
    system="Answer tersely.",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(completion.text, completion.usage, completion.latency_s)
```

Every call is appended as one JSON line to `.tbcore/calls.jsonl` in the working
directory (configurable via `CallLog` or `TBCORE_LOG_PATH`).

## Providers

Anthropic is the default. For Ollama, vLLM, LM Studio, OpenAI, or any other
OpenAI-compatible endpoint, build a spec with `make_model_spec` (or use a
registered shortcut):

```python
from tbcore import Client, make_model_spec

claude = Client("sonnet")  # ANTHROPIC_API_KEY
local = Client(make_model_spec("qwen3:8b", provider="ollama"))  # localhost:11434
served = Client(make_model_spec("org/model", provider="vllm"))  # localhost:8000
gpt = Client("gpt-4o-mini")  # OPENAI_API_KEY
```

Notes:

- An API key is sent as `Authorization: Bearer ...` on the OpenAI-compatible
  wire when available; Ollama/vLLM need none by default.
- `cache_prefix` uses Anthropic prompt caching and raises on the
  OpenAI-compatible wire; `tools` is Anthropic-wire only for now.
- Cost-equivalent uses registered prices. Models without registered prices (local
  models, unknown ids) log `cost_equiv: null`, not `0`; check `ModelSpec.priced`.
- `Client(..., base_url=...)` overrides the endpoint on both wires.
- Registered prices are snapshots, not provider invoices; check current rates
  before using cost estimates to set a budget.
- On the OpenAI-compatible wire, `Usage.input_tokens` excludes cached tokens (reported
  separately as `cache_read_input_tokens`), matching the Anthropic wire.

## Pool

Queue work in a SQLite file, then drain it with bounded concurrency. Re-running after a
crash resumes: finished rows are never redone.

```python
import asyncio

from tbcore import Client, ExecutionPool, WorkStore, enqueue

store = WorkStore("work.db")
enqueue(
    store,
    [
        {"messages": [{"role": "user", "content": f"Summarize item {i}"}], "max_tokens": 200}
        for i in range(100)
    ],
)

pool = ExecutionPool(Client("haiku"), store, concurrency=8)
print(asyncio.run(pool.run()))

# After a crash or Ctrl-C, run again with the same work.db: only unfinished rows run.
```

Payloads take `messages` (required) and optionally `system`, `cache_prefix`, `max_tokens`,
`temperature`, and `tools`. Rate-limit and transient errors are retried with exponential
backoff; a server `Retry-After` hint is honored, and a 429 pauses every worker until the
backoff window ends. Results (text, usage including cache tokens, latency) are stored in
the `result` column of the `work` table.

## Sandbox

Run model-generated commands in a disposable container with resource limits and
a network allowlist. The bundled image runs non-root, and tbcore drops container
capabilities. Docker is required. This is defense in depth; read
[SECURITY.md](https://github.com/adityaarakeri/tbcore/blob/main/SECURITY.md) for the complete threat model.

```python
from tbcore import Sandbox, build_default_images

# Builds tbcore/sandbox:latest and tbcore/proxy:latest from contexts bundled in the package.
images = build_default_images()

with Sandbox(images["sandbox"], timeout_s=30, allow_hosts=(), read_only=True) as box:
    result = box.run(
        ["python", "main.py"],
        files={"main.py": "import os; print(os.environ['GREETING'])"},
        env={"GREETING": "hello"},
    )
    assert result.exit_code == 0
    print(result.stdout.strip())  # hello
```

`allow_hosts=()` gives the container no network. The default allowlist is
`api.anthropic.com`; set it explicitly for each workload. `read_only=True` is
recommended, while the default remains `False` for compatibility.

### Published images

Each release also publishes the sandbox and proxy images to GitHub Container Registry as
`ghcr.io/adityaarakeri/tbcore-{sandbox,proxy}:<version>` (amd64 and arm64, with build
provenance attestations). `pull_default_images()` pulls the ones matching your installed
tbcore and returns digest-pinned references; use it instead of building when you run a
released version:

```python
from tbcore import Sandbox, pull_default_images

images = pull_default_images()  # {"sandbox": "ghcr.io/...@sha256:...", "proxy": "..."}

with Sandbox(images["sandbox"], proxy_image=images["proxy"], allow_hosts=()) as box:
    print(box.run(["python", "-c", "print('hello')"]).stdout)
```

tbcore never pulls the proxy implicitly. The proxy reports a protocol version when it
starts, and `Sandbox` refuses a proxy image whose version does not match this tbcore, so
an outdated local `tbcore/proxy:latest` fails with a rebuild hint instead of misbehaving.
Unreleased checkouts and offline hosts keep using `build_default_images()`.

`image_context("sandbox")` / `image_context("proxy")` return the bundled build contexts if
you want to build them yourself (or extend them with `build_image`). Containers run with all
capabilities dropped, as the image's own `USER` (uid 1000 in the bundled image); pass
`Sandbox(..., user="1000:1000")` to force a non-root user on images that default to root.

### Keep provider credentials in the proxy

Passing an API key via `env=` hands it to the untrusted code. `inject_headers` keeps it in the
proxy instead: the proxy sets the headers on every request to that host, and discards any
`Authorization` / `x-api-key` the sandbox sends.

```python
import os

from tbcore import Sandbox, build_default_images

images = build_default_images()

with Sandbox(
    images["sandbox"],
    inject_headers={"api.anthropic.com": {"x-api-key": os.environ["ANTHROPIC_API_KEY"]}},
) as box:
    print(box.base_url("api.anthropic.com"))  # http://proxy:8080/api.anthropic.com
```

The Anthropic SDK refuses to send a request without some key, which is why
`ANTHROPIC_API_KEY=sandbox-placeholder` is set; the proxy overwrites it on the wire. This
automatic setup also applies to OpenAI injection: `OPENAI_BASE_URL` points to the proxy's
`/v1` path and `OPENAI_API_KEY` is a placeholder. Codex CLI additionally needs a custom
Responses provider configured to use that URL (see below).
For other hosts or SDKs, point the client's base URL at `box.base_url(host)` (plain HTTP to the proxy,
which speaks verified HTTPS to the real host). Injected hosts are not reachable through
`HTTPS_PROXY`/CONNECT, so code cannot reach them with a key of its own. This protects against key
theft and against sending data to an attacker's account via that API. It does not stop sandboxed
code from using your key through the proxy (requests are billed to, and land in, your account).

### Codex CLI in a sandbox

Run `python examples/codex_sandbox.py` with `OPENAI_API_KEY` set on the host. Set
`TBCORE_CODEX_MODEL` to choose a model. The example
builds the optional `image_context("codex")` image and the proxy, creates a writable
session workspace, asks Codex to edit a file, checks the timeout, exit status, and file
contents. The default `build_default_images()` continues to build only the
Python sandbox and proxy; build the Codex image explicitly with `build_image`.

Codex's custom provider uses `base_url = "http://proxy:8080/api.openai.com/v1"`,
`env_key = "OPENAI_API_KEY"`, `wire_api = "responses"`, and
`supports_websockets = false`. The sandbox receives a placeholder key while the proxy adds
the host's real Bearer key to upstream requests. The proxy handles HTTP streaming; this
initial integration uses API-key auth only, without ChatGPT sign-in or WebSocket transport.
The Codex CLI's internal `danger-full-access` setting lets it reach the proxy and edit
`/work` inside the tbcore container. Docker remains the outer isolation boundary;
this setting does not change the Docker mounts or network allowlist. Set a finite
session timeout and close it after each workload. See
[OpenAI Docs configuration](https://developers.openai.com/codex/config-reference)
and [non-interactive mode](https://developers.openai.com/codex/noninteractive).

### Sessions

`run()` starts a fresh container per call. For an agent loop that works on one workspace, use a
session (one long-lived container, same hardening and limits):

```python
from tbcore import Sandbox, build_default_images

images = build_default_images()
with Sandbox(images["sandbox"], allow_hosts=()) as box, box.session() as session:
    session.put_files({"app.py": "print('v1')\n"})  # files belong to the container user
    session.exec(["python", "app.py"], timeout_s=10)  # killed in-container at the deadline
    print(session.read_text("app.py"))
```

Sessions replace the image's `ENTRYPOINT`/`CMD` with `sleep infinity`, and `exec`
needs coreutils `timeout` in the image. `session.copy_out(...)` transfers results
to the host with a default 100 MB size cap and rejects links outside the destination.

### Stronger isolation

Use `Sandbox(images["sandbox"], runtime="runsc", read_only=True)` when gVisor is
installed and registered with Docker.

- `read_only=True` mounts the root filesystem read-only (a 64 MB `/tmp` tmpfs and an anonymous
  `/work` volume stay writable), so code cannot modify binaries or libraries in its container.
  It is recommended; the default stays `False` for compatibility. Custom images need a `/work`
  directory writable by their user.
- `runtime="runsc"` runs sandbox containers under [gVisor](https://gvisor.dev/), which puts a
  user-space kernel between the code and the host kernel. It reduces the shared-kernel
  attack surface; it is not a guarantee. Install `runsc` and register it with Docker first
  (see [gVisor's Docker quick start](https://gvisor.dev/docs/user_guide/quick_start/docker/)).
  The proxy container always uses the default runtime.

### Container upgrades and maintenance

Image build contexts ship with the package under [`src/tbcore/images`](https://github.com/adityaarakeri/tbcore/tree/main/src/tbcore/images).
Base images are digest-pinned. Rebuild with a new tag, test the exact candidate image,
and record its image ID before switching workloads; updating tbcore does not rebuild
existing images or replace running containers. The optional Codex image must be built
separately with an exact `CODEX_VERSION`. See [CONTRIBUTING.md](https://github.com/adityaarakeri/tbcore/blob/main/CONTRIBUTING.md) for
the required checks and [the experiment guide](https://github.com/adityaarakeri/tbcore/blob/main/docs/experiment-workflow.md) for
recording environment provenance and recovery.

## Agent experiments

| Path | Status |
| --- | --- |
| Offline Docker fixture | Baseline/candidate, grade, export, and replay example with an installed-wheel integration test. |
| Codex CLI with an OpenAI API key | Experimental sandbox example; paid live-provider smoke is opt-in. |
| Claude Code, OpenCode, agy | No end-to-end managed-agent experiment fixture yet. |
| Kubernetes or managed execution | Deferred; Docker is the first backend. |

The [workflow guide](https://github.com/adityaarakeri/tbcore/blob/main/docs/experiment-workflow.md) covers telemetry gaps, recovery,
cost limits, and the benchmark recipe. Local samples and their limits are in
[`docs/performance-results.json`](https://github.com/adityaarakeri/tbcore/blob/main/docs/performance-results.json). CI includes
a separate installed-wheel Docker job.

## Stats

```python
from tbcore import regressed, summary

print(summary([0.8, 0.9, 0.85, 0.7]))  # mean, stddev, bootstrap CI

# Lower-is-better metric (cost, latency): is the candidate significantly higher?
print(regressed([1.0, 1.1, 0.9], [1.2, 1.3, 1.1]).regressed)

# Higher-is-better metric (accuracy): is the candidate significantly lower?
print(regressed([0.9, 0.8, 0.9], [0.7, 0.8, 0.7], higher_is_better=True).regressed)
```

`effect_size` is Cohen's d with a fixed sign convention (positive means the candidate mean
is higher) and is `nan` when undefined (too little data or zero variance).
`bootstrap_ci` and `cohens_d` are exported too.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy
python -m pytest -m "not installed_wheel and not live" --cov=tbcore --cov-report=term-missing
```

See [CONTRIBUTING.md](https://github.com/adityaarakeri/tbcore/blob/main/CONTRIBUTING.md) for test markers and image maintenance.
CI runs these checks on Python 3.11-3.13 (Linux), plus 3.12 on Windows and
macOS, with a separate installed-wheel Docker job. Docker tests skip when no
container runtime is available; paid live-provider tests are excluded.

## License

MIT - see [LICENSE](https://github.com/adityaarakeri/tbcore/blob/main/LICENSE).
