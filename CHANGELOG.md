# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- OpenAI-compatible wire (`openai-compat`): talk to Ollama, vLLM, LM Studio,
  OpenAI, or any `/v1/chat/completions` endpoint via `make_model_spec(...)`
  and `Client(spec)`; Bearer auth is sent when an API key is present.
- `temperature` passthrough and `tools` passthrough (Anthropic wire).
- Registry shortcuts `gpt-4o` and `gpt-4o-mini` targeting OpenAI cloud with
  registered prices; `vllm` provider convenience in `make_model_spec`.
- Sandbox image build contexts now ship inside the package
  (`tbcore/images/sandbox`, `tbcore/images/proxy`) with new helpers
  `image_context(name)` and `build_default_images(tag_prefix="tbcore")`.
- `Sandbox(..., user=None)` keyword argument to force a container user (`None` keeps
  the image's `USER`).
- `Sandbox(..., inject_headers={host: {header: value}})`: the proxy sets these headers on
  requests to the host through a new reverse-proxy listener (`http://proxy:8080/<host>/...`,
  verified HTTPS upstream, SSE streaming supported), so API keys never enter the sandbox.
  Injected hosts are added to the allowlist but refused on the CONNECT path. For
  `api.anthropic.com` the sandbox gets `ANTHROPIC_BASE_URL` and a placeholder
  `ANTHROPIC_API_KEY` (the SDK requires a key to be set). `Sandbox.base_url(host)` and
  `Sandbox.injected_hosts` expose the wiring for other SDKs.
- `Sandbox.session()` returns a `SandboxSession` (context manager): one long-lived container
  with `exec(command, env=, timeout_s=)` (in-container `timeout`), `put_files`,
  `copy_out(path, dest, max_bytes=)` (size-capped, `tarfile` data filter) and `read_text`.
- `Sandbox(..., runtime=None)` selects a docker runtime such as gVisor's `runsc`;
  `Sandbox(..., read_only=False)` mounts the root filesystem read-only with a `/tmp` tmpfs and
  an anonymous `/work` volume.
- `ModelSpec.priced`; registry entry `fable` (`claude-fable-5-1`).
- `RetryableError` base class and `retry_after` attribute (seconds) on
  `RateLimitError`/`TransientError`, parsed from `Retry-After` (seconds or HTTP date)
  on 429/5xx responses; `parse_retry_after` helper.
- `ExecutionPool(..., clock=...)` for tests; the pool forwards `max_tokens`,
  `temperature`, and `tools` from the payload when present.
- `regressed(..., higher_is_better=False)`; `bootstrap_ci` and `cohens_d` exported
  from `tbcore`.
- `TBCORE_LOG_PATH` environment variable overrides the call log location when no
  path is passed to `CallLog`. The default stays `.tbcore/calls.jsonl`.
- `py.typed` marker, mypy (strict) config, `SECURITY.md`, `CONTRIBUTING.md`, release
  workflow (trusted publishing), pytest-cov, and a Docker CI job.

### Changed

- Statistics reject invalid confidence levels and nonpositive or nonintegral bootstrap
  iteration counts with `ValueError`, including single-sample summaries.
- Bootstrap resampling runs in bounded batches instead of allocating all resamples at once,
  preserving seeded results while reducing peak memory use.
- Pull-request CI validates built distributions and an isolated installed wheel. Releases
  require checks on the tagged commit and an exact tag/package-version match before publishing.

- **Behavior change: model registry.** `opus`, `sonnet`, `haiku` now resolve to
  `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` (previously Claude 4.x ids).
  Prices (USD per million tokens, input/output): opus 4/20, sonnet 2/10, haiku 0.10/0.50;
  new `fable` 10/50. Cache reads are 0.05x input on opus/sonnet and 0.025x on fable
  (previously 0.1x); cache writes stay 1.25x. Haiku 5.5 switches to 0.50/2.50 for
  prompts over 100k tokens (new `ModelSpec.long_prompt_threshold` and
  `long_input_per_mtok`/`long_output_per_mtok`). `gpt-4o`/`gpt-4o-mini` prices
  re-verified, unchanged. Logged `cost_equiv` values change accordingly.
- **Behavior change: unpriced models log `cost_equiv: null`** instead of `0`. This
  covers unknown raw model ids and local Ollama/vLLM/openai-compat specs.
  `ModelSpec.cost()` still returns a float.
- **Behavior change: sandbox containers run with all capabilities dropped**
  (`--cap-drop ALL`). They still run as the image's own `USER` unless `user=` is passed
  (the bundled sandbox image uses uid 1000). Images whose entrypoint needs
  capabilities (e.g. `chown`, `su`/`gosu`) will fail.
- Sandbox environment variables are passed via a temporary `--env-file` (0600, deleted
  after the run) instead of `-e KEY=VALUE`. Names/values containing newlines, NUL, `=`
  in the name, or a leading `#` are rejected with `ValueError`.
- Sandbox and proxy base images are pinned by digest
  (`python:3.11-slim@sha256:e88e9763...`). The proxy container gets memory, CPU, and PID
  limits, dropped capabilities, and `no-new-privileges`.
- **Behavior change: the proxy container starts with `--pull=never`.** A missing
  `proxy_image` now fails with a hint to run `build_default_images()` instead of being
  pulled from a registry (the default `tbcore/proxy:latest` name is not a published image).
  Requires Docker 20.10+.
- Sandbox containers are now removed with `docker rm -f -v` so anonymous volumes cannot leak.
  The docker create arguments for `run()` are unchanged unless a new option is used.
- The proxy container now runs as the unprivileged `proxy` user and listens on 3128 (and 8080
  when credentials are injected). Rebuild the proxy image with `build_default_images()`.
- `sandbox/` moved to `src/tbcore/images/sandbox` and `src/tbcore/images/proxy`.
  `build_image` is unchanged.
- The pool's retry delay is `max(backoff, retry_after)`, and a rate-limit error now
  pauses all workers until its backoff window ends. Stored results include
  `cache_read_input_tokens` and `cache_creation_input_tokens`.
- `cohens_d` (and so `RegressionResult.effect_size`) returns `nan` instead of
  `inf`/`nan` with a division warning when there are fewer than three observations or
  zero pooled variance.
- Package version is read from `tbcore.__version__`; metadata gains author, keywords,
  and classifiers.
- CI runs Python 3.11-3.13 on Linux plus 3.12 on Windows and macOS, with mypy and
  coverage.

### Fixed

- Editable-install CI and release checks exclude the installed-wheel-only Docker fixture;
  the separate wheel job still runs it. Repository-wide Ruff lint and format gates pass.
- Retrying pool jobs remain exclusively owned during backoff, preventing duplicate API calls
  by other workers while retaining attempt counts across cancellation and resume.
- `CallLog(None)` retains records in memory as documented, with independent instances and
  isolated copies of recorded and returned data.

- `Client(..., base_url=...)` was ignored on the `openai-compat` wire.
- Duplicated HTTP error handling in `HttpTransport` and `OpenAICompatTransport` is now
  one shared helper.
- Cached-input tokens from OpenAI-style `prompt_tokens_details.cached_tokens`
  are now recorded as cache reads in accounting.
- OpenAI-style cached tokens were counted twice in cost (once inside `prompt_tokens`,
  again as cache reads). `Usage.input_tokens` now excludes them on that wire.

### Security

- `Sandbox.run(files=...)` validates every filename before staging writes, rejecting parent
  traversal and absolute or Windows-specific paths that could target host files.

- Sandbox hardening: `--cap-drop ALL`, optional `user=` override, secrets kept off argv, digest-pinned
  base images, proxy resource limits. See `SECURITY.md` for the threat model and how to
  report vulnerabilities.
- Credential injection: API keys can live only in the proxy container; sandboxed code can no
  longer exfiltrate through an injected API with its own key. Data sent to an injected host
  still lands in your account; see `SECURITY.md`.
- Optional `read_only=True` root filesystem and `runtime="runsc"` (gVisor); the proxy no
  longer runs as root. `read_only` and `runtime` are opt-in (off by default).
- The Docker-level behavior of the hardening is now exercised by
  `pytest -m "docker and not installed_wheel and not live"` from an editable install
  (non-root, no capabilities, env passthrough, read-only, sessions, proxy). gVisor itself is
  not exercised in CI: only the `--runtime` flag is unit-tested.

## [0.1.0] - 2026-08-24

### Added

- Request client (`tbcore.client`) with token/latency/cost accounting into an
  append-only JSONL call log, short-name model registry, and Anthropic-style
  prompt-prefix caching via cacheable system blocks.
- Rate-limit-aware resumable execution pool (`tbcore.pool`): SQLite work store,
  bounded async concurrency, exponential backoff on rate-limit/transient
  errors, and restart-safe resume without redoing completed work.
- Container sandbox (`tbcore.sandbox`) for untrusted content: pinned base
  image (`sandbox/Dockerfile`), ephemeral fresh workdir per run, CPU/memory/PID
  limits, no network by default, and allowlist-only egress through a proxy on
  an internal Docker network.
- Stats layer (`tbcore.stats`): distribution summaries, seeded bootstrap
  confidence intervals, one-sided Welch regression test, and Cohen's d.
- CI workflow running ruff lint/format checks and pytest on Python 3.11/3.12.
