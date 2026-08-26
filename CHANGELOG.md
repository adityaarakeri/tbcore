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
- Cached-input tokens from OpenAI-style `prompt_tokens_details.cached_tokens`
  are now recorded as cache reads in accounting.

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
