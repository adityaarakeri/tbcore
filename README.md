# tbcore

Token-build core: reusable plumbing for token-accounted model workloads.

Four components:

- **Client** (`tbcore.client`) - request wrapper with per-call accounting
  (tokens, latency, cost-equivalent) and prompt-prefix caching.
- **Pool** (`tbcore.pool`) - resumable, rate-limit-aware async execution pool
  backed by a SQLite work store.
- **Sandbox** (`tbcore.sandbox`) - container sandbox for untrusted content:
  no host access, fresh filesystem per run, network restricted to an
  allowlist (defaults to the API host only).
- **Stats** (`tbcore.stats`) - summaries, bootstrap confidence intervals,
  and two-sample regression tests.

## Install

```bash
pip install tbcore            # published
pip install -e ".[dev]"       # local development
```

Requires Python 3.11+.

## Quickstart

```python
from tbcore import Client

client = Client("sonnet")  # API key read from ANTHROPIC_API_KEY
completion = client.complete(
    system="Answer tersely.",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
    cache_prefix="<long stable prefix>",
)
print(completion.text, completion.usage, completion.latency_s)
```

Every call is appended as one JSON line to `.tbcore/calls.jsonl`
(configurable via `CallLog`).

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
- Cost-equivalent uses registered prices; unpriced local models log `$0`.

## Development

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"   # POSIX: .venv/bin/python
.venv\Scripts\ruff format .                       # POSIX: .venv/bin/ruff
.venv\Scripts\ruff check .
.venv\Scripts\python -m pytest
```

Lint and format checks run in CI (GitHub Actions) on Python 3.11 and 3.12.
Sandbox integration tests skip automatically when no container runtime is
available.

## License

MIT - see [LICENSE](LICENSE).
