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
