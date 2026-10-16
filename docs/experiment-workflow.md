# Reproducible local agent experiments

This is the first Docker-backed experiment path. It uses a fixture agent and no
paid model calls. From a clean checkout with [uv](https://docs.astral.sh/uv/) and a
working Docker daemon:

```bash
uv sync --extra dev
uv run python examples/experiment_workflow.py --store .tbcore/example
```

`uv sync` downloads Python 3.11+ if none is installed, and the commands are the same
on Windows. The example builds the bundled sandbox and proxy images unless `--image`
supplies a prebuilt sandbox image.
It opts in to workspace and artifact content capture, freezes the source task
and image ID, runs baseline and candidate configurations as separate trials,
grades their final `answer.txt`, writes `comparison.md`, exports
`candidate-bundle.json`, imports that bundle in a fresh store, and executes a
new seed from the imported inputs. The agent runs in Docker with no network.

The report counts one completed attempt per statistical trial. Baseline writes
`before`, candidate writes `after`, and the exact evaluator expects `after`.
The resulting quality rates are 0/1 and 1/1. Their Wilson intervals are wide
because this fixture has one trial per configuration; it is a workflow check,
not evidence of model quality. No model calls occur, so usage/cost coverage is
unknown and cost comparison has a zero denominator. Do not read that as $0
model cost. For real calls, ingest direct or proxy usage and apply a frozen
`PriceCard` before comparing. Unknown usage, unpriced models, failed
execution, missing artifacts, or incompatible evaluator/dataset versions are
shown as unknown, excluded with counts, or rejected as appropriate.

Inspect one unified record by opening `candidate-bundle.json`: it contains
the specification, run status/attempt identity, ordered command and evaluation
events, artifact hashes and content, immutable workspace manifest/blobs, and
environment provenance. Content capture may retain sensitive source material;
use an explicit `CapturePolicy(secret_values=...)`, limit workspace scope, and
protect or delete exported bundles according to local retention policy. A
redacted structural example is in `docs/sample-run-record.json`.

## Support and validation matrix

| Path | Evidence | Status |
| --- | --- | --- |
| Local metadata and artifact store, replay, comparison | Non-Docker tests and disposable-store backup test | Tested on Python 3.12 here |
| Docker sandbox, proxy, tool edit, streamed fake-provider usage, isolated grader | Installed-wheel `tests/test_experiment_workflow_docker.py` and full Docker selection | Passed locally on Docker 29.1.5; hosted CI pending |
| Live Codex CLI with API-key proxy | Opt-in version-pinned image smoke with bounded calls, tokens, and time | Experimental, not part of default CI |
| Claude Code, OpenCode, agy managed agent workflow | No end-to-end experiment fixture | Not claimed |
| Kubernetes or managed platform backend | No implementation | Deferred |

`SandboxTrialExecutor` records a Codex `exec --json` timeline only when given an
exact CLI version. For live trials, record the resolved image ID or registry
digest, exact CLI version, model, provider, resource limits, dataset version,
seed where supported, evaluator version, and price-card version. A fixed seed
does not make model sampling or external tools deterministic. Retry restores
captured files, but external side effects can repeat. Direct and proxy budget
controllers have separate counters; do not claim a shared spend ceiling for
mixed-path attempts. Usage-based spend limits can be exceeded by requests
already in flight.

The optional paid smoke is `tests/test_live_codex_smoke.py`. Set
`TBCORE_RUN_LIVE_CODEX=1`, `OPENAI_API_KEY`, an exact `TBCORE_CODEX_VERSION`, and
`TBCORE_LIVE_REPORT` to a local JSON output path, then run
`uv run pytest -m live tests/test_live_codex_smoke.py -q`. It permits at most
four model calls, has a 2,000-token admission limit, and bounds command and
session time. The report records the installed tbcore version, CLI version,
image ID, model, usage, and configured limits without the API key. These are
best-effort admission limits, not a guaranteed provider billing cap. The test
skips unless explicitly opted in and is excluded from the default Docker CI
selection.

## Backup, recovery, and rollout

Close the store, then copy its full directory (including `experiments.sqlite3`,
workspace blobs/manifests, environments, and artifacts) to a private backup.
For a live process, use SQLite's online backup API for the database and a
consistent snapshot of the content directories; copying only the SQLite file
can produce missing-artifact failures. The disposable-store test copies a
closed store, reopens it, and restores captured inputs. To recover, restore
the complete backup into a new directory, open it with a compatible tbcore
version, verify artifact hashes by reading completed runs, then resume trials.
The runner never re-executes completed trials, marks stale running attempts
interrupted, and assigns new attempt IDs to replacements. Registered orphan
`tbcore` containers are removed before retry; unknown containers are left
alone.

The store schema version is checked on open. Additive tables may be created
for existing version-1 stores; an unknown future version is rejected. There
is no automatic downgrade. Opt in with a disposable copied store first, pin
the package and image IDs, and retain the prior package/images. To roll back,
stop writers and restore the matching backup and package/image versions. A
portable run bundle includes captured input files, but a metadata-only
workspace cannot be reconstructed.

## Performance recipe

On the operator's Docker host, build an exact image and run:

```bash
uv sync --extra dev
uv run python -c "import tbcore; print(tbcore.build_default_images())"
uv run python scripts/benchmark_experiment.py --image tbcore/sandbox:latest --samples 20 --concurrency 2 --output benchmark.json
```

Record the machine, Docker version/runtime, image ID, concurrency, sample
count, workload, and storage device with the JSON. The script reports raw
samples and median/p95 for cold and warm launches, paired recording on/off
durations, host proxy direct/proxied first-byte streaming latency, artifact
bytes, throughput, and force-cleanup time/verification after an interrupted
session. Repeat on the same idle host and image; report failures and all
samples rather than selecting a best run. The recording comparison includes
SQLite event and artifact work and should not be interpreted as a provider
latency estimate. Host CPU, memory, Docker image storage, volume storage, and
operator time are separate from model API cost.

One local run is saved in `docs/performance-results.json` with all raw samples,
median/p95 distributions, image ID, Docker runtime/storage driver, and zero
observed failures. On this Windows host with Docker 29.1.5 and 16 logical CPUs,
the 20-sample median was 0.690 s with recording off and 0.714 s with recording
on; warm startup median was 0.673 s. The first launch was 0.649 s, but one
sample cannot establish a cold-start distribution. The proxy's host-only
first-byte fixture and Docker throughput samples are in the JSON report.
Docker Desktop's underlying host storage device was not identified. This is
one machine and one simple workload, so it supports no broader speed or cost
comparison.
