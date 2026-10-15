# Local experiment store

`ExperimentStore(root, policy)` keeps specifications, run records, and ordered events in
`experiments.sqlite3` under `root`. Artifact files live under `root/artifacts`. Pass an
explicit `CapturePolicy(content_enabled=True)` only when approved text artifacts or tool
content should be kept. Event fields pass through that policy before storage. A missing
or altered referenced artifact raises an error; it is never treated as a complete run.

## Backup and restore

Stop all writers and close store handles before copying the entire root directory,
including the SQLite database, WAL files if present, and artifacts. Keep the backup
private because opted-in content can contain information that configured redaction did
not recognize. Restore a copy to a new root and open it with a compatible tbcore
version; opening checks every recorded artifact hash. Do not copy only the database or
copy a live database file without its WAL state.

## Migration and recovery

The database uses SQLite `user_version=1`. Opening a version-0 database creates missing
version-1 tables and marks it version 1. Opening a newer or unsupported version raises
before changing that database; make a backup before upgrading the package or store.
There is no automatic downgrade. Restore a compatible backup if a newer schema cannot
be read.

An interrupted artifact write can leave a file before its metadata transaction commits.
Opening the store checks recorded hashes and removes only unreferenced artifact files.
It raises on missing or corrupt recorded files so callers can restore from backup or
investigate. Keep only one active writer process per store root; concurrent threads on
one store instance are serialized for ordered events. Run retention decisions are made
by the caller, which must preserve references to every artifact needed by retained runs.

`export_run` includes referenced workspace manifests and content-addressed blobs,
environment provenance, events, and artifacts. `import_bundle` verifies hashes and
restores those inputs before inserting the run, so a content-enabled captured workspace
can be replayed in a new store. A metadata-only workspace remains non-reconstructable.
The bundle is sensitive when capture was enabled; protect it like a full store backup.

## Input snapshots and environment provenance

`WorkspaceStore` captures files from an approved source tree without relying on Git's
tracked-file list, so dirty and untracked files are included. Default exclusions cover
`.git`, `.venv`, and `.agent`; callers can add patterns. Symlinks are rejected rather
than followed. Content capture must be enabled to reconstruct a workspace; otherwise
the manifest records counts and a fingerprint but explicitly marks reconstruction
incomplete. Binary capture needs a second opt-in (`allow_binary=True`) because binary
bytes cannot be redacted. Configured secret values found in captured bytes cause a
failure; this does not detect unknown secrets. Per-file and total size limits apply.

The returned snapshot ID and environment reference can be stored in `ExperimentSpec`
and `RunRecord`. Environment provenance records the resolved image ID, optional registry
digest, tbcore and agent CLI versions, model/provider, resource limits, runtime, and
non-secret policy settings. Verify the image identity against locally available images
before a rerun. The optional Codex image requires an exact CLI version at build time;
`build_codex_image(version)` supplies it. Capture final workspace artifacts before
closing a sandbox session when a before/after comparison is needed.

## Execution timeline

`ExecutionTimeline(store, run_id).record_command(event_id, result)` records a
`Sandbox.run` or `SandboxSession.exec` result as an outer command. Supply a
`parent_event_id` when the command belongs under an existing event. The event
includes the duration, exit code, timeout and cancellation flags, a command
digest, and an optional bounded output artifact reference. Raw command and
output content follow the store's capture policy; metadata-only capture omits
them. The caller must explicitly pass `cancelled=True` when cancelling work.

For a Codex CLI run, invoke `codex exec --json` and pass its `RunResult` to
`record_codex_exec(event_id, result, cli_version=...)`. The adapter reads only
Codex's JSONL stdout. Each completed `command_execution` item becomes a child
event of the outer CLI command, and a coverage event records the schema,
version, tool-call count, completeness flag, and gaps. Malformed, truncated,
unknown, conflicting, or unfinished events make coverage incomplete. The
adapter does not infer commands from assistant prose or from API response
streaming. Missing tool duration or output remains `null` when Codex does not
provide it; consumers must not treat those values as zero. `coverage_complete`
means the supported tool-event stream was complete, not that every optional
field was available.

## Model usage, estimates, and limits

Pass `CallContext(run_id, attempt_id, call_id)` to an experiment-backed
`Client.complete` call. The context is per invocation, so concurrent callers
cannot overwrite each other's identity. Direct-call events retain provider
request IDs and measured token categories; missing usage remains unknown.

For sandboxed agents, create a `Sandbox` with `telemetry_run_id` and
`telemetry_attempt_id`. Only the host passes those IDs to the proxy container.
Call `box.proxy_usage()` before `box.close()` and ingest each observation with
`ExperimentStore.ingest_proxy_usage`. The proxy supports `POST /v1/responses`
for OpenAI and `POST /v1/messages` for Anthropic on the injected reverse
listener. It labels unsupported, failed, truncated, and usage-less traffic as
unmetered or incomplete. It does not infer usage from response text. Multiple
observations with one provider request ID are reconciled without double
counting; conflicting measured values raise an error.

`CostLedger.price_run(run_id, card)` stores an original per-call estimate
using a caller-supplied `PriceCard` with source, version, effective date,
currency, and independent rates for uncached input, output, cache reads, and
cache writes. `run_totals` reports a known subtotal and unknown-call count;
its total is `None` until every call has measured usage and a rate. Pass
`recalculate=True` to retain a separately labeled estimate under a newer
price card. These are estimates, not a provider invoice.

`RunBudget` admits host-side model and tool calls under wall-clock, count,
token, and cost limits. A direct `Client` can receive the same budget and
checks admission before sending the request. Configure `on_cancel=box.close`
for an owned sandbox. For agent calls, a host-owned `proxy_budget` mapping on
`Sandbox` configures the reverse proxy's model-call, token, and spend limits;
provide model-specific rate strings under `rates`. Unpriced requests are
denied when a spend limit is set. Missing usage makes further token- or
cost-limited calls fail closed. Limits based on usage arriving after a call
are best-effort: concurrent in-flight calls can exceed them, and the
snapshot/telemetry reports that exposure. The host and proxy controllers
have separate counters; a future orchestrator must coordinate them if one
experiment mixes direct and proxy calls under a shared ceiling.

## Trials, grading, and comparisons

Capture input content with `CapturePolicy(content_enabled=True)`, then build a
`TrialCase` for each task, configuration, seed, and dataset version. A
`TrialRunner` restores the snapshot into a fresh temporary workspace for each
attempt. `max_workers` bounds parallelism; `max_attempts` bounds retries of
the same statistical trial. The runner skips completed trials on resume and
keeps failed and interrupted attempts under their original trial ID. Its
executor receives `(case, workspace, attempt_id, context)`. The context
exposes a cancellation event and requires any long-lived Docker workload to
be registered immediately, then forgotten only after removal. Recovery
reclaims dead-process owners and removes registered `tbcore` containers before
starting a new attempt. A live owner's claim is never stolen. Re-execution
starts from frozen inputs but can repeat external model or tool side effects.

`SandboxTrialExecutor(store, image, command)` provides a Docker adapter. It
stages the restored workspace, records the command timeline, copies final
files back before closing the session, and ingests host-bound proxy usage when
`inject_headers` is supplied. Pass expected output names to `TrialRunner`'s
`output_paths`; those files become integrity-checked run artifacts. The agent
command must be present in the image. Cancellation is cooperative for custom
executors and is checked before work and during evidence capture; a blocking
Docker command is bounded by the sandbox timeout. Use separate budgets for
direct and proxied calls; the runner does not promise a shared spend ceiling
when one attempt mixes them.

`Evaluator(store).evaluate(run_id, EvaluationSpec(...))` grades stored
artifacts without rerunning the agent. Exact text checks execute on the host
against verified artifacts. Optional command checks run in a separate
read-only Docker sandbox with no allowed hosts or injected credentials; put
grader code in the evaluator image, outside agent-controlled `/work`.
Evaluator identity, version, configuration hash, input hashes, component
results, duration, verdict, and evidence are retained as events. A timeout or
sandbox error produces an evaluation error while preserving the agent run's
status. Re-evaluation appends another evaluation event.

`compare_experiments(store, group, baseline, candidate, dataset_version,
evaluator_id, evaluator_version)` returns a report with Python fields plus
`to_json()` and `to_markdown()`. It uses one selected attempt per trial, pairs
tasks by task ID and seed, rejects incompatible dataset/evaluator versions,
and reports missing pairs. Quality, latency, token, and cost denominators are
separate. Failed execution and missing grading are excluded explicitly from
quality; incomplete usage and unknown cost are never treated as zero. The
success interval is a Wilson 95% interval and the paired success difference
is an effect estimate, not a significance test.

