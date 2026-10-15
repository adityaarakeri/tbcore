# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately using GitHub private security advisories:
open the repository's **Security** tab and choose **Report a vulnerability**. Do not
file public issues for security problems. Include a description, reproduction steps,
and the tbcore version. You can expect an acknowledgement within a few days; fixes
for confirmed issues are released as patch versions and noted under "Security" in the
changelog.

## Supported versions

Only the latest released version receives security fixes (the project is alpha).

## Threat model: `tbcore.sandbox`

The sandbox runs untrusted or model-generated commands in a Docker container. It is a
defense-in-depth layer, not a hardened isolation boundary.

### What it does

- Runs each command in a fresh, ephemeral container that is force-removed afterwards;
  the working directory (`/work`) is not shared with the host. Files are copied in with
  `docker cp`.
- Does not mount host paths or the Docker socket.
- Drops all Linux capabilities (`--cap-drop ALL`) and sets `no-new-privileges`. The
  container runs as the image's `USER`: the bundled sandbox image uses uid 1000. For
  custom images that default to root, pass `user="1000:1000"` (or similar) to force a
  non-root user.
- Applies CPU, memory, and PID limits, plus a wall-clock timeout after which the
  container is killed.
- With `allow_hosts=()` the container gets `--network none`. With a non-empty allowlist
  (the default is `api.anthropic.com`), the container sits on a
  Docker `--internal` network and can only reach the allowlisted hostnames through a
  small forward proxy container.
- Passes environment variables through a temporary `--env-file` (mode 0600, deleted
  after the run) so secrets do not appear in the host process list.
- **Credential injection** (`inject_headers`): the API key lives only in the proxy
  container (also delivered by a 0600 env-file, never argv). Sandboxed code reaches an
  injected host only through a plain-HTTP reverse listener on the proxy, which drops the
  caller's `Authorization`/`x-api-key`/`Proxy-*`/hop-by-hop headers, sets the configured
  ones, and forwards over verified TLS. Injected hosts are refused on the CONNECT/forward
  path. The key therefore never appears in the sandbox's environment, filesystem or
  memory, and code cannot use its own key against that host to send data to an attacker's
  account.
- **Sessions** (`Sandbox.session()`): same hardening as `run()` in a long-lived container;
  files go in as a tar stream (owned by the container user), come out through a size-capped
  tar extracted with `tarfile`'s `data` filter, and the container and its volumes are
  force-removed on close.
- **Optional** `read_only=True` (read-only root filesystem, 64 MB `/tmp`, anonymous `/work`
  volume) and `runtime="runsc"` (gVisor) for a user-space kernel. Both are off by default;
  `read_only=True` is recommended.
- **Codex CLI image** runs as uid 1000. The example sets Codex's own sandbox mode to
  `danger-full-access` so it can edit `/work` and reach the reverse proxy. The Docker
  sandbox, network allowlist, and proxy remain the outer isolation controls. The
  OpenAI API key stays in the proxy; the Codex process sees a placeholder key.
- Runs the proxy container as an unprivileged user (Debian's `proxy`, uid 13) with all
  capabilities dropped; its listen ports (3128, 8080) are unprivileged.
- Pins the base images by digest.

### What it does not protect against

- **Allowlisted hosts are fully trusted.** Any code in the sandbox can talk to an
  allowed host with any credentials you pass in via `env`. A leaked API key can be used
  against that host, or exfiltrated to it. Prefer `inject_headers` over `env` for API
  keys; pass the least-privileged, short-lived credentials you can, and keep allowlists
  minimal.
- **Injected hosts still receive whatever the sandbox sends.** Credential injection hides
  the key; it does not make the host untrusted-data-safe. For OpenAI and Anthropic model
  hosts the reverse proxy currently permits only supported POST model endpoints; data
  sent there still reaches your account and can spend quota. Other injected hosts do
  not have this path restriction. Configure finite proxy call/token/spend limits and
  use a scoped, rate-limited key. Usage-based limits are best-effort for in-flight calls.
- **Codex can spend API quota and send workspace data to OpenAI.** The example authorizes
  the Codex process to call `api.openai.com` through the injected reverse proxy. Limit
  the workspace to files appropriate for that model, use a spend-limited API key, and
  set a finite timeout. The proxy does not inspect model prompts; host-configured token
  and spend budgets reject further calls after measured limits, with in-flight exposure.
- **Non-injected allowlisted hosts remain exfiltration paths.** Anything in `allow_hosts`
  that is not in `inject_headers` is reachable by CONNECT with no inspection, so code can
  send data there (and to any account on a shared service). Keep that list empty if you
  can.
- **Kernel escapes need gVisor.** Without `runtime="runsc"` (or a VM), containers share
  the host kernel. gVisor shrinks that attack surface but is not a guarantee.
- **Session exec timeouts are best-effort.** `exec` kills the process group via
  `timeout`; code that detaches into its own session can outlive the deadline until the
  session closes. Files in a session persist across calls, so a session must not be shared
  between mutually untrusting workloads.
- **Allowlist matching is by hostname only.** The proxy checks the requested host name;
  it does not inspect TLS traffic, restrict paths or methods, or defend against an
  allowed host that itself relays data elsewhere (for example a hosting or paste
  service).
- **Kernel and container-runtime escapes are out of scope.** Containers share the host
  kernel. A kernel, runc, or Docker vulnerability can defeat every control here. For
  hostile code, run Docker inside a disposable VM or use a stronger runtime (gVisor,
  Kata, Firecracker).
- **Resource exhaustion is bounded, not prevented.** Limits cap a single container, but
  disk usage inside the container, many concurrent sandboxes, and the Docker daemon
  itself are not limited by tbcore.
- **The container filesystem is writable by default.** Pass `read_only=True` to mount the
  root filesystem read-only; otherwise code can modify anything the sandbox user can
  write inside its own container. `/work` and `/tmp` stay writable either way.
- **Output is untrusted.** stdout, stderr, and any returned files come from the
  sandboxed code; treat them as attacker-controlled data.
- **Docker access is privileged.** Anyone who can run `Sandbox` can run containers as
  the Docker user. tbcore does not sandbox the host process that calls it.
- **The proxy runs on the default bridge network** so it can reach the internet, and it
  resolves allowed hostnames itself; DNS-based tricks against the allowed hosts are not
  mitigated.

### Planned follow-ups

- Optional seccomp/AppArmor profiles.
- Extend path/method allowlists to custom injected hosts.

### Offline fake-provider routing

`upstream_overrides` is a host-controlled test setting that redirects an
injected OpenAI/Anthropic model host to a single-label Docker network alias
over plain HTTP. It rejects IP literals, dotted public hosts, paths, and
credentials in the destination. Use it only with fixture keys and an
operator-controlled provider container on the internal experiment network.
It intentionally skips upstream TLS verification for that route and must not
carry a production API key. The default route still uses verified HTTPS.

### Data and credentials outside the sandbox

`Client` sends prompts to the configured provider and logs token counts, latency and
cost (not prompt contents) to the call log. API keys are read from environment variables
or passed explicitly; they are never written to the log.

### Experiment evidence capture

`CapturePolicy` records only named, host-controlled metadata by default. Commands,
prompts, responses, stdout/stderr, tool results, workspace contents, and unrecognized
fields require `content_enabled=True`. Known credential fields (including headers,
environment, tokens, passwords, and API keys) are always replaced. Supply known secret
values to `secret_values` so they are also removed from opted-in text content before
export. Binary and oversized text output is represented only by omission reason and
byte count. Do not persist a policy object or raw proxy injection configuration as
evidence.

`EvidenceArtifacts` requires content capture to be enabled, accepts bounded UTF-8 text,
redacts it before writing, and rejects binary content and unsafe names. It creates
private directories and files on platforms with POSIX permissions. Cleanup takes the
set of artifacts referenced by retained runs and removes only unreferenced files within
the artifact store. The caller must decide run retention and pass its complete reference
set; automatic age-based deletion is not implied.

Redaction is limited to configured secret values and known field names. It cannot
identify every credential in arbitrary text or binary data. Only enable content capture
for approved workspaces, and review exported evidence before sharing it. The artifact
directory is intended for a trusted local operator; it does not defend against a
concurrent hostile process with write access to that directory.
