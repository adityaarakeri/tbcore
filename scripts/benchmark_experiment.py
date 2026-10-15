"""Operator-run, offline benchmark for experiment recording and Docker overhead."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tbcore import (
    CapturePolicy,
    ExecutionTimeline,
    ExperimentSpec,
    ExperimentStore,
    RunRecord,
    Sandbox,
)
from tbcore.images.proxy.proxy import ProxyConfig, start_servers


def summarize(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    if not ordered:
        return {"count": 0, "samples": [], "median": None, "p95": None}
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "count": len(ordered),
        "samples": samples,
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=False, timeout=60
    )


def _command(image: str) -> float:
    started = time.perf_counter()
    with Sandbox(image, allow_hosts=(), timeout_s=30) as box:
        result = box.run(("python", "-c", "print('fixture')"))
    if result.exit_code != 0:
        raise RuntimeError(f"fixture command failed: {result.stderr}")
    return time.perf_counter() - started


def _recorded_command(store: ExperimentStore, image: str) -> float:
    started = time.perf_counter()
    run_id = uuid.uuid4().hex
    pending = RunRecord("benchmark", run_id, run_id, run_id, "pending", _now())
    store.save_run(pending)
    store.save_run(pending.transition("running", at=_now()))
    with Sandbox(image, allow_hosts=(), timeout_s=30) as box:
        result = box.run(("python", "-c", "print('fixture')"))
    ExecutionTimeline(store, run_id).record_command("fixture-command", result)
    store.add_artifact(run_id, "output.txt", result.stdout.encode("utf-8"))
    store.save_run(
        replace(
            store.get_run(run_id).transition("completed", at=_now()), artifact_ids=("output.txt",)
        )
    )
    if result.exit_code != 0:
        raise RuntimeError(f"recorded fixture command failed: {result.stderr}")
    return time.perf_counter() - started


def _stream_latency(samples: int) -> dict[str, object]:
    first = b'data: {"type":"response.output_text.delta","delta":"x"}\n\n'
    last = b'data: {"type":"response.completed","response":{"id":"fixture"}}\n\n'

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(first) + len(last)))
            self.end_headers()
            self.wfile.write(first)
            self.wfile.flush()
            time.sleep(0.02)
            self.wfile.write(last)
            self.wfile.flush()

        def log_message(self, *_args: object) -> None:
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    import threading

    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    config = ProxyConfig(
        inject={"api.openai.com": {"Authorization": "Bearer fixture"}},
        connect=lambda _host: http.client.HTTPConnection(
            "127.0.0.1", upstream.server_address[1], timeout=5
        ),
    )
    forward, reverse = start_servers(config, host="127.0.0.1", forward_port=0, reverse_port=0)
    if reverse is None:
        raise RuntimeError("reverse proxy did not start")
    direct: list[float] = []
    proxied: list[float] = []
    body = b'{"model":"fixture","stream":true}'
    try:
        for _ in range(samples):
            for port, path, destination in (
                (upstream.server_address[1], "/v1/responses", direct),
                (reverse.server_address[1], "/api.openai.com/v1/responses", proxied),
            ):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                started = time.perf_counter()
                connection.request(
                    "POST", path, body=body, headers={"Content-Type": "application/json"}
                )
                response = connection.getresponse()
                if response.status != 200 or not response.read1(1):
                    raise RuntimeError("stream fixture failed")
                destination.append(time.perf_counter() - started)
                response.read()
                connection.close()
    finally:
        for server in (forward, reverse, upstream):
            server.shutdown()
            server.server_close()
    return {"direct_first_byte_s": summarize(direct), "proxy_first_byte_s": summarize(proxied)}


def benchmark(image: str, samples: int, concurrency: int) -> dict[str, object]:
    if samples < 2 or concurrency < 1:
        raise ValueError("samples must be >= 2 and concurrency positive")
    image_id = _docker("image", "inspect", "--format", "{{.Id}}", image)
    if image_id.returncode != 0:
        raise RuntimeError(f"cannot inspect image: {image_id.stderr}")
    report: dict[str, object] = {
        "captured_at_utc": _now(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "image": image,
        "image_id": image_id.stdout.strip(),
        "docker_server_version": _docker(
            "version", "--format", "{{.Server.Version}}"
        ).stdout.strip(),
        "docker_storage_driver": _docker("info", "--format", "{{.Driver}}").stdout.strip(),
        "docker_root_dir": _docker("info", "--format", "{{.DockerRootDir}}").stdout.strip(),
        "host_storage_device": "not identified by this recipe",
        "runtime": _docker("info", "--format", "{{.DefaultRuntime}}").stdout.strip(),
        "samples_requested": samples,
        "concurrency": concurrency,
        "workload": "python -c print('fixture') in fresh Docker containers",
        "model_api_cost_usd": None,
        "failures": [],
    }
    report["cold_start_s"] = summarize([_command(image)])
    report["warm_start_s"] = summarize([_command(image) for _ in range(samples)])
    with tempfile.TemporaryDirectory(prefix="tbcore-benchmark-") as directory:
        store = ExperimentStore(Path(directory) / "store", CapturePolicy(content_enabled=True))
        store.add_spec(ExperimentSpec("benchmark", "fixture", {}))
        off: list[float] = []
        on: list[float] = []
        for index in range(samples):
            # Alternate order to limit drift from image and filesystem cache effects.
            if index % 2:
                on.append(_recorded_command(store, image))
                off.append(_command(image))
            else:
                off.append(_command(image))
                on.append(_recorded_command(store, image))
        report["recording_disabled_s"] = summarize(off)
        report["recording_enabled_s"] = summarize(on)
        artifacts = list((store.artifacts.root).rglob("*.txt"))
        report["artifact_storage_bytes"] = sum(path.stat().st_size for path in artifacts)
        report["artifact_count"] = len(artifacts)
        store.close()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        concurrent = list(pool.map(lambda _: _command(image), range(samples)))
    elapsed = time.perf_counter() - started
    report["throughput_jobs_per_s"] = samples / elapsed
    report["throughput_job_durations_s"] = summarize(concurrent)
    report["stream_latency_host_proxy"] = _stream_latency(samples)
    box = Sandbox(image, allow_hosts=(), timeout_s=30)
    session = box.session()
    container = session.container
    started = time.perf_counter()
    box.close()
    report["interrupted_cleanup_s"] = time.perf_counter() - started
    report["cleanup_container_removed"] = _docker("inspect", container).returncode != 0
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="prebuilt exact image tag or digest")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.image, args.samples, args.concurrency)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
