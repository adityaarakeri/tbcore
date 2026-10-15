"""Offline installed-wheel experiment using real Docker isolation and proxy streaming."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

import tbcore
from tbcore import (
    CapturePolicy,
    CostLedger,
    EnvironmentProvenance,
    EvaluationSpec,
    Evaluator,
    ExperimentStore,
    PriceCard,
    Sandbox,
    TokenRates,
    TrialCase,
    TrialRunner,
)
from tbcore.timeline import ExecutionTimeline

pytestmark = pytest.mark.docker

SERVER = r"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == "/v1/responses"
        assert self.headers["Authorization"] == "Bearer fixture-key"
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        assert request["stream"] is True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"type":"response.output_text.delta","delta":"edited"}\n\n')
        self.wfile.flush()
        completed = {
            "type": "response.completed",
            "response": {
                "id": "fixture-response", "model": "fixture-model",
                "usage": {
                    "input_tokens": 12, "output_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 2},
                },
            },
        }
        self.wfile.write(("data: " + json.dumps(completed) + "\n\n").encode())
        self.wfile.flush()
    def log_message(self, *args):
        pass
server = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)
print("READY", flush=True)
server.serve_forever()
"""

AGENT = r"""
import json, os, urllib.request
from pathlib import Path
assert os.environ["OPENAI_API_KEY"] == "sandbox-placeholder"
assert Path("task.txt").read_text() == "before"
body = json.dumps({"model":"fixture-model","input":"edit","stream":True}).encode()
request = urllib.request.Request(
    os.environ["OPENAI_BASE_URL"] + "/responses", data=body,
    headers={"Content-Type":"application/json"}, method="POST",
)
with urllib.request.urlopen(request, timeout=15) as response:
    streamed = response.read().decode()
assert "response.completed" in streamed
Path("answer.txt").write_text("edited")
"""


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=False, timeout=45
    )


def _executor(store: ExperimentStore, image: str, proxy_image: str):
    def execute(case, workspace, attempt_id, context):
        with Sandbox(
            image,
            proxy_image=proxy_image,
            allow_hosts=(),
            timeout_s=30,
            inject_headers={"api.openai.com": {"Authorization": "Bearer fixture-key"}},
            upstream_overrides={"api.openai.com": "http://fake-provider:8000"},
            telemetry_run_id=attempt_id,
            telemetry_attempt_id=attempt_id,
            proxy_budget={"max_model_calls": 1, "max_total_tokens": 100},
        ) as box:
            box.prepare()
            assert box.network_name is not None
            name = f"tbcore-fixture-{uuid.uuid4().hex[:12]}"
            started = _docker(
                "run",
                "-d",
                "--name",
                name,
                "--network",
                box.network_name,
                "--network-alias",
                "fake-provider",
                image,
                "python",
                "-u",
                "-c",
                SERVER,
            )
            assert started.returncode == 0, started.stderr
            try:
                for _ in range(50):
                    if "READY" in _docker("logs", name).stdout:
                        break
                    time.sleep(0.1)
                else:
                    pytest.fail("fake provider did not start")
                session = box.session()
                context.register_container(session.container)
                try:
                    files = {path.name: path.read_bytes() for path in workspace.iterdir()}
                    session.put_files(files)
                    result = session.exec(("python", "agent.py"))
                    ExecutionTimeline(store, attempt_id).record_command("agent-command", result)
                    assert result.exit_code == 0, result.stderr
                    shutil.rmtree(workspace)
                    session.copy_out(".", workspace)
                finally:
                    session.close()
                    context.forget_container(session.container)
                observations = box.proxy_usage()
                assert len(observations) == 1
                store.ingest_proxy_usage(observations[0])
            finally:
                removed = _docker("rm", "-f", "-v", name)
                assert removed.returncode == 0, removed.stderr

    return execute


@pytest.mark.installed_wheel
def test_installed_wheel_full_experiment(sandbox_image, proxy_image, tmp_path: Path) -> None:
    package_dir = Path(tbcore.__file__).resolve().parent
    assert package_dir.is_relative_to(Path(sys.prefix).resolve()), package_dir
    source = tmp_path / "source"
    source.mkdir()
    (source / "task.txt").write_text("before", encoding="utf-8")
    (source / "agent.py").write_text(AGENT, encoding="utf-8")
    store = ExperimentStore(tmp_path / "store", CapturePolicy(content_enabled=True))
    snapshot = store.capture_workspace(source)
    image_id = _docker("image", "inspect", "--format", "{{.Id}}", sandbox_image).stdout.strip()
    environment = store.record_environment(
        EnvironmentProvenance(
            image_id=image_id,
            registry_digest=None,
            tbcore_version=tbcore.__version__,
            agent_cli_version="fixture-agent-1",
            model="fixture-model",
            provider="openai",
            resource_limits={"memory": "512m", "cpus": "1.0"},
            runtime="runc",
            policy_config={"network": "internal", "proxy": "offline"},
        )
    )
    case = TrialCase(
        "edit-task",
        "candidate",
        snapshot.snapshot_id,
        {},
        seed=1,
        environment_ref=environment,
        dataset_version="fixture-v1",
    )
    run = TrialRunner(
        store,
        "offline-fixture",
        _executor(store, sandbox_image, proxy_image),
        output_paths=("answer.txt",),
    ).run([case])[0]
    assert run.status == "completed"
    assert store._verified_artifact(run.run_id, "answer.txt")[0] == b"edited"
    calls = [
        event
        for event in store.events(run.run_id)
        if event["fields"].get("event_type") == "model_call"
    ]
    assert len(calls) == 1
    assert calls[0]["fields"]["attempt_id"] == run.attempt_id
    assert calls[0]["fields"]["input_tokens"] == 10
    assert calls[0]["fields"]["usage_complete"] is True
    card = PriceCard(
        "fixture-v1",
        "offline fixture",
        "2026-10-14",
        "USD",
        {"fixture-model": TokenRates("1", "2", "0.5", "1")},
    )
    ledger = CostLedger(store)
    ledger.price_run(run.run_id, card)
    assert ledger.run_totals(run.run_id).total is not None
    verdict = Evaluator(store).evaluate(
        run.run_id,
        EvaluationSpec(
            "exact",
            "v1",
            expected_text={"answer.txt": "edited"},
            command=("python", "-c", "assert open('answer.txt').read() == 'edited'"),
            image=image_id,
        ),
    )
    assert verdict.verdict == "pass"
    bundle = store.export_run(run.run_id)
    store.close()
    reopened = ExperimentStore(tmp_path / "reopened", CapturePolicy(content_enabled=True))
    reopened.import_bundle(json.loads(json.dumps(bundle)))
    assert reopened.get_run(run.run_id).status == "completed"
    replay_case = replace(case, seed=2)
    replay = TrialRunner(
        reopened,
        "offline-fixture",
        _executor(reopened, sandbox_image, proxy_image),
        output_paths=("answer.txt",),
    ).run([replay_case])[0]
    assert replay.status == "completed"
    assert replay.attempt_id != run.attempt_id
    reopened.close()
