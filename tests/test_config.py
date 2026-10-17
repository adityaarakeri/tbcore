import json
import logging
from pathlib import Path

import pytest

from tbcore import Client, ConfigError, ModelConfig, ModelSpec, load_model_config
from tbcore.accounting import CallLog
from tbcore.models import REGISTRY

VALID = """
[models.local-qwen]
provider = "ollama"
model = "qwen3:8b"

[models.lmstudio]
provider = "openai-compat"
model = "my-model"
base_url = "http://localhost:1234/v1"
api_key_env = "LMSTUDIO_KEY"

[models.house-claude]
provider = "anthropic"
model = "claude-sonnet-5-5"
base_url = "https://gateway.example.com"
input_per_mtok = 1.5
output_per_mtok = 7.5
long_prompt_threshold = 100000
"""


def write(tmp_path, text):
    path = tmp_path / "tbcore.toml"
    path.write_text(text, encoding="utf-8")
    return path


def bad(tmp_path, text, match):
    with pytest.raises(ConfigError, match=match):
        load_model_config(write(tmp_path, text))


def test_loads_specs_with_provider_defaults(tmp_path):
    cfg = load_model_config(write(tmp_path, VALID))

    qwen = cfg.models["local-qwen"]
    assert (qwen.short_name, qwen.model_id, qwen.provider) == ("local-qwen", "qwen3:8b", "ollama")
    assert qwen.base_url == "http://localhost:11434"
    assert qwen.wire == "openai-compat" and not qwen.api_key_required
    assert not qwen.priced

    lm = cfg.models["lmstudio"]
    assert lm.base_url == "http://localhost:1234/v1" and lm.api_key_env == "LMSTUDIO_KEY"

    claude = cfg.models["house-claude"]
    assert claude.wire == "anthropic" and claude.api_key_env == "ANTHROPIC_API_KEY"
    assert claude.api_key_required
    assert (claude.input_per_mtok, claude.output_per_mtok) == (1.5, 7.5)
    assert claude.long_prompt_threshold == 100000


def test_missing_file_is_not_swallowed(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_config(tmp_path / "nope.toml")


def test_malformed_toml(tmp_path):
    bad(tmp_path, "[models.x\nprovider =", "not valid TOML")


def test_no_models(tmp_path):
    bad(tmp_path, "", r"at least one \[models")
    bad(tmp_path, "[models]\n", r"at least one \[models")


def test_unknown_top_level_key(tmp_path):
    bad(tmp_path, VALID + "\n[logging]\nlevel = 'x'\n", "unknown top-level key")


def test_unknown_field_is_rejected(tmp_path):
    bad(tmp_path, '[models.x]\nprovider = "ollama"\nmodel = "m"\nbaseurl = "u"\n', "baseurl")


def test_literal_api_key_is_rejected(tmp_path):
    text = '[models.x]\nprovider = "anthropic"\nmodel = "m"\napi_key = "sk-secret"\n'
    with pytest.raises(ConfigError, match="api_key_env") as info:
        load_model_config(write(tmp_path, text))
    assert "sk-secret" not in str(info.value)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ('model = "m"', "provider is required"),
        ('provider = "ollama"', "model is required"),
        ('provider = "azure"\nmodel = "m"', "provider must be one of"),
        ('provider = "ollama"\nmodel = ""', "model must be a non-empty string"),
        ('provider = "ollama"\nmodel = 3', "model must be a non-empty string"),
        ('provider = "openai-compat"\nmodel = "m"', "requires an explicit base_url"),
        ('provider = "ollama"\nmodel = "m"\nbase_url = "ftp://h"', r"http\(s\) URL"),
        ('provider = "ollama"\nmodel = "m"\nbase_url = "localhost:11434"', r"http\(s\) URL"),
        ('provider = "ollama"\nmodel = "m"\nbase_url = "http://u:p@h"', "credentials"),
        ('provider = "ollama"\nmodel = "m"\napi_key_env = "has space"', "variable name"),
        ('provider = "ollama"\nmodel = "m"\ninput_per_mtok = -1', "non-negative"),
        ('provider = "ollama"\nmodel = "m"\ninput_per_mtok = inf', "non-negative"),
        ('provider = "ollama"\nmodel = "m"\ninput_per_mtok = "2"', "must be a number"),
        ('provider = "ollama"\nmodel = "m"\ninput_per_mtok = true', "must be a number"),
        ('provider = "ollama"\nmodel = "m"\nlong_prompt_threshold = 1.5', "integer"),
        ('provider = "ollama"\nmodel = "m"\napi_key_required = "yes"', "true or false"),
    ],
)
def test_field_validation(tmp_path, fields, match):
    bad(tmp_path, f"[models.x]\n{fields}\n", match)


def test_error_names_the_file_and_model(tmp_path):
    path = write(tmp_path, '[models.broken]\nprovider = "ollama"\n')
    with pytest.raises(ConfigError) as info:
        load_model_config(path)
    assert str(path) in str(info.value) and "[models.broken]" in str(info.value)


def test_resolve_precedence_config_then_builtin_then_error(tmp_path):
    text = '[models.sonnet]\nprovider = "anthropic"\nmodel = "mine"\noutput_per_mtok = 1\n'
    cfg = load_model_config(write(tmp_path, text))

    assert cfg.source("sonnet") == "config"
    assert cfg.resolve("sonnet").model_id == "mine"
    assert cfg.source("opus") == "built-in"
    assert cfg.resolve("opus") is REGISTRY["opus"]
    with pytest.raises(ConfigError, match="unknown model 'claude-opus-5-5'"):
        cfg.resolve("claude-opus-5-5")
    with pytest.raises(ConfigError, match="known: "):
        cfg.source("nope")


def test_config_entry_does_not_inherit_builtin_prices(tmp_path):
    text = '[models.haiku]\nprovider = "anthropic"\nmodel = "x"\n'
    spec = load_model_config(write(tmp_path, text)).resolve("haiku")

    assert not spec.priced and spec.long_prompt_threshold is None
    assert REGISTRY["haiku"].priced


class OkTransport:
    def __init__(self):
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        return {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def test_client_resolves_through_config(tmp_path):
    cfg = load_model_config(write(tmp_path, VALID))
    transport = OkTransport()

    client = Client("local-qwen", config=cfg, transport=transport, log=CallLog(None))

    assert client.spec is cfg.models["local-qwen"]
    assert client.complete(None, [{"role": "user", "content": "x"}]).text == "hi"
    assert json.loads(json.dumps(transport.requests[0].payload))["model"] == "qwen3:8b"


def test_client_with_config_rejects_unknown_name(tmp_path):
    cfg = load_model_config(write(tmp_path, VALID))

    with pytest.raises(ConfigError, match="unknown model"):
        Client("claude-sonnet-5-5", "key", config=cfg)


def test_client_without_config_keeps_raw_id_passthrough():
    client = Client("some-raw-id", "key")

    assert client.spec.model_id == "some-raw-id" and client.spec.provider == "anthropic"


def test_explicit_spec_beats_config(tmp_path):
    cfg = load_model_config(write(tmp_path, VALID))
    spec = ModelSpec("local-qwen", "other", api_key_required=False)

    assert Client(spec, config=cfg).spec is spec


def test_explicit_base_url_beats_config(tmp_path):
    cfg = load_model_config(write(tmp_path, VALID))

    client = Client("local-qwen", config=cfg, base_url="http://other:9999")

    assert client.transport.base_url == "http://other:9999"


def test_config_base_url_reaches_transport(tmp_path):
    cfg = load_model_config(write(tmp_path, VALID))

    assert Client("lmstudio", config=cfg).transport.base_url == "http://localhost:1234/v1"


def test_loading_does_not_read_the_environment_for_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("LMSTUDIO_KEY", "should-not-be-stored")
    cfg = load_model_config(write(tmp_path, VALID))

    assert "should-not-be-stored" not in repr(cfg)


def test_resolution_logs_the_winning_source(tmp_path, caplog):
    cfg = load_model_config(write(tmp_path, VALID))

    with caplog.at_level(logging.DEBUG, logger="tbcore.config"):
        cfg.resolve("local-qwen")
        cfg.resolve("opus")

    assert [r.getMessage() for r in caplog.records] == [
        "model 'local-qwen' resolved from config",
        "model 'opus' resolved from built-in",
    ]


def test_config_is_a_plain_value(tmp_path):
    cfg = load_model_config(write(tmp_path, VALID))

    assert isinstance(cfg, ModelConfig) and cfg.path == tmp_path / "tbcore.toml"


def test_model_name_must_not_have_surrounding_spaces(tmp_path):
    bad(tmp_path, '[models." x"]\nprovider = "ollama"\nmodel = "m"\n', "non-empty without")


def test_model_entry_must_be_a_table(tmp_path):
    bad(tmp_path, "[models]\nx = 1\n", r"\[models.x\] must be a table")


def test_shipped_example_config_loads():
    root = Path(__file__).resolve().parent.parent
    cfg = load_model_config(root / "tbcore.example.toml")

    assert set(cfg.models) == {"local-qwen", "served", "lmstudio", "house-claude"}
