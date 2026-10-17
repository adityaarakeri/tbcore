"""Explicitly loaded TOML file that defines named model specs.

Nothing here runs implicitly: a config exists only when the caller passes a path to
:func:`load_model_config` and hands the result to ``Client(..., config=...)``. The file
holds endpoints, the *names* of key variables, and prices. It never holds secret values.
"""

from __future__ import annotations

import logging
import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from tbcore.models import REGISTRY, ModelSpec, make_model_spec

logger = logging.getLogger(__name__)

PROVIDERS = ("anthropic", "ollama", "vllm", "openai-compat")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STRING_FIELDS = ("provider", "model", "base_url", "api_key_env")
_NONNEG_FLOAT_FIELDS = (
    "input_per_mtok",
    "output_per_mtok",
    "cache_write_multiplier",
    "cache_read_multiplier",
    "long_input_per_mtok",
    "long_output_per_mtok",
)
_ALLOWED_FIELDS = frozenset(
    (*_STRING_FIELDS, *_NONNEG_FLOAT_FIELDS, "api_key_required", "long_prompt_threshold")
)


class ConfigError(ValueError):
    """The config file is malformed, or a model name cannot be resolved against it."""


@dataclass(frozen=True)
class ModelConfig:
    """Named model specs loaded from one file. Resolution is strict: no raw-id fallthrough."""

    path: Path
    models: Mapping[str, ModelSpec]

    def source(self, name: str) -> Literal["config", "built-in"]:
        """Which registry answers ``name``: the config always wins over the built-ins."""
        if name in self.models:
            return "config"
        if name in REGISTRY:
            return "built-in"
        raise ConfigError(self._unknown(name))

    def resolve(self, name: str) -> ModelSpec:
        """Return the config entry, else the built-in registry entry, else raise.

        Unlike :func:`tbcore.models.resolve_model`, an unknown name is an error rather than
        a raw Anthropic model id. Pass a ``ModelSpec`` to ``Client`` for an ad-hoc model.
        """
        source = self.source(name)
        logger.debug("model %r resolved from %s", name, source)
        return self.models[name] if source == "config" else REGISTRY[name]

    def _unknown(self, name: str) -> str:
        known = ", ".join(sorted({*self.models, *REGISTRY}))
        return f"unknown model {name!r} for config {self.path} (known: {known})"


def load_model_config(path: str | PathLike[str]) -> ModelConfig:
    """Read and validate a model config file; raises :class:`ConfigError` on any problem.

    A missing file raises ``FileNotFoundError`` from the open call, unchanged.
    """
    file = Path(path)
    try:
        with file.open("rb") as fh:
            data = tomllib.load(fh)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{file}: not valid TOML: {exc}") from exc

    extra = sorted(set(data) - {"models"})
    if extra:
        raise ConfigError(
            f"{file}: unknown top-level key(s): {', '.join(extra)}; expected [models]"
        )
    tables = data.get("models")
    if not isinstance(tables, dict) or not tables:
        raise ConfigError(f"{file}: define at least one [models.<name>] table")

    models = {name: _build_spec(file, name, fields) for name, fields in tables.items()}
    logger.debug("loaded %d model(s) from %s", len(models), file)
    return ModelConfig(path=file, models=models)


def _build_spec(file: Path, name: str, fields: object) -> ModelSpec:
    where = f"{file}: [models.{name}]"
    if not name or name != name.strip():
        raise ConfigError(f"{file}: model names must be non-empty without surrounding spaces")
    if not isinstance(fields, dict):
        raise ConfigError(f"{where} must be a table")
    if "api_key" in fields:
        raise ConfigError(
            f"{where}: api_key is not allowed; keys never go in this file. "
            "Name the environment variable in api_key_env instead"
        )
    unknown = sorted(set(fields) - _ALLOWED_FIELDS)
    if unknown:
        raise ConfigError(
            f"{where}: unknown field(s): {', '.join(unknown)}; allowed: "
            f"{', '.join(sorted(_ALLOWED_FIELDS))}"
        )

    for key in _STRING_FIELDS:
        if key in fields and (not isinstance(fields[key], str) or not fields[key].strip()):
            raise ConfigError(f"{where}: {key} must be a non-empty string")
    for key in ("provider", "model"):
        if key not in fields:
            raise ConfigError(f"{where}: {key} is required")
    provider = fields["provider"]
    if provider not in PROVIDERS:
        raise ConfigError(f"{where}: provider must be one of {', '.join(PROVIDERS)}")
    base_url = fields.get("base_url")
    if base_url is not None:
        _check_base_url(where, base_url)
    api_key_env = fields.get("api_key_env")
    if api_key_env is not None and not _ENV_NAME.fullmatch(api_key_env):
        raise ConfigError(f"{where}: api_key_env must be an environment variable name")

    changes: dict[str, Any] = {"short_name": name}
    for key in _NONNEG_FLOAT_FIELDS:
        if key in fields:
            changes[key] = _nonneg_number(where, key, fields[key])
    if "long_prompt_threshold" in fields:
        value = fields["long_prompt_threshold"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"{where}: long_prompt_threshold must be a non-negative integer")
        changes["long_prompt_threshold"] = value
    if "api_key_required" in fields:
        if not isinstance(fields["api_key_required"], bool):
            raise ConfigError(f"{where}: api_key_required must be true or false")
        changes["api_key_required"] = fields["api_key_required"]

    if provider == "anthropic":
        spec = ModelSpec(
            short_name=name,
            model_id=fields["model"],
            base_url=base_url,
            **({"api_key_env": api_key_env} if api_key_env else {}),
        )
    else:
        try:
            spec = make_model_spec(
                fields["model"], provider=provider, base_url=base_url, api_key_env=api_key_env
            )
        except ValueError as exc:
            raise ConfigError(f"{where}: {exc}") from exc
    return replace(spec, **changes)


def _check_base_url(where: str, value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ConfigError(f"{where}: base_url must be an http(s) URL with a host")
    if parsed.username or parsed.password:
        raise ConfigError(f"{where}: base_url must not embed credentials")


def _nonneg_number(where: str, key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{where}: {key} must be a number")
    if not math.isfinite(value) or value < 0:
        raise ConfigError(f"{where}: {key} must be finite and non-negative")
    return float(value)
