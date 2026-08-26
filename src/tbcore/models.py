"""Registry mapping short model names to provider model identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
VLLM_DEFAULT_BASE_URL = "http://localhost:8000"


@dataclass(frozen=True)
class ModelSpec:
    short_name: str
    model_id: str
    provider: str = "anthropic"
    api_key_env: str = "ANTHROPIC_API_KEY"
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    cache_write_multiplier: float = 1.25
    cache_read_multiplier: float = 0.1
    # Wire protocol spoken to this provider: "anthropic" (Messages API) or
    # "openai-compat" (/v1/chat/completions, spoken by Ollama, LM Studio,
    # vLLM, OpenAI itself, and most gateways).
    wire: str = "anthropic"
    base_url: str | None = None
    api_key_required: bool = True

    def cost(self, usage: Any) -> float:
        """Cost-equivalent in USD for a usage object with token count fields."""
        billed_input = (
            usage.input_tokens
            + usage.cache_creation_input_tokens * self.cache_write_multiplier
            + usage.cache_read_input_tokens * self.cache_read_multiplier
        )
        return (
            billed_input * self.input_per_mtok + usage.output_tokens * self.output_per_mtok
        ) / 1_000_000


REGISTRY: dict[str, ModelSpec] = {
    "opus": ModelSpec(
        "opus", "claude-opus-4-1-20250805", input_per_mtok=15.0, output_per_mtok=75.0
    ),
    "sonnet": ModelSpec(
        "sonnet", "claude-sonnet-4-5-20250929", input_per_mtok=3.0, output_per_mtok=15.0
    ),
    "haiku": ModelSpec(
        "haiku", "claude-haiku-4-5-20251001", input_per_mtok=1.0, output_per_mtok=5.0
    ),
    "gpt-4o": ModelSpec(
        "gpt-4o",
        "gpt-4o",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        input_per_mtok=2.50,
        output_per_mtok=10.00,
        cache_read_multiplier=0.5,
        wire="openai-compat",
        base_url="https://api.openai.com/v1",
    ),
    "gpt-4o-mini": ModelSpec(
        "gpt-4o-mini",
        "gpt-4o-mini",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        input_per_mtok=0.15,
        output_per_mtok=0.60,
        cache_read_multiplier=0.5,
        wire="openai-compat",
        base_url="https://api.openai.com/v1",
    ),
}


def resolve_model(name_or_id: str) -> ModelSpec:
    """Resolve a short name from the registry, or pass a raw model id through."""
    spec = REGISTRY.get(name_or_id)
    if spec is not None:
        return spec
    return ModelSpec(short_name=name_or_id, model_id=name_or_id)


def make_model_spec(
    name_or_id: str,
    *,
    provider: str = "anthropic",
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> ModelSpec:
    """Build a :class:`ModelSpec` for any supported provider.

    Providers:

    - ``anthropic`` (default) - registry lookup or raw model id on the
      Messages API wire; requires ``ANTHROPIC_API_KEY``.
    - ``ollama`` - OpenAI-compatible chat completions against a local server
      (default ``http://localhost:11434``); no API key required.
    - ``vllm`` - OpenAI-compatible chat completions against a vLLM server
      (default ``http://localhost:8000``); an API key is optional and sent as
      a Bearer header when provided or present in ``VLLM_API_KEY``.
    - ``openai-compat`` - any OpenAI-compatible endpoint; ``base_url``
      required. An API key is optional and sent as a Bearer header when
      provided.
    """
    p = provider.lower()
    if p == "anthropic":
        spec = resolve_model(name_or_id)
        if base_url is not None or api_key_env is not None:
            spec = _replace(spec, base_url=base_url, api_key_env=api_key_env)
        return spec
    if p == "ollama":
        return ModelSpec(
            short_name=name_or_id,
            model_id=name_or_id,
            provider="ollama",
            api_key_env=api_key_env or "OLLAMA_API_KEY",
            wire="openai-compat",
            base_url=base_url or OLLAMA_DEFAULT_BASE_URL,
            api_key_required=False,
        )
    if p == "vllm":
        return ModelSpec(
            short_name=name_or_id,
            model_id=name_or_id,
            provider="vllm",
            api_key_env=api_key_env or "VLLM_API_KEY",
            wire="openai-compat",
            base_url=base_url or VLLM_DEFAULT_BASE_URL,
            api_key_required=False,
        )
    if p in ("openai-compat", "openai"):
        if not base_url:
            raise ValueError(
                f"provider {p!r} requires an explicit base_url (e.g. http://localhost:1234/v1)"
            )
        return ModelSpec(
            short_name=name_or_id,
            model_id=name_or_id,
            provider="openai-compat",
            api_key_env=api_key_env or "OPENAI_COMPAT_API_KEY",
            wire="openai-compat",
            base_url=base_url,
            api_key_required=False,
        )
    raise ValueError(
        f"unsupported provider {provider!r}; expected anthropic, ollama, vllm, or openai-compat"
    )


def _replace(spec: ModelSpec, **changes: Any) -> ModelSpec:
    from dataclasses import replace as _dataclass_replace

    changes = {k: v for k, v in changes.items() if v is not None}
    return _dataclass_replace(spec, **changes) if changes else spec
