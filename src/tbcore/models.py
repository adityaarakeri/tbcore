"""Registry mapping short model names to provider model identifiers."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    # Optional long-prompt tier: when the full prompt (input + cache reads + cache
    # writes) exceeds the threshold, the long rates replace the base rates.
    long_prompt_threshold: int | None = None
    long_input_per_mtok: float = 0.0
    long_output_per_mtok: float = 0.0

    @property
    def priced(self) -> bool:
        """True when this spec has registered prices (otherwise cost is unknown, not zero)."""
        return self.input_per_mtok > 0.0 or self.output_per_mtok > 0.0

    def cost(self, usage: Any) -> float:
        """Cost-equivalent in USD for a usage object with token count fields.

        Returns 0.0 for unpriced specs; check :attr:`priced` to tell that apart from free.
        """
        prompt_tokens = (
            usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
        )
        input_rate, output_rate = self.input_per_mtok, self.output_per_mtok
        if self.long_prompt_threshold is not None and prompt_tokens > self.long_prompt_threshold:
            input_rate, output_rate = self.long_input_per_mtok, self.long_output_per_mtok
        billed_input = (
            usage.input_tokens
            + usage.cache_creation_input_tokens * self.cache_write_multiplier
            + usage.cache_read_input_tokens * self.cache_read_multiplier
        )
        return float((billed_input * input_rate + usage.output_tokens * output_rate) / 1_000_000)


# Anthropic prices: https://platform.claude.com/docs/en/about-claude/pricing (2026-10-13).
# Cache writes are 1.25x base input (5m); cache reads are 0.05x on Opus/Sonnet 5.5 and 0.025x
# on Fable 5.1. Haiku 5.5 switches to its long-prompt tier above 100k prompt tokens.
REGISTRY: dict[str, ModelSpec] = {
    "opus": ModelSpec(
        "opus",
        "claude-opus-5-5",
        input_per_mtok=4.0,
        output_per_mtok=20.0,
        cache_read_multiplier=0.05,
    ),
    "sonnet": ModelSpec(
        "sonnet",
        "claude-sonnet-5-5",
        input_per_mtok=2.0,
        output_per_mtok=10.0,
        cache_read_multiplier=0.05,
    ),
    "haiku": ModelSpec(
        "haiku",
        "claude-haiku-5-5",
        input_per_mtok=0.10,
        output_per_mtok=0.50,
        long_prompt_threshold=100_000,
        long_input_per_mtok=0.50,
        long_output_per_mtok=2.50,
    ),
    "fable": ModelSpec(
        "fable",
        "claude-fable-5-1",
        input_per_mtok=10.0,
        output_per_mtok=50.0,
        cache_read_multiplier=0.025,
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
    changes = {k: v for k, v in changes.items() if v is not None}
    return replace(spec, **changes) if changes else spec
