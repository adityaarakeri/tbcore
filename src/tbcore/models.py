"""Registry mapping short model names to provider model identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
}


def resolve_model(name_or_id: str) -> ModelSpec:
    """Resolve a short name from the registry, or pass a raw model id through."""
    spec = REGISTRY.get(name_or_id)
    if spec is not None:
        return spec
    return ModelSpec(short_name=name_or_id, model_id=name_or_id)
