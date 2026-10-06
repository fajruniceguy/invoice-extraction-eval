"""Environment-driven settings.

The provider endpoint, model, key and prices all come from the environment (or a local
`.env`). `llm_base_url` and `llm_model` deliberately have no defaults so a missing value
fails loudly instead of silently calling the wrong model.
"""

from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

StructuredMode = Literal["auto", "json_schema", "json_object"]

_TOKENS_PER_MTOK = Decimal(1_000_000)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_base_url: str
    llm_model: str
    llm_api_key: SecretStr = SecretStr("")

    # USD per 1M tokens, as vendors publish them. 0 means "free / unknown" (e.g. local models).
    llm_price_input_per_mtok: Decimal = Field(default=Decimal(0), ge=0)
    llm_price_output_per_mtok: Decimal = Field(default=Decimal(0), ge=0)

    llm_timeout_s: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=4, ge=0)
    llm_backoff_base_s: float = Field(default=1.0, ge=0)
    llm_backoff_max_s: float = Field(default=30.0, ge=0)

    # Re-asks when the model's output is not valid JSON / does not match the schema (0 = never).
    llm_max_repairs: int = Field(default=1, ge=0)

    # auto: try json_schema, fall back to json_object. The others pin a single tier.
    llm_structured_mode: StructuredMode = "auto"

    # Passed through as `reasoning_effort` when set (e.g. none/minimal/low/medium/high; accepted
    # values differ per provider, so it is not validated here). Blank means unset.
    llm_reasoning_effort: str | None = None

    @field_validator("llm_reasoning_effort", mode="before")
    @classmethod
    def _blank_reasoning_effort_is_unset(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("llm_base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith(("http://", "https://")):
            raise ValueError("LLM_BASE_URL must start with http:// or https://")
        return value

    @field_validator("llm_model")
    @classmethod
    def _check_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("LLM_MODEL must not be empty")
        return value

    def cost_usd(self, input_tokens: int, output_tokens: int) -> Decimal:
        """Cost of one call in USD, from the configured per-1M-token prices."""
        return (
            self.llm_price_input_per_mtok * input_tokens
            + self.llm_price_output_per_mtok * output_tokens
        ) / _TOKENS_PER_MTOK


@lru_cache
def get_settings() -> Settings:
    return Settings()
