"""Environment-driven settings.

The provider endpoint, model, key and prices all come from the environment (or a local
`.env`). `llm_base_url` and `llm_model` deliberately have no defaults so a missing value
fails loudly instead of silently calling the wrong model.
"""

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

StructuredMode = Literal["auto", "json_schema", "json_object"]
# on: read and write the response cache; off: bypass it entirely;
# refresh: skip reads but overwrite entries (re-ask the model, keep the cache current)
CacheMode = Literal["on", "off", "refresh"]

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

    # Content-hash cache of raw completions (see providers/cache.py). On by default; the live
    # smoke check and anything that must hit the real API turn it off.
    llm_cache: CacheMode = "on"
    llm_cache_dir: Path = Path("outputs/cache")

    # Passed through as `reasoning_effort` when set (e.g. none/minimal/low/medium/high; accepted
    # values differ per provider, so it is not validated here). Blank means unset.
    llm_reasoning_effort: str | None = None

    # Longest edge (px) of the page images SENT to the model. Applied in memory at request time
    # to the stored page (which is never touched), so a resolution sweep needs no re-ingest.
    # Unset = send the stored page as is. Same shape rule as ingestion (docextract.imaging).
    llm_image_max_long_edge: int | None = Field(default=None, ge=1)

    @field_validator(
        "llm_reasoning_effort", "max_run_cost_usd", "llm_image_max_long_edge", mode="before"
    )
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """`NAME=` in an env file means "not set", not "an empty string to parse"."""
        if isinstance(value, str):
            return value.strip() or None
        return value

    # Hard cap on what one run may spend, in USD. Unset = no cap. Enforced by RunBudget.
    max_run_cost_usd: Decimal | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _cost_cap_needs_prices(self) -> Self:
        if (
            self.max_run_cost_usd is not None
            and self.llm_price_input_per_mtok == 0
            and self.llm_price_output_per_mtok == 0
        ):
            raise ValueError(
                "MAX_RUN_COST_USD is set but LLM_PRICE_INPUT_PER_MTOK and "
                "LLM_PRICE_OUTPUT_PER_MTOK are both 0, so every call costs $0 and the cap "
                "could never trigger. Set the prices, or unset MAX_RUN_COST_USD."
            )
        return self

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


class IngestSettings(BaseSettings):
    """How raw documents become stored pages. Separate from `Settings` so that ingestion needs no
    LLM configuration. Variables: INGEST_DPI, INGEST_MAX_LONG_EDGE, INGEST_JPEG_QUALITY."""

    model_config = SettingsConfigDict(
        env_prefix="INGEST_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    dpi: int = Field(default=200, ge=36, le=1200)  # PDF rasterization
    # Ceiling for the STORED page (see docextract.imaging for the rule). 0 = no cap, except the
    # 8000 px hard limit. What a model is sent is a separate, request-time setting.
    max_long_edge: int = Field(default=2400, ge=0)
    jpeg_quality: int = Field(default=92, ge=1, le=100)  # for pages that come from photos


@lru_cache
def get_settings() -> Settings:
    return Settings()
