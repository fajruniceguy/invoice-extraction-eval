from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from docextract.config import Settings, get_settings

LLM_ENV_VARS = [
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_PRICE_INPUT_PER_MTOK",
    "LLM_PRICE_OUTPUT_PER_MTOK",
    "LLM_TIMEOUT_S",
    "LLM_MAX_RETRIES",
    "LLM_BACKOFF_BASE_S",
    "LLM_BACKOFF_MAX_S",
    "LLM_MAX_REPAIRS",
    "LLM_REASONING_EFFORT",
    "LLM_STRUCTURED_MODE",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def make(**overrides: object) -> Settings:
    values: dict[str, object] = {"llm_base_url": "http://localhost:1/v1", "llm_model": "m"}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_MTOK", "0.30")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_MTOK", "2.50")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    s = Settings(_env_file=None)
    assert s.llm_base_url == "https://example.test/v1/"
    assert s.llm_model == "some-model"
    assert s.llm_api_key.get_secret_value() == "sk-secret"
    assert s.llm_price_input_per_mtok == Decimal("0.30")
    assert s.llm_max_retries == 2
    assert s.llm_structured_mode == "auto"


def test_reads_dotenv_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("LLM_BASE_URL=http://localhost:11434/v1\nLLM_MODEL=from-dotenv\n")
    s = Settings(_env_file=env)
    assert s.llm_model == "from-dotenv"
    assert s.llm_api_key.get_secret_value() == ""


def test_model_and_base_url_have_no_defaults() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None)
    missing = {e["loc"][0] for e in exc.value.errors() if e["type"] == "missing"}
    assert missing == {"llm_base_url", "llm_model"}


def test_api_key_is_not_leaked_in_repr_or_str() -> None:
    s = make(llm_api_key="sk-very-secret")
    assert "sk-very-secret" not in repr(s)
    assert "sk-very-secret" not in str(s)
    assert "sk-very-secret" not in s.model_dump_json()


@pytest.mark.parametrize("bad", ["localhost:11434/v1", "ftp://x", ""])
def test_rejects_non_http_base_url(bad: str) -> None:
    with pytest.raises(ValidationError):
        make(llm_base_url=bad)


def test_rejects_blank_model() -> None:
    with pytest.raises(ValidationError):
        make(llm_model="   ")


def test_rejects_unknown_structured_mode() -> None:
    with pytest.raises(ValidationError):
        make(llm_structured_mode="xml")


def test_cost_usd_uses_per_million_prices() -> None:
    s = make(llm_price_input_per_mtok="0.30", llm_price_output_per_mtok="2.50")
    assert s.cost_usd(1_000_000, 500_000) == Decimal("1.55")
    assert s.cost_usd(0, 0) == Decimal(0)


def test_cost_is_zero_when_prices_unset() -> None:
    assert make().cost_usd(12_345, 678) == Decimal(0)


def test_rejects_negative_price() -> None:
    with pytest.raises(ValidationError):
        make(llm_price_input_per_mtok="-1")


def test_max_repairs_defaults_to_one_and_can_be_set(monkeypatch: pytest.MonkeyPatch) -> None:
    assert make().llm_max_repairs == 1
    monkeypatch.setenv("LLM_MAX_REPAIRS", "0")
    assert make().llm_max_repairs == 0
    monkeypatch.setenv("LLM_MAX_REPAIRS", "3")
    assert make().llm_max_repairs == 3


def test_rejects_negative_max_repairs() -> None:
    with pytest.raises(ValidationError):
        make(llm_max_repairs=-1)


def test_reasoning_effort_is_unset_by_default() -> None:
    assert make().llm_reasoning_effort is None


@pytest.mark.parametrize(
    ("raw", "expected"), [("low", "low"), ("  high ", "high"), ("", None), ("   ", None)]
)
def test_reasoning_effort_from_env_is_trimmed_and_blank_means_unset(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str | None
) -> None:
    monkeypatch.setenv("LLM_REASONING_EFFORT", raw)
    assert make().llm_reasoning_effort == expected


def test_reasoning_effort_is_not_restricted_to_a_known_list() -> None:
    # accepted values differ per provider (none/minimal/low/medium/high/...), so pass through
    assert make(llm_reasoning_effort="xhigh").llm_reasoning_effort == "xhigh"


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
