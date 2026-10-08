from decimal import Decimal

import pytest
from pydantic import ValidationError

from docextract.config import Settings

PRELABEL_ENV_VARS = [
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_PRICE_INPUT_PER_MTOK",
    "LLM_PRICE_OUTPUT_PER_MTOK",
    "LLM_STRUCTURED_MODE",
    "LLM_REASONING_EFFORT",
    "LLM_IMAGE_MAX_LONG_EDGE",
    "MAX_RUN_COST_USD",
    "PRELABEL_BASE_URL",
    "PRELABEL_API_KEY",
    "PRELABEL_MODEL",
    "PRELABEL_PRICE_INPUT_PER_MTOK",
    "PRELABEL_PRICE_OUTPUT_PER_MTOK",
    "PRELABEL_STRUCTURED_MODE",
    "PRELABEL_IMAGE_MAX_LONG_EDGE",
    "PRELABEL_MAX_PAGES",
    "PRELABEL_MAX_TOKENS",
    "PRELABEL_MAX_RUN_COST_USD",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PRELABEL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def make(**overrides: object) -> Settings:
    values: dict[str, object] = {"llm_base_url": "http://localhost:1/v1", "llm_model": "m"}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def prelabel(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_base_url": "http://extractor.test/v1/",
        "llm_model": "gemini-x",
        "llm_api_key": "sk-extractor",
        "llm_reasoning_effort": "low",
        "max_run_cost_usd": "5",
        "llm_price_input_per_mtok": "1.5",
        "llm_price_output_per_mtok": "9",
        "llm_max_repairs": 2,
        "prelabel_base_url": "https://api.anthropic.com/v1/",
        "prelabel_api_key": "sk-prelabel",
        "prelabel_model": "claude-haiku-4-5",
        "prelabel_price_input_per_mtok": "1",
        "prelabel_price_output_per_mtok": "5",
    }
    values.update(overrides)
    return make(**values)


class TestPrelabelSettings:
    def test_pre_labeling_is_not_configured_by_default(self) -> None:
        s = make()
        assert (s.prelabel_base_url, s.prelabel_model) == (None, None)
        assert s.prelabel_structured_mode == "prompt"
        assert s.prelabel_send_temperature is True
        assert s.prelabel_image_max_long_edge == 1568
        assert (s.prelabel_max_pages, s.prelabel_max_tokens) == (15, 8000)
        assert s.prelabel_max_run_cost_usd is None

    def test_the_client_settings_point_at_the_prelabel_endpoint(self) -> None:
        derived = prelabel().prelabel_settings()
        assert derived.llm_base_url == "https://api.anthropic.com/v1/"
        assert derived.llm_model == "claude-haiku-4-5"
        assert derived.llm_api_key.get_secret_value() == "sk-prelabel"
        assert derived.llm_price_input_per_mtok == Decimal(1)
        assert derived.llm_price_output_per_mtok == Decimal(5)
        assert derived.cost_usd(1_000_000, 1_000_000) == Decimal(6)

    def test_it_pins_the_prompt_only_mode_and_the_prelabel_image_size(self) -> None:
        derived = prelabel(
            llm_structured_mode="json_schema", llm_image_max_long_edge=3000
        ).prelabel_settings()
        assert derived.llm_structured_mode == "prompt"
        assert derived.llm_image_max_long_edge == 1568

    def test_the_prelabel_temperature_switch_is_its_own_not_the_extractors(self) -> None:
        assert prelabel().prelabel_settings().llm_send_temperature is True
        derived = prelabel(prelabel_send_temperature=False).prelabel_settings()
        assert derived.llm_send_temperature is False
        # the extractor keeps sending it whatever the pre-labeler does
        assert prelabel(prelabel_send_temperature=False).llm_send_temperature is True
        assert prelabel(llm_send_temperature=False).prelabel_settings().llm_send_temperature is True

    def test_the_structured_mode_can_be_any_tier_for_the_prelabeler(self) -> None:
        for mode in ("json_object", "json_schema", "prompt"):
            derived = prelabel(prelabel_structured_mode=mode).prelabel_settings()
            assert derived.llm_structured_mode == mode

    def test_the_extractors_reasoning_effort_and_cost_cap_do_not_carry_over(self) -> None:
        derived = prelabel().prelabel_settings()
        assert derived.llm_reasoning_effort is None
        assert derived.max_run_cost_usd is None  # the extractor's cap is not the pre-labeler's
        capped = prelabel(prelabel_max_run_cost_usd="0.5").prelabel_settings()
        assert capped.max_run_cost_usd == Decimal("0.5")

    def test_everything_else_is_inherited(self) -> None:
        derived = prelabel(
            llm_max_repairs=3, llm_timeout_s=33, llm_cache="refresh"
        ).prelabel_settings()
        assert (derived.llm_max_repairs, derived.llm_timeout_s) == (3, 33)
        assert derived.llm_cache == "refresh"

    def test_the_extractors_own_settings_are_untouched(self) -> None:
        s = prelabel()
        s.prelabel_settings()
        assert (s.llm_model, s.llm_structured_mode, s.llm_reasoning_effort) == (
            "gemini-x",
            "auto",
            "low",
        )

    @pytest.mark.parametrize(
        ("unset", "message"),
        [
            ({"prelabel_base_url": None}, "PRELABEL_BASE_URL not set"),
            ({"prelabel_model": None}, "PRELABEL_MODEL not set"),
            (
                {"prelabel_base_url": None, "prelabel_model": None},
                "PRELABEL_BASE_URL and PRELABEL_MODEL not set",
            ),
        ],
    )
    def test_it_refuses_to_run_unconfigured(self, unset: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            prelabel(**unset).prelabel_settings()

    def test_blank_values_mean_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRELABEL_BASE_URL", "  ")
        monkeypatch.setenv("PRELABEL_MODEL", "")
        monkeypatch.setenv("PRELABEL_MAX_RUN_COST_USD", "")
        s = Settings(_env_file=None, llm_base_url="http://x/v1", llm_model="m")
        assert (s.prelabel_base_url, s.prelabel_model, s.prelabel_max_run_cost_usd) == (
            None,
            None,
            None,
        )

    def test_it_reads_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name, value in {
            "PRELABEL_BASE_URL": " https://api.anthropic.com/v1/ ",
            "PRELABEL_MODEL": "claude-haiku-4-5",
            "PRELABEL_API_KEY": "sk-p",
            "PRELABEL_PRICE_INPUT_PER_MTOK": "1.00",
            "PRELABEL_MAX_PAGES": "9",
            "PRELABEL_MAX_TOKENS": "4000",
        }.items():
            monkeypatch.setenv(name, value)
        s = Settings(_env_file=None, llm_base_url="http://x/v1", llm_model="m")
        assert s.prelabel_base_url == "https://api.anthropic.com/v1/"
        assert s.prelabel_api_key.get_secret_value() == "sk-p"
        assert (s.prelabel_max_pages, s.prelabel_max_tokens) == (9, 4000)
        assert s.prelabel_price_input_per_mtok == Decimal("1.00")

    @pytest.mark.parametrize("bad", ["api.anthropic.com/v1", "ftp://x", "claude"])
    def test_the_base_url_must_be_http(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="PRELABEL_BASE_URL must start with"):
            prelabel(prelabel_base_url=bad)

    @pytest.mark.parametrize(
        "field", ["prelabel_price_input_per_mtok", "prelabel_price_output_per_mtok"]
    )
    def test_negative_prices_are_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            prelabel(**{field: "-1"})

    @pytest.mark.parametrize(
        "overrides",
        [
            {"prelabel_max_pages": 0},
            {"prelabel_max_tokens": 100},
            {"prelabel_image_max_long_edge": 0},
            {"prelabel_max_run_cost_usd": "0"},
        ],
    )
    def test_bounds(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            prelabel(**overrides)

    def test_a_cap_needs_prices_just_like_the_extractors(self) -> None:
        with pytest.raises(ValidationError, match="PRELABEL_MAX_RUN_COST_USD is set but"):
            prelabel(
                prelabel_max_run_cost_usd="1",
                prelabel_price_input_per_mtok="0",
                prelabel_price_output_per_mtok="0",
            )

    def test_the_prelabel_key_is_not_leaked(self) -> None:
        s = prelabel()
        assert "sk-prelabel" not in repr(s) and "sk-prelabel" not in str(s)
        assert "sk-prelabel" not in repr(s.prelabel_settings())


class TestSameModel:
    def test_same_endpoint_and_model_is_detected_despite_slash_and_case(self) -> None:
        s = prelabel(llm_base_url="https://api.anthropic.com/v1", llm_model="Claude-Haiku-4-5")
        assert s.same_model_as_prelabel()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"llm_model": "claude-sonnet-5-5", "llm_base_url": "https://api.anthropic.com/v1/"},
            {"llm_model": "claude-haiku-4-5", "llm_base_url": "http://localhost:8080/v1"},
        ],
    )
    def test_a_different_model_or_a_different_endpoint_is_fine(
        self, overrides: dict[str, str]
    ) -> None:
        assert not prelabel(**overrides).same_model_as_prelabel()

    def test_unconfigured_pre_labeling_is_never_the_same(self) -> None:
        assert not make().same_model_as_prelabel()
