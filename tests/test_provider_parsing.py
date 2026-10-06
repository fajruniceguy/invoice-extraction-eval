import base64
from decimal import Decimal
from pathlib import Path

import pytest

from docextract.providers.openai_compat import (
    Usage,
    build_image_message,
    parse_json_payload,
    parse_usage,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 16


class TestParseJsonPayload:
    def test_plain_object(self) -> None:
        assert parse_json_payload('{"a": 1}') == {"a": 1}

    def test_surrounding_whitespace(self) -> None:
        assert parse_json_payload('\n  {"a": 1}  \n') == {"a": 1}

    @pytest.mark.parametrize("fence", ["```json", "```JSON", "```"])
    def test_code_fence(self, fence: str) -> None:
        assert parse_json_payload(f'{fence}\n{{"a": [1, 2]}}\n```') == {"a": [1, 2]}

    def test_prose_before_and_after(self) -> None:
        text = 'Sure! Here is the JSON:\n{"a": {"b": 2}}\nHope that helps.'
        assert parse_json_payload(text) == {"a": {"b": 2}}

    def test_braces_inside_strings(self) -> None:
        assert parse_json_payload('{"a": "}{ not a brace {"}') == {"a": "}{ not a brace {"}

    def test_stray_braces_in_prose_before_the_real_object(self) -> None:
        assert parse_json_payload('Note {see below}. {"a": 1}') == {"a": 1}

    def test_first_object_wins(self) -> None:
        assert parse_json_payload('{"a": 1} {"a": 2}') == {"a": 1}

    def test_unicode(self) -> None:
        assert parse_json_payload('{"name": "Café Müller ¥"}') == {"name": "Café Müller ¥"}

    def test_non_integers_are_decimal_not_float(self) -> None:
        data = parse_json_payload('{"a": 0.1, "b": 0.2, "c": [1.5, {"d": -2.25}]}')
        assert all(isinstance(v, Decimal) for v in (data["a"], data["b"], data["c"][0]))
        assert isinstance(data["c"][1]["d"], Decimal)

    def test_point_one_plus_point_two_is_exactly_point_three(self) -> None:
        data = parse_json_payload('{"a": 0.1, "b": 0.2}')
        assert data["a"] + data["b"] == Decimal("0.3")
        assert 0.1 + 0.2 != 0.3  # the float answer this avoids: 0.30000000000000004

    def test_trailing_zero_keeps_its_scale(self) -> None:
        value = parse_json_payload('{"t": 1250000.10}')["t"]
        assert value == Decimal("1250000.10")
        assert str(value) == "1250000.10"
        assert value.as_tuple().exponent == -2  # a float would have given 1250000.1 (exponent -1)

    def test_more_digits_than_a_float_can_hold_are_kept(self) -> None:
        value = parse_json_payload('{"t": 12345678901234567.89}')["t"]
        assert value == Decimal("12345678901234567.89")
        assert float(value) != value  # as a float this would have been 12345678901234568.0

    def test_float_noise_in_the_text_is_not_rounded_away(self) -> None:
        value = parse_json_payload('{"t": 0.30000000000000004}')["t"]
        assert value == Decimal("0.30000000000000004")

    def test_exponent_notation(self) -> None:
        assert parse_json_payload('{"t": 1.25e6}')["t"] == Decimal(1250000)

    def test_integers_stay_integers(self) -> None:
        data = parse_json_payload('{"qty": 3, "page": 1, "big": 123456789012345678901234567890}')
        assert data["qty"] == 3
        assert type(data["qty"]) is int
        assert data["big"] == 123456789012345678901234567890

    def test_numbers_inside_strings_are_untouched(self) -> None:
        assert parse_json_payload('{"s": "1250000.10"}') == {"s": "1250000.10"}

    @pytest.mark.parametrize(
        "text",
        ["", "   ", "no json here", "[1, 2, 3]", '{"a": 1', '{"a": [1, 2', "{'a': 1}"],
    )
    def test_not_an_object_is_an_error(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_json_payload(text)


class TestParseUsage:
    def test_openai_names(self) -> None:
        usage = parse_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        assert usage == Usage(input_tokens=10, output_tokens=5, reported=True)

    def test_input_output_names(self) -> None:
        usage = parse_usage({"input_tokens": 7, "output_tokens": 3})
        assert usage == Usage(input_tokens=7, output_tokens=3, reported=True)

    def test_partial_usage_is_still_reported(self) -> None:
        assert parse_usage({"prompt_tokens": 9}) == Usage(
            input_tokens=9, output_tokens=0, reported=True
        )

    @pytest.mark.parametrize("raw", [None, {}, [], "x", {"total_tokens": 4}])
    def test_missing_usage_is_not_reported(self, raw: object) -> None:
        assert parse_usage(raw) == Usage(input_tokens=0, output_tokens=0, reported=False)

    def test_non_integer_counts_are_ignored(self) -> None:
        usage = parse_usage({"prompt_tokens": "10", "completion_tokens": True})
        assert usage.reported is False


class TestBuildImageMessage:
    def test_text_then_images_in_order(self) -> None:
        message = build_image_message("read this", [PNG, JPEG, WEBP])
        assert message["role"] == "user"
        parts = message["content"]
        assert parts[0] == {"type": "text", "text": "read this"}
        urls = [part["image_url"]["url"] for part in parts[1:]]
        assert [url.split(";")[0] for url in urls] == [
            "data:image/png",
            "data:image/jpeg",
            "data:image/webp",
        ]
        assert all(part["type"] == "image_url" for part in parts[1:])

    def test_base64_round_trips(self) -> None:
        url = build_image_message("x", [PNG])["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == PNG

    def test_reads_paths(self, tmp_path: Path) -> None:
        path = tmp_path / "page.png"
        path.write_bytes(PNG)
        url = build_image_message("x", [path])["content"][1]["image_url"]["url"]
        assert base64.b64decode(url.split(",", 1)[1]) == PNG

    def test_custom_role(self) -> None:
        assert build_image_message("x", [PNG], role="assistant")["role"] == "assistant"

    def test_no_images_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="at least one image"):
            build_image_message("x", [])

    def test_unknown_format_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="unsupported image format"):
            build_image_message("x", [b"GIF89a" + b"\x00" * 10])
