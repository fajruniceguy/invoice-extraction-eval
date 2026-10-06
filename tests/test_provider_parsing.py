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
        assert usage == Usage(input_tokens=10, output_tokens=5, total_tokens=15, reported=True)

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


def usage_block(
    prompt: int, completion: int, total: int | None = None, reasoning: object = None
) -> dict[str, object]:
    block: dict[str, object] = {"prompt_tokens": prompt, "completion_tokens": completion}
    if total is not None:
        block["total_tokens"] = total
    if reasoning is not None:
        block["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return block


class TestReasoningTokens:
    def test_openai_style_reasoning_is_inside_completion_and_not_added_again(self) -> None:
        usage = parse_usage(usage_block(100, 300, total=400, reasoning=200))
        assert usage.output_tokens == 300
        assert usage.reasoning_tokens == 200
        assert usage.reasoning_outside_output == 0
        assert usage.billable_output_tokens == 300  # not 500: that would double count
        assert usage.output_includes_reasoning is True
        assert usage.reasoning_inferred is False
        assert usage.total_tokens == 400

    def test_google_style_reasoning_left_out_of_completion_is_billed_on_top(self) -> None:
        usage = parse_usage(usage_block(100, 100, total=400, reasoning=200))
        assert usage.output_tokens == 100
        assert usage.reasoning_tokens == 200
        assert usage.reasoning_outside_output == 200
        assert usage.billable_output_tokens == 300
        assert usage.output_includes_reasoning is False
        assert usage.reasoning_inferred is False

    def test_reasoning_not_reported_is_inferred_from_the_total(self) -> None:
        usage = parse_usage(usage_block(100, 100, total=500))
        assert usage.reasoning_tokens == 300
        assert usage.reasoning_inferred is True
        assert usage.reasoning_outside_output == 300
        assert usage.billable_output_tokens == 400
        assert usage.output_includes_reasoning is False

    def test_no_hidden_tokens_when_the_total_matches(self) -> None:
        usage = parse_usage(usage_block(100, 50, total=150))
        assert usage.reasoning_tokens == 0
        assert usage.reasoning_outside_output == 0
        assert usage.billable_output_tokens == 50
        assert usage.output_includes_reasoning is None  # nothing to say either way
        assert usage.reasoning_inferred is False

    def test_reported_zero_reasoning_with_a_matching_total(self) -> None:
        usage = parse_usage(usage_block(100, 50, total=150, reasoning=0))
        assert usage.reasoning_tokens == 0
        assert usage.output_includes_reasoning is None
        assert usage.billable_output_tokens == 50

    def test_without_a_total_the_openai_convention_is_assumed_and_flagged_unverified(self) -> None:
        usage = parse_usage(usage_block(100, 300, reasoning=200))
        assert usage.reasoning_tokens == 200
        assert usage.reasoning_outside_output == 0
        assert usage.billable_output_tokens == 300
        assert usage.output_includes_reasoning is None  # could not be verified
        assert usage.total_tokens is None

    def test_without_a_total_or_details(self) -> None:
        usage = parse_usage(usage_block(100, 300))
        assert (usage.reasoning_tokens, usage.billable_output_tokens) == (0, 300)
        assert usage.output_includes_reasoning is None

    def test_a_total_smaller_than_prompt_plus_completion_is_ignored(self) -> None:
        usage = parse_usage(usage_block(100, 100, total=50))
        assert usage.reasoning_outside_output == 0
        assert usage.billable_output_tokens == 100

    def test_when_reported_reasoning_disagrees_with_the_gap_everything_hidden_is_billed(
        self,
    ) -> None:
        usage = parse_usage(usage_block(100, 100, total=700, reasoning=300))
        assert usage.reasoning_tokens == 300  # what the provider said
        assert usage.reasoning_outside_output == 500  # what the total says is hidden
        assert usage.billable_output_tokens == 600
        assert usage.reasoning_inferred is False

    @pytest.mark.parametrize("details", ["many", 7, ["x"], {"reasoning_tokens": "5"}, {"x": 1}])
    def test_malformed_details_are_ignored(self, details: object) -> None:
        raw: dict[str, object] = {
            **usage_block(100, 100, total=400),
            "completion_tokens_details": details,
        }
        usage = parse_usage(raw)
        assert usage.reasoning_inferred is True  # fell back to the gap
        assert usage.reasoning_tokens == 200

    def test_boolean_reasoning_count_is_ignored(self) -> None:
        usage = parse_usage(usage_block(100, 100, total=400, reasoning=True))
        assert usage.reasoning_inferred is True

    def test_input_output_spelling_works_with_reasoning_too(self) -> None:
        usage = parse_usage({"input_tokens": 100, "output_tokens": 100, "total_tokens": 400})
        assert usage.reasoning_outside_output == 200

    def test_missing_usage_has_no_reasoning(self) -> None:
        usage = parse_usage(None)
        assert (usage.reasoning_tokens, usage.billable_output_tokens, usage.reported) == (
            0,
            0,
            False,
        )

    def test_billable_output_is_part_of_the_serialized_stats(self) -> None:
        usage = parse_usage(usage_block(100, 100, total=400, reasoning=200))
        dumped = usage.model_dump()
        assert dumped["billable_output_tokens"] == 300
        assert dumped["reasoning_tokens"] == 200
        assert dumped["output_tokens"] == 100


class TestUsageAggregation:
    def test_sums_every_counter(self) -> None:
        a = parse_usage(usage_block(100, 100, total=400, reasoning=200))  # google style
        b = parse_usage(usage_block(10, 30, total=40, reasoning=20))  # openai style
        total = a + b
        assert total.input_tokens == 110
        assert total.output_tokens == 130
        assert total.reasoning_tokens == 220
        assert total.reasoning_outside_output == 200
        assert total.total_tokens == 440
        assert total.billable_output_tokens == 330  # 300 + 30, each judged on its own call

    def test_reasoning_left_out_of_completion_is_summed_when_both_calls_have_it(self) -> None:
        first = parse_usage(usage_block(100, 100, total=400, reasoning=200))  # 200 outside
        second = parse_usage(usage_block(10, 10, total=130))  # 110 outside, inferred
        total = first + second
        assert total.reasoning_outside_output == 310
        assert total.reasoning_tokens == 310
        assert total.billable_output_tokens == 110 + 310
        assert (second + first).billable_output_tokens == total.billable_output_tokens

    def test_total_is_unknown_if_either_side_is_unknown(self) -> None:
        a = parse_usage(usage_block(100, 100, total=200))
        b = parse_usage(usage_block(10, 10))
        assert (a + b).total_tokens is None
        assert (b + a).total_tokens is None

    def test_includes_flag_is_kept_only_when_both_calls_agree(self) -> None:
        inside = parse_usage(usage_block(100, 300, total=400, reasoning=200))
        outside = parse_usage(usage_block(100, 100, total=400, reasoning=200))
        unknown = parse_usage(usage_block(100, 100))
        assert (inside + inside).output_includes_reasoning is True
        assert (outside + outside).output_includes_reasoning is False
        assert (inside + outside).output_includes_reasoning is None
        assert (inside + unknown).output_includes_reasoning is None

    def test_inferred_is_sticky(self) -> None:
        inferred = parse_usage(usage_block(100, 100, total=500))
        reported = parse_usage(usage_block(100, 100, total=400, reasoning=200))
        assert (reported + inferred).reasoning_inferred is True

    def test_reported_if_either_was(self) -> None:
        assert (Usage() + parse_usage(usage_block(1, 1))).reported is True
        assert (Usage() + Usage()).reported is False


class TestReasoningNote:
    def test_not_reported(self) -> None:
        assert "no usage" in Usage().reasoning_note

    def test_inside_completion(self) -> None:
        note = parse_usage(usage_block(100, 300, total=400, reasoning=200)).reasoning_note
        assert "200 tokens" in note
        assert "already inside completion_tokens" in note
        assert "not added again" in note

    def test_outside_completion_reported(self) -> None:
        note = parse_usage(usage_block(100, 100, total=400, reasoning=200)).reasoning_note
        assert "reported by the provider" in note
        assert "NOT inside completion_tokens" in note
        assert "200 added to billable output" in note

    def test_outside_completion_inferred(self) -> None:
        note = parse_usage(usage_block(100, 100, total=500)).reasoning_note
        assert "inferred from total_tokens" in note
        assert "300 added to billable output" in note

    def test_unverified_without_a_total(self) -> None:
        note = parse_usage(usage_block(100, 300, reasoning=200)).reasoning_note
        assert "unverified" in note

    def test_none(self) -> None:
        note = parse_usage(usage_block(100, 50, total=150)).reasoning_note
        assert note == "none: total_tokens equals prompt + completion"


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
