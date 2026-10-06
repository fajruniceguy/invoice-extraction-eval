import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from docextract.providers.cache import (
    CACHE_VERSION,
    CacheEntry,
    ResponseCache,
    fingerprint,
    schema_hash,
)

PIXELS = "data:image/png;base64,QUJDREVGRw=="  # a stand-in "page image"
MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You extract receipts."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract the fields."},
            {"type": "image_url", "image_url": {"url": PIXELS}},
        ],
    },
]
BASE: dict[str, Any] = {
    "model": "m",
    "base_url": "https://llm.test/v1/",
    "reasoning_effort": None,
    "temperature": 0.0,
    "max_tokens": None,
    "structured_mode": "auto",
    "schema_hash": "abc",
}


def digest(messages: list[dict[str, Any]] = MESSAGES, **overrides: Any) -> str:
    return fingerprint(messages, **{**BASE, **overrides}).digest


class TestFingerprint:
    def test_is_stable(self) -> None:
        assert digest() == digest() == digest(copy.deepcopy(MESSAGES))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model", "other-model"),
            ("base_url", "https://elsewhere.test/v1/"),
            ("reasoning_effort", "low"),
            ("temperature", 0.5),
            ("max_tokens", 1024),
            ("structured_mode", "json_object"),
            ("schema_hash", "def"),
        ],
    )
    def test_every_request_setting_changes_the_key(self, field: str, value: Any) -> None:
        assert digest(**{field: value}) != digest()

    def test_prompt_text_changes_the_key(self) -> None:
        changed = copy.deepcopy(MESSAGES)
        changed[0]["content"] = "You extract invoices."
        assert digest(changed) != digest()

    def test_message_order_and_roles_matter(self) -> None:
        assert digest(list(reversed(MESSAGES))) != digest()
        swapped = copy.deepcopy(MESSAGES)
        swapped[0]["role"] = "user"
        assert digest(swapped) != digest()

    def test_image_content_changes_the_key(self) -> None:
        changed = copy.deepcopy(MESSAGES)
        changed[1]["content"][1]["image_url"]["url"] = "data:image/png;base64,UVJTVFVWVw=="
        assert digest(changed) != digest()

    def test_the_same_image_gives_the_same_key_wherever_it_comes_from(self) -> None:
        rebuilt = [
            {"role": "system", "content": "You extract receipts."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the fields."},
                    {"type": "image_url", "image_url": {"url": str(PIXELS)}},
                ],
            },
        ]
        assert digest(rebuilt) == digest()

    def test_image_order_matters(self) -> None:
        two = copy.deepcopy(MESSAGES)
        two[1]["content"].append(
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,Wlla"}}
        )
        flipped = copy.deepcopy(two)
        flipped[1]["content"][1:] = list(reversed(flipped[1]["content"][1:]))
        assert digest(two) != digest(flipped)

    def test_parts_hold_hashes_not_pixels(self) -> None:
        key = fingerprint(MESSAGES, **BASE)
        assert len(key.parts["image_hashes"]) == 1
        assert "QUJDREVGRw" not in json.dumps(key.parts)
        assert key.parts["model"] == "m"
        assert key.parts["version"] == CACHE_VERSION

    def test_does_not_mutate_the_messages(self) -> None:
        before = copy.deepcopy(MESSAGES)
        fingerprint(MESSAGES, **BASE)
        assert before == MESSAGES

    def test_plain_string_content_is_handled(self) -> None:
        assert digest([{"role": "user", "content": "hello"}]) != digest(
            [{"role": "user", "content": "hello!"}]
        )


class Alpha(BaseModel):
    total: str


class Beta(BaseModel):
    total: str
    tax: str


class TestSchemaHash:
    def test_none_means_no_schema(self) -> None:
        assert schema_hash(None) == "none"

    def test_is_stable_and_distinguishes_schemas(self) -> None:
        assert schema_hash(Alpha) == schema_hash(Alpha)
        assert schema_hash(Alpha) != schema_hash(Beta)
        assert schema_hash(Alpha) != "none"


def entry(text: str = "hello") -> CacheEntry:
    return CacheEntry(
        text=text,
        finish_reason="stop",
        tier="json_schema",
        payload={"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}},
        latency_ms=123.0,
        created_at="2026-10-06T00:00:00+00:00",
    )


class TestResponseCache:
    def test_round_trip(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        key = fingerprint(MESSAGES, **BASE)
        assert cache.get(key.digest) is None
        cache.put(key, entry())
        assert cache.get(key.digest) == entry()

    def test_layout_is_sharded_by_digest_prefix(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        key = fingerprint(MESSAGES, **BASE)
        cache.put(key, entry())
        assert cache.path_for(key.digest) == tmp_path / key.digest[:2] / f"{key.digest}.json"
        assert cache.path_for(key.digest).is_file()

    def test_the_file_records_the_key_parts_for_debugging(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        key = fingerprint(MESSAGES, **BASE)
        cache.put(key, entry())
        document = json.loads(cache.path_for(key.digest).read_text(encoding="utf-8"))
        assert document["key_parts"]["model"] == "m"
        assert document["version"] == CACHE_VERSION

    def test_overwrite_replaces_the_entry(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        key = fingerprint(MESSAGES, **BASE)
        cache.put(key, entry("first"))
        cache.put(key, entry("second"))
        loaded = cache.get(key.digest)
        assert loaded is not None
        assert loaded.text == "second"

    def test_no_temp_files_are_left_behind(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        cache.put(fingerprint(MESSAGES, **BASE), entry())
        assert list(tmp_path.rglob("*.tmp")) == []

    @pytest.mark.parametrize(
        "garbage",
        [
            "",
            "not json at all",
            "[]",
            "{}",
            '{"version": 999, "entry": {}}',
            '{"version": 1}',
            '{"version": 1, "entry": {"text": "x"}}',
            '{"version": 1, "entry": {"text": 5, "finish_reason": null, "tier": "t", '
            '"payload": {}, "latency_ms": 1, "created_at": "x"}}',
            '{"version": 1, "entry": {"text": "x", "finish_reason": null, "tier": "t", '
            '"payload": [], "latency_ms": 1, "created_at": "x"}}',
            '{"version": 1, "entry": {"text": "x", "finish_reason": null, "tier": "t", '
            '"payload": {}, "latency_ms": "slow", "created_at": "x"}}',
        ],
    )
    def test_an_unusable_entry_is_a_miss_not_an_error(self, tmp_path: Path, garbage: str) -> None:
        cache = ResponseCache(tmp_path)
        key = fingerprint(MESSAGES, **BASE)
        path = cache.path_for(key.digest)
        path.parent.mkdir(parents=True)
        path.write_text(garbage, encoding="utf-8")
        assert cache.get(key.digest) is None

    def test_an_otherwise_valid_entry_from_another_cache_version_is_a_miss(
        self, tmp_path: Path
    ) -> None:
        cache = ResponseCache(tmp_path)
        key = fingerprint(MESSAGES, **BASE)
        cache.put(key, entry())
        assert cache.get(key.digest) is not None
        path = cache.path_for(key.digest)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["version"] = CACHE_VERSION + 1  # everything else is still perfectly valid
        path.write_text(json.dumps(document), encoding="utf-8")
        assert cache.get(key.digest) is None

    def test_a_non_string_finish_reason_is_dropped(self, tmp_path: Path) -> None:
        cache = ResponseCache(tmp_path)
        key = fingerprint(MESSAGES, **BASE)
        cache.put(key, CacheEntry("x", None, "none", {}, 1.0, "t"))
        loaded = cache.get(key.digest)
        assert loaded is not None
        assert loaded.finish_reason is None

    def test_a_failed_write_never_raises(self, tmp_path: Path) -> None:
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("i am a file")
        cache = ResponseCache(blocker / "cache")  # cannot be created under a file
        cache.put(fingerprint(MESSAGES, **BASE), entry())  # must not raise
        assert cache.get(fingerprint(MESSAGES, **BASE).digest) is None
