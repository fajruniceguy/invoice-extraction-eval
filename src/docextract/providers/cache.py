"""Content-hash response cache for LLM calls.

One entry per HTTP exchange, keyed by a sha256 over everything that can change the answer:
the messages (image bytes hashed, so a page image is identified by its content), model,
endpoint, reasoning effort, temperature, max_tokens, structured-output mode and the response
schema. What is stored is the RAW completion (text, finish reason, provider payload), never a
parsed object: parsing, normalization and validation rerun on every replay, so a validator
change can never serve a stale result, and a repair sequence replays deterministically.

A cache problem must never break a run: an unreadable entry is a miss, a failed write is
logged and ignored. Entries hold no secrets (no API key; the key parts are only hashes and
settings), but they do hold model output for your documents, so the directory is gitignored.
"""

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

CACHE_VERSION = 1  # bump when the entry layout or the key recipe changes


def _sha256(data: str | bytes) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class CacheKey:
    digest: str
    parts: dict[str, Any]  # human-readable breakdown, stored next to the entry for debugging


@dataclass(frozen=True)
class CacheEntry:
    text: str
    finish_reason: str | None
    tier: str
    payload: dict[str, Any]
    latency_ms: float  # how long the original call took (replays are instant)
    created_at: str


def schema_hash(model: type[BaseModel] | None) -> str:
    """Identity of the response schema: any schema change gives a different cache key."""
    if model is None:
        return "none"
    return _sha256(_canonical(model.model_json_schema()))[:16]


def fingerprint(
    messages: Sequence[dict[str, Any]],
    *,
    model: str,
    base_url: str,
    reasoning_effort: str | None,
    temperature: float,
    max_tokens: int | None,
    structured_mode: str,
    schema_hash: str,
) -> CacheKey:
    image_hashes: list[str] = []
    scrubbed: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = str((part.get("image_url") or {}).get("url", ""))
                    digest = _sha256(url)
                    image_hashes.append(digest)
                    parts.append({"type": "image_url", "image_sha256": digest})
                else:
                    parts.append(part)
            scrubbed.append({**message, "content": parts})
        else:
            scrubbed.append(dict(message))
    parts_summary: dict[str, Any] = {
        "version": CACHE_VERSION,
        "prompt_hash": _sha256(_canonical(scrubbed)),
        "image_hashes": image_hashes,
        "model": model,
        "base_url": base_url,
        "reasoning_effort": reasoning_effort,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "structured_mode": structured_mode,
        "schema_hash": schema_hash,
    }
    return CacheKey(digest=_sha256(_canonical(parts_summary)), parts=parts_summary)


class ResponseCache:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def path_for(self, digest: str) -> Path:
        return self.directory / digest[:2] / f"{digest}.json"

    def get(self, digest: str) -> CacheEntry | None:
        path = self.path_for(digest)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            logger.warning("ignoring unreadable cache entry %s: %s", path.name, exc)
            return None
        try:
            if raw["version"] != CACHE_VERSION:
                return None
            entry = raw["entry"]
            payload = entry["payload"]
            if not isinstance(payload, dict) or not isinstance(entry["text"], str):
                raise TypeError("malformed entry")
            finish_reason = entry["finish_reason"]
            return CacheEntry(
                text=entry["text"],
                finish_reason=finish_reason if isinstance(finish_reason, str) else None,
                tier=str(entry["tier"]),
                payload=payload,
                latency_ms=float(entry["latency_ms"]),
                created_at=str(entry["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("ignoring malformed cache entry %s: %s", path.name, exc)
            return None

    def put(self, key: CacheKey, entry: CacheEntry) -> None:
        document = {
            "version": CACHE_VERSION,
            "key_parts": key.parts,
            "entry": {
                "text": entry.text,
                "finish_reason": entry.finish_reason,
                "tier": entry.tier,
                "payload": entry.payload,
                "latency_ms": entry.latency_ms,
                "created_at": entry.created_at,
            },
        }
        path = self.path_for(key.digest)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # write to a temp file in the same directory, then rename: readers never see half
            handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    json.dump(document, stream, ensure_ascii=False)
                os.replace(temp_name, path)
            except BaseException:
                Path(temp_name).unlink(missing_ok=True)
                raise
        except OSError as exc:
            logger.warning("could not write cache entry %s: %s", path.name, exc)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
