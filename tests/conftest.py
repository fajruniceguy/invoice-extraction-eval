from collections.abc import Iterator
from pathlib import Path

import pytest

from fakes.fake_llm import FakeLLMServer


@pytest.fixture(autouse=True)
def isolate_llm_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The response cache is ON by default, so every test would otherwise write into the repo's
    outputs/cache and could replay an answer cached by another test. Default it to off and to a
    throwaway directory; tests that exercise the cache opt in explicitly (init kwargs beat env)."""
    monkeypatch.setenv("LLM_CACHE", "off")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "llm-cache"))


@pytest.fixture
def fake_llm() -> Iterator[FakeLLMServer]:
    """A fake OpenAI-compatible server on an ephemeral local port (real sockets)."""
    with FakeLLMServer() as server:
        yield server
