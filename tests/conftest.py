from collections.abc import Iterator

import pytest

from fakes.fake_llm import FakeLLMServer


@pytest.fixture
def fake_llm() -> Iterator[FakeLLMServer]:
    """A fake OpenAI-compatible server on an ephemeral local port (real sockets)."""
    with FakeLLMServer() as server:
        yield server
