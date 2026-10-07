"""Document-level extraction: a document's pages -> ONE multi-page model call -> a validated result.

    pages, in order -> the system prompt + one user message carrying every page as an image
      -> chat(DocumentExtraction, preprocess=extraction_preprocess(N)):
           parse JSON -> normalize -> validate + evidence contract -> repair only if that fails
      -> validate_document (business rules, field checks, evidence grounding, confidence caps
         and abstention; a copy, the model's parsed output is kept)

Both what the model said (`parsed`) and what the validators concluded (`validated`) are returned.
The pages are whatever the caller passes; the runner passes STORED pages only
(docextract.ingest.stored), never raw files.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from docextract.config import CacheMode, Settings
from docextract.extract.preprocess import extraction_preprocess
from docextract.extract.prompt import SYSTEM_PROMPT, user_text
from docextract.extract.schema import DocumentExtraction
from docextract.providers.openai_compat import (
    LLMResponse,
    Message,
    OpenAICompatClient,
    build_image_message,
)
from docextract.validate import validate_document


class ExtractionRefused(Exception):
    """The document is not sent to the model (and why). No request was made."""


@dataclass(frozen=True)
class Extraction:
    response: LLMResponse[DocumentExtraction]
    parsed: DocumentExtraction  # what the model said (after normalization)
    validated: DocumentExtraction  # what the validators made of it
    pages: int


def build_messages(pages: Sequence[Path], settings: Settings) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        build_image_message(
            user_text(len(pages)), list(pages), max_long_edge=settings.llm_image_max_long_edge
        ),
    ]


async def extract_pages(
    pages: Sequence[Path],
    *,
    client: OpenAICompatClient,
    settings: Settings,
    cache: CacheMode | None = None,
) -> Extraction:
    """Extract one document from its pages (in order) in a single call. `settings` are the
    client's; `cache` overrides LLM_CACHE for this call."""
    if not pages:
        raise ExtractionRefused("no pages to extract from")
    if len(pages) > settings.llm_max_pages:
        raise ExtractionRefused(
            f"{len(pages)} pages, but LLM_MAX_PAGES is {settings.llm_max_pages}: "
            "split the bundle with page_range first"
        )
    response = await client.chat(
        build_messages(pages, settings),
        DocumentExtraction,
        temperature=0.0,
        cache=cache,
        preprocess=extraction_preprocess(len(pages)),
    )
    parsed = response.parsed
    assert parsed is not None  # chat() with a response model either parses or raises
    validated = validate_document(
        parsed,
        abstain_below=settings.abstain_below,
        normalizations=response.normalizations,
    )
    return Extraction(response, parsed, validated, len(pages))
