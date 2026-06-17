from __future__ import annotations

"""
Azure Document Intelligence Layout API wrapper.

Returns OCR markdown (with table/checkbox structure) and per-word
confidence scores for downstream validation.
All Azure SDK instantiation is delegated to shared/azure_client.py.
"""

import io
import time
from dataclasses import dataclass, field
from pathlib import Path

from azure.core.exceptions import HttpResponseError, ServiceRequestError

from shared.azure_client import document_intelligence_client
from shared.logger import get_logger

logger = get_logger(__name__)

_CONTENT_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB (Azure hard limit)
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds


@dataclass
class OCRResult:
    markdown: str
    word_confidences: list[tuple[str, float]] = field(default_factory=list)
    avg_confidence: float = 1.0
    min_confidence: float = 1.0
    page_count: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_document(file_bytes: bytes, filename: str) -> OCRResult:
    """
    Run the Document Intelligence Layout model on *file_bytes*.

    Returns an OCRResult with:
      - markdown: full extracted text with table/checkbox markup
      - word_confidences: [(word_text, confidence), …] from all pages
      - avg_confidence / min_confidence: overall OCR quality signals
      - page_count: number of pages in the document

    Raises:
      ValueError  – unsupported file type, file too large, or empty result
      HttpResponseError – non-transient Azure error (re-raised after retries)
    """
    suffix = Path(filename).suffix.lower()
    content_type = _CONTENT_TYPES.get(suffix)
    if not content_type:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Accepted formats: PDF, JPG."
        )

    if len(file_bytes) > _MAX_FILE_SIZE:
        mb = len(file_bytes) // (1024 * 1024)
        raise ValueError(f"File too large ({mb} MB). Maximum allowed size is 50 MB.")

    logger.info(
        "OCR analysis started",
        extra={"doc_file": filename, "size_bytes": len(file_bytes), "content_type": content_type},
    )
    t0 = time.perf_counter()

    result = _backoff_retry(
        lambda: _call_layout_api(file_bytes, content_type),
        retries=_MAX_RETRIES,
    )

    elapsed = time.perf_counter() - t0

    if not result.content:
        raise ValueError(
            "Document Intelligence returned no text. "
            "The file may be corrupted, blank, or not a supported form."
        )

    word_confidences = _extract_word_confidences(result)
    avg_conf, min_conf = _aggregate_confidence(word_confidences)
    page_count = len(result.pages) if result.pages else 0

    logger.info(
        "OCR analysis complete",
        extra={
            "doc_file": filename,
            "page_count": page_count,
            "chars": len(result.content),
            "avg_confidence": round(avg_conf, 3),
            "min_confidence": round(min_conf, 3),
            "latency_s": round(elapsed, 2),
        },
    )

    return OCRResult(
        markdown=result.content,
        word_confidences=word_confidences,
        avg_confidence=avg_conf,
        min_confidence=min_conf,
        page_count=page_count,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_layout_api(file_bytes: bytes, content_type: str):  # noqa: ARG001
    """Single attempt at the Document Intelligence Layout API call."""
    # content_type is validated by the caller; the SDK infers the MIME type
    # itself (it internally sends application/octet-stream — passing it again
    # raises 'got multiple values for keyword argument content_type').
    poller = document_intelligence_client.begin_analyze_document(
        "prebuilt-layout",
        io.BytesIO(file_bytes),
        pages="1",
    )
    return poller.result()


def _backoff_retry(fn, retries: int):
    """
    Call fn(). On transient Azure errors (429 / 5xx / connection errors)
    retry up to *retries* times with exponential backoff.
    Non-transient errors are re-raised immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except HttpResponseError as exc:
            if exc.status_code in (429, 500, 502, 503, 504):
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "Azure transient error — retrying",
                    extra={
                        "status_code": exc.status_code,
                        "attempt": attempt + 1,
                        "wait_s": wait,
                    },
                )
                time.sleep(wait)
                last_exc = exc
            else:
                raise
        except ServiceRequestError as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "Azure connection error — retrying",
                extra={"attempt": attempt + 1, "wait_s": wait},
            )
            time.sleep(wait)
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def _extract_word_confidences(result) -> list[tuple[str, float]]:
    """Collect per-word confidence scores from all pages."""
    confidences: list[tuple[str, float]] = []
    for page in result.pages or []:
        for word in page.words or []:
            if word.confidence is not None:
                confidences.append((word.content, float(word.confidence)))
    return confidences


def _aggregate_confidence(word_confidences: list[tuple[str, float]]) -> tuple[float, float]:
    if not word_confidences:
        return 1.0, 1.0
    scores = [c for _, c in word_confidences]
    return sum(scores) / len(scores), min(scores)
