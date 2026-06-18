from __future__ import annotations

"""
Vision-based correction of fields that OCR read poorly or that fail validation,
followed by a re-validation pass.

Two triggers send a field back to GPT-4o vision for a fresh read of the SOURCE
IMAGE:
  1. Low OCR confidence (< 0.70) — the OCR markdown the text extractor saw is
     unreliable for that field.
  2. Failed deterministic validation — the extracted value is provably wrong or
     suspect (bad ID length, impossible date, malformed phone, …).

Flow:  OCR -> extract -> [correct_and_validate] -> result
    a. collect the trigger fields (low-confidence ∪ validation failures);
    b. one GPT-4o vision call re-reads ONLY those fields from the image;
    c. verified values are written back and the whole record is validated again;
    d. any field that STILL fails validation is kept as-is and its issue is noted
       — in the logs and (via the validation reason) in the UI.

Native Azure OpenAI SDK only (chat.completions with an image_url content part) —
no frameworks. The single openai_client lives in shared/azure_client.py.
"""

import base64
import copy
import json
import re
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF — rasterise PDF page 1 for the vision model

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from shared.azure_client import openai_client, GPT4O_DEPLOYMENT
from shared.logger import get_logger, hash_id
from part1.backend.prompts import (
    VISION_CORRECTION_SYSTEM_PROMPT,
    VISION_CORRECTION_USER_PROMPT_TEMPLATE,
    VISION_FIELD_LABELS,
)
from part1.backend.schema import ValidationResult
from part1.backend.validator import validate, failing_fields

logger = get_logger(__name__)

_LOW_CONF_THRESHOLD = 0.70   # must mirror validator._OCR_LOW_THRESHOLD
_RENDER_DPI = 200            # PDF rasterisation quality for the vision read
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0          # seconds

# Fields whose extracted value is matched word-by-word against OCR confidences
# (mirrors validator Signal 2's per-word checks).
_PER_WORD_FIELDS = (
    "lastName", "firstName", "idNumber", "jobType", "address.city", "address.street",
)
# Additional fields treated as suspect when the whole page is low-confidence
# (mirrors validator._flag_low_ocr_string_fields).
_LOW_AVG_FIELDS = (
    "lastName", "firstName", "idNumber", "gender", "jobType",
    "timeOfInjury", "accidentLocation", "accidentAddress",
    "accidentDescription", "injuredBodyPart",
)

# Top-level fields stored as a {day, month, year} dict rather than a string.
_DATE_FIELDS = frozenset(
    {"dateOfBirth", "dateOfInjury", "formFillingDate", "formReceiptDateAtClinic"}
)

_TRUSTED_OUTCOMES = ("corrected", "confirmed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def correct_and_validate(
    extracted: dict[str, Any],
    ocr_result,                 # OCRResult | None
    file_bytes: bytes,
    filename: str,
) -> tuple[dict[str, Any], ValidationResult]:
    """
    Re-read trigger fields from the source image, write back verified values, and
    return (corrected_extraction, ValidationResult) for the corrected data.

    Trigger fields = low-OCR-confidence fields ∪ fields that fail deterministic
    validation. Fields the vision pass verified are passed to the validator as
    trusted so they are not re-flagged for low confidence — but they are still
    subject to the deterministic checks. Any field that still fails validation
    after the re-read keeps its value and has the issue appended to its reason
    (and logged).
    """
    low_conf = find_low_confidence_fields(extracted, ocr_result)
    failed = failing_fields(extracted)
    targets = list(dict.fromkeys([*low_conf, *sorted(failed)]))  # de-dupe, keep order

    if not targets:
        return extracted, validate(extracted, ocr_result=ocr_result)

    logger.info(
        "Vision re-read triggered",
        extra={"low_confidence": low_conf, "failed_validation": sorted(failed)},
    )

    corrected, corrections = _reread_fields(extracted, ocr_result, file_bytes, filename, targets)

    result = validate(
        corrected, ocr_result=ocr_result, trusted_fields=trusted_fields(corrections)
    )
    _annotate_unresolved(result, targets, corrections)
    return corrected, result


def find_low_confidence_fields(
    extracted: dict[str, Any], ocr_result, threshold: float = _LOW_CONF_THRESHOLD
) -> list[str]:
    """Return the dotted paths of populated fields read with low OCR confidence.

    Combines the same two signals validator uses: per-word low confidence, plus
    (when the whole page averages below threshold) the OCR-sensitive string set.
    """
    if ocr_result is None:
        return []

    word_conf = {_norm(w): c for w, c in ocr_result.word_confidences}
    targets: list[str] = []

    for path in _PER_WORD_FIELDS:
        value = _get_path(extracted, path)
        if not value:
            continue
        if any(word_conf.get(_norm(tok), 1.0) < threshold for tok in value.split()):
            targets.append(path)

    if ocr_result.avg_confidence < threshold:
        for path in _LOW_AVG_FIELDS:
            if _get_path(extracted, path) and path not in targets:
                targets.append(path)

    return targets


def trusted_fields(corrections: dict[str, dict]) -> set[str]:
    """Fields the vision stage verified (corrected or confirmed) — these should
    not be re-flagged as low-OCR-confidence by the validator."""
    return {p for p, c in corrections.items() if c["outcome"] in _TRUSTED_OUTCOMES}


# ---------------------------------------------------------------------------
# Vision re-read
# ---------------------------------------------------------------------------

def _reread_fields(
    extracted: dict[str, Any],
    ocr_result,
    file_bytes: bytes,
    filename: str,
    targets: list[str],
) -> tuple[dict[str, Any], dict[str, dict]]:
    """Run one GPT-4o vision call over *targets* and write verified values back.

    Returns (corrected, corrections). On any failure the original extraction is
    returned unchanged with an empty corrections map — this stage is additive and
    never blocks the pipeline.
    """
    try:
        data_url = _to_image_data_url(file_bytes, filename)
        review = _with_backoff(lambda: _call_vision(data_url, targets, extracted))
    except Exception as exc:
        logger.warning("Vision re-read skipped", extra={"error": str(exc)[:200]})
        return extracted, {}

    corrected = copy.deepcopy(extracted)
    corrections: dict[str, dict] = {}

    for path in targets:
        old = _current_display(extracted, path)
        raw = str(review.get(path, "")).strip()
        if path == "idNumber":
            raw = re.sub(r"\D", "", raw)

        outcome = _apply_reread(corrected, path, old, raw)
        corrections[path] = {"outcome": outcome}
        _log_field_outcome(path, outcome, old, raw)

    n_corrected = sum(1 for c in corrections.values() if c["outcome"] == "corrected")
    logger.info(
        "Vision re-read complete",
        extra={"checked": len(targets), "corrected": n_corrected},
    )
    return corrected, corrections


def _apply_reread(corrected: dict, path: str, old: str, raw: str) -> str:
    """Write a re-read value back into *corrected*; return the outcome label.

    - "unread"    : model returned nothing (or an unparseable date) → keep original
    - "confirmed" : model agrees with the existing value
    - "corrected" : model returned a different, legible value → written back
    """
    if not raw:
        return "unread"

    if path in _DATE_FIELDS:
        parsed = _parse_date(raw)
        if parsed is None:
            return "unread"
        if _date_display(parsed) == old:
            return "confirmed"
        _set_date(corrected, path, parsed)
        return "corrected"

    if raw == old:
        return "confirmed"
    _set_path(corrected, path, raw)
    return "corrected"


def _annotate_unresolved(
    result: ValidationResult, targets: list[str], corrections: dict[str, dict]
) -> None:
    """For fields sent to vision that STILL fail validation, keep the value but
    append the issue to the field's reason and log it (issue surfaced in both
    logs and — via the reason — the UI)."""
    unresolved: list[str] = []
    for path in targets:
        status = result.fields.get(path)
        if status is None or status.status == "ok":
            continue  # resolved by the re-read (or never a hard failure)

        outcome = corrections.get(path, {}).get("outcome", "unread")
        note = (
            "could not be re-read from form image; original kept"
            if outcome == "unread"
            else "re-read from form image but still fails validation"
        )
        if note not in status.reason:
            status.reason = f"{status.reason} — {note}".strip(" —")
        unresolved.append(path)
        logger.warning(
            "Field unresolved after vision re-read",
            extra={"field": path, "status": status.status, "outcome": outcome,
                   "issue": status.reason},
        )

    if unresolved:
        logger.warning(
            "Vision correction left unresolved fields",
            extra={"fields": unresolved},
        )


# ---------------------------------------------------------------------------
# Azure OpenAI call
# ---------------------------------------------------------------------------

def _call_vision(data_url: str, targets: list[str], extracted: dict) -> dict:
    """Single GPT-4o vision call; returns the parsed {field: value} JSON."""
    field_lines = "\n".join(
        f'- {path} — {VISION_FIELD_LABELS.get(path, path)} — previously: "{_current_display(extracted, path)}"'
        for path in targets
    )
    user_text = VISION_CORRECTION_USER_PROMPT_TEMPLATE.format(
        fields_block=field_lines,
        keys_list=", ".join(targets),
    )
    response = openai_client.chat.completions.create(
        model=GPT4O_DEPLOYMENT,
        messages=[
            {"role": "system", "content": VISION_CORRECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def _with_backoff(fn):
    """Retry transient Azure OpenAI errors with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                "Vision re-read transient error — retrying",
                extra={"attempt": attempt + 1, "wait_s": wait},
            )
            time.sleep(wait)
            last_exc = exc
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def _to_image_data_url(file_bytes: bytes, filename: str) -> str:
    """Turn the uploaded file into a base64 image data URL for the vision API.

    JPGs are sent as-is; PDFs are rasterised — page 1 only, matching the OCR step
    (ocr_client analyses pages="1"). GPT-4o vision cannot read a PDF directly.
    """
    suffix = Path(filename).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "data:image/jpeg;base64," + base64.b64encode(file_bytes).decode()
    if suffix == ".pdf":
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            pixmap = doc.load_page(0).get_pixmap(dpi=_RENDER_DPI)
            png_bytes = pixmap.tobytes("png")
        return "data:image/png;base64," + base64.b64encode(png_bytes).decode()
    raise ValueError(f"Unsupported file type for vision correction: {suffix!r}")


# ---------------------------------------------------------------------------
# Value access (string leaves + date dicts)
# ---------------------------------------------------------------------------

def _norm(token: str) -> str:
    return token.lower().strip(".,;:\"'()")


def _current_display(extracted: dict, path: str) -> str:
    """Human-readable current value for *path* (dates rendered DD/MM/YYYY)."""
    if path in _DATE_FIELDS:
        return _date_display(extracted.get(path) or {})
    return _get_path(extracted, path)


def _date_display(d: dict) -> str:
    return "/".join(
        filter(None, [str(d.get("day", "")), str(d.get("month", "")), str(d.get("year", ""))])
    )


def _parse_date(value: str) -> dict | None:
    """Parse a 'DD/MM/YYYY' (any common separator) string into components."""
    parts = [p for p in re.split(r"[/.\-\s]+", value.strip()) if p]
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    day, month, year = parts
    return {"day": day, "month": month, "year": year}


def _set_date(data: dict, path: str, parsed: dict) -> None:
    data[path] = {"day": parsed["day"], "month": parsed["month"], "year": parsed["year"]}


def _get_path(data: dict, path: str) -> str:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(part, "")
    return cur if isinstance(cur, str) else ""


def _set_path(data: dict, path: str, value: str) -> None:
    parts = path.split(".")
    cur = data
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_field_outcome(path: str, outcome: str, old: str, new: str) -> None:
    """Log per-field provenance without leaking raw PII (idNumber is hashed)."""
    if path == "idNumber":
        logger.info(
            "Vision field result",
            extra={"field": path, "outcome": outcome,
                   "old_hash": hash_id(old), "new_hash": hash_id(new)},
        )
    else:
        logger.info(
            "Vision field result",
            extra={"field": path, "outcome": outcome, "changed": outcome == "corrected"},
        )
