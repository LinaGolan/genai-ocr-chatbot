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
    b. GPT-4o vision re-reads ONLY those fields — each from a crop of the page
       zoomed to that field's region (located via OCR word boxes), falling back to
       the full page for fields that can't be located confidently;
    c. verified values are written back and the whole record is validated again;
    d. any field that STILL fails validation is kept as-is and its issue is noted
       — in the logs and (via the validation reason) in the UI.

Native Azure OpenAI SDK only (chat.completions with an image_url content part) —
no frameworks. The single openai_client lives in shared/azure_client.py.
"""

import base64
import copy
import io
import json
import re
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF — rasterise PDF page 1 for the vision model
from PIL import Image  # crop the rendered page down to a single field's region

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
# Cap fields per vision call: the page image dominates cost and is re-sent each
# call, so batching is cheap — but reading too many fields in one call makes the
# model read each cursorily (worst on the poor scans that flag the most fields).
# Chunking bounds per-call field count; a clean form still resolves in one call.
# (Only the full-page fallback batches; located fields get a focused crop each.)
_MAX_FIELDS_PER_VISION_CALL = 6

# --- Field-region cropping ---------------------------------------------------
# We locate a flagged field on the page from its OCR word boxes and send GPT-4o a
# zoomed crop instead of the whole page: higher effective resolution on the exact
# digits/handwriting (accuracy) and a fraction of the image tokens (cost). When a
# field can't be located confidently we fall back to the full page — a mislocated
# crop would be worse than no crop, so location is deliberately conservative.
_REGION_PAD_FRAC = 0.04        # pad the located box by this fraction of page size
_MAX_REGION_AREA_FRAC = 0.5    # matches spanning >half the page are deemed unreliable
_MAX_WORDS_PER_TOKEN = 3       # a token hitting more words than this is too ambiguous

# Selection / checkbox fields are NEVER cropped to their value's location: every
# option's label is printed on the page, so locating by the (possibly wrong)
# extracted value crops to that option and biases vision into confirming it
# instead of seeing the whole option row and re-judging which box is ticked.
_NO_CROP_FIELDS = frozenset({
    "gender",
    "accidentLocation",
    "medicalInstitutionFields.healthFundMember",
})

# Template-based crops for fields whose position on the standard BL283 form is
# fixed regardless of content. (x0, y0, x1, y1) as fractions of page dimensions.
# Used instead of value-based location — gives focused resolution without
# confirmation bias from cropping to the extracted (possibly wrong) value.
_TEMPLATE_CROPS: dict[str, tuple[float, float, float, float]] = {
    # Medical-institution / HMO checkbox row is always in the bottom ~28% of the form
    "medicalInstitutionFields.healthFundMember": (0.0, 0.72, 1.0, 1.0),
}

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

# Fields that should be present on any completed BL283 submission.  When the
# LLM extractor leaves one of these empty (common on handwritten forms where the
# OCR markdown layout differs from typed examples), we trigger a vision re-read
# so GPT-4o can read the handwriting directly from the source image.
_CRITICAL_FIELDS = frozenset({
    "firstName", "lastName", "idNumber",
    "dateOfBirth", "dateOfInjury",
    "accidentLocation", "accidentDescription", "injuredBodyPart", "jobType",
})


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
    # Signature presence is a visual call OCR text can't make reliably and tends to
    # over-claim. Verify it from the image whenever the extractor claimed one exists
    # (asymmetric: catches false positives at the cost of one field, only when claimed).
    claimed_signature = ["signature"] if str(extracted.get("signature", "")).strip() else []
    # healthFundMember comes from a checkbox row whose order OCR routinely garbles (it
    # attaches the ☒ to the wrong HMO), so the OCR-text read is unreliable — always
    # re-read it from the image.
    always_verify = ["medicalInstitutionFields.healthFundMember"]
    # Empty critical fields: handwritten forms often cause the LLM to miss values
    # that are clearly visible in the image — re-read them from the source image.
    empty_critical = find_empty_critical_fields(extracted)
    targets = list(dict.fromkeys(
        [*low_conf, *sorted(failed), *claimed_signature, *always_verify, *empty_critical]
    ))  # de-dupe, keep order

    if not targets:
        return extracted, validate(extracted, ocr_result=ocr_result)

    logger.info(
        "Vision re-read triggered",
        extra={
            "low_confidence": low_conf,
            "failed_validation": sorted(failed),
            "empty_critical": empty_critical,
        },
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


def find_empty_critical_fields(extracted: dict[str, Any]) -> list[str]:
    """Return critical fields the extractor left empty.

    Handwritten forms often cause the OCR→text pipeline to miss values that
    are visible in the image; sending these to vision lets GPT-4o read the
    handwriting directly instead of relying on the OCR markdown.
    Only truly empty values are returned — a field with any content is skipped.
    """
    return [
        path for path in sorted(_CRITICAL_FIELDS)
        if not _current_display(extracted, path).strip()
    ]


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
        full_img = _render_full_page_image(file_bytes, filename)
        review = _gather_reviews(full_img, targets, extracted, ocr_result)
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


def _gather_reviews(
    full_img: Image.Image, targets: list[str], extracted: dict, ocr_result
) -> dict:
    """Re-read each target, cropping the page to the field's region when we can
    locate it (one focused call per field) and falling back to the full page —
    batched — for fields we can't locate. Results merge into one {field: value} map.
    """
    review: dict = {}
    unlocated: list[str] = []

    for path in targets:
        box = _template_region(path, full_img.size)
        if box is None:
            box = _locate_field_region(path, extracted, ocr_result, full_img.size)
        if box is None:
            unlocated.append(path)
            continue
        crop_url = _pil_to_data_url(full_img.crop(box))
        review.update(_with_backoff(lambda u=crop_url, p=path: _call_vision(u, [p], extracted)))

    if unlocated:
        full_url = _pil_to_data_url(full_img)
        for start in range(0, len(unlocated), _MAX_FIELDS_PER_VISION_CALL):
            chunk = unlocated[start:start + _MAX_FIELDS_PER_VISION_CALL]
            review.update(_with_backoff(lambda u=full_url, c=chunk: _call_vision(u, c, extracted)))

    logger.info(
        "Vision re-read dispatch",
        extra={"cropped_fields": len(targets) - len(unlocated), "fullpage_fields": len(unlocated)},
    )
    return review


def _template_region(
    path: str, img_size: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """Return a fixed pixel crop box for fields with a known template position,
    or None if no template region is defined for *path*."""
    fracs = _TEMPLATE_CROPS.get(path)
    if fracs is None:
        return None
    w, h = img_size
    x0, y0, x1, y1 = fracs
    return (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))


def _locate_field_region(
    path: str, extracted: dict, ocr_result, img_size: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """Find the pixel-space crop box for *path* on the rendered page, or None.

    Matches the field's extracted value against OCR word boxes, unions the hits,
    scales page units → rendered pixels (via page_width/height — unit-agnostic),
    and pads. Returns None when the field can't be located confidently, so the
    caller falls back to the full page rather than crop the wrong spot.
    """
    if path in _NO_CROP_FIELDS:
        return None  # selection field — must see the whole option row, never a value-located crop
    if ocr_result is None or not ocr_result.words:
        return None
    if not (ocr_result.page_width and ocr_result.page_height):
        return None

    candidates = _match_candidates(_current_display(extracted, path))
    if not candidates:
        return None

    matched: list = []
    for cand in candidates:
        hits = [w for w in ocr_result.words if _norm(w.content) == cand]
        if 0 < len(hits) <= _MAX_WORDS_PER_TOKEN:
            matched.extend(hits)
    if not matched:
        return None

    sx = img_size[0] / ocr_result.page_width
    sy = img_size[1] / ocr_result.page_height
    x0 = min(w.bbox[0] for w in matched) * sx
    y0 = min(w.bbox[1] for w in matched) * sy
    x1 = max(w.bbox[2] for w in matched) * sx
    y1 = max(w.bbox[3] for w in matched) * sy

    if (x1 - x0) * (y1 - y0) > _MAX_REGION_AREA_FRAC * img_size[0] * img_size[1]:
        return None  # scattered match across the page — not a single field region

    pad_x = _REGION_PAD_FRAC * img_size[0]
    pad_y = _REGION_PAD_FRAC * img_size[1]
    box = (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(img_size[0], x1 + pad_x),
        min(img_size[1], y1 + pad_y),
    )
    return tuple(int(round(v)) for v in box)


def _match_candidates(value: str) -> list[str]:
    """Normalised strings to look for in OCR words: the whole value, its digit run
    (for IDs / phones / concatenated dates), and each token. Short, ambiguous
    fragments are dropped to avoid matching unrelated words."""
    value = value.strip()
    if not value:
        return []
    cands: set[str] = set()
    full = _norm(value)
    if full:
        cands.add(full)
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 4:
        cands.add(digits)
    for tok in re.split(r"[\s/.\-]+", value):
        tok_n = _norm(tok)
        if len(tok_n) >= 2:
            cands.add(tok_n)
    return [c for c in cands if c]


def _apply_reread(corrected: dict, path: str, old: str, raw: str) -> str:
    """Write a re-read value back into *corrected*; return the outcome label.

    - "unread"    : model returned nothing (or an unparseable date) → keep original
    - "confirmed" : model agrees with the existing value
    - "corrected" : model returned a different, legible value → written back
    """
    if path == "signature":
        # Presence field: vision is authoritative and "" is a real verdict (no
        # signature), so it must overwrite a false-positive rather than be treated
        # as "unread". Normalise any non-empty reading to the canonical "קיימת".
        new = "קיימת" if raw else ""
        if new == old:
            return "confirmed"
        _set_path(corrected, path, new)
        return "corrected"

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

# Fields where showing the previous OCR reading anchors vision into confirming
# a wrong value instead of independently re-judging. For these we omit the
# "previously" hint and ask vision to determine the answer from scratch.
# Note: healthFundMember is NOT included here — it gets a template crop of just
# the checkbox row, so vision can see the checkboxes clearly without anchoring.
_NO_PREV_HINT_FIELDS = frozenset({
    "gender",
    "accidentLocation",
})


def _field_line(path: str, extracted: dict) -> str:
    label = VISION_FIELD_LABELS.get(path, path)
    if path in _NO_PREV_HINT_FIELDS:
        return (
            f"- {path} — {label} — "
            "OCR reading is unreliable for this field; determine the answer "
            "independently from the image (do NOT anchor on any prior reading)"
        )
    return f'- {path} — {label} — previously: "{_current_display(extracted, path)}"'


def _call_vision(data_url: str, targets: list[str], extracted: dict) -> dict:
    """Single GPT-4o vision call; returns the parsed {field: value} JSON."""
    field_lines = "\n".join(
        _field_line(path, extracted) for path in targets
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

def _render_full_page_image(file_bytes: bytes, filename: str) -> Image.Image:
    """Render the uploaded file to a PIL image of page 1 (matching the OCR step,
    which analyses pages="1"). PDFs are rasterised at _RENDER_DPI; JPGs are opened
    as-is. The image is returned so callers can crop it per field before encoding.
    """
    suffix = Path(filename).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    if suffix == ".pdf":
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            pixmap = doc.load_page(0).get_pixmap(dpi=_RENDER_DPI)
            png_bytes = pixmap.tobytes("png")
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    raise ValueError(f"Unsupported file type for vision correction: {suffix!r}")


def _pil_to_data_url(img: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG data URL for the vision API."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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
