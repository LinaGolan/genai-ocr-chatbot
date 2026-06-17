from __future__ import annotations

"""
Three-signal validation for BL283 extracted fields.

Priority order (a higher-priority 'invalid' cannot be downgraded):
  1. Deterministic checks  — ID checksum, date ranges, phone format, postal, cross-field
  2. OCR confidence        — per-word confidence from Document Intelligence
  3. LLM self-review       — GPT-4o semantic plausibility (supplementary only)

All data models live in schema.py; this module is validation logic only.
"""

import json
import re
from typing import Any, Literal

from shared.azure_client import openai_client, GPT4O_DEPLOYMENT
from shared.logger import get_logger, hash_id
from part1.backend.prompts import SELF_REVIEW_SYSTEM_PROMPT, SELF_REVIEW_USER_PROMPT_TEMPLATE
from part1.backend.schema import FieldStatus, ValidationResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate(
    extracted: dict[str, Any],
    ocr_result=None,   # OCRResult | None
    ocr_markdown: str = "",
) -> ValidationResult:
    """
    Run all three validation signals; return a merged ValidationResult.

    Parameters
    ----------
    extracted:    dict output from extractor.extract_fields()
    ocr_result:   OCRResult from ocr_client.analyze_document() (or None)
    ocr_markdown: raw OCR text (used by LLM self-review)
    """
    statuses: dict[str, FieldStatus] = {}

    # 1. Deterministic (primary — cannot be overridden)
    _run_deterministic(extracted, statuses)

    # 2. OCR confidence (supporting)
    if ocr_result is not None:
        _run_ocr_confidence(extracted, ocr_result, statuses)

    # 3. LLM self-review (supplementary — cannot downgrade 'invalid')
    if ocr_markdown:
        word_conf: dict[str, float] = (
            {w.lower().strip(".,;:\"'()"): c for w, c in ocr_result.word_confidences}
            if ocr_result else {}
        )
        try:
            _run_llm_review(extracted, ocr_markdown, statuses, word_conf)
        except Exception as exc:
            logger.warning("LLM self-review skipped", extra={"error": str(exc)[:200]})

    completeness = _compute_completeness(extracted)
    accuracy_estimate = _compute_accuracy_estimate(statuses, ocr_result)

    logger.info(
        "Validation complete",
        extra={
            "completeness": round(completeness, 3),
            "accuracy_estimate": accuracy_estimate,
            "invalid_fields": [k for k, s in statuses.items() if s.status == "invalid"],
            "uncertain_count": sum(1 for s in statuses.values() if s.status == "uncertain"),
        },
    )
    return ValidationResult(
        fields=statuses,
        completeness=completeness,
        accuracy_estimate=accuracy_estimate,
    )


# ---------------------------------------------------------------------------
# Signal 1 — Deterministic checks
# ---------------------------------------------------------------------------

def _run_deterministic(extracted: dict, statuses: dict[str, FieldStatus]) -> None:
    _check_id_number(extracted, statuses)
    _check_dates(extracted, statuses)
    _check_date_ordering(extracted, statuses)
    _check_phones(extracted, statuses)
    _check_postal(extracted, statuses)
    _check_gender(extracted, statuses)


def _check_id_number(extracted: dict, statuses: dict) -> None:
    id_num = extracted.get("idNumber", "")
    if not id_num:
        return
    if _validate_israeli_id(id_num):
        statuses.setdefault("idNumber", FieldStatus("ok"))
    else:
        statuses["idNumber"] = FieldStatus("invalid", "ID must be exactly 9 digits")
        logger.warning("ID length invalid", extra={"id_hash": hash_id(id_num)})


def _validate_israeli_id(id_str: str) -> bool:
    """Israeli ID must be exactly 9 digits."""
    digits = re.sub(r"\D", "", id_str)
    return len(digits) == 9


def _check_dates(extracted: dict, statuses: dict) -> None:
    date_fields = {
        "dateOfBirth": extracted.get("dateOfBirth") or {},
        "dateOfInjury": extracted.get("dateOfInjury") or {},
        "formFillingDate": extracted.get("formFillingDate") or {},
        "formReceiptDateAtClinic": extracted.get("formReceiptDateAtClinic") or {},
    }
    for field_name, date_val in date_fields.items():
        if not isinstance(date_val, dict):
            continue
        if not any(date_val.get(k, "") for k in ("day", "month", "year")):
            continue  # all empty — skip
        err = _validate_date(date_val)
        if err:
            statuses[field_name] = FieldStatus("invalid", err)
        else:
            statuses.setdefault(field_name, FieldStatus("ok"))


def _validate_date(d: dict) -> str:
    """Return an error description, or '' if valid."""
    try:
        day = int(d.get("day") or 0)
        month = int(d.get("month") or 0)
        year = int(d.get("year") or 0)
    except (ValueError, TypeError):
        return "Non-numeric date component"
    if day and not (1 <= day <= 31):
        return f"Invalid day: {day}"
    if month and not (1 <= month <= 12):
        return f"Invalid month: {month}"
    if year and not (1900 <= year <= 2100):
        return f"Implausible year: {year}"
    return ""


def _date_to_int(d: dict) -> int | None:
    """Convert date dict to YYYYMMDD int for ordering checks; None if incomplete."""
    try:
        y = int(d.get("year") or 0)
        m = int(d.get("month") or 0)
        day = int(d.get("day") or 0)
        if y and m and day:
            return y * 10000 + m * 100 + day
    except (ValueError, TypeError):
        pass
    return None


def _check_date_ordering(extracted: dict, statuses: dict) -> None:
    dob = _date_to_int(extracted.get("dateOfBirth") or {})
    doi = _date_to_int(extracted.get("dateOfInjury") or {})
    ffd = _date_to_int(extracted.get("formFillingDate") or {})

    if dob and doi and dob >= doi:
        if statuses.get("dateOfBirth", FieldStatus()).status != "invalid":
            statuses["dateOfBirth"] = FieldStatus(
                "uncertain", "Date of birth is not before date of injury"
            )
    if doi and ffd and doi > ffd:
        if statuses.get("dateOfInjury", FieldStatus()).status != "invalid":
            statuses["dateOfInjury"] = FieldStatus(
                "uncertain", "Date of injury is after form filling date"
            )


def _check_phones(extracted: dict, statuses: dict) -> None:
    for key in ("landlinePhone", "mobilePhone"):
        val = extracted.get(key, "")
        if not val:
            continue
        digits = re.sub(r"\D", "", val)
        if _validate_israeli_phone(digits):
            statuses.setdefault(key, FieldStatus("ok"))
        else:
            statuses[key] = FieldStatus("uncertain", f"Unexpected Israeli phone format: {val!r}")


def _validate_israeli_phone(digits: str) -> bool:
    if not digits.startswith("0"):
        return False
    if len(digits) == 10 and digits[1] in "345678":
        return True  # mobile (05x) or some landlines
    if 8 <= len(digits) <= 11:
        return True  # landline area codes vary in length
    return False


def _check_postal(extracted: dict, statuses: dict) -> None:
    postal = (extracted.get("address") or {}).get("postalCode", "")
    if not postal:
        return
    clean = re.sub(r"\s", "", postal)
    if re.fullmatch(r"\d{1,7}", clean):
        statuses.setdefault("address.postalCode", FieldStatus("ok"))
    else:
        statuses["address.postalCode"] = FieldStatus(
            "uncertain", f"Postal code should be up to 7 digits, got: {postal!r}"
        )


def _check_gender(extracted: dict, statuses: dict) -> None:
    gender = extracted.get("gender", "")
    if not gender:
        return
    valid = {"זכר", "נקבה"}
    if gender in valid:
        statuses.setdefault("gender", FieldStatus("ok"))
    else:
        statuses["gender"] = FieldStatus(
            "uncertain", f"Unexpected gender value extracted: {gender!r}"
        )


# ---------------------------------------------------------------------------
# Signal 2 — OCR confidence
# ---------------------------------------------------------------------------

_OCR_LOW_THRESHOLD = 0.70


def _run_ocr_confidence(extracted: dict, ocr_result, statuses: dict) -> None:
    avg_conf: float = ocr_result.avg_confidence

    if avg_conf < _OCR_LOW_THRESHOLD:
        logger.warning("Low overall OCR confidence", extra={"avg_confidence": avg_conf})
        _flag_low_ocr_string_fields(extracted, statuses, avg_conf)

    # Per-value word lookup: flag fields whose extracted words have low confidence
    word_conf: dict[str, float] = {
        w.lower().strip(".,;:\"'()") : c
        for w, c in ocr_result.word_confidences
    }
    _check_field_word_confidence("lastName", extracted.get("lastName", ""), word_conf, statuses)
    _check_field_word_confidence("firstName", extracted.get("firstName", ""), word_conf, statuses)
    _check_field_word_confidence("idNumber", extracted.get("idNumber", ""), word_conf, statuses)
    _check_field_word_confidence("jobType", extracted.get("jobType", ""), word_conf, statuses)
    addr = extracted.get("address") or {}
    _check_field_word_confidence("address.city", addr.get("city", ""), word_conf, statuses)
    _check_field_word_confidence("address.street", addr.get("street", ""), word_conf, statuses)


def _flag_low_ocr_string_fields(
    extracted: dict, statuses: dict, avg_conf: float
) -> None:
    affected = [
        "lastName", "firstName", "idNumber", "gender", "jobType",
        "timeOfInjury", "accidentLocation", "accidentAddress",
        "accidentDescription", "injuredBodyPart",
    ]
    reason = f"Low overall OCR confidence ({avg_conf:.2f})"
    for f in affected:
        if extracted.get(f) and statuses.get(f, FieldStatus()).status != "invalid":
            statuses.setdefault(f, FieldStatus("uncertain", reason))


def _check_field_word_confidence(
    field_name: str, value: str, word_conf: dict, statuses: dict
) -> None:
    if not value or statuses.get(field_name, FieldStatus()).status == "invalid":
        return
    low_words = [
        w for w in value.split()
        if word_conf.get(w.lower().strip(".,;:\"'()"), 1.0) < _OCR_LOW_THRESHOLD
    ]
    if low_words:
        statuses.setdefault(
            field_name,
            FieldStatus("uncertain", f"Low OCR confidence for: {', '.join(low_words[:3])}"),
        )


# ---------------------------------------------------------------------------
# Signal 3 — LLM self-review (GPT-4o)
# ---------------------------------------------------------------------------

_OCR_REVIEW_CHAR_LIMIT = 4_000  # cap to avoid token overflow
_HIGH_CONF_THRESHOLD   = 0.85   # minimum per-word confidence to trust the OCR value


def _resolve_field_value(extracted: dict, fname: str) -> str:
    """Resolve a dotted field key to its string value from the extracted dict."""
    val: Any = extracted
    for part in fname.split("."):
        val = val.get(part, "") if isinstance(val, dict) else ""
    if isinstance(val, dict):
        return "/".join(filter(None, [val.get("day"), val.get("month"), val.get("year")]))
    return str(val) if val else ""


def _ocr_confirms_value(value: str, word_conf: dict[str, float], ocr_markdown: str) -> bool:
    """Return True when every token of *value* appears in the OCR with high confidence
    AND the value itself is present verbatim in the OCR text.

    Used to discard LLM self-review false-positives where the extracted value is
    demonstrably supported by high-confidence OCR evidence.
    """
    if not value or value not in ocr_markdown:
        return False
    tokens = value.split()
    return all(
        word_conf.get(t.lower().strip(".,;:\"'()"), 0.0) >= _HIGH_CONF_THRESHOLD
        for t in tokens
    )


def _run_llm_review(extracted: dict, ocr_markdown: str, statuses: dict, word_conf: dict[str, float] | None = None) -> None:
    prompt = SELF_REVIEW_USER_PROMPT_TEMPLATE.format(
        ocr_text=ocr_markdown[:_OCR_REVIEW_CHAR_LIMIT],
        extracted_json=json.dumps(extracted, ensure_ascii=False, indent=2),
    )
    response = openai_client.chat.completions.create(
        model=GPT4O_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SELF_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    review = json.loads(response.choices[0].message.content)

    flagged = []
    discarded = []
    for item in review.get("uncertain_fields", []):
        fname = item.get("field", "").strip()
        reason = item.get("reason", "Flagged by LLM self-review")
        ocr_quote = item.get("ocr_quote", "").strip()
        if not fname:
            continue
        # Discard flags where the claimed OCR evidence doesn't actually appear in the text
        if ocr_quote and ocr_quote not in ocr_markdown:
            discarded.append(fname)
            continue
        # Discard false positives: extracted value is verbatim in OCR with high confidence
        if word_conf and _ocr_confirms_value(_resolve_field_value(extracted, fname), word_conf, ocr_markdown):
            discarded.append(fname)
            continue
        # Cannot downgrade a deterministic 'invalid'
        current = statuses.get(fname, FieldStatus())
        if current.status != "invalid":
            statuses[fname] = FieldStatus("uncertain", reason)
        flagged.append(fname)

    logger.info(
        "LLM self-review complete",
        extra={"flagged_fields": flagged, "discarded_hallucinations": discarded},
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _walk_fields(data: Any) -> tuple[int, int]:
    """Return (populated, total) leaf-field counts."""
    if isinstance(data, dict):
        populated, total = 0, 0
        for v in data.values():
            p, t = _walk_fields(v)
            populated += p
            total += t
        return populated, total
    # Leaf value
    return (1 if data and data != "" else 0), 1


def _compute_completeness(extracted: dict) -> float:
    populated, total = _walk_fields(extracted)
    return populated / total if total > 0 else 0.0


_CRITICAL_FIELDS = {"idNumber", "dateOfBirth", "dateOfInjury"}


def _compute_accuracy_estimate(
    statuses: dict[str, FieldStatus], ocr_result
) -> Literal["high", "medium", "low"]:
    invalid_count = sum(1 for s in statuses.values() if s.status == "invalid")
    uncertain_count = sum(1 for s in statuses.values() if s.status == "uncertain")
    critical_invalid = any(
        s.status == "invalid" for f, s in statuses.items() if f in _CRITICAL_FIELDS
    )

    # LOW only when a critical field is invalid, or multiple fields are invalid
    if critical_invalid or invalid_count >= 2:
        return "low"

    low_ocr = ocr_result is not None and ocr_result.avg_confidence < _OCR_LOW_THRESHOLD
    if invalid_count == 1 or uncertain_count > 2 or low_ocr:
        return "medium"
    if uncertain_count > 0:
        return "medium"
    return "high"
