from __future__ import annotations

"""
Two-signal validation for BL283 extracted fields.

Priority order (a higher-priority 'invalid' cannot be downgraded):
  1. Deterministic checks  — ID 9-digit length, date ranges, phone format, postal, cross-field
  2. OCR confidence        — per-word confidence from Document Intelligence

Low-confidence fields are repaired upstream by vision_corrector (GPT-4o re-reads
them from the source image) before validation runs; fields it verified are passed
in via `trusted_fields` so this module does not re-flag them as low-confidence.

All data models live in schema.py; this module is validation logic only.
"""

import re
from typing import Any, Literal

from shared.logger import get_logger, hash_id
from part1.backend.schema import FieldStatus, ValidationResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate(
    extracted: dict[str, Any],
    ocr_result=None,   # OCRResult | None
    trusted_fields: set[str] | None = None,
) -> ValidationResult:
    """
    Run both validation signals; return a merged ValidationResult.

    Parameters
    ----------
    extracted:      dict output from extractor.extract_fields()
    ocr_result:     OCRResult from ocr_client.analyze_document() (or None)
    trusted_fields: dotted paths the vision corrector re-read from the source
                    image; these are not flagged as low-OCR-confidence (the value
                    no longer comes from the low-confidence OCR read).
    """
    statuses: dict[str, FieldStatus] = {}
    trusted = trusted_fields or set()

    # 1. Deterministic (primary — cannot be overridden)
    _run_deterministic(extracted, statuses)

    # 2. OCR confidence (supporting)
    if ocr_result is not None:
        _run_ocr_confidence(extracted, ocr_result, statuses, trusted)

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


def deterministic_statuses(extracted: dict[str, Any]) -> dict[str, FieldStatus]:
    """Run ONLY the deterministic checks (no OCR / image signals) and return the
    per-field statuses. Used by the vision corrector to decide which fields fail
    validation and therefore need re-reading from the source image."""
    statuses: dict[str, FieldStatus] = {}
    _run_deterministic(extracted, statuses)
    return statuses


def failing_fields(extracted: dict[str, Any]) -> set[str]:
    """Dotted paths whose value does not pass deterministic validation
    (status 'invalid' or 'uncertain')."""
    return {
        field for field, status in deterministic_statuses(extracted).items()
        if status.status in ("invalid", "uncertain")
    }


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


def _run_ocr_confidence(
    extracted: dict, ocr_result, statuses: dict, trusted: set[str]
) -> None:
    avg_conf: float = ocr_result.avg_confidence

    if avg_conf < _OCR_LOW_THRESHOLD:
        logger.warning("Low overall OCR confidence", extra={"avg_confidence": avg_conf})
        _flag_low_ocr_string_fields(extracted, statuses, avg_conf, trusted)

    # Per-value word lookup: flag fields whose extracted words have low confidence
    word_conf: dict[str, float] = {
        w.lower().strip(".,;:\"'()") : c
        for w, c in ocr_result.word_confidences
    }
    addr = extracted.get("address") or {}
    per_word = {
        "lastName": extracted.get("lastName", ""),
        "firstName": extracted.get("firstName", ""),
        "idNumber": extracted.get("idNumber", ""),
        "jobType": extracted.get("jobType", ""),
        "address.city": addr.get("city", ""),
        "address.street": addr.get("street", ""),
    }
    for field_name, value in per_word.items():
        if field_name in trusted:
            continue
        _check_field_word_confidence(field_name, value, word_conf, statuses)

    # Fields the vision corrector verified from the source image are no longer a
    # low-confidence OCR read. Record that provenance only where deterministic
    # validation has no verdict of its own — a deterministic 'invalid' OR
    # 'uncertain' on the re-read value must stay visible (the value still fails
    # validation even though it came from the image).
    for field_name in trusted:
        if field_name not in statuses:
            statuses[field_name] = FieldStatus("ok", "Verified from form image")


def _flag_low_ocr_string_fields(
    extracted: dict, statuses: dict, avg_conf: float, trusted: set[str]
) -> None:
    affected = [
        "lastName", "firstName", "idNumber", "gender", "jobType",
        "timeOfInjury", "accidentLocation", "accidentAddress",
        "accidentDescription", "injuredBodyPart",
    ]
    reason = f"Low overall OCR confidence ({avg_conf:.2f})"
    for f in affected:
        if f in trusted:
            continue
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
