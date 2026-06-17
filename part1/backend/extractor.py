from __future__ import annotations

"""
GPT-4o field extraction: OCR markdown → FormExtraction dict.

Strategy:
  1. Try Azure OpenAI Structured Outputs (json_schema, strict=True) for
     guaranteed schema conformance (requires API version 2024-08-01-preview+).
  2. Fall back to response_format={"type":"json_object"} on older API versions.

The Pydantic FormExtraction model in validator.py is the single schema source
of truth — the JSON schema submitted to the API is derived from it at runtime.
"""

import json
import re
import time
from typing import Any

from openai import BadRequestError

from shared.azure_client import openai_client, GPT4O_DEPLOYMENT
from shared.logger import get_logger
from part1.backend.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE,
)
from part1.backend.validator import FormExtraction

logger = get_logger(__name__)

_TOKEN_WARN_CHARS = 80_000  # ~20k tokens — warn but continue


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_fields(ocr_markdown: str) -> dict[str, Any]:
    """
    Extract BL283 form fields from OCR markdown using GPT-4o.

    Returns a dict conforming to the FormExtraction schema with all keys
    present and string values (never null).
    Raises ValueError on unrecoverable extraction failure.
    """
    if len(ocr_markdown) > _TOKEN_WARN_CHARS:
        logger.warning(
            "OCR text is very long — may approach token budget",
            extra={"chars": len(ocr_markdown)},
        )

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": EXTRACTION_USER_PROMPT_TEMPLATE.format(ocr_text=ocr_markdown),
        },
    ]

    t0 = time.perf_counter()
    raw = _try_structured_outputs(messages) or _call_json_object(messages)
    elapsed = time.perf_counter() - t0

    # Merge with defaults so all keys are present, then validate shape
    complete = _merge_with_empty(raw)
    _pad_id_number(complete)
    _fix_date_fields(complete)
    parsed = FormExtraction.model_validate(complete)
    result = parsed.model_dump()

    empty_count = _count_empty(result)
    logger.info(
        "Extraction complete",
        extra={
            "empty_fields": empty_count,
            "latency_s": round(elapsed, 2),
            "mode": "structured_outputs" if raw is not _SENTINEL else "json_object",
        },
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()  # used only for the logger mode label


def _try_structured_outputs(messages: list[dict]) -> dict | None:
    """
    Attempt a Structured Outputs call (json_schema).
    Returns the parsed dict on success, None if the API version doesn't support it.
    Re-raises unexpected errors.
    """
    try:
        schema = _build_strict_schema()
        response = openai_client.chat.completions.create(
            model=GPT4O_DEPLOYMENT,
            messages=messages,
            temperature=0,
            seed=42,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "FormExtraction",
                    "description": "Extracted fields from BL283 form",
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        content = response.choices[0].message.content
        logger.info("Extraction used Structured Outputs (json_schema)")
        return json.loads(content)
    except BadRequestError as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("json_schema", "response_format", "structured", "unsupported")):
            logger.info(
                "Structured Outputs not supported by this API version — falling back",
                extra={"hint": str(exc)[:150]},
            )
            return None
        raise
    except Exception as exc:
        # Broad catch: surface as a warning and fall through to json_object
        logger.warning(
            "Structured Outputs call failed unexpectedly — falling back",
            extra={"error": str(exc)[:200]},
        )
        return None


def _call_json_object(messages: list[dict]) -> dict:
    """Standard json_object fallback extraction."""
    response = openai_client.chat.completions.create(
        model=GPT4O_DEPLOYMENT,
        messages=messages,
        temperature=0,
        seed=42,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    logger.info("Extraction used json_object mode")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GPT-4o returned invalid JSON: {exc}") from exc


def _build_strict_schema() -> dict:
    """
    Derive a JSON Schema from the Pydantic model and patch it for strict mode:
    every object gets additionalProperties: false.
    """
    schema = FormExtraction.model_json_schema()
    _add_additional_properties_false(schema)
    return schema


def _add_additional_properties_false(node: dict) -> None:
    """Recursively ensure every object node has additionalProperties: false."""
    if node.get("type") == "object":
        node.setdefault("additionalProperties", False)
        for prop_schema in node.get("properties", {}).values():
            _add_additional_properties_false(prop_schema)
    for sub in node.get("$defs", {}).values():
        _add_additional_properties_false(sub)
    for item in node.get("allOf", []):
        _add_additional_properties_false(item)


_DATE_KEYS = ("dateOfBirth", "dateOfInjury", "formFillingDate", "formReceiptDateAtClinic")


def _fix_date_fields(data: dict) -> None:
    """Correct field misassignments for all date dicts."""
    for key in _DATE_KEYS:
        d = data.get(key)
        if not isinstance(d, dict):
            continue
        _normalize_date(d)


def _normalize_date(d: dict) -> None:
    """
    Fix two detectable reversal patterns:
      1. year/day swap  — 'day' holds a 4-digit year (≥1900) and 'year' holds a day value (≤31)
      2. month/day swap — 'month' holds an impossible month (>12) while 'day' is a valid month (≤12)
    """
    day_s = d.get("day", "")
    month_s = d.get("month", "")
    year_s = d.get("year", "")
    try:
        dv = int(day_s) if day_s else 0
        mv = int(month_s) if month_s else 0
        yv = int(year_s) if year_s else 0
    except (ValueError, TypeError):
        return

    # Pattern 1: year value ended up in 'day', day value in 'year'
    if dv >= 1900 and 1 <= yv <= 31:
        d["day"], d["year"] = year_s, day_s
        dv, yv = yv, dv  # keep locals in sync for pattern 2

    # Pattern 2: day and month swapped (month > 12 is impossible)
    if mv > 12 and 1 <= dv <= 12:
        d["day"], d["month"] = month_s, day_s


def _pad_id_number(data: dict) -> None:
    """Strip non-digits, trim 10→9 (ס״ב branch code), left-pad to 9 digits."""
    digits = re.sub(r"\D", "", str(data.get("idNumber") or ""))
    if not digits:
        return
    if len(digits) == 10:
        digits = digits[:9]  # drop trailing ס״ב branch code digit
    data["idNumber"] = digits.zfill(9)


_EMPTY_EXTRACTION: dict = {
    "lastName": "", "firstName": "", "idNumber": "", "gender": "",
    "dateOfBirth": {"day": "", "month": "", "year": ""},
    "address": {
        "street": "", "houseNumber": "", "entrance": "", "apartment": "",
        "city": "", "postalCode": "", "poBox": "",
    },
    "landlinePhone": "", "mobilePhone": "", "jobType": "",
    "dateOfInjury": {"day": "", "month": "", "year": ""},
    "timeOfInjury": "", "accidentLocation": "", "accidentAddress": "",
    "accidentDescription": "", "injuredBodyPart": "", "signature": "",
    "formFillingDate": {"day": "", "month": "", "year": ""},
    "formReceiptDateAtClinic": {"day": "", "month": "", "year": ""},
    "medicalInstitutionFields": {
        "healthFundMember": "", "natureOfAccident": "", "medicalDiagnoses": "",
    },
}


def _merge_with_empty(data: dict) -> dict:
    """
    Deep-merge *data* into a fully-populated empty template so every
    required key is present even when the model omits some.
    Converts None values to "".
    """
    def _merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = _merge(result[k], v)
            elif v is None:
                result[k] = ""
            else:
                result[k] = v
        return result

    return _merge(_EMPTY_EXTRACTION, data)


def _count_empty(data: Any) -> int:
    if isinstance(data, dict):
        return sum(_count_empty(v) for v in data.values())
    return 1 if data == "" else 0
