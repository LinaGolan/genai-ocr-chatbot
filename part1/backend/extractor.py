from __future__ import annotations

"""
GPT-4o field extraction: OCR markdown → FormExtraction dict.

Strategy:
  1. Try Azure OpenAI Structured Outputs (json_schema, strict=True) for
     guaranteed schema conformance (requires API version 2024-08-01-preview+).
  2. Fall back to response_format={"type":"json_object"} on older API versions.

The Pydantic FormExtraction model in schema.py is the single schema source
of truth — the JSON schema submitted to the API is derived from it at runtime.
"""

import json
import os
import re
import time
from typing import Any

from openai import BadRequestError

from shared.azure_client import openai_client, GPT4O_DEPLOYMENT, GPT4O_MINI_DEPLOYMENT

# Extraction model is overridable for cost/accuracy A/B testing via the harness.
# EXTRACTION_MODEL=mini selects the cheaper deployment; anything else keeps 4o.
_EXTRACTION_DEPLOYMENT = (
    GPT4O_MINI_DEPLOYMENT if os.getenv("EXTRACTION_MODEL", "").lower() == "mini"
    else GPT4O_DEPLOYMENT
)
from shared.logger import get_logger
from part1.backend.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE,
    FEW_SHOT_HEBREW,
    FEW_SHOT_ENGLISH,
)
from part1.backend.schema import FormExtraction

logger = get_logger(__name__)

_TOKEN_WARN_CHARS = 80_000  # ~20k tokens — warn but continue


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(ocr_text: str) -> str:
    """
    Classify the fill language of a BL283 form as 'english' or 'hebrew'.

    The form template is always in Hebrew, so the OCR always contains Hebrew
    chars. English-filled forms additionally have Latin alphabetic chars in the
    value fields. A Latin ratio above 12% of all alphabetic chars reliably
    separates the two cases.
    """
    latin = sum(1 for c in ocr_text if "A" <= c <= "Z" or "a" <= c <= "z")
    hebrew = sum(1 for c in ocr_text if "א" <= c <= "ת")
    total = latin + hebrew
    if total == 0:
        return "hebrew"
    return "english" if latin / total > 0.12 else "hebrew"


# ---------------------------------------------------------------------------
# Message assembly
# ---------------------------------------------------------------------------

def _build_extraction_messages(ocr_text: str, language: str) -> list[dict]:
    """
    Build the full messages list for the extraction call:
      [system, few-shot user/assistant pairs..., real user request]

    Language-matched few-shot pairs are injected as real chat turns so the
    model sees concrete input→output examples before the actual form.
    """
    snippets = FEW_SHOT_ENGLISH if language == "english" else FEW_SHOT_HEBREW

    messages: list[dict] = [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT}]
    for ocr_snippet, expected_json in snippets:
        messages.append({
            "role": "user",
            "content": EXTRACTION_USER_PROMPT_TEMPLATE.format(ocr_text=ocr_snippet),
        })
        messages.append({"role": "assistant", "content": expected_json})

    messages.append({
        "role": "user",
        "content": EXTRACTION_USER_PROMPT_TEMPLATE.format(ocr_text=ocr_text),
    })
    return messages


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

    language = detect_language(ocr_markdown)
    logger.info("Form language detected", extra={"language": language})
    messages = _build_extraction_messages(ocr_markdown, language)

    t0 = time.perf_counter()
    raw = _try_structured_outputs(messages) or _call_json_object(messages)
    elapsed = time.perf_counter() - t0

    # Merge with defaults so all keys are present, then validate shape
    complete = _merge_with_empty(raw)
    _pad_id_number(complete)
    _fix_mobile_phone(complete)
    _fix_landline_phone(complete)
    parsed = FormExtraction.model_validate(complete)
    result = parsed.model_dump()

    empty_count = _count_empty(result)
    logger.info(
        "Extraction complete",
        extra={
            "language": language,
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
            model=_EXTRACTION_DEPLOYMENT,
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
    Derive a JSON Schema from the Pydantic model and patch it for strict mode.
    """
    schema = FormExtraction.model_json_schema()
    _patch_for_strict_mode(schema)
    return schema


def _patch_for_strict_mode(node: dict) -> None:
    """Patch a JSON Schema node for Azure Strict Structured Outputs:
    - Every object: required = all property keys, additionalProperties = false
    - Strip default values (strict mode rejects them)
    All three are required; missing any one causes a BadRequestError.
    """
    node.pop("default", None)
    if node.get("type") == "object":
        props = node.get("properties", {})
        node["required"] = list(props.keys())
        node["additionalProperties"] = False
        for prop_schema in props.values():
            _patch_for_strict_mode(prop_schema)
    for sub in node.get("$defs", {}).values():
        _patch_for_strict_mode(sub)
    for item in node.get("allOf", []):
        _patch_for_strict_mode(item)


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
    """Strip non-digits; left-pad to 9 if fewer than 9 digits (OCR may drop leading zero).
    10+ digit strings are preserved as-is so the validator can flag the length error."""
    digits = re.sub(r"\D", "", str(data.get("idNumber") or ""))
    if not digits:
        return
    data["idNumber"] = digits if len(digits) >= 9 else digits.zfill(9)


def _fix_landline_phone(data: dict) -> None:
    """Strip non-digit characters from landlinePhone; preserve digit content for validation."""
    raw = str(data.get("landlinePhone") or "")
    digits = re.sub(r"\D", "", raw)
    if digits:
        data["landlinePhone"] = digits


def _fix_mobile_phone(data: dict) -> None:
    """Strip non-digit characters from mobilePhone; preserve digit content for validation."""
    raw = str(data.get("mobilePhone") or "")
    digits = re.sub(r"\D", "", raw)
    if digits:
        data["mobilePhone"] = digits


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
