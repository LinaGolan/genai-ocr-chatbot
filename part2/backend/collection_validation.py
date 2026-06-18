from __future__ import annotations

"""
Deterministic, code-side validation of the collected fields.

This is the backstop that lets GPT-4o Mini — not GPT-4o — drive the collection
conversation. Mini converses well but is unreliable at the one thing the
assignment makes mandatory: counting digits (the 9-digit ID / card checks) and
range/enum checks. So we take that judgement away from the model entirely.

The model emits a state block on EVERY turn listing what it has collected so far
(empty string for anything not yet given). This module validates that block in
plain Python every turn:

  - ``field_errors``   — format problems in the fields that ARE filled in. The
                         per-turn signal: if a value is wrong, we hand the error
                         back to the model so it re-asks. Empty fields are "not
                         collected yet", never an error.
  - ``missing_fields`` — which of the 8 required fields are still empty.

Collection is complete (advance to confirmation) only when there are no missing
fields AND no field errors. The model never counts digits and never decides
completion — code does both.

This does NOT make collection a hardcoded form: the LLM still decides what to
ask, in what order, how to phrase it, and in which language. Code only verifies
the values it reports and decides when they are all valid.
"""

import re

# Canonical Hebrew enum values (the state block must use these — the prompt
# instructs the model to map English/other phrasings to them before recording).
VALID_HMOS = ("מכבי", "מאוחדת", "כללית")
VALID_TIERS = ("זהב", "כסף", "ארד")

REQUIRED_FIELDS = (
    "firstName", "lastName", "idNumber", "gender",
    "age", "hmo", "hmoCardNumber", "insuranceTier",
)

_NINE_DIGITS_RE = re.compile(r"^\d{9}$")


def field_errors(data: dict) -> list[str]:
    """
    Validate only the fields that are present (non-empty) in the state block.
    An empty field means "not collected yet" and is NOT an error here.

    Returns human-readable English error strings (one per failing field), written
    for the collection model to relay to the user. Empty list = nothing filled in
    so far is invalid.
    """
    errors: list[str] = []

    id_number = str(data.get("idNumber", "")).strip()
    if id_number and not _NINE_DIGITS_RE.match(id_number):
        errors.append(
            f"idNumber: must be exactly 9 digits (received {_describe(id_number)})."
        )

    card_number = str(data.get("hmoCardNumber", "")).strip()
    if card_number and not _NINE_DIGITS_RE.match(card_number):
        errors.append(
            f"hmoCardNumber: must be exactly 9 digits (received {_describe(card_number)})."
        )

    age = data.get("age")
    if age not in (None, ""):
        age_error = _validate_age(age)
        if age_error:
            errors.append(age_error)

    hmo = str(data.get("hmo", "")).strip()
    if hmo and hmo not in VALID_HMOS:
        errors.append(f"hmo: must be one of {' / '.join(VALID_HMOS)} (received '{hmo}').")

    tier = str(data.get("insuranceTier", "")).strip()
    if tier and tier not in VALID_TIERS:
        errors.append(
            f"insuranceTier: must be one of {' / '.join(VALID_TIERS)} (received '{tier}')."
        )

    return errors


def missing_fields(data: dict) -> list[str]:
    """Which of the 8 required fields are still empty (not yet collected)."""
    return [f for f in REQUIRED_FIELDS if not str(data.get(f, "")).strip()]


def is_complete_and_valid(data: dict) -> bool:
    """True when every required field is filled and every value passes its check."""
    return not missing_fields(data) and not field_errors(data)


def _validate_age(age) -> str | None:
    """Age must be a whole number 0–120. Accepts an int or a digit string."""
    if isinstance(age, bool):  # bool is an int subclass — reject explicitly
        return "age: must be a whole number between 0 and 120."
    if isinstance(age, int):
        value = age
    else:
        text = str(age).strip()
        if not text.isdigit():
            return "age: must be a whole number between 0 and 120."
        value = int(text)
    if not (0 <= value <= 120):
        return f"age: must be between 0 and 120 (received {value})."
    return None


def _describe(value: str) -> str:
    """A short, PII-free description of a bad ID/card value for the error text."""
    if not value:
        return "an empty value"
    digits = sum(c.isdigit() for c in value)
    if digits == len(value):
        return f"{len(value)} digits"
    return f"{len(value)} characters, including non-digits"


def format_validation_feedback(errors: list[str]) -> str:
    """
    Turn the deterministic errors into a system instruction for the collection
    model. The model — not this code — decides how to re-ask; we only state which
    fields are wrong and forbid completing until they are fixed.
    """
    bullet_list = "\n".join(f"- {e}" for e in errors)
    return (
        "AUTOMATED VALIDATION found problems with the value(s) the user just gave. "
        "You did NOT detect these yourself and must trust this validator over your "
        "own judgement (you cannot reliably count digits):\n"
        f"{bullet_list}\n\n"
        "Ask the user to correct ONLY the field(s) listed above, conversationally "
        "and in the user's language. Keep every other value you already have — it "
        "is fine. Do not claim any other field is wrong."
    )
