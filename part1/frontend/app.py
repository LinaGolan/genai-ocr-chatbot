from __future__ import annotations

# Part 1 — Streamlit UI for BL283 form field extraction.
# Layout: left = colour-coded extracted fields, right = raw JSON, bottom = OCR (collapsed).

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from typing import Any

import streamlit as st

from part1.backend.ocr_client import analyze_document, OCRResult
from part1.backend.extractor import extract_fields
from part1.backend.vision_corrector import correct_and_validate
from part1.backend.schema import ValidationResult, FieldStatus

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BL283 Form Extractor",
    page_icon="📋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

_STATUS_BG     = {"ok": "#f1f8f1", "uncertain": "#fffbf0", "invalid": "#fff5f5"}
_STATUS_BORDER = {"ok": "#43a047", "uncertain": "#fb8c00", "invalid": "#e53935"}
_STATUS_BADGE  = {"ok": "#43a047", "uncertain": "#fb8c00", "invalid": "#e53935"}
_STATUS_LABEL  = {"ok": "OK",      "uncertain": "CHECK",   "invalid": "ERROR"}
_STATUS_ICON   = {"ok": "✅",       "uncertain": "⚠️",      "invalid": "❌"}

# Maps the (lowercase) start of a date-ordering reason to the two field keys involved.
_DATE_COMPARISON_REASONS: dict[str, tuple[str, str, str, str]] = {
    "date of birth is not before date of injury": (
        "dateOfBirth", "Date of Birth", "dateOfInjury", "Date of Injury"),
    "date of injury is after form filling date": (
        "dateOfInjury", "Date of Injury", "formFillingDate", "Form Filling Date"),
    "form receipt date at clinic is before date of injury": (
        "formReceiptDateAtClinic", "Receipt at Clinic", "dateOfInjury", "Date of Injury"),
    "form receipt date at clinic is before form filling date": (
        "formReceiptDateAtClinic", "Receipt at Clinic", "formFillingDate", "Form Filling Date"),
}

_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container { padding-top: 2rem !important; padding-bottom: 1rem !important; }
div[data-testid="column"] { overflow: visible !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cfd8dc; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #b0bec5; }

code { font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important; }

details > summary { font-weight: 600; font-size: 0.9rem; letter-spacing: 0.2px; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
"""


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------

def _hero_html(title: str, subtitle: str) -> str:
    return (
        '<div style="padding:8px 2px 12px">'
        '<div style="display:flex;align-items:flex-start;gap:14px">'
        '<div style="width:4px;min-height:52px;background:linear-gradient(180deg,#1976d2 0%,#42a5f5 100%);'
        'border-radius:3px;flex-shrink:0"></div>'
        '<div>'
        f'<div style="font-size:1.45rem;font-weight:700;color:#1a237e;letter-spacing:-0.3px;line-height:1.2">{title}</div>'
        f'<p style="margin:6px 0 0;color:#607d8b;font-size:0.84rem;line-height:1.5">{subtitle}</p>'
        '</div>'
        '</div>'
        '</div>'
    )


def _metric_cards_html(
    completeness: float,
    total_n: int,
    ok_n: int,
    invalid_n: int,
    uncertain_n: int,
) -> str:
    def _std_card(color: str, value: str, label: str) -> str:
        return (
            f'<div style="flex:1;background:white;border:1px solid #e8eaf6;border-radius:12px;'
            f'padding:16px 20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.07)">'
            f'<div style="font-size:1.5rem;font-weight:700;color:{color};letter-spacing:-0.5px">{value}</div>'
            f'<div style="font-size:0.7rem;color:#90a4ae;text-transform:uppercase;'
            f'letter-spacing:0.7px;margin-top:3px">{label}</div>'
            f'</div>'
        )

    if not invalid_n and not uncertain_n:
        issues_inner = (
            '<div style="font-size:1.5rem;font-weight:700;color:#2e7d32">✓</div>'
            '<div style="font-size:0.7rem;color:#90a4ae;text-transform:uppercase;'
            'letter-spacing:0.7px;margin-top:3px">No Issues</div>'
        )
    else:
        rows = ""
        if invalid_n:
            rows += (
                f'<div style="font-size:1.05rem;font-weight:700;color:#c62828;line-height:1.4">'
                f'{invalid_n} invalid</div>'
            )
        if uncertain_n:
            rows += (
                f'<div style="font-size:1.05rem;font-weight:700;color:#e65100;line-height:1.4">'
                f'{uncertain_n} uncertain</div>'
            )
        issues_inner = (
            rows +
            '<div style="font-size:0.7rem;color:#90a4ae;text-transform:uppercase;'
            'letter-spacing:0.7px;margin-top:4px">Issues</div>'
        )

    issues_card = (
        '<div style="flex:1;background:white;border:1px solid #e8eaf6;border-radius:12px;'
        'padding:16px 20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.07)">'
        + issues_inner
        + '</div>'
    )

    items = (
        _std_card("#1565c0", f"{completeness:.0%}", "Completeness")
        + _std_card("#37474f", f"{ok_n} / {total_n}", "Fields Valid")
        + issues_card
    )
    return f'<div style="display:flex;gap:12px;margin:10px 0 14px">{items}</div>'


def _section_header(title: str) -> str:
    return (
        '<div style="display:flex;align-items:center;gap:9px;margin:20px 0 6px;'
        'padding-bottom:5px;border-bottom:1.5px solid #eceff1">'
        '<div style="width:3px;height:15px;background:#1976d2;border-radius:2px;flex-shrink:0"></div>'
        f'<span style="font-size:0.77rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.9px;color:#546e7a">{title}</span>'
        '</div>'
    )


def _field_html(label: str, value: str, field_key: str, statuses: dict[str, FieldStatus]) -> str:
    status_obj = statuses.get(field_key, FieldStatus("ok"))
    status     = status_obj.status
    bg         = _STATUS_BG.get(status, "#fafafa")
    border     = _STATUS_BORDER.get(status, "#90a4ae")
    tooltip    = f' title="{status_obj.reason}"' if status_obj.reason else ""

    val_html = (
        f'<span style="font-family:monospace;font-size:0.9rem;color:#1a237e;word-break:break-all">{value}</span>'
        if value else
        '<span style="color:#b0bec5;font-style:italic;font-size:0.88rem">—</span>'
    )

    badge_html = ""
    reason_html = ""
    if status != "ok":
        badge_bg   = _STATUS_BADGE.get(status, "#607d8b")
        badge_text = _STATUS_LABEL.get(status, status.upper())
        badge_html = (
            f'<span style="flex-shrink:0;margin-left:10px;background:{badge_bg};color:white;'
            f'font-size:0.62rem;font-weight:700;padding:2px 8px;border-radius:10px;'
            f'letter-spacing:0.5px;white-space:nowrap">{badge_text}</span>'
        )
        # (a) Surface the issue inline — visible without hovering the tooltip.
        if status_obj.reason:
            reason_html = (
                f'<div style="font-size:0.7rem;color:{badge_bg};margin-top:3px;'
                f'font-style:italic;word-break:break-word">⚠ {status_obj.reason}</div>'
            )

    return (
        f'<div style="display:flex;align-items:center;background:{bg};'
        f'border-left:4px solid {border};padding:8px 14px;margin:3px 0;'
        f'border-radius:0 8px 8px 0;box-shadow:0 1px 3px rgba(0,0,0,0.04)"{tooltip}>'
        f'<div style="flex:1;min-width:0">'
        f'<div style="font-size:0.67rem;color:#78909c;text-transform:uppercase;'
        f'letter-spacing:0.6px;font-weight:600;margin-bottom:2px">{label}</div>'
        f'{val_html}'
        f'{reason_html}'
        f'</div>'
        f'{badge_html}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Composite renderers
# ---------------------------------------------------------------------------

def _render_summary(val_result: ValidationResult) -> None:
    invalid_n   = sum(1 for s in val_result.fields.values() if s.status == "invalid")
    uncertain_n = sum(1 for s in val_result.fields.values() if s.status == "uncertain")
    total_n     = len(val_result.fields)
    ok_n        = total_n - invalid_n - uncertain_n
    st.markdown(
        _metric_cards_html(val_result.completeness, total_n, ok_n, invalid_n, uncertain_n),
        unsafe_allow_html=True,
    )


def _render_json_panel(extracted: dict[str, Any], val_result: ValidationResult) -> None:
    statuses   = val_result.fields
    html_parts: list[str] = []

    def field(label: str, value: str, key: str) -> None:
        html_parts.append(_field_html(label, value, key, statuses))

    def date_block(label: str, key: str) -> None:
        d   = extracted.get(key) or {}
        val = "/".join(filter(None, [d.get("day"), d.get("month"), d.get("year")]))
        html_parts.append(_field_html(label, val, key, statuses))

    html_parts.append(_section_header("👤 Personal Information"))
    field("Last Name / שם משפחה",      extracted.get("lastName", ""),  "lastName")
    field("First Name / שם פרטי",      extracted.get("firstName", ""), "firstName")
    field("ID Number / מספר זהות",     extracted.get("idNumber", ""),  "idNumber")
    field("Gender / מין",              extracted.get("gender", ""),    "gender")
    date_block("Date of Birth / תאריך לידה", "dateOfBirth")

    html_parts.append(_section_header("🏠 Address / כתובת"))
    addr = extracted.get("address") or {}
    field("Street / רחוב",          addr.get("street", ""),      "address.street")
    field("House No. / מספר בית",   addr.get("houseNumber", ""), "address.houseNumber")
    field("Entrance / כניסה",       addr.get("entrance", ""),    "address.entrance")
    field("Apartment / דירה",       addr.get("apartment", ""),   "address.apartment")
    field("City / ישוב",            addr.get("city", ""),        "address.city")
    field("Postal Code / מיקוד",    addr.get("postalCode", ""),  "address.postalCode")
    field("P.O. Box / תא דואר",     addr.get("poBox", ""),       "address.poBox")

    html_parts.append(_section_header("📞 Contact"))
    field("Landline / טלפון קווי", extracted.get("landlinePhone", ""), "landlinePhone")
    field("Mobile / טלפון נייד",   extracted.get("mobilePhone", ""),   "mobilePhone")

    html_parts.append(_section_header("💼 Employment"))
    field("Job Type / סוג העבודה", extracted.get("jobType", ""), "jobType")

    html_parts.append(_section_header("⚕️ Incident / פגיעה"))
    date_block("Date of Injury / תאריך הפגיעה",    "dateOfInjury")
    field("Time of Injury / שעת הפגיעה",           extracted.get("timeOfInjury", ""),       "timeOfInjury")
    field("Accident Location / מקום התאונה",        extracted.get("accidentLocation", ""),   "accidentLocation")
    field("Accident Address / כתובת מקום התאונה",  extracted.get("accidentAddress", ""),    "accidentAddress")
    field("Description / תיאור התאונה",            extracted.get("accidentDescription", ""),"accidentDescription")
    field("Injured Body Part / האיבר שנפגע",       extracted.get("injuredBodyPart", ""),    "injuredBodyPart")

    html_parts.append(_section_header("📝 Submission"))
    field("Signature / חתימה", extracted.get("signature", ""), "signature")
    date_block("Form Filling Date / תאריך מילוי הטופס",       "formFillingDate")
    date_block("Clinic Receipt Date / תאריך קבלת הטופס בקופה","formReceiptDateAtClinic")

    html_parts.append(_section_header("🏥 Medical Institution"))
    med = extracted.get("medicalInstitutionFields") or {}
    field("Health Fund Member / חבר בקופת חולים", med.get("healthFundMember", ""),  "medicalInstitutionFields.healthFundMember")
    field("Nature of Accident / מהות התאונה",     med.get("natureOfAccident", ""),  "medicalInstitutionFields.natureOfAccident")
    field("Medical Diagnoses / אבחנות רפואיות",   med.get("medicalDiagnoses", ""),  "medicalInstitutionFields.medicalDiagnoses")

    st.markdown("\n".join(html_parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Validation proof cards
# ---------------------------------------------------------------------------

def _get_field_value(extracted: dict, field_key: str) -> str:
    """Resolve a dotted field key (e.g. 'address.city') to its extracted value."""
    val: Any = extracted
    for part in field_key.split("."):
        if isinstance(val, dict):
            val = val.get(part, "")
        else:
            return ""
    if isinstance(val, dict):
        # date sub-object — format as DD/MM/YYYY
        return "/".join(filter(None, [val.get("day"), val.get("month"), val.get("year")]))
    return str(val) if val else ""


_DETERMINISTIC_PREFIXES = (
    "id must be", "invalid day", "invalid month", "implausible year",
    "unexpected israeli phone", "postal code should", "unexpected gender",
    "date of birth is not", "date of injury is after", "non-numeric date",
    "date is in the future", "form receipt date at clinic is before",
    "unexpected time-of-injury", "id number is empty", "last name is empty",
    "first name is empty",
)


def _validation_source(reason: str) -> tuple[str, str]:
    """Return (icon, label) describing which signal produced the reason."""
    r = reason.lower()
    # Deterministic verdict wins, even when a vision re-read note is appended.
    if any(r.startswith(p) for p in _DETERMINISTIC_PREFIXES):
        return "🔢", "Deterministic check"
    if "ocr confidence" in r:
        return "📡", "OCR confidence signal"
    if "form image" in r:
        return "🖼️", "Vision re-read from image"
    return "✅", "Passed checks"


def _date_pair_html(reason: str, extracted: dict) -> str:
    """Return a small comparison block showing both conflicting dates, or '' if not applicable."""
    r = reason.lower()
    for prefix, (key_a, lbl_a, key_b, lbl_b) in _DATE_COMPARISON_REASONS.items():
        if r.startswith(prefix):
            def _fmt(d: Any) -> str:
                if not d or not isinstance(d, dict):
                    return "—"
                return "/".join(filter(None, [d.get("day"), d.get("month"), d.get("year")])) or "—"
            val_a = _fmt(extracted.get(key_a))
            val_b = _fmt(extracted.get(key_b))
            return (
                f'<div style="margin:6px 0 6px;padding:7px 12px;background:#fff8e1;'
                f'border-left:3px solid #fb8c00;border-radius:0 6px 6px 0;font-size:0.78rem;'
                f'display:flex;gap:10px;flex-wrap:wrap;align-items:center">'
                f'<span><span style="color:#607d8b;font-weight:600">{lbl_a}:</span>&nbsp;'
                f'<code style="color:#c62828">{val_a}</code></span>'
                f'<span style="color:#b0bec5">→</span>'
                f'<span><span style="color:#607d8b;font-weight:600">{lbl_b}:</span>&nbsp;'
                f'<code style="color:#1a237e">{val_b}</code></span>'
                f'</div>'
            )
    return ""


def _proof_card_html(
    fname: str,
    value: str,
    fstatus: FieldStatus,
    extracted: dict,
) -> str:
    status     = fstatus.status
    border     = _STATUS_BORDER.get(status, "#90a4ae")
    header_bg  = _STATUS_BG.get(status, "#fafafa")
    badge_bg   = _STATUS_BADGE.get(status, "#607d8b")
    badge_text = _STATUS_LABEL.get(status, status.upper())
    icon       = _STATUS_ICON.get(status, "")
    src_icon, src_label = _validation_source(fstatus.reason)

    # ── Extracted value column ───────────────────────────────────────────────
    if value:
        val_display = (
            f'<span style="font-family:monospace;font-size:0.88rem;color:#1a237e;'
            f'background:#e8eaf6;padding:6px 12px;border-radius:6px;'
            f'display:inline-block;word-break:break-all;max-width:100%">{value}</span>'
        )
    else:
        val_display = '<span style="color:#b0bec5;font-style:italic;font-size:0.85rem">— empty —</span>'

    # ── Validation column ────────────────────────────────────────────────────
    reason_text = fstatus.reason or "Passed all checks"
    date_pair   = _date_pair_html(fstatus.reason or "", extracted)
    check_html = (
        f'<div style="margin-bottom:6px">'
        f'<span style="background:{badge_bg};color:white;font-size:0.63rem;font-weight:700;'
        f'padding:2px 9px;border-radius:9px;letter-spacing:0.4px">{badge_text}</span>'
        f'</div>'
        f'<div style="font-size:0.8rem;color:#546e7a;line-height:1.45;margin-bottom:4px">'
        f'{reason_text}</div>'
        f'{date_pair}'
        f'<div style="font-size:0.7rem;color:#90a4ae">'
        f'{src_icon}&nbsp;{src_label}</div>'
    )

    col_style = 'style="background:white;padding:14px 18px;min-width:0"'
    label_style = (
        'style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
        'letter-spacing:0.7px;color:#90a4ae;margin-bottom:8px"'
    )

    return (
        f'<div style="border:1px solid {border};border-radius:10px;margin:10px 0;'
        f'overflow:hidden;box-shadow:0 2px 6px rgba(0,0,0,0.06)">'

        # header
        f'<div style="background:{header_bg};border-bottom:1px solid {border};'
        f'padding:9px 16px;display:flex;align-items:center;justify-content:space-between">'
        f'<span style="font-weight:700;font-size:0.9rem;font-family:monospace;color:#263238">'
        f'{icon} {fname}</span>'
        f'<span style="background:{badge_bg};color:white;font-size:0.63rem;font-weight:700;'
        f'padding:3px 10px;border-radius:10px;letter-spacing:0.5px">{badge_text}</span>'
        f'</div>'

        # 2-column grid
        f'<div style="display:grid;grid-template-columns:1fr 1fr;background:#f5f5f5;gap:1px">'

        f'<div {col_style}>'
        f'<div {label_style}>Extracted Value</div>'
        f'<div style="line-height:1.6">{val_display}</div>'
        f'</div>'

        f'<div {col_style}>'
        f'<div {label_style}>Validation</div>'
        f'{check_html}'
        f'</div>'

        f'</div>'  # grid
        f'</div>'  # card
    )


def _render_validation_details(
    val_result: ValidationResult,
    extracted: dict,
) -> None:
    if not val_result.fields:
        st.write("No fields were specifically validated.")
        return

    issues = {k: v for k, v in val_result.fields.items() if v.status != "ok"}
    if not issues:
        st.success("All validated fields passed — no issues found.")
        return

    sorted_fields = sorted(
        issues.items(),
        key=lambda kv: (0 if kv[1].status == "invalid" else 1, kv[0]),
    )

    cards_html = "\n".join(
        _proof_card_html(fname, _get_field_value(extracted, fname), fstatus, extracted)
        for fname, fstatus in sorted_fields
    )
    st.markdown(cards_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline helper (cached per file bytes)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _run_pipeline(file_bytes: bytes, filename: str) -> tuple[OCRResult, dict, dict]:
    ocr       = analyze_document(file_bytes, filename)
    extracted = extract_fields(ocr.markdown)
    # Re-read low-confidence / validation-failing fields from the source image
    # (GPT-4o vision), then validate the corrected record.
    extracted, val = correct_and_validate(extracted, ocr, file_bytes, filename)
    return ocr, extracted, val.to_dict()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    # ── Header: title left, uploader right ──────────────────────────────────
    hdr_left, hdr_right = st.columns([3, 2], gap="large")

    with hdr_left:
        st.markdown(
            _hero_html(
                "📋 BL283 Form Field Extractor",
                "Upload a ביטוח לאומי BL283 form (PDF or JPG). "
                "Azure Document Intelligence + GPT-4o extract and validate all fields.",
            ),
            unsafe_allow_html=True,
        )

    with hdr_right:
        st.markdown(
            '<div style="padding-top:10px;font-size:0.82rem;font-weight:600;'
            'color:#546e7a;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:4px">'
            '📂 Upload BL283 Form</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload BL283 form",
            type=["pdf", "jpg", "jpeg"],
            help="Accepted: PDF, JPG — max 50 MB",
            label_visibility="collapsed",
        )

    if uploaded is None:
        return

    file_bytes = uploaded.read()

    try:
        with st.spinner("Running OCR and field extraction… this may take 20–40 seconds."):
            ocr_result, extracted, val_dict = _run_pipeline(file_bytes, uploaded.name)
    except ValueError as exc:
        st.error(f"**Processing failed:** {exc}")
        return
    except Exception as exc:
        st.error(f"**Unexpected error:** {exc}")
        st.exception(exc)
        return

    val_result = ValidationResult(
        fields={
            k: FieldStatus(v["status"], v.get("reason", ""))
            for k, v in val_dict["fields"].items()
        },
        completeness=val_dict["completeness"],
        accuracy_estimate=val_dict["accuracy_estimate"],
    )

    st.success(
        f"Processed **{uploaded.name}** — {ocr_result.page_count} page(s), "
        f"{len(ocr_result.markdown):,} characters extracted."
    )

    # ── Two-panel layout ────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🗂️ Extracted Fields")
        _render_summary(val_result)
        _render_json_panel(extracted, val_result)

    with col_right:
        st.subheader("{ } Raw JSON")
        st.code(
            json.dumps(extracted, ensure_ascii=False, indent=2),
            language="json",
        )
        with st.expander("🔍 Validation details"):
            _render_validation_details(val_result, extracted)

    # ── OCR output (collapsed) ───────────────────────────────────────────────
    with st.expander("📄 OCR Output (Document Intelligence)"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Confidence", f"{ocr_result.avg_confidence:.3f}")
        c2.metric("Min Confidence", f"{ocr_result.min_confidence:.3f}")
        c3.metric("Words Analysed", len(ocr_result.word_confidences))
        st.text_area(
            label="Raw OCR markdown",
            value=ocr_result.markdown,
            height=400,
            label_visibility="collapsed",
        )


if __name__ == "__main__":
    main()
