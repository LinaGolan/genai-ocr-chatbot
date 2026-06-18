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
_ACCURACY_COLOR = {"high": "#2e7d32", "medium": "#e65100", "low": "#c62828"}
_ACCURACY_ICON  = {"high": "🟢",      "medium": "🟡",       "low": "🔴"}

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
    # Outer padding-top gives the box-shadow room above the card so it isn't clipped.
    return (
        '<div style="padding:10px 2px 2px">'
        '<div style="background:linear-gradient(135deg,#1565c0 0%,#1976d2 55%,#42a5f5 100%);'
        'border-radius:12px;padding:18px 22px 16px;color:white;'
        'box-shadow:0 3px 14px rgba(21,101,192,0.22)">'
        f'<div style="font-size:1.35rem;font-weight:700;letter-spacing:-0.2px;margin-bottom:6px">{title}</div>'
        f'<p style="margin:0;opacity:0.88;font-size:0.85rem;line-height:1.5">{subtitle}</p>'
        '</div>'
        '</div>'
    )


def _metric_cards_html(
    completeness: float,
    total_validated: int,
    invalid_n: int,
    uncertain_n: int,
) -> str:
    issue_color = "#c62828" if invalid_n else ("#e65100" if uncertain_n else "#2e7d32")
    cards = [
        ("#1565c0", f"{completeness:.0%}",              "Completeness"),
        ("#37474f", str(total_validated),               "Fields Validated"),
        (issue_color, f"{invalid_n} ✗ / {uncertain_n} ⚠", "Issues"),
    ]
    items = "".join(
        f'<div style="flex:1;background:white;border:1px solid #e8eaf6;border-radius:12px;'
        f'padding:16px 20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.07)">'
        f'<div style="font-size:1.5rem;font-weight:700;color:{color};letter-spacing:-0.5px">{value}</div>'
        f'<div style="font-size:0.7rem;color:#90a4ae;text-transform:uppercase;'
        f'letter-spacing:0.7px;margin-top:3px">{label}</div>'
        f'</div>'
        for color, value, label in cards
    )
    return f'<div style="display:flex;gap:12px;margin:10px 0 14px">{items}</div>'


def _accuracy_badge_html(acc: str) -> str:
    color = _ACCURACY_COLOR.get(acc, "#607d8b")
    icon  = _ACCURACY_ICON.get(acc, "⚪")
    return (
        f'<div style="display:inline-flex;align-items:center;gap:8px;background:{color};'
        f'color:white;padding:9px 20px;border-radius:22px;font-weight:700;font-size:0.92rem;'
        f'letter-spacing:0.2px;box-shadow:0 3px 10px rgba(0,0,0,0.18);margin:4px 0 14px">'
        f'{icon}&nbsp; Accuracy Estimate: {acc.upper()}'
        f'</div>'
    )


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
    if status != "ok":
        badge_bg   = _STATUS_BADGE.get(status, "#607d8b")
        badge_text = _STATUS_LABEL.get(status, status.upper())
        badge_html = (
            f'<span style="flex-shrink:0;margin-left:10px;background:{badge_bg};color:white;'
            f'font-size:0.62rem;font-weight:700;padding:2px 8px;border-radius:10px;'
            f'letter-spacing:0.5px;white-space:nowrap">{badge_text}</span>'
        )

    return (
        f'<div style="display:flex;align-items:center;background:{bg};'
        f'border-left:4px solid {border};padding:8px 14px;margin:3px 0;'
        f'border-radius:0 8px 8px 0;box-shadow:0 1px 3px rgba(0,0,0,0.04)"{tooltip}>'
        f'<div style="flex:1;min-width:0">'
        f'<div style="font-size:0.67rem;color:#78909c;text-transform:uppercase;'
        f'letter-spacing:0.6px;font-weight:600;margin-bottom:2px">{label}</div>'
        f'{val_html}'
        f'</div>'
        f'{badge_html}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Composite renderers
# ---------------------------------------------------------------------------

def _render_summary(val_result: ValidationResult) -> None:
    invalid_n  = sum(1 for s in val_result.fields.values() if s.status == "invalid")
    uncertain_n = sum(1 for s in val_result.fields.values() if s.status == "uncertain")
    st.markdown(
        _metric_cards_html(val_result.completeness, len(val_result.fields), invalid_n, uncertain_n),
        unsafe_allow_html=True,
    )
    st.markdown(_accuracy_badge_html(val_result.accuracy_estimate), unsafe_allow_html=True)


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


def _proof_card_html(
    fname: str,
    value: str,
    fstatus: FieldStatus,
    word_conf: dict[str, float],
) -> str:
    status     = fstatus.status
    border     = _STATUS_BORDER.get(status, "#90a4ae")
    header_bg  = _STATUS_BG.get(status, "#fafafa")
    badge_bg   = _STATUS_BADGE.get(status, "#607d8b")
    badge_text = _STATUS_LABEL.get(status, status.upper())
    icon       = _STATUS_ICON.get(status, "")
    src_icon, src_label = _validation_source(fstatus.reason)

    # ── OCR evidence column ──────────────────────────────────────────────────
    ocr_parts: list[str] = []
    if value:
        for token in value.split():
            clean = token.lower().strip(".,;:\"'()")
            conf  = word_conf.get(clean)
            if conf is None:
                ocr_parts.append(
                    f'<span style="color:#b0bec5;font-style:italic;font-size:0.8rem">'
                    f'"{token}" not found</span>'
                )
            else:
                if conf >= 0.90:
                    pill_bg, pill_fg = "#e8f5e9", "#2e7d32"
                elif conf >= 0.70:
                    pill_bg, pill_fg = "#fff8e1", "#e65100"
                else:
                    pill_bg, pill_fg = "#ffebee", "#c62828"
                ocr_parts.append(
                    f'<span style="display:inline-flex;align-items:center;gap:4px;'
                    f'background:{pill_bg};border:1px solid {pill_fg}33;'
                    f'border-radius:5px;padding:2px 7px;margin:2px 2px 0 0;'
                    f'font-family:monospace;font-size:0.8rem;color:#263238">'
                    f'{token} <b style="color:{pill_fg}">{conf:.2f}</b></span>'
                )
    else:
        ocr_parts.append('<span style="color:#b0bec5;font-style:italic;font-size:0.8rem">no value to look up</span>')
    ocr_html = " ".join(ocr_parts)

    # ── LLM output column ────────────────────────────────────────────────────
    if value:
        llm_html = (
            f'<span style="font-family:monospace;font-size:0.88rem;color:#1a237e;'
            f'background:#e8eaf6;padding:4px 10px;border-radius:5px;'
            f'display:inline-block;word-break:break-all;max-width:100%">{value}</span>'
        )
    else:
        llm_html = '<span style="color:#b0bec5;font-style:italic;font-size:0.85rem">— empty —</span>'

    # ── Validation column ────────────────────────────────────────────────────
    reason_text = fstatus.reason or "Passed all checks"
    val_html = (
        f'<div style="margin-bottom:5px">'
        f'<span style="background:{badge_bg};color:white;font-size:0.63rem;font-weight:700;'
        f'padding:2px 9px;border-radius:9px;letter-spacing:0.4px">{badge_text}</span>'
        f'</div>'
        f'<div style="font-size:0.8rem;color:#546e7a;line-height:1.45;margin-bottom:6px">'
        f'{reason_text}</div>'
        f'<div style="font-size:0.7rem;color:#90a4ae">'
        f'{src_icon} {src_label}</div>'
    )

    col_style = (
        'style="background:white;padding:12px 16px;'
        'border-right:1px solid #f0f0f0;min-width:0"'
    )
    label_style = (
        'style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
        'letter-spacing:0.7px;color:#90a4ae;margin-bottom:6px"'
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

        # evidence grid
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;'
        f'background:#f5f5f5;gap:1px">'

        f'<div {col_style}>'
        f'<div {label_style}>📄 OCR Evidence</div>'
        f'<div style="line-height:1.6">{ocr_html}</div>'
        f'</div>'

        f'<div {col_style}>'
        f'<div {label_style}>🤖 LLM Output</div>'
        f'{llm_html}'
        f'</div>'

        f'<div style="background:white;padding:12px 16px;min-width:0">'
        f'<div {label_style}>🔍 Validation</div>'
        f'{val_html}'
        f'</div>'

        f'</div>'  # grid
        f'</div>'  # card
    )


def _render_validation_details(
    val_result: ValidationResult,
    extracted: dict,
    ocr_result: OCRResult,
) -> None:
    if not val_result.fields:
        st.write("No fields were specifically validated.")
        return

    word_conf: dict[str, float] = {
        w.lower().strip(".,;:\"'()"): c
        for w, c in ocr_result.word_confidences
    }

    issues = {k: v for k, v in val_result.fields.items() if v.status != "ok"}
    if not issues:
        st.success("All validated fields passed — no issues found.")
        return

    sorted_fields = sorted(
        issues.items(),
        key=lambda kv: (0 if kv[1].status == "invalid" else 1, kv[0]),
    )

    cards_html = "\n".join(
        _proof_card_html(fname, _get_field_value(extracted, fname), fstatus, word_conf)
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
    hdr_left, hdr_right = st.columns([2, 3], gap="large")

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
        if uploaded is not None:
            if st.button("🔍 Extract Fields", type="primary"):
                st.cache_data.clear()

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
        with st.expander("🔍 Validation details — with proof"):
            _render_validation_details(val_result, extracted, ocr_result)

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
