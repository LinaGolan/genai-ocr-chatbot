from __future__ import annotations

# Part 1 — Streamlit UI for BL283 form field extraction.
# Layout: left panel = raw OCR markdown, right panel = colour-coded JSON + badges.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from typing import Any

import streamlit as st

from part1.backend.ocr_client import analyze_document, OCRResult
from part1.backend.extractor import extract_fields
from part1.backend.validator import validate
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
# Styling helpers
# ---------------------------------------------------------------------------

_STATUS_BG = {"ok": "#e8f5e9", "uncertain": "#fff9c4", "invalid": "#ffebee"}
_STATUS_BORDER = {"ok": "#66bb6a", "uncertain": "#ffa726", "invalid": "#ef5350"}
_STATUS_ICON = {"ok": "✅", "uncertain": "⚠️", "invalid": "❌"}
_ACCURACY_COLOR = {"high": "#4caf50", "medium": "#ff9800", "low": "#f44336"}


def _field_html(label: str, value: str, field_key: str, statuses: dict[str, FieldStatus]) -> str:
    """Return an HTML div for one leaf field with status colouring."""
    status_obj = statuses.get(field_key, FieldStatus("ok"))
    status = status_obj.status
    bg = _STATUS_BG.get(status, "#f5f5f5")
    border = _STATUS_BORDER.get(status, "#9e9e9e")
    icon = _STATUS_ICON.get(status, "")
    val_html = f"<code>{value}</code>" if value else '<span style="color:#9e9e9e;font-style:italic">—</span>'
    tooltip = f' title="{status_obj.reason}"' if status_obj.reason else ""
    return (
        f'<div style="background:{bg};border-left:3px solid {border};'
        f'padding:5px 10px;margin:2px 0;border-radius:4px;'
        f'font-family:monospace;font-size:0.85rem"{tooltip}>'
        f'{icon} <b>{label}:</b> {val_html}</div>'
    )


def _section_header(title: str) -> str:
    return (
        f'<div style="background:#eceff1;padding:4px 10px;margin:6px 0 2px;'
        f'border-radius:4px;font-weight:600;color:#37474f">{title}</div>'
    )


def _render_json_panel(extracted: dict[str, Any], val_result: ValidationResult) -> None:
    """Render the full extracted JSON as colour-coded field rows."""
    statuses = val_result.fields
    html_parts: list[str] = []

    def field(label: str, value: str, key: str) -> None:
        html_parts.append(_field_html(label, value, key, statuses))

    def date_block(label: str, key: str) -> None:
        d = extracted.get(key) or {}
        val = "/".join(filter(None, [d.get("day"), d.get("month"), d.get("year")]))
        html_parts.append(_field_html(label, val, key, statuses))

    # -- Personal info -------------------------------------------------------
    html_parts.append(_section_header("👤 Personal Information"))
    field("Last Name (שם משפחה)", extracted.get("lastName", ""), "lastName")
    field("First Name (שם פרטי)", extracted.get("firstName", ""), "firstName")
    field("ID Number (מספר זהות)", extracted.get("idNumber", ""), "idNumber")
    field("Gender (מין)", extracted.get("gender", ""), "gender")
    date_block("Date of Birth (תאריך לידה)", "dateOfBirth")

    # -- Address -------------------------------------------------------------
    html_parts.append(_section_header("🏠 Address (כתובת)"))
    addr = extracted.get("address") or {}
    field("Street (רחוב)", addr.get("street", ""), "address.street")
    field("House No. (מספר בית)", addr.get("houseNumber", ""), "address.houseNumber")
    field("Entrance (כניסה)", addr.get("entrance", ""), "address.entrance")
    field("Apartment (דירה)", addr.get("apartment", ""), "address.apartment")
    field("City (ישוב)", addr.get("city", ""), "address.city")
    field("Postal Code (מיקוד)", addr.get("postalCode", ""), "address.postalCode")
    field("P.O. Box (תא דואר)", addr.get("poBox", ""), "address.poBox")

    # -- Contact -------------------------------------------------------------
    html_parts.append(_section_header("📞 Contact"))
    field("Landline (טלפון קווי)", extracted.get("landlinePhone", ""), "landlinePhone")
    field("Mobile (טלפון נייד)", extracted.get("mobilePhone", ""), "mobilePhone")

    # -- Employment ----------------------------------------------------------
    html_parts.append(_section_header("💼 Employment"))
    field("Job Type (סוג העבודה)", extracted.get("jobType", ""), "jobType")

    # -- Incident ------------------------------------------------------------
    html_parts.append(_section_header("⚕️ Incident (פגיעה)"))
    date_block("Date of Injury (תאריך הפגיעה)", "dateOfInjury")
    field("Time of Injury (שעת הפגיעה)", extracted.get("timeOfInjury", ""), "timeOfInjury")
    field("Accident Location (מקום התאונה)", extracted.get("accidentLocation", ""), "accidentLocation")
    field("Accident Address (כתובת מקום התאונה)", extracted.get("accidentAddress", ""), "accidentAddress")
    field("Description (תיאור התאונה)", extracted.get("accidentDescription", ""), "accidentDescription")
    field("Injured Body Part (האיבר שנפגע)", extracted.get("injuredBodyPart", ""), "injuredBodyPart")

    # -- Submission ----------------------------------------------------------
    html_parts.append(_section_header("📝 Submission"))
    field("Signature (חתימה)", extracted.get("signature", ""), "signature")
    date_block("Form Filling Date (תאריך מילוי הטופס)", "formFillingDate")
    date_block("Clinic Receipt Date (תאריך קבלת הטופס בקופה)", "formReceiptDateAtClinic")

    # -- Medical institution -------------------------------------------------
    html_parts.append(_section_header("🏥 Medical Institution Fields"))
    med = extracted.get("medicalInstitutionFields") or {}
    field("Health Fund Member (חבר בקופת חולים)", med.get("healthFundMember", ""), "medicalInstitutionFields.healthFundMember")
    field("Nature of Accident (מהות התאונה)", med.get("natureOfAccident", ""), "medicalInstitutionFields.natureOfAccident")
    field("Medical Diagnoses (אבחנות רפואיות)", med.get("medicalDiagnoses", ""), "medicalInstitutionFields.medicalDiagnoses")

    st.markdown("\n".join(html_parts), unsafe_allow_html=True)


def _render_badges(val_result: ValidationResult) -> None:
    """Render the accuracy badge and completeness metric."""
    acc = val_result.accuracy_estimate
    acc_color = _ACCURACY_COLOR.get(acc, "#9e9e9e")
    completeness_pct = f"{val_result.completeness:.0%}"

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Completeness", completeness_pct)
    col_b.metric("Fields Validated", len(val_result.fields))

    invalid_n = sum(1 for s in val_result.fields.values() if s.status == "invalid")
    uncertain_n = sum(1 for s in val_result.fields.values() if s.status == "uncertain")
    col_c.metric("Issues", f"{invalid_n} invalid / {uncertain_n} uncertain")

    st.markdown(
        f'<div style="display:inline-block;background:{acc_color};color:white;'
        f'padding:4px 14px;border-radius:12px;font-weight:600;font-size:0.9rem;margin:4px 0">'
        f'Accuracy estimate: {acc.upper()}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pipeline helper (cached per file bytes)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _run_pipeline(file_bytes: bytes, filename: str) -> tuple[OCRResult, dict, dict]:
    """
    Returns (ocr_result, extracted_dict, validation_dict).
    Cached so reruns (e.g. UI interactions) don't re-call Azure.
    """
    ocr = analyze_document(file_bytes, filename)
    extracted = extract_fields(ocr.markdown)
    val = validate(extracted, ocr_result=ocr, ocr_markdown=ocr.markdown)
    return ocr, extracted, val.to_dict()


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("📋 BL283 Form Field Extractor")
    st.caption(
        "Upload a ביטוח לאומי (National Insurance) BL283 form (PDF or JPG). "
        "The system uses Azure Document Intelligence + GPT-4o to extract and validate all fields."
    )

    uploaded = st.file_uploader(
        "Choose a BL283 form file",
        type=["pdf", "jpg", "jpeg"],
        help="Accepted: PDF, JPG — max 50 MB",
    )

    if uploaded is None:
        st.info("Upload a form file to begin.")
        return

    file_bytes = uploaded.read()

    if st.button("🔍 Extract Fields", type="primary"):
        # Clear any cached result for a fresh run
        st.cache_data.clear()

    # Run pipeline (uses cache if button wasn't pressed again)
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

    # Reconstruct ValidationResult for rendering
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
        st.subheader("📄 OCR Output")
        st.text_area(
            label="Raw OCR markdown from Document Intelligence",
            value=ocr_result.markdown,
            height=700,
            label_visibility="collapsed",
        )
        with st.expander("OCR confidence details"):
            st.write(f"**Average confidence:** {ocr_result.avg_confidence:.3f}")
            st.write(f"**Minimum confidence:** {ocr_result.min_confidence:.3f}")
            st.write(f"**Words analysed:** {len(ocr_result.word_confidences)}")

    with col_right:
        st.subheader("🗂️ Extracted Fields")
        _render_badges(val_result)
        st.divider()
        _render_json_panel(extracted, val_result)

        with st.expander("📊 View raw JSON"):
            st.code(
                json.dumps(extracted, ensure_ascii=False, indent=2),
                language="json",
            )

        with st.expander("🔍 Validation details"):
            if not val_result.fields:
                st.write("No specific field issues detected.")
            else:
                for fname, fstatus in sorted(val_result.fields.items()):
                    icon = _STATUS_ICON.get(fstatus.status, "")
                    reason = f" — {fstatus.reason}" if fstatus.reason else ""
                    st.markdown(f"{icon} **{fname}**: `{fstatus.status}`{reason}")


if __name__ == "__main__":
    main()
