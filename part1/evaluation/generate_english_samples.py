"""
Generate English-filled BL283 forms for testing the extraction pipeline.
Overlays English data values onto the blank 283_raw.pdf template.

Outputs:
  phase1_data/283_en1.pdf  (and en2, en3)
  part1/evaluation/ground_truth/283_en1.json  (and en2, en3)

Run from the repo root:
  python -m part1.evaluation.generate_english_samples
"""

import fitz  # PyMuPDF
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "phase1_data" / "283_raw.pdf"
PDF_OUT = ROOT / "phase1_data"
GT_OUT = Path(__file__).parent / "ground_truth"

FONT = "helv"
FS = 8
FS_SM = 7

# Three valid Israeli IDs (check-digit verified: weighted-digit sum divisible by 10)
# 123456782: sum=40 ✓   987654324: sum=50 ✓   555444330: sum=40 ✓

SAMPLES = [
    {
        "pdf": "283_en1.pdf",
        "gt":  "283_en1.json",
        "data": {
            "lastName":   "Smith",
            "firstName":  "John",
            "idNumber":   "123456782",
            "gender":     "Male",
            "dateOfBirth":            {"day": "15", "month": "03", "year": "1985"},
            "address": {
                "street":      "42 Oak Street",
                "houseNumber": "42",
                "entrance":    "A",
                "apartment":   "3",
                "city":        "Tel Aviv",
                "postalCode":  "6120001",
                "poBox":       "",
            },
            "landlinePhone":  "036541234",
            "mobilePhone":    "0521234567",
            "jobType":        "Construction Worker",
            "dateOfInjury":   {"day": "10", "month": "06", "year": "2024"},
            "timeOfInjury":   "14:30",
            "accidentLocation":    "At workplace",
            "accidentAddress":     "42 Industrial Zone, Tel Aviv",
            "accidentDescription": "Fell from scaffolding on third floor, landed on left arm",
            "injuredBodyPart":     "Left arm",
            "signature":           "John Smith",
            "formFillingDate":         {"day": "10", "month": "06", "year": "2024"},
            "formReceiptDateAtClinic": {"day": "11", "month": "06", "year": "2024"},
            "medicalInstitutionFields": {
                "healthFundMember":  "maccabi",
                "natureOfAccident":  "Workplace fall",
                "medicalDiagnoses":  "Fracture of left radius",
            },
        },
        # Canonical values the pipeline extracts (differ from the raw data above for
        # fields the pipeline normalises: gender → Hebrew, signature → "קיימת" or "",
        # healthFundMember → canonical Hebrew HMO name).
        "gt_overrides": {
            "gender": "זכר",
            "signature": "קיימת",
            "medicalInstitutionFields": {
                "healthFundMember": "מכבי",
                "natureOfAccident": "Workplace fall",
                "medicalDiagnoses": "Fracture of left radius",
            },
        },
    },
    {
        "pdf": "283_en2.pdf",
        "gt":  "283_en2.json",
        "data": {
            "lastName":   "Johnson",
            "firstName":  "Sarah",
            "idNumber":   "987654324",
            "gender":     "Female",
            "dateOfBirth":            {"day": "22", "month": "08", "year": "1990"},
            "address": {
                "street":      "15 Pine Avenue",
                "houseNumber": "15",
                "entrance":    "B",
                "apartment":   "7",
                "city":        "Haifa",
                "postalCode":  "3200001",
                "poBox":       "",
            },
            "landlinePhone":  "048765432",
            "mobilePhone":    "0529876543",
            "jobType":        "Software Developer",
            "dateOfInjury":   {"day": "03", "month": "11", "year": "2023"},
            "timeOfInjury":   "09:15",
            "accidentLocation":    "Road accident on way to work",
            "accidentAddress":     "Route 2 near Haifa interchange",
            "accidentDescription": "Slipped on wet entrance floor, injured right knee",
            "injuredBodyPart":     "Right knee",
            "signature":           "Sarah Johnson",
            "formFillingDate":         {"day": "03", "month": "11", "year": "2023"},
            "formReceiptDateAtClinic": {"day": "04", "month": "11", "year": "2023"},
            "medicalInstitutionFields": {
                "healthFundMember":  "clalit",
                "natureOfAccident":  "Road accident on way to work",
                "medicalDiagnoses":  "Right knee sprain, grade II",
            },
        },
        "gt_overrides": {
            "gender": "נקבה",
            "signature": "קיימת",
            "medicalInstitutionFields": {
                "healthFundMember": "כללית",
                "natureOfAccident": "Road accident on way to work",
                "medicalDiagnoses": "Right knee sprain, grade II",
            },
        },
    },
    {
        "pdf": "283_en3.pdf",
        "gt":  "283_en3.json",
        "data": {
            "lastName":   "Cohen",
            "firstName":  "David",
            "idNumber":   "555444330",
            "gender":     "Male",
            "dateOfBirth":            {"day": "07", "month": "12", "year": "1978"},
            "address": {
                "street":      "8 Herzl Boulevard",
                "houseNumber": "8",
                "entrance":    "",
                "apartment":   "12",
                "city":        "Jerusalem",
                "postalCode":  "9100001",
                "poBox":       "",
            },
            "landlinePhone":  "025678901",
            "mobilePhone":    "0523456789",
            "jobType":        "Electrician",
            "dateOfInjury":   {"day": "25", "month": "08", "year": "2024"},
            "timeOfInjury":   "11:00",
            "accidentLocation":    "At workplace",
            "accidentAddress":     "City Hall, 1 Safra Square, Jerusalem",
            "accidentDescription": "Electric shock while repairing panel in basement",
            "injuredBodyPart":     "Right hand and forearm",
            "signature":           "David Cohen",
            "formFillingDate":         {"day": "25", "month": "08", "year": "2024"},
            "formReceiptDateAtClinic": {"day": "26", "month": "08", "year": "2024"},
            "medicalInstitutionFields": {
                "healthFundMember":  "meuhedet",
                "natureOfAccident":  "Workplace electrical accident",
                "medicalDiagnoses":  "Electrical burns right hand, superficial",
            },
        },
        "gt_overrides": {
            "gender": "זכר",
            "signature": "קיימת",
            "medicalInstitutionFields": {
                "healthFundMember": "מאוחדת",
                "natureOfAccident": "Workplace electrical accident",
                "medicalDiagnoses": "Electrical burns right hand, superficial",
            },
        },
    },
]


def txt(page, x, y, text, fs=FS):
    """Insert Latin text at PDF point (x, y) — y is the text baseline."""
    if text:
        page.insert_text((x, y), text, fontname=FONT, fontsize=fs, color=(0, 0, 0))


def cross(page, x, y, size=7):
    """Draw an X at (x,y) to mark a checkbox."""
    page.draw_line((x, y), (x + size, y + size), color=(0, 0, 0), width=0.8)
    page.draw_line((x + size, y), (x, y + size), color=(0, 0, 0), width=0.8)


def date_boxes(page, x_day, x_month, x_year, y, d):
    """Place day / month / year values into three boxes."""
    txt(page, x_day,   y, d["day"],   FS_SM)
    txt(page, x_month, y, d["month"], FS_SM)
    txt(page, x_year,  y, d["year"],  FS_SM)


def render(sample: dict):
    doc = fitz.open(str(TEMPLATE))
    page = doc[0]
    d = sample["data"]

    # ── Header date boxes (top of form, outside the numbered sections) ──────
    # Page coords: day box is rightmost, year box leftmost (Hebrew RTL ordering)

    # "תאריך קבלת הטופס בקופה" – receipt date at clinic (top-centre boxes)
    rdate = d["formReceiptDateAtClinic"]
    txt(page, 305, 100, rdate["day"],   FS_SM)
    txt(page, 263, 100, rdate["month"], FS_SM)
    txt(page, 218, 100, rdate["year"],  FS_SM)   # full 4-digit year

    # "תאריך מילוי הטופס" – form filling date (top-left boxes)
    fdate = d["formFillingDate"]
    txt(page, 155, 100, fdate["day"],   FS_SM)
    txt(page, 113, 100, fdate["month"], FS_SM)
    txt(page, 70,  100, fdate["year"],  FS_SM)   # full 4-digit year

    # ── Section 1: injury date ───────────────────────────────────────────────
    date_boxes(page, x_day=445, x_month=360, x_year=265, y=175,
               d=d["dateOfInjury"])

    # ── Section 2: claimant personal details ────────────────────────────────
    # Row 1: surname (right col) | first name (mid col) | ID (left col)
    txt(page, 400, 238, d["lastName"])
    txt(page, 255, 238, d["firstName"])
    txt(page, 80,  238, d["idNumber"])

    # Gender checkboxes: זכר (male) checkbox ~ x=505, נקבה (female) ~ x=430
    if d["gender"].lower() in ("male", "m", "זכר"):
        cross(page, 503, 281)
    else:
        cross(page, 428, 281)

    # Date of birth: boxes right → left = day | month | year
    date_boxes(page, x_day=370, x_month=290, x_year=180, y=287,
               d=d["dateOfBirth"])

    # Address row (right→left): street | house# | entrance | apt | city | postal
    addr = d["address"]
    txt(page, 395, 343, addr["street"],      FS_SM)
    txt(page, 327, 343, addr["houseNumber"], FS_SM)
    txt(page, 287, 343, addr["entrance"],    FS_SM)
    txt(page, 257, 343, addr["apartment"],   FS_SM)
    txt(page, 142, 343, addr["city"],        FS_SM)
    txt(page, 62,  343, addr["postalCode"],  FS_SM)

    # Phones: landline (right half), mobile (left half)
    txt(page, 362, 376, d["landlinePhone"])
    txt(page, 107, 376, d["mobilePhone"])

    # ── Section 3: accident details ─────────────────────────────────────────
    # "בתאריך __ בשעה __ כאשר עבדתי ב __"
    inj = d["dateOfInjury"]
    txt(page, 393, 441, f"{inj['day']}/{inj['month']}/{inj['year']}", FS_SM)
    txt(page, 282, 441, d["timeOfInjury"], FS_SM)
    txt(page, 60,  441, d["jobType"],      FS_SM)   # employer / type of work

    # Job type label line (סוג העבודה) — blank extends left
    txt(page, 60, 452, d["jobType"], FS_SM)

    # Accident-location checkboxes (right→left): במפעל | דרכים בעבודה | דרכים לעבודה | ללא רכב | אחר
    loc = d["accidentLocation"].lower()
    if "workplace" in loc or "factory" in loc:
        cross(page, 393, 459)
    elif "to work" in loc or "from work" in loc:
        cross(page, 207, 459)
    elif "road" in loc:
        cross(page, 315, 459)
    else:
        cross(page, 115, 459)   # אחר (other)

    # Accident address
    txt(page, 60, 483, d["accidentAddress"], FS_SM)

    # Accident description (up to two lines)
    desc = d["accidentDescription"]
    if len(desc) > 68:
        txt(page, 60, 520, desc[:68], FS_SM)
        txt(page, 60, 535, desc[68:], FS_SM)
    else:
        txt(page, 60, 520, desc, FS_SM)

    # Injured body part
    txt(page, 60, 554, d["injuredBodyPart"], FS_SM)

    # ── Section 4: declaration ──────────────────────────────────────────────
    # שם המבקש (applicant name) on the right, חתימה (signature) on the left
    txt(page, 355, 662, d["signature"])
    txt(page, 70,  662, d["signature"])

    # ── Section 5: medical-institution fields ───────────────────────────────
    # HMO checkboxes (right→left): כללית | מאוחדת | מכבי | לאומית
    hmo = d["medicalInstitutionFields"]["healthFundMember"].lower()
    hmo_x = {"clalit": 450, "כללית": 450,
              "meuhedet": 394, "מאוחדת": 394,
              "maccabi": 346, "מכבי": 346,
              "leumit": 298, "לאומית": 298}
    for key, x in hmo_x.items():
        if key in hmo:
            cross(page, x, 707)
            break

    # Nature of accident / medical diagnoses
    txt(page, 60, 752, d["medicalInstitutionFields"]["medicalDiagnoses"], FS_SM)

    out = PDF_OUT / sample["pdf"]
    doc.save(str(out))
    doc.close()
    rasterize(out)
    print(f"  PDF  -> {out}")


def rasterize(path: Path, dpi: int = 150) -> None:
    """
    Re-save the PDF as a raster-image PDF (one grayscale PNG per page).

    This makes Azure Document Intelligence treat the file like a scanned document
    rather than a native-text PDF, so its OCR pipeline (and the resulting markdown
    format) is consistent with the handwritten Hebrew forms.  150 DPI grayscale
    keeps the file small (<3 MB/page) while remaining legible for OCR.
    """
    import shutil
    scale = dpi / 72
    tmp = path.with_suffix(".tmp.pdf")
    src = fitz.open(str(path))
    raster_doc = fitz.open()
    for page in src:
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY)
        new_page = raster_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, pixmap=pix)
    raster_doc.save(str(tmp))
    raster_doc.close()
    src.close()
    shutil.move(str(tmp), str(path))


def main():
    PDF_OUT.mkdir(exist_ok=True)
    GT_OUT.mkdir(exist_ok=True)

    for s in SAMPLES:
        render(s)

        # Ground truth uses canonical pipeline output values, not the raw form data.
        gt_data = {**s["data"], **s["gt_overrides"]}
        gt_path = GT_OUT / s["gt"]
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, ensure_ascii=False, indent=2)
        print(f"  JSON -> {gt_path}")


if __name__ == "__main__":
    main()
