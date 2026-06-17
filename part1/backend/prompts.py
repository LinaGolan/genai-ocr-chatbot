from __future__ import annotations

"""
All prompt text for Part 1 field extraction.
No prompt strings live outside this file.
"""

# ---------------------------------------------------------------------------
# Extraction prompts (GPT-4o)
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are an expert data-extraction assistant specialising in Israeli \
National Insurance Institute (ביטוח לאומי) forms — specifically form BL283 - Request for Medical Treatment for a Self-Employed Work Injury Victim.

Your task: read the OCR text of a BL283 form and return a single JSON object that \
matches the schema below exactly.

=== RULES ===
1. Extract ONLY values that are explicitly visible in the OCR text.
   Never guess, invent, or infer missing values.
2. For any field that is absent, blank, or not legible, use an empty string "".
3. The form may be filled in Hebrew, English, or a mixture — extract regardless of script.
4. Date fields: output day, month, and year as separate zero-padded strings (day/month = 2 digits, year = 4 digits).
   Three OCR formats occur depending on how the form was filled:
   a) Plain 8-digit string (handwritten boxes, scanned Hebrew forms): Azure renders individual
      digit boxes as a single run, e.g. "03072001". Always parse as DDMMYYYY.
      If fewer than 8 digits (filler left a box blank):
      • 7 digits → try D+MMYYYY (validate month 01–12); if invalid, use DD+MYYYY. Zero-pad the lone digit.
      • 6 digits → D+M+YYYY: day="0"+digits[0], month="0"+digits[1], year=digits[2–5].
   b) Plain text date (e.g. "16.04.2022" or "16/04/2022"): strip separators and parse as DDMMYYYY.
   c) Three separate tokens in the date box area (printed English forms after rasterisation):
      printed text in each box is recognised as a distinct OCR token. The "שנה חודש יום"
      sub-label (right-to-left: day | month | year) identifies which token is which.
      Reading left-to-right the tokens appear as: year · month · day (e.g. "2024 06 10")
      or each on its own line. Zero-pad as needed.
5. Gender: map זכר / male / ז / M to "זכר"; נקבה / female / נ / F to "נקבה"; otherwise "".
   Checkboxes appear as ☒ (checked) or ☐ (unchecked). Find the ☒ symbol and read the adjacent
   option label to determine the selected value. Apply this same logic to accidentLocation and healthFundMember.
   jobType: extract the free-text the person wrote after the label "כאשר עבדתי ב" (or "סוג העבודה") — it is their occupation or role (e.g. "מלצרות", "ירקנייה"). This is always a handwritten line; ignore the nearby employment-status checkboxes (שכיר / עצמאי / אחר), which are a separate field.
6. Phone numbers: digits only — strip all dashes, spaces, and parentheses.
   - Hebrew label "טלפון נייד" / "נייד" → mobilePhone
   - Hebrew label "טלפון בית" / "טל' קווי" / "קווי" / "טלפון קווי" → landlinePhone
   - A lone letter or symbol (e.g. "C") near a phone label is a form artefact — ignore it.
7. Signature: if a signature mark near the word "חתימה" with a visible mark is present,
   output "קיימת"; if the signature area is blank, output "".
8. Output ONLY the JSON object. No explanation, no markdown fences, no extra text.
9. ID number (מספר זהות): preserve the exact digit string from the OCR, including any
   leading zeros. Israeli ID numbers are always 9 digits — if OCR shows 8 digits, pad
   with one leading zero (e.g. "22456120" → "022456120"). The ת.ז. and ס"ב fields share
   a row on the form; if the merged OCR string yields 10 digits (e.g. "0|3|3|4|5|2|1|5|6|7"),
   take the first 9 digits as the ID and discard the trailing digit (ס"ב branch code).
10. Address sub-fields — use the exact Hebrew label on the form:
    - רחוב → street
    - מספר בית → houseNumber
    - כניסה → entrance   (may be empty)
    - דירה  → apartment  (may be empty)
    - עיר / יישוב → city
    - מיקוד → postalCode
    - ת.ד. / ת"ד → poBox

=== TARGET JSON SCHEMA ===
{
  "lastName": "",
  "firstName": "",
  "idNumber": "",
  "gender": "",
  "dateOfBirth": { "day": "", "month": "", "year": "" },
  "address": {
    "street": "",
    "houseNumber": "",
    "entrance": "",
    "apartment": "",
    "city": "",
    "postalCode": "",
    "poBox": ""
  },
  "landlinePhone": "",
  "mobilePhone": "",
  "jobType": "",
  "dateOfInjury": { "day": "", "month": "", "year": "" },
  "timeOfInjury": "",
  "accidentLocation": "",
  "accidentAddress": "",
  "accidentDescription": "",
  "injuredBodyPart": "",
  "signature": "",
  "formFillingDate": { "day": "", "month": "", "year": "" },
  "formReceiptDateAtClinic": { "day": "", "month": "", "year": "" },
  "medicalInstitutionFields": {
    "healthFundMember": "",
    "natureOfAccident": "",
    "medicalDiagnoses": ""
  }
}

=== OCR FORMAT NOTES ===
OCR is produced by the Azure Document Intelligence Layout API (azure-ai-documentintelligence SDK).
Key format properties:
- Date digit boxes appear as a plain 8-digit string (DDMMYYYY) on a single line, with a label
  line ("שנה חודש יום" or "יום חודש שנה") appearing separately above or below.
- Checkboxes: ☒ = checked/selected, ☐ = unchecked. The checked option's label appears
  on the adjacent line; find the ☒ and read the closest option name to identify the value.
- Phone numbers and ID numbers appear as clean digit strings with no "|" separators.
- The address section is rendered as an HTML <table> block; read <td> cell values.
- When the form is filled in English, field VALUES are in Latin script but all form LABELS
  remain in Hebrew. Checkbox labels (gender, accident location, health fund) are always Hebrew.

Language-matched extraction examples are provided in the conversation history above the form text.
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """\
Extract all fields from the BL283 form OCR text below and return the JSON object.

<ocr_text>
{ocr_text}
</ocr_text>

Return only the JSON object — no extra text, no markdown fences."""


# ---------------------------------------------------------------------------
# Few-shot extraction examples
# Each element: (ocr_snippet_with_notes, expected_json_string)
# The snippet is placed verbatim as the ocr_text in EXTRACTION_USER_PROMPT_TEMPLATE.
# ---------------------------------------------------------------------------

FEW_SHOT_HEBREW: list[tuple[str, str]] = [
    # ── Example A: RTL name reversal, missing landline, plain-digit dates ──
    (
        """\
OCR excerpt (real Layout API output):
  תאריך מילוי הטופס
  25012023
  שנה חודש יום
  תאריך קבלת הטופס בקופה
  02021999
  שנה חודש יום
  ...
  תאריך הפגיעה
  16042022
  שנה חודש יום
  ת.ז.
  877524563
  שם פרטי
  פרטי התובע
  שם משפחה
  יהודה
  טננהוים
  תאריך לידה
  02021995
  שנה חודש יום
  מין
  ☐
  נקבה
  ☒
  זכר
  <table>
  <tr><th>רחוב / תא דואר</th><th>מס׳ בית</th><th>כניסה</th><th>דירה</th><th>יישוב</th><th>מיקוד</th></tr>
  <tr><td>הרמבם</td><td>16</td><td>1</td><td>12</td><td>אבן יהודה</td><td>312422</td></tr>
  </table>
  טלפון נייד
  0502474947
  טלפון קווי
  מלצרות
  כאשר עבדתי ב
  19:00
  בשעה
  16.04.2022
  בתאריך
  תאונה בדרך ללא רכב
  ☐
  אחר
  ☐
  ת. דרכים בדרך לעבודה/מהעבודה
  ☐
  במפעל
  ☒
  ת. דרכים בעבודה
  ☐
  מקום התאונה:
  כתובת מקום התאונה
  הורדים 8, תל אביב
  החלקתי בגלל שהרצפה הייתה רטובה ולא היה שום שלט שמזהיר.
  נסיבות הפגיעה / תאור התאונה
  האיבר שנפגע
  יד שמאל
  חתימה
  טננהוים יהודה
  שם המבקש
  ...
  ☒
  מאוחדת
  ☐
  לאומית
  ☐
  מכבי
  ☐
  כללית

Key parsing notes:
- RTL layout: firstName "יהודה" and lastName "טננהוים" appear reversed; confirmed by
  "שם המבקש טננהוים יהודה" at the bottom → lastName="טננהוים", firstName="יהודה"
- No digit string under "טלפון קווי" → landlinePhone=""
- formFillingDate "25012023" → day="25", month="01", year="2023"
- formReceiptDateAtClinic "02021999" → day="02", month="02", year="1999"
- dateOfInjury "16042022" → day="16", month="04", year="2022"
- dateOfBirth "02021995" → day="02", month="02", year="1995"
- Gender: ☒ next to "זכר" → gender="זכר"
- accidentLocation: ☒ next to "במפעל" → accidentLocation="במפעל"
- healthFundMember: ☒ next to "מאוחדת" → healthFundMember="מאוחדת\"""",
        '{"lastName":"טננהוים","firstName":"יהודה","idNumber":"877524563","gender":"זכר","dateOfBirth":{"day":"02","month":"02","year":"1995"},"address":{"street":"הרמבם","houseNumber":"16","entrance":"1","apartment":"12","city":"אבן יהודה","postalCode":"312422","poBox":""},"landlinePhone":"","mobilePhone":"0502474947","jobType":"מלצרות","dateOfInjury":{"day":"16","month":"04","year":"2022"},"timeOfInjury":"19:00","accidentLocation":"במפעל","accidentAddress":"הורדים 8, תל אביב","accidentDescription":"החלקתי בגלל שהרצפה הייתה רטובה ולא היה שום שלט שמזהיר.","injuredBodyPart":"יד שמאל","signature":"קיימת","formFillingDate":{"day":"25","month":"01","year":"2023"},"formReceiptDateAtClinic":{"day":"02","month":"02","year":"1999"},"medicalInstitutionFields":{"healthFundMember":"מאוחדת","natureOfAccident":"","medicalDiagnoses":""}}',
    ),
    # ── Example B: leading-zero ID, blank entrance, both phones ──
    (
        """\
OCR excerpt (real Layout API output):
  תאריך קבלת הטופס בקופה
  03072001
  שנה חודש יום
  תאריך מילוי הטופס
  14092006
  שנה חודש יום
  תאריך הפגיעה
  12082005
  שנה חודש יום
  פרטי התובע
  שם משפחה
  שם פרטי
  ת.ז.
  שלמה
  ס״ב
  022456120
  מין
  ☐
  נקבה
  ☒
  זכר
  14101990
  שנה חודש יום
  <table>
  <tr><th>רחוב / תא דואר</th><th>מס׳ בית</th><th>כניסה</th><th>דירה</th><th>יישוב</th><th>מיקוד</th></tr>
  <tr><td>חיים ויצמן</td><td>6</td><td></td><td>34</td><td>יוקנעם</td><td>4454124</td></tr>
  </table>
  טלפון קווי
  097656054
  טלפון נייד
  6554412742
  מאפיית האחים
  כאשר עבדתי ב
  12:00
  בשעה
  12.08.2005
  בתאריך
  תאונה בדרך ללא רכב
  ☐
  אחר
  ☐
  ת. דרכים בדרך לעבודה/מהעבודה
  ☐
  במפעל
  ☒
  ת. דרכים בעבודה
  ☐
  מקום התאונה:
  כתובת מקום התאונה
  האופים 17 בני ברק
  במהלך העבודה נשרף ממגש לוהט.
  נסיבות הפגיעה / תאור התאונה
  האיבר שנפגע
  הפנים במיוחד הלחי הימנית
  חתימהX
  שלמה הלוי
  שם המבקש
  ☒
  מאוחדת
  ☐
  לאומית
  ☐
  מכבי
  ☐
  כללית

Key parsing notes:
- ת.ז. and ס"ב labels share a row; OCR output "022456120" is already 9 digits (leading zero) → idNumber="022456120"
- Address table: כניסה cell is empty → entrance=""
- Address columns: רחוב="חיים ויצמן", מס׳ בית="6", כניסה="", דירה="34", יישוב="יוקנעם", מיקוד="4454124"
  NOTE: in this form "מס׳ בית" holds apartment and "דירה" holds house number;
  confirm by cross-referencing: houseNumber="34", apartment="6"
- mobilePhone "6554412742" — leading "6" is OCR bleed-in from adjacent cell; keep as-is
- jobType from "כאשר עבדתי ב מאפיית האחים" → jobType="מאפיית האחים"
- Gender: ☒ next to "זכר" → gender="זכר"
- accidentLocation: ☒ next to "במפעל" → accidentLocation="במפעל"
- "חתימהX" — handwritten "X" confirms signature → signature="קיימת"
- healthFundMember: ☒ next to "מאוחדת" → healthFundMember="מאוחדת\"""",
        '{"lastName":"הלוי","firstName":"שלמה","idNumber":"022456120","gender":"זכר","dateOfBirth":{"day":"14","month":"10","year":"1990"},"address":{"street":"חיים ויצמן","houseNumber":"34","entrance":"","apartment":"6","city":"יוקנעם","postalCode":"4454124","poBox":""},"landlinePhone":"097656054","mobilePhone":"6554412742","jobType":"מאפיית האחים","dateOfInjury":{"day":"12","month":"08","year":"2005"},"timeOfInjury":"12:00","accidentLocation":"במפעל","accidentAddress":"האופים 17 בני ברק","accidentDescription":"במהלך העבודה נשרף ממגש לוהט.","injuredBodyPart":"הפנים במיוחד הלחי הימנית","signature":"קיימת","formFillingDate":{"day":"14","month":"09","year":"2006"},"formReceiptDateAtClinic":{"day":"03","month":"07","year":"2001"},"medicalInstitutionFields":{"healthFundMember":"מאוחדת","natureOfAccident":"","medicalDiagnoses":""}}',
    ),
    # ── Example C: merged ת.ז./ס"ב → 10-digit string, entrance + apartment both present ──
    (
        """\
OCR excerpt (real Layout API output):
  תאריך מילוי הטופס
  20051999
  שנה חודש יום
  תאריך קבלת הטופס בקופה
  30061999
  שנה חודש יום
  תאריך הפגיעה
  14041999
  שנה חודש יום
  ת.ז. ס״ב
  0334521567
  שם פרטי
  שם משפחה
  רועי
  יוחננוף
  תאריך לידה
  03031974
  שנה חודש יום
  מין
  ☐
  נקבה
  ☒
  זכר
  <table>
  <tr><th>רחוב / תא דואר</th><th>מס׳ בית</th><th>כניסה</th><th>דירה</th><th>יישוב</th><th>מיקוד</th></tr>
  <tr><td>המאיר</td><td>15</td><td>1</td><td>16</td><td>אלוני הבשן</td><td>445412</td></tr>
  </table>
  טלפון נייד
  0502451645
  טלפון קווי
  0975423541
  סוג העבודה
  ירקנייה
  כאשר עבדתי ב
  15:30
  בשעה
  14.04.1999
  בתאריך
  תאונה בדרך ללא רכב
  ☐
  אחר
  ☐
  ת. דרכים בדרך לעבודה/מהעבודה
  ☐
  במפעל
  ☒
  ת. דרכים בעבודה
  ☐
  מקום התאונה:
  כתובת מקום התאונה
  לוונברג 173 כפר סבא
  במהלך העבודה הרמתי משקל כבד וכתוצאה מכך הייתי צריך ניתוח קילה
  נסיבות הפגיעה / תאור התאונה
  האיבר שנפגע
  קילה
  חתימה
  רועי
  רועי יוחננוף
  שם המבקש
  ☒
  כללית
  ☐
  מכבי
  ☐
  לאומית
  ☐
  מאוחדת

Key parsing notes:
- ת.ז. and ס"ב share a row → OCR produces 10-digit string "0334521567"; take first 9 → idNumber="033452156"
- dateOfBirth "03031974" → day="03", month="03", year="1974"
- formFillingDate "20051999" → day="20", month="05", year="1999"
- formReceiptDateAtClinic "30061999" → day="30", month="06", year="1999"
- dateOfInjury "14041999" → day="14", month="04", year="1999"
- jobType: "סוג העבודה ירקנייה" → jobType="ירקנייה"
- accidentLocation: ☒ next to "במפעל" → accidentLocation="במפעל"
- Signature: "רועי" written under "חתימה", confirmed by "רועי יוחננוף" on שם המבקש → signature="קיימת"
- healthFundMember: ☒ next to "כללית" → healthFundMember="כללית\"""",
        '{"lastName":"יוחננוף","firstName":"רועי","idNumber":"033452156","gender":"זכר","dateOfBirth":{"day":"03","month":"03","year":"1974"},"address":{"street":"המאיר","houseNumber":"15","entrance":"1","apartment":"16","city":"אלוני הבשן","postalCode":"445412","poBox":""},"landlinePhone":"0975423541","mobilePhone":"0502451645","jobType":"ירקנייה","dateOfInjury":{"day":"14","month":"04","year":"1999"},"timeOfInjury":"15:30","accidentLocation":"במפעל","accidentAddress":"לוונברג 173 כפר סבא","accidentDescription":"במהלך העבודה הרמתי משקל כבד וכתוצאה מכך הייתי צריך ניתוח קילה","injuredBodyPart":"קילה","signature":"קיימת","formFillingDate":{"day":"20","month":"05","year":"1999"},"formReceiptDateAtClinic":{"day":"30","month":"06","year":"1999"},"medicalInstitutionFields":{"healthFundMember":"כללית","natureOfAccident":"","medicalDiagnoses":""}}',
    ),
]

FEW_SHOT_ENGLISH: list[tuple[str, str]] = [
    # ── Example D: English-filled, male, workplace accident, Maccabi ──
    # Date format note: this form was printed and rasterised; Azure DI OCRs each
    # date-box value as a separate token (year · month · day, left-to-right) rather
    # than the concatenated 8-digit string produced by handwritten Hebrew forms.
    (
        """\
OCR excerpt (real Layout API output — rasterised printed-English form):
  תאריך מילוי הטופס
  2024
  06
  10
  שנה חודש יום
  תאריך קבלת הטופס בקופה
  2024
  06
  11
  שנה חודש יום
  תאריך הפגיעה
  2024
  06
  10
  שנה חודש יום
  שם משפחה
  שם פרטי
  ת.ז.
  Smith
  John
  123456782
  מין
  ☒
  זכר
  ☐
  נקבה
  1985
  03
  15
  שנה חודש יום
  <table>
  <tr><th>רחוב / תא דואר</th><th>מס׳ בית</th><th>כניסה</th><th>דירה</th><th>יישוב</th><th>מיקוד</th></tr>
  <tr><td>42 Oak Street</td><td>42</td><td>A</td><td>3</td><td>Tel Aviv</td><td>6120001</td></tr>
  </table>
  טלפון נייד
  0521234567
  טלפון קווי
  036541234
  Construction Worker
  כאשר עבדתי ב
  14:30
  בשעה
  10/06/2024
  בתאריך
  תאונה בדרך ללא רכב
  ☐
  אחר
  ☐
  ת. דרכים בדרך לעבודה/מהעבודה
  ☐
  במפעל
  ☒
  ת. דרכים בעבודה
  ☐
  מקום התאונה:
  כתובת מקום התאונה
  42 Industrial Zone, Tel Aviv
  Fell from scaffolding on third floor, landed on left arm
  נסיבות הפגיעה / תאור התאונה
  האיבר שנפגע
  Left arm
  חתימה
  John Smith
  שם המבקש
  ☒
  מכבי
  ☐
  לאומית
  ☐
  מאוחדת
  ☐
  כללית
  Fracture of left radius
  מהות התאונה(אבחנות רפואיות):

Key parsing notes:
- Printed form (rasterised): each date box is a separate OCR token; the "שנה חודש יום"
  sub-label identifies order (left-to-right = year · month · day).
  So "2024 / 06 / 10" → year="2024", month="06", day="10"
- Form values are in English/Latin; all form labels remain in Hebrew
- Name area: "Smith" then "John" follow the Hebrew name labels; "שם המבקש John Smith"
  at bottom confirms → lastName="Smith", firstName="John"
- Gender: ☒ next to "זכר" → gender="זכר" (checkbox labels are always Hebrew)
- jobType: "Construction Worker" appears before "כאשר עבדתי ב" (RTL layout) → jobType="Construction Worker"
- Accident date "10/06/2024" → day="10", month="06", year="2024" (rule 4b slash-separated)
- accidentLocation: ☒ next to "במפעל" → accidentLocation="במפעל"
- Signature: "John Smith" near "חתימה" → signature="קיימת"
- healthFundMember: ☒ next to "מכבי" → healthFundMember="מכבי"
- medicalDiagnoses: "Fracture of left radius" after "מהות התאונה(אבחנות רפואיות):" → medicalDiagnoses="Fracture of left radius\"""",
        '{"lastName":"Smith","firstName":"John","idNumber":"123456782","gender":"זכר","dateOfBirth":{"day":"15","month":"03","year":"1985"},"address":{"street":"42 Oak Street","houseNumber":"42","entrance":"A","apartment":"3","city":"Tel Aviv","postalCode":"6120001","poBox":""},"landlinePhone":"036541234","mobilePhone":"0521234567","jobType":"Construction Worker","dateOfInjury":{"day":"10","month":"06","year":"2024"},"timeOfInjury":"14:30","accidentLocation":"במפעל","accidentAddress":"42 Industrial Zone, Tel Aviv","accidentDescription":"Fell from scaffolding on third floor, landed on left arm","injuredBodyPart":"Left arm","signature":"קיימת","formFillingDate":{"day":"10","month":"06","year":"2024"},"formReceiptDateAtClinic":{"day":"11","month":"06","year":"2024"},"medicalInstitutionFields":{"healthFundMember":"מכבי","natureOfAccident":"","medicalDiagnoses":"Fracture of left radius"}}',
    ),
    # ── Example E: English-filled, female, road-accident on way to work, Clalit ──
    (
        """\
OCR excerpt (real Layout API output — rasterised printed-English form):
  תאריך מילוי הטופס
  2023
  11
  03
  שנה חודש יום
  תאריך קבלת הטופס בקופה
  2023
  11
  04
  שנה חודש יום
  תאריך הפגיעה
  2023
  11
  03
  שנה חודש יום
  שם משפחה
  שם פרטי
  ת.ז.
  Johnson
  Sarah
  987654324
  מין
  ☐
  זכר
  ☒
  נקבה
  1990
  08
  22
  שנה חודש יום
  <table>
  <tr><th>רחוב / תא דואר</th><th>מס׳ בית</th><th>כניסה</th><th>דירה</th><th>יישוב</th><th>מיקוד</th></tr>
  <tr><td>15 Pine Avenue</td><td>15</td><td>B</td><td>7</td><td>Haifa</td><td>3200001</td></tr>
  </table>
  טלפון נייד
  0529876543
  טלפון קווי
  048765432
  Software Developer
  כאשר עבדתי ב
  09:15
  בשעה
  03/11/2023
  בתאריך
  תאונה בדרך ללא רכב
  ☐
  אחר
  ☐
  ת. דרכים בדרך לעבודה/מהעבודה
  ☒
  במפעל
  ☐
  ת. דרכים בעבודה
  ☐
  מקום התאונה:
  כתובת מקום התאונה
  Route 2 near Haifa interchange
  Slipped on wet entrance floor, injured right knee
  נסיבות הפגיעה / תאור התאונה
  האיבר שנפגע
  Right knee
  חתימה
  Sarah Johnson
  שם המבקש
  ☒
  כללית
  ☐
  מכבי
  ☐
  לאומית
  ☐
  מאוחדת
  Right knee sprain, grade II
  מהות התאונה(אבחנות רפואיות):

Key parsing notes:
- Printed form (rasterised): date tokens appear year · month · day (left-to-right per "שנה חודש יום")
  "2023 / 11 / 03" → year="2023", month="11", day="03"
  "1990 / 08 / 22" → year="1990", month="08", day="22"
- Form values in English; form labels in Hebrew
- Name: "Johnson" then "Sarah" in name area; "שם המבקש Sarah Johnson" confirms → lastName="Johnson", firstName="Sarah"
- Gender: ☒ next to "נקבה" → gender="נקבה"
- Accident date "03/11/2023" → day="03", month="11", year="2023" (rule 4b slash-separated)
- accidentLocation: ☒ next to "ת. דרכים בדרך לעבודה/מהעבודה" → accidentLocation="ת. דרכים בדרך לעבודה/מהעבודה"
- jobType: "Software Developer" before "כאשר עבדתי ב" (RTL) → jobType="Software Developer"
- Signature: "Sarah Johnson" near "חתימה" → signature="קיימת"
- healthFundMember: ☒ next to "כללית" → healthFundMember="כללית"
- medicalDiagnoses: "Right knee sprain, grade II" → medicalDiagnoses="Right knee sprain, grade II\"""",
        '{"lastName":"Johnson","firstName":"Sarah","idNumber":"987654324","gender":"נקבה","dateOfBirth":{"day":"22","month":"08","year":"1990"},"address":{"street":"15 Pine Avenue","houseNumber":"15","entrance":"B","apartment":"7","city":"Haifa","postalCode":"3200001","poBox":""},"landlinePhone":"048765432","mobilePhone":"0529876543","jobType":"Software Developer","dateOfInjury":{"day":"03","month":"11","year":"2023"},"timeOfInjury":"09:15","accidentLocation":"ת. דרכים בדרך לעבודה/מהעבודה","accidentAddress":"Route 2 near Haifa interchange","accidentDescription":"Slipped on wet entrance floor, injured right knee","injuredBodyPart":"Right knee","signature":"קיימת","formFillingDate":{"day":"03","month":"11","year":"2023"},"formReceiptDateAtClinic":{"day":"04","month":"11","year":"2023"},"medicalInstitutionFields":{"healthFundMember":"כללית","natureOfAccident":"","medicalDiagnoses":"Right knee sprain, grade II"}}',
    ),
]


# ---------------------------------------------------------------------------
# Self-review prompts (GPT-4o)
# ---------------------------------------------------------------------------

SELF_REVIEW_SYSTEM_PROMPT = """You are a data-quality reviewer for Israeli National Insurance \
form extractions (BL283).

You receive:
  - The raw OCR text of the form.
  - The JSON extracted from it.

Your job: identify fields where the extracted value clearly contradicts what the OCR text shows.

STRICT RULES:
1. Only flag a field if you can copy a short, exact substring from the OCR text that contradicts
   the extracted value. Put that substring in "ocr_quote". If you cannot find the exact quote,
   do NOT flag the field.
2. Never invent, guess, or paraphrase OCR content — only quote verbatim.
3. Do NOT flag fields that are legitimately empty because the form is blank there.
4. Do NOT flag minor normalisation differences (stripped dashes/spaces in phones, etc.).
5. Do NOT flag a field just because the same area of the form also contains other text
   that belongs to a different field (e.g. accidentAddress text near accidentLocation).

Return a JSON object with exactly this structure:
{
  "uncertain_fields": [
    { "field": "<fieldName>", "reason": "<short explanation>", "ocr_quote": "<exact OCR substring>" }
  ]
}

If everything looks correct, return: {"uncertain_fields": []}"""""

SELF_REVIEW_USER_PROMPT_TEMPLATE = """\
OCR text (may be truncated):
<ocr_text>
{ocr_text}
</ocr_text>

Extracted JSON:
<extracted_json>
{extracted_json}
</extracted_json>

List any fields whose extracted value is suspicious or inconsistent with the OCR text."""
