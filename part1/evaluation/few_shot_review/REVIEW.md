# Few-shot review — OCR + ground truth

Real Azure Layout API OCR (`*.ocr.md`) and proposed ground-truth JSON (`*.gt.json`) for the
four source forms. Review/correct the `.gt.json` values, then I'll build the few-shots.

- **English (replace the 2 current synthetic D/E):** `283_en1` (PDF), `283_en4` (JPEG), `283_en2` (PDF)
- **Hebrew (new few-shot):** `283_ex4` (JPEG)

> Note: `283_en1`/`283_en2` are the *real* forms the current synthetic few-shots D/E were modelled
> on (same names/IDs), but several values in those synthetic examples are **wrong** vs the actual
> forms — see flags below. `283_en4` and `283_ex4` are the **same template filled twice** — once with
> Latin-script values (en4), once with Hebrew values (ex4); most fields are identical.

---

## 283_en1 — English, John Smith  (clean 9-digit ID, signed, workplace traffic accident)
All fields read with high confidence. Changes vs the old synthetic few-shot D:
- **accidentLocation = `ת. דרכים בעבודה`** (old D said `במפעל` — the X is on the 2nd box, verified by zoom).
- **healthFundMember = `מאוחדת`** (old D said `מכבי` — the X is on the מאוחדת box).
- ID is a clean 9 digits in the ת.ז box (ס״ב empty) → `123456782`.

## 283_en2 — English, Sarah Johnson  (female, no HMO box marked)
Changes vs the old synthetic few-shot E:
- **accidentLocation = `תאונה בדרך ללא רכב`** (old E said `ת. דרכים בדרך לעבודה/מהעבודה`).
- **healthFundMember = `""`** — on the real form **no HMO box is ticked**; only the
  `הנפגע חבר בקופת חולים` box is X'd, and the top `אל קופ״ח/ביה״ח ___` header is **blank**.
  (Old E said `כללית`.) → per rule 11, blank.

## 283_en4 — "English" variant of the shared template, golan / Lina
- ⚠️ **idNumber**: ת.ז box = `394055869` (9 digits) + ס״ב box = `1`. **OCR merges them → `3940558691`**
  (10 digits). GT uses the true 9-digit `394055869`. *(Decide: do you want the few-shot's expected
  output to show the merged 10 digits — like example C — or the clean 9?)*
- ⚠️ **dateOfInjury conflict**: section-1 `תאריך הפגיעה` box = **03/09/2024**, but the section-3
  free text reads `2.1.1990`. GT uses the section-1 dated field (03/09/2024). Confirm which you want.
- dateOfBirth box also reads `03/09/2024` (implausible year, but that's what's written).
- accidentLocation = `ת. דרכים בדרך לעבודה/מהעבודה` (✓ on box 3). **OCR mis-attributes the glyph to
  `סוג העבודה`/`לאומית`** — good teaching example.
- healthFundMember = `מכבי` (✓ on the מכבי box). **OCR reports `לאומית ☒`** (RTL mis-attribution).
- jobType = `MMD`; mobile `05566991110` (11 digits as written); landline `034566391`.
- accidentAddress is just `-` → `""`; description / injured body part / signature / diagnoses all blank.
- formReceiptDateAtClinic = 03/08/2024 (note: differs from ex4).

## 283_ex4 — Hebrew variant of the shared template
Same numbers as en4 (ID, phones, address numbers, dates, HMO, accident location, MMD). Differences:
- formReceiptDateAtClinic = **03/07/2024** (en4 was 03/08/2024).
- ⚠️ **Handwritten Hebrew — please verify these reads:**
  - lastName ≈ **`זינה`** (could be רינה/דינה)
  - firstName ≈ **`לוז`** (could be גוז/לוף)
  - street ≈ **`ניר`** (OCR read `מיר`)
  - city ≈ **`יס`** (OCR read `e.`)
- Same idNumber/date caveats as en4 above.
