# Part 1 — Form Field Extraction

Extract the assignment's target JSON from an uploaded PDF/JPG of the BL283 form. Full design: @docs/part1-extraction.md

# Structure
- `backend/schema.py` — all part1 data models: the Pydantic output schema (`FormExtraction` + nested), the single source of truth for the JSON shape shared by extractor and validator; plus the validation result types (`FieldStatus`, `ValidationResult`)
- `backend/ocr_client.py` — Document Intelligence **Layout API** wrapper (not Read/General — the form has tables and checkboxes); returns markdown + confidence scores
- `backend/extractor.py` — GPT-4o extraction, OCR markdown → JSON
- `backend/vision_corrector.py` — `correct_and_validate`: re-reads fields that OCR read with confidence < 0.70 **or** that fail deterministic validation from the source image with GPT-4o vision, writes verified values back, then validates again (replaces the old text self-review). Fields still failing after the re-read are kept as-is with the issue noted in logs + the validation reason. Runs between extractor and the final validation.
- `backend/validator.py` — two-signal validation: deterministic checks + OCR confidence (vision-verified fields arrive as `trusted_fields` and skip the low-confidence flag but still face deterministic checks). Exposes `failing_fields()` so the corrector knows which fields to re-read.
- `backend/prompts.py` — all prompt text
- `frontend/app.py` — Streamlit: raw OCR (left) + JSON viewer with confidence highlights (right)
- `evaluation/` — offline accuracy harness + ground-truth JSON for all 6 `phase1_data` samples (3 Hebrew + 3 English). `generate_english_samples.py` creates the English PDFs and their canonical ground-truth JSONs; ground-truth files live in `evaluation/ground_truth/` named after their PDF stem.

# Constraints specific to Part 1
- Forms come in **Hebrew or English** — extract regardless of language.
- For any field not present or not legible, output an **empty string** (never null, never omitted). Never guess/invent values.
- Extraction runs at **`temperature=0`**. Prefer Structured Outputs (`response_format` `json_schema`, `strict`); fall back to `{"type":"json_object"}` only on older API versions.
- The output JSON shape is fixed by the assignment — match it exactly (see `schema.py` Pydantic models, the single source of truth).
- **Validation is deterministic-first.** The Israeli `idNumber` MUST be exactly 9 digits (length is the only rule — no check-digit/checksum); dates/phones/postal validated by format. Low-confidence fields are repaired upstream by `vision_corrector.py` (GPT-4o re-reads them from the image); vision-corrected values still pass through the deterministic checks and must NOT override a deterministic `invalid`.
- Accuracy is **measured offline** via `evaluation/` against labeled samples — not asserted by the model at runtime.
- Validate uploads (type/size/pages) and retry transient Azure errors with backoff. Never log raw extracted PII values.
- Azure calls here are synchronous (single-user Streamlit app); do not add async complexity.

# Run
`streamlit run part1/frontend/app.py`
