# Part 1 — Form Field Extraction

Extract the assignment's target JSON from an uploaded PDF/JPG of the BL283 form. Full design: @docs/part1-extraction.md

# Structure
- `backend/ocr_client.py` — Document Intelligence **Layout API** wrapper (not Read/General — the form has tables and checkboxes); returns markdown + confidence scores
- `backend/extractor.py` — GPT-4o extraction, OCR markdown → JSON
- `backend/validator.py` — three-signal validation: deterministic checks + OCR confidence + GPT-4o Mini self-review
- `backend/prompts.py` — all prompt text
- `frontend/app.py` — Streamlit: raw OCR (left) + JSON viewer with confidence highlights (right)
- `evaluation/` — offline accuracy harness + ground-truth JSON for the `phase1_data` samples

# Constraints specific to Part 1
- Forms come in **Hebrew or English** — extract regardless of language.
- For any field not present or not legible, output an **empty string** (never null, never omitted). Never guess/invent values.
- Extraction runs at **`temperature=0`**. Prefer Structured Outputs (`response_format` `json_schema`, `strict`); fall back to `{"type":"json_object"}` only on older API versions.
- The output JSON shape is fixed by the assignment — match it exactly (see `validator.py` Pydantic schema, the single source of truth).
- **Validation is deterministic-first.** The Israeli `idNumber` MUST be checked with its check-digit (checksum) algorithm; dates/phones/postal validated by format. The LLM self-review is supplementary and must NOT override a deterministic `invalid`.
- Accuracy is **measured offline** via `evaluation/` against labeled samples — not asserted by the model at runtime.
- Validate uploads (type/size/pages) and retry transient Azure errors with backoff. Never log raw extracted PII values.
- Azure calls here are synchronous (single-user Streamlit app); do not add async complexity.

# Run
`streamlit run part1/frontend/app.py`
