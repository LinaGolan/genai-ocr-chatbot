# KPMG GenAI Developer Assessment

Two independent systems that share one Azure resource layer, built with the
**native Azure OpenAI SDK only** (no LangChain or other LLM frameworks).

| Part | What it does | Stack |
|---|---|---|
| **Part 1** | Extracts structured JSON from ביטוח לאומי **BL283** forms (PDF/JPG) using Document Intelligence OCR + GPT-4o, with three-signal validation. | Streamlit |
| **Part 2** | Stateless microservice chatbot answering HMO medical-service questions (Maccabi / Meuhedet / Clalit) for the user's HMO + tier, in Hebrew or English. | FastAPI backend + Streamlit frontend |

Design docs live in [`docs/`](docs/): [shared infrastructure](docs/shared-infrastructure.md),
[Part 1 extraction](docs/part1-extraction.md), [Part 2 chatbot](docs/part2-chatbot.md).

### Which model (or code) does what

Every step below is handled by exactly one of these — the design deliberately uses the
**cheapest tool that's reliable** for each job, and pushes anything deterministic to plain code.

| Tool | Where it's used |
|---|---|
| **Document Intelligence** (Layout API) | Part 1 — OCR of the form into Markdown + per-word confidence |
| **GPT-4o** | Part 1 — field extraction & vision re-read of weak fields · Part 2 — offline KB build + the Q&A "all-topics" escalation |
| **GPT-4o Mini** | Part 2 — collection (field extraction + chat reply) · default single-file Q&A · English→Hebrew query translation |
| **ADA-002** (embeddings) | Part 2 — picking the relevant topic for a question |
| **Plain code** (no model) | Part 1 — all validation · Part 2 — field validation, completion decision, language lock, HMO/tier routing |

---

## Prerequisites

- **Python 3.8+** (developed and tested on 3.8).
- Azure access to the pre-deployed resources: Document Intelligence, GPT-4o,
  GPT-4o Mini, and ADA-002. Credentials are provided with the assignment.

## Setup

```powershell
# 1. (recommended) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install dependencies
pip install -r requirements.txt

# 3. configure credentials — copy the example and fill in the values
copy .env.example .env   # then edit .env
```

`.env` (git-ignored) supplies all endpoints, keys, and deployment names — nothing
is hardcoded in source. Required variables:

```
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=
AZURE_DOCUMENT_INTELLIGENCE_KEY=

AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
AZURE_OPENAI_API_VERSION=

AZURE_OPENAI_GPT4O_DEPLOYMENT=
AZURE_OPENAI_GPT4O_MINI_DEPLOYMENT=
AZURE_OPENAI_ADA_DEPLOYMENT=
```

The shared client (`shared/azure_client.py`) reads these once at import and fails
fast with a clear error if any are missing.

---

## Part 1 — BL283 Form Extraction

```powershell
streamlit run part1/frontend/app.py
```

Upload a PDF/JPG of a BL283 form. The UI shows the raw OCR (left) and the
extracted JSON with per-field validation highlights (right). Hebrew and English
forms are both supported; missing fields are returned as empty strings.

### How it works (upload → JSON)

```
   PDF / JPG upload
        │
   ① file check ─────────── code            (type, size, page-count limits)
        │
   ② OCR ────────────────── Document Intelligence (Layout API)
        │                                    → Markdown + per-word confidence
   ③ extract fields ─────── GPT-4o           (OCR Markdown → JSON, Structured
        │                                      Outputs, temperature 0)
   ④ vision re-read ─────── GPT-4o (vision)  (only fields with low OCR
        │                                      confidence or that fail a check —
        │                                      re-read straight from the image)
   ⑤ validate ───────────── code            (format + cross-field + confidence)
        │
   ⑥ show result ────────── Streamlit        (raw OCR + JSON + ERROR/CHECK badges)
```

Each value is always kept and shown as-is; problems are surfaced as badges, never
silently "fixed" or dropped. Steps ③–④ are the only LLM calls; everything that can be
decided by a rule (③ aside) is done in code.

### Validation

Validation is **deterministic-first**: a field's extracted value is always kept and
shown as-is, with any problem surfaced inline as an `ERROR` (invalid) or `CHECK`
(uncertain) badge plus a one-line reason — e.g. a 10-digit ID is displayed unchanged
with *"ID must be exactly 9 digits"*. Low-confidence OCR reads are first repaired by
the vision corrector (GPT-4o re-reads them from the image); the remaining checks are:

**Field-format checks**
- **ID number** — must be exactly 9 digits (length only; no check-digit/checksum) → `invalid`.
- **Dates** (birth / injury / filling / receipt) — day 1–31, month 1–12, plausible year 1900–2100 → `invalid`.
- **Phones** — Israeli landline/mobile format → `uncertain`.
- **Postal code** — up to 7 digits → `uncertain`.
- **Gender** — must be `זכר` / `נקבה` → `uncertain`.
- **Time of injury** — must parse to a valid `HH:MM` (00–23 / 00–59) → `uncertain`.

**Cross-field & logical checks**
- **Date ordering** — date of birth before date of injury; injury on/before form filling date → `uncertain`.
- **Form receipt date at clinic** — cannot be before the injury date or before the form filling date → `uncertain`.
- **Future dates** — no form date may be later than today → `uncertain`.
- **Required identity fields** — ID number, last name, first name flagged when empty → `uncertain`.

**OCR-confidence signal** — fields whose source words scored below 0.70 confidence (and
weren't vision-verified) are flagged `uncertain`. These signals feed an overall
`high` / `medium` / `low` accuracy estimate (critical-field or multi-field errors → `low`).

Offline accuracy harness against labelled samples:

```powershell
python -m part1.evaluation.evaluate
```

## Part 2 — HMO Medical-Services Chatbot

A **stateless** FastAPI microservice with a Streamlit frontend. The frontend holds
all session state and sends the full conversation history (and confirmed user info)
with every request, so the backend keeps no per-user state and scales horizontally.

There are two phases: **collection** (gather 8 user fields by conversation) and
**Q&A** (answer from the knowledge base for that user's HMO + tier).

### Phase 1 — collection (per user turn)

The conversation is LLM-driven, but every value is checked by code — so the cheap
mini model can drive it without being trusted to count digits (which it gets wrong).

```
   user message
        │
   ① extract the 8 fields so far ── GPT-4o Mini   (dedicated JSON call — the
        │                                           authoritative state)
   ② validate + decide done ─────── code           (9-digit ID/card, age 0–120,
        │                                           HMO/tier values; all 8 valid?)
   ③ chat reply ─────────────────── GPT-4o Mini    (asks for what's missing, or to
        │                                           fix a rejected value — steered
        │                                           by a code-built status note)
        ▼
   all 8 valid?  ──► confirmation card ── Streamlit  (user reviews & edits, then
                                                       confirms → Phase 2)
```

Reply language is **locked to the user's first message** (code), and the field choices
(HMO/tier) are shown in that language. The model never decides validity or completion —
code does both.

### Phase 2 — Q&A (per question)

```
   user question
        │
   ① (English only) → Hebrew keywords ── GPT-4o Mini   (KB is Hebrew; helps matching)
        │
   ② pick the topic ─────────────────── ADA-002         (embed query, rank 6 topics)
        │  + pick the folder ─────────── code            (from the user's HMO + tier)
        │
   ③ answer from that one file ──────── GPT-4o Mini     (or emit a sentinel if the
        │                                                 answer isn't in it)
        │
   ④ if insufficient → answer over ──── GPT-4o           (all 6 topic files for the
        all topics in the folder                          user's HMO/tier)
```

Answers come **only** from the knowledge base, already scoped to the user's HMO **and
tier**; if it's not covered, the bot says so. Response language is forced to match the
question (code-computed directive).

### Where the knowledge base comes from (offline build)

The Q&A flow above is cheap because the knowledge is pre-shaped **once, offline**. The
`phase2_data/*.html` pages (each covering all 3 HMOs × 3 tiers in one table) are
rewritten by `build_knowledge_base.py` (**GPT-4o**, `temperature=0`, faithful copy of
every number/contact) into one focused Markdown file per (HMO, tier, topic):
`phase2_data/processed/<hmo>/<tier>/<topic>.md` — **54 files**.

Because each file already holds only that HMO+tier's content, request-time retrieval is
just "pick the folder (code) → pick the topic (ADA-002) → answer (Mini, escalate to
GPT-4o)" as shown above — no per-request filtering, and the common case stays cheap.

### Run

```powershell
# 0. one-time: build the Markdown knowledge base (rerun if the source HTML changes)
python -m part2.backend.build_knowledge_base

# Terminal 1 — backend (embeds the 6 topics at startup)
uvicorn part2.backend.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run part2/frontend/app.py
```

If the backend runs somewhere other than `http://localhost:8000`, point the frontend
at it via the `CHATBOT_BACKEND_URL` environment variable.

API (also documented at `http://localhost:8000/docs`):

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat/collect` | Advance info collection; returns `user_info` once complete |
| `POST` | `/api/chat/qa` | Answer a question from the user's HMO/tier knowledge |
| `GET`  | `/health` | Liveness + topic-index status |

---

## Logging

All activity is logged as JSON lines via the shared `get_logger` factory
(`logs/part1.log`, `logs/part2.log`): requests, latency, the selected topic and
whether the answer escalated, and errors. Every LLM call also logs **which model
served it, token usage, and the `finish_reason`** (with a warning if a response was
truncated or content-filtered) — so the model/cost split described above is
measurable, not just claimed. ID numbers are SHA-256 hashed before logging —
**no raw PII is ever written**.
```
