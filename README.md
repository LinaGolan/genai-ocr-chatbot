# KPMG GenAI Developer Assessment

Two independent systems that share one Azure resource layer, built with the
**native Azure OpenAI SDK only** (no LangChain or other LLM frameworks).

| Part | What it does | Stack |
|---|---|---|
| **Part 1** | Extracts structured JSON from ביטוח לאומי **BL283** forms (PDF/JPG) using Document Intelligence OCR + GPT-4o, with three-signal validation. | Streamlit |
| **Part 2** | Stateless microservice chatbot answering HMO medical-service questions (Maccabi / Meuhedet / Clalit) for the user's HMO + tier, in Hebrew or English. | FastAPI backend + Streamlit frontend |

Design docs live in [`docs/`](docs/): [shared infrastructure](docs/shared-infrastructure.md),
[Part 1 extraction](docs/part1-extraction.md), [Part 2 chatbot](docs/part2-chatbot.md).

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

Offline accuracy harness against labelled samples:

```powershell
python -m part1.evaluation.evaluate
```

## Part 2 — HMO Medical-Services Chatbot

A **stateless** FastAPI microservice with a Streamlit frontend. The frontend holds
all session state and sends the full conversation history (and confirmed user info)
with every request, so the backend keeps no per-user state and scales horizontally.

**Flow:** the bot first collects 8 fields through natural, LLM-driven conversation
(first/last name, ID, gender, age, HMO, card number, tier) with inline validation,
shows a confirmation card, then answers questions for the user's HMO **and tier** —
strictly from the knowledge base. If something isn't covered, it says so.

### Knowledge base & retrieval

The `phase2_data/*.html` pages are pre-rendered **offline** into one focused Markdown
file per (HMO, tier, topic) — `phase2_data/processed/<hmo>/<tier>/<topic>.md`, 54
files — by `build_knowledge_base.py` (GPT-4o, `temperature=0`, faithful copy of every
number/contact). Each file already holds only that HMO+tier's content, so retrieval
is cheap:

1. The user's HMO + tier select the **folder** — no search needed.
2. **ADA-002** picks the most relevant **topic** for the question.
3. **GPT-4o Mini** answers from that single file; if the answer isn't there it emits a
   sentinel and the backend **escalates** to **GPT-4o** over all topics in the folder.

The common case stays cheap (one small file + the mini model), while a wrong topic
guess or a cross-topic question still gets answered from the full HMO/tier context.

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
whether the answer escalated, and errors. ID numbers are SHA-256 hashed before
logging — **no raw PII is ever written**.
```
