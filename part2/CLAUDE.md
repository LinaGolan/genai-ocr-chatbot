# Part 2 — Microservice Chatbot

Stateless FastAPI chatbot answering HMO medical-service questions (Maccabi / Meuhedet / Clalit). Full design: @docs/part2-chatbot.md

# Structure
- `backend/main.py` — FastAPI app + endpoints (`/api/chat/collect`, `/api/chat/qa`, `/health`)
- `backend/chat_service.py` — LLM orchestration for both phases (incl. the Q&A single-file → all-files escalation)
- `backend/build_knowledge_base.py` — OFFLINE build: GPT-4o rewrites each `phase2_data/*.html` into per-(HMO, tier) Markdown under `phase2_data/processed/<hmo>/<tier>/<topic>.md` (54 files)
- `backend/knowledge_base.py` — loads the `processed/` Markdown tree; ADA-002 topic-selection embeddings + file readers
- `backend/prompts.py` — all prompt text (collection, Q&A single-file, Q&A fallback, KB-build, query translation)
- `frontend/app.py` — Streamlit UI; state in `st.session_state`
- `frontend/api_client.py` — the ONLY place that makes HTTP calls to the backend (httpx, async)

# Hard constraints specific to Part 2
- **The backend is STATELESS.** No DB, no in-memory session store. Every request carries full `conversation_history` + `user_info`. Derive everything from the payload.
- **Info collection is LLM-driven.** No hardcoded question-answer logic or form-based UI filling. The LLM decides what to ask and validates inline.
- All endpoints are `async def`; all Azure calls are awaited (concurrent users must not block).
- Q&A answers come ONLY from the retrieved Markdown content, already filtered to the user's HMO **and** tier. If not in the KB, say so.
- The knowledge base is the pre-built `phase2_data/processed/` Markdown tree — regenerate it with `build_knowledge_base.py` if the source HTML changes. Retrieval = pick the `<hmo>/<tier>` folder from `user_info`, then select the topic (ADA-002), with a single-file → all-files answer escalation.
- Supports Hebrew and English — match the user's language (the backend injects an explicit response-language directive computed in code).

# Collection completion signal
When all 8 fields are gathered, the collection LLM emits `<user_info>{...}</user_info>`; the backend parses this to advance to the confirmation phase. Do not change this contract without updating both backend and frontend.

# Run
- Build KB (offline, once / when HTML changes): `python -m part2.backend.build_knowledge_base`
- Backend: `uvicorn part2.backend.main:app --reload --port 8000`
- Frontend: `streamlit run part2/frontend/app.py`
