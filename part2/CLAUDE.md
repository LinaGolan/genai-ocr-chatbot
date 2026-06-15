# Part 2 — Microservice Chatbot

Stateless FastAPI chatbot answering HMO medical-service questions (Maccabi / Meuhedet / Clalit). Full design: @docs/part2-chatbot.md

# Structure
- `backend/main.py` — FastAPI app + endpoints (`/api/chat/collect`, `/api/chat/qa`, `/health`)
- `backend/chat_service.py` — LLM orchestration for both phases
- `backend/knowledge_base.py` — HTML parsing, per-HMO content split, ADA-002 heading embeddings, retrieval
- `backend/prompts.py` — all prompt text (separate prompts for collection vs Q&A)
- `frontend/app.py` — Streamlit UI; state in `st.session_state`
- `frontend/api_client.py` — the ONLY place that makes HTTP calls to the backend (httpx, async)

# Hard constraints specific to Part 2
- **The backend is STATELESS.** No DB, no in-memory session store. Every request carries full `conversation_history` + `user_info`. Derive everything from the payload.
- **Info collection is LLM-driven.** No hardcoded question-answer logic or form-based UI filling. The LLM decides what to ask and validates inline.
- All endpoints are `async def`; all Azure calls are awaited (concurrent users must not block).
- Q&A answers come ONLY from retrieved knowledge-base content, filtered to the user's HMO. If not in the KB, say so.
- Supports Hebrew and English — match the user's language.

# Collection completion signal
When all 8 fields are gathered, the collection LLM emits `<user_info>{...}</user_info>`; the backend parses this to advance to the confirmation phase. Do not change this contract without updating both backend and frontend.

# Run
- Backend: `uvicorn part2.backend.main:app --reload --port 8000`
- Frontend: `streamlit run part2/frontend/app.py`
