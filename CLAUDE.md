# KPMG GenAI Assessment

Two independent systems sharing Azure resources:
- **part1/** — extract structured JSON from ביטוח לאומי (BL283) forms via OCR + LLM. See @part1/CLAUDE.md
- **part2/** — stateless microservice chatbot for HMO medical services. See @part2/CLAUDE.md

Design docs (read before non-trivial changes): @docs/shared-infrastructure.md, @docs/part1-extraction.md, @docs/part2-chatbot.md

# Hard constraints
- **YOU MUST use the native Azure OpenAI SDK only. NO LangChain or other LLM frameworks.** This is a graded requirement.
- All Azure SDK clients are created ONLY in `shared/azure_client.py`. Never instantiate `AzureOpenAI` or Document Intelligence clients elsewhere — import from there.
- No credentials, endpoints, or deployment names in source. Read them from env vars (loaded from `.env`). Copy `.env.example` → `.env` to configure.

# Layout convention
- Each part splits into `backend/` and `frontend/`.
- HTTP transport lives in a dedicated client file (`part2/frontend/api_client.py`) — never inline raw HTTP in UI code.
- All prompt text lives in `prompts.py` within each part's backend — not inlined in logic.

# Commands (PowerShell, Windows)
- Install: `pip install -r requirements.txt`
- Part 1 UI: `streamlit run part1/frontend/app.py`
- Part 2 backend: `uvicorn part2.backend.main:app --reload --port 8000`
- Part 2 UI: `streamlit run part2/frontend/app.py`

# Git
- **Do NOT set or override `git config user.name` / `user.email`** — the global git config is already set to the user's identity; touching it would break the signature.
- Commits may be made when asked — the global identity is correct.

# Conventions
- Logging goes through the shared `get_logger(name)` factory; logs are JSON lines to `logs/`. Never log raw PII (hash ID numbers).
- Supports Hebrew and English input throughout — never assume Latin script.
