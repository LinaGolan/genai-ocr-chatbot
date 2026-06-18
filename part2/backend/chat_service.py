from __future__ import annotations

"""
LLM orchestration for both chatbot phases.

  run_collection() — two GPT-4o Mini calls per turn: a dedicated json_object
                     extraction call produces the authoritative field state, which
                     a code-side validator (collection_validation.py) checks; a
                     second conversational call, steered by a status note built from
                     that result, produces the user-facing reply. Mini never judges
                     values; the backend decides completion when all 8 are valid.
  run_qa()         — GPT-4o Mini answers from a single retrieved topic file, with
                     escalation to GPT-4o over all of the HMO/tier's files.

Model strategy: keep the cheap GPT-4o Mini on the common paths (collection, query
translation, single-file Q&A) and spend GPT-4o only on the hard cross-topic Q&A
fallback. Mini is made reliable on collection by moving the deterministic checks
(9-digit ID/card, age 0–120, HMO/tier enum) into code instead of trusting the
model to count digits — see collection_validation.py.

The backend stays stateless: every call receives the full conversation history
and (for Q&A) the confirmed user_info from the request payload.
"""

import json
import re
import time
from typing import Any

from shared.azure_client import (
    async_openai_client,
    GPT4O_DEPLOYMENT,
    GPT4O_MINI_DEPLOYMENT,
)
from shared.logger import get_logger, hash_id
from part2.backend import knowledge_base as kb
from part2.backend.collection_validation import (
    field_errors,
    missing_fields,
    is_complete_and_valid,
    format_validation_feedback,
)
from part2.backend.prompts import (
    COLLECTION_SYSTEM_PROMPT,
    COLLECTION_EXTRACTION_PROMPT,
    QA_SYSTEM_PROMPT_TEMPLATE,
    QA_SINGLE_TOPIC_PROMPT_TEMPLATE,
    INSUFFICIENT_CONTEXT_SIGNAL,
    QUERY_TRANSLATION_PROMPT,
    NO_KNOWLEDGE_PLACEHOLDER,
)

logger = get_logger(__name__)

_REQUIRED_FIELDS = (
    "firstName", "lastName", "idNumber", "gender",
    "age", "hmo", "hmoCardNumber", "insuranceTier",
)


# ---------------------------------------------------------------------------
# Phase 1 — collection
# ---------------------------------------------------------------------------

async def run_collection(
    history: list[dict[str, str]],
    user_message: str,
) -> dict[str, Any]:
    """
    Advance the info-collection conversation by one turn.

    Returns:
      {
        "reply": str,                  # assistant text shown to the user
        "user_info": dict | None,      # populated only once all fields gathered
        "phase": "collection" | "confirmation",
      }
    """
    convo = _sanitize_history(history) + [{"role": "user", "content": user_message}]

    # Two cheap GPT-4o Mini calls per turn, decoupling extraction from chat:
    #   1. _extract_state — a dedicated json_object call whose output IS the
    #      authoritative state. The conversational model used to "forget" to keep a
    #      running block up to date (so an 8-digit ID surfaced only at the end);
    #      doing extraction as its own focused task fixes that.
    #   2. validate the extracted state in code (digit/range/enum), then
    #   3. _collection_reply — the user-facing turn, STEERED by a status note built
    #      from the validation result (ask next fields / fix invalid ones / wrap up).
    # Completion is decided here in code, never by the chat model.
    t0 = time.perf_counter()

    state = await _extract_state(convo)
    errors = field_errors(state)
    missing = missing_fields(state)
    complete = is_complete_and_valid(state)

    reply, response = await _collection_reply(convo, history, user_message, errors, missing, complete)
    user_info = _normalize_user_info(state) if complete else None

    logger.info(
        "Collection turn complete",
        extra={
            "phase": "confirmation" if complete else "collection",
            "complete": complete,
            "invalid_fields": [e.split(":", 1)[0] for e in errors] or None,
            "missing": missing or None,
            "latency_s": round(time.perf_counter() - t0, 2),
            "tokens": _usage(response),
        },
    )
    return {
        "reply": reply,
        "user_info": user_info,
        "phase": "confirmation" if complete else "collection",
    }


async def _extract_state(convo: list[dict[str, str]]) -> dict:
    """
    Extract the 8 collected fields from the conversation with a dedicated
    json_object Mini call. This output — not the chat model's prose — is the
    authoritative state the backend validates. On any failure, return an all-empty
    state so collection simply continues rather than crashing.
    """
    messages = [{"role": "system", "content": COLLECTION_EXTRACTION_PROMPT}] + convo
    try:
        response = await async_openai_client.chat.completions.create(
            model=GPT4O_MINI_DEPLOYMENT,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        _log_response(response, "collection.extract")
        data = json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001 — collection must still proceed
        logger.warning("State extraction failed; treating as empty", extra={"error": str(exc)[:150]})
        return {f: "" for f in _REQUIRED_FIELDS}
    if not isinstance(data, dict):
        return {f: "" for f in _REQUIRED_FIELDS}
    return {f: data.get(f, "") for f in _REQUIRED_FIELDS}


async def _collection_reply(
    convo: list[dict[str, str]],
    history: list[dict[str, str]],
    user_message: str,
    errors: list[str],
    missing: list[str],
    complete: bool,
) -> tuple[str, Any]:
    """
    Produce the user-facing collection turn, steered by a code-built STATUS note
    (validation errors / still-missing fields / complete) plus a code-computed
    language directive. Returns (reply, response).
    """
    system_prompt = (
        COLLECTION_SYSTEM_PROMPT
        + "\n\n=== STATUS (authoritative — follow this) ===\n"
        + _collection_status_directive(errors, missing, complete)
        + "\n\n=== RESPONSE LANGUAGE (authoritative — overrides everything above) ===\n"
        + _collection_language_directive(history, user_message)
    )
    response = await async_openai_client.chat.completions.create(
        model=GPT4O_MINI_DEPLOYMENT,
        messages=[{"role": "system", "content": system_prompt}] + convo,
        temperature=0.3,
    )
    _log_response(response, "collection.reply")
    return (response.choices[0].message.content or "").strip(), response


def _collection_status_directive(errors: list[str], missing: list[str], complete: bool) -> str:
    """The per-turn instruction telling the chat model what to do, from validation."""
    if errors:
        return format_validation_feedback(errors)
    if complete:
        return (
            "All 8 required fields are collected and valid. Reply with ONE short, warm "
            "sentence telling the user that's everything and they can review their details "
            "in the summary below. Do NOT list the values back and do NOT ask them to confirm "
            "— the app shows a confirmation screen for that. Ask no further questions."
        )
    return (
        "Collection is still in progress. Fields still NEEDED (internal keys): "
        f"{', '.join(missing)}. Ask the user for ONE or TWO of these still-missing fields, "
        "conversationally and in their language. Never re-ask for a field already provided."
    )


def _normalize_user_info(data: dict) -> dict:
    """Coerce types/strings into the canonical shape used downstream."""
    out = {f: data.get(f, "") for f in _REQUIRED_FIELDS}
    out["firstName"] = str(out["firstName"]).strip()
    out["lastName"] = str(out["lastName"]).strip()
    out["idNumber"] = str(out["idNumber"]).strip()
    out["gender"] = str(out["gender"]).strip()
    out["hmo"] = str(out["hmo"]).strip()
    out["hmoCardNumber"] = str(out["hmoCardNumber"]).strip()
    out["insuranceTier"] = str(out["insuranceTier"]).strip()
    try:
        out["age"] = int(out["age"])
    except (TypeError, ValueError):
        out["age"] = str(out["age"]).strip()
    return out


# ---------------------------------------------------------------------------
# Phase 2 — Q&A
# ---------------------------------------------------------------------------

async def run_qa(
    history: list[dict[str, str]],
    user_info: dict[str, Any],
    user_message: str,
) -> dict[str, Any]:
    """
    Answer a medical-services question, scoped to the user's HMO + tier folder.

    Two-tier escalation:
      1. ADA-002 selects the single most relevant topic; GPT-4o Mini tries to
         answer from just that one (already HMO/tier-filtered) file.
      2. If Mini signals the answer isn't in that file, fall back to GPT-4o over
         ALL topic files for the user's HMO/tier.

    Returns {"reply": str}.
    """
    hmo = str(user_info.get("hmo", "")).strip()
    tier = str(user_info.get("insuranceTier", "")).strip()
    t0 = time.perf_counter()

    # 1. Pick the single best topic. English questions are distilled to Hebrew
    #    keywords first — the topic embeddings are Hebrew and cross-lingual cosine
    #    is weak.
    search_query = await _search_query(user_message)
    topic = await kb.select_topic(search_query)
    topic_slug = topic.slug if topic else None

    # 2. Cheap path: GPT-4o Mini answers from that one file — unless it says no.
    single = kb.load_topic_content(hmo, tier, topic_slug) if topic_slug else ""
    reply = await _answer_from_single(history, user_info, user_message, single)

    if reply is not None:
        escalated, answer_model = False, "gpt-4o-mini"
    else:
        # 3. Fallback: GPT-4o over every topic file for this HMO/tier.
        all_content = kb.load_all_topics(hmo, tier)
        reply = await _answer_from_all(history, user_info, user_message, all_content)
        escalated, answer_model = True, "gpt-4o"

    logger.info(
        "Q&A turn complete",
        extra={
            "user_hash": hash_id(str(user_info.get("idNumber", ""))),
            "hmo": hmo,
            "tier": tier,
            "selected_topic": topic_slug,
            "escalated": escalated,
            "answer_model": answer_model,
            "latency_s": round(time.perf_counter() - t0, 2),
        },
    )
    return {"reply": reply}


async def _answer_from_single(
    history: list[dict[str, str]],
    user_info: dict[str, Any],
    user_message: str,
    content: str,
) -> str | None:
    """
    Try to answer from a single topic file with GPT-4o Mini. Returns the answer,
    or None to signal escalation (no content, or Mini emitted the sentinel).
    """
    if not content.strip():
        return None

    system_prompt = QA_SINGLE_TOPIC_PROMPT_TEMPLATE.format(
        knowledge=content,
        insufficient_signal=INSUFFICIENT_CONTEXT_SIGNAL,
        language_directive=_language_directive(user_message),
        **_profile_kwargs(user_info),
    )
    messages = (
        [{"role": "system", "content": system_prompt}]
        + _sanitize_history(history)
        + [{"role": "user", "content": user_message}]
    )
    response = await async_openai_client.chat.completions.create(
        model=GPT4O_MINI_DEPLOYMENT,
        messages=messages,
        temperature=0.2,
    )
    _log_response(response, "qa.single")
    raw = (response.choices[0].message.content or "").strip()
    if not raw or INSUFFICIENT_CONTEXT_SIGNAL in raw:
        return None
    return raw


async def _answer_from_all(
    history: list[dict[str, str]],
    user_info: dict[str, Any],
    user_message: str,
    all_content: dict[str, str],
) -> str:
    """Answer from every topic file for the user's HMO/tier with GPT-4o."""
    system_prompt = QA_SYSTEM_PROMPT_TEMPLATE.format(
        knowledge=_format_all_topics(all_content),
        language_directive=_language_directive(user_message),
        **_profile_kwargs(user_info),
    )
    messages = (
        [{"role": "system", "content": system_prompt}]
        + _sanitize_history(history)
        + [{"role": "user", "content": user_message}]
    )
    response = await async_openai_client.chat.completions.create(
        model=GPT4O_DEPLOYMENT,
        messages=messages,
        temperature=0.2,
    )
    _log_response(response, "qa.fallback")
    return (response.choices[0].message.content or "").strip()


def _profile_kwargs(user_info: dict[str, Any]) -> dict[str, Any]:
    """The user-profile fields shared by both Q&A prompt templates."""
    return {
        "first_name": user_info.get("firstName", ""),
        "last_name": user_info.get("lastName", ""),
        "hmo": str(user_info.get("hmo", "")).strip() or "—",
        "tier": user_info.get("insuranceTier", "") or "—",
        "age": user_info.get("age", ""),
        "gender": user_info.get("gender", ""),
    }


_LATIN_RE = re.compile(r"[A-Za-z]")
_HEBREW_RE = re.compile(r"[֐-׿]")


def _language_directive(user_message: str) -> str:
    """
    A deterministic 'answer in language X' instruction for the Q&A prompts.

    The knowledge base is entirely Hebrew and the models (GPT-4o Mini especially)
    tend to drift into Hebrew when the context is Hebrew, ignoring a softer
    'mirror the user' instruction. We detect the question's script in code (same
    Latin-letter signal that triggers query translation) and command the language
    explicitly, which the models follow reliably.
    """
    if _LATIN_RE.search(user_message):
        return "Write your ENTIRE reply in English, regardless of the Hebrew knowledge content."
    return "כתוב את כל התשובה בעברית בלבד."


def _is_english_conversation(history: list[dict[str, str]], user_message: str) -> bool:
    """
    Decide the collection reply language ONCE, from the dominant script of the
    user's FIRST lettered message, and stick with it for the whole session. The
    backend is stateless, so we recompute this each turn from the full history —
    but because we always read the earliest lettered user message, the verdict is
    stable and won't flip if a later answer happens to be in the other script or
    is digits-only (an ID, a card number, an age carry no script). Defaults to
    Hebrew when nothing decisive has been said yet (e.g. the opening greeting).
    """
    texts = [t["content"] for t in _sanitize_history(history) if t["role"] == "user"]
    texts.append(user_message)
    for text in texts:
        latin = len(_LATIN_RE.findall(text))
        hebrew = len(_HEBREW_RE.findall(text))
        if latin or hebrew:
            return latin > hebrew
    return False


def _collection_language_directive(history: list[dict[str, str]], user_message: str) -> str:
    """
    Code-computed reply-language command for collection — the same proven fix the
    Q&A path uses. GPT-4o Mini otherwise drifts into Hebrew because the collection
    prompt's field labels are Hebrew, ignoring the softer 'mirror the user' line.
    """
    if _is_english_conversation(history, user_message):
        return "Write your ENTIRE reply to the user in English, even though the field labels above are in Hebrew."
    return "כתוב את כל הפנייה אל המשתמש בעברית בלבד."


async def _search_query(user_message: str) -> str:
    """
    Hebrew/other-script questions are used verbatim for retrieval. Questions that
    contain Latin letters (English) are distilled into Hebrew search keywords —
    the KB is Hebrew and ADA-002 cross-lingual matching is too weak otherwise.
    Falls back to the original text on any translation error.
    """
    if not _LATIN_RE.search(user_message):
        return user_message
    try:
        # GPT-4o Mini suffices for this narrow noun-extraction task — no digit
        # counting or reasoning, just a few Hebrew keywords for the embedding step.
        response = await async_openai_client.chat.completions.create(
            model=GPT4O_MINI_DEPLOYMENT,
            temperature=0,
            messages=[
                {"role": "system", "content": QUERY_TRANSLATION_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        _log_response(response, "qa.translate")
        translated = (response.choices[0].message.content or "").strip()
        return translated or user_message
    except Exception as exc:  # noqa: BLE001 — retrieval must still proceed
        logger.warning("Query translation failed; using original text", extra={"error": str(exc)[:150]})
        return user_message


def _format_all_topics(all_content: dict[str, str]) -> str:
    """Join every topic file for the HMO/tier into the {knowledge} section. Each
    file already carries its own H1 title, so we just delimit them."""
    usable = [c for c in all_content.values() if c.strip()]
    if not usable:
        return NO_KNOWLEDGE_PLACEHOLDER
    return "\n\n---\n\n".join(usable)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sanitize_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only well-formed {role, content} turns with valid roles."""
    clean: list[dict[str, str]] = []
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            clean.append({"role": role, "content": content})
    return clean


def _usage(response) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    out = {
        "prompt": usage.prompt_tokens,
        "completion": usage.completion_tokens,
        "total": usage.total_tokens,
    }
    # Prompt-cache hit count, when the API reports it (saves cost on repeated
    # system prompts). Absent on older API versions, so guard for it.
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details else None
    if cached:
        out["cached"] = cached
    return out


def _log_response(response, call: str):
    """
    Per-call LLM telemetry — model, token usage, and finish_reason — plus a warning
    when the model truncated its output (finish_reason 'length') or Azure's content
    filter stopped it. Purely observational: returns the response unchanged and
    never raises, so it can sit beside any call without affecting the flow.
    """
    try:
        finish = getattr(response.choices[0], "finish_reason", None)
    except (AttributeError, IndexError):
        finish = None

    logger.info(
        "LLM call",
        extra={
            "call": call,
            "model": getattr(response, "model", None),
            "finish_reason": finish,
            "tokens": _usage(response),
        },
    )
    if finish == "length":
        logger.warning("LLM output was truncated at the token limit", extra={"call": call})
    elif finish == "content_filter":
        logger.warning("LLM output was stopped by the content filter", extra={"call": call})
    return response
