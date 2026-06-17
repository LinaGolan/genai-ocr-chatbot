from __future__ import annotations

"""
LLM orchestration for both chatbot phases.

  run_collection() — GPT-4o Mini drives the 8-field info-collection conversation
                     and signals completion via a <user_info>{...}</user_info> block.
  run_qa()         — GPT-4o answers from HMO-filtered knowledge retrieved by
                     knowledge_base.retrieve().

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
from part2.backend.prompts import (
    COLLECTION_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT_TEMPLATE,
    QA_SINGLE_TOPIC_PROMPT_TEMPLATE,
    INSUFFICIENT_CONTEXT_SIGNAL,
    QUERY_TRANSLATION_PROMPT,
    NO_KNOWLEDGE_PLACEHOLDER,
    USER_INFO_OPEN,
    USER_INFO_CLOSE,
)

logger = get_logger(__name__)

# Matches the completion block the collection model emits. DOTALL so the JSON
# may span lines; non-greedy so we stop at the first closing tag.
USER_INFO_PATTERN = re.compile(
    re.escape(USER_INFO_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(USER_INFO_CLOSE),
    re.DOTALL,
)

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
    messages = (
        [{"role": "system", "content": COLLECTION_SYSTEM_PROMPT}]
        + _sanitize_history(history)
        + [{"role": "user", "content": user_message}]
    )

    # GPT-4o (not Mini) drives collection: inline validation requires reliable
    # digit counting (ID / card = 9 digits, age 0–120), and GPT-4o Mini routinely
    # miscounts digits — falsely rejecting valid 9-digit IDs. Correctness of the
    # required field validation outweighs Mini's lower cost for this low-volume flow.
    t0 = time.perf_counter()
    response = await async_openai_client.chat.completions.create(
        model=GPT4O_DEPLOYMENT,
        messages=messages,
        temperature=0.3,
    )
    elapsed = time.perf_counter() - t0

    raw_reply = response.choices[0].message.content or ""
    user_info, visible_reply = _extract_user_info(raw_reply)
    phase = "confirmation" if user_info else "collection"

    logger.info(
        "Collection turn complete",
        extra={
            "phase": phase,
            "complete": user_info is not None,
            "latency_s": round(elapsed, 2),
            "tokens": _usage(response),
        },
    )
    return {"reply": visible_reply, "user_info": user_info, "phase": phase}


def _extract_user_info(reply: str) -> tuple[dict | None, str]:
    """
    Look for the <user_info> completion block. On success return
    (parsed_and_validated_dict, reply_with_block_removed). Otherwise (None, reply).
    A malformed or incomplete block is treated as not-yet-complete.
    """
    match = USER_INFO_PATTERN.search(reply)
    if not match:
        return None, reply.strip()

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        logger.warning("Found <user_info> block but JSON failed to parse")
        return None, _strip_block(reply)

    missing = [f for f in _REQUIRED_FIELDS if f not in data or data[f] in ("", None)]
    if missing:
        logger.warning("Incomplete <user_info> block", extra={"missing": missing})
        return None, _strip_block(reply)

    normalized = _normalize_user_info(data)
    return normalized, _strip_block(reply)


def _strip_block(reply: str) -> str:
    """Remove the machine-readable block so the user never sees raw tags."""
    cleaned = USER_INFO_PATTERN.sub("", reply).strip()
    # Defensive: drop any dangling open tag if the model malformed the block.
    cleaned = cleaned.replace(USER_INFO_OPEN, "").replace(USER_INFO_CLOSE, "").strip()
    return cleaned


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
        response = await async_openai_client.chat.completions.create(
            model=GPT4O_DEPLOYMENT,
            temperature=0,
            messages=[
                {"role": "system", "content": QUERY_TRANSLATION_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
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
    return {
        "prompt": usage.prompt_tokens,
        "completion": usage.completion_tokens,
        "total": usage.total_tokens,
    }
