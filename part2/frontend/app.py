from __future__ import annotations

# Part 2 — Streamlit chat UI for the HMO medical-services chatbot.
#
# All session state lives client-side in st.session_state (the backend is
# stateless). Three phases:
#   collection   — LLM-driven gathering of the 8 user fields
#   confirmation — user reviews/corrects the collected info
#   qa           — questions answered from the user's HMO knowledge base
#
# Every backend turn sends the full conversation_history (+ user_info in Q&A),
# satisfying the statelessness requirement.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio
import re

import streamlit as st

from part2.frontend.api_client import ChatAPIClient, ChatAPIError

# Hebrew canonical HMO/tier values → English display names (for English chats).
# The stored user_info keeps the Hebrew values; these are display-only.
_HMO_DISPLAY_EN = {"מכבי": "Maccabi", "מאוחדת": "Meuhedet", "כללית": "Clalit"}
_TIER_DISPLAY_EN = {"זהב": "Gold", "כסף": "Silver", "ארד": "Bronze"}

_LATIN_RE = re.compile(r"[A-Za-z]")
_HEBREW_RE = re.compile(r"[֐-׿]")


def _user_wrote_english() -> bool:
    """
    Detect the conversation language from the user's FIRST lettered message — the
    same language the backend locked onto — so the transition card matches the
    rest of collection. Digits-only answers (an ID, a card number) carry no script
    and are skipped. Defaults to Hebrew.
    """
    for msg in st.session_state.get("messages", []):
        if msg.get("role") != "user":
            continue
        text = msg.get("content", "")
        latin = len(_LATIN_RE.findall(text))
        hebrew = len(_HEBREW_RE.findall(text))
        if latin or hebrew:
            return latin > hebrew
    return False

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="HMO Medical Services Chatbot", page_icon="🏥", layout="centered")

_client = ChatAPIClient()

_FIELD_LABELS = {
    "firstName": "First name (שם פרטי)",
    "lastName": "Last name (שם משפחה)",
    "idNumber": "ID number (מספר זהות)",
    "gender": "Gender (מין)",
    "age": "Age (גיל)",
    "hmo": "HMO (קופת חולים)",
    "hmoCardNumber": "HMO card number (מספר כרטיס)",
    "insuranceTier": "Insurance tier (מסלול)",
}

_GREETING = (
    "שלום! 👋 אני כאן כדי לעזור לך עם שאלות על שירותים רפואיים בקופות החולים "
    "(מכבי, מאוחדת, כללית). תחילה אאסוף ממך כמה פרטים. מה שמך הפרטי?\n\n"
    "_(Hello! I'll first collect a few details, then answer your medical-service "
    "questions. You can chat in Hebrew or English. What's your first name?)_"
)


def _run(coro):
    """Run an async coroutine from Streamlit's synchronous context."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# State init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    ss = st.session_state
    if "phase" not in ss:
        ss.phase = "collection"
        ss.conversation_history = []  # list[{role, content}] for the backend
        ss.user_info = None
        ss.messages = [{"role": "assistant", "content": _GREETING}]  # for display


def _reset() -> None:
    for key in ("phase", "conversation_history", "user_info", "messages"):
        st.session_state.pop(key, None)
    _init_state()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _append(role: str, content: str, *, to_backend: bool = True) -> None:
    """Add a message to the display log and (optionally) the backend history."""
    st.session_state.messages.append({"role": role, "content": content})
    if to_backend:
        st.session_state.conversation_history.append({"role": role, "content": content})


# ---------------------------------------------------------------------------
# Phase handlers
# ---------------------------------------------------------------------------

def _handle_collection(user_text: str) -> None:
    _append("user", user_text)
    with st.spinner("…"):
        try:
            result = _run(
                _client.send_collect_message(
                    history=st.session_state.conversation_history[:-1],
                    message=user_text,
                )
            )
        except ChatAPIError as exc:
            st.error(str(exc))
            st.session_state.conversation_history.pop()
            st.session_state.messages.pop()
            return

    _append("assistant", result["reply"])

    if result.get("user_info"):
        st.session_state.user_info = result["user_info"]
        st.session_state.phase = "confirmation"


def _render_confirmation() -> None:
    info = st.session_state.user_info
    st.subheader("📋 Please review your details / אנא בדקו את הפרטים")

    with st.form("confirm_form"):
        edited: dict = {}
        for key, label in _FIELD_LABELS.items():
            value = info.get(key, "")
            edited[key] = st.text_input(label, value=str(value))

        col_ok, col_edit = st.columns(2)
        confirmed = col_ok.form_submit_button("✅ Confirm / אישור", type="primary")
        corrected = col_edit.form_submit_button("✏️ Update & continue / עדכון")

    if confirmed or corrected:
        # Normalize age back to int when possible.
        age_raw = str(edited.get("age", "")).strip()
        edited["age"] = int(age_raw) if age_raw.isdigit() else age_raw
        st.session_state.user_info = edited
        st.session_state.phase = "qa"
        # Reset history for the Q&A phase (collection chat is no longer relevant).
        st.session_state.conversation_history = []
        name = edited.get("firstName", "")
        hmo = edited.get("hmo", "")
        tier = edited.get("insuranceTier", "")
        if _user_wrote_english():
            hmo_en = _HMO_DISPLAY_EN.get(hmo, hmo)
            tier_en = _TIER_DISPLAY_EN.get(tier, tier)
            transition = (
                f"Thanks {name}! ✅ Your details are saved. You can now ask me "
                f"anything about your medical services with {hmo_en} on the "
                f"{tier_en} tier."
            )
        else:
            transition = (
                f"תודה {name}! ✅ הפרטים נשמרו. אפשר לשאול אותי כל שאלה על "
                f"השירותים הרפואיים ב{hmo} במסלול {tier}."
            )
        st.session_state.messages.append({"role": "assistant", "content": transition})
        st.rerun()


def _handle_qa(user_text: str) -> None:
    _append("user", user_text)
    with st.spinner("…"):
        try:
            result = _run(
                _client.send_qa_message(
                    history=st.session_state.conversation_history[:-1],
                    user_info=st.session_state.user_info,
                    message=user_text,
                )
            )
        except ChatAPIError as exc:
            st.error(str(exc))
            st.session_state.conversation_history.pop()
            st.session_state.messages.pop()
            return
    _append("assistant", result["reply"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()

    st.title("🏥 HMO Medical Services Chatbot")
    with st.sidebar:
        st.markdown("### Status")
        st.write(f"**Phase:** `{st.session_state.phase}`")
        if st.session_state.user_info:
            st.write(f"**HMO:** {st.session_state.user_info.get('hmo','')}")
            st.write(f"**Tier:** {st.session_state.user_info.get('insuranceTier','')}")
        if st.button("🔄 Restart conversation"):
            _reset()
            st.rerun()
        st.caption("Maccabi · Meuhedet · Clalit | Hebrew & English")

    _render_history()

    if st.session_state.phase == "confirmation":
        _render_confirmation()
        return

    placeholder = (
        "Type your answer… / כתבו כאן…"
        if st.session_state.phase == "collection"
        else "Ask about your medical services… / שאלו על השירותים…"
    )
    user_text = st.chat_input(placeholder)
    if not user_text:
        return

    if st.session_state.phase == "collection":
        _handle_collection(user_text)
    elif st.session_state.phase == "qa":
        _handle_qa(user_text)
    st.rerun()


if __name__ == "__main__":
    main()
