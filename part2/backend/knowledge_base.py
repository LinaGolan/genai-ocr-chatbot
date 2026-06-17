from __future__ import annotations

"""
Knowledge base for the Q&A phase — folder-scoped Markdown retrieval.

The knowledge base is the pre-rendered Markdown tree under
``phase2_data/processed/<hmo>/<tier>/<topic>.md`` (3 HMOs × 3 tiers × 6 topics =
54 files, built offline by ``build_knowledge_base.py``). Each file is already
filtered to ONE HMO and ONE tier, so retrieval reduces to two cheap steps:

  1. Map the user's HMO + tier (from ``user_info``) to a folder.
  2. Pick the most relevant *topic* for the question.

Topic selection uses ADA-002. At startup we embed each topic's *shared* text —
the title, intro, and the service names + descriptions, but NOT the tier-specific
benefit lines or the contact section (those differ per file and would only add
noise). That shared text is identical across a topic's nine (HMO, tier) files, so
we embed it once per topic (6 embeddings). Per question we embed the query and
cosine-rank the 6 topics, returning the single best one.

Choosing which file(s) to actually feed the answering model — and the cheap
single-file vs. full-folder escalation — lives in ``chat_service.py``; this module
only embeds topics, ranks them, and reads files off disk.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from shared.azure_client import async_openai_client, ADA_DEPLOYMENT
from shared.logger import get_logger

logger = get_logger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "phase2_data" / "processed"

# user_info carries the Hebrew HMO / tier names; the folder tree uses English
# slugs. These maps translate one to the other (English values pass through).
HMO_SLUG_BY_NAME: dict[str, str] = {"מכבי": "maccabi", "מאוחדת": "meuhedet", "כללית": "clalit"}
TIER_SLUG_BY_NAME: dict[str, str] = {"זהב": "gold", "כסף": "silver", "ארד": "bronze"}

# Markers used to strip the per-file (tier-specific) parts out of the text we
# embed, so the embedded text is the shared topic/service description only.
_BENEFIT_MARKER = "**ההטבה שלך:**"
_CONTACT_HEADING = "## יצירת קשר"


@dataclass
class TopicEntry:
    slug: str                  # e.g. "pregnancy" (the .md filename stem)
    title: str                 # display name from the H1, e.g. "הריון"
    embedding: np.ndarray | None = field(default=None, repr=False)


# Module-level topic index, populated by build_index() at FastAPI startup.
topic_index: list[TopicEntry] = []


# ---------------------------------------------------------------------------
# HMO / tier → folder slug resolution
# ---------------------------------------------------------------------------

def resolve_hmo_slug(hmo: str) -> str | None:
    key = (hmo or "").strip()
    if key in HMO_SLUG_BY_NAME.values():
        return key
    return HMO_SLUG_BY_NAME.get(key)


def resolve_tier_slug(tier: str) -> str | None:
    key = (tier or "").strip()
    if key in TIER_SLUG_BY_NAME.values():
        return key
    return TIER_SLUG_BY_NAME.get(key)


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

async def build_index() -> list[TopicEntry]:
    """
    Discover the topics from the processed tree and embed each one's shared text.
    Populates and returns the module-level ``topic_index``. Safe to call once at
    startup.
    """
    representatives = _discover_topics()
    if not representatives:
        raise RuntimeError(f"No knowledge-base Markdown files found under {_DATA_DIR}")

    slugs = sorted(representatives)
    md_by_slug = {s: representatives[s].read_text(encoding="utf-8") for s in slugs}
    entries = [TopicEntry(slug=s, title=_extract_title(md_by_slug[s])) for s in slugs]

    # One embedding per topic — the embedded text is identical across a topic's
    # nine (HMO, tier) files. All 6 run concurrently; negligible startup cost.
    embeddings = await asyncio.gather(
        *(_embed_text(_embedding_text(md_by_slug[s])) for s in slugs)
    )
    for entry, emb in zip(entries, embeddings):
        entry.embedding = emb

    topic_index.clear()
    topic_index.extend(entries)

    logger.info(
        "Topic index built",
        extra={"topics": [(e.slug, e.title) for e in entries]},
    )
    return topic_index


def _discover_topics() -> dict[str, Path]:
    """
    Map each topic slug to one representative .md file. The embedded text is
    folder-independent, so any one of a topic's nine files is fine; we take the
    first in sorted order.
    """
    found: dict[str, Path] = {}
    for path in sorted(_DATA_DIR.glob("*/*/*.md")):
        found.setdefault(path.stem, path)
    return found


def _extract_title(md: str) -> str:
    """The H1 topic name, with the '— <hmo> — מסלול <tier>' suffix stripped off."""
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].split("—")[0].strip()
    return ""


def _embedding_text(md: str) -> str:
    """
    The text embedded for topic selection: title + intro + service names and
    their descriptions. The tier-specific benefit lines and the contact section
    are excluded — they differ per file and add noise without helping topic
    discrimination.
    """
    parts = [_extract_title(md)]
    for line in md.splitlines():
        stripped = line.strip()
        if stripped == _CONTACT_HEADING:
            break  # everything below is contact info — stop
        if stripped.startswith("# ") or stripped.startswith(_BENEFIT_MARKER):
            continue  # H1 (already captured via title) and benefit lines
        cleaned = stripped.lstrip("#").strip()
        if cleaned:
            parts.append(cleaned)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

async def select_topic(question: str) -> TopicEntry | None:
    """Embed *question* and return the single best-matching topic (or None if the
    index is empty)."""
    if not topic_index:
        logger.warning("select_topic() called before index was built")
        return None

    query_emb = await _embed_text(question)
    scored = [
        (_cosine(query_emb, e.embedding) if e.embedding is not None else 0.0, e)
        for e in topic_index
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best = scored[0]

    logger.info(
        "Topic selection",
        extra={"ranking": [(e.slug, round(s, 4)) for s, e in scored[:3]]},
    )
    return best


def load_topic_content(hmo: str, tier: str, topic_slug: str) -> str:
    """Read one topic's Markdown for the user's HMO/tier. '' if not resolvable."""
    path = _topic_path(hmo, tier, topic_slug)
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def load_all_topics(hmo: str, tier: str) -> dict[str, str]:
    """Read every topic's Markdown for the user's HMO/tier. {} if not resolvable."""
    folder = _folder(hmo, tier)
    if folder is None or not folder.is_dir():
        return {}
    return {
        p.stem: p.read_text(encoding="utf-8").strip()
        for p in sorted(folder.glob("*.md"))
    }


def _folder(hmo: str, tier: str) -> Path | None:
    hslug, tslug = resolve_hmo_slug(hmo), resolve_tier_slug(tier)
    if not hslug or not tslug:
        logger.warning("Unresolvable HMO/tier", extra={"hmo": hmo, "tier": tier})
        return None
    return _DATA_DIR / hslug / tslug


def _topic_path(hmo: str, tier: str, topic_slug: str) -> Path | None:
    folder = _folder(hmo, tier)
    return folder / f"{topic_slug}.md" if folder is not None else None


# ---------------------------------------------------------------------------
# Embedding / math helpers
# ---------------------------------------------------------------------------

async def _embed_text(text: str) -> np.ndarray:
    """Single ADA-002 embedding call → numpy vector."""
    response = await async_openai_client.embeddings.create(
        model=ADA_DEPLOYMENT,
        input=text,
    )
    return np.asarray(response.data[0].embedding, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
