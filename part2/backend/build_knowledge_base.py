"""
Offline build step: convert the phase2_data topic HTML pages into focused
per-(HMO, tier) Markdown knowledge files using GPT-4o.

Each topic HTML page covers all three HMOs (מכבי / מאוחדת / כללית) and all three
tiers (זהב / כסף / ארד) in a single 4-column services table. For retrieval we want
one coherent, pre-filtered document per audience, so this script asks GPT-4o to
rewrite each page into a clean Markdown file per (HMO, tier) combination — 9 files
per topic — containing the shared topic intro plus ONLY that HMO+tier's service
benefits and that HMO's contact details.

This is a build-time tool, not part of the request path, so it uses the synchronous
Azure OpenAI client (single process, no concurrency needed).

Run (from the repository root):
  python -m part2.backend.build_knowledge_base             # all configured topics
  python -m part2.backend.build_knowledge_base pregnancy   # one topic only

Output: phase2_data/processed/<topic>_<hmo>_<tier>.md
"""

from __future__ import annotations

import sys
from pathlib import Path

from shared.azure_client import openai_client, GPT4O_DEPLOYMENT
from shared.logger import get_logger
from part2.backend.prompts import (
    KB_EXTRACTION_SYSTEM_PROMPT,
    KB_EXTRACTION_USER_TEMPLATE,
)

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIR = _REPO_ROOT / "phase2_data"
_OUTPUT_DIR = _SOURCE_DIR / "processed"

# English slug → Hebrew name exactly as it appears in the source HTML.
HMO_NAMES: dict[str, str] = {
    "maccabi": "מכבי",
    "meuhedet": "מאוחדת",
    "clalit": "כללית",
}
TIER_NAMES: dict[str, str] = {
    "gold": "זהב",
    "silver": "כסף",
    "bronze": "ארד",
}

# Topic slug → source HTML filename. The slug becomes the Markdown filename inside
# each <hmo>/<tier> folder (e.g. processed/maccabi/gold/pregnancy.md).
TOPICS: dict[str, str] = {
    "pregnancy": "pragrency_services.html",
    "dental": "dentel_services.html",
    "optometry": "optometry_services.html",
    "alternative": "alternative_services.html",
    "communication": "communication_clinic_services.html",
    "workshops": "workshops_services.html",
}


def _extract_markdown(html: str, hmo_he: str, tier_he: str) -> str:
    """One GPT-4o call → focused Markdown for a single (HMO, tier). temperature=0
    for deterministic, faithful extraction (we must not paraphrase numbers)."""
    response = openai_client.chat.completions.create(
        model=GPT4O_DEPLOYMENT,
        temperature=0,
        messages=[
            {"role": "system", "content": KB_EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": KB_EXTRACTION_USER_TEMPLATE.format(
                    hmo=hmo_he, tier=tier_he, html=html
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def build_topic(topic_slug: str) -> list[Path]:
    """Generate the 9 (HMO × tier) Markdown files for one topic. Returns the paths."""
    if topic_slug not in TOPICS:
        raise ValueError(
            f"Unknown topic '{topic_slug}'. Known topics: {', '.join(sorted(TOPICS))}"
        )

    source = _SOURCE_DIR / TOPICS[topic_slug]
    if not source.exists():
        raise FileNotFoundError(f"Source HTML not found: {source}")

    html = source.read_text(encoding="utf-8")

    logger.info("Building topic", extra={"topic": topic_slug, "source": source.name})

    written: list[Path] = []
    for hmo_slug, hmo_he in HMO_NAMES.items():
        for tier_slug, tier_he in TIER_NAMES.items():
            markdown = _extract_markdown(html, hmo_he, tier_he)
            # Layout: processed/<hmo>/<tier>/<topic>.md — mirrors how retrieval
            # reads them (pick the folder from user_info, then match the topic).
            out_path = _OUTPUT_DIR / hmo_slug / tier_slug / f"{topic_slug}.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(markdown + "\n", encoding="utf-8")
            written.append(out_path)
            logger.info(
                "Wrote KB file",
                extra={
                    "file": str(out_path.relative_to(_OUTPUT_DIR)),
                    "hmo": hmo_he,
                    "tier": tier_he,
                    "chars": len(markdown),
                },
            )

    logger.info("Topic complete", extra={"topic": topic_slug, "files": len(written)})
    return written


def main(argv: list[str]) -> None:
    topics = argv or list(TOPICS)
    for topic in topics:
        paths = build_topic(topic)
        print(f"\n{topic}: wrote {len(paths)} files to {_OUTPUT_DIR}")
        for p in paths:
            print(f"  - {p.relative_to(_OUTPUT_DIR)}")


if __name__ == "__main__":
    main(sys.argv[1:])
