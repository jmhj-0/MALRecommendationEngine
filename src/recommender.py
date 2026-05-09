"""Score candidate titles against the user's preference profile."""

import logging
import random
import re
from dataclasses import dataclass

from .profile import PreferenceProfile

logger = logging.getLogger(__name__)

_W_GENRE = 0.45
_W_STUDIO = 0.15
_W_SOURCE = 0.05
_W_QUALITY = 0.30
_W_NOVELTY = 0.05


@dataclass
class Recommendation:
    """A scored recommendation with metadata."""

    mal_id: int
    title: str
    media_type: str
    genres: list[str]
    studios: list[str]
    mean_score: float
    num_scoring_users: int
    source: str
    synopsis: str
    url: str
    final_score: float
    genre_score: float
    studio_score: float
    quality_score: float
    matching_genres: list[str]
    matching_studios: list[str]


def _score_candidate(
    node: dict,
    profile: PreferenceProfile,
    media: str,
) -> Recommendation | None:
    """Score a single candidate; return None if filtered out."""
    mal_id = node.get("id")
    if not mal_id:
        return None

    if media == "anime" and mal_id in profile.known_anime_ids:
        return None
    if media == "manga" and mal_id in profile.known_manga_ids:
        return None

    mean = node.get("mean") or 0
    if mean < 6.0:
        return None
    num_scoring = node.get("num_scoring_users") or 0
    if num_scoring < 1000:
        return None

    genres = [g["name"] for g in node.get("genres", [])]
    studios = [s["name"] for s in node.get("studios", [])]
    source = node.get("source", "")

    # Genre affinity
    gv = [profile.genres[g] for g in genres if g in profile.genres]
    genre_score = sum(gv) / len(gv) if gv else 0.0

    # Studio affinity
    sv = [
        profile.studios[s] for s in studios if s in profile.studios
    ]
    studio_score = sum(sv) / len(sv) if sv else 0.0

    # Source material affinity
    source_score = profile.sources.get(source, 0.0)

    # Community quality (normalised ~0-1)
    quality_score = (mean - 5.0) / 5.0

    # Small random jitter for variety between runs
    novelty = random.uniform(-0.1, 0.1)

    final = (
        _W_GENRE * genre_score
        + _W_STUDIO * studio_score
        + _W_SOURCE * source_score
        + _W_QUALITY * quality_score
        + _W_NOVELTY * novelty
    )

    matching_genres = [
        g for g in genres if profile.genres.get(g, 0) > 0
    ]
    matching_studios = [
        s for s in studios if profile.studios.get(s, 0) > 0
    ]

    return Recommendation(
        mal_id=mal_id,
        title=node.get("title", "Unknown"),
        media_type=node.get("media_type", "unknown"),
        genres=genres,
        studios=studios,
        mean_score=mean,
        num_scoring_users=num_scoring,
        source=source,
        synopsis=(node.get("synopsis") or "")[:200],
        url=f"https://myanimelist.net/{media}/{mal_id}",
        final_score=final,
        genre_score=genre_score,
        studio_score=studio_score,
        quality_score=quality_score,
        matching_genres=matching_genres,
        matching_studios=matching_studios,
    )


_STRIP_SUFFIX = re.compile(
    r"[\s:]*("
    r"season\s*\d+|2nd season|3rd season|\d+th season"
    r"|part\s*\d+|cour\s*\d+"
    r"|movie|ova|special|recap|recollections"
    r"|act\s*(ii|iii|iv|[2-9])"
    r"|\d+(st|nd|rd|th)\s+season"
    r").*",
    re.IGNORECASE,
)


def _franchise_key(title: str) -> str:
    """Reduce a title to a rough franchise key for dedup."""
    # Split at ": " to drop subtitles (preserves "Re:Zero" etc.)
    key = title.split(": ")[0]
    key = _STRIP_SUFFIX.sub("", key).strip()
    key = re.sub(r"[^\w\s]", "", key).lower().strip()
    return key


def recommend(
    candidates: list[dict],
    profile: PreferenceProfile,
    media: str,
    top_n: int = 10,
) -> list[Recommendation]:
    """Score all candidates and return the top N (one per franchise)."""
    scored = []
    for node in candidates:
        rec = _score_candidate(node, profile, media)
        if rec:
            scored.append(rec)

    scored.sort(key=lambda r: r.final_score, reverse=True)

    # Deduplicate by franchise — keep only the best entry
    seen_franchises: set[str] = set()
    recs: list[Recommendation] = []
    for rec in scored:
        key = _franchise_key(rec.title)
        if key in seen_franchises:
            continue
        seen_franchises.add(key)
        recs.append(rec)
        if len(recs) >= top_n:
            break

    logger.info(
        "Scored %d %s candidates, returning top %d",
        len(scored),
        media,
        top_n,
    )
    return recs
