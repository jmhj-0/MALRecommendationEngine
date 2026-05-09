"""Score candidate titles against the user's preference profile."""

import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass

from .profile import BROAD_GENRES, PreferenceProfile

logger = logging.getLogger(__name__)

# ── scoring weights (sum to 1.0) ────────────────────────────
_W_GENRE = 0.25
_W_THEME = 0.10
_W_STUDIO = 0.10
_W_SOURCE = 0.05
_W_QUALITY = 0.30
_W_COLLAB = 0.10
_W_AIRING = 0.05
_W_NOVELTY = 0.05

_GENRE_CAP = 5  # max appearances per broad genre in top N

# ── recap / compilation detection ────────────────────────────
_RECAP_TITLE = re.compile(
    r"\brecap\b|\brecollections?\b|\bcompilation\b"
    r"|\bsoukatsu-hen\b|\bsoushuuhen\b|\bdigest\b",
    re.IGNORECASE,
)
_RECAP_SYNOPSIS = re.compile(
    r"^recap\s+of\b|^a\s+compilation\b|^compilation\s+of\b"
    r"|^summary\s+of\b|^a\s+retelling\s+of\b"
    r"|^summarizes\s|^recaps?\s+",
    re.IGNORECASE,
)

# ── franchise dedup ──────────────────────────────────────────
_STRIP_SUFFIX = re.compile(
    r"[\s:]*("
    r"season\s*\d+|2nd season|3rd season|\d+th season"
    r"|second season|third season"
    r"|part\s*\d+|cour\s*\d+"
    r"|movie|ova|special|recap|recollections"
    r"|act\s*(ii|iii|iv|[2-9])"
    r"|\d+(st|nd|rd|th)\s+season"
    r").*",
    re.IGNORECASE,
)


@dataclass
class Recommendation:
    """A scored recommendation with full metadata."""

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
    theme_score: float
    studio_score: float
    quality_score: float
    collab_score: float
    airing_score: float
    matching_genres: list[str]
    matching_studios: list[str]


def _is_recap(title: str, synopsis: str) -> bool:
    """Return True if the title looks like a recap or compilation."""
    return bool(
        _RECAP_TITLE.search(title)
        or _RECAP_SYNOPSIS.search(synopsis.strip())
    )


def _franchise_key(title: str) -> str:
    """Reduce a title to a rough franchise key for dedup."""
    key = title.split(": ")[0]
    key = _STRIP_SUFFIX.sub("", key).strip()
    key = re.sub(r"[^\w\s]", "", key).lower().strip()
    return key


def _has_unwatched_prequel(
    node: dict,
    media: str,
    profile: PreferenceProfile,
) -> bool:
    """Return True if the candidate has a prequel not completed."""
    rel_key = (
        "related_anime" if media == "anime"
        else "related_manga"
    )
    completed = (
        profile.completed_anime_ids
        if media == "anime"
        else profile.completed_manga_ids
    )
    for rel in node.get(rel_key, []):
        if rel.get("relation_type") == "prequel":
            pid = rel.get("node", {}).get("id")
            if pid and pid not in completed:
                return True
    return False


def _score_candidate(
    node: dict,
    profile: PreferenceProfile,
    media: str,
) -> Recommendation | None:
    """Score a single candidate; return None if filtered out."""
    mal_id = node.get("id")
    if not mal_id:
        return None

    # Already on user's list
    if media == "anime" and mal_id in profile.known_anime_ids:
        return None
    if media == "manga" and mal_id in profile.known_manga_ids:
        return None

    # Sequel filter — skip if prequel not completed
    if _has_unwatched_prequel(node, media, profile):
        return None

    # Recap / compilation filter
    title = node.get("title", "Unknown")
    synopsis = node.get("synopsis") or ""
    if _is_recap(title, synopsis):
        return None

    # Dynamic quality floor
    avg = (
        profile.avg_anime_score
        if media == "anime"
        else profile.avg_manga_score
    )
    floor = max(6.0, avg - 1.5)
    mean = node.get("mean") or 0
    if mean < floor:
        return None
    num_scoring = node.get("num_scoring_users") or 0
    if num_scoring < 1000:
        return None

    # Classify tags into genres vs themes
    all_tags = [g["name"] for g in node.get("genres", [])]
    genres = [t for t in all_tags if t in BROAD_GENRES]
    themes = [t for t in all_tags if t not in BROAD_GENRES]
    studios = [s["name"] for s in node.get("studios", [])]
    source = node.get("source", "")

    # Genre affinity
    gv = [
        profile.genres[g]
        for g in genres
        if g in profile.genres
    ]
    genre_score = sum(gv) / len(gv) if gv else 0.0

    # Theme affinity
    tv = [
        profile.themes[t]
        for t in themes
        if t in profile.themes
    ]
    theme_score = sum(tv) / len(tv) if tv else 0.0

    # Studio affinity
    sv = [
        profile.studios[s]
        for s in studios
        if s in profile.studios
    ]
    studio_score = sum(sv) / len(sv) if sv else 0.0

    # Source material affinity
    source_score = profile.sources.get(source, 0.0)

    # Quality score (relative to user's own average)
    quality_score = (mean - avg) / 3.0

    # Collaborative filtering bonus
    collab_ids = (
        profile.collab_anime_ids
        if media == "anime"
        else profile.collab_manga_ids
    )
    collab_score = 1.0 if mal_id in collab_ids else 0.0

    # Currently-airing boost
    airing_status = node.get("status", "")
    airing_score = (
        1.0 if airing_status == "currently_airing" else 0.0
    )

    # Random jitter for weekly variety
    novelty = random.uniform(-0.1, 0.1)

    final = (
        _W_GENRE * genre_score
        + _W_THEME * theme_score
        + _W_STUDIO * studio_score
        + _W_SOURCE * source_score
        + _W_QUALITY * quality_score
        + _W_COLLAB * collab_score
        + _W_AIRING * airing_score
        + _W_NOVELTY * novelty
    )

    matching_genres = [
        t
        for t in all_tags
        if profile.genres.get(t, 0) > 0
        or profile.themes.get(t, 0) > 0
    ]
    matching_studios = [
        s for s in studios if profile.studios.get(s, 0) > 0
    ]

    return Recommendation(
        mal_id=mal_id,
        title=title,
        media_type=node.get("media_type", "unknown"),
        genres=all_tags,
        studios=studios,
        mean_score=mean,
        num_scoring_users=num_scoring,
        source=source,
        synopsis=synopsis[:200],
        url=f"https://myanimelist.net/{media}/{mal_id}",
        final_score=final,
        genre_score=genre_score,
        theme_score=theme_score,
        studio_score=studio_score,
        quality_score=quality_score,
        collab_score=collab_score,
        airing_score=airing_score,
        matching_genres=matching_genres,
        matching_studios=matching_studios,
    )


def recommend(
    candidates: list[dict],
    profile: PreferenceProfile,
    media: str,
    top_n: int = 10,
    history_ids: set[int] | None = None,
) -> list[Recommendation]:
    """Score candidates and return top N with diversity."""
    scored: list[Recommendation] = []
    for node in candidates:
        rec = _score_candidate(node, profile, media)
        if rec:
            scored.append(rec)

    scored.sort(key=lambda r: r.final_score, reverse=True)

    seen_franchises: set[str] = set()
    genre_counts: dict[str, int] = defaultdict(int)
    history = history_ids or set()
    recs: list[Recommendation] = []

    for rec in scored:
        # Skip recently recommended titles
        if rec.mal_id in history:
            continue

        # Franchise dedup
        key = _franchise_key(rec.title)
        if key in seen_franchises:
            continue

        # Genre diversity cap (broad genres only)
        broad = [g for g in rec.genres if g in BROAD_GENRES]
        if broad and any(
            genre_counts[g] >= _GENRE_CAP for g in broad
        ):
            continue

        seen_franchises.add(key)
        for g in broad:
            genre_counts[g] += 1
        recs.append(rec)
        if len(recs) >= top_n:
            break

    logger.info(
        "Scored %d %s candidates, selected %d",
        len(scored),
        media,
        len(recs),
    )
    return recs
