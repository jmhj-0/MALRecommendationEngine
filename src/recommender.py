"""Score candidate titles against the user's preference profile."""

import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from .profile import (
    BROAD_GENRES,
    PreferenceProfile,
    synopsis_similarity,
)

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {
    "genre": 0.20,
    "theme": 0.08,
    "creator": 0.10,
    "source": 0.04,
    "quality": 0.30,
    "collab": 0.12,
    "synopsis": 0.06,
    "airing": 0.05,
    "novelty": 0.05,
}

DEFAULT_GENRE_CAP = 5
DEFAULT_MIN_SCORERS = 1000

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
    creator_score: float
    quality_score: float
    collab_score: float
    synopsis_score: float
    airing_score: float
    combo_penalty: float
    matching_genres: list[str]
    matching_studios: list[str]


def _is_recap(title: str, synopsis: str) -> bool:
    return bool(
        _RECAP_TITLE.search(title)
        or _RECAP_SYNOPSIS.search(synopsis.strip())
    )


def _franchise_key(title: str) -> str:
    key = title.split(": ")[0]
    key = _STRIP_SUFFIX.sub("", key).strip()
    key = re.sub(r"[^\w\s]", "", key).lower().strip()
    return key


def _has_unwatched_prequel(
    node: dict, media: str, profile: PreferenceProfile
) -> bool:
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


def _combo_penalty(
    tags: list[str],
    dropped_combos: dict[tuple[str, str], int],
) -> float:
    if not dropped_combos:
        return 0.0
    penalty = 0.0
    for combo in combinations(sorted(tags), 2):
        count = dropped_combos.get(combo, 0)
        if count >= 2:
            penalty += 0.05 * count
    return min(penalty, 0.3)


def _author_name(author: dict) -> str:
    first = author.get("first_name", "")
    last = author.get("last_name", "")
    return f"{first} {last}".strip()


def _score_candidate(
    node: dict,
    profile: PreferenceProfile,
    media: str,
    weights: dict[str, float],
    min_scorers: int,
) -> Recommendation | None:
    mal_id = node.get("id")
    if not mal_id:
        return None

    # Already on list
    known = (
        profile.known_anime_ids
        if media == "anime"
        else profile.known_manga_ids
    )
    if mal_id in known:
        return None

    # Sequel filter
    if _has_unwatched_prequel(node, media, profile):
        return None

    # Recap filter
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
    if num_scoring < min_scorers:
        return None

    # ── extract features ─────────────────────────────────────
    all_tags = [g["name"] for g in node.get("genres", [])]
    genres = [t for t in all_tags if t in BROAD_GENRES]
    themes = [t for t in all_tags if t not in BROAD_GENRES]
    studios = [s["name"] for s in node.get("studios", [])]
    authors = [
        _author_name(a)
        for a in node.get("authors", [])
        if _author_name(a)
    ]
    source = node.get("source", "")

    # ── score components ─────────────────────────────────────
    # Genre
    gv = [profile.genres[g] for g in genres if g in profile.genres]
    genre_score = sum(gv) / len(gv) if gv else 0.0

    # Theme
    tv = [profile.themes[t] for t in themes if t in profile.themes]
    theme_score = sum(tv) / len(tv) if tv else 0.0

    # Creator (studio + author combined)
    sv = [profile.studios[s] for s in studios if s in profile.studios]
    studio_score = sum(sv) / len(sv) if sv else 0.0
    av = [profile.authors[a] for a in authors if a in profile.authors]
    author_score = sum(av) / len(av) if av else 0.0
    if media == "manga" and av:
        creator_score = studio_score * 0.4 + author_score * 0.6
    else:
        creator_score = studio_score

    # Source
    source_score = profile.sources.get(source, 0.0)

    # Quality (relative to user avg)
    quality_score = (mean - avg) / 3.0

    # Collaborative (weighted + cross-media)
    collab_w = (
        profile.collab_anime_weights
        if media == "anime"
        else profile.collab_manga_weights
    )
    cross_ids = (
        profile.cross_anime_ids
        if media == "anime"
        else profile.cross_manga_ids
    )
    collab_score = collab_w.get(mal_id, 0.0)
    if mal_id in cross_ids:
        collab_score = max(collab_score, 0.5)

    # Synopsis similarity
    syn_score = synopsis_similarity(synopsis, profile.synopsis_vocab)

    # Currently airing
    airing_status = node.get("status", "")
    airing_score = (
        1.0 if airing_status == "currently_airing" else 0.0
    )

    # Negative combo penalty
    cp = _combo_penalty(all_tags, profile.dropped_combos)

    # Jitter
    novelty = random.uniform(-0.1, 0.1)

    final = (
        weights["genre"] * genre_score
        + weights["theme"] * theme_score
        + weights["creator"] * creator_score
        + weights["source"] * source_score
        + weights["quality"] * quality_score
        + weights["collab"] * collab_score
        + weights["synopsis"] * syn_score
        + weights["airing"] * airing_score
        + weights["novelty"] * novelty
        - cp
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
        creator_score=creator_score,
        quality_score=quality_score,
        collab_score=collab_score,
        synopsis_score=syn_score,
        airing_score=airing_score,
        combo_penalty=cp,
        matching_genres=matching_genres,
        matching_studios=matching_studios,
    )


def recommend(
    candidates: list[dict],
    profile: PreferenceProfile,
    media: str,
    top_n: int = 10,
    history_ids: set[int] | None = None,
    weights: dict[str, float] | None = None,
    genre_cap: int = DEFAULT_GENRE_CAP,
    min_scorers: int = DEFAULT_MIN_SCORERS,
) -> list[Recommendation]:
    """Score candidates and return top N with diversity."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    scored: list[Recommendation] = []
    for node in candidates:
        rec = _score_candidate(node, profile, media, w, min_scorers)
        if rec:
            scored.append(rec)

    scored.sort(key=lambda r: r.final_score, reverse=True)

    seen_franchises: set[str] = set()
    genre_counts: dict[str, int] = defaultdict(int)
    history = history_ids or set()
    recs: list[Recommendation] = []

    for rec in scored:
        if rec.mal_id in history:
            continue

        key = _franchise_key(rec.title)
        if key in seen_franchises:
            continue

        broad = [g for g in rec.genres if g in BROAD_GENRES]
        if broad and any(
            genre_counts[g] >= genre_cap for g in broad
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
