"""Build a weighted preference profile from MAL history."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _status_weight(status: str, score: int) -> float:
    """Return a preference weight based on list status and score.

    Completed/watching titles with a score use (score - 5.5) so that
    high-rated titles push genres positive and low-rated ones push
    them negative.  Dropped titles are a strong negative signal.
    """
    if status == "dropped":
        return -3.0
    if status in ("completed", "watching", "reading"):
        if score > 0:
            return score - 5.5
        return 1.5 if status in ("watching", "reading") else 1.0
    if status in ("plan_to_watch", "plan_to_read"):
        return 0.5
    return 0.0


@dataclass
class PreferenceProfile:
    """Normalised preference scores across several dimensions."""

    genres: dict[str, float] = field(default_factory=dict)
    studios: dict[str, float] = field(default_factory=dict)
    sources: dict[str, float] = field(default_factory=dict)

    known_anime_ids: set[int] = field(default_factory=set)
    known_manga_ids: set[int] = field(default_factory=set)

    avg_anime_score: float = 0.0
    avg_manga_score: float = 0.0


def _normalise(prefs: dict[str, float]) -> dict[str, float]:
    """Scale values into the [-1, 1] range."""
    if not prefs:
        return prefs
    peak = max(abs(v) for v in prefs.values())
    if peak == 0:
        return prefs
    return {k: v / peak for k, v in prefs.items()}


def build_profile(
    anime_list: list[dict],
    manga_list: list[dict],
) -> PreferenceProfile:
    """Analyse the user's lists and return a PreferenceProfile."""
    genre_acc: dict[str, float] = defaultdict(float)
    studio_acc: dict[str, float] = defaultdict(float)
    source_acc: dict[str, float] = defaultdict(float)
    known_anime: set[int] = set()
    known_manga: set[int] = set()
    anime_scores: list[int] = []
    manga_scores: list[int] = []

    for item in anime_list:
        node = item.get("node", {})
        ls = item.get("list_status", {})
        mal_id = node.get("id")
        if mal_id:
            known_anime.add(mal_id)

        status = ls.get("status", "")
        score = ls.get("score", 0)
        w = _status_weight(status, score)

        if score > 0:
            anime_scores.append(score)
        for g in node.get("genres", []):
            genre_acc[g["name"]] += w
        for s in node.get("studios", []):
            studio_acc[s["name"]] += w
        src = node.get("source", "")
        if src:
            source_acc[src] += w

    for item in manga_list:
        node = item.get("node", {})
        ls = item.get("list_status", {})
        mal_id = node.get("id")
        if mal_id:
            known_manga.add(mal_id)

        status = ls.get("status", "")
        score = ls.get("score", 0)
        w = _status_weight(status, score)

        if score > 0:
            manga_scores.append(score)
        for g in node.get("genres", []):
            genre_acc[g["name"]] += w

    profile = PreferenceProfile(
        genres=_normalise(dict(genre_acc)),
        studios=_normalise(dict(studio_acc)),
        sources=_normalise(dict(source_acc)),
        known_anime_ids=known_anime,
        known_manga_ids=known_manga,
        avg_anime_score=(
            sum(anime_scores) / len(anime_scores)
            if anime_scores
            else 0.0
        ),
        avg_manga_score=(
            sum(manga_scores) / len(manga_scores)
            if manga_scores
            else 0.0
        ),
    )

    logger.info(
        "Profile: %d genres, %d studios, %d sources, "
        "%d known anime, %d known manga",
        len(profile.genres),
        len(profile.studios),
        len(profile.sources),
        len(profile.known_anime_ids),
        len(profile.known_manga_ids),
    )
    return profile
