"""Build a weighted preference profile from MAL history."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

BROAD_GENRES = frozenset({
    "Action", "Adventure", "Comedy", "Drama", "Ecchi",
    "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi",
    "Slice of Life", "Sports", "Supernatural",
    "Avant Garde", "Girls Love", "Boys Love", "Suspense",
})
DEMOGRAPHICS = frozenset({
    "Shounen", "Shoujo", "Seinen", "Josei", "Kids",
})


def _status_weight(
    status: str,
    score: int,
    episodes_watched: int = 0,
    total_episodes: int = 0,
    chapters_read: int = 0,
    total_chapters: int = 0,
) -> float:
    """Return a preference weight from list status and score.

    Dropped titles use a proportional penalty: dropping after
    watching most of a show is a stronger negative signal than
    dropping after one episode.
    """
    if status == "dropped":
        watched = episodes_watched or chapters_read
        total = total_episodes or total_chapters
        if total > 0 and watched > 0:
            proportion = watched / total
            return -1.0 - (proportion * 4.0)
        return -2.0
    if status in ("completed", "watching", "reading"):
        if score > 0:
            return score - 5.5
        return 1.5 if status in ("watching", "reading") else 1.0
    if status in ("plan_to_watch", "plan_to_read"):
        return 0.5
    return 0.0


def _classify(name: str) -> str:
    """Classify a MAL genre/theme/demographic tag."""
    if name in DEMOGRAPHICS:
        return "genre"  # treat demographics like broad genres
    if name in BROAD_GENRES:
        return "genre"
    return "theme"


@dataclass
class PreferenceProfile:
    """Normalised preference scores across dimensions."""

    genres: dict[str, float] = field(default_factory=dict)
    themes: dict[str, float] = field(default_factory=dict)
    studios: dict[str, float] = field(default_factory=dict)
    sources: dict[str, float] = field(default_factory=dict)

    known_anime_ids: set[int] = field(default_factory=set)
    known_manga_ids: set[int] = field(default_factory=set)
    completed_anime_ids: set[int] = field(default_factory=set)
    completed_manga_ids: set[int] = field(default_factory=set)

    top_anime_ids: list[int] = field(default_factory=list)
    top_manga_ids: list[int] = field(default_factory=list)

    collab_anime_ids: set[int] = field(default_factory=set)
    collab_manga_ids: set[int] = field(default_factory=set)

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
    """Analyse both lists and return a PreferenceProfile."""
    genre_acc: dict[str, float] = defaultdict(float)
    theme_acc: dict[str, float] = defaultdict(float)
    studio_acc: dict[str, float] = defaultdict(float)
    source_acc: dict[str, float] = defaultdict(float)

    known_anime: set[int] = set()
    known_manga: set[int] = set()
    completed_anime: set[int] = set()
    completed_manga: set[int] = set()
    anime_scored: list[tuple[int, int]] = []
    manga_scored: list[tuple[int, int]] = []
    anime_score_vals: list[int] = []
    manga_score_vals: list[int] = []

    # ── anime ────────────────────────────────────────────────
    for item in anime_list:
        node = item.get("node", {})
        ls = item.get("list_status", {})
        mal_id = node.get("id")
        if mal_id:
            known_anime.add(mal_id)
        status = ls.get("status", "")
        if status == "completed" and mal_id:
            completed_anime.add(mal_id)

        score = ls.get("score", 0)
        w = _status_weight(
            status,
            score,
            episodes_watched=ls.get(
                "num_episodes_watched", 0
            ),
            total_episodes=node.get("num_episodes", 0),
        )

        if score > 0:
            anime_scored.append((mal_id, score))
            anime_score_vals.append(score)

        for g in node.get("genres", []):
            if _classify(g["name"]) == "genre":
                genre_acc[g["name"]] += w
            else:
                theme_acc[g["name"]] += w

        for s in node.get("studios", []):
            studio_acc[s["name"]] += w

        src = node.get("source", "")
        if src:
            source_acc[src] += w

    # ── manga ────────────────────────────────────────────────
    for item in manga_list:
        node = item.get("node", {})
        ls = item.get("list_status", {})
        mal_id = node.get("id")
        if mal_id:
            known_manga.add(mal_id)
        status = ls.get("status", "")
        if status == "completed" and mal_id:
            completed_manga.add(mal_id)

        score = ls.get("score", 0)
        w = _status_weight(
            status,
            score,
            chapters_read=ls.get("num_chapters_read", 0),
            total_chapters=node.get("num_chapters", 0),
        )

        if score > 0:
            manga_scored.append((mal_id, score))
            manga_score_vals.append(score)

        for g in node.get("genres", []):
            if _classify(g["name"]) == "genre":
                genre_acc[g["name"]] += w
            else:
                theme_acc[g["name"]] += w

    # ── top-rated IDs for collaborative filtering ────────────
    anime_scored.sort(key=lambda x: x[1], reverse=True)
    manga_scored.sort(key=lambda x: x[1], reverse=True)
    top_anime = [mid for mid, s in anime_scored if s >= 9][:20]
    top_manga = [mid for mid, s in manga_scored if s >= 9][:20]

    profile = PreferenceProfile(
        genres=_normalise(dict(genre_acc)),
        themes=_normalise(dict(theme_acc)),
        studios=_normalise(dict(studio_acc)),
        sources=_normalise(dict(source_acc)),
        known_anime_ids=known_anime,
        known_manga_ids=known_manga,
        completed_anime_ids=completed_anime,
        completed_manga_ids=completed_manga,
        top_anime_ids=top_anime,
        top_manga_ids=top_manga,
        avg_anime_score=(
            sum(anime_score_vals) / len(anime_score_vals)
            if anime_score_vals
            else 0.0
        ),
        avg_manga_score=(
            sum(manga_score_vals) / len(manga_score_vals)
            if manga_score_vals
            else 0.0
        ),
    )

    logger.info(
        "Profile: %d genres, %d themes, %d studios, "
        "%d completed anime, %d completed manga, "
        "%d top anime, %d top manga",
        len(profile.genres),
        len(profile.themes),
        len(profile.studios),
        len(profile.completed_anime_ids),
        len(profile.completed_manga_ids),
        len(profile.top_anime_ids),
        len(profile.top_manga_ids),
    )
    return profile
