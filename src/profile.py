"""Build a weighted preference profile from MAL history."""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations

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

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "as", "is", "was",
    "are", "were", "been", "be", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "not", "it", "its", "he", "she", "they",
    "them", "their", "his", "her", "him", "this", "that",
    "these", "those", "who", "whom", "which", "what", "where",
    "when", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor",
    "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there",
    "also", "however", "one", "two", "new", "first", "last",
    "long", "great", "over", "still", "even", "being",
    "while", "many", "much", "your", "you", "out", "up",
    "down", "off", "away", "back", "now", "get", "got",
    "make", "made", "take", "come", "see", "know", "say",
    "said", "tell", "told", "find", "give", "think", "want",
    "use", "way", "like", "well", "day", "time", "look",
    "upon", "yet", "must", "shall", "though", "since",
    "until", "another", "any", "our", "we", "my", "me",
})


# ── time decay ───────────────────────────────────────────────

def _time_decay(updated_at: str | None) -> float:
    """Return a multiplier between 0.5 and 1.0 based on recency."""
    if not updated_at:
        return 0.75
    try:
        ts = datetime.fromisoformat(
            updated_at.replace("Z", "+00:00")
        )
        days = (datetime.now(timezone.utc) - ts).days
        return max(0.5, 1.0 - (days / 1460) * 0.5)  # 4-year half-life
    except (ValueError, TypeError):
        return 0.75


# ── status weight ────────────────────────────────────────────

def _status_weight(
    status: str,
    score: int,
    episodes_watched: int = 0,
    total_episodes: int = 0,
    chapters_read: int = 0,
    total_chapters: int = 0,
) -> float:
    """Return a preference weight from list status and score."""
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
    if name in DEMOGRAPHICS or name in BROAD_GENRES:
        return "genre"
    return "theme"


def _author_name(author: dict) -> str:
    first = author.get("first_name", "")
    last = author.get("last_name", "")
    return f"{first} {last}".strip()


# ── synopsis keyword analysis ────────────────────────────────

def _tokenize(text: str) -> list[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [
        w for w in text.split()
        if len(w) > 2 and w not in _STOPWORDS
    ]


def build_synopsis_vocab(
    synopses: list[str], min_count: int = 3
) -> set[str]:
    """Build taste vocabulary from synopses of top-rated titles."""
    doc_freq: dict[str, int] = defaultdict(int)
    for synopsis in synopses:
        for word in set(_tokenize(synopsis)):
            doc_freq[word] += 1
    vocab = {w for w, c in doc_freq.items() if c >= min_count}
    logger.info(
        "Synopsis vocab: %d words from %d synopses",
        len(vocab),
        len(synopses),
    )
    return vocab


def synopsis_similarity(
    synopsis: str, vocab: set[str]
) -> float:
    """Score a candidate synopsis against the taste vocabulary."""
    if not vocab or not synopsis:
        return 0.0
    words = set(_tokenize(synopsis))
    if not words:
        return 0.0
    overlap = len(words & vocab)
    return min(1.0, overlap / 8.0)


# ── profile dataclass ────────────────────────────────────────

@dataclass
class PreferenceProfile:
    genres: dict[str, float] = field(default_factory=dict)
    themes: dict[str, float] = field(default_factory=dict)
    studios: dict[str, float] = field(default_factory=dict)
    authors: dict[str, float] = field(default_factory=dict)
    sources: dict[str, float] = field(default_factory=dict)

    known_anime_ids: set[int] = field(default_factory=set)
    known_manga_ids: set[int] = field(default_factory=set)
    completed_anime_ids: set[int] = field(default_factory=set)
    completed_manga_ids: set[int] = field(default_factory=set)

    top_anime_ids: list[int] = field(default_factory=list)
    top_manga_ids: list[int] = field(default_factory=list)

    dropped_combos: dict[tuple[str, str], int] = field(
        default_factory=dict
    )

    # Set after initial build via collab fetch
    collab_anime_weights: dict[int, float] = field(
        default_factory=dict
    )
    collab_manga_weights: dict[int, float] = field(
        default_factory=dict
    )
    cross_anime_ids: set[int] = field(default_factory=set)
    cross_manga_ids: set[int] = field(default_factory=set)
    synopsis_vocab: set[str] = field(default_factory=set)

    avg_anime_score: float = 0.0
    avg_manga_score: float = 0.0


def _normalise(prefs: dict[str, float]) -> dict[str, float]:
    if not prefs:
        return prefs
    peak = max(abs(v) for v in prefs.values())
    if peak == 0:
        return prefs
    return {k: v / peak for k, v in prefs.items()}


# ── profile builder ──────────────────────────────────────────

def build_profile(
    anime_list: list[dict],
    manga_list: list[dict],
) -> PreferenceProfile:
    genre_acc: dict[str, float] = defaultdict(float)
    theme_acc: dict[str, float] = defaultdict(float)
    studio_acc: dict[str, float] = defaultdict(float)
    author_acc: dict[str, float] = defaultdict(float)
    source_acc: dict[str, float] = defaultdict(float)
    combo_acc: dict[tuple[str, str], int] = defaultdict(int)

    known_anime: set[int] = set()
    known_manga: set[int] = set()
    completed_anime: set[int] = set()
    completed_manga: set[int] = set()
    anime_scored: list[tuple[int, int]] = []
    manga_scored: list[tuple[int, int]] = []
    anime_vals: list[int] = []
    manga_vals: list[int] = []

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
            episodes_watched=ls.get("num_episodes_watched", 0),
            total_episodes=node.get("num_episodes", 0),
        )
        decay = _time_decay(ls.get("updated_at"))
        w *= decay

        if score > 0:
            anime_scored.append((mal_id, score))
            anime_vals.append(score)

        tag_names = [g["name"] for g in node.get("genres", [])]
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

        if status == "dropped":
            for combo in combinations(sorted(tag_names), 2):
                combo_acc[combo] += 1

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
        decay = _time_decay(ls.get("updated_at"))
        w *= decay

        if score > 0:
            manga_scored.append((mal_id, score))
            manga_vals.append(score)

        tag_names = [g["name"] for g in node.get("genres", [])]
        for g in node.get("genres", []):
            if _classify(g["name"]) == "genre":
                genre_acc[g["name"]] += w
            else:
                theme_acc[g["name"]] += w

        for a in node.get("authors", []):
            name = _author_name(a)
            if name:
                author_acc[name] += w

        if status == "dropped":
            for combo in combinations(sorted(tag_names), 2):
                combo_acc[combo] += 1

    # ── top-rated IDs ────────────────────────────────────────
    anime_scored.sort(key=lambda x: x[1], reverse=True)
    manga_scored.sort(key=lambda x: x[1], reverse=True)
    top_anime = [mid for mid, s in anime_scored if s >= 9][:20]
    top_manga = [mid for mid, s in manga_scored if s >= 9][:20]

    # Only keep combos that appeared 2+ times
    sig_combos = {k: v for k, v in combo_acc.items() if v >= 2}

    profile = PreferenceProfile(
        genres=_normalise(dict(genre_acc)),
        themes=_normalise(dict(theme_acc)),
        studios=_normalise(dict(studio_acc)),
        authors=_normalise(dict(author_acc)),
        sources=_normalise(dict(source_acc)),
        known_anime_ids=known_anime,
        known_manga_ids=known_manga,
        completed_anime_ids=completed_anime,
        completed_manga_ids=completed_manga,
        top_anime_ids=top_anime,
        top_manga_ids=top_manga,
        dropped_combos=sig_combos,
        avg_anime_score=(
            sum(anime_vals) / len(anime_vals) if anime_vals else 0.0
        ),
        avg_manga_score=(
            sum(manga_vals) / len(manga_vals) if manga_vals else 0.0
        ),
    )

    logger.info(
        "Profile: %d genres, %d themes, %d studios, "
        "%d authors, %d dropped combos, "
        "%d top anime, %d top manga",
        len(profile.genres),
        len(profile.themes),
        len(profile.studios),
        len(profile.authors),
        len(profile.dropped_combos),
        len(profile.top_anime_ids),
        len(profile.top_manga_ids),
    )
    return profile
