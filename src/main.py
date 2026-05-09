"""MAL Recommendation Engine -- weekly anime and manga picks."""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from .mal_client import (
    fetch_anime_candidates,
    fetch_collab_ids,
    fetch_manga_candidates,
    fetch_user_anime_list,
    fetch_user_manga_list,
)
from .profile import build_profile
from .recommender import (
    Recommendation,
    _W_AIRING,
    _W_COLLAB,
    _W_GENRE,
    _W_QUALITY,
    _W_SOURCE,
    _W_STUDIO,
    _W_THEME,
    recommend,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HISTORY_PATH = _PROJECT_ROOT / "data" / "history.json"
_HISTORY_WEEKS = 8


# ── history ──────────────────────────────────────────────────

def _load_history() -> tuple[set[int], dict]:
    """Load recent recommendation IDs and the raw history dict."""
    if not _HISTORY_PATH.exists():
        return set(), {"history": []}

    data = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
    recent_ids: set[int] = set()
    cutoff = datetime.now() - timedelta(weeks=_HISTORY_WEEKS)

    for entry in data.get("history", []):
        try:
            entry_date = datetime.strptime(
                entry["date"], "%Y-%m-%d"
            )
        except (KeyError, ValueError):
            continue
        if entry_date >= cutoff:
            recent_ids.update(entry.get("anime_ids", []))
            recent_ids.update(entry.get("manga_ids", []))

    return recent_ids, data


def _save_history(
    data: dict,
    anime_recs: list[Recommendation],
    manga_recs: list[Recommendation],
) -> None:
    """Append today's picks and write the history file."""
    data["history"].append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "anime_ids": [r.mal_id for r in anime_recs],
        "manga_ids": [r.mal_id for r in manga_recs],
    })
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY_PATH.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )


# ── output formatting ───────────────────────────────────────

def _print_profile_summary(profile) -> None:
    """Print top preferences."""
    print("\n  YOUR TASTE PROFILE")
    print(f"  {'-' * 50}")

    top_genres = sorted(
        profile.genres.items(), key=lambda x: x[1], reverse=True
    )
    loved = [g for g, v in top_genres if v > 0.3][:8]
    avoided = [g for g, v in top_genres if v < -0.1][:5]
    print(f"  Favourite genres  : {', '.join(loved)}")
    if avoided:
        print(f"  Avoided genres    : {', '.join(avoided)}")

    top_themes = sorted(
        profile.themes.items(), key=lambda x: x[1], reverse=True
    )
    liked_themes = [t for t, v in top_themes if v > 0.3][:8]
    if liked_themes:
        print(f"  Favourite themes  : {', '.join(liked_themes)}")

    top_studios = sorted(
        profile.studios.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    if top_studios:
        print(
            f"  Top studios       : "
            f"{', '.join(s for s, _ in top_studios)}"
        )

    print(
        f"  Avg anime score   : "
        f"{profile.avg_anime_score:.1f} / 10"
    )
    print(
        f"  Avg manga score   : "
        f"{profile.avg_manga_score:.1f} / 10"
    )
    print(
        f"  Collab sources    : "
        f"{len(profile.top_anime_ids)} anime, "
        f"{len(profile.top_manga_ids)} manga "
        f"(scored 9+)"
    )


def _score_breakdown(rec: Recommendation) -> str:
    """Compact string showing weighted score components."""
    parts = [
        f"genre {_W_GENRE * rec.genre_score:+.2f}",
        f"theme {_W_THEME * rec.theme_score:+.2f}",
        f"studio {_W_STUDIO * rec.studio_score:+.2f}",
        f"quality {_W_QUALITY * rec.quality_score:+.2f}",
    ]
    if rec.collab_score > 0:
        parts.append(f"collab +{_W_COLLAB:.2f}")
    if rec.airing_score > 0:
        parts.append(f"airing +{_W_AIRING:.2f}")
    return " | ".join(parts)


def _format_rec(rec: Recommendation, rank: int) -> str:
    """Format a single recommendation for console output."""
    lines = [
        f"  {rank:>2}. {rec.title}",
        f"      MAL: {rec.mean_score:.2f}  |  "
        f"Type: {rec.media_type}  |  "
        f"Scored by: {rec.num_scoring_users:,}",
        f"      Genres: {', '.join(rec.genres)}",
    ]
    if rec.studios:
        lines.append(
            f"      Studios: {', '.join(rec.studios)}"
        )

    reasons: list[str] = []
    if rec.matching_genres:
        reasons.append(
            f"genre match ({', '.join(rec.matching_genres)})"
        )
    if rec.matching_studios:
        reasons.append(
            f"studio match "
            f"({', '.join(rec.matching_studios)})"
        )
    if rec.collab_score > 0:
        reasons.append("similar to your 9/10-rated titles")
    if rec.airing_score > 0:
        reasons.append("currently airing")
    if rec.quality_score > 0.5:
        reasons.append("highly rated")
    if reasons:
        lines.append(f"      Why: {'; '.join(reasons)}")

    lines.append(
        f"      Score: {rec.final_score:.2f} "
        f"({_score_breakdown(rec)})"
    )

    if rec.synopsis:
        synopsis = rec.synopsis.rstrip()
        if len(synopsis) >= 200:
            synopsis = synopsis[:197] + "..."
        lines.append(f"      {synopsis}")
    lines.append(f"      {rec.url}")
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    load_dotenv(_PROJECT_ROOT / ".env")

    client_id = os.environ.get("MAL_CLIENT_ID")
    username = os.environ.get("MAL_USERNAME")
    if not client_id or not username:
        logging.error(
            "Set MAL_CLIENT_ID and MAL_USERNAME in .env"
        )
        sys.exit(1)

    # ── fetch user data ──────────────────────────────────────
    anime_list = fetch_user_anime_list(client_id, username)
    manga_list = fetch_user_manga_list(client_id, username)
    profile = build_profile(anime_list, manga_list)

    # ── collaborative filtering ──────────────────────────────
    if profile.top_anime_ids:
        profile.collab_anime_ids = fetch_collab_ids(
            client_id, "anime", profile.top_anime_ids
        )
    if profile.top_manga_ids:
        profile.collab_manga_ids = fetch_collab_ids(
            client_id, "manga", profile.top_manga_ids
        )

    # ── fetch candidates ─────────────────────────────────────
    anime_candidates = fetch_anime_candidates(client_id)
    manga_candidates = fetch_manga_candidates(client_id)

    # ── load history and generate recommendations ────────────
    history_ids, history_data = _load_history()

    anime_recs = recommend(
        anime_candidates, profile, "anime",
        history_ids=history_ids,
    )
    manga_recs = recommend(
        manga_candidates, profile, "manga",
        history_ids=history_ids,
    )

    _save_history(history_data, anime_recs, manga_recs)

    # ── output ───────────────────────────────────────────────
    date = datetime.now().strftime("%Y-%m-%d")
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  MAL WEEKLY RECOMMENDATIONS -- {date}")
    print(
        f"  User: {username}  |  "
        f"{len(anime_list)} anime, {len(manga_list)} manga"
    )
    print(sep)

    _print_profile_summary(profile)

    print(f"\n  ANIME PICKS ({len(anime_recs)})")
    print(f"  {'-' * 50}")
    for i, rec in enumerate(anime_recs, 1):
        print(_format_rec(rec, i))
        print()

    print(f"  MANGA PICKS ({len(manga_recs)})")
    print(f"  {'-' * 50}")
    for i, rec in enumerate(manga_recs, 1):
        print(_format_rec(rec, i))
        print()

    print(sep)


if __name__ == "__main__":
    main()
