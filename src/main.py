"""MAL Recommendation Engine — weekly anime and manga picks."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .mal_client import (
    fetch_anime_candidates,
    fetch_manga_candidates,
    fetch_user_anime_list,
    fetch_user_manga_list,
)
from .profile import PreferenceProfile, build_profile
from .recommender import Recommendation, recommend

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _print_profile_summary(
    profile: PreferenceProfile,
) -> None:
    """Print the top positive and negative preferences."""
    print("\n  YOUR TASTE PROFILE")
    print(f"  {'-' * 40}")

    top_genres = sorted(
        profile.genres.items(), key=lambda x: x[1], reverse=True
    )
    loved = [g for g, v in top_genres if v > 0.3][:8]
    avoided = [g for g, v in top_genres if v < -0.1][:5]
    print(f"  Favourite genres : {', '.join(loved)}")
    if avoided:
        print(f"  Avoided genres   : {', '.join(avoided)}")

    top_studios = sorted(
        profile.studios.items(), key=lambda x: x[1], reverse=True
    )[:5]
    if top_studios:
        studio_names = [s for s, _ in top_studios]
        print(f"  Top studios      : {', '.join(studio_names)}")

    print(
        f"  Avg anime score  : {profile.avg_anime_score:.1f} / 10"
    )
    print(
        f"  Avg manga score  : {profile.avg_manga_score:.1f} / 10"
    )


def _format_rec(rec: Recommendation, rank: int) -> str:
    """Format a single recommendation for console output."""
    lines = [
        f"  {rank:>2}. {rec.title}",
        f"      MAL score: {rec.mean_score:.2f}  |  "
        f"Type: {rec.media_type}  |  "
        f"Scored by: {rec.num_scoring_users:,}",
        f"      Genres: {', '.join(rec.genres)}",
    ]
    if rec.studios:
        lines.append(f"      Studios: {', '.join(rec.studios)}")
    reasons: list[str] = []
    if rec.matching_genres:
        reasons.append(
            f"genre match ({', '.join(rec.matching_genres)})"
        )
    if rec.matching_studios:
        reasons.append(
            f"studio match ({', '.join(rec.matching_studios)})"
        )
    if rec.quality_score > 0.5:
        reasons.append("highly rated")
    if reasons:
        lines.append(f"      Why: {'; '.join(reasons)}")
    if rec.synopsis:
        synopsis = rec.synopsis.rstrip()
        if len(synopsis) >= 200:
            synopsis = synopsis[:197] + "..."
        lines.append(f"      {synopsis}")
    lines.append(f"      {rec.url}")
    return "\n".join(lines)


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

    # ── fetch candidates ─────────────────────────────────────
    anime_candidates = fetch_anime_candidates(client_id)
    manga_candidates = fetch_manga_candidates(client_id)

    # ── generate recommendations ─────────────────────────────
    anime_recs = recommend(anime_candidates, profile, "anime")
    manga_recs = recommend(manga_candidates, profile, "manga")

    # ── output ───────────────────────────────────────────────
    date = datetime.now().strftime("%Y-%m-%d")
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  MAL WEEKLY RECOMMENDATIONS -- {date}")
    print(f"  User: {username}  |  "
          f"{len(anime_list)} anime, {len(manga_list)} manga")
    print(sep)

    _print_profile_summary(profile)

    print(f"\n  ANIME PICKS ({len(anime_recs)})")
    print(f"  {'-' * 40}")
    for i, rec in enumerate(anime_recs, 1):
        print(_format_rec(rec, i))
        print()

    print(f"  MANGA PICKS ({len(manga_recs)})")
    print(f"  {'-' * 40}")
    for i, rec in enumerate(manga_recs, 1):
        print(_format_rec(rec, i))
        print()

    print(sep)


if __name__ == "__main__":
    main()
