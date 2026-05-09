"""MAL Recommendation Engine -- weekly anime and manga picks."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from .mal_client import (
    CollabData,
    configure as configure_client,
    fetch_anime_candidates,
    fetch_collab_data,
    fetch_manga_candidates,
    fetch_user_anime_list,
    fetch_user_manga_list,
)
from .profile import (
    PreferenceProfile,
    build_profile,
    build_synopsis_vocab,
)
from .recommender import (
    DEFAULT_WEIGHTS,
    Recommendation,
    recommend,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "config.json"
_HISTORY_PATH = _PROJECT_ROOT / "data" / "history.json"


# ── config ───────────────────────────────────────────────────

def _load_config() -> dict:
    """Load config.json, falling back to defaults."""
    defaults = {
        "top_n": 10,
        "history_weeks": 8,
        "candidate_limit": 500,
        "collab_per_title": 10,
        "min_scorers": 1000,
        "genre_cap": 5,
        "cache_ttl_hours": 6,
        "weights": dict(DEFAULT_WEIGHTS),
    }
    if _CONFIG_PATH.exists():
        try:
            user = json.loads(
                _CONFIG_PATH.read_text(encoding="utf-8")
            )
            if "weights" in user:
                defaults["weights"] = {
                    **defaults["weights"],
                    **user["weights"],
                }
                del user["weights"]
            defaults.update(user)
        except (json.JSONDecodeError, OSError) as exc:
            logging.warning("Bad config.json, using defaults: %s", exc)
    return defaults


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MAL Recommendation Engine"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of recommendations per category",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use cached data only, no API calls",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip cache, always fetch fresh data",
    )
    return parser.parse_args()


# ── history ──────────────────────────────────────────────────

def _load_history(weeks: int) -> tuple[set[int], dict]:
    if not _HISTORY_PATH.exists():
        return set(), {"history": []}

    try:
        data = json.loads(
            _HISTORY_PATH.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        logging.warning("Corrupt history.json, starting fresh")
        return set(), {"history": []}

    recent: set[int] = set()
    cutoff = datetime.now() - timedelta(weeks=weeks)
    for entry in data.get("history", []):
        try:
            d = datetime.strptime(entry["date"], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue
        if d >= cutoff:
            recent.update(entry.get("anime_ids", []))
            recent.update(entry.get("manga_ids", []))
    return recent, data


def _save_history(
    data: dict,
    anime_recs: list[Recommendation],
    manga_recs: list[Recommendation],
) -> None:
    data["history"].append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "anime_ids": [r.mal_id for r in anime_recs],
        "manga_ids": [r.mal_id for r in manga_recs],
    })
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY_PATH.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )


# ── output ───────────────────────────────────────────────────

def _print_profile(profile: PreferenceProfile) -> None:
    print("\n  YOUR TASTE PROFILE")
    print(f"  {'-' * 50}")

    loved = [
        g
        for g, v in sorted(
            profile.genres.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        if v > 0.3
    ][:8]
    avoided = [
        g
        for g, v in sorted(
            profile.genres.items(), key=lambda x: x[1]
        )
        if v < -0.1
    ][:5]
    print(f"  Favourite genres  : {', '.join(loved)}")
    if avoided:
        print(f"  Avoided genres    : {', '.join(avoided)}")

    themes = [
        t
        for t, v in sorted(
            profile.themes.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        if v > 0.3
    ][:8]
    if themes:
        print(f"  Favourite themes  : {', '.join(themes)}")

    studios = [
        s
        for s, _ in sorted(
            profile.studios.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
    ]
    if studios:
        print(f"  Top studios       : {', '.join(studios)}")

    authors = [
        a
        for a, _ in sorted(
            profile.authors.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
    ]
    if authors:
        print(f"  Top authors       : {', '.join(authors)}")

    print(
        f"  Avg anime score   : "
        f"{profile.avg_anime_score:.1f} / 10"
    )
    print(
        f"  Avg manga score   : "
        f"{profile.avg_manga_score:.1f} / 10"
    )
    if profile.dropped_combos:
        top_combos = sorted(
            profile.dropped_combos.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        combos_str = ", ".join(
            f"{a}+{b} ({c}x)" for (a, b), c in top_combos
        )
        print(f"  Avoided combos    : {combos_str}")


def _breakdown(rec: Recommendation, w: dict) -> str:
    parts = [
        f"genre {w['genre'] * rec.genre_score:+.2f}",
        f"theme {w['theme'] * rec.theme_score:+.2f}",
        f"creator {w['creator'] * rec.creator_score:+.2f}",
        f"quality {w['quality'] * rec.quality_score:+.2f}",
    ]
    if rec.collab_score > 0:
        parts.append(
            f"collab {w['collab'] * rec.collab_score:+.2f}"
        )
    if rec.synopsis_score > 0:
        parts.append(
            f"synopsis {w['synopsis'] * rec.synopsis_score:+.2f}"
        )
    if rec.airing_score > 0:
        parts.append(f"airing +{w['airing']:.2f}")
    if rec.combo_penalty > 0:
        parts.append(f"combo -{rec.combo_penalty:.2f}")
    return " | ".join(parts)


def _format_rec(
    rec: Recommendation, rank: int, w: dict
) -> str:
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
            f"studio match ({', '.join(rec.matching_studios)})"
        )
    if rec.collab_score > 0:
        reasons.append("similar to your top-rated titles")
    if rec.synopsis_score > 0.3:
        reasons.append("thematically similar")
    if rec.airing_score > 0:
        reasons.append("currently airing")
    if rec.quality_score > 0.5:
        reasons.append("highly rated")
    if rec.combo_penalty > 0:
        reasons.append("has a dropped genre combo (penalised)")
    if reasons:
        lines.append(f"      Why: {'; '.join(reasons)}")

    lines.append(
        f"      Score: {rec.final_score:.2f} "
        f"({_breakdown(rec, w)})"
    )

    if rec.synopsis:
        s = rec.synopsis.rstrip()
        if len(s) >= 200:
            s = s[:197] + "..."
        lines.append(f"      {s}")
    lines.append(f"      {rec.url}")
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    args = _parse_args()
    cfg = _load_config()

    if args.top_n is not None:
        cfg["top_n"] = args.top_n

    load_dotenv(_PROJECT_ROOT / ".env")
    client_id = os.environ.get("MAL_CLIENT_ID")
    username = os.environ.get("MAL_USERNAME")
    if not client_id or not username:
        logging.error(
            "Set MAL_CLIENT_ID and MAL_USERNAME in .env"
        )
        sys.exit(1)

    configure_client(
        cache_ttl_hours=cfg["cache_ttl_hours"],
        dry_run=args.dry_run,
        no_cache=args.no_cache,
    )

    # ── fetch user data ──────────────────────────────────────
    anime_list = fetch_user_anime_list(client_id, username)
    manga_list = fetch_user_manga_list(client_id, username)
    profile = build_profile(anime_list, manga_list)

    # ── collaborative filtering (graceful degradation) ───────
    anime_collab = CollabData()
    manga_collab = CollabData()

    try:
        if profile.top_anime_ids:
            anime_collab = fetch_collab_data(
                client_id,
                "anime",
                profile.top_anime_ids,
                cfg["collab_per_title"],
            )
        if profile.top_manga_ids:
            manga_collab = fetch_collab_data(
                client_id,
                "manga",
                profile.top_manga_ids,
                cfg["collab_per_title"],
            )
    except Exception as exc:
        logging.warning(
            "Collaborative filtering failed, continuing "
            "without it: %s",
            exc,
        )

    # Enrich profile with collab data
    profile.collab_anime_weights = anime_collab.weights
    profile.collab_manga_weights = manga_collab.weights
    profile.cross_anime_ids = manga_collab.cross_media_ids
    profile.cross_manga_ids = anime_collab.cross_media_ids

    all_synopses = anime_collab.synopses + manga_collab.synopses
    if all_synopses:
        profile.synopsis_vocab = build_synopsis_vocab(
            all_synopses
        )

    # ── fetch candidates ─────────────────────────────────────
    anime_candidates = fetch_anime_candidates(
        client_id, cfg["candidate_limit"]
    )
    manga_candidates = fetch_manga_candidates(
        client_id, cfg["candidate_limit"]
    )

    # ── history + recommendations ────────────────────────────
    history_ids, history_data = _load_history(
        cfg["history_weeks"]
    )
    w = cfg["weights"]

    anime_recs = recommend(
        anime_candidates,
        profile,
        "anime",
        top_n=cfg["top_n"],
        history_ids=history_ids,
        weights=w,
        genre_cap=cfg["genre_cap"],
        min_scorers=cfg["min_scorers"],
    )
    manga_recs = recommend(
        manga_candidates,
        profile,
        "manga",
        top_n=cfg["top_n"],
        history_ids=history_ids,
        weights=w,
        genre_cap=cfg["genre_cap"],
        min_scorers=cfg["min_scorers"],
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

    _print_profile(profile)

    print(f"\n  ANIME PICKS ({len(anime_recs)})")
    print(f"  {'-' * 50}")
    for i, rec in enumerate(anime_recs, 1):
        print(_format_rec(rec, i, w))
        print()

    print(f"  MANGA PICKS ({len(manga_recs)})")
    print(f"  {'-' * 50}")
    for i, rec in enumerate(manga_recs, 1):
        print(_format_rec(rec, i, w))
        print()

    print(sep)


if __name__ == "__main__":
    main()
