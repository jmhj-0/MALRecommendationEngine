"""MAL API v2 client — fetches user lists and candidate titles."""

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.myanimelist.net/v2"

_USER_ANIME_FIELDS = (
    "list_status{status,score,num_episodes_watched,is_rewatching,"
    "start_date,finish_date,updated_at},"
    "num_episodes,mean,media_type,status,genres,studios,source,rating"
)
_USER_MANGA_FIELDS = (
    "list_status{status,score,num_volumes_read,num_chapters_read,"
    "is_rereading,start_date,finish_date,updated_at},"
    "num_volumes,num_chapters,mean,media_type,genres"
)
_CANDIDATE_ANIME_FIELDS = (
    "genres,studios,mean,media_type,num_scoring_users,"
    "source,rating,num_episodes,synopsis,start_season"
)
_CANDIDATE_MANGA_FIELDS = (
    "genres,mean,media_type,num_scoring_users,"
    "num_volumes,num_chapters,synopsis"
)


def _paginate(
    endpoint: str,
    client_id: str,
    params: dict[str, Any],
    max_items: int = 0,
) -> list[dict]:
    """Fetch all pages from a paginated MAL v2 endpoint."""
    headers = {"X-MAL-CLIENT-ID": client_id}
    url = f"{_BASE_URL}/{endpoint}"
    items: list[dict] = []

    while url:
        resp = requests.get(
            url, headers=headers, params=params, timeout=15
        )
        resp.raise_for_status()
        body = resp.json()
        items.extend(body.get("data", []))

        if max_items and len(items) >= max_items:
            items = items[:max_items]
            break

        url = body.get("paging", {}).get("next")
        params = {}
        time.sleep(0.3)

    return items


def fetch_user_anime_list(
    client_id: str, username: str
) -> list[dict]:
    """Fetch the complete anime list for a MAL user."""
    logger.info("Fetching anime list for %s", username)
    items = _paginate(
        f"users/{username}/animelist",
        client_id,
        {
            "fields": _USER_ANIME_FIELDS,
            "limit": 1000,
            "nsfw": "true",
        },
    )
    logger.info("Fetched %d anime entries", len(items))
    return items


def fetch_user_manga_list(
    client_id: str, username: str
) -> list[dict]:
    """Fetch the complete manga list for a MAL user."""
    logger.info("Fetching manga list for %s", username)
    items = _paginate(
        f"users/{username}/mangalist",
        client_id,
        {
            "fields": _USER_MANGA_FIELDS,
            "limit": 1000,
            "nsfw": "true",
        },
    )
    logger.info("Fetched %d manga entries", len(items))
    return items


def fetch_anime_candidates(
    client_id: str, per_ranking: int = 500
) -> list[dict]:
    """Pull anime from several ranking lists, deduplicated."""
    candidates: dict[int, dict] = {}

    for ranking_type in ("all", "bypopularity", "airing"):
        logger.info("Fetching anime ranking: %s", ranking_type)
        items = _paginate(
            "anime/ranking",
            client_id,
            {
                "ranking_type": ranking_type,
                "fields": _CANDIDATE_ANIME_FIELDS,
                "limit": per_ranking,
                "nsfw": "true",
            },
            max_items=per_ranking,
        )
        for item in items:
            node = item.get("node", {})
            mal_id = node.get("id")
            if mal_id and mal_id not in candidates:
                candidates[mal_id] = node

    logger.info(
        "Total unique anime candidates: %d", len(candidates)
    )
    return list(candidates.values())


def fetch_manga_candidates(
    client_id: str, per_ranking: int = 500
) -> list[dict]:
    """Pull manga from several ranking lists, deduplicated."""
    candidates: dict[int, dict] = {}

    for ranking_type in ("all", "bypopularity"):
        logger.info("Fetching manga ranking: %s", ranking_type)
        items = _paginate(
            "manga/ranking",
            client_id,
            {
                "ranking_type": ranking_type,
                "fields": _CANDIDATE_MANGA_FIELDS,
                "limit": per_ranking,
                "nsfw": "true",
            },
            max_items=per_ranking,
        )
        for item in items:
            node = item.get("node", {})
            mal_id = node.get("id")
            if mal_id and mal_id not in candidates:
                candidates[mal_id] = node

    logger.info(
        "Total unique manga candidates: %d", len(candidates)
    )
    return list(candidates.values())
