"""MAL API v2 client with caching, retry, and rich collab data."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.myanimelist.net/v2"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"

# ── module-level settings (set via configure()) ──────────────
_cache_ttl: int = 6 * 3600
_dry_run: bool = False
_no_cache: bool = False

_USER_ANIME_FIELDS = (
    "list_status{status,score,num_episodes_watched,is_rewatching,"
    "start_date,finish_date,updated_at},"
    "num_episodes,mean,media_type,status,genres,studios,source,rating"
)
_USER_MANGA_FIELDS = (
    "list_status{status,score,num_volumes_read,num_chapters_read,"
    "is_rereading,start_date,finish_date,updated_at},"
    "num_volumes,num_chapters,mean,media_type,genres,"
    "authors{first_name,last_name}"
)
_CANDIDATE_ANIME_FIELDS = (
    "genres,studios,mean,media_type,num_scoring_users,"
    "source,rating,num_episodes,synopsis,start_season,"
    "status,related_anime"
)
_CANDIDATE_MANGA_FIELDS = (
    "genres,mean,media_type,num_scoring_users,"
    "num_volumes,num_chapters,synopsis,"
    "status,related_manga,authors{first_name,last_name}"
)


@dataclass
class CollabData:
    """Rich collaborative-filtering payload."""

    weights: dict[int, float] = field(default_factory=dict)
    cross_media_ids: set[int] = field(default_factory=set)
    synopses: list[str] = field(default_factory=list)


def configure(
    *,
    cache_ttl_hours: int | None = None,
    dry_run: bool | None = None,
    no_cache: bool | None = None,
) -> None:
    """Set module-level API client options."""
    global _cache_ttl, _dry_run, _no_cache
    if cache_ttl_hours is not None:
        _cache_ttl = cache_ttl_hours * 3600
    if dry_run is not None:
        _dry_run = dry_run
    if no_cache is not None:
        _no_cache = no_cache


# ── caching ──────────────────────────────────────────────────

def _cache_key(url: str, params: dict) -> str:
    raw = f"{url}?{json.dumps(params, sort_keys=True)}" if params else url
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_get(key: str) -> dict | None:
    if _no_cache:
        return None
    path = _CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - data.get("ts", 0) < _cache_ttl:
            return data.get("body")
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _cache_set(key: str, body: dict) -> None:
    if _no_cache:
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _CACHE_DIR / f"{key}.json"
        path.write_text(
            json.dumps({"ts": time.time(), "body": body}),
            encoding="utf-8",
        )
    except OSError:
        pass


# ── HTTP with retry ──────────────────────────────────────────

def _api_get(
    url: str, headers: dict, params: dict | None = None
) -> dict:
    """GET with cache lookup, retry on 429/5xx, exponential backoff."""
    params = params or {}
    key = _cache_key(url, params)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    if _dry_run:
        logger.debug("Dry-run: no cache for %s", url[:80])
        return {}

    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                headers=headers,
                params=params or None,
                timeout=15,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                delay = 2 ** attempt
                logger.warning(
                    "HTTP %d, retry %d/3 in %ds",
                    resp.status_code,
                    attempt + 1,
                    delay,
                )
                time.sleep(delay)
                continue
            resp.raise_for_status()
            body = resp.json()
            _cache_set(key, body)
            return body
        except requests.RequestException as exc:
            if attempt == 2:
                logger.error("Request failed after 3 attempts: %s", exc)
                return {}
            delay = 2 ** attempt
            logger.warning(
                "Request error (%s), retry %d/3 in %ds",
                exc,
                attempt + 1,
                delay,
            )
            time.sleep(delay)

    return {}


# ── pagination ───────────────────────────────────────────────

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
        body = _api_get(url, headers, params)
        if not body:
            break
        items.extend(body.get("data", []))

        if max_items and len(items) >= max_items:
            items = items[:max_items]
            break

        url = body.get("paging", {}).get("next")
        params = {}
        time.sleep(0.3)

    return items


# ── user list fetchers ───────────────────────────────────────

def fetch_user_anime_list(
    client_id: str, username: str
) -> list[dict]:
    logger.info("Fetching anime list for %s", username)
    items = _paginate(
        f"users/{username}/animelist",
        client_id,
        {"fields": _USER_ANIME_FIELDS, "limit": 1000, "nsfw": "true"},
    )
    logger.info("Fetched %d anime entries", len(items))
    return items


def fetch_user_manga_list(
    client_id: str, username: str
) -> list[dict]:
    logger.info("Fetching manga list for %s", username)
    items = _paginate(
        f"users/{username}/mangalist",
        client_id,
        {"fields": _USER_MANGA_FIELDS, "limit": 1000, "nsfw": "true"},
    )
    logger.info("Fetched %d manga entries", len(items))
    return items


# ── candidate fetchers ───────────────────────────────────────

def _current_season() -> tuple[int, str]:
    month = datetime.now().month
    seasons = {
        1: "winter", 2: "winter", 3: "winter",
        4: "spring", 5: "spring", 6: "spring",
        7: "summer", 8: "summer", 9: "summer",
        10: "fall", 11: "fall", 12: "fall",
    }
    return datetime.now().year, seasons[month]


def fetch_anime_candidates(
    client_id: str, per_ranking: int = 500
) -> list[dict]:
    """Pull anime from rankings + current season, deduplicated."""
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

    # Current season anime
    year, season = _current_season()
    logger.info("Fetching seasonal anime: %s %d", season, year)
    items = _paginate(
        f"anime/season/{year}/{season}",
        client_id,
        {
            "fields": _CANDIDATE_ANIME_FIELDS,
            "limit": 100,
            "nsfw": "true",
        },
        max_items=100,
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
    """Pull manga from rankings, deduplicated."""
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


# ── collaborative filtering ──────────────────────────────────

def fetch_collab_data(
    client_id: str,
    media: str,
    mal_ids: list[int],
    per_title: int = 10,
) -> CollabData:
    """Fetch weighted collab IDs, cross-media IDs, and synopses."""
    headers = {"X-MAL-CLIENT-ID": client_id}
    counts: dict[int, int] = {}
    cross: set[int] = set()
    synopses: list[str] = []

    cross_key = (
        "related_manga" if media == "anime" else "related_anime"
    )
    fields = f"recommendations,synopsis,{cross_key}"

    for mid in mal_ids:
        body = _api_get(
            f"{_BASE_URL}/{media}/{mid}",
            headers,
            {"fields": fields},
        )
        if not body:
            continue

        # Collab recommendations (with counts)
        for rec in body.get("recommendations", [])[:per_title]:
            rid = rec.get("node", {}).get("id")
            if rid:
                counts[rid] = counts.get(rid, 0) + 1

        # Cross-media related entries
        for rel in body.get(cross_key, []):
            rtype = rel.get("relation_type", "")
            if rtype in ("adaptation", "alternative_version"):
                rid = rel.get("node", {}).get("id")
                if rid:
                    cross.add(rid)

        # Synopsis for keyword profile
        synopsis = body.get("synopsis", "")
        if synopsis:
            synopses.append(synopsis)

        time.sleep(0.3)

    # Normalise counts to 0-1
    peak = max(counts.values()) if counts else 1
    weights = {mid: c / peak for mid, c in counts.items()}

    logger.info(
        "Collab %s: %d weighted IDs, %d cross-media, "
        "%d synopses from %d titles",
        media,
        len(weights),
        len(cross),
        len(synopses),
        len(mal_ids),
    )
    return CollabData(
        weights=weights,
        cross_media_ids=cross,
        synopses=synopses,
    )
