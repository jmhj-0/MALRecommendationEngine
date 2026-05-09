# MAL Recommendation Engine

Weekly anime and manga recommendations based on your [MyAnimeList](https://myanimelist.net/) watch and read history.

## How It Works

The engine runs in four stages:

### 1. Profile Building

Pulls your complete anime and manga lists from the MAL API and builds weighted preference vectors across:

- **Genres** (Action, Drama, Romance, etc.) and **themes** (School, Isekai, Psychological, etc.) — tracked separately
- **Studios** and **manga authors** — creator affinity
- **Source material** preferences (manga adaptation, light novel, original, etc.)

Weights are based on your scores (centred at 5.5, so a score of 8 = +2.5, a score of 3 = -2.5), with:

- **Time decay** — recent ratings carry more weight than older ones (4-year half-life)
- **Proportional drop penalty** — dropping at episode 1/24 is a weaker signal than dropping at 20/24
- **Negative combo detection** — tracks which genre *pairs* appear on dropped titles (e.g. dropping multiple Comedy+Romance shows penalises that specific combination)

### 2. Collaborative Filtering

For your top 20 highest-rated titles (scored 9+), fetches MAL's "users who liked this also liked..." recommendations.

- **Weighted scoring** — a title recommended by 5 of your favourites scores higher than one recommended by 1
- **Cross-media linking** — if you rated a manga 9+, its anime adaptation gets a boost (and vice versa)
- **Synopsis keyword analysis** — builds a "taste vocabulary" from synopses of your top-rated titles (215 keywords from 40 synopses), then scores candidates by thematic overlap

### 3. Candidate Sourcing

Fetches ~1,200 anime and ~800 manga candidates from:

- MAL rankings (top rated, most popular, currently airing)
- Current season anime (catches new shows not yet in rankings)
- All responses are cached locally (6-hour TTL) for fast reruns

### 4. Scoring and Filtering

Each candidate is scored against your profile:

| Component | Weight | Description |
|---|---|---|
| Quality | 30% | MAL community score relative to your own average |
| Genre affinity | 20% | Match against your broad genre preferences |
| Collaborative | 12% | Weighted collab signal + cross-media boost |
| Creator affinity | 10% | Studio and manga author preferences |
| Theme affinity | 8% | Match against specific themes |
| Synopsis similarity | 6% | Keyword overlap with your top-rated titles |
| Currently airing | 5% | Small boost for airing titles |
| Source material | 4% | Match against preferred adaptation sources |
| Random jitter | 5% | Variety between runs |
| Combo penalty | -(var) | Penalises genre pairs you frequently drop |

**Filters applied before scoring:**

- Titles already on your list
- Sequels with unwatched prequels
- Recaps and compilation films
- Titles below a dynamic quality floor (based on your scoring average)
- Titles with fewer than 1,000 MAL ratings

**Diversity controls after scoring:**

- One entry per franchise (title-based deduplication)
- No broad genre appears in more than 5 of the 10 picks
- Titles recommended in the last 8 weeks are excluded

## Setup

```bash
git clone https://github.com/jmhj-0/MALRecommendationEngine.git
cd MALRecommendationEngine

cp .env.example .env
# Edit .env with your MAL Client ID and username

pip install -r requirements.txt
```

Get a MAL Client ID at https://myanimelist.net/apiconfig.

## Usage

```bash
# Normal run (fetches live data, caches responses)
python -m src.main

# Fast rerun using cached data (no API calls)
python -m src.main --dry-run

# Force fresh data (skip cache)
python -m src.main --no-cache

# Override number of recommendations
python -m src.main --top-n 15
```

## Configuration

All settings can be tuned via `config.json`:

```json
{
  "top_n": 10,
  "history_weeks": 8,
  "candidate_limit": 500,
  "collab_per_title": 10,
  "min_scorers": 1000,
  "genre_cap": 5,
  "cache_ttl_hours": 6,
  "weights": {
    "genre": 0.20,
    "theme": 0.08,
    "creator": 0.10,
    "source": 0.04,
    "quality": 0.30,
    "collab": 0.12,
    "synopsis": 0.06,
    "airing": 0.05,
    "novelty": 0.05
  }
}
```

## Weekly Schedule

A GitHub Actions workflow runs every Monday at 10:00 BST (09:00 UTC) and:

1. Generates 10 anime + 10 manga recommendations
2. Commits the updated recommendation history to `data/history.json`
3. Creates a GitHub issue with the results
4. On failure, creates a separate issue linking to the run logs

Manual trigger:

```bash
gh workflow run "Weekly Recommendations"
```

Repository secrets required: `MAL_CLIENT_ID`, `MAL_USERNAME`.

## Robustness

- **Retry with backoff** — API calls retry up to 3 times on 429/5xx errors with exponential backoff
- **Graceful degradation** — if collaborative filtering fails, recommendations still generate from rankings alone; corrupt history files are reset rather than crashing
- **Response caching** — all API responses cached to `data/cache/` with configurable TTL; enables fast `--dry-run` iterations
- **Workflow failure alerts** — failed GitHub Actions runs automatically create an issue with a link to the logs

## Project Structure

```
MALRecommendationEngine/
├── .github/workflows/
│   └── recommend.yml      # Weekly GitHub Actions workflow
├── data/
│   ├── cache/             # Cached API responses (gitignored)
│   └── history.json       # Past recommendation IDs (8-week window)
├── src/
│   ├── mal_client.py      # MAL API client — caching, retry, pagination
│   ├── profile.py         # Preference profile — genres, themes, studios,
│   │                      #   authors, combos, synopsis vocab, time decay
│   ├── recommender.py     # Scoring, filtering, and diversity ranking
│   └── main.py            # CLI, config, orchestration, output formatting
├── config.json            # Tunable weights and parameters
├── .env.example
├── requirements.txt
└── README.md
```
