# Now Showing: what to watch in theaters, based on movies you love

Pick a few movies you've loved, and this app ranks what's currently in theaters
by how well each one matches your taste, using hybrid search (BM25 + dense
embeddings) and a learning-to-rank re-ranker.

New releases have almost no ratings, so collaborative filtering can't
recommend them. This project treats recommendation as a **content-based
retrieval problem** and evaluates it with a time-split backtest on 32M
MovieLens ratings.

> Status: **Milestone 1 — data pipeline** ✅

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your TMDB API Read Access Token into .env
```

Get a free TMDB token at <https://www.themoviedb.org/settings/api>. Use the
long **API Read Access Token**, not the short API key.

## Data pipeline

```bash
# Quick iteration: ~9.7K movies, ~100K ratings (about 5 minutes of TMDB calls)
python -m movierec.build_catalog --dataset small

# Full dataset: ~87K movies, 32M ratings (~240 MB download, roughly an hour of TMDB calls)
python -m movierec.build_catalog --dataset 32m

# Snapshot what's in theaters today (run daily to build a history)
python -m movierec.now_playing --region US
```

TMDB responses are cached in `data/cache/tmdb.sqlite`, so re-runs only fetch
what's missing and an interrupted run picks up where it left off.

### Outputs

| File | Contents |
|---|---|
| `data/processed/<dataset>/catalog.parquet` | One row per movie: MovieLens id/title/year/genres, rating stats, and TMDB overview, keywords, genres, directors, writers, top-5 cast, runtime, popularity, release date |
| `data/processed/<dataset>/ratings.parquet` | `userId, movieId, rating, timestamp` with compact dtypes |
| `data/now_playing/<date>_<region>.parquet` | Movies in theaters on that date, same TMDB columns as the catalog |

## Project layout

```
movierec/
  config.py         paths, env vars
  movielens.py      download + load MovieLens, rating stats
  tmdb.py           TMDB client (retries, rate limits), SQLite cache, flattening
  build_catalog.py  CLI: MovieLens + TMDB -> catalog.parquet
  now_playing.py    CLI: today's in-theater movies -> parquet
tests/              pytest suite (no network or token needed)
```

## Roadmap

1. ✅ Data pipeline: MovieLens + TMDB catalog, now-playing snapshots
2. Time-split backtest + popularity baseline
3. BM25, dense retrieval, reciprocal rank fusion
4. LambdaMART re-ranker + ablation table (NDCG@10, Recall@5, MRR)
5. App with live now-playing list and "because you liked…" explanations
6. Write-up

## Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.
Ratings data from [MovieLens](https://grouplens.org/datasets/movielens/)
(F. Maxwell Harper and Joseph A. Konstan, 2015).
