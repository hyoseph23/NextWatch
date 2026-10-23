"""Build the movie catalog: MovieLens movies + ratings stats + TMDB metadata.

Usage:
    python -m movierec.build_catalog --dataset small
    python -m movierec.build_catalog --dataset 32m --workers 8
"""

import argparse
import logging

import pandas as pd

from movierec import movielens
from movierec.config import PROCESSED_DIR, TMDB_CACHE_PATH, tmdb_token
from movierec.tmdb import TMDBCache, TMDBClient, fetch_movies, flatten_many

log = logging.getLogger("movierec.build_catalog")


def build_catalog(
    movies: pd.DataFrame, stats: pd.DataFrame, tmdb: pd.DataFrame
) -> pd.DataFrame:
    """Join MovieLens movies, rating stats and flattened TMDB rows."""
    tmdb = tmdb.drop(columns="imdb_id")  # MovieLens already has it for every movie
    catalog = (
        movies.merge(stats, on="movieId", how="left")
        .merge(tmdb, on="tmdb_id", how="left")
    )
    catalog["n_ratings"] = catalog["n_ratings"].fillna(0).astype("int32")
    catalog["has_tmdb"] = catalog["tmdb_title"].notna()
    return catalog


def report(catalog: pd.DataFrame) -> None:
    n = len(catalog)

    def pct(mask) -> str:
        return f"{mask.sum():>7,} ({mask.mean():6.1%})"

    print(f"\nCatalog: {n:,} movies")
    print(f"  with a TMDB id     {pct(catalog['tmdb_id'].notna())}")
    print(f"  with TMDB metadata {pct(catalog['has_tmdb'])}")
    print(f"  with an overview   {pct(catalog['overview'].fillna('').str.len() > 0)}")
    print(f"  with keywords      {pct(catalog['keywords'].map(lambda k: isinstance(k, list) and len(k) > 0))}")
    print(f"  with a director    {pct(catalog['directors'].map(lambda d: isinstance(d, list) and len(d) > 0))}")
    print(f"  with >= 1 rating   {pct(catalog['n_ratings'] > 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=movielens.DATASETS, default="small")
    parser.add_argument("--limit", type=int, help="Only use the first N movies (for testing)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent TMDB requests")
    parser.add_argument("--skip-fetch", action="store_true", help="Use only what's already cached")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    folder = movielens.download(args.dataset)
    movies = movielens.load_movies(folder)
    if args.limit:
        movies = movies.head(args.limit)

    log.info("Loading ratings from %s", folder.name)
    ratings = movielens.load_ratings(folder)
    ratings = ratings[ratings["movieId"].isin(movies["movieId"])]
    stats = movielens.rating_stats(ratings)

    tmdb_ids = movies["tmdb_id"].dropna().astype(int).tolist()
    cache = TMDBCache(TMDB_CACHE_PATH)
    try:
        if not args.skip_fetch:
            client = TMDBClient(tmdb_token())
            client.check_auth()
            fetch_movies(client, cache, tmdb_ids, workers=args.workers)
        tmdb = flatten_many(cache.payloads(tmdb_ids))
    finally:
        cache.close()

    catalog = build_catalog(movies, stats, tmdb)

    out_dir = PROCESSED_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out_dir / "catalog.parquet", index=False)
    ratings.to_parquet(out_dir / "ratings.parquet", index=False)
    log.info("Wrote %s and ratings.parquet (%s rows)", out_dir / "catalog.parquet", f"{len(ratings):,}")
    report(catalog)


if __name__ == "__main__":
    main()
