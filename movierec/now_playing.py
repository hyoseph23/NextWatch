"""Snapshot the movies currently in theaters, with full TMDB metadata.

Usage:
    python -m movierec.now_playing --region US

Writes data/now_playing/<date>_<region>.parquet. Run it daily to build up a
history of what was in theaters when.

TMDB's now-playing list includes re-screenings of old movies (anniversaries,
Halloween horror nights), and their release_date is the original one. Movies
first released more than --rerelease-days before the snapshot are kept but
flagged with is_rerelease=True so they can be excluded from recommendations.
"""

import argparse
import logging
from datetime import date

import pandas as pd

from movierec.config import NOW_PLAYING_DIR, TMDB_CACHE_PATH, tmdb_token
from movierec.tmdb import TMDBCache, TMDBClient, fetch_movies, flatten_many

log = logging.getLogger("movierec.now_playing")

# Long enough to keep late US releases of foreign and festival films.
RERELEASE_MAX_AGE_DAYS = 365


def flag_rereleases(
    df: pd.DataFrame, snapshot_date: date, max_age_days: int = RERELEASE_MAX_AGE_DAYS
) -> pd.DataFrame:
    """Mark movies whose original release is older than max_age_days.

    Movies with no release date are treated as new.
    """
    cutoff = pd.Timestamp(snapshot_date) - pd.Timedelta(days=max_age_days)
    return df.assign(is_rerelease=df["release_date"] < cutoff)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", default="US", help="ISO 3166-1 country code")
    parser.add_argument(
        "--rerelease-days",
        type=int,
        default=RERELEASE_MAX_AGE_DAYS,
        help="Flag movies first released more than this many days ago as re-releases",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    today = date.today()

    client = TMDBClient(tmdb_token())
    listing = client.now_playing(args.region)
    ids = list(dict.fromkeys(m["id"] for m in listing))
    log.info("%d movies now playing in %s", len(ids), args.region)

    cache = TMDBCache(TMDB_CACHE_PATH)
    try:
        # Always refresh: new releases gain keywords, credits and votes quickly.
        fetch_movies(client, cache, ids, refresh=True)
        df = flatten_many(cache.payloads(ids))
    finally:
        cache.close()

    df = flag_rereleases(df, today, args.rerelease_days)
    df["snapshot_date"] = today.isoformat()
    df["region"] = args.region

    NOW_PLAYING_DIR.mkdir(parents=True, exist_ok=True)
    out = NOW_PLAYING_DIR / f"{today.isoformat()}_{args.region}.parquet"
    df = df.sort_values("popularity", ascending=False)
    df.to_parquet(out, index=False)
    log.info("Wrote %s", out)

    new = df[~df["is_rerelease"]]
    log.info(
        "%d new releases, %d re-releases of older movies (flagged is_rerelease)",
        len(new),
        df["is_rerelease"].sum(),
    )
    print(new[["tmdb_title", "release_date", "genres", "directors"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
