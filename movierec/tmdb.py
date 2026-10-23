"""TMDB API client, an on-disk cache of raw responses, and flattening to rows."""

import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

API_BASE = "https://api.themoviedb.org/3"
TOP_CAST = 5
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class TMDBError(Exception):
    pass


class TMDBAuthError(TMDBError):
    pass


class TMDBClient:
    """Thin wrapper over the TMDB v3 API with retries and backoff.

    Thread-safe: each thread gets its own requests.Session.
    """

    def __init__(
        self,
        token: str,
        max_retries: int = 5,
        timeout: float = 10.0,
        session_factory=requests.Session,
    ):
        self._headers = {"Authorization": f"Bearer {token}", "accept": "application/json"}
        self._max_retries = max_retries
        self._timeout = timeout
        self._session_factory = session_factory
        self._local = threading.local()

    def _session(self):
        if not hasattr(self._local, "session"):
            self._local.session = self._session_factory()
        return self._local.session

    def _get(self, path: str, params: dict | None = None) -> tuple[int, dict | None]:
        """GET a path. Returns (status, json); 404 returns (404, None)."""
        url = f"{API_BASE}{path}"
        for attempt in range(self._max_retries):
            try:
                resp = self._session().get(
                    url, headers=self._headers, params=params, timeout=self._timeout
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                log.debug("Network error on %s: %s", path, exc)
                time.sleep(2**attempt)
                continue

            if resp.status_code in RETRYABLE_STATUS:
                retry_after = resp.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else 2**attempt)
                continue
            if resp.status_code == 404:
                return 404, None
            if resp.status_code == 401:
                raise TMDBAuthError("TMDB rejected the token (401). Check TMDB_API_TOKEN.")
            resp.raise_for_status()
            return resp.status_code, resp.json()

        raise TMDBError(f"Gave up on {path} after {self._max_retries} attempts")

    def check_auth(self) -> None:
        self._get("/authentication")

    def movie(self, tmdb_id: int) -> tuple[int, dict | None]:
        """Movie details with keywords and credits in a single request."""
        return self._get(
            f"/movie/{tmdb_id}",
            {"append_to_response": "keywords,credits", "language": "en-US"},
        )

    def now_playing(self, region: str = "US") -> list[dict]:
        """All pages of the now-playing list for a region."""
        results, page, total_pages = [], 1, 1
        while page <= total_pages:
            _, data = self._get(
                "/movie/now_playing", {"region": region, "page": page, "language": "en-US"}
            )
            results.extend(data["results"])
            total_pages = data["total_pages"]
            page += 1
        return results


class TMDBCache:
    """SQLite store of raw TMDB movie responses, keyed by TMDB id.

    404s are cached too (with a NULL payload) so we don't keep re-requesting
    movies TMDB has removed.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS movies (
                tmdb_id INTEGER PRIMARY KEY,
                status INTEGER NOT NULL,
                fetched_at TEXT NOT NULL,
                payload TEXT
            )"""
        )
        self._conn.commit()

    def cached_ids(self) -> set[int]:
        return {row[0] for row in self._conn.execute("SELECT tmdb_id FROM movies")}

    def put(self, tmdb_id: int, status: int, payload: dict | None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO movies VALUES (?, ?, ?, ?)",
            (
                tmdb_id,
                status,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(payload) if payload is not None else None,
            ),
        )

    def commit(self) -> None:
        self._conn.commit()

    def payloads(self, ids: Iterable[int] | None = None) -> list[dict]:
        """Successful payloads, optionally restricted to the given ids."""
        rows = self._conn.execute(
            "SELECT tmdb_id, payload FROM movies WHERE payload IS NOT NULL"
        )
        wanted = set(ids) if ids is not None else None
        return [
            json.loads(payload)
            for tmdb_id, payload in rows
            if wanted is None or tmdb_id in wanted
        ]

    def close(self) -> None:
        self._conn.close()


def fetch_movies(
    client: TMDBClient,
    cache: TMDBCache,
    ids: Iterable[int],
    workers: int = 8,
    refresh: bool = False,
) -> None:
    """Fetch details for any ids not already cached (or all of them if refresh)."""
    ids = list(dict.fromkeys(int(i) for i in ids))
    cached = set() if refresh else cache.cached_ids()
    todo = [i for i in ids if i not in cached]
    if not todo:
        log.info("All %d movies already cached", len(ids))
        return

    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(client.movie, i): i for i in todo}
        for n, fut in enumerate(
            tqdm(as_completed(futures), total=len(todo), desc="Fetching TMDB"), 1
        ):
            tmdb_id = futures[fut]
            try:
                status, payload = fut.result()
            except TMDBAuthError:
                pool.shutdown(cancel_futures=True)
                raise
            except TMDBError as exc:
                failures += 1
                log.warning("Failed to fetch %s: %s", tmdb_id, exc)
                continue
            cache.put(tmdb_id, status, payload)
            if n % 500 == 0:
                cache.commit()
    cache.commit()
    if failures:
        log.warning("%d movies failed; re-run to retry them", failures)


FLAT_COLUMNS = [
    "tmdb_id", "imdb_id", "tmdb_title", "original_title", "overview", "tagline",
    "release_date", "runtime", "original_language", "genres", "keywords",
    "directors", "writers", "cast", "popularity", "vote_average", "vote_count",
    "poster_path", "adult",
]


def flatten_movie(payload: dict) -> dict:
    """Turn a raw /movie/{id} response into one flat catalog row."""
    credits = payload.get("credits") or {}
    crew = credits.get("crew") or []
    cast = sorted(credits.get("cast") or [], key=lambda c: c.get("order", 999))
    keywords = (payload.get("keywords") or {}).get("keywords") or []

    return {
        "tmdb_id": payload["id"],
        "imdb_id": payload.get("imdb_id") or None,
        "tmdb_title": payload.get("title"),
        "original_title": payload.get("original_title"),
        "overview": payload.get("overview") or "",
        "tagline": payload.get("tagline") or "",
        "release_date": payload.get("release_date") or None,
        "runtime": payload.get("runtime") or None,
        "original_language": payload.get("original_language"),
        "genres": [g["name"] for g in payload.get("genres") or []],
        "keywords": [k["name"] for k in keywords],
        "directors": [c["name"] for c in crew if c.get("job") == "Director"],
        "writers": list(
            dict.fromkeys(c["name"] for c in crew if c.get("department") == "Writing")
        ),
        "cast": [c["name"] for c in cast[:TOP_CAST]],
        "popularity": payload.get("popularity"),
        "vote_average": payload.get("vote_average"),
        "vote_count": payload.get("vote_count"),
        "poster_path": payload.get("poster_path"),
        "adult": payload.get("adult", False),
    }


def flatten_many(payloads: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([flatten_movie(p) for p in payloads], columns=FLAT_COLUMNS)
    df["tmdb_id"] = df["tmdb_id"].astype("Int64")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df
