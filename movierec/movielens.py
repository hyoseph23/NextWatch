"""Download and load the MovieLens datasets."""

import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from movierec.config import RAW_DIR

BASE_URL = "https://files.grouplens.org/datasets/movielens/"

# "small" (~100K ratings) is for quick iteration; "32m" is the full dataset.
DATASETS = {
    "small": "ml-latest-small",
    "32m": "ml-32m",
}

_YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")


def download(name: str, raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Download and unzip a MovieLens dataset, returning its folder."""
    folder = DATASETS[name]
    target = raw_dir / folder
    if target.exists() and not force:
        return target

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{folder}.zip"
    with requests.get(f"{BASE_URL}{folder}.zip", stream=True, timeout=30) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {folder}"
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    zip_path.unlink()
    return target


def split_title(raw_title: str) -> tuple[str, int | None]:
    """Split 'Toy Story (1995)' into ('Toy Story', 1995)."""
    match = _YEAR_RE.search(raw_title)
    if not match:
        return raw_title.strip(), None
    return raw_title[: match.start()].strip(), int(match.group(1))


def load_movies(folder: Path) -> pd.DataFrame:
    """Movies joined with their IMDb/TMDB ids, one row per MovieLens movieId."""
    movies = pd.read_csv(folder / "movies.csv")
    links = pd.read_csv(
        folder / "links.csv",
        dtype={"movieId": "int32", "imdbId": "string", "tmdbId": "Int64"},
    )

    parsed = movies["title"].map(split_title)
    movies["title"] = parsed.str[0]
    movies["year"] = pd.array(parsed.str[1], dtype="Int64")
    movies["ml_genres"] = movies["genres"].map(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    movies = movies.drop(columns="genres").astype({"movieId": "int32"})

    links = links.rename(columns={"imdbId": "imdb_id", "tmdbId": "tmdb_id"})
    links["imdb_id"] = "tt" + links["imdb_id"]
    return movies.merge(links, on="movieId", how="left")


def load_ratings(folder: Path) -> pd.DataFrame:
    """All ratings with compact dtypes and a proper datetime column."""
    ratings = pd.read_csv(
        folder / "ratings.csv",
        engine="pyarrow",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    return ratings


def rating_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """Per-movie popularity and timing stats from the ratings."""
    return (
        ratings.groupby("movieId")
        .agg(
            n_ratings=("rating", "size"),
            mean_rating=("rating", "mean"),
            first_rated_at=("timestamp", "min"),
        )
        .reset_index()
    )
