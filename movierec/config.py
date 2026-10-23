"""Paths and settings shared across the project."""

import os
from pathlib import Path

import truststore
from dotenv import load_dotenv

# Verify HTTPS against the OS trust store. files.grouplens.org chains to a root
# that certifi (requests' default bundle) doesn't include.
truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("MOVIEREC_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
NOW_PLAYING_DIR = DATA_DIR / "now_playing"
TMDB_CACHE_PATH = DATA_DIR / "cache" / "tmdb.sqlite"


def tmdb_token() -> str:
    token = os.environ.get("TMDB_API_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "TMDB_API_TOKEN is not set. Copy .env.example to .env and add your "
            "TMDB API Read Access Token (https://www.themoviedb.org/settings/api)."
        )
    return token
