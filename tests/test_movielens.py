import pandas as pd
import pytest

from movierec.build_catalog import build_catalog
from movierec.movielens import load_movies, load_ratings, rating_stats, split_title
from movierec.tmdb import flatten_many


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Toy Story (1995)", ("Toy Story", 1995)),
        ("City of Lost Children, The (Cité des enfants perdus, La) (1995)",
         ("City of Lost Children, The (Cité des enfants perdus, La)", 1995)),
        ("Babylon 5", ("Babylon 5", None)),
        ("Death Note: Desu nôto (2006–2007)", ("Death Note: Desu nôto (2006–2007)", None)),
    ],
)
def test_split_title(raw, expected):
    assert split_title(raw) == expected


@pytest.fixture
def ml_folder(tmp_path):
    (tmp_path / "movies.csv").write_text(
        "movieId,title,genres\n"
        "1,Toy Story (1995),Adventure|Animation\n"
        "2,Mystery Film (2001),(no genres listed)\n"
    )
    (tmp_path / "links.csv").write_text(
        "movieId,imdbId,tmdbId\n"
        "1,0114709,862\n"
        "2,0000002,\n"
    )
    (tmp_path / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n"
        "1,1,4.0,964982703\n"
        "2,1,5.0,964982224\n"
    )
    return tmp_path


def test_load_movies(ml_folder):
    movies = load_movies(ml_folder)
    toy = movies.iloc[0]
    assert toy["title"] == "Toy Story" and toy["year"] == 1995
    assert toy["imdb_id"] == "tt0114709" and toy["tmdb_id"] == 862
    assert toy["ml_genres"] == ["Adventure", "Animation"]
    assert movies.iloc[1]["ml_genres"] == []
    assert pd.isna(movies.iloc[1]["tmdb_id"])


def test_build_catalog_joins_everything(ml_folder):
    movies = load_movies(ml_folder)
    stats = rating_stats(load_ratings(ml_folder))
    tmdb = flatten_many([{"id": 862, "title": "Toy Story", "overview": "Toys come alive."}])

    catalog = build_catalog(movies, stats, tmdb)
    assert len(catalog) == 2
    toy, mystery = catalog.iloc[0], catalog.iloc[1]
    assert toy["n_ratings"] == 2 and toy["mean_rating"] == 4.5
    assert toy["has_tmdb"] and toy["overview"] == "Toys come alive."
    assert mystery["n_ratings"] == 0 and not mystery["has_tmdb"]
