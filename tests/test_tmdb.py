import pytest

from movierec import tmdb
from movierec.tmdb import (
    TMDBAuthError,
    TMDBCache,
    TMDBClient,
    fetch_movies,
    flatten_many,
    flatten_movie,
)

PAYLOAD = {
    "id": 101,
    "imdb_id": "tt0000101",
    "title": "Test Movie",
    "original_title": "Test Movie",
    "overview": "A linguist decodes a message.",
    "tagline": "",
    "release_date": "2016-11-11",
    "runtime": 116,
    "original_language": "en",
    "genres": [{"id": 878, "name": "Science Fiction"}, {"id": 18, "name": "Drama"}],
    "keywords": {"keywords": [{"id": 1, "name": "alien"}, {"id": 2, "name": "language"}]},
    "credits": {
        "cast": [
            {"name": "Second Billed", "order": 1},
            {"name": "Lead Actor", "order": 0},
        ],
        "crew": [
            {"name": "Some Director", "job": "Director", "department": "Directing"},
            {"name": "A Writer", "job": "Screenplay", "department": "Writing"},
            {"name": "A Writer", "job": "Novel", "department": "Writing"},
            {"name": "Composer", "job": "Original Music Composer", "department": "Sound"},
        ],
    },
    "popularity": 42.5,
    "vote_average": 7.6,
    "vote_count": 18000,
    "poster_path": "/abc.jpg",
    "adult": False,
}


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    """Returns queued responses in order, recording each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(tmdb.time, "sleep", lambda s: None)


def client_with(responses):
    session = FakeSession(responses)
    return TMDBClient("token", session_factory=lambda: session), session


def test_flatten_movie_extracts_fields():
    row = flatten_movie(PAYLOAD)
    assert row["genres"] == ["Science Fiction", "Drama"]
    assert row["keywords"] == ["alien", "language"]
    assert row["directors"] == ["Some Director"]
    assert row["writers"] == ["A Writer"]  # deduplicated
    assert row["cast"] == ["Lead Actor", "Second Billed"]  # sorted by billing order


def test_flatten_handles_sparse_payload():
    row = flatten_movie({"id": 5, "title": "Bare"})
    assert row["overview"] == ""
    assert row["genres"] == [] and row["keywords"] == [] and row["cast"] == []


def test_flatten_many_empty_has_columns():
    df = flatten_many([])
    assert "tmdb_id" in df.columns and len(df) == 0


def test_client_retries_on_rate_limit():
    client, session = client_with(
        [FakeResponse(429, headers={"Retry-After": "1"}), FakeResponse(200, PAYLOAD)]
    )
    status, body = client.movie(101)
    assert status == 200 and body["id"] == 101
    assert len(session.calls) == 2
    assert session.calls[0][1]["append_to_response"] == "keywords,credits"


def test_client_returns_none_on_404():
    client, _ = client_with([FakeResponse(404)])
    assert client.movie(999) == (404, None)


def test_client_raises_on_bad_token():
    client, _ = client_with([FakeResponse(401)])
    with pytest.raises(TMDBAuthError):
        client.check_auth()


def test_now_playing_reads_all_pages():
    client, session = client_with(
        [
            FakeResponse(200, {"results": [{"id": 1}], "total_pages": 2}),
            FakeResponse(200, {"results": [{"id": 2}], "total_pages": 2}),
        ]
    )
    assert [m["id"] for m in client.now_playing()] == [1, 2]
    assert [c[1]["page"] for c in session.calls] == [1, 2]


def test_fetch_movies_caches_and_skips_cached(tmp_path):
    cache = TMDBCache(tmp_path / "cache.sqlite")
    client, session = client_with([FakeResponse(200, PAYLOAD), FakeResponse(404)])

    fetch_movies(client, cache, [101, 999, 101], workers=1)
    assert cache.cached_ids() == {101, 999}
    assert [p["id"] for p in cache.payloads()] == [101]  # 404 cached without payload

    fetch_movies(client, cache, [101, 999], workers=1)
    assert len(session.calls) == 2  # nothing re-requested
    cache.close()
