import numpy as np
import pandas as pd
import pytest

from movierec.backtest import BacktestConfig, RatingsIndex, build_window, content_catalog
from movierec.baselines import LeakyPopularityRanker, PopularityRanker
from movierec.evaluate import evaluate_window, summarize

CUTOFF = pd.Timestamp("2010-02-01")
CONFIG = BacktestConfig(min_total_ratings=10, min_history=2)

CATALOG = pd.DataFrame(
    {
        "movieId": [1, 2, 3, 4],
        "title": ["In theaters A", "In theaters B", "Too old", "Too obscure"],
        "release_date": pd.to_datetime(["2010-01-10", "2010-01-20", "2009-06-01", "2010-01-25"]),
        "n_ratings": [100, 100, 100, 2],
        "popularity": [1.0, 2.0, 3.0, 4.0],
    }
)


def r(user, movie, rating, when):
    return (user, movie, rating, pd.Timestamp(when))


RATINGS = pd.DataFrame(
    [
        # user 1: enough history, saw movie 1 early, later loved movie 2
        r(1, 3, 4.0, "2009-07-01"), r(1, 1, 5.0, "2010-01-15"), r(1, 2, 5.0, "2010-03-01"),
        # user 2: loved movie 1 later, but only one rating before the cutoff
        r(2, 3, 4.0, "2009-07-01"), r(2, 1, 5.0, "2010-03-01"),
        # user 3: disliked movie 1, loved movie 2 only after the one-year horizon
        r(3, 10, 4.0, "2009-01-01"), r(3, 11, 4.0, "2009-02-01"),
        r(3, 1, 3.0, "2010-03-01"), r(3, 2, 4.5, "2011-06-01"),
        # user 4: loved movie 1 exactly at the cutoff (counts as after)
        r(4, 10, 4.0, "2009-01-01"), r(4, 11, 4.0, "2009-02-01"), r(4, 1, 4.0, "2010-02-01"),
        # users 5 and 6: early ratings of movie 2 (popularity signal before the cutoff)
        r(5, 2, 3.0, "2010-01-25"), r(6, 2, 3.0, "2010-01-26"),
        # lots of future ratings of movie 1, which must not leak into popularity
        *[r(100 + i, 1, 5.0, "2010-06-01") for i in range(5)],
    ],
    columns=["userId", "movieId", "rating", "timestamp"],
)


@pytest.fixture(scope="module")
def window():
    return build_window(RatingsIndex(RATINGS), CATALOG, CUTOFF, "test", CONFIG)


def test_candidate_pool(window):
    # movie 3 was released too long ago, movie 4 never reached min_total_ratings
    assert window.candidates.tolist() == [1, 2]


def test_users_need_history_and_a_relevant_rating(window):
    # user 2: too little history; user 3: nothing relevant within the horizon;
    # users 100+: no history at all
    assert window.users.tolist() == [1, 4]


def test_relevance_and_seen(window):
    assert window.relevant.tolist() == [[False, True], [True, False]]
    assert window.seen.tolist() == [[True, False], [False, False]]


def test_rankers_only_see_the_past(window):
    assert (window.history["timestamp"] < CUTOFF).all()
    assert (window.candidate_ratings["timestamp"] < CUTOFF).all()
    assert set(window.history["userId"]) == {1, 4}


def test_popularity_uses_only_pre_cutoff_ratings(window):
    # movie 1: one rating before the cutoff (and many after); movie 2: two before
    assert PopularityRanker().score(window)[0].tolist() == [1.0, 2.0]
    assert LeakyPopularityRanker(CATALOG).score(window)[0].tolist() == [100.0, 100.0]


def test_evaluate_window(window):
    results = evaluate_window(window, [PopularityRanker()]).set_index("userId")
    # user 1 already saw movie 1, so movie 2 (relevant) is ranked first
    assert results.loc[1, "ndcg@10"] == pytest.approx(1.0)
    assert results.loc[1, "n_candidates"] == 1
    # user 4's relevant movie 1 is ranked second behind the more popular movie 2
    assert results.loc[4, "ndcg@10"] == pytest.approx(1 / np.log2(3))
    assert results.loc[4, "mrr"] == pytest.approx(0.5)


def test_summarize_reports_lift_over_popularity():
    per_query = pd.DataFrame(
        {
            "method": ["popularity"] * 4 + ["other"] * 4,
            "split": "test",
            "cutoff": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"] * 2),
            "userId": [1, 2, 1, 2] * 2,
            "n_candidates": 10,
            "ndcg@10": [0.2, 0.4, 0.2, 0.4, 0.3, 0.5, 0.3, 0.5],
            "recall@5": 0.5,
            "mrr": 0.5,
        }
    )
    summary = summarize(per_query, n_boot=200).set_index("method")
    assert summary.loc["popularity", "ndcg@10"] == pytest.approx(0.3)
    assert summary.loc["other", "ndcg@10_vs_popularity"] == pytest.approx(0.4 / 0.3 - 1)
    lo, hi = summary.loc["other", ["ndcg@10_lo", "ndcg@10_hi"]]
    assert lo <= 0.4 <= hi


def test_ratings_index_lookups_skip_missing_ids():
    index = RatingsIndex(RATINGS)
    assert sorted(index.movie[index.rows_for_users(np.array([1, 999]))].tolist()) == [1, 2, 3]
    assert set(index.user[index.rows_for_movies(np.array([2, 999]))]) == {1, 3, 5, 6}


def test_splits_must_leave_a_gap():
    with pytest.raises(ValueError):
        BacktestConfig(splits=(("train", "2000-01-01", "2014-12-01"), ("test", "2015-01-01", "2016-01-01")))


def test_content_catalog_drops_leaky_columns():
    assert {"n_ratings", "popularity"}.isdisjoint(content_catalog(CATALOG).columns)
