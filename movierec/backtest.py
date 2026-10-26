"""Leakage-free, time-split backtest for "what should I watch in theaters?".

For each monthly cutoff date T we replay the app as if it were used on that day:

- Candidates ("in theaters"): catalog movies first released in the
  ``in_theaters_days`` before T. To mimic a wide-release lineup rather than every
  festival title, the pool keeps movies that eventually received at least
  ``min_total_ratings`` ratings. That uses future information to *define* the
  pool (identically for every method), never to rank within it.
- Users: anyone who rated a candidate at least ``relevant_rating`` within
  ``horizon_days`` after T, and who had at least ``min_history`` ratings before T.
- Relevance: (user, candidate) pairs rated >= ``relevant_rating`` in [T, T + horizon).
- What rankers may use: ratings strictly before T, plus content metadata from
  ``content_catalog`` (which drops columns computed from future data).

Cutoffs are grouped into train/val/test splits separated by gaps of at least one
horizon, so the labels of one split never overlap the next split's period.
"""

from dataclasses import dataclass, field
from typing import Iterator, Protocol

import numpy as np
import pandas as pd

DAY = 86_400

# Catalog columns computed from all ratings or from today's TMDB numbers. Using
# them to rank a past window would leak the future.
LEAKY_COLUMNS = [
    "n_ratings", "mean_rating", "first_rated_at", "popularity", "vote_average", "vote_count",
]


@dataclass(frozen=True)
class BacktestConfig:
    in_theaters_days: int = 42
    horizon_days: int = 365
    relevant_rating: float = 4.0
    min_history: int = 5
    min_total_ratings: int = 50
    # (name, first cutoff, last cutoff); cutoffs fall on the 1st of each month.
    splits: tuple = field(
        default=(
            ("train", "2000-01-01", "2014-12-01"),
            ("val", "2016-01-01", "2017-12-01"),
            ("test", "2019-01-01", "2022-09-01"),
        )
    )

    def __post_init__(self):
        horizon = pd.Timedelta(days=self.horizon_days)
        for (prev, _, prev_end), (name, start, _) in zip(self.splits, self.splits[1:]):
            if pd.Timestamp(prev_end) + horizon >= pd.Timestamp(start):
                raise ValueError(
                    f"{prev!r} labels run past the start of {name!r}; "
                    f"leave a gap of at least {self.horizon_days} days"
                )

    def cutoffs(self, splits: list[str] | None = None) -> list[tuple[pd.Timestamp, str]]:
        return [
            (cutoff, name)
            for name, start, end in self.splits
            if splits is None or name in splits
            for cutoff in pd.date_range(start, end, freq="MS")
        ]


@dataclass
class Window:
    """One simulated day: who is asking, what's in theaters, and the answers."""

    cutoff: pd.Timestamp
    split: str
    candidates: np.ndarray  # (C,) movieIds in theaters, sorted
    users: np.ndarray  # (U,) userIds being evaluated, sorted
    relevant: np.ndarray  # (U, C) bool: rated >= relevant_rating after the cutoff
    seen: np.ndarray  # (U, C) bool: already rated before the cutoff; not ranked
    history: pd.DataFrame  # these users' ratings before the cutoff
    candidate_ratings: pd.DataFrame  # everyone's ratings of candidates before the cutoff


class Ranker(Protocol):
    name: str

    def score(self, window: Window) -> np.ndarray:
        """Scores of shape (len(window.users), len(window.candidates)); higher is better."""
        ...


def content_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """The catalog without columns that would leak future information."""
    return catalog.drop(columns=[c for c in LEAKY_COLUMNS if c in catalog.columns])


def _to_seconds(values) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[s]").astype(np.int64)


def _runs(sorted_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, starts, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    return ids, starts, counts


def _gather(ids, starts, counts, wanted) -> np.ndarray:
    """Positions of every row belonging to the wanted ids (missing ids are skipped).

    Rows for one id are a contiguous run [start, start + count), so the result is
    those runs concatenated, built without a Python loop.
    """
    wanted = np.asarray(wanted)
    if len(ids) == 0 or len(wanted) == 0:
        return np.empty(0, dtype=np.int64)
    pos = np.minimum(np.searchsorted(ids, wanted), len(ids) - 1)
    pos = pos[ids[pos] == wanted]
    starts, counts = starts[pos], counts[pos]
    run_offsets = np.cumsum(counts) - counts
    return np.repeat(starts - run_offsets, counts) + np.arange(counts.sum())


class RatingsIndex:
    """All ratings as numpy arrays, with fast lookup by user or by movie."""

    def __init__(self, ratings: pd.DataFrame):
        user = ratings["userId"].to_numpy(np.int32)
        ts = _to_seconds(ratings["timestamp"])
        order = np.lexsort((ts, user))
        self.user = user[order]
        self.movie = ratings["movieId"].to_numpy(np.int32)[order]
        self.rating = ratings["rating"].to_numpy(np.float32)[order]
        self.ts = ts[order]
        self._users = _runs(self.user)
        self._by_movie = np.argsort(self.movie, kind="stable")
        self._movies = _runs(self.movie[self._by_movie])

    def rows_for_users(self, users: np.ndarray) -> np.ndarray:
        return _gather(*self._users, users)

    def rows_for_movies(self, movies: np.ndarray) -> np.ndarray:
        return self._by_movie[_gather(*self._movies, movies)]

    def frame(self, rows: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "userId": self.user[rows],
                "movieId": self.movie[rows],
                "rating": self.rating[rows],
                "timestamp": self.ts[rows].astype("datetime64[s]"),
            }
        )


def build_window(
    index: RatingsIndex,
    catalog: pd.DataFrame,
    cutoff: pd.Timestamp,
    split: str,
    config: BacktestConfig = BacktestConfig(),
) -> Window | None:
    """Build the evaluation window for one cutoff, or None if nobody qualifies."""
    t0 = int(_to_seconds([cutoff])[0])
    t_end = t0 + config.horizon_days * DAY

    released = catalog["release_date"]
    in_pool = (
        (released >= cutoff - pd.Timedelta(days=config.in_theaters_days))
        & (released < cutoff)
        & (catalog["n_ratings"] >= config.min_total_ratings)
    )
    candidates = np.sort(catalog.loc[in_pool, "movieId"].to_numpy(np.int32))
    if len(candidates) == 0:
        return None

    cand_rows = index.rows_for_movies(candidates)
    cand_ts = index.ts[cand_rows]
    before_rows = cand_rows[cand_ts < t0]
    target_rows = cand_rows[
        (cand_ts >= t0) & (cand_ts < t_end) & (index.rating[cand_rows] >= config.relevant_rating)
    ]

    target_users = np.unique(index.user[target_rows])
    hist_rows = index.rows_for_users(target_users)
    hist_rows = hist_rows[index.ts[hist_rows] < t0]
    hist_users, hist_counts = np.unique(index.user[hist_rows], return_counts=True)
    users = hist_users[hist_counts >= config.min_history]
    if len(users) == 0:
        return None
    hist_rows = hist_rows[np.isin(index.user[hist_rows], users)]

    def matrix(rows: np.ndarray) -> np.ndarray:
        rows = rows[np.isin(index.user[rows], users)]
        out = np.zeros((len(users), len(candidates)), dtype=bool)
        out[
            np.searchsorted(users, index.user[rows]),
            np.searchsorted(candidates, index.movie[rows]),
        ] = True
        return out

    seen = matrix(before_rows)
    relevant = matrix(target_rows) & ~seen

    return Window(
        cutoff=cutoff,
        split=split,
        candidates=candidates,
        users=users,
        relevant=relevant,
        seen=seen,
        history=index.frame(hist_rows),
        candidate_ratings=index.frame(before_rows),
    )


def iter_windows(
    index: RatingsIndex,
    catalog: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
    splits: list[str] | None = None,
) -> Iterator[Window]:
    for cutoff, split in config.cutoffs(splits):
        window = build_window(index, catalog, cutoff, split, config)
        if window is not None:
            yield window
