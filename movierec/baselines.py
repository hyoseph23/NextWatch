"""Non-personalized baselines that every later ranker has to beat."""

import numpy as np
import pandas as pd

from movierec.backtest import Window


class RandomRanker:
    """Sanity floor: a fresh random order for every query."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def score(self, window: Window) -> np.ndarray:
        rng = np.random.default_rng([self.seed, window.cutoff.value // 10**9])
        return rng.random((len(window.users), len(window.candidates)))


class PopularityRanker:
    """The leak-free baseline: rank by ratings received *before* the cutoff.

    This is what an app could actually know on that day. New releases have few
    ratings yet, which is the cold-start problem content-based ranking addresses.
    """

    name = "popularity"

    def score(self, window: Window) -> np.ndarray:
        counts = (
            window.candidate_ratings.groupby("movieId").size()
            .reindex(window.candidates, fill_value=0)
            .to_numpy(dtype=float)
        )
        return np.broadcast_to(counts, (len(window.users), len(window.candidates)))


class LeakyPopularityRanker:
    """NOT a valid baseline: ranks by each movie's eventual total rating count.

    Those totals include the very ratings being predicted. It is reported only to
    show how much a careless (leaky) evaluation would overstate performance.
    """

    name = "popularity_leaky"

    def __init__(self, catalog: pd.DataFrame):
        self._totals = catalog.set_index("movieId")["n_ratings"]

    def score(self, window: Window) -> np.ndarray:
        totals = self._totals.reindex(window.candidates, fill_value=0).to_numpy(dtype=float)
        return np.broadcast_to(totals, (len(window.users), len(window.candidates)))
