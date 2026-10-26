"""Ranking metrics, vectorized over many queries at once.

Each function takes ``ranked_rel``: a (queries, items) bool array giving the
relevance of each query's items in ranked order (best first). Queries with no
relevant items get NaN, since there is nothing to find.
"""

import numpy as np


def rank_order(scores: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Item indices per row, highest score first; ties go by ``tiebreak`` (lower first)."""
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")
    tiebreak = np.broadcast_to(tiebreak, scores.shape)
    return np.lexsort((tiebreak, -scores), axis=-1)


def _n_relevant(ranked_rel: np.ndarray) -> np.ndarray:
    return ranked_rel.sum(axis=1)


def ndcg_at_k(ranked_rel: np.ndarray, k: int) -> np.ndarray:
    """Normalized discounted cumulative gain with binary relevance."""
    k = min(k, ranked_rel.shape[1])
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = ranked_rel[:, :k] @ discounts
    n_rel = _n_relevant(ranked_rel)
    ideal = np.cumsum(discounts)[np.clip(n_rel, 1, k) - 1]
    return np.where(n_rel > 0, dcg / ideal, np.nan)


def recall_at_k(ranked_rel: np.ndarray, k: int) -> np.ndarray:
    """Share of a query's relevant items that appear in its top k."""
    n_rel = _n_relevant(ranked_rel)
    hits = ranked_rel[:, :k].sum(axis=1)
    return np.where(n_rel > 0, hits / np.maximum(n_rel, 1), np.nan)


def reciprocal_rank(ranked_rel: np.ndarray) -> np.ndarray:
    """1 / rank of the first relevant item (averaged over queries, this is MRR)."""
    n_rel = _n_relevant(ranked_rel)
    first = ranked_rel.argmax(axis=1)
    return np.where(n_rel > 0, 1.0 / (first + 1), np.nan)
