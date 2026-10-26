import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from movierec.metrics import ndcg_at_k, rank_order, recall_at_k, reciprocal_rank

RANKED = np.array(
    [
        [1, 0, 1, 0],  # relevant at ranks 1 and 3
        [0, 0, 0, 1],  # relevant only at the bottom
        [0, 0, 0, 0],  # nothing relevant
    ],
    dtype=bool,
)


def test_ndcg_hand_computed():
    ndcg = ndcg_at_k(RANKED, 10)
    assert ndcg[0] == pytest.approx((1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3)))
    assert ndcg[1] == pytest.approx(1 / np.log2(5))
    assert np.isnan(ndcg[2])


def test_ndcg_cutoff_ignores_items_below_k():
    assert ndcg_at_k(RANKED, 2)[0] == pytest.approx(1 / (1 + 1 / np.log2(3)))
    assert ndcg_at_k(RANKED, 2)[1] == 0


def test_recall_and_reciprocal_rank():
    np.testing.assert_allclose(recall_at_k(RANKED, 2)[:2], [0.5, 0.0])
    np.testing.assert_allclose(reciprocal_rank(RANKED)[:2], [1.0, 0.25])
    assert np.isnan(recall_at_k(RANKED, 2)[2]) and np.isnan(reciprocal_rank(RANKED)[2])


def test_ndcg_matches_scikit_learn():
    rng = np.random.default_rng(0)
    scores = rng.random((200, 40))  # continuous, so no ties
    relevant = rng.random((200, 40)) < 0.1
    relevant[relevant.sum(axis=1) == 0, 0] = True

    order = rank_order(scores, np.arange(40))
    ours = ndcg_at_k(np.take_along_axis(relevant, order, axis=1), 10)
    theirs = [ndcg_score([rel], [s], k=10) for rel, s in zip(relevant.astype(int), scores)]
    np.testing.assert_allclose(ours, theirs)


def test_rank_order_breaks_ties_with_tiebreak():
    scores = np.array([[1.0, 2.0, 2.0, -np.inf]])
    assert rank_order(scores, np.array([0, 1, 0, 0])).tolist() == [[2, 1, 0, 3]]


def test_rank_order_rejects_nan():
    with pytest.raises(ValueError):
        rank_order(np.array([[np.nan, 1.0]]), np.arange(2))
