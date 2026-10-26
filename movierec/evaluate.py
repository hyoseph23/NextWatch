"""Run rankers through the time-split backtest and report metrics.

Usage:
    python -m movierec.evaluate --dataset 32m
    python -m movierec.evaluate --dataset 32m --plot docs/backtest_baselines.png

Writes per-query results and a summary to data/results/<dataset>/. Confidence
intervals come from a bootstrap over windows (cutoff dates), since queries in
the same window share a candidate pool and are not independent.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from movierec.backtest import BacktestConfig, RatingsIndex, Ranker, Window, iter_windows
from movierec.baselines import LeakyPopularityRanker, PopularityRanker, RandomRanker
from movierec.config import DATA_DIR, PROCESSED_DIR
from movierec.metrics import ndcg_at_k, rank_order, recall_at_k, reciprocal_rank

log = logging.getLogger("movierec.evaluate")

METRICS = {
    "ndcg@10": lambda rel: ndcg_at_k(rel, 10),
    "recall@5": lambda rel: recall_at_k(rel, 5),
    "mrr": reciprocal_rank,
}
BASELINE = "popularity"


def evaluate_window(window: Window, rankers: list[Ranker], seed: int = 0) -> pd.DataFrame:
    """Per-query metrics for every ranker on one window.

    All rankers share the same random tie-break so ties can't favor one method.
    Movies a user already rated before the cutoff are pushed to the bottom; they
    are never relevant, so they don't affect the metrics.
    """
    rng = np.random.default_rng([seed, window.cutoff.value // 10**9])
    tiebreak = rng.permutation(len(window.candidates))
    shape = (len(window.users), len(window.candidates))

    frames = []
    for ranker in rankers:
        scores = np.asarray(ranker.score(window), dtype=float)
        if scores.shape != shape:
            raise ValueError(f"{ranker.name} returned {scores.shape}, expected {shape}")
        order = rank_order(np.where(window.seen, -np.inf, scores), tiebreak)
        ranked_rel = np.take_along_axis(window.relevant, order, axis=1)
        frames.append(
            pd.DataFrame(
                {
                    "method": ranker.name,
                    "cutoff": window.cutoff,
                    "split": window.split,
                    "userId": window.users,
                    "n_candidates": shape[1] - window.seen.sum(axis=1),
                    "n_relevant": window.relevant.sum(axis=1),
                    **{name: fn(ranked_rel) for name, fn in METRICS.items()},
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def summarize(per_query: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Mean of each metric per method and split, with 95% window-bootstrap CIs."""
    rng = np.random.default_rng(seed)
    rows = []
    for (method, split), df in per_query.groupby(["method", "split"], sort=False):
        by_window = df.groupby("cutoff")
        counts = by_window.size().to_numpy()
        boot = rng.integers(0, len(counts), size=(n_boot, len(counts)))
        row = {
            "method": method,
            "split": split,
            "windows": len(counts),
            "queries": len(df),
            "avg_candidates": df["n_candidates"].mean(),
        }
        for metric in METRICS:
            sums = by_window[metric].sum().to_numpy()
            samples = sums[boot].sum(axis=1) / counts[boot].sum(axis=1)
            row[metric] = sums.sum() / counts.sum()
            row[f"{metric}_lo"], row[f"{metric}_hi"] = np.percentile(samples, [2.5, 97.5])
        rows.append(row)

    summary = pd.DataFrame(rows)
    base = summary[summary["method"] == BASELINE].set_index("split")["ndcg@10"]
    summary["ndcg@10_vs_popularity"] = summary["ndcg@10"] / summary["split"].map(base) - 1
    return summary


def to_markdown(summary: pd.DataFrame, splits=("val", "test")) -> str:
    lines = [
        "| Split | Method | NDCG@10 | Recall@5 | MRR | NDCG@10 vs. popularity |",
        "|---|---|---|---|---|---|",
    ]
    for split in splits:
        for _, r in summary[summary["split"] == split].iterrows():
            cells = [
                f"{r[m]:.3f} ({r[f'{m}_lo']:.3f}–{r[f'{m}_hi']:.3f})" for m in METRICS
            ]
            lift = "—" if r["method"] == BASELINE else f"{r['ndcg@10_vs_popularity']:+.1%}"
            lines.append(f"| {split} | {r['method']} | " + " | ".join(cells) + f" | {lift} |")
    return "\n".join(lines)


def plot_over_time(per_query: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yearly = (
        per_query.assign(year=per_query["cutoff"].dt.year)
        .groupby(["method", "year"])["ndcg@10"].mean()
        .unstack("method")
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for method in yearly.columns:
        style = "--" if method.endswith("_leaky") else "-"
        ax.plot(yearly.index, yearly[method], style, marker="o", markersize=3, label=method)
    ax.set(xlabel="Cutoff year", ylabel="NDCG@10", title="Baselines on the time-split backtest")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="32m")
    parser.add_argument("--splits", nargs="+", default=None, help="Default: all splits")
    parser.add_argument("--min-total-ratings", type=int, default=BacktestConfig.min_total_ratings)
    parser.add_argument("--min-history", type=int, default=BacktestConfig.min_history)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--plot", type=Path, help="Save an NDCG@10-over-time chart here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_dir = PROCESSED_DIR / args.dataset
    catalog = pd.read_parquet(data_dir / "catalog.parquet")
    log.info("Indexing ratings")
    start = time.perf_counter()
    index = RatingsIndex(pd.read_parquet(data_dir / "ratings.parquet"))
    log.info("Indexed %s ratings in %.0fs", f"{len(index.user):,}", time.perf_counter() - start)

    config = BacktestConfig(
        min_total_ratings=args.min_total_ratings, min_history=args.min_history
    )
    rankers = [RandomRanker(), PopularityRanker(), LeakyPopularityRanker(catalog)]
    n_cutoffs = len(config.cutoffs(args.splits))
    per_query = pd.concat(
        [
            evaluate_window(w, rankers)
            for w in tqdm(iter_windows(index, catalog, config, args.splits), total=n_cutoffs, desc="Windows")
        ],
        ignore_index=True,
    )

    out_dir = DATA_DIR / "results" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    per_query.to_parquet(out_dir / "per_query.parquet", index=False)
    summary = summarize(per_query, n_boot=args.bootstrap)
    summary.to_csv(out_dir / "summary.csv", index=False)
    table = to_markdown(summary)
    (out_dir / "summary.md").write_text(table + "\n")
    if args.plot:
        plot_over_time(per_query, args.plot)
        log.info("Saved plot to %s", args.plot)

    counts = per_query[per_query["method"] == BASELINE].groupby("split").agg(
        windows=("cutoff", "nunique"), queries=("userId", "size"), users=("userId", "nunique")
    )
    print("\n" + counts.to_string())
    print("\n" + table)


if __name__ == "__main__":
    main()
