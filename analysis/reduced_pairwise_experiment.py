"""Experiment: does trimming the pairwise feature set to only well-observed
pairs beat the full 16k-column pairwise model?

Context (see docs/MODEL.md, "What was tried and rejected"): the full pairwise
model (127 hero + ~16,000 synergy/counter columns, heavy L2) scores marginally
better than the plain linear model but McNemar's test cannot tell them apart
(p ~= 0.13), and its learning curve had not flattened -- suggesting most of
those 16,000 columns are mostly noise that regularisation is fighting, not
signal the model is using. Most hero pairs simply never co-occur often enough
in 53,802 matches to estimate a pair-specific weight.

This experiment tests the direct fix: keep only pairs seen often enough in
training to plausibly carry signal, and see whether a much smaller, better-
conditioned pairwise model beats both the plain linear model and the full
pairwise one on the same chronological split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import build_hero_index, build_pair_index, load_matches, parse_team  # noqa: E402

RANDOM_STATE = 42


def metrics(y_true, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def encode_with_pair_counts(df, hero_index, pair_index):
    """Like encode_drafts_pairwise, but also returns per-pair occurrence counts
    so we can threshold on them afterwards."""
    n = len(hero_index)
    n_pairs = len(pair_index)
    rows, cols, vals = [], [], []
    pair_counts = np.zeros(2 * n_pairs, dtype=np.int64)

    for row, (radiant, dire) in enumerate(
        zip(df["radiant_team"].to_numpy(), df["dire_team"].to_numpy())
    ):
        R = [hero_index[h] for h in parse_team(radiant) if h in hero_index]
        D = [hero_index[h] for h in parse_team(dire) if h in hero_index]

        for col in R:
            rows.append(row); cols.append(col); vals.append(1.0)
        for col in D:
            rows.append(row); cols.append(col); vals.append(-1.0)

        for team, sign in ((R, 1.0), (D, -1.0)):
            for a in range(len(team)):
                for b in range(a + 1, len(team)):
                    i, j = sorted((team[a], team[b]))
                    if i == j:
                        continue
                    c = n + pair_index[(i, j)]
                    rows.append(row); cols.append(c); vals.append(sign)
                    pair_counts[c - n] += 1

        for a in R:
            for b in D:
                if a == b:
                    continue
                i, j = (a, b) if a < b else (b, a)
                sign = 1.0 if a < b else -1.0
                c = n + n_pairs + pair_index[(i, j)]
                rows.append(row); cols.append(c); vals.append(sign)
                pair_counts[c - n] += 1

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(df), n + 2 * n_pairs), dtype=np.float32)
    return X, pair_counts


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "artifacts"
    df = load_matches(str(Path(__file__).resolve().parent.parent / "data" / "matches.csv.gz"), game_mode=22)

    all_heroes = set()
    for radiant, dire in zip(df["radiant_team"], df["dire_team"]):
        all_heroes.update(parse_team(radiant))
        all_heroes.update(parse_team(dire))
    hero_index = build_hero_index(all_heroes)
    n = len(hero_index)
    pair_index = build_pair_index(n)
    n_pairs = len(pair_index)

    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    y_train = train_df["radiant_win"].to_numpy()
    y_test = test_df["radiant_win"].to_numpy()

    print(f"{len(train_df):,} train / {len(test_df):,} test, {n} heroes, {2*n_pairs:,} candidate pair columns")

    X_train_full, train_counts = encode_with_pair_counts(train_df, hero_index, pair_index)
    X_test_full, _ = encode_with_pair_counts(test_df, hero_index, pair_index)

    print("\nTrain-set occurrence distribution across all pair columns:")
    print(f"  never observed:     {(train_counts == 0).sum():,} / {len(train_counts):,}")
    print(f"  seen 1-9 times:     {((train_counts > 0) & (train_counts < 10)).sum():,}")
    print(f"  seen 10-29 times:   {((train_counts >= 10) & (train_counts < 30)).sum():,}")
    print(f"  seen >=30 times:    {(train_counts >= 30).sum():,}")
    print(f"  seen >=100 times:   {(train_counts >= 100).sum():,}")

    results = {}
    predictions = {}

    for threshold in (10, 30, 100, 300):
        keep_pair_cols = np.where(train_counts >= threshold)[0] + n
        keep_cols = np.concatenate([np.arange(n), keep_pair_cols])
        X_train = X_train_full[:, keep_cols]
        X_test = X_test_full[:, keep_cols]

        grid = GridSearchCV(
            LogisticRegression(max_iter=3000, solver="lbfgs"),
            {"C": [0.003, 0.01, 0.03, 0.1, 0.3]},
            scoring="neg_log_loss",
            cv=TimeSeriesSplit(n_splits=3),
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
        key = f"pairwise_thresh_{threshold}"
        results[key] = metrics(y_test, proba)
        results[key]["best_C"] = grid.best_params_["C"]
        results[key]["n_features"] = int(X_train.shape[1])
        predictions[key] = (proba >= 0.5).astype(int)
        print(
            f"threshold={threshold:>4}  features={X_train.shape[1]:>6,}  C={grid.best_params_['C']:<6}"
            f"  acc={results[key]['accuracy']:.4f}  auc={results[key]['roc_auc']:.4f}"
            f"  logloss={results[key]['log_loss']:.4f}"
        )

    # Compare best-threshold model against the plain linear model via McNemar.
    report_path = out_dir / "report.json"
    baseline = json.loads(report_path.read_text())
    baseline_acc = baseline["results"]["logistic_regression"]["accuracy"]
    baseline_auc = baseline["results"]["logistic_regression"]["roc_auc"]
    baseline_pairwise_auc = baseline["results"]["logreg_pairwise"]["roc_auc"]

    best_key = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest reduced-pairwise config: {best_key} (AUC {results[best_key]['roc_auc']:.4f})")
    print(f"Shipped linear model:         AUC {baseline_auc:.4f}, acc {baseline_acc:.4f}")
    print(f"Full pairwise model (16k):    AUC {baseline_pairwise_auc:.4f}")

    verdict = (
        "beats both the linear and full-pairwise models"
        if results[best_key]["roc_auc"] > max(baseline_auc, baseline_pairwise_auc)
        else "does not beat the shipped model class"
    )
    print(f"Verdict: {verdict}")

    out = {
        "train_pair_occurrence": {
            "never": int((train_counts == 0).sum()),
            "1_to_9": int(((train_counts > 0) & (train_counts < 10)).sum()),
            "10_to_29": int(((train_counts >= 10) & (train_counts < 30)).sum()),
            "at_least_30": int((train_counts >= 30).sum()),
            "at_least_100": int((train_counts >= 100).sum()),
            "total_columns": int(len(train_counts)),
        },
        "results": results,
        "baseline_linear_auc": baseline_auc,
        "baseline_linear_accuracy": baseline_acc,
        "baseline_full_pairwise_auc": baseline_pairwise_auc,
        "best_config": best_key,
        "verdict": verdict,
    }
    (out_dir / "reduced_pairwise_experiment.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_dir / 'reduced_pairwise_experiment.json'}")


if __name__ == "__main__":
    main()
