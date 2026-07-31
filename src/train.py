"""Train and honestly evaluate draft-only win prediction models.

Run:
    python src/train.py --data data/matches.csv.gz

What this script deliberately does:

  * Splits chronologically, not randomly. Test matches are strictly newer than
    train matches, so nothing from the future leaks backwards.
  * Compares against two baselines, one trivial and one non-trivial. Beating
    "always predict Radiant" is not evidence of anything; beating a hero
    win-rate lookup is.
  * Reports calibration (Brier, reliability curve) alongside accuracy, because
    a win-probability model that is 58% accurate but badly calibrated is not
    useful for the thing people actually want it for.
  * Runs McNemar's test on the paired test-set predictions, so the reported
    improvement comes with a p-value rather than a vibe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from statsmodels.stats.contingency_tables import mcnemar

from features import (
    build_hero_index,
    build_pair_index,
    encode_drafts,
    encode_drafts_pairwise,
    load_matches,
    parse_team,
)

RANDOM_STATE = 42


def metrics(y_true, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def hero_winrate_baseline(train_df, test_df, hero_index):
    """Non-trivial baseline: sum of solo hero win rates, Radiant minus Dire.

    This is what a competent person would do without machine learning -- look
    up each hero's win rate and add them up. Any credit the model claims has to
    be credit over this, not over coin-flipping.
    """
    wins = np.zeros(len(hero_index))
    games = np.zeros(len(hero_index))

    for radiant, dire, radiant_win in zip(
        train_df["radiant_team"], train_df["dire_team"], train_df["radiant_win"]
    ):
        for hero in parse_team(radiant):
            col = hero_index.get(hero)
            if col is not None:
                games[col] += 1
                wins[col] += radiant_win
        for hero in parse_team(dire):
            col = hero_index.get(hero)
            if col is not None:
                games[col] += 1
                wins[col] += 1 - radiant_win

    prior = train_df["radiant_win"].mean()
    # Laplace-style shrinkage towards the global prior for rarely picked heroes.
    winrate = (wins + 25 * prior) / (games + 25)

    scores = []
    for radiant, dire in zip(test_df["radiant_team"], test_df["dire_team"]):
        r = sum(winrate[hero_index[h]] for h in parse_team(radiant) if h in hero_index)
        d = sum(winrate[hero_index[h]] for h in parse_team(dire) if h in hero_index)
        scores.append(r - d)

    scores = np.asarray(scores)
    # Map the score differential onto a probability with a 1-D logistic fit so
    # that log loss and Brier are meaningful rather than arbitrary.
    calibrator = LogisticRegression()
    train_scores = []
    for radiant, dire in zip(train_df["radiant_team"], train_df["dire_team"]):
        r = sum(winrate[hero_index[h]] for h in parse_team(radiant) if h in hero_index)
        d = sum(winrate[hero_index[h]] for h in parse_team(dire) if h in hero_index)
        train_scores.append(r - d)
    calibrator.fit(np.asarray(train_scores).reshape(-1, 1), train_df["radiant_win"])
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1], winrate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/matches.csv.gz")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--assets-dir", default="assets")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    df = load_matches(args.data, game_mode=22)
    print(f"Loaded {len(df):,} All Draft matches")

    all_heroes = set()
    for radiant, dire in zip(df["radiant_team"], df["dire_team"]):
        all_heroes.update(parse_team(radiant))
        all_heroes.update(parse_team(dire))
    hero_index = build_hero_index(all_heroes)
    print(f"{len(hero_index)} distinct heroes")

    split = int(len(df) * (1 - args.test_frac))
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    print(f"Chronological split: {len(train_df):,} train / {len(test_df):,} test")

    X_train = encode_drafts(train_df, hero_index)
    X_test = encode_drafts(test_df, hero_index)
    y_train = train_df["radiant_win"].to_numpy()
    y_test = test_df["radiant_win"].to_numpy()

    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}

    # --- Baseline 1: majority class -------------------------------------
    prior = float(y_train.mean())
    majority_proba = np.full(len(y_test), prior)
    results["majority_class"] = metrics(y_test, majority_proba)
    predictions["majority_class"] = (majority_proba >= 0.5).astype(int)
    # ROC-AUC is undefined for a constant predictor; record it as 0.5 exactly.
    results["majority_class"]["roc_auc"] = 0.5

    # --- Baseline 2: solo hero win-rate differential --------------------
    wr_proba, winrate = hero_winrate_baseline(train_df, test_df, hero_index)
    results["hero_winrate_sum"] = metrics(y_test, wr_proba)
    predictions["hero_winrate_sum"] = (wr_proba >= 0.5).astype(int)

    # --- Model 1: L2 logistic regression on signed hero features --------
    grid = GridSearchCV(
        LogisticRegression(max_iter=2000, solver="lbfgs"),
        {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
        scoring="neg_log_loss",
        cv=TimeSeriesSplit(n_splits=4),
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    logreg = grid.best_estimator_
    print(f"Logistic regression best C = {grid.best_params_['C']}")
    lr_proba = logreg.predict_proba(X_test)[:, 1]
    results["logistic_regression"] = metrics(y_test, lr_proba)
    results["logistic_regression"]["best_C"] = grid.best_params_["C"]
    predictions["logistic_regression"] = (lr_proba >= 0.5).astype(int)

    # --- Model 2: logistic regression + synergy/counter pair features ---
    # Roughly 16k columns against ~43k rows, so this only stands up under
    # strong L2. Included to test the standard claim that draft models need
    # pairwise interactions, rather than to assume it.
    pair_index = build_pair_index(len(hero_index))
    Xp_train = encode_drafts_pairwise(train_df, hero_index, pair_index)
    Xp_test = encode_drafts_pairwise(test_df, hero_index, pair_index)
    pair_grid = GridSearchCV(
        LogisticRegression(max_iter=3000, solver="lbfgs"),
        {"C": [0.0003, 0.001, 0.003, 0.01]},
        scoring="neg_log_loss",
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=-1,
    )
    pair_grid.fit(Xp_train, y_train)
    print(f"Pairwise model best C = {pair_grid.best_params_['C']}")
    pair_proba = pair_grid.best_estimator_.predict_proba(Xp_test)[:, 1]
    results["logreg_pairwise"] = metrics(y_test, pair_proba)
    results["logreg_pairwise"]["best_C"] = pair_grid.best_params_["C"]
    results["logreg_pairwise"]["n_features"] = int(Xp_train.shape[1])
    predictions["logreg_pairwise"] = (pair_proba >= 0.5).astype(int)

    # --- Model 3: gradient boosting -------------------------------------
    gbm = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=RANDOM_STATE,
    )
    gbm.fit(X_train, y_train)
    gbm_proba = gbm.predict_proba(X_test)[:, 1]
    results["gradient_boosting"] = metrics(y_test, gbm_proba)
    predictions["gradient_boosting"] = (gbm_proba >= 0.5).astype(int)

    # --- McNemar's test: logistic regression vs each baseline -----------
    significance = {}
    lr_correct = predictions["logistic_regression"] == y_test
    for name in (
        "majority_class",
        "hero_winrate_sum",
        "logreg_pairwise",
        "gradient_boosting",
    ):
        other_correct = predictions[name] == y_test
        table = [
            [
                int(np.sum(lr_correct & other_correct)),
                int(np.sum(lr_correct & ~other_correct)),
            ],
            [
                int(np.sum(~lr_correct & other_correct)),
                int(np.sum(~lr_correct & ~other_correct)),
            ],
        ]
        result = mcnemar(table, exact=False, correction=True)
        significance[f"logreg_vs_{name}"] = {
            "table": table,
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
        }

    # --- Calibration ------------------------------------------------------
    prob_true, prob_pred = calibration_curve(y_test, lr_proba, n_bins=10, strategy="quantile")
    calibration = {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }

    report = {
        "n_matches": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_heroes": len(hero_index),
        "radiant_win_rate_train": prior,
        "radiant_win_rate_test": float(y_test.mean()),
        "results": results,
        "significance": significance,
        "calibration": calibration,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    joblib.dump(
        {
            "model": logreg,
            "hero_index": hero_index,
            "solo_winrate": winrate,
            "n_train": int(len(train_df)),
        },
        out_dir / "model.joblib",
    )

    # --- Console summary --------------------------------------------------
    print()
    header = f"{'model':<22}{'acc':>8}{'logloss':>10}{'auc':>8}{'brier':>8}"
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        print(
            f"{name:<22}{scores['accuracy']:>8.4f}{scores['log_loss']:>10.4f}"
            f"{scores['roc_auc']:>8.4f}{scores['brier']:>8.4f}"
        )
    print()
    for name, sig in significance.items():
        print(f"{name}: chi2={sig['statistic']:.2f}  p={sig['p_value']:.3e}")


if __name__ == "__main__":
    main()
