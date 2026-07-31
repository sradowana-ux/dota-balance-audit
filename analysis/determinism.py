"""How much does the draft decide? Using the model as a measuring instrument.

    python analysis/determinism.py

Hero win rates answer "is any single hero too strong". They do not answer the
question a designer actually cares about: **can a team win the game in the
draft?** That is a property of ten heroes interacting, not of one.

The win-probability model is repurposed here as an instrument rather than a
product. If drafts routinely produced lopsided matchups, a model that sees only
the draft would be able to separate winners from losers. The degree to which it
*cannot* is a direct measurement of how little the draft decides -- which is
what "the game is balanced" means operationally.

Three numbers come out of this:

* **Draft-implied win probability spread.** How far from 50/50 does the most
  lopsided real draft get? This is the headline: it bounds how much advantage
  is available at the draft stage.
* **Share of matches meaningfully decided.** What fraction of drafts move the
  win probability more than 5 points off even?
* **The information ceiling.** Refitting on all data with no regularisation and
  scoring in-sample gives the best a hero-additive model could ever do here.
  If the honest held-out number is close to that ceiling, the limit is the
  information in a draft, not the modelling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_hero_index, encode_drafts, load_matches, parse_team  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
MEANINGFUL = 0.05      # 5 percentage points off even


def main() -> None:
    df = load_matches(ROOT / "data" / "matches.csv.gz", game_mode=22)
    heroes = set()
    for r, d in zip(df["radiant_team"], df["dire_team"]):
        heroes.update(parse_team(r)); heroes.update(parse_team(d))
    hero_index = build_hero_index(heroes)
    X = encode_drafts(df, hero_index)
    y = df["radiant_win"].to_numpy()

    split = int(len(df) * 0.8)

    # Honest held-out model, same settings as the shipped one.
    bundle = joblib.load(ARTIFACTS / "model.joblib")
    model = bundle["model"]
    proba_test = model.predict_proba(X[split:])[:, 1]

    # Information ceiling: fit and score on everything, no regularisation.
    ceiling_model = LogisticRegression(max_iter=3000, C=100).fit(X, y)
    proba_all = ceiling_model.predict_proba(X)[:, 1]
    ceiling_acc = accuracy_score(y, proba_all >= 0.5)
    ceiling_auc = roc_auc_score(y, proba_all)

    edge = np.abs(proba_test - 0.5)
    report = {
        "n_test_matches": int(len(proba_test)),
        "held_out_accuracy": float(accuracy_score(y[split:], proba_test >= 0.5)),
        "held_out_auc": float(roc_auc_score(y[split:], proba_test)),
        "ceiling_accuracy": float(ceiling_acc),
        "ceiling_auc": float(ceiling_auc),
        "gap_to_ceiling_pp": float((ceiling_acc - accuracy_score(y[split:], proba_test >= 0.5)) * 100),
        "probability_spread": {
            "min": float(proba_test.min()),
            "max": float(proba_test.max()),
            "p1": float(np.percentile(proba_test, 1)),
            "p99": float(np.percentile(proba_test, 99)),
            "sd_pp": float(proba_test.std() * 100),
        },
        "share_meaningfully_decided": float((edge > MEANINGFUL).mean()),
        "meaningful_threshold_pp": MEANINGFUL * 100,
        "median_edge_pp": float(np.median(edge) * 100),
    }
    (ARTIFACTS / "determinism.json").write_text(json.dumps(report, indent=2))

    s = report["probability_spread"]
    print(f"Held-out accuracy {report['held_out_accuracy']:.4f}  "
          f"AUC {report['held_out_auc']:.4f}")
    print(f"Information ceiling {ceiling_acc:.4f}  AUC {ceiling_auc:.4f}  "
          f"(gap {report['gap_to_ceiling_pp']:.2f}pp)")
    print()
    print(f"Draft-implied win probability across {report['n_test_matches']:,} real drafts:")
    print(f"  full range      {s['min']:.3f} - {s['max']:.3f}")
    print(f"  1st-99th pct    {s['p1']:.3f} - {s['p99']:.3f}")
    print(f"  sd              {s['sd_pp']:.2f}pp")
    print()
    print(f"Median draft advantage: {report['median_edge_pp']:.2f}pp off even")
    print(f"Drafts moving win probability >{MEANINGFUL*100:.0f}pp: "
          f"{report['share_meaningfully_decided']*100:.1f}%")


if __name__ == "__main__":
    main()
