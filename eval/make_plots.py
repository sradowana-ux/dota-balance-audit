"""Generate the evaluation figures referenced by the README.

    python eval/make_plots.py

Produces:
    assets/calibration.png   reliability curve for the logistic model
    assets/hero_effects.png  the 15 strongest positive and negative hero weights
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

INK = "#1b1f24"
ACCENT = "#c8aa6e"
RADIANT = "#3f8f4f"
DIRE = "#a33a3a"

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "font.size": 9,
        "axes.edgecolor": "#c9ced6",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def calibration_plot(report: dict) -> None:
    prob_pred = np.asarray(report["calibration"]["prob_pred"])
    prob_true = np.asarray(report["calibration"]["prob_true"])

    # The model only ever emits probabilities in a narrow band around 0.5 --
    # that is the honest consequence of a weak signal, not a bug -- so plotting
    # the full 0-1 range would compress all the informative structure into one
    # corner. Zoom to the observed range instead and state the range on the plot.
    lo = min(prob_pred.min(), prob_true.min()) - 0.03
    hi = max(prob_pred.max(), prob_true.max()) + 0.03

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([lo, hi], [lo, hi], "--", color="#9aa2ad", lw=1, label="perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", color=ACCENT, lw=1.8, ms=5, label="model")

    brier = report["results"]["logistic_regression"]["brier"]
    ax.set_xlabel("predicted Radiant win probability")
    ax.set_ylabel("observed Radiant win rate")
    ax.set_title(f"Calibration on held-out matches (Brier {brier:.4f})", pad=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.annotate(
        f"model output spans {prob_pred.min():.2f}-{prob_pred.max():.2f}\n"
        "(deciles of the held-out set)",
        xy=(0.97, 0.04),
        xycoords="axes fraction",
        ha="right",
        fontsize=7.5,
        color="#5a6270",
    )
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "calibration.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/calibration.png")


def hero_effects_plot(bundle: dict, heroes: pd.DataFrame, top_n: int = 15) -> None:
    coefs = bundle["model"].coef_[0]
    hero_index = bundle["hero_index"]
    id_to_name = dict(zip(heroes["hero_id"], heroes["name"]))

    rows = [
        (id_to_name.get(hero_id, str(hero_id)), float(coefs[col]))
        for hero_id, col in hero_index.items()
    ]
    frame = pd.DataFrame(rows, columns=["hero", "weight"]).sort_values("weight")
    subset = pd.concat([frame.head(top_n), frame.tail(top_n)])

    fig, ax = plt.subplots(figsize=(5.6, 7.2))
    colours = [RADIANT if w > 0 else DIRE for w in subset["weight"]]
    ax.barh(subset["hero"], subset["weight"], color=colours, height=0.72)
    ax.axvline(0, color="#9aa2ad", lw=0.9)
    ax.set_xlabel("logistic-regression weight (log-odds contribution to its own side)")
    ax.set_title(
        f"Strongest {top_n} hero effects in each direction\n"
        "positive = the hero raises its own team's win odds",
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "hero_effects.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/hero_effects.png")


def main() -> None:
    report = json.loads((ARTIFACTS / "report.json").read_text())
    bundle = joblib.load(ARTIFACTS / "model.joblib")
    heroes = pd.read_csv(ROOT / "data" / "heroes.csv")

    calibration_plot(report)
    hero_effects_plot(bundle, heroes)


if __name__ == "__main__":
    main()
