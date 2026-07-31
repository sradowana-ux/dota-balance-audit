"""Figures for the balance audit.

    python analysis/figures.py

Writes:
    assets/hero_balance.png     forest plot, every hero with its CI
    assets/power_curve.png      matches needed to detect a patch effect
    assets/bracket_divergence.png  does balance mean the same thing at every rank
    assets/draft_determinism.png   distribution of draft-implied win probability
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

INK = "#1b1f24"
ACCENT = "#c8aa6e"
HOT = "#a33a3a"
COLD = "#2f6f9f"
MUTED = "#9aa2ad"

plt.rcParams.update({
    "figure.dpi": 160, "font.size": 9,
    "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def hero_balance_plot(balance: dict) -> None:
    df = pd.read_csv(ARTIFACTS / "hero_balance.csv").sort_values("win_rate")
    tol = balance["tolerance_pp"] / 100

    fig, ax = plt.subplots(figsize=(6.2, 12.5))
    y = np.arange(len(df))

    colours = np.where(
        df["ci_high"] < 0.5 - tol, COLD,
        np.where(df["ci_low"] > 0.5 + tol, HOT, MUTED),
    )

    ax.axvspan(0.5 - tol, 0.5 + tol, color=ACCENT, alpha=0.16, lw=0,
               label=f"±{balance['tolerance_pp']:.0f}pp tolerance band")
    ax.axvline(0.5, color=INK, lw=0.9)

    ax.hlines(y, df["ci_low"], df["ci_high"], color=colours, lw=1.5, alpha=0.9)
    ax.scatter(df["win_rate"], y, s=9, color=colours, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(df["name"], fontsize=6.2)
    ax.set_ylim(-1, len(df))
    ax.set_xlabel("win rate (95% Wilson interval)")
    ax.set_title(
        f"Hero balance across {balance['n_matches']:,} matches\n"
        f"{balance['hero_tests']['outside_tolerance']} of "
        f"{balance['roster']['heroes_tested']} heroes fall outside the tolerance band",
        pad=12,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "hero_balance.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/hero_balance.png")


def power_curve_plot(power: dict) -> None:
    df = pd.read_csv(ARTIFACTS / "hero_power.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    # Left: sample size vs effect size
    effects = np.linspace(0.005, 0.05, 200)
    from scipy import stats as st
    z_a, z_b = st.norm.ppf(0.975), st.norm.ppf(0.80)
    n = ((z_a * np.sqrt(2 * 0.25) + z_b * np.sqrt(2 * 0.25)) ** 2) / effects ** 2
    ax1.plot(effects * 100, n, color=ACCENT, lw=2)
    for e in (1, 2, 3):
        ne = ((z_a * np.sqrt(2 * 0.25) + z_b * np.sqrt(2 * 0.25)) ** 2) / (e / 100) ** 2
        ax1.plot([e], [ne], "o", color=HOT, ms=5)
        ax1.annotate(f"{e}pp → {ne:,.0f}", (e, ne), textcoords="offset points",
                     xytext=(8, 6), fontsize=8, color=INK)
    ax1.set_yscale("log")
    ax1.set_xlabel("win-rate change to detect (percentage points)")
    ax1.set_ylabel("hero-games needed per period (log scale)")
    ax1.set_title("Detecting a balance patch\n80% power, α=0.05", pad=10)

    # Right: matches needed, by pick rate
    ax2.scatter(df["pick_rate"] * 100, df["matches_needed"], s=14,
                color=ACCENT, edgecolor="none", alpha=0.85)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.axhline(power["dataset_matches"], color=HOT, lw=1.2, ls="--",
                label=f"this dataset ({power['dataset_matches']:,})")
    for _, r in df.nsmallest(2, "pick_rate").iterrows():
        ax2.annotate(r["name"], (r["pick_rate"] * 100, r["matches_needed"]),
                     textcoords="offset points", xytext=(6, -3), fontsize=7.5, color=INK)
    for _, r in df.nlargest(1, "pick_rate").iterrows():
        ax2.annotate(r["name"], (r["pick_rate"] * 100, r["matches_needed"]),
                     textcoords="offset points", xytext=(-38, -3), fontsize=7.5, color=INK)
    ax2.set_xlabel("hero pick rate (%, log scale)")
    ax2.set_ylabel("matches that must be played (log scale)")
    ax2.set_title(f"Cost of confirming a {power['target_effect_pp']:.0f}pp change\n"
                  "per hero, using real pick rates", pad=10)
    ax2.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(ASSETS / "power_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/power_curve.png")


def bracket_plot(balance: dict) -> None:
    bc = balance["bracket_consistency"]
    if "correlation" not in bc:
        return

    con = duckdb.connect()
    con.execute(f"SET VARIABLE matches_path = '{ROOT / 'data' / 'matches.csv.gz'}'")
    con.execute(f"SET VARIABLE heroes_path  = '{ROOT / 'data' / 'heroes.csv'}'")
    for script in sorted((ROOT / "sql").glob("*.sql")):
        con.execute(script.read_text())
    d = con.execute(
        "SELECT bracket, hero_id, name, win_rate FROM hero_winrates_by_bracket"
    ).fetchdf()

    pivot = d.pivot(index="hero_id", columns="bracket", values="win_rate")
    names = d.drop_duplicates("hero_id").set_index("hero_id")["name"]
    low, high = bc["low_bracket"], bc["high_bracket"]
    both = pivot[[low, high]].dropna()

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.axvline(0.5, color=MUTED, lw=0.8)
    lim = [both.values.min() - 0.02, both.values.max() + 0.02]
    ax.plot(lim, lim, "--", color=MUTED, lw=1, label="identical at both ranks")
    ax.scatter(both[low], both[high], s=18, color=ACCENT, edgecolor="none")

    diverge = (both[high] - both[low]).abs().nlargest(4).index
    for idx in diverge:
        ax.annotate(names.get(idx, str(idx)),
                    (both.loc[idx, low], both.loc[idx, high]),
                    textcoords="offset points", xytext=(6, 3), fontsize=7.5, color=HOT)

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"win rate at bracket {low} (low rank)")
    ax.set_ylabel(f"win rate at bracket {high} (high rank)")
    ax.set_title(
        f"Does balance mean the same thing at every rank?\n"
        f"r = {bc['correlation']:.2f} across {bc['n_heroes']} heroes", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "bracket_divergence.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/bracket_divergence.png")


def determinism_plot(det: dict) -> None:
    import joblib, sys
    sys.path.insert(0, str(ROOT / "src"))
    from features import build_hero_index, encode_drafts, load_matches, parse_team

    df = load_matches(ROOT / "data" / "matches.csv.gz", game_mode=22)
    heroes = set()
    for r, dd in zip(df["radiant_team"], df["dire_team"]):
        heroes.update(parse_team(r)); heroes.update(parse_team(dd))
    hi = build_hero_index(heroes)
    split = int(len(df) * 0.8)
    X = encode_drafts(df.iloc[split:], hi)
    model = joblib.load(ARTIFACTS / "model.joblib")["model"]
    p = model.predict_proba(X)[:, 1]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(p, bins=60, color=ACCENT, edgecolor="white", linewidth=0.3)
    ax.axvline(0.5, color=INK, lw=1)
    ax.axvspan(0.45, 0.55, color=MUTED, alpha=0.18, lw=0,
               label="within 5pp of even")
    ax.set_xlabel("win probability implied by the draft alone")
    ax.set_ylabel("matches")
    ax.set_title(
        f"How much is settled at the draft?\n"
        f"{det['share_meaningfully_decided']*100:.0f}% of drafts move it more than 5pp; "
        f"range {det['probability_spread']['min']:.2f}–{det['probability_spread']['max']:.2f}",
        pad=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "draft_determinism.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote assets/draft_determinism.png")


def main() -> None:
    balance = json.loads((ARTIFACTS / "balance.json").read_text())
    power = json.loads((ARTIFACTS / "power.json").read_text())
    det = json.loads((ARTIFACTS / "determinism.json").read_text())

    hero_balance_plot(balance)
    power_curve_plot(power)
    bracket_plot(balance)
    determinism_plot(det)


if __name__ == "__main__":
    main()
