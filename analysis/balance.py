"""Is Dota 2 balanced? The statistics layer.

    python analysis/balance.py

Reads the DuckDB aggregates built by sql/, applies the tests, and writes
artifacts/balance.json.

Three things this does that a win-rate table does not:

1. **Confidence intervals, not point estimates.** A hero at 47% over 400 games
   and a hero at 47% over 9,000 games are different claims. Wilson intervals
   are used rather than normal-approximation intervals because they behave
   correctly for proportions near 0.5 at modest n, which is exactly this case.

2. **Multiple-comparison control.** Testing ~126 heroes against 50% at
   alpha = 0.05 produces about six "significant" heroes by chance alone. Without
   correction, a balanced game looks broken. Benjamini-Hochberg controls the
   false discovery rate, which is the right choice here: we are screening for
   candidates to investigate, not trying to avoid a single false positive at
   any cost, so FDR is more appropriate (and more powerful) than Bonferroni.

3. **A tolerance band, separate from significance.** With 50,000 matches, a
   hero at 51.2% can be statistically distinguishable from 50% while being
   completely irrelevant to a balance decision. Statistical significance and
   practical significance are different questions, and conflating them is the
   most common way analysts mislead their own stakeholders. The band is set at
   +/- 2 percentage points and is stated as a judgement, not a discovery.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
SQL = ROOT / "sql"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

MIN_GAMES = 500          # below this a hero win rate is too noisy to act on
TOLERANCE = 0.02         # +/- 2pp is the practical balance band
ALPHA = 0.05


def build_database() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET VARIABLE matches_path = '{ROOT / 'data' / 'matches.csv.gz'}'")
    con.execute(f"SET VARIABLE heroes_path  = '{ROOT / 'data' / 'heroes.csv'}'")
    for script in sorted(SQL.glob("*.sql")):
        con.execute(script.read_text())

    bad = con.execute("SELECT COUNT(*) FROM integrity_check").fetchone()[0]
    if bad:
        raise SystemExit(
            f"{bad} matches did not unpivot to exactly 10 hero rows - aborting."
        )
    return con


def wilson_interval(wins: np.ndarray, games: np.ndarray, alpha: float = ALPHA):
    """Wilson score interval for a binomial proportion."""
    z = stats.norm.ppf(1 - alpha / 2)
    p = wins / games
    denom = 1 + z**2 / games
    centre = (p + z**2 / (2 * games)) / denom
    half = z * np.sqrt(p * (1 - p) / games + z**2 / (4 * games**2)) / denom
    return centre - half, centre + half


def test_heroes(con) -> pd.DataFrame:
    df = con.execute(
        f"SELECT * FROM hero_winrates WHERE games >= {MIN_GAMES} ORDER BY win_rate"
    ).fetchdf()

    # Two-sided exact binomial test against a fair 50%.
    df["p_value"] = [
        stats.binomtest(int(w), int(n), 0.5).pvalue
        for w, n in zip(df["wins"], df["games"])
    ]

    reject, q_values, _, _ = multipletests(df["p_value"], alpha=ALPHA, method="fdr_bh")
    df["q_value"] = q_values
    df["significant"] = reject

    lo, hi = wilson_interval(df["wins"].to_numpy(), df["games"].to_numpy())
    df["ci_low"], df["ci_high"] = lo, hi

    # Practical significance: is the whole interval outside the tolerance band?
    # Requiring the interval (not the point estimate) to clear the band means we
    # only flag heroes where the data rules out "merely a bit off".
    df["outside_tolerance"] = (df["ci_low"] > 0.5 + TOLERANCE) | (
        df["ci_high"] < 0.5 - TOLERANCE
    )
    df["action"] = np.where(
        df["outside_tolerance"] & (df["win_rate"] > 0.5), "nerf candidate",
        np.where(df["outside_tolerance"] & (df["win_rate"] < 0.5), "buff candidate", "within tolerance"),
    )
    return df


def test_side(con) -> dict:
    row = con.execute("SELECT * FROM side_balance").fetchdf().iloc[0]
    wins, games = int(row["radiant_wins"]), int(row["matches"])
    result = stats.binomtest(wins, games, 0.5)
    lo, hi = wilson_interval(np.array([wins]), np.array([games]))
    return {
        "matches": games,
        "radiant_wins": wins,
        "radiant_win_rate": wins / games,
        "p_value": float(result.pvalue),
        "ci_low": float(lo[0]),
        "ci_high": float(hi[0]),
        "advantage_pp": (wins / games - 0.5) * 100,
    }


def bracket_consistency(con, heroes: pd.DataFrame) -> dict:
    """Does a hero's balance depend on who is playing it?

    A roster can look balanced on average while being badly balanced at every
    individual skill level, if the errors cancel. Correlating bracket win rates
    against each other tests whether "balanced" means the same thing at Herald
    and at Divine.
    """
    by_bracket = con.execute(
        "SELECT bracket, hero_id, games, wins, win_rate FROM hero_winrates_by_bracket"
    ).fetchdf()

    pivot = by_bracket.pivot(index="hero_id", columns="bracket", values="win_rate")
    counts = by_bracket.pivot(index="hero_id", columns="bracket", values="games")

    # Compare the lowest and highest brackets that have enough data.
    usable = [b for b in pivot.columns if counts[b].notna().sum() >= 60]
    out = {"brackets_compared": [int(b) for b in usable]}
    if len(usable) >= 2:
        low, high = usable[0], usable[-1]
        both = pivot[[low, high]].dropna()
        r, p = stats.pearsonr(both[low], both[high])
        out.update(
            {
                "low_bracket": int(low),
                "high_bracket": int(high),
                "n_heroes": int(len(both)),
                "correlation": float(r),
                "p_value": float(p),
                "biggest_divergence": [
                    {
                        "hero_id": int(idx),
                        "low": float(both.loc[idx, low]),
                        "high": float(both.loc[idx, high]),
                        "delta_pp": float((both.loc[idx, high] - both.loc[idx, low]) * 100),
                    }
                    for idx in (both[high] - both[low]).abs().nlargest(5).index
                ],
            }
        )
    return out


def main() -> None:
    con = build_database()

    spread = con.execute("SELECT * FROM roster_spread").fetchdf().iloc[0]
    heroes = test_heroes(con)
    side = test_side(con)
    brackets = bracket_consistency(con, heroes)

    n_tested = len(heroes)
    n_significant = int(heroes["significant"].sum())
    n_outside = int(heroes["outside_tolerance"].sum())
    expected_false = ALPHA * n_tested

    report = {
        "n_matches": int(con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]),
        "min_games_threshold": MIN_GAMES,
        "tolerance_pp": TOLERANCE * 100,
        "alpha": ALPHA,
        "roster": {
            "heroes_tested": n_tested,
            "min_win_rate": float(spread["min_win_rate"]),
            "max_win_rate": float(spread["max_win_rate"]),
            "spread_pp": float(spread["spread"]) * 100,
            "sd_pp": float(spread["sd"]) * 100,
        },
        "hero_tests": {
            "significant_uncorrected": int((heroes["p_value"] < ALPHA).sum()),
            "expected_false_positives": expected_false,
            "significant_after_bh": n_significant,
            "outside_tolerance": n_outside,
            "nerf_candidates": heroes.loc[heroes["action"] == "nerf candidate", "name"].tolist(),
            "buff_candidates": heroes.loc[heroes["action"] == "buff candidate", "name"].tolist(),
        },
        "side_balance": side,
        "bracket_consistency": brackets,
    }

    (ARTIFACTS / "balance.json").write_text(json.dumps(report, indent=2))
    heroes.to_csv(ARTIFACTS / "hero_balance.csv", index=False)

    print(f"Matches analysed:            {report['n_matches']:,}")
    print(f"Heroes with >= {MIN_GAMES} games:   {n_tested}")
    print(
        f"Win-rate range:              {spread['min_win_rate']:.3f} - "
        f"{spread['max_win_rate']:.3f}  (sd {spread['sd']*100:.2f}pp)"
    )
    print()
    print(f"Significant before correction: {report['hero_tests']['significant_uncorrected']}"
          f"  (~{expected_false:.0f} expected by chance)")
    print(f"Significant after BH:          {n_significant}")
    print(f"Outside +/-{TOLERANCE*100:.0f}pp tolerance:      {n_outside}")
    print()
    print(f"Radiant win rate: {side['radiant_win_rate']:.4f} "
          f"[{side['ci_low']:.4f}, {side['ci_high']:.4f}]  p={side['p_value']:.2e}")
    if "correlation" in brackets:
        print(
            f"Bracket {brackets['low_bracket']} vs {brackets['high_bracket']} "
            f"hero win-rate correlation: r={brackets['correlation']:.3f} "
            f"(n={brackets['n_heroes']}, p={brackets['p_value']:.2e})"
        )


if __name__ == "__main__":
    main()
