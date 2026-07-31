"""How much data does it take to know a balance patch worked?

    python analysis/power.py

The balance analysis flags heroes whose win rate is off. The obvious next
question from anyone who has to act on that is: if we change the hero, how long
before we can tell whether the change did what we wanted?

That is a power calculation, and it is the difference between "Wraith King is
at 54.6%" (an observation) and "a 2-point nerf becomes detectable after N
matches, so do not re-tune before then" (a decision).

Setup: comparing a hero's win rate before and after a patch is a two-proportion
test. We need n hero-games in each period. Because a hero only appears in some
fraction of matches, the number of *matches* the population must play is much
larger than the number of hero-games needed -- and that conversion is the part
that actually matters operationally.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SQL = ROOT / "sql"
ARTIFACTS = ROOT / "artifacts"

ALPHA = 0.05
POWER = 0.80


def n_per_group(p1: float, p2: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """Sample size per group for a two-sided two-proportion z-test.

    Uses the pooled-variance form under H0 and the unpooled form under H1,
    which is the standard formulation and slightly more conservative than the
    simple version that uses one variance for both.
    """
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    numerator = (
        z_a * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return numerator / (p2 - p1) ** 2


def mde(n: float, p1: float = 0.5, alpha: float = ALPHA, power: float = POWER) -> float:
    """Smallest win-rate shift detectable with n hero-games per period."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return (z_a + z_b) * np.sqrt(2 * p1 * (1 - p1) / n)


def main() -> None:
    con = duckdb.connect()
    con.execute(f"SET VARIABLE matches_path = '{ROOT / 'data' / 'matches.csv.gz'}'")
    con.execute(f"SET VARIABLE heroes_path  = '{ROOT / 'data' / 'heroes.csv'}'")
    for script in sorted(SQL.glob("*.sql")):
        con.execute(script.read_text())

    heroes = con.execute(
        "SELECT hero_id, name, games, win_rate, pick_rate FROM hero_winrates "
        "WHERE games >= 500 ORDER BY pick_rate DESC"
    ).fetchdf()
    n_matches = int(con.execute("SELECT COUNT(*) FROM matches").fetchone()[0])

    # --- Sample size for a range of effect sizes -------------------------
    effects = [0.005, 0.01, 0.02, 0.03, 0.05]
    table = []
    for e in effects:
        n = n_per_group(0.50, 0.50 + e)
        table.append({"effect_pp": e * 100, "hero_games_per_period": int(np.ceil(n))})

    # --- Translate into matches, per hero, using real pick rates ---------
    # A hero appearing in `pick_rate` of matches accumulates one hero-game per
    # match it appears in, so matches_needed = hero_games / pick_rate.
    target = 0.02
    n_needed = n_per_group(0.50, 0.50 + target)
    heroes["matches_needed"] = np.ceil(n_needed / heroes["pick_rate"]).astype(int)
    heroes["mde_at_current_n"] = [mde(g) * 100 for g in heroes["games"]]

    most_picked = heroes.nlargest(5, "pick_rate")
    least_picked = heroes.nsmallest(5, "pick_rate")

    report = {
        "alpha": ALPHA,
        "power": POWER,
        "sample_size_table": table,
        "target_effect_pp": target * 100,
        "hero_games_needed_for_target": int(np.ceil(n_needed)),
        "dataset_matches": n_matches,
        "median_matches_needed": int(heroes["matches_needed"].median()),
        "median_mde_at_current_n_pp": float(heroes["mde_at_current_n"].median()),
        "most_picked": most_picked[
            ["name", "pick_rate", "games", "matches_needed", "mde_at_current_n"]
        ].to_dict("records"),
        "least_picked": least_picked[
            ["name", "pick_rate", "games", "matches_needed", "mde_at_current_n"]
        ].to_dict("records"),
    }
    (ARTIFACTS / "power.json").write_text(json.dumps(report, indent=2, default=float))
    heroes.to_csv(ARTIFACTS / "hero_power.csv", index=False)

    print(f"Two-proportion test, alpha={ALPHA}, power={POWER:.0%}\n")
    print("Effect to detect   Hero-games needed per period")
    for row in table:
        print(f"  {row['effect_pp']:>4.1f} pp          {row['hero_games_per_period']:>10,}")

    print(
        f"\nTo confirm a {target*100:.0f}pp shift you need "
        f"{int(np.ceil(n_needed)):,} hero-games before and after."
    )
    print(f"Median across heroes: {report['median_matches_needed']:,} matches must be played.")
    print(f"This dataset has {n_matches:,} matches.\n")

    print("Most-picked heroes (cheapest to evaluate):")
    for r in report["most_picked"]:
        print(
            f"  {r['name']:<20} pick {r['pick_rate']*100:>5.1f}%  "
            f"needs {r['matches_needed']:>8,} matches   "
            f"current MDE {r['mde_at_current_n']:.2f}pp"
        )
    print("\nLeast-picked heroes (most expensive):")
    for r in report["least_picked"]:
        print(
            f"  {r['name']:<20} pick {r['pick_rate']*100:>5.1f}%  "
            f"needs {r['matches_needed']:>8,} matches   "
            f"current MDE {r['mde_at_current_n']:.2f}pp"
        )


if __name__ == "__main__":
    main()
