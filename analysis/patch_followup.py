"""Patch 7.41e follow-up: did the audit's flagged heroes move the way it said?

The balance audit (see ../README.md, finding #2) flagged 18 of 126 heroes as
"actionable" -- 3 whose win rate sits above tolerance ("nerf candidate") and 15
below it ("buff candidate") -- from a dataset collected in a ~1.9 hour window
inside patch 7.41d, on 2026-07-29. Patch 7.41e shipped roughly 30 hours later.
This script checks the flagged heroes' win rates against live public data
pulled on 2026-08-01 (data/live_herostats_2026-08-01.csv, from OpenDota's
rolling /api/heroStats aggregate, summed across all public skill brackets).

Two things this script deliberately does NOT let you conclude on their own:

  1. "N of 18 moved the predicted direction" is not, by itself, evidence the
     audit works. My snapshot was noisy (a two-hour window); the live figure
     is drawn from a vastly larger sample. Extreme values in a noisy sample
     regress toward the population mean on their own, with no patch involved.
     This script fits that regression-to-the-mean relationship across *all*
     126 heroes (not just the flagged ones) and reports it, so the flagged-
     hero hit rate can be read against that baseline instead of in isolation.

  2. Not every flagged hero was actually touched by patch 7.41e. PATCH_741E_
     CHANGES below is manually curated from patch-note trackers (see the
     `source` field in each entry) and only covers heroes this audit flagged.
     Untouched flagged heroes serve as a same-population control: whatever
     they did between the two snapshots is meta drift or reversion, not a
     patch effect, because Valve did not touch them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

# Manually curated from dota2protracker.com/patches/7.41e (accessed 2026-08-01)
# and cross-checked against the official 7.41e patch notes. "direction" is my
# read of whether the change is pro-hero (buff) or anti-hero (nerf) from the
# hero's own perspective; several patches are mixed and are marked as such.
# Only heroes this audit flagged are listed -- the other ~10 flagged heroes
# received no changes in 7.41e and serve as the control group below.
PATCH_741E_CHANGES = {
    "Snapfire": {
        "direction": "nerf",
        "detail": "Mortimer Kisses burn DPS down, Firesnap Cookie cooldown "
                   "talent replaced, Mortimer Kisses launch-count talent down.",
    },
    "Spectre": {
        "direction": "nerf",
        "detail": "Level 25 illusion-damage talent down; Haunt illusion "
                   "damage rescaled down at high levels.",
    },
    "Doom": {
        "direction": "mixed",
        "detail": "Attack range cut 200->175 (nerf), but Infernal Blade burn "
                   "damage and max-HP-as-damage scaling both increased (buff).",
    },
    "Gyrocopter": {
        "direction": "buff",
        "detail": "Rocket Barrage and Homing Missile mana costs cut; Homing "
                   "Missile cooldown cut.",
    },
    "Beastmaster": {
        "direction": "nerf",
        "detail": "Level 20 damage talent and Level 25 Primal Roar cooldown "
                   "talent both weakened.",
    },
    "Queen of Pain": {
        "direction": "buff",
        "detail": "Shadow Strike mana cost cut; Sonic Wave damage now stacks "
                   "on quick double-cast.",
    },
    "Muerta": {
        "direction": "neutral",
        "detail": "Gunslinger toggle no longer breaks invisibility -- a "
                   "quality-of-life fix, not a power change.",
    },
    "Tiny": {
        "direction": "nerf",
        "detail": "Base health regen reduced.",
    },
}


def load_data():
    mine = pd.read_csv(ARTIFACTS / "hero_balance.csv")
    live_paths = sorted(ROOT.glob("data/live_herostats_*.csv"))
    live = pd.read_csv(live_paths[-1])
    live_date = live_paths[-1].stem.replace("live_herostats_", "")
    return mine, live, live_date


def main() -> None:
    mine, live, live_date = load_data()
    df = mine.merge(live, on="hero_id", suffixes=("_mine", "_live"))
    assert len(df) == len(mine), "not every audited hero found in live snapshot"

    df["delta_pp"] = (df["win_rate_live"] - df["win_rate_mine"]) * 100
    df["mine_offset_pp"] = (df["win_rate_mine"] - 0.5) * 100

    def predicted_dir(action):
        return {"nerf candidate": -1, "buff candidate": 1}.get(action, 0)

    df["predicted_dir"] = df["action"].apply(predicted_dir)
    df["actual_dir"] = np.sign(df["delta_pp"])
    df["matches_prediction"] = np.where(
        df["predicted_dir"] != 0, df["actual_dir"] == df["predicted_dir"], np.nan
    )
    df["patch_touched"] = df["name_mine"].map(
        lambda n: PATCH_741E_CHANGES.get(n, {}).get("direction", "untouched")
    )

    flagged = df[df["action"] != "within tolerance"].copy()
    n_flagged = len(flagged)
    n_match = int(flagged["matches_prediction"].sum())

    from scipy.stats import binomtest
    direction_test = binomtest(n_match, n_flagged, 0.5)

    # Regression-to-the-mean control: fit on ALL heroes, not just flagged.
    x = df["mine_offset_pp"].to_numpy()
    y = df["delta_pp"].to_numpy()
    slope, intercept, r, p, se = stats.linregress(x, y)

    # Same regression fit on the *unflagged* heroes only, then used to predict
    # what the flagged heroes "should" have done from noise alone. If a
    # flagged hero's actual delta is more extreme in the predicted direction
    # than this null model expects, that's evidence beyond pure reversion.
    unflagged = df[df["action"] == "within tolerance"]
    null_slope, null_intercept, null_r, null_p, null_se = stats.linregress(
        unflagged["mine_offset_pp"], unflagged["delta_pp"]
    )
    flagged["null_predicted_delta"] = null_intercept + null_slope * flagged["mine_offset_pp"]
    flagged["residual_vs_null"] = flagged["delta_pp"] - flagged["null_predicted_delta"]
    # Positive residual in the predicted direction = beats the noise-only null.
    flagged["residual_beats_null"] = np.where(
        flagged["predicted_dir"] == 1, flagged["residual_vs_null"] > 0,
        np.where(flagged["predicted_dir"] == -1, flagged["residual_vs_null"] < 0, np.nan),
    )
    n_beats_null = int(flagged["residual_beats_null"].sum())
    beats_null_test = binomtest(n_beats_null, n_flagged, 0.5)

    touched = flagged[flagged["patch_touched"] != "untouched"]
    untouched = flagged[flagged["patch_touched"] == "untouched"]

    print(f"Live snapshot date: {live_date}")
    print(f"\n{n_match}/{n_flagged} flagged heroes moved in the predicted direction "
          f"(binomial p={direction_test.pvalue:.3f})")
    print(f"{n_beats_null}/{n_flagged} beat the reversion-to-the-mean null model in "
          f"the predicted direction (binomial p={beats_null_test.pvalue:.3f})")
    print(f"\nReversion-to-mean, all 126 heroes: delta_pp = {intercept:.3f} + "
          f"{slope:.4f} * offset_from_50pp  (r={r:.3f}, p={p:.2e})")
    print(f"Same fit, unflagged heroes only:    slope={null_slope:.4f}, r={null_r:.3f}, p={null_p:.2e}")
    print(f"\n{len(touched)} of {n_flagged} flagged heroes were touched by 7.41e patch notes; "
          f"{len(untouched)} were not (control group).")

    print("\n=== Flagged heroes, full detail ===")
    cols = ["name_mine", "action", "win_rate_mine", "win_rate_live", "delta_pp",
            "matches_prediction", "patch_touched"]
    print(flagged[cols].sort_values("action").to_string(index=False))

    out = {
        "live_snapshot_date": live_date,
        "n_heroes_compared": len(df),
        "n_flagged": n_flagged,
        "n_matches_prediction": n_match,
        "direction_test_pvalue": direction_test.pvalue,
        "n_beats_null": n_beats_null,
        "beats_null_pvalue": beats_null_test.pvalue,
        "reversion_all_heroes": {"slope": slope, "intercept": intercept, "r": r, "p": p},
        "reversion_unflagged_only": {"slope": null_slope, "intercept": null_intercept, "r": null_r, "p": null_p},
        "n_touched_by_741e": int(len(touched)),
        "n_untouched_by_741e": int(len(untouched)),
        "patch_changes": PATCH_741E_CHANGES,
        "heroes": df[[
            "name_mine", "action", "win_rate_mine", "win_rate_live", "delta_pp",
            "predicted_dir", "matches_prediction", "patch_touched",
        ]].rename(columns={"name_mine": "name"}).to_dict(orient="records"),
    }
    (ARTIFACTS / "patch_followup.json").write_text(json.dumps(out, indent=2, default=float))
    df.to_csv(ARTIFACTS / "patch_followup.csv", index=False)
    print(f"\nWrote {ARTIFACTS / 'patch_followup.json'} and patch_followup.csv")


if __name__ == "__main__":
    main()
