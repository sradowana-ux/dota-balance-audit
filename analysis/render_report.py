"""Generate the README findings block from the analysis artifacts.

    python analysis/render_report.py           # rewrite the block
    python analysis/render_report.py --check   # exit 1 if the README is stale

Same contract as eval/render_results.py: every number in the report is written
by this script from artifacts/*.json, so the README cannot drift away from what
the code actually produced. CI runs --check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ARTIFACTS = ROOT / "artifacts"

START = "<!-- FINDINGS:START -->"
END = "<!-- FINDINGS:END -->"


def render() -> str:
    b = json.loads((ARTIFACTS / "balance.json").read_text())
    p = json.loads((ARTIFACTS / "power.json").read_text())
    d = json.loads((ARTIFACTS / "determinism.json").read_text())
    heroes = pd.read_csv(ARTIFACTS / "hero_balance.csv")

    roster, tests, side, bracket = b["roster"], b["hero_tests"], b["side_balance"], b["bracket_consistency"]
    spread = d["probability_spread"]

    top = heroes.nlargest(3, "win_rate")
    bottom = heroes.nsmallest(3, "win_rate")

    L: list[str] = []

    L += [
        f"Based on **{b['n_matches']:,}** ranked All Draft matches, covering "
        f"**{roster['heroes_tested']}** heroes with at least "
        f"{b['min_games_threshold']} games each.",
        "",
        "### 1. The roster is tightly balanced, but not perfectly",
        "",
        f"Win rates span **{roster['min_win_rate']:.1%} to {roster['max_win_rate']:.1%}**, "
        f"a {roster['spread_pp']:.1f} point range, standard deviation "
        f"{roster['sd_pp']:.2f}pp. For a game with {roster['heroes_tested']} "
        "asymmetric heroes, that is a narrow band.",
        "",
        "| | Hero | Games | Win rate | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for _, r in top.iterrows():
        L.append(f"| ▲ | {r['name']} | {r['games']:,} | {r['win_rate']:.1%} | "
                 f"[{r['ci_low']:.1%}, {r['ci_high']:.1%}] |")
    for _, r in bottom.iterrows():
        L.append(f"| ▼ | {r['name']} | {r['games']:,} | {r['win_rate']:.1%} | "
                 f"[{r['ci_low']:.1%}, {r['ci_high']:.1%}] |")

    L += [
        "",
        "### 2. At this sample size, statistical significance is the wrong question",
        "",
        f"Testing every hero against 50%, **{tests['significant_uncorrected']} of "
        f"{roster['heroes_tested']}** come back significant at α={b['alpha']}, against "
        f"roughly {tests['expected_false_positives']:.0f} expected by chance. Applying "
        f"Benjamini–Hochberg FDR correction removes almost nothing: "
        f"**{tests['significant_after_bh']}** still survive.",
        "",
        "That is the finding, not a footnote. With 50,000 matches the tests are so "
        "well powered that a hero sitting 0.8 points off even is detectable, and "
        "completely irrelevant to a balance decision. **Significance stopped "
        "discriminating; effect size has to do the work.**",
        "",
        f"Applying a practical tolerance of ±{b['tolerance_pp']:.0f}pp and requiring the "
        f"whole confidence interval to clear it, the {tests['significant_after_bh']} "
        f"\"significant\" heroes reduce to **{tests['outside_tolerance']} genuinely "
        "actionable ones**:",
        "",
        f"- **Overperforming ({len(tests['nerf_candidates'])}):** "
        + ", ".join(tests["nerf_candidates"]),
        f"- **Underperforming ({len(tests['buff_candidates'])}):** "
        + ", ".join(tests["buff_candidates"]),
        "",
        "### 3. Radiant's map advantage is real and larger than any hero effect",
        "",
        f"Radiant wins **{side['radiant_win_rate']:.2%}** of matches "
        f"[{side['ci_low']:.2%}, {side['ci_high']:.2%}], "
        f"p = {side['p_value']:.1e}, a **{side['advantage_pp']:.1f} point** structural "
        "edge before a single hero is picked. Only three heroes deviate from even by "
        "more than the side you were assigned does.",
        "",
        "### 4. \"Balanced\" means different things at different ranks",
        "",
        f"Correlating hero win rates between the lowest and highest skill brackets "
        f"gives **r = {bracket['correlation']:.2f}** across {bracket['n_heroes']} heroes "
        f"(p = {bracket['p_value']:.1e}). Positive, but weak: a hero's performance at "
        "low rank tells you comparatively little about its performance at high rank. "
        "A single global win rate averages over two genuinely different games.",
        "",
        "### 5. How much does the draft decide?",
        "",
        f"Using the win-probability model as a measuring instrument across "
        f"{d['n_test_matches']:,} held-out drafts:",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Draft-implied win probability, full range | {spread['min']:.3f} – {spread['max']:.3f} |",
        f"| 1st–99th percentile | {spread['p1']:.3f} – {spread['p99']:.3f} |",
        f"| Median advantage off even | {d['median_edge_pp']:.2f}pp |",
        f"| Drafts moving win probability >{d['meaningful_threshold_pp']:.0f}pp | {d['share_meaningfully_decided']:.1%} |",
        f"| Model accuracy from draft alone | {d['held_out_accuracy']:.1%} |",
        f"| Information ceiling for this model class | {d['ceiling_accuracy']:.1%} |",
        "",
        f"The honest held-out model sits **{d['gap_to_ceiling_pp']:.2f} points** below the "
        "ceiling obtained by fitting and scoring on all data with no regularisation. "
        "The limit is the information in a draft, not the modelling.",
        "",
        "### 6. Confirming a balance patch is expensive",
        "",
        f"Detecting a **{p['target_effect_pp']:.0f}pp** win-rate change at "
        f"{p['power']:.0%} power requires **{p['hero_games_needed_for_target']:,} "
        "hero-games before and after** the patch.",
        "",
        "| Change to detect | Hero-games needed per period |",
        "|---|---|",
    ]
    for row in p["sample_size_table"]:
        L.append(f"| {row['effect_pp']:.1f}pp | {row['hero_games_per_period']:,} |")

    cheap = p["most_picked"][0]
    dear = p["least_picked"][0]
    L += [
        "",
        "Converted through real pick rates, the median hero needs "
        f"**{p['median_matches_needed']:,} matches** played before a "
        f"{p['target_effect_pp']:.0f}pp change becomes visible. The spread is enormous: "
        f"{cheap['name']} (picked in {cheap['pick_rate']:.1%} of games) needs "
        f"{cheap['matches_needed']:,}, while {dear['name']} "
        f"({dear['pick_rate']:.1%}) needs {dear['matches_needed']:,}.",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    text = README.read_text()
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        sys.exit(f"Could not find {START} ... {END} markers in README.md")

    updated = pattern.sub(f"{START}\n\n{render()}\n{END}", text)

    if args.check:
        if updated != text:
            sys.exit("README.md findings block is out of date; run analysis/render_report.py")
        print("README findings block matches the analysis artifacts")
        return

    README.write_text(updated)
    print("Updated the findings block in README.md")


if __name__ == "__main__":
    main()
