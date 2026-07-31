"""Dota 2 balance audit — interactive dashboard.

Three tabs:
  1. Hero balance   every hero with its confidence interval and verdict
  2. Patch planner  how many matches to confirm a change of a given size
  3. Draft explorer the win-probability model, used as the instrument

Runs entirely off the committed artifacts, so the app needs no database and no
network at startup.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
REPO_URL = "https://github.com/sradowana-ux/dota-balance-audit"

balance = json.loads((ARTIFACTS / "balance.json").read_text())
power_report = json.loads((ARTIFACTS / "power.json").read_text())
determinism = json.loads((ARTIFACTS / "determinism.json").read_text())

heroes_stats = pd.read_csv(ARTIFACTS / "hero_balance.csv")
heroes_power = pd.read_csv(ARTIFACTS / "hero_power.csv")
hero_meta = pd.read_csv(ROOT / "data" / "heroes.csv")

bundle = joblib.load(ARTIFACTS / "model.joblib")
MODEL = bundle["model"]
HERO_INDEX = bundle["hero_index"]
COEFS = MODEL.coef_[0]
INTERCEPT = float(MODEL.intercept_[0])

ID_TO_NAME = dict(zip(hero_meta["hero_id"], hero_meta["name"]))
NAME_TO_ID = {v: k for k, v in ID_TO_NAME.items()}
HERO_NAMES = sorted(NAME_TO_ID)

TOL = balance["tolerance_pp"] / 100


# --------------------------------------------------------------------------
# Tab 1: hero balance
# --------------------------------------------------------------------------

def balance_table(verdict: str, search: str):
    df = heroes_stats.copy()
    if verdict == "Outside tolerance only":
        df = df[df["outside_tolerance"]]
    elif verdict == "Overperforming":
        df = df[df["action"] == "nerf candidate"]
    elif verdict == "Underperforming":
        df = df[df["action"] == "buff candidate"]
    if search:
        df = df[df["name"].str.contains(search, case=False, na=False)]

    out = pd.DataFrame({
        "Hero": df["name"],
        "Games": df["games"].map("{:,}".format),
        "Win rate": (df["win_rate"] * 100).map("{:.2f}%".format),
        "95% CI": [f"[{lo*100:.2f}%, {hi*100:.2f}%]" for lo, hi in zip(df["ci_low"], df["ci_high"])],
        "q-value (BH)": df["q_value"].map(lambda v: f"{v:.2e}" if v < 0.001 else f"{v:.4f}"),
        "Verdict": df["action"],
    })
    return out.sort_values("Hero").reset_index(drop=True)


# --------------------------------------------------------------------------
# Tab 2: patch planner
# --------------------------------------------------------------------------

def plan_patch(hero: str, effect_pp: float, power: float, alpha: float):
    if not hero:
        return "Pick a hero to size the experiment."

    row = heroes_power[heroes_power["name"] == hero]
    if row.empty:
        return f"No data for {hero} (fewer than {balance['min_games_threshold']} games)."
    row = row.iloc[0]

    p1, p2 = 0.50, 0.50 + effect_pp / 100
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    n = ((z_a * np.sqrt(2 * p_bar * (1 - p_bar))
          + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2

    hero_games = int(np.ceil(n))
    matches = int(np.ceil(hero_games / row["pick_rate"]))
    current_mde = (z_a + z_b) * np.sqrt(2 * 0.25 / row["games"]) * 100
    already = row["games"] >= hero_games

    verdict = (
        f"The committed dataset already has {int(row['games']):,} games of "
        f"{hero} — enough to detect this."
        if already else
        f"The committed dataset has {int(row['games']):,} games of {hero}, "
        f"which is {hero_games - int(row['games']):,} short."
    )

    return (
        f"## {hero}\n\n"
        f"Currently **{row['win_rate']*100:.2f}%** win rate over "
        f"**{int(row['games']):,}** games, picked in **{row['pick_rate']*100:.1f}%** "
        f"of matches.\n\n"
        f"To detect a **{effect_pp:.1f}pp** change at **{power:.0%}** power "
        f"(α={alpha}):\n\n"
        f"| | |\n|---|---|\n"
        f"| Hero-games needed, each period | **{hero_games:,}** |\n"
        f"| Matches the population must play | **{matches:,}** |\n"
        f"| Smallest change visible at current sample | {current_mde:.2f}pp |\n\n"
        f"{verdict}\n\n"
        f"*Both periods need that many games — before the patch and after — so "
        f"the total cost is twice the match figure.*"
    )


# --------------------------------------------------------------------------
# Tab 3: draft explorer
# --------------------------------------------------------------------------

def predict(*picks):
    radiant_names, dire_names = list(picks[:5]), list(picks[5:])
    chosen = [n for n in radiant_names + dire_names if n]

    if len(chosen) < 10:
        return "### Pick all ten heroes.", None
    if len(set(chosen)) != 10:
        dupes = sorted({n for n in chosen if chosen.count(n) > 1})
        return f"### Duplicate pick: {', '.join(dupes)}", None

    x = np.zeros((1, len(HERO_INDEX)), dtype=np.float32)
    rows = []
    for name in radiant_names:
        col = HERO_INDEX.get(int(NAME_TO_ID[name]))
        if col is not None:
            x[0, col] = 1.0
            rows.append(("Radiant", name, float(COEFS[col])))
    for name in dire_names:
        col = HERO_INDEX.get(int(NAME_TO_ID[name]))
        if col is not None:
            x[0, col] = -1.0
            rows.append(("Dire", name, -float(COEFS[col])))

    proba = float(MODEL.predict_proba(x)[0, 1])
    contrib = pd.DataFrame(rows, columns=["Side", "Hero", "Log-odds"])
    contrib["Effect"] = contrib["Log-odds"].map(
        lambda v: f"{v:+.3f} ({'Radiant' if v >= 0 else 'Dire'})"
    )
    contrib = (
        contrib.reindex(contrib["Log-odds"].abs().sort_values(ascending=False).index)
        .loc[:, ["Side", "Hero", "Effect"]].reset_index(drop=True)
    )

    edge = abs(proba - 0.5) * 100
    med = determinism["median_edge_pp"]
    comparison = "more lopsided than" if edge > med else "less lopsided than"

    return (
        f"## Radiant {proba:.1%} · Dire {1 - proba:.1%}\n\n"
        f"**{edge:.1f} points** off even — {comparison} the median real draft "
        f"({med:.1f}pp).\n\n"
        f"Side-advantage intercept `{INTERCEPT:+.3f}`; everything else below is heroes.",
        contrib,
    )


# --------------------------------------------------------------------------

roster = balance["roster"]
tests = balance["hero_tests"]
side = balance["side_balance"]

with gr.Blocks(title="Is Dota 2 Balanced?") as demo:
    gr.Markdown(
        f"""
# Is Dota 2 Balanced?

A statistical audit of **{balance['n_matches']:,}** ranked All Draft matches.
Hero win rates span **{roster['min_win_rate']:.1%}–{roster['max_win_rate']:.1%}**;
**{tests['significant_after_bh']}** of {roster['heroes_tested']} heroes are
statistically distinguishable from 50%, but only **{tests['outside_tolerance']}**
are outside a ±{balance['tolerance_pp']:.0f}pp practical tolerance — which is the
number that should drive decisions. Radiant's structural advantage
(**{side['radiant_win_rate']:.2%}**) is larger than all but three hero effects.

[Full method and recommendations]({REPO_URL})
"""
    )

    with gr.Tab("Hero balance"):
        with gr.Row():
            verdict = gr.Radio(
                ["All heroes", "Outside tolerance only", "Overperforming", "Underperforming"],
                value="Outside tolerance only", label="Filter",
            )
            search = gr.Textbox(label="Search hero", placeholder="e.g. Pudge")
        table = gr.Dataframe(wrap=True)
        for control in (verdict, search):
            control.change(balance_table, [verdict, search], table)
        demo.load(balance_table, [verdict, search], table)
        gr.Markdown(
            f"*q-values are Benjamini–Hochberg adjusted across "
            f"{roster['heroes_tested']} heroes. Verdict requires the entire "
            f"confidence interval to clear ±{balance['tolerance_pp']:.0f}pp, so "
            "statistical and practical significance stay separate.*"
        )

    with gr.Tab("Patch planner"):
        gr.Markdown(
            "**If you changed this hero, how long until you could tell it worked?** "
            "A two-proportion power calculation using the hero's real pick rate."
        )
        with gr.Row():
            hero_in = gr.Dropdown(sorted(heroes_power["name"]), label="Hero", filterable=True)
            effect_in = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="Change to detect (pp)")
        with gr.Row():
            power_in = gr.Slider(0.5, 0.95, value=0.8, step=0.05, label="Power")
            alpha_in = gr.Slider(0.01, 0.10, value=0.05, step=0.01, label="α")
        plan_out = gr.Markdown()
        for control in (hero_in, effect_in, power_in, alpha_in):
            control.change(plan_patch, [hero_in, effect_in, power_in, alpha_in], plan_out)

    with gr.Tab("Draft explorer"):
        gr.Markdown(
            f"The win-probability model used as a measuring instrument. Across "
            f"{determinism['n_test_matches']:,} held-out drafts it reaches "
            f"**{determinism['held_out_accuracy']:.1%}** accuracy — about a point "
            f"below the **{determinism['ceiling_accuracy']:.1%}** ceiling for this "
            "model class. That gap between the draft and the outcome is the point: "
            "it is how little the draft settles."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Radiant")
                radiant = [gr.Dropdown(HERO_NAMES, label=f"Radiant {i+1}", filterable=True)
                           for i in range(5)]
            with gr.Column():
                gr.Markdown("### Dire")
                dire = [gr.Dropdown(HERO_NAMES, label=f"Dire {i+1}", filterable=True)
                        for i in range(5)]
        go = gr.Button("Predict", variant="primary")
        summary = gr.Markdown()
        contrib_table = gr.Dataframe(headers=["Side", "Hero", "Effect"], wrap=True)
        go.click(predict, radiant + dire, [summary, contrib_table])

if __name__ == "__main__":
    demo.launch()
