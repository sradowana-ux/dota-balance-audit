"""Gradio demo: draft-only win probability for Dota 2.

Pick ten heroes, get a calibrated Radiant win probability plus the per-hero
breakdown of where that number came from.

The breakdown is exact, not an approximation. The model is a logistic
regression on signed hero indicators, so the log-odds decompose additively:

    logit(p) = intercept + sum over Radiant heroes of w_h
                         - sum over Dire heroes of w_h

Each row in the contributions table is one of those terms, in log-odds units.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

bundle = joblib.load(ARTIFACTS / "model.joblib")
MODEL = bundle["model"]
HERO_INDEX = bundle["hero_index"]

heroes_df = pd.read_csv(ROOT / "data" / "heroes.csv")
ID_TO_NAME = dict(zip(heroes_df["hero_id"], heroes_df["name"]))
NAME_TO_ID = {v: k for k, v in ID_TO_NAME.items()}
HERO_NAMES = sorted(NAME_TO_ID)

REPORT = json.loads((ARTIFACTS / "report.json").read_text())
TEST_ACC = REPORT["results"]["logistic_regression"]["accuracy"]
TEST_AUC = REPORT["results"]["logistic_regression"]["roc_auc"]
N_TEST = REPORT["n_test"]
N_TOTAL = REPORT["n_matches"]

COEFS = MODEL.coef_[0]
INTERCEPT = float(MODEL.intercept_[0])


def predict(*picks):
    radiant_names = list(picks[:5])
    dire_names = list(picks[5:])
    chosen = [n for n in radiant_names + dire_names if n]

    if len(chosen) < 10:
        return "### Pick all ten heroes to get a prediction.", None
    if len(set(chosen)) != 10:
        duplicates = sorted({n for n in chosen if chosen.count(n) > 1})
        return (
            f"### Duplicate pick: {', '.join(duplicates)}\n"
            "A Dota draft cannot contain the same hero twice.",
            None,
        )

    radiant_ids = [NAME_TO_ID[n] for n in radiant_names]
    dire_ids = [NAME_TO_ID[n] for n in dire_names]

    x = np.zeros((1, len(HERO_INDEX)), dtype=np.float32)
    rows = []
    for hero_id, name in zip(radiant_ids, radiant_names):
        col = HERO_INDEX.get(int(hero_id))
        if col is None:
            continue
        x[0, col] = 1.0
        rows.append(("Radiant", name, float(COEFS[col])))
    for hero_id, name in zip(dire_ids, dire_names):
        col = HERO_INDEX.get(int(hero_id))
        if col is None:
            continue
        x[0, col] = -1.0
        rows.append(("Dire", name, -float(COEFS[col])))

    proba = float(MODEL.predict_proba(x)[0, 1])
    favourite = "Radiant" if proba >= 0.5 else "Dire"
    edge = max(proba, 1 - proba)

    contributions = pd.DataFrame(rows, columns=["Side", "Hero", "Log-odds"])
    contributions["Effect"] = contributions["Log-odds"].map(
        lambda v: f"{'+' if v >= 0 else ''}{v:.3f} ({'favours Radiant' if v >= 0 else 'favours Dire'})"
    )
    contributions = (
        contributions.reindex(
            contributions["Log-odds"].abs().sort_values(ascending=False).index
        )
        .loc[:, ["Side", "Hero", "Effect"]]
        .reset_index(drop=True)
    )

    summary = (
        f"## Radiant {proba:.1%} &nbsp;·&nbsp; Dire {1 - proba:.1%}\n\n"
        f"**{favourite} favoured**, {edge:.1%} win probability from the draft alone.\n\n"
        f"Side-advantage intercept: `{INTERCEPT:+.3f}` log-odds. "
        "Everything else in the table below is hero effects."
    )
    return summary, contributions


EXAMPLES = [
    # A greedy late-game Radiant lineup against an early-pressure Dire lineup.
    ["Anti-Mage", "Medusa", "Crystal Maiden", "Dazzle", "Enigma",
     "Huskar", "Bristleback", "Undying", "Ogre Magi", "Tusk"],
    # Mirror-ish draft: both sides fairly standard.
    ["Juggernaut", "Lion", "Tidehunter", "Zeus", "Mirana",
     "Phantom Assassin", "Shadow Shaman", "Axe", "Lina", "Vengeful Spirit"],
]
EXAMPLES = [row for row in EXAMPLES if all(n in NAME_TO_ID for n in row)]

REPO_URL = "https://github.com/sradowana-ux/dota-draft-predictor"

with gr.Blocks(title="Dota 2 Draft Win Predictor") as demo:
    gr.Markdown(
        f"""
# Dota 2 Draft Win Predictor

Win probability from the **ten hero picks alone** — no player skill, no items,
no in-game events. Trained on {N_TOTAL:,} ranked All Draft matches from the
OpenDota public API, evaluated on {N_TEST:,} strictly later matches
(**{TEST_ACC:.1%}** accuracy, **{TEST_AUC:.3f}** ROC-AUC).

A draft is a weak signal by construction, so treat these numbers as a small
edge over a coin flip rather than a prediction of the outcome. The
[README]({REPO_URL}) explains exactly how much of an edge, and against which
baselines.
"""
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Radiant")
            radiant = [
                gr.Dropdown(HERO_NAMES, label=f"Radiant {i + 1}", filterable=True)
                for i in range(5)
            ]
        with gr.Column():
            gr.Markdown("### Dire")
            dire = [
                gr.Dropdown(HERO_NAMES, label=f"Dire {i + 1}", filterable=True)
                for i in range(5)
            ]

    button = gr.Button("Predict", variant="primary")
    summary = gr.Markdown()
    table = gr.Dataframe(
        headers=["Side", "Hero", "Effect"],
        label="Where the number comes from (log-odds contributions)",
        wrap=True,
    )

    button.click(predict, inputs=radiant + dire, outputs=[summary, table])

    if EXAMPLES:
        gr.Examples(examples=EXAMPLES, inputs=radiant + dire)

    gr.Markdown(
        """
---
**How to read this.** The model is a logistic regression on signed hero
indicators (`+1` Radiant, `-1` Dire), so the log-odds are exactly the sum of
the intercept and the ten hero terms shown above. Positive values push towards
a Radiant win.

**What it cannot do.** It has no notion of player skill, lane assignment, item
choices, or patch. Two identical drafts played by very different players get
identical predictions. Hero synergies and counters are not modelled — only
each hero's independent contribution.
"""
    )

if __name__ == "__main__":
    demo.launch()
