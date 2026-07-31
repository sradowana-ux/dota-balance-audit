# Dota 2 Draft Win Predictor

Predicting match outcome from **the ten hero picks alone** — no player skill, no
items, no in-game events — on public ranked matches from the OpenDota API.

[![Demo](https://img.shields.io/badge/LIVE_DEMO-c8aa6e?style=flat-square&labelColor=0d1117)](https://huggingface.co/spaces/Radowana/dota-draft-predictor)
[![Python](https://img.shields.io/badge/PYTHON-3.10+-c8aa6e?style=flat-square&labelColor=0d1117)](https://www.python.org)
[![License](https://img.shields.io/badge/LICENSE-MIT-c8aa6e?style=flat-square&labelColor=0d1117)](LICENSE)

This is a deliberately small question asked carefully. Draft-only prediction has
a low ceiling — most of what decides a Dota match happens after the draft — so
the interesting part is not the headline accuracy but whether the model beats
the baselines a sceptic would actually propose, and whether the improvement
survives a significance test.

---

## Results

<!-- RESULTS:START -->

Trained on **43,041** matches, evaluated on **10,761** strictly later matches (127 heroes, 53,802 total after filtering). Radiant won 53.3% of the held-out matches.

| Model | Accuracy | Log loss | ROC-AUC | Brier |
|---|---|---|---|---|
| Always predict Radiant *(baseline)* | 0.5326 | 0.6910 | — | 0.2490 |
| Solo hero win-rate sum *(baseline)* | 0.5591 | 0.6809 | 0.5816 | 0.2439 |
| **Logistic regression, signed hero features** | 0.5594 | 0.6805 | 0.5825 | 0.2437 |
| Logistic regression + synergy/counter pairs | 0.5649 | **0.6798** | 0.5855 | 0.2434 |
| Gradient boosting (HistGBM) | 0.5575 | 0.6836 | 0.5711 | 0.2452 |

**McNemar's test** on paired held-out predictions, logistic regression vs:

| Compared against | χ² | p | Verdict |
|---|---|---|---|
| Always predict Radiant (baseline) | 23.01 | 1.61e-06 | significant at α=0.05 |
| Solo hero win-rate sum (baseline) | 0.02 | 0.8913 | no significant difference |
| Logistic regression + synergy/counter pairs | 2.33 | 0.1265 | no significant difference |
| Gradient boosting (HistGBM) | 0.19 | 0.6633 | no significant difference |

<!-- RESULTS:END -->

Every number above is generated from `artifacts/report.json` by
`eval/render_results.py`, not typed by hand. `python eval/render_results.py --check`
fails if the README and the artifacts disagree.

<p align="center">
  <img src="assets/calibration.png" alt="Calibration curve" width="46%"/>
  <img src="assets/hero_effects.png" alt="Strongest hero effects" width="46%"/>
</p>

---

## How to read these numbers

**The signal is real but small, and the honest framing matters more than the
decimal places.** A draft-only model that reports ~56% accuracy is not a weak
attempt at a 90% problem; it is close to the information ceiling of the inputs.
Ten hero IDs cannot tell you who played better.

Several things are worth pointing out, including the ones that do not flatter the
model:

**The trivial baseline is not the one to beat.** "Always predict Radiant" scores
above 50% purely because Radiant has a real map advantage. Any model that only
clears *that* bar has learned which side of the map it is on and nothing else.
The second baseline — add up each hero's solo win rate, pick the higher side —
is what a competent analyst would do without machine learning, and it is a much
harder bar.

**Accuracy and log loss disagree, and log loss is the one to trust here.** The
solo win-rate baseline is competitive on raw accuracy, because near the decision
boundary the two methods flip the same coins. The logistic model wins on log
loss and Brier score, which is to say it is better at knowing *how* confident to
be. For a win-probability model that is the property that matters; accuracy
throws away everything except which side of 0.5 you landed on.

**Pairwise synergy and counter features help slightly, but not provably.**
The standard claim about draft models is that hero interactions carry much of
the signal. Adding ~16,000 synergy and counter columns does edge ahead of the
127-column linear model on every metric — but McNemar's test cannot distinguish
them (p ≈ 0.13), so the honest reading is "probably a small real gain, not
demonstrated at this sample size". Each hero pair appears in only a few dozen
matches out of 54,000, so most of those 16,000 weights are still fit largely
from noise.

**The shipped model is therefore the 127-feature linear one**, not the
best-scoring one. When two models are statistically indistinguishable, the
tie-breaker is the one whose predictions decompose exactly into per-hero terms,
because that is what the demo shows the user. Picking the marginally better
number over the explainable model would be optimising the leaderboard rather
than the product.

**Face validity.** The learned weights are not just numerically better than
chance, they line up with how the game is understood: the strongest negative
weights land on high-skill-floor heroes (Storm Spirit, Tinker, Monkey King,
Nature's Prophet) and the strongest positive ones on forgiving, low-execution
heroes (Wraith King, Spectre, Abaddon, Dazzle). That pattern is well known to
Dota players and was nowhere in the training signal — the model only ever saw
hero IDs and win/loss. It is a sanity check, not evidence of accuracy, but a
model that got this backwards would be worth distrusting.

---

## Method

### Signed hero encoding

A draft is two sets of five heroes. The obvious encoding gives each hero two
columns, one per side, for `2 × n_heroes` features. That lets the model learn
two unrelated weights for the same hero, which is wrong — Dota is close to
side-symmetric, so a hero worth `+w` to Radiant should be worth `−w` to Dire.

This project uses one column per hero instead:

```
x[h] = +1  if hero h is on Radiant
       -1  if hero h is on Dire
        0  otherwise
```

That halves the parameter count and makes the antisymmetry exact rather than
approximate: mirroring a draft flips the log-odds about the intercept, so
`P(radiant | draft) = 1 − P(radiant | mirrored draft)` holds by construction.
Radiant's map advantage is then absorbed cleanly into the intercept, which is
where a side effect belongs — it is a property of the map, not of any hero.

Because the model is linear in these features, the prediction decomposes
exactly:

```
logit(p) = intercept + Σ w_h over Radiant heroes − Σ w_h over Dire heroes
```

The demo shows that decomposition per hero. Nothing is approximated for the sake
of interpretability, because nothing needs to be.

### Evaluation design

* **Chronological split.** Test matches are strictly newer than every training
  match. A random split would let the model see matches from either side of any
  patch or meta shift inside the window.
* **Two baselines, one trivial and one not.** Described above.
* **Calibration, not just accuracy.** Reliability curve, Brier score, and log
  loss, because the demo hands users a probability and that probability should
  mean something.
* **McNemar's test** on the paired held-out predictions, which is the right test
  for comparing two classifiers on the same test set — it conditions on the
  cases where the models disagree instead of treating the two accuracy figures
  as independent samples.
* **Hyperparameters chosen by `TimeSeriesSplit` CV on the training set only.**
  The test set is touched once, at the end.

### Data

Public matches from the OpenDota API, filtered to:

| Filter | Reason |
|---|---|
| `game_mode == 22` (All Draft) | Turbo has roughly half the duration and a different economy, so hero win rates there come from a different distribution |
| Ten distinct non-zero hero IDs | OpenDota returns zero-filled hero arrays for matches its parser hasn't reached |
| `duration >= 900s` | Excludes early abandons, where the outcome reflects a disconnect rather than the draft |
| `avg_rank_tier` present | Guarantees every row can be bucketed by skill |

The exact dataset used for the reported numbers is committed to `data/matches.csv.gz`,
so the results reproduce without re-hitting the API.

---

## Reproduce

```bash
git clone https://github.com/sradowana-ux/dota-draft-predictor
cd dota-draft-predictor
pip install -r requirements-dev.txt

python src/train.py                  # trains, evaluates, writes artifacts/
python eval/make_plots.py            # calibration + hero-effect figures
python eval/render_results.py        # regenerates the README table
```

To rebuild the dataset from scratch (~20 minutes; the public API allows 60
calls/minute and roughly 100 matches arrive per call):

```bash
python src/collect.py --target 60000 --out data/matches.csv
```

Run the demo locally:

```bash
pip install -r requirements.txt
python app.py
```

---

## Project structure

```
src/collect.py         OpenDota collection with the quality filters above
src/features.py        signed hero encoding + synergy/counter pair encoding
src/train.py           baselines, models, McNemar's test, artifacts/report.json
eval/make_plots.py     calibration curve, strongest hero effects
eval/render_results.py generates the README results block from report.json
app.py                 Gradio demo with per-hero log-odds breakdown
data/matches.csv.gz       the exact dataset behind the reported numbers
data/heroes.csv        hero ID → name, attribute, roles
```

---

## Limitations

Stated plainly, because the model is easy to over-sell:

1. **The ceiling is low and this model is near it.** Draft explains a small
   fraction of match outcome. Anyone reporting draft-only accuracy far above
   these figures on public matches is either using post-draft information,
   evaluating on a random rather than chronological split, or leaking player
   identity.

2. **No player skill.** Two identical drafts played by a Herald stack and an
   Immortal stack get identical predictions. `avg_rank_tier` is collected but
   deliberately not used as a feature — it says nothing about which *side* wins,
   and including it would inflate apparent performance without improving the
   thing the demo actually does.

3. **The chronological split spans a narrow window.** The matches were collected
   in one pass over a short slice of time, so while the split is genuinely
   forward-looking, it does not test robustness across a patch boundary. Hero
   balance changes would degrade these weights, and this evaluation cannot say
   how fast.

4. **Public matchmaking, not professional drafts.** In All Draft there is no
   ban phase and no coordinated counter-picking, so the drafts are far less
   structured than in captains mode. Weights learned here should not be read as
   claims about competitive Dota.

5. **Hero interactions are not in the shipped model.** The pairwise experiment
   above scored slightly better but could not be distinguished from the linear
   model, so shipping it would have meant trading exact interpretability for an
   unproven gain. More data — on the order of hundreds of thousands of matches —
   is the prerequisite for revisiting that, not a better optimiser.

6. **Single-window sampling.** Matches come from a contiguous ID range rather
   than a stratified sample across regions and times of day, so the skill and
   region mix reflects whoever was playing during that window.

---

## Licence

MIT — see [LICENSE](LICENSE).

Match data from the [OpenDota API](https://docs.opendota.com/), used under its
terms. This project is not affiliated with Valve Corporation.
