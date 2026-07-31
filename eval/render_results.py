"""Write the results table in README.md straight from artifacts/report.json.

    python eval/render_results.py            # rewrite the README block
    python eval/render_results.py --check    # exit 1 if the README is stale

The README's numbers are generated, never typed. `--check` runs in CI so a
README that disagrees with the artifacts fails the build instead of quietly
overstating the model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "docs" / "MODEL.md"
REPORT = ROOT / "artifacts" / "report.json"

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

LABELS = {
    "majority_class": "Always predict Radiant *(baseline)*",
    "hero_winrate_sum": "Solo hero win-rate sum *(baseline)*",
    "logistic_regression": "**Logistic regression, signed hero features**",
    "logreg_pairwise": "Logistic regression + synergy/counter pairs",
    "gradient_boosting": "Gradient boosting (HistGBM)",
}

ORDER = [
    "majority_class",
    "hero_winrate_sum",
    "logistic_regression",
    "logreg_pairwise",
    "gradient_boosting",
]


def render(report: dict) -> str:
    results = report["results"]
    lines = [
        f"Trained on **{report['n_train']:,}** matches, evaluated on "
        f"**{report['n_test']:,}** strictly later matches "
        f"({report['n_heroes']} heroes, {report['n_matches']:,} total after filtering). "
        f"Radiant won {report['radiant_win_rate_test']:.1%} of the held-out matches.",
        "",
        "| Model | Accuracy | Log loss | ROC-AUC | Brier |",
        "|---|---|---|---|---|",
    ]

    best_ll = min(results[k]["log_loss"] for k in ORDER if k in results)
    for key in ORDER:
        if key not in results:
            continue
        r = results[key]
        auc = "—" if key == "majority_class" else f"{r['roc_auc']:.4f}"
        ll = f"{r['log_loss']:.4f}"
        if abs(r["log_loss"] - best_ll) < 1e-12:
            ll = f"**{ll}**"
        lines.append(
            f"| {LABELS.get(key, key)} | {r['accuracy']:.4f} | {ll} | {auc} | {r['brier']:.4f} |"
        )

    lines += ["", "**McNemar's test** on paired held-out predictions, logistic regression vs:", ""]
    lines += ["| Compared against | χ² | p | Verdict |", "|---|---|---|---|"]
    for key, sig in report["significance"].items():
        other = key.replace("logreg_vs_", "")
        p = sig["p_value"]
        verdict = (
            "significant at α=0.05" if p < 0.05 else "no significant difference"
        )
        p_str = f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}"
        lines.append(
            f"| {LABELS.get(other, other).replace('**', '').replace('*', '')} "
            f"| {sig['statistic']:.2f} | {p_str} | {verdict} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    report = json.loads(REPORT.read_text())
    block = render(report)
    text = README.read_text()

    pattern = re.compile(
        re.escape(START) + r".*?" + re.escape(END), re.DOTALL
    )
    if not pattern.search(text):
        sys.exit(f"Could not find {START} ... {END} markers in README.md")

    updated = pattern.sub(f"{START}\n\n{block}\n\n{END}", text)

    if args.check:
        if updated != text:
            sys.exit("README.md results block is out of date; run eval/render_results.py")
        print("README results block matches artifacts/report.json")
        return

    README.write_text(updated)
    print("Updated the results block in README.md")


if __name__ == "__main__":
    main()
