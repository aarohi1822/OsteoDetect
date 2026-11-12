"""Build the README results tables from reports/metrics_*.json, preds_*.csv and thresholds.json.

  python src/results_table.py           # print the tables
  python src/results_table.py --write   # replace the block between the markers in README.md
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
START, END = "<!-- results:start -->", "<!-- results:end -->"

MODELS = [  # (metrics file stem, display name)
    ("cnn", "Custom CNN (notebook architecture)"),
    ("resnet_frozen", "ResNet50, frozen (notebook recipe)"),
    ("resnet_ft", "ResNet50, fine-tuned, 12 epochs"),
    ("resnet_ft25", "ResNet50, fine-tuned, 25 epochs"),
]


def pct(x):
    return f"{100 * x:.1f}%"


def ci(lo_hi, scale=100, fmt="{:.1f}"):
    lo, hi = lo_hi
    return f"[{fmt.format(lo * scale)}–{fmt.format(hi * scale)}]"


def build() -> str:
    m = {k: json.loads((REPORTS / f"metrics_{k}.json").read_text()) for k, _ in MODELS}
    thr = json.loads((REPORTS / "thresholds.json").read_text())
    best = max(m, key=lambda k: m[k]["test"]["f1"])
    n, cm0 = m[best]["test"]["n"], m[best]["test"]["confusion_matrix"]
    n_tb, n_norm = cm0["tp"] + cm0["fn"], cm0["tn"] + cm0["fp"]
    out = []

    out.append(f"**Held-out test set** (n = {n}: {n_norm} Normal, {n_tb} TB), threshold 0.5, "
               "95% bootstrap CIs in brackets. TB is the positive class.\n")
    out.append("| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | FN / FP |")
    out.append("|---|---|---|---|---|---|---|---|")
    for k, name in MODELS:
        t, c = m[k]["test"], m[k]["test_ci95"]
        cm = t["confusion_matrix"]
        b = "**" if k == best else ""
        out.append(f"| {b}{name}{b} | {pct(t['accuracy'])} {ci(c['accuracy'])} | "
                   f"{pct(t['precision'])} {ci(c['precision'])} | {b}{pct(t['recall'])}{b} {ci(c['recall'])} | "
                   f"{pct(t['specificity'])} | {t['f1']:.3f} {ci(c['f1'], 1, '{:.3f}')} | "
                   f"{t['roc_auc']:.3f} {ci(c['roc_auc'], 1, '{:.3f}')} | {cm['fn']} / {cm['fp']} |")

    nlm = m[best]["test_nlm_only"]
    out.append(f"\n**Source-matched subset** (test images from NLM only, n = {nlm['n']}). NLM is the "
               "only source with both classes, so here the image source can't give away the label.\n")
    out.append("| Model | Accuracy | Recall | Specificity | F1 | ROC-AUC |")
    out.append("|---|---|---|---|---|---|")
    for k, name in MODELS:
        t = m[k]["test_nlm_only"]
        out.append(f"| {name} | {pct(t['accuracy'])} | {pct(t['recall'])} | "
                   f"{pct(t['specificity'])} | {t['f1']:.3f} | {t['roc_auc']:.3f} |")

    groups = [("RSNA", 0, "RSNA Normal"), ("NLM", 0, "NLM Normal"),
              ("Belarus", 1, "Belarus TB"), ("NLM", 1, "NLM TB")]
    out.append("\n**Correct predictions by source and class** (test, threshold 0.5). Almost every "
               "error is a missed TB case. RSNA Normals, the bulk of the Normal class, are "
               "nearly always right.\n")
    out.append("| Model | " + " | ".join(g[2] for g in groups) + " |")
    out.append("|---|" + "---|" * len(groups))
    for k, name in MODELS:
        p = pd.read_csv(REPORTS / f"preds_{k}.csv")
        ok = (p.p_tb > 0.5).astype(int) == p.label
        cells = []
        for src, y, _ in groups:
            sel = (p.source == src) & (p.label == y)
            cells.append(f"{int(ok[sel].sum())}/{int(sel.sum())}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")

    out.append("\n**Threshold chosen on val, applied to test** (`sens95` = highest threshold with "
               "≥ 95% recall on val; `youden` = max sensitivity + specificity − 1 on val).\n")
    out.append("| Model | Rule | Threshold | Test recall | Test specificity | Test F1 | FN / FP |")
    out.append("|---|---|---|---|---|---|---|")
    for k, name in MODELS:
        for rule in ("youden", "sens95"):
            r = thr[k][rule]
            t, cm = r["test"], r["test"]["confusion_matrix"]
            out.append(f"| {name} | {rule} | {r['threshold']:.3f} | {pct(t['recall'])} | "
                       f"{pct(t['specificity'])} | {t['f1']:.3f} | {cm['fn']} / {cm['fp']} |")
    return "\n".join(out)


def main():
    table = build()
    if "--write" not in sys.argv:
        print(table)
        return
    readme = ROOT / "README.md"
    text = readme.read_text()
    a, b = text.index(START) + len(START), text.index(END)
    readme.write_text(text[:a] + "\n" + table + "\n" + text[b:])
    print("README results block updated")


if __name__ == "__main__":
    main()
