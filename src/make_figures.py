"""Render assets/before_after.png and assets/source_heatmap.png from saved outputs only.

Sources, all read at run time:
  * reports/metrics_<model>.json -> held-out test recall ("test" -> "recall")
  * reports/preds_<model>.csv    -> per-source correct-prediction counts (threshold 0.5)
  * BONE_TB.ipynb at commit 521f48c, cell 13 output -> the notebook's "Validation Accuracy: 95.69%"
    (that output was cleared from the current file in commit 398b052; git history is its only copy)

EXPECTED holds the numbers quoted in the request. The script exits non-zero if any saved
value, rounded the same way, differs from them.

  python src/make_figures.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS, ASSETS = ROOT / "reports", ROOT / "assets"

MODELS = [("cnn", "Custom CNN"), ("resnet_frozen", "ResNet50\nfrozen"),
          ("resnet_ft", "ResNet50\nfine-tuned (12 ep)"), ("resnet_ft25", "ResNet50\nfine-tuned (25 ep)")]
GROUPS = [("RSNA", 0, "RSNA\nNormal"), ("NLM", 0, "NLM\nNormal"),
          ("Belarus", 1, "Belarus\nTB"), ("NLM", 1, "NLM\nTB")]
EXPECTED = {"notebook_val_acc": 95.69,  # percent, 2 dp as printed
            "recall": {"cnn": 69.3, "resnet_frozen": 84.2, "resnet_ft": 93.1, "resnet_ft25": 96.0}}

# Reference palette (dataviz skill, light mode)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a8a7a1", "#e4e3df"
BLUE = "#2a78d6"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def fail(msg):
    sys.exit(f"MISMATCH: {msg}")


def notebook_val_acc() -> float:
    nb = json.loads(subprocess.check_output(["git", "-C", str(ROOT), "show", "521f48c:BONE_TB.ipynb"]))
    text = "".join("".join(o.get("text", [])) for o in nb["cells"][13].get("outputs", []))
    m = re.search(r"Validation Accuracy: ([\d.]+)%", text)
    if not m:
        fail("could not find 'Validation Accuracy' in BONE_TB.ipynb@521f48c cell 13")
    return float(m.group(1))


def load():
    acc = notebook_val_acc()
    if acc != EXPECTED["notebook_val_acc"]:
        fail(f"notebook accuracy {acc} != expected {EXPECTED['notebook_val_acc']}")
    recall = {}
    for k, _ in MODELS:
        r = 100 * json.loads((REPORTS / f"metrics_{k}.json").read_text())["test"]["recall"]
        if round(r, 1) != EXPECTED["recall"][k]:
            fail(f"{k} recall {r:.4f}% rounds to {round(r, 1)}, expected {EXPECTED['recall'][k]}")
        recall[k] = r
    counts = {}
    for k, _ in MODELS:
        p = pd.read_csv(REPORTS / f"preds_{k}.csv")
        if set(p.split) != {"test"}:
            fail(f"preds_{k}.csv contains non-test rows")
        ok = (p.p_tb > 0.5).astype(int) == p.label
        counts[k] = [(int(ok[(p.source == s) & (p.label == y)].sum()),
                      int(((p.source == s) & (p.label == y)).sum())) for s, y, _ in GROUPS]
        # cross-check: per-source correct counts must reproduce the metrics file's confusion matrix
        cm = json.loads((REPORTS / f"metrics_{k}.json").read_text())["test"]["confusion_matrix"]
        (rsna, _), (nlm_n, _), (bel, _), (nlm_t, _) = counts[k]
        if rsna + nlm_n != cm["tn"] or bel + nlm_t != cm["tp"]:
            fail(f"{k}: per-source counts disagree with metrics_{k}.json confusion matrix")
    return acc, recall, counts


def before_after(acc, recall):
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    labels = ["Original notebook\n“validation” accuracy"] + [n for _, n in MODELS]
    values = [acc] + [recall[k] for k, _ in MODELS]
    x = range(len(values))
    bars = ax.bar(x, values, width=0.62, color=[SURFACE] + [BLUE] * 4,
                  edgecolor=[MUTED] + [SURFACE] * 4, linewidth=[1.6] + [2] * 4,
                  hatch=["///", "", "", "", ""], zorder=3)
    bars[0].set_hatch_linewidth(1.0) if hasattr(bars[0], "set_hatch_linewidth") else None
    for i, (b, v) in enumerate(zip(bars, values)):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.2f}%" if i == 0 else f"{v:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK2 if i == 0 else INK)
    ax.text(0, 46, "INVALID\nmeasured on the\ntraining data\n(train = val folder)", ha="center",
            va="center", fontsize=10, fontweight="bold", color=INK2, zorder=4,
            bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=MUTED, lw=1))
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(0.62, 104, "Held-out test set, TB recall (n = 101 TB images)", fontsize=10.5,
            color=INK2, va="bottom")
    ax.set_xticks(list(x), labels, fontsize=10.5)
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20), [f"{t}%" for t in range(0, 101, 20)])
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_title("The notebook said 95.69%. On unseen data, the CNN catches 69% of TB cases.",
                 loc="left", fontsize=14, fontweight="bold", pad=26)
    fig.text(0.012, 0.015, "Recall from reports/metrics_*.json (threshold 0.5). Notebook value from "
             "BONE_TB.ipynb, first commit, cell 13. Different metrics: shown for contrast, "
             "not comparison.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ASSETS / "before_after.png")
    plt.close(fig)


def source_heatmap(counts):
    rates = [[c / n for c, n in counts[k]] for k, _ in MODELS]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    vmin, vmax = 0.6, 1.0
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=200)
    im = ax.imshow(rates, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i, (k, _) in enumerate(MODELS):
        for j, (c, n) in enumerate(counts[k]):
            r = c / n
            dark = (r - vmin) / (vmax - vmin) > 0.55
            ax.text(j, i - 0.12, f"{100 * r:.1f}%", ha="center", va="center", fontsize=13,
                    fontweight="bold", color="#ffffff" if dark else INK)
            ax.text(j, i + 0.2, f"{c}/{n}", ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(GROUPS)), [g[2] for g in GROUPS], fontsize=11)
    ax.set_yticks(range(len(MODELS)), [n.replace("\n", " ") for _, n in MODELS], fontsize=11)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(GROUPS))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(MODELS))], minor=True)
    ax.grid(which="minor", color=SURFACE, lw=3)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_ticks([0.6, 0.7, 0.8, 0.9, 1.0], labels=["≤60%", "70%", "80%", "90%", "100%"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=INK2)
    cb.set_label("Correct predictions", color=INK2)
    fig.text(0.012, 0.965, "RSNA Normals are almost always right; TB cases are where models miss",
             fontsize=14, fontweight="bold", va="top")
    fig.text(0.012, 0.015, "Held-out test set, threshold 0.5. Counts from reports/preds_*.csv. "
             "NLM is the only source that contributes both classes.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(ASSETS / "source_heatmap.png")
    plt.close(fig)


def main():
    acc, recall, counts = load()
    ASSETS.mkdir(exist_ok=True)
    before_after(acc, recall)
    source_heatmap(counts)
    print(f"verified: notebook {acc}%, recall " +
          ", ".join(f"{k}={recall[k]:.2f}%" for k, _ in MODELS))
    for k, _ in MODELS:
        print(f"  {k:14s} " + "  ".join(f"{g[2].replace(chr(10), ' ')} {c}/{n}"
                                        for g, (c, n) in zip(GROUPS, counts[k])))
    print("wrote assets/before_after.png, assets/source_heatmap.png")


if __name__ == "__main__":
    main()
