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


# ---------------------------------------------------------------- LinkedIn versions -----
LI_DIR = ASSETS / "linkedin"
LI_EXPECTED = {"before_after_title": "94.8% accuracy, but it misses 31% of TB cases",
               "heatmap_title": "The fair test: NLM is the only source with both classes"}


def li_before_after(nb_acc, recall):
    """Square 1200x1200. Title numbers are computed from reports/ and must equal LI_EXPECTED."""
    m = {k: json.loads((REPORTS / f"metrics_{k}.json").read_text()) for k, _ in MODELS}
    cnn_acc = 100 * m["cnn"]["test"]["accuracy"]
    cnn_miss = 100 * (1 - m["cnn"]["test"]["recall"])
    title = f"{cnn_acc:.1f}% accuracy, but it misses {cnn_miss:.0f}% of TB cases"
    if title != LI_EXPECTED["before_after_title"]:
        return (f"before_after title from reports/metrics_cnn.json is '{title}' "
                f"(accuracy {cnn_acc:.4f}%, missed {cnn_miss:.4f}%), "
                f"requested '{LI_EXPECTED['before_after_title']}'")
    # bars: invalid | CNN accuracy, CNN recall | other models' recall
    bars = [("invalid", nb_acc, None, "Original\nnotebook\n“validation”"),
            ("acc", cnn_acc, None, "CNN\naccuracy"),
            ("recall", recall["cnn"], m["cnn"]["test_ci95"]["recall"], "CNN\nrecall")]
    li_names = {"resnet_frozen": "ResNet50\nfrozen", "resnet_ft": "ResNet50\nfine-tuned\n12 ep",
                "resnet_ft25": "ResNet50\nfine-tuned\n25 ep"}
    bars += [("recall", recall[k], m[k]["test_ci95"]["recall"], li_names[k]) for k, _ in MODELS[1:]]
    xs = [0, 1.25, 2.05, 3.3, 4.4, 5.5]  # gap after the invalid bar; CNN pair sits together
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    for x, (kind, v, ci, _) in zip(xs, bars):
        style = {"invalid": dict(color=SURFACE, edgecolor=MUTED, hatch="///", linewidth=1.6),
                 "acc": dict(color="#9ec5f4", edgecolor=SURFACE, linewidth=2),
                 "recall": dict(color=BLUE, edgecolor=SURFACE, linewidth=2)}[kind]
        ax.bar(x, v, width=0.72, zorder=3, **style)
        top = v
        if ci:
            lo, hi = 100 * ci[0], 100 * ci[1]
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=1.4,
                        capsize=5, capthick=1.4, zorder=4)
            top = hi
        ax.text(x, top + 1.5, f"{v:.2f}%" if kind == "invalid" else f"{v:.1f}%", ha="center",
                va="bottom", fontsize=13, fontweight="bold", color=INK2 if kind == "invalid" else INK)
    ax.text(xs[0], 45, "INVALID\nmeasured on\ntraining data", ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK2, zorder=5,
            bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=MUTED, lw=1))
    ax.axvline((xs[0] + xs[1]) / 2, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(xs[1] - 0.36, 109, "Held-out test set (621 images, 101 TB)", fontsize=11.5, color=INK2)
    ax.set_xticks(xs, [b[3] for b in bars], fontsize=10.5)
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20), [f"{t}%" for t in range(0, 101, 20)], fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0)
    fig.text(0.03, 0.965, title, fontsize=19, fontweight="bold", va="top")
    fig.text(0.03, 0.915, "Recall = share of TB cases caught. Light bar = accuracy; dark bars = recall.",
             fontsize=11.5, color=INK2, va="top")
    fig.text(0.03, 0.02, "Error bars: 95% bootstrap confidence intervals (2000 resamples). Threshold 0.5. "
             "The notebook figure\nwas computed on images it trained on; shown for contrast, not comparison.",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    fig.savefig(LI_DIR / "before_after.png")
    plt.close(fig)
    return None


def li_source_heatmap(counts):
    from matplotlib.patches import Rectangle
    title = LI_EXPECTED["heatmap_title"]
    # the claim in the title must hold in the data: NLM is the only source with both labels
    idx = pd.read_csv(REPORTS / "image_index.csv")
    both = sorted(s for s, g in idx.groupby("source") if g.label.nunique() == 2)
    if both != ["NLM"]:
        return f"heatmap title claims NLM is the only two-class source; image_index.csv gives {both}"
    rates = [[c / n for c, n in counts[k]] for k, _ in MODELS]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    vmin, vmax = 0.6, 1.0
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    im = ax.imshow(rates, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i, (k, _) in enumerate(MODELS):
        for j, (c, n) in enumerate(counts[k]):
            r = c / n
            dark = (r - vmin) / (vmax - vmin) > 0.55
            col = "#ffffff" if dark else INK
            ax.text(j, i - 0.12, f"{100 * r:.1f}%", ha="center", va="center", fontsize=15,
                    fontweight="bold", color=col)
            ax.text(j, i + 0.2, f"{c}/{n}", ha="center", va="center", fontsize=11, color=col)
    ax.set_xticks(range(len(GROUPS)), [g[2] for g in GROUPS], fontsize=12)
    ax.set_yticks(range(len(MODELS)), [n for _, n in MODELS], fontsize=11.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(GROUPS))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(MODELS))], minor=True)
    ax.grid(which="minor", color=SURFACE, lw=3)
    ax.tick_params(which="minor", length=0)
    for j, (src, _, _) in enumerate(GROUPS):  # outline the NLM columns
        if src == "NLM":
            ax.add_patch(Rectangle((j - 0.5, -0.5), 1, len(MODELS), fill=False, ec="#eb6834",
                                   lw=3.5, zorder=5, clip_on=False))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_ticks([0.6, 0.7, 0.8, 0.9, 1.0], labels=["≤60%", "70%", "80%", "90%", "100%"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=INK2, labelsize=10.5)
    cb.set_label("Correct predictions", color=INK2, fontsize=11)
    fig.text(0.03, 0.965, title, fontsize=17.5, fontweight="bold", va="top")
    fig.text(0.03, 0.918, "Outlined: NLM images. RSNA has only Normals and Belarus only TB, so there\n"
             "a model could score well just by recognising the source.",
             fontsize=11, color=INK2, va="top")
    fig.text(0.03, 0.02, "Held-out test set, threshold 0.5. Cells show % correct and count.",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.85))
    fig.savefig(LI_DIR / "source_heatmap.png")
    plt.close(fig)
    return None


def main():
    acc, recall, counts = load()
    ASSETS.mkdir(exist_ok=True)
    before_after(acc, recall)
    source_heatmap(counts)
    LI_DIR.mkdir(exist_ok=True)
    li_errors = [e for e in (li_source_heatmap(counts), li_before_after(acc, recall)) if e]
    print(f"verified: notebook {acc}%, recall " +
          ", ".join(f"{k}={recall[k]:.2f}%" for k, _ in MODELS))
    for k, _ in MODELS:
        print(f"  {k:14s} " + "  ".join(f"{g[2].replace(chr(10), ' ')} {c}/{n}"
                                        for g, (c, n) in zip(GROUPS, counts[k])))
    print("wrote assets/before_after.png, assets/source_heatmap.png")
    if li_errors:
        fail("; ".join(li_errors))
    print("wrote assets/linkedin/before_after.png, assets/linkedin/source_heatmap.png")


if __name__ == "__main__":
    main()
