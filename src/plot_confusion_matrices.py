"""Plot confusion-matrix heatmaps from saved eval metrics.

Purpose: visually compare per-class error patterns across experiments.
Reads outputs/<exp>/metrics.json (written by evaluate.py) and saves a single
combined heatmap figure to outputs/confusion_matrices/confusion_matrices.png.
Each panel shows raw prediction counts (sklearn/seaborn confusion-matrix
style); pass --normalize true/pred to shade by row/column percentage instead
(cell labels still show the raw count).
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUTPUTS_DIR

DEFAULT_EXPERIMENTS = [
    ("base", "Base"),
    ("full_ft_mp", "Full fine tuning"),
    ("lora_mp", "LoRA"),
]

CMAP = "Blues"


def load_metrics(exp):
    """Return the metrics.json dict for an experiment, or None if missing."""
    path = os.path.join(OUTPUTS_DIR, exp, "metrics.json")
    if not os.path.isfile(path):
        print(f"[skip] no metrics.json found for '{exp}' at {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_matrix(mat, mode="true"):
    """Row-normalize (mode='true') so each true label's row sums to 1.0."""
    mat = mat.astype(float)
    if mode == "true":
        sums = mat.sum(axis=1, keepdims=True)
    elif mode == "pred":
        sums = mat.sum(axis=0, keepdims=True)
    else:
        return mat
    sums[sums == 0] = 1
    return mat / sums


def plot_confusion_matrices(experiments, normalize="true", out_name="confusion_matrices.png"):
    """experiments: list of (exp_dir_name, display_title) tuples."""
    datasets = []
    for exp, title in experiments:
        m = load_metrics(exp)
        if m is None:
            continue
        raw = np.array(m["confusion_matrix"])
        datasets.append({
            "title": title,
            "subtitle": f"Accuracy {m['accuracy'] * 100:.1f}%  ·  Macro-F1 {m['macro_f1']:.3f}",
            "raw": raw,
            "color": normalize_matrix(raw, normalize) if normalize != "none" else raw.astype(float),
            "labels": m["labels"],
        })
    if not datasets:
        print("[skip] no metrics found for any requested experiment")
        return

    vmax = 1.0 if normalize != "none" else max(ds["color"].max() for ds in datasets)

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.2), facecolor="white")
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        color_mat = ds["color"]
        raw_mat = ds["raw"]
        labels = ds["labels"]
        im = ax.imshow(color_mat, cmap=CMAP, vmin=0, vmax=vmax)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        thresh = color_mat.max() / 2.0
        for r in range(len(labels)):
            for c in range(len(labels)):
                ax.text(c, r, str(raw_mat[r, c]), ha="center", va="center",
                         color="white" if color_mat[r, c] > thresh else "black", fontsize=11)

        title = "Normalized Confusion Matrix" if normalize != "none" else "Confusion Matrix"
        ax.set_title(f"{title}\n{ds['title']}  ({ds['subtitle']})", fontsize=11)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    out_dir = os.path.join(OUTPUTS_DIR, "confusion_matrices")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[ok] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiments", nargs="+",
        help="experiment dir names to plot (default: base full_ft_mp lora_mp)",
    )
    ap.add_argument(
        "--normalize", choices=["true", "pred", "none"], default="none",
        help="'none' = raw counts (default), 'true' = row-normalize shading (recall), 'pred' = column-normalize shading (precision); cell labels always show raw counts",
    )
    ap.add_argument("--out-name", default="confusion_matrices.png")
    args = ap.parse_args()

    if args.experiments:
        pairs = [(exp, exp) for exp in args.experiments]
    else:
        pairs = DEFAULT_EXPERIMENTS

    plot_confusion_matrices(pairs, normalize=args.normalize, out_name=args.out_name)


if __name__ == "__main__":
    main()
