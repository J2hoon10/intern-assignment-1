"""Plot training/eval loss curves from saved training logs.

Purpose: verify that the loss decreases properly during training.
Reads outputs/<exp>/trainer_state.json (falls back to train_log.jsonl) and
saves outputs/<exp>/loss_curve.png. Metrics/params are NOT plotted here.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUTPUTS_DIR

TRAINABLE_EXPERIMENTS = ["full_ft", "lora"]
MULTIPHASE_EXPERIMENTS = ["full_ft_mp", "lora_mp"]


def load_log_history(log_dir):
    """Return the Trainer log_history list of records."""
    state_path = os.path.join(log_dir, "trainer_state.json")
    if os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as f:
            return json.load(f).get("log_history", [])
    jsonl_path = os.path.join(log_dir, "train_log.jsonl")
    if os.path.isfile(jsonl_path):
        recs = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        return recs
    return None


def best_cv_fold(exp):
    """Return the best_fold index selected in outputs/<exp>/phase2/cv_selection.json."""
    sel_path = os.path.join(OUTPUTS_DIR, exp, "phase2", "cv_selection.json")
    if not os.path.isfile(sel_path):
        return None
    with open(sel_path, encoding="utf-8") as f:
        return json.load(f).get("best_fold")


def plot_experiment(exp, phase_subdir=None, out_name="loss_curve.png", title_suffix=None):
    exp_dir = os.path.join(OUTPUTS_DIR, exp)
    log_dir = os.path.join(exp_dir, phase_subdir) if phase_subdir else exp_dir
    history = load_log_history(log_dir)
    if not history:
        print(f"[skip] no training log found for '{exp}' in {log_dir}")
        return

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    lr_steps, lr_vals = [], []
    for rec in history:
        step = rec.get("step")
        if step is None:
            continue
        if "loss" in rec:
            train_steps.append(step)
            train_loss.append(rec["loss"])
        if "eval_loss" in rec:
            eval_steps.append(step)
            eval_loss.append(rec["eval_loss"])
        if "learning_rate" in rec:
            lr_steps.append(step)
            lr_vals.append(rec["learning_rate"])

    fig, ax1 = plt.subplots(figsize=(9, 5))
    if train_loss:
        ax1.plot(train_steps, train_loss, label="train loss", color="#1f77b4", linewidth=1.5)
    if eval_loss:
        ax1.plot(eval_steps, eval_loss, label="eval loss", color="#d62728",
                 marker="o", markersize=4, linewidth=1.5)
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    title = f"Training curve — {exp}"
    if title_suffix:
        title += f" ({title_suffix})"
    elif phase_subdir:
        title += f" ({phase_subdir})"
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    if lr_vals:
        ax2 = ax1.twinx()
        ax2.plot(lr_steps, lr_vals, label="learning rate", color="#2ca02c",
                 linestyle="--", alpha=0.6)
        ax2.set_ylabel("learning rate")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    fig.tight_layout()
    out_path = os.path.join(exp_dir, out_name)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[ok] saved {out_path}")


def plot_selected_cv_fold(exp):
    """Plot the training curve of the phase-2 CV fold selected as best in cv_selection.json."""
    fold = best_cv_fold(exp)
    if fold is None:
        print(f"[skip] no phase2/cv_selection.json found for '{exp}'")
        return
    plot_experiment(
        exp,
        phase_subdir=os.path.join("phase2", f"fold{fold}"),
        out_name="loss_curve_phase2.png",
        title_suffix=f"phase2/fold{fold}, selected",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", help="single experiment name (e.g. full_ft, lora, full_ft_mp, lora_mp)")
    ap.add_argument("--all", action="store_true", help="plot all trainable experiments")
    args = ap.parse_args()

    if args.all or not args.experiment:
        for exp in TRAINABLE_EXPERIMENTS:
            plot_experiment(exp)
        for exp in MULTIPHASE_EXPERIMENTS:
            plot_experiment(exp, phase_subdir="phase1")
            plot_selected_cv_fold(exp)
    elif args.experiment in MULTIPHASE_EXPERIMENTS:
        plot_experiment(args.experiment, phase_subdir="phase1")
        plot_selected_cv_fold(args.experiment)
    else:
        plot_experiment(args.experiment)


if __name__ == "__main__":
    main()
