"""Evaluate a model on the official PubMedQA test set (500 instances).

Metrics (Accuracy & Macro-F1 + per-class F1 + confusion matrix) are written to
logs and JSON only — no charts. Also writes predictions.json in the official
PubMedQA format (pubid -> label) for cross-checking with the repo's evaluation.py,
and refreshes outputs/summary.{txt,json} comparing the three experiments.
"""
import argparse
import glob
import json
import os

import torch

from common import DATA_DIR, OUTPUTS_DIR, load_config, output_dir_for, set_seed, setup_logging
from data import LABELS, format_prompt, read_jsonl
from metrics import compute_metrics, predict
from model_utils import load_lora_adapter, load_model, load_tokenizer, pick_dtype

EXPERIMENTS = ["base", "full_ft", "lora", "lora_attn_only", "full_ft_mp", "lora_mp", "lora_mp_flush", "lora_mp_attn_flush"]


def load_eval_model(cfg, logger):
    method = cfg["method"]
    exp = cfg["experiment"]
    dtype, dtype_name = pick_dtype()
    logger.info("Loading model for eval | method=%s dtype=%s", method, dtype_name)
    if method == "base":
        model = load_model(cfg["model_name"], dtype=dtype)
    elif method == "full_ft":
        final_dir = os.path.join(output_dir_for(exp), "final")
        assert os.path.isdir(final_dir), f"Missing trained model: {final_dir} (run train.py first)"
        model = load_model(final_dir, dtype=dtype)
    elif method == "lora":
        adapter_dir = os.path.join(output_dir_for(exp), "adapter")
        assert os.path.isdir(adapter_dir), f"Missing LoRA adapter: {adapter_dir} (run train.py first)"
        model = load_lora_adapter(cfg["model_name"], adapter_dir, dtype=dtype)
    else:
        raise ValueError(f"Unknown method: {method}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)


def verify_test_set(rows, logger):
    dist = {lab: 0 for lab in LABELS}
    for r in rows:
        dist[r["label"]] = dist.get(r["label"], 0) + 1
    logger.info("Test set | n=%d | dist=%s", len(rows), dist)
    assert len(rows) == 500, f"Expected 500 test instances, got {len(rows)}"


def find_param_count(exp_dir):
    """Locate param_count.json for an experiment.

    train.py (single-stage) writes it directly to outputs/<exp>/param_count.json.
    train_multiphase.py writes one per unit instead (outputs/<exp>/phase1/,
    outputs/<exp>/phase2/fold{k}/, ...) and never aggregates one at the root.
    Prefer the unit that produced the deployed adapter (phase2's best fold, per
    cv_selection.json); otherwise fall back to any unit's file -- trainable/total
    are identical across units for a fixed LoRA config (same target_modules/r).
    """
    top = os.path.join(exp_dir, "param_count.json")
    if os.path.isfile(top):
        return top
    for phase_dir in sorted(glob.glob(os.path.join(exp_dir, "phase*"))):
        sel_path = os.path.join(phase_dir, "cv_selection.json")
        if os.path.isfile(sel_path):
            sel = json.load(open(sel_path, encoding="utf-8"))
            cand = os.path.join(phase_dir, f"fold{sel['best_fold']}", "param_count.json")
            if os.path.isfile(cand):
                return cand
    matches = sorted(glob.glob(os.path.join(exp_dir, "**", "param_count.json"), recursive=True))
    return matches[0] if matches else None


def write_summary(logger):
    """Aggregate metrics.json + param_count.json across experiments into a text table."""
    rows = []
    for exp in EXPERIMENTS:
        d = os.path.join(OUTPUTS_DIR, exp)
        m = os.path.join(d, "metrics.json")
        p = find_param_count(d)
        entry = {"experiment": exp}
        if os.path.isfile(m):
            md = json.load(open(m, encoding="utf-8"))
            entry["accuracy"] = md["accuracy"]
            entry["macro_f1"] = md["macro_f1"]
        if p and os.path.isfile(p):
            pd_ = json.load(open(p, encoding="utf-8"))
            entry["trainable"] = pd_["trainable"]
            entry["total"] = pd_["total"]
            entry["trainable_pct"] = pd_["trainable_pct"]
        rows.append(entry)

    with open(os.path.join(OUTPUTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    header = f"{'experiment':<10} {'accuracy':>9} {'macro_f1':>9} {'trainable':>15} {'total':>15} {'train%':>9}"
    lines = [header, "-" * len(header)]
    for e in rows:
        acc = f"{e['accuracy']:.4f}" if "accuracy" in e else "-"
        f1 = f"{e['macro_f1']:.4f}" if "macro_f1" in e else "-"
        tr = f"{e['trainable']:,}" if "trainable" in e else "-"
        tot = f"{e['total']:,}" if "total" in e else "-"
        pct = f"{e['trainable_pct']:.3f}" if "trainable_pct" in e else "-"
        lines.append(f"{e['experiment']:<10} {acc:>9} {f1:>9} {tr:>15} {tot:>15} {pct:>9}")
    table = "\n".join(lines)
    with open(os.path.join(OUTPUTS_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    logger.info("Experiment summary:\n%s", table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    out_dir = output_dir_for(exp)
    logger = setup_logging(f"eval.{exp}", f"{exp}_eval.log")
    set_seed(cfg["seed"])
    logger.info("=== EVAL %s (method=%s) ===", exp, cfg["method"])

    test_rows = read_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
    verify_test_set(test_rows, logger)

    model = load_eval_model(cfg, logger)
    tokenizer = load_tokenizer(cfg["model_name"])

    prompts = [format_prompt(r["question"], r["contexts"]) for r in test_rows]
    y_true = [r["label"] for r in test_rows]
    logger.info("Scoring %d prompts (batch_size=%d)...", len(prompts), cfg["eval"]["batch_size"])
    y_pred = predict(model, tokenizer, prompts,
                     max_seq_len=cfg["max_seq_len"], batch_size=cfg["eval"]["batch_size"])

    metrics = compute_metrics(y_true, y_pred)
    logger.info("Accuracy = %.4f | Macro-F1 = %.4f", metrics["accuracy"], metrics["macro_f1"])
    for lab in LABELS:
        pc = metrics["per_class"][lab]
        logger.info("  %-6s | P=%.3f R=%.3f F1=%.3f (support=%d)",
                    lab, pc["precision"], pc["recall"], pc["f1"], pc["support"])
    logger.info("Confusion matrix (rows=true %s):\n%s", LABELS, metrics["confusion_matrix"])

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    preds_map = {str(r["pubid"]): p for r, p in zip(test_rows, y_pred)}
    with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(preds_map, f, indent=2)
    logger.info("Saved metrics.json and predictions.json to %s", out_dir)

    write_summary(logger)
    logger.info("=== EVAL %s done ===", exp)


if __name__ == "__main__":
    main()
