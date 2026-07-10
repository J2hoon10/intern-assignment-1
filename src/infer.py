"""Qualitative inference: run a trained model on a small handful of labeled
examples and save side-by-side comparisons (input / prediction / gold label).

Unlike evaluate.py (which scores the full 500-example official test set for
aggregate Accuracy/Macro-F1), this script is for eyeballing individual cases:
it prints and saves, per example, the question, the model's prediction, the
gold label, whether they match, and the per-label confidence (length-normalized
log-prob from the same constrained yes/no/maybe scoring used in evaluate.py).

Usage:
    python src/infer.py --config configs/lora.yaml
    python src/infer.py --config configs/lora.yaml --input data/infer_sample.jsonl
"""
import argparse
import json
import os

from common import DATA_DIR, load_config, output_dir_for, set_seed, setup_logging
from data import LABELS, format_prompt, read_jsonl
from evaluate import load_eval_model
from metrics import predict
from model_utils import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", default=os.path.join(DATA_DIR, "infer_sample.jsonl"),
                     help="JSONL with {pubid, question, contexts[, label]} rows")
    ap.add_argument("--output", default=None,
                     help="Where to write results JSON (default: outputs/<exp>/inference_samples.json)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    out_dir = output_dir_for(exp)
    out_path = args.output or os.path.join(out_dir, "inference_samples.json")
    logger = setup_logging(f"infer.{exp}", f"{exp}_infer.log")
    set_seed(cfg["seed"])
    logger.info("=== INFER %s (method=%s) ===", exp, cfg["method"])

    rows = read_jsonl(args.input)
    assert rows, f"No rows read from {args.input}"
    logger.info("Loaded %d sample(s) from %s", len(rows), args.input)

    model = load_eval_model(cfg, logger)
    tokenizer = load_tokenizer(cfg["model_name"])

    prompts = [format_prompt(r["question"], r["contexts"]) for r in rows]
    y_true = [r.get("label") for r in rows]
    y_pred, scores = predict(model, tokenizer, prompts,
                              max_seq_len=cfg["max_seq_len"],
                              batch_size=cfg["eval"]["batch_size"],
                              return_scores=True)

    results = []
    n_labeled, n_correct = 0, 0
    for r, prompt, true, pred, score in zip(rows, prompts, y_true, y_pred, scores):
        conf = {lab: float(score[i]) for i, lab in enumerate(LABELS)}
        correct = None
        if true is not None:
            correct = (pred == true)
            n_labeled += 1
            n_correct += int(correct)
        results.append({
            "pubid": r.get("pubid"),
            "question": r["question"],
            "contexts": r["contexts"],
            "prompt": prompt,
            "true_label": true,
            "pred_label": pred,
            "correct": correct,
            "confidence": conf,
        })

        logger.info("-" * 80)
        logger.info("pubid=%s", r.get("pubid"))
        logger.info("Q: %s", r["question"])
        status = "OK" if correct else ("WRONG" if correct is False else "N/A")
        logger.info("true=%-6s pred=%-6s [%s]", true, pred, status)
        logger.info("confidence: %s", {k: round(v, 3) for k, v in conf.items()})

    logger.info("=" * 80)
    if n_labeled:
        logger.info("Accuracy on this sample: %d/%d = %.3f", n_correct, n_labeled, n_correct / n_labeled)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved inference results to %s", out_path)


if __name__ == "__main__":
    main()
