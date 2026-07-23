"""자유 형식 생성: 실제 autoregressive 디코딩(model.generate)으로 학습된
모델을 돌려서, 모델이 실제로 생성하는 텍스트를 그대로 출력한다.

고정된 " yes"/" no"/" maybe" continuation을 teacher forcing으로 채점만 하고
샘플링은 하지 않는 evaluate.py / infer.py / metrics.predict와 달리, 이
스크립트는 출력을 어떤 라벨 집합으로도 제약하지 않는다. 모델을 그대로 두었을
때 실제로 뭐라고 답하는지 보기 위한 것이다.

사용법:
    python src/generate.py --config configs/lora.yaml
    python src/generate.py --config configs/lora.yaml --input data/infer_sample.jsonl
    python src/generate.py --config configs/lora.yaml --sample --temperature 0.7 --top_p 0.9
"""
import argparse
import json
import os

import torch

from common import DATA_DIR, load_config, output_dir_for, set_seed, setup_logging
from data import format_prompt, read_jsonl
from evaluate import load_eval_model
from model_utils import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", default=os.path.join(DATA_DIR, "infer_sample.jsonl"),
                     help="JSONL with {pubid, question, contexts[, label]} rows")
    ap.add_argument("--output", default=None,
                     help="Where to write results JSON (default: outputs/<exp>/generation_samples.json)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--sample", action="store_true",
                     help="Sample instead of greedy decoding (default: greedy, deterministic)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    out_dir = output_dir_for(exp)
    out_path = args.output or os.path.join(out_dir, "generation_samples.json")
    logger = setup_logging(f"generate.{exp}", f"{exp}_generate.log")
    set_seed(cfg["seed"])
    logger.info("=== GENERATE %s (method=%s, sample=%s) ===", exp, cfg["method"], args.sample)

    rows = read_jsonl(args.input)
    assert rows, f"No rows read from {args.input}"
    logger.info("Loaded %d sample(s) from %s", len(rows), args.input)

    model = load_eval_model(cfg, logger)
    model.eval()
    tokenizer = load_tokenizer(cfg["model_name"])
    tokenizer.padding_side = "left"  # 배치 causal generation이 올바르게 동작하려면 필요

    device = next(model.parameters()).device
    results = []
    for r in rows:
        prompt = format_prompt(r["question"], r["contexts"])
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                         max_length=cfg["max_seq_len"]).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.sample,
                temperature=args.temperature if args.sample else None,
                top_p=args.top_p if args.sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = gen_ids[0, enc["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        results.append({
            "pubid": r.get("pubid"),
            "question": r["question"],
            "true_label": r.get("label"),
            "generated_text": text,
        })
        logger.info("-" * 80)
        logger.info("pubid=%s", r.get("pubid"))
        logger.info("Q: %s", r["question"])
        logger.info("true=%s", r.get("label"))
        logger.info("generated: %r", text)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("=" * 80)
    logger.info("Saved generation results to %s", out_path)


if __name__ == "__main__":
    main()
