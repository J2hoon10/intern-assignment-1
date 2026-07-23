"""정성적 추론: 라벨이 있는 소수의 예시에 대해 학습된 모델을 돌려 (입력 /
예측 / 정답) 비교 결과를 나란히 저장한다.

evaluate.py(공식 500개 test 전체를 채점해 집계 Accuracy/Macro-F1을 구함)와
달리, 이 스크립트는 개별 사례를 눈으로 확인하기 위한 것이다: 예시마다 질문,
모델의 예측, 정답, 둘의 일치 여부, 그리고 (evaluate.py와 동일한 제약
yes/no/maybe 스코어링에서 나온 길이정규화 로그확률인) 라벨별 신뢰도를
출력하고 저장한다.

사용법:
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
