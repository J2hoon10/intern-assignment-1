"""PQA-L 10-fold CV를 포함한 2단계 학습용 데이터셋을 준비한다.

Phase 1은 PQA-A로 학습하고, Phase 2는 (공식 PubMedQA 프로토콜인) 10-fold
교차검증으로 PQA-L을 fine-tune한다. 검증 결과가 가장 좋은 fold를 테스트한다.

생성물:
  data/test.jsonl                     공식 500개 test 셋 (정규)
  data/mp/pqaa.jsonl                  PQA-A train subset          (Phase 1)
  data/mp/pqaa_dev.jsonl              PQA-A held-out dev           (Phase 1 검증)
  data/mp/pqal_fold{k}/train.jsonl    450 train  (k = 0..n_folds-1)  (Phase 2)
  data/mp/pqal_fold{k}/val.jsonl      50  val    (k = 0..n_folds-1)  (Phase 2)

누수 보장(fail-fast): 모든 PQA-A/PQA-L row에서 500개 test pubid를 제거하고;
각 fold 내에서 train ∩ val == 0; (train ∪ val) ∩ test == 0.
"""
import argparse
import os

from sklearn.model_selection import StratifiedKFold

from common import DATA_DIR, set_seed, setup_logging
from data import write_jsonl
from prepare_data import build_labeled_cv, build_pqaa_subset, build_test_set, label_dist

MP_DIR = os.path.join(DATA_DIR, "mp")


def assert_no_test_overlap(rows, test_pubids, name, logger):
    overlap = {r["pubid"] for r in rows} & test_pubids
    if overlap:
        raise RuntimeError(f"LEAKAGE: {name} shares {len(overlap)} pubids with test: {list(overlap)[:5]}")
    logger.info("Leakage OK | %s: n=%d, overlap with test = 0", name, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pqaa_size", type=int, default=50000, help="Phase 1: PQA-A train size")
    ap.add_argument("--pqaa_dev_size", type=int, default=500, help="Phase 1: PQA-A held-out dev size")
    ap.add_argument("--n_folds", type=int, default=10, help="PQA-L CV folds")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(MP_DIR, exist_ok=True)
    logger = setup_logging("prepare_data_mp", "prepare_data_mp.log")
    set_seed(args.seed)
    logger.info("=== prepare_data_multiphase (PQA-A -> PQA-L %d-fold) | pqaa=%d ===",
                args.n_folds, args.pqaa_size)

    # 1) 정규 test 셋.
    test_rows, test_pubids = build_test_set(logger)
    write_jsonl(test_rows, os.path.join(DATA_DIR, "test.jsonl"))

    # 2) Phase 1: PQA-A train + held-out PQA-A dev (Phase-1 모델 선택을
    #    PQA-L fold와 독립적으로 유지하기 위함).
    pqaa_all = build_pqaa_subset(test_pubids, args.pqaa_size + args.pqaa_dev_size, args.seed, logger)
    pqaa_train = pqaa_all[: args.pqaa_size]
    pqaa_dev = pqaa_all[args.pqaa_size:]
    assert_no_test_overlap(pqaa_train, test_pubids, "PQA-A train", logger)
    assert_no_test_overlap(pqaa_dev, test_pubids, "PQA-A dev", logger)
    write_jsonl(pqaa_train, os.path.join(MP_DIR, "pqaa.jsonl"))
    write_jsonl(pqaa_dev, os.path.join(MP_DIR, "pqaa_dev.jsonl"))

    # 3) Phase 2: PQA-L CV 500 -> 층화 n-fold (각 450 train / 50 val).
    cv_rows = build_labeled_cv(test_pubids, logger)
    assert_no_test_overlap(cv_rows, test_pubids, "PQA-L CV", logger)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    labels = [r["label"] for r in cv_rows]
    for k, (tr_idx, va_idx) in enumerate(skf.split(cv_rows, labels)):
        fold_train = [cv_rows[i] for i in tr_idx]
        fold_val = [cv_rows[i] for i in va_idx]
        tr_ids = {r["pubid"] for r in fold_train}
        va_ids = {r["pubid"] for r in fold_val}
        if tr_ids & va_ids:
            raise RuntimeError(f"LEAKAGE: fold{k} train/val share {len(tr_ids & va_ids)} pubids")
        if (tr_ids | va_ids) & test_pubids:
            raise RuntimeError(f"LEAKAGE: fold{k} overlaps test set")
        fold_dir = os.path.join(MP_DIR, f"pqal_fold{k}")
        os.makedirs(fold_dir, exist_ok=True)
        write_jsonl(fold_train, os.path.join(fold_dir, "train.jsonl"))
        write_jsonl(fold_val, os.path.join(fold_dir, "val.jsonl"))
        logger.info("fold%d | train=%d val=%d | val dist=%s | train/val & test overlap=0",
                    k, len(fold_train), len(fold_val), label_dist(fold_val))

    logger.info("Wrote pqaa_train=%d pqaa_dev=%d + %d PQA-L folds | test=%d",
                len(pqaa_train), len(pqaa_dev), args.n_folds, len(test_rows))
    logger.info("Class dist | PQA-A train=%s test=%s", label_dist(pqaa_train), label_dist(test_rows))
    logger.info("=== prepare_data_multiphase done ===")


if __name__ == "__main__":
    main()
