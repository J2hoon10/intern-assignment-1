"""실험용 PubMedQA 데이터를 준비한다.

생성물 (``data/`` 아래):
  - test.jsonl          공식 500개 test 셋 (question, contexts, label)
  - train.jsonl         PQA-A subset + PQA-L CV train (혼합, 셔플됨)
  - dev.jsonl           PQA-L CV dev split
  - test_pubids.json    500개 test pubid의 정규 집합
  - split_pubids.json   train / dev에 들어간 pubid (감사용 기록)

누수 방지(하드, fail-fast) — 계획서 §3.1 참고:
  1. test_set.json과 test_ground_truth.json 모두에서 정규 test pubid 집합을
     구함(둘이 일치해야 하고 크기는 500이어야 함).
  2. PQA-L과 PQA-A 양쪽 모두에서 test pubid를 제거.
  3. train/dev pubid가 test와(그리고 서로) 겹치지 않는지 assert, 위반 시 RAISE.
  4. 2차 콘텐츠 가드: 정규화한 질문 텍스트 해시가 어떤 test 질문과 충돌하는
     train/dev row는 제거.
  5. 감사 요약을 logs/prepare_data.log에 기록.
"""
import argparse
import json
import os
import re
import urllib.request

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from common import DATA_DIR, set_seed, setup_logging
from data import LABELS, write_jsonl

TEST_SET_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_set.json"
TEST_GT_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
HF_DATASET = "qiaojin/PubMedQA"


# ---------------------------------------------------------------------------
def _download_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", str(q).strip().lower())


def qhash(q: str) -> str:
    return normalize_question(q)


def label_dist(rows):
    d = {lab: 0 for lab in LABELS}
    for r in rows:
        d[r["label"]] = d.get(r["label"], 0) + 1
    return d


# ---------------------------------------------------------------------------
def build_test_set(logger):
    """공식 test 셋을 다운로드한다. 실패 시 HF에서 재구성한다."""
    ground_truth = _download_json(TEST_GT_URL)          # {pmid(str): label}
    gt_pubids = {int(k) for k in ground_truth}
    try:
        test_set = _download_json(TEST_SET_URL)          # {pmid(str): {QUESTION, CONTEXTS, ...}}
        rows = []
        for pmid, v in test_set.items():
            rows.append({
                "pubid": int(pmid),
                "question": v["QUESTION"],
                "contexts": v["CONTEXTS"],
                "label": ground_truth[pmid],
            })
        source = "github test_set.json"
    except Exception as e:  # noqa: BLE001 - HF 재구성으로 폴백
        logger.warning("Could not download test_set.json (%s); reconstructing from HF pqa_labeled.", e)
        labeled = load_dataset(HF_DATASET, "pqa_labeled")["train"]
        by_id = {int(r["pubid"]): r for r in labeled}
        rows = []
        for pmid_str, label in ground_truth.items():
            r = by_id[int(pmid_str)]
            rows.append({
                "pubid": int(pmid_str),
                "question": r["question"],
                "contexts": r["context"]["contexts"],
                "label": label,
            })
        source = "HF pqa_labeled (reconstructed)"

    pubid_set = {r["pubid"] for r in rows}
    assert pubid_set == gt_pubids, "test_set.json keys and test_ground_truth.json keys disagree!"
    assert len(rows) == 500, f"Expected 500 test instances, got {len(rows)}"
    logger.info("Test set built from %s | %d rows | dist=%s", source, len(rows), label_dist(rows))
    return rows, pubid_set


def build_labeled_cv(test_pubids, logger):
    """PQA-L(1000)에서 500개 test pubid를 뺀 나머지 = 500개 CV 인스턴스."""
    labeled = load_dataset(HF_DATASET, "pqa_labeled")["train"]
    cv = []
    for r in labeled:
        pid = int(r["pubid"])
        if pid in test_pubids:
            continue
        cv.append({
            "pubid": pid,
            "question": r["question"],
            "contexts": r["context"]["contexts"],
            "label": str(r["final_decision"]),
        })
    logger.info("PQA-L: %d total -> %d CV after removing test pubids | dist=%s",
                len(labeled), len(cv), label_dist(cv))
    assert len(cv) == 500, f"Expected 500 CV instances, got {len(cv)}"
    return cv


def build_pqaa_subset(test_pubids, size, seed, logger):
    """PQA-A(artificial)를 셔플한 subset. 여기에는 'maybe' 라벨이 없다."""
    artificial = load_dataset(HF_DATASET, "pqa_artificial")["train"]
    artificial = artificial.shuffle(seed=seed)
    rows, removed = [], 0
    for r in artificial:
        pid = int(r["pubid"])
        if pid in test_pubids:            # 방어적 처리: PQA-A는 설계상 test와 겹치지 않음
            removed += 1
            continue
        rows.append({
            "pubid": pid,
            "question": r["question"],
            "contexts": r["context"]["contexts"],
            "label": str(r["final_decision"]),
        })
        if len(rows) >= size:
            break
    logger.info("PQA-A: sampled %d rows (removed %d test-overlap) | dist=%s",
                len(rows), removed, label_dist(rows))
    return rows


# ---------------------------------------------------------------------------
def hard_leakage_check(train_rows, dev_rows, test_pubids, test_questions, logger):
    """Fail-fast 방식의 누수 방지. 정제된 (train, dev)를 반환한다."""
    # (1) 콘텐츠 기반 가드: 질문이 test 질문과 충돌하는 row는 제거.
    def drop_question_overlap(rows, name):
        kept, dropped = [], 0
        for r in rows:
            if qhash(r["question"]) in test_questions:
                dropped += 1
                continue
            kept.append(r)
        if dropped:
            logger.warning("%s: dropped %d rows overlapping test question text.", name, dropped)
        else:
            logger.info("%s: 0 rows overlap test question text.", name)
        return kept

    train_rows = drop_question_overlap(train_rows, "train")
    dev_rows = drop_question_overlap(dev_rows, "dev")

    # (2) pubid 겹침 여부 — 위반 시 RAISE.
    train_ids = {r["pubid"] for r in train_rows}
    dev_ids = {r["pubid"] for r in dev_rows}
    train_leak = train_ids & test_pubids
    dev_leak = dev_ids & test_pubids
    tr_dev_overlap = train_ids & dev_ids
    if train_leak:
        raise RuntimeError(f"LEAKAGE: {len(train_leak)} train pubids are in the test set: {list(train_leak)[:5]}")
    if dev_leak:
        raise RuntimeError(f"LEAKAGE: {len(dev_leak)} dev pubids are in the test set: {list(dev_leak)[:5]}")
    if tr_dev_overlap:
        raise RuntimeError(f"OVERLAP: {len(tr_dev_overlap)} pubids shared between train and dev.")

    logger.info("Leakage check PASSED | train n=%d dev n=%d | train/dev vs test overlap=0 | train vs dev overlap=0",
                len(train_rows), len(dev_rows))
    return train_rows, dev_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pqaa_size", type=int, default=30000, help="number of PQA-A examples to sample")
    ap.add_argument("--dev_size", type=int, default=50, help="dev split size from the 500 CV")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    logger = setup_logging("prepare_data", "prepare_data.log")
    set_seed(args.seed)
    logger.info("=== prepare_data | pqaa_size=%d dev_size=%d seed=%d ===",
                args.pqaa_size, args.dev_size, args.seed)

    # 1. 정규 test 셋.
    test_rows, test_pubids = build_test_set(logger)
    test_questions = {qhash(r["question"]) for r in test_rows}

    # 2. PQA-L CV(500)와 PQA-A subset.
    cv_rows = build_labeled_cv(test_pubids, logger)
    pqaa_rows = build_pqaa_subset(test_pubids, args.pqaa_size, args.seed, logger)

    # 3. 층화(stratified) CV -> train / dev.
    cv_train, cv_dev = train_test_split(
        cv_rows, test_size=args.dev_size, stratify=[r["label"] for r in cv_rows], random_state=args.seed
    )

    # 4. PQA-A subset + PQA-L train 혼합.
    train_rows = pqaa_rows + cv_train
    import random
    random.Random(args.seed).shuffle(train_rows)
    dev_rows = cv_dev

    # 5. 하드 누수 검사 (fail-fast).
    train_rows, dev_rows = hard_leakage_check(train_rows, dev_rows, test_pubids, test_questions, logger)

    # 6. 결과물 + 감사 기록 저장.
    write_jsonl(test_rows, os.path.join(DATA_DIR, "test.jsonl"))
    write_jsonl(train_rows, os.path.join(DATA_DIR, "train.jsonl"))
    write_jsonl(dev_rows, os.path.join(DATA_DIR, "dev.jsonl"))
    with open(os.path.join(DATA_DIR, "test_pubids.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(test_pubids), f)
    with open(os.path.join(DATA_DIR, "split_pubids.json"), "w", encoding="utf-8") as f:
        json.dump({"train": sorted({r["pubid"] for r in train_rows}),
                   "dev": sorted({r["pubid"] for r in dev_rows})}, f)

    logger.info("Wrote train=%d dev=%d test=%d to %s", len(train_rows), len(dev_rows), len(test_rows), DATA_DIR)
    logger.info("Class dist | train=%s dev=%s test=%s",
                label_dist(train_rows), label_dist(dev_rows), label_dist(test_rows))
    logger.info("=== prepare_data done ===")


if __name__ == "__main__":
    main()
