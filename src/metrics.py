"""yes/no/maybe 제약 스코어링(constrained scoring) 및 분류 지표.

각 프롬프트에 대해, 모델이 후속 텍스트로 각 후보 답("yes" / "no" / "maybe")에
부여하는 길이정규화 로그확률을 계산한 뒤 arg-max 후보를 예측값으로 삼는다. 이
방식은 학습되지 않은 base 모델(순수 zero-shot)뿐 아니라 fine-tuned 모델에도
그대로 적용되며, 자유 형식 텍스트 파싱이 필요 없다.
"""
from typing import List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from data import LABELS


@torch.no_grad()
def predict(model, tokenizer, prompts: List[str], max_seq_len: int = 1024,
            batch_size: int = 16, device=None, return_scores: bool = False):
    """각 프롬프트에 대한 예측 라벨(yes/no/maybe)을 반환한다.

    ``return_scores``가 True이면, 후보별 길이정규화 로그확률로 이루어진 [N, 3]
    배열도 함께 반환한다(열 순서 = LABELS). pseudo-label 신뢰도 필터링에 쓰인다.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # 후보 continuation들을 미리 한 번만 토큰화해 둔다.
    cand_ids = [tokenizer(" " + lab, add_special_tokens=False)["input_ids"] for lab in LABELS]
    pad_id = tokenizer.pad_token_id

    preds: List[str] = []
    all_scores: List[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        prompt_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in batch_prompts]
        scores = np.full((len(batch_prompts), len(LABELS)), -np.inf, dtype=np.float64)

        for ci, cids in enumerate(cand_ids):
            seqs = []
            for pids in prompt_ids:
                max_p = max(1, max_seq_len - len(cids))
                pids2 = pids[-max_p:] if len(pids) > max_p else pids
                seqs.append(pids2 + cids)

            maxlen = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for i, s in enumerate(seqs):
                input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
                attn[i, :len(s)] = 1
            input_ids = input_ids.to(device)
            attn = attn.to(device)

            logits = model(input_ids=input_ids, attention_mask=attn).logits
            logprobs = torch.log_softmax(logits.float(), dim=-1)

            clen = len(cids)
            for i, s in enumerate(seqs):
                total = 0.0
                real_len = len(s)
                for k in range(clen):
                    pos = real_len - clen + k          # 후보 토큰의 위치
                    tok_id = s[pos]
                    total += logprobs[i, pos - 1, tok_id].item()   # pos-1에서 예측된 값
                scores[i, ci] = total / clen           # 길이정규화
        for i in range(len(batch_prompts)):
            preds.append(LABELS[int(np.argmax(scores[i]))])
        if return_scores:
            all_scores.append(scores)
    if return_scores:
        return preds, (np.vstack(all_scores) if all_scores else np.zeros((0, len(LABELS))))
    return preds


def compute_metrics(y_true: List[str], y_pred: List[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    per_class = {
        lab: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, lab in enumerate(LABELS)
    }
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": LABELS,
    }
