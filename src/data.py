"""프롬프트 포맷팅, 토큰화(정답 부분에만 loss 부여), collation.

이 과제는 causal LM으로 진행되며, 모델은 ``Answer (yes/no/maybe):``로 끝나는
프롬프트를 읽고 한 단어(`yes`/`no`/`maybe`)를 생성해야 한다. 학습 중에는
프롬프트 토큰을 -100으로 마스킹해서 loss가 정답 토큰에서만 계산되도록 한다.
"""
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset as TorchDataset

LABELS = ["yes", "no", "maybe"]

PROMPT_TEMPLATE = "Context: {context}\nQuestion: {question}\nAnswer (yes/no/maybe):"


def join_contexts(contexts) -> str:
    """PubMedQA `context`는 초록 문단들의 리스트이므로 하나의 문자열로 합친다."""
    if isinstance(contexts, str):
        return contexts.strip()
    return " ".join(c.strip() for c in contexts if c and c.strip())


def format_prompt(question: str, contexts) -> str:
    return PROMPT_TEMPLATE.format(context=join_contexts(contexts), question=str(question).strip())


# ---------------------------------------------------------------------------
# JSONL 헬퍼
# ---------------------------------------------------------------------------
def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 학습용 토큰화
# ---------------------------------------------------------------------------
def tokenize_example(tokenizer, question, contexts, label: str, max_seq_len: int) -> Dict[str, List[int]]:
    """학습 예시 하나에 대해 input_ids / attention_mask / labels를 구성한다.

    프롬프트 토큰은 -100으로 마스킹되어 loss가 정답에만 걸리도록 한다.
    시퀀스가 너무 길면 프롬프트를 앞쪽에서 자르므로(left-truncate), 프롬프트
    끝부분의 질문과 정답은 항상 보존된다.
    """
    prompt = format_prompt(question, contexts)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    max_prompt_len = max_seq_len - len(answer_ids)
    if max_prompt_len < 1:
        max_prompt_len = 1
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]  # 뒷부분(질문 + 정답 유도 문구)을 남긴다

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class ListDataset(TorchDataset):
    """feature dict 리스트를 기반으로 하는 최소한의 map-style 데이터셋.

    여기서는 의도적으로 ``datasets.Dataset``(pyarrow)을 쓰지 않는다: Windows에서는
    pyarrow와 accelerate/torch가 서로 충돌하는 네이티브 런타임을 끌어들여서 같은
    프로세스에서 둘 다 import하면 segfault가 난다. 일반 torch Dataset을 쓰면
    pyarrow를 아예 거치지 않고 transformers.Trainer + 우리 collator와 바로 동작한다.
    """

    def __init__(self, items: List[Dict[str, List[int]]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.items[idx]


def build_dataset(tokenizer, rows: List[dict], max_seq_len: int) -> "ListDataset":
    """{question, contexts, label} row 리스트를 토큰화해 torch Dataset으로 만든다."""
    features = [
        tokenize_example(tokenizer, r["question"], r["contexts"], r["label"], max_seq_len)
        for r in rows
    ]
    return ListDataset(features)


# ---------------------------------------------------------------------------
# Collator: input_ids / attention_mask / labels의 동적 패딩
# ---------------------------------------------------------------------------
@dataclass
class CausalCollator:
    """배치 길이를 (배치 자체의 최댓값이 아니라) 고정된 배수 단위로 패딩한다.
    TrainingArguments의 group_by_length와 결합하면, 학습 중 거의 매 스텝마다
    새로운 shape이 나오는 대신 적고 반복되는 패딩 shape 집합만 보게 된다.
    이 Windows/WDDM 환경에서는 이런 shape 변동이 CUDA 캐싱 allocator를
    파편화시켜 예약 메모리가 16GB 카드를 초과했고, 결국 (잡을 수 있는 OOM이
    아니라) 치명적인 CUBLAS 에러로 학습이 죽었었다."""

    tokenizer: Any
    label_pad_token_id: int = -100
    pad_to_multiple_of: int = 64
    # 진단용 데이터(CudaMemoryProbeCallback이 읽음): probe가 마지막으로 비운 이후
    # 발생한 모든 패딩된 (batch, seq_len) shape과, 누적 히스토그램.
    # train/eval 배치 모두 여기 쌓이며, probe가 배치 크기로 이를 구분해서 표시한다.
    shapes_since_probe: List[Tuple[int, int]] = field(default_factory=list, repr=False)
    shape_counts: Counter = field(default_factory=Counter, repr=False)

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        batch_max = max(len(f["input_ids"]) for f in features)
        max_len = -(-batch_max // self.pad_to_multiple_of) * self.pad_to_multiple_of
        shape = (len(features), max_len)
        self.shapes_since_probe.append(shape)
        self.shape_counts[shape] += 1
        input_ids, attn, labels = [], [], []
        for f in features:
            n = max_len - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [pad_id] * n)
            attn.append(list(f["attention_mask"]) + [0] * n)
            labels.append(list(f["labels"]) + [self.label_pad_token_id] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
