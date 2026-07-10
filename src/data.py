"""Prompt formatting, tokenization (loss on the answer only), and collation.

The task is framed as causal LM: the model reads a prompt ending in
``Answer (yes/no/maybe):`` and must produce one word (`yes`/`no`/`maybe`).
During training we mask the prompt tokens (-100) so the loss is computed on
the answer tokens only.
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
    """PubMedQA `context` is a list of abstract paragraphs; join into one string."""
    if isinstance(contexts, str):
        return contexts.strip()
    return " ".join(c.strip() for c in contexts if c and c.strip())


def format_prompt(question: str, contexts) -> str:
    return PROMPT_TEMPLATE.format(context=join_contexts(contexts), question=str(question).strip())


# ---------------------------------------------------------------------------
# JSONL helpers
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
# Tokenization for training
# ---------------------------------------------------------------------------
def tokenize_example(tokenizer, question, contexts, label: str, max_seq_len: int) -> Dict[str, List[int]]:
    """Build input_ids / attention_mask / labels for one training example.

    Prompt tokens are masked with -100 so that loss is only on the answer.
    If the sequence is too long, the prompt is left-truncated so the question
    (at the end of the prompt) and the answer are always preserved.
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
        prompt_ids = prompt_ids[-max_prompt_len:]  # keep the tail (question + answer cue)

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class ListDataset(TorchDataset):
    """Minimal map-style dataset backed by a list of feature dicts.

    We deliberately avoid ``datasets.Dataset`` (pyarrow) here: on Windows, pyarrow
    and accelerate/torch bring conflicting native runtimes and importing both in the
    same process segfaults. A plain torch Dataset sidesteps pyarrow entirely and
    works directly with transformers.Trainer + our collator.
    """

    def __init__(self, items: List[Dict[str, List[int]]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.items[idx]


def build_dataset(tokenizer, rows: List[dict], max_seq_len: int) -> "ListDataset":
    """Tokenize a list of {question, contexts, label} rows into a torch Dataset."""
    features = [
        tokenize_example(tokenizer, r["question"], r["contexts"], r["label"], max_seq_len)
        for r in rows
    ]
    return ListDataset(features)


# ---------------------------------------------------------------------------
# Collator: dynamic padding of input_ids / attention_mask / labels
# ---------------------------------------------------------------------------
@dataclass
class CausalCollator:
    """Pads each batch's length up to a fixed multiple (not just the batch's own
    max) so that, combined with group_by_length in TrainingArguments, training
    sees only a small, recurring set of padded shapes instead of a new shape
    almost every step. On this Windows/WDDM box that shape churn fragmented the
    CUDA caching allocator until reserved memory exceeded the 16GB card and
    training crashed with a fatal CUBLAS error (not a catchable OOM)."""

    tokenizer: Any
    label_pad_token_id: int = -100
    pad_to_multiple_of: int = 64
    # Diagnostics (read by CudaMemoryProbeCallback): every padded (batch, seq_len)
    # shape emitted since the probe last drained, plus a cumulative histogram.
    # Train and eval batches both land here; the probe labels them by batch size.
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
