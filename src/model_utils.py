"""Model / tokenizer loading, LoRA wrapping, and trainable-parameter counting."""
import json
import logging
import os
from typing import Dict, Optional

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_dtype():
    """Choose a compute dtype: bf16 if supported, else fp16 on GPU, else fp32."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16"
        return torch.float16, "fp16"
    return torch.float32, "fp32"


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_name: str, dtype: Optional[torch.dtype] = None):
    """Load a causal LM. Pass dtype=torch.float32 for training (AMP handles the
    mixed-precision), or pick_dtype() for fast inference."""
    if dtype is None:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    return model


def load_lora_adapter(base_model_name: str, adapter_dir: str, dtype: Optional[torch.dtype] = None):
    """Load base model + a trained LoRA adapter for evaluation, merged into a plain
    causal LM (no PeftModel wrapper). Eval-only: merging is only valid when the
    adapter will not be trained further, and it also avoids the base(bf16) /
    adapter(fp32) dtype split that PEFT otherwise casts around on every forward."""
    if dtype is None:
        dtype, _ = pick_dtype()
    base = load_model(base_model_name, dtype=dtype)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    return model


def apply_lora(model, lora_cfg: dict):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
    )
    return get_peft_model(model, config)


def count_parameters(model) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (100.0 * trainable / total) if total else 0.0
    return {"total": int(total), "trainable": int(trainable), "trainable_pct": pct}


def log_and_save_param_count(model, out_dir: str, logger: logging.Logger) -> Dict[str, float]:
    stats = count_parameters(model)
    logger.info(
        "Trainable parameters | total=%s | trainable=%s | %.4f%%",
        f"{stats['total']:,}",
        f"{stats['trainable']:,}",
        stats["trainable_pct"],
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "param_count.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats
