"""Train Qwen2.5-0.5B on PubMedQA — full fine-tuning or LoRA.

Features:
  - trainable-parameter count logged + saved (param_count.json)
  - training progress logged to logs/<exp>.log, trainer_state.json, train_log.jsonl
  - automatic resume from the latest checkpoint (or --fresh to start over)
"""
import argparse
import json
import os

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from common import DATA_DIR, load_config, output_dir_for, set_seed, setup_logging
from data import CausalCollator, build_dataset, read_jsonl
from model_utils import apply_lora, load_model, load_tokenizer, log_and_save_param_count, pick_dtype


class JsonlLoggingCallback(TrainerCallback):
    """Append every Trainer log record (loss, lr, eval_loss, ...) to a JSONL file."""

    def __init__(self, path: str):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        rec = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def build_training_args(cfg, out_dir, logger):
    t = cfg["train"]
    _, dtype_name = pick_dtype()
    use_bf16 = dtype_name == "bf16"
    use_fp16 = dtype_name == "fp16"
    use_gc = bool(t.get("gradient_checkpointing", False))
    use_tf32 = torch.cuda.is_available() and use_bf16  # bf16 support (Ampere+) implies TF32
    logger.info(
        "AMP/memory | dtype=%s (bf16=%s fp16=%s) | gradient_checkpointing=%s | tf32=%s | "
        "micro_bsz=%d grad_accum=%d (effective=%d)",
        dtype_name, use_bf16, use_fp16, use_gc, use_tf32,
        int(t["per_device_train_batch_size"]), int(t["gradient_accumulation_steps"]),
        int(t["per_device_train_batch_size"]) * int(t["gradient_accumulation_steps"]),
    )

    return TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=False,
        num_train_epochs=float(t["num_train_epochs"]),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        warmup_ratio=float(t["warmup_ratio"]),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        logging_steps=int(t["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(t["eval_steps"]),
        save_strategy="steps",
        save_steps=int(t["save_steps"]),
        save_total_limit=int(t["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,                      # bf16 AMP (autocast); fp32 master weights kept
        fp16=use_fp16,
        tf32=use_tf32,                      # faster matmuls on Ampere+/Ada
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,               # cluster similar lengths -> far fewer distinct
                                             # padded shapes -> avoids CUDA allocator
                                             # fragmentation (see CausalCollator)
        dataloader_num_workers=0,           # 0 is safest on Windows
        report_to=["tensorboard"],
        logging_dir=os.path.join(out_dir, "tb"),
        seed=cfg["seed"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints and start over")
    ap.add_argument("--max_train_samples", type=int, default=None, help="cap train size (smoke test)")
    ap.add_argument("--max_steps", type=int, default=None, help="cap training steps (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiment"]
    method = cfg["method"]
    assert method in ("full_ft", "lora"), f"train.py only supports full_ft/lora, got {method}"

    out_dir = output_dir_for(exp)
    logger = setup_logging(f"train.{exp}", f"{exp}.log")
    set_seed(cfg["seed"])
    logger.info("=== TRAIN %s (method=%s) ===", exp, method)
    logger.info("Config: %s", json.dumps(cfg, ensure_ascii=False))

    # Model + tokenizer (fp32 weights; bf16 AMP handles mixed precision at run time).
    tokenizer = load_tokenizer(cfg["model_name"])
    model = load_model(cfg["model_name"], dtype=torch.float32)
    if method == "lora":
        model = apply_lora(model, cfg["lora"])
        model.print_trainable_parameters()
    # Gradient checkpointing needs use_cache off; PEFT additionally needs input grads enabled
    # so gradients can flow back to the adapters through the checkpointed graph.
    if bool(cfg["train"].get("gradient_checkpointing", False)):
        model.config.use_cache = False
        if method == "lora":
            model.enable_input_require_grads()
    log_and_save_param_count(model, out_dir, logger)

    # Data.
    train_rows = read_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    dev_rows = read_jsonl(os.path.join(DATA_DIR, "dev.jsonl"))
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
        logger.info("Capped train set to %d samples (smoke test).", len(train_rows))
    train_ds = build_dataset(tokenizer, train_rows, cfg["max_seq_len"])
    dev_ds = build_dataset(tokenizer, dev_rows, cfg["max_seq_len"])
    logger.info("Datasets | train=%d dev=%d", len(train_ds), len(dev_ds))

    training_args = build_training_args(cfg, out_dir, logger)
    if args.max_steps:
        training_args.max_steps = args.max_steps
        logger.info("Overriding max_steps=%d (smoke test).", args.max_steps)

    # Re-seed right before the Trainer builds its DataLoader: model/adapter init
    # consumes the global RNG by a method-dependent amount (LoRA's Kaiming-uniform
    # adapter draws vs none for full_ft), and Trainer's plain RandomSampler seeds
    # itself from torch's *current* global RNG state rather than the configured
    # seed -- so without this reset, full_ft and lora would get different,
    # uncontrolled batch-shuffle orders, and allocator fragmentation is
    # order-sensitive.
    set_seed(cfg["seed"])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=CausalCollator(tokenizer),
        callbacks=[JsonlLoggingCallback(os.path.join(out_dir, "train_log.jsonl"))],
    )

    # Resume handling.
    resume = None
    last_ckpt = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    if last_ckpt and not args.fresh:
        resume = last_ckpt
        logger.info("Resuming from checkpoint: %s", resume)
    elif last_ckpt and args.fresh:
        logger.info("--fresh set: ignoring existing checkpoint %s", last_ckpt)

    train_result = trainer.train(resume_from_checkpoint=resume)
    logger.info("Training finished | metrics=%s", train_result.metrics)

    # Save final model / adapter to a stable path for evaluate.py.
    if method == "lora":
        adapter_dir = os.path.join(out_dir, "adapter")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        logger.info("Saved LoRA adapter to %s", adapter_dir)
    else:
        final_dir = os.path.join(out_dir, "final")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info("Saved full model to %s", final_dir)

    logger.info("=== TRAIN %s done ===", exp)


if __name__ == "__main__":
    main()
