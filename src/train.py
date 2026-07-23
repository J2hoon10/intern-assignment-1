"""PubMedQA로 Qwen2.5-0.5B를 학습 — full fine-tuning 또는 LoRA.

기능:
  - 학습 가능 파라미터 수를 로깅 + 저장 (param_count.json)
  - 학습 진행 상황을 logs/<exp>.log, trainer_state.json, train_log.jsonl에 기록
  - 최신 체크포인트에서 자동 재개 (또는 --fresh로 처음부터 시작)
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
    """Trainer의 로그 레코드(loss, lr, eval_loss, ...)를 매번 JSONL 파일에 이어붙인다."""

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
    use_tf32 = torch.cuda.is_available() and use_bf16  # bf16 지원(Ampere 이상)이면 TF32도 지원됨
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
        bf16=use_bf16,                      # bf16 AMP (autocast); fp32 마스터 가중치는 유지
        fp16=use_fp16,
        tf32=use_tf32,                      # Ampere+/Ada에서 matmul을 더 빠르게
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,               # 비슷한 길이끼리 묶음 -> 서로 다른 패딩
                                             # shape 수가 훨씬 줄어듦 -> CUDA allocator
                                             # 파편화 방지 (CausalCollator 참고)
        dataloader_num_workers=0,           # Windows에서는 0이 가장 안전
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

    # 모델 + 토크나이저 (fp32 가중치; 실행 시 혼합정밀도는 bf16 AMP가 처리).
    tokenizer = load_tokenizer(cfg["model_name"])
    model = load_model(cfg["model_name"], dtype=torch.float32)
    if method == "lora":
        model = apply_lora(model, cfg["lora"])
        model.print_trainable_parameters()
    # gradient checkpointing을 쓰려면 use_cache를 꺼야 하고, PEFT는 추가로 input grads를
    # 활성화해야 checkpoint된 그래프를 통해 adapter까지 gradient가 흘러갈 수 있다.
    if bool(cfg["train"].get("gradient_checkpointing", False)):
        model.config.use_cache = False
        if method == "lora":
            model.enable_input_require_grads()
    log_and_save_param_count(model, out_dir, logger)

    # 데이터.
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

    # Trainer가 DataLoader를 만들기 직전에 시드를 다시 설정한다: 모델/adapter
    # 초기화가 전역 RNG를 소비하는 양이 방식마다 다르고(LoRA는 Kaiming-uniform으로
    # adapter를 뽑지만 full_ft는 그렇지 않음), Trainer의 기본 RandomSampler는
    # 설정된 시드가 아니라 torch의 *현재* 전역 RNG 상태로 스스로 시드를 정한다
    # -- 따라서 이 재설정이 없으면 full_ft와 lora가 서로 다른, 통제되지 않은
    # 배치 셔플 순서를 갖게 되고, allocator 파편화는 순서에 민감하다.
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

    # 재개(resume) 처리.
    resume = None
    last_ckpt = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    if last_ckpt and not args.fresh:
        resume = last_ckpt
        logger.info("Resuming from checkpoint: %s", resume)
    elif last_ckpt and args.fresh:
        logger.info("--fresh set: ignoring existing checkpoint %s", last_ckpt)

    train_result = trainer.train(resume_from_checkpoint=resume)
    logger.info("Training finished | metrics=%s", train_result.metrics)

    # evaluate.py가 참조할 고정된 경로에 최종 모델/adapter를 저장.
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
