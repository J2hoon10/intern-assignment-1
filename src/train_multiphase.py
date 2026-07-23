"""PQA-L 10-fold CV를 포함한 2단계 순차 fine-tuning.

Phase 1: PQA-A로 학습 (한 번).
Phase 2: PQA-L에 대한 10-fold 교차검증. 각 fold마다 Phase-1 가중치에서 이어받아
         450개 예시로 학습하고, 남겨둔 50개로 검증한다. 검증 Macro-F1이 가장 좋은
         fold를 선택하고, 그 모델/adapter를 outputs/<exp>/{final|adapter}로
         복사하여 evaluate.py가 그 최고 모델을 테스트하게 한다.

단계 간 가중치 초기화 방식:
  - full_ft: 이전 단계의 전체 모델을 로드
  - lora:    동일한 adapter를 계속 학습 (is_trainable=True)

*** 메모리 격리 ***: 각 학습 유닛(Phase 1, 그리고 각 CV fold)은 새로운
서브프로세스에서 실행된다. 한 프로세스 안에서 transformers.Trainer를 여러 번
생성하면 GPU 메모리가 샌다(옵티마이저 상태 / accelerate 싱글턴이 완전히
해제되지 않음). 이게 누적되면 결국 OOM이 난다. 각 유닛을 독립된 프로세스로
실행하면 유닛 사이에서 GPU가 완전히 해제됨이 보장된다. 부모 프로세스는 GPU
작업을 전혀 하지 않는다.

재개 가능(resume-safe): 완료된 phase/fold는 건너뛴다(val_metrics.json이 있으면
그 fold는 "완료"된 것으로 간주). 중단된 학습은 최신 체크포인트에서 자동 재개된다.
최고 fold가 아닌 full_ft fold의 가중치는 선택 이후 디스크 절약을 위해 삭제된다.
train.py의 build_training_args / JsonlLoggingCallback을 재사용한다.
"""
import argparse
import gc
import glob
import json
import os
import shutil
import subprocess
import sys
from collections import Counter

import torch
from peft import PeftModel
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from common import OUTPUTS_DIR, ROOT, load_config, set_seed, setup_logging
from data import CausalCollator, build_dataset, format_prompt, read_jsonl
from metrics import compute_metrics, predict
from model_utils import (
    apply_lora,
    load_lora_adapter,
    load_model,
    load_tokenizer,
    log_and_save_param_count,
    pick_dtype,
)
from train import JsonlLoggingCallback, build_training_args

PHASE_MERGE_KEYS = ("learning_rate", "num_train_epochs", "eval_steps", "save_steps",
                    "save_total_limit", "warmup_ratio", "weight_decay", "lr_scheduler_type",
                    "per_device_train_batch_size", "gradient_accumulation_steps",
                    "empty_cache_every_steps", "empty_cache_min_reserved_gb")


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def art_of(out_dir, method):
    return os.path.join(out_dir, "adapter" if method == "lora" else "final")


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class CudaMemoryProbeCallback(TrainerCallback):
    """진단용: 매 옵티마이저 스텝마다 memory_reserved()를 확인하고, 값이 바뀌었을
    때만 그 스텝에서 collator가 만든 패딩 배치 shape들과 그중 처음 본 것이
    무엇인지와 함께 한 줄 로그를 남긴다. allocator가 의도대로 동작한다면
    (group_by_length + pad_to_multiple_of=64), reserved는 실행 초반 처음 보는
    shape에서만 늘어나고 그 이후로는 조용해져야 한다. 또한 `every_n_steps`마다
    누적 distinct-shape 개수와 함께 reserved/allocated를 로깅하고, 에폭마다 한 번
    empty_cache()를 호출해서 캐시를 해제했을 때 Windows sysmem(Shared GPU
    Memory) 폴백이 실제로 줄어드는지 확인한다. (첫 주기적 probe에서의 전체
    torch.cuda.memory_summary() 덤프는 비활성화했다 -- 한 줄짜리 probe로도
    같은 정보를 알 수 있다.)"""

    def __init__(self, logger, every_n_steps: int = 10, collator=None,
                 empty_cache_every_steps: int = 0, empty_cache_min_reserved_gb: float = 12.0):
        self.logger = logger
        self.every_n_steps = every_n_steps
        self.collator = collator
        # empty_cache_every_steps > 0: N 옵티마이저 스텝마다 캐시된 allocator
        # 세그먼트를 드라이버로 반환한다 -- 단, reserved가
        # empty_cache_min_reserved_gb를 넘을 때만. VRAM 상한 아래에 머무는
        # reserved는 무해한 캐시이므로, 그걸 해제하면 재-cudaMalloc 비용만 든다.
        # 이게 필요한 이유는 PYTORCH_CUDA_ALLOC_CONF의
        # garbage_collection_threshold가 프로세스가
        # set_per_process_memory_fraction도 호출하지 않는 한 아무 동작도 하지
        # 않기 때문이다(allocator는 GC를 set_fraction에 게이팅함). 그래서 이거
        # 말고는 에폭 도중에 메모리를 반환하는 방법이 없다.
        self.empty_cache_every_steps = empty_cache_every_steps
        self.empty_cache_min_reserved_gb = empty_cache_min_reserved_gb
        self._prev_reserved = None
        self._seen_shapes = set()

    def _drain_shapes(self):
        if self.collator is None:
            return [], []
        shapes = self.collator.shapes_since_probe
        self.collator.shapes_since_probe = []
        new = [s for s in dict.fromkeys(shapes) if s not in self._seen_shapes]
        self._seen_shapes.update(new)
        return shapes, new

    def on_step_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        reserved = torch.cuda.memory_reserved()
        shapes, new_shapes = self._drain_shapes()
        if self._prev_reserved is not None and reserved != self._prev_reserved:
            self.logger.info(
                "[mem-probe] step=%d reserved %.3f->%.3fGB (%+.0fMiB) | shapes this step: %s | first-seen: %s",
                state.global_step, self._prev_reserved / 1e9, reserved / 1e9,
                (reserved - self._prev_reserved) / 2**20,
                dict(Counter(shapes)), new_shapes or "-")
        if (self.empty_cache_every_steps
                and state.global_step % self.empty_cache_every_steps == 0
                and reserved / 1e9 >= self.empty_cache_min_reserved_gb):
            torch.cuda.empty_cache()
            after = torch.cuda.memory_reserved()
            self.logger.info("[mem-probe] step=%d empty_cache(): reserved %.2fGB -> %.2fGB",
                              state.global_step, reserved / 1e9, after / 1e9)
            reserved = after
        self._prev_reserved = reserved
        if state.global_step % self.every_n_steps != 0:
            return
        # 전체 allocator 테이블 덤프는 요청에 따라 비활성화(너무 크고 반복적임):
        # self.logger.info("=== torch.cuda.memory_summary() (step=%d) ===\n%s",
        #                   state.global_step, torch.cuda.memory_summary())
        self.logger.info("[mem-probe] step=%d reserved=%.2fGB allocated=%.2fGB distinct_shapes=%d",
                          state.global_step, reserved / 1e9, torch.cuda.memory_allocated() / 1e9,
                          len(self.collator.shape_counts) if self.collator else -1)

    def on_epoch_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        before = torch.cuda.memory_reserved() / 1e9
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved() / 1e9
        self.logger.info("[mem-probe] epoch=%.3f empty_cache(): reserved %.2fGB -> %.2fGB",
                          state.epoch, before, after)


# ---------------------------------------------------------------------------
# 자식 프로세스 작업(학습 유닛 하나): 자신의 프로세스에서 실행 후 종료된다.
# ---------------------------------------------------------------------------
def load_phase_model(method, model_name, lora_cfg, init_from, logger):
    if method == "full_ft":
        src = init_from if init_from else model_name
        logger.info("full_ft init from: %s", src)
        return load_model(src, dtype=torch.float32)

    base = load_model(model_name, dtype=torch.float32)
    if init_from:
        logger.info("lora continue adapter from: %s", init_from)
        model = PeftModel.from_pretrained(base, init_from, is_trainable=True)
    else:
        logger.info("lora fresh adapter")
        model = apply_lora(base, lora_cfg)
    model.print_trainable_parameters()
    return model


def train_one_phase(method, model_name, max_seq_len, seed, lora_cfg,
                    phase_train, init_from, train_file, dev_file, out_dir, logger):
    tokenizer = load_tokenizer(model_name)
    model = load_phase_model(method, model_name, lora_cfg, init_from, logger)
    if bool(phase_train.get("gradient_checkpointing", False)):
        model.config.use_cache = False
        if method == "lora":
            model.enable_input_require_grads()
    log_and_save_param_count(model, out_dir, logger)

    train_ds = build_dataset(tokenizer, read_jsonl(train_file), max_seq_len)
    dev_ds = build_dataset(tokenizer, read_jsonl(dev_file), max_seq_len)
    logger.info("Datasets | train=%d (%s) dev=%d", len(train_ds), os.path.basename(train_file), len(dev_ds))

    collator = CausalCollator(tokenizer)
    training_args = build_training_args({"train": phase_train, "seed": seed}, out_dir, logger)
    # Trainer가 DataLoader를 만들기 직전에 시드를 다시 설정한다: 모델/adapter
    # 초기화(LoRA의 168개 Kaiming-uniform 추출)가 소비하는 전역 RNG 양이
    # full_ft와 다르므로, 이 재설정이 없으면 Trainer의 기본 RandomSampler(설정된
    # 시드가 아니라 torch의 *현재* 전역 RNG 상태로 스스로 시드를 정함)가
    # full_ft와 lora에 서로 다른, 통제되지 않은 배치 셔플 순서를 주게 된다 --
    # 그리고 allocator 파편화는 순서에 민감하다.
    set_seed(seed)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[JsonlLoggingCallback(os.path.join(out_dir, "train_log.jsonl")),
                   CudaMemoryProbeCallback(
                       logger, collator=collator,
                       empty_cache_every_steps=int(phase_train.get("empty_cache_every_steps", 0)),
                       empty_cache_min_reserved_gb=float(phase_train.get("empty_cache_min_reserved_gb", 12.0)))],
    )
    last_ckpt = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    if last_ckpt:
        logger.info("Resuming from checkpoint: %s", last_ckpt)
    trainer.train(resume_from_checkpoint=last_ckpt)
    # 유닛 종료 시 allocator 테이블 덤프는 요청에 따라 비활성화(step probe와 같은
    # 테이블이며, 스텝별 [mem-probe] 로그로 같은 정보를 알 수 있음):
    # if torch.cuda.is_available():
    #     logger.info("=== torch.cuda.memory_summary() ===\n%s", torch.cuda.memory_summary())

    art = art_of(out_dir, method)
    if method == "lora":
        model.save_pretrained(art)
    else:
        trainer.save_model(art)
    tokenizer.save_pretrained(art)
    logger.info("Saved artifact -> %s", art)
    del trainer, model
    _free_cuda()
    return art


def eval_model_on(method, model_name, art_dir, rows, max_seq_len, batch_size, logger):
    dtype, _ = pick_dtype()
    if method == "full_ft":
        model = load_model(art_dir, dtype=dtype)
    else:
        model = load_lora_adapter(model_name, art_dir, dtype=dtype)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(model_name)
    prompts = [format_prompt(r["question"], r["contexts"]) for r in rows]
    y_true = [r["label"] for r in rows]
    y_pred = predict(model, tokenizer, prompts, max_seq_len=max_seq_len, batch_size=batch_size)
    metrics = compute_metrics(y_true, y_pred)
    del model
    _free_cuda()
    return metrics


def run_unit(spec_path):
    """자식 프로세스 진입점: 유닛 하나를 학습하고(fold라면 평가까지) 종료한다.

    각 유닛은 새 서브프로세스에서 실행되므로(모듈 docstring 참고), 부모의
    set_seed()는 여기까지 전달되지 않는다 -- 이 호출이 없으면 자식의 RNG(따라서
    DataLoader 셔플 순서)가 spec["seed"]가 아니라 OS 엔트로피로 시드되어, 설정된
    시드와 무관하게 매 실행이 재현 불가능해진다.
    """
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    set_seed(spec["seed"])
    logger = setup_logging("train_mp.unit", spec["log_file"])
    logger.info(">>> UNIT %s (%s) init_from=%s", spec["kind"], os.path.relpath(spec["out_dir"], OUTPUTS_DIR),
                spec.get("init_from"))
    train_one_phase(spec["method"], spec["model_name"], spec["max_seq_len"], spec["seed"],
                    spec.get("lora_cfg"), spec["phase_train"], spec.get("init_from"),
                    spec["train_file"], spec["dev_file"], spec["out_dir"], logger)
    if spec["kind"] == "fold":
        m = eval_model_on(spec["method"], spec["model_name"], art_of(spec["out_dir"], spec["method"]),
                          read_jsonl(spec["val_file"]), spec["max_seq_len"], spec["eval_batch_size"], logger)
        with open(spec["metrics_path"], "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
        logger.info("fold val | accuracy=%.4f macro_f1=%.4f", m["accuracy"], m["macro_f1"])


# ---------------------------------------------------------------------------
# 부모 프로세스 오케스트레이션 (여기서는 GPU 작업을 하지 않는다).
# ---------------------------------------------------------------------------
def _launch_unit(spec, logger):
    """학습 유닛 하나를 새 서브프로세스에서 실행한다(종료 시 GPU가 완전히 해제됨).

    max_split_size_mb를 설정해 CUDA allocator 파편화를 줄인다. 파편화가 있으면
    전체 여유 메모리가 충분해도 학습 도중 OOM이 날 수 있다.
    garbage_collection_threshold는 reserved 메모리가 VRAM 상한에 도달하기 전에
    캐시된 블록을 해제한다; 상한에 도달하면 Windows 드라이버의 조용한 sysmem
    폴백이 발동해서 OOM을 내는 대신 학습이 몇 배로 느려진다.
    (expandable_segments는 Windows에서 지원되지 않는다.)
    """
    os.makedirs(spec["out_dir"], exist_ok=True)
    spec_path = os.path.join(spec["out_dir"], "_unit_spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f)
    env = {**os.environ,
           "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256,garbage_collection_threshold:0.7"}
    logger.info("Launching subprocess: %s", os.path.relpath(spec["out_dir"], OUTPUTS_DIR))
    ret = subprocess.run([sys.executable, os.path.abspath(__file__), "--_unit", spec_path], env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"Training subprocess failed (exit {ret.returncode}) for {spec['out_dir']}")


def _merge_phase_train(train_common, phase):
    pt = dict(train_common)
    for k in PHASE_MERGE_KEYS:
        if k in phase:
            pt[k] = phase[k]
    return pt


def _cleanup_fold_weights(fold_dir, logger):
    removed = 0
    for d in [os.path.join(fold_dir, "final")] + glob.glob(os.path.join(fold_dir, "checkpoint-*")):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    if removed:
        logger.info("Cleaned %d weight dir(s) in %s (disk)", removed, os.path.basename(fold_dir))


def run_cv_phase(phase, exp, method, model_name, max_seq_len, seed, lora_cfg,
                 train_common, init_from, eval_bs, log_file, logger, force):
    n = int(phase["cv_folds"])
    phase_root = os.path.join(OUTPUTS_DIR, exp, phase["name"])
    phase_train = _merge_phase_train(train_common, phase)

    results = []  # (fold, val_macro_f1, art_dir)
    for k in range(n):
        fold_out = os.path.join(phase_root, f"fold{k}")
        art = art_of(fold_out, method)
        metrics_path = os.path.join(fold_out, "val_metrics.json")

        if os.path.exists(metrics_path) and not force:
            m = json.load(open(metrics_path, encoding="utf-8"))
            logger.info("fold%d already complete | val macro_f1=%.4f (skip)", k, m["macro_f1"])
        else:
            logger.info("=== CV fold %d/%d ===", k, n - 1)
            spec = {
                "kind": "fold", "method": method, "model_name": model_name,
                "max_seq_len": max_seq_len, "seed": seed, "lora_cfg": lora_cfg,
                "phase_train": phase_train, "init_from": init_from,
                "train_file": resolve(phase["fold_train_pattern"].format(k=k)),
                "dev_file": resolve(phase["fold_val_pattern"].format(k=k)),
                "val_file": resolve(phase["fold_val_pattern"].format(k=k)),
                "eval_batch_size": eval_bs, "metrics_path": metrics_path,
                "out_dir": fold_out, "log_file": log_file,
            }
            _launch_unit(spec, logger)
            m = json.load(open(metrics_path, encoding="utf-8"))
        results.append((k, m["macro_f1"], art))

    best_k, best_f1, best_art = max(results, key=lambda x: x[1])
    logger.info("CV per-fold val macro_f1: %s", {k: round(f, 4) for k, f, _ in results})
    logger.info("=== Best fold = fold%d (val macro_f1=%.4f) ===", best_k, best_f1)
    with open(os.path.join(phase_root, "cv_selection.json"), "w", encoding="utf-8") as f:
        json.dump({"folds": [{"fold": k, "val_macro_f1": f} for k, f, _ in results],
                   "best_fold": best_k, "best_val_macro_f1": best_f1}, f, indent=2)

    if method == "full_ft":
        for k, _, _ in results:
            if k != best_k:
                _cleanup_fold_weights(os.path.join(phase_root, f"fold{k}"), logger)

    if not os.path.isdir(best_art):
        raise RuntimeError(f"Best fold artifact missing: {best_art}")
    return best_art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--force", action="store_true", help="retrain even completed phases/folds")
    ap.add_argument("--_unit", help=argparse.SUPPRESS)   # 내부용: 서브프로세스에서 유닛 하나 실행
    args = ap.parse_args()

    if args._unit:
        run_unit(args._unit)
        return

    assert args.config, "--config is required"
    cfg = load_config(args.config)
    exp, method = cfg["experiment"], cfg["method"]
    model_name, max_seq_len, seed = cfg["model_name"], cfg["max_seq_len"], cfg["seed"]
    lora_cfg = cfg.get("lora")
    train_common = cfg["train_common"]
    eval_bs = cfg["eval"]["batch_size"]
    log_file = f"{exp}.log"

    logger = setup_logging(f"train_mp.{exp}", log_file)
    set_seed(seed)
    logger.info("=== TWO-PHASE TRAIN %s (method=%s) | subprocess-isolated ===", exp, method)
    logger.info("Config: %s", json.dumps(cfg, ensure_ascii=False))

    prev_art = None
    for phase in cfg["phases"]:
        name = phase["name"]
        if "cv_folds" in phase:
            prev_art = run_cv_phase(phase, exp, method, model_name, max_seq_len, seed, lora_cfg,
                                    train_common, prev_art, eval_bs, log_file, logger, args.force)
            continue

        out_dir = os.path.join(OUTPUTS_DIR, exp, name)
        art = art_of(out_dir, method)
        if os.path.isdir(art) and not args.force:
            logger.info("Phase %s already complete (%s) - skipping.", name, art)
            prev_art = art
            continue

        spec = {
            "kind": "phase", "method": method, "model_name": model_name,
            "max_seq_len": max_seq_len, "seed": seed, "lora_cfg": lora_cfg,
            "phase_train": _merge_phase_train(train_common, phase), "init_from": prev_art,
            "train_file": resolve(phase["train_file"]), "dev_file": resolve(phase["dev_file"]),
            "out_dir": out_dir, "log_file": log_file,
        }
        logger.info("=== PHASE %s | init_from=%s ===", name, prev_art)
        _launch_unit(spec, logger)
        prev_art = art

    dest = os.path.join(OUTPUTS_DIR, exp, "adapter" if method == "lora" else "final")
    if prev_art and os.path.abspath(prev_art) != os.path.abspath(dest):
        if os.path.isdir(dest):
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(prev_art, dest)
        logger.info("Best-model artifact copied %s -> %s (evaluate.py target)", prev_art, dest)
    logger.info("=== TWO-PHASE TRAIN %s done ===", exp)


if __name__ == "__main__":
    main()
