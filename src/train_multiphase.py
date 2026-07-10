"""Two-phase sequential fine-tuning with 10-fold CV on PQA-L.

Phase 1: train on PQA-A (once).
Phase 2: 10-fold cross-validation on PQA-L. For each fold, continue training from
         the Phase-1 weights on 450 examples and validate on the held-out 50. The
         fold with the best validation Macro-F1 is selected; its model/adapter is
         copied to outputs/<exp>/{final|adapter} so evaluate.py tests the best model.

Weight initialization across phases:
  - full_ft: load the previous phase's full model
  - lora:    continue training the SAME adapter (is_trainable=True)

*** Memory isolation ***: each training unit (Phase 1, and every CV fold) runs in a
FRESH SUBPROCESS. Instantiating transformers.Trainer many times in one process leaks
GPU memory (optimizer state / accelerate singletons are not fully released), which
accumulates and eventually OOMs. Running each unit as its own process guarantees the
GPU is fully freed between units. The parent process does NO GPU work.

Resume-safe: completed phases/folds are skipped (a fold is "done" once its
val_metrics.json exists); interrupted training auto-resumes from the latest
checkpoint. Non-best full_ft fold weights are deleted after selection to save disk.
Reuses build_training_args / JsonlLoggingCallback from train.py.
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
    """Diagnostic: checks memory_reserved() after EVERY optimizer step and logs a
    line only when it changed, together with the padded batch shapes the collator
    produced during that step and which of them were first-seen. If the allocator
    behaves as intended (group_by_length + pad_to_multiple_of=64), reserved should
    grow only on first-seen shapes early in the run and then go silent. Also logs
    reserved/allocated every `every_n_steps` steps with the cumulative distinct-
    shape count, and calls empty_cache() once per epoch to check whether the
    Windows sysmem (Shared GPU Memory) fallback actually shrinks once the cache
    is released. (The full torch.cuda.memory_summary() dump at the first periodic
    probe is disabled -- the one-line probes carry the same signal.)"""

    def __init__(self, logger, every_n_steps: int = 10, collator=None,
                 empty_cache_every_steps: int = 0, empty_cache_min_reserved_gb: float = 12.0):
        self.logger = logger
        self.every_n_steps = every_n_steps
        self.collator = collator
        # empty_cache_every_steps > 0: every N optimizer steps, release cached
        # allocator segments back to the driver -- but only while reserved is
        # above empty_cache_min_reserved_gb. Reserved that stays under the VRAM
        # ceiling is harmless cache; releasing it would just cost re-cudaMalloc.
        # This exists because garbage_collection_threshold in
        # PYTORCH_CUDA_ALLOC_CONF is a no-op unless the process also calls
        # set_per_process_memory_fraction (allocator gates GC on set_fraction),
        # so nothing else ever returns memory mid-epoch.
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
        # Full allocator table dump, disabled on request (huge and repetitive):
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
# Child-process work (one training unit): runs in its own process, then exits.
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
    # Re-seed right before the Trainer builds its DataLoader: model/adapter init
    # (LoRA's 168 Kaiming-uniform draws) consumes the global RNG by a different
    # amount than full_ft, so without this reset, Trainer's plain RandomSampler
    # (which seeds itself from torch's *current* global RNG state, not the
    # configured seed) would hand full_ft and lora different, uncontrolled
    # batch-shuffle orders -- and allocator fragmentation is order-sensitive.
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
    # End-of-unit allocator table dump, disabled on request (same table as the
    # step probe; the per-step [mem-probe] lines carry the signal):
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
    """Child entry point: train (and, for folds, evaluate) one unit, then exit.

    Each unit runs in a fresh subprocess (see module docstring), so the parent's
    set_seed() never reaches it -- without this call the child's RNG (and thus
    its DataLoader shuffle order) was seeded from OS entropy, not spec["seed"],
    making every run non-reproducible regardless of the configured seed.
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
# Parent-process orchestration (no GPU work here).
# ---------------------------------------------------------------------------
def _launch_unit(spec, logger):
    """Run one training unit in a fresh subprocess (GPU fully freed on exit).

    Sets max_split_size_mb to reduce CUDA allocator fragmentation, which otherwise
    causes an OOM mid-training even when total free memory would suffice.
    garbage_collection_threshold frees cached blocks before reserved memory reaches
    the VRAM ceiling; hitting it triggers the Windows driver's silent sysmem
    fallback, which slows training several-fold instead of raising an OOM.
    (expandable_segments is unsupported on Windows.)
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
    ap.add_argument("--_unit", help=argparse.SUPPRESS)   # internal: run one unit in a subprocess
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
