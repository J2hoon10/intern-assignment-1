"""Shared helpers: project paths, config loading, logging setup, seeding."""
import logging
import os
import random
import sys

import numpy as np
import yaml

# Project root = parent of this file's directory (src/..).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
LOGS_DIR = os.path.join(ROOT, "logs")


def load_config(path: str) -> dict:
    """Load a YAML experiment config."""
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def output_dir_for(experiment: str) -> str:
    d = os.path.join(OUTPUTS_DIR, experiment)
    os.makedirs(d, exist_ok=True)
    return d


def setup_logging(name: str, log_file: str = None) -> logging.Logger:
    """Return a logger that writes to stdout and (optionally) a file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        if not os.path.isabs(log_file):
            log_file = os.path.join(LOGS_DIR, log_file)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
