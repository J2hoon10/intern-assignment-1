"""공통 헬퍼: 프로젝트 경로, config 로딩, 로깅 설정, 시드 고정."""
import logging
import os
import random
import sys

import numpy as np
import yaml

# 프로젝트 루트 = 이 파일이 있는 디렉터리(src)의 상위 디렉터리.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
LOGS_DIR = os.path.join(ROOT, "logs")


def load_config(path: str) -> dict:
    """YAML 실험 config를 로드한다."""
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
    """stdout과 (선택적으로) 파일에 기록하는 로거를 반환한다."""
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
