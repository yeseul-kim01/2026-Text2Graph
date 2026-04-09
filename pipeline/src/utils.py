"""
============================================================
유틸리티 함수 (utils.py)
============================================================
역할: 학습/평가에 필요한 공통 유틸리티

담당: 전체 (공용)

TODO (김예슬):
  - [완] WandB 로깅 통합
  - [완] 학습 재시작(resume) 기능
============================================================
"""

import os
import yaml
import json
import random
import torch
import numpy as np
from typing import Dict


def load_config(config_path: str) -> Dict:
    """
    YAML config 파일 로드.
    INPUT:  config 파일 경로
    OUTPUT: Dict
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"[Utils] Config loaded from {config_path}")
    return config


def set_seed(seed: int):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, path, extra=None):
    """모델 체크포인트 저장"""
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra:
        state.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[Utils] Checkpoint saved: {path}")


def load_checkpoint(model, path, optimizer=None, device="cpu"):
    """모델 체크포인트 로드"""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    print(f"[Utils] Checkpoint loaded: {path}")
    return state.get("epoch", 0)


def count_parameters(model) -> int:
    """학습 가능한 파라미터 수 카운트"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
