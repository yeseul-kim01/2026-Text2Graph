"""
============================================================
DREEAM Silver Evidence 추론 (scripts/infer_evidence.py)
============================================================
역할: Teacher 모델로 distant data에 대해 token importance를 추론하여
      silver evidence (.attns 파일)를 생성 (Stage 2 Step 2)

사용법:
  python scripts/infer_evidence.py --config configs/stage2.yaml \
      --teacher_checkpoint checkpoints/stage2/teacher/best_model.pt

INPUT:  Teacher 모델 체크포인트, train_distant.json
OUTPUT: train_distant.attns — silver evidence attention 파일

기반 논문: Ma et al. (2023) DREEAM — self-training pipeline

담당: 모델 담당

TODO (팀원 수정 포인트):
  - [ ] Teacher 모델의 attention weight 추출 로직 (DREEAM 논문 Section 3.2)
  - [ ] 대용량 distant data 처리를 위한 chunk/batch 처리
  - [ ] .attns 파일 포맷 정의 및 저장
============================================================
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, load_checkpoint
from src.preprocessing import load_rel2id, create_dataloader
from src.model import DocREModel


def main():
    parser = argparse.ArgumentParser(description="DREEAM Silver Evidence Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    device = torch.device(
        config["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )

    print("[InferEvidence] DREEAM Silver Evidence Generation")
    print(f"  Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"  Distant data: {config['data']['distant_file']}")

    # ── TODO: 팀원이 구현할 부분 ──
    # Step 1: Teacher 모델 로드
    # model = DocREModel(config).to(device)
    # load_checkpoint(model, args.teacher_checkpoint, device=str(device))
    # model.eval()

    # Step 2: Distant data를 순회하며 attention 추출
    # tokenizer = AutoTokenizer.from_pretrained(config["encoder"]["model_name"])
    # rel2id = load_rel2id(config["data"]["meta_dir"])
    # distant_loader = create_dataloader(...)

    # Step 3: 각 문서에 대해 token importance 계산
    # with torch.no_grad():
    #     for batch in tqdm(distant_loader, desc="Inferring evidence"):
    #         # BERT attention weights 추출
    #         # → evidence distribution 계산
    #         # → .attns 딕셔너리에 저장

    # Step 4: .attns 파일로 저장
    # save_path = os.path.join(config["evidence"]["teacher_signal_dir"], "train_distant.attns")
    # torch.save(attns_dict, save_path)

    print("[InferEvidence] TODO: Implement teacher attention extraction")
    print("  참고: https://github.com/YoumiMa/dreeam")
    print("  참고: DREEAM 논문 Section 3.2, Eq. 13")


if __name__ == "__main__":
    main()
