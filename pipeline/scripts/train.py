"""
============================================================
학습 스크립트 (scripts/train.py)
============================================================
역할: Stage 1/2/3 학습을 통합 실행하는 메인 스크립트

사용법:
  python scripts/train.py --config configs/stage1.yaml
  python scripts/train.py --config configs/stage2.yaml
  python scripts/train.py --config configs/stage3.yaml

Colab:
  !python scripts/train.py --config configs/stage1.yaml

INPUT:  config YAML 파일
OUTPUT: checkpoints, logs, 평가 결과

담당: 전체 (공용 실행 스크립트)

TODO (수정 포인트):
  - [ ] WandB 로깅 통합
  - [ ] Mixed precision training (torch.cuda.amp)
  - [ ] 멀티 GPU 지원 (DistributedDataParallel)
  - [ ] dev set 평가 루프 완성
============================================================
"""

import os
import sys
import argparse
from tqdm import tqdm

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# ── 프로젝트 루트를 path에 추가 (Colab 호환) ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, save_checkpoint, load_checkpoint, count_parameters
from src.preprocessing import DocREDDataset, docred_collate_fn, load_rel2id, create_dataloader
from src.model import DocREModel
from src.losses import compute_loss
from src.evaluation import evaluate_re


def train_one_epoch(model, dataloader, optimizer, scheduler, config, device):
    """
    1 epoch 학습 수행.

    INPUT:
      - model      : DocREModel
      - dataloader  : 학습 DataLoader
      - optimizer   : AdamW
      - scheduler   : linear warmup scheduler
      - config      : 설정 Dict
      - device      : torch.device
    OUTPUT:
      - avg_loss : float — 평균 loss
    """
    model.train()
    total_loss = 0
    num_steps = 0
    loss_cfg = config["training"]
    evi_cfg = config.get("evidence", {})

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)

        # ── Forward ──
        all_outputs = model(batch)

        # ── Loss 계산 (batch 내 문서별) ──
        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_count = 0

        for b, outputs in enumerate(all_outputs):
            labels = batch["labels"][b].to(device)
            if labels.size(0) == 0:
                continue

            loss_dict = compute_loss(
                outputs=outputs,
                labels=labels,
                evidence_labels=batch["evidence_labels"][b],
                num_sents=batch["num_sents"][b],
                loss_type=loss_cfg.get("loss_type", "bce"),
                lambda_evidence=evi_cfg.get("lambda_evidence", 0.0),
                no_relation_weight=loss_cfg.get("no_relation_weight", 0.1),
                num_relations=config["data"]["num_relations"],
            )
            batch_loss = batch_loss + loss_dict["total_loss"]
            valid_count += 1

        if valid_count == 0:
            continue

        # ── Backward ──
        batch_loss = batch_loss / valid_count
        batch_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), loss_cfg.get("max_grad_norm", 1.0)
        )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += batch_loss.item()
        num_steps += 1

    avg_loss = total_loss / max(num_steps, 1)
    return avg_loss


@torch.no_grad()
def evaluate_on_dev(model, dataloader, config, device):
    """
    Dev set 평가. (간이 버전 — 팀원이 확장)

    INPUT:  model, dev dataloader, config, device
    OUTPUT: Dict with loss, prediction count
    """
    model.eval()
    total_loss = 0
    total_preds = 0
    loss_cfg = config["training"]
    evi_cfg = config.get("evidence", {})

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)

        all_outputs = model(batch)

        for b, outputs in enumerate(all_outputs):
            labels = batch["labels"][b].to(device)
            if labels.size(0) == 0:
                continue

            loss_dict = compute_loss(
                outputs=outputs,
                labels=labels,
                evidence_labels=batch["evidence_labels"][b],
                num_sents=batch["num_sents"][b],
                loss_type=loss_cfg.get("loss_type", "bce"),
                lambda_evidence=evi_cfg.get("lambda_evidence", 0.0),
                no_relation_weight=loss_cfg.get("no_relation_weight", 0.1),
                num_relations=config["data"]["num_relations"],
            )
            total_loss += loss_dict["total_loss"].item()

            # 예측 수 카운트
            if "relation_logits" in outputs:
                preds = torch.sigmoid(outputs["relation_logits"])
                total_preds += (preds > 0.5).sum().item()

    return {
        "dev_loss": total_loss / max(len(dataloader), 1),
        "num_predictions": total_preds,
    }


def main():
    parser = argparse.ArgumentParser(description="DocRE Training Script")
    parser.add_argument("--config", type=str, required=True, help="Config YAML 파일 경로")
    args = parser.parse_args()

    # ── Config 로드 ──
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    device = torch.device(
        config["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"[Train] Device: {device}")
    print(f"[Train] Stage: {config['experiment']['stage']}")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(config["encoder"]["model_name"])

    # ── Relation 매핑 로드 ──
    rel2id = load_rel2id(config["data"]["meta_dir"], config["data"]["rel2id_file"])

    # ── DataLoader 생성 ──
    train_loader = create_dataloader(
        data_dir=config["data"]["data_dir"],
        data_file=config["data"]["train_file"],
        tokenizer=tokenizer,
        rel2id=rel2id,
        max_seq_len=config["data"]["max_seq_length"],
        batch_size=config["training"]["finetune_batch_size"],
        shuffle=True,
        stage=config["experiment"]["stage"],
    )

    dev_loader = create_dataloader(
        data_dir=config["data"]["data_dir"],
        data_file=config["data"]["dev_file"],
        tokenizer=tokenizer,
        rel2id=rel2id,
        max_seq_len=config["data"]["max_seq_length"],
        batch_size=config["training"]["finetune_batch_size"],
        shuffle=False,
        stage=config["experiment"]["stage"],
    )

    # ── 모델 생성 ──
    model = DocREModel(config).to(device)
    print(f"[Train] Trainable parameters: {count_parameters(model):,}")

    # ── Stage 3: Stage 2 체크포인트 로드 ──
    if config["training"].get("load_checkpoint"):
        ckpt_path = config["training"]["load_checkpoint"]
        if os.path.exists(ckpt_path):
            load_checkpoint(model, ckpt_path, device=str(device))
            print(f"[Train] Loaded Stage 2 checkpoint from {ckpt_path}")

    # ── Optimizer (Encoder vs 나머지 분리 LR) ──
    train_cfg = config["training"]
    param_groups = [
        {"params": model.get_encoder_params(), "lr": train_cfg["encoder_lr"]},
        {"params": model.get_non_encoder_params(), "lr": train_cfg["classifier_lr"]},
    ]
    optimizer = AdamW(param_groups, weight_decay=train_cfg.get("weight_decay", 0.0))

    # ── Scheduler ──
    num_epochs = train_cfg["finetune_epochs"]
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.06))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── 학습 루프 ──
    best_f1 = 0.0
    save_dir = config["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print(f"[Train] Starting training for {num_epochs} epochs...")
    print(f"[Train] Total steps: {total_steps}, Warmup: {warmup_steps}")

    for epoch in range(1, num_epochs + 1):
        # ── Train ──
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, config, device
        )
        print(f"\n[Epoch {epoch}/{num_epochs}] Train Loss: {avg_loss:.4f}")

        # ── Dev Eval ──
        dev_metrics = evaluate_on_dev(model, dev_loader, config, device)
        print(f"  Dev Loss: {dev_metrics['dev_loss']:.4f}")
        print(f"  Dev Predictions: {dev_metrics['num_predictions']}")

        # ── 체크포인트 저장 ──
        save_checkpoint(
            model, optimizer, epoch,
            os.path.join(save_dir, f"model_epoch{epoch}.pt"),
            extra={"dev_metrics": dev_metrics},
        )

    # ── 최종 모델 저장 ──
    save_checkpoint(
        model, optimizer, num_epochs,
        os.path.join(save_dir, "best_model.pt"),
    )
    print("\n[Train] Training complete!")
    print(f"[Train] Best model saved to {save_dir}/best_model.pt")


if __name__ == "__main__":
    main()
