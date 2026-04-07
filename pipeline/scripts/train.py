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
  - [ ] 멀티 GPU 지원 (DistributedDataParallel)
============================================================
"""

import copy
import math
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
    노트북 Cell 11 기반: AMP(Mixed Precision) + Gradient Accumulation 적용.

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
    grad_accum = loss_cfg.get("finetune_gradient_accumulation", 1)

    # AMP: CUDA 환경에서만 활성화 (노트북 Cell 11과 동일)
    use_amp = (str(device) != "cpu")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)

        # ── Forward (AMP autocast) ──
        with torch.cuda.amp.autocast(enabled=use_amp):
            all_outputs = model(batch)

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

        # gradient accumulation: 유효 배치 수로 나눔
        batch_loss = batch_loss / valid_count / grad_accum
        scaler.scale(batch_loss).backward()

        total_loss += batch_loss.item() * grad_accum
        num_steps += 1

        # ── Gradient Accumulation: N step마다 업데이트 ──
        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), loss_cfg.get("max_grad_norm", 1.0)
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    # ── 남은 gradient flush (마지막 불완전 accumulation) ──
    if num_steps % grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), loss_cfg.get("max_grad_norm", 1.0)
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(num_steps, 1)
    return avg_loss


@torch.no_grad()
def evaluate_on_dev(model, dataloader, config, device):
    """
    Dev set 평가 — 실제 Micro F1 계산.
    노트북 Cell 9(micro_f1)와 Cell 12(evaluate) 기반으로 재구현.

    INPUT:  model, dev dataloader, config, device
    OUTPUT: Dict with F1, precision, recall
    """
    model.eval()
    threshold = config["relation_head"].get("fixed_threshold", 0.5)

    # DocRED 표준: 'Na'(no_relation, index 0) 제외하고 F1 계산
    tp = fp = fn = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)

        all_outputs = model(batch)

        for b, outputs in enumerate(all_outputs):
            labels = batch["labels"][b].to(device)   # [num_pairs, num_relations]
            if labels.size(0) == 0:
                continue

            logits = outputs["relation_logits"]       # [num_pairs, num_relations]
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # index 0('Na') 제외: [num_pairs, 96]
            preds_pos = preds[:, 1:]
            labels_pos = labels[:, 1:]

            tp += int(((preds_pos == 1) & (labels_pos == 1)).sum())
            fp += int(((preds_pos == 1) & (labels_pos == 0)).sum())
            fn += int(((preds_pos == 0) & (labels_pos == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "F1"       : round(f1 * 100, 2),
        "precision": round(precision * 100, 2),
        "recall"   : round(recall * 100, 2),
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
        {"params": model.get_encoder_params(),     "lr": train_cfg["encoder_lr"]},
        {"params": model.get_non_encoder_params(), "lr": train_cfg["classifier_lr"]},
    ]
    optimizer = AdamW(param_groups, weight_decay=train_cfg.get("weight_decay", 0.0))

    # ── Scheduler ──
    # gradient accumulation 반영: 실제 업데이트 step 수로 계산 (노트북 Cell 13과 동일)
    num_epochs   = train_cfg["finetune_epochs"]
    grad_accum   = train_cfg.get("finetune_gradient_accumulation", 1)
    total_steps  = math.ceil(len(train_loader) / grad_accum) * num_epochs
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.06))
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"[Train] Total steps: {total_steps:,}  |  Warmup: {warmup_steps:,}")

    # ── 학습 루프 (Early Stopping + Best F1 저장) ──
    # 노트북 Cell 13의 early stopping / best model 로직 반영
    best_f1      = 0.0
    patience_counter = 0
    early_stopping_patience = train_cfg.get("early_stopping_patience", 15)
    save_dir     = config["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print(f"[Train] Starting training for {num_epochs} epochs  "
          f"(early stopping patience={early_stopping_patience})")
    print("=" * 65)

    for epoch in range(1, num_epochs + 1):
        # ── Train ──
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, config, device
        )

        # ── Dev F1 평가 ──
        dev_metrics = evaluate_on_dev(model, dev_loader, config, device)

        print(f"[Ep {epoch:02d}/{num_epochs}] "
              f"Loss={avg_loss:.4f} | "
              f"Dev F1={dev_metrics['F1']:.2f}%  "
              f"(P={dev_metrics['precision']:.2f}% / R={dev_metrics['recall']:.2f}%)")

        # ── Best model 저장 (F1 기준) ──
        if dev_metrics["F1"] > best_f1:
            best_f1 = dev_metrics["F1"]
            patience_counter = 0

            # 전체 모델 저장
            save_checkpoint(
                model, optimizer, epoch,
                os.path.join(save_dir, "best_model.pt"),
                extra={"dev_metrics": dev_metrics},
            )
            # BERT encoder 가중치 별도 저장 (Stage 2 초기화용, 노트북 Cell 14 참조)
            torch.save(
                model.encoder.bert.state_dict(),
                os.path.join(save_dir, "best_bert_encoder_weights.pt"),
            )
            print(f"  → Best model saved (F1={best_f1:.2f}%)")
        else:
            patience_counter += 1
            print(f"  (patience {patience_counter}/{early_stopping_patience})")
            if patience_counter >= early_stopping_patience:
                print(f"\n[Train] Early stopping at epoch {epoch}.")
                break

    print(f"\n[Train] Training complete! Best Dev F1: {best_f1:.2f}%")
    print(f"[Train] Best model saved to {save_dir}/best_model.pt")


if __name__ == "__main__":
    main()
