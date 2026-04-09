"""
============================================================
학습 스크립트 (scripts/train.py)
============================================================
역할: Stage 1/2/3 학습을 통합 실행하는 메인 스크립트
      2단계 학습: Distant Pre-training → Annotated Fine-tuning

사용법:
  # Phase 1 + Phase 2 모두 실행 (처음부터)
  python scripts/train.py --config configs/stage1.yaml

  # Phase 2만 실행 (이미 pre-train 완료 시)
  python scripts/train.py --config configs/stage1.yaml --skip_pretrain

  # Phase 1만 실행 (pre-train만)
  python scripts/train.py --config configs/stage1.yaml --pretrain_only

Colab:
  !python scripts/train.py --config configs/stage1.yaml

INPUT:  config YAML 파일
OUTPUT: checkpoints, logs, 평가 결과

담당: 전체 (공용 실행 스크립트)

TODO (김예슬):
  - [ ] WandB 로깅 통합
  - [ ] 멀티 GPU 지원 (DistributedDataParallel)

TODO (박재윤):
    fix: evaluate_on_dev에서 Adaptive Threshold 적용

    - 기존: 고정 threshold 0.5로 relation 채택 여부 판단 (Stage 1 방식)
    - 수정: model이 학습한 threshold_logits와 비교하여 판단 (ATLOP 방식)
    - 원인: 학습은 adaptive threshold로 했는데 평가는 고정 0.5를 사용하여
            학습된 threshold가 무시되고 있었음
    - 파일: scripts/train.py → evaluate_on_dev()
    
TODO (김예슬 - 완료):
    feat: main()에 2단계 학습 (Distant Pre-training → Annotated Fine-tuning) 추가

    - 기존: train_annotated.json (3K)만으로 학습 → F1 ~35% 한계
    - 수정: Phase 1에서 train_distant.json (101K)로 pre-training 후
            Phase 2에서 train_annotated.json (3K)로 fine-tuning
    - 근거: ATLOP (Zhou 2021) Table 1에서 distant pre-train으로 ~10%p F1 향상.
            DREEAM, GAIN, SSAN 등 모든 SOTA 모델이 이 전략 사용.
    - 추가 함수: create_optimizer_and_scheduler(), run_phase()
    - 추가 플래그: --skip_pretrain, --pretrain_only
    - 파일: scripts/train.py → main()
    - train_one_epoch(), evaluate_on_dev()는 팀원 코드 그대로 유지
    

수정 이력:
  [DONE] v1 — 김예슬: 기본 학습 루프 구현
  [DONE] v2 — 박재윤: evaluate_on_dev adaptive threshold 지원
  [DONE] v3 — 김예슬: main()에 2단계 학습 추가
    - Phase 1: distant data (101K)로 pre-training
    - Phase 2: annotated data (3K)로 fine-tuning
    - --skip_pretrain / --pretrain_only 플래그 추가
    - train_one_epoch(), evaluate_on_dev()는 팀원 코드 그대로 유지
    
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


# ══════════════════════════════════════════════════════════════
# train_one_epoch — 팀원 원본 그대로 (이수민 v1)
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# evaluate_on_dev — 팀원 원본 그대로 (박재윤 v2: adaptive threshold)
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_on_dev(model, dataloader, config, device):
    model.eval()

    tp = fp = fn = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)

        all_outputs = model(batch)

        for b, outputs in enumerate(all_outputs):
            labels = batch["labels"][b].to(device)
            if labels.size(0) == 0:
                continue

            logits = outputs["relation_logits"]

            # ── 박재윤 수정: adaptive threshold 지원 ──
            if "threshold_logits" in outputs:
                # Adaptive Threshold: 모델이 학습한 기준으로 판단
                th = outputs["threshold_logits"]  # [num_pairs, 1]
                preds = (logits > th).float()
            else:
                # 고정 threshold (Stage 1 호환)
                threshold = config["relation_head"].get("fixed_threshold", 0.5)
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()

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


# ══════════════════════════════════════════════════════════════
# [v3 추가 — 김예슬] 2단계 학습용 헬퍼 함수들
# ══════════════════════════════════════════════════════════════

def create_optimizer_and_scheduler(model, config, num_train_steps):
    """
    Optimizer + Scheduler 생성 헬퍼.
    Phase 1과 Phase 2에서 각각 새로 생성해야 함.

    ── 왜 Phase마다 새로 만드는가? ──
    Phase 1 (distant 101K)과 Phase 2 (annotated 3K)는
    데이터 크기가 33배 차이. Scheduler의 warmup/decay가
    total_steps에 맞춰져야 lr이 올바르게 조절됨.
    ATLOP 원본도 pre-train과 fine-tune에서 별도 optimizer 사용.
    """
    train_cfg = config["training"]
    param_groups = [
        {"params": model.get_encoder_params(),     "lr": train_cfg["encoder_lr"]},
        {"params": model.get_non_encoder_params(), "lr": train_cfg["classifier_lr"]},
    ]
    optimizer = AdamW(param_groups, weight_decay=train_cfg.get("weight_decay", 0.0))
    warmup_steps = int(num_train_steps * train_cfg.get("warmup_ratio", 0.06))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_train_steps)
    return optimizer, scheduler


def run_phase(
    phase_name, model, train_loader, dev_loader, config, device,
    num_epochs, save_dir, phase_label="Phase",
):
    """
    하나의 학습 Phase를 실행하는 통합 함수.
    Phase 1 (Pre-training)과 Phase 2 (Fine-tuning) 모두 이 함수를 사용.

    내부적으로 train_one_epoch(), evaluate_on_dev()를 호출하므로
    실제 학습/평가 로직은 팀원 코드 그대로 사용됨.
    """
    train_cfg = config["training"]
    grad_accum = train_cfg.get("finetune_gradient_accumulation", 1)

    # ── Optimizer / Scheduler 생성 ──
    total_steps = math.ceil(len(train_loader) / grad_accum) * num_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, total_steps)
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.06))

    print(f"\n{'=' * 65}")
    print(f"[{phase_label}] {phase_name}")
    print(f"  Data: {len(train_loader.dataset):,} documents")
    print(f"  Epochs: {num_epochs}")
    print(f"  Total steps: {total_steps:,}  |  Warmup: {warmup_steps:,}")
    print(f"  Save dir: {save_dir}")
    print(f"{'=' * 65}")

    os.makedirs(save_dir, exist_ok=True)
    best_f1 = 0.0
    patience_counter = 0
    early_stopping_patience = train_cfg.get("early_stopping_patience", 15)

    for epoch in range(1, num_epochs + 1):
        # ── Train (팀원 함수 그대로 호출) ──
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, config, device
        )

        # ── Dev F1 평가 (팀원 함수 그대로 호출) ──
        dev_metrics = evaluate_on_dev(model, dev_loader, config, device)

        print(f"[{phase_label} Ep {epoch:02d}/{num_epochs}] "
              f"Loss={avg_loss:.4f} | "
              f"Dev F1={dev_metrics['F1']:.2f}%  "
              f"(P={dev_metrics['precision']:.2f}% / R={dev_metrics['recall']:.2f}%)")

        # ── Best model 저장 ──
        if dev_metrics["F1"] > best_f1:
            best_f1 = dev_metrics["F1"]
            patience_counter = 0

            save_checkpoint(
                model, optimizer, epoch,
                os.path.join(save_dir, "best_model.pt"),
                extra={"dev_metrics": dev_metrics, "phase": phase_name},
            )
            torch.save(
                model.encoder.bert.state_dict(),
                os.path.join(save_dir, "best_bert_encoder_weights.pt"),
            )
            print(f"  → Best model saved (F1={best_f1:.2f}%)")
        else:
            patience_counter += 1
            print(f"  (patience {patience_counter}/{early_stopping_patience})")
            if patience_counter >= early_stopping_patience:
                print(f"\n[{phase_label}] Early stopping at epoch {epoch}.")
                break

    print(f"\n[{phase_label}] {phase_name} complete! Best Dev F1: {best_f1:.2f}%")
    return best_f1


# ══════════════════════════════════════════════════════════════
# [v3 수정 — 김예슬] main() — 2단계 학습 구조
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DocRE Training Script")
    parser.add_argument("--config", type=str, required=True, help="Config YAML 파일 경로")
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Phase 1 (distant pre-training) 건너뛰기")
    parser.add_argument("--pretrain_only", action="store_true",
                        help="Phase 1만 실행하고 종료")
    args = parser.parse_args()

    # ── Config 로드 ──
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    device = torch.device(
        config["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )
    train_cfg = config["training"]

    print(f"[Train] Device: {device}")
    print(f"[Train] Stage: {config['experiment']['stage']}")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(config["encoder"]["model_name"])

    # ── Relation 매핑 로드 ──
    rel2id = load_rel2id(config["data"]["meta_dir"], config["data"]["rel2id_file"])

    # ── Dev DataLoader (Phase 1, 2 공통) ──
    dev_loader = create_dataloader(
        data_dir=config["data"]["data_dir"],
        data_file=config["data"]["dev_file"],
        tokenizer=tokenizer,
        rel2id=rel2id,
        max_seq_len=config["data"]["max_seq_length"],
        batch_size=train_cfg["finetune_batch_size"],
        shuffle=False,
        stage=config["experiment"]["stage"],
    )

    # ── 모델 생성 ──
    model = DocREModel(config).to(device)
    print(f"[Train] Trainable parameters: {count_parameters(model):,}")

    # ── 기존 체크포인트 로드 (Stage 전환 시) ──
    if train_cfg.get("load_checkpoint"):
        ckpt_path = train_cfg["load_checkpoint"]
        if os.path.exists(ckpt_path):
            load_checkpoint(model, ckpt_path, device=str(device))
            print(f"[Train] Loaded checkpoint from {ckpt_path}")

    save_dir = config["output"]["save_dir"]

    # ══════════════════════════════════════════════════════════
    # Phase 1: Distant Pre-training (101K 문서)
    #
    # 논문 근거:
    #   ATLOP (Zhou 2021): distant pre-train으로 약 10%p F1 향상
    #   DREEAM (Ma 2023): 동일 전략 사용
    #   DocRED 벤치마크 모든 SOTA 모델이 이 전략 사용
    #
    # 주의: distant data 전처리에 10~30분 소요 (Colab 기준)
    # ══════════════════════════════════════════════════════════
    pretrain_epochs = train_cfg.get("pretrain_epochs", 0)

    if pretrain_epochs > 0 and not args.skip_pretrain:
        print("\n" + "=" * 65)
        print("[Phase 1] Distant Pre-training 준비 중...")
        print(f"  데이터: {config['data']['distant_file']} (101K 문서)")
        print(f"  Epochs: {pretrain_epochs}")
        print("  ※ 전처리에 시간이 걸릴 수 있습니다 (Colab: 10~30분)")
        print("=" * 65)

        # ── Distant DataLoader 생성 ──
        distant_loader = create_dataloader(
            data_dir=config["data"]["data_dir"],
            data_file=config["data"]["distant_file"],
            tokenizer=tokenizer,
            rel2id=rel2id,
            max_seq_len=config["data"]["max_seq_length"],
            batch_size=train_cfg.get("pretrain_batch_size", 4),
            shuffle=True,
            stage=config["experiment"]["stage"],
        )

        # ── Phase 1 실행 ──
        pretrain_save_dir = os.path.join(save_dir, "pretrain")
        phase1_f1 = run_phase(
            phase_name="Pre-training (Distant)",
            model=model,
            train_loader=distant_loader,
            dev_loader=dev_loader,
            config=config,
            device=device,
            num_epochs=pretrain_epochs,
            save_dir=pretrain_save_dir,
            phase_label="Phase 1",
        )

        # ── Pre-trained best model 로드 → Phase 2 시작점 ──
        pretrain_ckpt = os.path.join(pretrain_save_dir, "best_model.pt")
        if os.path.exists(pretrain_ckpt):
            load_checkpoint(model, pretrain_ckpt, device=str(device))
            print(f"[Phase 1→2] Pre-trained best model 로드 완료")

        if args.pretrain_only:
            print("\n[Train] --pretrain_only: Phase 1만 실행하고 종료.")
            return
    else:
        if args.skip_pretrain:
            print("\n[Train] --skip_pretrain: Phase 1 건너뜀")
            pretrain_ckpt = os.path.join(save_dir, "pretrain", "best_model.pt")
            if os.path.exists(pretrain_ckpt):
                load_checkpoint(model, pretrain_ckpt, device=str(device))
                print(f"[Train] Pre-trained checkpoint 로드: {pretrain_ckpt}")
        else:
            print("\n[Train] pretrain_epochs=0: Phase 1 비활성화")

    # ══════════════════════════════════════════════════════════
    # Phase 2: Annotated Fine-tuning (3K 문서)
    #
    # Phase 1에서 distant data로 기본 패턴을 배운 모델을
    # 깨끗한 human-annotated 데이터로 정밀 조정.
    #
    # Phase 1 → Phase 2 전환 시:
    #   1. Optimizer 새로 생성 (lr decay 초기화)
    #   2. Scheduler 새로 생성 (Phase 2 total_steps에 맞게)
    #   3. 모델 가중치는 Phase 1에서 이어받음
    # ══════════════════════════════════════════════════════════
    print("\n[Phase 2] Annotated Fine-tuning 준비 중...")

    train_loader = create_dataloader(
        data_dir=config["data"]["data_dir"],
        data_file=config["data"]["train_file"],
        tokenizer=tokenizer,
        rel2id=rel2id,
        max_seq_len=config["data"]["max_seq_length"],
        batch_size=train_cfg["finetune_batch_size"],
        shuffle=True,
        stage=config["experiment"]["stage"],
    )

    finetune_epochs = train_cfg["finetune_epochs"]
    finetune_save_dir = os.path.join(save_dir, "finetune")
    phase2_f1 = run_phase(
        phase_name="Fine-tuning (Annotated)",
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        config=config,
        device=device,
        num_epochs=finetune_epochs,
        save_dir=finetune_save_dir,
        phase_label="Phase 2",
    )

    # ── 최종 best model을 메인 save_dir에 복사 ──
    finetune_best = os.path.join(finetune_save_dir, "best_model.pt")
    final_best = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(finetune_best):
        import shutil
        shutil.copy2(finetune_best, final_best)

    print(f"\n{'=' * 65}")
    print(f"[Train] 전체 학습 완료!")
    print(f"  Phase 2 Best Dev F1: {phase2_f1:.2f}%")
    print(f"  최종 모델: {final_best}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()