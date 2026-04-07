"""
============================================================
Loss Functions (losses.py)
============================================================
역할: 각 Stage에 맞는 손실 함수 제공

Stage 1: BCE Loss (고정 threshold)
Stage 2+: ATLOP Adaptive Threshold Loss + DREEAM Evidence Loss

INPUT:  모델 출력 logits, 정답 labels
OUTPUT: scalar loss 값

기반 논문:
  - Zhou et al. (2021) ATLOP — ATL (Adaptive Threshold Loss)
  - Ma et al. (2023) DREEAM — Evidence KL-divergence loss

담당: 모델 담당

TODO ( 수정 포인트):
  - [ ] Focal loss 실험 (class imbalance 대응)
  - [ ] Label smoothing 적용
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BCEWithWeightLoss(nn.Module):
    """
    Stage 1용: Weighted Binary Cross-Entropy Loss.
    no_relation class의 가중치를 조절하여 클래스 불균형 대응.

    INPUT:
      - logits : [num_pairs, num_relations]
      - labels : [num_pairs, num_relations] (multi-hot)
    OUTPUT:
      - loss   : scalar
    """

    def __init__(self, num_relations: int = 97, no_relation_weight: float = 0.1):
        super().__init__()
        # no_relation (보통 index 0) 가중치를 낮춤
        weights = torch.ones(num_relations)
        weights[0] = no_relation_weight
        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, weight=self.weights.unsqueeze(0),
        )
        return loss


class ATLOPLoss(nn.Module):
    """
    Stage 2+ 용: ATLOP Adaptive Threshold Loss.
    각 pair에 대해 TH class를 기준으로 positive/negative relation을 구분.

    INPUT:
      - relation_logits  : [num_pairs, num_relations]
      - threshold_logits : [num_pairs, 1]
      - labels           : [num_pairs, num_relations]
    OUTPUT:
      - loss : scalar
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        relation_logits: torch.Tensor,
        threshold_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # TH logit을 relation logits에 concat
        # logits: [num_pairs, num_relations + 1] (마지막이 TH)
        logits = torch.cat([relation_logits, threshold_logits], dim=-1)

        # Positive relations: label == 1
        # Negative relations: label == 0 → TH가 이겨야 함

        num_relations = relation_logits.size(-1)

        # TH class의 label: positive relation이 하나도 없는 pair에서 1
        th_label = (labels.sum(dim=-1) == 0).float().unsqueeze(-1)  # [num_pairs, 1]
        full_labels = torch.cat([labels, th_label], dim=-1)

        # Multi-label softmax loss (ATLOP 스타일)
        # log_sum_exp over positive classes vs negative classes
        loss = F.binary_cross_entropy_with_logits(logits, full_labels)
        return loss


class EvidenceLoss(nn.Module):
    """
    DREEAM Evidence Loss: KL-divergence between
    predicted attention and evidence distribution.

    INPUT:
      - evidence_logits    : [num_pairs, num_sents] — 모델 예측
      - evidence_labels    : Dict[pair_idx -> {rel_idx -> [sent_ids]}]
      - num_sents          : int
    OUTPUT:
      - loss : scalar
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        evidence_logits: torch.Tensor,
        evidence_labels: Dict,
        num_sents: int,
    ) -> torch.Tensor:
        if len(evidence_labels) == 0 or evidence_logits.size(0) == 0:
            return torch.tensor(0.0, device=evidence_logits.device)

        # Evidence distribution 생성
        target_dist = torch.zeros_like(evidence_logits)
        count = 0

        for pair_idx, rel_dict in evidence_labels.items():
            if pair_idx >= target_dist.size(0):
                continue
            for rel_idx, sent_ids in rel_dict.items():
                for sid in sent_ids:
                    if sid < num_sents:
                        target_dist[pair_idx][sid] = 1.0
                        count += 1

        if count == 0:
            return torch.tensor(0.0, device=evidence_logits.device)

        # Normalize to distribution
        target_sum = target_dist.sum(dim=-1, keepdim=True).clamp(min=1)
        target_dist = target_dist / target_sum

        # Predicted distribution (softmax)
        pred_dist = F.softmax(evidence_logits, dim=-1)

        # KL divergence
        loss = F.kl_div(
            pred_dist.log().clamp(min=-100),
            target_dist,
            reduction="batchmean",
        )
        return loss


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    evidence_labels: Dict,
    num_sents: int,
    loss_type: str = "bce",
    lambda_evidence: float = 0.1,
    no_relation_weight: float = 0.1,
    num_relations: int = 97,
) -> Dict[str, torch.Tensor]:
    """
    Stage에 따른 통합 loss 계산 함수.

    INPUT:
      - outputs   : 모델 forward 출력
      - labels    : [num_pairs, num_relations]
      - 기타 설정
    OUTPUT:
      - Dict with 'total_loss', 're_loss', 'evidence_loss'
    """
    result = {}

    if loss_type == "bce":
        # Stage 1: BCE
        loss_fn = BCEWithWeightLoss(num_relations, no_relation_weight)
        re_loss = loss_fn(outputs["relation_logits"], labels)
    elif loss_type == "atlop":
        # Stage 2+: ATLOP ATL
        if "threshold_logits" in outputs:
            loss_fn = ATLOPLoss()
            re_loss = loss_fn(
                outputs["relation_logits"],
                outputs["threshold_logits"],
                labels,
            )
        else:
            loss_fn = BCEWithWeightLoss(num_relations, no_relation_weight)
            re_loss = loss_fn(outputs["relation_logits"], labels)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    result["re_loss"] = re_loss
    total_loss = re_loss

    # ── Evidence Loss (DREEAM) ──
    if "evidence_logits" in outputs and lambda_evidence > 0:
        evi_loss_fn = EvidenceLoss()
        evi_loss = evi_loss_fn(outputs["evidence_logits"], evidence_labels, num_sents)
        result["evidence_loss"] = evi_loss
        total_loss = total_loss + lambda_evidence * evi_loss

    result["total_loss"] = total_loss
    return result
