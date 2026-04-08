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

TODO(정현)
 - RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! 수정
    -  labels = labels.to(outputs["relation_logits"].device) 
    - labels를 모델 출력값(GPU)과 똑같은 디바이스로 보내도록 수정
    - loss_fn = BCEWithWeightLoss(num_relations, no_relation_weight).to(outputs["relation_logits"].device)
        re_loss = loss_fn(outputs["relation_logits"], labels)

============================================================

TODO (박재윤):
    fix: ATLOPLoss를 논문 원본 ranking loss로 교체

    - 기존: BCE with threshold class concat (단순 이진 분류)
    - 수정: ATLOP 논문의 ranking loss 구현
            positive relation은 TH보다 높게,
            negative relation은 TH보다 낮게 학습하는 구조
    - 수식: loss = log(1+Σexp(neg-TH)) + log(1+Σexp(TH-pos))
    - 근거: Zhou et al. (2021) ATLOP Section 3.1
    - 파일: src/losses.py → ATLOPLoss class
    
============================================================

TODO (이수민):
    fix: BCEWithWeightLoss 버그 수정

    - 문제1: `self.num_relations`를 `__init__`에서 저장하지 않아 `forward`에서 AttributeError 발생 가능
    - 수정: `self.num_relations = num_relations` 추가
    - 문제2: `pos_weight`가 BCEWithLogitsLoss에 제대로 전달되지 않음
    - 원인: `F.binary_cross_entropy_with_logits` 호출에서 `pos_weight` 인자가 누락

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
        self.num_relations = num_relations
        # no_relation (보통 index 0) 가중치를 낮춤 -- 클래스 불균형 완화 / 이수민
        weights = torch.ones(num_relations)
        weights[0] = no_relation_weight
        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.ones(self.num_relations, device=logits.device) * 10.0
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, 
            weight=self.weights.unsqueeze(0).to(logits.device),
            pos_weight=pos_weight,
        )
        return loss


class ATLOPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, relation_logits, threshold_logits, labels):
        # [num_pairs, num_relations + 1]
        logits = torch.cat([relation_logits, threshold_logits], dim=-1)
        num_rels = relation_logits.size(-1)

        # TH label: positive가 하나도 없으면 TH가 1
        th_label = (labels[:, 1:].sum(dim=-1) == 0).float().unsqueeze(-1)
        full_labels = torch.cat([labels, th_label], dim=-1)

        # ── 핵심: ATLOP ranking loss ──
        # positive class는 TH보다 높아야 함
        # negative class는 TH보다 낮아야 함
        th_logit = logits[:, -1:]  # [num_pairs, 1]

        # log(1 + sum_neg(exp(neg - TH))) + log(1 + sum_pos(exp(TH - pos)))
        pos_mask = full_labels[:, :-1]   # [num_pairs, num_rels]
        neg_mask = 1 - pos_mask

        pos_logits = logits[:, :-1]  # [num_pairs, num_rels]

        # negative part: exp(logit_neg - TH)
        neg_loss = torch.logsumexp(
            pos_logits * neg_mask + (-1e30) * pos_mask - th_logit, dim=-1
        )
        neg_loss = torch.log1p(torch.exp(neg_loss))

        # positive part: exp(TH - logit_pos)
        pos_loss = torch.logsumexp(
            -pos_logits * pos_mask + (-1e30) * neg_mask + th_logit, dim=-1
        )
        pos_loss = torch.log1p(torch.exp(pos_loss))

        loss = (neg_loss + pos_loss).mean()
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
    
    # labels를 모델 출력값(GPU)과 똑같은 디바이스로 이동
    labels = labels.to(outputs["relation_logits"].device)

    if loss_type == "bce":
        # Stage 1: BCE
        #LOSS_FN을 GOU로 전송
        loss_fn = BCEWithWeightLoss(num_relations, no_relation_weight).to(outputs["relation_logits"].device)
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
