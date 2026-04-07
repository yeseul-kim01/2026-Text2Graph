"""
============================================================
Relation Extraction Layer (relation_head.py)
============================================================
역할: Entity pair representation을 받아 multi-label 관계 분류 수행.
      Stage별로 Fixed/ATLOP/DREEAM classifier를 선택적 사용.

INPUT:
  - entity_vectors : [num_entities, hidden_size]
  - entity_pairs   : List[Tuple(h, t)]

OUTPUT:
  - relation_logits  : [num_pairs, num_relations]
  - evidence_logits  : Optional[num_pairs, num_sents] (DREEAM)

기반 논문:
  - Zhou et al. (2021) ATLOP — Adaptive Threshold, bilinear classifier
  - Ma et al. (2023) DREEAM — Evidence-guided attention head

담당: 모델 담당

TODO ( 수정 포인트):
  - [ ] Bilinear classifier 대신 MLP classifier 실험
  - [ ] Evidence head의 attention 시각화 기능 추가
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


class RelationHead(nn.Module):
    """
    Relation Classification Head.
    Stage 1: bilinear + fixed threshold
    Stage 2+: ATLOP adaptive threshold + DREEAM evidence head

    Args:
        hidden_size     : entity vector 차원 (768)
        num_relations   : relation 수 (97 = 96 + no_relation)
        classifier_type : 'bilinear' | 'atlop'
        threshold_type  : 'fixed' | 'adaptive'
        use_evidence    : evidence head 사용 여부
        max_num_sents   : 문서 내 최대 문장 수 (evidence용)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_relations: int = 97,
        classifier_type: str = "bilinear",
        threshold_type: str = "fixed",
        fixed_threshold: float = 0.5,
        use_evidence: bool = False,
        max_num_sents: int = 25,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_relations = num_relations
        self.classifier_type = classifier_type
        self.threshold_type = threshold_type
        self.fixed_threshold = fixed_threshold
        self.use_evidence = use_evidence

        # ── Pair Representation: [e_h; e_t; e_h ⊙ e_t] ──
        pair_dim = hidden_size * 3  # concat + element-wise product

        # ── Relation Classifier ──
        if classifier_type == "atlop":
            # ATLOP: bilinear with head/tail 분리
            # head와 tail에 별도 projection 적용 후 bilinear
            self.head_proj = nn.Linear(hidden_size, hidden_size)
            self.tail_proj = nn.Linear(hidden_size, hidden_size)
            self.bilinear = nn.Linear(hidden_size, num_relations, bias=True)

            if threshold_type == "adaptive":
                # Adaptive Threshold: 학습 가능한 TH class
                # num_relations + 1 (마지막이 threshold class)
                self.threshold_linear = nn.Linear(hidden_size, 1, bias=True)
        else:
            # 기본 bilinear classifier (Stage 1)
            # 노트북 Stage1DocREModel과 동일: Linear(hidden*3, num_relations) 단일 레이어
            self.classifier = nn.Linear(pair_dim, num_relations)

        # ── DREEAM Evidence Head ──
        if use_evidence:
            self.evidence_head = nn.Linear(hidden_size, max_num_sents)
            self.max_num_sents = max_num_sents

    def forward(
        self,
        entity_vectors: torch.Tensor,
        entity_pairs: List[Tuple],
        num_sents: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        INPUT:
          - entity_vectors : [num_entities, hidden_size]
          - entity_pairs   : [(h_id, t_id), ...]
          - num_sents      : 문장 수 (evidence 계산용)

        OUTPUT:
          - Dict with:
            - 'relation_logits': [num_pairs, num_relations]
            - 'threshold_logits': Optional[num_pairs, 1] (ATLOP)
            - 'evidence_logits': Optional[num_pairs, num_sents] (DREEAM)
        """
        if len(entity_pairs) == 0:
            device = entity_vectors.device
            return {
                "relation_logits": torch.zeros(0, self.num_relations, device=device),
            }

        # ── Pair representation 구성 ──
        head_ids = [p[0] for p in entity_pairs]
        tail_ids = [p[1] for p in entity_pairs]

        head_vecs = entity_vectors[head_ids]  # [num_pairs, hidden]
        tail_vecs = entity_vectors[tail_ids]  # [num_pairs, hidden]

        outputs = {}

        if self.classifier_type == "atlop":
            # ── ATLOP Classifier ──
            h_proj = self.head_proj(head_vecs)  # [num_pairs, hidden]
            t_proj = self.tail_proj(tail_vecs)  # [num_pairs, hidden]

            # Element-wise product for bilinear-like interaction
            pair_repr = h_proj * t_proj  # [num_pairs, hidden]

            relation_logits = self.bilinear(pair_repr)  # [num_pairs, num_relations]
            outputs["relation_logits"] = relation_logits

            # Adaptive Threshold
            if self.threshold_type == "adaptive":
                threshold_logits = self.threshold_linear(pair_repr)  # [num_pairs, 1]
                outputs["threshold_logits"] = threshold_logits

            # ── DREEAM Evidence Head ──
            if self.use_evidence and num_sents > 0:
                evidence_logits = self.evidence_head(pair_repr)  # [num_pairs, max_sents]
                evidence_logits = evidence_logits[:, :num_sents]  # 실제 문장 수만큼
                outputs["evidence_logits"] = evidence_logits
        else:
            # ── 기본 Bilinear Classifier (Stage 1) ──
            pair_repr = torch.cat([
                head_vecs,
                tail_vecs,
                head_vecs * tail_vecs,  # element-wise product
            ], dim=-1)  # [num_pairs, hidden*3]

            relation_logits = self.classifier(pair_repr)  # [num_pairs, num_relations]
            outputs["relation_logits"] = relation_logits

        return outputs

    def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Logits → 최종 예측 (threshold 적용).

        INPUT:  forward()의 출력 dict
        OUTPUT: [num_pairs, num_relations] — binary predictions
        """
        logits = outputs["relation_logits"]

        if self.threshold_type == "adaptive" and "threshold_logits" in outputs:
            # ATLOP: relation score > threshold score인 것만 positive
            th_logits = outputs["threshold_logits"]  # [num_pairs, 1]
            predictions = (logits > th_logits).float()
        else:
            # Fixed threshold
            probs = torch.sigmoid(logits)
            predictions = (probs > self.fixed_threshold).float()

        return predictions
