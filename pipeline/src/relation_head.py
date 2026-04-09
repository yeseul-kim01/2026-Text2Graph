"""
============================================================
Relation Extraction Layer (relation_head.py)
============================================================
역할:
  Entity pair representation을 받아 multi-label 관계 분류 수행.
  Stage별로 baseline / ATLOP / DREEAM 스타일 classifier를 지원한다.

입력:
  - entity_vectors : [num_entities, hidden_size]
  - entity_pairs   : List[Tuple(h, t)]
  - rs_vectors     : Optional[num_pairs, hidden_size]
                     ATLOP localized context pooling으로 얻은 pair-specific context
  - num_sents      : evidence head 출력 길이 결정용

출력:
  - relation_logits   : [num_pairs, num_relations]
  - threshold_logits  : Optional[num_pairs, 1]        (adaptive threshold)
  - evidence_logits   : Optional[num_pairs, num_sents]
============================================================
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict


class RelationHead(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_relations: int = 97,
        classifier_type: str = "bilinear",
        threshold_type: str = "fixed",
        fixed_threshold: float = 0.5,
        use_evidence: bool = False,
        max_num_sents: int = 25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_relations = num_relations
        self.classifier_type = classifier_type
        self.threshold_type = threshold_type
        self.fixed_threshold = fixed_threshold
        self.use_evidence = use_evidence
        self.max_num_sents = max_num_sents

        self.dropout = nn.Dropout(dropout)

        # ---------------------------------------------------------
        # Stage 1 baseline
        #   pair_repr = [h ; t ; h*t]
        # ---------------------------------------------------------
        if classifier_type != "atlop":
            pair_dim = hidden_size * 3
            self.classifier = nn.Linear(pair_dim, num_relations)

            if use_evidence:
                self.evidence_head = nn.Linear(hidden_size, max_num_sents)

        # ---------------------------------------------------------
        # Stage 2~4: ATLOP / DREEAM 스타일
        #   [h ; rs], [t ; rs] -> projection -> grouped bilinear
        # ---------------------------------------------------------
        else:
            self.emb_size = hidden_size
            self.block_size = 64
            assert self.emb_size % self.block_size == 0, \
                "hidden_size must be divisible by block_size"

            # 순정 ATLOP/DREEAM 흐름:
            # head/tail 각각 [entity ; context(rs)]를 받아 투영
            self.head_extractor = nn.Linear(hidden_size * 2, self.emb_size)
            self.tail_extractor = nn.Linear(hidden_size * 2, self.emb_size)

            classifier_input_dim = self.emb_size * self.block_size
            self.bilinear = nn.Linear(classifier_input_dim, num_relations, bias=True)

            if threshold_type == "adaptive":
                self.threshold_linear = nn.Linear(classifier_input_dim, 1, bias=True)

            if use_evidence:
                self.evidence_head = nn.Linear(classifier_input_dim, max_num_sents)

    def forward(
        self,
        entity_vectors: torch.Tensor,
        entity_pairs: List[Tuple[int, int]],
        rs_vectors: Optional[torch.Tensor] = None,
        num_sents: int = 0,
    ) -> Dict[str, torch.Tensor]:
        device = entity_vectors.device

        if len(entity_pairs) == 0:
            outputs = {
                "relation_logits": torch.zeros(0, self.num_relations, device=device),
            }
            if self.threshold_type == "adaptive":
                outputs["threshold_logits"] = torch.zeros(0, 1, device=device)
            if self.use_evidence and num_sents > 0:
                outputs["evidence_logits"] = torch.zeros(0, num_sents, device=device)
            return outputs

        head_ids = [p[0] for p in entity_pairs]
        tail_ids = [p[1] for p in entity_pairs]

        head_vecs = entity_vectors[head_ids]  # [num_pairs, hidden]
        tail_vecs = entity_vectors[tail_ids]  # [num_pairs, hidden]

        outputs: Dict[str, torch.Tensor] = {}

        # =========================================================
        # Stage 2~4: ATLOP / DREEAM classifier
        # =========================================================
        if self.classifier_type == "atlop":
            if rs_vectors is None:
                rs_vectors = torch.zeros_like(head_vecs)

            # [entity ; context]
            h_input = torch.cat([head_vecs, rs_vectors], dim=-1)
            t_input = torch.cat([tail_vecs, rs_vectors], dim=-1)

            # projection
            h_proj = torch.tanh(self.head_extractor(h_input))  # [num_pairs, emb]
            t_proj = torch.tanh(self.tail_extractor(t_input))  # [num_pairs, emb]

            # grouped bilinear
            # [B, emb] -> [B, emb/block, block]
            b1 = h_proj.view(-1, self.emb_size // self.block_size, self.block_size)
            b2 = t_proj.view(-1, self.emb_size // self.block_size, self.block_size)

            # block outer product
            pair_repr = (b1.unsqueeze(3) * b2.unsqueeze(2)).reshape(
                -1, self.emb_size * self.block_size
            )

            pair_repr = self.dropout(pair_repr)

            relation_logits = self.bilinear(pair_repr)
            outputs["relation_logits"] = relation_logits

            if self.threshold_type == "adaptive":
                outputs["threshold_logits"] = self.threshold_linear(pair_repr)

            if self.use_evidence and num_sents > 0:
                evidence_logits = self.evidence_head(pair_repr)
                outputs["evidence_logits"] = evidence_logits[:, :num_sents]

        # =========================================================
        # Stage 1 baseline classifier
        # =========================================================
        else:
            pair_repr = torch.cat(
                [head_vecs, tail_vecs, head_vecs * tail_vecs],
                dim=-1,
            )
            pair_repr = self.dropout(pair_repr)

            relation_logits = self.classifier(pair_repr)
            outputs["relation_logits"] = relation_logits

            if self.use_evidence and num_sents > 0:
                evidence_logits = self.evidence_head(head_vecs * tail_vecs)
                outputs["evidence_logits"] = evidence_logits[:, :num_sents]

        return outputs

    def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = outputs["relation_logits"]

        if self.threshold_type == "adaptive" and "threshold_logits" in outputs:
            th_logits = outputs["threshold_logits"]
            predictions = (logits > th_logits).float()
        else:
            probs = torch.sigmoid(logits)
            predictions = (probs > self.fixed_threshold).float()

        return predictions