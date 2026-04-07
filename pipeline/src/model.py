"""
============================================================
통합 모델 (model.py)
============================================================
역할: 전체 파이프라인을 하나의 nn.Module로 통합.
      Stage flag에 따라 각 레이어를 선택적으로 활성화.

      Stage 1: Encoder → MeanPool → Bilinear Classifier
      Stage 2: Encoder → LogSumExp → ATLOP + DREEAM
      Stage 3: Encoder → LogSumExp → GNN → ATLOP + DREEAM

INPUT:
  - batch: collate_fn이 반환하는 Dict (input_ids, entity_spans, 등)

OUTPUT:
  - Dict with relation_logits, evidence_logits, threshold_logits

담당: 모델 담당 (전체 통합)

TODO ( 수정 포인트):
  - [ ] Stage 전환 시 checkpoint 로드/세이브 로직
  - [ ] Mixed precision training 지원
  - [ ] DistributedDataParallel 지원
============================================================
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .encoder import DocumentEncoder
from .entity_repr import EntityRepresentation
from .relation_head import RelationHead
from .graph_encoder import GraphEncoder


class DocREModel(nn.Module):
    """
    Document-level Relation Extraction 통합 모델.
    3-Stage Incremental Stacking 아키텍처.

    Args:
        config : YAML config를 로드한 Dict
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.stage = config["experiment"]["stage"]

        enc_cfg = config["encoder"]
        ent_cfg = config["entity_repr"]
        rel_cfg = config["relation_head"]
        graph_cfg = config.get("graph_encoder", {})

        hidden_size = enc_cfg["hidden_size"]
        num_relations = config["data"]["num_relations"]

        # ── Layer 1: Document Encoder ──
        self.encoder = DocumentEncoder(
            model_name=enc_cfg["model_name"],
            hidden_size=hidden_size,
        )

        # ── Layer 2: Entity Representation ──
        self.entity_repr = EntityRepresentation(
            hidden_size=hidden_size,
            pooling=ent_cfg["pooling"],  # 'mean' (stage1) or 'logsumexp' (stage2+)
        )

        # ── Layer 3: Graph Encoder (Stage 3 only) ──
        self.graph_encoder = None
        if graph_cfg.get("enabled", False):
            self.graph_encoder = GraphEncoder(
                hidden_dim=graph_cfg.get("hidden_dim", hidden_size),
                num_layers=graph_cfg.get("num_layers", 2),
                gnn_type=graph_cfg.get("gnn_type", "gcn"),
                dropout=graph_cfg.get("dropout", 0.1),
                cross_sent_window=graph_cfg.get("cross_sent_window", 1),
            )

        # ── Layer 4: Relation Head ──
        self.relation_head = RelationHead(
            hidden_size=hidden_size,
            num_relations=num_relations,
            classifier_type=rel_cfg.get("classifier_type", "bilinear"),
            threshold_type=rel_cfg.get("threshold_type", "fixed"),
            fixed_threshold=rel_cfg.get("fixed_threshold", 0.5),
            use_evidence=rel_cfg.get("use_evidence_head", False),
        )

    def forward(self, batch: Dict) -> List[Dict]:
        """
        전체 모델 Forward Pass.

        INPUT:
          batch — collate_fn 출력:
            - input_ids      : [B, seq_len]
            - attention_mask  : [B, seq_len]
            - entity_spans   : List[List[List[Tuple]]]
            - entity_pairs   : List[List[Tuple]]
            - sent_map       : List[List[int]]
            - num_sents      : List[int]

        OUTPUT:
          List[Dict] — 각 batch 원소마다:
            - relation_logits  : [num_pairs, num_relations]
            - threshold_logits : Optional
            - evidence_logits  : Optional
        """
        input_ids = batch["input_ids"]           # [B, seq_len]
        attention_mask = batch["attention_mask"]  # [B, seq_len]
        device = input_ids.device

        # ── Step 1: Document Encoding ──
        hidden_states = self.encoder(input_ids, attention_mask)
        # [B, seq_len, hidden_size]

        # ── Step 2: Entity Representation ──
        batch_entity_vecs = self.entity_repr(
            hidden_states, batch["entity_spans"]
        )
        # List[Tensor[num_entities_i, hidden_size]]

        # ── Step 3: (Stage 3) Graph Reasoning ──
        if self.graph_encoder is not None and self.stage == "stage3":
            refined_vecs = []
            for b in range(len(batch_entity_vecs)):
                ev = batch_entity_vecs[b]  # [num_entities, hidden]
                refined = self.graph_encoder(
                    entity_vectors=ev,
                    entity_spans=batch["entity_spans"][b],
                    sent_map=batch["sent_map"][b],
                    num_sents=batch["num_sents"][b],
                )
                refined_vecs.append(refined)
            batch_entity_vecs = refined_vecs

        # ── Step 4: Relation Classification ──
        all_outputs = []
        for b in range(len(batch_entity_vecs)):
            outputs = self.relation_head(
                entity_vectors=batch_entity_vecs[b],
                entity_pairs=batch["entity_pairs"][b],
                num_sents=batch["num_sents"][b],
            )
            all_outputs.append(outputs)

        return all_outputs

    def get_encoder_params(self):
        """인코더 파라미터 (낮은 lr 적용)"""
        return self.encoder.parameters()

    def get_non_encoder_params(self):
        """인코더 제외 파라미터 (높은 lr 적용)"""
        encoder_param_ids = set(id(p) for p in self.encoder.parameters())
        return [p for p in self.parameters() if id(p) not in encoder_param_ids]
