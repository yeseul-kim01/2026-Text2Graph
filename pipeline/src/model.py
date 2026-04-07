"""
============================================================
통합 모델 (model.py)
============================================================
역할: 전체 파이프라인을 하나의 nn.Module로 통합.
      Stage flag에 따라 각 레이어를 선택적으로 활성화.

      Stage 1: Encoder → MeanPool → Bilinear Classifier
      Stage 2: Encoder → LogSumExp → ATLOP + DREEAM
      Stage 3: Encoder → LogSumExp → GNN → ATLOP + DREEAM
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
            pooling=ent_cfg["pooling"], 
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
        input_ids = batch["input_ids"]           # [B, seq_len]
        attention_mask = batch["attention_mask"]  # [B, seq_len]
        device = input_ids.device

        # ── Step 1: Document Encoding ──
        hidden_states = self.encoder(input_ids, attention_mask)

        # ── Step 2: Entity Representation ──
        batch_entity_vecs = self.entity_repr(
            hidden_states, batch["entity_spans"]
        )

        # ── Step 3: (Stage 3) Graph Reasoning ──
        if self.graph_encoder is not None and self.stage == "stage3":
            refined_vecs = []
            for b in range(len(batch_entity_vecs)):
                ev = batch_entity_vecs[b]
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
            
            # =========================================================
            # [배관 공사] Stage 2 (ATLOP/DREEAM) 전용 문맥(rs) 추출 로직
            # =========================================================
            rs_vectors = None
            if self.relation_head.classifier_type == "atlop":
                h_states = hidden_states[b]  # [seq_len, hidden_size]
                e_vecs = batch_entity_vecs[b]  # [num_entities, hidden_size]
                pairs = batch["entity_pairs"][b]

                if len(pairs) > 0:
                    head_ids = [p[0] for p in pairs]
                    tail_ids = [p[1] for p in pairs]
                    
                    h_vecs = e_vecs[head_ids]  # [num_pairs, hidden_size]
                    t_vecs = e_vecs[tail_ids]  # [num_pairs, hidden_size]
                    
                    # 1. 문서 전체의 단어들과 Head/Tail 간의 연관성(유사도)을 계산합니다.
                    h_att = torch.softmax(torch.matmul(h_vecs, h_states.T), dim=-1)  # [num_pairs, seq_len]
                    t_att = torch.softmax(torch.matmul(t_vecs, h_states.T), dim=-1)  # [num_pairs, seq_len]
                    
                    # 2. 두 엔티티가 공통으로 주목하는 문맥(교집합)을 구합니다.
                    ht_att = h_att * t_att
                    ht_att = ht_att / (ht_att.sum(dim=-1, keepdim=True) + 1e-30)
                    
                    # 3. 그 문맥의 정보만 쏙 뽑아서 문맥 벡터(rs)를 만듭니다!
                    rs_vectors = torch.matmul(ht_att, h_states)  # [num_pairs, hidden_size]
            # =========================================================
            
            # 뽑아낸 문맥(rs_vectors)을 방금 우리가 만든 RelationHead로 쏴줍니다!
            outputs = self.relation_head(
                entity_vectors=batch_entity_vecs[b],
                entity_pairs=batch["entity_pairs"][b],
                rs_vectors=rs_vectors,  # <--- 파이프라인 연결 완료!
                num_sents=batch["num_sents"][b],
            )
            all_outputs.append(outputs)

        return all_outputs

    def get_encoder_params(self):
        return self.encoder.parameters()

    def get_non_encoder_params(self):
        encoder_param_ids = set(id(p) for p in self.encoder.parameters())
        return [p for p in self.parameters() if id(p) not in encoder_param_ids]