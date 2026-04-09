"""
============================================================
통합 모델 (model.py)
============================================================
역할:
  전체 파이프라인을 하나의 nn.Module로 통합하고,
  stage 설정에 따라 각 레이어를 선택적으로 활성화한다.

  Stage 1: Encoder → Mean Pooling → Bilinear Classifier
  Stage 2: Encoder → LogSumExp → ATLOP + DREEAM-style context
  Stage 3: Encoder → LogSumExp → Graph Encoder → ATLOP + DREEAM-style context
  Stage 4: Encoder → LogSumExp → Graph U-Net → ATLOP + DREEAM-style context

설계 원칙:
  - Stage 1은 단순 baseline 유지
  - Stage 2~4는 ATLOP 스타일 localized context pooling(rs) 사용
  - Graph reasoning은 entity vector를 refinement하는 역할
  - relation 분류는 최종 entity vector + pair-specific context(rs)로 수행

TODO (김예슬):

[완료]
  - encoder 출력에서 attention 반환 옵션 연결
  - entity_repr에 attention 전달
  - entity_attns 기반 localized_context_pooling(rs) 연결
  - Stage 3 / Stage 4에서 graph encoder 분기 처리
  - 중복된 rs 계산 블록 제거 및 fallback 로직 정리
  - 전체 주석 정리 및 stage 흐름 명시

[현재 파일에서 반영 가능한 범위 내 완료]
  - 기존 inner-product 기반 context 계산은 fallback으로만 유지
  - Stage 2~4에서 relation_head로 rs_vectors 전달 완료

[별도 파일에서 진행 필요]
  - compute_loss GPU 고정 로직 정리 (losses.py / train.py 쪽 확인 필요)
  - teacher attention 기반 silver evidence 생성 (infer_evidence.py)
  - evidence supervision alignment 강화
  - evaluate.py의 실제 F1 계산 마무리
  - preprocessing.py no_relation 라벨 순서 수정

담당:
  - 모델 담당: 김예슬
============================================================
"""

from typing import Dict, List

import torch
import torch.nn as nn

from .encoder import DocumentEncoder
from .entity_repr import EntityRepresentation, localized_context_pooling
from .relation_head import RelationHead
from .graph_encoder import GraphEncoder
from .structural_encorder import GraphUNetEncoder


class DocREModel(nn.Module):
    """
    문서 단위 관계 추출 전체 모델.

    구성:
      1) Document Encoder
      2) Entity Representation
      3) Graph Reasoning (선택)
      4) Relation Head

    forward 입력:
      batch = {
        "input_ids": Tensor[B, L],
        "attention_mask": Tensor[B, L],
        "entity_spans": List,
        "entity_pairs": List,
        "sent_map": List,
        "num_sents": List,
        ...
      }

    forward 출력:
      List[Dict]
        문서별 relation_head 출력 결과
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

        # =========================================================
        # Layer 1. Document Encoder
        # =========================================================
        # BERT / RoBERTa 계열 문서 인코더.
        # Stage 2~4에서 ATLOP localized context pooling을 쓰기 위해
        # 필요 시 attention도 함께 반환하도록 사용한다.
        self.encoder = DocumentEncoder(
            model_name=enc_cfg["model_name"],
            hidden_size=hidden_size,
        )

        # =========================================================
        # Layer 2. Entity Representation
        # =========================================================
        # mention-level hidden states를 entity-level representation으로 통합.
        # Stage 1: mean pooling
        # Stage 2~4: logsumexp pooling (ATLOP 방식)
        self.entity_repr = EntityRepresentation(
            hidden_size=hidden_size,
            pooling=ent_cfg["pooling"],
        )

        # =========================================================
        # Layer 3. Graph Encoder (Stage 3 / 4)
        # =========================================================
        # Stage 3: 기본 GraphEncoder (GAIN-lite / GCN/GAT 계열)
        # Stage 4: GraphUNetEncoder
        self.graph_encoder = None
        if graph_cfg.get("enabled", False):
            arch_type = graph_cfg.get("architecture", "gain")

            if arch_type == "unet":
                print("🚀 [Model Init] Graph U-Net Encoder 활성화!")
                self.graph_encoder = GraphUNetEncoder(
                    hidden_dim=graph_cfg.get("hidden_dim", hidden_size),
                    num_layers=graph_cfg.get("num_layers", 2),
                    gnn_type=graph_cfg.get("gnn_type", "gcn"),
                    dropout=graph_cfg.get("dropout", 0.1),
                    cross_sent_window=graph_cfg.get("cross_sent_window", 1),
                    pool_ratio=graph_cfg.get("pool_ratio", 0.5),
                )
            else:
                print("🟢 [Model Init] 기본 GAIN Graph Encoder 활성화!")
                self.graph_encoder = GraphEncoder(
                    hidden_dim=graph_cfg.get("hidden_dim", hidden_size),
                    num_layers=graph_cfg.get("num_layers", 2),
                    gnn_type=graph_cfg.get("gnn_type", "gcn"),
                    dropout=graph_cfg.get("dropout", 0.1),
                    cross_sent_window=graph_cfg.get("cross_sent_window", 1), 
                    residual_alpha=graph_cfg.get("residual_alpha", 0.3),
                    num_heads=graph_cfg.get("num_heads", 4),
                )

        # =========================================================
        # Layer 4. Relation Head
        # =========================================================
        # classifier_type:
        #   - bilinear : Stage 1 baseline
        #   - atlop    : Stage 2~4
        self.relation_head = RelationHead(
            hidden_size=hidden_size,
            num_relations=num_relations,
            classifier_type=rel_cfg.get("classifier_type", "bilinear"),
            threshold_type=rel_cfg.get("threshold_type", "fixed"),
            fixed_threshold=rel_cfg.get("fixed_threshold", 0.5),
            use_evidence=rel_cfg.get("use_evidence_head", False),
            dropout=enc_cfg.get("dropout", 0.1),
        )

    def _compute_rs_vectors(
        self,
        hidden_states_single: torch.Tensor,
        entity_vectors_single: torch.Tensor,
        entity_pairs_single: List,
        entity_attns_single: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Stage 2~4에서 사용하는 pair-specific context vector(rs) 생성 함수.

        우선순위:
          1) localized_context_pooling (ATLOP 정식 방식)
          2) attention이 없으면 inner-product 기반 fallback

        Args:
          hidden_states_single : [seq_len, hidden_size]
          entity_vectors_single: [num_entities, hidden_size]
          entity_pairs_single  : [(h_id, t_id), ...]
          entity_attns_single  : [num_entities, num_heads, seq_len] or None

        Returns:
          rs_vectors : [num_pairs, hidden_size] or None
        """
        if len(entity_pairs_single) == 0:
            return None

        # ---------------------------------------------------------
        # Case 1. 정식 경로: ATLOP localized context pooling
        # ---------------------------------------------------------
        if entity_attns_single is not None:
            return localized_context_pooling(
                hidden_states=hidden_states_single,
                entity_attns=entity_attns_single,
                entity_pairs=entity_pairs_single,
            )

        # ---------------------------------------------------------
        # Case 2. fallback: attention이 없는 경우
        # ---------------------------------------------------------
        # entity vector와 token hidden state 간 inner-product를 이용해
        # soft attention을 만들고, head/tail attention의 교집합으로
        # context를 구성한다.
        head_ids = [p[0] for p in entity_pairs_single]
        tail_ids = [p[1] for p in entity_pairs_single]

        h_vecs = entity_vectors_single[head_ids]  # [num_pairs, hidden]
        t_vecs = entity_vectors_single[tail_ids]  # [num_pairs, hidden]

        h_att = torch.softmax(torch.matmul(h_vecs, hidden_states_single.T), dim=-1)
        t_att = torch.softmax(torch.matmul(t_vecs, hidden_states_single.T), dim=-1)

        ht_att = h_att * t_att
        ht_att = ht_att / (ht_att.sum(dim=-1, keepdim=True) + 1e-30)

        rs_vectors = torch.matmul(ht_att, hidden_states_single)  # [num_pairs, hidden]
        return rs_vectors

    def forward(self, batch: Dict) -> List[Dict]:
        """
        전체 forward 흐름

        Step 1. Document Encoding
        Step 2. Entity Representation
        Step 3. Graph Reasoning (선택)
        Step 4. Relation Classification
        """
        input_ids = batch["input_ids"]            # [B, seq_len]
        attention_mask = batch["attention_mask"]  # [B, seq_len]

        # =========================================================
        # Step 1. Document Encoding
        # =========================================================
        # ATLOP / DREEAM 스타일 문맥 벡터(rs)를 계산하려면
        # encoder attention이 필요하므로 classifier_type이 atlop인 경우에만
        # attention까지 함께 반환받는다.
        use_atlop_context = self.relation_head.classifier_type == "atlop"

        encoder_output = self.encoder(
            input_ids,
            attention_mask,
            return_attention=use_atlop_context,
        )

        if use_atlop_context:
            hidden_states = encoder_output["hidden_states"]  # [B, L, H]
            attentions = encoder_output["attentions"]        # [B, heads, L, L]
        else:
            hidden_states = encoder_output                   # [B, L, H]
            attentions = None

        # =========================================================
        # Step 2. Entity Representation
        # =========================================================
        # entity_repr는
        #   - entity_vectors
        #   - entity_attns (attention 입력 시)
        # 를 반환한다.
        repr_output = self.entity_repr(
            hidden_states,
            batch["entity_spans"],
            attention=attentions,
        )

        batch_entity_vecs = repr_output["entity_vectors"]
        batch_entity_attns = repr_output.get("entity_attns", None)

        # =========================================================
        # Step 3. Graph Reasoning (Stage 3 / 4)
        # =========================================================
        # graph encoder는 entity vector만 refinement한다.
        # attention은 encoder 단계에서 얻은 entity_attns를 그대로 사용한다.
        if self.graph_encoder is not None and self.stage in ["stage3", "stage4"]:
            refined_vecs = []

            for b in range(len(batch_entity_vecs)):
                entity_vectors_single = batch_entity_vecs[b]

                refined = self.graph_encoder(
                    entity_vectors=entity_vectors_single,
                    entity_spans=batch["entity_spans"][b],
                    sent_map=batch["sent_map"][b],
                    num_sents=batch["num_sents"][b],
                )
                refined_vecs.append(refined)

            batch_entity_vecs = refined_vecs

        # =========================================================
        # Step 4. Relation Classification
        # =========================================================
        all_outputs = []

        for b in range(len(batch_entity_vecs)):
            entity_vectors_single = batch_entity_vecs[b]
            entity_pairs_single = batch["entity_pairs"][b]
            hidden_states_single = hidden_states[b]

            # -----------------------------------------------------
            # Stage 2~4: pair-specific context(rs) 계산
            # -----------------------------------------------------
            rs_vectors = None
            if self.relation_head.classifier_type == "atlop":
                entity_attns_single = None
                if batch_entity_attns is not None and b < len(batch_entity_attns):
                    entity_attns_single = batch_entity_attns[b]

                rs_vectors = self._compute_rs_vectors(
                    hidden_states_single=hidden_states_single,
                    entity_vectors_single=entity_vectors_single,
                    entity_pairs_single=entity_pairs_single,
                    entity_attns_single=entity_attns_single,
                )

            # -----------------------------------------------------
            # Relation Head
            # -----------------------------------------------------
            outputs = self.relation_head(
                entity_vectors=entity_vectors_single,
                entity_pairs=entity_pairs_single,
                rs_vectors=rs_vectors,
                num_sents=batch["num_sents"][b],
            )
            all_outputs.append(outputs)

        return all_outputs

    def get_encoder_params(self):
        """optimizer에서 encoder 전용 lr를 따로 줄 때 사용"""
        return self.encoder.parameters()

    def get_non_encoder_params(self):
        """encoder를 제외한 나머지 파라미터 반환"""
        encoder_param_ids = set(id(p) for p in self.encoder.parameters())
        return [p for p in self.parameters() if id(p) not in encoder_param_ids]