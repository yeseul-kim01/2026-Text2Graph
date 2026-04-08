"""
============================================================
Relation Extraction Layer (relation_head.py)
============================================================
역할: Entity pair representation을 받아 multi-label 관계 분류 수행.
      Stage별로 Fixed/ATLOP/DREEAM classifier를 선택적 사용.

INPUT:
  - entity_vectors : [num_entities, hidden_size]
  - entity_pairs   : List[Tuple(h, t)]
  - rs_vectors     : (옵션) [num_pairs, hidden_size] - DREEAM 문맥 벡터

OUTPUT:
  - relation_logits  : [num_pairs, num_relations]
  - evidence_logits  : Optional[num_pairs, num_sents] (DREEAM)
============================================================

TODO (박재윤):
  - [수정] 삭제) 기존 오리지널 ATLOP 방식의 단순 projection(head_proj, tail_proj) 로직 비활성화 (AttributeError 원인 해결).

    (활성화) DREeAM 방식의 Context-aware extractor(head_extractor, tail_extractor) 로직 활성화. 문맥 벡터(rs_vectors) 정상 반영됨.

    (활성화) 논문 디폴트 세팅인 블록 단위 연산(Grouped Bilinear) 주석 해제.

    (추가) pair_repr 생성 직후 dropout 적용 코드 복구 (과적합 방지).
  
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.dropout = nn.Dropout(dropout)

        # ── Pair Representation: [e_h; e_t; e_h ⊙ e_t] ──
        pair_dim = hidden_size * 3

        # ── Relation Classifier ──
        if classifier_type == "atlop":
            # [수정] 오리지널 DREEAM/ATLOP의 Grouped Bilinear 분류기 세팅
            self.emb_size = hidden_size
            self.block_size = 64  # 논문 디폴트값
            
            # 문맥(rs)과 합쳐지므로 입력 차원이 hidden_size * 2가 됨
            self.head_extractor = nn.Linear(hidden_size * 2, self.emb_size)
            self.tail_extractor = nn.Linear(hidden_size * 2, self.emb_size)
            
            # 최종 분류기 차원 (emb_size * block_size)
            classifier_input_dim = self.emb_size * self.block_size
            self.bilinear = nn.Linear(classifier_input_dim, num_relations, bias=True)

            if threshold_type == "adaptive":
                self.threshold_linear = nn.Linear(classifier_input_dim, 1, bias=True)
                
            if use_evidence:
                self.evidence_head = nn.Linear(classifier_input_dim, max_num_sents)
                
        else:
            # 기본 bilinear classifier (Stage 1)
            # 노트북 Stage1DocREModel과 동일: Dropout → Linear(hidden*3, num_relations)
            self.classifier = nn.Linear(pair_dim, num_relations)
#             self.classifier = nn.Sequential(
#                 nn.Linear(pair_dim, hidden_size),
#                 nn.ReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(hidden_size, num_relations),
#             )
#             if use_evidence:
#                 self.evidence_head = nn.Linear(hidden_size, max_num_sents)

        self.max_num_sents = max_num_sents

    def forward(
        self,
        entity_vectors: torch.Tensor,
        entity_pairs: List[Tuple],
        rs_vectors: Optional[torch.Tensor] = None,  # [수정] 문맥 벡터(rs) 입력 추가!
        num_sents: int = 0,
    ) -> Dict[str, torch.Tensor]:
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
            # # ── ATLOP Classifier ──
            # h_proj = self.head_proj(head_vecs)  # [num_pairs, hidden]
            # t_proj = self.tail_proj(tail_vecs)  # [num_pairs, hidden]

            # # Element-wise product for bilinear-like interaction
            # pair_repr = self.dropout(h_proj * t_proj)  # [num_pairs, hidden]

            # ── ATLOP / DREEAM Classifier ──
            if rs_vectors is not None:
                # [수정] 문맥(rs)이 들어오면 순정 DREEAM 방식 작동!
                h_proj = torch.tanh(self.head_extractor(torch.cat([head_vecs, rs_vectors], dim=-1)))
                t_proj = torch.tanh(self.tail_extractor(torch.cat([tail_vecs, rs_vectors], dim=-1)))
            else:
                # 에러 방지용 (rs가 없을 때)
                h_proj = torch.tanh(self.head_extractor(torch.cat([head_vecs, torch.zeros_like(head_vecs)], dim=-1)))
                t_proj = torch.tanh(self.tail_extractor(torch.cat([tail_vecs, torch.zeros_like(tail_vecs)], dim=-1)))

            # [수정] 블록 단위 연산 (Grouped Bilinear)
            b1 = h_proj.view(-1, self.emb_size // self.block_size, self.block_size)
            b2 = t_proj.view(-1, self.emb_size // self.block_size, self.block_size)
            pair_repr = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)

            pair_repr = self.dropout(pair_repr)

            relation_logits = self.bilinear(pair_repr)  # [num_pairs, num_relations]
            outputs["relation_logits"] = relation_logits

            # Adaptive Threshold
            if self.threshold_type == "adaptive":
                threshold_logits = self.threshold_linear(pair_repr)  # [num_pairs, 1]
                outputs["threshold_logits"] = threshold_logits

            # DREEAM Evidence Head
            if self.use_evidence and num_sents > 0:
                evidence_logits = self.evidence_head(pair_repr)  # [num_pairs, max_sents]
                evidence_logits = evidence_logits[:, :num_sents]  # 실제 문장 수만큼 자르기
                outputs["evidence_logits"] = evidence_logits
                
        else:
            # ── 기본 Bilinear Classifier (Stage 1) ──
            # 노트북과 동일: dropout → classifier
            pair_repr = torch.cat([
                head_vecs,
                tail_vecs,
                head_vecs * tail_vecs,
            ], dim=-1)


            relation_logits = self.classifier(self.dropout(pair_repr))  # [num_pairs, num_relations]

#             relation_logits = self.classifier(pair_repr)

            outputs["relation_logits"] = relation_logits
            
            if self.use_evidence and num_sents > 0:
                evidence_logits = self.evidence_head(head_vecs * tail_vecs) # Stage1의 임시 Evidence
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