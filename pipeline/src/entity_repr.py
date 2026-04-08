"""
============================================================
Entity Representation Layer (entity_repr.py)
============================================================
역할: Document Encoder의 token-level hidden state에서 각 entity에
      해당하는 mention span을 추출하고, 여러 mention을 통합하여
      entity-level representation을 생성

INPUT:
  - hidden_states  : [batch, seq_len, hidden_size] — Encoder 출력
  - entity_spans   : List[List[List[Tuple]]] — batch별 entity별 mention spans

OUTPUT:
  - entity_vectors : List[Tensor[num_entities, hidden_size]] — batch별 entity 벡터

기반 논문:
  - Zhou et al. (2021) ATLOP — LogSumExp pooling
  - Zeng et al. (2020) GAIN — mention-level pooling

담당: 모델 담당

TODO ( 수정 포인트):
  - [ ] Attention-based mention aggregation 구현 (GAIN 스타일)
  - [ ] Entity marker 기반 representation 추가
============================================================
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class EntityRepresentation(nn.Module):
    """
    Mention span → Entity vector 변환 모듈.

    Args:
        hidden_size : 인코더 출력 차원 (768)
        pooling     : 'mean' (Stage 1) | 'logsumexp' (Stage 2+)
    """

    def __init__(self, hidden_size: int = 768, pooling: str = "mean"):
        super().__init__()
        self.hidden_size = hidden_size
        self.pooling = pooling

        # LogSumExp 안정화를 위한 learnable temperature (optional)
        if pooling == "logsumexp":
            self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        entity_spans: List[List[Tuple]],
    ) -> List[torch.Tensor]:
        """
        INPUT:
          - hidden_states : [batch, seq_len, hidden_size]
          - entity_spans  : batch별 [entity별 [(start, end), ...]]

        OUTPUT:
          - entity_vectors: List[Tensor[num_entities, hidden_size]]
        """
        batch_entity_vectors = []

        for b in range(hidden_states.size(0)):
            h = hidden_states[b]  # [seq_len, hidden_size]
            entities = entity_spans[b]  # [num_entities][num_mentions]
            entity_vecs = []

            for mention_list in entities:
                if len(mention_list) == 0:
                    entity_vecs.append(torch.zeros(self.hidden_size, device=h.device))
                    continue

                # ── Mention Representation ──
                mention_vecs = []
                for start, end in mention_list:
                    # [핵심 수정 1] 오리지널 DREEAM: mention의 첫 번째 토큰만 대표로 사용
                    start = max(0, start)
                    if start < h.size(0):
                        mention_vecs.append(h[start])

                if len(mention_vecs) == 0:
                    entity_vecs.append(torch.zeros(self.hidden_size, device=h.device))
                    continue

                # 여러 개의 mention을 하나로 묶기
                mention_stack = torch.stack(mention_vecs, dim=0)  # [num_mentions, hidden_size]

                # ── Entity Representation 통합 ──
                # [핵심 수정 2] forward 안에서 if문으로 Stage 1과 Stage 2를 나눕니다!
                if self.pooling == "logsumexp":
                    # Stage 2 (DREEAM): LogSumExp로 뭉치기
                    entity_vec = torch.logsumexp(mention_stack, dim=0) 
                else:
                    # Stage 1 (Baseline): 평균(mean)으로 뭉치기
                    entity_vec = mention_stack.mean(dim=0)

                entity_vecs.append(entity_vec)

            # 모든 entity를 하나의 텐서로
            if len(entity_vecs) > 0:
                batch_entity_vectors.append(torch.stack(entity_vecs, dim=0))
            else:
                batch_entity_vectors.append(
                    torch.zeros(1, self.hidden_size, device=h.device)
                )

        return batch_entity_vectors
