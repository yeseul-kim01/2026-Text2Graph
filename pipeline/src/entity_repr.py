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
            각 batch 원소마다 entity 수가 다를 수 있으므로 List로 반환
        """
        batch_entity_vectors = []

        for b in range(hidden_states.size(0)):
            h = hidden_states[b]  # [seq_len, hidden_size]
            entities = entity_spans[b]  # [num_entities][num_mentions]
            entity_vecs = []

            for mention_list in entities:
                if len(mention_list) == 0:
                    # mention이 없는 경우 (truncation으로 잘림)
                    entity_vecs.append(torch.zeros(self.hidden_size, device=h.device))
                    continue

                # ── Mention Representation ──
                mention_vecs = []
                for start, end in mention_list:
                    # 범위 체크
                    start = max(0, start)
                    end = min(end, h.size(0))
                    if start >= end:
                        continue

                    span_hidden = h[start:end]  # [span_len, hidden_size]

                    if self.pooling == "logsumexp":
                        # LogSumExp pooling (ATLOP)
                        # log(sum(exp(h_i))) — smooth max approximation
                        m_vec = torch.logsumexp(span_hidden, dim=0)  # [hidden_size]
                    else:
                        # Mean pooling (Stage 1 baseline)
                        m_vec = span_hidden.mean(dim=0)  # [hidden_size]

                    mention_vecs.append(m_vec)

                if len(mention_vecs) == 0:
                    entity_vecs.append(torch.zeros(self.hidden_size, device=h.device))
                    continue

                # ── Entity Representation 통합 ──
                # 동일 entity의 여러 mention vector를 평균
                mention_stack = torch.stack(mention_vecs, dim=0)  # [num_mentions, hidden_size]
                entity_vec = mention_stack.mean(dim=0)  # [hidden_size]
                entity_vecs.append(entity_vec)

            # 모든 entity를 하나의 텐서로
            if len(entity_vecs) > 0:
                batch_entity_vectors.append(torch.stack(entity_vecs, dim=0))
            else:
                batch_entity_vectors.append(
                    torch.zeros(1, self.hidden_size, device=h.device)
                )

        return batch_entity_vectors
