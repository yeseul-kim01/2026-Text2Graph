"""
============================================================
Entity Representation Layer (entity_repr.py)
============================================================
역할: Document Encoder의 token-level hidden state에서 각 entity에
      해당하는 mention span을 추출하고, 여러 mention을 통합하여
      entity-level representation을 생성

INPUT:
  - hidden_states  : [batch, seq_len, hidden_size] — Encoder 출력
  - attention      : [batch, num_heads, seq_len, seq_len] — BERT attention weights
  - entity_spans   : List[List[List[Tuple]]] — batch별 entity별 mention spans

OUTPUT:
  - entity_vectors : List[Tensor[num_entities, hidden_size]] — batch별 entity 벡터
  - entity_attns   : List[Tensor[num_entities, num_heads, seq_len]] — entity별 attention
                     (Localized Context Pooling과 DREEAM evidence에 사용)

============================================================
논문 근거 및 원본 코드 출처:

1. ATLOP (Zhou et al., 2021) — github.com/wzhouad/ATLOP/blob/main/model.py
   - get_hrt() 메서드 (라인 33~74)
   - 핵심: 각 mention의 첫 번째 토큰만 대표로 사용
   - 핵심: 여러 mention을 logsumexp로 통합 (smooth max)
   - 핵심: BERT attention weight도 mention별로 추출하여 저장
   - 핵심: Localized Context Pooling — head/tail attention을 곱해
           문서 전체에서 pair-specific context를 추출

2. DREEAM (Ma et al., 2023) — github.com/YoumiMa/dreeam
   - ATLOP backbone을 그대로 사용 (README: "backbone follows ATLOP")
   - 추가: attention weight를 evidence supervision 신호로 활용
   - 추가: 마지막 3개 layer의 hidden states 평균 (footnote 6)

3. GAIN (Zeng et al., 2020) — github.com/PKUnlp-icler/GAIN
   - mention-level에서 entity-level로의 aggregation 구조 참고
   - 우리 구현에서는 ATLOP 방식(logsumexp)을 기본으로 채택

============================================================
담당: 모델 담당 (김예슬)

TODO(완 - 김예슬):
  - [v1] 기존 span mean pooling 구현 (초기 뼈대)
  - [v2] ATLOP 원본 코드 기반 전면 재작성
    - [변경 1] span 전체 pooling → mention 첫 토큰만 사용 (ATLOP 원본)
    - [변경 2] entity 통합 시 mean → logsumexp (ATLOP 원본)
    - [변경 3] BERT attention weight 추출 추가 (Context Pooling + DREEAM용)
    - [변경 4] Localized Context Pooling 함수 추가 (ATLOP Section 3.2)
============================================================
"""

import torch
import torch.nn as nn
from opt_einsum import contract  # ATLOP 원본에서 사용하는 einsum 라이브러리
from typing import List, Tuple, Optional, Dict


class EntityRepresentation(nn.Module):
    """
    ATLOP/DREEAM 방식의 Entity Representation 모듈.

    ── ATLOP 원본 코드(model.py get_hrt)와의 대응 관계 ──

    ATLOP 원본:
        for e in entity_pos[i]:
            e_emb = sequence_output[i, start + offset]   # mention 첫 토큰
            e_att = attention[i, :, start + offset]       # 해당 토큰의 attention
            ...
            e_emb = torch.logsumexp(torch.stack(e_emb), dim=0)  # mention 통합

    구현:
        동일 로직을 EntityRepresentation 클래스로 모듈화.
        Stage 1에서는 mean pooling, Stage 2+에서는 logsumexp pooling 선택 가능.

    Args:
        hidden_size : 인코더 출력 차원 (768 for BERT-base)
        pooling     : 'mean' (Stage 1) | 'logsumexp' (Stage 2+, ATLOP 방식)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        pooling: str = "mean",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pooling = pooling

    def forward(
        self,
        hidden_states: torch.Tensor,
        entity_spans: List[List[List[Tuple]]],
        attention: Optional[torch.Tensor] = None,
    ) -> Dict[str, list]:
        """
        ── 전체 흐름 ──

        Step 1: 각 mention의 첫 번째 토큰 hidden state 추출
                (ATLOP model.py 라인 42: sequence_output[i, start + offset])
        Step 2: 동일 entity의 여러 mention을 logsumexp/mean으로 통합
                (ATLOP model.py 라인 50: torch.logsumexp(torch.stack(e_emb), dim=0))
        Step 3: (선택) BERT attention weight도 mention별로 추출
                (ATLOP model.py 라인 43: attention[i, :, start + offset])

        INPUT:
          - hidden_states : [batch, seq_len, hidden_size]
                            Document Encoder(BERT)의 출력.
                            DREEAM에서는 마지막 3개 layer 평균을 사용.
          - entity_spans  : batch별 -> entity별 -> mention별 [(start, end), ...]
                            Preprocessing에서 생성한 mention 위치 정보.
                            ⚠ start는 subword 기준 인덱스 (BERT tokenizer 기준)
          - attention     : [batch, num_heads, seq_len, seq_len] (Optional)
                            BERT의 attention weight. Localized Context Pooling과
                            DREEAM evidence head에서 사용.
                            Stage 1에서는 None 가능.

        OUTPUT:
          Dict with:
          - 'entity_vectors': List[Tensor[num_entities, hidden_size]]
              각 entity의 통합된 representation 벡터.
              entity가 20개면 [20, 768] shape의 텐서.
          - 'entity_attns': List[Tensor[num_entities, num_heads, seq_len]]
              각 entity의 attention pattern (Context Pooling용).
              attention이 None이면 이 키도 없음.
        """
        batch_entity_vectors = []
        batch_entity_attns = []

        for b in range(hidden_states.size(0)):
            h = hidden_states[b]  # [seq_len, hidden_size]
            entities = entity_spans[b]  # entity별 mention 리스트

            # attention이 있으면 추출 준비
            has_attn = (attention is not None)
            if has_attn:
                attn = attention[b]  # [num_heads, seq_len, seq_len]
                n_heads = attn.size(0)
                seq_len = attn.size(-1)

            entity_vecs = []
            entity_atts = []

            for mention_list in entities:
                # ─────────────────────────────────────────────
                # mention이 없는 경우 (truncation으로 잘린 entity)
                # ATLOP 원본 model.py 라인 53~54:
                #   e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                #   e_att = torch.zeros(h, c).to(attention)
                # ─────────────────────────────────────────────
                if len(mention_list) == 0:
                    entity_vecs.append(torch.zeros(self.hidden_size, device=h.device))
                    if has_attn:
                        entity_atts.append(torch.zeros(n_heads, seq_len, device=h.device))
                    continue

                # ─────────────────────────────────────────────
                # Step 1: 각 mention의 대표 토큰 추출
                #
                # [ATLOP 원본 코드 — model.py 라인 42~43]
                #   e_emb.append(sequence_output[i, start + offset])
                #   e_att.append(attention[i, :, start + offset])
                #
                # ※ ATLOP은 mention의 **첫 번째 토큰**만 사용한다.
                #   span 전체를 pooling하지 않는다!
                #   이유: BERT에서 첫 토큰([CLS] 또는 entity 첫 subword)이
                #         해당 span의 의미를 가장 잘 압축하고 있기 때문.
                #         (BERT 논문에서 [CLS]가 문장 의미를 대표하는 것과 동일 원리)
                #
                # ※ offset = 1 (BERT의 [CLS] 토큰 보정)
                #   ATLOP 원본에서는 entity_pos가 [CLS] 제외 기준이라 +1 보정.
                #   우리 preprocessing은 subword 기준이므로 보정 불필요.
                #   만약 preprocessing에서 [CLS] 제외 기준이면 여기서 +1 해야 함.
                # ─────────────────────────────────────────────
                mention_embs = []
                mention_atts = []

                for start, end in mention_list:
                    # 범위 체크 (truncation 대비)
                    # ATLOP 원본 라인 44: if start + offset < c:
                    if start >= h.size(0):
                        continue

                    # mention 첫 번째 토큰의 hidden state
                    # (ATLOP 원본: sequence_output[i, start + offset])
                    mention_embs.append(h[start])

                    # mention 첫 번째 토큰의 attention weight
                    # (ATLOP 원본: attention[i, :, start + offset])
                    if has_attn:
                        mention_atts.append(attn[:, start])  # [num_heads, seq_len]

                # mention을 하나도 못 추출한 경우
                if len(mention_embs) == 0:
                    entity_vecs.append(torch.zeros(self.hidden_size, device=h.device))
                    if has_attn:
                        entity_atts.append(torch.zeros(n_heads, seq_len, device=h.device))
                    continue

                # ─────────────────────────────────────────────
                # Step 2: 여러 mention을 하나의 entity vector로 통합
                #
                # [ATLOP 원본 코드 — model.py 라인 46~50]
                #   if len(e) > 1:
                #       e_emb, e_att = [], []
                #       for start, end in e:
                #           e_emb.append(sequence_output[i, start + offset])
                #       e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                #
                # ※ LogSumExp pooling (Stage 2+, ATLOP 방식):
                #   log(Σ exp(m_i)) — smooth max approximation.
                #   여러 mention 중 가장 강한 신호를 가진 mention에
                #   자연스럽게 가중치를 부여한다.
                #   ATLOP 논문 Table 3: mean → logsumexp로 약 1.5 F1 향상.
                #
                # ※ Mean pooling (Stage 1, Baseline):
                #   모든 mention을 균등하게 평균.
                #   단순하지만 비교 기준(baseline)으로 사용.
                #
                # ※ ATLOP 원본에서는 mention이 1개면 logsumexp 없이 그냥 사용.
                #   mention이 2개 이상일 때만 logsumexp 적용.
                #   우리도 동일하게 처리.
                # ─────────────────────────────────────────────
                mention_stack = torch.stack(mention_embs, dim=0)  # [num_mentions, hidden_size]

                if len(mention_embs) == 1:
                    # mention 1개: 그대로 사용 (ATLOP 원본과 동일)
                    entity_vec = mention_stack.squeeze(0)  # [hidden_size]
                else:
                    # mention 2개 이상: pooling 적용
                    if self.pooling == "logsumexp":
                        # ── ATLOP 방식: LogSumExp ──
                        # ATLOP model.py 라인 50:
                        #   e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        entity_vec = torch.logsumexp(mention_stack, dim=0)  # [hidden_size]
                    else:
                        # ── Baseline: Mean Pooling ──
                        entity_vec = mention_stack.mean(dim=0)  # [hidden_size]

                entity_vecs.append(entity_vec)

                # ─────────────────────────────────────────────
                # Step 3: Attention weight 통합
                #
                # [ATLOP 원본 코드 — model.py 라인 51]
                #   e_att = torch.stack(e_att, dim=0).mean(0)
                #
                # ※ attention은 항상 mean으로 통합한다 (logsumexp 아님).
                #   이유: attention weight는 확률 분포(합≈1)이므로
                #         logsumexp보다 mean이 의미적으로 적합.
                # ─────────────────────────────────────────────
                if has_attn and len(mention_atts) > 0:
                    attn_stack = torch.stack(mention_atts, dim=0)  # [num_mentions, heads, seq]
                    entity_att = attn_stack.mean(0)  # [num_heads, seq_len]
                    entity_atts.append(entity_att)

            # ── batch 내 하나의 문서에 대한 entity 텐서 구성 ──
            if len(entity_vecs) > 0:
                batch_entity_vectors.append(torch.stack(entity_vecs, dim=0))
            else:
                batch_entity_vectors.append(
                    torch.zeros(1, self.hidden_size, device=h.device)
                )

            if has_attn and len(entity_atts) > 0:
                batch_entity_attns.append(torch.stack(entity_atts, dim=0))

        # ── 출력 구성 ──
        result = {"entity_vectors": batch_entity_vectors}
        if len(batch_entity_attns) > 0:
            result["entity_attns"] = batch_entity_attns

        return result


# ════════════════════════════════════════════════════════════════
# Localized Context Pooling (ATLOP Section 3.2)
# ════════════════════════════════════════════════════════════════
#
# ATLOP 논문의 핵심 기여 중 하나.
# head entity와 tail entity의 attention을 element-wise로 곱해서
# 해당 pair에 특화된 "context" representation을 추출한다.
#
# 직관: head가 주목하는 토큰 AND tail이 주목하는 토큰 → 둘 다 중요한 토큰
# → 그 토큰들의 가중합이 이 pair의 context
#
# [ATLOP 원본 코드 — model.py 라인 69~73]
#   h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
#   t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
#   ht_att = (h_att * t_att).mean(1)
#   ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
#   rs = contract("ld,rl->rd", sequence_output[i], ht_att)
#
# 이 context vector(rs)는 relation_head.py에서 head/tail vector와 함께
# concat되어 최종 classification에 사용된다:
#   hs = tanh(W_h * [e_h ; rs])   ← head + context
#   ts = tanh(W_t * [e_t ; rs])   ← tail + context
#   logits = bilinear(hs, ts)
# ════════════════════════════════════════════════════════════════
def localized_context_pooling(
    hidden_states: torch.Tensor,
    entity_attns: torch.Tensor,
    entity_pairs: List[Tuple],
) -> torch.Tensor:
    """
    ATLOP의 Localized Context Pooling.
    각 (head, tail) pair에 대해 pair-specific context vector를 생성.

    ── ATLOP 원본 대응 (model.py 라인 69~73) ──
    h_att = entity_atts[head_ids]            # head의 attention
    t_att = entity_atts[tail_ids]            # tail의 attention
    ht_att = (h_att * t_att).mean(1)         # head*tail attention (head별 평균)
    ht_att = ht_att / (ht_att.sum + 1e-5)    # 정규화하여 확률분포로 만듦
    rs = einsum("ld,rl->rd", hidden, ht_att) # attention-weighted sum

    INPUT:
      - hidden_states : [seq_len, hidden_size] — 단일 문서의 토큰 벡터
      - entity_attns  : [num_entities, num_heads, seq_len] — entity별 attention
      - entity_pairs  : [(h_id, t_id), ...] — 평가할 pair 목록

    OUTPUT:
      - context_vectors : [num_pairs, hidden_size]
        각 pair에 특화된 context representation.
        relation_head.py에서 head/tail vector와 concat하여 사용:
          hs = tanh(W_h * [e_h ; context])
          ts = tanh(W_t * [e_t ; context])
    """
    if len(entity_pairs) == 0:
        return torch.zeros(0, hidden_states.size(-1), device=hidden_states.device)

    head_ids = [p[0] for p in entity_pairs]
    tail_ids = [p[1] for p in entity_pairs]

    # head/tail의 attention 추출
    # ATLOP 원본: h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
    h_att = entity_attns[head_ids]  # [num_pairs, num_heads, seq_len]
    t_att = entity_attns[tail_ids]  # [num_pairs, num_heads, seq_len]

    # head와 tail attention을 element-wise 곱 → head 차원 평균 → 정규화
    #
    # ATLOP 원본:
    #   ht_att = (h_att * t_att).mean(1)                    # [num_pairs, seq_len]
    #   ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)  # 정규화
    #
    # 의미: head가 주목하는 토큰 * tail이 주목하는 토큰
    #       → 양쪽 모두에게 중요한 토큰이 높은 가중치를 받음
    #       → 이 가중치로 문서 토큰의 가중합을 구하면 pair-specific context
    ht_att = (h_att * t_att).mean(1)  # [num_pairs, seq_len]
    ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)  # 정규화

    # Attention-weighted sum으로 context vector 생성
    #
    # ATLOP 원본: rs = contract("ld,rl->rd", sequence_output[i], ht_att)
    #
    # einsum 설명:
    #   "ld,rl->rd" 는:
    #   hidden_states [L, D] × ht_att [R, L] → context [R, D]
    #   L=seq_len, D=hidden_size, R=num_pairs
    #   즉, 각 pair(R)에 대해 문서 전체 토큰(L)의 가중합(D차원)을 구함
    context_vectors = contract("ld,rl->rd", hidden_states, ht_att)
    # [num_pairs, hidden_size]

    return context_vectors
