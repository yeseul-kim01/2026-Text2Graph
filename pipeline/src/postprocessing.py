"""
============================================================
Postprocessing Layer (postprocessing.py)
============================================================
역할: 모델 예측 결과를 실제 저장/활용 가능한 triple로 정제

INPUT:
  - 모델 출력 (relation_logits, threshold_logits)
  - Entity 정보 (canonical names, types)

OUTPUT:
  - List of (head, relation, tail, score, evidence) triples

기반 논문: Wang et al. (2019) ATLOP — adaptive threshold

담당: 후처리 + 그래프 담당

TODO ( 수정 포인트):
  - [ ] Relation label → 사람이 읽을 수 있는 이름 매핑
  - [ ] Duplicate 제거 로직 강화
  - [ ] CodaLab 제출 형식 생성
============================================================
"""

import torch
from typing import Dict, List, Tuple, Optional


def postprocess_predictions(
    outputs: Dict[str, torch.Tensor],
    entity_pairs: List[Tuple],
    id2rel: Dict[int, str],
    entity_names: Optional[List[str]] = None,
    threshold_type: str = "fixed",
    fixed_threshold: float = 0.5,
) -> List[Dict]:
    """
    모델 출력 → 구조화된 triple 리스트.

    INPUT:
      - outputs        : 모델 forward 출력
      - entity_pairs   : [(h_id, t_id), ...]
      - id2rel         : relation id → name 매핑
      - entity_names   : entity canonical name 리스트
      - threshold_type : 'fixed' | 'adaptive'

    OUTPUT:
      - List of {head, tail, relation, score, h_id, t_id}
    """
    relation_logits = outputs["relation_logits"]  # [num_pairs, num_rels]
    probs = torch.sigmoid(relation_logits)

    triples = []

    for pair_idx, (h_id, t_id) in enumerate(entity_pairs):
        if threshold_type == "adaptive" and "threshold_logits" in outputs:
            th = outputs["threshold_logits"][pair_idx].item()
            scores = relation_logits[pair_idx]
            positive_rels = (scores > th).nonzero(as_tuple=True)[0]
        else:
            scores = probs[pair_idx]
            positive_rels = (scores > fixed_threshold).nonzero(as_tuple=True)[0]

        for rel_idx in positive_rels:
            rel_idx = rel_idx.item()
            if rel_idx in id2rel:
                triple = {
                    "h": h_id,
                    "t": t_id,
                    "r": id2rel[rel_idx],
                    "score": scores[rel_idx].item(),
                }
                if entity_names:
                    triple["head_name"] = entity_names[h_id]
                    triple["tail_name"] = entity_names[t_id]
                triples.append(triple)

    # ── Duplicate 제거 ──
    seen = set()
    unique_triples = []
    for t in triples:
        key = (t["h"], t["t"], t["r"])
        if key not in seen:
            seen.add(key)
            unique_triples.append(t)

    return unique_triples
