"""
============================================================
Postprocessing Layer (postprocessing.py)
============================================================
역할:
  모델의 raw 출력(logits, threshold, evidence)을 실제 저장/평가/활용 가능한
  relation triple 형태로 정제한다.

입력:
  - 모델 출력:
      relation_logits
      threshold_logits (optional, ATLOP adaptive threshold)
      evidence_logits / token_importance (optional)
  - entity pair 정보:
      [(h_id, t_id), ...]
  - relation id ↔ name 매핑
  - entity canonical names / entity types (optional)

출력:
  - List[Dict]
    {
      "h": int,
      "t": int,
      "r": str,
      "score": float,
      "head_name": str (optional),
      "tail_name": str (optional),
      "head_type": str (optional),
      "tail_type": str (optional),
      "evidence": List[int] (optional)
    }

기반 논문:
  - Wang et al. (2019) DocRED evaluation setting
  - Zhou et al. (2021) ATLOP — adaptive threshold
  - Ma et al. (2023) DREEAM — evidence supervision / token importance 활용

구현 원칙:
  - fixed threshold / adaptive threshold 모두 지원
  - adaptive threshold는 logit 기준으로 판정
  - 사용자/평가 저장용 score는 sigmoid probability로 통일
  - no_relation / NA 계열 label은 triple 결과에서 제외
  - duplicate는 (h, t, r) 기준으로 제거하되 최고 score만 유지
  - evidence 정보가 있으면 함께 저장
  - CodaLab 제출용 포맷 생성 함수 제공

TODO (김예슬)

[완료]
  - relation id → label name 매핑 반영
  - fixed / adaptive threshold 후처리 분기 구현
  - no_relation / NA 필터링 추가
  - duplicate 제거 로직 개선 (최고 score 유지)
  - evidence 추출 및 triple에 포함
  - CodaLab 제출 형식 생성 함수 추가

[추가 예정]
  - relation별 custom threshold 실험
  - calibration score / margin score 저장 옵션
  - evidence top-k 저장 및 token index → sentence index 변환
  - official DocRED evaluation script와 직접 연결

담당:
  - 후처리 + 그래프 담당
============================================================
"""

from typing import Dict, List, Tuple, Optional, Any

import torch


# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------
def _to_cpu_tensor(x: Any) -> Optional[torch.Tensor]:
    """Tensor이면 detach + cpu 후 반환, 아니면 None."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu()
    return None


def _normalize_relation_name(rel_name: str) -> str:
    """비교용 relation name 정규화."""
    return rel_name.strip().lower().replace(" ", "_")


def _is_no_relation(rel_name: str) -> bool:
    """
    no_relation / NA 계열 relation 필터링.
    프로젝트마다 표기가 달라질 수 있어 여러 케이스를 허용.
    """
    normalized = _normalize_relation_name(rel_name)
    no_rel_aliases = {
        "na",
        "n/a",
        "no_relation",
        "no-relation",
        "none",
        "null",
        "other",
        "no relation",
    }
    return normalized in no_rel_aliases


def _extract_positive_relations(
    relation_logits_row: torch.Tensor,
    prob_row: torch.Tensor,
    threshold_type: str = "fixed",
    threshold_logit: Optional[torch.Tensor] = None,
    fixed_threshold: float = 0.5,
) -> List[int]:
    """
    pair 하나에 대해 positive relation id 목록 반환.

    규칙:
      - adaptive threshold: raw logit > threshold_logit
      - fixed threshold   : sigmoid(logit) > fixed_threshold
    """
    if threshold_type == "adaptive" and threshold_logit is not None:
        th = threshold_logit.item() if torch.is_tensor(threshold_logit) else float(threshold_logit)
        positive = (relation_logits_row > th).nonzero(as_tuple=True)[0].tolist()
        return positive

    positive = (prob_row > fixed_threshold).nonzero(as_tuple=True)[0].tolist()
    return positive


def _extract_evidence(
    outputs: Dict[str, torch.Tensor],
    pair_idx: int,
    evidence_threshold: float = 0.5,
    top_k: Optional[int] = None,
) -> Optional[List[int]]:
    """
    evidence token/sentence index 추출.

    지원 우선순위:
      1) outputs["token_importance"]  : [num_pairs, seq_len]
      2) outputs["evidence_logits"]   : [num_pairs, seq_len] 또는 [num_pairs, num_sents]

    반환:
      - evidence index list
      - evidence 정보가 없으면 None
    """
    token_importance = outputs.get("token_importance", None)
    evidence_logits = outputs.get("evidence_logits", None)

    if token_importance is not None:
        row = token_importance[pair_idx]
        if top_k is not None and top_k > 0:
            k = min(top_k, row.size(0))
            indices = torch.topk(row, k=k, dim=-1).indices.tolist()
            return sorted(indices)

        indices = (row > evidence_threshold).nonzero(as_tuple=True)[0].tolist()
        return indices

    if evidence_logits is not None:
        row = evidence_logits[pair_idx]
        row_prob = torch.sigmoid(row)

        if top_k is not None and top_k > 0:
            k = min(top_k, row_prob.size(0))
            indices = torch.topk(row_prob, k=k, dim=-1).indices.tolist()
            return sorted(indices)

        indices = (row_prob > evidence_threshold).nonzero(as_tuple=True)[0].tolist()
        return indices

    return None


# -------------------------------------------------------------
# Main Postprocessing
# -------------------------------------------------------------
def postprocess_predictions(
    outputs: Dict[str, torch.Tensor],
    entity_pairs: List[Tuple[int, int]],
    id2rel: Dict[int, str],
    entity_names: Optional[List[str]] = None,
    entity_types: Optional[List[str]] = None,
    threshold_type: str = "fixed",
    fixed_threshold: float = 0.5,
    evidence_threshold: float = 0.5,
    evidence_top_k: Optional[int] = None,
    remove_duplicates: bool = True,
    skip_no_relation: bool = True,
) -> List[Dict]:
    """
    모델 출력 → 구조화된 triple 리스트.

    Args:
      outputs:
        relation_head 출력 dict
        필수:
          - relation_logits : [num_pairs, num_relations]
        선택:
          - threshold_logits
          - evidence_logits
          - token_importance
      entity_pairs:
        [(h_id, t_id), ...]
      id2rel:
        relation id → relation name
      entity_names:
        entity canonical name 리스트
      entity_types:
        entity type 리스트
      threshold_type:
        'fixed' | 'adaptive'
      fixed_threshold:
        fixed threshold 값
      evidence_threshold:
        evidence selection threshold
      evidence_top_k:
        evidence를 top-k로 자를지 여부
      remove_duplicates:
        True면 (h,t,r) 기준 최고 score만 유지
      skip_no_relation:
        True면 no_relation / NA 계열 제거

    Returns:
      List[Dict]
    """
    relation_logits = _to_cpu_tensor(outputs["relation_logits"])
    threshold_logits = _to_cpu_tensor(outputs.get("threshold_logits", None))
    token_importance = _to_cpu_tensor(outputs.get("token_importance", None))
    evidence_logits = _to_cpu_tensor(outputs.get("evidence_logits", None))

    # evidence 추출 함수에서 동일 dict 형태 사용하도록 재구성
    cpu_outputs = {
        "relation_logits": relation_logits,
        "threshold_logits": threshold_logits,
        "token_importance": token_importance,
        "evidence_logits": evidence_logits,
    }

    prob_matrix = torch.sigmoid(relation_logits)
    triples: List[Dict] = []

    for pair_idx, (h_id, t_id) in enumerate(entity_pairs):
        relation_logits_row = relation_logits[pair_idx]
        prob_row = prob_matrix[pair_idx]

        threshold_logit = None
        if threshold_logits is not None:
            threshold_logit = threshold_logits[pair_idx]

        positive_rels = _extract_positive_relations(
            relation_logits_row=relation_logits_row,
            prob_row=prob_row,
            threshold_type=threshold_type,
            threshold_logit=threshold_logit,
            fixed_threshold=fixed_threshold,
        )

        for rel_idx in positive_rels:
            if rel_idx not in id2rel:
                continue

            rel_name = id2rel[rel_idx]

            if skip_no_relation and _is_no_relation(rel_name):
                continue

            triple = {
                "h": int(h_id),
                "t": int(t_id),
                "r": rel_name,
                # 저장 score는 probability로 통일
                "score": float(prob_row[rel_idx].item()),
                # 디버깅/분석용으로 raw logit도 같이 저장
                "logit": float(relation_logits_row[rel_idx].item()),
                "rel_id": int(rel_idx),
            }

            if entity_names is not None:
                if 0 <= h_id < len(entity_names):
                    triple["head_name"] = entity_names[h_id]
                if 0 <= t_id < len(entity_names):
                    triple["tail_name"] = entity_names[t_id]

            if entity_types is not None:
                if 0 <= h_id < len(entity_types):
                    triple["head_type"] = entity_types[h_id]
                if 0 <= t_id < len(entity_types):
                    triple["tail_type"] = entity_types[t_id]

            evidence = _extract_evidence(
                outputs=cpu_outputs,
                pair_idx=pair_idx,
                evidence_threshold=evidence_threshold,
                top_k=evidence_top_k,
            )
            if evidence is not None:
                triple["evidence"] = evidence

            triples.append(triple)

    if not remove_duplicates:
        return triples

    # ---------------------------------------------------------
    # Duplicate 제거: (h, t, r) 기준 최고 score만 유지
    # ---------------------------------------------------------
    best_by_key: Dict[Tuple[int, int, str], Dict] = {}

    for triple in triples:
        key = (triple["h"], triple["t"], triple["r"])

        if key not in best_by_key:
            best_by_key[key] = triple
            continue

        if triple["score"] > best_by_key[key]["score"]:
            best_by_key[key] = triple

    unique_triples = list(best_by_key.values())

    # 정렬: score 내림차순
    unique_triples.sort(key=lambda x: x["score"], reverse=True)
    return unique_triples


# -------------------------------------------------------------
# Batch Helper
# -------------------------------------------------------------
def postprocess_batch_predictions(
    batch_outputs: List[Dict[str, torch.Tensor]],
    batch_entity_pairs: List[List[Tuple[int, int]]],
    id2rel: Dict[int, str],
    batch_entity_names: Optional[List[List[str]]] = None,
    batch_entity_types: Optional[List[List[str]]] = None,
    threshold_type: str = "fixed",
    fixed_threshold: float = 0.5,
    evidence_threshold: float = 0.5,
    evidence_top_k: Optional[int] = None,
    remove_duplicates: bool = True,
    skip_no_relation: bool = True,
) -> List[List[Dict]]:
    """
    문서 batch 단위 postprocessing.
    """
    results = []

    for i, outputs in enumerate(batch_outputs):
        entity_names = None
        entity_types = None

        if batch_entity_names is not None and i < len(batch_entity_names):
            entity_names = batch_entity_names[i]

        if batch_entity_types is not None and i < len(batch_entity_types):
            entity_types = batch_entity_types[i]

        triples = postprocess_predictions(
            outputs=outputs,
            entity_pairs=batch_entity_pairs[i],
            id2rel=id2rel,
            entity_names=entity_names,
            entity_types=entity_types,
            threshold_type=threshold_type,
            fixed_threshold=fixed_threshold,
            evidence_threshold=evidence_threshold,
            evidence_top_k=evidence_top_k,
            remove_duplicates=remove_duplicates,
            skip_no_relation=skip_no_relation,
        )
        results.append(triples)

    return results


# -------------------------------------------------------------
# DocRED / CodaLab Export
# -------------------------------------------------------------
def to_docred_submission(
    triples: List[Dict],
    title: str,
) -> List[Dict]:
    """
    DocRED / CodaLab 제출용 기본 포맷 생성.

    일반 형식:
      {
        "title": doc_title,
        "h_idx": head entity id,
        "t_idx": tail entity id,
        "r": relation name,
        "evidence": [...]
      }
    """
    submission = []

    for triple in triples:
        item = {
            "title": title,
            "h_idx": triple["h"],
            "t_idx": triple["t"],
            "r": triple["r"],
        }

        if "evidence" in triple:
            item["evidence"] = triple["evidence"]

        submission.append(item)

    return submission


def to_docred_submission_batch(
    batch_triples: List[List[Dict]],
    titles: List[str],
) -> List[Dict]:
    """
    여러 문서의 triple을 하나의 제출 리스트로 병합.
    """
    merged = []

    for triples, title in zip(batch_triples, titles):
        merged.extend(to_docred_submission(triples, title))

    return merged