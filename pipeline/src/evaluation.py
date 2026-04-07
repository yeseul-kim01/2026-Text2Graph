"""
============================================================
Evaluation Module (evaluation.py)
============================================================
역할: DocRE 커뮤니티 표준 평가 지표 계산

지표:
  - Micro F1    : 모든 relation 예측을 동등 가중치로 평가
  - Ign F1      : 훈련/테스트 공유 triple 제외 F1
  - Evidence F1 : 근거 문장 예측 F1 (DREEAM)
  - Intra/Inter F1 : 문장 내/간 관계별 F1 (분석용)

INPUT:  예측 결과 리스트, 정답 리스트
OUTPUT: Dict with F1 scores

기반 논문: Yao et al. (2019) DocRED 공식 평가 코드

담당: 전체 (공용)

TODO ( 수정 포인트):
  - [ ] CodaLab 제출 형식 JSON 생성 함수
  - [ ] Intra/Inter F1 분리 계산 구현
============================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Set


def compute_micro_f1(
    predictions: List[Dict],
    gold_labels: List[Dict],
    ignore_train_triples: Set[Tuple] = None,
) -> Dict[str, float]:
    """
    Micro Precision, Recall, F1 계산.

    INPUT:
      predictions : [{'h': int, 't': int, 'r': str, 'score': float}, ...]
      gold_labels : [{'h': int, 't': int, 'r': str}, ...]
      ignore_train_triples : Ign F1 계산용 (훈련셋에도 있는 triple)

    OUTPUT:
      Dict with 'precision', 'recall', 'f1', 'ign_f1'
    """
    pred_set = set()
    for p in predictions:
        pred_set.add((p["h"], p["t"], p["r"]))

    gold_set = set()
    for g in gold_labels:
        gold_set.add((g["h"], g["t"], g["r"]))

    # Micro F1
    tp = len(pred_set & gold_set)
    precision = tp / max(len(pred_set), 1)
    recall = tp / max(len(gold_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Ign F1 (공유 triple 제외)
    ign_f1 = f1
    if ignore_train_triples is not None:
        ign_gold = gold_set - ignore_train_triples
        ign_pred = pred_set - ignore_train_triples
        ign_tp = len(ign_pred & ign_gold)
        ign_p = ign_tp / max(len(ign_pred), 1)
        ign_r = ign_tp / max(len(ign_gold), 1)
        ign_f1 = 2 * ign_p * ign_r / max(ign_p + ign_r, 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ign_f1": ign_f1,
    }


def evaluate_re(
    all_predictions: List[List[Dict]],
    all_gold_labels: List[List[Dict]],
) -> Dict[str, float]:
    """
    전체 데이터셋에 대한 RE 평가.

    INPUT:
      all_predictions : 문서별 예측 리스트
      all_gold_labels : 문서별 정답 리스트
    OUTPUT:
      Dict with 'f1', 'ign_f1', 'precision', 'recall'
    """
    flat_preds = [p for doc in all_predictions for p in doc]
    flat_golds = [g for doc in all_gold_labels for g in doc]
    return compute_micro_f1(flat_preds, flat_golds)


def evaluate_evidence(
    pred_evidence: List[Dict],
    gold_evidence: List[Dict],
) -> Dict[str, float]:
    """
    Evidence F1 계산 (DREEAM).

    INPUT:
      pred_evidence : [{pair_key: [sent_ids]}, ...]
      gold_evidence : [{pair_key: [sent_ids]}, ...]
    OUTPUT:
      Dict with 'evidence_f1'
    """
    tp, fp, fn = 0, 0, 0
    for pred, gold in zip(pred_evidence, gold_evidence):
        for key in set(list(pred.keys()) + list(gold.keys())):
            p_sents = set(pred.get(key, []))
            g_sents = set(gold.get(key, []))
            tp += len(p_sents & g_sents)
            fp += len(p_sents - g_sents)
            fn += len(g_sents - p_sents)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"evidence_f1": f1}
