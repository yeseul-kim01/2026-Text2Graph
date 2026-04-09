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

from typing import Dict, List, Tuple, Set


def _safe_div(a, b):
    return a / b if b > 0 else 0.0


def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# -------------------------------------------------------------
# Micro / Ign F1 (DocRED 공식 방식)
# -------------------------------------------------------------
def compute_micro_f1(
    predictions: List[Dict],
    gold_labels: List[Dict],
    train_facts: Set[Tuple] = None,
) -> Dict[str, float]:

    pred_set = set((p["title"], p["h"], p["t"], p["r"]) for p in predictions)
    gold_set = set((g["title"], g["h"], g["t"], g["r"]) for g in gold_labels)

    # Micro F1
    tp = len(pred_set & gold_set)
    precision = _safe_div(tp, len(pred_set))
    recall = _safe_div(tp, len(gold_set))
    f1 = _f1(precision, recall)

    # Ign F1 (DocRED 방식)
    ign_f1 = f1
    if train_facts is not None:

        # gold만 filtering
        ign_gold = set(
            (t, h, ta, r)
            for (t, h, ta, r) in gold_set
            if (h, ta, r) not in train_facts
        )

        ign_tp = len(pred_set & ign_gold)

        ign_precision = _safe_div(ign_tp, len(pred_set))
        ign_recall = _safe_div(ign_tp, len(ign_gold))
        ign_f1 = _f1(ign_precision, ign_recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ign_f1": ign_f1,
    }


# -------------------------------------------------------------
# Evidence F1 (DREEAM 방식)
# -------------------------------------------------------------
def compute_evidence_f1(
    predictions: List[Dict],
    gold_labels: List[Dict],
) -> Dict[str, float]:

    pred_set = set()
    gold_set = set()

    for p in predictions:
        for e in p.get("evidence", []):
            pred_set.add((p["title"], p["h"], p["t"], p["r"], e))

    for g in gold_labels:
        for e in g.get("evidence", []):
            gold_set.add((g["title"], g["h"], g["t"], g["r"], e))

    tp = len(pred_set & gold_set)
    precision = _safe_div(tp, len(pred_set))
    recall = _safe_div(tp, len(gold_set))
    f1 = _f1(precision, recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# -------------------------------------------------------------
# Intra / Inter F1
# -------------------------------------------------------------
def compute_inter_intra_f1(
    predictions: List[Dict],
    gold_labels: List[Dict],
    sent_info: Dict,
):

    pred_intra, pred_inter = set(), set()
    gold_intra, gold_inter = set(), set()

    for g in gold_labels:
        key = (g["title"], g["h"], g["t"], g["r"])

        if sent_info[g["title"]][g["h"]] & sent_info[g["title"]][g["t"]]:
            gold_intra.add(key)
        else:
            gold_inter.add(key)

    for p in predictions:
        key = (p["title"], p["h"], p["t"], p["r"])

        if sent_info[p["title"]][p["h"]] & sent_info[p["title"]][p["t"]]:
            pred_intra.add(key)
        else:
            pred_inter.add(key)

    # intra
    intra_tp = len(pred_intra & gold_intra)
    intra_p = _safe_div(intra_tp, len(pred_intra))
    intra_r = _safe_div(intra_tp, len(gold_intra))
    intra_f1 = _f1(intra_p, intra_r)

    # inter
    inter_tp = len(pred_inter & gold_inter)
    inter_p = _safe_div(inter_tp, len(pred_inter))
    inter_r = _safe_div(inter_tp, len(gold_inter))
    inter_f1 = _f1(inter_p, inter_r)

    return {
        "intra_f1": intra_f1,
        "inter_f1": inter_f1,
    }


# -------------------------------------------------------------
# 전체 평가
# -------------------------------------------------------------
def evaluate_re(
    all_predictions: List[List[Dict]],
    all_gold_labels: List[List[Dict]],
    train_facts: Set[Tuple] = None,
) -> Dict[str, float]:

    flat_preds = []
    flat_golds = []

    for doc_preds in all_predictions:
        flat_preds.extend(doc_preds)

    for doc_golds in all_gold_labels:
        flat_golds.extend(doc_golds)

    return compute_micro_f1(flat_preds, flat_golds, train_facts)