"""
============================================================
Document-level Relation Extraction & Knowledge Graph System
============================================================
패키지 구조:
  src/
    preprocessing.py   — DocRED 데이터 전처리
    encoder.py         — BERT Document Encoder
    entity_repr.py     — Entity Representation (Mean / LogSumExp)
    relation_head.py   — Relation Classification (Fixed / ATLOP / DREEAM)
    graph_encoder.py   — GNN Graph Reasoning (GAIN-lite)
    model.py           — 전체 모델 (Stage 1/2/3 통합 forward)
    postprocessing.py  — Triple 정제 및 필터링
    kg_builder.py      — Neo4j Knowledge Graph 구축
    losses.py          — 손실 함수 (BCE / ATLOP ATL / Evidence KL)
    evaluation.py      — 평가 지표 (Micro F1, Ign F1, Evidence F1)
    utils.py           — 유틸리티 함수
============================================================
"""

from .model import DocREModel
from .preprocessing import DocREDDataset, docred_collate_fn
from .evaluation import evaluate_re, evaluate_evidence

__all__ = [
    "DocREModel",
    "DocREDDataset",
    "docred_collate_fn",
    "evaluate_re",
    "evaluate_evidence",
]
