# Document-level Relation Extraction & Knowledge Graph System

DocRED 기반 End-to-End 정보 추출 파이프라인.  
ATLOP + DREEAM + GAIN 구조 기반 **3단계 Incremental Stacking** 설계.

## 프로젝트 구조

```
docre-kg/
├── configs/                    # Stage별 YAML 설정
│   ├── stage1.yaml             # Baseline (BERT + Mean + BCE)
│   ├── stage2.yaml             # + ATLOP + DREEAM
│   └── stage3.yaml             # + GAIN-lite GNN
├── src/                        # 핵심 소스 코드 (모듈별 분리)
│   ├── __init__.py
│   ├── preprocessing.py        # DocRED 데이터 전처리
│   ├── encoder.py              # BERT Document Encoder
│   ├── entity_repr.py          # Entity Representation (Mean/LogSumExp)
│   ├── relation_head.py        # Relation Head (Fixed/ATLOP/DREEAM)
│   ├── graph_encoder.py        # GNN Graph Reasoning (GAIN-lite)
│   ├── model.py                # 통합 모델 (Stage flag 분기)
│   ├── losses.py               # 손실 함수 (BCE/ATL/Evidence)
│   ├── evaluation.py           # 평가 지표 (F1/Ign F1/Evidence F1)
│   ├── postprocessing.py       # Triple 정제
│   ├── kg_builder.py           # Neo4j KG 구축
│   └── utils.py                # 유틸리티
├── scripts/                    # 실행 스크립트
│   ├── train.py                # 학습
│   ├── evaluate.py             # 평가
│   ├── infer_evidence.py       # DREEAM silver evidence 생성
│   └── build_kg.py             # Neo4j KG 구축
├── notebooks/
│   └── run_pipeline.ipynb      # Colab 실행용 노트북
├── data/
│   ├── docred/                 # DocRED JSON 파일 (다운로드 필요)
│   └── meta/                   # rel2id.json 등
├── requirements.txt
└── README.md
```

## 빠른 시작

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 데이터 다운로드
[DocRED (HuggingFace)](https://huggingface.co/datasets/thunlp/docred)에서 다운로드 후 `data/docred/`에 배치.

### 3. 학습 실행
```bash
# Stage 1: Baseline
python scripts/train.py --config configs/stage1.yaml

# Stage 2: ATLOP + DREEAM
python scripts/train.py --config configs/stage2.yaml

# Stage 3: + GAIN-lite GNN
python scripts/train.py --config configs/stage3.yaml
```

### 4. 평가
```bash
python scripts/evaluate.py --config configs/stage1.yaml \
    --checkpoint checkpoints/stage1/best_model.pt
```

## 3단계 Incremental Stacking

| Stage | 구성 | 핵심 변경 |
|-------|------|----------|
| Stage 1 | BERT + Mean Pool + BCE | Baseline 수립 |
| Stage 2 | + LogSumExp + ATLOP + DREEAM | Classifier 강화 |
| Stage 3 | + GCN/GAT (GAIN-lite) | Representation 강화 |

##  역할

| 담당 | 파일 | 역할 |
|------|------|------|
| 전처리 | `preprocessing.py` | DocRED 파싱, mention alignment, pair 생성 |
| 모델 | `encoder.py`, `entity_repr.py`, `relation_head.py`, `losses.py` | BERT, pooling, ATLOP/DREEAM |
| 그래프 | `graph_encoder.py`, `postprocessing.py`, `kg_builder.py` | GNN, Neo4j, multi-hop |

## 참고 논문 & GitHub
- [ATLOP](https://github.com/wzhouad/ATLOP) — Zhou et al. (2021)
- [DREEAM](https://github.com/YoumiMa/dreeam) — Ma et al. (2023)
- [GAIN](https://github.com/PKUnlp-icler/GAIN) — Zeng et al. (2020)
- [SSAN](https://github.com/BenfengXu/SSAN) — Xu et al. (2021)


## 남은 작업 
- Graph Encoder 의 분기 처리, stage 4 생성하기 