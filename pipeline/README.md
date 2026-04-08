# 2026-Text2Graph Pipeline

**DocRED 기반 Document-level Relation Extraction → Knowledge Graph 구축 파이프라인**

BERT 인코더부터 KG 저장까지 4단계 Incremental Stacking 구조로 설계된 NLP 파이프라인입니다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [데이터셋 DocRED](#3-데이터셋-docred)
4. [전체 파이프라인 흐름](#4-전체-파이프라인-흐름)
5. [Stage 1 — Baseline RE 상세](#5-stage-1--baseline-re-상세)
6. [Stage 2 — ATLOP + DREEAM](#6-stage-2--atlop--dreeam)
7. [Stage 3 — GAIN-lite GNN](#7-stage-3--gain-lite-gnn)
8. [Stage 4 — Graph U-Net](#8-stage-4--graph-u-net)
9. [Stage 간 비교](#9-stage-간-비교)
10. [환경 설정 및 실행](#10-환경-설정-및-실행)
11. [참고 논문](#11-참고-논문)

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 문서(Document) 내 다중 문장에 걸친 Entity 간 관계를 추출하여 Knowledge Graph 구축 |
| **데이터셋** | DocRED (Document-level Relation Extraction Dataset) |
| **기반 논문** | BERT, ATLOP (Zhou 2021), DREEAM (Ma 2023), GAIN (Zeng 2020), Graph U-Net (Gao 2019) |
| **구조** | 4단계 Incremental Stacking (Stage 1 → 2 → 3 → 4) |
| **언어** | Python 3.8+, PyTorch, HuggingFace Transformers |

### Incremental Stacking 전략

각 Stage는 이전 Stage의 결과물을 그대로 이어받아 **점진적으로 성능을 향상**시키는 구조입니다.

```
Stage 1  ──────►  Stage 2  ──────►  Stage 3  ──────►  Stage 4
Baseline          ATLOP             + GAIN-lite         + Graph U-Net
(BERT +           + DREEAM          GNN                 (U-Net 구조
 MeanPool +       (LogSumExp +      (Flat GNN           계층적 풀링 +
 BCE Loss)         Adaptive TH +    Message             글로벌 추론)
                   Evidence Head)   Passing)
                       │                                    │
                       └──────── Ablation 비교 ─────────────┘
                              (둘 다 Stage 2 기반)
```

### 팀 담당 분배

| 역할 | 이름 |
|------|------|
| 총괄 PM | `김예슬` |
| Stage 1 담당 | `이수민` |
| Stage 2 담당 | `박재윤` |
| Stage 3/4 담당 | `박정현` |

#### Stage별 세부 담당

**Stage 1** — 담당: `이수민`
| 레이어 담당 | 파일 | 역할 |
|------------|------|------|
| `이수민` | `파일명.py` | 담당 역할 |

**Stage 2** — 담당: `박재윤`
| 레이어 담당 | 파일 | 역할 |
|------------|------|------|
| `박재윤` | `파일명.py` | 담당 역할 |

**Stage 3** — 담당: `박정현`
| 레이어 담당 | 파일 | 역할 |
|------------|------|------|
| `박정현` | `파일명.py` | 담당 역할 |

**공용** — 총괄 PM: `김예슬`
| 레이어 담당 | 파일 | 역할 |
|------------|------|------|
| `김예슬` | `pipeline/` | 파이프라인 구축, 통합 모델, 학습/평가 스크립트, 도식화 |

---

## 2. 디렉토리 구조

```
pipeline/
│
├── configs/                    # Stage별 YAML 설정 파일
│   ├── stage1.yaml             # Baseline (BERT + MeanPool + BCE)
│   ├── stage2.yaml             # ATLOP + DREEAM
│   ├── stage3.yaml             # Stage 2 + GAIN-lite GNN
│   └── stage4.yaml             # Stage 2 + Graph U-Net (Ablation vs Stage 3)
│
├── data/
│   ├── docred/                 # DocRED 원본 JSON 데이터
│   │   ├── train_annotated.json
│   │   ├── dev.json
│   │   └── test.json
│   └── meta/
│       ├── rel2id.json         # relation → id 매핑 (96종 + NA)
│       └── rel_info.json       # relation 상세 정보
│
├── src/                        # 핵심 모듈 (파이프라인 레이어)
│   ├── preprocessing.py        # [Layer 0] 데이터 전처리 + DataLoader
│   ├── encoder.py              # [Layer 1] BERT Document Encoder
│   ├── entity_repr.py          # [Layer 2] Entity Representation
│   ├── graph_encoder.py        # [Layer 3-A] GAIN-lite GNN Graph Encoder (Stage 3)
│   ├── structural_encorder.py  # [Layer 3-B] Graph U-Net Encoder (Stage 4)
│   ├── relation_head.py        # [Layer 4] Relation Classifier
│   ├── losses.py               # Stage별 Loss 함수
│   ├── postprocessing.py       # 예측 결과 → Triple 변환
│   ├── evaluation.py           # F1 평가 지표 계산
│   ├── kg_builder.py           # Neo4j KG 저장
│   ├── model.py                # 통합 모델 (DocREModel)
│   └── utils.py                # 공통 유틸리티
│
├── scripts/
│   ├── train.py                # 학습 실행 스크립트
│   ├── evaluate.py             # 평가 실행 스크립트
│   ├── infer_evidence.py       # Silver evidence 추론 (Stage 2용)
│   └── build_kg.py             # KG 구축 스크립트
│
├── notebooks/
│   └── run_pipeline.ipynb      # 전체 파이프라인 실행 노트북 (Colab)
│
└── requirements.txt
```

---

## 3. 데이터셋 DocRED

### DocRED 원본 JSON 구조

각 문서(document)는 아래 구조로 구성됩니다.

```json
{
  "title": "문서 제목",
  "sents": [
    ["token1", "token2", ...],
    ["token1", "token2", ...]
  ],
  "vertexSet": [
    [
      {"name": "Obama", "sent_id": 0, "pos": [2, 3], "type": "PER"}
    ],
    [
      {"name": "USA", "sent_id": 0, "pos": [5, 6], "type": "ORG"},
      {"name": "United States", "sent_id": 2, "pos": [1, 3], "type": "ORG"}
    ]
  ],
  "labels": [
    {"h": 0, "t": 1, "r": "P17", "evidence": [0, 2]}
  ]
}
```

| 필드 | 설명 |
|------|------|
| `sents` | 문서를 구성하는 문장별 토큰 리스트 |
| `vertexSet` | entity별 mention 목록 (sent_id, pos, type 포함) |
| `labels` | (head_idx, tail_idx, relation_id, 근거 문장 목록) |
| `evidence` | 해당 relation의 근거가 된 문장 인덱스 (DREEAM용) |

### Relation 정보

- 총 **97가지** relation: 96종 Wikidata property + 1 NA(관계 없음)
- 매핑 파일: `data/meta/rel2id.json`
- 예: `"P17"` → 국가, `"P131"` → 행정구역, `"P710"` → 참가자

---

## 4. 전체 파이프라인 흐름

```
[DocRED JSON]
      │
      ▼
┌───────────────────────────────────────────────────────────┐
│  Layer 0. Preprocessing  (preprocessing.py)               │
│  - 문서 토큰화 및 subword 변환                                │
│  - entity mention → subword span 정렬                      │
│  - entity pair 생성 (N×(N-1) 방향성 쌍)                      │
│  - relation multi-hot label 구성                           │
└───────────────────────────┬───────────────────────────────┘
                            │ input_ids [B, 512]
                            │ attention_mask [B, 512]
                            │ entity_spans, entity_pairs
                            │ labels [num_pairs, 97]
                            │ sent_map, evidence_labels
                            ▼
┌───────────────────────────────────────────────────────────┐
│  Layer 1. Document Encoder  (encoder.py)                  │
│  - BERT 마지막 3개 layer hidden states 평균                  │
└───────────────────────────┬───────────────────────────────┘
                            │ hidden_states [B, seq_len, 768]
                            ▼
┌───────────────────────────────────────────────────────────┐
│  Layer 2. Entity Representation  (entity_repr.py)         │
│  - Stage 1: Mean Pooling                                  │
│  - Stage 2/3: LogSumExp Pooling                           │
└───────────────────────────┬───────────────────────────────┘
                            │ entity_vectors [num_entities, 768]
                            ▼
              ┌──────────────────────────────────────────────────┐
              │  Layer 3. Graph Encoder  ← Stage 3 / 4 선택     │
              │                                                  │
              │  [Stage 3] graph_encoder.py (GAIN-lite)          │
              │    - Entity Graph 구성 (Co-ref/Co-occur/Cross)   │
              │    - GCN/GAT Flat Message Passing (2 layers)     │
              │                                                  │
              │  [Stage 4] structural_encorder.py (Graph U-Net) │
              │    - 동일 Entity Graph 구성                       │
              │    - Encoder GNN → TopK Pooling (Down-sampling) │
              │    - Bottleneck GNN (Global Reasoning)           │
              │    - Unpooling + Skip Connection (Up-sampling)  │
              │    - Decoder GNN → Residual + LayerNorm         │
              └──────────────────────┬───────────────────────────┘
                                     │ refined_entity_vectors [num_entities, 768]
                                     ▼
┌───────────────────────────────────────────────────────────┐
│  Layer 4. Relation Head  (relation_head.py)               │
│  - Stage 1: Bilinear MLP + Fixed Threshold (0.5)          │
│  - Stage 2/3/4: ATLOP Grouped Bilinear + Adaptive TH      │
│               + DREEAM Evidence Head                      │
└───────────────────────────┬───────────────────────────────┘
                            │ relation_logits [num_pairs, 97]
                            │ (+ threshold_logits, evidence_logits)
                            ▼
┌───────────────────────────────────────────────────────────┐
│  Postprocessing  (postprocessing.py)                      │
│  - 임계값 적용 → positive relation 추출                       │
│  - Triple (head, relation, tail, score) 리스트 생성          │
│  - 중복 제거                                                │
└───────────────────────────┬───────────────────────────────┘
                            │ triples: [{h, t, r, score}, ...]
                            ▼
┌───────────────────────────────────────────────────────────┐
│  KG Builder  (kg_builder.py)                              │
│  - Neo4j에 Entity Node + Relation Edge 저장                 │
│  - Multi-hop 경로 탐색 지원                                  │
└───────────────────────────────────────────────────────────┘
```

---

## 5. Stage 1 — Baseline RE 상세

### 개요

BERT 인코더 + Mean Pooling + Bilinear 분류기로 구성된 기본 관계 추출 모델입니다.
Baseline을 수립하고, 이후 Stage에서 각 구성 요소를 개선하는 기준점 역할을 합니다.

---

### 레이어별 Input / Output 요약

```
Raw DocRED JSON
      │
      ▼ [Layer 0] preprocessing.py
      │  IN  : doc["sents"], doc["vertexSet"], doc["labels"]
      │  OUT : input_ids [512]
      │        attention_mask [512]
      │        entity_spans  List[ List[Tuple(start, end)] ]
      │        entity_pairs  List[Tuple(h_id, t_id)]
      │        labels        [num_pairs, 97]
      │        sent_map      List[int]
      │        evidence_labels  Dict
      │
      ▼ [Layer 1] encoder.py
      │  IN  : input_ids [B, 512], attention_mask [B, 512]
      │  OUT : hidden_states [B, 512, 768]
      │
      ▼ [Layer 2] entity_repr.py  (pooling="mean")
      │  IN  : hidden_states [B, 512, 768]
      │        entity_spans
      │  OUT : entity_vectors  List[ [num_entities, 768] ]
      │
      ▼ [Layer 4] relation_head.py  (classifier_type="bilinear")
      │  IN  : entity_vectors [num_entities, 768]
      │        entity_pairs   [(h_id, t_id), ...]
      │  OUT : relation_logits [num_pairs, 97]
      │
      ▼ losses.py  (loss_type="bce")
      │  IN  : relation_logits [num_pairs, 97]
      │        labels [num_pairs, 97]
      │  OUT : scalar BCE loss
      │
      ▼ postprocessing.py
         IN  : relation_logits, threshold=0.5
         OUT : List[{h, t, r, score}]
```

---

### 각 모듈 상세 설명

#### Layer 0 — Preprocessing (`src/preprocessing.py`)

DocRED 원본 JSON을 BERT가 처리할 수 있는 텐서 형태로 변환합니다.

**처리 단계**

| Step | 내용 |
|------|------|
| **Step 1** | 문장별 토큰 리스트를 하나의 문서 시퀀스로 연결, `sent_map` 생성 |
| **Step 2** | HuggingFace Tokenizer로 subword 토큰화, `word_ids` 추출 |
| **Step 3** | entity mention의 `sent_id + pos` → 문서 전체 subword 인덱스로 변환 |
| **Step 4** | N개 entity → N×(N-1)개 방향성 entity pair 생성 |
| **Step 5** | pair별 multi-hot relation label 벡터 [num_pairs, 97] 생성 |

**주요 클래스 및 함수**

```python
DocREDDataset(
    data_dir,           # DocRED JSON 파일 경로
    data_file,          # 파일명 (예: "train_annotated.json")
    tokenizer,          # HuggingFace AutoTokenizer
    rel2id,             # {"P17": 0, "P131": 1, ...} 딕셔너리
    max_seq_len=512,
    stage="stage1",
)

create_dataloader(data_dir, data_file, tokenizer, rel2id, ...)
load_rel2id(meta_dir, filename="rel2id.json")
```

**출력 Feature 구조**

| 키 | 형태 | 설명 |
|----|------|------|
| `input_ids` | `[512]` | BERT subword 토큰 인덱스 |
| `attention_mask` | `[512]` | 패딩 마스크 (1=실제 토큰, 0=패딩) |
| `entity_spans` | `List[List[Tuple]]` | entity별 mention의 subword 범위 목록 |
| `entity_pairs` | `List[Tuple]` | (head_id, tail_id) 방향성 쌍 |
| `labels` | `[num_pairs, 97]` | multi-hot relation label |
| `sent_map` | `List[int]` | subword별 소속 문장 인덱스 (Stage 3 그래프 구성용) |
| `num_sents` | `int` | 문서 내 총 문장 수 |
| `evidence_labels` | `Dict` | pair별 근거 문장 정보 (DREEAM용) |

**subword span 변환 로직 (`_find_subword_span`)**

원본 단어 인덱스 `[abs_start, abs_end)` → tokenizer의 `word_ids`를 순회하여
해당 범위에 매핑되는 첫 번째 / 마지막 subword 인덱스를 반환합니다.

```
원본 토큰: ["Barack", "Obama", "visited", "France"]
              0          1         2          3

BERT subword: [CLS, "Barack", "O", "##bam", "##a", "visited", "France", SEP]
word_ids:      [None,  0,       1,    1,       1,      2,         3,      None]

entity "Obama" pos=[1,2]  →  sw_start=2, sw_end=5
```

**DataLoader collate_fn**

가변 길이 entity/pair 정보를 배치로 묶기 위해 커스텀 `docred_collate_fn`을 사용합니다.
`input_ids`, `attention_mask`는 스택 텐서로, 나머지는 리스트 형태로 배치됩니다.

---

#### Layer 1 — Document Encoder (`src/encoder.py`)

사전 학습된 BERT를 사용하여 문서 전체의 token-level contextual representation을 생성합니다.

```
INPUT : input_ids      [B, 512]
        attention_mask [B, 512]
        ↓
       BERT (bert-base-uncased, 12 layers)
        ↓
       마지막 3개 layer hidden states 평균
        ↓
OUTPUT: hidden_states  [B, 512, 768]
```

**구현 포인트**

- `AutoConfig`에서 `output_hidden_states=True` 설정으로 모든 레이어의 hidden states 추출
- ATLOP/DREEAM 논문 방식을 따라 **마지막 3개 레이어의 평균** 사용

```python
# 마지막 3개 layer 평균 (단일 레이어보다 더 풍부한 표현)
hidden_states = (all_hidden[-1] + all_hidden[-2] + all_hidden[-3]) / 3.0
```

---

#### Layer 2 — Entity Representation (`src/entity_repr.py`)

token-level hidden states에서 각 entity의 벡터 표현을 추출하고 통합합니다.

```
INPUT : hidden_states [B, 512, 768]
        entity_spans  (batch별 entity별 mention span 목록)
        ↓
       각 mention span의 첫 번째 subword 토큰 추출
        ↓
       여러 mention → 하나의 entity 벡터로 통합 (Pooling)
        ↓
OUTPUT: entity_vectors  List[ [num_entities, 768] ]
```

**Stage 1: Mean Pooling**

동일 entity의 여러 mention 벡터를 **산술 평균**합니다.

```python
entity_vec = mention_stack.mean(dim=0)  # [num_mentions, 768] → [768]
```

- 구현이 단순하고 안정적
- mention별 중요도 차이를 반영하지 못한다는 한계 존재

**Stage 2에서의 개선: LogSumExp Pooling**

```python
entity_vec = torch.logsumexp(mention_stack, dim=0)
```

- 중요한 mention에 더 큰 가중치가 자연스럽게 부여됨 (ATLOP 논문 방식)

---

#### Layer 4 — Relation Head (`src/relation_head.py`)

Entity pair의 벡터를 결합하여 relation을 multi-label 분류합니다.

**Stage 1: Bilinear Classifier**

```
INPUT : entity_vectors [num_entities, 768]
        entity_pairs   [(h_id, t_id), ...]
        ↓
       head/tail 벡터 추출
       pair_repr = concat(e_h, e_t, e_h ⊙ e_t)  → [num_pairs, 768×3]
        ↓
       MLP: Linear(2304→768) → ReLU → Dropout(0.1) → Linear(768→97)
        ↓
OUTPUT: relation_logits [num_pairs, 97]
```

**Pair Representation 구성**

| 구성 요소 | 차원 | 역할 |
|-----------|------|------|
| `e_h` | 768 | head entity 벡터 |
| `e_t` | 768 | tail entity 벡터 |
| `e_h ⊙ e_t` | 768 | element-wise 곱 (상호작용 포착) |
| → concatenate | 2304 | 최종 pair representation |

**예측 (Inference)**

```python
probs = torch.sigmoid(relation_logits)
predictions = (probs > 0.5).float()   # Fixed Threshold = 0.5
```

> **Stage 2에서의 개선**: Bilinear MLP → ATLOP Grouped Bilinear + Adaptive Threshold
>
> - head/tail 벡터에 문맥 벡터(rs)를 결합하여 관계 분류
> - 고정 임계값(0.5) 대신 pair마다 학습되는 adaptive threshold 사용

---

#### Loss 함수 (`src/losses.py`)

**Stage 1: Weighted BCE Loss**

```
INPUT : relation_logits [num_pairs, 97]
        labels          [num_pairs, 97]  (multi-hot)
        ↓
       Binary Cross-Entropy with Logits
       (NA relation class 가중치 0.1로 낮춰 클래스 불균형 대응)
        ↓
OUTPUT: scalar loss
```

```python
weights = torch.ones(97)
weights[0] = 0.1   # NA(no_relation) class에 낮은 가중치 부여
loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
```

> **클래스 불균형 이유**: DocRED에서 대부분의 entity pair는 관계가 없음(NA).
> NA class에 낮은 가중치를 부여해 모델이 positive relation에 집중하도록 합니다.

---

#### Postprocessing (`src/postprocessing.py`)

모델의 `relation_logits`를 구조화된 Triple 리스트로 변환합니다.

```
INPUT : relation_logits [num_pairs, 97]
        entity_pairs    [(h_id, t_id), ...]
        id2rel          {rel_idx: "P17", ...}
        threshold=0.5
        ↓
       sigmoid → 임계값 적용 → positive relation 추출
        ↓
       중복 triple 제거
        ↓
OUTPUT: [
  {"h": 0, "t": 1, "r": "P17", "score": 0.83,
   "head_name": "Obama", "tail_name": "USA"},
  ...
]
```

---

#### Evaluation (`src/evaluation.py`)

**주요 지표**

| 지표 | 설명 | 활용 Stage |
|------|------|-----------|
| **Micro F1** | 모든 relation triple을 동등하게 취급한 F1 | 전체 |
| **Ign F1** | 학습/테스트 셋 공통 triple 제외 F1 (DocRED 표준 지표) | 전체 |
| **Evidence F1** | 근거 문장 예측 F1 | Stage 2/3/4 |
| **Intra/Inter F1** | 문장 내 / 문장 간 관계별 F1 분석 | Stage 3/4 |

```python
compute_micro_f1(predictions, gold_labels, ignore_train_triples)
# → {"precision": ..., "recall": ..., "f1": ..., "ign_f1": ...}

evaluate_evidence(pred_evidence, gold_evidence)
# → {"evidence_f1": ...}
```

---

#### 통합 모델 (`src/model.py`)

`DocREModel`은 위 모든 레이어를 하나의 `nn.Module`로 통합합니다.
`config["experiment"]["stage"]` 값에 따라 레이어가 선택적으로 활성화됩니다.

```python
model = DocREModel(config)

# forward 흐름
hidden_states = model.encoder(input_ids, attention_mask)            # Layer 1
entity_vecs   = model.entity_repr(hidden_states, entity_spans)      # Layer 2
# (Stage 3/4) entity_vecs = model.graph_encoder(entity_vecs, ...)  # Layer 3
#   Stage 3: GraphEncoder (GAIN-lite Flat GNN)
#   Stage 4: GraphUNetEncoder (U-Net 계층적 GNN)
outputs       = model.relation_head(entity_vecs, entity_pairs)      # Layer 4
```

---

### 학습 로직 (`scripts/train.py`)

```
1. Config 로드 (configs/stage1.yaml)
2. Tokenizer + rel2id 로드
3. DataLoader 생성 (train / dev)
4. DocREModel 초기화
5. Optimizer 구성
   - BERT Encoder params : lr = 2e-5  (사전 학습 가중치 → 낮은 lr)
   - 나머지 레이어 params : lr = 1e-4  (새로 초기화 → 높은 lr)
6. Linear Warmup Scheduler (warmup_ratio=0.06)

for epoch in range(30):
    ┌─ Train ──────────────────────────────────────────────┐
    │  for batch in train_loader:                          │
    │      forward() → relation_logits                     │
    │      compute_loss() → BCE Loss                       │
    │      backward()                                      │
    │      clip_grad_norm(max_norm=1.0)                    │
    │      optimizer.step() / scheduler.step()             │
    └──────────────────────────────────────────────────────┘
    ┌─ Dev Eval ────────────────────────────────────────────┐
    │  evaluate_on_dev() → dev_loss, num_predictions        │
    └───────────────────────────────────────────────────────┘
    save_checkpoint(epoch)

save_checkpoint(best_model.pt)
```

**Optimizer 파라미터 그룹 분리 이유**

BERT는 사전 학습된 모델이므로 낮은 lr(2e-5)로 fine-tuning하고,
새로 추가된 분류 레이어는 높은 lr(1e-4)로 빠르게 학습합니다.

---

### Stage 1 설정 (`configs/stage1.yaml`)

```yaml
experiment:
  name: "stage1_baseline"
  stage: "stage1"
  seed: 42
  device: "cuda:0"

data:
  num_relations: 97
  max_seq_length: 512

encoder:
  model_name: "bert-base-uncased"
  hidden_size: 768

entity_repr:
  pooling: "mean"                  # Stage 1 핵심: 평균 풀링

relation_head:
  classifier_type: "bilinear"      # Stage 1 핵심: MLP 분류기
  threshold_type: "fixed"
  fixed_threshold: 0.5
  use_evidence_head: false

graph_encoder:
  enabled: false                   # Stage 1: GNN 미사용

training:
  loss_type: "bce"                 # Stage 1 핵심: Binary Cross-Entropy
  encoder_lr: 2.0e-5
  classifier_lr: 1.0e-4
  finetune_epochs: 30
  max_grad_norm: 1.0
  no_relation_weight: 0.1
```

---

## 6. Stage 2 — ATLOP + DREEAM

### 개요

Stage 1 Baseline에서 다음 두 논문의 핵심 기법을 적용하여 성능을 강화합니다.

- **ATLOP** (Zhou et al., 2021): Adaptive Threshold + Grouped Bilinear Classifier
- **DREEAM** (Ma et al., 2023): Evidence-guided 문맥 벡터 활용 Self-training

### Stage 1 → Stage 2 변경 사항

| 구성 요소 | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Entity Pooling | `mean` | **`logsumexp`** |
| Classifier | Bilinear MLP | **ATLOP Grouped Bilinear** |
| Threshold | Fixed (0.5) | **Adaptive (pair별 학습)** |
| Context Vector | 없음 | **rs 벡터 (head-tail 공통 문맥)** |
| Loss | BCE | **ATL Loss** |
| Evidence Head | 없음 | **DREEAM Evidence Head + KL Loss** |

### 레이어 흐름

```
[Layer 1] BERT Encoder
      ↓  hidden_states [B, 512, 768]
[Layer 2] Entity Representation  (LogSumExp Pooling)
      ↓  entity_vectors [num_entities, 768]
[model.py] rs_vector 추출
      - head/tail 벡터와 hidden_states 간 attention 계산
      - 두 entity가 공통으로 주목하는 문맥 벡터(rs) 추출
      ↓  rs_vectors [num_pairs, 768]
[Layer 4] ATLOP Relation Head
      - head_proj = tanh(Linear([e_h; rs]))
      - tail_proj = tanh(Linear([e_t; rs]))
      - Grouped Bilinear: b1 × b2 → [num_pairs, 97]
      - Adaptive Threshold: threshold_logit per pair
      - Evidence Head: evidence_logits [num_pairs, num_sents]
      ↓
[Loss] ATL Loss + λ × Evidence KL Loss  (λ=0.1)
```

### 주요 변경 모듈 상세

#### LogSumExp Pooling (`entity_repr.py`)

Stage 1의 Mean Pooling을 대체하여, mention별 중요도에 자동 가중치를 부여합니다.

```python
# Mean (Stage 1): 모든 mention을 동등하게 취급
entity_vec = mention_stack.mean(dim=0)

# LogSumExp (Stage 2): 값이 큰 mention이 자연스럽게 더 큰 가중치를 받음
entity_vec = torch.logsumexp(mention_stack, dim=0)
```

#### rs 벡터 추출 (`model.py` forward 내부)

ATLOP 논문 방식으로 head/tail entity가 **공통으로 주목하는 문맥**을 추출합니다.

```python
# 1. head/tail 벡터와 문서 전체 토큰 간 유사도(attention) 계산
h_att = softmax(h_vecs @ h_states.T)     # [num_pairs, seq_len]
t_att = softmax(t_vecs @ h_states.T)     # [num_pairs, seq_len]

# 2. 두 attention의 교집합 (공통 문맥)
ht_att = h_att * t_att
ht_att = ht_att / (ht_att.sum(dim=-1, keepdim=True) + 1e-30)

# 3. 공통 문맥 벡터 생성
rs_vectors = ht_att @ h_states           # [num_pairs, hidden_size]
```

#### ATLOP Grouped Bilinear (`relation_head.py`)

```python
# head/tail 벡터에 문맥 벡터(rs)를 concat 후 투영
h_proj = tanh(head_extractor(cat([head_vecs, rs_vectors])))   # [num_pairs, emb_size]
t_proj = tanh(tail_extractor(cat([tail_vecs, rs_vectors])))   # [num_pairs, emb_size]

# Block-wise Bilinear (Grouped Bilinear)
b1 = h_proj.view(-1, emb_size // block_size, block_size)
b2 = t_proj.view(-1, emb_size // block_size, block_size)
pair_repr = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, emb_size * block_size)

# Relation 분류 + Adaptive Threshold
relation_logits = bilinear(pair_repr)          # [num_pairs, 97]
threshold_logits = threshold_linear(pair_repr)  # [num_pairs, 1]
```

#### ATL Loss + Evidence KL Loss (`losses.py`)

```python
# ATL Loss: TH class를 concat하여 positive/negative relation 분리
logits = cat([relation_logits, threshold_logits], dim=-1)   # [num_pairs, 98]
th_label = (labels.sum(dim=-1) == 0).float().unsqueeze(-1)  # positive 없으면 TH=1
re_loss = bce_with_logits(logits, cat([labels, th_label]))

# Evidence Loss (DREEAM): 예측 attention과 근거 문장 분포 간 KL divergence
target_dist = evidence_labels → softmax normalized distribution
pred_dist   = softmax(evidence_logits)
evi_loss    = kl_div(pred_dist.log(), target_dist)

total_loss = re_loss + λ(0.1) × evi_loss
```

### Stage 2 설정 (`configs/stage2.yaml`) 핵심

```yaml
entity_repr:
  pooling: "logsumexp"              # ← mean에서 변경

relation_head:
  classifier_type: "atlop"          # ← bilinear에서 변경
  threshold_type: "adaptive"        # ← fixed에서 변경
  use_evidence_head: true           # ← DREEAM Evidence Head 활성화

evidence:
  lambda_evidence: 0.1              # evidence loss 가중치

training:
  loss_type: "atlop"                # ← bce에서 변경
```

---

## 7. Stage 3 — GAIN-lite GNN

### 개요

Stage 2에 **Graph Neural Network(GNN)** 을 추가하여 문장 간(inter-sentence) 관계와
multi-hop 추론 능력을 강화합니다.
Stage 2 체크포인트를 기반으로 전이 학습하며, Stage 4와의 Ablation 비교 대상입니다.

- **GAIN** (Zeng et al., 2020): 이기종 Entity Graph + GCN 기반 message passing

### Stage 2 → Stage 3 변경 사항

| 구성 요소 | Stage 2 | Stage 3 |
|-----------|---------|---------|
| Graph Encoder | 없음 | **GAIN-lite GCN / GAT** |
| Entity Graph | 없음 | **Co-ref + Co-occur + Cross-sent 3종 Edge** |
| 모델 초기화 | 랜덤 | **Stage 2 체크포인트에서 전이 학습** |

### 레이어 흐름

```
[Layer 1] BERT Encoder
      ↓  hidden_states [B, 512, 768]
[Layer 2] Entity Representation  (LogSumExp)
      ↓  entity_vectors [num_entities, 768]
[Layer 3] Graph Encoder  ← Stage 3 신규 추가
      - Entity Graph 구성 (build_entity_graph)
        · Co-occurrence : 같은 문장에 등장하는 entity 간 연결
        · Cross-sentence: 인접 문장(window=1)의 entity 간 연결
        · Self-loop     : 자기 자신 연결
      - 인접 행렬 Row-Normalize
      - GCN/GAT Message Passing (2 layers)
      - 각 layer: GNN → ReLU → Residual Connection + LayerNorm
      ↓  refined_entity_vectors [num_entities, 768]
[model.py] rs_vector 추출 (Stage 2와 동일)
      ↓
[Layer 4] ATLOP Relation Head (Stage 2와 동일)
      ↓
[Loss] ATL Loss + Evidence KL Loss
```

### Entity Graph 구성 (`build_entity_graph`)

`preprocessing.py`에서 생성한 `entity_spans`와 `sent_map`을 사용하여
entity 간 관계를 인접 행렬로 구성합니다.

```python
adj = torch.zeros(num_entities, num_entities)

for i, j in all_entity_pairs:
    # 1. Self-loop
    adj[i][i] = 1.0

    # 2. Co-occurrence: entity i, j가 같은 문장에 등장
    common_sents = entity_sents[i] & entity_sents[j]
    if len(common_sents) > 0:
        adj[i][j] = 1.0

    # 3. Cross-sentence: 인접 문장 (|sent_i - sent_j| <= window)
    for si in entity_sents[i]:
        for sj in entity_sents[j]:
            if abs(si - sj) <= cross_sent_window:
                adj[i][j] = 1.0

# Row-normalize
degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
adj = adj / degree
```

### GCN / GAT Message Passing (`graph_encoder.py`)

| 타입 | 수식 | 설명 |
|------|------|------|
| **GCN** (기본) | `h' = Dropout(A_norm @ Linear(h))` | 이웃 노드 feature의 가중 합산 |
| **GAT** (ablation) | `h' = Σ α_ij · W · h_j` | Attention 기반 이웃 가중치 학습 (4 heads) |

**GAT Attention 계산**

```python
# Multi-head attention (4 heads)
h = W(x).view(N, num_heads, head_dim)        # [N, 4, 192]
attn = (h * attn_src).sum(-1) + (h * attn_dst).sum(-1).T   # [N, 4, N]
attn = softmax(LeakyReLU(attn), dim=-1)       # adj=0 위치는 -inf masking
out = attn @ h → reshape → [N, 768]
```

**Residual Connection + LayerNorm**

각 GNN layer 후 입력을 더해주고 LayerNorm을 적용하여 깊은 레이어에서도 안정적으로 학습합니다.

```python
for layer in gnn_layers:
    h_new = layer(h, adj)
    h_new = relu(h_new)
    h = layer_norm(h + h_new)   # Residual + Normalize
```

### Stage 3 설정 (`configs/stage3.yaml`) 핵심

```yaml
graph_encoder:
  enabled: true
  gnn_type: "gcn"               # gcn | gat
  num_layers: 2
  hidden_dim: 768
  dropout: 0.1
  cross_sent_window: 1

training:
  load_checkpoint: "checkpoints/stage2/best_model.pt"   # Stage 2 전이 학습
```

---

## 8. Stage 4 — Graph U-Net

### 개요

Stage 3의 Flat GNN(GAIN-lite) 대신 **계층적 Graph U-Net 구조**를 적용하여
더 넓은 범위의 Multi-hop 전역 추론 능력을 강화합니다.
Stage 3과의 **Ablation Study** 목적으로, 둘 다 동일하게 **Stage 2 체크포인트**에서 시작합니다.

- **Graph U-Net** (Gao & Ji, 2019): TopK 풀링 기반 계층적 그래프 표현 학습

### Stage 3 vs Stage 4 핵심 차이

| 구성 요소 | Stage 3 (GAIN-lite) | Stage 4 (Graph U-Net) |
|-----------|--------------------|-----------------------|
| 그래프 구조 | Flat 2-layer GNN | **U-Net 계층 구조** |
| 노드 처리 | 전체 노드 균등 처리 | **중요 노드 선별 (TopK Pool)** |
| 추론 범위 | 1~2 hop 이웃 | **글로벌 Bottleneck 추론** |
| Skip Connection | 없음 | **있음 (로컬+글로벌 융합)** |
| 시작 체크포인트 | Stage 2 best_model.pt | **Stage 2 best_model.pt** |
| 모듈 파일 | `graph_encoder.py` | **`structural_encorder.py`** |

### Graph U-Net 레이어 흐름

```
entity_vectors [num_entities, 768]   ← Stage 2 LogSumExp 결과물
      │
      ▼ ① Entity Graph 구성  (build_entity_graph 재활용)
        Co-occur + Cross-sent + Self-loop Edge (Stage 3과 동일)
      │
      ▼ ② Encoder GNN  (enc_gnn)
        GCN/GAT로 로컬 이웃 정보 통합
        skip_connection = h_enc.clone()   ← 로컬 문맥 저장
      │
      ▼ ③ TopK Pooling  (GraphTopKPool, pool_ratio=0.5)
        score = sigmoid(Linear(h))        ← 노드별 중요도 계산
        상위 50% 노드만 선별 + top_idx 저장
        adj 도 선별된 노드끼리만 축소
      │
      ▼ ④ Bottleneck GNN  (bottleneck_gnn)
        압축된 그래프에서 전역(Global) 추론
        → 멀리 떨어진 Entity 간 간접 관계 포착
      │
      ▼ ⑤ Unpooling  (GraphUnpool)
        top_idx 기반으로 원래 크기 복원
        선택 안 된 노드 자리는 Zero-padding
      │
      ▼ ⑥ Skip Connection 융합
        h_dec = h_unpooled + skip_connection
        → Zero-padded 공간을 로컬 피처로 채움
      │
      ▼ ⑦ Decoder GNN  (dec_gnn)
        융합된 로컬+글로벌 정보로 최종 정제
      │
      ▼ ⑧ Residual Connection + LayerNorm
        out = LayerNorm(entity_vectors + h_out)
        → 오리지널 Stage 2 벡터에 U-Net 델타값 추가
      │
      ▼
refined_entity_vectors [num_entities, 768]
```

### 서브모듈 상세 (`src/structural_encorder.py`)

#### GraphTopKPool — 학습 가능한 노드 선별

학습 가능한 투영 벡터로 노드별 중요도를 평가하고, 상위 K개만 남깁니다.

```python
scores = sigmoid(Linear(x))             # [N] 노드별 중요도
_, top_idx = topk(scores, k=N * ratio)   # 상위 50% 선별
top_idx = sort(top_idx)                  # 원래 순서 유지

x_pooled = x[top_idx] * scores[top_idx]  # Gate: 중요도를 곱해 역전파 가능
adj_pooled = adj[top_idx][:, top_idx]     # 인접 행렬도 축소
```

- `pool_ratio=0.5` → entity가 10개면 5개만 남김
- `scores`를 곱해주는 Gate 메커니즘으로 어떤 노드를 살릴지 역전파로 학습

#### GraphUnpool — 원래 크기 복원

```python
x_unpooled = zeros(orig_size, dim)       # [N, 768] 빈 텐서
x_unpooled[top_idx] = x_pooled           # 살아남은 위치에 Bottleneck 결과 배치
# → 나머지는 0 (Skip Connection에서 로컬 피처로 채워짐)
```

#### GraphUNetEncoder — U-Net 전체 통합

Stage 3의 `GraphEncoder`와 **동일한 Input/Output 인터페이스**를 가지므로
`model.py`에서 `architecture` config 값 하나로 교체 가능합니다.

```python
# model.py에서의 분기
if arch_type == "unet":
    self.graph_encoder = GraphUNetEncoder(...)   # Stage 4
else:
    self.graph_encoder = GraphEncoder(...)       # Stage 3
```

### `model.py`에서의 Stage 3/4 분기 로직

`graph_encoder.enabled=true`이고 stage가 `"stage3"` 또는 `"stage4"`일 때
`self.graph_encoder`가 호출됩니다. Stage 3/4는 같은 forward 경로를 공유하며,
초기화 시 `architecture` 값에 따라 어떤 Graph Encoder가 생성되었는지만 다릅니다.

```python
# model.py forward (Stage 3/4 공통)
if self.graph_encoder is not None and self.stage in ["stage3", "stage4"]:
    for b in range(len(batch_entity_vecs)):
        refined = self.graph_encoder(
            entity_vectors=batch_entity_vecs[b],
            entity_spans=batch["entity_spans"][b],
            sent_map=batch["sent_map"][b],
            num_sents=batch["num_sents"][b],
        )
        refined_vecs.append(refined)
    batch_entity_vecs = refined_vecs
```

### Stage 4 설정 (`configs/stage4.yaml`) 핵심

```yaml
experiment:
  stage: "stage4"

graph_encoder:
  enabled: true
  architecture: "unet"      # ← Stage 3(gain)과의 핵심 차이
  pool_ratio: 0.5           # TopK 노드 압축 비율 (50%)
  gnn_type: "gcn"           # gcn | gat
  num_layers: 2
  hidden_dim: 768
  dropout: 0.1
  cross_sent_window: 1

training:
  load_checkpoint: "checkpoints/stage2/best_model.pt"  # Stage 2 기반 (Stage 3과 동일)
```

---

## 9. Stage 간 비교

### 모델 구조 비교

```
Stage 1:
  BERT → MeanPool → concat(e_h, e_t, e_h⊙e_t) → MLP → 97 classes
                                                         ↑ sigmoid > 0.5

Stage 2:
  BERT → LogSumExp → rs 추출 → ATLOP Bilinear → 97 classes
                                                  ↑ Adaptive TH
                                               + Evidence Head

Stage 3:                               (Stage 2 ckpt 기반)
  BERT → LogSumExp → [GAIN-lite GCN/GAT] → rs 추출 → ATLOP Bilinear → 97 classes
                            ↑                                           ↑ Adaptive TH
                       Flat 2-layer GNN                              + Evidence Head
                  (Co-ref + Co-occur + Cross-sent)

Stage 4:                               (Stage 2 ckpt 기반, Stage 3 Ablation)
  BERT → LogSumExp → [Graph U-Net] → rs 추출 → ATLOP Bilinear → 97 classes
                           ↑                                      ↑ Adaptive TH
              Enc GNN → TopK Pool → Bottleneck GNN             + Evidence Head
              → Unpool + Skip → Dec GNN → Residual
```

### 설정 파일 핵심 비교

| 설정 키 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---------|---------|---------|---------|---------|
| `entity_repr.pooling` | `mean` | `logsumexp` | `logsumexp` | `logsumexp` |
| `relation_head.classifier_type` | `bilinear` | `atlop` | `atlop` | `atlop` |
| `relation_head.threshold_type` | `fixed` | `adaptive` | `adaptive` | `adaptive` |
| `relation_head.use_evidence_head` | `false` | `true` | `true` | `true` |
| `graph_encoder.enabled` | `false` | `false` | `true` | `true` |
| `graph_encoder.architecture` | - | - | `gain` (기본) | **`unet`** |
| `graph_encoder.pool_ratio` | - | - | - | **`0.5`** |
| `training.loss_type` | `bce` | `atlop` | `atlop` | `atlop` |
| `training.load_checkpoint` | - | - | Stage 2 ckpt | **Stage 2 ckpt** |

---

## 10. 환경 설정 및 실행

### 설치

```bash
pip install -r requirements.txt
```

### 주요 의존성

```
torch >= 2.0
transformers >= 4.30
numpy
pyyaml
tqdm
neo4j          # KG 저장 시 필요 (선택)
```

### 데이터 준비

```bash
mkdir -p data/docred data/meta

# DocRED 데이터를 data/docred/ 에 배치
# (HuggingFace: https://huggingface.co/datasets/thunlp/docred)
# - train_annotated.json
# - dev.json
# - test.json

# rel2id.json을 data/meta/ 에 배치
```

### 학습 실행

```bash
# Stage 1 (Baseline)
python scripts/train.py --config configs/stage1.yaml

# Stage 2 (ATLOP + DREEAM)
python scripts/train.py --config configs/stage2.yaml

# Stage 3 (+ GAIN-lite GNN)
python scripts/train.py --config configs/stage3.yaml

# Stage 4 (+ Graph U-Net, Ablation vs Stage 3)
python scripts/train.py --config configs/stage4.yaml
```

### 평가

```bash
python scripts/evaluate.py \
    --config configs/stage1.yaml \
    --checkpoint checkpoints/stage1/best_model.pt
```

### Colab 실행

```python
# 환경 설정
!pip install -r requirements.txt -q
import sys, os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

# Stage 1 학습
!python scripts/train.py --config configs/stage1.yaml
```

### GPU 설정

`configs/stage1.yaml`의 `experiment.device` 값을 수정하세요.

```yaml
experiment:
  device: "cuda:0"   # GPU 사용
  # device: "cpu"    # CPU 사용
```

CUDA를 사용할 수 없는 환경에서는 자동으로 CPU로 fallback 처리됩니다.

---

## 11. 참고 논문

| 논문 | 내용 | 적용 Stage |
|------|------|-----------|
| Yao et al. (2019) DocRED | 데이터셋 구성 및 평가 기준 | 전체 |
| Devlin et al. (2019) BERT | Document Encoder 기반 | 전체 |
| Zhou et al. (2021) ATLOP | Adaptive Threshold + LogSumExp Pooling | Stage 2/3 |
| Ma et al. (2023) DREEAM | Evidence-guided Self-training | Stage 2/3 |
| Zeng et al. (2020) GAIN | 이기종 Entity Graph + GCN | Stage 3 |
| Kipf & Welling (2017) GCN | Graph Convolutional Network | Stage 3/4 |
| Veličković et al. (2018) GAT | Graph Attention Network | Stage 3/4 |
| Gao & Ji (2019) Graph U-Net | 계층적 TopK 풀링 기반 Graph U-Net | Stage 4 |

### 참고 GitHub

- [ATLOP](https://github.com/wzhouad/ATLOP) — Zhou et al. (2021)
- [DREEAM](https://github.com/YoumiMa/dreeam) — Ma et al. (2023)
- [GAIN](https://github.com/PKUnlp-icler/GAIN) — Zeng et al. (2020)
- [SSAN](https://github.com/BenfengXu/SSAN) — Xu et al. (2021)
- [Graph U-Net](https://arxiv.org/abs/1905.05178) — Gao & Ji (2019)
