# 📑 Document-Level Relation Extraction with DocRED

<p align="center">
  <img src="https://img.shields.io/badge/Task-Information_Extraction-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-DocRED-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/🤗-HuggingFace-FFD21E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Difficulty-★★★★☆-red?style=for-the-badge" />
</p>

<p align="center">
  <b>자연어 문서에서 Entity를 식별하고, 문서 전체에 걸친 관계를 추론해<br/>구조화된 지식(Knowledge Graph)으로 변환하는 프로젝트</b>
</p>

<p align="center">
  📅 2026.04.02 ~ 2026.04.10 · 🔥 어서5시조 🔥
</p>

---

## 📌 Overview

자연어 텍스트에는 수많은 **객체(Entity)** 와 **관계(Relation)** 가 존재하지만,  
그 정보는 문장 단위로 분산되어 있으며 명시적으로 드러나지 않는 경우가 많습니다.

이 프로젝트는 **DocRED 데이터셋**을 활용해 문서 수준(Document-Level)의 관계 추출을 수행하고,  
최종적으로 **Knowledge Graph** 형태의 구조화된 데이터를 생성하는 것을 목표로 합니다.

```
📥 입력 (Raw Text)
"Elon Musk is the CEO of Tesla. Tesla is an American electric vehicle company."

📤 출력 (Structured Triple)
(Elon Musk)  ──[CEO_OF]──▶  (Tesla)
(Tesla)      ──[COUNTRY]──▶  (United States)
```

> **왜 중요한가?**  
> Information Extraction은 단순 정보 추출을 넘어 **Knowledge Graph 구축**, **RAG 검색 품질 개선**,  
> **LLM 출력 안정화** 등에 활용되며, 온톨로지 기반 구조화 레이어로서 그 중요성이 재조명되고 있습니다.

---

## 🎯 Task Description

### Information Extraction이란?

텍스트로부터 **의미 있는 객체와 그 관계를 추출**하여 구조화된 데이터로 변환하는 작업입니다.

<table>
<tr><td>🗺️</td><td><b>Knowledge Graph 구축</b></td><td>Entity 간 관계를 그래프로 표현</td></tr>
<tr><td>🔍</td><td><b>RAG 검색 품질 개선</b></td><td>구조화된 정보로 검색 정확도 향상</td></tr>
<tr><td>🤖</td><td><b>LLM 출력 안정화</b></td><td>팩트 기반 검증으로 hallucination 감소</td></tr>
</table>

### 파이프라인

```
Raw Document → NER → Coreference Resolution → Relation Extraction → Multi-hop Reasoning → Knowledge Graph
```

| 단계 | 작업 | 설명 | 난이도 |
|:---:|------|------|:---:|
| 1 | **Named Entity Recognition (NER)** | 문서 내 등장하는 Entity 식별 | ⭐⭐ |
| 2 | **Coreference Resolution** | 동일 Entity의 다양한 표현 통합 | ⭐⭐⭐ |
| 3 | **Relation Extraction** | Entity 간 관계 식별 | ⭐⭐⭐⭐ |
| 4 | **Multi-hop Reasoning** | 여러 문장에 걸친 관계 추론 | ⭐⭐⭐⭐⭐ |

### 종합 난이도

```
★★★★☆  (4/5)

NER까지는 접근이 어렵지 않으나,
Relation Extraction과 Multi-hop 기반 schema 정립이 핵심 난관입니다.
```

---

## 📦 Dataset: DocRED

> **DocRED**: A Large-Scale Document-Level Relation Extraction Dataset  
> *Yao et al., ACL 2019* · [📄 논문](https://aclanthology.org/P19-1074) · [💻 GitHub](https://github.com/thunlp/DocRED) · [🤗 HuggingFace](https://huggingface.co/datasets/thunlp/docred)

Wikipedia와 Wikidata로부터 구축된 **문서 수준 관계 추출 데이터셋**입니다.  
기존 문장 단위 RE 데이터셋과 달리, **40.7%의 관계가 여러 문장에 걸쳐 추론**되어야 합니다.

### 데이터 통계

| Split | 문서 수 | 비고 |
|-------|------:|------|
| `train_annotated` | 3,053 | 사람이 직접 annotation |
| `train_distant` | 101,873 | Distant Supervision으로 자동 생성 |
| `validation` | 998 | 평가용 |
| `test` | 1,000 | 평가용 |

- 📁 총 다운로드 크기: **21.00 MB** · 💾 생성 데이터: **20.12 MB**
- 🏷️ Entity 수: **132,375개** · 🔗 Relation 유형: **96종** · 📊 Relational Fact: **56,354개**

### Reasoning Type 분포

DocRED에서 관계를 추출하기 위해 필요한 추론 유형입니다:

| 추론 유형 | 비율 | 설명 |
|----------|-----:|------|
| Pattern Recognition | 38.9% | 단순 패턴 매칭으로 추출 가능 |
| **Logical Reasoning** | **26.6%** | Bridge Entity를 통한 간접 추론 |
| Coreference Reasoning | 17.6% | 동일 Entity의 다른 표현 식별 필요 |
| Common-sense Reasoning | 16.6% | 상식 기반 추론 필요 |

### 데이터 구조

```json
{
  "title": "Kungliga Hovkapellet",
  "sents": [
    ["Kungliga", "Hovkapellet", "is", "a", "Swedish", "orchestra", "..."],
    ["The", "orchestra", "originally", "consisted", "of", "..."]
  ],
  "vertexSet": [
    [
      {"name": "Kungliga Hovkapellet", "pos": [0, 1], "sent_id": 0, "type": "ORG"},
      {"name": "the orchestra",        "pos": [0, 1], "sent_id": 1, "type": "ORG"}
    ],
    [
      {"name": "Swedish", "pos": [4], "sent_id": 0, "type": "LOC"}
    ]
  ],
  "labels": {
    "head": [0],
    "tail": [1],
    "relation_id": ["P17"],
    "relation_text": ["country"],
    "evidence": [[0]]
  }
}
```

### 주요 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `title` | `string` | Wikipedia 문서 제목 |
| `sents` | `list[list[str]]` | 문장별 토큰 리스트 |
| `vertexSet` | `list[list[dict]]` | Entity 목록 (하나의 Entity가 여러 문장에 등장 가능 → coreference) |
| `vertexSet[i][j].sent_id` | `int` | 해당 mention이 등장하는 문장 번호 |
| `vertexSet[i][j].pos` | `list[int]` | 문장 내 토큰 위치 |
| `vertexSet[i][j].type` | `string` | Entity 타입 (PER, LOC, ORG, TIME, NUM, MISC) |
| `labels.head` / `tail` | `int` | 관계 주체/대상 (vertexSet 인덱스) |
| `labels.relation_id` | `string` | Wikidata Relation ID (e.g., P17 = country) |
| `labels.relation_text` | `string` | 관계 유형 이름 |
| `labels.evidence` | `list[int]` | 관계의 근거가 되는 문장 번호 |

### Entity 타입 분포

| 타입 | 비율 | 포함 내용 |
|------|-----:|---------|
| PER | 18.5% | 사람 (실존/가상 인물) |
| LOC | 30.9% | 지리적/정치적 위치, 시설 |
| ORG | 14.4% | 기업, 대학, 기관, 정치/종교 단체 |
| TIME | 15.8% | 절대/상대 날짜, 기간 |
| NUM | 5.1% | 퍼센트, 금액, 수량 |
| MISC | 15.2% | 이벤트, 예술작품, 법률 등 |

---

## 🏗️ Project Structure

```
📁 2026-Text2Graph/
│
├── 📁 data/
│   ├── train_annotated.json      # Human-annotated 학습 데이터
│   ├── train_distant.json        # Distantly supervised 데이터
│   ├── dev.json                  # Validation 세트
│   └── test.json                 # Test 세트
│
├── 📁 src/
│   ├── preprocess.py             # 데이터 전처리 & 토큰화
│   ├── ner.py                    # Named Entity Recognition
│   ├── coref.py                  # Coreference Resolution
│   ├── model.py                  # Relation Extraction 모델 정의
│   ├── train.py                  # 학습 루프
│   ├── evaluate.py               # 평가 (F1, Ign F1, AUC)
│   └── utils.py                  # 유틸리티 함수
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb              # 데이터 탐색 & 시각화
│   ├── 02_NER_experiment.ipynb   # NER 실험
│   ├── 03_RE_experiment.ipynb    # Relation Extraction 실험
│   └── 04_KG_visualization.ipynb # Knowledge Graph 시각화
│
├── 📁 outputs/
│   ├── predictions/              # 모델 예측 결과
│   └── graphs/                   # 생성된 Knowledge Graph
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Getting Started

### 환경 설정

```bash
git clone https://github.com/yeseul-kim01/2026-Text2Graph.git
cd 2026-Text2Graph
pip install -r requirements.txt
```

### 데이터 로드 (HuggingFace)

```python
from datasets import load_dataset

dataset = load_dataset("thunlp/docred")
train = dataset["train_annotated"]

# 데이터 확인
print(f"학습 데이터 수: {len(train)}")
print(f"첫 번째 문서 제목: {train[0]['title']}")
print(f"Entity 수: {len(train[0]['vertexSet'])}")
```

### 학습 실행

```bash
python src/train.py --model bert-base-uncased --epochs 10 --batch_size 40
```

---

## 📊 Evaluation Metrics

DocRED 논문에서 제안한 평가 지표를 사용합니다:

| 지표 | 설명 |
|------|------|
| **F1** | 전체 relational fact에 대한 F1 score |
| **Ign F1** | Train/Dev(Test) 공통 relational fact를 **제외**한 F1 (평가 편향 방지) |
| **AUC** | Area Under the Curve |
| **Ign AUC** | 공통 fact 제외 AUC |

### Baseline 성능 (논문 기준, Supervised Setting)

| Model | Dev Ign F1 | Dev F1 | Test Ign F1 | Test F1 |
|-------|----------:|-------:|-----------:|--------:|
| CNN | 37.99 | 43.45 | 36.44 | 42.33 |
| LSTM | 44.41 | 50.66 | 43.60 | 50.12 |
| BiLSTM | **45.12** | 50.95 | **44.73** | **51.06** |
| Context-Aware | 44.84 | **51.10** | 43.93 | 50.64 |
| **Human** | - | - | - | **88.0** |

> 💡 모델 최고 성능(~51 F1)과 인간 성능(88.0 F1) 사이에 **큰 격차**가 존재합니다.

---

## 📤 Expected Output

최종 결과는 아래와 같은 구조로 확장 가능해야 합니다:

```
Knowledge Graph
  └─ (Entity A) ──[relation]──▶ (Entity B)
     예: (Tesla) ──[country]──▶ (United States)

Relational Database
  └─ head        | relation | tail          | evidence
     Tesla       | country  | United States | [0, 3]

Graph-based Reasoning System
  └─ Multi-hop path: A → B → C
     예: Riddarhuset → Stockholm → Sweden
```

---

## 🗓️ Timeline

| 날짜 | 마일스톤 |
|------|---------|
| 04/02 (목) | 킥오프 · 데이터 탐색(EDA) · 역할 분담 |
| 04/03 (금) | 전처리 파이프라인 구축 · NER 실험 시작 |
| 04/04 (토) | NER 완료 · Coreference Resolution 진행 |
| 04/05 (일) | Relation Extraction 모델 설계 |
| 04/06 (월) | RE 모델 학습 · 중간 점검 |
| 04/07 (화) | Multi-hop Reasoning 실험 |
| 04/08 (수) | Knowledge Graph 생성 · 시각화 |
| 04/09 (목) | 평가 · 결과 분석 · 문서화 |
| 04/10 (금) | 최종 발표 준비 · README 정리 · 제출 |

---

## 📚 References

```bibtex
@inproceedings{yao-etal-2019-docred,
    title     = "{D}oc{RED}: A Large-Scale Document-Level Relation Extraction Dataset",
    author    = "Yao, Yuan and Ye, Deming and Li, Peng and Han, Xu and Lin, Yankai and
                 Liu, Zhenghao and Liu, Zhiyuan and Huang, Lixin and Zhou, Jie and Sun, Maosong",
    booktitle = "Proceedings of the 57th Annual Meeting of the ACL",
    year      = "2019",
    url       = "https://aclanthology.org/P19-1074",
    pages     = "764--777",
}
```

---

## 👥 Team — 🔥 어서5시조 🔥

| 이름 | 역할 | 담당 |
|------|------|------|
| 김예슬 | TBD | TBD |
| 박재윤 | TBD | TBD |
| 박정현 | TBD | TBD |
| 이수민 | TBD | TBD |

---

## 📜 License

본 프로젝트는 학술 목적으로 진행되었으며, DocRED 데이터셋의 라이선스를 따릅니다.

---

<p align="center">
  Made with 🔥 by <b>어서5시조</b> · 2026-04-02 ~ 2026-04-10
</p>
