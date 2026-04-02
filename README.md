# 📑 Document-Level Relation Extraction with DocRED

<p align="center">
  <img src="https://img.shields.io/badge/Task-Information_Extraction-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-DocRED-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/🤗-HuggingFace-FFD21E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Difficulty-★★★★☆-red?style=for-the-badge" />
  <br/>
  나중에 더 추가할 예정.
</p>

<p align="center">
  <b>자연어 문서에서 Entity를 식별하고, 문서 전체에 걸친 관계를 추론해 구조화된 지식으로 변환합니다.</b>
</p>

---

## 📌 Overview

자연어 텍스트에는 수많은 **객체(Entity)** 와 **관계(Relation)** 가 존재하지만,  
그 정보는 문장 단위로 분산되어 있으며 명시적으로 드러나지 않는 경우가 많습니다.

이 프로젝트는 **DocRED 데이터셋**을 활용해 문서 수준(Document-Level)의 관계 추출을 수행하고,  
최종적으로 **Knowledge Graph** 형태의 구조화된 데이터를 생성하는 것을 목표로 합니다.

```
입력 (Raw Text)
"Elon Musk is the CEO of Tesla. Tesla is an American electric vehicle company."

출력 (Structured Triple)
(Elon Musk)  ──[CEO_OF]──▶  (Tesla)
(Tesla)      ──[COUNTRY]──▶  (United States)
```

---

## 🎯 Task Description

### Information Extraction이란?

텍스트로부터 **의미있는 객체와 그 관계를 추출**하여 구조화된 데이터로 변환하는 작업입니다.  
최근에는 단순 정보 추출을 넘어 아래와 같은 목적으로 활용됩니다.

- 🗺️ **Knowledge Graph** 구축
- 🔍 **RAG(Retrieval-Augmented Generation)** 검색 품질 개선
- 🤖 **LLM 출력** 안정화 및 팩트 검증

### 수행 내용

| 단계 | 작업 | 설명 |
|------|------|------|
| 1 | **Named Entity Recognition (NER)** | 문서 내 등장하는 Entity 식별 |
| 2 | **Coreference Resolution** | 동일 Entity의 다양한 표현 통합 |
| 3 | **Relation Extraction** | Entity 간 관계 식별 |
| 4 | **Multi-hop Reasoning** | 여러 문장에 걸친 관계 추론 |

### 난이도

```
★★★★☆  (4/5)

NER까지는 접근이 어렵지 않으나,
relation 생성과 multi-hop 기반 schema 정립이 핵심 난관입니다.
```

---

## 📦 Dataset: DocRED

> **DocRED**: A Large-Scale Document-Level Relation Extraction Dataset  
> *Yao et al., ACL 2019* · [논문 링크](https://aclanthology.org/P19-1074)

Wikipedia와 Wikidata로부터 구축된 문서 수준 관계 추출 데이터셋입니다.

### 데이터 통계

| Split | 수량 |
|-------|------|
| train_annotated | 3,053 |
| train_distant | 101,873 |
| validation | 998 |
| test | 1,000 |

- 📁 다운로드 크기: **21.00 MB**
- 💾 생성 데이터 크기: **20.12 MB**

### 데이터 구조

```json
{
  "title": "Title of the document",
  "sents": [
    ["This", "is", "a", "sentence"],
    ["This", "is", "another", "sentence"]
  ],
  "vertexSet": [
    [
      {"name": "sentence", "pos": [3], "sent_id": 0, "type": "NN"},
      {"name": "sentence", "pos": [3], "sent_id": 1, "type": "NN"}
    ],
    [
      {"name": "This", "pos": [0], "sent_id": 0, "type": "NN"}
    ]
  ],
  "labels": {
    "head": [0],
    "tail": [0],
    "relation_id": ["P1"],
    "relation_text": ["is_a"],
    "evidence": [[0]]
  }
}
```

### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `title` | `string` | 문서 제목 |
| `sents` | `list[list[str]]` | 문장 리스트 (토큰 단위로 분리) |
| `vertexSet` | `list[list[dict]]` | Entity 목록. 하나의 Entity가 여러 문장에 등장 가능 (coreference) |
| `vertexSet[i][j].sent_id` | `int` | 해당 mention이 등장하는 문장 번호 |
| `vertexSet[i][j].pos` | `list[int]` | 문장 내 토큰 위치 |
| `labels.head` / `tail` | `int` | 관계 주체 / 대상 (vertexSet 인덱스) |
| `labels.relation_text` | `string` | 관계 유형 |
| `labels.evidence` | `list[int]` | 관계의 근거가 되는 문장 번호 |

---

## 🏗️ Project Structure
이 또한 미정

```
📁 project-root/
├── 📁 data/
│   ├── train_annotated.json
│   ├── train_distant.json
│   ├── dev.json
│   └── test.json
├── 📁 src/
│   ├── preprocess.py       # 데이터 전처리
│   ├── model.py            # 모델 정의
│   ├── train.py            # 학습 루프
│   └── evaluate.py         # 평가
├── 📁 notebooks/
│   └── EDA.ipynb           # 데이터 탐색
├── requirements.txt
└── README.md
```

---

## ⚙️ Getting Started

### 설치

```bash
https://github.com/yeseul-kim01/2026-Text2Graph.git
pip install -r requirements.txt
```

### 데이터 로드 (HuggingFace)

```python
from datasets import load_dataset

dataset = load_dataset("docred")
train = dataset["train_annotated"]
```

### 학습 실행

```bash
python src/train.py --model bert-base-uncased --epochs 10
```

---

## 📊 Expected Output

최종 결과는 아래 형태로 확장 가능해야 합니다.

```
Knowledge Graph
  └─ (Entity A) ──[relation]──▶ (Entity B)

Relational Database
  └─ head | relation | tail | evidence

Graph-based Reasoning System
  └─ Multi-hop path: A → B → C
```

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

## 👥 Team

| 이름 | 역할 |
|------|------|
| 김예슬 | 미정 |
| 박재윤 | 미정 |
| 박정현 | 미정 |
| 이수민 | 미정 |

---

<p align="center">Made with 🔥</strong>어서5시조🔥 · 2026-04-03 ~ 2026-04-10</p>
