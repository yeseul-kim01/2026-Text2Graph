"""
============================================================
Preprocessing Layer (preprocessing.py)
============================================================
역할: DocRED raw JSON 데이터를 Transformer 모델이 학습할 수 있는
      입력 구조로 변환하는 핵심 전처리 모듈

INPUT:
  - DocRED raw JSON (sents, vertexSet, labels)
  - BERT Tokenizer

OUTPUT:
  - input_ids       : [batch, seq_len]  — BERT 토큰 인덱스
  - attention_mask   : [batch, seq_len]  — 패딩 마스크
  - entity_spans     : List[List[Tuple]] — entity별 mention span 목록
  - entity_pairs     : List[Tuple]       — (head_id, tail_id) pair 목록
  - labels           : [num_pairs, num_relations] — multi-hot label
  - sent_map         : List[int]         — 토큰별 문장 인덱스 (Stage 3용)
  - evidence_labels  : Optional          — 근거 문장 정보 (DREEAM용)

기반 논문:
  - Yao et al. (2019) DocRED
  - Zhou et al. (2021) ATLOP — prepro.py 참고
  - Ma et al. (2023) DREEAM — evidence label 처리

담당: 전처리 담당

TODO ( 수정 포인트):
  - [ ] DocRED 데이터 경로 확인 및 다운로드 스크립트 추가
  - [ ] Entity marker 삽입 로직 (Stage 2 확장 시)
  - [ ] Re-DocRED 등 다른 데이터셋 지원
============================================================
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# ──────────────────────────────────────────────────────────────
# RelationMapper — 노트북 Cell 6 기반
# ──────────────────────────────────────────────────────────────
class RelationMapper:
    """
    DocRED relation string ↔ integer id 변환.
    노트북 Cell 6 기반: sorted key 정렬로 재현성 보장.

    사용 방법 두 가지:
      1. from_rel_info(rel_info) : rel_info.json 딕셔너리에서 생성 (노트북 방식)
      2. from_rel2id(rel2id)     : 기존 rel2id 딕셔너리 래핑 (파이프라인 방식)
    """

    def __init__(self, rel2id: Dict[str, int]):
        self.rel2id: Dict[str, int] = rel2id
        self.id2rel: Dict[int, str] = {v: k for k, v in rel2id.items()}

    @classmethod
    def from_rel_info(cls, rel_info: Dict) -> "RelationMapper":
        """
        노트북 방식: rel_info.json 딕셔너리에서 sorted key로 rel2id 생성.
        (노트북 Cell 6의 RelationMapper.__init__과 동일)
        """
        rel2id = {rel: idx for idx, rel in enumerate(sorted(rel_info.keys()))}
        return cls(rel2id)

    @classmethod
    def from_rel2id(cls, rel2id: Dict[str, int]) -> "RelationMapper":
        """파이프라인 방식: 기존 rel2id 딕셔너리(rel2id.json) 래핑."""
        return cls(rel2id)

    def get_id(self, rel: str) -> int:
        return self.rel2id.get(rel, -1)

    def get_rel(self, idx: int) -> str:
        return self.id2rel.get(idx, "UNK")

    def __len__(self) -> int:
        return len(self.rel2id)


def load_rel2id_from_rel_info(meta_dir: str, filename: str = "rel_info.json") -> Dict[str, int]:
    """
    rel_info.json에서 sorted key로 rel2id 생성.
    노트북의 RelationMapper.from_rel_info()와 동일한 ID 할당.
    (rel2id.json이 없고 rel_info.json만 있을 때 사용)
    """
    filepath = os.path.join(meta_dir, filename)
    with open(filepath, "r") as f:
        rel_info = json.load(f)
    rel2id = {rel: idx for idx, rel in enumerate(sorted(rel_info.keys()))}
    print(f"[Preprocessing] Built rel2id from {filename}: {len(rel2id)} relations")
    return rel2id


# ──────────────────────────────────────────────────────────────
# DocRED 데이터셋 클래스
# ──────────────────────────────────────────────────────────────
class DocREDDataset(Dataset):
    """
    DocRED 데이터셋을 PyTorch Dataset으로 래핑.

    Args:
        data_dir    : DocRED JSON 파일이 있는 디렉토리
        data_file   : JSON 파일명 (e.g., 'train_annotated.json')
        tokenizer   : HuggingFace tokenizer
        rel2id      : relation -> id 매핑 딕셔너리
        max_seq_len : 최대 시퀀스 길이 (기본 512)
        stage       : 'stage1' | 'stage2' | 'stage3'
        teacher_attns : Optional, DREEAM silver evidence attention (Stage 2)
    """

    def __init__(
        self,
        data_dir: str,
        data_file: str,
        tokenizer,
        rel2id: Dict[str, int],
        max_seq_len: int = 512,
        stage: str = "stage1",
        teacher_attns: Optional[Dict] = None,
    ):
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.num_relations = len(rel2id)
        self.max_seq_len = max_seq_len
        self.stage = stage
        self.teacher_attns = teacher_attns

        # ── JSON 로드 ──
        filepath = os.path.join(data_dir, data_file)
        with open(filepath, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        # ── 전처리된 features 생성 ──
        self.features = []
        for doc_idx, doc in enumerate(self.raw_data):
            feature = self._process_document(doc, doc_idx)
            if feature is not None:
                self.features.append(feature)

        print(f"[Preprocessing] Loaded {len(self.features)} documents from {data_file}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    # ──────────────────────────────────────────────────────────
    # Step 1~4: 문서 단위 전처리 파이프라인
    # ──────────────────────────────────────────────────────────
    def _process_document(self, doc: Dict, doc_idx: int) -> Optional[Dict]:
        """
        하나의 DocRED 문서를 모델 입력 feature로 변환.

        INPUT:  doc — DocRED raw JSON 문서 1개
        OUTPUT: Dict with {input_ids, attention_mask, entity_spans,
                entity_pairs, labels, sent_map, ...}
        """
        sents = doc["sents"]
        vertex_set = doc["vertexSet"]
        labels = doc.get("labels", [])
        title = doc.get("title", "")

        # ── Step 1: Sentence/Token 정리 ──
        # 문장별 토큰 리스트 → 하나의 문서 시퀀스로 연결
        tokens = []
        sent_map = []  # 각 토큰이 속한 문장 인덱스
        sent_start_positions = []  # 각 문장의 시작 위치

        for sent_idx, sent_tokens in enumerate(sents):
            sent_start_positions.append(len(tokens))
            for token in sent_tokens:
                tokens.append(token)
                sent_map.append(sent_idx)

        # ── BERT Tokenizer 적용 ──
        # 원본 토큰 → subword 토큰 변환 & offset mapping
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=False,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)          # [seq_len]
        attention_mask = encoded["attention_mask"].squeeze(0)  # [seq_len]

        # word_ids: 각 subword가 원본 몇 번째 토큰에 해당하는지
        word_ids = encoded.word_ids(batch_index=0)

        # subword → 문장 인덱스 매핑 (Stage 3 그래프 구성에 사용)
        subword_sent_map = []
        for wid in word_ids:
            if wid is not None and wid < len(sent_map):
                subword_sent_map.append(sent_map[wid])
            else:
                subword_sent_map.append(-1)  # [CLS], [SEP], [PAD]

        # ── Step 2: Entity Mention Alignment ──
        # sent_id + pos → 문서 전체 subword 인덱스로 변환
        entity_spans = []  # entity별 [(start, end), ...]
        entity_types = []  # entity별 type

        for entity in vertex_set:
            spans = []
            etype = entity[0].get("type", "UNK")
            for mention in entity:
                sent_id = mention["sent_id"]
                pos_start = mention["pos"][0]
                pos_end = mention["pos"][1]

                # 원본 토큰 인덱스 계산
                if sent_id < len(sent_start_positions):
                    abs_start = sent_start_positions[sent_id] + pos_start
                    abs_end = sent_start_positions[sent_id] + pos_end
                else:
                    continue

                # 원본 토큰 인덱스 → subword 인덱스로 변환
                sw_start, sw_end = self._find_subword_span(word_ids, abs_start, abs_end)
                if sw_start is not None and sw_end is not None:
                    spans.append((sw_start, sw_end))

            entity_spans.append(spans)
            entity_types.append(etype)

        num_entities = len(entity_spans)
        if num_entities < 2:
            return None  # entity 2개 미만이면 관계 추출 불가

        # ── Step 3: Entity Pair Generation ──
        # N개 entity → N*(N-1) 방향성 pair 생성
        entity_pairs = []
        for h in range(num_entities):
            for t in range(num_entities):
                if h != t:
                    entity_pairs.append((h, t))

        # ── Step 4: Relation Label 생성 ──
        # pair별 multi-hot vector (96 relations + 1 no_relation)
        pair_to_idx = {(h, t): i for i, (h, t) in enumerate(entity_pairs)}
        num_pairs = len(entity_pairs)
        relation_labels = np.zeros((num_pairs, self.num_relations), dtype=np.float32)

        # Evidence labels (DREEAM용)
        evidence_labels = {}  # pair_idx -> {rel_id: [sent_ids]}

        for label in labels:
            h = label["h"]
            t = label["t"]
            rel = label["r"]
            evidence = label.get("evidence", [])

            if (h, t) in pair_to_idx and rel in self.rel2id:
                pair_idx = pair_to_idx[(h, t)]
                rel_idx = self.rel2id[rel]
                relation_labels[pair_idx][rel_idx] = 1.0

                # Evidence 정보 저장
                if pair_idx not in evidence_labels:
                    evidence_labels[pair_idx] = {}
                evidence_labels[pair_idx][rel_idx] = evidence

        # no_relation: 어떤 relation도 없는 pair에 대해 표시
        # (ATLOP에서는 별도 TH class로 처리하므로 여기서는 생략)

        # ── [버그 수정 — 김예슬] Na 라벨 부여 ──
        # 어떤 relation도 없는 pair에 Na(index 0) = 1.0 부여.
        # ⚠ 반드시 torch.tensor() 호출 전에 해야 함!
        #    torch.tensor()는 numpy를 복사하므로, 이후에 numpy를 수정해도
        #    이미 만들어진 텐서에는 반영되지 않음.
        for pair_idx in range(num_pairs):
            if relation_labels[pair_idx].sum() == 0:
                relation_labels[pair_idx][0] = 1.0  # Na 라벨 부여

        feature = {
            "doc_idx": doc_idx,
            "title": title,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "entity_spans": entity_spans,
            "entity_types": entity_types,
            "entity_pairs": entity_pairs,
            "labels": torch.tensor(relation_labels, dtype=torch.float32),
            "sent_map": subword_sent_map,
            "num_sents": len(sents),
            "num_entities": num_entities,
            "evidence_labels": evidence_labels,
        }

        # DREEAM teacher attention (silver evidence)
        if self.teacher_attns is not None and doc_idx in self.teacher_attns:
            feature["teacher_attns"] = self.teacher_attns[doc_idx]

        return feature

    # ──────────────────────────────────────────────────────────
    # 유틸리티: 원본 토큰 인덱스 → subword span 변환
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _find_subword_span(word_ids, abs_start, abs_end):
        """
        원본 토큰 범위 [abs_start, abs_end) → subword 인덱스 범위로 변환.

        INPUT:  word_ids — tokenizer의 word_ids 출력
                abs_start, abs_end — 원본 토큰 인덱스 범위
        OUTPUT: (sw_start, sw_end) — subword 인덱스 범위, 못 찾으면 (None, None)
        """
        sw_start = None
        sw_end = None
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid == abs_start and sw_start is None:
                sw_start = i
            if wid == abs_end - 1:
                sw_end = i + 1
        return sw_start, sw_end


# ──────────────────────────────────────────────────────────────
# DataLoader용 collate function
# ──────────────────────────────────────────────────────────────
def docred_collate_fn(batch: List[Dict]) -> Dict:
    """
    DataLoader에서 사용할 collate function.
    가변 길이 entity/pair 정보를 배치로 묶는다.

    INPUT:  List of feature dicts
    OUTPUT: Batched dict with padded tensors
    """
    input_ids = torch.stack([f["input_ids"] for f in batch])
    attention_mask = torch.stack([f["attention_mask"] for f in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "entity_spans": [f["entity_spans"] for f in batch],
        "entity_types": [f["entity_types"] for f in batch],
        "entity_pairs": [f["entity_pairs"] for f in batch],
        "labels": [f["labels"] for f in batch],
        "sent_map": [f["sent_map"] for f in batch],
        "num_sents": [f["num_sents"] for f in batch],
        "num_entities": [f["num_entities"] for f in batch],
        "evidence_labels": [f["evidence_labels"] for f in batch],
        "doc_idx": [f["doc_idx"] for f in batch],
        "title": [f["title"] for f in batch],
    }


# ──────────────────────────────────────────────────────────────
# rel2id 로드 유틸리티
# ──────────────────────────────────────────────────────────────
def load_rel2id(meta_dir: str, filename: str = "rel2id.json") -> Dict[str, int]:
    """
    relation → id 매핑 딕셔너리 로드.

    INPUT:  meta_dir — rel2id.json 경로
    OUTPUT: Dict[str, int] — e.g., {"P17": 0, "P131": 1, ...}
    """
    filepath = os.path.join(meta_dir, filename)
    with open(filepath, "r") as f:
        rel2id = json.load(f)
    print(f"[Preprocessing] Loaded {len(rel2id)} relations from {filename}")
    return rel2id


# ──────────────────────────────────────────────────────────────
# DataLoader 생성 함수
# ──────────────────────────────────────────────────────────────
def create_dataloader(
    data_dir: str,
    data_file: str,
    tokenizer,
    rel2id: Dict,
    max_seq_len: int = 512,
    batch_size: int = 4,
    shuffle: bool = True,
    stage: str = "stage1",
    teacher_attns: Optional[Dict] = None,
) -> DataLoader:
    """
    DataLoader 생성 헬퍼 함수.

    INPUT:  데이터 경로, tokenizer, 설정값
    OUTPUT: PyTorch DataLoader
    """
    dataset = DocREDDataset(
        data_dir=data_dir,
        data_file=data_file,
        tokenizer=tokenizer,
        rel2id=rel2id,
        max_seq_len=max_seq_len,
        stage=stage,
        teacher_attns=teacher_attns,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=docred_collate_fn,
        num_workers=0,  # Colab 호환성
        pin_memory=True,
    )
