"""
============================================================
Document Encoder Layer (encoder.py)
============================================================
역할: 전처리된 문서를 BERT에 입력하여 token-level contextual
      representation을 생성

INPUT:
  - input_ids      : [batch, seq_len]
  - attention_mask  : [batch, seq_len]

OUTPUT:
  - hidden_states  : [batch, seq_len, hidden_size(768)]
    각 토큰의 문맥 반영 벡터

기반 논문:
  - Devlin et al. (2019) BERT
  - Zhou et al. (2021) ATLOP — last 3 layers 활용

담당: 모델 담당

TODO ( 수정 포인트):
  - [ ] RoBERTa-large 인코더 지원 추가
  - [ ] Entity marker 삽입 시 special token 등록
============================================================
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DocumentEncoder(nn.Module):
    """
    BERT 기반 Document Encoder.
    ATLOP/DREEAM 논문을 따라 마지막 3개 layer의 hidden states를
    결합하여 더 풍부한 representation을 생성할 수 있음.

    Args:
        model_name  : HuggingFace 모델명 (e.g., 'bert-base-uncased')
        hidden_size : 인코더 hidden dimension (768 for base)
    """

    def __init__(self, model_name: str = "bert-base-uncased", hidden_size: int = 768):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size

        # ── Pre-trained BERT 로드 ──
        # Stage 1: last_hidden_state만 사용하므로 output_hidden_states 불필요
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        INPUT:
          - input_ids      : [batch, seq_len]
          - attention_mask  : [batch, seq_len]

        OUTPUT:
          - hidden_states  : [batch, seq_len, hidden_size]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Stage 1 (노트북 Stage1DocREModel과 동일): last_hidden_state만 사용
        # Stage 2+로 전환 시 ATLOP 스타일 3-layer 평균으로 교체 예정
        hidden_states = outputs.last_hidden_state

        return hidden_states  # [batch, seq_len, hidden_size]

    def get_output_dim(self) -> int:
        """인코더 출력 차원 반환"""
        return self.hidden_size
