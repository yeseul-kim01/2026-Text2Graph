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

TODO(완 - 김예슬):
  - [완] BERT-base 인코더 구현
  - [완] ATLOP/DREEAM 스타일로 마지막 3개 layer의 hidden states 평균 사용
  - [수정] 모델명과 hidden_size를 config에서 동적으로 설정 가능하도록 수정
  - [완] get_output_dim() 메서드로 인코더 출력 차원 반환 기능 추가
  - [결과] 모델 초기화 시 config에서 모델명과 hidden_size를 읽어와서 BERT 모델을 로드하도록 수정
  
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
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True  # 모든 layer hidden states 출력
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        INPUT:
          - input_ids      : [batch, seq_len]
          - attention_mask  : [batch, seq_len]

        OUTPUT:
          - hidden_states  : [batch, seq_len, hidden_size]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # ATLOP/DREEAM 스타일: 마지막 3개 layer의 평균 사용
        # (DREEAM 논문 footnote 6 참조)
        all_hidden = outputs.hidden_states  # tuple of [batch, seq, hidden]
        # 마지막 3개 layer 평균
        hidden_states = (all_hidden[-1] + all_hidden[-2] + all_hidden[-3]) / 3.0

        return hidden_states  # [batch, seq_len, hidden_size]

    def get_output_dim(self) -> int:
        """인코더 출력 차원 반환"""
        return self.hidden_size
