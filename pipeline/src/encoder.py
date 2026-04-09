"""
============================================================
Document Encoder Layer (encoder.py)
============================================================
역할:
  전처리된 문서를 Transformer Encoder(BERT 계열)에 입력하여
  token-level contextual representation을 생성한다.

입력:
  - input_ids       : [batch, seq_len]
  - attention_mask  : [batch, seq_len]

출력:
  1) return_attention=False
     - hidden_states : [batch, seq_len, hidden_size]

  2) return_attention=True
     - {
         "hidden_states": [batch, seq_len, hidden_size],
         "attentions":    [batch, num_heads, seq_len, seq_len]
       }

기반 논문:
  - Devlin et al. (2019), BERT
  - Zhou et al. (2021), ATLOP
  - Ma et al. (2023), DREEAM

구현 원칙:
  - hidden state는 마지막 3개 layer 평균 사용
    (ATLOP / DREEAM 스타일)
  - attention은 마지막 layer attention 사용
    (ATLOP localized context pooling, evidence supervision용)

TODO (김예슬)

[완료]
  - BERT-base 인코더 구현
  - config 기반 model_name / hidden_size 동적 설정
  - 마지막 3개 layer hidden states 평균 사용
  - get_output_dim() 메서드 추가
  - attention weights 반환 옵션 추가
  - return_attention=True일 때 hidden_states + attentions 함께 반환

[추가 예정]
  - RoBERTa-large / DeBERTa 계열 인코더 실험
  - entity marker 사용 시 special token 등록 및 resize
  - 마지막 layer attention 외 multi-layer attention aggregation 실험

담당:
  - 모델 담당: 김예슬
============================================================
"""

from typing import Dict, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class DocumentEncoder(nn.Module):
    """
    Transformer 기반 Document Encoder.

    기능:
      - 문서 전체 토큰에 대한 contextual hidden state 생성
      - 필요 시 attention weights도 함께 반환

    Args:
        model_name: HuggingFace 모델명
        hidden_size: 인코더 hidden dimension
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size

        # ---------------------------------------------------------
        # Pretrained Transformer 설정 로드
        # ---------------------------------------------------------
        # output_hidden_states=True:
        #   모든 layer의 hidden states를 받아 마지막 3개 layer 평균에 사용
        #
        # output_attentions=True:
        #   마지막 layer attention을 ATLOP / DREEAM 문맥 추출에 사용
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.config.output_attentions = True

        self.bert = AutoModel.from_pretrained(model_name, config=self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        문서 인코딩 수행.

        Args:
            input_ids:
                [batch, seq_len]
            attention_mask:
                [batch, seq_len]
            return_attention:
                True이면 hidden_states와 attentions를 함께 반환.
                False이면 hidden_states만 반환.

        Returns:
            1) return_attention=False
               Tensor[batch, seq_len, hidden_size]

            2) return_attention=True
               {
                 "hidden_states": Tensor[batch, seq_len, hidden_size],
                 "attentions": Tensor[batch, num_heads, seq_len, seq_len]
               }
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # ---------------------------------------------------------
        # Hidden States
        # ---------------------------------------------------------
        # ATLOP / DREEAM 스타일:
        # 마지막 3개 layer의 hidden states 평균 사용
        #
        # outputs.hidden_states:
        #   tuple(length = num_layers + 1)
        #   각 원소 shape = [batch, seq_len, hidden_size]
        all_hidden = outputs.hidden_states
        hidden_states = (all_hidden[-1] + all_hidden[-2] + all_hidden[-3]) / 3.0

        # attention이 필요 없는 경우 hidden_states만 반환
        if not return_attention:
            return hidden_states

        # ---------------------------------------------------------
        # Attentions
        # ---------------------------------------------------------
        # ATLOP localized context pooling에서는
        # entity mention의 attention을 사용해 pair-specific context(rs)를 만든다.
        #
        # 여기서는 마지막 layer attention만 사용:
        # [batch, num_heads, seq_len, seq_len]
        attentions = outputs.attentions[-1]

        return {
            "hidden_states": hidden_states,
            "attentions": attentions,
        }

    def get_output_dim(self) -> int:
        """
        인코더 출력 차원 반환.
        relation head / graph encoder 초기화 시 사용 가능.
        """
        return self.hidden_size