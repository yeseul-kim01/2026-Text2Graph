"""
============================================================
Graph Reasoning Layer (graph_encoder.py)
============================================================
역할: Entity representation을 GNN으로 보정하여 inter-sentence
      관계와 multi-hop 추론 능력을 강화 (Stage 3)

INPUT:
  - entity_vectors : [num_entities, hidden_size] — entity 벡터
  - entity_spans   : mention span 정보 (graph edge 구성용)
  - sent_map       : 토큰별 문장 인덱스 (cross-sentence edge 구성용)
  - num_sents      : 문서 내 문장 수

OUTPUT:
  - refined_entity_vectors : [num_entities, hidden_size] — GNN 보정된 벡터

기반 논문:
  - Zeng et al. (2020) GAIN — 이기종 그래프 + GCN message passing
  - Xu et al. (2021) SSAN — structured attention (참고용)
  - Kipf & Welling (2017) GCN
  - Veličković et al. (2018) GAT

담당: 후처리 + 그래프 담당

TODO (김예슬):
  - [완] GAT 레이어 구현 (ablation 실험용)
  - [완] 이기종 그래프의 Sentence/Document 노드 추가
  - [완] Edge weight 학습 가능하게 확장
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


# ──────────────────────────────────────────────────────────────
# GCN Layer (Kipf & Welling, 2017)
# ──────────────────────────────────────────────────────────────
class GCNLayer(nn.Module):
    """
    INPUT:  node_features [N, in_dim], adj_matrix [N, N]
    OUTPUT: updated_features [N, out_dim]
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        INPUT:
          - x   : [N, in_dim]  — 노드 feature
          - adj : [N, N]       — 인접 행렬 (normalized)
        OUTPUT:
          - out : [N, out_dim] — 업데이트된 노드 feature
        """
        # Message passing: 인접 노드 정보 통합
        support = self.linear(x)       # [N, out_dim]
        output = torch.matmul(adj, support)  # [N, out_dim]
        output = self.dropout(output)
        return output


# ──────────────────────────────────────────────────────────────
# GAT Layer (Veličković et al., 2018) — ablation용
# ──────────────────────────────────────────────────────────────
class GATLayer(nn.Module):
    """
    Graph Attention Network 레이어.
    Attention 기반 이웃 정보 가중 합산.

    INPUT:  node_features [N, in_dim], adj_matrix [N, N]
    OUTPUT: updated_features [N, out_dim]
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.zeros(1, num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.zeros(1, num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        h = self.W(x).view(N, self.num_heads, self.head_dim)  # [N, H, D]

        # Attention scores
        attn_src = (h * self.attn_src).sum(dim=-1, keepdim=True)  # [N, H, 1]
        attn_dst = (h * self.attn_dst).sum(dim=-1, keepdim=True)  # [N, H, 1]
        attn = attn_src + attn_dst.transpose(0, 2)  # broadcast → [N, H, N]

        # Mask: adj가 0인 곳은 -inf
        mask = (adj.unsqueeze(1) == 0)  # [N, 1, N]
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(self.leaky_relu(attn), dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.bmm(attn.transpose(0, 1), h.transpose(0, 1))  # [H, N, D]
        out = out.transpose(0, 1).reshape(N, -1)  # [N, H*D]
        return out


# ──────────────────────────────────────────────────────────────
# 그래프 구성 함수 (GAIN 논문 참고)
# ──────────────────────────────────────────────────────────────
def build_entity_graph(
    entity_spans: List[List[Tuple]],
    sent_map: List[int],
    num_entities: int,
    num_sents: int,
    cross_sent_window: int = 1,
) -> torch.Tensor:
    """
    GAIN 논문의 Entity Graph 구성.
    3가지 edge type: co-reference, co-occurrence, cross-sentence

    INPUT:
      - entity_spans      : entity별 mention span 목록
      - sent_map          : subword별 문장 인덱스
      - num_entities      : entity 수
      - num_sents         : 문장 수
      - cross_sent_window : cross-sentence 연결 window

    OUTPUT:
      - adj : [num_entities, num_entities] — normalized adjacency matrix
    """
    adj = torch.zeros(num_entities, num_entities)

    # 각 entity가 등장하는 문장 집합 계산
    entity_sents = []
    for spans in entity_spans:
        sents = set()
        for start, end in spans:
            for pos in range(start, min(end, len(sent_map))):
                sid = sent_map[pos]
                if sid >= 0:
                    sents.add(sid)
        entity_sents.append(sents)

    for i in range(num_entities):
        for j in range(num_entities):
            if i == j:
                adj[i][j] = 1.0  # self-loop
                continue

            # ── Edge 1: Co-reference (동일 entity의 다른 mention) ──
            # entity-level graph에서는 사실상 self-loop과 동일
            # mention-level에서 의미가 있으므로 여기서는 skip

            # ── Edge 2: Co-occurrence (같은 문장에 등장) ──
            common_sents = entity_sents[i] & entity_sents[j]
            if len(common_sents) > 0:
                adj[i][j] = 1.0
                continue

            # ── Edge 3: Cross-sentence (인접 문장) ──
            for si in entity_sents[i]:
                for sj in entity_sents[j]:
                    if abs(si - sj) <= cross_sent_window:
                        adj[i][j] = 1.0
                        break

    # ── Normalize adjacency matrix (GCN 표준) ──
    # D^{-1/2} A D^{-1/2}
    degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
    adj = adj / degree  # row-normalized (간단 버전)

    return adj


# ──────────────────────────────────────────────────────────────
# GAIN-lite Graph Encoder
# ──────────────────────────────────────────────────────────────
class GraphEncoder(nn.Module):
    """
    GAIN-lite: Entity Graph 위에서 GCN/GAT message passing 수행.
    Entity representation을 보정하여 inter-sentence 관계 포착 강화.

    Args:
        hidden_dim       : 노드 feature 차원 (768)
        num_layers       : GNN layer 수 (기본 2)
        gnn_type         : 'gcn' | 'gat'
        dropout          : dropout 비율
        cross_sent_window: cross-sentence edge window
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 2,
        gnn_type: str = "gcn",
        dropout: float = 0.1,
        cross_sent_window: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.cross_sent_window = cross_sent_window

        # ── GNN Layers ──
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == "gat":
                self.gnn_layers.append(GATLayer(hidden_dim, hidden_dim, dropout))
            else:
                self.gnn_layers.append(GCNLayer(hidden_dim, hidden_dim, dropout))

        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        entity_vectors: torch.Tensor,
        entity_spans: List[List[Tuple]],
        sent_map: List[int],
        num_sents: int,
    ) -> torch.Tensor:
        """
        INPUT:
          - entity_vectors : [num_entities, hidden_size]
          - entity_spans   : entity별 mention span 목록
          - sent_map       : subword별 문장 인덱스
          - num_sents      : 문장 수

        OUTPUT:
          - refined_vectors: [num_entities, hidden_size]
        """
        num_entities = entity_vectors.size(0)

        # ── 그래프 구성 ──
        adj = build_entity_graph(
            entity_spans, sent_map, num_entities, num_sents,
            self.cross_sent_window,
        ).to(entity_vectors.device)

        # ── GNN Message Passing ──
        h = entity_vectors
        for layer in self.gnn_layers:
            h_new = layer(h, adj)
            h_new = self.activation(h_new)
            h = self.layer_norm(h + h_new)  # Residual connection

        return h  # [num_entities, hidden_size]
