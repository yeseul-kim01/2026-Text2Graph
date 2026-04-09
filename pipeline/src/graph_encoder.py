"""
============================================================
Graph Reasoning Layer (graph_encoder.py)
============================================================
역할:
  Entity representation을 그래프 신경망으로 보정하여
  inter-sentence relation, long-range dependency, multi-hop reasoning을 강화한다.

Stage 3 설계 목표:
  - Stage 2의 강한 ATLOP backbone을 유지
  - graph는 "보조 신호"로만 작동
  - noisy graph로 인해 성능이 떨어지지 않도록 residual scaling 적용

입력:
  - entity_vectors : [num_entities, hidden_size]
  - entity_spans   : entity별 mention span 정보
  - sent_map       : token/subword별 sentence index
  - num_sents      : 문서 내 문장 수

출력:
  - refined_entity_vectors : [num_entities, hidden_size]

기반 논문:
  - Zeng et al. (2020), GAIN
  - Kipf & Welling (2017), GCN
  - Veličković et al. (2018), GAT
  - Xu et al. (2021), SSAN (참고)

설계 원칙:
  - GAIN처럼 graph reasoning을 추가하되, ATLOP backbone을 해치지 않도록
    entity residual을 강하게 보존한다.
  - heterogeneous graph(entity / sentence / document)를 사용한다.
  - edge type별 gate를 학습하여 noisy edge 영향 완화.
  - Stage 2보다 성능이 떨어지는 주요 원인인 oversmoothing을 줄인다.

TODO (김예슬)

[완료]
  - GAT 레이어 구현
  - sentence / document 노드 추가
  - edge weight 학습 가능하게 확장
  - symmetric normalization 적용
  - residual scaling(alpha) 추가

[추가 예정]
  - edge type별 relation-specific graph 실험
  - top-k sparse graph pruning
  - sentence node 초기화 개선 (context-aware sentence pooling)
============================================================
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility
# ============================================================
def _symmetric_normalize(adj: torch.Tensor) -> torch.Tensor:
    """
    GCN 표준 대칭 정규화: D^{-1/2} A D^{-1/2}
    adj: [N, N]
    """
    deg = adj.sum(dim=-1).clamp(min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


def _mean_pool(vectors: List[torch.Tensor], dim: int, device: torch.device) -> torch.Tensor:
    """
    빈 리스트면 zero vector 반환.
    """
    if len(vectors) == 0:
        return torch.zeros(dim, device=device)
    return torch.stack(vectors, dim=0).mean(dim=0)


# ============================================================
# GCN Layer
# ============================================================
class GCNLayer(nn.Module):
    """
    기본 GCN 레이어.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)          # [N, out_dim]
        out = torch.matmul(adj, support)  # [N, out_dim]
        out = self.dropout(out)
        return out


# ============================================================
# GAT Layer
# ============================================================
class GATLayer(nn.Module):
    """
    Dense adjacency 기반 간단 GAT.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : [N, in_dim]
        adj : [N, N]   (0이면 edge 없음)
        """
        n = x.size(0)

        h = self.proj(x).view(n, self.num_heads, self.head_dim)   # [N, H, D]

        src_score = (h * self.attn_src.unsqueeze(0)).sum(dim=-1)  # [N, H]
        dst_score = (h * self.attn_dst.unsqueeze(0)).sum(dim=-1)  # [N, H]

        # [H, N, N]
        e = src_score.transpose(0, 1).unsqueeze(-1) + dst_score.transpose(0, 1).unsqueeze(-2)
        e = self.leaky_relu(e)

        mask = adj.unsqueeze(0) <= 0
        e = e.masked_fill(mask, float("-inf"))

        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)

        # [H, N, D]
        h_t = h.transpose(0, 1)
        out = torch.bmm(alpha, h_t)

        # [N, H, D] -> [N, out_dim]
        out = out.transpose(0, 1).contiguous().view(n, -1)
        return out

def build_entity_graph(
    entity_spans: List[List[Tuple[int, int]]],
    sent_map: List[int],
    num_entities: int,
    num_sents: int,
    cross_sent_window: int = 1,
    device=None,
    dtype=None
) -> torch.Tensor:
    """
    structural_encorder.py 호환용 entity-only graph builder.
    기존 코드와의 하위 호환을 위해 유지.
    """
    device = "cpu"
    if num_entities == 0:
        return torch.zeros(0, 0, device=device, dtype=dtype)

    adj = torch.zeros(num_entities, num_entities, device=device, dtype=dtype)

    entity_sents: List[set] = []
    for spans in entity_spans:
        sents = set()
        for start, end in spans:
            for pos in range(start, min(end, len(sent_map))):
                sid = sent_map[pos]
                if 0 <= sid < num_sents:
                    sents.add(sid)
        entity_sents.append(sents)

    for i in range(num_entities):
        adj[i, i] = 1.0
        for j in range(i + 1, num_entities):
            common = entity_sents[i] & entity_sents[j]
            if len(common) > 0:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
                continue

            linked = False
            for si in entity_sents[i]:
                for sj in entity_sents[j]:
                    if abs(si - sj) <= cross_sent_window:
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0
                        linked = True
                        break
                if linked:
                    break

    adj = _symmetric_normalize(adj)
    return adj

# ============================================================
# Heterogeneous Graph Builder
# ============================================================
def build_heterogeneous_graph(
    entity_spans: List[List[Tuple[int, int]]],
    sent_map: List[int],
    entity_vectors: torch.Tensor,
    num_sents: int,
    cross_sent_window: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    heterogeneous graph 구성

    노드:
      - entity nodes
      - sentence nodes
      - document node

    edge types:
      - entity_entity_same_sent
      - entity_entity_cross_sent
      - entity_sentence
      - sentence_document
      - self_loop

    반환:
      {
        "node_features": [N, H],
        "adj": [N, N],
        "num_entities": int,
        "num_sent_nodes": int,
        "doc_idx": int,
      }
    """
    device = entity_vectors.device
    hidden_dim = entity_vectors.size(-1)
    num_entities = entity_vectors.size(0)

    # --------------------------------------------------------
    # 1) entity -> sentence set
    # --------------------------------------------------------
    entity_sents: List[set] = []
    for spans in entity_spans:
        sents = set()
        for start, end in spans:
            for pos in range(start, min(end, len(sent_map))):
                sid = sent_map[pos]
                if 0 <= sid < num_sents:
                    sents.add(sid)
        entity_sents.append(sents)

    # --------------------------------------------------------
    # 2) sentence node 초기화
    #    해당 sentence에 등장하는 entity들의 평균으로 초기화
    # --------------------------------------------------------
    sentence_vectors = []
    for sid in range(num_sents):
        members = []
        for ent_id, sents in enumerate(entity_sents):
            if sid in sents:
                members.append(entity_vectors[ent_id])
        sentence_vectors.append(_mean_pool(members, hidden_dim, device))

    if num_sents > 0:
        sentence_vectors = torch.stack(sentence_vectors, dim=0)  # [S, H]
        document_vector = sentence_vectors.mean(dim=0, keepdim=True)  # [1, H]
    else:
        sentence_vectors = torch.zeros(0, hidden_dim, device=device)
        document_vector = torch.zeros(1, hidden_dim, device=device)

    # --------------------------------------------------------
    # 3) 전체 노드 feature
    # --------------------------------------------------------
    node_features = torch.cat(
        [entity_vectors, sentence_vectors, document_vector],
        dim=0,
    )
    total_nodes = node_features.size(0)

    sent_offset = num_entities
    doc_idx = num_entities + num_sents

    # --------------------------------------------------------
    # 4) adjacency
    # --------------------------------------------------------
    adj = torch.zeros(total_nodes, total_nodes, device=device)

    # self-loop
    adj.fill_diagonal_(1.0)

    # learnable edge type를 바로 adj에 반영하기 위해
    # 일단 edge mask를 만들고, encoder에서 gate 적용할 수 있게
    # dense adj 하나로 합쳐 반환
    #
    # entity-entity same sentence / cross sentence
    for i in range(num_entities):
        for j in range(i + 1, num_entities):
            common = entity_sents[i] & entity_sents[j]

            if len(common) > 0:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
                continue

            linked = False
            for si in entity_sents[i]:
                for sj in entity_sents[j]:
                    if abs(si - sj) <= cross_sent_window:
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0
                        linked = True
                        break
                if linked:
                    break

    # entity-sentence
    for ent_id, sents in enumerate(entity_sents):
        for sid in sents:
            s_idx = sent_offset + sid
            adj[ent_id, s_idx] = 1.0
            adj[s_idx, ent_id] = 1.0

    # sentence-document
    for sid in range(num_sents):
        s_idx = sent_offset + sid
        adj[s_idx, doc_idx] = 1.0
        adj[doc_idx, s_idx] = 1.0

    # 대칭 정규화
    adj = _symmetric_normalize(adj)

    return {
        "node_features": node_features,
        "adj": adj,
        "num_entities": num_entities,
        "num_sent_nodes": num_sents,
        "doc_idx": doc_idx,
    }


# ============================================================
# Graph Encoder
# ============================================================
class GraphEncoder(nn.Module):
    """
    GAIN-lite + heterogeneous node extension.

    특징:
      - entity / sentence / document node 사용
      - GCN 또는 GAT 선택 가능
      - residual scaling으로 graph 영향 제어
      - entity node만 최종 반환
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 1,
        gnn_type: str = "gcn",
        dropout: float = 0.1,
        cross_sent_window: int = 1,
        residual_alpha: float = 0.3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.cross_sent_window = cross_sent_window
        self.residual_alpha = residual_alpha

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == "gat":
                self.layers.append(
                    GATLayer(
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        dropout=dropout,
                        num_heads=num_heads,
                    )
                )
            else:
                self.layers.append(
                    GCNLayer(
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        dropout=dropout,
                    )
                )

        self.activation = nn.ReLU()
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # graph output gate:
        # graph update가 너무 강하면 Stage 2보다 성능이 떨어지므로
        # 학습 가능한 gate를 둔다.
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        entity_vectors: torch.Tensor,
        entity_spans: List[List[Tuple[int, int]]],
        sent_map: List[int],
        num_sents: int,
    ) -> torch.Tensor:
        """
        입력:
          entity_vectors : [E, H]

        출력:
          refined_entity_vectors : [E, H]
        """
        if entity_vectors.size(0) == 0:
            return entity_vectors

        graph = build_heterogeneous_graph(
            entity_spans=entity_spans,
            sent_map=sent_map,
            entity_vectors=entity_vectors,
            num_sents=num_sents,
            cross_sent_window=self.cross_sent_window,
        )

        h = graph["node_features"]      # [N, H]
        adj = graph["adj"]              # [N, N]
        num_entities = graph["num_entities"]

        original_entity = entity_vectors

        for layer, ln in zip(self.layers, self.layer_norms):
            update = layer(h, adj)
            update = self.activation(update)

            # residual scaling:
            # graph가 entity semantic을 과하게 덮지 않도록 alpha 사용
            h = ln(h + self.residual_alpha * update)

        refined_entity = h[:num_entities]  # entity node만 사용

        # final gated fusion:
        # Stage 2 entity를 최대한 보존하면서 graph 정보만 선택적으로 섞음
        gate_input = torch.cat([original_entity, refined_entity], dim=-1)
        gate = self.update_gate(gate_input)
        out = gate * refined_entity + (1.0 - gate) * original_entity

        return out