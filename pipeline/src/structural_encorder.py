# ──────────────────────────────────────────────────────────────
# Graph U-Net 전용 Pooling & Unpooling 모듈
# ──────────────────────────────────────────────────────────────
class GraphTopKPool(nn.Module):
    """
    학습 가능한 투영 벡터를 사용해 중요도가 높은 상위 K개의 노드만 남기는 풀링 레이어
    """
    def __init__(self, in_dim: int, ratio: float = 0.5):
        super().__init__()
        self.ratio = ratio
        self.score_layer = nn.Linear(in_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        k = max(1, int(num_nodes * self.ratio))  # 최소 1개의 노드는 보존

        # 1. 노드별 중요도 점수 계산 및 0~1 사이로 정규화 (Sigmoid)
        scores = self.score_layer(x).squeeze(-1)  # [N]
        scores = torch.sigmoid(scores)

        # 2. 상위 K개 노드의 인덱스 추출 및 원래 등장 순서대로 정렬
        _, top_idx = torch.topk(scores, k)
        top_idx = torch.sort(top_idx)[0]

        # 3. 노드 피처 축소 (중요도 점수를 곱해 역전파가 가능하도록 Gate 역할 수행)
        x_pooled = x[top_idx] * scores[top_idx].unsqueeze(-1)
        
        # 4. 인접 행렬 축소 (선택된 노드들끼리의 연결만 남김)
        adj_pooled = adj[top_idx][:, top_idx]

        return x_pooled, adj_pooled, top_idx


class GraphUnpool(nn.Module):
    """
    저장해둔 인덱스를 바탕으로 축소되었던 그래프를 원래 크기 N으로 복원하는 레이어
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_pooled: torch.Tensor, top_idx: torch.Tensor, orig_size: int) -> torch.Tensor:
        dim = x_pooled.size(-1)
        device = x_pooled.device
        
        # 원래 노드 개수(orig_size)만큼 빈 텐서(0) 생성
        x_unpooled = torch.zeros((orig_size, dim), device=device)
        
        # 풀링에서 살아남았던 노드들의 위치(top_idx)에 Bottleneck 피처를 채워 넣음
        x_unpooled[top_idx] = x_pooled
        return x_unpooled


# ──────────────────────────────────────────────────────────────
# Graph U-Net Encoder (기존 GraphEncoder와 교체 가능한 모듈)
# ──────────────────────────────────────────────────────────────
class GraphUNetEncoder(nn.Module):
    """
    Zero-Padding + Skip Connection 방식을 적용한 U-Net 형태의 Graph Reasoning Layer.
    기존 GraphEncoder와 동일한 Input/Output Interface를 가집니다.
    """
    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 2,  # U-Net에서는 깊이를 조절하는 용도로 사용 가능
        gnn_type: str = "gcn",
        dropout: float = 0.1,
        cross_sent_window: int = 1,
        pool_ratio: float = 0.5,  # U-Net 전용 파라미터 추가
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_sent_window = cross_sent_window
        
        # 사용자가 선택한 GNN 타입에 맞춰 레이어 생성
        GNNLayer = GATLayer if gnn_type == "gat" else GCNLayer

        # ── 1. Encoder (수축 경로) ──
        self.enc_gnn = GNNLayer(hidden_dim, hidden_dim, dropout)
        self.pool = GraphTopKPool(hidden_dim, pool_ratio)
        
        # ── 2. Bottleneck (글로벌 추론) ──
        self.bottleneck_gnn = GNNLayer(hidden_dim, hidden_dim, dropout)
        
        # ── 3. Decoder (확장 경로) ──
        self.unpool = GraphUnpool()
        self.dec_gnn = GNNLayer(hidden_dim, hidden_dim, dropout)

        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        entity_vectors: torch.Tensor,
        entity_spans: List[List[Tuple]],
        sent_map: List[int],
        num_sents: int,
    ) -> torch.Tensor:
        orig_size = entity_vectors.size(0)

        # ── 그래프 구성 (기존 모듈 활용) ──
        adj = build_entity_graph(
            entity_spans, sent_map, orig_size, num_sents, self.cross_sent_window
        ).to(entity_vectors.device)

        # ── Step 1: 인코딩 및 Skip Connection 저장 ──
        h_enc = self.enc_gnn(entity_vectors, adj)
        h_enc = self.activation(h_enc)
        skip_connection = h_enc.clone()  # 원본 노드들의 로컬 문맥 보존

        # ── Step 2: 풀링 (Down-sampling) ──
        h_pooled, adj_pooled, top_idx = self.pool(h_enc, adj)

        # ── Step 3: 병목 구간 (Global Reasoning) ──
        h_mid = self.bottleneck_gnn(h_pooled, adj_pooled)
        h_mid = self.activation(h_mid)

        # ── Step 4: 언풀링 (Up-sampling) 및 융합 ──
        h_unpooled = self.unpool(h_mid, top_idx, orig_size)
        
        # [핵심] Skip Connection 융합 (Zero-padded 공간을 로컬 피처로 채움)
        h_dec = h_unpooled + skip_connection
        
        h_out = self.dec_gnn(h_dec, adj)
        h_out = self.activation(h_out)

        # ── Step 5: 최종 Residual Connection ──
        # 오리지널(Stage 2 RoBERTa) 벡터에 U-Net이 추론한 글로벌/로컬 그래프 델타값을 더해줌
        out = self.layer_norm(entity_vectors + h_out)
        
        return out