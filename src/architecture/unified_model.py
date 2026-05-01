"""Unified enforcement prediction model: FT-Transformer + LSTM + GNN.

Architecture (Phase 1-3):
  FT-Transformer backbone → [optional GNN spatial] → [optional LSTM temporal] → task heads

Phase 1: FT-Transformer backbone + Occurrence/Speed heads
Phase 2: LSTM temporal encoder — processes last 168 hours of per-cell activity
Phase 3: GNN spatial encoder — propagates signal between neighboring H3 cells

Each numeric feature gets a Linear(1, d_token) projection.
Each categorical feature gets an nn.Embedding → Linear(d_cat, d_token).
Self-attention learns ALL pairwise feature interactions at once.
The [CLS] token's output is the shared cell/stop embedding.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Feature Tokenizer ----------

class NumericTokenizer(nn.Module):
    """Project each numeric feature independently into d_token space."""

    def __init__(self, n_numeric: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_numeric, d_token))
        self.bias = nn.Parameter(torch.empty(n_numeric, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_numeric)
        # out: (batch, n_numeric, d_token)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalTokenizer(nn.Module):
    """Embed each categorical feature, then project to d_token."""

    def __init__(self, cardinalities: List[int], d_token: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, min(card, 32))  # +1 for unknown/padding
            for card in cardinalities
        ])
        self.projections = nn.ModuleList([
            nn.Linear(min(card, 32), d_token)
            for card in cardinalities
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_categorical) — integer indices
        tokens = []
        for i, (emb, proj) in enumerate(zip(self.embeddings, self.projections)):
            tokens.append(proj(emb(x[:, i])))
        return torch.stack(tokens, dim=1)  # (batch, n_cat, d_token)


# ---------- Transformer ----------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> MHSA -> residual -> LN -> FFN -> residual.

    Canonical FT-Transformer (rtdl-revisiting-models make_default):
        - FFN activation: ReGLU (ReLU(gate) * value), hidden = 4/3 * d_token
        - attention_dropout = 0.2
        - ffn_dropout = 0.1
        - residual_dropout = 0.1 (was missing before 2026-04-24)
    """

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
        attention_dropout: Optional[float] = None,
        residual_dropout: Optional[float] = None,
    ):
        super().__init__()
        attn_p = attention_dropout if attention_dropout is not None else 0.2
        res_p = residual_dropout if residual_dropout is not None else 0.1
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=attn_p, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        # ReGLU FFN: project to 2 * d_ffn, split into (gate, value), apply
        # ReLU to gate then elementwise-multiply, project back to d_token.
        self.ffn_up = nn.Linear(d_token, 2 * d_ffn)
        self.ffn_down = nn.Linear(d_ffn, d_token)
        self.ffn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(res_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm + residual dropout
        normed = self.norm1(x)
        attn_out = self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.residual_dropout(attn_out)
        # ReGLU FFN with pre-norm + residual dropout
        n = self.norm2(x)
        gate, value = self.ffn_up(n).chunk(2, dim=-1)
        h = torch.relu(gate) * value
        ffn_out = self.ffn_down(self.ffn_dropout(h))
        x = x + self.residual_dropout(ffn_out)
        return x


class FTTransformerBackbone(nn.Module):
    """Feature Tokenizer Transformer: tokenize → [CLS] + tokens → L blocks → [CLS] output."""

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: List[int],
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ffn: int = 256,
        dropout: float = 0.1,
        output_dim: int = 128,
    ):
        super().__init__()
        self.numeric_tokenizer = NumericTokenizer(n_numeric, d_token)
        self.categorical_tokenizer = CategoricalTokenizer(cat_cardinalities, d_token) if cat_cardinalities else None

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_token)
        self.head_proj = nn.Linear(d_token, output_dim)

    def forward(
        self,
        x_numeric: torch.Tensor,
        x_categorical: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Tokenize numeric features
        tokens = self.numeric_tokenizer(x_numeric)  # (B, N_num, d)

        # Tokenize categoricals if present
        if self.categorical_tokenizer is not None and x_categorical is not None:
            cat_tokens = self.categorical_tokenizer(x_categorical)  # (B, N_cat, d)
            tokens = torch.cat([tokens, cat_tokens], dim=1)

        # Prepend [CLS] token
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+N, d)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Extract [CLS] output and project
        cls_out = self.norm(tokens[:, 0])  # (B, d_token)
        return self.head_proj(cls_out)  # (B, output_dim)


# ---------- Task Heads ----------

class OccurrenceHead(nn.Module):
    """Predicts expected stop count (Poisson).

    Input: backbone_output (128-d) + time_features (12-d)
    Output: log(lambda) → exp() gives expected count
    """

    def __init__(self, backbone_dim: int, time_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(backbone_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, backbone_out: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([backbone_out, time_features], dim=-1)
        return self.mlp(x).squeeze(-1)  # (B,) — log(lambda)


class SpeedHead(nn.Module):
    """Predicts P(is_speed_related).

    Input: backbone_output (128-d) + per_stop_features (~70-d)
    Output: logit → sigmoid gives probability
    """

    def __init__(self, backbone_dim: int, stop_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(backbone_dim + stop_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, backbone_out: torch.Tensor, stop_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([backbone_out, stop_features], dim=-1)
        return self.mlp(x).squeeze(-1)  # (B,) — logit


# ---------- Phase 2: Temporal Encoder ----------

class TemporalEncoder(nn.Module):
    """LSTM that processes recent enforcement history per cell.

    Input:  (batch, seq_len, input_dim) — last 168 hours of per-cell activity
            Each timestep: (stop_count, hour_sin, hour_cos, dow_sin, dow_cos, is_weekend)
    Output: (batch, hidden_dim) — temporal state
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: (batch, 168, 6)
        _, (h_n, _) = self.lstm(sequence)
        return h_n[-1]  # (batch, hidden_dim) — final layer's hidden state


# ---------- Phase 3: Spatial Encoder ----------

class SpatialEncoder(nn.Module):
    """Graph Attention Network for spatial message passing between H3 cells.

    Uses simple attention-weighted neighbor aggregation (no torch_geometric
    dependency for Phase 3 MVP — can upgrade to GATConv later).

    Input:  (N_cells, in_dim) — all cell embeddings + edge_index
    Output: (N_cells, out_dim) — neighbor-enriched embeddings
    """

    def __init__(self, in_dim: int = 128, out_dim: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        # Attention parameters per head
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        # Second layer
        self.W2 = nn.Linear(out_dim, out_dim, bias=False)
        self.a_src2 = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.a_dst2 = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.norm2 = nn.LayerNorm(out_dim)

    def _gat_layer(self, x, edge_index, W, a_src, a_dst, norm):
        """One GAT attention layer."""
        N = x.size(0)
        h = W(x).view(N, self.n_heads, self.head_dim)  # (N, H, D)

        # Attention scores
        src_score = (h * a_src.unsqueeze(0)).sum(-1)  # (N, H)
        dst_score = (h * a_dst.unsqueeze(0)).sum(-1)  # (N, H)

        # For each edge, compute attention
        src_idx = edge_index[0]  # (E,)
        dst_idx = edge_index[1]  # (E,)
        edge_attn = F.leaky_relu(
            src_score[src_idx] + dst_score[dst_idx], 0.2
        )  # (E, H)

        # Softmax over neighbors (sparse)
        # Build per-destination normalization
        attn_exp = torch.exp(edge_attn - edge_attn.max())  # numerical stability
        denom = torch.zeros(N, self.n_heads, device=x.device)
        denom.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(attn_exp), attn_exp)
        attn_norm = attn_exp / (denom[dst_idx] + 1e-8)  # (E, H)
        attn_norm = self.dropout(attn_norm)

        # Aggregate: for each destination, sum(attn * src_features)
        msg = h[src_idx] * attn_norm.unsqueeze(-1)  # (E, H, D)
        out = torch.zeros(N, self.n_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst_idx.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)

        out = out.view(N, -1)  # (N, out_dim)
        return norm(F.elu(out) + x)  # residual

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self._gat_layer(x, edge_index, self.W, self.a_src, self.a_dst, self.norm)
        x = self._gat_layer(x, edge_index, self.W2, self.a_src2, self.a_dst2, self.norm2)
        return x


# ---------- Unified Model ----------

class UnifiedModel(nn.Module):
    """Full v4.0.0 unified enforcement prediction model.

    Shared FT-Transformer backbone → two task heads:
      - Occurrence: Poisson count per (cell, hour, day)
      - Speed: binary P(is_speed_related) per stop

    Phase 2 adds: TemporalEncoder (LSTM) between backbone and heads
    Phase 3 adds: SpatialEncoder (GNN) between backbone and LSTM
    """

    def __init__(
        self,
        n_cell_numeric: int,
        cell_cat_cardinalities: List[int],
        n_time_features: int,
        n_stop_numeric: int,
        stop_cat_cardinalities: List[int],
        d_token: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        backbone_dim: int = 128,
        dropout: float = 0.1,
        use_temporal: bool = False,
        temporal_input_dim: int = 6,
        temporal_hidden_dim: int = 64,
        use_spatial: bool = False,
    ):
        super().__init__()
        self.backbone_dim = backbone_dim

        # Shared backbone processes cell-level features.
        # d_ffn is the ReGLU-hidden width (effective FFN hidden dim);
        # canonical FT-T uses 4/3 * d_token, not 4 * d_token.
        self.backbone = FTTransformerBackbone(
            n_numeric=n_cell_numeric,
            cat_cardinalities=cell_cat_cardinalities,
            d_token=d_token,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ffn=int(d_token * 4 / 3),
            dropout=dropout,
            output_dim=backbone_dim,
        )

        # Phase 3: Spatial encoder (GNN over H3 cell graph)
        self.spatial_encoder = SpatialEncoder(
            backbone_dim, backbone_dim, n_heads=4, dropout=dropout,
        ) if use_spatial else None

        # Phase 2: Temporal encoder (LSTM over recent activity)
        self.temporal_encoder = TemporalEncoder(
            temporal_input_dim, temporal_hidden_dim, num_layers=2, dropout=dropout,
        ) if use_temporal else None

        # Fuse backbone + temporal into backbone_dim (keeps head dims unchanged)
        if use_temporal:
            self.temporal_fuse = nn.Sequential(
                nn.Linear(backbone_dim + temporal_hidden_dim, backbone_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # Per-stop feature encoder (simple MLP, not transformer — these
        # features only exist for the speed head, not occurrence)
        total_stop_cat_dim = sum(min(c, 32) for c in stop_cat_cardinalities)
        self.stop_cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, min(card, 32))
            for card in stop_cat_cardinalities
        ])
        stop_input_dim = n_stop_numeric + total_stop_cat_dim
        self.stop_encoder = nn.Sequential(
            nn.Linear(stop_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task heads — Phase 2: 7 total (3 Poisson cell-level + 4 binary per-stop)
        # Poisson heads share the same OccurrenceHead shape (backbone + time_features)
        self.occurrence_head = OccurrenceHead(backbone_dim, n_time_features)
        self.speed_occurrence_head = OccurrenceHead(backbone_dim, n_time_features)
        self.trap_head = OccurrenceHead(backbone_dim, n_time_features)
        # Binary heads share the SpeedHead shape (backbone + stop_encoder output)
        self.speed_head = SpeedHead(backbone_dim, 128)
        self.search_head = SpeedHead(backbone_dim, 128)
        self.accident_head = SpeedHead(backbone_dim, 128)
        self.injury_head = SpeedHead(backbone_dim, 128)
        # v3.3.0: disposition head — predicts P(citation) for each stop.
        # violation_type is REMOVED from STOP_CAT_FEATURES inputs (would
        # leak the target).
        self.disposition_head = SpeedHead(backbone_dim, 128)

    def encode_stops(
        self,
        stop_numeric: torch.Tensor,
        stop_categorical: torch.Tensor,
    ) -> torch.Tensor:
        """Encode per-stop features into a fixed-dim vector (shared by all 4 binary heads)."""
        cat_embeds = []
        for i, emb in enumerate(self.stop_cat_embeddings):
            cat_embeds.append(emb(stop_categorical[:, i]))
        cat_concat = torch.cat(cat_embeds, dim=-1)
        stop_input = torch.cat([stop_numeric, cat_concat], dim=-1)
        return self.stop_encoder(stop_input)  # (B, 128)

    def forward(
        self,
        cell_numeric: torch.Tensor,
        cell_categorical: torch.Tensor,
        time_features: torch.Tensor,
        stop_numeric: Optional[torch.Tensor] = None,
        stop_categorical: Optional[torch.Tensor] = None,
        cell_sequence: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        task: str = 'both',
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        task values:
          'occurrence'  — return 3 Poisson heads only (no per-stop features needed)
          'stop_multi'  — return 4 binary heads only (stop_numeric required)
          'both' / 'multi' — return all 7 heads (stop features required)
          (legacy) 'speed' — returns just speed_logit for Phase 1 backward compat
        """
        # Shared backbone: cell features → embedding
        backbone_out = self.backbone(cell_numeric, cell_categorical)

        # Phase 3: spatial refinement (GNN over full cell graph)
        if self.spatial_encoder is not None and edge_index is not None:
            backbone_out = self.spatial_encoder(backbone_out, edge_index)

        # Phase 2: temporal conditioning (LSTM over recent activity)
        if self.temporal_encoder is not None and cell_sequence is not None:
            temporal_out = self.temporal_encoder(cell_sequence)
            backbone_out = self.temporal_fuse(
                torch.cat([backbone_out, temporal_out], dim=-1)
            )

        outputs: Dict[str, torch.Tensor] = {}

        want_occ = task in ('both', 'multi', 'occurrence')
        want_stop = task in ('both', 'multi', 'stop_multi', 'speed') and stop_numeric is not None

        if want_occ:
            outputs['occurrence'] = self.occurrence_head(backbone_out, time_features)
            outputs['speed_occurrence'] = self.speed_occurrence_head(backbone_out, time_features)
            outputs['trap'] = self.trap_head(backbone_out, time_features)

        if want_stop:
            stop_enc = self.encode_stops(stop_numeric, stop_categorical)
            outputs['speed'] = self.speed_head(backbone_out, stop_enc)
            if task != 'speed':  # legacy path returns just speed
                outputs['search'] = self.search_head(backbone_out, stop_enc)
                outputs['accident'] = self.accident_head(backbone_out, stop_enc)
                outputs['injury'] = self.injury_head(backbone_out, stop_enc)
                outputs['disposition'] = self.disposition_head(backbone_out, stop_enc)

        return outputs
