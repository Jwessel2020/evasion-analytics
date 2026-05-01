"""Phase 3A: PLR (Periodic-Linear) numeric tokenizer.

Per Gorishniy et al. 2022 "On Embeddings for Numerical Features in Tabular
Deep Learning" (NumEmbeddings paper). Replaces the per-feature linear
projection in `deep.model.NumericTokenizer` with a richer basis that
includes learned-frequency periodic components.

Design (per docs/next-gen/planning/05_model_architecture.md Phase 3A):

    PLR(x) = Linear_proj( concat[ x, sin(2πWx), cos(2πWx) ] )

where:
    x: (B, n_features)            — z-scored numeric features
    W: (n_features, n_freqs)      — learned per-feature frequency matrix
    Wx: (B, n_features, n_freqs)  — feature×freq products (broadcast)
    concat: (B, n_features, 1 + 2*n_freqs)
    Linear_proj: (1 + 2*n_freqs) → d_token, applied per-feature
    output: (B, n_features, d_token) — one token per feature, like NumericTokenizer

Key hyperparameter: `n_periodic_freqs = 8` (per planning/06_hyperparameters.md
Phase 3 hyperparameter additions table).

Initialization (Gorishniy 2022 §3.1): W ~ N(0, σ²) with σ = 0.5 (paper
calls this `sigma_w` and defaults it to 1.0 but our z-scored inputs have
std ≈ 1 so 0.5 keeps initial periodic frequencies in a reasonable range).

This is a DROP-IN replacement for `deep.model.NumericTokenizer`. Same
output shape (B, n_features, d_token); same constructor args plus the
two PLR-specific kwargs.

References:
- Gorishniy et al. 2022, arXiv:2203.05556
- next_gen/tokenizers/README.md (design overview)
- docs/next-gen/papers/tabular-transformers/README.md (Tier 1 paper)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PLRNumericTokenizer(nn.Module):
    """Periodic-Linear numeric tokenizer (Gorishniy et al. 2022).

    Per-feature output is concat([x, sin(2πWx), cos(2πWx)]) → Linear → d_token.
    Each feature gets its OWN learned frequency matrix W and projection bias,
    so the model can learn different periodicities for hour, day-of-week,
    lat/lng, etc.

    Args:
        n_features: Number of numeric input features.
        d_token: Output dimension per feature token.
        n_periodic_freqs: Number of learned frequencies per feature
            (default 8 per planning/06_hyperparameters.md).
        sigma_w: Std-dev for W initialization (default 0.5, tuned for
            z-scored inputs with std ≈ 1).

    Shape:
        Input:  (B, n_features) float32
        Output: (B, n_features, d_token) float32
    """

    def __init__(
        self,
        n_features: int,
        d_token: int,
        n_periodic_freqs: int = 8,
        sigma_w: float = 0.5,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.n_periodic_freqs = n_periodic_freqs

        # Learned per-feature frequency matrix.
        # Shape: (n_features, n_periodic_freqs)
        self.W = nn.Parameter(torch.randn(n_features, n_periodic_freqs) * sigma_w)

        # Per-feature projection from (1 + 2*n_freqs) → d_token.
        # We use a per-feature linear (n_features × d_token weights, plus
        # bias). Implemented as a single weight matrix + manual matmul to
        # avoid n_features individual nn.Linear modules.
        in_dim = 1 + 2 * n_periodic_freqs
        # weight shape: (n_features, in_dim, d_token)
        # bias shape:   (n_features, d_token)
        self.proj_weight = nn.Parameter(
            torch.empty(n_features, in_dim, d_token)
        )
        self.proj_bias = nn.Parameter(torch.zeros(n_features, d_token))
        # Kaiming-style init for the projection (matches NumericTokenizer
        # which uses Kaiming uniform per deep/model.py docs).
        bound = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(self.proj_weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_features) z-scored numeric features.

        Returns:
            tokens: (B, n_features, d_token) per-feature embeddings.
        """
        B, F = x.shape
        if F != self.n_features:
            raise ValueError(
                f"PLRNumericTokenizer expected {self.n_features} features, "
                f"got {F}"
            )

        # Compute periodic features: (B, F, n_freqs)
        # x[:, :, None] has shape (B, F, 1)
        # self.W[None, :, :] has shape (1, F, n_freqs)
        # broadcast multiply → (B, F, n_freqs)
        scaled = x.unsqueeze(-1) * self.W.unsqueeze(0)
        sin_part = torch.sin(2 * math.pi * scaled)  # (B, F, n_freqs)
        cos_part = torch.cos(2 * math.pi * scaled)  # (B, F, n_freqs)

        # Concat raw + sin + cos: (B, F, 1 + 2*n_freqs)
        concat = torch.cat(
            [x.unsqueeze(-1), sin_part, cos_part],
            dim=-1,
        )

        # Per-feature projection: einsum over the in_dim axis.
        # concat: (B, F, in_dim)
        # proj_weight: (F, in_dim, d_token)
        # → tokens: (B, F, d_token)
        tokens = torch.einsum("bfi,fid->bfd", concat, self.proj_weight)
        tokens = tokens + self.proj_bias.unsqueeze(0)  # (B, F, d_token)

        return tokens

    def extra_repr(self) -> str:
        return (
            f"n_features={self.n_features}, d_token={self.d_token}, "
            f"n_periodic_freqs={self.n_periodic_freqs}"
        )
