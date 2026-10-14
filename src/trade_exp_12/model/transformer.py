"""Patch-based time-series Transformer for market state classification."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Project each time step [n_features] -> [d_model]."""

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window, n_features]
        return self.projection(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, : x.size(1)]


class MarketTransformer(nn.Module):
    """Patch-based Transformer encoder with classification head.

    Architecture:
        input [batch, window, n_features]
        -> PatchEmbedding [batch, window, d_model]
        -> + CLS token [batch, window+1, d_model]
        -> + positional encoding
        -> TransformerEncoder
        -> CLS output [batch, d_model]
        -> Linear -> num_classes
    """

    def __init__(
        self,
        n_features: int,
        num_classes: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_embed = PatchEmbedding(n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len + 1)
        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, window, n_features]

        Returns:
            logits: [batch, num_classes]
        """
        batch_size = x.size(0)
        h = self.patch_embed(x)  # [B, W, D]
        cls = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        h = torch.cat([cls, h], dim=1)  # [B, W+1, D]
        h = self.pos_enc(h)
        h = self.input_norm(h)
        h = self.encoder(h)  # [B, W+1, D]
        cls_output = h[:, 0]  # [B, D]
        return self.head(cls_output)  # [B, C]

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS embedding without classification head."""
        batch_size = x.size(0)
        h = self.patch_embed(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_enc(h)
        h = self.input_norm(h)
        h = self.encoder(h)
        return h[:, 0]
