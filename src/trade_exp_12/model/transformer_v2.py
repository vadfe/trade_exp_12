"""Masked Autoencoder Transformer for market state representation learning.

v2 architecture:
  Pretraining: mask random timesteps, reconstruct features (MAE)
  Finetuning:  freeze encoder, train classification head on reversal labels

The encoder learns universal market-state embeddings that transfer
across instruments because relative features (log_return, body_atr, etc.)
are already instrument-agnostic.
"""

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
        return self.projection(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MarketMAE(nn.Module):
    """Masked Autoencoder for market time series.

    Pretraining mode:
        - Randomly mask `mask_ratio` of timesteps
        - Encode only visible timesteps (efficient)
        - Decode with lightweight decoder to reconstruct masked features
        - Loss = MSE on masked positions only

    Finetuning mode:
        - No masking
        - Encode full window
        - CLS token -> classification head
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        mask_ratio: float = 0.4,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.mask_ratio = mask_ratio

        # Encoder
        self.patch_embed = PatchEmbedding(n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len + 1)
        self.encoder_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_encoder_layers,
        )

        # Decoder (lightweight, for pretraining only)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=n_decoder_layers,
        )
        self.decoder_proj = nn.Linear(d_model, n_features)

        # Classification head (for finetuning)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _random_mask(self, batch_size: int, seq_len: int, device: torch.device):
        """Generate random mask: True = masked (to reconstruct)."""
        n_mask = max(1, int(seq_len * self.mask_ratio))
        noise = torch.rand(batch_size, seq_len, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        mask_indices = ids_shuffle[:, :n_mask]
        mask.scatter_(1, mask_indices, True)
        return mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode full window (no masking). Returns CLS embedding [B, D]."""
        B = x.size(0)
        h = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_enc(h)
        h = self.encoder_norm(h)
        h = self.encoder(h)
        return h[:, 0]  # CLS token

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Encode full window. Returns all token embeddings [B, W+1, D]."""
        B = x.size(0)
        h = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_enc(h)
        h = self.encoder_norm(h)
        return self.encoder(h)

    def pretrain_forward(self, x: torch.Tensor):
        """MAE forward: mask, encode visible, decode, reconstruct masked.

        Returns (reconstruction, target, mask) where:
            reconstruction: [B, W, n_features] — decoder output
            target: [B, W, n_features] — original features
            mask: [B, W] bool — True at masked positions
        """
        B, W, F = x.shape
        device = x.device

        mask = self._random_mask(B, W, device)  # [B, W]

        # Embed all positions
        h = self.patch_embed(x)  # [B, W, D]

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)

        # For encoder: replace masked positions with nothing (just encode visible)
        # But for simplicity and stability, encode all with positional encoding,
        # then pass to decoder which sees mask tokens at masked positions
        h_full = torch.cat([cls, h], dim=1)  # [B, W+1, D]
        h_full = self.pos_enc(h_full)
        h_full = self.encoder_norm(h_full)
        encoded = self.encoder(h_full)  # [B, W+1, D]

        # Decoder input: encoded visible + mask tokens at masked positions
        decoder_input = encoded[:, 1:]  # skip CLS, [B, W, D]
        mask_tokens = self.mask_token.expand(B, W, -1)
        # Replace masked positions with mask_token
        mask_expanded = mask.unsqueeze(-1).expand_as(decoder_input)
        decoder_input = torch.where(mask_expanded, mask_tokens, decoder_input)

        # Add positional encoding to decoder input
        decoder_input = self.pos_enc(
            torch.cat([encoded[:, :1], decoder_input], dim=1)
        )[:, 1:]  # skip CLS position

        decoded = self.decoder(decoder_input)  # [B, W, D]
        reconstruction = self.decoder_proj(decoded)  # [B, W, F]

        return reconstruction, x, mask

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Classification forward: encode -> CLS -> head. Returns logits."""
        cls_emb = self.encode(x)
        return self.head(cls_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward = classify (for compatibility with v1 predict)."""
        return self.classify(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS embedding (for compatibility with v1)."""
        return self.encode(x)
