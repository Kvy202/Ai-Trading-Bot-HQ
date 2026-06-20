"""
Advanced neural network architectures for time–series forecasting and classification.

This module defines more expressive models than the baseline TCN/LSTM/Transformer
implementations found in ``ml_dl/dl_models.py``.  The goal is to provide a
drop‑in upgrade path that captures richer temporal dependencies in financial
features without materially increasing inference cost.  The main model defined
here uses a multi‑layer Transformer encoder coupled with a positional encoding
module.  It outputs three heads (regression, volatility and classification)
similar to the baseline models.  You can instantiate this model by passing
``kind=adv`` to the training script, or import it directly.

Note: The default hyper‑parameters are conservative to avoid overfitting on
small datasets.  Feel free to increase ``d_model`` or ``num_layers`` when
training on larger corpora.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class SinusoidalPositionalEncoding(nn.Module):
    """Injects positional information using sin/cos functions.

    This implementation follows the formulation in the original Transformer
    paper (Vaswani et al., 2017).  The module precomputes a matrix of shape
    ``[max_len, d_model]`` and adds it to the input sequence.  If
    ``learned=False`` the positional encodings are fixed; when ``learned=True``
    the matrix becomes a trainable parameter.  We expose both options to allow
    experimentation.

    Args:
        d_model: embedding dimension.
        max_len: maximum sequence length expected during training/inference.
        learned: if ``True``, use a trainable position embedding instead of
            fixed sinusoidal values.
    """

    def __init__(self, d_model: int, max_len: int = 512, learned: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        if learned:
            self.pe = nn.Embedding(max_len, d_model)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * -(math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            self.register_buffer("pe", pe)
        self.learned = learned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to the input tensor.

        Args:
            x: input of shape ``[B, T, d_model]``.
        Returns:
            Tensor of the same shape as ``x`` with positional encodings added.
        """
        if self.learned:
            positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
            pos_embed = self.pe(positions)[None, :, :]
            return x + pos_embed
        else:
            return x + self.pe[:, : x.size(1), :]


class AdvancedTransformer(nn.Module):
    """A transformer‑based model for time‑series prediction.

    The network embeds the input features to a latent dimension, injects
    positional encodings, processes the sequence with a stack of Transformer
    encoder layers, and finally uses separate fully connected heads to predict
    next‑horizon returns, realized volatility and binary direction.  It is
    inspired by Temporal Fusion Transformers but simplified for efficient
    training and inference.

    Args:
        in_dim: Number of input features per timestep.
        d_model: Dimension of the latent representation (default 128).
        nhead: Number of attention heads (default 8).
        num_layers: Number of Transformer encoder layers (default 3).
        dropout: Dropout probability in the encoder layers (default 0.1).
        max_seq_len: Maximum expected sequence length (default 512).
        learned_pos: If ``True``, use learned positional encodings instead of
            fixed sinusoidal encodings.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        learned_pos: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        # Project inputs to the model dimension
        self.proj = nn.Linear(in_dim, d_model)
        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, learned_pos)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Heads for outputs
        self.head_ret_reg = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        self.head_ret_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2)
        )
        self.head_rv_reg = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for the model.

        Args:
            x: Input tensor of shape ``[B, T, F]`` (batch, time, features).
        Returns:
            Dictionary with keys ``ret_reg`` (regression prediction for next
            return), ``ret_cls_logits`` (logits for binary classification of
            next move), and ``rv_reg`` (regression prediction for realized
            volatility).  Each tensor has shape ``[B]`` except the logits
            tensor which has shape ``[B, 2]``.
        """
        # Project and add positional encodings
        z = self.proj(x)  # [B, T, d_model]
        z = self.pos_enc(z)
        # Encode sequence; take the last timestep's representation
        h = self.encoder(z)[:, -1, :]  # [B, d_model]
        return {
            "ret_reg": self.head_ret_reg(h).squeeze(-1),
            "ret_cls_logits": self.head_ret_cls(h),
            "rv_reg": self.head_rv_reg(h).squeeze(-1),
        }


__all__ = ["AdvancedTransformer"]