from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (query * cosine) + (rotate_half(query) * sine), (key * cosine) + (rotate_half(key) * sine)


class RotaryPositionEmbedding(nn.Module):
    """Rotary embedding module kept checkpoint-compatible with the original code."""

    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def update_cached(self, x: torch.Tensor, seq_dim: int) -> None:
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            time = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            frequencies = torch.einsum("i,j->ij", time, self.inv_freq)
            embedding = torch.cat((frequencies, frequencies), dim=-1).to(x.device)
            self.cos_cached = embedding.cos()[None, None, :, :]
            self.sin_cached = embedding.sin()[None, None, :, :]

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.update_cached(key, seq_dim=-2)
        return apply_rotary_pos_emb(query, key, self.cos_cached, self.sin_cached)
