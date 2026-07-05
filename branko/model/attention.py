from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .rope import RotaryPositionEmbedding


class BidirectionalMultiHeadAttention(nn.Module):
    def __init__(
        self,
        c_in: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        use_rot_emb: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if c_in % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by num_heads.")

        self.c_in = c_in
        self.num_heads = num_heads
        self.c_head = c_in // num_heads
        self.c_qkv = self.c_head * num_heads

        self.use_rot_emb = use_rot_emb
        if self.use_rot_emb:
            self.rotary_emb = RotaryPositionEmbedding(self.c_head)

        self.to_qkv = nn.Linear(self.c_in, self.c_qkv * 3, bias=bias)
        self.attention_dropout = nn.Dropout(p=attention_dropout)
        self.out_proj = nn.Linear(c_in, c_in, bias=bias)

    def forward(self, x: torch.Tensor, bidirectional_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        query, key, value = self.to_qkv(x).chunk(3, dim=-1)

        query = query.view(batch_size, seq_len, self.num_heads, self.c_head).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.c_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.c_head).transpose(1, 2)

        if self.use_rot_emb:
            query, key = self.rotary_emb(query, key)

        bidirectional_mask = bidirectional_mask.to(dtype=torch.bool, device=x.device)

        # All-zero rows appear outside the currently visible inward region.
        valid_query_rows = bidirectional_mask.any(dim=-1)
        if not torch.all(valid_query_rows):
            diag = torch.eye(seq_len, dtype=torch.bool, device=x.device).view(1, 1, seq_len, seq_len)
            bidirectional_mask = bidirectional_mask | (~valid_query_rows.unsqueeze(-1) & diag)

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bidirectional_mask,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.c_in)
        out = self.out_proj(out)
        return out * valid_query_rows.squeeze(1).unsqueeze(-1).to(out.dtype)


class BidirectionalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        c_in: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        use_rot_emb: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.mh_attn = BidirectionalMultiHeadAttention(
            c_in=c_in,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            use_rot_emb=use_rot_emb,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, bidirectional_mask: torch.Tensor) -> torch.Tensor:
        return self.mh_attn(x, bidirectional_mask)
