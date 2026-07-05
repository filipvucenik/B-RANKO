from __future__ import annotations

import math

import torch
from torch import nn

from .attention import BidirectionalMultiHeadSelfAttention


class SwiGLU(nn.Module):
    def __init__(self, size_in: int, size_out: int, beta_is_learnable: bool = True, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(size_in, size_out, bias=bias)
        self.linear_gate = nn.Linear(size_in, size_out, bias=bias)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=beta_is_learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = self.beta.to(x.dtype)
        linear_out = self.linear(x)
        swish_out = linear_out * torch.sigmoid(beta * linear_out)
        return swish_out * self.linear_gate(x)


class TransformerBlockBidirectionalDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_rot_emb: bool = True,
        attn_qkv_bias: bool = False,
        transition_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        transition_factor: int = 4,
    ) -> None:
        super().__init__()
        self.mh_attn = BidirectionalMultiHeadSelfAttention(
            c_in=embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            use_rot_emb=use_rot_emb,
            bias=attn_qkv_bias,
        )
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.transition = nn.Sequential(
            SwiGLU(embed_dim, int(2 / 3 * transition_factor * embed_dim), beta_is_learnable=True, bias=True),
            nn.Dropout(p=transition_dropout),
            nn.Linear(int(2 / 3 * transition_factor * embed_dim), embed_dim, bias=True),
        )
        self.out_layer_norm = nn.LayerNorm(embed_dim)
        self.residual_dropout_1 = nn.Dropout(p=residual_dropout)
        self.residual_dropout_2 = nn.Dropout(p=residual_dropout)

    def forward(self, x: torch.Tensor, bidirectional_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.attn_layer_norm(x)
        x = residual + self.residual_dropout_1(self.mh_attn(x, bidirectional_mask=bidirectional_mask))

        residual = x
        x = self.out_layer_norm(x)
        x = residual + self.residual_dropout_2(self.transition(x))
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        num_heads: int,
        use_rot_emb: bool = True,
        attn_qkv_bias: bool = False,
        transition_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlockBidirectionalDecoder(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    use_rot_emb=use_rot_emb,
                    attn_qkv_bias=attn_qkv_bias,
                    transition_dropout=transition_dropout,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, bidirectional_mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, bidirectional_mask=bidirectional_mask)
        return self.final_layer_norm(x)


class NextTokenModelHead(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
