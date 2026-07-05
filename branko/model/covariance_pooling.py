from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def strip_special_tokens(attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(dtype=torch.bool)
    if mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape (B, T), got {tuple(mask.shape)}.")
    if mask.shape[1] == 0:
        return mask

    trimmed = mask.clone()
    batch_positions = torch.arange(mask.shape[0], device=mask.device)
    token_positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    has_tokens = mask.any(dim=1)
    if has_tokens.any():
        first_positions = mask.long().argmax(dim=1)
        last_positions = (mask.long() * token_positions).max(dim=1).values
        trimmed[batch_positions[has_tokens], first_positions[has_tokens]] = False
        trimmed[batch_positions[has_tokens], last_positions[has_tokens]] = False
    return trimmed


class CovariancePooling(nn.Module):
    def __init__(
        self,
        input_dim: int,
        compressed_dim: int,
        output_dim: Optional[int] = None,
        *,
        exclude_special_tokens: bool = True,
        tied_projections: bool = False,
    ) -> None:
        super().__init__()
        if compressed_dim <= 0:
            raise ValueError("compressed_dim must be positive.")

        self.input_dim = int(input_dim)
        self.compressed_dim = int(compressed_dim)
        self.exclude_special_tokens = bool(exclude_special_tokens)
        self.left_projection = nn.Linear(self.input_dim, self.compressed_dim, bias=False)
        if tied_projections:
            self.right_projection = self.left_projection
        else:
            self.right_projection = nn.Linear(self.input_dim, self.compressed_dim, bias=False)

        flat_dim = self.compressed_dim * self.compressed_dim
        self.output_dim = flat_dim if output_dim is None else int(output_dim)
        self.output_projection = None
        if self.output_dim != flat_dim:
            self.output_projection = nn.Linear(flat_dim, self.output_dim, bias=False)

    def effective_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        else:
            mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
            if mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    f"attention_mask must have shape {tuple(hidden_states.shape[:2])}, "
                    f"got {tuple(mask.shape)}."
                )
        if self.exclude_special_tokens:
            mask = strip_special_tokens(mask)
        return mask

    @staticmethod
    def masked_second_moment(
        left_states: torch.Tensor,
        right_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(dtype=left_states.dtype, device=left_states.device)
        left_states = left_states * weights
        right_states = right_states * weights
        counts = mask.sum(dim=1).clamp_min(1).to(dtype=left_states.dtype, device=left_states.device)
        return torch.einsum("bti,btj->bij", left_states, right_states) / counts[:, None, None]

    def covariance_matrix(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = self.effective_mask(hidden_states, attention_mask)
        left_states = self.left_projection(hidden_states)
        right_states = self.right_projection(hidden_states)
        return self.masked_second_moment(left_states, right_states, mask)

    def pooled_embedding(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        covariance = self.covariance_matrix(hidden_states, attention_mask=attention_mask)
        embedding = covariance.flatten(start_dim=1)
        if self.output_projection is not None:
            embedding = self.output_projection(embedding)
        return embedding

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.pooled_embedding(hidden_states, attention_mask=attention_mask)
