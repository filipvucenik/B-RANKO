from __future__ import annotations

import torch


def build_full_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return a full bidirectional mask with shape (B, 1, T, T)."""

    valid = attention_mask.to(dtype=torch.bool)
    return valid[:, None, :, None] & valid[:, None, None, :]


def build_baa_mask(
    tokens: torch.Tensor,
    merge_indices: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the bidirectional autoregressive attention mask with shape (B, 1, T, T)."""

    batch_size, max_length = tokens.shape
    device = tokens.device

    if lengths is None:
        lengths = torch.full((batch_size,), max_length, device=device, dtype=torch.long)
    else:
        lengths = lengths.to(device=device, dtype=torch.long)

    merge_indices = merge_indices.to(device=device, dtype=torch.long)

    row_index = torch.arange(max_length, device=device).view(1, max_length, 1)
    col_index = torch.arange(max_length, device=device).view(1, 1, max_length)

    sequence_lengths = lengths.view(batch_size, 1, 1)
    merge_points = merge_indices.view(batch_size, 1, 1).clamp(min=0)
    merge_points = torch.minimum(merge_points, sequence_lengths)

    valid_rows = row_index < sequence_lengths
    valid_cols = col_index < sequence_lengths

    query_step = torch.where(
        row_index < merge_points,
        row_index + 1,
        sequence_lengths - row_index,
    )
    query_step = torch.where(valid_rows, query_step, torch.zeros_like(query_step))

    left_visible = (col_index < merge_points) & (col_index < query_step)
    right_visible = (
        (col_index >= merge_points)
        & (col_index >= (sequence_lengths - query_step))
        & valid_cols
    )

    shared_mask = valid_rows & valid_cols & (left_visible | right_visible)
    return shared_mask.unsqueeze(1)
