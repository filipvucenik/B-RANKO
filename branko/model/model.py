from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ..data.tokenizer import RNATokenizer, decode_tokens
from .covariance_pooling import CovariancePooling
from .modules import BidirectionalTransformer, NextTokenModelHead, PositionalEncoding
from ..masks import build_baa_mask, build_full_attention_mask
from ..utils import load_model_bundle, load_model_config, load_model_state


@dataclass
class EncoderOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to_tuple(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.last_hidden_state, self.pooler_output


class BidirectionalRNAModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config["model"]

        self.tokenizer = RNATokenizer()
        self.pad_token_id = self.tokenizer.pad_token_id
        self.sos_token_id = self.tokenizer.sos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.vocab_size = self.tokenizer.vocab_size

        embedding_dim = int(model_config["embedding_dim"])
        num_layers = int(model_config["num_layers"])
        num_heads = int(model_config["num_heads"])
        dropout = float(model_config.get("dropout", 0.0))

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=self.pad_token_id,
        )
        self.pos_encoding = PositionalEncoding(d_model=embedding_dim, dropout=dropout)
        self.shared_blocks = BidirectionalTransformer(
            embed_dim=embedding_dim,
            num_blocks=num_layers,
            num_heads=num_heads,
            use_rot_emb=bool(model_config.get("use_rotary", True)),
            attention_dropout=dropout,
            transition_dropout=dropout,
            residual_dropout=dropout,
        )
        self.left_head = NextTokenModelHead(embed_dim=embedding_dim, vocab_size=self.vocab_size)
        self.right_head = NextTokenModelHead(embed_dim=embedding_dim, vocab_size=self.vocab_size)
        self.covariance_pooler = self.build_covariance_pooler(config, embedding_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        shared_attention_mask: torch.Tensor,
        left_query_index: int | None = None,
        right_query_index: int | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.encode_tokens(tokens, shared_attention_mask)

        left_states = hidden_states
        right_states = hidden_states
        if left_query_index is not None:
            left_states = hidden_states[:, left_query_index : left_query_index + 1]
        if right_query_index is not None:
            right_states = hidden_states[:, right_query_index : right_query_index + 1]

        return {
            "left": self.left_head(left_states),
            "right": self.right_head(right_states),
        }

    def encode_tokens(self, tokens: torch.Tensor, shared_attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding(tokens)
        hidden_states = self.pos_encoding(hidden_states)
        return self.shared_blocks(hidden_states, bidirectional_mask=shared_attention_mask)

    def encode(
        self,
        sequences: str | list[str] | torch.Tensor,
        pooling: str = "mean",
        mask_mode: str = "full",
        merge_index: int | list[int] | torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> EncoderOutput | tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        tokens = self.as_token_tensor(sequences, device=device)
        attention_mask = tokens.ne(self.pad_token_id)

        if mask_mode == "full":
            shared_attention_mask = build_full_attention_mask(attention_mask)
        elif mask_mode == "inward":
            lengths = attention_mask.sum(dim=-1)
            merge_indices = self.resolve_merge_indices(lengths, merge_index, device=device)
            shared_attention_mask = build_baa_mask(tokens, merge_indices, lengths=lengths)
        else:
            raise ValueError("mask_mode must be either 'full' or 'inward'.")

        hidden_states = self.encode_tokens(tokens, shared_attention_mask)
        pooled = self.pool_hidden_states(hidden_states, tokens, pooling=pooling)

        output = EncoderOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            input_ids=tokens,
            attention_mask=attention_mask.long(),
        )
        return output if return_dict else output.to_tuple()

    def generate(
        self,
        sequence_length: int,
        merge_index: int | None = None,
        strategy: str = "random",
        temperature: float = 1.0,
    ) -> str:
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1.")

        token_length = sequence_length + 2
        token_merge_index = token_length // 2 if merge_index is None else int(merge_index) + 1
        if not 0 < token_merge_index < token_length:
            raise ValueError(f"merge_index must be between 0 and {sequence_length}, got {merge_index}.")

        device = next(self.parameters()).device
        tokens = torch.full((1, token_length), self.pad_token_id, dtype=torch.long, device=device)
        tokens[0, 0] = self.sos_token_id
        tokens[0, -1] = self.eos_token_id

        lengths = torch.tensor([token_length], device=device)
        merge_indices = torch.tensor([token_merge_index], device=device)
        shared_attention_mask = build_baa_mask(tokens, merge_indices, lengths=lengths)

        left_query_index = 0
        right_query_index = token_length - 1

        with torch.no_grad():
            while left_query_index + 1 < right_query_index:
                logits = self(
                    tokens=tokens,
                    shared_attention_mask=shared_attention_mask,
                    left_query_index=left_query_index,
                    right_query_index=right_query_index,
                )
                left_logits = self.mask_generation_logits(logits["left"][0, 0])
                right_logits = self.mask_generation_logits(logits["right"][0, 0])

                next_left = left_query_index + 1
                next_right = right_query_index - 1

                if next_left < next_right:
                    tokens[0, next_left] = self.select_token(left_logits, strategy, temperature)
                    tokens[0, next_right] = self.select_token(right_logits, strategy, temperature)
                    left_query_index = next_left
                    right_query_index = next_right
                    continue

                if next_left == next_right:
                    center_logits = 0.5 * (left_logits + right_logits)
                    tokens[0, next_left] = self.select_token(center_logits, strategy, temperature)
                break

        return decode_tokens(self.tokenizer, tokens.squeeze(0))

    def as_token_tensor(self, sequences: str | list[str] | torch.Tensor, device: torch.device) -> torch.Tensor:
        if isinstance(sequences, torch.Tensor):
            tokens = sequences.to(device=device, dtype=torch.long)
            if tokens.ndim == 1:
                tokens = tokens.unsqueeze(0)
            if tokens.ndim != 2:
                raise ValueError("Token tensor must have shape (T,) or (B, T).")
            return tokens

        if isinstance(sequences, str):
            sequences = [sequences]

        if not isinstance(sequences, (list, tuple)) or not sequences:
            raise ValueError("Expected a sequence string, a list of strings, or a token tensor.")

        tokenized = [self.tokenizer.tokenize(sequence).to(device=device) for sequence in sequences]
        return torch.nn.utils.rnn.pad_sequence(
            tokenized,
            batch_first=True,
            padding_value=self.pad_token_id,
        )

    def resolve_merge_indices(
        self,
        lengths: torch.Tensor,
        merge_index: int | list[int] | torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if merge_index is None:
            return lengths // 2
        if isinstance(merge_index, torch.Tensor):
            merge_indices = merge_index.to(device=device, dtype=torch.long)
        elif isinstance(merge_index, (list, tuple)):
            merge_indices = torch.tensor(merge_index, device=device, dtype=torch.long)
        else:
            merge_indices = torch.full_like(lengths, int(merge_index))
        if merge_indices.ndim == 0:
            merge_indices = merge_indices.expand_as(lengths)
        return merge_indices

    def pool_hidden_states(self, hidden_states: torch.Tensor, tokens: torch.Tensor, pooling: str) -> torch.Tensor:
        if pooling == "first":
            return hidden_states[:, 0]
        if pooling == "covariance":
            if self.covariance_pooler is None:
                raise ValueError("pooling='covariance' requires a model bundle with a covariance-pooling head.")
            return self.covariance_pooler(
                hidden_states,
                attention_mask=tokens.ne(self.pad_token_id),
            )
        if pooling != "mean":
            raise ValueError("pooling must be one of: mean, first, covariance.")

        valid = tokens.ne(self.pad_token_id) & tokens.ne(self.sos_token_id) & tokens.ne(self.eos_token_id)
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1)
        return (hidden_states * valid.unsqueeze(-1)).sum(dim=1) / denominator

    def mask_generation_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        logits[self.pad_token_id] = float("-inf")
        logits[self.sos_token_id] = float("-inf")
        logits[self.eos_token_id] = float("-inf")
        logits[self.unk_token_id] = float("-inf")
        return logits

    def select_token(self, logits: torch.Tensor, strategy: str, temperature: float) -> torch.Tensor:
        if strategy == "greedy":
            return torch.argmax(logits, dim=-1)
        if strategy == "random":
            probs = torch.softmax(logits / float(temperature), dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(0)
        raise ValueError("strategy must be either 'greedy' or 'random'.")

    @staticmethod
    def build_covariance_pooler(config: dict, embedding_dim: int) -> CovariancePooling | None:
        inference_config = config.get("inference", {})
        covariance_config = inference_config.get("covariance_pooling", {})
        if not covariance_config or not covariance_config.get("enabled", False):
            return None
        return CovariancePooling(
            input_dim=embedding_dim,
            compressed_dim=int(covariance_config["compressed_dim"]),
            output_dim=int(covariance_config.get("output_dim", embedding_dim)),
            exclude_special_tokens=bool(covariance_config.get("exclude_special_tokens", True)),
            tied_projections=bool(covariance_config.get("tied_projections", False)),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        device: str | torch.device | None = None,
        strict: bool = True,
    ) -> "BidirectionalRNAModel":
        config, state_dict, _ = load_model_bundle(model_path, map_location="cpu")
        model = cls(config)
        model.load_state_dict(state_dict, strict=strict)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        strict: bool = True,
        config_path: str | Path | None = None,
    ) -> "BidirectionalRNAModel":
        config = load_model_config(checkpoint_path, config_path=config_path, map_location="cpu")
        state_dict = load_model_state(checkpoint_path, map_location="cpu")
        model = cls(config)
        model.load_state_dict(state_dict, strict=strict)
        if device is not None:
            model = model.to(device)
        model.eval()
        return model
