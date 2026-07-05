from __future__ import annotations

import torch


class RNATokenizer:

    def __init__(self) -> None:
        self.vocab = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
            "A": 4,
            "C": 5,
            "G": 6,
            "U": 7,
            "T": 7,
        }
        self.inv_vocab = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>",
            4: "A",
            5: "C",
            6: "G",
            7: "U",
        }

    @property
    def pad_token_id(self) -> int:
        return self.vocab["<PAD>"]

    @property
    def sos_token_id(self) -> int:
        return self.vocab["<SOS>"]

    @property
    def eos_token_id(self) -> int:
        return self.vocab["<EOS>"]

    @property
    def unk_token_id(self) -> int:
        return self.vocab["<UNK>"]

    @property
    def vocab_size(self) -> int:
        return 8

    def tokenize(self, sequence: str) -> torch.Tensor:
        sequence = sequence.strip().upper().replace("T", "U")
        token_ids = [self.sos_token_id]
        token_ids.extend(self.vocab.get(base, self.unk_token_id) for base in sequence)
        token_ids.append(self.eos_token_id)
        return torch.tensor(token_ids, dtype=torch.long)

    def detokenize(self, token_ids: torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return "".join(self.inv_vocab.get(int(token_id), "N") for token_id in token_ids)


def decode_tokens(tokenizer: RNATokenizer, token_ids: torch.Tensor) -> str:
    if token_ids.ndim != 1:
        raise ValueError("decode_tokens expects a 1D tensor.")

    token_ids = token_ids.detach().cpu()
    if len(token_ids) > 0 and token_ids[0].item() == tokenizer.sos_token_id:
        token_ids = token_ids[1:]
    if len(token_ids) > 0 and token_ids[-1].item() == tokenizer.eos_token_id:
        token_ids = token_ids[:-1]
    return tokenizer.detokenize(token_ids)
