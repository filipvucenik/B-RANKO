from __future__ import annotations

from array import array
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ..masks import build_baa_mask
from ..utils import normalize_merge_index_name as normalize_merge_index_strategy
from .tokenizer import RNATokenizer


def build_fasta_index(fasta_path: str | Path) -> Path:
    """Create a simple .fai index so large FASTA files can be streamed."""

    fasta_path = Path(fasta_path)
    index_path = Path(f"{fasta_path}.fai")

    with fasta_path.open("rb") as fasta_handle, index_path.open("w", encoding="utf-8") as index_handle:
        current_name = None
        current_length = 0
        current_offset = None
        line_bases = None
        line_bytes = None

        while True:
            line = fasta_handle.readline()
            if not line:
                break

            if line.startswith(b">"):
                if current_name is not None:
                    index_handle.write(
                        f"{current_name}\t{current_length}\t{current_offset}\t{line_bases}\t{line_bytes}\n"
                    )

                current_name = line[1:].strip().split()[0].decode("ascii")
                current_length = 0
                current_offset = fasta_handle.tell()
                line_bases = None
                line_bytes = None
                continue

            stripped = line.rstrip(b"\r\n")
            if not stripped:
                continue

            if line_bases is None:
                line_bases = len(stripped)
                line_bytes = len(line)

            current_length += len(stripped)

        if current_name is not None:
            index_handle.write(
                f"{current_name}\t{current_length}\t{current_offset}\t{line_bases}\t{line_bytes}\n"
            )

    return index_path


class FastaSequenceDataset(Dataset):
    """Memory-light FASTA reader backed by a .fai index."""

    def __init__(self, fasta_path: str | Path, tokenizer: RNATokenizer | None = None) -> None:
        self.fasta_path = Path(fasta_path)
        self.index_path = Path(f"{self.fasta_path}.fai")
        if not self.index_path.exists():
            build_fasta_index(self.fasta_path)

        self.tokenizer = tokenizer or RNATokenizer()
        self._fasta_handle = None
        self._index_handle = None
        self._index_offsets = self.read_index_offsets()

    def __len__(self) -> int:
        return len(self._index_offsets)

    def __getitem__(self, index: int) -> dict:
        record_id, seq_len, seq_offset, line_bases, line_bytes = self.read_index_entry(index)
        sequence = self.read_sequence(seq_len, seq_offset, line_bases, line_bytes)
        return {
            "id": record_id,
            "tokenized_sequence": self.tokenizer.tokenize(sequence),
        }

    def padding_index(self) -> int:
        return self.tokenizer.pad_token_id

    def read_index_offsets(self) -> array:
        offsets = array("Q")
        current_offset = 0
        with self.index_path.open("rb") as handle:
            for line in handle:
                offsets.append(current_offset)
                current_offset += len(line)
        return offsets

    def ensure_handles(self) -> None:
        if self._fasta_handle is None:
            self._fasta_handle = self.fasta_path.open("rb")
        if self._index_handle is None:
            self._index_handle = self.index_path.open("rb")

    def read_index_entry(self, index: int) -> tuple[str, int, int, int, int]:
        self.ensure_handles()
        self._index_handle.seek(self._index_offsets[index])
        fields = self._index_handle.readline().decode("utf-8").rstrip("\r\n").split("\t")
        if len(fields) < 5:
            raise ValueError(f"Malformed FASTA index entry at item {index}.")
        return fields[0], int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4])

    def read_sequence(self, seq_len: int, seq_offset: int, line_bases: int, line_bytes: int) -> str:
        self._fasta_handle.seek(seq_offset)

        full_lines, remainder = divmod(seq_len, line_bases)
        bytes_to_read = full_lines * line_bytes + remainder
        raw = self._fasta_handle.read(bytes_to_read)
        sequence = raw.replace(b"\n", b"").replace(b"\r", b"")

        if len(sequence) == seq_len:
            return sequence.decode("ascii").upper()

        # Fallback for files with uneven trailing lines.
        self._fasta_handle.seek(seq_offset)
        remaining = seq_len
        chunks = []
        while remaining > 0:
            line = self._fasta_handle.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            chunks.append(stripped[:remaining])
            remaining -= len(stripped[:remaining])

        sequence = b"".join(chunks)
        if len(sequence) != seq_len:
            raise ValueError(f"Could not reconstruct sequence of length {seq_len}.")
        return sequence.decode("ascii").upper()

    def __del__(self) -> None:
        if self._fasta_handle is not None:
            self._fasta_handle.close()
        if self._index_handle is not None:
            self._index_handle.close()

class BidirectionalBatchCollator:
    def __init__(self, pad_token_id: int, merge_index_strategy: str = "uniform") -> None:
        self.pad_token_id = pad_token_id
        self.merge_index_strategy = normalize_merge_index_strategy(merge_index_strategy, default="uniform")

    def __call__(self, batch: list[dict]) -> dict:
        batch_ids = [item["id"] for item in batch]
        raw_sequences = [item["tokenized_sequence"] for item in batch]

        lengths = torch.tensor([len(sequence) for sequence in raw_sequences], dtype=torch.long)
        merge_indices = torch.tensor(
            [self.sample_merge_index(int(length)) for length in lengths],
            dtype=torch.long,
        )

        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            raw_sequences,
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        shared_attention_mask = build_baa_mask(
            padded_sequences,
            merge_indices=merge_indices.to(padded_sequences.device),
            lengths=lengths.to(padded_sequences.device),
        )

        return {
            "ids": batch_ids,
            "sequences": padded_sequences,
            "merge_index": merge_indices,
            "shared_attention_mask": shared_attention_mask,
        }

    def sample_merge_index(self, token_length: int) -> int:
        if token_length <= 2:
            return 1

        left_edge = 1
        right_edge = token_length - 1

        if self.merge_index_strategy == "half":
            return token_length // 2
        if self.merge_index_strategy == "uniform":
            return int(torch.randint(left_edge, right_edge + 1, (1,)).item())

        raise ValueError(f"Unsupported merge-index strategy '{self.merge_index_strategy}'.")


class SequenceDataModule(pl.LightningDataModule):
    """Shared FASTA datamodule used for both pre-training and fine-tuning."""

    def __init__(
        self,
        train_path: str,
        val_path: str,
        batch_size: int,
        num_workers: int,
        tokenizer: RNATokenizer | None = None,
        train_merge_index: str = "uniform",
        val_merge_index: str = "half",
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.tokenizer = tokenizer or RNATokenizer()
        self.train_merge_index = normalize_merge_index_strategy(train_merge_index, default="uniform")
        self.val_merge_index = normalize_merge_index_strategy(val_merge_index, default="half")
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = FastaSequenceDataset(self.train_path, tokenizer=self.tokenizer)
            self.val_dataset = FastaSequenceDataset(self.val_path, tokenizer=self.tokenizer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=BidirectionalBatchCollator(
                pad_token_id=self.train_dataset.padding_index(),
                merge_index_strategy=self.train_merge_index,
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=BidirectionalBatchCollator(
                pad_token_id=self.val_dataset.padding_index(),
                merge_index_strategy=self.val_merge_index,
            ),
        )
