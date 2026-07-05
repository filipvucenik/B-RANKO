#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from branko import BidirectionalRNAModel


def load_lengths(path: str | None, sequence_length: int | None, num_sequences: int) -> np.ndarray:
    if sequence_length is not None:
        return np.full(num_sequences, int(sequence_length), dtype=int)
    if path is None:
        raise ValueError("Pass either --sequence-length or --length-distribution.")

    dist_path = Path(path)
    delimiter = "\t" if dist_path.suffix.lower() in {".tsv", ".txt"} else ","
    with dist_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None or "Length" not in reader.fieldnames:
            raise ValueError("Length-distribution file must contain a 'Length' column.")
        lengths = [int(row["Length"]) for row in reader if row.get("Length")]
    if not lengths:
        raise ValueError("Length-distribution file did not contain any lengths.")
    return np.random.choice(np.array(lengths, dtype=int), size=num_sequences, replace=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RNA sequences with B-RANKO.")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a bundled B-RANKO model file or a raw Lightning checkpoint.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config for raw Lightning checkpoints. Defaults to config.yaml next to the checkpoint.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated FASTA output.")
    parser.add_argument("--num-sequences", type=int, required=True, help="Number of sequences to generate.")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Fixed RNA sequence length, excluding special tokens.",
    )
    parser.add_argument(
        "--length-distribution",
        default=None,
        help="CSV or TSV file with a 'Length' column.",
    )
    parser.add_argument(
        "--strategy",
        choices=("random", "greedy"),
        default="random",
        help="Token selection strategy.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for random generation.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BidirectionalRNAModel.from_checkpoint(args.model, device=device, config_path=args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lengths = load_lengths(args.length_distribution, args.sequence_length, args.num_sequences)

    output_path = output_dir / "generated_sequences.fasta"
    with output_path.open("w", encoding="utf-8") as handle:
        for index, sequence_length in tqdm(
            enumerate(lengths),
            total=args.num_sequences,
            desc="Generating sequences",
        ):
            sequence = model.generate(
                sequence_length=int(sequence_length),
                strategy=args.strategy,
                temperature=args.temperature,
            )
            sequence_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
            handle.write(f">seq_{index}_{sequence_hash}\n{sequence}\n")

    print(f"Wrote {args.num_sequences} sequences to {output_path}")


if __name__ == "__main__":
    main()
