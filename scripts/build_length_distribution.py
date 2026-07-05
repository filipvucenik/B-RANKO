#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


COMMON_SEQUENCE_COLUMNS = ("seq", "sequence", "Sequence", "SEQ", "rna", "dna")


def normalize_sequence(sequence: str) -> str | None:
    sequence = (sequence or "").strip()
    return sequence or None


def detect_sequence_column(fieldnames: list[str], explicit_column: str | None) -> str:
    if explicit_column is not None:
        if explicit_column not in fieldnames:
            raise ValueError(
                f"Requested sequence column '{explicit_column}' was not found. "
                f"Available columns: {fieldnames}"
            )
        return explicit_column

    for column in COMMON_SEQUENCE_COLUMNS:
        if column in fieldnames:
            return column

    raise ValueError(
        "Could not infer the sequence column automatically. "
        f"Available columns: {fieldnames}"
    )


def iter_fasta_lengths(path: Path) -> Iterable[int]:
    current_chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_chunks:
                    sequence = normalize_sequence("".join(current_chunks))
                    if sequence is not None:
                        yield len(sequence)
                    current_chunks = []
                continue
            current_chunks.append(line)

    if current_chunks:
        sequence = normalize_sequence("".join(current_chunks))
        if sequence is not None:
            yield len(sequence)


def iter_csv_lengths(path: Path, sequence_column: str | None) -> Iterable[int]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file '{path}' is missing a header row.")
        column = detect_sequence_column(reader.fieldnames, sequence_column)
        for row in reader:
            sequence = normalize_sequence(row.get(column, ""))
            if sequence is not None:
                yield len(sequence)


def iter_lengths(path: Path, sequence_column: str | None) -> Iterable[int]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from iter_csv_lengths(path, sequence_column)
        return
    if suffix in {".fa", ".fasta", ".fna"}:
        yield from iter_fasta_lengths(path)
        return
    raise ValueError(f"Unsupported file type '{path.suffix}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CSV length distribution from FASTA or CSV files.")
    parser.add_argument("--input-files", nargs="+", required=True, help="One or more FASTA or CSV files.")
    parser.add_argument("--output-file", required=True, help="Output CSV file.")
    parser.add_argument("--sequence-column", default=None, help="Sequence column name for CSV inputs.")
    parser.add_argument("--include-source", action="store_true", help="Include a Source column.")
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for input_file in args.input_files:
        input_path = Path(input_file).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input file '{input_path}' does not exist.")
        lengths = list(iter_lengths(input_path, args.sequence_column))
        if not lengths:
            raise ValueError(f"Input file '{input_path}' did not yield any sequences.")
        for length in lengths:
            row = {"Length": length}
            if args.include_source:
                row["Source"] = input_path.name
            rows.append(row)

    fieldnames = ["Length"] + (["Source"] if args.include_source else [])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
