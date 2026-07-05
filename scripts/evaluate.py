#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import random
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path


def progress(iterable, description: str, total: int | None = None, unit: str = "seq"):
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return iter(iterable)
    return tqdm(iterable, desc=description, total=total, unit=unit)


def progress_bar(description: str, total: int, unit: str):
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return None
    return tqdm(desc=description, total=total, unit=unit)


def parse_named_fasta(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(
            f"Expected NAME=PATH, got '{value}'. Example: --generated branko=runs/demo/generated.fasta"
        )
    name, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"FASTA file '{path}' does not exist.")
    return name, path


def iter_fasta_records(path: Path):
    current_id = None
    current_chunks: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    yield current_id, "".join(current_chunks).upper()
                current_id = line[1:].strip().split()[0]
                current_chunks = []
                continue
            current_chunks.append(line)

    if current_id is not None:
        yield current_id, "".join(current_chunks).upper()


def count_fasta_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def load_records(path: Path, dataset_name: str) -> list[dict[str, str]]:
    total = count_fasta_records(path)
    records = []
    iterator = progress(iter_fasta_records(path), f"Loading {dataset_name}", total=total)
    for record_id, sequence in iterator:
        records.append({"sequence_id": record_id, "sequence": sequence})
    print(f"Loaded {len(records)} sequences for {dataset_name}.", flush=True)
    return records


def gc_content(sequence: str) -> float:
    if not sequence:
        return 0.0
    gc_count = sequence.count("G") + sequence.count("C")
    return 100.0 * gc_count / len(sequence)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": std(values),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
    }


def compute_basic_metrics(records: list[dict[str, str]], dataset_name: str) -> tuple[list[dict], dict]:
    rows = []
    lengths = []
    gc_values = []
    sequences = []

    for record in progress(records, f"Length/GC {dataset_name}", total=len(records)):
        sequence = record["sequence"]
        sequence_length = len(sequence)
        gc_value = gc_content(sequence)
        lengths.append(sequence_length)
        gc_values.append(gc_value)
        sequences.append(sequence)
        rows.append(
            {
                "sequence_id": record["sequence_id"],
                "length": sequence_length,
                "gc_content": gc_value,
            }
        )

    unique_fraction = len(set(sequences)) / max(len(sequences), 1)
    summary = {
        "n_sequences": len(records),
        "unique_fraction": unique_fraction,
        "length": metric_summary(lengths),
        "gc_content": metric_summary(gc_values),
    }
    return rows, summary


def compute_amfe(
    records: list[dict[str, str]],
    dataset_name: str,
    max_samples: int | None,
) -> tuple[list[float], list[int]]:
    try:
        import RNA
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Average MFE evaluation requires the ViennaRNA Python package (`import RNA`)."
        ) from exc

    selected_records = records if max_samples is None else records[:max_samples]
    amfe_values = []
    selected_indices = []
    iterator = progress(
        enumerate(selected_records),
        f"Average MFE {dataset_name}",
        total=len(selected_records),
    )
    for index, record in iterator:
        sequence = record["sequence"]
        if not sequence:
            continue
        _, mfe = RNA.fold(sequence)
        amfe_values.append(mfe / len(sequence))
        selected_indices.append(index)
    return amfe_values, selected_indices


def stable_label_seed(seed: int, label: str) -> int:
    return seed + sum((index + 1) * ord(char) for index, char in enumerate(label))


def subsample_records(
    records: list[dict[str, str]],
    max_samples: int | None,
    seed: int,
    label: str,
) -> tuple[list[dict[str, str]], bool]:
    if max_samples is None or len(records) <= max_samples:
        return records, False

    rng = random.Random(stable_label_seed(seed, label))
    selected_indices = sorted(rng.sample(range(len(records)), max_samples))
    return [records[index] for index in selected_indices], True


def prepare_myers_pattern(pattern: str) -> dict:
    peq = {}
    for index, char in enumerate(pattern):
        peq[char] = peq.get(char, 0) | (1 << index)

    length = len(pattern)
    return {
        "peq": peq,
        "length": length,
        "mask": (1 << length) - 1,
        "high_bit": 1 << (length - 1),
    }


def myers_levenshtein_distance_prepared(prepared_pattern: dict, text: str) -> int:
    pattern_length = prepared_pattern["length"]
    if pattern_length == 0:
        return len(text)

    peq = prepared_pattern["peq"]
    mask = prepared_pattern["mask"]
    high_bit = prepared_pattern["high_bit"]

    pv = mask
    mv = 0
    score = pattern_length

    for char in text:
        eq = peq.get(char, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh

        if ph & high_bit:
            score += 1
        elif mh & high_bit:
            score -= 1

        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | (~(xv | ph) & mask)) & mask
        mv = ph & xv

    return score


def dp_levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for row_index, a_char in enumerate(a, start=1):
        current_row = [row_index]
        for column_index, b_char in enumerate(b, start=1):
            insertion = current_row[column_index - 1] + 1
            deletion = previous_row[column_index] + 1
            substitution = previous_row[column_index - 1] + (a_char != b_char)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_distance(a: str, b: str, prepared_a: dict | None = None, prepared_b: dict | None = None) -> int:
    if a == b:
        return 0

    if len(a) <= len(b):
        if prepared_a is None and len(a) <= 63:
            prepared_a = prepare_myers_pattern(a)
        if prepared_a is not None and prepared_a["length"] <= 63:
            return myers_levenshtein_distance_prepared(prepared_a, b)
    else:
        if prepared_b is None and len(b) <= 63:
            prepared_b = prepare_myers_pattern(b)
        if prepared_b is not None and prepared_b["length"] <= 63:
            return myers_levenshtein_distance_prepared(prepared_b, a)

    return dp_levenshtein_distance(a, b)


def compute_min_sequence_levenshtein(records: list[dict[str, str]], dataset_name: str) -> tuple[dict[str, float], list[float]]:
    value_to_ids: dict[str, list[str]] = {}
    for record in records:
        value_to_ids.setdefault(record["sequence"], []).append(record["sequence_id"])

    unique_sequences = list(value_to_ids)
    distance_by_sequence: dict[str, float] = {}

    for sequence, ids in value_to_ids.items():
        if len(ids) > 1:
            distance_by_sequence[sequence] = 0.0

    singleton_sequences = [sequence for sequence in unique_sequences if sequence not in distance_by_sequence]
    if len(singleton_sequences) == 1:
        distance_by_sequence[singleton_sequences[0]] = float("nan")
    elif len(singleton_sequences) > 1:
        prepared_patterns = {
            sequence: prepare_myers_pattern(sequence) if len(sequence) <= 63 else None
            for sequence in singleton_sequences
        }
        best_distances: dict[str, float | None] = {sequence: None for sequence in singleton_sequences}

        total_pairs = len(singleton_sequences) * (len(singleton_sequences) - 1) // 2
        print(
            f"Computing minimum sequence Levenshtein distances for {dataset_name} "
            f"({total_pairs:,} sequence pairs)...",
            flush=True,
        )
        pair_bar = progress_bar(f"Levenshtein {dataset_name}", total=total_pairs, unit="pair")
        for left_index, sequence_i in enumerate(singleton_sequences):
            for sequence_j in singleton_sequences[left_index + 1 :]:
                raw_distance = levenshtein_distance(
                    sequence_i,
                    sequence_j,
                    prepared_a=prepared_patterns[sequence_i],
                    prepared_b=prepared_patterns[sequence_j],
                )
                normalized_distance = raw_distance / max(len(sequence_i), len(sequence_j), 1)

                current_i = best_distances[sequence_i]
                if current_i is None or normalized_distance < current_i:
                    best_distances[sequence_i] = normalized_distance

                current_j = best_distances[sequence_j]
                if current_j is None or normalized_distance < current_j:
                    best_distances[sequence_j] = normalized_distance
                if pair_bar is not None:
                    pair_bar.update(1)
        if pair_bar is not None:
            pair_bar.close()

        distance_by_sequence.update(
            {sequence: distance for sequence, distance in best_distances.items() if distance is not None}
        )

    distance_by_id = {}
    values = []
    for sequence, ids in value_to_ids.items():
        value = distance_by_sequence.get(sequence, float("nan"))
        for sequence_id in ids:
            distance_by_id[sequence_id] = value
        if not math.isnan(value):
            values.extend([value] * len(ids))

    return distance_by_id, values


def classify_novelty(evalue: float, percent_identity: float, query_coverage: float) -> str:
    if evalue <= 1e-10 and percent_identity >= 70.0 and query_coverage >= 0.7:
        return "known"
    if evalue <= 1e-5 and query_coverage >= 0.3:
        return "novel/known"
    return "novel"


def run_mmseqs_novelty(
    generated_path: Path,
    reference_path: Path,
    dataset_name: str,
    threads: int,
) -> dict[str, str]:
    if shutil.which("mmseqs") is None:
        raise RuntimeError("MMseqs2 is not available on PATH, but novelty evaluation was requested.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        result_path = temp_dir / "mmseqs_results.tsv"
        command = [
            "mmseqs",
            "easy-search",
            str(generated_path),
            str(reference_path),
            str(result_path),
            str(temp_dir / "work"),
            "--search-type",
            "3",
            "--threads",
            str(threads),
            "--cov-mode",
            "5",
            "-s",
            "7.5",
            "--max-seqs",
            "10000",
            "-c",
            "0.9",
            "--min-seq-id",
            "0.8",
            "-e",
            "1e-5",
            "--format-output",
            "query,target,pident,alnlen,qlen,evalue",
        ]
        print(f"Running MMseqs2 novelty search for {dataset_name}...", flush=True)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

        best_hits: dict[str, tuple[float, float, float]] = {}
        if result_path.exists() and result_path.stat().st_size > 0:
            with result_path.open("r", encoding="utf-8") as handle:
                reader = csv.reader(handle, delimiter="\t")
                for row in reader:
                    query, _target, pident, alnlen, qlen, evalue = row
                    score = float(evalue)
                    current = best_hits.get(query)
                    if current is not None and score >= current[0]:
                        continue
                    query_coverage = float(alnlen) / max(float(qlen), 1.0)
                    best_hits[query] = (score, float(pident), query_coverage)

    total = count_fasta_records(generated_path)
    novelty_by_id = {}
    records = progress(
        iter_fasta_records(generated_path),
        f"Novelty labels {dataset_name}",
        total=total,
    )
    for sequence_id, _sequence in records:
        if sequence_id not in best_hits:
            novelty_by_id[sequence_id] = "novel"
            continue
        evalue, pident, qcov = best_hits[sequence_id]
        novelty_by_id[sequence_id] = classify_novelty(evalue, pident, qcov)
    return novelty_by_id


def write_records_to_fasta(records: list[dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in progress(records, f"Writing {output_path.name}", total=len(records)):
            handle.write(f">{record['sequence_id']}\n{record['sequence']}\n")


def run_mmseqs_diversity(
    records: list[dict[str, str]],
    dataset_name: str,
    threads: int,
    max_samples: int | None,
    seed: int,
) -> tuple[dict[str, str], dict]:
    if shutil.which("mmseqs") is None:
        raise RuntimeError("MMseqs2 is not available on PATH, but diversity evaluation was requested.")

    selected_records, was_subsampled = subsample_records(records, max_samples, seed, dataset_name)
    if was_subsampled:
        print(
            f"Diversity {dataset_name}: using {len(selected_records):,} / {len(records):,} sampled sequences.",
            flush=True,
        )
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "input.fasta"
        output_prefix = temp_dir / "clusters"
        print(f"Writing temporary FASTA for diversity clustering: {dataset_name}", flush=True)
        write_records_to_fasta(selected_records, input_path)

        command = [
            "mmseqs",
            "easy-cluster",
            str(input_path),
            str(output_prefix),
            str(temp_dir / "work"),
            "--min-seq-id",
            "0.9",
            "-c",
            "0.8",
            "--threads",
            str(threads),
        ]
        print(f"Running MMseqs2 diversity clustering for {dataset_name}...", flush=True)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

        cluster_file = Path(f"{output_prefix}_cluster.tsv")
        if not cluster_file.exists():
            raise RuntimeError(f"MMseqs2 did not write a cluster file for dataset '{dataset_name}'.")

        cluster_by_id = {}
        cluster_counts = Counter()
        with cluster_file.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in progress(reader, f"Reading clusters {dataset_name}"):
                if len(row) < 2:
                    continue
                cluster_id, sequence_id = row[0], row[1]
                cluster_by_id[sequence_id] = cluster_id
                cluster_counts[cluster_id] += 1

    total_sequences = len(selected_records)
    unique_sequences = len({record["sequence"] for record in selected_records})
    n_clusters = len(cluster_counts)
    n_singletons = sum(count == 1 for count in cluster_counts.values())
    summary = {
        "n_sequences_clustered": total_sequences,
        "subsampled": was_subsampled,
        "n_clusters": n_clusters,
        "diversity_ratio": n_clusters / total_sequences if total_sequences else 0.0,
        "exact_duplicate_rate": (total_sequences - unique_sequences) / total_sequences if total_sequences else 0.0,
        "singleton_sequence_rate": n_singletons / total_sequences if total_sequences else 0.0,
        "singleton_cluster_ratio": n_singletons / n_clusters if n_clusters else 0.0,
    }
    return cluster_by_id, summary


def plot_metric(metric_name: str, metric_by_dataset: dict[str, list[float]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"Skipping plot for {metric_name}: matplotlib is not installed.")
        return

    labels = [label for label, values in metric_by_dataset.items() if values]
    values = [metric_by_dataset[label] for label in labels]
    if not values:
        return

    plt.figure(figsize=(10, 6))
    plt.violinplot(values, showmeans=True, showextrema=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=15)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_novelty_counts(novelty_counts: dict[str, dict[str, int]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Skipping novelty plot: matplotlib is not installed.")
        return

    if not novelty_counts:
        return

    classes = ["known", "novel/known", "novel"]
    labels = list(novelty_counts)
    bottoms = [0.0] * len(labels)

    plt.figure(figsize=(10, 6))
    for novelty_class in classes:
        values = []
        for label in labels:
            counts = novelty_counts[label]
            total = sum(counts.values())
            values.append(100.0 * counts.get(novelty_class, 0) / total if total else 0.0)
        plt.bar(labels, values, bottom=bottoms, label=novelty_class)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    plt.ylabel("Sequences (%)")
    plt.title("Novelty classification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_bar_metric(metric_name: str, values_by_dataset: dict[str, float], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(f"Skipping plot for {metric_name}: matplotlib is not installed.")
        return

    labels = list(values_by_dataset)
    values = [values_by_dataset[label] for label in labels]
    if not labels:
        return

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def round_nested(value):
    if isinstance(value, dict):
        return {key: round_nested(item) for key, item in value.items()}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return round(value, 6)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated FASTA sequences against a reference FASTA.")
    parser.add_argument(
        "--generated",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Generated FASTA dataset. Repeat this flag for multiple datasets.",
    )
    parser.add_argument("--reference", required=True, help="Reference FASTA file.")
    parser.add_argument("--reference-name", default="Reference", help="Label used for the reference dataset.")
    parser.add_argument("--output-dir", default=None, help="Directory for TSV summaries and plots.")
    parser.add_argument("--skip-mfe", action="store_true", help="Skip average MFE evaluation.")
    parser.add_argument("--skip-novelty", action="store_true", help="Skip MMseqs2 novelty evaluation.")
    parser.add_argument("--skip-diversity", action="store_true", help="Skip MMseqs2 diversity clustering.")
    parser.add_argument(
        "--compute-levenshtein",
        action="store_true",
        help="Compute optional minimum normalized sequence Levenshtein distance within each dataset.",
    )
    parser.add_argument("--max-mfe-samples", type=int, default=10000, help="Maximum sequences per dataset for average MFE.")
    parser.add_argument(
        "--max-diversity-samples",
        type=int,
        default=None,
        help="Maximum sequences per dataset for MMseqs2 diversity clustering.",
    )
    parser.add_argument(
        "--max-levenshtein-samples",
        type=int,
        default=1000,
        help="Maximum sequences per dataset for optional Levenshtein evaluation.",
    )
    parser.add_argument("--mmseqs-threads", type=int, default=8, help="Number of threads for MMseqs2.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for metric subsampling.")
    args = parser.parse_args()

    generated_specs = [parse_named_fasta(item) for item in args.generated]
    reference_path = Path(args.reference).expanduser().resolve()
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference FASTA '{reference_path}' does not exist.")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else generated_specs[0][1].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    length_plot_data: dict[str, list[float]] = {}
    gc_plot_data: dict[str, list[float]] = {}
    amfe_plot_data: dict[str, list[float]] = {}
    levenshtein_plot_data: dict[str, list[float]] = {}
    novelty_count_plot_data: dict[str, dict[str, int]] = {}
    diversity_ratio_plot_data: dict[str, float] = {}
    summary_json = {"datasets": {}}

    all_specs = generated_specs + [(args.reference_name, reference_path)]
    novelty_results = {}

    for name, path in all_specs:
        print(f"\nEvaluating {name}: {path}", flush=True)
        records = load_records(path, name)
        per_sequence_rows, dataset_summary = compute_basic_metrics(records, name)

        length_plot_data[name] = [row["length"] for row in per_sequence_rows]
        gc_plot_data[name] = [row["gc_content"] for row in per_sequence_rows]

        if not args.skip_mfe:
            amfe_values, selected_indices = compute_amfe(
                records,
                dataset_name=name,
                max_samples=args.max_mfe_samples,
            )
            amfe_plot_data[name] = amfe_values
            dataset_summary["amfe"] = metric_summary(amfe_values)
            for row_index, amfe_value in zip(selected_indices, amfe_values):
                per_sequence_rows[row_index]["amfe"] = amfe_value

        if not args.skip_novelty and name != args.reference_name:
            novelty_by_id = run_mmseqs_novelty(
                generated_path=path,
                reference_path=reference_path,
                dataset_name=name,
                threads=args.mmseqs_threads,
            )
            novelty_results[name] = novelty_by_id
            counts = {"known": 0, "novel/known": 0, "novel": 0}
            for row in per_sequence_rows:
                novelty_class = novelty_by_id[row["sequence_id"]]
                row["novelty_class"] = novelty_class
                counts[novelty_class] += 1
            dataset_summary["novelty"] = counts
            novelty_count_plot_data[name] = counts

        if not args.skip_diversity:
            cluster_by_id, diversity_summary = run_mmseqs_diversity(
                records=records,
                dataset_name=name,
                threads=args.mmseqs_threads,
                max_samples=args.max_diversity_samples,
                seed=args.seed,
            )
            for row in per_sequence_rows:
                if row["sequence_id"] in cluster_by_id:
                    row["cluster_id"] = cluster_by_id[row["sequence_id"]]
            dataset_summary["diversity"] = diversity_summary
            diversity_ratio_plot_data[name] = diversity_summary["diversity_ratio"]

        if args.compute_levenshtein:
            sampled_records, was_subsampled = subsample_records(
                records=records,
                max_samples=args.max_levenshtein_samples,
                seed=args.seed,
                label=name,
            )
            distance_by_id, levenshtein_values = compute_min_sequence_levenshtein(sampled_records, name)
            for row in per_sequence_rows:
                if row["sequence_id"] in distance_by_id:
                    row["min_seq_levenshtein"] = distance_by_id[row["sequence_id"]]
            dataset_summary["min_seq_levenshtein"] = metric_summary(levenshtein_values)
            dataset_summary["min_seq_levenshtein"]["n_sequences_evaluated"] = len(sampled_records)
            dataset_summary["min_seq_levenshtein"]["subsampled"] = was_subsampled
            levenshtein_plot_data[name] = levenshtein_values

        summary_rows.append(
            {
                "dataset": name,
                "n_sequences": dataset_summary["n_sequences"],
                "unique_fraction": dataset_summary["unique_fraction"],
                "length_mean": dataset_summary["length"]["mean"],
                "length_std": dataset_summary["length"]["std"],
                "gc_mean": dataset_summary["gc_content"]["mean"],
                "gc_std": dataset_summary["gc_content"]["std"],
                "amfe_mean": dataset_summary.get("amfe", {}).get("mean"),
                "amfe_std": dataset_summary.get("amfe", {}).get("std"),
                "novelty_known": dataset_summary.get("novelty", {}).get("known"),
                "novelty_novel_known": dataset_summary.get("novelty", {}).get("novel/known"),
                "novelty_novel": dataset_summary.get("novelty", {}).get("novel"),
                "diversity_ratio": dataset_summary.get("diversity", {}).get("diversity_ratio"),
                "n_clusters": dataset_summary.get("diversity", {}).get("n_clusters"),
                "exact_duplicate_rate": dataset_summary.get("diversity", {}).get("exact_duplicate_rate"),
                "singleton_sequence_rate": dataset_summary.get("diversity", {}).get("singleton_sequence_rate"),
                "min_seq_levenshtein_mean": dataset_summary.get("min_seq_levenshtein", {}).get("mean"),
                "min_seq_levenshtein_std": dataset_summary.get("min_seq_levenshtein", {}).get("std"),
            }
        )
        summary_json["datasets"][name] = round_nested(dataset_summary)
        write_tsv(output_dir / f"evaluation_results_{name}.tsv", per_sequence_rows)
        print(f"Finished {name}.", flush=True)

    write_tsv(output_dir / "evaluation_summary.tsv", summary_rows)
    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(round_nested(summary_json), handle, indent=2, sort_keys=True)

    plot_metric("Length", length_plot_data, output_dir / "length_violin.png")
    plot_metric("GC Content (%)", gc_plot_data, output_dir / "gc_content_violin.png")
    if not args.skip_mfe:
        plot_metric("Average MFE (MFE/nt)", amfe_plot_data, output_dir / "amfe_violin.png")
    if not args.skip_novelty:
        plot_novelty_counts(novelty_count_plot_data, output_dir / "novelty_barplot.png")
    if not args.skip_diversity:
        plot_bar_metric("Diversity ratio", diversity_ratio_plot_data, output_dir / "diversity_ratio.png")
    if args.compute_levenshtein:
        plot_metric(
            "Minimum normalized sequence Levenshtein distance",
            levenshtein_plot_data,
            output_dir / "min_seq_levenshtein_violin.png",
        )

    print(f"Wrote evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
