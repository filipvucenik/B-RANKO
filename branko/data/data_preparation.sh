#!/bin/bash
# ===================================================
# RNA Dataset Preparation Script
# Accepts a YAML config file as an argument
# Handles duplicates, filtering, train/test split
# ===================================================

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_file.yml>"
    exit 1
fi

conf_file="$1"
if [ ! -f "$conf_file" ]; then
    echo "Error: Config file '$conf_file' not found!"
    exit 1
fi

echo "Using config file: $conf_file"

# Extract variables from YAML
output_dir=$(yq -r '.output_dir' "$conf_file")
readarray -t rna_datasets < <(yq -r '.rna_datasets[]' "$conf_file")
verbose=$(yq -r '.verbose' "$conf_file")
special_char_threshold=$(yq -r '.special_char_threshold' "$conf_file")
min_seq_length=$(yq -r '.min_seq_length' "$conf_file")
max_seq_length=$(yq -r '.max_seq_length' "$conf_file")
mmseqs_min_seq_id=$(yq -r '.mmseqs.min_seq_id' "$conf_file")
mmseqs_coverage=$(yq -r '.mmseqs.coverage' "$conf_file")
threads=$(yq -r '.threads' "$conf_file")
prepare_representatives_dataset=$(yq -r '.prepare_representatives_dataset' "$conf_file")
train_test_split_test_size=$(yq -r '.train_test_split.test_size' "$conf_file")
train_test_split_random_seed=$(yq -r '.train_test_split.random_seed' "$conf_file")
keep_all_files=$(yq -r '.keep_all_files' "$conf_file")
log_file="$output_dir/log.txt"

# Ensure output_dir exists
mkdir -p "$output_dir" || { echo "Cannot create output_dir: $output_dir"; exit 1; }
cp "$conf_file" "$output_dir/" 
echo "Output directory: $output_dir"
touch "$log_file"
echo "Log file: $log_file"
echo "Data preparation started at: $(date)" > "$log_file"

# Download datasets
sequences=()
for dataset in "${rna_datasets[@]}"; do
    filename=$(basename "$dataset")
    if [[ "$dataset" == https://*.gz ]]; then
        # remote gz file
        filename="${filename%.gz}"
        wget -qO- "$dataset" | gunzip > "$output_dir/$filename"
    elif [[ "$dataset" == *.gz ]]; then
        # local gz file
        relfile="${dataset%.gz}"
        filename="${filename%.gz}"
        # echo "procesing local gz file: $output_dir/$filename.gz"
        gunzip -c "$output_dir/$relfile.gz" > "$output_dir/$filename"
    elif [[ "$dataset" == https://*.zip ]]; then
        # remote zip file
        wget -O "$output_dir/$filename" "$dataset"
        unzip -o "$output_dir/$filename" -d "$output_dir"
    else
        # local or remote plain file
        wget -O "$output_dir/$filename" "$dataset" 2>/dev/null || cp "$dataset" "$output_dir/$filename"
    fi

    sequences+=("$filename")
    echo "Downloaded and processed: $filename"
    echo "Downloaded and processed: $filename" >> "$log_file"
done

# Remove all non-FASTA/FA files from output_dir
find "$output_dir" -type f ! \( -iname "*.fasta" -o -iname "*.fa" -o -iname "log.txt" \) -delete
echo "Removed all non-FASTA/FA files from $output_dir"
echo "Removed all non-FASTA/FA files from $output_dir" >> "$log_file"
find "$output_dir" -type d -empty -delete
echo "Removed all empty directories from $output_dir"
echo "Removed all empty directories from $output_dir" >> "$log_file"

# Merge datasets into one FASTA
if [ ${#sequences[@]} -gt 1 ]; then
    cat "${sequences[@]/#/$output_dir/}" > "$output_dir/dataset.fasta"
    for seq in "${sequences[@]}"; do
        seqkit stats "$output_dir/$seq" 
        seqkit stats "$output_dir/$seq" >> "$log_file"
    done
    echo "Merged all datasets into: $output_dir/dataset.fasta"
    echo "Merged all datasets into: $output_dir/dataset.fasta" >> "$log_file"
    rm "${sequences[@]/#/$output_dir/}"
else
    mv "$output_dir/${sequences[0]}" "$output_dir/dataset.fasta"
    echo "Single dataset renamed to: $output_dir/dataset.fasta"
    echo "Single dataset renamed to: $output_dir/dataset.fasta" >> "$log_file"
fi
seqkit stats "$output_dir/dataset.fasta"
seqkit stats "$output_dir/dataset.fasta" >> "$log_file"

# Convert sequences to uppercase
seqkit seq -u -o "$output_dir/dataset_upper.fasta" "$output_dir/dataset.fasta"
mv "$output_dir/dataset_upper.fasta" "$output_dir/dataset.fasta"
echo "Converted sequences to uppercase"
echo "Converted sequences to uppercase" >> "$log_file"

# Remove duplicates
seqkit rmdup -s -o "$output_dir/unique_dataset.fasta" "$output_dir/dataset.fasta"
mv "$output_dir/unique_dataset.fasta" "$output_dir/dataset.fasta"
echo "Removed duplicate sequences"
echo "Removed duplicate sequences" >> "$log_file"

seqkit stats "$output_dir/dataset.fasta"
seqkit stats "$output_dir/dataset.fasta" >> "$log_file"
# Filter sequences by special characters
awk -v t="$special_char_threshold" '
BEGIN {RS=">"; ORS=""} 
NR > 1 {
    n = split($0, lines, "\n")
    header = lines[1]
    seq = ""
    for (i = 2; i <= n; i++) seq = seq lines[i]
    temp = seq
    n_count = gsub(/[NXRYSWKMBDHVnxryswkmbdhv]/, "", temp)
    if (n_count / length(seq) <= t)
        print ">" header "\n" seq "\n"
}
' "$output_dir/dataset.fasta" > "$output_dir/filtered_dataset.fasta"
mv "$output_dir/filtered_dataset.fasta" "$output_dir/dataset.fasta"
echo "Filtered sequences with too many special characters"
echo "Filtered sequences with too many special characters" >> "$log_file"
seqkit stats "$output_dir/dataset.fasta"
seqkit stats "$output_dir/dataset.fasta" >> "$log_file"

# Filter sequences by length
awk -v min_seq_length="$min_seq_length" -v max_seq_length="$max_seq_length" '
BEGIN {RS=">"; ORS=""} 
NR>1 {
    n = split($0, lines, "\n")
    header = lines[1]
    seq = ""
    for (i=2; i<=n; i++) seq = seq lines[i]
    if (length(seq) >= min_seq_length && length(seq) <= max_seq_length)
        print ">" header "\n" seq "\n"
}
' "$output_dir/dataset.fasta" > "$output_dir/cleaned_dataset.fasta"
mv "$output_dir/cleaned_dataset.fasta" "$output_dir/dataset.fasta"
echo "Filtered sequences by length"
echo "Filtered sequences by length" >> "$log_file"
seqkit stats "$output_dir/dataset.fasta"
seqkit stats "$output_dir/dataset.fasta" >> "$log_file"

# Clustering with MMseqs2 (optional)
if [ "$prepare_representatives_dataset" = true ] ; then
    echo "Running MMseqs2 clustering..."
    echo "MMseqs2 parameters: min_seq_id=$mmseqs_min_seq_id, coverage=$mmseqs_coverage, threads=$threads"
    echo "Running MMseqs2 clustering..." >> "$log_file"
    echo "MMseqs2 parameters: min_seq_id=$mmseqs_min_seq_id, coverage=$mmseqs_coverage, threads=$threads" >> "$log_file"

    mmseqs easy-linclust \
        --min-seq-id "$mmseqs_min_seq_id" \
        -c "$mmseqs_coverage" \
        --threads "$threads" \
        "$output_dir/dataset.fasta" \
        "$output_dir/mmseqs" \
        "$output_dir/mmseqs_tmp"

    mv "$output_dir/mmseqs_all_seqs.fasta" "$output_dir/dataset.fasta"
    mv "$output_dir/mmseqs_rep_seq.fasta" "$output_dir/dataset_representatives.fasta"
    mv "$output_dir/mmseqs_cluster.tsv" "$output_dir/dataset_clusters.tsv"

    rm -rf "$output_dir/mmseqs_tmp"

    echo "Clustering completed"
    echo "Clustering completed" >> "$log_file"
fi

# Prepare dataset (representatives or full)
dataset="dataset"
if [ "$prepare_representatives_dataset" = true ] ; then
    dataset="dataset_representatives"
fi
# Make FASTA headers unique
python - <<EOF
from pyfaidx import Fasta
from collections import Counter

fasta_file = '$output_dir/$dataset.fasta'
lines = open(fasta_file).readlines()
seen = Counter()
with open(fasta_file, 'w') as f:
    for line in lines:
        if line.startswith('>'):
            header = line.strip()
            if seen[header]:
                header = f"{header}_{seen[header]}"
            seen[line.strip()] += 1
            f.write(header + '\n')
        else:
            f.write(line)
EOF

# Index dataset
python -c "from pyfaidx import Fasta; Fasta('$output_dir/$dataset.fasta', rebuild=True)"
echo "Indexed FASTA: $output_dir/$dataset.fasta"
echo "Indexed FASTA: $output_dir/$dataset.fasta" >> "$log_file"

# Train/test split
echo "Making train/test split"
echo "Making train/test split" >> "$log_file"

input_fasta="$output_dir/$dataset.fasta"
train_fasta="$output_dir/${dataset}_train.fasta"
val_fasta="$output_dir/${dataset}_val.fasta"

# Fast sequence count (no Python, no indexing)
SEQ_COUNT=$(grep -c '^>' "$input_fasta")
export SEQ_COUNT

python - <<EOF
from pyfaidx import Fasta
import random
import os

random.seed($train_test_split_random_seed)

fasta = Fasta('$input_fasta', rebuild=False)

seq_count = int(os.environ['SEQ_COUNT'])
test_size = int($train_test_split_test_size)

# Sample EXACT test indices without building huge lists
test_indices = set(random.sample(range(seq_count), test_size))

with open('$train_fasta', 'w') as train_file, open('$val_fasta', 'w') as val_file:
    for i, seq_id in enumerate(fasta.keys()):
        seq = fasta[seq_id]
        out = val_file if i in test_indices else train_file
        out.write(f'>{seq_id}\n{seq}\n')
EOF

echo "Created train and test splits: ${dataset}_train.fasta and ${dataset}_val.fasta"
echo "Created train and test splits: ${dataset}_train.fasta and ${dataset}_val.fasta" >> "$log_file"

# Index train/test FASTA
python -c "from pyfaidx import Fasta; Fasta('$output_dir/${dataset}_train.fasta', rebuild=True)"
python -c "from pyfaidx import Fasta; Fasta('$output_dir/${dataset}_val.fasta', rebuild=True)"
echo "Indexed train and test FASTA files"
echo "Indexed train and test FASTA files" >> "$log_file"

# Cleanup intermediate files
if [ "$keep_all_files" = false ] ; then
    rm -f "$output_dir/$dataset.fasta"
    rm -f "$output_dir/$dataset.fasta.fai"
    rm -f "$output_dir/dataset_representatives.fasta"
    rm -f "$output_dir/dataset_clusters.tsv"
fi

echo "Data preparation completed. Final dataset is split into train and test sets at: $output_dir/"
echo "Data preparation completed. Final dataset is split into train and test sets at: $output_dir/" >> "$log_file"
echo "Data preparation finished at: $(date)" >> "$log_file"
