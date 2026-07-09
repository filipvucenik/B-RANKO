#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="branko"
OUTPUT_DIR="weights"
PRETRAIN_MODEL_NAME="branko_mega_cpool128.ckpt"
APTAMER_MODEL_NAME="branko_aptamer_igfbp3_cpool128.ckpt"
PRETRAIN_MODEL_URL="https://zenodo.org/records/21242783/files/branko_mega_cpool128.ckpt?download=1"
APTAMER_MODEL_URL="https://zenodo.org/records/21242783/files/branko_aptamer_igfbp3_cpool128.ckpt?download=1"
ZENODO_RECORD_URL="https://zenodo.org/records/21242783"
SKIP_WEIGHTS=0
FORCE_DOWNLOAD=0

print_usage() {
  cat <<'EOF'
Usage:
  ./install.sh [--skip-weights] [--force-download]

This script:
  1. creates or updates the `branko` conda environment
  2. installs the package in editable mode
  3. downloads released model files into `weights/`

Options:
  --skip-weights    skip downloading released model files
  --force-download  re-download model files even if they already exist

Notes:
  - released weights are hosted on Zenodo:
    https://zenodo.org/records/21242783
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      print_usage
      exit 0
      ;;
    --skip-weights)
      SKIP_WEIGHTS=1
      shift
      ;;
    --force-download)
      FORCE_DOWNLOAD=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

if command -v mamba >/dev/null 2>&1; then
  CONDA_CMD="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA_CMD="conda"
else
  echo "Could not find mamba or conda."
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # `conda activate` is a shell function, so it must be initialized first.
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  ACTIVATE_CMD="conda"
elif CONDA_HOOK="$("${CONDA_CMD}" shell hook --shell bash 2>/dev/null)"; then
  eval "${CONDA_HOOK}"
  ACTIVATE_CMD="${CONDA_CMD}"
else
  echo "Could not initialize conda/mamba shell activation."
  exit 1
fi

if "${CONDA_CMD}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_CMD}" env update -n "${ENV_NAME}" -f environment.yml --prune
else
  "${CONDA_CMD}" env create -n "${ENV_NAME}" -f environment.yml
fi

set +u
"${ACTIVATE_CMD}" activate "${ENV_NAME}"
set -u
pip install -e .
pip install --no-cache-dir --no-build-isolation flash-attn==2.7.3

mkdir -p "${OUTPUT_DIR}"

download_model() {
  local name="$1"
  local url="$2"
  local destination="${OUTPUT_DIR}/${name}"
  local temp_file

  if [[ -f "${destination}" && "${FORCE_DOWNLOAD}" -eq 0 ]]; then
    echo "Model already present: ${destination}"
    return 0
  fi

  temp_file="$(mktemp "${destination}.tmp.XXXXXX")"

  echo "Downloading ${name} from Zenodo..."
  if [[ "${DOWNLOAD_CMD}" == "curl" ]]; then
    curl --fail --location --progress-bar --output "${temp_file}" "${url}"
  else
    wget --show-progress -O "${temp_file}" "${url}"
  fi

  mv "${temp_file}" "${destination}"
  echo "Saved ${destination}"
}

echo "Environment ready."

if [[ "${SKIP_WEIGHTS}" -eq 1 ]]; then
  echo "Skipping model downloads."
else
  if command -v curl >/dev/null 2>&1; then
    DOWNLOAD_CMD="curl"
  elif command -v wget >/dev/null 2>&1; then
    DOWNLOAD_CMD="wget"
  else
    echo "Could not find curl or wget for model downloads."
    exit 1
  fi

  download_model "${PRETRAIN_MODEL_NAME}" "${PRETRAIN_MODEL_URL}"
  download_model "${APTAMER_MODEL_NAME}" "${APTAMER_MODEL_URL}"
fi

echo "Zenodo record: ${ZENODO_RECORD_URL}"
