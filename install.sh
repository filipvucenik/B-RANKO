#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="branko"
OUTPUT_DIR="weights"
PRETRAIN_MODEL_NAME="branko_mega_cpool128.ckpt"
APTAMER_MODEL_NAME="branko_aptamer_igfbp3_cpool128.ckpt"

print_usage() {
  cat <<'EOF'
Usage:
  ./install.sh

This script:
  1. creates or updates the `branko` conda environment
  2. installs the package in editable mode
  3. leaves the current `weights/` directory untouched

Notes:
  - model downloads are currently mocked
  - if you already have the released model files in `weights/`, nothing else is needed
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      echo "install.sh does not accept options right now."
      print_usage
      exit 1
      ;;
  esac
fi

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

echo "Environment ready."
echo "Model downloading is currently mocked."
echo "Expected model files:"
echo "  - ${OUTPUT_DIR}/${PRETRAIN_MODEL_NAME}"
echo "  - ${OUTPUT_DIR}/${APTAMER_MODEL_NAME}"
