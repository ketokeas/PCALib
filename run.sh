#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="jax-nb"

# --- Find Conda base dir (works even if 'conda init' was never run) ---
detect_conda_base() {
  if command -v conda >/dev/null 2>&1; then
    conda info --base
    return
  fi
  # Fallbacks (common install locations)
  for d in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/anaconda3" "/opt/anaconda3" "/opt/miniconda3" "$HOME/miniconda3"; do
    if [ -d "$d" ]; then
      echo "$d"
      return
    fi
  done
  echo "ERROR: Could not find a Conda installation. Install Miniforge/Anaconda first." >&2
  exit 1
}

CONDA_BASE="$(detect_conda_base)"

# --- Enable 'conda activate' in this shell ---
# (don’t rely on 'conda init'; source the profile script directly)
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- Ensure the env exists ---
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' not found."
  echo "Run ./setup.sh first to create it."
  exit 1
fi

# --- Activate and launch ---
conda activate "$ENV_NAME"

# Optional: set Jupyter’s working dir to the repo root (this script’s dir)
cd "$(dirname "$0")"

# Launch JupyterLab
exec jupyter lab
