#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="jax-nb"

# --- locate conda and enable 'conda activate' in this shell ---
detect_conda_base() {
  if command -v conda >/dev/null 2>&1; then conda info --base && return; fi
  for d in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/anaconda3" "/opt/anaconda3" "/opt/miniconda3" "$HOME/miniconda3"; do
    [ -d "$d" ] && { echo "$d"; return; }
  done
  echo "ERROR: Conda not found. Install Miniforge/Anaconda first." >&2; exit 1
}
CONDA_BASE="$(detect_conda_base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"

# (optional) recreate clean env
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
conda env create -n "$ENV_NAME" -f environment.yml
conda activate "$ENV_NAME"

# --- choose the right JAX ---
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "OS=$OS ARCH=$ARCH"
# CPU
python -m pip install -U "jax[cpu]"

# install your package (editable)
python -m pip install -e .

# register kernel
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# quick sanity printout
python - <<'PY'
import jax
print("JAX:", jax.__version__)
print("Devices:", jax.devices())
PY

echo "Setup complete. Use ./run.sh to start JupyterLab."
