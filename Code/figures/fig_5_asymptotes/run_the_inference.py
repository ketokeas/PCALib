#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import numpy as np

# --- project imports ---
sys.path.insert(0, os.path.relpath("../../"))

from pcalib.utils import generate_periodic_exponential_kernel, convolve_data
from pcalib.functions import (
    determine_dimensionality,
    fit_statistics_from_dataset_diagonal,
)


# ====================== CONFIG ======================

DATASET_NAME = "3 Luo et al"

DATA_ROOT = Path("../../../Data")
DATA_PATH = DATA_ROOT / DATASET_NAME / "preformatted_data.npy"
G_PATH = DATA_ROOT / DATASET_NAME / "G.npy"

OUT = Path("cached_results") / DATASET_NAME / "potentials"

MODE = "trial-averaged"   # or "trial-concatenated"
TAU_SIGMA = 2.0
GAMMA = 0.07

SMOOTH_DATA = True


# ====================== CORE ======================

def load_and_preprocess():
    data = np.load(DATA_PATH).astype(float)   # [trials, T, N]
    G = np.load(G_PATH).astype(int)           # [D, N, N]

    print("Loaded data:", data.shape)
    print("Loaded G:", G.shape)
    print("Neurons per animal:", np.einsum("dii->d", G))

    # Same normalization convention as in the main inference script.
    scale = np.std(np.mean(data, axis=0, keepdims=True), axis=1, keepdims=True)
    scale = np.where(scale == 0, 1.0, scale)
    data = data / scale

    if SMOOTH_DATA:
        kernel = generate_periodic_exponential_kernel(data.shape[1], TAU_SIGMA)
        data = convolve_data(data, kernel)

    return data, G


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data, G = load_and_preprocess()

    print("Determining K on the full dataset...")
    K = determine_dimensionality(data, MODE, plot=True)
    print("K =", K)

    np.save(OUT / "K.npy", np.array(K, dtype=int))

    print("Fitting potentials on the full dataset...")
    potentials, _ = fit_statistics_from_dataset_diagonal(
        data,
        K,
        G,
        TAU_SIGMA,
        mode=MODE,
        gamma=GAMMA,
    )

    for k, pot in enumerate(potentials, start=1):
        path = OUT / f"pot_pc{k}.npz"
        pot.save_as_npz(str(path))
        print(f"Saved PC{k}: {path}")

    print("Done.")
    print("Saved potentials to:", OUT.resolve())


if __name__ == "__main__":
    main()