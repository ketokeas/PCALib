from pathlib import Path

import numpy as np
from scipy.io import loadmat


folder = Path("preprocessed")

data_path = folder / "binned_dStr.mat"
G_path = folder / "G_binned_dStr.mat"

out_data_path = Path("preformatted_data.npy")
out_G_path = Path("G.npy")


def load_mat_variable(path: Path, name: str):
    data = loadmat(path)
    if name not in data:
        keys = [k for k in data.keys() if not k.startswith("__")]
        raise KeyError(f"Variable '{name}' not found in {path}. Available: {keys}")
    return data[name]


def remove_zero_firing_neurons(X: np.ndarray, G: np.ndarray):
    """
    Keep neurons that fire at least once in every trial.

    X: [trials, time, neurons]
    G: [animals, neurons, neurons]
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape [trials, time, neurons], got {X.shape}")

    if G.ndim != 3:
        raise ValueError(f"Expected G with shape [animals, neurons, neurons], got {G.shape}")

    if G.shape[1:] != (X.shape[2], X.shape[2]):
        raise ValueError(f"Incompatible shapes: X={X.shape}, G={G.shape}")

    spike_counts = np.sum(X, axis=1)          # [trials, neurons]
    mask = np.min(spike_counts, axis=0) > 0   # [neurons]

    return X[:, :, mask], G[:, mask, :][:, :, mask], mask


if __name__ == "__main__":
    X = load_mat_variable(data_path, "X")
    G = load_mat_variable(G_path, "G")

    print(f"Original X shape: {X.shape}")
    print(f"Original G shape: {G.shape}")

    X_masked, G_masked, mask = remove_zero_firing_neurons(X, G)

    print(f"Neurons before masking: {mask.size}")
    print(f"Neurons after masking:  {mask.sum()}")
    print(f"Masked X shape: {X_masked.shape}")
    print(f"Masked G shape: {G_masked.shape}")
    print("Neurons per animal:", np.sum(np.sum(G_masked, axis=1), axis=1))

    np.save(out_data_path, X_masked)
    np.save(out_G_path, G_masked)

    print(f"Saved {out_data_path}")
    print(f"Saved {out_G_path}")
