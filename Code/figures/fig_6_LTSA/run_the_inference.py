#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors

from pcalib.synthetic import build_corridor_potential
from pcalib.classes import Potential
from pcalib.functions import extrapolate_potential, asymptotic_regime_trials
from pcalib.utils import PCA_matlab_like, align_pca_to_reference_svd


# ============================================================
# Output directory
# ============================================================

try:
    OUT = Path(__file__).resolve().parent / "cached_results"
except NameError:
    OUT = Path.cwd() / "cached_results"

OUT.mkdir(parents=True, exist_ok=True)

for axis_name in ["trials", "neurons", "animals"]:
    (OUT / axis_name).mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def make_block_G(G0, n_animals):
    """Build block-diagonal G for n_animals copies of one-animal G0."""
    I = np.eye(n_animals)
    blocks = np.einsum("ab,ac,ij->abicj", I, I, G0[0])
    return blocks.reshape(n_animals, n_animals * G0.shape[1], n_animals * G0.shape[2])


def true_local_varx(Z_true, n_neighbors):
    """
    Local variance of the true latent trajectory in true-space neighborhoods.

    Returns
    -------
    varx_local : array, shape (T,)
    neighbor_ids : array, shape (T, n_neighbors)
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nbrs.fit(Z_true)

    neighbor_ids = nbrs.kneighbors(Z_true, return_distance=False)[:, 1:]

    X = Z_true[neighbor_ids]
    Xc = X - X.mean(axis=1, keepdims=True)

    varx_local = np.mean(np.sum(Xc**2, axis=2), axis=1)

    return varx_local, neighbor_ids


def affine_align_to_true(Z_true, Z_hat):
    """
    Globally align Z_hat to Z_true by translation and full affine transformation.
    """
    X = Z_true - Z_true.mean(axis=0, keepdims=True)
    Y = Z_hat - Z_hat.mean(axis=0, keepdims=True)

    A, _, _, _ = np.linalg.lstsq(Y, X, rcond=None)

    return Y @ A, A


def local_affine_epsilon(Z_true, Z_hat, neighbor_ids):
    """
    Local mean-centered epsilon after local affine alignment.

    For each true-space neighborhood, the corresponding neighborhood of Z_hat is
    mapped to the neighborhood of Z_true by a full linear map.
    """
    X = Z_true[neighbor_ids]
    Y = Z_hat[neighbor_ids]

    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)

    T_local, _, K = Xc.shape

    eps_local = np.zeros(T_local)
    A_local = np.zeros((T_local, K, K))

    for t in range(T_local):
        A, _, _, _ = np.linalg.lstsq(Yc[t], Xc[t], rcond=None)
        Y_aligned = Yc[t] @ A

        A_local[t] = A
        eps_local[t] = np.mean(np.sum((Y_aligned - Xc[t]) ** 2, axis=1))

    return eps_local.mean(), eps_local, A_local


def run_ltsa(data, n_neighbors, n_components):
    """Fit LTSA embedding."""
    model = LocallyLinearEmbedding(
        n_neighbors=n_neighbors,
        n_components=n_components,
        method="ltsa",
        eigen_solver="dense",
    )
    return model.fit_transform(data)


def pca_epsilon(data, potential, n_neurons, K):
    """PCA trajectory epsilon after alignment to true modes."""
    coeff, score, _ = PCA_matlab_like(data)

    v = coeff[:, :K] * np.sqrt(n_neurons)
    y = score[:, :K] / np.sqrt(n_neurons)

    v, y, _, _ = align_pca_to_reference_svd(
        v,
        potential.bar_e,
        y,
    )

    return np.mean([
        np.mean((y[:, k] - potential.bar_x[:, k]) ** 2)
        for k in range(K)
    ])


def run_axis(axis_name, grid, make_potential, n_neurons_of_potential):
    """
    Run PCA and LTSA over one extrapolation axis.

    LTSA is run for k = 10, 15, 20.
    The same k is used for LTSA fitting and local epsilon evaluation.
    """
    pca_eps = np.zeros((len(grid), n_attempts))
    ltsa_eps = np.zeros((len(n_neighbors_array), len(grid), n_attempts))
    ltsa_affine_map = np.zeros((len(n_neighbors_array), len(grid), n_attempts, K, K))
    ltsa_local_varx = np.zeros((len(n_neighbors_array), len(grid), T))

    for i, value in enumerate(grid):
        print(f"{axis_name} = {value}")

        potential = make_potential(value)
        potential.save_as_npz(OUT / axis_name / f"{value}.npz")

        data = potential.generate_sample_data(n_samples=n_attempts)
        n_neurons = n_neurons_of_potential(value)

        for q, k in enumerate(n_neighbors_array):
            varx_local, neighbor_ids = true_local_varx(
                potential.bar_x,
                n_neighbors=k,
            )
            ltsa_local_varx[q, i] = varx_local

            for j in range(n_attempts):
                data_local = data[j]

                if q == 0:
                    pca_eps[i, j] = pca_epsilon(
                        data_local,
                        potential,
                        n_neurons=n_neurons,
                        K=K,
                    )

                Z_hat_raw = run_ltsa(
                    data_local,
                    n_neighbors=k,
                    n_components=K,
                )

                Z_hat, A = affine_align_to_true(
                    Z_true=potential.bar_x,
                    Z_hat=Z_hat_raw,
                )

                ltsa_affine_map[q, i, j] = A

                ltsa_eps[q, i, j], _, _ = local_affine_epsilon(
                    Z_true=potential.bar_x,
                    Z_hat=Z_hat,
                    neighbor_ids=neighbor_ids,
                )

    np.save(OUT / f"pca_epsilon_{axis_name}.npy", pca_eps)
    np.save(OUT / f"ltsa_epsilon_{axis_name}.npy", ltsa_eps)
    np.save(OUT / f"ltsa_affine_map_{axis_name}.npy", ltsa_affine_map)
    np.save(OUT / f"ltsa_local_varx_{axis_name}.npy", ltsa_local_varx)

    return pca_eps, ltsa_eps


def plot_quick_check(x, pca_eps, ltsa_eps, xlabel, title):
    """Small sanity-check plot."""
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        x,
        pca_eps.mean(axis=1),
        yerr=pca_eps.std(axis=1),
        fmt="o-",
        capsize=4,
        label="PCA",
    )

    for q, k in enumerate(n_neighbors_array):
        plt.errorbar(
            x,
            ltsa_eps[q].mean(axis=1),
            yerr=ltsa_eps[q].std(axis=1),
            fmt="o-",
            capsize=3,
            label=f"LTSA k={k}",
        )

    plt.xlabel(xlabel)
    plt.ylabel(r"$\epsilon$")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ============================================================
# Parameters
# ============================================================

N_array = np.arange(30, 351, 30).astype(int)
n_trials_array = np.arange(5, 101, 10).astype(int)
A_array = np.arange(1, 11, 1).astype(int)

n_neighbors_array = np.array([10, 15, 20])

T = 200
K = 2

N0 = int(N_array[0])
A0 = int(A_array[0])
n_trials0 = int(n_trials_array[0])

n_attempts = 30


# ============================================================
# Save grids
# ============================================================

np.save(OUT / "neurons_grid.npy", N_array)
np.save(OUT / "trials_grid.npy", n_trials_array)
np.save(OUT / "animals_grid.npy", A_array)
np.save(OUT / "n_neighbors_grid.npy", n_neighbors_array)


# ============================================================
# Initial potential
# ============================================================

potential_init = build_corridor_potential(
    T,
    A0,
    int(N0 / A0),
    var_array=(4.0, 2.0),
    epsilon_corridor=0.1,
    tau_sigma=0,
    tau_xi=0,
    sigma_mean=0.1,
    sigma_std=0.01,
    xi_mean=(0, 0),
    xi_std=(0, 0),
    Xi=None,
    rng=None,
)

potential_init.save_as_npz(OUT / "potential_init.npz")
np.save(OUT / "signal_init.npy", potential_init.bar_x)
np.save(OUT / "var_signal_init.npy", np.mean(np.var(potential_init.bar_x, axis=0)))

_, asymp_epsilon = asymptotic_regime_trials(
    potential_init,
    n_trials=n_trials0,
)

print("PCA epsilon asymptotic onset estimate:", asymp_epsilon)

potential_init = Potential.from_npz(OUT / "potential_init.npz")

'''
# ============================================================
# Trials axis
# ============================================================

pca_trials, ltsa_trials = run_axis(
    axis_name="trials",
    grid=n_trials_array,
    make_potential=lambda n_trials: extrapolate_potential(
        potential_init,
        new_trials=int(n_trials),
        existing_number_of_trials=n_trials0,
        mode="trial-averaged",
    ),
    n_neurons_of_potential=lambda n_trials: N0,
)

plot_quick_check(
    n_trials_array,
    pca_trials,
    ltsa_trials,
    xlabel="Number of trials",
    title="PCA vs LTSA epsilon: trials axis",
)
'''

# ============================================================
# Neurons axis
# ============================================================

pca_neurons, ltsa_neurons = run_axis(
    axis_name="neurons",
    grid=N_array,
    make_potential=lambda n_neurons: extrapolate_potential(
        potential_init,
        new_neurons=int(n_neurons),
        mode="trial-averaged",
    ),
    n_neurons_of_potential=lambda n_neurons: int(n_neurons),
)

plot_quick_check(
    N_array,
    pca_neurons,
    ltsa_neurons,
    xlabel="Number of neurons",
    title="PCA vs LTSA epsilon: neurons axis",
)

'''
# ============================================================
# Animals axis
# ============================================================

pca_animals, ltsa_animals = run_axis(
    axis_name="animals",
    grid=A_array,
    make_potential=lambda n_animals: extrapolate_potential(
        potential_init,
        new_neurons=N0 * int(n_animals),
        new_G=make_block_G(potential_init.G, int(n_animals)),
        mode="trial-averaged",
    ),
    n_neurons_of_potential=lambda n_animals: N0 * int(n_animals),
)

plot_quick_check(
    A_array,
    pca_animals,
    ltsa_animals,
    xlabel="Number of animals",
    title="PCA vs LTSA epsilon: animals axis",
)
'''