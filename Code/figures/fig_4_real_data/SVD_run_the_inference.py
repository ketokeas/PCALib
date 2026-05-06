#!/usr/bin/env python3
# --------------------------------------------------------------------
# Figures 4–6 pipeline (single dataset; three axes) with unified "available data"
#
# MODIFIED VERSION:
# - Prediction/theory side uses the THREE LOWEST VALUES of each grid
#   as separate base fits, exactly in the spirit of run_the_inference_fig3.py
# - No theory "attempts": one random ordering is chosen once, then everything
#   is fit/extrapolated from that ordering
# - Empirical resampling side is kept as before
# - NEW: if base potentials for a given axis/base/K are already saved in the
#   expected subfolder, they are loaded and reused instead of being inferred again
# - EMPIRICAL UPDATE: empirical rho/epsilon now uses the SVD/canonical-overlap
#   approach on the cross-overlap matrix M = (V_A^T V_B)/N
# - EPSILON UPDATE: empirical epsilon is additionally divided by
#   kernel_integral_squared = sum(kernel**2)
#
# Saves to: cached_results/.../real_gc_grid/
# --------------------------------------------------------------------

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- project imports ---
sys.path.insert(0, os.path.relpath("../../"))
from pcalib.utils import (
    PCA_matlab_like,
    reduce_to_2d,
    generate_periodic_exponential_kernel,
    convolve_data,
)
from pcalib.functions import (
    determine_dimensionality,
    fit_statistics_from_dataset_diagonal,
    make_predictions,
    extrapolate_potential,
)
from pcalib.classes import Potential

# ====================== CONFIG ======================

# Dataset folder name (used in multiple paths)
DATASET_NAME = "2 Chillale et al"

# Dataset
DATA_PATH = Path("../../../Data") / DATASET_NAME / "preformatted_data.npy"
G_PATH = Path("../../../Data") / DATASET_NAME / "G.npy"

MODE = "trial-averaged"  # or "trial-concatenated"
TAU_SIGMA = 2.0          # smoothing kernel width + used in fitting
GAMMA = 0.04             # step/damping for the diagonal fitter

SMOOTH_BEFORE_SPLIT = True

# Reproducibility
SEED = 2 #1000
RNG = np.random.default_rng(SEED)

# Work factors
SIGVAR_N_STEPS = 30
SIGVAR_N_SAMPLES = 200
N_REPETITIONS = 200

# --- Grids you control ---
TRIALS_GRID = None#[10, 11, 12, 13, 14, 15]         # e.g., [4, 6, 8, 10, 12]; default is auto
N_PRE_PER_ANIMAL = None     # per-animal neurons in available data; auto if None
NEURONS_GRID_STEPS = 6

# Number of base points used for infer->extrapolate on each axis
N_BASE_SERIES = 3

# Reuse already saved initial-fit potentials if present
REUSE_SAVED_POTENTIALS = True

# Output directory
OUT = Path("cached_results") / DATASET_NAME / "real_gc_grid"
OUT.mkdir(parents=True, exist_ok=True)

# ==================== Helpers ====================


def save_np(name, arr):
    np.save(OUT / f"{name}.npy", arr)


def infer_animals_from_G(G):
    D, N, _ = G.shape
    diag_stack = np.stack([np.diag(G[d]) for d in range(D)], axis=0)  # [D, N]
    animal_idx = np.argmax(diag_stack, axis=0)
    groups = [np.where(animal_idx == d)[0] for d in range(D)]
    return animal_idx, groups


def build_subG(G, kept_idxs, order_animals=None):
    D, N, _ = G.shape
    animal_idx, _ = infer_animals_from_G(G)
    kept_animals = sorted(set(int(animal_idx[i]) for i in kept_idxs))
    if order_animals is None:
        order_animals = kept_animals
    D_new = len(order_animals)
    out = np.zeros((D_new, len(kept_idxs), len(kept_idxs)))
    mapping = {old: new for new, old in enumerate(order_animals)}
    for old_d in kept_animals:
        new_d = mapping[old_d]
        block = G[old_d][np.ix_(kept_idxs, kept_idxs)]
        out[new_d] = block
    return out


def _empirical_two_groups(A_data, B_data, K, kernel, mode):
    """
    Empirical rho/epsilon between two datasets of the same shape: [n_trials_sel, T, N_sel].

    Uses SVD of the cross-overlap matrix
        M = (V_A^T V_B) / N
    to define canonical overlaps.

    Returns:
        rho[k]   = 1 - sqrt(sigma_k),
                   where sigma_k are singular values of M
                   (i.e. canonical squared overlaps)
        eps[k]   = score mismatch in the corresponding canonical basis,
                   divided by kernel_integral_squared
        mu_hat   = sqrt( (1/K) ||M||_F^2 )
    """

    XA = reduce_to_2d(A_data, mode)
    XB = reduce_to_2d(B_data, mode)
    XA -= np.mean(XA, axis=0, keepdims=True)
    XB -= np.mean(XB, axis=0, keepdims=True)

    coeffA, scoreA, _ = PCA_matlab_like(XA)
    coeffB, scoreB, _ = PCA_matlab_like(XB)

    coeffA = coeffA[:, :K]
    scoreA = scoreA[:, :K]
    coeffB = coeffB[:, :K]
    scoreB = scoreB[:, :K]

    N = coeffA.shape[0]
    T = scoreA.shape[0]

    # Rescale to match your paper normalization: ||v^(k)||^2 = N
    coeffA = np.array(coeffA, dtype=float) * np.sqrt(N)
    coeffB = np.array(coeffB, dtype=float) * np.sqrt(N)
    scoreA = np.array(scoreA, dtype=float) / np.sqrt(N)
    scoreB = np.array(scoreB, dtype=float) / np.sqrt(N)

    # Cross-overlap matrix in the same normalization as R = (1/N) V^T E
    M = (coeffA.T @ coeffB) / N

    # SVD gives canonical overlaps between the two empirical subspaces
    U, svals, Vt = np.linalg.svd(M, full_matrices=False)

    # Numerical safety
    svals = np.clip(svals, 0.0, 1.0)

    # Rotate coefficients and scores into the canonical bases
    coeffA_can = coeffA @ U
    coeffB_can = coeffB @ Vt.T
    scoreA_can = scoreA @ U
    scoreB_can = scoreB @ Vt.T

    # Optional sign alignment in canonical basis
    for k in range(K):
        if float(np.dot(coeffA_can[:, k], coeffB_can[:, k])) < 0:
            coeffB_can[:, k] *= -1.0
            scoreB_can[:, k] *= -1.0

    # Canonical overlap estimate: R_k ≈ sqrt(singular value of M)
    canonical_R = np.sqrt(svals)
    rho = 1.0 - canonical_R

    # Epsilon in the same canonical basis, with kernel normalization
    eps = np.zeros(K)
    kernel_integral_squared = np.sum((kernel/sum(kernel)) ** 2)
    kernel_integral_squared = 1

    for k in range(K):
        eps[k] = (
            (1.0 / (2.0 * T))
            * np.sum((scoreA_can[:, k] - scoreB_can[:, k]) ** 2)
            / kernel_integral_squared
        )

    # Subspace overlap estimator
    S12 = np.sum(M ** 2) / K
    mu_hat = np.sqrt(S12)

    return rho, eps, mu_hat


def first_k_values_sorted_unique(x, k):
    vals = sorted(set(int(v) for v in x))
    return vals[: min(k, len(vals))]


def nested_neuron_ids_for_target(base_order, pre_ids, D_pre, Ncur):
    """
    Build nested neuron ids for D_pre animals:
    start from pre_ids, then add extras in round-robin from the remaining pool.
    """
    pre_set = set(pre_ids.tolist())
    pools = []
    for a in range(D_pre):
        pools.append([x for x in base_order[a] if x not in pre_set])

    needed = max(0, int(Ncur) - len(pre_ids))
    extra = []
    pools_copy = [p[:] for p in pools]
    while needed > 0 and sum(len(p) for p in pools_copy) > 0:
        for a in range(D_pre):
            if needed == 0:
                break
            if pools_copy[a]:
                extra.append(pools_copy[a].pop(0))
                needed -= 1

    return np.r_[pre_ids, np.array(extra, dtype=int)]


def neuron_ids_for_base_count(base_order, D_pre, Nbase):
    """
    For a base neuron fit on the neurons axis:
    choose the first Nbase nested neurons from the D_pre animals, starting from zero.
    """
    pools = [base_order[a][:] for a in range(D_pre)]
    needed = int(Nbase)
    chosen = []
    pools_copy = [p[:] for p in pools]
    while needed > 0 and sum(len(p) for p in pools_copy) > 0:
        for a in range(D_pre):
            if needed == 0:
                break
            if pools_copy[a]:
                chosen.append(pools_copy[a].pop(0))
                needed -= 1
    return np.array(chosen, dtype=int)


def animal_ids_for_base_count(base_order, capacities, Dbase, n_pre_per_animal):
    """
    For a base animal fit on the animals axis:
    take the first Dbase animals and the fixed number of neurons per animal (clipped).
    """
    idxs = []
    for a in range(int(Dbase)):
        take = min(int(n_pre_per_animal), capacities[a])
        idxs.extend(base_order[a][:take])
    return np.array(idxs, dtype=int)


def axis_base_dir(axis_dir, axis_kind, base_value):
    axis_dir = Path(axis_dir)

    if axis_kind == "trials":
        return axis_dir / f"base_trials_{int(base_value)}"
    elif axis_kind == "neurons":
        return axis_dir / f"base_neurons_{int(base_value)}"
    elif axis_kind == "animals":
        return axis_dir / f"base_animals_{int(base_value)}"
    else:
        raise ValueError(f"Unknown axis_kind={axis_kind}")


def axis_pot_paths(axis_dir, axis_kind, base_value, K):
    base_dir = axis_base_dir(axis_dir, axis_kind, base_value)
    paths = [base_dir / f"pot_pc{k}.npz" for k in range(1, K + 1)]
    return base_dir, paths


def all_saved_pots_exist(axis_dir, axis_kind, base_value, K):
    _, paths = axis_pot_paths(axis_dir, axis_kind, base_value, K)
    return all(p.exists() for p in paths)


def save_pots_for_axis(axis_dir, axis_kind, base_value, pots):
    base_dir, paths = axis_pot_paths(axis_dir, axis_kind, base_value, len(pots))
    base_dir.mkdir(parents=True, exist_ok=True)
    for pot, path in zip(pots, paths):
        pot.save_as_npz(str(path))


def load_pots_for_axis(axis_dir, axis_kind, base_value, K):
    _, paths = axis_pot_paths(axis_dir, axis_kind, base_value, K)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing saved potentials for axis={axis_kind}, base={base_value}: {missing}"
        )
    return [Potential.from_npz(str(p)) for p in paths]


# ==================== Core ====================


def load_and_prep():
    data = np.load(DATA_PATH).astype(float)  # [trials, T, N]
    G = np.load(G_PATH).astype(int)          # [D, N, N]
    n_trials, T, N = data.shape
    D = G.shape[0]

    print("Loaded raw data:", data.shape)
    print("Loaded G:", G.shape)
    print("Firing rates (trial means):", np.mean(data, axis=1, keepdims=False))

    # Normalize by std of trial-average trace per neuron; guard against zero
    scale = np.std(np.mean(data, axis=0, keepdims=True), axis=1, keepdims=True)
    scale = np.where(scale == 0, 1.0, scale)
    data = data / scale

    # Always define kernel because empirical epsilon uses it
    kernel = generate_periodic_exponential_kernel(T, TAU_SIGMA)

    if SMOOTH_BEFORE_SPLIT:
        data = convolve_data(data, kernel)

    return data, G, kernel, n_trials, T, N, D


def permute_trials(full_data):
    """Single global trial permutation (for defining 'available trials')."""
    n_trials = full_data.shape[0]
    p_trials = RNG.permutation(n_trials)
    X = full_data[p_trials]
    return X, p_trials


def permute_animals(G):
    """Single global animal permutation."""
    n_animals = G.shape[0]
    p_animals = RNG.permutation(n_animals)
    G_new = G[p_animals]
    return G_new, p_animals

def permute_neurons_within_animals(full_data, G):
    """Permute neurons within each animal and update data/G consistently."""
    D, N, _ = G.shape
    animal_idx = np.argmax(np.stack([np.diag(G[d]) for d in range(D)]), axis=0)

    p = np.arange(N)
    for a in range(D):
        idx = np.where(animal_idx == a)[0]
        p[idx] = RNG.permutation(idx)

    return full_data[:, :, p]


def estimate_signal_variance_std_per_pc(pots, n_steps, n_samples):
    stds = []
    for k in range(len(pots)):
        s = pots[k].estimate_signal_variance_uncertainty(
            n_steps=n_steps, n_samples=n_samples
        )
        s = np.asarray(s).reshape(-1)
        stds.append(float(s[0]))
    return np.array(stds)


def build_grids_and_prelim(G, n_trials, N):
    """
    Build:
      - D_grid = [1..D_total]
      - trials_grid (auto if None)
      - available per-animal neurons (N_PRE_PER_ANIMAL or auto)
      - neurons_grid
      - base_order per animal
      - pre_ids for the smallest available dataset (D=1 animal)
    """
    D_total = G.shape[0]
    _, animal_groups = infer_animals_from_G(G)
    capacities = [len(g) for g in animal_groups]

    # D grid
    D_grid = list(range(1, D_total + 1))
    D_pre = 1  # smallest animal count

    # per-animal base order
    base_order = [RNG.permutation(list(g)).tolist() for g in animal_groups]

    # trials grid
    if TRIALS_GRID is None:
        low = max(3, n_trials // 3)
        high = n_trials // 2
        steps = min(5, max(1, high - low + 1))
        trials_grid = np.linspace(low, high, steps, dtype=int).tolist()
    else:
        trials_grid = list(map(int, TRIALS_GRID))
    trials_grid = sorted(set(trials_grid))

    # available per-animal neurons
    if N_PRE_PER_ANIMAL is None:
        n_pre_per_animal = max(8, capacities[0] // 3)
    else:
        n_pre_per_animal = int(N_PRE_PER_ANIMAL)
    n_pre_per_animal = min(n_pre_per_animal, capacities[0])

    # smallest available neuron ids: first animal only
    pre_ids = []
    for a in range(D_pre):
        take = min(n_pre_per_animal, capacities[a])
        pre_ids += base_order[a][:take]
    pre_ids = np.array(pre_ids, dtype=int)
    N_pre = len(pre_ids)

    # neurons grid (D fixed = 1)
    cap_Dpre = sum(capacities[:D_pre])
    steps = max(2, int(NEURONS_GRID_STEPS))
    neurons_grid = list(np.linspace(N_pre, cap_Dpre, steps, dtype=int))
    neurons_grid[0] = N_pre
    neurons_grid = sorted(set(neurons_grid))

    print("D_grid:", D_grid)
    print("trials_grid:", trials_grid)
    print("neurons_grid:", neurons_grid)

    return (
        D_grid,
        trials_grid,
        D_pre,
        n_pre_per_animal,
        neurons_grid,
        base_order,
        pre_ids,
        capacities,
    )


# ----------------- Three-series theory fits -----------------


def fit_pots_on_trials_base(data_perm, G, pre_ids, base_trials, K):
    avail_trials = np.arange(int(base_trials), dtype=int)
    data_fit = data_perm[avail_trials][:, :, pre_ids]
    G_fit = build_subG(G, pre_ids, order_animals=[0])

    pots, _ = fit_statistics_from_dataset_diagonal(
        data_fit, K, G_fit, TAU_SIGMA, mode=MODE, gamma=GAMMA
    )
    return pots, data_fit, G_fit


def fit_pots_on_neurons_base(data_perm, G, base_order, D_pre, pre_trials, Nbase, K):
    idxs = neuron_ids_for_base_count(base_order, D_pre, Nbase)
    avail_trials = np.arange(int(pre_trials), dtype=int)
    data_fit = data_perm[avail_trials][:, :, idxs]
    G_fit = build_subG(G, idxs, order_animals=list(range(D_pre)))

    pots, _ = fit_statistics_from_dataset_diagonal(
        data_fit, K, G_fit, TAU_SIGMA, mode=MODE, gamma=GAMMA
    )
    return pots, idxs, data_fit, G_fit


def fit_pots_on_animals_base(
    data_perm, G, base_order, capacities, base_D, pre_trials, n_pre_per_animal, K
):
    idxs = animal_ids_for_base_count(
        base_order, capacities, base_D, n_pre_per_animal
    )
    avail_trials = np.arange(int(pre_trials), dtype=int)
    data_fit = data_perm[avail_trials][:, :, idxs]
    G_fit = build_subG(G, idxs, order_animals=list(range(int(base_D))))

    pots, _ = fit_statistics_from_dataset_diagonal(
        data_fit, K, G_fit, TAU_SIGMA, mode=MODE, gamma=GAMMA
    )
    return pots, idxs, data_fit, G_fit


def predict_trials_axis_three_series(
    data_perm,
    G,
    pre_ids,
    K,
    trials_grid,
    base_trials_series,
    save_pots_dir=None,
    reuse_saved_pots=True,
):
    """
    Returns dict with arrays of shape (S_trials, L_trials, K).
    """
    S = len(base_trials_series)
    L = len(trials_grid)

    mean_rho = np.full((S, L, K), np.nan)
    eps = np.full((S, L, K), np.nan)
    mean_rho_std = np.full((S, L, K), np.nan)
    eps_std = np.full((S, L, K), np.nan)
    sigvar_std_all = np.full((S, K), np.nan)

    for s, base_trials in enumerate(base_trials_series):
        print(f"[theory:trials] series {s+1}/{S}, base_trials={base_trials}")

        if (
            reuse_saved_pots
            and save_pots_dir is not None
            and all_saved_pots_exist(save_pots_dir, "trials", base_trials, K)
        ):
            print(f"[theory:trials] loading saved potentials for base_trials={base_trials}")
            pots = load_pots_for_axis(save_pots_dir, "trials", base_trials, K)
        else:
            print(f"[theory:trials] fitting potentials for base_trials={base_trials}")
            pots, data_fit, G_fit = fit_pots_on_trials_base(
                data_perm, G, pre_ids, base_trials, K
            )

            if save_pots_dir is not None:
                save_pots_for_axis(save_pots_dir, "trials", base_trials, pots)

        sigvar_std = estimate_signal_variance_std_per_pc(
            pots, SIGVAR_N_STEPS, SIGVAR_N_SAMPLES
        )
        sigvar_std_all[s] = sigvar_std

        for i, tcur in enumerate(trials_grid):
            if int(tcur) < int(base_trials):
                continue

            for k in range(K):
                pot_xt = extrapolate_potential(
                    original=pots[k],
                    new_trials=int(tcur),
                    existing_number_of_trials=int(base_trials),
                    mode=MODE,
                    random_state=SEED + 10_000 * s + 100 * k + i,
                )
                pred = make_predictions(
                    pot_xt,
                    predict_errorbars=True,
                    var_std_list=np.array([float(sigvar_std[k])]),
                )
                mean_rho[s, i, k] = float(np.mean(np.asarray(pred["rho"])[:, 0, 0]))
                eps[s, i, k] = float(np.asarray(pred["epsilon"]).reshape(-1)[0])
                mean_rho_std[s, i, k] = float(
                    np.asarray(pred["mean_rho_std"]).reshape(-1)[0]
                )
                eps_std[s, i, k] = float(
                    np.asarray(pred.get("epsilon_std", 0.0)).reshape(-1)[0]
                )

    return {
        "mean_rho": mean_rho,
        "epsilon": eps,
        "mean_rho_std": mean_rho_std,
        "epsilon_std": eps_std,
        "signal_variance_std": sigvar_std_all,
    }


def predict_neurons_axis_three_series(
    data_perm,
    G,
    base_order,
    D_pre,
    pre_trials,
    K,
    neurons_grid,
    base_neurons_series,
    save_pots_dir=None,
    reuse_saved_pots=True,
):
    """
    Returns dict with arrays of shape (S_neurons, L_neurons, K).
    """
    S = len(base_neurons_series)
    L = len(neurons_grid)

    mean_rho = np.full((S, L, K), np.nan)
    eps = np.full((S, L, K), np.nan)
    mean_rho_std = np.full((S, L, K), np.nan)
    eps_std = np.full((S, L, K), np.nan)
    sigvar_std_all = np.full((S, K), np.nan)

    for s, Nbase in enumerate(base_neurons_series):
        print(f"[theory:neurons] series {s+1}/{S}, base_neurons={Nbase}")

        if (
            reuse_saved_pots
            and save_pots_dir is not None
            and all_saved_pots_exist(save_pots_dir, "neurons", Nbase, K)
        ):
            print(f"[theory:neurons] loading saved potentials for base_neurons={Nbase}")
            pots = load_pots_for_axis(save_pots_dir, "neurons", Nbase, K)
        else:
            print(f"[theory:neurons] fitting potentials for base_neurons={Nbase}")
            pots, base_ids, data_fit, G_fit = fit_pots_on_neurons_base(
                data_perm, G, base_order, D_pre, pre_trials, Nbase, K
            )

            if save_pots_dir is not None:
                save_pots_for_axis(save_pots_dir, "neurons", Nbase, pots)

        sigvar_std = estimate_signal_variance_std_per_pc(
            pots, SIGVAR_N_STEPS, SIGVAR_N_SAMPLES
        )
        sigvar_std_all[s] = sigvar_std

        # Reconstruct the deterministic base ids corresponding to this Nbase
        base_ids = neuron_ids_for_base_count(base_order, D_pre, Nbase)
        base_set = set(base_ids.tolist())
        pools = []
        for a in range(D_pre):
            pools.append([x for x in base_order[a] if x not in base_set])

        for i, Ncur in enumerate(neurons_grid):
            if int(Ncur) < int(Nbase):
                continue

            needed = max(0, int(Ncur) - len(base_ids))
            extra = []
            pools_copy = [p[:] for p in pools]
            while needed > 0 and sum(len(p) for p in pools_copy) > 0:
                for a in range(D_pre):
                    if needed == 0:
                        break
                    if pools_copy[a]:
                        extra.append(pools_copy[a].pop(0))
                        needed -= 1

            idxs = np.r_[base_ids, np.array(extra, dtype=int)]
            G_sub = build_subG(G, idxs, order_animals=list(range(D_pre)))

            for k in range(K):
                pot_xt = extrapolate_potential(
                    original=pots[k],
                    new_neurons=len(idxs),
                    new_G=G_sub,
                    mode=MODE,
                    random_state=SEED + 20_000 * s + 100 * k + i,
                )
                pred = make_predictions(
                    pot_xt,
                    predict_errorbars=True,
                    var_std_list=np.array([float(sigvar_std[k])]),
                )
                mean_rho[s, i, k] = float(np.mean(np.asarray(pred["rho"])[:, 0, 0]))
                eps[s, i, k] = float(np.asarray(pred["epsilon"]).reshape(-1)[0])
                mean_rho_std[s, i, k] = float(
                    np.asarray(pred["mean_rho_std"]).reshape(-1)[0]
                )
                eps_std[s, i, k] = float(
                    np.asarray(pred.get("epsilon_std", 0.0)).reshape(-1)[0]
                )

    return {
        "mean_rho": mean_rho,
        "epsilon": eps,
        "mean_rho_std": mean_rho_std,
        "epsilon_std": eps_std,
        "signal_variance_std": sigvar_std_all,
    }


def predict_animals_axis_three_series(
    data_perm,
    G,
    base_order,
    capacities,
    pre_trials,
    n_pre_per_animal,
    K,
    D_grid,
    base_animals_series,
    save_pots_dir=None,
    reuse_saved_pots=True,
):
    """
    Returns dict with arrays of shape (S_animals, L_animals, K).
    """
    S = len(base_animals_series)
    L = len(D_grid)

    mean_rho = np.full((S, L, K), np.nan)
    eps = np.full((S, L, K), np.nan)
    mean_rho_std = np.full((S, L, K), np.nan)
    eps_std = np.full((S, L, K), np.nan)
    sigvar_std_all = np.full((S, K), np.nan)

    for s, Dbase in enumerate(base_animals_series):
        print(f"[theory:animals] series {s+1}/{S}, base_animals={Dbase}")

        if (
            reuse_saved_pots
            and save_pots_dir is not None
            and all_saved_pots_exist(save_pots_dir, "animals", Dbase, K)
        ):
            print(f"[theory:animals] loading saved potentials for base_animals={Dbase}")
            pots = load_pots_for_axis(save_pots_dir, "animals", Dbase, K)
        else:
            print(f"[theory:animals] fitting potentials for base_animals={Dbase}")
            pots, base_ids, data_fit, G_fit = fit_pots_on_animals_base(
                data_perm,
                G,
                base_order,
                capacities,
                Dbase,
                pre_trials,
                n_pre_per_animal,
                K,
            )

            if save_pots_dir is not None:
                save_pots_for_axis(save_pots_dir, "animals", Dbase, pots)

        sigvar_std = estimate_signal_variance_std_per_pc(
            pots, SIGVAR_N_STEPS, SIGVAR_N_SAMPLES
        )
        sigvar_std_all[s] = sigvar_std

        for i, Dcur in enumerate(D_grid):
            if int(Dcur) < int(Dbase):
                continue

            idxs = []
            for a in range(int(Dcur)):
                take = min(int(n_pre_per_animal), capacities[a])
                idxs.extend(base_order[a][:take])
            idxs = np.array(idxs, dtype=int)
            G_sub = build_subG(G, idxs, order_animals=list(range(int(Dcur))))

            for k in range(K):
                pot_xt = extrapolate_potential(
                    original=pots[k],
                    new_neurons=len(idxs),
                    new_G=G_sub,
                    mode=MODE,
                    random_state=SEED + 30_000 * s + 100 * k + i,
                )
                pred = make_predictions(
                    pot_xt,
                    predict_errorbars=True,
                    var_std_list=np.array([float(sigvar_std[k])]),
                )
                mean_rho[s, i, k] = float(np.mean(np.asarray(pred["rho"])[:, 0, 0]))
                eps[s, i, k] = float(np.asarray(pred["epsilon"]).reshape(-1)[0])
                mean_rho_std[s, i, k] = float(
                    np.asarray(pred["mean_rho_std"]).reshape(-1)[0]
                )
                eps_std[s, i, k] = float(
                    np.asarray(pred.get("epsilon_std", 0.0)).reshape(-1)[0]
                )

    return {
        "mean_rho": mean_rho,
        "epsilon": eps,
        "mean_rho_std": mean_rho_std,
        "epsilon_std": eps_std,
        "signal_variance_std": sigvar_std_all,
    }


# -------------- Empirical (resampling, no fixed A/B) --------------


def empirical_trials_with_reps_resample(
    full_data_perm, K, pre_trials, trials_grid, pre_ids, kernel, n_reps, seed
):
    """
    A: contains 'available' trials (0..pre_trials-1 in the permuted ordering) and grows nested.
    B: fresh random sample each size & repetition (with replacement to keep variability at max size).
    Neurons are fixed to pre_ids on both sides.
    """
    n_trials = full_data_perm.shape[0]
    all_trials = np.arange(n_trials)
    avail_trials = np.arange(pre_trials, dtype=int)

    L = len(trials_grid)
    rho_all = np.zeros((n_reps, L, K))
    eps_all = np.zeros((n_reps, L, K))
    mu_all = np.zeros((n_reps, L))

    rng0 = np.random.default_rng(seed)

    for r in range(n_reps):
        rng = np.random.default_rng(rng0.integers(0, 2**31 - 1))
        rest = np.arange(pre_trials, n_trials)
        rng.shuffle(rest)

        for i, tcur in enumerate(trials_grid):
            tcur = int(tcur)
            add = max(0, tcur - pre_trials)
            A_sel = np.r_[avail_trials, rest[:add]]
            B_sel = rng.choice(all_trials, size=tcur, replace=True)

            A_sample = full_data_perm[A_sel][:, :, pre_ids]
            B_sample = full_data_perm[B_sel][:, :, pre_ids]
            rho, eps, mu = _empirical_two_groups(A_sample, B_sample, K, kernel, MODE)
            rho_all[r, i] = rho
            eps_all[r, i] = eps
            mu_all[r, i] = mu

    return {
        "mu_mean": np.sqrt(np.mean(mu_all**2, axis=0)),
        "mu_std": mu_all.std(0, ddof=1) if n_reps > 1 else np.zeros_like(mu_all.mean(0)),
        "mu_all": mu_all,
        "mean_rho_mean": 1 - np.sqrt(((1 - rho_all) ** 2).mean(0)),
        "epsilon_mean": eps_all.mean(0),
        "mean_rho_std": rho_all.std(0, ddof=1) if n_reps > 1 else np.zeros_like(rho_all.mean(0)),
        "epsilon_std": eps_all.std(0, ddof=1) if n_reps > 1 else np.zeros_like(eps_all.mean(0)),
        "mean_rho_all": rho_all,
        "epsilon_all": eps_all,
    }


def empirical_neurons_with_reps_resample(
    full_data_perm,
    G,
    K,
    pre_trials,
    D_pre,
    base_order,
    pre_ids,
    neurons_grid,
    kernel,
    n_reps,
    seed,
):
    """
    Trials: A uses fixed 'available' trials (first pre_trials in permuted order).
            B uses fresh random pre_trials trials each repetition (with replacement).
    Neurons: nested within a repetition; A and B use the SAME neuron IDs for a given size.
    """
    n_trials = full_data_perm.shape[0]
    all_trials = np.arange(n_trials)
    avail_trials = np.arange(pre_trials, dtype=int)

    L = len(neurons_grid)
    rho_all = np.zeros((n_reps, L, K))
    eps_all = np.zeros((n_reps, L, K))
    mu_all = np.zeros((n_reps, L))

    pre_set = set(pre_ids.tolist())
    pools0 = []
    for a in range(D_pre):
        pools0.append([x for x in base_order[a] if x not in pre_set])

    rng0 = np.random.default_rng(seed)

    for r in range(n_reps):
        rng = np.random.default_rng(rng0.integers(0, 2**31 - 1))
        B_sel = rng.choice(all_trials, size=pre_trials, replace=True)

        for i, Ncur in enumerate(neurons_grid):
            Ncur = int(Ncur)
            needed = max(0, Ncur - len(pre_ids))
            extra = []
            pools_copy = [p[:] for p in pools0]
            while needed > 0 and sum(len(p) for p in pools_copy) > 0:
                for a in range(D_pre):
                    if needed == 0:
                        break
                    if pools_copy[a]:
                        extra.append(pools_copy[a].pop(0))
                        needed -= 1
            idxs = np.r_[pre_ids, np.array(extra, dtype=int)]

            A_sample = full_data_perm[avail_trials][:, :, idxs]
            B_sample = full_data_perm[B_sel][:, :, idxs]
            rho, eps, mu = _empirical_two_groups(A_sample, B_sample, K, kernel, MODE)
            rho_all[r, i] = rho
            eps_all[r, i] = eps
            mu_all[r, i] = mu

    return {
        "mu_mean": np.sqrt(np.mean(mu_all**2, axis=0)),
        "mu_std": mu_all.std(0, ddof=1) if n_reps > 1 else np.zeros_like(mu_all.mean(0)),
        "mu_all": mu_all,
        "mean_rho_mean": 1 - np.sqrt(((1 - rho_all) ** 2).mean(0)),
        "epsilon_mean": eps_all.mean(0),
        "mean_rho_std": rho_all.std(0, ddof=1) if n_reps > 1 else np.zeros_like(rho_all.mean(0)),
        "epsilon_std": eps_all.std(0, ddof=1) if n_reps > 1 else np.zeros_like(eps_all.mean(0)),
        "mean_rho_all": rho_all,
        "epsilon_all": eps_all,
    }


def empirical_animals_with_reps_resample(
    full_data_perm, G, K, pre_trials, D_grid, n_pre_per_animal, base_order, kernel, n_reps, seed
):
    """
    Trials: A fixed to 'available' trials (first pre_trials).
            B freshly sampled each repetition (with replacement).
    Animals: nested by adding animals; per-animal neurons fixed to n_pre_per_animal (clipped).
    """
    n_trials = full_data_perm.shape[0]
    all_trials = np.arange(n_trials)
    avail_trials = np.arange(pre_trials, dtype=int)

    _, animal_groups = infer_animals_from_G(G)
    capacities = [len(g) for g in animal_groups]

    L = len(D_grid)
    rho_all = np.zeros((n_reps, L, K))
    eps_all = np.zeros((n_reps, L, K))
    mu_all = np.zeros((n_reps, L))

    rng0 = np.random.default_rng(seed)

    for r in range(n_reps):
        rng = np.random.default_rng(rng0.integers(0, 2**31 - 1))
        B_sel = rng.choice(all_trials, size=pre_trials, replace=True)

        for i, Dcur in enumerate(D_grid):
            idxs = []
            for a in range(Dcur):
                take = min(n_pre_per_animal, capacities[a])
                idxs.extend(base_order[a][:take])
            idxs = np.array(idxs, dtype=int)

            A_sample = full_data_perm[avail_trials][:, :, idxs]
            B_sample = full_data_perm[B_sel][:, :, idxs]
            rho, eps, mu = _empirical_two_groups(A_sample, B_sample, K, kernel, MODE)
            rho_all[r, i] = rho
            eps_all[r, i] = eps
            mu_all[r, i] = mu

    return {
        "mu_mean": np.sqrt(np.mean(mu_all**2, axis=0)),
        "mu_std": mu_all.std(0, ddof=1) if n_reps > 1 else np.zeros_like(mu_all.mean(0)),
        "mu_all": mu_all,
        "mean_rho_mean": 1 - np.sqrt(((1 - rho_all) ** 2).mean(0)),
        "epsilon_mean": eps_all.mean(0),
        "mean_rho_std": rho_all.std(0, ddof=1) if n_reps > 1 else np.zeros_like(rho_all.mean(0)),
        "epsilon_std": eps_all.std(0, ddof=1) if n_reps > 1 else np.zeros_like(eps_all.mean(0)),
        "mean_rho_all": rho_all,
        "epsilon_all": eps_all,
    }


# ==================== MAIN ====================


def main():
    full_data, G, kernel, n_trials, T, N, D_total = load_and_prep()

    reduced = reduce_to_2d(full_data, mode=MODE)
    print("Reduced shape:", reduced.shape)
    print("Neurons per animal:", np.einsum("dii->d", G))
    print(f"Loaded: trials={n_trials}, T={T}, N={N}, D={D_total}")

    # One global trial permutation (defines the nested available trials)
    data_perm, _ = permute_trials(full_data)
    data_perm = permute_neurons_within_animals(full_data,G)
    G, _ = permute_animals(G)

    # Build grids + smallest available slice
    (
        D_grid,
        trials_grid,
        D_pre,
        n_pre_per_animal,
        neurons_grid,
        base_order,
        pre_ids,
        capacities,
    ) = build_grids_and_prelim(G, n_trials, N)

    # The smallest fit settings are still used for:
    # - global K selection
    # - empirical curves
    pre_trials = int(trials_grid[0])

    print(
        f"Smallest available data: D_pre={D_pre}, per_animal={n_pre_per_animal}, "
        f"pre_trials={pre_trials}, pre_neurons={len(pre_ids)}"
    )

    # Determine K once from the smallest available slice, then keep it fixed
    avail_trials = np.arange(pre_trials, dtype=int)
    data_fit_smallest = data_perm[avail_trials][:, :, pre_ids]
    print(
        f"Estimating K on smallest available slice: trials={pre_trials}, "
        f"neurons={len(pre_ids)}, animals={D_pre}"
    )
    #K = determine_dimensionality(data_fit_smallest, MODE, plot=True)
    #print("K =", K)
    K=2

    save_np("K", np.array(K, dtype=int))
    

    # Save grids
    save_np("trials_grid", np.array(trials_grid, dtype=int))
    save_np("neurons_grid", np.array(neurons_grid, dtype=int))
    save_np("D_grid", np.array(D_grid, dtype=int))

    # Base series = three lowest values of each grid
    base_trials_series = first_k_values_sorted_unique(trials_grid, N_BASE_SERIES)
    base_neurons_series = first_k_values_sorted_unique(neurons_grid, N_BASE_SERIES)
    base_animals_series = first_k_values_sorted_unique(D_grid, N_BASE_SERIES)

    save_np("base_trials_series", np.array(base_trials_series, dtype=int))
    save_np("base_neurons_series", np.array(base_neurons_series, dtype=int))
    save_np("base_animals_series", np.array(base_animals_series, dtype=int))

    print("base_trials_series:", base_trials_series)
    print("base_neurons_series:", base_neurons_series)
    print("base_animals_series:", base_animals_series)

    # Save base order for reproducibility
    base_order_obj = np.array([np.array(x, dtype=int) for x in base_order], dtype=object)
    np.save(OUT / "base_order.npy", base_order_obj, allow_pickle=True)
    save_np("pre_ids", pre_ids)


    # -------- Prediction / theory side: three-series extrapolation
    pots_root = OUT / "pots_initial_fit_three_series"
    pots_root.mkdir(exist_ok=True)

    print("Predicting | trials axis (three base series) ...")
    pred_trials = predict_trials_axis_three_series(
        data_perm=data_perm,
        G=G,
        pre_ids=pre_ids,
        K=K,
        trials_grid=trials_grid,
        base_trials_series=base_trials_series,
        save_pots_dir=pots_root / "trials_axis",
        reuse_saved_pots=REUSE_SAVED_POTENTIALS,
    )

    for name, bundle in [("trials", pred_trials)]:
        save_np(f"pred_{name}_mean_rho", bundle["mean_rho"])
        save_np(f"pred_{name}_epsilon", bundle["epsilon"])
        save_np(f"pred_{name}_mean_rho_std", bundle["mean_rho_std"])
        save_np(f"pred_{name}_epsilon_std", bundle["epsilon_std"])
        save_np(f"pred_{name}_signal_variance_std", bundle["signal_variance_std"])

    print("Predicting | neurons axis (three base series) ...")
    pred_neurons = predict_neurons_axis_three_series(
        data_perm=data_perm,
        G=G,
        base_order=base_order,
        D_pre=D_pre,
        pre_trials=pre_trials,
        K=K,
        neurons_grid=neurons_grid,
        base_neurons_series=base_neurons_series,
        save_pots_dir=pots_root / "neurons_axis",
        reuse_saved_pots=REUSE_SAVED_POTENTIALS,
    )

    for name, bundle in [("neurons", pred_neurons)]:
        save_np(f"pred_{name}_mean_rho", bundle["mean_rho"])
        save_np(f"pred_{name}_epsilon", bundle["epsilon"])
        save_np(f"pred_{name}_mean_rho_std", bundle["mean_rho_std"])
        save_np(f"pred_{name}_epsilon_std", bundle["epsilon_std"])
        save_np(f"pred_{name}_signal_variance_std", bundle["signal_variance_std"])
    
    print("Predicting | animals axis (three base series) ...")
    pred_animals = predict_animals_axis_three_series(
        data_perm=data_perm,
        G=G,
        base_order=base_order,
        capacities=capacities,
        pre_trials=pre_trials,
        n_pre_per_animal=n_pre_per_animal,
        K=K,
        D_grid=D_grid,
        base_animals_series=base_animals_series,
        save_pots_dir=pots_root / "animals_axis",
        reuse_saved_pots=REUSE_SAVED_POTENTIALS,
    )

    for name, bundle in [("animals", pred_animals)]:
        save_np(f"pred_{name}_mean_rho", bundle["mean_rho"])
        save_np(f"pred_{name}_epsilon", bundle["epsilon"])
        save_np(f"pred_{name}_mean_rho_std", bundle["mean_rho_std"])
        save_np(f"pred_{name}_epsilon_std", bundle["epsilon_std"])
        save_np(f"pred_{name}_signal_variance_std", bundle["signal_variance_std"])


    # -------- Empirical side
    print(f"Empirical (resampling) | repetitions = {N_REPETITIONS}")

    emp_trials = empirical_trials_with_reps_resample(
        data_perm,
        K,
        pre_trials,
        trials_grid,
        pre_ids,
        kernel,
        N_REPETITIONS,
        seed=SEED + 1000,
    )
    emp_neurons = empirical_neurons_with_reps_resample(
        data_perm,
        G,
        K,
        pre_trials,
        D_pre,
        base_order,
        pre_ids,
        neurons_grid,
        kernel,
        N_REPETITIONS,
        seed=SEED + 2000,
    )
    emp_animals = empirical_animals_with_reps_resample(
        data_perm,
        G,
        K,
        pre_trials,
        D_grid,
        n_pre_per_animal,
        base_order,
        kernel,
        N_REPETITIONS,
        seed=SEED + 3000,
    )

    for name, bundle in [
        ("trials", emp_trials),
        ("neurons", emp_neurons),
        ("animals", emp_animals),
    ]:
        save_np(f"emp_{name}_mu_mean", bundle["mu_mean"])
        save_np(f"emp_{name}_mu_std", bundle["mu_std"])
        save_np(f"emp_{name}_mu_all", bundle["mu_all"])

        save_np(f"emp_{name}_mean_rho_mean", bundle["mean_rho_mean"])
        save_np(f"emp_{name}_epsilon_mean", bundle["epsilon_mean"])
        save_np(f"emp_{name}_mean_rho_std", bundle["mean_rho_std"])
        save_np(f"emp_{name}_epsilon_std", bundle["epsilon_std"])
        save_np(f"emp_{name}_mean_rho_all", bundle["mean_rho_all"])
        save_np(f"emp_{name}_epsilon_all", bundle["epsilon_all"])

    print("Done. Results saved to", OUT.resolve())
    print("Shapes:")
    print("  pred_trials_mean_rho   :", pred_trials["mean_rho"].shape)
    print("  pred_neurons_mean_rho  :", pred_neurons["mean_rho"].shape)
    print("  pred_animals_mean_rho  :", pred_animals["mean_rho"].shape)
    print("  emp_trials_mean_rho    :", emp_trials["mean_rho_mean"].shape)
    print("  emp_neurons_mean_rho   :", emp_neurons["mean_rho_mean"].shape)
    print("  emp_animals_mean_rho   :", emp_animals["mean_rho_mean"].shape)


if __name__ == "__main__":
    main()