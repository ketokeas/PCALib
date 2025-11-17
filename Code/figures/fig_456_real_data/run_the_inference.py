#!/usr/bin/env python3
# --------------------------------------------------------------------
# Figures 4–6 pipeline (single dataset; three axes) with unified "available data"
# - Resampling-based empirical estimates (no fixed A/B split):
#     * A includes the "available" trials and grows nested across sizes.
#     * B is freshly (re)sampled each repetition (with replacement when needed).
# - Inferred per-PC Potentials from the initial fit are saved as NPZs.
#
# Saves to: cached_results/.../real_gc_grid/
# --------------------------------------------------------------------

import os, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- project imports ---
sys.path.insert(0, os.path.relpath("../../"))
from pcalib.classes import Potential
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

# ====================== CONFIG ======================

# Dataset folder name (used in multiple paths)
DATASET_NAME = "2 Li et al (Svoboda lab)"

# Dataset
DATA_PATH = Path("../../../Data") / DATASET_NAME / "preformatted_data.npy"
G_PATH = Path("../../../Data") / DATASET_NAME / "G.npy"

MODE = "trial-averaged"  # or "trial-concatenated"
TAU_SIGMA = 2.0  # smoothing kernel width + used in fitting
GAMMA = 0.1  # step/damping for the diagonal fitter

SMOOTH_BEFORE_SPLIT = True

# Reproducibility
SEED = 2025
RNG = np.random.default_rng(SEED)

# Work factors
SIGVAR_N_STEPS = 30
SIGVAR_N_SAMPLES = 200
N_REPETITIONS = 50

# --- Grids you control ---
TRIALS_GRID = None  # e.g., [4, 6, 8, 10, 12]; default is auto
N_PRE_PER_ANIMAL = None  # per-animal neurons in available data; auto if None
NEURONS_GRID_STEPS = 5

# Output directory
OUT = Path("cached_results") / DATASET_NAME / "real_gc_grid"
OUT.mkdir(parents=True, exist_ok=True)

# ==================== Helpers ====================


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


def _empirical_two_groups(A_data, B_data, K, mode):
    """
    Empirical rho/epsilon between two datasets of the same shape: [n_trials_sel, T, N_sel],
    using your coeff/score rescaling and sign-alignment.
    """
    XA = reduce_to_2d(A_data, mode)
    XB = reduce_to_2d(B_data, mode)
    XA -= np.mean(XA, 0)
    XB -= np.mean(XB, 0)
    coeffA, scoreA, _ = PCA_matlab_like(XA)
    coeffB, scoreB, _ = PCA_matlab_like(XB)
    coeffA = coeffA[:, :K]
    scoreA = scoreA[:, :K]
    coeffB = coeffB[:, :K]
    scoreB = scoreB[:, :K]

    N = np.shape(coeffA)[0]
    T = np.shape(scoreA)[0]

    coeffA = np.array(coeffA) * np.sqrt(N)
    coeffB = np.array(coeffB) * np.sqrt(N)
    scoreA = np.array(scoreA) / np.sqrt(N)
    scoreB = np.array(scoreB) / np.sqrt(N)

    # sign alignment
    for k in range(K):
        if float(np.dot(coeffA[:, k], coeffB[:, k])) < 0:
            scoreB[:, k] *= -1.0
            coeffB[:, k] *= -1.0

    rho = np.zeros(K)
    eps = np.zeros(K)
    for k in range(K):
        rho[k] = 1 - np.sqrt(np.dot(coeffA[:, k], coeffB[:, k]) / N)
        eps[k] = 1 / (2 * T) * np.sum((scoreA[:, k] - scoreB[:, k]) ** 2)
    return rho, eps


# ==================== Core ====================


def load_and_prep():
    data = np.load(DATA_PATH).astype(float)  # [trials, T, N]
    G = np.load(G_PATH).astype(int)  # [D, N, N]
    n_trials, T, N = data.shape
    D = G.shape[0]

    plt.hist(np.sum(np.sum(data, axis=0), axis=0), 1000)
    plt.show()

    print("Firing rates:", np.mean(data, 1, keepdims=False))

    # Normalize by std of trial-average trace per neuron; guard against zero
    scale = np.std(np.mean(data, 0, keepdims=True), 1, keepdims=True)
    scale = np.where(scale == 0, 1.0, scale)
    data = data / scale

    if SMOOTH_BEFORE_SPLIT:
        kernel = generate_periodic_exponential_kernel(T, TAU_SIGMA)
        data = convolve_data(data, kernel)
    return data, G, n_trials, T, N, D


def permute_trials(full_data):
    """Single global trial permutation (for defining 'available trials')."""
    n_trials = full_data.shape[0]
    p_trials = RNG.permutation(n_trials)
    X = full_data[p_trials]
    return X, p_trials


def determine_K_safe(data_fit, mode):
    # Use your direct API
    K = determine_dimensionality(data_fit, mode)
    return int(K)


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
      - D_grid = [1..D_total], choose D_pre = 1
      - trials_grid (auto if None), pre_trials = trials_grid[0]
      - available per-animal neurons (N_PRE_PER_ANIMAL or auto), choose prelim ids from first D_pre animals
      - neurons_grid: grow total neurons for those D_pre animals (nested)
      - animals_grid (not saved): grow D, fixed per-animal count clipped to capacity
      - base_order per animal; pre_ids list
    """
    D_total = G.shape[0]
    _, animal_groups = infer_animals_from_G(G)
    capacities = [len(g) for g in animal_groups]

    # D grid and D_pre
    D_grid = list(range(1, D_total + 1))
    D_pre = D_grid[0]  # == 1 by construction

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
    pre_trials = int(trials_grid[0])

    # available per-animal neurons
    if N_PRE_PER_ANIMAL is None:
        n_pre_per_animal = max(8, capacities[0] // 3)
    else:
        n_pre_per_animal = int(N_PRE_PER_ANIMAL)
    n_pre_per_animal = min(n_pre_per_animal, capacities[0])

    # prelim neuron ids from first D_pre animals
    pre_ids = []
    for a in range(D_pre):
        take = min(n_pre_per_animal, capacities[a])
        pre_ids += base_order[a][:take]
    pre_ids = np.array(pre_ids, dtype=int)
    N_pre = len(pre_ids)

    # neurons grid (D fixed = D_pre)
    cap_Dpre = sum(capacities[:D_pre])
    steps = max(2, int(NEURONS_GRID_STEPS))
    neurons_grid = list(np.linspace(N_pre, cap_Dpre, steps, dtype=int))

    neurons_grid[0] = N_pre
    neurons_grid = sorted(set(neurons_grid))

    # animals grid (grow D; fixed per-animal count clipped)
    animals_grid = []
    for d in range(len(D_grid)):
        per = [min(n_pre_per_animal, capacities[a]) for a in range(d + 1)]
        animals_grid.append({"D": (d + 1), "neurons_per_animal": per})

    print("D grid", D_grid)
    print("Trials grid:", trials_grid)
    print("neurons_grid:", neurons_grid)

    return (
        D_grid,
        trials_grid,
        D_pre,
        n_pre_per_animal,
        pre_trials,
        neurons_grid,
        animals_grid,
        base_order,
        pre_ids,
        capacities,
    )


# ----------------- Predictions (direct API calls) -----------------


def predict_trials_axis(pots, pre_trials, trials_grid, sigvar_std):
    K = len(pots)
    L = len(trials_grid)
    mean_rho = np.zeros((L, K))
    eps = np.zeros((L, K))
    mean_rho_std = np.zeros((L, K))
    eps_std = np.zeros((L, K))
    for i, tcur in enumerate(trials_grid):
        for k in range(K):
            pot_xt = extrapolate_potential(
                original=pots[k],
                new_trials=int(tcur),
                existing_number_of_trials=int(pre_trials),
                mode=MODE,
                random_state=SEED + 10 * k + i,
            )
            pred = make_predictions(
                pot_xt,
                predict_errorbars=True,
                var_std_list=np.array([float(sigvar_std[k])]),
            )
            mean_rho[i, k] = float(np.mean(np.asarray(pred["rho"])[:, 0, 0]))
            eps[i, k] = float(np.asarray(pred["epsilon"]).reshape(-1)[0])
            mean_rho_std[i, k] = float(np.asarray(pred["mean_rho_std"].reshape(-1)[0]))
            eps_std[i, k] = float(
                np.asarray(pred.get("epsilon_std", 0.0)).reshape(-1)[0]
            )
    return {
        "mean_rho": mean_rho,
        "epsilon": eps,
        "mean_rho_std": mean_rho_std,
        "epsilon_std": eps_std,
    }


def predict_neurons_axis(pots, G, D_pre, base_order, pre_ids, neurons_grid, sigvar_std):
    K = len(pots)
    L = len(neurons_grid)
    mean_rho = np.zeros((L, K))
    eps = np.zeros((L, K))
    mean_rho_std = np.zeros((L, K))
    eps_std = np.zeros((L, K))

    pre_set = set(pre_ids.tolist())
    pools = []
    for a in range(D_pre):
        pools.append([x for x in base_order[a] if x not in pre_set])

    for i, Ncur in enumerate(neurons_grid):
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
        idxs = np.r_[pre_ids, np.array(extra, dtype=int)]
        G_sub = build_subG(G, idxs, order_animals=list(range(D_pre)))

        for k in range(K):
            pot_xt = extrapolate_potential(
                original=pots[k],
                new_neurons=len(idxs),
                new_G=G_sub,
                mode=MODE,
                random_state=SEED + 20 * k + i,
            )
            pred = make_predictions(
                pot_xt,
                predict_errorbars=True,
                var_std_list=np.array([float(sigvar_std[k])]),
            )
            mean_rho[i, k] = float(np.mean(np.asarray(pred["rho"])[:, 0, 0]))
            eps[i, k] = float(np.asarray(pred["epsilon"]).reshape(-1)[0])
            mean_rho_std[i, k] = float(np.asarray(pred["mean_rho_std"].reshape(-1)[0]))
            eps_std[i, k] = float(
                np.asarray(pred.get("epsilon_std", 0.0)).reshape(-1)[0]
            )

    return {
        "mean_rho": mean_rho,
        "epsilon": eps,
        "mean_rho_std": mean_rho_std,
        "epsilon_std": eps_std,
    }


def predict_animals_axis(pots, G, D_grid, n_pre_per_animal, base_order, sigvar_std):
    K = len(pots)
    L = len(D_grid)
    mean_rho = np.zeros((L, K))
    eps = np.zeros((L, K))
    mean_rho_std = np.zeros((L, K))
    eps_std = np.zeros((L, K))

    _, animal_groups = infer_animals_from_G(G)
    capacities = [len(g) for g in animal_groups]

    for i, Dcur in enumerate(D_grid):
        idxs = []
        for a in range(Dcur):
            n_a = min(n_pre_per_animal, capacities[a])
            idxs.extend(base_order[a][:n_a])
        idxs = np.array(idxs, dtype=int)
        G_sub = build_subG(G, idxs, order_animals=list(range(Dcur)))

        for k in range(K):
            pot_xt = extrapolate_potential(
                original=pots[k],
                new_neurons=len(idxs),
                new_G=G_sub,
                mode=MODE,
                random_state=SEED + 30 * k + i,
            )
            pred = make_predictions(
                pot_xt,
                predict_errorbars=True,
                var_std_list=np.array([float(sigvar_std[k])]),
            )
            mean_rho[i, k] = float(np.mean(np.asarray(pred["rho"])[:, 0, 0]))
            eps[i, k] = float(np.asarray(pred["epsilon"]).reshape(-1)[0])
            mean_rho_std[i, k] = float(np.asarray(pred["mean_rho_std"].reshape(-1)[0]))
            eps_std[i, k] = float(
                np.asarray(pred.get("epsilon_std", 0.0)).reshape(-1)[0]
            )

    return {
        "mean_rho": mean_rho,
        "epsilon": eps,
        "mean_rho_std": mean_rho_std,
        "epsilon_std": eps_std,
    }


# -------------- Empirical (resampling, no fixed A/B) --------------


def empirical_trials_with_reps_resample(
    full_data_perm, K, pre_trials, trials_grid, pre_ids, n_reps, seed
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

    rng0 = np.random.default_rng(seed)

    for r in range(n_reps):
        rng = np.random.default_rng(rng0.integers(0, 2**31 - 1))
        rest = np.arange(pre_trials, n_trials)  # trials not in the available set
        rng.shuffle(rest)

        for i, tcur in enumerate(trials_grid):
            tcur = int(tcur)
            add = max(0, tcur - pre_trials)
            # A: nested
            A_sel = np.r_[avail_trials, rest[:add]]
            # B: bootstrap (replacement) to keep variability even at boundary
            B_sel = rng.choice(all_trials, size=tcur, replace=True)

            A_sample = full_data_perm[A_sel][:, :, pre_ids]
            B_sample = full_data_perm[B_sel][:, :, pre_ids]
            rho, eps = _empirical_two_groups(A_sample, B_sample, K, MODE)
            rho_all[r, i] = rho
            eps_all[r, i] = eps

    return {
        "mean_rho_mean": rho_all.mean(0),
        "epsilon_mean": eps_all.mean(0),
        "mean_rho_std": rho_all.std(0, ddof=1)
        if n_reps > 1
        else np.zeros_like(rho_all.mean(0)),
        "epsilon_std": eps_all.std(0, ddof=1)
        if n_reps > 1
        else np.zeros_like(eps_all.mean(0)),
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

    # Pools per animal (excluding prelim)
    pre_set = set(pre_ids.tolist())
    pools0 = []
    for a in range(D_pre):
        pools0.append([x for x in base_order[a] if x not in pre_set])

    rng0 = np.random.default_rng(seed)

    for r in range(n_reps):
        rng = np.random.default_rng(rng0.integers(0, 2**31 - 1))
        # B trials: bootstrap each repetition
        B_sel = rng.choice(all_trials, size=pre_trials, replace=True)

        # per-rep neuron pools (keep nested across sizes)
        pools = [p[:] for p in pools0]

        for i, Ncur in enumerate(neurons_grid):
            Ncur = int(Ncur)
            needed = max(0, Ncur - len(pre_ids))
            extra = []
            pools_copy = [p[:] for p in pools]
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
            rho, eps = _empirical_two_groups(A_sample, B_sample, K, MODE)
            rho_all[r, i] = rho
            eps_all[r, i] = eps

    return {
        "mean_rho_mean": rho_all.mean(0),
        "epsilon_mean": eps_all.mean(0),
        "mean_rho_std": rho_all.std(0, ddof=1)
        if n_reps > 1
        else np.zeros_like(rho_all.mean(0)),
        "epsilon_std": eps_all.std(0, ddof=1)
        if n_reps > 1
        else np.zeros_like(eps_all.mean(0)),
        "mean_rho_all": rho_all,
        "epsilon_all": eps_all,
    }


def empirical_animals_with_reps_resample(
    full_data_perm, G, K, pre_trials, D_grid, n_pre_per_animal, base_order, n_reps, seed
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
            rho, eps = _empirical_two_groups(A_sample, B_sample, K, MODE)
            rho_all[r, i] = rho
            eps_all[r, i] = eps

    return {
        "mean_rho_mean": rho_all.mean(0),
        "epsilon_mean": eps_all.mean(0),
        "mean_rho_std": rho_all.std(0, ddof=1)
        if n_reps > 1
        else np.zeros_like(rho_all.mean(0)),
        "epsilon_std": eps_all.std(0, ddof=1)
        if n_reps > 1
        else np.zeros_like(eps_all.mean(0)),
        "mean_rho_all": rho_all,
        "epsilon_all": eps_all,
    }


# ==================== MAIN ====================


def main():
    full_data, G, n_trials, T, N, D_total = load_and_prep()

    reduced = reduce_to_2d(full_data[:, :, :], mode="trial-averaged")
    print("neurons per animal:", np.einsum("dii->d", G))
    print("N=", np.sum(G))

    print(f"Loaded: trials={n_trials}, T={T}, N={N}, D={D_total}")

    # Global trial permutation (defines the "available trials")
    data_perm, p_trials = permute_trials(full_data)
    np.save(OUT / "perm_trials.npy", p_trials)

    # Build grids + available slice
    (
        D_grid,
        trials_grid,
        D_pre,
        n_pre_per_animal,
        pre_trials,
        neurons_grid,
        animals_grid,
        base_order,
        pre_ids,
        capacities,
    ) = build_grids_and_prelim(G, n_trials, N)

    print(
        f"Available data: D_pre={D_pre}, per_animal={n_pre_per_animal}, "
        f"pre_trials={pre_trials}, pre_neurons={len(pre_ids)}"
    )

    # Fit on available slice (from the permuted data)
    avail_trials = np.arange(pre_trials, dtype=int)
    data_fit = data_perm[avail_trials][:, :, pre_ids]

    G_fit = build_subG(G, pre_ids, order_animals=list(range(D_pre)))

    print(
        f"Estimating K on available slice: trials={pre_trials}, neurons={len(pre_ids)}, animals={D_pre}"
    )
    K = determine_K_safe(data_fit, MODE)
    print("K =", K)
    np.save(OUT / "K.npy", np.array(K, dtype=int))

    print("Fitting diagonal per-PC potentials on available slice...")
    pots, _ = fit_statistics_from_dataset_diagonal(
        data_fit, K, G_fit, TAU_SIGMA, mode=MODE, gamma=GAMMA
    )

    # Save the inferred potentials (one NPZ per PC)
    pots_dir = OUT / "pots_initial_fit"
    pots_dir.mkdir(exist_ok=True)
    for k, pot in enumerate(pots, start=1):
        pot.save_as_npz(str(pots_dir / f"pot_pc{k}.npz"))
    print(f"Saved initial-fit potentials to: {pots_dir}")

    print("Estimating signal-variance uncertainty per PC...")
    sigvar_std = estimate_signal_variance_std_per_pc(
        pots, SIGVAR_N_STEPS, SIGVAR_N_SAMPLES
    )
    np.save(OUT / "signal_variance_std_smallfit.npy", sigvar_std)

    # -------- Predictions
    print("Predicting | trials axis ...")
    pred_trials = predict_trials_axis(pots, pre_trials, trials_grid, sigvar_std)

    print("Predicting | neurons axis ...")
    pred_neurons = predict_neurons_axis(
        pots, G, D_pre, base_order, pre_ids, neurons_grid, sigvar_std
    )

    print("Predicting | animals axis ...")
    pred_animals = predict_animals_axis(
        pots, G, D_grid, n_pre_per_animal, base_order, sigvar_std
    )

    # Save predictions + grids (only .npy; no JSON)
    np.save(OUT / "trials_grid.npy", np.array(trials_grid, dtype=int))
    np.save(OUT / "neurons_grid.npy", np.array(neurons_grid, dtype=int))
    np.save(OUT / "D_grid.npy", np.array(D_grid, dtype=int))

    for name, bundle in [
        ("trials", pred_trials),
        ("neurons", pred_neurons),
        ("animals", pred_animals),
    ]:
        np.save(OUT / f"pred_{name}_mean_rho.npy", bundle["mean_rho"])
        np.save(OUT / f"pred_{name}_epsilon.npy", bundle["epsilon"])
        np.save(OUT / f"pred_{name}_mean_rho_std.npy", bundle["mean_rho_std"])
        np.save(OUT / f"pred_{name}_epsilon_std.npy", bundle["epsilon_std"])

    # -------- Empirical (resampling; no fixed A/B)
    print(f"Empirical (resampling) | repetitions = {N_REPETITIONS}")

    emp_trials = empirical_trials_with_reps_resample(
        data_perm, K, pre_trials, trials_grid, pre_ids, N_REPETITIONS, seed=SEED + 1000
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
        N_REPETITIONS,
        seed=SEED + 3000,
    )

    for name, bundle in [
        ("trials", emp_trials),
        ("neurons", emp_neurons),
        ("animals", emp_animals),
    ]:
        np.save(OUT / f"emp_{name}_mean_rho_mean.npy", bundle["mean_rho_mean"])
        np.save(OUT / f"emp_{name}_epsilon_mean.npy", bundle["epsilon_mean"])
        np.save(OUT / f"emp_{name}_mean_rho_std.npy", bundle["mean_rho_std"])
        np.save(OUT / f"emp_{name}_epsilon_std.npy", bundle["epsilon_std"])
        # raw reps
        np.save(OUT / f"emp_{name}_mean_rho_all.npy", bundle["mean_rho_all"])
        np.save(OUT / f"emp_{name}_epsilon_all.npy", bundle["epsilon_all"])

    print("Done. Results saved to", OUT.resolve())


if __name__ == "__main__":
    main()
