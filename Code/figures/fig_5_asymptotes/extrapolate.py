#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import numpy as np

# --- project imports ---
sys.path.insert(0, os.path.relpath("../../"))

from pcalib.classes import Potential
from pcalib.functions import (
    extrapolate_potential,
    make_predictions,
    asymptotic_regime_trials,
    asymptotic_regime_neurons,
    asymptotic_regime_animals,
)


# ====================== CONFIG ======================

DATASET_NAME = "2 Chillale et al"

AXIS = "trials"          # "neurons", "animals", or "trials"
EXTRAPOLATION_POINTS = 8
N_EXTRAPOLATION_REPEATS = 10

MODE = "trial-averaged"
SEED = 2

DATA_ROOT = Path("../../../Data")
DATA_PATH = DATA_ROOT / DATASET_NAME / "preformatted_data.npy"

POTENTIALS_DIR = Path("cached_results") / DATASET_NAME / "potentials"
OUT = Path("cached_results") / DATASET_NAME / AXIS

# Needed only for model-based prediction error bars.
SIGVAR_N_STEPS = 30
SIGVAR_N_SAMPLES = 200


# ====================== HELPERS ======================

def infer_animal_index(G):
    """Return animal index for each neuron from G[d, i, i]."""
    diag_stack = np.stack([np.diag(G[d]) for d in range(G.shape[0])], axis=0)
    return np.argmax(diag_stack, axis=0)


def make_G_for_more_neurons(G, new_N):
    """Expand G to new_N neurons by cyclically reusing original animal labels."""
    G = np.asarray(G)
    D, N, _ = G.shape

    if new_N < N:
        raise ValueError("new_N must be >= current N.")

    animal_idx = infer_animal_index(G)

    new_G = np.zeros((D, new_N, new_N), dtype=G.dtype)
    new_G[:, :N, :N] = G

    for i in range(N, new_N):
        d = int(animal_idx[i % N])
        new_G[d, i, i] = 1

    return new_G


def make_G_for_more_animals(G, new_D):
    """Expand G to new_D animals by duplicating the original animal layouts."""
    G = np.asarray(G)
    D, N, _ = G.shape

    if new_D < D:
        raise ValueError("new_D must be >= current D.")

    animal_idx = infer_animal_index(G)
    groups = [np.where(animal_idx == d)[0] for d in range(D)]

    new_groups = [groups[d % D] for d in range(new_D)]
    new_N = sum(len(g) for g in new_groups)

    new_G = np.zeros((new_D, new_N, new_N), dtype=G.dtype)

    offset = 0
    for d, group in enumerate(new_groups):
        n = len(group)
        new_G[d, offset : offset + n, offset : offset + n] = np.eye(
            n,
            dtype=G.dtype,
        )
        offset += n

    return new_G


def load_potentials(K=2):
    """Load first two saved PC potentials."""

    pots = []
    for k in range(1, K + 1):
        path = POTENTIALS_DIR / f"pot_pc{k}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing potential: {path}")
        pots.append(Potential.from_npz(str(path)))

    return pots


def extrapolate_one(pot, axis, size, initial_size, n_trials, seed):
    """Create one extrapolated Potential for one axis/size/random seed."""
    if axis == "trials":
        return extrapolate_potential(
            original=pot,
            new_trials=int(size),
            existing_number_of_trials=int(initial_size),
            mode=MODE,
            random_state=seed,
        )

    if axis == "neurons":
        G_new = make_G_for_more_neurons(np.asarray(pot.G), int(size))
        return extrapolate_potential(
            original=pot,
            new_neurons=int(size),
            new_G=G_new,
            mode=MODE,
            random_state=seed,
        )

    if axis == "animals":
        G_new = make_G_for_more_animals(np.asarray(pot.G), int(size))
        return extrapolate_potential(
            original=pot,
            new_neurons=G_new.shape[1],
            new_G=G_new,
            mode=MODE,
            random_state=seed,
        )

    raise ValueError("axis must be 'neurons', 'animals', or 'trials'.")


# ====================== MAIN ======================

def main():
    if AXIS not in {"neurons", "animals", "trials"}:
        raise ValueError("AXIS must be 'neurons', 'animals', or 'trials'.")

    OUT.mkdir(parents=True, exist_ok=True)

    pots = load_potentials()
    K = len(pots)

    data_init = np.load(DATA_PATH)
    n_trials = int(data_init.shape[0])

    N0 = int(np.asarray(pots[0].bar_e).shape[0])
    D0 = int(np.asarray(pots[0].G).shape[0])

    if AXIS == "trials":
        initial_size = n_trials
    elif AXIS == "neurons":
        initial_size = N0
    else:
        initial_size = D0

    axis_grid = np.linspace(
        initial_size,
        2 * initial_size,
        EXTRAPOLATION_POINTS,
        dtype=int,
    )

    print(f"Loaded {K} potentials from {POTENTIALS_DIR}")
    print("Initial data shape:", data_init.shape)
    print("Axis:", AXIS)
    print("Grid:", axis_grid)
    print("Repeats:", N_EXTRAPOLATION_REPEATS)

    # ----- analytical asymptotic-regime estimates -----
    asymptotic_functions = {
        "trials": asymptotic_regime_trials,
        "neurons": asymptotic_regime_neurons,
        "animals": asymptotic_regime_animals,
    }

    asymptotic_rho = np.full(K, np.nan)
    asymptotic_epsilon = np.full(K, np.nan)

    for k, pot in enumerate(pots):
        rho_start, epsilon_start = asymptotic_functions[AXIS](
            pot,
            n_trials=n_trials,
        )
        asymptotic_rho[k] = float(np.asarray(rho_start).reshape(-1)[0])
        asymptotic_epsilon[k] = float(np.asarray(epsilon_start).reshape(-1)[0])

    np.save(OUT / f"{AXIS}_asymptotic_rho.npy", asymptotic_rho)
    np.save(OUT / f"{AXIS}_asymptotic_epsilon.npy", asymptotic_epsilon)

    "a"/4

    # ----- uncertainty of fitted signal variance -----
    signal_variance_std = np.full(K, np.nan)

    for k, pot in enumerate(pots):
        s = pot.estimate_signal_variance_uncertainty(
            n_steps=SIGVAR_N_STEPS,
            n_samples=SIGVAR_N_SAMPLES,
        )
        signal_variance_std[k] = float(np.asarray(s).reshape(-1)[0])

    # ----- repeated stochastic extrapolation -----
    shape = (N_EXTRAPOLATION_REPEATS, len(axis_grid), K)

    rho_rep = np.full(shape, np.nan)
    eps_rep = np.full(shape, np.nan)

    rho_model_std_rep = np.full(shape, np.nan)
    eps_model_std_rep = np.full(shape, np.nan)

    for r in range(N_EXTRAPOLATION_REPEATS):
        print(f"Repeat {r + 1}/{N_EXTRAPOLATION_REPEATS}")

        for i, size in enumerate(axis_grid):
            print(f"  {AXIS}: size = {size}")

            for k, pot in enumerate(pots):
                seed = SEED + 100000 * r + 1000 * i + k

                pot_xt = extrapolate_one(
                    pot=pot,
                    axis=AXIS,
                    size=size,
                    initial_size=initial_size,
                    n_trials=n_trials,
                    seed=seed,
                )

                pred = make_predictions(
                    pot_xt,
                    predict_errorbars=True,
                    var_std_list=np.array([float(signal_variance_std[k])]),
                )

                rho_rep[r, i, k] = float(np.mean(np.asarray(pred["rho"])[:, 0, 0]))
                eps_rep[r, i, k] = float(np.asarray(pred["epsilon"]).reshape(-1)[0])

                rho_model_std_rep[r, i, k] = float(
                    np.asarray(pred["mean_rho_std"]).reshape(-1)[0]
                )
                eps_model_std_rep[r, i, k] = float(
                    np.asarray(pred.get("epsilon_std", 0.0)).reshape(-1)[0]
                )

    # ----- combine repeats and error bars -----
    mean_rho = np.nanmean(rho_rep, axis=0)
    epsilon = np.nanmean(eps_rep, axis=0)

    mean_rho_random_std = np.nanstd(rho_rep, axis=0, ddof=1)
    epsilon_random_std = np.nanstd(eps_rep, axis=0, ddof=1)

    mean_rho_model_std = np.sqrt(np.nanmean(rho_model_std_rep**2, axis=0))
    epsilon_model_std = np.sqrt(np.nanmean(eps_model_std_rep**2, axis=0))

    mean_rho_total_std = np.sqrt(mean_rho_random_std**2 + mean_rho_model_std**2)
    epsilon_total_std = np.sqrt(epsilon_random_std**2 + epsilon_model_std**2)

    # ----- save -----
    np.save(OUT / f"{AXIS}_grid.npy", axis_grid)

    np.save(OUT / f"{AXIS}_mean_rho.npy", mean_rho)
    np.save(OUT / f"{AXIS}_epsilon.npy", epsilon)

    np.save(OUT / f"{AXIS}_mean_rho_random_std.npy", mean_rho_random_std)
    np.save(OUT / f"{AXIS}_epsilon_random_std.npy", epsilon_random_std)

    np.save(OUT / f"{AXIS}_mean_rho_model_std.npy", mean_rho_model_std)
    np.save(OUT / f"{AXIS}_epsilon_model_std.npy", epsilon_model_std)

    np.save(OUT / f"{AXIS}_mean_rho_total_std.npy", mean_rho_total_std)
    np.save(OUT / f"{AXIS}_epsilon_total_std.npy", epsilon_total_std)

    # Backward-compatible names for existing plotting code.
    np.save(OUT / f"{AXIS}_mean_rho_std.npy", mean_rho_total_std)
    np.save(OUT / f"{AXIS}_epsilon_std.npy", epsilon_total_std)

    np.save(OUT / f"{AXIS}_mean_rho_repeats.npy", rho_rep)
    np.save(OUT / f"{AXIS}_epsilon_repeats.npy", eps_rep)

    np.save(OUT / f"{AXIS}_asymptotic_rho.npy", asymptotic_rho)
    np.save(OUT / f"{AXIS}_asymptotic_epsilon.npy", asymptotic_epsilon)

    np.save(OUT / f"{AXIS}_signal_variance_std.npy", signal_variance_std)

    print("Done.")
    print("Saved results to:", OUT.resolve())
    print("mean_rho shape:", mean_rho.shape)
    print("epsilon shape:", epsilon.shape)
    print("total rho std shape:", mean_rho_total_std.shape)
    print("total epsilon std shape:", epsilon_total_std.shape)


if __name__ == "__main__":
    main()