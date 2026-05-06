# Figure 2 — synthetic dataset + inference
from pathlib import Path

import numpy as np

from pcalib.functions import (
    fit_statistics_from_dataset_diagonal,
    make_predictions,
)
from pcalib.classes import Potential
from pcalib.synthetic import build_corridor_potential
from pcalib.utils import (
    PCA_matlab_like,
    align_pca_to_reference,
    reduce_to_2d,
)

# results directory
try:
    OUT = Path(__file__).resolve().parent / "cached_results"
except NameError:  # __file__ is not defined in notebooks
    OUT = Path.cwd() / "cached_results"

OUT.mkdir(parents=True, exist_ok=True)


# ---------- helpers to cap/resume ----------
def attempts_done(path: Path):
    return np.load(path).shape[0] if path.exists() else 0


def append_rows_capped(path: Path, new_block, cap: int):
    """
    Append new rows on axis 0, but ensure the saved file has <= cap rows total.
    new_block must have shape [num_new, ...].
    """
    if new_block is None or len(new_block) == 0:
        return
    if path.exists():
        old = np.load(path)
        need = max(0, cap - old.shape[0])
        if need == 0:
            return  # already at cap
        out = np.concatenate([old, new_block[:need]], axis=0)
    else:
        out = new_block[:cap]

    np.save(path, out)  # overwrite existing file directly


# ---- tiny helper to coerce arrays (incl. 1x1, JAX) to Python float ----
def _to_scalar(x):
    a = np.asarray(x)
    if a.ndim == 0:
        return float(a)
    if a.ndim == 1:
        return float(a[0])
    if a.ndim == 2:
        return float(a[0, 0])
    return float(a.reshape(-1)[0])


load_potential = True
save_potential = True

if load_potential is False:
    ###################
    # Size parameters #
    ###################
    T = 100
    N_per_animal = 50
    K = 2
    D = 2
    n_trials = 40
    n_trials_array = np.arange(5, 50, 5)
    D_array = np.arange(1, 6)

    #####################
    # Model parameters  #
    #####################
    tau_sigma = 2
    tau_xi = 5
    var_array = (2.0, 1.0)
    epsilon_corridor = 0.1

    rng = np.random.default_rng()

    # Build the largest synthetic model once, then slice it for the reference case
    many_animals_potential = build_corridor_potential(
        T=T,
        n_animals=D_array[-1],
        neurons_per_animal=N_per_animal,
        var_array=var_array,
        epsilon_corridor=epsilon_corridor,
        tau_sigma=tau_sigma,
        tau_xi=tau_xi,
        sigma_mean=1.0,
        sigma_std=0.1,
        rng=rng,
    )

    N = N_per_animal * D

    bar_x = np.array(many_animals_potential.bar_x)
    bar_e_largest = np.array(many_animals_potential.bar_e)
    bar_sigma_largest = np.array(many_animals_potential.bar_sigma)
    bar_xi_largest = np.array(many_animals_potential.bar_xi)
    G_largest = np.array(many_animals_potential.G)
    Z = np.array(many_animals_potential.Z)
    Delta = np.array(many_animals_potential.Delta)
    Xi = np.array(many_animals_potential.Xi)

    # Reference-size potential: first D animals only
    bar_e = bar_e_largest[:N, :]
    bar_sigma = bar_sigma_largest[:N]
    bar_xi = bar_xi_largest[:D, :]
    G = G_largest[:D, :N, :N]

    np.save(
        OUT / "true_mean_noise_variance.npy",
        np.sqrt(np.mean(bar_sigma**2 / n_trials)),
    )
    np.save(OUT / "true_signal_variability.npy", np.var(bar_x, axis=0))

    many_animals_potential.save_as_npz(str(OUT / "many_animals_potential.npz"))
    np.save(OUT / "D_array.npy", D_array)
    np.save(OUT / "D_reference.npy", D)

    many_trials_potential = Potential(bar_sigma, bar_e, G, bar_xi, Z, Delta, bar_x, Xi)
    many_trials_potential.save_as_npz(str(OUT / "many_trials_potential.npz"))

    np.save(OUT / "n_trials_array.npy", n_trials_array)
    np.save(OUT / "n_trials_reference.npy", n_trials)
    np.save(OUT / "tau_sigma.npy", tau_sigma)

else:
    many_animals_potential = Potential.from_npz(str(OUT / "many_animals_potential.npz"))
    many_trials_potential = Potential.from_npz(str(OUT / "many_trials_potential.npz"))

    D_array = np.load(OUT / "D_array.npy")
    n_trials_array = np.load(OUT / "n_trials_array.npy")
    D = int(np.load(OUT / "D_reference.npy"))
    n_trials = int(np.load(OUT / "n_trials_reference.npy"))
    tau_sigma = np.load(OUT / "tau_sigma.npy").item()

    bar_e_largest = np.array(many_animals_potential.bar_e)
    bar_e = np.array(many_trials_potential.bar_e)
    bar_xi_largest = np.array(many_animals_potential.bar_xi)
    G_largest = np.array(many_animals_potential.G)

    T, K = np.shape(many_animals_potential.bar_x)
    N = np.shape(many_trials_potential.bar_sigma)[0]
    N_per_animal = N // D

    # Predicted rho at the reference setting
    sample_data = many_trials_potential.generate_sample_data(n_samples=n_trials)
    inferred_potentials, _ = fit_statistics_from_dataset_diagonal(
        sample_data, K, many_trials_potential.G, tau_sigma, gamma=0.1
    )

    rho_predictions = np.zeros((N, K))
    for k in range(K):
        preds = make_predictions(inferred_potentials[k], return_R=True)
        rho_predictions[:, k] = preds["rho"][:, 0, 0]

    np.save(OUT / "predicted_rho_40_trials.npy", rho_predictions)

    # Inferred PCA-aligned trajectory and loadings at the reference setting
    coeff, score, eigs = PCA_matlab_like(
        reduce_to_2d(sample_data, mode="trial-averaged")
    )
    inferred_v_i_40_trials, inferred_y_40_trials, _ = align_pca_to_reference(
        coeff,
        bar_e,
        score=score,
        scale_loadings=np.sqrt(N),
        scale_scores=1 / np.sqrt(N),
    )

    # Empirical rho across many attempts
    n_attempts = 1000
    rho_emp = np.zeros((n_attempts, N, K))

    bar_e_norm = np.array(bar_e, copy=True)
    for k in range(K):
        bar_e_norm[:, k] /= np.linalg.norm(bar_e_norm[:, k])
        bar_e_norm[:, k] *= np.sqrt(N)

    for attempt in range(n_attempts):
        print("attempt", attempt + 1)
        sample_data = many_trials_potential.generate_sample_data(n_samples=n_trials)
        coeff, score, eigs = PCA_matlab_like(
            reduce_to_2d(sample_data, mode="trial-averaged")
        )
        inferred_v_i_40_trials, _, _ = align_pca_to_reference(
            coeff,
            bar_e_norm,
            scale_loadings=np.sqrt(N),
        )

        for k in range(K):
            rho_emp[attempt, :, k] = (
                0.5 * (inferred_v_i_40_trials[:, k] - bar_e_norm[:, k]) ** 2
            )

    np.save(OUT / "empirical_rho_40_trials.npy", rho_emp)
    np.save(OUT / "inferred_y_40_trials.npy", inferred_y_40_trials)
    np.save(OUT / "inferred_v_i_40_trials.npy", inferred_v_i_40_trials)

#################################
# Now we can generate some data #
#################################

# We have to generate and do inference on:
# - array of animals from 1 to 5 (n_trials=40)
# - number of trials from 5 to 45 (n_animals=2)

n_attempts = 50  # cap

# ---------------------------
# Animals sweep (capped)
# ---------------------------
done_animals = max(
    attempts_done(OUT / "epsilon_animals.npy"),
    attempts_done(OUT / "rho_animals.npy"),
    attempts_done(OUT / "signal_variability_animals.npy"),
)
remaining_animals = max(0, n_attempts - done_animals)

if remaining_animals > 0:
    for attempt in range(remaining_animals):
        print(f"[animals] attempt {done_animals + attempt + 1} of {n_attempts}")
        synth_data_large = many_animals_potential.generate_sample_data(
            n_samples=n_trials
        )

        epsilon_animals_new = np.zeros((len(D_array), K))
        rho_animals_new = np.zeros((len(D_array), K))
        signal_variability_new = np.zeros((len(D_array), K))

        for i, D_current in enumerate(D_array):
            N_current = N_per_animal * D_current
            current_data = synth_data_large[:, :, :N_current]
            current_bar_e = np.array(bar_e_largest[:N_current, :], copy=True)
            current_G = G_largest[:D_current, :N_current, :N_current]

            current_bar_e_norm = np.array(current_bar_e, copy=True)
            for k in range(K):
                current_bar_e_norm[:, k] /= np.linalg.norm(current_bar_e_norm[:, k])
                current_bar_e_norm[:, k] *= np.sqrt(N_current)

            coeff, score, _ = PCA_matlab_like(np.mean(current_data, axis=0))
            aligned_v, aligned_y, _ = align_pca_to_reference(
                coeff,
                current_bar_e_norm,
                score=score,
                scale_loadings=np.sqrt(N_current),
                scale_scores=1 / np.sqrt(N_current),
            )

            if done_animals + attempt == 0 and D_current == D:
                np.save(OUT / f"inferred_y_{n_trials}_trials.npy", aligned_y)

            inferred_potentials, _ = fit_statistics_from_dataset_diagonal(
                current_data, K, current_G, tau_sigma, gamma=0.1
            )

            for k in range(K):
                prediction_dict = make_predictions(inferred_potentials[k])
                epsilon_animals_new[i, k] = _to_scalar(prediction_dict["epsilon"])
                rho_animals_new[i, k] = _to_scalar(np.mean(prediction_dict["rho"], 0))
                signal_variability_new[i, k] = np.var(
                    inferred_potentials[k].bar_x, axis=0
                )[0]

        append_rows_capped(
            OUT / "epsilon_animals.npy",
            epsilon_animals_new[np.newaxis, :, :],
            n_attempts,
        )
        append_rows_capped(
            OUT / "rho_animals.npy",
            rho_animals_new[np.newaxis, :, :],
            n_attempts,
        )
        append_rows_capped(
            OUT / "signal_variability_animals.npy",
            signal_variability_new[np.newaxis, :, :],
            n_attempts,
        )
else:
    print(f"[animals] already at cap ({n_attempts}) attempts; skipping.")

# ---------------------------
# Trials sweep (capped)
# ---------------------------
done_trials = max(
    attempts_done(OUT / "epsilon_trials.npy"),
    attempts_done(OUT / "rho_trials.npy"),
    attempts_done(OUT / "sqrt_mean_sigma_squared.npy"),
)
remaining_trials = max(0, n_attempts - done_trials)

if remaining_trials > 0:
    G = many_trials_potential.G

    for attempt in range(remaining_trials):
        print(f"[trials] attempt {done_trials + attempt + 1} of {n_attempts}")
        synth_data_large = many_trials_potential.generate_sample_data(
            n_samples=n_trials_array[-1]
        )

        epsilon_trials_new = np.zeros((len(n_trials_array), K))
        rho_trials_new = np.zeros((len(n_trials_array), K))
        sqrt_mean_sigma_squared_new = np.zeros(len(n_trials_array))

        for i, n_trials_current in enumerate(n_trials_array):
            current_data = synth_data_large[:n_trials_current, :, :]

            coeff, score, _ = PCA_matlab_like(np.mean(current_data, axis=0))
            aligned_v, aligned_y, _ = align_pca_to_reference(
                coeff,
                bar_e,
                score=score,
                scale_loadings=np.sqrt(N),
                scale_scores=1 / np.sqrt(N),
            )

            inferred_potentials, _ = fit_statistics_from_dataset_diagonal(
                current_data, K, G, tau_sigma, gamma=0.1
            )

            sqrt_mean_sigma_squared_new[i] = np.sqrt(
                np.mean(inferred_potentials[0].bar_sigma**2)
            )

            for k in range(K):
                prediction_dict = make_predictions(
                    inferred_potentials[k], return_R=True
                )
                epsilon_trials_new[i, k] = _to_scalar(prediction_dict["epsilon"])
                rho_trials_new[i, k] = _to_scalar(np.mean(prediction_dict["rho"], 0))

        append_rows_capped(
            OUT / "epsilon_trials.npy",
            epsilon_trials_new[np.newaxis, :, :],
            n_attempts,
        )
        append_rows_capped(
            OUT / "rho_trials.npy",
            rho_trials_new[np.newaxis, :, :],
            n_attempts,
        )
        append_rows_capped(
            OUT / "sqrt_mean_sigma_squared.npy",
            sqrt_mean_sigma_squared_new[np.newaxis, :],
            n_attempts,
        )
else:
    print(f"[trials] already at cap ({n_attempts}) attempts; skipping.")