# Figure 2 — synthetic dataset + inference
from pathlib import Path

import numpy as np
#import matplotlib.pyplot as plt

from pcalib.functions import (
    fit_statistics_from_dataset_diagonal,
    make_predictions,
    extrapolate_potential
)
from pcalib.classes import Potential
from pcalib.utils import (
    PCA_matlab_like,
    generate_gaussian_correlation_matrix,
    reduce_to_2d
)

# results directory (use consistently!)
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
# -------------------------------------------

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
# -----------------------------------------------------------------------

# Helper function to generate the 2d corridor function
def corridor_signal(T, var_array, epsilon_corridor):
    signal_array = np.zeros([T, 2])  # two components: K=2
    # Break everything into quarters
    signal_array[:T//4, 0] = np.linspace(-2, 0, T//4)

    n_points_middle = np.shape(np.arange(T//4, (3*T)//4))[0]
    signal_array[T//4:(3*T)//4, 0] = -np.cos(2*np.pi*np.arange(n_points_middle)/n_points_middle) + 1
    signal_array[T//4:(3*T)//4, 1] = -np.sin(2*np.pi*np.arange(n_points_middle)/n_points_middle)

    n_points_end = np.shape(signal_array[(3*T)//4:, 0])[0]
    signal_array[(3*T)//4:, 0] = np.linspace(0, -2, n_points_end)
    signal_array[:T//2, 1] -= epsilon_corridor
    signal_array[T//2:, 1] += epsilon_corridor

    signal_array -= np.mean(signal_array, 0, keepdims=True)
    signal_array /= np.sqrt(np.var(signal_array, 0, keepdims=True))
    signal_array[:, 0] *= np.sqrt(var_array[0])
    signal_array[:, 1] *= np.sqrt(var_array[1])

    return signal_array

load_potential = True
save_potential = True

if load_potential == False:
    ###################
    # Size parameters #
    ###################
    T = 100
    N_per_animal = 50
    K = 2
    D = 2
    N = N_per_animal * D
    n_trials = 40
    n_trials_array = np.arange(5, 50, 5)
    D_array = np.arange(1, 6)

    #####################
    # Define the signal #
    #####################
    var_array = [2, 1]
    epsilon_corridor = 0.1
    bar_x = corridor_signal(T, var_array, epsilon_corridor)
    # generate the mode basis
    bar_e_largest = np.random.normal(0, 1, [N_per_animal * D_array[-1], K])
    bar_e_largest, _ = np.linalg.qr(bar_e_largest)  # orthogonalizes the normal vectors
    bar_e_largest *= np.sqrt(N_per_animal * D_array[-1])

    bar_e = bar_e_largest[:N, :]

    #################
    # General noise #
    #################
    tau_sigma = 2
    Z = generate_gaussian_correlation_matrix(T, tau_sigma * np.sqrt(2))  # smoothing kernel
    bar_sigma_largest = np.abs(np.random.normal(1, 0.1, N_per_animal * D_array[-1]))
    bar_sigma = bar_sigma_largest[:N]
    ##############################
    # Trial-to-trial variability #
    ##############################
    tau_xi = 5
    Delta = generate_gaussian_correlation_matrix(T, tau_xi)  # direct correlations in time

    bar_xi_largest = np.zeros([D_array[-1], K])
    for k in range(K):
        bar_xi_largest[:, k] = np.sqrt(1) * np.sqrt(np.abs(np.random.normal(2/(k+1), 0.1/(k+1), D_array[-1])))

    G_largest = np.zeros([D_array[-1], N_per_animal * (D_array[-1]), N_per_animal * (D_array[-1])])
    for d in range(D_array[-1]):
        G_largest[d, d*N_per_animal:(d+1)*N_per_animal, d*N_per_animal:(d+1)*N_per_animal] = np.eye(N_per_animal)

    bar_xi = bar_xi_largest[:D, :]
    G = G_largest[:D, :N, :N]

    ###########################
    # Neuron-dependent kernel #
    ###########################
    Xi = np.zeros([T, T])  # Absent for this paper

    np.save(OUT / "true_mean_noise_variance.npy", np.sqrt(np.mean(bar_sigma**2 / n_trials)))
    np.save(OUT / "true_signal_variability.npy", np.var(bar_x, 0))

    many_animals_potential = Potential(bar_sigma_largest, bar_e_largest, G_largest, bar_xi_largest, Z, Delta, bar_x, Xi)
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
    D = np.load(OUT / "D_reference.npy")
    n_trials = np.load(OUT / "n_trials_reference.npy")
    bar_e_largest = many_animals_potential.bar_e
    bar_e = np.array(many_trials_potential.bar_e)
    [T, K] = np.shape(many_animals_potential.bar_x)
    N = np.shape(many_trials_potential.bar_sigma)[0]
    G_largest = many_animals_potential.G
    bar_xi_largest = many_animals_potential.bar_xi
    N_per_animal = N // D
    tau_sigma = np.load(OUT / "tau_sigma.npy").item()

    # Test: look at old rho and new rho.
    trial_averaged_potential = extrapolate_potential(many_trials_potential, new_trials=n_trials,
                                                     existing_number_of_trials=1)
    sample_data = many_trials_potential.generate_sample_data(n_samples=n_trials)
    inferred_potentials, _ = fit_statistics_from_dataset_diagonal(sample_data, K, many_trials_potential.G, tau_sigma,
                                                                  gamma=0.1)
    rho_predictions = np.zeros([N, K])
    for k in range(K):
        preds = make_predictions(inferred_potentials[k], return_R=True)
        rho_predictions[:, k] = preds["rho"][:, 0, 0]

    np.save("cached_results/predicted_rho_40_trials", rho_predictions)

    # Here generate an inferred neural trajectory, and also inferred v_i
    coeff, score, eigs = PCA_matlab_like(reduce_to_2d(sample_data, mode="trial-averaged"))
    inferred_v_i_40_trials = np.array(coeff[:, :K] * np.sqrt(N))
    inferred_y_40_trials = np.array(score[:, :K] / np.sqrt(N))
    for k in range(K):
        sign_k = np.sign(np.dot(inferred_v_i_40_trials[:, k], bar_e[:, k]))
        inferred_v_i_40_trials[:, k] *= sign_k
        inferred_y_40_trials[:, k] *= sign_k


    # Time to find empirical rho
    n_attempts=1000
    rho_emp = np.zeros([n_attempts, N, K])
    for attempt in range(n_attempts):
        print("attempt", attempt + 1)
        sample_data = many_trials_potential.generate_sample_data(n_samples=n_trials)
        coeff, score, eigs = PCA_matlab_like(reduce_to_2d(sample_data, mode="trial-averaged"))
        inferred_v_i_40_trials = np.array(coeff[:, :K] * np.sqrt(N))
        for k in range(K):
            sign_k = np.sign(np.dot(inferred_v_i_40_trials[:, k], bar_e[:, k]))
            inferred_v_i_40_trials[:, k] *= sign_k
        for k in range(K):
            bar_e[:, k] /= np.linalg.norm(bar_e[:, k])
            bar_e[:, k] *= np.sqrt(N)
            for i in range(N):
                rho_emp[attempt, i, k] = (inferred_v_i_40_trials[i, k] - bar_e[i, k]) ** 2 / 2

    np.save(OUT / "empirical_rho_40_trials", rho_emp)
    np.save(OUT / "inferred_y_40_trials", inferred_y_40_trials)
    np.save(OUT / "inferred_v_i_40_trials", inferred_v_i_40_trials)

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
# figure out how many attempts already saved (use any one file or max across them)
done_animals = max(
    attempts_done(OUT / "epsilon_animals.npy"),
    attempts_done(OUT / "rho_animals.npy"),
    attempts_done(OUT / "signal_variability_animals.npy"),
)
remaining_animals = max(0, n_attempts - done_animals)

if remaining_animals > 0:
    for attempt in range(remaining_animals):
        print(f"[animals] attempt {done_animals + attempt + 1} of {n_attempts}")
        synth_data_large = many_animals_potential.generate_sample_data(n_samples=n_trials)
        epsilon_animals_new = np.zeros([np.shape(D_array)[0], K])
        rho_animals_new = np.zeros([np.shape(D_array)[0], K])
        signal_variability_new = np.zeros([np.shape(D_array)[0], K])

        for i, D_current in enumerate(D_array):
            current_data = synth_data_large[:, :, :N_per_animal * D_current]
            current_bar_e = np.array(bar_e_largest[:N_per_animal*D_current, :])  # ensure mutable NumPy
            current_G = G_largest[:D_current, :N_per_animal * D_current, :N_per_animal * D_current]
            coeff, score, _ = PCA_matlab_like(np.mean(current_data, 0))
            sign_array = np.zeros(K)

            for k in range(K):
                current_bar_e[:, k] = current_bar_e[:, k] / np.linalg.norm(current_bar_e[:, k]) * np.sqrt(N_per_animal * D_current)
                sign_array[k] = np.sign(np.dot(coeff[:, k], current_bar_e[:, k]))

            coeff = coeff[:, :K] * sign_array[np.newaxis, :] * np.sqrt(N_per_animal * D_current)
            score = score[:, :K] * sign_array[np.newaxis, :] / np.sqrt(N_per_animal * D_current)

            if done_animals + attempt == 0 and D_current == D:
                np.save(OUT / f"inferred_y_{n_trials}_trials", score)

            # Time to make an inference:
            inferred_potentials, _ = fit_statistics_from_dataset_diagonal(current_data, K, current_G, tau_sigma, gamma=0.1)

            for k in range(K):
                prediction_dict = make_predictions(inferred_potentials[k])
                epsilon_animals_new[i, k] = _to_scalar(prediction_dict["epsilon"])
                rho_animals_new[i, k]     = _to_scalar(np.mean(prediction_dict["rho"],0))
                signal_variability_new[i, k] = np.var(inferred_potentials[k].bar_x, 0)[0]

        # append one attempt (row) to each file, capped
        append_rows_capped(OUT / "epsilon_animals.npy", epsilon_animals_new[np.newaxis, :, :], n_attempts)
        append_rows_capped(OUT / "rho_animals.npy", rho_animals_new[np.newaxis, :, :], n_attempts)
        append_rows_capped(OUT / "signal_variability_animals.npy", signal_variability_new[np.newaxis, :, :], n_attempts)
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
        synth_data_large = many_trials_potential.generate_sample_data(n_samples=n_trials_array[-1])
        epsilon_trials_new = np.zeros([np.shape(n_trials_array)[0], K])
        rho_trials_new = np.zeros([np.shape(n_trials_array)[0], K])
        sqrt_mean_sigma_squared_new = np.zeros([np.shape(n_trials_array)[0]])

        for i, n_trials_current in enumerate(n_trials_array):
            current_data = synth_data_large[:n_trials_current, :, :]
            coeff, score, _ = PCA_matlab_like(np.mean(current_data, 0))
            sign_array = np.zeros(K)
            for k in range(K):
                sign_array[k] = np.sign(np.dot(coeff[:, k], bar_e[:, k]))

            coeff = coeff[:, :K] * sign_array[np.newaxis, :] * np.sqrt(N)
            score = score[:, :K] * sign_array[np.newaxis, :] / np.sqrt(N)

            # Time to make an inference:
            inferred_potentials, _ = fit_statistics_from_dataset_diagonal(current_data, K, G, tau_sigma, gamma=0.1)
            sqrt_mean_sigma_squared_new[i] = np.sqrt(np.mean(inferred_potentials[0].bar_sigma**2))
            for k in range(K):
                prediction_dict = make_predictions(inferred_potentials[k],return_R=True)
                epsilon_trials_new[i, k] = _to_scalar(prediction_dict["epsilon"])
                rho_trials_new[i, k]     = _to_scalar(np.mean(prediction_dict["rho"],0))

        # append one attempt (row) to each file, capped
        append_rows_capped(OUT / "epsilon_trials.npy", epsilon_trials_new[np.newaxis, :, :], n_attempts)
        append_rows_capped(OUT / "rho_trials.npy", rho_trials_new[np.newaxis, :, :], n_attempts)
        append_rows_capped(OUT / "sqrt_mean_sigma_squared.npy", sqrt_mean_sigma_squared_new[np.newaxis, :], n_attempts)
else:
    print(f"[trials] already at cap ({n_attempts}) attempts; skipping.")
