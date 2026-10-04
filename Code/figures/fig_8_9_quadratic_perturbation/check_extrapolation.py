import numpy as np
import matplotlib.pyplot as plt

from pcalib.functions import (
    determine_dimensionality,
    extrapolate_potential,
    fit_statistics_from_dataset_diagonal,
    make_predictions,
)
from pcalib.synthetic import make_group_matrix
from pcalib.utils import (
    PCA_matlab_like,
    align_pca_to_reference,
    generate_gaussian_correlation_matrix,
    reduce_to_2d,
)


def generate_e_f(N, overlap=0.6, seed=None):
    rng = np.random.default_rng(seed)

    e = rng.normal(size=N)
    e *= np.sqrt(N) / np.linalg.norm(e)

    g = rng.normal(size=N)
    g -= np.dot(g, e) / N * e
    g *= np.sqrt(N) / np.linalg.norm(g)

    f = overlap * e + np.sqrt(1 - overlap**2) * g

    return e, f


def generate_bad_data(
    x, e, f, a, sigmas, n_trials, tau_sigma, seed=None
):
    rng = np.random.default_rng(seed)
    N = len(e)
    T = len(x)

    q = x**2 - np.mean(x**2)
    signal = np.outer(x, e) + a * np.outer(q, f)

    Z = generate_gaussian_correlation_matrix(
        T, tau_sigma * np.sqrt(2)
    )

    noise = rng.multivariate_normal(
        np.zeros(T),
        Z,
        size=(n_trials, N),
    ).transpose(0, 2, 1)

    noise *= sigmas[None, None, :] * np.sqrt(N)

    return signal[None, :, :] + noise


def empirical_errors(data, x, e):
    coeff, score, _ = PCA_matlab_like(
        reduce_to_2d(data, mode="trial-averaged")
    )

    estimated_e, estimated_x, _ = align_pca_to_reference(
        coeff,
        e[:, None],
        score=score,
        scale_loadings=np.sqrt(len(e)),
        scale_scores=1 / np.sqrt(len(e)),
    )

    rho = np.mean(0.5 * (estimated_e[:, 0] - e) ** 2)
    epsilon = np.mean((estimated_x[:, 0] - x) ** 2)

    return rho, epsilon


# Parameters
N = 100
T = 200

initial_n_trials = 10
maximum_n_trials = 50
n_repetitions = 50

a = 0.5
x_variance = 1.0
overlap = 0.6
gaussian_kernel_width = 3.0

trial_values = np.arange(initial_n_trials, maximum_n_trials + 1)
sigmas = np.ones(N)

t = np.arange(T)
x = np.sin(2 * np.pi * t / T)
x *= np.sqrt(x_variance / np.var(x))

e, f = generate_e_f(N, overlap=overlap, seed=0)
G = make_group_matrix(1, N)


# Infer and fit once at initial_n_trials
inference_data = generate_bad_data(
    x,
    e,
    f,
    a,
    sigmas,
    initial_n_trials,
    gaussian_kernel_width,
    seed=10_000,
)

K = determine_dimensionality(
    inference_data,
    mode="trial-averaged",
    n_samples=100,
    random_seed=0,
)

potentials, _ = fit_statistics_from_dataset_diagonal(
    inference_data,
    K,
    G,
    gaussian_kernel_width,
    mode="trial-averaged",
    gamma=0.1,
    improvement_cutoff=5e-2,
)


# Extrapolated predictions
predicted_rho = np.zeros(len(trial_values))
predicted_epsilon = np.zeros(len(trial_values))

for j, n_trials in enumerate(trial_values):
    potential = extrapolate_potential(
        potentials[0],
        new_trials=n_trials,
        existing_number_of_trials=initial_n_trials,
        mode="trial-averaged",
    )

    prediction = make_predictions(potential)

    predicted_rho[j] = np.mean(
        np.asarray(prediction["rho"])[:, 0, 0]
    )
    predicted_epsilon[j] = np.asarray(
        prediction["epsilon"]
    )[0, 0]


# Empirical errors
empirical_rho = np.zeros((n_repetitions, len(trial_values)))
empirical_epsilon = np.zeros_like(empirical_rho)

for repetition in range(n_repetitions):
    print("empirical values calculation. repetition", repetition+1)
    for j, n_trials in enumerate(trial_values):
        data = generate_bad_data(
            x,
            e,
            f,
            a,
            sigmas,
            n_trials,
            gaussian_kernel_width,
            seed=repetition * len(trial_values) + j,
        )

        empirical_rho[repetition, j], empirical_epsilon[repetition, j] = (
            empirical_errors(data, x, e)
        )


rho_mean = empirical_rho.mean(axis=0)
rho_std = empirical_rho.std(axis=0)

epsilon_mean = empirical_epsilon.mean(axis=0)
epsilon_std = empirical_epsilon.std(axis=0)


# Save results
a_tag = str(a).replace(".", "p")

np.save(f"trial_values_a_{a_tag}.npy", trial_values)
np.save(f"predicted_rho_a_{a_tag}.npy", predicted_rho)
np.save(f"predicted_epsilon_a_{a_tag}.npy", predicted_epsilon)
np.save(f"empirical_rho_a_{a_tag}.npy", empirical_rho)
np.save(f"empirical_epsilon_a_{a_tag}.npy", empirical_epsilon)


# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(
    trial_values,
    rho_mean,
    label="Empirical",
)
axes[0].fill_between(
    trial_values,
    rho_mean - rho_std,
    rho_mean + rho_std,
    alpha=0.2,
)
axes[0].plot(
    trial_values,
    predicted_rho,
    label="Predicted extrapolation",
)
axes[0].set_xlabel("Number of trials")
axes[0].set_ylabel(r"$\rho_{1,1}$")
axes[0].legend()

axes[1].plot(
    trial_values,
    epsilon_mean,
    label="Empirical",
)
axes[1].fill_between(
    trial_values,
    epsilon_mean - epsilon_std,
    epsilon_mean + epsilon_std,
    alpha=0.2,
)
axes[1].plot(
    trial_values,
    predicted_epsilon,
    label="Predicted extrapolation",
)
axes[1].set_xlabel("Number of trials")
axes[1].set_ylabel(r"$\epsilon_{1,1}$")
axes[1].legend()

fig.suptitle(f"Trial extrapolation, a = {a}, inferred K = {K}")
fig.tight_layout()

plt.savefig(
    f"trial_extrapolation_a_{a_tag}.png",
    dpi=200,
    bbox_inches="tight",
)
plt.show()