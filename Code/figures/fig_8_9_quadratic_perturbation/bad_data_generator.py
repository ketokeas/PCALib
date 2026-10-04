import numpy as np
import matplotlib.pyplot as plt

from pcalib.functions import (
    determine_dimensionality,
    fit_statistics_from_dataset_diagonal,
    make_predictions,
)
from pcalib.synthetic import make_group_matrix
from pcalib.utils import (
    PCA_matlab_like,
    align_pca_to_reference,
    reduce_to_2d, generate_gaussian_correlation_matrix
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
    x, e, f, a, sigmas, n_trials,
    tau_sigma, seed=None,
):
    rng = np.random.default_rng(seed)
    N = len(e)
    T = len(x)

    q = x**2 - np.mean(x**2)
    signal = np.outer(x, e) + a * np.outer(q, f)

    Z = generate_gaussian_correlation_matrix(
        T,
        tau_sigma * np.sqrt(2),
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
    epsilon = np.mean( (estimated_x[:, 0] - x) ** 2)

    return rho, epsilon


# Parameters
N = 100
T = 200
n_trials = 20
n_repetitions = 50

x_variance = 1.0
overlap = 0.6
gaussian_kernel_width = 3.0

a_values = np.linspace(0, 2, 10)
sigmas = np.ones(N)

t = np.arange(T)
x = np.sin(2 * np.pi * t / T)
x *= np.sqrt(x_variance / np.var(x))

e, f = generate_e_f(N, overlap=overlap, seed=0)
G = make_group_matrix(1, N)

empirical_rho = np.zeros((n_repetitions, len(a_values)))
empirical_epsilon = np.zeros_like(empirical_rho)

inferred_K = np.zeros(len(a_values), dtype=int)
predicted_rho = np.full(len(a_values), np.nan)
predicted_epsilon = np.full(len(a_values), np.nan)


for j, a in enumerate(a_values):

    # Empirical errors over repeated datasets
    for repetition in range(n_repetitions):
        data = generate_bad_data(
            x, e, f, a, sigmas, n_trials, gaussian_kernel_width,
            seed=repetition * len(a_values) + j,
        )

        empirical_rho[repetition, j], empirical_epsilon[repetition, j] = (
            empirical_errors(data, x, e)
        )

    # One PCALib inference for this value of a
    data = generate_bad_data(
        x, e, f, a, sigmas, n_trials, gaussian_kernel_width,
        seed=10_000 + j,
    )

    K = determine_dimensionality(
        data,
        mode="trial-averaged",
        n_samples=100,
        random_seed=0,
    )
    inferred_K[j] = K

    if K > 0:
        potentials, _ = fit_statistics_from_dataset_diagonal(
            data,
            K,
            G,
            gaussian_kernel_width,
            mode="trial-averaged",
            gamma=0.1,improvement_cutoff=5e-2
        )

        prediction = make_predictions(potentials[0])

        predicted_rho[j] = np.mean(
            np.asarray(prediction["rho"])[:, 0, 0]
        )
        predicted_epsilon[j] = np.asarray(
            prediction["epsilon"]
        )[0, 0]

        np.save('predicted_rho', predicted_rho)
        np.save('predicted_epsilon', predicted_epsilon)
        np.save('inferred_K', inferred_K)
        np.save('empirical_rho',empirical_rho)
        np.save('empirical_epsilon', empirical_epsilon)


        rho_mean = empirical_rho.mean(axis=0)
        rho_std = empirical_rho.std(axis=0)

        epsilon_mean = empirical_epsilon.mean(axis=0)
        epsilon_std = empirical_epsilon.std(axis=0)


        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(a_values, inferred_K, marker="o")
        axes[0].axhline(1, linestyle="--", label="Ground truth")
        axes[0].set_xlabel("a")
        axes[0].set_ylabel("Inferred K")
        axes[0].legend()

        axes[1].plot(a_values, rho_mean, label="Empirical")
        axes[1].fill_between(
            a_values,
            rho_mean - rho_std,
            rho_mean + rho_std,
            alpha=0.2,
        )
        axes[1].plot(a_values, predicted_rho, label="Predicted")
        axes[1].set_xlabel("a")
        axes[1].set_ylabel(r"$\rho_{1,1}$")
        axes[1].legend()

        axes[2].plot(a_values, epsilon_mean, label="Empirical")
        axes[2].fill_between(
            a_values,
            epsilon_mean - epsilon_std,
            epsilon_mean + epsilon_std,
            alpha=0.2,
        )
        axes[2].plot(a_values, predicted_epsilon, label="Predicted")
        axes[2].set_xlabel("a")
        axes[2].set_ylabel(r"$\epsilon_{1,1}$")
        axes[2].legend()

        fig.tight_layout()
        plt.savefig('inference_bad_data.png')