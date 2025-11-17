import copy

import matplotlib.pyplot as plt

from . import utils, classes
import numpy as np
import jax
import jax.numpy as jnp
import scipy
import time


def determine_dimensionality(
    non_flattened_data, mode, n_samples=1000, random_seed=0, plot=False
):
    """
    Determine the dimensionality of signal components in the flattened_data.

    Parameters:
    - flattened_data: (T, N) numpy array (already flattened trial x neuron activity)
    - Z: (T, T) numpy array, noise correlation matrix
    - n_samples: number of synthetic datasets to generate
    - random_seed: for reproducibility

    Returns:
    - K: inferred number of significant components
    """
    np.random.seed(random_seed)

    n_trials, T, N = non_flattened_data.shape

    flattened_data = utils.reduce_to_2d(non_flattened_data, mode)
    _, _, eigenvalues_true = utils.PCA_matlab_like(flattened_data)

    print("True eigenvalues:", eigenvalues_true)

    # ---------------------- Step 1: Center the data ----------------------
    non_flattened_data = non_flattened_data - np.mean(
        non_flattened_data, axis=1, keepdims=True
    )
    non_flattened_data = non_flattened_data - np.mean(
        non_flattened_data, axis=0, keepdims=True
    )

    # ------------------ Step 6: PCA on synthetic datasets ------------------
    synthetic_eigenvalues = np.zeros((n_samples, N))

    for i in range(n_samples):
        # synth_sample = np.array(synthetic_data[i])  # Convert JAX array to numpy
        # synth_sample_centered = synth_sample - np.mean(synth_sample, axis=0)
        # Make sure that the variance is the one we want.
        # synth_sample_centered /= np.std(synth_sample_centered, axis=0, keepdims=True)
        # synth_sample_centered *= (sigma_i_array * np.sqrt(N))[np.newaxis,:]
        print("Performing shuffling", i + 1, "out of", n_samples)
        for trial in range(n_trials):
            for neuron in range(N):
                non_flattened_data[trial, :, neuron] = non_flattened_data[
                    trial, np.roll(np.arange(T), np.random.randint(T)), neuron
                ]

        if mode == "trial-averaged":
            flattened_data = np.mean(non_flattened_data, axis=0)
        else:
            flattened_data = np.zeros((n_trials * T, N))
            for i in range(N):
                flattened_data[:, i] = np.ndarray.flatten(non_flattened_data[:, :, i])

        synth_sample = flattened_data
        _, _, eigvals = utils.PCA_matlab_like(synth_sample)
        synthetic_eigenvalues[i, :] = eigvals  # Save eigenvalues

    # ------------------ Step 7: Determine dimensionality ------------------
    # Take the maximum across synthetic samples for each eigenvalue index

    synthetic_max_eigenvalue = np.max(np.max(synthetic_eigenvalues))  # (N,)
    K = np.sum(eigenvalues_true > synthetic_max_eigenvalue)
    if plot:
        plt.hist(eigenvalues_true, bins=100, density=True, label="our data")
        plt.hist(
            np.ndarray.flatten(synthetic_eigenvalues),
            bins=100,
            density=True,
            label="Null model",
            alpha=0.5,
        )
        plt.axvline(x=synthetic_max_eigenvalue)
        plt.legend()
        plt.show()

    return K


def make_predictions(
    pot: classes.Potential,
    scale_factor: float = None,
    compute_derivative: bool = False,
    return_R: bool = False,
    predict_errorbars: bool = False,
    var_std_list=None,
) -> dict:
    """
    Make predictions from a scaled version of the potential, then rescale them back to the original units.

    Parameters:
    - pot: The original Potential object
    - scale_factor: Scalar by which bar_x, bar_sigma, and bar_xi are multiplied
    - compute_derivative: Whether to compute derivative of top eigenvalues w.r.t. sqrt(signal variances)
    - return_R: If True, include the R matrix in the output (for stability checks)

    Returns:
    - Dictionary with keys: "rho", "epsilon", "top_eigenvalues", optionally "d_lambda_d_sqrt_var" and "R"
    """

    if scale_factor is None:
        N = np.shape(pot.bar_e)[0]
        # Calculate the mean sigma
        if np.mean(pot.bar_sigma) > 0:
            scale_factor = 1 / (np.mean(pot.bar_sigma) * np.sqrt(N))
        else:
            # Take the largest variance?
            scale_factor = 1 / np.sqrt(
                np.var(pot.bar_x[:, 0]) + 1e-12
            )  # added small term to avoid divide-by-zero

    pot_scaled = classes.Potential(
        pot.bar_sigma * scale_factor,
        pot.bar_e,
        pot.G,
        pot.bar_xi * scale_factor,
        pot.Z,
        pot.Delta,
        pot.bar_x * scale_factor,
        pot.Xi,
    )
    point = pot_scaled.find_solution(verbose=True)
    # print("Time to find solution:", time.time() - beg)
    rho_pred = pot_scaled.rho(point)
    epsilon_pred = pot_scaled.epsilon(point) / (scale_factor**2)
    lambda_pred = pot_scaled.top_eigenvalues(point) / (scale_factor**2)
    result = {
        "rho": rho_pred,
        "epsilon": epsilon_pred,
        "top_eigenvalues": lambda_pred,
    }

    if compute_derivative:
        N, T, K = (
            pot_scaled.bar_sigma.shape[0],
            pot_scaled.Z.shape[0],
            pot_scaled.bar_x.shape[1],
        )
        tilde_Z = pot_scaled.Z @ (jnp.eye(T) - jnp.ones((T, T)) / T)
        normalized_y_t = pot_scaled.bar_x / jnp.sqrt(jnp.var(pot_scaled.bar_x, axis=0))

        W_Z = jnp.kron(point.W, tilde_Z)
        v_X = jnp.reshape(
            jnp.moveaxis(
                pot_scaled.tensor_contraction_with_X(point.v, pot_scaled.X), 2, 1
            ),
            (K * T, K * T),
        )
        A = jnp.eye(T * K) - 2 * (W_Z + v_X)
        InvA = jnp.transpose(jnp.reshape(jnp.linalg.inv(A), (K, T, K, T)), (0, 2, 1, 3))

        derivs = (
            jnp.einsum(
                "klts,dlm,sm,un,con,okut->mk",
                InvA,
                point.R,
                normalized_y_t,
                pot_scaled.bar_x,
                point.R,
                InvA,
            )
            / T
        )
        derivs += (
            jnp.einsum(
                "klts,dlm,sm,un,con,okut->nk",
                InvA,
                point.R,
                pot_scaled.bar_x,
                normalized_y_t,
                point.R,
                InvA,
            )
            / T
        )

        derivs /= scale_factor  # scale back derivative
        result["d_lambda_d_sqrt_var"] = derivs

    if return_R:
        result["R"] = np.array(point.R)

    if predict_errorbars:
        # var_std_list = pot.estimate_signal_variance_uncertainty(n_steps=30, n_samples=300)
        (
            eps_std,
            rho_std,
            mean_rho_std,
        ) = pot_scaled.estimate_theory_error_bars_from_signal_variance(
            point, var_std_list
        )
        result["epsilon_std"] = eps_std / (scale_factor**2)
        result["rho_std"] = rho_std
        result["mean_rho_std"] = mean_rho_std

    return result


if True:

    def fit_statistics_from_dataset(
        dataset,
        K,
        G,
        gaussian_kernel_width,
        mode="trial-averaged",
        gamma=0.5,
        eig_difference_cutoff=1e-2,
        improvement_cutoff=1e-3,
        scale_factor=10.0,
    ):
        jax.config.update("jax_enable_x64", True)

        init_gamma = gamma
        n_trials, T_single_trial, N = dataset.shape

        # ---------- Build flattened data & keep pre-centering means ----------
        if mode == "trial-averaged":
            flattened_dataset = np.mean(dataset, axis=0)  # shape [T, N], this is s_i(t)
            T = T_single_trial
        elif mode == "trial-concatenated":
            flattened_dataset = np.reshape(
                dataset, [n_trials * T_single_trial, N]
            )  # s_i(t)
            T = n_trials * T_single_trial
        else:
            raise ValueError(
                'Please select the way to treat your trials: "trial-averaged" or "trial-concatenated".'
            )

        # Save pre-centering per-neuron mean  \bar{s_i}  for Poisson diagonal correction
        pre_center_mean = np.mean(flattened_dataset, axis=0)  # vector length N

        # Mean-center data (standard PCA step)
        flattened_dataset = flattened_dataset - pre_center_mean[None, :]

        # Raw total variance (per neuron) from centered data
        total_variance_raw = (
            np.sum(flattened_dataset**2, axis=0) / T
        )  # Var[s_i] after centering

        # ---------- Compute Poisson-aware diagonal correction Δ_i ----------
        # Build discrete Gaussian kernel (L1-normalized) and get its L2 energy E = sum g^2
        def _gaussian_kernel_energy(width):
            # width is in time-bin units; use support ±4σ (safe & standard)
            L = int(np.ceil(4.0 * max(width, 1e-8)))
            t = np.arange(-L, L + 1)
            g = np.exp(-0.5 * (t / width) ** 2)
            g = g / np.sum(g)  # L1 normalization (sum g = 1)
            E = np.sum(g**2)  # L2 energy (sum g^2)
            return E

        E_kernel = _gaussian_kernel_energy(
            gaussian_kernel_width
        )  # scalar if same kernel for all i
        # If kernels differ across neurons, replace with a vector E_i.

        # Δ_i = (sum_tau F_i(τ)^2) * \bar{s_i}  ; here F is L1-normalized Gaussian with width given
        Delta = E_kernel * pre_center_mean  # vector length N

        # ---------- Do PCA on the corrected covariance ----------
        # C = (1/T) X^T X with X mean-centered. Then subtract Δ on the diagonal.
        X = flattened_dataset  # shape [T, N]
        C = (X.T @ X) / T  # empirical covariance across neurons
        # Apply correction on diagonal; clip to ensure PSD numerical stability if needed
        C[np.diag_indices(N)] = np.maximum(C[np.diag_indices(N)] - Delta, 0.0)

        # Eigen-decomposition of symmetric covariance; sort descending
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argsort(eigvals)[::-1]
        eigs = eigvals[idx]
        coeff = eigvecs[:, idx]

        # Scores (time factors) consistent with coeff from corrected C
        score = X @ coeff

        # Everything below proceeds as before, but using corrected eigs, coeff, score
        v_i = coeff[:, :K] * np.sqrt(N)
        y_t = score[:, :K] / np.sqrt(N)

        lambdas_empirical = eigs[:K] / N
        normalized_y_t = y_t / np.sqrt(np.var(y_t, axis=0))

        Z = utils.generate_gaussian_correlation_matrix(
            T, np.sqrt(2) * gaussian_kernel_width
        )
        (
            tau_kernel,
            tau_dx,
            xi_to_sigma_ratios,
        ) = fit_correlation_times_and_amplitude_ratios(dataset, K, G, Z, mode=mode)

        print("Inferred tau kernel:", tau_kernel)
        print("Inferred tau_dx:", tau_dx)
        print("Inferred xi_to_sigma_ratios:", xi_to_sigma_ratios)

        Delta_mat = utils.generate_gaussian_correlation_matrix(T, tau_dx)
        Xi = np.zeros((T, T))

        # IMPORTANT: use a variance baseline consistent with the corrected diagonal
        # total_variance_corr = Var[s_i] - Δ_i   (Δ_i is an additive diagonal inflation)
        total_variance_corr = np.maximum(total_variance_raw - Delta, 0.0)

        def recalculate_noise(sqrt_signal_variances):
            noise_variance = jnp.clip(
                total_variance_corr
                - jnp.einsum("k,ik->i", sqrt_signal_variances**2, v_i**2),
                0,
            )
            denum_non_sigma = (
                jnp.einsum("dk,ik,dii->", xi_to_sigma_ratios, v_i**2, G) / N
            )
            mean_sigma_squared_estimation = (
                jnp.sum(noise_variance) / N / (N + denum_non_sigma)
            )
            xi_k_estimation = jnp.sqrt(
                mean_sigma_squared_estimation * xi_to_sigma_ratios
            )
            sigma_i_estimation = jnp.sqrt(
                jnp.clip(
                    noise_variance
                    - jnp.einsum("dk,ik,dii->i", xi_k_estimation**2, v_i**2, G),
                    0,
                )
                / N
            )
            return sigma_i_estimation, xi_k_estimation

        sigma_i, xi_k = recalculate_noise(np.sqrt(np.var(y_t, axis=0)))
        potential = classes.Potential(sigma_i, v_i, G, xi_k, Z, Delta_mat, y_t, Xi)

        relative_top_eigenvalue_difference = np.inf
        improvement = np.inf
        eps = 1e-4

        predictions_old = make_predictions(
            potential, scale_factor=scale_factor, compute_derivative=True, return_R=True
        )

        while (
            relative_top_eigenvalue_difference > eig_difference_cutoff
            and improvement > improvement_cutoff
            and gamma > 1e-4
        ):
            gradient = predictions_old["d_lambda_d_sqrt_var"] @ (
                predictions_old["top_eigenvalues"] - lambdas_empirical
            )
            new_sqrt_variance = np.abs(
                np.sqrt(np.var(potential.bar_x, axis=0)) - gamma * gradient
            )
            potential.bar_x = normalized_y_t * new_sqrt_variance
            potential.bar_sigma, potential.bar_xi = recalculate_noise(new_sqrt_variance)
            potential.update_X()

            predictions = make_predictions(
                potential, scale_factor=scale_factor, return_R=True
            )

            R_diag_elements_old = jnp.einsum("dii->di", predictions_old["R"]) + np.sqrt(
                eps
            )
            R_diag_elements = jnp.einsum("dii->di", predictions["R"]) + np.sqrt(eps)

            if jnp.min(R_diag_elements / R_diag_elements_old) < 0.5:
                gamma /= 2
            else:
                predictions_old = predictions
                gamma = init_gamma

            top_eigenvalue_difference = jnp.sum(
                jnp.abs(predictions["top_eigenvalues"] - lambdas_empirical)
            )
            old_relative = relative_top_eigenvalue_difference
            relative_top_eigenvalue_difference = top_eigenvalue_difference / jnp.sum(
                jnp.abs(lambdas_empirical)
            )
            improvement = (
                np.abs(old_relative - relative_top_eigenvalue_difference)
                / relative_top_eigenvalue_difference
            )

            print("Relative eigenvalue difference:", relative_top_eigenvalue_difference)
            print("Improvement:", improvement)
            print("Rho:", predictions["rho"])

        return potential

    def fit_statistics_from_dataset_diagonal(
        dataset,
        K,
        G,
        gaussian_kernel_width,
        mode="trial-averaged",
        gamma=0.7,
        eig_difference_cutoff=1e-2,
        improvement_cutoff=1e-3,
    ):
        import copy
        import numpy as np
        import jax
        import jax.numpy as jnp
        from . import utils, classes
        from .functions import make_predictions

        jax.config.update("jax_enable_x64", True)

        D = np.shape(G)[0]
        init_gamma = gamma
        n_trials, T_single_trial, N = dataset.shape

        # ---------- Build flattened data & keep pre-centering means ----------
        if mode == "trial-averaged":
            flattened_dataset = np.mean(dataset, axis=0)  # s_i(t)
            T = T_single_trial
        elif mode == "trial-concatenated":
            flattened_dataset = np.reshape(
                dataset, [n_trials * T_single_trial, N]
            )  # s_i(t)
            T = n_trials * T_single_trial
        else:
            raise ValueError(
                'Please select either "trial-averaged" or "trial-concatenated".'
            )

        pre_center_mean = np.mean(flattened_dataset, axis=0)  # \bar{s_i}
        flattened_dataset = flattened_dataset - pre_center_mean[None, :]

        total_variance_raw = np.sum(flattened_dataset**2, axis=0) / T

        # ---------- Poisson-aware diagonal correction ----------
        def _gaussian_kernel_energy(width):
            L = int(np.ceil(4.0 * max(width, 1e-8)))
            t = np.arange(-L, L + 1)
            g = np.exp(-0.5 * (t / width) ** 2)
            g = g / np.sum(g)
            E = np.sum(g**2)
            return E

        E_kernel = _gaussian_kernel_energy(gaussian_kernel_width)
        Delta = E_kernel * pre_center_mean

        # ---------- PCA on corrected covariance ----------
        X = flattened_dataset
        C = np.array((X.T @ X) / T)
        C[np.diag_indices(N)] = np.maximum(C[np.diag_indices(N)] - Delta, 0.0)

        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argsort(eigvals)[::-1]
        eigs = eigvals[idx]
        coeff = eigvecs[:, idx]
        score = X @ coeff

        v_i = coeff[:, :K] * np.sqrt(N)
        y_t = score[:, :K] / np.sqrt(N)

        lambdas_empirical = eigs[:K] / N
        normalized_y_t = y_t / np.sqrt(np.var(y_t, axis=0))

        print("Inferring the xi...")
        Z = utils.generate_gaussian_correlation_matrix(
            T, np.sqrt(2) * gaussian_kernel_width
        )
        (
            tau_kernel,
            tau_dx,
            xi_to_sigma_ratios,
        ) = fit_correlation_times_and_amplitude_ratios(dataset, K, G, Z, mode=mode)
        print("Done!")
        print("Inferred tau kernel:", tau_kernel)
        print("Inferred tau_dx:", tau_dx)
        print("Inferred xi_to_sigma_ratios:", xi_to_sigma_ratios)

        Delta_mat = utils.generate_gaussian_correlation_matrix(T, tau_dx)
        Xi = np.zeros((T, T))

        # Use corrected per-neuron variance baseline
        total_variance_corr = np.maximum(total_variance_raw - Delta, 0.0)

        def recalculate_noise(sqrt_signal_variances):
            signal_variances = sqrt_signal_variances**2
            noise_variance = jnp.clip(
                total_variance_corr - jnp.einsum("k,ik->i", signal_variances, v_i**2),
                0,
            )
            denum_non_sigma = (
                jnp.einsum("dk,ik,dii->", xi_to_sigma_ratios, v_i**2, G) / N
            )
            mean_sigma_squared_estimation = (
                jnp.sum(noise_variance) / N / (N + denum_non_sigma)
            )
            xi_k_estimation = jnp.sqrt(
                mean_sigma_squared_estimation * xi_to_sigma_ratios
            )
            sigma_i_estimation = jnp.sqrt(
                jnp.clip(
                    noise_variance
                    - jnp.einsum("dk,ik,dii->i", xi_k_estimation**2, v_i**2, G),
                    0,
                )
                / N
            )
            return sigma_i_estimation, xi_k_estimation

        sigma_i, xi_k = recalculate_noise(jnp.sqrt(jnp.var(y_t, axis=0)))

        # Build diagonal potentials (unchanged)
        potentials = [
            classes.Potential(
                sigma_i,
                v_i[:, k : k + 1],  # [N,1]
                G,
                xi_k[:, k : k + 1],  # [D,1]
                Z,
                Delta_mat,
                y_t[:, k : k + 1],  # [T,1]
                Xi,
            )
            for k in range(K)
        ]

        old_Rs = np.zeros([D, K, K])
        for k in range(K):
            old_Rs[:, k, k] = potentials[k].find_solution().R[:, 0, 0]

        relative_top_eigenvalue_difference = np.inf
        improvement = np.inf
        eps = 1e-4

        while (
            relative_top_eigenvalue_difference > eig_difference_cutoff
            and improvement > improvement_cutoff
            and gamma > 1e-4
        ):
            top_eigenvalues = np.zeros(K)
            derivs = np.zeros((K, K))
            Rs = np.zeros([D, K, K])

            for k in range(K):
                preds = make_predictions(
                    potentials[k], compute_derivative=True, return_R=True
                )
                top_eigenvalues[k] = preds["top_eigenvalues"][0]
                derivs[k, k] = preds["d_lambda_d_sqrt_var"][0, 0]
                Rs[:, k, k] = preds["R"][:, 0, 0]

            R_diag_old = jnp.einsum("dii->di", old_Rs) + np.sqrt(eps)
            R_diag_new = jnp.einsum("dii->di", Rs) + np.sqrt(eps)

            gradient = derivs @ (top_eigenvalues - lambdas_empirical)
            new_sqrt_variance = np.zeros(K)

            for k in range(K):
                new_sqrt_variance[k] = np.abs(
                    np.sqrt(np.var(potentials[k].bar_x, axis=0)) - gamma * gradient[k]
                )
                potentials[k].bar_x = (
                    normalized_y_t[:, k, np.newaxis] * new_sqrt_variance[k]
                )

            new_sigma, new_xi = recalculate_noise(new_sqrt_variance)

            for k in range(K):
                potentials[k].bar_sigma = new_sigma
                potentials[k].bar_xi = new_xi[:, k, np.newaxis]
                potentials[k].update_X()

            old_Rs = Rs

            top_eigenvalue_difference = jnp.sum(
                jnp.abs(top_eigenvalues - lambdas_empirical)
            )
            old_relative = relative_top_eigenvalue_difference
            relative_top_eigenvalue_difference = top_eigenvalue_difference / jnp.sum(
                jnp.abs(lambdas_empirical)
            )
            improvement = (
                np.abs(old_relative - relative_top_eigenvalue_difference)
                / relative_top_eigenvalue_difference
            )
            gamma = init_gamma

            print(
                "Improvement:",
                improvement,
                "Lambdas (predicted):",
                top_eigenvalues,
                "Lambdas (empirical):",
                lambdas_empirical,
            )
            print("Predicted R:", R_diag_old)

        return potentials, tau_dx

else:

    def fit_statistics_from_dataset(
        dataset,
        K,
        G,
        gaussian_kernel_width,
        mode="trial-averaged",
        gamma=0.5,
        eig_difference_cutoff=1e-2,
        improvement_cutoff=1e-3,
        scale_factor=10.0,
    ):
        jax.config.update("jax_enable_x64", True)

        init_gamma = gamma
        n_trials, T_single_trial, N = dataset.shape

        if mode == "trial-averaged":
            flattened_dataset = np.mean(dataset, axis=0)
            T = T_single_trial
        elif mode == "trial-concatenated":
            flattened_dataset = np.reshape(dataset, [n_trials * T_single_trial, N])
            T = n_trials * T_single_trial
        else:
            raise ValueError(
                'Please select the way to treat your trials: "trial-averaged" or "trial-concatenated".'
            )

        flattened_dataset -= np.mean(flattened_dataset, axis=0, keepdims=True)
        total_variance = np.sum(flattened_dataset**2, axis=0) / T

        coeff, score, eigs = utils.PCA_matlab_like(flattened_dataset)
        v_i = coeff[:, :K] * np.sqrt(N)
        y_t = score[:, :K] / np.sqrt(N)

        lambdas_empirical = eigs[:K] / N
        normalized_y_t = y_t / np.sqrt(np.var(y_t, axis=0))

        Z = utils.generate_gaussian_correlation_matrix(
            T, np.sqrt(2) * gaussian_kernel_width
        )
        (
            tau_kernel,
            tau_dx,
            xi_to_sigma_ratios,
        ) = fit_correlation_times_and_amplitude_ratios(dataset, K, G, Z, mode=mode)

        print("Inferred tau kernel:", tau_kernel)
        print("Inferred tau_dx:", tau_dx)
        print("Inferred xi_to_sigma_ratios:", xi_to_sigma_ratios)
        # Debug
        # xi_to_sigma_ratios = np.zeros(K)
        # tau_dx=0

        Delta = utils.generate_gaussian_correlation_matrix(T, tau_dx)
        Xi = np.zeros((T, T))

        def recalculate_noise(sqrt_signal_variances):
            noise_variance = jnp.clip(
                total_variance
                - jnp.einsum("k,ik->i", sqrt_signal_variances**2, v_i**2),
                0,
            )
            denum_non_sigma = (
                jnp.einsum("dk,ik,dii->", xi_to_sigma_ratios, v_i**2, G) / N
            )
            mean_sigma_squared_estimation = (
                jnp.sum(noise_variance) / N / (N + denum_non_sigma)
            )
            xi_k_estimation = jnp.sqrt(
                mean_sigma_squared_estimation * xi_to_sigma_ratios
            )
            sigma_i_estimation = jnp.sqrt(
                jnp.clip(
                    noise_variance
                    - jnp.einsum("dk,ik,dii->i", xi_k_estimation**2, v_i**2, G),
                    0,
                )
                / N
            )
            return sigma_i_estimation, xi_k_estimation

        sigma_i, xi_k = recalculate_noise(np.sqrt(np.var(y_t, axis=0)))
        potential = classes.Potential(sigma_i, v_i, G, xi_k, Z, Delta, y_t, Xi)

        relative_top_eigenvalue_difference = np.inf
        improvement = np.inf
        eps = 1e-4

        predictions_old = make_predictions(
            potential, scale_factor=scale_factor, compute_derivative=True, return_R=True
        )

        while (
            relative_top_eigenvalue_difference > eig_difference_cutoff
            and improvement > improvement_cutoff
            and gamma > 1e-4
        ):
            gradient = predictions_old["d_lambda_d_sqrt_var"] @ (
                predictions_old["top_eigenvalues"] - lambdas_empirical
            )
            new_sqrt_variance = np.abs(
                np.sqrt(np.var(potential.bar_x, axis=0)) - gamma * gradient
            )
            potential.bar_x = normalized_y_t * new_sqrt_variance
            potential.bar_sigma, potential.bar_xi = recalculate_noise(new_sqrt_variance)
            potential.update_X()

            predictions = make_predictions(
                potential, scale_factor=scale_factor, return_R=True
            )

            R_diag_elements_old = jnp.einsum("dii->di", predictions_old["R"]) + np.sqrt(
                eps
            )
            R_diag_elements = jnp.einsum("dii->di", predictions["R"]) + np.sqrt(eps)

            if jnp.min(R_diag_elements / R_diag_elements_old) < 0.5:
                gamma /= 2
            else:
                predictions_old = predictions
                gamma = init_gamma

            top_eigenvalue_difference = jnp.sum(
                jnp.abs(predictions["top_eigenvalues"] - lambdas_empirical)
            )
            old_relative = relative_top_eigenvalue_difference
            relative_top_eigenvalue_difference = top_eigenvalue_difference / jnp.sum(
                jnp.abs(lambdas_empirical)
            )
            improvement = (
                np.abs(old_relative - relative_top_eigenvalue_difference)
                / relative_top_eigenvalue_difference
            )

            print("Relative eigenvalue difference:", relative_top_eigenvalue_difference)
            print("Improvement:", improvement)
            print("Rho:", predictions["rho"])

        return potential

    def fit_statistics_from_dataset_diagonal(
        dataset,
        K,
        G,
        gaussian_kernel_width,
        mode="trial-averaged",
        gamma=0.7,
        eig_difference_cutoff=1e-2,
        improvement_cutoff=1e-3,
    ):
        import copy
        import numpy as np
        import jax
        import jax.numpy as jnp
        from . import utils, classes
        from .functions import make_predictions

        jax.config.update("jax_enable_x64", True)

        D = np.shape(G)[0]
        init_gamma = gamma
        n_trials, T_single_trial, N = dataset.shape

        if mode == "trial-averaged":
            flattened_dataset = np.mean(dataset, axis=0)
            T = T_single_trial
        elif mode == "trial-concatenated":
            flattened_dataset = np.reshape(dataset, [n_trials * T_single_trial, N])
            T = n_trials * T_single_trial
        else:
            raise ValueError(
                'Please select either "trial-averaged" or "trial-concatenated".'
            )

        flattened_dataset -= np.mean(flattened_dataset, axis=0, keepdims=True)
        total_variance = np.sum(flattened_dataset**2, axis=0) / T

        coeff, score, eigs = utils.PCA_matlab_like(flattened_dataset)
        v_i = coeff[:, :K] * np.sqrt(N)
        y_t = score[:, :K] / np.sqrt(N)

        lambdas_empirical = eigs[:K] / N
        normalized_y_t = y_t / np.sqrt(np.var(y_t, axis=0))

        print("Inferring the xi...")
        Z = utils.generate_gaussian_correlation_matrix(
            T, np.sqrt(2) * gaussian_kernel_width
        )
        (
            tau_kernel,
            tau_dx,
            xi_to_sigma_ratios,
        ) = fit_correlation_times_and_amplitude_ratios(dataset, K, G, Z, mode=mode)
        print("Done!")
        print("Inferred tau kernel:", tau_kernel)
        print("Inferred tau_dx:", tau_dx)
        print("Inferred xi_to_sigma_ratios:", xi_to_sigma_ratios)
        # Debug
        # xi_to_sigma_ratios = np.zeros([D,K])
        # tau_dx=0

        Delta = utils.generate_gaussian_correlation_matrix(T, tau_dx)
        Xi = np.zeros((T, T))

        def recalculate_noise(sqrt_signal_variances):
            signal_variances = sqrt_signal_variances**2
            noise_variance = jnp.clip(
                total_variance - jnp.einsum("k,ik->i", signal_variances, v_i**2), 0
            )
            denum_non_sigma = (
                jnp.einsum("dk,ik,dii->", xi_to_sigma_ratios, v_i**2, G) / N
            )
            mean_sigma_squared_estimation = (
                jnp.sum(noise_variance) / N / (N + denum_non_sigma)
            )
            xi_k_estimation = jnp.sqrt(
                mean_sigma_squared_estimation * xi_to_sigma_ratios
            )
            sigma_i_estimation = jnp.sqrt(
                jnp.clip(
                    noise_variance
                    - jnp.einsum("dk,ik,dii->i", xi_k_estimation**2, v_i**2, G),
                    0,
                )
                / N
            )
            return sigma_i_estimation, xi_k_estimation

        sigma_i, xi_k = recalculate_noise(jnp.sqrt(jnp.var(y_t, axis=0)))

        # Build diagonal potentials
        potentials = [
            classes.Potential(
                sigma_i,
                v_i[:, k : k + 1],  # shape [N,1]
                G,
                xi_k[:, k : k + 1],  # shape [D,1]
                Z,
                Delta,
                y_t[:, k : k + 1],  # shape [T,1]
                Xi,
            )
            for k in range(K)
        ]

        old_Rs = np.zeros([D, K, K])
        for k in range(K):
            old_Rs[:, k, k] = potentials[k].find_solution().R[:, 0, 0]

        relative_top_eigenvalue_difference = np.inf
        improvement = np.inf
        eps = 1e-4

        while (
            relative_top_eigenvalue_difference > eig_difference_cutoff
            and improvement > improvement_cutoff
            and gamma > 1e-4
        ):
            top_eigenvalues = np.zeros(K)
            derivs = np.zeros((K, K))
            Rs = np.zeros([D, K, K])

            for k in range(K):
                preds = make_predictions(
                    potentials[k], compute_derivative=True, return_R=True
                )
                top_eigenvalues[k] = preds["top_eigenvalues"][0]
                derivs[k, k] = preds["d_lambda_d_sqrt_var"][0, 0]
                Rs[:, k, k] = preds["R"][:, 0, 0]

            # Check R-stability condition
            R_diag_old = jnp.einsum("dii->di", old_Rs) + np.sqrt(eps)
            R_diag_new = jnp.einsum("dii->di", Rs) + np.sqrt(eps)

            # if jnp.min(R_diag_new / R_diag_old) < 0.5:
            #    gamma /= 2
            #    print("Decreased step. gamma=", gamma)
            # else:
            # Update signal amplitudes
            gradient = derivs @ (top_eigenvalues - lambdas_empirical)
            new_sqrt_variance = np.zeros(K)

            for k in range(K):
                new_sqrt_variance[k] = np.abs(
                    np.sqrt(np.var(potentials[k].bar_x, axis=0)) - gamma * gradient[k]
                )
                potentials[k].bar_x = (
                    normalized_y_t[:, k, np.newaxis] * new_sqrt_variance[k]
                )

            new_sigma, new_xi = recalculate_noise(new_sqrt_variance)

            for k in range(K):
                potentials[k].bar_sigma = new_sigma
                potentials[k].bar_xi = new_xi[:, k, np.newaxis]
                potentials[k].update_X()

            old_Rs = Rs

            top_eigenvalue_difference = jnp.sum(
                jnp.abs(top_eigenvalues - lambdas_empirical)
            )
            old_relative = relative_top_eigenvalue_difference
            relative_top_eigenvalue_difference = top_eigenvalue_difference / jnp.sum(
                jnp.abs(lambdas_empirical)
            )
            improvement = (
                np.abs(old_relative - relative_top_eigenvalue_difference)
                / relative_top_eigenvalue_difference
            )
            gamma = init_gamma

            print(
                "Improvement:",
                improvement,
                "Lambdas (predicted):",
                top_eigenvalues,
                "Lambdas (empirical):",
                lambdas_empirical,
            )
            print("Predicted R:", R_diag_old)

        return potentials, tau_dx


def estimate_xi_sigma_ratio_knownK(
    data, Z, Delta, G, K, n_boot=100, band_threshold=1e-6, rng=None
):
    """
    Estimate (xi^{(k)}_d)^2 / mean_i sigma_i^2 via phase-randomized surrogates.

    Args
    ----
    data : (n_trials, T, N)
    Z, Delta : (T, T) known periodic correlation matrices (first row is the lag profile)
    G : (D,N,N) diagonal selectors or (D,N) boolean mask per animal
    K : int, known latent rank
    n_boot : # of bootstrap surrogates for Γ^{shuf}
    band_threshold : keep lags with |kernel| >= threshold * max(|kernel|)
    rng : np.random.Generator or None

    Returns
    -------
    ratio : (D,K)   estimates of (xi_d^{(k)})^2 / mean_i sigma_i^2
    """

    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, float)
    Z = np.asarray(Z, float)
    Delta = np.asarray(Delta, float)
    n_trials, T, N = data.shape
    assert Z.shape == (T, T) and Delta.shape == (T, T)
    assert 1 <= K <= min(T, N)

    # Parse G -> masks (D,N)
    G = np.asarray(G)
    if G.ndim == 3 and G.shape[1:] == (N, N):
        masks = np.diagonal(G, axis1=1, axis2=2)
    elif G.ndim == 2 and G.shape[1] == N:
        masks = G
    else:
        raise ValueError("G must be (D,N,N) diagonals or (D,N) masks")
    masks = (masks > 0).astype(float)
    D = masks.shape[0]

    # --- PCA on trial-average (rank K), subtract reconstruction ---
    coeff, score, eigs = utils.PCA_matlab_like(np.mean(data, 0))
    v = coeff[:, :K] * np.sqrt(N)
    y = score[:, :K] / np.sqrt(N)
    resid = data - np.einsum("ik,tk->ti", v, y)[np.newaxis, :, :]  # (n_trials,T,N)

    # --- choose lags (banded periodic kernels) ---
    dprof = Delta[0].copy()
    zprof = Z[0].copy()
    scale = max(np.max(np.abs(dprof)), np.max(np.abs(zprof)))
    thr = band_threshold * scale if scale > 0 else 0.0
    lags = np.where((np.abs(dprof) >= thr) | (np.abs(zprof) >= thr))[0]
    alpha = float(np.sum(dprof[lags] ** 2))  # Σ Δ(τ)^2

    ratios = np.zeros((D, K))
    xi2 = np.zeros((D, K))
    mu = np.zeros(D)
    C_list = []

    # helper: symmetric LS to get xi^2 from A and C without inversion
    def xi2_from_A_C(A, C):
        iu = np.triu_indices_from(A)
        y = A[iu]
        X = np.stack(
            [np.outer(C[:, k], C[:, k])[iu] for k in range(C.shape[1])], axis=1
        )
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.clip(coef, 0.0, None)

    # --- per-animal loop ---
    for d in range(D):
        m = masks[d].astype(float)  # (N,)
        idx = np.where(m > 0.5)[0]
        if idx.size == 0:
            C_list.append(np.zeros((K, K)))
            ratios[d, :], xi2[d, :], mu[d] = np.nan, np.nan, np.nan
            continue

        Vm = v * m[:, None]  # (N,K)
        C = Vm.T @ v  # (K,K)
        C_list.append(C)

        # --- Γ(τ): original (no shuffle) ---
        S_D = np.zeros((K, K))
        for r in range(n_trials):
            Ur = resid[r] @ Vm  # (T,K)
            for tau in lags:
                M = Ur.T @ np.roll(Ur, -int(tau), axis=0)
                S_D += dprof[tau] * M
        S_D /= n_trials

        # --- Γ^{shuf}(τ): average over bootstraps with phase randomization ---
        S_D_sh = np.zeros((K, K))
        for b in range(n_boot):
            print(b, " out of ", n_boot)
            S_D_b = np.zeros((K, K))
            for r in range(n_trials):
                R = resid[r].copy()  # (T,N)
                # ### CHANGED: independent phase randomization per neuron
                R[:, idx] = utils.phase_randomize(
                    R[:, idx], rng=rng, keep_dc_nyquist=True
                )
                Ur = R @ Vm
                for tau in lags:
                    M = Ur.T @ np.roll(Ur, -int(tau), axis=0)
                    S_D_b += dprof[tau] * M
            S_D_sh += S_D_b / n_trials
        S_D_sh /= n_boot

        # --- latent part by difference ---
        A_hat = (S_D - S_D_sh) / alpha

        # --- recover xi^2 via symmetric LS on A_hat and C ---
        xi2_d = xi2_from_A_C(A_hat, C)
        xi2[d, :] = xi2_d

        # --- mean sigma^2 per animal ---
        Theta = np.stack([dprof[lags], zprof[lags]], axis=1)  # (L,2)
        NT = Theta.T @ Theta
        sig2_vals = []
        for i in idx:
            g = np.zeros(lags.size)
            for r in range(n_trials):
                x = resid[r, :, i]
                g += np.array([float(np.dot(x, np.roll(x, -int(tau)))) for tau in lags])
            g /= n_trials
            coef = np.linalg.lstsq(NT, Theta.T @ g, rcond=None)[0]  # [a_i, b_i]
            sigma2_i = max(coef[1] / float(N), 0.0)  # b_i = N * sigma_i^2
            sig2_vals.append(sigma2_i)
        mu[d] = float(np.mean(sig2_vals)) if sig2_vals else np.nan

        ratios[d, :] = xi2_d / mu[d] if (mu[d] > 0 and np.isfinite(mu[d])) else np.nan

    return ratios


import numpy as np
from scipy import optimize

# ------------------------------------------------------------------
# Assumed available in your utils (as you provided):
#   - generate_periodic_exponential_kernel(T, tau, eps=1e-10)
#   - generate_gaussian_correlation_matrix(T, tau, eps=1e-10)
# ------------------------------------------------------------------


def fit_correlation_times_and_amplitude_ratios(
    dataset,  # shape [n_trials, T_single_trial, N]
    K,  # number of PCs
    G,  # shape [D, I, I] (usually diagonal per animal/group)
    Z,  # temporal covariance kernel/matrix to seed tau_1
    mode="trial-averaged",  # or "trial-concatenated"
    n_attempts=100,  # for L1 tuning (synthetic curves)
    n_boot=200,  # bootstraps for ratio-of-means
    rng=None,
):
    """
    Returns:
      tau1_est   : float
      tau2_est   : float      (tau1 + extra_dx, i.e. the wider timescale)
      amp_ratio  : array [D,K]   (Jensen-corrected, ratio-of-means with your scaling applied)
    """
    # --------------------------- small utilities ---------------------------

    def centering_matrix(T: int):
        I = np.eye(T)
        one = np.ones((T, 1))
        return I - (one @ one.T) / T

    def frobenius_nnls_2d(S, B1, B2, ridge=1e-10):
        """
        Closed-form active-set NNLS for:
            min_{c1,c2>=0} ||S - c1 B1 - c2 B2||_F^2
        Returns (c1, c2).
        """
        ip = lambda A, B: np.tensordot(A, B)  # Frobenius inner product
        G11 = ip(B1, B1) + ridge
        G22 = ip(B2, B2) + ridge
        G12 = ip(B1, B2)
        b1 = ip(B1, S)
        b2 = ip(B2, S)
        det = G11 * G22 - G12 * G12
        if det > 0:
            c1_u = (G22 * b1 - G12 * b2) / det
            c2_u = (-G12 * b1 + G11 * b2) / det
        else:
            # extremely rare fallback
            c1_u = b1 / max(G11, 1e-18)
            c2_u = b2 / max(G22, 1e-18)

        if c1_u >= 0 and c2_u >= 0:
            return float(c1_u), float(c2_u)

        # Axis projections
        c1_only = max(b1 / max(G11, 1e-18), 0.0)
        c2_only = max(b2 / max(G22, 1e-18), 0.0)
        S2 = ip(S, S)

        def obj(c1, c2):
            return (
                S2
                - 2 * (c1 * b1 + c2 * b2)
                + (c1 * c1 * G11 + 2 * c1 * c2 * G12 + c2 * c2 * G22)
            )

        if obj(c1_only, 0.0) <= obj(0.0, c2_only):
            return float(c1_only), 0.0
        else:
            return 0.0, float(c2_only)

    def build_demeaned_bases(T, tau1, tau2):
        H = centering_matrix(T)
        C1 = utils.generate_gaussian_correlation_matrix(T, tau1)
        C2 = utils.generate_gaussian_correlation_matrix(T, tau2)
        B1 = H @ C1 @ H
        B2 = H @ C2 @ H
        return B1, B2

    if rng is None:
        rng = np.random.default_rng()

    n_trials = dataset.shape[0]
    T_single = dataset.shape[1]
    N = dataset.shape[2]
    D = G.shape[0]
    I = N  # number of channels (same as data's last dim)
    # ----------------------------------------------------------------------

    # 0) Demean over time per trial/channel
    dataset = dataset - dataset.mean(axis=1, keepdims=True)

    # 1) Build data matrix for PCA
    if mode == "trial-averaged":
        flattened = dataset.mean(axis=0)  # [T, N]
        T = T_single
    elif mode == "trial-concatenated":
        flattened = dataset.reshape(n_trials * T_single, N)  # [(n*T), N]
        T = n_trials * T_single
    else:
        raise ValueError('mode must be "trial-averaged" or "trial-concatenated".')

    # 2) PCA (MATLAB-like), then take top K
    coeff, score, eigs = utils.PCA_matlab_like(flattened)
    v_i = coeff[:, :K] * np.sqrt(N)  # [N,K]
    y_t = score[:, :K] / np.sqrt(N)  # [T or nT, K]
    if mode == "trial-concatenated":
        # reshape y_t back to [n_trials, T_single, K] and average across trials in time-aligned sense
        y_t = y_t.reshape(n_trials, T_single, -1).mean(axis=0)  # [T_single,K]
        T = T_single

    # 3) Build G-weighted residuals per (trial,d,t,k)  -> shape [n_trials, D, T, K]
    #    First term: projection of data into PC space with G weighting
    #    Second term: low-rank reconstruction (y_t, v_i) also through G
    ones_trials = np.ones(n_trials)
    residual_noise = np.einsum("ati,ik,dii->adtk", dataset, v_i, G) - np.einsum(
        "a,tl,il,ik,dii->adtk", ones_trials, y_t, v_i, v_i, G
    )

    # 4) Autocovariance (periodic, demeaned) per (trial,d,k); then average over trials,d,k to a single curve
    mean_over_T_sq = residual_noise.mean(axis=2, keepdims=False) ** 2  # [n_trials,D,K]
    autocov = np.zeros((n_trials, D, T, K), dtype=float)
    for t in range(T):
        shifted = np.roll(residual_noise, t, axis=2)
        autocov[:, :, t, :] = (shifted * residual_noise).mean(axis=2) - mean_over_T_sq

    to_fit_xi = autocov.sum(axis=(0, 1, 3))  # sum over trials, D, K  -> length T
    to_fit_xi = to_fit_xi / to_fit_xi[0]

    # 5) Estimate tau_1 by fitting Z's first row with a Gaussian periodic kernel
    #    Z is a T×T correlation (or row), we fit its row-0 with kernel(tau)
    zrow = Z[0, :] if Z.ndim == 2 else Z
    zrow = np.asarray(zrow).reshape(-1)
    if zrow.size != T:
        raise ValueError("Z (or Z[0,:]) must have length T.")

    def fitfun(x, tau):
        # vectorized: return the kernel values at positions x
        k = utils.generate_periodic_exponential_kernel(T, float(tau))
        return k[np.asarray(x, dtype=int)]

    tau1 = optimize.curve_fit(fitfun, np.arange(T), zrow, bounds=(0, T))[0][0]

    # 6) Tune L1 on synthetic curves so that a2->0 when only kernel(tau1) is present
    def loss_amp(x, gamma, series):
        a1, a2, tau_dx = x[0], x[1], x[2]
        k1 = utils.generate_periodic_exponential_kernel(T, tau1)
        k2 = utils.generate_periodic_exponential_kernel(T, tau1 + tau_dx)
        fun = a1 * (k1 - k1.mean()) + a2 * (k2 - k2.mean())
        fun = fun / fun[0]
        return np.sum((series - fun) ** 2) + gamma * np.abs(a2)

    # synthesize n_attempts residual curves with covariance Z (only first kernel)
    autocorr_to_fit = np.zeros((n_attempts, T), dtype=float)
    LZ = np.linalg.cholesky(Z + 1e-12 * np.eye(T)) if Z.ndim == 2 else None
    for i in range(n_attempts):
        # synthetic residuals: shape [n_trials, D, K, T] → we'll just do many iid draws length T
        Xsyn = rng.standard_normal((n_trials, D, K, T))  # white
        if LZ is not None:
            Xsyn = Xsyn @ LZ.T
        # periodic, demeaned autocovariance per (trial,d,k)
        m2 = Xsyn.mean(axis=3, keepdims=True) ** 2
        ac = np.zeros((n_trials, D, T, K))
        for t in range(T):
            ac[:, :, t, :] = (np.roll(Xsyn, t, axis=3) * Xsyn).mean(axis=3) - m2[..., 0]
        to_fit = ac.sum(axis=(0, 1, 3))
        to_fit /= to_fit[0]
        autocorr_to_fit[i, :] = to_fit

    # Binary search gamma so that fitted a2 ~ 0 on synthetic curves
    left_gamma, right_gamma = 0.0, 1.0

    def fit_with_gamma(gamma, series):
        # optimize [a1, a2, tau_dx] with tau1 fixed
        x0 = np.array([1.0, 1.0, 1.0])
        bnds = ((1e-6, T), (0.0, T), (0.0, T))
        res = optimize.minimize(lambda x: loss_amp(x, gamma, series), x0, bounds=bnds)
        return res.x  # a1, a2, tau_dx

    # grow right_gamma until it kills a2 on all attempts
    vals = np.zeros((n_attempts, 3))
    for i in range(n_attempts):
        vals[i, :] = fit_with_gamma(right_gamma, autocorr_to_fit[i, :])
    while np.max(vals[:, 1]) > 1e-3:
        right_gamma *= 2.0
        for i in range(n_attempts):
            vals[i, :] = fit_with_gamma(right_gamma, autocorr_to_fit[i, :])
    init_right = right_gamma

    # refine by bisection
    while (right_gamma - left_gamma) > (1e-6) * init_right:
        gamma = 0.5 * (left_gamma + right_gamma)
        for i in range(n_attempts):
            vals[i, :] = fit_with_gamma(gamma, autocorr_to_fit[i, :])
        if np.max(vals[:, 1]) > 1e-3:
            left_gamma = gamma
        else:
            right_gamma = gamma
    gamma = 0.5 * (left_gamma + right_gamma)

    # 7) Fit extra tau on real averaged curve (with gamma fixed and tau1 fixed)
    res_real = optimize.minimize(
        lambda x: loss_amp([x[0], x[1], x[2]], gamma, to_fit_xi),
        x0=np.array([1.0, 1.0, 1.0]),
        bounds=((1e-6, T), (0.0, T), (0.0, T)),
    )
    a1_hat, a2_hat, tau_dx = res_real.x
    tau2 = tau1 + tau_dx

    # 8) Amplitude ratios per (d,k) via demeaned covariance + NNLS + ratio-of-means
    # Build demeaned bases
    B1, B2 = build_demeaned_bases(T, tau1, tau2)

    # Scaling coefficients (your formulas)
    diag_weight = np.einsum("dii,ik->dk", G, (v_i**2))  # [D,K]
    coeff_with_xi = diag_weight**2
    coeff_with_sigma = N * diag_weight

    # Residuals are already built above: residual_noise [n_trials, D, T, K]
    # (Recompute with current v_i, y_t, G if you want to be 100% consistent with tau fit.)
    # We'll compute ratios with trial bootstrap → ratio-of-means.
    if rng is None:
        rng = np.random.default_rng()

    amp_ratio_rom = np.zeros((D, K), dtype=float)  # final Jensen-corrected
    amp_ratio_mor = np.zeros((D, K), dtype=float)  # diagnostic

    for d in range(D):
        for k in range(K):
            c1_list, c2_list, r_list = [], [], []
            for _ in range(n_boot):
                idx = rng.integers(0, n_trials, size=n_trials)  # bootstrap trials
                X = residual_noise[idx, d, :, k]  # [n_trials, T]
                X = X - X.mean(axis=1, keepdims=True)  # ensure demeaning per trial
                Sigma_hat = (X.T @ X) / X.shape[0]  # [T,T]

                c1, c2 = frobenius_nnls_2d(Sigma_hat, B1, B2, ridge=1e-10)

                # scale back to source amplitudes (your geometry factors)
                c1_tilde = c1 / max(coeff_with_sigma[d, k], 1e-18)
                c2_tilde = c2 / max(coeff_with_xi[d, k], 1e-18)
                c1_list.append(c1_tilde)
                c2_list.append(c2_tilde)
                if c1_tilde > 0:
                    r_list.append(c2_tilde / c1_tilde)

            c1_arr = np.asarray(c1_list)
            c2_arr = np.asarray(c2_list)
            r_arr = np.asarray(r_list)

            # Jensen-corrected: ratio of means
            amp_ratio_rom[d, k] = c2_arr.mean() / max(c1_arr.mean(), 1e-18)
            # Diagnostic: mean of ratios (shows Jensen bias)
            amp_ratio_mor[d, k] = r_arr.mean() if r_arr.size > 0 else np.nan

    print("Inferred amplitude ratios:", amp_ratio_mor)
    # 9) Return value
    return (
        tau1 / np.sqrt(2.0),
        tau2,
        amp_ratio_rom,
    )  # (you can also return amp_ratio_mor for diagnostics)


def test_if_inference_is_meaningful(
    dataset, K, G, Z, mode="trial-averaged", n_attempts=10
):
    # first, do the inference of the noise parameters on the real data
    tau_kernel, tau_dx, amplitude_ratio = fit_correlation_times_and_amplitude_ratios(
        dataset, K, G, Z, mode=mode
    )
    # Now, generate the synthetic data with the inferred parameters
    tau_dx_from_synth = np.zeros(n_attempts)
    amplitude_ratio_from_synth = np.zeros([n_attempts, G.shape[0], K])
    T = dataset.shape[1]
    n_trials = dataset.shape[0]
    N = dataset.shape[2]
    print("Generating some data to test the inference")
    for attempt in range(n_attempts):
        key = jax.random.PRNGKey(np.random.randint(10**8))
        print("Attempt ", attempt + 1, "out of ", n_attempts)
        synthetic_data = jnp.transpose(
            jax.random.multivariate_normal(
                key, jnp.zeros(T), Z, shape=(n_trials, N), method="svd"
            ),
            (0, 2, 1),
        )  # (n_trials,T,N)
        (
            _,
            tau_dx_from_synth[attempt],
            amplitude_ratio_from_synth[attempt, :, :],
        ) = fit_correlation_times_and_amplitude_ratios(
            synthetic_data, K, G, Z, mode=mode
        )
    # How plot the histogram of the inferred tau_dx from the synthetic data
    fig, ax = plt.subplots(1, 2)
    print(
        "Amplitude ratio from synth:,",
        np.mean(np.mean(amplitude_ratio_from_synth, axis=1), axis=1),
    )
    ax[0].hist(
        np.mean(np.mean(amplitude_ratio_from_synth, axis=1), axis=1),
        bins=30,
        label="amplitude ratio from synthetic data",
    )
    ax[0].axvline(
        np.mean(np.ndarray.flatten(amplitude_ratio)),
        color="r",
        linestyle="dashed",
        linewidth=2,
        label="amplitude ratio from real data",
    )
    ax[0].legend()
    ax[1].hist(
        tau_dx_from_synth,
        bins=np.linspace(0, T, T + 1),
        label="tau_dx from synthetic data",
    )
    ax[1].axvline(
        tau_dx,
        color="r",
        linestyle="dashed",
        linewidth=2,
        label="tau_dx from real data (=" + str(tau_dx) + ")",
    )
    ax[1].legend()
    plt.show()


def extrapolate_potential(
    original: classes.Potential,
    new_neurons: int = None,
    new_trials: int = None,
    mode: str = "trial-averaged",
    existing_number_of_trials: int = None,
    new_bar_e: np.ndarray = None,
    new_bar_sigma: np.ndarray = None,
    new_G: np.ndarray = None,
    random_state: int = 42,
) -> classes.Potential:
    """
    Create a new Potential instance extrapolated to larger number of neurons or trials.

    Parameters:
    - original: The original Potential object
    - new_neurons: New total number of neurons (optional)
    - new_trials: New total number of trials (optional)
    - mode: "trial-averaged" or "trial-concatenated"
    - existing_number_of_trials: Required if extrapolating trials
    - new_bar_e: Optional custom bar_e for new neurons
    - new_bar_sigma: Optional custom bar_sigma for new neurons
    - new_G: Optional custom G matrix
    - random_state: Integer seed for reproducibility

    Returns:
    - New extrapolated Potential object
    """
    rng = np.random.default_rng(random_state)
    pot = copy.deepcopy(original)
    K = pot.bar_e.shape[1]
    N = pot.bar_e.shape[0]
    T = pot.bar_x.shape[0]
    D = pot.G.shape[0]

    # 1. Extrapolate trials first (if requested)
    if new_trials is not None:
        assert (
            existing_number_of_trials is not None
        ), "Must provide existing_number_of_trials when extrapolating trials."
        if mode == "trial-averaged":
            scale = np.sqrt(existing_number_of_trials / new_trials)
            pot.bar_sigma = pot.bar_sigma * scale
            pot.bar_xi = pot.bar_xi * scale
        elif mode == "trial-concatenated":
            assert (
                T % existing_number_of_trials == 0
            ), "Time length not divisible by number of trials."
            T_per_trial = T // existing_number_of_trials
            chunks = np.split(pot.bar_x, existing_number_of_trials, axis=0)
            extra_needed = new_trials - existing_number_of_trials
            new_chunks = rng.choice(chunks, size=extra_needed, replace=True)
            pot.bar_x = np.concatenate(chunks + list(new_chunks), axis=0)
            T_new = pot.bar_x.shape[0]
            pot.Z = jnp.kron(
                jnp.eye(new_trials), original.Z[:T_per_trial, :T_per_trial]
            )
            pot.Delta = jnp.kron(
                jnp.eye(new_trials), original.Delta[:T_per_trial, :T_per_trial]
            )
            pot.Xi = jnp.kron(
                jnp.eye(new_trials), original.Xi[:T_per_trial, :T_per_trial]
            )
        else:
            raise ValueError("Mode must be 'trial-averaged' or 'trial-concatenated'")

    # 2. Extrapolate neurons (if requested)
    if new_neurons is not None and new_neurons > N:
        additional = new_neurons - N

        # Extend bar_e
        if new_bar_e is not None:
            bar_e = np.vstack([pot.bar_e, new_bar_e])
        else:
            extra_e = rng.choice(pot.bar_e, size=additional, replace=True)
            bar_e = np.vstack([pot.bar_e, extra_e])

        # Normalize bar_e: ensure that each e^k is normalized to new number of neurons
        for k in range(K):
            bar_e[:, k] *= np.sqrt(new_neurons) / np.linalg.norm(bar_e[:, k])

        # Extend bar_sigma
        if new_bar_sigma is not None:
            bar_sigma = np.concatenate([pot.bar_sigma, new_bar_sigma])
        else:
            extra_sigma = rng.choice(pot.bar_sigma, size=additional, replace=True)
            bar_sigma = np.concatenate([pot.bar_sigma, extra_sigma])

        # Normalize bar_sigma: since the total per-neuron noise variance is sigma_i^2*N, we have to rescale
        bar_sigma *= np.sqrt(N / new_neurons)

        # Extend G
        if new_G is not None:
            old_D = np.shape(pot.G)[0]
            new_D = np.shape(new_G)[0]
            # Sampling missing xi:
            new_bar_xi = np.zeros([new_D, K])
            for k in range(K):
                new_bar_xi[old_D:, k] = rng.choice(
                    pot.bar_xi[:, k], size=new_D - old_D, replace=True
                )
            new_bar_xi[:old_D, :] = pot.bar_xi
            G = new_G
        else:
            G = np.zeros((D, new_neurons, new_neurons))
            G[:, :N, :N] = pot.G
            # Assign new neurons to the last animal (D-1)
            for i in range(N, new_neurons):
                G[D - 1, i, i] = 1.0

        # Update potential attributes
        pot.bar_e = bar_e
        pot.bar_sigma = bar_sigma
        pot.G = G
        if new_G is not None:
            pot.bar_xi = new_bar_xi

    # Recompute X based on potentially updated bar_x
    pot.update_X()
    return pot
