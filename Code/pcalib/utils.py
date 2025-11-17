import jax.random
import numpy as np
from jax import lax
import jax.numpy as jnp
from numpy.linalg import lstsq


def poissonize(non_convoluted_data):
    key = np.random.randint(10000)
    poisson_process = jax.random.poisson(
        jax.random.PRNGKey(key), lam=non_convoluted_data
    )
    return poisson_process


def approximate_inverse(deltaA, num_terms):
    I = jnp.eye(deltaA.shape[0])
    result = I
    term = I
    for _ in range(1, num_terms):
        term = -deltaA @ term
        result += term
    return result


def phase_randomize(x, rng=None, keep_dc_nyquist=True):
    """
    Independent phase randomization (phase-scrambling) for real-valued signals.

    Parameters
    ----------
    x : array_like
        Real signal of shape (T,) or (n_signals, T).
    rng : np.random.Generator or None
        Random generator to control reproducibility. If None, uses default.
    keep_dc_nyquist : bool
        If True, keep DC (0 Hz) and Nyquist (if T even) phases fixed so the
        output stays exactly real (recommended).

    Returns
    -------
    y : np.ndarray
        Phase-scrambled surrogate(s), same shape as x. Each signal keeps the
        original amplitude spectrum (and thus its autocorrelation), but with
        randomized phases (independently across signals).

    Notes
    -----
    - For real inputs, we use rFFT/irFFT and replace phases at positive freqs.
    - This preserves the power spectrum exactly; differences in time domain
      reflect only phase changes.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        X = np.fft.rfft(x)
        npos = X.shape[0]
        if rng is None:
            rng = np.random.default_rng()
        phases = rng.uniform(0, 2 * np.pi, size=npos)
        if keep_dc_nyquist:
            phases[0] = 0.0
            if x.shape[0] % 2 == 0:
                phases[-1] = 0.0
        Xs = np.abs(X) * np.exp(1j * phases)
        y = np.fft.irfft(Xs, n=x.shape[0]).real
        return y

    elif x.ndim == 2:
        n, T = x.shape
        X = np.fft.rfft(x, axis=1)  # shape (n, npos)
        npos = X.shape[1]
        if rng is None:
            rng = np.random.default_rng()
        phases = rng.uniform(0, 2 * np.pi, size=(n, npos))
        if keep_dc_nyquist:
            phases[:, 0] = 0.0
            if T % 2 == 0:
                phases[:, -1] = 0.0
        Xs = np.abs(X) * np.exp(1j * phases)
        y = np.fft.irfft(Xs, n=T, axis=1).real
        return y

    else:
        raise ValueError("x must be 1D (T,) or 2D (n_signals, T).")


@jax.jit
def PCA_matlab_like(data):
    """
    JAX-compatible PCA that mimics MATLAB behavior and sklearn PCA.
    Returns:
    - coeff: eigenvectors (N, N)
    - score: projection of data onto principal components (T, N)
    - eigs: eigenvalues (N,)
    """
    Tdim, Ndim = data.shape

    # Center data
    data_centered = data - jnp.mean(data, axis=0, keepdims=True)

    # Perform SVD
    U, S, Vh = jnp.linalg.svd(data_centered, full_matrices=False)
    eigs = (S**2) / (Tdim - 1)  # Match sklearn's variance scaling

    coeff = Vh.T  # Principal components
    score = jnp.dot(data_centered, coeff)

    # Pad with orthogonal vectors and zeros if needed (when N > T)
    if Ndim > Tdim:
        # Extend coeff to (N, N) with orthogonal basis
        U_full, _, _ = jnp.linalg.svd(coeff, full_matrices=True)
        extra_basis = U_full[:, Tdim:]  # shape (N, N - T)
        coeff = jnp.hstack([coeff, extra_basis])
        eigs = jnp.concatenate([eigs, jnp.zeros(Ndim - Tdim)])
        score = jnp.hstack([score, jnp.zeros((Tdim, Ndim - Tdim))])

    return coeff, score, eigs


def generate_periodic_exponential_kernel(T, tau, eps=10 ** (-10)):
    if tau == 0:
        kernel = np.zeros(T)
        kernel[0] = 1
        return kernel
    else:
        nper = 0
        while np.exp(-((nper * T) ** 2) / (2 * tau**2)) > eps:
            nper += 1

        kernel = np.exp(-np.arange(T) ** 2 / (2 * tau**2))
        for i in range(1, nper + 1):
            kernel += np.exp(-((i * T + np.arange(T)) ** 2) / (2 * tau**2)) + np.exp(
                -((-i * T + np.arange(T)) ** 2) / (2 * tau**2)
            )

        kernel /= kernel[0]
        return kernel


def generate_gaussian_correlation_matrix(T, tau, eps=10 ** (-10)):
    # Decide how many periods to take into account:
    row = generate_periodic_exponential_kernel(T, tau, eps=10 ** (-10))
    Z = np.array([np.roll(row, t) for t in range(T)])
    Z += Z.T
    return Z / 2


def reduce_to_2d(data, mode="trial-averaged"):
    """
    Reduces a 3D dataset [trials, time, neurons] to 2D for PCA.

    Parameters:
    - data: np.ndarray of shape [trials, time, neurons]
    - mode: "trial-averaged" or "trial-concatenated"

    Returns:
    - 2D array [time, neurons] or [time * trials, neurons]
    """
    if mode == "trial-averaged":
        return np.mean(data, axis=0)  # shape: [time, neurons]
    elif mode == "trial-concatenated":
        trials, time, neurons = data.shape
        return data.transpose(0, 1, 2).reshape(trials * time, neurons)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def convolve_data(data, kernel):
    """
    Convolve data with a kernel using FFT for periodic convolution.

    Parameters:
    - data: np.ndarray of shape [trials, time, neurons]
    - kernel: 1D array of shape [time]

    Returns:
    - Convolved data of the same shape as input
    """
    kernel_fft = np.fft.fft(kernel)
    convolved = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            signal_fft = np.fft.fft(data[i, :, j])
            filtered = np.real(np.fft.ifft(signal_fft * kernel_fft))
            convolved[i, :, j] = filtered
    return convolved


def get_empirical_accuracy_many_attempts(
    n_attempts, full_data, K, G, mode, size_axis, size_values
):
    rho_empirical = np.zeros([n_attempts, np.shape(size_values)[0], K])
    epsilon_empirical = np.zeros([n_attempts, np.shape(size_values)[0], K])
    n_trials = np.shape(full_data)[0]
    for attempt in range(n_attempts):
        full_data_shuffled = full_data[np.random.permutation(n_trials), :, :]
        (
            rho_empirical[attempt, :, :],
            epsilon_empirical[attempt, :, :],
        ) = get_empirical_accuracy_array(
            full_data_shuffled, K, G, mode, size_axis, size_values
        )
    return rho_empirical, epsilon_empirical


def get_empirical_accuracy_array(full_data, K, G, mode, size_axis, size_values):
    """
    Convolve data with a kernel using FFT for periodic convolution.

    Parameters:
    - data: np.ndarray of shape [trials, time, neurons]
    - K: int, dimensionality of latent neural dynamics
    - G: 3D np.array indicating which neurons belong to which animal
    - mode: string - either "trial-averaged" or "trial-concatenated"
    - size_axis: string - either "trials", "neurons" or "animals"
    - size_values: np.ndarray with numerical values for target size

    Returns:
    - rho_array - an array of rho values
    - epsilon_array - an array of epsilon values
    """
    [n_trials_full, T, N_full] = np.shape(full_data)
    D_full = np.shape(G)[0]
    size_axis_legth = np.shape(size_values)[0]
    rho_array = np.zeros([size_axis_legth, K])
    epsilon_array = np.zeros([size_axis_legth, K])

    if size_axis == "trials":
        n_trials_max = n_trials_full // 2
        N = N_full
        if size_values[-1] > n_trials_max:
            raise Exception(
                "Not enough trials in the dataset to use the subsampling technique!"
            )
        # Now, separate the dataset into two parts.
        subdata_1_full = full_data[:n_trials_max, :, :]
        subdata_2_full = full_data[n_trials_max:, :, :]

        for i, n_trials in enumerate(size_values):
            subdata_1 = reduce_to_2d(subdata_1_full[:n_trials, :, :], mode)
            subdata_2 = reduce_to_2d(subdata_2_full[:n_trials, :, :], mode)
            T_eff = subdata_1.shape[0]
            coeff_1, score_1, eigs_1 = PCA_matlab_like(subdata_1)
            coeff_2, score_2, eigs_2 = PCA_matlab_like(subdata_2)
            coeff_1 = np.array(coeff_1)
            coeff_2 = np.array(coeff_2)
            score_1 = np.array(score_1)
            score_2 = np.array(score_2)

            score_1 /= np.sqrt(N)
            score_2 /= np.sqrt(N)
            for k in range(K):
                if np.dot(coeff_1[:, k], coeff_2[:, k]) < 0:
                    coeff_2[:, k] *= -1
                    score_2[:, k] *= -1

                rho_array[i, k] = 1 - np.sqrt(np.dot(coeff_1[:, k], coeff_2[:, k]))
                epsilon_array[i, k] = (
                    1 / (2 * T_eff) * np.sum((score_1[:, k] - score_2[:, k]) ** 2)
                )

            R_array = np.zeros([K, K])
            for k1 in range(K):
                for k2 in range(K):
                    R_array[k1, k2] = np.sqrt(
                        np.abs(np.dot(coeff_1[:, k1], coeff_2[:, k2]))
                    )
            print("R=", R_array)

    elif size_axis == "neurons":
        n_trials = n_trials_full // 2
        if size_values[-1] > N_full:
            raise Exception("Not enough neurons in the dataset!")
        # Now, separate the dataset into two parts.
        subdata_1_full = full_data[:n_trials, :, :]
        subdata_2_full = full_data[n_trials : 2 * n_trials, :, :]

        for i, N in enumerate(size_values):
            subdata_1 = reduce_to_2d(subdata_1_full[:, :, :N], mode)
            subdata_2 = reduce_to_2d(subdata_2_full[:, :, :N], mode)
            T_eff = subdata_1.shape[0]
            coeff_1, score_1, eigs_1 = PCA_matlab_like(subdata_1)
            coeff_2, score_2, eigs_2 = PCA_matlab_like(subdata_2)
            coeff_1 = np.array(coeff_1)
            coeff_2 = np.array(coeff_2)
            score_1 = np.array(score_1)
            score_2 = np.array(score_2)

            score_1 /= np.sqrt(N)
            score_2 /= np.sqrt(N)
            for k in range(K):
                if np.dot(coeff_1[:, k], coeff_2[:, k]) < 0:
                    coeff_2[:, k] *= -1
                    score_2[:, k] *= -1

                rho_array[i, k] = 1 - np.sqrt(np.dot(coeff_1[:, k], coeff_2[:, k]))
                epsilon_array[i, k] = (
                    1 / (2 * T_eff) * np.sum((score_1[:, k] - score_2[:, k]) ** 2)
                )

    elif size_axis == "animals":
        n_trials = n_trials_full // 2
        if size_values[-1] > D_full:
            raise Exception("Not enough animals in the dataset!")
        # Now, separate the dataset into two parts.
        subdata_1_full = full_data[:n_trials, :, :]
        subdata_2_full = full_data[n_trials : 2 * n_trials, :, :]

        for i, D in enumerate(size_values):
            N = np.sum(G[:D, :, :])
            neuron_mask = np.einsum("dii->i", G[:D, :, :]).astype(bool)
            subdata_1 = reduce_to_2d(subdata_1_full[:, :, neuron_mask], mode)
            subdata_2 = reduce_to_2d(subdata_2_full[:, :, neuron_mask], mode)
            T_eff = subdata_1.shape[0]
            coeff_1, score_1, eigs_1 = PCA_matlab_like(subdata_1)
            coeff_2, score_2, eigs_2 = PCA_matlab_like(subdata_2)
            coeff_1 = np.array(coeff_1)
            coeff_2 = np.array(coeff_2)
            score_1 = np.array(score_1)
            score_2 = np.array(score_2)

            score_1 /= np.sqrt(N)
            score_2 /= np.sqrt(N)
            for k in range(K):
                if np.dot(coeff_1[:, k], coeff_2[:, k]) < 0:
                    coeff_2[:, k] *= -1
                    score_2[:, k] *= -1

                rho_array[i, k] = 1 - np.sqrt(np.dot(coeff_1[:, k], coeff_2[:, k]))
                epsilon_array[i, k] = (
                    1 / (2 * T_eff) * np.sum((score_1[:, k] - score_2[:, k]) ** 2)
                )
    else:
        raise ValueError(
            'Please select one of the sizes to iterate from: "trials","neurons" or "animals"'
        )

    return rho_array, epsilon_array
