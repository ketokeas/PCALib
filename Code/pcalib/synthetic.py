import numpy as np

from .classes import Potential
from .utils import generate_gaussian_correlation_matrix


def corridor_signal(T, var_array=(1.0, 1.0), epsilon_corridor=0.1):
    """
    Build a 2D corridor-shaped latent trajectory.

    Parameters
    ----------
    T : int
        Number of time bins.
    var_array : tuple[float, float]
        Target variances of the two latent components.
    epsilon_corridor : float
        Vertical separation between the two corridor halves.

    Returns
    -------
    bar_x : np.ndarray, shape (T, 2)
        Mean-centered latent trajectory with requested per-component variance.
    """
    if T < 4:
        raise ValueError("T must be at least 4.")
    if len(var_array) != 2:
        raise ValueError("var_array must have length 2.")

    bar_x = np.zeros((T, 2), dtype=float)

    # First quarter: horizontal approach
    q1 = T // 4
    q3 = (3 * T) // 4
    bar_x[:q1, 0] = np.linspace(-2, 0, q1)

    # Middle half: curved corridor
    n_mid = q3 - q1
    mid_idx = np.arange(n_mid)
    bar_x[q1:q3, 0] = -np.cos(2 * np.pi * mid_idx / n_mid) + 1
    bar_x[q1:q3, 1] = -np.sin(2 * np.pi * mid_idx / n_mid)

    # Final quarter: horizontal return
    n_end = T - q3
    bar_x[q3:, 0] = np.linspace(0, -2, n_end)

    # Corridor opening
    bar_x[: T // 2, 1] -= epsilon_corridor
    bar_x[T // 2 :, 1] += epsilon_corridor

    # Normalize and set target variances
    bar_x -= np.mean(bar_x, axis=0, keepdims=True)
    std = np.sqrt(np.var(bar_x, axis=0, keepdims=True))
    std = np.where(std == 0, 1.0, std)
    bar_x = bar_x / std
    bar_x[:, 0] *= np.sqrt(var_array[0])
    bar_x[:, 1] *= np.sqrt(var_array[1])

    return bar_x


def make_group_matrix(n_animals, neurons_per_animal):
    """
    Create a grouping tensor G of shape [D, N, N].

    Parameters
    ----------
    n_animals : int
        Number of animals.
    neurons_per_animal : int | list[int]
        Either a single integer (same number for each animal) or a list
        giving the neuron count for each animal.

    Returns
    -------
    G : np.ndarray, shape (D, N, N)
        Binary block-diagonal grouping tensor.
    """
    if isinstance(neurons_per_animal, int):
        counts = [neurons_per_animal] * n_animals
    else:
        counts = list(neurons_per_animal)
        if len(counts) != n_animals:
            raise ValueError("Length of neurons_per_animal must equal n_animals.")

    N = sum(counts)
    G = np.zeros((n_animals, N, N), dtype=float)

    start = 0
    for d, count in enumerate(counts):
        stop = start + count
        G[d, start:stop, start:stop] = np.eye(count)
        start = stop

    return G


def random_orthonormal_loadings(N, K, rng=None, scale_to_sqrt_N=True):
    """
    Draw random orthonormal loadings and optionally scale them so that
    each column has norm sqrt(N).

    Parameters
    ----------
    N : int
        Number of neurons.
    K : int
        Number of latent components.
    rng : np.random.Generator | None
        Random generator.
    scale_to_sqrt_N : bool
        If True, scale columns so ||e^(k)|| = sqrt(N).

    Returns
    -------
    bar_e : np.ndarray, shape (N, K)
        Loading matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    mat = rng.normal(size=(N, K))
    q, _ = np.linalg.qr(mat)

    if scale_to_sqrt_N:
        q = q * np.sqrt(N)

    return q


def build_synthetic_potential(
    T,
    K,
    n_animals,
    neurons_per_animal,
    *,
    bar_x=None,
    signal_type="corridor",
    var_array=None,
    epsilon_corridor=0.1,
    tau_sigma=2.0,
    tau_xi=5.0,
    sigma_mean=1.0,
    sigma_std=0.1,
    xi_mean=None,
    xi_std=None,
    Xi=None,
    rng=None,
):
    """
    Build a synthetic Potential for benchmarking.

    Parameters
    ----------
    T : int
        Number of time bins.
    K : int
        Number of latent components.
    n_animals : int
        Number of animals.
    neurons_per_animal : int | list[int]
        Neuron counts per animal.
    bar_x : np.ndarray | None
        Optional latent trajectory of shape (T, K). If provided, overrides
        signal_type / var_array / epsilon_corridor.
    signal_type : str
        Currently supports "corridor" for K=2.
    var_array : sequence | None
        Target signal variances. If None, defaults to ones.
    epsilon_corridor : float
        Corridor opening parameter for the built-in 2D trajectory.
    tau_sigma : float
        Correlation width for fast noise kernel.
    tau_xi : float
        Correlation width for trial-to-trial variability kernel.
    sigma_mean : float
        Mean of per-neuron noise amplitude before abs().
    sigma_std : float
        Std of per-neuron noise amplitude before abs().
    xi_mean : sequence | None
        Mean trial-to-trial variability amplitudes per component.
        If None, uses 2/(k+1).
    xi_std : sequence | None
        Std of trial-to-trial variability amplitudes per component.
        If None, uses 0.1/(k+1).
    Xi : np.ndarray | None
        Optional neuron-dependent temporal kernel matrix.
    rng : np.random.Generator | None
        Random generator.

    Returns
    -------
    pot : Potential
        Synthetic model instance.
    """
    if rng is None:
        rng = np.random.default_rng()

    G = make_group_matrix(n_animals, neurons_per_animal)
    N = G.shape[1]

    if bar_x is None:
        if signal_type != "corridor":
            raise ValueError(f"Unknown signal_type: {signal_type}")
        if K != 2:
            raise ValueError('signal_type="corridor" currently requires K=2.')
        if var_array is None:
            var_array = (1.0, 1.0)
        bar_x = corridor_signal(
            T=T,
            var_array=var_array,
            epsilon_corridor=epsilon_corridor,
        )
    else:
        bar_x = np.asarray(bar_x, dtype=float)
        if bar_x.shape != (T, K):
            raise ValueError(f"bar_x must have shape ({T}, {K}).")

    bar_e = random_orthonormal_loadings(N, K, rng=rng, scale_to_sqrt_N=True)

    bar_sigma = np.abs(rng.normal(loc=sigma_mean, scale=sigma_std, size=N))

    if xi_mean is None:
        xi_mean = [2.0 / (k + 1) for k in range(K)]
    if xi_std is None:
        xi_std = [0.1 / (k + 1) for k in range(K)]

    xi_mean = np.asarray(xi_mean, dtype=float)
    xi_std = np.asarray(xi_std, dtype=float)
    if xi_mean.shape != (K,) or xi_std.shape != (K,):
        raise ValueError("xi_mean and xi_std must have shape (K,).")

    bar_xi = np.zeros((n_animals, K), dtype=float)
    for k in range(K):
        bar_xi[:, k] = np.sqrt(
            np.abs(rng.normal(loc=xi_mean[k], scale=xi_std[k], size=n_animals))
        )

    Z = generate_gaussian_correlation_matrix(T, tau_sigma * np.sqrt(2))
    Delta = generate_gaussian_correlation_matrix(T, tau_xi)

    if Xi is None:
        Xi = np.zeros((T, T), dtype=float)
    else:
        Xi = np.asarray(Xi, dtype=float)
        if Xi.shape != (T, T):
            raise ValueError(f"Xi must have shape ({T}, {T}).")

    return Potential(bar_sigma, bar_e, G, bar_xi, Z, Delta, bar_x, Xi)


def build_corridor_potential(
    T,
    n_animals,
    neurons_per_animal,
    *,
    var_array=(1.0, 1.0),
    epsilon_corridor=0.1,
    tau_sigma=2.0,
    tau_xi=5.0,
    sigma_mean=1.0,
    sigma_std=0.1,
    xi_mean=None,
    xi_std=None,
    Xi=None,
    rng=None,
):
    """
    Convenience wrapper for building a 2D corridor synthetic Potential.

    Parameters
    ----------
    T : int
        Number of time bins.
    n_animals : int
        Number of animals.
    neurons_per_animal : int | list[int]
        Neuron counts per animal.
    var_array : tuple[float, float]
        Target variances of the two latent components.
    epsilon_corridor : float
        Vertical separation between the two corridor halves.
    tau_sigma : float
        Correlation width for fast noise kernel.
    tau_xi : float
        Correlation width for trial-to-trial variability kernel.
    sigma_mean : float
        Mean of per-neuron noise amplitude before abs().
    sigma_std : float
        Std of per-neuron noise amplitude before abs().
    xi_mean : sequence | None
        Mean trial-to-trial variability amplitudes per component.
    xi_std : sequence | None
        Std trial-to-trial variability amplitudes per component.
    Xi : np.ndarray | None
        Optional neuron-dependent temporal kernel matrix.
    rng : np.random.Generator | None
        Random generator.

    Returns
    -------
    pot : Potential
        Synthetic corridor-model instance.
    """
    return build_synthetic_potential(
        T=T,
        K=2,
        n_animals=n_animals,
        neurons_per_animal=neurons_per_animal,
        signal_type="corridor",
        var_array=var_array,
        epsilon_corridor=epsilon_corridor,
        tau_sigma=tau_sigma,
        tau_xi=tau_xi,
        sigma_mean=sigma_mean,
        sigma_std=sigma_std,
        xi_mean=xi_mean,
        xi_std=xi_std,
        Xi=Xi,
        rng=rng,
    )