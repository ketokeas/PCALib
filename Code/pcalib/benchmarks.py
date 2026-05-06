import numpy as np

from .functions import (
    fit_statistics_from_dataset,
    fit_statistics_from_dataset_diagonal,
    make_predictions,
    extrapolate_potential,
)
from .utils import to_scalar


def do_single_small_inference(
    true_potential,
    K,
    gaussian_kernel_width,
    *,
    n_trials_small,
    mode="trial-averaged",
    gamma=0.05,
    method="diagonal",
):
    """
    Generate one synthetic dataset from a ground-truth Potential and fit a
    benchmark model from it.

    Parameters
    ----------
    true_potential : Potential
        Ground-truth synthetic model used to generate data.
    K : int
        Number of latent components.
    gaussian_kernel_width : float
        Kernel width passed to the fitting routine.
    n_trials_small : int
        Number of trials in the pilot dataset used for fitting.
    mode : str
        "trial-averaged" or "trial-concatenated".
    gamma : float
        Step size / damping for the diagonal fitter.
    method : str
        Currently only "diagonal" is implemented.

    Returns
    -------
    fitted_model
        For method="diagonal", a list of length K containing one fitted
        Potential per component.

    Notes
    -----
    Benchmark workflows currently support only the diagonal approximation.
    """
    data = true_potential.generate_sample_data(n_samples=n_trials_small)

    if method == "diagonal":
        pots_diag, _ = fit_statistics_from_dataset_diagonal(
            data,
            K,
            true_potential.G,
            gaussian_kernel_width,
            mode=mode,
            gamma=gamma,
        )
        return pots_diag

    if method == "full":
        raise NotImplementedError(
            "Full-model benchmark workflows are not implemented yet."
        )

    raise ValueError(f"Unknown method: {method}")


def extrapolate_predictions_from_small(
    fitted_small,
    *,
    n_trials_small,
    target_trials,
    mode="trial-averaged",
    method="diagonal",
):
    """
    Extrapolate a fitted benchmark model from one pilot dataset size to a list
    of target trial counts.

    Parameters
    ----------
    fitted_small
        Output of do_single_small_inference(...).
        For method="diagonal", this must be a list of fitted one-component
        potentials, one per latent component.
    n_trials_small : int
        Number of trials used for the initial fit.
    target_trials : array_like
        Trial counts to extrapolate to.
    mode : str
        "trial-averaged" or "trial-concatenated".
    method : str
        Currently only "diagonal" is implemented.

    Returns
    -------
    result : dict
        For method="diagonal", dictionary with keys:
        - "sqrt_mean_sigma": shape (L,)
        - "xi": shape (L, D, K)
        - "epsilon": shape (L, K)
        - "rho": shape (L, N, K)
    """
    target_trials = np.asarray(target_trials, dtype=int)

    if method != "diagonal":
        if method == "full":
            raise NotImplementedError(
                "Full-model benchmark workflows are not implemented yet."
            )
        raise ValueError(f"Unknown method: {method}")

    pots_small = fitted_small
    K = len(pots_small)

    sqrt_mean_sigma_list = []
    xi_list = []
    epsilon_list = []
    rho_list = []

    for ntr in target_trials:
        sigmas_this = []
        xi_cols = []
        epsilon_this = []
        rho_cols = []

        for k in range(K):
            pot_small = pots_small[k]
            pot_xt = extrapolate_potential(
                original=pot_small,
                new_trials=int(ntr),
                existing_number_of_trials=int(n_trials_small),
                mode=mode,
            )

            sigmas_this.append(np.sqrt(np.mean(np.asarray(pot_xt.bar_sigma) ** 2)))

            xi_k = np.asarray(pot_xt.bar_xi)
            if xi_k.ndim == 1:
                xi_k = xi_k[:, np.newaxis]
            elif xi_k.shape[1] != 1:
                xi_k = xi_k[:, [k]]
            xi_cols.append(xi_k)

            pred = make_predictions(pot_xt)
            epsilon_this.append(to_scalar(pred["epsilon"]))

            rho_k = np.asarray(pred["rho"])[:, 0, 0]
            rho_cols.append(rho_k)

        sqrt_mean_sigma_list.append(np.mean(sigmas_this))
        xi_list.append(np.concatenate(xi_cols, axis=1))
        epsilon_list.append(np.array(epsilon_this))
        rho_list.append(np.stack(rho_cols, axis=1))

    return {
        "sqrt_mean_sigma": np.array(sqrt_mean_sigma_list),  # (L,)
        "xi": np.stack(xi_list, axis=0),                    # (L, D, K)
        "epsilon": np.stack(epsilon_list, axis=0),          # (L, K)
        "rho": np.stack(rho_list, axis=0),                  # (L, N, K)
    }


def extrapolate_three_series(
    true_potential,
    *,
    K,
    gaussian_kernel_width,
    base_trials_series,
    target_trials,
    mode="trial-averaged",
    gamma=0.05,
    method="diagonal",
):
    """
    Run infer-then-extrapolate from several pilot dataset sizes and stack the
    resulting theory series.

    Parameters
    ----------
    true_potential : Potential
        Ground-truth synthetic model used to generate synthetic pilot datasets.
    K : int
        Number of latent components.
    gaussian_kernel_width : float
        Kernel width passed to the fitting routine.
    base_trials_series : array_like
        Pilot trial counts used for the initial fits, e.g. [5, 10, 15].
    target_trials : array_like
        Trial counts to extrapolate to.
    mode : str
        "trial-averaged" or "trial-concatenated".
    gamma : float
        Step size / damping for the diagonal fitter.
    method : str
        Currently only "diagonal" is implemented.

    Returns
    -------
    result : dict
        For method="diagonal", dictionary with keys:
        - "sqrt_mean_sigma": shape (S, L)
        - "xi": shape (S, L, D, K)
        - "epsilon": shape (S, L, K)
        - "rho": shape (S, L, N, K)

        For any series s, entries with target_trials < base_trials_series[s]
        are filled with NaN.
    """
    base_trials_series = np.asarray(base_trials_series, dtype=int)
    target_trials = np.asarray(target_trials, dtype=int)

    if method != "diagonal":
        if method == "full":
            raise NotImplementedError(
                "Full-model benchmark workflows are not implemented yet."
            )
        raise ValueError(f"Unknown method: {method}")

    S = len(base_trials_series)
    L = len(target_trials)
    D = true_potential.G.shape[0]
    N = true_potential.bar_e.shape[0]

    sqrt_mean_sigma_all = np.full((S, L), np.nan, dtype=float)
    xi_all = np.full((S, L, D, K), np.nan, dtype=float)
    epsilon_all = np.full((S, L, K), np.nan, dtype=float)
    rho_all = np.full((S, L, N, K), np.nan, dtype=float)

    for s, n_trials_small in enumerate(base_trials_series):
        fitted_small = do_single_small_inference(
            true_potential,
            K,
            gaussian_kernel_width,
            n_trials_small=int(n_trials_small),
            mode=mode,
            gamma=gamma,
            method=method,
        )

        extrap = extrapolate_predictions_from_small(
            fitted_small,
            n_trials_small=int(n_trials_small),
            target_trials=target_trials,
            mode=mode,
            method=method,
        )

        valid = target_trials >= n_trials_small
        sqrt_mean_sigma_all[s, valid] = extrap["sqrt_mean_sigma"][valid]
        xi_all[s, valid, :, :] = extrap["xi"][valid]
        epsilon_all[s, valid, :] = extrap["epsilon"][valid]
        rho_all[s, valid, :, :] = extrap["rho"][valid]

    return {
        "sqrt_mean_sigma": sqrt_mean_sigma_all,
        "xi": xi_all,
        "epsilon": epsilon_all,
        "rho": rho_all,
    }