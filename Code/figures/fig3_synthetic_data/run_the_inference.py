# run_the_inference_fig3.py  (3-series infer→extrapolate + per-neuron rho with capped appends)
import os
import sys
from pathlib import Path

import numpy as np

# --- package imports ---
sys.path.insert(0, os.path.relpath("../../"))

from pcalib.synthetic import build_corridor_potential
from pcalib.benchmarks import extrapolate_three_series
from pcalib.utils import (
    PCA_matlab_like,
    align_pca_to_reference,
    get_empirical_accuracy_array,
    reduce_to_2d,
    to_scalar,
)
from pcalib.functions import make_predictions, extrapolate_potential

# ----------------------- Config -----------------------
OUT = Path("cached_results")
OUT.mkdir(exist_ok=True)
OUT_BAD_SNR = Path("cached_results/bad_SNR")
OUT_BAD_SNR.mkdir(exist_ok=True)

T = 100
K = 2
D = 2
N_per_animal = 50
N = D * N_per_animal

TAU_SIGMA = 1.0
TAU_XI = 7.0
EPS_CORRIDOR = 0.1

# Shared trials axis (x-axis for most figures)
n_trials_array = np.arange(5, 55, 5)  # 5..50 inclusive
L = len(n_trials_array)

# Small dataset sizes for parameter inference (THREE SERIES)
BASE_TRIALS_SERIES = np.array([5, 10, 15], dtype=int)
S = len(BASE_TRIALS_SERIES)

# Example sizes for panel C trajectories
TRIALS_REF_SMALL = 5
TRIALS_REF_LARGE = 50

# Empirical error bars: how many independent datasets to average over
N_ATTEMPTS_EMPIRICAL = 100

# Theoretical error bars: maximum number of infer→extrapolate attempts to keep
N_ATTEMPTS_THEORETICAL = 2

RNG = np.random.default_rng(1234)


# ----------------------- helpers: capped append/resume -----------------------
def attempts_done(path: Path) -> int:
    """Return #rows (axis 0) if file exists, else 0."""
    return np.load(path).shape[0] if path.exists() else 0


def append_rows_capped(path: Path, new_block: np.ndarray, cap: int) -> None:
    """
    Append new rows on axis 0, but ensure saved file has <= cap rows total.
    new_block must have shape [num_new, ...].
    Overwrites the file in place (idempotent).
    """
    if new_block is None:
        return
    new_block = np.asarray(new_block)
    if new_block.shape[0] == 0:
        return

    if path.exists():
        old = np.load(path)
        need = max(0, cap - old.shape[0])
        if need == 0:
            return
        out = np.concatenate([old, new_block[:need]], axis=0)
    else:
        out = new_block[:cap]

    np.save(path, out)


def save_np(name: str, arr, suffix: str = "") -> None:
    np.save(OUT / f"{name}{suffix}.npy", arr)


# ----------------------- synthetic helper -----------------------
def build_true_potential(var_array=(4.0, 1.0), sigma_mu=1.0, sigma_sd=0.1):
    """
    Potential used to generate synthetic data (ground truth) for this figure.
    """
    return build_corridor_potential(
        T=T,
        n_animals=D,
        neurons_per_animal=N_per_animal,
        var_array=var_array,
        epsilon_corridor=EPS_CORRIDOR,
        tau_sigma=TAU_SIGMA,
        tau_xi=TAU_XI,
        sigma_mean=sigma_mu,
        sigma_std=sigma_sd,
        rng=RNG,
    )


# ----------------------- empirical accuracy -----------------------
def empirical_accuracy_curves(
    true_pot, targets, attempts=N_ATTEMPTS_EMPIRICAL, mode="trial-averaged"
):
    """
    Compute empirical rho/epsilon vs trials using split-half via get_empirical_accuracy_array.

    Returns
    -------
    epsilon_emp_mean, epsilon_emp_std, rho_emp_mean, rho_emp_std
        Arrays of shape [len(targets), K].
    """
    eps_all = []
    rho_all = []
    max_tr = int(targets.max())

    for _ in range(attempts):
        # Need at least 2*max_tr trials for split-half
        full_data = true_pot.generate_sample_data(n_samples=2 * max_tr)
        rho_emp, eps_emp = get_empirical_accuracy_array(
            full_data,
            K=K,
            G=true_pot.G,
            mode=mode,
            size_axis="trials",
            size_values=targets,
        )
        rho_all.append(rho_emp)
        eps_all.append(eps_emp)

    rho_all = np.stack(rho_all, axis=0)  # [attempts, len, K]
    eps_all = np.stack(eps_all, axis=0)

    # Average rho in the overlap-squared space, then map back
    R_all_squared = (1 - rho_all) ** 2
    mean_rho = 1 - np.sqrt(np.mean(R_all_squared, axis=0))

    return eps_all.mean(axis=0), eps_all.std(axis=0), mean_rho, rho_all.std(axis=0)


# ----------------------- one-time assets -----------------------
def save_one_time_assets(true_pot):
    """Save constants/true curves that don't depend on attempts."""
    sqrt_mean_sigma_true = np.sqrt(np.mean(true_pot.bar_sigma**2) / n_trials_array)
    xi_true_tiled = np.einsum("bc,a->abc", true_pot.bar_xi, 1 / np.sqrt(n_trials_array))

    save_np("n_trials_array", n_trials_array)
    save_np("sqrt_mean_sigma_true", sqrt_mean_sigma_true)
    save_np("xi_true", xi_true_tiled)

    true_pot.save_as_npz(OUT / "potential_true")


# ----------------------- MAIN routine with capped appends -----------------------
def run_regular():
    # 0) Build ground-truth potential (deterministic for given RNG seed)
    pot_true = build_true_potential(var_array=(4.0, 1.0))
    print(
        "True xi to sigma ratios:",
        pot_true.bar_xi**2 / np.mean(pot_true.bar_sigma**2),
    )

    # 1) Save constants/true curves (safe to overwrite)
    save_one_time_assets(pot_true)

    # 2) Figure out how many THEORY attempts already saved
    f_sig_attempts = OUT / "sqrt_mean_sigma_extrap_attempts.npy"   # (A, S, L)
    f_xi_attempts = OUT / "xi_extrap_attempts.npy"                 # (A, S, L, D, K)
    f_eps_attempts = OUT / "epsilon_pred_attempts.npy"             # (A, S, L, K)
    f_rho_attempts = OUT / "rho_pred_attempts.npy"                 # (A, S, L, N, K)

    done_theory = max(
        attempts_done(f_sig_attempts),
        attempts_done(f_xi_attempts),
        attempts_done(f_eps_attempts),
        attempts_done(f_rho_attempts),
    )
    remaining = max(0, N_ATTEMPTS_THEORETICAL - done_theory)
    print(
        f"[theory] attempts done: {done_theory} / {N_ATTEMPTS_THEORETICAL} (remaining {remaining})"
    )

    # 3) Add new THEORY attempts, appending rows up to the cap
    for a in range(remaining):
        attempt_idx = done_theory + a + 1
        print(f"[theory] generating attempt {attempt_idx}")

        theory = extrapolate_three_series(
            pot_true,
            K=K,
            gaussian_kernel_width=TAU_SIGMA,
            base_trials_series=BASE_TRIALS_SERIES,
            target_trials=n_trials_array,
            mode="trial-averaged",
            gamma=0.05,
            method="diagonal",
        )

        sig_ex_S = theory["sqrt_mean_sigma"]
        xi_ex_S = theory["xi"]
        eps_ex_S = theory["epsilon"]
        rho_ex_S = theory["rho"]

        # Save first attempt as single-run reference
        if attempt_idx == 1:
            save_np("sqrt_mean_sigma_extrap", sig_ex_S)  # (S, L)
            save_np("xi_extrap", xi_ex_S)                # (S, L, D, K)
            save_np("epsilon_pred", eps_ex_S)            # (S, L, K)
            save_np("rho_pred", rho_ex_S)                # (S, L, N, K)

        append_rows_capped(
            f_sig_attempts, sig_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL
        )
        append_rows_capped(
            f_xi_attempts, xi_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL
        )
        append_rows_capped(
            f_eps_attempts, eps_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL
        )
        append_rows_capped(
            f_rho_attempts, rho_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL
        )

    # 4) Aggregate theory attempts for plotting
    if f_sig_attempts.exists():
        sig_all = np.load(f_sig_attempts)   # (A, S, L)
        xi_all = np.load(f_xi_attempts)     # (A, S, L, D, K)
        eps_all = np.load(f_eps_attempts)   # (A, S, L, K)
        rho_all = np.load(f_rho_attempts)   # (A, S, L, N, K)

        save_np("sqrt_mean_sigma_theory_mean", np.nanmean(sig_all, axis=0))  # (S, L)
        save_np("sqrt_mean_sigma_theory_std", np.nanstd(sig_all, axis=0))    # (S, L)

        save_np("xi_theory_mean", np.nanmean(xi_all, axis=0))                # (S, L, D, K)
        save_np("xi_theory_std", np.nanstd(xi_all, axis=0))                  # (S, L, D, K)

        save_np("epsilon_theory_mean", np.nanmean(eps_all, axis=0))          # (S, L, K)
        save_np("epsilon_theory_std", np.nanstd(eps_all, axis=0))            # (S, L, K)

        save_np("rho_theory_mean", np.nanmean(rho_all, axis=0))              # (S, L, N, K)
        save_np("rho_theory_std", np.nanstd(rho_all, axis=0))                # (S, L, N, K)

    # 5) Empirical accuracy curves
    print("Calculating empirical accuracy...")
    eps_mean, eps_std, rho_mean, rho_std = empirical_accuracy_curves(
        pot_true, n_trials_array, attempts=N_ATTEMPTS_EMPIRICAL, mode="trial-averaged"
    )

    save_np("epsilon_emp_mean", eps_mean)  # (L, K)
    save_np("epsilon_emp_std", eps_std)    # (L, K)
    save_np("rho_emp_mean", rho_mean)      # (L, K)
    save_np("rho_emp_std", rho_std)        # (L, K)

    # 6) Panel C assets
    print("Saving Panel C assets:")
    save_np("bar_x_true", pot_true.bar_x)

    for ntr in (TRIALS_REF_SMALL, TRIALS_REF_LARGE):
        data = pot_true.generate_sample_data(n_samples=ntr)
        coeff, score, _ = PCA_matlab_like(np.mean(data, axis=0))
        _, aligned_score, _ = align_pca_to_reference(
            coeff,
            pot_true.bar_e,
            score=score,
            scale_scores=1 / np.sqrt(N),
        )
        save_np(f"inferred_y_{ntr}", aligned_score)

    # epsilon_ref (per-component at two trial counts)
    print("Reference values of epsilon for the bar plots:")
    theory_ref = extrapolate_three_series(
        pot_true,
        K=K,
        gaussian_kernel_width=TAU_SIGMA,
        base_trials_series=[TRIALS_REF_SMALL],
        target_trials=[TRIALS_REF_SMALL, TRIALS_REF_LARGE],
        mode="trial-averaged",
        gamma=0.05,
        method="diagonal",
    )
    epsilon_ref = theory_ref["epsilon"][0].T  # shape (K, 2)
    save_np("epsilon_ref", epsilon_ref)

    # 7) Direct PCA-at-50 block (no split-half), per-neuron rho_i^{(k)}
    print("Calculating direct per-neuron rho at n=50 via PCA on reduce_to_2d...")
    attempts = N_ATTEMPTS_EMPIRICAL * 10
    rho_emp_PCA50 = np.full((attempts, N, K), np.nan, dtype=float)

    bar_e_norm = np.array(pot_true.bar_e, copy=True)
    for k in range(K):
        bar_e_norm[:, k] = (
            bar_e_norm[:, k] / np.linalg.norm(bar_e_norm[:, k]) * np.sqrt(N)
        )

    for a in range(attempts):
        data = pot_true.generate_sample_data(n_samples=50)
        X = reduce_to_2d(data, mode="trial-averaged")
        coeff, _, _ = PCA_matlab_like(X)

        aligned_v, _, _ = align_pca_to_reference(
            coeff,
            bar_e_norm,
            scale_loadings=np.sqrt(N),
        )

        for k in range(K):
            rho_emp_PCA50[a, :, k] = 0.5 * (bar_e_norm[:, k] - aligned_v[:, k]) ** 2

    save_np("rho_emp_per_neuron_PCA50_attempts", rho_emp_PCA50)
    save_np("rho_emp_per_neuron_PCA50_mean", np.nanmean(rho_emp_PCA50, axis=0))
    save_np("rho_emp_per_neuron_PCA50_std", np.nanstd(rho_emp_PCA50, axis=0))


# ----------------------- "Bad signal" routine with capped appends -----------------------
def run_bad_SNR():
    pot_true_bad_SNR = build_true_potential(var_array=(4.0, 0.05))
    pot_true_bad_SNR.save_as_npz("cached_results/bad_SNR/pot_true_bad_SNR")

    f_sig_attempts = OUT / "bad_SNR/sqrt_mean_sigma_extrap_attempts.npy"
    f_xi_attempts = OUT / "bad_SNR/xi_extrap_attempts.npy"
    f_eps_attempts = OUT / "bad_SNR/epsilon_pred_attempts.npy"
    f_rho_attempts = OUT / "bad_SNR/rho_pred_attempts.npy"

    done_theory = max(
        attempts_done(f_sig_attempts),
        attempts_done(f_xi_attempts),
        attempts_done(f_eps_attempts),
        attempts_done(f_rho_attempts),
    )
    remaining = max(0, N_ATTEMPTS_THEORETICAL - done_theory)
    print(
        f"[theory] attempts done: {done_theory} / {N_ATTEMPTS_THEORETICAL} (remaining {remaining})"
    )

    for a in range(remaining):
        attempt_idx = done_theory + a + 1
        print(f"[theory] generating attempt {attempt_idx}")

        theory = extrapolate_three_series(
            pot_true_bad_SNR,
            K=K,
            gaussian_kernel_width=TAU_SIGMA,
            base_trials_series=BASE_TRIALS_SERIES,
            target_trials=n_trials_array,
            mode="trial-averaged",
            gamma=0.01,
            method="diagonal",
        )

        sig_ex_S = theory["sqrt_mean_sigma"]
        xi_ex_S = theory["xi"]
        eps_ex_S = theory["epsilon"]
        rho_ex_S = theory["rho"]

        if attempt_idx == 1:
            save_np("bad_SNR/sqrt_mean_sigma_extrap", sig_ex_S)
            save_np("bad_SNR/xi_extrap", xi_ex_S)
            save_np("bad_SNR/epsilon_pred", eps_ex_S)
            save_np("bad_SNR/rho_pred", rho_ex_S)

        append_rows_capped(
            f_sig_attempts, sig_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL
        )
        append_rows_capped(
            f_xi_attempts, xi_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL
        )
        append_rows_capped(
            f_eps_attempts, eps_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL
        )
        append_rows_capped(
            f_rho_attempts, rho_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL
        )

    if f_sig_attempts.exists():
        sig_all = np.load(f_sig_attempts)
        xi_all = np.load(f_xi_attempts)
        eps_all = np.load(f_eps_attempts)
        rho_all = np.load(f_rho_attempts)

        np.save(OUT / "bad_SNR/sqrt_mean_sigma_theory_mean.npy", np.nanmean(sig_all, axis=0))
        np.save(OUT / "bad_SNR/sqrt_mean_sigma_theory_std.npy", np.nanstd(sig_all, axis=0))

        np.save(OUT / "bad_SNR/xi_theory_mean.npy", np.nanmean(xi_all, axis=0))
        np.save(OUT / "bad_SNR/xi_theory_std.npy", np.nanstd(xi_all, axis=0))

        np.save(OUT / "bad_SNR/epsilon_theory_mean.npy", np.nanmean(eps_all, axis=0))
        np.save(OUT / "bad_SNR/epsilon_theory_std.npy", np.nanstd(eps_all, axis=0))

        np.save(OUT / "bad_SNR/rho_theory_mean.npy", np.nanmean(rho_all, axis=0))
        np.save(OUT / "bad_SNR/rho_theory_std.npy", np.nanstd(rho_all, axis=0))

    print("Calculating empirical accuracy...")
    eps_mean, eps_std, rho_mean, rho_std = empirical_accuracy_curves(
        pot_true_bad_SNR,
        n_trials_array,
        attempts=N_ATTEMPTS_EMPIRICAL,
        mode="trial-averaged",
    )

    np.save(OUT / "bad_SNR/epsilon_emp_mean.npy", eps_mean)
    np.save(OUT / "bad_SNR/epsilon_emp_std.npy", eps_std)
    np.save(OUT / "bad_SNR/rho_emp_mean.npy", rho_mean)
    np.save(OUT / "bad_SNR/rho_emp_std.npy", rho_std)


# ----------------------- Entrypoint -----------------------
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT_BAD_SNR, exist_ok=True)
    run_regular()
    run_bad_SNR()