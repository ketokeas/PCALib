# run_the_inference_fig3.py  (3-series infer→extrapolate + per-neuron rho with capped appends)
import os, sys
import numpy as np
from pathlib import Path

# --- package imports ---
sys.path.insert(0, os.path.relpath("../../"))
from pcalib.classes import Potential
from pcalib.utils import PCA_matlab_like, generate_gaussian_correlation_matrix, get_empirical_accuracy_array, \
    reduce_to_2d
from pcalib.functions import fit_statistics_from_dataset_diagonal, make_predictions, extrapolate_potential

# ----------------------- Config -----------------------
OUT = Path("cached_results"); OUT.mkdir(exist_ok=True)
OUT_BAD_SNR = Path("cached_results/bad_SNR"); OUT.mkdir(exist_ok=True)

T = 100
K = 2
D = 2
N_per_animal = 50
N = D * N_per_animal

TAU_SIGMA = 1.0
TAU_XI    = 7.0
EPS_CORRIDOR = 0.1

# Shared trials axis (x-axis for most figures)
n_trials_array = np.arange(5, 55, 5)   # 5..50 inclusive
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
N_ATTEMPTS_THEORETICAL = 10

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
            return  # already at cap
        out = np.concatenate([old, new_block[:need]], axis=0)
    else:
        out = new_block[:cap]
    np.save(path, out)

def save_np(name: str, arr, suffix: str = "") -> None:
    np.save(OUT / f"{name}{suffix}.npy", arr)

def to_scalar(x):
    a = np.asarray(x)
    return float(a.reshape(-1)[0])

# ----------------------- synthetic helpers -----------------------
def corridor_signal(T, var_array, epsilon_corridor):
    sig = np.zeros((T, 2))
    sig[:T//4, 0] = np.linspace(-2, 0, T//4)
    mid = np.arange(T//4, (3*T)//4); n_mid = mid.size
    sig[T//4:(3*T)//4, 0] = -np.cos(2*np.pi*np.arange(n_mid)/n_mid) + 1
    sig[T//4:(3*T)//4, 1] = -np.sin(2*np.pi*np.arange(n_mid)/n_mid)
    n_end = T - (3*T)//4
    sig[(3*T)//4:, 0] = np.linspace(0, -2, n_end)
    sig[:T//2, 1] -= epsilon_corridor
    sig[T//2:, 1] += epsilon_corridor
    sig -= sig.mean(axis=0, keepdims=True)
    sig /= np.sqrt(sig.var(axis=0, keepdims=True))
    sig[:, 0] *= np.sqrt(var_array[0])
    sig[:, 1] *= np.sqrt(var_array[1])
    return sig

def make_G(D, N_per_animal):
    N = D * N_per_animal
    G = np.zeros((D, N, N))
    for d in range(D):
        s = d * N_per_animal
        G[d, s:s+N_per_animal, s:s+N_per_animal] = np.eye(N_per_animal)
    return G

def build_true_potential(var_array=(4.0, 1.0), sigma_mu=1.0, sigma_sd=0.1):
    """Potential used to GENERATE synthetic data (ground truth) for this figure."""
    bar_x = corridor_signal(T, var_array, EPS_CORRIDOR)   # [T, K]
    # loadings
    bar_e = RNG.normal(0, 1, size=(N, K))
    bar_e, _ = np.linalg.qr(bar_e)
    bar_e *= np.sqrt(N)
    # per-neuron noise
    bar_sigma = np.abs(RNG.normal(sigma_mu, sigma_sd, size=N))
    # per-animal per-component trial-to-trial variability
    bar_xi = np.reshape(np.linspace(0.5, 1, D*K), (D, K))
    # structure
    G = make_G(D, N_per_animal)
    Z = generate_gaussian_correlation_matrix(T, TAU_SIGMA*np.sqrt(2))
    Delta = generate_gaussian_correlation_matrix(T, TAU_XI)
    Xi = np.zeros((T, T))
    return Potential(bar_sigma, bar_e, G, bar_xi, Z, Delta, bar_x, Xi)

# ----------------------- core pipeline primitives -----------------------
def do_single_small_inference(true_pot, n_trials_small=TRIALS_REF_SMALL, mode="trial-averaged"):
    """Generate ONE small dataset and fit diagonal potentials (one per PC)."""
    data = true_pot.generate_sample_data(n_samples=n_trials_small)
    pots_diag, _ = fit_statistics_from_dataset_diagonal(
        data, K, true_pot.G, TAU_SIGMA, gamma=0.05
    )
    return pots_diag  # list length K (one Potential per PC)

def extrapolate_predictions_from_small(pots_small, n_trials_small, targets, mode="trial-averaged"):
    """
    Extrapolate from a single small fitted set to each target trials count.
    Returns:
      sqrt_mean_sigma_extrap: (L,)
      xi_extrap:              (L, D, K)
      epsilon_pred:           (L, K)
      rho_pred_diag_per_neuron: (L, N, K)  <-- NEW: store per-neuron diagonal of rho[N,K,K]
    """
    sig_list = []
    xi_list  = []
    eps_list = []
    rho_list = []  # will hold (N,K) per target

    for ntr in targets:
        sigs_this = []
        xi_cols   = []   # list of K arrays each (D,1)
        eps_this  = []
        rho_cols_per_k = []  # list of K arrays each (N,)

        for k in range(K):
            pot_sm = pots_small[k]
            pot_xt = extrapolate_potential(
                original=pot_sm,
                new_trials=ntr,
                existing_number_of_trials=n_trials_small,
                mode=mode,
            )
            # noise scalar for this component's extrapolated pot
            sigs_this.append(np.sqrt(np.mean(pot_xt.bar_sigma**2)))

            # collect xi column for this component
            xi_k = np.asarray(pot_xt.bar_xi)
            if xi_k.ndim == 1:
                xi_k = xi_k.reshape(-1, 1)          # (D,) -> (D,1)
            elif xi_k.shape[1] != 1:
                xi_k = xi_k[:, [k]]
            xi_cols.append(xi_k)

            # predictions
            pred = make_predictions(pot_xt)
            eps_this.append(to_scalar(pred["epsilon"]))  # scalar per k (unchanged)

            # rho now [N,K,K] — store diagonal slice for this k: (N,)
            rho_nkk = np.asarray(pred["rho"])[:, 0, 0]
            rho_cols_per_k.append(rho_nkk)

        # average the sigma scalars across components (one scalar per ntr)
        sig_list.append(np.mean(sigs_this))
        # concatenate K (D,1) columns -> (D,K)
        xi_mat = np.concatenate(xi_cols, axis=1)     # (D,K)
        xi_list.append(xi_mat)
        eps_list.append(np.array(eps_this))          # (K,)
        # stack per-k vectors into (N,K)
        rho_mat = np.stack(rho_cols_per_k, axis=1)   # (N,K)
        rho_list.append(rho_mat)

    return (np.array(sig_list),             # (L,)
            np.stack(xi_list, 0),           # (L, D, K)
            np.stack(eps_list, 0),          # (L, K)
            np.stack(rho_list, 0))          # (L, N, K)  <-- NEW

def extrapolate_three_series(true_pot, base_trials_series, targets, mode="trial-averaged"):
    """
    Run infer→extrapolate for each base size in `base_trials_series`,
    and stack results on a new leading axis S (series). For each series s,
    set entries at target indices where targets < base_trials_series[s] to NaN.
    Returns:
      sig_all: (S, L)
      xi_all:  (S, L, D, K)
      eps_all: (S, L, K)
      rho_all_diag_per_neuron: (S, L, N, K)  <-- NEW
    """
    S = len(base_trials_series)
    L = len(targets)

    sig_all = np.full((S, L), np.nan, dtype=float)
    xi_all  = np.full((S, L, D, K), np.nan, dtype=float)
    eps_all = np.full((S, L, K), np.nan, dtype=float)
    rho_all = np.full((S, L, N, K), np.nan, dtype=float)

    for s, n_small in enumerate(base_trials_series):
        pots_small = do_single_small_inference(true_pot, n_trials_small=int(n_small), mode=mode)
        sig_ex, xi_ex, eps_ex, rho_ex = extrapolate_predictions_from_small(
            pots_small, int(n_small), targets, mode=mode
        )
        mask_valid = targets >= n_small
        sig_all[s, mask_valid] = sig_ex[mask_valid]
        xi_all[s,  mask_valid, :, :] = xi_ex[mask_valid]
        eps_all[s, mask_valid, :] = eps_ex[mask_valid]
        rho_all[s, mask_valid, :, :] = rho_ex[mask_valid]   # (mask L, N, K)

    return sig_all, xi_all, eps_all, rho_all

def empirical_accuracy_curves(true_pot, targets, attempts=N_ATTEMPTS_EMPIRICAL, mode="trial-averaged"):
    """
    Compute empirical rho/epsilon vs trials using split-half via get_empirical_accuracy_array.
    Returns mean and std across attempts:
      epsilon_emp_mean/std: [len(targets), K]
      rho_emp_mean/std:     [len(targets), K]
    """
    eps_all = []
    rho_all = []
    max_tr = int(targets.max())
    for _ in range(attempts):
        # Need at least 2*max_tr trials for split-half
        full_data = true_pot.generate_sample_data(n_samples=2*max_tr)
        rho_emp, eps_emp = get_empirical_accuracy_array(
            full_data, K=K, G=true_pot.G,
            mode=mode,
            size_axis="trials",
            size_values=targets
        )
        # Shapes: [len(targets), K]
        rho_all.append(rho_emp)
        eps_all.append(eps_emp)
    rho_all = np.stack(rho_all, 0)   # [attempts, len, K]
    eps_all = np.stack(eps_all, 0)

    # Average rho in the right space: E[(1-rho)^2] then back-transform
    R_all_squared = (1 - rho_all)**2
    mean_rho = 1 - np.sqrt(np.mean(R_all_squared, 0))

    return (eps_all.mean(0), eps_all.std(0),
            mean_rho, rho_all.std(0))

# ----------------------- one-time assets -----------------------
def save_one_time_assets(true_pot):
    """Save constants/true curves that don't depend on attempts."""
    # True parameters repeated across trials
    sqrt_mean_sigma_true = np.sqrt(np.mean(true_pot.bar_sigma**2)/n_trials_array)
    xi_true_tiled = np.einsum("bc,a->abc", true_pot.bar_xi, 1/np.sqrt(n_trials_array))

    save_np("n_trials_array", n_trials_array)
    save_np("sqrt_mean_sigma_true", sqrt_mean_sigma_true)
    save_np("xi_true", xi_true_tiled)

    true_pot.save_as_npz(OUT / "potential_true")

# ----------------------- MAIN routine with capped appends -----------------------
def run_regular():
    # 0) Build ground-truth potential (deterministic for given RNG seed)
    pot_true = build_true_potential(var_array=(4.0, 1.0))
    print("True xi to sigma ratios:", pot_true.bar_xi**2 / np.mean(pot_true.bar_sigma**2))

    # 1) Save constants/true curves (safe to overwrite)
    save_one_time_assets(pot_true)

    # 2) Figure out how many THEORY attempts already saved (use any one file as a guide)
    f_sig_attempts = OUT / "sqrt_mean_sigma_extrap_attempts.npy"  # shape (A, S, L)
    f_xi_attempts  = OUT / "xi_extrap_attempts.npy"               # shape (A, S, L, D, K)
    f_eps_attempts = OUT / "epsilon_pred_attempts.npy"            # shape (A, S, L, K)
    f_rho_attempts = OUT / "rho_pred_attempts.npy"                # shape (A, S, L, N, K)  <-- NEW

    done_theory = max(
        attempts_done(f_sig_attempts),
        attempts_done(f_xi_attempts),
        attempts_done(f_eps_attempts),
        attempts_done(f_rho_attempts),
    )
    remaining = max(0, N_ATTEMPTS_THEORETICAL - done_theory)
    print(f"[theory] attempts done: {done_theory} / {N_ATTEMPTS_THEORETICAL} (remaining {remaining})")

    # 3) Add new THEORY attempts, appending rows up to the cap
    for a in range(remaining):
        attempt_idx = done_theory + a + 1
        print(f"[theory] generating attempt {attempt_idx}")

        # run 3-series infer→extrapolate
        sig_ex_S, xi_ex_S, eps_ex_S, rho_ex_S = extrapolate_three_series(
            pot_true, BASE_TRIALS_SERIES, n_trials_array, mode="trial-averaged"
        )
        # Save single-run (overwrite) for optional overlay in plots
        if attempt_idx == 1:  # only save first attempt as "single-run" reference
            save_np("sqrt_mean_sigma_extrap", sig_ex_S)   # (S, L)
            save_np("xi_extrap", xi_ex_S)                 # (S, L, D, K)
            save_np("epsilon_pred", eps_ex_S)             # (S, L, K)
            save_np("rho_pred", rho_ex_S)                 # (S, L, N, K)  <-- NEW

        # append one row (axis 0) to each attempts file (capped)
        append_rows_capped(f_sig_attempts, sig_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL)
        append_rows_capped(f_xi_attempts,  xi_ex_S[np.newaxis, ...],  N_ATTEMPTS_THEORETICAL)
        append_rows_capped(f_eps_attempts, eps_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL)
        append_rows_capped(f_rho_attempts, rho_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL)

    # 4) After appending, compute and save MEAN/STD aggregates for plotting (NaN-aware over attempts)
    if f_sig_attempts.exists():
        sig_all = np.load(f_sig_attempts)        # (A, S, L)
        xi_all  = np.load(f_xi_attempts)         # (A, S, L, D, K)
        eps_all = np.load(f_eps_attempts)        # (A, S, L, K)
        rho_all = np.load(f_rho_attempts)        # (A, S, L, N, K)

        # Nan-aware aggregates across attempts (axis 0)
        save_np("sqrt_mean_sigma_theory_mean", np.nanmean(sig_all, axis=0))   # (S, L)
        save_np("sqrt_mean_sigma_theory_std",  np.nanstd(sig_all, axis=0))    # (S, L)

        save_np("xi_theory_mean", np.nanmean(xi_all, axis=0))   # (S, L, D, K)
        save_np("xi_theory_std",  np.nanstd(xi_all, axis=0))    # (S, L, D, K)

        save_np("epsilon_theory_mean", np.nanmean(eps_all, axis=0))  # (S, L, K)
        save_np("epsilon_theory_std",  np.nanstd(eps_all, axis=0))   # (S, L, K)

        # Store per-neuron rho aggregates too (rarely needed, but consistent)
        save_np("rho_theory_mean", np.nanmean(rho_all, axis=0))      # (S, L, N, K)
        save_np("rho_theory_std",  np.nanstd(rho_all, axis=0))       # (S, L, N, K)

    # 5) Empirical accuracy curves (computed fresh; overwrite mean/std files)
    print("Calculating empirical accuracy...")
    eps_mean, eps_std, rho_mean, rho_std = empirical_accuracy_curves(
        pot_true, n_trials_array, attempts=N_ATTEMPTS_EMPIRICAL, mode="trial-averaged"
    )

    save_np("epsilon_emp_mean", eps_mean)   # (L, K)
    save_np("epsilon_emp_std",  eps_std)    # (L, K)
    save_np("rho_emp_mean",     rho_mean)   # (L, K)
    save_np("rho_emp_std",      rho_std)    # (L, K)

    # 6) Panel C assets (overwrite)
    print("Saving Panel C assets:")
    save_np("bar_x_true", pot_true.bar_x)
    for ntr in (TRIALS_REF_SMALL, TRIALS_REF_LARGE):
        data = pot_true.generate_sample_data(n_samples=ntr)
        coeff, score, _ = PCA_matlab_like(np.mean(data, axis=0))
        # sign-align to true loadings
        signs = np.sign([np.dot(coeff[:, k], pot_true.bar_e[:, k]) for k in range(K)])
        score = score[:, :K] * signs[np.newaxis, :] / np.sqrt(N)
        save_np(f"inferred_y_{ntr}", score)

    # epsilon_ref (per-component at two trial counts)
    print("Reference values of epsilon for the bar plots:")
    eps_5 = []
    eps_50 = []
    for k in range(K):
        pot5 = extrapolate_potential(
            original=do_single_small_inference(pot_true, TRIALS_REF_SMALL)[k],
            new_trials=TRIALS_REF_SMALL,
            existing_number_of_trials=TRIALS_REF_SMALL,
            mode="trial-averaged",
        )
        pot50 = extrapolate_potential(
            original=do_single_small_inference(pot_true, TRIALS_REF_SMALL)[k],
            new_trials=TRIALS_REF_LARGE,
            existing_number_of_trials=TRIALS_REF_SMALL,
            mode="trial-averaged",
        )
        print("Potentials successfully extrapolated")
        eps_5.append(to_scalar(make_predictions(pot5)["epsilon"]))
        eps_50.append(to_scalar(make_predictions(pot50)["epsilon"]))
    epsilon_ref = np.array([eps_5, eps_50]).T  # (2,2)
    save_np("epsilon_ref", epsilon_ref)

    # 5b) NEW: Direct PCA-at-50 block (no split-half), per-neuron rho_i^{(k)}
    print("Calculating direct per-neuron rho at n=50 via PCA on reduce_to_2d...")
    attempts = N_ATTEMPTS_EMPIRICAL*10
    rho_emp_PCA50 = np.full((attempts, N, K), np.nan, dtype=float)

    # Pre-normalize true modes to ||e|| = sqrt(N) (defensive; generation already does this)
    bar_e_norm = np.array(pot_true.bar_e)
    for k in range(K):
        bar_e_norm[:, k] = bar_e_norm[:, k] / np.linalg.norm(bar_e_norm[:,k]) * np.sqrt(N)

    for a in range(attempts):
        data = pot_true.generate_sample_data(n_samples=50)  # trials = 50
        X = reduce_to_2d(data, mode="trial-averaged")  # shape (T, N) for PCA
        coeff, _, _ = PCA_matlab_like(X)  # coeff: (N, N)
        v = np.array(coeff[:,:K]) * np.sqrt(N)
        for k in range(K):
            v[:,k] *= np.sign(np.dot(bar_e_norm[:, k], v[:,k]))
            rho_emp_PCA50[a, :, k] = (1 / (2)) * (bar_e_norm[:, k] - v[:,k]) ** 2

    save_np("rho_emp_per_neuron_PCA50_attempts", rho_emp_PCA50)  # (A, N, K)
    save_np("rho_emp_per_neuron_PCA50_mean", np.nanmean(rho_emp_PCA50, 0))  # (N, K)
    save_np("rho_emp_per_neuron_PCA50_std", np.nanstd(rho_emp_PCA50, 0))  # (N, K)


# ----------------------- "Bad signal" routine with capped appends -----------------------
def run_bad_SNR():
    # 0) Build ground-truth potential (deterministic for given RNG seed)
    pot_true_bad_SNR = build_true_potential(var_array=(4.0, 0.05))

    # 1) Save constants/true curves (safe to overwrite)
    pot_true_bad_SNR.save_as_npz("cached_results/bad_SNR/pot_true_bad_SNR")

    # 2) Figure out how many THEORY attempts already saved (use any one file as a guide)
    f_sig_attempts = OUT / "bad_SNR/sqrt_mean_sigma_extrap_attempts.npy"  # shape (A, S, L)
    f_xi_attempts  = OUT / "bad_SNR/xi_extrap_attempts.npy"               # shape (A, S, L, D, K)
    f_eps_attempts = OUT / "bad_SNR/epsilon_pred_attempts.npy"            # shape (A, S, L, K)
    f_rho_attempts = OUT / "bad_SNR/rho_pred_attempts.npy"                # shape (A, S, L, N, K)

    done_theory = max(
        attempts_done(f_sig_attempts),
        attempts_done(f_xi_attempts),
        attempts_done(f_eps_attempts),
        attempts_done(f_rho_attempts),
    )
    remaining = max(0, N_ATTEMPTS_THEORETICAL - done_theory)
    print(f"[theory] attempts done: {done_theory} / {N_ATTEMPTS_THEORETICAL} (remaining {remaining})")

    # 3) Add new THEORY attempts, appending rows up to the cap
    for a in range(remaining):
        attempt_idx = done_theory + a + 1
        print(f"[theory] generating attempt {attempt_idx}")

        # run 3-series infer→extrapolate
        sig_ex_S, xi_ex_S, eps_ex_S, rho_ex_S = extrapolate_three_series(
            pot_true_bad_SNR, BASE_TRIALS_SERIES, n_trials_array, mode="trial-averaged"
        )

        # Save single-run (overwrite) for optional overlay in plots
        if attempt_idx == 1:
            save_np("bad_SNR/sqrt_mean_sigma_extrap", sig_ex_S)  # (S, L)
            save_np("bad_SNR/xi_extrap", xi_ex_S)                # (S, L, D, K)
            save_np("bad_SNR/epsilon_pred", eps_ex_S)            # (S, L, K)
            save_np("bad_SNR/rho_pred", rho_ex_S)                # (S, L, N, K)

        # append one row to each attempts file (capped)
        append_rows_capped(f_sig_attempts, sig_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL)
        append_rows_capped(f_xi_attempts,  xi_ex_S[np.newaxis, ...],  N_ATTEMPTS_THEORETICAL)
        append_rows_capped(f_eps_attempts, eps_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL)
        append_rows_capped(f_rho_attempts, rho_ex_S[np.newaxis, ...], N_ATTEMPTS_THEORETICAL)

    # 4) After appending, compute and save MEAN/STD aggregates for plotting (NaN-aware over attempts)
    if f_sig_attempts.exists():
        sig_all = np.load(f_sig_attempts)        # (A, S, L)
        xi_all  = np.load(f_xi_attempts)         # (A, S, L, D, K)
        eps_all = np.load(f_eps_attempts)        # (A, S, L, K)
        rho_all = np.load(f_rho_attempts)        # (A, S, L, N, K)

        np.save(OUT / "bad_SNR/sqrt_mean_sigma_theory_mean.npy", np.nanmean(sig_all, axis=0))
        np.save(OUT / "bad_SNR/sqrt_mean_sigma_theory_std.npy",  np.nanstd(sig_all, axis=0))

        np.save(OUT / "bad_SNR/xi_theory_mean.npy", np.nanmean(xi_all, axis=0))
        np.save(OUT / "bad_SNR/xi_theory_std.npy",  np.nanstd(xi_all, axis=0))

        np.save(OUT / "bad_SNR/epsilon_theory_mean.npy", np.nanmean(eps_all, axis=0))
        np.save(OUT / "bad_SNR/epsilon_theory_std.npy",  np.nanstd(eps_all, axis=0))

        np.save(OUT / "bad_SNR/rho_theory_mean.npy", np.nanmean(rho_all, axis=0))  # (S,L,N,K)
        np.save(OUT / "bad_SNR/rho_theory_std.npy",  np.nanstd(rho_all, axis=0))   # (S,L,N,K)

    # 5) Empirical accuracy curves (computed fresh; overwrite mean/std files)
    print("Calculating empirical accuracy...")
    eps_mean, eps_std, rho_mean, rho_std = empirical_accuracy_curves(
        pot_true_bad_SNR, n_trials_array, attempts=N_ATTEMPTS_EMPIRICAL, mode="trial-averaged")

    np.save(OUT / "bad_SNR/epsilon_emp_mean.npy", eps_mean)
    np.save(OUT / "bad_SNR/epsilon_emp_std.npy",  eps_std)
    np.save(OUT / "bad_SNR/rho_emp_mean.npy",     rho_mean)
    np.save(OUT / "bad_SNR/rho_emp_std.npy",      rho_std)


# ----------------------- Entrypoint -----------------------
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(OUT_BAD_SNR, exist_ok=True)
    run_regular()
    #run_bad_SNR()
