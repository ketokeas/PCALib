import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# High-DPI figures
mpl.rcParams["figure.dpi"] = 200

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()  # fallback when run inside Jupyter/jupytext

# --- Robust results directory ---
OUT = HERE / "cached_results"

# -------------------------------------------------------------
# Utility: add subfigure label (A/B/C)
def add_subfigure_label(subfig, label, x=0, y=1.0, fontsize=18):
    subfig.text(x, y, label, fontsize=fontsize, weight="bold", va="top", ha="left")


# -------------------------------------------------------------
# Subfigure A (top-left): 2D latent trajectory: true vs inferred
# Subfigure A (top-left, 2 columns)
def create_subfigure_A(
    subfig, true_trajectories, inferred_trajectories, empirical_rho, inferred_rho
):
    axs = subfig.subplots(1, 2)
    T = np.shape(true_trajectories)[0]
    mpl.rc("image", cmap="viridis")

    axs[0].scatter(
        inferred_rho[:, 0],
        np.mean(empirical_rho[:, :, 0], 0),
        color="blue",
        alpha=0.4,
        label="k=1",
    )
    axs[0].scatter(
        inferred_rho[:, 1],
        np.mean(empirical_rho[:, :, 1], 0),
        color="red",
        alpha=0.4,
        label="k=2",
    )
    axs[0].plot(
        [0, axs[0].get_ylim()[1]],
        [0, axs[0].get_ylim()[1]],
        linestyle="--",
        color=[0, 0, 0],
    )
    # axs[0].set_title("Error on PC modes inference", fontweight="bold")
    axs[0].set_xlabel(r"Predicted $\rho^{(k)}_i$")
    axs[0].set_ylabel(r"Average $\frac{1}{2} (\bar{e}^{(k)}_i - \bar{v}^{(k)}_i)^2$")

    axs[0].legend()

    axs[1].scatter(
        inferred_trajectories[:, 0], inferred_trajectories[:, 1], c=np.arange(T)
    )
    axs[1].plot(
        true_trajectories[:, 0],
        true_trajectories[:, 1],
        linestyle="--",
        linewidth=2,
        color="red",
    )

    axs[1].set_title("Inferred neural trajectory", fontweight="bold")
    axs[1].set_xlabel("First signal component")
    axs[1].set_ylabel("Second signal component")

    # xlimits = [np.min([axs[0].get_xlim()[0],axs[1].get_xlim()[0]]),np.max([axs[0].get_xlim()[1],axs[1].get_xlim()[1]])]
    # ylimits = [np.min([axs[0].get_ylim()[0],axs[1].get_ylim()[0]]),np.max([axs[0].get_ylim()[1],axs[1].get_ylim()[1]])]

    # axs[0].set_xlim(left=xlimits[0],right=xlimits[1])
    # axs[0].set_ylim(bottom=ylimits[0], top=ylimits[1])
    # axs[1].set_xlim(left=xlimits[0], right=xlimits[1])
    # axs[1].set_ylim(bottom=ylimits[0], top=ylimits[1])

    add_subfigure_label(subfig, "A")


# -------------------------------------------------------------
# Subfigure C (top-right): distributions across attempts
# Left column: rho for selected #animals; Right column: epsilon for selected #trials
# Uses CONSISTENT BINS per column across rows and both modes.
def create_subfigure_C(subfig, rho_array, sqrt_epsilon_array):
    """
    rho_array:   (attempts, 3, K)  for e.g. 2,3,4 animals
    epsilon_array: (attempts, 3, K) for e.g. 10,20,30 trials
    """
    outer_gs = subfig.add_gridspec(1, 2)

    neuron_labels = ["2 animals", "3 animals", "4 animals"]
    trial_labels = ["10 trials", "20 trials", "40 trials"]

    axarray = [[None for _ in range(3)] for _ in range(2)]

    # --- Precompute consistent bins per column (use all rows and both modes) ---
    # Left column bins (rho)
    rho_min = np.inf
    rho_max = -np.inf
    for r in range(3):
        rho_min = min(rho_min, rho_array[:, r, 0].min(), rho_array[:, r, 1].min())
        rho_max = max(rho_max, rho_array[:, r, 0].max(), rho_array[:, r, 1].max())
    rho_bins = np.linspace(rho_min, rho_max, 21)

    # Right column bins (epsilon)
    eps_min = np.inf
    eps_max = -np.inf
    for r in range(3):
        eps_min = min(
            eps_min,
            sqrt_epsilon_array[:, r, 0].min(),
            sqrt_epsilon_array[:, r, 1].min(),
        )
        eps_max = max(
            eps_max,
            sqrt_epsilon_array[:, r, 0].max(),
            sqrt_epsilon_array[:, r, 1].max(),
        )
    eps_bins = np.linspace(eps_min, eps_max, 21)

    for col_idx in range(2):
        inner_gs = outer_gs[col_idx].subgridspec(
            5, 1, height_ratios=[0.5, 1, 1, 1, 0.05], hspace=0.05
        )
        for row_idx in range(3):
            ax = subfig.add_subplot(inner_gs[row_idx + 1])
            axarray[col_idx][row_idx] = ax

            if col_idx == 0:
                data1 = rho_array[:, row_idx, 0]  # mode 1 over attempts
                data2 = rho_array[:, row_idx, 1]  # mode 2 over attempts
                bins = rho_bins
            else:
                data1 = sqrt_epsilon_array[:, row_idx, 0]
                data2 = sqrt_epsilon_array[:, row_idx, 1]
                bins = eps_bins

            ax.hist(data1, bins=bins, color="blue", alpha=0.6)
            ax.hist(data2, bins=bins, color="red", alpha=0.6)

            # Mean lines (both black, as requested)
            ax.axvline(np.mean(data1), color="black", linewidth=2, linestyle="--")
            ax.axvline(np.mean(data2), color="black", linewidth=2, linestyle="--")

            # Clean x-ticks on upper rows
            if row_idx < 2:
                ax.set_xticks([])
            else:
                ax.set_xlabel(
                    r"$\rho$" if col_idx == 0 else r"$\sqrt{\epsilon}$", fontsize=14
                )

            # Titles per row
            if col_idx == 0:
                ax.set_title(neuron_labels[row_idx], fontsize=10)
            else:
                ax.set_title(trial_labels[row_idx], fontsize=10)

    # (x-lims are already aligned by common bins, but keep the unifier for robustness)
    for col_idx in range(2):
        left_x = [axarray[col_idx][row_idx].get_xlim()[0] for row_idx in range(3)]
        right_x = [axarray[col_idx][row_idx].get_xlim()[1] for row_idx in range(3)]
        common_left = np.min(left_x)
        common_right = np.max(right_x)
        for row_idx in range(3):
            axarray[col_idx][row_idx].set_xlim(common_left, common_right)

    subfig.text(0, 1.0, "C", fontsize=16, fontweight="bold", ha="left", va="top")


# -------------------------------------------------------------
# Subfigure B (bottom, full width): accuracy & stability summaries
def create_subfigure_B(
    subfig,
    true_varX,
    inferred_varX,
    true_sigma,
    inferred_sigmas,
    reference_n_animals,
    n_animals_array,
    std_varX_array,
    n_trials_reference,
    n_trials_array,
    std_sigma_array,
):
    axs = subfig.subplots(1, 4)

    # (B1) Signal variability hist (per-mode) at selected D
    axs[0].hist(inferred_varX[:, 0], bins=20, color="blue", alpha=0.6)
    axs[0].hist(inferred_varX[:, 1], bins=20, color="red", alpha=0.6)
    axs[0].axvline(true_varX[0], color="black", linewidth=2, linestyle="--")
    axs[0].axvline(true_varX[1], color="black", linewidth=2, linestyle="--")
    axs[0].set_xlabel("Signal variability")
    axs[0].set_ylabel("count")

    # (B2) sqrt(mean sigma^2) hist at selected #trials
    # NOTE: Your inference uses trial-averaged data, so the inferred sigma summary
    # matches your intended comparison scale (see your point #2).
    axs[1].hist(inferred_sigmas, bins=20, color="green", alpha=0.6)
    axs[1].axvline(true_sigma, color="black", linewidth=2, linestyle="--")
    axs[1].set_xlabel(r"$\sqrt{\langle\bar{\sigma}^2\rangle}$")
    axs[1].set_ylabel("count")
    axs[1].set_xlim(left=0, right=axs[1].get_xlim()[1] * 1.5)

    # (B3) std across attempts of signal variability vs #animals (both modes)
    axs[2].plot(n_animals_array, std_varX_array[:, 0], color="blue", linewidth=2)
    axs[2].plot(n_animals_array, std_varX_array[:, 1], color="red", linewidth=2)
    axs[2].axvline(reference_n_animals, color="black", linewidth=2, linestyle="--")
    axs[2].set_xlabel("number of animals")
    axs[2].set_ylabel("std(signal variability)")

    # (B4) std across attempts of sqrt(mean sigma^2) vs #trials
    axs[3].plot(n_trials_array, std_sigma_array, color="green", linewidth=2)
    axs[3].axvline(n_trials_reference, color="black", linewidth=2, linestyle="--")
    axs[3].set_xlabel("number of trials")
    axs[3].set_ylabel(r"$\mathrm{std}\left(\sqrt{\langle\bar{\sigma}^2\rangle}\right)$")

    subfig.text(0, 1.05, "B", fontsize=16, fontweight="bold", ha="left", va="top")


# -------------------------------------------------------------
# Data loading (with robust true signal extraction from saved Potential)
load = lambda name: np.load(OUT / name)

# Grids and references
n_trials_array = load("n_trials_array.npy")
n_trials_reference = load("n_trials_reference.npy")
D_array = load("D_array.npy")
D_reference = load("D_reference.npy")

# --- True latent trajectory: extract from saved Potential (preferred) ---
# Falls back to cached .npy if present, but you can rely solely on the .npz
true_trajectories_path = OUT / "bar_x_40_trials.npy"
try:
    true_trajectories = np.load(true_trajectories_path)
except FileNotFoundError:
    # Extract from the saved Potential bundle
    from pcalib.classes import Potential

    true_trajectories = Potential.from_npz(str(OUT / "many_trials_potential.npz")).bar_x

# Inferred PCA-aligned trajectory at the reference setting
inferred_trajectories = load("inferred_y_40_trials.npy")

# Empirical and inferred rho specifically for 2 animals and 40 trials
predicted_rho = load("predicted_rho_40_trials.npy")
empirical_rho = load("empirical_rho_40_trials.npy")


# Sweep outputs
rho_animals = load("rho_animals.npy")
epsilon_trials = load("epsilon_trials.npy")
signal_variability_animals = load("signal_variability_animals.npy")
sqrt_mean_sigma_squared = load("sqrt_mean_sigma_squared.npy")
true_sigma = load("true_mean_noise_variance.npy")

T, K = true_trajectories.shape

# Summaries for panel B (pick a particular D and trial count; here index -2)
true_varX = np.var(true_trajectories, axis=0)  # (K,)
inferred_varX = signal_variability_animals[:, -2, :]  # (attempts, K)
inferred_sigma = sqrt_mean_sigma_squared[:, -2]  # (attempts,)

# Across-attempts variability vs sweep variable
std_varX_array = np.std(signal_variability_animals, axis=0)  # (|D_array|, K)
std_sigma_array = np.std(sqrt_mean_sigma_squared, axis=0)  # (|n_trials_array|,)

# -------------------------------------------------------------
# Build the figure
fig = plt.figure(constrained_layout=True, figsize=(12, 6))
outer_gs = fig.add_gridspec(3, 4, height_ratios=[1, 0.1, 1], hspace=0.1)

# Top row: A (left) and C (right)
top_gs = outer_gs[0, :].subgridspec(1, 2, wspace=0.1)

subfig_A = fig.add_subfigure(top_gs[0])
create_subfigure_A(
    subfig_A, true_trajectories, inferred_trajectories, empirical_rho, predicted_rho
)

subfig_C = fig.add_subfigure(top_gs[1])
# Choose 3 entries: animals indices 1..3 and trials indices [1,3,5]
create_subfigure_C(
    subfig_C,
    rho_animals[:, 1:4, :],  # e.g., 2,3,4 animals
    np.sqrt(epsilon_trials[:, [1, 3, 7], :]),  # e.g., 10,20,30 trials
)

# Bottom row: B (full width)
bottom_gs = outer_gs[2, 0:4].subgridspec(1, 1)
subfig_B = fig.add_subfigure(bottom_gs[0])
create_subfigure_B(
    subfig_B,
    true_varX,
    inferred_varX,
    true_sigma,
    inferred_sigma,
    D_reference,
    D_array,
    std_varX_array,
    n_trials_reference,
    n_trials_array,
    std_sigma_array,
)

plt.show()
