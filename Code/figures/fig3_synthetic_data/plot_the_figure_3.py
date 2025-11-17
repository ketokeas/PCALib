# plot_fig3.py  (overlays 3 theory series; rho summed over N per attempt with NaN preservation, then aggregated)
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path as FSPath
from shapely.geometry import LineString
from shapely.affinity import scale
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

# ------------------------------ Global style (compact, publication-friendly) ------------------------------
mpl.rcParams.update(
    {
        "figure.dpi": 200,
        "font.size": 9,  # base
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "lines.linewidth": 1.8,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
    }
)

DATA = FSPath("cached_results")


# ------------------------------ I/O ------------------------------
def load(path_str: str) -> np.ndarray:
    """np.load wrapper that accepts filenames relative to DATA as strings."""
    return np.load(str(DATA / path_str))


# ------------------------------ Small helpers ------------------------------
def add_subfigure_label(
    subfig: mpl.figure.SubFigure,
    label: str,
    x: float = 0.0,
    y: float = 1.0,
    fontsize: int = 14,
) -> None:
    subfig.text(x, y, label, fontsize=fontsize, weight="bold", va="top", ha="left")


def polygon_to_path(polygon) -> MplPath:
    vertices, codes = [], []

    # exterior
    x, y = polygon.exterior.xy
    coords = list(zip(x, y))
    vertices.extend(coords)
    codes.extend(
        [MplPath.MOVETO] + [MplPath.LINETO] * (len(coords) - 2) + [MplPath.CLOSEPOLY]
    )

    # holes
    for ring in polygon.interiors:
        x, y = ring.xy
        coords = list(zip(x, y))
        vertices.extend(coords)
        codes.extend(
            [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(coords) - 2)
            + [MplPath.CLOSEPOLY]
        )

    return MplPath(vertices, codes)


def nanmean_preserve_all_nan(x: np.ndarray, axis: int) -> np.ndarray:
    """
    Like np.nanmean, but returns NaN when *all* entries along `axis` are NaN.
    """
    s = np.nanmean(x, axis=axis)
    all_nan = np.all(np.isnan(x), axis=axis)
    return np.where(all_nan, np.nan, s)


def choose_index_for_trials(
    n_trials_array: np.ndarray, n_target: int
) -> tuple[int, int]:
    """Return (idx, actual_n) where idx points to the entry equal/closest to n_target."""
    idxs = np.where(n_trials_array == n_target)[0]
    idx = (
        int(idxs[0]) if idxs.size else int(np.argmin(np.abs(n_trials_array - n_target)))
    )
    return idx, int(n_trials_array[idx])


# ------------------------------ Subfigure A ------------------------------
# ------------------------------ Subfigure A (xi only; 4 panels) ------------------------------
def create_subfigure_A(
    subfig: mpl.figure.SubFigure,
    n_trials_array: np.ndarray,
    sqrt_mean_sigma_true: np.ndarray,  # kept for API compatibility
    sig_mean_S: np.ndarray,
    sig_std_S: np.ndarray,  # kept for API compatibility
    xi_true: np.ndarray,
    xi_mean_S: np.ndarray,
    xi_std_S: np.ndarray,
    ref_ticks=(5, 50),
    series_labels=("fit from 5", "fit from 10", "fit from 15"),
) -> None:
    """
    A now shows only xi: 4 panels (2x2). The former sigma panel is commented out.
    Each small panel overlays the 3 theory series and the true curve.
    """
    Tlen, D, K = xi_true.shape
    S = xi_mean_S.shape[0]
    series_order = list(range(S))  # 0 (5) -> 1 (10) -> 2 (15)

    # --- Old left sigma panel (commented out to preserve intent) ---
    # gs = subfig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 2.3], wspace=0.12)
    # ax_sig = subfig.add_subplot(gs[0, 0])
    # ax_sig.plot(n_trials_array, sqrt_mean_sigma_true, color=[0.2, 0.8, 0.2],
    #             linewidth=2, label="true")
    # base_col = np.array([0, 0.2, 0])
    # for s in series_order:
    #     ylo = sig_mean_S[s] - sig_std_S[s]
    #     yhi = sig_mean_S[s] + sig_std_S[s]
    #     ax_sig.fill_between(n_trials_array, ylo, yhi, color=base_col, alpha=0.20,
    #                         label=series_labels[s] if s == series_order[0] else None)
    # for rt in ref_ticks:
    #     ax_sig.axvline(rt, color='black', linewidth=0.9, linestyle="--", alpha=0.6)
    # ax_sig.set_xlabel("Number of trials")
    # ax_sig.set_ylabel(r"$\sqrt{\langle \bar{\sigma}^2 \rangle}$")
    # ax_sig.legend(frameon=False, handlelength=1.6, loc="best")
    # ax_sig.tick_params(length=3)

    # --- New: xi only, 4 panels (2x2) ---
    gs_right = subfig.add_gridspec(nrows=2, ncols=2, wspace=0.08, hspace=0.10)

    # choose first four (d, k) pairs in row-major order
    dk_pairs = []
    for d in range(D):
        for k in range(K):
            dk_pairs.append((d, k))
    dk_pairs = dk_pairs[:4]  # up to four panels

    # colors
    animal_colors = [plt.cm.hsv(i / max(1, D)) for i in range(D)]
    shades = np.linspace(0.35, 0.8, max(1, K))

    # global y-lims for comparability (use theory range if available)
    if xi_std_S is not None:
        y_low = float(np.nanmin(xi_mean_S - xi_std_S))
        y_high = float(np.nanmax(xi_mean_S + xi_std_S))
    else:
        y_low = float(np.min(xi_true))
        y_high = float(np.max(xi_true))
    y_pad = 0.03 * (y_high - y_low + 1e-12)
    y_low -= y_pad
    y_high += y_pad

    for idx, (d, k) in enumerate(dk_pairs):
        r, c = divmod(idx, 2)
        ax = subfig.add_subplot(gs_right[r, c])

        base_col_d = np.array(animal_colors[d][:3])
        col = np.array(base_col_d * shades[k])

        # overlay the 3 series in order
        for s in series_order:
            ylo = xi_mean_S[s, :, d, k] - xi_std_S[s, :, d, k]
            yhi = xi_mean_S[s, :, d, k] + xi_std_S[s, :, d, k]
            ax.fill_between(
                n_trials_array,
                ylo,
                yhi,
                color=col * (s + 1) / S,
                alpha=0.30,
                linewidth=0,
            )

        # true curve on top
        ax.plot(n_trials_array, xi_true[:, d, k], color=col, linewidth=1.4, alpha=1)

        for rt in ref_ticks:
            ax.axvline(rt, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

        ax.set_xlim(0, n_trials_array[-1] + (n_trials_array[-1] - n_trials_array[-2]))
        ax.set_ylim(y_low, y_high)
        ax.tick_params(labelsize=7, length=2.5)

        ax.set_title(f"k={k+1}, d={d+1}", fontsize=9, pad=1.5)
        if c == 0:
            ax.set_ylabel(r"$\xi$", fontsize=9)
        if r == 1:
            ax.set_xlabel("trials", fontsize=8, labelpad=1)
        else:
            ax.set_xticklabels([])

    add_subfigure_label(subfig, "A", y=1.01)


# ------------------------------ Subfigure B ------------------------------
def create_subfigure_B(
    subfig: mpl.figure.SubFigure,
    title_prefix: str,
    n_trials_array: np.ndarray,
    eps_mean_S: np.ndarray,
    rho_mean_S: np.ndarray,  # theory means [S, L, K]
    eps_emp_mean: np.ndarray,
    eps_emp_std: np.ndarray,  # empirical (dots+bars)
    rho_emp_mean: np.ndarray,
    rho_emp_std: np.ndarray,
    ref_ticks=(5, 50),
    eps_std_S: np.ndarray | None = None,
    rho_std_S: np.ndarray | None = None,  # [S, L, K]
    series_labels=("fit from 5", "fit from 10", "fit from 15"),
):
    # Close panels, but not cramped
    gs = subfig.add_gridspec(1, 2, wspace=0.10)
    ax_eps = subfig.add_subplot(gs[0])
    ax_rho = subfig.add_subplot(gs[1])

    K = eps_emp_mean.shape[1] if eps_emp_mean.ndim == 2 else eps_mean_S.shape[-1]
    colors = [plt.cm.hsv(i / max(1, K)) for i in range(K)]
    S = eps_mean_S.shape[0]
    series_order = list(range(S))  # 0 (from 5) → 1 (from 10) → 2 (from 15)

    # epsilon
    for k in range(K):
        col = np.array(colors[k][:3])
        # overlay series bands (same color, increasing opacity via stacking)
        for s in series_order:
            if eps_std_S is not None:
                ylo = eps_mean_S[s, :, k] - eps_std_S[s, :, k]
                yhi = eps_mean_S[s, :, k] + eps_std_S[s, :, k]
                ax_eps.fill_between(
                    n_trials_array,
                    ylo,
                    yhi,
                    color=(s + 1) / S * col,
                    alpha=0.5,
                    linewidth=0,
                )

        # empirical
        ax_eps.errorbar(
            n_trials_array,
            eps_emp_mean[:, k],
            eps_emp_std[:, k],
            linestyle="",
            capsize=3,
            color=col * 0.5,
            alpha=1,
            label=None,
        )

    for rt in ref_ticks:
        ax_eps.axvline(rt, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax_eps.set_xlabel("Number of trials")
    ax_eps.set_ylabel(r"$\epsilon$")
    ax_eps.set_title(f"{title_prefix}: ε", fontsize=10, pad=2)
    ax_eps.legend(frameon=False, ncol=1, handlelength=1.6, loc="upper right")
    ax_eps.tick_params(length=3)

    # rho
    for k in range(K):
        col = np.array(colors[k][:3])
        for s in series_order:
            if rho_std_S is not None:
                ylo = rho_mean_S[s, :, k] - rho_std_S[s, :, k]
                yhi = rho_mean_S[s, :, k] + rho_std_S[s, :, k]
                ax_rho.fill_between(
                    n_trials_array,
                    ylo,
                    yhi,
                    color=(s + 1) / S * col,
                    alpha=0.5,
                    linewidth=0,
                )

        ax_rho.errorbar(
            n_trials_array,
            rho_emp_mean[:, k],
            rho_emp_std[:, k],
            linestyle="",
            capsize=3,
            color=col * 0.5,
            alpha=1,
            label=None,
        )

    for rt in ref_ticks:
        ax_rho.axvline(rt, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax_rho.set_xlabel("Number of trials")
    ax_rho.set_ylabel(r"$\rho$")
    ax_rho.set_title(f"{title_prefix}: ρ", fontsize=10, pad=2)
    ax_rho.legend(frameon=False, ncol=1, handlelength=1.6, loc="upper right")
    ax_rho.tick_params(length=3)
    add_subfigure_label(subfig, "B", y=1.01)

    return (ax_eps, ax_rho)


# ------------------------------ Subfigure C ------------------------------
def create_subfigure_C(
    subfig: mpl.figure.SubFigure,
    ref_trials: tuple[int, int],
    true_modes: np.ndarray,
    inferred_5: np.ndarray,
    inferred_50: np.ndarray,
    epsilon_ref: np.ndarray,
) -> None:
    # Two panels, modest gap
    gs = subfig.add_gridspec(1, 2, wspace=0.10)
    axL = subfig.add_subplot(gs[0])
    axR = subfig.add_subplot(gs[1])
    axs = (axL, axR)
    T = true_modes.shape[0]
    mpl.rc("image", cmap="viridis")

    # left (n=5)
    line_5 = LineString(
        [
            (
                inferred_5[t, 0] / np.sqrt(epsilon_ref[0, 0]),
                inferred_5[t, 1] / np.sqrt(epsilon_ref[1, 0]),
            )
            for t in range(T)
        ]
    )
    descaled_5 = scale(
        line_5.buffer(1),
        xfact=np.sqrt(epsilon_ref[0, 0]),
        yfact=np.sqrt(epsilon_ref[1, 0]),
        origin=(0, 0),
    )
    patch_5 = PathPatch(polygon_to_path(descaled_5), facecolor="grey", alpha=0.4, lw=0)
    axL.add_patch(patch_5)
    axL.scatter(inferred_5[:, 0], inferred_5[:, 1], c=np.arange(T), s=8)
    axL.plot(
        true_modes[:, 0], true_modes[:, 1], color="red", linewidth=2, linestyle="--"
    )
    axL.set_title(f"{ref_trials[0]} trials", fontweight="bold", fontsize=10, pad=2)
    axL.set_xlabel("PC1")
    axL.set_ylabel("PC2")

    # right (n=50)
    line_50 = LineString(
        [
            (
                inferred_50[t, 0] / np.sqrt(epsilon_ref[0, 1]),
                inferred_50[t, 1] / np.sqrt(epsilon_ref[1, 1]),
            )
            for t in range(T)
        ]
    )
    descaled_50 = scale(
        line_50.buffer(1),
        xfact=np.sqrt(epsilon_ref[0, 1]),
        yfact=np.sqrt(epsilon_ref[1, 1]),
        origin=(0, 0),
    )
    patch_50 = PathPatch(
        polygon_to_path(descaled_50), facecolor="grey", alpha=0.4, lw=0
    )
    axR.add_patch(patch_50)
    axR.scatter(inferred_50[:, 0], inferred_50[:, 1], c=np.arange(T), s=8)
    axR.plot(
        true_modes[:, 0], true_modes[:, 1], color="red", linewidth=2, linestyle="--"
    )
    axR.set_title(f"{ref_trials[1]} trials", fontweight="bold", fontsize=10, pad=2)
    axR.set_xlabel("PC1")
    axR.set_ylabel("PC2")

    # unify limits
    xlims = [
        min(axs[0].get_xlim()[0], axs[1].get_xlim()[0]),
        max(axs[0].get_xlim()[1], axs[1].get_xlim()[1]),
    ]
    ylims = [
        min(axs[0].get_ylim()[0], axs[1].get_ylim()[0]),
        max(axs[0].get_ylim()[1], axs[1].get_ylim()[1]),
    ]
    pad_x = 0.03 * (xlims[1] - xlims[0] + 1e-12)
    pad_y = 0.03 * (ylims[1] - ylims[0] + 1e-12)
    xlims = [xlims[0] - pad_x, xlims[1] + pad_x]
    ylims = [ylims[0] - pad_y, ylims[1] + pad_y]

    for ax in axs:
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.tick_params(length=3)

    add_subfigure_label(subfig, "C", y=1.01, fontsize=14)


# ------------------------------ Subfigure D: per-neuron bars ------------------------------
def create_subfigure_D_hist(
    subfig: mpl.figure.SubFigure,
    n_trials_array: np.ndarray,
    rho_attempts_S: np.ndarray,  # (A, S, L, N, K) per-neuron inferred diag ρ
    rho_emp_per_neuron_attempts: np.ndarray
    | None,  # (A, N, K) per-neuron empirical attempts
    neuron_ids_1based=(1, 2, 3, 4),
    k_select: int = 0,
    series_labels=("fit from 5", "fit from 10", "fit from 15"),
    n_target: int = 40,
) -> None:
    """
    Single grouped-bar plot:
      x-axis groups = selected neurons (e.g., i = 1,2,3,4)
      within each group: 3 theory bars (series) + 1 empirical bar

    Bars show mean ± std across attempts (NaN-aware). If theory values at n_target
    are NaN for a given (series, neuron), that bar will be empty (height NaN).
    """
    # Resolve the trials index closest to n_target
    idx, actual_n = choose_index_for_trials(n_trials_array, n_target)

    # Shapes
    A, S, L, N, K = rho_attempts_S.shape
    have_emp = (
        (rho_emp_per_neuron_attempts is not None)
        and rho_emp_per_neuron_attempts.ndim == 3
        and rho_emp_per_neuron_attempts.shape[1:] == (N, K)
    )

    # Prepare list of valid neuron indices (0-based) and their labels
    valid_i0, xlabels = [], []
    for nid in neuron_ids_1based:
        i0 = int(nid) - 1
        if 0 <= i0 < N:
            valid_i0.append(i0)
            xlabels.append(f"Neuron {nid}")

    if not valid_i0:
        ax = subfig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, "No valid neuron indices", ha="center", va="center")
        ax.axis("off")
        add_subfigure_label(subfig, "D", y=1.01, fontsize=14)
        return

    num_groups = len(valid_i0)
    num_bars_per_group = S + 1  # S theory + 1 empirical
    colors_series = [plt.cm.tab10(i) for i in range(S)]
    emp_color = (0.25, 0.25, 0.25)

    # Collect means/stds for each group (neuron) and each bar (series/empirical)
    means = np.full((num_groups, num_bars_per_group), np.nan, dtype=float)
    stds = np.full((num_groups, num_bars_per_group), np.nan, dtype=float)

    for g, i0 in enumerate(valid_i0):
        # Theory bars (S of them)
        for s in range(S):
            vals = rho_attempts_S[:, s, idx, i0, k_select]  # (A,)
            means[g, s] = np.nanmean(vals)
            stds[g, s] = np.nanstd(vals)

        # Empirical bar (last position)
        if have_emp:
            vemp = rho_emp_per_neuron_attempts[:, i0, k_select]  # (A,)
            means[g, S] = np.nanmean(vemp)
            stds[g, S] = np.nanstd(vemp)

    # ---- Plot: grouped bars on a single axis ----
    ax = subfig.add_subplot(1, 1, 1)

    x = np.arange(num_groups, dtype=float)  # group centers
    total_width = 0.78
    bar_width = total_width / num_bars_per_group
    offsets = (
        np.arange(num_bars_per_group) - (num_bars_per_group - 1) / 2.0
    ) * bar_width

    # Draw theory bars
    handles, labels = [], []
    for s in range(S):
        h = ax.bar(
            x + offsets[s],
            means[:, s],
            yerr=stds[:, s],
            width=bar_width * 0.95,
            capsize=3,
            color=colors_series[s],
            alpha=0.95,
            label=series_labels[s]
            if s == 0
            else None,  # keep legend appearance identical
        )
        if s == 0:
            handles.append(h)
            labels.append(series_labels[s])

    # Draw empirical bar
    he = ax.bar(
        x + offsets[S],
        means[:, S],
        yerr=stds[:, S],
        width=bar_width * 0.95,
        capsize=3,
        color=emp_color,
        alpha=0.95,
        label="empirical",
    )
    handles.append(he)
    labels.append("empirical")

    ax.set_xticks(x, xlabels)
    ax.set_ylabel(rf"$\rho_i^{{(k={k_select+1})}}$")
    ax.set_title(f"Per-neuron ρ at {actual_n} trials", fontsize=10)
    ax.tick_params(length=3)
    ax.margins(x=0.02)
    ax.set_ylim(0, ax.get_ylim()[1])

    # Legend (distinct entries for all S theory series + empirical)
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=colors_series[s]) for s in range(S)]
        + [plt.Rectangle((0, 0), 1, 1, color=emp_color)],
        [*series_labels, "empirical"],
        frameon=False,
        ncol=2,
    )

    add_subfigure_label(subfig, "D", y=1.01, fontsize=14)


# ------------------------------ Data loading & aggregation ------------------------------
def load_all_inputs():
    # Common axes
    n_trials_array = load("n_trials_array.npy")

    # True parameters (for A)
    sqrt_mean_sigma_true = load("sqrt_mean_sigma_true.npy")
    xi_true = load("xi_true.npy")  # [L, D, K]

    # Attempt stacks WITH series axis and per-neuron rho:
    sig_attempts_S = load("sqrt_mean_sigma_extrap_attempts.npy")  # (A, S, L)
    xi_attempts_S = load("xi_extrap_attempts.npy")  # (A, S, L, D, K)
    eps_attempts_S = load("epsilon_pred_attempts.npy")  # (A, S, L, K)
    rho_attempts_S = load("rho_pred_attempts.npy")  # (A, S, L, N, K)
    rho_emp_per_neuron_attempts = load("rho_emp_per_neuron_PCA50_attempts.npy")

    # Empirical accuracy (global; already saved as mean/std; no series axis)
    eps_emp_mean = load("epsilon_emp_mean.npy")  # (L, K)
    eps_emp_std = load("epsilon_emp_std.npy")  # (L, K)
    rho_emp_mean = load("rho_emp_mean.npy")  # (L, K)
    rho_emp_std = load("rho_emp_std.npy")  # (L, K)

    # Panel C assets
    bar_x_true = load("bar_x_true.npy")  # [T, 2]
    inferred_y_5 = load("inferred_y_5.npy")  # [T, 2]
    inferred_y_50 = load("inferred_y_50.npy")  # [T, 2]
    epsilon_ref = load("epsilon_ref.npy")  # [[eps1@5, eps1@50],[eps2@5, eps2@50]]

    return (
        n_trials_array,
        sqrt_mean_sigma_true,
        xi_true,
        sig_attempts_S,
        xi_attempts_S,
        eps_attempts_S,
        rho_attempts_S,
        rho_emp_per_neuron_attempts,
        eps_emp_mean,
        eps_emp_std,
        rho_emp_mean,
        rho_emp_std,
        bar_x_true,
        inferred_y_5,
        inferred_y_50,
        epsilon_ref,
    )


def aggregate_theory(sig_attempts_S, xi_attempts_S, eps_attempts_S, rho_attempts_S):
    # Rho: sum over N per attempt (preserve all-NaN), then aggregate
    rho_meanN_attempts = nanmean_preserve_all_nan(
        rho_attempts_S, axis=3
    )  # (A, S, L, K)
    rho_pred_mean_S = np.nanmean(rho_meanN_attempts, axis=0)  # (S, L, K)
    rho_pred_std_S = np.nanstd(rho_meanN_attempts, axis=0)

    # Other theory aggregates (NaN-aware across attempts)
    sqrt_mean_sigma_theory_mean_S = np.nanmean(sig_attempts_S, axis=0)  # (S, L)
    sqrt_mean_sigma_theory_std_S = np.nanstd(sig_attempts_S, axis=0)  # (S, L)
    xi_theory_mean_S = np.nanmean(xi_attempts_S, axis=0)  # (S, L, D, K)
    xi_theory_std_S = np.nanstd(xi_attempts_S, axis=0)  # (S, L, D, K)
    eps_pred_mean_S = np.nanmean(eps_attempts_S, axis=0)  # (S, L, K)
    eps_pred_std_S = np.nanstd(eps_attempts_S, axis=0)  # (S, L, K)

    return (
        rho_pred_mean_S,
        rho_pred_std_S,
        sqrt_mean_sigma_theory_mean_S,
        sqrt_mean_sigma_theory_std_S,
        xi_theory_mean_S,
        xi_theory_std_S,
        eps_pred_mean_S,
        eps_pred_std_S,
    )


# ------------------------------ Main figure ------------------------------
def main():
    (
        n_trials_array,
        sqrt_mean_sigma_true,
        xi_true,
        sig_attempts_S,
        xi_attempts_S,
        eps_attempts_S,
        rho_attempts_S,
        rho_emp_per_neuron_attempts,
        eps_emp_mean,
        eps_emp_std,
        rho_emp_mean,
        rho_emp_std,
        bar_x_true,
        inferred_y_5,
        inferred_y_50,
        epsilon_ref,
    ) = load_all_inputs()

    (
        rho_pred_mean_S,
        rho_pred_std_S,
        sqrt_mean_sigma_theory_mean_S,
        sqrt_mean_sigma_theory_std_S,
        xi_theory_mean_S,
        xi_theory_std_S,
        eps_pred_mean_S,
        eps_pred_std_S,
    ) = aggregate_theory(sig_attempts_S, xi_attempts_S, eps_attempts_S, rho_attempts_S)

    ref_trials = (5, 50)
    SERIES_LABELS = ("fit from 5", "fit from 10", "fit from 15")

    fig = plt.figure(figsize=(12, 6.8))

    # Top row: A | B ; spacer ; Bottom row: C | D
    outer_gs = fig.add_gridspec(
        3, 4, height_ratios=[1.0, 0.04, 1.0], hspace=0.02, wspace=0.06
    )
    top_gs = outer_gs[0, :].subgridspec(1, 2, wspace=0.08)  # A | B
    bottom_gs = outer_gs[2, :].subgridspec(1, 2, wspace=0.10)  # C | D

    # A: sigma & xi (grid), with 3 overlaid theory series
    subfig_A = fig.add_subfigure(top_gs[0])
    create_subfigure_A(
        subfig_A,
        n_trials_array,
        sqrt_mean_sigma_true,
        sqrt_mean_sigma_theory_mean_S,
        sqrt_mean_sigma_theory_std_S,
        xi_true,
        xi_theory_mean_S,
        xi_theory_std_S,
        ref_ticks=ref_trials,
        series_labels=SERIES_LABELS,
    )

    # B: epsilon & rho — overlay 3 theory series + empirical points
    subfig_B = fig.add_subfigure(top_gs[1])
    create_subfigure_B(
        subfig_B,
        "Regular",
        n_trials_array,
        eps_pred_mean_S,
        rho_pred_mean_S,  # (S, L, K) — rho already summed over N
        eps_emp_mean,
        eps_emp_std,
        rho_emp_mean,
        rho_emp_std,
        ref_ticks=ref_trials,
        eps_std_S=eps_pred_std_S,
        rho_std_S=rho_pred_std_S,
        series_labels=SERIES_LABELS,
    )

    # C: trajectories with uncertainty ribbons based on epsilon_ref
    subfig_C = fig.add_subfigure(bottom_gs[0])
    create_subfigure_C(
        subfig_C, ref_trials, bar_x_true, inferred_y_5, inferred_y_50, epsilon_ref
    )

    # D: per-neuron bars (3 theory + empirical per-neuron)
    subfig_D = fig.add_subfigure(bottom_gs[1])
    create_subfigure_D_hist(
        subfig_D,
        n_trials_array,
        rho_attempts_S,  # (A, S, L, N, K) per-neuron inferred
        rho_emp_per_neuron_attempts,  # (A, N, K) per-neuron empirical
        neuron_ids_1based=(1, 2, 3, 4),
        k_select=0,  # choose component (0 or 1)
        series_labels=SERIES_LABELS,
        n_target=50,
    )

    plt.show()


if __name__ == "__main__":
    main()
