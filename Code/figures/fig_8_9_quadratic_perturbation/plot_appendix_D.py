from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA = Path(".")  # Folder containing the saved arrays.
a_values = np.linspace(0, 2, 10)  # Same array as in bad_data_generator.py.
a_tags = ["0", "0p5", "1", "2p0"]  # Exact suffixes used in the saved filenames.

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "lines.linewidth": 2, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def plot_errors(ax, x, empirical, predicted):
    mean, sd = empirical.mean(axis=0), empirical.std(axis=0)
    ax.fill_between(x, np.maximum(0, mean - sd), mean + sd,
                    color="#247BA0", alpha=0.18, linewidth=0)
    ax.plot(x, mean, color="#247BA0", label="Empirical")
    ax.plot(x, predicted, color="#D55E00", linestyle="--", label="Predicted")
    ax.set_ylim(bottom=0)
    ax.set_xlim(x[0], x[-1])


# Figure 1: amplitude sweep at a fixed trial count.
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), layout="constrained")
K = np.load(DATA / "inferred_K.npy")
axes[0].plot(a_values, K, "o-", color="#247BA0", markersize=4)
axes[0].axhline(1, color="0.5", linestyle="--", label="Original model")
axes[0].set(ylabel="Inferred K", yticks=np.arange(K.max() + 1),
            ylim=(max(0, K.min() - 0.15), K.max() + 0.15))
axes[0].legend(frameon=False, fontsize=9)

for ax, metric, label in zip(axes[1:], ["rho", "epsilon"],
                            [r"$\rho_{1,1}$", r"$\varepsilon_{1,1}$"]):
    plot_errors(ax, a_values, np.load(DATA / f"empirical_{metric}.npy"),
                np.load(DATA / f"predicted_{metric}.npy"))
    ax.set_ylabel(label)
for ax, title in zip(axes, ["(A) Dimensionality", "(B) Loading error", "(C) Trajectory error"]):
    ax.set_xlabel("Quadratic amplitude a")
    ax.set_title(title, loc="left", pad=10)
axes[1].legend(frameon=False, fontsize=9)
fig.savefig(DATA / "appendix_D_amplitude.pdf")
fig.savefig(DATA / "appendix_D_amplitude.png", dpi=300)

# Figure 2: trial extrapolation, one column per quadratic amplitude.
fig, axes = plt.subplots(2, 4, figsize=(12, 5.4), sharex="col", layout="constrained")
for col, tag in enumerate(a_tags):
    trials = np.load(DATA / f"trial_values_a_{tag}.npy")
    for row, metric in enumerate(["rho", "epsilon"]):
        ax = axes[row, col]
        plot_errors(ax, trials,
                    np.load(DATA / f"empirical_{metric}_a_{tag}.npy"),
                    np.load(DATA / f"predicted_{metric}_a_{tag}.npy"))
        ax.set_title(f"({chr(65 + row * 4 + col)})", loc="left", fontsize=10)
    axes[0, col].set_title(f"a = {float(tag.replace('p', '.')):g}", pad=10)
    axes[1, col].set_xlabel("Number of trials")
axes[0, 0].set_ylabel(r"$\rho_{1,1}$")
axes[1, 0].set_ylabel(r"$\varepsilon_{1,1}$")
axes[0, 0].legend(frameon=False, fontsize=9)
fig.savefig(DATA / "appendix_D_extrapolation.pdf")
fig.savefig(DATA / "appendix_D_extrapolation.png", dpi=300)
plt.show()
