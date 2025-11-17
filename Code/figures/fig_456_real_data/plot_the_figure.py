#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 200
import matplotlib.pyplot as plt

OUT_DIRS = [Path("cached_results/1 Gallego-Carracedo et al/real_gc_grid")]
IMAGE_PATHS = [Path("cached_results/1 Gallego-Carracedo et al/Subfig_A.png")]


def pick_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def safe_slice_K(a, Kplot):
    if a is None:
        return None
    if a.ndim >= 2 and a.shape[-1] > Kplot:
        return a[..., :Kplot]
    return a


def try_load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[WARN] Could not parse JSON {p}: {e}")
        return None


def load_arrays(base: Path, name: str, required=True):
    p = base / name
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {p}")
        else:
            return None
    return np.load(p)


def load_saved(out_dir: Path):
    # --- grids first (so we can set refs even if meta is broken)
    trials_grid = load_arrays(out_dir, "trials_grid.npy")
    neurons_grid = load_arrays(out_dir, "neurons_grid.npy")

    # Animals x-values (prefer JSON; else D_grid.npy; else build 1..L)
    animals_json = out_dir / "animals_grid.json"
    D_grid = None
    if animals_json.exists():
        ag = try_load_json(animals_json)
        if isinstance(ag, list) and len(ag) and "D" in ag[0]:
            D_grid = np.array([int(x["D"]) for x in ag], dtype=int)
    if D_grid is None:
        # New inference saves D_grid.npy
        D_grid = load_arrays(out_dir, "D_grid.npy", required=False)
    if D_grid is None:
        # Deduce from predictions shape as a last resort
        # (we'll load the animals rho next and infer rows)
        pass

    # --- predictions (new filenames from the updated inference script)
    pr_tr_rho = load_arrays(out_dir, "pred_trials_mean_rho.npy")
    pr_tr_eps = load_arrays(out_dir, "pred_trials_epsilon.npy")
    pr_tr_rho_std = load_arrays(out_dir, "pred_trials_mean_rho_std.npy")
    pr_tr_eps_std = load_arrays(out_dir, "pred_trials_epsilon_std.npy")

    pr_ne_rho = load_arrays(out_dir, "pred_neurons_mean_rho.npy")
    pr_ne_eps = load_arrays(out_dir, "pred_neurons_epsilon.npy")
    pr_ne_rho_std = load_arrays(out_dir, "pred_neurons_mean_rho_std.npy")
    pr_ne_eps_std = load_arrays(out_dir, "pred_neurons_epsilon_std.npy")

    pr_an_rho = load_arrays(out_dir, "pred_animals_mean_rho.npy")
    pr_an_eps = load_arrays(out_dir, "pred_animals_epsilon.npy")
    pr_an_rho_std = load_arrays(out_dir, "pred_animals_mean_rho_std.npy")
    pr_an_eps_std = load_arrays(out_dir, "pred_animals_epsilon_std.npy")

    # If D grid wasn’t available, deduce it from the number of rows in pred_animals_*
    if D_grid is None:
        L = pr_an_rho.shape[0]
        D_grid = np.arange(1, L + 1, dtype=int)

    # --- infer K, clamp to 2 for plotting
    K_from_preds = pr_tr_rho.shape[1]
    Kplot = int(min(2, K_from_preds))

    # --- empirical means/stds (new names)
    em_tr_rho = load_arrays(out_dir, "emp_trials_mean_rho_mean.npy")
    em_tr_eps = load_arrays(out_dir, "emp_trials_epsilon_mean.npy")
    em_tr_rho_std = load_arrays(out_dir, "emp_trials_mean_rho_std.npy")
    em_tr_eps_std = load_arrays(out_dir, "emp_trials_epsilon_std.npy")

    em_ne_rho = load_arrays(out_dir, "emp_neurons_mean_rho_mean.npy")
    em_ne_eps = load_arrays(out_dir, "emp_neurons_epsilon_mean.npy")
    em_ne_rho_std = load_arrays(out_dir, "emp_neurons_mean_rho_std.npy")
    em_ne_eps_std = load_arrays(out_dir, "emp_neurons_epsilon_std.npy")

    em_an_rho = load_arrays(out_dir, "emp_animals_mean_rho_mean.npy")
    em_an_eps = load_arrays(out_dir, "emp_animals_epsilon_mean.npy")
    em_an_rho_std = load_arrays(out_dir, "emp_animals_mean_rho_std.npy")
    em_an_eps_std = load_arrays(out_dir, "emp_animals_epsilon_std.npy")

    # --- slice to Kplot
    pred = {
        "trials": {
            "mean_rho": safe_slice_K(pr_tr_rho, Kplot),
            "mean_rho_std": safe_slice_K(pr_tr_rho_std, Kplot),
            "eps": safe_slice_K(pr_tr_eps, Kplot),
            "eps_std": safe_slice_K(pr_tr_eps_std, Kplot),
        },
        "neurons": {
            "mean_rho": safe_slice_K(pr_ne_rho, Kplot),
            "mean_rho_std": safe_slice_K(pr_ne_rho_std, Kplot),
            "eps": safe_slice_K(pr_ne_eps, Kplot),
            "eps_std": safe_slice_K(pr_ne_eps_std, Kplot),
        },
        "animals": {
            "mean_rho": safe_slice_K(pr_an_rho, Kplot),
            "mean_rho_std": safe_slice_K(pr_an_rho_std, Kplot),
            "eps": safe_slice_K(pr_an_eps, Kplot),
            "eps_std": safe_slice_K(pr_an_eps_std, Kplot),
        },
    }
    emp = {
        "trials": {
            "mean_rho": safe_slice_K(em_tr_rho, Kplot),
            "mean_rho_std": safe_slice_K(em_tr_rho_std, Kplot),
            "eps": safe_slice_K(em_tr_eps, Kplot),
            "eps_std": safe_slice_K(em_tr_eps_std, Kplot),
        },
        "neurons": {
            "mean_rho": safe_slice_K(em_ne_rho, Kplot),
            "mean_rho_std": safe_slice_K(em_ne_rho_std, Kplot),
            "eps": safe_slice_K(em_ne_eps, Kplot),
            "eps_std": safe_slice_K(em_ne_eps_std, Kplot),
        },
        "animals": {
            "mean_rho": safe_slice_K(em_an_rho, Kplot),
            "mean_rho_std": safe_slice_K(em_an_rho_std, Kplot),
            "eps": safe_slice_K(em_an_eps, Kplot),
            "eps_std": safe_slice_K(em_an_eps_std, Kplot),
        },
    }

    # --- reference markers: try meta.json, else first grid entries
    meta = try_load_json(out_dir / "meta.json")
    if meta is not None and "available_data" in meta:
        ref_trials = int(meta["available_data"].get("pre_trials", trials_grid[0]))
        ref_neurons = int(
            len(meta["available_data"].get("pre_neuron_ids", []))
            or int(neurons_grid[0])
        )
        ref_animals = int(meta["available_data"].get("D_pre", int(D_grid[0])))
    else:
        print(
            "[WARN] Using grid first entries as reference markers (meta.json missing/invalid)."
        )
        ref_trials, ref_neurons, ref_animals = (
            int(trials_grid[0]),
            int(neurons_grid[0]),
            int(D_grid[0]),
        )

    return {
        "Kplot": Kplot,
        "ref_trials": ref_trials,
        "ref_neurons": ref_neurons,
        "ref_animals": ref_animals,
        "trials_grid": trials_grid,
        "neurons_grid": neurons_grid,
        "D_grid": D_grid,
        "pred": pred,
        "emp": emp,
    }


def add_subfigure_label(subfig, label, x=0, y=1.0, fontsize=18):
    subfig.text(x, y, label, fontsize=fontsize, weight="bold", va="top", ha="left")


def _plot_axis_pair(
    ax_eps,
    ax_rho,
    x,
    pred_eps,
    pred_eps_std,
    emp_eps,
    emp_eps_std,
    pred_rho,
    pred_rho_std,
    emp_rho,
    emp_rho_std,
    xlabel,
    ref_value,
    Kplot,
):
    colors = [plt.cm.hsv(i / max(1, Kplot)) for i in range(Kplot)]

    # --- epsilon panel (with labels) ---
    for k in range(Kplot):
        ax_eps.fill_between(
            x,
            pred_eps[:, k] - pred_eps_std[:, k],
            pred_eps[:, k] + pred_eps_std[:, k],
            alpha=0.25,
            color=colors[k][:3],
            label=f"k={k+1} prediction",
        )
        ax_eps.errorbar(
            x,
            emp_eps[:, k],
            emp_eps_std[:, k],
            linewidth=2,
            linestyle="",
            capsize=5,
            color=colors[k][:3],
            label=f"k={k+1} empirical",
        )
    ax_eps.axvline(ref_value, color="black", linewidth=1.5, linestyle="--")
    ax_eps.set_xlabel(xlabel)
    ax_eps.set_ylabel(r"$\epsilon$", fontsize=14)

    # --- rho panel (no labels to avoid duplicates) ---
    for k in range(Kplot):
        ax_rho.fill_between(
            x,
            pred_rho[:, k] - pred_rho_std[:, k],
            pred_rho[:, k] + pred_rho_std[:, k],
            alpha=0.25,
            color=colors[k][:3],
        )
        ax_rho.errorbar(
            x,
            emp_rho[:, k],
            emp_rho_std[:, k],
            linewidth=2,
            linestyle="",
            capsize=5,
            color=colors[k][:3],
        )
    ax_rho.axvline(ref_value, color="black", linewidth=1.5, linestyle="--")
    ax_rho.set_xlabel(xlabel)
    ax_rho.set_ylabel(r"$\rho$", fontsize=14)

    # one combined legend (from epsilon axis only)
    handles, labels = ax_eps.get_legend_handles_labels()
    if handles:
        ax_rho.legend(handles, labels, fontsize=8, loc="best")


def make_figure(saved):
    # optional image for A
    img_path = pick_existing(IMAGE_PATHS)
    img = None
    if img_path:
        try:
            from PIL import Image

            img = np.asarray(Image.open(img_path))
        except Exception:
            img = None

    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    outer_gs = fig.add_gridspec(3, 4, height_ratios=[1, 0.1, 1], hspace=0.1)
    top_gs = outer_gs[0, :].subgridspec(1, 2, wspace=0.2)
    bottom_gs = outer_gs[2, :].subgridspec(1, 2, wspace=0.2)

    # A
    subfig_A = fig.add_subfigure(top_gs[0])
    axA = subfig_A.subplots(1, 1)
    if img is not None:
        axA.imshow(img)
    axA.set_axis_off()
    add_subfigure_label(subfig_A, "A")

    # B (trials)
    subfig_B = fig.add_subfigure(top_gs[1])
    axB_eps, axB_rho = subfig_B.subplots(1, 2)
    add_subfigure_label(subfig_B, "B")
    _plot_axis_pair(
        axB_eps,
        axB_rho,
        saved["trials_grid"],
        saved["pred"]["trials"]["eps"],
        saved["pred"]["trials"]["eps_std"],
        saved["emp"]["trials"]["eps"],
        saved["emp"]["trials"]["eps_std"],
        saved["pred"]["trials"]["mean_rho"],
        saved["pred"]["trials"]["mean_rho_std"],
        saved["emp"]["trials"]["mean_rho"],
        saved["emp"]["trials"]["mean_rho_std"],
        xlabel="Number of trials",
        ref_value=saved["ref_trials"],
        Kplot=saved["Kplot"],
    )

    # C (neurons)
    subfig_C = fig.add_subfigure(bottom_gs[0])
    axC_eps, axC_rho = subfig_C.subplots(1, 2)
    add_subfigure_label(subfig_C, "C")
    _plot_axis_pair(
        axC_eps,
        axC_rho,
        saved["neurons_grid"],
        saved["pred"]["neurons"]["eps"],
        saved["pred"]["neurons"]["eps_std"],
        saved["emp"]["neurons"]["eps"],
        saved["emp"]["neurons"]["eps_std"],
        saved["pred"]["neurons"]["mean_rho"],
        saved["pred"]["neurons"]["mean_rho_std"],
        saved["emp"]["neurons"]["mean_rho"],
        saved["emp"]["neurons"]["mean_rho_std"],
        xlabel="Number of neurons",
        ref_value=saved["ref_neurons"],
        Kplot=saved["Kplot"],
    )

    # D (animals)
    subfig_D = fig.add_subfigure(bottom_gs[1])
    axD_eps, axD_rho = subfig_D.subplots(1, 2)
    add_subfigure_label(subfig_D, "D")
    _plot_axis_pair(
        axD_eps,
        axD_rho,
        saved["D_grid"],
        saved["pred"]["animals"]["eps"],
        saved["pred"]["animals"]["eps_std"],
        saved["emp"]["animals"]["eps"],
        saved["emp"]["animals"]["eps_std"],
        saved["pred"]["animals"]["mean_rho"],
        saved["pred"]["animals"]["mean_rho_std"],
        saved["emp"]["animals"]["mean_rho"],
        saved["emp"]["animals"]["mean_rho_std"],
        xlabel="Number of animals",
        ref_value=saved["ref_animals"],
        Kplot=saved["Kplot"],
    )
    return fig


def main():
    out_dir = pick_existing(OUT_DIRS)
    if out_dir is None:
        raise FileNotFoundError("Could not find results directory.")
    print(f"Loading saved outputs from: {out_dir}")
    saved = load_saved(out_dir)
    fig = make_figure(saved)
    png_path = out_dir / "fig456_empirical_vs_pred.png"
    fig.savefig(png_path, bbox_inches="tight")
    print(f"Saved figure: {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
