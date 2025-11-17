import os
import re
import numpy as np
from typing import Dict, Iterable, List, Sequence, Tuple, Optional
from scipy.io import loadmat

# -----------------------------------------------------------------------------
# Helpers to read MATLAB structs
# -----------------------------------------------------------------------------


def _load_mat(path: str):
    """
    Load a MATLAB .mat file robustly across SciPy versions.
    - simplify_cells=True makes MATLAB cells -> Python lists, structs -> dicts (SciPy >=1.4)
    - Fallback path supports older SciPy.
    """
    try:
        return loadmat(path, simplify_cells=True)
    except TypeError:
        return loadmat(path, squeeze_me=True, struct_as_record=False)


def _get_trial_windows(tsa: Dict) -> Dict[int, Tuple[float, float]]:
    """
    Extract per-trial begin/end times from `obj.timeSeriesArrayHash.value{1,1}`-like dict.

    Parameters
    ----------
    tsa : dict
        Expects fields 'trial' (vector of trial IDs) and 'time' (same length)

    Returns
    -------
    dict
        {trial_id: (t_begin, t_end)} with times in seconds.
    """
    trial_vec = np.asarray(tsa["trial"]).astype(int)
    time_vec = np.asarray(tsa["time"]).astype(float)
    trial_ids = np.unique(trial_vec)
    windows: Dict[int, Tuple[float, float]] = {}
    for tid in trial_ids:
        idx = trial_vec == tid
        if np.any(idx):
            t0 = float(time_vec[idx].min())
            t1 = float(time_vec[idx].max())
            if t1 > t0:
                windows[int(tid)] = (t0, t1)
    return windows


def _iter_neurons(es_value: Iterable):
    """
    Iterate over neurons from `obj.eventSeriesHash.value` which may be nested.

    Yields
    ------
    dict
        Neuron/event dicts with keys: 'eventTrials', 'eventTimes', 'cellType'
    """
    if isinstance(es_value, dict):
        es_value = es_value.get("value", es_value)
    if isinstance(es_value, np.ndarray):
        es_value = es_value.tolist()
    if len(es_value) == 1 and isinstance(es_value[0], (list, np.ndarray)):
        es_value = es_value[0]
    for es in es_value:
        yield es


def _is_pyramidal(cell_type) -> bool:
    """Return True if cell_type indicates a pyramidal neuron."""
    if cell_type is None:
        return False
    if isinstance(cell_type, str):
        return cell_type.lower() == "pyramidal"
    if isinstance(cell_type, (list, tuple)) and len(cell_type) > 0:
        return str(cell_type[0]).lower() == "pyramidal"
    return False


# -----------------------------------------------------------------------------
# Binning
# -----------------------------------------------------------------------------


def _bin_trial_event_times(
    event_times_rel: np.ndarray, bin_size: float, T_bins: int
) -> np.ndarray:
    """
    Bin spike times relative to trial start into fixed-width bins.

    Parameters
    ----------
    event_times_rel : 1D array-like
        Spike times (s) relative to trial start.
    bin_size : float
        Bin width in seconds.
    T_bins : int
        Number of bins to keep.

    Returns
    -------
    np.ndarray
        1D float32 array of length T_bins with spike counts per bin.
    """
    event_times_rel = np.asarray(event_times_rel, dtype=float)
    if event_times_rel.size == 0 or T_bins <= 0:
        return np.zeros(T_bins, dtype=np.float32)
    mask = (event_times_rel >= 0) & (event_times_rel < bin_size * T_bins)
    if not np.any(mask):
        return np.zeros(T_bins, dtype=np.float32)
    bins = np.floor(event_times_rel[mask] / bin_size).astype(int)
    bins = bins[bins < T_bins]
    counts = np.bincount(bins, minlength=T_bins).astype(np.float32)
    return counts


# -----------------------------------------------------------------------------
# Per-animal builder (as provided, slightly tidied, returns arrays + meta)
# -----------------------------------------------------------------------------


def build_pseudopop_for_animal_from_raw(
    data_root: str,
    animal_id: str,
    n_trials: int = 20,
    bin_size: float = 0.010,  # 10 ms
    left_row: int = 0,
    right_row: int = 1,
    stim_row: int = 6,
    early_row: int = 7,
    rng: Optional[np.random.Generator] = None,
):
    """
    Build a per-animal pseudopopulation directly from raw .mat sessions.

    Combines ALL sessions for the animal; keeps only pyramidal neurons; filters to
    correct left/right trials excluding stim & early; aligns to a GLOBAL truncation
    equal to the shortest duration across all included trials (L+R, all sessions).
    Requires >= n_trials non-silent trials in BOTH L and R per neuron; randomly
    samples exactly n_trials from each.

    Returns
    -------
    left_arr  : (n_trials, T_bins, N_d)
    right_arr : (n_trials, T_bins, N_d)
    both_arr  : (n_trials, 2*T_bins, N_d)
    meta      : dict with bookkeeping
    """
    if rng is None:
        rng = np.random.default_rng()

    # Find all sessions for this animal (letter after date optional)
    # pat = re.compile(rf"^data_structure_{re.escape(animal_id)}_(\d{{8}}[a-z]?)\.mat$")
    pat = re.compile(rf"^{re.escape(animal_id)}_(\d{{8}}[a-z]?)\.mat$")
    session_paths: List[str] = []
    for dirpath, _, files in os.walk(data_root):
        for fname in files:
            if pat.match(fname):
                session_paths.append(os.path.join(dirpath, fname))
    session_paths.sort()
    if not session_paths:
        raise FileNotFoundError(f"No sessions for {animal_id} under {data_root}")

    # ---------------- PASS 1: decide global truncation (T_bins) ---------------- #
    min_duration = None  # in seconds
    session_info: List[Dict] = []  # cache trial masks & windows

    for spath in session_paths:
        mat = _load_mat(spath)
        obj = mat.get("obj", mat)  # sometimes fields live at top-level; be permissive

        # trialTypeMat can live in obj or top-level
        ttm = obj.get("trialTypeMat", mat.get("trialTypeMat", None))
        if ttm is None:
            raise KeyError(f"trialTypeMat not found in {spath}")
        trialTypeMat = np.asarray(ttm)

        # time series array: obj.timeSeriesArrayHash.value{1,1}
        tsa_value = obj["timeSeriesArrayHash"]["value"]
        tsa = tsa_value[0] if isinstance(tsa_value, list) else tsa_value
        trial_windows = _get_trial_windows(tsa)

        # select trial indices (MATLAB columns) to keep
        try:
            left_mask = trialTypeMat[left_row, :].astype(bool)
            right_mask = trialTypeMat[right_row, :].astype(bool)
            stim_mask = trialTypeMat[stim_row, :].astype(bool)
            early_mask = trialTypeMat[early_row, :].astype(bool)
        except Exception as e:
            raise IndexError(f"trialTypeMat indexing failed in {spath}: {e}")

        keep_left_cols = (
            np.where(left_mask & ~stim_mask & ~early_mask)[0] + 1
        )  # MATLAB->Python IDs
        keep_right_cols = np.where(right_mask & ~stim_mask & ~early_mask)[0] + 1

        durations: List[float] = []
        for tid in np.concatenate([keep_left_cols, keep_right_cols]):
            win = trial_windows.get(int(tid), None)
            if win is not None:
                durations.append(win[1] - win[0])
        if durations:
            sess_min = float(np.min(durations))
            min_duration = (
                sess_min if min_duration is None else min(min_duration, sess_min)
            )

        session_info.append(
            {
                "path": spath,
                "trial_windows": trial_windows,  # dict tid -> (t0, t1)
                "keep_left_tids": keep_left_cols.astype(int),
                "keep_right_tids": keep_right_cols.astype(int),
            }
        )

    if min_duration is None or min_duration <= 0:
        raise RuntimeError(
            "Could not determine a positive minimal trial duration across sessions."
        )
    T_bins = int(np.floor(min_duration / bin_size))
    if T_bins <= 0:
        raise RuntimeError(
            f"Computed T_bins={T_bins} from min_duration={min_duration:.6f}s and bin_size={bin_size}s"
        )

    # ---------------- PASS 2: bin per neuron & select trials ---------------- #
    left_blocks: List[np.ndarray] = []
    right_blocks: List[np.ndarray] = []
    neuron_sources: List[Tuple[str, int]] = []  # (session_path, neuron_idx)

    for info in session_info:
        spath = info["path"]
        mat = _load_mat(spath)
        obj = mat.get("obj", mat)

        # event series list (one per neuron)
        es_list = list(_iter_neurons(obj["eventSeriesHash"]["value"]))
        N = len(es_list)
        if N == 0:
            continue

        trial_windows = info["trial_windows"]
        keep_L = info["keep_left_tids"]
        keep_R = info["keep_right_tids"]

        for n_idx, es in enumerate(es_list):
            if not _is_pyramidal(es.get("cellType", None)):
                continue

            ev_trials = np.asarray(es.get("eventTrials", []), dtype=int).ravel()
            ev_times = np.asarray(es.get("eventTimes", []), dtype=float).ravel()

            # collect per-trial binned rows
            L_rows: List[np.ndarray] = []
            R_rows: List[np.ndarray] = []

            # LEFT trials
            for tid in keep_L:
                win = trial_windows.get(int(tid), None)
                if win is None:
                    continue
                t0 = win[0]
                idx = ev_trials == int(tid)
                rel_times = ev_times[idx] - t0
                counts = _bin_trial_event_times(rel_times, bin_size, T_bins)
                L_rows.append(counts)

            # RIGHT trials
            for tid in keep_R:
                win = trial_windows.get(int(tid), None)
                if win is None:
                    continue
                t0 = win[0]
                idx = ev_trials == int(tid)
                rel_times = ev_times[idx] - t0
                counts = _bin_trial_event_times(rel_times, bin_size, T_bins)
                R_rows.append(counts)

            if not L_rows or not R_rows:
                continue

            L_mat = np.stack(L_rows, axis=0)  # [nL, T_bins]
            R_mat = np.stack(R_rows, axis=0)  # [nR, T_bins]

            # "non-silent" trials (at least one spike)
            L_nonzero = np.where(L_mat.sum(axis=1) > 0)[0]
            R_nonzero = np.where(R_mat.sum(axis=1) > 0)[0]

            if L_nonzero.size < n_trials or R_nonzero.size < n_trials:
                continue

            # Randomly pick exactly n_trials from each side
            L_pick = rng.choice(L_nonzero, size=n_trials, replace=False)
            R_pick = rng.choice(R_nonzero, size=n_trials, replace=False)

            left_blocks.append(L_mat[L_pick, :][:, :, None])
            right_blocks.append(R_mat[R_pick, :][:, :, None])
            neuron_sources.append((spath, n_idx))

    if not left_blocks:
        raise RuntimeError(
            f"No pyramidal neurons met the ≥{n_trials} trials per side criterion for {animal_id}."
        )

    left_arr = np.concatenate(left_blocks, axis=2)  # [n_trials, T_bins, N_d]
    right_arr = np.concatenate(right_blocks, axis=2)  # [n_trials, T_bins, N_d]
    both_arr = np.concatenate(
        [left_arr, right_arr], axis=1
    )  # [n_trials, 2*T_bins, N_d]

    meta = {
        "animal_id": animal_id,
        "bin_size_s": float(bin_size),
        "T_bins": int(T_bins),
        "min_duration_s": float(min_duration),
        "n_trials_per_side": int(n_trials),
        "N_neurons": int(left_arr.shape[2]),
        "sessions_used": [s["path"] for s in session_info],
        "neuron_sources": neuron_sources,
        "rows": {
            "left_row": left_row,
            "right_row": right_row,
            "stim_row": stim_row,
            "early_row": early_row,
        },
    }
    return left_arr, right_arr, both_arr, meta


# -----------------------------------------------------------------------------
# Cross-animal builder (efficient): build per animal, then truncate at the end
# -----------------------------------------------------------------------------


def _list_animal_ids_by_subfolders(data_root: str) -> List[str]:
    """
    Treat each subdirectory of `data_root` as an animal ID.
    Returns a sorted list of folder names that are directories.
    """
    ids: List[str] = []
    for name in os.listdir(data_root):
        path = os.path.join(data_root, name)
        if os.path.isdir(path) and not name.startswith("."):
            ids.append(name)
    return sorted(ids)


def build_cross_animal_pseudopop(
    data_root: str,
    animal_ids: Optional[Sequence[str]] = None,
    n_trials: int = 20,
    bin_size: float = 0.010,
    rng: Optional[np.random.Generator] = None,
    expect_same_bin_size: bool = True,
    verbose: bool = True,
):
    """
    Build a cross-animal pseudopopulation by:
      1) Building each animal independently (one pass each, fast).
      2) Truncating all animals' time axes to the global minimum T_bins.
      3) Concatenating along the neuron axis.

    This avoids a redundant global PASS 1 while keeping time alignment valid
    (all trials align at t=0; trimming only removes late bins).

    Parameters
    ----------
    data_root : str
        Root directory that contains per-animal subfolders.
    animal_ids : sequence of str or None
        If None, uses immediate subfolders of `data_root` as animal IDs.
    n_trials : int
        Number of non-silent trials to sample per side per neuron.
    bin_size : float
        Bin width (seconds). Should match across animals.
    rng : np.random.Generator or None
        RNG for reproducible trial sampling.
    expect_same_bin_size : bool
        If True, sanity-check that per-animal metas report the same bin size.
    verbose : bool
        If True, print progress and decisions.

    Returns
    -------
    left_all  : (n_trials, T_bins_min, N_total_neurons)
    right_all : (n_trials, T_bins_min, N_total_neurons)
    both_all  : (n_trials, 2*T_bins_min, N_total_neurons)
    meta_all  : dict
        Combined metadata including per-animal contributions and truncation info.
    """
    if rng is None:
        rng = np.random.default_rng()

    if animal_ids is None:
        animal_ids = _list_animal_ids_by_subfolders(data_root)
        if verbose:
            print(
                f"Discovered {len(animal_ids)} animal(s): {', '.join(animal_ids) if animal_ids else 'NONE'}"
            )

    left_blocks_all: List[np.ndarray] = []
    right_blocks_all: List[np.ndarray] = []
    both_blocks_all: List[np.ndarray] = []
    metas: List[Dict] = []

    for aid in animal_ids:
        animal_root = os.path.join(data_root, aid)
        if not os.path.isdir(animal_root):
            if verbose:
                print(f"Skipping {aid}: not a directory under {data_root}")
            continue
        try:
            left_arr, right_arr, both_arr, meta = build_pseudopop_for_animal_from_raw(
                data_root=animal_root,
                animal_id=aid,
                n_trials=n_trials,
                bin_size=bin_size,
                rng=rng,
            )
        except (FileNotFoundError, RuntimeError, KeyError, IndexError) as e:
            if verbose:
                print(f"Skipping {aid}: {e}")
            continue

        if (
            expect_same_bin_size
            and abs(meta.get("bin_size_s", bin_size) - bin_size) > 1e-12
        ):
            raise ValueError(
                f"Bin size mismatch for {aid}: meta has {meta.get('bin_size_s')} vs requested {bin_size}"
            )

        if verbose:
            print(f"Animal {aid}: T_bins={meta['T_bins']}, neurons={meta['N_neurons']}")

        left_blocks_all.append(left_arr)
        right_blocks_all.append(right_arr)
        both_blocks_all.append(both_arr)  # we'll re-make this after truncation anyway
        metas.append(meta)

    if not left_blocks_all:
        raise RuntimeError("No valid animals yielded pseudopop data.")

    # Determine global minimum T across animals (based on left/right which match per animal)
    T_bins_per_animal = [arr.shape[1] for arr in left_blocks_all]
    min_T_bins = int(min(T_bins_per_animal))
    if verbose:
        print(f"Truncating all animals to min T_bins = {min_T_bins}")

    # Truncate all arrays to min_T_bins and rebuild both-arrays accordingly
    left_blocks_all = [arr[:, :min_T_bins, :] for arr in left_blocks_all]
    right_blocks_all = [arr[:, :min_T_bins, :] for arr in right_blocks_all]
    both_blocks_all = [
        np.concatenate([L, R], axis=1)
        for L, R in zip(left_blocks_all, right_blocks_all)
    ]

    # Concatenate along neuron axis
    left_all = np.concatenate(left_blocks_all, axis=2)
    right_all = np.concatenate(right_blocks_all, axis=2)
    both_all = np.concatenate(both_blocks_all, axis=2)

    # Build combined meta
    meta_all = {
        "n_trials_per_side": int(n_trials),
        "bin_size_s": float(bin_size),
        "T_bins_min": int(min_T_bins),
        "animals": [m["animal_id"] for m in metas],
        "N_neurons_per_animal": [int(m["N_neurons"]) for m in metas],
        "N_total_neurons": int(left_all.shape[2]),
        "per_animal": metas,
        "truncation": {
            "strategy": "post-hoc",
            "note": "Truncated arrays after per-animal build to global min T_bins; alignment preserved from t=0.",
            "min_T_bins": int(min_T_bins),
            "T_bins_per_animal": T_bins_per_animal,
        },
    }

    return left_all, right_all, both_all, meta_all


# -----------------------------------------------------------------------------
# Optional: simple CLI / usage example (safe to remove in library use)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build cross-animal pseudopopulation")
    parser.add_argument(
        "data_root",
        type=str,
        nargs="?",
        default=os.getcwd(),
        help="Root directory with per-animal subfolders (default: current directory)",
    )
    parser.add_argument(
        "--n_trials", type=int, default=20, help="Trials per side per neuron"
    )
    parser.add_argument(
        "--bin_size", type=float, default=0.010, help="Bin size in seconds"
    )
    parser.add_argument(
        "--animals",
        type=str,
        nargs="*",
        default=None,
        help="Explicit list of animal IDs (subfolder names)",
    )
    parser.add_argument(
        "--no_verbose", action="store_true", help="Silence progress prints"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(12345)

    left_all, right_all, both_all, meta_all = build_cross_animal_pseudopop(
        data_root=args.data_root,
        animal_ids=args.animals,
        n_trials=args.n_trials,
        bin_size=args.bin_size,
        rng=rng,
        verbose=not args.no_verbose,
    )

    np.save(os.path.join(args.data_root, "preformatted_data.npy"), both_all)

    # --- Build G matrix ---
    num_animals = len(meta_all["animals"])
    total_neurons = meta_all["N_total_neurons"]
    G = np.zeros((num_animals, total_neurons, total_neurons), dtype=np.float32)

    start_idx = 0
    for d, n_neurons in enumerate(meta_all["N_neurons_per_animal"]):
        end_idx = start_idx + n_neurons
        for i in range(start_idx, end_idx):
            G[d, i, i] = 1.0
        start_idx = end_idx

    # Save G
    np.save(os.path.join(args.data_root, "G.npy"), G)

    print("\nSummary:")
    print(f"  Animals: {meta_all['animals']}")
    print(f"  Total neurons: {meta_all['N_total_neurons']}")
    print(f"  T_bins (post-truncation): {meta_all['T_bins_min']}")
    print(f"  left_all shape : {left_all.shape}")
    print(f"  right_all shape: {right_all.shape}")
    print(f"  both_all shape : {both_all.shape}")
