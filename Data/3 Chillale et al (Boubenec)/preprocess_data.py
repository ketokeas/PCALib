import os
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import random
from typing import Dict, List, Tuple, Optional, Set

# =====================
# CONFIGURATION
# =====================
root_dir = Path(os.getcwd())
sessions_folder = root_dir  # folder containing Pelardon/Timanoix subfolders

bin_width = 0.060           # 60 ms bins
min_trials_per_type = 10    # required trials per (condition, stimulus) for a neuron to be kept
animals = ["Pelardon", "Timanoix"]

# Exact stimulus-note sets per animal (as in your original script)
go_stims_by_animal = {
    "Pelardon": {"Stim , [16] , Target , 0dB", "Stim , [20] , Target , 0dB", "Stim , [24] , Target , 0dB"},
    "Timanoix": {"Stim , [4] , Target , 0dB", "Stim , [8] , Target , 0dB", "Stim , [12] , Target , 0dB"}
}
nogo_stims_by_animal = {
    "Pelardon": {"Stim , [4] , Reference", "Stim , [8] , Reference", "Stim , [12] , Reference"},
    "Timanoix": {"Stim , [16] , Reference", "Stim , [20] , Reference", "Stim , [24] , Reference"}
}

# Reproducibility
random.seed(1234)
np.random.seed(1234)


# =====================
# HELPERS
# =====================
def load_spike_file(file_path: Path):
    return loadmat(file_path, squeeze_me=True, struct_as_record=False)

def decode_note(note):
    return note.decode("utf-8") if isinstance(note, (bytes, bytearray)) else note

def parse_level_from_note(note: str) -> Optional[int]:
    try:
        i = note.index('[') + 1
        j = note.index(']', i)
        return int(note[i:j].strip())
    except Exception:
        return None

def map_level_to_note(stim_set: Set[str]) -> Dict[int, str]:
    out = {}
    for s in stim_set:
        lvl = parse_level_from_note(s)
        if lvl is not None:
            out[lvl] = s
    return out

def build_condition_order_for_animal(animal: str) -> List[Tuple[str, str]]:
    """Return desired (cond, stim_note) order for time-concat per animal."""
    go_map = map_level_to_note(go_stims_by_animal[animal])
    nogo_map = map_level_to_note(nogo_stims_by_animal[animal])

    if animal == "Pelardon":
        desired_levels = [4, 8, 12, 16, 20, 24]  # 4-24
    elif animal == "Timanoix":
        desired_levels = [24, 20, 16, 12, 8, 4]  # 24-4 (reverse)
    else:
        raise ValueError(f"Unknown animal {animal}")

    order: List[Tuple[str, str]] = []
    for lvl in desired_levels:
        if lvl in nogo_map:
            order.append(("nogo", nogo_map[lvl]))
        elif lvl in go_map:
            order.append(("go", go_map[lvl]))
        else:
            raise ValueError(f"Level {lvl} not found in provided stimulus sets for {animal}.")
    return order

def classify_trials(exptevents, go_stims: Set[str], nogo_stims: Set[str]) -> Dict[Tuple[str, str], List[int]]:
    events_by_trial: Dict[int, List[str]] = {}
    for evt in exptevents:
        trial = int(evt.Trial)
        note = decode_note(evt.Note)
        events_by_trial.setdefault(trial, []).append(note)

    groups = {("go", stim): [] for stim in go_stims}
    groups.update({("nogo", stim): [] for stim in nogo_stims})

    for trial, notes in events_by_trial.items():
        for stim in go_stims:
            if stim in notes and "LICK,HIT" in notes:
                groups[("go", stim)].append(trial)
        for stim in nogo_stims:
            if stim in notes and "LICK,FA" not in notes:
                groups[("nogo", stim)].append(trial)
    return groups

def extract_trial_times(exptevents):
    start_times, stop_times, stim_onsets = {}, {}, {}
    for evt in exptevents:
        trial = int(evt.Trial)
        note = decode_note(evt.Note)
        if note == "TRIALSTART":
            start_times[trial] = evt.StartTime
        elif note == "TRIALSTOP":
            stop_times[trial] = evt.StartTime
        elif note.startswith("Stim"):
            if trial not in stim_onsets:  # take the first stim as onset
                stim_onsets[trial] = evt.StartTime
    return start_times, stop_times, stim_onsets

def eligible_prepost_for_groups(start_times, stop_times, stim_onsets, groups_trials) -> Tuple[List[float], List[float]]:
    pre_times, post_times = [], []
    all_trials = set()
    for tids in groups_trials.values():
        all_trials.update(tids)
    for t in all_trials:
        if t in start_times and t in stop_times and t in stim_onsets:
            pre = stim_onsets[t] - start_times[t]
            post = stop_times[t] - stim_onsets[t]
            if pre > 0 and post > 0:
                pre_times.append(pre)
                post_times.append(post)
    return pre_times, post_times

def bin_and_align_single_neuron(unitSpikes, trial_ids, stim_onsets, pre_bins, post_bins, bin_width, rate):
    T = pre_bins + post_bins
    binned = np.zeros((len(trial_ids), T), dtype=np.int32)

    spikes = np.array(unitSpikes)
    if spikes.size == 0:
        return None
    trial_num = spikes[0, :]
    spike_times = spikes[1, :] / rate  # convert to seconds

    for k, trial_id in enumerate(trial_ids):
        mask = trial_num == trial_id
        if not np.any(mask):
            continue
        local_times = spike_times[mask] - stim_onsets[trial_id]  # align: time 0 = stimulus onset
        bins = np.floor(local_times / bin_width).astype(int) + pre_bins
        valid = (bins >= 0) & (bins < T)
        if np.any(valid):
            np.add.at(binned[k, :], bins[valid], 1)
    return binned


# =====================
# PASS 1: SCAN to compute GLOBAL pre/post bins across animals/sessions
# =====================
def scan_for_global_bins(animals: List[str]) -> Tuple[int, int]:
    global_pre_times: List[float] = []
    global_post_times: List[float] = []

    for animal in animals:
        animal_path = sessions_folder / animal / "Spike_sorting"
        if not animal_path.exists():
            continue
        cond_order = build_condition_order_for_animal(animal)
        go_stims = {stim for (cond, stim) in cond_order if cond == "go"}
        nogo_stims = {stim for (cond, stim) in cond_order if cond == "nogo"}

        session_folders = [f for f in animal_path.iterdir() if f.is_dir()]
        for sess in session_folders:
            files = list(sess.glob("*_a_CLT.spk.mat"))
            if not files:
                continue
            mat = load_spike_file(files[0])
            start_times, stop_times, stim_onsets = extract_trial_times(mat["exptevents"])
            groups_trials = classify_trials(mat["exptevents"], go_stims, nogo_stims)
            pre_times, post_times = eligible_prepost_for_groups(start_times, stop_times, stim_onsets, groups_trials)
            global_pre_times.extend(pre_times)
            global_post_times.extend(post_times)

    if not global_pre_times or not global_post_times:
        raise RuntimeError("No valid trials with start/stop/stim across sessions; cannot compute global bins.")

    global_pre_bins = int(np.floor(min(global_pre_times) / bin_width))
    global_post_bins = int(np.floor(min(global_post_times) / bin_width))

    if global_pre_bins <= 0 or global_post_bins <= 0:
        raise RuntimeError(f"Computed non-positive global bins: pre={global_pre_bins}, post={global_post_bins}")

    print(f"[GLOBAL] pre_bins={global_pre_bins}, post_bins={global_post_bins}, T={global_pre_bins + global_post_bins}")
    return global_pre_bins, global_post_bins


# =====================
# PASS 2: Process sessions with GLOBAL bins
# =====================
def process_session_with_global_bins(
    file_path: Path,
    stim_orders: List[Tuple[str, str]],
    min_trials_per_type: int,
    pre_bins: int,
    post_bins: int
):
    mat = load_spike_file(file_path)
    rate = mat["rate"]

    # Events & classification
    start_times, stop_times, stim_onsets = extract_trial_times(mat["exptevents"])
    go_stims = {stim for (cond, stim) in stim_orders if cond == "go"}
    nogo_stims = {stim for (cond, stim) in stim_orders if cond == "nogo"}
    groups_trials = classify_trials(mat["exptevents"], go_stims, nogo_stims)

    # Build neuron data per condition
    neuron_data = []
    for cell in mat["sortinfo"]:
        if not hasattr(cell, "__len__"):
            continue
        for unit in cell:
            if not hasattr(unit, "unitSpikes"):
                continue
            cond_binned = {}
            ok = True
            for cond in stim_orders:
                trial_ids = groups_trials[cond]
                if len(trial_ids) < min_trials_per_type:
                    ok = False
                    break
                selected = random.sample(trial_ids, min_trials_per_type)
                binned = bin_and_align_single_neuron(
                    unit.unitSpikes, selected, stim_onsets,
                    pre_bins, post_bins, bin_width, rate
                )
                if binned is None or binned.shape[0] != min_trials_per_type:
                    ok = False
                    break
                cond_binned[cond] = binned  # (trials, T)
            if ok:
                neuron_data.append(cond_binned)

    if not neuron_data:
        return None

    # Stack neurons for each condition → (trials, T, neurons_in_session)
    cond_arrays = {cond: np.stack([nd[cond] for nd in neuron_data], axis=2) for cond in stim_orders}
    return cond_arrays


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    # 1) Global bins across all animals and sessions
    global_pre_bins, global_post_bins = scan_for_global_bins(animals)
    T_global = global_pre_bins + global_post_bins

    per_animal_arrays: Dict[str, np.ndarray] = {}
    per_animal_neuron_counts: Dict[str, int] = {}

    # 2) Process & save per-animal arrays in requested order
    for animal in animals:
        print(f"\nProcessing {animal}")
        stim_orders = build_condition_order_for_animal(animal)  # explicit time order for this animal

        animal_path = sessions_folder / animal / "Spike_sorting"
        if not animal_path.exists():
            print(f"  No path: {animal_path}")
            continue

        session_folders = [f for f in animal_path.iterdir() if f.is_dir()]
        if not session_folders:
            print(f"  No session folders for {animal}")
            continue

        all_session_data = {cond: [] for cond in stim_orders}

        for sess in session_folders:
            files = list(sess.glob("*_a_CLT.spk.mat"))
            if not files:
                continue
            processed = process_session_with_global_bins(
                files[0], stim_orders, min_trials_per_type,
                pre_bins=global_pre_bins, post_bins=global_post_bins
            )
            if processed is None:
                continue
            for cond in stim_orders:
                all_session_data[cond].append(processed[cond])

        # Ensure we have at least one session’s data
        if not all_session_data[stim_orders[0]]:
            print(f"  No valid neurons found for {animal}")
            continue

        # Merge sessions along neurons (axis=2); bins already match globally
        merged_cond_arrays = {
            cond: np.concatenate(all_session_data[cond], axis=2) for cond in stim_orders
        }

        # Concatenate conditions along time in the requested order
        animal_array = np.concatenate([merged_cond_arrays[cond] for cond in stim_orders], axis=1)
        # Save per-animal
        np.save(f"preprocessed_data_{animal}.npy", animal_array)

        per_animal_arrays[animal] = animal_array
        per_animal_neuron_counts[animal] = animal_array.shape[2]

        print(f"  Saved preprocessed_data_{animal}.npy with shape {animal_array.shape} "
              f"# (trials={min_trials_per_type}, time=6*T, neurons)")

    # 3) Combined 2-block array (first + last condition block for each animal), and G mask
    if "Pelardon" in per_animal_arrays and "Timanoix" in per_animal_arrays:
        pel = per_animal_arrays["Pelardon"]   # (trials, 6*T, Np)
        tim = per_animal_arrays["Timanoix"]   # (trials, 6*T, Nt)

        trials = pel.shape[0]
        assert tim.shape[0] == trials, "Mismatch in trials per condition across animals."

        # First and last blocks for each animal (block 0 and block 5 in their own orders)
        b0 = slice(0, T_global)
        b5 = slice(5 * T_global, 6 * T_global)

        pel_2 = np.concatenate([pel[:, b0, :], pel[:, b5, :]], axis=1)  # (trials, 2*T, Np)
        tim_2 = np.concatenate([tim[:, b0, :], tim[:, b5, :]], axis=1)  # (trials, 2*T, Nt)

        combined_2blocks = np.concatenate([pel_2, tim_2], axis=2)        # (trials, 2*T, Np+Nt)
        np.save("preformatted_data.npy", combined_2blocks)
        print(f"\nSaved combined_2blocks.npy with shape {combined_2blocks.shape} "
              f"# (trials, 2*time, neurons_pelardon+neurons_timanoix)")

        # Build G: [2, N, N] with diagonal membership indicators
        Np = per_animal_neuron_counts["Pelardon"]
        Nt = per_animal_neuron_counts["Timanoix"]
        N = Np + Nt

        G = np.zeros((2, N, N), dtype=np.int8)
        G[0, np.arange(Np), np.arange(Np)] = 1               # Pelardon on diag
        G[1, Np + np.arange(Nt), Np + np.arange(Nt)] = 1     # Timanoix on diag

        np.save("G.npy", G)
        print(f"Saved G.npy with shape {G.shape}  # [2, N, N]")

    else:
        print("\nCombined outputs skipped: one or both per-animal arrays are missing.")