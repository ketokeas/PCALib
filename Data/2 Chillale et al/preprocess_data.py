import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from scipy.io import loadmat


# =====================
# CONFIGURATION
# =====================
root_dir = Path(os.getcwd())
sessions_folder = root_dir

bin_width = 0.060  # 60 ms bins
min_trials_per_type = 10
animals = ["Pelardon", "Timanoix"]

rng = np.random.default_rng(1234)


go_stims_by_animal = {
    "Pelardon": {
        "Stim , [16] , Target , 0dB",
        "Stim , [20] , Target , 0dB",
        "Stim , [24] , Target , 0dB",
    },
    "Timanoix": {
        "Stim , [4] , Target , 0dB",
        "Stim , [8] , Target , 0dB",
        "Stim , [12] , Target , 0dB",
    },
}

nogo_stims_by_animal = {
    "Pelardon": {
        "Stim , [4] , Reference",
        "Stim , [8] , Reference",
        "Stim , [12] , Reference",
    },
    "Timanoix": {
        "Stim , [16] , Reference",
        "Stim , [20] , Reference",
        "Stim , [24] , Reference",
    },
}


# =====================
# HELPERS
# =====================
def load_spike_file(file_path: Path):
    return loadmat(file_path, squeeze_me=True, struct_as_record=False)


def decode_note(note):
    return note.decode("utf-8") if isinstance(note, (bytes, bytearray)) else note


def parse_level_from_note(note: str) -> Optional[int]:
    try:
        i = note.index("[") + 1
        j = note.index("]", i)
        return int(note[i:j].strip())
    except Exception:
        return None


def map_level_to_note(stim_set: Set[str]) -> Dict[int, str]:
    out = {}
    for stim in stim_set:
        level = parse_level_from_note(stim)
        if level is not None:
            out[level] = stim
    return out


def build_condition_order_for_animal(animal: str) -> List[Tuple[str, str]]:
    """
    Return a reproducible order of the six stimulus-condition groups.

    The order is not used to concatenate all six groups in time anymore.
    It is only used to define a stable order for pooling Go groups and
    No-Go groups into the trial axis.
    """
    go_map = map_level_to_note(go_stims_by_animal[animal])
    nogo_map = map_level_to_note(nogo_stims_by_animal[animal])

    if animal == "Pelardon":
        desired_levels = [4, 8, 12, 16, 20, 24]
    elif animal == "Timanoix":
        desired_levels = [24, 20, 16, 12, 8, 4]
    else:
        raise ValueError(f"Unknown animal: {animal}")

    order = []
    for level in desired_levels:
        if level in nogo_map:
            order.append(("nogo", nogo_map[level]))
        elif level in go_map:
            order.append(("go", go_map[level]))
        else:
            raise ValueError(f"Level {level} missing for {animal}.")

    return order


def classify_trials(
    exptevents,
    go_stims: Set[str],
    nogo_stims: Set[str],
) -> Dict[Tuple[str, str], List[int]]:
    events_by_trial: Dict[int, List[str]] = {}

    for evt in np.atleast_1d(exptevents):
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

    for evt in np.atleast_1d(exptevents):
        trial = int(evt.Trial)
        note = decode_note(evt.Note)

        if note == "TRIALSTART":
            start_times[trial] = evt.StartTime
        elif note == "TRIALSTOP":
            stop_times[trial] = evt.StartTime
        elif note.startswith("Stim") and trial not in stim_onsets:
            stim_onsets[trial] = evt.StartTime

    return start_times, stop_times, stim_onsets


def eligible_prepost_for_groups(
    start_times,
    stop_times,
    stim_onsets,
    groups_trials,
) -> Tuple[List[float], List[float]]:
    pre_times, post_times = [], []

    all_trials = set()
    for trial_ids in groups_trials.values():
        all_trials.update(trial_ids)

    for trial in all_trials:
        if trial not in start_times or trial not in stop_times or trial not in stim_onsets:
            continue

        pre = stim_onsets[trial] - start_times[trial]
        post = stop_times[trial] - stim_onsets[trial]

        if pre > 0 and post > 0:
            pre_times.append(pre)
            post_times.append(post)

    return pre_times, post_times


def bin_and_align_single_neuron(
    unit_spikes,
    trial_ids,
    stim_onsets,
    pre_bins,
    post_bins,
    bin_width,
    rate,
):
    T = pre_bins + post_bins
    binned = np.zeros((len(trial_ids), T), dtype=np.int32)

    spikes = np.asarray(unit_spikes)
    if spikes.size == 0:
        return binned

    trial_num = spikes[0, :]
    spike_times = spikes[1, :] / rate

    for r, trial_id in enumerate(trial_ids):
        mask = trial_num == trial_id
        if not np.any(mask):
            continue

        local_times = spike_times[mask] - stim_onsets[trial_id]
        bins = np.floor(local_times / bin_width).astype(int) + pre_bins

        valid = (bins >= 0) & (bins < T)
        if np.any(valid):
            np.add.at(binned[r], bins[valid], 1)

    return binned


def iter_units(sortinfo):
    for cell in np.atleast_1d(sortinfo):
        if not hasattr(cell, "__len__"):
            continue
        for unit in np.atleast_1d(cell):
            if hasattr(unit, "unitSpikes"):
                yield unit


# =====================
# PASS 1: GLOBAL PRE/POST WINDOW
# =====================
def scan_for_global_bins(animals: List[str]) -> Tuple[int, int]:
    global_pre_times = []
    global_post_times = []

    for animal in animals:
        animal_path = sessions_folder / animal / "Spike_sorting"
        if not animal_path.exists():
            continue

        stim_order = build_condition_order_for_animal(animal)
        go_stims = {stim for cond, stim in stim_order if cond == "go"}
        nogo_stims = {stim for cond, stim in stim_order if cond == "nogo"}

        for sess in animal_path.iterdir():
            if not sess.is_dir():
                continue

            files = list(sess.glob("*_a_CLT.spk.mat"))
            if not files:
                continue

            mat = load_spike_file(files[0])
            start_times, stop_times, stim_onsets = extract_trial_times(mat["exptevents"])
            groups_trials = classify_trials(mat["exptevents"], go_stims, nogo_stims)

            pre_times, post_times = eligible_prepost_for_groups(
                start_times,
                stop_times,
                stim_onsets,
                groups_trials,
            )

            global_pre_times.extend(pre_times)
            global_post_times.extend(post_times)

    if not global_pre_times or not global_post_times:
        raise RuntimeError("No valid trials found; cannot compute global bins.")

    pre_bins = int(np.floor(min(global_pre_times) / bin_width))
    post_bins = int(np.floor(min(global_post_times) / bin_width))

    if pre_bins <= 0 or post_bins <= 0:
        raise RuntimeError(f"Invalid global bins: pre={pre_bins}, post={post_bins}")

    print(f"[GLOBAL] pre_bins={pre_bins}, post_bins={post_bins}, T={pre_bins + post_bins}")
    return pre_bins, post_bins


# =====================
# PASS 2: PROCESS ONE SESSION
# =====================
def process_session_with_global_bins(
    file_path: Path,
    stim_order: List[Tuple[str, str]],
    min_trials_per_type: int,
    pre_bins: int,
    post_bins: int,
):
    mat = load_spike_file(file_path)
    rate = mat["rate"]

    start_times, stop_times, stim_onsets = extract_trial_times(mat["exptevents"])

    go_stims = {stim for cond, stim in stim_order if cond == "go"}
    nogo_stims = {stim for cond, stim in stim_order if cond == "nogo"}

    groups_trials = classify_trials(mat["exptevents"], go_stims, nogo_stims)

    # Select the same trials for all neurons in this session.
    selected_trials = {}
    for cond in stim_order:
        valid_trials = [
            trial
            for trial in groups_trials[cond]
            if trial in start_times and trial in stop_times and trial in stim_onsets
        ]

        if len(valid_trials) < min_trials_per_type:
            return None

        selected_trials[cond] = rng.choice(
            valid_trials,
            size=min_trials_per_type,
            replace=False,
        ).tolist()

    neuron_data = []
    for unit in iter_units(mat["sortinfo"]):
        cond_binned = {}

        for cond in stim_order:
            cond_binned[cond] = bin_and_align_single_neuron(
                unit.unitSpikes,
                selected_trials[cond],
                stim_onsets,
                pre_bins,
                post_bins,
                bin_width,
                rate,
            )

        neuron_data.append(cond_binned)

    if not neuron_data:
        return None

    return {
        cond: np.stack([nd[cond] for nd in neuron_data], axis=2)
        for cond in stim_order
    }


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    pre_bins, post_bins = scan_for_global_bins(animals)
    T_global = pre_bins + post_bins

    per_animal_arrays: Dict[str, np.ndarray] = {}
    per_animal_neuron_counts: Dict[str, int] = {}

    for animal in animals:
        print(f"\nProcessing {animal}")

        stim_order = build_condition_order_for_animal(animal)
        animal_path = sessions_folder / animal / "Spike_sorting"

        if not animal_path.exists():
            print(f"  Missing folder: {animal_path}")
            continue

        all_session_data = {cond: [] for cond in stim_order}

        for sess in animal_path.iterdir():
            if not sess.is_dir():
                continue

            files = list(sess.glob("*_a_CLT.spk.mat"))
            if not files:
                continue

            processed = process_session_with_global_bins(
                files[0],
                stim_order,
                min_trials_per_type,
                pre_bins,
                post_bins,
            )

            if processed is None:
                continue

            for cond in stim_order:
                all_session_data[cond].append(processed[cond])

        if not all_session_data[stim_order[0]]:
            print(f"  No valid sessions for {animal}")
            continue

        # Merge all sessions along neurons:
        # each condition becomes [10, T, Na].
        merged_cond_arrays = {
            cond: np.concatenate(all_session_data[cond], axis=2)
            for cond in stim_order
        }

        go_conds = [cond for cond in stim_order if cond[0] == "go"]
        nogo_conds = [cond for cond in stim_order if cond[0] == "nogo"]

        if len(go_conds) != 3 or len(nogo_conds) != 3:
            raise RuntimeError(
                f"Expected 3 Go and 3 No-Go groups for {animal}, got "
                f"{len(go_conds)} Go and {len(nogo_conds)} No-Go."
            )

        # Pool the three Go frequencies as additional trials:
        # three [10, T, Na] arrays -> [30, T, Na].
        go_array = np.concatenate(
            [merged_cond_arrays[cond] for cond in go_conds],
            axis=0,
        )

        # Pool the three No-Go frequencies as additional trials:
        # three [10, T, Na] arrays -> [30, T, Na].
        nogo_array = np.concatenate(
            [merged_cond_arrays[cond] for cond in nogo_conds],
            axis=0,
        )

        # Concatenate Go and No-Go along time:
        # [30, T, Na] + [30, T, Na] -> [30, 2T, Na].
        animal_array = np.concatenate([go_array, nogo_array], axis=1)

        expected_shape = (3 * min_trials_per_type, 2 * T_global)
        if animal_array.shape[:2] != expected_shape:
            raise RuntimeError(
                f"Unexpected shape for {animal}: {animal_array.shape}; "
                f"expected first two dimensions {expected_shape}."
            )

        np.save(f"preprocessed_data_{animal}.npy", animal_array)

        per_animal_arrays[animal] = animal_array
        per_animal_neuron_counts[animal] = animal_array.shape[2]

        print(
            f"  Saved preprocessed_data_{animal}.npy with shape {animal_array.shape} "
            f"# [30, 2T, neurons]"
        )

    if all(animal in per_animal_arrays for animal in animals):
        animal_arrays = [per_animal_arrays[animal] for animal in animals]

        first_shape = animal_arrays[0].shape[:2]
        for animal, arr in zip(animals, animal_arrays):
            if arr.shape[:2] != first_shape:
                raise RuntimeError(
                    f"Shape mismatch for {animal}: {arr.shape[:2]} vs {first_shape}"
                )

        # Concatenate animals along neurons:
        # [30, 2T, Np] + [30, 2T, Nt] -> [30, 2T, Np + Nt].
        preformatted_data = np.concatenate(animal_arrays, axis=2)
        np.save("preformatted_data.npy", preformatted_data)

        print(
            f"\nSaved preformatted_data.npy with shape {preformatted_data.shape} "
            f"# [30, 2T, total_neurons]"
        )

        neuron_counts = [per_animal_neuron_counts[animal] for animal in animals]
        N = sum(neuron_counts)

        G = np.zeros((len(animals), N, N), dtype=np.int8)

        start = 0
        for a, count in enumerate(neuron_counts):
            stop = start + count
            idx = np.arange(start, stop)
            G[a, idx, idx] = 1
            start = stop

        np.save("G.npy", G)
        print(f"Saved G.npy with shape {G.shape} # [animals, N, N]")

    else:
        missing = [animal for animal in animals if animal not in per_animal_arrays]
        print(f"\nCombined outputs skipped. Missing animals: {missing}")
