import mat73
import numpy as np

dataset_names = ["Chewie_CO_20150313.mat", "Mihili_CO_20140304.mat"]
dictionary_keys = ["M1_spikes", "M1_spikes"]
trials_1 = []
trials_2 = []
minimal_time_before_goCue = []
minimal_time_after_goCue = []
goCue_times_1 = []
goCue_times_2 = []
D = len(dataset_names)
N_array = np.zeros(D).astype(int)

for animal_idx, data_name in enumerate(dataset_names):
    data = mat73.loadmat(data_name)["trial_data"]

    spikes = data[dictionary_keys[animal_idx]]
    tgtDir = np.array(data["tgtDir"])
    result = data["result"]
    all_possible_directions = np.unique(tgtDir)

    direction1 = all_possible_directions[0]
    direction2 = all_possible_directions[4]

    trial_indices_type_1 = np.where(tgtDir == direction1)[0]
    trial_indices_type_2 = np.where(tgtDir == direction2)[0]

    goCue_times_1.append(
        np.array([data["idx_goCueTime"][trial] for trial in trial_indices_type_1])
    )
    goCue_times_2.append(
        np.array([data["idx_goCueTime"][trial] for trial in trial_indices_type_2])
    )

    start_times_1 = np.array(
        [data["idx_startTime"][trial] for trial in trial_indices_type_1]
    )
    start_times_2 = np.array(
        [data["idx_startTime"][trial] for trial in trial_indices_type_2]
    )
    minimal_time_before_goCue.append(
        min(
            np.min(goCue_times_1[animal_idx] - start_times_1),
            np.min(goCue_times_2[animal_idx] - start_times_2),
        )
    )

    end_times_1 = np.array(
        [data["idx_endTime"][trial] for trial in trial_indices_type_1]
    )
    end_times_2 = np.array(
        [data["idx_endTime"][trial] for trial in trial_indices_type_2]
    )
    minimal_time_after_goCue.append(
        min(
            np.min(end_times_1 - goCue_times_1[animal_idx]),
            np.min(end_times_2 - goCue_times_2[animal_idx]),
        )
    )

    trials_1.append([spikes[trial] for trial in trial_indices_type_1])
    trials_2.append([spikes[trial] for trial in trial_indices_type_2])

    N_array[animal_idx] = np.shape(trials_1[animal_idx][0])[1]

n_trials = min(
    np.min([len(trials_1[animal_idx]) for animal_idx in range(D)]),
    np.min([len(trials_2[animal_idx]) for animal_idx in range(D)]),
)
before_goCue = int(min(minimal_time_before_goCue))
after_goCue = int(max(minimal_time_after_goCue))

N_total = int(np.sum(N_array))
G = np.zeros([D, N_total, N_total])

for animal_idx in range(D):
    N_already_used = np.sum(N_array[:animal_idx])
    G[
        animal_idx,
        N_already_used : N_already_used + N_array[animal_idx],
        N_already_used : N_already_used + N_array[animal_idx],
    ] = np.eye(N_array[animal_idx])

    local_trials_1 = trials_1[animal_idx]
    local_trials_2 = trials_2[animal_idx]
    # crop all trials to have the same before and after length
    local_trials_1 = [
        local_trials_1[trial][
            int(goCue_times_1[animal_idx][trial] - before_goCue) : int(
                goCue_times_1[animal_idx][trial] + after_goCue
            ),
            :,
        ]
        for trial in range(len(local_trials_1))
    ]
    local_trials_2 = [
        local_trials_2[trial][
            int(goCue_times_2[animal_idx][trial] - before_goCue) : int(
                goCue_times_2[animal_idx][trial] + after_goCue
            ),
            :,
        ]
        for trial in range(len(local_trials_2))
    ]
    # select random n_trials trials
    indices1 = np.random.permutation(len(local_trials_1))[:n_trials]
    indices2 = np.random.permutation(len(local_trials_2))[:n_trials]

    trials_1[animal_idx] = [local_trials_1[trial] for trial in indices1]
    trials_2[animal_idx] = [local_trials_2[trial] for trial in indices2]

full_data = np.concatenate(
    [np.concatenate(trials_1, axis=2), np.concatenate(trials_2, axis=2)], axis=1
)
np.save("preformatted_data", full_data)
np.save("G_array", G)
