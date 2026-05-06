import numpy as np


def infer_animals_from_G(G):
    """
    Infer neuron-to-animal assignments from a grouping tensor G.

    Parameters
    ----------
    G : np.ndarray, shape (D, N, N)
        Grouping tensor where G[d, i, i] = 1 if neuron i belongs to animal d,
        and 0 otherwise.

    Returns
    -------
    animal_idx : np.ndarray, shape (N,)
        animal_idx[i] is the animal index assigned to neuron i.
    groups : list[np.ndarray]
        groups[d] contains the neuron indices belonging to animal d.
    """
    G = np.asarray(G)
    if G.ndim != 3 or G.shape[1] != G.shape[2]:
        raise ValueError("G must have shape (D, N, N).")

    D, N, _ = G.shape
    diag_stack = np.stack([np.diag(G[d]) for d in range(D)], axis=0)  # (D, N)
    animal_idx = np.argmax(diag_stack, axis=0)
    groups = [np.where(animal_idx == d)[0] for d in range(D)]
    return animal_idx, groups


def build_subG(G, kept_idxs, order_animals=None):
    """
    Build the reduced grouping tensor after selecting a subset of neurons.

    Parameters
    ----------
    G : np.ndarray, shape (D, N, N)
        Original grouping tensor.
    kept_idxs : array_like
        Indices of neurons to keep.
    order_animals : list[int] | None
        Optional explicit ordering of the kept animals in the output tensor.
        If None, uses the sorted order of animals present in kept_idxs.

    Returns
    -------
    G_sub : np.ndarray, shape (D_new, N_sub, N_sub)
        Grouping tensor restricted to the selected neurons.
    """
    G = np.asarray(G)
    kept_idxs = np.asarray(kept_idxs, dtype=int)

    animal_idx, _ = infer_animals_from_G(G)
    kept_animals = sorted(set(int(animal_idx[i]) for i in kept_idxs))

    if order_animals is None:
        order_animals = kept_animals

    D_new = len(order_animals)
    N_sub = len(kept_idxs)
    G_sub = np.zeros((D_new, N_sub, N_sub), dtype=G.dtype)

    mapping = {old: new for new, old in enumerate(order_animals)}

    for old_d in kept_animals:
        new_d = mapping[old_d]
        block = G[old_d][np.ix_(kept_idxs, kept_idxs)]
        G_sub[new_d] = block

    return G_sub


def neuron_ids_for_base_count(base_order, D_pre, Nbase):
    """
    Choose the first Nbase nested neurons from the first D_pre animals.

    Selection is performed in round-robin order across the first D_pre animals,
    using the per-animal neuron order specified in base_order.

    Parameters
    ----------
    base_order : list[list[int]]
        base_order[a] is the ordered neuron list for animal a.
    D_pre : int
        Number of animals included in the base subset.
    Nbase : int
        Total number of neurons to select.

    Returns
    -------
    idxs : np.ndarray, shape (N_selected,)
        Selected neuron indices.
    """
    pools = [list(base_order[a]) for a in range(D_pre)]
    needed = int(Nbase)
    chosen = []

    pools_copy = [p[:] for p in pools]
    while needed > 0 and sum(len(p) for p in pools_copy) > 0:
        for a in range(D_pre):
            if needed == 0:
                break
            if pools_copy[a]:
                chosen.append(pools_copy[a].pop(0))
                needed -= 1

    return np.array(chosen, dtype=int)


def nested_neuron_ids_for_target(base_order, pre_ids, D_pre, Ncur):
    """
    Extend a base neuron subset to a larger nested target size.

    Starting from pre_ids, add extra neurons in round-robin order from the
    remaining pool of the first D_pre animals.

    Parameters
    ----------
    base_order : list[list[int]]
        base_order[a] is the ordered neuron list for animal a.
    pre_ids : array_like
        Base neuron indices that must be included.
    D_pre : int
        Number of animals included in the nested construction.
    Ncur : int
        Target total number of neurons.

    Returns
    -------
    idxs : np.ndarray, shape (N_selected,)
        Extended nested neuron indices.
    """
    pre_ids = np.asarray(pre_ids, dtype=int)
    pre_set = set(pre_ids.tolist())

    pools = []
    for a in range(D_pre):
        pools.append([x for x in base_order[a] if x not in pre_set])

    needed = max(0, int(Ncur) - len(pre_ids))
    extra = []

    pools_copy = [p[:] for p in pools]
    while needed > 0 and sum(len(p) for p in pools_copy) > 0:
        for a in range(D_pre):
            if needed == 0:
                break
            if pools_copy[a]:
                extra.append(pools_copy[a].pop(0))
                needed -= 1

    return np.r_[pre_ids, np.array(extra, dtype=int)]


def animal_ids_for_base_count(base_order, capacities, Dbase, n_pre_per_animal):
    """
    Choose a base subset for the animals axis.

    Takes the first Dbase animals and keeps up to n_pre_per_animal neurons from
    each animal, clipped by that animal's capacity.

    Parameters
    ----------
    base_order : list[list[int]]
        base_order[a] is the ordered neuron list for animal a.
    capacities : list[int]
        capacities[a] is the number of available neurons for animal a.
    Dbase : int
        Number of animals to include.
    n_pre_per_animal : int
        Number of neurons to keep per animal before clipping.

    Returns
    -------
    idxs : np.ndarray
        Selected neuron indices.
    """
    idxs = []
    for a in range(int(Dbase)):
        take = min(int(n_pre_per_animal), int(capacities[a]))
        idxs.extend(base_order[a][:take])

    return np.array(idxs, dtype=int)


def first_k_values_sorted_unique(x, k):
    """
    Return the first k values of x after sorting and removing duplicates.

    Parameters
    ----------
    x : array_like
        Input values.
    k : int
        Maximum number of unique sorted values to return.

    Returns
    -------
    vals : list[int]
        Sorted unique values, truncated to length k.
    """
    vals = sorted(set(int(v) for v in x))
    return vals[: min(k, len(vals))]