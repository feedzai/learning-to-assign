import numpy as np


def randomly_invert_labels(labels, p, seed):
    rng = np.random.default_rng(seed=seed)
    mask = rng.choice(a=[False, True], p=[1-p, p], size=labels.shape)

    new_labels = labels.copy()
    new_labels[mask] = np.abs(new_labels[mask] - 1)  # inverts binary labels

    return new_labels

def invert_labels_with_probabilities(labels_arr, p_arr, seed):
    rng = np.random.default_rng(seed=seed)
    mask = rng.binomial(n=1, p=p_arr).astype(bool)

    new_labels = labels_arr.copy()
    new_labels[mask] = np.abs(new_labels[mask] - 1)  # inverts labels

    return new_labels
