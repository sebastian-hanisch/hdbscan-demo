"""Rand-Index und Noise-Anteil je wahrer Gruppe - dieselben Kennzahlen wie in
agglomerative-demo (Rand-Index) und dbscan-demo (per_group_noise_fraction), hier
eingesetzt um live nachzuweisen, dass HDBSCAN beide geerbten Härtefälle löst."""

import numpy as np


def rand_index(true_labels, pred_labels):
    """Anteil der Punktpaare, bei denen beide Partitionen uebereinstimmen. Punkte mit
    true_label -1 (Bruecken-/Ausreisserpunkte ohne echte Gruppenzugehoerigkeit) werden
    ausgeschlossen."""
    true_arr = np.asarray(true_labels)
    pred_arr = np.asarray(pred_labels)
    mask = true_arr != -1
    t, p = true_arr[mask], pred_arr[mask]
    n = len(t)
    if n < 2:
        return 1.0
    iu = np.triu_indices(n, k=1)
    same_true = (t[:, None] == t[None, :])[iu]
    same_pred = (p[:, None] == p[None, :])[iu]
    return float(np.mean(same_true == same_pred))


def per_group_noise_fraction(true_labels, final_labels):
    """Anteil Noise je wahrer Gruppe (echte Bruecken-/Ausreisserpunkte mit true_label -1
    ausgenommen)."""
    true_arr = np.asarray(true_labels)
    final_arr = np.asarray(final_labels)
    groups = sorted(set(true_arr.tolist()) - {-1})
    return {g: float(np.mean(final_arr[true_arr == g] == -1)) for g in groups}
