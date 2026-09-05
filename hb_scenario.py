"""Zufällige 2D-Punktwolken für die HDBSCAN-Demo: k Gauß-Cluster mit einstellbarem
Dichte-Ungleichgewicht (dbscan-demo-Mechanik) UND optionaler Punktbrücke zwischen den
ersten beiden Clustern (agglomerative-demo-Mechanik) - beide unabhängig einstellbar, damit
auch ein kombinierter Härtefall (beide Effekte gleichzeitig) möglich ist."""

from dataclasses import dataclass

import numpy as np

from hb_constants import MAX_BRIDGE_POINTS, RING_RADIUS

MIN_STD_FRACTION = 0.05


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple  # ((x, y), ...)
    true_labels: tuple  # Gruppenindex, oder -1 fuer Bruecken-/Ausreisserpunkte
    k: int

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def _cluster_stds(k, spread, density_imbalance):
    """Wie dbscan-demo/db_scenario.py: Cluster 0 wird mit wachsendem density_imbalance
    diffuser (groessere Streuung bei gleicher Punktzahl, also geringere lokale Dichte)."""
    base_std = max(spread, MIN_STD_FRACTION) * RING_RADIUS
    stds = np.full(k, base_std)
    if k > 1:
        stds[0] *= 1 + density_imbalance
        stds[1:] *= max(1 - 0.6 * density_imbalance, MIN_STD_FRACTION)
    return stds


def _generate_blobs(n_points, k, spread, density_imbalance, rng):
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.15, 0.15, size=k)
    centers = np.stack([RING_RADIUS * np.cos(angles), RING_RADIUS * np.sin(angles)], axis=1)
    stds = _cluster_stds(k, spread, density_imbalance)

    counts = np.full(k, n_points // k)
    counts[-1] += n_points - counts.sum()

    points_per_cluster, labels_per_cluster = [], []
    for i in range(k):
        pts = rng.normal(loc=centers[i], scale=stds[i], size=(counts[i], 2))
        points_per_cluster.append(pts)
        labels_per_cluster.append(np.full(counts[i], i))
    return np.concatenate(points_per_cluster, axis=0), np.concatenate(labels_per_cluster, axis=0), centers


def _add_bridge(points, labels, center_a, center_b, bridge_strength, rng):
    """Wie agglomerative-demo/ag_scenario.py: zusaetzliche Punkte entlang der vollen
    Verbindungslinie zwischen zwei Zentren - das Vehikel fuer Single-Linkage-Chaining,
    hier zusaetzlich mit density_imbalance kombinierbar."""
    n_bridge = int(round(bridge_strength * MAX_BRIDGE_POINTS))
    if n_bridge <= 0:
        return points, labels
    t = rng.uniform(0.0, 1.0, n_bridge)
    base = center_a[None, :] + t[:, None] * (center_b - center_a)[None, :]
    jitter_std = 0.08 * RING_RADIUS
    bridge_points = base + rng.normal(scale=jitter_std, size=base.shape)
    bridge_labels = np.full(n_bridge, -1)
    return np.concatenate([points, bridge_points]), np.concatenate([labels, bridge_labels])


def generate_instance(n_points, k, spread, density_imbalance, bridge_strength, seed):
    rng = np.random.default_rng(seed)
    points, labels, centers = _generate_blobs(n_points, k, spread, density_imbalance, rng)
    if k >= 2 and bridge_strength > 0:
        points, labels = _add_bridge(points, labels, centers[0], centers[1], bridge_strength, rng)
    return ClusteringInstance(
        points=tuple(map(tuple, points.tolist())),
        true_labels=tuple(int(l) for l in labels),
        k=k,
    )
