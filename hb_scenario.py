"""Zufällige 2D-Punktwolken für die HDBSCAN-Demo: k Gauß-Cluster ("blobs") oder k
nicht-konvexe Halbkreis-Bögen ("moons", generalisiert wie in dbscan-demo/kmeans-demo),
beide mit einstellbarem Dichte-Ungleichgewicht (dbscan-demo-Mechanik) UND optionaler
Punktbrücke zwischen den ersten beiden Gruppen (agglomerative-demo-Mechanik) - alle drei
Achsen unabhängig einstellbar, damit auch kombinierte Härtefälle möglich sind."""

from dataclasses import dataclass

import numpy as np

from hb_constants import ARC_RADIUS, ARC_RING_RADIUS, MAX_BRIDGE_POINTS, RING_RADIUS

MIN_STD_FRACTION = 0.05


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple  # ((x, y), ...)
    true_labels: tuple  # Gruppenindex, oder -1 fuer Bruecken-/Ausreisserpunkte
    shape: str  # "blobs" oder "moons"
    k: int

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def _cluster_stds(k, density_imbalance, base_std):
    """Wie dbscan-demo/db_scenario.py's _group_scales: Cluster 0 wird mit wachsendem
    density_imbalance diffuser (groessere Streuung bei gleicher Punktzahl, also geringere
    lokale Dichte) - die uebrigen entsprechend enger. `base_std` ist die Basis-Streuung vor
    der Ungleichgewichts-Skalierung (blobs und moons rufen dieselbe Funktion mit
    unterschiedlichem `base_std` auf, statt die Formel je Form zu duplizieren)."""
    stds = np.full(k, base_std)
    if k > 1:
        stds[0] *= 1 + density_imbalance
        stds[1:] *= max(1 - 0.6 * density_imbalance, MIN_STD_FRACTION)
    return stds


def _generate_blobs(n_points, k, spread, density_imbalance, rng):
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.15, 0.15, size=k)
    centers = np.stack([RING_RADIUS * np.cos(angles), RING_RADIUS * np.sin(angles)], axis=1)
    base_std = max(spread, MIN_STD_FRACTION) * RING_RADIUS
    stds = _cluster_stds(k, density_imbalance, base_std)

    counts = np.full(k, n_points // k)
    counts[-1] += n_points - counts.sum()

    points_per_cluster, labels_per_cluster = [], []
    for i in range(k):
        pts = rng.normal(loc=centers[i], scale=stds[i], size=(counts[i], 2))
        points_per_cluster.append(pts)
        labels_per_cluster.append(np.full(counts[i], i))
    return np.concatenate(points_per_cluster, axis=0), np.concatenate(labels_per_cluster, axis=0), centers


def _generate_moons(n_points, k, spread, density_imbalance, rng):
    """k=2: das klassische "two moons"-Beispiel (wie in dbscan-demo/kmeans-demo). k>2: k
    Halbkreis-Bögen wie Blütenblätter auf einem Ring, konkave Seite zum Zentrum.
    density_imbalance wirkt wie bei "blobs": Gruppe 0 wird diffuser (geringere lokale
    Dichte bei GLEICHER Punktzahl), die übrigen entsprechend enger."""
    counts = np.full(k, n_points // k)
    counts[-1] += n_points - counts.sum()
    base_noise_std = max(spread, MIN_STD_FRACTION) * ARC_RADIUS * 0.3
    noise_stds = _cluster_stds(k, density_imbalance, base_noise_std)

    if k == 2:
        t1 = rng.uniform(0, np.pi, counts[0])
        x1 = ARC_RADIUS * np.cos(t1)
        y1 = ARC_RADIUS * np.sin(t1)

        t2 = rng.uniform(0, np.pi, counts[1])
        x2 = ARC_RADIUS * (1 - np.cos(t2))
        y2 = ARC_RADIUS * (0.5 - np.sin(t2))

        pts1 = np.stack([x1, y1], axis=1) + rng.normal(scale=noise_stds[0], size=(counts[0], 2))
        pts2 = np.stack([x2, y2], axis=1) + rng.normal(scale=noise_stds[1], size=(counts[1], 2))
        points = np.concatenate([pts1, pts2], axis=0)
        labels = np.concatenate([np.zeros(counts[0], dtype=int), np.ones(counts[1], dtype=int)])
        centers = np.stack([points[labels == i].mean(axis=0) for i in range(k)])
        return points, labels, centers

    layout_angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.1, 0.1, size=k)
    arc_centers = np.stack(
        [ARC_RING_RADIUS * np.cos(layout_angles), ARC_RING_RADIUS * np.sin(layout_angles)], axis=1
    )

    points_per_group, labels_per_group = [], []
    for i in range(k):
        t = rng.uniform(0, np.pi, counts[i])
        local_x = ARC_RADIUS * np.cos(t)
        local_y = ARC_RADIUS * np.sin(t)
        # Um layout_angle_i + pi rotieren, damit die konkave Seite des Bogens zum
        # Ringzentrum zeigt (Blütenblatt-Anordnung), statt nach außen.
        rot = layout_angles[i] + np.pi
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        rx = cos_r * local_x - sin_r * local_y
        ry = sin_r * local_x + cos_r * local_y
        pts = np.stack([rx, ry], axis=1) + arc_centers[i] + rng.normal(scale=noise_stds[i], size=(counts[i], 2))
        points_per_group.append(pts)
        labels_per_group.append(np.full(counts[i], i))

    points = np.concatenate(points_per_group, axis=0)
    labels = np.concatenate(labels_per_group, axis=0)
    centers = np.stack([points[labels == i].mean(axis=0) for i in range(k)])
    return points, labels, centers


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


def generate_instance(n_points, k, spread, density_imbalance, bridge_strength, seed, shape="blobs"):
    rng = np.random.default_rng(seed)
    if shape == "moons":
        points, labels, centers = _generate_moons(n_points, k, spread, density_imbalance, rng)
    else:
        points, labels, centers = _generate_blobs(n_points, k, spread, density_imbalance, rng)
    if k >= 2 and bridge_strength > 0:
        points, labels = _add_bridge(points, labels, centers[0], centers[1], bridge_strength, rng)
    return ClusteringInstance(
        points=tuple(map(tuple, points.tolist())),
        true_labels=tuple(int(l) for l in labels),
        shape=shape,
        k=k,
    )
