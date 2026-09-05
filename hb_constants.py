"""Defaults, Regler-Grenzen, Sicherheitsgrenzen und Presets für die HDBSCAN-Demo."""

DEFAULT_N_POINTS = 120
DEFAULT_K = 2
DEFAULT_SPREAD = 0.15
DEFAULT_DENSITY_IMBALANCE = 0.0
DEFAULT_BRIDGE_STRENGTH = 0.0
DEFAULT_SEED = 1
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MIN_SAMPLES = 5

N_POINTS_MIN, N_POINTS_MAX = 30, 200
K_MIN, K_MAX = 2, 6
SPREAD_MIN, SPREAD_MAX = 0.05, 0.9
DENSITY_IMBALANCE_MIN, DENSITY_IMBALANCE_MAX = 0.0, 1.0
BRIDGE_STRENGTH_MIN, BRIDGE_STRENGTH_MAX = 0.0, 1.0
MIN_CLUSTER_SIZE_MIN, MIN_CLUSTER_SIZE_MAX = 3, 20
MIN_SAMPLES_MIN, MIN_SAMPLES_MAX = 2, 15
MAX_BRIDGE_POINTS = 80

RING_RADIUS = 2.0

PRESETS = {
    "Einfaches Beispiel": {
        "n_points": 90, "k": 3, "spread": 0.15, "density_imbalance": 0.0,
        "bridge_strength": 0.0, "min_cluster_size": 5, "min_samples": 5, "seed": 1,
    },
    "Der Fall, an dem DBSCAN scheiterte": {
        "n_points": 150, "k": 2, "spread": 0.15, "density_imbalance": 0.9,
        "bridge_strength": 0.0, "min_cluster_size": 5, "min_samples": 5, "seed": 3,
    },
    "Der Fall, an dem Single-Linkage scheiterte": {
        "n_points": 100, "k": 2, "spread": 0.225, "density_imbalance": 0.0,
        "bridge_strength": 0.32, "min_cluster_size": 5, "min_samples": 5, "seed": 2,
    },
    "Kombinierter Härtefall": {
        "n_points": 150, "k": 2, "spread": 0.2, "density_imbalance": 0.7,
        "bridge_strength": 0.3, "min_cluster_size": 5, "min_samples": 5, "seed": 5,
    },
}
