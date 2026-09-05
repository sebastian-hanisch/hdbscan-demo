import pytest

from hb_algorithm import run
from hb_evaluation import per_group_noise_fraction, rand_index
from hb_scenario import generate_instance


def test_rand_index_identical_partitions_is_one():
    assert rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert rand_index([0, 0, 1, 1], [5, 5, 9, 9]) == 1.0


def test_rand_index_excludes_true_label_minus_one():
    with_bridge = rand_index([0, 0, 1, 1, -1], [0, 0, 1, 1, 2])
    without_bridge = rand_index([0, 0, 1, 1], [0, 0, 1, 1])
    assert with_bridge == without_bridge == 1.0


def test_hdbscan_solves_the_density_mismatch_that_broke_dbscan():
    """Kern-Nachweis 1: beim dbscan-demo-Haertefall-Szenario (starkes Dichte-
    Ungleichgewicht) braucht HDBSCAN kein eps und bedient beide Gruppen mit 0% Noise -
    anders als ein einzelnes globales eps bei DBSCAN, das die diffuse Gruppe
    grossteils als Noise einstufte."""
    instance = generate_instance(
        n_points=150, k=2, spread=0.15, density_imbalance=0.9, bridge_strength=0.0, seed=3
    )
    result = run(instance.as_array(), min_cluster_size=5, min_samples=5)
    fractions = per_group_noise_fraction(instance.true_labels, result.final_labels)
    assert all(f < 0.1 for f in fractions.values())


def test_hdbscan_solves_the_chaining_that_broke_single_linkage():
    """Kern-Nachweis 2: beim agglomerative-demo-Haertefall-Szenario (duenne Bruecke)
    erreicht rohes Single-Linkage nur einen Rand-Index nahe 0.5 (Zufallsniveau), waehrend
    HDBSCANs Mutual-Reachability-Distanz die Bruecke aufblaeht und einen Rand-Index nahe
    1.0 erreicht."""
    instance = generate_instance(
        n_points=100, k=2, spread=0.225, density_imbalance=0.0, bridge_strength=0.32, seed=2
    )
    result = run(instance.as_array(), min_cluster_size=5, min_samples=5)
    ri = rand_index(instance.true_labels, result.final_labels)
    assert ri > 0.9


def test_hdbscan_handles_the_combined_hard_case_reasonably_well():
    instance = generate_instance(
        n_points=150, k=2, spread=0.2, density_imbalance=0.7, bridge_strength=0.3, seed=5
    )
    result = run(instance.as_array(), min_cluster_size=5, min_samples=5)
    fractions = per_group_noise_fraction(instance.true_labels, result.final_labels)
    ri = rand_index(instance.true_labels, result.final_labels)
    assert ri > 0.9
    assert all(f < 0.2 for f in fractions.values())
