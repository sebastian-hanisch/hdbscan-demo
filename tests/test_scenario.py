import numpy as np

from hb_scenario import generate_instance


def test_point_count_matches_request_including_bridge():
    instance = generate_instance(n_points=100, k=2, spread=0.2, density_imbalance=0.0, bridge_strength=0.5, seed=1)
    assert instance.n_points == 100 + 40  # 50% von MAX_BRIDGE_POINTS (80) = 40
    assert instance.true_labels.count(-1) == 40


def test_no_bridge_by_default_strength_zero():
    instance = generate_instance(n_points=60, k=3, spread=0.2, density_imbalance=0.0, bridge_strength=0.0, seed=1)
    assert -1 not in instance.true_labels


def test_reproducible_given_same_seed():
    kwargs = dict(n_points=80, k=2, spread=0.2, density_imbalance=0.5, bridge_strength=0.3, seed=42)
    a = generate_instance(**kwargs)
    b = generate_instance(**kwargs)
    assert a.points == b.points
    assert a.true_labels == b.true_labels


def test_density_imbalance_and_bridge_work_independently_and_together():
    only_density = generate_instance(
        n_points=150, k=2, spread=0.15, density_imbalance=0.9, bridge_strength=0.0, seed=3
    )
    only_bridge = generate_instance(
        n_points=100, k=2, spread=0.225, density_imbalance=0.0, bridge_strength=0.32, seed=2
    )
    both = generate_instance(n_points=150, k=2, spread=0.2, density_imbalance=0.7, bridge_strength=0.3, seed=5)

    assert -1 not in only_density.true_labels
    assert only_bridge.true_labels.count(-1) > 0
    assert both.true_labels.count(-1) > 0


def test_density_imbalance_lowers_local_density_of_group_zero():
    instance = generate_instance(n_points=150, k=3, spread=0.2, density_imbalance=0.9, bridge_strength=0.0, seed=3)
    points = np.array(instance.points)
    labels = np.array(instance.true_labels)

    def mean_nn_distance(group_points):
        d = np.sqrt(((group_points[:, None, :] - group_points[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1).mean()

    nn_group0 = mean_nn_distance(points[labels == 0])
    nn_others = np.mean([mean_nn_distance(points[labels == i]) for i in (1, 2)])
    assert nn_group0 > nn_others * 1.5
