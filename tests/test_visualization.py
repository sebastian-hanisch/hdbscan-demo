"""Regressionstest: die rohe Single-Linkage-Animation durchlaeuft frueh Schritte mit
weit mehr als len(CLUSTER_PALETTE) gleichzeitig existierenden Clustern - ein Modulo auf
die feste Palette machte dort GENUINELY verschiedene Cluster faelschlich farbgleich
(gefunden 2026-09-05, beim Untersuchen eines Nutzer-Bugreports zu Preset 2)."""

import numpy as np
import pytest

from hb_algorithm import labels_at_step, run
from hb_scenario import generate_instance
from hb_visualization import build_scatter_figure


def test_every_cluster_gets_a_distinct_color_even_with_many_clusters():
    instance = generate_instance(150, 2, 0.15, 0.9, 0.0, seed=3)
    result = run(instance.as_array(), min_cluster_size=5, min_samples=5)
    max_step = len(result.merges) - 1

    for step in range(0, max_step + 1, 5):
        labels = labels_at_step(instance.n_points, result.merges, step)
        fig = build_scatter_figure(instance, labels)
        colors = [tr.marker.color for tr in fig.data if tr.name != "Noise"]
        assert len(colors) == len(set(colors)), f"color collision at step {step}"


@pytest.mark.parametrize("n_clusters", [1, 2, 8, 9, 15, 40])
def test_cluster_colors_are_pairwise_distinct_for_various_cluster_counts(n_clusters):
    from hb_visualization import _cluster_color

    colors = [_cluster_color(i, n_clusters) for i in range(n_clusters)]
    assert len(colors) == len(set(colors))
