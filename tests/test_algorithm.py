import numpy as np
import pytest

from hb_algorithm import core_distances, mutual_reachability_matrix, run
from hb_scenario import generate_instance


def test_core_distance_hand_computed():
    data = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
    core = core_distances(data, min_samples=2)  # k=1: Distanz zum naechsten ANDEREN Punkt
    assert tuple(core) == pytest.approx((1.0, 1.0, 2.0, 3.0))


def test_mutual_reachability_is_max_of_core_and_raw():
    data = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    core = core_distances(data, min_samples=2)
    mreach = mutual_reachability_matrix(data, min_samples=2)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            raw = abs(data[i, 0] - data[j, 0])
            assert mreach[i, j] == pytest.approx(max(core[i], core[j], raw))


def test_hand_computed_two_clear_pairs_no_noise():
    """Zwei eng benachbarte Punktpaare, weit voneinander entfernt, min_cluster_size=2:
    von Hand nachgerechnet (siehe hb_algorithm-Kommentar in der Planungs-Session) -
    jedes Paar bildet sein eigenes stabiles Cluster, kein Rauschen, weil beide Ketten
    beim Wurzel-Split bereits die Mindestgroesse erreichen."""
    data = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]])
    result = run(data, min_cluster_size=2, min_samples=2)

    assert -1 not in result.final_labels
    assert result.final_labels[0] == result.final_labels[1]
    assert result.final_labels[2] == result.final_labels[3]
    assert result.final_labels[0] != result.final_labels[2]


def test_far_isolated_point_becomes_noise():
    """Ein einzelner, extrem weit entfernter Punkt haengt sich zwangslaeufig erst an
    der WURZEL des Baums an (er ist die letzte, groesste Fusion ueberhaupt) - da die
    Wurzel per Excess-of-Mass (allow_single_cluster=False-Konvention) nie selbst
    ausgewaehlt werden kann, wird ein Punkt, der DIREKT von der Wurzel abgeht, immer
    Noise, unabhaengig von Details wie exakten Distanz-Gleichstaenden anderswo im Baum."""
    rng = np.random.default_rng(1)
    blob0 = rng.normal(loc=[0, 0], scale=0.1, size=(15, 2))
    blob1 = rng.normal(loc=[3, 3], scale=0.1, size=(15, 2))
    outlier = np.array([[100.0, 100.0]])
    data = np.concatenate([blob0, blob1, outlier])

    result = run(data, min_cluster_size=5, min_samples=5)
    assert result.final_labels[-1] == -1
    assert -1 not in result.final_labels[:15]
    assert -1 not in result.final_labels[15:30]


def test_every_point_gets_a_valid_label_or_noise():
    instance = generate_instance(n_points=100, k=3, spread=0.2, density_imbalance=0.4, bridge_strength=0.3, seed=5)
    result = run(instance.as_array(), min_cluster_size=5, min_samples=5)
    assert len(result.final_labels) == instance.n_points
    non_noise = [l for l in result.final_labels if l != -1]
    if non_noise:
        assert set(non_noise) == set(range(max(non_noise) + 1))


def test_all_stabilities_are_non_negative():
    instance = generate_instance(n_points=90, k=2, spread=0.2, density_imbalance=0.0, bridge_strength=0.5, seed=7)
    result = run(instance.as_array(), min_cluster_size=5, min_samples=5)
    for node in result.condensed_nodes.values():
        assert node.stability >= -1e-9


def _agreement_fraction(labels_a, labels_b):
    """Anteil der Punktpaare, bei denen beide Label-Zuweisungen uebereinstimmen (beide
    Noise, beide im selben Cluster, oder beide in unterschiedlichen Clustern)."""
    n = len(labels_a)
    iu = np.triu_indices(n, k=1)
    same_a = (labels_a[:, None] == labels_a[None, :])[iu]
    same_b = (labels_b[:, None] == labels_b[None, :])[iu]
    return float(np.mean(same_a == same_b))


@pytest.mark.parametrize("seed", range(5))
def test_matches_sklearn_hdbscan_partition(seed):
    """Unabhaengiger Kreuzvergleich gegen sklearn.cluster.HDBSCAN. Erwartet hohe, aber
    NICHT zwingend exakte Uebereinstimmung: die Mutual-Reachability-Distanz erzeugt oft
    exakte Gleichstaende (mehrere Punktpaare mit identischer, vom selben core-distance-
    Wert dominierter Distanz) - bei einem Gleichstand entscheidet die Verarbeitungs-
    reihenfolge, welches Paar zuerst fusioniert wird, und diese Reihenfolge unterscheidet
    sich zwischen unserer einfachen Lance-Williams-Schleife und sklearns Boruvka-basierter
    Implementierung. Beide resultierenden Baeume sind gueltige, gleich gute Loesungen -
    kein Korrektheitsfehler, siehe [[project_hdbscan_demo_venv]]. Deshalb: hoher
    Uebereinstimmungs-Schwellwert statt exaktem Vergleich. sklearn ist ausschliesslich
    ein Test-Dependency, kein Laufzeit-Dependency der App."""
    sklearn_cluster = pytest.importorskip("sklearn.cluster")

    instance = generate_instance(
        n_points=80, k=2, spread=0.2, density_imbalance=0.5, bridge_strength=0.4, seed=seed
    )
    data = instance.as_array()
    ours = run(data, min_cluster_size=5, min_samples=5)

    sk_model = sklearn_cluster.HDBSCAN(min_cluster_size=5, min_samples=5)
    sk_model.fit(data)

    our_labels = np.array(ours.final_labels)
    sk_labels = sk_model.labels_

    agreement = _agreement_fraction(our_labels, sk_labels)
    assert agreement > 0.8, f"agreement too low seed={seed}: {agreement:.3f}"
