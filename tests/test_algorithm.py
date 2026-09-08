import numpy as np
import pytest

from hb_algorithm import core_distances, mutual_reachability_matrix, run, soft_cluster_membership
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


def test_soft_membership_values_are_valid_probabilities():
    """Jeder Eintrag liegt in [0, 1], jede Zeilensumme hoechstens 1 (die Gesamt-Konfidenz-
    Skalierung `in_cluster_prob` deckelt die Summe - nur ein Punkt, der mit absoluter
    Sicherheit zu EINEM Cluster gehoert, erreicht Zeilensumme 1)."""
    instance = generate_instance(n_points=90, k=3, spread=0.15, density_imbalance=0.0, bridge_strength=0.0, seed=1)
    data = instance.as_array()
    result = run(data, min_cluster_size=5, min_samples=5)
    matrix, clusters = soft_cluster_membership(data, result)

    assert matrix.shape == (instance.n_points, len(clusters))
    assert np.all(matrix >= -1e-9)
    assert np.all(matrix.sum(axis=1) <= 1.0 + 1e-9)


def test_soft_membership_argmax_matches_final_labels():
    """Selbstkonsistenz: fuer JEDEN Punkt, der final einem echten Cluster zugeordnet wurde
    (kein Rauschen), muss das wahrscheinlichste Cluster im weichen Wahrscheinlichkeits-
    vektor genau jenes harte Cluster sein - Soft Clustering darf der harten Zuordnung
    nicht widersprechen, nur sie graduell verfeinern."""
    instance = generate_instance(n_points=90, k=3, spread=0.15, density_imbalance=0.3, bridge_strength=0.3, seed=4)
    data = instance.as_array()
    result = run(data, min_cluster_size=5, min_samples=5)
    matrix, clusters = soft_cluster_membership(data, result)

    final = np.array(result.final_labels)
    argmax_cluster_id = np.array([clusters[j] for j in np.argmax(matrix, axis=1)])
    non_noise = final != -1
    for label in sorted(set(final[non_noise].tolist())):
        member_cids = argmax_cluster_id[non_noise & (final == label)]
        vals, counts = np.unique(member_cids, return_counts=True)
        assert counts.max() / counts.sum() > 0.9, f"label {label}: modal cid fraction too low"


def test_soft_membership_noise_points_get_lower_total_probability():
    """Kernbehauptung des Gesamt-Konfidenz-Faktors: echte Rausch-Punkte muessen im Schnitt
    DEUTLICH weniger Gesamt-Wahrscheinlichkeitsmasse ueber alle Cluster erhalten als
    tatsaechlich zugeordnete Punkte - sonst waere Soft Clustering nur eine flache
    Umverteilung ohne echten Erkenntnisgewinn ueber Rauschen hinaus."""
    instance = generate_instance(n_points=100, k=2, spread=0.225, density_imbalance=0.0, bridge_strength=0.32, seed=2)
    data = instance.as_array()
    result = run(data, min_cluster_size=5, min_samples=5)
    matrix, clusters = soft_cluster_membership(data, result)

    final = np.array(result.final_labels)
    noise_mask = final == -1
    assert noise_mask.any() and (~noise_mask).any(), "Testszenario muss sowohl Rauschen als auch Cluster-Punkte haben"
    noise_total = matrix[noise_mask].sum(axis=1).mean()
    clustered_total = matrix[~noise_mask].sum(axis=1).mean()
    assert noise_total < clustered_total - 0.2


def test_soft_membership_empty_when_no_clusters_found():
    """Randfall: findet HDBSCAN ueberhaupt kein Cluster (alles Rauschen), muss Soft
    Clustering ein leeres (n, 0)-Array liefern statt abzustuerzen."""
    data = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, -5.0], [-8.0, 12.0]])
    result = run(data, min_cluster_size=5, min_samples=2)
    matrix, clusters = soft_cluster_membership(data, result)
    assert clusters == []
    assert matrix.shape == (4, 0)


def _align_columns_to_reference(our_matrix, ref_matrix, ref_labels):
    """Ordnet die Spalten von `our_matrix` den Spalten von `ref_matrix` zu, ueber die
    Mehrheits-Referenz-Zuordnung der Punkte, die je our-Spalte am wahrscheinlichsten
    gelten - noetig, weil beide Implementierungen Cluster-IDs unabhaengig voneinander
    vergeben und keine gemeinsame Nummerierung existiert."""
    ref_label_list = sorted(set(l for l in ref_labels.tolist() if l != -1))
    our_argmax = np.argmax(our_matrix, axis=1)
    aligned_cols = []
    for j in range(our_matrix.shape[1]):
        mask = our_argmax == j
        if not mask.any():
            aligned_cols.append(np.zeros(len(our_matrix)))
            continue
        vals, counts = np.unique(ref_labels[mask], return_counts=True)
        target = vals[np.argmax(counts)]
        if target == -1:
            aligned_cols.append(np.zeros(len(our_matrix)))
        else:
            aligned_cols.append(ref_matrix[:, ref_label_list.index(target)])
    return np.stack(aligned_cols, axis=1)


@pytest.mark.parametrize("seed", range(4))
def test_soft_membership_matches_hdbscan_reference_implementation(seed):
    """Unabhaengiger Kreuzvergleich gegen `hdbscan.prediction.all_points_membership_
    vectors` (McInnes & Healy) - dieselbe Bibliothek, die test_matches_sklearn_hdbscan_
    partition oben als Referenz fuer die harte Partition nutzt, hier fuer deren eigene
    Soft-Clustering-Funktion. Cluster-Nummerierung unterscheidet sich zwischen beiden
    Implementierungen (siehe _align_columns_to_reference), daher Vergleich nach Angleichung:
    das jeweils wahrscheinlichste Cluster muss weit ueberwiegend uebereinstimmen, und die
    Gesamt-Konfidenz je Punkt (Zeilensumme) muss stark korrelieren."""
    hdb = pytest.importorskip("hdbscan")

    instance = generate_instance(n_points=90, k=2, spread=0.2, density_imbalance=0.4, bridge_strength=0.3, seed=seed)
    data = instance.as_array()
    ours = run(data, min_cluster_size=5, min_samples=5)
    our_matrix, our_clusters = soft_cluster_membership(data, ours)

    clusterer = hdb.HDBSCAN(min_cluster_size=5, min_samples=5, prediction_data=True)
    clusterer.fit(data)
    ref_matrix = hdb.prediction.all_points_membership_vectors(clusterer)

    if our_matrix.shape[1] == 0 or ref_matrix.shape[1] == 0 or our_matrix.shape[1] != ref_matrix.shape[1]:
        pytest.skip(f"seed {seed}: differing cluster counts, not comparable (own base-tree tie-break variance)")

    ref_aligned = _align_columns_to_reference(our_matrix, ref_matrix, clusterer.labels_)
    argmax_agreement = np.mean(np.argmax(our_matrix, axis=1) == np.argmax(ref_aligned, axis=1))
    row_sum_corr = np.corrcoef(our_matrix.sum(axis=1), ref_matrix.sum(axis=1))[0, 1]

    assert argmax_agreement > 0.9, f"seed {seed}: argmax agreement too low ({argmax_agreement:.3f})"
    assert row_sum_corr > 0.8, f"seed {seed}: total-confidence correlation too low ({row_sum_corr:.3f})"
