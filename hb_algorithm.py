"""HDBSCAN from scratch: Kern-Distanz, Mutual-Reachability-Distanz, Single-Linkage-
Hierarchie darauf (wiederverwendete Lance-Williams-Technik aus agglomerative-demo,
verallgemeinert auf eine vorgegebene Distanzmatrix statt roher Koordinaten - mathematisch
aequivalent zu einem Minimum Spanning Tree ueber die Mutual-Reachability-Distanzen),
kondensierter Baum und "Excess of Mass"-Stabilitaetsauswahl (Campello, Moulavi & Sander,
2013).

Bewusst ohne sklearn/hdbscan zur Laufzeit implementiert. sklearn.cluster.HDBSCAN dient in
tests/ nur als unabhaengiger Kreuzvergleich fuer die eigene Implementierung.
"""

from dataclasses import dataclass

import numpy as np

NOISE = -1


def core_distances(data, min_samples):
    """Kern-Distanz je Punkt: Distanz zum (min_samples - 1)-ten naechsten ANDEREN Punkt -
    dieselbe Selbsteinschluss-Konvention wie dbscan-demos _region_query/k_distance_values."""
    n = len(data)
    k = max(1, min_samples - 1)
    core = np.empty(n)
    for i in range(n):
        dists = np.sort(np.sqrt(((data - data[i]) ** 2).sum(axis=1)))
        core[i] = dists[k] if k < len(dists) else dists[-1]
    return core


def mutual_reachability_matrix(data, min_samples):
    """d_mreach(a,b) = max(core(a), core(b), d(a,b)) - blaeht Abstaende durch duenn
    besiedelte Regionen kuenstlich auf und unterdrueckt dadurch Single-Linkage-Chaining."""
    n = len(data)
    diffs = data[:, None, :] - data[None, :, :]
    raw = np.sqrt((diffs ** 2).sum(axis=2))
    core = core_distances(data, min_samples)
    mreach = np.maximum(raw, np.maximum(core[:, None], core[None, :]))
    np.fill_diagonal(mreach, 0.0)
    return mreach


@dataclass(frozen=True)
class Merge:
    step: int
    cluster_a: int
    cluster_b: int
    distance: float
    new_cluster: int
    members: tuple


def single_linkage_from_matrix(dist_matrix):
    """Single-Linkage-Hierarchie ueber eine VORGEGEBENE Distanzmatrix (statt roher
    Koordinaten) - dieselbe Lance-Williams-Rekursion wie agglomerative-demos
    _lance_williams_coeffs("single", ...), hier auf die Mutual-Reachability-Distanz
    angewendet. Fuer Single-Linkage gilt alpha_i=alpha_j=0.5, beta=0, gamma=-0.5."""
    n = dist_matrix.shape[0]
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = float(dist_matrix[i, j])

    active = set(range(n))
    members = {i: (i,) for i in range(n)}
    next_id = n
    merges = []

    def key(i, j):
        return (i, j) if i < j else (j, i)

    for step in range(n - 1):
        a, b = min(dist, key=dist.get)
        d_ab = dist[(a, b)]
        others = active - {a, b}

        new_vals = {}
        for c in others:
            da, db = dist[key(a, c)], dist[key(b, c)]
            new_vals[c] = 0.5 * da + 0.5 * db - 0.5 * abs(da - db)  # = min(da, db)

        for k_ in [k for k in dist if a in k or b in k]:
            del dist[k_]
        new_id = next_id
        next_id += 1
        for c, v in new_vals.items():
            dist[key(new_id, c)] = v

        members[new_id] = members[a] + members[b]
        active.discard(a)
        active.discard(b)
        active.add(new_id)
        merges.append(Merge(step, a, b, d_ab, new_id, members[new_id]))

    return tuple(merges)


def labels_at_step(n_points, merges, step):
    """Partition nach genau `step + 1` angewendeten Single-Linkage-Fusionen (0-indiziert),
    als fortlaufend nummerierte Labels 0..k-1 - fuer die animierte Zwischenansicht (rohe
    Partition VOR der Kondensierung), analog zu agglomerative-demos labels_at_step."""
    label = list(range(n_points))
    for merge in merges[: step + 1]:
        for p in merge.members:
            label[p] = merge.new_cluster

    seen = {}
    result = []
    for l in label:
        if l not in seen:
            seen[l] = len(seen)
        result.append(seen[l])
    return tuple(result)


def _lam(distance):
    return 1.0 / distance if distance > 0 else float("inf")


@dataclass(frozen=True)
class CondensedNode:
    id: int
    parent: object  # int oder None
    birth_lambda: float
    children: tuple  # (id, id) oder () fuer einen Blattknoten des kondensierten Baums
    stability: float


@dataclass(frozen=True)
class HdbscanResult:
    merges: tuple  # Single-Linkage-Fusionen auf der Mutual-Reachability-Distanz
    condensed_nodes: dict  # id -> CondensedNode
    selected: frozenset  # IDs der ausgewaehlten (finalen) Cluster
    final_labels: tuple  # Cluster-ID (fortlaufend ab 0) oder NOISE (-1) je Punkt
    root_condensed_id: int


def _build_condensed_tree(n_points, merges, min_cluster_size):
    size_of = {i: 1 for i in range(n_points)}
    members_of = {i: (i,) for i in range(n_points)}
    children_of = {}
    distance_of = {}
    for m in merges:
        size_of[m.new_cluster] = len(m.members)
        members_of[m.new_cluster] = m.members
        children_of[m.new_cluster] = (m.cluster_a, m.cluster_b)
        distance_of[m.new_cluster] = m.distance

    condensed = {}  # id -> dict(parent, birth_lambda, children, stability_acc)
    point_departure = {}  # point -> (condensed_id, lambda)
    next_cid = [0]

    def new_node(parent, birth_lambda):
        cid = next_cid[0]
        next_cid[0] += 1
        condensed[cid] = {"parent": parent, "birth_lambda": birth_lambda, "children": (), "stability": 0.0}
        return cid

    def mark_departure(node_id, condensed_id, lam):
        for p in members_of[node_id]:
            point_departure[p] = (condensed_id, lam)

    def process(node_id, condensed_id, birth_lambda):
        if node_id not in children_of:
            return
        a, b = children_of[node_id]
        size_a, size_b = size_of[a], size_of[b]
        lam = _lam(distance_of[node_id])
        qualifies_a = size_a >= min_cluster_size
        qualifies_b = size_b >= min_cluster_size

        if qualifies_a and qualifies_b:
            # Echter Split: wie beim individuellen Herausfallen zaehlt auch dieser
            # Uebergang zur Stabilitaet des ALTEN Knotens (Referenz-Implementierung
            # hdbscan._hdbscan_tree.compute_stability summiert ueber JEDE Zeile des
            # kondensierten Baums, auch Split-Zeilen, nicht nur individuelle Abgaenge).
            remaining = size_of[node_id]
            condensed[condensed_id]["stability"] += (lam - birth_lambda) * remaining
            child_a = new_node(condensed_id, lam)
            child_b = new_node(condensed_id, lam)
            condensed[condensed_id]["children"] = (child_a, child_b)
            process(a, child_a, lam)
            process(b, child_b, lam)
        elif not qualifies_a and not qualifies_b:
            remaining = size_of[node_id]
            condensed[condensed_id]["stability"] += (lam - birth_lambda) * remaining
            mark_departure(a, condensed_id, lam)
            mark_departure(b, condensed_id, lam)
        elif not qualifies_a:
            condensed[condensed_id]["stability"] += (lam - birth_lambda) * size_a
            mark_departure(a, condensed_id, lam)
            process(b, condensed_id, birth_lambda)
        else:
            condensed[condensed_id]["stability"] += (lam - birth_lambda) * size_b
            mark_departure(b, condensed_id, lam)
            process(a, condensed_id, birth_lambda)

    root_id = n_points + len(merges) - 1
    root_condensed = new_node(parent=None, birth_lambda=0.0)
    process(root_id, root_condensed, 0.0)

    return condensed, point_departure, root_condensed


def _select_clusters(condensed, root_id):
    """Excess-of-Mass: ein Knoten wird ausgewaehlt, wenn seine eigene Stabilitaet die
    Summe der (rekursiv besten) Stabilitaet seiner Kinder erreicht oder uebertrifft.

    Die Referenzimplementierung (hdbscan._hdbscan_tree.get_clusters, Default
    allow_single_cluster=False) nimmt die WURZEL selbst NIE als Kandidaten in Betracht -
    "alles ein einziges Cluster" ist damit ausgeschlossen, die Entscheidung startet
    stattdessen direkt bei den Kindern der Wurzel. Ohne diesen Ausschluss gewinnt die
    Wurzel in der Praxis fast immer (ihre Stabilitaet summiert ueber die gesamte
    Baumtiefe), was jede echte Unterteilung unterdruecken wuerde."""
    propagated = {}

    def compute_propagated(cid):
        node = condensed[cid]
        if not node["children"]:
            propagated[cid] = node["stability"]
            return propagated[cid]
        c1, c2 = node["children"]
        children_sum = compute_propagated(c1) + compute_propagated(c2)
        propagated[cid] = max(node["stability"], children_sum)
        return propagated[cid]

    compute_propagated(root_id)

    selected = set()

    def decide(cid):
        node = condensed[cid]
        if not node["children"]:
            selected.add(cid)
            return
        c1, c2 = node["children"]
        if node["stability"] >= propagated[c1] + propagated[c2]:
            selected.add(cid)
        else:
            decide(c1)
            decide(c2)

    root_children = condensed[root_id]["children"]
    if root_children:
        decide(root_children[0])
        decide(root_children[1])
    # Falls die Wurzel nie in zwei qualifizierende Cluster splittet, bleibt `selected`
    # leer - konsistent mit allow_single_cluster=False: kein Cluster entsteht, alle
    # Punkte werden Noise (siehe _assign_labels).
    return frozenset(selected)


def _assign_labels(n_points, condensed, point_departure, selected):
    raw_labels = np.full(n_points, NOISE)
    for p, (cid, _lam) in point_departure.items():
        cur = cid
        while cur is not None:
            if cur in selected:
                raw_labels[p] = cur
                break
            cur = condensed[cur]["parent"]

    remap = {}
    labels = np.full(n_points, NOISE)
    for i, l in enumerate(raw_labels):
        if l == NOISE:
            continue
        if l not in remap:
            remap[l] = len(remap)
        labels[i] = remap[l]
    return tuple(int(l) for l in labels)


def run(data, min_cluster_size, min_samples):
    mreach = mutual_reachability_matrix(data, min_samples)
    merges = single_linkage_from_matrix(mreach)
    condensed_raw, point_departure, root_id = _build_condensed_tree(len(data), merges, min_cluster_size)
    selected = _select_clusters(condensed_raw, root_id)
    final_labels = _assign_labels(len(data), condensed_raw, point_departure, selected)

    condensed_nodes = {
        cid: CondensedNode(cid, node["parent"], node["birth_lambda"], node["children"], node["stability"])
        for cid, node in condensed_raw.items()
    }
    return HdbscanResult(
        merges=merges, condensed_nodes=condensed_nodes, selected=selected,
        final_labels=final_labels, root_condensed_id=root_id,
    )
