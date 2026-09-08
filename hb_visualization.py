"""Plotly-Visualisierungen: Single-Linkage-Dendrogramm auf Mutual-Reachability-Distanz
(schrittanimierbar wie in agglomerative-demo), Scatter fuer die rohe Zwischen-Partition
und das finale HDBSCAN-Ergebnis, der kondensierte Baum als eigene Grafik, sowie
Kleinmultiples und ein Noise-Anteil-Balkendiagramm."""

import numpy as np

from hb_algorithm import NOISE, labels_at_step

CLUSTER_PALETTE = [
    "#1f77b4", "#d68a2e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]
NOISE_COLOR = "#9aa6ba"
MAX_LEGEND_CLUSTERS = 12


def _axis_range(data):
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    padx = (xmax - xmin) * 0.1 or 1.0
    pady = (ymax - ymin) * 0.1 or 1.0
    return [xmin - padx, xmax + padx], [ymin - pady, ymax + pady]


def _cluster_color(index, n_clusters):
    """Bei bis zu len(CLUSTER_PALETTE) Clustern exakt die feste, qualitative Palette (wie
    in den uebrigen Demos). Die rohe Single-Linkage-Animation durchlaeuft aber auch sehr
    frueh Schritte mit weit mehr als 8 gleichzeitig existierenden (noch kaum fusionierten)
    Clustern - ein Modulo auf die 8-Farben-Palette wuerde dort GENUINELY verschiedene
    Cluster optisch ununterscheidbar machen. Ab mehr als 8 Clustern deshalb ein Farbrad
    (HSL, gleichmaessig ueber alle aktuell existierenden Cluster verteilt), das fuer jede
    Clusteranzahl paarweise unterschiedliche Farben garantiert."""
    if n_clusters <= len(CLUSTER_PALETTE):
        return CLUSTER_PALETTE[index % len(CLUSTER_PALETTE)]
    hue = (index * 360.0 / n_clusters) % 360
    return f"hsl({hue:.1f}, 65%, 50%)"


def _cluster_traces(data, labels, legend, confidence=None):
    """confidence (optional, array in [0, 1] je Punkt): mappt sich auf Marker-Opacity der
    Cluster-Punkte (NICHT der Noise-Punkte, die bleiben immer gleich subtil) - visualisiert
    weiche Zuordnungs-Konfidenz (Soft Clustering) ueber die bestehende harte Faerbung
    hinweg, ohne eine zweite Grafik zu brauchen. Auf [0.25, 1.0] gestaucht, damit selbst
    sehr unsichere Punkte noch sichtbar bleiben statt fast zu verschwinden."""
    import plotly.graph_objects as go

    traces = []
    cluster_ids = sorted(l for l in set(labels.tolist()) if l != NOISE)
    n_clusters = len(cluster_ids)
    show_legend = legend and n_clusters <= MAX_LEGEND_CLUSTERS

    mask_noise = labels == NOISE
    if mask_noise.any():
        traces.append(
            go.Scatter(
                x=data[mask_noise, 0], y=data[mask_noise, 1], mode="markers", name="Noise",
                showlegend=show_legend,
                marker=dict(symbol="x", color=NOISE_COLOR, size=7),
                hoverinfo="skip",
            )
        )
    for index, cid in enumerate(cluster_ids):
        mask = labels == cid
        color = _cluster_color(index, n_clusters)
        marker = dict(color=color, size=7, line=dict(width=0.5, color="white"))
        if confidence is not None:
            marker["opacity"] = (0.25 + 0.75 * confidence[mask]).tolist()
        traces.append(
            go.Scatter(
                x=data[mask, 0], y=data[mask, 1], mode="markers", name=f"Cluster {cid + 1}",
                showlegend=show_legend, marker=marker, hoverinfo="skip",
            )
        )
    return traces


def build_scatter_figure(instance, labels, legend=True, height=460, confidence=None):
    import plotly.graph_objects as go

    data = np.array(instance.points)
    fig = go.Figure()
    for trace in _cluster_traces(data, np.array(labels), legend, confidence=confidence):
        fig.add_trace(trace)

    xr, yr = _axis_range(data)
    layout_kwargs = dict(
        template="plotly_white", height=height,
        xaxis=dict(visible=False, range=xr, fixedrange=True),
        yaxis=dict(visible=False, range=yr, fixedrange=True, scaleanchor="x", scaleratio=1),
        showlegend=legend,
        margin=dict(t=40 if legend else 5, l=10 if legend else 5, r=10 if legend else 5, b=10 if legend else 5),
    )
    if legend:
        layout_kwargs["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(**layout_kwargs)
    return fig


def build_mini_scatter_figure(instance, labels):
    return build_scatter_figure(instance, labels, legend=False, height=200)


def _compute_dendrogram_layout(merges, n_points):
    children = {m.new_cluster: (m.cluster_a, m.cluster_b) for m in merges}
    x_of = {}
    next_leaf_slot = [0]

    def assign(node_id):
        if node_id not in children:
            x_of[node_id] = next_leaf_slot[0]
            next_leaf_slot[0] += 1
            return x_of[node_id]
        a, b = children[node_id]
        xa, xb = assign(a), assign(b)
        x_of[node_id] = (xa + xb) / 2
        return x_of[node_id]

    root = n_points + len(merges) - 1
    assign(root)
    return x_of


def build_dendrogram_figure(n_points, merges, step):
    import plotly.graph_objects as go

    x_of = _compute_dendrogram_layout(merges, n_points)
    height_of = {i: 0.0 for i in range(n_points)}

    edge_x, edge_y = [], []
    for merge in merges[: step + 1]:
        ya = height_of.get(merge.cluster_a, 0.0)
        yb = height_of.get(merge.cluster_b, 0.0)
        xa, xb = x_of[merge.cluster_a], x_of[merge.cluster_b]
        y = merge.distance
        edge_x += [xa, xa, None, xb, xb, None, xa, xb, None]
        edge_y += [ya, y, None, yb, y, None, y, y, None]
        height_of[merge.new_cluster] = y

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#1f77b4", width=1.5), hoverinfo="skip"))
    fig.update_layout(
        template="plotly_white", height=320,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(title="Mutual-Reachability-Distanz", fixedrange=True, rangemode="tozero"),
        margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig


def build_condensed_tree_figure(condensed_nodes, selected):
    """Vereinfachte, aber erkennbare Darstellung des kondensierten Baums: je Knoten ein
    horizontales Segment von seiner Geburts- bis zu seiner (durch die Kinder oder das
    Ende seiner Mitgliederliste bestimmten) Sterbe-Lambda, x-Position nach Blattreihenfolge
    wie beim Dendrogramm. Ausgewaehlte Cluster sind hervorgehoben."""
    import plotly.graph_objects as go

    x_of = {}
    next_slot = [0]

    def assign(cid):
        node = condensed_nodes[cid]
        if not node.children:
            x_of[cid] = next_slot[0]
            next_slot[0] += 1
            return x_of[cid]
        xs = [assign(c) for c in node.children]
        x_of[cid] = sum(xs) / len(xs)
        return x_of[cid]

    root_id = min(condensed_nodes.keys())
    assign(root_id)

    death_of = {}
    for cid, node in condensed_nodes.items():
        if node.children:
            death_of[cid] = condensed_nodes[node.children[0]].birth_lambda
    max_lambda = max([n.birth_lambda for n in condensed_nodes.values()] + [1.0])
    for cid, node in condensed_nodes.items():
        if cid not in death_of:
            death_of[cid] = max_lambda * 1.05

    fig = go.Figure()
    for cid, node in condensed_nodes.items():
        color = "#2ca02c" if cid in selected else "#9aa6ba"
        width = 4 if cid in selected else 2
        fig.add_trace(
            go.Scatter(
                x=[node.birth_lambda, death_of[cid]], y=[x_of[cid], x_of[cid]], mode="lines",
                line=dict(color=color, width=width), showlegend=False, hoverinfo="skip",
            )
        )
        if node.parent is not None:
            fig.add_trace(
                go.Scatter(
                    x=[node.birth_lambda, node.birth_lambda], y=[x_of[cid], x_of[node.parent]], mode="lines",
                    line=dict(color="#c4cbd8", width=1, dash="dot"), showlegend=False, hoverinfo="skip",
                )
            )

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#2ca02c", width=4), name="Ausgewählt"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#9aa6ba", width=2), name="Verworfen"))

    fig.update_layout(
        template="plotly_white", height=320,
        xaxis=dict(title="λ (= 1 / Distanz)", fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=30, l=10, r=10, b=10),
    )
    return fig


def build_noise_fraction_bar_chart(fractions):
    import plotly.graph_objects as go

    groups = sorted(fractions.keys())
    fig = go.Figure(
        go.Bar(
            x=[f"Gruppe {g + 1}" for g in groups], y=[fractions[g] * 100 for g in groups],
            marker_color=[CLUSTER_PALETTE[g % len(CLUSTER_PALETTE)] for g in groups],
        )
    )
    fig.update_layout(
        template="plotly_white", height=280,
        yaxis=dict(title="Noise-Anteil (%)", range=[0, 100], fixedrange=True),
        xaxis=dict(fixedrange=True), margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig
