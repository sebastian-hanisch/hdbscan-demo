"""HDBSCAN für Sammel-Routen ohne eps UND ohne Chaining-Risiko - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im
Vergleich) zeigt diese Demo EIN Verfahren - HDBSCAN - als den Zusammenfluss der beiden
vorigen Stücke der Clustering-Linie: dbscan-demo (Dichte-Ungleichgewicht-Problem) und
agglomerative-demo (Single-Linkage-Chaining-Problem). Zwei der vier Presets recyceln
exakt jene Härtefälle und zeigen live, dass HDBSCAN beide löst. Fünftes Stück der
"Konzepte"-Reihe (siehe README für die Einordnung).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import hb_constants as C
from hb_algorithm import labels_at_step, run
from hb_evaluation import per_group_noise_fraction, rand_index
from hb_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from hb_scenario import generate_instance
from hb_visualization import (
    build_condensed_tree_figure,
    build_dendrogram_figure,
    build_mini_scatter_figure,
    build_noise_fraction_bar_chart,
    build_scatter_figure,
)

st.set_page_config(page_title="HDBSCAN – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _compute_run(n_points, k, spread, density_imbalance, bridge_strength, shape, seed, min_cluster_size, min_samples):
    instance = generate_instance(n_points, k, spread, density_imbalance, bridge_strength, seed, shape=shape)
    result = run(instance.as_array(), min_cluster_size, min_samples)
    return instance, result


@st.cache_data(show_spinner=False)
def _compute_mini_run(instance, min_cluster_size, min_samples):
    return run(instance.as_array(), min_cluster_size, min_samples)


st.title("🧩 HDBSCAN: die Kombination, die Dichte UND Chaining löst")
st.markdown(
    """
Lieferadressen sollen zu Sammel-Routen gruppiert werden, **ohne eps festlegen zu müssen
UND ohne Chaining-Risiko** - beide Probleme, die die vorigen zwei Stücke dieser Reihe
aufgeworfen haben. **HDBSCAN** kombiniert genau die zwei Zutaten, die es im Namen trägt:
eine **h**ierarchische Fusionsstruktur (wie in agglomerative-demo), aber angewendet auf
eine **dichte-angepasste Distanz** (den DBSCAN-Kerngedanken aus dbscan-demo) statt der
rohen euklidischen Distanz. Genau **wie** das funktioniert, erklärt der aufgeklappte
Abschnitt direkt darunter - bevor weiter unten live geprüft wird, ob HDBSCAN tatsächlich
beide geerbten Härtefälle löst.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren "
    "vergleichen, zeigt diese Demo - Teil der wachsenden \"Konzepte\"-Reihe - **ein** "
    "Verfahren: HDBSCAN selbst, nicht eine vierte Methode neben k-Means/DBSCAN/"
    "agglomerativem Clustering, sondern der Punkt, auf den die beiden letzten Stücke "
    "dieser Reihe unabhängig voneinander zulaufen."
)

with st.expander("So funktioniert HDBSCAN", expanded=True):
    st.markdown(
        """
HDBSCAN durchläuft vier Schritte:

1. **Kern-Distanz** je Punkt (wie bei DBSCAN): wie weit muss man suchen, bis man
   `min_samples` Punkte gefunden hat?
2. **Mutual-Reachability-Distanz**: der größere Wert aus der Kern-Distanz beider Punkte
   und ihrer tatsächlichen Distanz - das bläht Abstände durch dünn besiedelte Regionen
   künstlich auf und unterdrückt dadurch Single-Linkage-Chaining, bevor es überhaupt
   entstehen kann.
3. **Single-Linkage-Hierarchie** auf dieser neuen Distanz (wie bei agglomerativem
   Clustering) - mathematisch äquivalent zu einem Minimum Spanning Tree über die
   Mutual-Reachability-Distanzen.
4. **Kondensierter Baum**: statt eines einzelnen Schnitts (wie bei k-Means' Ziel-k) wird
   der gesamte Baum durchlaufen - Äste, die nie `min_cluster_size` erreichen, werden
   Punkt für Punkt zu Noise, stabile Äste werden zu Clustern. Welcher Ast am Ende zählt,
   entscheidet die **Stabilität** (wie lange ein Cluster "überlebt", bevor er sich
   auflöst oder aufspaltet) - kein Regler mehr nötig, der wie eps global für alle
   Dichten gleichzeitig passen müsste.

Die Grafiken weiter unten zeigen zuerst die rohe Single-Linkage-Zwischenpartition (Schritt
für Schritt, wie in agglomerative-demo) - und direkt danach den kondensierten Baum und das
finale HDBSCAN-Ergebnis, damit der Unterschied sichtbar wird, den die Kondensierung macht.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = {
    "Einfaches Beispiel": "Gleichmäßige Dichte, keine Brücke - eine Baseline, bei der ohnehin alles funktioniert.",
    "Der Fall, an dem DBSCAN scheiterte": "Starkes Dichte-Ungleichgewicht - HDBSCAN braucht kein eps und bedient beide Dichten gut.",
    "Der Fall, an dem Single-Linkage scheiterte": "Eine dünne Punktbrücke - Mutual-Reachability-Distanz verhindert das Chaining, das rohes Single-Linkage zum Scheitern brachte.",
    "Kombinierter Härtefall": "Dichte-Ungleichgewicht UND Brücke gleichzeitig - der eigentliche Beweis, dass die Kombination mehr kann als jede Zutat allein.",
    "Nicht-konvexe Formen (auch das meistert HDBSCAN)": "Zwei ineinander verschlungene Halbmonde, ganz ohne eps - HDBSCAN braucht dafür etwas größere min_cluster_size/min_samples-Werte als bei runden Gruppen, meistert die Form aber genauso wie DBSCAN.",
}
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_points = st.slider("Anzahl Adressen", *bounds("n_points_slider"), key="n_points_slider")
    k = st.slider("Anzahl wahrer Gruppen", *bounds("k_slider"), key="k_slider")
    spread = st.slider(
        "Streuung", *bounds("spread_slider"), key="spread_slider", step=0.05,
        help="Klein = Gruppen klar getrennt. Groß = Gruppen überlappen sich spürbar.",
    )
    density_imbalance = st.slider(
        "Dichte-Ungleichgewicht", *bounds("density_imbalance_slider"), key="density_imbalance_slider", step=0.05,
        help="0 = alle Gruppen gleich dicht. 1 = eine Gruppe wird deutlich lockerer/diffuser "
        "als die übrigen, bei gleicher Punktzahl - der dbscan-demo-Härtefall.",
    )
    bridge_strength = st.slider(
        "Brücken-Stärke (zwischen den ersten beiden Gruppen)", *bounds("bridge_strength_slider"),
        key="bridge_strength_slider", step=0.02,
        help="0 = keine Brücke. Höher = mehr verbindende Punkte zwischen Gruppe 1 und 2 - "
        "der agglomerative-demo-Härtefall.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Punktwolken-Form**")
    shape = st.radio(
        "Form", options=C.SHAPES, key="shape_radio", format_func=lambda s: C.SHAPE_LABELS[s],
        help="„Gruppen“: runde, konvexe Cluster. „Halbmonde“: nicht-konvexe Bögen - Dichte-"
        "Ungleichgewicht und Brücken-Stärke wirken auf beide Formen.",
    )

    st.markdown("**HDBSCAN-Parameter**")
    min_cluster_size = st.slider(
        "min_cluster_size", *bounds("min_cluster_size_slider"), key="min_cluster_size_slider",
        help="Ersetzt eps komplett: die kleinste Punktzahl, die ein Ast im Fusionsbaum "
        "erreichen muss, um als eigenes Cluster zu gelten.",
    )
    min_samples = st.slider(
        "min_samples", *bounds("min_samples_slider"), key="min_samples_slider",
        help="Wie bei DBSCAN: steuert die Kern-Distanz (wie viele Punkte inklusive des "
        "Punktes selbst mindestens 'nah genug' sein müssen).",
    )

    st.button(
        "🎲 Neue Punktwolke generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für die Adressen.",
    )

sync_query_params(
    n_points, k, spread, density_imbalance, bridge_strength, seed, shape, min_cluster_size, min_samples
)

with st.spinner("Führe HDBSCAN aus..."):
    instance, result = _compute_run(
        int(n_points), int(k), spread, density_imbalance, bridge_strength, shape, int(seed),
        int(min_cluster_size), int(min_samples),
    )

max_step = len(result.merges) - 1
run_key = (n_points, k, spread, density_imbalance, bridge_strength, shape, seed, min_cluster_size, min_samples)
if "hb_step" not in st.session_state or st.session_state.get("hb_step_owner") != run_key:
    st.session_state["hb_step"] = max_step
    st.session_state["hb_step_owner"] = run_key

st.markdown("## 🎯 Die rohe Single-Linkage-Zwischenpartition")
st.caption(
    "Dieselbe Fusionslogik wie in agglomerative-demo, aber auf der Mutual-Reachability-"
    "Distanz statt der rohen Distanz berechnet - noch OHNE Kondensierung. Ein Schritt = "
    "eine Fusion."
)

if max_step == 0:
    step = 0
    st.caption("Nur eine einzige Fusion möglich - kein Regler nötig.")
else:
    step = st.slider(
        "Schritt (Fusion)", 0, max_step, key="hb_step",
        help="Reglerposition steht standardmäßig auf der letzten Fusion (ein einziges "
        "Gesamtcluster) - frei verschiebbar, um die Fusionsgeschichte zu erkunden.",
    )

raw_labels = labels_at_step(instance.n_points, result.merges, step)
dendro_col, scatter_col = st.columns(2)
with dendro_col:
    st.plotly_chart(
        build_dendrogram_figure(instance.n_points, result.merges, step), width="stretch",
        key=f"dendrogram_{step}",
    )
with scatter_col:
    st.plotly_chart(build_scatter_figure(instance, raw_labels), width="stretch", key=f"raw_scatter_{step}")

st.markdown("---")

st.markdown("## 🌲 Kondensierter Baum und finales HDBSCAN-Ergebnis")
st.caption(
    "Derselbe Baum wie oben, aber jetzt kondensiert: Äste, die nie min_cluster_size "
    "erreichen, lösen sich Punkt für Punkt in Noise auf. Grün hervorgehoben sind die "
    "letztlich ausgewählten (stabilsten) Cluster."
)

tree_col, final_col = st.columns(2)
with tree_col:
    st.plotly_chart(
        build_condensed_tree_figure(result.condensed_nodes, result.selected), width="stretch",
        key="condensed_tree",
    )
with final_col:
    st.plotly_chart(build_scatter_figure(instance, result.final_labels), width="stretch", key="final_scatter")

n_clusters = len(set(l for l in result.final_labels if l != -1))
n_noise = sum(1 for l in result.final_labels if l == -1)
lm1, lm2, lm3 = st.columns(3)
lm1.metric("Cluster gefunden", n_clusters)
lm2.metric("Noise-Punkte", f"{n_noise}/{instance.n_points}")
lm3.metric(
    "Fusionen bis Endergebnis", f"{max_step + 1}",
    help="Gesamtzahl der Single-Linkage-Fusionen - unabhängig vom aktuellen Schritt-Regler oben.",
)

st.markdown("**Und mit anderen min_cluster_size-Werten?**")
st.caption(
    "Gleiche Adressen, gleiches min_samples wie oben - nur min_cluster_size unterscheidet "
    "sich. Deutlich sanftere Übergänge als DBSCANs eps-Empfindlichkeit."
)
example_values = sorted({
    max(C.MIN_CLUSTER_SIZE_MIN, min(C.MIN_CLUSTER_SIZE_MAX, int(round(min_cluster_size * factor))))
    for factor in (0.5, 0.75, 1.5, 2.0)
} - {int(min_cluster_size)})
example_cols = st.columns(len(example_values)) if example_values else []
for col, example_mcs in zip(example_cols, example_values):
    with col:
        example_result = _compute_mini_run(instance, example_mcs, int(min_samples))
        st.plotly_chart(
            build_mini_scatter_figure(instance, example_result.final_labels), width="stretch",
            key=f"mini_{example_mcs}",
        )
        ex_clusters = len(set(l for l in example_result.final_labels if l != -1))
        ex_noise = sum(1 for l in example_result.final_labels if l == -1)
        st.caption(f"min_cluster_size={example_mcs} · {ex_clusters} Cluster · {ex_noise} Noise")

st.markdown("---")

st.subheader("📐 Meistert HDBSCAN beide Härtefälle gleichzeitig?")
st.markdown(
    """
Live für Ihr aktuelles Szenario berechnet: der **Noise-Anteil je wahrer Gruppe**
(dbscan-demo-Maßstab - ein einzelnes eps konnte diffuse Gruppen dort nicht gleich gut
bedienen wie dichte) und der **Rand-Index** gegen die tatsächliche Gruppenzugehörigkeit
(agglomerative-demo-Maßstab - Single-Linkage chainte dort durch die Brücke).
"""
)

group_fractions = per_group_noise_fraction(instance.true_labels, result.final_labels)
ri = rand_index(instance.true_labels, result.final_labels)

bar_col, metric_col = st.columns([3, 2])
with bar_col:
    st.plotly_chart(build_noise_fraction_bar_chart(group_fractions), width="stretch", key="noise_fraction_bar")
with metric_col:
    st.metric(
        "Rand-Index gegen wahre Gruppen", f"{ri:.2f}",
        help="1.0 = perfekte Übereinstimmung, ~0.5 = kaum besser als Zufall - genau die "
        "Kennzahl, an der rohes Single-Linkage beim Brücken-Härtefall scheiterte.",
    )
    max_noise_gap = (max(group_fractions.values()) - min(group_fractions.values())) if group_fractions else 0.0
    st.metric(
        "Größte Noise-Differenz zwischen Gruppen", f"{max_noise_gap * 100:.0f} Prozentpunkte",
        help="Groß bedeutet: mindestens eine Gruppe wird deutlich schlechter bedient als "
        "eine andere - genau das Symptom, an dem ein einzelnes globales eps bei DBSCAN scheiterte.",
    )

if ri > 0.9 and max_noise_gap < 0.15:
    st.success(
        f"✅ Rand-Index {ri:.2f} und eine Noise-Differenz von nur "
        f"{max_noise_gap * 100:.0f} Prozentpunkten zwischen den Gruppen - HDBSCAN bedient "
        f"dieses Szenario gut, ganz ohne eps festzulegen und ohne durch eine Brücke zu chainen."
    )
else:
    st.info(
        "Bei diesem Szenario ist noch Luft nach oben - probieren Sie eines der Härtefall-"
        "Presets oder erhöhen Sie Dichte-Ungleichgewicht bzw. Brücken-Stärke, um zu sehen, "
        "wo auch HDBSCANs Grenzen liegen."
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Kern-Distanz** $\text{core}_k(p)$: Distanz von $p$ zu seinem $(\text{min\_samples}-1)$-ten
nächsten ANDEREN Punkt (Selbsteinschluss-Konvention wie bei DBSCAN).

**Mutual-Reachability-Distanz**:

$$
d_{\text{mreach}}(a,b) = \max\big(\text{core}_k(a),\ \text{core}_k(b),\ \lVert a-b \rVert\big)
$$

**Single-Linkage-Hierarchie** auf $d_{\text{mreach}}$: iteratives Verschmelzen des jeweils
nächstgelegenen Clusterpaars - mathematisch äquivalent zu einem **Minimum Spanning Tree**
über die Mutual-Reachability-Distanzen (in produktiven Implementierungen tatsächlich so
berechnet, hier aus Wiederverwendungsgründen über dieselbe Lance-Williams-Rekursion wie in
agglomerative-demo, nur mit dieser Distanzmatrix statt roher Koordinaten).

**Kondensierter Baum**: rekursiver Abstieg von der Wurzel; an jeder Fusion mit Distanz $d$
(also $\lambda = 1/d$) wird geprüft, ob beide Teilmengen mindestens `min_cluster_size`
Punkte haben. Wenn ja: echter Split, zwei neue Cluster geboren bei $\lambda$. Wenn nein:
die kleinere Teilmenge fällt Punkt für Punkt als Noise heraus, die größere führt das
bisherige Cluster fort.

**Stabilität** eines Clusters $C$, geboren bei $\lambda_{\text{birth}}$:

$$
S(C) = \sum_{p \in C} \big(\lambda_{\max}(p, C) - \lambda_{\text{birth}}(C)\big)
$$

wobei $\lambda_{\max}(p, C)$ der Lambda-Wert ist, bei dem $p$ entweder einzeln aus $C$
herausfällt oder $C$ selbst sich aufspaltet.

**Cluster-Auswahl ("Excess of Mass"):** bottom-up durch den kondensierten Baum - ein
Knoten wird ausgewählt, wenn $S(C) \ge \sum_{\text{Kinder}} S(\text{Kind})$, sonst gewinnen
die (rekursiv besseren) Kinder. Die Wurzel selbst steht dabei **nie** zur Wahl - "alles ein
einziges Cluster" ist damit ausgeschlossen.

Naive Laufzeit $O(n^3)$ für die Hierarchie (wie in agglomerative-demo) - produktive
Implementierungen nutzen eine Boruvka-MST-Variante mit $O(n \log n)$.

Implementiert in `hb_algorithm.py` (vollständige Pipeline) und `hb_evaluation.py`
(Rand-Index, Noise-Anteil je Gruppe).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
