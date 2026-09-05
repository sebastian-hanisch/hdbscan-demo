# HDBSCAN: die Kombination, die Dichte UND Chaining löst – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-hdbscan-demo.streamlit.app/)**

Fünftes Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning" - der **Zusammenfluss** der Clustering-Linie:

```
kmeans-demo → dbscan-demo ──┐
                             ├──> hdbscan-demo (dieses Stück)
agglomerative-demo ──────────┘
```

HDBSCAN kombiniert genau die zwei Zutaten, die es im Namen trägt: eine **h**ierarchische
Fusionsstruktur (wie in [agglomerative-demo](../agglomerative-demo)), angewendet auf eine
**dichte-angepasste Distanz** (den DBSCAN-Kerngedanken aus [dbscan-demo](../dbscan-demo))
statt der rohen euklidischen Distanz. Zwei der vier Presets recyceln exakt die
Härtefall-Szenarien, an denen die beiden Vorläufer nachweislich scheiterten, und zeigen
live, dass HDBSCAN beide löst - der Zusammenfluss wird nicht nur behauptet, sondern
konkret nachvollziehbar gemacht.

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im
Vergleich) zeigt diese Demo **ein** Verfahren – HDBSCAN selbst, nicht eine vierte
Methode neben k-Means/DBSCAN/agglomerativem Clustering. Vehikel-Problem: Lieferadressen
zu Sammel-Routen gruppieren, **ohne eps festlegen zu müssen UND ohne Chaining-Risiko**.

## Warum diese Demo anders aufgebaut ist

- **Einfaches Beispiel**: gleichmäßige Dichte, keine Brücke - Baseline.
- **Der Fall, an dem DBSCAN scheiterte**: starkes Dichte-Ungleichgewicht (Parameter nah
  an dbscan-demos "Schwerer Fall") - HDBSCAN braucht kein eps und erreicht 0% Noise in
  beiden Gruppen.
- **Der Fall, an dem Single-Linkage scheiterte**: eine dünne Punktbrücke, sparsam genug
  angelegt, dass ihre Punkte einzeln hohe Kern-Distanzen haben (Parameter nah an
  agglomerative-demos "Schwerer Fall", aber mit angepasster Brücken-Dichte - siehe
  Verifikation) - rohes Single-Linkage erreicht dort nur einen Rand-Index von ~0.5
  (Zufallsniveau), HDBSCAN ~0.96.
- **Kombinierter Härtefall**: beides gleichzeitig - Rand-Index und Noise-Anteil bleiben
  auch hier gut, der eigentliche Beweis, dass die Kombination mehr kann als jede Zutat
  allein.

## Visualisierung

Zwei Blöcke nacheinander machen den Unterschied sichtbar, den die Kondensierung macht:
zuerst die **rohe Single-Linkage-Zwischenpartition** auf der Mutual-Reachability-Distanz
(schrittanimiert wie in agglomerative-demo, aber noch ohne Kondensierung - hier würde
ohne den nächsten Schritt weiterhin gechaint werden können), dann der **kondensierte
Baum** und das **finale HDBSCAN-Ergebnis** direkt daneben. Die "Und mit anderen
min_cluster_size-Werten?"-Kleinmultiples zeigen, wie viel sanfter dieser verbleibende
Regler auf Änderungen reagiert als DBSCANs eps.

## Sicherheitsgrenzen

Keine eigene Iterationsgrenze nötig - die zugrunde liegende Single-Linkage-Hierarchie
terminiert immer nach $n-1$ Fusionen. `N_POINTS_MAX` (200) hält die naive $O(n^3)$-Suche
schnell genug für eine flüssige Animation.

## Verifikation

Diese Demo war die aufwendigste Debugging-Session der ganzen Reihe - zwei echte
Algorithmus-Bugs wurden erst durch den Kreuzvergleich gegen eine Referenzimplementierung
gefunden:

- **Kreuzvergleich gegen sklearn.cluster.HDBSCAN**: hohe (nicht zwingend exakte)
  Partitions-Übereinstimmung über mehrere Szenarien - "nicht zwingend exakt", weil die
  Mutual-Reachability-Distanz oft echte Gleichstände erzeugt (mehrere Punktpaare mit
  identischer, von derselben Kern-Distanz dominierter Distanz), bei denen die
  Verarbeitungsreihenfolge entscheidet, welches Paar zuerst fusioniert - unsere einfache
  Lance-Williams-Schleife und sklearns Boruvka-Implementierung lösen solche Gleichstände
  unterschiedlich auf, beide Bäume sind aber gleich gültig (kein Korrektheitsfehler).
- **Handgerechnete Kern-Distanz und Mutual-Reachability-Distanz** an kleinen Beispielen.
- **Handgerechnetes Kondensierungs-Beispiel**: zwei klar getrennte Punktpaare, beide
  erreichen beim Wurzel-Split direkt die Mindestgröße - kein Noise.
- **Die beiden zentralen Nachweise direkt getestet**: HDBSCAN erreicht <10% Noise je
  Gruppe auf dem geerbten DBSCAN-Härtefall, und einen Rand-Index >0.9 auf dem geerbten
  Single-Linkage-Härtefall.

Zwei bemerkenswerte Bugs unterwegs (Details in [[project_hdbscan_demo_venv]]): die
Stabilitätsformel zählte zunächst eine Fusion doppelt, und die Cluster-Auswahl ließ
ursprünglich die Wurzel selbst als Kandidaten zu (wodurch fast immer "alles ein einziges
Cluster" gewann) - die Referenzimplementierung schließt die Wurzel bewusst aus
(`allow_single_cluster=False`). Außerdem musste die Brücken-Geometrie aus
agglomerative-demo neu kalibriert werden: eine *dichte* Brücke (viele, eng beieinander
liegende Punkte) täuscht zwar Single-Linkage, hat aber selbst niedrige Kern-Distanzen und
wird von der Mutual-Reachability-Distanz kaum aufgebläht - eine *sparsame* Brücke
(weniger, aber pro Punkt "einsamere" Punkte) ist der ehrliche Testfall für HDBSCANs
eigentliches Versprechen.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, rohe Zwischenpartition, kondensierter Baum, finales Ergebnis, Kleinmultiples, Härtefall-Nachweis, Formulierungs-Expander |
| `hb_constants.py` | Defaults, Regler-Grenzen, Sicherheitsgrenzen, `PRESETS` |
| `hb_presets.py` | `SettingSpec`/`SETTING_SPECS`, Permalink-Logik, Presets, Zufalls-Seed-Button |
| `hb_scenario.py` | Kombinierter Generator: Gauß-Gruppen mit Dichte-Ungleichgewicht UND optionaler Punktbrücke, unabhängig einstellbar |
| `hb_algorithm.py` | Kern-Distanz, Mutual-Reachability-Distanz, Single-Linkage auf dieser Distanz, kondensierter Baum, Stabilität, Excess-of-Mass-Auswahl, finale Labels |
| `hb_evaluation.py` | Rand-Index, Noise-Anteil je wahrer Gruppe |
| `hb_visualization.py` | Dendrogramm-, Scatter-, kondensierter-Baum- und Noise-Balkendiagramm (Plotly) |
| `tests/` | Handinstanzen, sklearn-Kreuzvergleich, Struktur-Invarianten, die beiden zentralen Härtefall-Nachweise |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
