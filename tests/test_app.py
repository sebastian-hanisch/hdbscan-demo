"""End-to-end Regressionstest fuer einen echten Nutzer-Bugreport (2026-09-05):
`streamlit.errors.StreamlitDuplicateElementId` beim zweiten Preset am vorletzten
Fusionsschritt. Ursache: mehrere `st.plotly_chart`-Aufrufe in app.py hatten kein
explizites `key=` und verliessen sich auf Streamlits inhaltsbasierte Auto-ID - am
vorletzten Schritt ist die ROHE Zwischenpartition (2 Cluster) bei diesem Preset
zufaellig BYTE-IDENTISCH mit dem finalen HDBSCAN-Ergebnis (ebenfalls 2 Cluster, 0
Noise), wodurch zwei verschiedene plotly_chart-Aufrufe dieselbe Auto-ID erzeugten.
Fix: explizite, eindeutige `key=`-Werte fuer jeden plotly_chart-Aufruf. Nutzt
Streamlits offizielles `AppTest`-Framework, um die App wirklich end-to-end
auszufuehren (reine Modul-Tests haetten diesen App.py-spezifischen Bug nicht
gefunden - er lag nicht in hb_algorithm.py oder hb_visualization.py selbst)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

# Exakt Preset 2 ("Der Fall, an dem DBSCAN scheiterte"), siehe hb_constants.PRESETS.
PRESET_2_SESSION_STATE = {
    "n_points_slider": 150,
    "k_slider": 2,
    "spread_slider": 0.15,
    "density_imbalance_slider": 0.9,
    "bridge_strength_slider": 0.0,
    "seed_input": 3,
    "min_cluster_size_slider": 5,
    "min_samples_slider": 5,
}


def _run_preset_2():
    at = AppTest.from_file(APP_PATH)
    for key, value in PRESET_2_SESSION_STATE.items():
        at.session_state[key] = value
    at.run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_preset_2_loads_without_exception():
    _run_preset_2()


def test_preset_2_second_to_last_raw_step_does_not_raise_duplicate_element_id():
    """Der urspruengliche Bugreport: Schritt-Regler auf max_step - 1 stellen."""
    at = _run_preset_2()
    step_slider = at.slider(key="hb_step")
    max_step = int(step_slider.max)

    at.slider(key="hb_step").set_value(max_step - 1).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]


def test_every_raw_step_of_preset_2_renders_without_exception():
    """Nicht nur der eine gemeldete Schritt - die GESAMTE Fusionsgeschichte durchlaufen,
    falls noch andere Schritte betroffen waeren."""
    at = _run_preset_2()
    max_step = int(at.slider(key="hb_step").max)

    for step in range(0, max_step + 1, 7):
        at.slider(key="hb_step").set_value(step).run(timeout=60)
        assert not at.exception, f"step {step}: {[str(e) for e in at.exception]}"
