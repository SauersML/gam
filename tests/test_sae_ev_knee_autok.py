"""#977/#1026 measure→improve, action 2: EV-knee auto-K + the manifold-vs-linear
wager verdict, computed from a real EV-vs-K reconstruction-parity frontier.

Two arms:
  * SYNTHETIC (live now) — the knee detector stops K at the saturation point on
    a planted frontier; the wager verdict CONFIRMS when manifold ties the linear
    ceiling at fewer atoms and REFUTES (honestly) when it never reaches it.
  * OLMo (filled from the battery JSON) — runs the same two functions on the
    measured recon_parity frontier and asserts the verdict is internally
    consistent; skipped (not faked) until olmo_battery_results.json lands.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

_SRC = (Path(__file__).resolve().parents[1] / "gamfit" / "_sae_manifold.py").read_text()


def _load() -> dict:
    ns: dict = {"np": __import__("numpy"), "Any": object, "Mapping": dict}
    for fn in ("ev_knee_k", "wager_verdict"):
        m = re.search(r"\ndef " + fn + r"\(.*?\n(?=\ndef |\Z)", _SRC, re.S)
        assert m is not None, f"{fn} not found"
        exec(m.group(0), ns)
    return ns


H = _load()
ev_knee_k = H["ev_knee_k"]
wager_verdict = H["wager_verdict"]

BATTERY_JSON = Path(__file__).resolve().parents[1] / "olmo_battery_results.json"
OLMO_FIXTURE = Path(__file__).resolve().parent / "data" / "olmo_mixedlayer_pca64_768.npy"


def test_knee_stops_at_saturation():
    # EV jumps to 0.8 at K=2 then crawls: knee at K=2.
    frontier = {1: 0.50, 2: 0.80, 3: 0.805, 4: 0.806}
    selection = ev_knee_k(frontier, return_details=True)
    assert selection["k"] == 2
    assert selection["flag"] == "knee"


def test_knee_keeps_climbing_while_gain_is_real():
    frontier = {1: 0.30, 2: 0.50, 3: 0.70, 4: 0.90}
    selection = ev_knee_k(frontier, return_details=True)
    assert selection["k"] == 4
    assert selection["flag"] == "linear"


def test_knee_single_point():
    assert ev_knee_k({3: 0.7}) == 3


def test_wager_confirmed_when_manifold_ties_linear_at_lower_k():
    # Linear ceiling 0.80 at K=8; manifold reaches 0.80 already at K=2.
    manifold = {1: 0.60, 2: 0.82, 3: 0.83}
    linear = {2: 0.50, 4: 0.70, 8: 0.82}
    v = wager_verdict(manifold, linear)
    assert v["confirmed"] is True
    assert v["manifold_k"] == 2
    assert v["linear_k"] == 8
    assert v["efficiency_ratio"] == 4.0


def test_wager_refuted_honestly_when_manifold_never_reaches_linear():
    # Manifold tops out below the linear ceiling — the wager loses measurably.
    #
    # The advantage is adjudicated at the EV the auto-K choice actually reaches
    # (recommend_auto_k: target_ev = selection.ev, the manifold's own knee EV),
    # NOT at the linear ceiling. So the manifold always reaches its own operating
    # EV (manifold_k is a real K, here at most len(manifold)); refutation shows up
    # as no compression — manifold_k is not strictly below linear_k — so confirmed
    # is False, and the residual ev_gap to the linear ceiling is positive.
    manifold = {1: 0.40, 2: 0.55, 3: 0.58}
    linear = {2: 0.60, 4: 0.75, 8: 0.85}
    v = wager_verdict(manifold, linear)
    assert v["confirmed"] is False
    # The manifold reaches its own knee EV at some K; the wager is refused
    # because that buys no parameter-efficiency over linear (manifold_k >= linear_k).
    assert v["manifold_k"] is not None
    assert v["linear_k"] is not None
    assert v["manifold_k"] >= v["linear_k"]
    assert v["efficiency_ratio"] is None or v["efficiency_ratio"] <= 1.0
    # The wager loses measurably: the linear ceiling sits above the manifold's best.
    assert v["ev_gap"] == pytest.approx(0.85 - 0.58)


def _ev(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1.0e-12)


def _olmo_linear_frontier() -> dict[int, float]:
    z = np.load(OLMO_FIXTURE).astype(np.float64)
    assert z.shape == (768, 64)
    train = z[:384]
    test = z[384:512]
    mu = train.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(train - mu, full_matrices=False)
    frontier: dict[int, float] = {}
    for k in (1, 2, 4, 8, 16, 32):
        proj = vt[:k].T @ vt[:k]
        frontier[k] = _ev(test, (test - mu) @ proj)
    return frontier


def test_olmo_fixture_linear_frontier_is_real_and_bounded():
    frontier = _olmo_linear_frontier()
    assert frontier[1] == pytest.approx(0.2961278177, abs=1e-9)
    assert frontier[8] == pytest.approx(0.5497310414, abs=1e-9)
    assert frontier[32] == pytest.approx(0.7669424107, abs=1e-9)
    selection = ev_knee_k(frontier, return_details=True)
    assert selection["k"] in frontier
    assert selection["flag"] in {"knee", "no_knee", "linear"}


@pytest.mark.skipif(not BATTERY_JSON.exists(), reason="OLMo battery artefact not present yet")
def test_olmo_recon_parity_verdict_is_consistent():
    blob = json.loads(BATTERY_JSON.read_text())
    rp = blob.get("recon_parity", [])
    manifold = {int(r["K"]): float(r["manifold_ev"]) for r in rp if "manifold_ev" in r}
    linear = {int(r["K"]): float(r["linear_ev"]) for r in rp if "linear_ev" in r}
    if not manifold or not linear:
        pytest.skip("recon_parity frontier incomplete in artefact")
    knee = ev_knee_k(manifold)
    v = wager_verdict(manifold, linear)
    assert knee in manifold
    if v["confirmed"]:
        assert v["manifold_k"] is not None and v["efficiency_ratio"] is not None
    else:
        assert v["ev_gap"] >= 0.0
