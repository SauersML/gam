"""#977 measure→improve, action 3: topology-verdict → dictionary warm-start seed.

The cross-class adjudicator (#907/#977) labels each recovered atom
circle/torus/sphere/euclidean/mixture. `topology_prior_seed` turns those
verdicts into the per-atom `atom_basis` + `d_atom` lists that warm-start a
subsequent `sae_manifold_fit`, so the refit starts from the discovered structure
instead of a flat circle prior.

These arms are live (pure mapping); they assert the seed mirrors the verdicts
and is shaped to pass straight into the fit API. The OLMo arm (reading real
adjudication verdicts from the battery JSON) is gated on the artefact.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parents[1] / "gamfit" / "_sae_manifold.py").read_text()


def _load() -> dict:
    ns: dict = {"np": __import__("numpy"), "Any": object, "Mapping": dict}
    # topology_prior_seed depends on the module-level _TOPOLOGY_TO_BASIS const.
    const = re.search(r"\n_TOPOLOGY_TO_BASIS = \{.*?\n\}\n", _SRC, re.S)
    assert const is not None
    exec(const.group(0), ns)
    m = re.search(r"\ndef topology_prior_seed\(.*?\n(?=\ndef |\Z)", _SRC, re.S)
    assert m is not None
    exec(m.group(0), ns)
    return ns


seed = _load()["topology_prior_seed"]

BATTERY_JSON = Path(__file__).resolve().parents[1] / "olmo_battery_results.json"


def test_seed_from_verdict_dicts():
    verdicts = [
        {"winner": "circle"},
        {"winner": "torus"},
        {"winner": "mixture_k7"},
        {"winner": "euclidean"},
    ]
    out = seed(verdicts)
    assert out["atom_basis"] == ["periodic", "torus", "euclidean", "euclidean"]
    assert out["d_atom"] == [2, 2, 1, 2]


def test_seed_from_bare_strings():
    out = seed(["sphere", "circle"])
    assert out["atom_basis"] == ["sphere", "periodic"]
    assert out["d_atom"] == [2, 2]


def test_unknown_verdict_falls_back_to_default():
    out = seed([{"winner": "n/a"}, {}])
    assert out["atom_basis"] == ["periodic", "periodic"]
    assert out["d_atom"] == [2, 2]


def test_seed_length_matches_atom_count():
    verdicts = [{"winner": "circle"}] * 5
    out = seed(verdicts)
    assert len(out["atom_basis"]) == 5 == len(out["d_atom"])


@pytest.mark.skipif(not BATTERY_JSON.exists(), reason="OLMo battery artefact not present yet")
def test_olmo_adjudication_seeds_a_valid_dictionary():
    blob = json.loads(BATTERY_JSON.read_text())
    verdicts = [a["shape_verdict"] for a in blob.get("adjudication", []) if "shape_verdict" in a]
    if not verdicts:
        pytest.skip("no adjudication verdicts in artefact")
    out = seed(verdicts)
    assert len(out["atom_basis"]) == len(verdicts)
    assert all(b in {"periodic", "torus", "sphere", "euclidean"} for b in out["atom_basis"])
    assert all(d in (1, 2) for d in out["d_atom"])
