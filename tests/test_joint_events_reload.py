"""#2961 A11 across processes: a joint event model fitted and saved through
gamfit reloads in a fresh interpreter and forecasts bit for bit what the
in-memory model forecasts (``docs/latent-signatures.md``). Floats are
compared as ``float.hex``, never within a tolerance.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import gamfit

MARKS = {"diagnosis": "once", "cvd_death": "terminal", "other_death": "terminal", "visit": "recurrent"}
HISTORY = (0.5, 2.5, [(1.0, "visit")])
HORIZONS = [0.25, 3.0, 40.0]
FIELDS = ("survival", "incidence", "incidence_error")


def _hex(values) -> list[str]:
    return [float(v).hex() for v in np.asarray(values, dtype=float).ravel()]


def test_a_saved_joint_model_forecasts_bit_identically_in_a_fresh_process(tmp_path) -> None:
    subjects = pd.DataFrame({"id": ["a", "b"], "entry": [0.0, 0.0], "exit": [4.0, 6.0]})
    events = pd.DataFrame(
        {"id": ["a", "b", "b", "b"], "time": [4.0, 0.0, 1.0, 5.0], "mark": ["cvd_death", "diagnosis", "visit", "visit"]}
    )
    model = gamfit.fit_joint_event_model(subjects, events, marks=MARKS)
    in_memory = model.forecast(*HISTORY, HORIZONS)
    # The once-only diagnosis against both terminal causes takes the bracketed
    # quadrature route, so its reported error is part of what must reproduce.
    assert any(e > 0.0 for e in np.asarray(in_memory["incidence_error"]).ravel())

    path = tmp_path / "model.json"
    model.save(path)
    script = (
        "import json, sys, numpy as np, gamfit\n"
        "m = gamfit.load_joint_event_model(sys.argv[1])\n"
        f"f = m.forecast({HISTORY[0]!r}, {HISTORY[1]!r}, {HISTORY[2]!r}, {HORIZONS!r})\n"
        "print(json.dumps({k: [float(v).hex() for v in np.asarray(f[k], dtype=float).ravel()]"
        f" for k in {FIELDS!r}}}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)], check=True, capture_output=True, text=True
    )
    reloaded = json.loads(completed.stdout)
    for key in FIELDS:
        assert reloaded[key] == _hex(in_memory[key]), key


def test_gamfit_load_reads_a_saved_joint_model_by_its_envelope_kind_3053(tmp_path) -> None:
    """``gamfit.load`` is the inverse of a joint model's ``save``: the saved
    document's header names the kind ``joint``, and the load dispatches on it."""
    subjects = pd.DataFrame({"id": ["a", "b"], "entry": [0.0, 0.0], "exit": [4.0, 6.0]})
    events = pd.DataFrame(
        {"id": ["a", "b", "b", "b"], "time": [4.0, 0.0, 1.0, 5.0], "mark": ["cvd_death", "diagnosis", "visit", "visit"]}
    )
    model = gamfit.fit_joint_event_model(subjects, events, marks=MARKS)
    in_memory = model.forecast(*HISTORY, HORIZONS)
    path = tmp_path / "model.json"
    model.save(path)
    for reloaded in (gamfit.load(path), gamfit.loads(path.read_bytes())):
        assert isinstance(reloaded, gamfit.JointEventModel)
        forecast = reloaded.forecast(*HISTORY, HORIZONS)
        for key in FIELDS:
            assert _hex(forecast[key]) == _hex(in_memory[key]), key

    document = json.loads(path.read_text())
    document["version"] += 1
    with pytest.raises(gamfit.errors.GamfitError, match="refit"):
        gamfit.loads(json.dumps(document).encode())
    document["kind"] = "not-a-model-kind"
    with pytest.raises(gamfit.errors.SchemaMismatchError, match="not-a-model-kind"):
        gamfit.loads(json.dumps(document).encode())
