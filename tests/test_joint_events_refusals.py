"""#2961: gamfit hands the joint event model its tables as given. Event rows may
come in any order, and an invalid table is refused with the encoder's own
message; gamfit keeps no table checks of its own (``docs/latent-signatures.md``).
"""
from __future__ import annotations

import pandas as pd
import pytest

import gamfit

MARKS = {"diagnosis": "once", "cvd_death": "terminal", "visit": "recurrent"}
SUBJECTS = pd.DataFrame({"id": ["a", "b"], "entry": [0.0, 0.0], "exit": [4.0, 6.0]})
SORTED = pd.DataFrame(
    {
        "id": ["a", "a", "b", "b"],
        "time": [4.0, 4.0, 1.0, 5.0],
        "mark": ["diagnosis", "cvd_death", "visit", "visit"],
    }
)


def _hex(values) -> list[str]:
    return [float(v).hex() for v in values.ravel()]


def test_event_rows_in_any_order_fit_and_forecast_bit_identically(tmp_path) -> None:
    # The same rows interleaved across subjects, the later visit first, and the
    # terminal event listed before its simultaneous diagnosis.
    shuffled = SORTED.iloc[[3, 1, 2, 0]].reset_index(drop=True)
    saved = []
    for name, events in (("sorted", SORTED), ("shuffled", shuffled)):
        path = tmp_path / f"{name}.json"
        gamfit.fit_joint_event_model(SUBJECTS, events, marks=MARKS).save(path)
        saved.append(path.read_bytes())
    assert saved[0] == saved[1]

    model = gamfit.load_joint_event_model(tmp_path / "sorted.json")
    in_order = model.forecast(0.5, 2.5, [(1.0, "visit"), (2.0, "visit")], [0.25, 3.0])
    out_of_order = model.forecast(0.5, 2.5, [(2.0, "visit"), (1.0, "visit")], [0.25, 3.0])
    for key in ("survival", "incidence", "incidence_error"):
        assert _hex(in_order[key]) == _hex(out_of_order[key]), key


def test_invalid_tables_are_refused_with_the_encoders_message() -> None:
    visit = pd.DataFrame({"id": ["a"], "time": [1.0], "mark": ["visit"]})
    # Positive control: the same visit fits against distinct, known subjects.
    assert gamfit.fit_joint_event_model(SUBJECTS, visit, marks=MARKS).mark_kinds == MARKS
    unknown = pd.DataFrame({"id": ["c"], "time": [1.0], "mark": ["visit"]})
    with pytest.raises(ValueError, match='subject "c" is not in the subjects table'):
        gamfit.fit_joint_event_model(SUBJECTS, unknown, marks=MARKS)
    duplicated = pd.DataFrame({"id": ["a", "a"], "entry": [0.0, 0.0], "exit": [4.0, 6.0]})
    with pytest.raises(ValueError, match='subject "a" has two rows'):
        gamfit.fit_joint_event_model(duplicated, visit, marks=MARKS)
