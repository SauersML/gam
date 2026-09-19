"""Regression test for issue #395.

``gamfit.topology.select_topology`` raised ``ValueError: select_topology could not
determine fitted basis size`` on every clean dataset because its summary
reader gated on ``isinstance(payload, collections.abc.Mapping)``. The object
returned by ``Model.summary()`` is a ``gamfit._summary.Summary`` frozen
dataclass: it duck-types the mapping protocol (``__getitem__`` /
``__contains__`` / ``__iter__`` / ``.get`` / ``.to_dict``) but is **not** a
registered ``Mapping`` instance, so the ``isinstance`` check was always
``False`` and the entire summary payload was silently discarded.

The probing helpers that carried that gate are gone (#2899): each fitted
candidate's metrics are read once, through ``Summary.to_dict()``, by
``_select_topology._fitted_candidate_outcome``. These tests pin that reader
against the *real* ``Summary`` type (not a dict stand-in) so the regression
cannot reappear by reintroducing an ``isinstance`` gate.
"""

import collections.abc as abc
from typing import Any

import gamfit._select_topology as st
from gamfit._summary import Summary


def _make_summary(n_coeffs: int, edf: float) -> Summary:
    coefficients = [{"index": i, "estimate": 0.0} for i in range(n_coeffs)]
    return Summary.from_dict(
        {
            "formula": "y ~ s(theta, type=circle)",
            "family_name": "Gaussian Identity",
            "coefficients": coefficients,
            "edf_total": edf,
            "null_dim": 1.0,
        }
    )


class _Fit:
    def __init__(self, summary: Summary):
        self._summary = summary

    def summary(self) -> Summary:
        return self._summary


def _outcome(summary: Summary) -> dict[str, Any]:
    candidate = st._Candidate("circle", object.__new__(st.PeriodicSplineCurve))
    return st._fitted_candidate_outcome(candidate, _Fit(summary), raw_reml=-1.0, n_obs=24)


def test_summary_is_not_a_mapping_instance():
    # The premise of the bug: the real Summary type is NOT an abc.Mapping, so
    # any isinstance(summary, Mapping) gate discards it. If this ever flips
    # (Summary becomes a registered Mapping), the bug is moot but the guard
    # below still documents the contract.
    summary = _make_summary(24, 7.5)
    assert not isinstance(summary, abc.Mapping), (
        "Summary must not be a Mapping instance — select_topology must read it "
        "through Summary.to_dict(), not isinstance(..., Mapping)."
    )


def test_summary_payload_reads_real_summary_object():
    outcome = _outcome(_make_summary(24, 7.5))
    assert outcome["status"] == "fitted", (
        "_fitted_candidate_outcome must read the flattened summary of a real "
        "Summary object (issue #395)."
    )
    assert outcome["null_dim"] == 1.0


def test_basis_size_resolves_from_real_summary():
    # The basis size is the quantity whose resolution raised ValueError in the
    # issue repro.
    assert _outcome(_make_summary(24, 7.5))["basis_size"] == 24


def test_effective_dim_resolves_from_real_summary():
    assert _outcome(_make_summary(24, 7.5))["effective_dim"] == 7.5
