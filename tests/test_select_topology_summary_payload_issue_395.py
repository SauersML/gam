"""Regression test for issue #395.

``gamfit.select_topology`` raised ``ValueError: select_topology could not
determine fitted basis size`` on every clean dataset because
``_select_topology._summary_payload`` gated on
``isinstance(payload, collections.abc.Mapping)``. The object returned by
``Model.summary()`` is a ``gamfit._summary.Summary`` frozen dataclass: it
duck-types the mapping protocol (``__getitem__`` / ``__contains__`` /
``__iter__`` / ``.get`` / ``.to_dict``) but is **not** a registered
``Mapping`` instance, so the ``isinstance`` check was always ``False`` and the
entire summary payload was silently discarded.

These tests pin the behaviour against the *real* ``Summary`` type (not a dict
stand-in) so the regression cannot reappear by reintroducing an ``isinstance``
gate.
"""

import collections.abc as abc

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


def test_summary_is_not_a_mapping_instance():
    # The premise of the bug: the real Summary type is NOT an abc.Mapping, so
    # any isinstance(summary, Mapping) gate discards it. If this ever flips
    # (Summary becomes a registered Mapping), the bug is moot but the guard
    # below still documents the contract.
    summary = _make_summary(24, 7.5)
    assert not isinstance(summary, abc.Mapping), (
        "Summary must not be a Mapping instance — _summary_payload must read it "
        "through its public duck-typed interface, not isinstance(..., Mapping)."
    )


def test_summary_payload_reads_real_summary_object():
    class _Fit:
        def __init__(self, summary: Summary):
            self._summary = summary

        def summary(self) -> Summary:
            return self._summary

    fit = _Fit(_make_summary(24, 7.5))
    payload = st._summary_payload(fit)
    assert payload is not None, (
        "_summary_payload must return the flattened summary for a real "
        "Summary object, not None (issue #395)."
    )
    coefficients = payload.get("coefficients")
    assert coefficients is not None and len(coefficients) == 24


def test_basis_size_resolves_from_real_summary():
    class _Fit:
        def __init__(self, summary: Summary):
            self._summary = summary

        def summary(self) -> Summary:
            return self._summary

    fit = _Fit(_make_summary(24, 7.5))
    # This is the exact call that raised ValueError in the issue repro.
    assert st._basis_size(fit) == 24


def test_effective_dim_resolves_from_real_summary():
    class _Fit:
        def __init__(self, summary: Summary):
            self._summary = summary

        def summary(self) -> Summary:
            return self._summary

    fit = _Fit(_make_summary(24, 7.5))
    assert st._effective_dim(fit) == 7.5
