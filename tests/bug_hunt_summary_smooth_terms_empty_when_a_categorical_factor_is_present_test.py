"""Bug hunt: ``summary().smooth_terms`` is EMPTY for every model whose formula
carries a categorical main effect, even though the model has smooths and reports
their EDF in ``edf_total``.

``y ~ s(x) + group(g)`` and ``y ~ factor(g) + s(x)`` compile to the same fit --
identical ``edf_total`` (11.471127141633094), identical lambdas
(``[58.3026050, 31.4486350, 0.0512120]``), identical deviance
(48.89499867068343), 17 coefficients, identical ``term_blocks``
(``intercept`` / ``g`` random_effect / ``s(x)`` smooth_bspline1d), and identical
``smooth_significance`` output (``statistic_lr = 908.7175395435211`` in both).
Only the summary table differs:

    y ~ s(x) + group(g)     smooth_terms = [
        {'name': 'g',    'edf': 2.5143, 'ref_df': 2.5143},
        {'name': 's(x)', 'edf': 7.9569, 'ref_df': 9.1928,
         'chi_sq': 2520.2388, 'p_value': 2.27e-186}]

    y ~ factor(g) + s(x)    smooth_terms = []
                            smooth_terms_frame() -> Empty DataFrame, Columns: []

It reproduces for ``factor(g)``, a bare categorical column ``g``, a numeric
``factor(gi)``, and with ``te(x, z)`` in place of ``s(x)`` -- i.e. for the most
ordinary GAM shape there is, a smooth plus a categorical main effect. The
per-term EDF / ref_df / chi-square / p-value table is simply unavailable there.

``Summary.smooth_terms_frame``'s own docstring states the contract:

    This is the canonical mgcv ``summary.gam`` per-smooth significance table:
    columns ``name``, ``edf``, ``ref_df``, ``chi_sq``, ``p_value`` (``chi_sq`` /
    ``p_value`` are absent for random-effect smooths and any shape-constrained
    term, matching the engine, which only computes the Wood Wald test for
    ordinary penalized smooths).

``s(x)`` is an ordinary penalized smooth; it is not excluded by that carve-out,
and the ``group(g)`` spelling of the same model proves the engine computes its
Wald test.

``Model.smooth_significance(data)`` -- the likelihood-ratio sibling -- finds
``s(x)`` in both spellings, so the loss is specific to the summary's Wald table.

Root cause, and the discriminator that isolates it. ``summary_smooth_terms``
(``crates/gam-pyffi/src/manifold/manifold_and_posterior_ffi.rs:1288``) rebuilds a
representative design from the frozen spec and bails to an EMPTY table when that
rebuild fails::

    let factor_levels = frozen_factor_levels_by_col(spec);
    let data = representative_data_from_ranges(ranges, &factor_levels);
    let Ok(design) = gam::terms::smooth::build_term_collection_design(data.view(), spec)
        else { return Vec::new(); };

and ``frozen_factor_levels_by_col`` (~line 996) collects levels by walking ONLY
``spec.smooth_terms``::

    for term in &spec.smooth_terms {
        walk(&term.basis, &mut record);
    }

so a categorical column that appears only as a MAIN EFFECT contributes no frozen
levels, ``representative_data_from_ranges`` fills it with an axis midpoint
instead of a level bit pattern, and the rebuild fails. The comment directly above
that bail-out records the same class of bug being fixed once already, for factor
smooths and factor ``by=`` (#1370): "that failure was being swallowed to an EMPTY
table for the whole model, erasing co-fitted ``s(x)`` rows too".

The discriminator, on one dataset with one factor ``b``:

    y ~ s(x) + b                     smooth_terms = []
    y ~ s(x, by=b) + b               smooth_terms = ['b', 's(x, by=b):by=b[b0]',
                                                     's(x, by=b):by=b[b1]',
                                                     's(x, by=b):by=b[b2]']
    y ~ s(x) + s(x, by=b) + b        smooth_terms = ['b', 's(x)', ...4 more]
    y ~ s(x) + s(b, x, bs='sz')      smooth_terms = ['s(x)', "s(b, x, bs='sz')"]
    y ~ s(x) + s(x, b, bs='fs')      smooth_terms = ['s(x)', "s(x, b, bs='fs')"]

Adding one ``by=b`` smooth restores the whole table -- INCLUDING the row for the
very main effect that was erasing it -- because the ``by=`` envelope is what puts
``b``'s frozen levels into the collector.

Observed: ``summary().smooth_terms == []`` whenever a categorical main effect is
in the formula.

Expected: the smooth rows are present and equal to the ones the identical
``group(...)`` spelling reports.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_N = 500


def _data() -> dict[str, Any]:
    rng = np.random.default_rng(31)
    x = rng.uniform(0.0, 1.0, _N)
    return {
        "x": x,
        "z": rng.uniform(0.0, 1.0, _N),
        "g": np.array([f"g{i % 5}" for i in range(_N)]),
        "gi": np.array([float(i % 5) for i in range(_N)]),
        "y": np.sin(2.0 * np.pi * x) + 0.3 * rng.standard_normal(_N),
    }


def _summary(formula: str) -> Any:
    return gamfit.fit(_data(), formula, family="gaussian").summary()


def _row(summary: Any, name: str) -> dict[str, Any] | None:
    return next((t for t in (summary.smooth_terms or []) if t.get("name") == name), None)


def _assert_same_inference_row(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Require the full per-term inference contract, including absent fields."""
    assert actual.keys() == expected.keys()
    assert actual["name"] == expected["name"]
    for field in actual.keys() - {"name"}:
        assert actual[field] == pytest.approx(expected[field], rel=1e-8, abs=0.0)


def test_reference_group_spelling_reports_the_smooth_row() -> None:
    """Green today: the same model, spelled with group(), has the table."""
    row = _row(_summary("y ~ s(x) + group(g)"), "s(x)")
    assert row is not None
    assert row["edf"] > 1.0 and row["chi_sq"] > 0.0 and 0.0 <= row["p_value"] <= 1.0


def test_the_two_spellings_are_the_same_fit() -> None:
    """Premise: this is one model, so any summary difference is a reporting bug."""
    a = _summary("y ~ s(x) + group(g)")
    b = _summary("y ~ factor(g) + s(x)")
    assert b.edf_total == pytest.approx(a.edf_total, rel=1e-12)
    assert b.deviance == pytest.approx(a.deviance, rel=1e-12)
    np.testing.assert_allclose(np.asarray(b.lambdas), np.asarray(a.lambdas), rtol=1e-10)
    for name in ("g", "s(x)"):
        expected = _row(a, name)
        actual = _row(b, name)
        assert expected is not None and actual is not None
        _assert_same_inference_row(actual, expected)


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ factor(g) + s(x)",
        "y ~ g + s(x)",
        "y ~ s(x) + factor(g)",
        "y ~ factor(gi) + s(x)",
    ],
)
def test_smooth_row_survives_a_categorical_main_effect(formula: str) -> None:
    reference = _row(_summary("y ~ s(x) + group(g)"), "s(x)")
    assert reference is not None

    summary = _summary(formula)
    row = _row(summary, "s(x)")
    assert row is not None, (
        f"{formula}: summary().smooth_terms has no row for s(x); it is "
        f"{summary.smooth_terms!r} while edf_total is {summary.edf_total!r}"
    )
    _assert_same_inference_row(row, reference)


def test_tensor_smooth_row_survives_a_categorical_main_effect() -> None:
    summary = _summary("y ~ factor(g) + te(x, z)")
    assert _row(summary, "te(x, z)") is not None, (
        "summary().smooth_terms has no row for te(x, z): "
        f"{summary.smooth_terms!r} (edf_total {summary.edf_total!r})"
    )


@pytest.mark.parametrize(
    "restoring_term",
    ["s(x, by=b)", "s(b, x, bs='sz')", "s(x, b, bs='fs')"],
)
def test_control_a_by_or_factor_smooth_restores_the_whole_table(restoring_term: str) -> None:
    """Green today: the same factor, once it also appears inside a smooth, is fine.

    This is the discriminator for the root cause -- the level collector walks
    only ``spec.smooth_terms``, so a factor that reaches it through a ``by=`` /
    ``sz`` / ``fs`` envelope rebuilds cleanly while a main-effect-only factor
    does not.
    """
    rng = np.random.default_rng(31)
    n = 500
    x = rng.uniform(0.0, 1.0, n)
    data = {
        "x": x,
        "b": np.array([f"b{i % 3}" for i in range(n)]),
        "y": np.sin(2.0 * np.pi * x) + 0.3 * rng.standard_normal(n),
    }
    summary = gamfit.fit(data, f"y ~ s(x) + {restoring_term}", family="gaussian").summary()
    assert summary.smooth_terms, f"y ~ s(x) + {restoring_term} lost the table too"


def test_main_effect_only_factor_keeps_the_table() -> None:
    """The same factor, present only as a main effect, erases every row."""
    rng = np.random.default_rng(31)
    n = 500
    x = rng.uniform(0.0, 1.0, n)
    data = {
        "x": x,
        "b": np.array([f"b{i % 3}" for i in range(n)]),
        "y": np.sin(2.0 * np.pi * x) + 0.3 * rng.standard_normal(n),
    }
    plain = gamfit.fit(data, "y ~ s(x) + b", family="gaussian").summary()
    withby = gamfit.fit(data, "y ~ s(x) + s(x, by=b) + b", family="gaussian").summary()
    assert [t.get("name") for t in (withby.smooth_terms or [])].count("s(x)") == 1
    assert plain.smooth_terms, (
        "y ~ s(x) + b reports smooth_terms == [] while y ~ s(x) + s(x, by=b) + b "
        "on the same data reports the s(x) row plus a row for b itself"
    )


def test_smooth_terms_frame_has_the_documented_columns() -> None:
    pytest.importorskip("pandas")
    frame = _summary("y ~ factor(g) + s(x)").smooth_terms_frame()
    for column in ("name", "edf", "ref_df"):
        assert column in frame.columns, (
            f"smooth_terms_frame() is missing the documented {column!r} column; "
            f"got {list(frame.columns)!r}"
        )
