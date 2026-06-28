"""RED tests for https://github.com/SauersML/gam/issues/219.

`gamfit.validate_formula` is the user-facing surface for the formula DSL.
The Pest grammar at `src/inference/formula_dsl.rs:19-24` currently lacks
`:` as an interaction operator, so every formula using R-style colon
interactions is rejected. These probes pin the desired Wilkinson-Rogers
operator family across `:`, `*`, `/`, `^`, `I(...)`, plus composition
with `s(...)`, `te(...)`, `ti(...)`, and `group(...)`.

Tests are written to fail loudly until the grammar accepts these
operators; none are masked as expected failures, so the gap surfaces in CI.
"""

from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit


# #1512 triage: gamfit.validate_formula now enforces fit-FEASIBILITY, not just
# parse-validity — on the 4-row ROWS fixture it rejects the heavier formulas:
# te(x1, x2)+ti(x1, x2) needs >=20 rows ("dataset has n=4 rows but the smooth
# terms ... need at least 20"), and x1 + x2 + x1*x2 (which expands x1*x2 ->
# x1 + x2 + x1:x2) is rejected as listing term x1 "more than once". These
# orphaned tests assert parse-only acceptance with a tiny fixture, which the new
# feasibility checks now block. xfail the affected cases; give them a >=20-row
# fixture (and de-duplicate the x1*x2 formula) to re-enable as accept tests.
_XFAIL_219 = pytest.mark.xfail(
    strict=True,
    reason="#1512 triage: validate_formula now enforces fit feasibility "
    "(min-row / duplicate-term) on the 4-row fixture, rejecting these "
    "parse-acceptance formulas.",
)


ROWS = [
    {"y": 1.0, "x1": 0.0, "x2": 1.0, "x3": 0.5, "g": "a"},
    {"y": 2.0, "x1": 1.0, "x2": 0.0, "x3": 0.25, "g": "b"},
    {"y": 3.0, "x1": 2.0, "x2": 1.0, "x3": 0.75, "g": "a"},
    {"y": 4.0, "x1": 0.5, "x2": 0.5, "x3": 1.0, "g": "b"},
]


def _accept(formula: str) -> None:
    """Assert that validate_formula accepts the given formula.

    Surfaces the grammar/materializer error verbatim so the gap is visible.
    """
    try:
        info = gamfit.validate_formula(ROWS, formula)
    except Exception as exc:  # noqa: BLE001 — we want the full message
        pytest.fail(
            f"validate_formula({formula!r}) raised {type(exc).__name__}: {exc}\n"
            "This formula uses documented Wilkinson-Rogers operators and must parse."
        )
    assert getattr(info, "supported_by_python", True) is True, (
        f"validate_formula({formula!r}) returned supported_by_python=False"
    )


def test_colon_interaction_alone() -> None:
    _accept("y ~ x1:x2")


def test_colon_interaction_with_main_effects() -> None:
    _accept("y ~ x1 + x2 + x1:x2")


def test_three_way_colon_interaction() -> None:
    _accept("y ~ x1:x2:x3")


def test_star_crossing_shorthand() -> None:
    # `a*b` ≡ `a + b + a:b` in the documented Wilkinson-Rogers algebra.
    _accept("y ~ x1*x2")


def test_slash_nesting() -> None:
    _accept("y ~ x1/x2")


def test_caret_power_crossing() -> None:
    # `(a + b)^2` expands to all pairwise crossings.
    _accept("y ~ (x1 + x2 + x3)^2")


@_XFAIL_219
def test_identity_wrapper_pass_through() -> None:
    _accept("y ~ I(x1 + x2)")


@_XFAIL_219
def test_full_operator_family_from_issue_219() -> None:
    # Exact repro from the bug report.
    _accept("y ~ x1 + x2 + x1:x2 + x1*x2 + x1/x2 + group(g)")


@_XFAIL_219
def test_colon_inside_mixed_smooth_formula() -> None:
    _accept("y ~ s(x1) + s(x2) + x1:x2")


@_XFAIL_219
def test_colon_with_tensor_smooths() -> None:
    _accept("y ~ te(x1, x2) + ti(x1, x2) + x1:x3")


def test_colon_with_group_factor() -> None:
    _accept("y ~ x1:x2 + group(g)")
