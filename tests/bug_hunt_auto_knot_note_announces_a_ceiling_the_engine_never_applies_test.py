"""Bug hunt: the automatic-knot note (once a ``GamInferenceWarning``, now an
informational entry in ``model.notes``) states a rule the engine does not
implement. It announces a ceiling of ``max(20, cbrt(unique))`` and then reports
a count that is capped at 8 -- on every default 1-D ``s()`` fit with more than
35 unique covariate values, i.e. essentially every real dataset.

The note is self-describing: it prints the rule, the unique count, the resolved
ceiling, and the value it chose, so its internal consistency is checkable without
any reference to the implementation. Measured (``y ~ s(x)``, ``x`` an ``n``-point
grid on [0, 1]), reading the note's own numbers back:

    unique   announced ceiling   rule evaluates to   APPLIED   coefficients
        16          20                   4              4           8
        20          20                   5              5           9
        24          20                   6              6          10
        32          20                   8              8          12
        40          20                  10              8          12   <- diverges
        60          20                  15              8          12
       100          20                  20              8          12
      5000          20                  20              8          12
    100000          46                  46              8          12   <- 5.75x

``clamp(unique/4, 4 .. announced_ceiling)`` is what the note itself says the
engine did; it agrees with what the engine actually did only while
``unique <= 35``. The announced ceiling is ``max(20, ...)``, so it is at least
20 for EVERY dataset, and the applied ceiling is 8 for every dataset: the two can
never coincide. This is not a corner case.

Mechanism. ``crates/gam-terms/src/term_builder.rs:4410-4416`` is the rule that
runs::

    pub fn heuristic_knots_for_column(col: ArrayView1<'_, f64>) -> usize {
        /// Default cubic basis = `MAX_DEFAULT_INTERNAL_KNOTS + degree + 1` = 12
        /// functions, matching mgcv's lean univariate default.
        const MAX_DEFAULT_INTERNAL_KNOTS: usize = 8;
        let unique = unique_count_column(col);
        (unique / 4).clamp(4, MAX_DEFAULT_INTERNAL_KNOTS)
    }

and ``crates/gam-terms/src/term_builder.rs:2600-2609`` is the note, which
computes a SECOND ceiling that is used for nothing but the string::

    let ceiling = ((unique as f64).cbrt() as usize).max(20);
    inference_notes.push(format!(
        "Automatically set {} internal knots for smooth '{}' from {} unique values \
         (rule: clamp(unique/4, 4..max(20, cbrt(unique))) = clamp(unique/4, 4..{})). \
         Override with knots=... or k=....",
        n_knots, vars.join(","), unique, ceiling,
    ));

``ceiling`` never reaches ``heuristic_knots_for_column``; ``n_knots`` never sees
``ceiling``.

Why it matters beyond the string: at ``n = 200_000`` (``y = sin(6*pi*x) +
0.7*cos(2*pi*x) + N(0, 0.4)``) the default fit realizes 11 smooth columns, its
RMSE against the noiseless truth stops improving at ``n >= 50_000``
(0.0406 at both 50k and 200k, against 0.0413 at 10k), and the engine's own #2774
basis-adequacy check reports ``p = 5.6e-107`` -- the residuals carry structure
this basis cannot represent. A user who reads the note is told the rule granted
58 internal knots on those rows. It granted 8.

Observed: the note's reported value disagrees with the note's own stated rule
for every column with 36 or more unique values, and the ceiling it announces is
unreachable for every dataset.

Expected: the note and the engine agree -- either the note prints the ceiling the
engine applies, or the engine applies the ceiling the note prints. This test
asserts only their agreement, so either repair turns it green.

Repaired by having the note print the ceiling the engine applies: it now reads
``(rule: clamp(unique/4, 4..8) = <value>; ...)``, and the regex below parses
that form. The agreement assertions are unchanged.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_NOTE = re.compile(
    r"Automatically set (?P<applied>\d+) internal knots for smooth '[^']*' "
    r"from (?P<unique>\d+) unique values "
    r"\(rule: clamp\(unique/4, 4\.\.(?P<ceiling>\d+)\) = (?P<stated>\d+);"
)

#: Unique counts where the note and the implementation still agree.
CONSISTENT = [16, 20, 24, 32]
#: Unique counts where they diverge.
DIVERGENT = [40, 60, 100, 5000, 100000]


def _note_for(unique: int) -> tuple[int, int, int, int]:
    """Return ``(applied, unique, announced_ceiling, n_coefficients)``.

    The note is informational (a default the engine chose, not a departure from
    the request), so it is read from ``model.notes`` rather than from warnings.
    """
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 1.0, unique)
    y = np.sin(4.0 * np.pi * x) + 0.3 * rng.standard_normal(unique)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="gaussian")
    notes = [_NOTE.search(note) for note in model.notes]
    matched = [m for m in notes if m is not None]
    assert matched, f"unique={unique}: no automatic-knot note was recorded"
    hit = matched[0]
    return (
        int(hit.group("applied")),
        int(hit.group("unique")),
        int(hit.group("ceiling")),
        len(model.summary().coefficients),
    )


def _rule(unique: int, ceiling: int) -> int:
    """``clamp(unique/4, 4..ceiling)`` -- the rule the note itself states."""
    return min(max(unique // 4, 4), ceiling)


@pytest.mark.parametrize("unique", CONSISTENT)
def test_control_the_note_is_self_consistent_below_the_hidden_cap(unique: int) -> None:
    """Green today: while ``unique/4 <= 8`` the two rules coincide, so the
    failures below are the divergence itself, not the parse or the fixture."""
    applied, reported_unique, ceiling, _ = _note_for(unique)
    assert reported_unique == unique
    assert applied == _rule(unique, ceiling)


@pytest.mark.parametrize("unique", CONSISTENT + DIVERGENT)
def test_control_the_coefficient_count_matches_the_reported_knot_count(unique: int) -> None:
    """Green today: a cubic B-spline with ``m`` internal knots realizes
    ``m + degree + 1`` columns, one of which the sum-to-zero chart trades for the
    intercept. The note's number is the number the basis was actually built
    with -- it is the RULE that is misreported, not the count."""
    applied, _, _, n_coefficients = _note_for(unique)
    assert n_coefficients == applied + 4


@pytest.mark.parametrize("unique", DIVERGENT)
def test_the_note_reports_the_value_its_own_stated_rule_produces(unique: int) -> None:
    applied, reported_unique, ceiling, _ = _note_for(unique)
    expected = _rule(reported_unique, ceiling)
    assert applied == expected, (
        f"unique={reported_unique}: the note states "
        f"clamp(unique/4, 4..{ceiling}) = {expected} and reports {applied}"
    )


def test_the_announced_ceiling_is_reachable_by_some_dataset() -> None:
    """The old ``max(20, cbrt(unique))`` ceiling was at least 20 for every
    dataset while the engine capped at 8, so it bound on nothing, ever. The
    announced ceiling must be one the engine can actually reach."""
    reached = False
    ceilings: list[tuple[int, int, int]] = []
    for unique in DIVERGENT:
        applied, reported_unique, ceiling, _ = _note_for(unique)
        ceilings.append((reported_unique, ceiling, applied))
        reached = reached or applied >= ceiling
    assert reached, (
        "the ceiling the note announces is never the ceiling the basis is "
        f"built with: (unique, announced_ceiling, applied) = {ceilings}"
    )
