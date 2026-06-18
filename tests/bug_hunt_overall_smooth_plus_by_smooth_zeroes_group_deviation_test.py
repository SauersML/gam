"""Bug hunt: an overall `s(x)` plus a by-factor smooth `s(x, by=g)` silently
zeroes the group-specific deviation.

`y ~ s(x) + s(x, by=g)` is the canonical mgcv "shared trend + per-group
deviation" model, explicitly recommended in `docs/difference-smooths.md`. The
overall `s(x)` carries the population shape; each `by=g` level smooth is that
group's deviation from it.

The smooth-overlap / ownership machinery treats the narrower `s(x)` as an
"owner" of the wider `s(x, by=g)` purely by feature-column subset
(`src/terms/smooth/structure_analysis.rs:160`, with `by_col` folded into the
feature set at `structure_analysis.rs:11`), then orthogonalizes the by-level
basis against the realized `s(x)` design
(`src/terms/smooth/design_construction.rs:640-724`). Because a per-level by-smooth
is *gated* (zero off-level) it is wrongly judged numerically spanned by the
ungated overall smooth, and its identifiability transform collapses to zero
columns (`design_construction.rs:720` returns an `(ncols, 0)` matrix). The fit
succeeds and predicts without error, but every group's shape collapses to the
shared trend plus a constant offset.

This is distinct from the closed #978 (which was a *predict* crash / design
mismatch on the `s(x)+fs(x,g)` / `bs=sz` forms): here predict works fine and the
fit is silently wrong.

Repro: B's true shape is A's plus cos(2*pi*x), so the recovered B-A difference
must track that cos (span ~1.9). With the by-smooth zeroed the recovered B-A is
a flat constant (span ~0).
"""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import pandas as pd

import gamfit


def _recovered_difference(formula, seed=0, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    g = rng.choice(["A", "B"], n)
    shape_a = np.sin(2.0 * np.pi * x)
    shape_b = np.sin(2.0 * np.pi * x) + np.cos(2.0 * np.pi * x)  # B - A = cos(2*pi*x)
    y = np.where(g == "A", shape_a, shape_b) + rng.normal(0.0, 0.1, n)
    df = pd.DataFrame({"x": x, "g": g, "y": y})

    m = gamfit.fit(df, formula, family="gaussian")
    xs = np.linspace(0.05, 0.95, 21)
    p_a = np.asarray(
        m.predict(pd.DataFrame({"x": xs, "g": ["A"] * len(xs)})), dtype=float
    ).ravel()
    p_b = np.asarray(
        m.predict(pd.DataFrame({"x": xs, "g": ["B"] * len(xs)})), dtype=float
    ).ravel()
    return xs, (p_b - p_a)


def test_overall_plus_by_smooth_preserves_group_shape_difference():
    # The recovered B - A must be a genuine x-varying curve (true span ~1.9),
    # not a flat constant. A by-only model recovers it; adding the overall s(x)
    # must not destroy it.
    xs, diff = _recovered_difference("y ~ s(x) + s(x, by=g)")
    span = float(diff.max() - diff.min())
    assert span > 0.8, (
        "y ~ s(x) + s(x, by=g): the recovered B-A group difference is nearly "
        f"constant (span={span:.4f}); the by-factor deviation smooth was "
        "orthogonalized away to zero columns by the overall s(x). True B-A is "
        "cos(2*pi*x) with span ~1.9."
    )
    # And it should actually look like cos(2*pi*x) (the by-only fit reaches
    # ~0.22 max error; a non-broken combined fit must be at least this good).
    true_diff = np.cos(2.0 * np.pi * xs)
    max_err = float(np.max(np.abs(diff - true_diff)))
    assert max_err < 0.5, (
        "recovered B-A does not track the true cos(2*pi*x) group difference "
        f"(max abs error={max_err:.4f})"
    )


def test_by_only_smooth_recovers_difference_control():
    # Control: the by-only model is unaffected and recovers the difference.
    # This guards against the assertion failing for some unrelated reason.
    xs, diff = _recovered_difference("y ~ s(x, by=g)")
    span = float(diff.max() - diff.min())
    assert span > 0.8, span
