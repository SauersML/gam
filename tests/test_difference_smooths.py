from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gamfit._rust")

import gamfit


def _frame(n: int = 40) -> pd.DataFrame:
    x = np.linspace(0.0, 1.0, n)
    g = np.where(np.arange(n) % 2 == 0, "A", "B")
    y = np.sin(2 * np.pi * x) + (g == "B") * (0.25 + 0.5 * x)
    return pd.DataFrame({"y": y, "x": x, "g": g})


def test_unordered_by_factor_smooth_and_difference_api_smoke() -> None:
    df = _frame()
    model = gamfit.fit(df, "y ~ s(x, k=6, by=g)")
    out = model.difference_smooth(view="x", group="g", n=12, n_sim=128, simultaneous=True)
    assert len(out) == 12
    assert set(["x", "level_1", "level_2", "diff", "se", "lower", "upper", "critical"]).issubset(out.columns)
    assert np.all(np.isfinite(out["diff"]))
    assert np.all(out["se"] >= 0)
    assert float(out["critical"].iloc[0]) >= 1.0


def test_group_means_false_targets_only_group_main_effect() -> None:
    """`group_means=False` must drop the grouping factor's parametric
    main-effect columns specifically (not "every random effect block"),
    so that with `marginalise_random=False` it leaves any *other* random
    effects in place. We don't rely on observable contrast differences
    here (the contrast `xr - xl` zeroes columns that don't depend on the
    changed factor anyway); we assert the semantic invariant that the
    main-effect ranges this code targets are exactly the term blocks
    belonging to the grouping factor."""
    df = _frame()
    model = gamfit.fit(df, "y ~ s(x, k=6, by=g) + g")
    state = model._coefficient_state()
    term_blocks = state.get("term_blocks") or []
    group_main_ranges = [
        (int(tb["start"]), int(tb["end"]))
        for tb in term_blocks
        if str(tb.get("name", "")) == "g"
        and str(tb.get("kind", "")) in ("linear", "random_effect")
    ]
    # `+ g` on a categorical column must produce exactly one parametric
    # block for the factor (one column per level under random-effect
    # encoding); without that block group_means=False would have nothing
    # specific to drop and silently degenerate into a no-op.
    assert len(group_main_ranges) == 1, (
        f"expected exactly one parametric main-effect block for 'g', "
        f"got {group_main_ranges!r} from term_blocks={term_blocks!r}"
    )
    start, stop = group_main_ranges[0]
    assert stop > start, "group main-effect range must be non-empty"

    # The block must be flagged as a recognised parametric kind; otherwise
    # the `group_means=False` filter above (kind in {linear, random_effect})
    # would skip it and revert to the buggy no-op behaviour.
    block_kinds = {tb.get("name"): tb.get("kind") for tb in term_blocks}
    assert block_kinds.get("g") in ("linear", "random_effect"), block_kinds

    # End-to-end sanity: both branches must run without error and produce
    # finite contrasts at every grid point.
    out_true = model.difference_smooth(
        view="x", group="g", n=8, group_means=True, marginalise_random=False
    )
    out_false = model.difference_smooth(
        view="x", group="g", n=8, group_means=False, marginalise_random=False
    )
    assert np.all(np.isfinite(out_true["diff"]))
    assert np.all(np.isfinite(out_false["diff"]))

