"""Bug hunt: a factor-smooth (`fs(x, g)` / `s(x, g, bs="fs")` / `bs="sz"`) makes
``Model.summary().smooth_terms`` come back **empty** — and when an ordinary
`s(x)` is also in the formula, that normal smooth term silently disappears too.

The Python `summary()` rebuilds the per-smooth significance table by replaying the
frozen term spec on *representative* covariate values reconstructed from the
saved per-feature ``(lo, hi)`` ranges:

```
// crates/gam-pyffi/src/manifold_and_posterior_ffi.rs:1818
fn representative_data_from_ranges(ranges) -> Array2<f64> {
    ... data[[row, col]] = lo + frac * (hi - lo);   // 16 LINEARLY SPACED values
}
```

For a factor column those 16 linearly-spaced values are not valid levels (a
3-level factor encoded as {0.0, 1.0, 2.0} becomes 0.0, 0.133, …, 2.0). When
``build_term_collection_design`` rebuilds a factor-level-gated smooth
(`FactorSumToZero`, i.e. `fs`/`sz`) on that fabricated data the rebuild fails,
and ``summary_smooth_terms`` swallows the error and returns an **empty vector**
for the WHOLE table (manifold_and_posterior_ffi.rs:1875-1877, the
``let Ok(design) = build_term_collection_design(...) else { return Vec::new() }``
guard). So every smooth term vanishes — not just the factor-smooth one.

Observed:

```
y~s(x)               n_smooth_terms=1   edf_total=12.08
y~s(x)+fs(x,gg)      n_smooth_terms=0   edf_total=13.94   <-- s(x) vanished too
y~fs(x,gg)           n_smooth_terms=0   edf_total=22.61
```

The factor-smooth is genuinely fitted (3 active smoothing parameters,
edf_total≈22) — it is simply invisible to ``summary()``, taking any co-fitted
smooth down with it.

Related: the same representative-data design replay also mishandles `s(x, by=g)`
factor smooths (last level's EDF dropped, term names are f64 bit patterns) —
see #1368 and #1369.
"""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import pandas as pd

import gamfit


def _fit(formula, seed=11, n=1800):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    g = rng.choice(["p", "q", "r"], n)
    y = np.sin(2.0 * np.pi * x) + 0.3 * rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "gg": g, "y": y})
    return gamfit.fit(df, formula, family="gaussian")


def test_factor_smooth_appears_in_summary_smooth_terms():
    """An `fs(x, g)` factor-smooth is the whole model (edf_total ≫ 1, several
    active smoothing parameters), so it must contribute at least one row to
    summary().smooth_terms."""
    model = _fit("y~fs(x,gg)")
    summary = model.summary()
    assert float(summary.edf_total) > 2.0, "factor-smooth model should be non-trivial"
    assert len(summary.smooth_terms) >= 1, (
        "fs(x, gg) is genuinely fitted (edf_total="
        f"{float(summary.edf_total):.2f}, {len(summary.lambdas)} smoothing "
        "parameters) but summary().smooth_terms is empty: the representative-data "
        "design replay fails on the factor column and the whole table is dropped."
    )


def test_factor_smooth_does_not_erase_a_coexisting_plain_smooth():
    """Adding an `fs(x, g)` term must not make the ordinary `s(x)` term vanish
    from summary().smooth_terms — `y ~ s(x)` alone reports it fine."""
    plain = _fit("y~s(x)").summary()
    assert len(plain.smooth_terms) == 1, "control: y~s(x) should report its smooth"

    combined = _fit("y~s(x)+fs(x,gg)").summary()
    assert len(combined.smooth_terms) >= 1, (
        "y~s(x)+fs(x,gg) reports zero smooth terms: the factor-smooth makes the "
        "representative-data design rebuild fail, and summary_smooth_terms returns "
        "an empty table for the ENTIRE model, erasing the perfectly ordinary s(x) "
        f"term too (edf_total={float(combined.edf_total):.2f})."
    )
