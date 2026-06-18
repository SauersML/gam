"""Bug hunt: tensor-product (`te`/`ti`) per-term EDF in `Model.summary()`
exceeds the model's total EDF and the term's own column count.

A term's effective degrees of freedom is tr(F) restricted to that term's
coefficient block, where F is the (whole-model) influence/hat operator. A
sub-block trace can never exceed the full-model trace, so per-term EDF must
satisfy `edf_term <= edf_total`; it also cannot exceed the number of
coefficients the term spans.

For a tensor smooth with several marginal penalty blocks sharing one coefficient
range, the user-facing `Model.summary()` (persisted-model path) reports a
per-term EDF that is the *sum of per-penalty-block EDFs*, double-counting the
shared tensor coefficients — exactly the #1219 defect. #1219 is marked fixed and
its in-tree Rust unit test
(`src/main/model_summary.rs::tensor_product_per_term_edf_does_not_exceed_total`)
passes, but that test builds the summary from a fresh in-memory fit that still
carries the influence matrix (the basis-invariant tier-1 estimate). The Python
`summary()` reads a serialized / column-conditioned model where the influence
matrix is dropped (`src/solver/estimate/penalty.rs:419`) and the per-term EDF
falls through to the legacy unclamped block-sum fallback
(`src/main/model_summary.rs:61-64`). So the bug still reaches every user through
the persisted summary path that #1219's fix and test never covered.

Observed: `te(x, z, k=[6,6])` reports term EDF ~40.0 with edf_total ~28.0 and
only 36 coefficients in the whole model.
"""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import pandas as pd

import gamfit


def _tensor_summary(formula, seed=11, n=600):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + np.cos(2.0 * np.pi * z) + 0.3 * rng.standard_normal(n)
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    m = gamfit.fit(df, formula, family="gaussian")
    return m.summary()


def test_te_per_term_edf_does_not_exceed_model_total():
    s = _tensor_summary("y ~ te(x, z, k=[6,6])")
    n_coef = len(s.coefficients)
    edf_total = float(s.edf_total)
    for term in s.smooth_terms:
        edf = float(term["edf"])
        # A sub-block trace cannot exceed the whole-model trace.
        assert edf <= edf_total + 1e-6, (
            f"per-term EDF {edf:.4f} for {term['name']!r} exceeds the model "
            f"total EDF {edf_total:.4f}: per-penalty-block EDFs are being summed, "
            "double-counting the shared tensor coefficients (persisted-summary "
            "path of #1219)"
        )
        # And it cannot exceed the number of coefficients in the whole model,
        # let alone the term's own column count.
        assert edf <= n_coef + 1e-6, (
            f"per-term EDF {edf:.4f} for {term['name']!r} exceeds the model's "
            f"total coefficient count {n_coef}"
        )


def test_ti_per_term_edf_does_not_exceed_model_total():
    s = _tensor_summary("y ~ ti(x, z, k=[6,6])")
    edf_total = float(s.edf_total)
    for term in s.smooth_terms:
        edf = float(term["edf"])
        assert edf <= edf_total + 1e-6, (
            f"per-term EDF {edf:.4f} for {term['name']!r} exceeds the model "
            f"total EDF {edf_total:.4f}"
        )
