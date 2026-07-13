"""Bug hunt: expectile (Newey-Powell LAWS) regression is fully implemented in
the Rust orchestration layer but is UNREACHABLE from the public Python API (and
the CLI), because both call ``materialize`` / ``fit_model`` directly and bypass
the only place the expectile family is dispatched.

The engine has a complete asymmetric-least-squares expectile estimator:
``crates/gam-models/src/fit_orchestration/entry.rs`` carries
``expectile_tau_for_config`` (parses ``family="expectile"`` / ``"expectile(0.9)"``
and the ``expectile_tau`` field, validating ``tau in (0,1)``),
``expectile_row_weights`` (the per-row ``w_i = tau if y_i>mu_i else 1-tau``
asymmetric weights), and ``fit_expectile_laws`` (the iteratively-reweighted
Gaussian-identity GAM fixed-point driver, ~130 lines). It works: called through
``gam::fit_from_formula`` it produces a correct, strictly tau-monotone family of
expectiles (verified directly in Rust: tau=0.1 lies below the median fit at
every grid point, tau=0.9 above it).

But that dispatch lives ONLY in ``fit_from_formula`` (entry.rs ~line 380):

    if let Some(tau) = expectile_tau_for_config(config)? {
        return fit_expectile_laws(formula, data, config, tau);
    }
    let mat = materialize(formula, data, config)?;   // <- normal path

The Python FFI ``fit_dataset_impl``
(``crates/gam-pyffi/src/manifold/geometry_ffi.rs`` ~line 4861) and the CLI
``run_fit`` (``crates/gam-cli/src/main/run_fit.rs`` ~line 829) both skip
``fit_from_formula`` and call ``materialize(...)`` then ``fit_model(...)``
themselves. ``materialize``'s family resolver
(``crates/gam-models/src/fit_orchestration/materialize/family.rs``) has no
``expectile`` arm, so it raises:

    InvalidConfigurationError: unknown family 'expectile(0.9)'; expected one of:
    auto, gaussian, binomial/bernoulli, ..., tweedie/tw, ...,
    royston-parmar, transformation-normal

i.e. an estimator the codebase ships, tests, and documents internally cannot be
invoked from either user-facing interface.

Observed: ``gamfit.fit(df, "y ~ s(x)", family="expectile(0.9)")`` raises
``unknown family``. Expected: it fits the 0.9-expectile, exactly as the Rust
``fit_from_formula`` path does.

This test asserts the public Python API can fit expectiles AND that the result
is a genuine expectile family (strictly monotone in tau: the 0.9-expectile sits
above the median fit and the 0.1-expectile below it, at every grid point). The
monotonicity is the defining property of expectiles, so this is not a tautology;
it pins that the family, once routed, behaves as the engine already computes it.
It fails today at the first expectile ``fit`` call. When the FFI/CLI route the
expectile family the way ``fit_from_formula`` does (e.g. share the
``expectile_tau_for_config`` shortcut, or have ``materialize`` recognize it),
the test passes without edits.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _make_data(seed: int = 7, n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    # Homoscedastic noise around a smooth mean: the conditional expectiles are
    # vertical shifts of the mean curve, so a correct estimator must order them
    # strictly by tau everywhere.
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.4, n)
    return pd.DataFrame({"x": x, "y": y})


def _predict_grid(model: Any) -> np.ndarray:
    grid = pd.DataFrame({"x": np.linspace(0.05, 0.95, 9)})
    return np.asarray(model.predict(grid)).ravel()


def test_expectile_family_is_fittable_from_python_and_monotone_in_tau(
    tmp_path: Any,
) -> None:
    df = _make_data()

    # The bug: this call raises InvalidConfigurationError("unknown family
    # 'expectile(0.9)'") today, even though gam::fit_from_formula fits it.
    lo = _predict_grid(gamfit.fit(df, "y ~ s(x)", family="expectile(0.1)"))
    mid = _predict_grid(gamfit.fit(df, "y ~ s(x)", family="expectile(0.5)"))
    hi_model = gamfit.fit(df, "y ~ s(x)", family="expectile(0.9)")
    hi = _predict_grid(hi_model)

    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(mid)) and np.all(np.isfinite(hi))

    # Defining property of expectiles: strictly ordered in tau at every point.
    assert np.all(lo < mid), f"0.1-expectile must lie below the median fit:\nlo={lo}\nmid={mid}"
    assert np.all(hi > mid), f"0.9-expectile must lie above the median fit:\nhi={hi}\nmid={mid}"

    # An expectile is an asymmetric-loss target, not a probability law. The
    # saved artifact must retain tau and refuse observation generation rather
    # than silently interpreting the Gaussian inner solver as Gaussian noise.
    model_path = tmp_path / "expectile.gam"
    hi_model.save(model_path)
    saved = json.loads(model_path.read_text(encoding="utf-8"))
    assert saved["payload"]["estimator"] == {
        "estimator_kind": "expectile",
        "tau": 0.9,
    }
    restored = gamfit.load(model_path)
    with pytest.raises(Exception, match="expectile.*no observation-replicate sampler"):
        restored.sample_replicates(pd.DataFrame({"x": [0.25, 0.75]}), 3, seed=4)
