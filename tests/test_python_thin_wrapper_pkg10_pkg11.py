"""PKG-10 / PKG-11: logic the Python wrapper used to own now lives in Rust.

Each test pins a behaviour that used to be decided in ``gamfit/*.py`` and is
now decided by one Rust function shared with the CLI and the formula front
door:

* ``bspline_basis`` / ``duchon_basis`` with no size used a Python ``K = 10``,
  so ``bspline_basis(x)`` had 14 columns where ``s(x)`` builds 12 (or fewer
  on sparse data). They now take the formula default from Rust.
* The multinomial alias set lived in a Python ``set`` literal next to a second
  copy in Rust, and the Python branch dropped ``fisher_rao_w`` silently. The
  single Rust predicate now routes every spelling and refuses what the
  multinomial driver cannot honour.
* Python computed the classification panel's null prevalence and clipped
  ``predict_proba``; the Rust panel owns the null model and the posterior mean
  is returned as fitted.
* ``persistent_warm_start_root`` (an on-disk cache directory that does not
  change the fitted model) is no longer a fit keyword.
"""

from __future__ import annotations

import importlib
import inspect
import typing

np = typing.cast(typing.Any, importlib.import_module("numpy"))
pytest = typing.cast(typing.Any, importlib.import_module("pytest"))

import gamfit
from gamfit import _api
from gamfit._binding import rust_module


def _formula_default_open_dim(t: typing.Any) -> int:
    """``s(x)``'s default cubic basis dimension: ``clamp(unique/4, 4, 8) + 4``."""
    unique = len(np.unique(np.asarray(t, dtype=float)))
    return min(max(unique // 4, 4), 8) + 4


@pytest.mark.parametrize("n", [20, 40, 400])
def test_bspline_basis_default_has_the_formula_dimension(n: int) -> None:
    t = np.linspace(0.0, 1.0, n)
    basis = gamfit.bspline_basis(t)
    assert basis.shape == (n, _formula_default_open_dim(t))
    derivative = gamfit.bspline_basis_derivative(t)
    assert derivative.shape == basis.shape


def test_periodic_bspline_basis_default_has_the_cyclic_formula_dimension() -> None:
    t = np.linspace(0.0, 1.0, 400)
    # cyclic(x) default: min(internal + degree + 1, 12).
    assert gamfit.bspline_basis(t, periodic=True).shape == (400, 12)
    # An explicit K still names K + degree + 1 cyclic controls.
    assert gamfit.bspline_basis(t, 5, periodic=True).shape == (400, 5 + 3 + 1)


def test_duchon_default_centers_are_not_coarser_than_the_default_spline() -> None:
    t = np.linspace(-1.0, 1.0, 400)
    centers = _api._resolve_centers(None, t)
    assert centers.ndim == 1
    assert len(centers) >= _formula_default_open_dim(t)
    assert len(_api._resolve_centers(7, t)) == 7


def test_duchon_basis_size_without_centers_is_data_dependent() -> None:
    with pytest.raises(ValueError, match="depends on the data"):
        gamfit.Duchon().basis_size()


def test_python_owns_no_basis_defaults_or_multinomial_entry() -> None:
    for name in ("_DEFAULT_BASIS_K", "_periodic_uniform_grid"):
        assert not hasattr(_api, name), name
    rust = rust_module()
    for name in ("fit_multinomial_formula_pyfunc", "auto_knots_1d", "auto_centers_1d"):
        assert not hasattr(rust, name), name
    assert hasattr(rust, "resolve_basis_locations_1d")


@pytest.mark.parametrize(
    ("family", "multinomial"),
    [
        ("multinomial", True),
        ("Categorical", True),
        ("softmax", True),
        ("multinomial_logit", True),
        ("categorical-logit", True),
        ("binomial", False),
        ("gaussian", False),
        (None, False),
    ],
)
def test_multinomial_spelling_list_lives_in_the_engine(
    family: str | None, multinomial: bool
) -> None:
    # The sklearn classifier's binary/multi-class split asks the engine
    # predicate `fit_table` and the CLI route on; Python keeps no copy.
    assert not hasattr(_api, "MULTINOMIAL_FAMILY_NAMES")
    assert _api.is_multinomial_family(family) is multinomial
    if family is not None:
        assert rust_module().is_multinomial_family_name(family) is multinomial


def _three_class_table(n: int = 240) -> dict[str, typing.Any]:
    rng = np.random.default_rng(20260919)
    x = rng.uniform(-2.0, 2.0, n)
    eta = np.stack([0.0 * x, 1.0 + 0.8 * x, -0.5 - 1.2 * x], axis=1)
    p = np.exp(eta)
    p /= p.sum(axis=1, keepdims=True)
    labels = (rng.uniform(size=n)[:, None] < np.cumsum(p, axis=1)).argmax(axis=1)
    return {"x": x.tolist(), "y": ["abc"[int(c)] for c in labels]}


@pytest.mark.parametrize("family", ["multinomial", "Categorical", "softmax", "multinomial_logit"])
def test_every_multinomial_spelling_reaches_the_vector_driver(family: str) -> None:
    model = gamfit.fit(_three_class_table(), "y ~ x", family=family)
    assert isinstance(model, gamfit.MultinomialModel)
    probabilities = np.asarray(model.predict({"x": np.array([-1.0, 0.0, 1.0])}), dtype=float)
    assert probabilities.shape == (3, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)


def test_multinomial_refuses_fisher_rao_weights_instead_of_dropping_them() -> None:
    data = _three_class_table()
    with pytest.raises(Exception, match="fisher_rao_w"):
        gamfit.fit(data, "y ~ x", family="softmax", fisher_rao_w=np.ones(len(data["x"])))


def test_multinomial_refuses_warm_start_from() -> None:
    data = _three_class_table()
    scalar = gamfit.fit({"x": data["x"], "z": data["x"]}, "z ~ x", family="gaussian")
    with pytest.raises(Exception, match="warm_start_from"):
        gamfit.fit(data, "y ~ x", family="multinomial", warm_start_from=scalar)


def test_classification_panel_owns_its_null_prevalence() -> None:
    rust = rust_module()
    observed = [0.0, 0.0, 0.0, 1.0]
    predicted = [0.1, 0.2, 0.3, 0.7]
    in_sample = dict(rust.classification_metrics(observed, predicted))
    explicit = dict(rust.classification_metrics(observed, predicted, 0.25))
    assert in_sample["nagelkerke_r2"] == explicit["nagelkerke_r2"]
    # Held-out scoring passes the training prevalence, which changes R².
    held_out = dict(rust.classification_metrics(observed, predicted, 0.5))
    assert held_out["nagelkerke_r2"] != in_sample["nagelkerke_r2"]


def test_predict_proba_returns_the_posterior_mean_as_fitted() -> None:
    pd = importlib.import_module("pandas")
    from gamfit.sklearn import GAMClassifier

    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, 200)
    y = (rng.uniform(size=200) < 1.0 / (1.0 + np.exp(-x))).astype(int)
    X = pd.DataFrame({"x1": x})
    clf = GAMClassifier(formula="y ~ s(x1)", family="binomial").fit(X, y)
    proba = clf.predict_proba(X)
    posterior = np.asarray(
        clf.model_.predict(clf._strip_response_column(X), return_type="dict")["posterior_mean"],
        dtype=float,
    )
    np.testing.assert_array_equal(proba[:, 1], posterior)
    np.testing.assert_array_equal(proba[:, 0], 1.0 - posterior)


@pytest.mark.parametrize("entry", [gamfit.fit, gamfit.fit_array, gamfit.validate_formula])
def test_on_disk_warm_start_root_is_not_a_keyword(entry: typing.Any) -> None:
    assert "persistent_warm_start_root" not in inspect.signature(entry).parameters
