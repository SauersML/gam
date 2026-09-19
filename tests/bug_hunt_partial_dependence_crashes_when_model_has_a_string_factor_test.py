"""Bug hunt: ``Model.partial_dependence`` crashes on ANY model that contains a
string-valued categorical term -- it feeds the FFI's own encoding sentinel back
in as if it were a user-facing level.

    y ~ s(x) + factor(b)          # b in {"b0", "b1", "b2"}
    model.partial_dependence("s(x)", data)
    -> gamfit._rust.GamError: unseen level '\\x00b0' in categorical column 'b' at row 1

``y ~ s(x) + b`` (bare string column) fails identically. Drop the factor and the
same call succeeds, so nothing about the smooth is at fault -- the mgcv
``plot.gam()`` analogue is simply unavailable for the most ordinary GAM there is,
a smooth plus a categorical main effect.

Mechanism (``gamfit/_model.py``, ``Model.partial_dependence``):

    data_headers, data_rows, _ = normalize_table(data)      # ~line 1276
    if data_rows:
        first = data_rows[0]
        template.update({h: first[i] for i, h in enumerate(data_headers)})

``normalize_table``'s second return value is the ENCODED ``_EncodedTable``, not
the user's frame. Its categorical cells still carry
``gam::data::CATEGORICAL_CELL_SENTINEL`` -- the marker
``crates/gam-pyffi/src/model/model_ffi.rs:850-856`` prepends before level
inference:

    >>> normalize_table({"x": x, "b": b, "y": y})[1][0]
    ['0.8647975870165865', '\\x00b0', '-1.2135545809440378']

So ``template["b"]`` becomes ``"\\x00b0"``. Numeric cells survive the round trip
because the grid builder re-parses them with ``float(row[h])`` (~line 1372),
which is exactly why only categorical columns break. The categorical branch
passes the string through untouched, ``normalize_table(columns)`` re-encodes it
with a second sentinel, and the model's encoder reports the result as a level it
has never seen. The persisted schema is not a workaround either: its levels come
back quoted (``["'b0'", "'b1'", "'b2'"]``), so the ``levels[0]`` fallback a few
lines below would inject ``"'b0'"``.

Observed: ``GamError: unseen level '\\x00...'`` for every model with a
categorical term.

Expected: ``partial_dependence`` returns the term's partial effect. Because the
``s(x)`` design block does not depend on ``b`` at all, the answer is pinned
exactly, independent of which level the reference template chooses:
``predicted == X[:, block] @ beta[block]`` and
``standard_error == sqrt(diag(X_block V_block X_block^T))``, both of which this
file verifies against the factor-free model to 1e-12 before applying them to the
factor model.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_N = 400


def _data() -> dict[str, Any]:
    rng = np.random.default_rng(13)
    x = rng.uniform(0.0, 1.0, _N)
    return {
        "x": x,
        "b": np.array([f"b{i % 3}" for i in range(_N)]),
        "y": np.sin(2.0 * np.pi * x) + 0.3 * rng.standard_normal(_N),
    }


def _oracle(model: Any, grid: np.ndarray, frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """``X_t beta_t`` and ``sqrt(diag(X_t V_t X_t^T))`` for the ``s(x)`` block."""
    summary = model.summary()
    beta = np.asarray([c["estimate"] for c in summary.coefficients], dtype=float)
    cov = summary.covariance
    block = next(b for b in model.term_blocks if b.name == "s(x)")
    design = np.asarray(model.design_matrix(frame).matrix, dtype=float)
    columns = design[:, block.start : block.end]
    sub = cov[block.start : block.end, block.start : block.end]
    return columns @ beta[block.start : block.end], np.sqrt(
        np.einsum("ij,jk,ik->i", columns, sub, columns)
    )


def test_oracle_agrees_with_partial_dependence_without_a_factor() -> None:
    """Calibrates the oracle used below; must be green today."""
    data = _data()
    model = gamfit.fit({"x": data["x"], "y": data["y"]}, "y ~ s(x)", family="gaussian")

    result = model.partial_dependence("s(x)", n_points=9)
    grid = np.asarray(result["grid"], dtype=float)
    expected_fit, expected_se = _oracle(model, grid, {"x": grid})

    np.testing.assert_allclose(np.asarray(result["predicted"], dtype=float), expected_fit, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(result["standard_error"], dtype=float), expected_se, atol=1e-12
    )


@pytest.mark.parametrize("formula", ["y ~ s(x) + factor(b)", "y ~ s(x) + b"])
def test_partial_dependence_with_a_string_factor(formula: str) -> None:
    data = _data()
    model = gamfit.fit(data, formula, family="gaussian")

    result = model.partial_dependence("s(x)", n_points=9)

    grid = np.asarray(result["grid"], dtype=float)
    frame = {"x": grid, "b": np.array(["b0"] * grid.size)}
    expected_fit, expected_se = _oracle(model, grid, frame)

    np.testing.assert_allclose(
        np.asarray(result["predicted"], dtype=float),
        expected_fit,
        atol=1e-10,
        err_msg="partial_dependence disagrees with X_t @ beta_t for the s(x) block",
    )
    np.testing.assert_allclose(
        np.asarray(result["standard_error"], dtype=float),
        expected_se,
        atol=1e-10,
        err_msg="partial_dependence SE disagrees with sqrt(diag(X_t V_t X_t^T))",
    )
