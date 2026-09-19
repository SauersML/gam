"""A latent coordinate that reaches a center must price its design gradient, not panic.

Latent rows move freely during the joint REML search, and data-derived centers
start on latent rows, so `t = c` is reachable on the first evaluation. The design
row's gradient `∂φ(‖t − c‖)/∂t` has norm `|φ'(r)|`:

* a kernel that is C¹ at the origin (`φ'(0⁺) = 0`: Matérn ν ≥ 3/2, the
  two-dimensional order-2 Duchon block `r² log r`) has gradient 0 there, and the
  fit proceeds;
* a kernel with a cone point at the center (Matérn ν = 1/2) has no gradient there,
  and the fit is refused by name when the latent operator is built.

Before, both classes died on "fit_table panicked inside Rust boundary: ...
DegenerateAtCollision": the collision classifier read the Duchon block's
opposite-signed ψ-scaling exponent and demanded a finite Hessian scalar where the
gradient needs only `φ'(0⁺) = 0`, and `LatentCoordDerivativeOp::materialize_local`
turned the error into a panic.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

N = 96


def _fixture() -> tuple[Any, Any, Any]:
    rng = np.random.default_rng(7)
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    rgb = np.column_stack(
        [0.5 + 0.45 * np.cos(theta), 0.5 + 0.45 * np.sin(theta), rng.uniform(0.0, 1.0, N)]
    )
    y = np.sin(theta) + 0.15 * rng.normal(size=N)
    return pd.DataFrame({"y": y}), theta, rgb


def _fit(formula: str, d: int) -> Any:
    data, theta, rgb = _fixture()
    latent = gamfit.smooth.LatentCoord(
        n=N,
        d=d,
        init="pca" if d == 2 else theta[:, None],
        aux_prior={"u": rgb if d == 2 else theta[:, None]},
    )
    return gamfit.fit(data, formula, latents={"t": latent})


@pytest.mark.parametrize(
    "formula,d",
    [
        ("y ~ s(t, type='duchon', centers=32)", 2),
        ("y ~ s(t, type='matern', nu=3/2, centers=12)", 1),
    ],
)
def test_a_c1_latent_kernel_fits_through_a_center_collision(formula: str, d: int) -> None:
    model = _fit(formula, d)
    assert np.isfinite(model.summary().reml_score)


def test_a_cone_point_latent_kernel_is_refused_by_name_not_by_a_panic() -> None:
    with pytest.raises(gamfit.errors.GamError, match="cone point") as refusal:
        _fit("y ~ s(t, type='matern', nu=1/2, centers=12)", 1)
    assert "panicked inside Rust boundary" not in str(refusal.value)
