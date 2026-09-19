from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

import gamfit
from gamfit._response_geometry import (
    ResponseGeometryModel,
    alr,
    closure,
    clr,
    geometry_log_map,
    fit_response_geometry,
    inverse_alr,
    simplex_exp_map,
    simplex_frechet_mean,
    sphere_exp_map,
    sphere_frechet_mean,
    sphere_log_map,
)


def test_fit_response_geometry_runtime_type_hints_resolve() -> None:
    hints = get_type_hints(fit_response_geometry)

    assert {"latents", "smooths", "constraints"} <= hints.keys()


def test_simplex_frechet_mean_is_geometric_not_extrinsic() -> None:
    y = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], dtype=float)
    mean = simplex_frechet_mean(y)
    expected = closure(np.exp(np.log(y).mean(axis=0)).reshape(1, -1))[0]
    extrinsic = y.mean(axis=0)

    np.testing.assert_allclose(mean, expected, rtol=1e-12, atol=1e-12)
    assert not np.allclose(mean, extrinsic)
    assert np.isclose(mean.sum(), 1.0)
    assert np.all(mean > 0.0)


def test_simplex_clr_and_alr_roundtrip_at_frechet_base() -> None:
    y = np.array(
        [[0.55, 0.30, 0.15], [0.20, 0.50, 0.30], [0.10, 0.15, 0.75]],
        dtype=float,
    )
    base = simplex_frechet_mean(y)

    clr_tangent = clr(y) - clr(base.reshape(1, -1))
    np.testing.assert_allclose(
        simplex_exp_map(clr_tangent, base, coordinates="clr"), y, rtol=1e-12, atol=1e-12
    )

    alr_tangent = alr(y, reference=2) - alr(base.reshape(1, -1), reference=2)
    np.testing.assert_allclose(
        simplex_exp_map(alr_tangent, base, coordinates="alr", reference=2),
        y,
        rtol=1e-12,
        atol=1e-12,
    )


def test_inverse_alr_accepts_single_coordinate_vector_roundtrip() -> None:
    x = np.array([0.2, 0.5, 0.3], dtype=float)
    expected = closure(x)

    for reference in (-1, 0, 1):
        coords = alr(x, reference=reference)
        reconstructed = inverse_alr(coords, reference=reference)

        assert coords.shape == (2,)
        assert reconstructed.shape == x.shape
        np.testing.assert_allclose(
            reconstructed,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )


def test_sphere_frechet_mean_is_intrinsic_and_log_exp_roundtrip() -> None:
    angles = np.array([-0.4, 0.1, 0.7], dtype=float)
    y = np.column_stack([np.cos(angles), np.sin(angles)])
    mean = sphere_frechet_mean(y)
    extrinsic = y.mean(axis=0)

    assert np.isclose(np.linalg.norm(mean), 1.0)
    assert not np.isclose(np.linalg.norm(extrinsic), 1.0)

    tangent = sphere_log_map(y, mean)
    np.testing.assert_allclose(sphere_exp_map(tangent, mean), y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tangent @ mean, np.zeros(y.shape[0]), atol=1e-12)


def test_sphere_antipodal_log_is_undefined_and_mean_is_not_endpoint() -> None:
    y = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)

    try:
        sphere_log_map(y[1:], y[0])
    except ValueError as exc:
        assert "antipodal" in str(exc)
    else:
        raise AssertionError("sphere_log_map should reject antipodal points")

    mean = sphere_frechet_mean(y)
    assert np.isclose(np.linalg.norm(mean), 1.0)
    assert abs(float(mean @ y[0])) < 1e-8

    endpoint_obj = 0.5 * np.pi**2
    mean_obj = float(np.mean(np.arccos(np.clip(y @ mean, -1.0, 1.0)) ** 2))
    assert mean_obj < endpoint_obj
    assert np.isclose(mean_obj, 0.25 * np.pi**2)


def test_geometry_log_map_resolves_simplex_aliases() -> None:
    y = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]], dtype=float)
    tangent_simplex, base_simplex, coords_simplex = geometry_log_map(y, geometry="simplex")
    tangent_clr, base_clr, coords_clr = geometry_log_map(y, geometry="clr")
    tangent_alr, _base_alr, coords_alr = geometry_log_map(y, geometry="alr")

    assert coords_simplex == "clr"
    assert coords_clr == "clr"
    assert coords_alr == "alr"
    np.testing.assert_allclose(tangent_simplex, tangent_clr)
    np.testing.assert_allclose(base_simplex, base_clr)
    assert tangent_alr.shape == (2, 2)


def test_response_geometry_model_predict_projects_back_to_manifold() -> None:
    class DummyCoordinateModel:
        """A coordinate model's predict table: the estimand-explicit schema a
        plain Gaussian tangent ``Model`` publishes, point under ``posterior_mean``."""

        def __init__(self, values: list[float]) -> None:
            self.values = values

        def predict(
            self,
            data: object,
            return_type: str | None = None,
            **kwargs: object,
        ) -> dict[str, list[float]]:
            return {
                "linear_predictor_plugin": self.values,
                "mean_plugin": self.values,
                "posterior_mean": self.values,
            }

        def summary(self) -> dict[str, str]:
            return {"ok": "yes"}

    base = np.array([0.25, 0.25, 0.50], dtype=float)
    target = np.array([[0.50, 0.25, 0.25], [0.20, 0.30, 0.50]], dtype=float)
    tangent = clr(target) - clr(base.reshape(1, -1))
    model = ResponseGeometryModel(
        models=[DummyCoordinateModel(tangent[:, j].tolist()) for j in range(3)],
        response_geometry="simplex",
        response_columns=("a", "b", "c"),
        base_point=base,
        coordinates="clr",
    )

    pred = model.predict({"x": [1.0, 2.0]}, return_type="dict")
    out = np.column_stack([pred[name] for name in ("a", "b", "c")])
    np.testing.assert_allclose(out, target, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(2), atol=1e-12)


def _simplex_frame(n: int) -> dict[str, list[float]]:
    x = np.linspace(0.0, 1.0, n)
    y = closure(np.column_stack([0.3 + 0.2 * x, 0.3 + 0.0 * x, 0.4 - 0.2 * x]))
    return {
        "x": x.tolist(),
        "g": ["a", "b"] * (n // 2),
        "y0": y[:, 0].tolist(),
        "y1": y[:, 1].tolist(),
        "y2": y[:, 2].tolist(),
    }


_DROPPED_BY_THE_JOINT_TANGENT_FIT = {
    "latents": lambda n: {
        "t": gamfit.smooth.LatentCoord(n=n, d=1, init=np.zeros((n, 1)), aux_prior={"u": np.zeros((n, 1))})
    },
    "smooths": lambda n: {"x": gamfit.smooth.BSpline()},
    "constraints": lambda n: {"s(x)": "monotone_increasing"},
    "penalties": lambda n: [
        gamfit.penalties.AuxConditionalPriorPenalty(
            lambda_per_row=np.ones((n, 1, 1)), weight=1.0, n_eff=n, target="t"
        )
    ],
    "precision_hyperpriors": lambda n: {"g": (3.0, 0.5)},
    "scale_dimensions": lambda n: True,
}


@pytest.mark.parametrize("argument", sorted(_DROPPED_BY_THE_JOINT_TANGENT_FIT))
def test_response_geometry_refuses_an_argument_its_joint_tangent_fit_would_drop(
    argument: str,
) -> None:
    """The joint tangent REML materializes only the formula and the weights.

    An argument it cannot see would shape the template model the predictions use
    but not the fit the coefficients come from, so `fit` must refuse it by name
    rather than return a model of something the user did not ask for.
    """
    n = 8
    with pytest.raises(ValueError, match=f"^{argument} is not supported with response_geometry$"):
        gamfit.fit(
            _simplex_frame(n),
            "y0 ~ s(x)",
            response_geometry="simplex",
            response_columns=["y0", "y1", "y2"],
            **{argument: _DROPPED_BY_THE_JOINT_TANGENT_FIT[argument](n)},
        )
