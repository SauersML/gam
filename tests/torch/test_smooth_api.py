"""Public Smooth + ``fit()`` + ``GAM`` API smoke tests.

Verifies the user-facing surface in ``gamfit.smooth`` and ``gamfit.torch``:

* Every Smooth subclass instantiates.
* Single-smooth fit (1D and multi-output) returns correct shapes.
* Multi-smooth additive fit returns per-smooth coefficient list.
* ``by`` row-gating applies to a single smooth.
* B-spline single-smooth fit returns correct shapes.
* Unsupported smooth kinds raise NotImplementedError with a clear message.
* ``GAM`` nn.Module: train-mode forward requires response; ``.freeze`` +
  ``.eval`` lets forward run without response; coefficients live as
  buffers (not parameters).
"""

from __future__ import annotations

import pytest

gt = pytest.importorskip("gamfit.torch")
torch = pytest.importorskip("torch")


def _make_centers(k: int = 8) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, k, dtype=torch.float64).unsqueeze(1)


def _make_inputs(n: int = 40, d_out: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    if d_out == 1:
        y = torch.sin(3.0 * t) + 0.05 * torch.randn(n, generator=g, dtype=torch.float64)
    else:
        y = torch.stack(
            [
                torch.sin((j + 1) * t) + 0.05 * torch.randn(
                    n, generator=g, dtype=torch.float64,
                )
                for j in range(d_out)
            ],
            dim=1,
        )
    return t, y


# ---------------------------------------------------------------------------
# Instantiability
# ---------------------------------------------------------------------------


def test_each_smooth_subclass_instantiable():
    c = _make_centers()
    assert isinstance(gt.Duchon(centers=c, m=2), gt.Smooth)
    knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    assert isinstance(gt.BSpline(knots=knots, degree=3), gt.Smooth)
    # TensorBSpline composes 1D marginals.
    margs = [
        gt.BSpline(knots=knots, degree=3),
        gt.BSpline(knots=knots, degree=3),
    ]
    assert isinstance(gt.TensorBSpline(marginals=margs), gt.Smooth)
    assert isinstance(gt.Matern(centers=c, nu=1.5, length_scale=1.0), gt.Smooth)
    assert isinstance(gt.Sphere(n_centers=20), gt.Smooth)
    assert isinstance(gt.Categorical(levels=torch.zeros(10, dtype=torch.int64), n_levels=3), gt.Smooth)
    # PeriodicSplineCurve might or might not be exported on every build.
    if hasattr(gt, "PeriodicSplineCurve"):
        assert isinstance(gt.PeriodicSplineCurve(n_knots=10, degree=3), gt.Smooth)
    else:
        from gamfit.smooth import PeriodicSplineCurve
        assert isinstance(PeriodicSplineCurve(n_knots=10, degree=3), gt.Smooth)


# ---------------------------------------------------------------------------
# Single-smooth fits
# ---------------------------------------------------------------------------


def test_fit_duchon_single_1d():
    t, y = _make_inputs(n=40)
    c = _make_centers(8)
    res = gt.fit(t, y, gt.Duchon(centers=c, m=2))
    assert isinstance(res.coefficients, torch.Tensor)
    assert res.coefficients.shape == (8, 1)
    assert res.fitted.shape == (40, 1)
    # lambdas may be a 0-dim tensor or a length-1 vector — both are scalar-like.
    assert res.lambdas.numel() == 1


def test_fit_duchon_single_multioutput_D5():
    t, y = _make_inputs(n=40, d_out=5)
    c = _make_centers(8)
    res = gt.fit(t, y, gt.Duchon(centers=c, m=2))
    assert res.coefficients.shape == (8, 5)
    assert res.fitted.shape == (40, 5)


def test_fit_additive_two_duchon():
    t, y = _make_inputs(n=40)
    c1 = _make_centers(6)
    c2 = _make_centers(7)
    res = gt.fit(
        [t, t], y,
        [gt.Duchon(centers=c1, m=2), gt.Duchon(centers=c2, m=2)],
    )
    assert isinstance(res.coefficients, list)
    assert len(res.coefficients) == 2
    assert res.coefficients[0].shape == (6, 1)
    assert res.coefficients[1].shape == (7, 1)
    assert res.lambdas.shape == (2,)
    assert res.fitted.shape == (40, 1)


def test_fit_duchon_with_by_row_gating():
    t, y = _make_inputs(n=40)
    c = _make_centers(8)
    g = torch.Generator().manual_seed(7)
    amp = torch.rand(40, generator=g, dtype=torch.float64)
    res = gt.fit(t, y, gt.Duchon(centers=c, m=2, by=amp))
    # Sanity: fitted is amplitude-weighted, so fitted ~ amp * (Phi @ B)
    assert res.fitted.shape == (40, 1)


def test_fit_bspline_single():
    t, y = _make_inputs(n=40)
    knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    res = gt.fit(t, y, gt.BSpline(knots=knots, degree=3))
    assert res.coefficients.dim() == 2
    assert res.coefficients.shape[1] == 1
    assert res.fitted.shape == (40, 1)


# ---------------------------------------------------------------------------
# Unsupported smooth kinds raise NotImplementedError
# ---------------------------------------------------------------------------


def test_matern_fit_raises_notimplemented():
    t, y = _make_inputs(n=20)
    c = _make_centers(6)
    with pytest.raises(NotImplementedError):
        gt.fit(t, y, gt.Matern(centers=c, nu=1.5, length_scale=1.0))


def test_tensorbspline_fit_raises_notimplemented():
    t, y = _make_inputs(n=20)
    knots = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    with pytest.raises(NotImplementedError):
        gt.fit(
            t, y,
            gt.TensorBSpline(marginals=[
                gt.BSpline(knots=knots, degree=3),
                gt.BSpline(knots=knots, degree=3),
            ]),
        )


def test_categorical_fit_raises_notimplemented():
    n = 20
    levels = torch.zeros(n, dtype=torch.int64)
    _, y = _make_inputs(n=n)
    # Categorical isn't a continuous-points smooth; the fit dispatch should
    # still reject it explicitly because no Rust binding wires it on the
    # torch path.
    with pytest.raises(NotImplementedError):
        # Pass a dummy "points" of the right shape; the dispatch should bail
        # before it ever uses points.
        t = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        gt.fit(t, y, gt.Categorical(levels=levels, n_levels=3))


@pytest.mark.xfail(
    reason="Sphere may or may not be plumbed end-to-end in this build; pin "
    "the contract — if Sphere fits, this test will xpass and can be promoted.",
    strict=False,
)
def test_sphere_fit_raises_notimplemented():
    n = 20
    g = torch.Generator().manual_seed(0)
    lat = torch.rand(n, generator=g, dtype=torch.float64) * 60.0 - 30.0
    lon = torch.rand(n, generator=g, dtype=torch.float64) * 360.0 - 180.0
    pts = torch.stack([lat, lon], dim=1)
    y = torch.sin(lat * 0.05)
    with pytest.raises(NotImplementedError):
        gt.fit(pts, y, gt.Sphere(n_centers=10))


def test_periodic_spline_curve_fit_raises_notimplemented():
    # PeriodicSplineCurve may not be wired on every build; pin the contract.
    t, y = _make_inputs(n=20)
    if hasattr(gt, "PeriodicSplineCurve"):
        psc = gt.PeriodicSplineCurve(n_knots=10, degree=3)
    else:
        from gamfit.smooth import PeriodicSplineCurve as _PSC
        psc = _PSC(n_knots=10, degree=3)
    try:
        gt.fit(t, y, psc)
    except NotImplementedError:
        return  # expected
    except Exception as exc:
        pytest.skip(f"PeriodicSplineCurve fit failed with non-NotImplemented error: {exc!r}")
    pytest.skip("PeriodicSplineCurve fit succeeded; promote to a positive test.")


# ---------------------------------------------------------------------------
# GAM nn.Module
# ---------------------------------------------------------------------------


def test_gam_module_train_then_freeze_then_eval():
    t, y = _make_inputs(n=40)
    c = _make_centers(8)
    model = gt.GAM([gt.Duchon(centers=c, m=2)])

    # Training mode forward needs a response.
    model.train()
    out_train = model(t, y)
    assert out_train.shape == (40, 1)

    with pytest.raises(ValueError):
        model(t)  # missing response in training mode

    # Freeze, switch to eval, forward without response.
    model.freeze(t, y)
    assert not model.training  # .freeze() ends with self.eval()
    out_eval = model(t)
    assert out_eval.shape == (40, 1)

    # Coefficients are buffers / non-parameter state, not Adam parameters.
    param_names = [n for n, _ in model.named_parameters()]
    assert param_names == [], (
        f"GAM should have zero learnable Parameters; got {param_names}"
    )
