"""Public smooth, fit, and GAM torch API smoke tests."""

import pytest

gt = pytest.importorskip("gamfit.torch")
torch = pytest.importorskip("torch")


def _centers(k=8):
    return torch.linspace(0.0, 1.0, k, dtype=torch.float64).unsqueeze(1)


def _inputs(n=40, d_out=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    if d_out == 1:
        y = torch.sin(3.0 * t) + 0.05 * torch.randn(
            n, generator=g, dtype=torch.float64,
        )
    else:
        y = torch.stack(
            [
                torch.sin((j + 1) * t)
                + 0.05 * torch.randn(n, generator=g, dtype=torch.float64)
                for j in range(d_out)
            ],
            dim=1,
        )
    return t, y


def test_smooth_subclasses_instantiable():
    centers = _centers()
    knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    marginals = [
        gt.BSpline(knots=knots, degree=3),
        gt.BSpline(knots=knots, degree=3),
    ]

    assert isinstance(gt.Duchon(centers=centers, m=2), gt.Smooth)
    assert isinstance(gt.BSpline(knots=knots, degree=3), gt.Smooth)
    assert isinstance(gt.TensorBSpline(marginals=marginals), gt.Smooth)
    assert isinstance(gt.Matern(centers=centers, nu=1.5, length_scale=1.0), gt.Smooth)
    assert isinstance(gt.Pca(K=2), gt.Smooth)
    assert isinstance(gt.Sphere(n_centers=20), gt.Smooth)
    assert isinstance(
        gt.Categorical(levels=torch.zeros(10, dtype=torch.int64), n_levels=3),
        gt.Smooth,
    )
    assert isinstance(gt.PeriodicSplineCurve(n_knots=10, degree=3), gt.Smooth)


def test_fit_duchon_single_1d():
    t, y = _inputs()
    res = gt.fit(t, y, gt.Duchon(centers=_centers(), m=2))

    assert isinstance(res.coefficients, torch.Tensor)
    assert res.coefficients.shape == (8, 1)
    assert res.fitted.shape == (40, 1)
    assert res.lambdas.numel() == 1


def test_fit_duchon_single_multioutput_D5():
    t, y = _inputs(d_out=5)
    res = gt.fit(t, y, gt.Duchon(centers=_centers(), m=2))

    assert res.coefficients.shape == (8, 5)
    assert res.fitted.shape == (40, 5)


def test_fit_additive_two_duchon():
    t, y = _inputs()
    res = gt.fit(
        [t, t],
        y,
        [gt.Duchon(centers=_centers(6), m=2), gt.Duchon(centers=_centers(7), m=2)],
    )

    assert isinstance(res.coefficients, list)
    assert len(res.coefficients) == 2
    assert res.coefficients[0].shape == (6, 1)
    assert res.coefficients[1].shape == (7, 1)
    assert res.lambdas.shape == (2,)
    assert res.fitted.shape == (40, 1)


def test_fit_duchon_with_by_row_gating():
    t, y = _inputs()
    g = torch.Generator().manual_seed(7)
    by = torch.rand(40, generator=g, dtype=torch.float64)
    res = gt.fit(t, y, gt.Duchon(centers=_centers(), m=2, by=by))

    assert res.coefficients.shape == (8, 1)
    assert res.fitted.shape == (40, 1)


def test_fit_bspline_single():
    t, y = _inputs()
    knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    res = gt.fit(t, y, gt.BSpline(knots=knots, degree=3))

    assert res.coefficients.dim() == 2
    assert res.coefficients.shape[1] == 1
    assert res.fitted.shape == (40, 1)


def test_fit_sphere_single():
    n = 20
    g = torch.Generator().manual_seed(0)
    lat = torch.rand(n, generator=g, dtype=torch.float64) * 60.0 - 30.0
    lon = torch.rand(n, generator=g, dtype=torch.float64) * 360.0 - 180.0
    y = torch.sin(lat * 0.05)
    res = gt.fit(torch.stack([lat, lon], dim=1), y, gt.Sphere(n_centers=10))

    assert res.coefficients.dim() == 2
    assert res.coefficients.shape[1] == 1
    assert res.fitted.shape == (n, 1)


def test_fit_periodic_spline_curve_single():
    t, y = _inputs(n=20)
    res = gt.fit(t, y, gt.PeriodicSplineCurve(n_knots=10, degree=3))

    assert res.coefficients.shape == (10, 1)
    assert res.fitted.shape == (20, 1)


def test_matern_fit_raises_notimplemented():
    t, y = _inputs(n=20)
    with pytest.raises(NotImplementedError):
        gt.fit(t, y, gt.Matern(centers=_centers(6), nu=1.5, length_scale=1.0))


def test_tensorbspline_fit_raises_notimplemented():
    t, y = _inputs(n=20)
    knots = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    with pytest.raises(NotImplementedError):
        gt.fit(
            t,
            y,
            gt.TensorBSpline(
                marginals=[
                    gt.BSpline(knots=knots, degree=3),
                    gt.BSpline(knots=knots, degree=3),
                ],
            ),
        )


def test_categorical_fit_raises_notimplemented():
    n = 20
    t, y = _inputs(n=n)
    levels = torch.zeros(n, dtype=torch.int64)

    with pytest.raises(NotImplementedError):
        gt.fit(t, y, gt.Categorical(levels=levels, n_levels=3))


def test_gam_module_train_then_freeze_then_eval():
    t, y = _inputs()
    model = gt.GAM([gt.Duchon(centers=_centers(), m=2)])

    model.train()
    assert model(t, y).shape == (40, 1)

    with pytest.raises(ValueError):
        model(t)

    model.freeze(t, y)
    assert not model.training
    assert model(t).shape == (40, 1)
    assert [name for name, _ in model.named_parameters()] == []
