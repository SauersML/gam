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


def test_matern_fit_single_1d():
    """Matern tensor backend is wired (#1105): kernel design vs centers, RKHS
    covariance-Gram penalty. The fit must produce finite coefficients and a
    fitted vector matching the response shape."""
    t, y = _inputs(n=20)
    centers = _centers(6)
    res = gt.fit(t, y, gt.Matern(centers=centers, nu=1.5, length_scale=1.0))

    # One coefficient per center, single output column.
    assert res.coefficients.shape == (6, 1)
    assert res.fitted.shape == (20, 1)
    assert torch.isfinite(res.coefficients).all()
    assert torch.isfinite(res.fitted).all()
    # The kernel ridge must actually track the smooth target, not collapse.
    y2d = y.unsqueeze(1)
    ss_res = ((res.fitted - y2d) ** 2).sum()
    ss_tot = ((y2d - y2d.mean()) ** 2).sum()
    assert float(ss_res / ss_tot) < 0.5


def test_matern_fit_autograd_flows_to_points():
    """The Matern design carries the input-location VJP back to ``points``
    (the scalar-kernel autograd path), so a scalar of the fitted values has a
    finite, non-zero gradient w.r.t. the input coordinates."""
    t, _y = _inputs(n=20)
    t = t.clone().requires_grad_(True)
    y = torch.sin(3.0 * t.detach())
    res = gt.fit(t, y, gt.Matern(centers=_centers(6), nu=1.5, length_scale=1.0))
    loss = res.fitted.sum()
    (grad,) = torch.autograd.grad(loss, t)
    assert grad.shape == t.shape
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def _tensor_bspline_inputs(n=24, seed=1):
    """2D (x, z) grid-ish points and a separable interaction target."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=g, dtype=torch.float64)
    z = torch.rand(n, generator=g, dtype=torch.float64)
    points = torch.stack([x, z], dim=1)
    y = torch.sin(3.0 * x) * torch.cos(2.0 * z) + 0.02 * torch.randn(
        n, generator=g, dtype=torch.float64,
    )
    return points, y


def test_tensorbspline_fit_te_2d():
    """TensorBSpline (te) tensor backend is wired (#1105): Khatri-Rao design
    over both marginal B-spline bases, Kronecker-sum tensor penalty. The fit
    must recover the interaction target on a 2D (x, z) input."""
    points, y = _tensor_bspline_inputs()
    knots = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    res = gt.fit(
        points,
        y,
        gt.TensorBSpline(
            marginals=[
                gt.BSpline(knots=knots, degree=3),
                gt.BSpline(knots=knots, degree=3),
            ],
        ),
    )

    # Coefficient space is the tensor product of the two marginal bases: its
    # size is the product of each marginal B-spline's column count. Derive the
    # marginal column count from the basis itself (avoids hardcoding the
    # open-knot convention) and assert the Kronecker dimension.
    from gamfit.torch._basis import bspline_basis as _bspline_basis

    marg_cols = _bspline_basis(
        points[:, 0], knots, degree=3, periodic=False,
    ).shape[1]
    assert marg_cols > 1
    assert res.coefficients.dim() == 2
    assert res.coefficients.shape[1] == 1
    assert res.coefficients.shape[0] == marg_cols * marg_cols
    assert res.fitted.shape == (points.shape[0], 1)
    assert torch.isfinite(res.coefficients).all()
    # The interaction surface must actually be tracked (additive s(x)+s(z)
    # cannot represent sin(x)cos(z); the te must beat the variance floor).
    y2d = y.unsqueeze(1)
    ss_res = ((res.fitted - y2d) ** 2).sum()
    ss_tot = ((y2d - y2d.mean()) ** 2).sum()
    assert float(ss_res / ss_tot) < 0.6


def test_tensorbspline_dim_mismatch_rejected():
    """A 2-marginal TensorBSpline against 1D points is a shape error, not a
    silent broadcast."""
    t, y = _inputs(n=20)
    knots = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    with pytest.raises(ValueError):
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


def test_categorical_fit_wired_backend_shapes_and_recovery():
    # Regression for the torch `Categorical` backend seam (#1133): the
    # sum-to-zero categorical contrast must build a real (design, penalty)
    # pair and fit, instead of raising NotImplementedError on the unwired
    # branch. The drop-last sum-to-zero coding gives n_levels-1 contrast
    # coefficients; the fitted per-group means must recover the data means
    # up to ridge shrinkage.
    n = 60
    n_levels = 3
    g = torch.Generator().manual_seed(7)
    # Round-robin level codes so every level is well populated.
    levels = torch.arange(n, dtype=torch.int64) % n_levels
    t = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    # Distinct, well-separated per-level means + small noise.
    group_means = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    y = group_means[levels] + 0.01 * torch.randn(
        n, generator=g, dtype=torch.float64,
    )

    result = gt.fit(t, y, gt.Categorical(levels=levels, n_levels=n_levels))

    # n_levels-1 contrast coefficients, single output column.
    assert tuple(result.coefficients.shape) == (n_levels - 1, 1)
    assert tuple(result.fitted.shape) == (n, 1)
    assert torch.isfinite(result.fitted).all()

    # Each row's fitted value should track its group; with light ridge
    # shrinkage the fitted group means stay near the data group means.
    fitted = result.fitted.reshape(-1)
    for k in range(n_levels):
        mask = levels == k
        fitted_k = fitted[mask].mean()
        data_k = y[mask].mean()
        assert torch.abs(fitted_k - data_k) < 0.5


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


def test_gam_frozen_coefficients_are_persistent_migrating_buffers(monkeypatch):
    import importlib

    gam_module = importlib.import_module("gamfit.torch.module")
    source = torch.arange(6, dtype=torch.float64).reshape(6, 1)

    class FakeFitResult:
        coefficients = [source]

    monkeypatch.setattr(gam_module, "fit", lambda *_args, **_kwargs: FakeFitResult())
    model = gt.GAM([gt.Duchon(centers=_centers(6), m=2)])
    model.freeze(torch.zeros(4), torch.zeros(4))

    # freeze() is a snapshot, not an alias into the returned FitResult.
    source.fill_(-1.0)
    frozen = model._frozen_coefficients()
    assert frozen is not None
    assert torch.equal(frozen[0], torch.arange(6, dtype=torch.float64).reshape(6, 1))

    state = model.state_dict()
    assert list(state) == ["_frozen_coefficient_0"]

    restored = gt.GAM([gt.Duchon(centers=_centers(6), m=2)])
    restored.load_state_dict(state)
    restored_frozen = restored._frozen_coefficients()
    assert restored_frozen is not None
    assert torch.equal(restored_frozen[0], frozen[0])

    restored.to(dtype=torch.float32)
    migrated = restored._frozen_coefficients()
    assert migrated is not None
    assert migrated[0].dtype == torch.float32

    # The meta device exercises nn.Module's device migration machinery on CPU CI
    # without requiring CUDA hardware.
    restored.to(device="meta")
    on_meta = restored._frozen_coefficients()
    assert on_meta is not None
    assert on_meta[0].device.type == "meta"


@pytest.mark.parametrize("block_count", [1, 3])
def test_gam_frozen_eval_rejects_points_block_count_mismatch(block_count):
    model = gt.GAM([
        gt.Duchon(centers=_centers(6), m=2),
        gt.Duchon(centers=_centers(7), m=2),
    ])
    model._install_frozen_coefficients([torch.zeros(6, 1), torch.zeros(7, 1)])
    model.eval()

    with pytest.raises(
        ValueError,
        match=rf"{block_count} points tensors for 2 smooths",
    ):
        model([torch.zeros(4)] * block_count)


@pytest.mark.parametrize("block_count", [1, 3])
def test_gam_frozen_eval_rejects_coefficient_block_count_mismatch(block_count):
    model = gt.GAM([
        gt.Duchon(centers=_centers(6), m=2),
        gt.Duchon(centers=_centers(7), m=2),
    ])
    with pytest.raises(
        RuntimeError,
        match=rf"{block_count} coefficient blocks for 2 smooths",
    ):
        model._install_frozen_coefficients([torch.zeros(6, 1)] * block_count)


@pytest.mark.parametrize("block_count", [1, 3])
def test_gam_freeze_rejects_fit_coefficient_block_count_mismatch(
    monkeypatch, block_count,
):
    import importlib

    gam_module = importlib.import_module("gamfit.torch.module")
    model = gt.GAM([
        gt.Duchon(centers=_centers(6), m=2),
        gt.Duchon(centers=_centers(7), m=2),
    ])

    class BadFitResult:
        coefficients = [torch.zeros(6, 1)] * block_count

    monkeypatch.setattr(gam_module, "fit", lambda *_args, **_kwargs: BadFitResult())

    with pytest.raises(
        RuntimeError,
        match=rf"{block_count} coefficient blocks for 2 smooths",
    ):
        model.freeze(torch.zeros(4), torch.zeros(4))

    assert model._frozen_coefficients() is None
    assert model.last_fit is None
    assert model.training
