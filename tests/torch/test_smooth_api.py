"""Public smooth, fit, and GAM torch API smoke tests."""

import numpy as np
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
    # Each Duchon block leaves its global mean unpenalized, so two ungated
    # Duchon blocks share that direction and the joint coefficient map is not
    # identified. Per-row amplitude gates, the documented additive form, give
    # each block its own mean direction.
    t, y = _inputs()
    g = torch.Generator().manual_seed(11)
    gates = [torch.rand(40, generator=g, dtype=torch.float64) for _ in range(2)]
    with pytest.raises(ValueError, match="joint coefficient map is not identified"):
        gt.fit(
            [t, t],
            y,
            [gt.Duchon(centers=_centers(6), m=2), gt.Duchon(centers=_centers(7), m=2)],
        )
    res = gt.fit(
        [t, t],
        y,
        [
            gt.Duchon(centers=_centers(6), m=2, by=gates[0]),
            gt.Duchon(centers=_centers(7), m=2, by=gates[1]),
        ],
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


def test_fit_two_periodic_splines_uses_identified_chart_and_lifts_coefficients():
    n = 80
    t1 = torch.arange(n, dtype=torch.float64) / n
    t2 = torch.remainder(
        0.137 + 0.6180339887498948 * torch.arange(n, dtype=torch.float64),
        1.0,
    )
    y = torch.sin(2.0 * torch.pi * t1) + 0.6 * torch.cos(2.0 * torch.pi * t2)
    y = y - y.mean()
    smooths = [
        gt.PeriodicSplineCurve(n_knots=7, degree=3),
        gt.PeriodicSplineCurve(n_knots=8, degree=3),
    ]

    res = gt.fit([t1, t2], y, smooths, mode="joint")

    assert isinstance(res.coefficients, list)
    assert [tuple(coef.shape) for coef in res.coefficients] == [(7, 1), (8, 1)]
    assert res.fitted.shape == (n, 1)
    assert torch.isfinite(res.fitted).all()
    assert torch.isfinite(res.lambdas).all()

    # Public coefficients are lifted to the raw cyclic bases while retaining
    # the terms-layer weighted sum-to-zero gauge used by the fit.
    for t, smooth, coefficient in zip([t1, t2], smooths, res.coefficients):
        raw_design, _ = gt.periodic_spline_curve_basis(
            t, smooth.n_knots, degree=smooth.degree,
        )
        weighted_mean = raw_design.sum(dim=0) @ coefficient[:, 0]
        torch.testing.assert_close(
            weighted_mean,
            torch.zeros_like(weighted_mean),
            rtol=0.0,
            atol=1.0e-10,
        )


def test_matern_fit_refuses_until_the_block_backend_takes_a_penalty_list():
    """`gt.fit` refuses a Matern term, and that refusal is the #4492 contract.

    This used to assert a successful fit with one coefficient per centre and
    an R^2 floor. It passed because the torch path built its OWN penalty --
    the raw symmetrised covariance Gram `K_cc` on the raw kernel columns under
    a single lambda -- which is not the penalty `gamfit.fit` prices for the
    same spec. 047b45f06d routed the realization through the term builder, so
    the Matern term now arrives with the nu-gated collocation operator
    candidates, several of them, one smoothing parameter each.

    `gaussian_reml_fit_blocks_exact` prices `P = blockdiag(lambda_k S_k)`, one
    lambda per COEFFICIENT BLOCK. Several penalties on one block have no
    coordinate in that criterion, and summing them under a single lambda is
    exactly the divergence #4492 exists to remove, so the fit refuses (step 2
    of the issue is the block API taking a penalty list per block).

    The old assertions cannot be restored as they were: the realized design is
    the builder's, through the kernel identifiability chart, so its width is
    not the centre count either.

    WHEN THIS REFUSAL STOPS, this test fails, and that failure is the
    instruction to replace this body with the fit and the parity assertions in
    `tests/torch/test_torch_penalty_parity_with_rust_4492.py`, which already
    carry the bar the parity is stated in.
    """
    t, y = _inputs(n=20)
    centers = _centers(6)
    with pytest.raises(NotImplementedError) as refusal:
        gt.fit(t, y, gt.Matern(centers=centers, nu=1.5, length_scale=1.0))
    assert "penalt" in str(refusal.value).lower(), (
        "Matern refused, but not for the several-penalties reason this test "
        f"is about: {refusal.value}"
    )


def test_matern_fit_no_longer_carries_autograd_back_to_points():
    """The input-location VJP through a Matern design is WITHDRAWN (#4492).

    This used to assert a finite non-zero gradient of the fitted values with
    respect to `points`. That held only while the torch path built the design
    itself as a torch expression of the inputs. 047b45f06d takes the design
    from the term builder, because a penalty is meaningful only in the chart
    its design is expressed in, and that chart -- the joint-null rotation, the
    term's identifiability transform, any unabsorbed global orthogonality, and
    the affine parametric residualization -- is not a right-multiplication the
    torch side could apply without reimplementing the thing the entry exists
    to stop duplicating.

    So there is no autograd path back to `points` for an engine-routed term,
    by construction and not by accident. Today the fit refuses before the
    question arises, and the refusal is what is asserted.

    WHEN THE REFUSAL STOPS, this test fails. The replacement is NOT the old
    assertion: it is that `torch.autograd.grad(res.fitted.sum(), t)` raises,
    or returns a gradient that is None, because `points` is not in the graph.
    Restoring the old assertion would mean restoring a torch-built design, and
    with it the divergence.
    """
    t, _y = _inputs(n=20)
    t = t.clone().requires_grad_(True)
    y = torch.sin(3.0 * t.detach())
    with pytest.raises(NotImplementedError) as refusal:
        gt.fit(t, y, gt.Matern(centers=_centers(6), nu=1.5, length_scale=1.0))
    assert "penalt" in str(refusal.value).lower(), (
        "Matern refused, but not for the several-penalties reason this test "
        f"is about: {refusal.value}"
    )


def _tensor_bspline_inputs(n=200, seed=1):
    """2D (x, z) grid-ish points and a separable interaction target."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=g, dtype=torch.float64)
    z = torch.rand(n, generator=g, dtype=torch.float64)
    points = torch.stack([x, z], dim=1)
    y = torch.sin(3.0 * x) * torch.cos(2.0 * z) + 0.02 * torch.randn(
        n, generator=g, dtype=torch.float64,
    )
    return points, y


def test_tensorbspline_fit_te_2d_refuses_one_lambda_per_margin():
    """`gt.fit` refuses a te term, one realized penalty per margin (#4492).

    This used to assert a successful fit whose coefficient count is the
    product of the two marginal column counts. It passed because the torch
    path built its own tensor penalty, summing `I (x) S_a (x) I` over the
    margins under ONE lambda. The term builder emits one candidate per margin,
    each measured by its neighbours' FUNCTION Grams rather than by their
    coefficients, and each normalized first -- a different model, with a
    different number of smoothing parameters.

    Two margins therefore realize two penalties on one coefficient block, and
    the block criterion prices one lambda per block, so the fit refuses.

    The old coefficient-count assertion could not be restored either: the
    realized design is the builder's identifiable block, not the raw
    Khatri-Rao product.

    DEPENDS ON `fix/4492-2`. Without it the term this fit lowers is a bare
    `te(x0, x1)`, whose margins are natural cubic regression splines, and the
    explicit knot VECTOR below is refused first, with "an explicit knot vector
    cannot replace the value knots of a natural cubic regression spline" --
    a different refusal, raised as a different exception type. If this test
    fails with that message, 4492-2 has not landed and this file is not what
    is wrong.

    WHEN THE REFUSAL STOPS, replace this body with the fit and the parity
    assertions in `test_torch_penalty_parity_with_rust_4492.py`.
    """
    points, y = _tensor_bspline_inputs()
    # An explicit knot tensor is the FULL knot vector, so a cubic basis needs
    # its boundary knots repeated to cover the data on [0, 1]. The bare grid
    # linspace(0, 1, 8) spans only [t_3, t_4] = [3/7, 4/7]; the rows outside it
    # fall on the linear boundary continuation and the design loses rank.
    knots = torch.cat(
        [
            torch.zeros(3, dtype=torch.float64),
            torch.linspace(0.0, 1.0, 8, dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
        ]
    )
    with pytest.raises(NotImplementedError) as refusal:
        gt.fit(
            points,
            y,
            gt.TensorBSpline(
                marginals=[
                    gt.BSpline(knots=knots, degree=3),
                    gt.BSpline(knots=knots, degree=3),
                ],
            ),
        )
    assert "penalt" in str(refusal.value).lower(), (
        "te refused, but not for the several-penalties reason this test is "
        f"about: {refusal.value}"
    )


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
    # The contrast is sum-to-zero and carries no constant
    # (`gamfit.smooth.Categorical`), so it represents the level effects about
    # the grand mean. Centre the response so its level means are such effects.
    y = y - y.mean()

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


def test_categorical_fit_is_invariant_to_level_relabeling():
    # Level codes are labels: relabeling them permutes the level effects and
    # must leave the fit unchanged. The ridge therefore prices the level
    # effects e = C·c the drop-last design produces, not the contrast
    # coordinates c. An identity ridge on c gives the last-coded level K - 1
    # times the prior variance of the others, so the fit moved with the coding.
    n = 12
    n_levels = 3
    g = torch.Generator().manual_seed(3)
    levels = torch.arange(n, dtype=torch.int64) % n_levels
    t = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    effects = torch.tensor([0.6, -0.4, -0.2], dtype=torch.float64)
    y = effects[levels] + torch.randn(n, generator=g, dtype=torch.float64)
    y = y - y.mean()

    ref = gt.fit(t, y, gt.Categorical(levels=levels, n_levels=n_levels))
    for relabel in ([2, 0, 1], [1, 2, 0]):
        codes = torch.tensor(relabel, dtype=torch.int64)[levels]
        got = gt.fit(t, y, gt.Categorical(levels=codes, n_levels=n_levels))
        torch.testing.assert_close(got.fitted, ref.fitted)
        torch.testing.assert_close(got.lambdas, ref.lambdas)
        torch.testing.assert_close(got.reml_score, ref.reml_score)


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


def test_fit_and_frozen_forward_split_a_non_list_points_sequence_alike():
    # gam#3117: fit() used to copy any non-list/tuple sequence to every smooth
    # while the frozen forward split it per smooth.
    import collections

    # The identified two-periodic fixture above, on two distinct inputs, so a
    # copied or reordered split changes the fit.
    n = 80
    t1 = torch.arange(n, dtype=torch.float64) / n
    t2 = torch.remainder(
        0.137 + 0.6180339887498948 * torch.arange(n, dtype=torch.float64),
        1.0,
    )
    y = torch.sin(2.0 * torch.pi * t1) + 0.6 * torch.cos(2.0 * torch.pi * t2)
    y = y - y.mean()
    smooths = [
        gt.PeriodicSplineCurve(n_knots=7, degree=3),
        gt.PeriodicSplineCurve(n_knots=8, degree=3),
    ]
    ref = gt.fit([t1, t2], y, smooths)
    res = gt.fit(collections.deque([t1, t2]), y, smooths)
    for a, b in zip(res.coefficients, ref.coefficients, strict=True):
        torch.testing.assert_close(a, b)
    model = gt.GAM(smooths)
    model.freeze(collections.deque([t1, t2]), y)
    torch.testing.assert_close(model(collections.deque([t1, t2])), model([t1, t2]))


@pytest.mark.parametrize("bad", ["ndarray", "entry"])
def test_fit_and_frozen_forward_refuse_non_tensor_points_alike(bad):
    t, y = _inputs()
    smooths = [gt.Duchon(centers=_centers(6), m=2), gt.Duchon(centers=_centers(7), m=2)]
    pts = np.stack([t.numpy(), t.numpy()]) if bad == "ndarray" else [t, t.numpy()]
    with pytest.raises(TypeError, match="torch.Tensor"):
        gt.fit(pts, y, smooths)
    model = gt.GAM(smooths)
    model._install_frozen_coefficients([torch.zeros(6, 1), torch.zeros(7, 1)])
    model.eval()
    with pytest.raises(TypeError, match="torch.Tensor"):
        model(pts)


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
