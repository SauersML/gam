"""``Model.partial_dependence`` intervals, bands, factor and surface effects, and
``Model.plot_terms`` (pyGAM audit: inference G2, docs DOC-2..5 and DOC-21, tests F4).

- Coverage: on a fixed-seed simulation of ``y = sin(2πx) + ε`` the pointwise
  intervals cover the centred truth at about ``level`` of the grid points on
  average, and the simultaneous band covers the whole curve in about ``level``
  of the replicates. The pointwise interval covers the whole curve far less
  often, which is the reason the band exists.
- A factor term gives one effect per level, labelled, with fit and se equal to
  the term's design block times the coefficients and their covariance.
- A ``te(x, z)`` term gives a product grid that ``surface()`` reshapes.
- ``plot_terms`` draws every kind of term on the Agg backend, and
  ``model.plot(kind="prediction")`` refuses multi-feature data.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gamfit

LEVEL = 0.95


def _beta_and_cov(model):
    summary = model.summary()
    beta = np.asarray([c["estimate"] for c in summary.coefficients], dtype=float)
    cov = np.asarray(summary.covariance_flat, dtype=float).reshape(
        summary.covariance_n, summary.covariance_n
    )
    return beta, cov


def _block_fit_and_se(model, term, frame):
    block = next(b for b in model.term_blocks if b.name == term)
    design = np.asarray(model.design_matrix(frame).matrix, dtype=float)
    columns = design[:, block.start : block.end]
    beta, cov = _beta_and_cov(model)
    sub = cov[block.start : block.end, block.start : block.end]
    fit = columns @ beta[block.start : block.end]
    se = np.sqrt(np.einsum("ij,jk,ik->i", columns, sub, columns))
    return fit, se


def _assert_band_ordering(effect):
    assert np.all(effect.se > 0.0)
    assert np.all(effect.simultaneous_lower < effect.lower)
    assert np.all(effect.lower < effect.fit)
    assert np.all(effect.fit < effect.upper)
    assert np.all(effect.upper < effect.simultaneous_upper)
    np.testing.assert_allclose(
        effect.upper - effect.fit, effect.pointwise_critical * effect.se, rtol=1e-12
    )
    np.testing.assert_allclose(
        effect.simultaneous_upper - effect.fit,
        effect.simultaneous_critical * effect.se,
        rtol=1e-12,
    )


def test_pointwise_and_simultaneous_coverage_on_a_fixed_seed_simulation():
    n, replicates, sigma = 200, 200, 0.3
    x = (np.arange(n) + 0.5) / n
    truth = np.sin(2 * np.pi * x)
    grid = np.linspace(x.min(), x.max(), 50)
    rng = np.random.default_rng(20260919)

    pointwise_rates, simultaneous_hits, pointwise_whole_hits = [], 0, 0
    for _ in range(replicates):
        frame = pd.DataFrame({"x": x, "y": truth + sigma * rng.standard_normal(n)})
        model = gamfit.fit(frame, "y ~ s(x)")
        if not pointwise_rates:
            # The term is centred over the training rows, so its truth is
            # sin(2πx) minus that function's training mean.
            at_training = model.partial_dependence("s(x)", grid=x)
            assert abs(at_training.fit.mean()) < 1e-10 * np.abs(at_training.fit).max()
        effect = model.partial_dependence("s(x)", grid=grid, level=LEVEL)
        centred = np.sin(2 * np.pi * grid) - truth.mean()
        inside = (effect.lower <= centred) & (centred <= effect.upper)
        pointwise_rates.append(inside.mean())
        pointwise_whole_hits += bool(inside.all())
        simultaneous_hits += bool(
            np.all((effect.simultaneous_lower <= centred) & (centred <= effect.simultaneous_upper))
        )

    pointwise = float(np.mean(pointwise_rates))
    simultaneous = simultaneous_hits / replicates
    whole_by_pointwise = pointwise_whole_hits / replicates
    tolerance = 3 * math.sqrt(LEVEL * (1 - LEVEL) / replicates)
    print(
        f"pointwise average coverage {pointwise:.4f}, simultaneous whole-curve coverage "
        f"{simultaneous:.4f}, pointwise whole-curve coverage {whole_by_pointwise:.4f}"
    )
    assert abs(pointwise - LEVEL) < tolerance
    assert simultaneous > LEVEL - tolerance
    assert whole_by_pointwise < simultaneous


def test_band_draw_count_and_seed_are_fixed_by_the_level():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, 150)
    frame = pd.DataFrame({"x": x, "y": np.sin(2 * np.pi * x) + 0.2 * rng.standard_normal(150)})
    model = gamfit.fit(frame, "y ~ s(x)")

    at_95 = model.partial_dependence("s(x)", n_points=30)
    again = model.partial_dependence("s(x)", n_points=30)
    at_99 = model.partial_dependence("s(x)", n_points=30, level=0.99)
    assert (at_95.simulations, at_95.seed, at_95.level) == (7600, 12345, 0.95)
    assert at_99.simulations == 39600
    np.testing.assert_array_equal(at_95.simultaneous_lower, again.simultaneous_lower)
    assert at_95.pointwise_critical == pytest.approx(1.959963984540054, rel=1e-12)
    assert at_99.pointwise_critical == pytest.approx(2.5758293035489004, rel=1e-12)
    assert at_99.simultaneous_critical > at_95.simultaneous_critical
    assert at_95.covariance_source in ("smoothing-corrected", "conditional")
    _assert_band_ordering(at_95)
    _assert_band_ordering(at_99)
    with pytest.raises(ValueError):
        model.partial_dependence("s(x)", level=1.0)


def _factor_frame(seed=5, n=240):
    rng = np.random.default_rng(seed)
    effects = {"a": 0.0, "b": 1.0, "c": -0.5}
    g = np.array(["a", "b", "c"])[np.arange(n) % 3]
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x) + np.array([effects[v] for v in g]) + 0.2 * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "g": g, "y": y}), effects


def test_a_factor_term_gives_one_labelled_effect_per_level():
    frame, effects = _factor_frame()
    model = gamfit.fit(frame, "y ~ s(x) + g")
    effect = model.partial_dependence("g")

    assert effect.axes == ("g",)
    assert effect.axis_values is not None
    labels = effect.labels()
    assert sorted(labels) == ["a", "b", "c"]
    assert len(effect.fit) == 3

    at_levels = pd.DataFrame({"x": np.full(3, 0.5), "g": labels})
    fit, se = _block_fit_and_se(model, "g", at_levels)
    np.testing.assert_allclose(effect.fit, fit, atol=1e-12)
    np.testing.assert_allclose(effect.se, se, atol=1e-12)

    # A factor main effect is a penalized block with one column per level, so
    # only its contrasts are identified; they recover the simulated shifts.
    by_label = dict(zip(labels, effect.fit))
    for level in ("b", "c"):
        contrast = by_label[level] - by_label["a"]
        assert contrast == pytest.approx(effects[level] - effects["a"], abs=0.15)
    assert np.all(effect.simultaneous_upper - effect.simultaneous_lower > effect.upper - effect.lower)
    _assert_band_ordering(effect)


def test_a_tensor_term_gives_a_surface_on_its_product_grid():
    rng = np.random.default_rng(9)
    n = 400
    frame = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n), "z": rng.uniform(-1.0, 1.0, n)})
    frame["y"] = (
        np.sin(2 * np.pi * frame["x"]) * frame["z"] + 0.2 * rng.standard_normal(n)
    )
    model = gamfit.fit(frame, "y ~ te(x, z)")
    effect = model.partial_dependence("te(x, z)", n_points=12)

    assert effect.axes == ("x", "z")
    xs, zs = effect.axis_values
    assert xs.shape == (12,) and zs.shape == (12,)
    assert xs[0] == pytest.approx(frame["x"].min()) and xs[-1] == pytest.approx(frame["x"].max())
    assert zs[0] == pytest.approx(frame["z"].min()) and zs[-1] == pytest.approx(frame["z"].max())
    for name in ("fit", "se", "lower", "upper", "simultaneous_lower", "simultaneous_upper"):
        assert effect.surface(name).shape == (12, 12)
    # The last axis varies fastest.
    mesh_x, mesh_z = np.meshgrid(xs, zs, indexing="ij")
    np.testing.assert_array_equal(effect.grid[:, 0], mesh_x.ravel())
    np.testing.assert_array_equal(effect.grid[:, 1], mesh_z.ravel())

    fit, se = _block_fit_and_se(
        model, "te(x, z)", pd.DataFrame({"x": effect.grid[:, 0], "z": effect.grid[:, 1]})
    )
    np.testing.assert_allclose(effect.fit, fit, atol=1e-10)
    np.testing.assert_allclose(effect.se, se, atol=1e-10)
    np.testing.assert_allclose(effect.surface("fit")[3, 7], effect.fit[3 * 12 + 7])
    _assert_band_ordering(effect)

    own = model.partial_dependence("te(x, z)", grid=effect.grid[:5])
    assert own.axis_values is None
    with pytest.raises(ValueError, match="default product grid"):
        own.surface("fit")
    with pytest.raises(ValueError, match="unknown series"):
        effect.surface("estimate")
    with pytest.raises(ValueError):
        _ = effect.x


def test_plot_terms_draws_every_kind_of_term():
    frame, _ = _factor_frame(seed=13, n=300)
    other = np.random.default_rng(14)
    frame["z"] = other.uniform(-1.0, 1.0, len(frame))
    frame["w"] = other.uniform(0.0, 1.0, len(frame))
    model = gamfit.fit(frame, "y ~ s(x) + g + te(z, w)")

    axes = model.plot_terms(n_points=15)
    names = [b.name for b in model.term_blocks if b.kind != "intercept"]
    assert sorted(names) == ["g", "s(x)", "te(z, w)"]
    assert [ax.get_title() for ax in axes] == names
    by_title = {ax.get_title(): ax for ax in axes}
    curve, levels, surface = by_title["s(x)"], by_title["g"], by_title["te(z, w)"]
    assert len(curve.lines) >= 1 and len(curve.collections) >= 2
    assert [tick.get_text() for tick in levels.get_xticklabels()] == model.partial_dependence(
        "g"
    ).labels()
    assert len(surface.collections) >= 1
    plt.close("all")

    figure, own = plt.subplots(1, 1)
    drawn = model.plot_terms("s(x)", axes=[own], level=0.9)
    assert drawn == [own]
    plt.close(figure)

    with pytest.raises(ValueError):
        model.plot_terms(["s(x)", "g"], axes=[own])


def test_plot_terms_draws_a_factor_by_smooth_as_one_curve_per_level():
    frame, _ = _factor_frame(seed=17, n=300)
    model = gamfit.fit(frame, "y ~ g + s(x, by=g)")
    by_terms = [b.name for b in model.term_blocks if "by=" in b.name]
    axes = model.plot_terms(by_terms, n_points=15)
    assert len(axes) == len(by_terms)
    plt.close("all")


def test_prediction_plot_refuses_multi_feature_data():
    frame, _ = _factor_frame(seed=19)
    model = gamfit.fit(frame, "y ~ s(x) + g")
    with pytest.raises(ValueError, match="plot_terms"):
        model.plot(frame, kind="prediction")
    plt.close("all")
