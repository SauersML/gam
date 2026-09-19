"""Hostile-input cases shared by the gamfit and pyGAM runners.

Each case returns a dict:
  data:     dict of numpy columns (gamfit input)
  formula:  gamfit formula
  family:   gamfit family (default "auto")
  weights:  optional weights column name
  pyg:      (GAMClassName, terms_expr, X, y, weights or None)
  pred:     dict of numpy columns for prediction (gamfit); pyg X for predict = pred_X
  truth:    optional true mean at pred points
"""
import numpy as np


def _rng(seed=0):
    return np.random.default_rng(seed)


def base(n=200, seed=0):
    r = _rng(seed)
    x = r.uniform(0, 1, n)
    y = np.sin(6 * x) + r.normal(0, 0.3, n)
    return x, y


def c_tiny_n5():
    r = _rng(1)
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y = np.sin(6 * x) + r.normal(0, 0.1, 5)
    px = np.linspace(0, 1, 5)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": px}, truth=np.sin(6 * px))


def c_tiny_n5_k20():
    d = c_tiny_n5()
    d["formula"] = "y ~ s(x, k=20)"
    return d


def c_tiny_n12_k20():
    r = _rng(2)
    x = np.linspace(0, 1, 12)
    y = np.sin(6 * x) + r.normal(0, 0.1, 12)
    px = np.linspace(0, 1, 5)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x, k=20)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": px}, truth=np.sin(6 * px))


def c_n1():
    x = np.array([0.5]); y = np.array([1.0])
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None), pred={"x": x})


def c_duplicated_x():
    r = _rng(3)
    xu = np.linspace(0, 1, 10)
    x = np.repeat(xu, 30)
    y = np.sin(6 * x) + r.normal(0, 0.3, x.size)
    px = np.linspace(0, 1, 7)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": px}, truth=np.sin(6 * px))


def c_constant_covariate():
    x, y = base()
    z = np.full_like(x, 3.0)
    return dict(data={"x": x, "z": z, "y": y}, formula="y ~ s(x) + s(z)",
                pyg=("LinearGAM", "s(0) + s(1)", np.c_[x, z], y, None),
                pred={"x": np.array([0.2, 0.5]), "z": np.array([3.0, 3.0])},
                truth=np.sin(6 * np.array([0.2, 0.5])))


def c_constant_only():
    x, y = base()
    z = np.full_like(x, 3.0)
    return dict(data={"z": z, "y": y}, formula="y ~ s(z)",
                pyg=("LinearGAM", "s(0)", z[:, None], y, None),
                pred={"z": np.array([3.0])})


def c_two_unique():
    r = _rng(4)
    x = r.integers(0, 2, 300).astype(float)
    y = 2 * x + r.normal(0, 0.5, 300)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": np.array([0.0, 1.0, 0.5])}, truth=np.array([0, 2, 1.0]))


def c_three_unique():
    r = _rng(5)
    x = r.integers(0, 3, 300).astype(float)
    y = (x - 1) ** 2 + r.normal(0, 0.5, 300)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": np.array([0.0, 1.0, 2.0])}, truth=np.array([1, 0, 1.0]))


def c_scale_1e9():
    x, y = base()
    X = 1e9 + x * 1e9
    px = 1e9 + np.linspace(0, 1, 5) * 1e9
    return dict(data={"x": X, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", X[:, None], y, None),
                pred={"x": px}, truth=np.sin(6 * np.linspace(0, 1, 5)))


def c_offset_1e9_small_range():
    # large location, tiny spread: x in [1e9, 1e9+1]
    x, y = base()
    X = 1e9 + x
    px = 1e9 + np.linspace(0, 1, 5)
    return dict(data={"x": X, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", X[:, None], y, None),
                pred={"x": px}, truth=np.sin(6 * np.linspace(0, 1, 5)))


def c_scale_1e_9():
    x, y = base()
    X = x * 1e-9
    px = np.linspace(0, 1, 5) * 1e-9
    return dict(data={"x": X, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", X[:, None], y, None),
                pred={"x": px}, truth=np.sin(6 * np.linspace(0, 1, 5)))


def c_y_scale_1e12():
    x, y = base()
    Y = y * 1e12
    px = np.linspace(0, 1, 5)
    return dict(data={"x": x, "y": Y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], Y, None),
                pred={"x": px}, truth=1e12 * np.sin(6 * px))


def c_y_scale_1e_12():
    x, y = base()
    Y = y * 1e-12
    px = np.linspace(0, 1, 5)
    return dict(data={"x": x, "y": Y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], Y, None),
                pred={"x": px}, truth=1e-12 * np.sin(6 * px))


def c_heavy_tail_y():
    r = _rng(6)
    x = r.uniform(0, 1, 400)
    y = np.sin(6 * x) + 0.3 * r.standard_cauchy(400)
    px = np.linspace(0.05, 0.95, 5)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": px}, truth=np.sin(6 * px))


def c_outliers():
    x, y = base(300)
    y = y.copy(); y[:3] = 1e6
    px = np.linspace(0.05, 0.95, 5)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": px}, truth=np.sin(6 * px))


def c_nan_x():
    x, y = base(); x = x.copy(); x[5] = np.nan
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None), pred={"x": np.array([0.5])})


def c_inf_x():
    x, y = base(); x = x.copy(); x[5] = np.inf
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None), pred={"x": np.array([0.5])})


def c_nan_y():
    x, y = base(); y = y.copy(); y[5] = np.nan
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None), pred={"x": np.array([0.5])})


def c_inf_y():
    x, y = base(); y = y.copy(); y[5] = np.inf
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None), pred={"x": np.array([0.5])})


def c_nan_at_predict():
    x, y = base()
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": np.array([0.5, np.nan, np.inf])})


def c_logit_perfect_sep():
    r = _rng(7)
    x = r.uniform(0, 1, 200)
    y = (x > 0.5).astype(float)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)", family="binomial",
                pyg=("LogisticGAM", "s(0)", x[:, None], y, None),
                pred={"x": np.array([0.1, 0.49, 0.51, 0.9])})


def c_logit_all_ones():
    r = _rng(8)
    x = r.uniform(0, 1, 200)
    y = np.ones(200)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)", family="binomial",
                pyg=("LogisticGAM", "s(0)", x[:, None], y, None),
                pred={"x": np.array([0.1, 0.9])})


def c_logit_linear_sep_param():
    r = _rng(9)
    x = r.uniform(-1, 1, 200)
    y = (x > 0).astype(float)
    return dict(data={"x": x, "y": y}, formula="y ~ x", family="binomial",
                pyg=("LogisticGAM", "l(0)", x[:, None], y, None),
                pred={"x": np.array([-0.5, 0.5])})


def c_poisson_all_zero():
    r = _rng(10)
    x = r.uniform(0, 1, 200)
    y = np.zeros(200)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)", family="poisson",
                pyg=("PoissonGAM", "s(0)", x[:, None], y, None),
                pred={"x": np.array([0.1, 0.9])})


def c_poisson_negative_y():
    r = _rng(11)
    x = r.uniform(0, 1, 200)
    y = r.poisson(3, 200).astype(float); y[0] = -1
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)", family="poisson",
                pyg=("PoissonGAM", "s(0)", x[:, None], y, None), pred={"x": np.array([0.5])})


def c_poisson_fractional_y():
    r = _rng(12)
    x = r.uniform(0, 1, 200)
    y = r.poisson(3, 200).astype(float) + 0.5
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)", family="poisson",
                pyg=("PoissonGAM", "s(0)", x[:, None], y, None), pred={"x": np.array([0.5])})


def c_binomial_y_2():
    r = _rng(13)
    x = r.uniform(0, 1, 200)
    y = r.integers(0, 2, 200).astype(float); y[0] = 2
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)", family="binomial",
                pyg=("LogisticGAM", "s(0)", x[:, None], y, None), pred={"x": np.array([0.5])})


def c_gamma_zero_y():
    r = _rng(14)
    x = r.uniform(0, 1, 200)
    y = r.gamma(2, 1, 200); y[0] = 0.0
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)", family="gamma",
                pyg=("GammaGAM", "s(0)", x[:, None], y, None), pred={"x": np.array([0.5])})


def c_gamma_negative_y():
    r = _rng(15)
    x = r.uniform(0, 1, 200)
    y = r.gamma(2, 1, 200); y[0] = -1.0
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)", family="gamma",
                pyg=("GammaGAM", "s(0)", x[:, None], y, None), pred={"x": np.array([0.5])})


def c_collinear_smooths():
    r = _rng(16)
    x = r.uniform(0, 1, 300)
    x2 = x + r.normal(0, 1e-8, 300)
    y = np.sin(6 * x) + r.normal(0, 0.3, 300)
    px = np.linspace(0.05, 0.95, 5)
    return dict(data={"x": x, "x2": x2, "y": y}, formula="y ~ s(x) + s(x2)",
                pyg=("LinearGAM", "s(0) + s(1)", np.c_[x, x2], y, None),
                pred={"x": px, "x2": px}, truth=np.sin(6 * px))


def c_exact_duplicate_smooths():
    r = _rng(17)
    x = r.uniform(0, 1, 300)
    y = np.sin(6 * x) + r.normal(0, 0.3, 300)
    px = np.linspace(0.05, 0.95, 5)
    return dict(data={"x": x, "x2": x.copy(), "y": y}, formula="y ~ s(x) + s(x2)",
                pyg=("LinearGAM", "s(0) + s(1)", np.c_[x, x], y, None),
                pred={"x": px, "x2": px}, truth=np.sin(6 * px))


def c_many_levels():
    r = _rng(18)
    n = 3000
    L = 500
    g = r.integers(0, L, n)
    eff = r.normal(0, 1, L)
    x = r.uniform(0, 1, n)
    y = np.sin(6 * x) + eff[g] + r.normal(0, 0.3, n)
    gs = np.array([f"L{i}" for i in g], dtype=object)
    return dict(data={"x": x, "g": gs, "y": y}, formula="y ~ s(x) + group(g)",
                pyg=("LinearGAM", "s(0) + f(1)", np.c_[x, g.astype(float)], y, None),
                pred={"x": np.array([0.5, 0.5]), "g": np.array(["L0", "L1"], dtype=object)},
                truth=np.sin(3.0) + eff[[0, 1]])


def c_unseen_level():
    r = _rng(19)
    n = 300
    g = r.integers(0, 3, n)
    x = r.uniform(0, 1, n)
    y = np.sin(6 * x) + g + r.normal(0, 0.3, n)
    gs = np.array([f"L{i}" for i in g], dtype=object)
    return dict(data={"x": x, "g": gs, "y": y}, formula="y ~ s(x) + g",
                pyg=("LinearGAM", "s(0) + f(1)", np.c_[x, g.astype(float)], y, None),
                pred={"x": np.array([0.5, 0.5]), "g": np.array(["L0", "LNEW"], dtype=object)},
                pred_X=np.array([[0.5, 0.0], [0.5, 7.0]]))


def c_unseen_level_group():
    d = c_unseen_level()
    d["formula"] = "y ~ s(x) + group(g)"
    return d


def c_extrapolate():
    x, y = base()
    px = np.array([-100.0, -1.0, 0.5, 2.0, 100.0])
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None), pred={"x": px})


def c_extrapolate_logit():
    r = _rng(20)
    x = r.uniform(0, 1, 300)
    y = (r.uniform(size=300) < 1 / (1 + np.exp(-(4 * x - 2)))).astype(float)
    px = np.array([-100.0, 0.5, 100.0])
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)", family="binomial",
                pyg=("LogisticGAM", "s(0)", x[:, None], y, None), pred={"x": px})


def c_zero_weights():
    x, y = base()
    w = np.ones_like(x); w[:50] = 0.0
    px = np.linspace(0.05, 0.95, 5)
    return dict(data={"x": x, "y": y, "w": w}, formula="y ~ s(x)", weights="w",
                pyg=("LinearGAM", "s(0)", x[:, None], y, w),
                pred={"x": px}, truth=np.sin(6 * px))


def c_all_zero_weights():
    x, y = base()
    w = np.zeros_like(x)
    return dict(data={"x": x, "y": y, "w": w}, formula="y ~ s(x)", weights="w",
                pyg=("LinearGAM", "s(0)", x[:, None], y, w), pred={"x": np.array([0.5])})


def c_negative_weights():
    x, y = base()
    w = np.ones_like(x); w[0] = -1.0
    return dict(data={"x": x, "y": y, "w": w}, formula="y ~ s(x)", weights="w",
                pyg=("LinearGAM", "s(0)", x[:, None], y, w), pred={"x": np.array([0.5])})


def c_nan_weights():
    x, y = base()
    w = np.ones_like(x); w[0] = np.nan
    return dict(data={"x": x, "y": y, "w": w}, formula="y ~ s(x)", weights="w",
                pyg=("LinearGAM", "s(0)", x[:, None], y, w), pred={"x": np.array([0.5])})


def c_float32():
    x, y = base()
    x = x.astype(np.float32); y = y.astype(np.float32)
    px = np.linspace(0.05, 0.95, 5).astype(np.float32)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": px}, truth=np.sin(6 * px.astype(float)))


def c_noncontig():
    r = _rng(21)
    M = r.uniform(0, 1, (400, 3))
    x = M[::2, 1]  # strided, non-contiguous
    y = (np.sin(6 * x) + r.normal(0, 0.3, x.size))
    Y = np.c_[y, y][:, 0]
    px = np.linspace(0.05, 0.95, 5)
    return dict(data={"x": x, "y": Y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", M[::2, 1:2], Y, None),
                pred={"x": px}, truth=np.sin(6 * px))


def c_huge_p_small_n():
    r = _rng(22)
    n, p = 60, 30
    X = r.uniform(0, 1, (n, p))
    y = np.sin(6 * X[:, 0]) + r.normal(0, 0.3, n)
    data = {f"x{j}": X[:, j] for j in range(p)}
    data["y"] = y
    formula = "y ~ " + " + ".join(f"s(x{j})" for j in range(p))
    terms = " + ".join(f"s({j})" for j in range(p))
    P = r.uniform(0, 1, (5, p))
    return dict(data=data, formula=formula,
                pyg=("LinearGAM", terms, X, y, None),
                pred={f"x{j}": P[:, j] for j in range(p)}, pred_X=P,
                truth=np.sin(6 * P[:, 0]))


def c_int_input():
    r = _rng(23)
    x = r.integers(0, 50, 300)
    y = np.sin(x / 8.0) + r.normal(0, 0.3, 300)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": np.array([10, 25])}, truth=np.sin(np.array([10, 25]) / 8.0))


def c_bool_y_logit():
    r = _rng(24)
    x = r.uniform(0, 1, 300)
    y = r.uniform(size=300) < 1 / (1 + np.exp(-(4 * x - 2)))
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LogisticGAM", "s(0)", x[:, None], y.astype(float), None),
                pred={"x": np.array([0.1, 0.9])},
                truth=1 / (1 + np.exp(-(4 * np.array([0.1, 0.9]) - 2))))


def c_constant_y():
    x, _ = base()
    y = np.full_like(x, 2.0)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": np.array([0.1, 0.9])}, truth=np.array([2.0, 2.0]))


def c_pure_noise():
    r = _rng(25)
    x = r.uniform(0, 1, 200)
    y = r.normal(0, 1, 200)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": np.array([0.1, 0.5, 0.9])}, truth=np.zeros(3))


def c_te_tiny():
    r = _rng(26)
    x = r.uniform(0, 1, 15); z = r.uniform(0, 1, 15)
    y = x * z + r.normal(0, 0.1, 15)
    return dict(data={"x": x, "z": z, "y": y}, formula="y ~ te(x, z)",
                pyg=("LinearGAM", "te(0, 1)", np.c_[x, z], y, None),
                pred={"x": np.array([0.5]), "z": np.array([0.5])}, truth=np.array([0.25]))


def c_huge_n_constant_x_dup():
    # extreme ties: 100k rows, 5 unique x
    r = _rng(27)
    x = r.integers(0, 5, 100000).astype(float)
    y = np.sin(x) + r.normal(0, 0.3, 100000)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": np.arange(5.0)}, truth=np.sin(np.arange(5.0)))


def c_x_with_outlier_x():
    # one x at 1e6, rest in [0,1] -> knots placement
    x, y = base()
    x = x.copy(); x[0] = 1e6
    px = np.linspace(0.05, 0.95, 5)
    return dict(data={"x": x, "y": y}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", x[:, None], y, None),
                pred={"x": px}, truth=np.sin(6 * px))


def c_empty():
    return dict(data={"x": np.array([], float), "y": np.array([], float)}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", np.zeros((0, 1)), np.zeros(0), None), pred={"x": np.array([0.5])})


def c_mismatched_len():
    return dict(data={"x": np.arange(10.0), "y": np.arange(9.0)}, formula="y ~ s(x)",
                pyg=("LinearGAM", "s(0)", np.arange(10.0)[:, None], np.arange(9.0), None),
                pred={"x": np.array([0.5])})


CASES = {k[2:]: v for k, v in globals().items() if k.startswith("c_")}
