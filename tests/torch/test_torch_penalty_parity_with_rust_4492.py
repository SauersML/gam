"""gam#4492: the torch path's penalty for a term is the term builder's penalty.

`gamfit/torch/fit.py` used to build its own penalty for `TensorBSpline` and
`Matern`. Both diverged from what `gamfit.fit` builds for the same spec: the
tensor summed `I ⊗ S_a ⊗ I` under ONE λ where the builder emits one candidate
per margin, each measured by its neighbours' function Grams and normalized
first; the Matérn took the raw symmetrised `K_cc` on the raw kernel columns
where the builder emits ν-gated collocation operator candidates through the
kernel identifiability chart. So `gamfit.torch.fit(x, y, TensorBSpline(...))`
fitted a different model, with a different number of smoothing parameters, than
`gamfit.fit` did for the same term -- which SPEC forbids outright.

Both kinds are now fitted by the engine whole: `gamfit.torch.fit` hands a
single `TensorBSpline` or `Matern` term to the fit `gamfit.fit` runs for the
same spec (`_fit_engine_term`), so the term's basis, identifiability chart,
penalties and REML criterion have one definition. Taking only the builder's
penalties (`smooth_term_realized_penalties`) could not close the divergence:
the torch block backends price one λ per coefficient block, both terms realize
several penalties on one block, and the realized block is centred against a
model intercept a torch block does not carry.

Two claims are asserted here:

* `test_engine_realizes_the_penalties_the_fit_carries`: the realization entry
  returns, for each term, as many penalties as `gamfit.fit` carries smoothing
  parameters, each square on the realized design block.
* `test_torch_fit_is_the_engine_fit`: the torch fit and `gamfit.fit` agree on
  the per-margin λ̂, the EDF, the REML score and the fitted values, EXACTLY.
  They are one computation on the same rows, so any difference at all -- not
  just one beyond a rounding band -- is two definitions of the term again. The
  positive control is the torch path this replaced: its tensor fit carried ONE
  λ against the engine's two, and fails the λ-count assertion before any value
  is compared.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")
torch = pytest.importorskip("torch")
gt = pytest.importorskip("gamfit.torch")

import gamfit
from gamfit._api import _jsonable_array
from gamfit._binding import rust_module
from gamfit.smooth import BSpline, Matern, TensorBSpline

# Both arms name `bs='ps'`, and that is part of the claim rather than a detail.
# A bare `te(x1, x2)` is mgcv's default tensor, whose margins are natural cubic
# regression splines -- `margin_wants_cr` in `term_builder.rs` treats an unset
# `bs` as `cr`. `TensorBSpline(marginals=[BSpline(...), ...])` is a B-spline
# tensor. Comparing one against the other is comparing two models, which is the
# very failure gam#4492 exists to stop, so the two arms had to be made the same
# basis before any parity assertion between them means anything. A descriptor
# cannot do it: the marginal bridge tunes a margin and does not set its family.

# One fixture for both arms: the two fits must see the same rows, so the data
# are built once and handed to each as its own frontend takes them.
N_ROWS = 160
SEED = 4492


def _surface_frame() -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(0.0, 1.0, N_ROWS)
    x2 = rng.uniform(0.0, 1.0, N_ROWS)
    truth = np.sin(2.0 * np.pi * x1) * (0.5 + x2)
    y = truth + 0.05 * rng.standard_normal(N_ROWS)
    return {"x1": x1, "x2": x2, "y": y}, truth


def _tensor_smooth() -> TensorBSpline:
    # Both arms name the same marginals, so neither side's default can stand in
    # for the other's and make a disagreement look like agreement.
    return TensorBSpline(
        marginals=[BSpline(knots=8, degree=3), BSpline(knots=8, degree=3)]
    )


def _matern_smooth(centers: np.ndarray) -> Matern:
    return Matern(centers=centers, nu=1.5, length_scale=0.35)


def _matern_centers() -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 12)
    return grid.reshape(-1, 1)


def _realized(points: np.ndarray, term: str, smooth: Any) -> tuple[Any, list[Any]]:
    descriptor = _jsonable_array(dict(smooth.to_rust_descriptor()))
    return rust_module().smooth_term_realized_penalties(
        np.ascontiguousarray(points, dtype=np.float64), term, json.dumps(descriptor)
    )


def test_engine_realizes_the_penalties_the_fit_carries() -> None:
    frame, _ = _surface_frame()

    tensor_points = np.column_stack([frame["x1"], frame["x2"]])
    tensor_smooth = _tensor_smooth()
    tensor_design, tensor_penalties = _realized(
        tensor_points, "te(x0, x1, bs='ps')", tensor_smooth
    )
    tensor_fit = gamfit.fit(
        frame, "y ~ te(x1, x2, bs='ps')", smooths={("x1", "x2"): tensor_smooth}
    )
    assert len(tensor_penalties) == len(tensor_fit.smoothing_parameters()), (
        "the entry must realize one penalty per smoothing parameter the fit "
        f"carries: entry={len(tensor_penalties)}, "
        f"fit={len(tensor_fit.smoothing_parameters())}"
    )
    width = int(np.asarray(tensor_design).shape[1])
    assert int(np.asarray(tensor_design).shape[0]) == N_ROWS
    for index, penalty in enumerate(tensor_penalties):
        block = np.asarray(penalty)
        assert block.shape == (width, width), (
            f"te penalty {index} is {block.shape}, not the realized block "
            f"({width}, {width})"
        )
        # A Gram accumulated over `width` summands may disagree between its
        # triangles by the rounding that accumulation left, `γ_width · max|S|`
        # (Higham, ASNA 2nd ed., §3.1), and by no more. The bar is that band,
        # read off this block, not a chosen closeness.
        scale = float(np.abs(block).max())
        band = width * float(np.finfo(np.float64).eps) * scale
        assert float(np.abs(block - block.T).max()) <= band, (
            f"te penalty {index} disagrees between its triangles by more than "
            f"its own accumulation band {band:.3e}"
        )

    centers = _matern_centers()
    matern_points = frame["x1"].reshape(-1, 1)
    matern_smooth = _matern_smooth(centers)
    matern_design, matern_penalties = _realized(
        matern_points, "matern(x0)", matern_smooth
    )
    matern_fit = gamfit.fit(
        frame, "y ~ matern(x1)", smooths={"x1": matern_smooth}
    )
    assert len(matern_penalties) == len(matern_fit.smoothing_parameters()), (
        "the entry must realize one penalty per smoothing parameter the fit "
        f"carries: entry={len(matern_penalties)}, "
        f"fit={len(matern_fit.smoothing_parameters())}"
    )
    width = int(np.asarray(matern_design).shape[1])
    for index, penalty in enumerate(matern_penalties):
        block = np.asarray(penalty)
        assert block.shape == (width, width), (
            f"matern penalty {index} is {block.shape}, not the realized block "
            f"({width}, {width})"
        )


def test_the_entry_refuses_a_descriptor_that_is_not_the_terms(
) -> None:
    """A descriptor whose axis count disagrees with the term is an error, not a
    silently different model: `apply_smooth_overrides` matches the registry key
    to the term's columns, and a mismatch is what it reports."""
    frame, _ = _surface_frame()
    points = np.column_stack([frame["x1"], frame["x2"]])
    # Three marginals named for a two-axis term.
    wrong = TensorBSpline(
        marginals=[
            BSpline(knots=8, degree=3),
            BSpline(knots=8, degree=3),
            BSpline(knots=8, degree=3),
        ]
    )
    with pytest.raises(Exception) as caught:
        _realized(points, "te(x0, x1, bs='ps')", wrong)
    assert "marginals" in str(caught.value), str(caught.value)


def test_torch_fit_is_the_engine_fit() -> None:
    """`gamfit.torch.fit` on a te and a Matérn term IS `gamfit.fit` on them.

    The torch arm names its axes `x0, x1`; the engine arm is written here with
    the frame's own names, independently of the torch path's lowering, so a
    torch-side term string that drifted from the spec would show as a
    disagreement rather than as the same call twice.
    """
    frame, _ = _surface_frame()
    response = torch.as_tensor(frame["y"], dtype=torch.float64)

    tensor_smooth = _tensor_smooth()
    tensor_points = torch.as_tensor(
        np.column_stack([frame["x1"], frame["x2"]]), dtype=torch.float64
    )
    torch_tensor_fit = gt.fit(tensor_points, response, tensor_smooth)
    rust_tensor_fit = gamfit.fit(
        frame, "y ~ te(x1, x2, bs='ps')", smooths={("x1", "x2"): tensor_smooth}
    )
    assert len(rust_tensor_fit.smoothing_parameters()) >= 2, (
        "te(x1, x2): the engine carries "
        f"{len(rust_tensor_fit.smoothing_parameters())} smoothing parameters, "
        "fewer than one per margin; the per-margin claim below would be vacuous"
    )
    _assert_fit_identity("te(x1, x2)", torch_tensor_fit, rust_tensor_fit, frame)

    centers = _matern_centers()
    matern_smooth = _matern_smooth(centers)
    matern_points = torch.as_tensor(frame["x1"], dtype=torch.float64)
    torch_matern_fit = gt.fit(matern_points, response, matern_smooth)
    rust_matern_fit = gamfit.fit(
        frame, "y ~ matern(x1)", smooths={"x1": matern_smooth}
    )
    _assert_fit_identity("matern(x1)", torch_matern_fit, rust_matern_fit, frame)


def test_torch_fit_refuses_what_the_engine_fit_does_not_carry() -> None:
    """Autograd, a multi-column response and a λ warm start are refused by
    name, not dropped: the engine fit has no backward, fits one response, and
    takes no warm start."""
    frame, _ = _surface_frame()
    points = torch.as_tensor(frame["x1"], dtype=torch.float64)
    response = torch.as_tensor(frame["y"], dtype=torch.float64)
    smooth = _matern_smooth(_matern_centers())

    with pytest.raises(NotImplementedError, match="backward"):
        gt.fit(points.clone().requires_grad_(True), response, smooth)
    with pytest.raises(NotImplementedError, match="response column"):
        gt.fit(points, torch.stack([response, response], dim=1), smooth)
    with pytest.raises(NotImplementedError, match="warm start"):
        gt.fit(points, response, smooth, init_lambdas=torch.tensor(1.0))


def _assert_fit_identity(
    label: str, torch_fit: Any, rust_fit: Any, frame: dict[str, np.ndarray]
) -> None:
    """Per-margin λ̂, EDF, REML score and fitted values, bit for bit."""
    torch_lambdas = np.asarray(torch_fit.lambdas, dtype=float).reshape(-1)
    rust_parameters = rust_fit.smoothing_parameters()
    rust_lambdas = np.asarray(
        [rust_parameters[index] for index in sorted(rust_parameters)], dtype=float
    )
    assert torch_lambdas.shape == rust_lambdas.shape, (
        f"{label}: the two fits carry different numbers of smoothing "
        f"parameters: torch={torch_lambdas.shape[0]}, rust={rust_lambdas.shape[0]}"
    )
    assert np.array_equal(torch_lambdas, rust_lambdas), (
        f"{label}: lambda-hat differs: torch={torch_lambdas.tolist()}, "
        f"rust={rust_lambdas.tolist()}"
    )

    summary = rust_fit.summary()
    rust_edf = float(sum(rust_fit.smooth_edf.values()))
    torch_edf = float(torch_fit.edf)
    assert torch_edf == rust_edf, (
        f"{label}: smooth EDF differs: torch={torch_edf}, rust={rust_edf}"
    )
    assert float(torch_fit.reml_score) == float(summary.reml_score), (
        f"{label}: REML score differs: torch={float(torch_fit.reml_score)}, "
        f"rust={summary.reml_score}"
    )

    torch_fitted = np.asarray(torch_fit.fitted, dtype=float).reshape(-1)
    rust_fitted = np.asarray(rust_fit.predict(frame), dtype=float).reshape(-1)
    assert np.array_equal(torch_fitted, rust_fitted), (
        f"{label}: fitted values differ by up to "
        f"{float(np.abs(torch_fitted - rust_fitted).max()):.3e}"
    )
