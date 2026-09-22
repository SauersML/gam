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

The penalties now come from the engine entry `smooth_term_realized_penalties`,
which runs the fit's own lowering (`parse_formula` → `build_termspec` →
`apply_smooth_overrides` → `build_term_collection_design`) and returns what
`gamfit.fit` would realize.

Two things are asserted here, and they are different claims:

* `test_engine_realizes_the_penalties_the_fit_carries` is step 1's own claim and
  passes now: the entry returns, for each of the two terms, as many penalties as
  `gamfit.fit` carries smoothing parameters for that term, each square on the
  realized design block.
* `test_torch_fit_matches_rust_fit_on_te_and_matern` is the parity the issue
  asks for -- per-margin λ̂, EDF and fitted values -- and is a STRICT xfail on
  `NotImplementedError` until the block backend takes a penalty list per block
  (gam#4492 step 2). `gaussian_reml_fit_blocks_exact` prices
  `P = blockdiag(λ_k S_k)`, one λ per coefficient block, and both terms realize
  several penalties on ONE block. The torch fit refuses rather than summing them
  under a single λ, because that sum IS the divergence. Strict xfail means the
  day step 2 lands this test reports the parity instead of the refusal, and any
  failure that is not that refusal fails the suite today.
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
        tensor_points, "te(x0, x1)", tensor_smooth
    )
    tensor_fit = gamfit.fit(
        frame, "y ~ te(x1, x2)", smooths={("x1", "x2"): tensor_smooth}
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
        _realized(points, "te(x0, x1)", wrong)
    assert "marginals" in str(caught.value), str(caught.value)


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "gam#4492 step 2: gaussian_reml_fit_blocks_exact prices one lambda per "
        "coefficient block, and both terms realize several penalties on one "
        "block. The torch fit refuses rather than summing them under a single "
        "lambda. This becomes the parity assertion when the block API takes a "
        "penalty list per block."
    ),
)
def test_torch_fit_matches_rust_fit_on_te_and_matern() -> None:
    frame, _ = _surface_frame()
    response = torch.as_tensor(frame["y"], dtype=torch.float64).reshape(-1, 1)

    tensor_smooth = _tensor_smooth()
    tensor_points = torch.as_tensor(
        np.column_stack([frame["x1"], frame["x2"]]), dtype=torch.float64
    )
    torch_tensor_fit = gt.fit(tensor_points, response, tensor_smooth)
    rust_tensor_fit = gamfit.fit(
        frame, "y ~ te(x1, x2)", smooths={("x1", "x2"): tensor_smooth}
    )
    _assert_fit_parity("te(x1, x2)", torch_tensor_fit, rust_tensor_fit, frame)

    centers = _matern_centers()
    matern_smooth = _matern_smooth(centers)
    matern_points = torch.as_tensor(
        frame["x1"].reshape(-1, 1), dtype=torch.float64
    )
    torch_matern_fit = gt.fit(matern_points, response, matern_smooth)
    rust_matern_fit = gamfit.fit(
        frame, "y ~ matern(x1)", smooths={"x1": matern_smooth}
    )
    _assert_fit_parity("matern(x1)", torch_matern_fit, rust_matern_fit, frame)


def _assert_fit_parity(
    label: str, torch_fit: Any, rust_fit: Any, frame: dict[str, np.ndarray]
) -> None:
    """Per-margin λ̂, EDF and fitted values, on the identical rows.

    ONE derived bar, `√ε`, applied in the coordinate each quantity is smooth in.

    The two arms minimize the SAME profiled REML criterion over the same
    matrices, so they differ only in the order their floating-point operations
    run, and a criterion whose VALUE is resolved to `ε` relative locates its
    ARGMIN only to `√ε`: at a minimum `V(ρ) − V(ρ̂) ≈ ½ V''(ρ̂)(ρ − ρ̂)²`, so a
    value perturbation of `ε|V|` moves `ρ̂` by `√(2ε|V|/V'')`. `ρ = log λ` is the
    coordinate the outer search works in and the one the criterion is curved in,
    so the smoothing parameters are compared there. EDF and the fitted vector are
    differentiable functions of `ρ` with O(1) sensitivity in the interior, so
    they inherit the same bar against their own scale.

    A failure AT this bar, with the two criterion values agreeing, is a
    statement about `V''` at the optimum rather than a disagreement about the
    model; a failure well beyond it is two different criteria, which is what
    gam#4492 was.
    """
    root_eps = float(np.sqrt(np.finfo(np.float64).eps))

    torch_lambdas = np.sort(
        np.asarray(torch_fit.lambdas.detach().cpu(), dtype=float).reshape(-1)
    )
    rust_lambdas = np.sort(
        np.asarray(list(rust_fit.smoothing_parameters().values()), dtype=float)
    )
    assert torch_lambdas.shape == rust_lambdas.shape, (
        f"{label}: the two fits carry different numbers of smoothing "
        f"parameters: torch={torch_lambdas.shape[0]}, "
        f"rust={rust_lambdas.shape[0]}"
    )
    assert float(torch_lambdas.min()) > 0.0 and float(rust_lambdas.min()) > 0.0, (
        f"{label}: a fitted smoothing parameter is not positive: "
        f"torch={torch_lambdas.tolist()}, rust={rust_lambdas.tolist()}"
    )
    log_gap = float(np.abs(np.log(torch_lambdas) - np.log(rust_lambdas)).max())
    assert log_gap <= root_eps, (
        f"{label}: per-margin lambda-hat differs by {log_gap:.3e} in log lambda, "
        f"beyond the sqrt(eps)={root_eps:.3e} an argmin of a criterion resolved "
        f"to eps is located to"
    )

    torch_edf = float(
        np.asarray(torch_fit.edf.detach().cpu(), dtype=float).sum()
    )
    rust_edf = float(sum(rust_fit.smooth_edf.values()))
    assert abs(torch_edf - rust_edf) <= root_eps * max(1.0, abs(rust_edf)), (
        f"{label}: smooth EDF differs: torch={torch_edf}, rust={rust_edf}"
    )

    torch_fitted = np.asarray(
        torch_fit.fitted.detach().cpu(), dtype=float
    ).reshape(-1)
    rust_fitted = np.asarray(rust_fit.predict(frame), dtype=float).reshape(-1)
    # The Rust collection carries an intercept the torch block does not, so the
    # two fitted vectors agree up to that one constant. Comparing the centred
    # vectors states the shared claim -- the same surface -- without asserting a
    # column one side does not have. The bar is the same sqrt(eps), against the
    # surface's own scale.
    torch_centred = torch_fitted - torch_fitted.mean()
    rust_centred = rust_fitted - rust_fitted.mean()
    surface_scale = float(np.abs(rust_centred).max())
    gap = float(np.abs(torch_centred - rust_centred).max())
    assert gap <= root_eps * max(1.0, surface_scale), (
        f"{label}: fitted surfaces differ by {gap:.3e}, beyond sqrt(eps) times "
        f"the surface scale {surface_scale:.3e}"
    )
