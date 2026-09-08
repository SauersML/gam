"""Bug hunt: ``firth=True`` on a binomial GAM with **two or more penalized
smooths** stalls the outer search far from stationarity and refuses the fit --
11 of 30 seeds on clean, non-separated data -- with a terminal gradient that is
exactly ``-rank(S_k)/2`` on the railed coordinate.

Firth / Jeffreys bias reduction is a headline feature ("Firth / Jeffreys bias
reduction handles separation in binomial fits", ``README.md``).  On the fixture
below -- 120 rows, ``eta = 2*(sin(5x) + 0.6*cos(4z))``, an ordinary Bernoulli
sample nowhere near separation -- the *only* thing that changes between success
and failure is the ``firth=`` flag:

| formula (n=120) | ``firth=False`` | ``firth=True`` |
|---|---|---|
| ``yb ~ s(x, k=8) + s(z, k=8)`` | 30/30 fit, 12 s | **11/30 raise**, 184 s |
| ``yb ~ s(x)`` | 24/24 fit | 24/24 fit |
| ``yb ~ x + z`` | 24/24 fit | 24/24 fit |

and the Firth fits that *do* succeed recover the truth exactly as well as the
ones without it (``corr(predicted, true p)`` min ``0.880`` / median ``0.963``
with Firth; min ``0.873`` / median ``0.956`` without).  Neither the data nor the
model class is the problem: one penalized smooth plus Firth is fine, two
penalized smooths without Firth are fine, the combination is not.

## The terminal gradient is a constant, not a number about the data

Every failure reports a large terminal gradient against a tiny stationarity
bound, and the diagnostic attributes it to a coordinate pinned at
``RHO_BOUND = 30`` (``crates/gam-solve/src/estimate/smoothing_correction.rs:110``):

    ... |g|=5.045e0 |Pg|=5.045e0 bound=4.168e-2 ... railed=[0]
    [#0 theta=3.000000e1 box=[-3.000000e1, 3.000000e1] ...]
    tail-snap declined: ... k=1: g=-5.000e0 H_kk=4.355e-4 ...

That ``-5.000e0`` is not data.  Varying only the basis dimension ``k`` (12 seeds
each, ``yb ~ s(x,k) + s(z,k)``, ``n=200``) moves it in exact lockstep with the
**rank of the block's wiggliness penalty**, ``rank(S_k) = k - 2``:

| ``k`` | block width | ``rank(S_k)`` | railed ``g_k`` | stalled ``|Pg|`` |
|---|---|---|---|---|
| 5  | 4  | 3  | ``-1.5`` | 1.60 |
| 8  | 7  | 6  | ``-3.0`` | 3.04 - 3.08 |
| 10 | 9  | 8  | ``-4.0`` | 4.02 - 4.05 |
| 12 | 11 | 10 | ``-5.0`` | 4.70 - 5.10 |
| 16 | 15 | 14 | (n/a)    | 7.00 - 7.13 |

i.e. ``g_railed = -rank(S_k) / 2``, to every digit printed, on unrelated
datasets.  Every railed value observed anywhere in this sweep is an exact
half-integer -- ``-0.5``, ``-1.5``, ``-2.5``, ``-3.0``, ``-4.0``, ``-5.0`` --
including ``-5.000e-1`` for the rank-1 null-space shrinkage block of the double
penalty, which is ``-1/2`` by the same law.  Data does not produce exact
half-integers.

The REML ``rho``-gradient of a penalized block is
``0.5 * (lambda_k * tr(H^-1 S_k) - rank(S_k) + tau * beta' S_k beta)``.  As
``lambda_k -> inf`` the trace term ``-> rank(S_k)`` and the quadratic term
``-> 0``, so the gradient goes to **zero** and the rail is a legitimate
stationary point.  Landing on exactly ``-rank(S_k)/2`` is the signature of the
``+0.5 * lambda_k * tr(H^-1 S_k)`` half being contributed as **zero** under
Firth, leaving only the ``-0.5 * rank(S_k)`` half.  The search then sees a
constant pull toward ``rho -> inf`` that never vanishes, the line search dies
(``line_search=StepSizeTooSmall`` / ``MaxAttempts after 50 attempt(s)``,
``origin=BfgsCostStallExit``), and the certificate refuses the fit.

Two further checks that rule out "it is just a hard problem":

* **It is a fixed point, not an iteration shortfall.**  Re-running seed 0 with
  ``config={"outer_max_iter": 200 / 1000 / 4000}`` returns the *bit-identical*
  terminal state each time -- ``|Pg| = 5.045e0`` and ``rho_checkpoint =
  [30.0, 18.483753006549268, -6.327676121185661, -3.8418351356240077]``.
* **The interior coordinates repeat across datasets too.**  Six of ten
  ``rho_checkpoint``s (n=200) carry a coordinate at ``18.483753``,
  ``18.486321``, ``18.486927``, ``18.488036``, ``18.849776``, ``18.637189`` --
  five digits of agreement on unrelated samples.

Separately worth a look: at seed 0 the railed coordinate has an *outward*
gradient (``theta = 30`` at the upper box bound, ``g = -5.000``), yet the
reported projected norm is ``|Pg| == |g| == 5.045``.  A KKT projection would
drop that component, leaving ``sqrt(5.045^2 - 5.000^2) = 0.67``.

Suggested place to look: ``crates/gam-solve/src/reml/firth.rs`` (the Jeffreys
``rho``/``tau`` derivative machinery) and ``ExactJeffreysTerm`` in
``crates/gam-solve/src/reml/reml_outer_engine/derivative_providers.rs:520``.
The signature -- exact with one lambda, ``-rank_k/2`` with two -- is what a
per-block ``rho_k`` trace term looks like when only one lane contributes.

The assertions are fix-agnostic: the model must fit and must recover the
probability surface it was generated from.  No solver, gradient, bound or rail
is named.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

import gamfit

_N = 120
_K = 8
_FORMULA = "yb ~ s(x, k=8) + s(z, k=8)"

# Seeds observed to raise at HEAD with firth=True on this fixture.
_FIRTH_FAILING_SEEDS = [0, 2, 9, 12, 16, 18]
# Two that already fit, kept as in-test controls.
_FIRTH_PASSING_SEEDS = [1, 4]


def _dataset(
    seed: int,
) -> tuple[dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, _N))
    z = rng.uniform(0.0, 1.0, _N)
    eta = 2.0 * (np.sin(5.0 * x) + 0.6 * np.cos(4.0 * z))
    probability = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=_N) < probability).astype(np.float64)
    return {"x": x, "z": z, "yb": y}, probability


def _recovery(
    model: gamfit.Model,
    data: dict[str, npt.NDArray[np.float64]],
    probability: npt.NDArray[np.float64],
) -> float:
    predicted = np.asarray(
        model.predict(data, return_type="pandas")["posterior_mean"], dtype=np.float64
    )
    assert np.isfinite(predicted).all()
    return float(np.corrcoef(predicted, probability)[0, 1])


@pytest.mark.parametrize("seed", _FIRTH_FAILING_SEEDS + _FIRTH_PASSING_SEEDS)
def test_firth_binomial_two_smooths_fits_and_recovers_the_probability_surface(
    seed: int,
) -> None:
    data, probability = _dataset(seed)
    model = gamfit.fit(data, _FORMULA, family="binomial", firth=True)
    # The 19 seeds that already fit under Firth reach corr >= 0.880, and all 30
    # seeds without Firth reach corr >= 0.873. 0.80 is a floor with real slack
    # that still rejects a "fix" returning a degenerate or flat surface.
    correlation = _recovery(model, data, probability)
    assert correlation > 0.80, (
        f"Firth fit recovered the surface at corr={correlation:.4f}"
    )


@pytest.mark.parametrize("seed", _FIRTH_FAILING_SEEDS[:3])
def test_firth_does_not_make_a_fittable_binomial_gam_unfittable(seed: int) -> None:
    """The Firth fit must exist wherever the plain fit does.

    Firth/Jeffreys adds a bounded penalty to the likelihood; it cannot turn a
    well-posed, converging REML problem into one with no stationary point. On
    these seeds the identical model without ``firth=True`` converges and
    recovers the surface, which the first assertion here pins as a control.
    """
    data, probability = _dataset(seed)

    plain = gamfit.fit(data, _FORMULA, family="binomial")
    assert _recovery(plain, data, probability) > 0.80  # control: well posed

    firth = gamfit.fit(data, _FORMULA, family="binomial", firth=True)
    assert _recovery(firth, data, probability) > 0.80
