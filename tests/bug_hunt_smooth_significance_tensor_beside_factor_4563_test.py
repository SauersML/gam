"""gam#4563: ``Model.smooth_significance`` on a tensor smooth beside a factor.

That combination reaches the multi-scale draw selection in
``smooth_term_lr.rs``: a ``te(x, z)`` term carries one smoothing scale per
margin, so the replay's per-draw selection sweeps TWO open axes rather than the
single axis a ``s(x)`` term gives it. #4563 records that no test on main drove
the surviving production entry (``smooth_term_lr_inference_json``, reached from
this method) on that shape at all, so the route with more than one scale was
exercised by nothing. The branch's own reproducer drove
``smooth_term_summary_rows``, whose smooth row comes from the variance-component
score test and never reaches the replay, so it did not cover this either.

The two arms are the point. A row coming back is not evidence the replay did
anything: a test that only asserted "it returns" would pass just as well if the
selection collapsed to the fitted point on every draw. So the same formula runs
twice on the same design, once over a response carrying a genuine ``te(x, z)``
surface and once over a response carrying only the factor, and the term has to
be found in the first and not manufactured in the second.

* planted -- a real interaction surface must be detected;
* null -- a response with no ``te(x, z)`` structure must NOT come back with an
  astronomically small p-value. A calibrated p-value falls below ``1e-6`` with
  probability ``1e-6``, so this bar is not flaky; what it catches is the failure
  where the replay's tail estimate is anti-conservative, which is the direction
  a broken selection fails in.

``n`` is 240 rather than #4563's 500 because the cost is the issue's subject:
the measurement there read 5.1 s to fit and 21.7 s to replay at 500 rows, and
the number of scales -- which is what selects this route -- does not depend on
``n``. To read the sweep count the stop must eventually be denominated against,
run this file under ``GAMFIT_LOG=debug`` and grep for
``[#4563 multiscale selection]``.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit

FORMULA = "y ~ factor(g) + te(x, z)"


def _design(n: int, seed: int) -> dict[str, list[float]]:
    """The covariates both arms share, so only the response differs."""
    rng = np.random.default_rng(seed)
    return {
        "x": list(rng.uniform(-1.0, 1.0, n)),
        "z": list(rng.uniform(-1.0, 1.0, n)),
        "g": list(rng.integers(0, 3, n).astype(float)),
    }


def _te_row(frame: dict[str, list[float]]) -> dict[str, Any]:
    model = gamfit.fit(frame, FORMULA, family="gaussian")
    rows = model.smooth_significance(frame)
    # `factor(g)` is parametric, so the only penalized smooth is the tensor
    # term: one row here is what says the fit really is "te beside a factor"
    # and not a tensor term on its own.
    assert len(rows) == 1, f"expected the tensor term alone as a penalized smooth; got {rows}"
    row = rows[0]
    assert "te(" in row["name"], f"the tested smooth must be the tensor term; got {row['name']}"
    assert "p_value_unavailable" not in row, (
        f"the tensor term beside a factor must be testable, not withheld: {row}"
    )
    p_value = float(row["p_value"])
    assert 0.0 <= p_value <= 1.0, f"p_value out of range: {row}"
    assert np.isfinite(float(row["statistic_lr"])), f"non-finite LR statistic: {row}"
    return row


def test_smooth_significance_detects_a_planted_tensor_beside_a_factor_4563() -> None:
    n = 240
    frame = _design(n, seed=31)
    x = np.asarray(frame["x"], dtype=float)
    z = np.asarray(frame["z"], dtype=float)
    g = np.asarray(frame["g"], dtype=float)
    noise = np.random.default_rng(4563).standard_normal(n)
    # A genuine interaction: the surface is not additive in x and z, so it
    # cannot be absorbed by either margin's main effect.
    frame = dict(frame)
    frame["y"] = list(0.8 * g + 2.0 * np.sin(2.5 * x * z) + 0.25 * noise)

    row = _te_row(frame)
    assert float(row["p_value"]) < 0.01, (
        f"a planted te(x, z) surface beside a factor must be detected; got {row}"
    )


def test_smooth_significance_does_not_manufacture_a_tensor_beside_a_factor_4563() -> None:
    n = 240
    frame = _design(n, seed=31)
    g = np.asarray(frame["g"], dtype=float)
    noise = np.random.default_rng(45631).standard_normal(n)
    # The factor is the whole signal; nothing in (x, z) carries structure.
    frame = dict(frame)
    frame["y"] = list(0.8 * g + 0.25 * noise)

    row = _te_row(frame)
    assert float(row["p_value"]) > 1.0e-6, (
        "a response with no te(x, z) structure must not come back with an astronomically "
        f"small p-value; got {row}"
    )
