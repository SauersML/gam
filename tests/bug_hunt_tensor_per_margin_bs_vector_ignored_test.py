"""Bug hunt: per-margin basis types in a tensor smooth's ``bs=c(...)`` vector
are silently discarded — ``te(x, z, bs=c("cc","cc"))`` (both margins cyclic),
``te(x, z, bs=c("ps","ps"))`` (neither cyclic), and ``te(x, z, bs=c("cc","ps"))``
(mixed) all fit the *byte-identical* surface.

mgcv's ``te(...)`` takes a per-margin basis vector: ``bs=c("cc","cc")`` makes
both tensor margins cyclic (penalized cyclic cubic splines), ``bs=c("ps","ps")``
makes both ordinary P-splines, and a mixed ``c("cc","ps")`` makes only the first
margin cyclic. These are genuinely different bases and must produce different
fitted surfaces — in particular the cyclic margins impose periodicity that the
P-spline margins do not.

Observed: on the same data, ``bs=c("cc","cc")``, ``bs=c("ps","ps")`` and
``bs=c("cc","ps")`` return predictions that agree to 0.0 (bit-for-bit), while
they each differ from the bare ``te(x, z)`` default by ~2e-2. So the engine does
*something* with the presence of a ``bs=`` option, but the per-margin **vector**
content is thrown away: every spelling collapses to the same single basis on
both margins. The cyclic request in particular is lost (it does not impose
periodicity — a sibling of the periodic-tensor axis-0 wrap defect).

Likely origin: the tensor ``bs=c(...)`` parsing in
``crates/gam-terms/src/term_builder.rs`` (``parse_tensor_periodic_axes`` /
``parse_periodic_axes`` ~lines 993-1141 and the surrounding margin-basis
resolution) does not translate each margin's basis token into that margin's
basis/periodicity, so the per-axis spec is dropped.

Expected: at minimum, specifying different per-margin bases yields different
fitted surfaces. Concretely, ``bs=c("cc","cc")`` (both cyclic) must differ
materially from ``bs=c("ps","ps")`` (neither cyclic) on data with a real signal.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")
pd: Any = pytest.importorskip("pandas")

import gamfit


def _fit_predict(spec: str, seed: int):
    rng = np.random.default_rng(seed)
    n = 2000
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * z) + rng.normal(0.0, 0.05, n)
    df = pd.DataFrame({"x": x, "z": z, "y": y})
    model = gamfit.fit(df, f"y ~ {spec}", family="gaussian")
    grid = pd.DataFrame(
        {"x": [0.1, 0.3, 0.5, 0.7, 0.9], "z": [0.2, 0.4, 0.6, 0.8, 0.15]}
    )
    return np.asarray(model.predict(grid)).ravel()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tensor_per_margin_bs_vector_changes_fit(seed: int) -> None:
    both_cyclic = _fit_predict('te(x, z, bs=c("cc","cc"))', seed)
    neither_cyclic = _fit_predict('te(x, z, bs=c("ps","ps"))', seed)

    diff = float(np.max(np.abs(both_cyclic - neither_cyclic)))
    # Two genuinely different per-margin basis specs must not produce the
    # bit-identical surface. Today diff == 0.0 (the bs=c(...) vector is dropped).
    assert diff > 1e-6, (
        f"seed {seed}: te(..., bs=c('cc','cc')) and te(..., bs=c('ps','ps')) "
        f"produced indistinguishable fits (max prediction diff {diff:.3e}); the "
        "per-margin bs=c(...) vector is being discarded."
    )
