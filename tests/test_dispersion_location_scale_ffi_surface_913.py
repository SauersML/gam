"""#913: the genuine-dispersion location-scale families (NB / Gamma / Beta /
Tweedie) are reachable from ``gamfit.fit`` via ``noise_formula``.

Before this surface landed, ``gamfit.fit(..., family="gamma",
noise_formula=...)`` reached the core dispersion-GAMLSS engine but the Python
FFI payload builder rejected the result with "dispersion location-scale fits
are not yet surfaced through the Python FFI payload builder". These tests pin
that the four dispersion families now fit and serialize cleanly through the
Python surface, that the family tag is magic-routed from ``family`` + the
presence of a ``noise_formula``, and that a saved dispersion model round-trips.

``family_name`` is the observable these tests key on precisely because it is the
one place the two channels can be told apart: it must name the *location-scale*
family the fit actually used, not the mean channel's likelihood. See the note
below the imports for the defect that made it report the latter.
"""

from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


# #1512 / #913 — FIXED, and the diagnosis this header used to carry was WRONG.
#
# It said the `family=<nb|gamma|tweedie|beta>` + `noise_formula=` magic-routing
# was "not wired through the current Python surface" and told the reader to
# "route a present noise_formula to the dispersion location-scale family". The
# routing was already wired and had been all along: `entry.rs` sends a present
# `noise_formula` to `materialize_location_scale`,
# `dispersion_location_scale_kind` returns the `DispersionFamilyKind`, the fit
# runs as `FitRequest::DispersionLocationScale`, and
# `assemble_location_scale_payload` persists the right tag. Measured on the saved
# payload of a real fit, before any fix:
#
#     nb       summary='Negative-Binomial Log'  payload.family='negbin-location-scale'
#     gamma    summary='Gamma Log'              payload.family='gamma-location-scale'
#     tweedie  summary='Tweedie Log'            payload.family='tweedie-location-scale'
#     beta     summary='Beta Regression Logit'  payload.family='beta-location-scale'
#
# Every fit had already routed correctly and persisted the correct tag. The
# defect was one layer later, on the SUMMARY surface: `summary_json_impl` read
# `model.likelihood().pretty_name()`, and `FittedFamily::LocationScale` carries
# only the MEAN channel's law — so a two-channel dispersion GAMLSS and the
# mean-only GLM of the same family were indistinguishable through `family_name`.
# The CLI's own fit line already printed the persisted tag, so the Python summary
# contradicted the CLI (SPEC rule 9). `FittedModel::display_family_name` is now
# the single authority both read.
#
# Recorded at this length because acting on the old header would have sent the
# next reader to rewrite routing that was already correct.


def _heteroscedastic_count_rows(n: int, seed: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n)
    mu = np.exp(0.5 + 0.8 * x)
    # Overdispersion that grows with x so the noise channel has signal.
    theta = np.exp(1.5 - 1.0 * x)
    # Negative-binomial draw via Gamma-Poisson mixture.
    lam = rng.gamma(shape=theta, scale=mu / theta)
    y = rng.poisson(lam).astype(float)
    return [{"y": float(y[i]), "x": float(x[i])} for i in range(n)]


@pytest.mark.parametrize(
    "family, expected_tag",
    [
        ("negative-binomial", "negbin-location-scale"),
        ("gamma", "gamma-location-scale"),
        ("tweedie", "tweedie-location-scale"),
    ],
)
def test_dispersion_location_scale_fits_and_serializes(family, expected_tag) -> None:
    rows = _heteroscedastic_count_rows(160, seed=11)
    if family == "gamma":
        # Gamma needs strictly-positive responses.
        rows = [{"y": r["y"] + 0.5, "x": r["x"]} for r in rows]

    model = gamfit.fit(rows, "y ~ x", family=family, noise_formula="x")

    # Magic-routing: family + a noise_formula selects the dispersion
    # location-scale family, surfaced in the model's family name.
    family_tag = model.family_name.lower()
    assert expected_tag in family_tag, (
        f"family={family} with a noise_formula must magic-route to the "
        f"dispersion location-scale family '{expected_tag}', got '{family_tag}'"
    )

    # The model must serialize and round-trip without the historical FFI
    # rejection. ``dumps``/``loads`` exercise the saved-model payload the
    # builder now constructs.
    blob = model.dumps()
    assert isinstance(blob, (bytes, bytearray)) and len(blob) > 0
    reloaded = gamfit.loads(blob)
    assert reloaded.family_name.lower() == family_tag


def test_beta_dispersion_location_scale_fits_and_serializes() -> None:
    rng = np.random.default_rng(7)
    n = 160
    x = np.linspace(-1.0, 1.0, n)
    mu = 1.0 / (1.0 + np.exp(-(0.3 + 0.7 * x)))
    phi = np.exp(2.0 - 0.8 * x)
    a = mu * phi
    b = (1.0 - mu) * phi
    y = np.clip(rng.beta(a, b), 1e-4, 1.0 - 1e-4)
    rows = [{"y": float(y[i]), "x": float(x[i])} for i in range(n)]

    model = gamfit.fit(rows, "y ~ x", family="beta", noise_formula="x")
    family_tag = model.family_name.lower()
    assert "beta-location-scale" in family_tag, (
        f"beta + noise_formula must route to beta-location-scale, got '{family_tag}'"
    )

    blob = model.dumps()
    assert len(blob) > 0
    reloaded = gamfit.loads(blob)
    assert "beta-location-scale" in reloaded.family_name.lower()


def test_constant_noise_formula_still_routes_dispersion_family() -> None:
    # A pure-intercept noise formula still selects the dispersion family (a
    # scalar-dispersion GAMLSS); it must not silently fall back to the
    # mean-only fit or the Gaussian location-scale path.
    rows = _heteroscedastic_count_rows(120, seed=3)
    rows = [{"y": r["y"] + 0.5, "x": r["x"]} for r in rows]
    model = gamfit.fit(rows, "y ~ x", family="gamma", noise_formula="1")
    assert "gamma-location-scale" in model.family_name.lower()
