"""Every key ``_build_fit_payload`` emits must be a field the Rust wire config knows.

``gamfit.fit`` marshals its model-spec kwargs into a JSON object that Rust
deserializes as ``FitRequestConfigDocument``, which is ``#[serde(deny_unknown_fields)]``.
A key the document does not name is therefore not "ignored" — it is a hard
``unknown field`` error that kills the call before any fitting happens.

That is exactly how three documented public kwargs died: ``smooths=``,
``penalties=`` and ``latents=`` were marshalled under their *kwarg* names while
the document names them ``smooth_descriptors``, ``analytic_penalties`` and
``latent_coordinates``. Every call that used them raised ``unknown field``, and
nothing arbitrated between the two spellings (fixed in ``54aa3340c``).

It is the same failure class as #2631, where the survival time-anchor rule was
written once in the engine and once in the CLI and the two disagreed: a front end
and the engine each owning their own copy of one contract, with no seam that
makes disagreement impossible. This test is that seam for the payload keys. It
does not check one key; it checks the whole emitted set against the parser's own
field list, so a *new* misspelled kwarg fails here rather than at a user's call
site.

The accepted-field list is read back from Rust rather than duplicated here —
duplicating it would recreate the very problem being guarded against. serde emits
it in the ``unknown field `x`, expected one of `a`, `b`, ...`` message, which is
the authoritative enumeration of the struct's fields.
"""

from __future__ import annotations

import importlib
import re
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))

import gamfit
from gamfit._api import _build_fit_payload


class _Latent:
    """Minimal duck-type accepted by ``_normalize_latents``."""

    def __init__(self) -> None:
        self.name = "u"
        self.n = 8
        self.d = 2


_UNSET_KWARGS: dict[str, typing.Any] = {
    "family": "gaussian",
    "offset": None,
    "weights": None,
    "persistent_warm_start_root": None,
    "transformation_normal": None,
    "transformation_normal_stage1": None,
    "survival_likelihood": None,
    "survival_time_anchor": None,
    "baseline_target": None,
    "baseline_scale": None,
    "baseline_shape": None,
    "baseline_rate": None,
    "baseline_makeham": None,
    "z_column": None,
    "link": None,
    "slope_formula": None,
    "frailty_kind": None,
    "frailty_sd": None,
    "hazard_loading": None,
    "scale_dimensions": None,
    "adaptive_regularization": None,
    "firth": None,
    "noise_formula": None,
    "noise_offset": None,
    "flexible_link": None,
    "precision_hyperpriors": None,
    "latents": None,
    "penalties": None,
    "smooths": None,
    "config": None,
}


_UNKNOWN_PROBE_KEY = "zz_not_a_wire_config_field"


def _accepted_wire_config_fields() -> frozenset[str]:
    """Ask the Rust parser which config fields it accepts.

    Sends one deliberately unknown key through the same entry point ``fit()``
    uses. The config object is deserialized before any data is touched, so this
    costs a parse, not a fit.
    """

    pytest.importorskip("gamfit._rust")
    frame = {"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]}
    with pytest.raises(Exception) as excinfo:
        gamfit.fit(frame, "y ~ x", config={_UNKNOWN_PROBE_KEY: 1})
    message = str(excinfo.value)
    assert _UNKNOWN_PROBE_KEY in message, (
        "the probe key must be reported as unknown; if the wire config stopped "
        f"rejecting unknown fields this guard is no longer sound. Got: {message}"
    )
    _, _, tail = message.partition("expected one of")
    assert tail, (
        "could not read the accepted-field list out of the parser's error; this "
        "test reads it from Rust on purpose rather than duplicating it. Got: "
        f"{message}"
    )
    fields = frozenset(re.findall(r"`([A-Za-z0-9_]+)`", tail))
    assert len(fields) > 20, (
        f"parsed an implausibly small field list from {tail!r}; the message "
        "format may have changed"
    )
    return fields


def _bspline_descriptor() -> typing.Any:
    from gamfit.smooth import BSpline

    return BSpline(degree=3)


def _fully_populated_payload() -> dict[str, typing.Any]:
    """A payload with every optional kwarg supplied, so every key is emitted.

    Values only have to survive Python-side normalization — the assertion is
    about key NAMES, and an unknown key is rejected before any value is read.
    """

    return _build_fit_payload(
        **{
            **_UNSET_KWARGS,
            "offset": "off",
            "weights": "w",
            "transformation_normal": True,
            "survival_likelihood": "location-scale",
            "survival_time_anchor": 25.0,
            "baseline_target": "weibull",
            "baseline_scale": 2.5,
            "baseline_shape": 1.0,
            "baseline_rate": 0.1,
            "baseline_makeham": 0.05,
            "z_column": "z",
            "link": "logit",
            "slope_formula": "s(x)",
            "frailty_kind": "gaussian-shift",
            "frailty_sd": 0.4,
            "hazard_loading": "full",
            "persistent_warm_start_root": "warm-start-fixture",
            "scale_dimensions": True,
            "adaptive_regularization": True,
            "firth": True,
            "noise_formula": "s(x)",
            "noise_offset": "logvar",
            "flexible_link": True,
            "precision_hyperpriors": {"block": {"shape": 2.0, "rate": 1.0}},
            "latents": {"u": _Latent()},
            "penalties": [{"kind": "ard", "target": "u"}],
            "smooths": {"x": _bspline_descriptor()},
        }
    )


def test_every_emitted_fit_payload_key_is_a_known_wire_config_field() -> None:
    accepted = _accepted_wire_config_fields()
    payload = _fully_populated_payload()

    unknown = sorted(set(payload) - accepted)
    assert not unknown, (
        f"gamfit.fit marshals {unknown} into the Rust config object, but "
        "FitRequestConfigDocument does not name "
        f"{'them' if len(unknown) > 1 else 'it'}. Because the document is "
        "deny_unknown_fields these are not ignored — every fit() call that "
        "supplies the corresponding kwarg dies with `unknown field`. Rename the "
        "payload key to the document's spelling (the kwarg name stays whatever "
        "reads best in Python), or add the field to the document."
    )


def test_the_three_historically_misspelled_keys_stay_fixed() -> None:
    """The specific spellings that were dead, pinned by name.

    The set test above is the general guard; this one names the three so a
    regression reads as itself rather than as an anonymous set difference.
    """

    payload = _fully_populated_payload()

    for wire_name, kwarg_name in (
        ("latent_coordinates", "latents"),
        ("analytic_penalties", "penalties"),
        ("smooth_descriptors", "smooths"),
    ):
        source = payload
        assert wire_name in source, (
            f"fit(..., {kwarg_name}=...) must be marshalled as {wire_name!r}"
        )
        assert kwarg_name not in source, (
            f"{kwarg_name!r} is the Python kwarg name, not the wire field name; "
            "emitting it makes the call fail with `unknown field`"
        )
