"""The one owner of a KL measurement's floor, reached through the Python binding (#2263).

The dose-ledger drivers floor every row here instead of re-deriving the band in Python,
so the binding must return the Rust owner's values and refuse what the owner cannot
price.
"""

from __future__ import annotations

import math

import pytest

from gamfit import _rust


def test_the_binding_returns_the_owners_band_and_floor_for_bfloat16_and_float32_logits() -> None:
    # bfloat16 logits of magnitude 30 are spaced 2^(4 - 7) = 1/8 apart, so logit
    # rounding alone produces 1/128 nats; float32 ones are spaced 2^(4 - 23).
    bf16 = _rust.kl_measurement_floor("bfloat16", 151_936, 30.0, 0.25, 0.0)
    assert bf16["evaluation_band_nats"] > 0.0
    assert bf16["measurement_band_nats"] == 1.0 / 128.0 + bf16["evaluation_band_nats"]
    assert bf16["floor_nats"] == bf16["measurement_band_nats"]
    f32 = _rust.kl_measurement_floor("float32", 151_936, 30.0, 0.25, 0.0)
    assert f32["evaluation_band_nats"] == bf16["evaluation_band_nats"]
    assert f32["measurement_band_nats"] == 2.0**-39 + f32["evaluation_band_nats"]


def test_only_control_evidence_above_the_band_raises_the_floor() -> None:
    band = _rust.kl_measurement_floor("bfloat16", 151_936, 30.0, 0.25, 0.0)["measurement_band_nats"]
    for control in (-1e-15, 0.5 * band):
        assert _rust.kl_measurement_floor("bfloat16", 151_936, 30.0, 0.25, control)["floor_nats"] == band
    assert _rust.kl_measurement_floor("bfloat16", 151_936, 30.0, 0.25, 3.0 * band)["floor_nats"] == 3.0 * band


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("int8", 151_936, 30.0, 0.25, 0.0), "logit format"),
        (("bfloat16", 0, 30.0, 0.25, 0.0), "vocab_size"),
        (("bfloat16", 151_936, -1.0, 0.25, 0.0), "logit_max_abs"),
        (("bfloat16", 151_936, 30.0, math.nan, 0.0), "logit_max_abs_change"),
        (("bfloat16", 151_936, 1.0, 3.0, 0.0), "twice the largest"),
        (("bfloat16", 151_936, 30.0, 0.25, math.inf), "control_nats"),
    ],
)
def test_the_binding_refuses_what_the_owner_cannot_price(
    arguments: tuple[str, int, float, float, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _rust.kl_measurement_floor(*arguments)
