from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qwen_l17_exact_dose_replay.py"
SPEC = importlib.util.spec_from_file_location("qwen_l17_exact_dose_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)

EPS = 2.0**-52
FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.2)
# (atom, base prompt, split, reference dose, gamma, chord-level scatter eta). The
# held-out scatter is symmetric about 0, so the mean r0 is exactly 1 in exact
# arithmetic. Base prompts repeat across atoms: a chord is an (atom, prompt) pair.
CHORDS = (
    (0, "c-a", "calibration", 0.5, 0.8, 0.0),
    (0, "c-b", "calibration", 1.0, -0.4, 0.0),
    (0, "c-c", "calibration", 2.0, 1.2, 0.0),
    (1, "c-a", "calibration", 0.5, 0.3, 0.0),
    (1, "c-d", "calibration", 1.0, -0.9, 0.0),
    (1, "c-e", "calibration", 2.0, 0.6, 0.0),
    (0, "h0", "heldout", 0.6, 0.5, 0.02),
    (0, "h1", "heldout", 0.8, -0.3, -0.02),
    (0, "h2", "heldout", 1.0, 1.1, 0.01),
    (0, "h3", "heldout", 1.2, 0.2, -0.01),
    (0, "h4", "heldout", 1.4, -0.7, 0.03),
    (0, "h5", "heldout", 1.5, 0.9, -0.03),
    (1, "h0", "heldout", 0.6, -0.2, 0.03),
    (1, "h1", "heldout", 0.8, 0.4, -0.03),
    (1, "h2", "heldout", 1.0, -1.0, 0.02),
    (1, "h3", "heldout", 1.2, 0.7, -0.02),
    (1, "h4", "heldout", 1.4, 0.1, 0.01),
    (1, "h5", "heldout", 1.5, -0.6, -0.01),
)


def _protocol(row_count: int) -> dict[str, object]:
    return {
        "model": "Qwen3.6-35B-A3B",
        "model_revision": "frozen-revision",
        "model_sha256": "model-hash",
        "layer": 17,
        "hook_module": "model.layers.17",
        "steering_mode": "steer_to_target_along_the_fitted_chart",
        "model_dtype": "bfloat16",
        "harvest_dtype": "float32",
        "gam_sha": "gam-sha",
        "wheel_sha256": "wheel-hash",
        "driver_sha256": "driver-hash",
        "prompt_bank_sha256": "prompt-hash",
        "harvest_cache_sha256": "cache-hash",
        "seed": 0,
        "fractions": list(FRACTIONS),
        "floor_multiplier": 30,
        "floor_repetitions": 5,
        "max_templates": 6,
        "bases": 10,
        "fit_iterations": 40,
        "router_modules": 40,
        "row_count": row_count,
    }


def _row(
    atom: int,
    prompt: str,
    split: str,
    fraction_index: int,
    *,
    predicted: float,
    measured: float,
) -> dict[str, object]:
    return {
        "intervention_id": f"{atom}-{prompt}-f{fraction_index}",
        "atom": atom,
        "base_prompt_id": prompt,
        "split": split,
        "fraction_index": fraction_index,
        "predicted_nats": predicted,
        "predicted_nats_kind": "exact_directional",
        "exact_directional_nats": predicted,
        "measured_nats": measured,
        "effective_delta": [predicted, -predicted],
        "resident_metric_nats": predicted * 0.4,
        "resident_metric_nats_kind": "uncertified_approximation",
        "router_topk_changes": 0,
    }


def _ledger(scale: float = 1.0, heldout_eta: tuple[float, ...] | None = None) -> dict[str, object]:
    """A meter exact to leading order: ``m = q (1 + γ √q)(1 + η)``, times ``scale``.

    ``heldout_eta`` replaces the held-out chords' scatter, in ``CHORDS`` order.
    """
    chords = list(CHORDS)
    if heldout_eta is not None:
        heldout = [index for index, chord in enumerate(chords) if chord[2] == "heldout"]
        assert len(heldout_eta) == len(heldout)
        for index, eta in zip(heldout, heldout_eta):
            chords[index] = chords[index][:5] + (eta,)
    rows = [
        _row(
            atom,
            prompt,
            split,
            k,
            predicted=reference * fraction,
            measured=scale
            * (reference * fraction)
            * (1.0 + gamma * math.sqrt(reference * fraction))
            * (1.0 + eta),
        )
        for atom, prompt, split, reference, gamma, eta in chords
        for k, fraction in enumerate(FRACTIONS)
    ]
    return {"protocol": _protocol(len(rows)), "rows": rows}


def _ratio_rounding(reference: float, gamma: float, eta: float) -> tuple[list[float], float]:
    """The fixture's three smallest sqrt-doses of a chord and a bound on each ratio's rounding.

    A measurement and its ratio take eight roundings of at most EPS relative each.
    """
    x = [math.sqrt(reference * fraction) for fraction in FRACTIONS[:3]]
    ratio = max(abs((1.0 + gamma * value) * (1.0 + eta)) for value in x)
    return x, 8.0 * EPS * ratio


def _report(ledger: dict[str, object]) -> dict[str, object]:
    return replay.acceptance_report(ledger, bootstrap_draws=400, seed=2249)


def _refresh(ledger: dict[str, object]) -> dict[str, object]:
    ledger["protocol"]["row_count"] = len(ledger["rows"])
    return ledger


def test_r0_is_the_zero_dose_limit_along_the_square_root_of_the_dose() -> None:
    # m/q = 1 + 5 sqrt(q) exactly, at q = 0.01 and 0.04 (x = 0.1, 0.2). The line in
    # sqrt(q) meets 1 at zero dose; each ratio carries a few roundings, amplified by
    # the extrapolation lever x2/(x2 - x1).
    window = [
        _row(0, "p", "heldout", k, predicted=q, measured=q * (1.0 + 5.0 * math.sqrt(q)))
        for k, q in enumerate((0.01, 0.04))
    ]
    extrapolation = replay._chord_extrapolation(window)
    bound = 8 * EPS * (1.0 + 5.0 * 0.2) * 0.2 / (0.2 - 0.1)
    assert abs(extrapolation["r0"] - 1.0) <= bound
    assert abs(extrapolation["gamma"] - 5.0) <= bound / 0.1
    assert extrapolation["r0_truncation"] is None
    # The same two ratios extrapolated along the dose itself miss the limit by a
    # third: the abscissa is sqrt(q), not q.
    r1, r2 = 1.0 + 5.0 * 0.1, 1.0 + 5.0 * 0.2
    along_the_dose = r1 - 0.01 * (r2 - r1) / (0.04 - 0.01)
    assert abs(along_the_dose - 4.0 / 3.0) <= 8 * EPS


def test_an_unbiased_meter_is_accepted_and_reports_its_resolution() -> None:
    report = _report(_ledger())
    assert report["verdict"] == "PASS" and report["accepted"]
    assert report["unresolved_against"] == []
    lo, hi = report["r0_chord_bootstrap_95_ci"]
    assert lo < 1.0 < hi
    assert report["r0_ci_half_width"] == 0.5 * (hi - lo) > 0.0
    assert report["r0_resolution"] == {
        "upward_mis_scale": 1.0 / lo - 1.0,
        "downward_mis_scale": 1.0 - 1.0 / hi,
    }
    # The repeated prompt ids belong to two atoms and therefore to two chords each.
    assert len(report["heldout_chords"]) == 12
    # Each chord's ratio is exactly linear in sqrt(q), so the third dose sees no
    # curvature beyond rounding: the second divided difference of the rounded
    # ratios, times x1 x2.
    for atom, prompt, split, reference, gamma, eta in CHORDS:
        if split != "heldout":
            continue
        x, dr = _ratio_rounding(reference, gamma, eta)
        curvature = (2 * dr / (x[2] - x[1]) + 2 * dr / (x[1] - x[0])) / (x[2] - x[0])
        truncation = report["heldout_chords"][f"{atom}:{prompt}"]["r0_truncation"]
        assert abs(truncation) <= curvature * x[0] * x[1]
    # Calibration chords publish the error law: gamma is each chord's slope, up to
    # the rounding of two ratios over the first sqrt-dose gap.
    calibration = [chord for chord in CHORDS if chord[0] == 0 and chord[2] == "calibration"]
    slope_rounding = max(
        2 * dr / (x[1] - x[0]) for x, dr in (_ratio_rounding(*chord[3:]) for chord in calibration)
    )
    gamma_mean = sum(chord[4] for chord in calibration) / len(calibration)
    assert abs(report["by_atom"]["atom_0"]["gamma_mean"] - gamma_mean) <= slope_rounding + 4 * EPS
    assert report["by_atom"]["atom_0"]["calibrated_region_nats"] == [0.5 * 0.01, 2.0 * 0.2]


def test_a_mis_scale_just_beyond_the_reported_resolution_is_refused() -> None:
    # r0 is linear in the measurements and the bootstrap draws do not read them, so
    # scaling every measurement by c scales the interval to c times itself. The
    # reported resolution is therefore exact: a mis-scale past it is refused and one
    # short of it is accepted. The margin is the rounding of the scaled pipeline: a
    # few roundings per ratio amplified by this ladder's lever x1/(x2 - x1) =
    # 1/(sqrt 2 - 1), one per chord in the mean, and a few in the quantile.
    report = _report(_ledger())
    lo, hi = report["r0_chord_bootstrap_95_ci"]
    lever = 1.0 / (math.sqrt(2.0) - 1.0)
    margin = (4.0 * (1.0 + 2.0 * lever) + len(report["heldout_chords"]) + 8.0) * EPS

    def verdict(scale: float) -> str:
        return str(_report(_ledger(scale))["verdict"])

    assert verdict((1.0 / lo) * (1.0 + margin)) == "FAIL"
    assert verdict((1.0 / lo) * (1.0 - margin)) == "PASS"
    assert verdict((1.0 / hi) * (1.0 - margin)) == "FAIL"
    assert verdict((1.0 / hi) * (1.0 + margin)) == "PASS"
    # A meter mis-scaled like the historical defects is refused outright.
    assert verdict(0.541) == "FAIL"
    assert verdict(4.30) == "FAIL"


def test_a_ledger_too_noisy_to_tell_the_meter_from_a_known_defect_is_unresolved() -> None:
    # Scatter of +1.6, -0.8, -0.8 per chord still averages to an unbiased meter, and
    # the interval contains 1, but it also covers the path-integrated and first-order
    # ratios #2249 measured. Containing 1 alone would pass it vacuously; the ledger
    # needs more rows, and it does not pass.
    report = _report(_ledger(heldout_eta=(1.6, -0.8, -0.8) * 4))
    lo, hi = report["r0_chord_bootstrap_95_ci"]
    assert lo <= 1.0 <= hi
    assert report["verdict"] == "UNRESOLVED"
    assert report["unresolved_against"] == ["path_integrated_2249", "tangent_first_order_2249"]
    assert not report["accepted"]


def test_doses_past_a_router_topk_change_do_not_enter_r0() -> None:
    clean = _report(_ledger())
    ledger = _ledger()
    # Chord (0, h1): the second dose flips a router's expert set, and from there on
    # the readout is ten times the meter. Only the first dose lies before the
    # change, so the chord has no two smooth doses and is not scored at all.
    for row in ledger["rows"]:
        if (row["atom"], row["base_prompt_id"]) == (0, "h1") and row["fraction_index"] >= 1:
            row["measured_nats"] *= 10.0
            if row["fraction_index"] == 1:
                row["router_topk_changes"] = 3
    report = _report(ledger)
    assert report["heldout_chords_unresolved"] == ["0:h1"]
    assert "0:h1" not in report["heldout_chords"]
    assert report["heldout_past_router_change"] == [f"0-h1-f{k}" for k in range(1, 5)]
    others = [chord["r0"] for key, chord in clean["heldout_chords"].items() if key != "0:h1"]
    assert report["r0_mean"] == float(np.mean(np.asarray(others)))


def test_heldout_doses_outside_the_calibrated_region_are_not_scored() -> None:
    clean = _report(_ledger())
    ledger = _ledger()
    # Atom 1's calibration occupies [0.005, 0.4]. A held-out dose below it would be
    # chord (1, h2)'s smallest and would set its r0; one above it lies past every
    # calibration dose. Neither was measured by the calibration, and both read ten
    # times the meter.
    ledger["rows"].append(_row(1, "h2", "heldout", 90, predicted=0.001, measured=0.01))
    ledger["rows"].append(_row(1, "h2", "heldout", 91, predicted=1.0, measured=10.0))
    report = _report(_refresh(ledger))
    assert report["heldout_outside_calibrated_region"] == ["1-h2-f90", "1-h2-f91"]
    assert report["heldout_chords"]["1:h2"] == clean["heldout_chords"]["1:h2"]
    assert report["r0_chord_bootstrap_95_ci"] == clean["r0_chord_bootstrap_95_ci"]


def test_the_calibrated_region_never_uses_heldout_measurements() -> None:
    first = _report(_ledger())
    ledger = _ledger()
    for row in ledger["rows"]:
        if row["split"] == "heldout":
            row["measured_nats"] *= 1000.0
    second = _report(ledger)
    assert first["by_atom"]["atom_0"]["calibrated_region_nats"] == second["by_atom"]["atom_0"]["calibrated_region_nats"]
    assert first["by_atom"]["atom_1"]["gamma_mean"] == second["by_atom"]["atom_1"]["gamma_mean"]
    assert second["verdict"] == "FAIL"


def test_acceptance_rejects_non_public_or_unstable_rows() -> None:
    ledger = _ledger()
    ledger["rows"][0]["predicted_nats_kind"] = "uncertified_approximation"
    with pytest.raises(ValueError, match="is not exact"):
        _report(ledger)

    ledger = _ledger()
    ledger["rows"][1]["intervention_id"] = ledger["rows"][0]["intervention_id"]
    with pytest.raises(ValueError, match="duplicate intervention_id"):
        _report(ledger)

    for value in (None, True, -1, 0.0):
        ledger = _ledger()
        ledger["rows"][0]["router_topk_changes"] = value
        with pytest.raises(ValueError, match="router_topk_changes"):
            _report(ledger)
