"""Strict pairing regressions for the #2283 Eq-4 comparator."""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import numpy as np


def _load_compare():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "1026_close"
        / "compare_bits.py"
    )
    spec = importlib.util.spec_from_file_location("issue_2283_compare", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GROUPS = 20
N_ROWS = 24_000
TARGETS = ("0.99", "0.95", "0.9", "0.8")


def _scorer_record(support, code, residual, dictionary, prefix):
    """One scorer output, total reconciled with its components at every target."""
    record = {f"{prefix}support_bits": support, f"{prefix}dictionary_bits": dictionary}
    for offset, target in enumerate(TARGETS):
        record[f"{prefix}code_bits_at_r2_{target}"] = code - offset
        record[f"{prefix}resid_bits_at_r2_{target}"] = residual - 10 * offset
        record[f"{prefix}bits_at_r2_{target}"] = (
            support + (code - offset) + (residual - 10 * offset) + dictionary
        )
    return record


def _rows(compare, residual=(2714.0, 2600.0, 2590.0), bias=(-60.0, -58.0, -57.0)):
    """External, control and hybrid rows of one run, in ``ARMS`` order.

    Each arm's residual term carries a plug-in bias ``bias[arm]`` at the full
    ``N_ROWS``, scaled as ``1/n`` in its deleted-block records, plus a zero-sum
    per-block wobble that sets its standard error.
    """
    identity = {
        "schema": compare.PAIR_SCHEMA,
        "run_id": "issue2283-seed0",
        "source": {"code_revision": "a" * 40, "wheel_sha256": "b" * 64},
        "data": {
            "manifest_sha256": "c" * 64,
            "bits_test_positions_sha256": "d" * 64,
            "bits_row_ids_sha256": "e" * 64,
        },
        # The v3 pair schema this fixture declares carries the DECLARED
        # amortization horizon, separately from the bits estimation subsample, and
        # the jackknife's group count; the comparator rejects a row without either,
        # so a fixture that claims v3 has to supply both or the pairing assertions
        # below never reach the pairing logic they are about.
        "config": {
            "K": 32768,
            "top_k": 32,
            "amortization_horizon": 120_000,
            "jackknife_groups": GROUPS,
        },
    }
    supports = {"external_topk": 377.18, "gam_flat": 377.18, "hybrid_rust": 357.05}
    wobble = np.sin(np.arange(GROUPS) + 1.0)
    wobble -= wobble.mean()

    def record(arm, index):
        truth = residual[index] - bias[index]
        deleted_rows = N_ROWS - N_ROWS // GROUPS
        row = {
            "issue": 2283,
            "record_type": "result",
            "arm": arm,
            "run_id": identity["run_id"],
            "pair_identity": copy.deepcopy(identity),
            "data_manifest": {"sha256": identity["data"]["manifest_sha256"]},
            "bits_test_positions_sha256": identity["data"]["bits_test_positions_sha256"],
            "bits_row_ids_sha256": identity["data"]["bits_row_ids_sha256"],
            "bits_dict_params_faithful": True,
            "bits_amortization_horizon": identity["config"]["amortization_horizon"],
            "bits_rows": N_ROWS,
            "p": 2048,
            "ev": 0.9,
            "K": 32768,
            "top_k": 32,
            "k_flat": 32672,
            "curved_atoms": 32,
            "curved_k": 2,
            "d_atom": 1,
            "sparse_score_mode": None if arm == "external_topk" else "off",
            "sparse_minibatch": None if arm == "external_topk" else 8192,
            "hybrid_phase": "curved-resume" if arm == "hybrid_rust" else None,
            "flat_checkpoint_sha256": "f" * 64 if arm == "hybrid_rust" else None,
            **_scorer_record(supports[arm], 150.0, residual[index], 4717.94, "bits_"),
        }
        row["bits_jackknife"] = {
            "groups": GROUPS,
            "block_sizes": [N_ROWS // GROUPS] * GROUPS,
            "block_positions": "9" * 64,
            "leave_one_out": [
                _scorer_record(
                    supports[arm],
                    150.0,
                    truth + bias[index] * N_ROWS / deleted_rows + (index + 1) * wobble[block],
                    4717.94,
                    "",
                )
                for block in range(GROUPS)
            ],
        }
        row["pair_identity_sha256"] = compare._payload_sha256(row["pair_identity"])
        return row

    return tuple(record(arm, index) for index, arm in enumerate(compare.ARMS))


def test_authoritative_pair_requires_one_exact_shared_identity():
    compare = _load_compare()
    external, control, hybrid = _rows(compare)
    assert compare._authoritative_pair(
        [external, control, hybrid], "issue2283-seed0"
    ) == (external, control, hybrid)

    mismatched = copy.deepcopy(hybrid)
    mismatched["pair_identity"]["source"]["code_revision"] = "9" * 40
    mismatched["pair_identity_sha256"] = compare._payload_sha256(
        mismatched["pair_identity"]
    )
    with np.testing.assert_raises_regex(ValueError, "different config/provenance"):
        compare._authoritative_pair(
            [external, control, mismatched], "issue2283-seed0"
        )

    with np.testing.assert_raises_regex(ValueError, "exactly one"):
        compare._authoritative_pair(
            [external, copy.deepcopy(external), control, hybrid], "issue2283-seed0"
        )

    # The paired control is required: an external bar alone confounds the curved
    # tier with the difference between two trainers.
    with np.testing.assert_raises_regex(ValueError, "exactly one 'gam_flat'"):
        compare._authoritative_pair([external, hybrid], "issue2283-seed0")


def test_control_must_share_the_hybrid_flat_tiers_trainer_settings():
    compare = _load_compare()
    external, control, hybrid = _rows(compare)
    other_mode = copy.deepcopy(control)
    other_mode["sparse_score_mode"] = "required"
    with np.testing.assert_raises_regex(ValueError, "share one trainer setting"):
        compare._authoritative_pair([external, other_mode, hybrid], "issue2283-seed0")


def test_every_row_must_carry_the_identitys_jackknife_over_one_partition():
    compare = _load_compare()
    external, control, hybrid = _rows(compare)
    short = copy.deepcopy(hybrid)
    short["bits_jackknife"]["leave_one_out"].pop()
    with np.testing.assert_raises_regex(ValueError, "deleted blocks"):
        compare._authoritative_pair([external, control, short], "issue2283-seed0")
    other_blocks = copy.deepcopy(control)
    other_blocks["bits_jackknife"]["block_positions"] = "8" * 64
    with np.testing.assert_raises_regex(ValueError, "different row blocks"):
        compare._authoritative_pair([external, other_blocks, hybrid], "issue2283-seed0")


def test_paired_jackknife_removes_a_first_order_bias_exactly():
    """Plug-in values that sit a/n off the truth at every sample size.

    The residual term of each arm is truth + bias·N/n over n scored rows, plus a
    zero-sum wobble across deleted blocks. The corrected paired difference must be
    the difference of the truths, to rounding, and its standard error the
    jackknife's ``sqrt((J-1)/J · Σ (d_j - mean)²)`` of the wobble difference.
    """
    compare = _load_compare()
    residual = (2714.0, 2600.0, 2590.0)
    bias = (-60.0, -58.0, -57.0)
    external, control, hybrid = _rows(compare, residual=residual, bias=bias)
    estimate = compare.paired_difference(hybrid, control, "0.99")["residual"]
    truth = (residual[2] - bias[2]) - (residual[1] - bias[1])
    assert estimate.plug_in == residual[2] - residual[1]
    # J·B and (J-1)·mean(B_(j)) each carry up to J·ε relative rounding of values
    # up to J·max|B|.
    bar = 8 * GROUPS**2 * np.finfo(np.float64).eps * max(abs(value) for value in residual)
    assert abs(estimate.corrected - truth) <= bar, (estimate.corrected, truth, bar)
    wobble = np.sin(np.arange(GROUPS) + 1.0)
    wobble -= wobble.mean()
    expected_se = math.sqrt((GROUPS - 1) / GROUPS * float(((3 - 2) ** 2 * wobble**2).sum()))
    assert abs(estimate.standard_error - expected_se) <= bar, (
        estimate.standard_error,
        expected_se,
    )
    # The support, code and dictionary terms carry no bias and no wobble here, so
    # their paired differences are exact with zero standard error.
    for name, value in (("support", 357.05 - 377.18), ("code", 0.0), ("dictionary", 0.0)):
        component = compare.paired_difference(hybrid, control, "0.99")[name]
        assert abs(component.corrected - value) <= bar, (name, component)
        assert component.standard_error <= bar, (name, component)


def test_verdict_reads_the_corrected_difference_against_two_standard_errors():
    compare = _load_compare()
    jackknife = compare._sibling("eq4_jackknife")
    estimate = jackknife.JackknifeEstimate
    assert compare.verdict(estimate(0.0, -3.0, 1.0)) == "hybrid shorter"
    assert compare.verdict(estimate(0.0, 3.0, 1.0)) == "hybrid longer"
    assert compare.verdict(estimate(-5.0, -1.9, 1.0)) == "unresolved at this n"
    assert compare.verdict(estimate(-5.0, 1.9, 1.0)) == "unresolved at this n"


def test_the_measurement_exits_zero_whichever_arm_is_shorter(tmp_path, monkeypatch, capsys):
    compare = _load_compare()
    for residual in ((2714.0, 2600.0, 2590.0), (2714.0, 2600.0, 2650.0)):
        rows = _rows(compare, residual=residual)
        path = tmp_path / f"rows-{residual[2]}.jsonl"
        path.write_text("\n".join(compare._canonical_json(row) for row in rows) + "\n")
        monkeypatch.setattr(
            "sys.argv", ["compare_bits.py", str(path), "--run-id", "issue2283-seed0"]
        )
        assert compare.main() == 0
        printed = capsys.readouterr().out
        assert "VERDICT R2=0.99 vs gam_flat: " in printed
        assert "VERDICT R2=0.99 vs external: " in printed


def test_component_reconciliation_includes_dictionary_bits():
    compare = _load_compare()
    row = {
        "arm": "external_topk",
        "bits_support_bits": 3.0,
        "bits_code_bits_at_r2_0.99": 5.0,
        "bits_resid_bits_at_r2_0.99": 7.0,
        "bits_dictionary_bits": 11.0,
        "bits_bits_at_r2_0.99": 26.0,
    }
    assert compare._components(row, "0.99") == (
        26.0,
        {"support": 3.0, "code": 5.0, "residual": 7.0, "dictionary": 11.0},
    )
    row["bits_bits_at_r2_0.99"] = 25.0
    with np.testing.assert_raises_regex(ValueError, "does not reconcile"):
        compare._components(row, "0.99")


def _load_theorem():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "1026_close"
        / "crossover_theorem_check.py"
    )
    spec = importlib.util.spec_from_file_location("issue_2283_theorem", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_support_code_bits_is_a_kraft_complete_cardinality_then_subset_code():
    theorem = _load_theorem()
    eps = np.finfo(np.float64).eps
    for atoms in range(1, 9):
        bar = 8 * (atoms + 1) * eps
        kraft = sum(
            math.comb(atoms, k) * 2.0 ** -theorem.support_code_bits(atoms, k)
            for k in range(atoms + 1)
        )
        assert abs(kraft - 1.0) <= bar, (atoms, kraft)
        # The retired price, `log2 C(G, k)` with no cardinality term, sums to
        # `G + 1` over the same supports, so the Kraft sum separates the two.
        retired = sum(
            math.comb(atoms, k) * 2.0 ** -theorem.selection_bits(atoms, k)
            for k in range(atoms + 1)
        )
        assert abs(retired - (atoms + 1)) <= bar, (atoms, retired)


def test_predicted_support_margin_prices_each_config_with_the_scorers_support_code():
    compare = _load_compare()
    hybrid = {
        "K": 32768,
        "top_k": 32,
        "curved_k": 2,
        "d_atom": 1,
        "k_flat": 32672,
        "curved_atoms": 32,
    }
    paired_atoms = hybrid["k_flat"] + hybrid["curved_atoms"]
    paired_cardinality = (
        hybrid["top_k"] - hybrid["curved_k"] * (1 + hybrid["d_atom"]) + hybrid["curved_k"]
    )
    external_code = math.log2(hybrid["K"] + 1) + math.log2(
        math.comb(hybrid["K"], hybrid["top_k"])
    )
    paired_code = math.log2(paired_atoms + 1) + math.log2(
        math.comb(paired_atoms, paired_cardinality)
    )
    # `selection_bits` sums `top_k` rounded log terms, each below the total.
    bar = 4 * hybrid["top_k"] * np.finfo(np.float64).eps * external_code
    predicted = compare._predicted_support_margin(hybrid)
    assert abs(predicted - (external_code - paired_code)) <= bar, (
        predicted,
        external_code - paired_code,
    )
    # The cardinality term the retired `log2 C(G, L0)` margin omitted is resolvable
    # at this bar, so a margin without it fails the assertion above.
    assert abs(math.log2(hybrid["K"] + 1) - math.log2(paired_atoms + 1)) > bar
