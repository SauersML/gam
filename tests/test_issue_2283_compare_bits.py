"""Strict pairing regressions for the #2283 Eq-4 comparator."""

from __future__ import annotations

import copy
import importlib.util
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


def _pair(compare):
    identity = {
        "schema": compare.PAIR_SCHEMA,
        "run_id": "issue2283-seed0",
        "source": {"code_revision": "a" * 40, "wheel_sha256": "b" * 64},
        "data": {
            "manifest_sha256": "c" * 64,
            "bits_test_positions_sha256": "d" * 64,
            "bits_row_ids_sha256": "e" * 64,
        },
        "config": {"K": 32768, "top_k": 32},
    }

    def record(arm):
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
            "hybrid_phase": "curved-resume" if arm == "hybrid_rust" else None,
            "flat_checkpoint_sha256": "f" * 64 if arm == "hybrid_rust" else None,
        }
        row["pair_identity_sha256"] = compare._payload_sha256(row["pair_identity"])
        return row

    return record("external_topk"), record("hybrid_rust")


def test_authoritative_pair_requires_one_exact_shared_identity():
    compare = _load_compare()
    external, hybrid = _pair(compare)
    assert compare._authoritative_pair(
        [external, hybrid], "issue2283-seed0"
    ) == (external, hybrid)

    mismatched = copy.deepcopy(hybrid)
    mismatched["pair_identity"]["source"]["code_revision"] = "9" * 40
    mismatched["pair_identity_sha256"] = compare._payload_sha256(
        mismatched["pair_identity"]
    )
    with np.testing.assert_raises_regex(ValueError, "different config/provenance"):
        compare._authoritative_pair(
            [external, mismatched], "issue2283-seed0"
        )

    with np.testing.assert_raises_regex(ValueError, "exactly one"):
        compare._authoritative_pair(
            [external, copy.deepcopy(external), hybrid], "issue2283-seed0"
        )


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
