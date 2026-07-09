"""Golden round-trip contract for ``ManifoldSAE`` serialization (issue #2091).

These fixtures are the byte-for-byte output of the *current* Python
``ManifoldSAE.to_dict()`` on a representative model exercising every optional
field (see ``tests/fixtures/manifold_sae/generate_golden.py``). They pin the
on-disk schema so the Rust-owned ``ManifoldSaePayload`` port (and the eventual
Python-dataclass cutover) cannot silently drift a field name, default, or
None-handling and thereby corrupt saved models.

The contract is a *fixed point*: loading a golden payload and re-serializing it
must reproduce the same payload value-for-value. That is exactly the invariant
the Rust serde round-trip (``manifold_sae_payload.rs``) must also satisfy, so
these two suites (Python here, Rust there) share one contract file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gamfit._sae_manifold import ManifoldSAE

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "manifold_sae"
GOLDEN_FULL = FIXTURE_DIR / "golden_full.json"
GOLDEN_COV = FIXTURE_DIR / "golden_with_covariance.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_golden_fixture_exists_and_is_schema_v1() -> None:
    payload = _load(GOLDEN_FULL)
    assert payload["schema"] == "gamfit.ManifoldSAE/v1"
    # The representative model exercises the full optional surface.
    assert len(payload["atoms"]) == 3
    assert payload["fisher_factors"] is not None
    assert payload["structure_certificate"] is not None
    assert payload["selected_log_lambda_smooth"] is not None


def test_to_dict_from_dict_is_a_fixed_point() -> None:
    """to_dict -> from_dict -> to_dict reproduces the golden payload exactly."""
    golden = _load(GOLDEN_FULL)
    model = ManifoldSAE.from_dict(golden)
    again = model.to_dict()
    # Compare as parsed values (key order is irrelevant); pinpoint any drift.
    if again != golden:
        keyset_diff = sorted(set(golden) ^ set(again))
        mismatched = {
            k: (golden.get(k), again.get(k))
            for k in set(golden) & set(again)
            if golden[k] != again[k]
        }
        raise AssertionError(
            f"round-trip drift: key-set-diff={keyset_diff} "
            f"value-mismatch-keys={sorted(mismatched)}"
        )


def test_structured_residual_diagnostics_is_write_dropped() -> None:
    """to_dict never emits structured_residual_diagnostics (from_dict tolerates
    it with a [] default). The Rust serde port mirrors this: Serialize skips the
    field, Deserialize accepts + ignores it."""
    golden = _load(GOLDEN_FULL)
    assert "structured_residual_diagnostics" not in golden
    # A payload that *does* carry the key must still load (read-tolerance).
    with_extra = dict(golden)
    with_extra["structured_residual_diagnostics"] = [{"pass": 0, "lambda_hat": 0.5}]
    model = ManifoldSAE.from_dict(with_extra)
    assert "structured_residual_diagnostics" not in model.to_dict()


def test_reml_score_is_a_duplicate_alias_of_penalized_loss_score() -> None:
    golden = _load(GOLDEN_FULL)
    assert golden["reml_score"] == golden["penalized_loss_score"]
    # from_dict must also accept a legacy dict that carries ONLY reml_score.
    legacy = dict(golden)
    legacy.pop("penalized_loss_score")
    model = ManifoldSAE.from_dict(legacy)
    assert model.penalized_loss_score == golden["reml_score"]


def test_assignment_is_canonicalized_on_load() -> None:
    """Only canonical assignment spellings round-trip stably; a non-canonical
    alias (e.g. "jumprelu") is rewritten by from_dict, so the golden fixture
    stores the canonical form."""
    golden = _load(GOLDEN_FULL)
    assert golden["assignment"] == "topk"  # already canonical
    aliased = dict(golden)
    aliased["assignment"] = "jumprelu"  # canonicalizes to "threshold_gate"
    model = ManifoldSAE.from_dict(aliased)
    assert model.assignment == "threshold_gate"


@pytest.mark.skipif(
    not GOLDEN_COV.exists(),
    reason="covariance-bearing fixture requires a built wheel to (re)generate",
)
def test_covariance_bearing_fixture_round_trips() -> None:
    golden = _load(GOLDEN_COV)
    # Atom 1 carries the compact per-channel covariance factor on disk.
    assert golden["atoms"][1]["decoder_covariance_channel_factors"] is not None
    model = ManifoldSAE.from_dict(golden)
    assert model.to_dict() == golden
