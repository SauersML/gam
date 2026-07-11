"""Golden round-trip contract for ``ManifoldSAE`` serialization (issue #2091).

These fixtures are the exact v3 output of the Rust-owned
``ManifoldSAE.to_dict()`` on a representative model exercising every optional
field (see ``tests/fixtures/manifold_sae/generate_golden.py``). They pin the
on-disk schema so the Rust-owned ``ManifoldSaePayload`` port (and the eventual
PyO3 model cannot silently drift a field name or null representation.

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


def test_golden_fixture_exists_and_is_schema_v3() -> None:
    payload = _load(GOLDEN_FULL)
    assert payload["schema"] == "gamfit.ManifoldSAE/v3"
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


def test_all_fields_are_explicit_and_runtime_diagnostics_round_trip() -> None:
    golden = _load(GOLDEN_FULL)
    assert golden["structured_residual_diagnostics"] == []
    changed = dict(golden)
    changed["structured_residual_diagnostics"] = [{"pass": 0, "lambda_hat": 0.5}]
    assert ManifoldSAE.from_dict(changed).to_dict() == changed
    missing = dict(golden)
    missing.pop("fisher_factors")
    with pytest.raises(ValueError, match="missing field.*fisher_factors"):
        ManifoldSAE.from_dict(missing)


def test_deprecated_score_alias_is_rejected() -> None:
    golden = _load(GOLDEN_FULL)
    assert "reml_score" not in golden
    aliased = dict(golden)
    aliased["reml_score"] = aliased["penalized_loss_score"]
    with pytest.raises(ValueError, match="unknown field.*reml_score"):
        ManifoldSAE.from_dict(aliased)


def test_complete_native_criterion_is_required() -> None:
    golden = _load(GOLDEN_FULL)
    missing = dict(golden)
    missing.pop("penalized_laml_criterion")
    with pytest.raises(ValueError, match="missing field.*penalized_laml_criterion"):
        ManifoldSAE.from_dict(missing)


@pytest.mark.parametrize("obsolete", ["top_k_projection", "pre_topk"])
def test_obsolete_projected_model_payloads_are_rejected(obsolete: str) -> None:
    golden = _load(GOLDEN_FULL)
    aliased = dict(golden)
    aliased[obsolete] = {}
    with pytest.raises(ValueError, match=rf"unknown field.*{obsolete}"):
        ManifoldSAE.from_dict(aliased)


def test_noncanonical_assignment_is_rejected() -> None:
    golden = _load(GOLDEN_FULL)
    assert golden["assignment"] == "topk"
    aliased = dict(golden)
    aliased["assignment"] = "legacy_assignment"
    with pytest.raises(ValueError, match="recognized assignment"):
        ManifoldSAE.from_dict(aliased)


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
