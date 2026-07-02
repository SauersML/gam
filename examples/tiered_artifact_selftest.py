"""Self-contained regression check for the tiered-artifact contract (WS-J).

Exercises the invariants the other workstreams depend on, with no fitter and no
Rust build: save→load hash stability, T2 canonical-hash order/scale/reflection
invariance (the ``dictionary_artifact.rs`` mirror), tamper detection, the WS-D
finalized-T0 round-trip (nested ``rogue_dims``), and the ``diff`` self-identity.
Run: ``python examples/tiered_artifact_selftest.py`` — exits non-zero on any
failure so it can gate the contract in CI without a pytest dependency.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiered_artifact import TieredArtifact, load_t0_from_manifest, t2_dictionary_hash  # noqa: E402


def _check(name: str, cond: bool) -> None:
    print(f"[{'ok' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(f"selftest failed: {name}")


def main() -> None:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        # 1. save → load content-hash stability across every present tier.
        art = TieredArtifact(
            t0={"mean": rng.standard_normal(8).tolist(),
                "norm": np.abs(rng.standard_normal(8)).tolist(),
                "d_model": 8, "total_tokens": 1000, "scale": 1.3, "rogue_dims": [0, 3]},
            t1_decoder=rng.standard_normal((6, 8)).astype(np.float32),
            t2_atoms=[{"topology": "circle", "frame": rng.standard_normal((3, 2)), "residual_gauge": "O(2)"},
                      {"topology": "circle", "frame": rng.standard_normal((3, 2)), "residual_gauge": "O(2)"}],
            t2_meta={"ev_trace": [0.1, 0.3, 0.42], "k": 2, "gauge_certificate": "O(2)"})
        d = os.path.join(tmp, "a")
        h1 = art.save(d)
        reloaded = TieredArtifact.load(d)
        _check("save==reload content hash", h1 == reloaded.content_hash())
        _check("no spurious tamper flag on clean reload",
               reloaded.provenance.get("hash_mismatch") is None)

        # 2. T2 hash invariance to atom order / scale / reflection (Rust mirror).
        base = [{"topology": "circle", "frame": np.array([[2.0], [0.0]]), "residual_gauge": "g"},
                {"topology": "circle", "frame": np.array([[0.0], [3.0]]), "residual_gauge": "g"}]
        perturbed = [{"topology": "circle", "frame": np.array([[0.0], [-9.0]]), "residual_gauge": "g"},
                     {"topology": "circle", "frame": np.array([[-4.0], [0.0]]), "residual_gauge": "g"}]
        _check("T2 hash invariant to order/scale/reflection",
               t2_dictionary_hash(base, "c") == t2_dictionary_hash(perturbed, "c"))
        _check("T2 hash changes on a real decoder-row perturbation",
               t2_dictionary_hash(base, "c") != t2_dictionary_hash(
                   [{"topology": "circle", "frame": np.array([[2.0], [0.2]]), "residual_gauge": "g"},
                    base[1]], "c"))

        # 3. tamper detection: corrupt the T1 array file → recorded mismatch.
        np.save(os.path.join(d, "t1_decoder.npy"),
                rng.standard_normal((6, 8)).astype(np.float32))
        tampered = TieredArtifact.load(d)
        _check("tampered T1 file trips hash_mismatch",
               tampered.provenance.get("hash_mismatch") is not None)

        # 4. WS-D finalized-T0 round-trip (nested rogue_dims / scale_median_std).
        import json
        fin = {"format": "residual_shard_bf16", "d_model": 5, "total_tokens": 10, "provisional": False,
               "t0": {"d_model": 5, "mean": [0.0] * 5, "std": [1.0] * 5, "rms": [1.0] * 5,
                      "scale_median_std": 1.0, "scale_median_rms": 1.0,
                      "rogue_dims": {"index": [1, 4], "rms": [9.0, 8.0],
                                     "rms_over_median": [9.0, 8.0], "mad_z": [40.0, 35.0],
                                     "rule": "rms>5*median_rms OR MAD-z>8"}}}
        mp = os.path.join(tmp, "manifest.json")
        json.dump(fin, open(mp, "w"))
        t0 = load_t0_from_manifest(mp)
        _check("finalized T0 rogue_dims passthrough", t0["rogue_dims"]["index"] == [1, 4])
        art2 = TieredArtifact(t0=t0, t1_decoder=np.eye(5, dtype=np.float32))
        d2 = os.path.join(tmp, "b")
        _check("finalized-T0 artifact hash round-trips",
               art2.save(d2) == TieredArtifact.load(d2).content_hash())

        # 5. diff self-identity.
        from dictionary_cli import diff_artifacts
        dd = diff_artifacts(d2, d2)
        _check("self-diff hash_equal", dd["hash_equal"] is True)

    print("\nALL TIERED-ARTIFACT SELFTESTS PASSED")


if __name__ == "__main__":
    main()
