"""#1103 Python round-trip: the per-atom split-LRT smooth-structure e-value.

The Rust SAE fit emits, for every fitted atom, an ``atom_inference`` report
whose ``smooth_significance.log_e_nonconstant`` is the any-n-valid
split-likelihood-ratio e-value for "the atom's inner decoder smooth is
non-constant" (null = constant) — the same universal-inference instrument the
atom-birth gate uses, honest at the ``df ≈ n`` regime (``E_{H0}[E] <= 1``).

This test pins that the e-value actually ROUND-TRIPS to a Python caller: it is
computed during a normal ``gamfit.sae_manifold_fit`` (no opt-in flag — magic by
default), surfaced through ``ManifoldSAE.atom_inference()``, and survives a
``save``/``load`` JSON round-trip. Before #1103's Python wiring the value was
emitted by the FFI but dropped by the ``ManifoldSAE`` wrapper, so a caller could
never see it — this guards against that regression.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _circle_data(n: int, p: int, noise: float, seed: int) -> np.ndarray:
    """A single circular harmonic mixed into ``p`` output dims: a genuinely
    curved (non-constant) 1-atom truth, so the smooth-structure e-value has real
    non-constant signal to report."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    harm = np.column_stack([np.cos(theta), np.sin(theta)])
    mixing = rng.normal(size=(harm.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = harm @ mixing + noise * rng.normal(size=(n, p))
    z -= z.mean(axis=0, keepdims=True)
    return z


def _fit_one_atom_periodic(seed: int = 0):
    z = _circle_data(n=300, p=48, noise=0.04, seed=seed)
    return gamfit.sae_manifold_fit(
        X=z,
        K=1,
        atom_basis="periodic",
        d_atom=2,
        assignment="ibp_map",
        n_iter=50,
        learning_rate=0.04,
        random_state=seed,
    )


def test_atom_inference_surfaces_log_e_nonconstant():
    fit = _fit_one_atom_periodic()

    # The accessor exists and returns one report per fitted atom (magic: it is
    # populated by the normal fit, with no flag to opt into).
    assert hasattr(fit, "atom_inference"), (
        "ManifoldSAE must expose atom_inference(); the #1103 split-LRT "
        "smooth-structure e-value is otherwise unreachable from Python"
    )
    reports = fit.atom_inference()
    assert isinstance(reports, list) and len(reports) == len(fit.atoms), (
        f"atom_inference() must report one entry per atom; got {len(reports)} "
        f"for {len(fit.atoms)} atoms"
    )

    report = reports[0]
    assert "smooth_significance" in report, (
        "per-atom inference report must carry a smooth_significance block "
        f"(keys present: {sorted(report)})"
    )
    sig = report["smooth_significance"]
    assert sig is not None, (
        "a harvested curved atom must carry a populated smooth_significance "
        "block, not None"
    )
    assert "log_e_nonconstant" in sig, (
        "smooth_significance must carry the #1103 log_e_nonconstant e-value "
        f"(keys present: {sorted(sig)})"
    )
    log_e = sig["log_e_nonconstant"]
    assert log_e is not None and math.isfinite(float(log_e)), (
        f"log_e_nonconstant must be a finite e-value; got {log_e!r}"
    )
    # A genuinely curved (circular-harmonic) atom carries POSITIVE any-n-valid
    # evidence for the non-constant alternative — the honest discrimination
    # signal, not merely a populated field.
    assert float(log_e) > 0.0, (
        "a circular-harmonic (strongly non-constant) atom must accumulate "
        f"positive split-LRT smooth-structure evidence; got log_e={log_e}"
    )


def test_atom_inference_survives_save_load_roundtrip(tmp_path):
    fit = _fit_one_atom_periodic(seed=1)
    before = fit.atom_inference()
    log_e_before = before[0]["smooth_significance"]["log_e_nonconstant"]

    path = tmp_path / "sae_fit.json"
    fit.save(path)
    reloaded = gamfit.load(path)

    after = reloaded.atom_inference()
    assert len(after) == len(before), (
        "atom_inference() length must be preserved across save/load"
    )
    log_e_after = after[0]["smooth_significance"]["log_e_nonconstant"]
    assert log_e_after is not None and math.isclose(
        float(log_e_after), float(log_e_before), rel_tol=0.0, abs_tol=1e-9
    ), (
        "the #1103 log_e_nonconstant e-value must survive a JSON save/load "
        f"round-trip exactly; before={log_e_before}, after={log_e_after}"
    )
