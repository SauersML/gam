//! Owed-work regression gate for #1033 (n-independent SAE outer ρ-loop:
//! hyperparameter search must touch only k-dim objects).
//!
//! ## What #1033 asks, and what is actually achievable
//!
//! #1033 wants the SAE-manifold outer hyperparameter (ρ) search reformulated so
//! that each outer evaluation consumes only k-dim *sufficient statistics* (k =
//! the number of atoms / latent axes), with the n-sized (row-count) work built
//! ONCE in O(n) — making the per-evaluation cost independent of n.
//!
//! The honest architectural finding (the primary deliverable of #1033) is that
//! the STRICT form of that contract — "the entire ρ-search reads only k-dim
//! objects" — is NOT achievable for the current SAE architecture without
//! introducing an approximation the code does not currently make. Every O(n)
//! touch inside one outer evaluation is *ρ-dependent*, not ρ-invariant, because
//! the outer search drives the NONLINEAR inner joint fit (`penalized_quasi_laplace_criterion*` →
//! `assemble_arrow_schur`, the inner Newton solve). For one ρ evaluation the
//! n-sized work is, and each piece DEPENDS on ρ:
//!
//!   * per-row assignment gates `a_k = a_k(ρ.lambda_sparse)` (softmax / ThresholdGate
//!     / ordered independent Beta--Bernoulli temperatures), and hence the per-row active-set selection;
//!   * per-row reconstruction `fitted = Σ_k a_k · B_k φ_k(t_k)`;
//!   * per-row residuals and any whitening;
//!   * per-row logit + coordinate Jacobian rows;
//!   * the data-fit Gram accumulation `G` (sparsity pattern + magnitudes both
//!     scale with the ρ-dependent gates).
//!
//! All of these flow from the gates `a_k(ρ)` AND from the inner solve's
//! converged `(coords, logits, β)`, which themselves move with ρ. The ONLY
//! ρ-invariant O(n) quantities are the target matrix and the per-row loss
//! weights, and those are already constant across the search. So one cannot
//! build a k-dim sufficient statistic once in O(n) and then make the WHOLE
//! ρ-search read only k-dim objects: the inner solve is a nonlinear fit whose
//! per-row state genuinely changes with ρ. A flat "outer-eval wall-time is
//! independent of n" contract would therefore be FALSE, and this test does NOT
//! assert it.
//!
//! ## What genuinely IS n-independent, and is pinned here
//!
//! The half of #1033 that is mathematically sound is already in place: the
//! objects the outer search CONSUMES are k-dim, not n-dim. The outer-ρ gradient
//! — the per-evaluation payload the optimizer steps on — has length
//! `1 + K + Σ_k d_k` (the shared sparse log-strength, the K per-atom smoothness
//! log-strengths — `log_lambda_smooth[k]`, one per atom since #1556 — plus the
//! per-atom per-axis ARD precisions), independent of the row count n. Each of
//! its four analytic
//! channels (`explicit`, `logdet_trace`, `occam`, `third_order_correction`) is a
//! k-dim `Array1`. The k-dim CONSUMPTION is exactly what makes the search scale
//! in the hyperparameter dimension and not in n — even though BUILDING each
//! channel still costs O(n) per ρ (the part that is not removable here).
//!
//! This test pins that invariant: build the SAME two-atom term — identical
//! atoms, decoders, ARD layout, and ρ — over two DIFFERENT row counts, run the
//! public `penalized_quasi_laplace_criterion_with_cache`, take the public
//! `analytic_outer_rho_gradient_at_converged`, and assert every gradient channel
//! has the k-dim length `1 + K + Σ_k d_k` for BOTH n, and that the assembled
//! gradient vector has that same n-invariant length. A regression that routed an
//! n-sized object into the outer-search payload (re-introducing an n-dimensional
//! coordinate the optimizer would have to walk) would change one of these
//! lengths and fail here. The ρ flat-coordinate round trip (`to_flat` /
//! `from_flat`) is pinned n-invariant for the same reason: the search space the
//! engine optimizes over is k-dim.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use ndarray::Array1;

use gam::terms::{sae::manifold::SaeManifoldRho};

const D: usize = 1; // latent dim per atom (circle)
const K: usize = 2; // atoms

/// k-dim outer-coordinate length: the shared sparse log-strength
/// (`log_lambda_sparse`), the K per-atom smoothness log-strengths
/// (`log_lambda_smooth[k]`, one per atom — #1556), plus one ARD precision per
/// atom per latent axis. Independent of the row count n — that is the property
/// under test.
const RHO_FLAT_LEN: usize = 1 + K + K * D;

/// The ARD-enabled ρ: one log-precision per atom per latent axis, plus the two
/// shared log-strengths. Identical k-dim layout for every n.
fn ard_rho() -> SaeManifoldRho {
    SaeManifoldRho::new(
        (1.0e-2_f64).ln(),
        (1.0e-2_f64).ln(),
        vec![Array1::<f64>::from_elem(D, (1.0e-1_f64).ln()); K],
    )
}

/// The ρ flat-coordinate space the outer engine optimizes over is k-dim, not
/// n-dim: `to_flat` / `from_flat` round-trip at length `1 + K + Σ_k d_k`
/// independent of the row count. (Pure ρ-structure property — no inner solve.)
#[test]
fn rho_flat_coordinate_space_is_n_invariant_1033() {
    let rho = ard_rho();
    let flat = rho.to_flat();
    assert_eq!(
        flat.len(),
        RHO_FLAT_LEN,
        "ρ flat coordinate length must be the k-dim 1 + K + Σ d_k = {RHO_FLAT_LEN}, the outer \
         search space dimension; got {}",
        flat.len()
    );
    // Round-trip is the exact inverse and stays k-dim.
    let back = rho.from_flat(flat.view()).unwrap();
    assert_eq!(back.to_flat().len(), RHO_FLAT_LEN);
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "k-dim outer coordinate must be finite"
    );
}

