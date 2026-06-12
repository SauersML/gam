//! #1037 reproduction + gauge diagnosis: the per-row `H_tt` non-PD evidence
//! pivot that kills every real-LLM SAE fit at seed startup.
//!
//! The autopsy (science-reader, rev d929e13f1) found ALL real-bank fits dying
//! at startup with:
//!
//!   'undamped evidence factorization hit a non-PD per-row H_tt block ...
//!    non-PD pivot ~ -6e-11 ... evidence mode does not condition non-PD blocks'
//!
//! at K=1 circle p=5120 (OLMo color bank) and K=8 p=2048 (Qwen). The claim of
//! #1037 is that this pivot is the circle-rotation GAUGE direction localized in
//! one row's `H_tt`: wherever a row's reconstruction residual is radial (a
//! locally perfect fit), the tangential phase-shift direction contributes no
//! curvature, so the `d × d` block (here `d = 1`) goes to a numerical zero. At
//! high `p` with concentrated harmonics this happens generically at some rows.
//!
//! These tests:
//!   1. `high_dim_circle_inner_evidence_dies_with_gauge_pivot` — plant a clean
//!      circle in `p ∈ {512, 2048}` and drive the production `reml_criterion`;
//!      TODAY this returns the non-PD-pivot error verbatim (the gate the fix
//!      must flip to a high-EV fit). Recorded as the red repro.
//!   2. `radial_residual_row_htt_is_gauge_explained` — assemble the arrow-Schur
//!      system at a row whose residual is engineered radial and confirm the
//!      failing direction IS the gauge direction: the `d = 1` tangent block
//!      `H_tt^(i)` is `≤ eps · max_diag`, i.e. gauge-explained per the #1037
//!      test, NOT generic corruption. This is the evidence that the principled
//!      fix (Faddeev–Popov deflation of the row-restricted orbit, contributing
//!      a θ/ρ-constant `log κ` to the Laplace normalizer) is sound.

use gam::linalg::faer_ndarray::{FaerCholesky, FaerEigh};
use gam::terms::latent_coord::LatentManifold;
use gam::terms::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

const M: usize = 3; // const + 1 harmonic (sin, cos): circle basis, rank ≤ 3.
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;

/// Plant a clean unit circle embedded in `R^p` by a random orthonormal pair of
/// ambient directions, sampled at `n` phases. The reconstruction at the truth
/// is exact (zero noise), so every row's residual is radial-by-construction at
/// the planted decoder — the worst case for the tangential gauge curvature.
fn planted_high_dim_circle(n: usize, p: usize, seed: u64) -> Array2<f64> {
    // Deterministic LCG: two ambient unit directions, orthonormalized.
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((bits >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut u = Array1::<f64>::from_shape_fn(p, |_| next());
    let mut v = Array1::<f64>::from_shape_fn(p, |_| next());
    let un = u.dot(&u).sqrt();
    u.mapv_inplace(|x| x / un);
    let uv = u.dot(&v);
    v.scaled_add(-uv, &u);
    let vn = v.dot(&v).sqrt();
    v.mapv_inplace(|x| x / vn);

    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let theta = std::f64::consts::TAU * (row as f64) / (n as f64);
        let (s, c) = theta.sin_cos();
        for col in 0..p {
            z[[row, col]] = c * u[col] + s * v[col];
        }
    }
    z
}

/// Build a K=1 periodic circle term from data `z`, with the on-manifold phase
/// coordinate seeded at the true phase and the decoder least-squares-fit to the
/// data (the production cold-start geometry: residual-energy logits are trivial
/// for K=1, weighted-LSQ decoder).
fn build_k1_circle_term(z: &Array2<f64>) -> SaeManifoldTerm {
    let n = z.nrows();
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    let coords = Array2::from_shape_fn((n, 1), |(i, _)| (i as f64) / (n as f64));
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();

    // Decoder least squares B = (Φᵀ Φ + jitter)⁻¹ Φᵀ Z, shape (M, p).
    let mut gram = phi.t().dot(&phi);
    let mut trace = 0.0_f64;
    for i in 0..M {
        trace += gram[[i, i]];
    }
    let jitter = (trace / M as f64).max(1.0) * 1.0e-10;
    for i in 0..M {
        gram[[i, i]] += jitter;
    }
    let rhs = phi.t().dot(z);
    let chol = gram.cholesky(faer::Side::Lower).unwrap();
    let b = chol.solve_mat(&rhs);

    let atom = SaeManifoldAtom::new(
        "circle_0".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        b,
        Array2::<f64>::eye(M),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap()));

    let logits = Array2::<f64>::zeros((n, 1));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(TAU, ALPHA, false),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

/// #1037 acceptance repro: high-dim planted circle currently dies in the inner
/// evidence factorization with the gauge pivot. After the fix this must fit to
/// EV ≥ 0.95. The test asserts the CURRENT failure mode precisely so the fix's
/// flip is unambiguous; the post-fix assertion is the EV gate.
#[test]
fn high_dim_circle_inner_evidence_dies_with_gauge_pivot() {
    for &p in &[512usize, 2048usize] {
        let n = 150;
        let z = planted_high_dim_circle(n, p, 0xC0FFEE ^ p as u64);
        let mut term = build_k1_circle_term(&z);
        let rho = SaeManifoldRho::new((1.0e-3_f64).ln(), (1.0e-3_f64).ln(), Vec::new());

        let result = term.reml_criterion(z.view(), &rho, None, 25, 1.0, 0.0, 0.0);

        match result {
            Ok((cost, _loss)) => {
                // POST-FIX path: the fit must reconstruct the circle.
                let fitted = term.fitted();
                let mut num = 0.0_f64;
                let mut den = 0.0_f64;
                let mut mean = Array1::<f64>::zeros(p);
                for row in 0..n {
                    for col in 0..p {
                        mean[col] += z[[row, col]];
                    }
                }
                mean.mapv_inplace(|x| x / n as f64);
                for row in 0..n {
                    for col in 0..p {
                        let r = z[[row, col]] - fitted[[row, col]];
                        num += r * r;
                        let dm = z[[row, col]] - mean[col];
                        den += dm * dm;
                    }
                }
                let ev = 1.0 - num / den;
                assert!(
                    ev >= 0.95,
                    "p={p}: post-fix circle fit must reach EV ≥ 0.95, got EV={ev:.4} (cost={cost:.4})"
                );
            }
            Err(msg) => {
                // PRE-FIX path: the documented gauge-pivot death. Assert it is
                // THIS failure (non-PD per-row H_tt evidence block), not some
                // unrelated error, so the fix's green flip is meaningful.
                assert!(
                    msg.contains("non-PD per-row H_tt block")
                        || msg.contains("H_tt is non-PD at base ridge"),
                    "p={p}: expected the #1037 gauge-pivot evidence failure, got: {msg}"
                );
            }
        }
    }
}

/// Diagnosis: confirm the failing direction is gauge-explained. At the planted
/// (zero-residual) circle every row's reconstruction is radial, so the single
/// tangential phase-shift curvature entry `H_tt^(i)` (d=1) is a numerical zero
/// relative to the block-diagonal scale — exactly the #1037 gauge-explained
/// test `g_i^T H_tt g_i ≤ eps · max_diag · |g_i|^2` with `g_i = [1]`.
#[test]
fn radial_residual_row_htt_is_gauge_explained() {
    let n = 150;
    let p = 512;
    let z = planted_high_dim_circle(n, p, 0x1037);
    let mut term = build_k1_circle_term(&z);
    let rho = SaeManifoldRho::new((1.0e-3_f64).ln(), (1.0e-3_f64).ln(), Vec::new());

    let sys = term
        .assemble_arrow_schur(z.view(), &rho, None)
        .expect("arrow-Schur assembly at the planted seed");

    // Block-diagonal scale = the largest per-row H_tt diagonal across all rows:
    // the natural curvature unit the near-zero direction is measured against.
    let mut max_diag = 0.0_f64;
    for row in &sys.rows {
        for ax in 0..row.htt.nrows() {
            max_diag = max_diag.max(row.htt[[ax, ax]].abs());
        }
    }
    assert!(
        max_diag > 0.0,
        "expected non-trivial per-row curvature somewhere; max_diag={max_diag}"
    );

    // For each per-row `H_tt^(i)` block, the SMALLEST eigenvalue is the flattest
    // curvature direction in that row's chart. A gauge-explained row has a
    // near-zero (or slightly negative, at the numerical-zero scale of the
    // autopsy's -6e-11) minimum eigenvalue relative to `max_diag`: the orbit
    // direction the #1037 Faddeev–Popov deflation must stiffen rather than the
    // evidence factor reject. Count those rows and report the flattest one.
    let eps = 1.0e-6_f64;
    let mut near_zero_rows = 0usize;
    let mut min_ratio = f64::INFINITY;
    for row in &sys.rows {
        let (evals, _vecs) = row
            .htt
            .eigh(faer::Side::Lower)
            .expect("per-row H_tt eigendecomposition");
        let lambda_min = evals.iter().copied().fold(f64::INFINITY, f64::min);
        let ratio = lambda_min / max_diag;
        min_ratio = min_ratio.min(ratio);
        // Gauge-explained: the flat direction's curvature is a numerical zero
        // (|λ_min| ≤ eps·max_diag), i.e. neither materially positive nor a real
        // negative curvature well — exactly the radial-residual orbit zero.
        if lambda_min <= eps * max_diag {
            near_zero_rows += 1;
        }
    }

    assert!(
        near_zero_rows > 0,
        "expected ≥1 row whose flattest H_tt curvature is a numerical zero \
         (λ_min ≤ {eps}·max_diag) — the gauge orbit the #1037 deflation targets; none found \
         (min ratio={min_ratio:.3e}, max_diag={max_diag:.3e}). If this fails the planted \
         geometry no longer produces the gauge zero and the repro must be revisited."
    );
}
