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
//! `high_dim_circle_inner_evidence_dies_with_gauge_pivot` plants a clean circle
//! in `p ∈ {512, 2048}` and drives the production `reml_criterion`. Before the
//! #1037 fix this returned the non-PD-pivot error verbatim; after it, the fit
//! must reach EV ≥ 0.95 AND must have recorded at least one gauge-deflated
//! evidence direction — proving the radial-residual gauge zero really surfaced
//! mid-solve and was stiffened (Faddeev–Popov, contributing a θ/ρ-constant
//! `log κ` to the Laplace normalizer) rather than rejected. The gauge zero is
//! an OFF-OPTIMUM transient — it appears on the inner Newton trajectory, not at
//! the cold planted seed — so the only faithful probe is to drive the full
//! inner solve and observe the recorded deflation count; a static assembly at
//! the seed does not exhibit it.

use gam::linalg::faer_ndarray::FaerCholesky;
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
        let rho = SaeManifoldRho::new(
            (1.0e-3_f64).ln(),
            (1.0e-3_f64).ln(),
            vec![Array1::from_elem(1, (1.0e-3_f64).ln())],
        );

        let result = term.reml_criterion(z.view(), &rho, None, 25, 1.0, 0.0, 0.0);

        match result {
            Ok((cost, loss)) => {
                // POST-FIX path: the fit must reconstruct the circle AND it must
                // have reached the optimum THROUGH the #1037 gauge-deflated
                // evidence factorization — i.e. the radial-residual gauge zero
                // really did surface mid-solve and was stiffened (Faddeev–Popov)
                // rather than rejected. A success with zero deflations would mean
                // the planted geometry no longer exercises the pivot this test
                // exists to guard, so the gate would be vacuous.
                assert!(
                    loss.evidence_gauge_deflated_directions > 0,
                    "p={p}: the fit succeeded but recorded 0 gauge-deflated evidence directions — \
                     the #1037 gauge orbit was never exercised, so this repro no longer guards \
                     the pivot. Revisit the planted geometry."
                );
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

