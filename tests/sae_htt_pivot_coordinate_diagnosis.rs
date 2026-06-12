//! #1037 DISCRIMINATING DIAGNOSIS (review gate for the gauge-deflation landing).
//!
//! The non-PD per-row `H_tt` evidence pivot (~-6e-11) that killed every real-LLM
//! SAE fit has two candidate sources, which demand DIFFERENT fixes:
//!
//!   (a) LATENT tangent gauge zero — the circle-rotation orbit direction in the
//!       per-atom chart block. Where a row's reconstruction residual is radial
//!       (a locally perfect fit) the tangential phase-shift curvature vanishes.
//!       This is a true gauge orbit → #1037's Faddeev–Popov deflation (the
//!       deflation field spans ONLY the latent chart directions, never the gate
//!       logit — see `push_atom_row_gauge_deflations`).
//!
//!   (b) GATE-logit raw curvature going indefinite — the un-majorized IBP/softmax
//!       `score·(1−2z)` logit curvature, which is genuinely indefinite off the
//!       optimum (see the long H-consistency note at the `H_tt` assembly site).
//!       That is NOT a gauge orbit; deflating it would be wrong. The principled
//!       fix there is a CONVENTION/consistency fix (assemble the same majorized
//!       PSD assignment curvature the step path + #1006 Γ-adjoint already use).
//!
//! This test decomposes each per-row `H_tt^(i)` block into its GATE sub-block
//! (`0..assignment_coord_dim`) and its LATENT sub-block (`assignment_coord_dim..`)
//! and reports, per coordinate group, the minimum eigenvalue relative to the
//! block-diagonal scale — across a sweep of states from the diffuse cold seed
//! toward the heavy-smoothing continuation entry. The verdict (which coordinate
//! group carries the numerical-zero / negative curvature, and whether the latent
//! zeros track residual radiality) determines whether the deflation landing is
//! accepted (a), amended to a convention fix (b), or both (c).
//!
//! It asserts only that SOME row exhibits the pivot-class flat/negative direction
//! (so the diagnosis is non-vacuous) and PRINTS the full per-coordinate breakdown
//! under `--nocapture` for the human verdict.

use gam::linalg::faer_ndarray::{FaerCholesky, FaerEigh};
use gam::terms::latent_coord::LatentManifold;
use gam::terms::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

const M: usize = 3;
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;

fn planted_high_dim_circle(n: usize, p: usize, seed: u64) -> Array2<f64> {
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

/// Build the K=1 periodic circle term. `coord_jitter` shifts the seeded phase
/// off the true value, and `radial_rows` forces a subset of rows to an EXACTLY
/// radial residual (phase set so the decoded point is collinear with the data —
/// the worst case for the tangential gauge curvature) so the latent-zero branch
/// is reproducible if it is real.
fn build_k1_circle_term(z: &Array2<f64>, coord_jitter: f64) -> SaeManifoldTerm {
    let n = z.nrows();
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    let coords = Array2::from_shape_fn((n, 1), |(i, _)| {
        ((i as f64) / (n as f64) + coord_jitter).rem_euclid(1.0)
    });
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();

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

/// Minimum eigenvalue of a symmetric sub-block (rows/cols `idx`), or `None` for
/// an empty selection.
fn sub_block_min_eig(htt: &Array2<f64>, idx: &[usize]) -> Option<f64> {
    if idx.is_empty() {
        return None;
    }
    let mut sub = Array2::<f64>::zeros((idx.len(), idx.len()));
    for (a, &i) in idx.iter().enumerate() {
        for (b, &j) in idx.iter().enumerate() {
            sub[[a, b]] = htt[[i, j]];
        }
    }
    let (evals, _v) = sub.eigh(faer::Side::Lower).ok()?;
    Some(evals.iter().copied().fold(f64::INFINITY, f64::min))
}

#[test]
fn htt_pivot_coordinate_diagnosis_gate_vs_latent() {
    let n = 150;
    let p = 512;
    let z = planted_high_dim_circle(n, p, 0x1037);

    // Sweep states from the exact planted seed toward off-optimum phase jitter —
    // the latent gauge zero is an OFF-OPTIMUM transient, so we probe several
    // iterates the cold continuation walk would traverse.
    let mut any_pivot_class = false;
    for &jitter in &[0.0_f64, 0.05, 0.15, 0.30] {
        let mut term = build_k1_circle_term(&z, jitter);
        let assignment_coord_dim = term.assignment.assignment_coord_dim();
        let rho = SaeManifoldRho::new(
            (1.0e-3_f64).ln(),
            (1.0e-3_f64).ln(),
            vec![Array1::from_elem(1, (1.0e-3_f64).ln())],
        );
        let sys = term
            .assemble_arrow_schur(z.view(), &rho, None)
            .expect("arrow-Schur assembly");

        // Block-diagonal curvature unit.
        let mut max_diag = 0.0_f64;
        for row in &sys.rows {
            for ax in 0..row.htt.nrows() {
                max_diag = max_diag.max(row.htt[[ax, ax]].abs());
            }
        }
        let max_diag = max_diag.max(1.0e-300);

        // Per-coordinate-group flattest curvature across rows.
        let mut gate_min = f64::INFINITY;
        let mut latent_min = f64::INFINITY;
        let mut gate_zero_rows = 0usize;
        let mut latent_zero_rows = 0usize;
        let eps = 1.0e-6_f64;
        for row in &sys.rows {
            let q = row.htt.nrows();
            let gate_idx: Vec<usize> = (0..assignment_coord_dim.min(q)).collect();
            let latent_idx: Vec<usize> = (assignment_coord_dim.min(q)..q).collect();
            if let Some(g) = sub_block_min_eig(&row.htt, &gate_idx) {
                gate_min = gate_min.min(g);
                if g <= eps * max_diag {
                    gate_zero_rows += 1;
                }
            }
            if let Some(l) = sub_block_min_eig(&row.htt, &latent_idx) {
                latent_min = latent_min.min(l);
                if l <= eps * max_diag {
                    latent_zero_rows += 1;
                }
            }
        }

        println!(
            "[#1037 diag] jitter={jitter:.2} max_diag={max_diag:.3e} \
             GATE min_eig/max_diag={:.3e} ({gate_zero_rows} zero rows) \
             LATENT min_eig/max_diag={:.3e} ({latent_zero_rows} zero rows)",
            gate_min / max_diag,
            latent_min / max_diag
        );
        if gate_zero_rows > 0 || latent_zero_rows > 0 {
            any_pivot_class = true;
        }
    }

    // Non-vacuous: SOME probed state must exhibit a pivot-class flat/negative
    // direction in SOME coordinate group, or the planted geometry no longer
    // reproduces the failure family this diagnosis exists to attribute.
    assert!(
        any_pivot_class,
        "no pivot-class near-zero/negative H_tt direction found in any coordinate group across \
         the probed states; the planted circle no longer reproduces the #1037 failure family"
    );
}
