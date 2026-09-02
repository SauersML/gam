//! Rolled-sheet recovery regression for the intrinsic-metric seeder (#2240/#2280).
//!
//! The thesis (validated in numpy by fable-mobius, the geometry owner): on a
//! FOLDED manifold — a swiss roll, a flat 2-D sheet rolled up in 3-D so that
//! geodesically-distant points are ambient-close — a global-LINEAR (PCA) seed
//! projects the two highest-variance ambient directions, a NON-INJECTIVE map that
//! folds the sheet's layers onto each other; a flexible decoder reading that 2-D
//! chart cannot separate the collapsed layers, so its held-out reconstruction R²
//! craters. The geodesic (Isomap) seed unrolls the sheet into a faithful,
//! single-valued 2-D chart, so the same decoder recovers the full ambient image
//! (held-out R² → 1). On a NON-fold the two seeds are equivalent charts and tie.
//!
//! The reconstruction metric is fable-mobius's validated held-out thin-plate-spline
//! R² (deterministic, RNG-free): fit a thin-plate RBF decoder + affine tail from
//! the 2-D seed chart to the ambient image on the training rows, predict the
//! held-out rows, report R². This is the exact objective (grid fixture, decoder,
//! split, thresholds) her landed numpy/Rust falsifier uses — intrinsic R²=0.9996,
//! PCA-2 R²=0.825 — lifted here to run against the CANONICAL seeder
//! [`sae_intrinsic_seed_initial_coords`] and its end-to-end auto-seed path.

use super::*;
use gam_linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use ndarray::{Array2, Array3};

/// fable-mobius's deterministic ~2-turn swiss-roll grid in R³ (no RNG): a flat
/// `(arclength, height)` sheet rolled as `(t·cos t, height, t·sin t)`. Height
/// carries a minority of the ambient variance, so the PCA-2 projection folds the
/// roll-plane layers together and drops height.
fn swiss_roll_grid() -> Array2<f64> {
    let (n_t, n_h) = (45usize, 10usize);
    let n = n_t * n_h;
    let mut z = Array2::<f64>::zeros((n, 3));
    for ti in 0..n_t {
        let t = 1.2 * std::f64::consts::PI
            + (3.2 - 1.2) * std::f64::consts::PI * ti as f64 / (n_t - 1) as f64;
        for hi in 0..n_h {
            let h = 10.0 * hi as f64 / (n_h - 1) as f64;
            let row = ti * n_h + hi;
            z[[row, 0]] = t * t.cos();
            z[[row, 1]] = h;
            z[[row, 2]] = t * t.sin();
        }
    }
    z
}

/// Held-out thin-plate-spline reconstruction R² of the ambient image `z` from a
/// 2-D latent chart `coords` — fable-mobius's exact validated decoder: standardize
/// coords per-axis, 70 strided thin-plate centers (`0.5·r²·ln r²`) plus an affine
/// `[1, u, v]` tail, ridge `max_diag·1e-8`, rows with `i % 4 == 0` held out.
fn heldout_tps_r2(coords: &Array2<f64>, z: &Array2<f64>) -> f64 {
    let n = z.nrows();
    let p = z.ncols();
    let cmean = coords
        .mean_axis(ndarray::Axis(0))
        .expect("the latent chart has at least one row");
    let mut cstd = [0.0_f64; 2];
    for k in 0..2 {
        cstd[k] = (coords
            .column(k)
            .iter()
            .map(|&v| (v - cmean[k]).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt()
            .max(1e-12);
    }
    let n_centers = 70usize;
    let centers: Vec<usize> = (0..n_centers)
        .map(|i| i * (n - 1) / (n_centers - 1))
        .collect();
    let width = 3 + n_centers;
    let mut phi = Array2::<f64>::zeros((n, width));
    let cs = |row: usize, k: usize| (coords[[row, k]] - cmean[k]) / cstd[k];
    for row in 0..n {
        phi[[row, 0]] = 1.0;
        phi[[row, 1]] = cs(row, 0);
        phi[[row, 2]] = cs(row, 1);
        for (ci, &cr) in centers.iter().enumerate() {
            let r2 = (0..2)
                .map(|k| {
                    let d = cs(row, k) - cs(cr, k);
                    d * d
                })
                .sum::<f64>()
                .max(1e-12);
            phi[[row, 3 + ci]] = 0.5 * r2 * r2.ln();
        }
    }
    let train: Vec<usize> = (0..n).filter(|r| r % 4 != 0).collect();
    let test: Vec<usize> = (0..n).filter(|r| r % 4 == 0).collect();
    let phi_tr = phi.select(ndarray::Axis(0), &train);
    let z_tr = z.select(ndarray::Axis(0), &train);
    let mut gram = fast_ata(&phi_tr);
    let scale = gram.diag().iter().copied().fold(0.0_f64, f64::max);
    for dgn in gram.diag_mut().iter_mut() {
        *dgn += scale * 1e-8;
    }
    let rhs = fast_atb(&phi_tr, &z_tr);
    let decoder = gram
        .cholesky(faer::Side::Lower)
        .expect("the thin-plate Gram carries a max_diag*1e-8 ridge, so it is positive definite")
        .solve_mat(&rhs);
    let mut mean_t = vec![0.0_f64; p];
    for &row in &test {
        for c in 0..p {
            mean_t[c] += z[[row, c]];
        }
    }
    for c in 0..p {
        mean_t[c] /= test.len() as f64;
    }
    let (mut resid, mut total) = (0.0_f64, 0.0_f64);
    for &row in &test {
        for c in 0..p {
            let mut fit = 0.0_f64;
            for a in 0..width {
                fit += phi[[row, a]] * decoder[[a, c]];
            }
            resid += (z[[row, c]] - fit).powi(2);
            total += (z[[row, c]] - mean_t[c]).powi(2);
        }
    }
    1.0 - resid / total
}

/// Extract an atom's `(n, 2)` latent chart from a `(K, n, d_max)` seed array.
fn chart_of(seed: &Array3<f64>, atom_idx: usize) -> Array2<f64> {
    let n = seed.shape()[1];
    let mut out = Array2::<f64>::zeros((n, 2));
    for row in 0..n {
        out[[row, 0]] = seed[[atom_idx, row, 0]];
        out[[row, 1]] = seed[[atom_idx, row, 1]];
    }
    out
}

/// END-TO-END PRIMARY (#2280 guardrail, fable-mobius Q2): the FULL auto-seed path
/// — discover → race → resolve_auto_primary_atoms → minimal_seed — must install the
/// UNFOLDED geodesic chart as the final seed coordinates, not a PCA-folded rebuild.
/// This is red unless the winning intrinsic coords actually PROPAGATE through
/// PrimaryTopologyChoice into the coord block, so it guards the whole verdict→coords
/// chain, not just the primitive. Asserts the FINAL seed's held-out R² clears 0.99.
#[test]
fn swiss_roll_auto_seed_propagates_unfolded_coords_end_to_end() {
    let z = swiss_roll_grid();
    let report = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: z.view(),
        atom_basis: vec!["auto".to_string()],
        atom_dim: vec![2],
        assignment_kind: SaeFitAssignmentKind::Softmax,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        random_state: 0,
        initial_logits: None,
        initial_coords: None,
    })
    .expect("auto swiss-roll seed builds");
    assert_eq!(
        report.geometry_plans[0].latent_dim(),
        2,
        "a swiss roll is an intrinsically 2-D sheet; auto discovery must resolve d=2"
    );
    let r2 = heldout_tps_r2(&chart_of(&report.initial_coords, 0), &z);
    assert!(
        r2 > 0.99,
        "the auto-seed path must install the UNFOLDED geodesic chart as the final \
         seed coords (the intrinsic race winner must reach the coordinates, not just \
         the kind); held-out R²={r2}"
    );
}

