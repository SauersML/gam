//! #2022 GATE — in-loop unit-speed (arc-length) chart retraction: (1) the
//! assembled latent/decoder gradient stays FD-consistent with the objective at
//! the chart the fit sits on — INCLUDING the ARD coordinate gradient, the exact
//! term the in-loop re-gauge moves and that a desync would corrupt in the line
//! search; and (2) the in-loop retraction hook NEVER corrupts the fit (image-
//! frozen data-fit + smoothness; honest-skip is a strict no-op; if it does fire,
//! the ARD change equals exactly the von-Mises coordinate-energy delta).
//!
//! Why the gradient FDs are the load-bearing part (residual's adversarial
//! review): the earlier β-only FD is ARD⊥β blind (∂ARD/∂β = 0), so it did NOT
//! guard the coordinate gradient. The t-coordinate FD below does. A t-perturbation
//! is invisible unless the cached basis Φ is refreshed, so we call
//! `refresh_basis_from_current_coords` around each probe (the β case needs no
//! refresh — the decoder is not cached in Φ).
//!
//! NOTE on reachability (see #2022 review): a non-trivial d=1 arc-length reparam
//! is nonlinear, so its recomposition against a finite basis exceeds the 1e-9
//! image-freeze gate and `canonicalize_atom_unit_speed_chart` HONEST-SKIPS for a
//! harmonic fixture. The gate therefore pins the two RELIABLY-reachable
//! contracts: gradient consistency (always) and no-op safety (honest-skip must
//! not move the fit). The active-retraction ARD-invariance is asserted
//! conditionally (only if it fires).

use crate::assignment::{AssignmentMode, SaeAssignment};
use crate::chart_canonicalization::CanonicalChartTopology;
use crate::manifold::{
    ArdAxisPrior, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use gam_terms::latent::LatentManifold;
use ndarray::{array, Array2};
use std::sync::Arc;

use super::tests::{periodic_basis, TestPeriodicEvaluator};

fn build_circle_term(coords_col: &Array2<f64>, decoder: &Array2<f64>) -> SaeManifoldTerm {
    let n = coords_col.nrows();
    let (phi, jet) = periodic_basis(coords_col);
    let m = phi.ncols();
    assert_eq!(decoder.nrows(), m, "decoder rows must equal basis width");
    let atom = SaeManifoldAtom::new(
        "circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder.clone(),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
    let logits = Array2::<f64>::from_elem((n, 1), 2.0); // gated ON (logit 2 > threshold 0)
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords_col.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::jumprelu(1.0, 0.0),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom], assignment).unwrap()
}

#[test]
fn unit_speed_gate_gradient_consistency_and_noop_safety_2022() {
    let period = 1.0_f64;
    // Uneven circle coordinates (non-uniform curve speed) with row 3 well away
    // from the origin so the ARD coordinate gradient at that row is non-trivial.
    let coords_col = array![
        [0.02_f64],
        [0.10],
        [0.17],
        [0.31],
        [0.55],
        [0.66],
        [0.80],
        [0.95]
    ];
    let n = coords_col.nrows();
    let p = 3usize;
    let (phi0, _) = periodic_basis(&coords_col);
    let m = phi0.ncols();
    let decoder = Array2::<f64>::from_shape_fn((m, p), |(a, b)| {
        0.3 * ((a + 1) as f64) - 0.15 * (b as f64) + 0.05 * ((a * p + b) as f64)
    });

    let mut term = build_circle_term(&coords_col, &decoder);
    let target =
        Array2::<f64>::from_shape_fn((n, p), |(r, c)| 0.2 - 0.05 * (r as f64) + 0.1 * (c as f64));
    let rho = SaeManifoldRho::new(0.0, -4.0, vec![array![0.0]]); // λ_sparse=1, α=1 ARD

    let sys = term.assemble_arrow_schur(target.view(), &rho, None).unwrap();
    let h = 1.0e-6_f64;

    // ---- (1a) t-COORDINATE gradient FD — guards ∂(data_fit+ARD)/∂t. ----
    // Row block layout for the single gated-on atom is [logit, coord], so the
    // coordinate gradient at observation `row` is `gt[1]`.
    let row = 3usize;
    let base_flat = term.assignment.coords[0].as_matrix().column(0).to_owned();
    let mut fp = base_flat.clone();
    fp[row] = (fp[row] + h).rem_euclid(period);
    term.assignment.coords[0].set_flat(fp.view());
    term.refresh_basis_from_current_coords().unwrap();
    let lp = term.loss(target.view(), &rho).unwrap().total();
    let mut fm = base_flat.clone();
    fm[row] = (fm[row] - h).rem_euclid(period);
    term.assignment.coords[0].set_flat(fm.view());
    term.refresh_basis_from_current_coords().unwrap();
    let lm = term.loss(target.view(), &rho).unwrap().total();
    term.assignment.coords[0].set_flat(base_flat.view());
    term.refresh_basis_from_current_coords().unwrap();
    let gt_fd = (lp - lm) / (2.0 * h);
    let gt_analytic = sys.rows[row].gt[1];
    assert!(
        gt_fd.abs() > 1.0e-3,
        "coord gradient must be non-trivial so the ARD term is actually guarded (got {gt_fd})"
    );
    assert!(
        (gt_analytic - gt_fd).abs() <= 1.0e-4 * (1.0 + gt_fd.abs()),
        "assembled coord gradient {gt_analytic} must match FD {gt_fd} (ARD/coord desync guard)"
    );

    // ---- (1b) β gradient FD — ≥2 entries, tight tol (β is Euclidean; no refresh). ----
    for &(bm, bp) in &[(0usize, 0usize), (1usize, 1usize)] {
        let beta_idx = bm * p + bp; // single atom: β flattened row-major (m × p)
        let g_analytic = sys.gb[beta_idx];
        let base = term.atoms[0].decoder_coefficients[[bm, bp]];
        term.atoms[0].decoder_coefficients[[bm, bp]] = base + h;
        let lp = term.loss(target.view(), &rho).unwrap().total();
        term.atoms[0].decoder_coefficients[[bm, bp]] = base - h;
        let lm = term.loss(target.view(), &rho).unwrap().total();
        term.atoms[0].decoder_coefficients[[bm, bp]] = base; // restore
        let g_fd = (lp - lm) / (2.0 * h);
        assert!(
            (g_analytic - g_fd).abs() <= 1.0e-6 * (1.0 + g_fd.abs()),
            "β-gradient[{bm},{bp}] {g_analytic} must match FD {g_fd}"
        );
    }

    // ---- (2) retraction NO-OP SAFETY: the in-loop hook never corrupts the fit. ----
    let l0 = term.loss(target.view(), &rho).unwrap();
    let coords0 = term.assignment.coords[0].as_matrix().column(0).to_owned();
    let topo = CanonicalChartTopology::Circle { period };
    let applied = term.canonicalize_atom_unit_speed_chart(0, &topo).unwrap();
    let l1 = term.loss(target.view(), &rho).unwrap();
    // Image-frozen ⇒ data-fit + intrinsic smoothness invariant whether the
    // retraction fired or honest-skipped.
    assert!(
        (l1.data_fit - l0.data_fit).abs() <= 1.0e-6 * (1.0 + l0.data_fit.abs()),
        "data_fit must be invariant under the retraction: {} vs {}",
        l1.data_fit,
        l0.data_fit
    );
    assert!(
        (l1.smoothness - l0.smoothness).abs() <= 1.0e-6 * (1.0 + l0.smoothness.abs()),
        "smoothness must be invariant under the retraction: {} vs {}",
        l1.smoothness,
        l0.smoothness
    );
    let coords1 = term.assignment.coords[0].as_matrix().column(0).to_owned();
    if applied {
        // If it fired, the ARD change equals EXACTLY the von-Mises coordinate-
        // energy delta at the reparam'd coords (the ard_value normalizer depends
        // only on α/n/period and cancels) — proving it's the reparam effect.
        let alpha = SaeManifoldRho::stable_exp_strength(0.0);
        let mut expected = 0.0_f64;
        for i in 0..n {
            expected += ArdAxisPrior::eval(alpha, coords1[i], Some(period)).value
                - ArdAxisPrior::eval(alpha, coords0[i], Some(period)).value;
        }
        assert!(
            ((l1.ard - l0.ard) - expected).abs() <= 1.0e-9 * (1.0 + expected.abs()),
            "ARD delta {} must equal the von-Mises energy delta {}",
            l1.ard - l0.ard,
            expected
        );
    } else {
        // Honest-skip ⇒ strict no-op: nothing moved, ARD unchanged.
        let drift = coords0
            .iter()
            .zip(coords1.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(drift <= 1.0e-12, "honest-skip must not move coords; drift {drift}");
        assert!(
            (l1.ard - l0.ard).abs() <= 1.0e-12 * (1.0 + l0.ard.abs()),
            "honest-skip must not move the ARD objective"
        );
    }

    // The in-loop hook itself runs safely at a chart-refresh boundary.
    term.retract_unit_speed_charts_in_loop().unwrap();
}
