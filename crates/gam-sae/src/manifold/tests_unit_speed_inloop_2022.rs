//! #2022 GATE — in-loop unit-speed (arc-length) chart retraction is
//! objective-CONSISTENT and moves ONLY the identifying ARD coordinate prior.
//!
//! This is the mandatory behavior/FD gate for promoting the arc-length
//! canonicalization from post-fit to in-loop. Because the retraction re-gauges
//! the chart coordinates `t`, the ARD *coordinate* prior energy changes (it is
//! the term that pins the residual gauge to `t → ±t + c`), so this is a
//! behavior-changing optimizer edit that MUST carry its own gate. The retraction
//! is the exact, image-frozen `unit_speed_reparameterization` applied through the
//! existing `canonicalize_atom_unit_speed_chart` apply-path — the SAME operation
//! the in-loop wiring calls at chart-refresh boundaries.
//!
//! Asserted:
//!  (data-fit)   invariant — the reparam is image-frozen (`Φ(t̃)B̃ ≈ Φ(t)B`).
//!  (smoothness) invariant — the congruence transport preserves `BᵀSB`.
//!  (ARD)        MOVES — and the change equals EXACTLY the von-Mises coordinate
//!               energy delta at the reparam'd coords (FD confirmation that it is
//!               the intended reparam effect, not a gradient/bookkeeping bug).
//!  (gradient)   the assembled β-gradient at the canonical chart matches a
//!               central finite-difference of the total loss (assemble↔loss stay
//!               consistent across the re-gauge — the Armijo/ARD-regression guard).
//!  (idempotence) re-applying at the next boundary is a no-op — safe to call every
//!               refresh without perturbing the line search.

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
    // Single atom, gated ON (logit 2 > threshold 0).
    let logits = Array2::<f64>::from_elem((n, 1), 2.0);
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
fn unit_speed_retraction_objective_consistent_moves_only_ard_2022() {
    let period = 1.0_f64;
    // Unevenly-spaced circle coordinates ⇒ the fitted decoder curve has
    // non-uniform speed, so the arc-length retraction is non-trivial.
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
    // alpha = exp(0) = 1 ARD precision; modest smoothness.
    let rho = SaeManifoldRho::new(0.0, -4.0, vec![array![0.0]]);
    let alpha = SaeManifoldRho::stable_exp_strength(0.0);

    let loss0 = term.loss(target.view(), &rho).unwrap();
    let old_coords: Vec<f64> = term.assignment.coords[0]
        .as_matrix()
        .column(0)
        .to_vec();

    // ---- apply the (in-loop) unit-speed retraction via the shared apply-path ----
    let topo = CanonicalChartTopology::Circle { period };
    let applied = term.canonicalize_atom_unit_speed_chart(0, &topo).unwrap();
    assert!(
        applied,
        "a non-uniform-speed circle chart must admit the arc-length retraction"
    );

    let loss1 = term.loss(target.view(), &rho).unwrap();
    let new_coords: Vec<f64> = term.assignment.coords[0]
        .as_matrix()
        .column(0)
        .to_vec();

    // (data-fit) invariant — image-frozen reconstruction.
    let df_scale = 1.0 + loss0.data_fit.abs();
    assert!(
        (loss1.data_fit - loss0.data_fit).abs() <= 1.0e-6 * df_scale,
        "data_fit must be invariant under the image-frozen retraction: {} vs {}",
        loss1.data_fit,
        loss0.data_fit
    );
    // (smoothness) invariant — the transport preserves BᵀSB.
    let sm_scale = 1.0 + loss0.smoothness.abs();
    assert!(
        (loss1.smoothness - loss0.smoothness).abs() <= 1.0e-6 * sm_scale,
        "smoothness must be invariant (transport preserves BᵀSB): {} vs {}",
        loss1.smoothness,
        loss0.smoothness
    );
    // (ARD) the identifying coordinate prior actually MOVED.
    assert!(
        (loss1.ard - loss0.ard).abs() > 1.0e-6,
        "ARD coordinate prior must change under the reparam (it pins t → ±t + c)"
    );

    // (ARD is EXACTLY the reparam effect) the coord-dependent von-Mises energy
    // delta at the moved coords equals the ARD change; the ard_value normalizer
    // depends only on (alpha, n, period) and cancels.
    let mut expected_delta = 0.0_f64;
    for i in 0..n {
        expected_delta += ArdAxisPrior::eval(alpha, new_coords[i], Some(period)).value
            - ArdAxisPrior::eval(alpha, old_coords[i], Some(period)).value;
    }
    let ard_delta = loss1.ard - loss0.ard;
    assert!(
        (ard_delta - expected_delta).abs() <= 1.0e-9 * (1.0 + expected_delta.abs()),
        "ARD delta {ard_delta} must equal the von-Mises energy delta at the reparam'd coords \
         {expected_delta} (confirms it is the reparam effect, not a bookkeeping bug)"
    );

    // (gradient) assemble↔loss consistency at the canonical chart: central FD of
    // the total loss w.r.t. a decoder (β) entry — β is Euclidean, no manifold
    // projection — must match the assembled β-gradient. Guards against an
    // assemble/line-search desync introduced by the re-gauge.
    let sys = term.assemble_arrow_schur(target.view(), &rho, None).unwrap();
    let (bm, bp) = (1usize, 0usize);
    let beta_idx = bm * p + bp; // single atom: β flattened row-major (m × p)
    let g_analytic = sys.gb[beta_idx];
    let h = 1.0e-6_f64;
    let base = term.atoms[0].decoder_coefficients[[bm, bp]];
    term.atoms[0].decoder_coefficients[[bm, bp]] = base + h;
    let lp = term.loss(target.view(), &rho).unwrap().total();
    term.atoms[0].decoder_coefficients[[bm, bp]] = base - h;
    let lm = term.loss(target.view(), &rho).unwrap().total();
    term.atoms[0].decoder_coefficients[[bm, bp]] = base; // restore
    let g_fd = (lp - lm) / (2.0 * h);
    assert!(
        (g_analytic - g_fd).abs() <= 1.0e-4 * (1.0 + g_fd.abs()),
        "assembled β-gradient {g_analytic} must match FD {g_fd} at the canonical chart"
    );

    // (idempotence) re-applying at the next refresh boundary is a no-op (already
    // unit-speed) — safe to call every boundary without perturbing the fit.
    let coords_before: Vec<f64> = term.assignment.coords[0].as_matrix().column(0).to_vec();
    let _ = term.canonicalize_atom_unit_speed_chart(0, &topo).unwrap();
    let coords_after: Vec<f64> = term.assignment.coords[0].as_matrix().column(0).to_vec();
    let drift: f64 = coords_before
        .iter()
        .zip(coords_after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        drift <= 1.0e-6,
        "the retraction must be idempotent at a refresh boundary; max coord drift {drift}"
    );
    let loss2 = term.loss(target.view(), &rho).unwrap();
    assert!(
        (loss2.ard - loss1.ard).abs() <= 1.0e-6 * (1.0 + loss1.ard.abs()),
        "a second retraction must not move the objective (idempotent)"
    );
}
