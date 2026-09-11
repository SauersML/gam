//! #2412: "railed" must mean the same box to the search loop and to the
//! terminal certificate.
//!
//! A smoothing coordinate whose variance component is genuinely zero drives
//! its λ toward the infinite-smoothing ceiling asymptotically: every outer step
//! shrinks the gap but none closes it, so the iterate lands strictly inside the
//! box. The rail detector calls that railed by margin; the exact `x >= upper`
//! test in `project_gradient_vector` calls it interior. While those two
//! disagreed, the cost-stall guard measured a stationarity residual made almost
//! entirely of an outward rail pull that the certificate would have discarded,
//! never reached a stationary verdict, and let the solver run to its iteration
//! cap on points the certificate goes on to accept.

use super::bridges::{
    project_gradient_vector, projected_gradient_norm, rail_projected_gradient_norm,
    rail_relaxed_bounds, reduced_hessian_psd_at_point,
};
use super::{
    OuterConfig, StationarityBound, StationarityBoundSource, certificate_hessian_is_psd_off_railed,
    certificate_railed_coordinates, certify_interior_stationarity, interior_face_indices,
    newton_predicted_decrease,
};
use crate::model_types::CERTIFICATE_RAIL_MARGIN;
use gam_terms::smooth::CONSTANT_CURVATURE_KAPPA_CHART_FRACTION;
use ndarray::{Array1, array};

/// The default outer search box, wide enough that the margin is not width-capped.
fn wide_box(n: usize) -> (Array1<f64>, Array1<f64>) {
    (
        Array1::from_elem(n, -30.0),
        Array1::from_elem(n, 30.0),
    )
}

#[test]
fn a_coordinate_creeping_onto_the_ceiling_is_railed_for_the_residual() {
    // The reported shape: ρ parked just short of the ceiling with a large
    // gradient pulling further out (λ wants to keep growing). At the upper
    // bound the outward direction is g < 0, since the step is -g.
    let bounds = wide_box(1);
    let x = Array1::from_vec(vec![29.9938]);
    let gradient = Array1::from_vec(vec![-158.8]);

    // The raw box calls it interior and keeps the whole outward pull, which is
    // what let a rail masquerade as a stationarity residual.
    let raw = projected_gradient_norm(&x, &gradient, Some(&bounds));
    assert!((raw - 158.8).abs() < 1e-9, "raw projection kept {raw}");

    // The rail-relaxed box recognizes it and drops the KKT-multiplier part.
    let railed = rail_projected_gradient_norm(&x, &gradient, Some(&bounds));
    assert_eq!(railed, 0.0, "rail-relaxed projection kept {railed}");
}

#[test]
fn the_lower_ceiling_behaves_the_same_way() {
    // Mirror case: ρ creeping onto the λ→0 floor, outward pull is g > 0.
    let bounds = wide_box(1);
    let x = Array1::from_vec(vec![-29.9938]);
    let gradient = Array1::from_vec(vec![42.5]);

    assert!((projected_gradient_norm(&x, &gradient, Some(&bounds)) - 42.5).abs() < 1e-9);
    assert_eq!(rail_projected_gradient_norm(&x, &gradient, Some(&bounds)), 0.0);
}

#[test]
fn feasible_descent_at_a_rail_is_never_discarded() {
    // The safety property that makes relaxing the box sound: only the OUTWARD
    // half is zeroed. A near-ceiling coordinate whose gradient still points
    // back INTO the box is not at an optimum, and must keep its residual under
    // both projections or the relaxation would manufacture certifications.
    let bounds = wide_box(1);
    let x = Array1::from_vec(vec![29.9938]);
    let gradient = Array1::from_vec(vec![7.25]);

    assert!((projected_gradient_norm(&x, &gradient, Some(&bounds)) - 7.25).abs() < 1e-9);
    assert!((rail_projected_gradient_norm(&x, &gradient, Some(&bounds)) - 7.25).abs() < 1e-9);
}

#[test]
fn interior_coordinates_are_untouched_by_the_relaxation() {
    // Well inside the box, the two projections must agree exactly — the
    // relaxation may not perturb ordinary interior optimization.
    let bounds = wide_box(3);
    let x = Array1::from_vec(vec![0.0, -4.5, 11.25]);
    let gradient = Array1::from_vec(vec![1.5, -2.5, 0.75]);

    let raw = projected_gradient_norm(&x, &gradient, Some(&bounds));
    let railed = rail_projected_gradient_norm(&x, &gradient, Some(&bounds));
    assert!((raw - railed).abs() < 1e-12, "raw {raw} vs railed {railed}");
}

#[test]
fn a_railed_coordinate_does_not_mask_an_interior_one() {
    // The mixed case the guard actually sees: one coordinate on the ceiling,
    // one genuinely far from stationary. The rail is removed; the interior
    // residual survives in full, so the guard still refuses to certify.
    let bounds = wide_box(2);
    let x = Array1::from_vec(vec![29.9938, 0.0]);
    let gradient = Array1::from_vec(vec![-158.8, 3.0]);

    assert!((rail_projected_gradient_norm(&x, &gradient, Some(&bounds)) - 3.0).abs() < 1e-9);
}

#[test]
fn unbounded_problems_are_unaffected() {
    let x = Array1::from_vec(vec![1.0, -2.0]);
    let gradient = Array1::from_vec(vec![3.0, -4.0]);
    let expected = 5.0_f64;
    assert!((rail_projected_gradient_norm(&x, &gradient, None) - expected).abs() < 1e-12);
}

#[test]
fn the_margin_is_the_certificate_margin_on_a_wide_box() {
    let (lower, upper) = rail_relaxed_bounds(&wide_box(1));
    assert!((lower[0] - (-30.0 + CERTIFICATE_RAIL_MARGIN)).abs() < 1e-12);
    assert!((upper[0] - (30.0 - CERTIFICATE_RAIL_MARGIN)).abs() < 1e-12);
}

#[test]
fn a_narrow_box_is_relaxed_without_inverting() {
    // A box narrower than twice the margin would invert under a flat
    // relaxation, making every point read as railed at BOTH ends and zeroing
    // the coordinate's residual outright — a silent false certification on
    // exactly the tightly-boxed coordinates that most need a real one. The
    // margin is capped at a quarter width, so the relaxed interval keeps at
    // least half the original and the endpoints stay ordered.
    let bounds = (
        Array1::from_vec(vec![-0.4]),
        Array1::from_vec(vec![0.4]),
    );
    let (lower, upper) = rail_relaxed_bounds(&bounds);
    assert!(lower[0] < upper[0], "relaxed box inverted: {lower:?} {upper:?}");
    assert!((upper[0] - lower[0]) >= 0.5 * 0.8 - 1e-12);

    // A point at the centre of a narrow box is not railed, so a real residual
    // there still counts.
    let x = Array1::from_vec(vec![0.0]);
    let gradient = Array1::from_vec(vec![2.0]);
    assert!((rail_projected_gradient_norm(&x, &gradient, Some(&bounds)) - 2.0).abs() < 1e-12);

    // Ask the OTHER consumer the same question. This assertion is the one this
    // test was missing: the claim above was only ever put to the projector,
    // and the certificate's rail flag answered it the opposite way on this
    // very box (`|0.0 − (−0.4)| = 0.4 ≤ CERTIFICATE_RAIL_MARGIN`) for as long
    // as it carried its own margin law (#2462).
    let config = boxed_config(bounds.0.clone(), bounds.1.clone());
    assert!(
        certificate_railed_coordinates(&x, &config).is_empty(),
        "the centre of a narrow box must read interior to the certificate too, \
         not only to the residual projector",
    );
}

#[test]
fn a_rail_direction_does_not_poison_the_curvature_verdict() {
    // The same disagreement reaching the curvature test. A coordinate running
    // its smoothing parameter to the ceiling is still descending along that
    // direction, so its curvature reads indefinite. The `1e-10` strict-activity
    // test inside the reduction cannot see a bound that is only approached in
    // the limit, so it keeps the direction in the critical cone and returns
    // NOT-PSD — blocking the guard's converged verdict on a direction that is
    // not in the cone at all.
    let bounds = wide_box(2);
    let x = array![29.9938, 0.0];
    let gradient = array![-4.0, 0.0]; // outward at the upper bound
    let hessian = array![[-3.0, 0.0], [0.0, 2.0]];

    assert_eq!(
        reduced_hessian_psd_at_point(&x, &gradient, &hessian, Some((&bounds.0, &bounds.1)), 0.0),
        Some(false),
        "the raw box keeps the rail direction in the critical cone"
    );

    let relaxed = rail_relaxed_bounds(&bounds);
    assert_eq!(
        reduced_hessian_psd_at_point(&x, &gradient, &hessian, Some((&relaxed.0, &relaxed.1)), 0.0),
        Some(true),
        "the rail direction leaves the cone; the interior block is judged alone"
    );
}

#[test]
fn a_near_bound_coordinate_with_feasible_descent_keeps_its_curvature() {
    // Only coordinates pushing OUT of the box leave the critical cone. One near
    // the ceiling whose gradient points back INTO it is not at an optimum, so
    // its curvature must still be judged — otherwise the relaxation would start
    // certifying saddles rather than rails.
    let bounds = wide_box(2);
    let relaxed = rail_relaxed_bounds(&bounds);
    let x = array![29.9938, 0.0];
    let gradient = array![4.0, 0.0]; // feasible descent, back into the box
    let hessian = array![[-3.0, 0.0], [0.0, 2.0]];

    assert_eq!(
        reduced_hessian_psd_at_point(&x, &gradient, &hessian, Some((&relaxed.0, &relaxed.1)), 0.0),
        Some(false),
        "a near-bound coordinate that is not railed keeps its negative curvature"
    );
}

#[test]
fn the_guard_stays_no_more_permissive_than_the_certificate() {
    // The certificate drops EVERY margin-railed coordinate before testing PSD,
    // with no gradient-sign condition. The guard's rule additionally requires
    // an outward gradient, so the guard's free set is a superset of the
    // certificate's and its verdict can never be more permissive. Here the
    // railed coordinate has feasible-descent gradient: the certificate would
    // drop it, the guard keeps it, and the guard therefore reports the stricter
    // NOT-PSD.
    let bounds = wide_box(2);
    let relaxed = rail_relaxed_bounds(&bounds);
    let x = array![29.9938, 0.0];
    let gradient = array![1.0, 0.0];
    let hessian = array![[-3.0, 0.0], [0.0, 2.0]];

    let guard = reduced_hessian_psd_at_point(
        &x,
        &gradient,
        &hessian,
        Some((&relaxed.0, &relaxed.1)),
        0.0,
    );
    assert_eq!(guard, Some(false));
}

#[test]
fn a_fixed_coordinate_gets_no_relaxation() {
    // `lower == upper` pins the coordinate; there is no interior to relax into,
    // and the exact bound test already handles it.
    let bounds = (
        Array1::from_vec(vec![2.0]),
        Array1::from_vec(vec![2.0]),
    );
    let (lower, upper) = rail_relaxed_bounds(&bounds);
    assert_eq!(lower[0], 2.0);
    assert_eq!(upper[0], 2.0);
}

// ─── #2462: the certificate's rail FLAG obeys the same margin law ────────
//
// The projector above caps the margin at a quarter of each coordinate's width.
// The flag did not: it applied a flat `CERTIFICATE_RAIL_MARGIN` to whatever box
// it was handed, so on any box narrower than twice that constant the two margin
// bands covered the whole interval and every feasible point read railed at both
// ends. The non-ρ blocks a joint search carries in the same θ vector are exactly
// that narrow.

/// The raw-κ window a constant-curvature term installs on the outer box:
/// `±CONSTANT_CURVATURE_KAPPA_CHART_FRACTION / R²`, as built by
/// `gam_terms::smooth::term_specs::constant_curvature_kappa_bounds`. Its
/// documented contract is that flat κ = 0 is the window's INTERIOR centre — the
/// reachability the raw-κ (not log-κ) coordinate exists to preserve.
fn constant_curvature_kappa_window(max_chart_radius2: f64) -> (Array1<f64>, Array1<f64>) {
    let half = CONSTANT_CURVATURE_KAPPA_CHART_FRACTION / max_chart_radius2;
    (array![-half], array![half])
}

fn boxed_config(lower: Array1<f64>, upper: Array1<f64>) -> OuterConfig {
    OuterConfig {
        model_domain_bounds: Some((lower, upper)),
        ..OuterConfig::default()
    }
}

#[test]
fn flat_curvature_at_the_centre_of_its_own_window_is_not_railed() {
    // R² = 18 is what standardised 2-D features (|z| ≲ 3) produce, so the κ
    // window is ±0.028 — two orders narrower than the rail margin.
    let (lower, upper) = constant_curvature_kappa_window(18.0);
    let width = upper[0] - lower[0];
    assert!(
        width < 2.0 * CERTIFICATE_RAIL_MARGIN,
        "this fixture only reaches the defect in the narrow-box regime, and its \
         width {width} is not below 2 x {CERTIFICATE_RAIL_MARGIN}",
    );
    // The reading the flat margin produced, stated directly: both bands reach
    // the centre, so "railed" carried no information about this coordinate.
    assert!(
        (0.0 - lower[0]).abs() <= CERTIFICATE_RAIL_MARGIN
            && (upper[0] - 0.0).abs() <= CERTIFICATE_RAIL_MARGIN,
    );

    let config = boxed_config(lower.clone(), upper.clone());
    assert_eq!(
        certificate_railed_coordinates(&array![0.0], &config),
        Vec::<usize>::new(),
        "flat kappa is the centre of its own window and must read interior",
    );
    // Both walls still register, so the cap did not simply switch rails off.
    assert_eq!(certificate_railed_coordinates(&lower, &config), vec![0]);
    assert_eq!(certificate_railed_coordinates(&upper, &config), vec![0]);
}

#[test]
fn a_narrow_window_still_gets_a_real_second_order_verdict() {
    // Why the margin law is load-bearing rather than cosmetic: railed rows and
    // columns are DELETED from the certificate's reduced Hessian, so a window
    // covered end to end by its own margin bands leaves an empty interior
    // sub-block and an indefinite curvature passes vacuously.
    let (lower, upper) = constant_curvature_kappa_window(18.0);
    let config = boxed_config(lower, upper);
    let railed = certificate_railed_coordinates(&array![0.0], &config);
    let indefinite = array![[-1.0]];
    assert_eq!(
        certificate_hessian_is_psd_off_railed(&indefinite, &railed, None),
        Some(false),
        "an indefinite curvature at an interior kappa must refuse",
    );
    assert_eq!(
        certificate_hessian_is_psd_off_railed(&indefinite, &[0], None),
        Some(true),
        "control: with the coordinate railed there is nothing left to judge",
    );
}

#[test]
fn the_rail_flag_is_the_exact_test_on_the_relaxed_box() {
    // One law, two layers: `railed` must be precisely `x <= relaxed_lower ||
    // x >= relaxed_upper`, on wide, narrow, tied and degenerate boxes alike.
    let boxes = [
        (-30.0, 30.0),                              // the default rho box
        (-0.5 / 18.0, 0.5 / 18.0),                  // a raw-kappa chart window
        (-0.39196610102242435, 0.6226807260860918), // the psi box measured on #2462
        (-1.0, 3.0),                                // width 4: cap and constant tie
        (2.0, 2.0),                                 // a fixed coordinate
    ];
    for (lo, hi) in boxes {
        let bounds = (array![lo], array![hi]);
        let config = boxed_config(bounds.0.clone(), bounds.1.clone());
        let (relaxed_lower, relaxed_upper) = rail_relaxed_bounds(&bounds);
        let width = hi - lo;
        let outside = width.max(1.0);
        for x in [
            lo - outside,
            lo,
            lo + 0.25 * width,
            0.5 * (lo + hi),
            hi - 0.25 * width,
            hi,
            hi + outside,
        ] {
            let flagged = !certificate_railed_coordinates(&array![x], &config).is_empty();
            let projected = x <= relaxed_lower[0] || x >= relaxed_upper[0];
            assert_eq!(
                flagged, projected,
                "flag and relaxed box disagree at x={x} on [{lo}, {hi}]",
            );
        }
    }
}

#[test]
fn the_centre_of_a_box_is_never_railed() {
    // The invariant the quarter-width cap buys: the two bands together cover at
    // most half the interval, so a coordinate sitting at the middle of its own
    // domain is interior at every width.
    for width in [1e-6, 1e-3, 0.05, 0.5, 1.0, 1.9, 2.0, 2.1, 4.0, 60.0] {
        let config = boxed_config(array![-0.5 * width], array![0.5 * width]);
        assert!(
            certificate_railed_coordinates(&array![0.0], &config).is_empty(),
            "the centre of a width-{width} box must read interior",
        );
    }
}

#[test]
fn a_coordinate_outside_its_box_is_railed() {
    // The absolute-value form reported a point far OUTSIDE the box as interior,
    // because `|theta - lower|` exceeded the margin on the infeasible side.
    let config = boxed_config(array![-5.0], array![5.0]);
    assert_eq!(
        certificate_railed_coordinates(&array![-12.0], &config),
        vec![0]
    );
    assert_eq!(
        certificate_railed_coordinates(&array![12.0], &config),
        vec![0]
    );
}

// ─── #2471: the interior face PROJECTS railed rows, it does not DELETE them ──

#[test]
fn a_railed_coordinate_that_can_still_descend_stays_in_the_interior() {
    // Coordinate 1 sits on its upper bound with a gradient pointing back INTO
    // the box, so `project_gradient_vector` keeps it — that is feasible
    // descent, not a KKT multiplier. Coordinate 2 sits on the same wall with
    // its pull heading out, and is zeroed.
    let bounds = (array![-30.0, -30.0, -30.0], array![30.0, 30.0, 30.0]);
    let x = array![0.0, 30.0, 30.0];
    let gradient = array![0.1, 12.018, -4.0];
    let projected = project_gradient_vector(&x, &gradient, Some(&bounds));
    assert_eq!(projected[1], 12.018, "inward pull at an upper bound survives");
    assert_eq!(projected[2], 0.0, "outward pull at an upper bound is dropped");

    // Both 1 and 2 are railed by the flag; only 2 has actually been pinned.
    let railed = [1usize, 2];
    assert_eq!(interior_face_indices(&projected, &railed), vec![0, 1]);
}

#[test]
fn the_interior_face_norm_is_exactly_the_projected_gradient_norm() {
    // The identity the fix rests on: the projection either keeps a component
    // unchanged or zeroes it, never partially, so summing the kept indices
    // reproduces |Pg| bit-for-bit. Deleting every railed row does not.
    let bounds = (array![-30.0, -30.0, -30.0], array![30.0, 30.0, 30.0]);
    let x = array![0.0, 30.0, -30.0];
    let gradient = array![0.1986, 12.018, 3.5];
    let projected = project_gradient_vector(&x, &gradient, Some(&bounds));
    let railed = [1usize, 2];

    let face = interior_face_indices(&projected, &railed);
    let face_norm = face
        .iter()
        .map(|&k| projected[k] * projected[k])
        .sum::<f64>()
        .sqrt();
    let pg_norm = projected_gradient_norm(&x, &gradient, Some(&bounds));
    assert_eq!(
        face_norm.to_bits(),
        pg_norm.to_bits(),
        "interior-face norm must BE |Pg|, not merely approximate it",
    );

    // What deletion reported instead, on this issue's own magnitudes.
    let deleted_norm = (0..3)
        .filter(|k| !railed.contains(k))
        .map(|k| projected[k] * projected[k])
        .sum::<f64>()
        .sqrt();
    assert!(
        (deleted_norm - 0.1986).abs() < 1e-12 && face_norm > 12.0,
        "deletion drops a coordinate carrying 60x the residual: \
         deleted={deleted_norm:.4e} projected={face_norm:.4e}",
    );
}

#[test]
fn a_full_space_curvature_rung_cannot_bypass_the_active_face_decrement_2559() {
    // Coordinate 1 is a KKT-pinned rail: its projected residual is exactly
    // zero, but its enormous curvature controls the FULL decrement's shared
    // regularization shift. That makes the full-space decrement appear below
    // objective resolution while coordinate 0 still carries order-one descent
    // on the exact active face.
    let projected = array![1.0, 0.0];
    let hessian = array![[1.0, 0.0], [0.0, 1.0e16]];
    let objective_tol = 1.0e-6;
    let full_decrement =
        newton_predicted_decrease(&hessian, &projected).expect("full PD decrement");
    assert!(
        full_decrement <= objective_tol,
        "the fixture must make the full-space curvature rung admissible: \
         decrement={full_decrement:.3e}"
    );
    let projected_norm = projected.dot(&projected).sqrt();
    let full_curvature_bound = projected_norm * (objective_tol / full_decrement).sqrt();
    assert!(
        projected_norm <= full_curvature_bound,
        "control: the old early return must accept this full-space rung"
    );

    let face = interior_face_indices(&projected, &[1]);
    assert_eq!(face, vec![0]);
    let face_decrement =
        newton_predicted_decrease(&array![[1.0]], &array![1.0]).expect("face PD decrement");
    assert!(
        face_decrement > objective_tol,
        "the exact active face must retain resolvable descent: \
         decrement={face_decrement:.3e}"
    );
    let refusal = certify_interior_stationarity(
        &projected,
        &hessian,
        &face,
        StationarityBound::from_ladder(
            full_curvature_bound,
            StationarityBoundSource::CurvatureResolvability,
        ),
        objective_tol,
    )
    .expect_err("full-space curvature evidence must not certify a different face");
    assert!(
        refusal.contains("active-face Newton decrement")
            && refusal.contains("curvature-resolvability"),
        "the refusal must name the face-local evidence and rejected currency: {refusal}"
    );
}

#[test]
fn an_admissible_curvature_rung_is_reminted_from_the_active_face_2559() {
    // Both spaces have a sub-resolution decrement, but the railed coordinate's
    // scale makes their bounds observably different. The certificate must
    // publish the bound formed from H_face/g_face, never the caller's value.
    let projected = array![1.0, 0.0];
    let hessian = array![[1.0e8, 0.0], [0.0, 1.0e16]];
    let objective_tol = 1.0e-6;
    let full_decrement =
        newton_predicted_decrease(&hessian, &projected).expect("full PD decrement");
    let full_curvature_bound = (objective_tol / full_decrement).sqrt();
    let face = interior_face_indices(&projected, &[1]);
    let face_decrement =
        newton_predicted_decrease(&array![[1.0e8]], &array![1.0]).expect("face PD decrement");
    assert!(face_decrement <= objective_tol);
    let expected_face_bound = (objective_tol / face_decrement).sqrt();
    assert_ne!(
        full_curvature_bound.to_bits(),
        expected_face_bound.to_bits(),
        "the fixture must distinguish the full-space and active-face currencies"
    );

    let (face_norm, effective_bound) = certify_interior_stationarity(
        &projected,
        &hessian,
        &face,
        StationarityBound::from_ladder(
            full_curvature_bound,
            StationarityBoundSource::CurvatureResolvability,
        ),
        objective_tol,
    )
    .expect("a sub-resolution active-face decrement must certify");
    assert_eq!(face_norm.to_bits(), 1.0_f64.to_bits());
    assert_eq!(
        effective_bound.value().to_bits(),
        expected_face_bound.to_bits(),
        "the emitted bound must be recomputed from the active-face decrement"
    );
    assert_eq!(effective_bound.rung().label, "curvature-resolvability");
    assert!(effective_bound.rung().derived_standard);
}

#[test]
fn a_gradient_magnitude_rung_remains_direct_first_order_currency_2559() {
    // The exact face residual is already the same gradient-magnitude currency
    // as a solver band. Its own configured first-order authority may admit the
    // point without consulting the optional curvature widening; the caller's
    // separate reduced-PSD gate still owns second-order admissibility.
    let projected = array![1.0e-6, 0.0];
    let hessian = array![[1.0e-12, 0.0], [0.0, 1.0e16]];
    let face = interior_face_indices(&projected, &[1]);
    let objective_tol = 1.0e-12;
    let face_decrement =
        newton_predicted_decrease(&array![[1.0e-12]], &array![1.0e-6]).expect("face PD decrement");
    assert!(
        face_decrement > objective_tol,
        "the fixture must distinguish first-order authority from curvature widening"
    );
    let solver_bound = 2.0e-6;
    let (_, effective_bound) = certify_interior_stationarity(
        &projected,
        &hessian,
        &face,
        StationarityBound::from_ladder(solver_bound, StationarityBoundSource::SolverBand),
        objective_tol,
    )
    .expect("a same-currency gradient band must remain authoritative");
    assert_eq!(effective_bound.value().to_bits(), solver_bound.to_bits());
    assert_eq!(effective_bound.rung().label, "solver-band");
    assert!(!effective_bound.rung().derived_standard);
}

#[test]
fn the_interior_face_can_only_grow_never_shrink() {
    // The safety argument, made executable: whatever the railed set says, the
    // projected face is a SUPERSET of the deleted one, so no nonzero KKT
    // residual disappears. Curvature is not inferred from that set relation:
    // the consumer recomputes it on the exact face (#2559).
    let bounds = (Array1::from_elem(6, -30.0), Array1::from_elem(6, 30.0));
    for pattern in 0..64u32 {
        let x = Array1::from_iter(
            (0..6).map(|k| if pattern >> k & 1 == 1 { 30.0 } else { 0.0 }),
        );
        let gradient = array![1.0, -1.0, 0.0, 4.0, -4.0, 1e-13];
        let projected = project_gradient_vector(&x, &gradient, Some(&bounds));
        let railed: Vec<usize> = (0..6).filter(|&k| x[k] >= 30.0).collect();
        let face = interior_face_indices(&projected, &railed);
        let deleted: Vec<usize> = (0..6).filter(|k| !railed.contains(k)).collect();
        assert!(
            deleted.iter().all(|k| face.contains(k)),
            "pattern {pattern}: deleted set {deleted:?} escaped the face {face:?}",
        );
    }
}
