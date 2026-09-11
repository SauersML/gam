//! #1082: the search route's reduced-Hessian definiteness verdict, judged at the
//! criterion's curvature resolution. A negative eigenvalue that one e-fold of
//! `log λ` cannot turn into a resolvable decrease is not a strict saddle; a
//! resolvable one still is, and a zero resolution is today's verdict.

use super::{incumbent_curvature, rail_relaxed_bounds, reduced_hessian_psd_at_point};
use ndarray::array;

/// The penguin arm-1 unbiased probe incumbent (job 383312): the analytic
/// `λ_min` on the judged sub-block against the criterion resolution the
/// terminal adjudication measured at the same point.
const PENGUIN_LAMBDA_MIN_1082: f64 = -6.695751e-7;
const PENGUIN_RESOLUTION_1082: f64 = 2.525e-5;

#[test]
fn sub_resolution_negative_curvature_is_not_a_strict_saddle_1082() {
    let x = array![0.5, 0.5];
    let gradient = array![1.0e-4, 1.0e-4];
    let (lower, upper) = (array![-30.0, -30.0], array![30.0, 30.0]);
    let bounds = Some((&lower, &upper));
    let incumbent = array![[1.0, 0.0], [0.0, PENGUIN_LAMBDA_MIN_1082]];
    assert_eq!(
        reduced_hessian_psd_at_point(&x, &gradient, &incumbent, bounds, 0.0),
        Some(false),
        "at the arithmetic shift alone the incumbent reads as a strict saddle"
    );
    assert_eq!(
        reduced_hessian_psd_at_point(
            &x,
            &gradient,
            &incumbent,
            bounds,
            2.0 * PENGUIN_RESOLUTION_1082,
        ),
        Some(true),
        "a direction whose unit-step decrease ½|λ| is below the criterion's resolution is not \
         resolvable curvature"
    );
    // Straddle the resolution: `½|λ|` twice the resolution must still refuse.
    let resolvable = array![[1.0, 0.0], [0.0, -4.0 * PENGUIN_RESOLUTION_1082]];
    assert_eq!(
        reduced_hessian_psd_at_point(
            &x,
            &gradient,
            &resolvable,
            bounds,
            2.0 * PENGUIN_RESOLUTION_1082,
        ),
        Some(false),
        "a direction whose unit-step decrease exceeds the resolution is still a strict saddle"
    );
}

/// #1082: a strict-saddle refusal names the curvature it refused on. Coordinate 1
/// sits on its upper rail with an inward-descent gradient, so the guard keeps it in
/// its free set and reads a resolvable negative eigenvalue there, while the
/// certificate's interior (margin-railed coordinates removed) is positive definite.
#[test]
fn strict_saddle_facts_separate_the_free_set_from_the_interior_1082() {
    let (lower, upper) = (array![-30.0, -30.0], array![30.0, 30.0]);
    let relaxed = rail_relaxed_bounds(&(lower, upper));
    let x = array![0.5, 30.0];
    let gradient = array![1.0e-4, 1.0e-4];
    let hessian = array![[1.0, 0.0], [0.0, -1.0e-2]];
    let resolution = 2.0 * PENGUIN_RESOLUTION_1082;
    let bounds = Some((&relaxed.0, &relaxed.1));
    assert_eq!(
        reduced_hessian_psd_at_point(&x, &gradient, &hessian, bounds, resolution),
        Some(false),
        "the railed inward-gradient coordinate stays in the guard's free set"
    );
    let facts = incumbent_curvature(&x, &gradient, &hessian, bounds, resolution)
        .expect("facts at a strict-saddle verdict");
    let rendered = facts.to_string();
    assert!(rendered.contains("free-set λ_min=-1.000e-2"), "{rendered}");
    assert!(rendered.contains("interior λ_min=1.000e0"), "{rendered}");
    assert!(rendered.contains("railed=[#1 ρ=3.0000e1"), "{rendered}");
}
