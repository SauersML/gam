//! #1082: the search route's reduced-Hessian definiteness verdict, judged at the
//! criterion's curvature resolution. A negative eigenvalue that one e-fold of
//! `log λ` cannot turn into a resolvable decrease is not a strict saddle; a
//! resolvable one still is, and a zero resolution is today's verdict.

use super::reduced_hessian_psd_at_point;
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
