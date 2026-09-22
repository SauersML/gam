//! gam#3173: the fold record's softest direction, and the start past the saddle it names.
//!
//! The record already measured how far the Laplace series' leading correction along `v` is from
//! being a correction. What it did not carry was `v` itself, so nothing could act on the verdict.
//! These pins are on the closed form of the cubic model the record is derived from, not on any
//! fitted surface.

use super::*;
use ndarray::{Array1, Array2, array};

/// A record with the fields the crossing is derived from, and nothing else pretending to be
/// measured.
fn record(sigma: f64, third: Option<f64>, share: Option<f64>) -> InnerModeFold {
    InnerModeFold {
        sigma,
        rounding_band: 0.0,
        softest_direction: array![1.0, 0.0],
        third_derivative: third,
        cubic_correction: share,
        quartic_correction: QuarticShare::NotPriced,
        completion: third.map(|_| CompletionShare::Priced),
    }
}

/// gam#3173: `saddle_crossing_displacement` lands past the OTHER stationary point of the cubic
/// model, and names no crossing where the model has no other stationary point.
///
/// Along `v` the model is `f(β̂ + s·v) = f̂ + ½σs² + t₃s³/6`, so `f'(s) = σs + t₃s²/2` vanishes at
/// `s = 0` and at `s* = −2σ/t₃`. The displacement must be `2s*·v`: past the saddle, so a solve
/// started there leaves this basin. The values are chosen so `s*` and `f'(s*)` are exact in binary
/// — `σ = 1/4`, `t₃ = −1` puts `s* = 1/2` and `f'(s*) = 1/8 − 1/8` — and the assertion is
/// therefore equality, not a band.
#[test]
fn the_crossing_lands_past_the_cubic_model_s_other_stationary_point_3173() {
    let sigma = 0.25_f64;
    let third = -1.0_f64;
    let saddle = -2.0 * sigma / third;
    assert_eq!(saddle, 0.5, "the fixture's saddle is exact in binary");
    assert_eq!(
        sigma * saddle + third * saddle * saddle / 2.0,
        0.0,
        "the saddle is a stationary point of the cubic model, exactly"
    );

    let displacement = record(sigma, Some(third), Some(13.0))
        .saddle_crossing_displacement()
        .expect("a resolved, non-zero t3 places the saddle");
    assert_eq!(
        displacement,
        array![2.0 * saddle, 0.0],
        "the crossing is twice the saddle's step along the softest direction"
    );

    // A vanishing cubic term puts the saddle at infinity: the model is a parabola along `v` and
    // there is no second basin to cross into.
    assert!(
        record(sigma, Some(0.0), Some(0.0))
            .saddle_crossing_displacement()
            .is_none(),
        "a curvature that does not move along v names no crossing"
    );
    // And a record whose `t3` was never priced names none either.
    assert!(
        record(sigma, None, None)
            .saddle_crossing_displacement()
            .is_none(),
        "an unpriced third derivative places no saddle"
    );
}

/// gam#3173: the probe's condition is that the leading correction is not below the term it
/// corrects — the cubic share at or above one — and it is read from the share alone.
///
/// The share is `5t₃²/(24σ³) = 5/(36·ΔF)` for the cubic model's barrier `ΔF = (2/3)σ³/t₃²`, so the
/// condition reads "the barrier is at or below `5/36` in log-likelihood units". The boundary is
/// pinned at one and at the two representable neighbours of one, so the condition is the
/// inequality it claims and not an interval around it. An unpriced share is not a low barrier: it
/// is no evidence at all.
#[test]
fn the_probe_fires_where_the_correction_is_not_below_what_it_corrects_3173() {
    let at = |share: Option<f64>| record(0.25, Some(-1.0), share).barrier_is_below_its_own_correction();
    assert!(at(Some(1.0)), "a correction equal to its term is not below it");
    assert!(at(Some(1.0 + f64::EPSILON)));
    assert!(!at(Some(1.0 - f64::EPSILON / 2.0)), "one ulp below one is below one");
    assert!(!at(Some(0.0)));
    assert!(!at(None), "an unpriced share is no evidence");

    // The share and the barrier are one quantity: `share = 5/(36 ΔF)` at the fixture's own
    // numbers, so the boundary `share = 1` is the barrier `5/36`.
    let sigma = 0.25_f64;
    let third = -1.0_f64;
    let share = 5.0 * third * third / (24.0 * sigma.powi(3));
    let barrier = 2.0 * sigma.powi(3) / (3.0 * third * third);
    assert!(
        (share * barrier - 5.0 / 36.0).abs() <= 4.0 * f64::EPSILON * (5.0 / 36.0),
        "share x barrier is 5/36 to the four roundings of the two products: {share} x {barrier}"
    );
}

/// gam#3173: the grading carries the softest eigenvector on EVERY record, priced or refused.
///
/// The direction is what the record is measured along, and the band-refused path has it in hand
/// just as the priced path does. A record that reached the caller without it could not be acted
/// on, which is why `t₃` was never worth pricing before.
#[test]
fn every_graded_record_carries_the_direction_it_was_measured_along_3173() {
    let identity = Array2::<f64>::eye(2);
    let refused = grade_inner_mode_fold(
        &InvertedSpan {
            basis: identity.clone(),
            eigenvalues: vec![0.0, 5.0],
        },
        1.0,
        Some(&|_| panic!("an unresolved curvature is refused before t3 is priced")),
    )
    .expect("the rounding band grades without pricing");
    assert!(!refused.is_valid());
    assert_eq!(refused.softest_direction, array![1.0, 0.0]);

    let priced = grade_inner_mode_fold(
        &InvertedSpan {
            basis: identity,
            eigenvalues: vec![1.0e-7, 5.0],
        },
        1.0,
        Some(&|direction: &Array1<f64>| Ok((direction[0].powi(3), CompletionShare::Priced))),
    )
    .expect("a resolved curvature is graded");
    assert!(priced.is_valid());
    assert_eq!(
        priced.softest_direction,
        array![1.0, 0.0],
        "the priced path carries the same direction it handed the third-derivative hook"
    );
    assert!(
        priced.saddle_crossing_displacement().is_some(),
        "a priced, non-zero t3 on a resolved curvature names its crossing: {priced}"
    );
}
