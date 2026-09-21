//! What an I-spline basis does OUTSIDE its knot range, as a declared policy
//! rather than as a convention each consumer re-derives.
//!
//! # Why this is a module and not a comment
//!
//! An I-spline value basis and its M-spline derivative are two readouts of one
//! function, and every consumer needs both: the value enters `η` and the
//! derivative enters a Jacobian, a hazard, or a density. Inside the knot range
//! the two agree because they are built from the same Cox–de Boor recursion.
//! OUTSIDE it they agree only if someone made them, because the two are
//! produced by different code paths — `create_ispline_dense` holds `I_k`
//! constant past the boundary knots, while the clamped B-spline first-derivative
//! basis it is a cumulative sum of returns the *boundary slope*, since a clamped
//! B-spline VALUE extends linearly (`apply_linear_extension_from_first_derivative`).
//!
//! That mismatch has been found and repaired three times, in three places, each
//! time as a separate defect:
//!
//! * gam#1348 — the open-knot B-spline derivative did not match a finite
//!   difference of the constant-extended value;
//! * gam#2695 — the survival link warp's `m1 = 1 + Σ βw_j·I'_j(q₀)` picked up a
//!   slope the flat warp value does not have, and the joint-Newton RHS asserted
//!   a first-order change the objective does not make. Fixed by zeroing
//!   `create_ispline_derivative_dense`'s exterior;
//! * gam#2600 — the CTN transformation's whole exterior was a readout of the
//!   `1e-8` monotonicity floor, because `I_k` saturated and `M_k` was zero.
//!   Fixed by continuing BOTH affinely, in the CTN chart;
//! * gam#2705 (this module) — the Royston-Parmar baseline still hand-rolled the
//!   cumulative sum from a clamped B-spline derivative and so never received
//!   gam#2695's repair, publishing a FLAT `Λ(t)` next to a NONZERO `h(t)` past
//!   the training support. Two statements that cannot both describe one model.
//!
//! Four repairs of one defect is a missing abstraction, not four bugs. This
//! module is that abstraction: [`ISplineBoundary`] names the two coherent
//! conventions, [`ispline_value_and_first_derivative`] produces the value and
//! the derivative together under ONE of them, and a consumer that wants only a
//! value still says which convention it is in.
//!
//! # The two conventions, and when each is right
//!
//! Write `[left, right] = [t_{d+1}, t_{n_B}]` for the modelling interval of the
//! internal degree-`d+1` B-spline frame, and `M_k = I_k′`.
//!
//! * [`ISplineBoundary::Saturate`] — `I_k(x) = I_k(b)` and `M_k(x) = 0` for `x`
//!   outside. Every column stays in `[0, 1]` and non-decreasing, which is what a
//!   consumer reading the columns as a BOUNDED warp needs (gam#2695). Its
//!   modelling content is "the fitted object stops changing past the data".
//!
//! * [`ISplineBoundary::LinearTails`] — `I_k(x) = I_k(b) + (x − b)·M_k(b∓)` and
//!   `M_k(x) = M_k(b∓)`. `C¹`, still non-decreasing (`M_k ≥ 0`), and the columns
//!   leave `[0, 1]`. Its modelling content is "the fitted object keeps the trend
//!   it had at the edge of the data", which is the classical answer for a
//!   transformation or a log-cumulative-hazard: Royston & Parmar's restricted
//!   splines are LINEAR beyond the boundary knots by construction, and `mlt`'s
//!   `extrapolate` does the same thing for the same reason.
//!
//! Neither is universally right, which is exactly why it is a parameter. What is
//! never right is one of them for the value and the other for the derivative.

use ndarray::{Array1, Array2, ArrayView1};

use super::{
    BasisError, BasisOptions, Dense, KnotSource, create_basis, create_ispline_derivative_dense,
};

/// The boundary convention an I-spline value/derivative pair is evaluated
/// under. See the module documentation for the derivation and for which
/// consumers want which.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ISplineBoundary {
    /// Constant extension: `I_k(x) = I_k(b)`, `M_k(x) = 0`. The historical
    /// (and still default) convention of [`create_ispline_derivative_dense`]
    /// and of the shared I-spline evaluator.
    #[default]
    Saturate,
    /// Affine continuation at the one-sided boundary derivative:
    /// `I_k(x) = I_k(b) + (x − b)·M_k(b∓)`, `M_k(x) = M_k(b∓)`.
    LinearTails,
}

/// `[left, right] = [knots[d+1], knots[n_B]]` — the interval on which the
/// internal degree-`d+1` B-spline frame of a degree-`d` I-spline is a partition
/// of unity, and therefore the interval outside which an
/// [`ISplineBoundary`] applies.
///
/// Returns `Err` for a knot vector too short to carry the frame, and `Ok(None)`
/// for one whose interval is empty or non-finite — the caller then has no
/// exterior to extend and every convention agrees.
pub fn ispline_modelling_interval(
    knots: ArrayView1<'_, f64>,
    degree: usize,
) -> Result<Option<(f64, f64)>, BasisError> {
    let bspline_degree = degree
        .checked_add(1)
        .ok_or_else(|| BasisError::InvalidInput("I-spline degree overflow".to_string()))?;
    let bspline_columns = knots.len().checked_sub(bspline_degree + 1).ok_or_else(|| {
        BasisError::InvalidInput(format!(
            "I-spline knot vector of length {} cannot carry a degree-{bspline_degree} B-spline \
             frame",
            knots.len()
        ))
    })?;
    if bspline_columns == 0 || bspline_degree >= knots.len() {
        return Ok(None);
    }
    let left = knots[bspline_degree];
    let right = knots[bspline_columns];
    if !(left.is_finite() && right.is_finite() && left < right) {
        return Ok(None);
    }
    Ok(Some((left, right)))
}

/// The I-spline value basis and its first derivative at `data`, under ONE
/// declared boundary convention.
///
/// This is the entry point every consumer should use, because the failure mode
/// this module exists to remove is producing the two halves separately and
/// letting them disagree past the knots. Inside `[left, right]` the result is
/// bit-identical to `create_basis(.., i_spline())` /
/// `create_ispline_derivative_dense(.., 1)` under either convention.
pub fn ispline_value_and_first_derivative(
    data: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    boundary: ISplineBoundary,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let owned_knots = knots.to_owned();
    let (value_arc, _) = create_basis::<Dense>(
        data,
        KnotSource::Provided(knots),
        degree,
        BasisOptions::i_spline(),
    )?;
    let mut value = value_arc.as_ref().clone();
    let mut derivative = create_ispline_derivative_dense(data, &owned_knots, degree, 1)?;
    if derivative.ncols() != value.ncols() || derivative.nrows() != value.nrows() {
        return Err(BasisError::DimensionMismatch(format!(
            "I-spline derivative basis is {:?} but the value basis is {:?}",
            derivative.dim(),
            value.dim()
        )));
    }
    match boundary {
        ISplineBoundary::Saturate => {}
        ISplineBoundary::LinearTails => {
            extend_ispline_pair_affinely(data, knots, degree, &mut value, &mut derivative)?;
        }
    }
    Ok((value, derivative))
}

/// The I-spline value basis alone, under a declared boundary convention.
///
/// A consumer that needs only values still names the convention, so that a
/// later reader can see which function's exterior it is evaluating rather than
/// having to know the evaluator's default.
pub fn ispline_value(
    data: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    boundary: ISplineBoundary,
) -> Result<Array2<f64>, BasisError> {
    let (value, _derivative) = ispline_value_and_first_derivative(data, knots, degree, boundary)?;
    Ok(value)
}

/// Which I-spline columns take more than one value on `[lo, hi]`, under a
/// declared boundary convention.
///
/// `I_j` is continuous and non-decreasing with derivative `M_j ≥ 0`, so
/// `I_j(hi) − I_j(lo) = ∫_lo^hi M_j`, and that is positive iff `M_j > 0` on part
/// of `(lo, hi)`. Inside `[left, right]` the M-spline is a positive multiple of
/// a B-spline and is positive exactly on the open support `(t_{j+1}, t_{j+k+1})`,
/// `k = degree + 1`; under [`ISplineBoundary::LinearTails`] it is the constant
/// `M_j(left)` below the interval and `M_j(right)` above it. Each clause is a
/// comparison of knots with `lo`/`hi` or a sign of a Cox–de Boor value, so the
/// answer is structural: a column that is flat on the data is not kept because
/// rounding moved it, and a column that varies is not dropped because it moved
/// by less than an absolute cut-off (#3288).
///
/// For a finite set of evaluation points `[lo, hi]` is their range; a monotone
/// column that is constant across its extremes is constant across the set.
pub fn ispline_columns_varying_on(
    knots: ArrayView1<'_, f64>,
    degree: usize,
    lo: f64,
    hi: f64,
    boundary: ISplineBoundary,
) -> Result<Vec<bool>, BasisError> {
    if !(lo.is_finite() && hi.is_finite()) {
        return Err(BasisError::InvalidInput(format!(
            "I-spline column variation needs a finite range; got [{lo}, {hi}]"
        )));
    }
    let interval = ispline_modelling_interval(knots, degree)?;
    // `ispline_modelling_interval` has validated that the frame fits.
    let order = degree + 1;
    let columns = (knots.len() - order - 1).saturating_sub(1);
    let mut varies = vec![false; columns];
    let Some((left, right)) = interval else {
        return Ok(varies);
    };
    if !(lo < hi) {
        return Ok(varies);
    }
    let owned_knots = knots.to_owned();
    let boundary_slopes = match boundary {
        ISplineBoundary::Saturate => None,
        ISplineBoundary::LinearTails => Some(create_ispline_derivative_dense(
            Array1::from_vec(vec![left, right]).view(),
            &owned_knots,
            degree,
            1,
        )?),
    };
    for (column, flag) in varies.iter_mut().enumerate() {
        let support_lo = knots[column + 1].max(left);
        let support_hi = knots[column + order + 1].min(right);
        let interior = lo.max(support_lo) < hi.min(support_hi);
        let exterior = boundary_slopes.as_ref().is_some_and(|slopes| {
            (lo < left && slopes[[0, column]] > 0.0) || (hi > right && slopes[[1, column]] > 0.0)
        });
        *flag = interior || exterior;
    }
    Ok(varies)
}

/// Continue an already-evaluated `(value, derivative)` pair affinely past the
/// boundary knots, at the pair's own one-sided boundary derivative.
///
/// Rows strictly inside `[left, right]` — and rows AT either boundary, whose
/// saturating value already IS the anchor and whose derivative
/// `create_ispline_derivative_dense` deliberately keeps at the interior
/// one-sided slope — are left untouched, bit for bit. A non-finite evaluation
/// point compares false against both bounds and is therefore also left as the
/// raw evaluator produced it, rather than silently acquiring a tail.
fn extend_ispline_pair_affinely(
    data: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    value: &mut Array2<f64>,
    derivative: &mut Array2<f64>,
) -> Result<(), BasisError> {
    let Some((left, right)) = ispline_modelling_interval(knots, degree)? else {
        return Ok(());
    };
    if !data.iter().any(|&x| x < left || x > right) {
        return Ok(());
    }
    let boundary_points = Array1::from_vec(vec![left, right]);
    let (boundary_value_arc, _) = create_basis::<Dense>(
        boundary_points.view(),
        KnotSource::Provided(knots),
        degree,
        BasisOptions::i_spline(),
    )?;
    let boundary_value = boundary_value_arc.as_ref();
    let boundary_derivative =
        create_ispline_derivative_dense(boundary_points.view(), &knots.to_owned(), degree, 1)?;
    let columns = value.ncols();
    if boundary_value.ncols() != columns || boundary_derivative.ncols() != columns {
        return Err(BasisError::DimensionMismatch(format!(
            "I-spline boundary bases are {}/{} columns wide but the evaluated basis is {columns}",
            boundary_value.ncols(),
            boundary_derivative.ncols()
        )));
    }
    for (row, &x) in data.iter().enumerate() {
        let (end, anchor) = if x < left {
            (0usize, left)
        } else if x > right {
            (1usize, right)
        } else {
            continue;
        };
        let step = x - anchor;
        for column in 0..columns {
            let slope = boundary_derivative[[end, column]];
            value[[row, column]] = boundary_value[[end, column]] + step * slope;
            derivative[[row, column]] = slope;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests_ispline_boundary {
    use super::*;

    /// A clamped degree-3 I-spline knot vector on `[lo, hi]`: the internal
    /// B-spline frame is degree 4, so each end repeats `degree + 2 = 5` times.
    fn clamped_knots(lo: f64, hi: f64, internal: usize) -> Array1<f64> {
        let mut knots: Vec<f64> = vec![lo; 5];
        for k in 1..=internal {
            knots.push(lo + (hi - lo) * (k as f64) / ((internal + 1) as f64));
        }
        knots.extend(std::iter::repeat_n(hi, 5));
        Array1::from_vec(knots)
    }

    fn probe_points(lo: f64, hi: f64) -> Array1<f64> {
        Array1::from_vec(vec![
            lo - 4.0,
            lo - 0.7,
            lo,
            lo + 0.25 * (hi - lo),
            0.5 * (lo + hi),
            hi - 0.25 * (hi - lo),
            hi,
            hi + 0.7,
            hi + 4.0,
        ])
    }

    #[test]
    fn the_modelling_interval_is_the_frames_partition_of_unity_span() {
        let knots = clamped_knots(-1.0, 2.0, 3);
        let interval = ispline_modelling_interval(knots.view(), 3)
            .expect("interval")
            .expect("a usable interval");
        assert!(
            (interval.0 - (-1.0)).abs() < 1e-15 && (interval.1 - 2.0).abs() < 1e-15,
            "modelling interval {interval:?} is not the clamped support"
        );
    }

    #[test]
    fn both_conventions_agree_with_the_raw_evaluator_inside_the_knots() {
        let knots = clamped_knots(0.0, 1.0, 4);
        let inside = Array1::from_vec(vec![0.0, 0.1, 0.37, 0.5, 0.81, 1.0]);
        let (saturating_value, saturating_derivative) = ispline_value_and_first_derivative(
            inside.view(),
            knots.view(),
            3,
            ISplineBoundary::Saturate,
        )
        .expect("saturating pair");
        let (linear_value, linear_derivative) = ispline_value_and_first_derivative(
            inside.view(),
            knots.view(),
            3,
            ISplineBoundary::LinearTails,
        )
        .expect("linear-tail pair");
        assert_eq!(
            saturating_value, linear_value,
            "the two conventions must be BIT-identical inside the knots"
        );
        assert_eq!(
            saturating_derivative, linear_derivative,
            "the two conventions must be BIT-identical inside the knots"
        );
    }

    /// The property the whole module exists for, on both conventions and on
    /// both exteriors: the published derivative IS the derivative of the
    /// published value.
    #[test]
    fn the_derivative_is_a_finite_difference_of_the_value_2705() {
        let (lo, hi) = (0.0_f64, 1.0_f64);
        let knots = clamped_knots(lo, hi, 4);
        let points = probe_points(lo, hi);
        let step = 1e-6_f64;
        for boundary in [ISplineBoundary::Saturate, ISplineBoundary::LinearTails] {
            for &x in points.iter() {
                // Skip the boundary knots themselves: the derivative there is
                // one-sided by construction and a centred difference straddles
                // the two conventions.
                if (x - lo).abs() < 2.0 * step || (x - hi).abs() < 2.0 * step {
                    continue;
                }
                let center = Array1::from_vec(vec![x]);
                let plus = Array1::from_vec(vec![x + step]);
                let minus = Array1::from_vec(vec![x - step]);
                let (_, analytic) =
                    ispline_value_and_first_derivative(center.view(), knots.view(), 3, boundary)
                        .expect("analytic derivative");
                let (value_plus, _) =
                    ispline_value_and_first_derivative(plus.view(), knots.view(), 3, boundary)
                        .expect("forward value");
                let (value_minus, _) =
                    ispline_value_and_first_derivative(minus.view(), knots.view(), 3, boundary)
                        .expect("backward value");
                for column in 0..analytic.ncols() {
                    let difference =
                        (value_plus[[0, column]] - value_minus[[0, column]]) / (2.0 * step);
                    let gap = (difference - analytic[[0, column]]).abs();
                    assert!(
                        gap < 1e-6,
                        "{boundary:?}: column {column} at x={x}: analytic {} vs finite \
                         difference {difference} (gap {gap:.3e})",
                        analytic[[0, column]]
                    );
                }
            }
        }
    }

    /// #3288: the structural answer is the brute-force one — a column varies on
    /// `[lo, hi]` iff its published derivative is positive somewhere inside —
    /// on interior windows, whole-interval, and exterior windows, under both
    /// conventions, with no tolerance on either side.
    #[test]
    fn the_varying_columns_are_those_with_derivative_mass_in_the_range_3288() {
        let knots = clamped_knots(0.0, 1.0, 4);
        let ranges = [
            (0.05, 0.15),
            (0.45, 0.55),
            (0.85, 0.95),
            (0.21, 0.39),
            (0.0, 1.0),
            (-2.0, -1.0),
            (2.0, 3.0),
            (-2.0, 3.0),
        ];
        let mut saw_mixed = [false; 2];
        let mut saw_exterior_only = false;
        for (which, boundary) in [ISplineBoundary::Saturate, ISplineBoundary::LinearTails]
            .into_iter()
            .enumerate()
        {
            for &(lo, hi) in &ranges {
                let varies = ispline_columns_varying_on(knots.view(), 3, lo, hi, boundary)
                    .expect("structural column variation");
                let grid =
                    Array1::from_iter((1..2000).map(|i| lo + (hi - lo) * (i as f64) / 2000.0));
                let (_, derivative) =
                    ispline_value_and_first_derivative(grid.view(), knots.view(), 3, boundary)
                        .expect("derivative on the range");
                assert_eq!(varies.len(), derivative.ncols());
                for (column, &flag) in varies.iter().enumerate() {
                    let brute = derivative.column(column).iter().any(|&m| m > 0.0);
                    assert_eq!(
                        flag, brute,
                        "{boundary:?} on [{lo}, {hi}]: column {column} structural {flag} vs \
                         derivative mass {brute}"
                    );
                }
                saw_mixed[which] |= varies.iter().any(|&v| v) && varies.iter().any(|&v| !v);
                if boundary == ISplineBoundary::LinearTails && hi < 0.0 {
                    saw_exterior_only |= varies.iter().any(|&v| v);
                }
            }
        }
        assert!(
            saw_mixed[0] && saw_mixed[1],
            "each convention must see a window that keeps some columns and drops others"
        );
        assert!(
            saw_exterior_only,
            "a linear tail must make a column vary on a window below every knot"
        );
    }

    #[test]
    fn saturating_holds_the_boundary_value_and_kills_the_slope() {
        let (lo, hi) = (0.0_f64, 1.0_f64);
        let knots = clamped_knots(lo, hi, 4);
        let points = Array1::from_vec(vec![lo - 3.0, lo, hi, hi + 3.0]);
        let (value, derivative) = ispline_value_and_first_derivative(
            points.view(),
            knots.view(),
            3,
            ISplineBoundary::Saturate,
        )
        .expect("saturating pair");
        for column in 0..value.ncols() {
            assert_eq!(
                value[[0, column]],
                value[[1, column]],
                "column {column} must hold its left-boundary value"
            );
            assert_eq!(
                value[[3, column]],
                value[[2, column]],
                "column {column} must hold its right-boundary value"
            );
            assert_eq!(
                derivative[[0, column]],
                0.0,
                "column {column} must carry no exterior slope"
            );
            assert_eq!(
                derivative[[3, column]],
                0.0,
                "column {column} must carry no exterior slope"
            );
        }
    }

    #[test]
    fn linear_tails_continue_at_the_boundary_slope_and_stay_monotone() {
        let (lo, hi) = (0.0_f64, 1.0_f64);
        let knots = clamped_knots(lo, hi, 4);
        let far = 3.0_f64;
        let points = Array1::from_vec(vec![lo - far, lo, hi, hi + far]);
        let (value, derivative) = ispline_value_and_first_derivative(
            points.view(),
            knots.view(),
            3,
            ISplineBoundary::LinearTails,
        )
        .expect("linear-tail pair");
        let mut left_slope_total = 0.0_f64;
        let mut right_slope_total = 0.0_f64;
        for column in 0..value.ncols() {
            let left_slope = derivative[[1, column]];
            let right_slope = derivative[[2, column]];
            left_slope_total += left_slope;
            right_slope_total += right_slope;
            assert!(
                left_slope >= 0.0 && right_slope >= 0.0,
                "column {column} boundary slopes must be non-negative: {left_slope}, {right_slope}"
            );
            assert_eq!(
                derivative[[0, column]],
                left_slope,
                "column {column} must carry its LEFT boundary slope below the support"
            );
            assert_eq!(
                derivative[[3, column]],
                right_slope,
                "column {column} must carry its RIGHT boundary slope above the support"
            );
            let expected_below = value[[1, column]] - far * left_slope;
            let expected_above = value[[2, column]] + far * right_slope;
            assert!(
                (value[[0, column]] - expected_below).abs() < 1e-12,
                "column {column} below the support: {} vs {expected_below}",
                value[[0, column]]
            );
            assert!(
                (value[[3, column]] - expected_above).abs() < 1e-12,
                "column {column} above the support: {} vs {expected_above}",
                value[[3, column]]
            );
        }
        // Non-vacuity: a tail that carries no slope at all would satisfy every
        // assertion above and would BE the saturating convention.
        assert!(
            left_slope_total > 0.0 && right_slope_total > 0.0,
            "the fixture must have a nonzero boundary slope on both sides; got \
             {left_slope_total} and {right_slope_total}"
        );
    }
}
