//! Cyclic (periodic) B-spline basis and finite-difference penalty.
//!
//! For a periodic covariate — angle, time-of-day, day-of-year — December must
//! meet January without a seam. We bend the coefficient axis into a ring so
//! the smooth has no endpoints to privilege: the last coefficient is the first
//! one's neighbour, full stop.
//!
//! This module owns the self-contained periodic-domain math for 1-D cyclic
//! smooths: the wrap-around finite-difference penalty `S = D'D`, the half-open
//! period folding used by every periodic evaluator, the period-aware great-arc
//! distance, the uniform cyclic knot vector, and the dense cyclic B-spline
//! design assembly. Callers in `basis` handle term specs, streaming chunking,
//! and downstream matrix wiring.

use super::{BasisError, BasisOptions, Dense, KnotSource, create_basis};
use crate::faer_ndarray::fast_ata;
use ndarray::{Array1, Array2, ArrayView1};

/// Creates a cyclic finite-difference penalty `S = D' D`.
///
/// Unlike the ordinary P-spline Reinsch penalty, every difference stencil wraps
/// around the coefficient ring. For `order = 2`, each row is
/// `β_i - 2β_{i+1} + β_{i+2}` modulo the number of coefficients, so the
/// constant vector is the only null direction and no endpoint is privileged.
pub fn create_cyclic_difference_penalty_matrix(
    num_basis_functions: usize,
    order: usize,
) -> Result<Array2<f64>, BasisError> {
    if order == 0 || order >= num_basis_functions {
        return Err(BasisError::InvalidPenaltyOrder {
            order,
            num_basis: num_basis_functions,
        });
    }

    let mut d = Array2::<f64>::eye(num_basis_functions);
    for _ in 0..order {
        let previous = d;
        d = Array2::<f64>::zeros((num_basis_functions, num_basis_functions));
        for i in 0..num_basis_functions {
            let next = (i + 1) % num_basis_functions;
            for j in 0..num_basis_functions {
                d[[i, j]] = previous[[next, j]] - previous[[i, j]];
            }
        }
    }
    Ok(fast_ata(&d))
}

#[inline]
pub(crate) fn wrap_to_period(x: f64, start: f64, period: f64) -> f64 {
    // Keep wrapped values numerically inside the half-open period
    // `[start, start + period)`. `rem_euclid` can return exactly `period`
    // after extreme-roundoff cancellation (e.g. `(start + period - eps - start)`
    // rounding up to `period`), which lifts the result to `start + period`
    // and pushes basis evaluators that assume `x < start + period` over the
    // right knot. Fold that boundary back to `start` so every caller — the
    // dense cyclic B-spline evaluator, the periodic Duchon kernel matrix,
    // and the cylinder/torus tensor margins — agrees with the
    // `wrap_periodic_phase` convention used by the derivative path.
    let offset = (x - start).rem_euclid(period);
    if offset >= period {
        start
    } else {
        start + offset
    }
}

#[inline]
pub(crate) fn cyclic_distance_1d(x: f64, c: f64, period: f64) -> f64 {
    let delta = (x - c).abs().rem_euclid(period);
    delta.min(period - delta)
}

pub(crate) fn cyclic_uniform_knot_vector(
    start: f64,
    end: f64,
    degree: usize,
    num_basis: usize,
) -> Array1<f64> {
    let period = end - start;
    let h = period / num_basis as f64;
    let total_knots = num_basis + 2 * degree + 1;
    Array1::from_iter((0..total_knots).map(|i| start + (i as f64 - degree as f64) * h))
}

pub(crate) fn create_cyclic_bspline_basis_dense(
    data: ArrayView1<'_, f64>,
    start: f64,
    end: f64,
    degree: usize,
    num_basis: usize,
) -> Result<(Array2<f64>, Array1<f64>), BasisError> {
    if end <= start {
        return Err(BasisError::InvalidRange(start, end));
    }
    if num_basis <= degree {
        crate::bail_invalid_basis!(
            "cyclic B-spline basis requires more basis functions ({num_basis}) than degree ({degree})"
        );
    }
    let period = end - start;
    let wrapped = data.mapv(|x| wrap_to_period(x, start, period));
    let knots = cyclic_uniform_knot_vector(start, end, degree, num_basis);
    let (extended, _) = create_basis::<Dense>(
        wrapped.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )?;
    let mut cyclic = Array2::<f64>::zeros((data.len(), num_basis));
    for i in 0..extended.nrows() {
        for j in 0..extended.ncols() {
            let target = j % num_basis;
            cyclic[[i, target]] += extended[[i, j]];
        }
    }
    Ok((cyclic, knots))
}
