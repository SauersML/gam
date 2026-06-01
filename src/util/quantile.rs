/// Linear-interpolation quantile matching numpy.quantile default (method='linear').
pub fn quantile_from_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Exact 1-based order statistic from an already sorted slice.
///
/// Returns `NaN` when the slice is empty, `rank` is zero, or `rank` exceeds
/// the number of observations. This is intentionally not an interpolating
/// quantile: split-conformal calibration needs the observed `k`-th value to
/// preserve the finite-sample coverage proof.
pub fn order_statistic_from_sorted(sorted: &[f64], rank: usize) -> f64 {
    if sorted.is_empty() || rank == 0 || rank > sorted.len() {
        return f64::NAN;
    }
    sorted[rank - 1]
}

/// Exact 1-based order statistic from an unsorted slice.
///
/// This centralizes the sort+select path for code that needs an observed
/// sample value rather than `quantile_from_sorted`'s linear interpolation.
pub fn order_statistic(values: &[f64], rank: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    order_statistic_from_sorted(&sorted, rank)
}
