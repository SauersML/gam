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

#[cfg(test)]
mod tests {
    use super::*;

    // ── quantile_from_sorted ───────────────────────────────────────────────

    #[test]
    fn quantile_from_sorted_empty_returns_nan() {
        assert!(quantile_from_sorted(&[], 0.5).is_nan());
    }

    #[test]
    fn quantile_from_sorted_single_element_returns_it_for_any_q() {
        assert_eq!(quantile_from_sorted(&[7.0], 0.0), 7.0);
        assert_eq!(quantile_from_sorted(&[7.0], 0.5), 7.0);
        assert_eq!(quantile_from_sorted(&[7.0], 1.0), 7.0);
    }

    #[test]
    fn quantile_from_sorted_q0_returns_min_q1_returns_max() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile_from_sorted(&v, 0.0), 1.0);
        assert_eq!(quantile_from_sorted(&v, 1.0), 5.0);
    }

    #[test]
    fn quantile_from_sorted_median_five_element() {
        // numpy.quantile([1,2,3,4,5], 0.5, method='linear') == 3.0
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile_from_sorted(&v, 0.5), 3.0);
    }

    #[test]
    fn quantile_from_sorted_interpolates_between_adjacent_elements() {
        // [1.0, 3.0]: q=0.5 → pos=0.5, lo=0, hi=1, frac=0.5 → 1.0*0.5 + 3.0*0.5 = 2.0
        let v = [1.0, 3.0];
        assert_eq!(quantile_from_sorted(&v, 0.5), 2.0);
        // q=0.25 → pos=0.25, lo=0, frac=0.25 → 1.0*0.75 + 3.0*0.25 = 1.5
        assert!((quantile_from_sorted(&v, 0.25) - 1.5).abs() < 1e-14);
        // q=0.75 → pos=0.75, lo=0, frac=0.75 → 1.0*0.25 + 3.0*0.75 = 2.5
        assert!((quantile_from_sorted(&v, 0.75) - 2.5).abs() < 1e-14);
    }

    #[test]
    fn quantile_from_sorted_q_clamped_below_zero() {
        // q < 0 should clamp to q=0 → return minimum
        let v = [2.0, 4.0, 6.0];
        assert_eq!(quantile_from_sorted(&v, -1.0), 2.0);
        assert_eq!(quantile_from_sorted(&v, -0.0001), 2.0);
    }

    #[test]
    fn quantile_from_sorted_q_clamped_above_one() {
        // q > 1 should clamp to q=1 → return maximum
        let v = [2.0, 4.0, 6.0];
        assert_eq!(quantile_from_sorted(&v, 2.0), 6.0);
        assert_eq!(quantile_from_sorted(&v, 1.0001), 6.0);
    }

    #[test]
    fn quantile_from_sorted_two_element_boundary_ranks() {
        // q=0 → first, q=1 → last
        let v = [10.0, 20.0];
        assert_eq!(quantile_from_sorted(&v, 0.0), 10.0);
        assert_eq!(quantile_from_sorted(&v, 1.0), 20.0);
    }

    #[test]
    fn quantile_from_sorted_matches_numpy_linear_reference() {
        // Reference values from numpy.quantile([0,1,2,3,4,5,6,7,8,9], q, method='linear')
        let v: Vec<f64> = (0..10).map(|i| i as f64).collect();
        // q=0.1 → pos=0.9, lo=0, hi=1, frac=0.9 → 0*0.1 + 1*0.9 = 0.9
        assert!((quantile_from_sorted(&v, 0.1) - 0.9).abs() < 1e-14);
        // q=0.9 → pos=8.1, lo=8, hi=9, frac=0.1 → 8*0.9 + 9*0.1 = 8.1
        assert!((quantile_from_sorted(&v, 0.9) - 8.1).abs() < 1e-14);
        // q=0.5 → pos=4.5, lo=4, hi=5, frac=0.5 → 4*0.5 + 5*0.5 = 4.5
        assert!((quantile_from_sorted(&v, 0.5) - 4.5).abs() < 1e-14);
    }

    // ── order_statistic / order_statistic_from_sorted ─────────────────────

    #[test]
    fn order_statistic_returns_nan_on_empty() {
        assert!(order_statistic(&[], 1).is_nan());
        assert!(order_statistic_from_sorted(&[], 1).is_nan());
    }

    #[test]
    fn order_statistic_returns_nan_on_zero_rank() {
        let v = [3.0, 1.0, 2.0];
        assert!(order_statistic(&v, 0).is_nan());
        let sorted = [1.0, 2.0, 3.0];
        assert!(order_statistic_from_sorted(&sorted, 0).is_nan());
    }

    #[test]
    fn order_statistic_returns_nan_when_rank_exceeds_len() {
        let v = [3.0, 1.0, 2.0];
        assert!(order_statistic(&v, 4).is_nan());
        let sorted = [1.0, 2.0, 3.0];
        assert!(order_statistic_from_sorted(&sorted, 4).is_nan());
    }

    #[test]
    fn order_statistic_hits_mid_rank() {
        // Unsorted input; the 1-based 3rd-smallest of {1,2,3,4,5} is 3.
        let v = [5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(order_statistic(&v, 3), 3.0);
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(order_statistic_from_sorted(&sorted, 3), 3.0);
        // Boundary ranks: first and last.
        assert_eq!(order_statistic(&v, 1), 1.0);
        assert_eq!(order_statistic(&v, 5), 5.0);
    }
}
