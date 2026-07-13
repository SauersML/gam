//! Numerically stable Cox-de Boor B-spline point evaluation and clamped
//! knot-vector generation. Internal scratch + helpers shared across the
//! spline-evaluation submodules; kept private (`pub(super)`) to `basis`.

use super::*;

/// Thread-local scratch buffers for spline evaluation. These are reused across
/// points to reduce allocation and improve cache locality.
#[derive(Clone, Debug, Default)]
pub struct BsplineScratch {
    pub(crate) left: Vec<f64>,
    pub(crate) right: Vec<f64>,
    pub(crate) n: Vec<f64>,
    pub(crate) all_prev: Vec<f64>,
    pub(crate) all_curr: Vec<f64>,
}

impl BsplineScratch {
    #[inline]
    pub fn new(degree: usize) -> Self {
        let len = degree + 1;
        Self {
            left: vec![0.0; len],
            right: vec![0.0; len],
            n: vec![0.0; len],
            all_prev: Vec::new(),
            all_curr: Vec::new(),
        }
    }

    #[inline]
    pub(super) fn ensure_degree(&mut self, degree: usize) {
        let len = degree + 1;
        if self.left.len() != len {
            self.left.resize(len, 0.0);
            self.right.resize(len, 0.0);
            self.n.resize(len, 0.0);
        }
    }
}

/// Evaluates all B-spline basis functions using the full knot support instead
/// of clamping to the modeling interval `[t_degree, t_num_basis]`.
///
/// The ordinary value evaluator intentionally clamps for linear extension at
/// clamped boundaries. Derivative recurrences need the mathematical B-spline
/// values on the whole supplied knot vector, including the exterior support
/// spans used by cyclic fold-back constructions.
#[inline]
pub(crate) fn evaluate_splines_at_point_full_support_into(
    x: f64,
    degree: usize,
    knots: ArrayView1<f64>,
    basisvalues: &mut [f64],
    scratch: &mut BsplineScratch,
) {
    let num_knots = knots.len();
    let num_basis = num_knots - degree - 1;
    assert_eq!(basisvalues.len(), num_basis);
    basisvalues.fill(0.0);

    if !x.is_finite() || num_knots < 2 {
        return;
    }

    let zero_degree_len = num_knots - 1;
    scratch.all_prev.resize(zero_degree_len, 0.0);
    scratch.all_prev.fill(0.0);

    let last_knot = knots[num_knots - 1];
    for i in 0..zero_degree_len {
        if (knots[i] <= x && x < knots[i + 1]) || (x == last_knot && i + 1 == zero_degree_len) {
            scratch.all_prev[i] = 1.0;
        }
    }

    if degree == 0 {
        basisvalues.copy_from_slice(&scratch.all_prev[..num_basis]);
        return;
    }

    for d in 1..=degree {
        let level_len = num_knots - d - 1;
        scratch.all_curr.resize(level_len, 0.0);
        scratch.all_curr.fill(0.0);

        for i in 0..level_len {
            let denom_left = knots[i + d] - knots[i];
            let left = if !knot_span_is_degenerate(denom_left) {
                ((x - knots[i]) / denom_left) * scratch.all_prev[i]
            } else {
                0.0
            };

            let denom_right = knots[i + d + 1] - knots[i + 1];
            let right = if !knot_span_is_degenerate(denom_right) {
                ((knots[i + d + 1] - x) / denom_right) * scratch.all_prev[i + 1]
            } else {
                0.0
            };

            scratch.all_curr[i] = left + right;
        }

        std::mem::swap(&mut scratch.all_prev, &mut scratch.all_curr);
    }

    basisvalues.copy_from_slice(&scratch.all_prev[..num_basis]);
}

/// Generates the full knot vector with clamped boundary knots.
///
/// Standard B-spline construction: boundary values are repeated (degree + 1) times
/// to ensure the basis functions are well-supported across the entire data domain.
/// This prevents "ghost" basis functions with support mostly outside the data range,
/// which would create near-zero columns in the design matrix and ill-conditioned systems.
pub(super) fn generate_full_knot_vector(
    data_range: (f64, f64),
    num_internal_knots: usize,
    degree: usize,
) -> Result<Array1<f64>, BasisError> {
    let (minval, maxval) = data_range;

    // Double-check for degenerate range - this should be caught by the public function
    // but we add it here as a defensive measure
    if minval == maxval {
        return Err(BasisError::DegenerateRange(num_internal_knots));
    }

    let h = (maxval - minval) / (num_internal_knots as f64 + 1.0);
    let total_knots = num_internal_knots + 2 * (degree + 1);

    let mut knots = Vec::with_capacity(total_knots);

    // Clamped start: repeat minval (degree + 1) times
    for _ in 0..=degree {
        knots.push(minval);
    }

    // Internal knots: uniformly spaced
    for i in 1..=num_internal_knots {
        knots.push(minval + i as f64 * h);
    }

    // Clamped end: repeat maxval (degree + 1) times
    for _ in 0..=degree {
        knots.push(maxval);
    }

    Ok(Array::from_vec(knots))
}

/// Generates a clamped full knot vector with internal knots placed at empirical quantiles.
pub(super) fn generate_full_knot_vector_quantile(
    data: ArrayView1<'_, f64>,
    num_internal_knots: usize,
    degree: usize,
) -> Result<Array1<f64>, BasisError> {
    if data.is_empty() {
        crate::bail_invalid_basis!("cannot generate quantile knots from empty data");
    }
    if data.iter().any(|x| !x.is_finite()) {
        crate::bail_invalid_basis!("quantile knot placement requires finite data");
    }

    // Up-front minimum-n check (issue #340): auto-knot quantile placement
    // requires enough evaluation points to span both clamped boundaries and
    // every requested interior knot.  The interior support is everything
    // strictly between min(t) and max(t), which is `len(t) - 2` points in
    // the best case (one minimum, one maximum, the rest interior).  With
    // `num_internal_knots` knots we need at least `num_internal_knots`
    // distinct interior values; we also need `len(t) >= degree + 1` so the
    // clamped knot vector defines a non-degenerate basis.  When either
    // bound is violated, emit a user-correctable diagnostic rather than
    // the cryptic "non-interior knot" / "distinct interior support"
    // messages from deeper in the algorithm.
    let min_required_for_interior = num_internal_knots.saturating_add(2);
    let min_required_for_degree = degree.saturating_add(1);
    let min_required = min_required_for_interior.max(min_required_for_degree);
    if data.len() < min_required {
        crate::bail_invalid_basis!(
            "auto-knot placement requires at least {min} evaluation point(s) for \
             degree={deg} with {ki} interior knot(s) (got n={n}). Either provide \
             more points, reduce the requested interior-knot count, or supply an \
             explicit clamped knot vector.",
            min = min_required,
            deg = degree,
            ki = num_internal_knots,
            n = data.len(),
        );
    }

    let mut sorted: Vec<f64> = data.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    let minval = sorted[0];
    let maxval = *sorted.last().unwrap_or(&minval);
    if minval == maxval {
        return Err(BasisError::DegenerateRange(num_internal_knots));
    }
    let total_knots = num_internal_knots + 2 * (degree + 1);
    let mut knots = Vec::with_capacity(total_knots);
    for _ in 0..=degree {
        knots.push(minval);
    }

    if num_internal_knots > 0 {
        let mut support = Vec::with_capacity(sorted.len());
        let mut last: Option<f64> = None;
        for &x in &sorted {
            if x <= minval || x >= maxval {
                continue;
            }
            if last == Some(x) {
                continue;
            }
            support.push(x);
            last = Some(x);
        }
        if support.is_empty() {
            crate::bail_invalid_basis!(
                "quantile knot placement requires distinct interior support between {:.6e} and {:.6e}",
                minval,
                maxval
            );
        }
        let n = support.len();
        let mut prev_q = minval;
        for j in 1..=num_internal_knots {
            let p = j as f64 / (num_internal_knots + 1) as f64;
            let pos = p * (n.saturating_sub(1) as f64);
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            let frac = pos - lo as f64;
            let q = if lo == hi {
                support[lo]
            } else {
                support[lo] * (1.0 - frac) + support[hi] * frac
            };
            let q = q.clamp(minval, maxval);
            if q <= prev_q || q >= maxval {
                crate::bail_invalid_basis!(
                    "quantile knot placement produced a non-interior knot at index {}: {:.6e}",
                    j - 1,
                    q
                );
            }
            knots.push(q);
            prev_q = q;
        }
    }

    for _ in 0..=degree {
        knots.push(maxval);
    }

    Ok(Array::from_vec(knots))
}

/// Evaluates all B-spline basis functions at a single point `x`.
/// This uses a numerically stable implementation of the Cox-de Boor algorithm,
/// based on Algorithm A2.2 from "The NURBS Book" by Piegl and Tiller.
///
/// For x outside the spline domain [t_degree, tnum_basis], we apply constant
/// boundary extrapolation by clamping x to the nearest boundary before running
/// Cox-de Boor recursion.
#[inline]
pub(super) fn evaluate_splines_at_point_into(
    x: f64,
    degree: usize,
    knots: ArrayView1<f64>,
    basisvalues: &mut [f64],
    scratch: &mut BsplineScratch,
) {
    match degree {
        3 => evaluate_splines_at_point_fixed::<3>(x, knots, basisvalues, scratch),
        2 => evaluate_splines_at_point_fixed::<2>(x, knots, basisvalues, scratch),
        1 => evaluate_splines_at_point_fixed::<1>(x, knots, basisvalues, scratch),
        _ => evaluate_splines_at_point_dynamic(x, degree, knots, basisvalues, scratch),
    }
}

#[inline]
pub(crate) fn evaluate_spline_local_values(
    x: f64,
    degree: usize,
    knots: ArrayView1<f64>,
    scratch: &mut BsplineScratch,
) -> (usize, usize) {
    let num_knots = knots.len();
    let num_basis = num_knots - degree - 1;

    scratch.ensure_degree(degree);
    scratch.n.fill(0.0);
    scratch.left.fill(0.0);
    scratch.right.fill(0.0);

    let x_eval = x.clamp(knots[degree], knots[num_basis]);

    let mu = {
        if x_eval >= knots[num_basis] {
            num_basis - 1
        } else if x_eval < knots[degree] {
            degree
        } else {
            // Binary search to replace the linear scan
            //   while span < num_basis && x_eval >= knots[span + 1] { span += 1; }
            // The loop counts how many of knots[degree+1..=num_basis] are
            // `<= x_eval`, so `partition_point(|&k| k <= x_eval)` on that
            // slice gives the same offset from `degree`.
            let slice = knots
                .as_slice()
                .expect("B-spline knot vector is contiguous");
            degree + slice[degree + 1..=num_basis].partition_point(|&k| k <= x_eval)
        }
    };

    let left = &mut scratch.left;
    let right = &mut scratch.right;
    let n = &mut scratch.n;

    n[0] = 1.0;

    for d in 1..=degree {
        left[d] = x_eval - knots[mu + 1 - d];
        right[d] = knots[mu + d] - x_eval;

        let mut saved = 0.0;
        for r in 0..d {
            let den = right[r + 1] + left[d - r];
            let temp = if !knot_span_is_degenerate(den) {
                n[r] / den
            } else {
                0.0
            };
            n[r] = saved + right[r + 1] * temp;
            saved = left[d - r] * temp;
        }
        n[d] = saved;
    }

    (mu, num_basis)
}

#[inline]
pub(crate) fn evaluate_splines_at_point_fixed<const DEGREE: usize>(
    x: f64,
    knots: ArrayView1<f64>,
    basisvalues: &mut [f64],
    scratch: &mut BsplineScratch,
) {
    let (mu, num_basis) = evaluate_spline_local_values(x, DEGREE, knots, scratch);
    assert_eq!(basisvalues.len(), num_basis);
    let n = &scratch.n;
    basisvalues.fill(0.0);
    for i in 0..=DEGREE {
        let gi = mu as isize + i as isize - DEGREE as isize;
        if gi >= 0 {
            let global_idx = gi as usize;
            if global_idx < num_basis {
                basisvalues[global_idx] = n[i];
            }
        }
    }
}

#[inline]
pub(crate) fn evaluate_splines_at_point_dynamic(
    x: f64,
    degree: usize,
    knots: ArrayView1<f64>,
    basisvalues: &mut [f64],
    scratch: &mut BsplineScratch,
) {
    let (mu, num_basis) = evaluate_spline_local_values(x, degree, knots, scratch);
    assert_eq!(basisvalues.len(), num_basis);
    let n = &scratch.n;
    basisvalues.fill(0.0);
    for i in 0..=degree {
        let gi = mu as isize + i as isize - degree as isize;
        if gi >= 0 {
            let global_idx = gi as usize;
            if global_idx < num_basis {
                basisvalues[global_idx] = n[i];
            }
        }
    }
}

/// Evaluates only the non-zero B-spline basis values at a single point `x`.
/// Returns the start column for the contiguous support.
#[inline]
pub(super) fn evaluate_splines_sparse_into(
    x: f64,
    degree: usize,
    knots: ArrayView1<f64>,
    values: &mut [f64],
    scratch: &mut BsplineScratch,
) -> usize {
    let (mu, _) = evaluate_spline_local_values(x, degree, knots, scratch);
    assert_eq!(values.len(), degree + 1);
    let n = &scratch.n;
    for i in 0..=degree {
        values[i] = n[i];
    }

    mu.saturating_sub(degree)
}

/// Evaluate the sparse B-spline support at `x` and write its right-cumulative
/// sums into `offsets`, indexed by global basis column.
///
/// This is the I-spline left-boundary anchoring kernel: column `j` receives the
/// total active mass at and to the right of `j` within the support block. The
/// caller must pre-zero `offsets` (length = number of B-spline columns) and
/// supply `local` of length `degree + 1` as scratch. Columns outside
/// `offsets.len()` are skipped.
#[inline]
pub(super) fn cumulative_bspline_offsets_into(
    x: f64,
    degree: usize,
    knots: ArrayView1<f64>,
    local: &mut [f64],
    scratch: &mut BsplineScratch,
    offsets: &mut [f64],
) {
    let support = degree + 1;
    let start = evaluate_splines_sparse_into(x, degree, knots, local, scratch);
    let mut running = 0.0_f64;
    for offset in (0..support).rev() {
        let j = start + offset;
        if j >= offsets.len() {
            continue;
        }
        running += local[offset];
        offsets[j] = running;
    }
}

#[cfg(test)]
mod knot_scale_invariance_tests {
    use super::*;
    use crate::basis::{
        create_difference_penalty_matrix, evaluate_bspline_derivative_scalar,
        evaluate_bsplinesecond_derivative_scalar,
    };
    use ndarray::Array1;

    /// Clamped cubic knot vector with interior knots at `frac * scale` for
    /// `frac in {0.2, 0.4, 0.6, 0.8}` on `[0, scale]`.
    fn clamped_cubic_knots(scale: f64) -> Array1<f64> {
        let interior = [0.2, 0.4, 0.6, 0.8];
        let mut v = vec![0.0; 4];
        v.extend(interior.iter().map(|f| f * scale));
        v.extend(std::iter::repeat(scale).take(4));
        Array1::from(v)
    }

    fn partition_sum_at(x: f64, knots: ArrayView1<f64>, degree: usize) -> f64 {
        let num_basis = knots.len() - degree - 1;
        // Local (workhorse) path.
        let mut scratch = BsplineScratch::new(degree);
        let mut local = vec![0.0; degree + 1];
        let start = evaluate_splines_sparse_into(x, degree, knots, &mut local, &mut scratch);
        let local_sum: f64 = local.iter().sum();
        // Full-support path — must agree with the local path on partition of unity.
        let mut full = vec![0.0; num_basis];
        let mut full_scratch = BsplineScratch::new(degree);
        evaluate_splines_at_point_full_support_into(x, degree, knots, &mut full, &mut full_scratch);
        let full_sum: f64 = full.iter().sum();
        assert!(
            (local_sum - full_sum).abs() < 1e-9,
            "local ({local_sum}) and full-support ({full_sum}) partition sums disagree at x={x}, start={start}"
        );
        local_sum
    }

    /// Regression for #2292: the Cox–de Boor recurrence is scale-free (its terms
    /// are ratios `(x - t_i)/(t_{i+k} - t_i)`), so partition-of-unity must hold
    /// on a small-magnitude domain exactly as it does at unit scale. The old
    /// *absolute* `1e-12` knot-span floor zeroed legitimate distinct-but-small
    /// spans once the domain shrank below it, collapsing whole basis rows to
    /// zero. Exact repeated-knot detection leaves every distinct span live, so
    /// the basis values are invariant to a uniform rescaling of knots and x.
    #[test]
    fn bspline_partition_of_unity_is_scale_invariant() {
        let degree = 3;
        // Unit-scale sanity: partition of unity holds (this is the control).
        let ref_knots = clamped_cubic_knots(1.0);
        for &frac in &[0.05, 0.3, 0.55, 0.72, 0.95] {
            let s = partition_sum_at(frac, ref_knots.view(), degree);
            assert!(
                (s - 1.0).abs() < 1e-12,
                "unit-scale partition sum {s} != 1 at {frac}"
            );
        }

        // Tiny-scale domain (magnitude 1e-12): the smallest distinct knot span
        // here is 0.2e-12 = 2e-13, well below the old absolute 1e-12 floor, so
        // the pre-fix recurrence zeroed its de Boor terms and the row sums
        // collapsed. Structural equality leaves every distinct span live, so
        // the sums stay 1.
        let scale = 1e-12;
        let tiny_knots = clamped_cubic_knots(scale);
        for &frac in &[0.05, 0.3, 0.55, 0.72, 0.95] {
            let s = partition_sum_at(frac * scale, tiny_knots.view(), degree);
            assert!(
                (s - 1.0).abs() < 1e-9,
                "scale-invariance broken: partition sum {s} != 1 at x={} (scale {scale})",
                frac * scale
            );
        }
    }

    #[test]
    fn bspline_derivatives_transform_covariantly_under_coordinate_scaling() {
        let degree = 3;
        let scale = 1e-12;
        let knots = clamped_cubic_knots(1.0);
        let tiny_knots = clamped_cubic_knots(scale);
        let num_basis = knots.len() - degree - 1;
        let mut d1 = vec![0.0; num_basis];
        let mut d1_tiny = vec![0.0; num_basis];
        let mut d2 = vec![0.0; num_basis];
        let mut d2_tiny = vec![0.0; num_basis];
        evaluate_bspline_derivative_scalar(0.37, knots.view(), degree, &mut d1).unwrap();
        evaluate_bspline_derivative_scalar(0.37 * scale, tiny_knots.view(), degree, &mut d1_tiny)
            .unwrap();
        evaluate_bsplinesecond_derivative_scalar(0.37, knots.view(), degree, &mut d2).unwrap();
        evaluate_bsplinesecond_derivative_scalar(
            0.37 * scale,
            tiny_knots.view(),
            degree,
            &mut d2_tiny,
        )
        .unwrap();

        for i in 0..num_basis {
            assert!((d1_tiny[i] * scale - d1[i]).abs() < 1e-10);
            assert!((d2_tiny[i] * scale * scale - d2[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn divided_difference_penalty_is_invariant_on_tiny_coordinate_domains() {
        let unit = Array1::from(vec![0.0, 0.1, 0.35, 0.7, 1.0]);
        let scale = 1e-14;
        let tiny = unit.mapv(|x| x * scale);
        let reference = create_difference_penalty_matrix(5, 2, Some(unit.view())).unwrap();
        let observed = create_difference_penalty_matrix(5, 2, Some(tiny.view())).unwrap();
        for (&left, &right) in reference.iter().zip(observed.iter()) {
            assert!((left - right).abs() < 1e-10);
        }
    }

    /// The relative floor must NOT mask a genuine exactly-repeated-knot
    /// degeneracy: a basis function whose `degree + 1` span collapses to exactly
    /// zero (`t[i+degree+1] == t[i]`) still has zero support and must be
    /// rejected by `validate_knot_spans_nondegenerate`, at any scale.
    #[test]
    fn exact_repeated_knot_degeneracy_still_rejected() {
        let degree = 2;
        for &scale in &[1.0, 1e-12] {
            // t[3] == t[0] == 0 -> basis function 0 has zero support.
            let knots = Array1::from(vec![0.0, 0.0, 0.0, 0.0, scale, scale, scale]);
            let err = validate_knot_spans_nondegenerate(knots.view(), degree)
                .expect_err("zero-support basis must be rejected at scale {scale}");
            assert!(
                matches!(err, BasisError::InvalidKnotVector(_)),
                "expected InvalidKnotVector, got {err:?} at scale {scale}"
            );
        }
    }
}
