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
}

impl BsplineScratch {
    #[inline]
    pub fn new(degree: usize) -> Self {
        let len = degree + 1;
        Self {
            left: vec![0.0; len],
            right: vec![0.0; len],
            n: vec![0.0; len],
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
    let scale = (maxval - minval).abs().max(1.0);
    let tol = 1e-12 * scale;

    let total_knots = num_internal_knots + 2 * (degree + 1);
    let mut knots = Vec::with_capacity(total_knots);
    for _ in 0..=degree {
        knots.push(minval);
    }

    if num_internal_knots > 0 {
        let mut support = Vec::with_capacity(sorted.len());
        let mut last: Option<f64> = None;
        for &x in &sorted {
            if x <= minval + tol || x >= maxval - tol {
                continue;
            }
            if last.map(|prev| (x - prev).abs() <= tol).unwrap_or(false) {
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
            if q <= prev_q + tol || q >= maxval - tol {
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
            let temp = if den.abs() > 1e-12 { n[r] / den } else { 0.0 };
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
