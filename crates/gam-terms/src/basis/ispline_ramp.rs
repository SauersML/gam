//! The monotone I-spline **ramp** basis: one function on all of `ℝ`, for any
//! knot vector, with every derivative order read off that same function.
//!
//! This is the evaluator a monotone *warp* needs. `create_ispline_dense` and
//! [`create_ispline_derivative_dense`] answer the same question on the interval
//! where the degree-`bs` B-splines are a partition of unity, and impose `0` /
//! `1` outside it by convention. On a CLAMPED knot vector that convention is
//! exactly right and this module reproduces them bit for bit (pinned below).
//! On any other knot vector it is not: a ramp whose support runs past the
//! partition-of-unity interval gets truncated mid-rise, and the truncation is a
//! step.
//!
//! Why that distinction is worth a module (gam#2695): a clamped boundary knot
//! has multiplicity `bs + 1`, so the ramp is `C^{-1}` there — `I'_c` steps from
//! `0` outside to its interior one-sided value inside, and `I''_c` steps by
//! `2, 6, 12, 20` at public degree `2, 3, 4, 5`. A warp composed onto a
//! β-dependent index is differentiated three times by the inner objective (the
//! Firth value `Φ = ½Σ g(λ(Z_JᵀHZ_J))` reads `H`, which reads `w‴`), so a step
//! at any order `≤ 3` is a step in the OBJECTIVE. The cure is a knot vector
//! whose ends are SIMPLE — and a simple-ended vector is exactly the case the
//! partition-of-unity convention cannot evaluate. This evaluator is what makes
//! such a vector representable.

use super::*;

/// Dense matrix of the `order`-th derivative of every degree-`degree` B-spline
/// in `knot_vector`, evaluated at `data`.
///
/// `order = 0` is the plain basis. Orders above `degree` are identically zero
/// and are returned as such rather than refused, because the derivative tower a
/// warp consumer asks for is fixed by the objective, not by the degree it
/// happens to be built at.
pub(crate) fn bspline_derivative_dense_any_order(
    data: ArrayView1<'_, f64>,
    knot_vector: ArrayView1<'_, f64>,
    degree: usize,
    order: usize,
) -> Result<Array2<f64>, BasisError> {
    let knots_owned = knot_vector.to_owned();
    let num_cols = knot_vector.len().saturating_sub(degree + 1);
    if order > degree {
        return Ok(Array2::zeros((data.len(), num_cols)));
    }
    match order {
        0 => {
            let (basis, _) = create_basis::<Dense>(
                data,
                KnotSource::Provided(knots_owned.view()),
                degree,
                BasisOptions::value(),
            )?;
            Ok(basis.as_ref().clone())
        }
        1 => {
            let (basis, _) = create_basis::<Dense>(
                data,
                KnotSource::Provided(knots_owned.view()),
                degree,
                BasisOptions::first_derivative(),
            )?;
            Ok(basis.as_ref().clone())
        }
        2 => {
            let (basis, _) = create_basis::<Dense>(
                data,
                KnotSource::Provided(knots_owned.view()),
                degree,
                BasisOptions::second_derivative(),
            )?;
            Ok(basis.as_ref().clone())
        }
        order => {
            // Orders 3 and up run the shared de-Boor derivative recurrence
            // directly, so the tower has no ceiling below the degree: a warp
            // consumer asks for the order ITS objective differentiates, and
            // capping the evaluator at 4 would silently report zero for a
            // derivative the basis genuinely has.
            let mut out = Array2::<f64>::zeros((data.len(), num_cols));
            let mut workspace = BsplineDerivativeWorkspace::new();
            for (row_index, &x) in data.iter().enumerate() {
                let row = out.slice_mut(s![row_index, ..]).into_slice().ok_or_else(|| {
                    BasisError::InvalidInput("B-spline derivative row is not contiguous".into())
                })?;
                evaluate_bspline_derivative_recurrence_into(
                    order,
                    x,
                    knots_owned.view(),
                    degree,
                    row,
                    &mut workspace,
                    0,
                )?;
            }
            Ok(out)
        }
    }
}

/// The number of knots added at each end so that every ramp's support lies
/// strictly inside the padded partition-of-unity interval.
///
/// Ramp `c` is supported on `[t_{c+1}, t_{c+1+bs}]`, whose right end is at most
/// `t_{len-2}`; the partition of unity of the degree-`bs` B-splines on a padded
/// vector holds on `[τ_bs, τ_{len_pad-bs-1}]`. Padding by `bs + 1` on each side
/// puts every support inside that, with one span to spare, and is the smallest
/// padding that does. Nothing is chosen: it is read off the support.
#[inline]
fn ramp_padding(bs_degree: usize) -> usize {
    bs_degree + 1
}

/// Extend `knot_vector` by [`ramp_padding`] knots at each end, continuing the
/// first and last POSITIVE span so the padded vector stays strictly usable even
/// when the original one is clamped (where the boundary span is zero-width).
fn pad_knots(knot_vector: ArrayView1<'_, f64>, pad: usize) -> Result<Array1<f64>, BasisError> {
    let len = knot_vector.len();
    if len < 2 {
        return Err(BasisError::InvalidKnotVector(
            "I-spline ramp padding needs at least two knots".to_string(),
        ));
    }
    let first_gap = (1..len)
        .map(|i| knot_vector[i] - knot_vector[i - 1])
        .find(|gap| *gap > 0.0);
    let last_gap = (1..len)
        .rev()
        .map(|i| knot_vector[i] - knot_vector[i - 1])
        .find(|gap| *gap > 0.0);
    let (Some(low_gap), Some(high_gap)) = (first_gap, last_gap) else {
        return Err(BasisError::InvalidKnotVector(
            "I-spline ramp padding needs a knot vector with a positive span".to_string(),
        ));
    };
    let mut padded = Vec::with_capacity(len + 2 * pad);
    for step in (1..=pad).rev() {
        padded.push(knot_vector[0] - low_gap * step as f64);
    }
    padded.extend(knot_vector.iter().copied());
    for step in 1..=pad {
        padded.push(knot_vector[len - 1] + high_gap * step as f64);
    }
    Ok(Array1::from(padded))
}

/// The I-spline ramp basis and its `derivative_order`-th derivative.
///
/// Column `c` is
///
/// ```text
///     I_c(x) = ∫_{t_{c+1}}^{x} M_{c+1}(u) du
/// ```
///
/// the integral of the normalised M-spline of degree `ispline_degree` supported
/// on `[t_{c+1}, t_{c+1+bs}]` with `bs = ispline_degree + 1`. So every column is
/// exactly `0` before its support, exactly `1` after it, non-decreasing
/// throughout, and as smooth as the underlying spline: `C^{bs-1-m}` at a knot of
/// multiplicity `m`.
///
/// The column count matches `create_ispline_dense` exactly
/// (`knots.len() − bs − 2`), and on a clamped knot vector every entry matches it
/// too — see `the_ramp_reproduces_the_clamped_convention` below.
pub fn ispline_ramp_basis_dense(
    data: ArrayView1<'_, f64>,
    knot_vector: ArrayView1<'_, f64>,
    ispline_degree: usize,
    derivative_order: usize,
) -> Result<Array2<f64>, BasisError> {
    if ispline_degree < 1 {
        return Err(BasisError::InvalidDegree(ispline_degree));
    }
    let bs_degree = ispline_degree
        .checked_add(1)
        .ok_or_else(|| BasisError::InvalidInput("I-spline degree overflow".to_string()))?;
    validate_knots_for_degree(knot_vector, bs_degree)?;
    let num_bspline = knot_vector.len() - bs_degree - 1;
    let num_ramps = num_bspline.saturating_sub(1);
    if num_ramps == 0 {
        return Ok(Array2::zeros((data.len(), 0)));
    }

    let pad = ramp_padding(bs_degree);
    let padded = pad_knots(knot_vector, pad)?;
    let derivatives =
        bspline_derivative_dense_any_order(data, padded.view(), bs_degree, derivative_order)?;
    let padded_cols = padded.len() - bs_degree - 1;
    if derivatives.ncols() != padded_cols {
        return Err(BasisError::InvalidInput(format!(
            "padded B-spline evaluation produced {} columns, expected {padded_cols}",
            derivatives.ncols()
        )));
    }

    // The documented endpoint convention, kept (gam#2695).
    //
    // `create_ispline_derivative_dense` returns the INTERIOR one-sided slope at
    // the top of a clamped knot vector, and says why: "`right` is routinely the
    // largest observed value (knot vectors are built from the data range), and
    // the transformation-normal shape derivative `h'(y)` must stay positive
    // there." The padded evaluation below reads the RIGHT-hand limit at that
    // point instead, which on a clamped vector is `0` — the padding turns the
    // repeated boundary knot into an interior knot of multiplicity `bs + 1`,
    // where the ramp's derivative is genuinely two-valued.
    //
    // So rows sitting exactly on a REPEATED top knot are read from the original
    // vector, where `create_basis` evaluates that same one-sided limit. A
    // simple-ended (warp) vector has no repeated top knot and never takes this
    // path: its ramp has settled there, and `0` is the correct value, not a
    // truncation.
    let top = knot_vector[knot_vector.len() - 1];
    let top_is_repeated = knot_vector.len() >= 2 && knot_vector[knot_vector.len() - 2] == top;
    let left_limit_rows: Vec<usize> = if top_is_repeated && derivative_order >= 1 {
        (0..data.len()).filter(|&row| data[row] == top).collect()
    } else {
        Vec::new()
    };
    let left_limit_table = if left_limit_rows.is_empty() {
        None
    } else {
        let points = Array1::from_elem(1, top);
        Some(bspline_derivative_dense_any_order(
            points.view(),
            knot_vector,
            bs_degree,
            derivative_order,
        )?)
    };

    let mut out = Array2::<f64>::zeros((data.len(), num_ramps));
    // Right-cumulative sums, one pass per row: `sum_at[j] = Σ_{l ≥ j+pad} dB_l`
    // is the derivative tower of `Σ_{l ≥ j} B_l` on the ORIGINAL indexing, and
    // that sum is `∫ M_j` wherever the padded partition of unity holds — which,
    // by construction of `pad`, is everywhere inside every ramp's support.
    let mut sum_at = vec![0.0_f64; num_bspline];
    for row in 0..data.len() {
        let x = data[row];
        if !x.is_finite() {
            continue;
        }
        let mut running = 0.0_f64;
        for value in sum_at.iter_mut() {
            *value = 0.0;
        }
        for column in (0..padded_cols).rev() {
            let term = derivatives[[row, column]];
            if term.is_finite() {
                running += term;
            }
            if column >= pad {
                let original = column - pad;
                if original < num_bspline {
                    sum_at[original] = running;
                }
            }
        }
        for ramp in 0..num_ramps {
            let index = ramp + 1;
            let support_start = knot_vector[index];
            let support_end = knot_vector[index + bs_degree];
            out[[row, ramp]] = if x < support_start {
                0.0
            } else if x > support_end {
                if derivative_order == 0 { 1.0 } else { 0.0 }
            } else {
                sum_at[index]
            };
        }
    }
    if let Some(table) = left_limit_table.as_ref() {
        for &row in &left_limit_rows {
            let mut running = 0.0_f64;
            for column in (1..num_bspline).rev() {
                let term = table[[0, column]];
                if term.is_finite() {
                    running += term;
                }
                out[[row, column - 1]] = running;
            }
        }
    }
    Ok(out)
}

/// A knot vector for a monotone WARP: `num_internal_knots + 1` uniform spans
/// across `[low, high]`, continued by `degree` further spans at the same width
/// on each side, all knots SIMPLE.
///
/// The continuation is not padding for numerical comfort — it is what makes the
/// basis a warp. Ramp `c` is `C^{degree-1}` at both ends of its support only if
/// the knot there is simple; a clamped vector gives the two outer ramps a
/// multiplicity-`degree+1` end, which is a step in `I'` (gam#2695). Continuing
/// the same uniform grid outward is the smallest change that makes every end
/// simple while leaving the user's `[low, high]` and knot count alone, and the
/// column count is unchanged: `num_internal_knots + degree` either way.
pub fn monotone_warp_knots(
    low: f64,
    high: f64,
    degree: usize,
    num_internal_knots: usize,
) -> Result<Array1<f64>, String> {
    if !(low.is_finite() && high.is_finite() && high > low) {
        return Err(format!(
            "monotone warp knots need a finite non-degenerate range, got [{low}, {high}]"
        ));
    }
    if degree < 1 {
        return Err("monotone warp knots need degree >= 1".to_string());
    }
    let spans = num_internal_knots + 1;
    let width = (high - low) / spans as f64;
    if !(width > 0.0) {
        return Err(format!(
            "monotone warp knot spacing is not positive for range [{low}, {high}] and \
             {num_internal_knots} internal knots"
        ));
    }
    let total = spans + 2 * degree;
    let mut knots = Vec::with_capacity(total + 1);
    for step in 0..=total {
        knots.push(low + width * (step as f64 - degree as f64));
    }
    Ok(Array1::from(knots))
}

/// [`monotone_warp_knots`] over [`seed_knot_range`], the same seed domain
/// [`initializewiggle_knots_from_seed`] uses.
///
/// This is the warp-block entry point; the clamped
/// [`initializewiggle_knots_from_seed`] stays for consumers whose basis is
/// evaluated on FIXED data (a response transform, say), where a boundary knot's
/// multiplicity is invisible because the evaluation point never moves.
pub fn monotone_warp_knots_from_seed(
    seed: ArrayView1<'_, f64>,
    degree: usize,
    num_internal_knots: usize,
) -> Result<Array1<f64>, String> {
    let (low, high) = seed_knot_range(seed)
        .ok_or_else(|| "non-finite seed for monotone warp knot initialization".to_string())?;
    monotone_warp_knots(low, high, degree, num_internal_knots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    fn clamped(degree: usize, internal: &[f64], low: f64, high: f64) -> Array1<f64> {
        let mut knots = vec![low; degree + 1];
        knots.extend_from_slice(internal);
        knots.extend(std::iter::repeat_n(high, degree + 1));
        Array1::from(knots)
    }

    /// The ramp evaluator is a strict generalisation: on a clamped knot vector —
    /// the only kind the shipped I-spline convention can express — it agrees
    /// with `create_ispline_dense` and `create_ispline_derivative_dense`
    /// everywhere they are defined, INCLUDING outside the modelling interval.
    #[test]
    fn the_ramp_reproduces_the_clamped_convention() {
        for ispline_degree in 1..=4usize {
            let knots = clamped(ispline_degree + 1, &[0.0, 1.0], -1.0, 2.0);
            let x = Array1::linspace(-3.0, 4.0, 71);
            let ramp = ispline_ramp_basis_dense(x.view(), knots.view(), ispline_degree, 0)
                .expect("ramp value");
            let legacy = create_ispline_derivative_dense(x.view(), &knots, ispline_degree, 0)
                .expect("legacy value");
            assert_eq!(ramp.dim(), legacy.dim());
            for ((row, col), value) in ramp.indexed_iter() {
                assert!(
                    (value - legacy[[row, col]]).abs() <= 1e-12,
                    "value column {col} at x={} : ramp {value:.12e} vs clamped {:.12e}",
                    x[row],
                    legacy[[row, col]],
                );
            }
        }
    }

    /// The endpoint convention `create_ispline_derivative_dense` documents is
    /// kept: at the top of a CLAMPED vector — routinely the largest observed
    /// value — the derivative is the interior one-sided slope, not the
    /// right-hand limit `0`.
    #[test]
    fn a_clamped_top_knot_still_reports_its_interior_one_sided_slope() {
        for ispline_degree in 1..=4usize {
            let knots = clamped(ispline_degree + 1, &[0.0, 1.0], -1.0, 2.0);
            let at_top = array![2.0];
            let inside = array![2.0 - 1.0e-9];
            let ramp_at_top =
                ispline_ramp_basis_dense(at_top.view(), knots.view(), ispline_degree, 1)
                    .expect("ramp slope at the top");
            let ramp_inside =
                ispline_ramp_basis_dense(inside.view(), knots.view(), ispline_degree, 1)
                    .expect("ramp slope inside");
            let worst = (0..ramp_at_top.ncols())
                .map(|c| (ramp_at_top[[0, c]] - ramp_inside[[0, c]]).abs())
                .fold(0.0_f64, f64::max);
            let scale = (0..ramp_at_top.ncols())
                .map(|c| ramp_at_top[[0, c]].abs())
                .fold(0.0_f64, f64::max);
            assert!(
                scale > 0.1,
                "the clamped top must carry a real slope for this pin to bite; got {scale:.3e}"
            );
            assert!(
                worst <= 1.0e-6 * (1.0 + scale),
                "the clamped top slope must be the interior one-sided value: at-top vs \
                 inside differ by {worst:.3e}"
            );
        }
    }

    /// The same point on a SIMPLE-ended warp vector is genuinely zero — the
    /// ramp has settled there — so the convention above is a clamped-vector
    /// rule and not a blanket one.
    #[test]
    fn a_simple_top_knot_reports_zero_because_the_ramp_has_settled() {
        for degree in 2..=5usize {
            let knots = monotone_warp_knots(-1.0, 2.0, degree, 2).expect("warp knots");
            let at_top = array![knots[knots.len() - 1]];
            let slope = ispline_ramp_basis_dense(at_top.view(), knots.view(), degree - 1, 1)
                .expect("ramp slope at the top");
            for c in 0..slope.ncols() {
                assert_eq!(
                    slope[[0, c]],
                    0.0,
                    "column {c} at the simple top knot must be exactly zero"
                );
            }
        }
    }

    /// Every column is a ramp: `0` before its support, `1` after it,
    /// non-decreasing in between, on ANY knot vector.
    #[test]
    fn every_column_is_a_zero_to_one_ramp_on_a_simple_knot_vector() {
        for degree in 2..=5usize {
            let knots = monotone_warp_knots(-1.0, 2.0, degree, 2).expect("warp knots");
            let x = Array1::linspace(-8.0, 9.0, 341);
            let values = ispline_ramp_basis_dense(x.view(), knots.view(), degree - 1, 0)
                .expect("ramp value");
            for col in 0..values.ncols() {
                assert!(
                    values[[0, col]] == 0.0 && values[[values.nrows() - 1, col]] == 1.0,
                    "column {col} must run 0 -> 1, got {} -> {}",
                    values[[0, col]],
                    values[[values.nrows() - 1, col]],
                );
                for row in 1..values.nrows() {
                    assert!(
                        values[[row, col]] >= values[[row - 1, col]] - 1e-12,
                        "column {col} decreases between x={} and x={}",
                        x[row - 1],
                        x[row],
                    );
                }
            }
        }
    }

    /// The property the warp exists for (gam#2695): on a SIMPLE-ended knot
    /// vector every derivative order up to `degree - 1` is continuous
    /// everywhere, including where the shipped clamped vector steps.
    #[test]
    fn a_simple_ended_warp_basis_is_continuous_to_one_below_its_degree() {
        for degree in 2..=5usize {
            let knots = monotone_warp_knots(-1.0, 2.0, degree, 2).expect("warp knots");
            let probes: Vec<f64> = knots.iter().copied().collect();
            for order in 0..degree {
                for &knot in &probes {
                    let gap = |h: f64| -> f64 {
                        let x = array![knot - h, knot + h];
                        let table =
                            ispline_ramp_basis_dense(x.view(), knots.view(), degree - 1, order)
                                .expect("ramp derivative");
                        (0..table.ncols())
                            .map(|c| (table[[0, c]] - table[[1, c]]).abs())
                            .fold(0.0_f64, f64::max)
                    };
                    let coarse = gap(1.0e-3);
                    let fine = gap(1.0e-6);
                    assert!(
                        fine <= coarse / 100.0 + 1.0e-12,
                        "degree {degree} order {order} steps at knot {knot}: gap {coarse:.3e} \
                         at h=1e-3 and {fine:.3e} at h=1e-6, a ratio of {:.2e} against the \
                         1000x a continuous derivative must give",
                        coarse / fine.max(f64::MIN_POSITIVE),
                    );
                }
            }
        }
    }

    /// Non-vacuity for the pin above, and the measurement gam#2695 turns on: the
    /// CLAMPED vector does step, at order 2, at every degree.
    #[test]
    fn the_clamped_vector_steps_at_order_two_at_every_degree() {
        for degree in 2..=5usize {
            let knots = clamped(degree, &[0.0, 1.0], -1.0, 2.0);
            let gap = |h: f64| -> f64 {
                let x = array![2.0 - h, 2.0 + h];
                let table = ispline_ramp_basis_dense(x.view(), knots.view(), degree - 1, 2)
                    .expect("ramp derivative");
                (0..table.ncols())
                    .map(|c| (table[[0, c]] - table[[1, c]]).abs())
                    .fold(0.0_f64, f64::max)
            };
            let coarse = gap(1.0e-3);
            let fine = gap(1.0e-6);
            assert!(
                fine > coarse / 10.0,
                "degree {degree}: the clamped right edge must STEP at order 2 for the \
                 simple-ended pin to be measuring something; got {coarse:.3e} -> {fine:.3e}"
            );
        }
    }

    /// The warp knot vector keeps the column count the clamped one had, so
    /// nothing downstream changes shape.
    #[test]
    fn the_warp_knot_vector_keeps_the_clamped_column_count() {
        for degree in 2..=5usize {
            for internal in 0..=4usize {
                let internal_knots: Vec<f64> = (1..=internal)
                    .map(|i| -1.0 + 3.0 * (i as f64) / ((internal + 1) as f64))
                    .collect();
                let clamped_knots = clamped(degree, &internal_knots, -1.0, 2.0);
                let warp_knots = monotone_warp_knots(-1.0, 2.0, degree, internal).expect("knots");
                let cols = |knots: &Array1<f64>| knots.len() - degree - 2;
                assert_eq!(
                    cols(&clamped_knots),
                    cols(&warp_knots),
                    "degree {degree}, {internal} internal knots"
                );
                assert_eq!(cols(&warp_knots), internal + degree);
            }
        }
    }
}
