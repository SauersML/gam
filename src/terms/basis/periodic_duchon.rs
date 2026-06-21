use super::*;

/// N-D periodic-cyclic-B-spline first-derivative jet `∂Φ̃/∂t` per row.
///
/// One-dimensional periodic B-spline basis (one latent axis). `t` is the
/// `(n_rows, 1)` latent matrix; each row evaluates a length-`num_basis`
/// derivative stencil w.r.t. the scalar latent coordinate. The result is
/// `(n_rows, num_basis, 1)`. This is the derivative of the row-normalized
/// design returned by [`build_periodic_bspline_basis_1d`]. The raw
/// derivative formula `B'_i(x) = (B_{i,k−1}(x) − B_{i+1,k−1}(x)) / h` is
/// evaluated alongside the unnormalized basis row `Φ`; the returned row uses
/// the quotient rule for `Φ̃ = Φ / S`, where `S = Σ_j Φ_j`.
pub fn periodic_bspline_first_derivative_nd(
    t: ArrayView2<'_, f64>,
    data_range: (f64, f64),
    degree: usize,
    num_basis: usize,
) -> Result<Array3<f64>, BasisError> {
    if t.ncols() != 1 {
        crate::bail_invalid_basis!(
            "periodic_bspline_first_derivative_nd: t must have exactly 1 column; got {}",
            t.ncols()
        );
    }
    if degree == 0 {
        crate::bail_invalid_basis!("periodic_bspline_first_derivative_nd requires degree >= 1");
    }
    if num_basis < degree + 1 {
        crate::bail_invalid_basis!(
            "periodic_bspline_first_derivative_nd requires num_basis >= degree + 1 (got num_basis={num_basis}, degree={degree})"
        );
    }
    let (start, end) = data_range;
    if !(start.is_finite() && end.is_finite()) || end <= start {
        crate::bail_invalid_basis!(
            "periodic_bspline_first_derivative_nd: data_range must be finite and ordered, got {data_range:?}"
        );
    }
    let period = end - start;
    let n_rows = t.nrows();
    let t_col = t.column(0);

    let mut phi = vec![0.0_f64; num_basis];
    let mut dphi = vec![0.0_f64; num_basis];
    let mut out = Array3::<f64>::zeros((n_rows, num_basis, 1));
    for row in 0..n_rows {
        let xi = t_col[row];
        if !xi.is_finite() {
            crate::bail_invalid_basis!(
                "periodic_bspline_first_derivative_nd: non-finite latent at row {row}"
            );
        }
        let rowsum =
            fill_periodic_bspline_unnormalized_value_row(xi, start, period, degree, &mut phi);
        if !rowsum.is_finite() || rowsum <= 0.0 {
            crate::bail_invalid_basis!(
                "periodic_bspline_first_derivative_nd: non-positive rowsum at row {row}: {rowsum}"
            );
        }
        let rowsum_derivative =
            fill_periodic_bspline_unnormalized_derivative_row(xi, start, period, degree, &mut dphi);
        if !rowsum_derivative.is_finite() {
            crate::bail_invalid_basis!(
                "periodic_bspline_first_derivative_nd: non-finite rowsum derivative at row {row}: {rowsum_derivative}"
            );
        }
        let rowsum_squared = rowsum * rowsum;
        for i in 0..num_basis {
            out[[row, i, 0]] = dphi[i] / rowsum - phi[i] * rowsum_derivative / rowsum_squared;
        }
    }
    Ok(out)
}

/// Tensor-product 1-D-B-spline first-derivative jet `∂Φ/∂t` per row.
///
/// `t` is the `(n_rows, n_axes)` latent matrix and each axis carries its
/// own `(knots, degree)` univariate B-spline. The tensor-product basis is
///
/// ```text
///     Φ_{n, k}(t_n) = ∏_a B^{(a)}_{j_a(k)}(t_{n,a}),
/// ```
///
/// where `k` enumerates the row-major tensor product
/// `j_0 ∈ [0, K_0) × … × j_{n_axes−1} ∈ [0, K_{n_axes−1})`. The product
/// rule then gives, for the partial w.r.t. axis `axis`:
///
/// ```text
///     ∂Φ_{n,k} / ∂t_{n, axis}
///         = (B^{(axis)}_{j_axis})'(t_{n, axis})
///           · ∏_{a ≠ axis} B^{(a)}_{j_a}(t_{n,a}).
/// ```
///
/// Returned tensor shape: `(n_rows, K_total, n_axes)` where
/// `K_total = ∏_a K_a` and `K_a = knots[a].len() − degree[a] − 1`.
pub fn bspline_tensor_first_derivative(
    t: ArrayView2<'_, f64>,
    knots_per_axis: &[ArrayView1<'_, f64>],
    degrees: &[usize],
) -> Result<Array3<f64>, BasisError> {
    let n_axes = t.ncols();
    if knots_per_axis.len() != n_axes || degrees.len() != n_axes {
        crate::bail_invalid_basis!(
            "bspline_tensor_first_derivative: t has {n_axes} axes but received \
             {} knot vectors and {} degrees",
            knots_per_axis.len(),
            degrees.len(),
        );
    }
    if n_axes == 0 {
        crate::bail_invalid_basis!(
            "bspline_tensor_first_derivative: t must have at least one axis".into(),
        );
    }
    let n_rows = t.nrows();
    // Per-axis basis sizes and total tensor size.
    let mut k_per_axis = Vec::<usize>::with_capacity(n_axes);
    let mut total = 1usize;
    for a in 0..n_axes {
        let k = knots_per_axis[a]
            .len()
            .checked_sub(degrees[a] + 1)
            .ok_or_else(|| {
                BasisError::InvalidInput(format!(
                    "bspline_tensor_first_derivative: axis {a} knot vector too short \
                     for degree {}",
                    degrees[a]
                ))
            })?;
        k_per_axis.push(k);
        total = total.checked_mul(k).ok_or_else(|| {
            BasisError::InvalidInput(
                "bspline_tensor_first_derivative: tensor-product basis size overflow".into(),
            )
        })?;
    }
    let mut out = Array3::<f64>::zeros((n_rows, total, n_axes));
    // Scratch per row: per-axis value vector and derivative vector.
    let mut values_per_axis: Vec<Vec<f64>> = k_per_axis.iter().map(|&k| vec![0.0; k]).collect();
    let mut derivs_per_axis: Vec<Vec<f64>> = k_per_axis.iter().map(|&k| vec![0.0; k]).collect();
    // Hoist per-axis scratch allocations outside the row loop. Previously each
    // row reallocated a fresh `BsplineScratch` for the value path and (via
    // `evaluate_bspline_derivative_scalar`) a fresh lower-basis `Vec<f64>` and
    // lower-degree `BsplineScratch` for the derivative path on every axis,
    // turning the tensor evaluator into O(n_rows · n_axes) heap traffic.
    let mut value_scratch_per_axis: Vec<internal::BsplineScratch> = degrees
        .iter()
        .map(|&d| internal::BsplineScratch::new(d))
        .collect();
    let mut lower_basis_per_axis: Vec<Vec<f64>> = knots_per_axis
        .iter()
        .zip(degrees.iter())
        .map(|(knots, &d)| vec![0.0; knots.len().saturating_sub(d)])
        .collect();
    let mut lower_scratch_per_axis: Vec<internal::BsplineScratch> = degrees
        .iter()
        .map(|&d| internal::BsplineScratch::new(d.saturating_sub(1)))
        .collect();
    let mut idx = vec![0usize; n_axes];
    let mut prefix = vec![1.0; n_axes + 1];
    let mut suffix = vec![1.0; n_axes + 1];
    for n in 0..n_rows {
        // Evaluate B^{(a)} and (B^{(a)})' at t_{n, a} for each axis.
        for a in 0..n_axes {
            internal::evaluate_splines_at_point_into(
                t[[n, a]],
                degrees[a],
                knots_per_axis[a],
                &mut values_per_axis[a],
                &mut value_scratch_per_axis[a],
            );
            evaluate_bspline_derivative_scalar_into(
                t[[n, a]],
                knots_per_axis[a],
                degrees[a],
                &mut derivs_per_axis[a],
                &mut lower_basis_per_axis[a],
                &mut lower_scratch_per_axis[a],
            )?;
        }
        // Enumerate tensor product in row-major order matching
        // `j = j_0 * (K_1 K_2 … K_{n_axes-1}) + j_1 * (K_2 … K_{n_axes-1}) + … + j_{n_axes-1}`.
        for k in 0..total {
            // Reconstruct multi-index `idx` from flat `k`.
            let mut rem = k;
            for a in (0..n_axes).rev() {
                idx[a] = rem % k_per_axis[a];
                rem /= k_per_axis[a];
            }

            prefix[0] = 1.0;
            for a in 0..n_axes {
                prefix[a + 1] = prefix[a] * values_per_axis[a][idx[a]];
            }
            suffix[n_axes] = 1.0;
            for a in (0..n_axes).rev() {
                suffix[a] = suffix[a + 1] * values_per_axis[a][idx[a]];
            }

            // For each output axis, derivative of axis-`axis` factor times
            // values of the others.
            for axis in 0..n_axes {
                let leave_one_out = prefix[axis] * suffix[axis + 1];
                out[[n, k, axis]] = derivs_per_axis[axis][idx[axis]] * leave_one_out;
            }
        }
    }
    Ok(out)
}

#[inline]
pub(crate) fn periodic_distance_1d(x: f64, c: f64, period: f64) -> f64 {
    let dx = (x - c).rem_euclid(period).abs();
    dx.min(period - dx).abs()
}

/// 2m-th Bernoulli polynomial ``B_{2m}(t)``, evaluated on ``t ∈ [0, 1]``.
///
/// Closed forms for the orders the Duchon stack actually uses:
///   * ``B₂(t)  = t² − t + 1/6``
///   * ``B₄(t)  = t⁴ − 2t³ + t² − 1/30``
///   * ``B₆(t)  = t⁶ − 3t⁵ + (5/2)t⁴ − (1/2)t² + 1/42``
///   * ``B₈(t)  = t⁸ − 4t⁷ + (14/3)t⁶ − (7/3)t⁴ + (2/3)t² − 1/30``
///
/// Defined for ``t ∈ [0, 1]`` then extended periodically (the caller has
/// already reduced ``r/period`` modulo 1).
pub(crate) fn even_bernoulli_polynomial(degree: usize, t: f64) -> Result<f64, BasisError> {
    let t2 = t * t;
    match degree {
        2 => Ok(t2 - t + 1.0 / 6.0),
        4 => Ok(t2 * t2 - 2.0 * t2 * t + t2 - 1.0 / 30.0),
        6 => {
            let t4 = t2 * t2;
            let t6 = t4 * t2;
            Ok(t6 - 3.0 * t4 * t + 2.5 * t4 - 0.5 * t2 + 1.0 / 42.0)
        }
        8 => {
            let t4 = t2 * t2;
            let t6 = t4 * t2;
            let t8 = t4 * t4;
            Ok(
                t8 - 4.0 * t6 * t + (14.0 / 3.0) * t6 - (7.0 / 3.0) * t4 + (2.0 / 3.0) * t2
                    - 1.0 / 30.0,
            )
        }
        other => Err(BasisError::InvalidInput(format!(
            "periodic Duchon Bernoulli kernel only implemented for B_{{2m}} with m ∈ {{1, 2, 3, 4}}; got degree {other}"
        ))),
    }
}

/// Periodic Green's function of the iterated 1D Laplacian ``(d²/dx²)^m`` on
/// the circle of circumference ``period``, modulo the constant nullspace.
///
/// Returns ``(-1)^(m+1) · B_{2m}(r / period)`` where ``B_{2m}`` is the
/// ``2m``-th Bernoulli polynomial extended periodically. The Fourier series
/// is
///
/// ```text
///     2 · (-1)^(m+1) · (2π)^{2m} / (2m)! · Σ_{n≥1} cos(2π n t) / n^{2m}
/// ```
///
/// so every nonzero harmonic carries weight ``∝ 1/n^{2m}`` and the kernel
/// matrix is full rank (modulo the constant direction) on **any** lattice of
/// ``K`` distinct circle points — uniform or not, even or odd ``K``. The
/// sign ``(-1)^(m+1)`` makes every Fourier coefficient positive, so the
/// kernel matrix is positive semidefinite with rank ``K − 1`` (a single
/// zero eigenvalue along the constants).
///
/// **Contrast with the polyharmonic kernel evaluated at wrapped distance**:
/// for ``m = 2`` the polyharmonic path computes ``φ(r) = c · r``, which is
/// the triangle wave on the circle. The triangle wave's Fourier series
/// carries only **odd** harmonics; sampled on a uniform K-lattice with even
/// K, the discrete DFT lands exactly on the zero (even-harmonic) modes and
/// the kernel matrix loses ``K/2 − 1`` singular values. The Bernoulli
/// kernel is the actual Green's function the operator demands and does not
/// suffer that lattice-parity degeneracy.
pub(crate) fn periodic_duchon_kernel_bernoulli(
    r: f64,
    m: usize,
    period: f64,
) -> Result<f64, BasisError> {
    if !period.is_finite() || period <= 0.0 {
        crate::bail_invalid_basis!(
            "periodic Duchon kernel requires positive finite period; got {period}"
        );
    }
    if m == 0 {
        crate::bail_invalid_basis!("periodic Duchon order m must be at least 1");
    }
    let t = (r / period).rem_euclid(1.0);
    let sign = if m % 2 == 1 { 1.0 } else { -1.0 };
    Ok(sign * even_bernoulli_polynomial(2 * m, t)?)
}

/// First and second derivatives ``(B'_{2m}(s), B''_{2m}(s))`` of the even
/// Bernoulli polynomial w.r.t. its argument ``s``, for the orders the Duchon
/// stack uses (``m ∈ {1, 2, 3, 4}``).
///
/// Obtained by differentiating the closed forms in [`even_bernoulli_polynomial`]
/// (each is a plain polynomial in ``s``), so they are the EXACT derivatives of
/// the forward kernel value — the analytic backward of the periodic Bernoulli
/// Green's-function design (gam#580).
pub(crate) fn even_bernoulli_polynomial_derivatives(
    degree: usize,
    s: f64,
) -> Result<(f64, f64), BasisError> {
    let s2 = s * s;
    match degree {
        2 => Ok((2.0 * s - 1.0, 2.0)),
        4 => {
            let d1 = 4.0 * s2 * s - 6.0 * s2 + 2.0 * s;
            let d2 = 12.0 * s2 - 12.0 * s + 2.0;
            Ok((d1, d2))
        }
        6 => {
            let s3 = s2 * s;
            let s4 = s2 * s2;
            let s5 = s4 * s;
            let d1 = 6.0 * s5 - 15.0 * s4 + 10.0 * s3 - s;
            let d2 = 30.0 * s4 - 60.0 * s3 + 30.0 * s2 - 1.0;
            Ok((d1, d2))
        }
        8 => {
            let s3 = s2 * s;
            let s4 = s2 * s2;
            let s5 = s4 * s;
            let s6 = s4 * s2;
            let s7 = s6 * s;
            let d1 = 8.0 * s7 - 28.0 * s6 + 28.0 * s5 - (28.0 / 3.0) * s3 + (4.0 / 3.0) * s;
            let d2 = 56.0 * s6 - 168.0 * s5 + 140.0 * s4 - 28.0 * s2 + 4.0 / 3.0;
            Ok((d1, d2))
        }
        other => Err(BasisError::InvalidInput(format!(
            "periodic Duchon Bernoulli kernel derivative only implemented for B_{{2m}} with m ∈ {{1, 2, 3, 4}}; got degree {other}"
        ))),
    }
}

/// Radial jet ``(φ, dφ/dr, d²φ/dr²)`` of the periodic Bernoulli Green's-function
/// kernel ``φ(r) = (−1)^{m+1} · B_{2m}(r / period)``.
///
/// The forward design uses ``periodic_duchon_kernel_bernoulli``; this is its
/// EXACT radial derivative so the analytic backward (the position-API VJP) is
/// consistent with the Bernoulli forward, mirroring how the polyharmonic
/// triplet feeds the non-periodic derivative path. The caller already reduces
/// the signed offset to ``[−period/2, period/2]`` and passes ``r = |offset|``
/// with the sign applied separately, so ``s = r / period ∈ [0, 1/2]`` needs no
/// further modular reduction. Each ``d/dr`` brings a ``1/period`` factor by the
/// chain rule.
pub(crate) fn periodic_duchon_kernel_bernoulli_triplet(
    r: f64,
    m: usize,
    period: f64,
) -> Result<(f64, f64, f64), BasisError> {
    if !period.is_finite() || period <= 0.0 {
        crate::bail_invalid_basis!(
            "periodic Duchon kernel requires positive finite period; got {period}"
        );
    }
    if m == 0 {
        crate::bail_invalid_basis!("periodic Duchon order m must be at least 1");
    }
    let s = (r / period).rem_euclid(1.0);
    let sign = if m % 2 == 1 { 1.0 } else { -1.0 };
    let phi = sign * even_bernoulli_polynomial(2 * m, s)?;
    let (b1, b2) = even_bernoulli_polynomial_derivatives(2 * m, s)?;
    let dphi_dr = sign * b1 / period;
    let d2phi_dr2 = sign * b2 / (period * period);
    Ok((phi, dphi_dr, d2phi_dr2))
}

/// Scaled Bernoulli function ``kᵥ(t) = Bᵥ(t) / ν!`` and its first derivative
/// ``k'ᵥ(t) = B'ᵥ(t)/ν! = Bᵥ₋₁(t)/(ν−1)! = kᵥ₋₁(t)`` for the degrees the
/// mixed-periodicity Sobolev kernel needs (``ν ∈ {0,1,2,3,4}``).
///
/// These are the standard Sobolev-spline reproducing-kernel building blocks
/// (Wahba 1990; Gu, *Smoothing Spline ANOVA*). ``Bᵥ`` is the (ordinary, not
/// periodised) Bernoulli polynomial:
///   ``B₀=1``, ``B₁=t−½``, ``B₂=t²−t+1/6``, ``B₃=t³−(3/2)t²+(1/2)t``,
///   ``B₄=t⁴−2t³+t²−1/30``.
fn scaled_bernoulli_value_and_derivative(nu: usize, t: f64) -> Result<(f64, f64), BasisError> {
    // (value of Bᵥ(t), value of B'ᵥ(t)=ν·Bᵥ₋₁(t)); we then divide by ν!.
    let (bv, dbv) = match nu {
        0 => (1.0, 0.0),
        1 => (t - 0.5, 1.0),
        2 => (t * t - t + 1.0 / 6.0, 2.0 * t - 1.0),
        3 => {
            let t2 = t * t;
            (t2 * t - 1.5 * t2 + 0.5 * t, 3.0 * t2 - 3.0 * t + 0.5)
        }
        4 => {
            let t2 = t * t;
            (
                t2 * t2 - 2.0 * t2 * t + t2 - 1.0 / 30.0,
                4.0 * t2 * t - 6.0 * t2 + 2.0 * t,
            )
        }
        other => {
            crate::bail_invalid_basis!(
                "mixed-periodicity Sobolev kernel needs Bernoulli degree ν ≤ 4; got {other}"
            )
        }
    };
    let factorial = (1..=nu).map(|v| v as f64).product::<f64>();
    Ok((bv / factorial, dbv / factorial))
}

/// Penalised part of the 1-D Sobolev (smoothing-spline) reproducing kernel of
/// order ``m`` on ``[0, 1]`` for a NON-periodic axis:
///
/// ```text
///     R(x, y) = kₘ(x) kₘ(y) + (−1)^{m−1} k_{2m}(|x − y|)
/// ```
///
/// with ``kᵥ(t) = Bᵥ(t)/ν!`` (Wahba 1990; Gu 2013, the ``R₁`` reproducing
/// kernel of the seminorm ``∫ (f^{(m)})²``). This kernel is **positive
/// semidefinite** and its null space is exactly the polynomials of degree
/// ``< m`` — precisely the unpenalised directions the cylinder/torus Duchon
/// nullspace must contain on a non-periodic axis (gam#1423). Using it as the
/// per-axis factor (instead of the conditionally-PD chord-polyharmonic kernel)
/// is what restores positive semidefiniteness to the mixed-periodicity penalty
/// (gam#1422).
///
/// The caller passes the RAW axis coordinates ``x, y`` together with the
/// per-axis ``(lo, hi)`` from the centers; both are affine-mapped to ``[0, 1]``
/// so ``Bᵥ`` is evaluated on its canonical domain and the same map is replayed
/// identically at prediction time.
fn nonperiodic_sobolev_kernel_1d(
    x: f64,
    y: f64,
    m: usize,
    lo: f64,
    hi: f64,
) -> Result<f64, BasisError> {
    if m == 0 {
        crate::bail_invalid_basis!("non-periodic Sobolev kernel order m must be at least 1");
    }
    let span = (hi - lo).max(1e-300);
    let xs = ((x - lo) / span).clamp(0.0, 1.0);
    let ys = ((y - lo) / span).clamp(0.0, 1.0);
    let (kx, _) = scaled_bernoulli_value_and_derivative(m, xs)?;
    let (ky, _) = scaled_bernoulli_value_and_derivative(m, ys)?;
    let diff = (xs - ys).abs();
    let sign = if m % 2 == 1 { 1.0 } else { -1.0 };
    let k2m = even_bernoulli_polynomial(2 * m, diff)? / factorial_f64(2 * m);
    Ok(kx * ky + sign * k2m)
}

/// Radial-style jet ``(R, ∂R/∂x, ∂²R/∂x²)`` of
/// [`nonperiodic_sobolev_kernel_1d`] w.r.t. the first (data) coordinate ``x``,
/// for the prediction/position-API path. Derivatives carry the chain-rule
/// ``1/span`` factor from the affine map to ``[0, 1]``; the ``|x − y|`` term's
/// first derivative picks up ``sign(x − y)`` (its second derivative is the even
/// Bernoulli second derivative, continuous across ``x = y``).
pub(crate) fn nonperiodic_sobolev_kernel_1d_triplet(
    x: f64,
    y: f64,
    m: usize,
    lo: f64,
    hi: f64,
) -> Result<(f64, f64, f64), BasisError> {
    if m == 0 {
        crate::bail_invalid_basis!("non-periodic Sobolev kernel order m must be at least 1");
    }
    let span = (hi - lo).max(1e-300);
    let xs = ((x - lo) / span).clamp(0.0, 1.0);
    let ys = ((y - lo) / span).clamp(0.0, 1.0);
    let (kx, dkx) = scaled_bernoulli_value_and_derivative(m, xs)?;
    let (ky, _) = scaled_bernoulli_value_and_derivative(m, ys)?;
    // d/dx [kₘ(xs)] = k'ₘ(xs)/span; d²/dx² = k''ₘ(xs)/span². k''ₘ = kₘ₋₂ etc.;
    // reuse the even-Bernoulli second-derivative only via the |x−y| term and
    // build kₘ's second derivative from the (m−1) scaled-Bernoulli derivative.
    let (_, d2kx_inner) = if m >= 1 {
        scaled_bernoulli_value_and_derivative(m.saturating_sub(1), xs)?
    } else {
        (0.0, 0.0)
    };
    let sign = if m % 2 == 1 { 1.0 } else { -1.0 };
    let diff = xs - ys;
    let adiff = diff.abs();
    let (b1, b2) = even_bernoulli_polynomial_derivatives(2 * m, adiff)?;
    let fac2m = factorial_f64(2 * m);
    let k2m = even_bernoulli_polynomial(2 * m, adiff)? / fac2m;
    let dsign = if diff >= 0.0 { 1.0 } else { -1.0 };

    let value = kx * ky + sign * k2m;
    // ∂/∂x: kₘ'(xs)·ky/span + sign·(B'_{2m}(|d|)/((2m)!))·sgn(d)/span
    let d1 = (dkx * ky / span) + sign * (b1 / fac2m) * dsign / span;
    // ∂²/∂x²: kₘ''(xs)·ky/span² + sign·B''_{2m}(|d|)/((2m)!)/span²
    // kₘ''(xs) = (d/dxs) k'ₘ(xs) = (d/dxs) kₘ₋₁(xs) = k'ₘ₋₁(xs) = d2kx_inner.
    let d2 = (d2kx_inner * ky / (span * span)) + sign * (b2 / fac2m) / (span * span);
    Ok((value, d1, d2))
}

#[inline]
fn factorial_f64(n: usize) -> f64 {
    (1..=n).map(|v| v as f64).product::<f64>()
}

/// Per-axis ``[lo, hi]`` bounds of the centers along every NON-periodic axis,
/// used to affine-map that axis to ``[0, 1]`` for the Sobolev kernel. Periodic
/// axes carry a placeholder (their kernel uses the period, not these bounds).
/// Mirrored byte-for-byte between the forward builder and the prediction/jet
/// path so the realized design and the frozen-center replay agree.
pub(crate) fn mixed_periodicity_axis_bounds(
    centers: ArrayView2<'_, f64>,
    periodic_per_axis: &[bool],
) -> Vec<(f64, f64)> {
    let d = centers.ncols();
    (0..d)
        .map(|j| {
            if periodic_per_axis[j] {
                (0.0, 1.0)
            } else {
                let col = centers.column(j);
                let lo = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let hi = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                (lo, hi)
            }
        })
        .collect()
}

/// Additive mixed-periodicity Duchon kernel value
/// ``K(x, c) = Σ_j R_j(x_j, c_j)``, the sum of per-axis positive-semidefinite
/// reproducing kernels: the periodic Bernoulli Green's function on periodic
/// axes and the 1-D Sobolev kernel on non-periodic axes. As a SUM of PSD
/// kernels it is PSD, and its null space is the polynomials of degree ``< m``
/// in the non-periodic coordinates only (constants on the periodic axes),
/// which is exactly the correct cylinder/torus Duchon null space
/// (gam#1422 / gam#1423). This replaces the conditionally-PD
/// polyharmonic-of-chord-distance kernel.
pub(crate) fn mixed_periodicity_additive_kernel(
    x: ArrayView1<'_, f64>,
    c: ArrayView1<'_, f64>,
    m: usize,
    periodic_per_axis: &[bool],
    periods: &[f64],
    axis_bounds: &[(f64, f64)],
) -> Result<f64, BasisError> {
    let d = x.len();
    let mut acc = 0.0_f64;
    for j in 0..d {
        acc += if periodic_per_axis[j] {
            let r = periodic_distance_1d(x[j], c[j], periods[j]);
            periodic_duchon_kernel_bernoulli(r, m, periods[j])?
        } else {
            let (lo, hi) = axis_bounds[j];
            nonperiodic_sobolev_kernel_1d(x[j], c[j], m, lo, hi)?
        };
    }
    Ok(acc)
}

/// Per-axis value + first/second self-derivative of the additive
/// mixed-periodicity kernel for one ``(x, c)`` pair. Because the kernel is the
/// SUM of per-axis 1-D kernels, the gradient w.r.t. ``x`` is per-axis
/// (``∂K/∂x_a = R_a'(x_a, c_a)``, no cross terms) and the Hessian is DIAGONAL
/// (``∂²K/∂x_a∂x_c = δ_{ac} R_a''``). Returns ``(value, grad_a, hess_aa)`` with
/// length-``d`` per-axis vectors, so the caller can assemble the input-location
/// jet/Hessian directly. This is the exact analytic derivative of
/// [`mixed_periodicity_additive_kernel`].
#[allow(clippy::type_complexity)]
pub(crate) fn mixed_periodicity_additive_kernel_jet(
    x: ArrayView1<'_, f64>,
    c: ArrayView1<'_, f64>,
    m: usize,
    periodic_per_axis: &[bool],
    periods: &[f64],
    axis_bounds: &[(f64, f64)],
) -> Result<(f64, Vec<f64>, Vec<f64>), BasisError> {
    let d = x.len();
    let mut value = 0.0_f64;
    let mut grad = vec![0.0_f64; d];
    let mut hess = vec![0.0_f64; d];
    for a in 0..d {
        let (v, d1, d2) = if periodic_per_axis[a] {
            // The Bernoulli triplet differentiates w.r.t. the unsigned radial
            // distance `r = |x_a − c_a|` reduced mod period; convert to the
            // derivative w.r.t. `x_a` via the chain rule (sign of the reduced
            // signed offset). The second derivative is sign-independent.
            let p = periods[a];
            let signed = {
                let raw = (x[a] - c[a]).rem_euclid(p);
                if raw > 0.5 * p { raw - p } else { raw }
            };
            let r = signed.abs();
            let (phi, dphi_dr, d2phi_dr2) = periodic_duchon_kernel_bernoulli_triplet(r, m, p)?;
            let dsign = if signed >= 0.0 { 1.0 } else { -1.0 };
            (phi, dphi_dr * dsign, d2phi_dr2)
        } else {
            let (lo, hi) = axis_bounds[a];
            nonperiodic_sobolev_kernel_1d_triplet(x[a], c[a], m, lo, hi)?
        };
        value += v;
        grad[a] = d1;
        hess[a] = d2;
    }
    Ok((value, grad, hess))
}

/// Polynomial side-condition block for the mixed-periodicity Duchon null space:
/// monomials of total degree ``< m`` in the NON-periodic coordinates only
/// (periodic axes contribute only the constant, which is the degree-0 monomial
/// shared by all axes). For the cylinder ``(θ periodic, y free)`` with ``m = 2``
/// this yields ``{1, y}`` — so ``f(θ, y) = a + b y`` is correctly unpenalised
/// (gam#1423).
pub(crate) fn mixed_periodicity_nullspace_poly_block(
    points: ArrayView2<'_, f64>,
    m: usize,
    periodic_per_axis: &[bool],
) -> Array2<f64> {
    let n = points.nrows();
    let nonperiodic_axes: Vec<usize> = (0..points.ncols())
        .filter(|&j| !periodic_per_axis[j])
        .collect();
    let max_degree = m.saturating_sub(1);
    // Monomial exponents of total degree ≤ (m−1) over the non-periodic axes.
    let exps = monomial_exponents(nonperiodic_axes.len(), max_degree);
    let mut block = Array2::<f64>::zeros((n, exps.len()));
    for (col, exp) in exps.iter().enumerate() {
        for row in 0..n {
            let mut value = 1.0_f64;
            for (local_axis, &power) in exp.iter().enumerate() {
                let axis = nonperiodic_axes[local_axis];
                value *= points[[row, axis]].powi(power as i32);
            }
            block[[row, col]] = value;
        }
    }
    block
}

/// Drop centers that periodically identify with the leftmost anchor.
///
/// When the user describes a closed periodic lattice by including BOTH
/// endpoints of ``[left, left+period]``, the right endpoint is the same
/// circle point as ``left`` and produces an identical kernel column. We
/// remove every such duplicate (tested under the periodic metric with a
/// tolerance scaled to ``period``); the remaining centers correspond to
/// geometrically distinct points on the circle.
pub(crate) fn collapse_periodic_endpoint(
    centers: Array2<f64>,
    left: f64,
    period: f64,
) -> Array2<f64> {
    if period <= 0.0 || !period.is_finite() {
        return centers;
    }
    // Tolerance: relative to ``period``, well below any reasonable lattice
    // spacing (mgcv's smallest practical periodic ``k`` is around 3, giving a
    // spacing of ``period/3``).
    let tol = period.max(1.0) * 1.0e-10;
    let col = centers.column(0);
    let n_rows = col.len();
    // Keep the first center that maps to the circle point of ``left`` and
    // drop every subsequent center in the same equivalence class. A
    // naive "always keep index 0, drop other left-equivalents" rule loses
    // the geometric point entirely when the user passes centers in
    // unsorted order — e.g. ``[5, 0, period]`` would collapse to ``[5]``
    // because both ``0`` and ``period`` are left-equivalents at indices
    // ``> 0``.
    let mut seen_left = false;
    let keep: Vec<usize> = (0..n_rows)
        .filter(|&i| {
            if periodic_distance_1d(col[i], left, period) <= tol {
                if seen_left {
                    return false;
                }
                seen_left = true;
            }
            true
        })
        .collect();
    if keep.len() == n_rows {
        return centers;
    }
    let mut trimmed = Array2::<f64>::zeros((keep.len(), centers.ncols()));
    for (out_row, &src_row) in keep.iter().enumerate() {
        for c in 0..centers.ncols() {
            trimmed[[out_row, c]] = centers[[src_row, c]];
        }
    }
    trimmed
}

pub(crate) fn build_periodic_duchon_basis_1d(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    centers: Array2<f64>,
    workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    if data.ncols() != 1 {
        crate::bail_invalid_basis!(
            "periodic Duchon smooths currently require exactly one covariate"
        );
    }
    // ``left + period`` is the same circle point as ``left``. If the user
    // supplied centers spanning ``[left, left+period]`` (the natural way to
    // describe a closed periodic lattice and what the position-API validator
    // requires) the rightmost point is a duplicate of the leftmost under
    // periodic identification. Two identical kernel columns make the design
    // ``rank K−1`` instead of ``K``; ``X'X`` becomes singular (cond ~10¹⁷)
    // and the REML whitening transform amplifies machine noise into a ~10⁻⁶
    // negative eigenvalue, tripping the solver's PSD check.
    //
    // ``prepare_periodic_duchon_centers_1d_with_period`` validates the center
    // matrix, computes ``(left, period)`` and drops the periodically duplicate
    // center, in one place that every periodic Duchon code path shares. When
    // ``spec.periodic`` carries an explicit per-axis period (the position-API
    // half-open lattice path — gam#580), honor it as the domain wrap; otherwise
    // derive it from the center span (the closed lattice the formula DSL emits).
    let explicit_period = spec
        .periodic
        .as_ref()
        .and_then(|axes| axes.first().copied().flatten());
    let (centers, left, period) =
        prepare_periodic_duchon_centers_1d_with_period(centers, explicit_period)?;
    // The user encodes the Duchon order ``m`` in ``spec.nullspace_order``
    // (``Zero → m=1``, ``Linear → m=2``, ``Degree(d) → m=d+1``). Periodicity
    // forces the *constraint* nullspace to ``{constants}`` (the only
    // polynomial that is itself periodic), but the *kernel* must still
    // encode full ``m``-th-order smoothness. The right kernel for that is
    // the periodic Green's function of ``(d²/dx²)^m`` — the Bernoulli
    // polynomial ``B_{2m}(r/period)`` — not the polyharmonic kernel
    // ``r^{2p+2s-d}`` evaluated at wrapped distance (which collapses to the
    // triangle wave ``r^1`` after the periodic constraint forces ``p=1`` and
    // produces zero singular values on even-K uniform lattices).
    let user_m = duchon_p_from_nullspace_order(spec.nullspace_order);
    let effective_nullspace_order = DuchonNullspaceOrder::Zero;
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    let s_order = spec.power_as_usize();
    // Validate against the INTEGER `s` the hybrid kernel actually evaluates
    // (`power_as_usize` truncates a fractional `spec.power`), so the
    // well-posedness gate matches the realized kernel rather than the raw power.
    validate_duchon_kernel_orders(spec.length_scale, p_order, s_order as f64, 1)?;
    let z = kernel_constraint_nullspace(
        centers.view(),
        effective_nullspace_order,
        &mut workspace.cache,
    )?;
    let kernel_cols = z.ncols();
    let mut basis = Array2::<f64>::zeros((data.nrows(), kernel_cols + 1));
    let coeffs = spec
        .length_scale
        .map(|ls| duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / ls.max(1e-300)));
    let pure_poly_coeff = if spec.length_scale.is_none() {
        Some(PolyharmonicBlockCoeff::new(
            (pure_duchon_block_order(p_order, s_order as f64)) as f64,
            1,
        ))
    } else {
        None
    };
    let kernel_amp = duchon_kernel_amplification(
        centers.view(),
        spec.length_scale,
        p_order,
        s_order,
        1,
        None,
        coeffs.as_ref(),
        pure_poly_coeff.as_ref(),
    );
    // Step 1: build the N×K raw kernel matrix in parallel (each row is
    // independent; no shared writes). Step 2: design[:, :kernel_cols] =
    // K @ z via fast_ab (BLAS), which beats a hand-rolled per-row matvec
    // loop both at small K (compiler vectorizes the inner loop) and at
    // large K (one big matmul vs. many small ones).
    let centers_col0: Vec<f64> = centers.column(0).to_vec();
    let n_data = data.nrows();
    let k_centers = centers_col0.len();
    let len_scale = spec.length_scale;
    let mut raw_kernel = Array2::<f64>::zeros((n_data, k_centers));
    let err_flag = std::sync::atomic::AtomicBool::new(false);
    // Hoist the kernel-form choice out of the inner row × center loop. The
    // pure-Duchon vs. hybrid-Matern branch is the same for every row, so a
    // single-time dispatch saves N·K conditional branches at large scale.
    let amp = kernel_amp;
    if pure_poly_coeff.is_some() {
        // Pure polyharmonic case (no Matern length-scale). Use the periodic
        // Green's function — Bernoulli ``B_{2m}(r/period)`` — directly. This
        // is the actual Green's function of ``(d²/dx²)^m`` on the circle
        // modulo constants. Every Fourier mode contributes with weight
        // ``∝ 1/n^{2m}``, so the kernel matrix is full rank (modulo the
        // constant direction) on any K-point lattice — uniform or not, even
        // or odd K. The triangle-wave kernel ``r`` that the polyharmonic
        // dispatch would emit here only has odd Fourier modes and collapses
        // on even-K uniform lattices.
        raw_kernel
            .axis_chunks_iter_mut(ndarray::Axis(0), 1024)
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_idx, mut block)| {
                let row_offset = chunk_idx * 1024;
                for (local_i, mut out_row) in block.outer_iter_mut().enumerate() {
                    let i = row_offset + local_i;
                    let x = wrap_to_period(data[[i, 0]], left, period);
                    for j in 0..k_centers {
                        let r = periodic_distance_1d(x, centers_col0[j], period);
                        match periodic_duchon_kernel_bernoulli(r, user_m, period) {
                            Ok(v) => out_row[j] = v,
                            Err(_) => {
                                err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                }
            });
    } else {
        raw_kernel
            .axis_chunks_iter_mut(ndarray::Axis(0), 1024)
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_idx, mut block)| {
                let row_offset = chunk_idx * 1024;
                for (local_i, mut out_row) in block.outer_iter_mut().enumerate() {
                    let i = row_offset + local_i;
                    let x = wrap_to_period(data[[i, 0]], left, period);
                    for j in 0..k_centers {
                        let r = periodic_distance_1d(x, centers_col0[j], period);
                        match duchon_matern_kernel_general_from_distance(
                            r,
                            len_scale,
                            p_order,
                            s_order,
                            1,
                            coeffs.as_ref(),
                        ) {
                            Ok(v) => out_row[j] = v * amp,
                            Err(_) => {
                                err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                }
            });
    }
    if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
        crate::bail_invalid_basis!("periodic Duchon kernel evaluation produced a non-finite value");
    }
    // design[:, :kernel_cols] = raw_kernel @ z; design[:, kernel_cols] = 1
    let design_kernel = fast_ab(&raw_kernel, &z);
    basis
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&design_kernel);
    basis.column_mut(kernel_cols).fill(1.0);
    let mut center_kernel = Array2::<f64>::zeros((centers.nrows(), centers.nrows()));
    fill_symmetric_from_row_kernel(&mut center_kernel, |i, j| {
        let r = periodic_distance_1d(centers[[i, 0]], centers[[j, 0]], period);
        if pure_poly_coeff.is_some() {
            // Same Bernoulli Green's function the design uses — keeps the
            // penalty ``ω = z' K_centers z`` exactly the Gram matrix of the
            // smoother in its native basis, with no scale mismatch.
            periodic_duchon_kernel_bernoulli(r, user_m, period)
        } else {
            Ok(duchon_matern_kernel_general_from_distance(
                r,
                spec.length_scale,
                p_order,
                s_order,
                1,
                coeffs.as_ref(),
            )? * kernel_amp)
        }
    })?;
    let omega = fast_ab(&fast_atb(&z, &center_kernel), &z);
    let mut penalty = Array2::<f64>::zeros((basis.ncols(), basis.ncols()));
    penalty
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&omega);
    let base_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(basis));
    let identifiability_transform = spatial_identifiability_transform_from_design_matrix(
        data,
        &base_design,
        &spec.identifiability,
        "periodic Duchon",
    )?;
    let (design, primary) = if let Some(transform) = identifiability_transform.as_ref() {
        let design = wrap_dense_design_with_transform(base_design, transform, "periodic Duchon")?;
        let transformed = fast_ab(&fast_atb(transform, &penalty), transform);
        (design, transformed)
    } else {
        (base_design, penalty)
    };
    let candidates = vec![normalize_penalty_candidate(
        primary,
        1,
        PenaltySource::Primary,
    )];
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
        metadata: BasisMetadata::Duchon {
            centers,
            length_scale: spec.length_scale,
            periodic: Some(vec![Some(period)]),
            power: spec.power,
            nullspace_order: effective_nullspace_order,
            identifiability_transform,
            input_scales: None,
            aniso_log_scales: None,
            operator_collocation_points: None,
            radial_reparam: None,
        },
        kronecker_factored: None,
    })
}

/// Build a multi-dimensional Duchon basis with per-axis periodicity
/// (cylinder ``(True, False)``, torus ``(True, True)``, …).
///
/// ## Construction (gam#1422 / gam#1423)
///
/// The penalty is built from an **additive tensor (ANOVA) reproducing
/// kernel** — the sum of per-axis positive-semidefinite 1-D reproducing
/// kernels — NOT the polyharmonic kernel evaluated at the cylinder/torus
/// chord distance. The chord-polyharmonic kernel is only *conditionally*
/// positive-definite (PD orthogonal to the chord-embedding linear span), so
/// projecting out only the constants leaves indefinite linear modes and the
/// center Gram ``Ω = Zᵀ K Z`` carries large negative eigenvalues (gam#1422).
///
///   * **periodic** axis (period ``P_j``): the periodic Bernoulli Green's
///     function ``(−1)^{m+1} B_{2m}(Δ/P_j)`` ([`periodic_duchon_kernel_bernoulli`]),
///     which is PSD (every Fourier coefficient ``∝ 1/n^{2m} > 0``) with null
///     space ``{constants}``.
///   * **non-periodic** axis: the 1-D Sobolev smoothing-spline reproducing
///     kernel ([`nonperiodic_sobolev_kernel_1d`]), which is PSD with null
///     space the polynomials of degree ``< m``.
///
/// The total center kernel ``K_CC = Σ_j R_j`` is PSD (sum of PSD), and its
/// null space is the polynomials of total degree ``< m`` in the **non-periodic
/// coordinates only** (periodic coordinates contribute only constants) — the
/// correct cylinder/torus Duchon null space (gam#1423). We build
/// ``Z = null(Pᵀ)`` from that polynomial block ([`mixed_periodicity_nullspace_poly_block`]),
/// append the matching unpenalised polynomial columns to the design, and form
/// the single Primary penalty ``Ω = Zᵀ K_CC Z``, which is PSD by congruence.
/// The per-axis kernels are evaluated on each axis's own coordinate, so the
/// design wraps cleanly at every periodic seam.
pub(crate) fn build_duchon_basis_mixed_periodicity(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    centers: Array2<f64>,
    periodic_per_axis: &[bool],
    periods: &[f64],
    _workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    let d = data.ncols();
    if d == 0 {
        crate::bail_invalid_basis!("Duchon basis requires at least one covariate dimension");
    }
    if periodic_per_axis.len() != d {
        crate::bail_invalid_basis!(
            "periodic_per_axis must have length d={d}, got {}",
            periodic_per_axis.len()
        );
    }
    if periods.len() != d {
        crate::bail_invalid_basis!("periods must have length d={d}, got {}", periods.len());
    }
    for (j, (&per, &period)) in periodic_per_axis.iter().zip(periods.iter()).enumerate() {
        if per && !(period.is_finite() && period > 0.0) {
            crate::bail_invalid_basis!(
                "axis {j} is periodic but period={period} is not finite & positive"
            );
        }
    }
    if centers.ncols() != d {
        crate::bail_invalid_basis!(
            "centers ncols={} does not match data ncols={d}",
            centers.ncols()
        );
    }

    // Hybrid Matérn (length_scale = Some) is not supported on the
    // cylinder/torus path yet; the generalized chord distance plus the
    // partial-fraction Matérn chain has not been validated for periodic
    // axes. Surface a clear error instead of silently producing nonsense.
    if spec.length_scale.is_some() {
        crate::bail_invalid_basis!(
            "mixed-periodicity Duchon basis currently only supports the pure polyharmonic spectrum (length_scale=None)"
        );
    }
    // s_order > 0 (the Sobolev tail) is similarly unvalidated for periodic
    // axes — gate to s = 0 (pure polyharmonic).
    if spec.power != 0.0 {
        crate::bail_invalid_basis!(
            "mixed-periodicity Duchon basis currently requires power = 0 (pure polyharmonic); got power={}",
            spec.power
        );
    }

    let user_m = duchon_p_from_nullspace_order(spec.nullspace_order);
    let s_order_int = 0usize;
    validate_duchon_kernel_orders(None, user_m, s_order_int as f64, d)?;

    // gam#1422 / gam#1423 — PSD mixed-periodicity Duchon via an ADDITIVE
    // tensor (ANOVA) reproducing kernel, NOT the conditionally-PD
    // polyharmonic-of-chord-distance kernel. The center Gram is the sum of
    // per-axis positive-semidefinite reproducing kernels: the periodic
    // Bernoulli Green's function on periodic axes (PSD, null = constants) and
    // the 1-D Sobolev kernel on non-periodic axes (PSD, null = polynomials of
    // degree < m). The sum is PSD (sum of PSD), and its null space is the
    // polynomials of degree < m in the NON-periodic coordinates only — exactly
    // the cylinder/torus Duchon null space. We build `Z` from that null space
    // (`{1, y, …}` on the cylinder), append the matching polynomial columns to
    // the design (so `a + b·y` is representable AND unpenalised), and form
    // `Ω = Zᵀ K_CC Z`, which is PSD by congruence.
    let axis_bounds = mixed_periodicity_axis_bounds(centers.view(), periodic_per_axis);

    let centers_owned = centers.clone();
    let k_centers = centers_owned.nrows();
    let n_data = data.nrows();

    // Non-periodic-only polynomial side condition → translation-aware null
    // space `Z = null(Pᵀ)`. For the cylinder (m=2) `P = [1, y]`.
    let poly_block_centers =
        mixed_periodicity_nullspace_poly_block(centers_owned.view(), user_m, periodic_per_axis);
    let z = kernel_constraint_nullspace_from_matrix(poly_block_centers.view())?;
    let kernel_cols = z.ncols();
    let n_poly = poly_block_centers.ncols();

    // Row-parallel additive kernel: K[i, j] = Σ_a R_a(x_i[a], c_j[a]).
    let mut raw_kernel = Array2::<f64>::zeros((n_data, k_centers));
    let kernel_err: std::sync::Mutex<Option<BasisError>> = std::sync::Mutex::new(None);
    raw_kernel
        .axis_chunks_iter_mut(ndarray::Axis(0), 1024)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut block)| {
            let row_offset = chunk_idx * 1024;
            for (local_i, mut out_row) in block.outer_iter_mut().enumerate() {
                let i = row_offset + local_i;
                let x_row = data.row(i);
                for j in 0..k_centers {
                    match mixed_periodicity_additive_kernel(
                        x_row,
                        centers_owned.row(j),
                        user_m,
                        periodic_per_axis,
                        periods,
                        &axis_bounds,
                    ) {
                        Ok(v) => out_row[j] = v,
                        Err(e) => {
                            *kernel_err.lock().unwrap() = Some(e);
                            return;
                        }
                    }
                }
            }
        });
    if let Some(e) = kernel_err.into_inner().unwrap() {
        return Err(e);
    }

    // Design = [K @ Z, P(data)] — the kernel columns plus the explicit
    // unpenalised polynomial columns (in the non-periodic coordinates only).
    let design_kernel = fast_ab(&raw_kernel, &z);
    let poly_block_data = mixed_periodicity_nullspace_poly_block(data, user_m, periodic_per_axis);
    let mut basis = Array2::<f64>::zeros((n_data, kernel_cols + n_poly));
    basis
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&design_kernel);
    basis
        .slice_mut(s![.., kernel_cols..kernel_cols + n_poly])
        .assign(&poly_block_data);

    // Penalty: Ω = Zᵀ K_CC Z (kernel-Gram identity in the projected basis),
    // padded with zero rows/cols for the unpenalised polynomial columns. PSD
    // because K_CC is PSD and Z is real (congruence preserves PSD).
    let mut center_kernel = Array2::<f64>::zeros((k_centers, k_centers));
    fill_symmetric_from_row_kernel(&mut center_kernel, |i, j| {
        mixed_periodicity_additive_kernel(
            centers_owned.row(i),
            centers_owned.row(j),
            user_m,
            periodic_per_axis,
            periods,
            &axis_bounds,
        )
    })?;
    let omega = fast_ab(&fast_atb(&z, &center_kernel), &z);
    let mut penalty = Array2::<f64>::zeros((basis.ncols(), basis.ncols()));
    penalty
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&omega);

    let base_design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(basis));
    let identifiability_transform = spatial_identifiability_transform_from_design_matrix(
        data,
        &base_design,
        &spec.identifiability,
        "mixed-periodicity Duchon",
    )?;
    let (design, primary) = if let Some(transform) = identifiability_transform.as_ref() {
        let design =
            wrap_dense_design_with_transform(base_design, transform, "mixed-periodicity Duchon")?;
        let transformed = fast_ab(&fast_atb(transform, &penalty), transform);
        (design, transformed)
    } else {
        (base_design, penalty)
    };
    let candidates = vec![normalize_penalty_candidate(
        primary,
        1,
        PenaltySource::Primary,
    )];
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
        metadata: BasisMetadata::Duchon {
            centers: centers_owned,
            length_scale: None,
            // `periods[j]` is always present; the metadata convention is
            // `Some(period)` only for axes the caller marked periodic.
            periodic: Some(
                periodic_per_axis
                    .iter()
                    .zip(periods.iter())
                    .map(|(&is_periodic, &period)| if is_periodic { Some(period) } else { None })
                    .collect(),
            ),
            power: spec.power,
            // Record the user's requested order so the prediction/jet replay
            // rebuilds the SAME non-periodic-only polynomial null space and the
            // SAME order-`m` additive kernel (gam#1423).
            nullspace_order: spec.nullspace_order,
            identifiability_transform,
            input_scales: None,
            aniso_log_scales: None,
            operator_collocation_points: None,
            radial_reparam: None,
        },
        kronecker_factored: None,
    })
}

/// Public driver for the mixed-periodicity Duchon basis: derives per-axis
/// ``(left_j, period_j)`` from the supplied centers (mirroring how the 1D
/// periodic path infers the period from min/max), then dispatches into
/// [`build_duchon_basis_mixed_periodicity`].
///
/// `periods` may be `None` (auto-derive from centers along every periodic
/// axis) or `Some(vec![...])` (length == data.ncols(); entries for
/// non-periodic axes are ignored).
pub fn build_duchon_basis_mixed_periodicity_auto(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    periodic_per_axis: &[bool],
    periods: Option<&[f64]>,
) -> Result<BasisBuildResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    assert_spatial_centers_below_large_scale_cap(data.ncols(), centers.view())?;
    let d = data.ncols();
    if periodic_per_axis.len() != d {
        crate::bail_invalid_basis!(
            "periodic_per_axis must have length d={d}, got {}",
            periodic_per_axis.len()
        );
    }
    let resolved_periods: Vec<f64> = match periods {
        Some(p) => {
            if p.len() != d {
                crate::bail_invalid_basis!("periods must have length d={d}, got {}", p.len());
            }
            p.to_vec()
        }
        None => {
            // Auto-derive: along each periodic axis use (max - min) over centers.
            // Non-periodic axes get a placeholder 1.0 (unused).
            let mut out = vec![1.0_f64; d];
            for j in 0..d {
                if periodic_per_axis[j] {
                    let col = centers.column(j);
                    let left = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let right = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    if !left.is_finite() || !right.is_finite() || left >= right {
                        return Err(BasisError::InvalidRange(left, right));
                    }
                    out[j] = right - left;
                }
            }
            out
        }
    };
    // The 1D periodic circle is NOT a mixed-periodicity cylinder/torus: the
    // chord-embedding polyharmonic kernel ``φ(r) = c·r^{2m−d}`` is only
    // CONDITIONALLY positive-definite on ℝ and is genuinely indefinite under
    // the chord metric on the circle (its periodised Gram carries large
    // negative eigenvalues), so it cannot serve as a PSD penalty (gam#580).
    // The actual Green's function of ``(d²/dx²)^m`` on the circle is the
    // Bernoulli kernel built by ``build_periodic_duchon_basis_1d`` — full rank
    // modulo constants, PSD by construction. Route the 1D periodic case there
    // for EVERY caller (basis design and function-norm penalty alike) so the
    // two stay consistent; reserve the chord builder for true ``d ≥ 2``
    // cylinder/torus products where it is the right object.
    if d == 1 && periodic_per_axis[0] {
        let mut periodic_spec = spec.clone();
        periodic_spec.periodic = Some(vec![Some(resolved_periods[0])]);
        return build_periodic_duchon_basis_1d(data, &periodic_spec, centers, &mut workspace);
    }
    build_duchon_basis_mixed_periodicity(
        data,
        spec,
        centers,
        periodic_per_axis,
        &resolved_periods,
        &mut workspace,
    )
}

/// The magic *request-layer* default `(nullspace_order, power)` for a
/// non-periodic Euclidean Duchon basis of dimension `d`: the cubic polyharmonic
/// kernel in every dimension.
///
/// Returns an affine (`Linear`, `d+1` polynomial columns) null space and the
/// fractional spectral power `s = (d − 1)/2`. With `m = p + s = 2 + (d−1)/2` the
/// pure kernel exponent `2m − d = 3`, i.e. `φ(r) = r³` for every `d` — no order
/// escalation, no even/odd-`d` log special case. The smoothing structure is the
/// analytic native reproducing-norm Gram (`PenaltySource::Primary`) plus a
/// null-space ridge; only the global mean is left free.
///
/// This is applied by the FRONT-ENDS (formula / CLI / pyffi) when the user gives
/// no explicit `power`. The basis builder itself treats `spec.power` literally,
/// so an explicit `power = 0` is honored as `s = 0` — the integer-order Duchon
/// kernel `r²·log r` (≡ the thin-plate kernel) in even `d` — rather than being
/// upgraded to the cubic default.
pub fn duchon_cubic_default(dim: usize) -> (DuchonNullspaceOrder, f64) {
    (DuchonNullspaceOrder::Linear, (dim as f64 - 1.0) / 2.0)
}

/// Build the **analytic** Duchon penalty for a non-periodic Euclidean Duchon
/// basis: the native reproducing-norm Gram `ω = α²·Zᵀ K_CC Z` (the kernel
/// evaluated at center pairs, projected through the polynomial-constraint null
/// space `Z`) plus an analytic null-space shrinkage ridge. This is the exact
/// `(m+s)`-order Duchon seminorm — pure closed form, no quadrature — the same
/// object mgcv `bs="ds"` uses, mirroring the Matérn `double_penalty` path. The
/// design scales its kernel columns by the underflow amplification `α`, so the
/// coefficient-space penalty scales by `α²`. The null-space ridge penalizes the
/// affine trend's slope (mean-free: the constant is absorbed by the model
/// intercept) so the trend is not left fully unpenalized.
/// The constrained native bending penalty `Ω_c = α² · Zᵀ K_CC Z` (m×m, in the
/// kernel-coefficient frame, pre-identifiability). This is exactly the `omega`
/// block that `duchon_native_penalty_candidates` builds; it is exposed so the
/// data-metric radial reparameterization (#1355) can solve the generalized
/// eigenproblem `Ω_c v = μ G_c v` against the realized design Gram `G_c`.
pub(crate) fn duchon_constrained_bending_penalty(
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    kernel_transform: &Array2<f64>,
) -> Result<Array2<f64>, BasisError> {
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "duchon_constrained_bending_penalty: centers must have at least one column"
        );
    }
    let k = centers.nrows();
    let z = kernel_transform;
    let p_order = duchon_p_from_nullspace_order(nullspace_order);
    let s_int = duchon_power_to_usize(power);
    let pure = length_scale.is_none();
    let pure_poly_coeff = if pure {
        Some(PolyharmonicBlockCoeff::new(
            pure_duchon_block_order(p_order, power),
            dim,
        ))
    } else {
        None
    };
    let coeffs =
        length_scale.map(|ls| duchon_partial_fraction_coeffs(p_order, s_int, 1.0 / ls.max(1e-300)));
    let kernel_amp = duchon_kernel_amplification(
        centers,
        length_scale,
        p_order,
        s_int,
        dim,
        aniso_log_scales,
        coeffs.as_ref(),
        pure_poly_coeff.as_ref(),
    );
    let axis_scales = aniso_log_scales.map(aniso_axis_scales);

    // K_CC: kernel value at every center pair (anisotropic distance when set).
    let mut center_kernel = Array2::<f64>::zeros((k, k));
    fill_symmetric_from_row_kernel(&mut center_kernel, |i, j| {
        let r = if let Some(scales) = axis_scales.as_deref() {
            aniso_distance_rows_with_scales(centers, i, centers, j, scales)
        } else {
            euclidean_distance_rows(centers, i, centers, j)
        };
        if let Some(ppc) = pure_poly_coeff.as_ref() {
            Ok(ppc.eval(r))
        } else {
            duchon_matern_kernel_general_from_distance(
                r,
                length_scale,
                p_order,
                s_int,
                dim,
                coeffs.as_ref(),
            )
        }
    })?;

    let amp2 = kernel_amp * kernel_amp;
    let zt_k = fast_atb(z, &center_kernel);
    let omega = fast_ab(&zt_k, z).mapv(|v| v * amp2);

    // gam#1424 — the hybrid (Duchon–Matérn) kernel's exact spectral density
    // `ρ^{-2p}(κ²+ρ²)^{-s}` is nonnegative, so the constrained bending Gram
    // `Ω_c = α²·Zᵀ K_CC Z` is positive semidefinite in exact arithmetic. The
    // historical failure was numerical: `duchon_matern_kernel_general_from_distance`
    // assembled the kernel from an ALTERNATING partial-fraction expansion
    // (polyharmonic `r^{2m−d}` minus Matérn `r^ν K_ν(κr)` blocks) whose
    // individually-enormous terms cancel to the float noise floor at high
    // dimension / spectral power (d=16, s=7: largest block ~1e3, true value
    // ~1e-13), pushing λ_min to ≈ −0.26 after normalization. That kernel
    // evaluation now routes through the cancellation-free single-integral form
    // (`duchon_hybrid_kernel_stable_integral`), so the constrained spectrum is
    // genuinely nonnegative to machine precision. Rather than silently
    // projecting (which would mask a true loss of positive-definiteness), we
    // REJECT a materially-negative spectrum and only clamp float-noise
    // negatives — per gam#1424's required PSD check before normalization.
    reject_nonpsd_then_clamp_noise(&symmetrize_penalty(&omega))
}

/// Enforce the PSD contract on a constrained Duchon bending penalty before
/// normalization (gam#1424).
///
/// The kernel's spectral density is nonnegative, so the constrained Gram must
/// be PSD in exact arithmetic. A genuinely PSD matrix only ever carries
/// negative eigenvalues at the float noise floor; those are clamped to zero. A
/// *materially* negative eigenvalue means the numerical kernel has stopped
/// representing the stated kernel — that is rejected with a clear, actionable
/// error rather than masked, because clamping a −0.26 mode silently fabricates
/// a different penalty.
fn reject_nonpsd_then_clamp_noise(matrix: &Array2<f64>) -> Result<Array2<f64>, BasisError> {
    use crate::linalg::faer_ndarray::FaerEigh;
    use faer::Side;
    let sym = symmetrize_penalty(matrix);
    let n = sym.nrows();
    if n == 0 || n != sym.ncols() {
        return Ok(sym);
    }
    let (evals, _) = FaerEigh::eigh(&sym, Side::Lower).map_err(|e| {
        BasisError::InvalidInput(format!("Duchon penalty PSD check failed: {e}"))
    })?;
    if evals.is_empty() {
        return Ok(sym);
    }
    let max_abs_ev = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let min_ev = evals.iter().copied().fold(f64::INFINITY, f64::min);
    // Noise-floor tolerance in eigenvalue units, so uniform scaling of the
    // penalty does not change the PSD decision (matches `spectral_tolerance`).
    let tol = (n as f64) * 1e-10 * max_abs_ev;
    if min_ev < -tol {
        crate::bail_invalid_basis!(
            "Duchon constrained penalty is not positive semidefinite: λ_min={min_ev:.6e} \
             (tol=−{tol:.6e}, λ_max={max_abs_ev:.6e}). The hybrid kernel's spectral density is \
             nonnegative, so a materially-negative mode indicates the kernel evaluation lost \
             positive-definiteness numerically (see gam#1424)."
        );
    }
    // λ_min is at the noise floor: clamp the harmless negative residue to zero.
    Ok(project_penalty_to_psd_cone(&sym))
}

pub(crate) fn duchon_native_penalty_candidates(
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    kernel_transform: &Array2<f64>,
    outer_identifiability: Option<&Array2<f64>>,
    poly_cols: usize,
) -> Result<Vec<PenaltyCandidate>, BasisError> {
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "duchon_native_penalty_candidates: centers must have at least one column"
        );
    }
    let z = kernel_transform;
    let n_kernel = z.ncols();

    // ω = α² · Zᵀ K_CC Z, embedded in the kernel block of the
    // (n_kernel + poly) pre-identifiability frame (polynomial columns carry no
    // native roughness), then mapped through the outer identifiability `T`.
    let omega = duchon_constrained_bending_penalty(
        centers,
        length_scale,
        power,
        nullspace_order,
        aniso_log_scales,
        z,
    )?;
    let n_pre = n_kernel + poly_cols;
    let mut primary_pre = Array2::<f64>::zeros((n_pre, n_pre));
    primary_pre
        .slice_mut(s![..n_kernel, ..n_kernel])
        .assign(&omega);
    let primary = symmetrize(&project_penalty_matrix(&primary_pre, outer_identifiability));

    let shrink = if poly_cols > 1 {
        let mut shrink_pre = Array2::<f64>::zeros((n_pre, n_pre));
        for col in (n_kernel + 1)..n_pre {
            shrink_pre[[col, col]] = 1.0;
        }
        let shrink = symmetrize(&project_penalty_matrix(&shrink_pre, outer_identifiability));
        Some(shrink)
    } else {
        None
    };
    let mut out = Vec::new();
    out.push(normalize_penalty_candidate(
        primary,
        0,
        PenaltySource::Primary,
    ));
    if let Some(shrink) = shrink {
        out.push(normalize_penalty_candidate(
            shrink,
            0,
            PenaltySource::DoublePenaltyNullspace,
        ));
    }
    Ok(out)
}

/// Farthest-point collocation points per basis center for the lower-order
/// (mass / tension) operator penalties. The sample is space-filling over the
/// data SUPPORT (density-blind — sparse and dense regions weighted alike, which
/// is the regularization you want), `m = OVERSAMPLE·k` capped at `n`: dense
/// enough to resolve the `k`-bump basis, independent of `n`.
pub(crate) const DUCHON_COLLOCATION_OVERSAMPLE: usize = 3;

/// The lower two rungs of the Hilbert scale for a Duchon smooth, as FUNCTION
/// penalties collocated on a density-blind `O(k)` farthest-point sample of the
/// data support:
///   * `mass    = Σ(f−f̄)²` — centered value-design Gram (amplitude / distance
///     from the mean; kernel block only — the affine trend's slope is governed
///     by the null-space ridge, so only the global mean stays free).
///   * `tension = Σ‖∇f‖²`  — gradient-design Gram (first-order roughness).
///
/// Curvature is intentionally NOT here: it is the EXACT RKHS reproducing-norm
/// `Primary` Gram (`duchon_native_penalty_candidates`). These two orders have no
/// convergent continuous integral for the growing polyharmonic kernel, so the
/// data-support quadrature *is* their definition — and it is `O(k)`-in-`n` (the
/// sample size does not grow with the data). Each is a plain penalty (`op = None`)
/// with its own REML λ; REML drives an unhelpful one to zero. Stiffness (`D2`) is
/// absent on purpose — `Primary` is the exact, superior curvature.
/// Emit the lower-order Hilbert-scale penalties — mass `Σ(f−f̄)²` (q=0),
/// tension `Σ‖∇f‖²` (q=1), stiffness `Σ‖∇²f‖²` (q=2) — for a Duchon smooth.
///
/// Each active order routes through the shared closed-form factory, which uses
/// the EXACT continuous reproducing-norm Gram wherever the polyharmonic
/// integral converges (UV/IR + CPD adequacy — `n`-free, the high-`d` accuracy
/// and scale win) and falls back to the `D_qᵀ D_q` quadrature otherwise. That
/// quadrature is collocated on a density-blind, space-filling `O(k)`
/// farthest-point sample of the DATA SUPPORT (`select_thin_plate_knots(data,
/// 3k)`) — never the `k` sparse centers (which under-resolve a `k`-bump basis
/// and made these penalties explode), and never all `n` (which would scale with
/// the data). The collocation `D_q` is built with `max_op = max active order`,
/// so a disabled higher order never allocates its `O(d²)`-row Hessian.
///
/// The operators use the ISOTROPIC metric (`aniso = None`): the anisotropy
/// lives entirely in the curvature (`Primary`) RKHS Gram, which carries its own
/// exact `η`-derivative. Keeping these low-order stabilizers isotropic makes
/// their `η`-gradient identically zero, so the REML anisotropy optimization
/// stays consistent without per-axis operator derivatives.
pub(crate) fn duchon_operator_penalty_candidates(
    collocation_points: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    operator_penalties: &DuchonOperatorPenaltySpec,
    length_scale: Option<f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    per_axis_relevance: bool,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<Vec<PenaltyCandidate>, BasisError> {
    let want_mass = matches!(operator_penalties.mass, OperatorPenaltySpec::Active { .. });
    let mut want_tension = matches!(
        operator_penalties.tension,
        OperatorPenaltySpec::Active { .. }
    );
    let mut want_stiffness = matches!(
        operator_penalties.stiffness,
        OperatorPenaltySpec::Active { .. }
    );
    // Collocation validity: the gradient (D1) and Hessian (D2) operator
    // quadratures are defined only when `2(p+s) > d+1` / `> d+2` respectively
    // (mass/D0 needs only kernel existence, `2(p+s) > d`, guaranteed upstream).
    // Outside that regime the operator's radial limit is undefined, so the
    // order is SKIPPED — the higher Hilbert rungs (Primary curvature, mass,
    // trend) still regularize — rather than failing the whole basis build. E.g.
    // order=0, d=3, s=1 gives `2(p+s)=4`, so tension and stiffness drop out
    // cleanly and the smooth is curvature + mass + trend.
    let effective_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_order);
    let dim = centers.ncols();
    let two_pps = 2.0 * (p_order as f64 + power);
    want_tension = want_tension && two_pps > dim as f64 + 1.0;
    want_stiffness = want_stiffness && two_pps > dim as f64 + 2.0;
    if !want_mass && !want_tension && !want_stiffness {
        return Ok(Vec::new());
    }
    // Effective spec carrying only the collocation-valid active orders.
    let mut effective_spec = operator_penalties.clone();
    if !want_tension {
        effective_spec.tension = OperatorPenaltySpec::Disabled;
    }
    if !want_stiffness {
        effective_spec.stiffness = OperatorPenaltySpec::Disabled;
    }
    let max_op = duchon_max_active_operator_derivative_order(&effective_spec);
    let ops = build_duchon_collocation_operator_matriceswithworkspace(
        centers,
        collocation_points,
        None,
        length_scale,
        power,
        nullspace_order,
        None,
        identifiability_transform.map(|t| t.view()),
        max_op,
        workspace,
    )?;
    let kernel_nullspace = ops.kernel_nullspace_transform.as_ref();
    let poly_cols = ops.polynomial_block_cols;
    // When per-axis relevance is requested (`scale_dims`) and tension is a
    // collocation-valid active order, the single isotropic gradient penalty
    // `Σ‖∇f‖²` is REPLACED by `dim` per-axis penalties `Σ(∂f/∂x_a)²`, each its
    // own REML λ_a (ARD: REML shrinks an axis's nonlinear contribution toward
    // flat only when it does not earn its keep). The isotropic-order penalties
    // (mass, stiffness) still route through the shared factory; tension is
    // removed from its spec here and re-emitted per-axis below. The affine
    // slopes stay in the global trend ridge, so a smooth, linearly-useful axis
    // keeps its slope while its nonlinear λ_a may grow.
    let split_tension = per_axis_relevance && want_tension;
    let factory_spec = if split_tension {
        let mut spec = effective_spec.clone();
        spec.tension = OperatorPenaltySpec::Disabled;
        spec
    } else {
        effective_spec
    };
    // The collocation `D_q` already carry the kernel CPD nullspace `Z`, the
    // polynomial padding, and the identifiability transform (final β-basis), so
    // the factory's quadrature fallback `fast_ata(d_q)` is β-basis. Its
    // closed-form branch rebuilds the same β-basis from `centers` via the SAME
    // `kernel_nullspace` + `poly_cols` + `outer_identifiability`, so both
    // branches agree. q=0 mass is always the centered quadrature Gram.
    let mut candidates = if let Some(length_scale) = length_scale {
        operator_penalty_candidates_closed_form(
            centers,
            &ops.d0,
            &ops.d1,
            &ops.d2,
            &factory_spec,
            p_order,
            duchon_power_to_usize(power),
            length_scale,
            None,
            kernel_nullspace,
            poly_cols,
            identifiability_transform,
        )
    } else {
        operator_penalty_candidates_closed_form_pure(
            centers,
            &ops.d0,
            &ops.d1,
            &ops.d2,
            &factory_spec,
            p_order,
            power,
            None,
            kernel_nullspace,
            poly_cols,
            identifiability_transform,
        )
    };
    if split_tension {
        // `D1` rows are indexed `collocation_i · dim + axis`, so axis `a` owns
        // the strided row set `a, a+dim, a+2·dim, …`. `fast_ata` of that slice
        // is the density-blind support quadrature of `∫(∂f/∂x_a)²` in the final
        // β-basis (the poly null space is zeroed in `D1`, so this is the
        // NONLINEAR gradient energy; the affine slope is the trend ridge's job).
        for axis in 0..dim {
            let d1_axis = ops.d1.slice(s![axis..; dim, ..]).to_owned();
            candidates.push(normalize_penalty_candidate(
                symmetrize(&fast_ata(&d1_axis)),
                0,
                PenaltySource::OperatorRelevance { axis },
            ));
        }
    }
    Ok(candidates)
}

#[cfg(test)]
mod mixed_periodicity_psd_tests {
    //! Regression tests for gam#1422 (mixed-periodicity Duchon penalty must be
    //! PSD — the additive ANOVA reproducing kernel replaces the conditionally-PD
    //! polyharmonic-of-chord-distance kernel) and gam#1423 (the cylinder null
    //! space must contain polynomials of total degree `< m` in the NON-periodic
    //! coordinates, not just constants).
    use super::*;
    use crate::linalg::faer_ndarray::FaerEigh;
    use faer::Side;
    use ndarray::{Array2, array};

    fn cylinder_spec() -> DuchonBasisSpec {
        // m = 2 ⇒ Linear null-space order; pure polyharmonic (no length scale,
        // power = 0) — the only spectrum the mixed-periodicity path supports.
        DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(Array2::<f64>::zeros((0, 0))),
            periodic: None,
            length_scale: None,
            power: 0.0,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: None,
            operator_penalties: DuchonOperatorPenaltySpec::default(),
            boundary: OneDimensionalBoundary::Open,
            radial_reparam: None,
        }
    }

    /// Build the cylinder (`θ` periodic on `[0, 2π]`, `y` free on `[0, 1]`)
    /// Primary penalty via the public mixed-periodicity driver and return it.
    fn cylinder_primary_penalty() -> (Array2<f64>, usize) {
        // Anchor the periodic span to exactly [0, 2π] so the auto-derived period
        // is the geometric period; this mirrors the Python cylinder fixture.
        let two_pi = std::f64::consts::TAU;
        let theta = [
            0.0, 0.6, 1.2, 1.9, 2.5, 3.1, 3.8, 4.4, 5.0, 5.6, two_pi,
        ];
        let y = [
            0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.15, 0.5,
        ];
        let mut centers = Array2::<f64>::zeros((theta.len(), 2));
        for i in 0..theta.len() {
            centers[[i, 0]] = theta[i];
            centers[[i, 1]] = y[i];
        }
        let mut spec = cylinder_spec();
        spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
        let periodic_per_axis = [true, false];
        let built = build_duchon_basis_mixed_periodicity_auto(
            centers.view(),
            &spec,
            &periodic_per_axis,
            None,
        )
        .expect("cylinder mixed-periodicity basis must build");
        let idx = built
            .penaltyinfo
            .iter()
            .position(|info| matches!(info.source, PenaltySource::Primary))
            .expect("cylinder build must emit a Primary penalty");
        (built.penalties[idx].clone(), centers.nrows())
    }

    #[test]
    fn cylinder_penalty_is_symmetric_psd_gam1422() {
        // gam#1422: with the conditionally-PD chord-polyharmonic kernel this
        // fixture produced λ_min ≈ −0.426 (3 materially negative eigenvalues).
        // The additive ANOVA kernel (sum of PSD per-axis reproducing kernels)
        // is PSD by construction. Tolerance matches the Python
        // `_assert_symmetric_psd` slack (1e-8); do NOT weaken it.
        let (penalty, k) = cylinder_primary_penalty();
        assert_eq!(penalty.nrows(), k);
        assert_eq!(penalty.ncols(), k);
        // Symmetry.
        for i in 0..k {
            for j in 0..k {
                assert!(
                    (penalty[[i, j]] - penalty[[j, i]]).abs() <= 1e-9,
                    "cylinder penalty must be symmetric at ({i}, {j})"
                );
            }
        }
        let sym = symmetrize(&penalty);
        let (evals, _) = FaerEigh::eigh(&sym, Side::Lower).expect("eigh");
        let lambda_min = evals.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            lambda_min > -1e-8,
            "cylinder Duchon penalty not PSD; λ_min = {lambda_min:.3e} (gam#1422)"
        );
    }

    #[test]
    fn torus_penalty_is_psd_gam1422() {
        // gam#1422: the torus fixture previously gave λ_min ≈ −0.885 (4 negative
        // eigenvalues). With both axes periodic the additive kernel is the sum of
        // two PSD Bernoulli-Green kernels, hence PSD.
        let two_pi = std::f64::consts::TAU;
        let theta = [0.0, 0.7, 1.5, 2.3, 3.0, 3.9, 4.6, 5.4, two_pi];
        let phi = [0.0, 1.1, 2.0, 0.4, 3.3, 4.8, 5.9, 2.7, two_pi];
        let mut centers = Array2::<f64>::zeros((theta.len(), 2));
        for i in 0..theta.len() {
            centers[[i, 0]] = theta[i];
            centers[[i, 1]] = phi[i];
        }
        let mut spec = cylinder_spec();
        spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
        let periodic_per_axis = [true, true];
        let built = build_duchon_basis_mixed_periodicity_auto(
            centers.view(),
            &spec,
            &periodic_per_axis,
            None,
        )
        .expect("torus mixed-periodicity basis must build");
        let idx = built
            .penaltyinfo
            .iter()
            .position(|info| matches!(info.source, PenaltySource::Primary))
            .expect("torus build must emit a Primary penalty");
        let sym = symmetrize(&built.penalties[idx]);
        let (evals, _) = FaerEigh::eigh(&sym, Side::Lower).expect("eigh");
        let lambda_min = evals.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            lambda_min > -1e-8,
            "torus Duchon penalty not PSD; λ_min = {lambda_min:.3e} (gam#1422)"
        );
    }

    #[test]
    fn cylinder_nullspace_contains_one_and_y_gam1423() {
        // gam#1423: on S¹×ℝ with m = 2 the Duchon null space is {1, y} — both the
        // constant AND the linear-in-the-nonperiodic-coordinate term have zero
        // seminorm, so both must be unpenalised. The polynomial block driving the
        // null space must therefore have exactly 2 columns: a constant column and
        // a `y` column (NOT a θ column — periodic axes contribute only the
        // constant).
        let points = array![
            [0.0_f64, 0.10],
            [1.0, 0.40],
            [2.0, 0.70],
            [3.0, 0.95],
        ];
        let periodic_per_axis = [true, false];
        let block = mixed_periodicity_nullspace_poly_block(points.view(), 2, &periodic_per_axis);
        assert_eq!(
            block.ncols(),
            2,
            "cylinder (m=2) null space must be {{1, y}} — 2 columns (gam#1423)"
        );
        // One column must be the all-ones constant; another must equal the
        // non-periodic coordinate `y` (column 1 of `points`). The block does not
        // depend on θ (column 0).
        let n = points.nrows();
        let mut has_const = false;
        let mut has_y = false;
        let mut depends_on_theta = false;
        for col in 0..block.ncols() {
            let is_const = (0..n).all(|r| (block[[r, col]] - 1.0).abs() < 1e-12);
            let is_y = (0..n).all(|r| (block[[r, col]] - points[[r, 1]]).abs() < 1e-12);
            let is_theta = (0..n).all(|r| (block[[r, col]] - points[[r, 0]]).abs() < 1e-12);
            has_const |= is_const;
            has_y |= is_y;
            depends_on_theta |= is_theta && !is_const;
        }
        assert!(has_const, "null space must include the constant 1");
        assert!(has_y, "null space must include the nonperiodic coordinate y (gam#1423)");
        assert!(
            !depends_on_theta,
            "periodic axis θ must contribute only the constant, never a linear θ column"
        );
    }

    #[test]
    fn cylinder_linear_in_y_is_unpenalised_gam1423() {
        // gam#1423: build the center Gram K_CC and the non-periodic null-space
        // projector Z, then confirm the linear-in-y direction lies in the kernel
        // null space — i.e. f(θ, y) = a + b·y has EXACTLY zero penalty energy.
        let two_pi = std::f64::consts::TAU;
        let theta = [0.0, 0.7, 1.5, 2.3, 3.0, 3.9, 4.6, 5.4, two_pi];
        let y = [0.0, 0.2, 0.45, 0.6, 0.3, 0.8, 0.95, 0.1, 0.5];
        let mut centers = Array2::<f64>::zeros((theta.len(), 2));
        for i in 0..theta.len() {
            centers[[i, 0]] = theta[i];
            centers[[i, 1]] = y[i];
        }
        let periodic_per_axis = [true, false];
        let periods = [two_pi, 1.0];
        let axis_bounds = mixed_periodicity_axis_bounds(centers.view(), &periodic_per_axis);
        let k = centers.nrows();
        let m = 2usize;

        // Z = null(Pᵀ) for P = [1, y].
        let poly_block =
            mixed_periodicity_nullspace_poly_block(centers.view(), m, &periodic_per_axis);
        let z = kernel_constraint_nullspace_from_matrix(poly_block.view())
            .expect("null-space basis must build");

        // K_CC = additive kernel at center pairs.
        let mut k_cc = Array2::<f64>::zeros((k, k));
        fill_symmetric_from_row_kernel(&mut k_cc, |i, j| {
            mixed_periodicity_additive_kernel(
                centers.row(i),
                centers.row(j),
                m,
                &periodic_per_axis,
                &periods,
                &axis_bounds,
            )
        })
        .expect("center kernel must build");

        // Ω = Zᵀ K_CC Z is the realized penalty in the kernel-coefficient frame.
        let omega = fast_ab(&fast_atb(&z, &k_cc), &z);
        let sym = symmetrize(&omega);
        let (evals, _) = FaerEigh::eigh(&sym, Side::Lower).expect("eigh");
        let lambda_min = evals.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            lambda_min > -1e-8,
            "Ω = ZᵀK_CC Z must be PSD; λ_min = {lambda_min:.3e}"
        );

        // The linear-in-y trend is carried by the explicit polynomial columns
        // (which receive a zero penalty block), so the kernel-coefficient penalty
        // Ω never sees it. Confirm directly that evaluating the penalty quadratic
        // form on the constant and linear-in-y design directions yields zero: any
        // design coefficient vector that is purely in the polynomial block has
        // zero penalty because Ω only occupies the kernel block. We model that by
        // checking that K_CC applied to the {1, y} span sits inside the column
        // space the polynomial block already spans — i.e. Pᵀ K_CC P has no energy
        // that Z can pick up. Concretely, Z is orthogonal to {1, y} by
        // construction, so Zᵀ·(linear-in-y vector) = 0.
        let ones: Vec<f64> = vec![1.0; k];
        let yvec: Vec<f64> = (0..k).map(|i| centers[[i, 1]]).collect();
        for (label, v) in [("constant", &ones), ("linear-in-y", &yvec)] {
            let mut proj = vec![0.0f64; z.ncols()];
            for c in 0..z.ncols() {
                let mut acc = 0.0;
                for r in 0..k {
                    acc += z[[r, c]] * v[r];
                }
                proj[c] = acc;
            }
            let norm: f64 = proj.iter().map(|p| p * p).sum::<f64>().sqrt();
            assert!(
                norm < 1e-9,
                "{label} direction must lie in the unpenalised null space \
                 (Zᵀv = 0); got ‖Zᵀv‖ = {norm:.3e} (gam#1423)"
            );
        }
    }
}

#[cfg(test)]
mod hybrid_high_dim_psd_tests {
    //! Regression tests for gam#1424: the high-dimensional hybrid
    //! (Duchon–Matérn) constrained bending penalty must be positive
    //! semidefinite. Before the fix, the kernel was assembled from an
    //! alternating partial-fraction sum whose individually-huge polyharmonic /
    //! Matérn blocks cancelled to the float noise floor; for d=16, p=2, s=7 the
    //! constrained spectrum was λ_min ≈ −0.264 after normalization. The
    //! cancellation-free single-integral kernel evaluation
    //! (`duchon_hybrid_kernel_stable_integral`) restores genuine PSD-ness.
    use super::*;
    use crate::linalg::faer_ndarray::FaerEigh;
    use faer::Side;
    use ndarray::Array2;

    /// Deterministic pseudo-random centers in `[-1, 1]^d` (no RNG dependency in
    /// the core crate). A 64-bit SplitMix-style generator seeded from `(d, k)`.
    fn deterministic_centers(d: usize, k: usize) -> Array2<f64> {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15u64
            .wrapping_mul(d as u64 + 1)
            .wrapping_add(k as u64);
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Map to (−1, 1).
            (z as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        let mut c = Array2::<f64>::zeros((k, d));
        for i in 0..k {
            for j in 0..d {
                c[[i, j]] = next();
            }
        }
        c
    }

    /// Constrained bending-penalty spectrum λ_min for a hybrid Duchon smooth at
    /// the resolved (p, s) for the given dimension. Mirrors the construction in
    /// `duchon_native_penalty_candidates` (Z = null(Pᵀ), Ω = α²·ZᵀK_CC Z) but
    /// stops at the eigenvalues so the test can assert PSD-ness directly.
    fn hybrid_constrained_lambda_min(d: usize, s_order: usize) -> f64 {
        let centers = deterministic_centers(d, 4 * d);
        let nullspace = DuchonNullspaceOrder::Linear; // m = 2 ⇒ p = 2.
        let effective = duchon_effective_nullspace_order(centers.view(), nullspace);
        let poly_block = polynomial_block_from_order(centers.view(), effective);
        let z = kernel_constraint_nullspace_from_matrix(poly_block.view())
            .expect("kernel null-space basis must build");
        let omega = duchon_constrained_bending_penalty(
            centers.view(),
            Some(1.0),
            s_order as f64,
            effective,
            None,
            &z,
        )
        .expect("hybrid constrained bending penalty must build and pass the PSD check");
        let sym = symmetrize(&omega);
        let (evals, _) = FaerEigh::eigh(&sym, Side::Lower).expect("eigh");
        let max_abs = evals.iter().copied().fold(0.0_f64, |a, v| a.max(v.abs()));
        let lambda_min = evals.iter().copied().fold(f64::INFINITY, f64::min);
        // Report in spectral (scale-relative) units so the assertion is
        // invariant to the penalty's overall magnitude.
        lambda_min / max_abs.max(f64::MIN_POSITIVE)
    }

    #[test]
    fn hybrid_d16_m2_s7_constrained_spectrum_is_psd_gam1424() {
        // The exact spectral density ρ^{-2p}(κ²+ρ²)^{-s} is nonnegative, so the
        // constrained Gram is PSD in exact arithmetic. Before gam#1424 the
        // partial-fraction kernel evaluation lost every significant digit here
        // and the normalized λ_min was ≈ −0.264. The stable single-integral
        // kernel keeps λ_min at the float noise floor. Tolerance is the same
        // scale-relative noise floor used by the penalty pipeline
        // (`spectral_tolerance`: n·1e-10); do NOT weaken it.
        let d = 16;
        let n = 4 * d; // penalty dimension upper bound (kernel coeff frame).
        let tol = (n as f64) * 1e-10;
        let lambda_min_rel = hybrid_constrained_lambda_min(d, 7);
        assert!(
            lambda_min_rel >= -tol,
            "d=16, m=2, s=7 hybrid constrained spectrum not PSD: \
             λ_min/λ_max = {lambda_min_rel:.6e} (tol = −{tol:.6e}) (gam#1424)"
        );
    }

    #[test]
    fn hybrid_other_high_dims_constrained_spectrum_is_psd_gam1424() {
        // A spread of high-d hybrid orders that also lost positive-definiteness
        // through partial-fraction cancellation (d=8: ~7 digits lost; d=12: ~13;
        // d=10). Each must now be PSD to the float noise floor.
        for (d, s) in [(8usize, 3usize), (10, 4), (12, 5)] {
            let n = 4 * d;
            let tol = (n as f64) * 1e-10;
            let lambda_min_rel = hybrid_constrained_lambda_min(d, s);
            assert!(
                lambda_min_rel >= -tol,
                "d={d}, m=2, s={s} hybrid constrained spectrum not PSD: \
                 λ_min/λ_max = {lambda_min_rel:.6e} (tol = −{tol:.6e}) (gam#1424)"
            );
        }
    }
}
