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
        },
        kronecker_factored: None,
    })
}

/// Per-pair generalized distance for the mixed-periodicity Duchon basis.
///
/// For each axis ``j``:
///   * **periodic** axis with period ``P_j``: ``d_j(x, y) = (P_j / π) · sin(π·(x − y)/P_j)``,
///     the chord distance on the circle of circumference ``P_j``. The chord
///     metric recovers the Euclidean limit ``d_j → x − y`` as ``P_j → ∞`` and
///     is invariant under the periodic identification ``x ≡ x + P_j``.
///   * **non-periodic** axis: ``d_j(x, y) = x − y``.
///
/// Then ``r = sqrt(Σ d_j²)``. This is the cylinder/torus "extrinsic chord"
/// distance — the same metric used implicitly by the spherical S² basis when
/// embedding in ℝ³. The radial polyharmonic kernel φ(r) defined on this
/// distance yields a positive-definite kernel on the mixed-periodicity
/// product manifold whose nullspace contains the constant function.
#[inline]
pub(crate) fn duchon_mixed_periodicity_distance(
    x: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
    periodic_per_axis: &[bool],
    periods: &[f64],
) -> f64 {
    let d = x.len();
    assert_eq!(d, y.len());
    assert_eq!(d, periodic_per_axis.len());
    assert_eq!(d, periods.len());
    let mut acc = 0.0_f64;
    for j in 0..d {
        let delta = if periodic_per_axis[j] {
            let p = periods[j];
            // Chord distance on circle of circumference P_j.
            (p / std::f64::consts::PI) * (std::f64::consts::PI * (x[j] - y[j]) / p).sin()
        } else {
            x[j] - y[j]
        };
        acc += delta * delta;
    }
    acc.sqrt()
}

/// Build a multi-dimensional Duchon basis with per-axis periodicity.
///
/// Generalizes the 1D `build_periodic_duchon_basis_1d` to mixed-periodicity
/// settings (cylinder ``(True, False)``, torus ``(True, True)``, etc.) by:
///
///   1. Replacing the Euclidean per-pair distance with a generalized
///      cylinder/torus distance: for periodic axes use the chord distance
///      on the circle ``(P_j/π) · sin(π·(x−y)/P_j)``; for non-periodic axes
///      use the plain difference (see [`duchon_mixed_periodicity_distance`]).
///   2. Evaluating the radial polyharmonic Duchon kernel
///      ``φ(r) = c · r^(2m − d)`` (or ``r^(2m−d) · log r`` in the log case)
///      at the generalized distance. The polyharmonic coefficient ``c`` is
///      computed by [`PolyharmonicBlockCoeff::new(m, d)`].
///   3. Forcing the constraint nullspace to ``{constants}`` (the only
///      polynomial that is periodic on every periodic axis). This mirrors
///      the 1D periodic path.
///   4. Returning a single Primary penalty matrix
///      ``Ω = Zᵀ · K_centers · Z`` (the kernel-Gram identity).
///
/// Notes
/// -----
/// * **Math (1D)**: for ``d = 1`` with one periodic axis, this path uses the
///   polyharmonic-of-chord-distance kernel
///   ``c · |(P/π) sin(π Δ/P)|^(2m − 1)``. This is the principled
///   generalization on the circle and is also the kernel the pyffi
///   dispatcher uses for the 1D periodic case; the older Bernoulli
///   Green's-function ``B_{2m}(Δ/P)`` builder is no longer dispatched
///   from pyffi.
/// * **Nullspace audit**: a more principled choice for the cylinder
///   (``d = 2``, axis 0 periodic, axis 1 non-periodic) is the polynomial
///   nullspace ``{1, x_1, x_1², …, x_1^{m−1}}`` — polynomials in the
///   non-periodic axes only, of total degree ``< m``. We keep
///   ``{constants}`` here to match the existing periodic-Duchon convention
///   and avoid widening the polynomial-block construction; users who need
///   richer null spaces on the non-periodic factor can layer a separate
///   tensor smooth.
pub(crate) fn build_duchon_basis_mixed_periodicity(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    centers: Array2<f64>,
    periodic_per_axis: &[bool],
    periods: &[f64],
    workspace: &mut BasisWorkspace,
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
    // Force constant-only nullspace (only periodic-in-every-axis polynomial).
    let effective_nullspace_order = DuchonNullspaceOrder::Zero;
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    let s_order_int = 0usize;
    validate_duchon_kernel_orders(None, p_order, s_order_int as f64, d)?;

    let z = kernel_constraint_nullspace(
        centers.view(),
        effective_nullspace_order,
        &mut workspace.cache,
    )?;
    let kernel_cols = z.ncols();

    // Polyharmonic kernel coefficient for radial order ``m_kernel`` in
    // ``d`` dimensions. We use ``m_kernel = user_m`` so the kernel
    // smoothness order tracks the user's requested ``m``, not the
    // (forced-to-constant) nullspace order.
    let m_kernel = pure_duchon_block_order(user_m, s_order_int as f64);
    let ppc = PolyharmonicBlockCoeff::new(m_kernel, d);

    let centers_owned = centers.clone();
    let k_centers = centers_owned.nrows();
    let n_data = data.nrows();

    // Row-parallel raw kernel: K[i, j] = φ(r_mixed(x_i, c_j)).
    let mut raw_kernel = Array2::<f64>::zeros((n_data, k_centers));
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
                    let c_row = centers_owned.row(j);
                    let r =
                        duchon_mixed_periodicity_distance(x_row, c_row, periodic_per_axis, periods);
                    out_row[j] = ppc.eval(r);
                }
            }
        });

    // Design = [raw_kernel @ z, ones] (constant column carries the
    // constant-only nullspace).
    let design_kernel = fast_ab(&raw_kernel, &z);
    let mut basis = Array2::<f64>::zeros((n_data, kernel_cols + 1));
    basis
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&design_kernel);
    basis.column_mut(kernel_cols).fill(1.0);

    // Penalty: Ω = Zᵀ K_centers Z (kernel-Gram identity in the projected
    // basis), padded with a zero row/col for the constant column.
    let mut center_kernel = Array2::<f64>::zeros((k_centers, k_centers));
    fill_symmetric_from_row_kernel(&mut center_kernel, |i, j| {
        let r = duchon_mixed_periodicity_distance(
            centers_owned.row(i),
            centers_owned.row(j),
            periodic_per_axis,
            periods,
        );
        Ok(ppc.eval(r))
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
            nullspace_order: effective_nullspace_order,
            identifiability_transform,
            input_scales: None,
            aniso_log_scales: None,
            operator_collocation_points: None,
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
    let k = centers.nrows();
    let z = kernel_transform;
    let n_kernel = z.ncols();
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

    // ω = α² · Zᵀ K_CC Z, embedded in the kernel block of the
    // (n_kernel + poly) pre-identifiability frame (polynomial columns carry no
    // native roughness), then mapped through the outer identifiability `T`.
    let amp2 = kernel_amp * kernel_amp;
    let omega = {
        let zt_k = fast_atb(z, &center_kernel);
        fast_ab(&zt_k, z).mapv(|v| v * amp2)
    };
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
