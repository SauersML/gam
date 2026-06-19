use super::*;

pub fn build_spherical_spline_basis(
    data: ArrayView2<'_, f64>,
    spec: &SphericalSplineBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    if matches!(spec.method, SphereMethod::Harmonic) {
        return build_spherical_harmonic_basis(data, spec);
    }
    if matches!(spec.wahba_kernel, SphereWahbaKernel::Pseudo) {
        let mut harmonic_spec = spec.clone();
        harmonic_spec.method = SphereMethod::Harmonic;
        harmonic_spec.penalty_order = 2;
        harmonic_spec.max_degree = Some(
            spec.max_degree
                .unwrap_or_else(|| harmonic_degree_for_wahba_basis_width(spec, data.nrows())),
        );
        return build_spherical_harmonic_basis(data, &harmonic_spec);
    }
    validate_lat_lon_matrix(data, "spherical spline", spec.radians)?;
    if !(1..=4).contains(&spec.penalty_order) {
        crate::bail_invalid_basis!(
            "spherical spline penalty_order must be one of 1, 2, 3, 4; got {}",
            spec.penalty_order
        );
    }
    let centers = match realized_center_strategy(&spec.center_strategy) {
        CenterStrategy::FarthestPoint { num_centers } => {
            select_spherical_farthest_point_centers(data, *num_centers, spec.radians)?
        }
        _ => select_centers_by_strategy(data, &spec.center_strategy)?,
    };
    validate_lat_lon_matrix(centers.view(), "spherical spline centers", spec.radians)?;
    if centers.nrows() < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: centers.nrows(),
        });
    }
    let center_kernel = spherical_wahba_kernel_matrix_with_kind(
        centers.view(),
        centers.view(),
        spec.penalty_order,
        spec.radians,
        spec.wahba_kernel,
    )?;
    let decomposition =
        wahba_low_degree_decomposition(centers.view(), spec.radians, center_kernel.view())?;

    let raw_kernel_design = spherical_wahba_kernel_matrix_with_kind(
        data,
        centers.view(),
        spec.penalty_order,
        spec.radians,
        spec.wahba_kernel,
    )?;
    let raw_design =
        build_wahba_decomposed_design(raw_kernel_design.view(), data, spec.radians, &decomposition);
    let mut raw_penalty = build_wahba_decomposed_penalty(center_kernel.view(), &decomposition);
    let kernel_rank = decomposition.kernel_basis.ncols();
    let diag_scale = if kernel_rank > 0 {
        (0..kernel_rank)
            .map(|i| raw_penalty[[i, i]].abs())
            .sum::<f64>()
            / kernel_rank as f64
    } else {
        0.0
    };
    if diag_scale.is_finite() && diag_scale > 0.0 {
        // The raw finite-center chart is intentionally not coefficient-gauged.
        // Tie a small coefficient ridge to the primary RKHS penalty so REML
        // cannot disable all raw-chart stabilization by driving the separate
        // double-penalty block to zero. This damps sparse polar center leverage
        // without adding another smoothing parameter.
        for i in 0..kernel_rank {
            raw_penalty[[i, i]] += 10.0 * diag_scale;
        }
    }
    let raw_width = raw_design.ncols();
    // Realized-design transform. The Wahba kernels are built without the l=0
    // spherical-harmonic mode, so an additional finite-center coefficient
    // sum-to-zero gauge is not intrinsic to the smooth and can distort sparse
    // polar fits. Keep the raw center coefficients here; the global
    // identifiability pipeline still composes the realized parametric
    // orthogonalization onto this transform and freezes it for prediction.
    let z = match &spec.identifiability {
        SphericalSplineIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != raw_width {
                crate::bail_dim_basis!(
                    "frozen spherical identifiability transform mismatch: {} raw basis columns but transform has {} rows",
                    raw_width,
                    transform.nrows()
                );
            }
            transform.clone()
        }
        SphericalSplineIdentifiability::CenterSumToZero => Array2::<f64>::eye(raw_width),
    };
    let gauge = crate::solver::gauge::Gauge::from_block_transforms(&[z.clone()]);
    let penalty = gauge.restrict_penalty(&raw_penalty);
    let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
        gauge.restrict_design(&raw_design),
    ));
    let (penalty_norm, c_primary) = normalize_penalty(&((&penalty + &penalty.t()) * 0.5));
    let mut candidates = vec![PenaltyCandidate {
        matrix: penalty_norm,
        nullspace_dim_hint: 0,
        source: PenaltySource::Primary,
        normalization_scale: c_primary,
        kronecker_factors: None,
        op: None,
    }];
    if spec.double_penalty {
        let null_shrinkage = build_wahba_decomposed_null_shrinkage(&decomposition);
        let ridge = gauge.restrict_penalty(&null_shrinkage);
        let (ridge_norm, c_ridge) = normalize_penalty(&ridge);
        candidates.push(PenaltyCandidate {
            matrix: ridge_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::DoublePenaltyNullspace,
            normalization_scale: c_ridge,
            kronecker_factors: None,
            op: None,
        });
    }
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::Sphere {
            centers,
            penalty_order: spec.penalty_order,
            method: SphereMethod::Wahba,
            max_degree: None,
            wahba_kernel: spec.wahba_kernel,
            constraint_transform: Some(z),
        },
        kronecker_factored: None,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
    })
}

const SPHERE_UNPENALIZED_LOW_DEGREE: usize = 1;

fn harmonic_degree_for_wahba_basis_width(spec: &SphericalSplineBasisSpec, n_rows: usize) -> usize {
    let target = match &spec.center_strategy {
        CenterStrategy::Auto(inner) => match inner.as_ref() {
            CenterStrategy::FarthestPoint { num_centers }
            | CenterStrategy::EqualMass { num_centers }
            | CenterStrategy::EqualMassCovarRepresentative { num_centers }
            | CenterStrategy::KMeans { num_centers, .. } => *num_centers,
            CenterStrategy::UniformGrid { points_per_dim } => points_per_dim.saturating_pow(2),
            CenterStrategy::UserProvided(centers) => centers.nrows(),
            CenterStrategy::Auto(_) => default_num_centers(n_rows, 2),
        },
        CenterStrategy::FarthestPoint { num_centers } => *num_centers,
        CenterStrategy::EqualMass { num_centers } => *num_centers,
        CenterStrategy::EqualMassCovarRepresentative { num_centers } => *num_centers,
        CenterStrategy::KMeans { num_centers, .. } => *num_centers,
        CenterStrategy::UniformGrid { points_per_dim } => points_per_dim.saturating_pow(2),
        CenterStrategy::UserProvided(centers) => centers.nrows(),
    }
    .max(1);
    (1..=32)
        .find(|&l| l * (l + 2) >= target)
        .unwrap_or_else(|| default_spherical_harmonic_degree(n_rows))
        .max(8)
}

fn real_spherical_harmonic_design_up_to_degree(
    data: ArrayView2<'_, f64>,
    max_degree: usize,
    radians: bool,
) -> Array2<f64> {
    let p = max_degree * (max_degree + 2);
    let to_rad = if radians {
        1.0
    } else {
        std::f64::consts::PI / 180.0
    };
    let norms = precompute_harmonic_norms(max_degree);
    let l_cap = max_degree + 1;
    let mut out = Array2::<f64>::zeros((data.nrows(), p));
    let mut p_buf = vec![0.0_f64; l_cap * l_cap];
    for (i, mut row) in out.outer_iter_mut().enumerate() {
        let lat_raw = data[(i, 0)] * to_rad;
        let lat = lat_raw.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
        let lon = data[(i, 1)] * to_rad;
        fill_real_spherical_harmonics_row(
            lat,
            lon,
            max_degree,
            p_buf.as_mut_slice(),
            norms.as_slice(),
            row.view_mut(),
        );
    }
    out
}

fn orthonormal_column_basis(matrix: ArrayView2<'_, f64>, rel_tol: f64) -> Array2<f64> {
    let n = matrix.nrows();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    let mut scale = 0.0_f64;
    for col in matrix.columns() {
        scale = scale.max(col.iter().map(|v| v * v).sum::<f64>().sqrt());
    }
    let tol = rel_tol * scale.max(1.0);
    for col in matrix.columns() {
        let mut v = col.to_vec();
        for _ in 0..2 {
            for q in &cols {
                let dot = v.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f64>();
                for (vi, qi) in v.iter_mut().zip(q.iter()) {
                    *vi -= dot * qi;
                }
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > tol {
            for vi in &mut v {
                *vi /= norm;
            }
            cols.push(v);
        }
    }
    let mut q = Array2::<f64>::zeros((n, cols.len()));
    for (j, col) in cols.iter().enumerate() {
        for i in 0..n {
            q[(i, j)] = col[i];
        }
    }
    q
}

fn orthonormal_complement(q: ArrayView2<'_, f64>, rel_tol: f64) -> Array2<f64> {
    let n = q.nrows();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    let tol = rel_tol.max(0.0);
    for i in 0..n {
        let mut v = vec![0.0_f64; n];
        v[i] = 1.0;
        for _ in 0..2 {
            for q_col in q.columns() {
                let dot = v.iter().zip(q_col.iter()).map(|(a, b)| a * b).sum::<f64>();
                for (vi, qi) in v.iter_mut().zip(q_col.iter()) {
                    *vi -= dot * qi;
                }
            }
            for c in &cols {
                let dot = v.iter().zip(c.iter()).map(|(a, b)| a * b).sum::<f64>();
                for (vi, ci) in v.iter_mut().zip(c.iter()) {
                    *vi -= dot * ci;
                }
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > tol {
            for vi in &mut v {
                *vi /= norm;
            }
            cols.push(v);
        }
    }
    let mut out = Array2::<f64>::zeros((n, cols.len()));
    for (j, col) in cols.iter().enumerate() {
        for i in 0..n {
            out[[i, j]] = col[i];
        }
    }
    out
}

struct WahbaLowDegreeDecomposition {
    kernel_basis: Array2<f64>,
    low_degree_centers: Option<Array2<f64>>,
    kernel_low_projection: Option<Array2<f64>>,
    low_degree_cols: usize,
}

fn wahba_low_degree_decomposition(
    centers: ArrayView2<'_, f64>,
    radians: bool,
    center_kernel: ArrayView2<'_, f64>,
) -> Result<WahbaLowDegreeDecomposition, BasisError> {
    let low_cols = SPHERE_UNPENALIZED_LOW_DEGREE * (SPHERE_UNPENALIZED_LOW_DEGREE + 2);
    if centers.nrows() <= low_cols {
        return Ok(WahbaLowDegreeDecomposition {
            kernel_basis: Array2::<f64>::eye(centers.nrows()),
            low_degree_centers: None,
            kernel_low_projection: None,
            low_degree_cols: 0,
        });
    }
    let harmonics = real_spherical_harmonic_design_up_to_degree(
        centers,
        SPHERE_UNPENALIZED_LOW_DEGREE,
        radians,
    );
    let low_degree_coefficients = solve_spd_columns_ridged(center_kernel, harmonics.view())?;
    let low_coeff_basis = orthonormal_column_basis(low_degree_coefficients.view(), 1e-10);
    let kernel_basis = orthonormal_complement(low_coeff_basis.view(), 1e-10);
    let low_degree_centers = harmonics;
    if low_degree_centers.ncols() == 0 || kernel_basis.ncols() == centers.nrows() {
        return Ok(WahbaLowDegreeDecomposition {
            kernel_basis,
            low_degree_centers: None,
            kernel_low_projection: None,
            low_degree_cols: 0,
        });
    }
    let center_kernel_reduced = center_kernel.dot(&kernel_basis);
    let low_normal = low_degree_centers.t().dot(&low_degree_centers);
    let low_cross = low_degree_centers.t().dot(&center_kernel_reduced);
    let kernel_low_projection = solve_spd_columns_ridged(low_normal.view(), low_cross.view())?;
    let low_degree_cols = low_degree_centers.ncols();
    Ok(WahbaLowDegreeDecomposition {
        kernel_basis,
        low_degree_centers: Some(low_degree_centers),
        kernel_low_projection: Some(kernel_low_projection),
        low_degree_cols,
    })
}

fn hstack_dense(left: ArrayView2<'_, f64>, right: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = left.nrows();
    assert_eq!(right.nrows(), n);
    let mut out = Array2::<f64>::zeros((n, left.ncols() + right.ncols()));
    out.slice_mut(s![.., 0..left.ncols()]).assign(&left);
    out.slice_mut(s![.., left.ncols()..]).assign(&right);
    out
}

fn build_wahba_decomposed_design(
    raw_kernel_design: ArrayView2<'_, f64>,
    data: ArrayView2<'_, f64>,
    radians: bool,
    decomposition: &WahbaLowDegreeDecomposition,
) -> Array2<f64> {
    let mut kernel_design = raw_kernel_design.dot(&decomposition.kernel_basis);
    match (
        &decomposition.low_degree_centers,
        &decomposition.kernel_low_projection,
    ) {
        (Some(low_degree_centers), Some(kernel_low_projection)) => {
            let raw_low = real_spherical_harmonic_design_up_to_degree(
                data,
                SPHERE_UNPENALIZED_LOW_DEGREE,
                radians,
            );
            assert_eq!(
                raw_low.ncols(),
                low_degree_centers.ncols(),
                "low-degree spherical harmonic design width must match its centers"
            );
            let low_design = raw_low;
            kernel_design -= &low_design.dot(kernel_low_projection);
            hstack_dense(kernel_design.view(), low_design.view())
        }
        _ => kernel_design,
    }
}

fn build_wahba_decomposed_penalty(
    center_kernel: ArrayView2<'_, f64>,
    decomposition: &WahbaLowDegreeDecomposition,
) -> Array2<f64> {
    let kernel_penalty = decomposition
        .kernel_basis
        .t()
        .dot(&center_kernel.dot(&decomposition.kernel_basis));
    let p = kernel_penalty.nrows() + decomposition.low_degree_cols;
    let mut out = Array2::<f64>::zeros((p, p));
    out.slice_mut(s![0..kernel_penalty.nrows(), 0..kernel_penalty.ncols()])
        .assign(&kernel_penalty);
    (&out + &out.t()) * 0.5
}

fn build_wahba_decomposed_null_shrinkage(
    decomposition: &WahbaLowDegreeDecomposition,
) -> Array2<f64> {
    let p = decomposition.kernel_basis.ncols() + decomposition.low_degree_cols;
    let mut out = Array2::<f64>::zeros((p, p));
    if decomposition.low_degree_cols == 0 {
        for i in 0..p {
            out[[i, i]] = 1.0;
        }
    } else {
        for i in 0..decomposition.kernel_basis.ncols() {
            out[[i, i]] = 1.0;
        }
    }
    out
}

fn solve_spd_columns_ridged(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, BasisError> {
    use crate::linalg::faer_ndarray::{FaerArrayView, FaerLlt, FaerSolve};
    use faer::Side;

    let n = a.nrows();
    if n == 0 || a.ncols() != n || b.nrows() != n {
        crate::bail_dim_basis!(
            "ridged SPD solve needs square A and matching RHS rows, got A={}x{} and B={}x{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        );
    }
    let trace: f64 = (0..n).map(|i| a[[i, i]].abs()).sum();
    let ridge = if trace.is_finite() && trace > 0.0 {
        1e-8 * trace / n as f64
    } else {
        1e-12
    };
    let mut m = a.to_owned();
    for i in 0..n {
        m[[i, i]] += ridge;
    }
    let mview = FaerArrayView::new(&m);
    let factor = FaerLlt::new(mview.as_ref(), Side::Lower).map_err(|err| {
        BasisError::InvalidInput(format!(
            "sphere Wahba low-degree Gram solve failed after ridge {ridge:.3e}: {err:?}"
        ))
    })?;
    let rhs_owned = b.to_owned();
    let rhs = FaerArrayView::new(&rhs_owned);
    let solved = factor.solve(rhs.as_ref());
    let mut out = Array2::<f64>::zeros((n, b.ncols()));
    for j in 0..b.ncols() {
        for i in 0..n {
            out[[i, j]] = solved[(i, j)];
        }
    }
    if !out.iter().all(|v| v.is_finite()) {
        return Err(BasisError::InvalidInput(
            "sphere Wahba low-degree Gram solve produced non-finite coefficients".to_string(),
        ));
    }
    Ok(out)
}

/// Precomputed √(2)·N(l,m) coefficients for the real spherical-harmonic
/// real-orthonormal basis (m=0 row uses N(l,0) without the √2 factor).
///
/// Layout: index `[l * (l_cap) + m]` with l_cap = max_degree + 1; entry
/// for m = 0 is stored without the √2 prefactor (since cos(0·φ) = 1 has
/// no twin term to share normalization with).
pub(crate) fn precompute_harmonic_norms(max_degree: usize) -> Vec<f64> {
    let l_cap = max_degree + 1;
    let sqrt2 = std::f64::consts::SQRT_2;
    let mut out = vec![0.0_f64; l_cap * l_cap];
    for l in 0..=max_degree {
        let mut ratio = 1.0_f64;
        let base = ((2 * l + 1) as f64) / (4.0 * std::f64::consts::PI);
        out[l * l_cap] = base.sqrt(); // m = 0
        for m in 1..=l {
            ratio /= ((l - m + 1) * (l + m)) as f64;
            out[l * l_cap + m] = sqrt2 * (base * ratio).sqrt();
        }
    }
    out
}

/// Fill one row of the real-spherical-harmonic design at (lat, lon) in
/// radians. `p_buf` is a scratch buffer of length `(max_degree + 1) ^ 2`
/// owned by the caller; `norms` is the precomputed √2·N(l, m) table
/// produced by `precompute_harmonic_norms`.
///
/// Column order per degree l = 1..=L is:
///   sin(l·φ)·P_{l,l}, sin((l-1)φ)·P_{l,l-1}, …, sin(φ)·P_{l,1},
///   P_{l,0}, cos(φ)·P_{l,1}, …, cos(l·φ)·P_{l,l}
/// times the precomputed norm factor.
pub(crate) fn fill_real_spherical_harmonics_row(
    lat: f64,
    lon: f64,
    max_degree: usize,
    p_buf: &mut [f64],
    norms: &[f64],
    mut row: ndarray::ArrayViewMut1<'_, f64>,
) {
    let l_cap = max_degree + 1;
    assert_eq!(p_buf.len(), l_cap * l_cap);
    assert_eq!(norms.len(), l_cap * l_cap);
    // Recurrence for associated Legendre P_{l,m}(sin(lat)) — standard
    // formulation (no Condon-Shortley phase, since we apply the (-1)^m
    // factor implicitly through cos(m·φ)/sin(m·φ) sign cancellation).
    let x = lat.sin();
    let somx2 = (1.0 - x * x).max(0.0).sqrt();
    for slot in p_buf.iter_mut() {
        *slot = 0.0;
    }
    let idx = |l: usize, m: usize| l * l_cap + m;
    p_buf[idx(0, 0)] = 1.0;
    for m in 1..=max_degree {
        p_buf[idx(m, m)] = -((2 * m - 1) as f64) * somx2 * p_buf[idx(m - 1, m - 1)];
    }
    for m in 0..max_degree {
        p_buf[idx(m + 1, m)] = ((2 * m + 1) as f64) * x * p_buf[idx(m, m)];
    }
    for m in 0..=max_degree {
        for l in (m + 2)..=max_degree {
            p_buf[idx(l, m)] = (((2 * l - 1) as f64) * x * p_buf[idx(l - 1, m)]
                - ((l + m - 1) as f64) * p_buf[idx(l - 2, m)])
                / ((l - m) as f64);
        }
    }
    // sin(m·φ), cos(m·φ) via Chebyshev recurrence — one sin/cos call per
    // row total, instead of 2L.
    let (sin1, cos1) = lon.sin_cos();
    // sin/cos buffers indexed 0..=max_degree; index 0 stores (sin0, cos0).
    let mut sin_buf = [0.0_f64; 33];
    let mut cos_buf = [0.0_f64; 33];
    sin_buf[0] = 0.0;
    cos_buf[0] = 1.0;
    if max_degree >= 1 {
        sin_buf[1] = sin1;
        cos_buf[1] = cos1;
    }
    let two_cos1 = 2.0 * cos1;
    for m in 2..=max_degree {
        sin_buf[m] = two_cos1 * sin_buf[m - 1] - sin_buf[m - 2];
        cos_buf[m] = two_cos1 * cos_buf[m - 1] - cos_buf[m - 2];
    }
    let mut col = 0usize;
    for l in 1..=max_degree {
        // sin(m·φ) factor for m = l, l-1, ..., 1
        for m_pos in (1..=l).rev() {
            row[col] = norms[idx(l, m_pos)] * p_buf[idx(l, m_pos)] * sin_buf[m_pos];
            col += 1;
        }
        // m = 0 (no trig factor)
        row[col] = norms[idx(l, 0)] * p_buf[idx(l, 0)];
        col += 1;
        // cos(m·φ) factor for m = 1, ..., l
        for m in 1..=l {
            row[col] = norms[idx(l, m)] * p_buf[idx(l, m)] * cos_buf[m];
            col += 1;
        }
    }
}

/// Default L for the harmonic basis when the user does not set `max_degree`.
/// Targets ~k = 50 columns (mgcv `sos` default) for sample sizes large enough
/// to support that many parameters, scaling down toward L=2 for small n
/// (target ≈ n/4 columns), and capped at L=12 (168 cols) at the upper end.
///
/// Why these choices:
/// - mgcv's `bs="sos"` defaults to k=50 columns → L=6 (L(L+2)=48 ≈ 50).
/// - On tiny datasets (n=20) a 50-column basis would overfit; rule-of-thumb
///   keeps ≥ ~4 obs per basis column.
/// - The L=12 cap (168 cols) matches the historical wisdom that beyond
///   degree 12 the spherical-harmonic Gram conditioning starts to suffer
///   under realistic data densities.
pub fn default_spherical_harmonic_degree(n_rows: usize) -> usize {
    // Convert a target column count into the smallest L with L(L+2) >= target.
    // L=2 → 8 cols; L=3 → 15; L=4 → 24; L=5 → 35; L=6 → 48; L=7 → 63; L=12 → 168.
    let target_cols = ((n_rows as f64) * 0.25).min(50.0).max(3.0);
    let mut l = 1usize;
    while (l as f64) * (l as f64 + 2.0) < target_cols && l < 12 {
        l += 1;
    }
    l.max(2)
}

/// Build the spherical-harmonic basis (alternative `method == Harmonic`).
pub(crate) fn build_spherical_harmonic_basis(
    data: ArrayView2<'_, f64>,
    spec: &SphericalSplineBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    validate_lat_lon_matrix(data, "spherical-harmonic", spec.radians)?;
    let n = data.nrows();
    let l_max = spec
        .max_degree
        .unwrap_or_else(|| default_spherical_harmonic_degree(n));
    if l_max < 1 {
        crate::bail_invalid_basis!("spherical-harmonic max_degree must be >= 1");
    }
    if l_max > 32 {
        crate::bail_invalid_basis!("spherical-harmonic max_degree {l_max} too large; cap is 32");
    }
    if !(1..=4).contains(&spec.penalty_order) {
        crate::bail_invalid_basis!(
            "spherical-harmonic penalty_order must be one of 1, 2, 3, 4; got {}",
            spec.penalty_order
        );
    }
    let p = l_max * (l_max + 2);
    let to_rad = if spec.radians {
        1.0
    } else {
        std::f64::consts::PI / 180.0
    };
    let norms = precompute_harmonic_norms(l_max);
    let l_cap = l_max + 1;
    let mut design = Array2::<f64>::zeros((n, p));
    // Per-row buffer is small (≤ 33² ≈ 1KB at L=32), so per-thread allocation
    // dominates only at tiny n. For large-scale n we want rows to fan out across
    // threads; rayon::par_iter over a row range with thread-local scratch.
    {
        let mut row_blocks = design
            .axis_chunks_iter_mut(ndarray::Axis(0), 1024)
            .collect::<Vec<_>>();
        let chunks_iter = row_blocks.par_iter_mut().enumerate();
        let chunk_size = 1024usize;
        chunks_iter.try_for_each(|(chunk_idx, block)| -> Result<(), BasisError> {
            let mut p_buf = vec![0.0_f64; l_cap * l_cap];
            let row_offset = chunk_idx * chunk_size;
            for (local_i, mut out_row) in block.outer_iter_mut().enumerate() {
                let i = row_offset + local_i;
                let lat_raw = data[(i, 0)] * to_rad;
                let lat = lat_raw.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
                let lon = data[(i, 1)] * to_rad;
                fill_real_spherical_harmonics_row(
                    lat,
                    lon,
                    l_max,
                    p_buf.as_mut_slice(),
                    norms.as_slice(),
                    out_row.view_mut(),
                );
            }
            Ok(())
        })?;
    }
    // Diagonal Laplace-Beltrami eigenvalue penalty [l(l+1)]^m per (l, m).
    // Degree-2 modes carry real low-frequency sphere signal, but they still
    // need primary-penalty shrinkage under noisy finite samples. For explicit
    // high-degree bases (L > 2), fold an additional [l(l+1)] factor into modes
    // l > 2 in this same primary block. Keeping that tail shrinkage tied to the
    // primary lambda prevents REML from fitting dense equatorial noise with
    // separately under-penalized high-degree modes and then degrading in sparse
    // polar latitude bands.
    //
    // This is already in the natural coefficient coordinates for the real
    // spherical harmonics: the basis is orthonormal on S², so X'X/n is O(1)
    // under uniform sampling, while the diagonal entries are the physical
    // roughness eigenvalues of the final function. Frobenius-normalizing this
    // matrix would divide away that meaningful spectral scale (≈1261 for
    // L=4, m=2), making REML optimize against an artificially tiny physical
    // penalty. Keep the raw operator with normalization_scale=1 so optimizer
    // lambdas are physical lambdas for this smooth.
    // Split the diagonal Laplace-Beltrami curvature penalty into two blocks that
    // carry SEPARATE smoothing parameters (#1246, polar high-latitude quality).
    // The decisive observation is that REML re-selects each block's lambda, so
    // changing a penalty's SCALE within a block is absorbed by lambda and cannot
    // move the fit; only a STRUCTURAL split into an independently-tuned block can.
    //
    //   * Primary (equator-supported signal) — the degree-2 NON-zonal modes
    //     (m != 0: the sectoral/tesseral harmonics Y_{2,+-2}=x^2-y^2, xy and
    //     Y_{2,+-1}). Their basis functions carry a cos(lat) factor, so they
    //     VANISH at the poles and concentrate their amplitude near the equator.
    //     This is where the genuine low-degree sphere signal lives, so REML must
    //     keep this lambda moderate enough to fit it.
    //   * Tail (polar-supported / noise) — every ZONAL mode (m = 0, P_{l,0}) for
    //     l >= 2 PLUS the entire high-degree band (l > 2, all m). The zonal modes
    //     P_{l,0}(sin lat) reach their largest magnitude AT the poles, and the
    //     high-degree modes carry no low-degree signal. Under a single shared
    //     lambda the optimizer had to keep the curvature penalty small enough to
    //     fit the equator-dominant sectoral signal, which left these polar-heavy
    //     modes under-suppressed; their residual wiggle then piled up in the
    //     sparsely sampled polar latitude bands (south-polar/equator RMSE ratio
    //     > 1.4). Routing them to their OWN smoothing parameter lets REML drive
    //     this block independently: for a low-degree truth it crushes the
    //     polar-heavy noise band (no equatorial signal to protect) while leaving
    //     the equator-supported sectoral signal untouched, evening out the
    //     latitude-band error profile. For a genuinely high-degree truth REML
    //     keeps the tail lambda moderate, so signal recovery and the predict
    //     boundedness guarantees are preserved.
    let mut penalty = Array2::<f64>::zeros((p, p));
    let mut tail = Array2::<f64>::zeros((p, p));
    let mut col = 0usize;
    for l in 1..=l_max {
        let laplace = l as f64 * (l as f64 + 1.0);
        let eig = laplace.powi((spec.penalty_order + 2) as i32);
        // Column layout per degree l: [sin(l*phi)P_{l,l} .. sin(phi)P_{l,1},
        // P_{l,0}, cos(phi)P_{l,1} .. cos(l*phi)P_{l,l}], so the local index `l`
        // (0-based) within the (2l+1)-wide block is the zonal m=0 column.
        for local in 0..(2 * l + 1) {
            let is_zonal = local == l;
            if l <= SPHERE_UNPENALIZED_LOW_DEGREE {
                // unpenalized low-degree span
            } else if l > 2 || is_zonal {
                // polar-supported / noise band -> independent tail lambda
                tail[[col, col]] = eig;
            } else {
                // equator-supported degree-2 sectoral/tesseral signal
                penalty[[col, col]] = eig;
            }
            col += 1;
        }
    }
    let has_tail = tail.diag().iter().any(|&v| v > 0.0);
    // Double-penalty shrinkage lives on the primary penalty's null space.
    // The harmonic basis omits the constant mode, so the unpenalized Wahba
    // low-degree span is exactly the l <= SPHERE_UNPENALIZED_LOW_DEGREE block.
    // Penalizing the complement here would duplicate the curvature penalty and
    // leave the nominal null-space penalty inert.
    let mut ridge = Array2::<f64>::zeros((p, p));
    {
        let mut col = 0usize;
        for l in 1..=l_max {
            for _ in 0..(2 * l + 1) {
                if l <= SPHERE_UNPENALIZED_LOW_DEGREE {
                    ridge[[col, col]] = 1.0;
                }
                col += 1;
            }
        }
    }

    // Realized-design identifiability gauge (#1246). The global smooth
    // identifiability pipeline can residualize this harmonic term against the
    // model intercept / overlapping parametric columns and FREEZES the resulting
    // column transform onto `spec.identifiability` as a `FrozenTransform` so that
    // prediction reproduces the exact fit-time basis. Unlike the Wahba path the
    // harmonic builder used to ignore that frozen transform entirely: at predict
    // time it rebuilt the RAW (un-residualized) harmonic design while the fitted
    // coefficients lived in the transformed coordinate system. Applying those
    // coefficients to the raw design produced finite-but-wrong predictions even
    // on the training rows (observed RMSE-vs-truth ≈ 0.59 for a degree-≤2 truth
    // that the basis recovers exactly, i.e. the coefficients were correct but the
    // replayed design was not). Honor the frozen transform here exactly as the
    // Wahba builder does: apply it to the design and every penalty through a
    // `Gauge`, and re-freeze it into the metadata for the next rebuild. A
    // `CenterSumToZero` (fresh-fit) spec carries the identity, so first-fit
    // builds are unchanged.
    let transform = match &spec.identifiability {
        SphericalSplineIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != p {
                crate::bail_dim_basis!(
                    "frozen spherical-harmonic identifiability transform mismatch: {p} basis columns but transform has {} rows",
                    transform.nrows()
                );
            }
            transform.clone()
        }
        SphericalSplineIdentifiability::CenterSumToZero => Array2::<f64>::eye(p),
    };
    let gauge = crate::solver::gauge::Gauge::from_block_transforms(&[transform.clone()]);
    let design = gauge.restrict_design(&design);
    let penalty = gauge.restrict_penalty(&penalty);
    let tail = gauge.restrict_penalty(&tail);
    let ridge = gauge.restrict_penalty(&ridge);

    let mut candidates = vec![PenaltyCandidate {
        matrix: penalty,
        nullspace_dim_hint: 0,
        source: PenaltySource::Primary,
        normalization_scale: 1.0,
        kronecker_factors: None,
        op: None,
    }];
    if has_tail {
        // Independent smoothing parameter for the degree>2 curvature tail. Keep
        // the raw physical roughness operator (normalization_scale=1) so the
        // optimizer lambda stays a physical lambda, matching the primary block.
        candidates.push(PenaltyCandidate {
            matrix: tail,
            nullspace_dim_hint: 0,
            source: PenaltySource::Other("SphereHarmonicHighDegreeTail".to_string()),
            normalization_scale: 1.0,
            kronecker_factors: None,
            op: None,
        });
    }
    if spec.double_penalty {
        let (ridge_norm, c_ridge) = normalize_penalty(&ridge);
        candidates.push(PenaltyCandidate {
            matrix: ridge_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::DoublePenaltyNullspace,
            normalization_scale: c_ridge,
            kronecker_factors: None,
            op: None,
        });
    }
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    Ok(BasisBuildResult {
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design)),
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::Sphere {
            centers: Array2::<f64>::zeros((0, 2)),
            penalty_order: spec.penalty_order,
            method: SphereMethod::Harmonic,
            max_degree: Some(l_max),
            wahba_kernel: spec.wahba_kernel,
            constraint_transform: Some(transform),
        },
        kronecker_factored: None,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
    })
}

pub fn build_matern_basis(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_matern_basiswithworkspace(data, spec, &mut workspace)
}

/// Public forward Matérn design that honors an explicit all-zero
/// `aniso_log_scales` **literally** as the isotropic metric, rather than the
/// κ-optimizer's geometry-seeding sentinel. This is what the public
/// `matern_basis` FFI evaluates, so a caller's explicit isotropic request is not
/// silently hijacked into a data-driven anisotropic kernel (#1042). For every
/// internal/fit build, use [`build_matern_basis`] (auto-seed).
pub fn build_matern_basis_literal_aniso(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_matern_basis_seeded(data, spec, &mut workspace, AnisoSeedMode::Literal)
}

pub fn build_matern_basiswithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    build_matern_basis_seeded(data, spec, workspace, AnisoSeedMode::AutoSeedFromGeometry)
}

pub(crate) fn build_matern_basis_seeded(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
    workspace: &mut BasisWorkspace,
    aniso_seed_mode: AnisoSeedMode,
) -> Result<BasisBuildResult, BasisError> {
    let selected_centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    // Drop redundant centers when an over-specified `centers=K` exceeds the
    // Matérn kernel's numerical rank on the data cloud (#755). Reducing the base
    // (pre-periodic-expansion) center set keeps the stored metadata, the
    // periodic replication, the identifiability transform, and the penalty all
    // built from the same full-rank center subset. The contrasts used for the
    // rank Gram come from the selected centers so anisotropy is honored.
    //
    // The reduction depends on the realized kernel rank over *this* data cloud,
    // so it must run exactly once — at the cold (train-time) build that also
    // builds the identifiability transform over the surviving centers. A
    // `FrozenTransform` build replays a fit whose centers (pinned `UserProvided`)
    // and transform were already reduced and frozen mutually consistently; the
    // prediction/replay data cloud is different (and often smaller or degenerate)
    // and re-running RRQR here would prune to a *different* count (e.g. 16→0),
    // leaving a stale N-row transform over a reduced-column basis and a hard
    // "centers vs transform rows" dimension mismatch at predict time (#1090).
    // When frozen, keep the pinned centers verbatim.
    let original_centers = if matches!(
        spec.identifiability,
        MaternIdentifiability::FrozenTransform { .. }
    ) {
        selected_centers
    } else {
        let reduce_aniso = resolve_matern_forward_aniso(
            aniso_seed_mode,
            selected_centers.view(),
            spec.aniso_log_scales.as_deref(),
        );
        matern_rank_reduce_centers(
            data,
            &selected_centers,
            spec.length_scale,
            spec.nu,
            reduce_aniso.as_deref(),
        )?
    };
    let centers = expand_periodic_centers(&original_centers, spec.periodic.as_deref())?;
    // Resolve the anisotropy contrasts for the forward design (see
    // `resolve_matern_forward_aniso` / [`AnisoSeedMode`]): `Literal` honors an
    // explicit all-zero η as the isotropic metric (#1042), while
    // `AutoSeedFromGeometry` (the default for internal/fit builds) seeds η from
    // the knot cloud — the κ-optimizer's seeding sentinel.
    let aniso = resolve_matern_forward_aniso(
        aniso_seed_mode,
        centers.view(),
        spec.aniso_log_scales.as_deref(),
    );
    let z_opt = matern_identifiability_transform(centers.view(), &spec.identifiability)?;
    let identifiability_transform = z_opt.clone();
    let full_transform = z_opt.as_ref().map(|z| {
        if spec.include_intercept {
            append_intercept_to_transform(z)
        } else {
            z.clone()
        }
    });
    // Frozen double-penalty nullspace-shrinkage decision carried by a
    // FrozenTransform identifiability (gam#787/#860). `None` for cold/non-frozen
    // builds → decide via the κ-dependent spectral test; `Some(b)` (set at the
    // bootstrap-κ freeze) forces the decision so the learned-penalty count is
    // invariant across the κ optimizer's per-trial design rebuilds.
    let frozen_nullspace_shrinkage_survived = match &spec.identifiability {
        MaternIdentifiability::FrozenTransform {
            nullspace_shrinkage_survived,
            ..
        } => *nullspace_shrinkage_survived,
        _ => None,
    };
    // Realized decision, recorded into metadata so the freeze step can pin it.
    // Each candidate-emission arm below overwrites this with its actual outcome.
    let mut realized_nullspace_shrinkage_survived = false;
    let design_cols =
        z_opt.as_ref().map_or(centers.nrows(), Array2::ncols) + usize::from(spec.include_intercept);
    let dense_bytes = dense_design_bytes(data.nrows(), design_cols);
    let matern_auto_chunk = auto_streaming_chunk_size_for_dense(data.nrows(), design_cols);
    let use_streaming = matern_auto_chunk.is_some();
    let use_lazy = !use_streaming
        && should_use_lazy_spatial_design(data.nrows(), design_cols, workspace.policy());
    let (design, candidates) = if let Some(chunk) = matern_auto_chunk {
        log::info!(
            "Matérn basis auto-streaming evaluator: n={} p={} chunk_size={}",
            data.nrows(),
            design_cols,
            chunk,
        );
        let shared_data = shared_owned_data_matrix(data, &workspace.cache);
        let op = StreamingMaternEvaluator::new(
            shared_data,
            Arc::new(centers.clone()),
            spec.length_scale,
            spec.nu,
            aniso.clone(),
            z_opt.as_ref().map(|z| Arc::new(z.clone())),
            spec.include_intercept,
            Some(chunk),
        )
        .map_err(BasisError::InvalidInput)?;
        let design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(op)));
        let candidates = if spec.double_penalty {
            let penalty_kernel = build_matern_kernel_penalty(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                aniso.as_deref(),
            )?;
            let primary = project_penalty_matrix(&penalty_kernel, full_transform.as_ref());
            let (candidates, survived) = matern_double_penalty_candidates_with_decision(
                &primary,
                frozen_nullspace_shrinkage_survived,
            )?;
            realized_nullspace_shrinkage_survived = survived;
            candidates
        } else {
            build_matern_operator_penalty_candidates(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                z_opt.as_ref(),
                aniso.as_deref(),
            )?
        };
        (design, candidates)
    } else if use_lazy {
        // log::info! — deliberate memory-saving choice, not an anomaly.
        log::info!(
            "Matérn basis switching to lazy chunked design: n={} p={} ({:.1} MiB dense)",
            data.nrows(),
            design_cols,
            dense_bytes as f64 / (1024.0 * 1024.0),
        );
        let shared_data = shared_owned_data_matrix(data, &workspace.cache);
        let d = data.ncols();
        let length_scale = spec.length_scale;
        let nu = spec.nu;
        let poly_basis = if spec.include_intercept {
            Some(Arc::new(Array2::<f64>::ones((data.nrows(), 1))))
        } else {
            None
        };
        let kernel_gauge = z_opt.as_ref().map(|z| {
            Arc::new(crate::solver::gauge::Gauge::from_block_transforms(&[
                z.clone()
            ]))
        });
        let design = if let Some(eta) = aniso.as_ref() {
            let metric_weights = eta.iter().map(|&v| (2.0 * v).exp()).collect::<Vec<_>>();
            let kernel = move |data_row: &[f64], center_row: &[f64]| -> f64 {
                let mut q = 0.0f64;
                for axis in 0..data_row.len() {
                    let delta = data_row[axis] - center_row[axis];
                    q += metric_weights[axis] * delta * delta;
                }
                matern_kernel_from_distance(q.sqrt(), length_scale, nu)
                    .expect("validated Matérn inputs should not fail")
            };
            let op = ChunkedKernelDesignOperator::new(
                shared_data.clone(),
                Arc::new(centers.clone()),
                kernel,
                kernel_gauge.clone(),
                poly_basis.clone(),
            )
            .map_err(BasisError::InvalidInput)?;
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(op)))
        } else {
            let kernel = move |data_row: &[f64], center_row: &[f64]| -> f64 {
                let r = stable_euclidean_norm((0..d).map(|axis| data_row[axis] - center_row[axis]));
                matern_kernel_from_distance(r, length_scale, nu)
                    .expect("validated Matérn inputs should not fail")
            };
            let op = ChunkedKernelDesignOperator::new(
                shared_data,
                Arc::new(centers.clone()),
                kernel,
                kernel_gauge,
                poly_basis,
            )
            .map_err(BasisError::InvalidInput)?;
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(op)))
        };
        let candidates = if spec.double_penalty {
            let penalty_kernel = build_matern_kernel_penalty(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                aniso.as_deref(),
            )?;
            let primary = project_penalty_matrix(&penalty_kernel, full_transform.as_ref());
            let (candidates, survived) = matern_double_penalty_candidates_with_decision(
                &primary,
                frozen_nullspace_shrinkage_survived,
            )?;
            realized_nullspace_shrinkage_survived = survived;
            candidates
        } else {
            build_matern_operator_penalty_candidates(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                z_opt.as_ref(),
                aniso.as_deref(),
            )?
        };
        (design, candidates)
    } else {
        let m = create_matern_spline_basiswithworkspace(
            data,
            centers.view(),
            spec.length_scale,
            spec.nu,
            spec.include_intercept,
            aniso.as_deref(),
            workspace,
        )?;
        let design = if let Some(transform) = full_transform.as_ref() {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(fast_ab(
                &m.basis, transform,
            )))
        } else {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(m.basis.clone()))
        };
        let candidates = if spec.double_penalty {
            let (candidates, survived) = build_matern_double_penalty_candidates(
                &m,
                full_transform.as_ref(),
                frozen_nullspace_shrinkage_survived,
            )?;
            realized_nullspace_shrinkage_survived = survived;
            candidates
        } else {
            build_matern_operator_penalty_candidates(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                z_opt.as_ref(),
                aniso.as_deref(),
            )?
        };
        (design, candidates)
    };
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::Matern {
            centers: original_centers,
            length_scale: spec.length_scale,
            periodic: spec.periodic.clone(),
            nu: spec.nu,
            include_intercept: spec.include_intercept,
            identifiability_transform,
            input_scales: None,
            aniso_log_scales: aniso,
            nullspace_shrinkage_survived: realized_nullspace_shrinkage_survived,
        },
        kronecker_factored: None,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
    })
}

#[inline(always)]
pub(crate) fn eval_polywith_derivatives(coeffs: &[f64], a: f64) -> (f64, f64, f64) {
    let mut p = 0.0;
    let mut p1 = 0.0;
    let mut p2 = 0.0;
    for (i, &c) in coeffs.iter().enumerate() {
        p += c * a.powi(i as i32);
        if i >= 1 {
            p1 += (i as f64) * c * a.powi((i - 1) as i32);
        }
        if i >= 2 {
            p2 += (i as f64) * ((i - 1) as f64) * c * a.powi((i - 2) as i32);
        }
    }
    (p, p1, p2)
}

#[inline(always)]
pub(crate) fn maternvalue_psi_triplet(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<(f64, f64, f64), BasisError> {
    // Exact value + hyper-derivatives for psi = log(kappa)
    // Half-integer Matérn kernels are represented as:
    //   phi(r) = p(a) * exp(-a),
    //   a = s r,   s = sqrt(2 nu) * kappa,   kappa = 1/length_scale.
    //
    // Differentiating with respect to a:
    //   d/da [p(a)e^{-a}]       = (p' - p)e^{-a}
    //   d^2/da^2 [p(a)e^{-a}]   = (p'' - 2p' + p)e^{-a}.
    //
    // We need derivatives w.r.t. psi=log(kappa), not r:
    //   da/dpsi = a,
    // therefore
    //   phi_psi      = a * (dphi/da)
    //   phi_psi_psi  = a*(dphi/da) + a^2*(d^2phi/da^2).
    //
    // This path is fully analytic and avoids FD in the hyper-derivative chain.
    validate_matern_inputs(r, length_scale)?;

    let kappa = 1.0 / length_scale;
    let (s, p): (f64, &[f64]) = match nu {
        MaternNu::Half => (kappa, &[1.0]),
        MaternNu::ThreeHalves => (3.0_f64.sqrt() * kappa, &[1.0, 1.0]),
        MaternNu::FiveHalves => (5.0_f64.sqrt() * kappa, &[1.0, 1.0, 1.0 / 3.0]),
        MaternNu::SevenHalves => (7.0_f64.sqrt() * kappa, &[1.0, 1.0, 2.0 / 5.0, 1.0 / 15.0]),
        MaternNu::NineHalves => (
            9.0_f64.sqrt() * kappa,
            &[1.0, 1.0, 3.0 / 7.0, 2.0 / 21.0, 1.0 / 105.0],
        ),
    };
    let a = s * r;
    // When a > 700, exp(-a) underflows to 0 while p(a) can overflow to Inf,
    // producing 0 * Inf = NaN.  All terms carry exp(-a) as a factor, so the
    // triplet is exactly zero for large a.
    if a > 700.0 {
        return Ok((0.0, 0.0, 0.0));
    }
    let e = (-a).exp();
    let (p0, p1, p2) = eval_polywith_derivatives(p, a);
    let value = e * p0;
    // Chain through psi=log(kappa): da/dpsi = a.
    let value_psi = e * a * (p1 - p0);
    let value_psi_psi = e * (a * (p1 - p0) + a * a * (p2 - 2.0 * p1 + p0));
    Ok((value, value_psi, value_psi_psi))
}

#[inline(always)]
pub(crate) fn exp_poly_scaled_s2_psi_triplet(
    s: f64,
    a: f64,
    coeffs: &[f64],
    scalar: f64,
) -> (f64, f64, f64) {
    // Helper for operator terms of the form:
    //   y(psi) = scalar * s(psi)^2 * exp(-a) * P(a),
    // where
    //   a = s r,  ds/dpsi = s,  da/dpsi = a.
    //
    // Product/chain expansion gives:
    //   y'  = scalar*s^2*e^{-a} [2P + a(P' - P)]
    //   y'' = scalar*s^2*e^{-a} [4P + 5a(P' - P) + a^2(P'' - 2P' + P)].
    //
    // Used for:
    // - phi''(r) pieces
    // - phi'(r)/r closed forms for nu>=3/2
    // under psi-derivatives.
    // When a > 700, exp(-a) underflows to 0 while the polynomial can overflow,
    // giving 0 * Inf = NaN.  All terms carry exp(-a), so the result is exactly 0.
    if a > 700.0 {
        return (0.0, 0.0, 0.0);
    }
    let e = (-a).exp();
    let (p0, p1, p2) = eval_polywith_derivatives(coeffs, a);
    let d = p1 - p0;
    let y = scalar * s * s * e * p0;
    let y_psi = scalar * s * s * e * (2.0 * p0 + a * d);
    let y_psi_psi = scalar * s * s * e * (4.0 * p0 + 5.0 * a * d + a * a * (p2 - 2.0 * p1 + p0));
    (y, y_psi, y_psi_psi)
}

#[inline(always)]
pub(crate) fn matern_operator_psi_triplet(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
    dimension: usize,
) -> Result<
    (
        f64, // phi
        f64, // phi_psi
        f64, // phi_psi_psi
        f64, // phi_r_over_r
        f64, // derivative of phi_r_over_r with respect to psi
        f64, // second derivative of phi_r_over_r with respect to psi
        f64, // lap
        f64, // lap_psi
        f64, // lap_psi_psi
    ),
    BasisError,
> {
    // Operator-level analytic identities used by Thread-1 penalties:
    //   D0 uses phi,
    //   D1 uses phi'(r)/r,
    //   D2 uses full Hessian rows:
    //       ∂²φ/∂x_b∂x_c = δ_bc w_b q + (w_b h_b)(w_c h_c)t.
    //
    // For each half-integer nu, we use closed forms:
    //   phi''(r)      = s^2 * e^{-a} * R_nu(a),
    //   phi'(r)/r     = -s^2 * e^{-a} * Q_nu(a),  (nu>=3/2),
    // where Q_nu, R_nu are low-degree polynomials.
    // The `q` and `rr` arrays below are the literal coefficient arrays for
    // Q_nu(a) and R_nu(a), including the normalization factors such as 1/3,
    // 1/15, and 1/105.
    //
    // Then psi-derivatives are obtained exactly through
    // exp_poly_scaled_s2_psi_triplet.
    let (phi, phi_psi, phi_psi_psi) = maternvalue_psi_triplet(r, length_scale, nu)?;
    let kappa = 1.0 / length_scale;
    let d = dimension as f64;
    let (s, q, rr): (f64, &[f64], &[f64]) = match nu {
        MaternNu::Half => (kappa, &[1.0], &[1.0]),
        MaternNu::ThreeHalves => (3.0_f64.sqrt() * kappa, &[1.0], &[-1.0, 1.0]),
        MaternNu::FiveHalves => (
            5.0_f64.sqrt() * kappa,
            &[1.0 / 3.0, 1.0 / 3.0],
            &[-1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0],
        ),
        MaternNu::SevenHalves => (
            7.0_f64.sqrt() * kappa,
            &[1.0 / 5.0, 1.0 / 5.0, 1.0 / 15.0],
            &[-1.0 / 5.0, -1.0 / 5.0, 0.0, 1.0 / 15.0],
        ),
        MaternNu::NineHalves => (
            9.0_f64.sqrt() * kappa,
            &[1.0 / 7.0, 1.0 / 7.0, 2.0 / 35.0, 1.0 / 105.0],
            &[
                -1.0 / 7.0,
                -1.0 / 7.0,
                -1.0 / 35.0,
                2.0 / 105.0,
                1.0 / 105.0,
            ],
        ),
    };
    let a = s * r;
    let (phi_rr, phi_rr_psi, phi_rr_psi_psi) = exp_poly_scaled_s2_psi_triplet(s, a, rr, 1.0);

    // nu=1/2 has singular phi'(r)/r ~ -kappa/r as r->0.
    // We use the same finite r-floor regularization as operator assembly.
    let (ratio, ratio_psi, ratio_psi_psi) = if matches!(nu, MaternNu::Half) {
        let r_eff = r.max(1e-12);
        let e_eff = (-a).exp();
        let g = -(s / r_eff) * e_eff;
        let g_psi = -(s / r_eff) * e_eff * (1.0 - a);
        let g_psi_psi = -(s / r_eff) * e_eff * (1.0 - 3.0 * a + a * a);
        (g, g_psi, g_psi_psi)
    } else {
        exp_poly_scaled_s2_psi_triplet(s, a, q, -1.0)
    };

    let lap = phi_rr + (d - 1.0) * ratio;
    let lap_psi = phi_rr_psi + (d - 1.0) * ratio_psi;
    let lap_psi_psi = phi_rr_psi_psi + (d - 1.0) * ratio_psi_psi;

    if !phi.is_finite()
        || !phi_psi.is_finite()
        || !phi_psi_psi.is_finite()
        || !ratio.is_finite()
        || !ratio_psi.is_finite()
        || !ratio_psi_psi.is_finite()
        || !lap.is_finite()
        || !lap_psi.is_finite()
        || !lap_psi_psi.is_finite()
    {
        crate::bail_invalid_basis!(
            "non-finite Matérn psi-derivative operator values at r={r}, length_scale={length_scale}, nu={nu:?}"
        );
    }
    Ok((
        phi,
        phi_psi,
        phi_psi_psi,
        ratio,
        ratio_psi,
        ratio_psi_psi,
        lap,
        lap_psi,
        lap_psi_psi,
    ))
}

pub(crate) fn gram_and_psi_derivatives_from_operator(
    d: &Array2<f64>,
    d_psi: &Array2<f64>,
    d_psi_psi: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    // Raw Gram derivatives from operator-collocation matrix D(psi):
    //   S_raw(psi) = D(psi)^T D(psi)
    //   S_raw'     = D'^T D + D^T D'
    //   S_raw''    = D''^T D + 2 D'^T D' + D^T D''.
    //
    // These are exactly the product-rule formulas requested in the math spec.
    let s_raw = symmetrize(&fast_ata(d));
    let s_raw_psi = symmetrize(&(d_psi.t().dot(d) + d.t().dot(d_psi)));
    let s_raw_psi_psi =
        symmetrize(&(d_psi_psi.t().dot(d) + d.t().dot(d_psi_psi) + 2.0 * d_psi.t().dot(d_psi)));
    (s_raw, s_raw_psi, s_raw_psi_psi)
}

/// Cross second derivative of the Gram penalty w.r.t. two different axes a and b:
///   S_raw_{ab} = D_{ab}'^T D + D'^T D_b + D_a'^T D_b + D^T D_{ab}
/// where D_a = ∂D/∂ψ_a, D_b = ∂D/∂ψ_b, D_{ab} = ∂²D/∂ψ_a∂ψ_b.
pub(crate) fn gram_cross_psi_derivative_from_operator(
    d: &Array2<f64>,
    d_a: &Array2<f64>,
    d_b: &Array2<f64>,
    d_ab: &Array2<f64>,
) -> Array2<f64> {
    symmetrize(&(d_ab.t().dot(d) + d.t().dot(d_ab) + d_a.t().dot(d_b) + d_b.t().dot(d_a)))
}

/// Normalize a cross second derivative ∂²S~_m/∂ψ_a∂ψ_b using the Frobenius norm chain rule.
///
/// Given:
///   S     = the raw Gram penalty (axis-independent)
///   S_a   = ∂S/∂ψ_a (first derivative, axis a)
///   S_b   = ∂S/∂ψ_b (first derivative, axis b)
///   S_ab  = ∂²S/∂ψ_a∂ψ_b (cross second derivative, raw)
///   c     = ||S||_F (Frobenius norm)
///
/// The normalized cross second derivative is:
///   S~_{ab} = S_{ab}/c - (c_a/c²)·S_b - (c_b/c²)·S_a + (2·c_a·c_b/c³ - c_{ab}/c²)·S
///
/// where c_a = tr(S'·S_a)/c, c_b = tr(S'·S_b)/c, and
///   c_{ab} = [tr(S_a'·S_b) + tr(S'·S_{ab})]/c - c_a·c_b/c.
pub(crate) fn normalize_penalty_cross_psi_derivative(
    s: &Array2<f64>,
    s_a: &Array2<f64>,
    s_b: &Array2<f64>,
    s_ab: &Array2<f64>,
    c: f64,
) -> Array2<f64> {
    if !c.is_finite() || c <= 1e-12 {
        return Array2::<f64>::zeros(s.raw_dim());
    }

    let c2 = c * c;
    let c3 = c2 * c;

    // c_a = tr(S^T S_a) / c
    let a_val = trace_of_product(s, s_a);
    let c_a = a_val / c;

    // c_b = tr(S^T S_b) / c
    let b_val = trace_of_product(s, s_b);
    let c_b = b_val / c;

    // c_{ab} = [tr(S_a^T S_b) + tr(S^T S_{ab})] / c - c_a * c_b / c
    let cross_val = trace_of_product(s_a, s_b) + trace_of_product(s, s_ab);
    let c_ab = cross_val / c - c_a * c_b / c;

    // S~_{ab} = S_{ab}/c - (c_a/c²)·S_b - (c_b/c²)·S_a + (2·c_a·c_b/c³ - c_{ab}/c²)·S
    let coeff_s = 2.0 * c_a * c_b / c3 - c_ab / c2;
    s_ab.mapv(|v| v / c) - s_b.mapv(|v| c_a / c2 * v) - s_a.mapv(|v| c_b / c2 * v)
        + s.mapv(|v| coeff_s * v)
}

#[inline(always)]
pub(crate) fn trace_of_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.t().dot(b).diag().sum()
}

pub(crate) fn normalize_penaltywith_psi_derivatives(
    s: &Array2<f64>,
    s_psi: &Array2<f64>,
    s_psi_psi: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, f64) {
    // Exact constrained-space Frobenius normalization derivatives:
    //
    // Let S = S_con(psi), c = ||S||_F = sqrt(tr(S^T S)).
    // Define:
    //   a = tr(S^T S'),
    //   b = tr((S')^T S') + tr(S^T S'').
    //
    // Then:
    //   c'  = a/c,
    //   c'' = b/c - a^2/c^3.
    //
    // For normalized S~ = S/c:
    //   S~'  = S'/c - (c'/c^2) S
    //   S~'' = S''/c - 2(c'/c^2)S' + (2(c')^2/c^3 - c''/c^2)S.
    //
    // This keeps hyper-derivative scaling coherent with the constrained REML
    // objective and matches the user-provided trace-only derivation.
    let fro2 = trace_of_product(s, s);
    let c = fro2.sqrt();
    if !c.is_finite() || c <= 1e-12 {
        return (
            s.clone(),
            Array2::<f64>::zeros(s.raw_dim()),
            Array2::<f64>::zeros(s.raw_dim()),
            1.0,
        );
    }

    let a = trace_of_product(s, s_psi);
    let b = trace_of_product(s_psi, s_psi) + trace_of_product(s, s_psi_psi);
    let c_psi = a / c;
    let c_psi_psi = b / c - (a * a) / (c * c * c);

    let s_tilde = s.mapv(|v| v / c);
    let s_tilde_psi = s_psi.mapv(|v| v / c) - s.mapv(|v| (c_psi / (c * c)) * v);
    let s_tilde_psi_psi = s_psi_psi.mapv(|v| v / c) - s_psi.mapv(|v| 2.0 * c_psi / (c * c) * v)
        + s.mapv(|v| ((2.0 * c_psi * c_psi) / (c * c * c) - c_psi_psi / (c * c)) * v);

    (s_tilde, s_tilde_psi, s_tilde_psi_psi, c)
}

pub(crate) fn build_matern_operator_penalty_psi_derivatives(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
    aniso_log_scales: Option<&[f64]>,
) -> Result<(Vec<Array2<f64>>, Vec<Array2<f64>>), BasisError> {
    // Full operator-to-penalty derivative pipeline in constrained coordinates:
    //
    // 1. Build D0, D1, D2 and their psi-derivatives from analytic radial forms.
    // 2. Apply identifiability transform Z at operator level:
    //      D_con = D Z, D_con' = D' Z, D_con'' = D'' Z
    //    (valid because Z is psi-independent).
    // 3. Build raw Gram derivatives per operator block:
    //      S_raw = D_con^T D_con, etc.
    // 4. Normalize each block by constrained Frobenius norm and propagate
    //    derivatives with exact quotient rules.
    //
    // Returned vectors correspond to [S0, S1, S2] derivatives after
    // constrained-space normalization.
    let p = centers.nrows();
    let d = centers.ncols();
    let mut d0_raw = Array2::<f64>::zeros((p, p));
    let mut d1_raw = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw = Array2::<f64>::zeros((p * d * d, p));
    let mut d0_raw_psi = Array2::<f64>::zeros((p, p));
    let mut d1_raw_psi = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw_psi = Array2::<f64>::zeros((p * d * d, p));
    let mut d0_raw_psi_psi = Array2::<f64>::zeros((p, p));
    let mut d1_raw_psi_psi = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw_psi_psi = Array2::<f64>::zeros((p * d * d, p));
    let metric_weights = aniso_log_scales
        .map(centered_aniso_metric_weights)
        .unwrap_or_else(|| vec![1.0; d]);

    for k in 0..p {
        for j in 0..p {
            let (r, _s_vec) = if let Some(eta) = aniso_log_scales {
                aniso_distance_and_components(
                    centers.row(k).as_slice().unwrap(),
                    centers.row(j).as_slice().unwrap(),
                    eta,
                )
            } else {
                (
                    stable_euclidean_norm((0..d).map(|c| centers[[k, c]] - centers[[j, c]])),
                    (0..d)
                        .map(|c| {
                            let h = centers[[k, c]] - centers[[j, c]];
                            h * h
                        })
                        .collect(),
                )
            };
            let (
                phi,
                phi_psi,
                phi_psi_psi,
                ratio,
                ratio_psi,
                ratio_psi_psi,
                _lap,
                lap_psi,
                lap_psi_psi,
            ) = matern_operator_psi_triplet(r, length_scale, nu, d)?;
            let (_, _q_shape, t, _t_r, _t_rr) =
                matern_aniso_extended_radial_scalars(r, length_scale, nu)?;
            let q = ratio;
            let q_psi = ratio_psi;
            let q_psi_psi = ratio_psi_psi;
            // Mixed-curvature Hessian scalar t = (φ''(r) − φ'(r)/r)/r² and its
            // ψ-derivatives (ψ = log κ = −log ℓ, with r held fixed). The earlier
            // `4t + r·t_r` / `16t + 9 r·t_r + r²·t_rr` expressions confused the
            // fixed-ℓ radial derivative `t_r` with the fixed-r ψ-derivative and
            // were simply wrong (verified against finite differences, #1122):
            // they made the operator-penalty D₂ ψ-gradient inconsistent with the
            // penalty itself, so the isotropic-κ joint REML gradient never went
            // to zero and the Matérn fit stalled at the 80-iteration cap while
            // the thin-plate/Duchon siblings (different penalty path) converged.
            //
            // The Laplacian `lap = φ''(r) + (d−1)·φ'(r)/r` already returns exact
            // ψ-derivatives from `matern_operator_psi_triplet`, and `r` is fixed
            // under ψ, so the linear identity
            //   t = (lap − d·ratio)/r²  ⇒  t_ψ = (lap_ψ − d·ratio_ψ)/r²,
            //   t_ψψ = (lap_ψψ − d·ratio_ψψ)/r²
            // gives the correct, FD-matching derivatives. At a center collision
            // (r = 0) `t` and its ψ-derivatives are multiplied by displacement
            // factors that vanish identically, so we use the same 0 convention
            // as the value-side `t`.
            let (t_psi, t_psi_psi) = if r < 1e-14 {
                (0.0, 0.0)
            } else {
                let r2 = r * r;
                let d_f64 = d as f64;
                (
                    (lap_psi - d_f64 * ratio_psi) / r2,
                    (lap_psi_psi - d_f64 * ratio_psi_psi) / r2,
                )
            };
            d0_raw[[k, j]] = phi;
            d0_raw_psi[[k, j]] = phi_psi;
            d0_raw_psi_psi[[k, j]] = phi_psi_psi;
            for axis in 0..d {
                let delta = centers[[k, axis]] - centers[[j, axis]];
                let axis_scale = metric_weights[axis];
                let row = k * d + axis;
                d1_raw[[row, j]] = ratio * axis_scale * delta;
                d1_raw_psi[[row, j]] = ratio_psi * axis_scale * delta;
                d1_raw_psi_psi[[row, j]] = ratio_psi_psi * axis_scale * delta;
            }
            for b in 0..d {
                let h_b = centers[[k, b]] - centers[[j, b]];
                let w_b = metric_weights[b];
                for c in 0..d {
                    let h_c = centers[[k, c]] - centers[[j, c]];
                    let w_c = metric_weights[c];
                    let row = (k * d + b) * d + c;
                    d2_raw[[row, j]] = hessian_operator_entry(q, t, h_b, h_c, w_b, w_c, b, c);
                    d2_raw_psi[[row, j]] =
                        hessian_operator_entry(q_psi, t_psi, h_b, h_c, w_b, w_c, b, c);
                    d2_raw_psi_psi[[row, j]] =
                        hessian_operator_entry(q_psi_psi, t_psi_psi, h_b, h_c, w_b, w_c, b, c);
                }
            }
        }
    }

    let coefficient_gauge =
        z_opt.map(|z| crate::solver::gauge::Gauge::from_block_transforms(&[z.clone()]));
    let project = |mat: Array2<f64>| {
        if let Some(gauge) = coefficient_gauge.as_ref() {
            gauge.restrict_design(&mat)
        } else {
            mat
        }
    };
    // With psi-independent Z this is algebraically exact:
    //   S_con = Z^T (D^T D) Z = (DZ)^T (DZ),
    // and identically for S_con', S_con'' using D'Z, D''Z.
    // So we can project operators first, then build Gram derivatives.
    let d0_kernel = project(d0_raw);
    let d0_kernel_psi = project(d0_raw_psi);
    let d0_kernel_psi_psi = project(d0_raw_psi_psi);
    let d1_kernel = project(d1_raw);
    let d1_kernel_psi = project(d1_raw_psi);
    let d1_kernel_psi_psi = project(d1_raw_psi_psi);
    let d2_kernel = project(d2_raw);
    let d2_kernel_psi = project(d2_raw_psi);
    let d2_kernel_psi_psi = project(d2_raw_psi_psi);

    let kernel_cols = d0_kernel.ncols();
    let total_cols = kernel_cols + usize::from(include_intercept);
    let mut d0 = Array2::<f64>::zeros((p, total_cols));
    let mut d1 = Array2::<f64>::zeros((p * d, total_cols));
    let mut d2 = Array2::<f64>::zeros((p * d * d, total_cols));
    let mut d0_psi = Array2::<f64>::zeros((p, total_cols));
    let mut d1_psi = Array2::<f64>::zeros((p * d, total_cols));
    let mut d2_psi = Array2::<f64>::zeros((p * d * d, total_cols));
    let mut d0_psi_psi = Array2::<f64>::zeros((p, total_cols));
    let mut d1_psi_psi = Array2::<f64>::zeros((p * d, total_cols));
    let mut d2_psi_psi = Array2::<f64>::zeros((p * d * d, total_cols));
    d0.slice_mut(s![.., 0..kernel_cols]).assign(&d0_kernel);
    d1.slice_mut(s![.., 0..kernel_cols]).assign(&d1_kernel);
    d2.slice_mut(s![.., 0..kernel_cols]).assign(&d2_kernel);
    d0_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d0_kernel_psi);
    d1_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d1_kernel_psi);
    d2_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d2_kernel_psi);
    d0_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d0_kernel_psi_psi);
    d1_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d1_kernel_psi_psi);
    d2_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d2_kernel_psi_psi);
    if include_intercept {
        d0.column_mut(kernel_cols).fill(1.0);
    }

    // The forward Matérn operator-penalty path
    // (`operator_penalty_candidates_from_collocation`) builds the Mass block
    // as the RAW (un-centered) Gram `S0 = D0ᵀ D0`; only the Duchon runtime
    // overlay centers the mass rows. The analytic ∂S/∂ψ must mirror the forward
    // construction exactly, so the S0 derivative also uses the un-centered Gram
    // — centering here desynced the derivative from the forward penalty
    // (FD-mismatch ~3e-2, #839).
    let (s0, s0_psi, s0_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d0, &d0_psi, &d0_psi_psi);
    let (s1, s1_psi, s1_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d1, &d1_psi, &d1_psi_psi);
    let (s2, s2_psi, s2_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d2, &d2_psi, &d2_psi_psi);

    let (s0_norm, s0_norm_psi, s0_norm_psi_psi, c0) =
        normalize_penaltywith_psi_derivatives(&s0, &s0_psi, &s0_psi_psi);
    let (s1_norm, s1_norm_psi, s1_norm_psi_psi, c1) =
        normalize_penaltywith_psi_derivatives(&s1, &s1_psi, &s1_psi_psi);
    let (s2_norm, s2_norm_psi, s2_norm_psi_psi, c2) =
        normalize_penaltywith_psi_derivatives(&s2, &s2_psi, &s2_psi_psi);
    // Gate the operator dials on the Matérn-ν RKHS smoothness EXACTLY as the
    // forward builder `build_matern_operator_penalty_candidates` does (via
    // `operator_penalty_candidates_from_collocation` /
    // `matern_for_smoothness(nu, d)`): a rough kernel (e.g. ν=1/2, d=1) emits
    // only the admitted operator penalties, so the candidate list — and hence
    // its ψ-derivative list below — stays index-aligned with the forward penalty
    // construction. Omitting the gate here let a rough-ν, non-double-penalty
    // Matérn produce ψ-derivatives for tension/stiffness penalties the forward
    // path never built, desyncing the κ-gradient against a mismatched penalty
    // set (gam#902).
    let matern_spec = DuchonOperatorPenaltySpec::matern_for_smoothness(nu, d);
    let mut candidates = Vec::with_capacity(3);
    for (spec_gate, source, matrix, normalization_scale) in [
        (&matern_spec.mass, PenaltySource::OperatorMass, s0_norm, c0),
        (
            &matern_spec.tension,
            PenaltySource::OperatorTension,
            s1_norm,
            c1,
        ),
        (
            &matern_spec.stiffness,
            PenaltySource::OperatorStiffness,
            s2_norm,
            c2,
        ),
    ] {
        if !matches!(spec_gate, OperatorPenaltySpec::Active { .. }) {
            continue;
        }
        candidates.push(PenaltyCandidate {
            matrix,
            nullspace_dim_hint: 0,
            source,
            normalization_scale,
            kronecker_factors: None,
            op: None,
        });
    }
    // `active_operator_penalty_derivatives` selects the κ-derivative for each
    // SURVIVING penalty by its `source` kind out of the canonical
    // `[mass, tension, stiffness]` triple, so a gated-out (or rank-0-dropped)
    // operator is simply never requested and the returned derivative list stays
    // index-aligned with the forward penalty list.
    let (_, _, penaltyinfo) = filter_active_penalty_candidates(candidates)?;
    let penalties_derivative = active_operator_penalty_derivatives(
        &penaltyinfo,
        &[s0_norm_psi, s1_norm_psi, s2_norm_psi],
        "Matérn",
    )?;
    let penaltiessecond_derivative = active_operator_penalty_derivatives(
        &penaltyinfo,
        &[s0_norm_psi_psi, s1_norm_psi_psi, s2_norm_psi_psi],
        "Matérn",
    )?;
    Ok((penalties_derivative, penaltiessecond_derivative))
}

pub(crate) struct DuchonRawPenaltyPsiDerivativeBlocks {
    pub(crate) d0: Array2<f64>,
    pub(crate) d1: Array2<f64>,
    pub(crate) d2: Array2<f64>,
    pub(crate) d0_psi: Array2<f64>,
    pub(crate) d1_psi: Array2<f64>,
    pub(crate) d2_psi: Array2<f64>,
    pub(crate) d0_psi_psi: Array2<f64>,
    pub(crate) d1_psi_psi: Array2<f64>,
    pub(crate) d2_psi_psi: Array2<f64>,
}

impl DuchonRawPenaltyPsiDerivativeBlocks {
    pub(crate) fn zeros(p: usize, d: usize, cols: usize) -> Self {
        Self {
            d0: Array2::<f64>::zeros((p, cols)),
            d1: Array2::<f64>::zeros((p * d, cols)),
            d2: Array2::<f64>::zeros((p * d * d, cols)),
            d0_psi: Array2::<f64>::zeros((p, cols)),
            d1_psi: Array2::<f64>::zeros((p * d, cols)),
            d2_psi: Array2::<f64>::zeros((p * d * d, cols)),
            d0_psi_psi: Array2::<f64>::zeros((p, cols)),
            d1_psi_psi: Array2::<f64>::zeros((p * d, cols)),
            d2_psi_psi: Array2::<f64>::zeros((p * d * d, cols)),
        }
    }

    pub(crate) fn add_assign(&mut self, rhs: &Self) {
        self.d0 += &rhs.d0;
        self.d1 += &rhs.d1;
        self.d2 += &rhs.d2;
        self.d0_psi += &rhs.d0_psi;
        self.d1_psi += &rhs.d1_psi;
        self.d2_psi += &rhs.d2_psi;
        self.d0_psi_psi += &rhs.d0_psi_psi;
        self.d1_psi_psi += &rhs.d1_psi_psi;
        self.d2_psi_psi += &rhs.d2_psi_psi;
    }
}

pub(crate) fn build_duchon_operator_penalty_psi_derivatives(
    collocation_points: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<(Vec<PenaltySource>, Vec<Array2<f64>>, Vec<Array2<f64>>), BasisError> {
    let length_scale = spec.length_scale.ok_or_else(|| {
        BasisError::InvalidInput(
            "exact Duchon log-kappa derivatives require hybrid Duchon with length_scale"
                .to_string(),
        )
    })?;
    let effective_nullspace_order = duchon_effective_nullspace_order(centers, spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    let s_order = spec.power_as_usize();
    let dim = centers.ncols();
    let two_pps = 2.0 * (p_order as f64 + spec.power);
    let mut effective_operator_penalties = spec.operator_penalties.clone();
    if two_pps <= dim as f64 + 1.0 {
        effective_operator_penalties.tension = OperatorPenaltySpec::Disabled;
    }
    if two_pps <= dim as f64 + 2.0 {
        effective_operator_penalties.stiffness = OperatorPenaltySpec::Disabled;
    }
    let max_derivative_order =
        duchon_max_active_operator_derivative_order(&effective_operator_penalties);
    if max_derivative_order == 0
        && !matches!(
            effective_operator_penalties.mass,
            OperatorPenaltySpec::Active { .. }
        )
    {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    validate_duchon_collocation_orders(
        Some(length_scale),
        p_order,
        s_order as f64,
        dim,
        max_derivative_order,
    )?;
    // Hybrid Matérn partial-fraction expansion requires integer s; the
    // assertion fires here rather than at the spec layer so the
    // scale-free path stays fractional-clean.
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let z_kernel =
        kernel_constraint_nullspace(centers, effective_nullspace_order, &mut workspace.cache)?;
    let n_basis = centers.nrows();
    if collocation_points.ncols() != dim {
        crate::bail_dim_basis!(
            "Duchon psi-derivative collocation dim {} != centers dim {dim}",
            collocation_points.ncols()
        );
    }
    let p_colloc = collocation_points.nrows();
    let d = dim;
    let kernel_cols = z_kernel.ncols();

    let aniso = spec.aniso_log_scales.as_deref();
    if let Some(eta) = aniso
        && eta.len() != d
    {
        crate::bail_dim_basis!(
            "Duchon anisotropy dimension mismatch: got {}, expected {d}",
            eta.len()
        );
    }
    let metric_weights: Option<Vec<f64>> = aniso.map(centered_aniso_metric_weights);
    // Only assemble derivative-order blocks that the *enabled* operator
    // penalties actually consume. `max_derivative_order` is computed from
    // `effective_operator_penalties`, which has tension/stiffness already
    // disabled when the kernel is too rough to admit them (the
    // `two_pps <= dim + k` guards above). Computing higher-order blocks
    // anyway is not just wasted work — the d2 collision branch calls
    // `duchonphi_rr_collision_psi_triplet`, which requires
    // `2(p+s) > dim + 2` and aborts the whole fit when the upstream
    // auto-disable has correctly recognized the boundary case. Gating the
    // accumulators here keeps the contract between the operator-spec
    // validator and the per-pair worker consistent.
    let need_d1 = max_derivative_order >= 1;
    let need_d2 = max_derivative_order >= 2;
    let chunk_count = rayon::current_num_threads().max(1);
    let chunk_size = p_colloc.div_ceil(chunk_count).max(1);
    let chunks: Vec<(usize, usize)> = (0..p_colloc)
        .step_by(chunk_size)
        .map(|start| (start, (start + chunk_size).min(p_colloc)))
        .collect();
    let partial_blocks = chunks
        .into_par_iter()
        .map(
            |(start, end)| -> Result<DuchonRawPenaltyPsiDerivativeBlocks, BasisError> {
                let mut local =
                    DuchonRawPenaltyPsiDerivativeBlocks::zeros(p_colloc, d, kernel_cols);
                for i in start..end {
                    for j in 0..n_basis {
                        let r = if let Some(eta) = aniso {
                            let row_i: Vec<f64> =
                                (0..d).map(|a| collocation_points[[i, a]]).collect();
                            let row_j: Vec<f64> = (0..d).map(|a| centers[[j, a]]).collect();
                            let (r, _) = aniso_distance_and_components(&row_i, &row_j, eta);
                            r
                        } else {
                            stable_euclidean_norm(
                                (0..d)
                                    .map(|axis| collocation_points[[i, axis]] - centers[[j, axis]]),
                            )
                        };
                        let core = duchon_radial_core_psi_triplet(
                            r,
                            length_scale,
                            p_order,
                            s_order,
                            d,
                            &coeffs,
                        )?;
                        for col in 0..kernel_cols {
                            let z_jc = z_kernel[[j, col]];
                            local.d0[[i, col]] += core.phi.value * z_jc;
                            local.d0_psi[[i, col]] += core.phi.psi * z_jc;
                            local.d0_psi_psi[[i, col]] += core.phi.psi_psi * z_jc;
                        }
                        if !need_d1 && !need_d2 {
                            continue;
                        }
                        if r > 1e-10 {
                            let jets =
                                duchon_radial_jets(r, length_scale, p_order, s_order, d, &coeffs)?;
                            let q = jets.q;
                            let (q_psi, q_psi_psi) =
                                duchon_q_psi_triplet_from_jets(&jets, p_order, s_order, d, r);
                            let t_exponent = duchon_scaling_exponent(p_order, s_order, d) + 4.0;
                            let (t_psi, t_psi_psi) = scaled_log_kappa_derivatives(
                                jets.t, jets.t_r, jets.t_rr, t_exponent, r,
                            );
                            if need_d1 {
                                for axis in 0..d {
                                    let delta = collocation_points[[i, axis]] - centers[[j, axis]];
                                    let axis_scale = metric_weights
                                        .as_ref()
                                        .map(|weights| weights[axis])
                                        .unwrap_or(1.0);
                                    let row = i * d + axis;
                                    for col in 0..kernel_cols {
                                        let z_jc = z_kernel[[j, col]];
                                        local.d1[[row, col]] += q * axis_scale * delta * z_jc;
                                        local.d1_psi[[row, col]] +=
                                            q_psi * axis_scale * delta * z_jc;
                                        local.d1_psi_psi[[row, col]] +=
                                            q_psi_psi * axis_scale * delta * z_jc;
                                    }
                                }
                            }
                            if need_d2 {
                                for col in 0..kernel_cols {
                                    let z_jc = z_kernel[[j, col]];
                                    for axis_b in 0..d {
                                        let h_b =
                                            collocation_points[[i, axis_b]] - centers[[j, axis_b]];
                                        let w_b = metric_weights
                                            .as_ref()
                                            .map(|weights| weights[axis_b])
                                            .unwrap_or(1.0);
                                        for axis_c in 0..d {
                                            let h_c = collocation_points[[i, axis_c]]
                                                - centers[[j, axis_c]];
                                            let w_c = metric_weights
                                                .as_ref()
                                                .map(|weights| weights[axis_c])
                                                .unwrap_or(1.0);
                                            let row = (i * d + axis_b) * d + axis_c;
                                            local.d2[[row, col]] += hessian_operator_entry(
                                                q, jets.t, h_b, h_c, w_b, w_c, axis_b, axis_c,
                                            ) * z_jc;
                                            local.d2_psi[[row, col]] += hessian_operator_entry(
                                                q_psi, t_psi, h_b, h_c, w_b, w_c, axis_b, axis_c,
                                            ) * z_jc;
                                            local.d2_psi_psi[[row, col]] += hessian_operator_entry(
                                                q_psi_psi, t_psi_psi, h_b, h_c, w_b, w_c, axis_b,
                                                axis_c,
                                            ) * z_jc;
                                        }
                                    }
                                }
                            }
                        } else if need_d2 {
                            let (phi_rr, phi_rr_psi, phi_rr_psi_psi) =
                                duchonphi_rr_collision_psi_triplet(
                                    length_scale,
                                    p_order,
                                    s_order,
                                    d,
                                    &coeffs,
                                )?;
                            for col in 0..kernel_cols {
                                let z_jc = z_kernel[[j, col]];
                                for axis in 0..d {
                                    let w_axis = metric_weights
                                        .as_ref()
                                        .map(|weights| weights[axis])
                                        .unwrap_or(1.0);
                                    let row = (i * d + axis) * d + axis;
                                    local.d2[[row, col]] += w_axis * phi_rr * z_jc;
                                    local.d2_psi[[row, col]] += w_axis * phi_rr_psi * z_jc;
                                    local.d2_psi_psi[[row, col]] += w_axis * phi_rr_psi_psi * z_jc;
                                }
                            }
                        }
                    }
                }
                Ok(local)
            },
        )
        .collect::<Result<Vec<_>, BasisError>>()?;
    let mut raw = DuchonRawPenaltyPsiDerivativeBlocks::zeros(p_colloc, d, kernel_cols);
    for partial in &partial_blocks {
        raw.add_assign(partial);
    }
    let d0_raw = raw.d0;
    let d1_raw = raw.d1;
    let d2_raw = raw.d2;
    let d0_raw_psi = raw.d0_psi;
    let d1_raw_psi = raw.d1_psi;
    let d2_raw_psi = raw.d2_psi;
    let d0_raw_psi_psi = raw.d0_psi_psi;
    let d1_raw_psi_psi = raw.d1_psi_psi;
    let d2_raw_psi_psi = raw.d2_psi_psi;

    let poly = polynomial_block_from_order(centers, effective_nullspace_order);
    let kernel_cols = d0_raw.ncols();
    let poly_cols = poly.ncols();
    let total_cols = kernel_cols + poly_cols;
    let mut d0 = Array2::<f64>::zeros((p_colloc, total_cols));
    let mut d1 = Array2::<f64>::zeros((p_colloc * d, total_cols));
    let mut d2 = Array2::<f64>::zeros((p_colloc * d * d, total_cols));
    let mut d0_psi = Array2::<f64>::zeros((p_colloc, total_cols));
    let mut d1_psi = Array2::<f64>::zeros((p_colloc * d, total_cols));
    let mut d2_psi = Array2::<f64>::zeros((p_colloc * d * d, total_cols));
    let mut d0_psi_psi = Array2::<f64>::zeros((p_colloc, total_cols));
    let mut d1_psi_psi = Array2::<f64>::zeros((p_colloc * d, total_cols));
    let mut d2_psi_psi = Array2::<f64>::zeros((p_colloc * d * d, total_cols));
    d0.slice_mut(s![.., 0..kernel_cols]).assign(&d0_raw);
    d1.slice_mut(s![.., 0..kernel_cols]).assign(&d1_raw);
    d2.slice_mut(s![.., 0..kernel_cols]).assign(&d2_raw);
    d0_psi.slice_mut(s![.., 0..kernel_cols]).assign(&d0_raw_psi);
    d1_psi.slice_mut(s![.., 0..kernel_cols]).assign(&d1_raw_psi);
    d2_psi.slice_mut(s![.., 0..kernel_cols]).assign(&d2_raw_psi);
    d0_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d0_raw_psi_psi);
    d1_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d1_raw_psi_psi);
    d2_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&d2_raw_psi_psi);
    // The polynomial block is the Duchon nullspace. Keep it in the total
    // coordinate system so identifiability transforms line up, but leave its
    // operator columns and psi-derivatives at zero so it remains unpenalized.

    let coefficient_gauge = identifiability_transform
        .map(|z| crate::solver::gauge::Gauge::from_block_transforms(&[z.clone()]));
    let project = |mat: Array2<f64>| {
        if let Some(gauge) = coefficient_gauge.as_ref() {
            gauge.restrict_design(&mat)
        } else {
            mat
        }
    };
    let d0 = project(d0);
    let d1 = project(d1);
    let d2 = project(d2);
    let d0_psi = project(d0_psi);
    let d1_psi = project(d1_psi);
    let d2_psi = project(d2_psi);
    let d0_psi_psi = project(d0_psi_psi);
    let d1_psi_psi = project(d1_psi_psi);
    let d2_psi_psi = project(d2_psi_psi);

    let (s0, s0_psi, s0_psi_psi) =
        centered_operator_gram_and_psi_derivatives(&d0, &d0_psi, &d0_psi_psi);
    let (mut s1, mut s1_psi, mut s1_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d1, &d1_psi, &d1_psi_psi);
    let (mut s2, mut s2_psi, mut s2_psi_psi) =
        gram_and_psi_derivatives_from_operator(&d2, &d2_psi, &d2_psi_psi);

    // Match the value-side Duchon penalty exactly. q=0 mass remains the
    // collocation Gram; q∈{1,2} uses the continuous closed-form Lebesgue
    // penalty whenever the UV+IR+precondition predicate holds, independent of
    // the polynomial nullspace order. Polynomial columns are zero-padded in
    // the closed-form block because they are the unpenalized Duchon nullspace.
    let kappa = 1.0 / length_scale.max(1e-300);
    let aniso = spec.aniso_log_scales.as_deref();
    if duchon_closed_form_operator_penalty_converges(1, p_order, s_order as f64, d) {
        let (cf_s, cf_s_psi, cf_s_psi_psi) = closed_form_psi_derivatives_in_total_basis(
            centers,
            1,
            p_order,
            s_order,
            kappa,
            aniso,
            Some(&z_kernel),
            poly_cols,
            identifiability_transform,
        );
        s1 = cf_s;
        s1_psi = cf_s_psi;
        s1_psi_psi = cf_s_psi_psi;
    }
    if duchon_closed_form_operator_penalty_converges(2, p_order, s_order as f64, d) {
        let (cf_s, cf_s_psi, cf_s_psi_psi) = closed_form_psi_derivatives_in_total_basis(
            centers,
            2,
            p_order,
            s_order,
            kappa,
            aniso,
            Some(&z_kernel),
            poly_cols,
            identifiability_transform,
        );
        s2 = cf_s;
        s2_psi = cf_s_psi;
        s2_psi_psi = cf_s_psi_psi;
    }

    let (s0_norm, s0_norm_psi, s0_norm_psi_psi, c0) =
        normalize_penaltywith_psi_derivatives(&s0, &s0_psi, &s0_psi_psi);
    let (s1_norm, s1_norm_psi, s1_norm_psi_psi, c1) =
        normalize_penaltywith_psi_derivatives(&s1, &s1_psi, &s1_psi_psi);
    let (s2_norm, s2_norm_psi, s2_norm_psi_psi, c2) =
        normalize_penaltywith_psi_derivatives(&s2, &s2_psi, &s2_psi_psi);

    let candidates = vec![
        PenaltyCandidate {
            matrix: s0_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
            kronecker_factors: None,
            op: None,
        },
        PenaltyCandidate {
            matrix: s1_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
            kronecker_factors: None,
            op: None,
        },
        PenaltyCandidate {
            matrix: s2_norm,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
            kronecker_factors: None,
            op: None,
        },
    ];
    let candidates = operator_penalty_candidates_from_derivative_candidates(
        candidates,
        &effective_operator_penalties,
    );

    let first_derivs = vec![s0_norm_psi, s1_norm_psi, s2_norm_psi];
    let second_derivs = vec![s0_norm_psi_psi, s1_norm_psi_psi, s2_norm_psi_psi];

    let (_, _, penaltyinfo) = filter_active_penalty_candidates(candidates)?;
    let active_sources = penaltyinfo
        .iter()
        .filter(|info| info.active)
        .map(|info| info.source.clone())
        .collect::<Vec<_>>();
    let penalties_derivative =
        active_operator_penalty_derivatives(&penaltyinfo, &first_derivs, "Duchon")?;
    let penaltiessecond_derivative =
        active_operator_penalty_derivatives(&penaltyinfo, &second_derivs, "Duchon")?;
    Ok((
        active_sources,
        penalties_derivative,
        penaltiessecond_derivative,
    ))
}

pub(crate) fn operator_penalty_candidates_from_derivative_candidates(
    candidates: Vec<PenaltyCandidate>,
    spec: &DuchonOperatorPenaltySpec,
) -> Vec<PenaltyCandidate> {
    candidates
        .into_iter()
        .filter(|candidate| match candidate.source {
            PenaltySource::OperatorMass => matches!(spec.mass, OperatorPenaltySpec::Active { .. }),
            PenaltySource::OperatorTension => {
                matches!(spec.tension, OperatorPenaltySpec::Active { .. })
            }
            PenaltySource::OperatorStiffness => {
                matches!(spec.stiffness, OperatorPenaltySpec::Active { .. })
            }
            _ => true,
        })
        .collect()
}

pub(crate) fn build_duchon_native_penalty_psi_derivatives(
    centers: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<(Vec<PenaltySource>, Vec<Array2<f64>>, Vec<Array2<f64>>), BasisError> {
    let length_scale = spec.length_scale.ok_or_else(|| {
        BasisError::InvalidInput(
            "exact Duchon native penalty log-kappa derivatives require hybrid Duchon with length_scale"
                .to_string(),
        )
    })?;
    let effective_nullspace_order = duchon_effective_nullspace_order(centers, spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    let s_order = spec.power_as_usize();
    let dim = centers.ncols();
    let z = kernel_constraint_nullspace(centers, effective_nullspace_order, &mut workspace.cache)?;
    let kernel_cols = z.ncols();
    let poly_cols = polynomial_block_from_order(centers, effective_nullspace_order).ncols();
    let total_cols = kernel_cols + poly_cols;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale.max(1e-300));
    let kernel_amp = duchon_kernel_amplification(
        centers,
        Some(length_scale),
        p_order,
        s_order,
        dim,
        spec.aniso_log_scales.as_deref(),
        Some(&coeffs),
        None,
    );
    let axis_scales = spec.aniso_log_scales.as_deref().map(aniso_axis_scales);
    let n_centers = centers.nrows();
    let mut kernel = Array2::<f64>::zeros((n_centers, n_centers));
    let mut kernel_psi = Array2::<f64>::zeros((n_centers, n_centers));
    let mut kernel_psi_psi = Array2::<f64>::zeros((n_centers, n_centers));
    for i in 0..n_centers {
        for j in i..n_centers {
            let r = if let Some(scales) = axis_scales.as_deref() {
                aniso_distance_rows_with_scales(centers, i, centers, j, scales)
            } else {
                euclidean_distance_rows(centers, i, centers, j)
            };
            let core =
                duchon_radial_core_psi_triplet(r, length_scale, p_order, s_order, dim, &coeffs)?;
            kernel[[i, j]] = core.phi.value;
            kernel[[j, i]] = core.phi.value;
            kernel_psi[[i, j]] = core.phi.psi;
            kernel_psi[[j, i]] = core.phi.psi;
            kernel_psi_psi[[i, j]] = core.phi.psi_psi;
            kernel_psi_psi[[j, i]] = core.phi.psi_psi;
        }
    }

    let amp2 = kernel_amp * kernel_amp;
    let kernel_gauge = crate::solver::gauge::Gauge::from_block_transforms(&[z.clone()]);
    let project_kernel = |k: &Array2<f64>| kernel_gauge.restrict_penalty(k).mapv(|v| v * amp2);
    let omega = project_kernel(&kernel);
    let omega_psi = project_kernel(&kernel_psi);
    let omega_psi_psi = project_kernel(&kernel_psi_psi);

    let outer_gauge = identifiability_transform
        .map(|z| crate::solver::gauge::Gauge::from_block_transforms(&[z.clone()]));
    let embed = |block: Array2<f64>| {
        let mut out = Array2::<f64>::zeros((total_cols, total_cols));
        out.slice_mut(s![..kernel_cols, ..kernel_cols])
            .assign(&block);
        match outer_gauge.as_ref() {
            Some(gauge) => symmetrize(&gauge.restrict_penalty(&out)),
            None => symmetrize(&out),
        }
    };
    let primary = embed(omega);
    let primary_psi = embed(omega_psi);
    let primary_psi_psi = embed(omega_psi_psi);
    let (_, primary_psi_norm, primary_psi_psi_norm, _) =
        normalize_penaltywith_psi_derivatives(&primary, &primary_psi, &primary_psi_psi);
    let candidates = duchon_native_penalty_candidates(
        centers,
        spec.length_scale,
        spec.power,
        effective_nullspace_order,
        spec.aniso_log_scales.as_deref(),
        &z,
        identifiability_transform,
        poly_cols,
    )?;
    let (_, _, penaltyinfo) = filter_active_penalty_candidates(candidates)?;
    let mut sources = Vec::new();
    let mut first = Vec::new();
    let mut second = Vec::new();
    for info in penaltyinfo.iter().filter(|info| info.active) {
        sources.push(info.source.clone());
        match info.source {
            PenaltySource::Primary => {
                first.push(primary_psi_norm.clone());
                second.push(primary_psi_psi_norm.clone());
            }
            PenaltySource::DoublePenaltyNullspace => {
                first.push(Array2::<f64>::zeros(primary_psi_norm.raw_dim()));
                second.push(Array2::<f64>::zeros(primary_psi_psi_norm.raw_dim()));
            }
            ref other => {
                crate::bail_invalid_basis!(
                    "unexpected Duchon native penalty source in derivative path: {other:?}"
                );
            }
        }
    }
    Ok((sources, first, second))
}

pub(crate) fn prepare_duchon_derivative_contextwithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<(Array2<f64>, Option<Array2<f64>>), BasisError> {
    let original_centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let centers = expand_periodic_centers(&original_centers, spec.periodic.as_deref())?;
    assert_spatial_centers_below_large_scale_cap(data.ncols(), centers.view())?;
    let raw_design = build_duchon_basis_designwithworkspace(
        data,
        centers.view(),
        spec.length_scale,
        spec.power,
        spec.nullspace_order,
        spec.aniso_log_scales.as_deref(),
        workspace,
    )?;
    let identifiability_transform = spatial_identifiability_transform_from_design(
        data,
        raw_design.basis.view(),
        &spec.identifiability,
        "Duchon",
    )?;
    Ok((centers, identifiability_transform))
}

/// Validate a 1D periodic Duchon center matrix, compute the circular
/// domain ``[left, left + period]``, and drop any centers that are
/// periodically equivalent to ``left`` past the first occurrence.
///
/// All periodic Duchon code paths (`build_periodic_duchon_basis_1d`,
/// `build_periodic_duchon_basis_log_kappa_derivatives…`) must use the
/// *same* collapsed centers, otherwise the kernel-column count diverges
/// between the design build and its log-κ derivative — producing
/// `ShapeError` mismatches at the consumer (e.g. the finite-difference
/// regression test against the analytic derivative). Centralising the
/// collapse in one helper makes it impossible to add a new periodic path
/// that forgets the dedup.
pub(crate) fn prepare_periodic_duchon_centers_1d(
    centers: Array2<f64>,
) -> Result<(Array2<f64>, f64, f64), BasisError> {
    prepare_periodic_duchon_centers_1d_with_period(centers, None)
}

/// Variant of [`prepare_periodic_duchon_centers_1d`] that honors an explicit
/// domain-wrap `period`.
///
/// The period is the circumference of the circle the smooth lives on, NOT the
/// span of the supplied centers. On a half-open lattice — e.g.
/// `linspace(0, 1, K, endpoint=false)` with `period = 1.0` — the centers cover
/// only `period − one_spacing`, so deriving the period from the center span
/// (`right − left`) undershoots the true wrap. That undersized period made the
/// Bernoulli Green's-function kernel evaluate at the wrong argument and the
/// resulting Gram was no longer the operator's true reproducing-norm matrix
/// (gam#580). When `explicit_period` is `Some(P)` we use `P` as the wrap and
/// require every center to fit inside one period (`span ≤ P`); when `None` we
/// fall back to the legacy center-span period (the closed lattice the formula
/// DSL builds, where the endpoints span exactly one period).
pub(crate) fn prepare_periodic_duchon_centers_1d_with_period(
    centers: Array2<f64>,
    explicit_period: Option<f64>,
) -> Result<(Array2<f64>, f64, f64), BasisError> {
    if centers.ncols() != 1 {
        crate::bail_invalid_basis!(
            "periodic Duchon smooths currently require exactly one covariate"
        );
    }
    let left = centers
        .column(0)
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let right = centers
        .column(0)
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if !left.is_finite() || !right.is_finite() || left >= right {
        return Err(BasisError::InvalidRange(left, right));
    }
    let span = right - left;
    let period = match explicit_period {
        Some(p) => {
            if !p.is_finite() || p <= 0.0 {
                crate::bail_invalid_basis!(
                    "periodic Duchon period must be finite and positive; got {p}"
                );
            }
            // Every center must lie inside a single period; the wrap may be
            // larger than the center span (half-open lattice) but never
            // smaller (that would fold distinct centers onto each other).
            if p < span - 1.0e-10 * span.max(1.0) {
                crate::bail_invalid_basis!(
                    "periodic Duchon period ({p}) is smaller than the center span ({span}); \
                     every center must lie within a single period"
                );
            }
            p
        }
        None => span,
    };
    let centers = collapse_periodic_endpoint(centers, left, period);
    Ok((centers, left, period))
}

pub(crate) fn fill_periodic_duchon_kernel_psi_matrices(
    rows: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    left: f64,
    period: f64,
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), BasisError> {
    let n = rows.nrows();
    let k = centers.nrows();
    let mut kernel = Array2::<f64>::zeros((n, k));
    let mut kernel_psi = Array2::<f64>::zeros((n, k));
    let mut kernel_psi_psi = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        let x = wrap_to_period(rows[[i, 0]], left, period);
        for j in 0..k {
            let r = periodic_distance_1d(x, centers[[j, 0]], period);
            let core =
                duchon_radial_core_psi_triplet(r, length_scale, p_order, s_order, 1, coeffs)?;
            kernel[[i, j]] = core.phi.value;
            kernel_psi[[i, j]] = core.phi.psi;
            kernel_psi_psi[[i, j]] = core.phi.psi_psi;
        }
    }
    Ok((kernel, kernel_psi, kernel_psi_psi))
}

pub(crate) fn periodic_duchon_identifiability_transformwithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    centers: Array2<f64>,
    workspace: &mut BasisWorkspace,
) -> Result<Option<Array2<f64>>, BasisError> {
    let built = build_periodic_duchon_basis_1d(data, spec, centers, workspace)?;
    match built.metadata {
        BasisMetadata::Duchon {
            identifiability_transform,
            ..
        } => Ok(identifiability_transform),
        other => Err(BasisError::InvalidInput(format!(
            "periodic Duchon builder must return Duchon metadata, got {:?}",
            std::mem::discriminant(&other)
        ))),
    }
}

pub(crate) fn build_periodic_duchon_basis_log_kappa_derivativeswithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    if data.ncols() != 1 {
        crate::bail_invalid_basis!(
            "periodic Duchon log-kappa derivatives require exactly one covariate"
        );
    }
    let length_scale = spec.length_scale.ok_or_else(|| {
        BasisError::InvalidInput(
            "periodic Duchon log-kappa derivatives require hybrid Duchon with length_scale"
                .to_string(),
        )
    })?;
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    assert_spatial_centers_below_large_scale_cap(data.ncols(), centers.view())?;
    let (centers, left, period) = prepare_periodic_duchon_centers_1d(centers)?;
    let effective_nullspace_order = DuchonNullspaceOrder::Zero;
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    let s_order = spec.power_as_usize();
    // Validate against the INTEGER `s` the hybrid kernel actually evaluates
    // (`power_as_usize` truncates a fractional `spec.power` to the integer the
    // partial-fraction expansion uses). Validating the raw fractional power
    // would desync the well-posedness gate from the realized kernel.
    validate_duchon_kernel_orders(Some(length_scale), p_order, s_order as f64, 1)?;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale.max(1e-300));
    let z_kernel = kernel_constraint_nullspace(
        centers.view(),
        effective_nullspace_order,
        &mut workspace.cache,
    )?;
    let identifiability_transform = periodic_duchon_identifiability_transformwithworkspace(
        data,
        spec,
        centers.clone(),
        workspace,
    )?;

    let (_data_kernel, data_kernel_psi, data_kernel_psi_psi) =
        fill_periodic_duchon_kernel_psi_matrices(
            data,
            centers.view(),
            left,
            period,
            length_scale,
            p_order,
            s_order,
            &coeffs,
        )?;
    let kernel_amp = duchon_kernel_amplification(
        centers.view(),
        Some(length_scale),
        p_order,
        s_order,
        1,
        None,
        Some(&coeffs),
        None,
    );
    let kernel_cols = z_kernel.ncols();
    let total_cols = kernel_cols + 1;
    let kernel_gauge = crate::solver::gauge::Gauge::from_block_transforms(&[z_kernel.clone()]);
    let mut design_first = Array2::<f64>::zeros((data.nrows(), total_cols));
    let mut design_second = Array2::<f64>::zeros((data.nrows(), total_cols));
    design_first
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&(kernel_gauge.restrict_design(&data_kernel_psi) * kernel_amp));
    design_second
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&(kernel_gauge.restrict_design(&data_kernel_psi_psi) * kernel_amp));
    if let Some(gauge) = identifiability_transform
        .as_ref()
        .map(|z| crate::solver::gauge::Gauge::from_block_transforms(&[z.clone()]))
    {
        design_first = gauge.restrict_design(&design_first);
        design_second = gauge.restrict_design(&design_second);
    }

    let (center_kernel, center_kernel_psi, center_kernel_psi_psi) =
        fill_periodic_duchon_kernel_psi_matrices(
            centers.view(),
            centers.view(),
            left,
            period,
            length_scale,
            p_order,
            s_order,
            &coeffs,
        )?;
    let omega = kernel_gauge.restrict_penalty(&center_kernel);
    let omega_psi = kernel_gauge.restrict_penalty(&center_kernel_psi);
    let omega_psi_psi = kernel_gauge.restrict_penalty(&center_kernel_psi_psi);
    let mut penalty = Array2::<f64>::zeros((total_cols, total_cols));
    let mut penalty_psi = Array2::<f64>::zeros((total_cols, total_cols));
    let mut penalty_psi_psi = Array2::<f64>::zeros((total_cols, total_cols));
    penalty
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&omega);
    penalty_psi
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&omega_psi);
    penalty_psi_psi
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&omega_psi_psi);
    if let Some(gauge) = identifiability_transform
        .as_ref()
        .map(|z| crate::solver::gauge::Gauge::from_block_transforms(&[z.clone()]))
    {
        penalty = gauge.restrict_penalty(&penalty);
        penalty_psi = gauge.restrict_penalty(&penalty_psi);
        penalty_psi_psi = gauge.restrict_penalty(&penalty_psi_psi);
    }
    let (penalty_norm, penalty_norm_psi, penalty_norm_psi_psi, normalization_scale) =
        normalize_penaltywith_psi_derivatives(
            &symmetrize(&penalty),
            &symmetrize(&penalty_psi),
            &symmetrize(&penalty_psi_psi),
        );
    let (_, _, penaltyinfo) = filter_active_penalty_candidates(vec![PenaltyCandidate {
        matrix: penalty_norm,
        nullspace_dim_hint: 1,
        source: PenaltySource::Primary,
        normalization_scale,
        kronecker_factors: None,
        op: None,
    }])?;
    let mut penalties_derivative = Vec::new();
    let mut penaltiessecond_derivative = Vec::new();
    for info in penaltyinfo.iter().filter(|info| info.active) {
        match info.source {
            PenaltySource::Primary => {
                penalties_derivative.push(penalty_norm_psi.clone());
                penaltiessecond_derivative.push(penalty_norm_psi_psi.clone());
            }
            ref other => {
                crate::bail_invalid_basis!(
                    "unexpected periodic Duchon penalty source in derivative path: {other:?}"
                );
            }
        }
    }

    Ok(BasisPsiDerivativeBundle {
        first: BasisPsiDerivativeResult {
            design_derivative: design_first,
            penalties_derivative,
            implicit_operator: None,
        },
        second: BasisPsiSecondDerivativeResult {
            designsecond_derivative: design_second,
            penaltiessecond_derivative,
            implicit_operator: None,
        },
        implicit_operator: None,
    })
}

pub(crate) fn build_matern_design_psi_derivatives(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
    aniso_log_scales: Option<&[f64]>,
) -> Result<ScalarDesignPsiDerivatives, BasisError> {
    let k = centers.nrows();
    let kernel_cols = z_opt.map(|z| z.ncols()).unwrap_or(k);
    let total_cols = kernel_cols + usize::from(include_intercept);
    build_scalar_design_psi_derivatives_shared(
        data,
        centers,
        aniso_log_scales,
        total_cols,
        z_opt.cloned(),
        None,
        usize::from(include_intercept),
        RadialScalarKind::Matern { length_scale, nu },
        0.0,
    )
}

/// Build the Matérn double-penalty **primary** block (the projected kernel
/// Gram `A = Zᵀ K Z`, embedded into the `total_cols` coefficient space) and its
/// log-κ ψ-derivatives, in BOTH the un-normalized and the Frobenius-normalized
/// forms.
///
/// Returns `(s_norm, s_norm_psi, s_norm_psi_psi, c, a_raw, a_raw_psi,
/// a_raw_psi_psi)` where the `s_norm*` are the normalized primary penalty and
/// its ψ-derivatives (the active `PenaltySource::Primary` block) and the
/// `a_raw*` are the UN-normalized projected kernel and its ψ-derivatives. The
/// un-normalized triplet is what `build_nullspace_shrinkage_penalty` eigen-
/// decomposes in the value build, so it is exactly the matrix whose spectral
/// projector — and therefore the `DoublePenaltyNullspace` shrinkage block —
/// must be differentiated against (#1122).
pub(crate) fn build_matern_double_penalty_primarywith_psi_derivatives(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
    aniso_log_scales: Option<&[f64]>,
) -> Result<
    (
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        f64,
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
    ),
    BasisError,
> {
    let k = centers.nrows();
    let kernel_cols = z_opt.map(|z| z.ncols()).unwrap_or(k);
    let total_cols = kernel_cols + usize::from(include_intercept);
    let mut kernel = Array2::<f64>::zeros((k, k));
    let mut kernel_psi = Array2::<f64>::zeros((k, k));
    let mut kernel_psi_psi = Array2::<f64>::zeros((k, k));

    for i in 0..k {
        for j in i..k {
            let r = if let Some(eta) = aniso_log_scales {
                aniso_distance(
                    centers.row(i).as_slice().unwrap(),
                    centers.row(j).as_slice().unwrap(),
                    eta,
                )
            } else {
                stable_euclidean_norm(
                    (0..centers.ncols()).map(|axis| centers[[i, axis]] - centers[[j, axis]]),
                )
            };
            let value = matern_kernel_from_distance(r, length_scale, nu)?;
            let d1 = matern_kernel_log_kappa_derivative_from_distance(r, length_scale, nu)?;
            let d2 = matern_kernel_log_kappasecond_derivative_from_distance(r, length_scale, nu)?;
            kernel[[i, j]] = value;
            kernel[[j, i]] = value;
            kernel_psi[[i, j]] = d1;
            kernel_psi[[j, i]] = d1;
            kernel_psi_psi[[i, j]] = d2;
            kernel_psi_psi[[j, i]] = d2;
        }
    }

    let (kernel, kernel_psi, kernel_psi_psi) = if let Some(gauge) =
        z_opt.map(|z| crate::solver::gauge::Gauge::from_block_transforms(&[z.clone()]))
    {
        (
            gauge.restrict_penalty(&kernel),
            gauge.restrict_penalty(&kernel_psi),
            gauge.restrict_penalty(&kernel_psi_psi),
        )
    } else {
        (kernel, kernel_psi, kernel_psi_psi)
    };

    let mut s = Array2::<f64>::zeros((total_cols, total_cols));
    let mut s_psi = Array2::<f64>::zeros((total_cols, total_cols));
    let mut s_psi_psi = Array2::<f64>::zeros((total_cols, total_cols));
    s.slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&kernel);
    s_psi
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&kernel_psi);
    s_psi_psi
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&kernel_psi_psi);
    // `s`/`s_psi`/`s_psi_psi` are the UN-normalized projected kernel Gram
    // `A = Zᵀ K Z` (embedded into `total_cols`) and its exact log-κ
    // ψ-derivatives. `build_nullspace_shrinkage_penalty` (value build) eigen-
    // decomposes exactly this `A`, so the shrinkage-block derivative is the
    // spectral-projector derivative driven by `s_psi` / `s_psi_psi`.
    let (s_norm, s_norm_psi, s_norm_psi_psi, c) =
        normalize_penaltywith_psi_derivatives(&s, &s_psi, &s_psi_psi);
    Ok((s_norm, s_norm_psi, s_norm_psi_psi, c, s, s_psi, s_psi_psi))
}

/// Frozen-eigenbasis frame of the un-normalized projected Matérn kernel Gram
/// `A`, plus the constant data needed to differentiate the spectral projector
/// onto its near-null eigenspace `N = { i : |λ_i| ≤ tol }`.
///
/// The value build forms the `DoublePenaltyNullspace` block
/// (`build_nullspace_shrinkage_penalty`) as the Frobenius-normalized projector
///   `R~ = P / ‖P‖_F`,   `P = Σ_{i ∈ N} u_i u_iᵀ`,
/// at the SAME spectral tolerance used here. `P` is an orthogonal projector, so
/// `‖P‖_F = √r` with `r = |N|` the (FrozenTransform-pinned) null dimension — a
/// hyperparameter-independent constant. Hence every projector derivative is
/// scaled by `1/√r`.
///
/// `P` moves with the hyperparameters because `A` does, so its earlier hard-
/// coded zero derivative was an objective↔gradient desync that stalled the
/// isotropic-κ joint REML (#1122). Derivatives come from exact eigen-
/// perturbation in this frozen eigenbasis `U`.
pub(crate) struct ShrinkageProjectorFrame {
    /// Eigenvectors of `sym(A)`, columns ascending in eigenvalue.
    pub(crate) u: Array2<f64>,
    /// Eigenvalues (ascending).
    pub(crate) evals: Array1<f64>,
    /// `1` on near-null indices `N`, `0` elsewhere.
    pub(crate) in_null: Vec<f64>,
    /// Null dimension `r = |N|`.
    pub(crate) null_dim: usize,
    /// Connection-gap floor `tol` (pairs closer than this have no resolvable
    /// gap; their eigenvector sensitivity is ambiguous and they do not move the
    /// projector, so the connection entry is set to zero).
    pub(crate) gap_floor: f64,
}

impl ShrinkageProjectorFrame {
    /// Build the frame from the un-normalized projected kernel Gram `A`.
    /// Returns `None` when `A` has no near-null eigenspace at this tolerance
    /// (the value build emits no shrinkage block, so there is nothing to
    /// differentiate).
    pub(crate) fn build(a_raw: &Array2<f64>) -> Result<Option<Self>, BasisError> {
        if a_raw.nrows() == 0 {
            return Ok(None);
        }
        let (sym, evals, evecs) = spectral_summary(a_raw)?;
        let tol = spectral_tolerance(&sym, &evals);
        let in_null: Vec<f64> = evals
            .iter()
            .map(|&ev| if ev.abs() <= tol { 1.0 } else { 0.0 })
            .collect();
        let null_dim = in_null.iter().filter(|&&b| b != 0.0).count();
        if null_dim == 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            u: evecs,
            evals,
            in_null,
            null_dim,
            gap_floor: tol.max(f64::MIN_POSITIVE),
        }))
    }

    pub(crate) fn dim(&self) -> usize {
        self.u.nrows()
    }

    /// Skew connection `Ω_d[m,k] = (Uᵀ A_d U)[m,k] / (λ_k − λ_m)` for a single
    /// direction's `A_d = ∂A/∂η_d` (`m ≠ k`, floored at small gaps), together
    /// with the eigenbasis representation `B̂_d = Uᵀ A_d U` and the
    /// Hellmann–Feynman eigenvalue derivatives `λ_k' = B̂_d[k,k]`.
    pub(crate) fn connection(&self, a_dir: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Vec<f64>) {
        let p = self.dim();
        let b_hat = fast_atb(&self.u, &fast_ab(&symmetrize(a_dir), &self.u));
        let mut omega = Array2::<f64>::zeros((p, p));
        for m in 0..p {
            for k in 0..p {
                if m == k {
                    continue;
                }
                let gap = self.evals[k] - self.evals[m];
                if gap.abs() > self.gap_floor {
                    omega[[m, k]] = b_hat[[m, k]] / gap;
                }
            }
        }
        let lam_prime: Vec<f64> = (0..p).map(|k| b_hat[[k, k]]).collect();
        (omega, b_hat, lam_prime)
    }

    /// `P̂_d' = Ω_d I_N − I_N Ω_d` (frozen-frame projector first derivative for
    /// direction `d`; nonzero only across `N`↔`R`).
    pub(crate) fn projector_first_hat(&self, omega: &Array2<f64>) -> Array2<f64> {
        let p = self.dim();
        let mut out = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                let coeff = self.in_null[j] - self.in_null[i];
                if coeff != 0.0 {
                    out[[i, j]] = omega[[i, j]] * coeff;
                }
            }
        }
        out
    }

    /// Lab-frame, `1/√r`-normalized first derivative of the shrinkage block for
    /// direction `d`: `R~_d = U P̂_d' Uᵀ / √r`.
    pub(crate) fn first(&self, a_dir: &Array2<f64>) -> Array2<f64> {
        let (omega, _b_hat, _lam) = self.connection(a_dir);
        let p1_hat = self.projector_first_hat(&omega);
        self.to_lab(&p1_hat)
    }

    /// Lab-frame, `1/√r`-normalized mixed second derivative of the shrinkage
    /// block for directions `(a, b)`:
    ///   `Uᵀ P_ab U = ∂_b(P̂_a') + Ω_b P̂_a' − P̂_a' Ω_b`,
    /// `∂_b(P̂_a') = (∂_b Ω_a) I_N − I_N (∂_b Ω_a)`,
    /// `∂_b Ω_a[m,k] = B̂_a'^{(b)}[m,k]/(λ_k−λ_m) − B̂_a[m,k]·(λ_k'^{(b)}−λ_m'^{(b)})/(λ_k−λ_m)²`,
    /// `B̂_a'^{(b)} = Uᵀ A_ab U + B̂_a Ω_b − Ω_b B̂_a`,  `λ_k'^{(b)} = B̂_b[k,k]`.
    /// For the diagonal case `a == b` this is the ordinary second derivative.
    pub(crate) fn second(
        &self,
        a_dir_a: &Array2<f64>,
        a_dir_b: &Array2<f64>,
        a_cross: &Array2<f64>,
    ) -> Array2<f64> {
        let p = self.dim();
        let (omega_a, b_hat_a, _lam_a) = self.connection(a_dir_a);
        let (omega_b, _b_hat_b, lam_prime_b) = self.connection(a_dir_b);
        let p1a_hat = self.projector_first_hat(&omega_a);
        // B̂_a'^{(b)} = Uᵀ A_ab U + B̂_a Ω_b − Ω_b B̂_a.
        let c_hat = fast_atb(&self.u, &fast_ab(&symmetrize(a_cross), &self.u));
        let b_hat_a_prime = &c_hat + &(fast_ab(&b_hat_a, &omega_b) - fast_ab(&omega_b, &b_hat_a));
        // ∂_b Ω_a.
        let mut omega_a_db = Array2::<f64>::zeros((p, p));
        for m in 0..p {
            for k in 0..p {
                if m == k {
                    continue;
                }
                let gap = self.evals[k] - self.evals[m];
                if gap.abs() > self.gap_floor {
                    omega_a_db[[m, k]] = b_hat_a_prime[[m, k]] / gap
                        - b_hat_a[[m, k]] * (lam_prime_b[k] - lam_prime_b[m]) / (gap * gap);
                }
            }
        }
        // P̂_ab = (∂_b Ω_a) I_N − I_N (∂_b Ω_a) + Ω_b P̂_a' − P̂_a' Ω_b.
        let mut p2_hat = fast_ab(&omega_b, &p1a_hat) - fast_ab(&p1a_hat, &omega_b);
        for i in 0..p {
            for j in 0..p {
                let coeff = self.in_null[j] - self.in_null[i];
                if coeff != 0.0 {
                    p2_hat[[i, j]] += omega_a_db[[i, j]] * coeff;
                }
            }
        }
        self.to_lab(&p2_hat)
    }

    /// Map a frozen-frame projector derivative `P̂` back to the lab frame and
    /// apply the constant `1/√r` normalization: `symmetrize(U P̂ Uᵀ) / √r`.
    pub(crate) fn to_lab(&self, p_hat: &Array2<f64>) -> Array2<f64> {
        let inv_norm = 1.0 / (self.null_dim as f64).sqrt();
        symmetrize(&fast_ab(&self.u, &fast_abt(p_hat, &self.u))).mapv(|v| v * inv_norm)
    }
}

/// Exact isotropic-κ (`ρ = log κ`) first and second ψ-derivatives of the
/// Matérn double-penalty `DoublePenaltyNullspace` shrinkage block, driven by
/// the un-normalized projected-kernel Gram `A` and its log-κ derivatives.
/// Returns `(R~', R~'')`. Zeros when no shrinkage subspace exists at this ρ.
pub(crate) fn matern_nullspace_shrinkage_psi_derivatives(
    a_raw: &Array2<f64>,
    a_raw_psi: &Array2<f64>,
    a_raw_psi_psi: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let p = a_raw.nrows();
    let zero = || Array2::<f64>::zeros((p, p));
    let Some(frame) = ShrinkageProjectorFrame::build(a_raw)? else {
        return Ok((zero(), zero()));
    };
    let first = frame.first(a_raw_psi);
    let second = frame.second(a_raw_psi, a_raw_psi, a_raw_psi_psi);
    Ok((first, second))
}

/// Assemble the active Matérn double-penalty ψ-derivative blocks (first or
/// second order), index-aligned with `penaltyinfo`. The `Primary` block uses
/// the projected-kernel-Gram derivative; the `DoublePenaltyNullspace` block
/// uses the exact spectral-projector derivative (#1122) supplied per block.
pub(crate) fn active_matern_double_penalty_derivatives(
    penaltyinfo: &[PenaltyInfo],
    primary_derivative: &Array2<f64>,
    shrinkage_derivative: &Array2<f64>,
) -> Result<Vec<Array2<f64>>, BasisError> {
    penaltyinfo
        .iter()
        .filter(|info| info.active)
        .map(|info| match &info.source {
            PenaltySource::Primary => Ok(primary_derivative.clone()),
            PenaltySource::DoublePenaltyNullspace => Ok(shrinkage_derivative.clone()),
            other => Err(BasisError::InvalidInput(format!(
                "unexpected Matérn penalty source in double-penalty path: {other:?}"
            ))),
        })
        .collect()
}

pub fn build_matern_basis_log_kappa_derivative(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_matern_basis_log_kappa_derivativewithworkspace(data, spec, &mut workspace)
}

pub fn build_matern_basis_log_kappa_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let mut bundle = build_matern_basis_log_kappa_derivativeswithworkspace(data, spec, workspace)?;
    bundle.first.implicit_operator = bundle.implicit_operator;
    Ok(bundle.first)
}

pub fn build_matern_basis_log_kappa_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_matern_basis_log_kappa_derivativeswithworkspace(data, spec, &mut workspace)
}

pub fn build_matern_basis_log_kappa_derivativeswithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    // Analytic psi derivative assembly for the Matérn basis block.
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let z_opt = matern_identifiability_transform(centers.view(), &spec.identifiability)?;
    let aniso = spec.aniso_log_scales.as_deref();
    let design_derivatives = build_matern_design_psi_derivatives(
        data,
        centers.view(),
        spec.length_scale,
        spec.nu,
        spec.include_intercept,
        z_opt.as_ref(),
        aniso,
    )?;
    let (penalties_derivative, penaltiessecond_derivative) = if spec.double_penalty {
        let base = build_matern_basiswithworkspace(data, spec, workspace)?;
        let (_, primary_derivative, primarysecond_derivative, _, a_raw, a_raw_psi, a_raw_psi_psi) =
            build_matern_double_penalty_primarywith_psi_derivatives(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                z_opt.as_ref(),
                aniso,
            )?;
        // Exact log-κ ψ-derivatives of the `DoublePenaltyNullspace` shrinkage
        // projector, driven by the UN-normalized projected-kernel ψ-derivatives
        // (#1122). Computed only when an active shrinkage block exists.
        let (shrinkage_first, shrinkagesecond) =
            if base.penaltyinfo.iter().any(|info| {
                info.active && matches!(info.source, PenaltySource::DoublePenaltyNullspace)
            }) {
                matern_nullspace_shrinkage_psi_derivatives(&a_raw, &a_raw_psi, &a_raw_psi_psi)?
            } else {
                (
                    Array2::<f64>::zeros(a_raw.raw_dim()),
                    Array2::<f64>::zeros(a_raw.raw_dim()),
                )
            };
        (
            active_matern_double_penalty_derivatives(
                &base.penaltyinfo,
                &primary_derivative,
                &shrinkage_first,
            )?,
            active_matern_double_penalty_derivatives(
                &base.penaltyinfo,
                &primarysecond_derivative,
                &shrinkagesecond,
            )?,
        )
    } else {
        build_matern_operator_penalty_psi_derivatives(
            centers.view(),
            spec.length_scale,
            spec.nu,
            spec.include_intercept,
            z_opt.as_ref(),
            aniso,
        )?
    };

    Ok(BasisPsiDerivativeBundle {
        first: BasisPsiDerivativeResult {
            design_derivative: design_derivatives.design_first,
            penalties_derivative,
            implicit_operator: None,
        },
        second: BasisPsiSecondDerivativeResult {
            designsecond_derivative: design_derivatives.design_second_diag,
            penaltiessecond_derivative,
            implicit_operator: None,
        },
        implicit_operator: design_derivatives.implicit_operator,
    })
}

pub fn build_matern_basis_log_kappasecond_derivative(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_matern_basis_log_kappasecond_derivativewithworkspace(data, spec, &mut workspace)
}

pub fn build_matern_basis_log_kappasecond_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let mut bundle = build_matern_basis_log_kappa_derivativeswithworkspace(data, spec, workspace)?;
    bundle.second.implicit_operator = bundle.implicit_operator;
    Ok(bundle.second)
}

/// Build per-axis ψ_a design-matrix derivatives for anisotropic Matérn terms.
///
/// The optimized coordinates are the raw per-axis log-scales `psi_a`, so the
/// isotropic all-ones direction is part of this coordinate system. For Matérn
/// kernels there is no extra isotropic prefactor, so the raw-`psi` derivatives
/// are exactly the familiar shape-only terms.
pub(crate) fn build_matern_design_psi_aniso_derivatives(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    eta: &[f64],
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
) -> Result<AnisoBasisPsiDerivatives, BasisError> {
    let k = centers.nrows();
    let p_constrained = z_opt.map(|z| z.ncols()).unwrap_or(k);
    let n_poly = usize::from(include_intercept);
    let p_smooth = p_constrained + n_poly;
    build_aniso_design_psi_derivatives_shared(
        data,
        centers,
        eta,
        p_smooth,
        z_opt.cloned(),
        None,
        n_poly,
        RadialScalarKind::Matern { length_scale, nu },
    )
}

pub(crate) fn build_matern_aniso_primary_raw_derivative_matrices(
    centers: ArrayView2<'_, f64>,
    eta: &[f64],
    length_scale: f64,
    nu: MaternNu,
) -> Result<(Vec<Array2<f64>>, Vec<Array2<f64>>), BasisError> {
    let k = centers.nrows();
    let dim = centers.ncols();
    let row_blocks: Result<Vec<_>, BasisError> = (0..k)
        .into_par_iter()
        .map(|i| {
            let ci: Vec<f64> = (0..dim).map(|a| centers[[i, a]]).collect();
            let mut first_by_axis: Vec<Vec<f64>> =
                (0..dim).map(|_| Vec::with_capacity(k - i)).collect();
            let mut second_diag_by_axis: Vec<Vec<f64>> =
                (0..dim).map(|_| Vec::with_capacity(k - i)).collect();
            for j in i..k {
                let cj: Vec<f64> = (0..dim).map(|a| centers[[j, a]]).collect();
                let (r, s_vec) = aniso_distance_and_components(&ci, &cj, eta);
                let (_, q, t, _, _) = matern_aniso_extended_radial_scalars(r, length_scale, nu)?;
                for a in 0..dim {
                    let s_a = s_vec[a];
                    first_by_axis[a].push(q * s_a);
                    second_diag_by_axis[a].push(2.0 * q * s_a + t * s_a * s_a);
                }
            }
            Ok((first_by_axis, second_diag_by_axis))
        })
        .collect();

    let row_blocks = row_blocks?;
    let mut raw_first = vec![Array2::<f64>::zeros((k, k)); dim];
    let mut raw_second_diag = vec![Array2::<f64>::zeros((k, k)); dim];
    for (i, (first_by_axis, second_diag_by_axis)) in row_blocks.into_iter().enumerate() {
        for (offset, j) in (i..k).enumerate() {
            for a in 0..dim {
                let d1 = first_by_axis[a][offset];
                let d2 = second_diag_by_axis[a][offset];
                raw_first[a][[i, j]] = d1;
                raw_first[a][[j, i]] = d1;
                raw_second_diag[a][[i, j]] = d2;
                raw_second_diag[a][[j, i]] = d2;
            }
        }
    }

    Ok((raw_first, raw_second_diag))
}

pub(crate) fn build_matern_aniso_raw_cross_derivative_matrix(
    centers: ArrayView2<'_, f64>,
    eta: &[f64],
    length_scale: f64,
    nu: MaternNu,
    axis_a: usize,
    axis_b: usize,
) -> Result<Array2<f64>, BasisError> {
    let k = centers.nrows();
    let dim = centers.ncols();
    let row_blocks: Result<Vec<_>, BasisError> = (0..k)
        .into_par_iter()
        .map(|i| {
            let ci: Vec<f64> = (0..dim).map(|ax| centers[[i, ax]]).collect();
            let mut values = Vec::with_capacity(k - i);
            for j in i..k {
                let cj: Vec<f64> = (0..dim).map(|ax| centers[[j, ax]]).collect();
                let (r, s_vec) = aniso_distance_and_components(&ci, &cj, eta);
                let (_, _, t_val, _, _) =
                    matern_aniso_extended_radial_scalars(r, length_scale, nu)?;
                values.push(t_val * s_vec[axis_a] * s_vec[axis_b]);
            }
            Ok(values)
        })
        .collect();

    let row_blocks = row_blocks?;
    let mut raw_cross = Array2::<f64>::zeros((k, k));
    for (i, values) in row_blocks.into_iter().enumerate() {
        for (offset, j) in (i..k).enumerate() {
            let value = values[offset];
            raw_cross[[i, j]] = value;
            raw_cross[[j, i]] = value;
        }
    }
    Ok(raw_cross)
}

/// Build per-axis ψ_a derivatives for anisotropic Matérn terms, including
/// both design-matrix and penalty derivatives.
///
/// For each axis a (0..d), produces first and second derivative information.
/// The penalty derivatives use the fractional weighting approach for operator
/// penalties, and exact per-axis R-operator derivatives for double penalties.
pub fn build_matern_basis_log_kappa_aniso_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &MaternBasisSpec,
) -> Result<AnisoBasisPsiDerivatives, BasisError> {
    let eta = spec.aniso_log_scales.as_deref().ok_or_else(|| {
        BasisError::InvalidInput("aniso derivatives require aniso_log_scales to be set".to_string())
    })?;
    let dim = data.ncols();
    if eta.len() != dim {
        crate::bail_dim_basis!(
            "aniso_log_scales length {} != data dimension {dim}",
            eta.len()
        );
    }

    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let z_opt = matern_identifiability_transform(centers.view(), &spec.identifiability)?;

    let mut result = build_matern_design_psi_aniso_derivatives(
        data,
        centers.view(),
        spec.length_scale,
        spec.nu,
        eta,
        spec.include_intercept,
        z_opt.as_ref(),
    )?;

    // Penalty per-axis derivatives.
    if spec.double_penalty {
        // Double-penalty path: per-axis primary penalty derivatives via R-operators.
        let k = centers.nrows();
        let kernel_cols = z_opt.as_ref().map(|z| z.ncols()).unwrap_or(k);
        let total_cols = kernel_cols + usize::from(spec.include_intercept);
        let mut primary_first = vec![Array2::<f64>::zeros((total_cols, total_cols)); dim];
        let mut primary_second_diag = vec![Array2::<f64>::zeros((total_cols, total_cols)); dim];
        let coefficient_gauge = z_opt
            .as_ref()
            .map(|z| crate::solver::gauge::Gauge::from_block_transforms(&[z.clone()]));
        let (mut raw_first, mut raw_second_diag) =
            build_matern_aniso_primary_raw_derivative_matrices(
                centers.view(),
                eta,
                spec.length_scale,
                spec.nu,
            )?;
        for a in 0..dim {
            // raw_first[a] / raw_second_diag[a] are dropped after this loop.
            // When there is no identifiability transform we previously cloned
            // them just to slice into primary_*; move them instead.
            let projected_first = if let Some(gauge) = coefficient_gauge.as_ref() {
                gauge.restrict_penalty(&raw_first[a])
            } else {
                std::mem::take(&mut raw_first[a])
            };
            let projected_second = if let Some(gauge) = coefficient_gauge.as_ref() {
                gauge.restrict_penalty(&raw_second_diag[a])
            } else {
                std::mem::take(&mut raw_second_diag[a])
            };
            primary_first[a]
                .slice_mut(s![0..kernel_cols, 0..kernel_cols])
                .assign(&projected_first);
            primary_second_diag[a]
                .slice_mut(s![0..kernel_cols, 0..kernel_cols])
                .assign(&projected_second);
        }
        let mut dp_cross_pairs: Vec<(usize, usize)> = Vec::new();
        for a in 0..dim {
            for b in (a + 1)..dim {
                dp_cross_pairs.push((a, b));
            }
        }

        let base = build_matern_basiswithworkspace(data, spec, &mut BasisWorkspace::default())?;
        let has_shrinkage = base.penaltyinfo.iter().any(|info| {
            info.active && matches!(info.source, PenaltySource::DoublePenaltyNullspace)
        });
        // The un-normalized projected aniso kernel Gram `A = Zᵀ K Z` (embedded
        // into `total_cols`) is what the value build's shrinkage block eigen-
        // decomposes; its per-axis η_a derivatives are `primary_first[a]`. The
        // shrinkage-projector derivative (#1122) is driven by exactly these.
        let shrinkage_frame = if has_shrinkage {
            let kernel = build_matern_kernel_penalty(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                Some(eta),
            )?;
            let kblock = kernel.slice(s![0..k, 0..k]).to_owned();
            let mut a_raw = Array2::<f64>::zeros((total_cols, total_cols));
            let projected = if let Some(gauge) = coefficient_gauge.as_ref() {
                gauge.restrict_penalty(&kblock)
            } else {
                kblock
            };
            a_raw
                .slice_mut(s![0..kernel_cols, 0..kernel_cols])
                .assign(&projected);
            ShrinkageProjectorFrame::build(&a_raw)?
        } else {
            None
        };
        let shrinkage_first: Vec<Array2<f64>> = (0..dim)
            .map(|a| match &shrinkage_frame {
                Some(frame) => frame.first(&primary_first[a]),
                None => Array2::<f64>::zeros((total_cols, total_cols)),
            })
            .collect();
        let shrinkage_second_diag: Vec<Array2<f64>> = (0..dim)
            .map(|a| match &shrinkage_frame {
                Some(frame) => frame.second(
                    &primary_first[a],
                    &primary_first[a],
                    &primary_second_diag[a],
                ),
                None => Array2::<f64>::zeros((total_cols, total_cols)),
            })
            .collect();
        result.penalties_first = Vec::with_capacity(dim);
        result.penalties_second_diag = Vec::with_capacity(dim);
        for a in 0..dim {
            let pf = active_matern_double_penalty_derivatives(
                &base.penaltyinfo,
                &primary_first[a],
                &shrinkage_first[a],
            )?;
            let ps = active_matern_double_penalty_derivatives(
                &base.penaltyinfo,
                &primary_second_diag[a],
                &shrinkage_second_diag[a],
            )?;
            result.penalties_first.push(pf);
            result.penalties_second_diag.push(ps);
        }
        result.penalties_cross_pairs = dp_cross_pairs;
        let centers_owned = centers.to_owned();
        let eta_owned = eta.to_vec();
        let gauge_owned = z_opt
            .as_ref()
            .map(|z| crate::solver::gauge::Gauge::from_block_transforms(&[z.clone()]));
        let penaltyinfo = base.penaltyinfo.clone();
        let length_scale = spec.length_scale;
        let nu = spec.nu;
        // Per-axis projected first derivatives ∂A/∂η_a (embedded), so the
        // cross-pair shrinkage second derivative `∂²P/∂η_a∂η_b` can be formed
        // exactly inside the provider (#1122).
        let primary_first_owned = primary_first.clone();
        let include_intercept = spec.include_intercept;
        result.penalties_cross_provider = Some(AnisoPenaltyCrossProvider::new(
            move |axis_a: usize, axis_b: usize| {
                let (a, b) = if axis_a < axis_b {
                    (axis_a, axis_b)
                } else {
                    (axis_b, axis_a)
                };
                if a == b || b >= eta_owned.len() {
                    return Ok(Vec::new());
                }
                let raw_cross = build_matern_aniso_raw_cross_derivative_matrix(
                    centers_owned.view(),
                    &eta_owned,
                    length_scale,
                    nu,
                    a,
                    b,
                )?;
                let projected: Array2<f64> = if let Some(gauge) = gauge_owned.as_ref() {
                    gauge.restrict_penalty(&raw_cross)
                } else {
                    raw_cross
                };
                let mut padded = Array2::<f64>::zeros((total_cols, total_cols));
                padded
                    .slice_mut(s![0..kernel_cols, 0..kernel_cols])
                    .assign(&projected);
                // Exact cross second derivative of the shrinkage projector, if
                // an active shrinkage block exists at this hyperparameter.
                let shrinkage_cross = if penaltyinfo.iter().any(|info| {
                    info.active && matches!(info.source, PenaltySource::DoublePenaltyNullspace)
                }) {
                    let kernel = build_matern_kernel_penalty(
                        centers_owned.view(),
                        length_scale,
                        nu,
                        include_intercept,
                        Some(&eta_owned),
                    )?;
                    let k = centers_owned.nrows();
                    let kblock = kernel.slice(s![0..k, 0..k]).to_owned();
                    let projected_a = if let Some(gauge) = gauge_owned.as_ref() {
                        gauge.restrict_penalty(&kblock)
                    } else {
                        kblock
                    };
                    let mut a_raw = Array2::<f64>::zeros((total_cols, total_cols));
                    a_raw
                        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
                        .assign(&projected_a);
                    match ShrinkageProjectorFrame::build(&a_raw)? {
                        Some(frame) => {
                            frame.second(&primary_first_owned[a], &primary_first_owned[b], &padded)
                        }
                        None => Array2::<f64>::zeros((total_cols, total_cols)),
                    }
                } else {
                    Array2::<f64>::zeros((total_cols, total_cols))
                };
                active_matern_double_penalty_derivatives(&penaltyinfo, &padded, &shrinkage_cross)
            },
        ));
    } else {
        // Operator penalty path: exact per-axis η_a derivatives.
        // Replaces the former fractional approximation with exact analytic
        // derivatives of D₀, D₁, D₂ w.r.t. each aniso log-scale η_a,
        // assembled via the Gram product rule into penalty derivatives.
        let (per_axis, cross_pairs, cross_provider) =
            build_matern_operator_penalty_aniso_derivatives(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                z_opt.as_ref(),
                eta,
            )?;

        result.penalties_first = Vec::with_capacity(dim);
        result.penalties_second_diag = Vec::with_capacity(dim);
        for (pen_first, pen_second) in per_axis {
            result.penalties_first.push(pen_first);
            result.penalties_second_diag.push(pen_second);
        }
        result.penalties_cross_pairs = cross_pairs;
        result.penalties_cross_provider = Some(cross_provider);
    }

    if dim > 1 && !result.penalties_first.is_empty() {
        // The forward anisotropic Matérn design uses the CENTERED contrast
        // metric `w_a = exp(2·(η_a − mean(η)))` (see `centered_aniso_metric_weights`
        // / `aniso_distance`): a uniform shift of every η_a leaves the mean-
        // subtracted weights — and therefore the whole design, kernel, and
        // penalty — unchanged. The optimizer's per-axis ψ-coordinate is the raw
        // η_a, so the criterion is invariant along the all-ones direction and
        // the analytic penalty derivative w.r.t. raw η_a must be the centering
        // projection of the per-axis ψ-derivative:
        //
        //   ∂S/∂η_a = ∂S/∂ψ_a − (1/d) Σ_b ∂S/∂ψ_b   (η_a − mean(η) chain rule).
        //
        // The per-axis builders above produce `∂S/∂ψ_a`, treating each centered
        // contrast as independent; subtracting the mean across axes removes the
        // spurious common-mode. (#1259: the previous code added back a
        // `scalar_share = (1/d)·∂S/∂log κ` term, which is the gradient of a
        // GLOBAL length-scale move — but the centered forward design makes a
        // uniform η shift a no-op, not a log-κ change, so that add-back injected
        // a fake all-ones gradient component. The FD-audit of the outer REML
        // criterion flagged it as a per-axis analytic≠FD desync that misdirected
        // the κ-optimizer off the true signal-axis contrast.)
        let inv_dim = 1.0 / dim as f64;
        let num_blocks = result.penalties_first[0].len();
        for block in 0..num_blocks {
            let mut eta_mean = Array2::<f64>::zeros(result.penalties_first[0][block].raw_dim());
            for axis in 0..dim {
                if result.penalties_first[axis][block].raw_dim()
                    != result.penalties_first[0][block].raw_dim()
                {
                    return Err(BasisError::InvalidInput(format!(
                        "Matérn aniso raw-psi penalty derivative shape mismatch on axis {axis}, block {block}"
                    )));
                }
                eta_mean += &result.penalties_first[axis][block];
            }
            eta_mean.mapv_inplace(|value| value * inv_dim);
            for axis in 0..dim {
                result.penalties_first[axis][block] =
                    &result.penalties_first[axis][block] - &eta_mean;
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod harmonic_penalty_invariants_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn harmonic_double_penalty_targets_primary_nullspace() {
        let data = array![
            [-0.9, 0.0],
            [-0.4, 1.0],
            [0.0, 2.0],
            [0.4, 3.0],
            [0.9, 4.0],
            [0.2, 5.0],
        ];
        let spec = SphericalSplineBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
            penalty_order: 2,
            double_penalty: true,
            radians: true,
            method: SphereMethod::Harmonic,
            max_degree: Some(3),
            wahba_kernel: SphereWahbaKernel::Sobolev,
            identifiability: SphericalSplineIdentifiability::CenterSumToZero,
        };
        let built = build_spherical_harmonic_basis(data.view(), &spec).expect("harmonic basis");
        assert_eq!(built.penalties.len(), 2);
        let primary = &built.penalties[0];
        let shrink = &built.penalties[1];
        for col in 0..primary.ncols() {
            let primary_diag = primary[[col, col]].abs();
            let shrink_diag = shrink[[col, col]].abs();
            if col < SPHERE_UNPENALIZED_LOW_DEGREE * (SPHERE_UNPENALIZED_LOW_DEGREE + 2) {
                assert!(
                    primary_diag <= 1e-12,
                    "low-degree column {col} must be primary-null"
                );
                assert!(
                    shrink_diag > 0.0,
                    "low-degree column {col} must be shrink-penalized"
                );
            } else {
                assert!(
                    primary_diag > 0.0,
                    "higher-degree column {col} must carry roughness"
                );
                assert!(
                    shrink_diag <= 1e-12,
                    "higher-degree column {col} must not be in null shrinkage"
                );
            }
        }
    }
}
