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
    // The primary penalty is exactly the Wahba RKHS seminorm `c'Kc` of the
    // fitted function — no coefficient ridge is fused into it. Raw-chart
    // shrinkage (sparse polar center leverage) is statistically a prior on
    // the low-degree/null directions, so it lives ONLY in the separate
    // REML-selected double-penalty candidate below; if the evidence drives
    // that block to zero, zero shrinkage is the correct estimate. Numerical
    // conditioning of near-singular kernel blocks is the solver's job, not
    // the objective's.
    let raw_penalty = build_wahba_decomposed_penalty(center_kernel.view(), &decomposition);
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
    let gauge = gam_problem::Gauge::from_block_transforms(&[z.clone()]);
    let penalty = gauge.restrict_penalty(&raw_penalty);
    // The selected sphere centers are the compact domain quadrature carried by
    // this finite-rank Wahba representation. Evaluate the exact decomposed
    // chart `[K_CC Z - H C | H]` on those centers, then apply the FINAL active
    // coefficient chart before forming G. This makes the null ridge a penalty
    // on the represented low-degree function, covariant under every frozen
    // basis reparameterization.
    let raw_center_design = build_wahba_decomposed_design(
        center_kernel.view(),
        centers.view(),
        spec.radians,
        &decomposition,
    );
    let center_design = gauge.restrict_design(&raw_center_design);
    let function_gram = symmetrize_penalty(&fast_ata(&center_design));
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
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
        if let Some(ridge) = function_space_nullspace_shrinkage(&penalty, &function_gram)? {
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
    }
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    Ok(BasisBuildResult {
        design,
        affine_offset: None,
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

pub(crate) const SPHERE_UNPENALIZED_LOW_DEGREE: usize = 1;

pub(crate) fn harmonic_degree_for_wahba_basis_width(
    spec: &SphericalSplineBasisSpec,
    n_rows: usize,
) -> usize {
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

pub(crate) struct WahbaLowDegreeDecomposition {
    pub(crate) kernel_basis: Array2<f64>,
    pub(crate) low_degree_centers: Option<Array2<f64>>,
    pub(crate) kernel_low_projection: Option<Array2<f64>>,
    /// Columns of the complete degree-1 harmonic design retained by canonical
    /// RRQR. Prediction and input-location jets replay this exact selection.
    pub(crate) low_degree_columns: Vec<usize>,
    pub(crate) low_degree_cols: usize,
}

pub(crate) fn wahba_low_degree_decomposition(
    centers: ArrayView2<'_, f64>,
    radians: bool,
    center_kernel: ArrayView2<'_, f64>,
) -> Result<WahbaLowDegreeDecomposition, BasisError> {
    let complete_harmonics = real_spherical_harmonic_design_up_to_degree(
        centers,
        SPHERE_UNPENALIZED_LOW_DEGREE,
        radians,
    );
    let rank_alpha = gam_linalg::faer_ndarray::default_rrqr_rank_alpha();
    let low_rrqr = gam_linalg::faer_ndarray::rrqr_with_permutation(&complete_harmonics, rank_alpha)
        .map_err(BasisError::LinalgError)?;
    let low_degree_columns = low_rrqr.column_permutation[..low_rrqr.rank].to_vec();
    if low_degree_columns.is_empty() {
        return Ok(WahbaLowDegreeDecomposition {
            kernel_basis: Array2::<f64>::eye(centers.nrows()),
            low_degree_centers: None,
            kernel_low_projection: None,
            low_degree_columns,
            low_degree_cols: 0,
        });
    }

    let low_degree_centers = complete_harmonics.select(Axis(1), &low_degree_columns);
    // Native-space gauge for a conditionally positive-definite spherical
    // spline: the kernel coefficients satisfy H^T c = 0, where H is the
    // supported low-degree harmonic block. This removes the exact overlap
    // between Kc and H alpha without perturbing K. The previous chart solved
    // (K + delta I) C = H with a fixed delta and then complemented C, so the
    // realized function space changed with coefficient scaling and the fixed
    // shift was frozen into prediction. Canonical RRQR supplies the exact,
    // rank-adapted section instead.
    let (kernel_basis, low_rank) =
        gam_linalg::faer_ndarray::rrqr_nullspace_basis(&low_degree_centers, rank_alpha)
            .map_err(BasisError::LinalgError)?;
    if low_rank != low_degree_centers.ncols() {
        crate::bail_invalid_basis!(
            "sphere Wahba low-degree RRQR selection lost rank: selected {} harmonics but resolved rank {}",
            low_degree_centers.ncols(),
            low_rank
        );
    }
    let center_kernel_reduced = center_kernel.dot(&kernel_basis);
    // Express the empirical projection of KZ onto H through a pivoted-QR
    // least-squares solve. Forming (H^T H)^{-1} H^T KZ would square H's
    // condition number and can fail precisely for a supported direction near
    // the RRQR rank boundary. QR operates on the resolved design itself and
    // introduces no diagonal shift.
    let kernel_low_projection =
        solve_least_squares_columns(low_degree_centers.view(), center_kernel_reduced.view())?;
    let low_degree_cols = low_degree_centers.ncols();
    Ok(WahbaLowDegreeDecomposition {
        kernel_basis,
        low_degree_centers: Some(low_degree_centers),
        kernel_low_projection: Some(kernel_low_projection),
        low_degree_columns,
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
    // `(n × m) · (m × k)` reduction of the raw finite-center kernel onto the
    // penalized RKHS chart. At production sphere shapes (n ≳ 1e5, m ~ 200) this
    // is the single largest host cost in the basis build — ~1 s of generic
    // matrixmultiply — so route it through `fast_ab`, which dispatches to the
    // cuBLAS GEMM when the workload clears the policy flop floor and otherwise
    // falls back to the identical faer/SIMD path. Bit-for-bit identical on the
    // CPU branch; device GEMM differs only by IEEE reduction order, well inside
    // the documented Wahba parity tolerance.
    let mut kernel_design = fast_ab(&raw_kernel_design, &decomposition.kernel_basis);
    match (
        &decomposition.low_degree_centers,
        &decomposition.kernel_low_projection,
    ) {
        (Some(low_degree_centers), Some(kernel_low_projection)) => {
            let complete_low = real_spherical_harmonic_design_up_to_degree(
                data,
                SPHERE_UNPENALIZED_LOW_DEGREE,
                radians,
            );
            let low_design = complete_low.select(Axis(1), &decomposition.low_degree_columns);
            assert_eq!(low_design.ncols(), low_degree_centers.ncols());
            // `(n × low) · (low × k)` — `low` is tiny (3 for degree 1), so this
            // stays on the host SIMD path; `.dot` is fine here.
            kernel_design -= &low_design.dot(kernel_low_projection);
            hstack_dense(kernel_design.view(), low_design.view())
        }
        _ => kernel_design,
    }
}

/// Build the DESIGN jet `∂Φ/∂(lat, lon)` of the Wahba decomposed design,
/// matching the exact column layout of [`build_wahba_decomposed_design`].
///
/// The forward decomposed design is the linear map
///   `[ raw_kernel·kernel_basis − low·kernel_low_projection | low ]`
/// of the per-row evaluated kernel and low-degree harmonic functions, so its
/// derivative w.r.t. each angular axis is the SAME linear map applied to the
/// per-row jets:
///   `kernel_jet = raw_kernel_jet·kernel_basis − low_jet·kernel_low_projection`,
///   `out_jet    = [ kernel_jet | low_jet ]`.
/// If the realized harmonic block has rank zero, the design is just
/// `raw_kernel·kernel_basis`, so the jet is `raw_kernel_jet·kernel_basis`.
pub(crate) fn build_wahba_decomposed_jet(
    raw_kernel_jet: &Array3<f64>,
    low_jet: Option<&Array3<f64>>,
    decomposition: &WahbaLowDegreeDecomposition,
) -> Array3<f64> {
    let n = raw_kernel_jet.shape()[0];
    match (
        &decomposition.kernel_low_projection,
        low_jet,
        &decomposition.low_degree_centers,
    ) {
        (Some(kernel_low_projection), Some(low_jet), Some(_)) => {
            let kernel_cols = decomposition.kernel_basis.ncols();
            let low_cols = decomposition.low_degree_cols;
            let low_jet = low_jet.select(Axis(1), &decomposition.low_degree_columns);
            let mut out = Array3::<f64>::zeros((n, kernel_cols + low_cols, 2));
            for axis in 0..2 {
                let raw_axis = raw_kernel_jet.index_axis(ndarray::Axis(2), axis);
                let low_axis = low_jet.index_axis(ndarray::Axis(2), axis);
                let kernel_axis =
                    raw_axis.dot(&decomposition.kernel_basis) - low_axis.dot(kernel_low_projection);
                out.slice_mut(s![.., 0..kernel_cols, axis])
                    .assign(&kernel_axis);
                out.slice_mut(s![.., kernel_cols.., axis]).assign(&low_axis);
            }
            out
        }
        _ => {
            let kernel_cols = decomposition.kernel_basis.ncols();
            let mut out = Array3::<f64>::zeros((n, kernel_cols, 2));
            for axis in 0..2 {
                let raw_axis = raw_kernel_jet.index_axis(ndarray::Axis(2), axis);
                let projected = raw_axis.dot(&decomposition.kernel_basis);
                out.slice_mut(s![.., .., axis]).assign(&projected);
            }
            out
        }
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

fn solve_least_squares_columns(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, BasisError> {
    use faer::prelude::*;
    use gam_linalg::faer_ndarray::FaerArrayView;

    if a.nrows() == 0 || a.ncols() == 0 || a.nrows() < a.ncols() || b.nrows() != a.nrows() {
        crate::bail_dim_basis!(
            "least-squares solve needs a nonempty tall A and matching RHS rows, got A={}x{} and B={}x{}",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        );
    }
    if b.ncols() == 0 {
        return Ok(Array2::<f64>::zeros((a.ncols(), 0)));
    }
    let a_owned = a.to_owned();
    let b_owned = b.to_owned();
    let a_view = FaerArrayView::new(&a_owned);
    let b_view = FaerArrayView::new(&b_owned);
    let solved = a_view.as_ref().col_piv_qr().solve_lstsq(b_view.as_ref());
    let mut out = Array2::<f64>::zeros((a.ncols(), b.ncols()));
    for col in 0..out.ncols() {
        for row in 0..out.nrows() {
            out[[row, col]] = solved[(row, col)];
        }
    }
    if !out.iter().all(|v| v.is_finite()) {
        return Err(BasisError::InvalidInput(
            "sphere Wahba low-degree QR projection produced non-finite coefficients".to_string(),
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
    // Diagonal Laplace-Beltrami curvature penalty [l(l+1)]^order per (l, m).
    //
    // The real spherical harmonics diagonalize the Laplace-Beltrami operator on
    // S²: the degree-l block is an irreducible SO(3) representation with the
    // single eigenvalue l(l+1), constant across all 2l+1 modes in the block.
    // Raising every mode in a degree to the SAME power therefore yields a
    // ROTATION-INVARIANT penalty (it commutes with the SO(3) action), which is
    // exactly the iterated Laplacian Sobolev seminorm ‖Δ^order f‖². Because the
    // penalty is rotation-invariant and the sphere area measure is rotation-
    // invariant, the induced smoothing is latitude-independent by construction:
    // no pole/equator direction is privileged (#1246). Splitting the curvature
    // penalty by mode type (e.g. routing zonal m=0 modes to a separate block)
    // would BREAK this invariance and reintroduce a latitude-dependent error
    // profile, so the operator is kept as one isotropic block.
    //
    // The basis is orthonormal on S², so X'X/n is O(1) under uniform sampling
    // while the diagonal entries are the physical roughness eigenvalues of the
    // fitted function. Frobenius-normalizing would divide away that meaningful
    // spectral scale, making REML optimize against an artificially tiny physical
    // penalty, so the raw operator is kept with normalization_scale=1 and the
    // optimizer lambda is a physical lambda for this smooth.
    let mut penalty = Array2::<f64>::zeros((p, p));
    let mut col = 0usize;
    for l in 1..=l_max {
        let laplace = l as f64 * (l as f64 + 1.0);
        let eig = laplace.powi(spec.penalty_order as i32);
        for _ in 0..(2 * l + 1) {
            penalty[[col, col]] = eig;
            col += 1;
        }
    }
    // The curvature penalty above is full-rank within this basis (every degree
    // l >= 1 has eigenvalue l(l+1) > 0; the constant mode l=0 is not part of the
    // harmonic basis), so it has an empty null space. A double-penalty therefore
    // contributes a uniform isotropic ridge that shrinks all coefficients toward
    // zero under its own smoothing parameter — the standard mgcv-style extra
    // shrinkage term — and is only active when explicitly requested.
    let mut ridge = Array2::<f64>::zeros((p, p));
    for c in 0..p {
        ridge[[c, c]] = 1.0;
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
    let gauge = gam_problem::Gauge::from_block_transforms(&[transform.clone()]);
    let design = gauge.restrict_design(&design);
    let penalty = gauge.restrict_penalty(&penalty);
    let ridge = gauge.restrict_penalty(&ridge);

    let mut candidates = vec![PenaltyCandidate {
        matrix: penalty,
        nullspace_dim_hint: 0,
        source: PenaltySource::Primary,
        normalization_scale: 1.0,
        kronecker_factors: None,
        op: None,
    }];
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
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(design)),
        affine_offset: None,
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
        let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(op)));
        let candidates = if spec.double_penalty {
            let penalty_kernel = build_matern_kernel_penalty(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                aniso.as_deref(),
            )?;
            let primary = project_penalty_matrix(&penalty_kernel, full_transform.as_ref());
            let function_gram = matern_center_function_gram(
                &penalty_kernel,
                spec.include_intercept,
                full_transform.as_ref(),
            )?;
            matern_double_penalty_candidates(
                &primary,
                &function_gram,
                spec.include_intercept,
            )?
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
        let kernel_gauge = z_opt
            .as_ref()
            .map(|z| Arc::new(gam_problem::Gauge::from_block_transforms(&[z.clone()])));
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
                workspace.policy().material_policy(),
            )
            .map_err(BasisError::InvalidInput)?;
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(op)))
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
                workspace.policy().material_policy(),
            )
            .map_err(BasisError::InvalidInput)?;
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(op)))
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
            let function_gram = matern_center_function_gram(
                &penalty_kernel,
                spec.include_intercept,
                full_transform.as_ref(),
            )?;
            matern_double_penalty_candidates(
                &primary,
                &function_gram,
                spec.include_intercept,
            )?
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
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(fast_ab(
                &m.basis, transform,
            )))
        } else {
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(m.basis.clone()))
        };
        let candidates = if spec.double_penalty {
            build_matern_double_penalty_candidates(&m, full_transform.as_ref())?
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
        affine_offset: None,
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

pub fn build_matern_operator_penalty_psi_derivatives(
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

    let coefficient_gauge = z_opt.map(|z| gam_problem::Gauge::from_block_transforms(&[z.clone()]));
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

pub fn build_duchon_operator_penalty_psi_derivatives(
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
    // Fill the polynomial null-space columns of the operator design exactly as the
    // value-side `build_duchon_collocation_operator_matrices` does: the monomial
    // value (D0), gradient (D1), and Hessian (D2) evaluated at the collocation
    // sites. The operator penalties (mass `Σ(f−f̄)²`, tension `Σ‖∇f‖²`, stiffness
    // `Σ‖∇²f‖²`) penalize the WHOLE function, whose polynomial trend block is part
    // of the collocation Gram `S = DᵀD` — leaving these columns at zero silently
    // dropped the trend block from the derivative-side penalty. The polynomial
    // block is κ-independent, so its ψ-derivative columns stay zero; but its
    // presence in `D` (not `D_ψ`) is precisely the kernel↔trend cross term
    // `D_ψᵀ P + Pᵀ D_ψ` that the analytic `S_ψ = D_ψᵀ D + Dᵀ D_ψ` was missing.
    // Without it the analytic operator-penalty log-κ gradient disagreed with a
    // central finite difference of the rebuilt penalty (the native Primary Gram
    // was correct because it genuinely leaves the polynomial null space
    // unpenalized).
    if poly_cols > 0 {
        d0.slice_mut(s![.., kernel_cols..total_cols])
            .assign(&polynomial_block_from_order(
                collocation_points,
                effective_nullspace_order,
            ));
        if need_d1 {
            d1.slice_mut(s![.., kernel_cols..total_cols])
                .assign(&polynomial_derivative_block(
                    collocation_points,
                    effective_nullspace_order,
                    1,
                ));
        }
        if need_d2 {
            d2.slice_mut(s![.., kernel_cols..total_cols])
                .assign(&polynomial_derivative_block(
                    collocation_points,
                    effective_nullspace_order,
                    2,
                ));
        }
    }
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

    let coefficient_gauge =
        identifiability_transform.map(|z| gam_problem::Gauge::from_block_transforms(&[z.clone()]));
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

pub fn build_duchon_native_penalty_psi_derivatives(
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
    let mut z =
        kernel_constraint_nullspace(centers, effective_nullspace_order, &mut workspace.cache)?;
    // #1355: fold the frozen data-metric reparam `Z' = Z·V` so the penalty
    // ψ-derivatives project in the SAME rotated radial basis as the forward
    // penalty (`Vᵀ Ω(ψ) V`), staying bit-consistent with the design.
    if let Some(v) = spec.radial_reparam.as_ref() {
        if v.nrows() != z.ncols() {
            crate::bail_dim_basis!(
                "Duchon frozen radial reparam shape {:?} does not match constrained kernel dimension {}",
                v.dim(),
                z.ncols()
            );
        }
        z = fast_ab(&z, v);
    }
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
    let kernel_gauge = gam_problem::Gauge::from_block_transforms(&[z.clone()]);
    let project_kernel = |k: &Array2<f64>| kernel_gauge.restrict_penalty(k).mapv(|v| v * amp2);
    let omega = project_kernel(&kernel);
    let omega_psi = project_kernel(&kernel_psi);
    let omega_psi_psi = project_kernel(&kernel_psi_psi);

    let outer_gauge =
        identifiability_transform.map(|z| gam_problem::Gauge::from_block_transforms(&[z.clone()]));
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

    // The native trend block is a function penalty, so its κ derivative comes
    // from the moving center-chart Gram, not from a frozen Euclidean selector.
    // The chart matches `duchon_native_penalty_candidates` exactly:
    //
    //   B    = [alpha K_CC Z | P(C)] T,
    //   B_p  = [alpha K_CC,p Z | 0] T,
    //   B_pp = [alpha K_CC,pp Z | 0] T.
    //
    // `alpha` is the fixed numerical chart amplification used throughout the
    // existing analytic Duchon derivative surface.  The structural target is
    // the surviving polynomial subspace after T (or the explicit nonconstant
    // polynomial columns when no outer intercept constraint is present).
    let center_mean: Vec<f64> = (0..dim)
        .map(|axis| centers.column(axis).sum() / n_centers.max(1) as f64)
        .collect();
    let mut centered = centers.to_owned();
    for axis in 0..dim {
        let mean = center_mean[axis];
        centered.column_mut(axis).mapv_inplace(|value| value - mean);
    }
    let center_poly = polynomial_block_from_order(centered.view(), effective_nullspace_order);
    let kernel_center_design = fast_ab(&kernel, &z).mapv(|value| value * kernel_amp);
    let kernel_center_design_psi = fast_ab(&kernel_psi, &z).mapv(|value| value * kernel_amp);
    let kernel_center_design_psi_psi =
        fast_ab(&kernel_psi_psi, &z).mapv(|value| value * kernel_amp);
    let mut center_design = Array2::<f64>::zeros((n_centers, total_cols));
    let mut center_design_psi = Array2::<f64>::zeros((n_centers, total_cols));
    let mut center_design_psi_psi = Array2::<f64>::zeros((n_centers, total_cols));
    center_design
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&kernel_center_design);
    center_design
        .slice_mut(s![.., kernel_cols..])
        .assign(&center_poly);
    center_design_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&kernel_center_design_psi);
    center_design_psi_psi
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&kernel_center_design_psi_psi);
    let (center_design, center_design_psi, center_design_psi_psi, trend_frame) =
        if let Some(transform) = identifiability_transform {
            if transform.nrows() != total_cols {
                crate::bail_dim_basis!(
                    "Duchon identifiability transform has {} rows, expected {}",
                    transform.nrows(),
                    total_cols
                );
            }
            let kernel_coordinate_map = transform.slice(s![0..kernel_cols, ..]).to_owned();
            let (frame, _) = rrqr_nullspace_basis(
                &kernel_coordinate_map.t().to_owned(),
                default_rrqr_rank_alpha(),
            )
            .map_err(BasisError::LinalgError)?;
            (
                fast_ab(&center_design, transform),
                fast_ab(&center_design_psi, transform),
                fast_ab(&center_design_psi_psi, transform),
                frame,
            )
        } else {
            let mut frame = Array2::<f64>::zeros((total_cols, poly_cols.saturating_sub(1)));
            for column in 1..poly_cols {
                frame[[kernel_cols + column, column - 1]] = 1.0;
            }
            (
                center_design,
                center_design_psi,
                center_design_psi_psi,
                frame,
            )
        };
    let gram = symmetrize_penalty(&fast_ata(&center_design));
    let gram_psi = symmetrize_penalty(
        &(fast_atb(&center_design_psi, &center_design)
            + fast_atb(&center_design, &center_design_psi)),
    );
    let gram_psi_psi = symmetrize_penalty(
        &(fast_atb(&center_design_psi_psi, &center_design)
            + fast_atb(&center_design_psi, &center_design_psi).mapv(|value| 2.0 * value)
            + fast_atb(&center_design, &center_design_psi_psi)),
    );
    let trend_jet = function_space_subspace_shrinkage_derivatives(
        &trend_frame,
        &gram,
        &gram_psi,
        &gram_psi,
        &gram_psi_psi,
    )?;
    let (_, trend_psi_norm, trend_psi_psi_norm, _) = normalize_penaltywith_psi_derivatives(
        &trend_jet.value,
        &trend_jet.first_a,
        &trend_jet.mixed,
    );
    let candidates = duchon_native_penalty_candidates(
        centers,
        spec.length_scale,
        spec.power,
        effective_nullspace_order,
        spec.aniso_log_scales.as_deref(),
        &z,
        identifiability_transform,
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
                first.push(trend_psi_norm.clone());
                second.push(trend_psi_psi_norm.clone());
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
        None,
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
    let kernel_gauge = gam_problem::Gauge::from_block_transforms(&[z_kernel.clone()]);
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
        .map(|z| gam_problem::Gauge::from_block_transforms(&[z.clone()]))
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
        .map(|z| gam_problem::Gauge::from_block_transforms(&[z.clone()]))
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

fn matern_center_design_chart(
    kernel: &Array2<f64>,
    z_opt: Option<&Array2<f64>>,
    include_intercept: bool,
    intercept_value: f64,
) -> Array2<f64> {
    let k = kernel.nrows();
    let kernel_chart = match z_opt {
        Some(transform) => fast_ab(kernel, transform),
        None => kernel.clone(),
    };
    let mut design =
        Array2::<f64>::zeros((k, kernel_chart.ncols() + usize::from(include_intercept)));
    design
        .slice_mut(s![.., 0..kernel_chart.ncols()])
        .assign(&kernel_chart);
    if include_intercept {
        design
            .column_mut(kernel_chart.ncols())
            .fill(intercept_value);
    }
    design
}

/// Function-metric ridge jet on Matérn's frozen center measure.
///
/// The structural subspace is the explicit intercept; the kernel transform is
/// frozen and only the center-evaluation chart `K(θ)Z` moves. Product rules
/// form `(G, G_a, G_b, G_ab)` exactly, then the shared metric-projector inverse
/// rule supplies `(R, R_a, R_b, R_ab)` with no spectral-eigenvector sensitivity.
fn matern_center_function_metric_jet(
    kernel: &Array2<f64>,
    kernel_a: &Array2<f64>,
    kernel_b: &Array2<f64>,
    kernel_ab: &Array2<f64>,
    z_opt: Option<&Array2<f64>>,
    include_intercept: bool,
) -> Result<FunctionSpaceSubspaceShrinkageDerivatives, BasisError> {
    let design = matern_center_design_chart(kernel, z_opt, include_intercept, 1.0);
    let design_a = matern_center_design_chart(kernel_a, z_opt, include_intercept, 0.0);
    let design_b = matern_center_design_chart(kernel_b, z_opt, include_intercept, 0.0);
    let design_ab = matern_center_design_chart(kernel_ab, z_opt, include_intercept, 0.0);
    let gram = symmetrize_penalty(&fast_ata(&design));
    let gram_a = symmetrize_penalty(&(fast_atb(&design_a, &design) + fast_atb(&design, &design_a)));
    let gram_b = symmetrize_penalty(&(fast_atb(&design_b, &design) + fast_atb(&design, &design_b)));
    let gram_ab = symmetrize_penalty(
        &(fast_atb(&design_ab, &design)
            + fast_atb(&design_a, &design_b)
            + fast_atb(&design_b, &design_a)
            + fast_atb(&design, &design_ab)),
    );
    let p = design.ncols();
    let mut frame = Array2::<f64>::zeros((p, usize::from(include_intercept)));
    if include_intercept {
        frame[[p - 1, 0]] = 1.0;
    }
    function_space_subspace_shrinkage_derivatives(&frame, &gram, &gram_a, &gram_b, &gram_ab)
}

/// Build the Matérn double-penalty **primary** block (the projected kernel
/// Gram `A = Zᵀ K Z`, embedded into the `total_cols` coefficient space) and its
/// log-κ ψ-derivatives, in BOTH the un-normalized and the Frobenius-normalized
/// forms.
///
/// Returns normalized primary value/derivatives plus the normalized first and
/// second derivatives of the center-function-metric intercept ridge.
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

    let shrinkage_jet = matern_center_function_metric_jet(
        &kernel,
        &kernel_psi,
        &kernel_psi,
        &kernel_psi_psi,
        z_opt,
        include_intercept,
    )?;
    let (_, shrinkage_psi, shrinkage_psi_psi, _) = normalize_penaltywith_psi_derivatives(
        &shrinkage_jet.value,
        &shrinkage_jet.first_a,
        &shrinkage_jet.mixed,
    );

    let (kernel, kernel_psi, kernel_psi_psi) = if let Some(gauge) =
        z_opt.map(|z| gam_problem::Gauge::from_block_transforms(&[z.clone()]))
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
    let (s_norm, s_norm_psi, s_norm_psi_psi, _) =
        normalize_penaltywith_psi_derivatives(&s, &s_psi, &s_psi_psi);
    Ok((
        s_norm,
        s_norm_psi,
        s_norm_psi_psi,
        shrinkage_psi,
        shrinkage_psi_psi,
    ))
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
    //
    // The ψ-derivatives MUST be built over the EXACT same realized geometry as
    // the value build (`build_matern_basiswithworkspace`): the value build
    // rank-reduces an over-specified center set (`matern_rank_reduce_centers`,
    // #755), periodic-expands it, and resolves the anisotropy/identifiability
    // transform over the surviving centers. Re-deriving the centers here from
    // `select_centers_by_strategy` (un-reduced) produced a derivative penalty
    // sized to the FULL center set while `base.penaltyinfo` (the active-block
    // mask + the forward penalty list) is sized to the REDUCED set — an
    // index/shape desync that crashed the double-penalty ψ-derivative assembly
    // with an IncompatibleShape matmul and left the FD audit comparing
    // differently-shaped blocks (the matern double-penalty log-κ FD tests). Pull
    // the realized centers, transform, and anisotropy from the value build so the
    // two are byte-consistent by construction.
    let base = build_matern_basiswithworkspace(data, spec, workspace)?;
    let (base_centers, base_transform, base_aniso) = match &base.metadata {
        BasisMetadata::Matern {
            centers,
            identifiability_transform,
            aniso_log_scales,
            ..
        } => (
            centers.clone(),
            identifiability_transform.clone(),
            aniso_log_scales.clone(),
        ),
        other => {
            return Err(BasisError::InvalidInput(format!(
                "Matérn ψ-derivative build expected Matérn metadata, got {:?}",
                std::mem::discriminant(other)
            )));
        }
    };
    // Reproduce the value build's periodic expansion of the (reduced) base
    // centers so the kernel pairs and the realized design columns align.
    let centers = expand_periodic_centers(&base_centers, spec.periodic.as_deref())?;
    let z_opt = base_transform;
    let aniso = base_aniso.as_deref();
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
        let (_, primary_derivative, primarysecond_derivative, shrinkage_first, shrinkagesecond) =
            build_matern_double_penalty_primarywith_psi_derivatives(
                centers.view(),
                spec.length_scale,
                spec.nu,
                spec.include_intercept,
                z_opt.as_ref(),
                aniso,
            )?;
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
    if spec.aniso_log_scales.is_none() {
        return Err(BasisError::InvalidInput(
            "aniso derivatives require aniso_log_scales to be set".to_string(),
        ));
    }
    let dim = data.ncols();

    // gam#1376 / #755 — derive the κ-derivative geometry from the REALIZED value
    // build, NOT from `select_centers_by_strategy(spec)`. The value build
    // (`build_matern_basiswithworkspace`) rank-REDUCES an over-specified center
    // set over this data cloud (`matern_rank_reduce_centers`) and periodic-EXPANDS
    // it; re-selecting raw centers here produced a derivative sized to the FULL
    // center set while `base.penaltyinfo` / the realized design columns are sized
    // to the REDUCED+expanded set — a shape desync that the κ-gradient consumer
    // (`design_construction.rs` aniso entry) silently drops on a column-count
    // mismatch, disabling the analytic aniso κ-gradient for any rank-reduced or
    // periodic fit. Mirror the iso builder
    // (`build_matern_basis_log_kappa_derivativeswithworkspace`): pull the realized
    // centers / identifiability transform / resolved anisotropy from the value
    // build's metadata so the two are byte-consistent by construction.
    let base = build_matern_basis_seeded(
        data,
        spec,
        &mut BasisWorkspace::default(),
        AnisoSeedMode::Literal,
    )?;
    let (base_centers, z_opt, base_aniso) = match &base.metadata {
        BasisMetadata::Matern {
            centers,
            identifiability_transform,
            aniso_log_scales,
            ..
        } => (
            centers.clone(),
            identifiability_transform.clone(),
            aniso_log_scales.clone(),
        ),
        other => {
            return Err(BasisError::InvalidInput(format!(
                "Matérn aniso ψ-derivative build expected Matérn metadata, got {:?}",
                std::mem::discriminant(other)
            )));
        }
    };
    // Reproduce the value build's periodic expansion of the (reduced) base centers.
    let centers = expand_periodic_centers(&base_centers, spec.periodic.as_deref())?;
    let eta = base_aniso.as_deref().ok_or_else(|| {
        BasisError::InvalidInput(
            "aniso derivatives require resolved aniso_log_scales from the value build".to_string(),
        )
    })?;
    if eta.len() != dim {
        crate::bail_dim_basis!(
            "resolved aniso_log_scales length {} != data dimension {dim}",
            eta.len()
        );
    }

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
        let mut primary_first_raw = vec![Array2::<f64>::zeros((total_cols, total_cols)); dim];
        let mut primary_second_diag_raw = vec![Array2::<f64>::zeros((total_cols, total_cols)); dim];
        let coefficient_gauge = z_opt
            .as_ref()
            .map(|z| gam_problem::Gauge::from_block_transforms(&[z.clone()]));
        let (raw_first, raw_second_diag) = build_matern_aniso_primary_raw_derivative_matrices(
            centers.view(),
            eta,
            spec.length_scale,
            spec.nu,
        )?;
        for a in 0..dim {
            let projected_first = if let Some(gauge) = coefficient_gauge.as_ref() {
                gauge.restrict_penalty(&raw_first[a])
            } else {
                raw_first[a].clone()
            };
            let projected_second = if let Some(gauge) = coefficient_gauge.as_ref() {
                gauge.restrict_penalty(&raw_second_diag[a])
            } else {
                raw_second_diag[a].clone()
            };
            primary_first_raw[a]
                .slice_mut(s![0..kernel_cols, 0..kernel_cols])
                .assign(&projected_first);
            primary_second_diag_raw[a]
                .slice_mut(s![0..kernel_cols, 0..kernel_cols])
                .assign(&projected_second);
        }
        let mut dp_cross_pairs: Vec<(usize, usize)> = Vec::new();
        for a in 0..dim {
            for b in (a + 1)..dim {
                dp_cross_pairs.push((a, b));
            }
        }

        // Reuse the value build already constructed at the top of this function
        // (its metadata seeded the realized geometry) — `base.penaltyinfo` is the
        // active-block mask sized to the realized (reduced) basis.
        let has_shrinkage = base.penaltyinfo.iter().any(|info| {
            info.active && matches!(info.source, PenaltySource::DoublePenaltyNullspace)
        });
        // The value path emits each block after Frobenius normalization. Keep
        // the raw center kernel and its exact axis derivatives for both the
        // primary RKHS block and the moving function-metric intercept ridge.
        let kernel = build_matern_kernel_penalty(
            centers.view(),
            spec.length_scale,
            spec.nu,
            spec.include_intercept,
            Some(eta),
        )?;
        let kblock = kernel.slice(s![0..k, 0..k]).to_owned();
        let projected = if let Some(gauge) = coefficient_gauge.as_ref() {
            gauge.restrict_penalty(&kblock)
        } else {
            kblock
        };
        let mut a_raw = Array2::<f64>::zeros((total_cols, total_cols));
        a_raw
            .slice_mut(s![0..kernel_cols, 0..kernel_cols])
            .assign(&projected);
        let mut primary_first = Vec::with_capacity(dim);
        let mut primary_second_diag = Vec::with_capacity(dim);
        for a in 0..dim {
            let (_, first, second, _) = normalize_penaltywith_psi_derivatives(
                &a_raw,
                &primary_first_raw[a],
                &primary_second_diag_raw[a],
            );
            primary_first.push(first);
            primary_second_diag.push(second);
        }
        let mut shrinkage_first = Vec::with_capacity(dim);
        let mut shrinkage_second_diag = Vec::with_capacity(dim);
        for a in 0..dim {
            if has_shrinkage {
                let jet = matern_center_function_metric_jet(
                    &kernel.slice(s![0..k, 0..k]).to_owned(),
                    &raw_first[a],
                    &raw_first[a],
                    &raw_second_diag[a],
                    z_opt.as_ref(),
                    spec.include_intercept,
                )?;
                let (_, first, second, _) =
                    normalize_penaltywith_psi_derivatives(&jet.value, &jet.first_a, &jet.mixed);
                shrinkage_first.push(first);
                shrinkage_second_diag.push(second);
            } else {
                shrinkage_first.push(Array2::<f64>::zeros((total_cols, total_cols)));
                shrinkage_second_diag.push(Array2::<f64>::zeros((total_cols, total_cols)));
            }
        }
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
            .map(|z| gam_problem::Gauge::from_block_transforms(&[z.clone()]));
        let penaltyinfo = base.penaltyinfo.clone();
        let length_scale = spec.length_scale;
        let nu = spec.nu;
        let primary_first_raw_owned = primary_first_raw.clone();
        let a_raw_owned = a_raw.clone();
        let kernel_block_owned = kernel.slice(s![0..k, 0..k]).to_owned();
        let kernel_first_owned = raw_first.clone();
        let z_owned = z_opt.clone();
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
                let raw_cross_for_metric = raw_cross.clone();
                let projected: Array2<f64> = if let Some(gauge) = gauge_owned.as_ref() {
                    gauge.restrict_penalty(&raw_cross)
                } else {
                    raw_cross
                };
                let mut padded = Array2::<f64>::zeros((total_cols, total_cols));
                padded
                    .slice_mut(s![0..kernel_cols, 0..kernel_cols])
                    .assign(&projected);
                let primary_cross = normalize_penalty_cross_psi_derivative(
                    &a_raw_owned,
                    &primary_first_raw_owned[a],
                    &primary_first_raw_owned[b],
                    &padded,
                    trace_of_product(&a_raw_owned, &a_raw_owned).sqrt(),
                );
                // Exact mixed derivative of the function-metric ridge. The
                // structural intercept frame is fixed; only the compact center
                // Gram moves with the two anisotropy axes.
                let shrinkage_cross = if penaltyinfo.iter().any(|info| {
                    info.active && matches!(info.source, PenaltySource::DoublePenaltyNullspace)
                }) {
                    let jet = matern_center_function_metric_jet(
                        &kernel_block_owned,
                        &kernel_first_owned[a],
                        &kernel_first_owned[b],
                        &raw_cross_for_metric,
                        z_owned.as_ref(),
                        include_intercept,
                    )?;
                    normalize_penalty_cross_psi_derivative(
                        &jet.value,
                        &jet.first_a,
                        &jet.first_b,
                        &jet.mixed,
                        trace_of_product(&jet.value, &jet.value).sqrt(),
                    )
                } else {
                    Array2::<f64>::zeros((total_cols, total_cols))
                };
                active_matern_double_penalty_derivatives(
                    &penaltyinfo,
                    &primary_cross,
                    &shrinkage_cross,
                )
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

    // gam#1376 — NO cross-axis centering of the per-axis ψ derivatives.
    //
    // The κ-optimizer's per-axis coordinate is the RAW `psi_a`, decoded by
    // `spatial_term_psi_to_length_scale_and_aniso` into BOTH halves of the
    // metric at once: the global length scale `ℓ = exp(−mean(psi))` AND the
    // centered contrast `eta_a = psi_a − mean(psi)` (which is then passed as
    // `aniso_log_scales`, already mean-zero). In the kernel argument these two
    // recombine. The Matérn design uses `x = r/ℓ` with the centered-contrast
    // distance `r = √(Σ_a exp(2·eta_a)·h_a²)` (`aniso_distance_and_components`,
    // `centered_aniso_metric_weights`), so
    //
    //   x² = r²/ℓ² = Σ_a exp(2·(psi_a − mean(psi)))·exp(2·mean(psi))·h_a²
    //              = Σ_a exp(2·psi_a)·h_a²,
    //
    // i.e. the `mean(psi)` cancels EXACTLY and the effective per-axis exponent
    // is the raw `psi_a`. Therefore the criterion derivative w.r.t. the
    // optimizer coordinate `psi_a` is the NATIVE, un-centered per-axis ψ
    // derivative `∂φ/∂psi_a = q·s_a` that the per-axis builders already produce
    // (`design_first`, `penalties_first`, `penalties_second_diag`, and the cross
    // provider).
    //
    // An earlier #1376 change installed the centering projection
    // `P = I − 11ᵀ/d` (∂F/∂η_a = ∂F/∂ψ_a − (1/d)Σ_b ∂F/∂ψ_b) on the reasoning
    // that a uniform η-shift is a no-op of the centered metric. That reasoning
    // omits the length-scale half of the coordinate map: a uniform shift of the
    // optimizer coordinate `psi` is NOT a no-op — it rescales `ℓ`. P annihilated
    // the all-ones (global-scale) direction, making the analytic outer gradient
    // sum-zero / antisymmetric while the FD of the full criterion (which moves
    // raw `psi`, hence both `ℓ` and the contrast) is not, yielding the rel≈0.85
    // FD gap. Removing the projection restores agreement; the per-axis contrast
    // `g_signal − g_noise` (an FD-visible descent direction) is unchanged by the
    // removal, since centering only subtracts a common mean from every axis.

    Ok(result)
}

#[cfg(test)]
mod wahba_penalty_invariants_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn wahba_primary_is_exact_function_seminorm_without_coefficient_ridge() {
        let data = array![
            [-1.1, 0.1],
            [-0.7, 1.0],
            [-0.2, 2.0],
            [0.3, 2.8],
            [0.8, -2.5],
            [1.1, -1.4],
            [0.5, -0.4],
            [-0.4, -1.8],
        ];
        let spec = SphericalSplineBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
            penalty_order: 2,
            double_penalty: false,
            radians: true,
            method: SphereMethod::Wahba,
            max_degree: None,
            wahba_kernel: SphereWahbaKernel::Sobolev,
            identifiability: SphericalSplineIdentifiability::CenterSumToZero,
        };
        let built = build_spherical_spline_basis(data.view(), &spec).expect("Wahba basis");
        assert_eq!(built.penalties.len(), 1);
        let BasisMetadata::Sphere { centers, .. } = &built.metadata else {
            panic!("Wahba builder must return spherical metadata");
        };

        let center_kernel = spherical_wahba_kernel_matrix_with_kind(
            centers.view(),
            centers.view(),
            spec.penalty_order,
            spec.radians,
            spec.wahba_kernel,
        )
        .expect("center kernel");
        let decomposition =
            wahba_low_degree_decomposition(centers.view(), spec.radians, center_kernel.view())
                .expect("low-degree decomposition");
        let exact_raw = build_wahba_decomposed_penalty(center_kernel.view(), &decomposition);
        let (expected, _) = normalize_penalty(&exact_raw);

        let observed = &built.penalties[0];
        assert_eq!(observed.dim(), expected.dim());
        let error = observed
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            error <= 1.0e-12,
            "primary Wahba penalty must be exactly the normalized RKHS seminorm; max error={error:.3e}"
        );
    }

    #[test]
    fn wahba_null_shrinkage_is_center_function_metric_and_chart_covariant() {
        let centers = array![
            [-1.1, 0.1],
            [-0.7, 1.0],
            [-0.2, 2.0],
            [0.3, 2.8],
            [0.8, -2.5],
            [1.1, -1.4],
            [0.5, -0.4],
            [-0.4, -1.8],
        ];
        let kernel = spherical_wahba_kernel_matrix_with_kind(
            centers.view(),
            centers.view(),
            2,
            true,
            SphereWahbaKernel::Sobolev,
        )
        .expect("center kernel");
        let decomposition = wahba_low_degree_decomposition(centers.view(), true, kernel.view())
            .expect("low-degree decomposition");
        let design =
            build_wahba_decomposed_design(kernel.view(), centers.view(), true, &decomposition);
        let penalty = build_wahba_decomposed_penalty(kernel.view(), &decomposition);
        let gram = symmetrize_penalty(&fast_ata(&design));
        let ridge = function_space_nullspace_shrinkage(&penalty, &gram)
            .expect("metric construction")
            .expect("Wahba low-degree null space");

        // The decomposed Wahba chart is `[kernel | low degree]`, and the
        // primary RKHS seminorm has an exact zero block on the appended
        // low-degree columns.  Exercise those structural null directions
        // directly instead of reaching into the sibling B-spline module's
        // private generalized-eigensolver (which is an implementation detail
        // of the ridge constructor being tested).
        let low_degree = decomposition.low_degree_cols;
        assert!(low_degree > 0, "fixture must retain low-degree harmonics");
        let mut null = Array2::<f64>::zeros((penalty.nrows(), low_degree));
        let low_degree_start = penalty.nrows() - low_degree;
        for column in 0..low_degree {
            null[[low_degree_start + column, column]] = 1.0;
        }
        let metric_action_error = (&ridge.dot(&null) - &gram.dot(&null))
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            metric_action_error < 2.0e-9,
            "ridge must equal the function metric on null directions; error={metric_action_error:.3e}"
        );

        let p = penalty.nrows();
        let mut transform = Array2::<f64>::eye(p);
        for j in 0..p {
            transform[[j, j]] = if j % 2 == 0 { 0.2 } else { 3.0 };
            if j + 1 < p {
                transform[[j, j + 1]] = 0.07 * (j + 1) as f64;
            }
        }
        let congruence = |matrix: &Array2<f64>| fast_atb(&transform, &fast_ab(matrix, &transform));
        let transformed =
            function_space_nullspace_shrinkage(&congruence(&penalty), &congruence(&gram))
                .expect("transformed metric construction")
                .expect("transformed null space");
        let expected = congruence(&ridge);
        let covariance_error = (&transformed - &expected)
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            covariance_error < 2.0e-8,
            "Wahba null ridge changed under a harmless basis chart; error={covariance_error:.3e}"
        );
    }

    #[test]
    fn wahba_low_degree_chart_satisfies_exact_native_gauge() {
        let centers = array![
            [-1.1, 0.1],
            [-0.7, 1.0],
            [-0.2, 2.0],
            [0.3, 2.8],
            [0.8, -2.5],
            [1.1, -1.4],
            [0.5, -0.4],
            [-0.4, -1.8],
        ];
        let center_kernel = spherical_wahba_kernel_matrix_with_kind(
            centers.view(),
            centers.view(),
            2,
            true,
            SphereWahbaKernel::Sobolev,
        )
        .expect("center kernel");
        let decomposition =
            wahba_low_degree_decomposition(centers.view(), true, center_kernel.view())
                .expect("low-degree decomposition");
        let low = decomposition
            .low_degree_centers
            .as_ref()
            .expect("degree-1 harmonics are supported");
        let projection = decomposition
            .kernel_low_projection
            .as_ref()
            .expect("kernel/low projection");

        assert_eq!(
            decomposition.kernel_basis.ncols() + low.ncols(),
            centers.nrows(),
            "native gauge must remove exactly the supported harmonic rank"
        );
        let gauge_residual = low.t().dot(&decomposition.kernel_basis);
        let centered_kernel = center_kernel.dot(&decomposition.kernel_basis) - low.dot(projection);
        let design_residual = low.t().dot(&centered_kernel);
        let max_abs = |matrix: &Array2<f64>| {
            matrix
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max)
        };
        let roundoff_factor = gam_linalg::faer_ndarray::default_rrqr_rank_alpha()
            * f64::EPSILON
            * centers.nrows() as f64;
        let gauge_roundoff =
            roundoff_factor * max_abs(low).max(1.0) * max_abs(&decomposition.kernel_basis).max(1.0);
        let design_roundoff =
            roundoff_factor * max_abs(low).max(1.0) * max_abs(&centered_kernel).max(1.0);
        assert!(
            max_abs(&gauge_residual) <= gauge_roundoff,
            "Wahba kernel chart must satisfy H^T Z = 0 without a ridge; residual={:.3e}, bound={gauge_roundoff:.3e}",
            max_abs(&gauge_residual)
        );
        assert!(
            max_abs(&design_residual) <= design_roundoff,
            "centered kernel design must be orthogonal to H; residual={:.3e}, bound={design_roundoff:.3e}",
            max_abs(&design_residual)
        );
    }

    #[test]
    fn wahba_native_gauge_rank_adapts_at_minimum_center_counts() {
        let center_sets = vec![
            array![[-0.6, -0.4], [0.8, 1.7]],
            array![[-0.8, -0.2], [0.1, 2.1], [0.9, -1.3]],
        ];
        for centers in center_sets {
            let center_kernel = spherical_wahba_kernel_matrix_with_kind(
                centers.view(),
                centers.view(),
                2,
                true,
                SphereWahbaKernel::Sobolev,
            )
            .expect("center kernel");
            let decomposition =
                wahba_low_degree_decomposition(centers.view(), true, center_kernel.view())
                    .expect("minimum-center decomposition");
            assert_eq!(decomposition.low_degree_cols, centers.nrows());
            assert_eq!(decomposition.kernel_basis.ncols(), 0);
            assert_eq!(
                decomposition
                    .kernel_low_projection
                    .as_ref()
                    .expect("empty kernel projection")
                    .dim(),
                (centers.nrows(), 0)
            );
            let design = build_wahba_decomposed_design(
                center_kernel.view(),
                centers.view(),
                true,
                &decomposition,
            );
            assert_eq!(design.dim(), (centers.nrows(), centers.nrows()));
            assert!(design.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn wahba_great_circle_rank_adaptation_replays_selected_jet_columns() {
        let centers = array![[0.0, -2.4], [0.0, -1.1], [0.0, 0.2], [0.0, 1.5], [0.0, 2.8],];
        let center_kernel = spherical_wahba_kernel_matrix_with_kind(
            centers.view(),
            centers.view(),
            2,
            true,
            SphereWahbaKernel::Sobolev,
        )
        .expect("center kernel");
        let decomposition =
            wahba_low_degree_decomposition(centers.view(), true, center_kernel.view())
                .expect("great-circle decomposition");
        assert_eq!(decomposition.low_degree_cols, 2);
        assert_eq!(decomposition.kernel_basis.ncols(), centers.nrows() - 2);

        let points = array![[-0.4, -0.8], [0.7, 1.2]];
        let raw_kernel_jet = spherical_wahba_kernel_jet_with_kind(
            points.view(),
            centers.view(),
            2,
            true,
            SphereWahbaKernel::Sobolev,
        )
        .expect("raw kernel jet");
        let complete_low_jet =
            spherical_harmonic_jet(points.view(), SPHERE_UNPENALIZED_LOW_DEGREE, true)
                .expect("complete harmonic jet");
        let selected_low_jet = complete_low_jet.select(Axis(1), &decomposition.low_degree_columns);
        let decomposed =
            build_wahba_decomposed_jet(&raw_kernel_jet, Some(&complete_low_jet), &decomposition);
        let kernel_cols = decomposition.kernel_basis.ncols();
        for axis in 0..2 {
            let observed = decomposed.slice(s![.., kernel_cols.., axis]);
            let expected = selected_low_jet.slice(s![.., .., axis]);
            assert_eq!(observed, expected);
        }

        let near_great_circle = array![
            [-1.0e-10, -2.4],
            [0.0, -1.1],
            [1.0e-10, 0.2],
            [-1.0e-10, 1.5],
            [1.0e-10, 2.8],
        ];
        let near_kernel = spherical_wahba_kernel_matrix_with_kind(
            near_great_circle.view(),
            near_great_circle.view(),
            2,
            true,
            SphereWahbaKernel::Sobolev,
        )
        .expect("near-great-circle kernel");
        let near =
            wahba_low_degree_decomposition(near_great_circle.view(), true, near_kernel.view())
                .expect("QR projection must remain defined near the rank boundary");
        assert!(
            near.kernel_low_projection
                .as_ref()
                .expect("near-great-circle projection")
                .iter()
                .all(|value| value.is_finite())
        );
    }
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

        // The CURRENT harmonic builder (see build_spherical_harmonic_basis) uses
        // a FULL-RANK Laplace–Beltrami curvature penalty: the harmonic basis
        // starts at degree l=1 (there is no degree-0/constant column), and every
        // column carries the strictly-positive eigenvalue [l(l+1)]^order. There
        // is therefore NO primary null space, and the double penalty is a UNIFORM
        // isotropic ridge over ALL columns (Frobenius-normalized). The earlier
        // expectation — that degree-≤`SPHERE_UNPENALIZED_LOW_DEGREE` columns are
        // primary-null and shrink-penalized while higher degrees are the reverse
        // — pinned the Wahba low-degree-nullspace structure, which this Harmonic
        // method does not use. (`SPHERE_UNPENALIZED_LOW_DEGREE` governs the Wahba
        // decomposition only.) Assert the real Harmonic contract instead.
        let l_max = spec.max_degree.unwrap();
        let expected_p = l_max * (l_max + 2);
        assert_eq!(primary.ncols(), expected_p);

        // Reconstruct the per-column degree (block l has 2l+1 modes, l = 1..=l_max).
        let mut col_degree = Vec::with_capacity(expected_p);
        for l in 1..=l_max {
            for _ in 0..(2 * l + 1) {
                col_degree.push(l);
            }
        }
        assert_eq!(col_degree.len(), expected_p);

        for col in 0..primary.ncols() {
            let primary_diag = primary[[col, col]];
            let shrink_diag = shrink[[col, col]];
            // Every column carries strictly-positive curvature roughness.
            assert!(
                primary_diag > 0.0,
                "column {col} (degree {}) must carry positive curvature roughness, got {primary_diag}",
                col_degree[col]
            );
            // Curvature penalty equals the Laplace–Beltrami eigenvalue
            // [l(l+1)]^order on its diagonal.
            let l = col_degree[col] as f64;
            let eig = (l * (l + 1.0)).powi(spec.penalty_order as i32);
            assert!(
                (primary_diag - eig).abs() <= 1e-9 * eig,
                "column {col} primary diagonal must be [l(l+1)]^order={eig}, got {primary_diag}"
            );
            // The shrink ridge is uniform and positive on every column.
            assert!(
                shrink_diag > 0.0,
                "column {col} must be shrink-penalized by the uniform ridge, got {shrink_diag}"
            );
        }

        // The shrink ridge must be the SAME on every column (a uniform isotropic
        // ridge, not a degree-selective one).
        let ridge0 = shrink[[0, 0]];
        for col in 1..shrink.ncols() {
            assert!(
                (shrink[[col, col]] - ridge0).abs() <= 1e-12 * ridge0.max(1.0),
                "shrink ridge must be uniform; column {col} = {} differs from column 0 = {ridge0}",
                shrink[[col, col]]
            );
        }
    }
}
