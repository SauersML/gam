use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ConstraintNullspaceCacheKey {
    pub(crate) centersrows: usize,
    pub(crate) centers_cols: usize,
    pub(crate) centers_hash: u64,
    pub(crate) order: ConstraintNullspaceOrderKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ConstraintNullspaceOrderKey {
    Duchon(DuchonNullspaceOrder),
    ThinPlate,
}

#[derive(Default, Clone, Debug)]
pub(crate) struct ConstraintNullspaceCache {
    pub(crate) map: HashMap<ConstraintNullspaceCacheKey, Arc<Array2<f64>>>,
    pub(crate) order: Vec<ConstraintNullspaceCacheKey>,
}

pub(crate) const CONSTRAINT_NULLSPACE_CACHE_MAX_ENTRIES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct OwnedDataCacheKey {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) ptr: usize,
    pub(crate) stride0: isize,
    pub(crate) stride1: isize,
}

#[derive(Debug)]
pub(crate) struct BasisCacheContext {
    pub(crate) constraint_nullspace: ConstraintNullspaceCache,
    pub(crate) owned_data: gam_runtime::resource::ByteLruCache<OwnedDataCacheKey, Arc<Array2<f64>>>,
}

impl BasisCacheContext {
    pub(crate) fn with_policy(policy: &gam_runtime::resource::ResourcePolicy) -> Self {
        Self {
            constraint_nullspace: ConstraintNullspaceCache::default(),
            owned_data: gam_runtime::resource::ByteLruCache::with_max_entries(
                policy.max_owned_data_cache_bytes,
                gam_runtime::resource::OWNED_DATA_CACHE_MAX_ENTRIES,
            ),
        }
    }
}

impl Default for BasisCacheContext {
    fn default() -> Self {
        Self::with_policy(&gam_runtime::resource::ResourcePolicy::default_library())
    }
}

/// Explicit per-run workspace for reusable basis-construction caches.
///
/// Pass one workspace through repeated basis builds to avoid global mutable state
/// and to keep caching scoped to a caller-controlled lifecycle.
///
/// Owned-data cache entries are byte-limited via the
/// [`gam_runtime::resource::ResourcePolicy`] provided at construction; use
/// [`BasisWorkspace::with_policy`] for large-scale workloads where a single
/// entry can be multiple gigabytes.
#[derive(Debug)]
pub struct BasisWorkspace {
    pub(crate) cache: BasisCacheContext,
    pub(crate) policy: gam_runtime::resource::ResourcePolicy,
}

impl BasisWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_policy(policy: gam_runtime::resource::ResourcePolicy) -> Self {
        Self {
            cache: BasisCacheContext::with_policy(&policy),
            policy,
        }
    }

    pub fn default_library() -> Self {
        Self::with_policy(gam_runtime::resource::ResourcePolicy::default_library())
    }

    /// Returns the resource policy this workspace was configured with.
    pub fn policy(&self) -> &gam_runtime::resource::ResourcePolicy {
        &self.policy
    }
}

impl Default for BasisWorkspace {
    fn default() -> Self {
        Self::default_library()
    }
}

pub(crate) fn hash_arrayview2(values: ArrayView2<'_, f64>) -> u64 {
    let mut hasher = DefaultHasher::new();
    values.nrows().hash(&mut hasher);
    values.ncols().hash(&mut hasher);
    for v in values {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

pub(crate) fn shared_owned_data_matrix(
    data: ArrayView2<'_, f64>,
    cache: &BasisCacheContext,
) -> Arc<Array2<f64>> {
    let key = OwnedDataCacheKey {
        rows: data.nrows(),
        cols: data.ncols(),
        ptr: data.as_ptr() as usize,
        stride0: data.strides()[0],
        stride1: data.strides()[1],
    };
    if let Some(hit) = cache.owned_data.get(&key) {
        return hit;
    }

    let owned = Arc::new(data.to_owned());
    if let Some(hit) = cache.owned_data.get(&key) {
        return hit;
    }

    cache.owned_data.insert(key, owned.clone());
    owned
}

/// Minimal cache-less intern: wraps an `ArrayView2` into an `Arc<Array2<f64>>`.
///
/// Used by derivative-operator builders that don't have a `BasisCacheContext`
/// in scope (e.g. `build_aniso_design_psi_derivatives_shared`). The goal is the
/// same as `shared_owned_data_matrix`: move the owned payload into an `Arc`
/// once so that downstream `StreamingRadialState` copies share it via
/// `Arc::clone` instead of materializing a fresh n×d `Array2<f64>` per axis.
#[inline]
pub(crate) fn shared_owned_data_matrix_from_view(data: ArrayView2<'_, f64>) -> Arc<Array2<f64>> {
    Arc::new(data.to_owned())
}

/// Minimal cache-less intern for knot centers; mirrors
/// `shared_owned_data_matrix_from_view`. Centers are typically k×d with k
/// much smaller than n, but the `Arc::clone` pattern still avoids a k×d
/// copy per axis when the same operator feeds multiple derivative paths.
#[inline]
pub(crate) fn shared_owned_centers_matrix_from_view(
    centers: ArrayView2<'_, f64>,
) -> Arc<Array2<f64>> {
    Arc::new(centers.to_owned())
}

/// Compute the kernel reparameterisation transform `Z = null(P_centers^T)`.
///
/// `Z` is a `(k, k − C(d+r, r))` orthonormal matrix whose columns span the
/// null space of the polynomial side-condition system.  Reparameterising the
/// radial kernel coefficients as `α = Z γ` enforces `P_centers^T α = 0` and
/// reduces the kernel column count from `k` to `k − C(d+r, r)`.
///
/// After this projection the polynomial block `P_data` is appended as separate
/// explicit unpenalized columns (see `build_duchon_basis_designwithworkspace`),
/// so the pre-identifiability total width is always `k` (equal to the center
/// count), regardless of the polynomial null-space dimension.
///
/// This is the step that absorbs the full `C(d+r, r)`-dimensional polynomial
/// null space.  The subsequent `spatial_parametric_constraint_block` step only
/// removes the intercept.
pub(crate) fn kernel_constraint_nullspace(
    centers: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
    cache: &mut BasisCacheContext,
) -> Result<Array2<f64>, BasisError> {
    let effective_order = duchon_effective_nullspace_order(centers, order);
    let degraded = effective_order != order;
    // Translation-invariant side-condition frame (#1375, mirroring the #1269 tp
    // fix). `Z = null(P(centers)ᵀ)` is mathematically invariant to subtracting a
    // per-axis constant from `centers` (the polynomial columns `{1, x, …}` and
    // `{1, x − x̄, …}` span the same space, so `P` has the same column space and
    // `P^T` the same null space), but the RRQR pivoting that materialises `Z`
    // drifts under a large coordinate mean — landing on a different orthonormal
    // basis of the SAME null space, which would desync the design `K·Z` from the
    // penalty `ZᵀK_CC Z` across a covariate translation. Subtract the center-cloud
    // per-axis mean so the factorisation is location-standardized; both a raw and
    // an already-centered caller then produce bit-identical `Z`. The mean is a
    // fixed property of the (frozen `UserProvided`) centers, replayed identically
    // at predict.
    let k = centers.nrows();
    let d = centers.ncols();
    let center_mean: Vec<f64> = (0..d)
        .map(|c| centers.column(c).sum() / (k.max(1) as f64))
        .collect();
    let mut centers_centered = centers.to_owned();
    for c in 0..d {
        let mu = center_mean[c];
        centers_centered.column_mut(c).mapv_inplace(|v| v - mu);
    }
    let centers = centers_centered.view();
    let key = ConstraintNullspaceCacheKey {
        centersrows: centers.nrows(),
        centers_cols: centers.ncols(),
        centers_hash: hash_arrayview2(centers),
        order: ConstraintNullspaceOrderKey::Duchon(effective_order),
    };

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }

    let p_k = polynomial_block_from_order(centers, effective_order);
    let z = Arc::new(kernel_constraint_nullspace_from_matrix(p_k.view()).map_err(|err| {
        if degraded {
            BasisError::InvalidInput(format!(
                "Duchon degraded from order={:?} to order={:?} due to insufficient centers ({} in dim={}); order={:?} construction then failed: {err}",
                order,
                effective_order,
                centers.nrows(),
                centers.ncols(),
                effective_order,
            ))
        } else {
            err
        }
    })?);

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }
    cache.constraint_nullspace.map.insert(key, z.clone());
    cache.constraint_nullspace.order.push(key);
    while cache.constraint_nullspace.map.len() > CONSTRAINT_NULLSPACE_CACHE_MAX_ENTRIES {
        if cache.constraint_nullspace.order.is_empty() {
            break;
        }
        let oldkey = cache.constraint_nullspace.order.remove(0);
        cache.constraint_nullspace.map.remove(&oldkey);
    }

    Ok((*z).clone())
}

pub(crate) fn thin_plate_kernel_constraint_nullspace(
    centers: ArrayView2<'_, f64>,
    cache: &mut BasisCacheContext,
) -> Result<Array2<f64>, BasisError> {
    let key = ConstraintNullspaceCacheKey {
        centersrows: centers.nrows(),
        centers_cols: centers.ncols(),
        centers_hash: hash_arrayview2(centers),
        order: ConstraintNullspaceOrderKey::ThinPlate,
    };

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }

    let p_k = thin_plate_polynomial_block(centers);
    if centers.nrows() < p_k.ncols() {
        crate::bail_invalid_basis!(
            "thin-plate spline requires at least {} centers to span the degree-{} polynomial null space in dimension {}; got {}",
            p_k.ncols(),
            thin_plate_polynomial_degree(centers.ncols()),
            centers.ncols(),
            centers.nrows()
        );
    }
    let (z, rank) =
        rrqr_nullspace_basis(&p_k, default_rrqr_rank_alpha()).map_err(BasisError::LinalgError)?;
    if rank != p_k.ncols() {
        crate::bail_invalid_basis!(
            "thin-plate spline polynomial block is rank deficient at the selected centers: expected rank {}, got {}; choose geometrically independent centers for dimension {}",
            p_k.ncols(),
            rank,
            centers.ncols()
        );
    }
    let z = Arc::new(z);

    if let Some(hit) = cache.constraint_nullspace.map.get(&key) {
        return Ok((**hit).clone());
    }
    cache.constraint_nullspace.map.insert(key, z.clone());
    cache.constraint_nullspace.order.push(key);
    while cache.constraint_nullspace.map.len() > CONSTRAINT_NULLSPACE_CACHE_MAX_ENTRIES {
        if cache.constraint_nullspace.order.is_empty() {
            break;
        }
        let oldkey = cache.constraint_nullspace.order.remove(0);
        cache.constraint_nullspace.map.remove(&oldkey);
    }

    Ok((*z).clone())
}

pub(crate) fn matern_identifiability_transform(
    centers: ArrayView2<'_, f64>,
    identifiability: &MaternIdentifiability,
) -> Result<Option<Array2<f64>>, BasisError> {
    let k = centers.nrows();
    match identifiability {
        MaternIdentifiability::None => Ok(None),
        MaternIdentifiability::CenterSumToZero => {
            let q = Array2::<f64>::ones((k, 1));
            Ok(Some(kernel_constraint_nullspace_from_matrix(q.view())?))
        }
        MaternIdentifiability::CenterLinearOrthogonal => {
            // Mirror the Duchon path: auto-degrade to Zero (constant-only) when
            // there aren't enough centers to affinely span [1, x_1, ..., x_d].
            // kernel_constraint_nullspace_from_matrix would otherwise hard-error
            // via rrqr_nullspace_basis when centers.nrows() < d + 1.
            let effective_order =
                duchon_effective_nullspace_order(centers, DuchonNullspaceOrder::Linear);
            let q = polynomial_block_from_order(centers, effective_order);
            Ok(Some(kernel_constraint_nullspace_from_matrix(q.view())?))
        }
        MaternIdentifiability::FrozenTransform { transform, .. } => {
            if transform.nrows() != k {
                crate::bail_dim_basis!(
                    "frozen Matérn identifiability transform mismatch: centers={k}, transform rows={}",
                    transform.nrows()
                );
            }
            Ok(Some(transform.clone()))
        }
    }
}

pub(crate) fn build_matern_operator_penalty_candidates(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
    aniso_log_scales: Option<&[f64]>,
) -> Result<Vec<PenaltyCandidate>, BasisError> {
    let ops = build_matern_collocation_operator_matrices(
        centers,
        None,
        length_scale,
        nu,
        include_intercept,
        z_opt.map(|z| z.view()),
        aniso_log_scales,
    )?;
    // Gate the operator dials on the Matérn-ν RKHS smoothness so a rough kernel
    // (e.g. ν=1/2) is not over-smoothed by a higher-order roughness penalty its
    // own RKHS norm does not control (#707).
    let matern_spec = DuchonOperatorPenaltySpec::matern_for_smoothness(nu, centers.ncols());
    Ok(operator_penalty_candidates_from_collocation(
        &ops.d0,
        &ops.d1,
        &ops.d2,
        &matern_spec,
    ))
}

/// True when every entry of `m` is finite.
fn matrix_all_finite(m: &Array2<f64>) -> bool {
    m.iter().all(|v| v.is_finite())
}

/// Discrete function Gram on Matérn's frozen center support.
///
/// The embedded primary contains `K_CC` in its kernel block. Evaluating the
/// represented raw basis at the same centers gives `[K_CC | 1]`; applying the
/// final kernel-identifiability chart and taking `B_CᵀB_C` therefore provides
/// an exact compact Gram for this finite-rank representation without touching
/// the training rows.
pub(crate) fn matern_center_function_gram(
    embedded_kernel: &Array2<f64>,
    include_intercept: bool,
    full_transform: Option<&Array2<f64>>,
) -> Result<Array2<f64>, BasisError> {
    if embedded_kernel.nrows() != embedded_kernel.ncols() {
        crate::bail_dim_basis!("Matérn embedded kernel penalty must be square");
    }
    let total = embedded_kernel.nrows();
    let k = total
        .checked_sub(usize::from(include_intercept))
        .ok_or_else(|| BasisError::InvalidInput("Matérn basis width underflow".to_string()))?;
    if k == 0 {
        crate::bail_invalid_basis!("Matérn function metric requires at least one center");
    }
    let mut center_design = Array2::<f64>::zeros((k, total));
    center_design
        .slice_mut(s![.., 0..k])
        .assign(&embedded_kernel.slice(s![0..k, 0..k]));
    if include_intercept {
        center_design.column_mut(k).fill(1.0);
    }
    let center_design = match full_transform {
        Some(transform) => fast_ab(&center_design, transform),
        None => center_design,
    };
    Ok(symmetrize_penalty(&fast_ata(&center_design)))
}

pub(crate) fn matern_double_penalty_candidates(
    primary: &Array2<f64>,
    function_gram: &Array2<f64>,
    include_intercept: bool,
) -> Result<Vec<PenaltyCandidate>, BasisError> {
    // gam#1379 — guard the Primary projected kernel Gram itself. It is `Zᵀ K Z`
    // with a finite Matérn kernel `K`, so it is finite in exact arithmetic; if a
    // degenerate trial geometry made it non-finite we cannot ship it as a
    // penalty (the range-block eigensolve would abort the fit). Surface a clear
    // basis error instead of an opaque downstream "non-finite range penalty".
    if !matrix_all_finite(primary) {
        crate::bail_invalid_basis!(
            "Matérn double-penalty primary kernel Gram is non-finite; the projected \
             kernel `Zᵀ K Z` could not be formed at this length scale (degenerate \
             geometry). Widen the data spread, change the length scale, or drop the term."
        );
    }
    if primary.dim() != function_gram.dim() || !matrix_all_finite(function_gram) {
        crate::bail_invalid_basis!(
            "Matérn center function Gram is non-finite or does not match the primary penalty"
        );
    }
    let mut candidates = vec![normalize_penalty_candidate(
        primary.clone(),
        0,
        PenaltySource::Primary,
    )];
    // K_CC is strictly positive definite after center rank reduction. The ONLY
    // structural null direction is the explicitly appended intercept. Kernel
    // eigenvalues near a floating-point tolerance remain range directions; they
    // must be conditioned/reduced, never reclassified into a κ-dependent null
    // projector. This makes penalty topology structural and κ-invariant.
    if include_intercept {
        let p = primary.nrows();
        let mut intercept_frame = Array2::<f64>::zeros((p, 1));
        intercept_frame[[p - 1, 0]] = 1.0;
        let shrinkage = function_space_subspace_shrinkage(&intercept_frame, function_gram)?;
        candidates.push(normalize_penalty_candidate(
            shrinkage,
            0,
            PenaltySource::DoublePenaltyNullspace,
        ));
    }
    Ok(candidates)
}

pub(crate) fn build_matern_double_penalty_candidates(
    spline: &MaternSplineBasis,
    full_transform: Option<&Array2<f64>>,
) -> Result<Vec<PenaltyCandidate>, BasisError> {
    let primary = project_penalty_matrix(&spline.penalty_kernel, full_transform);
    let include_intercept = spline.num_polynomial_basis == 1;
    let function_gram =
        matern_center_function_gram(&spline.penalty_kernel, include_intercept, full_transform)?;
    matern_double_penalty_candidates(&primary, &function_gram, include_intercept)
}

/// Creates a Matérn spline basis from data and centers.
///
/// The design is `[K | 1]` when `include_intercept=true` and `[K]` otherwise, where:
/// - `K_ij = k(||x_i - c_j||; length_scale, nu)` is the Matérn kernel block.
///
/// The default kernel penalty is `alpha' S alpha` with `S_jl = k(||c_j - c_l||)`, embedded
/// in the full coefficient space. With intercept included, that column is unpenalized by
/// `penalty_kernel`; optional `penalty_ridge` is the center-function-metric
/// penalty for double-penalty shrinkage of the explicit intercept direction.
///
/// NOTE: This follows the RKHS Gram construction S = K_CC (not K_CC^{-1}) in
/// coefficient space, with global scaling absorbed by the smoothing parameter λ.
pub fn create_matern_spline_basiswithworkspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    aniso_log_scales: Option<&[f64]>,
    workspace: &mut BasisWorkspace,
) -> Result<MaternSplineBasis, BasisError> {
    let n = data.nrows();
    let d = data.ncols();
    let k = centers.nrows();
    let total_cols = k + usize::from(include_intercept);
    let dense_bytes = dense_design_bytes(n, total_cols);
    if dense_bytes > workspace.policy().max_single_materialization_bytes {
        crate::bail_invalid_basis!(
            "Matérn basis dense design exceeds resource policy: n={n}, p={total_cols}, dense={:.1} MiB, cap={:.1} MiB",
            dense_bytes as f64 / (1024.0 * 1024.0),
            workspace.policy().max_single_materialization_bytes as f64 / (1024.0 * 1024.0),
        );
    }

    if d == 0 {
        crate::bail_invalid_basis!("Matérn basis requires at least one covariate dimension");
    }
    if k == 0 {
        crate::bail_invalid_basis!("Matérn basis requires at least one center");
    }
    if centers.ncols() != d {
        crate::bail_dim_basis!(
            "Matérn basis dimension mismatch: data has {d} columns, centers have {}",
            centers.ncols()
        );
    }
    if data.iter().any(|v| !v.is_finite()) || centers.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("Matérn basis requires finite data and center values");
    }
    validate_matern_length_scale(length_scale)?;
    if let Some(eta) = aniso_log_scales {
        if eta.len() != d {
            crate::bail_dim_basis!(
                "aniso_log_scales length {} does not match data dimension {d}",
                eta.len()
            );
        }
        if eta.iter().any(|v| !v.is_finite()) {
            crate::bail_invalid_basis!("aniso_log_scales must contain finite values");
        }
    }

    // Practical safe operating range for κ from center geometry (document Eq. D.2):
    //   κ in [1e-2 / r_max, 1e2 / r_min], with κ = 1/length_scale.
    // Warn rather than silently clamp so callers keep explicit control.
    // Under anisotropy the kernel metric is y-space (y_a = exp(η_a) x_a), so
    // the relevant r_min/r_max are y-space pairwise distances, not raw.
    let warn_bounds = if let Some(eta) = aniso_log_scales {
        let y_centers = points_in_aniso_y_space(centers, eta);
        pairwise_distance_bounds(y_centers.view())
    } else {
        pairwise_distance_bounds(centers)
    };
    if let Some((r_min, r_max)) = warn_bounds {
        let kappa = 1.0 / length_scale.max(1e-300);
        let kappa_lo = 1e-2 / r_max;
        let kappa_hi = 1e2 / r_min;
        if kappa < kappa_lo || kappa > kappa_hi {
            log::debug!(
                "Matérn κ={} is outside recommended range [{}, {}] derived from centers (r_min={}, r_max={}); kernel conditioning may degrade",
                kappa,
                kappa_lo,
                kappa_hi,
                r_min,
                r_max
            );
        }
    }

    // Distance computation: anisotropic when eta is present, isotropic otherwise.
    // Under anisotropy we work in y-space (y = Ax), so r = |Ah| replaces |h|.
    let mut kernel_block = Array2::<f64>::zeros((n, k));
    let mut center_kernel = Array2::<f64>::zeros((k, k));
    let axis_scales = aniso_log_scales.map(aniso_axis_scales);
    let kernel_result: Result<(), BasisError> = kernel_block
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            for j in 0..k {
                let r = if let Some(scales) = axis_scales.as_deref() {
                    aniso_distance_rows_with_scales(data, i, centers, j, scales)
                } else {
                    euclidean_distance_rows(data, i, centers, j)
                };
                row[j] = matern_kernel_from_distance(r, length_scale, nu)?;
            }
            Ok(())
        });
    kernel_result?;
    // Center-center Gram matrix K_CC. In RKHS form, the kernel penalty on
    // radial coefficients is alpha^T K_CC alpha.
    fill_symmetric_from_row_kernel(&mut center_kernel, |i, j| {
        let r = if let Some(scales) = axis_scales.as_deref() {
            aniso_distance_rows_with_scales(centers, i, centers, j, scales)
        } else {
            euclidean_distance_rows(centers, i, centers, j)
        };
        matern_kernel_from_distance(r, length_scale, nu)
    })?;

    let mut basis = Array2::<f64>::zeros((n, total_cols));
    basis.slice_mut(s![.., 0..k]).assign(&kernel_block);
    if include_intercept {
        basis.column_mut(k).fill(1.0);
    }

    let mut penalty_kernel = Array2::<f64>::zeros((total_cols, total_cols));
    // RKHS coefficient penalty uses the center Gram matrix directly:
    //   S = K_CC  (not K_CC^{-1}).
    // This matches Duchon/Matérn spline theory where alpha^T K_CC alpha is the
    // native-space quadratic form up to a global scaling absorbed by lambda.
    penalty_kernel
        .slice_mut(s![0..k, 0..k])
        .assign(&center_kernel);
    let function_gram = matern_center_function_gram(&penalty_kernel, include_intercept, None)?;
    let penalty_ridge = if include_intercept {
        let mut intercept_frame = Array2::<f64>::zeros((total_cols, 1));
        intercept_frame[[total_cols - 1, 0]] = 1.0;
        function_space_subspace_shrinkage(&intercept_frame, &function_gram)?
    } else {
        Array2::<f64>::zeros((total_cols, total_cols))
    };

    Ok(MaternSplineBasis {
        basis,
        penalty_kernel,
        penalty_ridge,
        num_kernel_basis: k,
        num_polynomial_basis: usize::from(include_intercept),
        dimension: d,
    })
}

#[inline]
pub(crate) fn validate_lat_lon_matrix(
    data: ArrayView2<'_, f64>,
    context: &str,
    radians: bool,
) -> Result<(), BasisError> {
    if data.ncols() != 2 {
        crate::bail_dim_basis!(
            "{context} requires exactly two columns: latitude and longitude; got {}",
            data.ncols()
        );
    }
    if data.nrows() == 0 {
        crate::bail_invalid_basis!("{context} requires at least one row");
    }
    let (lat_lo, lat_hi, unit) = if radians {
        (
            -std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
            "radians",
        )
    } else {
        (-90.0, 90.0, "degrees")
    };
    for (i, row) in data.outer_iter().enumerate() {
        let lat = row[0];
        let lon = row[1];
        if !lat.is_finite() || !lon.is_finite() {
            crate::bail_invalid_basis!(
                "{context} requires finite latitude/longitude; row {i} has ({lat}, {lon})"
            );
        }
        if !(lat_lo..=lat_hi).contains(&lat) {
            crate::bail_invalid_basis!(
                "{context} latitude must be in [{lat_lo}, {lat_hi}] {unit}; row {i} has {lat}"
            );
        }
    }
    Ok(())
}

pub fn spherical_wahba_kernel_matrix(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    penalty_order: usize,
    radians: bool,
) -> Result<Array2<f64>, BasisError> {
    spherical_wahba_kernel_matrix_with_kind(
        data,
        centers,
        penalty_order,
        radians,
        SphereWahbaKernel::Sobolev,
    )
}

pub fn spherical_wahba_kernel_matrix_with_kind(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    penalty_order: usize,
    radians: bool,
    kernel: SphereWahbaKernel,
) -> Result<Array2<f64>, BasisError> {
    validate_lat_lon_matrix(data, "spherical spline data", radians)?;
    validate_lat_lon_matrix(centers, "spherical spline centers", radians)?;
    // GPU fast path for the truncated-spectral kernels. The CPU SIMD loop
    // (`spherical_wahba_kernel_matrix_cpu`) is the bit-defining oracle; the
    // device only engages when `sphere_kernel_decision` admits the work (large
    // `n·m`, `lmax ≤ 200`, memory budget). `None` ⇒ quiet CPU route (closed-form
    // variant, no device, or below threshold); `Some(Err)` ⇒ admitted device
    // failed ⇒ surface it (never a silent CPU degrade — the engagement-failure
    // class this path kills).
    if let Some(gpu_result) = crate::basis::sphere_gpu::try_build_truncated_kernel_matrix_gpu(
        data,
        centers,
        penalty_order,
        radians,
        kernel,
    ) {
        let gpu_matrix = gpu_result.map_err(|err| {
            BasisError::InvalidInput(format!(
                "spherical spline GPU truncated kernel was admitted but failed on device: {err}"
            ))
        })?;
        return Ok(gpu_matrix);
    }
    spherical_wahba_kernel_matrix_cpu(data, centers, penalty_order, radians, kernel)
}

/// CPU oracle for the Wahba S² kernel design matrix — the bit-defining
/// reference the GPU truncated path is held to. Always evaluates on host,
/// regardless of the GPU dispatch decision, so parity tests and any caller that
/// needs the deterministic reference can bypass device routing entirely.
pub fn spherical_wahba_kernel_matrix_cpu(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    penalty_order: usize,
    radians: bool,
    kernel: SphereWahbaKernel,
) -> Result<Array2<f64>, BasisError> {
    validate_lat_lon_matrix(data, "spherical spline data", radians)?;
    validate_lat_lon_matrix(centers, "spherical spline centers", radians)?;
    let n = data.nrows();
    let k = centers.nrows();
    let deg = if radians {
        1.0
    } else {
        std::f64::consts::PI / 180.0
    };
    // Precompute (sin_lat, cos_lat, sin_lon, cos_lon) for each center once.
    // Using cos(lon - lon_c) = cos(lon)·cos(lon_c) + sin(lon)·sin(lon_c)
    // collapses the inner-loop trig from one `.cos()` per (i, j) down to
    // four multiplies and an add — a ~10x speedup on the inner body at
    // large-scale N·K.
    let mut sin_lat_c = Vec::<f64>::with_capacity(k);
    let mut cos_lat_c = Vec::<f64>::with_capacity(k);
    let mut sin_lon_c = Vec::<f64>::with_capacity(k);
    let mut cos_lon_c = Vec::<f64>::with_capacity(k);
    for c in centers.outer_iter() {
        let lat = c[0] * deg;
        let lon = c[1] * deg;
        let (s_lat, c_lat) = lat.sin_cos();
        let (s_lon, c_lon) = lon.sin_cos();
        sin_lat_c.push(s_lat);
        cos_lat_c.push(c_lat);
        sin_lon_c.push(s_lon);
        cos_lon_c.push(c_lon);
    }
    let mut out = Array2::<f64>::zeros((n, k));
    let err_flag = std::sync::atomic::AtomicBool::new(false);
    out.axis_chunks_iter_mut(ndarray::Axis(0), 256)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut block)| {
            use wide::f64x4;
            let row_offset = chunk_idx * 256;
            let chunks = k / 4;
            let tail = k % 4;
            for (local_i, mut out_row) in block.outer_iter_mut().enumerate() {
                let i = row_offset + local_i;
                let lat = data[(i, 0)] * deg;
                let lon = data[(i, 1)] * deg;
                let (sin_lat, cos_lat) = lat.sin_cos();
                let (sin_lon, cos_lon) = lon.sin_cos();
                let sin_lat_v = f64x4::from(sin_lat);
                let cos_lat_v = f64x4::from(cos_lat);
                let sin_lon_v = f64x4::from(sin_lon);
                let cos_lon_v = f64x4::from(cos_lon);
                // SIMD over 4 centers at a time.
                for cidx in 0..chunks {
                    let base = cidx * 4;
                    let sl_c = f64x4::from([
                        sin_lat_c[base],
                        sin_lat_c[base + 1],
                        sin_lat_c[base + 2],
                        sin_lat_c[base + 3],
                    ]);
                    let cl_c = f64x4::from([
                        cos_lat_c[base],
                        cos_lat_c[base + 1],
                        cos_lat_c[base + 2],
                        cos_lat_c[base + 3],
                    ]);
                    let sn_c = f64x4::from([
                        sin_lon_c[base],
                        sin_lon_c[base + 1],
                        sin_lon_c[base + 2],
                        sin_lon_c[base + 3],
                    ]);
                    let cn_c = f64x4::from([
                        cos_lon_c[base],
                        cos_lon_c[base + 1],
                        cos_lon_c[base + 2],
                        cos_lon_c[base + 3],
                    ]);
                    let dlon_cos = cos_lon_v * cn_c + sin_lon_v * sn_c;
                    let cos_gamma = sin_lat_v * sl_c + cos_lat_v * cl_c * dlon_cos;
                    let vals =
                        wahba_sphere_kernel_from_cos_simd_kind(cos_gamma, penalty_order, kernel);
                    let arr = vals.to_array();
                    for lane in 0..4 {
                        if !arr[lane].is_finite() {
                            err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            return;
                        }
                        out_row[base + lane] = arr[lane];
                    }
                }
                // Scalar tail (0..3 elements).
                let tail_start = chunks * 4;
                for t in 0..tail {
                    let j = tail_start + t;
                    let dlon_cos = cos_lon * cos_lon_c[j] + sin_lon * sin_lon_c[j];
                    let cos_gamma = sin_lat * sin_lat_c[j] + cos_lat * cos_lat_c[j] * dlon_cos;
                    match wahba_sphere_kernel_from_cos_kind(cos_gamma, penalty_order, kernel) {
                        Ok(v) => out_row[j] = v,
                        Err(_) => {
                            err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            return;
                        }
                    }
                }
            }
        });
    if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
        crate::bail_invalid_basis!("spherical spline kernel produced a non-finite value");
    }
    Ok(out)
}

pub(crate) fn weighted_coefficient_sum_to_zero_transform(
    weights: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, BasisError> {
    let k = weights.len();
    if k < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: k });
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        crate::bail_invalid_basis!(
            "sphere coefficient constraint weights must be finite and non-negative"
        );
    }
    let norm = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
    if norm <= 0.0 {
        crate::bail_invalid_basis!("sphere coefficient constraint weights cannot all be zero");
    }
    let c = Array2::from_shape_vec((k, 1), weights.iter().map(|w| *w / norm).collect())
        .map_err(|e| BasisError::InvalidInput(format!("invalid sphere constraint weights: {e}")))?;
    let (z, rank) =
        rrqr_nullspace_basis(&c, default_rrqr_rank_alpha()).map_err(BasisError::LinalgError)?;
    if rank >= k {
        return Err(BasisError::ConstraintNullspaceCollapsed {
            site: "weighted_coefficient_sum_to_zero_transform",
            cross_rank: rank,
            coeff_dim: k,
            cross_frobenius: 1.0,
            gram_spectrum: "not computed (structural rank collapse before Gram eigendecomposition)"
                .to_string(),
        });
    }
    Ok(z)
}

const SPHERICAL_CENTER_COINCIDENT_TOL: f64 = 1.0e-12;

#[inline]
fn spherical_center_dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Lexicographic comparison of the sorted intrinsic distance profiles of two
/// spherical data rows. `Equal` means the unordered geometry cannot distinguish
/// the two candidates: choosing one by row index would violate permutation
/// invariance, so the selector must treat their whole class atomically.
fn spherical_dot_profile_cmp(units: &[[f64; 3]], i: usize, j: usize) -> std::cmp::Ordering {
    let mut pi: Vec<f64> = units
        .iter()
        .map(|u| spherical_center_dot(&units[i], u))
        .collect();
    let mut pj: Vec<f64> = units
        .iter()
        .map(|u| spherical_center_dot(&units[j], u))
        .collect();
    pi.sort_by(f64::total_cmp);
    pj.sort_by(f64::total_cmp);
    for (a, b) in pi.iter().zip(pj.iter()) {
        let ordering = a.total_cmp(b);
        if !ordering.is_eq() {
            return ordering;
        }
    }
    std::cmp::Ordering::Equal
}

/// Remove coincident directions from one invariant tie class without choosing
/// between genuinely distinct tied directions. Representatives of coincident
/// rows are geometrically interchangeable and produce the same kernel column;
/// every distinct member of the symmetry class is retained.
fn distinct_spherical_orbit(
    units: &[[f64; 3]],
    candidates: &[usize],
    already_selected: &[usize],
) -> Vec<usize> {
    let mut distinct = Vec::with_capacity(candidates.len());
    'candidate: for &candidate in candidates {
        for &selected in already_selected.iter().chain(distinct.iter()) {
            if spherical_center_dot(&units[candidate], &units[selected])
                >= 1.0 - SPHERICAL_CENTER_COINCIDENT_TOL
            {
                continue 'candidate;
            }
        }
        distinct.push(candidate);
    }
    distinct
}

/// Select spherical-spline basis centers by **geodesic** farthest-point sampling
/// of the data cloud, returning a well-spread subset of the actual data rows.
///
/// This is the rotation-EQUIVARIANT center rule Wahba's reproducing-kernel smooth
/// needs. The kernel is a function of the geodesic angle alone (`k(cos γ)`,
/// `cos γ = uᵢ·u_c` for unit vectors `u`), so the continuous estimator is exactly
/// SO(3)-invariant; the finite-center discretization inherits that invariance
/// **iff** the centers rotate rigidly with the data. Every ingredient of this
/// selection is a dot product of data unit vectors — the mean-direction seed key
/// `uᵢ·Σⱼuⱼ`, the maximin nearest-center dot `max_c uᵢ·u_c`, and the sorted
/// dot-profile tie-break — and a dot product is invariant under any rotation `R`
/// (`(Ruᵢ)·(Ru_c) = uᵢ·u_c`). So under ANY rotation of the data the SAME physical
/// rows are selected, the returned centers are exactly those rows rotated, every
/// kernel entry `k(uᵢ·u_c)` is preserved, and the fit and every prediction are
/// invariant to the arbitrary choice of frame (a longitude origin, a tilt, any
/// element of SO(3)) — matching the rotation-invariant `harmonic` control (#2127).
///
/// On a symmetric cloud, invariant scalar keys can leave several **distinct**
/// rows exactly tied. No row-permutation-equivariant rule can choose one member
/// of such an orbit: a symmetry exchanging two tied rows would have to both
/// preserve and change that choice. The selector therefore adds the complete
/// distinct-direction tie class atomically. Because `num_centers` is an exact
/// resource contract, a class that does not fit in the remaining budget is
/// refused as unrepresentable rather than truncated by row index. Coincident
/// rows remain one kernel column; consequently a request exceeding the number
/// of distinct directions is also refused rather than silently undersized.
///
/// The previous implementation ignored `data` and laid down a fixed golden-angle
/// (Fibonacci) lattice pinned in the (lat, lon) frame: a rigid rotation moved the
/// data relative to the STATIONARY centers, changed every data-to-center geodesic
/// angle, and reshaped the fitted surface. Anchoring only the lattice's longitude
/// origin to the data (a first pass at #2127) fixed rotations about the pole but
/// left the frame-pinned latitudes exposed to a tilt.
///
/// The selection mirrors the Euclidean thin-plate knot picker
/// ([`select_thin_plate_knots`]) — centroid-nearest seed, maximin recursion,
/// invariant tie-breaks — but with geodesic (great-circle) distance in place of
/// Euclidean distance, which is the correct SO(3) invariant on S². Coincident
/// data directions are not selected twice (a duplicate center makes the Wahba
/// Gram singular).
pub fn select_spherical_farthest_point_centers(
    data: ArrayView2<'_, f64>,
    num_centers: usize,
    radians: bool,
) -> Result<Array2<f64>, BasisError> {
    validate_lat_lon_matrix(data, "spherical farthest-point centers", radians)?;
    if num_centers == 0 {
        crate::bail_invalid_basis!("spherical farthest-point center count must be positive");
    }
    let n = data.nrows();
    if n < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: n });
    }
    if num_centers > n {
        crate::bail_invalid_basis!(
            "requested {num_centers} spherical farthest-point centers but only {n} rows are available"
        );
    }

    let to_rad = if radians {
        1.0
    } else {
        std::f64::consts::PI / 180.0
    };
    // Unit vectors on S² for each data row. The geodesic distance between rows is
    // a monotone-DECREASING function of the dot product `uᵢ·uⱼ = cos γ`, so every
    // "distance" comparison below is phrased directly in dot products — each of
    // which is exactly rotation invariant.
    let units: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let lat = data[[i, 0]] * to_rad;
            let lon = data[[i, 1]] * to_rad;
            let cos_lat = lat.cos();
            [cos_lat * lon.cos(), cos_lat * lon.sin(), lat.sin()]
        })
        .collect();
    // Mean-direction seed key `dot_to_sum[i] = uᵢ·Σⱼuⱼ`. `Σⱼuⱼ` is
    // rotation-EQUIVARIANT (it rotates rigidly with the data), so `argmax` is the
    // SAME physical row in every frame. Using the UNNORMALIZED resultant avoids
    // the fp blow-up of normalizing a near-zero mean direction on a well-covered
    // sphere (`dot_to_sum` is a homogeneous linear function of the resultant, so
    // its relative accuracy — and hence the argmax — is stable regardless of the
    // resultant's magnitude). Each component sum is taken in value-sorted order so
    // the key is also invariant to a pure row permutation (matching
    // `select_thin_plate_knots`).
    let mut sum = [0.0_f64; 3];
    for (c, sum_c) in sum.iter_mut().enumerate() {
        let mut col: Vec<f64> = units.iter().map(|u| u[c]).collect();
        col.sort_by(|a, b| a.total_cmp(b));
        *sum_c = col.iter().sum();
    }
    let dot_to_sum: Vec<f64> = units
        .iter()
        .map(|u| spherical_center_dot(u, &sum))
        .collect();

    // Seed class = rows nearest the mean direction (largest `dot_to_sum`), then
    // lexicographically smallest intrinsic dot profile. If that complete key is
    // tied, retain the whole symmetry orbit; a row-index tie-break is forbidden.
    let mut seed = 0usize;
    for i in 1..n {
        let take = match dot_to_sum[i].total_cmp(&dot_to_sum[seed]) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => spherical_dot_profile_cmp(&units, i, seed).is_lt(),
        };
        if take {
            seed = i;
        }
    }

    let target = num_centers;
    let seed_class: Vec<usize> = (0..n)
        .filter(|&i| {
            dot_to_sum[i].total_cmp(&dot_to_sum[seed]).is_eq()
                && spherical_dot_profile_cmp(&units, i, seed).is_eq()
        })
        .collect();
    let seed_orbit = distinct_spherical_orbit(&units, &seed_class, &[]);
    if seed_orbit.len() > target {
        crate::bail_invalid_basis!(
            "spherical farthest-point seed symmetry orbit has {} distinct directions, exceeding the requested center budget {target}; use a budget at least as large as the orbit or the harmonic sphere basis",
            seed_orbit.len()
        );
    }

    let mut selected = Vec::with_capacity(target);
    let mut chosen = vec![false; n];
    // `max_dot[i]` = `max` over chosen centers `c` of `uᵢ·u_c` = `cos` of the
    // geodesic distance to the NEAREST chosen center. The maximin step picks the
    // unchosen row MINIMIZING it (farthest from all chosen).
    let mut max_dot = vec![f64::NEG_INFINITY; n];
    for &i in &seed_class {
        chosen[i] = true;
    }
    selected.extend(seed_orbit);
    for i in 0..n {
        max_dot[i] = selected
            .iter()
            .map(|&center| spherical_center_dot(&units[i], &units[center]))
            .fold(f64::NEG_INFINITY, f64::max);
    }

    // A dot `≥ 1 − SPHERICAL_CENTER_COINCIDENT_TOL` is a geodesic angle
    // `≲ 1.4e-6` rad: the
    // candidate coincides with an already-chosen center, so selecting it would add
    // a duplicate kernel column and a singular Wahba Gram. Stopping here caps the
    // center set at the number of DISTINCT data directions.
    while selected.len() < target {
        let mut best: Option<usize> = None;
        for i in 0..n {
            if chosen[i] {
                continue;
            }
            match best {
                None => best = Some(i),
                Some(b) => {
                    // Maximin: prefer the larger geodesic distance to the chosen
                    // set (the SMALLER `max_dot`). Exact `max_dot` ties — common on
                    // symmetric clouds and in float arithmetic — break first toward
                    // the MORE PERIPHERAL row (smaller `dot_to_sum`, which spreads
                    // centers outward and is rotation invariant), then by the
                    // invariant dot-profile. A tie after all three keys is a
                    // symmetry orbit and is completed atomically below.
                    let take = match max_dot[i].total_cmp(&max_dot[b]) {
                        std::cmp::Ordering::Less => true,
                        std::cmp::Ordering::Greater => false,
                        std::cmp::Ordering::Equal => {
                            match dot_to_sum[i].total_cmp(&dot_to_sum[b]) {
                                std::cmp::Ordering::Less => true,
                                std::cmp::Ordering::Greater => false,
                                std::cmp::Ordering::Equal => {
                                    spherical_dot_profile_cmp(&units, i, b).is_lt()
                                }
                            }
                        }
                    };
                    if take {
                        best = Some(i);
                    }
                }
            }
        }
        let next = match best {
            Some(i) => i,
            None => break,
        };
        if max_dot[next] >= 1.0 - SPHERICAL_CENTER_COINCIDENT_TOL {
            break;
        }

        let tied_class: Vec<usize> = (0..n)
            .filter(|&i| {
                !chosen[i]
                    && max_dot[i].total_cmp(&max_dot[next]).is_eq()
                    && dot_to_sum[i].total_cmp(&dot_to_sum[next]).is_eq()
                    && spherical_dot_profile_cmp(&units, i, next).is_eq()
            })
            .collect();
        let orbit = distinct_spherical_orbit(&units, &tied_class, &selected);
        let remaining = target - selected.len();
        if orbit.len() > remaining {
            crate::bail_invalid_basis!(
                "spherical farthest-point tie class has {} distinct directions but only {remaining} of the exact {target}-center budget remain; choose a compatible center count or the harmonic sphere basis",
                orbit.len(),
            );
        }
        for &i in &tied_class {
            chosen[i] = true;
        }
        if orbit.is_empty() {
            continue;
        }
        selected.extend(orbit.iter().copied());
        for i in 0..n {
            if chosen[i] {
                continue;
            }
            for &center in &orbit {
                let d = spherical_center_dot(&units[i], &units[center]);
                if d > max_dot[i] {
                    max_dot[i] = d;
                }
            }
        }
    }

    if selected.len() < target {
        crate::bail_invalid_basis!(
            "requested {target} distinct spherical farthest-point centers but the data contain only {} numerically distinct directions",
            selected.len()
        );
    }
    if selected.len() < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: selected.len(),
        });
    }

    // Return the selected rows VERBATIM (in the data's own lat/lon units), so the
    // centers ARE data points and carry the rotation exactly.
    let mut centers = Array2::<f64>::zeros((selected.len(), 2));
    for (r, &idx) in selected.iter().enumerate() {
        centers[[r, 0]] = data[[idx, 0]];
        centers[[r, 1]] = data[[idx, 1]];
    }
    Ok(centers)
}

#[cfg(test)]
mod spherical_farthest_point_symmetry_tests {
    use super::*;
    use ndarray::{Array2, array};

    fn permute_rows(data: &Array2<f64>, order: &[usize]) -> Array2<f64> {
        Array2::from_shape_fn((order.len(), 2), |(row, col)| data[[order[row], col]])
    }

    fn sorted_center_rows(centers: &Array2<f64>) -> Vec<[f64; 2]> {
        let mut rows: Vec<[f64; 2]> = centers.outer_iter().map(|row| [row[0], row[1]]).collect();
        rows.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
        rows
    }

    /// The equatorial point is the unique centroid-nearest seed. North and
    /// south are then exactly tied by every intrinsic key and are exchanged by
    /// a data symmetry, so selecting either one by row index is impossible to
    /// reconcile with permutation invariance. The selector must add both as one
    /// atomic orbit. A duplicate equatorial row remains one kernel column.
    #[test]
    fn symmetric_tie_orbit_is_completed_under_every_row_permutation() {
        let data = array![[0.0_f64, 0.0], [0.0, 0.0], [90.0, 0.0], [-90.0, 0.0]];
        let permutations = [[0_usize, 1, 2, 3], [0, 1, 3, 2], [2, 0, 3, 1], [3, 1, 2, 0]];

        let mut reference: Option<Vec<[f64; 2]>> = None;
        for order in permutations {
            let permuted = permute_rows(&data, &order);
            let centers = select_spherical_farthest_point_centers(permuted.view(), 3, false)
                .expect("the complete three-direction symmetry orbit is representable");
            assert_eq!(
                centers.nrows(),
                3,
                "the exact three-center target must contain the complete north/south tie class"
            );
            let center_set = sorted_center_rows(&centers);
            if let Some(expected) = &reference {
                assert_eq!(
                    &center_set, expected,
                    "selected physical center set changed under row permutation"
                );
            } else {
                reference = Some(center_set);
            }
        }
    }

    #[test]
    fn incomplete_nonseed_tie_class_is_refused() {
        let data = array![[0.0_f64, 0.0], [0.0, 0.0], [90.0, 0.0], [-90.0, 0.0]];
        let error = select_spherical_farthest_point_centers(data.view(), 2, false)
            .expect_err("one remaining slot cannot split the north/south tie class");
        assert!(
            error
                .to_string()
                .contains("only 1 of the exact 2-center budget remain"),
            "unexpected refusal: {error}"
        );
    }

    /// A symmetry orbit is indivisible. If even one orbit is larger than the
    /// declared resource budget, refusing is the only bounded equivariant
    /// answer; silently choosing a row-index representative is mathematically
    /// false and expanding without a bound can turn an O(m) request into O(n).
    #[test]
    fn symmetry_orbit_larger_than_center_budget_is_refused() {
        let antipodal = array![[90.0_f64, 0.0], [-90.0, 0.0]];
        let error = select_spherical_farthest_point_centers(antipodal.view(), 1, false)
            .expect_err("a two-direction seed orbit cannot fit a one-center budget");
        assert!(
            error.to_string().contains("symmetry orbit"),
            "unexpected refusal: {error}"
        );
    }
}

#[cfg(test)]
mod matern_function_metric_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn center_metric_null_ridge_is_covariant_and_targets_only_intercept_function() {
        let center_kernel = array![[1.4, 0.3, 0.1], [0.3, 1.2, 0.2], [0.1, 0.2, 1.1]];
        let mut embedded = Array2::<f64>::zeros((4, 4));
        embedded.slice_mut(s![0..3, 0..3]).assign(&center_kernel);
        let gram =
            matern_center_function_gram(&embedded, true, None).expect("raw center function Gram");
        let base = matern_double_penalty_candidates(&embedded, &gram, true)
            .expect("raw candidates");
        assert_eq!(base.len(), 2);
        let raw_ridge = &base[1].matrix * base[1].normalization_scale;

        let intercept = array![[0.0], [0.0], [0.0], [1.0]];
        let action_error = (&raw_ridge.dot(&intercept) - &gram.dot(&intercept))
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            action_error < 2.0e-13,
            "ridge must equal G on the structural intercept; error={action_error:.3e}"
        );

        // A strongly non-orthogonal kernel chart plus intercept rescaling. The
        // block structure is exactly Matérn's supported final transform: kernel
        // coordinates may shear/rescale, while the explicit intercept remains a
        // separate structural coordinate.
        let transform = array![
            [0.2, 0.5, 0.0, 0.0],
            [0.0, 3.0, -0.4, 0.0],
            [0.0, 0.0, 1.7, 0.0],
            [0.0, 0.0, 0.0, 2.5]
        ];
        let primary_t = fast_atb(&transform, &fast_ab(&embedded, &transform));
        let gram_t = matern_center_function_gram(&embedded, true, Some(&transform))
            .expect("transformed center function Gram");
        let transformed = matern_double_penalty_candidates(&primary_t, &gram_t, true)
            .expect("transformed candidates");
        let ridge_t = &transformed[1].matrix * transformed[1].normalization_scale;
        let expected = fast_atb(&transform, &fast_ab(&raw_ridge, &transform));
        let covariance_error = (&ridge_t - &expected)
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            covariance_error < 2.0e-12,
            "Matérn function ridge changed under a basis chart; error={covariance_error:.3e}"
        );

        let no_intercept_gram = matern_center_function_gram(
            &center_kernel,
            false,
            Some(&transform.slice(s![0..3, 0..3]).to_owned()),
        )
        .expect("kernel-only Gram");
        let kernel_only = matern_double_penalty_candidates(
            &fast_atb(
                &transform.slice(s![0..3, 0..3]).to_owned(),
                &fast_ab(&center_kernel, &transform.slice(s![0..3, 0..3]).to_owned()),
            ),
            &no_intercept_gram,
            false,
        )
        .expect("kernel-only candidates");
        assert_eq!(kernel_only.len(), 1, "an SPD kernel has no null ridge");
    }
}

/// Auto-derive a streaming row chunk size for dense basis evaluation.
///
/// The opt-in `streaming_chunk_size` knob has been removed from public specs:
/// streaming activates automatically when the would-be dense buffer
/// `n_rows * n_basis_cols * 8 bytes` exceeds 1 GiB. When streaming is
/// active, the chunk size is sized so each resident chunk holds ~256 MiB
/// of `f64` (`chunk = (256 MiB) / (n_basis_cols * 8)`), clamped to
/// `[1024, n_rows]`. Returning `None` means "do not stream, materialize
/// densely".
pub fn auto_streaming_chunk_size_for_dense(n_rows: usize, n_basis_cols: usize) -> Option<usize> {
    if n_rows == 0 || n_basis_cols == 0 {
        return None;
    }
    const DENSE_THRESHOLD_BYTES: usize = 1024 * 1024 * 1024;
    const TARGET_CHUNK_BYTES: usize = 256 * 1024 * 1024;
    const MIN_CHUNK_ROWS: usize = 1024;
    let dense_bytes = n_rows.saturating_mul(n_basis_cols).saturating_mul(8);
    if dense_bytes <= DENSE_THRESHOLD_BYTES {
        return None;
    }
    let row_bytes = n_basis_cols.saturating_mul(8).max(1);
    let raw_chunk = TARGET_CHUNK_BYTES / row_bytes;
    let clamped = raw_chunk.max(MIN_CHUNK_ROWS).min(n_rows);
    Some(clamped)
}
