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
    pub(crate) owned_data:
        crate::resource::ByteLruCache<OwnedDataCacheKey, Arc<Array2<f64>>>,
}

impl BasisCacheContext {
    pub(crate) fn with_policy(policy: &crate::resource::ResourcePolicy) -> Self {
        Self {
            constraint_nullspace: ConstraintNullspaceCache::default(),
            owned_data: crate::resource::ByteLruCache::with_max_entries(
                policy.max_owned_data_cache_bytes,
                crate::resource::OWNED_DATA_CACHE_MAX_ENTRIES,
            ),
        }
    }
}

impl Default for BasisCacheContext {
    fn default() -> Self {
        Self::with_policy(&crate::resource::ResourcePolicy::default_library())
    }
}

/// Explicit per-run workspace for reusable basis-construction caches.
///
/// Pass one workspace through repeated basis builds to avoid global mutable state
/// and to keep caching scoped to a caller-controlled lifecycle.
///
/// Owned-data cache entries are byte-limited via the
/// [`crate::resource::ResourcePolicy`] provided at construction; use
/// [`BasisWorkspace::with_policy`] for large-scale workloads where a single
/// entry can be multiple gigabytes.
#[derive(Debug)]
pub struct BasisWorkspace {
    pub(crate) cache: BasisCacheContext,
    pub(crate) policy: crate::resource::ResourcePolicy,
}

impl BasisWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_policy(policy: crate::resource::ResourcePolicy) -> Self {
        Self {
            cache: BasisCacheContext::with_policy(&policy),
            policy,
        }
    }

    pub fn default_library() -> Self {
        Self::with_policy(crate::resource::ResourcePolicy::default_library())
    }

    /// Returns the resource policy this workspace was configured with.
    pub fn policy(&self) -> &crate::resource::ResourcePolicy {
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

/// Decide whether the matern double-penalty path emits the
/// `DoublePenaltyNullspace` shrinkage candidate, honoring a FROZEN bootstrap-κ
/// decision when one is present (gam#787/#860). `frozen` is
/// `MaternIdentifiability::FrozenTransform`'s `nullspace_shrinkage_survived`:
/// `Some(b)` forces the answer (so the learned-penalty count stays invariant as
/// the κ-optimizer rebuilds the design), `None` falls back to the κ-dependent
/// spectral test (the cold-build / non-frozen behavior). Returns the emitted
/// candidate list together with the realized decision so the caller can record
/// it into the basis metadata for the freeze step.
pub(crate) fn matern_double_penalty_candidates_with_decision(
    primary: &Array2<f64>,
    frozen: Option<bool>,
) -> Result<(Vec<PenaltyCandidate>, bool), BasisError> {
    let mut candidates = vec![normalize_penalty_candidate(
        primary.clone(),
        0,
        PenaltySource::Primary,
    )];
    let survived = match frozen {
        Some(forced) => {
            if forced && let Some(shrinkage) = build_nullspace_shrinkage_penalty(primary)? {
                candidates.push(normalize_penalty_candidate(
                    shrinkage.sym_penalty,
                    0,
                    PenaltySource::DoublePenaltyNullspace,
                ));
                true
            } else {
                // Forced ON but the projected kernel has no near-zero direction
                // at this κ (so there is literally no shrinkage subspace to
                // build), OR forced OFF: emit only the primary kernel penalty.
                // Forced-ON-without-a-subspace cannot manufacture a 7th penalty,
                // but the frozen path only sets `Some(true)` when the bootstrap κ
                // DID find a subspace, and the projected-kernel null space is a
                // geometric property of the centers/transform (κ rescales every
                // eigenvalue together), so the subspace persists across rebuilds.
                false
            }
        }
        None => {
            if let Some(shrinkage) = build_nullspace_shrinkage_penalty(primary)? {
                candidates.push(normalize_penalty_candidate(
                    shrinkage.sym_penalty,
                    0,
                    PenaltySource::DoublePenaltyNullspace,
                ));
                true
            } else {
                false
            }
        }
    };
    Ok((candidates, survived))
}

pub(crate) fn build_matern_double_penalty_candidates(
    spline: &MaternSplineBasis,
    full_transform: Option<&Array2<f64>>,
    frozen_nullspace_shrinkage_survived: Option<bool>,
) -> Result<(Vec<PenaltyCandidate>, bool), BasisError> {
    let primary = project_penalty_matrix(&spline.penalty_kernel, full_transform);
    matern_double_penalty_candidates_with_decision(&primary, frozen_nullspace_shrinkage_survived)
}

/// Creates a Matérn spline basis from data and centers.
///
/// The design is `[K | 1]` when `include_intercept=true` and `[K]` otherwise, where:
/// - `K_ij = k(||x_i - c_j||; length_scale, nu)` is the Matérn kernel block.
///
/// The default kernel penalty is `alpha' S alpha` with `S_jl = k(||c_j - c_l||)`, embedded
/// in the full coefficient space. With intercept included, that column is unpenalized by
/// `penalty_kernel`; optional `penalty_ridge` is a nullspace projector used for
/// double-penalty shrinkage of previously unpenalized directions.
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
    let penalty_ridge = build_nullspace_shrinkage_penalty(&penalty_kernel)?
        .map(|block| block.sym_penalty)
        .unwrap_or_else(|| Array2::<f64>::zeros((total_cols, total_cols)));

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

pub fn select_spherical_farthest_point_centers(
    data: ArrayView2<'_, f64>,
    num_centers: usize,
    radians: bool,
) -> Result<Array2<f64>, BasisError> {
    validate_lat_lon_matrix(data, "spherical farthest-point centers", radians)?;
    if num_centers == 0 {
        crate::bail_invalid_basis!("spherical farthest-point center count must be positive");
    }

    let to_units = if radians {
        1.0
    } else {
        180.0 / std::f64::consts::PI
    };
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let mut centers = Array2::<f64>::zeros((num_centers, 2));
    for i in 0..num_centers {
        let z = (2.0 * i as f64 + 1.0) / num_centers as f64 - 1.0;
        let lon = ((i as f64) * golden_angle + std::f64::consts::PI)
            .rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        centers[[i, 0]] = z.asin() * to_units;
        centers[[i, 1]] = lon * to_units;
    }
    Ok(centers)
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
