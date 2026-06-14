
pub fn build_spherical_spline_basis(
    data: ArrayView2<'_, f64>,
    spec: &SphericalSplineBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    if matches!(spec.method, SphereMethod::Harmonic) {
        return build_spherical_harmonic_basis(data, spec);
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
    let raw_penalty = spherical_wahba_kernel_matrix_with_kind(
        centers.view(),
        centers.view(),
        spec.penalty_order,
        spec.radians,
        spec.wahba_kernel,
    )?;
    // Realized-design constraint transform. At fit time this is the
    // area-weighted center sum-to-zero `z` (the global identifiability pipeline
    // then composes the parametric-orthogonalization onto it and freezes the
    // result). At predict time the frozen composed transform `z · z_parametric`
    // is replayed verbatim so the realized design reproduces the fit-time
    // basis exactly (#532) — recomputing `z` from the centers would drop the
    // parametric orthogonalization and resurrect the constant-vs-intercept
    // collision.
    let z = match &spec.identifiability {
        SphericalSplineIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != centers.nrows() {
                crate::bail_dim_basis!(
                    "frozen spherical identifiability transform mismatch: {} centers but transform has {} rows",
                    centers.nrows(),
                    transform.nrows()
                );
            }
            transform.clone()
        }
        SphericalSplineIdentifiability::CenterSumToZero => {
            let weights = sphere_area_weights(centers.view(), spec.radians);
            weighted_coefficient_sum_to_zero_transform(weights.view())?
        }
    };
    let penalty = z.t().dot(&raw_penalty).dot(&z);
    // Prefer the device truncated-spectral kernel whenever
    // `sphere_kernel_decision` reports the (n, m, lmax) workload is
    // worth GPU dispatch — this short-circuits the CPU streaming
    // evaluator at the same time. We still keep the streaming fallback
    // for the (rare) case where the kernel is Sobolev/Pseudo (untruncated)
    // or the GPU runtime refuses the call.
    let gpu_raw_design = try_build_truncated_sphere_design_gpu(
        data,
        centers.view(),
        spec.wahba_kernel,
        spec.penalty_order,
        spec.radians,
    );
    let sphere_auto_chunk = if gpu_raw_design.is_some() {
        None
    } else {
        auto_streaming_chunk_size_for_dense(data.nrows(), z.ncols())
    };
    let design = if let Some(raw_design) = gpu_raw_design {
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(raw_design.dot(&z)))
    } else if let Some(chunk) = sphere_auto_chunk {
        log::info!(
            "Sphere basis auto-streaming evaluator: n={} p={} chunk_size={}",
            data.nrows(),
            z.ncols(),
            chunk,
        );
        let op = StreamingSphereEvaluator::new(
            Arc::new(data.as_standard_layout().to_owned()),
            Arc::new(centers.clone()),
            spec.penalty_order,
            spec.radians,
            spec.wahba_kernel,
            Some(Arc::new(z.clone())),
            Some(chunk),
        )
        .map_err(BasisError::InvalidInput)?;
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(op)))
    } else {
        let raw_design = spherical_wahba_kernel_matrix_with_kind(
            data,
            centers.view(),
            spec.penalty_order,
            spec.radians,
            spec.wahba_kernel,
        )?;
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(raw_design.dot(&z)))
    };
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
        let ridge = Array2::<f64>::eye(design.ncols());
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


/// Precomputed √(2)·N(l,m) coefficients for the real spherical-harmonic
/// real-orthonormal basis (m=0 row uses N(l,0) without the √2 factor).
///
/// Layout: index `[l * (l_cap) + m]` with l_cap = max_degree + 1; entry
/// for m = 0 is stored without the √2 prefactor (since cos(0·φ) = 1 has
/// no twin term to share normalization with).
fn precompute_harmonic_norms(max_degree: usize) -> Vec<f64> {
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
fn fill_real_spherical_harmonics_row(
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
fn build_spherical_harmonic_basis(
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
    //
    // This is already in the natural coefficient coordinates for the real
    // spherical harmonics: the basis is orthonormal on S², so X'X/n is O(1)
    // under uniform sampling, while the diagonal entries are the physical
    // roughness eigenvalues of the final function. Frobenius-normalizing this
    // matrix would divide away that meaningful spectral scale (≈1261 for
    // L=4, m=2), making REML optimize against an artificially tiny physical
    // penalty. Keep the raw operator with normalization_scale=1 so optimizer
    // lambdas are physical lambdas for this smooth.
    let mut penalty = Array2::<f64>::zeros((p, p));
    let mut col = 0usize;
    for l in 1..=l_max {
        let eig = (l as f64 * (l as f64 + 1.0)).powi(spec.penalty_order as i32);
        for _ in 0..(2 * l + 1) {
            penalty[[col, col]] = eig;
            col += 1;
        }
    }
    let mut candidates = vec![PenaltyCandidate {
        matrix: penalty,
        nullspace_dim_hint: 0,
        source: PenaltySource::Primary,
        normalization_scale: 1.0,
        kronecker_factors: None,
        op: None,
    }];
    if spec.double_penalty {
        let ridge = Array2::<f64>::eye(p);
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
            constraint_transform: None,
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


fn build_matern_basis_seeded(
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
                z_opt.as_ref().map(|z| Arc::new(z.clone())),
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
                z_opt.as_ref().map(|z| Arc::new(z.clone())),
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
fn eval_polywith_derivatives(coeffs: &[f64], a: f64) -> (f64, f64, f64) {
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
fn maternvalue_psi_triplet(
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
fn exp_poly_scaled_s2_psi_triplet(s: f64, a: f64, coeffs: &[f64], scalar: f64) -> (f64, f64, f64) {
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
fn matern_operator_psi_triplet(
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


fn gram_and_psi_derivatives_from_operator(
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
fn gram_cross_psi_derivative_from_operator(
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
fn normalize_penalty_cross_psi_derivative(
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
fn trace_of_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.t().dot(b).diag().sum()
}


fn normalize_penaltywith_psi_derivatives(
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


fn build_matern_operator_penalty_psi_derivatives(
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

    let project = |mat: Array2<f64>| {
        if let Some(z) = z_opt {
            fast_ab(&mat, z)
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


struct DuchonRawPenaltyPsiDerivativeBlocks {
    d0: Array2<f64>,
    d1: Array2<f64>,
    d2: Array2<f64>,
    d0_psi: Array2<f64>,
    d1_psi: Array2<f64>,
    d2_psi: Array2<f64>,
    d0_psi_psi: Array2<f64>,
    d1_psi_psi: Array2<f64>,
    d2_psi_psi: Array2<f64>,
}


impl DuchonRawPenaltyPsiDerivativeBlocks {
    fn zeros(p: usize, d: usize, cols: usize) -> Self {
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

    fn add_assign(&mut self, rhs: &Self) {
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


fn build_duchon_operator_penalty_psi_derivatives(
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

    let project = |mat: Array2<f64>| {
        if let Some(z) = identifiability_transform {
            fast_ab(&mat, z)
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


fn operator_penalty_candidates_from_derivative_candidates(
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


fn build_duchon_native_penalty_psi_derivatives(
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
    let project_kernel = |k: &Array2<f64>| fast_ab(&fast_atb(&z, k), &z).mapv(|v| v * amp2);
    let omega = project_kernel(&kernel);
    let omega_psi = project_kernel(&kernel_psi);
    let omega_psi_psi = project_kernel(&kernel_psi_psi);

    let embed = |block: Array2<f64>| {
        let mut out = Array2::<f64>::zeros((total_cols, total_cols));
        out.slice_mut(s![..kernel_cols, ..kernel_cols])
            .assign(&block);
        symmetrize(&project_penalty_matrix(&out, identifiability_transform))
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


fn prepare_duchon_derivative_contextwithworkspace(
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
fn prepare_periodic_duchon_centers_1d(
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
fn prepare_periodic_duchon_centers_1d_with_period(
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


fn fill_periodic_duchon_kernel_psi_matrices(
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


fn periodic_duchon_identifiability_transformwithworkspace(
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


fn build_periodic_duchon_basis_log_kappa_derivativeswithworkspace(
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
    let mut design_first = Array2::<f64>::zeros((data.nrows(), total_cols));
    let mut design_second = Array2::<f64>::zeros((data.nrows(), total_cols));
    design_first
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&(fast_ab(&data_kernel_psi, &z_kernel) * kernel_amp));
    design_second
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&(fast_ab(&data_kernel_psi_psi, &z_kernel) * kernel_amp));
    if let Some(transform) = identifiability_transform.as_ref() {
        design_first = fast_ab(&design_first, transform);
        design_second = fast_ab(&design_second, transform);
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
    let omega = fast_ab(&fast_atb(&z_kernel, &center_kernel), &z_kernel);
    let omega_psi = fast_ab(&fast_atb(&z_kernel, &center_kernel_psi), &z_kernel);
    let omega_psi_psi = fast_ab(&fast_atb(&z_kernel, &center_kernel_psi_psi), &z_kernel);
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
    if let Some(transform) = identifiability_transform.as_ref() {
        penalty = fast_ab(&fast_atb(transform, &penalty), transform);
        penalty_psi = fast_ab(&fast_atb(transform, &penalty_psi), transform);
        penalty_psi_psi = fast_ab(&fast_atb(transform, &penalty_psi_psi), transform);
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


fn build_matern_design_psi_derivatives(
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
fn build_matern_double_penalty_primarywith_psi_derivatives(
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

    let (kernel, kernel_psi, kernel_psi_psi) = if let Some(z) = z_opt {
        let zt_s = z.t().dot(&kernel);
        let zt_d1 = z.t().dot(&kernel_psi);
        let zt_d2 = z.t().dot(&kernel_psi_psi);
        (zt_s.dot(z), zt_d1.dot(z), zt_d2.dot(z))
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
struct ShrinkageProjectorFrame {
    /// Eigenvectors of `sym(A)`, columns ascending in eigenvalue.
    u: Array2<f64>,
    /// Eigenvalues (ascending).
    evals: Array1<f64>,
    /// `1` on near-null indices `N`, `0` elsewhere.
    in_null: Vec<f64>,
    /// Null dimension `r = |N|`.
    null_dim: usize,
    /// Connection-gap floor `tol` (pairs closer than this have no resolvable
    /// gap; their eigenvector sensitivity is ambiguous and they do not move the
    /// projector, so the connection entry is set to zero).
    gap_floor: f64,
}

impl ShrinkageProjectorFrame {
    /// Build the frame from the un-normalized projected kernel Gram `A`.
    /// Returns `None` when `A` has no near-null eigenspace at this tolerance
    /// (the value build emits no shrinkage block, so there is nothing to
    /// differentiate).
    fn build(a_raw: &Array2<f64>) -> Result<Option<Self>, BasisError> {
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

    fn dim(&self) -> usize {
        self.u.nrows()
    }

    /// Skew connection `Ω_d[m,k] = (Uᵀ A_d U)[m,k] / (λ_k − λ_m)` for a single
    /// direction's `A_d = ∂A/∂η_d` (`m ≠ k`, floored at small gaps), together
    /// with the eigenbasis representation `B̂_d = Uᵀ A_d U` and the
    /// Hellmann–Feynman eigenvalue derivatives `λ_k' = B̂_d[k,k]`.
    fn connection(&self, a_dir: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Vec<f64>) {
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
    fn projector_first_hat(&self, omega: &Array2<f64>) -> Array2<f64> {
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
    fn first(&self, a_dir: &Array2<f64>) -> Array2<f64> {
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
    fn second(
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
        let b_hat_a_prime =
            &c_hat + &(fast_ab(&b_hat_a, &omega_b) - fast_ab(&omega_b, &b_hat_a));
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
    fn to_lab(&self, p_hat: &Array2<f64>) -> Array2<f64> {
        let inv_norm = 1.0 / (self.null_dim as f64).sqrt();
        symmetrize(&fast_ab(&self.u, &fast_abt(p_hat, &self.u))).mapv(|v| v * inv_norm)
    }
}

/// Exact isotropic-κ (`ρ = log κ`) first and second ψ-derivatives of the
/// Matérn double-penalty `DoublePenaltyNullspace` shrinkage block, driven by
/// the un-normalized projected-kernel Gram `A` and its log-κ derivatives.
/// Returns `(R~', R~'')`. Zeros when no shrinkage subspace exists at this ρ.
fn matern_nullspace_shrinkage_psi_derivatives(
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
fn active_matern_double_penalty_derivatives(
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
        let (
            _,
            primary_derivative,
            primarysecond_derivative,
            _,
            a_raw,
            a_raw_psi,
            a_raw_psi_psi,
        ) = build_matern_double_penalty_primarywith_psi_derivatives(
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
fn build_matern_design_psi_aniso_derivatives(
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


fn build_matern_aniso_primary_raw_derivative_matrices(
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


fn build_matern_aniso_raw_cross_derivative_matrix(
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
            let projected_first = if let Some(z) = z_opt.as_ref() {
                z.t().dot(&raw_first[a]).dot(z)
            } else {
                std::mem::take(&mut raw_first[a])
            };
            let projected_second = if let Some(z) = z_opt.as_ref() {
                z.t().dot(&raw_second_diag[a]).dot(z)
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
            let projected = if let Some(z) = z_opt.as_ref() {
                z.t().dot(&kblock).dot(z)
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
                Some(frame) => {
                    frame.second(&primary_first[a], &primary_first[a], &primary_second_diag[a])
                }
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
        let z_owned = z_opt.clone();
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
                let projected: Array2<f64> = if let Some(z) = z_owned.as_ref() {
                    z.t().dot(&raw_cross).dot(z)
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
                    let projected_a = if let Some(z) = z_owned.as_ref() {
                        z.t().dot(&kblock).dot(z)
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

    Ok(result)
}


fn duchon_coeff_exponents(p_order: usize, s_order: usize, m_or_n: usize) -> f64 {
    // In the partial fractions
    //   1 / (z^p (z + kappa^2)^s)
    // = Σ a_m(kappa) / z^m + Σ b_n(kappa) / (z + kappa^2)^n,
    // both a_m and b_n are pure powers of kappa:
    //   c(kappa) = C * kappa^{-2(p+s-index)}.
    // With psi = log(kappa), that gives c_psi = alpha c and
    // c_psipsi = alpha^2 c with alpha below. This is the exact coefficient
    // derivative rule from the Duchon spectral factorization.
    -2.0 * (p_order + s_order - m_or_n) as f64
}


#[inline(always)]
fn duchon_scaling_exponent(p_order: usize, s_order: usize, k_dim: usize) -> f64 {
    k_dim as f64 - 2.0 * (p_order + s_order) as f64
}


#[derive(Clone, Copy)]
struct DuchonMaternDerivativeTerm {
    coeff: f64,
    kappa_power: usize,
    r_power: f64,
    bessel_order: f64,
}


#[derive(Clone, Copy, Debug, Default)]
struct PsiTriplet {
    value: f64,
    psi: f64,
    psi_psi: f64,
}


#[derive(Clone, Copy, Debug, Default)]
struct DuchonRadialCore {
    phi: PsiTriplet,
}


#[derive(Clone, Copy, Debug, Default)]
struct DuchonRadialJets {
    phi: f64,
    phi_r: f64,
    phi_rr: f64,
    phi_rrr: f64,
    q: f64,
    q_r: f64,
    q_rr: f64,
    lap: f64,
    lap_r: f64,
    lap_rr: f64,
    /// R-operator radial scalar: t = R²φ = (φ'' - q) / r² = q' / r.
    /// At collision (r = 0): t = φ''''(0) / 3, computed via assembled
    /// fourth-derivative collision limits of the partial-fraction blocks.
    t: f64,
    /// First radial derivative of t:
    ///   t_r = dt/dr = (q_rr - t) / r  for r > 0.
    /// At collision, the exact radial limit is t_r(0) = 0.
    t_r: f64,
    /// Second radial derivative of t:
    ///   t_rr = d²t/dr² = [lap_rr + 2 t - (d + 4) q_rr] / r²  for r > 0,
    /// using Delta phi = d q + r² t.
    ///
    /// At collision, the exact radial limit is
    ///   t_rr(0) = φ⁽⁶⁾(0) / 15.
    t_rr: f64,
}


#[derive(Clone, Copy, Debug, Default)]
struct DuchonRegularizedOperatorCore {
    q: f64,
    t: f64,
    t_r: f64,
    t_rr: f64,
}


#[inline(always)]
fn duchon_operator_jets_from_primary_core(
    core: DuchonRegularizedOperatorCore,
    r: f64,
    d: f64,
) -> DuchonRadialJets {
    let r2 = r * r;
    let mut out = DuchonRadialJets {
        q: core.q,
        t: core.t,
        t_r: core.t_r,
        t_rr: core.t_rr,
        ..DuchonRadialJets::default()
    };
    out.q_r = r * out.t;
    out.q_rr = out.t + r * out.t_r;
    out.lap = d * out.q + r2 * out.t;
    out.lap_r = (d + 2.0) * r * out.t + r2 * out.t_r;
    out.lap_rr = (d + 2.0) * out.t + (d + 4.0) * r * out.t_r + r2 * out.t_rr;
    out.phi_r = r * out.q;
    out.phi_rr = out.q + r2 * out.t;
    out.phi_rrr = 3.0 * r * out.t + r2 * out.t_r;

    assert!(
        ((out.phi_rr - (out.q + r * out.q_r)).abs()) <= 1e-10 * out.phi_rr.abs().max(1.0),
        "radial scalar identity failed: phi_rr != q + r*q_r, phi_rr={}, q={}, r={}, q_r={}",
        out.phi_rr,
        out.q,
        r,
        out.q_r
    );
    assert!(
        ((out.phi_rr - (out.q + r2 * out.t)).abs()) <= 1e-10 * out.phi_rr.abs().max(1.0),
        "radial scalar identity failed: phi_rr != q + r2*t, phi_rr={}, q={}, r2={}, t={}",
        out.phi_rr,
        out.q,
        r2,
        out.t
    );
    assert!(
        ((out.lap - (d * out.q + r2 * out.t)).abs()) <= 1e-10 * out.lap.abs().max(1.0),
        "radial scalar identity failed: lap != d*q + r2*t, lap={}, d={}, q={}, r2={}, t={}",
        out.lap,
        d,
        out.q,
        r2,
        out.t
    );

    out
}


#[inline(always)]
fn scaled_log_kappa_derivatives(
    value: f64,
    radial_first: f64,
    radialsecond: f64,
    exponent: f64,
    r: f64,
) -> (f64, f64) {
    // Scaling-law differentiation template
    // For any radial quantity of the form
    //   F(r; kappa) = kappa^a G(kappa r),
    // with psi = log(kappa), one has d/dpsi = kappa d/dkappa.
    //
    // Writing t = kappa r,
    //   F_psi
    //   = kappa d/dkappa [kappa^a G(t)]
    //   = a kappa^a G(t) + kappa^a (kappa r) G'(t)
    //   = a F + r F_r.
    //
    // Differentiating again,
    //   F_psipsi
    //   = d/dpsi [a F + r F_r]
    //   = a F_psi + r (F_r)_psi
    //   = a (a F + r F_r) + r d/dr(F_psi)
    //   = a^2 F + (2a + 1) r F_r + r^2 F_rr.
    //
    // This helper is the common exact formula used for:
    //   - phi            with exponent delta
    //   - q = phi_r / r  with exponent delta + 2
    //   - Delta phi      with exponent delta + 2.
    let first = exponent * value + r * radial_first;
    let second = exponent * exponent * value
        + (2.0 * exponent + 1.0) * r * radial_first
        + r * r * radialsecond;
    (first, second)
}


#[inline(always)]
fn duchon_q_psi_triplet_from_jets(
    jets: &DuchonRadialJets,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    r: f64,
) -> (f64, f64) {
    scaled_log_kappa_derivatives(
        jets.q,
        jets.q_r,
        jets.q_rr,
        duchon_operator_scaling_exponent(p_order, s_order, k_dim),
        r,
    )
}


#[inline(always)]
fn duchon_operator_scaling_exponent(p_order: usize, s_order: usize, k_dim: usize) -> f64 {
    // For the hybrid Duchon spectrum
    //   1 / (|w|^(2p) (kappa^2 + |w|^2)^s),
    // the spatial kernel scales as
    //   phi(r; kappa) = kappa^delta H(kappa r),
    // where
    //   delta = d - 2p - 2s.
    //
    // A first spatial derivative contributes one extra factor of kappa, so
    // phi_r scales like kappa^(delta + 1). Dividing by r gives
    //   q(r; kappa) = phi_r / r = kappa^(delta + 2) Q(kappa r).
    //
    // The Laplacian also contributes two spatial derivatives, so
    //   Delta phi(r; kappa) = kappa^(delta + 2) L(kappa r).
    //
    // Thus both Duchon operator scalars use exponent delta + 2.
    duchon_scaling_exponent(p_order, s_order, k_dim) + 2.0
}


fn duchon_regularized_operator_core(
    r_eval: f64,
    kappa: f64,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<DuchonRegularizedOperatorCore, BasisError> {
    // Assemble the operator scalars with compensated summation because the
    // partial-fraction coefficients can alternate in sign and span many orders
    // of magnitude in higher dimensions.
    let mut q_sum = KahanSum::default();
    let mut t_sum = KahanSum::default();
    let mut t_r_sum = KahanSum::default();
    let mut t_rr_sum = KahanSum::default();

    for (m, coeff) in coeffs.a.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        let (q, t, t_r, t_rr) = duchon_polyharmonic_operator_block_jets(r_eval, m as f64, k_dim)?;
        q_sum.add(coeff * q);
        t_sum.add(coeff * t);
        t_r_sum.add(coeff * t_r);
        t_rr_sum.add(coeff * t_rr);
    }
    // One Bessel-K ladder at z = κ·r serves every Matérn block and every
    // term of their derivative lattices (see [`BesselKLadder`]); the old
    // per-term Bessel calls restarted the seed+recurrence hundreds of times
    // per evaluation point.
    let max_ladder_steps = coeffs
        .b
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, coeff)| **coeff != 0.0)
        .map(|(n, _)| duchon_matern_block_max_ladder_steps(n, k_dim))
        .max();
    if let Some(max_ladder_steps) = max_ladder_steps {
        let ladder =
            BesselKLadder::build(kappa * r_eval, !k_dim.is_multiple_of(2), max_ladder_steps);
        for (n, coeff) in coeffs.b.iter().enumerate().skip(1) {
            if *coeff == 0.0 {
                continue;
            }
            let (q, t, t_r, t_rr) =
                duchon_matern_operator_block_jets_with_ladder(r_eval, kappa, n, k_dim, &ladder)?;
            q_sum.add(coeff * q);
            t_sum.add(coeff * t);
            t_r_sum.add(coeff * t_r);
            t_rr_sum.add(coeff * t_rr);
        }
    }
    Ok(DuchonRegularizedOperatorCore {
        q: q_sum.sum(),
        t: t_sum.sum(),
        t_r: t_r_sum.sum(),
        t_rr: t_rr_sum.sum(),
    })
}


#[inline(always)]
fn duchon_collision_taylor_operator_core(
    r: f64,
    phi_rr_collision: f64,
    t_collision: f64,
    t_rr_collision: f64,
) -> DuchonRegularizedOperatorCore {
    let r2 = r * r;
    let r4 = r2 * r2;
    DuchonRegularizedOperatorCore {
        q: phi_rr_collision + 0.5 * t_collision * r2 + 0.125 * t_rr_collision * r4,
        t: t_collision + 0.5 * t_rr_collision * r2,
        t_r: t_rr_collision * r,
        t_rr: t_rr_collision,
    }
}


fn duchon_radial_jets(
    r: f64,
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<DuchonRadialJets, BasisError> {
    let kappa = 1.0 / length_scale.max(1e-300);
    let r_floor = DUCHON_DERIVATIVE_R_FLOOR_REL * length_scale.max(1e-8);
    let collision_taylor_radius = DUCHON_COLLISION_TAYLOR_REL * length_scale.max(1e-8);
    let r_eval = r.max(r_floor);
    let d = k_dim as f64;

    // Value path keeps the intrinsic diagonal convention used by the actual basis.
    let phi = duchon_matern_kernel_general_from_distance(
        r,
        Some(length_scale),
        p_order,
        s_order,
        k_dim,
        Some(coeffs),
    )?;
    if !phi.is_finite() {
        crate::bail_invalid_basis!(
            "non-finite Duchon radial kernel value at r={r}, length_scale={length_scale}, p={p_order}, s={s_order}, dim={k_dim}"
        );
    }

    // Assemble the operator scalars directly from the partial-fraction blocks.
    // This avoids the unstable off-origin subtraction
    //   t = (phi_rr - phi_r / r) / r^2
    // in high dimensions, where phi_rr and phi_r / r can be enormous and nearly
    // cancel long before the final Duchon operator stays moderate.
    let generic_jets = duchon_operator_jets_from_primary_core(
        duchon_regularized_operator_core(r_eval, kappa, k_dim, coeffs)?,
        r_eval,
        d,
    );
    let mut out = DuchonRadialJets {
        phi,
        ..generic_jets
    };

    // Smoothness check: the collision Taylor expansion requires analytic
    // collision limits (t(0) = φ''''(0)/3, etc.) which only exist when the
    // kernel is sufficiently smooth at the origin: 2(p+s) > d + 2j.
    // For the borderline case (2(p+s) == d+4), φ''''(0) diverges
    // logarithmically and the Taylor carrier cannot represent t(r) accurately.
    // In that regime, keep the generic-path values at r_eval = r_floor.
    let smoothness_order = 2 * (p_order + s_order);
    let collision_q_exists = smoothness_order > k_dim + 2;
    let collision_t_exists = smoothness_order > k_dim + 4;
    let collision_t_rr_exists = smoothness_order > k_dim + 6;

    if r <= collision_taylor_radius.max(r_floor) && collision_t_exists {
        // Tier 2+: full collision Taylor expansion using φ''(0), φ''''(0)/3,
        // and optionally φ⁽⁶⁾(0)/15.  Replaces the generic r_floor path for
        // all radial scalars in the near-origin region.
        let (analytic_phi_rr, _, _) =
            duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, coeffs)?;
        let analytic_t_collision =
            duchon_phi_rrrr_collision(length_scale, p_order, s_order, k_dim, coeffs)? / 3.0;
        let analytic_t_rr_collision = if collision_t_rr_exists {
            duchon_phi_rrrrrr_collision(length_scale, p_order, s_order, k_dim, coeffs)? / 15.0
        } else {
            // t_rr(0) does not exist as a finite limit for this smoothness
            // order, so the smooth-origin carrier must stop at the quadratic
            // term in t(r) and the quartic term in q(r), phi_r(r), phi_rr(r).
            0.0
        };
        let collision_jets = duchon_operator_jets_from_primary_core(
            duchon_collision_taylor_operator_core(
                r,
                analytic_phi_rr,
                analytic_t_collision,
                analytic_t_rr_collision,
            ),
            r,
            d,
        );
        out = DuchonRadialJets {
            phi: out.phi,
            ..collision_jets
        };
    } else if r < r_floor && collision_q_exists {
        // Tier 1: only lower-order collision identities exist.  φ''(0) is
        // finite but φ''''(0) diverges logarithmically at this smoothness
        // order.  Override phi_r, phi_rr, q, q_r, lap, lap_r with exact
        // values; leave t, t_r, t_rr, q_rr, lap_rr at their generic-path
        // values from r_eval = r_floor (best available for the divergent tier).
        let (analytic_phi_rr, _, _) =
            duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, coeffs)?;
        out.phi_r = analytic_phi_rr * r;
        out.phi_rr = analytic_phi_rr;
        out.q = analytic_phi_rr;
        out.q_r = 0.0;
        out.lap = d * analytic_phi_rr;
        out.lap_r = 0.0;
    }
    if !out.phi_r.is_finite()
        || !out.phi_rr.is_finite()
        || !out.phi_rrr.is_finite()
        || !out.q.is_finite()
        || !out.q_r.is_finite()
        || !out.q_rr.is_finite()
        || !out.lap.is_finite()
        || !out.lap_r.is_finite()
        || !out.lap_rr.is_finite()
        || !out.t.is_finite()
        || !out.t_r.is_finite()
        || !out.t_rr.is_finite()
    {
        crate::bail_invalid_basis!(
            "non-finite Duchon radial jets at r={r}, length_scale={length_scale}, p={p_order}, s={s_order}, dim={k_dim}"
        );
    }
    Ok(out)
}


fn duchon_radial_core_psi_triplet(
    r: f64,
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<DuchonRadialCore, BasisError> {
    // Duchon spectral derivation
    // Start from the isotropic spectrum
    //   K^(ω; kappa) ∝ 1 / (|ω|^(2p) * (kappa^2 + |ω|^2)^s),
    // with fixed integer orders p,s and continuous scale
    //   psi = log(kappa),   kappa = 1 / length_scale.
    //
    // Rescaling frequency by ω = kappa ξ gives the full spatial kernel scaling law
    //   phi(r; kappa) = kappa^delta H(kappa r),
    //   delta = d - 2p - 2s.
    //
    // Therefore the exact full-kernel psi derivatives are
    //   phi_psi     = delta * phi + r * phi_r
    //   phi_psipsi  = delta^2 * phi + (2 delta + 1) r phi_r + r^2 phi_rr.
    //
    // The operator scalars are
    //   q(r; kappa) = phi_r(r; kappa) / r
    //   ell(r; kappa) = Δphi(r; kappa) = phi_rr + (d-1) q.
    // Both q and ell scale with exponent delta + 2, so
    //   q_psi       = (delta + 2) q + r q_r
    //   q_psipsi    = (delta + 2)^2 q + (2 delta + 5) r q_r + r^2 q_rr
    // and identically for ell.
    //
    // Once {phi, q, ell} and their psi derivatives are known, the collocation
    // operators follow exactly:
    //   D0[k,j]         = phi(r_kj)
    //   D1[(k,a), j]    = q(r_kj) * (x_{k,a} - c_{j,a})
    //   D2[k,j]         = ell(r_kj)
    // and the penalty Hessians come from the Gram identities
    //   S_psi     = D_psi^T D + D^T D_psi
    //   S_psipsi  = D_psipsi^T D + 2 D_psi^T D_psi + D^T D_psipsi.
    //
    // This helper computes exactly that minimal scalar core:
    //   phi, q = phi_r / r, ell = Δphi
    // together with their first and second psi derivatives.
    //
    // Representation note:
    //   When p > 0 the Duchon kernel is only conditionally positive definite, so
    //   the spatial kernel is canonical only up to polynomial additions. The
    //   formulas in this helper are therefore tied to the specific representative
    //   encoded by the partial-fraction construction and the collision rules used
    //   below. The operator penalties, exact psi derivatives, and center-collision
    //   limits all have to use that same representative or the resulting penalty
    //   geometry will drift across code paths.
    let delta = duchon_scaling_exponent(p_order, s_order, k_dim);
    let jets = duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, coeffs)?;
    let phi = jets.phi;
    let (phi_psi, phi_psi_psi) =
        scaled_log_kappa_derivatives(phi, jets.phi_r, jets.phi_rr, delta, r);
    if r > 1e-10 {
        assert!(
            ((delta * phi + r * jets.phi_r) - phi_psi).abs() < 1e-7_f64.max(1e-7_f64 * phi.abs())
        );
        return Ok(DuchonRadialCore {
            phi: PsiTriplet {
                value: phi,
                psi: phi_psi,
                psi_psi: phi_psi_psi,
            },
        });
    }

    // Continuous center-collision extension for the scalar operator core:
    //   q(0; kappa) = phi_rr(0; kappa)
    //   L(0; kappa) = d * phi_rr(0; kappa).
    //
    // The value and psi derivatives are extracted from the same Taylor
    // coefficient of the assembled partial-fraction kernel. In even dimensions
    // this preserves the log-Riesz finite-part constants, so the collision
    // derivative is not the naive `(delta + 2) * phi_rr` scaling shortcut.
    Ok(DuchonRadialCore {
        phi: PsiTriplet {
            value: phi,
            psi: phi_psi,
            psi_psi: phi_psi_psi,
        },
    })
}


fn duchonphi_rr_collision_psi_triplet(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<(f64, f64, f64), BasisError> {
    // Center-collision rule
    // For a C^2 radial kernel one has
    //   lim_{r->0} phi_r(r)/r = phi_rr(0),
    //   lim_{r->0} Δphi(r)    = d * phi_rr(0).
    //
    // Assemble phi_rr and its psi derivatives by summing the partial-fraction
    // blocks directly.  Do not replace this with the tempting scaling shortcut
    // `phi_rr_psi = (delta + 2) phi_rr`: in even dimensions the log-Riesz
    // representative carries kappa-dependent finite-part constants at the
    // origin, so the shortcut gives the wrong center-collision derivative even
    // when the classical C^2 limit exists.
    duchon_phi_even_derivative_collision_psi_triplet(
        length_scale,
        p_order,
        s_order,
        k_dim,
        coeffs,
        1,
    )
}


/// Euler-Mascheroni constant γ ≈ 0.5772.
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;


/// Digamma function ψ(n) for positive integer n.
///
/// ψ(1) = −γ, ψ(n+1) = −γ + H_n where H_n = Σ_{j=1}^{n} 1/j.
#[inline(always)]
fn digamma_pos_int(n: usize) -> f64 {
    assert!(n >= 1, "digamma_pos_int requires n >= 1: n={n}");
    let mut h = 0.0_f64;
    for j in 1..n {
        h += 1.0 / j as f64;
    }
    -EULER_MASCHERONI + h
}


/// Extract the coefficient of r^{2j} (pure and log-r parts) from a single
/// Matérn partial-fraction block g_n(r) = c · r^ν · K_{|ν|}(κr), where
/// ν = n − d/2.
///
/// Returns `(pure_coeff, log_coeff)` such that the r^{2j} piece of g_n is
///   pure_coeff · r^{2j}  +  log_coeff · r^{2j} · ln(r).
///
/// For even d (integer ν) the expansion uses the DLMF 10.31.1 series for
/// K_n(z) at the origin, which involves digamma / harmonic-number terms.
///
/// For odd d (half-integer ν) the Bessel function is elementary; the Taylor
/// coefficients come from convolving a finite polynomial in 1/r with e^{−κr},
/// and there is no log-r contribution.
fn duchon_matern_block_taylor_r2j(
    kappa: f64,
    n_order: usize,
    k_dim: usize,
    j: usize,
) -> (f64, f64) {
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    // Normalization constant for the Matérn block.
    let c = kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));

    if k_dim.is_multiple_of(2) {
        // Integer ν.
        let nu_int = n_order as i64 - (k_dim as i64) / 2;
        duchon_matern_block_taylor_r2j_integer_nu(kappa, c, nu_int, j)
    } else {
        // Half-integer ν.
        duchon_matern_block_taylor_r2j_half_integer_nu(kappa, c, nu, j)
    }
}


#[inline(always)]
fn psi_power_triplet(value: f64, exponent: f64) -> (f64, f64, f64) {
    (value, exponent * value, exponent * exponent * value)
}


#[inline(always)]
fn psi_power_log_triplet(base: f64, exponent: f64, log_kappa_half: f64) -> (f64, f64, f64) {
    (
        base * log_kappa_half,
        base * (exponent * log_kappa_half + 1.0),
        base * (exponent * exponent * log_kappa_half + 2.0 * exponent),
    )
}


#[inline(always)]
fn add_triplet(dst: &mut (f64, f64, f64), inc: (f64, f64, f64)) {
    dst.0 += inc.0;
    dst.1 += inc.1;
    dst.2 += inc.2;
}


/// Like [`duchon_matern_block_taylor_r2j`], but also returns exact
/// derivatives of the pure/log Taylor coefficients with respect to
/// `psi = log(kappa)`.
fn duchon_matern_block_taylor_r2j_triplet(
    kappa: f64,
    n_order: usize,
    k_dim: usize,
    j: usize,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    let c_const = 1.0
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));
    let c_exp = k_half - n;

    let mut pure = (0.0, 0.0, 0.0);
    let mut log_part = (0.0, 0.0, 0.0);
    let log_kappa_half = (0.5 * kappa).ln();

    if k_dim.is_multiple_of(2) {
        let nu_int = n_order as i64 - (k_dim as i64) / 2;
        let mu = nu_int.unsigned_abs() as usize;
        let sign_mu = if mu.is_multiple_of(2) { 1.0 } else { -1.0 };

        if nu_int >= 0 {
            let nu_usize = nu_int as usize;

            if j < nu_usize {
                let sign = if j.is_multiple_of(2) { 1.0 } else { -1.0 };
                let power = 2 * j as i32 - nu_usize as i32;
                let coeff = 0.5 * sign * gamma_lanczos((nu_usize - j) as f64)
                    / gamma_lanczos((j + 1) as f64)
                    * 2.0_f64.powi(-power);
                let exponent = c_exp + power as f64;
                let value = c_const * coeff * kappa.powf(exponent);
                add_triplet(&mut pure, psi_power_triplet(value, exponent));
            }

            if j >= nu_usize {
                let k = j - nu_usize;
                let inv_fac = 1.0
                    / (gamma_lanczos((k + 1) as f64) * gamma_lanczos((nu_usize + k + 1) as f64));
                let power = (2 * k + nu_usize) as i32;
                let exponent = c_exp + power as f64;
                let kp_base = c_const * kappa.powf(exponent) * 2.0_f64.powi(-power);

                let log_base = -sign_mu * kp_base * inv_fac;
                add_triplet(&mut log_part, psi_power_triplet(log_base, exponent));
                add_triplet(
                    &mut pure,
                    psi_power_log_triplet(log_base, exponent, log_kappa_half),
                );

                let psi_sum = digamma_pos_int(k + 1) + digamma_pos_int(nu_usize + k + 1);
                let digamma_base = sign_mu * 0.5 * kp_base * inv_fac * psi_sum;
                add_triplet(&mut pure, psi_power_triplet(digamma_base, exponent));
            }
        } else {
            let k = j;
            let inv_fac =
                1.0 / (gamma_lanczos((k + 1) as f64) * gamma_lanczos((mu + k + 1) as f64));
            let power = (mu + 2 * k) as i32;
            let exponent = c_exp + power as f64;
            let kp_base = c_const * kappa.powf(exponent) * 2.0_f64.powi(-power);

            let log_base = -sign_mu * kp_base * inv_fac;
            add_triplet(&mut log_part, psi_power_triplet(log_base, exponent));
            add_triplet(
                &mut pure,
                psi_power_log_triplet(log_base, exponent, log_kappa_half),
            );

            let psi_sum = digamma_pos_int(k + 1) + digamma_pos_int(mu + k + 1);
            let digamma_base = sign_mu * 0.5 * kp_base * inv_fac * psi_sum;
            add_triplet(&mut pure, psi_power_triplet(digamma_base, exponent));
        }
    } else {
        let nu_abs = nu.abs();
        let l = (2.0 * nu_abs - 1.0).round().max(0.0) as usize;
        let prefactor_const = (std::f64::consts::PI / 2.0).sqrt();
        let prefactor_exp = -0.5;
        let target = 2 * j;

        for i in 0..=l {
            let c_i = gamma_lanczos((l + i + 1) as f64)
                / (gamma_lanczos((i + 1) as f64) * gamma_lanczos((l - i + 1) as f64));
            let p_f64 = nu - 0.5 - i as f64;
            let p_round = p_f64.round() as i64;
            if (p_f64 - p_round as f64).abs() > 1e-12 {
                continue;
            }
            let q_needed = target as i64 - p_round;
            if q_needed < 0 {
                continue;
            }
            let q = q_needed as usize;
            let sign = if q.is_multiple_of(2) { 1.0 } else { -1.0 };
            let exponent = c_exp + prefactor_exp - i as f64 + q as f64;
            let value = c_const * prefactor_const * c_i * 2.0_f64.powi(-(i as i32)) * sign
                / gamma_lanczos((q + 1) as f64)
                * kappa.powf(exponent);
            add_triplet(&mut pure, psi_power_triplet(value, exponent));
        }
    }

    (pure, log_part)
}


/// Taylor r^{2j} coefficients for integer-ν Matérn block.
///
/// Uses the K_μ(z) expansion for integer μ = |ν| ≥ 0 (A&S 9.6.11 / DLMF 10.31.1):
///
///   K_μ(z) = (−1)^{μ+1} I_μ(z) ln(z/2)
///          + ½ Σ_{k=0}^{μ−1} (−1)^k (μ−k−1)!/k! · (z/2)^{2k−μ}   [singular]
///          + (−1)^μ · ½ Σ_{k≥0} (z/2)^{μ+2k}/(k!(μ+k)!)
///                              · [ψ(k+1)+ψ(μ+k+1)]                  [regular]
///
/// Multiplied by r^ν, the r^{2j} coefficient is assembled from the singular
/// and/or regular+log series depending on the sign and magnitude of ν.
fn duchon_matern_block_taylor_r2j_integer_nu(
    kappa: f64,
    c: f64,
    nu_int: i64,
    j: usize,
) -> (f64, f64) {
    let mu = nu_int.unsigned_abs() as usize; // |ν|

    // Helper: compute (κ/2)^p for integer p.
    let kappa_half = 0.5 * kappa;

    if nu_int >= 0 {
        let nu = nu_int as usize;
        // Two potential sources for the r^{2j} coefficient:
        //
        // 1) Singular sum:  contributes when j ≤ ν−1 (the k=j term gives r^{2j}).
        // 2) Regular+log sum: contributes when 2ν+2k = 2j, i.e. k = j−ν ≥ 0.
        let mut pure = 0.0;
        let mut log_part = 0.0;

        // Source 1: singular sum at k = j.
        if j < nu {
            // (1/2) · (−1)^j · (ν−j−1)!/j! · (κ/2)^{2j−ν}
            let sign = if j.is_multiple_of(2) { 1.0 } else { -1.0 };
            let coeff = sign * gamma_lanczos((nu - j) as f64) / gamma_lanczos((j + 1) as f64)
                * kappa_half.powi(2 * j as i32 - nu as i32)
                * 0.5;
            pure += coeff;
        }

        // Source 2: regular+log sum at k = j − ν.
        if j >= nu {
            let k = j - nu;
            let inv_fac =
                1.0 / (gamma_lanczos((k + 1) as f64) * gamma_lanczos((nu + k + 1) as f64));
            let kp = kappa_half.powi(2 * k as i32 + nu as i32);
            let sign_mu = if mu.is_multiple_of(2) { 1.0 } else { -1.0 }; // (−1)^μ

            // Log coefficient: (−1)^{μ+1} · (κ/2)^{ν+2k} / (k!(ν+k)!)
            log_part += -sign_mu * kp * inv_fac;

            // Pure coefficient from the log series (ln(κ/2) piece):
            //   (−1)^{μ+1} · (κ/2)^{ν+2k} / (k!(ν+k)!) · ln(κ/2)
            // Plus the digamma series:
            //   (−1)^μ · ½ · (κ/2)^{ν+2k} / (k!(ν+k)!) · [ψ(k+1)+ψ(ν+k+1)]
            let psi_sum = digamma_pos_int(k + 1) + digamma_pos_int(nu + k + 1);
            pure += -sign_mu * kp * inv_fac * kappa_half.ln();
            pure += sign_mu * 0.5 * kp * inv_fac * psi_sum;
        }

        (c * pure, c * log_part)
    } else {
        // ν < 0: mu = |ν| > 0.
        // Singular sum gives powers r^{2ν}, ..., r^{−2} (all negative).
        // Regular+log sum gives r^0, r^2, r^4, ... at k = j.
        let k = j;
        let inv_fac = 1.0 / (gamma_lanczos((k + 1) as f64) * gamma_lanczos((mu + k + 1) as f64));
        let kp = kappa_half.powi(mu as i32 + 2 * k as i32);
        let sign_mu = if mu.is_multiple_of(2) { 1.0 } else { -1.0 };

        // Log coefficient: (−1)^{μ+1} · (κ/2)^{μ+2k} / (k!(μ+k)!)
        let log_part = -sign_mu * kp * inv_fac;

        // Pure coefficient: log-series ln(κ/2) piece + digamma piece.
        let psi_sum = digamma_pos_int(k + 1) + digamma_pos_int(mu + k + 1);
        let pure =
            -sign_mu * kp * inv_fac * kappa_half.ln() + sign_mu * 0.5 * kp * inv_fac * psi_sum;

        (c * pure, c * log_part)
    }
}


/// Taylor r^{2j} coefficients for half-integer-ν Matérn block.
///
/// For half-integer |ν| = l + ½, K_{l+½}(z) is elementary:
///   K_{l+½}(z) = √(π/(2z)) · e^{−z} · Σ_{i=0}^{l} C_i · (2z)^{−i}
/// where C_i = (l+i)! / (i! · (l−i)!).
///
/// The product r^ν · K_{|ν|}(κr) expands as an explicit polynomial in r
/// (including possible negative powers) times e^{−κr}.  The r^{2j} Taylor
/// coefficient is obtained by convolving with the exponential series
/// e^{−κr} = Σ_q (−κ)^q r^q / q!.  There is never a log-r contribution.
fn duchon_matern_block_taylor_r2j_half_integer_nu(
    kappa: f64,
    c: f64,
    nu: f64,
    j: usize,
) -> (f64, f64) {
    let nu_abs = nu.abs();
    let l = (2.0 * nu_abs - 1.0).round().max(0.0) as usize;
    // Compute the polynomial coefficients C_i / (2κ)^i for each r-power.
    //
    // r^ν · K_{l+½}(κr) = √(π/(2κ)) · e^{−κr} · Σ_{i=0}^{l} C_i (2κ)^{−i} r^{ν−½−i}
    //
    // (since K_{l+½}(z) = √(π/(2z)) e^{−z} Σ C_i (2z)^{−i}, multiplying by
    // r^ν gives r^{ν−½} from the √(π/(2κr)) factor, then each (2κr)^{−i}
    // contributes r^{−i}.)
    let prefactor = (std::f64::consts::PI / (2.0 * kappa)).sqrt();

    // Polynomial term i has r-power = ν − 0.5 − i.  We need to convolve
    // each monomial with e^{−κr} = Σ_q (−κ)^q r^q / q! and extract the
    // r^{2j} coefficient.
    //
    // For monomial r^p (p = ν−½−i) times e^{−κr}: the r^{2j} coefficient is
    //   (−κ)^{2j−p} / (2j−p)!   when 2j−p is a non-negative integer.
    let target = 2 * j;
    let mut pure = 0.0;

    for i in 0..=l {
        let c_i = gamma_lanczos((l + i + 1) as f64)
            / (gamma_lanczos((i + 1) as f64) * gamma_lanczos((l - i + 1) as f64));
        let inv_2kappa_i = (2.0 * kappa).powi(-(i as i32));

        // r-power of this polynomial term.
        let p_f64 = nu - 0.5 - i as f64;
        let p_round = p_f64.round() as i64;
        if (p_f64 - p_round as f64).abs() > 1e-12 {
            // Not integer/half-integer aligned — should not happen for half-integer ν.
            continue;
        }
        let q_needed = target as i64 - p_round;
        if q_needed < 0 {
            continue;
        }
        let q = q_needed as usize;
        let exp_coeff = (-kappa).powi(q as i32) / gamma_lanczos((q + 1) as f64);
        pure += c_i * inv_2kappa_i * exp_coeff;
    }

    (c * prefactor * pure, 0.0) // No log contribution for half-integer ν.
}


/// Extract the r^{2j} Taylor coefficient from a polyharmonic block Φ_m(r).
///
/// Non-log case (d odd, or d even with m < d/2): Φ_m = c · r^α with α = 2m − d.
///   Only contributes when α = 2j exactly: pure_coeff = c, log_coeff = 0.
///
/// Log case (d even, m ≥ d/2): Φ_m = c · r^α · ln(r).
///   Only contributes when α = 2j: pure_coeff = 0, log_coeff = c.
fn duchon_polyharmonic_block_taylor_r2j(m: usize, k_dim: usize, j: usize) -> (f64, f64) {
    let k_half = 0.5 * k_dim as f64;
    let alpha = 2 * m as i64 - k_dim as i64;

    if alpha != 2 * j as i64 {
        return (0.0, 0.0);
    }

    // α = 2j: compute the normalization constant.
    if k_dim.is_multiple_of(2) && m >= k_dim / 2 {
        // Log case: Φ_m = c · r^α · ln(r).
        let c = polyharmonic_log_sign(m, k_dim)
            / (2.0_f64.powi((2 * m - 1) as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m as f64)
                * gamma_lanczos((m - k_dim / 2 + 1) as f64));
        (0.0, c)
    } else {
        // Non-log case: Φ_m = c · r^α.
        let c = gamma_lanczos(k_half - m as f64)
            / (4.0_f64.powi(m as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m as f64));
        (c, 0.0)
    }
}


/// Compute the even-order radial derivative φ^{(2j)}(0) from analytic Taylor
/// coefficients of the partial-fraction blocks.
///
/// For a C^{2j} radial kernel with Taylor expansion φ(r) = Σ_k a_{2k} r^{2k},
/// φ^{(2j)}(0) = (2j)! · a_{2j}.  Each partial-fraction block (polyharmonic
/// and Matérn) has a computable r^{2j} Taylor coefficient (both pure and
/// ln(r) parts).  The ln(r) contributions cancel across blocks whenever the
/// kernel is sufficiently smooth; the pure coefficients sum to give a_{2j}.
///
/// Existence condition (kernel is C^{2j} at the origin):
///   2(p + s) > d + 2j.
///
/// When this condition fails (borderline or insufficient smoothness), the
/// derivative is not a finite collision limit. Callers must reject that model
/// upstream rather than regularize it at an arbitrary floor radius.
fn duchon_phi_even_derivative_collision(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
    j: usize,
) -> Result<f64, BasisError> {
    let smoothness_order = 2 * (p_order + s_order);
    let required = k_dim + 2 * j;

    if smoothness_order <= required {
        // Smallest integer power admitting phi^{(2j)}(0): 2(p+s) > k_dim+2j.
        let min_power = (required / 2 + 1).saturating_sub(p_order);
        crate::bail_invalid_basis!(
            "Duchon collision derivative phi^({}) requires 2*(p+s) > dimension+{}; got 2*(p+s)={}, dimension={}, p={}, s={}. \
             This path needs the {}-order radial-kernel derivative at the origin, which is finite only for a smoother spline: raise power to >= {} (or reduce the joint smooth's dimension).",
            2 * j,
            2 * j,
            smoothness_order,
            k_dim,
            p_order,
            s_order,
            2 * j,
            min_power
        );
    }

    // Analytic path: extract per-block Taylor r^{2j} coefficients and sum.
    let kappa = 1.0 / length_scale.max(1e-300);
    let mut total_pure = KahanSum::default();
    let mut total_log = KahanSum::default();
    let mut total_log_abs_scale = KahanSum::default();

    // Polyharmonic blocks.
    for (m, &a_m) in coeffs.a.iter().enumerate().skip(1) {
        if a_m == 0.0 {
            continue;
        }
        let (pure, log) = duchon_polyharmonic_block_taylor_r2j(m, k_dim, j);
        total_pure.add(a_m * pure);
        total_log.add(a_m * log);
        total_log_abs_scale.add((a_m * log).abs());
    }

    // Matérn blocks.
    for (n, &b_n) in coeffs.b.iter().enumerate().skip(1) {
        if b_n == 0.0 {
            continue;
        }
        let (pure, log) = duchon_matern_block_taylor_r2j(kappa, n, k_dim, j);
        total_pure.add(b_n * pure);
        total_log.add(b_n * log);
        total_log_abs_scale.add((b_n * log).abs());
    }
    let total_pure = total_pure.sum();
    let total_log = total_log.sum();
    let total_log_abs_scale = total_log_abs_scale.sum();

    // The ln(r) coefficients should cancel to zero (guaranteed by the PFD
    // identity when 2(p+s) > d+2j).  Check this as a sanity guard.
    let log_cancel_tol = 1e-10 * total_log_abs_scale.max(total_pure.abs()).max(1e-30);
    if total_log.abs() > log_cancel_tol {
        crate::bail_invalid_basis!(
            "Duchon Taylor a_{} log-coefficient did not cancel: log={total_log:.6e}, pure={total_pure:.6e}; \
             log_abs_scale={total_log_abs_scale:.6e}, tol={log_cancel_tol:.6e}; p={p_order}, s={s_order}, d={k_dim}",
            2 * j
        );
    }

    // φ^{(2j)}(0) = (2j)! · a_{2j}
    let factorial_2j = gamma_lanczos((2 * j + 1) as f64);
    Ok(factorial_2j * total_pure)
}


fn duchon_phi_even_derivative_collision_psi_triplet(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
    j: usize,
) -> Result<(f64, f64, f64), BasisError> {
    let smoothness_order = 2 * (p_order + s_order);
    let required = k_dim + 2 * j;

    if smoothness_order <= required {
        // Smallest integer power admitting the phi^{(2j)} psi triplet: 2(p+s) > k_dim+2j.
        let min_power = (required / 2 + 1).saturating_sub(p_order);
        crate::bail_invalid_basis!(
            "Duchon collision derivative phi^({}) psi triplet requires 2*(p+s) > dimension+{}; got 2*(p+s)={}, dimension={}, p={}, s={}. \
             The exact two-block / transformation-normal path needs analytic length-scale derivatives of the kernel, which are finite only for a smoother spline: raise power to >= {} (or reduce the joint smooth's dimension).",
            2 * j,
            2 * j,
            smoothness_order,
            k_dim,
            p_order,
            s_order,
            min_power
        );
    }

    let kappa = 1.0 / length_scale.max(1e-300);
    let mut value = KahanSum::default();
    let mut psi = KahanSum::default();
    let mut psi_psi = KahanSum::default();
    let mut log_value = KahanSum::default();
    let mut log_psi = KahanSum::default();
    let mut log_psi_psi = KahanSum::default();
    let mut log_abs_scale = KahanSum::default();

    for (m, &a_m) in coeffs.a.iter().enumerate().skip(1) {
        if a_m == 0.0 {
            continue;
        }
        let alpha_m = duchon_coeff_exponents(p_order, s_order, m);
        let (pure, log) = duchon_polyharmonic_block_taylor_r2j(m, k_dim, j);
        value.add(a_m * pure);
        psi.add(alpha_m * a_m * pure);
        psi_psi.add(alpha_m * alpha_m * a_m * pure);
        log_value.add(a_m * log);
        log_psi.add(alpha_m * a_m * log);
        log_psi_psi.add(alpha_m * alpha_m * a_m * log);
        log_abs_scale.add((a_m * log).abs());
        log_abs_scale.add((alpha_m * a_m * log).abs());
        log_abs_scale.add((alpha_m * alpha_m * a_m * log).abs());
    }

    for (n, &b_n) in coeffs.b.iter().enumerate().skip(1) {
        if b_n == 0.0 {
            continue;
        }
        let beta_n = duchon_coeff_exponents(p_order, s_order, n);
        let (pure, log) = duchon_matern_block_taylor_r2j_triplet(kappa, n, k_dim, j);
        value.add(b_n * pure.0);
        psi.add(beta_n * b_n * pure.0 + b_n * pure.1);
        psi_psi.add(beta_n * beta_n * b_n * pure.0 + 2.0 * beta_n * b_n * pure.1 + b_n * pure.2);
        log_value.add(b_n * log.0);
        log_psi.add(beta_n * b_n * log.0 + b_n * log.1);
        log_psi_psi.add(beta_n * beta_n * b_n * log.0 + 2.0 * beta_n * b_n * log.1 + b_n * log.2);
        let log_v = b_n * log.0;
        let log_p = beta_n * b_n * log.0 + b_n * log.1;
        let log_pp = beta_n * beta_n * b_n * log.0 + 2.0 * beta_n * b_n * log.1 + b_n * log.2;
        log_abs_scale.add(log_v.abs());
        log_abs_scale.add(log_p.abs());
        log_abs_scale.add(log_pp.abs());
    }

    let value = value.sum();
    let psi = psi.sum();
    let psi_psi = psi_psi.sum();
    let log_value = log_value.sum();
    let log_psi = log_psi.sum();
    let log_psi_psi = log_psi_psi.sum();
    let log_abs_scale = log_abs_scale.sum();
    let scale = value.abs().max(psi.abs()).max(psi_psi.abs()).max(1e-30);
    let log_cancel_tol = 1e-10 * log_abs_scale.max(scale);
    if log_value.abs().max(log_psi.abs()).max(log_psi_psi.abs()) > log_cancel_tol {
        crate::bail_invalid_basis!(
            "Duchon Taylor a_{} log-coefficient derivative did not cancel: \
             log=({log_value:.6e}, {log_psi:.6e}, {log_psi_psi:.6e}), \
             value=({value:.6e}, {psi:.6e}, {psi_psi:.6e}), log_abs_scale={log_abs_scale:.6e}, tol={log_cancel_tol:.6e}; \
             p={p_order}, s={s_order}, d={k_dim}",
            2 * j
        );
    }

    let factorial_2j = gamma_lanczos((2 * j + 1) as f64);
    Ok((
        factorial_2j * value,
        factorial_2j * psi,
        factorial_2j * psi_psi,
    ))
}


/// Assemble φ''''(0) from the partial-fraction blocks using analytic Taylor
/// coefficients.
///
/// For a radial kernel with Taylor expansion φ(r) = a₀ + a₂r² + a₄r⁴ + ...,
/// we have φ''''(0) = 24 a₄.  This is used to compute the collision limit
/// t(0) = φ''''(0) / 3, where t = R²φ = (φ'' - q) / r².
///
/// Each partial-fraction block (polyharmonic and Matérn) has a known Taylor
/// expansion around r = 0; the r⁴ coefficient a₄ is extracted from the series
/// and summed.  This avoids the catastrophic cancellation that occurs when
/// evaluating divergent block derivatives at a small floor radius.
fn duchon_phi_rrrr_collision(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<f64, BasisError> {
    duchon_phi_even_derivative_collision(length_scale, p_order, s_order, k_dim, coeffs, 2)
}


/// Assemble φ⁽⁶⁾(0) from the partial-fraction blocks using analytic Taylor
/// coefficients.
///
/// For a radial kernel with Taylor expansion φ(r) = a₀ + a₂r² + a₄r⁴ + a₆r⁶ + ...,
/// we have φ⁽⁶⁾(0) = 720 a₆. This gives the collision limit
///   t_rr(0) = φ⁽⁶⁾(0) / 15
/// for t = R²φ.
///
/// Like [`duchon_phi_rrrr_collision`], this extracts per-block Taylor
/// coefficients analytically rather than evaluating divergent derivatives at
/// a small floor radius.
fn duchon_phi_rrrrrr_collision(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<f64, BasisError> {
    duchon_phi_even_derivative_collision(length_scale, p_order, s_order, k_dim, coeffs, 3)
}


fn build_duchon_design_psi_derivativeswithworkspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<ScalarDesignPsiDerivatives, BasisError> {
    let length_scale = spec.length_scale.ok_or_else(|| {
        BasisError::InvalidInput(
            "exact Duchon log-kappa derivatives require hybrid Duchon with length_scale"
                .to_string(),
        )
    })?;
    // Exact Duchon design derivatives:
    // 1. evaluate phi_psi and phi_psipsi at each data/center distance
    // 2. project the kernel block with the same nullspace constraint used by the basis
    // 3. append polynomial columns; their psi derivatives are zero because p and s are fixed
    // 4. apply any frozen identifiability transform
    let effective_nullspace_order = duchon_effective_nullspace_order(centers, spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    let s_order = spec.power_as_usize();
    let kappa = 1.0 / length_scale;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, kappa);
    let z_kernel =
        kernel_constraint_nullspace(centers, effective_nullspace_order, &mut workspace.cache)?;
    let poly_cols = polynomial_block_from_order(data, effective_nullspace_order).ncols();
    let p_padded = z_kernel.ncols() + poly_cols;
    if let Some(zf) = identifiability_transform
        && p_padded != zf.nrows()
    {
        crate::bail_dim_basis!(
            "Duchon identifiability transform mismatch in design derivatives: local cols={}, transform rows={}",
            p_padded,
            zf.nrows()
        );
    }
    let p_final = identifiability_transform
        .map(|zf| zf.ncols())
        .unwrap_or(p_padded);
    build_scalar_design_psi_derivatives_shared(
        data,
        centers,
        spec.aniso_log_scales.as_deref(),
        p_final,
        Some(z_kernel),
        identifiability_transform.cloned(),
        poly_cols,
        RadialScalarKind::Duchon {
            length_scale,
            p_order,
            s_order,
            dim: data.ncols(),
            coeffs,
        },
        duchon_scaling_exponent(p_order, s_order, data.ncols()),
    )
}


pub fn build_duchon_basis_log_kappa_derivative(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_basis_log_kappa_derivativewithworkspace(data, spec, &mut workspace)
}


pub fn build_duchon_basis_log_kappa_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let mut bundle = build_duchon_basis_log_kappa_derivativeswithworkspace(data, spec, workspace)?;
    bundle.first.implicit_operator = bundle.implicit_operator;
    Ok(bundle.first)
}


pub fn build_duchon_basis_log_kappa_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_basis_log_kappa_derivativeswithworkspace(data, spec, &mut workspace)
}


fn duchon_operator_penalties_requested(spec: &DuchonOperatorPenaltySpec) -> bool {
    matches!(spec.mass, OperatorPenaltySpec::Active { .. })
        || matches!(spec.tension, OperatorPenaltySpec::Active { .. })
        || matches!(spec.stiffness, OperatorPenaltySpec::Active { .. })
}


pub fn build_duchon_basis_log_kappa_derivativeswithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    if spec.periodic.is_some() {
        return build_periodic_duchon_basis_log_kappa_derivativeswithworkspace(
            data, spec, workspace,
        );
    }
    let (centers, identifiability_transform) =
        prepare_duchon_derivative_contextwithworkspace(data, spec, workspace)?;
    let operator_collocation_points =
        if duchon_operator_penalties_requested(&spec.operator_penalties) {
            let m = (DUCHON_COLLOCATION_OVERSAMPLE * centers.nrows()).min(data.nrows());
            Some(select_thin_plate_knots(data, m)?)
        } else {
            None
        };
    build_duchon_basis_log_kappa_derivativeswith_collocationwithworkspace(
        data,
        spec,
        centers.view(),
        identifiability_transform.as_ref(),
        operator_collocation_points
            .as_ref()
            .map(|points| points.view()),
        workspace,
    )
}


pub(crate) fn build_duchon_basis_log_kappa_derivativeswith_collocationwithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    centers: ArrayView2<'_, f64>,
    identifiability_transform: Option<&Array2<f64>>,
    operator_collocation_points: Option<ArrayView2<'_, f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    let design_derivatives = build_duchon_design_psi_derivativeswithworkspace(
        data,
        centers,
        spec,
        identifiability_transform,
        workspace,
    )?;
    let (native_sources, native_first, native_second) =
        build_duchon_native_penalty_psi_derivatives(
            centers,
            spec,
            identifiability_transform,
            workspace,
        )?;
    let (operator_sources, operator_first, operator_second) = if duchon_operator_penalties_requested(
        &spec.operator_penalties,
    ) {
        let Some(collocation_points) = operator_collocation_points else {
            crate::bail_invalid_basis!(
                "Duchon log-kappa operator penalty derivatives require realized collocation points"
            );
        };
        build_duchon_operator_penalty_psi_derivatives(
            collocation_points,
            centers,
            spec,
            identifiability_transform,
            workspace,
        )?
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    let mut penalties_derivative = Vec::with_capacity(native_first.len() + operator_first.len());
    penalties_derivative.extend(native_first);
    penalties_derivative.extend(operator_first);
    let mut penaltiessecond_derivative =
        Vec::with_capacity(native_second.len() + operator_second.len());
    penaltiessecond_derivative.extend(native_second);
    penaltiessecond_derivative.extend(operator_second);
    let expected_derivative_count = native_sources.len() + operator_sources.len();
    if penalties_derivative.len() != expected_derivative_count {
        crate::bail_invalid_basis!(
            "Duchon penalty derivative count mismatch: assembled {}, expected {} from active penalty sources",
            penalties_derivative.len(),
            expected_derivative_count
        );
    }
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


pub fn build_duchon_basis_log_kappasecond_derivative(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_basis_log_kappasecond_derivativewithworkspace(data, spec, &mut workspace)
}


pub fn build_duchon_basis_log_kappasecond_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let mut bundle = build_duchon_basis_log_kappa_derivativeswithworkspace(data, spec, workspace)?;
    bundle.second.implicit_operator = bundle.implicit_operator;
    Ok(bundle.second)
}


/// Multiplicative amplification factor that lifts an underflowing Duchon
/// kernel back into a representable range. Probes max|K_CC| (the kernel at
/// every center pair) and returns `1/max` when the kernel collapses to the
/// double-precision noise floor; otherwise returns `1.0`.
///
/// **Why**: in high d with a small length scale the spectral normalization
/// `c = κ^{d/2-n} / ((2π)^{d/2}·2^{n-1}·Γ(n))` of the Matérn block is `~1e-14`,
/// driving every `K(r) = c · r^ν · K_ν(κr)` to `~1e-16`. Downstream
/// `B^T B` is then at `~1e-32` — below `eps²` — and the spectral whitener
/// truncates everything as noise, even though the basis is mathematically
/// well-defined.
///
/// Rescaling the basis by α = 1/max|K_CC| produces the same predictions
/// (β rescales by α, REML's λ adapts). Since the probe is computed from
/// `centers + kernel parameters` which are stored verbatim in
/// `BasisMetadata::Duchon`, prediction recomputes an identical α — so
/// fit-time and predict-time bases share a single coefficient frame.
fn duchon_kernel_amplification(
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    p_order: usize,
    s_order: usize,
    d: usize,
    aniso_log_scales: Option<&[f64]>,
    coeffs: Option<&DuchonPartialFractionCoeffs>,
    pure_poly_coeff: Option<&PolyharmonicBlockCoeff>,
) -> f64 {
    let k = centers.nrows();
    if k == 0 {
        return 1.0;
    }
    let axis_scales = aniso_log_scales.map(aniso_axis_scales);
    let mut max_abs = 0.0_f64;
    for i in 0..k {
        for j in i..k {
            let r = if let Some(scales) = axis_scales.as_deref() {
                aniso_distance_rows_with_scales(centers, i, centers, j, scales)
            } else {
                euclidean_distance_rows(centers, i, centers, j)
            };
            let val = if let Some(ppc) = pure_poly_coeff {
                ppc.eval(r)
            } else {
                match duchon_matern_kernel_general_from_distance(
                    r,
                    length_scale,
                    p_order,
                    s_order,
                    d,
                    coeffs,
                ) {
                    Ok(v) => v,
                    Err(_) => continue,
                }
            };
            if val.abs() > max_abs {
                max_abs = val.abs();
            }
        }
    }
    // Only amplify when the kernel has underflowed. The 1e-10 threshold is
    // well above any meaningful smoothing-relevant kernel scale yet far from
    // 1.0, so well-conditioned kernels pass through unchanged (α = 1).
    if max_abs > 0.0 && max_abs < 1e-10 {
        1.0 / max_abs
    } else {
        1.0
    }
}


/// Scalar kernel amplification `α` that [`build_duchon_basis`] applies to the
/// pure scale-free polyharmonic Duchon kernel block (`length_scale = None`,
/// `power = 0`, no anisotropy) for the given requested null-space `order`.
///
/// This is the exact factor the forward design multiplies into `K(t,C)` before
/// the null-space projection `Z`, so any derivative path that differentiates
/// that forward design (e.g. the `duchon_basis_with_jet` FFI, which builds its
/// forward via [`build_duchon_basis`] with these same parameters) must scale
/// its raw radial jet by the identical `α`. Returning it from the Rust core —
/// rather than recomputing the amplification probe in a wrapper — keeps the
/// derivative bit-for-bit consistent with the forward and avoids duplicating
/// the spectral-normalization math outside this module.
///
/// The requested `order` is internally degraded via
/// [`duchon_effective_nullspace_order`] exactly as the forward builder does, so
/// the polyharmonic order `p` used by the amplification probe matches.
pub fn duchon_pure_kernel_amplification(
    centers: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
    power: f64,
) -> f64 {
    let dim = centers.ncols();
    if dim == 0 || centers.nrows() == 0 {
        return 1.0;
    }
    let effective_order = duchon_effective_nullspace_order(centers, order);
    let p_order = duchon_p_from_nullspace_order(effective_order);
    let s_order: f64 = power;
    let pure_poly_coeff =
        PolyharmonicBlockCoeff::new(pure_duchon_block_order(p_order, s_order), dim);
    duchon_kernel_amplification(
        centers,
        None,
        p_order,
        duchon_power_to_usize(s_order),
        dim,
        None,
        None,
        Some(&pure_poly_coeff),
    )
}


fn build_duchon_basis_designwithworkspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    workspace: &mut BasisWorkspace,
) -> Result<DuchonBasisDesign, BasisError> {
    let n = data.nrows();
    let d = data.ncols();
    let k = centers.nrows();

    if d == 0 {
        crate::bail_invalid_basis!("Duchon basis requires at least one covariate dimension");
    }
    if k == 0 {
        crate::bail_invalid_basis!("Duchon basis requires at least one center");
    }
    if centers.ncols() != d {
        crate::bail_dim_basis!(
            "Duchon basis dimension mismatch: data has {d} columns, centers have {}",
            centers.ncols()
        );
    }
    if data.iter().any(|v| !v.is_finite()) || centers.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("Duchon basis requires finite data and center values");
    }
    // Auto-degrade the null-space order to Zero when centers are insufficient
    // to span the requested polynomial block; emits a warning inside the helper.
    let nullspace_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(nullspace_order);
    let s_order: f64 = power;
    // Gate on the spectral power the kernel actually evaluates: the scale-free
    // native Gram uses the literal fractional `power`, but the hybrid
    // (`length_scale=Some`) partial-fraction kernel reads `s` back through
    // `duchon_power_to_usize` (truncating a fractional `power`). Validating the
    // raw fractional power on the hybrid path would desync the `2(p+s) > d`
    // gate from the realized kernel and let the non-finite-at-origin case
    // through (gh#750).
    let validation_power = if length_scale.is_some() {
        duchon_power_to_usize(s_order) as f64
    } else {
        s_order
    };
    validate_duchon_kernel_orders(length_scale, p_order, validation_power, d)?;

    let poly_block = polynomial_block_from_order(data, nullspace_order);
    // Z spans null(Q^T), where Q contains polynomial side conditions at centers.
    // Reparameterizing alpha = Z gamma enforces conditional-PD constraints once
    // and yields free-parameter penalty gamma^T (Z^T K_CC Z) gamma.
    let z = kernel_constraint_nullspace(centers, nullspace_order, &mut workspace.cache)?;

    let coeffs = length_scale.map(|ls| {
        duchon_partial_fraction_coeffs(
            p_order,
            duchon_power_to_usize(s_order),
            1.0 / ls.max(1e-300),
        )
    });

    // Practical safe operating range (document Eq. D.2):
    //   κ in [1e-2 / r_max, 1e2 / r_min]
    // where r_min/r_max are pairwise center distance extrema. Under
    // anisotropy the kernel metric is y-space (y_a = exp(η_a) x_a), so
    // the relevant r_min/r_max are y-space pairwise distances, not raw.
    // We keep user-provided κ but emit a warning outside this regime.
    let warn_bounds = match (length_scale, aniso_log_scales) {
        (Some(_), Some(eta)) => {
            let y_centers = points_in_aniso_y_space(centers, eta);
            pairwise_distance_bounds(y_centers.view())
        }
        (Some(_), None) => pairwise_distance_bounds(centers),
        (None, _) => None,
    };
    if let (Some(length_scale), Some((r_min, r_max))) = (length_scale, warn_bounds) {
        let kappa = 1.0 / length_scale.max(1e-300);
        let kappa_lo = 1e-2 / r_max;
        let kappa_hi = 1e2 / r_min;
        if kappa < kappa_lo || kappa > kappa_hi {
            log::debug!(
                "Duchon κ={} is outside recommended range [{}, {}] derived from centers (r_min={}, r_max={}); numerical conditioning may degrade",
                kappa,
                kappa_lo,
                kappa_hi,
                r_min,
                r_max
            );
        }
    }

    let kernel_cols = z.ncols();
    let poly_cols = poly_block.ncols();
    let total_cols = kernel_cols + poly_cols;

    // Pre-compute polyharmonic coefficient for the pure Duchon case (no length_scale).
    // This avoids 2 gamma_lanczos calls per kernel evaluation (n × k total).
    let pure_poly_coeff = if length_scale.is_none() {
        Some(PolyharmonicBlockCoeff::new(
            (pure_duchon_block_order(p_order, s_order)) as f64,
            d,
        ))
    } else {
        None
    };

    let axis_scales = aniso_log_scales.map(aniso_axis_scales);
    let kernel_amp = duchon_kernel_amplification(
        centers,
        length_scale,
        p_order,
        duchon_power_to_usize(s_order),
        d,
        aniso_log_scales,
        coeffs.as_ref(),
        pure_poly_coeff.as_ref(),
    );
    // Certified radial value profile for the hybrid path (#979): one exact
    // hybrid-Duchon kernel value costs microseconds across its
    // partial-fraction blocks, and this n·k materialization loop runs on
    // every design rebuild of every κ-trial. For large sweeps, profile φ
    // once over the observed radius range (distance-only pre-pass) and
    // answer per-pair queries by Clenshaw; out-of-range radii and
    // uncertified builds fall back to the exact evaluator (the profile's
    // exact fallback IS `duchon_radial_jets`, whose value channel is the
    // same `duchon_matern_kernel_general_from_distance` evaluated below).
    let hybrid_kind = match (length_scale, coeffs.as_ref()) {
        (Some(ls), Some(c)) if pure_poly_coeff.is_none() => Some(RadialScalarKind::Duchon {
            length_scale: ls,
            p_order,
            s_order: duchon_power_to_usize(s_order),
            dim: d,
            coeffs: c.clone(),
        }),
        _ => None,
    };
    let value_profile = hybrid_kind.as_ref().and_then(|kind| {
        if n.saturating_mul(k) < RADIAL_PROFILE_MIN_PAIRS {
            return None;
        }
        let (r_lo, r_hi) = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut lo = f64::INFINITY;
                let mut hi = 0.0_f64;
                for j in 0..k {
                    let r = if let Some(scales) = axis_scales.as_deref() {
                        aniso_distance_rows_with_scales(data, i, centers, j, scales)
                    } else {
                        euclidean_distance_rows(data, i, centers, j)
                    };
                    if r > 0.0 {
                        lo = lo.min(r);
                        hi = hi.max(r);
                    }
                }
                (lo, hi)
            })
            .reduce(
                || (f64::INFINITY, 0.0_f64),
                |a, b| (a.0.min(b.0), a.1.max(b.1)),
            );
        if r_lo.is_finite() && r_hi > r_lo {
            radial_profile::RadialProfile::build(kind, r_lo, r_hi)
        } else {
            None
        }
    });
    let mut basis = Array2::<f64>::zeros((n, total_cols));
    // Process rows in chunks to amortize thread-local allocation across many rows.
    // Use larger chunks (1024) for better cache utilization at large scale.
    let chunk_size = 1024.min(n);
    let basis_result: Result<(), BasisError> = basis
        .axis_chunks_iter_mut(Axis(0), chunk_size)
        .into_par_iter()
        .enumerate()
        .try_for_each(|(ci, mut chunk)| {
            let mut kernel_row = vec![0.0; k];
            let chunk_start = ci * chunk_size;
            for local_i in 0..chunk.nrows() {
                let i = chunk_start + local_i;
                for j in 0..k {
                    let r = if let Some(scales) = axis_scales.as_deref() {
                        aniso_distance_rows_with_scales(data, i, centers, j, scales)
                    } else {
                        euclidean_distance_rows(data, i, centers, j)
                    };
                    let raw = if let Some(ref ppc) = pure_poly_coeff {
                        // Pure Duchon: use precomputed coefficient, skip gamma calls.
                        ppc.eval(r)
                    } else if let (Some(profile), Some(kind)) =
                        (value_profile.as_ref(), hybrid_kind.as_ref())
                    {
                        profile.eval_or_exact(kind, r)?.0
                    } else {
                        duchon_matern_kernel_general_from_distance(
                            r,
                            length_scale,
                            p_order,
                            duchon_power_to_usize(s_order),
                            d,
                            coeffs.as_ref(),
                        )?
                    };
                    kernel_row[j] = raw * kernel_amp;
                }
                // Write basis row = kernel_row^T × Z using scatter-accumulate
                // pattern: for each knot j with nonzero kernel, add its
                // contribution to all columns at once. This is more cache-
                // friendly than the column-by-column gather pattern since
                // Z rows are contiguous in memory.
                let mut row = chunk.row_mut(local_i);
                row.slice_mut(s![..kernel_cols]).fill(0.0);
                for j in 0..k {
                    let kv = kernel_row[j];
                    if kv != 0.0 {
                        let z_row = z.row(j);
                        for col in 0..kernel_cols {
                            row[col] += kv * z_row[col];
                        }
                    }
                }
            }
            Ok(())
        });
    basis_result?;
    if poly_cols > 0 {
        basis.slice_mut(s![.., kernel_cols..]).assign(&poly_block);
    }

    Ok(DuchonBasisDesign { basis })
}


fn build_cyclic_duchon_basis_1dwithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    start: f64,
    end: f64,
) -> Result<BasisBuildResult, BasisError> {
    if data.ncols() != 1 {
        crate::bail_invalid_basis!("cyclic Duchon smooths currently require exactly one covariate");
    }
    if end <= start {
        return Err(BasisError::InvalidRange(start, end));
    }
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    if centers.ncols() != 1 {
        crate::bail_dim_basis!(
            "cyclic Duchon centers must have one column, got {}",
            centers.ncols()
        );
    }
    let k = centers.nrows();
    let s_order_usize = spec.power_as_usize();
    if k <= s_order_usize.max(1) {
        crate::bail_invalid_basis!(
            "cyclic Duchon basis requires more centers ({k}) than power ({})",
            spec.power
        );
    }
    let period = end - start;
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Zero);
    // Hybrid kernel evaluates the truncated integer `s` (`power_as_usize`);
    // scale-free uses the literal fractional power. Gate on the realized value.
    let validation_power = if spec.length_scale.is_some() {
        s_order_usize as f64
    } else {
        spec.power
    };
    validate_duchon_kernel_orders(spec.length_scale, p_order, validation_power, 1)?;
    let coeffs = spec
        .length_scale
        .map(|ls| duchon_partial_fraction_coeffs(p_order, s_order_usize, 1.0 / ls.max(1e-300)));
    let pure_poly_coeff = if spec.length_scale.is_none() {
        Some(PolyharmonicBlockCoeff::new(
            pure_duchon_block_order(p_order, spec.power),
            1,
        ))
    } else {
        None
    };
    let kernel_amp = duchon_kernel_amplification(
        centers.view(),
        spec.length_scale,
        p_order,
        duchon_power_to_usize(spec.power),
        1,
        None,
        coeffs.as_ref(),
        pure_poly_coeff.as_ref(),
    );
    let mut basis = Array2::<f64>::zeros((data.nrows(), k + 1));
    for i in 0..data.nrows() {
        let x = wrap_to_period(data[[i, 0]], start, period);
        for j in 0..k {
            let c = wrap_to_period(centers[[j, 0]], start, period);
            let r = cyclic_distance_1d(x, c, period);
            let raw = if let Some(ref ppc) = pure_poly_coeff {
                ppc.eval(r)
            } else {
                duchon_matern_kernel_general_from_distance(
                    r,
                    spec.length_scale,
                    p_order,
                    s_order_usize,
                    1,
                    coeffs.as_ref(),
                )?
            };
            basis[[i, j]] = raw * kernel_amp;
        }
        basis[[i, k]] = 1.0;
    }

    let identifiability_transform = spatial_identifiability_transform_from_design(
        data,
        basis.view(),
        &spec.identifiability,
        "cyclic Duchon",
    )?;
    let (design_matrix, transform_for_penalty) = if let Some(z) = identifiability_transform.as_ref()
    {
        (fast_ab(&basis, z), Some(z))
    } else {
        (basis, None)
    };

    let mut s_kernel = Array2::<f64>::zeros((k + 1, k + 1));
    let s_cyclic = create_cyclic_difference_penalty_matrix(k, s_order_usize.max(1).min(k - 1))?;
    s_kernel.slice_mut(s![..k, ..k]).assign(&s_cyclic);
    let s_final = if let Some(z) = transform_for_penalty {
        fast_ab(&fast_atb(z, &s_kernel), z)
    } else {
        s_kernel
    };
    let candidates = vec![PenaltyCandidate {
        matrix: s_final,
        nullspace_dim_hint: 1,
        source: PenaltySource::Primary,
        normalization_scale: 1.0,
        kronecker_factors: None,
        op: None,
    }];
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    Ok(BasisBuildResult {
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design_matrix)),
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::Duchon {
            centers,
            length_scale: spec.length_scale,
            periodic: Some(vec![Some(period)]),
            power: spec.power,
            nullspace_order: DuchonNullspaceOrder::Zero,
            identifiability_transform,
            input_scales: None,
            aniso_log_scales: None,
            operator_collocation_points: None,
        },
        kronecker_factored: None,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
    })
}


/// Generic Duchon builder returning design + penalty list.
pub fn build_duchon_basis(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_basiswithworkspace(data, spec, &mut workspace)
}


pub fn create_duchon_basis_1d_derivative_dense(
    t: ArrayView1<'_, f64>,
    centers: ArrayView1<'_, f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    periodic: bool,
    period: Option<f64>,
    order: usize,
) -> Result<Array2<f64>, BasisError> {
    if order > 2 {
        crate::bail_invalid_basis!(
            "Duchon basis derivative supports orders 0, 1, and 2; got order={order}"
        );
    }
    if t.is_empty() || centers.is_empty() {
        crate::bail_invalid_basis!("Duchon basis derivative requires non-empty t and centers");
    }
    if t.iter().any(|v| !v.is_finite()) || centers.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("Duchon basis derivative requires finite t and center values");
    }
    if !periodic && period.is_some() {
        crate::bail_invalid_basis!(
            "Duchon basis derivative period is only valid when periodic=true"
        );
    }

    let data = t.to_owned().insert_axis(Axis(1));
    let center_matrix = centers.to_owned().insert_axis(Axis(1));
    let mut workspace = BasisWorkspace::default();
    // The user-requested Duchon order ``m`` is encoded in ``nullspace_order``;
    // the PERIODIC kernel is the Bernoulli Green's function of ``(d²/dx²)^m``
    // (PSD on the circle, gam#580) so it needs the original ``m`` even though
    // the periodic *constraint* nullspace is forced to constants only.
    let user_m = duchon_p_from_nullspace_order(nullspace_order);
    let effective_order = if periodic {
        DuchonNullspaceOrder::Zero
    } else {
        duchon_effective_nullspace_order(center_matrix.view(), nullspace_order)
    };
    let p_order = duchon_p_from_nullspace_order(effective_order);
    let s_order = duchon_power_to_usize(power);
    validate_duchon_kernel_orders(None, p_order, s_order as f64, 1)?;

    if periodic {
        // Periodic case: mirror the forward Bernoulli Green's-function design
        // (`build_periodic_duchon_basis_1d`) EXACTLY — same collapsed centers,
        // same domain-wrap period, same constant-only constraint nullspace —
        // so the analytic derivative is the true ∂/∂t of the forward design
        // (gam#580). Using the polyharmonic triangle-wave kernel here (the old
        // path) was inconsistent with the Bernoulli forward and silently wrong.
        let (collapsed_centers, left, resolved_period) =
            prepare_periodic_duchon_centers_1d_with_period(center_matrix, period)?;
        let z = kernel_constraint_nullspace(
            collapsed_centers.view(),
            effective_order,
            &mut workspace.cache,
        )?;
        let kernel_cols = z.ncols();
        let k_centers = collapsed_centers.nrows();
        let centers_col0: Vec<f64> = collapsed_centers.column(0).to_vec();
        let mut raw_kernel = Array2::<f64>::zeros((t.len(), k_centers));
        for i in 0..t.len() {
            let x = wrap_to_period(t[i], left, resolved_period);
            for j in 0..k_centers {
                // Signed offset reduced to [−period/2, period/2]; r = |offset|.
                let mut delta = (x - centers_col0[j]).rem_euclid(resolved_period);
                if delta > 0.5 * resolved_period {
                    delta -= resolved_period;
                }
                let r = delta.abs();
                let sign = if delta > 0.0 {
                    1.0
                } else if delta < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let (phi, phi_r, phi_rr) =
                    periodic_duchon_kernel_bernoulli_triplet(r, user_m, resolved_period)?;
                raw_kernel[[i, j]] = match order {
                    0 => phi,
                    1 => phi_r * sign,
                    2 => phi_rr,
                    other => {
                        crate::bail_invalid_basis!(
                            "Duchon basis derivative supports orders 0, 1, and 2; got order={other}"
                        );
                    }
                };
            }
        }
        // Forward design appends a single constant column; its t-derivative is
        // zero (order ≥ 1) or one (order 0). Match that layout exactly.
        let mut basis = Array2::<f64>::zeros((t.len(), kernel_cols + 1));
        let design_kernel = fast_ab(&raw_kernel, &z);
        basis
            .slice_mut(s![.., 0..kernel_cols])
            .assign(&design_kernel);
        if order == 0 {
            basis.column_mut(kernel_cols).fill(1.0);
        }
        return Ok(basis);
    }

    let z =
        kernel_constraint_nullspace(center_matrix.view(), effective_order, &mut workspace.cache)?;
    let kernel_cols = z.ncols();
    let poly_cols = polynomial_block_from_order(data.view(), effective_order).ncols();

    let pure_coeff =
        PolyharmonicBlockCoeff::new((pure_duchon_block_order(p_order, s_order as f64)) as f64, 1);
    let kernel_amp = duchon_kernel_amplification(
        center_matrix.view(),
        None,
        p_order,
        s_order,
        1,
        None,
        None,
        Some(&pure_coeff),
    );

    let mut raw_kernel = Array2::<f64>::zeros((t.len(), centers.len()));
    for i in 0..t.len() {
        let x = t[i];
        for j in 0..centers.len() {
            let delta = x - centers[j];
            let r = delta.abs();
            let sign = if delta > 0.0 {
                1.0
            } else if delta < 0.0 {
                -1.0
            } else {
                0.0
            };
            let (phi, phi_r, phi_rr) =
                duchon_kernel_radial_triplet(r, None, p_order, s_order as f64, 1, None)?;
            raw_kernel[[i, j]] = match order {
                0 => phi,
                1 => phi_r * sign,
                2 => phi_rr,
                other => {
                    crate::bail_invalid_basis!(
                        "Duchon basis derivative supports orders 0, 1, and 2; got order={other}"
                    );
                }
            } * kernel_amp;
        }
    }

    let mut basis = Array2::<f64>::zeros((t.len(), kernel_cols + poly_cols));
    let design_kernel = fast_ab(&raw_kernel, &z);
    basis
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&design_kernel);
    fill_duchon_1d_polynomial_derivative(&mut basis, kernel_cols, t, effective_order, order);
    Ok(basis)
}


/// Shared N-D Duchon radial-jet matrices: the per-`(row, center)` scalar
/// radial derivatives `φ'(r)`, `φ''(r)`, `φ'''(r)` of the Duchon kernel.
///
/// This is the single place that performs the expensive, error-prone work
/// behind every N-D Duchon derivative consumer: distance evaluation, the
/// effective-scale default, partial-fraction-coefficient derivation, and the
/// per-pair [`duchon_radial_jets`] call. The first/second/third radial
/// derivative helpers, and the analytic-penalty Cartesian-derivative tensors,
/// are all thin adapters over the matrices produced here.
///
/// Only the radial orders `1..=max_order` are materialized; higher matrices
/// are left empty. `max_order` must be in `1..=3` for the supported latent /
/// isometry paths.
struct DuchonRadialJetsNd {
    /// `(n_rows, n_centers)` matrix of `φ'(r_{nk})`; always populated.
    phi_r: Array2<f64>,
    /// `(n_rows, n_centers)` matrix of `φ''(r_{nk})`; populated iff `max_order ≥ 2`.
    phi_rr: Array2<f64>,
    /// `(n_rows, n_centers)` matrix of `φ'''(r_{nk})`; populated iff `max_order ≥ 3`.
    phi_rrr: Array2<f64>,
}


/// Effective length scale used by [`duchon_radial_jets`]'s near-origin guards
/// (`r_floor`, collision-Taylor radius) when the caller selects the scale-free
/// pure-Duchon spectrum via `length_scale = None`. This only sets the
/// numerical guards near `r = 0` and does not change the analytic kernel; we
/// pick the typical inter-center distance (or `1.0` as a last resort).
fn duchon_effective_length_scale(length_scale: Option<f64>, centers: ArrayView2<'_, f64>) -> f64 {
    length_scale.unwrap_or_else(|| {
        let n_centers = centers.nrows();
        let dim = centers.ncols();
        let mut acc = 0.0_f64;
        let mut cnt = 0usize;
        for i in 0..n_centers.min(8) {
            for j in (i + 1)..n_centers.min(8) {
                let mut r2 = 0.0_f64;
                for a in 0..dim {
                    let dv = centers[[i, a]] - centers[[j, a]];
                    r2 += dv * dv;
                }
                acc += r2.sqrt();
                cnt += 1;
            }
        }
        if cnt == 0 || acc <= 0.0 {
            1.0
        } else {
            acc / cnt as f64
        }
    })
}


/// Evaluate the shared N-D Duchon radial jets up to `max_order` (`1..=3`).
///
/// `caller` is used only to give callers a precise validation message.
fn duchon_radial_jets_nd(
    max_order: usize,
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    nullspace_order: DuchonNullspaceOrder,
    power: usize,
    caller: &str,
) -> Result<DuchonRadialJetsNd, BasisError> {
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!("{caller}: centers must have at least one column");
    }
    if t.ncols() != dim {
        crate::bail_invalid_basis!(
            "{caller}: t has {} cols but centers have {}",
            t.ncols(),
            dim
        );
    }
    assert!(
        (1..=3).contains(&max_order),
        "duchon_radial_jets_nd supports radial orders 1..=3; got {max_order}"
    );
    let effective_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_order);
    // Resolve the same hybrid `(p, s, κ)` triple the forward `build_duchon_basis`
    // uses, so the radial jets differentiate the *exact* forward Green's
    // function `φ_{p,s,κ}` rather than a hard-coded `s = 0` surrogate
    // (issue #440). `power` is the forward spectral order `s = spec.power`.
    //
    //   * Hybrid (`length_scale = Some`): spectrum `‖w‖^{2p}(κ²+‖w‖²)^s`, built
    //     from the integer partial-fraction blocks at the real κ = 1/length_scale.
    //   * Pure scale-free (`length_scale = None`): the forward kernel collapses
    //     to the polyharmonic of total order `p + s` (`pure_duchon_block_order`,
    //     `duchon_matern_kernel_general_from_distance`'s `None` branch). Folding
    //     `s` into the polyharmonic order (`s_jets = 0`, `a_{p+s} = 1`) makes the
    //     operator core κ-independent and reproduces that exact kernel and all
    //     of its radial derivatives.
    let (jet_p_order, s_order) = match length_scale {
        Some(_) => (p_order, power),
        None => (p_order + power, 0usize),
    };
    let kappa = length_scale.map(|l| 1.0 / l.max(1e-300)).unwrap_or(0.0);
    let coeffs = duchon_partial_fraction_coeffs(jet_p_order, s_order, kappa);
    let effective_length_scale = duchon_effective_length_scale(length_scale, centers);

    let mut phi_r = Array2::<f64>::zeros((n_rows, n_centers));
    let mut phi_rr = if max_order >= 2 {
        Array2::<f64>::zeros((n_rows, n_centers))
    } else {
        Array2::<f64>::zeros((0, 0))
    };
    let mut phi_rrr = if max_order >= 3 {
        Array2::<f64>::zeros((n_rows, n_centers))
    } else {
        Array2::<f64>::zeros((0, 0))
    };
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let dv = t[[n, a]] - centers[[k, a]];
                r2 += dv * dv;
            }
            let r = r2.sqrt();
            let jets = duchon_radial_jets(
                r,
                effective_length_scale,
                jet_p_order,
                s_order,
                dim,
                &coeffs,
            )?;
            phi_r[[n, k]] = jets.phi_r;
            if max_order >= 2 {
                phi_rr[[n, k]] = jets.phi_rr;
            }
            if max_order >= 3 {
                phi_rrr[[n, k]] = jets.phi_rrr;
            }
        }
    }
    Ok(DuchonRadialJetsNd {
        phi_r,
        phi_rr,
        phi_rrr,
    })
}


/// Map shared N-D Duchon radial jets into the Cartesian input-location
/// derivative tensor of `order` (2 or 3), contracted against per-center
/// decoder coefficients.
///
/// `coeffs` is the `(n_centers, p_out)` matrix of radial-basis coefficients
/// `c_{k,i}`. The result is a flat `(n_rows, p_out · dⁿ)` matrix whose entry
/// for row `n`, output `i`, and Cartesian multi-index `m` lives at column
/// `i · dⁿ + m`, where `m` enumerates the axes in row-major order
/// (`(a·d + c)` for `order = 2`, `((a·d + c)·d + e)` for `order = 3`).
///
/// The radial→Cartesian maps are:
///
/// ```text
/// order 2: ∂²Φ/∂t_a∂t_c = q δ_ac + (φ'' − q) u_a u_c,   q = φ'/r
/// order 3: ∂³Φ/∂t_a∂t_c∂t_e = a u_a u_c u_e
///                            + b (δ_ac u_e + δ_ae u_c + δ_ce u_a)
///          b = (φ'' − q)/r,   a = φ''' − 3b
/// ```
///
/// with `u = (t_n − c_k)/r`. At `r = 0` the order-2 collision limit is the
/// isotropic `φ''(0) δ_ac`; the order-3 tensor vanishes there.
pub(crate) fn radial_basis_cartesian_derivative(
    order: usize,
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    coeffs: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    nullspace_order: DuchonNullspaceOrder,
    power: usize,
) -> Result<Array2<f64>, BasisError> {
    assert!(
        order == 2 || order == 3,
        "radial_basis_cartesian_derivative supports Cartesian orders 2 and 3; got {order}"
    );
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let d = centers.ncols();
    let p_out = coeffs.ncols();
    assert_eq!(
        coeffs.nrows(),
        n_centers,
        "radial_basis_cartesian_derivative: coeffs has {} rows but centers have {n_centers}",
        coeffs.nrows()
    );
    let jets = duchon_radial_jets_nd(
        order,
        t,
        centers,
        length_scale,
        nullspace_order,
        power,
        "radial_basis_cartesian_derivative",
    )?;
    let d_pow = d.pow(order as u32);
    let mut out = Array2::<f64>::zeros((n_rows, p_out * d_pow));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..d {
                let delta = t[[n, a]] - centers[[k, a]];
                r2 += delta * delta;
            }
            let r = r2.sqrt();
            match order {
                2 => {
                    for a in 0..d {
                        for c in 0..d {
                            let basis_hess = if r == 0.0 {
                                if a == c { jets.phi_rr[[n, k]] } else { 0.0 }
                            } else {
                                let inv_r = 1.0 / r;
                                let u_a = (t[[n, a]] - centers[[k, a]]) * inv_r;
                                let u_c = (t[[n, c]] - centers[[k, c]]) * inv_r;
                                let q = jets.phi_r[[n, k]] * inv_r;
                                let eye = if a == c { 1.0 } else { 0.0 };
                                q * eye + (jets.phi_rr[[n, k]] - q) * u_a * u_c
                            };
                            if basis_hess == 0.0 {
                                continue;
                            }
                            let m = a * d + c;
                            for i in 0..p_out {
                                out[[n, i * d_pow + m]] += coeffs[[k, i]] * basis_hess;
                            }
                        }
                    }
                }
                _ => {
                    if r == 0.0 {
                        continue;
                    }
                    let inv_r = 1.0 / r;
                    let q = jets.phi_r[[n, k]] * inv_r;
                    let b_coef = (jets.phi_rr[[n, k]] - q) * inv_r;
                    let a_coef = jets.phi_rrr[[n, k]] - 3.0 * b_coef;
                    for a in 0..d {
                        let u_a = (t[[n, a]] - centers[[k, a]]) * inv_r;
                        for c in 0..d {
                            let u_c = (t[[n, c]] - centers[[k, c]]) * inv_r;
                            for e in 0..d {
                                let u_e = (t[[n, e]] - centers[[k, e]]) * inv_r;
                                let eye_ac = if a == c { 1.0 } else { 0.0 };
                                let eye_ae = if a == e { 1.0 } else { 0.0 };
                                let eye_ce = if c == e { 1.0 } else { 0.0 };
                                let basis_third = a_coef * u_a * u_c * u_e
                                    + b_coef * (eye_ac * u_e + eye_ae * u_c + eye_ce * u_a);
                                if basis_third == 0.0 {
                                    continue;
                                }
                                let m = (a * d + c) * d + e;
                                for i in 0..p_out {
                                    out[[n, i * d_pow + m]] += coeffs[[k, i]] * basis_third;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}


/// N-D Duchon radial first-derivative `φ'(r)` evaluated for every
/// `(row, center)` pair.
///
/// Returns an `(n_rows, n_centers)` matrix whose `(n, k)` entry is the
/// scalar radial derivative `φ'(r_{nk})` of the Duchon kernel,
/// where `r_{nk} = ‖t_n − c_k‖_2`.
///
/// This is the load-bearing primitive for differentiating a Duchon design
/// against its *first* kernel argument (i.e. per-row latent coordinates
/// `t_n`); the full per-row gradient is reconstructed at the call site as
/// `∂Φ_{n,k}/∂t_n = φ'(r_{n,k}) · (t_n − c_k) / r_{n,k}` (see
/// [`crate::terms::latent_coord::LatentCoordValues::design_gradient_wrt_t`]).
///
/// `length_scale = None` selects the scale-free pure-Duchon spectrum
/// (matches `gam_pyffi::position_basis_derivative` for the 1-D case).
///
/// Thin adapter over [`duchon_radial_jets_nd`].
pub fn duchon_radial_first_derivative_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    nullspace_order: DuchonNullspaceOrder,
    power: usize,
) -> Result<Array2<f64>, BasisError> {
    Ok(duchon_radial_jets_nd(
        1,
        t,
        centers,
        length_scale,
        nullspace_order,
        power,
        "duchon_radial_first_derivative_nd",
    )?
    .phi_r)
}


/// N-D Duchon radial second derivative `φ''(r)` evaluated for every
/// `(row, center)` pair.
///
/// Returns an `(n_rows, n_centers)` matrix whose `(n, k)` entry is the scalar
/// radial second derivative `φ''(r_{nk})` of the Duchon kernel, where
/// `r_{nk} = ‖t_n − c_k‖_2`.
///
/// This is the companion primitive to
/// [`duchon_radial_first_derivative_nd`]. Together the two scalars reconstruct
/// the full input-location Hessian:
///
/// ```text
/// ∂²φ/∂t_a∂t_b = (φ'(r)/r) δ_ab
///              + (φ''(r) − φ'(r)/r) (t_a − c_a)(t_b − c_b) / r².
/// ```
///
/// At `r = 0`, consumers should use the isotropic collision limit
/// `φ''(0) δ_ab`; `duchon_radial_jets` supplies that finite scalar whenever
/// the selected Duchon order is smooth enough for the supported latent path.
///
/// Thin adapter over [`duchon_radial_jets_nd`].
pub fn duchon_radial_second_derivative_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    nullspace_order: DuchonNullspaceOrder,
    power: usize,
) -> Result<Array2<f64>, BasisError> {
    Ok(duchon_radial_jets_nd(
        2,
        t,
        centers,
        length_scale,
        nullspace_order,
        power,
        "duchon_radial_second_derivative_nd",
    )?
    .phi_rr)
}


/// N-D Duchon radial third derivative `φ'''(r)` evaluated for every
/// `(row, center)` pair.
///
/// Returns an `(n_rows, n_centers)` matrix whose `(n, k)` entry is the scalar
/// third radial derivative `φ'''(r_{nk})` from the same
/// [`duchon_radial_jets`] path used by the first/second derivative helpers.
///
/// Thin adapter over [`duchon_radial_jets_nd`].
pub fn duchon_radial_third_derivative_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    nullspace_order: DuchonNullspaceOrder,
    power: usize,
) -> Result<Array2<f64>, BasisError> {
    Ok(duchon_radial_jets_nd(
        3,
        t,
        centers,
        length_scale,
        nullspace_order,
        power,
        "duchon_radial_third_derivative_nd",
    )?
    .phi_rrr)
}


fn fill_duchon_1d_polynomial_derivative(
    basis: &mut Array2<f64>,
    start_col: usize,
    t: ArrayView1<'_, f64>,
    nullspace_order: DuchonNullspaceOrder,
    derivative_order: usize,
) {
    let exponents: Vec<usize> = match nullspace_order {
        DuchonNullspaceOrder::Zero => vec![0],
        DuchonNullspaceOrder::Linear => vec![0, 1],
        DuchonNullspaceOrder::Degree(degree) => monomial_exponents(1, degree)
            .into_iter()
            .map(|exponent| exponent[0])
            .collect(),
    };
    for (offset, exponent) in exponents.into_iter().enumerate() {
        if exponent < derivative_order {
            continue;
        }
        let coefficient =
            (0..derivative_order).fold(1.0, |acc, step| acc * (exponent - step) as f64);
        let remaining = exponent - derivative_order;
        for row in 0..t.len() {
            basis[[row, start_col + offset]] = if remaining == 0 {
                coefficient
            } else {
                coefficient * t[row].powi(remaining as i32)
            };
        }
    }
}


/// N-D Duchon polynomial-nullspace first derivative `∂P/∂t` per row.
///
/// Generalises [`fill_duchon_1d_polynomial_derivative`] to arbitrary spatial
/// dimension `d` and arbitrary nullspace degree. For the monomial
/// `m_α(t) = ∏_a t_a^{α_a}`, the partial derivative w.r.t. `t_axis` is
///
/// ```text
///     ∂ m_α / ∂ t_axis = α_axis · t_axis^{α_axis − 1} · ∏_{a ≠ axis} t_a^{α_a}
/// ```
///
/// (and is zero if `α_axis == 0`).
///
/// Returned tensor shape: `(n_rows, n_poly_cols, d)`, ordered with the same
/// `monomial_exponents(d, max_total_degree)` enumeration that
/// `monomial_basis_block` uses to build the polynomial-tail columns of the
/// Duchon design, so the column index `k` aligns directly with the design.
pub fn duchon_polynomial_first_derivative_nd(
    t: ArrayView2<'_, f64>,
    nullspace_order: DuchonNullspaceOrder,
) -> Array3<f64> {
    let n_rows = t.nrows();
    let dim = t.ncols();
    let max_degree = match nullspace_order {
        DuchonNullspaceOrder::Zero => 0usize,
        DuchonNullspaceOrder::Linear => 1usize,
        DuchonNullspaceOrder::Degree(k) => k,
    };
    let exponents = monomial_exponents(dim, max_degree);
    let n_poly = exponents.len();
    let mut out = Array3::<f64>::zeros((n_rows, n_poly, dim));
    if dim == 0 || n_poly == 0 {
        return out;
    }
    for (col, alpha) in exponents.iter().enumerate() {
        for axis in 0..dim {
            let a_axis = alpha[axis];
            if a_axis == 0 {
                continue;
            }
            for row in 0..n_rows {
                let mut value = a_axis as f64;
                for a in 0..dim {
                    let exp_a = if a == axis { a_axis - 1 } else { alpha[a] };
                    if exp_a != 0 {
                        value *= t[[row, a]].powi(exp_a as i32);
                    }
                }
                out[[row, col, axis]] = value;
            }
        }
    }
    out
}


/// Per-`(row, center)` input-location jet of a radial kernel from its scalar
/// first derivative `φ'(r)`.
///
/// `phi_r[n, k] = φ'(r_{nk})`. The full gradient w.r.t. the latent input is
///
/// ```text
/// ∂Φ_{n,k}/∂t_{n,a} = φ'(r_{nk}) · (t_{n,a} − c_{k,a}) / r_{nk}.
/// ```
///
/// At a collision (`r ≤ 1e-12`) the gradient is the zero vector — the radial
/// kernel has a stationary point at the center, so every axis derivative
/// vanishes there. Output shape `(n_rows, n_centers, dim)`.
pub fn radial_input_location_jet_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    phi_r: ArrayView2<'_, f64>,
) -> Result<Array3<f64>, BasisError> {
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let dim = t.ncols();
    if phi_r.dim() != (n_rows, n_centers) {
        crate::bail_dim_basis!(
            "radial_input_location_jet_nd: phi_r shape {:?} != ({n_rows}, {n_centers})",
            phi_r.dim()
        );
    }
    if centers.ncols() != dim {
        crate::bail_dim_basis!(
            "radial_input_location_jet_nd: t has {dim} cols but centers have {}",
            centers.ncols()
        );
    }
    let mut out = Array3::<f64>::zeros((n_rows, n_centers, dim));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let delta = t[[n, a]] - centers[[k, a]];
                r2 += delta * delta;
            }
            let r = r2.sqrt();
            if r <= 1.0e-12 {
                continue;
            }
            let scale = phi_r[[n, k]] / r;
            for a in 0..dim {
                out[[n, k, a]] = scale * (t[[n, a]] - centers[[k, a]]);
            }
        }
    }
    Ok(out)
}


/// Forward design and input-location first jet of the scale-free Duchon atom
/// used by the SAE-manifold path, recomputed self-consistently at arbitrary
/// latent coordinates `t`.
///
/// The column layout matches [`build_duchon_basis`] under the SAE atom's spec
/// (`length_scale = None`, `power = 0`, no identifiability transform): the
/// kernel block `Φ_radial(t) · Z` (where `Z = null(P_centersᵀ)` is the
/// polynomial-constraint null space) followed by the polynomial block
/// `P(t)`. Both blocks of `Φ` **and** the matching blocks of the jet carry
/// the identical scalar kernel amplification `α` that
/// [`build_duchon_basis`] applies, so the returned `(Φ, ∂Φ/∂t)` pair is a
/// true jet — i.e. the kernel block of the jet is exactly the `t`-derivative
/// of the kernel block of `Φ`, with no stray `α` mismatch (the precise
/// failure mode of issue #247: a forward design and derivative jet built from
/// inconsistent scalings/column counts).
///
/// `t` is `(n_rows, dim)`, `centers` is `(n_centers, dim)`. Returns
/// `(Φ, jet)` with `Φ` shape `(n_rows, n_kernel + n_poly)` and `jet` shape
/// `(n_rows, n_kernel + n_poly, dim)`.
pub fn duchon_sae_atom_basis_with_jet(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    nullspace_order: DuchonNullspaceOrder,
) -> Result<(Array2<f64>, Array3<f64>), BasisError> {
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "duchon_sae_atom_basis_with_jet: centers must have at least one column"
        );
    }
    if t.ncols() != dim {
        crate::bail_dim_basis!(
            "duchon_sae_atom_basis_with_jet: t has {} cols but centers have {dim}",
            t.ncols()
        );
    }
    let effective_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_order);
    let s_order: f64 = 0.0;

    // Polynomial-constraint null space `Z` (same construction as the design).
    let poly_block_centers = polynomial_block_from_order(centers, effective_order);
    let z = kernel_constraint_nullspace_from_matrix(poly_block_centers.view())?;
    let n_kernel = z.ncols();

    // Scalar kernel amplification `α`, identical to the value the design path
    // applies (pure scale-free polyharmonic block).
    let pure_poly_coeff =
        PolyharmonicBlockCoeff::new(pure_duchon_block_order(p_order, s_order), dim);
    let kernel_amp = duchon_kernel_amplification(
        centers,
        None,
        p_order,
        duchon_power_to_usize(s_order),
        dim,
        None,
        None,
        Some(&pure_poly_coeff),
    );

    // Forward radial kernel block `(Φ_radial · α) · Z`.
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let mut radial_value = Array2::<f64>::zeros((n_rows, n_centers));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let r = euclidean_distance_rows(t, n, centers, k);
            radial_value[[n, k]] = pure_poly_coeff.eval(r) * kernel_amp;
        }
    }
    let kernel_design = fast_ab(&radial_value, &z);

    // Polynomial forward block `P(t)`.
    let poly_design = polynomial_block_from_order(t, effective_order);
    let n_poly = poly_design.ncols();

    let mut phi = Array2::<f64>::zeros((n_rows, n_kernel + n_poly));
    phi.slice_mut(s![.., ..n_kernel]).assign(&kernel_design);
    if n_poly > 0 {
        phi.slice_mut(s![.., n_kernel..]).assign(&poly_design);
    }

    // Input-location first jet, scaled by the *same* `α` on the kernel block.
    let radial_first = duchon_radial_first_derivative_nd(t, centers, None, effective_order, 0)?;
    let radial_jet = radial_input_location_jet_nd(t, centers, radial_first.view())?;
    let poly_jet = duchon_polynomial_first_derivative_nd(t, effective_order);
    if poly_jet.shape()[1] != n_poly {
        crate::bail_dim_basis!(
            "duchon_sae_atom_basis_with_jet: polynomial jet has {} columns but design has {n_poly}",
            poly_jet.shape()[1]
        );
    }
    let mut jet = Array3::<f64>::zeros((n_rows, n_kernel + n_poly, dim));
    for axis in 0..dim {
        let projected = radial_jet.index_axis(Axis(2), axis).dot(&z);
        let mut block = jet.slice_mut(s![.., ..n_kernel, axis]);
        block.assign(&projected);
        block *= kernel_amp;
    }
    jet.slice_mut(s![.., n_kernel.., ..]).assign(&poly_jet);

    Ok((phi, jet))
}


/// Second input-location jet of the scale-free Duchon SAE atom (the analytic
/// Hessian `∂²Φ / ∂t_a ∂t_c`), consistent with
/// [`duchon_sae_atom_basis_with_jet`] column-for-column and `α`-for-`α`.
///
/// The kernel block uses the standard radial Hessian decomposition
///
/// ```text
/// ∂²φ/∂t_a∂t_c = (φ'(r)/r) δ_ac + (φ''(r) − φ'(r)/r) (t−c)_a (t−c)_c / r²,
/// ```
///
/// projected through `Z` and scaled by `α`; the polynomial block carries the
/// monomial Hessian. Output shape `(n_rows, n_kernel + n_poly, dim, dim)`.
pub fn duchon_sae_atom_second_jet(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    nullspace_order: DuchonNullspaceOrder,
) -> Result<Array4<f64>, BasisError> {
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "duchon_sae_atom_second_jet: centers must have at least one column"
        );
    }
    if t.ncols() != dim {
        crate::bail_dim_basis!(
            "duchon_sae_atom_second_jet: t has {} cols but centers have {dim}",
            t.ncols()
        );
    }
    let effective_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_order);
    let s_order: f64 = 0.0;

    let poly_block_centers = polynomial_block_from_order(centers, effective_order);
    let z = kernel_constraint_nullspace_from_matrix(poly_block_centers.view())?;
    let n_kernel = z.ncols();

    let pure_poly_coeff =
        PolyharmonicBlockCoeff::new(pure_duchon_block_order(p_order, s_order), dim);
    let kernel_amp = duchon_kernel_amplification(
        centers,
        None,
        p_order,
        duchon_power_to_usize(s_order),
        dim,
        None,
        None,
        Some(&pure_poly_coeff),
    );

    let n_rows = t.nrows();

    let poly_block_t_cols = polynomial_block_from_order(t, effective_order).ncols();

    // Kernel-block Cartesian Hessian `∂²Φ/∂t_a∂t_c`, projected through `Z` and
    // scaled by `α`, via the shared radial→Cartesian engine. Folding `α` into
    // the per-center coefficient matrix `Z` (so `coeffs = α·Z`, shape
    // `(n_centers, n_kernel)`) makes the shared helper emit the already-
    // amplified, already-projected kernel Hessian directly: its flat
    // `(n_rows, n_kernel·d²)` output places output `i`, multi-index `(a,c)` at
    // column `i·d² + (a·d + c)`, including the `φ''(0) δ_ac` collision limit.
    let coeffs = &z * kernel_amp;
    let flat =
        radial_basis_cartesian_derivative(2, t, centers, coeffs.view(), None, effective_order, 0)?;

    let mut out = Array4::<f64>::zeros((n_rows, n_kernel + poly_block_t_cols, dim, dim));
    for n in 0..n_rows {
        for i in 0..n_kernel {
            for a in 0..dim {
                for c in 0..dim {
                    out[[n, i, a, c]] = flat[[n, i * dim * dim + a * dim + c]];
                }
            }
        }
    }

    // Polynomial Hessian block.
    let exponents = monomial_exponents(
        dim,
        match effective_order {
            DuchonNullspaceOrder::Zero => 0,
            DuchonNullspaceOrder::Linear => 1,
            DuchonNullspaceOrder::Degree(k) => k,
        },
    );
    if exponents.len() != poly_block_t_cols {
        crate::bail_dim_basis!(
            "duchon_sae_atom_second_jet: monomial count {} != polynomial block columns {poly_block_t_cols}",
            exponents.len()
        );
    }
    for (col, alpha) in exponents.iter().enumerate() {
        for a in 0..dim {
            for c in 0..dim {
                // ∂²(Π t_i^{α_i}) / ∂t_a ∂t_c.
                for row in 0..n_rows {
                    let coeff_a = alpha[a];
                    if coeff_a == 0 {
                        continue;
                    }
                    let coeff_c = if a == c {
                        alpha[c].saturating_sub(1)
                    } else {
                        alpha[c]
                    };
                    if a != c && coeff_c == 0 {
                        continue;
                    }
                    let lead = (coeff_a as f64) * (coeff_c as f64);
                    if lead == 0.0 {
                        continue;
                    }
                    let mut value = lead;
                    for axis in 0..dim {
                        let mut exp = alpha[axis];
                        if axis == a {
                            exp = exp.saturating_sub(1);
                        }
                        if axis == c {
                            exp = exp.saturating_sub(1);
                        }
                        if exp != 0 {
                            value *= t[[row, axis]].powi(exp as i32);
                        }
                    }
                    out[[row, n_kernel + col, a, c]] = value;
                }
            }
        }
    }

    Ok(out)
}


/// Third input-location jet of the scale-free Duchon SAE atom (the analytic
/// `∂³Φ / ∂t_a ∂t_c ∂t_e`), consistent with
/// [`duchon_sae_atom_basis_with_jet`] and [`duchon_sae_atom_second_jet`]
/// column-for-column and `α`-for-`α`.
///
/// The kernel block uses the standard radial third-derivative decomposition
///
/// ```text
/// ∂³φ/∂t_a∂t_c∂t_e = a_coef·u_a u_c u_e
///                  + b_coef·(δ_ac u_e + δ_ae u_c + δ_ce u_a),
/// a_coef = φ'''(r) − 3·b_coef,   b_coef = (φ''(r) − φ'(r)/r)/r,   u = (t−c)/r,
/// ```
///
/// projected through `Z` and scaled by `α` (emitted directly by the shared
/// [`radial_basis_cartesian_derivative`] engine at order 3); the polynomial
/// block carries the monomial third derivative. At a coincident point `r = 0`
/// the kernel third jet vanishes (odd-order radial derivative of an even kernel
/// in the collision limit), which the engine encodes by skipping `r == 0`.
/// Output shape `(n_rows, n_kernel + n_poly, dim, dim, dim)`. Symmetric in its
/// three trailing axes by construction.
pub fn duchon_sae_atom_third_jet(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    nullspace_order: DuchonNullspaceOrder,
) -> Result<Array5<f64>, BasisError> {
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "duchon_sae_atom_third_jet: centers must have at least one column"
        );
    }
    if t.ncols() != dim {
        crate::bail_dim_basis!(
            "duchon_sae_atom_third_jet: t has {} cols but centers have {dim}",
            t.ncols()
        );
    }
    let effective_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_order);
    let s_order: f64 = 0.0;

    let poly_block_centers = polynomial_block_from_order(centers, effective_order);
    let z = kernel_constraint_nullspace_from_matrix(poly_block_centers.view())?;
    let n_kernel = z.ncols();

    let pure_poly_coeff =
        PolyharmonicBlockCoeff::new(pure_duchon_block_order(p_order, s_order), dim);
    let kernel_amp = duchon_kernel_amplification(
        centers,
        None,
        p_order,
        duchon_power_to_usize(s_order),
        dim,
        None,
        None,
        Some(&pure_poly_coeff),
    );

    let n_rows = t.nrows();
    let poly_block_t_cols = polynomial_block_from_order(t, effective_order).ncols();

    // Kernel-block Cartesian third jet `∂³Φ/∂t_a∂t_c∂t_e`, projected through `Z`
    // and scaled by `α`, via the shared radial→Cartesian engine. Folding `α`
    // into the per-center coefficient matrix (`coeffs = α·Z`, shape
    // `(n_centers, n_kernel)`) makes the helper emit the already-amplified,
    // already-projected kernel third jet directly: its flat
    // `(n_rows, n_kernel·d³)` output places output `i`, multi-index `(a,c,e)` at
    // column `i·d³ + ((a·d + c)·d + e)`.
    let coeffs = &z * kernel_amp;
    let flat =
        radial_basis_cartesian_derivative(3, t, centers, coeffs.view(), None, effective_order, 0)?;

    let d_pow = dim * dim * dim;
    let mut out = Array5::<f64>::zeros((n_rows, n_kernel + poly_block_t_cols, dim, dim, dim));
    for n in 0..n_rows {
        for i in 0..n_kernel {
            for a in 0..dim {
                for c in 0..dim {
                    for e in 0..dim {
                        out[[n, i, a, c, e]] = flat[[n, i * d_pow + ((a * dim) + c) * dim + e]];
                    }
                }
            }
        }
    }

    // Polynomial third-derivative block: `∂³(Π t_i^{α_i}) / ∂t_a ∂t_c ∂t_e`.
    // Differentiating axis `j` a total of `k_j` times (its multiplicity in
    // `{a, c, e}`) contracts that factor to `falling(α_j, k_j)·t_j^{α_j − k_j}`,
    // with `falling(α, k) = α(α−1)…(α−k+1)`; the term vanishes when `α_j < k_j`.
    let exponents = monomial_exponents(
        dim,
        match effective_order {
            DuchonNullspaceOrder::Zero => 0,
            DuchonNullspaceOrder::Linear => 1,
            DuchonNullspaceOrder::Degree(k) => k,
        },
    );
    if exponents.len() != poly_block_t_cols {
        crate::bail_dim_basis!(
            "duchon_sae_atom_third_jet: monomial count {} != polynomial block columns {poly_block_t_cols}",
            exponents.len()
        );
    }
    let falling = |alpha: usize, k: usize| -> f64 {
        let mut acc = 1.0_f64;
        for j in 0..k {
            acc *= (alpha as f64) - (j as f64);
        }
        acc
    };
    for (col, alpha) in exponents.iter().enumerate() {
        for a in 0..dim {
            if alpha[a] == 0 {
                continue;
            }
            for c in 0..dim {
                for e in 0..dim {
                    // Per-axis differentiation order in this (a, c, e) cell.
                    let mut order = vec![0usize; dim];
                    order[a] += 1;
                    order[c] += 1;
                    order[e] += 1;
                    if (0..dim).any(|axis| order[axis] > alpha[axis]) {
                        continue;
                    }
                    let mut lead = 1.0_f64;
                    for axis in 0..dim {
                        lead *= falling(alpha[axis], order[axis]);
                    }
                    if lead == 0.0 {
                        continue;
                    }
                    for row in 0..n_rows {
                        let mut value = lead;
                        for axis in 0..dim {
                            let exp = alpha[axis] - order[axis];
                            if exp != 0 {
                                value *= t[[row, axis]].powi(exp as i32);
                            }
                        }
                        out[[row, n_kernel + col, a, c, e]] = value;
                    }
                }
            }
        }
    }

    Ok(out)
}


/// Forward design, input-location first jet, and input-location second jet
/// (Hessian) of the **general** Duchon basis — the same matrix
/// [`build_duchon_basis`] / [`build_duchon_basis_mixed_periodicity_auto`]
/// produce for the resolved spec, differentiated **exactly**.
///
/// This is the analytic-derivative companion to the basis-only Python FFI
/// `duchon_basis`. The forward Duchon design is **not** the raw centerwise
/// radial kernel `K(x, C)`; it is
///
/// ```text
/// X(x) = [ α · K(x, C) · Z ,  P(x) ],
/// ```
///
/// where `Z = null(P(C)ᵀ)` is the polynomial-constraint null space, `P(x)`
/// is the monomial nullspace block, and `α` is the kernel amplification the
/// design path applies. The jets returned here are the exact `x`-derivatives
/// of that built matrix, column-for-column:
///
/// ```text
/// J(x)   = [ α · ∂ₓK(x,C) · Z ,  ∂ₓP(x) ],
/// H(x)   = [ α · ∂²ₓK(x,C) · Z , ∂²ₓP(x) ].
/// ```
///
/// Both the non-periodic radial path (pure polyharmonic and hybrid Matérn
/// length-scale, `s_order = power`) and the mixed-periodicity chord-embedding
/// path are handled, matching whichever forward builder the spec selects.
///
/// `t` is `(n_rows, dim)`, `centers` is `(n_centers, dim)`. The returned
/// triple is `(Φ, J, H)` with `Φ` shape `(n_rows, n_kernel + n_poly)`,
/// `J` shape `(n_rows, n_kernel + n_poly, dim)`, `H` shape
/// `(n_rows, n_kernel + n_poly, dim, dim)`.
pub fn build_duchon_basis_design_and_jets(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    periodic_per_axis: &[bool],
    periods: &[f64],
) -> Result<(Array2<f64>, Array3<f64>, Array4<f64>), BasisError> {
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "build_duchon_basis_design_and_jets: centers must have at least one column"
        );
    }
    if t.ncols() != dim {
        crate::bail_dim_basis!(
            "build_duchon_basis_design_and_jets: t has {} cols but centers have {dim}",
            t.ncols()
        );
    }
    if periodic_per_axis.len() != dim {
        crate::bail_invalid_basis!(
            "build_duchon_basis_design_and_jets: periodic_per_axis must have length {dim}, got {}",
            periodic_per_axis.len()
        );
    }
    if periods.len() != dim {
        crate::bail_invalid_basis!(
            "build_duchon_basis_design_and_jets: periods must have length {dim}, got {}",
            periods.len()
        );
    }
    let any_periodic = periodic_per_axis.iter().any(|&p| p);

    let n_rows = t.nrows();
    let n_centers = centers.nrows();

    // ----------------------------------------------------------------- spec
    // The mixed-periodicity forward forces the constraint nullspace to
    // {constants} (the only polynomial periodic on every periodic axis),
    // gates the spectrum to the pure polyharmonic case (length_scale = None,
    // power = 0), and applies NO kernel amplification. The non-periodic
    // forward keeps the (auto-degraded) requested nullspace, the hybrid
    // length-scale / power, and the amplification α. Mirror both exactly.
    if any_periodic {
        if length_scale.is_some() {
            crate::bail_invalid_basis!(
                "mixed-periodicity Duchon basis currently only supports the pure polyharmonic spectrum (length_scale=None)"
            );
        }
        if power != 0.0 {
            crate::bail_invalid_basis!(
                "mixed-periodicity Duchon basis currently requires power = 0 (pure polyharmonic); got power={power}"
            );
        }
        for (j, (&per, &period)) in periodic_per_axis.iter().zip(periods.iter()).enumerate() {
            if per && !(period.is_finite() && period > 0.0) {
                crate::bail_invalid_basis!(
                    "axis {j} is periodic but period={period} is not finite & positive"
                );
            }
        }
    }

    let effective_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let kernel_nullspace_order = if any_periodic {
        DuchonNullspaceOrder::Zero
    } else {
        effective_order
    };
    let p_order = duchon_p_from_nullspace_order(kernel_nullspace_order);
    // For the periodic chord path the kernel smoothness order tracks the
    // user's requested m (carried in `nullspace_order`, not the forced-to-
    // constant kernel-nullspace order), matching the mixed-periodicity forward.
    let kernel_m = if any_periodic {
        duchon_p_from_nullspace_order(nullspace_order)
    } else {
        p_order
    };
    let s_order_int = duchon_power_to_usize(power);
    let s_order_f = power;

    // Polynomial-constraint null space `Z` (same construction as every design
    // path: null of the polynomial side-condition block at the centers).
    let poly_block_centers = polynomial_block_from_order(centers, kernel_nullspace_order);
    let z = kernel_constraint_nullspace_from_matrix(poly_block_centers.view())?;
    let n_kernel = z.ncols();

    // Hybrid partial-fraction coefficients (None ⇒ pure polyharmonic).
    let coeffs = length_scale
        .map(|ls| duchon_partial_fraction_coeffs(p_order, s_order_int, 1.0 / ls.max(1e-300)));
    let pure_poly_coeff = if length_scale.is_none() {
        Some(PolyharmonicBlockCoeff::new(
            pure_duchon_block_order(kernel_m, s_order_f),
            dim,
        ))
    } else {
        None
    };

    // Kernel amplification α — identical to the value the design path applies.
    // The mixed-periodicity forward applies no amplification (α = 1).
    let kernel_amp = if any_periodic {
        1.0
    } else {
        duchon_kernel_amplification(
            centers,
            length_scale,
            p_order,
            s_order_int,
            dim,
            None,
            coeffs.as_ref(),
            pure_poly_coeff.as_ref(),
        )
    };

    // --------------------------------------------------- radial value block
    // Per (row, center) metric distance `r` plus the scalar radial value
    // φ(r), first derivative φ'(r), and second derivative φ''(r) of the SAME
    // kernel the forward design evaluates. `metric_axis[(n,k,a)]` is the
    // per-axis chord-embedding partial dδ_a/dx_a and `metric_axis2` its second
    // derivative (zero on linear / non-periodic axes); the displacement
    // `delta_a = δ_a(x_a − c_a)` is the chord (or plain difference) used to
    // contract the radial scalars into the input-location jet.
    let mut radial_value = Array2::<f64>::zeros((n_rows, n_centers));
    let mut radial_first = Array2::<f64>::zeros((n_rows, n_centers));
    let mut radial_second = Array2::<f64>::zeros((n_rows, n_centers));
    // delta[(n,k,a)] = embedded displacement along axis a (chord or plain).
    let mut delta = Array3::<f64>::zeros((n_rows, n_centers, dim));
    // d1[(n,k,a)] = ∂δ_a/∂x_a, d2 = ∂²δ_a/∂x_a²  (chord-embedding metric jets).
    let mut metric_d1 = Array3::<f64>::zeros((n_rows, n_centers, dim));
    let mut metric_d2 = Array3::<f64>::zeros((n_rows, n_centers, dim));

    // Pure-polyharmonic radial scalars φ, φ', φ'' all come from the SAME
    // analytic jet of the polyharmonic block `c · r^(2m_pure − d)` (or its log
    // variant) that defines the forward value, via `polyharmonic_block_jet4`.
    // This is the *exact* derivative of `ppc.eval(r)` — `polyharmonic_block_jet4`
    // and `PolyharmonicBlockCoeff::new` share the identical coefficient `c`,
    // power, and log-case branch — so the returned φ' and φ'' differentiate the
    // forward kernel value column-for-column with no Matérn-regularized
    // surrogate. (The earlier `duchon_radial_jets` path injected a fabricated
    // length scale + partial-fraction coeffs, producing φ', φ'' of a *hybrid*
    // kernel that is NOT the derivative of the pure polyharmonic `ppc.eval`.)
    // The kernel smoothness order `m_pure` is the one that built `pure_poly_coeff`.
    let m_pure = pure_duchon_block_order(kernel_m, s_order_f);

    let pi = std::f64::consts::PI;
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let raw = t[[n, a]] - centers[[k, a]];
                let (d_a, d1_a, d2_a) = if periodic_per_axis[a] {
                    // Chord embedding on the circle of circumference P_a:
                    //   δ_a   = (P/π) sin(π·raw/P)
                    //   δ_a'  = cos(π·raw/P)
                    //   δ_a'' = −(π/P) sin(π·raw/P)
                    let p = periods[a];
                    let theta = pi * raw / p;
                    ((p / pi) * theta.sin(), theta.cos(), -(pi / p) * theta.sin())
                } else {
                    (raw, 1.0, 0.0)
                };
                delta[[n, k, a]] = d_a;
                metric_d1[[n, k, a]] = d1_a;
                metric_d2[[n, k, a]] = d2_a;
                r2 += d_a * d_a;
            }
            let r = r2.sqrt();
            let (phi, phi_r, phi_rr) = if pure_poly_coeff.is_some() {
                // Exact analytic (value, φ', φ'') of the pure polyharmonic block,
                // i.e. the true derivatives of the forward `ppc.eval(r)`.
                polyharmonic_kernel_triplet(r, m_pure, dim)?
            } else {
                let jets = duchon_radial_jets(
                    r,
                    length_scale.expect("hybrid Duchon requires length_scale"),
                    p_order,
                    s_order_int,
                    dim,
                    coeffs.as_ref().expect("hybrid Duchon requires coeffs"),
                )?;
                (jets.phi, jets.phi_r, jets.phi_rr)
            };
            radial_value[[n, k]] = phi * kernel_amp;
            radial_first[[n, k]] = phi_r;
            radial_second[[n, k]] = phi_rr;
        }
    }

    // ---------------------------------------------------- forward kernel block
    let kernel_design = fast_ab(&radial_value, &z);

    // Polynomial forward block P(x).  Periodic ⇒ constant-only column.
    let poly_design = polynomial_block_from_order(t, kernel_nullspace_order);
    let n_poly = poly_design.ncols();

    let mut phi_design = Array2::<f64>::zeros((n_rows, n_kernel + n_poly));
    phi_design
        .slice_mut(s![.., ..n_kernel])
        .assign(&kernel_design);
    if n_poly > 0 {
        phi_design
            .slice_mut(s![.., n_kernel..])
            .assign(&poly_design);
    }

    // -------------------------------------------------- radial input-location
    // jets, contracted through the chord-embedding metric.  For each (n, k):
    //   ∂φ/∂x_a   = φ'(r) · (δ_a · δ_a') / r
    //   ∂²φ/∂x_a∂x_c = (φ''(r) − φ'(r)/r)/r² · (δ_a δ_a')(δ_c δ_c')
    //                + [a == c] · (φ'(r)/r) · ( (δ_a')² + δ_a · δ_a'' )
    // The first form collapses to the standard radial jet when every axis is
    // non-periodic (δ_a = x_a − c_a, δ_a' = 1, δ_a'' = 0).
    let mut radial_jet = Array3::<f64>::zeros((n_rows, n_centers, dim));
    let mut radial_hess = Array4::<f64>::zeros((n_rows, n_centers, dim, dim));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let da = delta[[n, k, a]];
                r2 += da * da;
            }
            let r = r2.sqrt();
            let phi_r = radial_first[[n, k]];
            let phi_rr = radial_second[[n, k]];
            if r <= 1.0e-12 {
                // Collision limit: the radial gradient vanishes and the Hessian
                // is the isotropic φ''(0) δ_ac scaled by (δ_a')² (= 1 at the
                // center for both linear and chord embeddings).
                for a in 0..dim {
                    let d1a = metric_d1[[n, k, a]];
                    radial_hess[[n, k, a, a]] = phi_rr * kernel_amp * d1a * d1a;
                }
                continue;
            }
            let q = phi_r / r;
            let s_scalar = (phi_rr - q) / r2;
            // g_a = δ_a · δ_a' is ∂r/∂x_a · r (the metric-contracted gradient
            // of ½r²).
            for a in 0..dim {
                let ga = delta[[n, k, a]] * metric_d1[[n, k, a]];
                radial_jet[[n, k, a]] = q * ga * kernel_amp;
            }
            for a in 0..dim {
                let ga = delta[[n, k, a]] * metric_d1[[n, k, a]];
                for c in 0..dim {
                    let gc = delta[[n, k, c]] * metric_d1[[n, k, c]];
                    let mut value = s_scalar * ga * gc;
                    if a == c {
                        let d1a = metric_d1[[n, k, a]];
                        let curvature = d1a * d1a + delta[[n, k, a]] * metric_d2[[n, k, a]];
                        value += q * curvature;
                    }
                    radial_hess[[n, k, a, c]] = value * kernel_amp;
                }
            }
        }
    }

    // ---------------------------------------------------- assemble jet (J)
    let poly_jet = duchon_polynomial_first_derivative_nd(t, kernel_nullspace_order);
    if poly_jet.shape()[1] != n_poly {
        crate::bail_dim_basis!(
            "build_duchon_basis_design_and_jets: polynomial jet has {} columns but design has {n_poly}",
            poly_jet.shape()[1]
        );
    }
    let mut jet = Array3::<f64>::zeros((n_rows, n_kernel + n_poly, dim));
    for axis in 0..dim {
        let projected = radial_jet.index_axis(Axis(2), axis).dot(&z);
        jet.slice_mut(s![.., ..n_kernel, axis]).assign(&projected);
    }
    if n_poly > 0 {
        jet.slice_mut(s![.., n_kernel.., ..]).assign(&poly_jet);
    }

    // ---------------------------------------------------- assemble Hessian (H)
    let mut hess = Array4::<f64>::zeros((n_rows, n_kernel + n_poly, dim, dim));
    for a in 0..dim {
        for c in 0..dim {
            let slab = radial_hess.slice(s![.., .., a, c]);
            let projected = slab.dot(&z);
            hess.slice_mut(s![.., ..n_kernel, a, c]).assign(&projected);
        }
    }
    if n_poly > 0 {
        // Polynomial Hessian block: ∂²(Π t_i^{α_i}) / ∂t_a ∂t_c.
        let max_degree = match kernel_nullspace_order {
            DuchonNullspaceOrder::Zero => 0,
            DuchonNullspaceOrder::Linear => 1,
            DuchonNullspaceOrder::Degree(deg) => deg,
        };
        let exponents = monomial_exponents(dim, max_degree);
        if exponents.len() != n_poly {
            crate::bail_dim_basis!(
                "build_duchon_basis_design_and_jets: monomial count {} != polynomial block columns {n_poly}",
                exponents.len()
            );
        }
        for (col, alpha) in exponents.iter().enumerate() {
            for a in 0..dim {
                let coeff_a = alpha[a];
                if coeff_a == 0 {
                    continue;
                }
                for c in 0..dim {
                    let coeff_c = if a == c {
                        alpha[c].saturating_sub(1)
                    } else {
                        alpha[c]
                    };
                    if a != c && coeff_c == 0 {
                        continue;
                    }
                    let lead = (coeff_a as f64) * (coeff_c as f64);
                    if lead == 0.0 {
                        continue;
                    }
                    for row in 0..n_rows {
                        let mut value = lead;
                        for axis in 0..dim {
                            let mut exp = alpha[axis];
                            if axis == a {
                                exp = exp.saturating_sub(1);
                            }
                            if axis == c {
                                exp = exp.saturating_sub(1);
                            }
                            if exp != 0 {
                                value *= t[[row, axis]].powi(exp as i32);
                            }
                        }
                        hess[[row, n_kernel + col, a, c]] = value;
                    }
                }
            }
        }
    }

    Ok((phi_design, jet, hess))
}


/// N-D Matérn radial first-derivative `φ'(r)` evaluated for every
/// `(row, center)` pair.
///
/// Returns an `(n_rows, n_centers)` matrix whose `(n, k)` entry is the
/// scalar radial derivative `φ'(r_{nk})` of the Matérn kernel,
/// where `r_{nk} = ‖t_n − c_k‖_2`.
///
/// This is the Matérn analogue of [`duchon_radial_first_derivative_nd`].
/// The full per-row gradient is reconstructed at the call site as
/// `∂Φ_{n,k}/∂t_n = φ'(r_{n,k}) · (t_n − c_k) / r_{n,k}` (chain rule of the
/// radial kernel w.r.t. its first argument), reusing
/// [`crate::terms::latent_coord::LatentCoordValues::design_gradient_wrt_t`].
///
/// All radial derivatives are obtained in closed form from the half-integer
/// Matérn polynomial-times-exponential representation; the underlying scalar
/// arithmetic is [`matern_kernel_radial_tripletwith_safe_ratio`].
pub fn matern_radial_first_derivative_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
) -> Result<Array2<f64>, BasisError> {
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "matern_radial_first_derivative_nd: centers must have at least one column".into(),
        );
    }
    if t.ncols() != dim {
        crate::bail_invalid_basis!(
            "matern_radial_first_derivative_nd: t has {} cols but centers have {}",
            t.ncols(),
            dim
        );
    }
    let mut out = Array2::<f64>::zeros((n_rows, n_centers));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let dv = t[[n, a]] - centers[[k, a]];
                r2 += dv * dv;
            }
            let r = r2.sqrt();
            let (_phi, phi_r, _phi_rr, _ratio) =
                matern_kernel_radial_tripletwith_safe_ratio(r, length_scale, nu)?;
            out[[n, k]] = phi_r;
        }
    }
    Ok(out)
}


/// N-D Matérn radial second derivative `φ''(r)` evaluated for every
/// `(row, center)` pair.
///
/// Companion to [`matern_radial_first_derivative_nd`]. Together they give
/// the full input-location Hessian:
///
/// ```text
/// ∂²φ/∂t_i∂t_j = (φ'(r)/r) (δ_ij − u_i u_j) + φ''(r) u_i u_j,
/// ```
/// where `u_a = (t_a − c_a) / r`. At `r = 0`, the limit reduces to the
/// isotropic `φ''(0) δ_ij`.
pub fn matern_radial_second_derivative_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
) -> Result<Array2<f64>, BasisError> {
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "matern_radial_second_derivative_nd: centers must have at least one column".into(),
        );
    }
    if t.ncols() != dim {
        crate::bail_invalid_basis!(
            "matern_radial_second_derivative_nd: t has {} cols but centers have {}",
            t.ncols(),
            dim
        );
    }
    let mut out = Array2::<f64>::zeros((n_rows, n_centers));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let dv = t[[n, a]] - centers[[k, a]];
                r2 += dv * dv;
            }
            let r = r2.sqrt();
            let (_phi, _phi_r, phi_rr, _ratio) =
                matern_kernel_radial_tripletwith_safe_ratio(r, length_scale, nu)?;
            out[[n, k]] = phi_rr;
        }
    }
    Ok(out)
}


/// Resolve the per-axis metric weights `w_a = exp(2·ψ_a)` that the Matérn
/// forward design (`build_matern_basis` → `StreamingMaternEvaluator` /
/// `ChunkedKernelDesignOperator`) applies, **bit-for-bit**.
///
/// The forward path centres the supplied anisotropy log-scales through
/// [`centered_aniso_contrasts`] (subtract the mean, zero tiny residuals) and
/// then squares `exp(ψ_a)`. Replicating that exact transform here is what lets
/// the input-location jet/Hessian differentiate the *same* function the forward
/// evaluates under anisotropy. Like the forward design, this is a pure function
/// of the supplied `η`: an explicit all-zero vector yields the isotropic
/// all-ones metric, matching the closed-form isotropic Matérn (#437, #1042).
///
/// `None` (or a 1-D problem, where the centred contrast is a no-op) yields the
/// isotropic all-ones metric.
fn matern_metric_weights(dim: usize, aniso: Option<&[f64]>) -> Vec<f64> {
    match centered_aniso_contrasts(aniso) {
        Some(psi) => psi.iter().map(|&v| (2.0 * v).exp()).collect(),
        None => vec![1.0; dim],
    }
}


/// N-D Matérn input-location **jet** `∂Φ/∂t` under the anisotropic metric.
///
/// Returns an `(n_rows, n_centers, dim)` tensor whose `(n, k, a)` entry is the
/// exact partial derivative of the (un-projected) kernel value
/// `Φ_{n,k} = φ(r_A)` w.r.t. the input coordinate `t_{n,a}`, where the
/// anisotropic radius is `r_A = √(Σ_b w_b (t_b − c_b)²)` with the forward
/// metric weights `w_b` from [`matern_metric_weights`]:
///
/// ```text
/// ∂Φ_{n,k}/∂t_{n,a} = φ'(r_A) · w_a (t_{n,a} − c_{k,a}) / r_A.
/// ```
///
/// At `r_A = 0` the kernel is at a smooth maximum and the jet is exactly `0`.
/// This is the metric-aware analogue of pairing
/// [`matern_radial_first_derivative_nd`] with `(t − c)/r`; combining the two
/// isotropic helpers ignores `w_a` and therefore differentiates a *different*
/// function whenever anisotropy is active (issue #437).
pub fn matern_input_location_jet_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    aniso_log_scales: Option<&[f64]>,
) -> Result<Array3<f64>, BasisError> {
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "matern_input_location_jet_nd: centers must have at least one column".into(),
        );
    }
    if t.ncols() != dim {
        crate::bail_invalid_basis!(
            "matern_input_location_jet_nd: t has {} cols but centers have {}",
            t.ncols(),
            dim
        );
    }
    if let Some(eta) = aniso_log_scales
        && eta.len() != dim
    {
        crate::bail_invalid_basis!(
            "matern_input_location_jet_nd: aniso_log_scales len {} != dim {}",
            eta.len(),
            dim
        );
    }
    let weights = matern_metric_weights(dim, aniso_log_scales);
    let mut out = Array3::<f64>::zeros((n_rows, n_centers, dim));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let h = t[[n, a]] - centers[[k, a]];
                r2 += weights[a] * h * h;
            }
            let r = r2.sqrt();
            if r <= 0.0 {
                continue;
            }
            let (_phi, phi_r, _phi_rr, _ratio) =
                matern_kernel_radial_tripletwith_safe_ratio(r, length_scale, nu)?;
            let scale = phi_r / r;
            for a in 0..dim {
                let h = t[[n, a]] - centers[[k, a]];
                out[[n, k, a]] = scale * weights[a] * h;
            }
        }
    }
    Ok(out)
}


/// N-D Matérn input-location **Hessian** `∂²Φ/∂t∂tᵀ` under the anisotropic
/// metric.
///
/// Returns an `(n_rows, n_centers, dim, dim)` tensor whose `(n, k, a, c)`
/// entry is the exact second partial of `Φ_{n,k} = φ(r_A)` (same `r_A` and
/// forward metric weights `w` as [`matern_input_location_jet_nd`]):
///
/// ```text
/// H_{ac} = φ''(r_A) · (w_a h_a / r_A)(w_c h_c / r_A)
///        + (φ'(r_A)/r_A) · (w_a δ_{ac} − w_a h_a w_c h_c / r_A²),
/// ```
/// with `h = t − c`. At `r_A = 0` the smooth limit collapses to the diagonal
/// `(φ'/r)|_0 · w_a δ_{ac}` (the regularized ratio from
/// [`matern_kernel_radial_tripletwith_safe_ratio`], which equals `φ''(0)` for
/// ν ≥ 3/2 and carries the genuine ν = 1/2 singularity floor).
pub fn matern_input_location_hessian_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    aniso_log_scales: Option<&[f64]>,
) -> Result<Array4<f64>, BasisError> {
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "matern_input_location_hessian_nd: centers must have at least one column".into(),
        );
    }
    if t.ncols() != dim {
        crate::bail_invalid_basis!(
            "matern_input_location_hessian_nd: t has {} cols but centers have {}",
            t.ncols(),
            dim
        );
    }
    if let Some(eta) = aniso_log_scales
        && eta.len() != dim
    {
        crate::bail_invalid_basis!(
            "matern_input_location_hessian_nd: aniso_log_scales len {} != dim {}",
            eta.len(),
            dim
        );
    }
    let weights = matern_metric_weights(dim, aniso_log_scales);
    let mut out = Array4::<f64>::zeros((n_rows, n_centers, dim, dim));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let h = t[[n, a]] - centers[[k, a]];
                r2 += weights[a] * h * h;
            }
            let r = r2.sqrt();
            let (_phi, phi_r, phi_rr, phi_r_over_r) =
                matern_kernel_radial_tripletwith_safe_ratio(r, length_scale, nu)?;
            if r <= 0.0 {
                // Smooth collision limit: gradient component `w_a h_a / r` → 0,
                // so only the diagonal `(φ'/r)·w_a` term survives.
                for a in 0..dim {
                    out[[n, k, a, a]] = phi_r_over_r * weights[a];
                }
                continue;
            }
            let q = phi_r / r; // = φ'(r)/r at this r > 0.
            let inv_r2 = 1.0 / r2;
            for a in 0..dim {
                let ga = weights[a] * (t[[n, a]] - centers[[k, a]]); // W h, component a
                for c in 0..dim {
                    let gc = weights[c] * (t[[n, c]] - centers[[k, c]]);
                    // φ''(r)·(W h/r)_a (W h/r)_c
                    let mut value = phi_rr * (ga / r) * (gc / r);
                    // (φ'/r)·(−w_a h_a w_c h_c / r²)
                    value -= q * ga * gc * inv_r2;
                    if a == c {
                        // (φ'/r)·w_a δ_ac
                        value += q * weights[a];
                    }
                    out[[n, k, a, c]] = value;
                }
            }
        }
    }
    Ok(out)
}


/// N-D Sobolev-sphere first-derivative jet `∂Φ/∂t` per row, on the unit
/// sphere `S^{dim−1}`.
///
/// `points` is `(n_rows, dim)` ambient unit vectors `t_n ∈ ℝ^dim`,
/// `centers` is `(n_centers, dim)` ambient unit vectors `c_k`. The kernel
/// is `K(cos γ)` with `cos γ = t · c`, and the chain rule gives
///
/// ```text
///     ∂Φ_{n,k} / ∂t_n = K'(cos γ_{n,k}) · c_k,
/// ```
///
/// where `K'` is `dK/d(cos γ)` from
/// [`wahba_sphere_kernel_sobolev_derivative_dcos`].
///
/// When `project_to_tangent` is `true`, each per-row gradient is projected
/// through [`crate::terms::latent_coord::LatentManifold::Sphere`] onto
/// `T_{t_n} S^{dim-1}` as `g − (g · t_n) t_n`, which is the correct
/// Riemannian input-location derivative for embedded-sphere latent updates.
/// Passing `false` returns the un-projected ambient jet.
///
/// Returned tensor shape: `(n_rows, n_centers, dim)`.
pub fn sphere_first_derivative_nd(
    points: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    penalty_order: usize,
    project_to_tangent: bool,
) -> Result<Array3<f64>, BasisError> {
    let n_rows = points.nrows();
    let n_centers = centers.nrows();
    let dim = points.ncols();
    if !(1..=4).contains(&penalty_order) {
        crate::bail_invalid_basis!(
            "sphere_first_derivative_nd: penalty_order must be in 1..=4; got {penalty_order}"
        );
    }
    if dim == 0 {
        crate::bail_invalid_basis!(
            "sphere_first_derivative_nd: points must have at least one column".into(),
        );
    }
    if centers.ncols() != dim {
        crate::bail_invalid_basis!(
            "sphere_first_derivative_nd: points have dim {} but centers have dim {}",
            dim,
            centers.ncols()
        );
    }
    let tangent_projector =
        project_to_tangent.then_some(crate::terms::latent_coord::LatentManifold::Sphere { dim });
    let mut out = Array3::<f64>::zeros((n_rows, n_centers, dim));
    let mut ambient = Array1::<f64>::zeros(dim);
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut cos_g = 0.0_f64;
            for a in 0..dim {
                cos_g += points[[n, a]] * centers[[k, a]];
            }
            let dk = wahba_sphere_kernel_sobolev_derivative_dcos(cos_g, penalty_order);
            for a in 0..dim {
                ambient[a] = dk * centers[[k, a]];
            }
            if let Some(manifold) = &tangent_projector {
                let tangent = manifold.project_to_tangent(points.row(n), ambient.view());
                for a in 0..dim {
                    out[[n, k, a]] = tangent[a];
                }
            } else {
                for a in 0..dim {
                    out[[n, k, a]] = ambient[a];
                }
            }
        }
    }
    Ok(out)
}


/// Raw (pre-identifiability) Wahba sphere DESIGN jet `∂Φ_raw/∂(lat, lon)`.
///
/// `data` is `(N, 2)` lat/lon, `centers` is `(K, 2)` lat/lon, both in the same
/// angular convention selected by `radians`. Returns `(N, K, 2)` where the
/// last axis is `(∂col/∂lat, ∂col/∂lon)` in the SAME angular units as the
/// input — i.e. the radian-space derivative scaled by `deg = radians ? 1 :
/// π/180`.
///
/// With `cos γ = sinφ sinφc + cosφ cosφc cos(ψ − ψc)` (φ, ψ in radians):
///   ∂cosγ/∂φ = cosφ sinφc − sinφ cosφc cos(ψ − ψc),
///   ∂cosγ/∂ψ = −cosφ cosφc sin(ψ − ψc),
/// and ∂Φ/∂φ = K'(cosγ)·∂cosγ/∂φ, ∂Φ/∂ψ = K'(cosγ)·∂cosγ/∂ψ. The raw-radian
/// derivatives are multiplied by `deg` to express them per raw input unit.
fn spherical_wahba_kernel_jet_with_kind(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    penalty_order: usize,
    radians: bool,
    kernel: SphereWahbaKernel,
) -> Result<Array3<f64>, BasisError> {
    validate_lat_lon_matrix(data, "spherical spline jet data", radians)?;
    validate_lat_lon_matrix(centers, "spherical spline jet centers", radians)?;
    if !(1..=4).contains(&penalty_order) {
        crate::bail_invalid_basis!(
            "spherical spline jet penalty_order must be one of 1, 2, 3, 4; got {penalty_order}"
        );
    }
    let n = data.nrows();
    let k = centers.nrows();
    let deg = if radians {
        1.0
    } else {
        std::f64::consts::PI / 180.0
    };
    let mut sin_lat_c = Vec::<f64>::with_capacity(k);
    let mut cos_lat_c = Vec::<f64>::with_capacity(k);
    let mut sin_lon_c = Vec::<f64>::with_capacity(k);
    let mut cos_lon_c = Vec::<f64>::with_capacity(k);
    for c in centers.outer_iter() {
        let (s_lat, c_lat) = (c[0] * deg).sin_cos();
        let (s_lon, c_lon) = (c[1] * deg).sin_cos();
        sin_lat_c.push(s_lat);
        cos_lat_c.push(c_lat);
        sin_lon_c.push(s_lon);
        cos_lon_c.push(c_lon);
    }
    let mut out = Array3::<f64>::zeros((n, k, 2));
    let err_flag = std::sync::atomic::AtomicBool::new(false);
    out.axis_chunks_iter_mut(ndarray::Axis(0), 256)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut block)| {
            let row_offset = chunk_idx * 256;
            for (local_i, mut out_row) in block.outer_iter_mut().enumerate() {
                let i = row_offset + local_i;
                let (sin_lat, cos_lat) = (data[(i, 0)] * deg).sin_cos();
                let (sin_lon, cos_lon) = (data[(i, 1)] * deg).sin_cos();
                for j in 0..k {
                    // cos(ψ − ψc) and sin(ψ − ψc) via angle-subtraction.
                    let dlon_cos = cos_lon * cos_lon_c[j] + sin_lon * sin_lon_c[j];
                    let dlon_sin = sin_lon * cos_lon_c[j] - cos_lon * sin_lon_c[j];
                    let cos_gamma = sin_lat * sin_lat_c[j] + cos_lat * cos_lat_c[j] * dlon_cos;
                    let dk =
                        wahba_sphere_kernel_derivative_dcos_kind(cos_gamma, penalty_order, kernel);
                    // ∂cosγ/∂φ and ∂cosγ/∂ψ (radian space).
                    let dcos_dphi = cos_lat * sin_lat_c[j] - sin_lat * cos_lat_c[j] * dlon_cos;
                    let dcos_dpsi = -cos_lat * cos_lat_c[j] * dlon_sin;
                    let dphi = dk * dcos_dphi * deg;
                    let dpsi = dk * dcos_dpsi * deg;
                    if !dphi.is_finite() || !dpsi.is_finite() {
                        err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        return;
                    }
                    out_row[[j, 0]] = dphi;
                    out_row[[j, 1]] = dpsi;
                }
            }
        });
    if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
        crate::bail_invalid_basis!("spherical spline kernel jet produced a non-finite value");
    }
    Ok(out)
}


/// Apply the `(K × K')` identifiability transform `z` to a raw Wahba jet
/// `(N, K, 2)`, producing the realized-design jet `(N, K', 2)` whose column
/// `c` aligns with column `c` of `raw_design.dot(z)`. The transform is linear
/// in the kernel columns, so `∂(Φ_raw z)/∂t = (∂Φ_raw/∂t) z` axis-by-axis.
fn apply_identifiability_to_jet(raw_jet: &Array3<f64>, z: &Array2<f64>) -> Array3<f64> {
    let n = raw_jet.shape()[0];
    let k = raw_jet.shape()[1];
    let kp = z.ncols();
    assert_eq!(
        z.nrows(),
        k,
        "apply_identifiability_to_jet: identifiability transform rows ({}) must match raw jet basis dim ({})",
        z.nrows(),
        k
    );
    let mut out = Array3::<f64>::zeros((n, kp, 2));
    for axis in 0..2 {
        // raw_axis: (N, K); out_axis = raw_axis · z → (N, K').
        let raw_axis: ndarray::ArrayView2<'_, f64> = raw_jet.index_axis(ndarray::Axis(2), axis);
        let projected = raw_axis.dot(z);
        out.slice_mut(ndarray::s![.., .., axis]).assign(&projected);
    }
    out
}


/// Real-spherical-harmonic DESIGN jet `∂Φ/∂(lat, lon)`, shape `(N, p, 2)` with
/// `p = L(L+2)` and column order matching [`fill_real_spherical_harmonics_row`].
///
/// With `x = sinφ`, column `= N_{lm}·T_m(ψ)·P_{lm}(x)` where `T_m` is
/// `sin(mψ)`, `1`, or `cos(mψ)`:
///   ∂col/∂φ = N_{lm}·T_m(ψ)·P'_{lm}(x)·cosφ   (dx/dφ = cosφ),
///   ∂col/∂ψ = N_{lm}·T'_m(ψ)·P_{lm}(x)         (T' = m cos(mψ), 0, −m sin(mψ)).
/// `P'_{lm}(x)` from `(1 − x²) P'_{lm}(x) = −l x P_{lm}(x) + (l+m) P_{l−1,m}(x)`,
/// with the forward's latitude clamp and `somx2` floor reused for the poles.
/// The radian-space derivatives are scaled by `deg` to per-raw-unit values.
fn spherical_harmonic_jet(
    data: ArrayView2<'_, f64>,
    max_degree: usize,
    radians: bool,
) -> Result<Array3<f64>, BasisError> {
    validate_lat_lon_matrix(data, "spherical-harmonic jet", radians)?;
    if max_degree < 1 {
        crate::bail_invalid_basis!("spherical-harmonic jet max_degree must be >= 1");
    }
    if max_degree > 32 {
        crate::bail_invalid_basis!(
            "spherical-harmonic jet max_degree {max_degree} too large; cap is 32"
        );
    }
    let n = data.nrows();
    let p = max_degree * (max_degree + 2);
    let deg = if radians {
        1.0
    } else {
        std::f64::consts::PI / 180.0
    };
    let norms = precompute_harmonic_norms(max_degree);
    let l_cap = max_degree + 1;
    let mut out = Array3::<f64>::zeros((n, p, 2));
    let idx = |l: usize, m: usize| l * l_cap + m;
    {
        let mut row_blocks = out
            .axis_chunks_iter_mut(ndarray::Axis(0), 1024)
            .collect::<Vec<_>>();
        let chunk_size = 1024usize;
        row_blocks
            .par_iter_mut()
            .enumerate()
            .for_each(|(chunk_idx, block)| {
                let mut p_buf = vec![0.0_f64; l_cap * l_cap];
                let row_offset = chunk_idx * chunk_size;
                for (local_i, mut out_row) in block.outer_iter_mut().enumerate() {
                    let i = row_offset + local_i;
                    let lat_raw = data[(i, 0)] * deg;
                    let lat =
                        lat_raw.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
                    let lon = data[(i, 1)] * deg;
                    let cos_lat = lat.cos();
                    let x = lat.sin();
                    let somx2 = (1.0 - x * x).max(0.0).sqrt();
                    let one_minus_x2 = (1.0 - x * x).max(f64::EPSILON);
                    // Associated Legendre P_{l,m}(x) — identical recurrence to
                    // `fill_real_spherical_harmonics_row`.
                    for slot in p_buf.iter_mut() {
                        *slot = 0.0;
                    }
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
                    // P'_{l,m}(x) via (1 − x²) P'_{l,m} = −l x P_{l,m} + (l+m) P_{l−1,m}.
                    let dp = |l: usize, m: usize| -> f64 {
                        let p_lm1 = if l >= 1 { p_buf[idx(l - 1, m)] } else { 0.0 };
                        (-(l as f64) * x * p_buf[idx(l, m)] + ((l + m) as f64) * p_lm1)
                            / one_minus_x2
                    };
                    // sin(mψ), cos(mψ) via Chebyshev recurrence (mirror forward).
                    let (sin1, cos1) = lon.sin_cos();
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
                        // sin(mψ) columns for m = l, l-1, ..., 1.
                        for m_pos in (1..=l).rev() {
                            let nlm = norms[idx(l, m_pos)];
                            let mf = m_pos as f64;
                            // ∂/∂φ = N·sin(mψ)·P'·cosφ ; ∂/∂ψ = N·m cos(mψ)·P.
                            out_row[[col, 0]] = nlm * sin_buf[m_pos] * dp(l, m_pos) * cos_lat * deg;
                            out_row[[col, 1]] =
                                nlm * mf * cos_buf[m_pos] * p_buf[idx(l, m_pos)] * deg;
                            col += 1;
                        }
                        // m = 0: no trig factor → ∂/∂ψ = 0.
                        let nl0 = norms[idx(l, 0)];
                        out_row[[col, 0]] = nl0 * dp(l, 0) * cos_lat * deg;
                        out_row[[col, 1]] = 0.0;
                        col += 1;
                        // cos(mψ) columns for m = 1, ..., l.
                        for m in 1..=l {
                            let nlm = norms[idx(l, m)];
                            let mf = m as f64;
                            // ∂/∂φ = N·cos(mψ)·P'·cosφ ; ∂/∂ψ = −N·m sin(mψ)·P.
                            out_row[[col, 0]] = nlm * cos_buf[m] * dp(l, m) * cos_lat * deg;
                            out_row[[col, 1]] = -nlm * mf * sin_buf[m] * p_buf[idx(l, m)] * deg;
                            col += 1;
                        }
                    }
                }
            });
    }
    if out.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("spherical-harmonic jet produced a non-finite value");
    }
    Ok(out)
}


/// Realized-design DESIGN jet `∂Φ/∂(lat, lon)` for the spherical-spline basis,
/// matching the column layout of [`build_spherical_spline_basis`] with the
/// given `spec`. Returns `(N, K, 2)` where `K` equals the forward design's
/// column count and the last axis is `(∂col/∂lat, ∂col/∂lon)` in the same
/// angular units as the raw input.
///
/// - **Harmonic** (`spec.method == Harmonic`): `K = L(L+2)`, no transform.
/// - **Wahba** (Sobolev/Pseudo/truncated): centers are resolved exactly as the
///   forward does, the raw `(N, K_c, 2)` kernel jet is built, then contracted
///   with the same area-weighted sum-to-zero (or frozen) transform `z` so the
///   result aligns column-for-column with `raw_design · z`.
pub fn spherical_spline_design_jet(
    data: ArrayView2<'_, f64>,
    spec: &SphericalSplineBasisSpec,
) -> Result<Array3<f64>, BasisError> {
    if matches!(spec.method, SphereMethod::Harmonic) {
        let l_max = spec
            .max_degree
            .unwrap_or_else(|| default_spherical_harmonic_degree(data.nrows()));
        if !(1..=4).contains(&spec.penalty_order) {
            crate::bail_invalid_basis!(
                "spherical-harmonic jet penalty_order must be one of 1, 2, 3, 4; got {}",
                spec.penalty_order
            );
        }
        return spherical_harmonic_jet(data, l_max, spec.radians);
    }
    validate_lat_lon_matrix(data, "spherical spline jet", spec.radians)?;
    let centers = match realized_center_strategy(&spec.center_strategy) {
        CenterStrategy::FarthestPoint { num_centers } => {
            select_spherical_farthest_point_centers(data, *num_centers, spec.radians)?
        }
        _ => select_centers_by_strategy(data, &spec.center_strategy)?,
    };
    validate_lat_lon_matrix(centers.view(), "spherical spline jet centers", spec.radians)?;
    if centers.nrows() < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: centers.nrows(),
        });
    }
    let z = match &spec.identifiability {
        SphericalSplineIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != centers.nrows() {
                crate::bail_dim_basis!(
                    "frozen spherical identifiability transform mismatch: {} centers but transform has {} rows",
                    centers.nrows(),
                    transform.nrows()
                );
            }
            transform.clone()
        }
        SphericalSplineIdentifiability::CenterSumToZero => {
            let weights = sphere_area_weights(centers.view(), spec.radians);
            weighted_coefficient_sum_to_zero_transform(weights.view())?
        }
    };
    let raw_jet = spherical_wahba_kernel_jet_with_kind(
        data,
        centers.view(),
        spec.penalty_order,
        spec.radians,
        spec.wahba_kernel,
    )?;
    Ok(apply_identifiability_to_jet(&raw_jet, &z))
}


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
fn periodic_distance_1d(x: f64, c: f64, period: f64) -> f64 {
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
fn even_bernoulli_polynomial(degree: usize, t: f64) -> Result<f64, BasisError> {
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
fn periodic_duchon_kernel_bernoulli(r: f64, m: usize, period: f64) -> Result<f64, BasisError> {
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
fn even_bernoulli_polynomial_derivatives(degree: usize, s: f64) -> Result<(f64, f64), BasisError> {
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
fn periodic_duchon_kernel_bernoulli_triplet(
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
fn collapse_periodic_endpoint(centers: Array2<f64>, left: f64, period: f64) -> Array2<f64> {
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


fn build_periodic_duchon_basis_1d(
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
fn duchon_mixed_periodicity_distance(
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
fn build_duchon_basis_mixed_periodicity(
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
fn duchon_native_penalty_candidates(
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
const DUCHON_COLLOCATION_OVERSAMPLE: usize = 3;


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
fn duchon_operator_penalty_candidates(
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
