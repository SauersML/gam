use super::*;

/// Cross-disease Duchon basis cache.
///
/// The biobank workload fits many models (e.g. 17 diseases) over the SAME base
/// cohort: identical individuals, identical predictor columns (PC1..PC15, sex,
/// ages, geography); only the response/PRS column changes per fit. The Duchon
/// spatial basis — center/knot selection, the thin-plate kernel evaluation, the
/// kernel-constraint nullspace reparameterisation, the identifiability
/// transform, and the penalty Grams — is a PURE FUNCTION of `(data, spec)`: it
/// never reads the response. So the whole [`BasisBuildResult`] is identical
/// across diseases sharing those exact predictor columns, yet the per-disease
/// log shows it rebuilt + re-audited from scratch each time (~0.4–2.5 s each).
///
/// This is a content-addressed, size-bounded, recomputable memo mirroring the
/// FFI cross-disease column-encode cache (`encoded_column_cache` in
/// `crates/gam-pyffi/src/manifold_and_posterior_ffi.rs`): the key is a 128-bit
/// fingerprint of the data matrix CONTENT (shape + every element bit-pattern)
/// plus the basis spec, so a different cohort / subsample (different rows) MISSES
/// — preserving correctness — while the same columns across diseases HIT. A hit
/// clones the cached `BasisBuildResult` (cheap `Arc`/ndarray clones vs. the
/// kernel build + RRQR audit), so results are bit-identical to the miss path.
/// Eviction (LRU under a byte budget) only ever forfeits the perf benefit, never
/// correctness, since every value is exactly recomputable from its key.
type DuchonBasisCacheKey = (u64, u64);

#[derive(Clone)]
struct CachedDuchonBasis(BasisBuildResult);

impl gam_runtime::resource::ResidentBytes for CachedDuchonBasis {
    fn resident_bytes(&self) -> usize {
        // Coarse charge: the dominant resident cost is the dense design columns
        // and the penalty Grams. An estimate suffices — the byte budget only
        // bounds the cache, it never affects correctness.
        let design_bytes = self
            .0
            .design
            .nrows()
            .saturating_mul(self.0.design.ncols())
            .saturating_mul(std::mem::size_of::<f64>());
        let penalty_bytes: usize = self
            .0
            .penalties
            .iter()
            .map(|s| s.len().saturating_mul(std::mem::size_of::<f64>()))
            .sum();
        design_bytes
            .saturating_add(penalty_bytes)
            .saturating_add(4096)
    }
}

/// Process-wide Duchon basis memo. 1 GiB matches the established large-scale
/// densification ceiling used elsewhere; with ~17 diseases over one cohort the
/// working set is a single `BasisBuildResult`, so even a modest budget retains
/// the shared basis across the whole sweep.
fn duchon_basis_cache()
-> &'static gam_runtime::resource::ByteLruCache<DuchonBasisCacheKey, CachedDuchonBasis> {
    static CACHE: std::sync::OnceLock<
        gam_runtime::resource::ByteLruCache<DuchonBasisCacheKey, CachedDuchonBasis>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| gam_runtime::resource::ByteLruCache::new(1 << 30))
}

/// 128-bit content fingerprint of `(data, spec)`. Two independent hashers (one
/// unseeded, one seeded with a fixed golden-ratio constant) widen the key to
/// 128 bits so accidental collisions across a batch are negligible. The data
/// contribution hashes the shape plus EVERY element's IEEE-754 bit pattern, so
/// any change of rows, columns, or values — i.e. a different cohort / subsample
/// — produces a different key and misses. The spec is hashed via its serialized
/// form (the spec carries `serde` derives), capturing center strategy, power,
/// length scale, nullspace order, anisotropy, identifiability, and operator
/// penalty dials — every input the builder reads.
fn duchon_basis_fingerprint(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Option<DuchonBasisCacheKey> {
    let spec_bytes = serde_json::to_vec(spec).ok()?;
    let mut lo = DefaultHasher::new();
    let mut hi = DefaultHasher::new();
    // Seed `hi` so its stream is statistically independent of `lo`.
    0x9E37_79B9_7F4A_7C15u64.hash(&mut hi);

    let (nrows, ncols) = data.dim();
    for h in [&mut lo, &mut hi] {
        nrows.hash(h);
        ncols.hash(h);
    }
    // Hash element bit-patterns in a fixed (row-major) order, independent of the
    // view's underlying memory layout, so two views over the same logical matrix
    // fingerprint identically.
    for row in data.rows() {
        for &v in row {
            let bits = v.to_bits();
            bits.hash(&mut lo);
            bits.hash(&mut hi);
        }
    }
    for h in [&mut lo, &mut hi] {
        spec_bytes.len().hash(h);
        spec_bytes.hash(h);
    }
    Some((lo.finish(), hi.finish()))
}

pub fn build_duchon_basiswithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    if let Some(key) = duchon_basis_fingerprint(data, spec) {
        if let Some(hit) = duchon_basis_cache().get(&key) {
            return Ok(hit.0);
        }
        let result = build_duchon_basis_uncached(data, spec, workspace)?;
        duchon_basis_cache().insert(key, CachedDuchonBasis(result.clone()));
        return Ok(result);
    }
    build_duchon_basis_uncached(data, spec, workspace)
}

fn build_duchon_basis_uncached(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    if let Some((start, end, _period)) = spec.boundary.period() {
        return build_cyclic_duchon_basis_1dwithworkspace(data, spec, start, end);
    }
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    assert_spatial_centers_below_large_scale_cap(data.ncols(), centers.view())?;
    if let Some(periodic) = spec.periodic.as_ref() {
        if periodic.len() != data.ncols() {
            crate::bail_invalid_basis!(
                "periodic must have length d={}, got {}",
                data.ncols(),
                periodic.len()
            );
        }
        if data.ncols() > 1 && periodic.iter().any(Option::is_some) {
            let flags = periodic.iter().map(Option::is_some).collect::<Vec<_>>();
            let periods = periodic
                .iter()
                .map(|axis| axis.unwrap_or(1.0))
                .collect::<Vec<_>>();
            return build_duchon_basis_mixed_periodicity_auto(data, spec, &flags, Some(&periods));
        }
        return build_periodic_duchon_basis_1d(data, spec, centers, workspace);
    }
    // `spec.power` is the LITERAL Duchon spectral power `s` at the basis layer.
    // The kernel exponent is `2(p+s) − d`, so `power = 0` means `s = 0` — the
    // integer-order Duchon kernel `r^{2(p)−d}` (its `r²·log r` log case in even
    // `d`, which equals the thin-plate kernel) — and is honored verbatim, NOT
    // read as "apply a default". The magic cubic default (no explicit power ⇒
    // `s = (d−1)/2`, `φ(r)=r³`) is a REQUEST-LAYER choice the formula/CLI/pyffi
    // front-ends resolve via `duchon_cubic_default`; the builder uses whatever
    // `(nullspace_order, power)` it is handed, so both Duchon spectral powers —
    // `s = 0` (thin-plate kernel) and `s = (d−1)/2` (fractional cubic) — are
    // reachable through this one construction.
    //
    // Auto-degrade the requested null-space order to Zero when the selected
    // centers cannot span the requested polynomial block. Every downstream
    // consumer of `spec.nullspace_order` in this function MUST use the
    // effective order, otherwise the penalty/nullspace is built with a
    // different order than the basis.
    let effective_nullspace_order =
        duchon_effective_nullspace_order(centers.view(), spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    // Initialize anisotropy contrasts from knot cloud geometry when the caller
    // enabled scale-dimensions but left η at the zero default. Duchon η is a
    // FIXED, geometry-derived basis parameter (never a REML hyper-axis), so the
    // all-zero auto-seed sentinel is the intended seeding mechanism here — unlike
    // the Matérn forward path, whose η is optimized and must be honored literally.
    let aniso = auto_seed_aniso_contrasts(centers.view(), spec.aniso_log_scales.as_deref());
    // The native reproducing-norm Gram penalty (`Primary`) is assembled from
    // kernel VALUES at the center pairs (K_CC), not from collocated D1/D2
    // derivative operators, so the build only requires the pointwise kernel to
    // EXIST (`2(p+s) > d`). The stricter operator-collocation orders
    // (`2(p+s) > d+1` / `> d+2`) are a property of the old triple-operator
    // penalties that this path no longer builds; enforcing them here would
    // spuriously reject valid kernels — e.g. the `s=0` thin-plate `r²·log r`
    // (`2(p+s)=d+2` in 2D), which the native Gram handles fine.
    //
    // Validate against the spectral power the kernel actually evaluates. The
    // scale-free native Gram (`length_scale=None`) uses the literal fractional
    // `spec.power`. The hybrid Matérn-blended kernel (`length_scale=Some`) is
    // built from the integer partial-fraction expansion of `(κ²+‖w‖²)^s` and
    // reads `s` back through `power_as_usize` (a fractional `spec.power` is
    // truncated to that integer). Validating the raw fractional power on the
    // hybrid path desyncs the `2(p+s) > d` well-posedness gate from the realized
    // kernel: e.g. the cubic default `s=(d-1)/2=1.5` at p=2, d=4 truncates to
    // s=0 where `2(p+s)=4=d` is NOT finite at the origin, yet `spec.power=1.5`
    // passes the gate — the resulting non-finite Gram crashes the constraint
    // eigendecomposition (gh#750). Gate on the truncated integer for hybrid so
    // that case is rejected here with a clear message while every valid hybrid
    // config (e.g. 1D, where `2(2+0)=4>1` stays finite) still builds.
    let validation_power = if spec.length_scale.is_some() {
        spec.power_as_usize() as f64
    } else {
        spec.power
    };
    validate_duchon_kernel_orders(spec.length_scale, p_order, validation_power, data.ncols())?;
    let mut kernel_transform = kernel_constraint_nullspace(
        centers.view(),
        effective_nullspace_order,
        &mut workspace.cache,
    )?;
    let poly_cols = polynomial_block_from_order(data, effective_nullspace_order).ncols();
    let base_cols = kernel_transform.ncols() + poly_cols;
    let dense_bytes = dense_design_bytes(data.nrows(), base_cols);
    let use_lazy = should_use_lazy_spatial_design(data.nrows(), base_cols, workspace.policy());
    // #1355: data-metric radial reparameterization `V`, frozen into metadata so
    // predict / κ-trial rebuilds replay the exact fit-time rotated radial basis.
    // A FROZEN `V` (predict / κ-trial / replay) is folded into the constrained
    // kernel transform on EVERY path so the design stays consistent with the
    // frozen penalty. A FRESH `V` is computed only on the dense cold path; the
    // lazy/streaming cold path keeps the original constrained basis (`None`).
    let mut frozen_radial_reparam: Option<Array2<f64>> = None;
    if let Some(v) = spec.radial_reparam.as_ref() {
        if v.nrows() != kernel_transform.ncols() {
            crate::bail_dim_basis!(
                "Duchon frozen radial reparam shape {:?} does not match constrained kernel dimension {}",
                v.dim(),
                kernel_transform.ncols()
            );
        }
        kernel_transform = fast_ab(&kernel_transform, v);
        frozen_radial_reparam = Some(v.clone());
    }
    let (design, identifiability_transform) = if use_lazy {
        // log::info! — deliberate memory-saving choice, not an anomaly.
        log::info!(
            "Duchon basis switching to lazy chunked design: n={} p={} ({:.1} MiB dense)",
            data.nrows(),
            base_cols,
            dense_bytes as f64 / (1024.0 * 1024.0),
        );
        let d = data.ncols();
        let shared_data = shared_owned_data_matrix(data, &workspace.cache);
        let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
        let s_order: f64 = spec.power;
        let length_scale = spec.length_scale;
        let s_order_int = length_scale.map(|_| duchon_power_to_usize(s_order));
        let coeffs = length_scale.map(|ls| {
            // Hybrid Matérn (length_scale = Some) uses the integer
            // partial-fraction chain; assert at this boundary so the
            // scale-free path stays fractional-clean.
            duchon_partial_fraction_coeffs(
                p_order,
                s_order_int.expect("hybrid Duchon requires integer power"),
                1.0 / ls.max(1e-300),
            )
        });
        let pure_poly_coeff = if length_scale.is_none() {
            Some(PolyharmonicBlockCoeff::new(
                pure_duchon_block_order(p_order, s_order),
                d,
            ))
        } else {
            None
        };
        // Translation-invariant polynomial frame (#1375): build the explicit
        // poly null-space columns at coordinates centered by the center-cloud
        // per-axis mean, matching `build_duchon_basis_designwithworkspace` (dense
        // path) and the side-condition `Z` (centered inside
        // `kernel_constraint_nullspace`). The kernel block reads `data − centers`
        // differences, so it is already translation-invariant and stays raw.
        let center_mean: Vec<f64> = (0..d)
            .map(|c| centers.column(c).sum() / (centers.nrows().max(1) as f64))
            .collect();
        let mut data_centered = data.to_owned();
        for c in 0..d {
            let mu = center_mean[c];
            data_centered.column_mut(c).mapv_inplace(|v| v - mu);
        }
        let poly_block =
            polynomial_block_from_order(data_centered.view(), effective_nullspace_order);
        let kernel_amp = duchon_kernel_amplification(
            centers.view(),
            length_scale,
            p_order,
            duchon_power_to_usize(s_order),
            d,
            aniso.as_deref(),
            coeffs.as_ref(),
            pure_poly_coeff.as_ref(),
        );
        let base_design = if let Some(eta) = aniso.as_ref() {
            let metric_weights = eta.iter().map(|&v| (2.0 * v).exp()).collect::<Vec<_>>();
            let coeffs = coeffs.clone();
            let kernel = move |data_row: &[f64], center_row: &[f64]| -> f64 {
                let mut q = 0.0f64;
                for axis in 0..data_row.len() {
                    let delta = data_row[axis] - center_row[axis];
                    q += metric_weights[axis] * delta * delta;
                }
                let r = q.sqrt();
                let raw = if let Some(ppc) = pure_poly_coeff {
                    ppc.eval(r)
                } else {
                    duchon_matern_kernel_general_from_distance(
                        r,
                        length_scale,
                        p_order,
                        s_order_int.expect("hybrid Duchon requires integer power"),
                        d,
                        coeffs.as_ref(),
                    )
                    .expect("validated Duchon inputs should not fail")
                };
                raw * kernel_amp
            };
            let kernel_gauge = Arc::new(gam_problem::Gauge::from_block_transforms(&[
                kernel_transform.clone(),
            ]));
            let base_op = ChunkedKernelDesignOperator::new(
                shared_data.clone(),
                Arc::new(centers.clone()),
                kernel,
                Some(kernel_gauge),
                Some(Arc::new(poly_block.clone())),
            )
            .map_err(BasisError::InvalidInput)?;
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(
                base_op,
            )))
        } else {
            let coeffs = coeffs.clone();
            let make_kernel = || {
                let coeffs = coeffs.clone();
                let pure_poly_coeff = pure_poly_coeff;
                Arc::new(move |data_row: &[f64], center_row: &[f64]| -> f64 {
                    let r =
                        stable_euclidean_norm((0..d).map(|axis| data_row[axis] - center_row[axis]));
                    let raw = if let Some(ppc) = pure_poly_coeff {
                        ppc.eval(r)
                    } else {
                        duchon_matern_kernel_general_from_distance(
                            r,
                            length_scale,
                            p_order,
                            s_order_int.expect("hybrid Duchon requires integer power"),
                            d,
                            coeffs.as_ref(),
                        )
                        .expect("validated Duchon inputs should not fail")
                    };
                    raw * kernel_amp
                }) as Arc<dyn crate::chunked_kernel_design::SpatialKernelEvaluator>
            };
            let operators_active = matches!(
                spec.operator_penalties.mass,
                OperatorPenaltySpec::Active { .. }
            ) || matches!(
                spec.operator_penalties.tension,
                OperatorPenaltySpec::Active { .. }
            ) || matches!(
                spec.operator_penalties.stiffness,
                OperatorPenaltySpec::Active { .. }
            );
            if frozen_radial_reparam.is_none() && !operators_active {
                let raw_gauge = Arc::new(gam_problem::Gauge::from_block_transforms(&[
                    kernel_transform.clone(),
                ]));
                let raw_op = ChunkedKernelDesignOperator::new(
                    shared_data.clone(),
                    Arc::new(centers.clone()),
                    make_kernel(),
                    Some(raw_gauge),
                    Some(Arc::new(poly_block.clone())),
                )
                .map_err(BasisError::InvalidInput)?;
                let ones = Array1::<f64>::ones(raw_op.nrows());
                let raw_gram = raw_op.diag_xtw_x(&ones).map_err(BasisError::InvalidInput)?;
                let kernel_cols = kernel_transform.ncols();
                let design_gram = symmetrize_penalty(
                    &raw_gram.slice(s![..kernel_cols, ..kernel_cols]).to_owned(),
                );
                let omega_constrained = duchon_constrained_bending_penalty(
                    centers.view(),
                    spec.length_scale,
                    spec.power,
                    effective_nullspace_order,
                    aniso.as_deref(),
                    &kernel_transform,
                )?;
                let (v, _mu) =
                    thin_plate_radial_reparam_data_metric(&omega_constrained, &design_gram)?;
                if v.ncols() > 0 {
                    kernel_transform = fast_ab(&kernel_transform, &v);
                    frozen_radial_reparam = Some(v);
                }
            }
            let kernel_gauge = Arc::new(gam_problem::Gauge::from_block_transforms(&[
                kernel_transform.clone(),
            ]));
            let base_op = ChunkedKernelDesignOperator::new(
                shared_data,
                Arc::new(centers.clone()),
                make_kernel(),
                Some(kernel_gauge),
                Some(Arc::new(poly_block)),
            )
            .map_err(BasisError::InvalidInput)?;
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(
                base_op,
            )))
        };
        let identifiability_transform = spatial_identifiability_transform_from_design_matrix(
            data,
            &base_design,
            &spec.identifiability,
            "Duchon",
        )?;
        let design = if let Some(transform) = identifiability_transform.as_ref() {
            wrap_dense_design_with_transform(base_design, transform, "Duchon")?
        } else {
            base_design
        };
        (design, identifiability_transform)
    } else {
        // #1355: dense path applies the data-metric radial reparameterization
        // `V` (mirroring the thin-plate Wood-TPRS reparam) so the native
        // penalty's cliff-less Mercer spectrum is replaced by the
        // curvature-per-unit-data-variance spectrum (mgcv's cliff), removing the
        // REML over-smoothing collapse to EDF = 1. `V` is frozen at the cold
        // build and replayed verbatim from `spec.radial_reparam` on the
        // predict / κ-trial paths.
        // A FRESH `V` is computed only when no frozen reparam was supplied
        // (`frozen_radial_reparam` already folded above on the replay paths). At
        // that point `kernel_transform` is still the raw `Z`.
        //
        // The reparam is adopted for EVERY configuration, including the default
        // all-on Hilbert scale (mass+tension active). The frozen `V` is threaded
        // into the operator collocation builder (`duchon_operator_penalty_candidates`
        // → `build_duchon_collocation_operator_matriceswithworkspace`) so the
        // mass/tension blocks are assembled directly in the same `K·Z·V` frame as
        // the design and the native `Primary` penalty — no design↔penalty desync.
        // Skipping the reparam whenever operators were active (the old gate) left
        // the default Duchon on the raw cliff-less Mercer spectrum, so REML
        // over-selected EDF (a single 2-D bump fit to EDF≈30/49), which in turn
        // made the fit a knife-edge unstable to ulp-level covariate rotation and
        // unable to collapse toward the null on an irrelevant covariate. Restoring
        // the cliff for the default is what makes those recoveries hold.
        // When the fresh data-metric reparam is computed, its `raw` (un-rotated)
        // design is built here from a full `n×k` kernel evaluation. That SAME
        // realized design is the base of the final basis — rotating it by the
        // adopted `V` gives the fit-time design without a second kernel pass —
        // so carry it forward instead of rebuilding it below (#1718). This
        // halves the cold-build kernel work for explicit native-only Duchon
        // configurations (`all_disabled()`, no frozen reparam), closing their
        // wall-time gap to `thinplate(x, z)` without changing default terms.
        let mut prebuilt_raw_basis: Option<Array2<f64>> = None;
        if frozen_radial_reparam.is_none() {
            let kernel_cols = kernel_transform.ncols();
            if kernel_cols > 0 {
                // Build the un-rotated constrained kernel design once, take its
                // realized Gram `G_c = (K·Z)ᵀ(K·Z)`, and solve the generalized
                // eigenproblem `Ω_c v = μ G_c v` with `Ω_c = α²·ZᵀK_CC Z`.
                let raw = build_duchon_basis_designwithworkspace(
                    data,
                    centers.view(),
                    spec.length_scale,
                    spec.power,
                    effective_nullspace_order,
                    aniso.as_deref(),
                    None,
                    workspace,
                )?;
                let kernel_block = raw.basis.slice(s![.., 0..kernel_cols]);
                // Canonical row order so the realized Gram (and the near-degenerate
                // reparam it feeds) is bit-identical under a pure row permutation (#1378).
                let design_gram = data_metric_design_gram(kernel_block);
                let omega_constrained = duchon_constrained_bending_penalty(
                    centers.view(),
                    spec.length_scale,
                    spec.power,
                    effective_nullspace_order,
                    aniso.as_deref(),
                    &kernel_transform,
                )?;
                let (v, _mu) =
                    thin_plate_radial_reparam_data_metric(&omega_constrained, &design_gram)?;
                // A degenerate reparam (no retained modes) would gut the basis;
                // only adopt `V` when it preserves at least one radial column.
                if v.ncols() > 0 {
                    // The fit-time design is `[K·Z·V | P] = [(K·Z)·V | P]`,
                    // where `K·Z` and `P` are exactly the kernel/poly blocks of
                    // `raw` (the reparam only right-multiplies the constrained
                    // kernel columns; the poly block is reparam-independent). So
                    // rotate `raw`'s kernel block by `V` in place rather than
                    // re-evaluating the kernel: `(K·Z)·V = K·(Z·V)`, the same
                    // model space the un-fused rebuild would produce.
                    let rotated_kernel = fast_ab(&raw.basis.slice(s![.., 0..kernel_cols]), &v);
                    let poly_block = raw.basis.slice(s![.., kernel_cols..]);
                    let mut fused = Array2::<f64>::zeros((
                        raw.basis.nrows(),
                        rotated_kernel.ncols() + poly_block.ncols(),
                    ));
                    fused
                        .slice_mut(s![.., 0..rotated_kernel.ncols()])
                        .assign(&rotated_kernel);
                    if poly_block.ncols() > 0 {
                        fused
                            .slice_mut(s![.., rotated_kernel.ncols()..])
                            .assign(&poly_block);
                    }
                    prebuilt_raw_basis = Some(fused);
                    kernel_transform = fast_ab(&kernel_transform, &v);
                    frozen_radial_reparam = Some(v);
                } else {
                    // No reparam adopted: `raw` already IS the fit-time design
                    // (no rotation), so reuse it directly.
                    prebuilt_raw_basis = Some(raw.basis);
                }
            }
        }
        let basis = if let Some(basis) = prebuilt_raw_basis {
            basis
        } else {
            build_duchon_basis_designwithworkspace(
                data,
                centers.view(),
                spec.length_scale,
                spec.power,
                effective_nullspace_order,
                aniso.as_deref(),
                frozen_radial_reparam.as_ref(),
                workspace,
            )?
            .basis
        };
        let identifiability_transform = spatial_identifiability_transform_from_design(
            data,
            basis.view(),
            &spec.identifiability,
            "Duchon",
        )?;
        let design = if let Some(z) = identifiability_transform.as_ref() {
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(fast_ab(
                &basis, z,
            )))
        } else {
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(basis))
        };
        (design, identifiability_transform)
    };
    // The Duchon penalty is a HILBERT SCALE of pure function-penalties, each a
    // plain block with its own REML λ (REML deselects what the data don't need):
    //   * curvature = the EXACT RKHS reproducing-norm Gram (`Primary`), `n`-free;
    //   * trend     = the affine null-space slope ridge (`DoublePenaltyNullspace`);
    //   * tension `Σ‖∇f‖²` + mass `Σ(f−f̄)²` = collocated on a density-blind `O(k)`
    //     farthest-point sample of the data support (their continuous integrals
    //     diverge for the polyharmonic kernel, so the support quadrature *is* the
    //     penalty — `O(k)`-in-`n`, not the old sparse-center collocation that
    //     under-resolved the basis and exploded).
    let operator_collocation_points = {
        let any_operator = matches!(
            spec.operator_penalties.mass,
            OperatorPenaltySpec::Active { .. }
        ) || matches!(
            spec.operator_penalties.tension,
            OperatorPenaltySpec::Active { .. }
        ) || matches!(
            spec.operator_penalties.stiffness,
            OperatorPenaltySpec::Active { .. }
        );
        if any_operator {
            let m = (DUCHON_COLLOCATION_OVERSAMPLE * centers.nrows()).min(data.nrows());
            Some(select_thin_plate_knots(data, m)?)
        } else {
            None
        }
    };
    let mut candidates = duchon_native_penalty_candidates(
        centers.view(),
        spec.length_scale,
        spec.power,
        effective_nullspace_order,
        aniso.as_deref(),
        &kernel_transform,
        identifiability_transform.as_ref(),
        poly_cols,
    )?;
    if let Some(points) = operator_collocation_points.as_ref() {
        candidates.extend(duchon_operator_penalty_candidates(
            points.view(),
            centers.view(),
            &spec.operator_penalties,
            spec.length_scale,
            spec.power,
            effective_nullspace_order,
            aniso.is_some(),
            identifiability_transform.as_ref(),
            frozen_radial_reparam.as_ref(),
            workspace,
        )?);
    }
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
            periodic: spec.periodic.clone(),
            power: spec.power,
            nullspace_order: effective_nullspace_order,
            identifiability_transform,
            input_scales: None,
            aniso_log_scales: aniso,
            operator_collocation_points,
            radial_reparam: frozen_radial_reparam,
        },
        kronecker_factored: None,
    })
}

/// Rebuild the Duchon penalty list at a NEW `length_scale` purely from FROZEN
/// basis geometry — no data rows touched (#1033, n-free per-ψ penalty re-key).
///
/// The κ-loop fast path skips the n-row `reset_surface`, so it needs `S(ψ_new)`
/// reconstructed exactly and `n`-free at each trial length-scale. This mirrors
/// the cold penalty assembly (`build_duchon_basis_uncached` lines ~345-396)
/// EXACTLY, but every input is taken from the already-frozen
/// `BasisMetadata::Duchon` (centers, identifiability transform, operator
/// collocation points) plus the spec's `(power, nullspace_order,
/// aniso_log_scales, operator_penalties)`. The only thing that moves is
/// `length_scale`.
///
/// The polynomial-column count is `C(d + r, r)` — a pure function of `(d, r)` —
/// so it is recomputed from the centers (`polynomial_block_from_order(centers,
/// order).ncols()`), which equals the cold build's `polynomial_block_from_order(
/// data, order).ncols()` because `.ncols()` does not depend on the row count.
///
/// Returns the per-block penalty matrices (term-local frame, same order/count
/// the cold build emits) and the active per-block nullspace dims — exactly the
/// objects the cold build feeds into `filter_active_penalty_candidates_with_ops`.
pub fn duchon_penalties_at_length_scale(
    centers: ArrayView2<'_, f64>,
    identifiability_transform: Option<&Array2<f64>>,
    operator_collocation_points: Option<ArrayView2<'_, f64>>,
    operator_penalties: &DuchonOperatorPenaltySpec,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    radial_reparam: Option<&Array2<f64>>,
    length_scale: Option<f64>,
    workspace: &mut BasisWorkspace,
) -> Result<(Vec<Array2<f64>>, Vec<usize>), BasisError> {
    // Recompute the effective order + auto-seeded anisotropy exactly as the cold
    // build does (duchon_thinplate.rs:151/159). Both are pure functions of the
    // frozen centers + spec, so the κ trial replays the SAME structural choices.
    let effective_nullspace_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let aniso = auto_seed_aniso_contrasts(centers, aniso_log_scales);
    // n-free kernel-constraint nullspace (from centers; cached on the workspace).
    let mut kernel_transform =
        kernel_constraint_nullspace(centers, effective_nullspace_order, &mut workspace.cache)?;
    // #1355: fold the frozen data-metric reparam `Z' = Z·V` so the κ-trial
    // penalty `Z'ᵀ K_CC(ψ) Z' = diag(μ(ψ))` matches the rotated design.
    if let Some(v) = radial_reparam {
        if v.nrows() != kernel_transform.ncols() {
            crate::bail_dim_basis!(
                "Duchon frozen radial reparam shape {:?} does not match constrained kernel dimension {}",
                v.dim(),
                kernel_transform.ncols()
            );
        }
        kernel_transform = fast_ab(&kernel_transform, v);
    }
    // Polynomial column count: `C(d+r, r)`, independent of the row count, so the
    // n-free centers-based form equals the cold build's data-based `.ncols()`.
    let poly_cols = polynomial_block_from_order(centers, effective_nullspace_order).ncols();
    let mut candidates = duchon_native_penalty_candidates(
        centers,
        length_scale,
        power,
        effective_nullspace_order,
        aniso.as_deref(),
        &kernel_transform,
        identifiability_transform,
        poly_cols,
    )?;
    if let Some(points) = operator_collocation_points {
        candidates.extend(duchon_operator_penalty_candidates(
            points,
            centers,
            operator_penalties,
            length_scale,
            power,
            effective_nullspace_order,
            aniso.is_some(),
            identifiability_transform,
            radial_reparam,
            workspace,
        )?);
    }
    let (penalties, nullspace_dims, _info, _eig, _ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    Ok((penalties, nullspace_dims))
}

/// Materialise the polynomial null-space block for a Duchon basis.
///
/// Returns an `(n, C(d+r, r))` matrix whose columns are all monomials of total
/// degree `≤ r` evaluated at `points`, where `r` is the degree implied by
/// `order` and `d = points.ncols()`.
///
/// | `order`        | columns        | content                      |
/// |----------------|----------------|------------------------------|
/// | `Zero`         | 1              | constant `1`                 |
/// | `Linear`       | `d + 1`        | `[1, x₁, …, x_d]`           |
/// | `Degree(k)`    | `C(d+k, k)`   | all monomials ≤ degree `k`   |
///
/// **Role in basis construction:**
/// At *centers*, this block forms the side-condition matrix `Q` whose null
/// space `null(Q^T)` is the kernel reparameterisation transform `Z`.  At
/// *data rows*, the same block is appended as explicit unpenalized columns so
/// the smooth can represent low-degree polynomial trends.  The column count
/// equals `C(d + r, r)` by the stars-and-bars identity.
pub(crate) fn polynomial_block_from_order(
    points: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> Array2<f64> {
    let n = points.nrows();
    let d = points.ncols();
    match order {
        DuchonNullspaceOrder::Zero => Array2::<f64>::ones((n, 1)),
        DuchonNullspaceOrder::Linear => {
            let mut poly = Array2::<f64>::zeros((n, d + 1));
            poly.column_mut(0).fill(1.0);
            for c in 0..d {
                poly.column_mut(c + 1).assign(&points.column(c));
            }
            poly
        }
        DuchonNullspaceOrder::Degree(degree) => monomial_basis_block(points, degree),
    }
}

/// Range-floor the reparam'd Duchon Primary curvature block so its numerical
/// null space is exactly the polynomial null space, not inflated by the
/// ill-conditioned kernel Gram's low-curvature tail.
///
/// The default duchon adopts the data-metric radial reparam `V`, so the Primary
/// penalty kernel block is `Vᵀ Ω_c V` — diagonal in the `μ` (generalized
/// curvature) eigenvalues. The Duchon polyharmonic Gram is extremely
/// ill-conditioned (cond ≫ 1e10 at k=20), so most `μ` fall far below the
/// numerical-rank cutoff `spectral_tolerance = nrows·1e-10·λmax` that
/// [`analyze_penalty_block`] uses to partition range vs null. Those genuine
/// low-curvature directions are then mis-classified as UNPENALIZED null:
/// retained in the design (they clear the SEPARATE `k·ε` design-support floor)
/// but shrinkable by NO `λ`, so the smooth cannot collapse toward the null on an
/// irrelevant covariate (measured `nulldim = 19` vs the affine `{1,x} = 2`
/// expected on the gam#1815 null-recovery fixture) and REML over-selects EDF.
///
/// Lift the smallest eigenvalues to a relative floor one decade above that
/// cutoff (`nrows·1e-9·λmax`, with `nrows` the EMBEDDED penalty dimension
/// kernel+poly, so the floor clears the tolerance the assembled block is scored
/// against) so every retained mode is a genuine — if weak — penalized `Range`
/// direction; REML's `λ→∞` tail then collapses them. The floor is far below the
/// statistical scale and lifts only the lowest-curvature (near-linear) modes, so
/// signal recovery (e.g. the sin8 centers=50 escape) is unchanged — the
/// high-curvature signal modes sit orders of magnitude above the floor.
pub(crate) fn duchon_range_floor_curvature(
    omega: &Array2<f64>,
    embedded_penalty_dim: usize,
) -> Result<Array2<f64>, BasisError> {
    let n = omega.nrows();
    if n == 0 {
        return Ok(omega.clone());
    }
    let sym = symmetrize_penalty(omega);
    let (mut evals, evecs) = FaerEigh::eigh(&sym, Side::Lower).map_err(BasisError::LinalgError)?;
    let lam_max = evals.iter().copied().fold(0.0_f64, |a, v| a.max(v.abs()));
    if !lam_max.is_finite() || lam_max <= 0.0 {
        return Ok(sym);
    }
    // Two decades above `analyze_penalty_block`'s `nrows·1e-10·λmax` cutoff, so
    // the margin survives the later identifiability congruence `Tᵀ(·)T` and the
    // Frobenius renormalization before the block's rank is scored. Still far
    // below the statistical scale (`≤ nrows·1e-8·λmax`).
    let floor = (embedded_penalty_dim.max(n) as f64) * 1e-8 * lam_max;
    let mut floored = false;
    for v in evals.iter_mut() {
        if v.is_finite() && *v < floor {
            *v = floor;
            floored = true;
        }
    }
    if !floored {
        return Ok(sym);
    }
    // Reconstruct `U diag(evals) Uᵀ` with the floored spectrum.
    let mut out = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let lam = evals[j];
        for a in 0..n {
            let ua = evecs[[a, j]];
            if ua == 0.0 {
                continue;
            }
            for b in 0..n {
                out[[a, b]] += ua * lam * evecs[[b, j]];
            }
        }
    }
    Ok(symmetrize_penalty(&out))
}

pub fn monomial_exponents(dimension: usize, max_total_degree: usize) -> Vec<Vec<usize>> {
    fn recurse(
        axis: usize,
        remaining_degree: usize,
        current: &mut [usize],
        out: &mut Vec<Vec<usize>>,
    ) {
        if axis + 1 == current.len() {
            current[axis] = remaining_degree;
            out.push(current.to_vec());
            return;
        }
        for exponent in (0..=remaining_degree).rev() {
            current[axis] = exponent;
            recurse(axis + 1, remaining_degree - exponent, current, out);
        }
    }

    if dimension == 0 {
        return vec![Vec::new()];
    }

    let mut out = Vec::new();
    let mut current = vec![0usize; dimension];
    for total_degree in 0..=max_total_degree {
        recurse(0, total_degree, &mut current, &mut out);
    }
    out
}

pub fn duchon_nullspace_dimension(dimension: usize, max_total_degree: usize) -> usize {
    monomial_exponents(dimension, max_total_degree).len()
}

pub(crate) fn monomial_basis_block(
    points: ArrayView2<'_, f64>,
    max_total_degree: usize,
) -> Array2<f64> {
    let n = points.nrows();
    let exponents = monomial_exponents(points.ncols(), max_total_degree);
    let mut block = Array2::<f64>::zeros((n, exponents.len()));
    for (col, exponents) in exponents.iter().enumerate() {
        for row in 0..n {
            let mut value = 1.0;
            for axis in 0..points.ncols() {
                let exponent = exponents[axis];
                if exponent != 0 {
                    value *= points[[row, axis]].powi(exponent as i32);
                }
            }
            block[[row, col]] = value;
        }
    }
    block
}

#[inline(always)]
pub(crate) fn thin_plate_polynomial_degree(dimension: usize) -> usize {
    thin_plate_penalty_order(dimension).saturating_sub(1)
}

pub(crate) fn thin_plate_polynomial_block(points: ArrayView2<'_, f64>) -> Array2<f64> {
    monomial_basis_block(points, thin_plate_polynomial_degree(points.ncols()))
}

pub fn thin_plate_polynomial_basis_dimension(dimension: usize) -> usize {
    monomial_exponents(dimension, thin_plate_polynomial_degree(dimension)).len()
}

/// Row-order-canonical realized design Gram `symmetrize(KᵀK)` for the data-metric
/// radial reparam (#1347/#1355).
///
/// `KᵀK = Σ_row (row)ᵀ(row)` is mathematically invariant to a pure row
/// permutation of the training data, but `fast_atb` accumulates the outer
/// products in the kernel block's stored (data) row order, so floating-point
/// non-associativity lets a reordering perturb the Gram by an ulp. The reparam
/// eigendecomposition fed by this Gram is near-degenerate (the thin-plate radial
/// spectrum has a long low-curvature tail), so that ulp rotates its eigenvectors
/// and makes the fitted `s(x, bs="tp")` basis — and hence the curve — depend on
/// row order. That is the residual ~2e-7 row-permutation drift owed under
/// gam#1378 that survives the value-anchored knot set and centroid seed (the
/// local `bs="cr"/"ps"` bases never form this data-metric radial Gram, so they
/// stayed bit-stable). Summing the rows in a canonical lexicographic (`total_cmp`)
/// order gives the identical addition sequence for every permutation of the same
/// unordered row set — the rows are a pure function of the data and genuinely
/// equal rows contribute equal, order-free terms — so the Gram, its
/// eigendecomposition, and the reparam become bit-identical across row order.
fn data_metric_design_gram(kernel_block: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = kernel_block.nrows();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        for c in 0..kernel_block.ncols() {
            match kernel_block[[a, c]].total_cmp(&kernel_block[[b, c]]) {
                std::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    });
    let sorted = kernel_block.select(Axis(0), &order);
    symmetrize_penalty(&fast_atb(&sorted, &sorted))
}

/// Selects which radial penalty eigenmodes to expose as basis columns.
///
/// The constrained radial penalty `Ω` is SPD in exact arithmetic — the
/// polynomial null space `{1, x, …}` has already been removed by the gauge
/// restriction, so every nonzero eigenvalue is a genuine bending direction
/// and must be retained (this matches mgcv's thin-plate construction, which
/// keeps all `k − M` radial modes and relies on REML, not basis truncation,
/// to set the effective degrees of freedom). The only modes that are NOT
/// real curvature directions are **roundoff dust**: eigenvalues at or below
/// the LAPACK numerical-rank floor `K·ε·λ_max` (Golub & Van Loan, *Matrix
/// Computations*, §2.5.6) are exact zeros polluted by floating-point error
/// from the constraint restriction and carry no information.
///
/// The threshold is therefore the standard numerical-rank floor — derived,
/// scale-free, and tuning-free. It deliberately does NOT prune low-but-real
/// bending modes by magnitude: doing so was the #1271 hill-climb (a swept
/// `max_eval·tol` cutoff) that over-pruned the nonlinear arms (lidar /
/// by-factor truth recovery collapsed) while still missing the linear EDF
/// bar. The genuine over-fit on near-linear data is a REML smoothing issue
/// (the diagonalised radial penalty's wide eigenvalue spread under a single
/// `λ` leaves a flat REML profile, so the outer optimiser terminates at an
/// interior `λ` that under-smooths), not a basis-rank issue — pruning cannot
/// fix it without destroying the bending capacity real data needs.
fn thin_plate_retained_radial_indices(evals: &Array1<f64>) -> Vec<usize> {
    let k = evals.len();
    if k == 0 {
        return Vec::new();
    }
    let max_eval = evals
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    if !max_eval.is_finite() || max_eval <= 0.0 {
        return Vec::new();
    }
    // Numerical-rank floor: anything at or below `K·ε·λ_max` is roundoff dust
    // from the gauge restriction, not a real bending mode. Everything above it
    // is genuine curvature and is kept.
    let num_floor = (k as f64) * f64::EPSILON * max_eval;
    evals
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value.abs() > num_floor).then_some(idx))
        .collect()
}

pub(crate) fn thin_plate_radial_reparam_from_constrained_penalty(
    omega_constrained: &Array2<f64>,
) -> Result<(Array2<f64>, Array1<f64>), BasisError> {
    let kernel_cols = omega_constrained.nrows();
    if kernel_cols != omega_constrained.ncols() {
        crate::bail_dim_basis!(
            "thin-plate constrained radial penalty must be square: got {:?}",
            omega_constrained.dim()
        );
    }
    if kernel_cols == 0 {
        return Ok((Array2::<f64>::zeros((0, 0)), Array1::<f64>::zeros(0)));
    }
    let sym = symmetrize_penalty(omega_constrained);
    let (mut evals, evecs) = FaerEigh::eigh(&sym, Side::Lower).map_err(BasisError::LinalgError)?;
    for value in evals.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
    let keep = thin_plate_retained_radial_indices(&evals);
    Ok((evecs.select(Axis(1), &keep), evals.select(Axis(0), &keep)))
}

/// Thin-plate radial reparameterization in the **realized data metric** (#1347).
///
/// The penalty is the polyharmonic bending energy `Ω_c = Zᵀ K_CC Z` (the RKHS
/// reproducing-norm Gram on the constrained kernel coefficients). gam's old
/// reparam eigendecomposed `Ω_c` alone, laying its raw eigenvalues on the
/// penalty diagonal. But the constraint `Z` has already quotiented out the
/// `{1, x}` polynomial null space, so `Ω_c` is full-rank with a smooth Mercer
/// tail and **no cliff** — its smallest eigenvalues are genuine low-curvature
/// bending directions that nonetheless carry large variance over the data.
/// Under a single REML `λ` those near-null modes cost almost nothing yet absorb
/// EDF freely, over-fitting near-linear data (mean EDF ≈ 5.3 vs mgcv ≈ 2.1).
///
/// mgcv's TPRS instead penalizes bending energy **relative to the realized
/// design metric** — equivalently it solves the generalized eigenproblem
///
/// ```text
///   Ω_c v = μ G_c v ,   G_c = (K Z)ᵀ (K Z)
/// ```
///
/// where `G_c` is the Gram of the realized constrained kernel design columns.
/// The eigenvalue `μ = (vᵀ Ω_c v)/(vᵀ G_c v)` is curvature per unit
/// data-variance: it spreads the spectrum the way mgcv's does (top mode, a
/// `0.77` second mode, then a clean geometric cliff to the tail), so a single
/// `λ` can no longer buy near-free wiggle. The returned eigenvectors `V` are
/// `G_c`-orthonormal (`Vᵀ G_c V = I`), so the rotated design `K Z V` has an
/// identity Gram and the penalty is exactly `diag(μ) = Vᵀ Ω_c V` — which the
/// frozen-replay / length-scale paths already recover via `diag(Vᵀ Ω_c V)`, so
/// no downstream change is needed. The model space `span(K Z V) = span(K Z)` is
/// unchanged (`V` is invertible), preserving full nonlinear capacity.
pub(crate) fn thin_plate_radial_reparam_data_metric(
    omega_constrained: &Array2<f64>,
    design_gram: &Array2<f64>,
) -> Result<(Array2<f64>, Array1<f64>), BasisError> {
    let k = omega_constrained.nrows();
    if k != omega_constrained.ncols() || design_gram.nrows() != k || design_gram.ncols() != k {
        crate::bail_dim_basis!(
            "thin-plate data-metric reparam requires square k×k Ω_c and G_c: Ω_c={:?}, G_c={:?}",
            omega_constrained.dim(),
            design_gram.dim()
        );
    }
    if k == 0 {
        return Ok((Array2::<f64>::zeros((0, 0)), Array1::<f64>::zeros(0)));
    }
    // Whiten by G_c: G_c = U_g D_g U_gᵀ ; W = U_g D_g^{-1/2} (drop near-null G_c
    // directions, which are design columns with no realized data support).
    let g_sym = symmetrize_penalty(design_gram);
    let (g_evals, g_evecs) =
        FaerEigh::eigh(&g_sym, Side::Lower).map_err(BasisError::LinalgError)?;
    let gmax = g_evals.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs()));
    if !gmax.is_finite() || gmax <= 0.0 {
        // Degenerate design Gram: fall back to the plain bending eigenbasis.
        return thin_plate_radial_reparam_from_constrained_penalty(omega_constrained);
    }
    let g_floor = (k as f64) * f64::EPSILON * gmax;
    let mut cols: Vec<usize> = Vec::with_capacity(k);
    for j in 0..k {
        if g_evals[j] > g_floor {
            cols.push(j);
        }
    }
    let m = cols.len();
    if m == 0 {
        return thin_plate_radial_reparam_from_constrained_penalty(omega_constrained);
    }
    let mut w = Array2::<f64>::zeros((k, m));
    for (c, &j) in cols.iter().enumerate() {
        let inv_sqrt = 1.0 / g_evals[j].sqrt();
        for i in 0..k {
            w[[i, c]] = g_evecs[[i, j]] * inv_sqrt;
        }
    }
    // M = Wᵀ Ω_c W (m×m), eig(M) = (μ, P). Generalized eigenvectors V = W P.
    let omega_sym = symmetrize_penalty(omega_constrained);
    let wt_omega = fast_atb(&w, &omega_sym);
    let m_mat = symmetrize_penalty(&fast_ab(&wt_omega, &w));
    let (mut mu, p_mat) = FaerEigh::eigh(&m_mat, Side::Lower).map_err(BasisError::LinalgError)?;
    for value in mu.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
    let v_full = fast_ab(&w, &p_mat); // k×m, G_c-orthonormal columns
    let keep = thin_plate_retained_radial_indices(&mu);
    Ok((v_full.select(Axis(1), &keep), mu.select(Axis(0), &keep)))
}

pub(crate) fn thin_plate_radial_reparam_from_centers(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    kernel_transform: &Array2<f64>,
) -> Result<(Array2<f64>, Array1<f64>), BasisError> {
    let k = centers.nrows();
    let d = centers.ncols();
    let mut omega = Array2::<f64>::zeros((k, k));
    let length_scale_sq = length_scale * length_scale;
    fill_symmetric_from_row_kernel(&mut omega, |i, j| {
        let mut dist2 = 0.0;
        for c in 0..d {
            let delta = centers[[i, c]] - centers[[j, c]];
            dist2 += delta * delta;
        }
        thin_plate_kernel_from_dist2(dist2 / length_scale_sq, d)
    })?;
    let kernel_gauge = gam_problem::Gauge::from_block_transforms(&[kernel_transform.clone()]);
    let omega_constrained = symmetrize_penalty(&kernel_gauge.restrict_penalty(&omega));
    thin_plate_radial_reparam_from_constrained_penalty(&omega_constrained)
}

pub(crate) fn kernel_constraint_nullspace_from_matrix(
    constraint_matrix: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, BasisError> {
    let k = constraint_matrix.nrows();
    let q = constraint_matrix.ncols();
    if q == 0 {
        return Ok(Array2::<f64>::eye(k));
    }
    // Constraint system Q^T alpha = 0. The trailing columns of the orthogonal
    // factor in a column-pivoted QR of Q span null(Q^T).
    let (z, _) = rrqr_nullspace_basis(&constraint_matrix, default_rrqr_rank_alpha())
        .map_err(BasisError::LinalgError)?;
    Ok(z)
}

/// Relative tolerance (against the data's squared radius) below which two
/// farthest-point candidates' maximin — or centroid — distances are treated as
/// *tied* and resolved by the rotation/permutation-invariant support-distance
/// profile rather than by their exact floating-point ordering.
///
/// A generic (non-90°) rigid rotation of the covariates re-expresses every
/// coordinate with ~1 ulp of round-off, so the squared distances that drive the
/// farthest-point recursion differ from their exact rotation-invariant values by
/// ~`ε·‖x‖²`. This tolerance is set several orders of magnitude above that
/// round-off floor yet far below any genuine gap between geometrically-distinct
/// candidates, so it absorbs the sub-ulp perturbation without altering the
/// selection on data whose maximin values are genuinely separated.
const KNOT_MAXIMIN_TIE_REL_TOL: f64 = 1e-9;

/// Deterministically selects thin-plate knots via farthest-point sampling.
///
/// This produces a space-filling subset without introducing RNG/state coupling.
pub fn select_thin_plate_knots(
    data: ArrayView2<f64>,
    num_knots: usize,
) -> Result<Array2<f64>, BasisError> {
    let n = data.nrows();
    let d = data.ncols();
    if d == 0 {
        crate::bail_invalid_basis!("thin-plate spline requires at least one covariate dimension");
    }
    if n == 0 {
        crate::bail_invalid_basis!("cannot select thin-plate knots from empty data");
    }
    if data.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("thin-plate spline knot selection requires finite data");
    }
    if num_knots == 0 {
        crate::bail_invalid_basis!("thin-plate spline knot count must be positive");
    }
    if num_knots > n {
        crate::bail_invalid_basis!(
            "requested {} knots but only {} rows are available",
            num_knots,
            n
        );
    }

    // Rotation-equivariant maximin seed. The greedy farthest-point recursion
    // below uses ONLY Euclidean distances, which are invariant under any rigid
    // rotation of the covariates, so the only frame-dependent ingredients of
    // the selected knot set are the seed point and the tie-break. A thin-plate
    // spline is mathematically *exactly* rotation-invariant — its `r^{2m-d}`
    // (log r) kernel depends only on the pairwise distance `r`, and its
    // polynomial null space `span{1, x, …}` is mapped onto itself by any
    // orthogonal map — so rotating the data must leave the fitted surface
    // unchanged, which requires the knot SET to be rotation-invariant. The old
    // lexicographically-smallest-coordinate seed broke exactly this: a rigid
    // rotation changes which row is "lexicographically smallest", reseeding the
    // recursion at a different physical point and selecting a genuinely
    // different knot set — a 90° rotation about the centroid drifted the
    // default `thinplate(x, z)` surface by ~2% of its range while a pure row
    // permutation was bit-stable.
    //
    // Seed at the row nearest the data centroid instead. The centroid is
    // rotation-EQUIVARIANT (it rotates rigidly with the data) and the
    // nearest-row test is a Euclidean distance, so the SAME physical row is
    // chosen in every rotated frame; both are pure functions of the unordered
    // value set, so the seed also stays row-permutation invariant (gam#1378).
    //
    // The column sum is taken in CANONICAL (value-sorted) order rather than row
    // order. A plain `for i in 0..n { s += data[[i, c]] }` accumulates in the
    // data's ROW order, so floating-point round-off makes the result depend on
    // that order: a pure row permutation re-sequences the additions and shifts
    // the mean by an ulp. That ulp is enough to break the EXACT equidistance of
    // points that are symmetric about the mean (the common 1-D case), so the
    // `dist2_to_centroid` comparisons below stop reducing to the
    // value-lexicographic tie-break and the seed — and hence the whole knot set
    // — flips with row order. That is the residual ~1e-7 `s(x, bs="tp")`
    // row-permutation drift owed under gam#1378 (value-anchored `bs="cr"/"ps"`
    // stayed bit-stable because they never seed off this centroid). Sorting the
    // column values yields the identical addition sequence for every permutation
    // of the same data — all values are finite (guarded above), so `total_cmp`
    // is a total order — restoring a bit-identical, order-independent centroid.
    let centroid: Vec<f64> = (0..d)
        .map(|c| {
            let mut col: Vec<f64> = (0..n).map(|i| data[[i, c]]).collect();
            col.sort_by(|a, b| a.total_cmp(b));
            let s: f64 = col.iter().sum();
            s / n as f64
        })
        .collect();
    let dist2_to_centroid: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut d2 = 0.0;
            for c in 0..d {
                let delta = data[[i, c]] - centroid[c];
                d2 += delta * delta;
            }
            d2
        })
        .collect();

    // Rotation- and permutation-invariant tie-break on a candidate's distance
    // profile to the whole support.  Lexicographic coordinate order is
    // permutation-invariant, but it is NOT rotation-invariant: on symmetric
    // clouds (regular grids, rings, centred designs) the centroid/fill-distance
    // keys often tie exactly, and a rigid rotation can change which coordinate
    // tuple is lexicographically smallest.  That reseeds the farthest-point
    // recursion with a different physical row and breaks the isotropic Duchon /
    // thin-plate equivariance contract.  The sorted multiset
    // `{‖x_i - x_l‖² : l=1..n}` is a pure function of the unordered Euclidean
    // geometry, so it survives both row permutations and rigid rotations.  Only
    // genuinely duplicate/interchangeable rows fall through to the row index.
    let distance_profile_less = |i: usize, j: usize| -> bool {
        let mut profile_i = Vec::with_capacity(n);
        let mut profile_j = Vec::with_capacity(n);
        for row in 0..n {
            let mut d2_i = 0.0;
            let mut d2_j = 0.0;
            for c in 0..d {
                let delta_i = data[[i, c]] - data[[row, c]];
                let delta_j = data[[j, c]] - data[[row, c]];
                d2_i += delta_i * delta_i;
                d2_j += delta_j * delta_j;
            }
            profile_i.push(d2_i);
            profile_j.push(d2_j);
        }
        profile_i.sort_by(|a, b| a.total_cmp(b));
        profile_j.sort_by(|a, b| a.total_cmp(b));
        for (&di, &dj) in profile_i.iter().zip(profile_j.iter()) {
            match di.total_cmp(&dj) {
                std::cmp::Ordering::Less => return true,
                std::cmp::Ordering::Greater => return false,
                std::cmp::Ordering::Equal => {}
            }
        }
        i < j
    };

    // Round-off-robust tie tolerance (#1818). The data's squared radius sets the
    // scale of the maximin/centroid distances; a generic rigid rotation perturbs
    // each of them by ~`ε·radius²`, so exact-equality tie-break gates let that
    // round-off — rather than the intended rotation-invariant key — decide
    // near-equidistant candidates, and a single flip cascades into a materially
    // different knot set. `tie_tol` sits well above that round-off floor and far
    // below any genuine maximin gap, so near-ties are consistently resolved by
    // the invariant support-distance profile in every rotated frame.
    let knot_scale2 = dist2_to_centroid.iter().copied().fold(0.0_f64, f64::max).max(1.0);
    let tie_tol = KNOT_MAXIMIN_TIE_REL_TOL * knot_scale2;

    // Seed = centroid-nearest row; near-equidistant rows (within `tie_tol`) are
    // resolved by the invariant support-distance profile so the seed is a
    // deterministic, rotation- and permutation-invariant function of the data.
    let seed_min = dist2_to_centroid.iter().copied().fold(f64::INFINITY, f64::min);
    let seed_idx = (0..n)
        .filter(|&i| dist2_to_centroid[i] <= seed_min + tie_tol)
        .reduce(|a, b| if distance_profile_less(a, b) { a } else { b })
        .unwrap_or(0);

    let mut selected = Vec::with_capacity(num_knots);
    let mut chosen = vec![false; n];
    let mut min_dist2 = vec![f64::INFINITY; n];

    selected.push(seed_idx);
    chosen[seed_idx] = true;

    min_dist2.par_iter_mut().enumerate().for_each(|(i, slot)| {
        let mut d2 = 0.0;
        for c in 0..d {
            let delta = data[[i, c]] - data[[seed_idx, c]];
            d2 += delta * delta;
        }
        *slot = d2;
    });
    min_dist2[seed_idx] = 0.0;

    while selected.len() < num_knots {
        // Maximin: take the larger min-distance to the chosen set. Exact
        // `min_dist2` ties — common on regular grids and, under a generic
        // rotation, wherever round-off perturbs two near-equidistant candidates —
        // are resolved by a rotation-invariant key first (the larger distance to
        // the centroid, which spreads knots outward and is a pure function of the
        // unordered value set), and only by the invariant support-distance profile
        // for points that also tie there. Both the maximin and the centroid keys
        // use `tie_tol` (not exact equality) so sub-ulp coordinate perturbation
        // can never decide the selection; this keeps the knot SET invariant under
        // both rigid rotation and row permutation of the data.
        let max_val = min_dist2
            .par_iter()
            .enumerate()
            .filter(|(i, _)| !chosen[*i])
            .map(|(_, &cand)| cand)
            .reduce(|| f64::NEG_INFINITY, f64::max);
        if !max_val.is_finite() {
            break;
        }
        // Candidates within round-off tolerance of the maximin extremum, in
        // canonical (ascending) row order (parallel collect is index-ordered).
        let mut candidates: Vec<usize> = (0..n)
            .into_par_iter()
            .filter(|&i| !chosen[i] && min_dist2[i] >= max_val - tie_tol)
            .collect();
        if candidates.is_empty() {
            break;
        }
        // Secondary invariant key: farthest from the centroid, round-off-robust.
        let cand_max_centroid = candidates
            .iter()
            .map(|&i| dist2_to_centroid[i])
            .fold(f64::NEG_INFINITY, f64::max);
        candidates.retain(|&i| dist2_to_centroid[i] >= cand_max_centroid - tie_tol);
        // Tertiary invariant key: smallest support-distance profile (then row
        // index, for genuinely interchangeable duplicate points).
        let next_idx = candidates
            .into_iter()
            .reduce(|a, b| if distance_profile_less(a, b) { a } else { b })
            .expect("candidate set is non-empty");
        selected.push(next_idx);
        chosen[next_idx] = true;

        min_dist2.par_iter_mut().enumerate().for_each(|(i, slot)| {
            if chosen[i] {
                return;
            }
            let mut d2 = 0.0;
            for c in 0..d {
                let delta = data[[i, c]] - data[[next_idx, c]];
                d2 += delta * delta;
            }
            if d2 < *slot {
                *slot = d2;
            }
        });
    }

    let mut knots = Array2::<f64>::zeros((selected.len(), d));
    for (r, &idx) in selected.iter().enumerate() {
        knots.row_mut(r).assign(&data.row(idx));
    }
    Ok(knots)
}

#[inline(always)]
pub(crate) fn thin_plate_kernel_from_dist2(
    dist2: f64,
    dimension: usize,
) -> Result<f64, BasisError> {
    if !dist2.is_finite() || dist2 < 0.0 {
        crate::bail_invalid_basis!("thin-plate kernel distance must be finite and non-negative");
    }
    if dist2 == 0.0 {
        return Ok(0.0);
    }
    match dimension {
        // For d≤3, the minimum penalty order m=2 (biharmonic) suffices.
        // Hand-optimized closed forms avoid the overhead of the general evaluator.
        //   d=1:  r^3
        //   d=2:  r^2 log(r)
        //   d=3: -r
        1 => Ok(dist2 * dist2.sqrt()),
        2 => Ok(0.5 * dist2 * dist2.ln()),
        3 => Ok(-dist2.sqrt()),
        _ => {
            // General case: choose the smallest penalty order m with 2m > d,
            // i.e. m = floor(d/2) + 1, and evaluate via the Duchon polyharmonic
            // kernel which handles arbitrary (m, d) combinations.
            let m = dimension / 2 + 1;
            let r = dist2.sqrt();
            Ok(polyharmonic_kernel(r, (m) as f64, dimension))
        }
    }
}

#[inline(always)]
pub(crate) fn thin_plate_penalty_order(dimension: usize) -> usize {
    match dimension {
        1..=3 => 2,
        _ => dimension / 2 + 1,
    }
}

/// True when canonical TPS is mathematically infeasible at this (d, k) — the
/// polynomial nullspace P(C) has more columns than centers, so the side
/// constraint `P(C)^T α = 0` is overdetermined and the basis collapses.
#[inline(always)]
pub(crate) fn d_canonical_tps_infeasible(dimension: usize, num_centers: usize) -> bool {
    num_centers < thin_plate_polynomial_basis_dimension(dimension)
}

/// Whether canonical thin-plate splines are infeasible at THESE specific
/// centers — the single governing feasibility test for the auto-promotion gate.
///
/// Canonical TPS requires the polynomial nullspace block `P(C)` (`k × M(d)`) to
/// have full column rank `M(d)`; otherwise the side constraint `P(C)ᵀα = 0` is
/// overdetermined (count-short) or rank-deficient (degenerate geometry), and
/// `thin_plate_kernel_constraint_nullspace` hard-errors. There are two failure
/// modes and rank subsumes both:
///   * too few centers — `k < M(d)` (the cheap count short-circuit, which also
///     avoids materialising an oversized `k × M(d)` block when `M(d)` explodes
///     in high dimension, e.g. `M(16) = 735_471`); and
///   * enough centers but geometrically DEGENERATE — the selected centers are
///     affinely/polynomially dependent, so `rank P(C) < M(d)` even though
///     `k ≥ M(d)` (e.g. coplanar points in 3-D).
///
/// The prior gate tested only the count, so a degenerate-but-sufficient center
/// set slipped past it into canonical TPS and hard-errored instead of promoting
/// to the Duchon generalisation (which handles a rank-deficient nullspace
/// gracefully — it takes the RRQR nullspace at the *actual* rank and downgrades
/// the effective nullspace order). Making rank the test keeps the promotion gate
/// from drifting out of sync with the linear-algebra feasibility the builder
/// enforces downstream.
pub(crate) fn thin_plate_canonical_infeasible_at_centers(centers: ArrayView2<'_, f64>) -> bool {
    let dimension = centers.ncols();
    // Cheap count short-circuit; also guards high `d`, where forming the
    // `k × M(d)` polynomial block is itself intractable.
    if d_canonical_tps_infeasible(dimension, centers.nrows()) {
        return true;
    }
    // Enough centers by count (`M(d) ≤ k`), so the block is small enough to
    // form: check the ACTUAL rank so a degenerate center geometry promotes to
    // Duchon rather than hard-erroring in canonical TPS.
    let poly_block = thin_plate_polynomial_block(centers);
    let poly_cols = poly_block.ncols();
    match rrqr_nullspace_basis(&poly_block, default_rrqr_rank_alpha()) {
        Ok((_, rank)) => rank < poly_cols,
        // If the rank probe itself fails, defer to the canonical path, which
        // surfaces a precise error rather than silently promoting.
        Err(_) => false,
    }
}

/// Pick Duchon parameters for the TPS auto-promotion at infeasible (d, k).
/// Returns `Some((nullspace_order, power))` when a hybrid-Duchon spec exists
/// satisfying the collocation gate `2(p + s) > d + max_op` for max_op = 2
/// (default operator penalties: mass + tension + stiffness). The hybrid
/// kernel (Matern-blended) sidesteps the pure-Duchon `2s < d` gate, leaving
/// only the collocation/spectral-existence condition.
///
/// Strategy: prefer Linear nullspace (M' = d+1) so the polynomial trend
/// retains the affine span; fall back to Zero (M' = 1) when k < d+1. The
/// smallest admissible s in each case gives the most TPS-like behavior
/// (largest spectral roughness for a given polynomial nullspace).
pub(crate) fn duchon_thin_plate_fallback_params(
    dimension: usize,
    num_centers: usize,
) -> Option<(DuchonNullspaceOrder, usize)> {
    let d = dimension;
    let max_op = 2usize; // mass + tension + stiffness collocation
    for (order, p, m_poly) in [
        (DuchonNullspaceOrder::Linear, 2usize, d + 1),
        (DuchonNullspaceOrder::Zero, 1usize, 1usize),
    ] {
        if num_centers < m_poly {
            continue;
        }
        // Smallest integer s with 2(p + s) > d + max_op.
        let target = d + max_op;
        let s_min = if 2 * p > target {
            0
        } else {
            (target - 2 * p) / 2 + 1
        };
        return Some((order, s_min));
    }
    None
}

/// Length scale at which the auto-promoted hybrid-Duchon kernel is well
/// conditioned: the typical separation between centers.
///
/// The hybrid spectrum `||w||^(2p)·(kappa²+||w||²)^s` produces real-space
/// partial-fraction coefficients that scale as `length_scale^(2(p+s-n))`
/// (`duchon_partial_fraction_coeffs`). To keep every block O(1), `kappa·r`
/// must be O(1) at the center separations the kernel actually evaluates on,
/// i.e. `length_scale ≈ typical center distance`. We use the geometric mean
/// of the min and max pairwise center distances — robust to a few clustered
/// or far-flung centers and exactly the scale where the kernel's smooth and
/// Matern-tail parts are both resolved. Falls back to the requested length
/// scale when fewer than two distinct centers exist (no pairwise distance).
pub(crate) fn hybrid_duchon_promotion_length_scale(
    centers: ArrayView2<'_, f64>,
    requested_length_scale: f64,
) -> f64 {
    match pairwise_distance_bounds_sampled(centers) {
        Some((r_min, r_max)) => {
            // Geometric mean keeps the scale between the tightest and widest
            // center pairs; both are positive and finite by construction.
            (r_min * r_max).sqrt()
        }
        None => {
            if requested_length_scale.is_finite() && requested_length_scale > 0.0 {
                requested_length_scale
            } else {
                1.0
            }
        }
    }
}

#[inline(always)]
pub(crate) fn thin_plate_kernel_triplet_from_scaled_distance(
    scaled_distance: f64,
    dimension: usize,
) -> Result<(f64, f64, f64), BasisError> {
    if !scaled_distance.is_finite() || scaled_distance < 0.0 {
        crate::bail_invalid_basis!("thin-plate scaled distance must be finite and non-negative");
    }
    if scaled_distance == 0.0 {
        return Ok((0.0, 0.0, 0.0));
    }

    match dimension {
        1 => {
            let value = scaled_distance.powi(3);
            let first = 3.0 * scaled_distance.powi(2);
            let second = 6.0 * scaled_distance;
            Ok((value, first, second))
        }
        2 => {
            let log_r = scaled_distance.max(1e-300).ln();
            let value = scaled_distance.powi(2) * log_r;
            let first = 2.0 * scaled_distance * log_r + scaled_distance;
            let second = 2.0 * log_r + 3.0;
            Ok((value, first, second))
        }
        3 => Ok((-scaled_distance, -1.0, 0.0)),
        _ => polyharmonic_kernel_triplet(
            scaled_distance,
            thin_plate_penalty_order(dimension) as f64,
            dimension,
        ),
    }
}

#[inline(always)]
pub(crate) fn thin_plate_kernel_psi_triplet_from_distance(
    distance: f64,
    length_scale: f64,
    dimension: usize,
) -> Result<(f64, f64, f64), BasisError> {
    if !distance.is_finite() || distance < 0.0 {
        crate::bail_invalid_basis!("thin-plate kernel distance must be finite and non-negative");
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        crate::bail_invalid_basis!("thin-plate length_scale must be finite and positive");
    }

    // ThinPlate psi-derivative convention:
    // the optimizer uses psi = log(kappa) = -log(length_scale), so the scaled
    // radial argument is
    //   r(psi) = ||x - c|| / length_scale = ||x - c|| * exp(psi).
    //
    // Therefore
    //   dr/dpsi     = r
    //   d²r/dpsi²   = r
    //
    // and for any TPS radial kernel phi(r),
    //   d phi / dpsi       = phi_r(r) * r
    //   d²phi / dpsi²      = phi_rr(r) * r² + phi_r(r) * r.
    //
    // This is exactly the chain rule requested by the math spec, translated to
    // the code's stored inverse-length-scale parameterization.
    let scaled_distance = distance / length_scale;
    let (value, radial_first, radial_second) =
        thin_plate_kernel_triplet_from_scaled_distance(scaled_distance, dimension)?;
    let psi = radial_first * scaled_distance;
    let psi_psi = radial_second * scaled_distance * scaled_distance + psi;
    Ok((value, psi, psi_psi))
}

/// Creates a thin-plate regression spline basis from data and knot locations.
///
/// # Arguments
/// * `data` - `n x d` matrix of evaluation points
/// * `knots` - `k x d` matrix of knot locations
///
/// # Returns
/// `ThinPlateSplineBasis` containing:
/// - `basis`: `n x (k_c + M)` matrix (`[K_c | P]`) where `M` is the TPS
///   polynomial null-space dimension for the selected ambient dimension
/// - `penalty_bending`: constrained TPS curvature penalty
/// - `penalty_ridge`: identity penalty for null-space shrinkage
pub fn create_thin_plate_spline_basis(
    data: ArrayView2<f64>,
    knots: ArrayView2<f64>,
) -> Result<ThinPlateSplineBasis, BasisError> {
    let mut workspace = BasisWorkspace::default();
    create_thin_plate_spline_basiswithworkspace(data, knots, &mut workspace)
}

pub fn create_thin_plate_spline_basiswithworkspace(
    data: ArrayView2<f64>,
    knots: ArrayView2<f64>,
    workspace: &mut BasisWorkspace,
) -> Result<ThinPlateSplineBasis, BasisError> {
    create_thin_plate_spline_basis_scaledwithworkspace(data, knots, 1.0, None, workspace)
}

pub(crate) fn create_thin_plate_spline_basis_scaledwithworkspace(
    data: ArrayView2<f64>,
    knots: ArrayView2<f64>,
    length_scale: f64,
    frozen_radial_reparam: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<ThinPlateSplineBasis, BasisError> {
    let n = data.nrows();
    let k = knots.nrows();
    let d = data.ncols();

    if d == 0 {
        crate::bail_invalid_basis!("thin-plate spline requires at least one covariate dimension");
    }
    if d != knots.ncols() {
        crate::bail_dim_basis!(
            "thin-plate spline dimension mismatch: data has {} columns, knots have {} columns",
            d,
            knots.ncols()
        );
    }
    let poly_cols = thin_plate_polynomial_basis_dimension(d);
    if k < poly_cols {
        crate::bail_invalid_basis!(
            "thin-plate spline requires at least {} knots to span the degree-{} polynomial null space in dimension {}; got {}",
            poly_cols,
            thin_plate_polynomial_degree(d),
            d,
            k
        );
    }
    if data.iter().any(|v| !v.is_finite()) || knots.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("thin-plate spline requires finite data and knot values");
    }
    if !length_scale.is_finite() || length_scale <= 0.0 {
        crate::bail_invalid_basis!("thin-plate length_scale must be finite and positive");
    }

    // Translation-invariant frame (#1269). The thin-plate kernel reads only
    // coordinate *differences* `data − knots`, so it is already invariant to a
    // covariate translation `x → x + c`; the polynomial null-space block
    // `P = {1, x, x², …}` and the side-constraint nullspace `P(knots)ᵀα = 0`,
    // however, are assembled at the *absolute* coordinate. When the covariate is
    // offset (e.g. a centred-vs-raw "year", or this term's standardized axis
    // carrying a large mean), the `{1, x}` columns become near-collinear, the
    // design ill-conditions, and REML λ-selection lands in a slightly different
    // basin — moving the fit by ~1% of signal range even though the model space
    // is identical (`{1, x − x̄}` spans the same null space). Subtract the knot
    // cloud's per-axis mean from both `data` and `knots` so the polynomial block
    // is built in a location-standardized, well-conditioned frame. The knots are
    // frozen (`UserProvided`) after fit, so this offset is identical at predict;
    // and under `x → x + c` the knots (selected from the data) shift by the same
    // `c`, so the centred coordinate — hence the whole basis — is invariant.
    let knot_mean: Vec<f64> = (0..d)
        .map(|c| knots.column(c).sum() / (k.max(1) as f64))
        .collect();
    let mut data_centered = data.to_owned();
    let mut knots_centered = knots.to_owned();
    for c in 0..d {
        let mu = knot_mean[c];
        data_centered.column_mut(c).mapv_inplace(|v| v - mu);
        knots_centered.column_mut(c).mapv_inplace(|v| v - mu);
    }
    let data = data_centered.view();
    let knots = knots_centered.view();

    // K block: radial basis evaluations data -> knots
    let mut kernel_block = Array2::<f64>::zeros((n, k));
    let kernel_result: Result<(), BasisError> = kernel_block
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            for j in 0..k {
                let mut dist2 = 0.0;
                for c in 0..d {
                    let delta = data[[i, c]] - knots[[j, c]];
                    dist2 += delta * delta;
                }
                row[j] = thin_plate_kernel_from_dist2(dist2 / (length_scale * length_scale), d)?;
            }
            Ok(())
        });
    kernel_result?;

    // P block: all TPS null-space monomials of total degree < m.
    let poly_block = thin_plate_polynomial_block(data);

    // Omega block on knots
    let mut omega = Array2::<f64>::zeros((k, k));
    let length_scale_sq = length_scale * length_scale;
    fill_symmetric_from_row_kernel(&mut omega, |i, j| {
        let mut dist2 = 0.0;
        for c in 0..d {
            let delta = knots[[i, c]] - knots[[j, c]];
            dist2 += delta * delta;
        }
        thin_plate_kernel_from_dist2(dist2 / length_scale_sq, d)
    })?;

    // Enforce TPS side-constraint P(knots)^T α = 0 by projecting onto
    // the nullspace of P(knots)^T.
    let z = thin_plate_kernel_constraint_nullspace(knots, &mut workspace.cache)?;
    let kernel_constrained = fast_ab(&kernel_block, &z);
    let omega_constrained = {
        let zt_o = fast_atb(&z, &omega);
        symmetrize_penalty(&fast_ab(&zt_o, &z))
    };
    let omega_psd = validate_psd_penalty(
        &omega_constrained,
        &format!("thin_plate bending penalty (dimension={d})"),
        "thin-plate kernel and side-constraint assembly must yield a PSD penalty on the constrained subspace",
    )?;
    assert!(
        omega_psd.min_eigenvalue >= -omega_psd.tolerance,
        "thin-plate constrained penalty PSD validation violated tolerance after validation: min_eigenvalue={}, tolerance={}",
        omega_psd.min_eigenvalue,
        omega_psd.tolerance
    );
    assert!(
        omega_psd.max_abs_eigenvalue.is_finite(),
        "thin-plate constrained penalty has non-finite max eigenvalue after validation: max_abs_eigenvalue={}",
        omega_psd.max_abs_eigenvalue
    );
    assert!(
        omega_psd.effective_rank <= omega_constrained.nrows(),
        "thin-plate constrained penalty rank exceeds constrained rows: effective_rank={}, rows={}",
        omega_psd.effective_rank,
        omega_constrained.nrows()
    );

    let constrained_kernel_cols = kernel_constrained.ncols();

    // Radial penalty eigenspace reparameterization. Eigendecompose
    // Ω_constrained = V Λ V' and rotate the radial design columns into the
    // same basis. This preserves the TPS model space while making the bending
    // block diagonal. Numerically near-null radial directions are not part of
    // the polynomial null space; keeping them as almost-free columns lets REML
    // spend EDF on wiggle with effectively zero curvature cost (#1271). Drop
    // them from the exposed basis so only genuinely penalized radial directions
    // remain.
    let (radial_reparam, radial_eigvals): (Array2<f64>, Array1<f64>) = if let Some(frozen) =
        frozen_radial_reparam
    {
        if frozen.nrows() != constrained_kernel_cols {
            crate::bail_dim_basis!(
                "thin-plate frozen radial reparam shape {:?} does not match constrained radial dimension {}",
                frozen.dim(),
                constrained_kernel_cols
            );
        }
        let v = frozen.to_owned();
        let vt_omega_v = fast_atb(&v, &omega_constrained);
        let lambda_diag = fast_ab(&vt_omega_v, &v);
        let mut evals = Array1::<f64>::zeros(v.ncols());
        for i in 0..v.ncols() {
            evals[i] = lambda_diag[[i, i]].max(0.0);
        }
        (v, evals)
    } else if constrained_kernel_cols == 0 {
        (Array2::<f64>::zeros((0, 0)), Array1::<f64>::zeros(0))
    } else {
        // #1347: reparameterize in the realized data metric so the bending
        // spectrum acquires mgcv's cliff (curvature per unit data-variance),
        // rather than the cliff-less raw knot-Gram spectrum that lets REML buy
        // near-free wiggle on near-linear data. G_c = (K Z)ᵀ (K Z).
        // Canonical row order so the Gram is row-permutation invariant (#1378).
        let design_gram = data_metric_design_gram(kernel_constrained.view());
        thin_plate_radial_reparam_data_metric(&omega_constrained, &design_gram)?
    };
    let kernel_cols = radial_eigvals.len();
    let total_cols = kernel_cols + poly_cols;

    let kernel_rotated = if kernel_cols == 0 {
        Array2::<f64>::zeros((n, 0))
    } else {
        fast_ab(&kernel_constrained, &radial_reparam)
    };

    let mut basis = Array2::<f64>::zeros((n, total_cols));
    basis
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&kernel_rotated);
    basis.slice_mut(s![.., kernel_cols..]).assign(&poly_block);

    let mut penalty_bending = Array2::<f64>::zeros((total_cols, total_cols));
    for i in 0..kernel_cols {
        penalty_bending[[i, i]] = radial_eigvals[i];
    }
    let penalty_ridge = build_nullspace_shrinkage_penalty(&penalty_bending)?
        .map(|block| block.sym_penalty)
        .unwrap_or_else(|| Array2::<f64>::zeros((total_cols, total_cols)));

    Ok(ThinPlateSplineBasis {
        basis,
        penalty_bending,
        penalty_ridge,
        num_kernel_basis: kernel_cols,
        num_polynomial_basis: poly_cols,
        dimension: d,
        radial_reparam,
    })
}

pub(crate) fn active_thin_plate_penalty_derivatives(
    penaltyinfo: &[PenaltyInfo],
    primary_derivative: &Array2<f64>,
) -> Result<Vec<Array2<f64>>, BasisError> {
    penaltyinfo
        .iter()
        .filter(|info| info.active)
        .map(|info| match &info.source {
            PenaltySource::Primary => Ok(primary_derivative.clone()),
            PenaltySource::DoublePenaltyNullspace => {
                Ok(Array2::<f64>::zeros(primary_derivative.raw_dim()))
            }
            other => Err(BasisError::InvalidInput(format!(
                "unexpected ThinPlate penalty source in psi-derivative path: {other:?}"
            ))),
        })
        .collect()
}

// The dense per-pair ThinPlate ψ-derivative builder used to live here. It has
// been replaced by `build_thin_plate_scalar_design_psi_derivatives`, which
// drives the same math through the shared scalar streaming infrastructure
// (`build_scalar_design_psi_derivatives_shared`) so large-scale TPS terms no
// longer materialize dense `(n × p)` first/second derivative arrays.

pub fn build_thin_plate_penalty_psi_derivativeswithworkspace(
    centers: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    // Match build_thin_plate_basis exactly (Wood-TPRS path):
    //
    //   M(ψ)        = Z_kernel^T Ω(ψ) Z_kernel
    //   V, Λ(ψ)    = eigh(M)        (or V from spec.radial_reparam, frozen)
    //   S_raw(ψ)   = pad(diag(Λ(ψ)), total_cols)         // kernel block + zero poly
    //   S_norm(ψ)  = S_raw(ψ) / ||S_raw(ψ)||_F
    //   S_final(ψ) = Z_id^T S_norm(ψ) Z_id              // identifiability transform
    //
    // where Ω_ij(ψ) = φ(r_ij(ψ)), r_ij(ψ) = ||c_i - c_j|| · exp(ψ).
    //
    // We need d/dψ S_final and d²/dψ² S_final, applied in the same composition
    // order as the build path so the analytic derivative is of the exact
    // materialized penalty surface.
    let z_kernel = thin_plate_kernel_constraint_nullspace(centers, &mut workspace.cache)?;
    let constrained_kernel_cols = z_kernel.ncols();
    let poly_cols = thin_plate_polynomial_basis_dimension(centers.ncols());
    let k = centers.nrows();
    let d = centers.ncols();

    // 1) Build Ω, Ω_ψ, Ω_ψψ on centers (k × k). Ω is needed to recover Λ when
    //    V is frozen and to apply Hellmann-Feynman in the fresh-V path.
    let mut omega = Array2::<f64>::zeros((k, k));
    let mut omega_psi = Array2::<f64>::zeros((k, k));
    let mut omega_psi_psi = Array2::<f64>::zeros((k, k));

    // Evaluate the dense symmetric center-pair kernel blocks in independent
    // lower-triangular row tiles. Each rayon worker owns its tile-local entry
    // buffer (scratch workspace) and returns immutable results; the serial
    // assembly below is the only place that writes to the dense output arrays,
    // so no mutable ndarray storage is shared across workers.
    struct ThinPlatePsiTileEntry {
        pub(crate) i: usize,
        pub(crate) j: usize,
        pub(crate) phi: f64,
        pub(crate) phi_psi: f64,
        pub(crate) phi_psi_psi: f64,
    }

    let n_tiles = k.div_ceil(THIN_PLATE_PENALTY_PSI_TILE_ROWS);
    let omega_tiles: Result<Vec<Vec<ThinPlatePsiTileEntry>>, BasisError> = (0..n_tiles)
        .into_par_iter()
        .map(|tile_idx| {
            let row_start = tile_idx * THIN_PLATE_PENALTY_PSI_TILE_ROWS;
            let row_end = (row_start + THIN_PLATE_PENALTY_PSI_TILE_ROWS).min(k);
            let tile_pairs = (row_start..row_end).map(|i| i + 1).sum::<usize>();
            let mut entries = Vec::with_capacity(tile_pairs);
            for i in row_start..row_end {
                for j in 0..=i {
                    let mut dist2 = 0.0;
                    for axis in 0..d {
                        let delta = centers[[i, axis]] - centers[[j, axis]];
                        dist2 += delta * delta;
                    }
                    let (phi, phi_psi, phi_psi_psi) = thin_plate_kernel_psi_triplet_from_distance(
                        dist2.sqrt(),
                        spec.length_scale,
                        d,
                    )?;
                    entries.push(ThinPlatePsiTileEntry {
                        i,
                        j,
                        phi,
                        phi_psi,
                        phi_psi_psi,
                    });
                }
            }
            Ok(entries)
        })
        .collect();

    for tile in omega_tiles? {
        for entry in tile {
            omega[[entry.i, entry.j]] = entry.phi;
            omega_psi[[entry.i, entry.j]] = entry.phi_psi;
            omega_psi_psi[[entry.i, entry.j]] = entry.phi_psi_psi;
            if entry.i != entry.j {
                omega[[entry.j, entry.i]] = entry.phi;
                omega_psi[[entry.j, entry.i]] = entry.phi_psi;
                omega_psi_psi[[entry.j, entry.i]] = entry.phi_psi_psi;
            }
        }
    }

    // 2) Project to the constrained kernel space.
    let m_constrained = symmetrize_penalty(&z_kernel.t().dot(&omega).dot(&z_kernel));
    let m_psi_constrained = symmetrize_penalty(&z_kernel.t().dot(&omega_psi).dot(&z_kernel));
    let m_pp_constrained = symmetrize_penalty(&z_kernel.t().dot(&omega_psi_psi).dot(&z_kernel));

    // 3) Get V (frozen or fresh from eigh).
    let (v, lambda) = if let Some(frozen) = spec.radial_reparam.as_ref() {
        if frozen.nrows() != constrained_kernel_cols {
            crate::bail_dim_basis!(
                "thin-plate frozen radial reparam shape {:?} does not match constrained radial dimension {}",
                frozen.dim(),
                constrained_kernel_cols
            );
        }
        let v_owned = frozen.to_owned();
        let lambda_diag = fast_ab(&fast_atb(&v_owned, &m_constrained), &v_owned);
        let mut evals = Array1::<f64>::zeros(v_owned.ncols());
        for i in 0..v_owned.ncols() {
            evals[i] = lambda_diag[[i, i]].max(0.0);
        }
        (v_owned, evals)
    } else if constrained_kernel_cols == 0 {
        (Array2::<f64>::zeros((0, 0)), Array1::<f64>::zeros(0))
    } else {
        let (mut evals, evecs) =
            FaerEigh::eigh(&m_constrained, Side::Lower).map_err(BasisError::LinalgError)?;
        for ev in evals.iter_mut() {
            if *ev < 0.0 {
                *ev = 0.0;
            }
        }
        let keep = thin_plate_retained_radial_indices(&evals);
        (evecs.select(Axis(1), &keep), evals.select(Axis(0), &keep))
    };
    let kernel_cols = lambda.len();
    let total_cols = kernel_cols + poly_cols;
    let v_is_frozen = spec.radial_reparam.is_some();

    // 4) Rotate the constrained-space derivatives into V's basis. These are the
    //    coefficients used by Hellmann-Feynman / standard perturbation theory:
    //      A_ψ[i,j]  = v_i^T M_ψ  v_j
    //      A_ψψ[i,j] = v_i^T M_ψψ v_j.
    let a_psi = if kernel_cols > 0 {
        v.t().dot(&m_psi_constrained).dot(&v)
    } else {
        Array2::<f64>::zeros((0, 0))
    };
    let a_pp = if kernel_cols > 0 {
        v.t().dot(&m_pp_constrained).dot(&v)
    } else {
        Array2::<f64>::zeros((0, 0))
    };

    // 5) Build the un-normalized rotated penalty and its ψ-derivatives.
    //
    //    Frozen V (predict-time): the penalty is V^T M(ψ) V — a full kc×kc
    //    matrix that equals diag(Λ_0) only at fit-time ψ_0. Its ψ-derivatives
    //    are simply A_ψ and A_ψψ (full matrices).
    //
    //    Fresh V (fit-time, no frozen reparam): V(ψ) re-diagonalizes M(ψ) at
    //    each ψ, so the penalty is identically diag(Λ(ψ)). Off-diagonals
    //    vanish at every ψ; on-diagonals follow from non-degenerate eigenvalue
    //    perturbation:
    //      dΛ_i/dψ   = A_ψ[i,i]
    //      d²Λ_i/dψ² = A_ψψ[i,i] + 2 Σ_{k ≠ i} A_ψ[i,k]² / (Λ_i − Λ_k)
    //    For degenerate eigenvalues the off-diagonal correction is dropped on
    //    the offending pairs (their contribution is encoded in subspace
    //    rotations rather than scalar eigenvalue motion).
    let s_raw_kernel = Array2::from_diag(&lambda);
    let s_raw_psi_kernel = if v_is_frozen {
        a_psi.clone()
    } else {
        let mut diag = Array2::<f64>::zeros((kernel_cols, kernel_cols));
        for i in 0..kernel_cols {
            diag[[i, i]] = a_psi[[i, i]];
        }
        diag
    };
    let s_raw_pp_kernel = if v_is_frozen {
        a_pp.clone()
    } else {
        let mut diag = Array2::<f64>::zeros((kernel_cols, kernel_cols));
        for i in 0..kernel_cols {
            let mut acc = a_pp[[i, i]];
            for k_idx in 0..kernel_cols {
                if k_idx == i {
                    continue;
                }
                let denom = lambda[i] - lambda[k_idx];
                if denom.abs() > 1e-14 {
                    acc += 2.0 * a_psi[[i, k_idx]].powi(2) / denom;
                }
            }
            diag[[i, i]] = acc;
        }
        diag
    };

    // 6) Pad to total_cols (poly block has zero penalty).
    let pad = |kernel_block: &Array2<f64>| -> Array2<f64> {
        let mut s = Array2::<f64>::zeros((total_cols, total_cols));
        if kernel_cols > 0 {
            s.slice_mut(s![0..kernel_cols, 0..kernel_cols])
                .assign(kernel_block);
        }
        s
    };
    let s_raw = pad(&s_raw_kernel);
    let s_raw_psi = pad(&s_raw_psi_kernel);
    let s_raw_pp = pad(&s_raw_pp_kernel);

    // 7) Apply the Frobenius normalization chain rule. The build path divides
    //    by c(ψ)=||S_raw(ψ)||_F before applying the identifiability transform.
    //    Therefore:
    //      S_norm'  = S_raw'/c - c' S_raw/c²
    //      S_norm'' = S_raw''/c - 2c' S_raw'/c²
    //                  + (2(c')²/c³ - c''/c²) S_raw,
    //    exactly as implemented by `normalize_penaltywith_psi_derivatives`.
    let (_, s_norm_psi, s_norm_pp, _c) =
        normalize_penaltywith_psi_derivatives(&s_raw, &s_raw_psi, &s_raw_pp);

    // 8) Apply the identifiability transform last (matches build path order:
    //    `if let Some(z) = ... { Z^T penalty_norm Z }`).
    let s_psi_out = project_penalty_matrix(&s_norm_psi, identifiability_transform);
    let s_psi_psi_out = project_penalty_matrix(&s_norm_pp, identifiability_transform);

    Ok((s_psi_out, s_psi_psi_out))
}

/// Build the design ψ-derivatives for a Thin-Plate Spline term via the shared
/// scalar streaming infrastructure that Duchon already uses at large scale.
///
/// At small `n` this materializes both the first and second derivative arrays
/// just like the legacy dense path; at large scale the policy elects
/// streaming and both arrays come back as zero-sized — only an
/// `ImplicitDesignPsiDerivative` is returned, and downstream consumers
/// (`spatial_log_kappa_hyper_dirs_frominfo_list`) dispatch matvecs through it
/// instead of materializing dense `(n × p)` arrays per axis.
pub(crate) fn build_thin_plate_scalar_design_psi_derivatives(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<ScalarDesignPsiDerivatives, BasisError> {
    let z_kernel = thin_plate_kernel_constraint_nullspace(centers, &mut workspace.cache)?;
    let constrained_kernel_cols = z_kernel.ncols();
    let kernel_transform = if let Some(v) = spec.radial_reparam.as_ref() {
        if v.nrows() != constrained_kernel_cols {
            crate::bail_dim_basis!(
                "thin-plate radial reparam shape {:?} does not match constrained radial dimension {}",
                v.dim(),
                constrained_kernel_cols
            );
        }
        fast_ab(&z_kernel, v)
    } else {
        z_kernel
    };
    let kernel_cols = kernel_transform.ncols();
    let poly_cols = thin_plate_polynomial_basis_dimension(data.ncols());
    let p_after_pad = kernel_cols + poly_cols;
    let p_final = identifiability_transform
        .map(|zf| zf.ncols())
        .unwrap_or(p_after_pad);
    build_scalar_design_psi_derivatives_shared(
        data,
        centers,
        None,
        p_final,
        Some(kernel_transform),
        identifiability_transform.cloned(),
        poly_cols,
        RadialScalarKind::ThinPlate {
            length_scale: spec.length_scale,
            dim: data.ncols(),
        },
        0.0,
    )
}

pub fn build_thin_plate_basis_log_kappa_derivative(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_thin_plate_basis_log_kappa_derivativewithworkspace(data, spec, &mut workspace)
}

pub fn build_thin_plate_basis_log_kappa_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let mut bundle =
        build_thin_plate_basis_log_kappa_derivativeswithworkspace(data, spec, workspace)?;
    bundle.first.implicit_operator = bundle.implicit_operator;
    Ok(bundle.first)
}

pub fn build_thin_plate_basis_log_kappa_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_thin_plate_basis_log_kappa_derivativeswithworkspace(data, spec, &mut workspace)
}

pub fn build_thin_plate_basis_log_kappa_derivativeswithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    let base = build_thin_plate_basiswithworkspace(data, spec, workspace)?;
    let (centers, identifiability_transform, radial_reparam) = match &base.metadata {
        BasisMetadata::ThinPlate {
            centers,
            identifiability_transform,
            radial_reparam,
            ..
        } => (
            centers.clone(),
            identifiability_transform.clone(),
            radial_reparam.clone(),
        ),
        _ => {
            crate::bail_invalid_basis!("ThinPlate derivative path expected ThinPlate metadata");
        }
    };
    let mut derivative_spec = spec.clone();
    if derivative_spec.radial_reparam.is_none() {
        derivative_spec.radial_reparam = radial_reparam;
    }
    let scalar = build_thin_plate_scalar_design_psi_derivatives(
        data,
        centers.view(),
        &derivative_spec,
        identifiability_transform.as_ref(),
        workspace,
    )?;
    let (primary_derivative_opt, primarysecond_derivative_opt) =
        build_thin_plate_penalty_psi_derivativeswithworkspace(
            centers.view(),
            &derivative_spec,
            identifiability_transform.as_ref(),
            workspace,
        )?;
    let primary_derivative = primary_derivative_opt;
    let primarysecond_derivative = primarysecond_derivative_opt;
    let penalties_derivative =
        active_thin_plate_penalty_derivatives(&base.penaltyinfo, &primary_derivative)?;
    let penaltiessecond_derivative =
        active_thin_plate_penalty_derivatives(&base.penaltyinfo, &primarysecond_derivative)?;
    Ok(BasisPsiDerivativeBundle {
        first: BasisPsiDerivativeResult {
            design_derivative: scalar.design_first,
            penalties_derivative,
            implicit_operator: None,
        },
        second: BasisPsiSecondDerivativeResult {
            designsecond_derivative: scalar.design_second_diag,
            penaltiessecond_derivative,
            implicit_operator: None,
        },
        implicit_operator: scalar.implicit_operator,
    })
}

pub fn build_thin_plate_basis_log_kappasecond_derivative(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_thin_plate_basis_log_kappasecond_derivativewithworkspace(data, spec, &mut workspace)
}

pub fn build_thin_plate_basis_log_kappasecond_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let mut bundle =
        build_thin_plate_basis_log_kappa_derivativeswithworkspace(data, spec, workspace)?;
    bundle.second.implicit_operator = bundle.implicit_operator;
    Ok(bundle.second)
}

/// High-level TPS constructor: selects knots from data, then builds basis+penalty.
pub fn create_thin_plate_spline_basis_with_knot_count(
    data: ArrayView2<f64>,
    num_knots: usize,
) -> Result<(ThinPlateSplineBasis, Array2<f64>), BasisError> {
    let mut workspace = BasisWorkspace::default();
    create_thin_plate_spline_basis_with_knot_count_andworkspace(data, num_knots, &mut workspace)
}

pub fn create_thin_plate_spline_basis_with_knot_count_andworkspace(
    data: ArrayView2<f64>,
    num_knots: usize,
    workspace: &mut BasisWorkspace,
) -> Result<(ThinPlateSplineBasis, Array2<f64>), BasisError> {
    let knots = select_thin_plate_knots(data, num_knots)?;
    let basis = create_thin_plate_spline_basiswithworkspace(data, knots.view(), workspace)?;
    Ok((basis, knots))
}

/// Applies a sum-to-zero constraint to a basis matrix for model identifiability.
///
/// This is achieved by reparameterizing the basis to be orthogonal to the weighted intercept.
/// In GAMs, this constraint removes the confounding between the intercept and smooth functions.
/// For weighted models (e.g., GLM-IRLS), the constraint is B^T W 1 = 0 instead of B^T 1 = 0.
///
/// # Arguments
/// * `basis_matrix`: An `ArrayView2<f64>` of the original, unconstrained basis matrix.
/// * `weights`: Optional weights for the constraint. If None, uses unweighted constraint.
///
/// # Returns
/// A tuple containing:
/// - The new, constrained basis matrix (with `k - rank(c)` columns).
/// - The transformation matrix `Z` used to create it.
pub fn apply_sum_to_zero_constraint(
    basis_matrix: ArrayView2<f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let n = basis_matrix.nrows();
    let k = basis_matrix.ncols();
    if k < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: k });
    }

    // c = B^T w (weighted constraint) or B^T 1 (unweighted constraint)
    let constraintvector = match weights {
        Some(w) => {
            if w.len() != n {
                return Err(BasisError::WeightsDimensionMismatch {
                    expected: n,
                    found: w.len(),
                });
            }
            w.to_owned()
        }
        None => Array1::<f64>::ones(n),
    };
    let c = basis_matrix.t().dot(&constraintvector); // shape k

    // Orthonormal basis for nullspace of c^T from a pivoted QR of the k×1
    // constraint matrix.
    let mut c_mat = Array2::<f64>::zeros((k, 1));
    c_mat.column_mut(0).assign(&c);
    let (z, rank) =
        rrqr_nullspace_basis(&c_mat, default_rrqr_rank_alpha()).map_err(BasisError::LinalgError)?;
    if rank >= k {
        return Err(BasisError::ConstraintNullspaceCollapsed {
            site: "apply_sum_to_zero_constraint",
            cross_rank: rank,
            coeff_dim: k,
            cross_frobenius: c.iter().map(|v| v * v).sum::<f64>().sqrt(),
            gram_spectrum: "not computed (structural rank collapse before Gram eigendecomposition)"
                .to_string(),
        });
    }
    if rank == 0 {
        // Already orthogonal to the intercept constraint; keep full basis unchanged.
        return Ok((basis_matrix.to_owned(), Array2::eye(k)));
    }

    let gauge = gam_problem::Gauge::sum_to_zero(z);
    let constrained = gauge.restrict_design(&basis_matrix);
    let z = gauge.block_transform(0);
    Ok((constrained, z))
}

/// Build a sum-to-zero reparametrization for a sparse basis.
///
/// Returns `(B_c, Z)` where `Z` is an **orthonormal** basis for `null(c^T)`
/// with `c = B^T w` (the weighted column sums of `B`), and
/// `B_c = B Z` is the constrained design matrix.
///
/// Because `Z` has orthonormal columns, `Z Zᵀ` is the canonical
/// orthogonal projector onto `null(cᵀ)` — i.e. it is idempotent and
/// `cᵀ Z Zᵀ = 0`, so any vector projected by `Z Zᵀ` still satisfies the
/// sum-to-zero constraint. The previous "drop the pivot column" trick
/// produced a valid null-space basis but with non-orthogonal, non-unit
/// columns, breaking the projector identities downstream code may rely on.
///
/// `Z` is dense `(k × (k-1))`; consequently `B_c = B Z` is returned as a
/// dense matrix even when `B` is sparse. Callers that previously relied on
/// the constrained basis being sparse should wrap the result in
/// [`DenseDesignMatrix`].
pub fn apply_sum_to_zero_constraint_sparse(
    basis_matrix: &SparseColMat<usize, f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let n = basis_matrix.nrows();
    let k = basis_matrix.ncols();
    if k < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: k });
    }

    let constraint_weights = match weights {
        Some(w) => {
            if w.len() != n {
                return Err(BasisError::WeightsDimensionMismatch {
                    expected: n,
                    found: w.len(),
                });
            }
            w.to_owned()
        }
        None => Array1::<f64>::ones(n),
    };

    // c = Bᵀ w (k-vector of weighted column sums) computed directly from the
    // CSC storage.
    let mut c = Array1::<f64>::zeros(k);
    let (symbolic, values) = basis_matrix.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..k {
        let mut sum = 0.0;
        for idx in col_ptr[col]..col_ptr[col + 1] {
            sum += values[idx] * constraint_weights[row_idx[idx]];
        }
        c[col] = sum;
    }

    // Orthonormal basis for null(cᵀ) via a column-pivoted QR of the k×1
    // constraint matrix — exactly the same construction used by the dense
    // path `apply_sum_to_zero_constraint`. This guarantees ZᵀZ = I and hence
    // that ZZᵀ is the canonical orthogonal projector onto null(cᵀ).
    let mut c_mat = Array2::<f64>::zeros((k, 1));
    c_mat.column_mut(0).assign(&c);
    let (z, rank) =
        rrqr_nullspace_basis(&c_mat, default_rrqr_rank_alpha()).map_err(BasisError::LinalgError)?;
    if rank >= k {
        return Err(BasisError::ConstraintNullspaceCollapsed {
            site: "apply_sum_to_zero_constraint_sparse",
            cross_rank: rank,
            coeff_dim: k,
            cross_frobenius: c.iter().map(|v| v * v).sum::<f64>().sqrt(),
            gram_spectrum: "not computed (structural rank collapse before Gram eigendecomposition)"
                .to_string(),
        });
    }
    if rank == 0 {
        // Constraint is numerically zero (e.g. weights produced cᵀ ≈ 0):
        // the basis already lies in null(cᵀ), so the constrained basis is
        // the dense materialization of B with Z = I.
        let mut dense_b = Array2::<f64>::zeros((n, k));
        for col in 0..k {
            for idx in col_ptr[col]..col_ptr[col + 1] {
                dense_b[[row_idx[idx], col]] = values[idx];
            }
        }
        return Ok((dense_b, Array2::eye(k)));
    }

    // Constrained basis B_c = B Z. Iterate columns of Z and apply B as a
    // sparse-times-dense-vector product per column. Result is dense
    // `(n × (k-1))` since Z is dense.
    let kc = z.ncols();
    let mut constrained = Array2::<f64>::zeros((n, kc));
    for out_col in 0..kc {
        let z_col = z.column(out_col);
        let mut dst = constrained.column_mut(out_col);
        for src_col in 0..k {
            let coeff = z_col[src_col];
            if coeff == 0.0 {
                continue;
            }
            for idx in col_ptr[src_col]..col_ptr[src_col + 1] {
                dst[row_idx[idx]] += coeff * values[idx];
            }
        }
    }

    Ok((constrained, z))
}

/// Reparameterizes a basis matrix so its columns are orthogonal (with optional weights)
/// to a supplied constraint matrix.
///
/// Let:
/// - `B` be the raw basis (`n x k`)
/// - `C` be the constraint matrix (`n x q`)
/// - `W` be diagonal weights (`n x n`), or identity when `weights=None`
///
/// We seek a transformed basis `B_c = B K` (`n x k_c`) such that:
///   `B_c^T W C = 0`.
///
/// Expanding:
///   `B_c^T W C = (B K)^T W C = K^T (B^T W C)`.
///
/// So it is enough to choose columns of `K` in `null((B^T W C)^T)`.
/// This implementation computes:
///   `M = B^T W C` (`k x q`)
/// and extracts a basis for `null(M^T)` via column-pivoted Householder QR.
///
/// The result enforces orthogonality by construction while retaining the largest possible
/// smooth subspace under the given constraints.
pub fn applyweighted_orthogonality_constraint(
    basis_matrix: ArrayView2<f64>,
    constraint_matrix: ArrayView2<f64>,
    weights: Option<ArrayView1<f64>>,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let n = basis_matrix.nrows();
    let k = basis_matrix.ncols();
    if constraint_matrix.nrows() != n {
        return Err(BasisError::ConstraintMatrixRowMismatch {
            basisrows: n,
            constraintrows: constraint_matrix.nrows(),
        });
    }
    if k == 0 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: 0 });
    }
    let q = constraint_matrix.ncols();
    if q == 0 {
        return Ok((basis_matrix.to_owned(), Array2::eye(k)));
    }

    // Form W*C by row scaling because W is diagonal.
    let mut weighted_constraints = constraint_matrix.to_owned();
    if let Some(w) = weights {
        if w.len() != n {
            return Err(BasisError::WeightsDimensionMismatch {
                expected: n,
                found: w.len(),
            });
        }
        for (mut row, &weight) in weighted_constraints.axis_iter_mut(Axis(0)).zip(w.iter()) {
            row *= weight;
        }
    }

    // M = B^T W C. Its transpose M^T has nullspace directions in coefficient space
    // that produce basis columns orthogonal to C under the W-inner product.
    let constraint_cross = basis_matrix.t().dot(&weighted_constraints); // k×q
    let gram = fast_ata(&basis_matrix);
    let transform = orthogonality_transform_from_cross_and_gram(&constraint_cross, &gram)?;
    let basis_orthonormal = fast_ab(&basis_matrix, &transform);
    Ok((basis_orthonormal, transform))
}

/// Compute Greville abscissae for a B-spline basis.
///
/// The Greville abscissa for basis function j is defined as:
///   G_j = (1/d) × Σ_{k=1}^{d} t_{j+k}
///
/// These provide the "center" of support for each basis function and are used
/// for geometric constraints that don't depend on observed data. A key property
/// is that a linear function f(x) = a + bx has B-spline coefficients c_j = a + b·G_j,
/// so constraining coefficients to be orthogonal to [1, G] removes linear functions
/// from the representable space.
///
/// # Arguments
/// * `knot_vector` - Full knot vector including boundary repetitions
/// * `degree` - B-spline degree (typically 3 for cubic)
///
/// # Returns
/// Array of Greville abscissae, one per basis function (length = n_knots - degree - 1)
///
/// # Errors
/// Returns error if knot vector is too short or Greville abscissae are degenerate.
pub fn compute_greville_abscissae(
    knot_vector: &Array1<f64>,
    degree: usize,
) -> Result<Array1<f64>, BasisError> {
    let n_knots = knot_vector.len();
    if degree == 0 {
        // For degree 0, Greville abscissae are knot midpoints
        let n_basis = n_knots.saturating_sub(1);
        if n_basis == 0 {
            return Err(BasisError::InsufficientColumnsForConstraint { found: 0 });
        }
        let mut g = Array1::<f64>::zeros(n_basis);
        for j in 0..n_basis {
            g[j] = 0.5 * (knot_vector[j] + knot_vector[j + 1]);
        }
        return Ok(g);
    }

    // Number of basis functions: k = n_knots - degree - 1
    if n_knots <= degree + 1 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: n_knots.saturating_sub(degree + 1),
        });
    }
    let n_basis = n_knots - degree - 1;

    let mut g = Array1::<f64>::zeros(n_basis);
    let d_inv = 1.0 / (degree as f64);

    for j in 0..n_basis {
        // G_j = (1/d) × Σ_{k=1}^{d} t_{j+k}
        let mut sum = 0.0;
        for k in 1..=degree {
            sum += knot_vector[j + k];
        }
        g[j] = sum * d_inv;
    }

    // Check for degeneracy (all Greville abscissae equal)
    let g_min = g.iter().cloned().fold(f64::INFINITY, f64::min);
    let g_max = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (g_max - g_min) < 1e-10 {
        return Err(BasisError::DegenerateKnots);
    }

    Ok(g)
}

/// Compute the constraint transform Z using Greville abscissae (geometric constraints).
///
/// This creates a transform that removes constant and linear trends from spline
/// coefficients based purely on knot geometry, without reference to observed data.
/// This makes Z constant w.r.t. model parameters β, ensuring dZ/dβ = 0 exactly,
/// which enables exact analytic gradients.
///
/// # Mathematical Background
/// For B-splines, a linear function f(x) = a + bx has coefficients c_j = a + b·G_j
/// where G_j are the Greville abscissae. Therefore, constraining the coefficient
/// vector θ to satisfy:
///   - Σ θ_j = 0  (orthogonal to constants)
///   - Σ θ_j·G_j = 0  (orthogonal to linear in Greville coordinates)
/// removes the ability to represent any linear function.
///
/// # Arguments
/// * `knot_vector` - Full knot vector
/// * `degree` - B-spline degree
/// * `penalty_order` - Order of difference penalty (typically 2)
///
/// # Returns
/// Tuple of (transform Z, projected_penalty Z'SZ) where:
/// - Z: k × (k-2) matrix mapping raw coefficients to constrained space
/// - S_constrained: (k-2) × (k-2) projected second-difference penalty
pub fn compute_geometric_constraint_transform(
    knot_vector: &Array1<f64>,
    degree: usize,
    penalty_order: usize,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    // 1. Compute Greville abscissae
    let g = compute_greville_abscissae(knot_vector, degree)?;
    let k = g.len();

    if k < 3 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: k });
    }

    // 2. Build constraint matrix C_geom (2 × k)
    // Row 0: all ones (intercept constraint)
    // Row 1: Greville abscissae (linear constraint)
    let mut c_geom = Array2::<f64>::zeros((2, k));
    for j in 0..k {
        c_geom[[0, j]] = 1.0;
        c_geom[[1, j]] = g[j];
    }

    // 3. Standardize linear row for numerical conditioning
    let g_mean = g.mean().unwrap_or(0.0);
    let gvar = g.iter().map(|&x| (x - g_mean).powi(2)).sum::<f64>() / (k as f64);
    let g_std = gvar.sqrt().max(1e-10);
    for j in 0..k {
        c_geom[[1, j]] = (c_geom[[1, j]] - g_mean) / g_std;
    }

    // 4. Column-pivoted QR on C_geom^T; the trailing Q columns span null(C_geom).
    let (z, rank) = rrqr_nullspace_basis(&c_geom.t(), default_rrqr_rank_alpha())
        .map_err(BasisError::LinalgError)?;
    if rank >= k {
        return Err(BasisError::ConstraintNullspaceCollapsed {
            site: "compute_geometric_constraint_transform",
            cross_rank: rank,
            coeff_dim: k,
            cross_frobenius: f64::NAN,
            gram_spectrum: "not computed (structural rank collapse before Gram eigendecomposition)"
                .to_string(),
        });
    }

    if z.ncols() == 0 {
        return Err(BasisError::ConstraintNullspaceCollapsed {
            site: "compute_geometric_constraint_transform",
            cross_rank: 0,
            coeff_dim: k,
            cross_frobenius: f64::NAN,
            gram_spectrum: "not computed (structural rank collapse before Gram eigendecomposition)"
                .to_string(),
        });
    }

    // 5. Build raw penalty and project: S_c = Z' S Z
    let s_raw = create_difference_penalty_matrix(k, penalty_order, Some(g.view()))?;
    let s_constrained = {
        let zt_s = fast_atb(&z, &s_raw);
        fast_ab(&zt_s, &z)
    };

    Ok((z, s_constrained))
}

/// Result of auto-deriving a clamped B-spline knot vector from 1-D data.
///
/// The `degree` / `num_internal_knots` fields report the **effective** values
/// that were actually used to build `knots`. They may differ from the
/// requested values when the engine had to auto-shrink the configuration
/// (issue #340): with small `n`, cubic-by-default gracefully degrades to
/// quadratic / linear, and the interior-knot count shrinks toward zero.
///
/// `shrunk` is `true` iff at least one of the two parameters was reduced
/// relative to the request, so callers can surface the decision in model
/// summaries / logs without recomputing it.
#[derive(Debug, Clone)]
pub struct AutoBSplineKnots {
    pub knots: Array1<f64>,
    pub degree: usize,
    pub num_internal_knots: usize,
    pub shrunk: bool,
}

/// Build a clamped B-spline full knot vector from 1-D data.
///
/// Thin public wrapper around
/// [`internal::generate_full_knot_vector_quantile`] so external crates can
/// request auto-derived knots without reimplementing the placement logic.
///
/// When `n = data.len()` is too small to support the requested
/// `(num_internal_knots, degree)` combination, this function auto-shrinks the
/// configuration to the largest feasible one (see [`auto_shrink_bspline_config`]):
///   * `num_internal_knots` is capped at `n - 2`.
///   * `degree` is reduced (cubic → quadratic → linear) until `n >= degree + 1`.
///
/// Only when even linear placement is impossible (`n < 2` or the data range is
/// degenerate) does this raise an error. The returned [`AutoBSplineKnots`]
/// records the effective configuration so downstream evaluators stay in sync.
pub fn auto_knot_vector_1d_quantile(
    data: ArrayView1<'_, f64>,
    num_internal_knots: usize,
    degree: usize,
) -> Result<AutoBSplineKnots, BasisError> {
    let n = data.len();
    let Some((eff_knots, eff_degree, shrunk)) =
        auto_shrink_bspline_config(n, num_internal_knots, degree)
    else {
        crate::bail_invalid_basis!(
            "auto-knot placement needs at least 2 finite evaluation points (got n={n}); \
             cannot fit even a linear B-spline",
        );
    };
    let knots = internal::generate_full_knot_vector_quantile(data, eff_knots, eff_degree)?;
    Ok(AutoBSplineKnots {
        knots,
        degree: eff_degree,
        num_internal_knots: eff_knots,
        shrunk,
    })
}

/// Build a clamped full B-spline knot vector from explicit *internal* knot
/// positions (mgcv `knots=` semantics).
///
/// The user supplies the interior knots (those strictly between the data
/// endpoints). This wraps them in the standard clamped boundary stencil:
/// `data_range.0` repeated `degree + 1` times, the sorted distinct internal
/// positions, then `data_range.1` repeated `degree + 1` times — matching the
/// layout produced by [`internal::generate_full_knot_vector`] for the uniform
/// case, except the interior positions are taken verbatim from the caller.
///
/// Internal positions must lie strictly inside `(data_range.0, data_range.1)`,
/// be finite, and be strictly increasing after sorting (no duplicates, which
/// would create a degenerate knot span). The data range itself is derived from
/// the covariate so the spline domain still spans the observed data even when
/// the user only pins a few interior knots.
pub fn clamped_knot_vector_from_internal_positions(
    data_range: (f64, f64),
    internal_positions: &[f64],
    degree: usize,
) -> Result<Array1<f64>, BasisError> {
    let (minval, maxval) = data_range;
    if !(minval.is_finite() && maxval.is_finite()) {
        crate::bail_invalid_basis!(
            "explicit knots require a finite data range, got ({minval:.6e}, {maxval:.6e})"
        );
    }
    if minval >= maxval {
        return Err(BasisError::InvalidRange(minval, maxval));
    }
    let scale = (maxval - minval).abs().max(1.0);
    let tol = 1e-12 * scale;

    let mut interior: Vec<f64> = Vec::with_capacity(internal_positions.len());
    for &k in internal_positions {
        if !k.is_finite() {
            crate::bail_invalid_basis!("explicit knot position {k:.6e} is not finite");
        }
        if k <= minval + tol || k >= maxval - tol {
            crate::bail_invalid_basis!(
                "explicit internal knot {k:.6e} must lie strictly inside the data range \
                 ({minval:.6e}, {maxval:.6e}); boundary knots are added automatically"
            );
        }
        interior.push(k);
    }
    interior.sort_by(f64::total_cmp);
    for w in interior.windows(2) {
        if (w[1] - w[0]).abs() <= tol {
            crate::bail_invalid_basis!(
                "explicit internal knots must be strictly increasing; \
                 found a duplicate/near-duplicate near {:.6e}",
                w[0]
            );
        }
    }

    let total_knots = interior.len() + 2 * (degree + 1);
    let mut knots = Vec::with_capacity(total_knots);
    for _ in 0..=degree {
        knots.push(minval);
    }
    knots.extend_from_slice(&interior);
    for _ in 0..=degree {
        knots.push(maxval);
    }
    Ok(Array::from_vec(knots))
}

/// Place `num_centers` Duchon centers on 1-D data via the equal-mass strategy.
///
/// Thin public wrapper around [`select_equal_mass_centers`] specialised to a
/// single covariate dimension. The returned vector is sorted.
pub fn auto_centers_1d_equal_mass(
    data: ArrayView1<'_, f64>,
    num_centers: usize,
) -> Result<Array1<f64>, BasisError> {
    let column = data.to_owned().insert_axis(Axis(1));
    let centers = select_equal_mass_centers(column.view(), num_centers)?;
    let mut flat: Vec<f64> = centers.column(0).iter().copied().collect();
    flat.sort_by(f64::total_cmp);
    Ok(Array1::from_vec(flat))
}

#[cfg(test)]
mod knot_selection_invariance_tests {
    // Regression tests for the knot-selector invariance defects fixed by the
    // rotation-equivariant maximin seed (gam#1456 rotation, gam#1378 row
    // permutation). Both would FAIL on the OLD seed, which started the greedy
    // farthest-point recursion at the lexicographically-smallest-coordinate row:
    //   * a 90 degree rotation about the centroid changes which row is
    //     lexicographically smallest, reseeding at a different physical point and
    //     selecting a different knot SET (rotation leak, #1456);
    //   * a row permutation changes the row index of that smallest row only when
    //     two rows tie, but more fundamentally the index-based tie-breaks made the
    //     selected set order-dependent (#1378).
    // The fix seeds at the centroid-nearest row (rotation-equivariant, a pure
    // function of the unordered value set) with value-lexicographic tie-breaks, so
    // the selected SET is invariant under both transforms to machine precision.
    use super::select_thin_plate_knots;
    use ndarray::Array2;

    /// A deterministic, asymmetric 2-D point cloud. It is deliberately NOT a
    /// rotation-symmetric grid: the points have distinct distances to the
    /// centroid and distinct coordinate orderings, so the centroid-nearest seed
    /// is unique and the OLD lexicographic seed lands on a different physical
    /// point after a 90 degree rotation.
    fn sample_cloud() -> Array2<f64> {
        // 12 scattered points in the plane.
        let pts: Vec<[f64; 2]> = vec![
            [0.10, 0.20],
            [1.30, 0.05],
            [2.10, 1.40],
            [0.40, 2.30],
            [1.90, 2.80],
            [3.20, 0.70],
            [2.70, 3.10],
            [0.90, 1.10],
            [3.50, 2.20],
            [1.60, 3.60],
            [0.05, 3.05],
            [2.40, 0.30],
        ];
        let mut a = Array2::<f64>::zeros((pts.len(), 2));
        for (i, p) in pts.iter().enumerate() {
            a[[i, 0]] = p[0];
            a[[i, 1]] = p[1];
        }
        a
    }

    /// Canonicalise a knot set into a sorted multiset of (bit-pattern) coordinate
    /// tuples so two selections can be compared as SETS, independent of the order
    /// in which the rows were emitted. Using the IEEE-754 bit pattern makes the
    /// comparison exact (machine precision) and is valid here because the 90
    /// degree rotation `(x,z)->(-z,x)` about the centroid is built from exact
    /// f64 additions/negations of the same operands, so equal physical points
    /// have bit-identical coordinates.
    fn canonical(knots: &Array2<f64>) -> Vec<(u64, u64)> {
        let mut rows: Vec<(u64, u64)> = (0..knots.nrows())
            .map(|r| (knots[[r, 0]].to_bits(), knots[[r, 1]].to_bits()))
            .collect();
        rows.sort_unstable();
        rows
    }

    /// Centroid of a 2-D point set, as the rigid-rotation pivot.
    fn data_centroid_2d(data: &Array2<f64>) -> (f64, f64) {
        let n = data.nrows();
        let cx = (0..n).map(|i| data[[i, 0]]).sum::<f64>() / n as f64;
        let cz = (0..n).map(|i| data[[i, 1]]).sum::<f64>() / n as f64;
        (cx, cz)
    }

    /// Exact 90 degree rotation of every row about an EXPLICIT center
    /// `(cx, cz)`: `(x, z) -> (cx - (z - cz), cz + (x - cx))`. Built from f64
    /// add/sub only, so it introduces no rounding beyond the operands
    /// themselves. The center is passed in (rather than recomputed per array)
    /// so the data and a selected subset can be rotated about the SAME pivot —
    /// rotation invariance of the knot SET is `select(R·data) == R·select(data)`
    /// for one fixed `R`, which only holds bit-for-bit when both sides rotate
    /// about the identical center.
    fn rotate_90_about(data: &Array2<f64>, cx: f64, cz: f64) -> Array2<f64> {
        let n = data.nrows();
        let mut out = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let dx = data[[i, 0]] - cx;
            let dz = data[[i, 1]] - cz;
            out[[i, 0]] = cx - dz;
            out[[i, 1]] = cz + dx;
        }
        out
    }

    #[test]
    fn knot_set_is_rotation_invariant_gh1456() {
        let data = sample_cloud();
        let n = data.nrows();
        // FarthestPoint path: strictly fewer knots than rows (centers != n).
        let num_knots = 5;
        assert!(num_knots < n, "must exercise the farthest-point selector");

        let knots = select_thin_plate_knots(data.view(), num_knots).expect("select knots");
        assert_eq!(knots.nrows(), num_knots);

        // ONE rigid rotation R about the fixed data centroid, applied to both
        // the full data and the selected subset. Rotating the knots about their
        // OWN centroid instead would be a different map and could never match
        // bit-for-bit even under perfect invariance.
        let (cx, cz) = data_centroid_2d(&data);
        let rotated = rotate_90_about(&data, cx, cz);
        let knots_rot = select_thin_plate_knots(rotated.view(), num_knots).expect("select rotated");

        // The invariant: selecting in the rotated frame yields the SAME physical
        // points as rotating the originally-selected set. With an exact 90 degree
        // rotation this holds to machine precision (bit-identical coordinates).
        let knots_then_rotate = rotate_90_about(&knots, cx, cz);
        assert_eq!(
            canonical(&knots_then_rotate),
            canonical(&knots_rot),
            "rotating-then-selecting must equal selecting-then-rotating (gh#1456); \
             the OLD lexicographic seed picks a different physical point after rotation"
        );
    }

    #[test]
    fn knot_set_is_row_permutation_invariant_gh1378() {
        let data = sample_cloud();
        let n = data.nrows();
        let num_knots = 5;
        assert!(num_knots < n, "must exercise the farthest-point selector");

        let knots = select_thin_plate_knots(data.view(), num_knots).expect("select knots");

        // A non-trivial permutation of the rows (a fixed derangement-ish shuffle).
        let perm: Vec<usize> = vec![7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4, 6];
        assert_eq!(perm.len(), n);
        let mut permuted = Array2::<f64>::zeros((n, 2));
        for (new_row, &old_row) in perm.iter().enumerate() {
            permuted[[new_row, 0]] = data[[old_row, 0]];
            permuted[[new_row, 1]] = data[[old_row, 1]];
        }

        let knots_perm =
            select_thin_plate_knots(permuted.view(), num_knots).expect("select permuted");

        // The selected SET (as physical coordinate tuples) must be bit-identical
        // regardless of input row order (gh#1378).
        assert_eq!(
            canonical(&knots),
            canonical(&knots_perm),
            "reordering rows must not change the selected knot set (gh#1378)"
        );
    }
}

#[cfg(test)]
mod duchon_operator_gate_tests {
    use super::{DuchonOperatorPenaltySpec, OperatorPenaltySpec};

    #[test]
    fn default_duchon_operator_penalties_are_active() {
        let default_spec = DuchonOperatorPenaltySpec::default();

        assert!(
            default_spec.has_active_operator_penalty(),
            "default Duchon terms must bypass the native-only fused radial path"
        );
        assert!(
            matches!(default_spec.mass, OperatorPenaltySpec::Active { .. })
                && matches!(default_spec.tension, OperatorPenaltySpec::Active { .. })
                && matches!(default_spec.stiffness, OperatorPenaltySpec::Disabled),
            "the default is mass+tension active with stiffness disabled"
        );
    }

    #[test]
    fn all_disabled_duchon_operator_penalties_are_native_only() {
        let native_only = DuchonOperatorPenaltySpec::all_disabled();

        assert!(
            !native_only.has_active_operator_penalty(),
            "all_disabled() is the explicit native-Gram-only configuration"
        );
    }
}

#[cfg(test)]
mod retained_radial_indices_tests {
    use super::thin_plate_retained_radial_indices;
    use ndarray::Array1;

    // The eigenvalue spectra below were captured from the live thin-plate
    // builder (`s(x, bs="tp", k=20)`) on the #1271 regression data. They lock
    // in the derived selection behaviour: keep EVERY numerically-real bending
    // mode (matching mgcv, which truncates only at the numerical-rank floor),
    // dropping only sub-floor roundoff dust — no tuned magnitude cutoff.

    #[test]
    fn linear_data_spectrum_keeps_every_mode() {
        // Purely linear DGP: every eigenvalue is far above the numerical floor,
        // so all are genuine curvature directions and must be kept. REML (not
        // basis truncation) is responsible for the EDF on linear data.
        let evals = Array1::from_vec(vec![
            885.4, 119.98, 26.287, 10.030, 5.066, 2.330, 1.3953, 0.67709, 0.46814, 0.34210,
            0.26488, 0.17895, 0.14514,
        ]);
        let keep = thin_plate_retained_radial_indices(&evals);
        assert_eq!(
            keep.len(),
            evals.len(),
            "all numerically real modes must be retained"
        );
    }

    #[test]
    fn lidar_spectrum_keeps_every_real_mode() {
        // Real lidar fit: the smallest eigenvalues (~0.04) are still ~12 orders
        // of magnitude above the numerical floor (K*eps*lambda_max ~ 5e-12), so
        // they are real bending modes and are kept — pruning them by magnitude
        // was the #1271 over-prune that collapsed the nonlinear truth recovery.
        let evals = Array1::from_vec(vec![
            1212.2, 144.94, 37.270, 15.529, 6.0768, 3.5845, 1.8094, 1.1058, 0.73002, 0.43701,
            0.33814, 0.23136, 0.18267, 0.15702, 0.13654, 0.044936, 0.041844, 0.038235,
        ]);
        let keep = thin_plate_retained_radial_indices(&evals);
        assert_eq!(keep.len(), evals.len(), "every above-floor mode is kept");
    }

    #[test]
    fn pure_roundoff_modes_are_dropped() {
        // A mode below the K*eps*lambda_max numerical floor is roundoff dust.
        // Here K=5, lambda_max=1e3 => floor = 5*eps*1e3; put the dust an order
        // of magnitude below that floor.
        let big = 1.0e3;
        let dust = 0.1 * 5.0 * f64::EPSILON * big; // well below the K*eps*max floor
        let evals = Array1::from_vec(vec![big, 100.0, 10.0, 1.0, dust]);
        let keep = thin_plate_retained_radial_indices(&evals);
        assert_eq!(keep.len(), 4, "the sub-floor roundoff mode must be pruned");
        assert!(!keep.contains(&4));
    }

    #[test]
    fn empty_and_singleton_spectra_are_handled() {
        assert!(thin_plate_retained_radial_indices(&Array1::from_vec(vec![])).is_empty());
        assert_eq!(
            thin_plate_retained_radial_indices(&Array1::from_vec(vec![5.0])),
            vec![0]
        );
    }
}

#[cfg(test)]
mod gc_spectrum_diag_1757_tests {
    // ROOT-2 measurement for the perf cluster (#1757 duchon / #1689 thin-plate):
    // does the design Gram Gc = KᵀK (radial kernel evaluated at the selected
    // knots) have a REDUNDANCY CLIFF — a capacity-preserving low-rank truncation
    // à la Wood-2003, where dropping near-duplicate radial columns shrinks the
    // final basis dimension p WITHOUT removing function-space capacity — or only
    // a smooth power-law tail, in which case no magic-free p-reduction exists and
    // the current machine-eps whitening floor already keeps everything meaningful.
    //
    // This is a DIAGNOSTIC (no behavioural assertion beyond "it ran"): it prints
    // the Gc spectrum for the #1757/#1689 repro sizes so CI can grep the shard
    // log. It uses the PRODUCTION knot selector (`select_thin_plate_knots`,
    // farthest-point) and the PRODUCTION thin-plate kernel
    // (`thin_plate_kernel_from_dist2`), so the spectrum matches what the real
    // basis builder forms (the polynomial-null constraint Z removes only 3 dims
    // and cannot create or erase a spectral cliff, so the raw KᵀK Gram answers
    // the redundancy-tail question).
    use super::{select_thin_plate_knots, thin_plate_kernel_from_dist2};
    use crate::basis::default_num_centers;
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;
    use ndarray::Array2;

    // Deterministic uniform scatter in [-1, 1]^2 (SplitMix64; no `rand`
    // dependency, so the printed spectrum is reproducible across machines).
    fn scatter(n: usize, seed: u64) -> Array2<f64> {
        let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            ((z >> 11) as f64) / ((1u64 << 53) as f64) // in [0, 1)
        };
        let mut x = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            x[[i, 0]] = 2.0 * next() - 1.0;
            x[[i, 1]] = 2.0 * next() - 1.0;
        }
        x
    }

    /// Returns `(p, kk, cond)`: number of positive Gram eigenvalues, knot count,
    /// and condition number `λ_max / λ_min⁺`. The caller asserts the design-Gram
    /// invariants (`1 ≤ p ≤ kk`, `cond` finite and `≥ 1`); the printed spectrum is
    /// the diagnostic signal.
    fn report(label: &str, n: usize, seed: u64) -> (usize, usize, f64) {
        let x = scatter(n, seed);
        let k = default_num_centers(n, 2);
        let knots = select_thin_plate_knots(x.view(), k).expect("knot selection");
        let kk = knots.nrows();
        // Design K (n x kk): K[i,c] = phi(||x_i - knot_c||^2), thin-plate d=2.
        let mut kdes = Array2::<f64>::zeros((n, kk));
        for i in 0..n {
            for c in 0..kk {
                let dx = x[[i, 0]] - knots[[c, 0]];
                let dy = x[[i, 1]] - knots[[c, 1]];
                let d2 = dx * dx + dy * dy;
                kdes[[i, c]] = thin_plate_kernel_from_dist2(d2, 2).expect("kernel");
            }
        }
        let gc = kdes.t().dot(&kdes); // kk x kk design Gram
        let (evals, _evecs) = FaerEigh::eigh(&gc, Side::Lower).expect("eigh");
        let mut ev: Vec<f64> = evals.iter().copied().filter(|v| *v > 0.0).collect();
        ev.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
        let m = ev.len();
        if m == 0 {
            eprintln!("[GC-DIAG-1757] {label} n={n}: empty spectrum");
            return (0, kk, f64::INFINITY);
        }
        let lam_max = ev[0];
        let eps_floor = (kk as f64) * f64::EPSILON * lam_max; // current whitening floor
        let eps_kept = ev.iter().filter(|v| **v > eps_floor).count();
        let count_rel = |t: f64| ev.iter().filter(|v| **v / lam_max > t).count();
        // Largest multiplicative gap in the sorted spectrum (eigengap estimator).
        let mut best_gap = 0.0_f64;
        let mut gap_keep = m;
        for j in 0..m - 1 {
            let g = (ev[j] / ev[j + 1]).ln();
            if g > best_gap {
                best_gap = g;
                gap_keep = j + 1;
            }
        }
        eprintln!(
            "[GC-DIAG-1757] {label} n={n} k_req={k} kk={kk} p={m} cond={:.2e} eps_kept={eps_kept} eigengap_keep={gap_keep} log_gap={:.2} | #rel> 1e-2:{} 1e-4:{} 1e-6:{} 1e-8:{} 1e-10:{}",
            lam_max / ev[m - 1],
            best_gap,
            count_rel(1e-2),
            count_rel(1e-4),
            count_rel(1e-6),
            count_rel(1e-8),
            count_rel(1e-10),
        );
        let sampled: Vec<String> = (0..m)
            .step_by((m / 20).max(1))
            .map(|i| format!("{:.1}", (ev[i] / lam_max).log10()))
            .collect();
        eprintln!(
            "[GC-DIAG-1757] {label} log10(rel eigenvalue) sampled: {}",
            sampled.join(" ")
        );
        (m, kk, lam_max / ev[m - 1])
    }

    #[test]
    fn gc_spectrum_duchon_thinplate_repro_sizes() {
        // The redundancy-tail answer is left to the printed spectrum; these are
        // structural design-Gram invariants a broken Gram/knot/kernel would
        // violate (they do NOT presuppose the cliff-vs-power-law verdict).
        for (label, n, seed) in [
            ("duchon_n500", 500usize, 42u64),
            ("duchon_n1220", 1220, 43),
            ("thinplate_n1200", 1200, 7),
        ] {
            let (p, kk, cond) = report(label, n, seed);
            assert!(
                p >= 1 && p <= kk,
                "{label}: positive-eigenvalue count {p} must be in 1..={kk}"
            );
            assert!(
                cond.is_finite() && cond >= 1.0,
                "{label}: condition number {cond} must be finite and >= 1"
            );
        }
    }
}
