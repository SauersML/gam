use super::*;


pub fn build_duchon_basiswithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    if let Some((start, end, _period)) = spec.boundary.period() {
        return build_cyclic_duchon_basis_1dwithworkspace(data, spec, start, end);
    }
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    assert_spatial_centers_below_large_scale_cap(data.ncols(), centers.view())?;
    if spec.periodic.is_some() {
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
    let kernel_transform = kernel_constraint_nullspace(
        centers.view(),
        effective_nullspace_order,
        &mut workspace.cache,
    )?;
    let poly_cols = polynomial_block_from_order(data, effective_nullspace_order).ncols();
    let base_cols = kernel_transform.ncols() + poly_cols;
    let dense_bytes = dense_design_bytes(data.nrows(), base_cols);
    let use_lazy = should_use_lazy_spatial_design(data.nrows(), base_cols, workspace.policy());
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
        let poly_block = polynomial_block_from_order(data, effective_nullspace_order);
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
            let base_op = ChunkedKernelDesignOperator::new(
                shared_data.clone(),
                Arc::new(centers.clone()),
                kernel,
                Some(Arc::new(kernel_transform.clone())),
                Some(Arc::new(poly_block.clone())),
            )
            .map_err(BasisError::InvalidInput)?;
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(base_op)))
        } else {
            let coeffs = coeffs.clone();
            let kernel = move |data_row: &[f64], center_row: &[f64]| -> f64 {
                let r = stable_euclidean_norm((0..d).map(|axis| data_row[axis] - center_row[axis]));
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
            let base_op = ChunkedKernelDesignOperator::new(
                shared_data,
                Arc::new(centers.clone()),
                kernel,
                Some(Arc::new(kernel_transform.clone())),
                Some(Arc::new(poly_block)),
            )
            .map_err(BasisError::InvalidInput)?;
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(base_op)))
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
        let d = build_duchon_basis_designwithworkspace(
            data,
            centers.view(),
            spec.length_scale,
            spec.power,
            effective_nullspace_order,
            aniso.as_deref(),
            workspace,
        )?;
        let basis = d.basis;
        let identifiability_transform = spatial_identifiability_transform_from_design(
            data,
            basis.view(),
            &spec.identifiability,
            "Duchon",
        )?;
        let design = if let Some(z) = identifiability_transform.as_ref() {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(fast_ab(&basis, z)))
        } else {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(basis))
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
        },
        kronecker_factored: None,
    })
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


pub(crate) fn monomial_basis_block(points: ArrayView2<'_, f64>, max_total_degree: usize) -> Array2<f64> {
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

    // Deterministic seed point: lexicographically smallest row.
    let mut seed_idx = 0usize;
    for i in 1..n {
        let mut choose_i = false;
        for c in 0..d {
            let ai = data[[i, c]];
            let as_ = data[[seed_idx, c]];
            if ai < as_ {
                choose_i = true;
                break;
            }
            if ai > as_ {
                break;
            }
        }
        if choose_i {
            seed_idx = i;
        }
    }

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
        let best_idx = min_dist2
            .par_iter()
            .enumerate()
            .filter(|(i, _)| !chosen[*i])
            .map(|(i, &cand)| (i, cand))
            .reduce_with(|a, b| {
                if b.1 > a.1 || (b.1 == a.1 && b.0 < a.0) {
                    b
                } else {
                    a
                }
            })
            .map(|(i, _)| i);
        let next_idx = match best_idx {
            Some(i) => i,
            None => break,
        };
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
pub(crate) fn thin_plate_kernel_from_dist2(dist2: f64, dimension: usize) -> Result<f64, BasisError> {
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

    let kernel_cols = kernel_constrained.ncols();
    let total_cols = kernel_cols + poly_cols;

    // Radial penalty eigenspace reparameterization. Eigendecompose
    // Ω_constrained = V Λ V' and rotate the radial design columns into the
    // same basis. This preserves the TPS model space while making the bending
    // block diagonal.
    let (radial_reparam, radial_eigvals): (Array2<f64>, Array1<f64>) = if let Some(frozen) =
        frozen_radial_reparam
    {
        if frozen.nrows() != kernel_cols || frozen.ncols() != kernel_cols {
            crate::bail_dim_basis!(
                "thin-plate frozen radial reparam shape {:?} does not match radial dimension {}",
                frozen.dim(),
                kernel_cols
            );
        }
        let v = frozen.to_owned();
        let vt_omega_v = fast_atb(&v, &omega_constrained);
        let lambda_diag = fast_ab(&vt_omega_v, &v);
        let mut evals = Array1::<f64>::zeros(kernel_cols);
        for i in 0..kernel_cols {
            evals[i] = lambda_diag[[i, i]].max(0.0);
        }
        (v, evals)
    } else if kernel_cols == 0 {
        (Array2::<f64>::zeros((0, 0)), Array1::<f64>::zeros(0))
    } else {
        let sym = symmetrize_penalty(&omega_constrained);
        let (mut evals, evecs) =
            FaerEigh::eigh(&sym, Side::Lower).map_err(BasisError::LinalgError)?;
        for v in evals.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        (evecs, evals)
    };

    let kernel_rotated = if kernel_cols == 0 {
        kernel_constrained.clone()
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

pub(crate) fn build_thin_plate_penalty_psi_derivativeswithworkspace(
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
    let kernel_cols = z_kernel.ncols();
    let poly_cols = thin_plate_polynomial_basis_dimension(centers.ncols());
    let total_cols = kernel_cols + poly_cols;
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
        i: usize,
        j: usize,
        phi: f64,
        phi_psi: f64,
        phi_psi_psi: f64,
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
        if frozen.nrows() != kernel_cols || frozen.ncols() != kernel_cols {
            crate::bail_dim_basis!(
                "thin-plate frozen radial reparam shape {:?} does not match radial dimension {}",
                frozen.dim(),
                kernel_cols
            );
        }
        let v_owned = frozen.to_owned();
        let lambda_diag = fast_ab(&fast_atb(&v_owned, &m_constrained), &v_owned);
        let mut evals = Array1::<f64>::zeros(kernel_cols);
        for i in 0..kernel_cols {
            evals[i] = lambda_diag[[i, i]].max(0.0);
        }
        (v_owned, evals)
    } else if kernel_cols == 0 {
        (Array2::<f64>::zeros((0, 0)), Array1::<f64>::zeros(0))
    } else {
        let (mut evals, evecs) =
            FaerEigh::eigh(&m_constrained, Side::Lower).map_err(BasisError::LinalgError)?;
        for ev in evals.iter_mut() {
            if *ev < 0.0 {
                *ev = 0.0;
            }
        }
        (evecs, evals)
    };
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
    let kernel_cols = z_kernel.ncols();
    let kernel_transform = if let Some(v) = spec.radial_reparam.as_ref() {
        if v.nrows() != kernel_cols || v.ncols() != kernel_cols {
            crate::bail_dim_basis!(
                "thin-plate radial reparam shape {:?} does not match radial dimension {}",
                v.dim(),
                kernel_cols
            );
        }
        fast_ab(&z_kernel, v)
    } else {
        z_kernel
    };
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
            constrained_gram_max_eigenvalue: f64::NAN,
            constrained_gram_min_eigenvalue: f64::NAN,
            spectral_tolerance: f64::NAN,
        });
    }
    if rank == 0 {
        // Already orthogonal to the intercept constraint; keep full basis unchanged.
        return Ok((basis_matrix.to_owned(), Array2::eye(k)));
    }

    // Constrained basis
    let constrained = fast_ab(&basis_matrix, &z);
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
            constrained_gram_max_eigenvalue: f64::NAN,
            constrained_gram_min_eigenvalue: f64::NAN,
            spectral_tolerance: f64::NAN,
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
            constrained_gram_max_eigenvalue: f64::NAN,
            constrained_gram_min_eigenvalue: f64::NAN,
            spectral_tolerance: f64::NAN,
        });
    }

    if z.ncols() == 0 {
        return Err(BasisError::ConstraintNullspaceCollapsed {
            site: "compute_geometric_constraint_transform",
            cross_rank: 0,
            coeff_dim: k,
            cross_frobenius: f64::NAN,
            constrained_gram_max_eigenvalue: f64::NAN,
            constrained_gram_min_eigenvalue: f64::NAN,
            spectral_tolerance: f64::NAN,
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
