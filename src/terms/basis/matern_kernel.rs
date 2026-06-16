use super::*;

/// Generic thin-plate builder returning design + penalty list.
pub fn build_thin_plate_basis(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_thin_plate_basiswithworkspace(data, spec, &mut workspace)
}

pub fn build_thin_plate_basiswithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &ThinPlateBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisBuildResult, BasisError> {
    let original_centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    let centers = expand_periodic_centers(&original_centers, spec.periodic.as_deref())?;
    // Canonical TPS in dimension d uses penalty order m = ⌊d/2⌋+1 and a
    // polynomial nullspace of size M(d) = C(d+m-1, d). For d=16 this is
    // 735_471, well above any practical knot count. When the requested
    // knot count is below M(d), canonical TPS is mathematically infeasible
    // (the constraint P(C)^T α = 0 is overdetermined). Rather than reject,
    // delegate to a hybrid Matern-Duchon spline — TPS's proper generalization
    // with an additional Riesz fractional smoothness s and a Matern-blended
    // spectrum — using parameters that satisfy Duchon's collocation gates:
    //   2(p + s) > d + max_op   pointwise kernel + collocation existence
    // We pick p = 2 (Linear nullspace, M' = d+1, well below typical k) and
    // the smallest s satisfying the gate. Hybrid (length_scale=Some) is used
    // rather than pure Duchon so the spatial-scale optimizer's log-κ
    // derivatives have a tunable kernel parameter (pure Duchon has none).
    if d_canonical_tps_infeasible(data.ncols(), centers.nrows())
        && let Some((nullspace_order, s)) =
            duchon_thin_plate_fallback_params(data.ncols(), centers.nrows())
    {
        let d = data.ncols();
        // The hybrid-Duchon partial-fraction kernel coefficients scale as
        // `kappa^(-2(p+s-n)) = length_scale^(2(p+s-n))` (see
        // `duchon_partial_fraction_coeffs`). With the high spectral order `s`
        // this auto-promotion selects (s ≥ 3 for d ≥ 6) and the Matern-style
        // auto-init length_scale (`max_range / sqrt(n)`, which is far below the
        // center spacing for moderate n), kappa·r runs large at every center
        // pair, every kernel block underflows toward machine epsilon, and the
        // constrained radial Gram collapses to floating-point noise
        // (`positive_spectral_whitener_from_gram` then rejects a rank-0 smooth —
        // gam#1091). The natural operating scale of a radial kernel is the
        // typical center separation, where kappa·r ≈ O(1) keeps every block
        // O(1); promote at that scale rather than inheriting the (possibly
        // tiny) Matern init. The outer optimizer still tunes psi = log kappa
        // from here, but it now starts from a non-degenerate basis.
        let promotion_length_scale =
            hybrid_duchon_promotion_length_scale(centers.view(), spec.length_scale);
        let duchon_spec = DuchonBasisSpec {
            center_strategy: CenterStrategy::UserProvided(original_centers.clone()),
            periodic: spec.periodic.clone(),
            length_scale: Some(promotion_length_scale),
            power: s as f64,
            nullspace_order,
            identifiability: spec.identifiability.clone(),
            aniso_log_scales: None,
            operator_penalties: DuchonOperatorPenaltySpec::default(),
            boundary: OneDimensionalBoundary::Open,
        };
        log::info!(
            "thin-plate basis auto-promoted to hybrid Duchon ({:?}, s={}) in d={}: \
             canonical TPS would need {} centers but got {} — using Duchon's \
             Riesz-fractional generalization with finite kernel at r=0 \
             (length_scale={:.4e} from center spacing, was {:.4e})",
            nullspace_order,
            s,
            d,
            thin_plate_polynomial_basis_dimension(d),
            centers.nrows(),
            promotion_length_scale,
            spec.length_scale,
        );
        return build_duchon_basiswithworkspace(data, &duchon_spec, workspace);
    }
    let internal_kernel_transform =
        thin_plate_kernel_constraint_nullspace(centers.view(), &mut workspace.cache)?;
    let poly_cols = thin_plate_polynomial_basis_dimension(centers.ncols());
    let base_cols = internal_kernel_transform.ncols() + poly_cols;
    let dense_bytes = dense_design_bytes(data.nrows(), base_cols);
    let use_lazy = should_use_lazy_spatial_design(data.nrows(), base_cols, workspace.policy());
    if use_lazy {
        // log::info! — deliberate memory-saving choice, not an anomaly.
        log::info!(
            "thin-plate basis switching to lazy chunked design: n={} p={} ({:.1} MiB dense)",
            data.nrows(),
            base_cols,
            dense_bytes as f64 / (1024.0 * 1024.0),
        );
    }
    let (design, identifiability_transform, mut candidates, radial_reparam_meta) = if use_lazy {
        let poly_block = thin_plate_polynomial_block(data);
        let d = data.ncols();
        let length_scale_sq = spec.length_scale * spec.length_scale;
        let shared_data = shared_owned_data_matrix(data, &workspace.cache);
        let kernel_fn = move |data_row: &[f64], center_row: &[f64]| -> f64 {
            let mut dist2 = 0.0;
            for axis in 0..d {
                let delta = data_row[axis] - center_row[axis];
                dist2 += delta * delta;
            }
            thin_plate_kernel_from_dist2(dist2 / length_scale_sq, d)
                .expect("validated thin-plate inputs should not fail")
        };
        let base_op = ChunkedKernelDesignOperator::new(
            shared_data,
            Arc::new(centers.clone()),
            kernel_fn,
            Some(Arc::new(internal_kernel_transform.clone())),
            Some(Arc::new(poly_block)),
        )
        .map_err(BasisError::InvalidInput)?;
        let base_design =
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(base_op)));
        let identifiability_transform = thin_plate_identifiability_transform_from_design_matrix(
            &base_design,
            internal_kernel_transform.ncols(),
            poly_cols,
            &spec.identifiability,
        )?;
        let design = if let Some(transform) = identifiability_transform.as_ref() {
            wrap_dense_design_with_transform(base_design, transform, "ThinPlate")?
        } else {
            base_design
        };
        let (penalty_bending, penalty_ridge) = build_thin_plate_penalty_matrices(
            centers.view(),
            spec.length_scale,
            &internal_kernel_transform,
            spec.double_penalty,
        )?;
        let (penalty_bending_norm, c_bending) = normalize_penalty(&penalty_bending);
        let mut candidates = vec![PenaltyCandidate {
            matrix: penalty_bending_norm,
            nullspace_dim_hint: poly_cols,
            source: PenaltySource::Primary,
            normalization_scale: c_bending,
            kronecker_factors: None,
            op: None,
        }];
        if let Some(penalty_ridge) = penalty_ridge {
            let (penalty_ridge_norm, c_ridge) = normalize_penalty(&penalty_ridge);
            candidates.push(PenaltyCandidate {
                matrix: penalty_ridge_norm,
                nullspace_dim_hint: 0,
                source: PenaltySource::DoublePenaltyNullspace,
                normalization_scale: c_ridge,
                kronecker_factors: None,
                op: None,
            });
        }
        (design, identifiability_transform, candidates, None)
    } else {
        let tps = create_thin_plate_spline_basis_scaledwithworkspace(
            data,
            centers.view(),
            spec.length_scale,
            spec.radial_reparam.as_ref(),
            workspace,
        )?;
        let identifiability_transform = thin_plate_identifiability_transform_from_design(
            tps.basis.view(),
            tps.num_kernel_basis,
            tps.num_polynomial_basis,
            &spec.identifiability,
        )?;
        let design = if let Some(z) = identifiability_transform.as_ref() {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(fast_ab(
                &tps.basis, z,
            )))
        } else {
            DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(tps.basis.clone()))
        };
        let (penalty_bending_norm, c_bending) = normalize_penalty(&tps.penalty_bending);
        let mut candidates = vec![PenaltyCandidate {
            matrix: penalty_bending_norm,
            nullspace_dim_hint: tps.num_polynomial_basis,
            source: PenaltySource::Primary,
            normalization_scale: c_bending,
            kronecker_factors: None,
            op: None,
        }];
        if spec.double_penalty {
            let (penalty_ridge_norm, c_ridge) = normalize_penalty(&tps.penalty_ridge);
            candidates.push(PenaltyCandidate {
                matrix: penalty_ridge_norm,
                nullspace_dim_hint: 0,
                source: PenaltySource::DoublePenaltyNullspace,
                normalization_scale: c_ridge,
                kronecker_factors: None,
                op: None,
            });
        }
        let radial_reparam_meta = Some(tps.radial_reparam.clone());
        (
            design,
            identifiability_transform,
            candidates,
            radial_reparam_meta,
        )
    };
    if let Some(z) = identifiability_transform.as_ref() {
        candidates = candidates
            .into_iter()
            .map(|candidate| -> Result<PenaltyCandidate, BasisError> {
                let zt_s = z.t().dot(&candidate.matrix);
                let matrix = zt_s.dot(z);
                Ok(PenaltyCandidate {
                    nullspace_dim_hint: candidate.nullspace_dim_hint,
                    matrix,
                    source: candidate.source,
                    normalization_scale: candidate.normalization_scale,
                    kronecker_factors: None,
                    op: None,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
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
        metadata: BasisMetadata::ThinPlate {
            centers: original_centers,
            length_scale: spec.length_scale,
            periodic: spec.periodic.clone(),
            identifiability_transform,
            input_scales: None,
            radial_reparam: radial_reparam_meta,
        },
        kronecker_factored: None,
    })
}

/// Canonical domain guard for Matérn kernel evaluations: distance `r` must be
/// finite and non-negative, length scale must be finite and positive. Single
/// source of truth for the `(r, length_scale)` validity check shared by every
/// Matérn kernel/derivative function below.
#[inline(always)]
pub(crate) fn validate_matern_inputs(r: f64, length_scale: f64) -> Result<(), BasisError> {
    if !r.is_finite() || r < 0.0 {
        crate::bail_invalid_basis!("Matérn kernel distance must be finite and non-negative");
    }
    validate_matern_length_scale(length_scale)
}

/// Canonical guard for the length-scale-only Matérn sites: length scale must be
/// finite and positive. Shared by `validate_matern_inputs` and by callers that
/// validate the distance separately (or have no distance argument).
#[inline(always)]
pub(crate) fn validate_matern_length_scale(length_scale: f64) -> Result<(), BasisError> {
    if !length_scale.is_finite() || length_scale <= 0.0 {
        crate::bail_invalid_basis!("Matérn length_scale must be finite and positive");
    }
    Ok(())
}

#[inline(always)]
pub(crate) fn matern_kernel_from_distance(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<f64, BasisError> {
    validate_matern_inputs(r, length_scale)?;

    // Parameterization used here:
    //   x = r / length_scale
    //   a = sqrt(2ν) * x
    // and the half-integer Matérn closed forms are in terms of `a`:
    //   ν=1/2: exp(-a)
    //   ν=3/2: (1+a) exp(-a)
    //   ν=5/2: (1+a+a^2/3) exp(-a)
    // (for ν=1/2, a=x since sqrt(2ν)=1).
    let x = r / length_scale;
    let k = match nu {
        MaternNu::Half => stable_nonnegative_poly_times_exp_neg(x, &[1.0]),
        MaternNu::ThreeHalves => {
            let a = 3.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[1.0, 1.0])
        }
        MaternNu::FiveHalves => {
            let a = 5.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[1.0, 1.0, 1.0 / 3.0])
        }
        MaternNu::SevenHalves => {
            let a = 7.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[1.0, 1.0, 2.0 / 5.0, 1.0 / 15.0])
        }
        MaternNu::NineHalves => {
            let a = 9.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[1.0, 1.0, 3.0 / 7.0, 2.0 / 21.0, 1.0 / 105.0],
            )
        }
    };
    Ok(k)
}

#[inline(always)]
pub(crate) fn matern_kernel_log_kappa_derivative_from_distance(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<f64, BasisError> {
    validate_matern_inputs(r, length_scale)?;

    let x = r / length_scale;
    let deriv = match nu {
        MaternNu::Half => stable_nonnegative_poly_times_exp_neg(x, &[0.0, -1.0]),
        MaternNu::ThreeHalves => {
            let a = 3.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[0.0, 0.0, -1.0])
        }
        MaternNu::FiveHalves => {
            let a = 5.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[0.0, 0.0, -1.0 / 3.0, -1.0 / 3.0])
        }
        MaternNu::SevenHalves => {
            let a = 7.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[0.0, 0.0, -1.0 / 5.0, -1.0 / 5.0, -1.0 / 15.0],
            )
        }
        MaternNu::NineHalves => {
            let a = 9.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[0.0, 0.0, -1.0 / 7.0, -1.0 / 7.0, -2.0 / 35.0, -1.0 / 105.0],
            )
        }
    };
    Ok(deriv)
}

#[inline(always)]
pub(crate) fn matern_kernel_log_kappasecond_derivative_from_distance(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<f64, BasisError> {
    validate_matern_inputs(r, length_scale)?;

    let x = r / length_scale;
    let second = match nu {
        MaternNu::Half => stable_nonnegative_poly_times_exp_neg(x, &[0.0, -1.0, 1.0]),
        MaternNu::ThreeHalves => {
            let a = 3.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[0.0, 0.0, -2.0, 1.0])
        }
        MaternNu::FiveHalves => {
            let a = 5.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(a, &[0.0, 0.0, -2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0])
        }
        MaternNu::SevenHalves => {
            let a = 7.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[0.0, 0.0, -2.0 / 5.0, -2.0 / 5.0, -1.0 / 15.0, 1.0 / 15.0],
            )
        }
        MaternNu::NineHalves => {
            let a = 9.0_f64.sqrt() * x;
            stable_nonnegative_poly_times_exp_neg(
                a,
                &[
                    0.0,
                    0.0,
                    -2.0 / 7.0,
                    -2.0 / 7.0,
                    -3.0 / 35.0,
                    1.0 / 105.0,
                    1.0 / 105.0,
                ],
            )
        }
    };
    Ok(second)
}

#[inline(always)]
pub(crate) fn matern_kernel_radial_tripletwith_safe_ratio(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<(f64, f64, f64, f64), BasisError> {
    validate_matern_inputs(r, length_scale)?;

    // Full derivation used by collocation operators:
    //   phi(r) = P_nu(a) exp(-a), a=sr, s=sqrt(2nu)/length_scale.
    // For nu>=3/2 we use closed-form phi'(r)/r polynomials with finite r->0 limit.
    // For nu=1/2:
    //   phi'(r)/r = -kappa exp(-kappa r)/r,
    // which is genuinely singular at r=0 and must not be regularized here.
    // Closed forms used below (a = s r, E = exp(-a)):
    // nu=1/2:
    //   phi'    = -s E
    //   phi''   =  s^2 E
    //   phi'/r  diverges as -s/r (regularized via r floor).
    // nu=3/2:
    //   phi'    = -s E a
    //   phi''   =  s^2 E (a-1)
    //   phi'/r  = -s^2 E.
    // nu=5/2:
    //   phi'    = -(s/3) E a(a+1)
    //   phi''   =  (s^2/3) E (a^2-a-1)
    //   phi'/r  = -(s^2/3) E (a+1).
    // nu=7/2:
    //   phi'    = -(s/15) E a(a^2+3a+3)
    //   phi''   =  (s^2/15) E (a^3-3a-3)
    //   phi'/r  = -(s^2/15) E (a^2+3a+3).
    // nu=9/2:
    //   phi'    = -(s/105) E a(a^3+6a^2+15a+15)
    //   phi''   =  (s^2/105) E (a^4+2a^3-3a^2-15a-15)
    //   phi'/r  = -(s^2/105) E (a^3+6a^2+15a+15).
    let (phi, phi_r, phi_rr, phi_r_over_r) = match nu {
        MaternNu::Half => {
            let s = 1.0 / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = e;
            let phi_r = -s * e;
            let phi_rr = s * s * e;
            // Safe ratio regularization at r=0 to keep operator assembly finite.
            let r_eff = r.max(1e-12);
            let ratio = phi_r / r_eff;
            (phi, phi_r, phi_rr, ratio)
        }
        MaternNu::ThreeHalves => {
            let s = 3.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a) * e;
            let phi_r = -s * e * a;
            let phi_rr = s * s * e * (a - 1.0);
            let ratio = -s * s * e;
            (phi, phi_r, phi_rr, ratio)
        }
        MaternNu::FiveHalves => {
            let s = 5.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (a * a) / 3.0) * e;
            let phi_r = -(s / 3.0) * e * a * (a + 1.0);
            let phi_rr = (s * s / 3.0) * e * (a * a - a - 1.0);
            let ratio = -(s * s / 3.0) * e * (a + 1.0);
            (phi, phi_r, phi_rr, ratio)
        }
        MaternNu::SevenHalves => {
            let s = 7.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (2.0 / 5.0) * a * a + (1.0 / 15.0) * a * a * a) * e;
            let phi_r = -(s / 15.0) * e * a * (a * a + 3.0 * a + 3.0);
            let phi_rr = (s * s / 15.0) * e * (a * a * a - 3.0 * a - 3.0);
            let ratio = -(s * s / 15.0) * e * (a * a + 3.0 * a + 3.0);
            (phi, phi_r, phi_rr, ratio)
        }
        MaternNu::NineHalves => {
            let s = 9.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0
                + a
                + (3.0 / 7.0) * a * a
                + (2.0 / 21.0) * a * a * a
                + (1.0 / 105.0) * a * a * a * a)
                * e;
            let phi_r = -(s / 105.0) * e * a * (a * a * a + 6.0 * a * a + 15.0 * a + 15.0);
            let phi_rr = (s * s / 105.0)
                * e
                * (a * a * a * a + 2.0 * a * a * a - 3.0 * a * a - 15.0 * a - 15.0);
            let ratio = -(s * s / 105.0) * e * (a * a * a + 6.0 * a * a + 15.0 * a + 15.0);
            (phi, phi_r, phi_rr, ratio)
        }
    };

    if !phi.is_finite() || !phi_r.is_finite() || !phi_rr.is_finite() || !phi_r_over_r.is_finite() {
        crate::bail_invalid_basis!(
            "non-finite Matérn radial derivatives at r={r}, length_scale={length_scale}, nu={nu:?}"
        );
    }
    Ok((phi, phi_r, phi_rr, phi_r_over_r))
}

/// Extended radial scalars for exact per-axis eta_a derivatives of the Matérn
/// operator collocation matrices D1 (gradient) and D2 (full Hessian).
///
/// Returns `(phi, q, t, dt_dr, d2t_dr2)` where:
///   - phi   = φ(r)                  (kernel value)
///   - q     = φ'(r)/r               (used in D₁)
///   - t     = (phi''(r) - q) / r^2  (Hessian mixed-curvature scalar)
///   - dt_dr = dt/dr                 (needed for second eta-derivatives)
///   - d2t_dr2 = d2t/dr2             (needed for second eta-derivatives)
///
/// At r = 0 (center collision), the function returns zeros for all quantities
/// that would be multiplied by s_a (which also vanishes at collision).
///
/// For ν = 1/2 and ν = 3/2 where t and/or dt_dr diverge at r = 0, the
/// collision entries are safe because D₁ and D₂ derivatives at coincident
/// centers vanish via s_a = 0.
pub(crate) fn matern_aniso_extended_radial_scalars(
    r: f64,
    length_scale: f64,
    nu: MaternNu,
) -> Result<(f64, f64, f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        crate::bail_invalid_basis!(
            "Matérn extended radial scalar distance must be finite and non-negative"
        );
    }
    validate_matern_length_scale(length_scale)?;

    match nu {
        // ----------------------------------------------------------------
        // ν = 1/2:  φ = exp(-a), a = r / ℓ, s = 1/ℓ
        //   q = -s·E/r  (diverges at r=0)
        //   t = (s²E - q) / r²  (diverges at r=0)
        //   At r=0 all products with s_a vanish, so return 0 for dt_dr, d2t_dr2.
        // ----------------------------------------------------------------
        MaternNu::Half => {
            let s = 1.0 / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = e;
            if r < 1e-14 {
                // Center collision. φ(r) = exp(−s r) has a cusp at r = 0, so
                // the radial scalars q = φ'/r and t = (φ'' − q)/r² diverge.
                // But every consumer multiplies them by displacement factors
                // that vanish identically at a coincident center:
                //   * the design-matrix η-derivatives are q·s_a and t·s_a·s_b
                //     (true value 0 — φ ≡ 1 there, independent of length scale);
                //   * the operator-collocation gradient row is q·h_b (h_b = 0);
                //   * the only term not pre-multiplied by a vanishing factor is
                //     the D₂ operator diagonal q·w_b, which the *value* path
                //     defines via the same convention — `phi_r_over_r = 0` for
                //     ν = 1/2 in 1D (the 1D Laplacian Δφ = φ'' carries no φ'/r
                //     term; see the base assembly below), and bails for d ≥ 2
                //     (already rejected at term construction).
                // Returning the convention-consistent zeros keeps the analytic
                // κ-gradient in lockstep with its own value surface — mirroring
                // the ν = 3/2 branch — rather than hard-erroring on a quantity
                // that is multiplied away.
                return Ok((phi, 0.0, 0.0, 0.0, 0.0));
            }
            let q = -s * e / r;
            let phi_rr = s * s * e;
            let t = (phi_rr - q) / (r * r);
            // t' from: t = f/r² where f = φ'' - q.
            //   f'  = φ''' - q' = -s³E - t·r   (since q' = t·r)
            //   t'  = (f' - 2t·r) / r²  = (-s³E - 3t·r) / r²
            let dt_dr = (-s * s * s * e - 3.0 * t * r) / (r * r);
            // t'' from: t' = g/r² where g = -s³E - 3tr.
            //   g' = s⁴E - 3(t'r + t)
            //   t'' = (g' - 2t'r) / r² = (s⁴E - 3t'r - 3t - 2t'r) / r²
            //        = (s⁴E - 5t'r - 3t) / r²
            let d2t_dr2 = (s.powi(4) * e - 5.0 * dt_dr * r - 3.0 * t) / (r * r);
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
        // ----------------------------------------------------------------
        // ν = 3/2:  φ = (1 + a)E, a = √3·r/ℓ, s = √3/ℓ
        //   q  = -s²E         (finite at r=0)
        //   t  = s³E/r        (diverges at r=0)
        //   dt/dr = s³E(-sr - 1)/r²  (diverges at r=0)
        //   At r=0, s_a = 0 so all products vanish.
        // ----------------------------------------------------------------
        MaternNu::ThreeHalves => {
            let s = 3.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a) * e;
            let q = -s * s * e;
            if r < 1e-14 {
                return Ok((phi, q, 0.0, 0.0, 0.0));
            }
            let t = s * s * s * e / r;
            // dt/dr: d/dr [s³ E / r] = s³ [-s E r - E] / r² = -s³ E (sr + 1) / r²
            let dt_dr = -s * s * s * e * (a + 1.0) / (r * r);
            // d²t/dr²: d/dr [-s³ E (a+1) / r²]
            //   = -s³ [(-s E)(a+1)r² + s E r² - 2r E(a+1)] / r⁴ ... expand
            // Let g(r) = -s³ E (a+1) / r²
            // g'(r) = -s³ [E'(a+1) + E·s] / r² + 2s³ E(a+1) / r³
            //       = -s³ [-sE(a+1) + sE] / r² + 2s³ E(a+1) / r³
            //       = -s³ · sE[-a-1+1] / r² + 2s³ E(a+1) / r³
            //       = s⁴ a E / r² + 2s³ E(a+1) / r³
            //       = s³ E [s a r + 2(a+1)] / r³
            let d2t_dr2 = s * s * s * e * (s * a * r + 2.0 * (a + 1.0)) / (r * r * r);
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
        // ----------------------------------------------------------------
        // ν = 5/2:  φ = (1 + a + a²/3)E, a = √5·r/ℓ, s = √5/ℓ
        //   q = -(s²/3)(a+1)E
        //   t = (s⁴/3)E
        //   dt/dr = -(s⁵/3)E
        //   d²t/dr² = (s⁶/3)E
        // ----------------------------------------------------------------
        MaternNu::FiveHalves => {
            let s = 5.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (a * a) / 3.0) * e;
            let q = -(s * s / 3.0) * (a + 1.0) * e;
            let t = (s * s * s * s / 3.0) * e;
            let dt_dr = -(s * s * s * s * s / 3.0) * e;
            let d2t_dr2 = (s.powi(6) / 3.0) * e;
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
        // ----------------------------------------------------------------
        // ν = 7/2:  φ = (1 + a + (2/5)a² + (1/15)a³)E
        //   q = -(s²/15)(a² + 3a + 3)E
        //   t = (s⁴/15)(a + 1)E
        //   dt/dr = -(s⁵/15)aE
        //   d²t/dr² = (s⁶/15)(a - 1)E
        // ----------------------------------------------------------------
        MaternNu::SevenHalves => {
            let s = 7.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0 + a + (2.0 / 5.0) * a * a + (1.0 / 15.0) * a * a * a) * e;
            let q = -(s * s / 15.0) * (a * a + 3.0 * a + 3.0) * e;
            let t = (s * s * s * s / 15.0) * (a + 1.0) * e;
            let dt_dr = -(s.powi(5) / 15.0) * a * e;
            let d2t_dr2 = (s.powi(6) / 15.0) * (a - 1.0) * e;
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
        // ----------------------------------------------------------------
        // ν = 9/2:  φ = (1 + a + (3/7)a² + (2/21)a³ + (1/105)a⁴)E
        //   q = -(s²/105)(a³ + 6a² + 15a + 15)E
        //   t = (s⁴/105)(a² + 3a + 3)E
        //   dt/dr = -(s⁵/105)a(a + 1)E
        //   d²t/dr² = (s⁶/105)(a² - a - 1)E
        // ----------------------------------------------------------------
        MaternNu::NineHalves => {
            let s = 9.0_f64.sqrt() / length_scale;
            let a = s * r;
            let e = (-a).exp();
            let phi = (1.0
                + a
                + (3.0 / 7.0) * a * a
                + (2.0 / 21.0) * a * a * a
                + (1.0 / 105.0) * a * a * a * a)
                * e;
            let q = -(s * s / 105.0) * (a * a * a + 6.0 * a * a + 15.0 * a + 15.0) * e;
            let t = (s * s * s * s / 105.0) * (a * a + 3.0 * a + 3.0) * e;
            let dt_dr = -(s.powi(5) / 105.0) * a * (a + 1.0) * e;
            let d2t_dr2 = (s.powi(6) / 105.0) * (a * a - a - 1.0) * e;
            Ok((phi, q, t, dt_dr, d2t_dr2))
        }
    }
}

#[inline(always)]
pub(crate) fn hessian_operator_entry(
    q: f64,
    t: f64,
    h_b: f64,
    h_c: f64,
    w_b: f64,
    w_c: f64,
    axis_b: usize,
    axis_c: usize,
) -> f64 {
    let diagonal = if axis_b == axis_c { w_b * q } else { 0.0 };
    diagonal + (w_b * h_b) * (w_c * h_c) * t
}

#[inline(always)]
pub(crate) fn hessian_operator_eta_entry(
    q: f64,
    t: f64,
    t_r: f64,
    r: f64,
    s_a: f64,
    h_b: f64,
    h_c: f64,
    w_b: f64,
    w_c: f64,
    axis_a: usize,
    axis_b: usize,
    axis_c: usize,
) -> f64 {
    let a_is_b = usize::from(axis_a == axis_b) as f64;
    let a_is_c = usize::from(axis_a == axis_c) as f64;
    let q_a = t * s_a;
    let t_a = if r > 1e-14 { t_r * s_a / r } else { 0.0 };
    let diagonal = if axis_b == axis_c {
        w_b * (2.0 * a_is_b * q + q_a)
    } else {
        0.0
    };
    let mixed_multiplier = 2.0 * a_is_b + 2.0 * a_is_c;
    diagonal + (w_b * h_b) * (w_c * h_c) * (mixed_multiplier * t + t_a)
}

#[inline(always)]
pub(crate) fn hessian_operator_eta2_entry(
    q: f64,
    t: f64,
    t_r: f64,
    t_rr: f64,
    r: f64,
    s_a: f64,
    h_b: f64,
    h_c: f64,
    w_b: f64,
    w_c: f64,
    axis_a: usize,
    axis_b: usize,
    axis_c: usize,
) -> f64 {
    let a_is_b = usize::from(axis_a == axis_b) as f64;
    let a_is_c = usize::from(axis_a == axis_c) as f64;
    let q_a = t * s_a;
    let q_aa = if r > 1e-14 {
        t_r * s_a * s_a / r + 2.0 * t * s_a
    } else {
        0.0
    };
    let t_a = if r > 1e-14 { t_r * s_a / r } else { 0.0 };
    let t_aa = if r > 1e-14 {
        ((t_rr * r - t_r) / (r * r * r)) * s_a * s_a + 2.0 * t_r * s_a / r
    } else {
        0.0
    };
    let diagonal = if axis_b == axis_c {
        w_b * (4.0 * a_is_b * q + 4.0 * a_is_b * q_a + q_aa)
    } else {
        0.0
    };
    let mixed_multiplier = 2.0 * a_is_b + 2.0 * a_is_c;
    diagonal
        + (w_b * h_b)
            * (w_c * h_c)
            * (mixed_multiplier * mixed_multiplier * t + 2.0 * mixed_multiplier * t_a + t_aa)
}

#[inline(always)]
pub(crate) fn hessian_operator_eta_cross_entry(
    t: f64,
    t_r: f64,
    t_rr: f64,
    r: f64,
    s_i: f64,
    s_j: f64,
    h_b: f64,
    h_c: f64,
    w_b: f64,
    w_c: f64,
    axis_i: usize,
    axis_j: usize,
    axis_b: usize,
    axis_c: usize,
) -> f64 {
    assert_ne!(axis_i, axis_j);
    let i_is_b = usize::from(axis_i == axis_b) as f64;
    let i_is_c = usize::from(axis_i == axis_c) as f64;
    let j_is_b = usize::from(axis_j == axis_b) as f64;
    let j_is_c = usize::from(axis_j == axis_c) as f64;
    let q_i = t * s_i;
    let q_j = t * s_j;
    let q_ij = if r > 1e-14 { t_r * s_i * s_j / r } else { 0.0 };
    let t_i = if r > 1e-14 { t_r * s_i / r } else { 0.0 };
    let t_j = if r > 1e-14 { t_r * s_j / r } else { 0.0 };
    let t_ij = if r > 1e-14 {
        ((t_rr * r - t_r) / (r * r * r)) * s_i * s_j
    } else {
        0.0
    };
    let diagonal = if axis_b == axis_c {
        w_b * (2.0 * i_is_b * q_j + 2.0 * j_is_b * q_i + q_ij)
    } else {
        0.0
    };
    let m_i = 2.0 * i_is_b + 2.0 * i_is_c;
    let m_j = 2.0 * j_is_b + 2.0 * j_is_c;
    diagonal + (w_b * h_b) * (w_c * h_c) * (m_i * m_j * t + m_i * t_j + m_j * t_i + t_ij)
}

/// Build exact per-axis η_a derivatives of operator penalty matrices for
/// anisotropic Matérn terms.
///
/// Instead of the fractional approximation `dS_op/dη_a ≈ f_a · dS_op/dψ`,
/// this computes exact first and second η_a derivatives of each operator
/// collocation matrix (D₀, D₁, D₂) and assembles the Gram product-rule
/// derivatives:
///   S_{m,a}  = D_{m,a}ᵀ D_m + D_mᵀ D_{m,a}
///   S_{m,aa} = D_{m,aa}ᵀ D_m + 2 D_{m,a}ᵀ D_{m,a} + D_mᵀ D_{m,aa}
///
/// ## Per-axis derivative formulas (y-space operators)
///
/// With r = √(Σ exp(2η_a) h_a²) and s_a = exp(2η_a) h_a²:
///
/// **D₀[k,j] = φ(r):**
///   ∂φ/∂η_a = q · s_a
///   ∂²φ/∂η_a² = t · s_a² + 2q · s_a
///
/// **D₁[(k,b),j] = q(r) · h_b** (y-space gradient):
///   ∂D₁/∂η_a = t · s_a · h_b
///   ∂²D₁/∂η_a² = (dt/dr · s_a²/r + 2t · s_a) · h_b
///
/// **D₂[k,j] = φ''(r) + (d-1)·q(r)** (y-space Laplacian):
///   ∂D₂/∂η_a = [(d+2)·t + dt/dr · r] · s_a
///   ∂²D₂/∂η_a² = [(d+3)·dt/dr/r + d²t/dr²] · s_a² + 2·[(d+2)·t + dt/dr·r] · s_a
pub(crate) struct MaternCrossPenaltyContext {
    pub(crate) centers: Array2<f64>,
    pub(crate) aniso_log_scales: Vec<f64>,
    pub(crate) length_scale: f64,
    pub(crate) nu: MaternNu,
    pub(crate) z_transform: Option<Array2<f64>>,
    pub(crate) penaltyinfo: Vec<PenaltyInfo>,
    pub(crate) d0: Array2<f64>,
    pub(crate) d1: Array2<f64>,
    pub(crate) d2: Array2<f64>,
    pub(crate) d0_eta_proj: Vec<Array2<f64>>,
    pub(crate) d1_eta_proj: Vec<Array2<f64>>,
    pub(crate) d2_eta_proj: Vec<Array2<f64>>,
    pub(crate) op0_s_raw: Array2<f64>,
    pub(crate) op1_s_raw: Array2<f64>,
    pub(crate) op2_s_raw: Array2<f64>,
    pub(crate) op0_c: f64,
    pub(crate) op1_c: f64,
    pub(crate) op2_c: f64,
    pub(crate) op0_s_first_raw: Vec<Array2<f64>>,
    pub(crate) op1_s_first_raw: Vec<Array2<f64>>,
    pub(crate) op2_s_first_raw: Vec<Array2<f64>>,
}

impl MaternCrossPenaltyContext {
    pub(crate) fn project_operator(&self, mat: &Array2<f64>, row_dim: usize) -> Array2<f64> {
        let kernel = if let Some(z) = self.z_transform.as_ref() {
            fast_ab(mat, z)
        } else {
            mat.clone()
        };
        let mut padded = Array2::<f64>::zeros((row_dim, self.d0.ncols()));
        padded.slice_mut(s![.., 0..kernel.ncols()]).assign(&kernel);
        padded
    }

    pub(crate) fn compute_pair(
        &self,
        axis_a: usize,
        axis_b: usize,
    ) -> Result<Vec<Array2<f64>>, BasisError> {
        let p = self.centers.nrows();
        let d = self.centers.ncols();
        let mut d0_cross_raw = Array2::<f64>::zeros((p, p));
        let mut d1_cross_raw = Array2::<f64>::zeros((p * d, p));
        let mut d2_cross_raw = Array2::<f64>::zeros((p * d * d, p));
        let metric_weights = centered_aniso_metric_weights(&self.aniso_log_scales);

        for k in 0..p {
            for j in 0..p {
                let ci: Vec<f64> = (0..d).map(|axis| self.centers[[k, axis]]).collect();
                let cj: Vec<f64> = (0..d).map(|axis| self.centers[[j, axis]]).collect();
                let (r, s_vec) = aniso_distance_and_components(&ci, &cj, &self.aniso_log_scales);
                let (_, _, t, dt_dr, d2t_dr2) =
                    matern_aniso_extended_radial_scalars(r, self.length_scale, self.nu)?;
                let s_a = s_vec[axis_a];
                let s_b = s_vec[axis_b];
                let sa_sb = s_a * s_b;

                d0_cross_raw[[k, j]] = t * sa_sb;
                for axis in 0..d {
                    let h_axis = ci[axis] - cj[axis];
                    let w_axis = metric_weights[axis];
                    let row = k * d + axis;
                    d1_cross_raw[[row, j]] = if r > 1e-14 {
                        dt_dr * sa_sb / r * h_axis
                            + if axis == axis_a {
                                2.0 * t * s_b * h_axis
                            } else {
                                0.0
                            }
                            + if axis == axis_b {
                                2.0 * t * s_a * h_axis
                            } else {
                                0.0
                            }
                    } else {
                        0.0
                    } * w_axis;
                }
                for b in 0..d {
                    let h_b = ci[b] - cj[b];
                    let w_b = metric_weights[b];
                    for c in 0..d {
                        let h_c = ci[c] - cj[c];
                        let w_c = metric_weights[c];
                        let row = (k * d + b) * d + c;
                        d2_cross_raw[[row, j]] = hessian_operator_eta_cross_entry(
                            t, dt_dr, d2t_dr2, r, s_a, s_b, h_b, h_c, w_b, w_c, axis_a, axis_b, b,
                            c,
                        );
                    }
                }
            }
        }

        let d0_cross_proj = self.project_operator(&d0_cross_raw, p);
        let d1_cross_proj = self.project_operator(&d1_cross_raw, p * d);
        let d2_cross_proj = self.project_operator(&d2_cross_raw, p * d * d);

        let s0_cross = normalize_penalty_cross_psi_derivative(
            &self.op0_s_raw,
            &self.op0_s_first_raw[axis_a],
            &self.op0_s_first_raw[axis_b],
            &gram_cross_psi_derivative_from_operator(
                &self.d0,
                &self.d0_eta_proj[axis_a],
                &self.d0_eta_proj[axis_b],
                &d0_cross_proj,
            ),
            self.op0_c,
        );
        let s1_cross = normalize_penalty_cross_psi_derivative(
            &self.op1_s_raw,
            &self.op1_s_first_raw[axis_a],
            &self.op1_s_first_raw[axis_b],
            &gram_cross_psi_derivative_from_operator(
                &self.d1,
                &self.d1_eta_proj[axis_a],
                &self.d1_eta_proj[axis_b],
                &d1_cross_proj,
            ),
            self.op1_c,
        );
        let s2_cross = normalize_penalty_cross_psi_derivative(
            &self.op2_s_raw,
            &self.op2_s_first_raw[axis_a],
            &self.op2_s_first_raw[axis_b],
            &gram_cross_psi_derivative_from_operator(
                &self.d2,
                &self.d2_eta_proj[axis_a],
                &self.d2_eta_proj[axis_b],
                &d2_cross_proj,
            ),
            self.op2_c,
        );

        active_operator_penalty_derivatives(
            &self.penaltyinfo,
            &[s0_cross, s1_cross, s2_cross],
            "Matérn-aniso-cross",
        )
    }
}

pub(crate) fn build_matern_operator_penalty_aniso_derivatives(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    z_opt: Option<&Array2<f64>>,
    eta: &[f64],
) -> Result<
    (
        Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)>,
        Vec<(usize, usize)>,
        AnisoPenaltyCrossProvider,
    ),
    BasisError,
> {
    let p = centers.nrows();
    let d = centers.ncols();
    let dim = eta.len();
    assert_eq!(dim, d);

    // Per-axis: build raw D0, D1, and full-Hessian D2 plus their eta_a
    // first/second derivatives.
    let mut d0_raw = Array2::<f64>::zeros((p, p));
    let mut d1_raw = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw = Array2::<f64>::zeros((p * d * d, p));
    let mut d0_raw_eta: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p, p))).collect();
    let mut d1_raw_eta: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p * d, p))).collect();
    let mut d2_raw_eta: Vec<Array2<f64>> =
        (0..dim).map(|_| Array2::zeros((p * d * d, p))).collect();
    let mut d0_raw_eta2: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p, p))).collect();
    let mut d1_raw_eta2: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((p * d, p))).collect();
    let mut d2_raw_eta2: Vec<Array2<f64>> =
        (0..dim).map(|_| Array2::zeros((p * d * d, p))).collect();
    let num_cross = dim * (dim - 1) / 2;
    let mut cross_pairs: Vec<(usize, usize)> = Vec::with_capacity(num_cross);
    for a in 0..dim {
        for b in (a + 1)..dim {
            cross_pairs.push((a, b));
        }
    }
    let metric_weights = centered_aniso_metric_weights(eta);

    struct CenterRowAccumulator {
        pub(crate) k: usize,
        pub(crate) d0: Array1<f64>,
        pub(crate) d1: Array2<f64>,
        pub(crate) d2: Array2<f64>,
        pub(crate) d0_eta: Vec<Array1<f64>>,
        pub(crate) d1_eta: Vec<Array2<f64>>,
        pub(crate) d2_eta: Vec<Array2<f64>>,
        pub(crate) d0_eta2: Vec<Array1<f64>>,
        pub(crate) d1_eta2: Vec<Array2<f64>>,
        pub(crate) d2_eta2: Vec<Array2<f64>>,
    }

    let row_accumulators: Vec<CenterRowAccumulator> = (0..p)
        .into_par_iter()
        .map(|k| -> Result<CenterRowAccumulator, BasisError> {
            let ci: Vec<f64> = (0..d).map(|a| centers[[k, a]]).collect();
            let mut d0 = Array1::<f64>::zeros(p);
            let mut d1 = Array2::<f64>::zeros((d, p));
            let mut d2 = Array2::<f64>::zeros((d * d, p));
            let mut d0_eta: Vec<Array1<f64>> = (0..dim).map(|_| Array1::zeros(p)).collect();
            let mut d1_eta: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((d, p))).collect();
            let mut d2_eta: Vec<Array2<f64>> =
                (0..dim).map(|_| Array2::zeros((d * d, p))).collect();
            let mut d0_eta2: Vec<Array1<f64>> = (0..dim).map(|_| Array1::zeros(p)).collect();
            let mut d1_eta2: Vec<Array2<f64>> = (0..dim).map(|_| Array2::zeros((d, p))).collect();
            let mut d2_eta2: Vec<Array2<f64>> =
                (0..dim).map(|_| Array2::zeros((d * d, p))).collect();

            for j in 0..p {
                let cj: Vec<f64> = (0..d).map(|a| centers[[j, a]]).collect();
                let (r, s_vec) = aniso_distance_and_components(&ci, &cj, eta);

                let (phi, q, t, dt_dr, d2t_dr2) =
                    matern_aniso_extended_radial_scalars(r, length_scale, nu)?;

                // --- D₀ ---
                d0[j] = phi;

                // --- D₁ (gradient) ---
                for axis in 0..d {
                    let h_b = ci[axis] - cj[axis];
                    let w_b = metric_weights[axis];
                    d1[[axis, j]] = q * w_b * h_b;
                }

                // --- D₂ (full Hessian, row layout point × axis × axis) ---
                for b in 0..d {
                    let h_b = ci[b] - cj[b];
                    let w_b = metric_weights[b];
                    for c in 0..d {
                        let h_c = ci[c] - cj[c];
                        let w_c = metric_weights[c];
                        let row = b * d + c;
                        d2[[row, j]] = hessian_operator_entry(q, t, h_b, h_c, w_b, w_c, b, c);
                    }
                }

                // --- Per-axis η_a derivatives ---
                for a in 0..dim {
                    let s_a = s_vec[a];

                    // ∂D₀/∂η_a = q · s_a
                    d0_eta[a][j] = q * s_a;
                    // ∂²D₀/∂η_a² = t · s_a² + 2q · s_a
                    d0_eta2[a][j] = t * s_a * s_a + 2.0 * q * s_a;

                    // ∂D₁/∂η_a: for each axis b, ∂(q · h_b)/∂η_a = (dq/dη_a) · h_b = t · s_a · h_b
                    for b in 0..d {
                        let h_b = ci[b] - cj[b];
                        let w_b = metric_weights[b];
                        d1_eta[a][[b, j]] = if a == b {
                            w_b * h_b * (t * s_a + 2.0 * q)
                        } else {
                            w_b * h_b * t * s_a
                        };
                        d1_eta2[a][[b, j]] = if a == b && r > 1e-14 {
                            w_b * h_b * (dt_dr * s_a * s_a / r + 6.0 * t * s_a + 4.0 * q)
                        } else if a == b {
                            0.0
                        } else if r > 1e-14 {
                            w_b * h_b * (dt_dr * s_a * s_a / r + 2.0 * t * s_a)
                        } else {
                            0.0
                        };
                    }

                    for b in 0..d {
                        let h_b = ci[b] - cj[b];
                        let w_b = metric_weights[b];
                        for c in 0..d {
                            let h_c = ci[c] - cj[c];
                            let w_c = metric_weights[c];
                            let row = b * d + c;
                            d2_eta[a][[row, j]] = hessian_operator_eta_entry(
                                q, t, dt_dr, r, s_a, h_b, h_c, w_b, w_c, a, b, c,
                            );
                            d2_eta2[a][[row, j]] = hessian_operator_eta2_entry(
                                q, t, dt_dr, d2t_dr2, r, s_a, h_b, h_c, w_b, w_c, a, b, c,
                            );
                        }
                    }
                }
            }

            Ok(CenterRowAccumulator {
                k,
                d0,
                d1,
                d2,
                d0_eta,
                d1_eta,
                d2_eta,
                d0_eta2,
                d1_eta2,
                d2_eta2,
            })
        })
        .collect::<Result<Vec<_>, BasisError>>()?;

    for row in row_accumulators {
        let k = row.k;
        d0_raw.row_mut(k).assign(&row.d0);
        d1_raw.slice_mut(s![k * d..(k + 1) * d, ..]).assign(&row.d1);
        d2_raw
            .slice_mut(s![k * d * d..(k + 1) * d * d, ..])
            .assign(&row.d2);

        for a in 0..dim {
            d0_raw_eta[a].row_mut(k).assign(&row.d0_eta[a]);
            d1_raw_eta[a]
                .slice_mut(s![k * d..(k + 1) * d, ..])
                .assign(&row.d1_eta[a]);
            d2_raw_eta[a]
                .slice_mut(s![k * d * d..(k + 1) * d * d, ..])
                .assign(&row.d2_eta[a]);
            d0_raw_eta2[a].row_mut(k).assign(&row.d0_eta2[a]);
            d1_raw_eta2[a]
                .slice_mut(s![k * d..(k + 1) * d, ..])
                .assign(&row.d1_eta2[a]);
            d2_raw_eta2[a]
                .slice_mut(s![k * d * d..(k + 1) * d * d, ..])
                .assign(&row.d2_eta2[a]);
        }
    }

    // Project through identifiability transform Z (ψ-independent).
    let project = |mat: Array2<f64>| -> Array2<f64> {
        if let Some(z) = z_opt {
            fast_ab(&mat, z)
        } else {
            mat
        }
    };

    let d0_kernel = project(d0_raw);
    let d1_kernel = project(d1_raw);
    let d2_kernel = project(d2_raw);

    let kernel_cols = d0_kernel.ncols();
    let total_cols = kernel_cols + usize::from(include_intercept);

    // Pad with intercept column.
    let pad = |kernel_mat: Array2<f64>, nrows: usize, add_intercept_ones: bool| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((nrows, total_cols));
        out.slice_mut(s![.., 0..kernel_cols]).assign(&kernel_mat);
        if add_intercept_ones && include_intercept {
            out.column_mut(kernel_cols).fill(1.0);
        }
        out
    };

    let d0 = pad(d0_kernel, p, true);
    let d1 = pad(d1_kernel, p * d, false);
    let d2 = pad(d2_kernel, p * d * d, false);

    // Project and pad all per-axis operator derivative matrices upfront,
    // so they remain available for cross-term computation.
    let d0_eta_all: Vec<Array2<f64>> = d0_raw_eta
        .into_iter()
        .map(|m| pad(project(m), p, false))
        .collect();
    let d1_eta_all: Vec<Array2<f64>> = d1_raw_eta
        .into_iter()
        .map(|m| pad(project(m), p * d, false))
        .collect();
    let d2_eta_all: Vec<Array2<f64>> = d2_raw_eta
        .into_iter()
        .map(|m| pad(project(m), p * d * d, false))
        .collect();
    let d0_eta2_all: Vec<Array2<f64>> = d0_raw_eta2
        .into_iter()
        .map(|m| pad(project(m), p, false))
        .collect();
    let d1_eta2_all: Vec<Array2<f64>> = d1_raw_eta2
        .into_iter()
        .map(|m| pad(project(m), p * d, false))
        .collect();
    let d2_eta2_all: Vec<Array2<f64>> = d2_raw_eta2
        .into_iter()
        .map(|m| pad(project(m), p * d * d, false))
        .collect();

    // Build raw Gram penalties (axis-independent) and their per-axis
    // first/second derivatives + Frobenius norms.
    // We compute these once for axis 0 (the raw Gram S and norm c are the same
    // for all axes) and store them, then reuse c for cross-term normalization.
    struct PerOperatorInfo {
        pub(crate) s_raw: Array2<f64>,
        pub(crate) c: f64,
        pub(crate) s_first: Vec<Array2<f64>>, // per-axis first derivatives (normalized)
        pub(crate) s_second: Vec<Array2<f64>>, // per-axis second derivatives (normalized)
        pub(crate) s_first_raw: Vec<Array2<f64>>, // per-axis first derivatives (raw, for cross normalization)
    }

    let compute_operator_info = |d_op: &Array2<f64>,
                                 d_eta_all: &[Array2<f64>],
                                 d_eta2_all: &[Array2<f64>]|
     -> PerOperatorInfo {
        // Compute the raw Gram and its norm (same for all axes).
        let s_raw = symmetrize(&fast_ata(d_op));
        let fro2: f64 = s_raw.iter().map(|v| v * v).sum();
        let c = fro2.sqrt();

        let mut s_first = Vec::with_capacity(dim);
        let mut s_second = Vec::with_capacity(dim);
        let mut s_first_raw = Vec::with_capacity(dim);
        for a in 0..dim {
            let (_, sa, sa2) =
                gram_and_psi_derivatives_from_operator(d_op, &d_eta_all[a], &d_eta2_all[a]);
            let (_, sa_norm, sa2_norm, _) =
                normalize_penaltywith_psi_derivatives(&s_raw, &sa, &sa2);
            s_first_raw.push(sa);
            s_first.push(sa_norm);
            s_second.push(sa2_norm);
        }

        PerOperatorInfo {
            s_raw,
            c,
            s_first,
            s_second,
            s_first_raw,
        }
    };

    let op0_info = compute_operator_info(&d0, &d0_eta_all, &d0_eta2_all);
    let op1_info = compute_operator_info(&d1, &d1_eta_all, &d1_eta2_all);
    let op2_info = compute_operator_info(&d2, &d2_eta_all, &d2_eta2_all);

    // Build penalty candidates and determine which are active (using axis-0
    // normalized Gram, which is axis-independent).
    let (s0_norm, c0) = if op0_info.c > 1e-12 {
        (op0_info.s_raw.mapv(|v| v / op0_info.c), op0_info.c)
    } else {
        (op0_info.s_raw.clone(), 1.0)
    };
    let (s1_norm, c1) = if op1_info.c > 1e-12 {
        (op1_info.s_raw.mapv(|v| v / op1_info.c), op1_info.c)
    } else {
        (op1_info.s_raw.clone(), 1.0)
    };
    let (s2_norm, c2) = if op2_info.c > 1e-12 {
        (op2_info.s_raw.mapv(|v| v / op2_info.c), op2_info.c)
    } else {
        (op2_info.s_raw.clone(), 1.0)
    };

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
    let (_, _, penaltyinfo) = filter_active_penalty_candidates(candidates)?;

    // Build per-axis results.
    let mut per_axis_results = Vec::with_capacity(dim);
    for a in 0..dim {
        let pen_first = active_operator_penalty_derivatives(
            &penaltyinfo,
            &[
                op0_info.s_first[a].clone(),
                op1_info.s_first[a].clone(),
                op2_info.s_first[a].clone(),
            ],
            "Matérn-aniso",
        )?;
        let pen_second = active_operator_penalty_derivatives(
            &penaltyinfo,
            &[
                op0_info.s_second[a].clone(),
                op1_info.s_second[a].clone(),
                op2_info.s_second[a].clone(),
            ],
            "Matérn-aniso",
        )?;
        per_axis_results.push((pen_first, pen_second));
    }

    let cross_ctx = std::sync::Arc::new(MaternCrossPenaltyContext {
        centers: centers.to_owned(),
        aniso_log_scales: eta.to_vec(),
        length_scale,
        nu,
        z_transform: z_opt.cloned(),
        penaltyinfo,
        d0,
        d1,
        d2,
        d0_eta_proj: d0_eta_all,
        d1_eta_proj: d1_eta_all,
        d2_eta_proj: d2_eta_all,
        op0_s_raw: op0_info.s_raw,
        op1_s_raw: op1_info.s_raw,
        op2_s_raw: op2_info.s_raw,
        op0_c: op0_info.c,
        op1_c: op1_info.c,
        op2_c: op2_info.c,
        op0_s_first_raw: op0_info.s_first_raw,
        op1_s_first_raw: op1_info.s_first_raw,
        op2_s_first_raw: op2_info.s_first_raw,
    });
    let cross_provider = AnisoPenaltyCrossProvider::new(move |a: usize, b: usize| {
        let (axis_a, axis_b) = if a < b { (a, b) } else { (b, a) };
        if axis_a == axis_b || axis_b >= cross_ctx.d0_eta_proj.len() {
            return Ok(Vec::new());
        }
        cross_ctx.compute_pair(axis_a, axis_b)
    });

    Ok((per_axis_results, cross_pairs, cross_provider))
}

/// Build exact per-axis η_a derivatives of operator penalty matrices for
/// anisotropic hybrid Duchon terms.
///
/// Analogous to [`build_matern_operator_penalty_aniso_derivatives`] but for
/// the Duchon kernel. Uses `duchon_radial_jets` for the full radial jet
/// `(φ, q, t, t_r, t_rr)`.
///
/// The local y-space operator shape derivatives start from the same formulas as
/// Matérn, but the raw per-axis `psi_a` coordinates also inherit the Duchon
/// isotropic scaling law. After assembling the shape-only pieces, this routine
/// adds the exact raw-`psi` isotropic-share correction implied by
/// `phi(r; kappa) = kappa^delta H(kappa r)`.
pub(crate) fn duchon_kernel_radial_triplet(
    r: f64,
    length_scale: Option<f64>,
    p_order: usize,
    s_order: f64,
    k_dim: usize,
    coeffs: Option<&DuchonPartialFractionCoeffs>,
) -> Result<(f64, f64, f64), BasisError> {
    // Public Duchon (φ, φ_r, φ_rr) triplet.
    //
    // The Duchon spectral kernel is F(ρ) = 1 / [ρ^(2p)·(κ²+ρ²)^s]. The pure
    // case (κ=0) is the κ→0 limit and collapses to a single polyharmonic of
    // order m = p+s — value and both radial derivatives all come from one
    // normalization in `polyharmonic_block_jet4`. The hybrid case (κ>0) is the
    // partial-fraction sum; we route it through `duchon_radial_jets` so the
    // public triplet shares the same Taylor / collision tiering used by the
    // operator scalars (q, lap, t) in the penalty code.
    let triplet = match length_scale {
        None => {
            // Keep the block order in `f64`: fractional `s_order` rides
            // through `pure_duchon_block_order` → `polyharmonic_kernel_triplet`
            // → `polyharmonic_block_jet4` end-to-end. Truncating to
            // `usize` here was the bug — for `s=1.5, p=2, d=4` it
            // collapsed `m=3.5` to `m=3` and tripped the integer-only
            // log-case branch at `m=d/2`, producing NaN at `r=0`.
            let m = pure_duchon_block_order(p_order, s_order);
            polyharmonic_kernel_triplet(r, m, k_dim)?
        }
        Some(length_scale) => {
            if !length_scale.is_finite() || length_scale <= 0.0 {
                crate::bail_invalid_basis!(
                    "Duchon hybrid length_scale must be finite and positive"
                );
            }
            let kappa = 1.0 / length_scale.max(1e-300);
            let coeffs_local;
            let coeffs_ref = match coeffs {
                Some(c) => c,
                None => {
                    coeffs_local = duchon_partial_fraction_coeffs(p_order, s_order as usize, kappa);
                    &coeffs_local
                }
            };
            let jets = duchon_radial_jets(
                r,
                length_scale,
                p_order,
                s_order as usize,
                k_dim,
                coeffs_ref,
            )?;
            (jets.phi, jets.phi_r, jets.phi_rr)
        }
    };

    if !triplet.0.is_finite() || !triplet.1.is_finite() || !triplet.2.is_finite() {
        crate::bail_invalid_basis!(
            "non-finite Duchon radial triplet at r={r}, length_scale={length_scale:?}, p={p_order}, s={s_order}, dim={k_dim}"
        );
    }
    Ok(triplet)
}

#[inline(always)]
pub(crate) fn lower_triangular_offset(row: usize) -> usize {
    if row & 1 == 0 {
        (row / 2)
            .checked_mul(row + 1)
            .expect("lower-triangular row offset overflow")
    } else {
        row.checked_mul(row / 2 + 1)
            .expect("lower-triangular row offset overflow")
    }
}

pub(crate) fn lower_triangular_len(k: usize) -> usize {
    if k & 1 == 0 {
        (k / 2)
            .checked_mul(k.checked_add(1).expect("lower-triangular length overflow"))
            .expect("lower-triangular length overflow")
    } else {
        k.checked_mul(k / 2 + 1)
            .expect("lower-triangular length overflow")
    }
}

pub(crate) fn symmetric_matrix_from_lower_values(k: usize, values: &[f64]) -> Array2<f64> {
    assert_eq!(values.len(), lower_triangular_len(k));
    let mut g = Array2::<f64>::zeros((k, k));
    let mut idx = 0usize;
    for i in 0..k {
        for j in 0..=i {
            let v = values[idx];
            g[[i, j]] = v;
            if i != j {
                g[[j, i]] = v;
            }
            idx += 1;
        }
    }
    g
}

pub(crate) fn transform_closed_form_raw_block(
    raw: &Array2<f64>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> Array2<f64> {
    let kernel_block = if let Some(z) = kernel_nullspace {
        let zt = fast_atb(z, raw);
        fast_ab(&zt, z)
    } else {
        raw.clone()
    };
    let kernel_cols = kernel_block.nrows();
    let total_pre = kernel_cols + polynomial_block_cols;
    let padded = if polynomial_block_cols == 0 {
        kernel_block
    } else {
        let mut padded = Array2::<f64>::zeros((total_pre, total_pre));
        padded
            .slice_mut(s![0..kernel_cols, 0..kernel_cols])
            .assign(&kernel_block);
        padded
    };
    let total = if let Some(t) = outer_identifiability {
        let tt = fast_atb(t, &padded);
        fast_ab(&tt, t)
    } else {
        padded
    };
    symmetrize(&total)
}

pub(crate) fn symmetrize(matrix: &Array2<f64>) -> Array2<f64> {
    (matrix + &matrix.t().to_owned()) * 0.5
}

/// Centered design Gram: `(D − 1 μ')^T (D − 1 μ')` where `μ_j` is the
/// column mean of `D` across rows. The constant direction (intercept basis
/// = column of ones) sits in the exact null space because its centered
/// column is identically zero. Used as the spring-measure mass penalty on
/// the scale-free Duchon path so the magnitude term penalizes deviations
/// from the function's row-mean rather than absolute level — the intercept
/// is genuinely unpenalized regardless of what the row-mean is.
pub(crate) fn centered_design_gram(d: &Array2<f64>) -> Array2<f64> {
    let n_rows = d.nrows();
    let n_cols = d.ncols();
    if n_rows == 0 || n_cols == 0 {
        return Array2::<f64>::zeros((n_cols, n_cols));
    }
    let inv_n = 1.0 / n_rows as f64;
    let col_sum = d.sum_axis(Axis(0));
    let g_raw = fast_ata(d);
    // (D − 1μ')'(D − 1μ') = D'D − N μ μ'   where Σ = D'1 = col_sum,
    // so the rank-1 correction is `col_sum col_sum' / N`.
    let mut out = g_raw;
    for i in 0..n_cols {
        let ci = col_sum[i];
        let row = out.row_mut(i);
        // Subtract column i's contribution: out[i, j] -= (ci * col_sum[j]) / N.
        let mut row = row;
        for j in 0..n_cols {
            row[j] -= ci * col_sum[j] * inv_n;
        }
    }
    out
}

pub(crate) fn centered_operator_gram_and_psi_derivatives(
    d: &Array2<f64>,
    d_psi: &Array2<f64>,
    d_psi_psi: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let center_columns = |mat: &Array2<f64>| {
        let n_rows = mat.nrows();
        if n_rows == 0 || mat.ncols() == 0 {
            return mat.clone();
        }
        let means = mat.sum_axis(Axis(0)).mapv(|v| v / n_rows as f64);
        let mut centered = mat.clone();
        for mut row in centered.rows_mut() {
            row -= &means;
        }
        centered
    };
    let d_centered = center_columns(d);
    let d_psi_centered = center_columns(d_psi);
    let d_psi_psi_centered = center_columns(d_psi_psi);
    gram_and_psi_derivatives_from_operator(&d_centered, &d_psi_centered, &d_psi_psi_centered)
}

pub(crate) fn normalize_penalty(matrix: &Array2<f64>) -> (Array2<f64>, f64) {
    let norm = matrix.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    (matrix.mapv(|v| v / norm), norm)
}

pub(crate) fn closed_form_anisotropic_pair_value_with_powers(
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    eta_raw: &[f64],
    powers: &closed_form_penalty::AnisoMetricPowers,
    r: &[f64],
    diagonal_epsilon: f64,
) -> f64 {
    assert_eq!(
        eta_raw.len(),
        r.len(),
        "closed_form_anisotropic_pair_value_with_powers: eta/r dimension mismatch"
    );
    let j_prefactor = eta_raw.iter().sum::<f64>().exp();
    if r.iter().all(|&value| value == 0.0) {
        // Exact distributional diagonal first. In the convergent spectral
        // strip the radial pointwise chain is singular at R=0, but the
        // self-pair integral is finite and has a Gamma/Beta closed form.
        // Outside that strip, odd-d hybrid Taylor covers the smooth pointwise
        // diagonal; epsilon regularization is only the final non-convergent
        // diagonal convention.
        if let Some(bundle) =
            closed_form_penalty::analytic_self_pair_bundle(q, m, s, kappa, eta_raw)
        {
            return bundle.value;
        }
        let mut r_eps_buf = vec![0.0_f64; r.len()];
        if !r_eps_buf.is_empty() {
            r_eps_buf[0] = diagonal_epsilon * eta_raw[0].exp();
        }
        return j_prefactor
            * closed_form_penalty::anisotropic_duchon_penalty_radial_with_powers(
                q, m, s as f64, kappa, eta_raw, powers, &r_eps_buf,
            );
    }

    j_prefactor
        * closed_form_penalty::anisotropic_duchon_penalty_radial_with_powers(
            q, m, s as f64, kappa, eta_raw, powers, r,
        )
}

pub fn closed_form_anisotropic_pair_block(
    centers: ArrayView2<'_, f64>,
    q: usize,
    m: usize,
    s: usize,
    kappa: f64,
    aniso_log_scales: Option<&[f64]>,
) -> Array2<f64> {
    // Math team Letter A §9: G_q(η_raw, κ) ≠ G_q(η_centered, κ) in general;
    // the relation involves an exp((2d-4m-4s)μ) prefactor and a κ rescaling
    // that don't reduce to a uniform Jacobian. Use raw η directly so the
    // pair-block matches the closed-form Lebesgue penalty's natural raw-η
    // parameterization, and so its η-derivatives (computed elsewhere via
    // `pair_block_radial_with_j_second_derivatives` with raw η) are already
    // ∂G_q/∂η_raw without any chain-rule conversion.
    let k = centers.nrows();
    let d = centers.ncols();
    let zeros: Vec<f64>;
    let eta_raw: &[f64] = match aniso_log_scales {
        Some(eta) => eta,
        None => {
            zeros = vec![0.0_f64; d];
            &zeros
        }
    };
    let r_eps = if closed_form_penalty::analytic_self_pair_bundle(q, m, s, kappa, eta_raw).is_some()
    {
        0.0
    } else {
        pure_duchon_diagonal_epsilon(centers, eta_raw)
    };
    let powers = closed_form_penalty::AnisoMetricPowers::new(eta_raw);

    // Parallelize by independent lower-triangular rows. This keeps one lag
    // scratch buffer per worker row, avoids the sqrt-based flat-index decode
    // in the hot loop, and still evaluates each symmetric pair only once.
    let n_pairs = lower_triangular_len(k);
    let mut values = vec![0.0_f64; n_pairs];
    let values_ptr = SendPtr(values.as_mut_ptr());
    (0..k).into_par_iter().for_each(|i| {
        let row_ptr = values_ptr.add(lower_triangular_offset(i));
        let mut r_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
        r_buf.resize(d, 0.0);
        for j in 0..=i {
            for axis in 0..d {
                r_buf[axis] = centers[[i, axis]] - centers[[j, axis]];
            }
            let value = closed_form_anisotropic_pair_value_with_powers(
                q, m, s, kappa, eta_raw, &powers, &r_buf, r_eps,
            );
            // SAFETY: values has k(k+1)/2 slots; for i in 0..k and j ∈ 0..=i,
            // lower_triangular_offset(i)+j is in bounds. Each rayon iteration
            // owns a distinct lower-triangular row, so writes are disjoint.
            unsafe {
                *row_ptr.add(j) = value;
            }
        }
    });

    symmetric_matrix_from_lower_values(k, &values)
}

/// Pure-Duchon (κ=0) variant of [`closed_form_anisotropic_pair_block`].
///
/// Uses the analytic radial-derivative path
/// [`closed_form_penalty::anisotropic_duchon_penalty_radial`] which handles
/// κ=0 cleanly by delegating to pure-Riesz radial derivatives. The
/// Schoenberg path is undefined at κ=0 in low dimensions, so this variant
/// must be used in place of [`closed_form_anisotropic_pair_block`] for
/// `length_scale = None` (pure-Duchon) penalty assembly.
///
/// Self-pairs (R=0) are ε-regularized using a small fraction of the median
/// off-diagonal lag, since the radial form is singular at R=0 and the pure
/// Riesz Schoenberg fallback also doesn't converge for κ=0.
pub fn closed_form_anisotropic_pair_block_pure(
    centers: ArrayView2<'_, f64>,
    q: usize,
    m: usize,
    s: f64,
    aniso_log_scales: Option<&[f64]>,
) -> Array2<f64> {
    let k = centers.nrows();
    let d = centers.ncols();
    let eta_centered: Vec<f64> = if let Some(eta) = aniso_log_scales {
        let mean = centered_aniso_log_scale_mean(eta);
        eta.iter()
            .map(|&v| centered_aniso_log_scale(v, mean))
            .collect()
    } else {
        vec![0.0_f64; d]
    };
    let j_prefactor = eta_centered.iter().sum::<f64>().exp();

    // Median off-diagonal anisotropic distance is needed only when the
    // exact pure-Duchon finite self-pair is unavailable. The integer-only
    // self-pair helper is consulted only when `s` is whole-valued;
    // fractional `s` always falls through to the analytic radial chain
    // below, which now accepts `f64` via the threaded
    // `radial_derivatives_of_isotropic_duchon` cascade.
    let s_int = if s.fract() == 0.0 && s >= 0.0 {
        Some(s as usize)
    } else {
        None
    };
    let pure_diag_exact = s_int
        .and_then(|si| closed_form_penalty::pure_duchon_self_pair_value(q, d, m, si, &eta_centered))
        .is_some();
    let r_eps = if pure_diag_exact {
        0.0
    } else {
        pure_duchon_diagonal_epsilon(centers, &eta_centered)
    };
    let powers = closed_form_penalty::AnisoMetricPowers::new(&eta_centered);

    // Parallelize by independent lower-triangular rows and evaluate each
    // symmetric pair once while reusing a single lag scratch buffer per row.
    let n_pairs = lower_triangular_len(k);
    let eta_slice: &[f64] = eta_centered.as_slice();
    let mut values = vec![0.0_f64; n_pairs];
    let values_ptr = SendPtr(values.as_mut_ptr());
    (0..k).into_par_iter().for_each(|i| {
        let row_ptr = values_ptr.add(lower_triangular_offset(i));
        let mut r_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
        r_buf.resize(d, 0.0);
        for j in 0..=i {
            for axis in 0..d {
                r_buf[axis] = centers[[i, axis]] - centers[[j, axis]];
            }
            let value = if i == j {
                // Self-pair (R = 0). Prefer the exact finite-part limit
                // when available (integer s only); otherwise use the same
                // ε-regularized convention via the analytic radial chain
                // (which now accepts fractional s end-to-end).
                let closed_self = s_int.and_then(|si| {
                    closed_form_penalty::pure_duchon_self_pair_value(q, d, m, si, eta_slice)
                });
                if let Some(closed) = closed_self {
                    j_prefactor * closed
                } else {
                    let mut r_eps_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
                    r_eps_buf.resize(d, 0.0);
                    if d > 0 {
                        r_eps_buf[0] = r_eps * eta_slice[0].exp();
                    }
                    j_prefactor
                        * closed_form_penalty::anisotropic_duchon_penalty_radial_with_powers(
                            q, m, s, 0.0, eta_slice, &powers, &r_eps_buf,
                        )
                }
            } else {
                j_prefactor
                    * closed_form_penalty::anisotropic_duchon_penalty_radial_with_powers(
                        q, m, s, 0.0, eta_slice, &powers, &r_buf,
                    )
            };
            // SAFETY: values has k(k+1)/2 slots; for i in 0..k and j ∈ 0..=i,
            // lower_triangular_offset(i)+j is in bounds. Each rayon iteration
            // owns a distinct lower-triangular row, so writes are disjoint.
            unsafe {
                *row_ptr.add(j) = value;
            }
        }
    });

    symmetric_matrix_from_lower_values(k, &values)
}

/// Closed-form pair-block penalty for the pure Matérn basis at operator order
/// `q ∈ {0, 1, 2}`.
///
/// Spectral form: the Matérn kernel has Fourier symbol `K̂(ρ) = c · (κ² + ρ²)^{-ℓ}`
/// with `ℓ = ν + d/2`. For half-integer `MaternNu`, `2ℓ = 2ν + d` is always
/// a positive integer (even when ℓ itself is half-integer in even d). The
/// pair-block penalty is
///
///   `g_q(R; κ) = F^{-1}{|ρ|^{2q} / (κ² + ρ²)^{2ℓ}}(R)`.
///
/// Using the binomial expansion `|ρ|^{2q} = ((κ² + ρ²) − κ²)^q`:
///
///   `g_q(R; κ) = Σ_{j=0}^{q} C(q,j) (−κ²)^{q−j} · M_{2ℓ−j}^d(R; κ)`,
///
/// where each `M_ℓ'^d(R; κ)` is supplied by
/// [`closed_form_penalty::matern_kernel_value`]. Convergence (finite block at
/// R = 0) requires `4ℓ − 2q > d`, i.e. the Sobolev order strictly exceeds
/// the operator order plus dimension/2; otherwise the spectral integrand is
/// not integrable and the resulting matrix is not PSD-by-construction.
///
/// Length scale enters as `κ = √(2ν) / length_scale` (the standard Matérn
/// parameterization that makes `length_scale` the practical correlation
/// scale). Anisotropy is handled via `aniso_log_scales` by rescaling lags:
/// `r_axis ← r_axis · exp(η_axis)`. The spectral form is invariant under
/// this rescaling (Schoenberg) so the penalty matrix remains PSD when the
/// gate accepts.
///
/// Returns `None` when `q > 2` or when the spectral integral diverges
/// (`4ℓ ≤ 2q + d`); the caller should fall back to the collocation
/// `D_q^T D_q` path in those regimes.
pub fn closed_form_matern_pair_block(
    centers: ArrayView2<'_, f64>,
    q: usize,
    length_scale: f64,
    nu: MaternNu,
    aniso_log_scales: Option<&[f64]>,
) -> Option<Array2<f64>> {
    assert!(
        length_scale.is_finite() && length_scale > 0.0,
        "closed_form_matern_pair_block: length_scale must be finite and positive"
    );
    if q > 2 {
        return None;
    }
    let k = centers.nrows();
    let d = centers.ncols();
    if d == 0 || k == 0 {
        return Some(Array2::<f64>::zeros((k, k)));
    }

    // Convert MaternNu (half-integer ν) to integer 2ν, then 2ℓ = 2ν + d.
    let two_nu: usize = match nu {
        MaternNu::Half => 1,
        MaternNu::ThreeHalves => 3,
        MaternNu::FiveHalves => 5,
        MaternNu::SevenHalves => 7,
        MaternNu::NineHalves => 9,
    };
    let two_ell = two_nu + d;

    // IR convergence: 2·(2ℓ) > 2q + d.
    if 2 * two_ell <= 2 * q + d {
        return None;
    }
    // Each building block requires `2ℓ - j ≥ 1` for j ∈ [0, q].
    if two_ell < q + 1 {
        return None;
    }

    // Standard Matérn parameterization: κ = √(2ν) / length_scale.
    let kappa = (two_nu as f64).sqrt() / length_scale;
    let kappa_sq = kappa * kappa;

    // Per-axis multiplicative scale for anisotropic lags.
    let scale_per_axis: Option<Vec<f64>> = aniso_log_scales.map(|eta| {
        assert_eq!(
            eta.len(),
            d,
            "closed_form_matern_pair_block: aniso_log_scales length must match d"
        );
        eta.iter().map(|v| v.exp()).collect()
    });

    // Coefficients C(q, j) · (−κ²)^{q−j} for j = 0..q.
    let mut binom_coeffs: Vec<f64> = Vec::with_capacity(q + 1);
    for j in 0..=q {
        let cqj = crate::probability::binomial_coefficient_f64(q, j);
        let sign_pow = if (q - j).is_multiple_of(2) { 1.0 } else { -1.0 };
        let coeff = cqj * sign_pow * kappa_sq.powi((q - j) as i32);
        binom_coeffs.push(coeff);
    }

    let n_pairs = lower_triangular_len(k);
    let mut values = vec![0.0_f64; n_pairs];
    let values_ptr = SendPtr(values.as_mut_ptr());
    (0..k).into_par_iter().for_each(|i| {
        let row_ptr = values_ptr.add(lower_triangular_offset(i));
        for j in 0..=i {
            // Anisotropic distance: r_eff² = Σ_a (Δ_a · exp(η_a))².
            let mut r2 = 0.0_f64;
            for axis in 0..d {
                let delta = centers[[i, axis]] - centers[[j, axis]];
                let scaled = if let Some(sc) = &scale_per_axis {
                    delta * sc[axis]
                } else {
                    delta
                };
                r2 += scaled * scaled;
            }
            let r = r2.sqrt();

            let mut acc = 0.0_f64;
            for jj in 0..=q {
                let order = two_ell - jj; // ≥ 1 by the gate
                acc +=
                    binom_coeffs[jj] * closed_form_penalty::matern_kernel_value(d, order, kappa, r);
            }
            // SAFETY: values has k(k+1)/2 slots; for i in 0..k and j ∈ 0..=i,
            // lower_triangular_offset(i)+j is in bounds. Each rayon iteration
            // owns a distinct lower-triangular row, so writes are disjoint.
            unsafe {
                *row_ptr.add(j) = acc;
            }
        }
    });
    Some(symmetric_matrix_from_lower_values(k, &values))
}

/// Median off-diagonal anisotropic lag scaled by 1e-6, used for
/// regularizing self-pair R=0 evaluations in pure-Duchon (κ=0) closed-form
/// penalties. Matches the magnitude used by hybrid κ>0 collocation builders
/// where the ε-regularization is implicit in the Matérn kernel finiteness.
pub(crate) fn pure_duchon_diagonal_epsilon(
    centers: ArrayView2<'_, f64>,
    eta_log_scales: &[f64],
) -> f64 {
    let k = centers.nrows();
    let d = centers.ncols();
    if k <= 1 || d == 0 {
        return 1e-12;
    }
    let mut lags = Vec::with_capacity(k * (k - 1) / 2);
    for i in 0..k {
        for j in 0..i {
            let mut acc = 0.0_f64;
            for axis in 0..d {
                let delta = centers[[i, axis]] - centers[[j, axis]];
                let b = (-2.0 * eta_log_scales[axis]).exp();
                acc += b * delta * delta;
            }
            let r = acc.sqrt();
            if r > 0.0 {
                lags.push(r);
            }
        }
    }
    if lags.is_empty() {
        return 1e-12;
    }
    lags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = lags[lags.len() / 2];
    (median * 1e-6).max(1e-12)
}

pub fn closed_form_operator_penalty_in_total_basis(
    centers: ArrayView2<'_, f64>,
    q: usize,
    p_order: usize,
    s_order: usize,
    kappa: f64,
    aniso_log_scales: Option<&[f64]>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> Array2<f64> {
    // 1. Closed-form penalty in raw kernel basis (K×K).
    let g_raw =
        closed_form_anisotropic_pair_block(centers, q, p_order, s_order, kappa, aniso_log_scales);
    // 2. Apply kernel-constraint nullspace transform Z (K×kernel_cols).
    let g_kernel = if let Some(z) = kernel_nullspace {
        let zt_g = fast_atb(z, &g_raw);
        fast_ab(&zt_g, z)
    } else {
        g_raw
    };
    // 3. Block-diag pad polynomial nullspace (zero penalty in continuous form).
    let kernel_cols = g_kernel.nrows();
    let total_pre_cols = kernel_cols + polynomial_block_cols;
    let g_padded = if polynomial_block_cols == 0 {
        g_kernel
    } else {
        let mut padded = Array2::<f64>::zeros((total_pre_cols, total_pre_cols));
        padded
            .slice_mut(s![0..kernel_cols, 0..kernel_cols])
            .assign(&g_kernel);
        padded
    };
    // 4. Apply outer spatial identifiability transform if any.
    let g_total = if let Some(t) = outer_identifiability {
        let tt_g = fast_atb(t, &g_padded);
        fast_ab(&tt_g, t)
    } else {
        g_padded
    };
    symmetrize(&g_total)
}

/// Closed-form penalty value `S_q` and its log-κ derivatives `S_q_psi`,
/// `S_q_psi_psi` in the final (post-transform) basis space, computed via
/// `pair_block_radial_with_j_second_derivatives` from the closed_form_penalty
/// module. Bundle's `d_kappa` and `d2_kappa` are κ-derivatives; chain rule
/// converts to log-κ: `∂/∂ψ = κ·∂/∂κ`,
/// `∂²/∂ψ² = κ²·∂²/∂κ² + κ·∂/∂κ`.
///
/// All three matrices share the same Z + poly-pad + outer-T transform pipeline
/// as `closed_form_operator_penalty_in_total_basis` (the transforms are
/// linear and commute with parameter differentiation).
pub fn closed_form_psi_derivatives_in_total_basis(
    centers: ArrayView2<'_, f64>,
    q: usize,
    p_order: usize,
    s_order: usize,
    kappa: f64,
    aniso_log_scales: Option<&[f64]>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let k = centers.nrows();
    let d = centers.ncols();
    let zeros: Vec<f64>;
    let eta_raw: &[f64] = match aniso_log_scales {
        Some(eta) => eta,
        None => {
            zeros = vec![0.0_f64; d];
            &zeros
        }
    };
    let r_eps =
        if closed_form_penalty::analytic_self_pair_bundle(q, p_order, s_order, kappa, eta_raw)
            .is_some()
        {
            0.0
        } else {
            pure_duchon_diagonal_epsilon(centers, eta_raw)
        };
    let powers = closed_form_penalty::AnisoMetricPowers::new(eta_raw);

    let n_pairs = lower_triangular_len(k);
    let mut g_values = vec![0.0_f64; n_pairs];
    let mut g_psi_values = vec![0.0_f64; n_pairs];
    let mut g_psi_psi_values = vec![0.0_f64; n_pairs];
    let g_ptr = SendPtr(g_values.as_mut_ptr());
    let g_psi_ptr = SendPtr(g_psi_values.as_mut_ptr());
    let g_psi_psi_ptr = SendPtr(g_psi_psi_values.as_mut_ptr());
    (0..k).into_par_iter().for_each(|i| {
        let row_offset = lower_triangular_offset(i);
        let g_row = g_ptr.add(row_offset);
        let g_psi_row = g_psi_ptr.add(row_offset);
        let g_psi_psi_row = g_psi_psi_ptr.add(row_offset);
        let mut r_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
        r_buf.resize(d, 0.0);
        let mut r_eps_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
        r_eps_buf.resize(d, 0.0);
        if d > 0 {
            r_eps_buf[0] = r_eps * eta_raw[0].exp();
        }
        for j in 0..=i {
            for axis in 0..d {
                r_buf[axis] = centers[[i, axis]] - centers[[j, axis]];
            }
            let bundle = if i == j {
                closed_form_penalty::analytic_self_pair_bundle(q, p_order, s_order, kappa, eta_raw)
                    .unwrap_or_else(|| {
                        closed_form_penalty::pair_block_radial_with_j_second_derivatives_with_powers(
                            q, p_order, s_order, kappa, eta_raw, &powers, &r_eps_buf,
                        )
                    })
            } else {
                closed_form_penalty::pair_block_radial_with_j_second_derivatives_with_powers(
                    q, p_order, s_order, kappa, eta_raw, &powers, &r_buf,
                )
            };
            // SAFETY: each output has k(k+1)/2 slots; for i in 0..k and j ∈ 0..=i,
            // lower_triangular_offset(i)+j is in bounds. Each rayon iteration
            // owns that lower-triangular row in every output, so writes are disjoint.
            unsafe {
                *g_row.add(j) = bundle.value;
                *g_psi_row.add(j) = kappa * bundle.d_kappa;
                *g_psi_psi_row.add(j) = kappa * kappa * bundle.d2_kappa + kappa * bundle.d_kappa;
            }
        }
    });
    let g = symmetric_matrix_from_lower_values(k, &g_values);
    let g_psi = symmetric_matrix_from_lower_values(k, &g_psi_values);
    let g_psi_psi = symmetric_matrix_from_lower_values(k, &g_psi_psi_values);

    // Apply Z + poly-pad + T to each of g, g_psi, g_psi_psi identically.
    (
        transform_closed_form_raw_block(
            &g,
            kernel_nullspace,
            polynomial_block_cols,
            outer_identifiability,
        ),
        transform_closed_form_raw_block(
            &g_psi,
            kernel_nullspace,
            polynomial_block_cols,
            outer_identifiability,
        ),
        transform_closed_form_raw_block(
            &g_psi_psi,
            kernel_nullspace,
            polynomial_block_cols,
            outer_identifiability,
        ),
    )
}

/// Closed-form anisotropic penalty `S_q` and its raw-η derivatives — full
/// d×d Hessian materialized — in the final (post-transform) basis space.
/// Returns `(S_q, S_q_eta_a per axis, S_q_eta_a_a per axis, S_q_eta_a_b for
/// (a, b) with a < b)`. All derivatives are with respect to raw η components
/// directly per math team Letter A §9 — no centering, no apply_raw_psi_scaling.
///
/// Bundle is computed via `pair_block_radial_with_j_second_derivatives`,
/// which uses the radial analytic derivative chain in regular regimes and
/// the Schoenberg derivative bundle in convergent singular/log-Riesz regimes.
pub fn closed_form_aniso_psi_derivatives_in_total_basis(
    centers: ArrayView2<'_, f64>,
    q: usize,
    p_order: usize,
    s_order: usize,
    kappa: f64,
    aniso_log_scales: Option<&[f64]>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> (
    Array2<f64>,
    Vec<Array2<f64>>,
    Vec<Array2<f64>>,
    Vec<Vec<Array2<f64>>>,
) {
    let k = centers.nrows();
    let d = centers.ncols();
    let zeros: Vec<f64>;
    let eta_raw: &[f64] = match aniso_log_scales {
        Some(eta) => eta,
        None => {
            zeros = vec![0.0_f64; d];
            &zeros
        }
    };
    let r_eps =
        if closed_form_penalty::analytic_self_pair_bundle(q, p_order, s_order, kappa, eta_raw)
            .is_some()
        {
            0.0
        } else {
            pure_duchon_diagonal_epsilon(centers, eta_raw)
        };
    let powers = closed_form_penalty::AnisoMetricPowers::new(eta_raw);

    let cross_pairs: Vec<(usize, usize)> =
        (0..d).flat_map(|a| (a..d).map(move |b| (a, b))).collect();
    let n_pairs = lower_triangular_len(k);
    let mut g_values = vec![0.0_f64; n_pairs];
    let mut g_eta_values: Vec<Vec<f64>> = (0..d).map(|_| vec![0.0_f64; n_pairs]).collect();
    let mut g_eta2_diag_values: Vec<Vec<f64>> = (0..d).map(|_| vec![0.0_f64; n_pairs]).collect();
    let mut g_eta2_cross_values: Vec<Vec<f64>> =
        cross_pairs.iter().map(|_| vec![0.0_f64; n_pairs]).collect();

    let g_ptr = SendPtr(g_values.as_mut_ptr());
    let g_eta_ptrs: Vec<SendPtr> = g_eta_values
        .iter_mut()
        .map(|values| SendPtr(values.as_mut_ptr()))
        .collect();
    let g_eta2_diag_ptrs: Vec<SendPtr> = g_eta2_diag_values
        .iter_mut()
        .map(|values| SendPtr(values.as_mut_ptr()))
        .collect();
    let g_eta2_cross_ptrs: Vec<SendPtr> = g_eta2_cross_values
        .iter_mut()
        .map(|values| SendPtr(values.as_mut_ptr()))
        .collect();

    (0..k).into_par_iter().for_each(|i| {
        let row_offset = lower_triangular_offset(i);
        let g_row = g_ptr.add(row_offset);
        let g_eta_rows: Vec<*mut f64> = g_eta_ptrs
            .iter()
            .map(|ptr| ptr.add(row_offset))
            .collect();
        let g_eta2_diag_rows: Vec<*mut f64> = g_eta2_diag_ptrs
            .iter()
            .map(|ptr| ptr.add(row_offset))
            .collect();
        let g_eta2_cross_rows: Vec<*mut f64> = g_eta2_cross_ptrs
            .iter()
            .map(|ptr| ptr.add(row_offset))
            .collect();
        let mut r_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
        r_buf.resize(d, 0.0);
        let mut r_eps_buf: SmallVec<[f64; 16]> = SmallVec::with_capacity(d);
        r_eps_buf.resize(d, 0.0);
        if d > 0 {
            r_eps_buf[0] = r_eps * eta_raw[0].exp();
        }
        for j in 0..=i {
            for axis in 0..d {
                r_buf[axis] = centers[[i, axis]] - centers[[j, axis]];
            }
            let bundle = if i == j {
                closed_form_penalty::analytic_self_pair_bundle(q, p_order, s_order, kappa, eta_raw)
                    .unwrap_or_else(|| {
                        closed_form_penalty::pair_block_radial_with_j_second_derivatives_with_powers(
                            q, p_order, s_order, kappa, eta_raw, &powers, &r_eps_buf,
                        )
                    })
            } else {
                closed_form_penalty::pair_block_radial_with_j_second_derivatives_with_powers(
                    q, p_order, s_order, kappa, eta_raw, &powers, &r_buf,
                )
            };
            // SAFETY: every output has k(k+1)/2 slots; for i in 0..k and j ∈ 0..=i,
            // lower_triangular_offset(i)+j is in bounds. Each rayon iteration owns
            // that lower-triangular row in every output, so writes are disjoint.
            unsafe {
                *g_row.add(j) = bundle.value;
                for a in 0..d {
                    *g_eta_rows[a].add(j) = bundle.d_eta[a];
                    *g_eta2_diag_rows[a].add(j) = bundle.d2_eta[a][a];
                }
                for (idx, &(a, b)) in cross_pairs.iter().enumerate() {
                    *g_eta2_cross_rows[idx].add(j) = bundle.d2_eta[a][b];
                }
            }
        }
    });

    let g = symmetric_matrix_from_lower_values(k, &g_values);
    let g_eta: Vec<Array2<f64>> = g_eta_values
        .iter()
        .map(|values| symmetric_matrix_from_lower_values(k, values))
        .collect();
    let g_eta2_diag: Vec<Array2<f64>> = g_eta2_diag_values
        .iter()
        .map(|values| symmetric_matrix_from_lower_values(k, values))
        .collect();
    let g_eta2_cross_unique: Vec<Array2<f64>> = g_eta2_cross_values
        .iter()
        .map(|values| symmetric_matrix_from_lower_values(k, values))
        .collect();

    // Apply Z + poly-pad + T to raw K×K matrices.
    let s = transform_closed_form_raw_block(
        &g,
        kernel_nullspace,
        polynomial_block_cols,
        outer_identifiability,
    );
    let s_first: Vec<Array2<f64>> = g_eta
        .par_iter()
        .map(|raw| {
            transform_closed_form_raw_block(
                raw,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        })
        .collect();
    let s_second_diag: Vec<Array2<f64>> = g_eta2_diag
        .par_iter()
        .map(|raw| {
            transform_closed_form_raw_block(
                raw,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        })
        .collect();
    let transformed_cross_unique: Vec<Array2<f64>> = g_eta2_cross_unique
        .par_iter()
        .map(|raw| {
            transform_closed_form_raw_block(
                raw,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        })
        .collect();
    let out_dim = s.nrows();
    let mut s_second_cross: Vec<Vec<Array2<f64>>> = (0..d)
        .map(|_| {
            (0..d)
                .map(|_| Array2::<f64>::zeros((out_dim, out_dim)))
                .collect()
        })
        .collect();
    for (idx, &(a, b)) in cross_pairs.iter().enumerate() {
        let block = &transformed_cross_unique[idx];
        s_second_cross[a][b] = block.clone();
        if a != b {
            s_second_cross[b][a] = block.clone();
        }
    }
    (s, s_first, s_second_diag, s_second_cross)
}

#[inline(always)]
pub(crate) fn duchon_closed_form_operator_penalty_converges(
    q: usize,
    p_order: usize,
    s_order: f64,
    dimension: usize,
) -> bool {
    // Real-valued conditions so fractional `s_order` falls inside the
    // convergent regime when admissible. Integer values reduce to the
    // original strict inequalities exactly.
    let four_ms = 4.0 * (p_order as f64 + s_order);
    let dp2q = (dimension + 2 * q) as f64;
    let four_m = (4 * p_order) as f64;
    four_ms > dp2q && dp2q > four_m && 2 * p_order >= q + 1
}

/// CPD-adequacy gate for the *scale-free* (`length_scale = None`) Duchon
/// closed-form pair block. The pure-polyharmonic pair block is
/// `S_q[i, j] ∝ R_J^d(|c_i − c_j|)` with `J = 2(p+s) − q`, which is only
/// *conditionally* positive-definite — its projection onto the
/// polynomial-orthogonal complement is PSD iff the spec's polynomial
/// null-space order is at least the kernel's CPD order
/// (Wendland Thm 8.17 / 8.18).
///
/// For the *hybrid* Matérn-blended path (`length_scale = Some`) the kernel
/// is strictly positive definite (Matérn regularization at low
/// frequencies), so no CPD restriction is needed — the UV/IR convergence
/// check is sufficient. This function is therefore only consulted from
/// the pure-Duchon candidate factory.
///
/// Concrete tripwire: at `d=8, p_order=2, s_order=3.5` with the Linear
/// null space, the closed-form pair block for q ∈ {1, 2} has CPD order
/// `(2J − d)/2 + 1 ∈ {7, 6}` (log case fires because `2s = 7` makes
/// `2J − d` an even integer at even `d`). Without this gate the
/// closed-form matrix at centers was non-PSD (15 / 30 negative
/// eigenvalues); with this gate, both q's route to collocation
/// `D_qᵀD_q` (PSD by construction). TPS sanity: `d=2, p=2, s=0, q=2`
/// gives `2J − d = 2`, log case, CPD order = 2, matched exactly by
/// the Linear null space.
pub(crate) fn duchon_pure_closed_form_pair_block_cpd_adequate(
    q: usize,
    p_order: usize,
    s_order: f64,
    dimension: usize,
) -> bool {
    // β = 2J − d where J = 2(p+s) − q. Equivalently β = 4(p+s) − 2q − d.
    let beta = 4.0 * (p_order as f64 + s_order) - 2.0 * q as f64 - dimension as f64;
    if beta < 0.0 {
        return false;
    }
    const LOG_EPS: f64 = 1e-12;
    let n_f = (beta / 2.0).round();
    let is_log_case =
        dimension.is_multiple_of(2) && n_f >= 0.0 && (n_f * 2.0 - beta).abs() < LOG_EPS;
    let cpd_required = if is_log_case {
        // Log case: kernel `c · r^{2n}(ln r + A_n)` is CPD of order n + 1
        // (Wendland Thm 8.18).
        (n_f as usize).saturating_add(1)
    } else {
        // Non-log case: kernel `c · r^β` is CPD of order ⌈(β+1)/2⌉
        // (Wendland Thm 8.17). For odd β this is `(β+1)/2`; for
        // fractional β it rounds up.
        ((beta + 1.0) / 2.0).ceil() as usize
    };
    p_order >= cpd_required
}

pub fn operator_penalty_candidates_closed_form(
    centers: ArrayView2<'_, f64>,
    d0: &Array2<f64>,
    d1: &Array2<f64>,
    d2: &Array2<f64>,
    spec: &DuchonOperatorPenaltySpec,
    p_order: usize,
    s_order: usize,
    length_scale: f64,
    aniso_log_scales: Option<&[f64]>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> Vec<PenaltyCandidate> {
    let kappa = 1.0 / length_scale.max(1e-300);

    // Per-q Duchon convergence regime: closed-form Lebesgue kernel matrix is
    // PSD only when both UV `4(m+s) > d + 2q` and IR `d + 2q > 4m` hold,
    // and the partial-fraction expansion in `isotropic_duchon_penalty`
    // requires `2m - q ≥ 1`. Even-dimensional log-Riesz terms are handled
    // analytically by the canonical finite-part shift in `riesz_kernel_value`.
    // Outside this convergence regime the continuous Lebesgue Gram does not
    // define the desired PSD operator block, so the finite-K collocation
    // `D_q^T D_q` Gram remains the mathematically different object.
    let d = centers.ncols();

    // Threshold for emitting an operator-form handle alongside the dense
    // matrix. Above this raw kernel size, the closed-form factory attaches
    // a `ClosedFormPenaltyOperator` so downstream consumers (PCG-against-
    // implicit-H, Hutchinson EDF) can reuse the operator's matvec without
    // rebuilding the dense Gram. Below threshold, dense-only is preserved
    // (Cholesky on the small materialized H is faster).
    let emit_operator = centers.nrows() > CLOSED_FORM_OPERATOR_THRESHOLD;

    let make_op =
        |q: usize, c: f64| -> Option<std::sync::Arc<dyn crate::terms::penalties::op::PenaltyOp>> {
            if !emit_operator {
                return None;
            }
            if !duchon_closed_form_operator_penalty_converges(q, p_order, s_order as f64, d) {
                return None;
            }
            let raw_op = std::sync::Arc::new(
                crate::terms::basis::closed_form_operator::ClosedFormPenaltyOperator::new(
                    centers,
                    q,
                    p_order,
                    s_order,
                    kappa,
                    aniso_log_scales,
                    kernel_nullspace,
                    polynomial_block_cols,
                    outer_identifiability,
                ),
            );
            // The candidate's `matrix` is the closed-form Gram divided by its
            // Frobenius norm `c`. Wrap in `ScaledPenaltyOp` with factor `1/c`
            // so `op.as_dense()` matches the candidate's dense matrix.
            let scale = if c > 1e-12 { 1.0 / c } else { 1.0 };
            let scaled: std::sync::Arc<dyn crate::terms::penalties::op::PenaltyOp> =
                std::sync::Arc::new(crate::terms::penalties::op::ScaledPenaltyOp::new(
                    raw_op, scale,
                ));
            Some(scaled)
        };

    // Each order is materialized ONLY when its spec is active, so a disabled
    // order never touches its `d_q` operand (lets the caller build `D_q` with
    // `max_op = max active order` and skip the `O(d²)`-row Hessian — the
    // `D2`-skip). Mass is the *centered* collocation Gram `Σ(f−f̄)²`, identical
    // to the pure path, so the constant direction is genuinely unpenalized.
    let mut out = Vec::new();
    if matches!(spec.mass, OperatorPenaltySpec::Active { .. }) {
        let (s0, c0) = normalize_penalty(&symmetrize(&centered_design_gram(d0)));
        out.push(PenaltyCandidate {
            matrix: s0,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
            kronecker_factors: None,
            op: None,
        });
    }
    if matches!(spec.tension, OperatorPenaltySpec::Active { .. }) {
        let s1_raw = if duchon_closed_form_operator_penalty_converges(1, p_order, s_order as f64, d)
        {
            closed_form_operator_penalty_in_total_basis(
                centers,
                1,
                p_order,
                s_order,
                kappa,
                aniso_log_scales,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        } else {
            symmetrize(&fast_ata(d1))
        };
        let (s1, c1) = normalize_penalty(&s1_raw);
        let op = make_op(1, c1);
        out.push(PenaltyCandidate {
            matrix: s1,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
            kronecker_factors: None,
            op,
        });
    }
    if matches!(spec.stiffness, OperatorPenaltySpec::Active { .. }) {
        let s2_raw = if duchon_closed_form_operator_penalty_converges(2, p_order, s_order as f64, d)
        {
            closed_form_operator_penalty_in_total_basis(
                centers,
                2,
                p_order,
                s_order,
                kappa,
                aniso_log_scales,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        } else {
            symmetrize(&fast_ata(d2))
        };
        let (s2, c2) = normalize_penalty(&s2_raw);
        let op = make_op(2, c2);
        out.push(PenaltyCandidate {
            matrix: s2,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
            kronecker_factors: None,
            op,
        });
    }
    out
}

/// Threshold above which the closed-form factory emits an operator-form `op`
/// handle alongside the dense matrix. Above 1500 raw kernel rows, downstream
/// consumers (PCG-against-implicit-H, Hutchinson EDF) reuse the operator's
/// matvec without rebuilding the dense Gram. Below it, only the dense form is
/// emitted — direct Cholesky on the small materialized H is faster than
/// PCG-against-implicit-H. The crossover was measured by
/// `bench_hessian_solve_dense_vs_implicit` in `benches/closed_form_criterion.rs`
/// against the synthetic SPD-with-coupled-penalty fixture there.
pub(crate) const CLOSED_FORM_OPERATOR_THRESHOLD: usize = 1500;

/// Pure-Duchon (κ=0 / `length_scale = None`) counterpart of
/// [`closed_form_operator_penalty_in_total_basis`]. Uses
/// [`closed_form_anisotropic_pair_block_pure`] to evaluate the closed-form
/// penalty via analytic radial derivatives of the pure-Riesz kernel, which
/// is finite for R > 0 in any (m, s, d, q) regime where
/// `radial_derivatives_of_isotropic_duchon` is defined. Self-pair (R=0)
/// regularization is handled inside the pair-block routine.
pub fn closed_form_operator_penalty_in_total_basis_pure(
    centers: ArrayView2<'_, f64>,
    q: usize,
    p_order: usize,
    s_order: f64,
    aniso_log_scales: Option<&[f64]>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> Array2<f64> {
    // The whole scale-free Duchon chain — pair block, anisotropic radial,
    // uniform-metric branch, isotropic radial derivatives, Riesz kernel —
    // is now `f64`-threaded for `kappa = 0`. The integer-only self-pair /
    // partial-fraction helpers are still consulted opportunistically
    // (`s_int` gating inside the pair block) when `s` happens to be
    // whole-valued, but fractional `s` falls through cleanly to the
    // ε-regularized analytic radial chain.
    assert!(
        s_order.is_finite() && s_order >= 0.0,
        "closed_form_operator_penalty_in_total_basis_pure: s_order must be finite and ≥ 0, got {s_order}"
    );
    let g_raw =
        closed_form_anisotropic_pair_block_pure(centers, q, p_order, s_order, aniso_log_scales);
    let g_kernel = if let Some(z) = kernel_nullspace {
        let zt_g = fast_atb(z, &g_raw);
        fast_ab(&zt_g, z)
    } else {
        g_raw
    };
    let kernel_cols = g_kernel.nrows();
    let total_pre_cols = kernel_cols + polynomial_block_cols;
    let g_padded = if polynomial_block_cols == 0 {
        g_kernel
    } else {
        let mut padded = Array2::<f64>::zeros((total_pre_cols, total_pre_cols));
        padded
            .slice_mut(s![0..kernel_cols, 0..kernel_cols])
            .assign(&g_kernel);
        padded
    };
    let g_total = if let Some(t) = outer_identifiability {
        let tt_g = fast_atb(t, &g_padded);
        fast_ab(&tt_g, t)
    } else {
        g_padded
    };
    symmetrize(&g_total)
}

/// Pure-Duchon (κ=0 / `length_scale = None`) counterpart of
/// [`operator_penalty_candidates_closed_form`].
///
/// Builds the three operator penalty candidates (mass, tension, stiffness)
/// from the pure-Duchon closed-form path, which is a polyharmonic of order
/// `m + s = p_order + s_order` with no Matérn factor. `q ∈ {0,1,2}` rolls
/// the penalty differential order — same as the hybrid path.
///
/// The closed-form Lebesgue penalty is finite only when the Duchon
/// convergence conditions hold:
///   - UV (smoothness at origin): `4(m+s) > d + 2q`
///   - IR (decay at infinity):    `d + 2q > 4m`
/// When either condition fails for a given `q`, the closed-form integrand
/// diverges (UV) or vanishes identically (IR — kernel is a finite-degree
/// polynomial that drops out of `Δ_B^q`); in those regimes we fall back to
/// the collocation Gram `D_q^T D_q`, which is the same regularization the
/// pre-closed-form path used.
pub fn operator_penalty_candidates_closed_form_pure(
    centers: ArrayView2<'_, f64>,
    d0: &Array2<f64>,
    d1: &Array2<f64>,
    d2: &Array2<f64>,
    spec: &DuchonOperatorPenaltySpec,
    p_order: usize,
    s_order: f64,
    aniso_log_scales: Option<&[f64]>,
    kernel_nullspace: Option<&Array2<f64>>,
    polynomial_block_cols: usize,
    outer_identifiability: Option<&Array2<f64>>,
) -> Vec<PenaltyCandidate> {
    // q=0 mass is the *centered* collocation Gram — the data-density-weighted
    // spring penalty on deviations from the function's mean over the
    // collocation sites. Centering each design column by its mean across rows
    // before forming the Gram puts the constant direction exactly into the
    // penalty's null space (intercept genuinely unpenalized): for the
    // constant basis column (all-ones), the column mean is one and the
    // centered column is identically zero, so the resulting Gram row/column
    // for that direction is zero. Algebraically this is
    // `(D_0 - 1 μ')^T (D_0 - 1 μ')` where `μ_j = (1/N) Σ_i D_0[i, j]`. This
    // expresses the "springs to a floating flat sheet" semantics — the level
    // is free and only deviations get the spring force — while staying inside
    // the standard quadratic-penalty machinery.
    let d = centers.ncols();
    // Convergence predicate also requires `isotropic_duchon_penalty`'s
    // partial-fraction precondition `2m ≥ q + 1`; without it, closed-form
    // panics on configs like m=1, q=2. Even-dimensional log-Riesz branches are
    // admitted because `riesz_kernel_value` now uses the canonical finite part.
    // Closed-form pair block is admitted only when both the UV/IR
    // convergence predicate AND Wendland's CPD-adequacy condition hold;
    // the second guards against silently-non-PSD pair blocks when the
    // polynomial null space is too small to absorb the kernel's CPD
    // order (e.g. d=8, p_order=2, s_order=3.5 log-case). Failing
    // either test routes to collocation `D_qᵀD_q`, which is PSD by
    // construction.
    let closed_form_ok = |q: usize| -> bool {
        duchon_closed_form_operator_penalty_converges(q, p_order, s_order, d)
            && duchon_pure_closed_form_pair_block_cpd_adequate(q, p_order, s_order, d)
    };
    // Each order is materialized ONLY when its spec is active: a disabled order
    // never touches its `d_q` operand, so the caller can build `D_q` with
    // `max_op = max active order` and leave the higher-order designs empty (the
    // `D2`-skip — decisive in high `d`, where the Hessian has `O(d²)` rows).
    let mut out = Vec::new();
    if matches!(spec.mass, OperatorPenaltySpec::Active { .. }) {
        // q=0 mass is the *centered* collocation Gram — the data-density-weighted
        // spring penalty on deviations from the function's mean over the
        // collocation sites; centering puts the constant direction exactly into
        // the penalty null space (intercept genuinely unpenalized).
        let (s0, c0) = normalize_penalty(&symmetrize(&centered_design_gram(d0)));
        out.push(PenaltyCandidate {
            matrix: s0,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
            kronecker_factors: None,
            op: None,
        });
    }
    if matches!(spec.tension, OperatorPenaltySpec::Active { .. }) {
        let s1_raw = if closed_form_ok(1) {
            closed_form_operator_penalty_in_total_basis_pure(
                centers,
                1,
                p_order,
                s_order,
                aniso_log_scales,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        } else {
            symmetrize(&fast_ata(d1))
        };
        let (s1, c1) = normalize_penalty(&s1_raw);
        out.push(PenaltyCandidate {
            matrix: s1,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
            kronecker_factors: None,
            op: None,
        });
    }
    if matches!(spec.stiffness, OperatorPenaltySpec::Active { .. }) {
        let s2_raw = if closed_form_ok(2) {
            closed_form_operator_penalty_in_total_basis_pure(
                centers,
                2,
                p_order,
                s_order,
                aniso_log_scales,
                kernel_nullspace,
                polynomial_block_cols,
                outer_identifiability,
            )
        } else {
            symmetrize(&fast_ata(d2))
        };
        let (s2, c2) = normalize_penalty(&s2_raw);
        out.push(PenaltyCandidate {
            matrix: s2,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
            kronecker_factors: None,
            op: None,
        });
    }
    out
}

pub(crate) fn operator_penalty_candidates_from_collocation(
    d0: &Array2<f64>,
    d1: &Array2<f64>,
    d2: &Array2<f64>,
    spec: &DuchonOperatorPenaltySpec,
) -> Vec<PenaltyCandidate> {
    let s0_raw = symmetrize(&fast_ata(d0));
    let (s0, c0) = normalize_penalty(&s0_raw);
    let (s1, c1) = normalize_penalty(&symmetrize(&fast_ata(d1)));
    let (s2, c2) = normalize_penalty(&symmetrize(&fast_ata(d2)));
    let mut out = Vec::new();
    if matches!(spec.mass, OperatorPenaltySpec::Active { .. }) {
        out.push(PenaltyCandidate {
            matrix: s0,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorMass,
            normalization_scale: c0,
            kronecker_factors: None,
            op: None,
        });
    }
    if matches!(spec.tension, OperatorPenaltySpec::Active { .. }) {
        out.push(PenaltyCandidate {
            matrix: s1,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorTension,
            normalization_scale: c1,
            kronecker_factors: None,
            op: None,
        });
    }
    if matches!(spec.stiffness, OperatorPenaltySpec::Active { .. }) {
        out.push(PenaltyCandidate {
            matrix: s2,
            nullspace_dim_hint: 0,
            source: PenaltySource::OperatorStiffness,
            normalization_scale: c2,
            kronecker_factors: None,
            op: None,
        });
    }
    out
}

pub(crate) fn active_operator_penalty_derivatives(
    penaltyinfo: &[PenaltyInfo],
    operator_derivatives: &[Array2<f64>],
    label: &str,
) -> Result<Vec<Array2<f64>>, BasisError> {
    if operator_derivatives.len() != 3 {
        crate::bail_invalid_basis!(
            "{label} operator derivative path requires 3 canonical penalties; found {}",
            operator_derivatives.len()
        );
    }

    penaltyinfo
        .iter()
        .filter(|info| info.active)
        .map(|info| match &info.source {
            PenaltySource::OperatorMass => Ok(operator_derivatives[0].clone()),
            PenaltySource::OperatorTension => Ok(operator_derivatives[1].clone()),
            PenaltySource::OperatorStiffness => Ok(operator_derivatives[2].clone()),
            other => Err(BasisError::InvalidInput(format!(
                "unexpected {label} penalty source in canonical operator path: {other:?}"
            ))),
        })
        .collect()
}

pub(crate) fn frozen_spatial_identifiability_transform(
    identifiability: &SpatialIdentifiability,
    expectedrows: usize,
    label: &str,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None | SpatialIdentifiability::OrthogonalToParametric => Ok(None),
        SpatialIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != expectedrows {
                crate::bail_dim_basis!(
                    "frozen {label} identifiability transform mismatch: rows={}, expected {expectedrows}",
                    transform.nrows()
                );
            }
            Ok(Some(transform.clone()))
        }
    }
}

/// Returns the parametric-constraint columns used by the standalone
/// `OrthogonalToParametric` spatial identifiability transform.
///
/// **What it contains:** a single all-ones column (the global intercept).
///
/// **What it consumes vs. what it does not:**
///
/// - *Consumes* (orthogonalises the smooth against): the intercept direction
///   only.  After this transform the smooth columns have zero unweighted mean,
///   so they cannot absorb a global additive constant that belongs to the
///   intercept parameter.
///
/// - *Does not consume*: the full polynomial null space of the Duchon kernel
///   (constants + linear + higher-order monomials).  The linear and higher
///   monomial directions in `[1, x₁, …, x_d]` are already handled by the
///   kernel side-condition projection inside `kernel_constraint_nullspace` —
///   that step compresses the radial kernel block from `k` columns down to
///   `k − C(d+r, r)` columns, so those directions never appear as free
///   smooth columns in the first place.  The spatial identifiability transform
///   only needs to remove the global-intercept residual left over after that
///   projection.
///
/// - *Does not consume*: cross-block aliases that arise when the same Duchon
///   smooth appears in multiple formula channels (e.g. marginal and logslope).
///   Two channels with identical raw bases have cosine-similarity 1.0 between
///   the corresponding columns; that aliasing is detected and resolved by the
///   joint cross-block identifiability audit (`audit_identifiability` /
///   `audit_identifiability_channel_aware`), not here.
pub(crate) fn spatial_parametric_constraint_block(data: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = data.nrows();
    Array2::<f64>::ones((n, 1))
}

pub(crate) fn build_thin_plate_penalty_matrices(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    kernel_transform: &Array2<f64>,
    double_penalty: bool,
) -> Result<(Array2<f64>, Option<Array2<f64>>), BasisError> {
    let k = centers.nrows();
    let d = centers.ncols();
    let kernel_cols = kernel_transform.ncols();
    let poly_cols = thin_plate_polynomial_basis_dimension(d);
    let total_cols = kernel_cols + poly_cols;
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
    let omega_constrained = {
        let zt_o = fast_atb(kernel_transform, &omega);
        // `kernel_transform` spans the side-constraint nullspace, so the
        // congruence transform preserves the thin-plate PSD construction.
        // Symmetrize to remove roundoff asymmetry without paying for a full EVD
        // on the large lazy-path penalty.
        symmetrize_penalty(&fast_ab(&zt_o, kernel_transform))
    };
    let mut penalty_bending = Array2::<f64>::zeros((total_cols, total_cols));
    penalty_bending
        .slice_mut(s![0..kernel_cols, 0..kernel_cols])
        .assign(&omega_constrained);
    let penalty_ridge = if double_penalty {
        build_nullspace_shrinkage_penalty(&penalty_bending)?.map(|block| block.sym_penalty)
    } else {
        None
    };
    Ok((penalty_bending, penalty_ridge))
}

/// Drop redundant Matérn centers when an over-specified `centers=K` exceeds the
/// kernel's numerical rank on the data cloud (#755).
///
/// The Matérn kernel has a fixed `length_scale` (default 1.0 on standardized
/// inputs), so packing too many centers into a tight data cloud produces
/// overlapping, near-identical radial basis functions. The realized kernel
/// design `K(data, centers)` then carries exactly linearly-dependent columns,
/// which the downstream identifiability audit hard-FATALs as intra-block rank
/// deficiency. (Duchon is scale-free and never hits this.)
///
/// We detect the deficiency on the *realized* kernel design block (the same
/// matrix the audit RRQRs, before the identifiability transform) via a
/// column-pivoted rank-revealing QR at the crate-standard rank tolerance, so
/// the reduction fires exactly when — and only when — the audit would have
/// FATAL'd. When `rank < K`, we keep the leading `rank` pivoted centers
/// (restored to ascending original order so the basis layout stays
/// deterministic) and drop the redundant remainder. Returning a full-rank
/// center subset keeps the design, penalty, and identifiability machinery
/// mutually consistent because they are all rebuilt from the same centers.
pub(crate) fn matern_rank_reduce_centers(
    data: ArrayView2<'_, f64>,
    centers: &Array2<f64>,
    length_scale: f64,
    nu: MaternNu,
    aniso_log_scales: Option<&[f64]>,
) -> Result<Array2<f64>, BasisError> {
    let k = centers.nrows();
    let n = data.nrows();
    // Need at least as many rows as columns for a column rank to be meaningful;
    // the kernel design is n × K, and a 0/1-center basis can never be deficient.
    if k <= 1 || n < k {
        return Ok(centers.clone());
    }
    let mut kernel_block = Array2::<f64>::zeros((n, k));
    let axis_scales = aniso_log_scales.map(aniso_axis_scales);
    let centers_view = centers.view();
    kernel_block
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            for j in 0..k {
                let r = if let Some(scales) = axis_scales.as_deref() {
                    aniso_distance_rows_with_scales(data, i, centers_view, j, scales)
                } else {
                    euclidean_distance_rows(data, i, centers_view, j)
                };
                row[j] = matern_kernel_from_distance(r, length_scale, nu)?;
            }
            Ok::<(), BasisError>(())
        })?;
    let rrqr = rrqr_with_permutation(&kernel_block, default_rrqr_rank_alpha())
        .map_err(BasisError::LinalgError)?;
    if rrqr.rank >= k {
        return Ok(centers.clone());
    }
    // Rank 0 means the realized kernel design has no numerically independent
    // columns at all: every center collapses to the same (near-constant) radial
    // response on this data cloud at the chosen `length_scale`, so the Matérn
    // term carries no usable signal. Emitting a 0-center basis here would leave a
    // degenerate term whose 0-column design desyncs against its identifiability
    // transform and silently corrupts the fit (#1090). Fail loudly with an
    // actionable message instead — a length_scale this large relative to the data
    // spread (or a near-degenerate coordinate cloud) needs the user to widen the
    // domain, shrink the length scale, or drop the term.
    if rrqr.rank == 0 {
        crate::bail_invalid_basis!(
            "Matérn smooth has data-supported numerical rank 0: all {k} center(s) are \
             numerically collinear at length_scale={length_scale} on this data cloud, so the \
             kernel basis is degenerate (no independent columns). Reduce length_scale, spread \
             the coordinate cloud, or drop this term (#1090/#755)."
        );
    }
    let mut keep = rrqr.column_permutation[..rrqr.rank].to_vec();
    keep.sort_unstable();
    log::info!(
        "Matérn centers reduced from {k} to {} (data-supported numerical rank): \
         requested centers exceed the kernel's rank at length_scale={length_scale}, so \
         {} collinear basis column(s) were dropped to keep the basis full-rank (#755).",
        rrqr.rank,
        k - rrqr.rank,
    );
    let mut reduced = Array2::<f64>::zeros((keep.len(), centers.ncols()));
    for (new_row, &old_row) in keep.iter().enumerate() {
        reduced.row_mut(new_row).assign(&centers.row(old_row));
    }
    Ok(reduced)
}

pub(crate) fn build_matern_kernel_penalty(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    aniso_log_scales: Option<&[f64]>,
) -> Result<Array2<f64>, BasisError> {
    let k = centers.nrows();
    let total_cols = k + usize::from(include_intercept);
    let mut center_kernel = Array2::<f64>::zeros((k, k));
    let axis_scales = aniso_log_scales.map(aniso_axis_scales);
    fill_symmetric_from_row_kernel(&mut center_kernel, |i, j| {
        let r = if let Some(scales) = axis_scales.as_deref() {
            aniso_distance_rows_with_scales(centers, i, centers, j, scales)
        } else {
            euclidean_distance_rows(centers, i, centers, j)
        };
        matern_kernel_from_distance(r, length_scale, nu)
    })?;
    let mut penalty_kernel = Array2::<f64>::zeros((total_cols, total_cols));
    penalty_kernel
        .slice_mut(s![0..k, 0..k])
        .assign(&center_kernel);
    Ok(penalty_kernel)
}

/// Compute the spatial identifiability transform for a dense design matrix.
///
/// For the `OrthogonalToParametric` policy the transform orthogonalises `design`
/// against the **intercept only** (a column of ones built from `data`).  This
/// removes one direction from the smooth's column space, so a basis with
/// pre-transform width `p` yields a post-transform width of `p − 1`.
///
/// The polynomial null space of the Duchon kernel is consumed *upstream* by
/// `kernel_constraint_nullspace`, not here.  See
/// [`spatial_parametric_constraint_block`] for a precise description of what
/// this step does and does not consume.
pub(crate) fn spatial_identifiability_transform_from_design(
    data: ArrayView2<'_, f64>,
    design: ArrayView2<'_, f64>,
    identifiability: &SpatialIdentifiability,
    label: &str,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None => Ok(None),
        SpatialIdentifiability::OrthogonalToParametric => {
            let c = spatial_parametric_constraint_block(data);
            let (_, z) = applyweighted_orthogonality_constraint(design, c.view(), None)?;
            Ok(Some(z))
        }
        SpatialIdentifiability::FrozenTransform { .. } => {
            frozen_spatial_identifiability_transform(identifiability, design.ncols(), label)
        }
    }
}

pub(crate) fn spatial_identifiability_transform_from_design_matrix(
    data: ArrayView2<'_, f64>,
    design: &DesignMatrix,
    identifiability: &SpatialIdentifiability,
    label: &str,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None => Ok(None),
        SpatialIdentifiability::OrthogonalToParametric => {
            let c = spatial_parametric_constraint_block(data);
            let z = orthogonality_transform_for_design(design, c.view(), None)?;
            Ok(Some(z))
        }
        SpatialIdentifiability::FrozenTransform { .. } => {
            frozen_spatial_identifiability_transform(identifiability, design.ncols(), label)
        }
    }
}

pub(crate) fn thin_plate_intercept_transform_from_column_means(
    column_means: &Array1<f64>,
    kernel_cols: usize,
    poly_cols: usize,
) -> Result<Array2<f64>, BasisError> {
    let total_cols = kernel_cols + poly_cols;
    if column_means.len() != total_cols {
        crate::bail_dim_basis!(
            "thin-plate column-mean length mismatch: got {}, expected {total_cols}",
            column_means.len()
        );
    }
    if poly_cols == 0 {
        return Ok(Array2::<f64>::eye(total_cols));
    }
    let out_cols = total_cols
        .checked_sub(1)
        .ok_or_else(|| BasisError::InvalidInput("thin-plate basis has no columns".to_string()))?;
    let mut transform = Array2::<f64>::zeros((total_cols, out_cols));

    for j in 0..kernel_cols {
        transform[[j, j]] = 1.0;
        transform[[kernel_cols, j]] = -column_means[j];
    }
    for poly_j in 1..poly_cols {
        let src = kernel_cols + poly_j;
        let dst = kernel_cols + poly_j - 1;
        transform[[src, dst]] = 1.0;
        transform[[kernel_cols, dst]] = -column_means[src];
    }
    Ok(transform)
}

pub(crate) fn thin_plate_identifiability_transform_from_design(
    design: ArrayView2<'_, f64>,
    kernel_cols: usize,
    poly_cols: usize,
    identifiability: &SpatialIdentifiability,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None => Ok(None),
        SpatialIdentifiability::OrthogonalToParametric => {
            let n = design.nrows();
            if n == 0 {
                crate::bail_invalid_basis!("thin-plate identifiability requires at least one row");
            }
            let means = design.sum_axis(Axis(0)).mapv(|v| v / n as f64);
            Ok(Some(thin_plate_intercept_transform_from_column_means(
                &means,
                kernel_cols,
                poly_cols,
            )?))
        }
        SpatialIdentifiability::FrozenTransform { .. } => {
            frozen_spatial_identifiability_transform(identifiability, design.ncols(), "ThinPlate")
        }
    }
}

pub(crate) fn thin_plate_identifiability_transform_from_design_matrix(
    design: &DesignMatrix,
    kernel_cols: usize,
    poly_cols: usize,
    identifiability: &SpatialIdentifiability,
) -> Result<Option<Array2<f64>>, BasisError> {
    match identifiability {
        SpatialIdentifiability::None => Ok(None),
        SpatialIdentifiability::OrthogonalToParametric => {
            let n = design.nrows();
            if n == 0 {
                crate::bail_invalid_basis!("thin-plate identifiability requires at least one row");
            }
            let ones = Array1::<f64>::ones(n);
            let means = design.apply_transpose(&ones).mapv(|v| v / n as f64);
            Ok(Some(thin_plate_intercept_transform_from_column_means(
                &means,
                kernel_cols,
                poly_cols,
            )?))
        }
        SpatialIdentifiability::FrozenTransform { .. } => {
            frozen_spatial_identifiability_transform(identifiability, design.ncols(), "ThinPlate")
        }
    }
}

pub(crate) fn append_intercept_to_transform(transform: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((transform.nrows() + 1, transform.ncols() + 1));
    out.slice_mut(s![0..transform.nrows(), 0..transform.ncols()])
        .assign(transform);
    out[[transform.nrows(), transform.ncols()]] = 1.0;
    out
}

pub(crate) fn project_penalty_matrix(
    matrix: &Array2<f64>,
    transform: Option<&Array2<f64>>,
) -> Array2<f64> {
    let projected = if let Some(z) = transform {
        let zt_s = z.t().dot(matrix);
        zt_s.dot(z)
    } else {
        matrix.clone()
    };
    symmetrize(&projected)
}

pub(crate) fn normalize_penalty_candidate(
    matrix: Array2<f64>,
    nullspace_dim_hint: usize,
    source: PenaltySource,
) -> PenaltyCandidate {
    let (matrix, normalization_scale) = if matrix.iter().all(|v| v.abs() <= 1e-12) {
        (matrix, 1.0)
    } else {
        normalize_penalty(&matrix)
    };
    PenaltyCandidate {
        matrix,
        nullspace_dim_hint,
        source,
        normalization_scale,
        kronecker_factors: None,
        op: None,
    }
}

pub fn build_matern_collocation_operator_matrices(
    centers: ArrayView2<'_, f64>,
    collocationweights: Option<ArrayView1<'_, f64>>,
    length_scale: f64,
    nu: MaternNu,
    include_intercept: bool,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
    aniso_log_scales: Option<&[f64]>,
) -> Result<CollocationOperatorMatrices, BasisError> {
    // Specialized Matérn operator assembly using explicit half-integer formulas:
    // - one exp(-a) and small polynomials per pair,
    // - NaN-safe phi'(r)/r without dividing by r for nu>=3/2,
    // - exact Hessian rows for the stiffness operator, not just the Laplacian.
    let p = centers.nrows();
    let d = centers.ncols();
    let row_scales = if let Some(w) = collocationweights {
        if w.len() != p {
            crate::bail_dim_basis!(
                "collocation weight length mismatch: got {}, expected {p}",
                w.len()
            );
        }
        let mut out = Vec::with_capacity(p);
        for &wk in w {
            if !wk.is_finite() || wk < 0.0 {
                crate::bail_invalid_basis!(
                    "collocation weights must be finite and non-negative; got {wk}"
                );
            }
            out.push(wk.sqrt());
        }
        out
    } else {
        vec![1.0; p]
    };
    let mut d0_raw = Array2::<f64>::zeros((p, p));
    let mut d1_raw = Array2::<f64>::zeros((p * d, p));
    let mut d2_raw = Array2::<f64>::zeros((p * d * d, p));
    let metric_weights = aniso_log_scales.map(centered_aniso_metric_weights);
    const R_EPS: f64 = 1e-12;
    // Row blocks are independent: output rows [k] in d0, [k*d..(k+1)*d] in d1,
    // and [k*d*d..(k+1)*d*d] in d2 are disjoint for each collocation row k.
    // Keep small assemblies serial to avoid Rayon scheduling overhead.
    const MATERN_COLLOCATION_PAR_WORK_THRESHOLD: usize = 32_768;
    const MATERN_COLLOCATION_ROW_BLOCK: usize = 32;
    let assembly_work = p
        .saturating_mul(p)
        .saturating_mul(d.max(1))
        .saturating_mul(d.max(1));
    let row_block_size = MATERN_COLLOCATION_ROW_BLOCK.min(p.max(1));
    let assemble_chunk = |ci: usize,
                          mut d0_chunk: ArrayViewMut2<'_, f64>,
                          mut d1_chunk: ArrayViewMut2<'_, f64>,
                          mut d2_chunk: ArrayViewMut2<'_, f64>|
     -> Result<(), BasisError> {
        let chunk_start = ci * row_block_size;
        for local_k in 0..d0_chunk.nrows() {
            let k = chunk_start + local_k;
            let scale_k = row_scales[k];
            for j in 0..p {
                // Distance: anisotropic r = |Ah| when eta present, isotropic |h| otherwise.
                let r = if let Some(eta) = aniso_log_scales {
                    aniso_distance_and_components(
                        centers.row(k).as_slice().unwrap(),
                        centers.row(j).as_slice().unwrap(),
                        eta,
                    )
                    .0
                } else {
                    stable_euclidean_norm((0..d).map(|c| centers[[k, c]] - centers[[j, c]]))
                };
                if matches!(nu, MaternNu::Half) && r <= R_EPS && d > 1 {
                    crate::bail_invalid_basis!(
                        "Matérn nu=1/2 has singular Laplacian at center collisions for d>1; choose nu>=3/2 or avoid collocation at centers"
                    );
                }
                let (phi, _, phi_rr, phi_r_over_r) =
                    if matches!(nu, MaternNu::Half) && r <= R_EPS && d == 1 {
                        // In 1D: Delta phi = phi'' and the singular phi'/r term is absent.
                        let s = 1.0 / length_scale;
                        let e = 1.0;
                        (e, -s * e, s * s * e, 0.0)
                    } else {
                        matern_kernel_radial_tripletwith_safe_ratio(r, length_scale, nu)?
                    };
                d0_chunk[[local_k, j]] = scale_k * phi;
                if r > R_EPS {
                    for c in 0..d {
                        let delta = centers[[k, c]] - centers[[j, c]];
                        d1_chunk[[local_k * d + c, j]] = scale_k * phi_r_over_r * delta;
                    }
                } else {
                    // Symmetry at center-center coincidence.
                    for c in 0..d {
                        d1_chunk[[local_k * d + c, j]] = 0.0;
                    }
                }
                let t = if r > R_EPS {
                    (phi_rr - phi_r_over_r) / (r * r)
                } else {
                    0.0
                };
                for a in 0..d {
                    let h_a = centers[[k, a]] - centers[[j, a]];
                    let w_a = metric_weights.as_ref().map(|w| w[a]).unwrap_or(1.0);
                    for b in 0..d {
                        let h_b = centers[[k, b]] - centers[[j, b]];
                        let w_b = metric_weights.as_ref().map(|w| w[b]).unwrap_or(1.0);
                        let diagonal = if a == b { phi_r_over_r * w_a } else { 0.0 };
                        let mixed = if r > R_EPS {
                            t * w_a * h_a * w_b * h_b
                        } else {
                            0.0
                        };
                        let row = (local_k * d + a) * d + b;
                        d2_chunk[[row, j]] = scale_k * (diagonal + mixed);
                    }
                }
                if !d0_chunk[[local_k, j]].is_finite()
                    || ((local_k * d * d)..((local_k + 1) * d * d))
                        .any(|row| !d2_chunk[[row, j]].is_finite())
                {
                    crate::bail_invalid_basis!(
                        "non-finite Matérn collocation operator entry at row={k}, col={j}, r={r}, nu={nu:?}"
                    );
                }
            }
        }
        Ok(())
    };
    if d == 0 && p > 0 {
        for k in 0..p {
            let scale_k = row_scales[k];
            for j in 0..p {
                let (phi, _, _, _) =
                    matern_kernel_radial_tripletwith_safe_ratio(0.0, length_scale, nu)?;
                d0_raw[[k, j]] = scale_k * phi;
                if !d0_raw[[k, j]].is_finite() {
                    crate::bail_invalid_basis!(
                        "non-finite Matérn collocation operator entry at row={k}, col={j}, r=0, nu={nu:?}"
                    );
                }
            }
        }
    } else if assembly_work >= MATERN_COLLOCATION_PAR_WORK_THRESHOLD && p > 1 {
        d0_raw
            .axis_chunks_iter_mut(Axis(0), row_block_size)
            .into_par_iter()
            .zip(
                d1_raw
                    .axis_chunks_iter_mut(Axis(0), row_block_size * d)
                    .into_par_iter(),
            )
            .zip(
                d2_raw
                    .axis_chunks_iter_mut(Axis(0), row_block_size * d * d)
                    .into_par_iter(),
            )
            .enumerate()
            .try_for_each(|(ci, ((d0_chunk, d1_chunk), d2_chunk))| {
                assemble_chunk(ci, d0_chunk, d1_chunk, d2_chunk)
            })?;
    } else if p > 0 {
        d0_raw
            .axis_chunks_iter_mut(Axis(0), row_block_size)
            .zip(d1_raw.axis_chunks_iter_mut(Axis(0), row_block_size * d))
            .zip(d2_raw.axis_chunks_iter_mut(Axis(0), row_block_size * d * d))
            .enumerate()
            .try_for_each(|(ci, ((d0_chunk, d1_chunk), d2_chunk))| {
                assemble_chunk(ci, d0_chunk, d1_chunk, d2_chunk)
            })?;
    }
    let (d0_kernel, d1_kernel, d2_kernel) = if let Some(z) = identifiability_transform {
        let z = z.to_owned();
        (
            fast_ab(&d0_raw, &z),
            fast_ab(&d1_raw, &z),
            fast_ab(&d2_raw, &z),
        )
    } else {
        (d0_raw, d1_raw, d2_raw)
    };
    let p_colloc = centers.nrows();
    let dim = centers.ncols();
    let kernel_cols = d0_kernel.ncols();
    let total_cols = kernel_cols + usize::from(include_intercept);
    let mut d0 = Array2::<f64>::zeros((p_colloc, total_cols));
    let mut d1 = Array2::<f64>::zeros((p_colloc * dim, total_cols));
    let mut d2 = Array2::<f64>::zeros((p_colloc * dim * dim, total_cols));
    d0.slice_mut(s![.., 0..kernel_cols]).assign(&d0_kernel);
    d1.slice_mut(s![.., 0..kernel_cols]).assign(&d1_kernel);
    d2.slice_mut(s![.., 0..kernel_cols]).assign(&d2_kernel);
    if include_intercept {
        for (k, &scale_k) in row_scales.iter().enumerate() {
            d0[[k, kernel_cols]] = scale_k;
        }
    }
    Ok(CollocationOperatorMatrices {
        d0,
        d1,
        d2,
        collocation_points: centers.to_owned(),
        kernel_nullspace_transform: None,
        polynomial_block_cols: usize::from(include_intercept),
    })
}
