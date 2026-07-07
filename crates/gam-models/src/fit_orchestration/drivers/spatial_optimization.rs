fn try_build_spatial_term_log_kappa_derivative(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    term_idx: usize,
) -> Result<
    Option<(
        Range<usize>,
        usize,
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
        Vec<Array2<f64>>,
        Vec<Array2<f64>>,
        Option<std::sync::Arc<gam_terms::basis::ImplicitDesignPsiDerivative>>,
    )>,
    EstimationError,
> {
    let Some(smooth_term) = design.smooth.terms.get(term_idx) else {
        return Ok(None);
    };
    let Some(termspec) = resolvedspec.smooth_terms.get(term_idx) else {
        return Ok(None);
    };

    let derivative_bundle = match &termspec.basis {
        SmoothBasisSpec::ThinPlate {
            feature_cols,
            spec,
            input_scales,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            let mut spec_local = spec.clone();
            if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
                spec_local.length_scale =
                    compensate_length_scale_for_standardization(spec.length_scale, s);
            }
            build_thin_plate_basis_log_kappa_derivatives(x.view(), &spec_local)
                .map_err(EstimationError::from)?
        }
        SmoothBasisSpec::Sphere { .. } => return Ok(None),
        // Constant-curvature smooths expose κ as one signed, design-moving
        // outer ψ-coordinate (#944 stage 3 final wiring). Unlike the Matérn /
        // Duchon / TPS kernels — whose ψ-coordinate is `log κ = −log ℓ` — the
        // constant-curvature ψ-coordinate is the **raw curvature κ itself**, so
        // κ = 0 stays an interior point of the `S^d ← ℝ^d → H^d` family. The
        // bundle therefore carries `∂·/∂κ` / `∂²·/∂κ²` directly, and the chart
        // coordinates are consumed verbatim (no input standardization — the
        // gauge `1 + κ‖x‖²` defines what κ means; see the basis builder).
        SmoothBasisSpec::ConstantCurvature { feature_cols, spec } => {
            let x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            build_constant_curvature_basis_kappa_derivatives(x.view(), spec)
                .map_err(EstimationError::from)?
        }
        // Measure-jet routes through the GROUPED dial builder
        // (`try_build_spatial_term_log_kappa_aniso_derivativeinfos`):
        // `spatial_term_uses_per_axis_psi` is true for every enrolled
        // measure-jet term, so this isotropic path only sees unenrolled
        // terms (`measure_jet_enrolls_psi` = false), which expose no ψ bundle.
        SmoothBasisSpec::MeasureJet { .. } => return Ok(None),
        SmoothBasisSpec::Matern {
            feature_cols,
            spec,
            input_scales,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            let mut spec_local = spec.clone();
            if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
                spec_local.length_scale =
                    compensate_length_scale_for_standardization(spec.length_scale, s);
            }
            // The realized Matérn DESIGN penalty is ALWAYS the operator-collocation
            // {mass, tension, stiffness} triplet — the term-collection assembler
            // overrides whatever `double_penalty` produced at the basis level with
            // `matern_operator_penalty_triplet_from_metadata` (see
            // `gam_terms::smooth::term_specs`, "The Matérn design ALWAYS uses the
            // operator-collocation … triplet"; #1074/#1270). The ψ=log κ outer
            // gradient must differentiate the SAME penalty the REML cost is built
            // on, so the derivative is forced onto the operator-triplet path here.
            // Honoring `double_penalty: true` instead returned the kernel-Gram
            // double-penalty ψ-derivatives — a penalty the design does NOT carry —
            // which desynced the analytic iso-κ gradient from the cost's FD and
            // stalled the κ-optimizer at its iteration cap with a large residual
            // gradient (#1122). `double_penalty: false` reproduces the operator
            // triplet exactly (verified: the 2-D iso-κ FD matches to ~1e-9).
            spec_local.double_penalty = false;
            build_matern_basis_log_kappa_derivatives(x.view(), &spec_local)
                .map_err(EstimationError::from)?
        }
        SmoothBasisSpec::Duchon {
            feature_cols,
            spec,
            input_scales,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            let mut spec_local = spec.clone();
            if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
                spec_local.length_scale =
                    compensate_optional_length_scale_for_standardization(spec.length_scale, s);
            }
            let BasisMetadata::Duchon {
                centers,
                identifiability_transform,
                operator_collocation_points,
                radial_reparam,
                ..
            } = &smooth_term.metadata
            else {
                return Ok(None);
            };
            // #1355: replay the frozen data-metric reparam into the derivative
            // spec so the ψ-derivative arms assemble in the rotated radial basis.
            if spec_local.radial_reparam.is_none() {
                spec_local.radial_reparam = radial_reparam.clone();
            }
            gam_terms::basis::build_duchon_basis_log_kappa_derivativeswith_collocationwithworkspace(
                x.view(),
                &spec_local,
                centers.view(),
                identifiability_transform.as_ref(),
                operator_collocation_points
                    .as_ref()
                    .map(|points| points.view()),
                &mut BasisWorkspace::default(),
            )
            .map_err(EstimationError::from)?
        }
        SmoothBasisSpec::BSpline1D { .. }
        | SmoothBasisSpec::TensorBSpline { .. }
        | SmoothBasisSpec::ByVariable { .. }
        | SmoothBasisSpec::FactorSumToZero { .. }
        | SmoothBasisSpec::BySmooth { .. }
        | SmoothBasisSpec::FactorSmooth { .. }
        | SmoothBasisSpec::Pca { .. } => {
            return Ok(None);
        }
    };
    let mut implicit_operator = derivative_bundle.implicit_operator;
    let BasisPsiDerivativeResult {
        design_derivative: mut local_x_psi,
        penalties_derivative: mut local_s_psi,
        implicit_operator: local_implicit_first_unused,
    } = derivative_bundle.first;
    let BasisPsiSecondDerivativeResult {
        designsecond_derivative: mut local_x_psi_psi,
        penaltiessecond_derivative: mut local_s_psi_psi,
        implicit_operator: local_implicit_second_unused,
    } = derivative_bundle.second;
    assert!(local_implicit_first_unused.is_none());
    assert!(local_implicit_second_unused.is_none());

    if let Some(rotation) = smooth_term.joint_null_rotation.as_ref() {
        let q = &rotation.rotation;
        if let Some(op) = implicit_operator.take() {
            implicit_operator = Some(op.append_full_transform(q).map_err(EstimationError::from)?);
        } else {
            if local_x_psi.ncols() != q.nrows() || local_x_psi_psi.ncols() != q.nrows() {
                return Ok(None);
            }
            local_x_psi = fast_ab(&local_x_psi, q);
            local_x_psi_psi = fast_ab(&local_x_psi_psi, q);
        }
        let rotate_penalty = |s_local: Array2<f64>| -> Option<Array2<f64>> {
            if s_local.nrows() != q.nrows() || s_local.ncols() != q.nrows() {
                return None;
            }
            let qt_s = gam_linalg::faer_ndarray::fast_atb(q, &s_local);
            Some(gam_linalg::faer_ndarray::fast_ab(&qt_s, q))
        };
        let Some(rotated_s_psi) = local_s_psi
            .into_iter()
            .map(|s| rotate_penalty(s))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        local_s_psi = rotated_s_psi;
        let Some(rotated_s_psi_psi) = local_s_psi_psi
            .into_iter()
            .map(|s| rotate_penalty(s))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        local_s_psi_psi = rotated_s_psi_psi;
    }
    let implicit_operator = implicit_operator.map(std::sync::Arc::new);

    if let Some(ref op) = implicit_operator {
        if op.p_out() != smooth_term.coeff_range.len() {
            return Ok(None);
        }
    } else {
        if local_x_psi.ncols() != smooth_term.coeff_range.len() {
            return Ok(None);
        }
        if local_x_psi_psi.ncols() != smooth_term.coeff_range.len() {
            return Ok(None);
        }
    }
    if local_s_psi.is_empty() || local_s_psi.len() != local_s_psi_psi.len() {
        return Ok(None);
    }
    if local_s_psi.iter().any(|s| {
        s.nrows() != smooth_term.coeff_range.len() || s.ncols() != smooth_term.coeff_range.len()
    }) {
        return Ok(None);
    }
    if local_s_psi_psi.iter().any(|s| {
        s.nrows() != smooth_term.coeff_range.len() || s.ncols() != smooth_term.coeff_range.len()
    }) {
        return Ok(None);
    }

    let p_total = design.design.ncols();
    let smooth_start = p_total.saturating_sub(design.smooth.total_smooth_cols());
    let global_range = (smooth_start + smooth_term.coeff_range.start)
        ..(smooth_start + smooth_term.coeff_range.end);

    Ok(Some((
        global_range,
        p_total,
        local_x_psi,
        local_s_psi.iter().fold(
            Array2::<f64>::zeros((smooth_term.coeff_range.len(), smooth_term.coeff_range.len())),
            |acc, m| acc + m,
        ),
        local_x_psi_psi,
        local_s_psi_psi.iter().fold(
            Array2::<f64>::zeros((smooth_term.coeff_range.len(), smooth_term.coeff_range.len())),
            |acc, m| acc + m,
        ),
        local_s_psi,
        local_s_psi_psi,
        implicit_operator,
    )))
}

fn try_build_spatial_log_kappa_hyper_dirs(
    data: ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    spatial_terms: &[usize],
) -> Result<Option<Vec<DirectionalHyperParam>>, EstimationError> {
    // Each spatial term contributes one continuous scale hyperparameter
    //   psi = log(kappa) = -log(length_scale),
    // while rho = log(lambda) still indexes the smoothing parameters of the
    // three operator penalties. The joint outer vector is therefore
    //   theta = (rho_0, ..., rho_{K-1}, psi_1, ..., psi_q)
    // for q spatial terms participating in exact joint optimization.
    let Some(info_list) =
        try_build_spatial_log_kappa_derivativeinfo_list(data, resolvedspec, design, spatial_terms)?
    else {
        return Ok(None);
    };
    Ok(Some(spatial_log_kappa_hyper_dirs_frominfo_list(info_list)?))
}

pub(crate) fn try_build_latent_coord_hyper_dirs(
    latent: std::sync::Arc<gam_terms::latent::LatentCoordValues>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    latent_terms: &[gam_problem::types::SmoothTermIdx],
    analytic_rho_count: usize,
) -> Result<Option<Vec<DirectionalHyperParam>>, EstimationError> {
    if latent_terms.is_empty() || latent.is_empty() {
        return Ok(None);
    }
    if latent_terms.len() != 1 {
        crate::bail_invalid_estim!(
            "LatentCoord standard-fit hyper_dirs currently require exactly one latent smooth term"
                .to_string(),
        );
    }
    let term_idx = latent_terms[0];
    let smooth_term = design.smooth.terms.get(term_idx.get()).ok_or_else(|| {
        EstimationError::InvalidInput(format!(
            "LatentCoord term index {term_idx} out of bounds for realized smooth design"
        ))
    })?;
    let termspec = resolvedspec
        .smooth_terms
        .get(term_idx.get())
        .ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "LatentCoord term index {term_idx} out of bounds for resolved smooth spec"
            ))
        })?;
    let p_total = design.design.ncols();
    let smooth_start = p_total.saturating_sub(design.smooth.total_smooth_cols());
    let global_range = (smooth_start + smooth_term.coeff_range.start)
        ..(smooth_start + smooth_term.coeff_range.end);

    // Spline bases do not add a separate continuous basis-scale ψ coordinate
    // here. When they are latent-coordinate terms, their ψ directions are the
    // latent-coordinate axes below, using the same DirectionalHyperParam layout
    // as Matérn and Duchon.
    let operator = match (&termspec.basis, &smooth_term.metadata) {
        (
            SmoothBasisSpec::Matern { .. },
            BasisMetadata::Matern {
                centers,
                length_scale,
                nu,
                include_intercept,
                identifiability_transform,
                ..
            },
        ) => gam_terms::basis::LatentCoordDesignDerivative::new_matern(
            latent.clone(),
            std::sync::Arc::new(centers.clone()),
            *length_scale,
            *nu,
            *include_intercept,
            identifiability_transform.clone(),
        )
        .map_err(EstimationError::from)?,
        (
            SmoothBasisSpec::Duchon { .. },
            BasisMetadata::Duchon {
                centers,
                length_scale,
                power,
                nullspace_order,
                identifiability_transform,
                ..
            },
        ) => gam_terms::basis::LatentCoordDesignDerivative::new_duchon(
            latent.clone(),
            std::sync::Arc::new(centers.clone()),
            *length_scale,
            *power,
            *nullspace_order,
            identifiability_transform.clone(),
        )
        .map_err(EstimationError::from)?,
        (
            SmoothBasisSpec::Sphere { .. },
            BasisMetadata::Sphere {
                centers,
                penalty_order,
                method,
                constraint_transform,
                ..
            },
        ) if matches!(*method, gam_terms::basis::SphereMethod::Wahba) => {
            gam_terms::basis::LatentCoordDesignDerivative::new_sphere(
                latent.clone(),
                std::sync::Arc::new(centers.clone()),
                *penalty_order,
                constraint_transform.clone(),
            )
            .map_err(EstimationError::from)?
        }
        (
            SmoothBasisSpec::BSpline1D { spec, .. },
            BasisMetadata::BSpline1D {
                knots,
                identifiability_transform,
                periodic,
                degree: meta_degree,
                ..
            },
        ) => {
            // Issue #340: use the metadata-recorded effective degree so the
            // latent-design Jacobian matches what `build_bspline_basis_1d`
            // actually built at fit time after auto-shrink.
            let effective_degree = meta_degree.unwrap_or(spec.degree);
            if let Some((domain_start, period, num_basis)) = periodic {
                gam_terms::basis::LatentCoordDesignDerivative::new_periodic_bspline(
                    latent.clone(),
                    (*domain_start, *domain_start + *period),
                    effective_degree,
                    *num_basis,
                    identifiability_transform.clone(),
                )
                .map_err(EstimationError::from)?
            } else {
                gam_terms::basis::LatentCoordDesignDerivative::new_tensor_bspline(
                    latent.clone(),
                    vec![knots.clone()],
                    vec![effective_degree],
                    identifiability_transform.clone(),
                )
                .map_err(EstimationError::from)?
            }
        }
        (
            SmoothBasisSpec::TensorBSpline { .. },
            BasisMetadata::TensorBSpline {
                knots,
                degrees,
                identifiability_transform,
                ..
            },
        ) => gam_terms::basis::LatentCoordDesignDerivative::new_tensor_bspline(
            latent.clone(),
            knots.clone(),
            degrees.clone(),
            identifiability_transform.clone(),
        )
        .map_err(EstimationError::from)?,
        (SmoothBasisSpec::Pca { .. }, BasisMetadata::Pca { basis_matrix, .. }) => {
            gam_terms::basis::LatentCoordDesignDerivative::new_pca(
                latent.clone(),
                std::sync::Arc::new(basis_matrix.clone()),
            )
            .map_err(EstimationError::from)?
        }
        _ => return Ok(None),
    };
    if operator.p_out() != global_range.len() {
        crate::bail_invalid_estim!(
            "LatentCoord derivative width mismatch for term '{}': operator p={}, coeff range={}",
            smooth_term.name,
            operator.p_out(),
            global_range.len()
        );
    }
    let operator = std::sync::Arc::new(operator);
    let mut hyper_dirs = Vec::with_capacity(operator.n_axes());
    for flat_axis in 0..operator.n_axes() {
        let dir = DirectionalHyperParam::new_compact(
            gam_solve::estimate::reml::HyperDesignDerivative::from_latent_coord(
                operator.clone(),
                flat_axis,
                global_range.clone(),
                p_total,
            ),
            Vec::new(),
            None,
            None,
        )?
        .not_penalty_like();
        hyper_dirs.push(dir);
    }
    let direct_dim = latent_coord_direct_hyper_count(latent.id_mode(), latent.latent_dim());
    if analytic_rho_count + direct_dim > 0 {
        let zero_x = gam_solve::estimate::reml::HyperDesignDerivative::from(Array2::<f64>::zeros((
            design.design.nrows(),
            p_total,
        )));
        for _ in 0..analytic_rho_count {
            hyper_dirs.push(
                DirectionalHyperParam::new_compact(zero_x.clone(), Vec::new(), None, None)?
                    .not_penalty_like(),
            );
        }
        for _ in 0..direct_dim {
            hyper_dirs.push(
                DirectionalHyperParam::new_compact(zero_x.clone(), Vec::new(), None, None)?
                    .not_penalty_like(),
            );
        }
    }
    Ok(Some(hyper_dirs))
}

fn latent_coord_direct_hyper_count(
    id_mode: &gam_terms::latent::LatentIdMode,
    latent_dim: usize,
) -> usize {
    use gam_terms::latent::{AuxPriorStrength, LatentIdMode};
    match id_mode {
        LatentIdMode::AuxPrior { strength, .. } => match strength {
            AuxPriorStrength::Auto => 1,
            AuxPriorStrength::Fixed(_) => 0,
        },
        LatentIdMode::AuxPriorDimSelection { strength, .. } => {
            latent_dim
                + match strength {
                    AuxPriorStrength::Auto => 1,
                    AuxPriorStrength::Fixed(_) => 0,
                }
        }
        LatentIdMode::DimSelection { .. } => latent_dim,
        // A fixed-reference anchor carries at most the REML-selectable log-`μ`
        // (one direct hyper when `Auto`, none when `Fixed`), like `AuxPrior`.
        LatentIdMode::IsometryToReference { strength, .. } => match strength {
            AuxPriorStrength::Auto => 1,
            AuxPriorStrength::Fixed(_) => 0,
        },
        // The behavioral head appends one (1 + d) coefficient block per
        // η-channel, plus the composed per-axis ARD log-precisions.
        LatentIdMode::AuxOutcome { head, .. } => head.n_coeffs(latent_dim) + latent_dim,
        LatentIdMode::None => 0,
    }
}

fn latent_coord_initial_direct_hypers(
    id_mode: &gam_terms::latent::LatentIdMode,
    latent_dim: usize,
) -> Result<Array1<f64>, EstimationError> {
    use gam_terms::latent::{AuxPriorStrength, LatentIdMode};
    let mut values = Vec::with_capacity(latent_coord_direct_hyper_count(id_mode, latent_dim));
    match id_mode {
        LatentIdMode::AuxPrior { strength, .. } => {
            if matches!(strength, AuxPriorStrength::Auto) {
                values.push(0.0);
            }
        }
        LatentIdMode::AuxPriorDimSelection {
            strength,
            init_log_precision,
            ..
        } => {
            if matches!(strength, AuxPriorStrength::Auto) {
                values.push(0.0);
            }
            append_latent_ard_seed(&mut values, init_log_precision.as_ref(), latent_dim)?;
        }
        LatentIdMode::DimSelection { init_log_precision } => {
            append_latent_ard_seed(&mut values, init_log_precision.as_ref(), latent_dim)?;
        }
        LatentIdMode::IsometryToReference { strength, .. } => {
            if matches!(strength, AuxPriorStrength::Auto) {
                values.push(0.0);
            }
        }
        LatentIdMode::AuxOutcome {
            head,
            init_log_precision,
        } => {
            // Head coefficients seed at zero: intercept 0 ⇒ baseline rate, all
            // loadings 0 ⇒ no behavioral anchoring at start (REML/Newton move
            // them). One (1 + d) block per η-channel.
            values.extend(std::iter::repeat_n(0.0, head.n_coeffs(latent_dim)));
            append_latent_ard_seed(&mut values, init_log_precision.as_ref(), latent_dim)?;
        }
        LatentIdMode::None => {}
    }
    Ok(Array1::from_vec(values))
}

fn append_latent_ard_seed(
    values: &mut Vec<f64>,
    init: Option<&Array1<f64>>,
    latent_dim: usize,
) -> Result<(), EstimationError> {
    if let Some(init) = init {
        if init.len() != latent_dim {
            crate::bail_invalid_estim!(
                "latent dim_selection init_log_precision length mismatch: got {}, expected {}",
                init.len(),
                latent_dim
            );
        }
        values.extend(init.iter().copied());
    } else {
        values.extend(std::iter::repeat_n(0.0, latent_dim));
    }
    Ok(())
}

struct LatentIdObjectiveContribution {
    cost: f64,
    gradient: Array1<f64>,
}

fn latent_id_objective_contribution(
    theta: &Array1<f64>,
    rho_dim: usize,
    analytic_rho_count: usize,
    latent: &gam_terms::latent::LatentCoordValues,
) -> Result<LatentIdObjectiveContribution, EstimationError> {
    use gam_terms::latent::{AuxPriorStrength, LatentIdMode, aux_prior_targets};
    let n_obs = latent.n_obs();
    let latent_dim = latent.latent_dim();
    let flat_len = latent.len();
    let mut gradient = Array1::<f64>::zeros(theta.len());
    let t_start = rho_dim;
    let direct_start = t_start + flat_len + analytic_rho_count;
    if theta.len() < direct_start {
        crate::bail_invalid_estim!(
            "latent-coordinate theta too short for id objective: got {}, need at least {}",
            theta.len(),
            direct_start
        );
    }
    let t = latent.as_matrix();
    let mut cost = 0.0;
    let mut cursor = direct_start;

    match latent.id_mode() {
        LatentIdMode::AuxPrior {
            u,
            family,
            strength,
        }
        | LatentIdMode::AuxPriorDimSelection {
            u,
            family,
            strength,
            ..
        } => {
            let (log_mu, mu) = match strength {
                AuxPriorStrength::Fixed(mu) => (mu.ln(), *mu),
                AuxPriorStrength::Auto => {
                    let log_mu = theta[cursor];
                    cursor += 1;
                    (log_mu, log_mu.exp())
                }
            };
            let targets = aux_prior_targets(t.view(), u.view(), *family)
                .map_err(EstimationError::InvalidInput)?;
            let residual = &t - &targets;
            let q = residual.iter().map(|v| v * v).sum::<f64>();
            // The single shared precision `mu` governs every one of the
            // `n_obs · latent_dim` scalar latent coordinates, so the prior
            // log-determinant normalizer `−0.5·log det₊(mu · I_K)` counts
            // `K = n_obs · latent_dim`. (The per-axis ARD path below emits
            // `−0.5·n_obs·ln(α)` for each of `latent_dim` axes; one shared `mu`
            // must equal that sum.)
            let k = (n_obs * latent_dim) as f64;
            cost += 0.5 * mu * q - 0.5 * k * log_mu;

            let projected_residual = aux_prior_targets(residual.view(), u.view(), *family)
                .map_err(EstimationError::InvalidInput)?;
            let grad_base = residual - projected_residual;
            for n in 0..n_obs {
                for axis in 0..latent_dim {
                    gradient[t_start + n * latent_dim + axis] += mu * grad_base[[n, axis]];
                }
            }
            if matches!(strength, AuxPriorStrength::Auto) {
                gradient[direct_start] += 0.5 * mu * q - 0.5 * k;
            }
        }
        LatentIdMode::IsometryToReference { reference, strength } => {
            // Fixed-reference anchor `½ μ ‖t − reference‖²` with REML-selectable
            // `μ`. Identical structure to `AuxPrior` except the target is a
            // constant configuration (independent of `t`), so the latent
            // gradient is the plain `μ · (t − reference)` with no projection
            // term (`AuxPrior` subtracts the projected residual only because its
            // target `ĥ(u)` depends on `t` through the internal ridge fit).
            if reference.dim() != (n_obs, latent_dim) {
                crate::bail_invalid_estim!(
                    "IsometryToReference reference shape {:?} must equal (n_obs, latent_dim) = ({}, {})",
                    reference.dim(),
                    n_obs,
                    latent_dim
                );
            }
            let mu_slot = cursor;
            let (log_mu, mu) = match strength {
                AuxPriorStrength::Fixed(mu) => (mu.ln(), *mu),
                AuxPriorStrength::Auto => {
                    let log_mu = theta[cursor];
                    cursor += 1;
                    (log_mu, log_mu.exp())
                }
            };
            let residual = &t - reference;
            let q = residual.iter().map(|v| v * v).sum::<f64>();
            // Shared precision `mu` over all `K = n_obs · latent_dim` scalar
            // coordinates: the normalizer `−0.5·log det₊(mu · I_K)` counts `K`,
            // matching the AuxPrior arm and the ARD path's per-axis sum.
            let k = (n_obs * latent_dim) as f64;
            cost += 0.5 * mu * q - 0.5 * k * log_mu;
            for n in 0..n_obs {
                for axis in 0..latent_dim {
                    gradient[t_start + n * latent_dim + axis] += mu * residual[[n, axis]];
                }
            }
            if matches!(strength, AuxPriorStrength::Auto) {
                gradient[mu_slot] += 0.5 * mu * q - 0.5 * k;
            }
        }
        LatentIdMode::AuxOutcome { head, .. } => {
            // Behavioral head likelihood channel: the head's design columns are
            // the live latent codes, so its NLL enters the SAME joint objective
            // as the reconstruction term and REML balances the two channels.
            // The head coefficients occupy `head.n_coeffs(d)` direct-hyper slots
            // starting at `cursor`; their gradient drives the β-tier update and
            // the head's latent-code gradient flows into the `t` block (the
            // arrow-Schur cross-channel coupling).
            let n_coeffs = head.n_coeffs(latent_dim);
            let coeffs = theta
                .slice(ndarray::s![cursor..cursor + n_coeffs])
                .to_owned();
            let (head_nll, grad_coeffs, grad_t) = head
                .neg_loglik_and_grad(t.view(), coeffs.view())
                .map_err(EstimationError::InvalidInput)?;
            cost += head_nll;
            for (offset, &g) in grad_coeffs.iter().enumerate() {
                gradient[cursor + offset] += g;
            }
            for n in 0..n_obs {
                for axis in 0..latent_dim {
                    gradient[t_start + n * latent_dim + axis] += grad_t[[n, axis]];
                }
            }
            cursor += n_coeffs;
        }
        LatentIdMode::DimSelection { .. } | LatentIdMode::None => {}
    }

    match latent.id_mode() {
        LatentIdMode::AuxPriorDimSelection { .. }
        | LatentIdMode::DimSelection { .. }
        | LatentIdMode::AuxOutcome { .. } => {
            for axis in 0..latent_dim {
                let log_alpha = theta[cursor + axis];
                let alpha = log_alpha.exp();
                let mut q_axis = 0.0;
                for n in 0..n_obs {
                    let flat_idx = n * latent_dim + axis;
                    let value = latent.as_flat()[flat_idx];
                    q_axis += value * value;
                    gradient[t_start + flat_idx] += alpha * value;
                }
                cost += 0.5 * alpha * q_axis - 0.5 * n_obs as f64 * log_alpha;
                gradient[cursor + axis] += 0.5 * alpha * q_axis - 0.5 * n_obs as f64;
            }
            cursor += latent_dim;
        }
        LatentIdMode::AuxPrior { .. }
        | LatentIdMode::IsometryToReference { .. }
        | LatentIdMode::None => {}
    }

    if cursor != theta.len() {
        crate::bail_invalid_estim!(
            "latent-coordinate direct hyperparameter length mismatch: consumed {}, theta len {}",
            cursor,
            theta.len()
        );
    }
    Ok(LatentIdObjectiveContribution { cost, gradient })
}

fn add_latent_id_objective_to_eval(
    theta: &Array1<f64>,
    rho_dim: usize,
    analytic_rho_count: usize,
    latent: &gam_terms::latent::LatentCoordValues,
    eval: &mut (
        f64,
        Array1<f64>,
        gam_problem::HessianResult,
    ),
) -> Result<(), EstimationError> {
    let contribution =
        latent_id_objective_contribution(theta, rho_dim, analytic_rho_count, latent)?;
    eval.0 += contribution.cost;
    if eval.1.len() != contribution.gradient.len() {
        crate::bail_invalid_estim!(
            "latent-coordinate REML gradient length mismatch: base={}, id={}",
            eval.1.len(),
            contribution.gradient.len()
        );
    }
    eval.1 += &contribution.gradient;
    if eval.2.is_analytic() {
        eval.2 = gam_problem::HessianResult::Unavailable;
    }
    Ok(())
}

fn analytic_penalty_objective_contribution(
    theta: &Array1<f64>,
    rho_dim: usize,
    latent: &gam_terms::latent::LatentCoordValues,
    registry: &gam_terms::AnalyticPenaltyRegistry,
) -> Result<LatentIdObjectiveContribution, EstimationError> {
    let flat_len = latent.len();
    let t_start = rho_dim;
    let t_end = t_start + flat_len;
    let rho_start = t_end;
    let rho_end = rho_start + registry.total_rho_count();
    if theta.len() < rho_end {
        crate::bail_invalid_estim!(
            "latent-coordinate theta too short for analytic penalties: got {}, need at least {}",
            theta.len(),
            rho_end
        );
    }
    let target_t = theta.slice(s![t_start..t_end]);
    let rho = theta.slice(s![rho_start..rho_end]);
    let mut cost = 0.0_f64;
    let mut gradient = Array1::<f64>::zeros(theta.len());
    for (penalty, (rho_slice, tier, name)) in registry.penalties.iter().zip(registry.rho_layout()) {
        let rho_local = rho.slice(s![rho_slice.clone()]);
        match tier {
            gam_terms::PenaltyTier::Psi => {
                cost += penalty.value(target_t.view(), rho_local);
                let grad = penalty.grad_target(target_t.view(), rho_local);
                if grad.len() != flat_len {
                    crate::bail_invalid_estim!(
                        "analytic penalty {name:?} gradient length mismatch: got {}, expected {}",
                        grad.len(),
                        flat_len
                    );
                }
                for i in 0..flat_len {
                    gradient[t_start + i] += grad[i];
                }
                let grad_rho_local = penalty.grad_rho(target_t.view(), rho_local);
                if grad_rho_local.len() != rho_slice.len() {
                    crate::bail_invalid_estim!(
                        "analytic penalty {name:?} rho-gradient length mismatch: got {}, expected {}",
                        grad_rho_local.len(),
                        rho_slice.len()
                    );
                }
                for local_idx in 0..grad_rho_local.len() {
                    gradient[rho_start + rho_slice.start + local_idx] += grad_rho_local[local_idx];
                }
            }
            gam_terms::PenaltyTier::Beta => {}
            gam_terms::PenaltyTier::Rho => {}
        }
    }
    Ok(LatentIdObjectiveContribution { cost, gradient })
}

fn add_analytic_penalty_hessian_to_eval(
    theta: &Array1<f64>,
    rho_dim: usize,
    latent: &gam_terms::latent::LatentCoordValues,
    registry: &gam_terms::AnalyticPenaltyRegistry,
    eval: &mut (
        f64,
        Array1<f64>,
        gam_problem::HessianResult,
    ),
) -> Result<(), EstimationError> {
    let flat_len = latent.len();
    let t_start = rho_dim;
    let t_end = t_start + flat_len;
    let rho_start = t_end;
    let rho_end = rho_start + registry.total_rho_count();
    if theta.len() < rho_end {
        crate::bail_invalid_estim!(
            "latent-coordinate theta too short for analytic penalty Hessian: got {}, need at least {}",
            theta.len(),
            rho_end
        );
    }
    let gam_problem::HessianResult::Analytic(hessian) = &mut eval.2 else {
        if eval.2.is_analytic() {
            eval.2 = gam_problem::HessianResult::Unavailable;
        }
        return Ok(());
    };
    if hessian.dim() != (theta.len(), theta.len()) {
        crate::bail_invalid_estim!(
            "analytic penalty Hessian target shape mismatch: got {}x{}, expected {}x{}",
            hessian.nrows(),
            hessian.ncols(),
            theta.len(),
            theta.len()
        );
    }
    let target_t = theta.slice(s![t_start..t_end]);
    let rho = theta.slice(s![rho_start..rho_end]);
    for (penalty, (rho_slice, tier, _name)) in registry.penalties.iter().zip(registry.rho_layout())
    {
        let rho_local = rho.slice(s![rho_slice]);
        if !matches!(tier, gam_terms::PenaltyTier::Psi) {
            continue;
        }
        if let Some(diag) = penalty.hessian_diag(target_t.view(), rho_local) {
            if diag.len() != flat_len {
                crate::bail_invalid_estim!(
                    "analytic penalty Hessian diagonal length mismatch: got {}, expected {}",
                    diag.len(),
                    flat_len
                );
            }
            for i in 0..flat_len {
                hessian[[t_start + i, t_start + i]] += diag[i];
            }
            continue;
        }
        let mut probe = Array1::<f64>::zeros(flat_len);
        for col in 0..flat_len {
            probe[col] = 1.0;
            let hv = penalty.hvp(target_t.view(), rho_local, probe.view());
            if hv.len() != flat_len {
                crate::bail_invalid_estim!(
                    "analytic penalty Hessian-vector length mismatch: got {}, expected {}",
                    hv.len(),
                    flat_len
                );
            }
            for row in 0..flat_len {
                hessian[[t_start + row, t_start + col]] += hv[row];
            }
            probe[col] = 0.0;
        }
    }
    Ok(())
}

fn add_analytic_penalty_objective_to_eval(
    theta: &Array1<f64>,
    rho_dim: usize,
    latent: &gam_terms::latent::LatentCoordValues,
    registry: &gam_terms::AnalyticPenaltyRegistry,
    eval: &mut (
        f64,
        Array1<f64>,
        gam_problem::HessianResult,
    ),
) -> Result<(), EstimationError> {
    let contribution = analytic_penalty_objective_contribution(theta, rho_dim, latent, registry)?;
    eval.0 += contribution.cost;
    if eval.1.len() != contribution.gradient.len() {
        crate::bail_invalid_estim!(
            "latent-coordinate REML gradient length mismatch: base={}, analytic_penalty={}",
            eval.1.len(),
            contribution.gradient.len()
        );
    }
    eval.1 += &contribution.gradient;
    add_analytic_penalty_hessian_to_eval(theta, rho_dim, latent, registry, eval)?;
    Ok(())
}

fn spatial_log_kappa_hyper_dirs_frominfo_list(
    info_list: Vec<SpatialPsiDerivative>,
) -> Result<Vec<DirectionalHyperParam>, EstimationError> {
    use gam_solve::estimate::reml::ImplicitDerivLevel;
    use std::collections::HashMap;

    let log_kappa_dim = info_list.len();
    // Layout-only metadata (group_id per axis) is cheap to snapshot up front so
    // the consumption loop below can MOVE the dense (n × p) derivative arrays
    // out of each entry instead of cloning. At large scale (n≈3×10⁵, 16-axis
    // CTN) the prior `.clone()` sites doubled peak working memory for the
    // psi-derivative pass through several GiB.
    let group_ids: Vec<Option<usize>> = info_list.iter().map(|e| e.aniso_group_id).collect();
    let mut group_indices_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, gid) in group_ids.iter().enumerate() {
        if let Some(g) = gid {
            group_indices_map.entry(*g).or_default().push(idx);
        }
    }

    let mut hyper_dirs = Vec::with_capacity(log_kappa_dim);
    for (i, info) in info_list.into_iter().enumerate() {
        let SpatialPsiDerivative {
            penalty_index: _,
            penalty_indices,
            global_range,
            total_p,
            x_psi_local,
            s_psi_components_local,
            x_psi_psi_local,
            s_psi_psi_components_local,
            aniso_group_id,
            aniso_cross_designs,
            aniso_cross_penalty_provider,
            implicit_operator,
            implicit_axis,
        } = info;

        let mut xsecond = vec![None; log_kappa_dim];
        // Diagonal second derivative (same axis).
        xsecond[i] = Some(if let Some(ref op) = implicit_operator {
            gam_solve::estimate::reml::HyperDesignDerivative::from_implicit(
                op.clone(),
                ImplicitDerivLevel::SecondDiag(implicit_axis),
                global_range.clone(),
                total_p,
            )
        } else {
            gam_solve::estimate::reml::HyperDesignDerivative::from_embedded(
                x_psi_psi_local,
                global_range.clone(),
                total_p,
            )
        });
        // Cross second derivatives for axes in the same aniso group.
        if let Some(cross_designs) = aniso_cross_designs {
            // Use the base index of this aniso group in the original info_list.
            // Entries for the same group are contiguous: the first index in the
            // group gives the base, and axis b is at base+b.
            if let Some(gid) = aniso_group_id {
                let base = group_indices_map
                    .get(&gid)
                    .and_then(|v| v.first().copied())
                    .unwrap_or(i);
                for (b_axis, cross_mat) in cross_designs.into_iter() {
                    let j = base + b_axis;
                    if j < log_kappa_dim {
                        xsecond[j] = Some(if let Some(ref op) = implicit_operator {
                            gam_solve::estimate::reml::HyperDesignDerivative::from_implicit(
                                op.clone(),
                                ImplicitDerivLevel::SecondCross(implicit_axis, b_axis),
                                global_range.clone(),
                                total_p,
                            )
                        } else {
                            gam_solve::estimate::reml::HyperDesignDerivative::from_embedded(
                                cross_mat,
                                global_range.clone(),
                                total_p,
                            )
                        });
                    }
                }
            }
        }
        let s_components = penalty_indices
            .iter()
            .copied()
            .zip(s_psi_components_local.into_iter().map(|local| {
                gam_solve::estimate::reml::HyperPenaltyDerivative::from_embedded(
                    local,
                    global_range.clone(),
                    total_p,
                )
            }))
            .collect::<Vec<_>>();
        let s2_components = penalty_indices
            .iter()
            .copied()
            .zip(s_psi_psi_components_local.into_iter().map(|local| {
                gam_solve::estimate::reml::HyperPenaltyDerivative::from_embedded(
                    local,
                    global_range.clone(),
                    total_p,
                )
            }))
            .collect::<Vec<_>>();
        let mut ssecond_components = vec![None; log_kappa_dim];
        ssecond_components[i] = Some(s2_components);
        let mut penaltysecond_partner_indices: Option<Vec<usize>> = None;
        let penaltysecond_component_provider =
            if let (Some(provider), Some(gid)) = (aniso_cross_penalty_provider, aniso_group_id) {
                let group_indices = group_indices_map.get(&gid).cloned().unwrap_or_default();
                let axis_in_group =
                    group_indices
                        .iter()
                        .position(|&idx| idx == i)
                        .ok_or_else(|| {
                            EstimationError::InvalidInput(format!(
                                "missing spatial hyper axis {} in anisotropy group {}",
                                i, gid
                            ))
                        })?;
                penaltysecond_partner_indices = Some(
                    group_indices
                        .iter()
                        .copied()
                        .filter(|&idx| idx != i)
                        .collect(),
                );
                let penalty_indices_inner = penalty_indices.clone();
                let global_range_inner = global_range.clone();
                let total_p_inner = total_p;
                let group_indices_inner = group_indices;
                Some(std::sync::Arc::new(
                    move |j: usize| -> Result<
                        Option<Vec<gam_solve::estimate::reml::PenaltyDerivativeComponent>>,
                        EstimationError,
                    > {
                        let Some(other_axis_in_group) =
                            group_indices_inner.iter().position(|&idx| idx == j)
                        else {
                            return Ok(None);
                        };
                        if other_axis_in_group == axis_in_group {
                            return Ok(None);
                        }
                        let cross_pens = provider(other_axis_in_group)?;
                        if cross_pens.is_empty() {
                            return Ok(None);
                        }
                        Ok(Some(
                            penalty_indices_inner
                                .iter()
                                .copied()
                                .zip(cross_pens.into_iter().map(|local| {
                                    gam_solve::estimate::reml::HyperPenaltyDerivative::from_embedded(
                                        local,
                                        global_range_inner.clone(),
                                        total_p_inner,
                                    )
                                }))
                                .map(|(penalty_index, matrix)| {
                                    gam_solve::estimate::reml::PenaltyDerivativeComponent {
                                        penalty_index,
                                        matrix,
                                    }
                                })
                                .collect(),
                        ))
                    },
                )
                    as std::sync::Arc<
                        dyn Fn(
                                usize,
                            ) -> Result<
                                Option<Vec<gam_solve::estimate::reml::PenaltyDerivativeComponent>>,
                                EstimationError,
                            > + Send
                            + Sync
                            + 'static,
                    >)
            } else {
                None
            };
        // First derivative: use implicit operator when available to avoid
        // storing dense (n x p) matrices for all D axes simultaneously.
        let x_first_hyper = if let Some(ref op) = implicit_operator {
            gam_solve::estimate::reml::HyperDesignDerivative::from_implicit(
                op.clone(),
                ImplicitDerivLevel::First(implicit_axis),
                global_range.clone(),
                total_p,
            )
        } else {
            gam_solve::estimate::reml::HyperDesignDerivative::from_embedded(
                x_psi_local,
                global_range.clone(),
                total_p,
            )
        };
        let mut dir = DirectionalHyperParam::new_compact(
            x_first_hyper,
            s_components,
            Some(xsecond),
            Some(ssecond_components),
        )?
        .not_penalty_like();
        if let Some(provider) = penaltysecond_component_provider {
            dir = dir.with_penaltysecond_component_provider(provider);
        }
        if let Some(partner_indices) = penaltysecond_partner_indices {
            dir = dir.with_penaltysecond_partner_indices(partner_indices);
        }
        hyper_dirs.push(dir);
    }
    Ok(hyper_dirs)
}

/// Compute `dims_per_term` for a list of spatial term indices.
///
/// Returns a vector where entry i is the number of stored ψ values for
/// spatial term i: `d` for terms that enroll per-axis anisotropy in the
/// REML joint vector (`spatial_term_uses_per_axis_psi`), `1` otherwise.
pub(crate) fn spatial_dims_per_term(
    resolvedspec: &TermCollectionSpec,
    spatial_terms: &[usize],
) -> Vec<usize> {
    spatial_terms
        .iter()
        .map(|&term_idx| {
            if let Some(mj) = measure_jet_term_spec(resolvedspec, term_idx) {
                // Dial group, not per-axis anisotropy; layout owned by
                // `measure_jet_psi_dim`.
                measure_jet_psi_dim(mj)
            } else if spatial_term_uses_per_axis_psi(resolvedspec, term_idx) {
                get_spatial_feature_dim(resolvedspec, term_idx).unwrap_or(1)
            } else {
                1
            }
        })
        .collect()
}

/// Check whether any spatial terms enroll per-axis anisotropic ψ in the joint
/// outer vector. Mirrors the hyper_dirs builder's enrollment predicate so the
/// outer θ-layout cannot drift from the inner evaluator's ψ count.
fn has_aniso_terms(resolvedspec: &TermCollectionSpec, spatial_terms: &[usize]) -> bool {
    spatial_terms
        .iter()
        .any(|&term_idx| spatial_term_uses_per_axis_psi(resolvedspec, term_idx))
}

/// Emits the `theta`-keyed memoization accessors shared verbatim by the
/// single-block and n-block exact-joint design caches. Both carry the same
/// `current_theta` / `last_cost` / `last_eval` fields, so the cost/eval
/// lookups and the `store_eval` writer are identical; this macro is the single
/// source so the two inherent impls cannot drift.
macro_rules! impl_exact_joint_theta_memo {
    () => {
        fn memoized_cost(&self, theta: &Array1<f64>) -> Option<f64> {
            if self
                .current_theta
                .as_ref()
                .is_some_and(|cached| theta_values_match(cached, theta))
            {
                self.last_eval
                    .as_ref()
                    .map(|cached| cached.0)
                    .or(self.last_cost)
            } else {
                None
            }
        }

        fn memoized_eval(
            &self,
            theta: &Array1<f64>,
        ) -> Option<(
            f64,
            Array1<f64>,
            gam_problem::HessianResult,
        )> {
            if self
                .current_theta
                .as_ref()
                .is_some_and(|cached| theta_values_match(cached, theta))
            {
                self.last_eval.clone()
            } else {
                None
            }
        }

        fn store_eval(
            &mut self,
            eval: (
                f64,
                Array1<f64>,
                gam_problem::HessianResult,
            ),
        ) {
            self.last_cost = Some(eval.0);
            self.last_eval = Some(eval);
        }
    };
}

struct SingleBlockExactJointDesignCache<'d> {
    realizer: FrozenTermCollectionIncrementalRealizer<'d>,
    current_theta: Option<Array1<f64>>,
    // Memo key for `last_cost`/`last_eval`. Distinct from `current_theta` (which
    // tracks the θ the n×k design is REALIZED at): on the #1033 certified
    // Gaussian path `eval_full` evaluates a trial ψ WITHOUT re-realizing the
    // design (the tensor serves value+gradient n-free), so the eval θ and the
    // realized-design θ diverge. Keying the memo on a dedicated field keeps a
    // ψ-skip from ever mis-associating one ψ's cost/eval with another ψ's key.
    last_eval_theta: Option<Array1<f64>>,
    last_cost: Option<f64>,
    last_eval: Option<(
        f64,
        Array1<f64>,
        gam_problem::HessianResult,
    )>,
    // #1033: ψ-invariant hyper-direction slab cache. The κ hyper_dirs (the n×k
    // ∂X/∂ψ design-derivative slabs + their k×k penalty derivatives) are a pure
    // function of (data, frozen spec, REALIZED column layout) — they do NOT
    // depend on the trial ψ once the design is fixed. On the certified Gaussian
    // n-free path `eval_full` evaluates trial ψ WITHOUT re-realizing the design,
    // so the realized layout (and hence the hyper_dirs) is identical across an
    // entire run of skip-path trials. Rebuilding them each trial re-runs the
    // basis ψ-derivative over all n rows + an O(n·k²) `fast_ab` rotation — the
    // last per-trial O(n) pass in the κ loop. Cache them keyed by the realizer
    // `design_revision`: a skip-path trial (revision unchanged) reuses the
    // build; a slow-path trial (revision advanced) rebuilds and re-keys.
    cached_hyper_dirs: Option<(u64, Vec<DirectionalHyperParam>)>,
    spatial_terms: Vec<usize>,
    rho_dim: usize,
    dims_per_term: Vec<usize>,
}

impl<'d> SingleBlockExactJointDesignCache<'d> {
    fn new(
        data: ArrayView2<'d, f64>,
        spec: TermCollectionSpec,
        design: TermCollectionDesign,
        spatial_terms: Vec<usize>,
        rho_dim: usize,
        dims_per_term: Vec<usize>,
    ) -> Result<Self, String> {
        Ok(Self {
            realizer: FrozenTermCollectionIncrementalRealizer::new(data, spec, design)?,
            current_theta: None,
            last_eval_theta: None,
            last_cost: None,
            last_eval: None,
            cached_hyper_dirs: None,
            spatial_terms,
            rho_dim,
            dims_per_term,
        })
    }

    fn design_revision(&self) -> u64 {
        self.realizer.design_revision()
    }

    /// Build the κ hyper-directions for the CURRENT realized design, reusing the
    /// `cached_hyper_dirs` slab when the realizer revision has not advanced since
    /// the last build (#1033). The slab is ψ-invariant at a fixed realized
    /// layout, so a skip-path trial (which does not re-realize the design) gets a
    /// bit-identical clone instead of re-running the per-row basis ψ-derivative +
    /// O(n·k²) rotation. A revision change (slow-path re-realization) rebuilds and
    /// re-keys. The clone is an O(n·k) memcpy — far cheaper than the O(n·k²)
    /// rebuild, and the conditioning pass it feeds is itself skipped on the
    /// certified path (see `prepare_eval_state`'s fast path).
    fn hyper_dirs_for_current_design(
        &mut self,
        data: ArrayView2<'_, f64>,
        kind: SpatialHyperKind,
    ) -> Result<Vec<DirectionalHyperParam>, EstimationError> {
        let revision = self.realizer.design_revision();
        if let Some((cached_rev, dirs)) = self.cached_hyper_dirs.as_ref()
            && *cached_rev == revision
        {
            return Ok(dirs.clone());
        }
        let dirs = try_build_spatial_log_kappa_hyper_dirs(
            data,
            self.realizer.spec(),
            self.realizer.design(),
            &self.spatial_terms,
        )?
        .ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "failed to build {} hyper_dirs at current {}",
                kind.adjective(),
                kind.coord_name(),
            ))
        })?;
        self.cached_hyper_dirs = Some((revision, dirs.clone()));
        Ok(dirs)
    }

    fn nfree_tensor_gradient_hyper_dirs(
        &mut self,
        theta: &Array1<f64>,
    ) -> Result<Vec<DirectionalHyperParam>, EstimationError> {
        let psi = &theta.as_slice().ok_or_else(|| {
            EstimationError::InvalidInput(
                "nfree_tensor_gradient_hyper_dirs: theta is not contiguous".to_string(),
            )
        })?[self.rho_dim..];
        let (global_range, p_total, s_psi_components) = self
            .realizer
            .canonical_penalty_derivatives_at_psi(&self.spatial_terms, psi)
            .map_err(EstimationError::InvalidInput)?;
        let zero_x = gam_solve::estimate::reml::HyperDesignDerivative::zero(
            self.realizer.design().design.nrows(),
            p_total,
        );
        let components = s_psi_components
            .into_iter()
            .enumerate()
            .map(|(penalty_index, local)| {
                (
                    penalty_index,
                    gam_solve::estimate::reml::HyperPenaltyDerivative::from_embedded(
                        local,
                        global_range.clone(),
                        p_total,
                    ),
                )
            })
            .collect::<Vec<_>>();
        Ok(DirectionalHyperParam::new_compact(zero_x, components, None, None)?.not_penalty_like())
            .map(|dir| vec![dir])
    }

    fn ensure_theta(&mut self, theta: &Array1<f64>) -> Result<(), String> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            return Ok(());
        }
        let t_ensure = std::time::Instant::now();
        let log_kappa = SpatialLogKappaCoords::from_theta_tail_with_dims(
            theta,
            self.rho_dim,
            self.dims_per_term.clone(),
        );
        self.realizer
            .apply_log_kappa(&log_kappa, &self.spatial_terms)?;
        log::info!(
            "[STAGE] ensure_theta (apply_log_kappa, {} terms): {:.3}s",
            self.spatial_terms.len(),
            t_ensure.elapsed().as_secs_f64(),
        );
        self.current_theta = Some(theta.clone());
        self.last_eval_theta = None;
        self.last_cost = None;
        self.last_eval = None;
        Ok(())
    }

    // Memo methods keyed on `last_eval_theta` (NOT `current_theta`): the #1033
    // certified Gaussian path evaluates a trial ψ without re-realizing the
    // design, so the eval θ and the realized-design θ can differ. Keying the
    // memo on the eval θ keeps a ψ-skip from mis-associating one ψ's result
    // with another ψ's key. The other exact-joint caches still use the shared
    // `impl_exact_joint_theta_memo!` macro (they always realize before eval).
    fn memoized_cost(&self, theta: &Array1<f64>) -> Option<f64> {
        if self
            .last_eval_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            self.last_eval
                .as_ref()
                .map(|cached| cached.0)
                .or(self.last_cost)
        } else {
            None
        }
    }

    fn memoized_eval(
        &self,
        theta: &Array1<f64>,
    ) -> Option<(
        f64,
        Array1<f64>,
        gam_problem::HessianResult,
    )> {
        if self
            .last_eval_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            self.last_eval.clone()
        } else {
            None
        }
    }

    /// Record an eval result keyed to the θ it was computed at. Used in place of
    /// the macro's `store_eval` so the memo key reflects the EVAL θ even when the
    /// design was not re-realized at that θ (#1033 certified skip).
    fn store_eval_at(
        &mut self,
        theta: &Array1<f64>,
        eval: (
            f64,
            Array1<f64>,
            gam_problem::HessianResult,
        ),
    ) {
        self.last_eval_theta = Some(theta.clone());
        self.last_cost = Some(eval.0);
        self.last_eval = Some(eval);
    }

    /// Record a cost-only result keyed to the θ it was computed at, so
    /// `memoized_cost` keys on the EVAL θ (matching `store_eval_at`).
    fn store_cost_at(&mut self, theta: &Array1<f64>, cost: f64) {
        self.last_eval_theta = Some(theta.clone());
        self.last_cost = Some(cost);
        // A cost-only probe carries no gradient/Hessian, so drop any prior
        // full eval: `memoized_cost` prefers `last_eval.0`, and a stale
        // `last_eval` from a different θ must never answer for this θ.
        self.last_eval = None;
    }

    fn spec(&self) -> &TermCollectionSpec {
        self.realizer.spec()
    }

    fn design(&self) -> &TermCollectionDesign {
        self.realizer.design()
    }

    /// True when the single spatial term's frozen geometry admits an EXACT,
    /// n-free penalty re-key at a new length-scale (#1033). The κ-loop fast path
    /// gates its design-realization skip on this (replacing the old certified
    /// `psi_penalty_tensor_covers` gate): the skip leaves `reset_surface`
    /// un-run, so it is sound only when `S(ψ_new)` can be rebuilt n-free.
    fn supports_nfree_penalty_rekey(&self) -> bool {
        self.realizer
            .supports_nfree_penalty_rekey(&self.spatial_terms)
    }

    fn supports_nfree_gradient_only_routing(&self) -> bool {
        self.realizer
            .supports_nfree_gradient_only_routing(&self.spatial_terms)
    }

    /// Build the EXACT canonical penalty surface `S(ψ)` at the length-scale
    /// implied by `theta`'s ψ tail, entirely n-free (#1033). Maps ψ→length-scale
    /// with the IDENTICAL `spatial_term_psi_to_length_scale_and_aniso` the slow
    /// path uses, reuses the frozen basis geometry, and runs the SAME
    /// `canonicalize_penalty_specs` pipeline `reset_surface` runs — so the
    /// returned canonical list is the one the kept reference surface must be
    /// re-keyed with on the design-revision fast path. The caller (which holds
    /// `cache`) computes this and hands the owned result to the evaluator via
    /// `stage_fast_path_penalty`, avoiding a `&mut cache` borrow alias.
    fn canonical_penalties_at(
        &mut self,
        theta: &Array1<f64>,
    ) -> Result<(Vec<gam_terms::construction::CanonicalPenalty>, Vec<usize>), String> {
        let psi = &theta
            .as_slice()
            .ok_or_else(|| "canonical_penalties_at: theta is not contiguous".to_string())?
            [self.rho_dim..];
        self.realizer
            .canonical_penalties_at_psi(&self.spatial_terms, psi)
    }
}

struct SingleBlockLatentCoordDesignCache {
    data: Array2<f64>,
    spec: TermCollectionSpec,
    design: TermCollectionDesign,
    current_theta: Option<Array1<f64>>,
    current_latent: Option<std::sync::Arc<gam_terms::latent::LatentCoordValues>>,
    current_hyper_dirs: Option<Vec<gam_solve::estimate::reml::DirectionalHyperParam>>,
    current_design_cache_id: Option<u64>,
    latent_design_cache: gam_solve::latent_cache::LatentDesignCache,
    last_cost: Option<f64>,
    last_eval: Option<(
        f64,
        Array1<f64>,
        gam_problem::HessianResult,
    )>,
    term_index: gam_problem::types::SmoothTermIdx,
    feature_cols: Vec<usize>,
    rho_dim: usize,
    n_obs: usize,
    latent_dim: usize,
    id_mode: gam_terms::latent::LatentIdMode,
    manifold: gam_terms::latent::LatentManifold,
    retraction_registry: gam_solve::latent_cache::LatentRetractionRegistry,
    latent_id: u64,
    analytic_penalties: Option<std::sync::Arc<gam_terms::AnalyticPenaltyRegistry>>,
    analytic_rho_count: usize,
    design_revision: u64,
    // Stamp the outer-iter the cached cost/eval was computed under; analytic
    // penalty weight schedules advance with this counter, so a stale stamp
    // invalidates the memo even at unchanged θ.
    last_outer_iter: Option<u64>,
}

impl SingleBlockLatentCoordDesignCache {
    fn new(
        data: Array2<f64>,
        spec: TermCollectionSpec,
        design: TermCollectionDesign,
        latent: &StandardLatentCoordConfig,
        rho_dim: usize,
    ) -> Result<Self, String> {
        if latent.term_index.get() >= spec.smooth_terms.len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "latent-coordinate term index {} out of bounds for {} smooth terms",
                latent.term_index,
                spec.smooth_terms.len()
            ))
            .into());
        }
        if latent.feature_cols.len() != latent.values.latent_dim() {
            return Err(SmoothError::dimension_mismatch(format!(
                "latent-coordinate feature width mismatch: feature_cols={}, latent_dim={}",
                latent.feature_cols.len(),
                latent.values.latent_dim()
            ))
            .into());
        }
        if latent.values.n_obs() != data.nrows() {
            return Err(SmoothError::dimension_mismatch(format!(
                "latent-coordinate row mismatch: latent n={}, data n={}",
                latent.values.n_obs(),
                data.nrows()
            ))
            .into());
        }
        let analytic_rho_count = latent
            .analytic_penalties
            .as_ref()
            .map_or(0, |registry| registry.total_rho_count());
        Ok(Self {
            data,
            spec,
            design,
            current_theta: None,
            current_latent: None,
            current_hyper_dirs: None,
            current_design_cache_id: None,
            latent_design_cache: gam_solve::latent_cache::LatentDesignCache::default(),
            last_cost: None,
            last_eval: None,
            term_index: latent.term_index,
            feature_cols: latent.feature_cols.clone(),
            rho_dim,
            n_obs: latent.values.n_obs(),
            latent_dim: latent.values.latent_dim(),
            id_mode: latent.values.id_mode().clone(),
            manifold: latent.values.manifold().clone(),
            retraction_registry: latent.values.retraction_registry().clone(),
            latent_id: latent.values.latent_id(),
            analytic_penalties: latent.analytic_penalties.clone(),
            analytic_rho_count,
            design_revision: 0,
            last_outer_iter: None,
        })
    }

    fn design_revision(&self) -> u64 {
        self.design_revision
    }

    fn design(&self) -> &TermCollectionDesign {
        &self.design
    }

    fn latent(&self) -> Result<std::sync::Arc<gam_terms::latent::LatentCoordValues>, String> {
        self.current_latent
            .as_ref()
            .cloned()
            .ok_or_else(|| "latent-coordinate cache has not been realized".to_string())
    }

    fn analytic_penalties(&self) -> Option<std::sync::Arc<gam_terms::AnalyticPenaltyRegistry>> {
        self.analytic_penalties.clone()
    }

    fn analytic_penalty_rho_count(&self) -> usize {
        self.analytic_rho_count
    }

    fn hyper_dirs(&self) -> Result<Vec<gam_solve::estimate::reml::DirectionalHyperParam>, String> {
        self.current_hyper_dirs
            .as_ref()
            .cloned()
            .ok_or_else(|| "latent-coordinate hyper_dirs cache has not been realized".to_string())
    }

    fn latent_basis_kind(&self) -> Result<gam_solve::latent_cache::LatentBasisKind, String> {
        let smooth_term = self
            .design
            .smooth
            .terms
            .get(self.term_index.get())
            .ok_or_else(|| {
                SmoothError::dimension_mismatch(format!(
                    "LatentCoord term index {} out of bounds for realized smooth design",
                    self.term_index
                ))
            })?;
        let termspec = self
            .spec
            .smooth_terms
            .get(self.term_index.get())
            .ok_or_else(|| {
                SmoothError::dimension_mismatch(format!(
                    "LatentCoord term index {} out of bounds for resolved smooth spec",
                    self.term_index
                ))
            })?;
        match (&termspec.basis, &smooth_term.metadata) {
            (
                SmoothBasisSpec::Matern { .. },
                BasisMetadata::Matern {
                    centers,
                    length_scale,
                    nu,
                    aniso_log_scales,
                    ..
                },
            ) => Ok(gam_solve::latent_cache::LatentBasisKind::Matern {
                centers: centers.clone(),
                length_scale: *length_scale,
                nu: *nu,
                aniso_log_scales: aniso_log_scales
                    .clone()
                    .unwrap_or_else(|| vec![0.0; centers.ncols()]),
                chunk_size: gam_terms::basis::auto_streaming_chunk_size_for_dense(
                    self.n_obs,
                    centers.nrows(),
                ),
            }),
            (
                SmoothBasisSpec::Duchon { .. },
                BasisMetadata::Duchon {
                    centers,
                    length_scale,
                    power,
                    nullspace_order,
                    aniso_log_scales,
                    ..
                },
            ) => Ok(gam_solve::latent_cache::LatentBasisKind::Duchon {
                centers: centers.clone(),
                length_scale: *length_scale,
                power: *power,
                nullspace_order: *nullspace_order,
                aniso_log_scales: aniso_log_scales
                    .clone()
                    .unwrap_or_else(|| vec![0.0; centers.ncols()]),
            }),
            (
                SmoothBasisSpec::Sphere { .. },
                BasisMetadata::Sphere {
                    centers,
                    penalty_order,
                    method,
                    ..
                },
            ) if matches!(*method, gam_terms::basis::SphereMethod::Wahba) => {
                Ok(gam_solve::latent_cache::LatentBasisKind::Sphere {
                    centers: centers.clone(),
                    penalty_order: *penalty_order,
                    chunk_size: gam_terms::basis::auto_streaming_chunk_size_for_dense(
                        self.n_obs,
                        centers.nrows(),
                    ),
                })
            }
            (
                SmoothBasisSpec::BSpline1D { spec, .. },
                BasisMetadata::BSpline1D {
                    knots,
                    periodic,
                    degree: meta_degree,
                    ..
                },
            ) => {
                // Issue #340: prefer the metadata-recorded effective degree
                // (which reflects fit-time auto-shrink) over the upstream
                // user-requested `spec.degree`.
                let effective_degree = meta_degree.unwrap_or(spec.degree);
                if let Some((domain_start, period, num_basis)) = periodic {
                    Ok(
                        gam_solve::latent_cache::LatentBasisKind::PeriodicBspline {
                            domain_start: *domain_start,
                            period: *period,
                            degree: effective_degree,
                            num_basis: *num_basis,
                            chunk_size: gam_terms::basis::auto_streaming_chunk_size_for_dense(
                                self.n_obs, *num_basis,
                            ),
                        },
                    )
                } else {
                    let num_basis_est = knots.len().saturating_sub(effective_degree + 1);
                    Ok(
                        gam_solve::latent_cache::LatentBasisKind::TensorBspline {
                            knots: vec![knots.clone()],
                            degrees: vec![effective_degree],
                            chunk_size: gam_terms::basis::auto_streaming_chunk_size_for_dense(
                                self.n_obs,
                                num_basis_est,
                            ),
                        },
                    )
                }
            }
            (
                SmoothBasisSpec::TensorBSpline { .. },
                BasisMetadata::TensorBSpline { knots, degrees, .. },
            ) => Ok(
                gam_solve::latent_cache::LatentBasisKind::TensorBspline {
                    knots: knots.clone(),
                    degrees: degrees.clone(),
                    chunk_size: None,
                },
            ),
            (
                SmoothBasisSpec::Pca { .. },
                BasisMetadata::Pca {
                    basis_matrix,
                    centered,
                    smooth_penalty,
                    center_mean,
                    pca_basis_path,
                    chunk_size,
                    ..
                },
            ) => {
                let center_mean_fingerprint = if *centered && pca_basis_path.is_none() {
                    let mean = center_mean.as_ref().ok_or_else(|| {
                        SmoothError::invalid_config(
                            "latent-coordinate Pca cache key requires center_mean when centered",
                        )
                    })?;
                    Some(gam_solve::latent_cache::pca_center_mean_fingerprint(
                        mean,
                    ))
                } else {
                    None
                };
                Ok(gam_solve::latent_cache::LatentBasisKind::Pca {
                    basis_matrix: basis_matrix.clone(),
                    centered: *centered,
                    center_mean_fingerprint,
                    smooth_penalty: *smooth_penalty,
                    pca_basis_path: pca_basis_path.clone(),
                    chunk_size: *chunk_size,
                })
            }
            _ => Err(SmoothError::invalid_config(
                "latent-coordinate design cache could not key the realized latent smooth basis"
                    .to_string(),
            )
            .into()),
        }
    }

    fn ensure_theta(&mut self, theta: &Array1<f64>) -> Result<(), String> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            return Ok(());
        }
        let latent_flat_len = self.n_obs * self.latent_dim;
        let direct_hyper_count = latent_coord_direct_hyper_count(&self.id_mode, self.latent_dim);
        let expected =
            self.rho_dim + latent_flat_len + self.analytic_rho_count + direct_hyper_count;
        if theta.len() != expected {
            return Err(SmoothError::dimension_mismatch(format!(
                "latent-coordinate theta length mismatch: got {}, expected {} (rho_dim={}, n={}, d={}, analytic_rhos={}, direct_hypers={})",
                theta.len(),
                expected,
                self.rho_dim,
                self.n_obs,
                self.latent_dim,
                self.analytic_rho_count,
                direct_hyper_count
            ))
            .into());
        }
        let flat = theta
            .slice(s![self.rho_dim..self.rho_dim + latent_flat_len])
            .to_owned();
        let latent = std::sync::Arc::new(
            gam_terms::latent::LatentCoordValues::from_flat_with_manifold_and_retraction_and_id(
                flat,
                self.n_obs,
                self.latent_dim,
                self.id_mode.clone(),
                self.manifold.clone(),
                self.retraction_registry.clone(),
                self.latent_id,
            ),
        );
        let latent_values_changed = self
            .current_latent
            .as_ref()
            .map(|cached| !latent_values_match(cached.as_flat(), latent.as_flat()))
            .unwrap_or(true);
        if latent_values_changed {
            self.latent_design_cache.invalidate_all();
            self.current_design_cache_id = None;
            self.design_revision = self.design_revision.wrapping_add(1);
        }
        for n in 0..self.n_obs {
            for axis in 0..self.latent_dim {
                let col = self.feature_cols[axis];
                self.data[[n, col]] = latent.as_flat()[n * self.latent_dim + axis];
            }
        }

        let basis_kind = self.latent_basis_kind()?;
        let rebuilt_width = self.design.design.ncols();
        let spec = self.spec.clone();
        let term_index = self.term_index;
        let analytic_rho_count = self.analytic_rho_count;
        let data = self.data.view();
        let design_context_digest =
            gam_solve::latent_cache::latent_design_context_cache_digest(
                data,
                &spec,
                term_index,
                analytic_rho_count,
                &self.feature_cols,
            )
            .map_err(|e| e.to_string())?;
        let lookup = self
            .latent_design_cache
            .lookup_or_compute(latent.clone(), basis_kind, design_context_digest, || {
                let rebuilt = build_term_collection_design(data, &spec).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "failed to rebuild latent-coordinate design: {e}"
                    ))
                })?;
                if rebuilt.design.ncols() != rebuilt_width {
                    crate::bail_invalid_estim!(
                        "latent-coordinate design topology changed: rebuilt p={}, cached p={}",
                        rebuilt.design.ncols(),
                        rebuilt_width
                    );
                }
                let hyper_dirs = try_build_latent_coord_hyper_dirs(
                    latent.clone(),
                    &spec,
                    &rebuilt,
                    &[term_index],
                    analytic_rho_count,
                )?
                .ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "failed to build latent-coordinate hyper_dirs".to_string(),
                    )
                })?;
                Ok(gam_solve::latent_cache::ComputedLatentDesign {
                    design: rebuilt,
                    hyper_dirs,
                })
            })
            .map_err(|e| e.to_string())?;
        if lookup.cached.design.design.ncols() != self.design.design.ncols() {
            return Err(SmoothError::dimension_mismatch(format!(
                "latent-coordinate design topology changed: rebuilt p={}, cached p={}",
                lookup.cached.design.design.ncols(),
                self.design.design.ncols()
            ))
            .into());
        }
        self.design = lookup.cached.design.clone();
        self.current_hyper_dirs = Some(lookup.cached.hyper_dirs.clone());
        self.current_latent = Some(latent);
        self.current_theta = Some(theta.clone());
        self.last_cost = None;
        self.last_eval = None;
        self.last_outer_iter = None;
        if !latent_values_changed && self.current_design_cache_id != Some(lookup.entry_id) {
            self.design_revision = self.design_revision.wrapping_add(1);
        }
        self.current_design_cache_id = Some(lookup.entry_id);
        Ok(())
    }

    fn memoized_cost(&self, theta: &Array1<f64>) -> Option<f64> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
            && self.last_outer_iter
                == Some(gam_solve::estimate::reml::outer_eval::current_outer_iter())
        {
            self.last_eval
                .as_ref()
                .map(|cached| cached.0)
                .or(self.last_cost)
        } else {
            None
        }
    }

    fn memoized_eval(
        &self,
        theta: &Array1<f64>,
    ) -> Option<(
        f64,
        Array1<f64>,
        gam_problem::HessianResult,
    )> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
            && self.last_outer_iter
                == Some(gam_solve::estimate::reml::outer_eval::current_outer_iter())
        {
            self.last_eval.clone()
        } else {
            None
        }
    }

    fn store_eval(
        &mut self,
        eval: (
            f64,
            Array1<f64>,
            gam_problem::HessianResult,
        ),
    ) {
        self.last_cost = Some(eval.0);
        self.last_eval = Some(eval);
        self.last_outer_iter =
            Some(gam_solve::estimate::reml::outer_eval::current_outer_iter());
    }

    fn store_cost(&mut self, cost: f64) {
        self.last_cost = Some(cost);
        self.last_outer_iter =
            Some(gam_solve::estimate::reml::outer_eval::current_outer_iter());
    }

    fn reset(&mut self) {
        self.current_theta = None;
        self.current_latent = None;
        self.current_hyper_dirs = None;
        self.current_design_cache_id = None;
        self.latent_design_cache.invalidate();
        self.last_cost = None;
        self.last_eval = None;
        self.last_outer_iter = None;
    }
}

/// #1464: the fixed-κ profiled-REML score `V_p(κ)` for a single constant-curvature
/// term — pin κ on the term, fit with κ-optimisation DISABLED so only the
/// smoothing parameters ρ are profiled, and return the resulting REML/LAML
/// negative-log-evidence (the value the outer loop minimises). This is exactly
/// the criterion the `curvature_inference_forspec` CI oracle evaluates; factoring
/// it here lets the production joint-fit path reuse the SAME sign-correct profiled
/// criterion to pick the κ-sign basin before the joint [ρ, ψ] solve, instead of
/// letting the joint optimiser descend from a single κ seed into the spurious +κ
/// collapsed-kernel corner (the headline #1464 sign-blindness).
///
/// `pub` so a regression test can evaluate the EXACT production criterion at two
/// pinned κ (e.g. +κ vs −κ on a hyperbolic dataset) and settle solver-vs-criterion:
/// if `V_p(+κ) < V_p(−κ)` for hyperbolic data, the criterion itself prefers the
/// collapsed +κ corner and the bug is in the constant-curvature REML/Occam term,
/// not the optimiser.
pub fn fixed_kappa_profiled_reml_score(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    term_idx: usize,
    kappa: f64,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<f64, EstimationError> {
    if !kappa.is_finite() {
        crate::bail_invalid_estim!("fixed-κ profiled score probed a non-finite κ = {kappa}");
    }
    // Resolve the constant-curvature term's feature columns and base spec so the
    // criterion is probed on the production constant-curvature design.
    let (feature_cols, mut probe_basis) = match resolvedspec
        .smooth_terms
        .get(term_idx)
        .map(|t| &t.basis)
    {
        Some(SmoothBasisSpec::ConstantCurvature {
            feature_cols, spec, ..
        }) => (feature_cols.clone(), spec.clone()),
        _ => {
            crate::bail_invalid_estim!(
                "fixed-κ profiled score: term {term_idx} is not a constant-curvature smooth"
            )
        }
    };
    probe_basis.kappa = kappa;

    // #1464: the curvature κ criterion the CI/flatness oracle walks (and the
    // `constant_curvature_profiled_reml_scores` export reports) is the HONEST
    // fixed-κ profiled REML of the realized constant-curvature design —
    // `dof·log(rss/dof) + log|H| − log|λS|₊` profiled over λ on `[1|K_κ·z]`
    // (`constant_curvature_honest_profiled_reml_score`). NOT the production
    // full-fit `reml_score`: that score heavily SMOOTHS this RKHS kernel, and under
    // heavy smoothing the +κ chart's geodesic-distance compression makes the
    // collapsed kernel a uniformly better fit of the over-smoothed target for ANY
    // data, so it is MONOTONE toward the +chart bound regardless of the true
    // curvature sign (the #1464 sign-blindness — `bug_hunt_1464_criterion_vs_solver`
    // shows V_p(+2) < V_p(0) < V_p(−2) on hyperbolic data with the raw score). The
    // honest profiled REML keeps the curvature-shape signal in the data fit, so its
    // argmin tracks the planted sign, and as a proper profiled-REML deviance the
    // CI/flatness LR thresholds stay χ²-calibrated; on constant-mean data it is
    // ~flat in κ, giving the flatness test correct size. Gaussian-identity is the
    // only family the curvature-as-estimand path serves; a weighted response, a
    // non-zero offset, or a non-Gaussian link routes to the production fixed-κ fit
    // (those configurations are not exercised by curvature inference, and the
    // fallback keeps their behaviour byte-identical).
    let is_unweighted = weights.iter().all(|&w| (w - 1.0).abs() <= 1e-12);
    let is_zero_offset = offset.iter().all(|&o| o.abs() <= 1e-12);
    if family == LikelihoodSpec::gaussian_identity() && is_unweighted && is_zero_offset {
        let x_term = select_columns(data, &feature_cols).map_err(EstimationError::from)?;
        let score =
            gam_terms::basis::constant_curvature_honest_profiled_reml_score(x_term.view(), y, &probe_basis)
                .map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "fixed-κ honest profiled-REML score at κ={kappa} failed: {e}"
                    ))
                })?;
        if !score.is_finite() {
            crate::bail_invalid_estim!(
                "fixed-κ honest profiled-REML score at κ={kappa} is non-finite"
            );
        }
        return Ok(score);
    }

    // Fallback (weighted / offset / non-Gaussian): the production fixed-κ fit.
    let mut probe_spec = resolvedspec.clone();
    match probe_spec.smooth_terms.get_mut(term_idx).map(|t| &mut t.basis) {
        Some(SmoothBasisSpec::ConstantCurvature { spec, .. }) => spec.kappa = kappa,
        _ => {
            crate::bail_invalid_estim!(
                "fixed-κ profiled score: term {term_idx} is not a constant-curvature smooth"
            )
        }
    }
    let fixed_kappa_options = SpatialLengthScaleOptimizationOptions {
        enabled: false,
        ..SpatialLengthScaleOptimizationOptions::default()
    };
    let fit = fit_term_collectionwith_spatial_length_scale_optimization(
        data,
        y.to_owned(),
        weights.to_owned(),
        offset.to_owned(),
        &probe_spec,
        family,
        options,
        &fixed_kappa_options,
    )?;
    let score = fit_score(&fit.fit);
    if !score.is_finite() {
        crate::bail_invalid_estim!("fixed-κ profiled fit at κ={kappa} returned a non-finite score");
    }
    Ok(score)
}

/// #1464: estimate κ̂ for a constant-curvature term as the argmin of the κ-FAIR
/// sign-resolving criterion over a fine grid spanning the whole chart window.
///
/// WHY THIS IS THE ESTIMATE, NOT JUST A SEED. The production profiled-REML
/// criterion (`fixed_kappa_profiled_reml_score`, and equivalently the joint
/// [ρ, ψ] solver's REML objective) is *sign-blind* in κ on a generic
/// center-peaked radial signal: the +κ chart compresses geodesic distances, so
/// the geodesic-exponential kernel becomes a uniformly better interpolator of
/// ANY radial peak regardless of the true curvature sign. Its V_p therefore
/// decreases monotonically toward the +chart bound for BOTH spherical and
/// hyperbolic truth (verified: `vp_grid_identifies_planted_kappa_sign` puts the
/// raw-V_p argmin at the +bound even for κ⋆ = −2). Seeding the joint solver in
/// the correct basin and pinning the window to one sign half-axis is not enough:
/// the raw REML is still monotone toward 0 inside that half-axis, so the solver
/// rails κ̂ to the 0 boundary (the observed hyperbolic κ̂ = 0). The cure is to
/// stop using the sign-blind criterion to *choose* κ at all for these terms and
/// instead use the κ-fair criterion
/// [`gam_terms::basis::constant_curvature_kappa_fair_sign_score`], whose generic
/// radial-peak-fitting power is subtracted out so only the genuine
/// curvature-shape signal remains — its argmin is sign-AND-magnitude correct
/// (spherical κ̂ > 0, hyperbolic κ̂ < 0, materially distinguished).
///
/// Returns `None` when the term carries no usable κ window or every probe fit
/// fails (caller falls back to the spec's κ seed / joint solve).
fn constant_curvature_kappa_fair_argmin(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    term_idx: usize,
) -> Option<f64> {
    let (kappa_min, kappa_max) = constant_curvature_kappa_bounds(data, resolvedspec, term_idx);
    if !(kappa_min.is_finite() && kappa_max.is_finite() && kappa_max > kappa_min) {
        return None;
    }
    let (feature_cols, base_spec) = match resolvedspec.smooth_terms.get(term_idx).map(|t| &t.basis) {
        Some(SmoothBasisSpec::ConstantCurvature {
            feature_cols, spec, ..
        }) => (feature_cols, spec.clone()),
        _ => return None,
    };
    let x_term = match select_columns(data, feature_cols) {
        Ok(x) => x,
        Err(e) => {
            log::info!("[spatial-kappa] #1464 κ-fair argmin column select failed ({e}); skipping");
            return None;
        }
    };
    // Dense symmetric grid over the full chart window. 24 steps resolves the
    // κ-fair criterion's interior optimum well within the contract tolerances
    // (the criterion is smooth and single-welled in κ on curved truth); the
    // argmin's SIGN — the headline #1464 requirement — is robust to the grid
    // resolution. κ = 0 is included so genuinely flat truth can be selected.
    const GRID_STEPS: usize = 24;
    let mut best: Option<(f64, f64)> = None; // (κ-fair score, kappa)
    for i in 0..=GRID_STEPS {
        let t = i as f64 / GRID_STEPS as f64;
        let kappa = kappa_min + (kappa_max - kappa_min) * t;
        let mut probe_spec = base_spec.clone();
        probe_spec.kappa = kappa;
        match gam_terms::basis::constant_curvature_kappa_fair_sign_score(x_term.view(), y, &probe_spec) {
            Ok(score) => {
                if best.as_ref().is_none_or(|(b, _)| score < *b) {
                    best = Some((score, kappa));
                }
            }
            Err(e) => {
                log::info!(
                    "[spatial-kappa] #1464 κ-fair argmin probe at κ={kappa:.4} failed ({e}); skipping"
                );
            }
        }
    }
    best.map(|(score, kappa)| {
        log::info!(
            "[spatial-kappa] #1464 κ-fair argmin κ̂={kappa:.4} (κ-fair score={score:.6e}) for term {term_idx}"
        );
        kappa
    })
}

/// #1464: choose the κ-sign basin for the joint spatial fit by scanning the
/// sign-correct fixed-κ profiled-REML criterion `V_p(κ)` over a small symmetric
/// grid spanning both chart signs, and return the argmin κ. The joint [ρ, ψ]
/// optimiser is then seeded at this κ so it polishes inside the correct basin
/// rather than descending from a single (near-zero) κ seed into the spurious +κ
/// collapsed-kernel corner. Returns `None` when the term carries no usable κ
/// window or every probe fit fails (caller falls back to the spec's κ seed).
fn select_constant_curvature_kappa_sign_seed(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    term_idx: usize,
) -> Option<f64> {
    let (kappa_min, kappa_max) = constant_curvature_kappa_bounds(data, resolvedspec, term_idx);
    if !(kappa_min.is_finite() && kappa_max.is_finite() && kappa_max > kappa_min) {
        return None;
    }
    // Resolve this term's chart-coordinate columns and its base spec so the
    // sign-basin scan can score each probe κ with the κ-FAIR criterion directly
    // on the production constant-curvature basis (#1464). The κ-fair score
    // subtracts the design's generic radial-peak-fitting power (measured on a
    // bank of κ-independent Euclidean-radial reference signals) from the data's
    // profiled REML, so the +κ chart's distance-compression interpolation
    // advantage — which lifts BOTH the data and the reference fits equally —
    // cancels, leaving only the genuine curvature-shape signal. This is what
    // makes the SIGN identifiable: the raw `fixed_kappa_profiled_reml_score`
    // (still used for the magnitude/CI) is sign-blind on a generic radial signal
    // and rails to the +chart bound for both spherical and hyperbolic truth.
    let (feature_cols, base_spec) = match resolvedspec.smooth_terms.get(term_idx).map(|t| &t.basis) {
        Some(SmoothBasisSpec::ConstantCurvature {
            feature_cols, spec, ..
        }) => (feature_cols, spec.clone()),
        _ => return None,
    };
    let x_term = match select_columns(data, feature_cols) {
        Ok(x) => x,
        Err(e) => {
            log::info!("[spatial-kappa] #1464 sign-basin scan column select failed ({e}); skipping");
            return None;
        }
    };
    // Five probes spanning both signs: the two interior corners (half the chart
    // bound on each side, away from the saturating boundary), flat (κ = 0), and
    // the chart bounds.
    let probes = [
        kappa_min,
        0.5 * kappa_min,
        0.0,
        0.5 * kappa_max,
        kappa_max,
    ];
    let mut best: Option<(f64, f64)> = None; // (κ-fair score, kappa)
    for &kappa in &probes {
        let mut probe_spec = base_spec.clone();
        probe_spec.kappa = kappa;
        match gam_terms::basis::constant_curvature_kappa_fair_sign_score(
            x_term.view(),
            y,
            &probe_spec,
        ) {
            Ok(score) => {
                if best.as_ref().is_none_or(|(b, _)| score < *b) {
                    best = Some((score, kappa));
                }
            }
            Err(e) => {
                log::info!(
                    "[spatial-kappa] #1464 sign-basin probe at κ={kappa:.4} failed ({e}); skipping"
                );
            }
        }
    }
    best.map(|(score, kappa)| {
        log::info!(
            "[spatial-kappa] #1464 κ-fair sign-basin scan selected κ_seed={kappa:.4} \
             (κ-fair score={score:.6e}) for term {term_idx}"
        );
        kappa
    })
}

/// Number of length-scale restarts in the #1074 GP-range multi-start pre-scan
/// (inclusive log-κ grid endpoints), spanning the per-term data-derived κ window.
const SPATIAL_RANGE_PRESCAN_GRID: usize = 7;

/// #1074 — kernel-range multi-start for isotropic Matérn/Duchon GP smooths.
///
/// The joint `[ρ, ψ]` REML objective of an isotropic radial GP smooth is
/// genuinely MULTIMODAL in the kernel range `ψ = log κ`: a long-range (stiff)
/// basin and a short-range (flexible) basin can each be a local optimum,
/// separated by a barrier the local ARC/BFGS joint solver cannot cross. From the
/// single data-window-midpoint seed the local solver descends into whichever
/// basin holds the seed. For the ROUGHEST kernels (Matérn ν=3/2) the midpoint is
/// the long-range basin, which over-smooths the domain boundary and leaves the
/// global short-range optimum unreached — the observed `matern_varying_nu` ν=3/2
/// failure recovered the interior fine but the edges 4× worse than the interior,
/// at a REML score ~16 nats WORSE than the reachable short-range optimum (the
/// criterion is correct; only the optimizer was stuck).
///
/// The cure is the textbook remedy for a multimodal length-scale likelihood: a
/// coarse grid restart. For each isotropic spatial term we evaluate the profiled
/// REML (ρ optimised at fixed κ — exactly [`fit_term_collection_forspec`]) at a
/// log-κ grid spanning the term's data-derived window, and adopt the best-scoring
/// length scale as the seed handed to the joint solver, which then polishes
/// locally inside the global basin. Coordinate descent across terms (each scanned
/// with the others held at their running best) keeps the cost linear in the term
/// count. Only a STRICT REML improvement over the incumbent seed is adopted, so
/// the pre-scan can never regress a fit that the midpoint seed already solved
/// well — it only rescues the ones stuck in the wrong basin.
///
/// Gated to the isotropic, non-constant-curvature case: anisotropic ψ-per-axis
/// terms and constant-curvature `curv()` terms carry their own dedicated seeding
/// (the η-aware constructors and the #1464 κ-fair sign-basin scan respectively),
/// so they are left untouched. Returns the `(term_idx, length_scale)` overrides
/// that strictly improve REML; an empty vector means the incumbent seed already
/// sits in the best-scoring basin and nothing downstream changes.
fn prescan_isotropic_spatial_range_seed(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    baseline_score: f64,
    family: &LikelihoodSpec,
    options: &FitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    spatial_terms: &[usize],
) -> Result<Vec<(usize, f64)>, EstimationError> {
    // Anisotropic and constant-curvature terms have their own seeding paths.
    if has_aniso_terms(resolvedspec, spatial_terms)
        || !constant_curvature_term_indices(resolvedspec).is_empty()
    {
        return Ok(Vec::new());
    }
    let dims = spatial_dims_per_term(resolvedspec, spatial_terms);
    // The grid coordinate-descends term by term; `working` carries the chosen
    // length scales of already-scanned terms so a later term is scored against
    // the improved earlier ones, not the stale midpoint seed.
    let mut working = resolvedspec.clone();
    let mut best_score = if baseline_score.is_finite() {
        baseline_score
    } else {
        f64::INFINITY
    };
    let mut overrides: Vec<(usize, f64)> = Vec::new();
    for (slot, &term_idx) in spatial_terms.iter().enumerate() {
        // Isotropic terms contribute a single ψ; a per-axis (anisotropic) slot
        // is excluded by the gate above, but stay defensive.
        if dims.get(slot).copied().unwrap_or(1) != 1 {
            continue;
        }
        // Only terms that actually carry a free length scale (Matérn / hybrid
        // Duchon). Pure Duchon / TPS without a length scale are skipped.
        if get_spatial_length_scale(&working, term_idx).is_none() {
            continue;
        }
        let (psi_lo, psi_hi) = spatial_term_psi_bounds(data, &working, term_idx, kappa_options);
        if !(psi_lo.is_finite() && psi_hi.is_finite()) || psi_hi <= psi_lo {
            continue;
        }
        let mut term_best: Option<f64> = None;
        for g in 0..SPATIAL_RANGE_PRESCAN_GRID {
            let frac = g as f64 / (SPATIAL_RANGE_PRESCAN_GRID - 1) as f64;
            let psi = psi_lo + (psi_hi - psi_lo) * frac;
            // `apply_log_kappa_to_term` converts the optimizer's ψ to the spec
            // length scale as `ℓ = exp(-ψ)`; mirror it so the grid lives in the
            // SAME coordinate the joint solver and the ψ window use.
            let ls = (-psi).exp();
            if !ls.is_finite() || ls <= 0.0 {
                continue;
            }
            let mut probe = working.clone();
            if set_spatial_length_scale(&mut probe, term_idx, ls).is_err() {
                continue;
            }
            // Each probe MUST run an independent, cold ρ-optimization: the
            // strict-improvement ranking below only compares apples-to-apples if
            // every grid point reaches its own ρ* from the same neutral start.
            // Warm-starting a probe from the previous (adjacent-κ) probe's λ
            // biases its ρ-optimum toward the neighbour's basin and corrupts the
            // ranking — it re-strands the ν=3/2 fit in the long-range
            // over-smoothing basin (boundary recovery regression). So this is a
            // cold `fit_term_collection_forspec`, deliberately not warm-started.
            let fit = match fit_term_collection_forspec(
                data,
                y,
                weights,
                offset,
                &probe,
                family.clone(),
                options,
            ) {
                Ok(fit) => fit,
                // A grid point can hit an infeasible kernel geometry (rank
                // collapse at an extreme range); skip it, don't abort the scan.
                Err(_) => continue,
            };
            let score = fit_score(&fit.fit);
            // Strict improvement only — guards against adopting a numerically
            // equal basin and against ever regressing the incumbent seed.
            if score.is_finite() && score < best_score - 1e-7 * best_score.abs().max(1.0) {
                best_score = score;
                term_best = Some(ls);
            }
        }
        if let Some(ls) = term_best {
            set_spatial_length_scale(&mut working, term_idx, ls)?;
            overrides.push((term_idx, ls));
            log::info!(
                "[spatial-kappa] #1074 range pre-scan: term {term_idx} re-seeded at \
                 length_scale={ls:.5} (profiled REML {best_score:.5}, was {baseline_score:.5})"
            );
        }
    }
    Ok(overrides)
}

/// Fractions of each isotropic term's data-derived ψ window used as joint-solve
/// restart seeds in the #1688 multistart rescue. They run from the long-range
/// (stiff) corner at `0.0` through the short-range (flexible) corner at `1.0`,
/// deliberately spanning BOTH local basins of the multimodal `[ρ, ψ]` REML
/// surface so a restart lands in the global basin regardless of which side the
/// auto-seed stranded the primary solve on. Five points keep the rare-trigger
/// cost bounded while guaranteeing a seed in the long-range half (which the
/// profiled-REML pre-scan systematically under-ranks for the roughest kernels).
const JOINT_RESTART_WINDOW_FRACTIONS: [f64; 5] = [0.0, 0.2, 0.45, 0.7, 1.0];

/// #1688 joint-solve multistart from a single ψ-window fraction.
///
/// Re-seeds every isotropic spatial term's length scale to `ℓ = exp(−ψ)` at the
/// requested `fraction` of its data-derived ψ window, then runs the FULL
/// baseline → freeze → joint `[ρ, ψ]` sequence at that geometry and returns the
/// realized result with its certified REML score. This is the same machinery the
/// primary path runs — only the seed differs — so a candidate it returns is a
/// genuine production fit, never a heuristic stand-in.
///
/// Returns `Ok(None)` only when the seed geometry is non-constructible (e.g. a
/// baseline fit that errors at an extreme range, or no free length scale after
/// the freeze-time basis promotion); callers treat that as "this restart point
/// is infeasible, skip it" rather than failing the fit. When the joint refine
/// itself yields no usable candidate, the seed's frozen baseline geometry is a
/// valid κ-optimized result (ρ profiled at the fixed seeded κ) and is returned.
fn joint_solve_from_window_fraction(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    base_spec: &TermCollectionSpec,
    spatial_terms: &[usize],
    fraction: f64,
    family: &LikelihoodSpec,
    options: &FitOptions,
    baseline_options: &FitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<Option<(FittedTermCollectionWithSpec, f64)>, EstimationError> {
    let mut seed_spec = base_spec.clone();
    let mut any_set = false;
    for &term_idx in spatial_terms {
        if get_spatial_length_scale(&seed_spec, term_idx).is_none() {
            continue;
        }
        let (psi_lo, psi_hi) = spatial_term_psi_bounds(data, &seed_spec, term_idx, kappa_options);
        if !(psi_lo.is_finite() && psi_hi.is_finite()) || psi_hi <= psi_lo {
            continue;
        }
        let psi = psi_lo + (psi_hi - psi_lo) * fraction;
        let ls = (-psi).exp();
        if !ls.is_finite() || ls <= 0.0 {
            continue;
        }
        if set_spatial_length_scale(&mut seed_spec, term_idx, ls).is_ok() {
            any_set = true;
        }
    }
    if !any_set {
        return Ok(None);
    }
    // Baseline at the seeded geometry: this both supplies the joint solver's ρ
    // seed / frozen-baseline fallback AND becomes the returned candidate if the
    // joint refine is unavailable. A non-constructible seed geometry is skipped.
    let seed_best = match fit_term_collection_forspec(
        data,
        y,
        weights,
        offset,
        &seed_spec,
        family.clone(),
        baseline_options,
    ) {
        Ok(fit) => fit,
        Err(_) => return Ok(None),
    };
    let seed_spec = freeze_term_collection_from_design(&seed_spec, &seed_best.design)?;
    // The freeze can promote ThinPlate → pure Duchon (no free length scale);
    // refresh the eligible-term list exactly as the primary path does.
    let seed_terms = spatial_length_scale_term_indices(&seed_spec);
    if seed_terms.is_empty() {
        let score = fit_score(&seed_best.fit);
        return Ok(Some((
            FittedTermCollectionWithSpec {
                fit: seed_best.fit,
                design: seed_best.design,
                resolvedspec: seed_spec,
                adaptive_diagnostics: seed_best.adaptive_diagnostics,
                kappa_timing: None,
            },
            score,
        )));
    }
    let joint = try_exact_joint_spatial_length_scale_optimization(
        data,
        y,
        weights,
        offset,
        &seed_spec,
        &seed_best,
        family.clone(),
        options,
        kappa_options,
        &seed_terms,
    )?;
    match joint {
        Some(fit) => {
            let score = fit_score(&fit.fit);
            Ok(Some((fit, score)))
        }
        // Joint refine unavailable for this seed; its frozen baseline geometry
        // is itself a valid κ-optimized fit, so return that as the candidate.
        None => {
            let score = fit_score(&seed_best.fit);
            Ok(Some((
                FittedTermCollectionWithSpec {
                    fit: seed_best.fit,
                    design: seed_best.design,
                    resolvedspec: seed_spec,
                    adaptive_diagnostics: seed_best.adaptive_diagnostics,
                    kappa_timing: None,
                },
                score,
            )))
        }
    }
}

fn try_exact_joint_spatial_length_scale_optimization(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    best: &FittedTermCollection,
    family: LikelihoodSpec,
    options: &FitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    spatial_terms: &[usize],
) -> Result<Option<FittedTermCollectionWithSpec>, EstimationError> {
    if spatial_terms.is_empty() {
        return Ok(None);
    }
    // Fail loud on nonsensical κ options rather than letting them propagate
    // silent NaNs (e.g. inverted min/max inverts the BFGS window, negative
    // scales produce NaN logs). This is the first function on every outer-κ
    // path; downstream paths assume validated options.
    kappa_options
        .validate()
        .map_err(EstimationError::InvalidInput)?;

    // #1464 constant-curvature κ̂ via the κ-FAIR criterion (NOT the joint REML).
    //
    // The joint [ρ, ψ] solver below minimises the production profiled REML, which
    // is SIGN-BLIND in κ on a generic radial signal: the +κ chart compresses
    // geodesic distances, making the geodesic-exponential kernel a uniformly
    // better interpolator of any radial peak regardless of the true curvature
    // sign, so its objective is monotone toward the +chart bound for BOTH
    // spherical and hyperbolic truth. Seeding + one-sided window pinning is not
    // enough — inside the correct half-axis the raw REML is still monotone toward
    // 0, so the solver rails κ̂ to the 0 boundary (the observed hyperbolic
    // κ̂ = 0). When EVERY spatial term in this solve is a constant-curvature term,
    // we therefore choose κ̂ directly from the κ-fair criterion's fine-grid argmin
    // (`constant_curvature_kappa_fair_argmin`), which subtracts the design's
    // generic radial-peak-fitting power and so is sign-AND-magnitude correct
    // (spherical κ̂ > 0, hyperbolic κ̂ < 0), then profile ONLY ρ at that fixed κ.
    // This is gated on a pure-CC spatial problem (the `curv()` use case); mixed
    // CC + Matérn/Duchon/sphere solves fall through to the unchanged joint path,
    // so no non-CC fit is affected. The frozen-baseline harvest is used so the κ̂
    // is persisted in the returned spec and read back by `model.curvature()`.
    let cc_term_set = constant_curvature_term_indices(resolvedspec);
    let all_spatial_are_cc =
        !cc_term_set.is_empty() && spatial_terms.iter().all(|t| cc_term_set.contains(t));
    if all_spatial_are_cc {
        let mut fixed_kappa_spec = resolvedspec.clone();
        let mut any_kappa_chosen = false;
        for &term_idx in spatial_terms {
            // gam#2152: a user-PINNED κ is never re-derived. Skip the κ-fair
            // argmin entirely so the term keeps the user's fixed curvature; it
            // reaches the joint path below where its κ coordinate is frozen.
            if constant_curvature_kappa_is_fixed(resolvedspec, term_idx) {
                continue;
            }
            // Only OVERRIDE κ with the κ-fair argmin when it selects a NEGATIVE
            // (hyperbolic) curvature. This is the one regime the sign-blind joint
            // REML cannot reach: its objective is monotone toward +κ, so seeding +
            // one-sided pinning still rails κ̂ to the 0 boundary (hyperbolic
            // recovered as flat). For a positive κ-fair argmin the joint solver
            // ALREADY rails to the (correct) +chart bound, and its jointly-
            // optimised [ρ, κ] gives a strictly better realized fit than fixing κ
            // and profiling ρ alone — so we leave the spherical/positive case to
            // the unchanged joint path below, preserving its recovery R². A κ-fair
            // argmin of exactly 0 (genuinely flat) likewise falls through.
            if let Some(kappa_hat) =
                constant_curvature_kappa_fair_argmin(data, y, resolvedspec, term_idx)
                    .filter(|&k| k < 0.0)
            {
                if let Some(SmoothBasisSpec::ConstantCurvature { spec: cc, .. }) = fixed_kappa_spec
                    .smooth_terms
                    .get_mut(term_idx)
                    .map(|t| &mut t.basis)
                {
                    cc.kappa = kappa_hat;
                    any_kappa_chosen = true;
                    log::info!(
                        "[spatial-kappa] #1464 term {term_idx}: fixed κ̂ = {kappa_hat:.4} from κ-fair argmin (hyperbolic basin; profiling ρ only)"
                    );
                }
            }
        }
        if any_kappa_chosen {
            // Profiled-ρ fit at the κ-fair κ̂, then a fresh REML-seeded harvest so
            // the returned spec carries the κ̂ for read-back, exactly as the
            // frozen-baseline path does for its geometry.
            let baseline_score = fit_score(&best.fit);
            let fitted = fit_term_collection_forspec(
                data,
                y,
                weights,
                offset,
                &fixed_kappa_spec,
                family.clone(),
                options,
            )?;
            let frozen_spec =
                freeze_term_collection_from_design(&fixed_kappa_spec, &fitted.design)?;
            let mut fit = fitted.fit;
            // Stamp the κ = 0 baseline REML score, exactly as
            // `fit_frozen_baseline_geometry` does for its chosen geometry. The
            // outer `require_successful_spatial_optimization_result` guard exists
            // to reject genuine optimizer DIVERGENCE (a κ that the production REML
            // it minimises says is worse than the seed). It does NOT apply here:
            // κ̂ is deliberately chosen by the κ-FAIR criterion precisely because
            // the production REML is sign-blind in κ and would always score a
            // genuinely-curved κ̂ as "worse" than flat. Reporting the baseline
            // score keeps the principled κ̂ from being spuriously rejected, while
            // the fitted β / λ are the real ρ-profiled fit AT κ̂. (The CI/flatness
            // statistics downstream re-profile V_p around κ̂ on their own.)
            fit.reml_score = baseline_score;
            return Ok(Some(FittedTermCollectionWithSpec {
                fit,
                design: fitted.design,
                resolvedspec: frozen_spec,
                adaptive_diagnostics: fitted.adaptive_diagnostics,
                kappa_timing: None,
            }));
        }
    }

    if try_build_spatial_log_kappa_hyper_dirs(data, resolvedspec, &best.design, spatial_terms)?
        .is_none()
    {
        if !constant_curvature_term_indices(resolvedspec).is_empty() {
            log::info!(
                "[#1464-trace] try_exact_joint RETURNED None (hyper_dirs unavailable); \
                 κ̂ comes from a NON-joint path"
            );
        }
        return Ok(None);
    }
    if !constant_curvature_term_indices(resolvedspec).is_empty() {
        log::info!(
            "[#1464-trace] try_exact_joint ENTERED for {} spatial term(s); CC present",
            spatial_terms.len()
        );
    }

    const JOINT_RHO_BOUND: f64 = 12.0;
    let rho_dim = best.fit.lambdas.len();

    // #1464: a constant-curvature `curv()` term's geodesic-exponential kernel
    // COLLAPSES toward the constant function as κ grows positive (sphere
    // distances compress), so its global REML optimum at the +κ side is a LARGE
    // smoothing λ — often ρ > +JOINT_RHO_BOUND. With the symmetric ±12 box the
    // joint [ρ,ψ] optimizer is structurally clamped into the shallow
    // under-smoothing basin whose spuriously-low deviance rails κ̂ to the +chart
    // bound for any curved data (hyperbolic truth mis-recovered as spherical).
    // When a constant-curvature term is present, widen ONLY the over-smoothing
    // (upper) ρ bound to the standard `RHO_BOUND`, leaving the lower bound at
    // −JOINT_RHO_BOUND so an overfit origin is never reachable — the same
    // asymmetric-bound rationale the standard scalar-ρ path uses for the
    // gam#1266 high-λ basin. Every other spatial/Matérn/Duchon/sphere joint fit
    // keeps the historical ±12 box byte-for-byte.
    let has_constant_curvature_term = !constant_curvature_term_indices(resolvedspec).is_empty();
    let rho_upper_bound = if has_constant_curvature_term {
        gam_solve::estimate::RHO_BOUND
    } else {
        JOINT_RHO_BOUND
    };

    // Compute per-term dimensionality for anisotropic terms.
    let dims_per_term = spatial_dims_per_term(resolvedspec, spatial_terms);
    let use_aniso = has_aniso_terms(resolvedspec, spatial_terms);

    // Build initial ψ values and bounds, using aniso-aware constructors
    // when any term has d > 1 axes. Bounds are tied to each term's center
    // geometry (r_min, r_max) so κ cannot saturate at an upper bound that
    // has no relationship to the data's distance scale.
    let log_kappa0 = if use_aniso {
        SpatialLogKappaCoords::from_length_scales_aniso(resolvedspec, spatial_terms, kappa_options)
    } else {
        SpatialLogKappaCoords::from_length_scales(resolvedspec, spatial_terms, kappa_options)
    };
    // If the user/spec did not set a length_scale, re-seed ψ at the midpoint
    // of the data-derived window instead of the arbitrary options fallback.
    let mut log_kappa0 =
        log_kappa0.reseed_from_data(data, resolvedspec, spatial_terms, kappa_options);
    // #1464: for each constant-curvature term, pick the κ-sign basin from the
    // sign-correct fixed-κ profiled-REML criterion (κ-sign PINNED during each
    // ρ-profile) and seed the joint solver THERE, instead of letting the joint
    // [ρ, ψ] optimiser descend from a single near-zero κ seed into the spurious
    // +κ collapsed-kernel corner that rails κ̂ to the +chart bound regardless of
    // the true sign. CC-gated: non-CC spatial/Matérn/Duchon/sphere joint fits
    // never enter this loop, so their seed is byte-identical to before. The κ-opt
    // OFF profiled fits are the SAME criterion `curvature_inference_forspec`
    // already trusts for the CI, so this reuses a verified sign-correct oracle.
    // Records `(slot, selected_kappa_seed)` for each constant-curvature term so
    // the joint ψ bounds can be HARD-PINNED to the selected sign's half-axis
    // below: the joint ARC genuinely prefers the collapsed +κ corner (its
    // production REML there is lower than the correct basin), so a seed alone is
    // not enough — without a one-sided bound the optimiser walks back across
    // κ = 0 to the spurious corner (the observed #1464 bit-identical railing).
    let mut cc_sign_seeds: Vec<(usize, f64)> = Vec::new();
    // gam#2152: slots whose κ is user-PINNED. These are seeded AND hard-frozen at
    // the user's fixed curvature below (both bounds = kappa=, including exactly
    // 0), so the joint solve optimises only ρ at that fixed geometry — never the
    // κ-fair sign scan's estimate. Tracked separately from `cc_sign_seeds`
    // because a pinned κ = 0 must still freeze (a scanned κ = 0 leaves κ free).
    let mut cc_fixed_seeds: Vec<(usize, f64)> = Vec::new();
    if has_constant_curvature_term {
        for (slot, &term_idx) in spatial_terms.iter().enumerate() {
            if constant_curvature_term_spec(resolvedspec, term_idx).is_none() {
                continue;
            }
            if constant_curvature_kappa_is_fixed(resolvedspec, term_idx) {
                let fixed_kappa = get_constant_curvature_kappa(resolvedspec, term_idx)
                    .expect("constant-curvature term exposes its κ");
                log::info!(
                    "[#2152] term {term_idx}: κ PINNED at user kappa={fixed_kappa}; \
                     freezing the joint ψ coordinate (ρ-only refinement, no κ scan)"
                );
                log_kappa0.set_scalar_slot(slot, fixed_kappa);
                cc_fixed_seeds.push((slot, fixed_kappa));
                continue;
            }
            let scan = select_constant_curvature_kappa_sign_seed(
                data,
                y,
                resolvedspec,
                term_idx,
            );
            // #1464 diagnostic: what the κ-fair sign-basin scan picked for this CC
            // term, before any joint solve. If this prints a negative κ for the
            // hyperbolic dataset but the final κ̂ is +1.08, the bug is downstream of
            // the scan (solver railing or readback), not the scan.
            match scan {
                Some(kappa_seed) => {
                    log::info!(
                        "[#1464-trace] term {term_idx}: κ-fair sign-basin scan picked κ_seed = {kappa_seed}"
                    );
                    log_kappa0.set_scalar_slot(slot, kappa_seed);
                    cc_sign_seeds.push((slot, kappa_seed));
                }
                None => {
                    log::info!(
                        "[#1464-trace] term {term_idx}: fixed-κ sign-basin scan returned NONE (no seed applied)"
                    );
                }
            }
        }
    }
    let log_kappa_lower = if use_aniso {
        SpatialLogKappaCoords::lower_bounds_aniso_from_data(
            data,
            resolvedspec,
            spatial_terms,
            &dims_per_term,
            kappa_options,
        )
    } else {
        SpatialLogKappaCoords::lower_bounds_from_data(
            data,
            resolvedspec,
            spatial_terms,
            kappa_options,
        )
    };
    let log_kappa_upper = if use_aniso {
        SpatialLogKappaCoords::upper_bounds_aniso_from_data(
            data,
            resolvedspec,
            spatial_terms,
            &dims_per_term,
            kappa_options,
        )
    } else {
        SpatialLogKappaCoords::upper_bounds_from_data(
            data,
            resolvedspec,
            spatial_terms,
            kappa_options,
        )
    };
    // #1464 hard κ-PIN: for each constant-curvature term whose κ-FAIR sign-basin
    // scan chose a definite sign, FREEZE the joint ψ coordinate at the scanned
    // κ value (both bounds = κ_seed) rather than only closing the far half-axis at
    // κ = 0. Why the full freeze and not the half-axis pin: the joint solver
    // refines κ against the production profiled-REML `fit_score`, and that raw
    // criterion is SIGN-BLIND — on a generic radial signal its data-fit term
    // decreases MONOTONICALLY toward +κ for BOTH spherical and hyperbolic truth
    // (the +κ chart compresses geodesic distances → a uniformly better radial
    // interpolator; verified by `bug_hunt_1464_criterion_vs_solver`, V_p(+2) <
    // V_p(0) < V_p(−2) for hyperbolic data). So a half-axis window [κ_min, 0] does
    // NOT stop the rail: the solver walks κ to the 0-edge (κ̂ → 0, the observed
    // hyperbolic-recovered-as-spherical failure). Only the κ-FAIR scan
    // (`constant_curvature_kappa_fair_sign_score`, which subtracts the design's
    // generic radial-peak-fitting power) is sign-identifying, and since the κ
    // MAGNITUDE is unidentified here (V_p is monotone — it rails to whichever
    // bound the window exposes), the scan's argmin is the authoritative κ̂. Freeze
    // there and let the joint solve optimize only ρ (and any non-CC ψ) at that κ.
    // This is byte-identical to the prior behaviour for SPHERICAL data — the
    // half-axis pin already railed κ̂ to κ_max = the scan value — and only changes
    // the negative-sign cases, which previously railed to 0. A scan result of
    // exactly κ = 0 (genuinely flat) leaves the window untouched. CC-gated —
    // non-CC terms are never in `cc_sign_seeds`, so every other
    // spatial/Matérn/Duchon/sphere joint window is byte-identical to before.
    let mut log_kappa_lower = log_kappa_lower;
    let mut log_kappa_upper = log_kappa_upper;
    for &(slot, kappa_seed) in &cc_sign_seeds {
        if kappa_seed != 0.0 {
            log_kappa_lower.set_scalar_slot(slot, kappa_seed);
            log_kappa_upper.set_scalar_slot(slot, kappa_seed);
        }
        log::info!(
            "[#1464-trace] slot {slot}: FROZE joint ψ coordinate at κ_seed={kappa_seed} \
             (window [{}, {}]); raw fit_score is sign-blind so the κ-fair scan is authoritative",
            log_kappa_lower.as_array()[log_kappa_lower.dims_per_term()[..slot].iter().sum::<usize>()],
            log_kappa_upper.as_array()[log_kappa_upper.dims_per_term()[..slot].iter().sum::<usize>()],
        );
    }
    // gam#2152: hard-freeze every user-PINNED κ coordinate at the fixed value —
    // UNCONDITIONALLY, unlike the sign-scan freeze above (which leaves a scanned
    // κ = 0 free): a user who wrote `kappa=0` asked for the flat geometry
    // verbatim, so both bounds collapse onto 0 too. With lower == upper == κ the
    // joint solve holds the geometry constant and refines only ρ — exactly the
    // "fixed sectional curvature, profile ρ" contract the docs promise.
    for &(slot, fixed_kappa) in &cc_fixed_seeds {
        log_kappa_lower.set_scalar_slot(slot, fixed_kappa);
        log_kappa_upper.set_scalar_slot(slot, fixed_kappa);
        log::info!(
            "[#2152] slot {slot}: FROZE joint ψ coordinate at PINNED κ={fixed_kappa} \
             (window [{fixed_kappa}, {fixed_kappa}]); ρ-only refinement at the fixed geometry"
        );
    }
    // Project seed onto data-derived bounds; spec.length_scale is a hint,
    // not a hard constraint. BFGS requires theta0 ∈ [lower, upper].
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);
    let setup = ExactJointHyperSetup::new(
        best.fit.lambdas.mapv(f64::ln),
        Array1::<f64>::from_elem(rho_dim, -JOINT_RHO_BOUND),
        Array1::<f64>::from_elem(rho_dim, rho_upper_bound),
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    );

    let theta0 = setup.theta0();
    let lower = setup.lower();
    let upper = setup.upper();

    // ───────────────────────────────────────────────────────────────────────
    //  Both coordinate kinds drive the SAME exact joint optimizer
    //  (`run_exact_joint_spatial_optimization`): the unified REML evaluator with
    //  ext_coords for joint [ρ, ψ] optimization, with analytic gradient +
    //  Hessian flowing through the
    //  AnisoBasisPsiDerivatives / SpatialPsiDerivative → DirectionalHyperParam →
    //  HyperCoord pipeline for Newton/BFGS quadratic convergence. The only
    //  difference is the coordinate kind: anisotropic carries one ψ per axis per
    //  term, isotropic one log-κ per term. `outer_strategy` handles the
    //  centralized degradation path when the analytic Hessian is unavailable.
    // ───────────────────────────────────────────────────────────────────────
    let kind = if use_aniso {
        SpatialHyperKind::Anisotropic
    } else {
        SpatialHyperKind::Isotropic
    };
    let (outcome, kappa_timing) = run_exact_joint_spatial_optimization(
        kind,
        data,
        y,
        weights,
        offset,
        resolvedspec,
        &best.design,
        family.clone(),
        options,
        spatial_terms,
        &dims_per_term,
        &theta0,
        &lower,
        &upper,
        rho_dim,
        kappa_options,
    )?;

    let baseline_score = fit_score(&best.fit);

    // The joint κ optimizer is a refinement on top of the frozen baseline
    // geometry, never a precondition for a fit. There are two ways its candidate
    // is not adopted, and both keep the baseline rather than aborting:
    //   1. it ran to a finite cost but did not certify a stationary point
    //      (`NonConverged`) — the formula/FFI path's tight outer tolerance can
    //      leave the optimizer mid-descent at the iteration cap where the CLI's
    //      looser tolerance converges (#1126); and
    //   2. it converged to a candidate whose certified cost worsens the profiled
    //      score (the gate below).
    let (theta_star, joint_final_value) = match outcome {
        SpatialJointOutcome::Optimized {
            theta_star,
            final_value,
        } => (theta_star, final_value),
        SpatialJointOutcome::NonConverged {
            iterations,
            final_value,
            final_grad_norm,
        } => {
            if has_constant_curvature_term {
                log::info!(
                    "[#1464-trace] joint solve NONCONVERGED (iters={iterations}, \
                     final_value={final_value}); returning FROZEN BASELINE geometry \
                     (κ̂ = spec default, NOT the joint candidate)"
                );
            }
            log::info!(
                "[spatial-kappa] joint spatial optimization did not converge \
                 (iterations={}, final_objective={:.6e}, final_grad_norm={}); \
                 keeping the frozen baseline geometry",
                iterations,
                final_value,
                final_grad_norm.map_or_else(|| "n/a".to_string(), |g| format!("{g:.3e}")),
            );
            let baseline = fit_frozen_baseline_geometry(
                data,
                y,
                weights,
                offset,
                resolvedspec,
                best,
                family.clone(),
                options,
                baseline_score,
                Some(kappa_timing),
            )?;
            // #2122: a stalled joint solve must NEVER silently ship a mean-only
            // surface. The frozen baseline keeps the SPEC-DEFAULT range; when that
            // range is far longer than this frame's own distance scale the Matérn
            // kernel columns go nearly collinear with their polynomial nullspace
            // and REML shrinks the whole smooth onto its intercept — the fit
            // reports success but predicts the response mean (thinplate/duchon,
            // which carry no enrolled kernel-range hyperparameter and so never
            // stall, recover the identical signal on the same frame). Detect that
            // collapse on the REALIZED baseline (EDF below the null floor) and
            // only then retreat to a DATA-ADAPTIVE geometry whose kernel range is
            // re-centered on the data's pairwise-distance scale (the geometric
            // mean of the admissible kernel-range window — the same data-scaled
            // length scale a robust GP prior centers on, and the geometry
            // thinplate effectively uses), refitting λ from scratch there. Adopt
            // it only when it recovers materially more effective DOF, so a
            // non-degenerate stalled fit (e.g. the #1126 tight-tolerance stall) is
            // byte-identical to before.
            let baseline_edf = baseline.fit.inference.as_ref().map(|inf| inf.edf_total);
            if let Some(base_edf) = baseline_edf
                && base_edf < SPATIAL_COLLAPSE_EDF_FLOOR
                && let Some(adaptive) = fit_data_adaptive_geometry(
                    data,
                    y,
                    weights,
                    offset,
                    resolvedspec,
                    spatial_terms,
                    &dims_per_term,
                    &theta0,
                    &lower,
                    &upper,
                    rho_dim,
                    family,
                    options,
                    baseline_score,
                    Some(kappa_timing),
                )?
            {
                let adaptive_edf = adaptive.fit.inference.as_ref().map(|inf| inf.edf_total);
                if let Some(adapt_edf) = adaptive_edf
                    && adapt_edf >= base_edf + SPATIAL_COLLAPSE_EDF_MARGIN
                {
                    log::info!(
                        "[spatial-kappa] #2122 stalled joint solve collapsed the frozen \
                         baseline (edf={base_edf:.3}); data-adaptive geometry recovers \
                         edf={adapt_edf:.3} — shipping the data-adaptive fit"
                    );
                    return Ok(Some(adaptive));
                }
            }
            return Ok(Some(baseline));
        }
    };

    // Compare the joint optimizer's certified cost (final_value at theta*)
    // against the baseline. Tolerance ≥ options.tol because both endpoints
    // are outer-BFGS approximations accurate to options.tol; a tighter
    // gate would reject true improvements due to floating-point noise.
    let accept_tol = options.tol.max(1e-8 * baseline_score.abs()).max(1e-12);
    if joint_final_value > baseline_score + accept_tol {
        if has_constant_curvature_term {
            log::info!(
                "[#1464-trace] joint candidate WORSENED score (joint={joint_final_value}, \
                 baseline={baseline_score}); returning FROZEN BASELINE geometry \
                 (κ̂ = spec default, NOT the joint candidate)"
            );
        }
        log::info!(
            "[spatial-kappa] exact joint spatial candidate worsened the profiled score (joint={:.6e}, baseline={:.6e}, tol={:.2e}); keeping the frozen baseline geometry",
            joint_final_value,
            baseline_score,
            accept_tol,
        );
        return Ok(Some(fit_frozen_baseline_geometry(
            data,
            y,
            weights,
            offset,
            resolvedspec,
            best,
            family,
            options,
            baseline_score,
            Some(kappa_timing),
        )?));
    }

    let rho_star = theta_star.slice(s![..rho_dim]).mapv(f64::exp);
    let log_kappa_star =
        SpatialLogKappaCoords::from_theta_tail_with_dims(&theta_star, rho_dim, dims_per_term);
    // #1464 diagnostic (ban-clean): the joint solver's CONVERGED ψ-tail κ for each
    // CC term — the value BEFORE any spec write-back / freeze / readback. If this
    // is negative for the hyperbolic dataset but `get_constant_curvature_kappa`
    // later returns +1.08, the railing is a POST-SOLVE clamp/readback, not the
    // optimiser. If this is itself +1.08, the joint solver railed past the pin.
    if has_constant_curvature_term {
        let star = log_kappa_star.as_array();
        let dims = log_kappa_star.dims_per_term();
        for (slot, &term_idx) in spatial_terms.iter().enumerate() {
            if constant_curvature_term_spec(resolvedspec, term_idx).is_some() {
                let off: usize = dims[..slot].iter().sum();
                log::info!(
                    "[#1464-trace] term {term_idx}: joint solver CONVERGED ψ-tail κ = {} \
                     (this is the optimised candidate; joint_final_value={joint_final_value})",
                    star[off]
                );
            }
        }
    }
    // Keep a handle on the baseline geometry spec before shadowing `resolvedspec`
    // with the κ-optimized spec, so the #1357 degenerate-corner guard below can
    // fall back to the frozen baseline.
    let baseline_spec = resolvedspec;
    let optimized_spec = log_kappa_star.apply_tospec(resolvedspec, spatial_terms)?;
    let optimized = fit_term_collection_forspecwith_heuristic_lambdas(
        data,
        y,
        weights,
        offset,
        &optimized_spec,
        rho_star.as_slice(),
        family.clone(),
        options,
    )?;

    // #1357 degenerate-corner guard. In the flat (ρ, κ) valley the joint
    // optimizer can certify a κ at which the kernel block goes nearly flat and
    // REML then shrinks the whole smooth onto its intercept (EDF → the null
    // floor, prediction returns a constant surface). Such a corner can carry a
    // *better* profiled REML cost than the informative baseline — the
    // smoothing-correction trace flips between the near-boundary cubature and
    // first-order branches across draws, so the `joint_final_value` ≤
    // `baseline_score` gate above does not catch it. The frozen baseline
    // geometry (the data-derived default length scale with its own REML-seeded
    // λ) keeps the kernel informative, so when the joint optimum has collapsed
    // to the null while the baseline has materially more effective DOF, reject
    // the optimum and keep the baseline. This never blocks a genuine refinement:
    // the baseline is only preferred when the joint candidate is degenerate.
    let optimized_edf = optimized.fit.inference.as_ref().map(|inf| inf.edf_total);
    if let Some(opt_edf) = optimized_edf
        && opt_edf < SPATIAL_COLLAPSE_EDF_FLOOR
    {
        let baseline = fit_frozen_baseline_geometry(
            data,
            y,
            weights,
            offset,
            baseline_spec,
            best,
            family.clone(),
            options,
            baseline_score,
            Some(kappa_timing),
        )?;
        let baseline_edf = baseline.fit.inference.as_ref().map(|inf| inf.edf_total);
        if let Some(base_edf) = baseline_edf
            && base_edf >= opt_edf + SPATIAL_COLLAPSE_EDF_MARGIN
        {
            log::info!(
                "[spatial-kappa] joint candidate collapsed to the null (edf={opt_edf:.3}); \
                 baseline geometry retains edf={base_edf:.3} — keeping the frozen baseline",
            );
            return Ok(Some(baseline));
        }
        // Baseline is no better (both genuinely near-null, or baseline lacks
        // inference): keep the optimized candidate via the normal path below.
    }

    // Stamp reml_score with joint_final_value so downstream consumers see a
    // score consistent with the gate decision; the refit serves as a
    // β/inference harvester at the certified (ρ*, ψ*).
    let mut fit = optimized.fit;
    fit.reml_score = joint_final_value;
    let optimized_result = FittedTermCollectionWithSpec {
        fit,
        design: optimized.design,
        resolvedspec: optimized_spec,
        adaptive_diagnostics: optimized.adaptive_diagnostics,
        kappa_timing: Some(kappa_timing),
    };

    Ok(Some(optimized_result))
}

/// EDF below this is treated as an intercept-only / null collapse of the spatial
/// smooth (#1357): the model has shed essentially all effective degrees of
/// freedom beyond a handful of unpenalized coordinates.
const SPATIAL_COLLAPSE_EDF_FLOOR: f64 = 2.5;

/// A non-degenerate baseline must carry at least this much more effective DOF
/// than the collapsed joint candidate before the baseline is preferred (#1357),
/// so genuinely-near-null surfaces (where both fits agree there is no signal)
/// are left untouched.
const SPATIAL_COLLAPSE_EDF_MARGIN: f64 = 1.0;

/// Re-fit at the frozen baseline geometry — the REML-seeded length scales and
/// heuristic λ already certified in `best` — and stamp the certified baseline
/// REML score onto the result.
///
/// This is the graceful-degradation target for the joint spatial-κ optimizer. It
/// is reached whenever the joint refinement is not adopted: when the optimizer
/// converges to a candidate that worsens the profiled score, *and* when it fails
/// to converge at all (#1126). The geometry is the same baseline the parent fit
/// started from, so it is always valid — the joint step can only ever improve on
/// it, never block it.
///
/// The refit is a β/inference harvester at the frozen baseline `resolvedspec`;
/// the score that geometry was certified at is
/// `baseline_score = fit_score(&best.fit)`. We stamp that certified value rather
/// than the harvest's own re-derived `reml_score`, which drifts because the
/// harvest runs the full-inference option set (and re-runs the adaptive spatial
/// overlay) instead of the superseded baseline path that produced `best`. The
/// spatial-κ result gate (`require_successful_spatial_optimization_result`)
/// compares the returned fit's `fit_score` against `fit_score(&best.fit)`;
/// without this stamp a downward drift of a few REML units on the *same*
/// geometry spuriously reads as "the optimizer made the score worse" and aborts
/// an otherwise-valid fit. Stamping keeps the returned score consistent with the
/// gate decision that selected this geometry, identical to the optimized branch.
///
/// #1357: the harvest warm-starts REML from `best.fit.lambdas` (reproducing the
/// certified baseline cheaply), but on the flat (ρ, κ) Matérn valley that warm
/// start can slide the ρ search into a degenerate basin that collapses the smooth
/// onto its intercept (EDF → 1) even though `best` at the same geometry is
/// healthy — the double-penalty nullspace-shrinkage block of `best`'s λ sits near
/// the shrink-out corner, and the relaxed log-λ cap then lets it run away. When
/// the warm-started harvest collapses far below `best`'s certified EDF, refit the
/// same geometry from scratch (no λ seed, exactly how `best` was produced); the
/// scratch fit recovers the healthy baseline. This retry only fires on the
/// collapse pathology, so warm-starting's speed/uniformity is preserved for every
/// non-degenerate fallback.
fn fit_frozen_baseline_geometry(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    best: &FittedTermCollection,
    family: LikelihoodSpec,
    options: &FitOptions,
    baseline_score: f64,
    kappa_timing: Option<SpatialLengthScaleOptimizationTiming>,
) -> Result<FittedTermCollectionWithSpec, EstimationError> {
    let baseline = fit_term_collection_forspecwith_heuristic_lambdas(
        data,
        y,
        weights,
        offset,
        resolvedspec,
        best.fit.lambdas.as_slice(),
        family.clone(),
        options,
    )?;
    // #1357 collapse retry: if the warm-started harvest shed essentially all of
    // `best`'s certified effective DOF (a flat-valley collapse onto the
    // intercept), re-derive λ from scratch — `best` itself was fit from scratch
    // and is healthy, so the scratch harvest reproduces it.
    let best_edf = best.fit.inference.as_ref().map(|inf| inf.edf_total);
    let baseline_edf = baseline.fit.inference.as_ref().map(|inf| inf.edf_total);
    let baseline = match (best_edf, baseline_edf) {
        (Some(best_edf), Some(base_edf))
            if base_edf < SPATIAL_COLLAPSE_EDF_FLOOR
                && best_edf >= base_edf + SPATIAL_COLLAPSE_EDF_MARGIN =>
        {
            log::info!(
                "[spatial-kappa] warm-started frozen baseline collapsed (edf={base_edf:.3}) \
                 below the certified baseline (edf={best_edf:.3}); refitting from scratch",
            );
            fit_term_collection_forspec(data, y, weights, offset, resolvedspec, family, options)?
        }
        _ => baseline,
    };
    let mut fit = baseline.fit;
    fit.reml_score = baseline_score;
    Ok(FittedTermCollectionWithSpec {
        fit,
        design: baseline.design,
        resolvedspec: resolvedspec.clone(),
        adaptive_diagnostics: baseline.adaptive_diagnostics,
        kappa_timing,
    })
}

/// #2122: data-adaptive retreat geometry for a STALLED joint spatial solve.
///
/// This is the safety net that stops a non-converged joint κ/range solve from
/// silently shipping a mean-only surface. The frozen baseline
/// (`fit_frozen_baseline_geometry`) keeps the SPEC-DEFAULT length scale; on a
/// frame whose distance scale is far shorter than that default, the Matérn
/// kernel columns collapse into their polynomial nullspace and REML shrinks the
/// smooth onto its intercept (the fit succeeds but predicts the response mean).
///
/// This builds a geometry whose every plain spatial length-scale term (Matérn /
/// hybrid; NOT constant-curvature κ, NOT measure-jet dials) is re-centered at
/// the MIDPOINT of its data-derived ψ window `[ψ_lo, ψ_hi]` — i.e. the geometric
/// mean `√(r_min·r_max / (min_frac·max_mult))` of the admissible kernel-range
/// interval, the same data-scaled length scale a robust GP length-scale prior
/// centers on and the informative range thinplate/duchon effectively use.
/// Constant-curvature and measure-jet ψ coordinates carry non-log-scale
/// semantics (signed κ / geometry dials) and are left at their seed values, so
/// `apply_tospec` writes them back unchanged. λ is re-derived from scratch at the
/// new geometry (`best`'s λ belong to the spec-default range and would
/// mis-warm-start it).
///
/// The ψ window bounds are read back from the flat outer-optimizer seed/bound
/// vectors the joint solve already computed, so no distance pass is repeated.
/// Returns `Ok(None)` when no term is eligible for a length-scale override, so
/// the caller keeps the frozen baseline unchanged.
fn fit_data_adaptive_geometry(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    spatial_terms: &[usize],
    dims_per_term: &[usize],
    theta_seed: &Array1<f64>,
    theta_lower: &Array1<f64>,
    theta_upper: &Array1<f64>,
    rho_dim: usize,
    family: LikelihoodSpec,
    options: &FitOptions,
    baseline_score: f64,
    kappa_timing: Option<SpatialLengthScaleOptimizationTiming>,
) -> Result<Option<FittedTermCollectionWithSpec>, EstimationError> {
    let dims = dims_per_term.to_vec();
    let seed = SpatialLogKappaCoords::from_theta_tail_with_dims(theta_seed, rho_dim, dims.clone());
    let lower = SpatialLogKappaCoords::from_theta_tail_with_dims(theta_lower, rho_dim, dims.clone());
    let upper = SpatialLogKappaCoords::from_theta_tail_with_dims(theta_upper, rho_dim, dims.clone());

    let mut values = seed.as_array().clone();
    let mut cursor = 0usize;
    let mut any_override = false;
    for (slot, &term_idx) in spatial_terms.iter().enumerate() {
        let d = dims[slot];
        // Only plain spatial length-scale terms get a data-centered range.
        // Constant-curvature κ (sign-basin logic) and measure-jet dials carry
        // non-log-scale ψ coordinates and are left at their seed values.
        let is_length_scale_term = constant_curvature_term_spec(resolvedspec, term_idx).is_none()
            && measure_jet_term_spec(resolvedspec, term_idx).is_none();
        if is_length_scale_term {
            for off in 0..d {
                let lo = lower.as_array()[cursor + off];
                let hi = upper.as_array()[cursor + off];
                if lo.is_finite() && hi.is_finite() && lo < hi {
                    values[cursor + off] = 0.5 * (lo + hi);
                    any_override = true;
                }
            }
        }
        cursor += d;
    }
    if !any_override {
        return Ok(None);
    }
    let coords = SpatialLogKappaCoords::new_with_dims(values, dims);
    let adaptive_spec = coords.apply_tospec(resolvedspec, spatial_terms)?;
    let adaptive =
        fit_term_collection_forspec(data, y, weights, offset, &adaptive_spec, family, options)?;
    // Stamp the certified baseline REML score, exactly as the frozen-baseline
    // path does: this fit is the graceful-degradation target chosen because the
    // joint solve stalled, so the outer `require_successful_spatial_optimization_result`
    // gate (which compares against `fit_score(&best.fit)` = `baseline_score`) must
    // not read a same-geometry-class re-derivation drift as "the optimizer made
    // the score worse" and abort an otherwise-valid recovered fit.
    let mut fit = adaptive.fit;
    fit.reml_score = baseline_score;
    Ok(Some(FittedTermCollectionWithSpec {
        fit,
        design: adaptive.design,
        resolvedspec: adaptive_spec,
        adaptive_diagnostics: adaptive.adaptive_diagnostics,
        kappa_timing,
    }))
}

/// Coordinate kind for the exact joint spatial hyperparameter optimizer.
///
/// Anisotropic and isotropic spatial terms drive the *same* joint `[ρ, ψ]`
/// optimizer: identical outer-Hessian policy, identical
/// `ExternalJointHyperEvaluator` wiring, identical multistart problem, identical
/// convergence processing, and an identical `eval_full / eval_efs / eval_cost`
/// inner loop that routes ψ through `try_build_spatial_log_kappa_hyper_dirs`.
/// The only difference is the coordinate *kind*: the anisotropic path carries
/// one log-scale coordinate per axis per term (ψ_a) while the isotropic path
/// carries one log-κ coordinate per term. The kind selects diagnostic labels
/// only — the numerics are shared verbatim.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SpatialHyperKind {
    Anisotropic,
    Isotropic,
}

impl SpatialHyperKind {
    /// Stable diagnostic prefix used in every `log::*` line and as the
    /// `ExternalJointHyperEvaluator` / cost-only label root.
    fn label(self) -> &'static str {
        match self {
            SpatialHyperKind::Anisotropic => "spatial-aniso-joint",
            SpatialHyperKind::Isotropic => "spatial-iso-joint",
        }
    }

    /// Human-readable adjective for error strings ("anisotropic" / "isotropic").
    fn adjective(self) -> &'static str {
        match self {
            SpatialHyperKind::Anisotropic => "anisotropic",
            SpatialHyperKind::Isotropic => "isotropic",
        }
    }

    /// Name of the directional coordinate being optimized ("psi" / "kappa"),
    /// used only in hyper-direction construction error messages.
    fn coord_name(self) -> &'static str {
        match self {
            SpatialHyperKind::Anisotropic => "psi",
            SpatialHyperKind::Isotropic => "kappa",
        }
    }
}

/// Shared context for the exact joint spatial optimizer's closures. Holds the
/// realized-design cache and the joint REML evaluator, plus the coordinate
/// `kind` whose only effect is the diagnostic label routed into the cost-only
/// evaluation path. The `eval_full / eval_efs / eval_cost` methods are the
/// single source of truth for both anisotropic and isotropic spatial terms.
struct SpatialFrozenGlmInputs {
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    family: LikelihoodSpec,
}

/// True when the frozen-weight GLM ψ-tensor (#1111 / #1033 mechanism (c)) is a
/// faithful first-Fisher-step provider for this family.
///
/// The mechanism freezes the working weight `w = w(η_warm)` and working response
/// `z = z(η_warm)` once per outer ψ-sweep, so it is exact for ANY family whose
/// per-iteration PIRLS reduces to a Gaussian working model with a SINGLE
/// canonical Fisher weight at a FIXED dispersion — i.e. the one-parameter
/// exponential families Binomial, Poisson, Gamma, and Negative-Binomial (the
/// θ-fixed running-seed weight `W = μθ/(θ+μ)` is a clean per-row Fisher weight).
/// These are precisely the "Poisson/Binomial/etc" families the issue names.
///
/// Tweedie and Beta jointly estimate an extra dispersion parameter that moves
/// the working weight outside the frozen snapshot, so the frozen-W stand-in is
/// not faithful for them and they keep the exact per-trial PIRLS rebuild.
/// Gaussian-identity is served by the (exact, converged) `PsiGramTensor` lane,
/// and Royston-Parmar is the survival path, neither of which routes here.
fn frozen_glm_tensor_eligible_family(family: &LikelihoodSpec) -> bool {
    !family.is_gaussian_identity()
        && matches!(
            &family.response,
            ResponseFamily::Binomial
                | ResponseFamily::Poisson
                | ResponseFamily::Gamma
                | ResponseFamily::NegativeBinomial { .. }
        )
}

struct SpatialJointContext<'d> {
    data: ArrayView2<'d, f64>,
    rho_dim: usize,
    kind: SpatialHyperKind,
    cache: SingleBlockExactJointDesignCache<'d>,
    evaluator: gam_solve::estimate::ExternalJointHyperEvaluator<'d>,
    frozen_glm_inputs: Option<SpatialFrozenGlmInputs>,
    frozen_glm_psi_bounds: Option<(f64, f64)>,
    frozen_glm_tensor: Option<gam_solve::glm_sufficient_lane::FrozenWeightGramTensor>,
    frozen_glm_tensor_attempted: bool,
    /// #1033: memo of the frozen-W trial Fisher weights keyed on the warm β that
    /// produced them. `stage_frozen_glm_trial_statistics` runs on EVERY κ trial
    /// (every cost / gradient probe), and the only β-dependent quantity it needs
    /// is the current Fisher weight vector `W(η)` (η = Xβ + offset) for the
    /// drift check and the n-free gradient soundness gate. Computing `W` is an
    /// O(n·p) GEMV + O(n) family evaluation; β only changes when the inner solve
    /// re-converges (after an accepted outer step), so recomputing it on every
    /// same-β probe was a redundant per-trial n-touch. Cache `(β, W)` and reuse
    /// `W` whenever β is unchanged — the GEMV runs once per distinct β, i.e.
    /// O(outer steps), not O(trials). `None` until the first compute / when no
    /// frozen-W inputs are installed.
    frozen_glm_weight_memo: Option<(Array1<f64>, Array1<f64>)>,
}

#[derive(Clone, Copy, Debug, Default)]
struct NfreeSkipGateStatus {
    shape: bool,
    value: bool,
    gradient: bool,
    penalty: bool,
    revision: bool,
    second_order: bool,
}

impl NfreeSkipGateStatus {
    fn would_skip(self, require_gradient: bool) -> bool {
        self.shape
            && self.value
            && (!require_gradient || self.gradient)
            && self.penalty
            && self.revision
            && !self.second_order
    }
}

impl<'d> SpatialJointContext<'d> {
    fn nfree_skip_gate_status(
        &self,
        theta: &Array1<f64>,
        allow_second_order: bool,
        require_gradient: bool,
    ) -> NfreeSkipGateStatus {
        let shape = theta.len() == self.rho_dim + 1;
        let (value, gradient) = if shape {
            let psi = theta[self.rho_dim];
            (
                self.evaluator.psi_gram_tensor_covers(psi)
                    && self.evaluator.psi_gram_tensor_covers_skip(psi),
                !require_gradient || self.evaluator.psi_gram_tensor_covers_gradient(psi),
            )
        } else {
            (false, false)
        };
        NfreeSkipGateStatus {
            shape,
            value,
            gradient,
            penalty: self.evaluator.supports_nfree_penalty_rekey(),
            revision: self.evaluator.nfree_fast_path_revision().is_some(),
            second_order: allow_second_order,
        }
    }

    fn frozen_glm_working_state(
        &self,
        beta: &Array1<f64>,
    ) -> Result<Option<(Array1<f64>, Array1<f64>)>, EstimationError> {
        let Some(inputs) = self.frozen_glm_inputs.as_ref() else {
            return Ok(None);
        };
        if beta.len() != self.cache.design().design.ncols() {
            return Ok(None);
        }
        let mut eta = self.cache.design().design.matrixvectormultiply(beta);
        if eta.len() != inputs.offset.len() {
            crate::bail_invalid_estim!(
                "frozen GLM tensor warm-state row mismatch: eta={}, offset={}",
                eta.len(),
                inputs.offset.len()
            );
        }
        eta += &inputs.offset;
        let obs = evaluate_standard_familyobservations(
            inputs.family.clone(),
            None,
            None,
            None,
            &inputs.y,
            &inputs.weights,
            &eta,
        )?;
        let mut working_response = obs.eta.clone();
        for i in 0..working_response.len() {
            let wi = obs.fisherweight[i].max(1e-12);
            working_response[i] += obs.score[i] / wi;
        }
        Ok(Some((obs.fisherweight, working_response)))
    }

    /// #1033: the trial Fisher weight vector `W(η)` for `beta`, memoized on
    /// `beta`. `stage_frozen_glm_trial_statistics` consults `W` on EVERY κ trial
    /// (drift check + n-free gradient soundness gate) but `W` is a deterministic
    /// function of β (η = Xβ + offset), and β only changes when the inner solve
    /// re-converges — many cost / gradient probes share one β. Recompute the
    /// O(n·p) working state only when β differs from the memoized key; otherwise
    /// return the cached weights. Returns `None` exactly when
    /// `frozen_glm_working_state` does (no frozen-W inputs / β shape mismatch).
    fn frozen_glm_trial_weights(
        &mut self,
        beta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        if let Some((memo_beta, memo_w)) = self.frozen_glm_weight_memo.as_ref()
            && memo_beta.len() == beta.len()
            && memo_beta
                .iter()
                .zip(beta.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Ok(Some(memo_w.clone()));
        }
        match self.frozen_glm_working_state(beta)? {
            Some((current_w, _)) => {
                self.frozen_glm_weight_memo = Some((beta.clone(), current_w.clone()));
                Ok(Some(current_w))
            }
            None => Ok(None),
        }
    }

    fn ensure_frozen_glm_tensor(
        &mut self,
        theta: &Array1<f64>,
        warm_beta: Option<&Array1<f64>>,
    ) -> Result<(), EstimationError> {
        if self.frozen_glm_tensor.is_some() || self.frozen_glm_tensor_attempted {
            return Ok(());
        }
        let Some((psi_lo, psi_hi)) = self.frozen_glm_psi_bounds else {
            return Ok(());
        };
        if theta.len() != self.rho_dim + 1 {
            self.frozen_glm_tensor_attempted = true;
            return Ok(());
        }
        let Some(beta) = warm_beta else {
            return Ok(());
        };
        let Some((frozen_w, working_z)) = self.frozen_glm_working_state(beta)? else {
            self.frozen_glm_tensor_attempted = true;
            return Ok(());
        };
        let theta_probe_base = theta.clone();
        let rho_dim = self.rho_dim;
        // Build through the evaluator so the frozen-W Gram is assembled in the
        // SAME conditioned `x_fit` column frame the inner PIRLS solve uses
        // (the evaluator owns the ψ-invariant parametric conditioning). Disjoint
        // mutable borrows of `cache` (in the realizer) and `evaluator` (the
        // build host) — both fields of `self` — exactly as the Gaussian
        // `build_and_set_psi_gram_tensor` site does.
        let Self {
            cache, evaluator, ..
        } = self;
        let tensor = evaluator.build_frozen_glm_gram_tensor(
            |psi| {
                let mut theta_probe = theta_probe_base.clone();
                theta_probe[rho_dim] = psi;
                cache.ensure_theta(&theta_probe)?;
                Ok(cache.design().design.clone())
            },
            frozen_w.view(),
            working_z.view(),
            psi_lo,
            psi_hi,
        );
        self.cache
            .ensure_theta(theta)
            .map_err(EstimationError::InvalidInput)?;
        self.frozen_glm_tensor_attempted = true;
        if let Some(tensor) = tensor {
            self.frozen_glm_tensor = Some(tensor);
            log::info!(
                "[STAGE] {} certified frozen-W GLM ψ tensor over [{psi_lo:.3}, {psi_hi:.3}]",
                self.kind.label(),
            );
        } else {
            log::info!(
                "[STAGE] {} frozen-W GLM ψ tensor did not certify over [{psi_lo:.3}, {psi_hi:.3}]",
                self.kind.label(),
            );
        }
        Ok(())
    }

    fn stage_frozen_glm_trial_statistics(
        &mut self,
        theta: &Array1<f64>,
        warm_beta: Option<&Array1<f64>>,
        allow_gradient: bool,
    ) -> Result<(), EstimationError> {
        let kind = self.kind;
        let mut staged_gram: Option<Array2<f64>> = None;
        let mut staged_deriv: Option<(Array2<f64>, Array1<f64>)> = None;
        if theta.len() == self.rho_dim + 1 {
            let psi = theta[self.rho_dim];
            // Compute the β-memoized trial Fisher weights up front (mutable
            // self borrow) so the immutable `self.frozen_glm_tensor` borrow
            // below does not alias it. `frozen_glm_trial_weights` recomputes the
            // O(n·p) working state only on a β change, so a same-β probe pays
            // nothing here (#1033). Only proceed when a tensor is installed and
            // covers this ψ — otherwise skip the weight compute entirely.
            let tensor_covers = self
                .frozen_glm_tensor
                .as_ref()
                .is_some_and(|t| t.contains(psi));
            let current_w = if tensor_covers {
                match warm_beta {
                    Some(beta) => self.frozen_glm_trial_weights(beta)?,
                    None => None,
                }
            } else {
                None
            };
            if let (Some(tensor), Some(current_w)) =
                (self.frozen_glm_tensor.as_ref(), current_w.as_ref())
            {
                const FROZEN_GLM_WEIGHT_DRIFT_RTOL: f64 = 1e-3;
                if tensor.weight_drift_within(current_w.view(), FROZEN_GLM_WEIGHT_DRIFT_RTOL) {
                    staged_gram = Some(tensor.gram_at(psi));
                    log::debug!(
                        "[STAGE] {} trial at psi={psi:.6}: serving frozen-W GLM \
                         first-Fisher-step XᵀWX n-free (weight drift within tol)",
                        kind.label(),
                    );
                }
                if allow_gradient
                    && tensor.contains_for_gradient(psi)
                    && let Some((dgram_dpsi, drhs_dpsi)) =
                        tensor.gradient_pair_if_sound(psi, current_w.view())
                {
                    staged_deriv = Some((dgram_dpsi, drhs_dpsi));
                    log::debug!(
                        "[STAGE] {} trial at psi={psi:.6}: serving frozen-W GLM \
                         ψ-gradient (∂G/∂ψ, ∂b/∂ψ) n-free (gradient weight drift within \
                         tight tol); B_j stays exact",
                        kind.label(),
                    );
                }
            }
        }
        self.evaluator.stage_glm_first_step_gram(staged_gram);
        self.evaluator.stage_glm_psi_gram_deriv(staged_deriv);
        Ok(())
    }

    /// Full evaluation on the current realized design + hyper_dirs.
    fn eval_full(
        &mut self,
        theta: &Array1<f64>,
        order: gam_solve::rho_optimizer::OuterEvalOrder,
        analytic_outer_hessian_available: bool,
    ) -> Result<
        (
            f64,
            Array1<f64>,
            gam_problem::HessianResult,
        ),
        EstimationError,
    > {
        use gam_solve::rho_optimizer::OuterEvalOrder;
        let allow_second_order = matches!(order, OuterEvalOrder::ValueGradientHessian)
            && analytic_outer_hessian_available;
        if let Some(eval) = self.cache.memoized_eval(theta) {
            let cached_satisfies_order = !allow_second_order || eval.2.is_analytic();
            if cached_satisfies_order {
                return Ok(eval);
            }
        }
        let kind = self.kind;
        // #1033: the per-trial n×k design re-realization (`ensure_theta` →
        // `apply_log_kappa`) plus the downstream n-row reconditioning
        // (`reset_surface`) are the LAST n-passes in the certified κ loop. They
        // are redundant on the Gaussian-identity certified path: the inner
        // Gaussian PLS reads its `XᵀWX(ψ)/XᵀW(y−offset)(ψ)` entirely from the
        // ψ-keyed `GaussianFixedCache` the certified tensor installs (zero row
        // access), and the ψ-gradient HyperCoord is served from the k-space
        // `(∂G/∂ψ, ∂b/∂ψ)` tensor derivatives — never the n×k ∂X/∂ψ slab. So when
        //   (a) this is the single design-moving ψ coordinate (`rho_dim + 1`),
        //   (b) the certified ψ-Gram tensor covers ψ for BOTH the value lane
        //       (`psi_gram_tensor_covers`) AND the gradient window
        //       (`psi_gram_tensor_covers_gradient`) — so neither channel reads
        //       the realized rows,
        //   (c) this eval is gradient-only (`!allow_second_order`) — the exact
        //       outer-Hessian `B_j` path DOES read the slab, so a Hessian trial
        //       must keep a faithful (freshly realized) design, and
        //   (d) the evaluator has a pinned canonical slow-path revision — i.e.
        //       a prior slow-path eval already built a faithful reference surface,
        //       which `prepare_eval_state` will reuse while re-installing the
        //       ψ-keyed cache,
        // we SKIP `ensure_theta`. The realizer revision then does not advance, so
        // `prepare_eval_state` takes its design-revision fast path by receiving
        // that pinned revision back: it skips `reset_surface` + the n×k
        // `apply_to_design`, keeps the reference surface, and re-keys the
        // `GaussianFixedCache` to this ψ. The hyper_dirs built below are a pure
        // function of (data, frozen spec, column layout) — ψ-invariant — so they
        // are bit-identical whether or not the design was re-realized, and the
        // tensor branch never reads their n×k slab anyway. Net: criterion +
        // gradient + inner solve come from k-space statistics only, with no
        // per-trial O(n·k) pass.
        //
        // When ANY gate clause fails (non-Gaussian, off-window, off the gradient
        // sub-window, a Hessian eval, or no pinned canonical surface yet) we
        // realize the design as before so the slow path rebuilds a faithful
        // surface — the existing exact lane runs unchanged.
        let nfree_fast_path_revision = self.evaluator.nfree_fast_path_revision();
        let skip_design_realization = !allow_second_order && theta.len() == self.rho_dim + 1 && {
            let psi = theta[self.rho_dim];
            self.evaluator.psi_gram_tensor_covers(psi)
                    // #1033 gradient coverage: the skip serves the ψ-gradient n-free
                    // only where the analytic Chebyshev derivative is CERTIFIED.
                    // The kappa sufficient-statistic outer loop is routed here only
                    // when the certified gradient window spans the entire optimizer
                    // bounds, so a measured trial cannot pay an edge streamed
                    // ∂X/∂ψ pass after the initial priming eval.
                    && self.evaluator.psi_gram_tensor_covers_gradient(psi)
                    // #1264 (RESTORED) reduced-basis-rotation soundness precondition.
                    // The Gaussian inner penalized solve `(QsᵀGQs+S)β=b` runs in the
                    // CONDITIONED reduced basis. On the near-singular production
                    // Duchon Gram (κ(G)≈9.5e14) that basis ROTATES with ψ, and the
                    // skip installs the Chebyshev-interpolated `gram_at(ψ)` (≤1e-10
                    // vs streamed exact). When the trial-ψ basis differs from the
                    // reference surface's, the κ-amplified round-off moves β̂ by
                    // ~1.7e-5 — 17× the issue's 1e-6 bar — EVEN at a ψ the n-free
                    // VALUE window admits (cluster: β̂rel=1.749e-5 at ψ=2.803). The
                    // "stale-penalty-not-stale-basis" theory that dropped this gate
                    // was empirically refuted. So the skip is β̂-sound ONLY where the
                    // gauge-invariant range projector is unchanged vs the pinning ψ:
                    // `reduced_basis_equal(psi_ref, psi)`. Value coverage is NOT
                    // sufficient. This forces the exact O(n) `reset_surface` fallback
                    // across a basis rotation — correctness over n-independence
                    // (#1033 is frontier-blocked on rotating Duchon geometry).
                    && self.evaluator.psi_gram_tensor_covers_skip(psi)
                    // #1033 penalty lane: ψ moves S(ψ) too, and the skip leaves
                    // `reset_surface` un-run; only skip when the penalty can be
                    // rebuilt EXACTLY and n-free on the fast path, else the inner
                    // solve would pair XᵀWX(ψ_new) with the stale S(ψ_old).
                    && self.evaluator.supports_nfree_penalty_rekey()
                    && nfree_fast_path_revision.is_some()
        };
        // #1868: the #1033 n-free design-realization skip is armed above. A prior
        // debug override (`TEMP-SKIPOFF-1122`) hard-forced `skip_design_realization`
        // to `false` here to test whether the n-free ψ-Gram Chebyshev interpolant
        // was the source of the #1122 H-side FD-vs-analytic gap. That override was
        // never removed, so every in-window κ `eval_full` trial fell through to the
        // O(n) `ensure_theta` → `apply_log_kappa` + `reset_surface` lane — the O(n)
        // per-callback regression #1868 reports. The skip is already gated on
        // `!allow_second_order`, so it never fires on the H (Hessian) trials the
        // #1122 diagnostic was probing; the override only ever suppressed the
        // n-free gradient/value lane. Removing it routes the gradient eval through
        // the k-space `GaussianFixedCache` + ψ-derivative tensor as intended.
        if skip_design_realization {
            log::debug!(
                "[STAGE] {} eval_full at psi={:.6}: skipping n×k design re-realization \
                 + reconditioning — criterion/gradient/inner-solve served n-free from \
                 the certified ψ-gram tensor (GaussianFixedCache + k-space ψ-derivatives)",
                kind.label(),
                theta[self.rho_dim],
            );
        } else {
            self.cache
                .ensure_theta(theta)
                .map_err(EstimationError::InvalidInput)?;
        }
        let warm_beta = self.evaluator.current_beta();
        self.ensure_frozen_glm_tensor(theta, warm_beta.as_ref())?;
        // #1033 / #1111: stage the GLM frozen-W first-step Gram and conditioned
        // ψ-gradient whenever the certified frozen-weight tensor covers this
        // trial's ψ. The provider applies its drift guards, so misses clear the
        // staged slots and the exact streamed path runs.
        //
        // Stage through a shared helper because cost-only line-search probes use
        // the same first-Fisher-step Gram; they simply pass `allow_gradient=false`.
        self.stage_frozen_glm_trial_statistics(theta, warm_beta.as_ref(), !allow_second_order)?;
        // #1033: on the certified Gaussian skip path the value and ψ-gradient
        // are both served by k-space tensor statistics, so the row-wise X_ψ slab
        // is dead. Build only the exact n-free S_ψ components from frozen
        // geometry and attach a zero-storage design derivative placeholder.
        // Edge-gradient/Hessian/non-certified trials keep the exact row-wise
        // builder, because those lanes genuinely consume X_ψ.
        let hyper_dirs = if skip_design_realization {
            self.cache.nfree_tensor_gradient_hyper_dirs(theta)?
        } else {
            self.cache.hyper_dirs_for_current_design(self.data, kind)?
        };

        let design_revision = if skip_design_realization {
            nfree_fast_path_revision
        } else {
            Some(self.cache.design_revision())
        };
        // #1033 penalty lane: stage the EXACT n-free `S(ψ)` for this trial so the
        // evaluator's design-revision fast path can re-key the kept reference
        // surface without `reset_surface`. Built from the FROZEN basis geometry
        // (centers + identifiability transform + operator collocation points) at
        // the trial length-scale — no data rows — so it is valid even on the
        // design-realization skip path (where the design was not re-realized). The
        // caller (holding `cache`) computes it and hands the owned result to the
        // evaluator, sidestepping a `&mut cache` borrow alias. On the slow path
        // the evaluator ignores + clears the staged value (it rebuilds S from the
        // realized design). A build error here clears the stage; if the skip
        // already fired (fast path), the evaluator then hard-errors rather than
        // pairing a stale S — the safe outcome, since a rebuild from frozen
        // geometry should never fail in practice.
        if self.evaluator.supports_nfree_penalty_rekey() {
            match self.cache.canonical_penalties_at(theta) {
                Ok(penalty) => self.evaluator.stage_fast_path_penalty(Some(penalty)),
                Err(e) => {
                    log::warn!(
                        "[STAGE] {} eval_full at psi={:.6}: exact n-free S(ψ) rebuild failed \
                         ({e}); clearing stage (eval falls to slow path)",
                        kind.label(),
                        theta[self.rho_dim],
                    );
                    self.evaluator.stage_fast_path_penalty(None);
                }
            }
        }
        // Warm-start PIRLS from the previous outer step's converged β. This is
        // especially impactful for GLM families (Poisson, NB, Binomial) that
        // cannot use the Gaussian Gram tensor n-free shortcut: without the warm
        // β every outer step cold-solves a full PIRLS from β=0, paying the full
        // O(n·p²) cost × PIRLS-iters × outer-iters budget. With the warm β the
        // inner solve typically converges in 1-2 Newton steps instead of 4-8.
        let eval = evaluate_joint_reml_outer_eval_at_theta(
            &mut self.evaluator,
            self.cache.design(),
            theta,
            self.rho_dim,
            hyper_dirs,
            warm_beta.as_ref().map(|b: &Array1<f64>| b.view()),
            if allow_second_order {
                order
            } else {
                OuterEvalOrder::ValueAndGradient
            },
            design_revision,
        );
        if let Ok(ref value) = eval {
            self.cache.store_eval_at(theta, value.clone());
        }
        eval
    }

    fn eval_efs(
        &mut self,
        theta: &Array1<f64>,
    ) -> Result<gam_problem::EfsEval, EstimationError> {
        self.cache
            .ensure_theta(theta)
            .map_err(EstimationError::InvalidInput)?;
        let kind = self.kind;
        let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
            self.data,
            self.cache.spec(),
            self.cache.design(),
            &self.cache.spatial_terms,
        )?
        .ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "failed to build {} hyper_dirs for exact-joint EFS",
                kind.adjective(),
            ))
        })?;
        let design_revision = Some(self.cache.design_revision());
        let warm_beta = self.evaluator.current_beta();
        evaluate_joint_reml_efs_at_theta(
            &mut self.evaluator,
            self.cache.design(),
            theta,
            self.rho_dim,
            hyper_dirs,
            warm_beta.as_ref().map(|b: &Array1<f64>| b.view()),
            design_revision,
        )
    }

    /// Cost-only evaluation. BFGS line-search probes route through the
    /// evaluator's true value-only path so they neither construct
    /// `try_build_spatial_log_kappa_hyper_dirs` nor assemble a gradient that
    /// the line search will discard. Split-borrow on `self.cache` +
    /// `self.evaluator` matches the pattern already used by `eval_full`.
    fn eval_cost(&mut self, theta: &Array1<f64>) -> f64 {
        if let Some(cost) = self.cache.memoized_cost(theta) {
            return cost;
        }
        // #1029: a BFGS line-search VALUE probe. It converges the inner PIRLS to
        // the SAME tolerance the accepted-point full eval uses (NOT a capped
        // surrogate — a cap returns ∞ for a feasible point and re-imports the
        // #787/#808 outer stall), so probe and incumbent values live in ONE
        // refinement regime (measure-consistent Armijo). It is cheaper only
        // because it skips the gradient / hyper-dir assembly. Time the inner
        // cost-only solve and report it alongside the trial-θ distance from the
        // last evaluated point so this convergence-critical regression class is
        // visible in the STAGE trace (the spatial REML lane has no PROGRESS-
        // EXTENDED refine multiplier — that knob is SAE-only — so there is no
        // extended polish to strip from a probe here).
        //
        // Capture the previous evaluated θ BEFORE `ensure_theta` overwrites it,
        // so the logged distance reflects the backtracking step rather than 0.
        let probe_start = std::time::Instant::now();
        let psi_distance = self
            .cache
            .current_theta
            .as_ref()
            .filter(|reference| reference.len() == theta.len())
            .map(|reference| {
                reference
                    .iter()
                    .zip(theta.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt()
            })
            .unwrap_or(f64::NAN);
        // #1033: a VALUE-only line-search probe needs only the certified ψ-Gram
        // tensor's value lane (`XᵀWX(ψ)/XᵀW(y−offset)(ψ)`), which the inner
        // Gaussian PLS reads n-free from the ψ-keyed `GaussianFixedCache`. So when
        // the single design-moving ψ is covered for the VALUE lane and the
        // evaluator has a pinned canonical slow-path revision, skip the n×k
        // design re-realization: `evaluate_cost_only` receives that pinned
        // revision, takes its `prepare_eval_state_cost_only` fast path (which
        // skips `reset_surface` + the n×k `apply_to_design` and re-keys the cache
        // to this probe's ψ), and the probe cost comes from k-space statistics
        // only. Line-search probes are the bulk of the κ-loop per-trial work, so
        // this is the dominant n-flat lever. Any miss (non-Gaussian, off-window,
        // missing penalty re-key support, or no pinned surface yet) realizes the
        // design and runs the exact streamed probe unchanged.
        let nfree_fast_path_revision = self.evaluator.nfree_fast_path_revision();
        let skip_value_realization = theta.len() == self.rho_dim + 1 && {
            let psi = theta[self.rho_dim];
            self.evaluator.psi_gram_tensor_covers(psi)
                    // #1868: a VALUE-only line-search probe does NOT need the
                    // #1264 `reduced_basis_equal` (`covers_skip`) soundness gate the
                    // ACCEPTED gradient eval (`eval_full`, still gated) requires. That
                    // gate exists because the design-realization skip freezes the
                    // conditioned reduced basis at the pinning ψ, and on the near-
                    // singular Duchon Gram (κ(G)≈9.5e14) a ψ-rotation makes the
                    // frozen basis interpolate β̂ with a κ-amplified round-off of
                    // β̂rel≈1.7e-5 — which matters for the RETURNED coefficients/
                    // gradient. A cost probe returns only the scalar REML criterion
                    // for the line search, and that criterion is STATIONARY in β̂ at
                    // the inner minimizer (envelope theorem): a β̂ perturbation δβ
                    // moves the data-fit+penalty term by O(δβ²) and leaves the
                    // `log|H|` term (built from the EXACT tensor Gram G(ψ), not β̂)
                    // untouched, so the RELATIVE cost error is ~δβ² ≈ 3e-10 — orders
                    // below the line search's 1e-5 Armijo tolerance. So the probe
                    // cannot be mis-ranked, and the converged κ/β̂ (pinned by the
                    // covers_skip-gated `eval_full` at accepted iterates) is
                    // unchanged. Gating the probe on `covers_skip` instead forced the
                    // O(n) `reset_surface` lane for every line-search step that
                    // overshoots the (n-drifting) reduced-basis-stable band — the
                    // #1868 per-callback reset climb: the band's rotation rate dP/dψ
                    // grows with n (sample-std standardization), so more probes fall
                    // just past PSI_GRAM_SKIP_PROJ_ATOL as n grows, defeating the
                    // n-independence the tensor lane was built for. The evaluator's
                    // own value-probe fast path (`prepare_eval_state_cost_only`) is
                    // gated on VALUE coverage exactly for this reason; aligning the
                    // driver here lets the probe cost come from the n-free k-space
                    // Gram/penalty statistics across the rotation.
                    //
                    // #1033 penalty lane: the value-probe fast path also skips
                    // `reset_surface`, so the probe must be able to re-key S(ψ)
                    // EXACTLY and n-free; otherwise its cost would use the stale
                    // S(ψ_old) and mis-rank the line search.
                    && self.evaluator.supports_nfree_penalty_rekey()
                    && nfree_fast_path_revision.is_some()
        };
        if theta.len() == self.rho_dim + 1
            && self.evaluator.has_psi_gram_tensor()
            && !self.evaluator.psi_gram_tensor_covers(theta[self.rho_dim])
        {
            self.cache.store_cost_at(theta, f64::INFINITY);
            return f64::INFINITY;
        }
        if !skip_value_realization && self.cache.ensure_theta(theta).is_err() {
            return f64::INFINITY;
        }
        // #1033 penalty lane: stage the EXACT n-free `S(ψ)` for this probe's ψ so
        // the cost-only fast path re-keys the kept surface without `reset_surface`
        // (built from frozen geometry — valid even when the design was not
        // re-realized). The slow path clears it. A rebuild failure clears the
        // stage; the evaluator then takes the slow path or hard-errors (safe).
        if self.evaluator.supports_nfree_penalty_rekey() {
            match self.cache.canonical_penalties_at(theta) {
                Ok(penalty) => self.evaluator.stage_fast_path_penalty(Some(penalty)),
                Err(_) => self.evaluator.stage_fast_path_penalty(None),
            }
        }
        let warm_beta = self.evaluator.current_beta();
        if let Err(err) = self.ensure_frozen_glm_tensor(theta, warm_beta.as_ref()) {
            log::warn!(
                "[STAGE] {} value-probe at psi={:.6}: frozen-W GLM tensor setup failed ({err}); \
                 falling back to exact streamed Gram",
                self.kind.label(),
                if theta.len() > self.rho_dim {
                    theta[self.rho_dim]
                } else {
                    f64::NAN
                },
            );
            self.evaluator.stage_glm_first_step_gram(None);
            self.evaluator.stage_glm_psi_gram_deriv(None);
        } else if let Err(err) =
            self.stage_frozen_glm_trial_statistics(theta, warm_beta.as_ref(), false)
        {
            log::warn!(
                "[STAGE] {} value-probe at psi={:.6}: frozen-W GLM staging failed ({err}); \
                 falling back to exact streamed Gram",
                self.kind.label(),
                if theta.len() > self.rho_dim {
                    theta[self.rho_dim]
                } else {
                    f64::NAN
                },
            );
            self.evaluator.stage_glm_first_step_gram(None);
            self.evaluator.stage_glm_psi_gram_deriv(None);
        }
        let design_revision = if skip_value_realization {
            nfree_fast_path_revision
        } else {
            Some(self.cache.design_revision())
        };
        let cost_label = self.kind.label();
        let result = {
            let design = self.cache.design();
            self.evaluator.evaluate_cost_only(
                &design.design,
                &design.penalties,
                &design.nullspace_dims,
                design.linear_constraints.clone(),
                theta,
                self.rho_dim,
                warm_beta.as_ref().map(|b: &Array1<f64>| b.view()),
                cost_label,
                design_revision,
            )
        };
        match result {
            Ok(cost) => {
                log::debug!(
                    "[STAGE] {cost_label} value-probe (order=Value): elapsed={:.3}s \
                     cost={cost:.6e} trial_theta_distance={psi_distance:.3e}",
                    probe_start.elapsed().as_secs_f64(),
                );
                self.cache.store_cost_at(theta, cost);
                cost
            }
            Err(_) => f64::INFINITY,
        }
    }

    fn reset(&mut self) {
        self.cache.current_theta = None;
        self.cache.last_eval_theta = None;
        self.cache.last_cost = None;
        self.cache.last_eval = None;
    }
}

/// Exact joint `[ρ, ψ]` optimization for spatial terms using analytic
/// derivatives through the unified REML evaluator. This is the single shared
/// engine for both the anisotropic and isotropic coordinate kinds (selected by
/// `kind`).
///
/// At each outer iteration, the frozen term topology is reused and only the
/// spatial realized blocks affected by the current ψ are refreshed before the
/// unified evaluator returns cost + gradient + Hessian for the full
/// θ = [ρ, ψ] vector. The ψ derivatives flow through:
///
///   `AnisoBasisPsiDerivatives` / `SpatialPsiDerivative` → `DirectionalHyperParam`
///     → `build_tau_unified_objects` → `HyperCoord` ext_coords → unified evaluator
///
/// This gives Newton/BFGS quadratic convergence on the length-scale /
/// anisotropy parameters while jointly optimizing the smoothing parameters.
///
/// The ψ coordinates are parameterized as unconstrained log-scales. For the
/// anisotropic kind the decomposition into isotropic scale (ψ̄ = mean(ψ_a)) and
/// anisotropy (η_a = ψ_a − ψ̄, with Ση_a = 0) happens only on writeback via
/// `SpatialLogKappaCoords::apply_tospec`; the all-ones direction in ψ-space is
/// NOT a gauge direction — it controls the identifiable isotropic scale
/// κ = exp(ψ̄). The isotropic kind carries one log-κ coordinate per term. In
/// neither case is a sum-to-zero constraint enforced during optimization.
/// Outcome of the joint spatial hyperparameter `(ρ, ψ/κ)` optimization.
///
/// The joint κ optimizer refines an *already-valid* frozen baseline geometry
/// (the REML-seeded length scales in `best`); it is therefore best-effort. A run
/// that does not certify a stationary point must degrade to the baseline rather
/// than abort the parent fit (#1126), so this enum lets the caller distinguish a
/// usable iterate from a non-convergence that should fall back to the baseline.
/// Genuine numerical blowups (a non-finite terminal cost) still surface as
/// `Err` from [`run_exact_joint_spatial_optimization`] and never reach here.
enum SpatialJointOutcome {
    /// The optimizer produced a usable iterate: it either converged to a
    /// stationary point or its terminal iterate cleared the mgcv-style
    /// relative-to-cost REML acceptance gate. Carries `(θ*, final_value)`.
    Optimized {
        theta_star: Array1<f64>,
        final_value: f64,
    },
    /// The optimizer ran to a finite terminal cost but neither converged nor
    /// cleared the relative-to-cost gate. The caller keeps the frozen baseline
    /// geometry; the fields are diagnostics only.
    NonConverged {
        iterations: usize,
        final_value: f64,
        final_grad_norm: Option<f64>,
    },
}

fn kphase_log_norms(theta: &Array1<f64>, rho_dim: usize) -> (f64, f64) {
    let theta_norm = theta.iter().map(|v| v * v).sum::<f64>().sqrt();
    let log_kappa_norm = theta
        .iter()
        .skip(rho_dim)
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    (theta_norm, log_kappa_norm)
}

fn run_exact_joint_spatial_optimization(
    kind: SpatialHyperKind,
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    baseline_design: &TermCollectionDesign,
    family: LikelihoodSpec,
    options: &FitOptions,
    spatial_terms: &[usize],
    dims_per_term: &[usize],
    theta0: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    rho_dim: usize,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<(SpatialJointOutcome, SpatialLengthScaleOptimizationTiming), EstimationError> {
    let label = kind.label();
    // Use bounds and design metadata for validation.
    assert!(
        lower.len() == theta0.len() && upper.len() == theta0.len(),
        "spatial hyperparameter bounds must match theta length: lower_len={}, upper_len={}, theta_len={}",
        lower.len(),
        upper.len(),
        theta0.len()
    );
    assert!(
        baseline_design.smooth.terms.len() >= spatial_terms.len(),
        "baseline design must have at least one smooth term per spatial term: baseline_terms={}, spatial_terms={}",
        baseline_design.smooth.terms.len(),
        spatial_terms.len()
    );
    use gam_solve::rho_optimizer::OuterEvalOrder;
    use gam_problem::{DeclaredHessianForm, Derivative, OuterEval};

    let theta_dim = theta0.len();
    // Directional-coordinate dimension: psi-per-axis (anisotropic) or
    // kappa-per-term (isotropic). The numerics below are identical either way.
    let coord_dim = theta_dim - rho_dim;
    // Capability is declared solely from derivative coverage, not from
    // problem size. The unified REML evaluator now exposes exact matrix-free
    // outer Hessian operators for the costly third/fourth-derivative
    // contractions used by spatial ψ coordinates; its internal
    // `(n, p, K)` work model chooses `HessianResult::Operator` at large-scale
    // scale and the dense analytic matrix only below that crossover. Keeping
    // `Derivative::Analytic` here preserves ARC / trust-region-CG second-order
    // optimization for `n > 50_000` and `coord_dim > 30` instead of forcing the
    // obsolete HybridEFS compatibility path.
    let analytic_outer_hessian_available =
        exact_joint_spatial_outer_hessian_available(&family, baseline_design);
    if !analytic_outer_hessian_available {
        log::info!(
            "[{label}] analytic outer Hessian unavailable for family/design; routing without second-order geometry (coord_dim={coord_dim})"
        );
    }
    // Cost-aware second-order routing, mirroring the n-block path's
    // work-budget policy: past the pair budget gradient-only quasi-Newton
    // converges to the same optimum strictly cheaper per eval; below it,
    // exact second-order keeps the ARC/TR-CG geometry. The budget's
    // derivation is owned by `EXACT_JOINT_SECOND_ORDER_THETA_CAP`.
    let mut prefer_gradient_only = theta_dim > EXACT_JOINT_SECOND_ORDER_THETA_CAP;
    if prefer_gradient_only {
        log::info!(
            "[{label}] joint θ-dim {theta_dim} exceeds the exact pair-Hessian budget \
             ({EXACT_JOINT_SECOND_ORDER_THETA_CAP}); routing gradient-only quasi-Newton"
        );
    }
    // #1033: set when the n-free Gaussian ψ-lane arms below. It must SUPPRESS the
    // declared analytic outer Hessian (force `Unavailable`), not merely prefer
    // gradient-only: the planner keeps the second-order ARC solver whenever an
    // analytic Hessian is declared `Either`, even under `prefer_gradient_only`
    // (see `plan_prefer_gradient_only_does_not_hide_analytic_hessian`). A
    // `ValueGradientHessian` eval forces the O(n) design re-realization because
    // the outer Hessian curvature slab `B_j` is irreducibly n-dependent, so only
    // routing to a gradient-only solver (BFGS) keeps every in-window κ-trial on
    // the n-free `ValueAndGradient` skip.
    let mut suppress_outer_hessian_for_nfree = false;

    log::trace!(
        "[{}] starting analytic optimization: rho_dim={}, coord_dim={}, dims_per_term={:?}",
        label,
        rho_dim,
        coord_dim,
        dims_per_term,
    );

    let mut ctx = SpatialJointContext {
        data,
        rho_dim,
        kind,
        cache: SingleBlockExactJointDesignCache::new(
            data,
            resolvedspec.clone(),
            baseline_design.clone(),
            spatial_terms.to_vec(),
            rho_dim,
            dims_per_term.to_vec(),
        )
        .map_err(EstimationError::InvalidInput)?,
        evaluator: gam_solve::estimate::ExternalJointHyperEvaluator::new(
            y,
            weights,
            &baseline_design.design,
            offset,
            &baseline_design.penalties,
            &external_opts_for_design(&family, baseline_design, options),
            label,
        )?,
        frozen_glm_inputs: if coord_dim == 1 && frozen_glm_tensor_eligible_family(&family) {
            Some(SpatialFrozenGlmInputs {
                y: y.to_owned(),
                weights: weights.to_owned(),
                offset: offset.to_owned(),
                family: family.clone(),
            })
        } else {
            None
        },
        frozen_glm_psi_bounds: if coord_dim == 1 && frozen_glm_tensor_eligible_family(&family) {
            Some((lower[rho_dim], upper[rho_dim]))
        } else {
            None
        },
        frozen_glm_tensor: None,
        frozen_glm_tensor_attempted: false,
        frozen_glm_weight_memo: None,
    };

    // #1033b: single isotropic design-moving coordinate on a Gaussian-identity
    // fit — build the certified Chebyshev-in-ψ Gram tensor ONCE over the
    // optimizer's ψ window and hand it to the evaluator. Every in-window trial
    // then receives its Gaussian sufficient statistics (XᵀWX(ψ), XᵀW(y−offset),
    // (y−offset)ᵀW(y−offset)) assembled n-free instead of paying the per-trial
    // O(n·p²) Gram re-stream after the design rebuild. The realizer closure
    // returns the RAW realized design; the evaluator threads it through its
    // own (fixed, ψ-invariant) parametric column conditioning so the tensor
    // lives in the same frame as the streamed Gram. Certification failure,
    // off-window trials, or any other ineligibility silently keep the exact
    // streamed path (same numbers, the tensor is certified to
    // PSI_GRAM_SPOT_RTOL against the exact rebuild).
    // #1033 (rank-stable κ-floor): set to the lowest ψ at which the certified
    // tensor's conditioned Gram holds maximal numerical rank. Below it the
    // reduced basis collapses/rotates and the design-realization skip is SOUNDLY
    // refused (→ O(n) reset_surface); the κ window floor `ln(2/r_max)` lands
    // inside that degenerate sliver and DRIFTS with n through the sample-std
    // standardization, so n=2000's line search re-enters the slow lane while
    // n=1000's does not. Lifting the optimizer's lower bound to this n-FREE
    // (k-space) floor keeps every in-window trial on the fast path for all n,
    // and only excludes over-smoothed length scales the `2/r_max` geometry floor
    // already meant to exclude (the κ-optimum lives well above it).
    let mut psi_rank_stable_floor: Option<f64> = None;
    // #1033 (rank-stable κ-ceiling): symmetric twin of the floor. The conditioned
    // Gram is rank-deficient at the HIGH window edge too (the longest-frequency
    // radial mode goes collinear), so a line-search overshoot above the maximal-
    // rank band soundly refuses the design-realization skip → O(n) reset_surface,
    // and the deficient pinning ψ it records makes the NEXT in-band trial reset a
    // second time. Clamping the optimizer's UPPER bound to this n-free k-space
    // ceiling keeps every trial inside the band. The κ-optimum lives well inside
    // it, so the clamp only excludes over-fit (too-short) length scales.
    let mut psi_rank_stable_ceiling: Option<f64> = None;
    let nfree_penalty_capable = coord_dim == 1
        && family.is_gaussian_identity()
        && ctx.cache.supports_nfree_penalty_rekey();
    if nfree_penalty_capable {
        let psi_lo = lower[rho_dim];
        let psi_hi = upper[rho_dim];
        let z = Array1::from_iter(y.iter().zip(offset.iter()).map(|(yi, oi)| yi - oi));
        let theta_probe_base = theta0.clone();
        // Disjoint mutable borrows of `cache` (in the realizer) and
        // `evaluator` (the build target) — both fields of `ctx`.
        let SpatialJointContext {
            cache, evaluator, ..
        } = &mut ctx;
        let attached = evaluator.build_and_set_psi_gram_tensor(
            |psi| {
                let mut theta_probe = theta_probe_base.clone();
                theta_probe[rho_dim] = psi;
                cache.ensure_theta(&theta_probe)?;
                Ok(cache.design().design.clone())
            },
            weights,
            z.view(),
            psi_lo,
            psi_hi,
        );
        if attached {
            log::info!(
                "[{label}] certified ψ-gram tensor over [{psi_lo:.3}, {psi_hi:.3}]: \
                 in-window trials assemble Gaussian sufficient statistics n-free"
            );
            // #1033: read the n-free rank-stable κ-floor off the k-space tensor.
            // Only lift INTO the window (never below psi_lo, never above the seed
            // ψ — the seed is the geometric-mean midpoint and is well clear of the
            // degenerate band), so the optimizer never starts outside its bounds.
            let psi_anchor = theta0[rho_dim];
            psi_rank_stable_floor = evaluator
                .psi_gram_rank_stable_floor(psi_anchor)
                .filter(|&f| f.is_finite() && f > psi_lo && f < psi_anchor);
            log::info!(
                "[KAPPA-PHASE-FLOOR] n_rows={} psi_lo={psi_lo:.6} psi_anchor={psi_anchor:.6} \
                 rank_stable_floor={:?} lifted={}",
                data.nrows(),
                evaluator.psi_gram_rank_stable_floor(psi_anchor),
                psi_rank_stable_floor.is_some(),
            );
            if let Some(floor) = psi_rank_stable_floor {
                log::info!(
                    "[{label}] rank-stable κ-floor ψ_floor={floor:.6} > window floor \
                     ψ_lo={psi_lo:.6}: lifting the optimizer lower bound to keep every \
                     in-window trial on the n-free design-realization skip (#1033). The \
                     conditioned Gram is rank-deficient below ψ_floor (longest-length-scale \
                     radial mode collapses into the nullspace), where the skip is soundly \
                     refused; that band drifts with n via the sample-std standardization, \
                     so this n-free k-space floor is the n-independent fix."
                );
            }
            // #1033: read the n-free rank-stable κ-CEILING (symmetric twin of the
            // floor). Only clamp INTO the window (strictly below psi_hi, strictly
            // above the seed ψ — the seed is the geometric-mean midpoint, well
            // inside the maximal-rank band), so the optimizer never starts outside
            // its bounds. This is the fix for the n=16000 fast-ladder resets: the
            // line search overshot to ψ≈1.0 (rank 11→10 at the high edge), tripping
            // two O(n) reset_surface calls; clamping the upper bound keeps the
            // search inside the band where the n-free skip stays sound.
            psi_rank_stable_ceiling = evaluator
                .psi_gram_rank_stable_ceiling(psi_anchor)
                .filter(|&c| c.is_finite() && c < psi_hi && c > psi_anchor);
            log::info!(
                "[KAPPA-PHASE-CEIL] n_rows={} psi_hi={psi_hi:.6} psi_anchor={psi_anchor:.6} \
                 rank_stable_ceiling={:?} clamped={}",
                data.nrows(),
                evaluator.psi_gram_rank_stable_ceiling(psi_anchor),
                psi_rank_stable_ceiling.is_some(),
            );
            if let Some(ceiling) = psi_rank_stable_ceiling {
                log::info!(
                    "[{label}] rank-stable κ-ceiling ψ_ceil={ceiling:.6} < window ceiling \
                     ψ_hi={psi_hi:.6}: clamping the optimizer upper bound to keep every \
                     in-window trial on the n-free design-realization skip (#1033). The \
                     conditioned Gram is rank-deficient above ψ_ceil (longest-frequency \
                     radial mode goes collinear), where the skip is soundly refused; a \
                     line-search overshoot there trips the O(n) reset_surface lane (and the \
                     deficient pinning ψ it records resets the next in-band trial too)."
                );
            }
            let gradient_covers_full_window = evaluator.psi_gram_tensor_covers_gradient(psi_lo)
                && evaluator.psi_gram_tensor_covers_gradient(psi_hi);
            if gradient_covers_full_window {
                log::info!(
                    "[{label}] certified ψ-gram tensor gradient lane covers the full \
                     optimizer window [{psi_lo:.3}, {psi_hi:.3}]"
                );
            } else {
                log::info!(
                    "[{label}] ψ-gram tensor value lane certified, but the gradient lane \
                     does not cover the full optimizer window [{psi_lo:.3}, {psi_hi:.3}]; \
                     keeping exact streamed kappa routing"
                );
            }
            // #1033 penalty lane: ψ also moves the penalty `S(ψ)` (the
            // Duchon/ThinPlate Hilbert scale is an analytic function of the
            // length-scale, built from the FROZEN basis CENTERS — not the data
            // rows). The design-revision fast path that the Gram tensor enables
            // SKIPS `reset_surface`, the only place the canonical penalty surface
            // is rebuilt; without re-keying, the inner solve would pair
            // `XᵀWX(ψ_new)` with the stale `S(ψ_old)` and converge to the wrong
            // β̂ / κ-optimum. Rather than interpolate `S(ψ)`, the fast path rebuilds
            // it EXACTLY and n-free per trial from the frozen geometry via
            // `cache.canonical_penalties_at(theta)` (the SAME
            // `canonicalize_penalty_specs` pipeline the slow `reset_surface` runs).
            // Here we only DECLARE the capability to the evaluator; the per-trial
            // staging happens in `eval_full` / `eval_cost`. The skip is enabled
            // exactly when the single spatial term's frozen metadata
            // (Duchon/ThinPlate) admits the exact rebuild. Matérn deliberately
            // does not enter this block: mixing tensor value probes with exact
            // streamed gradients/Hessians changed its selected κ enough to miss
            // the truth-recovery quality gate, so Matérn stays on one exact
            // streamed objective for value, gradient, and Hessian.
            evaluator.set_supports_nfree_penalty_rekey(true);
            log::info!(
                "[{label}] exact n-free ψ-penalty re-key enabled over [{psi_lo:.3}, \
                 {psi_hi:.3}]: in-window fast-path trials rebuild S(ψ) n-free from frozen \
                 geometry (no reset_surface)"
            );
        } else {
            log::info!(
                "[{label}] ψ-gram tensor did not certify over [{psi_lo:.3}, {psi_hi:.3}]; \
                 keeping the exact per-trial path"
            );
        }
        // #1033 (n-independent outer loop): with the n-free Gaussian lane fully
        // armed (Gram tensor attached + exact n-free penalty re-key), the design-
        // realization skip serves the criterion AND the ψ-gradient `(a_j, g_j)`
        // n-free for every in-window trial — but ONLY a `ValueAndGradient` eval
        // takes that skip. A `ValueGradientHessian` eval sets `allow_second_order`,
        // which forces `ensure_theta` → `reset_surface` (the O(n) design re-
        // realization) because the outer Hessian curvature `B_j` is the exact
        // n-dependent slab. So second-order outer steps are the LAST O(n) per-trial
        // cost in the κ search, and they make the outer loop scale with n. Route
        // gradient-only here: the spatial length-scale objective is smooth and the
        // budget policy already establishes that gradient-only quasi-Newton
        // converges to the same optimum strictly cheaper per eval past the pair-
        // Hessian budget — and with the tensor, the realized Hessian is the only
        // remaining expensive operation, so the same argument applies for ANY n
        // once the lane is armed. This keeps every in-window κ-trial on the n-free
        // `ValueAndGradient` skip, delivering the n-independent outer loop. The
        // exact second-order geometry is preserved whenever the lane is NOT armed
        // for gradient-only routing (non-Gaussian, multi-term, Matérn, or an
        // uncertified window), where it still pays O(n) per Hessian but keeps the
        // quality-sensitive exact second-order path.
        if attached
            && evaluator.psi_gram_tensor_covers_gradient(psi_lo)
            && evaluator.psi_gram_tensor_covers_gradient(psi_hi)
            && evaluator.supports_nfree_penalty_rekey()
            && cache.supports_nfree_gradient_only_routing()
        {
            suppress_outer_hessian_for_nfree = true;
            prefer_gradient_only = true;
            log::info!(
                "[{label}] n-free Gaussian ψ-lane armed; suppressing the analytic outer \
                 Hessian and routing gradient-only (BFGS) so the κ outer loop never realizes \
                the O(n) second-order slab — n-independent outer loop (#1033)"
            );
        }
    } else if coord_dim == 1 && family.is_gaussian_identity() {
        log::info!(
            "[{label}] exact n-free ψ-penalty re-key unavailable; skipping ψ-gram tensor \
             attachment so value, gradient, and Hessian remain on the same exact streamed \
             objective"
        );
    }

    // ── Discriminating outer-gradient FD audit (issue #1040 / #944 merge gate) ──
    //
    // At θ₀, central-difference the outer criterion component-by-component and
    // compare it to the analytic outer gradient that drives this single-block
    // joint optimizer. This forks the two failure modes of a non-terminating
    // outer loop — an objective↔gradient DESYNC (analytic ≠ FD) vs weak
    // identifiability (analytic ≈ FD but a near-singular outer Hessian) — and is
    // the standing merge gate for any design-moving ψ-coordinate, including the
    // #944 raw-κ constant-curvature coordinate (labelled `psi_kappa[..]`).
    //
    // Gated strictly to diagnostic-sized problems (auto-derived from the
    // realized (n, θ_dim), no flag) so it never taxes a production fit. The
    // same gate the n-block driver uses.
    //
    // #1688: the audit's whole output is a logged verdict (`log_verdict`,
    // below) — it never feeds the optimizer — yet it costs `1 + 2·theta_dim`
    // extra full REML evaluations (one `ValueGradientHessian` plus a central
    // pair of `ValueOnly` per coordinate). On the common small spatial fit
    // (n≤4000, the gate ceiling) that is a real fraction of total fit time
    // spent purely to produce a diagnostic that is suppressed at the default
    // `Warn` verbosity anyway. So additionally gate on `Info` being enabled:
    // the gradient-FD-audit regression tests install an `Info` logger and keep
    // exercising it; an ordinary production fit at the quiet default skips the
    // extra evals entirely.
    // FD-OK: FD-audit of the analytic outer gradient (small-problem gate, never feeds the optimizer)
    const OUTER_FD_AUDIT_MAX_N: usize = 4_000; // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
    const OUTER_FD_AUDIT_MAX_THETA_DIM: usize = 32; // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
    let n_total = data.nrows();
    let outer_fd_audit_eligible = log::log_enabled!(log::Level::Info) // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
        && analytic_outer_hessian_available // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
        && n_total <= OUTER_FD_AUDIT_MAX_N // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
        && theta_dim <= OUTER_FD_AUDIT_MAX_THETA_DIM; // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
    log::info!(
        "[OUTER-FD-AUDIT/spatial-exact-joint] gate eligible={outer_fd_audit_eligible} \
         analytic_grad={analytic_outer_hessian_available} n_total={n_total} \
         theta_dim={theta_dim} rho_dim={rho_dim} psi_dim={coord_dim}"
    );
    if outer_fd_audit_eligible {
        // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
        let audit = (|| -> Result<gam_solve::rho_optimizer::OuterGradientFdAudit, String> {
            let mut eval_at = |theta: &Array1<f64>,
                               mode: gam_solve::estimate::reml::reml_outer_engine::EvalMode|
             -> Result<
                (
                    f64,
                    Array1<f64>,
                    gam_problem::HessianResult,
                ),
                String,
            > {
                use gam_solve::estimate::reml::reml_outer_engine::EvalMode;
                let order = if matches!(mode, EvalMode::ValueGradientHessian) {
                    OuterEvalOrder::ValueGradientHessian
                } else {
                    OuterEvalOrder::Value
                };
                ctx.eval_full(theta, order, analytic_outer_hessian_available)
                    .map_err(|e| format!("fd-audit eval_full: {e}"))
            };
            let rho_dim_audit = rho_dim;
            let label_fn = move |i: usize| -> String {
                if i < rho_dim_audit {
                    format!("rho[{i}]")
                } else {
                    format!("psi_kappa[{}]", i - rho_dim_audit)
                }
            };
            gam_solve::rho_optimizer::outer_gradient_fd_audit(
                // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
                theta0,
                1e-4,
                label_fn,
                &mut eval_at,
            )
        })();
        // END-FD-OK
        match audit {
            Ok(audit) => audit.log_verdict("spatial-exact-joint"),
            Err(e) => log::warn!("[OUTER-FD-AUDIT/spatial-exact-joint] skipped: {e}"),
        }
    }

    let kphase_prime_order = if analytic_outer_hessian_available && !suppress_outer_hessian_for_nfree {
        OuterEvalOrder::ValueGradientHessian
    } else {
        OuterEvalOrder::ValueAndGradient
    };
    let kphase_prime_start = std::time::Instant::now();
    drop(ctx.eval_full(theta0, kphase_prime_order, analytic_outer_hessian_available)?);
    log::info!(
        "[KAPPA-PHASE-PRIME] n_rows={} order={:?} elapsed_s={:.4} slow_path_resets_total={} design_revision={}",
        data.nrows(),
        kphase_prime_order,
        kphase_prime_start.elapsed().as_secs_f64(),
        ctx.evaluator.slow_path_reset_count(),
        ctx.cache.design_revision(),
    );

    let kphase_cost_calls = std::cell::Cell::new(0usize);
    let kphase_eval_calls = std::cell::Cell::new(0usize);
    let kphase_efs_calls = std::cell::Cell::new(0usize);
    let kphase_cost_total_s = std::cell::Cell::new(0.0);
    let kphase_eval_total_s = std::cell::Cell::new(0.0);
    let kphase_efs_total_s = std::cell::Cell::new(0.0);
    let kphase_nfree_miss_shape = std::cell::Cell::new(0u64);
    let kphase_nfree_miss_value = std::cell::Cell::new(0u64);
    let kphase_nfree_miss_gradient = std::cell::Cell::new(0u64);
    let kphase_nfree_miss_penalty = std::cell::Cell::new(0u64);
    let kphase_nfree_miss_revision = std::cell::Cell::new(0u64);
    let kphase_nfree_miss_second_order = std::cell::Cell::new(0u64);
    let kphase_nfree_miss_other = std::cell::Cell::new(0u64);
    let kphase_optim_start = std::time::Instant::now();
    let kphase_log_kappa_dim = coord_dim;
    let kphase_slow_resets_start = ctx.evaluator.slow_path_reset_count();
    let kphase_design_revision_start = ctx.cache.design_revision();
    // #1868: snapshot the deterministic n-free skip-path row-touch accumulator
    // AFTER the one-time priming eval above, so the reported delta measures only
    // the per-trial inner-synthesis row work across the κ-optimisation phase.
    let kphase_nfree_skip_touches_start = gam_solve::pirls::nfree_skip_row_element_touches();

    // #1033: lift the ψ (log-κ) lower bound to the n-free rank-stable floor so the
    // optimizer never line-searches into the rank-deficient sliver where the
    // design-realization skip is soundly refused (→ O(n) reset_surface). The lift
    // touches ONLY the single design-moving ψ coordinate at `rho_dim`; all ρ
    // bounds are untouched. `psi_rank_stable_floor` is already constrained to lie
    // strictly inside `(psi_lo, theta0[rho_dim])`, so theta0 stays feasible.
    let lower_effective: std::borrow::Cow<'_, Array1<f64>> = match psi_rank_stable_floor {
        Some(floor) if coord_dim == 1 && floor > lower[rho_dim] => {
            let mut lifted = lower.clone();
            lifted[rho_dim] = floor;
            std::borrow::Cow::Owned(lifted)
        }
        _ => std::borrow::Cow::Borrowed(lower),
    };
    let lower = lower_effective.as_ref();

    // #1033: clamp the ψ (log-κ) upper bound DOWN to the n-free rank-stable ceiling
    // so the optimizer never line-searches into the high-edge rank-deficient sliver
    // where the design-realization skip is soundly refused (→ O(n) reset_surface,
    // plus a second reset from the deficient pinning ψ). Touches ONLY the single
    // design-moving ψ coordinate at `rho_dim`; all ρ bounds are untouched.
    // `psi_rank_stable_ceiling` is already constrained to lie strictly inside
    // `(theta0[rho_dim], psi_hi)`, so theta0 stays feasible.
    let upper_effective: std::borrow::Cow<'_, Array1<f64>> = match psi_rank_stable_ceiling {
        Some(ceiling) if coord_dim == 1 && ceiling < upper[rho_dim] => {
            let mut clamped = upper.clone();
            clamped[rho_dim] = ceiling;
            std::borrow::Cow::Owned(clamped)
        }
        _ => std::borrow::Cow::Borrowed(upper),
    };
    let upper = upper_effective.as_ref();

    let problem = exact_joint_multistart_outer_problem(
        theta0,
        lower,
        upper,
        rho_dim,
        coord_dim,
        theta_dim,
        Derivative::Analytic,
        if analytic_outer_hessian_available && !suppress_outer_hessian_for_nfree {
            DeclaredHessianForm::Either
        } else {
            // `Unavailable` when the n-free Gaussian ψ-lane is armed (#1033): the
            // planner then selects BFGS instead of ARC, so the κ loop issues only
            // `ValueAndGradient` evals and every in-window trial takes the n-free
            // design-realization skip.
            DeclaredHessianForm::Unavailable
        },
        prefer_gradient_only,
        // Single-block spatial path: penalty-like rho + spatial psi.
        // EFS/HybridEFS remain eligible (the Wood-Fasiolo PSD structure holds
        // for single-block families with β-independent joint H_L) UNLESS the
        // n-free Gaussian ψ-lane is armed (#1033): HybridEFS forms the trace Gram
        // `tr(H⁻¹ B_d H⁻¹ B_e)` from the n-dependent curvature slab `B_d`, so it
        // realizes O(n) per step exactly like a Hessian eval. Disabling the
        // fixed-point lane there forces the planner to BFGS (`(Analytic,
        // Unavailable)` → `S::Bfgs`), keeping every in-window κ-trial on the
        // n-free `ValueAndGradient` skip even when `n_params` exceeds the small-
        // BFGS threshold (aniso / multi-ψ).
        suppress_outer_hessian_for_nfree,
        seed_risk_profile_for_likelihood_family(&family),
        kappa_options.rel_tol.max(1e-6),
        kappa_options.max_outer_iter.max(1),
        // Rho-axis BFGS cap: log-λ's natural step is ≈ 5 per
        // `first_order_bfgs_loglambda_step_cap`. Anything tighter throttles
        // BFGS on flat REML valleys.
        Some(5.0),
        // Psi-axis BFGS cap: kappa / aniso-log-scale needs ~ln 2 per iter.
        Some(kappa_options.log_step.clamp(0.25, 1.0)),
        None,
        // Calibrate the outer to the n-scaled profiled REML/LAML objective for
        // every family — the iso-κ non-convergence cure (#1053 1-D Matérn,
        // #1066 2-D binomial geo, #1069 GP/kriging). p = baseline design column
        // count.
        Some((data.nrows(), baseline_design.design.ncols())),
        // #1464: widen the over-smoothing ρ ceiling + seed a high-λ probe when a
        // constant-curvature term is present (collapsing +κ kernel needs a large
        // smoothing λ beyond the historical ±12 box).
        !constant_curvature_term_indices(resolvedspec).is_empty(),
    );

    let eval_outer = |ctx: &mut &mut SpatialJointContext<'_>,
                      theta: &Array1<f64>,
                      order: OuterEvalOrder|
     -> Result<OuterEval, EstimationError> {
        let t0 = std::time::Instant::now();
        let allow_second_order_for_call = matches!(order, OuterEvalOrder::ValueGradientHessian)
            && analytic_outer_hessian_available;
        let gate = ctx.nfree_skip_gate_status(theta, allow_second_order_for_call, true);
        let resets_before = ctx.evaluator.slow_path_reset_count();
        let raw = ctx.eval_full(theta, order, analytic_outer_hessian_available);
        let reset_delta = ctx
            .evaluator
            .slow_path_reset_count()
            .saturating_sub(resets_before);
        if reset_delta > 0 {
            if !gate.shape {
                kphase_nfree_miss_shape.set(kphase_nfree_miss_shape.get() + reset_delta);
            }
            if gate.shape && !gate.value {
                kphase_nfree_miss_value.set(kphase_nfree_miss_value.get() + reset_delta);
            }
            if gate.shape && gate.value && !gate.gradient {
                kphase_nfree_miss_gradient.set(kphase_nfree_miss_gradient.get() + reset_delta);
            }
            if gate.shape && gate.value && gate.gradient && !gate.penalty {
                kphase_nfree_miss_penalty.set(kphase_nfree_miss_penalty.get() + reset_delta);
            }
            if gate.shape && gate.value && gate.gradient && gate.penalty && !gate.revision {
                kphase_nfree_miss_revision.set(kphase_nfree_miss_revision.get() + reset_delta);
            }
            if gate.shape
                && gate.value
                && gate.gradient
                && gate.penalty
                && gate.revision
                && gate.second_order
            {
                kphase_nfree_miss_second_order
                    .set(kphase_nfree_miss_second_order.get() + reset_delta);
            }
            if gate.would_skip(true) {
                kphase_nfree_miss_other.set(kphase_nfree_miss_other.get() + reset_delta);
            }
        }
        let elapsed_s = t0.elapsed().as_secs_f64();
        kphase_eval_calls.set(kphase_eval_calls.get() + 1);
        kphase_eval_total_s.set(kphase_eval_total_s.get() + elapsed_s);
        let (theta_norm, log_kappa_norm) = kphase_log_norms(theta, rho_dim);
        log::info!(
            "[KAPPA-PHASE] phase=eval_outer call={} order={:?} design_revision={:?} theta_norm={:.4e} log_kappa_norm={:.4e} elapsed_s={:.4}",
            kphase_eval_calls.get(),
            order,
            Some(ctx.cache.design_revision()),
            theta_norm,
            log_kappa_norm,
            elapsed_s,
        );
        match raw {
            Ok((cost, grad, hess)) => Ok(OuterEval {
                cost,
                gradient: grad,
                hessian: hess,
                inner_beta_hint: None,
            }),
            // A trial hyperparameter at which the spatial kernel design /
            // ψ-derivatives are non-constructible is an infeasible point, not
            // a fatal error: the gradient/Hessian path must retreat exactly as
            // the cost-only path (which already returns +∞) does. Returning
            // `OuterEval::infeasible` keeps the two paths symmetric so a single
            // bad probe — e.g. an anisotropy that overflows the Duchon radial
            // kernel — no longer aborts the whole REML optimization.
            Err(err) if is_recoverable_trial_point_error(&err) => {
                log::debug!(
                    "[{label}] trial point infeasible (kernel design \
                     not constructible at theta={theta:?}): {err}; retreating",
                );
                Ok(OuterEval::infeasible(theta_dim))
            }
            Err(err) => Err(err),
        }
    };

    let mut obj = problem.build_objective_with_eval_order(
        &mut ctx,
        |ctx: &mut &mut SpatialJointContext<'_>, theta: &Array1<f64>| {
            let t0 = std::time::Instant::now();
            let gate = ctx.nfree_skip_gate_status(theta, false, false);
            let resets_before = ctx.evaluator.slow_path_reset_count();
            let cost = ctx.eval_cost(theta);
            let reset_delta = ctx
                .evaluator
                .slow_path_reset_count()
                .saturating_sub(resets_before);
            if reset_delta > 0 {
                if !gate.shape {
                    kphase_nfree_miss_shape.set(kphase_nfree_miss_shape.get() + reset_delta);
                }
                if gate.shape && !gate.value {
                    kphase_nfree_miss_value.set(kphase_nfree_miss_value.get() + reset_delta);
                }
                if gate.shape && gate.value && !gate.penalty {
                    kphase_nfree_miss_penalty.set(kphase_nfree_miss_penalty.get() + reset_delta);
                }
                if gate.shape && gate.value && gate.penalty && !gate.revision {
                    kphase_nfree_miss_revision.set(kphase_nfree_miss_revision.get() + reset_delta);
                }
                if gate.would_skip(false) {
                    kphase_nfree_miss_other.set(kphase_nfree_miss_other.get() + reset_delta);
                }
            }
            let elapsed_s = t0.elapsed().as_secs_f64();
            kphase_cost_calls.set(kphase_cost_calls.get() + 1);
            kphase_cost_total_s.set(kphase_cost_total_s.get() + elapsed_s);
            let (theta_norm, log_kappa_norm) = kphase_log_norms(theta, rho_dim);
            log::info!(
                "[KAPPA-PHASE] phase=cost call={} design_revision={:?} theta_norm={:.4e} log_kappa_norm={:.4e} elapsed_s={:.4}",
                kphase_cost_calls.get(),
                Some(ctx.cache.design_revision()),
                theta_norm,
                log_kappa_norm,
                elapsed_s,
            );
            Ok(cost)
        },
        |ctx: &mut &mut SpatialJointContext<'_>, theta: &Array1<f64>| {
            eval_outer(
                ctx,
                theta,
                // #1033: when the n-free Gaussian ψ-lane is armed we suppress the
                // outer Hessian and route BFGS — so this default gradient eval MUST
                // request `ValueAndGradient`, not `ValueGradientHessian`. A
                // second-order order sets `allow_second_order`, which forces
                // `ensure_theta` → the O(n) design re-realization (the Hessian slab
                // is irreducibly n-dependent), DISARMING the design-revision fast
                // path for every trial — exactly the O(n) κ-loop this lane exists to
                // remove. Gating only the planner's solver (Unavailable→BFGS)
                // without gating this eval-order left every trial second-order.
                if analytic_outer_hessian_available && !suppress_outer_hessian_for_nfree {
                    OuterEvalOrder::ValueGradientHessian
                } else {
                    OuterEvalOrder::ValueAndGradient
                },
            )
        },
        |ctx: &mut &mut SpatialJointContext<'_>, theta: &Array1<f64>, order: OuterEvalOrder| {
            eval_outer(ctx, theta, order)
        },
        Some(|ctx: &mut &mut SpatialJointContext<'_>| {
            ctx.reset();
        }),
        Some(|ctx: &mut &mut SpatialJointContext<'_>, theta: &Array1<f64>| {
            let t0 = std::time::Instant::now();
            let eval = ctx.eval_efs(theta);
            let elapsed_s = t0.elapsed().as_secs_f64();
            kphase_efs_calls.set(kphase_efs_calls.get() + 1);
            kphase_efs_total_s.set(kphase_efs_total_s.get() + elapsed_s);
            let (theta_norm, log_kappa_norm) = kphase_log_norms(theta, rho_dim);
            log::info!(
                "[KAPPA-PHASE] phase=efs call={} design_revision={:?} theta_norm={:.4e} log_kappa_norm={:.4e} elapsed_s={:.4}",
                kphase_efs_calls.get(),
                Some(ctx.cache.design_revision()),
                theta_norm,
                log_kappa_norm,
                elapsed_s,
            );
            eval
        }),
    );

    let run_label = match kind {
        SpatialHyperKind::Anisotropic => "aniso-psi joint REML",
        SpatialHyperKind::Isotropic => "iso-kappa joint REML",
    };
    let result = problem.run(&mut obj, run_label).map_err(|e| {
        EstimationError::InvalidInput(format!(
            "{} analytic optimization failed after exhausting strategy fallbacks: {e}",
            kind.adjective(),
        ))
    })?;
    drop(obj);
    let kphase_total_s = kphase_optim_start.elapsed().as_secs_f64();
    let kphase_slow_resets = ctx
        .evaluator
        .slow_path_reset_count()
        .saturating_sub(kphase_slow_resets_start);
    let kphase_design_revision_delta = ctx
        .cache
        .design_revision()
        .saturating_sub(kphase_design_revision_start);
    let kphase_nfree_skip_touches = gam_solve::pirls::nfree_skip_row_element_touches()
        .saturating_sub(kphase_nfree_skip_touches_start);
    log::info!(
        "[KAPPA-PHASE-SUMMARY] n_rows={} log_kappa_dim={} n_cost={} cost_total_s={:.4} n_eval={} eval_total_s={:.4} n_efs={} efs_total_s={:.4} slow_path_resets={} design_revision_delta={} nfree_skip_row_touches={} nfree_miss_shape={} nfree_miss_value={} nfree_miss_gradient={} nfree_miss_penalty={} nfree_miss_revision={} nfree_miss_second_order={} nfree_miss_other={} optim_total_s={:.4}",
        data.nrows(),
        kphase_log_kappa_dim,
        kphase_cost_calls.get(),
        kphase_cost_total_s.get(),
        kphase_eval_calls.get(),
        kphase_eval_total_s.get(),
        kphase_efs_calls.get(),
        kphase_efs_total_s.get(),
        kphase_slow_resets,
        kphase_design_revision_delta,
        kphase_nfree_skip_touches,
        kphase_nfree_miss_shape.get(),
        kphase_nfree_miss_value.get(),
        kphase_nfree_miss_gradient.get(),
        kphase_nfree_miss_penalty.get(),
        kphase_nfree_miss_revision.get(),
        kphase_nfree_miss_second_order.get(),
        kphase_nfree_miss_other.get(),
        kphase_total_s,
    );
    let timing = SpatialLengthScaleOptimizationTiming {
        log_kappa_dim: kphase_log_kappa_dim,
        cost_calls: kphase_cost_calls.get(),
        cost_total_s: kphase_cost_total_s.get(),
        eval_calls: kphase_eval_calls.get(),
        eval_total_s: kphase_eval_total_s.get(),
        efs_calls: kphase_efs_calls.get(),
        efs_total_s: kphase_efs_total_s.get(),
        slow_path_resets: kphase_slow_resets,
        design_revision_delta: kphase_design_revision_delta,
        nfree_skip_row_touches: kphase_nfree_skip_touches,
        nfree_miss_shape: kphase_nfree_miss_shape.get(),
        nfree_miss_value: kphase_nfree_miss_value.get(),
        nfree_miss_gradient: kphase_nfree_miss_gradient.get(),
        nfree_miss_penalty: kphase_nfree_miss_penalty.get(),
        nfree_miss_revision: kphase_nfree_miss_revision.get(),
        nfree_miss_second_order: kphase_nfree_miss_second_order.get(),
        nfree_miss_other: kphase_nfree_miss_other.get(),
        optim_total_s: kphase_total_s,
    };
    if !result.converged {
        // Mirror `fit_term_collectionwith_exact_spatial_adaptive_regularization`
        // (commit 0267d082): the strict absolute-floor gradient criterion is too
        // tight when the outer Hessian carries a near-null direction (η-anchor
        // drift, ill-conditioned operator-collocation Gram, etc.) — the iterate
        // settles into a flat valley with ‖g‖_proj at numerical-noise scale
        // (~1e-5 for cost ~1e1 in double precision) which is above the 1e-6
        // absolute floor but well below the textbook mgcv `magic` REML rule
        // ‖g‖_proj ≤ τ·(1 + |f|). Accept the iterate under the rel-to-cost
        // form when the absolute form has timed out; divergent runs (‖g‖
        // large relative to |f|) still surface as errors.
        let rel_to_cost_threshold = options.tol * (1.0_f64 + result.final_value.abs());
        if let Some(final_grad) = result
            .final_grad_norm
            .filter(|v| v.is_finite() && *v <= rel_to_cost_threshold)
        {
            log::info!(
                "[{}] outer optimization hit max_iter={} but \
                 projected gradient norm {:.3e} ≤ τ·(1+|f|) = {:.3e} \
                 (τ={:.3e}, |f|={:.3e}); accepting iterate under the mgcv-style \
                 relative-to-cost REML convergence criterion.",
                label,
                result.iterations,
                final_grad,
                rel_to_cost_threshold,
                options.tol,
                result.final_value.abs(),
            );
        } else if result.final_value.is_finite() {
            // The joint κ optimizer is a *refinement* layered on top of an
            // always-valid frozen baseline geometry (the REML-seeded length
            // scales in `best`); a run that hits the iteration cap without
            // certifying a stationary point — and without clearing the
            // relative-to-cost gate above — must degrade to that baseline, not
            // abort the parent fit. The `gam` CLI fits this exact data (#1126):
            // its looser outer tolerance (`tol=1e-6`) lets this same optimizer
            // converge in ≤80 iters, whereas the formula/FFI path's tightened
            // `tol=1e-10` (the #893 replication-invariance tolerance) leaves it
            // mid-descent at the cap. Loosening the tolerance would weaken that
            // invariant for every fit; instead we report the non-convergence and
            // let the caller keep the baseline. The terminal cost is finite, so
            // the iterate is well-defined — this is ordinary slow convergence,
            // not a numerical blowup.
            log::warn!(
                "[{}] {} did not converge after {} iterations \
                 (final_objective={:.6e}, final_grad_norm={}); keeping the \
                 frozen baseline geometry instead of aborting the fit.",
                label,
                kind.adjective(),
                result.iterations,
                result.final_value,
                result.final_grad_norm_report(),
            );
            return Ok((
                SpatialJointOutcome::NonConverged {
                    iterations: result.iterations,
                    final_value: result.final_value,
                    final_grad_norm: result.final_grad_norm,
                },
                timing,
            ));
        } else {
            // A non-finite terminal cost is a genuine numerical blowup (NaN/inf
            // propagating through the gradient/Hessian wiring), not the ordinary
            // slow convergence handled above — surface it rather than masking a
            // real defect behind the baseline fallback.
            crate::bail_invalid_estim!(
                "{} analytic optimization diverged after {} iterations (final_objective={:.6e}, final_grad_norm={})",
                kind.adjective(),
                result.iterations,
                result.final_value,
                result.final_grad_norm_report(),
            );
        }
    }
    log::trace!(
        "[{}] converged in {} iterations, final_value={:.6e}, grad_norm={}",
        label,
        result.iterations,
        result.final_value,
        result.final_grad_norm_report(),
    );
    // No sum-to-zero enforcement needed: ψ coordinates are unconstrained during
    // optimization. For the anisotropic kind the decomposition into (ψ̄, η)
    // happens later in apply_tospec.
    let theta_star = result.rho;
    Ok((
        SpatialJointOutcome::Optimized {
            theta_star,
            final_value: result.final_value,
        },
        timing,
    ))
}

/// Apply a length scale to a single `SmoothTermSpec` (independent of any
/// outer `TermCollectionSpec`). Mirrors `set_spatial_length_scale` but on a
/// term in isolation; used by the incremental realizer's cached planned spec.
fn set_single_term_spatial_length_scale(
    term: &mut SmoothTermSpec,
    length_scale: f64,
) -> Result<(), EstimationError> {
    match &mut term.basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => {
            spec.length_scale = length_scale;
            Ok(())
        }
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.length_scale = length_scale;
            Ok(())
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            spec.length_scale = Some(length_scale);
            Ok(())
        }
        _ => Err(EstimationError::InvalidInput(format!(
            "term '{}' does not expose a spatial length scale",
            term.name
        ))),
    }
}

/// Apply anisotropy contrasts to a single `SmoothTermSpec`. Mirrors
/// `set_spatial_aniso_log_scales` but on a term in isolation; used by the
/// incremental realizer's cached planned spec.
fn set_single_term_spatial_aniso_log_scales(
    term: &mut SmoothTermSpec,
    eta: Vec<f64>,
) -> Result<(), EstimationError> {
    let eta = center_aniso_log_scales(&eta);
    match &mut term.basis {
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.aniso_log_scales = Some(eta);
            Ok(())
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            spec.aniso_log_scales = Some(eta);
            Ok(())
        }
        _ => Err(EstimationError::InvalidInput(format!(
            "term '{}' does not support aniso_log_scales",
            term.name
        ))),
    }
}

/// Freeze the design-moving representer length-scale dial on every measure-jet
/// term in `spec` (sets `learn_length_scale = false`), so ℓ stays at its
/// realized auto value with no outer REML enrollment.
///
/// Used by COUPLED-block families (bernoulli marginal-slope: a shared mjs
/// surface feeds both the marginal mean and the log-slope). In that coupling a
/// design-moving kernel-scale dial on the shared covariates is an
/// identifiability hazard: the outer search can reach a sharp ℓ at which a
/// marginal smooth direction trades off against the log-slope into a
/// separation-scale runaway (#1116). A single Gaussian surface has no such
/// coupling and keeps ℓ learnable. Returns the number of terms frozen.
/// The signed sectional curvature κ of a constant-curvature smooth at
/// `term_idx`, or `None` if that term is not a `curv(...)` smooth. After a fit
/// with κ-optimization enabled this reads the **fitted κ̂** out of the resolved
/// spec (`freeze_term_collection_from_design` writes the optimized κ back into
/// the spec, and `BasisMetadata::ConstantCurvature.kappa` carries the same
/// value). This is the headline #944 estimand accessor — the κ̂ in
/// "κ̂ = −1.8 (95% CI …)". Mirrors [`get_spatial_length_scale`].
pub fn get_constant_curvature_kappa(spec: &TermCollectionSpec, term_idx: usize) -> Option<f64> {
    constant_curvature_term_spec(spec, term_idx).map(|cc| cc.kappa)
}

/// `true` when `term_idx` is a `curv(...)` smooth whose user PINNED the
/// sectional curvature with an explicit `kappa=` (the mgcv-`sp=` convention,
/// gam#2152). A pinned κ is a fixed geometry: the outer loop must hold it
/// constant and never run any of its κ re-derivation heuristics (the κ-fair
/// sign scan, the κ-fair argmin override, or the joint [ρ, κ] refinement) on
/// that term. Non-CC terms and CC terms whose `kappa=` was omitted (κ free,
/// #944/#1464 estimation) return `false`.
pub fn constant_curvature_kappa_is_fixed(spec: &TermCollectionSpec, term_idx: usize) -> bool {
    constant_curvature_term_spec(spec, term_idx).is_some_and(|cc| cc.kappa_fixed)
}

/// Indices of every constant-curvature (`curv(...)`) smooth term in `spec`.
pub fn constant_curvature_term_indices(spec: &TermCollectionSpec) -> Vec<usize> {
    (0..spec.smooth_terms.len())
        .filter(|&idx| constant_curvature_term_spec(spec, idx).is_some())
        .collect()
}


#[derive(Debug, Clone)]
struct SingleSmoothTermRealization {
    design_local: DesignMatrix,
    term: SmoothTerm,
    dropped_penaltyinfo: Vec<DroppedPenaltyBlockInfo>,
}

impl SingleSmoothTermRealization {
    fn active_penaltyinfo(&self) -> Vec<PenaltyInfo> {
        self.term
            .penaltyinfo_local
            .iter()
            .filter(|info| info.active)
            .cloned()
            .collect()
    }
}

fn build_single_smooth_term_realization(
    data: ArrayView2<'_, f64>,
    termspec: &SmoothTermSpec,
) -> Result<SingleSmoothTermRealization, BasisError> {
    let raw = build_smooth_design(data, std::slice::from_ref(termspec))?;
    finish_single_smooth_term_realization(raw)
}

fn finish_single_smooth_term_realization(
    raw: RawSmoothDesign,
) -> Result<SingleSmoothTermRealization, BasisError> {
    let RawSmoothDesign {
        term_designs,
        dropped_penaltyinfo,
        terms,
        ..
    } = raw;
    let term = terms.into_iter().next().ok_or_else(|| {
        BasisError::InvalidInput("single-term smooth build returned no term".to_string())
    })?;
    let design = term_designs.into_iter().next().ok_or_else(|| {
        BasisError::InvalidInput("single-term smooth build returned no term design".to_string())
    })?;

    Ok(SingleSmoothTermRealization {
        design_local: design,
        term,
        dropped_penaltyinfo,
    })
}

/// Wrap a fresh `LocalSmoothTermBuild` (produced by `build_single_local_smooth_term`)
/// into a `SingleSmoothTermRealization`. Mirrors the single-term portion of
/// `build_smooth_design_withworkspace_unvalidated`, but skips the joint center
/// planner and per-term workspace fork — the realizer drives κ-only rebuilds
/// directly with its persistent workspace so basis caches survive across BFGS
/// κ proposals.
fn wrap_local_build_as_realization(
    mut local: LocalSmoothTermBuild,
    termspec: &SmoothTermSpec,
) -> Result<SingleSmoothTermRealization, String> {
    let p_local = local.dim;
    let lb_local = if local.box_reparam {
        shape_lower_bounds_local(termspec.shape, p_local)
    } else {
        None
    };

    let active_count = local.penaltyinfo.iter().filter(|info| info.active).count();
    if active_count != local.penalties.len() {
        return Err(format!(
            "internal penalty info mismatch for term '{}': active_infos={}, penalties={}",
            termspec.name,
            active_count,
            local.penalties.len()
        ));
    }

    let mut dropped_penaltyinfo = Vec::<DroppedPenaltyBlockInfo>::new();
    for info in local.penaltyinfo.iter().filter(|info| !info.active) {
        dropped_penaltyinfo.push(DroppedPenaltyBlockInfo {
            termname: Some(termspec.name.clone()),
            penalty: info.clone(),
        });
    }
    for info in &local.pre_dropped_penaltyinfo {
        dropped_penaltyinfo.push(DroppedPenaltyBlockInfo {
            termname: Some(termspec.name.clone()),
            penalty: info.clone(),
        });
    }

    // Stage-2 joint-null absorption rotation, same logic as the main
    // aggregation loop in `build_smooth_design_withworkspace_unvalidated`:
    // apply Q when Some AND the smooth has no shape constraints.
    let applied_rotation: Option<gam_terms::basis::JointNullRotation> = match (
        local.joint_null_rotation.take(),
        lb_local.is_some(),
        local.linear_constraints.is_some(),
    ) {
        (Some(rot), false, false) => {
            let q = &rot.rotation;
            let dense = local
                .design
                .try_to_dense_by_chunks("joint-null absorption rotation (single realization)")
                .map_err(|e| {
                    format!(
                        "joint-null absorption rotation: dense conversion failed for term '{}': {}",
                        termspec.name, e
                    )
                })?;
            let rotated = gam_linalg::faer_ndarray::fast_ab(&dense, q);
            local.design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(rotated));
            local.penalties = local
                .penalties
                .into_iter()
                .map(|s_local| {
                    let qt_s = gam_linalg::faer_ndarray::fast_atb(q, &s_local);
                    gam_linalg::faer_ndarray::fast_ab(&qt_s, q)
                })
                .collect();
            local.ops = vec![None; local.penalties.len()];
            local.kronecker_factored = None;
            Some(rot)
        }
        (Some(_), _, _) => None,
        (None, _, _) => None,
    };

    let smooth_term = SmoothTerm {
        name: termspec.name.clone(),
        coeff_range: 0..p_local,
        shape: termspec.shape,
        penalties_local: local.penalties.clone(),
        nullspace_dims: local.nullspaces.clone(),
        penaltyinfo_local: local.penaltyinfo.clone(),
        metadata: local.metadata.clone(),
        lower_bounds_local: lb_local,
        linear_constraints_local: local.linear_constraints.clone(),
        kronecker_factored: local.kronecker_factored.take(),
        joint_null_rotation: applied_rotation,
        // Single-term realizations never run the global ownership pass, so
        // there is no overlap residualization to export here (#978).
        unabsorbed_global_orthogonality: None,
    };

    Ok(SingleSmoothTermRealization {
        design_local: local.design,
        term: smooth_term,
        dropped_penaltyinfo,
    })
}

/// Extract the κ-invariant pieces of a freshly-built spatial basis — center
/// cloud (in standardized coords) and `input_scales` — and bake them into a
/// `SmoothTermSpec` whose `center_strategy` becomes `UserProvided` and whose
/// `input_scales` is `Some`. Subsequent rebuilds driven from this cached spec
/// will short-circuit `select_centers_by_strategy` (KMeans / FarthestPoint /
/// EqualMass cluster searches over n×d data) and `compute_spatial_input_scales`
/// (per-axis variance over n rows), leaving only the κ-dependent kernel
/// values and basis assembly. Returns `None` for non-spatial families or when
/// the metadata does not yet expose the required pieces (for instance when a
/// ThinPlate request was auto-promoted to Duchon during the build).
fn freeze_geometry_from_metadata(
    termspec: &SmoothTermSpec,
    metadata: &BasisMetadata,
) -> Option<SmoothTermSpec> {
    let mut frozen = termspec.clone();
    match (&mut frozen.basis, metadata) {
        (
            SmoothBasisSpec::Matern {
                spec,
                input_scales: spec_scales,
                ..
            },
            BasisMetadata::Matern {
                centers,
                input_scales: meta_scales,
                identifiability_transform,
                nullspace_shrinkage_survived,
                ..
            },
        ) => {
            spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
            if spec_scales.is_none()
                && let Some(s) = meta_scales.clone()
            {
                *spec_scales = Some(s);
            }
            // Pin BOTH the cold-build identifiability transform `Z` AND the
            // double-penalty nullspace-shrinkage decision into a
            // `FrozenTransform` (gam#787/#860, #1122). Without this, the
            // κ-optimizer's per-trial value rebuild re-runs the κ-DEPENDENT
            // spectral test (`build_nullspace_shrinkage_penalty`), whose
            // tolerance scales with `λ_max(A(κ))`: as κ moves, near-null
            // eigenvalues of the projected kernel Gram `A` cross the threshold,
            // so the `DoublePenaltyNullspace` block `P/√r` (and its null
            // dimension `r`) JUMP discontinuously between line-search trials.
            // The analytic ψ-gradient — assembled in a fixed frozen eigenbasis
            // — cannot follow those discrete jumps, so the joint REML objective
            // V(κ) is piecewise-discontinuous while the gradient is smooth: an
            // objective↔gradient desync that stalls the isotropic-κ optimizer
            // with a large residual gradient at the iteration cap. Freezing the
            // decision (and the transform that `A` is built from) makes the
            // per-trial value rebuild and the analytic gradient share one fixed
            // `Z` and one fixed `r`, restoring a smooth, differentiable V(κ).
            if let Some(transform) = identifiability_transform.clone() {
                spec.identifiability = MaternIdentifiability::FrozenTransform {
                    transform,
                    nullspace_shrinkage_survived: Some(*nullspace_shrinkage_survived),
                };
            }
            Some(frozen)
        }
        (
            SmoothBasisSpec::Duchon {
                spec,
                input_scales: spec_scales,
                ..
            },
            BasisMetadata::Duchon {
                centers,
                input_scales: meta_scales,
                ..
            },
        ) => {
            spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
            if spec_scales.is_none()
                && let Some(s) = meta_scales.clone()
            {
                *spec_scales = Some(s);
            }
            Some(frozen)
        }
        (
            SmoothBasisSpec::ThinPlate {
                spec,
                input_scales: spec_scales,
                ..
            },
            BasisMetadata::ThinPlate {
                centers,
                input_scales: meta_scales,
                ..
            },
        ) => {
            spec.center_strategy = CenterStrategy::UserProvided(centers.clone());
            if spec_scales.is_none()
                && let Some(s) = meta_scales.clone()
            {
                *spec_scales = Some(s);
            }
            Some(frozen)
        }
        // Family mismatch (e.g. ThinPlate auto-promotion to Duchon) leaves the
        // cache empty; we'll retry materialization on the next κ apply.
        _ => None,
    }
}

fn rebuild_smooth_auxiliary_state(
    smooth: &mut SmoothDesign,
    dropped_penaltyinfo_by_term: &[Vec<DroppedPenaltyBlockInfo>],
) -> Result<(), String> {
    if dropped_penaltyinfo_by_term.len() != smooth.terms.len() {
        return Err(SmoothError::dimension_mismatch(format!(
            "smooth dropped-penalty cache mismatch: terms={}, dropped_sets={}",
            smooth.terms.len(),
            dropped_penaltyinfo_by_term.len()
        ))
        .into());
    }

    let total_p = smooth.total_smooth_cols();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(total_p, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintrows: Vec<Array1<f64>> = Vec::new();
    let mut linear_constraint_b: Vec<f64> = Vec::new();

    for term in &smooth.terms {
        let range = term.coeff_range.clone();
        if let Some(lb_local) = term.lower_bounds_local.as_ref() {
            if lb_local.len() != range.len() {
                return Err(SmoothError::dimension_mismatch(format!(
                    "smooth lower-bound cache mismatch for term '{}': bounds={}, coeffs={}",
                    term.name,
                    lb_local.len(),
                    range.len()
                ))
                .into());
            }
            coefficient_lower_bounds
                .slice_mut(s![range.clone()])
                .assign(lb_local);
            any_bounds = true;
        }
        if let Some(lin_local) = term.linear_constraints_local.as_ref() {
            if lin_local.a.ncols() != range.len() {
                return Err(SmoothError::dimension_mismatch(format!(
                    "smooth linear-constraint cache mismatch for term '{}': cols={}, coeffs={}",
                    term.name,
                    lin_local.a.ncols(),
                    range.len()
                ))
                .into());
            }
            for r in 0..lin_local.a.nrows() {
                let mut row = Array1::<f64>::zeros(total_p);
                row.slice_mut(s![range.clone()]).assign(&lin_local.a.row(r));
                linear_constraintrows.push(row);
                linear_constraint_b.push(lin_local.b[r]);
            }
        }
    }

    smooth.coefficient_lower_bounds = if any_bounds {
        Some(coefficient_lower_bounds)
    } else {
        None
    };
    smooth.linear_constraints = if linear_constraintrows.is_empty() {
        None
    } else {
        let mut a = Array2::<f64>::zeros((linear_constraintrows.len(), total_p));
        for (i, row) in linear_constraintrows.iter().enumerate() {
            a.row_mut(i).assign(row);
        }
        Some(LinearInequalityConstraints {
            a,
            b: Array1::from_vec(linear_constraint_b),
        })
    };
    smooth.dropped_penaltyinfo = dropped_penaltyinfo_by_term
        .iter()
        .flat_map(|infos| infos.iter().cloned())
        .collect();
    Ok(())
}

fn rebuild_term_collection_auxiliary_state(
    spec: &TermCollectionSpec,
    design: &mut TermCollectionDesign,
) -> Result<(), String> {
    if spec.linear_terms.len() != design.linear_ranges.len() {
        return Err(SmoothError::dimension_mismatch(format!(
            "term-collection linear bookkeeping mismatch: spec_terms={}, design_ranges={}",
            spec.linear_terms.len(),
            design.linear_ranges.len()
        ))
        .into());
    }

    let p_total = design.design.ncols();
    let smooth_start = p_total.saturating_sub(design.smooth.total_smooth_cols());
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(p_total, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintrows: Vec<Array1<f64>> = Vec::new();
    let mut linear_constraint_b: Vec<f64> = Vec::new();

    for (linear, (_, range)) in spec.linear_terms.iter().zip(design.linear_ranges.iter()) {
        if range.len() != 1 {
            return Err(SmoothError::dimension_mismatch(format!(
                "linear term '{}' expected one coefficient column, found {}",
                linear.name,
                range.len()
            ))
            .into());
        }
        let col = range.start;
        if let Some(lb) = linear.coefficient_min {
            let mut row = Array1::<f64>::zeros(p_total);
            row[col] = 1.0;
            linear_constraintrows.push(row);
            linear_constraint_b.push(lb);
        }
        if let Some(ub) = linear.coefficient_max {
            let mut row = Array1::<f64>::zeros(p_total);
            row[col] = -1.0;
            linear_constraintrows.push(row);
            linear_constraint_b.push(-ub);
        }
    }

    if let Some(lb_smooth) = design.smooth.coefficient_lower_bounds.as_ref() {
        if lb_smooth.len() != design.smooth.total_smooth_cols() {
            return Err(SmoothError::dimension_mismatch(format!(
                "smooth lower-bound width mismatch: bounds={}, smooth_cols={}",
                lb_smooth.len(),
                design.smooth.total_smooth_cols()
            ))
            .into());
        }
        coefficient_lower_bounds
            .slice_mut(s![
                smooth_start..(smooth_start + design.smooth.total_smooth_cols())
            ])
            .assign(lb_smooth);
        any_bounds = true;
    }
    if let Some(lin_smooth) = design.smooth.linear_constraints.as_ref() {
        if lin_smooth.a.ncols() != design.smooth.total_smooth_cols() {
            return Err(SmoothError::dimension_mismatch(format!(
                "smooth linear-constraint width mismatch: cols={}, smooth_cols={}",
                lin_smooth.a.ncols(),
                design.smooth.total_smooth_cols()
            ))
            .into());
        }
        let mut a_global = Array2::<f64>::zeros((lin_smooth.a.nrows(), p_total));
        a_global
            .slice_mut(s![
                ..,
                smooth_start..(smooth_start + design.smooth.total_smooth_cols())
            ])
            .assign(&lin_smooth.a);
        for r in 0..a_global.nrows() {
            linear_constraintrows.push(a_global.row(r).to_owned());
            linear_constraint_b.push(lin_smooth.b[r]);
        }
    }

    let lower_bound_constraints = if any_bounds {
        linear_constraints_from_lower_bounds_global(&coefficient_lower_bounds)
    } else {
        None
    };
    let explicit_linear_constraints = if linear_constraintrows.is_empty() {
        None
    } else {
        let mut a = Array2::<f64>::zeros((linear_constraintrows.len(), p_total));
        for (i, row) in linear_constraintrows.iter().enumerate() {
            a.row_mut(i).assign(row);
        }
        Some(LinearInequalityConstraints {
            a,
            b: Array1::from_vec(linear_constraint_b),
        })
    };

    design.coefficient_lower_bounds = if any_bounds {
        Some(coefficient_lower_bounds)
    } else {
        None
    };
    design.linear_constraints =
        merge_linear_constraints_global(explicit_linear_constraints, lower_bound_constraints);
    design.dropped_penaltyinfo = design.smooth.dropped_penaltyinfo.clone();
    Ok(())
}

fn theta_values_match(left: &Array1<f64>, right: &Array1<f64>) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(&l, &r)| l.to_bits() == r.to_bits())
}

fn latent_values_match(left: &Array1<f64>, right: &Array1<f64>) -> bool {
    theta_values_match(left, right)
}

fn spatial_aniso_matches(left: Option<&[f64]>, right: Option<&[f64]>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(&x, &y)| x.to_bits() == y.to_bits())
        }
        _ => false,
    }
}

fn spatial_length_scale_matches(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => a.to_bits() == b.to_bits(),
        _ => false,
    }
}

struct FrozenTermCollectionIncrementalRealizer<'d> {
    data: ArrayView2<'d, f64>,
    spec: TermCollectionSpec,
    design: TermCollectionDesign,
    fixed_blocks: Vec<DesignBlock>,
    dropped_penaltyinfo_by_term: Vec<Vec<DroppedPenaltyBlockInfo>>,
    smooth_penalty_ranges: Vec<Range<usize>>,
    full_penalty_ranges: Vec<Range<usize>>,
    /// Persistent workspace for basis cache reuse across κ proposals.
    /// Distance matrices are cached here so they're computed once and
    /// reused across repeated `apply_log_kappa_to_term` calls.
    basisworkspace: gam_terms::basis::BasisWorkspace,
    /// Per-term cached realization geometry for incremental κ updates.
    ///
    /// On the first κ-driven rebuild of term `i`, this slot is populated with a
    /// `SmoothTermSpec` whose κ-invariant geometry — center cloud (as
    /// `CenterStrategy::UserProvided`) and `input_scales` — has been frozen
    /// out of the realized basis metadata. Subsequent
    /// `apply_log_kappa_to_term` calls reuse this spec, mutating only the
    /// κ / aniso fields. This short-circuits `select_centers_by_strategy`
    /// (KMeans / FarthestPoint / EqualMass cluster searches over the n×d data
    /// matrix) and `compute_spatial_input_scales` (per-axis variance pass
    /// over n rows) on every BFGS κ-eval, leaving the kernel-value pass and
    /// basis assembly as the only work.
    spatial_realization_geometry: Vec<Option<SmoothTermSpec>>,
    /// Monotonic counter incremented every time `apply_log_kappa` actually
    /// rebuilds the realized design / smooth penalties. Read by the
    /// design-revision-counter fast path in `ExternalJointHyperEvaluator`
    /// to skip redundant canonical-penalty rebuilds and cache wipes when
    /// the outer BFGS loop probes the same ψ twice in a row.
    design_revision: u64,
}

impl<'d> std::fmt::Debug for FrozenTermCollectionIncrementalRealizer<'d> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrozenTermCollectionIncrementalRealizer")
            .field("data_shape", &(self.data.nrows(), self.data.ncols()))
            .field("fixed_blocks", &self.fixed_blocks.len())
            .finish_non_exhaustive()
    }
}

impl<'d> FrozenTermCollectionIncrementalRealizer<'d> {
    fn new(
        data: ArrayView2<'d, f64>,
        spec: TermCollectionSpec,
        design: TermCollectionDesign,
    ) -> Result<Self, String> {
        if spec.smooth_terms.len() != design.smooth.terms.len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "incremental realizer smooth term mismatch: spec_terms={}, design_terms={}",
                spec.smooth_terms.len(),
                design.smooth.terms.len()
            ))
            .into());
        }

        let mut smooth_cursor = 0usize;
        let mut smooth_penalty_ranges = Vec::with_capacity(design.smooth.terms.len());
        for term in &design.smooth.terms {
            let next = smooth_cursor + term.penalties_local.len();
            smooth_penalty_ranges.push(smooth_cursor..next);
            smooth_cursor = next;
        }
        if smooth_cursor != design.smooth.penalties.len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "incremental realizer smooth penalty mismatch: ranged={}, actual={}",
                smooth_cursor,
                design.smooth.penalties.len()
            ))
            .into());
        }

        let fixed_penalty_offset = design
            .penalties
            .len()
            .checked_sub(design.smooth.penalties.len())
            .ok_or_else(|| {
                "incremental realizer encountered invalid penalty bookkeeping".to_string()
            })?;
        let full_penalty_ranges = smooth_penalty_ranges
            .iter()
            .map(|range| (fixed_penalty_offset + range.start)..(fixed_penalty_offset + range.end))
            .collect::<Vec<_>>();
        let fixed_blocks = build_term_collection_fixed_blocks(data, &spec)
            .map_err(|e| format!("failed to cache fixed term-collection blocks: {e}"))?;

        let mut dropped_penaltyinfo_by_term = Vec::with_capacity(spec.smooth_terms.len());
        for (term_idx, termspec) in spec.smooth_terms.iter().enumerate() {
            let realization =
                build_single_smooth_term_realization(data, termspec).map_err(|e| {
                    format!(
                        "failed to build cached realization for smooth term '{}' (index {}): {e}",
                        termspec.name, term_idx
                    )
                })?;
            let expected_cols = design.smooth.terms[term_idx].coeff_range.len();
            if realization.design_local.ncols() != expected_cols {
                return Err(SmoothError::dimension_mismatch(format!(
                    "cached realization width mismatch for term '{}': cached_cols={}, design_cols={}",
                    termspec.name,
                    realization.design_local.ncols(),
                    expected_cols
                ))
                .into());
            }
            if realization.active_penaltyinfo().len()
                != design.smooth.terms[term_idx].penalties_local.len()
            {
                return Err(SmoothError::dimension_mismatch(format!(
                    "cached realization penalty mismatch for term '{}': cached_penalties={}, design_penalties={}",
                    termspec.name,
                    realization.active_penaltyinfo().len(),
                    design.smooth.terms[term_idx].penalties_local.len()
                ))
                .into());
            }
            dropped_penaltyinfo_by_term.push(realization.dropped_penaltyinfo);
        }

        let geometry_slots = spec.smooth_terms.len();
        Ok(Self {
            data,
            spec,
            design,
            fixed_blocks,
            dropped_penaltyinfo_by_term,
            smooth_penalty_ranges,
            full_penalty_ranges,
            basisworkspace: gam_terms::basis::BasisWorkspace::new(),
            spatial_realization_geometry: vec![None; geometry_slots],
            design_revision: 0,
        })
    }

    fn design_revision(&self) -> u64 {
        self.design_revision
    }

    fn spec(&self) -> &TermCollectionSpec {
        &self.spec
    }

    fn design(&self) -> &TermCollectionDesign {
        &self.design
    }

    /// True when this realizer carries exactly ONE spatial smooth term whose
    /// frozen basis geometry (`BasisMetadata::Duchon`/`ThinPlate`)
    /// admits an EXACT, n-free penalty rebuild at a new length-scale (#1033).
    /// The κ-loop fast path gates its design-realization skip on this: the skip
    /// leaves `reset_surface` un-run, so it is only sound when `S(ψ_new)` can be
    /// re-keyed n-free from the frozen geometry (centers + identifiability
    /// transform + operator collocation points), never from the data rows, AND
    /// the re-keyed penalty's block topology is IDENTICAL to the one the frozen
    /// design carries.
    ///
    /// Matérn stays on the exact slow re-key path here, but NOT for the reason
    /// #1270 originally pinned. The operator-triplet penalty re-key (#1274) IS
    /// fully landed: `canonical_penalties_at_psi` and
    /// `canonical_penalty_derivatives_at_psi` both rebuild the realized Matérn
    /// `{mass, tension, stiffness}` triplet (and its analytic ψ-derivative)
    /// n-free from the frozen collocation geometry, routed through the SAME
    /// shared `matern_operator_penalty_triplet_at_length_scale` builder the
    /// design uses — so the block topology is ψ-stable by construction and the
    /// surface is byte-identical to the slow path across the ψ window (pinned
    /// to <1e-10 by `matern_nfree_rekey_topology_tests`). The historical
    /// "the re-key cannot reproduce the operator triplet" rationale is resolved.
    ///
    /// Re-admission is nonetheless withheld because it is net-negative on the
    /// CURRENT architecture, for two independent reasons the #1274 acceptance
    /// gates surface:
    ///   1. NO SPEED WIN. Even with the penalty re-keyed, the #1264
    ///      reduced-basis-rotation soundness gate (`psi_gram_tensor_covers_skip`)
    ///      refuses Matérn's rotating collocation geometry, so the design-
    ///      realization skip still falls to the exact O(n) `reset_surface`
    ///      re-realization every trial — admitting the penalty rekey alone buys
    ///      no n-independence. Closing this needs an n-free re-key of the Matérn
    ///      *design* (Chebyshev-in-ψ Gram over the rotating basis), which is the
    ///      remaining design-scope work, not a flag flip.
    ///   2. QUALITY REGRESSION. Re-admitting Matérn (as #1033 `6a5a2e1` did,
    ///      reverted by `feb0eb5`) perturbs the selected fit enough to miss the
    ///      mgcv/GP truth-recovery bar (`matern_nu_sweep_*`) — slower AND worse.
    ///
    /// So Matérn is deliberately "slow-but-right". Duchon/ThinPlate are the
    /// #1033 acceptance lane. `matern_nfree_rekey_topology_tests` test (b) pins
    /// this negative admission contract: a flip must first re-clear both gates.
    fn supports_nfree_penalty_rekey(&self, spatial_terms: &[usize]) -> bool {
        if spatial_terms.len() != 1 {
            return false;
        }
        let term_idx = spatial_terms[0];
        matches!(
            self.design.smooth.terms.get(term_idx).map(|t| &t.metadata),
            Some(BasisMetadata::Duchon { .. } | BasisMetadata::ThinPlate { .. })
        )
    }

    /// True when the armed n-free Gaussian lane should suppress exact outer
    /// Hessians and route κ search through gradient-only BFGS.
    ///
    /// This is deliberately narrower than [`Self::supports_nfree_penalty_rekey`]:
    /// Matérn has an exact n-free operator-triplet `S(ψ)` re-key (#1274), but its
    /// quality gate still depends on the exact second-order outer route. Duchon
    /// and ThinPlate are the #1033 n-independent acceptance lane where the exact
    /// Hessian slab is the remaining O(n) per-trial cost.
    fn supports_nfree_gradient_only_routing(&self, spatial_terms: &[usize]) -> bool {
        if spatial_terms.len() != 1 {
            return false;
        }
        let term_idx = spatial_terms[0];
        matches!(
            self.design.smooth.terms.get(term_idx).map(|t| &t.metadata),
            Some(BasisMetadata::Duchon { .. } | BasisMetadata::ThinPlate { .. })
        )
    }

    /// Rebuild the EXACT canonical penalty surface `S(ψ)` at the length-scale
    /// implied by `psi`, entirely n-free (#1033). Reuses the FROZEN basis
    /// geometry from the single spatial term's `BasisMetadata` (centers,
    /// identifiability transform, operator collocation points — all `k × d`, no
    /// data rows) and the spec's `(power, nullspace_order, operator_penalties,
    /// nu, …)`; only the length-scale moves. The reconstructed term-local
    /// penalty matrices replace the `local` of the FROZEN
    /// `design.penalties` templates (whose `col_range` / `prior_mean` /
    /// `structure_hint` / `op` are ψ-invariant), so the resulting
    /// `PenaltySpec`s are bit-identical in topology to the slow path's; running
    /// them through the SAME `canonicalize_penalty_specs` pipeline yields the
    /// canonical list the kept reference surface must be re-keyed with.
    fn canonical_penalties_at_psi(
        &mut self,
        spatial_terms: &[usize],
        psi: &[f64],
    ) -> Result<(Vec<gam_terms::construction::CanonicalPenalty>, Vec<usize>), String> {
        if spatial_terms.len() != 1 {
            return Err(format!(
                "n-free penalty re-key requires exactly one spatial term, found {}",
                spatial_terms.len()
            ));
        }
        let term_idx = spatial_terms[0];
        // Decode ψ with the same chart used by the slow rebuild path. For
        // Matérn, per-axis ψ entries are REML hyper-coordinates, so the n-free
        // penalty rebuild must consume the trial η contrasts as well as the
        // scalar length scale. Duchon keeps η as fixed geometry and continues
        // to use frozen metadata below.
        let (ls_opt, aniso_from_psi) = spatial_term_psi_to_length_scale_and_aniso(psi);
        // Pull the spec-level penalty configuration (which operator orders are
        // active / double_penalty) — ψ-invariant, frozen at construction.
        let termspec =
            self.spec.smooth_terms.get(term_idx).ok_or_else(|| {
                format!("spatial term {term_idx} out of range for n-free penalty")
            })?;
        let term = self
            .design
            .smooth
            .terms
            .get(term_idx)
            .ok_or_else(|| format!("realized smooth term {term_idx} out of range"))?;
        // The per-term penalties live contiguously in the collection penalty
        // list at the term's `coeff_range` (single-spatial-term collection).
        let p_total = self.design.design.ncols();
        let (locals, nullspace_dims): (Vec<Array2<f64>>, Vec<usize>) = match &term.metadata {
            BasisMetadata::Duchon {
                centers,
                identifiability_transform,
                operator_collocation_points,
                power,
                nullspace_order,
                aniso_log_scales,
                input_scales,
                radial_reparam,
                ..
            } => {
                let operator_penalties = match &termspec.basis {
                    SmoothBasisSpec::Duchon { spec, .. } => spec.operator_penalties.clone(),
                    _ => gam_terms::basis::DuchonOperatorPenaltySpec::default(),
                };
                // Slow-path Duchon realization stores centers/collocation points
                // in standardized coordinates and compensates the user-facing
                // length_scale by σ_geom before building penalties. The n-free
                // re-key must use the same effective length scale, or the fast
                // path pairs G(ψ_new) with an S(ψ_new) from a different
                // coordinate scale.
                let effective_ls = match input_scales.as_deref() {
                    Some(scales) => {
                        compensate_optional_length_scale_for_standardization(ls_opt, scales)
                    }
                    None => ls_opt,
                };
                gam_terms::basis::duchon_penalties_at_length_scale(
                    centers.view(),
                    identifiability_transform.as_ref(),
                    operator_collocation_points.as_ref().map(|p| p.view()),
                    &operator_penalties,
                    *power,
                    *nullspace_order,
                    aniso_log_scales.as_deref(),
                    radial_reparam.as_ref(),
                    effective_ls,
                    &mut self.basisworkspace,
                )
                .map_err(|e| e.to_string())?
            }
            BasisMetadata::Matern {
                centers,
                periodic,
                nu,
                include_intercept,
                identifiability_transform,
                aniso_log_scales,
                input_scales,
                ..
            } => {
                // `spatial_term_psi_to_length_scale_and_aniso` decodes ψ to a
                // length scale in ORIGINAL data coordinates — exactly what the
                // slow-path rebuild writes into `spec.length_scale` before
                // `matern_operator_penalty_triplet_from_metadata` compensates it
                // by σ_geom. Compensate identically here so the n-free re-key
                // reproduces the slow-path penalty surface byte-for-byte (#706).
                let ls = ls_opt.ok_or_else(|| {
                    "Matérn n-free penalty re-key requires a finite length-scale".to_string()
                })?;
                let effective_ls = match input_scales.as_deref() {
                    Some(scales) => compensate_length_scale_for_standardization(ls, scales),
                    None => ls,
                };
                let aniso_for_penalty = aniso_from_psi.as_deref().or(aniso_log_scales.as_deref());
                // Route through the SAME canonical operator-triplet builder the
                // realized design uses (`matern_operator_penalty_triplet_from_
                // metadata`). The Matérn design ALWAYS uses this {mass, tension,
                // stiffness} triplet (see the Matérn penalty selection in
                // term_specs.rs; #1074 confirmed by MSI measurement that the RKHS
                // kernel penalty does not improve recovery and regresses the
                // high-frequency guard), so re-keying via the kernel path would
                // produce a 1-block surface against a 3-block frozen design — the
                // topology desync #1270 hard-errored on. Sharing the builder
                // makes the block count ψ-stable by construction.
                let (penalties, nullspace_dims, _info) =
                    matern_operator_penalty_triplet_at_length_scale(
                        centers.view(),
                        periodic.as_deref(),
                        identifiability_transform.as_ref(),
                        *nu,
                        *include_intercept,
                        aniso_for_penalty,
                        effective_ls,
                    )
                    .map_err(|e| e.to_string())?;
                (penalties, nullspace_dims)
            }
            BasisMetadata::ThinPlate {
                centers,
                identifiability_transform,
                radial_reparam,
                ..
            } => {
                let ls = ls_opt.ok_or_else(|| {
                    "thin-plate n-free penalty re-key requires a finite length-scale".to_string()
                })?;
                let double_penalty = match &termspec.basis {
                    SmoothBasisSpec::ThinPlate { spec, .. } => spec.double_penalty,
                    _ => false,
                };
                gam_terms::basis::thin_plate_penalties_at_length_scale(
                    centers.view(),
                    identifiability_transform.as_ref(),
                    radial_reparam.as_ref(),
                    ls,
                    double_penalty,
                    &mut self.basisworkspace,
                )
                .map_err(|e| e.to_string())?
            }
            other => {
                return Err(format!(
                    "n-free penalty re-key unsupported for basis metadata {:?}",
                    std::mem::discriminant(other)
                ));
            }
        };
        // The frozen collection penalties for THIS term are the templates whose
        // ψ-invariant structure (col_range / prior_mean / structure_hint / op)
        // we keep, swapping only the numeric `local`. For a single-spatial-term
        // collection the term owns the whole penalty list.
        let templates = &self.design.penalties;
        if templates.len() != locals.len() {
            return Err(format!(
                "n-free penalty re-key produced {} blocks but the frozen design carries {} \
                 — penalty topology is not ψ-stable",
                locals.len(),
                templates.len()
            ));
        }
        let specs: Vec<gam_solve::estimate::PenaltySpec> = templates
            .iter()
            .zip(locals.into_iter())
            .map(|(tmpl, local)| gam_solve::estimate::PenaltySpec::Block {
                local,
                col_range: tmpl.col_range.clone(),
                prior_mean: tmpl.prior_mean.clone(),
                structure_hint: tmpl.structure_hint.clone(),
                op: tmpl.op.clone(),
            })
            .collect();
        gam_terms::construction::canonicalize_penalty_specs(
            &specs,
            &nullspace_dims,
            p_total,
            "nfree-psi-penalty",
        )
        .map_err(|e| e.to_string())
    }

    fn canonical_penalty_derivatives_at_psi(
        &mut self,
        spatial_terms: &[usize],
        psi: &[f64],
    ) -> Result<(Range<usize>, usize, Vec<Array2<f64>>), String> {
        if spatial_terms.len() != 1 {
            return Err(format!(
                "n-free penalty derivative re-key requires exactly one spatial term, found {}",
                spatial_terms.len()
            ));
        }
        let term_idx = spatial_terms[0];
        let (ls_opt, aniso_from_psi) = spatial_term_psi_to_length_scale_and_aniso(psi);
        let termspec = self.spec.smooth_terms.get(term_idx).ok_or_else(|| {
            format!("spatial term {term_idx} out of range for n-free penalty derivative")
        })?;
        let term = self
            .design
            .smooth
            .terms
            .get(term_idx)
            .ok_or_else(|| format!("realized smooth term {term_idx} out of range"))?;
        let p_total = self.design.design.ncols();
        let smooth_start = p_total.saturating_sub(self.design.smooth.total_smooth_cols());
        let global_range =
            (smooth_start + term.coeff_range.start)..(smooth_start + term.coeff_range.end);

        let locals = match &term.metadata {
            BasisMetadata::Duchon {
                centers,
                identifiability_transform,
                operator_collocation_points,
                power,
                nullspace_order,
                aniso_log_scales,
                input_scales,
                radial_reparam,
                ..
            } => {
                let mut spec = match &termspec.basis {
                    SmoothBasisSpec::Duchon { spec, .. } => spec.clone(),
                    _ => {
                        return Err(
                            "Duchon n-free penalty derivative requires a Duchon term spec"
                                .to_string(),
                        );
                    }
                };
                let effective_ls = match input_scales.as_deref() {
                    Some(scales) => {
                        compensate_optional_length_scale_for_standardization(ls_opt, scales)
                    }
                    None => ls_opt,
                };
                spec.length_scale = effective_ls;
                spec.power = *power;
                spec.nullspace_order = *nullspace_order;
                spec.aniso_log_scales = aniso_log_scales.clone();
                // #1355: replay the frozen data-metric reparam so the n-free
                // penalty ψ-derivative matches the rotated forward penalty.
                spec.radial_reparam = radial_reparam.clone();
                if spec.length_scale.is_none() {
                    return Err(
                        "Duchon n-free penalty derivative requires a hybrid length-scale"
                            .to_string(),
                    );
                }
                let collocation = operator_collocation_points
                    .as_ref()
                    .map(|points| points.view())
                    .unwrap_or_else(|| centers.view());
                let (_native_sources, mut first, _native_second) =
                    gam_terms::basis::build_duchon_native_penalty_psi_derivatives(
                        centers.view(),
                        &spec,
                        identifiability_transform.as_ref(),
                        &mut self.basisworkspace,
                    )
                    .map_err(|e| e.to_string())?;
                let (_operator_sources, operator_first, _operator_second) =
                    gam_terms::basis::build_duchon_operator_penalty_psi_derivatives(
                        collocation,
                        centers.view(),
                        &spec,
                        identifiability_transform.as_ref(),
                        &mut self.basisworkspace,
                    )
                    .map_err(|e| e.to_string())?;
                first.extend(operator_first);
                first
            }
            BasisMetadata::Matern {
                centers,
                periodic,
                nu,
                include_intercept,
                identifiability_transform,
                aniso_log_scales,
                input_scales,
                ..
            } => {
                let ls = ls_opt.ok_or_else(|| {
                    "Matérn n-free penalty derivative requires a finite length-scale".to_string()
                })?;
                let effective_ls = match input_scales.as_deref() {
                    Some(scales) => compensate_length_scale_for_standardization(ls, scales),
                    None => ls,
                };
                let penalty_centers =
                    gam_terms::basis::expand_periodic_centers(&centers.to_owned(), periodic.as_deref())
                        .map_err(|e| e.to_string())?;
                let aniso_for_penalty = aniso_from_psi.as_deref().or(aniso_log_scales.as_deref());
                let (first, _second) = gam_terms::basis::build_matern_operator_penalty_psi_derivatives(
                    penalty_centers.view(),
                    effective_ls,
                    *nu,
                    *include_intercept,
                    identifiability_transform.as_ref(),
                    aniso_for_penalty,
                )
                .map_err(|e| e.to_string())?;
                first
            }
            BasisMetadata::ThinPlate {
                centers,
                identifiability_transform,
                radial_reparam,
                ..
            } => {
                let ls = ls_opt.ok_or_else(|| {
                    "thin-plate n-free penalty derivative requires a finite length-scale"
                        .to_string()
                })?;
                let mut spec = match &termspec.basis {
                    SmoothBasisSpec::ThinPlate { spec, .. } => spec.clone(),
                    _ => {
                        return Err(
                            "thin-plate n-free penalty derivative requires a ThinPlate term spec"
                                .to_string(),
                        );
                    }
                };
                spec.length_scale = ls;
                if spec.radial_reparam.is_none() {
                    spec.radial_reparam = radial_reparam.clone();
                }
                let (primary, _primary_second) =
                    gam_terms::basis::build_thin_plate_penalty_psi_derivativeswithworkspace(
                        centers.view(),
                        &spec,
                        identifiability_transform.as_ref(),
                        &mut self.basisworkspace,
                    )
                    .map_err(|e| e.to_string())?;
                if self.design.penalties.len() > 1 {
                    vec![primary.clone(), Array2::<f64>::zeros(primary.raw_dim())]
                } else {
                    vec![primary]
                }
            }
            other => {
                return Err(format!(
                    "n-free penalty derivative re-key unsupported for basis metadata {:?}",
                    std::mem::discriminant(other)
                ));
            }
        };
        if locals.len() != self.design.penalties.len() {
            return Err(format!(
                "n-free penalty derivative re-key produced {} blocks but the frozen design carries {} \
                 — penalty topology is not ψ-stable",
                locals.len(),
                self.design.penalties.len()
            ));
        }
        Ok((global_range, p_total, locals))
    }

    fn apply_log_kappa(
        &mut self,
        log_kappa: &SpatialLogKappaCoords,
        term_indices: &[usize],
    ) -> Result<(), String> {
        if term_indices.len() != log_kappa.dims_per_term().len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "incremental realizer log-kappa term mismatch: term_indices={}, dims_per_term={}",
                term_indices.len(),
                log_kappa.dims_per_term().len()
            ))
            .into());
        }

        let mut any_changed = false;
        for (slot, &term_idx) in term_indices.iter().enumerate() {
            any_changed |= self.apply_log_kappa_to_term(term_idx, log_kappa.term_slice(slot))?;
        }

        if any_changed {
            self.refresh_full_design_operator()?;
            rebuild_smooth_auxiliary_state(
                &mut self.design.smooth,
                &self.dropped_penaltyinfo_by_term,
            )?;
            rebuild_term_collection_auxiliary_state(&self.spec, &mut self.design)?;
            self.design_revision = self.design_revision.wrapping_add(1);
        }
        Ok(())
    }

    fn apply_log_kappa_to_term(&mut self, term_idx: usize, psi: &[f64]) -> Result<bool, String> {
        if !spatial_term_supports_hyper_optimization(&self.spec, term_idx) {
            return Err(SmoothError::invalid_config(format!(
                "incremental realizer term {term_idx} does not expose spatial hyperparameters"
            ))
            .into());
        }
        // Measure-jet ψ slots are dial coordinates, not log-κ (dial docs:
        // the MEASURE_JET_PSI_* bounds block); route through the dial setter
        // so the κ-translation below never misreads them as log-scales.
        let measure_jet_term = measure_jet_term_spec(&self.spec, term_idx).is_some();
        // Constant-curvature ψ is the raw signed curvature κ, NOT a log-scale;
        // route through the κ setter so `spatial_term_psi_to_length_scale_and_aniso`
        // never misreads it (and never hits the "no length scale" rejection).
        let constant_curvature_term = constant_curvature_term_spec(&self.spec, term_idx).is_some();
        let mut next_length_scale = None;
        let mut next_aniso: Option<Vec<f64>> = None;
        if measure_jet_term {
            if !set_measure_jet_psi_dials(&mut self.spec, term_idx, psi)
                .map_err(|e| e.to_string())?
            {
                return Ok(false);
            }
        } else if constant_curvature_term {
            if !set_constant_curvature_kappa(&mut self.spec, term_idx, psi)
                .map_err(|e| e.to_string())?
            {
                return Ok(false);
            }
        } else {
            let current_length_scale = get_spatial_length_scale(&self.spec, term_idx);
            let current_aniso = get_spatial_aniso_log_scales(&self.spec, term_idx);
            let (ls, eta) = spatial_term_psi_to_length_scale_and_aniso(psi);
            next_length_scale = ls;
            next_aniso = eta;
            let same_length = spatial_length_scale_matches(current_length_scale, next_length_scale);
            let same_aniso = spatial_aniso_matches(current_aniso.as_deref(), next_aniso.as_deref());
            if same_length && same_aniso {
                return Ok(false);
            }
            if let Some(length_scale) = next_length_scale {
                set_spatial_length_scale(&mut self.spec, term_idx, length_scale)
                    .map_err(|e| e.to_string())?;
            }
            if let Some(eta) = next_aniso.clone() {
                set_spatial_aniso_log_scales(&mut self.spec, term_idx, eta)
                    .map_err(|e| e.to_string())?;
            }
        }

        // Pick the spec to drive the rebuild. If the per-term geometry cache
        // is populated, it carries already-resolved centers
        // (`CenterStrategy::UserProvided`) and frozen `input_scales`; reusing
        // it short-circuits `select_centers_by_strategy` (KMeans /
        // FarthestPoint / EqualMass cluster searches) and
        // `compute_spatial_input_scales` (per-axis variance over n rows) in
        // the family builders. Centers in the cached spec live in
        // standardized coordinates (matching the cached `input_scales`), so
        // the same standardization + kernel path runs without recomputation
        // of the geometry.
        let geometry_slot = self
            .spatial_realization_geometry
            .get(term_idx)
            .ok_or_else(|| format!("incremental realizer geometry slot {term_idx} out of range"))?;
        let mut build_spec = match geometry_slot {
            Some(cached) => cached.clone(),
            None => self
                .spec
                .smooth_terms
                .get(term_idx)
                .ok_or_else(|| format!("incremental realizer smooth term {term_idx} out of range"))?
                .clone(),
        };
        if measure_jet_term {
            // The cached build spec carries the frozen geometry (UserProvided
            // barycenter nodes, frozen quadrature + transform); only the
            // dials move per trial.
            set_single_term_measure_jet_psi_dials(&mut build_spec, psi)
                .map_err(|e| e.to_string())?;
        } else if constant_curvature_term {
            // The cached build spec carries the κ-fixed geometry (UserProvided
            // centers, frozen ℓ and constraint transform); only κ moves per
            // trial, written through the raw-κ setter to match the collection
            // write-back above.
            set_single_term_constant_curvature_kappa(&mut build_spec, psi)
                .map_err(|e| e.to_string())?;
        } else {
            if let Some(length_scale) = next_length_scale {
                set_single_term_spatial_length_scale(&mut build_spec, length_scale)
                    .map_err(|e| e.to_string())?;
            }
            if let Some(eta) = next_aniso {
                set_single_term_spatial_aniso_log_scales(&mut build_spec, eta)
                    .map_err(|e| e.to_string())?;
            }
        }

        let termname = build_spec.name.clone();
        let local = build_single_local_smooth_term(
            self.data,
            &build_spec,
            &mut self.basisworkspace,
        )
        .map_err(|e| {
            format!(
                "failed to rebuild smooth term '{termname}' during incremental κ realization: {e}"
            )
        })?;

        // Populate the geometry cache from the realized metadata on first use.
        // Family auto-promotion (ThinPlate -> Duchon) is detected as a basis /
        // metadata mismatch in `freeze_geometry_from_metadata` and leaves the
        // cache empty so the next call re-tries with the (now stable) family.
        if self.spatial_realization_geometry[term_idx].is_none()
            && let Some(frozen) = freeze_geometry_from_metadata(&build_spec, &local.metadata)
        {
            // Mirror the frozen identifiability (pinned `Z` + double-penalty
            // nullspace-shrinkage decision, #787/#860/#1122) back onto the
            // collection spec the analytic ψ-gradient reads
            // (`try_build_spatial_log_kappa_hyper_dirs(self.spec(), …)`). The
            // value rebuild consumes the cached `build_spec`, so without this
            // copy the gradient would keep re-running the κ-DEPENDENT spectral
            // test on the un-frozen collection spec while the value uses the
            // frozen decision — re-introducing the very objective↔gradient
            // desync the freeze removes. Pinning both to the same frozen
            // transform keeps the per-trial value and its analytic gradient on
            // one fixed `Z` and one fixed null dimension `r`.
            if let (
                SmoothBasisSpec::Matern {
                    spec: frozen_spec, ..
                },
                Some(SmoothBasisSpec::Matern {
                    spec: live_spec, ..
                }),
            ) = (
                &frozen.basis,
                self.spec
                    .smooth_terms
                    .get_mut(term_idx)
                    .map(|t| &mut t.basis),
            ) {
                live_spec.identifiability = frozen_spec.identifiability.clone();
                live_spec.center_strategy = frozen_spec.center_strategy.clone();
            }
            self.spatial_realization_geometry[term_idx] = Some(frozen);
        }

        let realization = wrap_local_build_as_realization(local, &build_spec)?;
        self.replace_term_realization(term_idx, realization)?;
        Ok(true)
    }

    fn replace_term_realization(
        &mut self,
        term_idx: usize,
        realization: SingleSmoothTermRealization,
    ) -> Result<(), String> {
        let t_replace = std::time::Instant::now();
        let SingleSmoothTermRealization {
            design_local,
            term,
            dropped_penaltyinfo,
        } = realization;
        let SmoothTerm {
            name,
            penalties_local,
            nullspace_dims,
            penaltyinfo_local,
            metadata,
            lower_bounds_local,
            linear_constraints_local,
            joint_null_rotation,
            ..
        } = term;
        let coeff_range = self
            .design
            .smooth
            .terms
            .get(term_idx)
            .ok_or_else(|| format!("incremental realizer smooth term {term_idx} out of range"))?
            .coeff_range
            .clone();
        if design_local.ncols() != coeff_range.len() {
            return Err(SmoothError::dimension_mismatch(format!(
                "incremental realizer width mismatch for term {}: rebuilt_cols={}, cached_cols={}",
                term_idx,
                design_local.ncols(),
                coeff_range.len()
            ))
            .into());
        }
        if design_local.nrows() != self.design.design.nrows() {
            return Err(SmoothError::dimension_mismatch(format!(
                "incremental realizer row mismatch for term {}: rebuilt_rows={}, design_rows={}",
                term_idx,
                design_local.nrows(),
                self.design.design.nrows()
            ))
            .into());
        }

        let active_penaltyinfo = penaltyinfo_local
            .iter()
            .filter(|info| info.active)
            .cloned()
            .collect::<Vec<_>>();
        let smooth_penalty_range = self
            .smooth_penalty_ranges
            .get(term_idx)
            .ok_or_else(|| {
                format!("incremental realizer missing smooth penalty range for term {term_idx}")
            })?
            .clone();
        let full_penalty_range = self
            .full_penalty_ranges
            .get(term_idx)
            .ok_or_else(|| {
                format!("incremental realizer missing full penalty range for term {term_idx}")
            })?
            .clone();
        if active_penaltyinfo.len() != smooth_penalty_range.len()
            || penalties_local.len() != smooth_penalty_range.len()
            || nullspace_dims.len() != smooth_penalty_range.len()
        {
            return Err(SmoothError::dimension_mismatch(format!(
                "incremental realizer topology changed for term '{}': penalties={}, infos={}, nullspaces={}, cached_penalties={}",
                name,
                penalties_local.len(),
                active_penaltyinfo.len(),
                nullspace_dims.len(),
                smooth_penalty_range.len()
            ))
            .into());
        }

        self.design.smooth.term_designs[term_idx] = design_local;

        for (offset, penalty_local) in penalties_local.iter().enumerate() {
            let smooth_penalty_idx = smooth_penalty_range.start + offset;
            let full_penalty_idx = full_penalty_range.start + offset;
            let nullspace_dim = nullspace_dims[offset];
            let penalty_info = active_penaltyinfo[offset].clone();

            if penalty_local.nrows() != coeff_range.len()
                || penalty_local.ncols() != coeff_range.len()
            {
                return Err(SmoothError::dimension_mismatch(format!(
                    "incremental realizer penalty shape mismatch for term '{}' penalty {}: \
                     penalty is {}x{} but coeff_range has {} columns",
                    name,
                    offset,
                    penalty_local.nrows(),
                    penalty_local.ncols(),
                    coeff_range.len()
                ))
                .into());
            }

            let smooth_penalty = self
                .design
                .smooth
                .penalties
                .get_mut(smooth_penalty_idx)
                .ok_or_else(|| {
                    format!(
                        "incremental realizer smooth penalty {} out of range for term {}",
                        smooth_penalty_idx, term_idx
                    )
                })?;
            // With per-term block-local penalties, col_range already targets
            // this specific term, so .local is p_k × p_k.
            smooth_penalty.local.assign(penalty_local);

            let full_bp = self
                .design
                .penalties
                .get_mut(full_penalty_idx)
                .ok_or_else(|| {
                    format!(
                        "incremental realizer full penalty {} out of range for term {}",
                        full_penalty_idx, term_idx
                    )
                })?;
            // With per-term block-local penalties, col_range already targets
            // this specific term, so .local is p_k × p_k.
            full_bp.local.assign(penalty_local);

            self.design.smooth.nullspace_dims[smooth_penalty_idx] = nullspace_dim;
            self.design.nullspace_dims[full_penalty_idx] = nullspace_dim;

            self.design.smooth.penaltyinfo[smooth_penalty_idx].global_index = smooth_penalty_idx;
            self.design.smooth.penaltyinfo[smooth_penalty_idx].termname = Some(name.clone());
            self.design.smooth.penaltyinfo[smooth_penalty_idx].penalty = penalty_info.clone();

            self.design.penaltyinfo[full_penalty_idx].global_index = full_penalty_idx;
            self.design.penaltyinfo[full_penalty_idx].termname = Some(name.clone());
            self.design.penaltyinfo[full_penalty_idx].penalty = penalty_info;
        }

        let target_term = self.design.smooth.terms.get_mut(term_idx).ok_or_else(|| {
            format!("incremental realizer smooth term {term_idx} disappeared during replacement")
        })?;
        target_term.penalties_local = penalties_local;
        target_term.nullspace_dims = nullspace_dims;
        target_term.penaltyinfo_local = penaltyinfo_local;
        target_term.metadata = metadata;
        target_term.lower_bounds_local = lower_bounds_local;
        target_term.linear_constraints_local = linear_constraints_local;
        target_term.joint_null_rotation = joint_null_rotation;
        self.dropped_penaltyinfo_by_term[term_idx] = dropped_penaltyinfo;
        log::info!(
            "[STAGE] smooth basis rebuild (term {}, '{}', cols={}): {:.3}s",
            term_idx,
            target_term.name,
            coeff_range.len(),
            t_replace.elapsed().as_secs_f64(),
        );
        Ok(())
    }

    fn refresh_full_design_operator(&mut self) -> Result<(), String> {
        let mut blocks = Vec::<DesignBlock>::with_capacity(
            self.fixed_blocks.len() + self.design.smooth.term_designs.len(),
        );
        blocks.extend(self.fixed_blocks.iter().cloned());
        for term_design in &self.design.smooth.term_designs {
            blocks.push(DesignBlock::from(term_design));
        }
        self.design.design = assemble_term_collection_design_matrix(blocks)
            .map_err(|e| format!("failed to refresh term-collection design: {e}"))?;
        Ok(())
    }
}

fn build_term_collection_fixed_blocks(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<Vec<DesignBlock>, BasisError> {
    let mut blocks = Vec::<DesignBlock>::new();
    if !term_collection_has_one_sided_anchored_bspline(spec) {
        blocks.push(DesignBlock::Intercept(data.nrows()));
    }

    if !spec.linear_terms.is_empty() {
        let mut linear_block = Array2::<f64>::zeros((data.nrows(), spec.linear_terms.len()));
        for (j, linear) in spec.linear_terms.iter().enumerate() {
            // Single shared realizer: numeric product gated by any
            // categorical-level indicators (factor-aware `:` interaction),
            // mirroring `build_term_collection_design_inner`.
            let column = linear
                .realized_design_column(data)
                .map_err(BasisError::InvalidInput)?;
            linear_block.column_mut(j).assign(&column);
        }
        blocks.push(DesignBlock::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            linear_block,
        )));
    }

    for term in &spec.random_effect_terms {
        let block = build_random_effect_block(data, term)?;
        let re_op = RandomEffectOperator::new(block.group_ids, block.num_groups);
        blocks.push(DesignBlock::RandomEffect(Arc::new(re_op)));
    }

    Ok(blocks)
}

// ---------------------------------------------------------------------------
// N-block spatial length-scale optimizer.
// ---------------------------------------------------------------------------

pub struct SpatialLengthScaleOptimizationResult<FitOut> {
    pub resolved_specs: Vec<TermCollectionSpec>,
    pub designs: Vec<TermCollectionDesign>,
    pub fit: FitOut,
    pub timing: Option<SpatialLengthScaleOptimizationTiming>,
}

/// Exact-joint hyper-parameter setup for N-block spatial length-scale optimization.
#[derive(Debug, Clone)]
pub struct ExactJointHyperSetup {
    rho0: Array1<f64>,
    rho_lower: Array1<f64>,
    rho_upper: Array1<f64>,
    log_kappa0: SpatialLogKappaCoords,
    log_kappa_lower: SpatialLogKappaCoords,
    log_kappa_upper: SpatialLogKappaCoords,
    auxiliary0: Array1<f64>,
    auxiliary_lower: Array1<f64>,
    auxiliary_upper: Array1<f64>,
}

impl ExactJointHyperSetup {
    fn sanitize_rho_seed(
        rho0: Array1<f64>,
        rho_lower: &Array1<f64>,
        rho_upper: &Array1<f64>,
    ) -> Array1<f64> {
        Array1::from_iter(rho0.iter().enumerate().map(|(idx, &value)| {
            let lo = rho_lower[idx];
            let hi = rho_upper[idx];
            let fallback = 0.0_f64.clamp(lo, hi);
            if value.is_finite() {
                value.clamp(lo, hi)
            } else {
                fallback
            }
        }))
    }

    pub(crate) fn new(
        rho0: Array1<f64>,
        rho_lower: Array1<f64>,
        rho_upper: Array1<f64>,
        log_kappa0: SpatialLogKappaCoords,
        log_kappa_lower: SpatialLogKappaCoords,
        log_kappa_upper: SpatialLogKappaCoords,
    ) -> Self {
        let rho0 = Self::sanitize_rho_seed(rho0, &rho_lower, &rho_upper);
        Self {
            rho0,
            rho_lower,
            rho_upper,
            log_kappa0,
            log_kappa_lower,
            log_kappa_upper,
            auxiliary0: Array1::zeros(0),
            auxiliary_lower: Array1::zeros(0),
            auxiliary_upper: Array1::zeros(0),
        }
    }

    pub(crate) fn with_auxiliary(
        mut self,
        auxiliary0: Array1<f64>,
        auxiliary_lower: Array1<f64>,
        auxiliary_upper: Array1<f64>,
    ) -> Self {
        assert_eq!(
            auxiliary0.len(),
            auxiliary_lower.len(),
            "auxiliary lower bound length mismatch"
        );
        assert_eq!(
            auxiliary0.len(),
            auxiliary_upper.len(),
            "auxiliary upper bound length mismatch"
        );
        self.auxiliary0 = Self::sanitize_rho_seed(auxiliary0, &auxiliary_lower, &auxiliary_upper);
        self.auxiliary_lower = auxiliary_lower;
        self.auxiliary_upper = auxiliary_upper;
        self
    }

    pub(crate) fn rho_dim(&self) -> usize {
        self.rho0.len()
    }

    pub(crate) fn log_kappa_dim(&self) -> usize {
        self.log_kappa0.len()
    }

    pub(crate) fn auxiliary_dim(&self) -> usize {
        self.auxiliary0.len()
    }

    pub(crate) fn theta0(&self) -> Array1<f64> {
        let mut out =
            Array1::<f64>::zeros(self.rho_dim() + self.log_kappa_dim() + self.auxiliary_dim());
        out.slice_mut(s![..self.rho_dim()]).assign(&self.rho0);
        out.slice_mut(s![self.rho_dim()..self.rho_dim() + self.log_kappa_dim()])
            .assign(self.log_kappa0.as_array());
        out.slice_mut(s![self.rho_dim() + self.log_kappa_dim()..])
            .assign(&self.auxiliary0);
        out
    }

    pub(crate) fn lower(&self) -> Array1<f64> {
        let mut out =
            Array1::<f64>::zeros(self.rho_dim() + self.log_kappa_dim() + self.auxiliary_dim());
        out.slice_mut(s![..self.rho_dim()]).assign(&self.rho_lower);
        out.slice_mut(s![self.rho_dim()..self.rho_dim() + self.log_kappa_dim()])
            .assign(self.log_kappa_lower.as_array());
        out.slice_mut(s![self.rho_dim() + self.log_kappa_dim()..])
            .assign(&self.auxiliary_lower);
        out
    }

    pub(crate) fn upper(&self) -> Array1<f64> {
        let mut out =
            Array1::<f64>::zeros(self.rho_dim() + self.log_kappa_dim() + self.auxiliary_dim());
        out.slice_mut(s![..self.rho_dim()]).assign(&self.rho_upper);
        out.slice_mut(s![self.rho_dim()..self.rho_dim() + self.log_kappa_dim()])
            .assign(self.log_kappa_upper.as_array());
        out.slice_mut(s![self.rho_dim() + self.log_kappa_dim()..])
            .assign(&self.auxiliary_upper);
        out
    }

    /// Per-term dimensionality layout for the psi block.
    pub(crate) fn log_kappa_dims_per_term(&self) -> Vec<usize> {
        self.log_kappa0.dims_per_term().to_vec()
    }
}

/// N-block design cache for exact-joint spatial length-scale optimization.
///
/// Each block owns a `FrozenTermCollectionIncrementalRealizer` and a list of
/// spatial term indices within that block's spec. The cache splits the
/// combined psi vector into per-block slices using precomputed offsets.
struct ExactJointDesignCache<'d> {
    realizers: Vec<FrozenTermCollectionIncrementalRealizer<'d>>,
    block_term_indices: Vec<Vec<usize>>,
    current_theta: Option<Array1<f64>>,
    last_cost: Option<f64>,
    last_eval: Option<(
        f64,
        Array1<f64>,
        gam_problem::HessianResult,
    )>,
    rho_dim: usize,
    all_dims: Vec<usize>,
    log_kappa_dim: usize,
    block_term_counts: Vec<usize>,
}

impl<'d> ExactJointDesignCache<'d> {
    fn new(
        data: ArrayView2<'d, f64>,
        blocks: Vec<(TermCollectionSpec, TermCollectionDesign, Vec<usize>)>,
        rho_dim: usize,
        all_dims: Vec<usize>,
    ) -> Result<Self, String> {
        let n_blocks = blocks.len();
        let mut realizers = Vec::with_capacity(n_blocks);
        let mut block_term_indices = Vec::with_capacity(n_blocks);
        let mut block_term_counts = Vec::with_capacity(n_blocks);

        for (spec, design, terms) in blocks {
            block_term_counts.push(terms.len());
            block_term_indices.push(terms);
            realizers.push(FrozenTermCollectionIncrementalRealizer::new(
                data, spec, design,
            )?);
        }

        Ok(Self {
            realizers,
            block_term_indices,
            current_theta: None,
            last_cost: None,
            last_eval: None,
            rho_dim,
            log_kappa_dim: all_dims.iter().sum(),
            all_dims,
            block_term_counts,
        })
    }

    fn ensure_theta(&mut self, theta: &Array1<f64>) -> Result<(), String> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            return Ok(());
        }

        let t_ensure = std::time::Instant::now();
        let kappa_theta_len = self.rho_dim + self.log_kappa_dim;
        if theta.len() < kappa_theta_len {
            return Err(SmoothError::dimension_mismatch(format!(
                "exact-joint theta length mismatch: got {}, expected at least {} (rho_dim={}, log_kappa_dim={})",
                theta.len(),
                kappa_theta_len,
                self.rho_dim,
                self.log_kappa_dim
            ))
            .into());
        }
        let theta_kappa = theta.slice(s![..kappa_theta_len]).to_owned();
        let full_log_kappa = SpatialLogKappaCoords::from_theta_tail_with_dims(
            &theta_kappa,
            self.rho_dim,
            self.all_dims.clone(),
        );

        // Split the full log_kappa into per-block sub-coords using split_at.
        // We split from the front iteratively: after extracting block 0..N-2,
        // the remainder is the last block.
        let n = self.realizers.len();
        let mut remaining = full_log_kappa;
        for block_idx in 0..n {
            let count = self.block_term_counts[block_idx];
            if block_idx < n - 1 {
                let (block_lk, rest) = remaining.split_at(count);
                self.realizers[block_idx]
                    .apply_log_kappa(&block_lk, &self.block_term_indices[block_idx])?;
                remaining = rest;
            } else {
                // Last block gets the remainder.
                self.realizers[block_idx]
                    .apply_log_kappa(&remaining, &self.block_term_indices[block_idx])?;
            }
        }

        log::info!(
            "[STAGE] ensure_theta (n-block, {} blocks, {} realizers): {:.3}s",
            n,
            self.realizers.len(),
            t_ensure.elapsed().as_secs_f64(),
        );
        self.current_theta = Some(theta.clone());
        self.last_cost = None;
        self.last_eval = None;
        Ok(())
    }

    impl_exact_joint_theta_memo!();

    /// Cache a cost-only result. Called after `ensure_theta(theta)` for
    /// line-search probes that pay only for the cost evaluation. We
    /// intentionally do not populate `last_eval` because no gradient was
    /// computed; the next outer evaluation at this θ will recompute
    /// (V, ∇V) via `evaluate_with_order` if the optimizer asks for it.
    fn store_cost_only(&mut self, theta: &Array1<f64>, cost: f64) {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
        {
            self.last_cost = Some(cost);
        }
    }

    fn specs(&self) -> Vec<&TermCollectionSpec> {
        self.realizers.iter().map(|r| r.spec()).collect()
    }

    fn designs(&self) -> Vec<&TermCollectionDesign> {
        self.realizers.iter().map(|r| r.design()).collect()
    }

    /// Combined monotonic design revision across all per-block realizers.
    ///
    /// Mirrors `SingleBlockExactJointDesignCache::design_revision` for the
    /// n-block exact-joint path. Each realizer's `design_revision` counter
    /// advances iff `apply_log_kappa` actually rebuilt that block's realized
    /// design / smooth penalties; the wrapping sum therefore changes iff
    /// *any* block rebuilt. Equal values across two calls imply no realizer
    /// has been rebuilt in between, which is the invariant the
    /// `ExternalJointHyperEvaluator` canonical-penalty fast path needs.
    fn design_revision(&self) -> u64 {
        self.realizers
            .iter()
            .fold(0u64, |acc, r| acc.wrapping_add(r.design_revision()))
    }
}

pub(crate) fn seed_risk_profile_for_likelihood_family(
    family: &LikelihoodSpec,
) -> gam_problem::SeedRiskProfile {
    match &family.response {
        ResponseFamily::Gaussian => gam_problem::SeedRiskProfile::Gaussian,
        ResponseFamily::RoystonParmar => gam_problem::SeedRiskProfile::Survival,
        ResponseFamily::Binomial
        | ResponseFamily::Poisson
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Beta { .. }
        | ResponseFamily::Gamma => gam_problem::SeedRiskProfile::GeneralizedLinear,
    }
}

/// Joint-θ dimension above which the single-block exact-joint driver routes
/// gradient-only (this doc owns the derivation; the routing site only
/// compares against it). The exact outer Hessian builds θ(θ+1)/2 pairwise
/// hyper operators, so per-eval cost grows quadratically in θ-dim —
/// profiled: `TauTauPairHyperOperator::mul_vec` dominates wall-clock at
/// spectral-mode measure-jet candidate counts (θ ≈ 9–11), while θ ≤ 8
/// (classic Matérn κ/η fits) keeps cheap exact second-order geometry.
const EXACT_JOINT_SECOND_ORDER_THETA_CAP: usize = 8;

fn exact_joint_seed_config(
    risk_profile: gam_problem::SeedRiskProfile,
    auxiliary_dim: usize,
) -> gam_problem::SeedConfig {
    let mut config = gam_problem::SeedConfig {
        risk_profile,
        num_auxiliary_trailing: auxiliary_dim,
        ..Default::default()
    };
    match risk_profile {
        gam_problem::SeedRiskProfile::Gaussian
        | gam_problem::SeedRiskProfile::GaussianLocationScale => {
            config.max_seeds = 4;
            config.seed_budget = 2;
        }
        gam_problem::SeedRiskProfile::GeneralizedLinear => {
            // Bernoulli marginal-slope Matérn fits use the exact-joint spatial
            // driver rather than the family-local BMS outer. Mirror BMS proper:
            // screen one principled heuristic seed deeply enough to reach the
            // KKT basin instead of spending minutes screening equivalent starts.
            config.max_seeds = 1;
            config.seed_budget = 1;
            config.screen_max_inner_iterations = 8;
        }
        gam_problem::SeedRiskProfile::Survival => {
            // Survival marginal-slope has an additional time/hazard block and
            // is the most sensitive Matérn startup regime. Keep more of the
            // coherent SPDE candidate manifold alive through truncation and
            // validate enough starts that one bad transient does not report
            // "no candidate seeds" before reaching a viable basin.
            config.max_seeds = 8;
            config.seed_budget = 4;
            config.screen_max_inner_iterations = 8;
        }
    }
    config
}

#[cfg(test)]
mod exact_joint_seed_config_tests {
    use super::*;

    #[test]
    fn exact_joint_marginal_slope_profiles_get_deeper_startup_validation() {
        let bms = exact_joint_seed_config(gam_problem::SeedRiskProfile::GeneralizedLinear, 2);
        assert_eq!(bms.max_seeds, 1);
        assert_eq!(bms.seed_budget, 1);
        assert_eq!(bms.screen_max_inner_iterations, 8);
        assert_eq!(bms.num_auxiliary_trailing, 2);

        let survival = exact_joint_seed_config(gam_problem::SeedRiskProfile::Survival, 3);
        assert_eq!(survival.max_seeds, 8);
        assert_eq!(survival.seed_budget, 4);
        assert_eq!(survival.screen_max_inner_iterations, 8);
        assert_eq!(survival.num_auxiliary_trailing, 3);
    }

    #[test]
    fn exact_joint_gaussian_keeps_tight_historical_multistart_budget() {
        let gaussian = exact_joint_seed_config(gam_problem::SeedRiskProfile::Gaussian, 1);
        assert_eq!(gaussian.max_seeds, 4);
        assert_eq!(gaussian.seed_budget, 2);
        assert_eq!(
            gaussian.screen_max_inner_iterations,
            gam_problem::SeedConfig::default().screen_max_inner_iterations
        );
        assert_eq!(gaussian.num_auxiliary_trailing, 1);
    }
}

#[cfg(test)]
mod wood_reference_df_tests {
    use super::*;

    // The whole-term LR reference d.f. (#1766). `wood_reference_df` returns
    // Wood's smoothing-selection-corrected `edf1 = 2·tr(F) − tr(F²)` on the
    // coefficient-influence block, NOT the earlier Satterthwaite ratio
    // `tr(F)²/tr(F²)` that collapsed toward 0 for a boundary-shrunk term.

    #[test]
    fn edf1_equals_two_trace_minus_trace_of_square() {
        // A symmetric smoother block with eigenvalues {0.9, 0.4}: a partially
        // shrunk penalized term. edf = tr = 1.3; tr(F²) = 0.81 + 0.16 = 0.97;
        // edf1 = 2·1.3 − 0.97 = 1.63. (Diagonal ⇒ block F² trace = Σ λ².)
        let f = ndarray::array![[0.9_f64, 0.0], [0.0, 0.4]];
        let got = wood_reference_df(Some(&f), &(0..2)).unwrap();
        assert!(
            (got - 1.63).abs() < 1e-12,
            "edf1 should be 2*tr - tr(F^2) = 1.63, got {got}"
        );
        // And it must dominate the raw edf (edf1 >= edf) — the invariant the LR
        // reference relies on.
        let edf = 1.3;
        assert!(got >= edf - 1e-12, "edf1 {got} must be >= edf {edf}");
    }

    #[test]
    fn edf1_never_collapses_below_edf_when_offdiagonals_blow_up() {
        // The #1766 degeneracy: a NON-symmetric influence block whose
        // off-diagonal coupling makes tr(F²) = Σ_ij F_ij F_ji run away, so the
        // old Satterthwaite ratio tr(F)²/tr(F²) crashed toward 0. Here tr = 1.0
        // but tr(F²) = 1·1 + 50·(-50) ... take F with large opposite-sign
        // off-diagonals so tr(F²) is huge: edf1 = 2·tr − tr(F²) would go very
        // negative, and the `.max(tr)` guard must floor it back at edf = tr.
        let f = ndarray::array![[0.5_f64, 40.0], [40.0, 0.5]];
        let tr = 1.0_f64;
        let got = wood_reference_df(Some(&f), &(0..2)).unwrap();
        assert!(
            got >= tr - 1e-12,
            "edf1 must be floored at edf (=tr={tr}) even when tr(F^2) explodes, got {got}"
        );
        assert!(got.is_finite() && got > 0.0, "edf1 must stay finite/positive");
    }

    #[test]
    fn returns_none_on_nonpositive_or_missing_trace() {
        // No influence matrix at all → None (caller falls back to the
        // max(edf, null_dim, 1) floor).
        assert!(wood_reference_df(None, &(0..2)).is_none());
        // A fully-shrunk block with a non-positive trace → None.
        let zero = ndarray::array![[0.0_f64, 0.0], [0.0, 0.0]];
        assert!(wood_reference_df(Some(&zero), &(0..2)).is_none());
        // An out-of-bounds range → None, never a panic.
        let f = ndarray::array![[0.5_f64, 0.0], [0.0, 0.5]];
        assert!(wood_reference_df(Some(&f), &(0..5)).is_none());
    }
}

pub(crate) fn exact_joint_multistart_outer_problem(
    theta0: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    rho_dim: usize,
    auxiliary_dim: usize,
    n_params: usize,
    gradient: gam_problem::Derivative,
    hessian: gam_problem::DeclaredHessianForm,
    prefer_gradient_only: bool,
    disable_fixed_point: bool,
    risk_profile: gam_problem::SeedRiskProfile,
    tolerance: f64,
    max_iter: usize,
    // BFGS step caps split by parameter type. `bfgs_step_cap` (rho-axis cap)
    // bounds first-trial moves on log-λ; documented natural step is ≈ 5.
    // `bfgs_step_cap_psi` bounds moves on the trailing `auxiliary_dim`
    // psi-axes (kappa / aniso-log-scales), where ≈ ln 2 keeps the kernel
    // scale from oscillating across orders of magnitude per iter. Using a
    // single uniform cap (the old API) starved rho on the survival-marg-slope
    // joint solver because the psi-calibrated value (`ln 2 ≈ 0.69`) was
    // applied to log-λ, where |d|≈5 is the natural quasi-Newton magnitude.
    bfgs_step_cap: Option<f64>,
    bfgs_step_cap_psi: Option<f64>,
    screening_cap: Option<Arc<AtomicUsize>>,
    // `Some((n_obs, p_cols))` calibrates the outer solver to the n-scaled
    // profiled REML/LAML criterion exactly as the primary REML outer
    // (`solver/estimate.rs`) does. The profiled criterion is a sum over the n
    // observations, so its magnitude is O(n) (|f| ~ thousands at n ~ 10³) for
    // EVERY family — Gaussian, binomial, GP/kriging alike. A scale-blind outer
    // takes the bare `tolerance` (≈1e-6) as the *absolute* projected-gradient
    // floor, which is hopelessly tight against an n-scaled gradient: in-basin
    // iterates (e.g. ‖g‖≈7e-2 at |f|≈17, or single-digit ‖g‖ at |f|≈1.3e3)
    // never clear it and the fit bails at the iteration cap. Worse, ARC's
    // trust-region reduction ratios and default initial regularization are
    // referenced against the wrong curvature magnitude, so the first step can
    // overshoot and diverge (the ‖g‖≈½|f| blow-ups in #1053/#1066). Threading
    // the scale (→ absolute floor = max(tol, n·1e-9)) plus a warm ARC
    // regularization (σ₀ = 0.25) and operator trust radius (4.0) makes the
    // spatial exact-joint outer converge as robustly as the primary REML outer
    // across 1-D Matérn (#1053), 2-D binomial geo (#1066), and GP/kriging
    // (#1069). This is NOT a loosening of the `τ·(1+|f|)` REML acceptance gate
    // — that relative-to-cost criterion is unchanged; only the nonsensical
    // scale-free *absolute* floor and the solver's curvature reference are
    // corrected. `None` preserves the prior scale-free calibration.
    profiled_objective_size: Option<(usize, usize)>,
    // #1464: `true` when the fit carries a constant-curvature `curv()` term. Its
    // geodesic-exponential kernel collapses toward the constant function on the
    // +κ side, so the joint REML optimum there is a LARGE smoothing λ beyond the
    // historical ±12 ρ box. For that case the over-smoothing ρ ceiling is widened
    // to `RHO_BOUND` and an explicit high-ρ over-smoothing multistart probe is
    // seeded so the joint ARC can reach that basin. `false` keeps the historical
    // ±12 box and seed grid byte-for-byte for every other spatial/Matérn/Duchon/
    // sphere/survival joint fit.
    has_constant_curvature: bool,
) -> gam_solve::rho_optimizer::OuterProblem {
    let mut seed_heuristic = theta0.to_vec();
    for value in &mut seed_heuristic[..rho_dim] {
        *value = value.exp();
    }
    // Over-smoothing ρ ceiling: widened only for a constant-curvature fit (see
    // the `has_constant_curvature` param doc). Drives both the scalar saturation
    // reference and the seed-grid clamp; the actual box is the per-dim
    // `lower`/`upper` arrays passed in.
    let rho_ceiling = if has_constant_curvature {
        gam_solve::estimate::RHO_BOUND
    } else {
        12.0
    };
    let mut problem = gam_solve::rho_optimizer::OuterProblem::new(n_params)
        .with_gradient(gradient)
        .with_hessian(hessian)
        .with_prefer_gradient_only(prefer_gradient_only)
        .with_disable_fixed_point(disable_fixed_point)
        // Re-enable the automatic fallback ladder for exact joint spatial
        // problems. It was previously `Disabled` to suppress a geo-bench
        // fallback bug where HybridEFS ψ stagnation degraded silently to
        // BfgsApprox on a Charbonnier surface. With the ψ-stagnation guard
        // in OuterFixedPointBridge (`MAX_CONSECUTIVE_PSI_STAGNATION`) the
        // bridge now surfaces `EFS_FIRST_ORDER_FALLBACK_MARKER` when ψ
        // stationarity cannot be enforced, so the ladder routes correctly
        // to a joint gradient-based solver instead of grinding HybridEFS
        // for thousands of iterations.
        .with_fallback_policy(gam_solve::rho_optimizer::FallbackPolicy::Automatic)
        .with_psi_dim(auxiliary_dim)
        .with_tolerance(tolerance)
        .with_max_iter(max_iter)
        .with_bounds(lower.clone(), upper.clone())
        .with_initial_rho(theta0.clone())
        .with_bfgs_step_cap(bfgs_step_cap)
        .with_bfgs_step_cap_psi(bfgs_step_cap_psi)
        .with_seed_config({
            let mut sc = exact_joint_seed_config(risk_profile, auxiliary_dim);
            if has_constant_curvature {
                // Let the seed grid reach the widened over-smoothing ceiling so a
                // smooth whose true REML optimum genuinely lives at large λ can be
                // discovered (#1464).
                sc.bounds = (sc.bounds.0, rho_ceiling);
                // gam#1464: do NOT inject an explicit over-smoothing probe at
                // ρ ≈ +15 for constant-curvature terms. The probe seeds the joint
                // [ρ, ψ] solve at the collapsed-kernel corner where the geodesic
                // exponential exp(−d_κ/L) degenerates to a near-constant. There the
                // criterion is flat in κ (the kernel no longer resolves curvature)
                // and reduces to the monotone log-det Occam term, so keep-best
                // adopts the low-Occam collapsed null regardless of the true κ sign
                // — the bit-identical κ̂ → +chart-bound rail for both ±κ datasets
                // (the headline #1464 sign-blindness). The κ-sign basin is instead
                // seeded from the sign-correct fixed-κ profiled-REML scan
                // (`select_constant_curvature_kappa_sign_seed`, applied to
                // `log_kappa0` above), which routes through the same κ-opt-OFF
                // profiled fit the `curvature_inference_forspec` CI oracle trusts,
                // so the joint solve starts inside the correct sign basin with a
                // non-degenerate (κ-resolving) kernel rather than at the collapsed
                // corner. The widened ρ ceiling is retained: legitimate
                // over-smoothing is still reachable via the gradient solve and the
                // sweep grid, just not pre-pinned to the collapse point.
            }
            sc
        })
        .with_rho_bound(rho_ceiling)
        .with_heuristic_lambdas(seed_heuristic);
    if let Some((n_obs, p_cols)) = profiled_objective_size {
        // Calibrate to the n-scaled profiled criterion (see the param doc):
        // n-aware objective scale → sane absolute gradient floor + correct ARC
        // reduction-ratio reference, plus a warm ARC regularization / operator
        // trust radius that prevents the first-step overshoot. These are the
        // knobs the spatial exact-joint path was missing relative to the
        // primary REML outer; without them the iso-κ length-scale fit stalls or
        // diverges as |f| grows with n (#1053 / #1066 / #1069).
        problem = problem
            .with_objective_scale(Some(n_obs as f64))
            .with_problem_size(n_obs, p_cols)
            .with_arc_initial_regularization(Some(0.25))
            .with_operator_initial_trust_radius(Some(4.0));
    }
    if let Some(screening_cap) = screening_cap {
        problem = problem
            .with_screening_cap(screening_cap)
            .with_screen_initial_rho(true);
    }
    problem
}

/// True iff a κ-phase (`n-block exact-joint spatial`) optimizer failure is a
/// NUMERICAL pathology of the length-scale search that the fixed-κ fallback can
/// recover from (gam#787/#860), rather than a structural failure that must
/// propagate.
///
/// By the time the κ optimizer runs, the structural identifiability audits have
/// already passed upstream, so an all-seeds-rejected / rho-dimension-mismatch
/// here means a κ-driven design-rebuild penalty-topology flip starved the
/// startup validation — recoverable by fitting at the bootstrap κ. Any other
/// optimizer error (a genuine solver contract violation) still propagates.
fn kappa_phase_failure_is_fixed_kappa_recoverable(message: &str) -> bool {
    message.contains("no candidate seeds passed outer startup validation")
        || message.contains("joint hyper rho dimension mismatch")
        || message.contains("objective returned a non-finite cost")
}

pub fn optimize_spatial_length_scale_exact_joint<FitOut, FitFn, ExactFn, ExactEfsFn, SeedFn>(
    data: ArrayView2<'_, f64>,
    block_specs: &[TermCollectionSpec],
    block_term_indices: &[Vec<usize>],
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    joint_setup: &ExactJointHyperSetup,
    seed_risk_profile: gam_problem::SeedRiskProfile,
    analytic_joint_gradient_available: bool,
    analytic_joint_hessian_available: bool,
    disable_fixed_point: bool,
    screening_cap: Option<Arc<AtomicUsize>>,
    outer_derivative_policy: gam_model_api::families::custom_family::OuterDerivativePolicy,
    mut fit_fn: FitFn,
    mut exact_fn: ExactFn,
    mut exact_efs_fn: ExactEfsFn,
    mut seed_inner_beta_fn: SeedFn,
) -> Result<SpatialLengthScaleOptimizationResult<FitOut>, String>
where
    FitOut: Clone,
    FitFn: FnMut(
        &Array1<f64>,
        &[TermCollectionSpec],
        &[TermCollectionDesign],
    ) -> Result<FitOut, String>,
    ExactFn: FnMut(
        &Array1<f64>,
        &[TermCollectionSpec],
        &[TermCollectionDesign],
        gam_solve::estimate::reml::reml_outer_engine::EvalMode,
        &gam_problem::outer_subsample::RowSet,
    ) -> Result<
        (
            f64,
            Array1<f64>,
            gam_problem::HessianResult,
        ),
        String,
    >,
    ExactEfsFn: FnMut(
        &Array1<f64>,
        &[TermCollectionSpec],
        &[TermCollectionDesign],
    ) -> Result<gam_problem::EfsEval, String>,
    SeedFn:
        FnMut(&Array1<f64>) -> Result<gam_solve::rho_optimizer::SeedOutcome, EstimationError>,
{
    let n_blocks = block_specs.len();
    if block_term_indices.len() != n_blocks {
        return Err(SmoothError::dimension_mismatch(format!(
            "block_specs ({}) and block_term_indices ({}) length mismatch",
            n_blocks,
            block_term_indices.len()
        ))
        .into());
    }

    let log_kappa_dim = joint_setup.log_kappa_dim();

    log::warn!(
        "[OUTER-FD-AUDIT/spatial-exact-joint] driver entry: aux_dim={} log_kappa_dim={} kappa_enabled={} rho_dim={} theta0_len={}",
        joint_setup.auxiliary_dim(),
        log_kappa_dim,
        kappa_options.enabled,
        joint_setup.rho_dim(),
        joint_setup.theta0().len()
    );

    // -----------------------------------------------------------------------
    // Fast path: kappa disabled or no spatial terms — build designs once.
    // -----------------------------------------------------------------------
    if joint_setup.auxiliary_dim() == 0 && (!kappa_options.enabled || log_kappa_dim == 0) {
        log::warn!(
            "[OUTER-FD-AUDIT/spatial-exact-joint] taking FAST path (no outer theta optimization in this driver)"
        );
        let (designs, resolved_specs) = build_term_collection_designs_and_freeze_joint(
            data, block_specs,
        )
        .map_err(|e| {
            format!("failed to build and freeze joint block designs during exact joint kappa optimization: {e}")
        })?;
        let theta0 = joint_setup.theta0();

        // Build temporary owned slices for the closure call.
        let spec_refs: Vec<TermCollectionSpec> = resolved_specs.clone();
        let design_refs: Vec<TermCollectionDesign> = designs.clone();
        let fit = fit_fn(&theta0, &spec_refs, &design_refs)?;
        return Ok(SpatialLengthScaleOptimizationResult {
            resolved_specs,
            designs,
            fit,
            timing: None,
        });
    }

    // -----------------------------------------------------------------------
    // Full optimization path.
    // -----------------------------------------------------------------------
    let theta0 = joint_setup.theta0();
    let lower = joint_setup.lower();
    let upper = joint_setup.upper();
    if theta0.len() < log_kappa_dim || lower.len() != theta0.len() || upper.len() != theta0.len() {
        return Err(SmoothError::dimension_mismatch(format!(
            "invalid exact joint theta setup: theta0={}, lower={}, upper={}, required_log_kappa_dim={}",
            theta0.len(),
            lower.len(),
            upper.len(),
            log_kappa_dim
        ))
        .into());
    }
    let rho_dim = joint_setup.rho_dim();
    let all_dims = joint_setup.log_kappa_dims_per_term();

    // Build bootstrap designs and frozen specs for each block.
    let (boot_designs, best_specs) = build_term_collection_designs_and_freeze_joint(
        data,
        block_specs,
    )
    .map_err(|e| {
        format!(
            "failed to build and freeze joint block designs during exact joint kappa bootstrap: {e}"
        )
    })?;
    // Capability vs realized policy: the family may *advertise* an exact
    // analytic outer Hessian, but at this realized (n, psi_dim, rho_dim,
    // p_total) the predicted per-eval cost can still exceed the universal
    // outer-Hessian work budget. In that regime we route the outer optimizer
    // through gradient-only BFGS / L-BFGS, which is **convergent** to the
    // exact MLE — it just takes more line-search iterations. This is **not**
    // a feature drop: quasi-Newton picks up curvature from successive
    // analytic gradients, and the per-eval cost saving (`O(p)` instead of
    // `O(p²)`) more than pays for the iteration overhead at large scale.
    let policy_hessian_form = outer_derivative_policy.declared_hessian_form();
    let analytic_outer_hessian_available = analytic_joint_hessian_available
        && matches!(
            policy_hessian_form,
            gam_problem::DeclaredHessianForm::Either
                | gam_problem::DeclaredHessianForm::Dense
                | gam_problem::DeclaredHessianForm::Operator { .. }
        );
    let prefer_gradient_only = !analytic_outer_hessian_available;

    let theta_dim = theta0.len();
    let psi_dim = theta_dim - rho_dim;

    // Build the cache with one realizer per block.
    let cache_blocks: Vec<(TermCollectionSpec, TermCollectionDesign, Vec<usize>)> = best_specs
        .iter()
        .zip(boot_designs.iter())
        .zip(block_term_indices.iter())
        .map(|((spec, design), terms)| (spec.clone(), design.clone(), terms.clone()))
        .collect();

    struct NBlockExactJointState<'d> {
        cache: ExactJointDesignCache<'d>,
    }

    let mut state = NBlockExactJointState {
        cache: ExactJointDesignCache::new(data, cache_blocks, rho_dim, all_dims.clone())?,
    };

    // ── P7: staged-κ schedule ────────────────────────────────────────────
    //
    // The κ MLE for a stationary spatial process is asymptotically
    // *invariant* in `n` once `n` is past the Monte-Carlo resolution of
    // the cell-moment kernel. At large scale (`n ≥ STAGED_KAPPA_*`) the
    // Monte-Carlo error of a `K = 5_000`-row pilot is ≪ the κ posterior
    // width, so estimating θ on a stratified `K`-row pilot returns
    // statistically the *same* estimate as the full-data fit at a
    // fraction of the wall-clock cost. We then do one Gauss-Newton-style
    // polish at `K_polish` to absorb residual Monte-Carlo error before
    // the final coefficient fit at the polished θ on the full data.
    //
    // This is **not a heuristic shortcut**. It is the textbook
    // pilot-then-refine schedule for stationary-process likelihoods,
    // chosen here because the per-eval cost of the κ gradient grows
    // linearly in `n` and the pilot subsample reduces that cost by a
    // factor of `n / K`. The final coefficient fit at θ̂_polished on the
    // full data preserves estimation accuracy for β.
    //
    // At `n < STAGED_KAPPA_TRIGGER_N` the schedule collapses to one
    // full-data stage — identical to the pre-P7 behaviour.
    // Note: the n≥30_000 pilot trigger lives in
    // `outer_derivative_policy.should_use_staged_kappa(n_total)`; this fn
    // only carries the constants it consumes directly.
    const KAPPA_PILOT_K: usize = 5_000;
    const KAPPA_POLISH_K: usize = 25_000;
    const KAPPA_POLISH_TRIGGER_N: usize = 100_000;

    let n_total = data.nrows();
    let use_staged_kappa = outer_derivative_policy.should_use_staged_kappa(n_total);
    if use_staged_kappa {
        log::info!(
            "[KAPPA-STAGED] auto-engaging pilot+polish schedule: n={} pilot_k={} polish_k={}",
            n_total,
            KAPPA_PILOT_K,
            KAPPA_POLISH_K,
        );
    }

    // Build the initial row mask for the κ optimization.
    //
    // * `use_staged_kappa = false`: full data (`RowSet::All`). The
    //   schedule collapses to the historical single-stage path.
    // * `use_staged_kappa = true`: deterministic uniform pilot of size
    //   `min(KAPPA_PILOT_K, n_total)`, wrapped as a `RowSet::Subsample`
    //   with per-row HT weight `n_total / k_pilot`. The uniform pick is
    //   a valid unbiased estimator on its own; the stratified
    //   per-decile picker
    //   (`marginal_slope_shared::auto_outer_score_subsample`) requires
    //   the response vector `z`, which only the family evaluator can
    //   produce. **Agent C replaces this with the stratified pick once
    //   `exact_fn` exposes the per-row score.**
    //
    // Sampling RNG is seeded from `n_total` so the pilot is
    // deterministic across reruns at fixed `n`.
    fn build_uniform_pilot_subsample(
        n_total: usize,
        k_target: usize,
        seed: u64,
    ) -> gam_problem::outer_subsample::OuterScoreSubsample {
        use gam_problem::outer_subsample::OuterScoreSubsample;
        let k = k_target.min(n_total);
        if k == 0 || n_total == 0 {
            return OuterScoreSubsample::from_uniform_inclusion_mask(Vec::new(), n_total, seed);
        }
        // Reservoir-free deterministic pick: linear congruential walk
        // over a shuffled index set; for the pilot, a fast Floyd-style
        // sample is sufficient.
        let mut mask: Vec<usize> = Vec::with_capacity(k);
        // Splitmix64-driven Floyd's sampler.
        let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
        let splitmix = |s: &mut u64| -> u64 { gam_linalg::utils::splitmix64(s) };
        let mut taken = std::collections::HashSet::with_capacity(k);
        for j in (n_total - k)..n_total {
            let r = (splitmix(&mut state) % (j as u64 + 1)) as usize;
            if !taken.insert(r) {
                taken.insert(j);
                mask.push(j);
            } else {
                mask.push(r);
            }
        }
        mask.sort_unstable();
        mask.dedup();
        OuterScoreSubsample::from_uniform_inclusion_mask(mask, n_total, seed)
    }

    let current_row_set: std::cell::RefCell<gam_problem::outer_subsample::RowSet> = if use_staged_kappa {
        let pilot = build_uniform_pilot_subsample(n_total, KAPPA_PILOT_K, n_total as u64);
        std::cell::RefCell::new(gam_problem::outer_subsample::RowSet::Subsample {
            rows: std::sync::Arc::clone(&pilot.rows),
            n_full: n_total,
        })
    } else {
        std::cell::RefCell::new(gam_problem::outer_subsample::RowSet::All)
    };

    let exact_fn_cell = std::cell::RefCell::new(&mut exact_fn);
    let exact_efs_fn_cell = std::cell::RefCell::new(&mut exact_efs_fn);

    // ── κ-optimization scaling instrumentation ──
    //
    // Per-phase wall-clock counters for the three kinds of evaluator
    // invocation the κ outer drives: cost-only line-search probes,
    // value-and-gradient(/Hessian) evaluations at accepted iterates, and
    // EFS fixed-point evaluations. Each invocation emits one
    // `[KAPPA-PHASE]` log line with a per-call elapsed time, plus the
    // running call counter and a summary `theta_norm` /
    // `log_kappa_norm` so the bench runner can attribute cost to
    // particular trajectory regions. A single `[KAPPA-PHASE-SUMMARY]`
    // line is emitted on optimization exit. Grepping these is the
    // production-fit κ-scaling probe (task #32) — measurement happens
    // in real large-scale fits rather than a synthetic harness, so the
    // scaling law reflects the actual workload.
    use std::cell::Cell;
    let kphase_cost_calls: Cell<usize> = Cell::new(0);
    let kphase_cost_total_s: Cell<f64> = Cell::new(0.0);
    let kphase_eval_calls: Cell<usize> = Cell::new(0);
    let kphase_eval_total_s: Cell<f64> = Cell::new(0.0);
    let kphase_efs_calls: Cell<usize> = Cell::new(0);
    let kphase_efs_total_s: Cell<f64> = Cell::new(0.0);
    let kphase_optim_start = std::time::Instant::now();
    let kphase_log_kappa_dim = log_kappa_dim;
    let kphase_log_norms = |theta: &Array1<f64>| -> (f64, f64) {
        let theta_norm = theta.iter().map(|v| v * v).sum::<f64>().sqrt();
        let log_kappa_norm = if kphase_log_kappa_dim > 0 && theta.len() >= kphase_log_kappa_dim {
            let start = theta.len() - kphase_log_kappa_dim;
            theta.iter().skip(start).map(|v| v * v).sum::<f64>().sqrt()
        } else {
            0.0
        };
        (theta_norm, log_kappa_norm)
    };

    use gam_solve::rho_optimizer::OuterEvalOrder;
    use gam_problem::{DeclaredHessianForm, Derivative, OuterEval};

    // Joint design width across blocks → the `p` reported to the outer solver's
    // operator-vs-dense Hessian crossover. `n_total` is the load-bearing
    // profiled-objective scale (see `exact_joint_multistart_outer_problem`).
    let joint_p_cols: usize = boot_designs
        .iter()
        .map(|d| d.design.ncols())
        .sum::<usize>()
        .max(1);

    let problem = exact_joint_multistart_outer_problem(
        &theta0,
        &lower,
        &upper,
        rho_dim,
        psi_dim,
        theta_dim,
        if analytic_joint_gradient_available {
            Derivative::Analytic
        } else {
            Derivative::Unavailable
        },
        if analytic_outer_hessian_available {
            DeclaredHessianForm::Either
        } else {
            DeclaredHessianForm::Unavailable
        },
        prefer_gradient_only,
        disable_fixed_point,
        seed_risk_profile,
        kappa_options.rel_tol.max(1e-6),
        kappa_options.max_outer_iter.max(1),
        // Rho-axis cap: log-λ natural step ≈ 5.
        Some(5.0),
        // Psi-axis cap: kappa scale needs ~ln 2 per iter.
        Some(kappa_options.log_step.clamp(0.25, 1.0)),
        screening_cap.clone(),
        // n-scaled profiled-criterion calibration for every family (#1053 /
        // #1066 / #1069 iso-κ non-convergence cure).
        Some((n_total, joint_p_cols)),
        // #1464: widen the over-smoothing ρ ceiling + seed a high-λ probe when
        // any block carries a constant-curvature term.
        block_specs
            .iter()
            .any(|s| !constant_curvature_term_indices(s).is_empty()),
    );

    // Helper: collect specs and designs from cache into owned Vecs for closure calls.
    fn collect_specs(cache: &ExactJointDesignCache<'_>) -> Vec<TermCollectionSpec> {
        cache.specs().into_iter().cloned().collect()
    }
    fn collect_designs(cache: &ExactJointDesignCache<'_>) -> Vec<TermCollectionDesign> {
        cache.designs().into_iter().cloned().collect()
    }

    let result = {
        let eval_outer = |ctx: &mut &mut NBlockExactJointState<'_>,
                          theta: &Array1<f64>,
                          order: OuterEvalOrder|
         -> Result<OuterEval, EstimationError> {
            if let Some((cost, grad, hess)) = ctx.cache.memoized_eval(theta) {
                let cached_satisfies_order = match order {
                    OuterEvalOrder::Value => true,
                    OuterEvalOrder::ValueAndGradient => true,
                    OuterEvalOrder::ValueGradientHessian => hess.is_analytic(),
                };
                if cached_satisfies_order {
                    if !cost.is_finite() {
                        return Ok(OuterEval::infeasible(theta.len()));
                    }
                    // Symmetric with the non-finite-cost guard above: a non-finite
                    // gradient marks this θ as infeasible just as a non-finite cost
                    // does (e.g. degenerate tied / zero-gap survival times drive the
                    // analytic exact-joint gradient channel to NaN/Inf). Return the
                    // bounded infeasible sentinel so the outer optimizer rejects the
                    // step and shrinks its trust region — instead of hard-failing the
                    // entire REML fit and handing the driver an unbroken stream of
                    // objective failures whose recovery path deepens once per outer
                    // step until the worker stack overflows (the survival
                    // location-scale path is the one that routes through this analytic
                    // gradient, which is why it crashed where the cost-only paths only
                    // stall).
                    if grad.iter().any(|v| !v.is_finite()) {
                        return Ok(OuterEval::infeasible(theta.len()));
                    }
                    return Ok(OuterEval {
                        cost,
                        gradient: grad,
                        hessian: hess,
                        inner_beta_hint: None,
                    });
                }
            }
            if let Err(err) = ctx.cache.ensure_theta(theta) {
                log::warn!(
                    "[OUTER] n-block exact-joint spatial: ensure_theta failed during gradient evaluation: {err}"
                );
                return Ok(OuterEval::infeasible(theta.len()));
            }
            let design_revision = Some(ctx.cache.design_revision());
            let specs = collect_specs(&ctx.cache);
            let designs = collect_designs(&ctx.cache);
            // Clamp the requested order against the realized outer
            // derivative policy. The capability-aware
            // `analytic_outer_hessian_available` already encodes the
            // policy gate; re-checking through `order_for_evaluation`
            // here keeps the per-eval branch in lockstep with the
            // top-of-function declaration so the optimizer and the
            // evaluator never disagree on what was requested.
            let clamped = outer_derivative_policy.order_for_evaluation(order);
            let need_hessian = matches!(clamped, OuterEvalOrder::ValueGradientHessian)
                && analytic_outer_hessian_available;
            let eval_mode = if need_hessian {
                gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueGradientHessian
            } else {
                gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueAndGradient
            };
            let t0 = std::time::Instant::now();
            let result = {
                let row_set_borrow = current_row_set.borrow();
                (*exact_fn_cell.borrow_mut())(theta, &specs, &designs, eval_mode, &row_set_borrow)
            };
            let elapsed_s = t0.elapsed().as_secs_f64();
            kphase_eval_calls.set(kphase_eval_calls.get() + 1);
            kphase_eval_total_s.set(kphase_eval_total_s.get() + elapsed_s);
            let (theta_norm, log_kappa_norm) = kphase_log_norms(theta);
            log::info!(
                "[KAPPA-PHASE] phase=eval_outer call={} order={:?} design_revision={:?} theta_norm={:.4e} log_kappa_norm={:.4e} elapsed_s={:.4}",
                kphase_eval_calls.get(),
                order,
                design_revision,
                theta_norm,
                log_kappa_norm,
                elapsed_s,
            );
            match result {
                Ok((cost, grad, hess)) => {
                    ctx.cache.store_eval((cost, grad.clone(), hess.clone()));
                    if !cost.is_finite() {
                        return Ok(OuterEval::infeasible(theta.len()));
                    }
                    // Symmetric with the non-finite-cost guard above: a non-finite
                    // gradient marks this θ as infeasible just as a non-finite cost
                    // does (e.g. degenerate tied / zero-gap survival times drive the
                    // analytic exact-joint gradient channel to NaN/Inf). Return the
                    // bounded infeasible sentinel so the outer optimizer rejects the
                    // step and shrinks its trust region — instead of hard-failing the
                    // entire REML fit and handing the driver an unbroken stream of
                    // objective failures whose recovery path deepens once per outer
                    // step until the worker stack overflows (the survival
                    // location-scale path is the one that routes through this analytic
                    // gradient, which is why it crashed where the cost-only paths only
                    // stall).
                    if grad.iter().any(|v| !v.is_finite()) {
                        return Ok(OuterEval::infeasible(theta.len()));
                    }
                    Ok(OuterEval {
                        cost,
                        gradient: grad,
                        hessian: hess,
                        inner_beta_hint: None,
                    })
                }
                Err(err) => {
                    log::warn!(
                        "[OUTER] n-block exact-joint spatial: exact evaluation failed: {err}"
                    );
                    Ok(OuterEval::infeasible(theta.len()))
                }
            }
        };

        let obj = problem.build_objective_with_eval_order(
            &mut state,
            |ctx: &mut &mut NBlockExactJointState<'_>, theta: &Array1<f64>| {
                if let Some(cost) = ctx.cache.memoized_cost(theta) {
                    return Ok(cost);
                }
                if let Err(err) = ctx.cache.ensure_theta(theta) {
                    log::warn!(
                        "[OUTER] n-block exact-joint spatial: ensure_theta failed during cost evaluation: {err}"
                    );
                    return Ok(f64::INFINITY);
                }
                let design_revision = Some(ctx.cache.design_revision());
                let specs = collect_specs(&ctx.cache);
                let designs = collect_designs(&ctx.cache);
                // Cost-only line-search probe: pass `ValueOnly` so the closure
                // skips gradient and Hessian assembly. This is the principled
                // fix for the N-block joint optimization V+G-per-probe waste —
                // gradient construction (≈ 6.5·10⁹ FLOPs per CTN step at
                // n=320 000, n_grid=293, p_resp=32, p_cov=23) is now paid only
                // when the outer evaluator actually requests it.
                let t0 = std::time::Instant::now();
                let result = {
                    let row_set_borrow = current_row_set.borrow();
                    (*exact_fn_cell.borrow_mut())(
                        theta,
                        &specs,
                        &designs,
                        gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueOnly,
                        &row_set_borrow,
                    )
                };
                let elapsed_s = t0.elapsed().as_secs_f64();
                kphase_cost_calls.set(kphase_cost_calls.get() + 1);
                kphase_cost_total_s.set(kphase_cost_total_s.get() + elapsed_s);
                let (theta_norm, log_kappa_norm) = kphase_log_norms(theta);
                log::info!(
                    "[KAPPA-PHASE] phase=cost call={} design_revision={:?} theta_norm={:.4e} log_kappa_norm={:.4e} elapsed_s={:.4}",
                    kphase_cost_calls.get(),
                    design_revision,
                    theta_norm,
                    log_kappa_norm,
                    elapsed_s,
                );
                match result {
                    Ok((cost, _grad, _hess)) => {
                        // Don't `store_eval`: that path is only valid when the
                        // closure produced a real gradient. The next outer-eval
                        // call will recompute (V, ∇V) at this θ if needed; the
                        // memoized_cost path covers the common case where the
                        // line search returns to an accepted iterate.
                        ctx.cache.store_cost_only(theta, cost);
                        Ok(cost)
                    }
                    Err(err) => {
                        log::warn!(
                            "[OUTER] n-block exact-joint spatial: exact cost evaluation failed: {err}"
                        );
                        Ok(f64::INFINITY)
                    }
                }
            },
            |ctx: &mut &mut NBlockExactJointState<'_>, theta: &Array1<f64>| {
                eval_outer(
                    ctx,
                    theta,
                    if analytic_outer_hessian_available {
                        OuterEvalOrder::ValueGradientHessian
                    } else {
                        OuterEvalOrder::ValueAndGradient
                    },
                )
            },
            |ctx: &mut &mut NBlockExactJointState<'_>,
             theta: &Array1<f64>,
             order: OuterEvalOrder| { eval_outer(ctx, theta, order) },
            None::<fn(&mut &mut NBlockExactJointState<'_>)>,
            Some(
                |ctx: &mut &mut NBlockExactJointState<'_>, theta: &Array1<f64>| {
                    ctx.cache
                        .ensure_theta(theta)
                        .map_err(EstimationError::InvalidInput)?;
                    let design_revision = Some(ctx.cache.design_revision());
                    let specs = collect_specs(&ctx.cache);
                    let designs = collect_designs(&ctx.cache);
                    let t0 = std::time::Instant::now();
                    let eval_result = (*exact_efs_fn_cell.borrow_mut())(
                        theta,
                        &specs,
                        &designs,
                    );
                    let elapsed_s = t0.elapsed().as_secs_f64();
                    kphase_efs_calls.set(kphase_efs_calls.get() + 1);
                    kphase_efs_total_s.set(kphase_efs_total_s.get() + elapsed_s);
                    let (theta_norm, log_kappa_norm) = kphase_log_norms(theta);
                    log::info!(
                        "[KAPPA-PHASE] phase=efs call={} design_revision={:?} theta_norm={:.4e} log_kappa_norm={:.4e} elapsed_s={:.4}",
                        kphase_efs_calls.get(),
                        design_revision,
                        theta_norm,
                        log_kappa_norm,
                        elapsed_s,
                    );
                    let eval = eval_result.map_err(EstimationError::RemlOptimizationFailed)?;
                    Ok(eval)
                },
            ),
        );
        let mut obj = obj.with_seed_inner_state(
            move |_ctx: &mut &mut NBlockExactJointState<'_>, beta: &Array1<f64>| {
                (seed_inner_beta_fn)(beta)
            },
        );

        match problem.run(&mut obj, "n-block exact-joint spatial") {
            Ok(result) => result,
            Err(e) => {
                let message = e.to_string();
                // Kappa-phase graceful degradation (gam#787/#860). The
                // length-scale (κ) optimizer rebuilds the spatial design at each
                // trial κ; a κ-driven matern penalty-topology flip (the
                // FrozenTransform spectral-tolerance crossing in
                // `build_nullspace_shrinkage_penalty`) can make the rebuilt
                // design's learned-penalty count disagree with the frozen
                // joint-setup ρ dimension, so EVERY κ seed fails startup
                // validation ("joint hyper rho dimension mismatch" → all seeds
                // rejected → "no candidate seeds passed outer startup
                // validation"). That is a NUMERICAL pathology of the κ search on
                // a structurally-well-posed design (the structural audits already
                // passed upstream) — NOT a reason to fail the whole fit. Fall
                // back to a FIXED κ (the bootstrap length-scale, skipping κ
                // optimization): build + freeze the joint designs at the initial
                // κ and fit there. We lose κ tuning but return a REAL, valid
                // model — graceful degradation, exactly mirroring the
                // `kappa_options.enabled == false` fixed-κ path above. Only the
                // startup-validation / mismatch class is caught; any other κ
                // optimizer error still propagates.
                if kappa_phase_failure_is_fixed_kappa_recoverable(&message) {
                    drop(obj);
                    log::warn!(
                        "[KAPPA-PHASE] length-scale optimization could not validate any seed \
                         ({message}); falling back to a FIXED bootstrap κ (skipping κ \
                         optimization) and fitting there — a real model at the initial \
                         length-scale rather than raising (gam#787/#860)."
                    );
                    let (designs, resolved_specs) =
                        build_term_collection_designs_and_freeze_joint(data, block_specs).map_err(
                            |build_err| {
                                format!(
                                    "fixed-κ fallback failed to build and freeze joint block \
                                     designs after κ optimization could not validate a seed \
                                     ({message}): {build_err}"
                                )
                            },
                        )?;
                    let fixed_theta0 = joint_setup.theta0();
                    let spec_refs: Vec<TermCollectionSpec> = resolved_specs.clone();
                    let design_refs: Vec<TermCollectionDesign> = designs.clone();
                    let fit = fit_fn(&fixed_theta0, &spec_refs, &design_refs)?;
                    return Ok(SpatialLengthScaleOptimizationResult {
                        resolved_specs,
                        designs,
                        fit,
                        timing: None,
                    });
                }
                return Err(message);
            }
        }
    }; // obj dropped here, releasing mutable borrow on state

    // ── κ-optimization scaling summary ──
    //
    // Single line summarizing all per-call wall-clock counters
    // accumulated above. The bench runner / scaling-law analyzer
    // can pivot on this directly without parsing the per-call
    // [KAPPA-PHASE] markers (which remain available for
    // attribution).
    let kphase_total_s = kphase_optim_start.elapsed().as_secs_f64();
    log::info!(
        "[KAPPA-PHASE-SUMMARY] log_kappa_dim={} n_cost={} cost_total_s={:.4} n_eval={} eval_total_s={:.4} n_efs={} efs_total_s={:.4} optim_total_s={:.4}",
        kphase_log_kappa_dim,
        kphase_cost_calls.get(),
        kphase_cost_total_s.get(),
        kphase_eval_calls.get(),
        kphase_eval_total_s.get(),
        kphase_efs_calls.get(),
        kphase_efs_total_s.get(),
        kphase_total_s,
    );
    let timing = SpatialLengthScaleOptimizationTiming {
        log_kappa_dim: kphase_log_kappa_dim,
        cost_calls: kphase_cost_calls.get(),
        cost_total_s: kphase_cost_total_s.get(),
        eval_calls: kphase_eval_calls.get(),
        eval_total_s: kphase_eval_total_s.get(),
        efs_calls: kphase_efs_calls.get(),
        efs_total_s: kphase_efs_total_s.get(),
        slow_path_resets: 0,
        design_revision_delta: 0,
        nfree_skip_row_touches: 0,
        nfree_miss_shape: 0,
        nfree_miss_value: 0,
        nfree_miss_gradient: 0,
        nfree_miss_penalty: 0,
        nfree_miss_revision: 0,
        nfree_miss_second_order: 0,
        nfree_miss_other: 0,
        optim_total_s: kphase_total_s,
    };

    let theta_star = result.rho;

    // ── P7 stage rotation ────────────────────────────────────────────────
    //
    // The optimization above ran against `current_row_set` — the pilot
    // subsample under `use_staged_kappa`, otherwise the full data. We
    // now:
    //
    // 1. If `n_total ≥ KAPPA_POLISH_TRIGGER_N`, rotate to a larger
    //    polish subsample and request a single value+gradient evaluation
    //    at `theta_star` so the family caches its polished score. This
    //    is the Gauss-Newton-style polish in the schedule — one step
    //    rather than a full re-run because the pilot has already
    //    consumed most of the curvature information.
    //
    // 2. Always rotate back to `RowSet::All` before the final
    //    coefficient fit `fit_fn(theta_star)`. The final β estimate at
    //    θ̂ uses the full data so no estimation accuracy is lost.
    if use_staged_kappa && n_total >= KAPPA_POLISH_TRIGGER_N {
        let polish = build_uniform_pilot_subsample(
            n_total,
            KAPPA_POLISH_K,
            (n_total as u64).wrapping_add(0xA5A5A5A5),
        );
        *current_row_set.borrow_mut() = gam_problem::outer_subsample::RowSet::Subsample {
            rows: std::sync::Arc::clone(&polish.rows),
            n_full: n_total,
        };
        log::info!(
            "[KAPPA-STAGED] rotating to polish subsample: k={} at theta_star",
            polish.rows.len(),
        );
        // One V+G evaluation at theta_star on the polish subsample. The
        // returned objective pieces must be usable; the family-side cache
        // update inside `exact_fn` is consumed by the final fit.
        state.cache.ensure_theta(&theta_star)?;
        let (polish_cost, polish_grad, _) = {
            let specs = collect_specs(&state.cache);
            let designs = collect_designs(&state.cache);
            let row_set_borrow = current_row_set.borrow();
            exact_fn(
                &theta_star,
                &specs,
                &designs,
                gam_solve::estimate::reml::reml_outer_engine::EvalMode::ValueAndGradient,
                &row_set_borrow,
            )?
        };
        if !polish_cost.is_finite() || polish_grad.iter().any(|value| !value.is_finite()) {
            return Err(
                "polish subsample exact-joint evaluation produced non-finite objective pieces"
                    .to_string(),
            );
        }
    }
    *current_row_set.borrow_mut() = gam_problem::outer_subsample::RowSet::All;
    if use_staged_kappa {
        log::info!(
            "[KAPPA-STAGED] rotating to full data for final coefficient fit (n={})",
            n_total,
        );
    }

    state.cache.ensure_theta(&theta_star)?;

    let resolved_specs: Vec<TermCollectionSpec> = collect_specs(&state.cache);
    let designs: Vec<TermCollectionDesign> = collect_designs(&state.cache);

    let fit = fit_fn(&theta_star, &resolved_specs, &designs)?;

    for spec in &resolved_specs {
        log_spatial_aniso_scales(spec);
    }

    Ok(SpatialLengthScaleOptimizationResult {
        resolved_specs,
        designs,
        fit,
        timing: Some(timing),
    })
}

fn try_exact_joint_latent_coord_optimization(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    best: &FittedTermCollection,
    family: LikelihoodSpec,
    options: &FitOptions,
    latent: &StandardLatentCoordConfig,
) -> Result<FittedTermCollectionWithSpec, EstimationError> {
    use gam_solve::rho_optimizer::OuterEvalOrder;
    use gam_problem::{DeclaredHessianForm, Derivative, OuterEval};

    let rho_dim = best.fit.lambdas.len();
    let latent_flat_dim = latent.values.len();
    if latent_flat_dim == 0 {
        crate::bail_invalid_estim!(
            "latent-coordinate optimization requires a non-empty latent block"
        );
    }
    let direct_hypers =
        latent_coord_initial_direct_hypers(latent.values.id_mode(), latent.values.latent_dim())?;
    let analytic_rho_count = latent
        .analytic_penalties
        .as_ref()
        .map_or(0, |registry| registry.total_rho_count());
    let latent_coord_ext_dim = latent_flat_dim + analytic_rho_count + direct_hypers.len();

    let mut theta0 = Array1::<f64>::zeros(rho_dim + latent_coord_ext_dim);
    theta0
        .slice_mut(s![..rho_dim])
        .assign(&best.fit.lambdas.mapv(f64::ln));
    theta0
        .slice_mut(s![rho_dim..rho_dim + latent_flat_dim])
        .assign(latent.values.as_flat());
    if !direct_hypers.is_empty() {
        let direct_start = rho_dim + latent_flat_dim + analytic_rho_count;
        theta0
            .slice_mut(s![direct_start..direct_start + direct_hypers.len()])
            .assign(&direct_hypers);
    }

    let mut lower = Array1::<f64>::from_elem(theta0.len(), -12.0);
    let mut upper = Array1::<f64>::from_elem(theta0.len(), 12.0);
    let latent_bound = latent
        .values
        .as_flat()
        .iter()
        .fold(1.0_f64, |acc, &v| acc.max(v.abs()))
        + 10.0;
    for axis in rho_dim..rho_dim + latent_flat_dim {
        lower[axis] = -latent_bound;
        upper[axis] = latent_bound;
    }

    struct LatentJointContext<'d> {
        rho_dim: usize,
        cache: SingleBlockLatentCoordDesignCache,
        evaluator: gam_solve::estimate::ExternalJointHyperEvaluator<'d>,
    }

    impl<'d> LatentJointContext<'d> {
        fn eval_full(
            &mut self,
            theta: &Array1<f64>,
            order: OuterEvalOrder,
        ) -> Result<
            (
                f64,
                Array1<f64>,
                gam_problem::HessianResult,
            ),
            EstimationError,
        > {
            if let Some(eval) = self.cache.memoized_eval(theta) {
                return Ok(eval);
            }
            self.cache
                .ensure_theta(theta)
                .map_err(EstimationError::InvalidInput)?;
            let hyper_dirs = self
                .cache
                .hyper_dirs()
                .map_err(EstimationError::InvalidInput)?;
            let design_revision = Some(self.cache.design_revision());
            let registry_for_key = self.cache.analytic_penalties();
            self.evaluator
                .set_analytic_penalty_registry(registry_for_key.as_deref());
            let mut eval = evaluate_joint_reml_outer_eval_at_theta(
                &mut self.evaluator,
                self.cache.design(),
                theta,
                self.rho_dim,
                hyper_dirs,
                None,
                order,
                design_revision,
            )?;
            let latent = self.cache.latent().map_err(EstimationError::InvalidInput)?;
            if let Some(registry) = registry_for_key {
                let mut registry = registry.as_ref().clone();
                registry.apply_weight_schedules(
                    gam_solve::estimate::reml::outer_eval::current_outer_iter() as usize,
                );
                add_analytic_penalty_objective_to_eval(
                    theta,
                    self.rho_dim,
                    latent.as_ref(),
                    &registry,
                    &mut eval,
                )?;
            }
            add_latent_id_objective_to_eval(
                theta,
                self.rho_dim,
                self.cache.analytic_penalty_rho_count(),
                latent.as_ref(),
                &mut eval,
            )?;
            self.cache.store_eval(eval.clone());
            Ok(eval)
        }

        fn eval_efs(
            &mut self,
            theta: &Array1<f64>,
        ) -> Result<gam_problem::EfsEval, EstimationError> {
            self.cache
                .ensure_theta(theta)
                .map_err(EstimationError::InvalidInput)?;
            let hyper_dirs = self
                .cache
                .hyper_dirs()
                .map_err(EstimationError::InvalidInput)?;
            let registry_for_key = self.cache.analytic_penalties();
            self.evaluator
                .set_analytic_penalty_registry(registry_for_key.as_deref());
            let mut efs = evaluate_joint_reml_efs_at_theta(
                &mut self.evaluator,
                self.cache.design(),
                theta,
                self.rho_dim,
                hyper_dirs,
                None,
                Some(self.cache.design_revision()),
            )?;
            if let Some(registry) = registry_for_key {
                let mut registry = registry.as_ref().clone();
                registry.apply_weight_schedules(
                    gam_solve::estimate::reml::outer_eval::current_outer_iter() as usize,
                );
                let latent = self.cache.latent().map_err(EstimationError::InvalidInput)?;
                let contribution = analytic_penalty_objective_contribution(
                    theta,
                    self.rho_dim,
                    latent.as_ref(),
                    &registry,
                )?;
                efs.cost += contribution.cost;
                if let (Some(psi_gradient), Some(psi_indices)) =
                    (efs.psi_gradient.as_mut(), efs.psi_indices.as_ref())
                {
                    if psi_gradient.len() != psi_indices.len() {
                        crate::bail_invalid_estim!(
                            "latent-coordinate analytic penalty EFS psi gradient length mismatch: gradient={}, indices={}",
                            psi_gradient.len(),
                            psi_indices.len()
                        );
                    }
                    for (local_idx, &theta_idx) in psi_indices.iter().enumerate() {
                        psi_gradient[local_idx] += contribution.gradient[theta_idx];
                    }
                }
            }
            Ok(efs)
        }

        fn eval_cost(&mut self, theta: &Array1<f64>) -> f64 {
            if let Some(cost) = self.cache.memoized_cost(theta) {
                return cost;
            }
            if self.cache.ensure_theta(theta).is_err() {
                return f64::INFINITY;
            }
            let design_revision = Some(self.cache.design_revision());
            let registry_for_key = self.cache.analytic_penalties();
            self.evaluator
                .set_analytic_penalty_registry(registry_for_key.as_deref());
            let result = {
                let design = self.cache.design();
                self.evaluator.evaluate_cost_only(
                    &design.design,
                    &design.penalties,
                    &design.nullspace_dims,
                    design.linear_constraints.clone(),
                    theta,
                    self.rho_dim,
                    None,
                    "latent-coordinate-joint cost-only",
                    design_revision,
                )
            };
            match result {
                Ok(cost) => {
                    let latent = match self.cache.latent() {
                        Ok(latent) => latent,
                        Err(_) => return f64::INFINITY,
                    };
                    let contribution = match latent_id_objective_contribution(
                        theta,
                        self.rho_dim,
                        self.cache.analytic_penalty_rho_count(),
                        latent.as_ref(),
                    ) {
                        Ok(contribution) => contribution,
                        Err(_) => return f64::INFINITY,
                    };
                    let cost = cost + contribution.cost;
                    let cost = if let Some(registry) = registry_for_key {
                        let mut registry = registry.as_ref().clone();
                        registry.apply_weight_schedules(
                            gam_solve::estimate::reml::outer_eval::current_outer_iter()
                                as usize,
                        );
                        match analytic_penalty_objective_contribution(
                            theta,
                            self.rho_dim,
                            latent.as_ref(),
                            &registry,
                        ) {
                            Ok(contribution) => cost + contribution.cost,
                            Err(_) => return f64::INFINITY,
                        }
                    } else {
                        cost
                    };
                    self.cache.store_cost(cost);
                    cost
                }
                Err(_) => f64::INFINITY,
            }
        }
    }

    let mut ctx = LatentJointContext {
        rho_dim,
        cache: SingleBlockLatentCoordDesignCache::new(
            data.to_owned(),
            resolvedspec.clone(),
            best.design.clone(),
            latent,
            rho_dim,
        )
        .map_err(EstimationError::InvalidInput)?,
        evaluator: gam_solve::estimate::ExternalJointHyperEvaluator::new(
            y,
            weights,
            &best.design.design,
            offset,
            &best.design.penalties,
            &external_opts_for_design(&family, &best.design, options),
            "latent-coordinate-joint",
        )?,
    };
    let registry_for_key = ctx.cache.analytic_penalties();
    ctx.evaluator
        .set_analytic_penalty_registry(registry_for_key.as_deref());
    ctx.evaluator
        .set_persistent_latent_values_fingerprint(latent.values.id_mode());
    if let Some(cached_t) = ctx
        .evaluator
        .load_persistent_latent_values(latent.values.n_obs(), latent.values.latent_dim())
    {
        let cached_t: Array2<f64> = cached_t;
        for (dst, src) in theta0
            .slice_mut(s![rho_dim..rho_dim + latent_flat_dim])
            .iter_mut()
            .zip(cached_t.iter())
        {
            *dst = *src;
        }
    }

    let problem = exact_joint_multistart_outer_problem(
        &theta0,
        &lower,
        &upper,
        rho_dim,
        latent_coord_ext_dim,
        theta0.len(),
        Derivative::Analytic,
        DeclaredHessianForm::Unavailable,
        false,
        false,
        seed_risk_profile_for_likelihood_family(&family),
        options.tol,
        options.max_iter.max(1),
        Some(5.0),
        Some(0.5),
        None,
        // n-scaled profiled-criterion calibration (same absolute-gradient-floor
        // correction as the spatial paths; #1053 / #1066 / #1069).
        Some((data.nrows(), best.design.design.ncols().max(1))),
        // #1464: widen the over-smoothing ρ ceiling and seed the high-ρ probe
        // only when a constant-curvature curv() term is present in this fit.
        !constant_curvature_term_indices(resolvedspec).is_empty(),
    );

    let eval_outer = |ctx: &mut &mut LatentJointContext<'_>,
                      theta: &Array1<f64>,
                      order: OuterEvalOrder|
     -> Result<OuterEval, EstimationError> {
        let (cost, gradient, hessian) = ctx.eval_full(theta, order)?;
        Ok(OuterEval {
            cost,
            gradient,
            hessian,
            inner_beta_hint: None,
        })
    };

    let result = {
        let mut obj = problem.build_objective_with_eval_order(
            &mut ctx,
            |ctx: &mut &mut LatentJointContext<'_>, theta: &Array1<f64>| Ok(ctx.eval_cost(theta)),
            |ctx: &mut &mut LatentJointContext<'_>, theta: &Array1<f64>| {
                eval_outer(ctx, theta, OuterEvalOrder::ValueAndGradient)
            },
            |ctx: &mut &mut LatentJointContext<'_>, theta: &Array1<f64>, order: OuterEvalOrder| {
                eval_outer(ctx, theta, order)
            },
            Some(|ctx: &mut &mut LatentJointContext<'_>| {
                ctx.cache.reset();
            }),
            Some(|ctx: &mut &mut LatentJointContext<'_>, theta: &Array1<f64>| ctx.eval_efs(theta)),
        );

        problem
            .run(&mut obj, "latent-coordinate joint REML")
            .map_err(|e| {
                EstimationError::InvalidInput(format!(
                    "latent-coordinate joint optimization failed after exhausting strategy fallbacks: {e}"
                ))
            })?
    };
    if !result.converged {
        crate::bail_invalid_estim!(
            "latent-coordinate joint optimization did not converge after {} iterations (final_objective={:.6e}, final_grad_norm={})",
            result.iterations,
            result.final_value,
            result.final_grad_norm_report(),
        );
    }

    let theta_star = result.rho;
    let rho_star = theta_star.slice(s![..rho_dim]).mapv(f64::exp);
    let mut final_data = data.to_owned();
    let flat_t = theta_star
        .slice(s![rho_dim..rho_dim + latent_flat_dim])
        .to_owned();
    let mut fitted_latent_values =
        Array2::<f64>::zeros((latent.values.n_obs(), latent.values.latent_dim()));
    for n in 0..latent.values.n_obs() {
        for axis in 0..latent.values.latent_dim() {
            let value = flat_t[n * latent.values.latent_dim() + axis];
            fitted_latent_values[[n, axis]] = value;
            final_data[[n, latent.feature_cols[axis]]] = value;
        }
    }
    let optimized = fit_term_collection_forspecwith_heuristic_lambdas(
        final_data.view(),
        y,
        weights,
        offset,
        resolvedspec,
        rho_star.as_slice(),
        family,
        options,
    )?;
    ctx.evaluator
        .store_persistent_latent_values(&fitted_latent_values);
    let mut fit = optimized.fit;
    fit.reml_score = result.final_value;
    fit.penalized_objective = result.final_value;
    Ok(FittedTermCollectionWithSpec {
        fit,
        design: optimized.design,
        resolvedspec: resolvedspec.clone(),
        adaptive_diagnostics: optimized.adaptive_diagnostics,
        kappa_timing: None,
    })
}

pub fn fit_term_collectionwith_latent_coord_optimization(
    data: ArrayView2<'_, f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    spec: &TermCollectionSpec,
    latent: &StandardLatentCoordConfig,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<FittedTermCollectionWithSpec, EstimationError> {
    let n = data.nrows();
    if !(y.len() == n && weights.len() == n && offset.len() == n) {
        crate::bail_invalid_estim!(
            "fit_term_collectionwith_latent_coord_optimization row mismatch: n={}, y={}, weights={}, offset={}",
            n,
            y.len(),
            weights.len(),
            offset.len()
        );
    }
    let best = fit_term_collection_forspec(
        data,
        y.view(),
        weights.view(),
        offset.view(),
        spec,
        family.clone(),
        options,
    )?;
    let resolvedspec = freeze_term_collection_from_design(spec, &best.design)?;
    try_exact_joint_latent_coord_optimization(
        data,
        y.view(),
        weights.view(),
        offset.view(),
        &resolvedspec,
        &best,
        family,
        options,
        latent,
    )
}

pub fn fit_term_collectionwith_spatial_length_scale_optimization(
    data: ArrayView2<'_, f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    spec: &TermCollectionSpec,
    family: LikelihoodSpec,
    options: &FitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<FittedTermCollectionWithSpec, EstimationError> {
    // Spatial hyperparameters change kernel geometry nonlinearly, so each
    // proposal rebuilds the spatial basis. Hybrid/isotropic terms expose a
    // scalar κ (= 1/length_scale); pure Duchon anisotropy exposes only
    // per-axis shape coordinates.
    //
    // When exact derivative information is available for the rebuilt basis and
    // penalty, kappa is promoted to a first-class outer hyperparameter beside
    // rho = log(lambda). In that mode this routine runs a joint outer solve in
    // theta = [rho, psi], where psi = log(kappa) = -log(length_scale), and the
    // optimizer is expected to consume a real joint Hessian. ARC is not meant
    // to run on a gradient-only surrogate here.
    //
    // Any eligible spatial smooth participates in this outer solve. If an
    // eligible spatial basis does not expose derivative information, that is
    // now a hard error.
    let mut resolvedspec = spec.clone();
    let spatial_terms = spatial_length_scale_term_indices(&resolvedspec);
    let n = data.nrows();
    if !(y.len() == n && weights.len() == n && offset.len() == n) {
        crate::bail_invalid_estim!(
            "fit_term_collectionwith_spatial_length_scale_optimization row mismatch: n={}, y={}, weights={}, offset={}",
            n,
            y.len(),
            weights.len(),
            offset.len()
        );
    }
    if !kappa_options.enabled || spatial_terms.is_empty() {
        let out = fit_term_collection_forspec(
            data,
            y.view(),
            weights.view(),
            offset.view(),
            &resolvedspec,
            family,
            options,
        )?;
        let resolvedspec = freeze_term_collection_from_design(&resolvedspec, &out.design)?;
        return Ok(FittedTermCollectionWithSpec {
            fit: out.fit,
            design: out.design,
            resolvedspec,
            adaptive_diagnostics: out.adaptive_diagnostics,
            kappa_timing: None,
        });
    }
    if kappa_options.max_outer_iter == 0 {
        crate::bail_invalid_estim!("spatial kappa optimization requires max_outer_iter >= 1");
    }
    if !(kappa_options.log_step.is_finite() && kappa_options.log_step > 0.0) {
        crate::bail_invalid_estim!("spatial kappa optimization requires log_step > 0");
    }
    if !(kappa_options.min_length_scale.is_finite()
        && kappa_options.max_length_scale.is_finite()
        && kappa_options.min_length_scale > 0.0
        && kappa_options.max_length_scale >= kappa_options.min_length_scale)
    {
        crate::bail_invalid_estim!(
            "spatial kappa optimization requires valid positive length_scale bounds"
        );
    }

    let pilot_threshold = kappa_options.pilot_subsample_threshold;
    if pilot_threshold > 0 && n > pilot_threshold * 2 {
        log::info!(
            "[spatial-kappa] n={n} exceeds pilot threshold {}; using pilot geometry only for deterministic anisotropy initialization",
            pilot_threshold * 2,
        );
        apply_spatial_anisotropy_pilot_initializer(
            data,
            &mut resolvedspec,
            &spatial_terms,
            pilot_threshold,
            kappa_options,
        );
    }

    // #1376: the geometry-only anisotropy seed (`initial_aniso_contrasts`, from
    // per-axis knot-coordinate spread) is blind to the response, so a signal
    // axis and a nuisance axis with equal coordinate spread both seed to ~0 and
    // the κ optimizer can stall at the symmetric point (it found a weak/flat
    // antisymmetric gradient, amplified by double-penalty nullspace shrinkage).
    // Add a bounded, response-aware per-axis nudge so the optimizer starts in
    // the correct basin. This runs whether or not the pilot initializer fired
    // (the pilot path is gated on a large-n threshold).
    apply_response_aware_anisotropy_seed(data, y.view(), &mut resolvedspec, &spatial_terms);

    // #1464: pin each constant-curvature term's κ to the κ-FAIR sign-scan value
    // BEFORE the baseline fit. The production profiled-REML criterion
    // (`fixed_kappa_profiled_reml_score`) that drives BOTH the baseline geometry
    // and the joint solve's accept-vs-baseline gate (`joint_final_value >
    // baseline_score`) is SIGN-BLIND — its data-fit term decreases monotonically
    // toward +κ for either truth sign, so a baseline left at κ = 0 always beats a
    // correctly-signed-but-negative κ candidate on raw REML, and the gate discards
    // the right answer (hyperbolic κ̂ → 0, recovered as spherical). Only the κ-fair
    // scan (`constant_curvature_kappa_fair_sign_score`, which subtracts the
    // design's generic radial-peak-fitting power) identifies the sign; since the κ
    // MAGNITUDE is unidentified (raw V_p rails to a chart bound regardless), the
    // scan's argmin is the authoritative κ̂. Pinning the baseline there makes the
    // baseline, the frozen joint candidate (see the κ-PIN in
    // `try_exact_joint_spatial_length_scale_optimization`), and the gate all agree
    // on the sign-correct κ. Byte-identical for genuinely spherical data (the scan
    // and the raw criterion both pick the +bound there) and for non-CC spatial
    // terms (never entered). A scan result of κ = 0 (genuinely flat) leaves κ as-is.
    for term_idx in constant_curvature_term_indices(&resolvedspec) {
        // gam#2152: a user-PINNED κ is a fixed geometry — never overwrite it with
        // the κ-fair sign scan's estimate. The scan (and every downstream κ
        // re-derivation) is skipped for this term; its baseline, joint window,
        // and readback all stay at exactly the user's kappa=.
        if constant_curvature_kappa_is_fixed(&resolvedspec, term_idx) {
            continue;
        }
        if let Some(kappa_seed) =
            select_constant_curvature_kappa_sign_seed(data, y.view(), &resolvedspec, term_idx)
            && kappa_seed != 0.0
            && let Some(SmoothBasisSpec::ConstantCurvature { spec: cc, .. }) =
                resolvedspec.smooth_terms.get_mut(term_idx).map(|t| &mut t.basis)
        {
            log::info!(
                "[#1464] pinned CC term {term_idx} baseline κ to κ-fair scan value {kappa_seed} \
                 (raw profiled REML is sign-blind; scan is authoritative for the sign)"
            );
            cc.kappa = kappa_seed;
        }
    }

    let baseline_options = superseded_fit_options(options);
    let mut best = fit_term_collection_forspec(
        data,
        y.view(),
        weights.view(),
        offset.view(),
        &resolvedspec,
        family.clone(),
        &baseline_options,
    )?;
    resolvedspec = freeze_term_collection_from_design(&resolvedspec, &best.design)?;
    // The freeze step can rewrite a term's basis variant — most notably when
    // `build_thin_plate_basis_with_workspace` auto-promotes an infeasible
    // canonical-TPS request to a pure Duchon spline (length_scale = None,
    // no anisotropy). The pre-fit eligibility list was computed against the
    // ThinPlate spec, which has length_scale set, so it included that term.
    // After the rewrite the same term is a *pure* Duchon basis with no free
    // length-scale parameter to optimize, and the downstream kappa solver
    // (which assumes hybrid Duchon for log-κ derivatives) errors out. Refresh
    // the index list so it reflects the post-freeze spec.
    let mut spatial_terms = spatial_length_scale_term_indices(&resolvedspec);
    // Sync knot-cloud-derived aniso contrasts from the basis metadata back
    // into the spec so the optimizer starts from the geometry-informed η values
    // rather than the zero sentinel from --scale-dimensions.
    sync_aniso_contrasts_from_metadata(&mut resolvedspec, &best.design.smooth);
    // #1074: kernel-range multi-start. The single midpoint seed can strand the
    // joint [ρ, ψ] solver in a long-range local optimum for the roughest kernels
    // (Matérn ν=3/2); a coarse log-κ grid restart re-seeds the spec's length
    // scale in the globally best-scoring basin before the joint solve refines it.
    // Strict-improvement-only, so a fit the midpoint already solved well is left
    // byte-identical. Isotropic/non-CC only (gated inside the helper).
    let mut prescan_improved = false;
    if !spatial_terms.is_empty() {
        let baseline_score = fit_score(&best.fit);
        let range_overrides = prescan_isotropic_spatial_range_seed(
            data,
            y.view(),
            weights.view(),
            offset.view(),
            &resolvedspec,
            baseline_score,
            &family,
            &baseline_options,
            kappa_options,
            &spatial_terms,
        )?;
        if !range_overrides.is_empty() {
            prescan_improved = true;
            for (term_idx, length_scale) in range_overrides {
                set_spatial_length_scale(&mut resolvedspec, term_idx, length_scale)?;
            }
            // Recompute the baseline (and re-freeze) at the re-seeded geometry so
            // the joint solver's ψ seed, ρ seed, accept/reject gate, and the
            // frozen-baseline fallback all start from the better basin.
            best = fit_term_collection_forspec(
                data,
                y.view(),
                weights.view(),
                offset.view(),
                &resolvedspec,
                family.clone(),
                &baseline_options,
            )?;
            resolvedspec = freeze_term_collection_from_design(&resolvedspec, &best.design)?;
            // A re-seeded length scale can, in rare geometries, re-trigger the
            // freeze-time basis promotion (ThinPlate → pure Duchon); refresh the
            // spatial-term index list so the joint solve sees the current spec.
            spatial_terms = spatial_length_scale_term_indices(&resolvedspec);
        }
    }
    if spatial_terms.is_empty() {
        let fitted = fit_term_collection_forspecwith_heuristic_lambdas(
            data,
            y.view(),
            weights.view(),
            offset.view(),
            &resolvedspec,
            best.fit.lambdas.as_slice(),
            family,
            options,
        )?;
        return Ok(FittedTermCollectionWithSpec {
            fit: fitted.fit,
            design: fitted.design,
            resolvedspec,
            adaptive_diagnostics: fitted.adaptive_diagnostics,
            kappa_timing: None,
        });
    }
    let initial_score = fit_score(&best.fit);
    if !initial_score.is_finite() {
        log::debug!("[spatial-kappa] initial profiled score is non-finite");
    }
    // #1688: snapshot the per-term seed length scales the joint solve starts
    // from. If the joint solve neither improves the score NOR moves the geometry
    // off these values, it stalled and kept the frozen baseline — the precise
    // signature that triggers the ψ-window multistart rescue below. (A joint
    // solve that genuinely converged to its seed basin still moves ψ off the
    // exact seed, so this distinguishes a stall from a healthy local optimum.)
    let seed_length_scales: Vec<(usize, f64)> = spatial_terms
        .iter()
        .filter_map(|&t| get_spatial_length_scale(&resolvedspec, t).map(|ls| (t, ls)))
        .collect();
    let joint_result = try_exact_joint_spatial_length_scale_optimization(
        data,
        y.view(),
        weights.view(),
        offset.view(),
        &resolvedspec,
        &best,
        family.clone(),
        options,
        kappa_options,
        &spatial_terms,
    )
    .map(|opt| {
        opt.map(|fit| {
            let score = fit_score(&fit.fit);
            (fit, score)
        })
    });
    // #1074 / #2162: when the fixed-κ profiled seed is in a good, finite basin,
    // a HARD joint-solve failure caused only by inference-matrix finiteness (for
    // example a NaN frequentist covariance from κ railing into the
    // kernel-collapse corner during local polish) must not sink the whole fit.
    // The trial-evaluation path already classifies that exact condition as a
    // recoverable infeasible κ point and retreats; apply the same semantics to
    // the terminal polish by returning the finite frozen/pre-scan geometry with
    // ρ profiled at fixed κ. Keep unrelated errors fatal, and keep the older
    // pre-scan-improved rescue for cases where the multi-start scan found a
    // strictly better fixed-κ basin but the final polish produced no candidate.
    let terminal_joint_recoverable = matches!(
        &joint_result,
        Err(err) if is_recoverable_trial_point_error(err)
    );
    let exact_joint = if (prescan_improved && !matches!(joint_result, Ok(Some(_))))
        || terminal_joint_recoverable
    {
        let reason = match &joint_result {
            Err(e) => format!("error: {e}"),
            _ => "unavailable".to_string(),
        };
        log::info!(
            "[spatial-kappa] #1074/#2162 joint polish yielded no usable candidate \
             ({reason}); returning the fixed-κ profiled geometry (REML {initial_score:.5})"
        );
        FittedTermCollectionWithSpec {
            fit: best.fit,
            design: best.design,
            resolvedspec,
            adaptive_diagnostics: best.adaptive_diagnostics,
            kappa_timing: None,
        }
    } else {
        require_successful_spatial_optimization_result(initial_score, joint_result)?
    };

    // #1688: conditional joint-solve MULTISTART rescue.
    //
    // The joint [ρ, ψ] REML surface of an isotropic radial GP smooth is genuinely
    // multimodal in the kernel range, and the local ARC/BFGS solver descends into
    // whichever basin holds its seed. When the auto-seed (`max_range/√n`, the
    // wiggly side since #1629) lands in a flat valley between the two basins, the
    // local solver STALLS — it returns NonConverged at a large gradient and keeps
    // the frozen baseline geometry, so the realized score never improves on the
    // seed's profiled REML (the observed ν=3/2 boundary over-smoothing: edge RMSE
    // 4× the interior, REML ~7 nats worse than the reachable global basin).
    //
    // The #1074 pre-scan does not rescue this case: it ranks restart seeds by
    // FIXED-κ profiled REML (ρ-opt at fixed ψ), which is a poor predictor of the
    // JOINT outcome reachable from a seed — the global-basin seeds score WORSE on
    // the profiled proxy than the stalled auto-seed basin, so the strict-
    // improvement gate adopts none of them. The only honest ranking is the
    // realized joint REML itself.
    //
    // So: only when the primary joint solve produced NO REML improvement over its
    // baseline (the exact stall signature) do we re-run the full baseline → joint
    // sequence from a handful of seeds spanning the data-derived ψ window (long-
    // range corner through short-range corner) and keep the best REALIZED joint
    // score. This fires on a vanishing fraction of fits — a healthy joint solve
    // that moves ψ off the seed improves the score and skips the rescue entirely,
    // so the matern/GP fast path #1688 is about keeps its speed. Isotropic/non-CC
    // only: anisotropic and constant-curvature terms carry their own seeding.
    let exact_joint = {
        let primary_score = fit_score(&exact_joint.fit);
        let improved = primary_score.is_finite()
            && initial_score.is_finite()
            && primary_score < initial_score - 1e-7 * initial_score.abs().max(1.0);
        // Base the eligibility checks and the restart seeds on the realized
        // (frozen, κ-optimized) spec the primary path produced: it carries the
        // same term structure with resolved centers, and the parent `resolvedspec`
        // has already been moved into `exact_joint` on the pre-scan fallback path.
        let base_spec = exact_joint.resolvedspec.clone();
        // Did the joint solve keep the frozen baseline geometry (no length scale
        // moved off its seed)? That, combined with no score gain, is the stall.
        let geometry_unchanged = !seed_length_scales.is_empty()
            && seed_length_scales.iter().all(|&(t, seed_ls)| {
                get_spatial_length_scale(&base_spec, t)
                    .is_some_and(|ls| (ls - seed_ls).abs() <= 1e-6 * seed_ls.abs().max(1.0))
            });
        let eligible = !improved
            && geometry_unchanged
            && !has_aniso_terms(&base_spec, &spatial_terms)
            && constant_curvature_term_indices(&base_spec).is_empty()
            && spatial_terms
                .iter()
                .any(|&t| get_spatial_length_scale(&base_spec, t).is_some());
        if eligible {
            log::info!(
                "[spatial-kappa] #1688 joint solve stalled at REML {primary_score:.5} \
                 (no improvement over baseline {initial_score:.5}); running ψ-window \
                 multistart rescue across {} seeds",
                JOINT_RESTART_WINDOW_FRACTIONS.len(),
            );
            let mut best_fit = exact_joint;
            // Lower REML is better; the incumbent is the primary stalled result.
            let mut best_score = primary_score;
            for &fraction in JOINT_RESTART_WINDOW_FRACTIONS.iter() {
                match joint_solve_from_window_fraction(
                    data,
                    y.view(),
                    weights.view(),
                    offset.view(),
                    &base_spec,
                    &spatial_terms,
                    fraction,
                    &family,
                    options,
                    &baseline_options,
                    kappa_options,
                ) {
                    Ok(Some((candidate, score))) => {
                        if score.is_finite()
                            && (!best_score.is_finite()
                                || score < best_score - 1e-7 * best_score.abs().max(1.0))
                        {
                            log::info!(
                                "[spatial-kappa] #1688 multistart seed (ψ-window {fraction:.2}) \
                                 reached REML {score:.5}, improving on {best_score:.5}",
                            );
                            best_score = score;
                            best_fit = candidate;
                        }
                    }
                    // Infeasible seed geometry — skip this restart point.
                    Ok(None) => {}
                    // A restart's full re-fit can hit a genuine fatal error; the
                    // incumbent (the primary result) is already valid, so log and
                    // keep going rather than sinking the whole fit on one seed.
                    Err(e) => {
                        log::info!(
                            "[spatial-kappa] #1688 multistart seed (ψ-window {fraction:.2}) \
                             failed ({e}); skipping"
                        );
                    }
                }
            }
            best_fit
        } else {
            exact_joint
        }
    };

    log_spatial_aniso_scales(&exact_joint.resolvedspec);
    Ok(exact_joint)
}

/// The end-to-end curvature-as-an-estimand report for one `curv(...)` smooth:
/// the fitted κ̂, its profile-likelihood confidence interval, the interior
/// κ = 0 likelihood-ratio flatness test, and the topology-free geometry
/// verdict. This is the #944 headline — it turns "we chose hyperbolic space"
/// into "κ̂ = −1.8 (95% CI −2.6, −1.1), flat rejected at p = …".
#[derive(Clone, Debug)]
pub struct CurvatureInference {
    /// Smooth-term index of the `curv(...)` term this report is about.
    pub term_idx: usize,
    /// The fitted signed sectional curvature κ̂ (the outer optimiser's argmin of
    /// the profiled REML/LAML criterion over κ).
    pub kappa_hat: f64,
    /// Profile-likelihood CI for κ and the geometry verdict from its sign.
    pub ci: gam_geometry::curvature_estimand::KappaProfileCi,
    /// Interior-point κ = 0 likelihood-ratio flatness test (full χ²₁, no
    /// half-χ² boundary correction — κ = 0 is an interior point of the
    /// `S^d ← ℝ^d → H^d` family).
    pub flatness: gam_geometry::curvature_estimand::FlatnessTest,
}

/// Compute the #944 curvature inference for the constant-curvature smooth at
/// `term_idx`, given the already-fitted resolved spec (carrying κ̂) and the same
/// fit inputs used to produce it.
///
/// The profiled criterion `V_p(κ) = max_{ρ} V(κ, ρ)` is evaluated as an oracle:
/// for each probe κ, pin the term's curvature to κ, fit with κ-optimisation
/// **disabled** (so only the smoothing parameters ρ are profiled), and read the
/// resulting `reml_score` (the negative-log-evidence the outer loop minimises,
/// so κ̂ is its argmin). The exact same criterion the joint κ-fit minimised —
/// the only difference is which coordinates move — so κ̂ is a genuine stationary
/// point of this oracle. The statistics (profile-CI walk, interior κ=0 LR test)
/// are then the principled likelihood-set / Wilks constructions in
/// [`gam_geometry::curvature_estimand`].
///
/// `v_pp` (the initial Wald step size) is taken from a central finite difference
/// of `V_p` at κ̂; the CI itself is the exact χ²₁ likelihood crossing, not the
/// Wald ellipsoid, so this only sizes the first bracket step.
pub fn curvature_inference_forspec(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    term_idx: usize,
    family: LikelihoodSpec,
    options: &FitOptions,
    level: f64,
) -> Result<CurvatureInference, EstimationError> {
    let kappa_hat = get_constant_curvature_kappa(resolvedspec, term_idx).ok_or_else(|| {
        EstimationError::InvalidInput(format!(
            "curvature_inference_forspec: term {term_idx} is not a constant-curvature smooth"
        ))
    })?;
    let (kappa_min, kappa_max) = constant_curvature_kappa_bounds(data, resolvedspec, term_idx);

    // Profiled criterion oracle V_p(κ) for the CI walk and the κ = 0 flatness LR
    // test. This MUST be the same criterion that selected κ̂, otherwise the
    // statistics are inconsistent with the point estimate. For a constant-
    // curvature smooth κ̂ is chosen by the κ-FAIR criterion
    // (`constant_curvature_kappa_fair_sign_score`, #1464) — the raw
    // `fixed_kappa_profiled_reml_score` is sign-BLIND in κ on a generic radial
    // signal (the +κ chart's distance-compression is a uniformly better
    // interpolator regardless of the true sign, so raw V_p rails to the +chart
    // bound for both signs and would report `V_p(0) < V_p(κ̂)`, i.e. a flatness
    // p-value of 1 even for genuinely curved truth). We therefore evaluate the
    // CI/flatness criterion with the κ-fair score, which subtracts the design's
    // generic radial-peak-fitting power so only the genuine curvature-shape
    // signal remains and `V_fair(κ̂) < V_fair(0)` for curved truth. The κ-fair
    // score is the basis-level criterion; resolve this term's feature columns and
    // base spec so each κ-probe scores the production constant-curvature basis.
    // Use the κ-fair criterion for the CI/flatness ONLY when κ̂ is in the
    // hyperbolic basin (κ̂ < 0) — the regime where κ̂ was chosen by the κ-fair
    // fast-path (`constant_curvature_kappa_fair_argmin`), so the flatness LR and
    // CI must use the SAME criterion to be consistent (raw V_p is sign-blind and
    // would report `V_p(0) < V_p(κ̂)`, a flatness p-value of 1 even for genuinely
    // hyperbolic truth). For κ̂ ≥ 0 (spherical via the joint solver, or a genuinely
    // flat κ̂ ≈ 0) the raw production V_p is the right, scale-correct criterion and
    // already sizes flatness correctly, so we keep it — this preserves the
    // spherical and flat statistics unchanged.
    let cc_fair_inputs: Option<(Array2<f64>, gam_terms::basis::ConstantCurvatureBasisSpec)> =
        if kappa_hat < 0.0 {
            match resolvedspec.smooth_terms.get(term_idx).map(|t| &t.basis) {
                Some(SmoothBasisSpec::ConstantCurvature {
                    feature_cols, spec, ..
                }) => select_columns(data, feature_cols)
                    .ok()
                    .map(|x| (x, spec.clone())),
                _ => None,
            }
        } else {
            None
        };

    // Memoize across κ probes. The CI walk's bracketing/bisection, the
    // central-difference v_pp seed, and the flatness LR test all re-evaluate
    // the criterion at the SAME κ, so caching by the raw bits of κ removes
    // redundant evaluations with no change to the statistical answer.
    let v_p_cache: std::cell::RefCell<std::collections::HashMap<u64, f64>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    let v_p = |kappa: f64| -> Result<f64, String> {
        if !kappa.is_finite() {
            return Err(format!("V_p probed a non-finite κ = {kappa}"));
        }
        let key = kappa.to_bits();
        if let Some(&cached) = v_p_cache.borrow().get(&key) {
            return Ok(cached);
        }
        let score = if let Some((x_term, base_spec)) = &cc_fair_inputs {
            let mut probe_spec = base_spec.clone();
            probe_spec.kappa = kappa;
            gam_terms::basis::constant_curvature_kappa_fair_sign_score(x_term.view(), y, &probe_spec)
                .map_err(|e| format!("κ-fair criterion at κ={kappa} failed: {e}"))?
        } else {
            fixed_kappa_profiled_reml_score(
                data,
                y,
                weights,
                offset,
                resolvedspec,
                term_idx,
                kappa,
                family.clone(),
                options,
            )
            .map_err(|e| format!("V_p fixed-κ fit at κ={kappa} failed: {e}"))?
        };
        v_p_cache.borrow_mut().insert(key, score);
        Ok(score)
    };

    // Wald step seed: central FD of V_p at κ̂ (only sizes the first bracket; the
    // CI is the exact likelihood crossing). Step a small fraction of the κ
    // window so the FD straddles κ̂ without leaving the chart.
    let h = (1e-3 * (kappa_max - kappa_min)).max(1e-4);
    let v_pp = match (v_p(kappa_hat + h), v_p(kappa_hat), v_p(kappa_hat - h)) {
        (Ok(vp), Ok(v0), Ok(vm)) => (vp - 2.0 * v0 + vm) / (h * h),
        _ => f64::NAN, // profile_ci_walk falls back to a default step
    };

    let ci = gam_geometry::curvature_estimand::profile_ci_walk(
        &v_p, kappa_hat, v_pp, kappa_min, kappa_max, level, 1e-4,
    )
    .map_err(EstimationError::InvalidInput)?;
    let flatness = gam_geometry::curvature_estimand::flatness_lr_test(&v_p, kappa_hat)
        .map_err(EstimationError::InvalidInput)?;

    Ok(CurvatureInference {
        term_idx,
        kappa_hat,
        ci,
        flatness,
    })
}

/// Provenance tag for the smooth-term significance correction (#1063): which
/// statistic the reported p-value is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothLrCorrection {
    /// A per-term LR statistic corrected by the full estimated-λ Lawley factor,
    /// including the ρ̂-sampling-variation contribution from the regularized
    /// inverse REML/LAML outer Hessian.
    LawleyLrEstimatedLambda,
    /// A per-term likelihood-ratio statistic `W = 2(ℓ_full − ℓ_null)` that has
    /// been Bartlett-corrected with the fixed-λ Lawley factor `c = E[W|λ]/d`
    /// (`W* = W/c`, referenced against `χ²_d`). This is used only when the
    /// estimated-λ handoff is unavailable.
    LawleyLrFixedLambda,
    /// No second-order correction was applied — either the family has no
    /// closed-form Lawley cumulant jets or the null refit did not converge — so
    /// the uncorrected `χ²_d` of the raw LR statistic stands.
    None,
}

impl SmoothLrCorrection {
    /// The serialized provenance label surfaced in the summary table.
    pub fn label(self) -> &'static str {
        match self {
            SmoothLrCorrection::LawleyLrEstimatedLambda => "lawley_lr_estimated_lambda",
            SmoothLrCorrection::LawleyLrFixedLambda => "lawley_lr_fixed_lambda",
            SmoothLrCorrection::None => "none",
        }
    }
}

/// The Bartlett-corrected per-term significance report for one penalized smooth
/// term (#1063). Unlike the summary table's Wood rank-truncated **Wald**
/// statistic, this is a genuine **likelihood-ratio** statistic from a
/// constrained refit (the smooth dropped), so the exact Lawley LR Bartlett
/// factor corrects the right quantity.
#[derive(Clone, Debug)]
pub struct SmoothTermLrInference {
    /// Smooth-term name (matches the summary row).
    pub name: String,
    /// Smooth-term index within `resolvedspec.smooth_terms`.
    pub term_idx: usize,
    /// The uncorrected likelihood-ratio statistic `W = 2(ℓ_full − ℓ_null)`,
    /// floored at zero (a non-negative LR by construction).
    pub statistic_lr: f64,
    /// Reference degrees of freedom `d` (the Wood truncation `tr(F)²/tr(F²)` on
    /// the term's influence block, falling back to the term EDF).
    pub ref_df: f64,
    /// Lawley LR Bartlett factor `c = E[W]/d = 1 + Δε/d` when computable, else
    /// `1.0` (no correction).
    pub bartlett_factor: f64,
    /// Fixed-λ conditional factor `c_cond = 1 + Δε(ρ̂)/d` when the estimated-λ
    /// correction was applied. `None` means the applied factor was either the
    /// fixed-λ factor itself or no Lawley correction was available.
    pub bartlett_factor_conditional: Option<f64>,
    /// Increment in Lawley's LR mean shift due solely to ρ̂ sampling variation,
    /// `0.5 * tr(H_Δε Cov(ρ̂))`, when estimated-λ correction was applied.
    pub rho_variation_shift: Option<f64>,
    /// Bartlett-corrected statistic `W* = W / c`.
    pub statistic_corrected: f64,
    /// Uncorrected p-value `P(χ²_d > W)`.
    pub p_value_uncorrected: f64,
    /// Corrected p-value `P(χ²_d > W*)`; equals the uncorrected value when no
    /// correction was applied.
    pub p_value_corrected: f64,
    /// Whether the second-order correction is **material** (#939 deliverable 4):
    /// the per-test diagnostic "is `n` too small for first-order inference
    /// *here*?". `true` when a correction was applied and it moves the result by
    /// more than [`SMOOTH_LR_MATERIAL_THRESHOLD`] — measured as the larger of the
    /// relative Bartlett-factor distance from one `|c − 1|` and the relative
    /// p-value change `|p* − p| / max(p, p*, ε)`. `false` when `correction` is
    /// [`SmoothLrCorrection::None`] (no correction was applied).
    pub material: bool,
    /// Which statistic the corrected p-value is built from.
    pub correction: SmoothLrCorrection,
}

/// The materiality threshold for [`SmoothTermLrInference::material`] (#939
/// deliverable 4): a correction is flagged material when it changes the result
/// by more than 10%.
pub const SMOOTH_LR_MATERIAL_THRESHOLD: f64 = 0.10;

/// Build `S_b = lambda_b * S_b^unit` as global `p_total x p_total` matrices in
/// exactly the fitted rho/lambda ordering. This is the narrow handoff the
/// estimated-lambda Lawley correction needs: the same `design.penalties` order
/// already paired with `fit.lambdas`, without changing #740's outer-Hessian
/// algebra or the production penalty assembly.
fn fitted_rho_penalty_components(
    penalties: &[BlockwisePenalty],
    lambdas: &[f64],
    p_total: usize,
) -> Result<Vec<gam_terms::inference::lawley::RhoPenaltyComponent>, EstimationError> {
    if penalties.len() != lambdas.len() {
        return Err(EstimationError::InvalidInput(format!(
            "smooth_term_lr_inference: penalty/lambda count mismatch ({} penalties, {} lambdas)",
            penalties.len(),
            lambdas.len()
        )));
    }
    let mut components = Vec::with_capacity(penalties.len());
    for (idx, (penalty, &lambda)) in penalties.iter().zip(lambdas.iter()).enumerate() {
        if !(lambda.is_finite() && lambda >= 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "smooth_term_lr_inference: lambda[{idx}] is invalid: {lambda}"
            )));
        }
        let r = &penalty.col_range;
        if r.end > p_total {
            return Err(EstimationError::InvalidInput(format!(
                "smooth_term_lr_inference: penalty[{idx}] range {:?} exceeds coefficient dimension {p_total}",
                r
            )));
        }
        let mut s_component = Array2::<f64>::zeros((p_total, p_total));
        s_component
            .slice_mut(s![r.start..r.end, r.start..r.end])
            .scaled_add(lambda, &penalty.local);
        components.push(gam_terms::inference::lawley::RhoPenaltyComponent { s_component });
    }
    Ok(components)
}

/// The end-to-end per-term likelihood-ratio significance report for every
/// penalized (shape-unconstrained) smooth term in a fitted model, magically
/// Bartlett-corrected when the family carries closed-form Lawley cumulant jets
/// (#1063, follow-up to #939).
///
/// # Why an LR statistic (not the summary Wald)
///
/// The summary table's `wood_smooth_test` is Wood's rank-truncated **Wald**
/// statistic `T = β̂'Σ̂⁻β̂`. Lawley's ε corrects the **likelihood-ratio**
/// statistic, and under penalization the Wald form is already a weighted χ²
/// whose second-order mean is *not* `d + Δε` — dividing `T` by the LR factor
/// would correct the wrong statistic. The principled route (#1063 Option 1) is
/// to compute a real per-term LR statistic by a constrained refit and correct
/// *that*:
///
/// ```text
/// W = 2(ℓ_full − ℓ_null),   W* = W / c,   c = 1 + Δε/d,   p = P(χ²_d > W*).
/// ```
///
/// # Method
///
/// 1. Fit the full model and read `ℓ_full` and the per-term coefficient ranges /
///    EDF / influence block. The full design's column layout fixes the tested
///    block for the Lawley factor.
/// 2. For each penalized smooth term, refit a null model with that term dropped
///    from the spec; `W = max(2(ℓ_full − ℓ_null), 0)`.
/// 3. The reference d.f. `d` is the Wood truncation `tr(F)²/tr(F²)` on the
///    term's influence block (the same `ref_df` the summary Wald row reports),
///    floored at `max(edf, null_dim, 1)`: this LR test drops the whole term, so
///    `d` is at least the dimension the term spans when present (its null-space
///    dimension, never below 1). The non-symmetric `tr(F²)` can collapse toward
///    0 at a shrunk-to-null fit and violate that bound — see the inline note at
///    the `ref_df` binding.
/// 4. When the family has closed-form cumulant jets, evaluate Lawley's ε at the
///    **null** linear predictor (an expectation evaluated at the null fit), fold
///    the full λ-scaled penalty `S_λ` into the information, and Bartlett-correct
///    `W` with [`gam_terms::inference::lawley::lawley_lr_bartlett_factor`]. The
///    null annihilates the tested block's penalty (`S_λ β₀ = 0` on that block),
///    so the penalized Lawley expansion applies verbatim.
/// 5. Otherwise (no closed-form jets, or a null refit that did not converge) the
///    uncorrected `χ²_d` stands with provenance `none` — never weakened.
///
/// Random-effect smooths and shape-constrained smooths are skipped (their tests
/// are not a central-χ² LR), matching the summary table's policy.
pub fn smooth_term_lr_inference_forspec(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<Vec<SmoothTermLrInference>, EstimationError> {
    use gam_terms::inference::lawley::{
        LAWLEY_PAIR_MATRIX_MAX_ROWS, known_scale_expected_jets_with_dispersion,
        lawley_lr_bartlett_factor, lawley_lr_mean_shift_with_rho_variation,
    };

    let n = data.nrows();
    // Full fit: ℓ_full, the per-term coefficient ranges/EDF/influence, and the
    // full design whose column layout fixes each tested block for Lawley.
    let full = fit_term_collection_forspec(
        data,
        y,
        weights,
        offset,
        resolvedspec,
        family.clone(),
        options,
    )?;
    let ll_full = full.fit.log_likelihood;
    let p_total = full.design.design.ncols();
    let lambdas = full.fit.lambdas.as_slice().ok_or_else(|| {
        EstimationError::InvalidInput(
            "smooth_term_lr_inference: non-contiguous lambda vector".to_string(),
        )
    })?;
    let s_lambda = weighted_blockwise_penalty_sum(&full.design.penalties, lambdas, p_total);
    let rho_penalty_components =
        fitted_rho_penalty_components(&full.design.penalties, lambdas, p_total)?;
    let rho_covariance = full.fit.artifacts.rho_covariance.as_ref().filter(|cov| {
        cov.nrows() == rho_penalty_components.len() && cov.ncols() == rho_penalty_components.len()
    });
    // Full design as a dense n×p array for the Lawley pair-matrix reduction.
    let full_design_dense = full.design.design.to_dense();
    let influence = full.fit.coefficient_influence();
    let family_disp = lawley_dispersion_for_family(&family, &full.fit);

    // The penalty-block cursor walks the same block order the summary table
    // uses: any leading linear/random-effect penalty blocks first (skipped
    // here), then smooth terms.
    let mut penalty_cursor = full.design.leading_penalty_blocks_before_smooth();
    let mut out = Vec::<SmoothTermLrInference>::new();
    for (term_idx, design_term) in full.design.smooth.terms.iter().enumerate() {
        let k = design_term.penalties_local.len();
        let block_start = penalty_cursor;
        penalty_cursor += k;
        // Shape-constrained smooths get no central-χ² LR (cone-projected
        // boundary test); the summary table skips them too.
        if design_term.shape != ShapeConstraint::None {
            continue;
        }
        let coeff_range = design_term.coeff_range.clone();
        if coeff_range.start >= coeff_range.end || coeff_range.end > p_total {
            continue;
        }
        // Per-term EDF for the χ² reference df FALLBACK (used only when the
        // influence matrix `F` is unavailable). Route through `per_term_edf`,
        // which uses the ADDITIVE per-block trace channel
        // (`|coeff_range| − Σ_{kk∈term} tr_kk`) and caps at the model total,
        // rather than the raw `edf_by_block` block-sum `Σ_{kk}(rank_kk − tr_kk)`.
        // For a multi-penalty term (te/ti/double-penalty) the penalties share one
        // coefficient range, so the rank-based block-sum OVER-COUNTS the term EDF
        // (Σ rank_kk > |coeff_range|) and would inflate the LR reference df,
        // biasing the smooth-term test conservative on large/sparse fits where `F`
        // is not materialised. (Same per-block over-count class as the multinomial
        // `edf_per_class` fix.)
        let edf = full.fit.per_term_edf(coeff_range.clone(), block_start, k);
        // The term's **joint** unpenalized null-space dimension: the coefficient
        // directions penalized by *no* active penalty — the polynomial part a
        // penalized smooth always carries when present, which no penalty can
        // shrink. This is `dim(∩_k null(S_k)) = p_local − rank(Σ_k S_k)`, the
        // INTERSECTION of the per-penalty null spaces, computed by
        // `wald_unpenalized_dim()` — the very same scalar the summary Wald test
        // (`wood_smooth_test`) floors its reference d.f. at, so the LR and Wald
        // tests reference a consistent d.f.
        //
        // It must NOT be `nullspace_dims.iter().sum()`: that *unions* the null
        // spaces (the #1360 defect — see `joint_unpenalized_dim`'s docs). A
        // double-penalty smooth carries a bending penalty (null space = its
        // polynomial part) plus a complementary null-space ridge (which penalizes
        // exactly that polynomial part), so the two null spaces are disjoint and
        // the joint null space is EMPTY (dim 0) — yet the per-penalty dims sum to
        // ~`p_local`. Flooring `ref_df` at that sum pins it to the full basis
        // dimension for every fit (e.g. 11 for a k=12 s(x)), making the LR test
        // badly conservative for genuine moderate signals while only accidentally
        // masking the collapse.
        let null_dim = design_term.wald_unpenalized_dim();
        // χ² reference d.f. for the whole-term LR test. The statistic W tests the
        // term present vs entirely absent, so the reference d.f. must be at least
        // the dimension the term spans when present — its joint null-space
        // dimension (`null_dim`), the effective d.f. it uses in the fit (`edf`),
        // and never below 1 (you cannot test "is this function present" with
        // fewer than one degree of freedom). The primary reference is Wood's
        // smoothing-selection-corrected `edf1 = 2·tr(F) − tr(F²)`
        // (`wood_reference_df`), which dominates in calibrated and high-power
        // fits and removes the post-selection left-tail anti-conservatism the raw
        // `edf` reference leaves behind. The floor guards the DEGENERATE collapse:
        // as REML shrinks a term fully onto its null space `edf1 → edf → ~0` and
        // the non-symmetric `F = H⁻¹X'WX` can make the block `tr(F²)` numerically
        // unreliable, so `wood_reference_df` returns `None` and the floor supplies
        // `max(edf, null_dim, 1)`. Without a floor, a positive `W` referenced
        // against `χ²_{~0}` would report a flat, shrunk-to-null term as MAXIMALLY
        // significant (`p ~ 1e-12`) — a Type-I error decided by a degenerate
        // reference d.f., not the data (#1766). The floor binds ONLY on that
        // collapse; `edf` and `null_dim` never exceed `edf1` in a healthy fit.
        // This mirrors the summary Wald path, which floors its reference at the
        // statistic's own rank for the same reason.
        //
        // Guard the ONE state where `edf1` cannot be trusted: a non-converged fit
        // whose influence-based `edf` has collapsed BELOW the term's guaranteed
        // floor. The flat-valley REML stall on an unidentified term (#1762) rails
        // the outer iterations out and returns an INCONSISTENT state — the term
        // still carries a large, wiggly `β` (so the refit LR statistic `W` is
        // large) yet `tr(F) = edf` reads ~0, a combination a genuine penalized
        // least-squares solution cannot produce (large `β` ⇒ `edf ≫ 0`).
        // Referencing that large `W` against `edf1 ≈ edf ≈ 0` manufactures
        // significance: measured null FPR ~0.62 on those stalled noise fits vs a
        // well-calibrated ~0.06 on the converged ones — the entire residual
        // miscalibration lives here. The summary Wald path already sidesteps it
        // (its statistic truncates to the ~0-`edf` rank and the term is skipped);
        // the whole-term LR statistic cannot self-truncate (it is a refit deviance
        // difference), so we widen its REFERENCE instead: floor `ref_df` at the
        // term's full basis dimension — the maximally-honest reference for an
        // untrusted term (it may occupy up to all its columns). A genuine effect
        // still rejects (`W` ≫ its column count); a tiny-`W` collapse still gives
        // `p ≈ 1`; only the spurious large-`W`/zero-`edf` artifact is neutralised.
        //
        // The trigger is NARROW on purpose. `outer_converged` alone is far too
        // broad: the null-space double-penalty's shrinkage λ routinely rails to
        // its ~1e13 ceiling on perfectly good fits (a `te` interaction, a
        // multi-term model), flipping `outer_converged` to false while `edf`
        // stays healthy (~3, ~9). Flooring THOSE at the full basis dimension
        // would bury real effects (a `te` with `W ≈ 49` on a 48-column basis).
        // Requiring BOTH non-convergence AND `edf < max(null_dim, 1)` — the term
        // shrunk below even its unpenalized floor, the inconsistency fingerprint —
        // fires on the zero-`edf` stall alone and leaves every healthy-`edf` fit
        // on the tight `edf1` reference. The underlying stall itself is #1762;
        // this keeps the inference honest on top of whatever fit it is handed.
        let edf_floor = (null_dim.max(1)) as f64;
        let untrusted_edf_collapse = !full.fit.outer_converged && edf < edf_floor;
        let unconverged_dim_floor = if untrusted_edf_collapse {
            coeff_range.len() as f64
        } else {
            0.0
        };
        let rho_uncertainty_df = wps_block_uncertainty_df(
            full.fit.weighted_gram(),
            full.fit.smoothing_correction(),
            &coeff_range,
            family_disp,
        );
        let ref_df = (wood_reference_df(influence, &coeff_range)
            .unwrap_or(0.0)
            .max(edf)
            + rho_uncertainty_df)
            .max(null_dim as f64)
            .max(unconverged_dim_floor)
            .max(1.0);
        if !(ref_df.is_finite() && ref_df > 0.0) {
            continue;
        }

        // Null model: drop this smooth term from the spec and refit. The term's
        // name pins which spec entry to remove (design and spec share names).
        let mut null_spec = resolvedspec.clone();
        let Some(spec_pos) = null_spec
            .smooth_terms
            .iter()
            .position(|t| t.name == design_term.name)
        else {
            continue;
        };
        null_spec.smooth_terms.remove(spec_pos);
        let null_fit = fit_term_collection_forspec(
            data,
            y,
            weights,
            offset,
            &null_spec,
            family.clone(),
            options,
        );
        let (statistic_lr, eta_null) = match null_fit {
            Ok(null) if null.fit.log_likelihood.is_finite() => {
                let w = (2.0 * (ll_full - null.fit.log_likelihood)).max(0.0);
                // η at the null fit: X_null β_null + offset (per-row linear
                // predictor; design-layout independent — Lawley reads it on the
                // full design rows).
                let mut eta = null.design.design.dot(&null.fit.beta);
                eta += &offset;
                (w, Some(eta))
            }
            _ => (f64::NAN, None),
        };

        let chi2 = statrs::distribution::ChiSquared::new(ref_df).ok();
        let p_uncorrected = match (chi2.as_ref(), statistic_lr.is_finite()) {
            (Some(dist), true) => {
                use statrs::distribution::ContinuousCDF;
                (1.0 - dist.cdf(statistic_lr)).clamp(0.0, 1.0)
            }
            _ => f64::NAN,
        };

        // Magic Bartlett correction: only when the LR statistic is finite, the
        // family has closed-form jets, n is in the resolvable regime, and the
        // factor is computable. Otherwise the uncorrected χ² stands.
        let mut bartlett_factor = 1.0;
        let mut bartlett_factor_conditional = None;
        let mut rho_variation_shift = None;
        let mut statistic_corrected = statistic_lr;
        let mut p_corrected = p_uncorrected;
        let mut correction = SmoothLrCorrection::None;
        if let (Some(eta), true, true) = (
            eta_null.as_ref(),
            statistic_lr.is_finite(),
            n <= LAWLEY_PAIR_MATRIX_MAX_ROWS,
        ) {
            let kappas: Option<Vec<_>> = (0..n)
                .map(|i| {
                    known_scale_expected_jets_with_dispersion(&family, eta[i], family_disp)
                        .and_then(|jets| jets.kappas().ok())
                })
                .collect();
            if let (Some(kappas), Some(dist)) = (kappas, chi2.as_ref()) {
                let fixed_factor = lawley_lr_bartlett_factor(
                    full_design_dense.view(),
                    &kappas,
                    Some(s_lambda.view()),
                    coeff_range.clone(),
                    ref_df,
                );
                if let Ok(c_cond) = fixed_factor
                    && c_cond.is_finite()
                    && c_cond > 0.0
                {
                    let mut c_applied = c_cond;
                    correction = SmoothLrCorrection::LawleyLrFixedLambda;
                    if let Some(cov) = rho_covariance
                        && let Ok(total_shift) = lawley_lr_mean_shift_with_rho_variation(
                            full_design_dense.view(),
                            &kappas,
                            s_lambda.view(),
                            coeff_range.clone(),
                            &rho_penalty_components,
                            cov.view(),
                        )
                    {
                        let mean_w = ref_df + total_shift;
                        if let Some(c_est) =
                            gam_terms::inference::higher_order::bartlett_factor_from_mean(
                                mean_w, ref_df,
                            )
                            && c_est.is_finite()
                            && c_est > 0.0
                        {
                            let conditional_shift = (c_cond - 1.0) * ref_df;
                            c_applied = c_est;
                            bartlett_factor_conditional = Some(c_cond);
                            rho_variation_shift = Some(total_shift - conditional_shift);
                            correction = SmoothLrCorrection::LawleyLrEstimatedLambda;
                        }
                    }
                    use statrs::distribution::ContinuousCDF;
                    bartlett_factor = c_applied;
                    statistic_corrected = statistic_lr / c_applied;
                    p_corrected = (1.0 - dist.cdf(statistic_corrected)).clamp(0.0, 1.0);
                }
            }
        }

        // Materiality (#939 deliverable 4): only when a correction was actually
        // applied, flagged when it moves the result by more than the 10%
        // threshold — by the Bartlett factor's distance from one OR the relative
        // p-value shift, whichever is larger (a factor near one can still flip a
        // p-value sitting on the α boundary, and vice versa).
        let material = match correction {
            SmoothLrCorrection::LawleyLrEstimatedLambda
            | SmoothLrCorrection::LawleyLrFixedLambda => {
                let factor_move = (bartlett_factor - 1.0).abs();
                let p_denom = p_uncorrected.max(p_corrected).max(f64::MIN_POSITIVE);
                let p_move = if p_uncorrected.is_finite() && p_corrected.is_finite() {
                    (p_corrected - p_uncorrected).abs() / p_denom
                } else {
                    0.0
                };
                factor_move > SMOOTH_LR_MATERIAL_THRESHOLD || p_move > SMOOTH_LR_MATERIAL_THRESHOLD
            }
            SmoothLrCorrection::None => false,
        };

        out.push(SmoothTermLrInference {
            name: design_term.name.clone(),
            term_idx,
            statistic_lr,
            ref_df,
            bartlett_factor,
            bartlett_factor_conditional,
            rho_variation_shift,
            statistic_corrected,
            p_value_uncorrected: p_uncorrected,
            p_value_corrected: p_corrected,
            material,
            correction,
        });
    }
    Ok(out)
}

/// The dispersion `φ` Lawley needs for the family's cumulant scaling: Gaussian
/// `σ̂²`, Gamma `1/shape`, and `1` for the scale-free Poisson/Binomial.
fn lawley_dispersion_for_family(family: &LikelihoodSpec, fit: &UnifiedFitResult) -> f64 {
    match family.response {
        gam_spec::ResponseFamily::Gaussian => {
            let sd = fit.standard_deviation;
            (sd * sd).max(f64::MIN_POSITIVE)
        }
        gam_spec::ResponseFamily::Gamma => {
            let shape = fit.standard_deviation;
            if shape.is_finite() && shape > 0.0 {
                1.0 / shape
            } else {
                1.0
            }
        }
        _ => 1.0,
    }
}

fn wps_block_uncertainty_df(
    weighted_gram: Option<&Array2<f64>>,
    smoothing_correction: Option<&Array2<f64>>,
    coeff_range: &Range<usize>,
    phi: f64,
) -> f64 {
    let (Some(xwx), Some(corr)) = (weighted_gram, smoothing_correction) else {
        return 0.0;
    };
    let (start, end) = (coeff_range.start, coeff_range.end);
    if start >= end
        || end > xwx.nrows()
        || end > xwx.ncols()
        || end > corr.nrows()
        || end > corr.ncols()
        || !(phi.is_finite() && phi > 0.0)
    {
        return 0.0;
    }

    let mut trace = 0.0;
    for i in start..end {
        for j in start..end {
            trace += xwx[[i, j]] * corr[[j, i]];
        }
    }
    trace /= phi;
    if trace.is_finite() && trace > 0.0 {
        trace
    } else {
        0.0
    }
}

/// Wood's smoothing-selection-corrected reference d.f. `edf1 = 2·tr(F_jj) −
/// tr(F_jj²)` on the coefficient-influence block `F = H⁻¹ X'WX` restricted to
/// `coeff_range` (mgcv's `edf1`). This is the reference degrees of freedom
/// Wood (2013) recommends for a smooth-component significance test when the
/// smoothing parameter is *estimated* (as it always is here): it inflates the
/// raw effective d.f. `edf = tr(F)` by the selection-bias term
/// `tr(F) − tr(F²) = Σ_i λ_i(1 − λ_i) ≥ 0` (the F-block eigenvalues `λ_i` lie in
/// `[0, 1]`), which is exactly the excess variance the fit spends locating a
/// term on noise. Referencing the whole-term LR statistic against this larger
/// `edf1` instead of the raw `edf` removes the post-selection left-tail
/// anti-conservatism (a REML fit that turns a smooth *on* against pure noise no
/// longer over-rejects): with the raw `edf` reference the mean null p-value is a
/// calibrated ~0.5 but the 5%-level false-positive rate runs ~0.15; the `edf1`
/// correction pulls it toward the nominal α while leaving the collapsed-term
/// verdict (`W ≈ 0 ⇒ p ≈ 1`) untouched.
///
/// `edf1 ∈ [edf, 2·edf]` analytically, so it never collapses toward 0 the way
/// the earlier Satterthwaite ratio `tr(F)²/tr(F²)` did when the non-symmetric
/// block's `tr(F²)` ran away (the #1766 degeneracy). The `.max(tr)` guard keeps
/// it ≥ `edf` even if a corrupted block drives `tr(F²)` above `2·tr(F)`. Returns
/// `None` when the influence block is unavailable or its trace is non-finite /
/// non-positive (a fully shrunk term), in which case the caller falls back to
/// the `max(edf, null_dim, 1)` floor.
fn wood_reference_df(influence: Option<&Array2<f64>>, coeff_range: &Range<usize>) -> Option<f64> {
    let f = influence?;
    let (start, end) = (coeff_range.start, coeff_range.end);
    if start >= end || end > f.nrows() || end > f.ncols() {
        return None;
    }
    let block = f.slice(s![start..end, start..end]);
    let tr = (0..block.nrows()).map(|i| block[[i, i]]).sum::<f64>();
    let tr2 = block.dot(&block).diag().sum();
    (tr.is_finite() && tr2.is_finite() && tr > 0.0)
        .then(|| (2.0 * tr - tr2).max(tr).max(1e-12))
}
