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
        Option<std::sync::Arc<crate::terms::basis::ImplicitDesignPsiDerivative>>,
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
                ..
            } = &smooth_term.metadata
            else {
                return Ok(None);
            };
            crate::basis::build_duchon_basis_log_kappa_derivativeswith_collocationwithworkspace(
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
            let qt_s = crate::linalg::faer_ndarray::fast_atb(q, &s_local);
            Some(crate::linalg::faer_ndarray::fast_ab(&qt_s, q))
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
    latent: std::sync::Arc<crate::terms::latent::LatentCoordValues>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    latent_terms: &[crate::types::SmoothTermIdx],
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
        ) => crate::terms::basis::LatentCoordDesignDerivative::new_matern(
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
        ) => crate::terms::basis::LatentCoordDesignDerivative::new_duchon(
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
        ) if matches!(*method, crate::basis::SphereMethod::Wahba) => {
            crate::terms::basis::LatentCoordDesignDerivative::new_sphere(
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
                crate::terms::basis::LatentCoordDesignDerivative::new_periodic_bspline(
                    latent.clone(),
                    (*domain_start, *domain_start + *period),
                    effective_degree,
                    *num_basis,
                    identifiability_transform.clone(),
                )
                .map_err(EstimationError::from)?
            } else {
                crate::terms::basis::LatentCoordDesignDerivative::new_tensor_bspline(
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
        ) => crate::terms::basis::LatentCoordDesignDerivative::new_tensor_bspline(
            latent.clone(),
            knots.clone(),
            degrees.clone(),
            identifiability_transform.clone(),
        )
        .map_err(EstimationError::from)?,
        (SmoothBasisSpec::Pca { .. }, BasisMetadata::Pca { basis_matrix, .. }) => {
            crate::terms::basis::LatentCoordDesignDerivative::new_pca(
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
            crate::estimate::reml::HyperDesignDerivative::from_latent_coord(
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
        let zero_x = crate::estimate::reml::HyperDesignDerivative::from(Array2::<f64>::zeros((
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
    id_mode: &crate::terms::latent::LatentIdMode,
    latent_dim: usize,
) -> usize {
    use crate::terms::latent::{AuxPriorStrength, LatentIdMode};
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
        // The behavioral head appends one (1 + d) coefficient block per
        // η-channel, plus the composed per-axis ARD log-precisions.
        LatentIdMode::AuxOutcome { head, .. } => head.n_coeffs(latent_dim) + latent_dim,
        LatentIdMode::None => 0,
    }
}

fn latent_coord_initial_direct_hypers(
    id_mode: &crate::terms::latent::LatentIdMode,
    latent_dim: usize,
) -> Result<Array1<f64>, EstimationError> {
    use crate::terms::latent::{AuxPriorStrength, LatentIdMode};
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
    latent: &crate::terms::latent::LatentCoordValues,
) -> Result<LatentIdObjectiveContribution, EstimationError> {
    use crate::terms::latent::{AuxPriorStrength, LatentIdMode, aux_prior_targets};
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
            cost += 0.5 * mu * q - 0.5 * n_obs as f64 * log_mu;

            let projected_residual = aux_prior_targets(residual.view(), u.view(), *family)
                .map_err(EstimationError::InvalidInput)?;
            let grad_base = residual - projected_residual;
            for n in 0..n_obs {
                for axis in 0..latent_dim {
                    gradient[t_start + n * latent_dim + axis] += mu * grad_base[[n, axis]];
                }
            }
            if matches!(strength, AuxPriorStrength::Auto) {
                gradient[direct_start] += 0.5 * mu * q - 0.5 * n_obs as f64;
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
        LatentIdMode::AuxPrior { .. } | LatentIdMode::None => {}
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
    latent: &crate::terms::latent::LatentCoordValues,
    eval: &mut (
        f64,
        Array1<f64>,
        crate::solver::rho_optimizer::HessianResult,
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
        eval.2 = crate::solver::rho_optimizer::HessianResult::Unavailable;
    }
    Ok(())
}

fn analytic_penalty_objective_contribution(
    theta: &Array1<f64>,
    rho_dim: usize,
    latent: &crate::terms::latent::LatentCoordValues,
    registry: &crate::terms::AnalyticPenaltyRegistry,
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
            crate::terms::PenaltyTier::Psi => {
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
            crate::terms::PenaltyTier::Beta => {}
            crate::terms::PenaltyTier::Rho => {}
        }
    }
    Ok(LatentIdObjectiveContribution { cost, gradient })
}

fn add_analytic_penalty_hessian_to_eval(
    theta: &Array1<f64>,
    rho_dim: usize,
    latent: &crate::terms::latent::LatentCoordValues,
    registry: &crate::terms::AnalyticPenaltyRegistry,
    eval: &mut (
        f64,
        Array1<f64>,
        crate::solver::rho_optimizer::HessianResult,
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
    let crate::solver::rho_optimizer::HessianResult::Analytic(hessian) = &mut eval.2 else {
        if eval.2.is_analytic() {
            eval.2 = crate::solver::rho_optimizer::HessianResult::Unavailable;
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
        if !matches!(tier, crate::terms::PenaltyTier::Psi) {
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
    latent: &crate::terms::latent::LatentCoordValues,
    registry: &crate::terms::AnalyticPenaltyRegistry,
    eval: &mut (
        f64,
        Array1<f64>,
        crate::solver::rho_optimizer::HessianResult,
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
    use crate::estimate::reml::ImplicitDerivLevel;
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
            crate::estimate::reml::HyperDesignDerivative::from_implicit(
                op.clone(),
                ImplicitDerivLevel::SecondDiag(implicit_axis),
                global_range.clone(),
                total_p,
            )
        } else {
            crate::estimate::reml::HyperDesignDerivative::from_embedded(
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
                            crate::estimate::reml::HyperDesignDerivative::from_implicit(
                                op.clone(),
                                ImplicitDerivLevel::SecondCross(implicit_axis, b_axis),
                                global_range.clone(),
                                total_p,
                            )
                        } else {
                            crate::estimate::reml::HyperDesignDerivative::from_embedded(
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
                crate::estimate::reml::HyperPenaltyDerivative::from_embedded(
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
                crate::estimate::reml::HyperPenaltyDerivative::from_embedded(
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
                        Option<Vec<crate::estimate::reml::PenaltyDerivativeComponent>>,
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
                                    crate::estimate::reml::HyperPenaltyDerivative::from_embedded(
                                        local,
                                        global_range_inner.clone(),
                                        total_p_inner,
                                    )
                                }))
                                .map(|(penalty_index, matrix)| {
                                    crate::estimate::reml::PenaltyDerivativeComponent {
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
                                Option<Vec<crate::estimate::reml::PenaltyDerivativeComponent>>,
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
            crate::estimate::reml::HyperDesignDerivative::from_implicit(
                op.clone(),
                ImplicitDerivLevel::First(implicit_axis),
                global_range.clone(),
                total_p,
            )
        } else {
            crate::estimate::reml::HyperDesignDerivative::from_embedded(
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

/// Whether a spatial term contributes per-axis ψ entries to the outer joint
/// hyperparameter vector.
///
/// A term enrolls per-axis log-κ ψ when (a) it carries `aniso_log_scales`
/// matching its feature dimension, (b) its dimension is `> 1`, and (c) its
/// basis supports REML-side anisotropic κ optimization. Duchon is explicitly
/// excluded: its anisotropy η is a fixed, geometry-derived basis parameter,
/// never a REML hyper axis. A hybrid Duchon (explicit κ) still optimizes its
/// scalar length scale through the regular isotropic-κ path, so it
/// contributes a single ψ entry, not one per axis.
///
/// This predicate is the *single source of truth* for the joint-θ layout:
/// `has_aniso_terms`, `spatial_dims_per_term`, `from_length_scales_aniso`,
/// and the hyper_dirs builder `try_build_spatial_log_kappa_derivativeinfo_list`
/// all consult it, so the outer optimizer's `n_params = rho_dim + Σ ψ_per_term`
/// always matches the gradient length produced by the inner unified evaluator.
fn spatial_term_uses_per_axis_psi(resolvedspec: &TermCollectionSpec, term_idx: usize) -> bool {
    // Measure-jet enrolls a multi-coordinate DIAL group (α, lnτ[, s]) —
    // grouped like per-axis anisotropy in the θ layout, but the coordinates
    // are geometry dials, not axis scales. Enrollment is owned by
    // `measure_jet_enrolls_psi`.
    if let Some(mj) = measure_jet_term_spec(resolvedspec, term_idx) {
        return measure_jet_enrolls_psi(mj);
    }
    let Some(d) = get_spatial_feature_dim(resolvedspec, term_idx) else {
        return false;
    };
    if d <= 1 {
        return false;
    }
    let Some(eta) = get_spatial_aniso_log_scales(resolvedspec, term_idx) else {
        return false;
    };
    if eta.len() != d {
        return false;
    }
    !matches!(
        resolvedspec.smooth_terms.get(term_idx).map(|t| &t.basis),
        Some(SmoothBasisSpec::Duchon { .. })
    )
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
            crate::solver::rho_optimizer::HessianResult,
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
                crate::solver::rho_optimizer::HessianResult,
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
        crate::solver::rho_optimizer::HessianResult,
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
        let zero_x = crate::estimate::reml::HyperDesignDerivative::zero(
            self.realizer.design().design.nrows(),
            p_total,
        );
        let components = s_psi_components
            .into_iter()
            .enumerate()
            .map(|(penalty_index, local)| {
                (
                    penalty_index,
                    crate::estimate::reml::HyperPenaltyDerivative::from_embedded(
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
        crate::solver::rho_optimizer::HessianResult,
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
            crate::solver::rho_optimizer::HessianResult,
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
    ) -> Result<(Vec<crate::construction::CanonicalPenalty>, Vec<usize>), String> {
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
    current_latent: Option<std::sync::Arc<crate::terms::latent::LatentCoordValues>>,
    current_hyper_dirs: Option<Vec<crate::estimate::reml::DirectionalHyperParam>>,
    current_design_cache_id: Option<u64>,
    latent_design_cache: crate::solver::latent_cache::LatentDesignCache,
    last_cost: Option<f64>,
    last_eval: Option<(
        f64,
        Array1<f64>,
        crate::solver::rho_optimizer::HessianResult,
    )>,
    term_index: crate::types::SmoothTermIdx,
    feature_cols: Vec<usize>,
    rho_dim: usize,
    n_obs: usize,
    latent_dim: usize,
    id_mode: crate::terms::latent::LatentIdMode,
    manifold: crate::terms::latent::LatentManifold,
    retraction_registry: crate::solver::latent_cache::LatentRetractionRegistry,
    latent_id: u64,
    analytic_penalties: Option<std::sync::Arc<crate::terms::AnalyticPenaltyRegistry>>,
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
            latent_design_cache: crate::solver::latent_cache::LatentDesignCache::default(),
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

    fn latent(&self) -> Result<std::sync::Arc<crate::terms::latent::LatentCoordValues>, String> {
        self.current_latent
            .as_ref()
            .cloned()
            .ok_or_else(|| "latent-coordinate cache has not been realized".to_string())
    }

    fn analytic_penalties(&self) -> Option<std::sync::Arc<crate::terms::AnalyticPenaltyRegistry>> {
        self.analytic_penalties.clone()
    }

    fn analytic_penalty_rho_count(&self) -> usize {
        self.analytic_rho_count
    }

    fn hyper_dirs(&self) -> Result<Vec<crate::estimate::reml::DirectionalHyperParam>, String> {
        self.current_hyper_dirs
            .as_ref()
            .cloned()
            .ok_or_else(|| "latent-coordinate hyper_dirs cache has not been realized".to_string())
    }

    fn latent_basis_kind(&self) -> Result<crate::solver::latent_cache::LatentBasisKind, String> {
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
            ) => Ok(crate::solver::latent_cache::LatentBasisKind::Matern {
                centers: centers.clone(),
                length_scale: *length_scale,
                nu: *nu,
                aniso_log_scales: aniso_log_scales
                    .clone()
                    .unwrap_or_else(|| vec![0.0; centers.ncols()]),
                chunk_size: crate::basis::auto_streaming_chunk_size_for_dense(
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
            ) => Ok(crate::solver::latent_cache::LatentBasisKind::Duchon {
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
            ) if matches!(*method, crate::basis::SphereMethod::Wahba) => {
                Ok(crate::solver::latent_cache::LatentBasisKind::Sphere {
                    centers: centers.clone(),
                    penalty_order: *penalty_order,
                    chunk_size: crate::basis::auto_streaming_chunk_size_for_dense(
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
                        crate::solver::latent_cache::LatentBasisKind::PeriodicBspline {
                            domain_start: *domain_start,
                            period: *period,
                            degree: effective_degree,
                            num_basis: *num_basis,
                            chunk_size: crate::basis::auto_streaming_chunk_size_for_dense(
                                self.n_obs, *num_basis,
                            ),
                        },
                    )
                } else {
                    let num_basis_est = knots.len().saturating_sub(effective_degree + 1);
                    Ok(
                        crate::solver::latent_cache::LatentBasisKind::TensorBspline {
                            knots: vec![knots.clone()],
                            degrees: vec![effective_degree],
                            chunk_size: crate::basis::auto_streaming_chunk_size_for_dense(
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
                crate::solver::latent_cache::LatentBasisKind::TensorBspline {
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
                    Some(crate::solver::latent_cache::pca_center_mean_fingerprint(
                        mean,
                    ))
                } else {
                    None
                };
                Ok(crate::solver::latent_cache::LatentBasisKind::Pca {
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
            crate::terms::latent::LatentCoordValues::from_flat_with_manifold_and_retraction_and_id(
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
            crate::solver::latent_cache::latent_design_context_cache_digest(
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
                Ok(crate::solver::latent_cache::ComputedLatentDesign {
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
                == Some(crate::solver::estimate::reml::outer_eval::current_outer_iter())
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
        crate::solver::rho_optimizer::HessianResult,
    )> {
        if self
            .current_theta
            .as_ref()
            .is_some_and(|cached| theta_values_match(cached, theta))
            && self.last_outer_iter
                == Some(crate::solver::estimate::reml::outer_eval::current_outer_iter())
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
            crate::solver::rho_optimizer::HessianResult,
        ),
    ) {
        self.last_cost = Some(eval.0);
        self.last_eval = Some(eval);
        self.last_outer_iter =
            Some(crate::solver::estimate::reml::outer_eval::current_outer_iter());
    }

    fn store_cost(&mut self, cost: f64) {
        self.last_cost = Some(cost);
        self.last_outer_iter =
            Some(crate::solver::estimate::reml::outer_eval::current_outer_iter());
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
    if try_build_spatial_log_kappa_hyper_dirs(data, resolvedspec, &best.design, spatial_terms)?
        .is_none()
    {
        return Ok(None);
    }

    const JOINT_RHO_BOUND: f64 = 12.0;
    let rho_dim = best.fit.lambdas.len();

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
    let log_kappa0 = log_kappa0.reseed_from_data(data, resolvedspec, spatial_terms, kappa_options);
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
    // Project seed onto data-derived bounds; spec.length_scale is a hint,
    // not a hard constraint. BFGS requires theta0 ∈ [lower, upper].
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);
    let setup = ExactJointHyperSetup::new(
        best.fit.lambdas.mapv(f64::ln),
        Array1::<f64>::from_elem(rho_dim, -JOINT_RHO_BOUND),
        Array1::<f64>::from_elem(rho_dim, JOINT_RHO_BOUND),
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
            log::info!(
                "[spatial-kappa] joint spatial optimization did not converge \
                 (iterations={}, final_objective={:.6e}, final_grad_norm={}); \
                 keeping the frozen baseline geometry",
                iterations,
                final_value,
                final_grad_norm.map_or_else(|| "n/a".to_string(), |g| format!("{g:.3e}")),
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
    };

    // Compare the joint optimizer's certified cost (final_value at theta*)
    // against the baseline. Tolerance ≥ options.tol because both endpoints
    // are outer-BFGS approximations accurate to options.tol; a tighter
    // gate would reject true improvements due to floating-point noise.
    let accept_tol = options.tol.max(1e-8 * baseline_score.abs()).max(1e-12);
    if joint_final_value > baseline_score + accept_tol {
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
    let resolvedspec = log_kappa_star.apply_tospec(resolvedspec, spatial_terms)?;
    let optimized = fit_term_collection_forspecwith_heuristic_lambdas(
        data,
        y,
        weights,
        offset,
        &resolvedspec,
        rho_star.as_slice(),
        family,
        options,
    )?;

    // Stamp reml_score with joint_final_value so downstream consumers see a
    // score consistent with the gate decision; the refit serves as a
    // β/inference harvester at the certified (ρ*, ψ*).
    let mut fit = optimized.fit;
    fit.reml_score = joint_final_value;
    let optimized_result = FittedTermCollectionWithSpec {
        fit,
        design: optimized.design,
        resolvedspec,
        adaptive_diagnostics: optimized.adaptive_diagnostics,
        kappa_timing: Some(kappa_timing),
    };

    Ok(Some(optimized_result))
}

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
/// The refit is a β/inference harvester at `best`'s lambdas and the frozen
/// `resolvedspec`; the score that geometry was certified at is
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
#[allow(clippy::too_many_arguments)]
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
        family,
        options,
    )?;
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
    evaluator: crate::estimate::ExternalJointHyperEvaluator<'d>,
    frozen_glm_inputs: Option<SpatialFrozenGlmInputs>,
    frozen_glm_psi_bounds: Option<(f64, f64)>,
    frozen_glm_tensor: Option<crate::solver::glm_sufficient_lane::FrozenWeightGramTensor>,
    frozen_glm_tensor_attempted: bool,
}

impl<'d> SpatialJointContext<'d> {
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
            if let (Some(tensor), Some(beta)) = (self.frozen_glm_tensor.as_ref(), warm_beta)
                && tensor.contains(psi)
                && let Some((current_w, _)) = self.frozen_glm_working_state(beta)?
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
        order: crate::solver::rho_optimizer::OuterEvalOrder,
        analytic_outer_hessian_available: bool,
    ) -> Result<
        (
            f64,
            Array1<f64>,
            crate::solver::rho_optimizer::HessianResult,
        ),
        EstimationError,
    > {
        use crate::solver::rho_optimizer::OuterEvalOrder;
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
                    // #1033 (reduced-basis rotation, supersedes #1264 gate): the
                    // Gaussian-identity inner solve the skip serves reads its data
                    // statistics ONLY from re-keyed k×k objects — XᵀWX(ψ)/XᵀW(y−
                    // offset)(ψ) from the `GaussianFixedCache` (re-installed at this
                    // trial's ψ by `install_psi_gram_statistics`) and the stable
                    // reparametrization Qs, which is built from the PENALTY
                    // (`stable_reparameterization_engine_canonical`), NOT the data
                    // Gram, and is rebuilt every inner PLS call from the re-keyed
                    // `S(ψ)`. β̂, EDF, scale/deviance, and the j==0 ψ-gradient
                    // (from the re-keyed `∂G/∂ψ`,`∂b/∂ψ` derivative pair, conjugated
                    // by Qs) never read `self.x` rows. So the skip is sound whether
                    // or not the conditioned reduced basis ROTATES with ψ: the
                    // realized `self.x` frozen at the pinning ψ is never consumed on
                    // this lane. The earlier wrong-β̂ regression a wide-window MSI
                    // run found was the STALE PENALTY `S(ψ_old)` (fixed by re-keying
                    // it n-free below), not a stale reduced basis. The pairwise
                    // range-projector witness `reduced_basis_equal` is retained as a
                    // soundness reference, but is no longer a skip precondition — it
                    // refused sound rotated-basis skips, forcing the O(n)
                    // `reset_surface` fallback and breaking n-independence across a
                    // basis rotation.
                    //
                    // #1033 penalty lane: ψ moves S(ψ) too, and the skip leaves
                    // `reset_surface` un-run; only skip when the penalty can be
                    // rebuilt EXACTLY and n-free on the fast path, else the inner
                    // solve would pair XᵀWX(ψ_new) with the stale S(ψ_old).
                    && self.evaluator.supports_nfree_penalty_rekey()
                    && nfree_fast_path_revision.is_some()
                    // #1264 SOUNDNESS PRECONDITION (restored): the n-free fast path
                    // reads `XᵀWX(ψ)` from the Chebyshev `gram_at(ψ)` interpolant
                    // (`PSI_GRAM_SPOT_RTOL=1e-10` relative), while the slow path forms
                    // the EXACT assembled Gram. On production Duchon geometry the raw
                    // Gram is near-singular (κ(G)≈1e15 MSI-measured), so even that
                    // 1e-10 interpolation round-off amplifies through the penalized
                    // solve to β̂rel≈1.7e-5 — 17× the issue's 1e-6 contract — i.e. the
                    // skip MOVES the κ-optimum. The `82cf6049f` "stale-penalty-not-
                    // stale-basis" theory was empirically refuted: the divergence
                    // appears even at a ψ the witness ADMITS, so it is conditioning
                    // amplification of the interpolated Gram, not a stale-basis leak.
                    // Restore the `reduced_basis_equal` window (`covers_skip`) as a
                    // skip precondition so the fast path fires ONLY where the reduced
                    // basis is provably unchanged (in practice the interpolation nodes
                    // / non-rotating sub-window), and falls to the exact slow path on
                    // the rotating production geometry. Correctness over speed (#1264).
                    && self.evaluator.psi_gram_tensor_covers_skip(psi)
        };
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
            warm_beta.as_ref().map(|b| b.view()),
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
    ) -> Result<crate::solver::rho_optimizer::EfsEval, EstimationError> {
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
            warm_beta.as_ref().map(|b| b.view()),
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
                    // #1033 (reduced-basis rotation): a value-only line-search probe
                    // runs the inner Gaussian-identity solve from the re-keyed
                    // `GaussianFixedCache` (XᵀWX(ψ)/XᵀW(y−offset)(ψ)) and the stable
                    // reparametrization Qs (penalty-derived, re-keyed every call from
                    // the n-free `S(ψ)`). β̂ and the cost never read the frozen
                    // `self.x` rows, so the probe is correct whether or not the
                    // conditioned reduced basis rotates with ψ. The
                    // `reduced_basis_equal` precondition is therefore dropped here
                    // too (it refused sound rotated-basis probes and forced the O(n)
                    // `reset_surface` fallback); the witness is kept as a soundness
                    // reference. See the eval_full gate for the full justification.
                    // #1033 penalty lane: the value-probe fast path also skips
                    // `reset_surface`, so the probe must be able to re-key S(ψ)
                    // EXACTLY and n-free; otherwise its cost would use the stale
                    // S(ψ_old) and mis-rank the line search.
                    && self.evaluator.supports_nfree_penalty_rekey()
                    && nfree_fast_path_revision.is_some()
                    // #1264 SOUNDNESS PRECONDITION (restored): same conditioning
                    // hazard as the eval_full gate — the cost-only probe's inner
                    // Gaussian solve reads the Chebyshev-interpolated `gram_at(ψ)`,
                    // and κ(G)≈1e15 on production Duchon geometry amplifies that
                    // 1e-10 round-off into a moved κ-optimum (β̂rel≈1.7e-5 ≫ 1e-6,
                    // MSI-measured at a witness-ADMITTED ψ). Gate on the
                    // `reduced_basis_equal` window so the value probe re-realizes the
                    // exact design across the rotation. See the eval_full gate.
                    && self.evaluator.psi_gram_tensor_covers_skip(psi)
        };
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
                warm_beta.as_ref().map(|b| b.view()),
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
    use crate::solver::rho_optimizer::{
        DeclaredHessianForm, Derivative, OuterEval, OuterEvalOrder,
    };

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
        evaluator: crate::estimate::ExternalJointHyperEvaluator::new(
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
    // FD-OK: FD-audit of the analytic outer gradient (small-problem gate, never feeds the optimizer)
    const OUTER_FD_AUDIT_MAX_N: usize = 4_000; // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
    const OUTER_FD_AUDIT_MAX_THETA_DIM: usize = 32; // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
    let n_total = data.nrows();
    let outer_fd_audit_eligible = analytic_outer_hessian_available // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
        && n_total <= OUTER_FD_AUDIT_MAX_N // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
        && theta_dim <= OUTER_FD_AUDIT_MAX_THETA_DIM; // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
    log::warn!(
        "[OUTER-FD-AUDIT/spatial-exact-joint] gate eligible={outer_fd_audit_eligible} \
         analytic_grad={analytic_outer_hessian_available} n_total={n_total} \
         theta_dim={theta_dim} rho_dim={rho_dim} psi_dim={coord_dim}"
    );
    if outer_fd_audit_eligible {
        // fd-ok: FD-audit gate, runs diagnostic oracle only, not in fit math
        let audit = (|| -> Result<crate::solver::rho_optimizer::OuterGradientFdAudit, String> {
            let mut eval_at = |theta: &Array1<f64>,
                               mode: crate::solver::estimate::reml::reml_outer_engine::EvalMode|
             -> Result<
                (
                    f64,
                    Array1<f64>,
                    crate::solver::rho_optimizer::HessianResult,
                ),
                String,
            > {
                use crate::solver::estimate::reml::reml_outer_engine::EvalMode;
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
            crate::solver::rho_optimizer::outer_gradient_fd_audit(
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
        "[KAPPA-PHASE-PRIME] order={:?} elapsed_s={:.4} slow_path_resets_total={} design_revision={}",
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
    let kphase_optim_start = std::time::Instant::now();
    let kphase_log_kappa_dim = coord_dim;
    let kphase_slow_resets_start = ctx.evaluator.slow_path_reset_count();
    let kphase_design_revision_start = ctx.cache.design_revision();

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
    );

    let eval_outer = |ctx: &mut &mut SpatialJointContext<'_>,
                      theta: &Array1<f64>,
                      order: OuterEvalOrder|
     -> Result<OuterEval, EstimationError> {
        let t0 = std::time::Instant::now();
        let raw = ctx.eval_full(theta, order, analytic_outer_hessian_available);
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
            let cost = ctx.eval_cost(theta);
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
    log::info!(
        "[KAPPA-PHASE-SUMMARY] log_kappa_dim={} n_cost={} cost_total_s={:.4} n_eval={} eval_total_s={:.4} n_efs={} efs_total_s={:.4} slow_path_resets={} design_revision_delta={} optim_total_s={:.4}",
        kphase_log_kappa_dim,
        kphase_cost_calls.get(),
        kphase_cost_total_s.get(),
        kphase_eval_calls.get(),
        kphase_eval_total_s.get(),
        kphase_efs_calls.get(),
        kphase_efs_total_s.get(),
        kphase_slow_resets,
        kphase_design_revision_delta,
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

fn set_spatial_length_scale(
    spec: &mut TermCollectionSpec,
    term_idx: usize,
    length_scale: f64,
) -> Result<(), EstimationError> {
    let Some(term) = spec.smooth_terms.get_mut(term_idx) else {
        crate::bail_invalid_estim!("spatial length-scale term index {term_idx} out of range");
    };
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

pub fn get_spatial_length_scale(spec: &TermCollectionSpec, term_idx: usize) -> Option<f64> {
    spec.smooth_terms
        .get(term_idx)
        .and_then(|term| match &term.basis {
            SmoothBasisSpec::ThinPlate { spec, .. } => Some(spec.length_scale),
            SmoothBasisSpec::Matern { spec, .. } => Some(spec.length_scale),
            SmoothBasisSpec::Duchon { spec, .. } => spec.length_scale,
            _ => None,
        })
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
pub fn freeze_measure_jet_length_scale_learning(spec: &mut TermCollectionSpec) -> usize {
    let mut frozen = 0;
    for term in spec.smooth_terms.iter_mut() {
        if let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &mut term.basis
            && mj.learn_length_scale
        {
            mj.learn_length_scale = false;
            frozen += 1;
        }
    }
    frozen
}

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

/// Indices of every constant-curvature (`curv(...)`) smooth term in `spec`.
pub fn constant_curvature_term_indices(spec: &TermCollectionSpec) -> Vec<usize> {
    (0..spec.smooth_terms.len())
        .filter(|&idx| constant_curvature_term_spec(spec, idx).is_some())
        .collect()
}

/// Freeze a `TermCollectionSpec` by baking in the concrete knots, centers,
/// identifiability transforms, and random-effect levels that were resolved
/// during design-matrix construction.  The result passes `validate_frozen`
/// and is safe to serialize for prediction.
///
/// This is the single canonical freezer — every model-save path should call
/// this rather than rolling ad-hoc freezing logic.
/// Freeze a smooth basis spec from its fit-time metadata so that predict-time
/// rebuilds reproduce the exact fitted geometry instead of recomputing any
/// data-dependent construction (knot selection, radial reparameterization,
/// eigen-truncation, identifiability constraint) on the prediction rows.
///
/// This is the SINGLE source of truth for freezing, shared by stand-alone
/// terms and by `by=`-wrapped / factor-sum-to-zero inner smooths. The wrapper
/// arms recurse into this same function, so every inner basis kind is frozen
/// with identical logic. A previous split implementation froze only B-spline
/// inners, leaving spatial inner bases (`bs='tp'`/`matern`/`duchon`/`sos`)
/// unfrozen and recomputed on the prediction grid (#704).
fn freeze_smooth_basis_from_metadata(
    basis: &mut SmoothBasisSpec,
    metadata: &BasisMetadata,
    term_name: &str,
) -> Result<(), EstimationError> {
    match (&mut *basis, metadata) {
        (SmoothBasisSpec::ByVariable { inner, .. }, meta)
        | (SmoothBasisSpec::FactorSumToZero { inner, .. }, meta) => {
            freeze_smooth_basis_from_metadata(inner, meta, term_name)?;
        }
        (
            SmoothBasisSpec::BSpline1D { spec: s, .. },
            BasisMetadata::BSpline1D {
                knots,
                identifiability_transform,
                periodic,
                degree: meta_degree,
                ..
            },
        ) => {
            // Issue #340: bake the fit-time effective degree into the
            // frozen spec so reload sees a self-consistent
            // (degree, knots) pair.
            if let Some(d) = meta_degree {
                s.degree = *d;
            }
            s.knotspec = periodic
                .map(
                    |(domain_start, period, num_basis)| BSplineKnotSpec::PeriodicUniform {
                        data_range: (domain_start, domain_start + period),
                        num_basis,
                    },
                )
                .unwrap_or_else(|| BSplineKnotSpec::Provided(knots.clone()));
            s.identifiability = match identifiability_transform {
                Some(z) => BSplineIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => BSplineIdentifiability::None,
            };
            // Boundary projections are folded into `identifiability_transform`
            // by `build_bspline_basis_1d`. A frozen prediction spec rebuilds the
            // same raw knot basis and replays the captured `FrozenTransform`
            // exactly once; the builder now SKIPS re-deriving the boundary
            // nullspace transform whenever identifiability is `FrozenTransform`
            // (it is already baked into that transform), so re-projection can no
            // longer happen. We therefore KEEP the original
            // `boundary_conditions`: they are the single source of truth the
            // intercept-suppression decision reads
            // (`term_collection_has_one_sided_anchored_bspline`), and a
            // one-sided anchored smooth suppresses the global intercept at fit
            // time (#1238). Clearing them here flipped that decision at predict
            // and re-added a spurious intercept column → save→load→predict
            // 21-vs-22 design mismatch (#1265). Boundary conditions are left
            // exactly as fit.
        }
        (
            SmoothBasisSpec::ThinPlate {
                spec: s,
                input_scales,
                ..
            },
            BasisMetadata::ThinPlate {
                centers,
                length_scale,
                periodic: meta_periodic,
                identifiability_transform,
                input_scales: meta_scales,
                radial_reparam,
            },
        ) => {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            s.length_scale = *length_scale;
            s.identifiability = match identifiability_transform {
                Some(z) => SpatialIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => match &s.identifiability {
                    SpatialIdentifiability::FrozenTransform { .. } => s.identifiability.clone(),
                    _ => SpatialIdentifiability::None,
                },
            };
            s.radial_reparam = radial_reparam.clone();
            s.periodic = meta_periodic.clone();
            *input_scales = meta_scales.clone();
        }
        (
            SmoothBasisSpec::ThinPlate { feature_cols, .. },
            BasisMetadata::Duchon {
                centers,
                length_scale,
                periodic: meta_periodic,
                power,
                nullspace_order,
                identifiability_transform,
                input_scales: meta_scales,
                aniso_log_scales: meta_aniso,
                ..
            },
        ) => {
            // Auto-promotion path: the basis builder rewrote a canonical-TPS
            // request to a pure Duchon spline because k < polynomial-nullspace
            // size at this dimension. Bake the resolved Duchon parameters into
            // the spec so predict-time goes through the same Duchon code path.
            let identifiability = match identifiability_transform {
                Some(z) => SpatialIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => SpatialIdentifiability::None,
            };
            *basis = SmoothBasisSpec::Duchon {
                feature_cols: feature_cols.clone(),
                spec: DuchonBasisSpec {
                    periodic: meta_periodic.clone(),
                    center_strategy: crate::basis::CenterStrategy::UserProvided(centers.clone()),
                    length_scale: *length_scale,
                    power: *power,
                    nullspace_order: *nullspace_order,
                    identifiability,
                    aniso_log_scales: meta_aniso.clone(),
                    operator_penalties: Default::default(),
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scales: meta_scales.clone(),
            };
        }
        (
            SmoothBasisSpec::Sphere { spec: s, .. },
            BasisMetadata::Sphere {
                centers,
                penalty_order,
                method,
                max_degree,
                wahba_kernel,
                constraint_transform,
            },
        ) => {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            s.penalty_order = *penalty_order;
            s.method = *method;
            s.max_degree = *max_degree;
            s.wahba_kernel = *wahba_kernel;
            // #532: freeze the realized-design transform captured at fit time
            // so the predict-time rebuild reuses it verbatim instead of
            // dropping the parametric orthogonalization and resurrecting the
            // intercept collision. The Harmonic method never carries a
            // constraint transform.
            s.identifiability = match constraint_transform {
                Some(z) => SphericalSplineIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => SphericalSplineIdentifiability::CenterSumToZero,
            };
        }
        (
            SmoothBasisSpec::ConstantCurvature { spec: s, .. },
            BasisMetadata::ConstantCurvature {
                centers,
                kappa,
                length_scale,
                constraint_transform,
            },
        ) => {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            s.kappa = *kappa;
            // Pin the REALIZED kernel range so a `0.0` auto-sentinel spec
            // replays the exact fit-time geometry at predict time (and at the
            // future ψ-channel per-trial rebuilds) instead of re-deriving ℓ
            // from whatever rows the rebuild sees.
            s.length_scale = *length_scale;
            // #532 pattern: freeze the composed `z · z_parametric` so the
            // rebuild replays the fit-time realized transform verbatim.
            s.identifiability = match constraint_transform {
                Some(z) => ConstantCurvatureIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => ConstantCurvatureIdentifiability::CenterSumToZero,
            };
        }
        (
            SmoothBasisSpec::MeasureJet {
                spec: s,
                input_scales,
                ..
            },
            BasisMetadata::MeasureJet {
                centers,
                input_scales: meta_scales,
                length_scale,
                eps_band,
                order_s,
                alpha,
                tau0,
                masses,
                support_means,
                penalty_normalization_scales,
                raw_penalty_normalization_scales,
                fused_penalty_normalization_scale,
                constraint_transform,
            },
        ) => {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            // Pin the realized geometry so auto sentinels cannot re-derive it
            // from predict rows. Field semantics are owned elsewhere:
            // length_scale replays VERBATIM (build dispatch round-trip
            // contract); order_s is the sentinel-preserving mode marker and
            // masses + eps_band are the fit-data penalty quadrature (both
            // `BasisMetadata::MeasureJet`).
            s.length_scale = *length_scale;
            s.order_s = *order_s;
            s.alpha = *alpha;
            s.tau0 = *tau0;
            s.num_scales = eps_band.len();
            s.frozen_quadrature = Some(MeasureJetFrozenQuadrature {
                masses: masses.clone(),
                eps_band: eps_band.clone(),
                support_means: support_means.clone(),
                penalty_normalization_scales: penalty_normalization_scales.clone(),
                raw_penalty_normalization_scales: raw_penalty_normalization_scales.clone(),
                fused_penalty_normalization_scale: *fused_penalty_normalization_scale,
            });
            // #532 pattern: freeze the composed `z · z_parametric` so the
            // rebuild replays the fit-time realized transform verbatim.
            s.identifiability = match constraint_transform {
                Some(z) => MeasureJetIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => MeasureJetIdentifiability::CenterSumToZero,
            };
            *input_scales = meta_scales.clone();
        }
        (
            SmoothBasisSpec::Matern {
                spec: s,
                input_scales,
                ..
            },
            BasisMetadata::Matern {
                centers,
                length_scale,
                periodic: meta_periodic,
                nu,
                include_intercept,
                identifiability_transform,
                input_scales: meta_scales,
                aniso_log_scales: meta_aniso,
                nullspace_shrinkage_survived: meta_nullspace_survived,
            },
        ) => {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            s.length_scale = *length_scale;
            s.nu = *nu;
            s.include_intercept = *include_intercept;
            // Pin the bootstrap-κ double-penalty nullspace-shrinkage decision into
            // the frozen transform so the κ-optimizer's per-trial design rebuilds
            // reproduce the SAME learned-penalty count (gam#787/#860); without
            // this the κ-dependent spectral test in `build_nullspace_shrinkage_penalty`
            // flips the count 6↔7 and the rebuilt ρ dimension disagrees with the
            // frozen joint setup ("joint hyper rho dimension mismatch"). When there
            // is no transform to freeze we keep `None` (unconstrained kernel needs
            // no replayed survival decision).
            s.identifiability = match identifiability_transform {
                Some(z) => MaternIdentifiability::FrozenTransform {
                    transform: z.clone(),
                    nullspace_shrinkage_survived: Some(*meta_nullspace_survived),
                },
                None => MaternIdentifiability::None,
            };
            s.aniso_log_scales = meta_aniso.clone();
            s.periodic = meta_periodic.clone();
            *input_scales = meta_scales.clone();
        }
        (
            SmoothBasisSpec::Duchon {
                spec: s,
                input_scales,
                ..
            },
            BasisMetadata::Duchon {
                centers,
                length_scale,
                periodic: meta_periodic,
                power,
                nullspace_order,
                identifiability_transform,
                input_scales: meta_scales,
                aniso_log_scales: meta_aniso,
                ..
            },
        ) => {
            s.center_strategy = crate::basis::CenterStrategy::UserProvided(centers.clone());
            s.length_scale = *length_scale;
            s.power = *power;
            s.nullspace_order = *nullspace_order;
            s.identifiability = match identifiability_transform {
                Some(z) => SpatialIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => match &s.identifiability {
                    // If the spec already carries a frozen transform but the
                    // metadata lost it (e.g. raw rebuild stripped it), keep
                    // the existing frozen transform rather than downgrading.
                    SpatialIdentifiability::FrozenTransform { .. } => s.identifiability.clone(),
                    _ => SpatialIdentifiability::None,
                },
            };
            s.aniso_log_scales = meta_aniso.clone();
            s.periodic = meta_periodic.clone();
            *input_scales = meta_scales.clone();
        }
        (
            SmoothBasisSpec::Sphere { spec: s, .. },
            BasisMetadata::SphereHarmonics {
                max_degree,
                radians,
            },
        ) => {
            s.max_degree = Some(*max_degree);
            s.radians = *radians;
        }
        (
            SmoothBasisSpec::TensorBSpline {
                feature_cols,
                spec: s,
            },
            BasisMetadata::TensorBSpline {
                feature_cols: fitted_cols,
                knots,
                degrees,
                periods,
                identifiability_transform,
            },
        ) => {
            if s.marginalspecs.len() != knots.len() || s.marginalspecs.len() != degrees.len() {
                crate::bail_invalid_estim!(
                    "tensor freeze mismatch for '{}': marginalspecs={}, knots={}, degrees={}",
                    term_name,
                    s.marginalspecs.len(),
                    knots.len(),
                    degrees.len()
                );
            }
            *feature_cols = fitted_cols.clone();
            for i in 0..s.marginalspecs.len() {
                s.marginalspecs[i].degree = degrees[i];
                s.marginalspecs[i].knotspec = match (periods[i], knots[i].len()) {
                    (Some(period), num_basis) if num_basis >= 1 => {
                        // Periodic uniform reconstructs the open
                        // `[start, start + period)` data range from the
                        // first knot and the saved period. `knots[i].len()`
                        // is the periodic control-site count.
                        let domain_start = knots[i][0];
                        BSplineKnotSpec::PeriodicUniform {
                            data_range: (domain_start, domain_start + period),
                            num_basis,
                        }
                    }
                    _ => BSplineKnotSpec::Provided(knots[i].clone()),
                };
            }
            // Do NOT overwrite `s.periods` from `periods`: `frozen = spec.clone()`
            // already preserves the user's original `spec.periods`. The metadata
            // `periods` slot captures the effective per-margin period for
            // restoring `PeriodicUniform` knotspecs, but it may also reflect
            // periodicity implied by a `PeriodicUniform` knotspec on a margin
            // for which the user left `spec.periods` empty (intending the
            // periodic B-spline path, not the Fourier path). Overwriting
            // `s.periods` here would silently flip such a margin onto the
            // Fourier path at predict time and produce a basis distinct from
            // the one used at fit time.
            s.identifiability = match identifiability_transform {
                Some(z) => TensorBSplineIdentifiability::FrozenTransform {
                    transform: z.clone(),
                },
                None => TensorBSplineIdentifiability::None,
            };
        }
        (
            SmoothBasisSpec::FactorSmooth { spec: s },
            BasisMetadata::FactorSmooth {
                knots,
                degree,
                periodic,
                group_levels,
                ..
            },
        ) => {
            s.marginal.knotspec = periodic
                .map(
                    |(domain_start, period, num_basis)| BSplineKnotSpec::PeriodicUniform {
                        data_range: (domain_start, domain_start + period),
                        num_basis,
                    },
                )
                .unwrap_or_else(|| BSplineKnotSpec::Provided(knots.clone()));
            // Restore the FROZEN marginal degree (#555 predict-replay). With
            // a `Provided(knots)` knotspec the per-margin basis count is
            // `knots.len() - (degree + 1)`, so if fit-time auto-shrink
            // lowered the marginal degree (small per-group n: cubic →
            // quadratic/linear), rebuilding with the original spec degree
            // would yield a different per-level `p` and corrupt the
            // block-diagonal replay. Mirror the sibling BySmooth arm, which
            // restores `inner.degree = *degree` for exactly this reason.
            s.marginal.degree = *degree;
            s.group_frozen_levels = Some(group_levels.clone());
        }
        (
            SmoothBasisSpec::BySmooth { smooth, by_kind },
            BasisMetadata::FactorSmooth {
                knots,
                degree,
                periodic,
                group_levels,
                ..
            },
        ) => {
            if let ByVarKind::Factor { frozen_levels, .. } = by_kind {
                *frozen_levels = Some(group_levels.clone());
            }
            if let SmoothBasisSpec::BSpline1D { spec: inner, .. } = smooth.as_mut() {
                // Issue #340: FactorSmooth metadata records the per-axis
                // spline degree directly (not optional); reflect it on
                // the inner spec so reload sees a self-consistent
                // (degree, knots) pair after fit-time auto-shrink.
                inner.degree = *degree;
                inner.knotspec = periodic
                    .map(
                        |(domain_start, period, num_basis)| BSplineKnotSpec::PeriodicUniform {
                            data_range: (domain_start, domain_start + period),
                            num_basis,
                        },
                    )
                    .unwrap_or_else(|| BSplineKnotSpec::Provided(knots.clone()));
                inner.identifiability = BSplineIdentifiability::None;
            }
        }
        (
            SmoothBasisSpec::BySmooth { smooth, by_kind },
            BasisMetadata::BySmooth { inner, levels, .. },
        ) => {
            // A `by=` smooth whose metadata is wrapped in `BySmooth`:
            // restore the frozen grouping levels, then recurse so the inner
            // basis is frozen with EXACTLY the same logic used for a
            // stand-alone term.
            if let ByVarKind::Factor { frozen_levels, .. } = by_kind
                && let Some(levels) = levels
            {
                *frozen_levels = Some(levels.clone());
            }
            freeze_smooth_basis_from_metadata(smooth, inner, term_name)?;
        }
        (SmoothBasisSpec::BySmooth { smooth, .. }, metadata) => {
            // `by=` wrapper carrying the inner basis metadata directly
            // (numeric `by`, or a factor `by` lowered to one gated block).
            // Recurse so a spatial inner basis (thin-plate, Matern, Duchon,
            // sphere, tensor, …) is frozen identically to a stand-alone
            // term. Previously only a B-spline inner was frozen here, so a
            // `s(x, by=g, bs='tp')` smooth left its data-dependent kernel
            // and eigen-truncation to be recomputed on the prediction grid,
            // crashing the predict-time design rebuild (#704).
            freeze_smooth_basis_from_metadata(smooth, metadata, term_name)?;
        }
        _ => {
            crate::bail_invalid_estim!(
                "smooth metadata/spec type mismatch while freezing term '{}'",
                term_name
            );
        }
    }
    Ok(())
}

pub fn freeze_term_collection_from_design(
    spec: &TermCollectionSpec,
    design: &TermCollectionDesign,
) -> Result<TermCollectionSpec, EstimationError> {
    if spec.smooth_terms.len() != design.smooth.terms.len() {
        crate::bail_invalid_estim!(
            "freeze mismatch: smooth spec count {} != design smooth term count {}",
            spec.smooth_terms.len(),
            design.smooth.terms.len()
        );
    }
    if spec.random_effect_terms.len() != design.random_effect_levels.len() {
        crate::bail_invalid_estim!(
            "freeze mismatch: random-effect spec count {} != design random-effect term count {}",
            spec.random_effect_terms.len(),
            design.random_effect_levels.len()
        );
    }

    let mut frozen = spec.clone();

    // ── smooth terms ────────────────────────────────────────────────────
    for (term, fitted) in frozen
        .smooth_terms
        .iter_mut()
        .zip(design.smooth.terms.iter())
    {
        // Persist joint-null absorption rotation captured at fit time so
        // save → load → predict re-applies `X_new_raw · Q` identically to
        // in-memory prediction. `None` when the smooth had no joint null
        // space, or when rotation was suppressed (shape-constrained smooths
        // whose cone geometry would not survive arbitrary orthogonal
        // rotation). Without this propagation, models reloaded from disk
        // produce wrong η at predict-time for any smooth with `Some(Q)`.
        term.joint_null_rotation = fitted.joint_null_rotation.clone();
        freeze_smooth_basis_from_metadata(&mut term.basis, &fitted.metadata, &term.name)?;
        // Persist the global-orthogonality chart the metadata could not absorb
        // (factor-smooth kinds residualized against an overlapping owner
        // smooth, #978). Without this, save → load → predict rebuilds the
        // unresidualized full-width design and the fitted coefficients no
        // longer match it.
        if let Some(z) = fitted.unabsorbed_global_orthogonality.as_ref() {
            match &mut term.basis {
                SmoothBasisSpec::FactorSumToZero {
                    frozen_global_orthogonality,
                    ..
                } => *frozen_global_orthogonality = Some(z.clone()),
                SmoothBasisSpec::FactorSmooth { spec } => {
                    spec.frozen_global_orthogonality = Some(z.clone());
                }
                _ => {
                    crate::bail_invalid_estim!(
                        "freeze: term '{}' carries an unabsorbed global-orthogonality transform but its basis kind has no frozen carrier for it",
                        term.name
                    );
                }
            }
        }
    }

    // ── random-effect terms ─────────────────────────────────────────────
    for (idx, rt) in frozen.random_effect_terms.iter_mut().enumerate() {
        let (_, kept_levels) = &design.random_effect_levels[idx];
        rt.frozen_levels = Some(kept_levels.clone());
    }

    Ok(frozen)
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
    let applied_rotation: Option<crate::terms::basis::JointNullRotation> = match (
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
            let rotated = crate::linalg::faer_ndarray::fast_ab(&dense, q);
            local.design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(rotated));
            local.penalties = local
                .penalties
                .into_iter()
                .map(|s_local| {
                    let qt_s = crate::linalg::faer_ndarray::fast_atb(q, &s_local);
                    crate::linalg::faer_ndarray::fast_ab(&qt_s, q)
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
    basisworkspace: crate::basis::BasisWorkspace,
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
            basisworkspace: crate::basis::BasisWorkspace::new(),
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
    /// Matérn stays on the exact slow re-key path here. Its operator-triplet
    /// n-free rebuild exists, but the current quality gate shows that enabling
    /// the fast-path κ loop changes the selected fit enough to miss the mgcv
    /// truth-recovery bar. Duchon/ThinPlate are the #1033 acceptance lane.
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
    ) -> Result<(Vec<crate::construction::CanonicalPenalty>, Vec<usize>), String> {
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
                ..
            } => {
                let operator_penalties = match &termspec.basis {
                    SmoothBasisSpec::Duchon { spec, .. } => spec.operator_penalties.clone(),
                    _ => crate::basis::DuchonOperatorPenaltySpec::default(),
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
                crate::basis::duchon_penalties_at_length_scale(
                    centers.view(),
                    identifiability_transform.as_ref(),
                    operator_collocation_points.as_ref().map(|p| p.view()),
                    &operator_penalties,
                    *power,
                    *nullspace_order,
                    aniso_log_scales.as_deref(),
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
                // metadata`). The Matérn design ALWAYS overrides the kernel
                // double-penalty with this {mass, tension, stiffness} triplet
                // (see `build_inner_smooth_basis`), so re-keying via the kernel
                // path produced a 1-block surface against a 3-block frozen
                // design — the topology desync #1270 hard-errored on. Sharing
                // the builder makes the block count ψ-stable by construction.
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
                crate::basis::thin_plate_penalties_at_length_scale(
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
        let specs: Vec<crate::estimate::PenaltySpec> = templates
            .iter()
            .zip(locals.into_iter())
            .map(|(tmpl, local)| crate::estimate::PenaltySpec::Block {
                local,
                col_range: tmpl.col_range.clone(),
                prior_mean: tmpl.prior_mean.clone(),
                structure_hint: tmpl.structure_hint.clone(),
                op: tmpl.op.clone(),
            })
            .collect();
        crate::construction::canonicalize_penalty_specs(
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
                    crate::basis::build_duchon_native_penalty_psi_derivatives(
                        centers.view(),
                        &spec,
                        identifiability_transform.as_ref(),
                        &mut self.basisworkspace,
                    )
                    .map_err(|e| e.to_string())?;
                let (_operator_sources, operator_first, _operator_second) =
                    crate::basis::build_duchon_operator_penalty_psi_derivatives(
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
                    crate::basis::expand_periodic_centers(&centers.to_owned(), periodic.as_deref())
                        .map_err(|e| e.to_string())?;
                let aniso_for_penalty = aniso_from_psi.as_deref().or(aniso_log_scales.as_deref());
                let (first, _second) = crate::basis::build_matern_operator_penalty_psi_derivatives(
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
                    crate::basis::build_thin_plate_penalty_psi_derivativeswithworkspace(
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
        blocks.push(DesignBlock::Dense(crate::matrix::DenseDesignMatrix::from(
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
        crate::solver::rho_optimizer::HessianResult,
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
) -> crate::seeding::SeedRiskProfile {
    match &family.response {
        ResponseFamily::Gaussian => crate::seeding::SeedRiskProfile::Gaussian,
        ResponseFamily::RoystonParmar => crate::seeding::SeedRiskProfile::Survival,
        ResponseFamily::Binomial
        | ResponseFamily::Poisson
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Beta { .. }
        | ResponseFamily::Gamma => crate::seeding::SeedRiskProfile::GeneralizedLinear,
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
    risk_profile: crate::seeding::SeedRiskProfile,
    auxiliary_dim: usize,
) -> crate::seeding::SeedConfig {
    let mut config = crate::seeding::SeedConfig {
        risk_profile,
        num_auxiliary_trailing: auxiliary_dim,
        ..Default::default()
    };
    match risk_profile {
        crate::seeding::SeedRiskProfile::Gaussian => {
            config.max_seeds = 4;
            config.seed_budget = 2;
        }
        crate::seeding::SeedRiskProfile::GeneralizedLinear => {
            // Bernoulli marginal-slope Matérn fits use the exact-joint spatial
            // driver rather than the family-local BMS outer. Mirror BMS proper:
            // screen one principled heuristic seed deeply enough to reach the
            // KKT basin instead of spending minutes screening equivalent starts.
            config.max_seeds = 1;
            config.seed_budget = 1;
            config.screen_max_inner_iterations = 8;
        }
        crate::seeding::SeedRiskProfile::Survival => {
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
        let bms = exact_joint_seed_config(crate::seeding::SeedRiskProfile::GeneralizedLinear, 2);
        assert_eq!(bms.max_seeds, 1);
        assert_eq!(bms.seed_budget, 1);
        assert_eq!(bms.screen_max_inner_iterations, 8);
        assert_eq!(bms.num_auxiliary_trailing, 2);

        let survival = exact_joint_seed_config(crate::seeding::SeedRiskProfile::Survival, 3);
        assert_eq!(survival.max_seeds, 8);
        assert_eq!(survival.seed_budget, 4);
        assert_eq!(survival.screen_max_inner_iterations, 8);
        assert_eq!(survival.num_auxiliary_trailing, 3);
    }

    #[test]
    fn exact_joint_gaussian_keeps_tight_historical_multistart_budget() {
        let gaussian = exact_joint_seed_config(crate::seeding::SeedRiskProfile::Gaussian, 1);
        assert_eq!(gaussian.max_seeds, 4);
        assert_eq!(gaussian.seed_budget, 2);
        assert_eq!(
            gaussian.screen_max_inner_iterations,
            crate::seeding::SeedConfig::default().screen_max_inner_iterations
        );
        assert_eq!(gaussian.num_auxiliary_trailing, 1);
    }
}

pub(crate) fn exact_joint_multistart_outer_problem(
    theta0: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    rho_dim: usize,
    auxiliary_dim: usize,
    n_params: usize,
    gradient: crate::solver::rho_optimizer::Derivative,
    hessian: crate::solver::rho_optimizer::DeclaredHessianForm,
    prefer_gradient_only: bool,
    disable_fixed_point: bool,
    risk_profile: crate::seeding::SeedRiskProfile,
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
) -> crate::solver::rho_optimizer::OuterProblem {
    let mut seed_heuristic = theta0.to_vec();
    for value in &mut seed_heuristic[..rho_dim] {
        *value = value.exp();
    }
    let mut problem = crate::solver::rho_optimizer::OuterProblem::new(n_params)
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
        .with_fallback_policy(crate::solver::rho_optimizer::FallbackPolicy::Automatic)
        .with_psi_dim(auxiliary_dim)
        .with_tolerance(tolerance)
        .with_max_iter(max_iter)
        .with_bounds(lower.clone(), upper.clone())
        .with_initial_rho(theta0.clone())
        .with_bfgs_step_cap(bfgs_step_cap)
        .with_bfgs_step_cap_psi(bfgs_step_cap_psi)
        .with_seed_config(exact_joint_seed_config(risk_profile, auxiliary_dim))
        .with_rho_bound(12.0)
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
    seed_risk_profile: crate::seeding::SeedRiskProfile,
    analytic_joint_gradient_available: bool,
    analytic_joint_hessian_available: bool,
    disable_fixed_point: bool,
    screening_cap: Option<Arc<AtomicUsize>>,
    outer_derivative_policy: crate::families::custom_family::OuterDerivativePolicy,
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
        crate::solver::estimate::reml::reml_outer_engine::EvalMode,
        &crate::outer_subsample::RowSet,
    ) -> Result<
        (
            f64,
            Array1<f64>,
            crate::solver::rho_optimizer::HessianResult,
        ),
        String,
    >,
    ExactEfsFn: FnMut(
        &Array1<f64>,
        &[TermCollectionSpec],
        &[TermCollectionDesign],
    ) -> Result<crate::solver::rho_optimizer::EfsEval, String>,
    SeedFn:
        FnMut(&Array1<f64>) -> Result<crate::solver::rho_optimizer::SeedOutcome, EstimationError>,
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
            crate::solver::rho_optimizer::DeclaredHessianForm::Either
                | crate::solver::rho_optimizer::DeclaredHessianForm::Dense
                | crate::solver::rho_optimizer::DeclaredHessianForm::Operator { .. }
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
    ) -> crate::solver::outer_subsample::OuterScoreSubsample {
        use crate::solver::outer_subsample::OuterScoreSubsample;
        let k = k_target.min(n_total);
        if k == 0 || n_total == 0 {
            return OuterScoreSubsample::new(Vec::new(), n_total, seed);
        }
        // Reservoir-free deterministic pick: linear congruential walk
        // over a shuffled index set; for the pilot, a fast Floyd-style
        // sample is sufficient.
        let mut mask: Vec<usize> = Vec::with_capacity(k);
        // Splitmix64-driven Floyd's sampler.
        let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
        let splitmix = |s: &mut u64| -> u64 { crate::linalg::utils::splitmix64(s) };
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
        OuterScoreSubsample::new(mask, n_total, seed)
    }

    let current_row_set: std::cell::RefCell<crate::outer_subsample::RowSet> = if use_staged_kappa {
        let pilot = build_uniform_pilot_subsample(n_total, KAPPA_PILOT_K, n_total as u64);
        std::cell::RefCell::new(crate::outer_subsample::RowSet::Subsample {
            rows: std::sync::Arc::clone(&pilot.rows),
            n_full: n_total,
        })
    } else {
        std::cell::RefCell::new(crate::outer_subsample::RowSet::All)
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

    use crate::solver::rho_optimizer::{
        DeclaredHessianForm, Derivative, OuterEval, OuterEvalOrder,
    };

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
                    if grad.iter().any(|v| !v.is_finite()) {
                        return Err(EstimationError::RemlOptimizationFailed(
                            "n-block exact-joint gradient contained non-finite values".to_string(),
                        ));
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
                crate::solver::estimate::reml::reml_outer_engine::EvalMode::ValueGradientHessian
            } else {
                crate::solver::estimate::reml::reml_outer_engine::EvalMode::ValueAndGradient
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
                    if grad.iter().any(|v| !v.is_finite()) {
                        return Err(EstimationError::RemlOptimizationFailed(
                            "n-block exact-joint gradient contained non-finite values".to_string(),
                        ));
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
                        crate::solver::estimate::reml::reml_outer_engine::EvalMode::ValueOnly,
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
        *current_row_set.borrow_mut() = crate::outer_subsample::RowSet::Subsample {
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
                crate::solver::estimate::reml::reml_outer_engine::EvalMode::ValueAndGradient,
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
    *current_row_set.borrow_mut() = crate::outer_subsample::RowSet::All;
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
    use crate::solver::rho_optimizer::{
        DeclaredHessianForm, Derivative, OuterEval, OuterEvalOrder,
    };

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
        evaluator: crate::estimate::ExternalJointHyperEvaluator<'d>,
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
                crate::solver::rho_optimizer::HessianResult,
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
                    crate::solver::estimate::reml::outer_eval::current_outer_iter() as usize,
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
        ) -> Result<crate::solver::rho_optimizer::EfsEval, EstimationError> {
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
                    crate::solver::estimate::reml::outer_eval::current_outer_iter() as usize,
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
                            crate::solver::estimate::reml::outer_eval::current_outer_iter()
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
        evaluator: crate::estimate::ExternalJointHyperEvaluator::new(
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

    let baseline_options = superseded_fit_options(options);
    let best = fit_term_collection_forspec(
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
    let spatial_terms = spatial_length_scale_term_indices(&resolvedspec);
    // Sync knot-cloud-derived aniso contrasts from the basis metadata back
    // into the spec so the optimizer starts from the geometry-informed η values
    // rather than the zero sentinel from --scale-dimensions.
    sync_aniso_contrasts_from_metadata(&mut resolvedspec, &best.design.smooth);
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
    let exact_joint = require_successful_spatial_optimization_result(
        initial_score,
        try_exact_joint_spatial_length_scale_optimization(
            data,
            y.view(),
            weights.view(),
            offset.view(),
            &resolvedspec,
            &best,
            family,
            options,
            kappa_options,
            &spatial_terms,
        )
        .map(|opt| {
            opt.map(|fit| {
                let score = fit_score(&fit.fit);
                (fit, score)
            })
        }),
    )?;
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
    pub ci: crate::geometry::curvature_estimand::KappaProfileCi,
    /// Interior-point κ = 0 likelihood-ratio flatness test (full χ²₁, no
    /// half-χ² boundary correction — κ = 0 is an interior point of the
    /// `S^d ← ℝ^d → H^d` family).
    pub flatness: crate::geometry::curvature_estimand::FlatnessTest,
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
/// [`crate::geometry::curvature_estimand`].
///
/// `v_pp` (the initial Wald step size) is taken from a central finite difference
/// of `V_p` at κ̂; the CI itself is the exact χ²₁ likelihood crossing, not the
/// Wald ellipsoid, so this only sizes the first bracket step.
#[allow(clippy::too_many_arguments)]
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

    // Profiled criterion oracle V_p(κ): pin κ, fit with κ-optimisation OFF so
    // only ρ is profiled, return the REML/LAML negative-log-evidence. Disabling
    // κ-opt routes `fit_term_collectionwith_spatial_length_scale_optimization`
    // straight to `fit_term_collection_forspec` at the spec's κ.
    let fixed_kappa_options = SpatialLengthScaleOptimizationOptions {
        enabled: false,
        ..SpatialLengthScaleOptimizationOptions::default()
    };
    // Memoize V_p across κ probes. The CI walk's bracketing/bisection, the
    // central-difference v_pp seed, and the flatness LR test all re-evaluate
    // V_p at the SAME κ (κ̂ alone is fit ≥3×, and the bisection revisits its
    // bracket endpoints), each a full fixed-κ inner fit. Identical κ ⇒
    // identical fit ⇒ identical score, so caching by κ removes the redundant
    // fits with no change to the statistical answer — the dominant cost in the
    // otherwise-timing-out flat/hyperbolic e2e arms. The key is the raw bits of
    // κ so only EXACT repeats hit the cache (no tolerance fuzz that could
    // collapse distinct probes).
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
        let mut probe_spec = resolvedspec.clone();
        match probe_spec
            .smooth_terms
            .get_mut(term_idx)
            .map(|t| &mut t.basis)
        {
            Some(SmoothBasisSpec::ConstantCurvature { spec, .. }) => spec.kappa = kappa,
            _ => {
                return Err(format!(
                    "V_p oracle: term {term_idx} is not a constant-curvature smooth"
                ));
            }
        }
        let fit = fit_term_collectionwith_spatial_length_scale_optimization(
            data,
            y.to_owned(),
            weights.to_owned(),
            offset.to_owned(),
            &probe_spec,
            family.clone(),
            options,
            &fixed_kappa_options,
        )
        .map_err(|e| format!("V_p fixed-κ fit at κ={kappa} failed: {e}"))?;
        let score = fit_score(&fit.fit);
        if score.is_finite() {
            v_p_cache.borrow_mut().insert(key, score);
            Ok(score)
        } else {
            Err(format!(
                "V_p fixed-κ fit at κ={kappa} returned a non-finite score"
            ))
        }
    };

    // Wald step seed: central FD of V_p at κ̂ (only sizes the first bracket; the
    // CI is the exact likelihood crossing). Step a small fraction of the κ
    // window so the FD straddles κ̂ without leaving the chart.
    let h = (1e-3 * (kappa_max - kappa_min)).max(1e-4);
    let v_pp = match (v_p(kappa_hat + h), v_p(kappa_hat), v_p(kappa_hat - h)) {
        (Ok(vp), Ok(v0), Ok(vm)) => (vp - 2.0 * v0 + vm) / (h * h),
        _ => f64::NAN, // profile_ci_walk falls back to a default step
    };

    let ci = crate::geometry::curvature_estimand::profile_ci_walk(
        &v_p, kappa_hat, v_pp, kappa_min, kappa_max, level, 1e-4,
    )
    .map_err(EstimationError::InvalidInput)?;
    let flatness = crate::geometry::curvature_estimand::flatness_lr_test(&v_p, kappa_hat)
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
) -> Result<Vec<crate::inference::lawley::RhoPenaltyComponent>, EstimationError> {
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
        components.push(crate::inference::lawley::RhoPenaltyComponent { s_component });
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
///    term's influence block (falling back to the term EDF) — the same `ref_df`
///    the summary Wald row reports.
/// 4. When the family has closed-form cumulant jets, evaluate Lawley's ε at the
///    **null** linear predictor (an expectation evaluated at the null fit), fold
///    the full λ-scaled penalty `S_λ` into the information, and Bartlett-correct
///    `W` with [`crate::inference::lawley::lawley_lr_bartlett_factor`]. The
///    null annihilates the tested block's penalty (`S_λ β₀ = 0` on that block),
///    so the penalized Lawley expansion applies verbatim.
/// 5. Otherwise (no closed-form jets, or a null refit that did not converge) the
///    uncorrected `χ²_d` stands with provenance `none` — never weakened.
///
/// Random-effect smooths and shape-constrained smooths are skipped (their tests
/// are not a central-χ² LR), matching the summary table's policy.
#[allow(clippy::too_many_arguments)]
pub fn smooth_term_lr_inference_forspec(
    data: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    family: LikelihoodSpec,
    options: &FitOptions,
) -> Result<Vec<SmoothTermLrInference>, EstimationError> {
    use crate::inference::lawley::{
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
    let edf_blocks = full.fit.edf_by_block().to_vec();
    let family_disp = lawley_dispersion_for_family(&family, &full.fit);

    // The penalty-block cursor walks the same block order the summary table
    // uses: random-effect ranges first (skipped here), then smooth terms.
    let mut penalty_cursor = full.design.random_effect_ranges.len();
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
        let edf = edf_blocks
            .get(block_start..block_start + k)
            .map(|block: &[f64]| block.iter().sum::<f64>())
            .unwrap_or(0.0);
        let ref_df = wood_reference_df(influence, &coeff_range).unwrap_or(edf.max(1e-12));
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
                            crate::inference::higher_order::bartlett_factor_from_mean(
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
        crate::types::ResponseFamily::Gaussian => {
            let sd = fit.standard_deviation;
            (sd * sd).max(f64::MIN_POSITIVE)
        }
        crate::types::ResponseFamily::Gamma => {
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

/// Wood's rank-corrected reference d.f. `tr(F_jj)² / tr(F_jj²)` on the
/// coefficient-influence block `F = H⁻¹ X'WX` restricted to `coeff_range`. This
/// is the same reference the summary Wald row uses, so the corrected LR and the
/// Wald test reference the *same* `χ²_d`. Returns `None` when the influence
/// block is unavailable or degenerate.
fn wood_reference_df(influence: Option<&Array2<f64>>, coeff_range: &Range<usize>) -> Option<f64> {
    let f = influence?;
    let (start, end) = (coeff_range.start, coeff_range.end);
    if start >= end || end > f.nrows() || end > f.ncols() {
        return None;
    }
    let block = f.slice(s![start..end, start..end]);
    let tr = (0..block.nrows()).map(|i| block[[i, i]]).sum::<f64>();
    let tr2 = block.dot(&block).diag().sum();
    (tr.is_finite() && tr2.is_finite() && tr > 0.0 && tr2 > 0.0).then(|| (tr * tr / tr2).max(1e-12))
}
