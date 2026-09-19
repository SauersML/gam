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
            input_scale,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            let mut spec_local = spec.clone();
            if let Some(scale) = input_scale {
                scale.standardize(&mut x);
                spec_local.length_scale = scale
                    .to_standardized_units(gam_terms::OriginalUnits::new(spec.length_scale))
                    .standardized_value();
            }
            // Same fitted-chart replay as the Matérn arm below: the realizer's
            // spec is in `z_local`; the metadata carries the composed chart.
            if let BasisMetadata::ThinPlate {
                identifiability_transform: Some(transform),
                ..
            } = &smooth_term.metadata
            {
                spec_local.identifiability = SpatialIdentifiability::FrozenTransform {
                    transform: transform.clone(),
                };
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
            input_scale,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            let mut spec_local = spec.clone();
            if let Some(scale) = input_scale {
                scale.standardize(&mut x);
                let length_scale = spec.length_scale.resolved().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "Matérn Auto length_scale reached derivative construction unresolved"
                            .to_string(),
                    )
                })?;
                spec_local.length_scale.set_resolved(
                    scale
                        .to_standardized_units(gam_terms::OriginalUnits::new(length_scale))
                        .standardized_value(),
                );
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
            // The jet is built in the FITTED coefficient chart. The incremental
            // realizer hands this builder a spec whose identifiability has been
            // put back into the TERM-LOCAL chart `z_local`
            // (`restore_local_identifiability_chart`, gam#2760) so that a design
            // REBUILD can apply the collection gauge's fixed `T0` itself; the
            // design the criterion is built on lives in the composition
            // `z_local · T0` the realized term's metadata records. Built on
            // `z_local`, the jet has the fitted WIDTH whenever `T0` is square
            // (the Residualize arm), so nothing declines and the ψ-gradient is
            // silently wrong: measured on the #1379 seed-3 fixture, the same θ
            // and the same cost (−138.0325036) gave ∂V/∂ψ = −2.949e4 here
            // against +0.2253 from the composed chart (central differences
            // +0.2249), and the line search walked uphill for 50 attempts. The
            // Duchon arm below already replays its chart from metadata; the
            // measure-jet per-axis arm was fixed the same way (597003f2e).
            if let BasisMetadata::Matern {
                identifiability_transform: Some(transform),
                ..
            } = &smooth_term.metadata
            {
                spec_local.identifiability = MaternIdentifiability::FrozenTransform {
                    transform: transform.clone(),
                };
            }
            spec_local.double_penalty = false;
            build_matern_basis_log_kappa_derivatives(x.view(), &spec_local)
                .map_err(EstimationError::from)?
        }
        SmoothBasisSpec::Duchon {
            feature_cols,
            spec,
            input_scale,
        } => {
            let mut x = select_columns(data, feature_cols).map_err(EstimationError::from)?;
            let mut spec_local = spec.clone();
            if let Some(scale) = input_scale {
                scale.standardize(&mut x);
                spec_local.length_scale = spec.length_scale.map(|length| {
                    scale
                        .to_standardized_units(gam_terms::OriginalUnits::new(length))
                        .standardized_value()
                });
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

    // gam#2760: the collection owns a FIXED row-space constraint `C`, while
    // its per-realization coefficient whitening is only a coordinate chart.
    // Differentiate the statistical smooth in the current coefficient chart:
    //
    //     X_g(psi) = P_C X(psi) T_0,  P_C = I - C(C'C)^-C'
    //
    // rather than differentiating the arbitrary RRQR/eigenvectors that produce
    // a fresh `T(psi)`.  Penalty jets already use the fixed `T_0` through the
    // metadata transform; only the design jets need this left projection.
    if let Some(gauge) = smooth_term.collection_gauge.as_ref() {
        let projector = FixedRowSpaceProjector::from_constraint_block(
            gauge.constraint_block.view(),
        )
        .map_err(EstimationError::from)?;
        // The scalar builders return dense jets beside the implicit operator, so
        // the dense jets take the same projection whether or not an operator is
        // present. Leaving them unprojected beside a projected operator carried a
        // second, wrong design derivative: on the gam#2895 Matérn fixture it
        // differed from central differences of the realized design by rel 6.67,
        // against 5.6e-7 once projected (MSI job 434079).
        if local_x_psi.nrows() > 0 {
            projector
                .project_matrix_in_place(&mut local_x_psi)
                .map_err(EstimationError::from)?;
        }
        if local_x_psi_psi.nrows() > 0 {
            projector
                .project_matrix_in_place(&mut local_x_psi_psi)
                .map_err(EstimationError::from)?;
        }
        if let Some(op) = implicit_operator.take() {
            implicit_operator = Some(
                op.with_fixed_row_space_projection(projector)
                    .map_err(EstimationError::from)?,
            );
        }
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
                input_scale,
                ..
            },
        ) => gam_terms::basis::LatentCoordDesignDerivative::new_matern(
            latent.clone(),
            std::sync::Arc::new(centers.clone()),
            // The metadata's own frame pair: standardized `centers` above,
            // original-units range here. The constructor owns the single
            // conversion between them (#2643).
            *input_scale,
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
                input_scale,
                radial_reparam,
                ..
            },
        ) => gam_terms::basis::LatentCoordDesignDerivative::new_duchon(
            latent.clone(),
            std::sync::Arc::new(centers.clone()),
            // See the Matérn arm: the pair travels together (#2643).
            *input_scale,
            *length_scale,
            *power,
            *nullspace_order,
            // The frozen data-metric radial chart the shipped kernel block is
            // expressed in (gam#979).
            radial_reparam.as_ref(),
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
        let zero_x = gam_solve::estimate::reml::HyperDesignDerivative::from(Array2::<f64>::zeros(
            (design.design.nrows(), p_total),
        ));
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
                AuxPriorStrength::Fixed(mu) => (
                    gam_problem::checked_log_strength(*mu).map_err(|error| {
                        EstimationError::InvalidInput(format!(
                            "fixed latent auxiliary-prior precision is outside the canonical physical-strength domain: {error}"
                        ))
                    })?,
                    *mu,
                ),
                AuxPriorStrength::Auto => {
                    let log_mu = *theta.get(cursor).ok_or_else(|| {
                        EstimationError::InvalidInput(format!(
                            "latent auxiliary-prior precision coordinate {cursor} is missing from theta length {}",
                            theta.len(),
                        ))
                    })?;
                    cursor += 1;
                    let mu = gam_problem::checked_exp_log_strength(log_mu).map_err(|error| {
                        EstimationError::InvalidInput(format!(
                            "latent auxiliary-prior log precision is outside the canonical log-strength domain: {error}"
                        ))
                    })?;
                    (log_mu, mu)
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
        LatentIdMode::IsometryToReference {
            reference,
            strength,
        } => {
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
                AuxPriorStrength::Fixed(mu) => (
                    gam_problem::checked_log_strength(*mu).map_err(|error| {
                        EstimationError::InvalidInput(format!(
                            "fixed latent isometry precision is outside the canonical physical-strength domain: {error}"
                        ))
                    })?,
                    *mu,
                ),
                AuxPriorStrength::Auto => {
                    let log_mu = *theta.get(cursor).ok_or_else(|| {
                        EstimationError::InvalidInput(format!(
                            "latent isometry precision coordinate {cursor} is missing from theta length {}",
                            theta.len(),
                        ))
                    })?;
                    cursor += 1;
                    let mu = gam_problem::checked_exp_log_strength(log_mu).map_err(|error| {
                        EstimationError::InvalidInput(format!(
                            "latent isometry log precision is outside the canonical log-strength domain: {error}"
                        ))
                    })?;
                    (log_mu, mu)
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
            if cursor + n_coeffs > theta.len() {
                crate::bail_invalid_estim!(
                    "latent auxiliary-outcome coefficient block overruns theta: start={cursor}, width={n_coeffs}, theta_len={}",
                    theta.len(),
                );
            }
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
            if cursor + latent_dim > theta.len() {
                crate::bail_invalid_estim!(
                    "latent dimension-selection precision block overruns theta: start={cursor}, width={latent_dim}, theta_len={}",
                    theta.len(),
                );
            }
            let alphas = gam_problem::checked_exp_log_strengths(
                theta.slice(s![cursor..cursor + latent_dim]).iter().copied(),
            )
            .map_err(|error| {
                EstimationError::InvalidInput(format!(
                    "latent dimension-selection log precision is outside the canonical log-strength domain: {error}"
                ))
            })?;
            for axis in 0..latent_dim {
                let log_alpha = theta[cursor + axis];
                let alpha = alphas[axis];
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
    eval: &mut (f64, Array1<f64>, gam_problem::HessianValue),
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
        eval.2 = gam_problem::HessianValue::Unavailable;
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
    registry
        .validate_rho(rho)
        .map_err(EstimationError::InvalidInput)?;
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
    eval: &mut (f64, Array1<f64>, gam_problem::HessianValue),
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
    let gam_problem::HessianValue::Dense(hessian) = &mut eval.2 else {
        if eval.2.is_analytic() {
            eval.2 = gam_problem::HessianValue::Unavailable;
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
    registry
        .validate_rho(rho)
        .map_err(EstimationError::InvalidInput)?;
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
    eval: &mut (f64, Array1<f64>, gam_problem::HessianValue),
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
