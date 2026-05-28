use super::*;
use super::family::*;
use super::gradient_paths::*;
use super::install_flex::validate_spec;

// ── BmsFamilyScalars ──────────────────────────────────────────────────────────
//
// Per-row scalars required by the BMS block Jacobians at non-zero β.
// Mirrors the SurvivalMarginalSlopeFamilyScalars pattern (T26 hard contract).
//
// Fields:
//   q_i  = M[i,:] · β_m + offset_m[i]    (marginal linear predictor)
//   g_i  = G[i,:] · β_s + offset_s[i]    (log-slope linear predictor)
//   c_i  = sqrt(1 + (s · g_i)²)
//   s    = probit_frailty_scale
//   z_i  = calibrated latent covariate for row i
//
// The marginal block needs c_i.  The logslope block needs q_i, g_i, c_i, z_i.
pub struct BmsFamilyScalars {
    pub q_i: Vec<f64>,
    pub g_i: Vec<f64>,
    pub c_i: Vec<f64>,
    pub s: f64,
    pub z_i: Vec<f64>,
}

// ── BlockEffectiveJacobian impls for BMS ─────────────────────────────────────
//
// BMS has a single Bernoulli output per row (n_outputs = 1). The observed η is
//
//   η_i = q_i · c_i + s·g_i · z_i
//
// where
//   q_i   = marginal_design[i,:] · β_m + offset_m[i]      (marginal η)
//   g_i   = logslope_design[i,:] · β_s + offset_s[i]      (log-slope η)
//   s     = probit_frailty_scale(gaussian_frailty_sd)
//   c_i   = sqrt(1 + (s·g_i)²)
//
// Per-block Jacobians ∂η_i / ∂β_block:
//
//   Marginal block  → ∂η_i/∂β_m = c_i · M_i
//     (M_i = marginal_design row i; c_i is β-dependent but does not involve β_m)
//
//   Logslope block  → ∂η_i/∂β_s = (q_i · s²·g_i / c_i + s·z_i) · G_i
//     (G_i = logslope_design row i)
//
// score_warp_dev and link_dev blocks use IFT-corrected η, but their
// contribution to the identifiability audit is captured by the raw design
// columns (the IFT correction adds a direction already in the anchor span at
// compile time). These blocks leave jacobian_callback = None and rely on
// effective_design (= raw design) for the flat audit.

/// β-dependent Jacobian for the BMS marginal block.
///
/// ∂η_i/∂β_m = c_i · M[i,:]
/// where c_i = sqrt(1 + (s · g_i)²),
///       g_i = G[i,:] · β_s + offset_s[i],
///       s   = state.probit_frailty_scale.
///
/// `probit_frailty_scale` is read from the evaluation state at call time (not
/// captured at construction) so the callback remains correct across outer-loop
/// σ updates without rebuilding the block spec.
///
/// Designs are pre-densified at construction to avoid repeated materialisation.
pub struct BmsMarginalJacobian {
    /// Dense marginal design: n × p_m.
    pub marginal_dense: Arc<Array2<f64>>,
    /// Dense logslope design: n × p_s.
    pub logslope_dense: Arc<Array2<f64>>,
    pub offset_m: Array1<f64>,
    pub offset_s: Array1<f64>,
    /// Number of marginal columns (= size of β_m slice in the full β vector).
    pub p_marginal: usize,
}

impl BmsMarginalJacobian {
    pub fn new(
        marginal_dense: Arc<Array2<f64>>,
        logslope_dense: Arc<Array2<f64>>,
        offset_m: Array1<f64>,
        offset_s: Array1<f64>,
        p_marginal: usize,
    ) -> Self {
        Self {
            marginal_dense,
            logslope_dense,
            offset_m,
            offset_s,
            p_marginal,
        }
    }
}

impl BlockEffectiveJacobian for BmsMarginalJacobian {
    fn effective_jacobian_at(
        &self,
        state: &FamilyLinearizationState<'_>,
    ) -> Result<Array2<f64>, String> {
        let beta = state.beta;
        let s = state.probit_frailty_scale;
        let p_m = self.p_marginal;
        let p_s_block = self.logslope_dense.ncols();
        let beta_s_raw = if beta.len() > p_m { &beta[p_m..] } else { &[][..] };
        let p_s_use = p_s_block.min(beta_s_raw.len());
        let beta_s = &beta_s_raw[..p_s_use];
        let n = self.marginal_dense.nrows();
        let p_block = self.marginal_dense.ncols();

        // Compute per-row g_i = G[i, :p_s_use] · β_s + offset_s[i].
        // This block owns the logslope design so g is self-computable.
        // Hard contract (mirrors T26 LogslopeBlockJacobian): when any g_i is
        // nonzero the per-row c_i depends on g and the caller must have
        // populated BmsFamilyScalars in family_scalars.
        let mut g_rows = vec![0.0_f64; n];
        for i in 0..n {
            g_rows[i] = self.offset_s[i]
                + self
                    .logslope_dense
                    .row(i)
                    .slice(ndarray::s![..p_s_use])
                    .dot(&ArrayView1::from(beta_s));
        }

        let scalars: Option<&BmsFamilyScalars> = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<BmsFamilyScalars>());

        let any_nonzero_g = g_rows.iter().any(|&gi| gi != 0.0);
        if any_nonzero_g && scalars.is_none() {
            return Err(
                "bms-marginal block requires BmsFamilyScalars when beta != 0 \
                 (g_i != 0 for at least one row); got family_scalars: None. \
                 The caller must compute per-row (q_i, g_i, c_i, z_i) at the \
                 current beta and pass them via FamilyLinearizationState::family_scalars."
                    .to_string(),
            );
        }

        let mut out = Array2::<f64>::zeros((n, p_block));
        for i in 0..n {
            // c_i from family_scalars when present, otherwise g == 0 so c = 1.
            let c_i = match scalars {
                Some(sc) => sc.c_i[i],
                None => {
                    // g_i == 0.0 here (enforced by contract above), so c_i = 1.
                    let sg = s * g_rows[i];
                    (1.0 + sg * sg).sqrt()
                }
            };
            // J[i,:] = c_i · M[i,:]
            let m_row = self.marginal_dense.row(i);
            out.row_mut(i).assign(&m_row.mapv(|x| c_i * x));
        }
        Ok(out)
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

/// β-dependent Jacobian for the BMS logslope block.
///
/// ∂η_i/∂β_s = (q_i · s²·g_i / c_i + s·z_i) · G[i,:]
/// where q_i = M[i,:] · β_m + offset_m[i],
///       g_i = G[i,:] · β_s + offset_s[i],
///       c_i = sqrt(1 + (s·g_i)²),
///       s   = state.probit_frailty_scale.
///
/// `probit_frailty_scale` is read from the evaluation state at call time.
///
/// Designs are pre-densified at construction to avoid repeated materialisation.
pub struct BmsLogslopeJacobian {
    /// Dense marginal design: n × p_m.
    pub marginal_dense: Arc<Array2<f64>>,
    /// Dense logslope design: n × p_s.
    pub logslope_dense: Arc<Array2<f64>>,
    pub offset_m: Array1<f64>,
    pub offset_s: Array1<f64>,
    pub z: Arc<[f64]>,
    /// Number of marginal columns (= start of β_s in the full β vector).
    pub p_marginal: usize,
}

impl BmsLogslopeJacobian {
    pub fn new(
        marginal_dense: Arc<Array2<f64>>,
        logslope_dense: Arc<Array2<f64>>,
        offset_m: Array1<f64>,
        offset_s: Array1<f64>,
        z: Arc<[f64]>,
        p_marginal: usize,
    ) -> Self {
        Self {
            marginal_dense,
            logslope_dense,
            offset_m,
            offset_s,
            z,
            p_marginal,
        }
    }
}

impl BlockEffectiveJacobian for BmsLogslopeJacobian {
    fn effective_jacobian_at(
        &self,
        state: &FamilyLinearizationState<'_>,
    ) -> Result<Array2<f64>, String> {
        let beta = state.beta;
        let s = state.probit_frailty_scale;
        let p_m = self.p_marginal;
        let p_m_use = p_m.min(beta.len());
        let beta_m = &beta[..p_m_use];
        let beta_s_raw = if beta.len() > p_m { &beta[p_m..] } else { &[][..] };
        let p_s_block = self.logslope_dense.ncols();
        let p_s_use = p_s_block.min(beta_s_raw.len());
        let beta_s = &beta_s_raw[..p_s_use];
        let n = self.logslope_dense.nrows();

        // Compute per-row g_i to detect whether the hyperbolic formula is needed.
        // Hard contract (mirrors T26): when any g_i != 0 the caller must have
        // populated BmsFamilyScalars in family_scalars with the correct per-row
        // (q_i, g_i, c_i, z_i) at the current beta.
        let mut g_rows = vec![0.0_f64; n];
        for i in 0..n {
            g_rows[i] = self.offset_s[i]
                + self
                    .logslope_dense
                    .row(i)
                    .slice(ndarray::s![..p_s_use])
                    .dot(&ArrayView1::from(beta_s));
        }

        let scalars: Option<&BmsFamilyScalars> = state
            .family_scalars
            .as_ref()
            .and_then(|a| a.downcast_ref::<BmsFamilyScalars>());

        let any_nonzero_g = g_rows.iter().any(|&gi| gi != 0.0);
        if any_nonzero_g && scalars.is_none() {
            return Err(
                "bms-logslope block requires BmsFamilyScalars when beta != 0 \
                 (g_i != 0 for at least one row); got family_scalars: None. \
                 The caller must compute per-row (q_i, g_i, c_i, z_i) at the \
                 current beta and pass them via FamilyLinearizationState::family_scalars."
                    .to_string(),
            );
        }

        let mut out = Array2::<f64>::zeros((n, p_s_block));
        for i in 0..n {
            // q_i, g_i, c_i, z_i from family_scalars when present.
            // When scalars is None, g_i == 0 (enforced above), so c_i = 1,
            // q_i * s^2 * 0 / 1 = 0, and factor = s * z_i only.
            let (q_i, g_i_val, c_i, z_i) = match scalars {
                Some(sc) => (sc.q_i[i], sc.g_i[i], sc.c_i[i], sc.z_i[i]),
                None => {
                    // g_rows[i] == 0 here.
                    let q_i_self = self.offset_m[i]
                        + self
                            .marginal_dense
                            .row(i)
                            .slice(ndarray::s![..p_m_use])
                            .dot(&ArrayView1::from(beta_m));
                    let sg = s * g_rows[i];
                    let c_self = (1.0 + sg * sg).sqrt();
                    (q_i_self, g_rows[i], c_self, self.z[i])
                }
            };
            // per-row scalar factor: q_i · s²·g_i / c_i + s·z_i
            let factor = q_i * s * s * g_i_val / c_i + s * z_i;
            // J[i,:] = factor · G[i,:]
            let g_row = self.logslope_dense.row(i);
            out.row_mut(i).assign(&g_row.mapv(|x| factor * x));
        }
        Ok(out)
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

fn build_blockspec(
    name: &str,
    design: &TermCollectionDesign,
    baseline: f64,
    offset: &Array1<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: name.to_string(),
        design: design.design.clone(),
        offset: offset + baseline,
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
        gauge_priority: 100,
        jacobian_callback: None,
    }
}

fn build_marginal_blockspec_bms(
    design: &TermCollectionDesign,
    baseline: f64,
    offset: &Array1<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
    logslope_design: &TermCollectionDesign,
    logslope_offset: &Array1<f64>,
    logslope_baseline: f64,
    p_marginal: usize,
) -> Result<ParameterBlockSpec, String> {
    let offset_m = offset + baseline;
    let offset_s = logslope_offset + logslope_baseline;
    let marginal_dense = design
        .design
        .try_to_dense_arc("build_marginal_blockspec_bms::marginal")?;
    let logslope_dense = logslope_design
        .design
        .try_to_dense_arc("build_marginal_blockspec_bms::logslope")?;
    let callback: Arc<dyn BlockEffectiveJacobian> = Arc::new(BmsMarginalJacobian {
        marginal_dense: Arc::clone(&marginal_dense),
        logslope_dense,
        offset_m: offset_m.clone(),
        offset_s,
        p_marginal,
    });
    Ok(ParameterBlockSpec {
        name: "marginal_surface".to_string(),
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            (*marginal_dense).clone(),
        )),
        offset: offset_m,
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
        gauge_priority: 100,
        jacobian_callback: Some(callback),
    })
}

fn build_logslope_blockspec_bms(
    design: &TermCollectionDesign,
    baseline: f64,
    offset: &Array1<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
    marginal_design: &TermCollectionDesign,
    marginal_offset: &Array1<f64>,
    marginal_baseline: f64,
    z: Arc<[f64]>,
    p_marginal: usize,
) -> Result<ParameterBlockSpec, String> {
    let offset_s = offset + baseline;
    let offset_m = marginal_offset + marginal_baseline;
    let marginal_dense = marginal_design
        .design
        .try_to_dense_arc("build_logslope_blockspec_bms::marginal")?;
    let logslope_dense = design
        .design
        .try_to_dense_arc("build_logslope_blockspec_bms::logslope")?;
    let callback: Arc<dyn BlockEffectiveJacobian> = Arc::new(BmsLogslopeJacobian {
        marginal_dense,
        logslope_dense: Arc::clone(&logslope_dense),
        offset_m,
        offset_s: offset_s.clone(),
        z,
        p_marginal,
    });
    Ok(ParameterBlockSpec {
        name: "logslope_surface".to_string(),
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            (*logslope_dense).clone(),
        )),
        offset: offset_s,
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
        gauge_priority: 100,
        jacobian_callback: Some(callback),
    })
}

pub(crate) fn build_deviation_aux_blockspec(
    name: &str,
    prepared: &DeviationPrepared,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> Result<ParameterBlockSpec, String> {
    let mut block = prepared.block.clone();
    block.initial_log_lambdas = Some(rho);
    let candidate_beta = beta_hint.or_else(|| Some(Array1::<f64>::zeros(block.design.ncols())));
    block.initial_beta = candidate_beta
        .map(|beta| {
            let zero = Array1::<f64>::zeros(beta.len());
            project_monotone_feasible_beta(&prepared.runtime, &zero, &beta, name)
        })
        .transpose()?;
    let mut spec = block.intospec(name)?;
    // Deviation auxiliary blocks (score_warp_dev, link_dev, and any
    // future flex block routed through this builder) model pure
    // shape modifications on top of parametric anchors. They must
    // never own a shared affine direction with the parametric
    // (time / marginal / logslope) blocks. The canonical-gauge
    // selector drops shared directions from blocks with lower
    // gauge_priority first; assigning a value below the parametric
    // default (100) realises that contract automatically.
    spec.gauge_priority = match name {
        "link_dev" => 60,
        // score_warp_dev gets a slightly higher priority than link_dev
        // because in mixed-flex configurations (both blocks present)
        // link_dev is the residualised one (orthogonalised against the
        // parametric anchors PLUS the already-prepared score_warp
        // basis at construction time); link_dev should therefore yield
        // first when an alias still survives into the joint design.
        "score_warp_dev" => 80,
        _ => 70,
    };
    Ok(spec)
}

pub(crate) fn push_deviation_aux_blockspecs(
    blocks: &mut Vec<ParameterBlockSpec>,
    rho: &Array1<f64>,
    cursor: &mut usize,
    score_warp_prepared: Option<&DeviationPrepared>,
    link_dev_prepared: Option<&DeviationPrepared>,
    score_warp_beta_hint: Option<Array1<f64>>,
    link_dev_beta_hint: Option<Array1<f64>>,
) -> Result<(), String> {
    if let Some(prepared) = score_warp_prepared {
        let rho_h = rho
            .slice(s![*cursor..*cursor + prepared.block.penalties.len()])
            .to_owned();
        *cursor += prepared.block.penalties.len();
        blocks.push(build_deviation_aux_blockspec(
            "score_warp_dev",
            prepared,
            rho_h,
            score_warp_beta_hint,
        )?);
    }
    if let Some(prepared) = link_dev_prepared {
        let rho_w = rho
            .slice(s![*cursor..*cursor + prepared.block.penalties.len()])
            .to_owned();
        blocks.push(build_deviation_aux_blockspec(
            "link_dev",
            prepared,
            rho_w,
            link_dev_beta_hint,
        )?);
    }
    Ok(())
}

fn inner_fit(
    family: &BernoulliMarginalSlopeFamily,
    blocks: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    fit_custom_family(family, blocks, options).map_err(|e| e.to_string())
}

pub fn fit_bernoulli_marginal_slope_terms(
    data: ArrayView2<'_, f64>,
    spec: BernoulliMarginalSlopeTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    policy: &crate::resource::ResourcePolicy,
) -> Result<BernoulliMarginalSlopeFitResult, String> {
    let mut spec = spec;
    let data_view = data;
    validate_spec(data_view, &spec)?;
    let mut effective_kappa_options = kappa_options.clone();
    // Honor explicit `length_scale=X` in the user's formula: when every
    // spatial term in BOTH the marginal mean and log-slope blocks carries
    // a user-supplied scalar length scale and no per-axis anisotropy is
    // requested, there is nothing for the joint-spatial outer optimizer
    // to do. Routing through it anyway spends ~80 outer ARC iters stalled
    // at the user's chosen ρ (the n-block ARC's first proposed step lands
    // at the box corner and never recovers), then falls through to the
    // ρ-only "custom family" path which is what we wanted all along.
    // Short-circuit straight to the ρ-only path.
    let kappa_locked_marginal = crate::smooth::all_spatial_terms_kappa_fixed(&spec.marginalspec);
    let kappa_locked_logslope = crate::smooth::all_spatial_terms_kappa_fixed(&spec.logslopespec);
    if effective_kappa_options.enabled && kappa_locked_marginal && kappa_locked_logslope {
        log::info!(
            "[BMS spatial] disabling κ/ψ optimization: every spatial term has an \
             explicit length_scale and no anisotropy; user-supplied kernel scale is fixed"
        );
        effective_kappa_options.enabled = false;
    }
    let flex_spatial_pilot_path = (spec.score_warp.is_some() || spec.link_dev.is_some())
        && spec.y.len() >= BMS_FLEX_SPATIAL_OUTER_PILOT_ROW_THRESHOLD
        && effective_kappa_options.enabled;
    if flex_spatial_pilot_path {
        let marginal_terms = spatial_length_scale_term_indices(&spec.marginalspec);
        let logslope_terms = spatial_length_scale_term_indices(&spec.logslopespec);
        let marginal_updates = apply_spatial_anisotropy_pilot_initializer(
            data_view,
            &mut spec.marginalspec,
            &marginal_terms,
            effective_kappa_options.pilot_subsample_threshold,
            &effective_kappa_options,
        );
        let logslope_updates = apply_spatial_anisotropy_pilot_initializer(
            data_view,
            &mut spec.logslopespec,
            &logslope_terms,
            effective_kappa_options.pilot_subsample_threshold,
            &effective_kappa_options,
        );
        effective_kappa_options.enabled = false;
        log::info!(
            "[BMS spatial] n={} flex=true pilot_geometry_updates={} iterative_spatial_outer=false reason=large-flex-spatial-pilot",
            spec.y.len(),
            marginal_updates + logslope_updates,
        );
    }
    let (z_standardized, z_normalization) = standardize_latent_z_with_policy(
        &spec.z,
        &spec.weights,
        "bernoulli-marginal-slope",
        &spec.latent_z_policy,
    )?;
    spec.z = z_standardized;
    let sigma_learnable = matches!(
        &spec.frailty,
        FrailtySpec::GaussianShift { sigma_fixed: None }
    );
    let initial_sigma = match &spec.frailty {
        FrailtySpec::GaussianShift {
            sigma_fixed: Some(s),
        } => Some(*s),
        FrailtySpec::GaussianShift { sigma_fixed: None } => Some(0.5),
        FrailtySpec::None => None,
        FrailtySpec::HazardMultiplier { .. } => {
            return Err(
                "internal: validate_spec should have rejected unsupported marginal-slope frailty"
                    .to_string(),
            );
        }
    };
    let probit_scale = probit_frailty_scale(initial_sigma);
    let (mut joint_designs, mut joint_specs) = build_term_collection_designs_and_freeze_joint(
        data_view,
        &[spec.marginalspec.clone(), spec.logslopespec.clone()],
    )
    .map_err(|e| e.to_string())?;
    let marginal_design = joint_designs.remove(0);
    let logslope_design = joint_designs.remove(0);
    let marginalspec_boot = joint_specs.remove(0);
    let logslopespec_boot = joint_specs.remove(0);
    let (latent_measure, latent_z_calibration) =
        build_latent_measure_with_geometry(&spec.z, &spec.weights, &spec.latent_z_policy)?;
    if latent_measure.is_empirical() && sigma_learnable {
        return Err("empirical latent-measure marginal-slope calibration requires fixed GaussianShift sigma; learnable sigma derivatives must be fit under the standard-normal latent measure"
                    .to_string());
    }

    let y = Arc::new(spec.y.clone());
    let weights = Arc::new(spec.weights.clone());
    // Apply rank-INT calibration to training z before any downstream
    // consumer (pooled probit baseline, term-collection designs, the
    // family's PIRLS loops) sees it. The calibration is persisted on the
    // fit result so prediction applies the identical monotone map.
    let z = match &latent_z_calibration {
        LatentMeasureCalibration::None => Arc::new(spec.z.clone()),
        LatentMeasureCalibration::RankInverseNormal(cal) => {
            Arc::new(cal.apply_to_training(&spec.z)?)
        }
    };
    let z_train = z.as_ref();
    let pilot_baseline = pooled_probit_baseline(&spec.y, z_train, &spec.weights)?;
    let baseline = (
        bernoulli_marginal_slope_eta_from_probability(
            &spec.base_link,
            normal_cdf(pilot_baseline.0),
            "bernoulli marginal-slope baseline link inversion",
        )?,
        pilot_baseline.1 / probit_scale,
    );

    // Score-warp basis construction is β-independent (identifiability is
    // provided by the smoothness-null-space drop on the basis transform,
    // not by a data-distribution moment anchor at the rigid-pilot η₀), so
    // the standard-normal and empirical latent-measure branches build the
    // same block. There is no row-weight pilot to thread into the basis;
    // the latent-measure split is enforced upstream via the empirical
    // intercept solve in `build_row_exact_context_with_stats`, not in the
    // deviation basis.
    // Score-warp basis is built first, then immediately reparameterised
    // against the parametric span (marginal + logslope columns at the n
    // training rows) so its column span is orthogonal to span(X_marginal,
    // X_logslope) by construction. This is the first half of the joint-
    // design identifiability invariant; the second half (link-deviation
    // orthogonalised against parametric + the now-reparameterised score-
    // warp) runs inside the link-deviation closure below. Together they
    // ensure `[X_marginal | X_logslope | Φ_score_warp · T_sw |
    // Φ_link_dev · T_lw]` has full numerical column rank, structurally
    // bounding `σ_min(joint H + S) ≥ λ_min(S) > 0` regardless of how β
    // drifts the linear predictor distribution during PIRLS.
    // Cross-block W-metric pilot. The joint penalised Hessian during PIRLS
    // uses the probit-style data Hessian row metric
    //
    //   W_pirls[i] = spec.weights[i] · φ(η_i)² / (μ_i·(1−μ_i))
    //
    // which is the canonical IRLS row weight. The cross-block
    // orthogonalisation below must use this metric (not uniform
    // spec.weights) so that `Aᵀ W C̃ = 0` holds in the same inner product
    // the joint Hessian sees — otherwise A and C̃ are merely Euclidean-
    // orthogonal, `Aᵀ W_pirls C̃ ≠ 0`, the joint Hessian carries a near-
    // null direction along the W-metric alias, and REML can drive the
    // flex block's λ small enough that the alias direction's joint
    // Hessian eigenvalue collapses. β then runs away along the alias
    // (manifest as `rho≈2.0`, constant `step_inf`, growing `beta_inf`
    // during PIRLS, and the inner solve hitting `inner_max_cycles`
    // without satisfying the KKT residual).
    //
    // Use the rigid pooled-probit pilot η for score-warp (its basis is
    // β-independent in z, so the rigid pilot suffices) and the one-GN-
    // stepped pilot η for link-deviation (its basis is evaluated at the
    // same eta_pilot used here, so the orthogonalisation metric matches
    // the basis evaluation point exactly). Both are β-independent so the
    // orthogonalisation remains a one-shot construction-time step.
    let rigid_pilot_eta = rigid_pooled_probit_pilot_eta(
        &spec.base_link,
        z_train,
        &spec.marginal_offset,
        &spec.logslope_offset,
        baseline.0,
        baseline.1,
        probit_scale,
    )?;
    let cross_block_pilot_w_score_warp =
        pilot_irls_hessian_row_metric_at_eta(&rigid_pilot_eta, &spec.weights);
    let mut cross_block_warnings: Vec<CrossBlockIdentifiabilityWarning> = Vec::new();
    let score_warp_prepared = if let Some(cfg) = spec.score_warp.as_ref() {
        use super::deviation_runtime::ParametricAnchorBlock;
        let mut prepared = build_score_warp_deviation_block_from_seed(z_train, cfg)?;
        // `install_compiled_flex_block_into_runtime` now delegates
        // its math body to `identifiability_compiler::compile` (commit
        // 4e20b8dc8); the prior Phase-4a shadow compile here was a
        // duplicate of that internal call and has been removed.
        let outcome = install_compiled_flex_block_into_runtime(
            &mut prepared,
            z_train,
            cfg,
            &[
                (&marginal_design.design, ParametricAnchorBlock::Marginal),
                (&logslope_design.design, ParametricAnchorBlock::Logslope),
            ],
            &[],
            &cross_block_pilot_w_score_warp,
        )?;
        match outcome {
            FlexCompileOutcome::Reparameterised => Some(prepared),
            FlexCompileOutcome::FullyAliased { reason } => {
                // Record via the structured channel. Keep the original
                // (non-compiled) design so the unified audit sees score_warp_dev
                // and attributes the drop via dropped_columns (gauge_priority=80
                // is below marginal=150 / logslope=120, so RRQR correctly
                // demotes score_warp_dev when it aliases those blocks).
                cross_block_warnings.push(CrossBlockIdentifiabilityWarning {
                    candidate_label: "score_warp",
                    anchor_summary: "marginal+logslope".to_string(),
                    reason,
                });
                Some(prepared)
            }
        }
    } else {
        None
    };
    // Build the link-deviation block. The basis lives in η-space, and at
    // PIRLS time `runtime.design(η_current)` is re-evaluated at the
    // current β-dependent η, so the basis is genuinely β-dependent during
    // optimisation. The construction-time seed is used only for (a) knot
    // placement in η-space and (b) the cross-block identifiability check
    // that computes the basis-space transform `T` orthogonalising the
    // candidate against the parametric and score-warp anchors at training
    // rows.
    //
    // Using the rigid pooled probit pilot directly (`q0 = a₀·√(…) + s_f·
    // b₀·z`) is structurally degenerate: with zero per-row offsets it is
    // affine in z, so a degree-3 I-spline of `q0` spans the same column
    // space at training rows as a degree-3 I-spline of z, and the cross-
    // block check finds the candidate fully aliased by the score-warp
    // anchor even though at any non-rigid β the link-deviation carries
    // PC/age structure the score-warp cannot represent.
    //
    // Instead, seed both knot placement and the orthogonalisation pivot at
    // a non-rigid pilot η computed via one probit Gauss-Newton step from
    // the rigid pilot onto the full marginal design (see
    // `pilot_eta_for_link_dev_orthogonalisation`). The pilot is row-varying
    // in PCs/age and the resulting `T` drops only directions aliased
    // across all β. The score-warp basis at training rows is also threaded
    // in as a flex anchor when active so the kept directions are jointly
    // orthogonal to parametric ⊕ score-warp.
    let link_dev_prepared = if let Some(cfg) = spec.link_dev.as_ref() {
        let eta_pilot = pilot_eta_for_link_dev_orthogonalisation(
            &spec.base_link,
            &spec.y,
            z_train,
            &spec.weights,
            &marginal_design.design,
            &spec.marginal_offset,
            &spec.logslope_offset,
            baseline.0,
            baseline.1,
            probit_scale,
        )?;
        let link_dev_seed = padded_deviation_seed(&eta_pilot, 1.0, 0.5);
        let mut prepared = build_link_deviation_block_from_knots_design_seed_and_weights(
            &link_dev_seed,
            &eta_pilot,
            cfg,
        )?;
        // Cross-block identifiability for the link-deviation basis. The
        // anchor union covers BOTH possible aliasing channels:
        //
        //  - Parametric: location and logslope designs evaluated at the n
        //    training rows. Columns of `Φ_link_dev(q0)` that reproduce
        //    parametric features become null-direction targets in the
        //    joint penalised Hessian since `S_link_dev` has no mass on
        //    them.
        //
        //  - Score-warp (when active): the now-reparameterised score-warp
        //    basis, also evaluated at training rows. Both flex bases are
        //    cubic I-spline cubic combinations of an η-pilot scalar, and
        //    even with each block's own smoothness-null-space drop their
        //    column spans can still overlap inside the orthogonal
        //    complement of `{1, η_pilot}`.
        //
        // After the orthogonalisation, `[X_marginal | X_logslope |
        // Φ_score_warp · T_sw | Φ_link_dev · T_lw]` has full numerical
        // column rank at training rows, so `σ_min(joint H+S) ≥ λ_min(S)
        // > 0` for every β. This is the standard GAM `gam.side`
        // convention generalised to multi-anchor unions (mgcv applies it
        // sequentially across smooths sharing a covariate).
        // When `install_compiled_flex_block_into_runtime`
        // reparameterised the score-warp runtime against the parametric
        // anchor union (marginal + logslope), it installed an
        // `anchor_residual` and cached the training-row parametric
        // anchor matrix on the runtime. `runtime.design()` on a
        // residualised runtime returns the *raw* basis evaluation,
        // which `assert`s the caller hasn't conflated with the
        // reparameterised basis — we want the reparameterised one
        // here, so go through `design_at_training_with_residual` so
        // the cached anchor rows are folded in. For score-warp
        // configurations where reparameterisation was a no-op (no
        // residual installed) the same call falls back to the raw
        // `design()` path, so the residual-vs-no-residual branches
        // converge on the right matrix.
        let score_warp_anchor_design = score_warp_prepared
            .as_ref()
            .map(|sw| sw.runtime.design_at_training_with_residual(z_train))
            .transpose()?;
        use super::deviation_runtime::ParametricAnchorBlock;
        let parametric_anchors: [(&DesignMatrix, ParametricAnchorBlock); 2] = [
            (&marginal_design.design, ParametricAnchorBlock::Marginal),
            (&logslope_design.design, ParametricAnchorBlock::Logslope),
        ];
        let flex_anchor_slot: Option<&Array2<f64>> = score_warp_anchor_design.as_ref();
        let flex_anchors: Vec<&Array2<f64>> = flex_anchor_slot.into_iter().collect();
        // W-metric for link-deviation orthogonalisation: same IRLS-style
        // probit Hessian row weight as the score-warp path, but evaluated at
        // `eta_pilot` (the one-GN-stepped pilot at which the link-dev basis
        // itself is anchored).
        let cross_block_pilot_w_link_dev =
            pilot_irls_hessian_row_metric_at_eta(&eta_pilot, &spec.weights);
        let outcome = install_compiled_flex_block_into_runtime(
            &mut prepared,
            &eta_pilot,
            cfg,
            &parametric_anchors,
            &flex_anchors,
            &cross_block_pilot_w_link_dev,
        )?;
        match outcome {
            FlexCompileOutcome::Reparameterised => Some(prepared),
            FlexCompileOutcome::FullyAliased { reason } => {
                // Record via the structured channel. Keep the original
                // (non-compiled) design so the unified audit sees link_dev
                // and attributes the drop via dropped_columns (gauge_priority=60
                // is below all parametric blocks so RRQR correctly demotes
                // link_dev when it aliases marginal / logslope / score_warp).
                cross_block_warnings.push(CrossBlockIdentifiabilityWarning {
                    candidate_label: "link_deviation",
                    anchor_summary: "marginal+logslope+score_warp".to_string(),
                    reason,
                });
                Some(prepared)
            }
        }
    } else {
        None
    };
    let extra_rho0 = {
        let mut out = Vec::new();
        if let Some(ref prepared) = score_warp_prepared {
            out.extend(std::iter::repeat_n(0.0, prepared.block.penalties.len()));
        }
        if let Some(ref prepared) = link_dev_prepared {
            out.extend(std::iter::repeat_n(0.0, prepared.block.penalties.len()));
        }
        out
    };
    let setup = joint_setup(
        data_view,
        &marginalspec_boot,
        &logslopespec_boot,
        marginal_design.penalties.len(),
        logslope_design.penalties.len(),
        &extra_rho0,
        &effective_kappa_options,
    );
    let setup = if sigma_learnable {
        setup.with_auxiliary(
            Array1::from_vec(vec![initial_sigma.expect("learnable sigma seed").ln()]),
            Array1::from_vec(vec![0.01_f64.ln()]),
            Array1::from_vec(vec![5.0_f64.ln()]),
        )
    } else {
        setup
    };
    let final_sigma_cell = std::cell::Cell::new(initial_sigma);
    let exact_warm_start = RefCell::new(None::<CustomFamilyWarmStart>);
    // Outer ρ-cache β-seed staging slot. On a cache hit the spatial-joint
    // optimizer invokes `seed_inner_beta_fn` before the first eval at the
    // restored ρ: per-block column widths aren't known until the first
    // `build_blocks(rho, …)` runs, so we stash the flat β here and the eval
    // closures promote it into `exact_warm_start` (the slot the inner
    // PIRLS / Newton solve actually consumes) on their first invocation.
    let pending_beta_seed = RefCell::new(None::<Array1<f64>>);
    let hints = RefCell::new(ThetaHints::default());
    let score_warp_runtime = score_warp_prepared.as_ref().map(|p| p.runtime.clone());
    let link_dev_runtime = link_dev_prepared.as_ref().map(|p| p.runtime.clone());

    let build_blocks = |rho: &Array1<f64>,
                        marginal_design: &TermCollectionDesign,
                        logslope_design: &TermCollectionDesign|
     -> Result<Vec<ParameterBlockSpec>, String> {
        let hints = hints.borrow();
        let mut cursor = 0usize;
        let rho_marginal = rho
            .slice(s![cursor..cursor + marginal_design.penalties.len()])
            .to_owned();
        cursor += marginal_design.penalties.len();
        let rho_logslope = rho
            .slice(s![cursor..cursor + logslope_design.penalties.len()])
            .to_owned();
        cursor += logslope_design.penalties.len();
        let p_m = marginal_design.design.ncols();
        let mut blocks = vec![
            build_marginal_blockspec_bms(
                marginal_design,
                baseline.0,
                &spec.marginal_offset,
                rho_marginal,
                hints.marginal_beta.clone(),
                logslope_design,
                &spec.logslope_offset,
                baseline.1,
                p_m,
            )?,
            build_logslope_blockspec_bms(
                logslope_design,
                baseline.1,
                &spec.logslope_offset,
                rho_logslope,
                hints.logslope_beta.clone(),
                marginal_design,
                &spec.marginal_offset,
                baseline.0,
                Arc::clone(&z),
                p_m,
            )?,
        ];
        push_deviation_aux_blockspecs(
            &mut blocks,
            rho,
            &mut cursor,
            score_warp_prepared.as_ref(),
            link_dev_prepared.as_ref(),
            hints.score_warp_beta.clone(),
            hints.link_dev_beta.clone(),
        )?;
        Ok(blocks)
    };

    let intercept_warm_starts = new_intercept_warm_start_cache(y.len());
    let cell_moment_lru = new_cell_moment_lru_cache(policy);
    let cell_moment_cache_stats = new_cell_moment_cache_stats();
    let make_family = |marginal_design: &TermCollectionDesign,
                       logslope_design: &TermCollectionDesign,
                       sigma: Option<f64>|
     -> BernoulliMarginalSlopeFamily {
        BernoulliMarginalSlopeFamily {
            y: Arc::clone(&y),
            weights: Arc::clone(&weights),
            z: Arc::clone(&z),
            latent_measure: latent_measure.clone(),
            gaussian_frailty_sd: sigma,
            base_link: spec.base_link.clone(),
            marginal_design: marginal_design.design.clone(),
            logslope_design: logslope_design.design.clone(),
            score_warp: score_warp_runtime.clone(),
            link_dev: link_dev_runtime.clone(),
            policy: policy.clone(),
            cell_moment_lru: Arc::clone(&cell_moment_lru),
            cell_moment_cache_stats: Arc::clone(&cell_moment_cache_stats),
            intercept_warm_starts: Some(Arc::clone(&intercept_warm_starts)),
            auto_subsample_phase_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        }
    };

    let marginal_terms = spatial_length_scale_term_indices(&marginalspec_boot);
    let logslope_terms = spatial_length_scale_term_indices(&logslopespec_boot);
    let marginal_has_spatial = !marginal_terms.is_empty();
    let logslope_has_spatial = !logslope_terms.is_empty();
    let analytic_joint_derivatives_available =
        marginal_has_spatial || logslope_has_spatial || setup.log_kappa_dim() == 0;
    if setup.log_kappa_dim() > 0 && !analytic_joint_derivatives_available {
        return Err("exact bernoulli marginal-slope spatial optimization requires analytic joint psi derivatives"
                    .to_string());
    }
    let initial_rho = setup.theta0().slice(s![..setup.rho_dim()]).to_owned();
    let initial_blocks = build_blocks(&initial_rho, &marginal_design, &logslope_design)?;
    let initial_family = make_family(&marginal_design, &logslope_design, initial_sigma);
    let (joint_gradient, joint_hessian) =
        custom_family_outer_derivatives(&initial_family, &initial_blocks, options);
    let analytic_joint_gradient_available = analytic_joint_derivatives_available
        && matches!(
            joint_gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
    // Keep the analytic outer Hessian advertised at biobank scale. The
    // row-tensor terms below are represented through block-local
    // `HyperOperator`s and cached exact-Hessian workspaces, so ARC/trust-region
    // can consume exact HVPs without falling back to BFGS merely because the
    // realized problem is large.
    let analytic_joint_hessian_available =
        analytic_joint_derivatives_available && joint_hessian.is_analytic();
    let kappa_options_ref: &SpatialLengthScaleOptimizationOptions = &effective_kappa_options;
    let sigma_from_theta = |theta: &Array1<f64>| -> Option<f64> {
        if sigma_learnable {
            Some(theta[setup.rho_dim() + setup.log_kappa_dim()].exp())
        } else {
            initial_sigma
        }
    };
    let derivative_block_cache = RefCell::new(
        None::<(
            Array1<f64>,
            Arc<Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>>,
        )>,
    );
    let theta_matches = |left: &Array1<f64>, right: &Array1<f64>| -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right.iter())
                .all(|(lhs, rhs)| (*lhs - *rhs).abs() <= 1e-12 * (1.0 + lhs.abs().max(rhs.abs())))
    };
    let get_derivative_blocks = |theta: &Array1<f64>,
                                 specs: &[TermCollectionSpec],
                                 designs: &[TermCollectionDesign]|
     -> Result<
        Arc<Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>>,
        String,
    > {
        if let Some((cached_theta, cached_blocks)) = derivative_block_cache.borrow().as_ref()
            && theta_matches(cached_theta, theta)
        {
            return Ok(Arc::clone(cached_blocks));
        }

        let built = |specs: &[TermCollectionSpec],
                     designs: &[TermCollectionDesign]|
         -> Result<
            Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
            String,
        > {
            let marginal_psi_derivs = if marginal_has_spatial {
                build_block_spatial_psi_derivatives(data_view, &specs[0], &designs[0])?.ok_or_else(
                    || {
                        "bernoulli marginal-slope: marginal block has spatial terms \
                         but spatial psi derivatives are unavailable"
                            .to_string()
                    },
                )?
            } else {
                Vec::new()
            };
            let logslope_psi_derivs = if logslope_has_spatial {
                build_block_spatial_psi_derivatives(data_view, &specs[1], &designs[1])?.ok_or_else(
                    || {
                        "bernoulli marginal-slope: logslope block has spatial terms \
                         but spatial psi derivatives are unavailable"
                            .to_string()
                    },
                )?
            } else {
                Vec::new()
            };
            let mut derivative_blocks = vec![marginal_psi_derivs, logslope_psi_derivs];
            if score_warp_runtime.is_some() {
                derivative_blocks.push(Vec::new());
            }
            if link_dev_runtime.is_some() {
                derivative_blocks.push(Vec::new());
            }
            if sigma_learnable {
                derivative_blocks
                    .last_mut()
                    .expect("bernoulli derivative block list is non-empty")
                    .push(crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                        None,
                        Array2::zeros((0, 0)),
                        Array2::zeros((0, 0)),
                        None,
                        None,
                        None,
                        None,
                    ));
            }
            Ok(derivative_blocks)
        }(specs, designs)?;
        let built = Arc::new(built);
        derivative_block_cache.replace(Some((theta.clone(), Arc::clone(&built))));
        Ok(built)
    };

    // Bernoulli marginal-slope is a multi-block family with β-dependent
    // joint Hessian: EFS/HybridEFS fixed-point structural invariant fails,
    // so we disable fixed-point at plan time rather than burning cycles on
    // a stalled first attempt that silently falls back.
    let outer_policy = {
        let psi_dim = setup.theta0().len() - setup.rho_dim();
        initial_family.outer_derivative_policy(&initial_blocks, psi_dim, options)
    };
    let exact_spatial_outer_tol = kappa_options_ref.rel_tol.max(1e-6);
    let solved = optimize_spatial_length_scale_exact_joint(
        data_view,
        &[marginalspec_boot.clone(), logslopespec_boot.clone()],
        &[marginal_terms.clone(), logslope_terms.clone()],
        kappa_options_ref,
        &setup,
        crate::seeding::SeedRiskProfile::GeneralizedLinear,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        true,
        None,
        outer_policy,
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            assert_eq!(
                specs.len(),
                designs.len(),
                "spatial joint optimizer must supply one spec per design",
            );
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(&rho, &designs[0], &designs[1])?;
            let sigma = sigma_from_theta(theta);
            final_sigma_cell.set(sigma);
            let family = make_family(&designs[0], &designs[1], sigma);
            let fit = inner_fit(&family, &blocks, options)?;
            let mut hints_mut = hints.borrow_mut();
            let mut bidx = 0usize;
            if let Some(block) = fit.block_states.get(bidx) {
                hints_mut.marginal_beta = Some(block.beta.clone());
            }
            bidx += 1;
            if let Some(block) = fit.block_states.get(bidx) {
                hints_mut.logslope_beta = Some(block.beta.clone());
            }
            bidx += 1;
            if score_warp_prepared.is_some() {
                if let Some(block) = fit.block_states.get(bidx) {
                    hints_mut.score_warp_beta = Some(block.beta.clone());
                }
                bidx += 1;
            }
            if link_dev_prepared.is_some()
                && let Some(block) = fit.block_states.get(bidx)
            {
                hints_mut.link_dev_beta = Some(block.beta.clone());
            }
            Ok(fit)
        },
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         eval_mode,
         row_set: &crate::families::row_kernel::RowSet| {
            use crate::solver::estimate::reml::unified::EvalMode;
            let row_set_rows = match row_set {
                crate::families::row_kernel::RowSet::All => spec.y.len(),
                crate::families::row_kernel::RowSet::Subsample { rows, .. } => rows.len(),
            };
            log::debug!(
                "[BMS exact outer eval] mode={:?} row_set_rows={}",
                eval_mode,
                row_set_rows
            );
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(&rho, &designs[0], &designs[1])?;
            // Promote a staged β seed (deposited by the outer ρ-cache hit
            // before any eval ran) into the family warm-start slot now that
            // we know the per-block widths from the freshly built blocks.
            if let Some(beta_seed) = pending_beta_seed.borrow_mut().take() {
                let widths: Vec<usize> = blocks.iter().map(|b| b.design.ncols()).collect();
                match CustomFamilyWarmStart::from_cached_beta(&widths, &beta_seed) {
                    Ok(ws) => {
                        exact_warm_start.replace(Some(ws));
                    }
                    Err(e) => {
                        log::warn!(
                            "[BMS] outer ρ-cache β-warm-start rejected: {e}; falling back to cold β"
                        );
                    }
                }
            }
            let sigma = sigma_from_theta(theta);
            final_sigma_cell.set(sigma);
            let family = make_family(&designs[0], &designs[1], sigma);
            let derivative_blocks = get_derivative_blocks(theta, specs, designs)?;
            // Downgrade to ValueAndGradient when the caller asks for a
            // Hessian we can't provide; preserve ValueOnly probes for
            // line-search cost-only evaluation.
            let effective_mode = match eval_mode {
                EvalMode::ValueGradientHessian if !analytic_joint_hessian_available => {
                    EvalMode::ValueAndGradient
                }
                other => other,
            };
            let eval = evaluate_custom_family_joint_hyper_shared(
                &family,
                &blocks,
                &joint_hyper_options_for_outer_tolerance(options, exact_spatial_outer_tol),
                &rho,
                derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                effective_mode,
            )?;
            exact_warm_start.replace(Some(eval.warm_start.clone()));
            if !eval.inner_converged {
                return Err(
                    "exact bernoulli marginal-slope inner solve did not converge".to_string(),
                );
            }
            if matches!(eval_mode, EvalMode::ValueGradientHessian)
                && analytic_joint_hessian_available
                && !eval.outer_hessian.is_analytic()
            {
                return Err("exact bernoulli marginal-slope joint [rho, psi] objective did not return an outer Hessian"
                            .to_string());
            }
            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(&rho, &designs[0], &designs[1])?;
            if let Some(beta_seed) = pending_beta_seed.borrow_mut().take() {
                let widths: Vec<usize> = blocks.iter().map(|b| b.design.ncols()).collect();
                match CustomFamilyWarmStart::from_cached_beta(&widths, &beta_seed) {
                    Ok(ws) => {
                        exact_warm_start.replace(Some(ws));
                    }
                    Err(e) => {
                        log::warn!(
                            "[BMS] outer ρ-cache β-warm-start rejected (efs): {e}; falling back to cold β"
                        );
                    }
                }
            }
            let sigma = sigma_from_theta(theta);
            final_sigma_cell.set(sigma);
            let family = make_family(&designs[0], &designs[1], sigma);
            let derivative_blocks = get_derivative_blocks(theta, specs, designs)?;
            let eval = evaluate_custom_family_joint_hyper_efs_shared(
                &family,
                &blocks,
                &joint_hyper_options_for_outer_tolerance(options, exact_spatial_outer_tol),
                &rho,
                derivative_blocks,
                exact_warm_start.borrow().as_ref(),
            )?;
            exact_warm_start.replace(Some(eval.warm_start.clone()));
            if !eval.inner_converged {
                return Err(
                    "exact bernoulli marginal-slope EFS inner solve did not converge".to_string(),
                );
            }
            Ok(eval.efs_eval)
        },
        |beta: &Array1<f64>| {
            if beta.iter().any(|v| !v.is_finite()) {
                crate::bail_invalid_estim!("cached inner beta contains non-finite entries");
            }
            pending_beta_seed.replace(Some(beta.clone()));
            Ok(())
        },
    )?;

    let mut resolved_specs = solved.resolved_specs;
    let mut designs = solved.designs;
    let latent_z_rank_int_calibration = match latent_z_calibration {
        LatentMeasureCalibration::None => None,
        LatentMeasureCalibration::RankInverseNormal(cal) => Some(cal),
    };
    Ok(BernoulliMarginalSlopeFitResult {
        fit: solved.fit,
        marginalspec_resolved: resolved_specs.remove(0),
        logslopespec_resolved: resolved_specs.remove(0),
        marginal_design: designs.remove(0),
        logslope_design: designs.remove(0),
        baseline_marginal: baseline.0,
        baseline_logslope: baseline.1,
        z_normalization,
        latent_measure,
        score_warp_runtime,
        link_dev_runtime,
        gaussian_frailty_sd: final_sigma_cell.get(),
        cross_block_warnings,
        latent_z_rank_int_calibration,
    })
}
