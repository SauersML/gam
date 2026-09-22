//! Public input term-spec and fit-result types, the derivative-guard
//! tolerance defaults/helpers, and full input validation (`validate_spec`).
//! This is the user-facing data contract plus its integrity checks.

use super::*;

/// Family-owned survival-baseline coordinates for the joint LAML surface.
///
/// `Linear` is structurally fixed and contributes no family hyperparameter
/// axes. `Nonlinear` owns one frozen offset chart; its theta coordinates are
/// optimized jointly with smoothing, spatial, and learned-frailty axes.
#[derive(Clone, Debug)]
pub enum SurvivalMarginalSlopeBaselineHyperSpec {
    Linear {
        /// Exact fixed baseline represented by the prepared offset channels.
        /// It contributes zero optimizer axes but is carried through the fit
        /// result without reconstruction or fallback.
        config: crate::survival::construction::SurvivalBaselineConfig,
    },
    Nonlinear {
        chart: crate::survival::construction::SurvivalMarginalSlopeFrozenOffsetChart,
    },
}

#[derive(Clone)]
pub struct SurvivalMarginalSlopeTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<f64>,
    pub weights: Array1<f64>,
    pub z: Array2<f64>,
    pub base_link: InverseLink,
    pub marginalspec: TermCollectionSpec,
    pub marginal_offset: Array1<f64>,
    /// GaussianShift frailty on the final probit index: U ~ N(0, σ²) added
    /// to the scalar argument of Φ.  This is exact because the sextic
    /// microcell kernel is preserved — the Gaussian-decoupling identity
    /// E[Φ(η + U)] = Φ(η / √(1+σ²)) rescales the index by 1/τ where
    /// τ = √(1+σ²), and every derivative chain rule factor is polynomial
    /// in τ, so all six kernel derivatives remain closed-form.
    ///
    /// **HazardMultiplier frailty is NOT supported in this family.**
    /// HazardMultiplier frailty + score_warp/linkwiggle cubic marginal-slope
    /// is not finite-state exact.  For hazard-multiplier frailty, use the
    /// standalone LatentCloglogBinomial / LatentSurvival families instead.
    pub frailty: FrailtySpec,
    /// Strict lower bound on q'(t) used by both the likelihood domain and
    /// the monotonicity constraints.
    pub derivative_guard: f64,
    pub baseline_hyper: SurvivalMarginalSlopeBaselineHyperSpec,
    pub time_block: TimeBlockInput,
    pub timewiggle_block: Option<TimeWiggleBlockInput>,
    pub slopespec: TermCollectionSpec,
    pub slopespecs: Option<Vec<TermCollectionSpec>>,
    pub slope_offset: Array1<f64>,
    /// Time margin for the slope block (gam#2765, gam#2767).
    ///
    /// `Static` — the default and everything built before this existed — is a
    /// slope that is constant along follow-up. `TimeVarying` tensors the
    /// slope covariate design against a B-spline margin in `log t`, exactly
    /// as `threshold_time_k` / `sigma_time_k` already do for the location-scale
    /// family, and the row program then carries `b` at entry, at exit, and its
    /// exit-time rate instead of a single per-row scalar.
    pub slope_template: SurvivalCovariateTermBlockTemplate,
    pub score_warp: Option<DeviationBlockConfig>,
    pub link_dev: Option<DeviationBlockConfig>,
    /// Out-of-fold Stage-1 score-influence Jacobian `J = ∂z/∂θ₁` (n × p₁) for a
    /// CTN → marginal-slope chain (issue #461, §3 of
    /// `marginal_slope_orthogonal_design.md`). When `Some`, the score-warp build
    /// site installs the absorbed influence block
    /// `Z_infl = diag(s_f · β̂₀(x_i)) · J` instead of the free-spline score-warp:
    /// the realized x-dependent Stage-1 leakage directions in η-space are
    /// appended as a null-penalized absorbed block (gauge priority 80,
    /// orthogonalized against marginal ⊕ slope), making the β estimating
    /// equation Neyman-orthogonal to `span(Z_infl)`. When `None` (raw `z` with
    /// no Stage-1 model), the free-warp `score_warp` path is used unchanged.
    /// Populated out-of-fold by `crossfit_score_calibration` in
    /// `solver/workflow.rs`; mirrors the BMS spec field of the same name.
    pub score_influence_jacobian: Option<Array2<f64>>,
    /// Policy for the latent score `z`: the normalisation applied first (the
    /// default `Frozen { mean: 0, sd: 1 }` is an identity that only checks and
    /// warns), the standard-normal adequacy thresholds, and — through its
    /// `latent_measure` — whether the AUTOMATIC latent-measure gate runs.
    ///
    /// This family runs the SAME latent-law gate the Bernoulli marginal-slope
    /// runs (gam#2768, gam#2926). Under the default `LatentMeasureSpec::Auto`
    /// the fit estimates the law of the score on the marginal-index span `a(C)`
    /// and anchors on it — one global finite law, or local laws by context when
    /// the law moves on the span — with the score on its own axis.
    /// `LatentMeasureSpec::StandardNormal` declares the Gaussian closed form,
    /// refused when the score's conditional moments move on the span and warned
    /// about, with its excess anchoring loss, when the score fails the adequacy
    /// screen;
    /// `ConditionalLocationScale` declares `ζ = (z − m(C))/√v(C)` with an
    /// empirical residual law. The gate is engaged by
    /// `resolve_survival_latent_score_calibration` (`latent_measure.rs`) on the
    /// frozen marginal design, before any consumer reads `z`; a calibration is
    /// carried per score column in
    /// `SurvivalMarginalSlopeFitResult::latent_z_calibrations`, persisted
    /// through `persisted_latent_z_calibrations`, and replayed at predict
    /// against the marginal block of the q-design.
    pub latent_z_policy: LatentZPolicy,
    /// A DECLARED finite law of the latent score (gam#2923): nodes and weights
    /// the row index is anchored on as given, in place of anything the
    /// automatic gate would decide. The score is taken as supplied — no
    /// pre-transform is fitted to it, because the law is the caller's
    /// statement about that very score — and the fit persists the law as its
    /// latent measure. `None` leaves the measure to `latent_z_policy`.
    pub declared_latent_law: Option<crate::bms::EmpiricalZGrid>,
}

pub(crate) const DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD: f64 = 1e-6;

#[inline]
pub(crate) fn survival_derivative_guard_violated(qd1: f64, derivative_guard: f64) -> bool {
    if !qd1.is_finite() {
        return true;
    }
    // NEG_INFINITY is the "no lower bound" sentinel used by callers that want to
    // skip the guard entirely (e.g. GPU rowjet tests that don't compute a bound).
    // Production paths assert derivative_guard is finite and positive before
    // calling this function, so this branch only fires in the unbounded case.
    if derivative_guard == f64::NEG_INFINITY {
        return false;
    }
    // The band is the builder's offset-only band, so the domain test and the
    // constraint build agree; it admits the boundary-feasible iterates the
    // solver certifies (#788), and log(c·qd1) stays finite for any qd1 > 0.
    !derivative_guard.is_finite()
        || (qd1 + derivative_guard_feasibility_band(qd1, derivative_guard) < derivative_guard)
}

pub struct SurvivalMarginalSlopeFitResult {
    pub fit: UnifiedFitResult,
    pub marginalspec_resolved: TermCollectionSpec,
    pub slopespec_resolved: TermCollectionSpec,
    /// One frozen spec per slope surface when the slope is per-score (`K ≥ 2`
    /// surfaces, one per latent-score column); `None` for a shared slope.
    /// `slopespec_resolved` is their concatenation, which names the terms but
    /// builds one intercept where the surfaces own one each, so it cannot
    /// rebuild `slope_design`; these can, surface by surface (gam#2929).
    pub slope_surface_specs: Option<Vec<TermCollectionSpec>>,
    pub marginal_design: TermCollectionDesign,
    /// Learned or fixed Gaussian-shift frailty SD.  `None` = no frailty.
    pub gaussian_frailty_sd: Option<f64>,
    /// Certified nonlinear baseline selected by the joint LAML solve. Linear
    /// baselines have no family-owned coordinates and therefore return `None`.
    pub baseline_config: crate::survival::construction::SurvivalBaselineConfig,
    pub slope_design: TermCollectionDesign,
    /// The resolved follow-up time margin of the slope block, when the fit
    /// asked for a slope that varies along follow-up (gam#2765, gam#2767).
    ///
    /// This is fit state a predictor would have to replay: with a margin
    /// present, `slope_design` is the tensor product `X_cov ⊗ᵣ B(log t)` and
    /// `slopespec_resolved` still describes only the covariate factor, so a
    /// predictor that rebuilt the design from the spec alone would produce
    /// `p_cov` columns against a `p_cov·p_time` coefficient vector. The knots
    /// are carried here for exactly the reason the threshold and sigma margins
    /// carry theirs — a prediction sample must never be allowed to move the
    /// basis by re-estimating quantile knots.
    pub slope_time_basis: Option<crate::survival::location_scale::SurvivalCovariateTimeBasis>,
    pub baseline_slope: f64,
    pub baseline_offset_residuals: OffsetChannelResiduals,
    pub baseline_offset_curvatures: OffsetChannelCurvatures,
    /// The fitted marginal survival index `q̂(t_i, a_i)` at every training
    /// row's exit time (gam#2923). Under the family's defining identity this
    /// is the probit of the row's marginal survival at exit — on the
    /// standard-normal law by the closed-form lowering, on a declared law by
    /// the anchoring equation — so it is the quantity a calibration check of
    /// the marginal index reads.
    pub fitted_exit_index: Array1<f64>,
    pub z_normalization: LatentZNormalization,
    /// The automatic latent-measure gate's decision, one entry per latent-score
    /// column in column order (gam#2768). `LatentMeasureCalibration::None` means
    /// the gate did not fire on that column and z reached the kernel unchanged.
    ///
    /// This is *fit state that prediction must replay*: the fitted coefficients
    /// are defined on the calibrated axis, so a predictor that re-derived the
    /// map from its own sample — or skipped it — would evaluate a different
    /// model. The saved payload carries it for exactly that reason.
    ///
    /// The measure itself is [`Self::latent_measure`].
    pub latent_z_calibrations: Vec<crate::bms::LatentMeasureCalibration>,
    /// The latent measure the row program integrated against (gam#2923).
    ///
    /// `StandardNormal` is the Gaussian closed form `c(a) = √(1 + r(a)ᵀΣ(a)r(a))`.
    /// An empirical kind is the declared finite law the fit anchored the index
    /// on — `Σ_k w_k Φ(−(α + rᵀz_k)) = Φ(−q)` solved per row — and it is *fit
    /// state prediction must replay*: the coefficients are defined against that
    /// law's anchor, so a predictor lowering the same coefficients in closed
    /// form would evaluate a different model. Persisted as the payload's
    /// `latent_measure`, which the shared marginal-slope predictor already
    /// replays by the same anchoring equation.
    pub latent_measure: crate::bms::LatentMeasureKind,
    /// The certified compression the fit anchored on in place of a declared law
    /// with many atoms (gam#2928): atoms, bins and nodes, and the certified
    /// anchor error at every row's converged inputs against the anchor's
    /// sampling error. [`Self::latent_measure`] is then the compressed law, which
    /// is what the coefficients are defined against. `None` for a law anchored
    /// as declared.
    pub(crate) latent_law_compression:
        Option<crate::latent_law_compression::DeclaredLawCompressionRecord>,
    /// The declared atoms a compressed fit was certified against (gam#2928),
    /// persisted beside the compressed law it anchored on; `None` for a law
    /// anchored as declared, which [`Self::latent_measure`] already is.
    pub(crate) declared_latent_law: Option<crate::bms::EmpiricalZGrid>,
    /// Which latent law the fit consumed (gam#2926): the estimated law of the
    /// score, a declared finite law, the declared conditional location-scale
    /// law, or the declared Gaussian closed form. Persisted with the model.
    pub latent_law_consumed: crate::bms::LatentLawConsumed,
    /// Whether the conditioning span `a(C)` the conditional calibration was fit
    /// against is the span prediction will rebuild (gam#2768).
    ///
    /// The gate runs against the marginal design frozen *before* the spatial
    /// length-scale search, because the calibrated score has to exist before the
    /// fit can consume it. Prediction rebuilds `a(C)` from the *resolved*
    /// marginal spec. For every design whose columns do not depend on a searched
    /// hyperparameter — linear terms, fixed-knot splines, any `length_scale=`
    /// pinned smooth — those are the same matrix and this is `true`. When an
    /// automatic-κ spatial term in the MARGINAL formula moves the basis, they
    /// are not, and a saved model would apply a different map at predict than
    /// the one its coefficients were fitted under. Checked numerically at the
    /// end of the fit, and refused at persistence rather than at the fit: the
    /// point estimates are on a well-defined axis and are worth keeping; it is
    /// only the *replay* that is impossible.
    ///
    /// `true` whenever no conditional calibration fired — there is then nothing
    /// to reproduce.
    pub latent_conditioning_reproducible: bool,
    /// The POOLED weighted empirical score covariance. Still the fit's summary
    /// statistic and still what the on-disk contract carries; when
    /// [`Self::conditional_score_covariance`] is `Some` it is no longer what the
    /// row program consumed.
    pub score_covariance: Array2<f64>,
    /// The fitted conditional score covariance `Σ(a) = Var(z | a)`, when the
    /// pair-wise Rao gate escalated to one (gam#2766).
    ///
    /// This is fit state prediction would have to replay: the coefficients were
    /// estimated against a per-row `c(a) = √(1 + r(a)ᵀΣ(a)r(a))`, so a predictor
    /// that used the pooled `Σ̄` would evaluate a different model. It only exists
    /// at `K ≥ 2`, which the saved-model contract already refuses (that contract
    /// carries one `z_column` and validates a 1×1 score covariance), so today it
    /// is a fit-time object with a refusal rather than a serialization — see
    /// [`Self::persisted_latent_z_calibrations`].
    pub conditional_score_covariance: Option<crate::bms::ConditionalScoreCovariance>,
    /// The joint latent law of the score vector a `K ≥ 2` per-score fit anchored
    /// its index on (gam#2929): the pooled whitened residual law and its
    /// transport `μ + L(a)·ε`, with the conditional covariance inside it when the
    /// gam#2766 gate escalated. Fit state prediction must replay, for the same
    /// reason [`Self::latent_measure`] is; `None` whenever the fit ran the closed
    /// form or a single score.
    pub joint_latent_law: Option<SurvivalJointLatentLaw>,
    pub time_block_penalties_len: usize,
    pub time_wiggle_knots: Option<Array1<f64>>,
    pub time_wiggle_degree: Option<usize>,
    pub time_wiggle_ncols: usize,
    pub score_warp_runtime: Option<DeviationRuntime>,
    pub link_dev_runtime: Option<DeviationRuntime>,
    /// Width `p₁` of the absorbed Stage-1 influence block (#461) when the fit
    /// hosted a dedicated additive absorber (the trailing block). `None` when no
    /// CTN Stage-1 chain produced an influence Jacobian. The predictor drops the
    /// absorber's `γ`; this width lets it account for the extra trailing block
    /// and slice `γ` out of the joint covariance.
    pub influence_absorber_width: Option<usize>,
    /// Exact residualized training-row absorber design.  This is likelihood
    /// state, not prediction state: ordinary prediction drops the fitted
    /// absorber, while saved-model ALO must replay its row Jacobian exactly.
    pub influence_absorber_design: Option<Array2<f64>>,
}

impl SurvivalMarginalSlopeFitResult {
    /// The single-surface calibration the on-disk model contract carries
    /// (`latent_z_conditional_calibration`) out of
    /// [`Self::latent_z_calibrations`]. Only the declared conditional
    /// location-scale law calibrates a score (gam#2926).
    ///
    /// `K > 1` is a refusal, not a truncation. The saved payload carries one
    /// `z_column`, one slope term collection and one calibration, so a
    /// multi-surface fit whose second-or-later score was calibrated cannot be
    /// represented: prediction would rebuild that column's latent axis WITHOUT
    /// the map the fit applied to it and evaluate a different model. Dropping it
    /// silently is the failure mode this whole issue is about, so it is named at
    /// the point of loss.
    pub fn persisted_latent_z_calibrations(
        &self,
    ) -> Result<Option<crate::bms::LatentZConditionalCalibration>, String> {
        split_persisted_latent_calibrations(
            &self.latent_z_calibrations,
            self.latent_conditioning_reproducible,
            self.joint_latent_law
                .as_ref()
                .is_some_and(|law| !law.score_calibrations.is_empty()),
        )
    }

    /// Refuse to persist a fit whose row program consumed a CONDITIONAL score
    /// covariance (gam#2766).
    ///
    /// This is a refusal at the point of loss, not a silent drop. The saved
    /// contract carries one `z_column` and its loader validates a 1×1 score
    /// covariance, so a `K ≥ 2` model is already unloadable; what a silent write
    /// would additionally lose is the per-row `c(a) = √(1 + r(a)ᵀΣ(a)r(a))` every
    /// fitted coefficient is defined against, which is not recoverable from the
    /// pooled matrix the payload does carry. Naming it here means the failure
    /// arrives at save time with the reason attached rather than at load time as
    /// a shape mismatch.
    pub fn persistable_score_covariance(&self) -> Result<&Array2<f64>, String> {
        // On the joint-law path the conditional covariance is not a per-row `c(a)`
        // the row program consumed: it is the transport of the joint law, and it
        // travels inside that law (gam#2929).
        if self.conditional_score_covariance.is_some() && self.joint_latent_law.is_none() {
            return Err(
                "survival marginal-slope fit consumed a CONDITIONAL score covariance Σ(a) \
                 (gam#2766), and the saved-model contract carries only a single pooled matrix: \
                 persisting it would give prediction the pooled Σ̄ and therefore a different \
                 c(a) = √(1 + r(a)ᵀΣ(a)r(a)) at every row than the coefficients were fitted \
                 under. This state needs K ≥ 2 latent scores, which that contract already \
                 refuses at load (it validates a 1×1 score covariance); the refusal is raised \
                 here so the reason travels with it. Supply scores whose conditional covariance \
                 does not move on the marginal-index span if this model must be saved"
                    .to_string(),
            );
        }
        Ok(&self.score_covariance)
    }
}

/// The persistence decision itself, over exactly what it reads. Split out from
/// the fit result so both refusals are exercisable without standing up a whole
/// fit.
pub(crate) fn split_persisted_latent_calibrations(
    calibrations: &[crate::bms::LatentMeasureCalibration],
    conditioning_reproducible: bool,
    joint_law_carries_maps: bool,
) -> Result<Option<crate::bms::LatentZConditionalCalibration>, String> {
    {
        use crate::bms::LatentMeasureCalibration;
        let any_calibrated = calibrations
            .iter()
            .any(|calibration| !matches!(calibration, LatentMeasureCalibration::None));
        // gam#2949: a `K ≥ 2` fit's maps travel inside the joint latent law, one
        // per coordinate, because prediction reads every coordinate against that
        // law. The single-surface payload field is then EMPTY — one owner for
        // the map a score is read on, so no coordinate can be mapped twice.
        // The reproducibility refusal still binds: the span `a(C)` is rebuilt
        // from the resolved marginal spec whichever object holds the map.
        if joint_law_carries_maps {
            if any_calibrated && !conditioning_reproducible {
                return Err(CONDITIONING_NOT_REPRODUCIBLE.to_string());
            }
            return Ok(None);
        }
        for (column, calibration) in calibrations.iter().enumerate().skip(1) {
            if !matches!(calibration, LatentMeasureCalibration::None) {
                return Err(format!(
                    "survival marginal-slope latent-score column {column} carries a conditional \
                     location-scale calibration, and this fit's joint latent law does not carry \
                     that column's map: the single-surface payload holds exactly one score \
                     surface, so persisting it would give prediction an uncalibrated axis for \
                     that column and a different model from the one that was fitted. Fit the \
                     multi-surface model without latent_measure=\"conditional-location-scale\" if \
                     it must be saved"
                ));
            }
        }
        Ok(match calibrations.first() {
            None | Some(LatentMeasureCalibration::None) => None,
            Some(LatentMeasureCalibration::ConditionalLocationScale(cal)) => {
                if !conditioning_reproducible {
                    return Err(CONDITIONING_NOT_REPRODUCIBLE.to_string());
                }
                Some(cal.clone())
            }
        })
    }
}

/// Why a conditional latent calibration cannot be persisted when the spatial
/// length-scale search moved the design it was fitted against (gam#2926).
///
/// One refusal for one reason: the scalar payload field and the joint latent
/// law's per-coordinate maps (gam#2949) are both read by rebuilding `a(C)` from
/// the RESOLVED marginal spec, so both are refused here on the same evidence.
pub(crate) const CONDITIONING_NOT_REPRODUCIBLE: &str =
    "survival marginal-slope conditional latent calibration was fit against the marginal design \
     frozen before the spatial length-scale search, and that search then moved the design: \
     prediction rebuilds a(C) from the RESOLVED marginal spec, so a saved model would apply a \
     different latent map than the one its coefficients were fitted under. Pin the marginal \
     formula's spatial length_scale=, or supply an already conditionally-standardised score, if \
     this model must be saved";

/// Why a learned Gaussian-shift frailty is refused where the likelihood does not
/// identify it (gam#2938); see
/// [`crate::survival::lognormal_kernel::frailty_identification`].
pub(crate) const LEARNED_FRAILTY_NOT_IDENTIFIED: &str =
    "a learned Gaussian-shift frailty σ is refused: σ is not identified by the likelihood. The \
     survival marginal-slope likelihood reads σ only through the probit scale \
     s(σ) = 1/√(1+σ²) on the observed slope s(σ)·(o + Xβ), and with the slope offset o inside \
     every slope surface's span (a constant offset beside an intercept, by default), \
     (σ, β, λ) ↦ (σ′, β′, λ/c²) with o + Xβ′ = c·(o + Xβ), c = s(σ)/s(σ′), leaves the \
     likelihood and the slope penalty unchanged. The criterion then moves with σ only through \
     the slope block's prior and Laplace terms (by −m·ln c for m unpenalized slope directions \
     while λ is free). A fixed σ only rescales the reported slope (gam#2938)";

pub(crate) fn validate_spec(spec: &SurvivalMarginalSlopeTermSpec) -> Result<(), String> {
    let n = spec.age_entry.len();
    log::debug!(
        "[survival-marginal-slope] fit start n={} marginal_terms={} slope_terms={}",
        n,
        spec.marginalspec.linear_terms.len()
            + spec.marginalspec.random_effect_terms.len()
            + spec.marginalspec.smooth_terms.len(),
        spec.slopespec.linear_terms.len()
            + spec.slopespec.random_effect_terms.len()
            + spec.slopespec.smooth_terms.len(),
    );
    if spec.age_exit.len() != n
        || spec.event_target.len() != n
        || spec.weights.len() != n
        || spec.z.nrows() != n
        || spec.z.ncols() == 0
        || spec.marginal_offset.len() != n
        || spec.slope_offset.len() != n
    {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival-marginal-slope row mismatch: entry={}, exit={}, event={}, weights={}, z={}x{}, marginal_offset={}, slope_offset={}",
                n,
                spec.age_exit.len(),
                spec.event_target.len(),
                spec.weights.len(),
                spec.z.nrows(),
                spec.z.ncols(),
                spec.marginal_offset.len(),
                spec.slope_offset.len()
            ),
        }
        .into());
    }
    if spec.weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires finite non-negative weights".to_string(),
        }
        .into());
    }
    if let Some(jac) = spec.score_influence_jacobian.as_ref() {
        // #461 absorbed influence Jacobian `J = ∂z/∂θ₁` (n × p₁): must align with
        // the fit rows and be finite. A zero-column J carries no leakage
        // directions; the build site treats it as no absorber, but a row
        // mismatch or non-finite entry is a hard error (the residualization Gram
        // and the per-row Z̃ projection both assume `n` aligned finite rows).
        if jac.nrows() != n {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival-marginal-slope score_influence_jacobian has {} rows, expected {n}",
                    jac.nrows()
                ),
            }
            .into());
        }
        if jac.iter().any(|&v| !v.is_finite()) {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "survival-marginal-slope score_influence_jacobian must be finite"
                    .to_string(),
            }
            .into());
        }
    }
    if spec.z.iter().any(|&zi| !zi.is_finite()) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires finite z values".to_string(),
        }
        .into());
    }
    if spec.marginal_offset.iter().any(|&value| !value.is_finite()) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires finite marginal offsets".to_string(),
        }
        .into());
    }
    if spec.slope_offset.iter().any(|&value| !value.is_finite()) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires finite slope offsets".to_string(),
        }
        .into());
    }
    spec.frailty.validate_for_marginal_slope()?;
    match &spec.frailty {
        FrailtySpec::None => {}
        FrailtySpec::GaussianShift { .. } => {}
        FrailtySpec::HazardMultiplier { .. } => {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "survival-marginal-slope does not support FrailtySpec::HazardMultiplier"
                    .to_string(),
            }
            .into());
        }
    }
    if spec.event_target.iter().any(|&d| d != 0.0 && d != 1.0) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires binary event indicators (0.0 or 1.0)"
                .to_string(),
        }
        .into());
    }
    // Fast-fail on a degenerate all-censored design: the marginal-slope partial
    // likelihood has no events to anchor the hazard scale, so the outer/inner
    // solve cannot make progress and otherwise spins without termination (#789B).
    if !spec.event_target.is_empty() && spec.event_target.iter().all(|&d| d == 0.0) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires at least one event (event==1); the supplied design is entirely censored (all event==0), which has no finite marginal-slope fit"
                .to_string(),
        }
        .into());
    }
    if !spec.derivative_guard.is_finite() || spec.derivative_guard <= 0.0 {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: format!(
                "survival-marginal-slope requires derivative_guard > 0, got {}",
                spec.derivative_guard
            ),
        }
        .into());
    }
    for i in 0..n {
        if spec.age_exit[i] < spec.age_entry[i] {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: format!(
                    "survival-marginal-slope row {i}: exit time ({}) < entry time ({})",
                    spec.age_exit[i], spec.age_entry[i]
                ),
            }
            .into());
        }
    }
    let n_entry = spec.time_block.design_entry.nrows();
    let n_exit = spec.time_block.design_exit.nrows();
    let n_deriv = spec.time_block.design_derivative_exit.nrows();
    if n_entry != n || n_exit != n || n_deriv != n {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival-marginal-slope time block design row mismatch: \
                 data={n}, design_entry={n_entry}, design_exit={n_exit}, design_derivative_exit={n_deriv}"
            ),
        }
        .into());
    }
    let p_entry = spec.time_block.design_entry.ncols();
    let p_exit = spec.time_block.design_exit.ncols();
    let p_deriv = spec.time_block.design_derivative_exit.ncols();
    if p_exit != p_entry || p_deriv != p_entry {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival-marginal-slope time block design column mismatch: entry={p_entry}, exit={p_exit}, deriv={p_deriv}"
            ),
        }
        .into());
    }
    for (row, &offset) in spec.time_block.derivative_offset_exit.iter().enumerate() {
        if !offset.is_finite() {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival-marginal-slope coordinate-cone time block has non-finite derivative offset at row {row}: {offset}"
                ),
            }
            .into());
        }
        // The offset is the row's q' at β = 0, so it is held to the one
        // guard predicate the solver and the row kernels use (#3766).
        if survival_derivative_guard_violated(offset, spec.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival-marginal-slope coordinate-cone time block requires derivative offset >= guard at row {row}: offset={offset:.3e}, guard={:.3e}",
                    spec.derivative_guard
                ),
            }
            .into());
        }
    }
    let derivative_design = spec
        .time_block
        .design_derivative_exit
        .try_to_dense_by_chunks("survival marginal-slope coordinate-cone derivative audit")
        .map_err(|reason| SurvivalMarginalSlopeError::IncompatibleDimensions { reason })?;
    let p_time = derivative_design.ncols();
    // Each coordinate-cone derivative entry is analytically ≥ 0 but is formed
    // as a right-cumulative sum of at most `p_time` B-spline derivatives
    // `dB_k`; since `dB_k = D_k − D_{k+1}`, `Σ_k |dB_k| ≤ 2 Σ_j |D_j|`, so the
    // computed entry carries at most `accumulation_band(p_time, 2 Σ_j |D_j|)`
    // of roundoff (depth `p_time`: one rounding forming each `dB_k` plus the
    // `p_time − 1` additions). Only a negative entry beyond that band is a real sign
    // violation.
    let derivative_row_bands: Vec<f64> = derivative_design
        .rows()
        .into_iter()
        .map(|row| {
            gam_linalg::roundoff::accumulation_band(
                p_time,
                2.0 * row.iter().map(|v| v.abs()).sum::<f64>(),
            )
        })
        .collect();
    for ((row, col), &value) in derivative_design.indexed_iter() {
        if !value.is_finite() {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival-marginal-slope coordinate-cone time block has non-finite derivative design entry at row {row}, col {col}: {value}"
                ),
            }
            .into());
        }
        if value < -derivative_row_bands[row] {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival-marginal-slope coordinate-cone time block requires nonnegative derivative design entries; row {row}, col {col} = {value:.3e}"
                ),
            }
            .into());
        }
    }
    if let Some(beta0) = &spec.time_block.initial_beta {
        // Under a coordinate-cone time basis, the solver enforces β ≥ 0
        // directly. The row-wise derivative guard is redundant because
        // validation above proves D ≥ 0 up to its accumulation roundoff and
        // offset ≥ guard under the shared feasibility band. The seed's β ≥ 0
        // rows are the same active-set bound rows, so a seed coordinate is
        // held to that band with guard 0: a projected seed that the active
        // set certifies as primal feasible is accepted, a real negative is not.
        if spec.time_block.design_derivative_exit.ncols() != beta0.len() {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival-marginal-slope time_block initial_beta length mismatch under coordinate-cone monotonicity: got {}, expected {}",
                    beta0.len(),
                    spec.time_block.design_derivative_exit.ncols()
                ),
            }
            .into());
        }
        for (j, &g) in beta0.iter().enumerate() {
            if !g.is_finite() {
                return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                    reason: format!(
                        "survival-marginal-slope time_block initial_beta is non-finite at coordinate {j} under coordinate-cone monotonicity: got {g}"
                    ),
                }
                .into());
            }
            if survival_derivative_guard_violated(g, 0.0) {
                return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                    reason: format!(
                        "survival-marginal-slope time_block initial_beta violates β ≥ 0 at coordinate {j} under coordinate-cone monotonicity: got {g:.3e}"
                    ),
                }
                .into());
            }
        }
    }
    if let Some(timewiggle) = spec.timewiggle_block.as_ref() {
        if timewiggle.degree != 3 {
            return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
                reason: format!(
                    "survival-marginal-slope timewiggle requires cubic degree=3, got {}",
                    timewiggle.degree
                ),
            }
            .into());
        }
        let derived_ncols = time_wiggle_basis_ncols(&timewiggle.knots, timewiggle.degree)?;
        if derived_ncols == 0 {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason:
                    "survival-marginal-slope timewiggle requires at least one wiggle coefficient"
                        .to_string(),
            }
            .into());
        }
        if timewiggle.ncols != derived_ncols {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival-marginal-slope timewiggle metadata width mismatch: metadata={}, basis={derived_ncols}",
                    timewiggle.ncols
                ),
            }
            .into());
        }
        if spec.time_block.design_exit.ncols() < derived_ncols {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival-marginal-slope timewiggle requests {} tail columns but time block only has {} columns",
                    derived_ncols,
                    spec.time_block.design_exit.ncols()
                ),
            }
            .into());
        }
    }
    Ok(())
}
