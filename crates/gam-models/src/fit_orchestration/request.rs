use super::*;

#[derive(Clone, Debug)]
pub struct LinkWiggleConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
}

/// Configuration for the second-stage binomial-mean wiggle fit appended to a
/// standard pilot. The blockwise refit options live inside this struct so the
/// pilot config (`link_kind` + `wiggle`) and its required `refit_options` can
/// never disagree: either the whole standard-wiggle request is `Some`, or it
/// is `None`. The previous shape had two sibling `Option` fields on
/// `StandardFitRequest`, which allowed the materialize path to construct an
/// inconsistent state (#320: linkwiggle config without blockwise options).
#[derive(Clone)]
pub struct StandardBinomialWiggleConfig {
    pub link_kind: InverseLink,
    pub wiggle: LinkWiggleConfig,
    pub refit_options: BlockwiseFitOptions,
}

/// Clone-cheap training-matrix backing for a standard fit.
///
/// Ordinary formula fits borrow the projected [`Dataset`] matrix all the way
/// through fitting. A latent-coordinate fit has to augment that matrix during
/// materialization, so it moves the augmented allocation into an [`Arc`]. In
/// both cases cloning this handle aliases the same storage; outer estimators
/// such as expectile LAWS can therefore issue repeated fit requests without
/// copying the complete `n x p` dataset on every iteration.
#[derive(Clone)]
pub enum StandardFitData<'a> {
    Borrowed(ArrayView2<'a, f64>),
    Shared(Arc<Array2<f64>>),
}

impl<'a> StandardFitData<'a> {
    pub fn borrowed(data: ArrayView2<'a, f64>) -> Self {
        Self::Borrowed(data)
    }

    pub fn shared(data: Array2<f64>) -> Self {
        Self::Shared(Arc::new(data))
    }

    pub fn view(&self) -> ArrayView2<'_, f64> {
        match self {
            Self::Borrowed(data) => data.view(),
            Self::Shared(data) => data.view(),
        }
    }

    pub fn nrows(&self) -> usize {
        match self {
            Self::Borrowed(data) => data.nrows(),
            Self::Shared(data) => data.nrows(),
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(data) => data.ncols(),
            Self::Shared(data) => data.ncols(),
        }
    }

    pub fn column(&self, index: usize) -> ArrayView1<'_, f64> {
        match self {
            Self::Borrowed(data) => data.column(index),
            Self::Shared(data) => data.column(index),
        }
    }
}

pub struct StandardFitRequest<'a> {
    pub data: StandardFitData<'a>,
    /// Clone-cheap immutable response backing. Iterative estimators retain one
    /// allocation while issuing multiple standard-fit requests.
    pub y: Arc<Array1<f64>>,
    /// Clone-cheap prior/working-weight backing. A new allocation is made only
    /// when an estimator actually changes the weights.
    pub weights: Arc<Array1<f64>>,
    /// Clone-cheap immutable offset backing.
    pub offset: Arc<Array1<f64>>,
    pub spec: TermCollectionSpec,
    pub family: LikelihoodSpec,
    /// #2026: estimate the Tweedie variance power `p` by profile likelihood
    /// (mgcv `tw()` semantics) before the final fit, rather than trusting the
    /// `p` baked into `family`. Set only for a bare `family="tweedie"`/`"tw"`
    /// request that named no explicit power; an explicit `tweedie(1.6)` pins `p`
    /// and leaves this `false`. When `true`, `family` must carry
    /// `ResponseFamily::Tweedie` on a log link (the placeholder power is
    /// overwritten with the estimate).
    pub estimate_tweedie_p: bool,
    pub options: FitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub wiggle: Option<StandardBinomialWiggleConfig>,
    pub coefficient_groups: Vec<CoefficientGroupSpec>,
    pub penalty_block_gamma_priors: Vec<(String, f64, f64)>,
    pub latent_coord: Option<StandardLatentCoordConfig>,
}

pub struct GaussianLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: GaussianLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleConfig>,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

pub struct BinomialLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: BinomialLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleConfig>,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

pub struct DispersionLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: DispersionGlmLocationScaleTermSpec,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

pub struct SurvivalLocationScaleFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: SurvivalLocationScaleTermSpec,
    pub wiggle: Option<LinkWiggleConfig>,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub optimize_inverse_link: bool,
    /// See [`gam_custom_family::BlockwiseFitOptions::cache_session`].
    /// Threaded into the internally constructed `BlockwiseFitOptions` by
    /// `fit_survival_location_scale_model`.
    pub cache_session: Option<std::sync::Arc<gam_runtime::warm_start::Session>>,
}

pub struct SurvivalTransformationFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: SurvivalTransformationTermSpec,
    /// See [`gam_custom_family::BlockwiseFitOptions::cache_session`].
    /// Threaded into the internally constructed `BlockwiseFitOptions` by
    /// `fit_survival_transformation_model`.
    pub cache_session: Option<std::sync::Arc<gam_runtime::warm_start::Session>>,
}

#[derive(Clone)]
pub struct SurvivalTransformationTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub weights: Array1<f64>,
    pub covariate_spec: TermCollectionSpec,
    pub covariate_offset: Array1<f64>,
    pub baseline_cfg: crate::survival::SurvivalBaselineConfig,
    pub likelihood_mode: crate::survival::SurvivalLikelihoodMode,
    pub time_anchor: f64,
    pub time_build: crate::survival::SurvivalTimeBuildOutput,
    pub timewiggle: Option<LinkWiggleFormulaSpec>,
    pub weibull_seed: Option<(f64, f64)>,
    pub ridge_lambda: f64,
    pub penalty_block_gamma_priors: Vec<(String, f64, f64)>,
}
pub struct BernoulliMarginalSlopeFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: BernoulliMarginalSlopeTermSpec,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub policy: gam_runtime::resource::ResourcePolicy,
}

pub struct SurvivalMarginalSlopeFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: SurvivalMarginalSlopeTermSpec,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}
pub struct LatentSurvivalFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: LatentSurvivalTermSpec,
    pub frailty: FrailtySpec,
    pub options: BlockwiseFitOptions,
}

pub struct LatentBinaryFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: LatentBinaryTermSpec,
    pub frailty: FrailtySpec,
    pub options: BlockwiseFitOptions,
}

pub struct TransformationNormalFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub response: Array1<f64>,
    pub weights: Array1<f64>,
    pub offset: Array1<f64>,
    pub covariate_spec: TermCollectionSpec,
    pub config: TransformationNormalConfig,
    pub options: BlockwiseFitOptions,
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
    pub warm_start: Option<TransformationWarmStart>,
}
pub enum FitRequest<'a> {
    Standard(StandardFitRequest<'a>),
    GaussianLocationScale(GaussianLocationScaleFitRequest<'a>),
    BinomialLocationScale(BinomialLocationScaleFitRequest<'a>),
    DispersionLocationScale(DispersionLocationScaleFitRequest<'a>),
    SurvivalLocationScale(SurvivalLocationScaleFitRequest<'a>),
    SurvivalTransformation(SurvivalTransformationFitRequest<'a>),
    BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest<'a>),
    SurvivalMarginalSlope(SurvivalMarginalSlopeFitRequest<'a>),
    LatentSurvival(LatentSurvivalFitRequest<'a>),
    LatentBinary(LatentBinaryFitRequest<'a>),
    TransformationNormal(TransformationNormalFitRequest<'a>),
}

pub struct StandardFitResult {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    /// Which resolved smooth positions originated from an auto-sized radial
    /// spatial basis. Freeze replaces center strategies with explicit center
    /// matrices, so this provenance must travel beside the result for the
    /// adaptive resolution loop.
    pub adaptive_spatial_terms: Vec<bool>,
    /// Requested (pre-freeze) center counts aligned with
    /// `adaptive_spatial_terms`. Frozen specs store realized center matrices,
    /// whose row count can include periodic image expansion and is therefore
    /// not the next request size.
    pub adaptive_spatial_center_counts: Vec<Option<usize>>,
    pub adaptive_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
    pub kappa_timing: Option<SpatialLengthScaleOptimizationTiming>,
    pub saved_link_state: FittedLinkState,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
    /// Exact canonical function-penalty semantics and smoothing-parameter
    /// order used by the fitted link-wiggle block.
    pub wiggle_penalty_metadata: Option<WigglePenaltyMetadata>,
    /// Standard-basis link-warp coefficients `β_w = Z·γ` for the saved-model
    /// predict runtime when the frozen-basis de-aliasing engaged (#1596). The
    /// fit's coefficients stay in the reduced `γ` coordinate; this lift is
    /// persisted into the payload's `beta_link_wiggle`.
    pub wiggle_saved_warp_beta: Option<Vec<f64>>,
    /// Frozen-index mean-coordinate shift for the predict runtime (#2141),
    /// persisted into the payload's `link_wiggle_index_shift`. Lets predict
    /// evaluate the warp basis at the frozen index `η̂` the fit pinned it at,
    /// rather than at the de-aliased base predictor.
    pub wiggle_saved_index_shift: Option<Vec<f64>>,
}

pub(crate) fn adaptive_spatial_term_mask(spec: &TermCollectionSpec) -> Vec<bool> {
    fn auto_spatial(basis: &gam_terms::smooth::SmoothBasisSpec) -> bool {
        use gam_terms::smooth::SmoothBasisSpec as B;
        match basis {
            B::ByVariable { inner, .. } | B::FactorSumToZero { inner, .. } => auto_spatial(inner),
            B::BySmooth { smooth, .. } => auto_spatial(smooth),
            B::ThinPlate {
                feature_cols, spec, ..
            } => {
                !feature_cols.is_empty()
                    && gam_terms::basis::center_strategy_is_auto(&spec.center_strategy)
            }
            B::Duchon {
                feature_cols, spec, ..
            } => {
                !feature_cols.is_empty()
                    && gam_terms::basis::center_strategy_is_auto(&spec.center_strategy)
            }
            // Matérn's learned range changes both its basin and realized kernel
            // rank as centers move. It has no validated EDF-saturation growth
            // theorem yet, so the generic radial grow loop must not claim it.
            B::Matern { .. } => false,
            B::ConstantCurvature { feature_cols, spec } => {
                !feature_cols.is_empty()
                    && gam_terms::basis::center_strategy_is_auto(&spec.center_strategy)
            }
            B::MeasureJet {
                feature_cols, spec, ..
            } => {
                !feature_cols.is_empty()
                    && gam_terms::basis::center_strategy_is_auto(&spec.center_strategy)
            }
            _ => false,
        }
    }

    spec.smooth_terms
        .iter()
        .map(|term| auto_spatial(&term.basis))
        .collect()
}

pub(crate) fn adaptive_spatial_center_counts(spec: &TermCollectionSpec) -> Vec<Option<usize>> {
    fn center_count(basis: &gam_terms::smooth::SmoothBasisSpec) -> Option<usize> {
        use gam_terms::smooth::SmoothBasisSpec as B;
        match basis {
            B::ByVariable { inner, .. } | B::FactorSumToZero { inner, .. } => center_count(inner),
            B::BySmooth { smooth, .. } => center_count(smooth),
            B::ThinPlate {
                feature_cols, spec, ..
            } if !feature_cols.is_empty() => {
                Some(spec.center_strategy.planned_num_centers(feature_cols.len()))
            }
            B::Duchon {
                feature_cols, spec, ..
            } if !feature_cols.is_empty() => {
                Some(spec.center_strategy.planned_num_centers(feature_cols.len()))
            }
            B::Matern { .. } => None,
            B::ConstantCurvature { feature_cols, spec } if !feature_cols.is_empty() => {
                Some(spec.center_strategy.planned_num_centers(feature_cols.len()))
            }
            B::MeasureJet {
                feature_cols, spec, ..
            } if !feature_cols.is_empty() => {
                Some(spec.center_strategy.planned_num_centers(feature_cols.len()))
            }
            _ => None,
        }
    }

    spec.smooth_terms
        .iter()
        .map(|term| center_count(&term.basis))
        .collect()
}

pub struct SurvivalLocationScaleFitResult {
    pub fit: SurvivalLocationScaleTermFitResult,
    pub inverse_link: InverseLink,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
    /// Distinct proof for outer inverse-link profiling. The nested unified fit
    /// retains its own smoothing/spatial certificate; optimized-link results
    /// retain this sealed carrier instead of overwriting or dropping either
    /// optimization layer's evidence. `None` means the inverse link was fixed.
    pub(crate) inverse_link_outer: Option<gam_solve::rho_optimizer::CertifiedOuterResult>,
}

impl SurvivalLocationScaleFitResult {
    pub fn inverse_link_outer(&self) -> Option<&gam_solve::rho_optimizer::CertifiedOuterResult> {
        self.inverse_link_outer.as_ref()
    }
}

pub struct SurvivalTransformationFitResult {
    pub fit: UnifiedFitResult,
    pub resolvedspec: TermCollectionSpec,
    pub baseline_cfg: crate::survival::SurvivalBaselineConfig,
    pub likelihood_mode: crate::survival::SurvivalLikelihoodMode,
    /// Persistable snapshot of the time basis used during the fit. Replaces
    /// six previously flat fields (basisname / degree / knots / keep_cols /
    /// smooth_lambda / anchor) so the FFI save path consumes a single
    /// source-of-truth value rather than threading siblings independently.
    pub time_basis: crate::survival::SavedSurvivalTimeBasis,
    pub time_base_ncols: usize,
    pub baseline_timewiggle: Option<TimeWiggleBlockInput>,
}

pub enum FitResult {
    Standard(StandardFitResult),
    GaussianLocationScale(GaussianLocationScaleFitResult),
    BinomialLocationScale(BinomialLocationScaleFitResult),
    DispersionLocationScale(DispersionLocationScaleFitResult),
    SurvivalLocationScale(SurvivalLocationScaleFitResult),
    SurvivalTransformation(SurvivalTransformationFitResult),
    BernoulliMarginalSlope(BernoulliMarginalSlopeFitResult),
    SurvivalMarginalSlope(SurvivalMarginalSlopeFitResult),
    LatentSurvival(LatentSurvivalTermFitResult),
    LatentBinary(LatentBinaryTermFitResult),
    TransformationNormal(TransformationNormalFitResult),
    /// Exact O(n) state-space cubic/linear/quintic smoothing-spline scan
    /// (#1030/#1034). A scan-bearing model IS a Gaussian-identity model with a
    /// different (exact) representation: rather than a dense design + coefficient
    /// vector it carries the Durbin–Koopman smoother posterior directly (knots,
    /// smoothed states, pointwise variances, σ², log λ, exact diffuse-REML EDF,
    /// and an exact per-row `predict`). Library callers that want the fitted
    /// posterior get it here without paying the dense O(n·k²)+O(k³) route; the
    /// CLI/FFI save paths build the persistence payload from the same
    /// `SplineScanFit` via `assemble_spline_scan_payload`.
    SplineScan(gam_solve::spline_scan::SplineScanFit),
    /// O(n log n) multiresolution residual-cascade smooth (#1032). UNLIKE the
    /// 1-D scan, the cascade is NOT the same posterior as the Duchon/Matérn term
    /// it stands in for (a different finite basis — the multilevel Wendland
    /// frame), so it is never a silent swap: this variant is produced only when
    /// the structural detector [`residual_cascade_fast_path`] fires on an
    /// eligible scattered-low-d Gaussian fit past the dense-kernel cliff AND the
    /// in-cascade quasi-uniformity guard certifies the metric; every other shape
    /// (and a rejected metric) falls through to the dense `fit_model` path. The
    /// cascade-bearing model carries the
    /// [`ResidualCascadeFit`](gam_solve::residual_cascade::ResidualCascadeFit)
    /// directly — knots-free nested geometry, coefficients, the factored
    /// precision, and an exact per-row `predict`; the CLI/FFI save paths build
    /// the persistence payload from its `to_state` snapshot.
    ResidualCascade(gam_solve::residual_cascade::ResidualCascadeFit),
}

/// Result of a dispersion-channel GAMLSS location-scale fit (#913). Wraps the
/// shared two-block [`BlockwiseTermFitResult`] (mean + log-precision designs
/// and coefficients) plus the family kind so the save path can stamp the right
/// likelihood. These families have no link-wiggle and no response
/// standardization, so the result is a thin wrapper.
pub struct DispersionLocationScaleFitResult {
    pub fit: BlockwiseTermFitResult,
    pub kind: DispersionFamilyKind,
}

/// Out-of-fold Stage-1 latent score and its score-influence Jacobian for a
/// CTN → marginal-slope chain. `z_oof` (length n) replaces the in-sample `z`
/// the Stage-2 model consumes; `jac_oof` (n × p₁) is fed to the Stage-2 spec's
/// `score_influence_jacobian` so the joint solve absorbs the realized leakage
/// directions `Z_infl = diag(s_f·β̂₀)·J`.
pub struct CrossFitScoreCalibration {
    pub z_oof: Array1<f64>,
    pub jac_oof: Array2<f64>,
}

/// Internal recipe describing the CTN Stage-1 fit that produced a Stage-2 `z`
/// column. This is in-process plumbing — never a CLI flag, env var, or feature
/// gate. The orchestration layer populates [`FitConfig::ctn_stage1`] when (and
/// only when) the marginal-slope `z` was generated by a transformation-normal
/// Stage-1 fit; its presence is the sole auto-enable signal for cross-fitted
/// orthogonalization (design §5). When absent, Stage-2 falls back to the free
/// 1-D `score_warp` spline (which spans only the x-free leakage column).
#[derive(Clone, Debug)]
pub struct CtnStage1Recipe {
    /// Stage-1 response column name (the `y` the CTN transforms).
    pub response_column: String,
    /// Stage-1 covariate-side formula right-hand side (e.g. `"s(pc1) + s(pc2)"`),
    /// with no `~` and no response symbol. [`crossfit_score_calibration`] parses
    /// it and builds the CTN covariate basis exactly as
    /// `materialize_transformation_normal` does, then FREEZES that basis once on
    /// the full data and reuses the frozen spec for every fold's refit — so the
    /// rebuilt covariate design has an identical column geometry across folds,
    /// keeping `J`'s `p₁ = p_resp · p_cov` columns aligned (design §3).
    ///
    /// The recipe carries the formula RHS (a primitive string) rather than a
    /// resolved [`TermCollectionSpec`] because this struct is populated both via
    /// [`CtnStage1Recipe::new`] (set on [`FitConfig::ctn_stage1`], then
    /// [`fit_from_formula`]) and by the gamfit FFI marshaller
    /// (`gamfit/_calibrated_slope.py`), which can only serialize primitives over
    /// the JSON boundary — a `TermCollectionSpec` is not serializable. Freezing on
    /// the full Stage-2 data is equivalent to
    /// freezing on the Stage-1 data whenever the two stages share a frame (the
    /// calibrated-chain contract), so the column geometry still matches Stage-1.
    pub covariate_formula_rhs: String,
    /// Stage-1 CTN config (response basis degree / knot count / penalties).
    /// Its `response_num_internal_knots` is the FIXED response-basis size; the
    /// cross-fit pins it across folds so `p_resp` (and hence `p₁`) is
    /// fold-invariant (design §3).
    pub config: TransformationNormalConfig,
    /// Optional Stage-1 weight column name.
    pub weight_column: Option<String>,
    /// Optional Stage-1 offset column name.
    pub offset_column: Option<String>,
}

impl CtnStage1Recipe {
    /// Build a Stage-1 CTN recipe from the Stage-1 description. This is the public
    /// way to populate [`FitConfig::ctn_stage1`] — set it on a marginal-slope
    /// config and run [`fit_from_formula`] (the entry IS `fit_from_formula` with
    /// `ctn_stage1` set; there is no separate combined entry function). The
    /// materializer then cross-fits the CTN and installs the leakage-projection
    /// block; supplying the recipe *is* the request for orthogonalization.
    ///
    /// `response` is the Stage-1 CTN response column; `covariates` is the
    /// covariate-side formula right-hand side (e.g. `"s(pc1) + s(pc2)"` — no `~`,
    /// no response symbol). Validates both are non-empty and that `covariates`
    /// is an RHS only.
    pub fn new(
        response: &str,
        covariates: &str,
        config: TransformationNormalConfig,
        weight_column: Option<&str>,
        offset_column: Option<&str>,
    ) -> Result<Self, String> {
        let response_column = response.trim().to_string();
        if response_column.is_empty() {
            return Err("CtnStage1Recipe requires a non-empty Stage-1 response column".to_string());
        }
        let covariate_formula_rhs = covariates.trim().to_string();
        if covariate_formula_rhs.is_empty() {
            return Err(
                "CtnStage1Recipe requires a non-empty Stage-1 covariate formula RHS".to_string(),
            );
        }
        if covariate_formula_rhs.contains('~') {
            return Err(
                "CtnStage1Recipe covariates is a right-hand side only; pass 's(pc1) + s(pc2)', \
                 not 'score ~ s(pc1) + s(pc2)'"
                    .to_string(),
            );
        }
        Ok(Self {
            response_column,
            covariate_formula_rhs,
            config,
            weight_column: weight_column
                .map(str::to_string)
                .filter(|s| !s.trim().is_empty()),
            offset_column: offset_column
                .map(str::to_string)
                .filter(|s| !s.trim().is_empty()),
        })
    }
}
#[derive(Clone, Debug)]
pub struct FitConfig {
    /// Family: "gaussian", "binomial", "poisson", "negative-binomial",
    /// "gamma", "tweedie" (alias "tw"; variance power fixed at p = 1.5), or
    /// None for auto-detect.
    pub family: Option<String>,
    /// Fixed size/overdispersion parameter for `family="negative-binomial"`.
    pub negative_binomial_theta: Option<f64>,
    /// Link: "identity", "logit", "probit", "cloglog", "sas", "beta-logistic", or None.
    pub link: Option<String>,
    /// Whether to use flexible (wiggle-augmented) link.
    pub flexible_link: bool,
    /// Optional additive offset column for the primary linear predictor.
    pub offset_column: Option<String>,
    /// Optional additive offset column for the noise/log-scale predictor.
    pub noise_offset_column: Option<String>,
    /// Family-level frailty. `None` is represented only by
    /// [`FrailtySpec::None`]; an outer `Option` would create two null states.
    pub frailty: FrailtySpec,

    // Survival-specific
    /// Baseline target: "linear", "weibull", "gompertz", "gompertz-makeham".
    pub baseline_target: String,
    pub baseline_scale: Option<f64>,
    pub baseline_shape: Option<f64>,
    pub baseline_rate: Option<f64>,
    pub baseline_makeham: Option<f64>,
    /// Time basis: "ispline" or "none".
    pub time_basis: String,
    pub time_degree: usize,
    pub time_num_internal_knots: usize,
    pub time_smooth_lambda: f64,
    /// Survival likelihood mode: "location-scale", "transformation", "weibull",
    /// "marginal-slope", "latent", or "latent-binary".
    pub survival_likelihood: String,
    /// Residual distribution: "gaussian", "logistic", "gumbel".
    pub survival_distribution: String,
    pub threshold_time_k: Option<usize>,
    pub threshold_time_degree: usize,
    pub sigma_time_k: Option<usize>,
    pub sigma_time_degree: usize,

    // Location-scale (GAMLSS)
    /// If set, fit a location-scale model with this formula for the noise parameter.
    pub noise_formula: Option<String>,

    // Marginal-slope
    /// Formula for the log-slope model (survival marginal-slope or Bernoulli marginal-slope).
    pub logslope_formula: Option<String>,
    /// Column name for the z (exposure/dose) variable in marginal-slope models.
    pub z_column: Option<String>,
    /// Optional non-negative per-row training weights column.
    pub weight_column: Option<String>,
    /// Expectile asymmetry `τ ∈ (0, 1)` for `family = "expectile"`.
    ///
    /// When `family` resolves to `"expectile"` the fit minimizes the
    /// Newey–Powell asymmetric squared loss `Σ wᵢ(τ)·(yᵢ − μᵢ)²` with
    /// `wᵢ(τ) = τ` if `yᵢ > μᵢ` else `1 − τ`, tracing the conditional
    /// `τ`-expectile — the smooth analogue of the `τ`-quantile. `τ = 0.5`
    /// reduces exactly to the Gaussian-identity mean fit. The whole penalized
    /// smooth + REML `λ`-selection machinery is reused via a Least
    /// Asymmetrically Weighted Squares (LAWS) outer loop. `None` defaults to
    /// the median expectile `τ = 0.5` when the family is `"expectile"`; it is
    /// ignored for every other family. The asymmetry may also be written inline
    /// as `family = "expectile(0.9)"`, which fills this field at resolve time.
    pub expectile_tau: Option<f64>,
    /// Internal CTN Stage-1 provenance for the marginal-slope `z` column.
    ///
    /// When the marginal-slope `z` was generated by a transformation-normal
    /// Stage-1 fit, the orchestration layer fills this with the Stage-1 recipe.
    /// Its presence is the sole auto-enable signal for cross-fitted, Neyman-
    /// orthogonal score calibration (#461): the materializer cross-fits the CTN
    /// to produce out-of-fold `z` and the score-influence Jacobian `J`, replaces
    /// the raw `z` with `z_oof`, and absorbs `J` as a leakage-projection block in
    /// Stage-2. This is in-process plumbing only — there is no CLI flag, env var,
    /// or feature gate. `None` ⇒ raw `z` with the free-warp `score_warp`
    /// fallback. See [`CtnStage1Recipe`].
    pub ctn_stage1: Option<CtnStage1Recipe>,

    // Fitting options
    pub scale_dimensions: bool,
    /// Spatial length-scale/anisotropy optimization policy shared by every
    /// formula family. Front ends must set model-wide spatial knobs here rather
    /// than mutating a request after materialization.
    pub spatial_optimization: SpatialLengthScaleOptimizationOptions,
    /// Enable exact spatial adaptive regularization for standard formula fits.
    /// `None` uses the quality-first automatic policy. The current automatic
    /// policy leaves LAREG off unless explicitly requested because the
    /// optimizer's REML-selected local weights can over-regularize small
    /// high-yield spatial signals.
    pub adaptive_regularization: Option<bool>,
    pub ridge_lambda: f64,

    /// Route the fit through the transformation-normal family.  When set, the
    /// formula terms are treated as the covariate side of the transformation
    /// model and the response basis is built internally.  Incompatible with
    /// `noise_formula` and with `Surv(...)` responses.
    pub transformation_normal: bool,

    /// Enable Firth bias reduction for standard single-parameter families.
    pub firth: bool,
    /// Optional cap on the REML/LAML outer smoothing-parameter iterations for
    /// standard formula fits. `None` uses the production default.
    pub outer_max_iter: Option<usize>,

    /// GPU backend selection policy. `Auto` uses supported device kernels for
    /// large workloads, `Off` pins execution to CPU kernels, and `Required` fails
    /// loudly when a requested GPU kernel has no compiled backend.
    pub gpu_policy: gam_gpu::GpuPolicy,
    /// Optional override of the [`gam_runtime::resource::ResourcePolicy`] used when
    /// planning spatial bases (TPS / Matern / Duchon) during term construction.
    /// When `None`, the default-library policy is used.
    pub resource_policy: Option<gam_runtime::resource::ResourcePolicy>,

    /// Optional per-group metadata supplied by the caller. Fitting ignores this
    /// field; saved-model builders pass it through so deployment consumers can
    /// recover group provenance.
    pub group_metadata: Option<BTreeMap<String, JsonValue>>,

    /// Container type of the caller's training table (`"pandas"`, `"polars"`,
    /// `"pyarrow"`, `"numpy"`, or `"unknown"` outside a typed table frontend).
    /// Fitting ignores this field; saved-model builders persist it so every
    /// current frontend writes the same complete model schema.
    pub training_table_kind: String,

    /// Optional user-defined coefficient groups with separate precision
    /// parameters. Group-local priors, including catalog-metadata-informed
    /// Gamma precision hyperpriors, are resolved during design setup.
    pub coefficient_groups: Vec<CoefficientGroupSpec>,

    /// Optional per-existing-penalty-block Gamma(shape, rate) precision
    /// hyperpriors keyed by penalty-block label. This is the
    /// catalog-metadata-informed-prior hook for models that do not need a new
    /// user-defined coefficient group.
    pub penalty_block_gamma_priors: Vec<(String, f64, f64)>,

    /// Python `gamfit.fit(..., latents={...})` configuration. This reaches
    /// the standard formula workflow as an owned latent-coordinate block:
    /// the named smooth's synthetic covariates are rebuilt from `t`, and
    /// joint REML optimizes `[rho, vec(t)]` through latent design hyper-dirs.
    pub latents: Option<JsonValue>,
    /// Python `gamfit.fit(..., penalties=[...])` analytic-penalty descriptors,
    /// validated against the declared latent-coordinate blocks before a
    /// standard latent fit starts.
    pub analytic_penalties: Option<JsonValue>,
    /// `gamfit.fit(..., smooths={...})` Python kwarg routed through the FFI
    /// bridge. JSON object keyed by formula symbol (single column name or
    /// comma-joined tuple) → smooth descriptor (`{"kind": "duchon",
    /// "centers": [[...], ...], ...}`). Applied as a post-processing step on
    /// the [`TermCollectionSpec`] produced by the formula DSL: each smooth
    /// term whose `feature_cols` match a registry key has its kind-specific
    /// tunables (centers, knots, kernel hyperparameters) overridden with the
    /// user-supplied values. The single canonical lowering path guarantees
    /// `smooths={"x": Duchon(centers=K)}` (integer) produces a bit-identical
    /// block spec to writing `duchon(x, centers=K)` in the formula; only
    /// explicit array-valued `centers=` differs, routing through
    /// `CenterStrategy::UserProvided` instead of `FarthestPoint`/`EqualMass`.
    pub smooth_overrides: Option<JsonValue>,
    /// Engage the cross-process ON-DISK persistent checkpoint layer (#1082).
    ///
    /// Default `true`: formula fits survive process and wall interruptions.
    /// The flag threads
    /// `FitConfig → FitOptions → ExternalOptimOptions` down to the standard
    /// `RemlState`, which then calls `enable_persistent_warm_start_disk()`.
    /// Low-level embedding code may disable it explicitly when it owns a
    /// stronger external checkpoint transaction.
    pub persist_warm_start_disk: bool,
    /// Per-smooth spatial center requests maintained by the adaptive
    /// fit→expand→refit loop. Outer `None` means no loop owns this request, so
    /// raw materialization keeps the ordinary full basis. `Some` activates the
    /// canonical formula workflow: missing inner entries select the structural
    /// identifiable start and `Some(k)` requests the next evidence-backed
    /// resolution for that smooth only. This is in-process orchestration state,
    /// never a user knob or environment setting.
    pub spatial_center_counts: Option<Vec<Option<usize>>>,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            family: None,
            negative_binomial_theta: None,
            link: None,
            flexible_link: false,
            offset_column: None,
            noise_offset_column: None,
            frailty: FrailtySpec::None,
            baseline_target: "linear".into(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".into(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            survival_likelihood: "location-scale".into(),
            survival_distribution: "gaussian".into(),
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            noise_formula: None,
            logslope_formula: None,
            z_column: None,
            weight_column: None,
            expectile_tau: None,
            ctn_stage1: None,
            scale_dimensions: false,
            spatial_optimization: SpatialLengthScaleOptimizationOptions::default(),
            adaptive_regularization: None,
            ridge_lambda: 1e-6,
            transformation_normal: false,
            firth: false,
            outer_max_iter: None,
            gpu_policy: gam_gpu::GpuPolicy::Auto,
            resource_policy: None,
            group_metadata: None,
            training_table_kind: "unknown".to_string(),
            coefficient_groups: Vec::new(),
            penalty_block_gamma_priors: Vec::new(),
            latents: None,
            analytic_penalties: None,
            smooth_overrides: None,
            persist_warm_start_disk: true,
            spatial_center_counts: None,
        }
    }
}
/// The result of materializing a formula + config against a dataset.
pub struct MaterializedModel<'a> {
    pub request: FitRequest<'a>,
    pub inference_notes: Vec<String>,
}
pub struct SplineScanInputs {
    /// Abscissae of the single 1-D smooth (training rows of its feature column).
    pub x: Vec<f64>,
    /// Gaussian response.
    pub y: Vec<f64>,
    /// Observation weights (variance is `σ²/w`).
    pub w: Vec<f64>,
    /// Smoothing-spline order `m = penalty_order ∈ {1, 2, 3}`: `m = 1` the
    /// random-walk/linear smoother (penalty `λ∫f′²`), `m = 2` the cubic
    /// smoother (penalty `λ∫f″²`), `m = 3` the quintic smoother (penalty
    /// `λ∫(f‴)²`).
    pub order: usize,
}
pub struct ResidualCascadeInputs {
    /// One slice per coordinate axis (2 or 3) of the single scattered smooth.
    pub coords: Vec<Vec<f64>>,
    /// Gaussian response.
    pub y: Vec<f64>,
    /// Observation weights (variance is `σ²/w`).
    pub w: Vec<f64>,
    /// Per-axis positive metric scaling `diag(metric)` of `z = diag(metric)·x`.
    pub metric: Vec<f64>,
    /// Sobolev smoothness order `s` of the multilevel Wendland-(3,1) prior,
    /// clamped into the native-space window `(d/2, (d+3)/2]` (issue caveat 1).
    pub sobolev_s: f64,
}

#[cfg(test)]
mod default_workflow_policy_tests {
    use super::*;

    #[test]
    fn formula_fits_checkpoint_durably_by_default() {
        let config = FitConfig::default();
        assert!(config.persist_warm_start_disk);
        let options = canonical_standard_fit_options(&config, StandardFitOptionsInputs::default());
        assert!(options.persist_warm_start_disk);
    }

    #[test]
    fn raw_materialization_does_not_activate_adaptive_spatial_resolution() {
        assert!(
            FitConfig::default().spatial_center_counts.is_none(),
            "raw materialization must not activate a grow loop it does not own"
        );
    }

    #[test]
    fn explicit_external_checkpoint_owner_can_disable_disk_layer() {
        let config = FitConfig {
            persist_warm_start_disk: false,
            ..FitConfig::default()
        };
        let options = canonical_standard_fit_options(&config, StandardFitOptionsInputs::default());
        assert!(!options.persist_warm_start_disk);
    }
}
