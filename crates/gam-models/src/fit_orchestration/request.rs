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
}

pub struct SurvivalTransformationFitRequest<'a> {
    pub data: ArrayView2<'a, f64>,
    pub spec: SurvivalTransformationTermSpec,
    pub persistent_warm_start_store: Option<gam_runtime::warm_start::ConfiguredWarmStartStore>,
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
    /// Per-smooth basis-adequacy evidence (#2774): the residual lack-of-fit
    /// verdict for each smooth term, or a typed reason it could not be
    /// measured. Empty for a result assembled before the report ran — the
    /// report is attached by the single formula-fit seam that owns the
    /// materialized covariate frame it needs, not by `fit_model`, which does
    /// not know which numeric columns a smooth's covariates are.
    pub basis_adequacy: Vec<crate::fit_orchestration::drivers::BasisAdequacyRow>,
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
    /// eligible scattered-low-d Gaussian fit past the dense-kernel cliff and
    /// every in-cascade proof succeeds. Structurally ineligible shapes stay on
    /// the dense `fit_model` path; after the cascade route is selected, a
    /// quasi-uniformity, automatic-REML, or convergence refusal propagates
    /// instead of silently changing estimators. The cascade-bearing model carries the
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
    /// with no `~` and no response symbol. `crossfit_score_calibration` parses
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
    /// Survival likelihood mode: `Some("transformation" | "location-scale" |
    /// "weibull" | "marginal-slope" | "latent" | "latent-binary")`, or `None`
    /// (the default), which resolves to `"transformation"` at the `Surv(...)`
    /// materialization seam via [`FitConfig::resolved_survival_likelihood`]
    /// (#2301 — no library-side string default). `Some(_)` on a non-survival
    /// response is a typed configuration error.
    pub survival_likelihood: Option<String>,
    /// Explicit centering anchor for the baseline time basis, in the data's own
    /// time units. `None` (the default) lets
    /// [`resolve_survival_time_anchor_for_mode`] pick it from the likelihood mode
    /// and the truncation shape of the data: the robust interior median exit for
    /// marginal-slope and for any genuinely left-truncated dataset (#751/#1790),
    /// the earliest entry age otherwise.
    ///
    /// This is model configuration, not front-end transport (#2631). It used to
    /// exist only as the CLI's `--survival-time-anchor`, which meant the flag was
    /// silently dropped on the CLI's own default (transformation / Weibull)
    /// route — that route delegates to `fit_from_formula`, which had nowhere to
    /// receive it — and a `gam.fit-request` document could not express the anchor
    /// even though the flag declares a conflict with `--request`.
    ///
    /// [`resolve_survival_time_anchor_for_mode`]: crate::survival::resolve_survival_time_anchor_for_mode
    pub survival_time_anchor: Option<f64>,
    /// Residual distribution: "gaussian", "logistic", "gumbel".
    pub survival_distribution: String,
    pub threshold_time_k: Option<usize>,
    pub threshold_time_degree: usize,
    pub sigma_time_k: Option<usize>,
    pub sigma_time_degree: usize,
    /// Number of B-spline basis functions on the `log t` margin of the
    /// **slope** block for the survival marginal-slope family (gam#2765,
    /// gam#2767). `None` — the default — is a slope that does not move along
    /// follow-up. Any `Some(k)` makes `b` a fitted surface in `(x, t)`: the
    /// slope covariate design is tensored against the time margin and the
    /// row program carries `b` at the row's entry time, at its exit time, and
    /// the exit-time rate, so the event density picks up the `q₁·c′₁ + ḃᵀz`
    /// terms a constant slope zeroes out.
    pub slope_time_k: Option<usize>,
    /// Polynomial degree of that margin. Shares the default (`3`) and the
    /// `k >= degree + 1` admission rule with the threshold and sigma margins.
    pub slope_time_degree: usize,

    // Location-scale (GAMLSS)
    /// If set, fit a location-scale model with this formula for the noise parameter.
    pub noise_formula: Option<String>,

    // Marginal-slope
    /// Formula for the slope model (survival marginal-slope or Bernoulli marginal-slope).
    pub slope_formula: Option<String>,
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
    /// Explicit cross-process warm-start capability.
    ///
    /// Default `None`: ordinary fits never consult or write an ambient
    /// machine-global cache. Call
    /// [`FitConfig::with_persistent_warm_start_root`] to opt in with a
    /// caller-owned root. The configured capability is lazy and clone-shared,
    /// so validation creates no directories and every standard, survival, and
    /// custom-family owner uses one opened store handle.
    pub persistent_warm_start_store: Option<gam_runtime::warm_start::ConfiguredWarmStartStore>,
    /// Per-smooth spatial center requests maintained by the adaptive
    /// fit→expand→refit loop. Outer `None` means no loop owns this request, so
    /// raw materialization keeps the ordinary full basis. `Some` activates the
    /// canonical formula workflow: missing inner entries select the structural
    /// identifiable start and `Some(k)` requests the next evidence-backed
    /// resolution for that smooth only. This is in-process orchestration state,
    /// never a user knob or environment setting.
    pub spatial_center_counts: Option<Vec<Option<usize>>>,
    /// Whether to precompute the distribution-free conformal substrates (#942
    /// jackknife+, #1098 exact full-conformal) at fit time and persist them on
    /// the saved model. `None` keeps the historical behaviour of precomputing
    /// whenever the fit is eligible; `Some(false)` skips both.
    ///
    /// The trade-off, measured on `y ~ s(x1,k=6) + s(x2,k=6)` (#2633): the two
    /// substrates are **94% of a saved Gaussian model at n=20,000** (10.2 MB of
    /// 10.85 MB) and grow linearly with the training rows, because they are
    /// per-row. Rebuilding both costs **~5.6 ms**, 0.3% of the fit that produced
    /// them. So keeping them buys single-digit milliseconds at roughly half a
    /// kilobyte per training row, forever — turning the flag off yields a **~16x
    /// smaller** model (10.85 MB -> ~0.65 MB at n=20,000).
    ///
    /// It is opt-OUT rather than opt-in for one reason: rebuilding a substrate
    /// needs the training design AND response back, and a saved model
    /// deliberately does not carry the training rows. So a model that will be
    /// shipped to a host that never sees the training data must keep them, or it
    /// cannot produce a conformal interval at all. Turn this off when the caller
    /// retains its training data, fits in batch, or never asks for conformal
    /// intervals; leave it alone when the model has to stand on its own.
    pub precompute_conformal: Option<bool>,
    /// Whether the fit computes and publishes a coefficient covariance (and the
    /// standard errors derived from it). `None` keeps each family's own
    /// default, which for every path that reaches this field today is "yes";
    /// `Some(false)` asks for point estimates only.
    ///
    /// This exists because it was advice nobody could take (gam#2718). The
    /// bernoulli marginal-slope refusal for a non-StandardNormal latent measure
    /// told callers to "fit without inference if only point estimates are
    /// needed", while `materialize/marginal_slope.rs` set
    /// `compute_covariance = true` unconditionally, so there was no way to
    /// comply. The mechanism was never missing — the latent survival/binary CLI
    /// path has been passing `compute_covariance: false` in production all
    /// along — only a way for a caller to reach it.
    ///
    /// Declining inference is not a way to make a bad covariance acceptable: a
    /// fit that WOULD have withheld its covariance still withholds it and still
    /// declares why (see `CovarianceDeclined`). This only avoids paying for one
    /// that is never read.
    pub compute_covariance: Option<bool>,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            precompute_conformal: None,
            compute_covariance: None,
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
            survival_likelihood: None,
            survival_time_anchor: None,
            survival_distribution: "gaussian".into(),
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            slope_time_k: None,
            slope_time_degree: 3,
            noise_formula: None,
            slope_formula: None,
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
            persistent_warm_start_store: None,
            spatial_center_counts: None,
        }
    }
}
/// The result of materializing a formula + config against a dataset.
pub struct MaterializedModel<'a> {
    pub request: FitRequest<'a>,
    pub inference_notes: Vec<String>,
    /// The survival time basis THIS materialization built, including the time
    /// anchor it centered at. Persistence must record the basis the fit
    /// actually used; re-deriving it downstream from the `FitConfig` silently
    /// diverged whenever the two derivations disagreed — the left-truncation
    /// anchor switch in `materialize_survival` had no counterpart in the save
    /// path, so a left-truncated location-scale model persisted an anchor its
    /// own fit never used (#2470). `None` for every non-survival request.
    pub survival_time_basis: Option<crate::survival::SavedSurvivalTimeBasis>,
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
    fn formula_fits_are_disk_silent_by_default() {
        let config = FitConfig::default();
        assert!(config.persistent_warm_start_store.is_none());
        let options = canonical_standard_fit_options(&config, StandardFitOptionsInputs::default());
        assert!(options.persistent_warm_start_store.is_none());
    }

    #[test]
    fn raw_materialization_does_not_activate_adaptive_spatial_resolution() {
        assert!(
            FitConfig::default().spatial_center_counts.is_none(),
            "raw materialization must not activate a grow loop it does not own"
        );
    }

    #[test]
    fn explicit_root_threads_one_lazy_store_capability() {
        let directory = tempfile::tempdir().expect("create explicit store parent");
        let root = directory.path().join("chosen-root");
        let config = FitConfig::default().with_persistent_warm_start_root(root.clone());
        let configured = config
            .persistent_warm_start_store
            .as_ref()
            .expect("explicit root must configure persistence");
        assert_eq!(configured.root(), root);
        assert!(!root.exists(), "configuration must remain lazy");

        let options = canonical_standard_fit_options(&config, StandardFitOptionsInputs::default());
        let threaded = options
            .persistent_warm_start_store
            .as_ref()
            .expect("canonical options must retain the configured store");
        assert_eq!(threaded.root(), root);
    }
}
