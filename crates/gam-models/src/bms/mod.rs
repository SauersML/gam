use crate::cubic_cell_kernel as exact_kernel;
use crate::custom_family::{
    BatchedOuterGradientTerms, BlockEffectiveJacobian, BlockWorkingSet, BlockwiseFitOptions,
    CustomFamily, CustomFamilyJointHyperModeSelection, CustomFamilyWarmStart, EvalMode,
    ExactNewtonJointGradientEvaluation, ExactNewtonJointHessianWorkspace, FamilyEvaluation,
    FamilyLinearizationState, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
    custom_family_outer_derivatives, evaluate_custom_family_joint_hyper_best_mode_shared,
    fit_custom_family_fixed_log_lambdas_from_mode_selection,
    joint_hyper_options_for_outer_tolerance, upgrade_custom_family_joint_hyper_mode_shared,
};
use crate::exact_mode_branch::ExactCoefficientModeBranch;
use crate::fit_orchestration::drivers::{
    ExactJointEfsEvaluation, ExactJointEvaluation, ExactJointHyperSetup, SpatialFitProvenance,
    build_term_collection_designs_and_freeze_joint, optimize_spatial_length_scale_exact_joint_typed,
    spatial_length_scale_term_indices,
};
use crate::inference::predict_io::FittedLatentScoreMap;
use crate::marginal_slope_shared::{
    CoeffSupport, ObservedDenestedCellPartials, SparsePrimaryCoeffJetView, add_optional_matrix,
    add_optional_vector, add_two_surface_psi_outer,
    build_denested_partition_cells as shared_denested_partition_cells, chunked_row_reduction,
    eval_coeff4_at, first_parameter_directional_order2_terms, first_parameter_order2_terms,
    observed_denested_calibration_newton_coefficients as shared_observed_denested_calibration_newton_coefficients,
    observed_denested_cell_partials as shared_observed_denested_cell_partials, outer_row_indices,
    outer_weighted_rows, parameter_block_specs_match_rows, probit_frailty_scale,
    psi_derivative_location, scale_coeff4, second_parameter_order2_terms,
};
use crate::model_types::UnifiedFitResult;
use crate::outer_subsample::WeightedOuterRow;
use crate::parameter_block::ParameterBlockInput;
use crate::probability::{
    chi_square_sf, normal_cdf, normal_logcdf, normal_pdf, normal_two_sided_probability,
    signed_probit_logcdf_and_mills_ratio, standard_normal_quantile,
};
use crate::row_kernel::{
    RowKernel, RowKernelHessianWorkspace, build_row_kernel_cache, row_kernel_gradient,
    row_kernel_hessian_dense, row_kernel_log_likelihood,
};
use crate::spatial_psi_bridge::{
    CoefficientSpatialPsiBlockTransform, build_block_spatial_psi_derivatives,
    build_block_spatial_psi_derivatives_with_transform,
};
use crate::survival::lognormal_kernel::{FrailtyScale, FrailtySpec};
use gam_linalg::matrix::{DesignMatrix, SymmetricMatrix};
use gam_problem::{
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
    HyperOperator, InverseLink, StandardLink, WigglePenaltyConfig,
};
use gam_solve::estimate::reml::reml_outer_engine::{DenseSpectralOperator, HessianFactorization};
use gam_solve::pirls::LinearInequalityConstraints;
use gam_terms::smooth::{
    SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords, TermCollectionDesign,
    TermCollectionSpec,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, s};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

mod alo_replay;
pub mod deviation_runtime;
pub mod gpu;
pub(crate) use alo_replay::exact_runtime_from_saved;
pub use alo_replay::{BernoulliMarginalSlopeSavedAloReplay, BernoulliMarginalSlopeSavedAloRowGeometry};
pub(crate) use alo_replay::{
    BernoulliMarginalSlopeSavedAloReplayInput, replay_saved_bernoulli_marginal_slope_alo,
};
pub use deviation_runtime::DeviationRuntime;
pub use deviation_runtime::ParametricAnchorBlock;
pub use moving_law_rule::{MovingLawArm, MovingLawArmScore, MovingLawCertificate};

#[derive(Clone, Debug)]
pub struct DeviationBlockConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_order: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
    pub monotonicity_eps: f64,
}

impl Default for DeviationBlockConfig {
    fn default() -> Self {
        WigglePenaltyConfig::cubic_triple_operator_default().into()
    }
}

impl From<WigglePenaltyConfig> for DeviationBlockConfig {
    fn from(cfg: WigglePenaltyConfig) -> Self {
        let penalty_order = *cfg.penalty_orders.iter().max().unwrap_or(&2);
        Self {
            degree: cfg.degree,
            num_internal_knots: cfg.num_internal_knots,
            penalty_order,
            penalty_orders: cfg.penalty_orders,
            double_penalty: cfg.double_penalty,
            monotonicity_eps: cfg.monotonicity_eps,
        }
    }
}

#[derive(Clone)]
pub(crate) struct DeviationPrepared {
    pub(crate) block: ParameterBlockInput,
    pub(crate) runtime: DeviationRuntime,
}

impl std::fmt::Debug for DeviationPrepared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviationPrepared").finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub struct BernoulliMarginalSlopeTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub z: Array1<f64>,
    pub base_link: InverseLink,
    pub marginalspec: TermCollectionSpec,
    pub slopespec: TermCollectionSpec,
    pub marginal_offset: Array1<f64>,
    pub slope_offset: Array1<f64>,
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
    pub score_warp: Option<DeviationBlockConfig>,
    pub link_dev: Option<DeviationBlockConfig>,
    pub latent_z_policy: LatentZPolicy,
    /// Out-of-fold Stage-1 score-influence Jacobian `J = ∂z/∂θ₁` (n × p₁)
    /// from cross-fitting a CTN transformation-normal Stage-1 model (#461).
    /// When `Some`, the realized leakage directions `Z_infl = diag(s_f·β̂₀)·J`
    /// are absorbed as a null-penalized block so the joint solve makes the
    /// β estimating equation orthogonal to `span(Z_infl)` — the x-dependent
    /// realization of `ψ − Π_η[ψ]`. `None` ⇒ raw `--z-column` with no CTN
    /// Stage-1, in which case the free 1-D `score_warp` spline is the
    /// fallback basis (it spans only the x-free leakage column).
    pub score_influence_jacobian: Option<Array2<f64>>,
    /// Residual genetic repair block (gam#2924): `K` conditionally centred
    /// features entering the genetic drive with constant, ridge-shrunk
    /// coefficients, the anchor integrating the joint `(z, r)` law. `None` is
    /// the single-score family unchanged.
    pub residual: Option<residual_repair::ResidualRepairSpec>,
    /// A DECLARED finite law of the latent score (gam#2926): nodes and weights
    /// the intercept is anchored on as given, in place of anything
    /// `latent_z_policy` would estimate or check. The score is taken as supplied
    /// and the law is persisted as the fit's latent measure. `None` leaves the
    /// law to `latent_z_policy`.
    pub declared_latent_law: Option<EmpiricalZGrid>,
}

pub struct BernoulliMarginalSlopeFitResult {
    pub fit: UnifiedFitResult,
    pub marginalspec_resolved: TermCollectionSpec,
    pub slopespec_resolved: TermCollectionSpec,
    pub marginal_design: TermCollectionDesign,
    pub slope_design: TermCollectionDesign,
    pub baseline_marginal: f64,
    pub baseline_slope: f64,
    pub z_normalization: LatentZNormalization,
    pub latent_measure: LatentMeasureKind,
    pub score_warp_runtime: Option<DeviationRuntime>,
    pub link_dev_runtime: Option<DeviationRuntime>,
    /// Learned or fixed Gaussian-shift frailty SD.  `None` = no frailty.
    pub gaussian_frailty_sd: Option<f64>,
    /// Structured warnings emitted during fit-time setup when a flex
    /// block was fully aliased by its anchor union and got dropped. The
    /// fit proceeds without the dropped block (its contribution to the
    /// joint design was numerically reproducible by the anchor span, so
    /// keeping it would leave the joint Hessian rank-deficient). Empty
    /// for fits where every flex block carried independent directions.
    pub cross_block_warnings: Vec<CrossBlockIdentifiabilityWarning>,
    /// Which latent law the fit consumed (gam#2926): the estimated law of the
    /// score (global or local by context), a declared finite law, the declared
    /// conditional location-scale law, or the declared Gaussian closed form.
    /// The law itself is [`Self::latent_measure`]. Persisted with the model.
    pub latent_law_consumed: LatentLawConsumed,
    /// Conditional location-scale calibration of the latent score (#905),
    /// `Some(_)` only under the declared `conditional-location-scale` law when
    /// its `E[z|C]`/`Var(z|C)` Rao test fired: the training z was then replaced
    /// in place by `ζ = (z − m(C))/√v(C)` (through the fitted latent score map,
    /// gam#3016) before any downstream consumer saw it, and the residual is
    /// anchored on its empirical law. Persisted so prediction rebuilds `a(C)` from
    /// the (reproducible) marginal design and applies the identical map.
    pub latent_z_conditional_calibration: Option<LatentZConditionalCalibration>,
    /// The latent score of each training row as the kernel consumed it: the raw
    /// score through the fitted score map (the saved normalisation, then the
    /// conditional calibration when one was minted). Under the conditional law
    /// a saved model returns these same values at these rows through
    /// `FittedModel::latent_conditional_residual` (gam#3016).
    pub latent_score: Array1<f64>,
    /// The fitted residual repair geometry (gam#2924) when a residual block was
    /// supplied: column names, the pooled joint `(z, r)` covariance, the
    /// conditional model when the pairwise gate escalated, and the centring
    /// p-values. The coefficients live in `fit.block_states[2]`.
    pub residual_repair: Option<residual_repair::ResidualRepairGeometry>,
}

#[derive(Clone, Debug)]
pub enum LatentZCheckMode {
    Strict,
    WarnOnly,
    Off,
}

#[derive(Clone, Debug)]
pub enum LatentZNormalizationMode {
    None,
    FitWeighted,
    Frozen { mean: f64, sd: f64 },
}

pub(crate) const DEFAULT_EMPIRICAL_LATENT_GRID_SIZE: usize = 65;
/// The standard-normal adequacy screen's design false-fail rate (gam#2926): the
/// probability that a score drawn exactly N(0, 1) fails it. It is a stated
/// policy, not a tuning knob. Each clause's bound is the null quantile of its own
/// statistic at the sample's Kish effective size, at this level split evenly over
/// the clauses, so no bound is a constant and the screen's family-wise
/// false-fail rate is at most this. The screen decides only which route a fit
/// starts on; the post-fit `D̂` certificate carries the accuracy claim.
pub(crate) const AUTO_Z_NORMAL_SCREEN_ALPHA: f64 = 0.05;
/// The screen's clauses: mean, sd, skewness, excess kurtosis, KS distance, the
/// two tail masses and the largest `|z|`.
const AUTO_Z_NORMAL_SCREEN_CLAUSES: f64 = 8.0;
/// Inner σ level at which the empirical tail mass of latent z is compared
/// against the standard normal's two-sided tail in the adequacy screen.
pub(crate) const AUTO_Z_NORMAL_TAIL_SIGMA_INNER: f64 = 4.0;
/// Outer σ level for the same tail-mass comparison; catches heavier far-tail
/// excess that the inner level can miss.
pub(crate) const AUTO_Z_NORMAL_TAIL_SIGMA_OUTER: f64 = 6.0;
/// Significance level for the conditional `E[z|C]` / `Var(z|C)` Rao gate in the
/// core Auto path (#905). When the latent score's conditional mean or variance
/// on the marginal-index span `a(C)` is significant at this level, the Auto
/// path escalates from the pooled-marginal rank-INT to a conditional
/// location-scale correction. Chosen small (0.1%) so the escalation fires only
/// on clear conditional structure, not finite-sample noise — the gate runs once
/// over the whole training sample, so a per-test α this tight still has ample
/// power against the grouping mean-shift the issue names.
pub(crate) const AUTO_Z_CONDITIONAL_RAO_ALPHA: f64 = 1.0e-3;
/// Relative ridge added to the weighted normal equations when regressing the
/// latent score on the marginal-index span for the conditional correction.
/// Stabilizes the solve when `a(C)` is rank-deficient or collinear (penalized
/// spline marginal indices routinely are) without materially biasing the
/// conditional mean/variance fit.
pub(crate) const AUTO_Z_CONDITIONAL_RIDGE_REL: f64 = 1.0e-8;

/// Which law of the latent score a marginal-slope fit anchors on (gam#2926).
///
/// The anchoring equation `E_p[Φ(α + b·z) | a] = π(a)` has a unique solution on
/// every finite law, and the closed-form Gaussian lowering is its `N(0, 1)`
/// case, so the default is the law the score HAS and the Gaussian form is a
/// declaration a caller makes about the score, never the target of a transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentMeasureSpec {
    /// The default: estimate the conditional law of the score on the
    /// marginal-index span and anchor on it, the score on its own axis — the
    /// closed-form Gaussian law when the span shows no structure in that law and
    /// the score passes the standard-normal adequacy check, one global finite law
    /// when it fails the check, and when the law moves the simplest of the
    /// Gaussian, location-scale and local laws its cross-fitted certificate admits.
    Auto { grid_size: usize },
    /// A declared Gaussian law: the closed-form lowering, refused when the score's
    /// conditional moments move on the span, and warned about with its estimated
    /// excess anchoring loss when the pooled score fails the adequacy screen.
    StandardNormal,
    /// Always anchor on the pooled empirical law of the score, whatever the span
    /// shows.
    GlobalEmpirical { grid_size: usize },
    /// Declare the conditional law a location-scale family on the span: fit
    /// `m(a)`, `v(a)`, and anchor `ζ = (z − m)/√v` on the empirical law of the
    /// standardised residual. The slope then lives on the `ζ` axis.
    ConditionalLocationScale { grid_size: usize },
}

impl LatentMeasureSpec {
    pub fn auto_default() -> Self {
        Self::Auto {
            grid_size: DEFAULT_EMPIRICAL_LATENT_GRID_SIZE,
        }
    }
}

impl Default for LatentMeasureSpec {
    fn default() -> Self {
        Self::auto_default()
    }
}

/// What the conditional-law structure test measured on the marginal-index span
/// (gam#2926): robust Rao score tests of `E[z|a]`, `Var(z|a)` and the third
/// standardised moment against the span's non-constant directions, each at
/// level [`AUTO_Z_CONDITIONAL_RAO_ALPHA`].
///
/// A `None` p-value means the test could not be formed — no span was supplied
/// (a CTN influence absorber owns the conditional leakage), or the span has no
/// usable direction — which is not evidence of structure.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConditionalLawEvidence {
    pub mean_p_value: Option<f64>,
    pub variance_p_value: Option<f64>,
    pub skewness_p_value: Option<f64>,
    pub alpha: f64,
}

impl ConditionalLawEvidence {
    fn fires(p_value: Option<f64>, alpha: f64) -> bool {
        p_value.is_some_and(|p| p < alpha)
    }

    /// The conditional mean or variance of the score moves on the span: a
    /// declared `N(0, 1)` law is then false in context.
    pub fn mean_or_variance_moves(&self) -> bool {
        Self::fires(self.mean_p_value, self.alpha) || Self::fires(self.variance_p_value, self.alpha)
    }

    /// Any tested moment of the conditional law moves on the span: one pooled
    /// law then misstates the law in context.
    pub fn law_moves(&self) -> bool {
        self.mean_or_variance_moves() || Self::fires(self.skewness_p_value, self.alpha)
    }

    pub(crate) fn summary(&self) -> String {
        fn p(value: Option<f64>) -> String {
            value.map_or_else(|| "untestable".to_string(), |p| format!("{p:.3e}"))
        }
        format!(
            "p(E[z|a])={} p(Var(z|a))={} p(skew(z|a))={} at level {:.1e}",
            p(self.mean_p_value),
            p(self.variance_p_value),
            p(self.skewness_p_value),
            self.alpha
        )
    }
}

/// The closed form's anchoring choice at a converged fit (gam#2926): whichever of
/// the closed form and the estimated law's own anchor is expected to be the more
/// accurate on the rows of the fit.
///
/// Per anchor `i`, the residual `r_i = Σ_k w_k Φ(a_cf,i + h_k) − π_i` under the
/// estimated law `Ĝ`, at the closed-form intercept, is `bias_i + ε_i`. `bias_i` is
/// the closed form's probability error under the true law `G`, and
/// `ε_i = (E_Ĝ − E_G)[Φ(a_cf,i + h)]` is `Ĝ`'s sampling error, with variance
/// `se_i² = Var_Ĝ(Φ(a_cf,i + h))/n_eff`. The estimated law's own anchor errs by
/// about `−(E_Ĝ − E_G)[Φ(a_emp,i + h)]`, whose variance is `se_i²` to first order,
/// so the closed form's excess risk on the anchor is `bias_i² − se_i²`. Because
/// `E[r_i²] = bias_i² + se_i²`, `r_i² − 2·se_i²` estimates it without bias, whatever
/// the correlation between anchors that share `Ĝ`. Weighted by the log-loss
/// curvature, `KL ≈ Δp²/(2π(1−π))`, and summed over the fit:
///
/// ```text
/// D̂ = Σ_i w_i (r_i² − 2·se_i²) / (π_i(1−π_i))
/// ```
///
/// `D̂` is recorded, but the decision is not its sign. On an exactly Gaussian score
/// `bias = 0`, so the residuals are `Ĝ`'s sampling error alone, `r ~ N(0, Σ)`, and
/// `T = Σ_i c_i r_i²` (`c_i = w_i/(π_i(1−π_i))`, `T` the residual energy) is the
/// weighted chi-square `Σ_k λ_k χ²_1` over the eigenvalues `λ_k` of `C^{1/2} Σ C^{1/2}`.
/// Anchors that share `Ĝ` share its error, and on one law of `M` atoms
/// `Σ_ij = Σ_m w_m (p_im − p̄_i)(p_jm − p̄_j)/n_eff`, so the `λ_k` are those of the
/// `M × M` Gram `Σ_i c_i a_i a_iᵀ`, `a_im = √(w_m/n_eff)·(p_im − p̄_i)`
/// ([`AnchorNoiseGram`]). Their sum is the noise energy. One mode carries most of
/// it, so the sign of `D̂ = T − 2 Σ_k λ_k` fired on about `P(χ²_1 > 2) ≈ 16%` of exact
/// Gaussian fits. The closed form is now kept unless `T` exceeds the null law's
/// upper [`CLOSED_FORM_CERTIFICATE_ALPHA`] quantile, read from the null tail and its
/// derived error bound ([`crate::probability::signed_weighted_chi_square_sf`]), that
/// is unless its anchoring error is resolved above the estimated law's own sampling
/// error at this `n`. The design false-fire rate is that level at every `n`, and the
/// fire is where the data show the closed form's error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClosedFormAnchorResidual {
    /// `D̂`.
    pub excess_kl: f64,
    /// `Σ_i w_i r_i² / (π_i(1−π_i))`.
    pub residual_energy: f64,
    /// `Σ_i w_i se_i² / (π_i(1−π_i))`.
    pub noise_energy: f64,
    /// Kish effective size of the scores the estimated law was compressed from.
    pub effective_n: f64,
    /// Anchors measured: positive prior weight and `π(1−π) > 0`.
    pub anchors: usize,
    /// Nodes of the estimated law.
    pub nodes: usize,
    /// `P(Σ_k λ_k χ²_1 > residual_energy)`: the residual energy's upper tail under
    /// an exactly Gaussian score, `0` below the subnormal range. `None` in a payload
    /// written before it was recorded, whose decision was the sign of `excess_kl`.
    #[serde(default)]
    pub null_p_value: Option<f64>,
    /// The relative error bound on `null_p_value`
    /// ([`crate::probability::TailProbability::relative_error`]); `1` or more where
    /// the tail is not resolved, as below the subnormal range.
    #[serde(default)]
    pub null_p_value_relative_error: Option<f64>,
    /// `(Σλ)²/Σλ²`, the null law's effective number of modes.
    #[serde(default)]
    pub null_modes: Option<f64>,
    /// The decision, [`closed_form_kept_by_null_tail`]: the closed form was kept.
    pub closed_form_chosen: bool,
}

/// The closed-form certificate's design false-fire rate (gam#2926): the probability
/// that the certificate prefers the estimated law on an exactly Gaussian score. A
/// stated policy, not a tuning knob, at the adequacy screen's level
/// [`AUTO_Z_NORMAL_SCREEN_ALPHA`].
pub const CLOSED_FORM_CERTIFICATE_ALPHA: f64 = AUTO_Z_NORMAL_SCREEN_ALPHA;

/// Whether the closed-form certificate keeps the closed form (gam#2926), from the
/// null tail `tail = P(Q > statistic)` of `Q = Σ_k λ_k χ²_1`, whose mean is
/// `mean = Σλ` and variance `variance = 2Σλ²`, or `None` where no bound decides it.
///
/// A resolved tail (`relative_error = ε < 1`) puts `P` in `[p/(1 + ε), p/(1 − ε)]`.
/// The closed form is kept when all of it is at least
/// [`CLOSED_FORM_CERTIFICATE_ALPHA`], and the certificate fires when all of it is
/// below. An unresolved tail says only `0 ≤ P ≤ 1`. That happens below the
/// subnormal range, far beyond the rate, and there Cantelli's inequality
/// `P(Q − μ ≥ s) ≤ σ²/(σ² + s²)` decides it when its bound is below the rate.
/// Everything else, a bound that straddles the rate or a tail that is not a number,
/// is undecided.
pub(crate) fn closed_form_kept_by_null_tail(
    tail: crate::probability::TailProbability,
    statistic: f64,
    mean: f64,
    variance: f64,
) -> Option<bool> {
    let alpha = CLOSED_FORM_CERTIFICATE_ALPHA;
    if tail.probability.is_nan() || tail.relative_error.is_nan() {
        return None;
    }
    if tail.relative_error < 1.0 {
        if tail.probability / (1.0 + tail.relative_error) >= alpha {
            return Some(true);
        }
        return (tail.probability / (1.0 - tail.relative_error) < alpha).then_some(false);
    }
    let excess = statistic - mean;
    (excess > 0.0 && variance / (variance + excess * excess) < alpha).then_some(false)
}

/// The Gram `Σ_i c_i a_i a_iᵀ` of the closed-form certificate's anchors on the `M`
/// atoms of one estimated law (gam#2926): `c_i = w_i/(π_i(1−π_i))` and
/// `a_im = √(w_m/n_eff)·(p_im − Σ_k w_k p_ik)`, where `p_im` is the anchor's
/// probability at atom `m`. Its eigenvalues are the weights of the certificate's
/// null law ([`ClosedFormAnchorResidual`]). Only the lower triangle is accumulated.
#[derive(Clone, Debug)]
pub(crate) struct AnchorNoiseGram {
    gram: Array2<f64>,
}

impl AnchorNoiseGram {
    pub(crate) fn new(atoms: usize) -> Self {
        Self {
            gram: Array2::zeros((atoms, atoms)),
        }
    }

    /// Add the anchor with certificate coefficient `c = w/(π(1−π))` whose
    /// probabilities at the law's atoms are `probabilities`.
    pub(crate) fn add_anchor(
        &mut self,
        coefficient: f64,
        law_weights: &[f64],
        probabilities: &[f64],
        effective_n: f64,
    ) -> Result<(), String> {
        let atoms = self.gram.nrows();
        if law_weights.len() != atoms || probabilities.len() != atoms {
            return Err(format!(
                "closed-form certificate noise Gram of {atoms} atoms read an anchor over {} \
                 weights and {} probabilities",
                law_weights.len(),
                probabilities.len()
            ));
        }
        let mean: f64 = law_weights
            .iter()
            .zip(probabilities)
            .map(|(w, p)| w * p)
            .sum();
        let centered: Vec<f64> = law_weights
            .iter()
            .zip(probabilities)
            .map(|(w, p)| (w / effective_n).sqrt() * (p - mean))
            .collect();
        for i in 0..atoms {
            let scaled = coefficient * centered[i];
            for j in 0..=i {
                self.gram[[i, j]] += scaled * centered[j];
            }
        }
        Ok(())
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        self.gram += &other.gram;
    }

    /// The null law's weights: the Gram's eigenvalues, the round-off below zero of
    /// a positive semidefinite Gram dropped.
    fn null_weights(&self) -> Result<Vec<f64>, String> {
        let atoms = self.gram.nrows();
        let mut full = self.gram.clone();
        for i in 0..atoms {
            for j in 0..i {
                full[[j, i]] = full[[i, j]];
            }
        }
        let (eigenvalues, _) = gam_linalg::faer_ndarray::FaerEigh::eigh(&full, faer::Side::Lower)
            .map_err(|error| {
                format!("closed-form certificate noise Gram eigendecomposition failed: {error:?}")
            })?;
        if let Some(eigenvalue) = eigenvalues.iter().find(|l| !l.is_finite()) {
            return Err(format!(
                "closed-form certificate noise Gram has a non-finite eigenvalue {eigenvalue}"
            ));
        }
        Ok(eigenvalues.iter().copied().filter(|&l| l > 0.0).collect())
    }

    /// The declared-Gaussian loss test ([`declared_gaussian_loss_test`]) of the
    /// residual energy `residual_energy` against this Gram's null weights.
    pub(crate) fn declared_gaussian_loss_test(
        &self,
        residual_energy: f64,
    ) -> Result<DeclaredGaussianLossTest, String> {
        declared_gaussian_loss_test(&self.null_weights()?, residual_energy)
    }
}

/// The test of a declared Gaussian law's excess anchoring loss (gam#2968), at the
/// conditional-law gate's level [`AUTO_Z_CONDITIONAL_RAO_ALPHA`].
///
/// Under the declaration each anchor's residual is `r = b + e`: `b` the Gaussian
/// anchor's bias against the true law, `e ~ N(0, Σ)` the estimated law's sampling
/// error ([`ClosedFormAnchorResidual`]). The declaration's excess loss is
/// `D = bᵀCb − N`, the Gaussian anchor's weighted squared error beyond the estimated
/// law's own, `N = tr(CΣ) = Σ_k λ_k` the noise energy, and `D̂ = T − 2N` estimates it
/// without bias. The declaration is refused when `H₀: D ≤ 0` is rejected, which
/// needs the law of `T = ‖C^{1/2}(b + e)‖²` at its least favourable null bias.
/// In the eigenbasis of `C^{1/2}ΣC^{1/2}`,
///
/// ```text
/// T = Σ_k λ_k (z_k + μ_k)² + ‖β_⊥‖²,    Σ_k λ_k μ_k² + ‖β_⊥‖² = bᵀCb ≤ N,
/// ```
///
/// `z_k` independent standard normals, `β_⊥` the part of `C^{1/2}b` outside the
/// noise's range. `T`'s tail grows with the bias energy, so the supremum over the
/// null is on its boundary `bᵀCb = N`. A budget `B` on one mode gives
/// `(√λ·z + √B)²`, and `P(|√λ·z + √B| > √t)` grows with `λ` wherever `t > B`, so the
/// top mode `λ₁` carries the heaviest tail and the bias outside the range
/// (`λ → 0`, the constant `B`) the lightest. The least-favourable law is
///
/// ```text
/// T_LF = λ₁·χ²₁(N/λ₁) + Σ_{k≥2} λ_k χ²₁,
/// ```
///
/// with mean `2N` and variance `2Σλ² + 4λ₁N`; a bias split over modes is checked
/// against it by simulation in the tests. The noncentral term is the Poisson mixture
/// `χ²₁(δ) = Σ_j Pois(j; δ/2)·χ²_{1+2j}`, so the p-value `P(T_LF > T)` is
/// `Σ_j π_j P_j`, each `P_j` a central weighted chi-square tail
/// ([`crate::probability::signed_weighted_chi_square_sf`]) with `1 + 2j` degrees of
/// freedom on `λ₁` ([`least_favourable_declaration_tail`] bounds it). Cantelli's
/// inequality on the mean and variance above is a second upper bound. The declaration
/// is refused when the upper bound is below the level and kept when the lower bound
/// is at least it, and a bound that straddles the level is an error by name.
///
/// The test has size exactly `α` at the least-favourable bias and less everywhere
/// else in the null: on an exactly Gaussian score, `b = 0`, it is far below `α`.
/// No positive weight means no anchor varies over the law's atoms, every law gives
/// the same anchors, and there is nothing to refuse.
pub(crate) fn declared_gaussian_loss_test(
    weights: &[f64],
    residual_energy: f64,
) -> Result<DeclaredGaussianLossTest, String> {
    let alpha = AUTO_Z_CONDITIONAL_RAO_ALPHA;
    let top_weight = weights.iter().copied().fold(0.0, f64::max);
    let (noise_energy, weight_sq_sum) = weights
        .iter()
        .fold((0.0, 0.0), |(s, q), &l| (s + l, q + l * l));
    if weights.is_empty() {
        return Ok(DeclaredGaussianLossTest {
            p_value_lower: 1.0,
            p_value_upper: 1.0,
            top_weight,
            noise_energy,
            refused: false,
        });
    }
    let (p_value_lower, series_upper) = least_favourable_declaration_tail(weights, residual_energy)?;
    let excess = residual_energy - 2.0 * noise_energy;
    let variance = 2.0 * weight_sq_sum + 4.0 * top_weight * noise_energy;
    let cantelli = if excess > 0.0 {
        variance / (variance + excess * excess)
    } else {
        1.0
    };
    let p_value_upper = series_upper.min(cantelli);
    let refused = if p_value_upper < alpha {
        true
    } else if p_value_lower >= alpha {
        false
    } else {
        return Err(format!(
            "declared-Gaussian loss test does not decide against the level {alpha}: the \
             least-favourable p-value lies in [{p_value_lower:.6e}, {p_value_upper:.6e}] at \
             residual energy {residual_energy}, top null weight {top_weight} of {} summing to \
             {noise_energy}",
            weights.len()
        ));
    };
    Ok(DeclaredGaussianLossTest {
        p_value_lower,
        p_value_upper,
        top_weight,
        noise_energy,
        refused,
    })
}

/// Bounds `[lower, upper]` on `P(λ₁·χ²₁(N/λ₁) + Σ_{k≥2} λ_k χ²₁ > statistic)`, `λ₁`
/// the largest of the positive `weights` and `N` their sum (gam#2968,
/// [`declared_gaussian_loss_test`]).
///
/// With Poisson mean `μ = N/(2λ₁)` the tail is `Σ_j π_j P_j`, `π_j = e^{−μ} μ^j/j!`
/// and `P_j` the central tail with `1 + 2j` degrees of freedom on `λ₁`. Each bound
/// is derived:
/// - `π_j` is formed by `π_j = π_{j−1}·μ/j` from `e^{−μ}` (the exponential within one
///   ulp), so it carries at most `2j + 2` roundings, `γ_{2j+2}` relative;
/// - a resolved `P_j` (`relative_error ε < 1`) lies in `[p/(1 + ε), p/(1 − ε)]`. An
///   unresolved one says only `0 ≤ P_j ≤ 1`, but `χ²_{1+2j}` is stochastically
///   increasing in `j`, so `P_j` lies between the last resolved lower bound before it
///   and the first resolved upper bound after it;
/// - once `J + 2 > μ` the Poisson mass beyond `J` is at most
///   `π_{J+1}/(1 − μ/(J + 2))` (the ratio of consecutive terms is `μ/(j+1)`), charged
///   at `P_j ≤ 1`.
///
/// The series stops when that remainder is no wider than the interval the evaluated
/// terms already leave, or has underflowed to zero. The two sums round by at most
/// `γ` of twice their length. `e^{−μ}` underflows only past about 1400 null modes,
/// far beyond any estimated or joint law here, and is refused by name there.
fn least_favourable_declaration_tail(weights: &[f64], statistic: f64) -> Result<(f64, f64), String> {
    use crate::probability::{WeightedChiSquareTerm, signed_weighted_chi_square_sf};
    use gam_linalg::roundoff::accumulation_growth;
    if !(statistic > 0.0) {
        return if statistic.is_nan() {
            Err("declared-Gaussian loss test read a residual energy that is not a number".to_string())
        } else {
            Ok((1.0, 1.0))
        };
    }
    let (top_index, top_weight) = weights
        .iter()
        .copied()
        .enumerate()
        .fold((0, 0.0), |best, (index, weight)| {
            if weight > best.1 { (index, weight) } else { best }
        });
    let noise_energy: f64 = weights.iter().sum();
    let mean = 0.5 * noise_energy / top_weight;
    let mut mass = (-mean).exp();
    if !(mass > 0.0) {
        return Err(format!(
            "declared-Gaussian loss test: the least-favourable law's Poisson mixture of mean \
             {mean} ({} null weights, top {top_weight} of {noise_energy}) underflows at its \
             first term",
            weights.len()
        ));
    }
    let mut terms: Vec<WeightedChiSquareTerm> = weights
        .iter()
        .map(|&weight| WeightedChiSquareTerm {
            weight,
            degrees_of_freedom: 1.0,
        })
        .collect();
    let (mut lower, mut upper) = (0.0, 0.0);
    // Upper Poisson mass of the unresolved terms since the last resolved one, and
    // that resolved term's lower bound.
    let (mut pending, mut last_lower) = (0.0, 0.0);
    let mut resolved = false;
    let mut j = 0usize;
    loop {
        terms[top_index].degrees_of_freedom = 1.0 + 2.0 * j as f64;
        let tail = signed_weighted_chi_square_sf(&terms, statistic);
        if tail.probability.is_nan() || tail.relative_error.is_nan() {
            return Err(format!(
                "declared-Gaussian loss test: the central tail with {} degrees of freedom on the \
                 top null weight is not a number at residual energy {statistic}",
                1 + 2 * j
            ));
        }
        let rounding = accumulation_growth(2 * j + 2);
        let (mass_lower, mass_upper) = (mass / (1.0 + rounding), mass * (1.0 + rounding));
        if tail.relative_error < 1.0 {
            let p_lower = tail.probability / (1.0 + tail.relative_error);
            let p_upper = (tail.probability / (1.0 - tail.relative_error)).min(1.0);
            lower += mass_lower * p_lower;
            upper += (mass_upper + pending) * p_upper;
            pending = 0.0;
            last_lower = p_lower;
            resolved = true;
        } else {
            lower += mass_lower * last_lower;
            pending += mass_upper;
        }
        let next = mass * mean / (j as f64 + 1.0);
        let ratio = mean / (j as f64 + 2.0);
        if ratio < 1.0 {
            let remainder = next * (1.0 + accumulation_growth(2 * j + 6)) / (1.0 - ratio);
            if (resolved && remainder <= upper + pending - lower) || remainder == 0.0 {
                let summed = accumulation_growth(2 * j + 2);
                return Ok((
                    (lower / (1.0 + summed)).min(1.0),
                    ((upper + pending + remainder) * (1.0 + summed)).min(1.0),
                ));
            }
        }
        mass = next;
        j += 1;
    }
}

/// The outcome of [`declared_gaussian_loss_test`] (gam#2968).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeclaredGaussianLossTest {
    /// Lower bound on the least-favourable p-value `P(T_LF > T)`.
    pub p_value_lower: f64,
    /// Upper bound on it: the Poisson-mixture series' or Cantelli's, the tighter.
    pub p_value_upper: f64,
    /// `λ₁`, the null weight the least-favourable bias sits on.
    pub top_weight: f64,
    /// `N = Σλ`, the bias energy at the null's boundary.
    pub noise_energy: f64,
    /// `H₀: D ≤ 0` is rejected at [`AUTO_Z_CONDITIONAL_RAO_ALPHA`].
    pub refused: bool,
}

impl DeclaredGaussianLossTest {
    pub fn summary(&self) -> String {
        format!(
            "P(T beyond the least-favourable loss-free declaration's, λ₁ = {:.4e} carrying all \
             of N = {:.4e}) in [{:.3e}, {:.3e}] against α = {}: {}",
            self.top_weight,
            self.noise_energy,
            self.p_value_lower,
            self.p_value_upper,
            AUTO_Z_CONDITIONAL_RAO_ALPHA,
            if self.refused {
                "the excess anchoring loss is beyond its sampling noise"
            } else {
                "the excess anchoring loss is within its sampling noise"
            }
        )
    }
}

impl ClosedFormAnchorResidual {
    /// Aggregate per-anchor `(residual, standard error, π(1−π), prior weight)`
    /// beside the anchors' noise Gram on the law's atoms. Anchors whose `π(1−π)` is
    /// zero carry no probability to anchor and are not measured, and the caller adds
    /// none of them to `noise`.
    pub(crate) fn from_rows(
        rows: &[(f64, f64, f64, f64)],
        noise: &AnchorNoiseGram,
        nodes: usize,
        effective_n: f64,
    ) -> Result<Self, String> {
        let mut residual_energy = 0.0;
        let mut noise_energy = 0.0;
        let mut anchors = 0usize;
        for &(residual, standard_error, scale, weight) in rows {
            if !(weight > 0.0 && scale > 0.0) {
                continue;
            }
            if !(residual.is_finite() && standard_error.is_finite() && scale.is_finite()) {
                return Err(format!(
                    "closed-form anchoring residual is not measurable at an anchor: \
                     residual={residual}, standard error={standard_error}, π(1−π)={scale}"
                ));
            }
            residual_energy += weight * residual * residual / scale;
            noise_energy += weight * standard_error * standard_error / scale;
            anchors += 1;
        }
        if anchors == 0 {
            return Err(
                "closed-form anchoring residual pass saw no measurable positive-weight anchor"
                    .to_string(),
            );
        }
        let excess_kl = residual_energy - 2.0 * noise_energy;
        let weights = noise.null_weights()?;
        let (sum, sum_sq) = weights
            .iter()
            .fold((0.0, 0.0), |(s, q), &l| (s + l, q + l * l));
        let terms: Vec<crate::probability::WeightedChiSquareTerm> = weights
            .iter()
            .map(|&weight| crate::probability::WeightedChiSquareTerm {
                weight,
                degrees_of_freedom: 1.0,
            })
            .collect();
        // No positive weight means no anchor's probability varies over the law's
        // atoms, so every anchor is the same under any law of the score and there is
        // nothing to prefer.
        let (null_p_value, relative_error, closed_form_chosen) = if terms.is_empty() {
            (1.0, 0.0, true)
        } else {
            let tail = crate::probability::signed_weighted_chi_square_sf(&terms, residual_energy);
            let decided = closed_form_kept_by_null_tail(tail, residual_energy, sum, 2.0 * sum_sq);
            let kept = decided.ok_or_else(|| {
                format!(
                    "closed-form certificate null tail does not decide against the design rate \
                     {CLOSED_FORM_CERTIFICATE_ALPHA}: P = {} with relative error bound {}, \
                     residual energy = {residual_energy}, {} null weights summing to {sum} \
                     (squares {sum_sq})",
                    tail.probability,
                    tail.relative_error,
                    weights.len()
                )
            })?;
            (tail.probability, tail.relative_error, kept)
        };
        Ok(Self {
            excess_kl,
            residual_energy,
            noise_energy,
            effective_n,
            anchors,
            nodes,
            null_p_value: Some(null_p_value),
            null_p_value_relative_error: Some(relative_error),
            null_modes: Some(if sum_sq > 0.0 { sum * sum / sum_sq } else { 0.0 }),
            closed_form_chosen,
        })
    }

    pub(crate) fn summary(&self) -> String {
        format!(
            "D̂ = Σ w (r² − 2·se²)/π(1−π) = {:.4e} (Σ w r²/π(1−π) = {:.4e}, Σ w se²/π(1−π) = \
             {:.4e}) over {} anchors, n_eff = {:.1}, estimated law of {} nodes; P(Σ w r²/π(1−π) \
             beyond an exactly Gaussian score's) = {} (relative error {}) over {} null modes, \
             against {}: {}",
            self.excess_kl,
            self.residual_energy,
            self.noise_energy,
            self.anchors,
            self.effective_n,
            self.nodes,
            self.null_p_value
                .map_or_else(|| "not recorded".to_string(), |p| format!("{p:.3e}")),
            self.null_p_value_relative_error
                .map_or_else(|| "not recorded".to_string(), |e| format!("{e:.1e}")),
            self.null_modes
                .map_or_else(|| "unrecorded".to_string(), |m| format!("{m:.2}")),
            CLOSED_FORM_CERTIFICATE_ALPHA,
            if self.closed_form_chosen {
                "the closed form is kept"
            } else {
                "the estimated law is the more accurate anchor"
            }
        )
    }
}

/// Rows per chunk of the closed-form certificate's pass. A fixed size, so the
/// order the noise Gram sums in, and with it every recorded bit, is the same at
/// any thread count.
const CERTIFICATE_ROW_CHUNK: usize = 256;

/// The closed-form certificate's pass over `rows` training rows (gam#2926):
/// `measure(workspace, row)` gives the row's `K` anchors as `(residual, law sd,
/// π(1−π), probabilities at the law's atoms)`. Each becomes a
/// [`ClosedFormAnchorResidual::from_rows`] row `(residual, law sd/√n_eff, π(1−π),
/// weight)`, and each measured one (positive weight and `π(1−π)`) a term of the
/// anchors' [`AnchorNoiseGram`]. `init` builds one workspace per chunk.
pub(crate) fn closed_form_certificate_pass<const K: usize, W>(
    rows: usize,
    row_weights: &[f64],
    law_weights: &[f64],
    effective_n: f64,
    init: impl Fn() -> Result<W, String> + Sync,
    measure: impl Fn(&mut W, usize) -> Result<[(f64, f64, f64, Vec<f64>); K], String> + Sync,
) -> Result<(Vec<(f64, f64, f64, f64)>, AnchorNoiseGram), String> {
    let root_n = effective_n.sqrt();
    let atoms = law_weights.len();
    let partials = (0..rows.div_ceil(CERTIFICATE_ROW_CHUNK))
        .into_par_iter()
        .map(|chunk| -> Result<(Vec<(f64, f64, f64, f64)>, AnchorNoiseGram), String> {
            let mut workspace = init()?;
            let mut noise = AnchorNoiseGram::new(atoms);
            let start = chunk * CERTIFICATE_ROW_CHUNK;
            let end = (start + CERTIFICATE_ROW_CHUNK).min(rows);
            let mut measured = Vec::with_capacity((end - start) * K);
            for row in start..end {
                let weight = row_weights[row];
                for (residual, law_sd, scale, probabilities) in measure(&mut workspace, row)? {
                    if weight > 0.0 && scale > 0.0 {
                        noise.add_anchor(weight / scale, law_weights, &probabilities, effective_n)?;
                    }
                    measured.push((residual, law_sd / root_n, scale, weight));
                }
            }
            Ok((measured, noise))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut measured = Vec::with_capacity(rows * K);
    let mut noise = AnchorNoiseGram::new(atoms);
    for (chunk, gram) in partials {
        measured.extend(chunk);
        noise.merge(&gram);
    }
    Ok((measured, noise))
}

/// The law of the latent score a marginal-slope fit consumed, and how it came
/// to consume it (gam#2926). Persisted with the model beside
/// `latent_measure`, which carries the law itself; this records what the law
/// IS — an estimate, a declaration, or the Gaussian closed form — so a reader
/// of a saved model never has to infer it from the shape of the grid.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "law", rename_all = "kebab-case")]
pub enum LatentLawConsumed {
    /// Default, span without structure, a score the standard-normal adequacy
    /// screen cannot tell from `N(0, 1)`, and a converged closed-form fit whose
    /// excess-KL certificate ([`ClosedFormAnchorResidual`]) keeps the closed form:
    /// the closed-form Gaussian lowering, chosen by that evidence and never by
    /// transforming the score. `evidence` is the span test, `adequacy` every
    /// pooled statistic beside its bound, `residual` the certificate.
    EstimatedGaussianAdequate {
        evidence: ConditionalLawEvidence,
        adequacy: LatentNormalAdequacy,
        /// `None` only between the gate and the family's certificate pass; a
        /// model is never persisted without it.
        residual: Option<ClosedFormAnchorResidual>,
    },
    /// Default, span without structure, and a score that passes the adequacy
    /// screen, but a converged closed-form fit whose excess-KL certificate
    /// ([`ClosedFormAnchorResidual`]) says the estimated law is the more accurate
    /// anchor: the fit is re-solved on that estimated law, warm-started from the
    /// closed form.
    EstimatedGlobalByResidual {
        evidence: ConditionalLawEvidence,
        adequacy: LatentNormalAdequacy,
        residual: ClosedFormAnchorResidual,
    },
    /// Default on a configuration whose kernel evaluates only the closed form, where
    /// no certified choice exists yet: the score's law departs or moves and the
    /// kernel cannot anchor on a finite law, the anchoring residual cannot yet be
    /// evaluated on the configuration, or the estimated law is the more accurate
    /// anchor and nothing can re-solve on it. The closed-form lowering on the score as
    /// given, as the default fitted before gam#2926, recorded with what is missing and
    /// warned about, never as a certified closed form.
    GaussianUncertified {
        evidence: ConditionalLawEvidence,
        adequacy: Option<LatentNormalAdequacy>,
        certificate: Option<ClosedFormAnchorResidual>,
        missing: String,
    },
    /// Default, span without structure, and a score that fails the adequacy
    /// check: one finite law estimated from the training score on its own axis.
    EstimatedGlobal { evidence: ConditionalLawEvidence },
    /// Default, span with structure: the arm the moving-law certificate
    /// ([`MovingLawCertificate`]) chose at the converged fit, the simplest of the
    /// Gaussian, location-scale and local laws whose cross-fitted excess anchoring
    /// loss is within one paired standard error of the lowest. `contexts` is the
    /// local arm's context count, whichever arm was chosen.
    EstimatedMovingLaw {
        evidence: ConditionalLawEvidence,
        arm: MovingLawArm,
        contexts: usize,
        /// `None` between the gate and the family's certificate pass, and where no
        /// certificate can be taken, which `uncertified` then names; a model is
        /// never persisted with neither.
        certificate: Option<MovingLawCertificate>,
        /// Why no certificate was taken: the anchor is one the certificate does
        /// not evaluate. `arm` is then the arm the default fits first.
        #[serde(default)]
        uncertified: Option<String>,
    },
    /// `latent_measure = "global-empirical"`: the pooled law of the score,
    /// requested whatever the span shows.
    RequestedGlobalEmpirical,
    /// `declared_latent_law`: exactly the caller's finite law.
    DeclaredFiniteLaw { nodes: usize },
    /// `latent_measure = "conditional-location-scale"`: `m(a)`, `v(a)` fitted
    /// on the span and the standardised residual's empirical law.
    /// `calibrated = false` when neither moment moves, so the score reached the
    /// kernel unchanged on the pooled law.
    ConditionalLocationScale {
        calibrated: bool,
        evidence: ConditionalLawEvidence,
    },
    /// `latent_measure = "gaussian"`, `frozen_score`, or the CTN chain: the
    /// closed-form `N(0, 1)` lowering, admitted because the score's conditional
    /// moments do not move on the span. When the pooled score fails the adequacy
    /// screen, `adequacy` is the failing ledger and `residual` the declaration's
    /// estimated excess anchoring loss at the converged fit, both warned about. A
    /// failed screen alone does not refuse the declaration; a loss beyond its
    /// sampling noise does ([`declared_gaussian_loss_test`],
    /// [`LatentLawRefusal::DeclaredGaussianAnchoringLoss`], gam#2968), so a
    /// persisted declaration's loss is within it.
    DeclaredGaussian {
        evidence: ConditionalLawEvidence,
        adequacy: Option<LatentNormalAdequacy>,
        /// `None` when the screen passed, or between the gate and the family's
        /// certificate pass, or where the measurement cannot be taken, which
        /// `uncertified` then names; a model whose screen failed is never persisted
        /// with neither.
        residual: Option<ClosedFormAnchorResidual>,
        /// Why the excess anchoring loss of a failing declaration was not measured.
        #[serde(default)]
        uncertified: Option<String>,
    },
}

impl LatentLawConsumed {
    /// The stable spelling a report or log names this law by.
    pub fn label(&self) -> &'static str {
        match self {
            Self::EstimatedGaussianAdequate { .. } => "estimated-gaussian-adequate",
            Self::EstimatedGlobalByResidual { .. } => "estimated-global-by-residual",
            Self::GaussianUncertified { .. } => "gaussian-uncertified",
            Self::EstimatedGlobal { .. } => "estimated-global",
            Self::EstimatedMovingLaw { arm, .. } => match arm {
                MovingLawArm::Gaussian => "estimated-gaussian",
                MovingLawArm::PooledEmpirical => "estimated-pooled-empirical",
                MovingLawArm::LocationScaleGaussian => "estimated-location-scale-gaussian",
                MovingLawArm::LocationScaleEmpirical => "estimated-location-scale-empirical",
                MovingLawArm::Local => "estimated-local",
            },
            Self::RequestedGlobalEmpirical => "requested-global-empirical",
            Self::DeclaredFiniteLaw { .. } => "declared-finite-law",
            Self::ConditionalLocationScale { .. } => "conditional-location-scale",
            Self::DeclaredGaussian { .. } => "declared-gaussian",
        }
    }

    /// The reason no certificate was taken, when the fit recorded one instead: a
    /// default whose anchor the certificate cannot evaluate. `None` for every
    /// certified or declared law, and for a provisional record.
    pub fn uncertified_reason(&self) -> Option<&str> {
        match self {
            Self::GaussianUncertified { missing, .. } => Some(missing),
            Self::EstimatedMovingLaw { uncertified, .. }
            | Self::DeclaredGaussian { uncertified, .. } => uncertified.as_deref(),
            Self::EstimatedGaussianAdequate { .. }
            | Self::EstimatedGlobalByResidual { .. }
            | Self::EstimatedGlobal { .. }
            | Self::RequestedGlobalEmpirical
            | Self::DeclaredFiniteLaw { .. }
            | Self::ConditionalLocationScale { .. } => None,
        }
    }

    /// Refuse a fit whose law carries no certificate where one is due (gam#2926):
    /// a provisional closed form, a failing Gaussian declaration never measured, a
    /// moving law never certified, or a default recorded uncertified. A recorded
    /// reason is named in the refusal, never taken in place of the certificate.
    pub fn require_certified(&self, context: &str) -> Result<(), String> {
        self.require_recorded(context)?;
        match self.uncertified_reason() {
            Some(reason) => Err(format!(
                "{context}: the {} law carries no certificate: {reason}",
                self.label()
            )),
            None => Ok(()),
        }
    }

    /// Refuse a provisional record, whose certificate was due and never taken and
    /// which states no reason: the gate a saved model passes (gam#2926). A default
    /// the certificate cannot evaluate records why and is saved, uncertified.
    pub(crate) fn require_recorded(&self, context: &str) -> Result<(), String> {
        if let Self::EstimatedGaussianAdequate { residual: None, .. } = self {
            return Err(format!(
                "{context}: the closed form was chosen by the adequacy screen, and its anchoring \
                 residual was never certified at the converged fit"
            ));
        }
        if let Self::DeclaredGaussian {
            adequacy: Some(_),
            residual: None,
            uncertified: None,
            ..
        } = self
        {
            return Err(format!(
                "{context}: the declared Gaussian law failed the adequacy screen, and its excess \
                 anchoring loss was never measured at the converged fit"
            ));
        }
        if let Self::EstimatedMovingLaw {
            certificate: None,
            uncertified: None,
            arm,
            ..
        } = self
        {
            return Err(format!(
                "{context}: the {} law was fitted for a score whose law moves on the span, and \
                 the moving-law certificate was never taken at the converged fit",
                arm.label()
            ));
        }
        Ok(())
    }
}

/// Why a marginal-slope fit refused the latent law it was asked for, or could
/// not reach the default one (gam#2926). Rendered into the fit's error; the
/// variants are what a caller can act on.
#[derive(Clone, Debug, PartialEq)]
pub enum LatentLawRefusal {
    /// A Gaussian law was declared, but the score's conditional mean or
    /// variance moves on the marginal-index span.
    GaussianConditionalMomentsMove {
        context: String,
        evidence: ConditionalLawEvidence,
    },
    /// The requested law needs the empirical anchoring kernel, and this
    /// configuration's kernel is the closed-form Gaussian lowering only.
    EmpiricalKernelUnavailable { context: String, requested: String },
    /// The conditional law moves on the span, so the default law is local by
    /// context, and this caller supplied no context to estimate it on.
    LocalLawContextUnavailable {
        context: String,
        evidence: ConditionalLawEvidence,
    },
    /// `conditional-location-scale` needs the marginal-index span, and none was
    /// available.
    LocationScaleSpanUnavailable { context: String },
    /// The moving-law certificate could not build an arm's law without one fold's
    /// rows at the full-data law's configuration, so it cannot score that arm.
    MovingLawFoldUnfittable {
        context: String,
        fold: usize,
        arm: MovingLawArm,
        reason: String,
    },
    /// A Gaussian law was declared on a score that fails the standard-normal
    /// adequacy screen, and at the converged declared fit the declaration's excess
    /// anchoring loss is beyond its sampling noise ([`declared_gaussian_loss_test`],
    /// gam#2968).
    DeclaredGaussianAnchoringLoss {
        context: String,
        certificate: ClosedFormAnchorResidual,
        test: DeclaredGaussianLossTest,
        adequacy: String,
    },
}

impl std::fmt::Display for LatentLawRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GaussianConditionalMomentsMove { context, evidence } => write!(
                f,
                "{context}: the Gaussian latent law was declared (latent_measure=\"gaussian\", \
                 frozen_score, or the CTN chain), but the score's conditional mean or variance \
                 moves on the marginal-index span ({}), so z | a is not N(0, 1) and the \
                 closed-form lowering would put b(a)·E[z|a] into the marginal index. Refused. \
                 Drop the declaration to anchor on the estimated conditional law (the default), \
                 or supply a score that is conditionally standard normal",
                evidence.summary()
            ),
            Self::EmpiricalKernelUnavailable { context, requested } => write!(
                f,
                "{context}: the latent law requested ({requested}) is a finite law, and this \
                 configuration's row kernel evaluates only the closed-form Gaussian lowering. \
                 Declare latent_measure=\"gaussian\" to fit the Gaussian law (the score must pass \
                 the adequacy check), or remove what confines the kernel to the closed form"
            ),
            Self::LocalLawContextUnavailable { context, evidence } => write!(
                f,
                "{context}: the score's conditional law moves on the marginal-index span ({}), so \
                 the default law is local by context, and no context columns were available to \
                 estimate it on. Declare latent_measure=\"conditional-location-scale\" to anchor \
                 on a location-scale law on the span, or latent_measure=\"global-empirical\" to \
                 anchor on the pooled law (miscalibrated where the law moves)",
                evidence.summary()
            ),
            Self::LocationScaleSpanUnavailable { context } => write!(
                f,
                "{context}: latent_measure=\"conditional-location-scale\" fits m(a) and v(a) on \
                 the marginal-index span, and no span is available here (a CTN influence absorber \
                 owns the conditional leakage). Use the default latent measure"
            ),
            Self::MovingLawFoldUnfittable {
                context,
                fold,
                arm,
                reason,
            } => write!(
                f,
                "{context}: the score's conditional law moves on the marginal-index span, and the \
                 moving-law certificate cannot build the {} law without fold {fold}'s rows at the \
                 full-data law's configuration ({reason}), so it cannot choose among the laws. \
                 Declare latent_measure=\"conditional-location-scale\" or \
                 latent_measure=\"global-empirical\"",
                arm.label()
            ),
            Self::DeclaredGaussianAnchoringLoss {
                context,
                certificate,
                test,
                adequacy,
            } => write!(
                f,
                "{context}: the Gaussian latent law was declared (latent_measure=\"gaussian\", \
                 frozen_score, or the CTN chain) on a score that fails the standard-normal \
                 adequacy screen (ledger, x = statistic / bound, x<=1 passed: {adequacy}), and at \
                 the converged declared fit the closed form's estimated excess anchoring loss \
                 is beyond its sampling noise: D-hat = {:.4e}, {} ({}). The declared anchor \
                 misstates the probabilities it anchors. Refused. Drop the declaration to anchor \
                 on the estimated law (the default), or supply a score that is standard normal",
                certificate.excess_kl,
                test.summary(),
                certificate.summary()
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmpiricalZGrid {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

impl EmpiricalZGrid {
    /// Construct a grid whose node/weight invariants (equal length ≥ 2, finite
    /// ascending nodes, finite positive weights, weights summing to 1 within
    /// 1e-8) are enforced up-front. Sorted order is part of the input contract
    /// so hot denested-cell kernels can consume contiguous buckets without a
    /// constructor-side reorder or allocation. Prefer this over building the
    /// struct literally; every code path that goes through `new` satisfies the
    /// same contract that `validate_empirical_z_grid` checks on read.
    pub fn new(nodes: Vec<f64>, weights: Vec<f64>, context: &str) -> Result<Self, String> {
        validate_empirical_z_grid(&nodes, &weights, context)?;
        Ok(Self { nodes, weights })
    }

    /// Iterate over co-indexed `(node, weight)` pairs. Use this instead of
    /// reading `.nodes`/`.weights` separately whenever a loop wants both
    /// arrays in lockstep — eliminates the chance of mismatched indexing.
    #[inline]
    pub fn pairs(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.nodes.iter().copied().zip(self.weights.iter().copied())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
#[derive(Default)]
pub enum LatentMeasureKind {
    #[default]
    StandardNormal,
    GlobalEmpirical {
        grid: EmpiricalZGrid,
    },
    LocalEmpirical {
        feature_cols: Vec<usize>,
        #[serde(default)]
        input_scales: Option<Vec<f64>>,
        centers: Vec<Vec<f64>>,
        grids: Vec<EmpiricalZGrid>,
        top_k: usize,
        bandwidth: f64,
        /// How a row's mixture weights are formed from its centre distances.
        /// Absent on laws saved before gam#2926, which keep the meaning they
        /// were fitted under.
        #[serde(default)]
        mixture: LocalLawMixture,
        #[serde(skip)]
        train_row_mixtures: Arc<Vec<Vec<(usize, f64)>>>,
    },
}

/// How a row's local-law mixture weights are formed from its distances to the
/// context centres (gam#2926). Every row's weights come from one function,
/// [`estimated_latent_law::local_empirical_mixture_for_point`], at fit and at
/// prediction alike.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LocalLawMixture {
    /// The `top_k` nearest centres with Gaussian kernel weights, renormalised:
    /// the meaning a saved local law has always had (the fit-time builder that
    /// once minted one, removed after bd1c5ac5c5, used `top_k = 4` and
    /// `bandwidth = 1.0`). A row's law jumps where the ranking at rank `top_k`
    /// swaps, because a centre enters or leaves the mixture with a nonzero
    /// weight.
    #[default]
    NearestNormalized,
    /// Kernel weights `K(d) = exp(−d²/2h²)` of the `top_k` nearest centres less
    /// the `(top_k + 1)`-th centre's value, so a centre's weight reaches zero
    /// exactly where it leaves the top `top_k`, plus the pooled law — the grid
    /// after the context grids — at the weight `floor` in units of `K(0) = 1`,
    /// renormalised. The floor keeps the normaliser positive where the
    /// `top_k + 1` nearest centres tie, so the law is continuous in the
    /// covariates everywhere. New fits mint it with `top_k = 4`, and with the
    /// bandwidth in the scaled covariates and the floor that minimise the
    /// cross-fitted CRPS of the score
    /// ([`local_law_resolution::select_local_law_resolution`], gam#3610).
    VanishingAtTruncation { floor: f64 },
}

/// How the flexible row algebra integrates a row over its latent law — the one
/// property of a law that a row kernel has to implement (gam#3000). The CPU
/// row lowering takes one branch per form, and a device kernel declares the
/// forms it transcribes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LatentIntegral {
    /// Closed-form moments of the standard normal density on each denested
    /// cubic cell.
    GaussianCellMoments,
    /// A finite weighted sum over the law's grid nodes.
    DiscreteGrid,
}

impl LatentIntegral {
    /// The capability a kernel needs to compute a model of this form.
    pub(crate) const fn capability(self) -> &'static str {
        match self {
            Self::GaussianCellMoments => "the Gaussian cell-moment latent integral",
            Self::DiscreteGrid => "the discrete-grid latent integral of an empirical latent law",
        }
    }
}

impl LatentMeasureKind {
    /// The form in which the row algebra integrates over this law.
    pub(crate) fn integral(&self) -> LatentIntegral {
        match self {
            Self::StandardNormal => LatentIntegral::GaussianCellMoments,
            Self::GlobalEmpirical { .. } | Self::LocalEmpirical { .. } => {
                LatentIntegral::DiscreteGrid
            }
        }
    }

    pub fn validate(&self, context: &str) -> Result<(), String> {
        match self {
            Self::StandardNormal => Ok(()),
            Self::GlobalEmpirical { grid } => {
                validate_empirical_z_grid(&grid.nodes, &grid.weights, context)
            }
            Self::LocalEmpirical {
                feature_cols,
                input_scales,
                centers,
                grids,
                top_k,
                bandwidth,
                mixture,
                ..
            } => {
                if feature_cols.is_empty() {
                    return Err(format!(
                        "{context} local empirical latent measure needs feature columns"
                    ));
                }
                if centers.is_empty() {
                    return Err(format!(
                        "{context} local empirical latent measure needs centers"
                    ));
                }
                let pooled_grids =
                    usize::from(matches!(mixture, LocalLawMixture::VanishingAtTruncation { .. }));
                if grids.len() != centers.len() + pooled_grids {
                    return Err(format!(
                        "{context} local empirical latent measure center/grid length mismatch: \
                         centers={}, grids={}, expected {} (a floored mixture carries the pooled \
                         law after the context grids)",
                        centers.len(),
                        grids.len(),
                        centers.len() + pooled_grids
                    ));
                }
                if let LocalLawMixture::VanishingAtTruncation { floor } = mixture
                    && !(floor.is_finite() && *floor > 0.0)
                {
                    return Err(format!(
                        "{context} local empirical latent measure pooled floor must be finite \
                         and positive, got {floor}"
                    ));
                }
                if *top_k == 0 || *top_k > centers.len() {
                    return Err(format!(
                        "{context} local empirical latent measure top_k must be in 1..={}, got {top_k}",
                        centers.len()
                    ));
                }
                if !(*bandwidth).is_finite() || *bandwidth <= 0.0 {
                    return Err(format!(
                        "{context} local empirical latent measure bandwidth must be finite and positive, got {bandwidth}"
                    ));
                }
                if let Some(scales) = input_scales.as_ref() {
                    if scales.len() != feature_cols.len() {
                        return Err(format!(
                            "{context} local empirical latent measure input scale dimension mismatch: scales={}, features={}",
                            scales.len(),
                            feature_cols.len()
                        ));
                    }
                    for (scale_idx, scale) in scales.iter().enumerate() {
                        if !(scale.is_finite() && *scale > 0.0) {
                            return Err(format!(
                                "{context} local empirical latent measure input scale {scale_idx} must be finite and positive, got {scale}"
                            ));
                        }
                    }
                }
                for (center_idx, center) in centers.iter().enumerate() {
                    if center.len() != feature_cols.len() {
                        return Err(format!(
                            "{context} local empirical latent center {center_idx} dimension mismatch: got {}, expected {}",
                            center.len(),
                            feature_cols.len()
                        ));
                    }
                    if center.iter().any(|value| !value.is_finite()) {
                        return Err(format!(
                            "{context} local empirical latent center {center_idx} has non-finite coordinates"
                        ));
                    }
                }
                for (grid_idx, grid) in grids.iter().enumerate() {
                    validate_empirical_z_grid(
                        &grid.nodes,
                        &grid.weights,
                        &format!("{context} local empirical grid {grid_idx}"),
                    )?;
                }
                Ok(())
            }
        }
    }

    /// Per-row empirical latent grid, borrowed where possible. This sits in
    /// the innermost per-row loops of every criterion/gradient/Hessian
    /// evaluation, so the global grid MUST come back as a borrow — the old
    /// `grid.clone()` here allocated two `grid_size`-length vectors per row
    /// per evaluation across the whole fit. Only the local-mixture path,
    /// which genuinely synthesizes a new grid per row, returns an owned
    /// value.
    pub(crate) fn empirical_grid_for_training_row(
        &self,
        row: usize,
    ) -> Result<Option<std::borrow::Cow<'_, EmpiricalZGrid>>, String> {
        match self {
            Self::StandardNormal => Ok(None),
            Self::GlobalEmpirical { grid } => Ok(Some(std::borrow::Cow::Borrowed(grid))),
            Self::LocalEmpirical {
                grids,
                train_row_mixtures,
                ..
            } => {
                let mixture = train_row_mixtures.get(row).ok_or_else(|| {
                    format!(
                        "local empirical latent measure is missing training mixture for row {row}"
                    )
                })?;
                Ok(Some(std::borrow::Cow::Owned(combine_empirical_grids(
                    grids, mixture,
                )?)))
            }
        }
    }

    /// The same law with every training row's mixture rebuilt from the scaled
    /// conditioning covariates `conditioning`, one row per training row
    /// (gam#2929).
    ///
    /// A local law's `train_row_mixtures` is `#[serde(skip)]`, and that is not
    /// an omission: a row's mixture is a function of its own covariates and the
    /// saved centres, so the saved law carries the function and the values are
    /// rebuilt wherever the fitted rows are replayed. This is the one rule that
    /// rebuilds them, and it composes each row's weights with
    /// [`estimated_latent_law::local_empirical_mixture_for_point`] — the same
    /// composer the fit used for its training rows (gam#2926) and prediction
    /// uses for a prediction row — so the law a row was fitted under and the law
    /// it is replayed under are one object.
    ///
    /// A law with no per-row mixtures (the standard-normal closed form, one
    /// global grid) has nothing to rebuild and is returned unchanged, so a
    /// caller does not have to know which kind it holds.
    pub fn with_rebuilt_training_mixtures(
        &self,
        conditioning: ndarray::ArrayView2<'_, f64>,
    ) -> Result<Self, String> {
        let Self::LocalEmpirical {
            feature_cols,
            input_scales,
            centers,
            grids,
            top_k,
            bandwidth,
            mixture,
            ..
        } = self
        else {
            return Ok(self.clone());
        };
        let width = centers.first().map_or(0, Vec::len);
        if conditioning.ncols() != width {
            return Err(format!(
                "local empirical latent law conditioning is {} columns wide, but its centres are \
                 {width}-dimensional; a row's mixture reads the centres' own coordinates",
                conditioning.ncols()
            ));
        }
        let train_row_mixtures = conditioning
            .rows()
            .into_iter()
            .map(|row| {
                let point = row.to_vec();
                estimated_latent_law::local_empirical_mixture_for_point(
                    &point, centers, *top_k, *bandwidth, *mixture,
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self::LocalEmpirical {
            feature_cols: feature_cols.clone(),
            input_scales: input_scales.clone(),
            centers: centers.clone(),
            grids: grids.clone(),
            top_k: *top_k,
            bandwidth: *bandwidth,
            mixture: *mixture,
            train_row_mixtures: Arc::new(train_row_mixtures),
        })
    }
}

/// Allocation-free heapsort of parallel empirical node/weight storage.
/// Used by the local-mixture constructor, whose concatenated sorted component
/// grids are not globally ordered. Moving pairs in place avoids the third
/// temporary allocation that a `Vec<(node, weight)>` canonicalization would
/// add to that per-row path.
fn sort_empirical_node_weight_pairs(nodes: &mut [f64], weights: &mut [f64]) {
    assert_eq!(
        nodes.len(),
        weights.len(),
        "empirical grid nodes and weights must remain parallel"
    );
    fn sift_down(nodes: &mut [f64], weights: &mut [f64], mut root: usize, end: usize) {
        loop {
            let mut child = 2 * root + 1;
            if child >= end {
                return;
            }
            if child + 1 < end && nodes[child].total_cmp(&nodes[child + 1]).is_lt() {
                child += 1;
            }
            if !nodes[root].total_cmp(&nodes[child]).is_lt() {
                return;
            }
            nodes.swap(root, child);
            weights.swap(root, child);
            root = child;
        }
    }

    let len = nodes.len();
    for root in (0..len / 2).rev() {
        sift_down(nodes, weights, root, len);
    }
    for end in (1..len).rev() {
        nodes.swap(0, end);
        weights.swap(0, end);
        sift_down(nodes, weights, 0, end);
    }
}

pub(crate) fn validate_empirical_z_grid(
    nodes: &[f64],
    weights: &[f64],
    context: &str,
) -> Result<(), String> {
    if nodes.len() != weights.len() {
        return Err(format!(
            "{context} empirical latent measure node/weight length mismatch: nodes={}, weights={}",
            nodes.len(),
            weights.len()
        ));
    }
    if nodes.len() < 2 {
        return Err(format!(
            "{context} empirical latent measure requires at least two nodes"
        ));
    }
    let mut total = 0.0;
    let mut previous_node = f64::NEG_INFINITY;
    for (idx, (&node, &weight)) in nodes.iter().zip(weights.iter()).enumerate() {
        if !node.is_finite() {
            return Err(format!(
                "{context} empirical latent measure node {idx} is non-finite ({node})"
            ));
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "{context} empirical latent measure weight {idx} must be finite and positive, got {weight}"
            ));
        }
        if node < previous_node {
            return Err(format!(
                "{context} empirical latent measure nodes must be sorted ascending, but node {idx} ({node}) is below node {} ({previous_node})",
                idx - 1
            ));
        }
        previous_node = node;
        total += weight;
    }
    if !(total.is_finite() && (total - 1.0).abs() <= 1e-8) {
        return Err(format!(
            "{context} empirical latent measure weights must sum to 1, got {total}"
        ));
    }
    Ok(())
}

pub(crate) fn combine_empirical_grids(
    grids: &[EmpiricalZGrid],
    mixture: &[(usize, f64)],
) -> Result<EmpiricalZGrid, String> {
    if mixture.is_empty() {
        return Err("local empirical latent measure row mixture is empty".to_string());
    }
    let mut nodes = Vec::new();
    let mut weights = Vec::new();
    for &(grid_idx, grid_weight) in mixture {
        if !(grid_weight.is_finite() && grid_weight >= 0.0) {
            return Err(format!(
                "local empirical latent mixture weight must be finite and non-negative, got {grid_weight}"
            ));
        }
        let grid = grids.get(grid_idx).ok_or_else(|| {
            format!("local empirical latent mixture references missing grid {grid_idx}")
        })?;
        // A centre whose kernel weight underflowed to zero carries no mass.
        if grid_weight > 0.0 {
            for (node, weight) in grid.pairs() {
                nodes.push(node);
                weights.push(grid_weight * weight);
            }
        }
    }
    // Sort once, then coalesce equal nodes by summing their weights: the grids
    // of a mixture share nodes (every context grid and the pooled law are
    // compressions of one score), and the combined law carries each node once.
    sort_empirical_node_weight_pairs(&mut nodes, &mut weights);
    let mut merged = 0usize;
    for idx in 0..nodes.len() {
        if merged > 0 && nodes[idx].total_cmp(&nodes[merged - 1]).is_eq() {
            weights[merged - 1] += weights[idx];
        } else {
            nodes[merged] = nodes[idx];
            weights[merged] = weights[idx];
            merged += 1;
        }
    }
    nodes.truncate(merged);
    weights.truncate(merged);
    let total = weights.iter().copied().sum::<f64>();
    if !(total.is_finite() && total > 0.0) {
        return Err(
            "local empirical latent combined grid has non-positive total weight".to_string(),
        );
    }
    for weight in &mut weights {
        *weight /= total;
    }
    validate_empirical_z_grid(&nodes, &weights, "local empirical latent combined grid")?;
    Ok(EmpiricalZGrid { nodes, weights })
}

#[derive(Clone, Debug)]
pub struct LatentZPolicy {
    pub check_mode: LatentZCheckMode,
    pub normalization: LatentZNormalizationMode,
    pub latent_measure: LatentMeasureSpec,
    pub mean_tol_multiplier: f64,
    pub sd_tol_multiplier: f64,
    pub max_abs_skew: f64,
    pub max_abs_excess_kurtosis: f64,
}

impl LatentZPolicy {
    pub(crate) fn frozen_transformation_normal() -> Self {
        // Defaults relaxed to `WarnOnly` with the same thresholds the
        // exploratory-weighted preset uses (skew ≤ 4.0, |excess kurt| ≤ 20.0).
        // Rationale: the upstream conditional transformation-normal
        // preprocessor may be fit isotropically (no per-axis κ). At large-scale
        // dimensionality (16 PCs, 15 ancestries) an isotropic fit can leave
        // the global latent-z distribution mildly heavy-tailed (skew ≈ 4,
        // excess kurt ≈ 30–40 in synthetic studies) without violating per-
        // grouping mean/variance calibration. The downstream marginal-slope
        // model still uses the latent-Gaussian probit/score-warp link; the
        // emitted warning makes the deviation visible without aborting the
        // fit. Callers that need strict enforcement can construct a custom
        // `LatentZPolicy` with `check_mode: LatentZCheckMode::Strict`.
        Self {
            check_mode: LatentZCheckMode::WarnOnly,
            normalization: LatentZNormalizationMode::Frozen { mean: 0.0, sd: 1.0 },
            latent_measure: LatentMeasureSpec::auto_default(),
            mean_tol_multiplier: 4.0,
            sd_tol_multiplier: 4.0,
            max_abs_skew: 4.0,
            max_abs_excess_kurtosis: 20.0,
        }
    }
}

impl Default for LatentZPolicy {
    fn default() -> Self {
        Self::frozen_transformation_normal()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatentZNormalization {
    pub mean: f64,
    pub sd: f64,
}

impl LatentZNormalization {
    pub fn apply(&self, z: &Array1<f64>, context: &str) -> Result<Array1<f64>, String> {
        if !(self.mean.is_finite() && self.sd.is_finite() && self.sd > 0.0) {
            return Err(format!(
                "{context} requires finite latent z normalization with sd > 0; got mean={} sd={}",
                self.mean, self.sd
            ));
        }
        if z.iter().any(|value| !value.is_finite()) {
            return Err(format!("{context} requires finite z values"));
        }
        Ok(z.mapv(|zi| (zi - self.mean) / self.sd))
    }
}

/// Weighted mid-distribution rank inverse-normal transform for the
/// latent score.
///
/// When the latent z fails the standard-normal auto-detection
/// (`latent_z_normal_adequacy`), the BMS family applied to
/// pretend the score is N(0,1) anyway would distort the closed-form
/// probit log-CDF kernel. The historical fallback (local- or
/// global-empirical latent measure) is *mathematically correct* but
/// triggers the per-row intercept Newton solve in the empirical-grid
/// closed-form kernels (`empirical_rigid_primary_grad_hess_closed_form`
/// and its higher-order siblings); at large scale that is the dominant
/// cost.
///
/// **Rank-INT is a modeling choice, not a reparameterisation.** The
/// rigid BMS predictor is *affine* in the latent score
/// (`η = q·√(1+b²) + b·z`), so a nonlinear monotone map of `z` changes
/// the model class and its likelihood — it does not leave them
/// invariant. Applying the calibration redefines the latent axis: the
/// affine model is *specified on the calibrated score* `T(z)`. Nor is
/// the calibrated training sample exactly N(0,1): a finite set of
/// normal scores is discrete, and with heavy ties it can stay far from
/// Gaussian. The closed-form standard-normal kernel is therefore
/// adequate only when the calibrated sample itself passes the same
/// standard-normal adequacy gate applied to raw z
/// (`latent_z_normal_adequacy`). Since gam#2926 no fit mints one: the
/// default anchors on the estimated law of the score instead of making
/// the score look Gaussian. Models saved with one still carry it, and
/// prediction applies the same monotone map to incoming z and re-routes
/// through the closed-form kernel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatentZRankIntCalibration {
    /// Sorted unique positive-mass z values seen during training, ascending.
    /// Knot table for `apply_to_training` / `apply_at_predict`. Zero-weight
    /// knots carry no probability mass and are not stored.
    pub sorted_z: Vec<f64>,
    /// Weighted mid-distribution rank `(W_before + w_knot/2) / W_total` at
    /// each `sorted_z` knot. Strictly increasing, strictly inside `(0, 1)`
    /// (each knot is bounded away from the endpoints by half its own mass),
    /// and invariant to a common rescaling of the weights.
    pub weighted_cdf: Vec<f64>,
    /// Weighted mean of the calibrated training sample. Used as a
    /// sanity-check value on `fit`; should be very close to zero.
    pub post_mean: f64,
    /// Weighted SD of the calibrated training sample. Used as a
    /// sanity-check value on `fit`; should be very close to one.
    pub post_sd: f64,
}

impl LatentZRankIntCalibration {
    /// Fit the weighted rank-INT calibration from training z and weights.
    ///
    /// Algorithm:
    /// 1. Sort rows by ascending z and merge ties into one knot per unique
    ///    z with the tie group's total weight `w_g`; discard zero-mass
    ///    knots.
    /// 2. Weighted mid-distribution rank at each knot:
    ///    `p_g = (W_before + w_g/2) / W_total`,
    ///    with `W_before` the cumulative weight strictly below the knot.
    /// 3. Store `(sorted_z, weighted_cdf = p_g)`.
    ///
    /// The mid-rank depends only on *relative* weights (rescaling every
    /// weight by a common factor leaves every `p_g` unchanged), is strictly
    /// increasing across knots, and lies strictly inside `(0, 1)` — each
    /// knot is separated from the endpoints by half its own mass, so no
    /// ad-hoc clamp is needed and `Φ⁻¹(p_g)` is always finite.
    ///
    /// Returns the calibration plus the post-transform sample's weighted
    /// mean / SD for sanity-check logging.
    pub fn fit(z: &Array1<f64>, weights: &Array1<f64>) -> Result<Self, String> {
        if z.len() != weights.len() {
            return Err(format!(
                "rank-INT calibration: z length {} != weights length {}",
                z.len(),
                weights.len()
            ));
        }
        if z.is_empty() {
            return Err("rank-INT calibration requires at least one observation".to_string());
        }
        let w_total = weights.iter().copied().sum::<f64>();
        if !(w_total.is_finite() && w_total > 0.0) {
            return Err(format!(
                "rank-INT calibration requires positive finite total weight, got {w_total}"
            ));
        }
        for (idx, value) in z.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "rank-INT calibration: z[{idx}] = {value} not finite"
                ));
            }
        }
        for (idx, weight) in weights.iter().enumerate() {
            if !(weight.is_finite() && *weight >= 0.0) {
                return Err(format!(
                    "rank-INT calibration: weight[{idx}] = {weight} not finite/non-negative"
                ));
            }
        }
        let mut order: Vec<usize> = (0..z.len()).collect();
        order.sort_by(|&a, &b| z[a].partial_cmp(&z[b]).unwrap_or(std::cmp::Ordering::Equal));

        let mut sorted_z: Vec<f64> = Vec::with_capacity(z.len());
        let mut weighted_cdf: Vec<f64> = Vec::with_capacity(z.len());
        // Merge ties into one knot per unique z, then assign the weighted
        // mid-distribution rank p_g = (W_before + w_g/2) / W_total. This is
        // the mid-point of the tie group's probability mass, so it depends
        // only on relative weights, is strictly increasing, and sits
        // strictly inside (0, 1) without any clamp. Zero-mass tie groups
        // are not knots of the weighted empirical distribution and are
        // dropped.
        let mut cum_before = 0.0_f64;
        let mut pos = 0usize;
        while pos < order.len() {
            let zi = z[order[pos]];
            let mut w_group = 0.0_f64;
            let mut end = pos;
            while end < order.len() && z[order[end]] == zi {
                w_group += weights[order[end]];
                end += 1;
            }
            if w_group > 0.0 {
                sorted_z.push(zi);
                weighted_cdf.push((cum_before + 0.5 * w_group) / w_total);
                cum_before += w_group;
            }
            pos = end;
        }
        if sorted_z.is_empty() {
            return Err(
                "rank-INT calibration requires at least one positive-weight observation"
                    .to_string(),
            );
        }

        // Compute sanity-check post-mean and post-sd on the transformed
        // sample, weighted by the original weights.
        let mut sum_wz = 0.0_f64;
        let mut sum_w = 0.0_f64;
        for &idx in &order {
            let zi = z[idx];
            let calibrated = Self::apply_with_knots(zi, &sorted_z, &weighted_cdf);
            sum_wz += weights[idx] * calibrated;
            sum_w += weights[idx];
        }
        let post_mean = if sum_w > 0.0 { sum_wz / sum_w } else { 0.0 };
        let mut sum_w_dev = 0.0_f64;
        for &idx in &order {
            let zi = z[idx];
            let calibrated = Self::apply_with_knots(zi, &sorted_z, &weighted_cdf);
            let d = calibrated - post_mean;
            sum_w_dev += weights[idx] * d * d;
        }
        let post_sd = if sum_w > 0.0 {
            (sum_w_dev / sum_w).sqrt()
        } else {
            1.0
        };

        Ok(Self {
            sorted_z,
            weighted_cdf,
            post_mean,
            post_sd,
        })
    }

    /// Apply the calibration to the full training z vector, returning the
    /// calibrated sample. Equivalent to mapping each row's z through
    /// [`Self::apply_at_predict`], but vectorised.
    pub fn apply_to_training(&self, z: &Array1<f64>) -> Result<Array1<f64>, String> {
        if self.sorted_z.is_empty() {
            return Err("rank-INT calibration has no knots".to_string());
        }
        let mut out = Array1::<f64>::zeros(z.len());
        for (idx, &zi) in z.iter().enumerate() {
            if !zi.is_finite() {
                return Err(format!(
                    "rank-INT calibration apply: z[{idx}] = {zi} not finite"
                ));
            }
            out[idx] = self.apply_at_predict(zi);
        }
        Ok(out)
    }

    /// Apply the calibration to a single z at predict time.
    ///
    /// Linear interpolation on `(sorted_z, weighted_cdf)` to obtain
    /// `p ∈ [eps, 1 − eps]`, then `Φ⁻¹(p)` via
    /// [`standard_normal_quantile`]. Out-of-range z's clip to the
    /// boundary CDF before the quantile, so the calibration extrapolates
    /// monotonically beyond the training support.
    pub fn apply_at_predict(&self, z: f64) -> f64 {
        Self::apply_with_knots(z, &self.sorted_z, &self.weighted_cdf)
    }

    pub(crate) fn apply_with_knots(z: f64, sorted_z: &[f64], weighted_cdf: &[f64]) -> f64 {
        assert_eq!(sorted_z.len(), weighted_cdf.len());
        assert!(!sorted_z.is_empty());
        let n = sorted_z.len();
        let p = if z <= sorted_z[0] {
            weighted_cdf[0]
        } else if z >= sorted_z[n - 1] {
            weighted_cdf[n - 1]
        } else {
            // Binary search for the right knot.
            let mut lo = 0usize;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if sorted_z[mid] <= z {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let z_lo = sorted_z[lo];
            let z_hi = sorted_z[hi];
            let p_lo = weighted_cdf[lo];
            let p_hi = weighted_cdf[hi];
            if z_hi == z_lo {
                p_hi
            } else {
                let t = (z - z_lo) / (z_hi - z_lo);
                p_lo + t * (p_hi - p_lo)
            }
        };
        // Φ⁻¹(p); clip away from {0, 1} to keep the quantile finite.
        standard_normal_quantile(p).unwrap_or_else(|err| {
            let clipped = if p < 0.5 { -8.0 } else { 8.0 };
            log::trace!(
                "standard_normal_quantile({p}) failed ({err}); clipping the latent score to {clipped}"
            );
            clipped
        })
    }
}

/// The map a fit applies to the latent score before the kernel runs. Only the
/// declared conditional location-scale law has one (gam#2926); every other law
/// is anchored on the score as given. A rank inverse-normal calibration is never
/// minted by a fit any more — making the score look Gaussian is not a law — and
/// survives only as a saved-model field older models replay.
#[derive(Clone, Debug)]
pub enum LatentMeasureCalibration {
    None,
    ConditionalLocationScale(LatentZConditionalCalibration),
}

/// Conditional location-scale calibration of the latent score (#905).
///
/// The marginal-slope Auto trigger's pooled-z gate (KS / skewness / kurtosis +
/// the rank inverse-normal transform) only inspects the **marginal** law of
/// `z`. A conditional shift `E[z | C] = m(C) ≠ 0` — the allele-frequency-driven
/// grouping mean shift — passes the marginal gate while leaving `z | C`
/// off-center, so the slope contribution `b(C)·m(C)` leaks into the influence
/// channel `q`. Rank-INT provably cannot fix this: no transform `T` depending
/// only on the marginal `F_Z` can enforce `E[T(Z) | C] ≡ const` for all joint
/// laws.
///
/// The unique Fisher-orthogonal location-scale correction (for the Gaussian
/// working metric the closed-form probit kernel assumes) is
/// `ζ = (z − m(C)) / √v(C)`, where `m(C) = E[z|C]` is estimated by weighted
/// ridge regression of `z` on the marginal-index span `a(C) = [1 | X_marginal]`
/// and `v(C) = Var(z|C) = exp(γ·a(C))` by the Gaussian maximum-likelihood
/// log-linear variance fit of the mean residual on the same span (gam#4019).
/// The log link makes `v` positive at every `C` by construction: a linear
/// `v = γ·a(C)` goes non-positive on the low end of any span along which the
/// variance grows convexly, and needed a floor to stay a variance at all. The
/// corrected `ζ` is
/// conditionally centered (and homoskedastic when the variance block is
/// active) by construction, so the `b(C)·m(C)` leakage vanishes. Matching the
/// first two conditional moments does **not** by itself make `ζ` standard
/// normal (a two-point residual law survives location-scale correction
/// unchanged in shape), so the declared location-scale law always anchors `ζ`
/// on its empirical law and never on the closed form (gam#2926); only an
/// explicit Gaussian declaration reaches that kernel. Persisted so prediction
/// rebuilds `a(C)` from the
/// (reproducible) marginal design and applies the identical map to incoming
/// z.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatentZConditionalCalibration {
    /// Coefficients for the conditional mean `m(C) = β_m·[1 | a(C)]` over the
    /// basis `[1 | marginal-design row]`. Length `1 + basis_ncols` (leading
    /// entry is the intercept).
    pub mean_coeffs: Vec<f64>,
    /// Coefficients for the conditional log-variance
    /// `log v(C) = γ·[1 | a(C)]`. Length `1 + basis_ncols`, or empty when the
    /// conditional-variance block of the Rao gate was not significant
    /// (mean-only correction); then `v(C) ≡ homoskedastic_var`.
    ///
    /// The log parameterisation replaced a linear `v(C) = max(β_v·[1 | a(C)],
    /// var_floor)` stored as `var_coeffs` (gam#4019). No default: a payload that
    /// carries the old linear coefficients fails to deserialize on this field's
    /// absence rather than having them read as log-variance coefficients.
    pub log_var_coeffs: Vec<f64>,
    /// Number of marginal-design columns in the basis (excludes the leading
    /// intercept). The predict-time marginal design must present exactly this
    /// many columns.
    pub basis_ncols: usize,
    /// The homoskedastic conditional variance `v(C) ≡ Var(z | C)`, used when the
    /// Breusch-Pagan stage did not fire and `v` is therefore constant in `C`.
    ///
    /// This is the *residual* variance of the conditional-mean regression,
    /// `Σ w (z − m̂(C))² / Σ w`, and NOT the global (marginal) variance of z —
    /// which is what it used to hold, and what its old on-disk name still says
    /// (gam#2768).
    ///
    /// The distinction is the whole correction. `ζ = (z − m(C))/√v` is supposed
    /// to be conditionally standard normal, because the marginal-slope identity
    /// that makes `q` the MARGINAL index,
    /// `E_ζ[Φ(q√(1+b²) + bζ)] = Φ(q)`, holds only at `Var(ζ|C) = 1`; at
    /// `Var(ζ|C) = v` it becomes `Φ(q√(1+b²)/√(1+b²v))`, a multiplicative
    /// distortion of every marginal coefficient. Dividing by the marginal
    /// variance guaranteed `v ≠ 1`: with z standardised,
    /// `1 = Var(m(C)) + E[Var(z|C)]`, so the residual variance is `1 − R²` and
    /// is strictly below the marginal variance *whenever the gate fires at all*.
    /// The bug was therefore not a corner case — it was on every fired gate, and
    /// it grew with exactly the conditional structure the correction exists to
    /// remove. At `R² = 0.25` it left `sd(ζ) = 0.87` against the `post_sd ≈ 1`
    /// this struct documents, distorted the marginal coefficients by ~4%, and
    /// (worse) made the calibrated residual fail the standard-normal adequacy
    /// re-check on the SD clause alone at any appreciable n — sending BMS to the
    /// empirical measure and, per gam#2718, withholding the covariance.
    ///
    /// Its on-disk name is still `global_var`, the name it had before gam#2768.
    #[serde(rename = "global_var")]
    pub homoskedastic_var: f64,
    /// Weighted mean of the calibrated training sample (sanity-check, ≈ 0).
    pub post_mean: f64,
    /// Weighted SD of the calibrated training sample (sanity-check, ≈ 1).
    pub post_sd: f64,
    /// Joint first-stage (generated-regressor) sandwich covariance of
    /// `θ₁ = (mean_coeffs, variance stage)`, shape `dim θ₁ × dim θ₁` with
    /// `dim θ₁ =` [`Self::theta1_dim`]. The variance stage is `log_var_coeffs`
    /// when the Breusch-Pagan stage fired and the estimated constant
    /// `log homoskedastic_var` when it did not (gam#3030, gam#4019): both in log
    /// units, so the two stages are one parameterisation.
    ///
    /// Replaced a stored PAIR of per-stage sandwiches that
    /// [`Self::theta1_covariance`] assembled block-diagonally -- i.e. that
    /// asserted `Cov(mean_coeffs, variance stage) = 0`. The stages are not
    /// independent: stage B fits the variance of the MEAN residual on the same
    /// basis, so the stacked bread is block lower-triangular and the meat has a
    /// cross-block proportional to the residual's third moment. Both vanish
    /// under a Gaussian residual, and neither vanishes on the branch this
    /// covariance serves (gam#2484). Built as `ΨᵀΨ` from
    /// [`stacked_first_stage_row_influence`]; the two retired fields were its
    /// diagonal blocks.
    ///
    /// Fit-time only: predict applies the map from `mean_coeffs`/`log_var_coeffs`
    /// and never reads their uncertainty. Verified rather than assumed -- the
    /// only production consumer of this matrix is the Murphy-Topel assembly in
    /// `block_specs.rs`, which runs at fit.
    ///
    /// `#[serde(default)]` so a model saved before gam#2484 -- carrying the two
    /// retired per-stage blocks and no joint -- still deserializes on the
    /// predict path. An empty default is NOT a silent zero: it is refused at the
    /// single point of consumption (see
    /// [`Self::generated_regressor_correction`]), because a covariance that
    /// quietly vanishes is exactly the failure this issue exists to prevent.
    #[serde(default)]
    pub theta1_cov: Array2<f64>,
}

impl LatentZConditionalCalibration {
    #[inline]
    pub(crate) fn affine(coeffs: &[f64], a_row: ArrayView1<'_, f64>) -> f64 {
        let mut acc = coeffs[0];
        for (c, &x) in coeffs[1..].iter().zip(a_row.iter()) {
            acc += c * x;
        }
        acc
    }

    pub(crate) fn conditional_mean(&self, a_row: ArrayView1<'_, f64>) -> f64 {
        Self::affine(&self.mean_coeffs, a_row)
    }

    pub(crate) fn conditional_var(&self, a_row: ArrayView1<'_, f64>) -> f64 {
        if self.log_var_coeffs.is_empty() {
            self.homoskedastic_var
        } else {
            Self::affine(&self.log_var_coeffs, a_row).exp()
        }
    }

    /// Apply `ζ = (z − m(C))/√v(C)` to a batch. `a_block` is the marginal
    /// design (`n × basis_ncols`); `z` is the (normalized) latent score. Used
    /// at both training and predict time, so the map is identical. Crate-private:
    /// every reader goes through the fitted latent score map
    /// (`FittedLatentScoreMap`, gam#3016), outside the crate through
    /// `FittedModel::fitted_latent_score`.
    pub(crate) fn apply(
        &self,
        z: ArrayView1<'_, f64>,
        a_block: ArrayView2<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        if a_block.ncols() != self.basis_ncols {
            return Err(format!(
                "conditional latent calibration expects {} basis columns, got {}",
                self.basis_ncols,
                a_block.ncols()
            ));
        }
        if a_block.nrows() != z.len() {
            return Err(format!(
                "conditional latent calibration row mismatch: z={}, basis rows={}",
                z.len(),
                a_block.nrows()
            ));
        }
        if self.mean_coeffs.len() != self.basis_ncols + 1 {
            return Err(format!(
                "conditional latent calibration mean coefficient length {} != basis_ncols+1 ({})",
                self.mean_coeffs.len(),
                self.basis_ncols + 1
            ));
        }
        let mut out = Array1::<f64>::zeros(z.len());
        for i in 0..z.len() {
            let a_row = a_block.row(i);
            if !z[i].is_finite() {
                return Err(format!(
                    "conditional latent calibration: z[{i}] = {} not finite",
                    z[i]
                ));
            }
            let m = self.conditional_mean(a_row);
            let v = self.conditional_var(a_row);
            if !(v.is_finite() && v > 0.0) {
                return Err(format!(
                    "conditional latent calibration produced non-positive variance {v} at row {i}"
                ));
            }
            let zeta = (z[i] - m) / v.sqrt();
            if !zeta.is_finite() {
                return Err(format!(
                    "conditional latent calibration produced non-finite zeta at row {i}"
                ));
            }
            out[i] = zeta;
        }
        Ok(out)
    }

    /// Dimension of the first-stage parameter vector `θ₁ = (mean_coeffs,
    /// variance stage)` whose estimation uncertainty the generated-regressor
    /// correction propagates: `len(mean_coeffs) + len(log_var_coeffs)` when the
    /// Breusch-Pagan stage fired, otherwise `len(mean_coeffs) + 1` -- the
    /// constant `log homoskedastic_var` is estimated too (gam#3030).
    pub fn theta1_dim(&self) -> usize {
        self.mean_coeffs.len() + self.variance_stage_dim()
    }

    /// Width of the variance stage in `θ₁`: `len(log_var_coeffs)`, or 1 for the
    /// estimated constant `log homoskedastic_var`.
    fn variance_stage_dim(&self) -> usize {
        if self.log_var_coeffs.is_empty() {
            1
        } else {
            self.log_var_coeffs.len()
        }
    }

    /// Per-row sensitivity `∂ζ_i/∂θ₁` of the calibrated score to the first-stage
    /// calibration parameters, stacked as `[∂ζ/∂mean_coeffs | ∂ζ/∂variance]`
    /// (length [`Self::theta1_dim`]). With `ζ = (z − m(C))/√v(C)`,
    /// `A_i = [1 | a(C_i)]`, `m = A_iᵀ·mean_coeffs`:
    ///
    ///   `∂ζ/∂m = −1/√v`,  `∂ζ/∂log v = −(z − m)/(2√v) = −ζ/2`,
    ///
    /// and by the chain rule `∂ζ/∂mean_coeffs = (∂ζ/∂m)·A_i`. The variance
    /// block is `(∂ζ/∂log v)·A_i` for the fitted `log v = A_iᵀ·log_var_coeffs`,
    /// and the single entry `∂ζ/∂log v` for the constant
    /// `log v = log homoskedastic_var` (gam#3030, gam#4019). The applied map is
    /// smooth in every parameter at every row, so no row is exempt. `z` is the
    /// (normalized) raw latent score at this row.
    pub fn zeta_theta1_jacobian_row(&self, z: f64, a_row: ArrayView1<'_, f64>) -> Vec<f64> {
        let m = self.conditional_mean(a_row);
        let v = self.conditional_var(a_row);
        let inv_sqrt_v = 1.0 / v.sqrt();
        // Intercept-augmented basis row A_i = [1 | a(C_i)].
        let mut out = Vec::with_capacity(self.theta1_dim());
        let dzeta_dm = -inv_sqrt_v;
        out.push(dzeta_dm); // intercept column of A
        for &x in a_row.iter() {
            out.push(dzeta_dm * x);
        }
        let dzeta_dlog_v = -0.5 * (z - m) * inv_sqrt_v;
        out.push(dzeta_dlog_v);
        if !self.log_var_coeffs.is_empty() {
            for &x in a_row.iter() {
                out.push(dzeta_dlog_v * x);
            }
        }
        out
    }

    /// Joint first-stage covariance `V₁` of `θ₁`, ordered to match
    /// [`Self::zeta_theta1_jacobian_row`]: the stacked sandwich `ΨᵀΨ` of
    /// [`stacked_first_stage_row_influence`], which is NOT block-diagonal off a
    /// Gaussian residual (gam#2484).
    pub fn theta1_covariance(&self) -> Array2<f64> {
        self.theta1_cov.clone()
    }

    /// Murphy–Topel generated-regressor correction term for the second-stage
    /// slope covariance. Given the second-stage information `H_β` (the penalized
    /// joint Hessian of the slope fit, whose inverse is the naive `V_β`) and the
    /// cross-derivative `G = ∂(score_β)/∂θ₁` (`p_β × dim θ₁`), the corrected
    /// covariance is
    ///
    ///   `V_β = V_β^naive + (H_β⁻¹ G) V₁ (H_β⁻¹ G)ᵀ`.
    ///
    /// This returns the additive rank-`dim θ₁` term `(H_β⁻¹ G) V₁ (H_β⁻¹ G)ᵀ`
    /// given the already-formed `hbeta_inv_g = H_β⁻¹ G` (`p_β × dim θ₁`). The
    /// caller forms `G` by accumulating the per-row slope-score sensitivity to
    /// `ζ_i` times [`Self::zeta_theta1_jacobian_row`] (chain rule
    /// `∂score_β/∂θ₁ = Σ_i (∂score_β/∂ζ_i) (∂ζ_i/∂θ₁)`).
    pub fn generated_regressor_term(&self, hbeta_inv_g: ArrayView2<'_, f64>) -> Array2<f64> {
        let v1 = self.theta1_covariance();
        hbeta_inv_g.dot(&v1).dot(&hbeta_inv_g.t())
    }

    /// Assemble the full Murphy–Topel generated-regressor correction
    /// `(Vb·G)·V₁·(Vb·G)ᵀ` for the second-stage slope covariance, given the ONE
    /// engine-side quantity it cannot reconstruct post-fit: the per-row
    /// reduced-frame slope-score sensitivity to the calibrated score,
    /// `s_i = ∂score_β,i/∂ζ_i` (a `p_β`-vector in the joint flat-β reduced frame
    /// `solved_fit.beta_covariance()` lives in). With `score_β,i = ∂ℓ_i/∂β`,
    /// `s_i = ∂²ℓ_i/∂β∂ζ_i = J_iᵀ·(∂²ℓ_i/∂η_i∂ζ_i)` is the mixed `(β, ζ)`
    /// second derivative of the warped row kernel contracted through the slope
    /// design Jacobian `J_i` — exactly the #932 `RowProgram` z-jet
    /// channel (`z` is already a row-program input; one extra mixed `(β, z)` jet
    /// channel reads off `∂²ℓ/∂β∂z`). It must be evaluated at the converged `β̂`
    /// in the SAME reduced frame as `vb`.
    ///
    /// Everything else is built here from the stored first-stage quantities and
    /// the second-stage fit, dissolving the post-fit-reconstruction blocker:
    ///   - `G = Σ_i s_i · (∂ζ_i/∂θ₁)ᵀ` (`p_β × dim θ₁`), the chain-rule outer
    ///     product accumulated row-by-row with `∂ζ_i/∂θ₁ =
    ///     `[`Self::zeta_theta1_jacobian_row`]`(z_i, a_row_i)`;
    ///   - `Vb·G = vb·G` since the naive second-stage covariance `vb` IS
    ///     `H_β⁻¹` (the coordinator's `H_β⁻¹ G = Vb.dot(G)`);
    ///   - the term `(Vb·G)·V₁·(Vb·G)ᵀ` via [`Self::generated_regressor_term`].
    ///
    /// `score_zeta_sensitivity` is `n × p_β` (row `i` = `s_i`); `z` is the
    /// per-row normalized latent score (`n`); `a_block` is the marginal design
    /// `n × basis_ncols` whose rows feed `zeta_theta1_jacobian_row`; `vb` is the
    /// naive reduced-frame slope covariance `n_β × n_β`. The returned term is
    /// PSD (a congruence of the PSD `V₁`), so adding it to `vb` makes the
    /// corrected slope SE strictly ≥ the naive SE whenever the gate fires
    /// (`G ≠ 0`) and exactly equal when no row's slope score responds to `ζ`
    /// (`G = 0`).
    pub fn generated_regressor_correction(
        &self,
        score_zeta_sensitivity: ArrayView2<'_, f64>,
        z: ArrayView1<'_, f64>,
        a_block: ArrayView2<'_, f64>,
        vb: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        self.check_generated_regressor_inputs(score_zeta_sensitivity, z, a_block, vb)?;
        // G = Σ_i s_i ⊗ (∂ζ_i/∂θ₁)  (p_β × dim θ₁). Each row contributes the
        // rank-1 outer product `s_i ⊗ J_zeta_i`, so summed over the n rows this
        // is exactly the cross product `G = Sᵀ·J` of the score-sensitivity
        // matrix `S` (`n × p_β`, supplied) and the per-row ζ-Jacobian matrix
        // `J` (`n × dim θ₁`). Forming `J` row-by-row is O(n·dim θ₁); the cross
        // product is then a single BLAS-3 GEMM rather than the O(n·p_β·dim θ₁)
        // scalar triple loop (≈1.5e9 FMA at biobank scale, n≈194k, the dominant
        // ~13s/disease cost of the SE correction). Floored rows yield an exact
        // all-zero `J` row, so they contribute zero to the GEMM — bit-identical
        // to skipping them, no approximation.
        let j_mat = self.build_zeta_theta1_jacobian(z, a_block);
        let vb_g = self.beta_theta1_sensitivity(score_zeta_sensitivity, j_mat.view(), vb)?;
        Ok(self.generated_regressor_term(vb_g.view()))
    }

    /// The generated-regressor correction when the second-stage latent measure
    /// is ITSELF estimated from the calibrated score, as the global-empirical
    /// grid is (gam#3452).
    ///
    /// [`Self::generated_regressor_correction`] propagates `δθ₁` alone: the rows
    /// move with `θ₁`, and the grid moves with them through `D` (which
    /// `score_zeta_sensitivity` already carries). The grid has a sampling error
    /// of its own, though. At the true `θ₁` it is the equal-mass compression of
    /// the EMPIRICAL law of `ζ`, not of its population law, and the second-stage
    /// estimand is defined at the population grid. That error is a function of
    /// the same rows as `δθ₁`, so the two cannot be added as independent
    /// covariances; they are summed row by row and the Gram is taken once:
    ///
    /// ```text
    /// φ_i = G·ψ_i + g_i ,   Cov(β̂) ⊇ Vb·(Σ_i φ_i φ_iᵀ)·Vb
    /// ```
    ///
    /// with `ψ_i` the stacked first-stage row influence
    /// ([`Self::theta1_row_influence`]), `G = Sᵀ·J` as in the closed form, and
    /// `g_i` (`measure_influence`, `n × p_β`) the score's response to row `i`'s
    /// influence on the grid, from
    /// [`empirical_measure_sensitivity::EmpiricalZGridBuild::node_sampling_influence`].
    /// With `g = 0` this is exactly `(Vb·G)·V₁·(Vb·G)ᵀ`. The cross term with
    /// the second-stage score is zero: that score has conditional mean zero
    /// given the rows' covariates and scores at the true parameters.
    ///
    /// `weights` are the prior weights the calibration was fitted with.
    pub(crate) fn generated_regressor_correction_with_measure_influence(
        &self,
        score_zeta_sensitivity: ArrayView2<'_, f64>,
        z: ArrayView1<'_, f64>,
        a_block: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        vb: ArrayView2<'_, f64>,
        measure_influence: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        self.check_generated_regressor_inputs(score_zeta_sensitivity, z, a_block, vb)?;
        if measure_influence.dim() != score_zeta_sensitivity.dim() {
            return Err(format!(
                "generated_regressor_correction: the measure influence is {:?}, the score \
                 sensitivity {:?}",
                measure_influence.dim(),
                score_zeta_sensitivity.dim()
            ));
        }
        let j_mat = self.build_zeta_theta1_jacobian(z, a_block);
        let g = gam_linalg::faer_ndarray::fast_atb(&score_zeta_sensitivity, &j_mat);
        let psi = self.theta1_row_influence(z, a_block, weights)?;
        let mut phi = psi.dot(&g.t());
        phi += &measure_influence;
        let meat = gam_linalg::faer_ndarray::fast_ata(&phi);
        let mut correction = vb.dot(&meat).dot(&vb.t());
        gam_linalg::matrix::symmetrize_in_place(&mut correction);
        if correction.iter().any(|value| !value.is_finite()) {
            return Err("generated_regressor_correction is non-finite".to_string());
        }
        Ok(correction)
    }

    /// Per-row first-stage influence `Ψ` (`n × dim θ₁`) of this calibration on
    /// the rows it was fitted to: [`stacked_first_stage_row_influence`] at the
    /// stored coefficients, so `ΨᵀΨ` is `theta1_cov`. Recomputed rather than
    /// stored because it is `O(n)` and fit-time only.
    ///
    /// The residuals are those of the stored map: `û = z − Aᵀβ_m`, and
    /// `r = û² − Bᵀβ_v` fired or `û² − Σwû²/Σw` (the raw, unfloored constant
    /// stage) not.
    pub(crate) fn theta1_row_influence(
        &self,
        z: ArrayView1<'_, f64>,
        a_block: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let n = z.len();
        if a_block.nrows() != n || weights.len() != n || a_block.ncols() != self.basis_ncols {
            return Err(format!(
                "first-stage row influence shape mismatch: z={n}, weights={}, basis {}x{} \
                 (expected {} columns)",
                weights.len(),
                a_block.nrows(),
                a_block.ncols(),
                self.basis_ncols
            ));
        }
        let basis = build_intercept_basis(a_block);
        let (_, normal) = conditional_calibration_ridge_system(basis.view(), weights.view());
        let mean_residuals: Vec<f64> = (0..n)
            .map(|i| z[i] - self.conditional_mean(a_block.row(i)))
            .collect();
        // The `v_i` are read off this calibration's own applied map
        // (`conditional_var`), so the influence and the map it describes cannot
        // be built from different variances. The branch below chooses only the
        // basis the variance stage was fitted on; the variance score itself,
        // `r_i = û_i²/v_i − 1`, is formed inside
        // [`stacked_first_stage_row_influence`] from these `v_i` (gam#4019).
        let var_fitted: Vec<f64> = (0..n)
            .map(|i| self.conditional_var(a_block.row(i)))
            .collect();
        let constant_var_basis;
        let var_basis = if self.log_var_coeffs.is_empty() {
            // gam#3030: the constant `log v̂` is the same Gaussian likelihood on
            // the column of ones, and it is no less estimated:
            // `∂ζ/∂log v̂ = −ζ/2` is O(1), so dropping it would understate the
            // slope variance by a term of the SAME order as the mean stage's.
            constant_var_basis = Array2::<f64>::ones((n, 1));
            constant_var_basis.view()
        } else {
            // The Breusch-Pagan stage on the mean basis.
            basis.view()
        };
        stacked_first_stage_row_influence(
            basis.view(),
            var_basis,
            weights,
            &mean_residuals,
            &var_fitted,
            &normal,
        )
    }

    /// Shape checks shared by both generated-regressor corrections, including
    /// the refusal of an absent or pre-gam#3030 first-stage covariance.
    fn check_generated_regressor_inputs(
        &self,
        score_zeta_sensitivity: ArrayView2<'_, f64>,
        z: ArrayView1<'_, f64>,
        a_block: ArrayView2<'_, f64>,
        vb: ArrayView2<'_, f64>,
    ) -> Result<(), String> {
        let n = score_zeta_sensitivity.nrows();
        let p_beta = score_zeta_sensitivity.ncols();
        if z.len() != n || a_block.nrows() != n {
            return Err(format!(
                "generated_regressor_correction row mismatch: score_zeta_sensitivity rows={n}, \
                 z={}, a_block rows={}",
                z.len(),
                a_block.nrows()
            ));
        }
        if a_block.ncols() != self.basis_ncols {
            return Err(format!(
                "generated_regressor_correction expects {} basis columns, got {}",
                self.basis_ncols,
                a_block.ncols()
            ));
        }
        if vb.nrows() != p_beta || vb.ncols() != p_beta {
            return Err(format!(
                "generated_regressor_correction: vb must be {p_beta}×{p_beta}, got {}×{}",
                vb.nrows(),
                vb.ncols()
            ));
        }
        // gam#2484: refuse an absent or ill-shaped first-stage covariance rather
        // than multiplying by it. This fires for a payload written before the
        // joint covariance existed (`theta1_cov` defaults to `0×0` there); such
        // a model cannot supply the term, and silently contributing nothing
        // would understate the interval by exactly the amount the correction
        // exists to add.
        let dim_theta1 = self.theta1_dim();
        if self.theta1_cov.nrows() != dim_theta1 || self.theta1_cov.ncols() != dim_theta1 {
            return Err(format!(
                "generated_regressor_correction: the first-stage covariance is {}×{} but \
                 theta1 has dimension {dim_theta1}. A calibration deserialized from a payload \
                 written before gam#2484 carries no joint first-stage covariance, and one \
                 written before gam#3030 omits the constant variance stage; refit rather \
                 than publishing an uncorrected interval.",
                self.theta1_cov.nrows(),
                self.theta1_cov.ncols()
            ));
        }
        Ok(())
    }

    /// Per-row ζ-Jacobian matrix `J` (`n × dim θ₁`, row `i` = `∂ζ_i/∂θ₁`) built
    /// row-by-row from [`Self::zeta_theta1_jacobian_row`].
    fn build_zeta_theta1_jacobian(
        &self,
        z: ArrayView1<'_, f64>,
        a_block: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        let n = a_block.nrows();
        let dim_theta1 = self.theta1_dim();
        let mut j_mat = Array2::<f64>::zeros((n, dim_theta1));
        for i in 0..n {
            let j_zeta_row = self.zeta_theta1_jacobian_row(z[i], a_block.row(i));
            assert_eq!(
                j_zeta_row.len(),
                dim_theta1,
                "J_zeta row width must match the first-stage hyperparameter dimension"
            );
            let mut dst = j_mat.row_mut(i);
            for (slot, jz) in dst.iter_mut().zip(j_zeta_row.into_iter()) {
                *slot = jz;
            }
        }
        j_mat
    }

    /// Signed first-order sensitivity `∂β̂/∂θ₁ = Vb·G` (`p_β × dim θ₁`) of the
    /// converged second-stage slope to the first-stage calibration parameters,
    /// the SIGNED quantity the Murphy–Topel correction is built from.
    ///
    /// `G = Sᵀ·J = Σ_i s_i ⊗ (∂ζ_i/∂θ₁)` with `s_i = ∂score_β,i/∂ζ_i` the
    /// LOG-LIKELIHOOD-score sensitivity (the sign convention #1131 fixes at the
    /// source in [`gradient_paths::rigid_standard_normal_mixed_z_sensitivity`]),
    /// and `Vb = H_β⁻¹` the NLL-Hessian inverse. Under this convention the
    /// implicit-function theorem on `∂(log L)/∂β = 0` gives
    /// `∂β̂/∂θ₁ = +H_β⁻¹·G = +Vb·G`, so the returned matrix matches the finite
    /// difference of the refit slope in θ₁ in BOTH sign and magnitude — unlike
    /// the PSD correction term [`Self::generated_regressor_correction`], which is
    /// invariant to this sign. `j_zeta` is the per-row ζ-Jacobian matrix
    /// (`n × dim θ₁`, row `i` = `∂ζ_i/∂θ₁`).
    fn beta_theta1_sensitivity(
        &self,
        score_zeta_sensitivity: ArrayView2<'_, f64>,
        j_zeta: ArrayView2<'_, f64>,
        vb: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        // G = Sᵀ·J (p_β × dim θ₁) via the SIMD/GPU-routed cross product.
        let g = gam_linalg::faer_ndarray::fast_atb(&score_zeta_sensitivity, &j_zeta);
        // Vb·G = H_β⁻¹·G (vb is the naive reduced-frame covariance the fit
        // already produced — reused, never recomputed).
        Ok(vb.dot(&g))
    }
}

/// `M⁺` for a weighted-ridge normal matrix `M = AᵀWA + λR`, in RAW coordinates,
/// computed on the Jacobi-preconditioned matrix.
///
/// Pseudo-inverse rather than inverse: identifiable directions get the usual
/// `(λ_eff)⁻¹` weight, and directions `A` does not identify are zeroed -- they
/// carry no asymptotic distribution, so `V₁` is zero there and the Murphy-Topel
/// propagation through identifiable functionals of β stays finite. The ordinary
/// inverse let an unregularized direction's `1/ε` blow the sandwich through the
/// f64 range whenever a wide marginal-index span had a near-null direction (the
/// bug behind "conditional latent calibration sandwich covariance is
/// non-finite" on wide rank-deficient duchon/spline conditioning).
///
/// Jacobi preconditioning. When the conditioning basis spans many orders of
/// magnitude -- a power-9 Duchon RBF over 16 standardized PCs produces columns
/// differing by ~30 decades -- the relative truncation tolerance is set by
/// `λ_max(M)`, so an identified small-scale direction can be dropped while a
/// near-null one is kept. Precondition by `S = diag(1/√M_jj)`: because the
/// ridge penalty diagonal is the weighted Gram diagonal itself
/// (`R_jj = (AᵀWA)_jj`), `M̃ = S·M·S` has exact unit diagonal and
/// `M̃ = C + (λ/(1+λ))·I` with `C` the basis correlation matrix, so
/// `λ_min(M̃) ≥ λ/(1+λ)` even for a fully collinear basis. That clears the
/// formation band `jacobi_scaled_normal_relative_cutoff` whenever `n·p` stays
/// below about `λ/u`, so no direction is spuriously dropped and
/// `M⁺ = S·M̃⁻¹·S = M⁻¹`. The same holds for the constant variance stage's
/// `1 × 1` normal matrix `Σ w`, where `M̃ = 1`.
///
/// Returning `M⁺` in raw coordinates loses nothing to the scaling: in a raw
/// product `(M⁺ X)_ij = Σ_k M⁺_ik X_kj` against a matrix `X` carried in the
/// dual scale, every term carries the same factor `S_ii/S_jj`, so no sum mixes
/// scales. `rows` is the number of observation rows `M` was accumulated over.
pub(crate) fn preconditioned_normal_pseudoinverse(
    normal_matrix: &Array2<f64>,
    rows: usize,
) -> Result<Array2<f64>, String> {
    let p = normal_matrix.nrows();
    if normal_matrix.ncols() != p {
        return Err(format!(
            "stacked first-stage sandwich needs a square normal matrix, got {}x{}",
            normal_matrix.nrows(),
            normal_matrix.ncols()
        ));
    }
    let mut m_sym = normal_matrix.clone();
    gam_linalg::matrix::symmetrize_in_place(&mut m_sym);
    let scale: Vec<f64> = (0..p)
        .map(|j| 1.0 / m_sym[[j, j]].max(f64::MIN_POSITIVE).sqrt())
        .collect();
    let mut m_scaled = m_sym;
    for i in 0..p {
        for j in 0..p {
            m_scaled[[i, j]] *= scale[i] * scale[j];
        }
    }
    // Symmetrized, then scaled by the commutative `scale[i]·scale[j]`: mirrored.
    let mut pinv = gam_linalg::utils::rank_certified_psd_pseudoinverse(
        &m_scaled,
        gam_linalg::roundoff::SymmetricAssembly::Mirrored,
        jacobi_scaled_normal_relative_cutoff(rows, p),
    )
    .map_err(|e| format!("stacked first-stage sandwich pseudo-inverse failed: {e}"))?
    .into_pseudoinverse();
    for i in 0..p {
        for j in 0..p {
            pinv[[i, j]] *= scale[i] * scale[j];
        }
    }
    Ok(pinv)
}

/// Relative eigenvalue cutoff for the Jacobi-scaled weighted-ridge normal matrix
/// `M̃ = S·(AᵀWA + λR)·S`, `S = diag(1/√M_jj)`: its formation band against
/// `λ_max(M̃)`.
///
/// Each entry of `M` sums `rows` products `a_ij·(w_i·a_ik)`, two roundings each,
/// and the scaling adds four more. With non-negative weights that gives
/// `|E_jk| ≤ γ_{rows+6}·√(M̃_jj·M̃_kk) ≤ γ_{rows+6}` and
/// `‖E‖₂ ≤ ‖E‖_F ≤ p·γ_{rows+6}`, while `λ_max(M̃) ≥ max_j M̃_jj = 1`. The
/// `p·ε` term is the eigensolver's backward error. An eigenvalue at or below
/// the cutoff is not resolved from rounding, so its direction is not identified.
fn jacobi_scaled_normal_relative_cutoff(rows: usize, p: usize) -> f64 {
    gam_linalg::roundoff::accumulation_growth(rows + 6) * p as f64 + p as f64 * f64::EPSILON
}

/// Inverse bread `J⁻¹` of the STACKED first-stage estimating system of
/// `θ₁ = (β_m, γ)`, the conditional mean then the conditional log-variance
/// (gam#2484, gam#3047, gam#3030, gam#4019).
///
/// With `A` the mean basis `[1 | a(C)]`, `B` the variance basis, `W = diag(w)`,
/// `û_i = z_i − A_iᵀβ_m` and `v_i = exp(B_iᵀγ)`, the stages solve
///
/// ```text
/// ψ^m = Σ_i w_i A_i û_i − λR_m β_m = 0      ψ^v = Σ_i w_i B_i (û_i²/v_i − 1) = 0
/// ```
///
/// the ridge-stabilised mean regression and the Gaussian log-linear variance
/// score, whose root
/// [`conditional_score_covariance::fisher_score_log_linear_variance`] returns (the ridge of
/// its scoring step damps the step and never moves the root, so `ψ^v` carries
/// no penalty). With `M = AᵀWA + λR_m`, the observed variance information
/// `N = −∂ψ^v/∂γ = Σ_i w_i (û_i²/v_i) B_i B_iᵀ` and
/// `M_vm = ∂ψ^v/∂β_m = −2·Σ_i w_i (û_i/v_i) B_i A_iᵀ`, the bread is
///
/// ```text
/// J = −∂ψ/∂θ₁ = [  M      0 ]        J⁻¹ = [ M⁻¹   0   ]   K = N⁻¹·M_vm·M⁻¹
///               [ −M_vm   N ]              [ K     N⁻¹ ]
/// ```
///
/// (the lower-left block of `J·J⁻¹` is `−M_vm·M⁻¹ + N·K = 0`). The sign of `K`
/// is the whole of gam#3047: the variance stage's response to a mean-stage
/// error is `dγ̂ = N⁻¹·M_vm·dβ̂_m`, and `M_vm` is already the NEGATIVE of the
/// weighted `û/v`-Gram, so a second minus reverses the direction in which a
/// mean-stage error moves `γ̂`. A reversed `K` leaves the covariance PSD and
/// the diagonal blocks plausible, which is why only a refit of the system
/// itself pins it.
///
/// Two variance stages share this system:
///
///   - the Breusch-Pagan stage fired: `B = A` and `v_i` is the fitted
///     `exp(A_iᵀγ̂)`;
///   - it did not: `log v̂ = log(Σwû²/Σw)` is still ESTIMATED (gam#3030), and is
///     the same score on `B = 1` with `v_i ≡ v̂`, where `N = Σwû²/v̂ = Σw`.
///
/// `M⁺` and `N⁺` are [`preconditioned_normal_pseudoinverse`]; the pseudo-inverse
/// solves `J` exactly on the identified span, which is the only span on which
/// the Murphy-Topel propagation is defined. `var_fitted` are the `v_i`.
pub(crate) fn stacked_first_stage_inverse_bread(
    mean_basis: ArrayView2<'_, f64>,
    var_basis: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    mean_residuals: &[f64],
    var_fitted: &[f64],
    mean_normal: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    let n = mean_basis.nrows();
    let p = mean_basis.ncols();
    let q = var_basis.ncols();
    if var_basis.nrows() != n
        || mean_residuals.len() != n
        || var_fitted.len() != n
        || weights.len() != n
    {
        return Err(format!(
            "stacked first-stage bread length mismatch: mean basis rows={n}, variance basis \
             rows={}, mean_residuals={}, var_fitted={}, weights={}",
            var_basis.nrows(),
            mean_residuals.len(),
            var_fitted.len(),
            weights.len()
        ));
    }
    if mean_normal.dim() != (p, p) {
        return Err(format!(
            "stacked first-stage bread normal-matrix shape mismatch: mean basis cols={p}, mean \
             normal {:?}",
            mean_normal.dim()
        ));
    }
    if let Some((row, &v)) = var_fitted
        .iter()
        .enumerate()
        .find(|&(_, &v)| !(v.is_finite() && v > 0.0))
    {
        return Err(format!(
            "stacked first-stage bread: fitted conditional variance {v} at row {row} is not a \
             positive finite variance"
        ));
    }
    let m_pinv = preconditioned_normal_pseudoinverse(mean_normal, n)?;

    // N = Bᵀ diag(w û²/v) B  (q × q) and M_vm = −2·Bᵀ diag(w û/v) A  (q × p).
    let mut info_scaled_var_basis = var_basis.to_owned();
    let mut cross_scaled_mean_basis = mean_basis.to_owned();
    for i in 0..n {
        let standardized = mean_residuals[i] / var_fitted[i];
        let info_scale = weights[i] * mean_residuals[i] * standardized;
        let cross_scale = -2.0 * weights[i] * standardized;
        info_scaled_var_basis
            .row_mut(i)
            .iter_mut()
            .for_each(|entry| *entry *= info_scale);
        cross_scaled_mean_basis
            .row_mut(i)
            .iter_mut()
            .for_each(|entry| *entry *= cross_scale);
    }
    let var_normal = var_basis.t().dot(&info_scaled_var_basis);
    let n_pinv = preconditioned_normal_pseudoinverse(&var_normal, n)?;
    let m_vm = var_basis.t().dot(&cross_scaled_mean_basis);
    let k = n_pinv.dot(&m_vm).dot(&m_pinv);

    let mut j_inv = Array2::<f64>::zeros((p + q, p + q));
    j_inv.slice_mut(s![..p, ..p]).assign(&m_pinv);
    j_inv.slice_mut(s![p.., ..p]).assign(&k);
    j_inv.slice_mut(s![p.., p..]).assign(&n_pinv);
    Ok(j_inv)
}

/// `V₁ = ΨᵀΨ` for a first-stage row influence `Ψ`, refused when non-finite.
fn first_stage_covariance_from_row_influence(psi: &Array2<f64>) -> Result<Array2<f64>, String> {
    let mut v1 = gam_linalg::faer_ndarray::fast_ata(psi);
    gam_linalg::matrix::symmetrize_in_place(&mut v1);
    if v1.iter().any(|value| !value.is_finite()) {
        return Err("stacked first-stage sandwich covariance is non-finite".to_string());
    }
    Ok(v1)
}

/// Per-row influence `Ψ` (`n × (p+q)`) of the stacked first stage
/// [`stacked_first_stage_inverse_bread`] documents: row `i` is
/// `ψ_i = J⁻¹ S_i` with `S_i = [w_i û_i A_iᵀ | w_i r_i B_iᵀ]` and
/// `r_i = û_i²/v_i − 1`, the Gaussian log-linear variance score (gam#4019), so
/// that `θ̂₁ − θ₁ = Σ_i ψ_i + o_p(n^{-1/2})` and `V₁ = ΨᵀΨ = J⁻¹ Ω J⁻ᵀ` with the
/// robust (HC0) meat `Ω = SᵀS`. `var_fitted` are the `v_i`; `r_i` is formed here
/// rather than supplied, so the meat and the bread
/// [`stacked_first_stage_inverse_bread`] builds cannot read different variances.
///
/// Both off-diagonal channels, `M_vm` in the bread and `Ω_mv ∝ E[û³]` in the
/// meat, vanish for a Gaussian residual and neither vanishes on the branch this
/// serves (gam#2484). The mean block is the standalone HC0 sandwich `M⁺ Ω_mm M⁺`
/// exactly, because `J⁻¹` is block lower-triangular.
///
/// The covariance alone cannot combine the first stage with a second
/// influence that is a function of the same rows -- the sampling error of the
/// global-empirical latent measure built from `ζ` (gam#3452). The row influence
/// can: the total per-row influence is summed before the Gram is taken, so
/// every cross-covariance between the two channels is carried exactly.
pub(crate) fn stacked_first_stage_row_influence(
    mean_basis: ArrayView2<'_, f64>,
    var_basis: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    mean_residuals: &[f64],
    var_fitted: &[f64],
    mean_normal: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    let n = mean_basis.nrows();
    let p = mean_basis.ncols();
    let q = var_basis.ncols();
    let j_inv = stacked_first_stage_inverse_bread(
        mean_basis,
        var_basis,
        weights,
        mean_residuals,
        var_fitted,
        mean_normal,
    )?;
    let mut scores = Array2::<f64>::zeros((n, p + q));
    for i in 0..n {
        let mean_scale = weights[i] * mean_residuals[i];
        let var_scale =
            weights[i] * (mean_residuals[i] * mean_residuals[i] / var_fitted[i] - 1.0);
        for j in 0..p {
            scores[[i, j]] = mean_scale * mean_basis[[i, j]];
        }
        for j in 0..q {
            scores[[i, p + j]] = var_scale * var_basis[[i, j]];
        }
    }
    let psi = scores.dot(&j_inv.t());
    if psi.iter().any(|value| !value.is_finite()) {
        return Err("stacked first-stage row influence is non-finite".to_string());
    }
    Ok(psi)
}

/// The weighted-ridge system of the conditional calibration's mean stage on the
/// basis `A = [1 | a(C)]`: the per-column penalty `R = diag(Σ_i w_i A_ij²)` and
/// the normal matrix `M = AᵀWA + λR`, `λ = AUTO_Z_CONDITIONAL_RIDGE_REL`.
///
/// One definition, read by the fit and by the row influence the correction
/// recomputes at the seam, so the two cannot drift apart.
fn conditional_calibration_ridge_system(
    basis: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> (Array2<f64>, Array2<f64>) {
    // Per-column Tikhonov penalty scaled by the weighted Gram diagonal, so the
    // ridge is *relative* to each column's scale (a 1e-8 absolute ridge would
    // be negligible against an O(n) Gram and would not stabilize a
    // rank-deficient penalized-spline marginal index). `diag_jj = Σ_i w_i a_ij²`;
    // floored positive so the all-zero (already-dropped) directions still
    // receive a finite ridge and the factorization cannot fail.
    let mut penalty = Array2::<f64>::zeros((basis.ncols(), basis.ncols()));
    for j in 0..basis.ncols() {
        let diag_jj = basis
            .column(j)
            .iter()
            .zip(weights.iter())
            .map(|(&x, &w)| w * x * x)
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        penalty[[j, j]] = diag_jj;
    }
    // First-stage (generated-regressor) normal matrix `M = AᵀWA + λR`, the same
    // weighted-ridge system `gaussian_weighted_ridge` factorizes internally;
    // rebuilt here so its inverse can form the closed-form coefficient sandwich
    // `V₁` that the second-stage Murphy–Topel correction consumes. `p` is the
    // marginal-index width (small), so this is a cheap dense `(p+1)²` form.
    let mut wa = basis.to_owned();
    for i in 0..wa.nrows() {
        let wi = weights[i];
        wa.row_mut(i).iter_mut().for_each(|value| *value *= wi);
    }
    let mut normal = basis.t().dot(&wa);
    normal += &(penalty.to_owned() * AUTO_Z_CONDITIONAL_RIDGE_REL);
    (penalty, normal)
}

/// Weighted mean of a slice of values.
pub(crate) fn weighted_mean(
    values: &[f64],
    weights: ArrayView1<'_, f64>,
    total_weight: f64,
) -> f64 {
    values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| w * v)
        .sum::<f64>()
        / total_weight
}

/// Robust (heteroskedasticity-consistent) Rao/LM score-test p-value for the
/// null that the centered basis columns `ã(C)` carry no information about the
/// centered response `u`. This is the LAN locally-optimal statistic the issue
/// names: `s = Σ_i w_i u_i ã(C_i)`, `Ω̂ = Σ_i w_i² u_i² ã(C_i)ã(C_i)ᵀ`,
/// `D = sᵀ Ω̂⁺ s ⟶ χ²_{rank Ω̂}`. Both the conditional-mean test
/// (`u_i = z_i − z̄`) and the conditional-variance / Breusch-Pagan test
/// (`u_i = û_i² − σ̂²_û` on the conditional-mean residual `û = z − m̂(C)`)
/// are this statistic with the same centered basis.
///
/// Returns `None` when the test is degenerate (no usable basis directions),
/// otherwise the asymptotic p-value.
pub(crate) fn robust_conditional_score_pvalue(
    a_centered: ArrayView2<'_, f64>,
    u: &[f64],
    weights: ArrayView1<'_, f64>,
) -> Result<Option<f64>, String> {
    let n = a_centered.nrows();
    let r = a_centered.ncols();
    if r == 0 || n == 0 {
        return Ok(None);
    }
    if u.len() != n || weights.len() != n {
        return Err(format!(
            "conditional score test length mismatch: rows={n}, u={}, weights={}",
            u.len(),
            weights.len()
        ));
    }
    // Build the per-row scaled basis `B` with `B_i = (w_i u_i) ã_i` once, then
    // recover both the score and the HC0 robust meat from it:
    //   • score  `s   = ãᵀ (w ∘ u) = Bᵀ 1`     (column sums of `B`),
    //   • meat   `Ω̂  = Σ_i w_i² u_i² ã_i ã_iᵀ = BᵀB` since `(w_i u_i)² = w_i² u_i²`.
    // A non-positive weight zeroes that row of `B` (its score and meat
    // contributions both vanish), reproducing the `wi <= 0.0` skip EXACTLY.
    // `b_ij = (w_i·u_i)·ã_ij` takes three roundings.
    let mut b = a_centered.to_owned();
    for i in 0..n {
        let wi = weights[i];
        let scale = if wi > 0.0 { wi * u[i] } else { 0.0 };
        if scale == 0.0 {
            b.row_mut(i).fill(0.0);
            continue;
        }
        b.row_mut(i).iter_mut().for_each(|value| *value *= scale);
    }
    robust_score_contributions_pvalue(&b, 3)
}

/// The robust score test from its per-row contributions `ψ_i` (the rows of
/// `contributions`): `s = Σ_i ψ_i`, `Ω̂ = Σ_i ψ_i ψ_iᵀ`, `D = sᵀ Ω̂⁺ s ⟶
/// χ²_{rank Ω̂}`. A contribution that carries an estimated nuisance's
/// first-order effect makes `Ω̂` the variance of the score as it is actually
/// computed, not as if the nuisance were known. `row_roundings` bounds the
/// roundings that formed each `ψ_ij`, for the rank cutoff.
///
/// `None` means the test is degenerate — no contributions, or an `Ω̂` with no
/// usable direction — so there is no evidence either way. A score, meat,
/// statistic or tail probability that is not finite is a numerical failure,
/// not an absence of evidence, and is an error.
pub(crate) fn robust_score_contributions_pvalue(
    contributions: &Array2<f64>,
    row_roundings: usize,
) -> Result<Option<f64>, String> {
    let n = contributions.nrows();
    let r = contributions.ncols();
    if r == 0 || n == 0 {
        return Ok(None);
    }
    let s = contributions.sum_axis(ndarray::Axis(0));
    let omega = gam_linalg::faer_ndarray::fast_ata(contributions);
    if !s.iter().all(|v| v.is_finite()) || !omega.iter().all(|v| v.is_finite()) {
        return Err(format!(
            "robust score test over {n} rows and {r} directions: the score or its meat is not \
             finite"
        ));
    }
    // Rank cutoff: `Ω̂`'s formation band against `λ_max(Ω̂)`. Each entry sums `n`
    // products `ψ_ij·ψ_ik`, each factor formed with `row_roundings` roundings,
    // so `|E_jk| ≤ γ_{n+row_roundings}·√(Ω̂_jj·Ω̂_kk)` and
    // `‖E‖₂ ≤ γ_{n+row_roundings}·tr Ω̂`, against `λ_max(Ω̂) ≥ max_j Ω̂_jj`. The
    // `r·ε` term is the eigensolver's backward error. A PSD Gram with a zero
    // largest diagonal is the zero matrix, which has no usable direction.
    let omega_max_diagonal = omega.diag().iter().copied().fold(0.0_f64, f64::max);
    if omega_max_diagonal == 0.0 {
        return Ok(None);
    }
    let relative_cutoff = gam_linalg::roundoff::accumulation_growth(n + row_roundings)
        * omega.diag().sum()
        / omega_max_diagonal
        + r as f64 * f64::EPSILON;
    // `fast_ata` accumulates one triangle and mirrors it.
    let omega_geometry = gam_linalg::utils::rank_certified_psd_pseudoinverse(
        &omega,
        gam_linalg::roundoff::SymmetricAssembly::Mirrored,
        relative_cutoff,
    )
    .map_err(|e| format!("conditional score test pseudo-inverse failed: {e}"))?;
    let rank = omega_geometry.rank();
    let omega_pinv = omega_geometry.into_pseudoinverse();
    if rank == 0 {
        return Ok(None);
    }
    // `D = sᵀΩ̂⁺s` is formed as `‖Ψ Ω̂⁺ s‖²`, `Ψ` the contribution rows: the
    // rank-truncated pseudo-inverse satisfies `Ω̂⁺ Ω̂ Ω̂⁺ = Ω̂⁺` with `Ω̂ = ΨᵀΨ`, so
    // the two agree, and a sum of squares cannot round below zero the way the
    // quadratic form can when `s` lies in `Ω̂`'s discarded null space.
    let direction = omega_pinv.dot(&s);
    let d_stat = contributions
        .dot(&direction)
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    if !d_stat.is_finite() {
        return Err(format!(
            "robust score test over {n} rows at rank {rank}: the statistic sᵀΩ⁺s is not finite"
        ));
    }
    // The shared survival primitive owns both the direct upper-gamma identity
    // and its exact `Q(a, 0) = 1` boundary.
    let p_value = chi_square_sf(d_stat, rank as f64);
    if !p_value.is_finite() {
        return Err(format!(
            "robust score test: the chi-square({rank}) tail at D = {d_stat:e} is not finite"
        ));
    }
    Ok(Some(p_value))
}

/// Fit the conditional location-scale calibration (#905) if the conditional
/// `E[z|C]`/`Var(z|C)` Rao gate fires on the marginal-index basis `a_block`.
///
/// Returns `None` when there is no conditional structure to correct (the gate
/// does not fire, or the basis is degenerate) — in that case the caller falls
/// back to the existing pooled-marginal gate (rank-INT or no calibration).
pub(crate) fn fit_conditional_latent_calibration_if_needed(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    a_block: ArrayView2<'_, f64>,
) -> Result<Option<LatentZConditionalCalibration>, String> {
    let n = z.len();
    let p = a_block.ncols();
    if n != weights.len() {
        return Err(format!(
            "conditional latent gate length mismatch: z={n}, weights={}",
            weights.len()
        ));
    }
    if a_block.nrows() != n {
        return Err(format!(
            "conditional latent gate row mismatch: z={n}, basis rows={}",
            a_block.nrows()
        ));
    }
    if p == 0 {
        return Ok(None);
    }
    let total_weight = weights.iter().copied().sum::<f64>();
    if !(total_weight.is_finite() && total_weight > 0.0) {
        return Ok(None);
    }
    if z.iter().any(|v| !v.is_finite()) || a_block.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }

    // The conditional mean is tested on `z − z̄`, and the conditional variance
    // on the residual of the conditional mean, `û² − Σwû²/Σw` with
    // `û = z − m̂(C)`, as a Breusch-Pagan test is on OLS residuals. Testing
    // `(z − z̄)²` instead has score expectation `Cov(a, (m(a) − m̄)²) +
    // Cov(a, Var(z|a))`: a moving mean masks or fakes heteroskedasticity
    // (gam#3335). The estimated mean's first-order effect on that score is
    // propagated into its meat, so the χ² law holds under a curved mean too.
    let evidence = estimated_latent_law::conditional_law_evidence(z, weights, Some(a_block))?;
    let mean_fires = ConditionalLawEvidence::fires(evidence.mean_p_value, evidence.alpha);
    let var_fires = ConditionalLawEvidence::fires(evidence.variance_p_value, evidence.alpha);
    if !mean_fires && !var_fires {
        return Ok(None);
    }
    fit_conditional_latent_calibration(z, weights, a_block, var_fires).map(Some)
}

/// The conditional-mean stage `m(C)` of the location-scale calibration.
pub(crate) struct ConditionalMeanStage {
    /// `[1 | a(C)]`.
    pub(crate) basis: Array2<f64>,
    /// `M = AᵀWA + λR`, the normal matrix the ridge factorizes.
    pub(crate) normal: Array2<f64>,
    pub(crate) coeffs: Vec<f64>,
    /// `û = z − m̂(C)`.
    pub(crate) residuals: Vec<f64>,
}

/// Fit the conditional mean over the full basis `[1 | a(C)]` via a weighted
/// ridge (the ridge stabilizes a rank-deficient marginal-index span; it does
/// not meaningfully shrink the few directions that trigger the gate). Shared
/// by the Rao gates, whose variance tests run on this residual, and the fit.
pub(crate) fn fit_conditional_mean_stage(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    a_block: ArrayView2<'_, f64>,
) -> Result<ConditionalMeanStage, String> {
    let basis = build_intercept_basis(a_block);
    // The ridge penalty and the dense `(p+1)²` normal matrix it factorizes,
    // whose inverse propagates the mean stage into the Rao gates and the
    // Murphy–Topel `V₁` -- the one system the first-stage row influence
    // (gam#3452) rebuilds too.
    let (penalty, normal) = conditional_calibration_ridge_system(basis.view(), weights.view());
    let z_col = z.view().insert_axis(ndarray::Axis(1));
    let (coeffs_mat, fitted) = gam_linalg::utils::gaussian_weighted_ridge(
        basis.view(),
        z_col,
        penalty.view(),
        weights.view(),
        AUTO_Z_CONDITIONAL_RIDGE_REL,
    )?;
    let residuals = z
        .iter()
        .zip(fitted.column(0).iter())
        .map(|(&zi, &mi)| zi - mi)
        .collect();
    Ok(ConditionalMeanStage {
        basis,
        normal,
        coeffs: coeffs_mat.column(0).to_vec(),
        residuals,
    })
}

/// Fit the conditional location-scale calibration at a given structure, with no
/// gate: the mean `m(a)` always, and the variance `v(a)` when `fit_variance`.
/// [`fit_conditional_latent_calibration_if_needed`] fits it at the structure its
/// Rao gate chose; the moving-law certificate refits every fold at the full-data
/// fit's structure (gam#2926). Inputs that admit no fit are refused.
pub(crate) fn fit_conditional_latent_calibration(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    a_block: ArrayView2<'_, f64>,
    fit_variance: bool,
) -> Result<LatentZConditionalCalibration, String> {
    let var_fires = fit_variance;
    let n = z.len();
    let p = a_block.ncols();
    if n != weights.len() || a_block.nrows() != n {
        return Err(format!(
            "conditional latent calibration length mismatch: z={n}, weights={}, basis rows={}",
            weights.len(),
            a_block.nrows()
        ));
    }
    if p == 0 {
        return Err("conditional latent calibration needs a non-empty span".to_string());
    }
    let total_weight = weights.iter().copied().sum::<f64>();
    if !(total_weight.is_finite() && total_weight > 0.0) {
        return Err(format!(
            "conditional latent calibration needs positive finite total weight, got {total_weight}"
        ));
    }
    if z.iter().any(|v| !v.is_finite()) || a_block.iter().any(|v| !v.is_finite()) {
        return Err("conditional latent calibration needs a finite score and span".to_string());
    }
    let z_mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / total_weight;
    let global_var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - z_mean) * (zi - z_mean))
        .sum::<f64>()
        / total_weight;
    if !(global_var.is_finite() && global_var > 0.0) {
        return Err(format!(
            "conditional latent calibration needs a score with positive variance, got {global_var}"
        ));
    }

    // Escalation fires. The conditional-mean correction is applied whenever
    // the gate fires (a pure-variance trigger leaves the C-slopes of m(C) ≈ 0,
    // so it reduces to harmless global centering).
    let ConditionalMeanStage {
        basis,
        normal: _,
        coeffs: mean_coeffs,
        residuals: mean_residuals,
    } = fit_conditional_mean_stage(z, weights, a_block)?;

    // The constant variance `v̂ = Σ w û² / Σ w`: the variance stage when the
    // Breusch-Pagan stage does not fire, and the seed `log v̂` of the fired fit.
    let raw_homoskedastic_var = mean_residuals
        .iter()
        .zip(weights.iter())
        .map(|(&e, &w)| w * e * e)
        .sum::<f64>()
        / total_weight;
    if !(raw_homoskedastic_var.is_finite() && raw_homoskedastic_var > 0.0) {
        // `û ≡ 0` on every weighted row: the mean stage interpolates `z`, and no
        // conditional variance -- constant or not -- is identified.
        return Err(format!(
            "conditional latent calibration: the mean-stage residual variance is \
             {raw_homoskedastic_var}, so no conditional variance is identified"
        ));
    }
    // gam#4019: the fired variance stage is the Gaussian maximum-likelihood fit
    // of `log v(C) = γ·[1 | a(C)]` to the mean residuals, the same log-linear
    // Fisher-scoring fit the score-covariance innovation uses. The log link keeps
    // `v > 0` by construction; the linear regression of `û²` it replaced could go
    // negative and needed a hand-picked floor.
    let log_var_coeffs: Vec<f64> = if var_fires {
        crate::bms::conditional_score_covariance::fisher_score_log_linear_variance(
            &mean_residuals,
            basis.view(),
            weights.view(),
            raw_homoskedastic_var.ln(),
        )?
    } else {
        Vec::new()
    };

    // gam#2768: the homoskedastic branch's `v(C)` is the RESIDUAL variance of
    // the conditional-mean regression, not the marginal variance of z. See the
    // field doc for why the difference is the correction rather than a detail:
    // with z standardised, `1 = Var(m(C)) + E[Var(z|C)]`, so the marginal
    // variance overstates `Var(z|C)` by exactly the structure the gate just
    // detected, and dividing by it leaves `ζ` at `sd = √(1−R²)`.
    // Its estimation uncertainty is the variance stage of `theta1_cov` below.
    let mut calibration = LatentZConditionalCalibration {
        mean_coeffs,
        log_var_coeffs,
        basis_ncols: p,
        homoskedastic_var: raw_homoskedastic_var,
        post_mean: 0.0,
        post_sd: 1.0,
        theta1_cov: Array2::<f64>::zeros((0, 0)),
    };
    // The joint first-stage covariance is the Gram of the SAME row influence
    // the seam recomputes when the second-stage measure is estimated too
    // (gam#3452), so the two cannot come from different formulas.
    let psi = calibration.theta1_row_influence(z.view(), a_block, weights.view())?;
    calibration.theta1_cov = first_stage_covariance_from_row_influence(&psi)?;

    // Sanity-check post-correction moments on the training sample, whose
    // calibrated score is the fitted score map's (gam#3016).
    let calibrated = FittedLatentScoreMap::conditional_only(&calibration)
        .calibrate(z.view(), Some(a_block))?;
    let post_mean = weighted_mean(
        calibrated
            .as_slice()
            .expect("the fitted score map returns an owned standard-layout 1-D array"),
        weights.view(),
        total_weight,
    );
    let post_var = calibrated
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - post_mean) * (zi - post_mean))
        .sum::<f64>()
        / total_weight;
    calibration.post_mean = post_mean;
    calibration.post_sd = post_var.max(0.0).sqrt();

    Ok(calibration)
}

/// Prepend a column of ones to `a_block`, producing the `[1 | a(C)]` regression
/// basis used by the conditional location-scale fit.
pub(crate) fn build_intercept_basis(a_block: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = a_block.nrows();
    let p = a_block.ncols();
    let mut basis = Array2::<f64>::ones((n, p + 1));
    basis.slice_mut(s![.., 1..]).assign(&a_block);
    basis
}

/// Which latent measures the *calling family's row kernel* can actually
/// evaluate.
///
/// This is the only thing about the kernel that differs between the two
/// marginal-slope families' latent-measure decisions, so it is the only
/// capability [`build_latent_measure_decision`] takes to serve both. The
/// Bernoulli kernel owns an empirical-grid branch; the survival kernel owns one
/// (the anchored frame, gam#2923) on the configurations
/// `anchored_kernel_unavailable_reason` admits, and only the closed-form
/// Gaussian lowering elsewhere. A finite law handed to a closed-form-only
/// kernel would be a law it cannot evaluate, so the gate refuses it by name
/// instead (gam#2926).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EmpiricalLatentMeasureSupport {
    /// The caller can evaluate `GlobalEmpirical` and `LocalEmpirical` laws.
    Available,
    /// The caller can only evaluate `LatentMeasureKind::StandardNormal`.
    StandardNormalOnly,
}

/// The latent-measure gate's verdict: the law the kernel will integrate
/// against, the pre-transform applied to z before it reaches that kernel (only
/// the declared conditional location-scale law has one), and the record of
/// which law the fit consumed.
pub(crate) struct LatentMeasureDecision {
    pub(crate) kind: LatentMeasureKind,
    pub(crate) calibration: LatentMeasureCalibration,
    /// `Some` exactly when `kind` is a `GlobalEmpirical` law compressed from the
    /// conditionally standardised residual `ζ`: the record of the equal-mass
    /// compression that produced it, which is what makes the measure
    /// differentiable in the first stage it was built from.
    ///
    /// Fit-time only and deliberately not part of `kind`: the measure is on the
    /// persistence wire and its identity is its nodes and weights, while this is
    /// provenance about the rows behind them. The gam#2484 Murphy–Topel
    /// correction is its only consumer, and it is engaged only beside a
    /// conditional calibration; a law estimated from the raw score has no first
    /// stage to correct for.
    pub(crate) empirical_build: Option<empirical_measure_sensitivity::EmpiricalZGridBuild>,
    /// `Some` exactly when `consumed` is a provisional
    /// [`LatentLawConsumed::EstimatedGaussianAdequate`], or a
    /// [`LatentLawConsumed::DeclaredGaussian`] whose score failed the adequacy
    /// screen: the estimated law the family measures the converged closed-form
    /// fit's anchoring residual under, and, for the default only, re-solves on when
    /// the certificate prefers it.
    pub(crate) certificate_law: Option<EmpiricalZGrid>,
    pub(crate) consumed: LatentLawConsumed,
    /// `Some` exactly when `consumed` is a provisional
    /// [`LatentLawConsumed::EstimatedMovingLaw`]: the held-out laws its
    /// certificate scores at the converged fit, and the full-data law of every
    /// arm a re-solve can anchor on.
    pub(crate) moving_law: Option<Box<moving_law_rule::MovingLawCandidates>>,
}

/// The latent-measure gate, shared by both marginal-slope families (gam#2768,
/// gam#2926).
///
/// The default estimates the law the score has and anchors on it, the score on
/// its own axis; nothing is done to the score to make it look Gaussian:
///
/// 1. the conditional-law structure test on the marginal-index span `a(C)`
///    ([`estimated_latent_law::conditional_law_evidence`]);
/// 2. no structure, and a score the standard-normal adequacy check cannot tell
///    from `N(0, 1)`: the closed-form Gaussian law, with that evidence recorded
///    ([`LatentLawConsumed::EstimatedGaussianAdequate`]);
/// 3. no structure, and a score that fails the check: one finite law of the
///    score ([`LatentLawConsumed::EstimatedGlobal`]);
/// 4. structure: the location-scale Gaussian law (the closed form when neither
///    conditional moment moves), certified at the converged fit against the
///    Gaussian, location-scale empirical and local arms by the moving-law rule
///    ([`moving_law_rule`], [`LatentLawConsumed::EstimatedMovingLaw`]).
///
/// Otherwise the Gaussian closed form is reached only by declaration
/// (`LatentMeasureSpec::StandardNormal`). A declaration whose conditional mean or
/// variance moves on the span is refused with the typed [`LatentLawRefusal`]: no
/// single declared law can be right. One whose pooled score fails the fixed
/// adequacy screen is fitted and warned about, with its estimated excess anchoring
/// loss measured at the converged fit, because the screen's tolerances say nothing
/// about what the declaration costs. A finite law asked of a kernel that evaluates
/// only the closed form is refused, which is the one thing `support` changes.
pub(crate) fn build_latent_measure_decision(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    policy: &LatentZPolicy,
    conditioning: Option<ArrayView2<'_, f64>>,
    local_context: Option<&estimated_latent_law::LocalLawContext<'_>>,
    support: EmpiricalLatentMeasureSupport,
    context: &str,
) -> Result<LatentMeasureDecision, String> {
    let refuse_closed_form_only = |requested: &str| -> Result<(), String> {
        if support == EmpiricalLatentMeasureSupport::StandardNormalOnly {
            return Err(LatentLawRefusal::EmpiricalKernelUnavailable {
                context: context.to_string(),
                requested: requested.to_string(),
            }
            .to_string());
        }
        Ok(())
    };
    match policy.latent_measure {
        LatentMeasureSpec::Auto { grid_size } => {
            let evidence =
                estimated_latent_law::conditional_law_evidence(z, weights, conditioning)?;
            if !evidence.law_moves() {
                let adequacy = latent_z_normal_adequacy(z, weights, policy)?;
                if adequacy.passes() {
                    // The screen says nothing directly about the anchoring error,
                    // so the closed form is provisional: the family measures each
                    // row's anchoring residual under this estimated law at the
                    // converged fit, and re-solves on it when that is over
                    // tolerance.
                    let certificate_law = estimated_latent_law::build_empirical_law_on_own_axis(
                        z.view(),
                        weights.view(),
                        grid_size,
                        "estimated latent law",
                    )?;
                    log::debug!(
                        "[{context} latent-z] the conditional law of the score does not move on \
                         the marginal-index span ({}) and the score passes the standard-normal \
                         adequacy screen ({}); fitting the closed-form Gaussian law, to be \
                         certified by its anchoring residual at the converged fit (gam#2926)",
                        evidence.summary(),
                        adequacy.ledger(),
                    );
                    return Ok(LatentMeasureDecision {
                        kind: LatentMeasureKind::StandardNormal,
                        calibration: LatentMeasureCalibration::None,
                        empirical_build: None,
                        certificate_law: Some(certificate_law),
                        moving_law: None,
                        consumed: LatentLawConsumed::EstimatedGaussianAdequate {
                            evidence,
                            adequacy,
                            residual: None,
                        },
                    });
                }
                if support == EmpiricalLatentMeasureSupport::StandardNormalOnly {
                    let missing = format!(
                        "the score fails the standard-normal adequacy screen (adequacy ledger, x = \
                         statistic / bound, x<=1 passed: {}), and this configuration's row kernel \
                         evaluates only the closed form",
                        adequacy.ledger()
                    );
                    log::debug!(
                        "[{context} latent-z] fitting the closed form uncertified: {missing} \
                         (gam#2926)"
                    );
                    return Ok(LatentMeasureDecision {
                        kind: LatentMeasureKind::StandardNormal,
                        calibration: LatentMeasureCalibration::None,
                        empirical_build: None,
                        certificate_law: None,
                        moving_law: None,
                        consumed: LatentLawConsumed::GaussianUncertified {
                            evidence,
                            adequacy: Some(adequacy),
                            certificate: None,
                            missing,
                        },
                    });
                }
                let grid = estimated_latent_law::build_empirical_law_on_own_axis(
                    z.view(),
                    weights.view(),
                    grid_size,
                    "estimated latent law",
                )?;
                log::debug!(
                    "[{context} latent-z] the conditional law of the score does not move on the \
                     marginal-index span ({}) and the score fails the standard-normal adequacy \
                     check ({}); anchoring on its estimated law of {} nodes, the score on its own \
                     axis (gam#2926)",
                    evidence.summary(),
                    adequacy.ledger(),
                    grid.nodes.len(),
                );
                return Ok(LatentMeasureDecision {
                    kind: LatentMeasureKind::GlobalEmpirical { grid },
                    calibration: LatentMeasureCalibration::None,
                    empirical_build: None,
                    certificate_law: None,
                    moving_law: None,
                    consumed: LatentLawConsumed::EstimatedGlobal { evidence },
                });
            }
            if support == EmpiricalLatentMeasureSupport::StandardNormalOnly {
                let missing = format!(
                    "the conditional law of the score moves on the marginal-index span ({}), and \
                     this configuration's row kernel evaluates only the closed form",
                    evidence.summary()
                );
                log::debug!(
                    "[{context} latent-z] fitting the closed form uncertified: {missing} (gam#2926)"
                );
                return Ok(LatentMeasureDecision {
                    kind: LatentMeasureKind::StandardNormal,
                    calibration: LatentMeasureCalibration::None,
                    empirical_build: None,
                    certificate_law: None,
                    moving_law: None,
                    consumed: LatentLawConsumed::GaussianUncertified {
                        evidence,
                        adequacy: None,
                        certificate: None,
                        missing,
                    },
                });
            }
            let local = local_context.ok_or_else(|| {
                LatentLawRefusal::LocalLawContextUnavailable {
                    context: context.to_string(),
                    evidence: evidence.clone(),
                }
                .to_string()
            })?;
            // The law moves, so it is chosen among nested arms by the moving-law
            // certificate at the converged fit. The fit starts on the simplest
            // admissible arm that follows a moving mean and variance: the
            // location-scale Gaussian law only if its residual passes the adequacy
            // screen.
            let a_block = conditioning.ok_or_else(|| {
                format!(
                    "{context}: the conditional-law evidence moved without a marginal-index span \
                     to test it on"
                )
            })?;
            let candidates = moving_law_rule::MovingLawCandidates::build(
                z,
                weights,
                a_block,
                local,
                grid_size,
                policy,
                evidence.clone(),
                context,
            )
            .map_err(|error| error.to_string())?;
            let fitted = candidates.fitted_arm();
            log::debug!(
                "[{context} latent-z] the conditional law of the score moves on the \
                 marginal-index span ({}); fitting the {} law, to be certified against the \
                 other arms ({}) by their cross-fitted excess anchoring loss at the converged \
                 fit (gam#2926)",
                evidence.summary(),
                fitted.label(),
                candidates
                    .arms()
                    .iter()
                    .map(|arm| arm.label())
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            let mut decision = candidates
                .decision_for(fitted, None)
                .map_err(|error| error.to_string())?;
            decision.moving_law = Some(Box::new(candidates));
            Ok(decision)
        }
        LatentMeasureSpec::StandardNormal => {
            let evidence =
                estimated_latent_law::conditional_law_evidence(z, weights, conditioning)?;
            if evidence.mean_or_variance_moves() {
                return Err(LatentLawRefusal::GaussianConditionalMomentsMove {
                    context: context.to_string(),
                    evidence,
                }
                .to_string());
            }
            let adequacy = latent_z_normal_adequacy(z, weights, policy)?;
            if adequacy.passes() {
                return Ok(LatentMeasureDecision {
                    kind: LatentMeasureKind::StandardNormal,
                    calibration: LatentMeasureCalibration::None,
                    empirical_build: None,
                    certificate_law: None,
                    moving_law: None,
                    consumed: LatentLawConsumed::DeclaredGaussian {
                        evidence,
                        adequacy: None,
                        residual: None,
                        uncertified: None,
                    },
                });
            }
            // The shape screen's bounds are fixed tolerances, not tests of the
            // declaration's consequence: at small n they reject exact Gaussian
            // scores, at large n harmless departures. So a failed screen does not
            // refuse a declaration. The family measures what the declaration costs
            // at the converged fit instead, the excess anchoring loss `D̂` under the
            // estimated law, and warns with both.
            let certificate_law = estimated_latent_law::build_empirical_law_on_own_axis(
                z.view(),
                weights.view(),
                DEFAULT_EMPIRICAL_LATENT_GRID_SIZE,
                "estimated latent law",
            )?;
            log::debug!(
                "[{context} latent-z] the Gaussian latent law was declared, and the score fails \
                 the standard-normal adequacy screen (adequacy ledger, x = statistic / bound, \
                 x<=1 passed: {}); fitting the declared closed form, whose estimated excess \
                 anchoring loss is measured at the converged fit (gam#2926)",
                adequacy.ledger(),
            );
            Ok(LatentMeasureDecision {
                kind: LatentMeasureKind::StandardNormal,
                calibration: LatentMeasureCalibration::None,
                empirical_build: None,
                certificate_law: Some(certificate_law),
                moving_law: None,
                consumed: LatentLawConsumed::DeclaredGaussian {
                    evidence,
                    adequacy: Some(adequacy),
                    residual: None,
                    uncertified: None,
                },
            })
        }
        LatentMeasureSpec::GlobalEmpirical { grid_size } => {
            refuse_closed_form_only("latent_measure=\"global-empirical\"")?;
            let grid = estimated_latent_law::build_empirical_law_on_own_axis(
                z.view(),
                weights.view(),
                grid_size,
                "global-empirical latent law",
            )?;
            Ok(LatentMeasureDecision {
                kind: LatentMeasureKind::GlobalEmpirical { grid },
                calibration: LatentMeasureCalibration::None,
                empirical_build: None,
                certificate_law: None,
                moving_law: None,
                consumed: LatentLawConsumed::RequestedGlobalEmpirical,
            })
        }
        LatentMeasureSpec::ConditionalLocationScale { grid_size } => {
            refuse_closed_form_only("latent_measure=\"conditional-location-scale\"")?;
            let a_block = conditioning.ok_or_else(|| {
                LatentLawRefusal::LocationScaleSpanUnavailable {
                    context: context.to_string(),
                }
                .to_string()
            })?;
            let evidence =
                estimated_latent_law::conditional_law_evidence(z, weights, Some(a_block))?;
            match fit_conditional_latent_calibration_if_needed(z, weights, a_block)? {
                Some(cal) => {
                    // The residual law is anchored on as it is, never lowered in
                    // closed form: a two-point residual survives location-scale
                    // correction unchanged in shape, and only a declaration may
                    // make it Gaussian.
                    let zeta = FittedLatentScoreMap::conditional_only(&cal)
                        .calibrate(z.view(), Some(a_block))?;
                    let (kind, build) =
                        build_global_empirical_latent_measure(&zeta, weights, grid_size)?;
                    log::debug!(
                        "[{context} latent-z] declared conditional location-scale law: \
                         basis_ncols={} var_active={} post_mean={:.3e} post_sd={:.3e}; the \
                         residual is anchored on its empirical law (gam#2926)",
                        cal.basis_ncols,
                        !cal.log_var_coeffs.is_empty(),
                        cal.post_mean,
                        cal.post_sd,
                    );
                    Ok(LatentMeasureDecision {
                        kind,
                        calibration: LatentMeasureCalibration::ConditionalLocationScale(cal),
                        empirical_build: Some(build),
                        certificate_law: None,
                        moving_law: None,
                        consumed: LatentLawConsumed::ConditionalLocationScale {
                            calibrated: true,
                            evidence,
                        },
                    })
                }
                None => {
                    let grid = estimated_latent_law::build_empirical_law_on_own_axis(
                        z.view(),
                        weights.view(),
                        grid_size,
                        "conditional location-scale latent law",
                    )?;
                    Ok(LatentMeasureDecision {
                        kind: LatentMeasureKind::GlobalEmpirical { grid },
                        calibration: LatentMeasureCalibration::None,
                        empirical_build: None,
                        certificate_law: None,
                        moving_law: None,
                        consumed: LatentLawConsumed::ConditionalLocationScale {
                            calibrated: false,
                            evidence,
                        },
                    })
                }
            }
        }
    }
}

/// The standard-normal adequacy verdict on a latent-z sample: every statistic
/// the gate forms, beside the bound it was judged against.
///
/// The gate used to return a bare `bool`, and all three of its call sites threw
/// the evidence away. That is why "how far is the calibrated residual from
/// standard normal when the gate trips?" could not be answered from a fit --
/// the failing clause and its margin existed only inside the conjunction and
/// were discarded at the `&&`. The conjunction now lives in [`Self::passes`]
/// and the evidence survives it, which is what lets the conditional
/// location-scale branch say what the data did (gam#2484).
///
/// Every field is in the units of its own clause; no field is a ratio, so a
/// consumer can report either the raw statistic or its margin.
///
/// Persisted in [`LatentLawConsumed::EstimatedGaussianAdequate`], which is minted
/// only when every clause passed, so every persisted statistic is finite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatentNormalAdequacy {
    /// Kish effective sample size `(Σw)² / Σw²`, which sets the moment bounds.
    pub effective_n: f64,
    pub mean: f64,
    pub mean_tol: f64,
    pub sd: f64,
    pub sd_tol: f64,
    pub skew: f64,
    pub skew_tol: f64,
    pub excess_kurtosis: f64,
    pub excess_kurtosis_tol: f64,
    pub ks: f64,
    pub ks_tol: f64,
    pub tail_mass_inner: f64,
    pub tail_bound_inner: f64,
    pub tail_mass_outer: f64,
    pub tail_bound_outer: f64,
    pub max_abs: f64,
    pub max_abs_tol: f64,
}

impl LatentNormalAdequacy {
    /// The nine-clause conjunction that admits the closed-form standard-normal
    /// kernel. A `NaN` statistic never passes: the `is_finite` clauses and the
    /// comparisons both reject it, which is how a degenerate sample (constant
    /// or non-finite) is refused without being reported as a large deviation it
    /// was never measured to have.
    pub(crate) fn passes(&self) -> bool {
        self.mean.abs() <= self.mean_tol
            && (self.sd - 1.0).abs() <= self.sd_tol
            && self.skew.is_finite()
            && self.skew.abs() <= self.skew_tol
            && self.excess_kurtosis.is_finite()
            && self.excess_kurtosis.abs() <= self.excess_kurtosis_tol
            && self.ks.is_finite()
            && self.ks <= self.ks_tol
            && self.tail_mass_inner <= self.tail_bound_inner
            && self.tail_mass_outer <= self.tail_bound_outer
            && self.max_abs < self.max_abs_tol
    }

    /// One line naming every clause, its statistic, its bound, and the factor by
    /// which it missed -- so a reader can tell a marginal failure from a
    /// structural one without re-running the fit. `x` is the ratio of the
    /// statistic to its bound; a clause with `x <= 1` passed.
    pub(crate) fn ledger(&self) -> String {
        fn clause(name: &str, value: f64, bound: f64) -> String {
            format!(
                "{name}={value:.4e}/{bound:.4e}(x{:.2})",
                (value / bound).abs()
            )
        }
        format!(
            "n_eff={:.1} {} {} {} {} {} {} {} {}",
            self.effective_n,
            clause("|mean|", self.mean, self.mean_tol),
            clause("|sd-1|", self.sd - 1.0, self.sd_tol),
            clause("|skew|", self.skew, self.skew_tol),
            clause(
                "|excess_kurtosis|",
                self.excess_kurtosis,
                self.excess_kurtosis_tol
            ),
            clause("ks", self.ks, self.ks_tol),
            clause(
                "tail_mass_inner",
                self.tail_mass_inner,
                self.tail_bound_inner
            ),
            clause(
                "tail_mass_outer",
                self.tail_mass_outer,
                self.tail_bound_outer
            ),
            clause("max_abs", self.max_abs, self.max_abs_tol),
        )
    }
}

/// The adequacy screen's bounds at Kish effective size `n` (gam#2926). Each is the
/// level-`α/8` null quantile of its clause's statistic for `n` draws of an exact
/// N(0, 1) score, so such a score fails the screen with probability at most
/// [`AUTO_Z_NORMAL_SCREEN_ALPHA`], at every `n`. The mean and sd bounds are the
/// two-sided normal quantile over their standard errors, skewness and excess
/// kurtosis the same over their exact normal-sample moments, the KS distance
/// Kolmogorov's critical value in Stephens' finite-`n` form, each tail mass the
/// Poisson upper quantile of its expected count, and `max |z|` the level `n` draws
/// exceed with probability `α/8`. A stricter policy cap still binds the mean, sd,
/// skewness and kurtosis. Skewness and kurtosis are not testable below four
/// effective draws, and do not bound there.
struct NormalScreenBounds {
    mean: f64,
    sd: f64,
    skew: f64,
    excess_kurtosis: f64,
    ks: f64,
    tail_inner: f64,
    tail_outer: f64,
    max_abs: f64,
}

fn normal_screen_bounds(n: f64, policy: &LatentZPolicy) -> Result<NormalScreenBounds, String> {
    let alpha = AUTO_Z_NORMAL_SCREEN_ALPHA / AUTO_Z_NORMAL_SCREEN_CLAUSES;
    let two_sided = standard_normal_quantile(1.0 - alpha / 2.0)?;
    let (skew, excess_kurtosis) = if n > 3.0 {
        let skew_sd = (6.0 * (n - 2.0) / ((n + 1.0) * (n + 3.0))).sqrt();
        let kurtosis_mean = -6.0 / (n + 1.0);
        let kurtosis_sd = (24.0 * n * (n - 2.0) * (n - 3.0)
            / ((n + 1.0) * (n + 1.0) * (n + 3.0) * (n + 5.0)))
            .sqrt();
        (
            two_sided * skew_sd,
            kurtosis_mean.abs() + two_sided * kurtosis_sd,
        )
    } else {
        (f64::INFINITY, f64::INFINITY)
    };
    let root_n = n.sqrt();
    let ks = kolmogorov_upper_quantile(alpha) / (root_n + 0.12 + 0.11 / root_n);
    let tail = |sigma: f64| {
        poisson_upper_quantile(n * normal_two_sided_probability(sigma), alpha) / n
    };
    // 1 − (1 − α)^{1/n}: the per-draw exceedance that n draws reach with
    // probability α.
    let per_draw = -((-alpha).ln_1p() / n).exp_m1();
    let max_abs = -standard_normal_quantile(per_draw / 2.0)?;
    Ok(NormalScreenBounds {
        mean: policy.mean_tol_multiplier.min(two_sided) / root_n,
        sd: policy.sd_tol_multiplier.min(two_sided) / (2.0 * (n - 1.0).max(1.0)).sqrt(),
        skew: policy.max_abs_skew.min(skew),
        excess_kurtosis: policy.max_abs_excess_kurtosis.min(excess_kurtosis),
        ks,
        tail_inner: tail(AUTO_Z_NORMAL_TAIL_SIGMA_INNER),
        tail_outer: tail(AUTO_Z_NORMAL_TAIL_SIGMA_OUTER),
        max_abs,
    })
}

/// `λ` with `P(K > λ) = α` for Kolmogorov's limit law,
/// `P(K > λ) = 2 Σ_{j≥1} (−1)^{j−1} e^{−2 j² λ²}`, by bisection on its
/// decreasing survival function.
fn kolmogorov_upper_quantile(alpha: f64) -> f64 {
    let survival = |lambda: f64| {
        let mut sum = 0.0;
        for j in 1..=100_u32 {
            let jf = f64::from(j);
            let term = (-2.0 * jf * jf * lambda * lambda).exp();
            sum += if j % 2 == 1 { term } else { -term };
            if term < 1e-18 {
                break;
            }
        }
        2.0 * sum
    };
    let (mut low, mut high) = (0.2_f64, 5.0_f64);
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if survival(middle) > alpha {
            low = middle;
        } else {
            high = middle;
        }
    }
    high
}

/// The smallest count `k` with `P(X > k) ≤ α` for `X ~ Poisson(mean)`.
fn poisson_upper_quantile(mean: f64, alpha: f64) -> f64 {
    let mut log_pmf = -mean;
    let mut cdf = log_pmf.exp();
    let mut k = 0.0_f64;
    while 1.0 - cdf > alpha {
        k += 1.0;
        log_pmf += mean.ln() - k.ln();
        cdf += log_pmf.exp();
    }
    k
}

/// Measure a latent-z sample against the standard-normal adequacy gate,
/// returning every statistic and bound rather than only the verdict.
pub(crate) fn latent_z_normal_adequacy(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    policy: &LatentZPolicy,
) -> Result<LatentNormalAdequacy, String> {
    if z.len() != weights.len() {
        return Err(format!(
            "latent-measure auto-detection length mismatch: z={}, weights={}",
            z.len(),
            weights.len()
        ));
    }
    let weight_sum = weights.iter().copied().sum::<f64>();
    let weight_sq_sum = weights.iter().map(|&w| w * w).sum::<f64>();
    if !(weight_sum.is_finite()
        && weight_sum > 0.0
        && weight_sq_sum.is_finite()
        && weight_sq_sum > 0.0)
    {
        return Err("latent-measure auto-detection requires positive finite weights".to_string());
    }
    let effective_n = weight_sum * weight_sum / weight_sq_sum;
    if !(effective_n.is_finite() && effective_n > 1.0) {
        return Err(
            "latent-measure auto-detection requires at least two effective observations"
                .to_string(),
        );
    }
    let mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / weight_sum;
    let var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
        .sum::<f64>()
        / weight_sum;
    let sd = var.sqrt();
    // Every bound depends only on the effective sample size, so they are formed
    // before the degeneracy check and stated once for both exits.
    let bounds = normal_screen_bounds(effective_n, policy)?;
    if !(mean.is_finite() && sd.is_finite() && sd > 0.0) {
        // A constant or non-finite sample has no shape: every standardized
        // statistic divides by `sd`, so skewness, kurtosis and the tail masses
        // are UNDEFINED here rather than merely large. `NaN` is the honest
        // entry -- `passes` rejects it through the same `is_finite` clauses the
        // conjunction always had, so this exit refuses the standard-normal
        // kernel exactly as the previous `Ok(false)` did, without reporting a
        // deviation that was never measured.
        return Ok(LatentNormalAdequacy {
            effective_n,
            mean,
            mean_tol: bounds.mean,
            sd,
            sd_tol: bounds.sd,
            skew: f64::NAN,
            skew_tol: bounds.skew,
            excess_kurtosis: f64::NAN,
            excess_kurtosis_tol: bounds.excess_kurtosis,
            ks: f64::NAN,
            ks_tol: bounds.ks,
            tail_mass_inner: f64::NAN,
            tail_bound_inner: bounds.tail_inner,
            tail_mass_outer: f64::NAN,
            tail_bound_outer: bounds.tail_outer,
            max_abs: f64::NAN,
            max_abs_tol: bounds.max_abs,
        });
    }
    let skew = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| {
            let centered = (zi - mean) / sd;
            wi * centered.powi(3)
        })
        .sum::<f64>()
        / weight_sum;
    let excess_kurtosis = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| {
            let centered = (zi - mean) / sd;
            wi * centered.powi(4)
        })
        .sum::<f64>()
        / weight_sum
        - 3.0;
    let ks_to_normal = weighted_ks_to_standard_normal(z, weights, weight_sum)?;
    let tail_mass_4 = weighted_tail_mass(z, weights, weight_sum, AUTO_Z_NORMAL_TAIL_SIGMA_INNER);
    let tail_mass_6 = weighted_tail_mass(z, weights, weight_sum, AUTO_Z_NORMAL_TAIL_SIGMA_OUTER);
    let max_abs_z = z.iter().fold(0.0_f64, |acc, &zi| acc.max(zi.abs()));
    Ok(LatentNormalAdequacy {
        effective_n,
        mean,
        mean_tol: bounds.mean,
        sd,
        sd_tol: bounds.sd,
        skew,
        skew_tol: bounds.skew,
        excess_kurtosis,
        excess_kurtosis_tol: bounds.excess_kurtosis,
        ks: ks_to_normal,
        ks_tol: bounds.ks,
        tail_mass_inner: tail_mass_4,
        tail_bound_inner: bounds.tail_inner,
        tail_mass_outer: tail_mass_6,
        tail_bound_outer: bounds.tail_outer,
        max_abs: max_abs_z,
        max_abs_tol: bounds.max_abs,
    })
}

/// The global-empirical latent measure AND the fit-time record of how it was
/// built.
///
/// The two are returned together, never separately: the record is what makes
/// the measure differentiable in the sample it was compressed from (gam#2484),
/// and a record obtained from a second call to the builder would be a record of
/// a second compression.
pub(crate) fn build_global_empirical_latent_measure(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    grid_size: usize,
) -> Result<
    (
        LatentMeasureKind,
        empirical_measure_sensitivity::EmpiricalZGridBuild,
    ),
    String,
> {
    let build = empirical_measure_sensitivity::build_empirical_z_grid_with_alpha(
        z.view(),
        weights.view(),
        grid_size,
        "empirical latent measure",
    )?;
    let measure = LatentMeasureKind::GlobalEmpirical {
        grid: build.grid.clone(),
    };
    measure.validate("empirical latent measure")?;
    Ok((measure, build))
}

pub(crate) fn weighted_ks_to_standard_normal(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    total_weight: f64,
) -> Result<f64, String> {
    let mut pairs = Vec::<(f64, f64)>::with_capacity(z.len());
    for (&zi, &wi) in z.iter().zip(weights.iter()) {
        if !zi.is_finite() || !wi.is_finite() || wi < 0.0 {
            return Err(
                "latent-measure KS diagnostic requires finite z and non-negative finite weights"
                    .to_string(),
            );
        }
        if wi > 0.0 {
            pairs.push((zi, wi));
        }
    }
    pairs.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .expect("validated latent z values are finite")
    });
    let mut prev = 0.0;
    let mut ks = 0.0_f64;
    for (zi, wi) in pairs {
        let cdf = normal_cdf(zi);
        let next = prev + wi / total_weight;
        ks = ks.max((cdf - prev).abs()).max((cdf - next).abs());
        prev = next;
    }
    Ok(ks)
}

pub(crate) fn weighted_tail_mass(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    total_weight: f64,
    cutoff: f64,
) -> f64 {
    z.iter()
        .zip(weights.iter())
        .filter(|&(&zi, _)| zi.abs() > cutoff)
        .map(|(_, &wi)| wi)
        .sum::<f64>()
        / total_weight
}

// ---------------------------------------------------------------------------
// Cross-module constants — declared here so all submodules can reach them
// via `use super::*` without promoting implementation details to pub(crate).
// ---------------------------------------------------------------------------
/// Upper bound (and large-`n` default) for rows-per-chunk in the parallel
/// row-accumulation phases.
///
/// This is also a hard *ceiling* the [`bms_row_chunk_size`] chunk sizing must
/// respect: several per-chunk fast paths (block-Hessian / block-gradient
/// assembly) allocate fixed `[0.0f64; ROW_CHUNK_SIZE]` stack buffers and index
/// them by the chunk's local row position, so a chunk may never carry more than
/// `ROW_CHUNK_SIZE` rows.
pub(super) const ROW_CHUNK_SIZE: usize = 1024;
/// Floor for rows-per-chunk: below it the per-chunk scratch allocation +
/// scheduler hand-off cost dominates the row arithmetic. Small enough that a
/// moderate `n` on a many-core box still carves several chunks per worker.
pub(super) const ROW_CHUNK_MIN: usize = 64;
/// Target number of row-chunks per rayon worker for the BMS exact-Newton
/// row-fan-out phases (gradient / HVP / diagonal directional-derivative sweeps).
///
/// Several chunks per worker keeps the pool load-balanced across the uneven
/// per-row cost tail (work-stealing moves whole chunks, never partial sums) so
/// the heavy coord-corrections / row-stream phases saturate the cores instead
/// of stranding the tail on one worker.
pub(super) const ROW_CHUNKS_PER_WORKER: usize = 4;

/// Pool-aware rows-per-chunk for the BMS exact-Newton row fan-outs.
///
/// A *fixed* `ROW_CHUNK_SIZE` divisor makes the chunk **count** scale with `n`,
/// so at moderate `n` (e.g. `n = 10·ROW_CHUNK_SIZE` on a 64-core box) the
/// `into_par_iter` over `⌈n/ROW_CHUNK_SIZE⌉` chunks has far fewer tasks than
/// workers and most cores idle — the measured ~30-90% core utilization on the
/// biobank coord-corrections / row-stream phases. This sizes the chunk so the
/// chunk count targets `ROW_CHUNKS_PER_WORKER × worker_count` (the same policy
/// `chunked_row_reduction` uses), clamped to `[ROW_CHUNK_MIN, ROW_CHUNK_SIZE]`:
///
/// * the `ROW_CHUNK_SIZE` ceiling is mandatory — the block-assembly fast paths
///   index fixed `[…; ROW_CHUNK_SIZE]` stack buffers by local row, so a chunk
///   can never exceed it. At large `n` the per-1024-row count already exceeds
///   the worker count, so the clamp costs nothing there;
/// * the `ROW_CHUNK_MIN` floor stops sub-floor fan-out at tiny `n`.
///
/// Reproducibility contract (#1045): the worker count used here is the
/// process-stable machine parallelism (`reproducible_chunk_parallelism`), NOT
/// the live `rayon::current_num_threads()` of the executing (possibly scoped,
/// possibly shrunk) pool. Keying the chunk *count* — and hence the chunk
/// boundaries `chunk_idx·chunk → (chunk_idx+1)·chunk` — to the transient pool
/// size made the per-chunk row sums regroup when the pool was narrowed, so a
/// fit reduced over these chunks and fed into the iterative REML optimizer moved
/// its `(ρ, λ)` selection with the pool size. Anchoring to a process constant
/// makes the boundaries — and therefore the `try_fold`/`try_reduce` reduction
/// tree that round-trips through them — invariant to how many workers run the
/// fit, while rayon still fans the chunks across whatever workers exist. For a
/// given `n` the returned chunk size is stable across calls and pool sizes.
#[inline]
pub(super) fn bms_row_chunk_size(n: usize) -> usize {
    if n == 0 {
        return ROW_CHUNK_SIZE;
    }
    let workers = crate::marginal_slope_shared::reproducible_chunk_parallelism();
    let target_chunks = workers.saturating_mul(ROW_CHUNKS_PER_WORKER).max(1);
    // Rows per chunk that yields ≈ `target_chunks` chunks, clamped into
    // `[ROW_CHUNK_MIN, ROW_CHUNK_SIZE]`.
    n.div_ceil(target_chunks)
        .clamp(ROW_CHUNK_MIN, ROW_CHUNK_SIZE)
}
/// Row count from which `log_exact_work` turns on the BMS exact-path stage logs.
///
/// Work bound (#2469): result-invariant. Every `log_exact_work` gate encloses
/// only `log` macros, the elapsed times and sizes they print, and progress
/// counters that feed nothing but those lines.
pub(super) const EXACT_WORK_LOG_MIN_ROWS: usize = 50_000;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_EXPECTED_REUSE_PASSES: usize = 3;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_MIN_REUSE_PASSES: usize = 2;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_TILE_ROWS: usize = 8192;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_SINGLE_FRACTION_NUM: u64 = 1;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_SINGLE_FRACTION_DEN: u64 = 4;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_GLOBAL_FRACTION_NUM: u64 = 1;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_GLOBAL_FRACTION_DEN: u64 = 2;
pub(super) const BERNOULLI_MARGSLOPE_LINE_SEARCH_EARLY_EXIT_CHUNK_ROWS: usize = 10_000;

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------
pub(crate) mod block_specs;
pub mod conditional_score_covariance;
pub(crate) mod estimated_latent_law;
pub(crate) mod exact_eval_cache;
mod expected_information;
pub(crate) mod family;
pub(crate) mod flex_row_program;
pub(crate) mod gradient_paths;
pub(crate) mod hessian_paths;
mod information_third;
pub(crate) mod install_flex;
pub(crate) mod pilot_total_jacobian;
pub(crate) mod local_law_resolution;
pub(crate) mod moving_law_rule;
pub mod residual_repair;
mod residual_repair_kernel;
pub(crate) mod row_kernel;
#[cfg(test)]
mod tests_residual_repair_laws;
#[cfg(test)]
mod tests {
    include!("../../../../tests/src_modules/misc/families_bms_identifiability_rigid_tests.rs");
    include!(
        "../../../../tests/src_modules/optimization/families_bms_joint_hessian_hvp_correction_tests.rs"
    );

    #[test]
    fn empirical_grid_constructor_preserves_canonical_node_order() {
        let grid = EmpiricalZGrid::new(
            vec![-2.0, 0.5, 1.0],
            vec![0.3, 0.5, 0.2],
            "sorted-grid invariant",
        )
        .expect("canonical sorted grid");
        assert_eq!(grid.nodes, vec![-2.0, 0.5, 1.0]);
        assert_eq!(grid.weights, vec![0.3, 0.5, 0.2]);
    }

    #[test]
    fn empirical_grid_constructor_rejects_noncanonical_node_order() {
        let err = EmpiricalZGrid::new(vec![0.0, -1.0], vec![0.5, 0.5], "sorted-grid invariant")
            .expect_err("constructed grids must already be canonical");
        assert!(err.contains("nodes must be sorted ascending"), "{err}");
    }

    #[test]
    fn robust_score_test_statistic_is_the_score_quadratic_form() {
        // s = 1.5, Ω̂ = 1 + 1 + 0.25 = 2.25, so D = s²/Ω̂ = 1 on one direction.
        let contributions = ndarray::array![[1.0], [1.0], [-0.5]];
        let p = super::robust_score_contributions_pvalue(&contributions, 1)
            .expect("finite contributions")
            .expect("one usable direction");
        let expected = super::chi_square_sf(1.0, 1.0);
        assert!((p - expected).abs() <= 1e-12 * expected, "{p} vs {expected}");
    }

    #[test]
    fn robust_score_test_with_a_null_space_score_reports_p_one() {
        // The score is zero and Ω̂ = diag(2, 0) has rank 1: D = 0 is evidence of
        // no departure, p = 1, not a degenerate test.
        let contributions = ndarray::array![[1.0, 0.0], [-1.0, 0.0]];
        let p = super::robust_score_contributions_pvalue(&contributions, 1)
            .expect("finite contributions")
            .expect("one usable direction");
        assert_eq!(p, 1.0);
    }

    #[test]
    fn robust_score_test_refuses_a_non_finite_score() {
        let contributions = ndarray::array![[1.0e300, 1.0], [1.0e300, -1.0]];
        let err = super::robust_score_contributions_pvalue(&contributions, 1)
            .expect_err("an overflowed meat is a numerical failure, not a degenerate test");
        assert!(err.contains("not finite"), "{err}");
    }
}

#[cfg(test)]
mod stacked_first_stage_sandwich_2484_tests {
    use super::{
        first_stage_covariance_from_row_influence, preconditioned_normal_pseudoinverse,
        stacked_first_stage_row_influence,
    };
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};

    /// Compare the live row-influence covariance with an independently assembled
    /// stacked score meat and block inverse bread. The oracle deliberately does
    /// not call the production inverse-bread to build what it compares against:
    /// it rebuilds the variance stage's observed information
    /// `N = Σ w (û²/v) B Bᵀ`, its score `r = û²/v − 1` and the bread's cross
    /// block `M_vm = −2 Σ w (û/v) B Aᵀ` from `var_fitted` alone, which is what
    /// the log-linear variance stage makes them (gam#4019).
    fn assert_row_influence_matches_sandwich(
        mean_basis: ArrayView2<'_, f64>,
        var_basis: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        mean_residuals: &[f64],
        var_fitted: &[f64],
        mean_normal: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let psi = stacked_first_stage_row_influence(
            mean_basis,
            var_basis,
            weights,
            mean_residuals,
            var_fitted,
            mean_normal,
        )?;
        let actual = first_stage_covariance_from_row_influence(&psi)?;
        let n = mean_basis.nrows();
        let p = mean_basis.ncols();
        let q = var_basis.ncols();
        let m_inv = preconditioned_normal_pseudoinverse(mean_normal, n)?;
        let mut var_normal = Array2::<f64>::zeros((q, q));
        for i in 0..n {
            let info_scale = weights[i] * mean_residuals[i] * mean_residuals[i] / var_fitted[i];
            for j in 0..q {
                for k in 0..q {
                    var_normal[[j, k]] += info_scale * var_basis[[i, j]] * var_basis[[i, k]];
                }
            }
        }
        let v_inv = preconditioned_normal_pseudoinverse(&var_normal, n)?;
        let mut cross = Array2::<f64>::zeros((q, p));
        let mut meat = Array2::<f64>::zeros((p + q, p + q));
        for i in 0..n {
            let standardized = mean_residuals[i] / var_fitted[i];
            let var_residual = mean_residuals[i] * standardized - 1.0;
            let mut score = vec![0.0; p + q];
            for j in 0..p {
                score[j] = weights[i] * mean_residuals[i] * mean_basis[[i, j]];
            }
            for j in 0..q {
                score[p + j] = weights[i] * var_residual * var_basis[[i, j]];
                for k in 0..p {
                    cross[[j, k]] -=
                        2.0 * weights[i] * standardized * var_basis[[i, j]] * mean_basis[[i, k]];
                }
            }
            for j in 0..p + q {
                for k in 0..p + q {
                    meat[[j, k]] += score[j] * score[k];
                }
            }
        }
        let lower = v_inv.dot(&cross).dot(&m_inv);
        let mut inverse_bread = Array2::<f64>::zeros((p + q, p + q));
        for j in 0..p {
            for k in 0..p { inverse_bread[[j, k]] = m_inv[[j, k]]; }
        }
        for j in 0..q {
            for k in 0..p { inverse_bread[[p + j, k]] = lower[[j, k]]; }
            for k in 0..q { inverse_bread[[p + j, p + k]] = v_inv[[j, k]]; }
        }
        let expected = inverse_bread.dot(&meat).dot(&inverse_bread.t());
        let scale = expected.iter().fold(0.0_f64, |v, x| v.max(x.abs()));
        let allowance = 64.0 * (n + p + q) as f64 * f64::EPSILON * scale;
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= allowance,
                "row influence covariance {a:e} differs from independent sandwich {e:e}");
        }
        Ok(actual)
    }

    /// The standalone HC0 sandwich `M⁺ (Σ w² e² A Aᵀ) M⁺` of one stage with
    /// score `Σ w e A` and information `M` -- the block-diagonal form the joint
    /// sandwich replaced.
    fn standalone_sandwich(
        basis: &Array2<f64>,
        residuals: &[f64],
        weights: &Array1<f64>,
        m: &Array2<f64>,
    ) -> Array2<f64> {
        let mut scores = basis.clone();
        for (i, mut row) in scores.rows_mut().into_iter().enumerate() {
            let scale = weights[i] * residuals[i];
            row.iter_mut().for_each(|value| *value *= scale);
        }
        let m_pinv = preconditioned_normal_pseudoinverse(m, basis.nrows()).expect("M⁺");
        m_pinv.dot(&scores.t().dot(&scores)).dot(&m_pinv)
    }

    /// `A`, weights, and a normal matrix `M = AᵀWA + λR` built exactly the way
    /// the calibration builds it, so the sandwich is exercised on a realistic
    /// bread rather than on an identity.
    fn system(basis: &Array2<f64>, weights: &Array1<f64>) -> Array2<f64> {
        let mut wa = basis.clone();
        for (mut row, &w) in wa.rows_mut().into_iter().zip(weights.iter()) {
            row.iter_mut().for_each(|value| *value *= w);
        }
        let mut m = basis.t().dot(&wa);
        let diag: Vec<f64> = (0..m.nrows()).map(|j| m[[j, j]]).collect();
        for (j, value) in diag.iter().enumerate() {
            m[[j, j]] += value * 1.0e-8;
        }
        m
    }

    /// A log-linear fitted variance `v = exp(0.1 + 0.2·a)` that depends on the
    /// row only through its basis row, so it is equal within a pair.
    fn var_fitted(basis: &Array2<f64>) -> Vec<f64> {
        basis
            .rows()
            .into_iter()
            .map(|row| (0.1 * row[0] + 0.2 * row[1]).exp())
            .collect()
    }

    /// The variance stage's own quantities: the observed information
    /// `N = Σ w (û²/v) B Bᵀ` and the score residual `r = û²/v − 1`.
    fn variance_stage(
        basis: &Array2<f64>,
        weights: &Array1<f64>,
        mean_residuals: &[f64],
        var_fitted: &[f64],
    ) -> (Array2<f64>, Vec<f64>) {
        let mut scaled = basis.clone();
        for (i, mut row) in scaled.rows_mut().into_iter().enumerate() {
            let scale = weights[i] * mean_residuals[i] * mean_residuals[i] / var_fitted[i];
            row.iter_mut().for_each(|value| *value *= scale);
        }
        let residuals = mean_residuals
            .iter()
            .zip(var_fitted.iter())
            .map(|(&u, &v)| u * u / v - 1.0)
            .collect();
        (basis.t().dot(&scaled), residuals)
    }

    /// PAIRED fixture: every conditioning row appears twice with equal weight
    /// and mean residuals `+c` / `−c`.
    ///
    /// That makes both cross-terms cancel EXACTLY rather than approximately —
    /// the bread's `M_vm = −2Σ w·(û/v)·A Aᵀ` cancels because `û` flips sign
    /// while `v` and `A Aᵀ` do not, and the meat's `Σ w²·û·r·A Aᵀ` cancels
    /// because `r = û²/v − 1` is equal across the pair. So this arm asserts an
    /// identity, not a tolerance: with no third moment, the joint sandwich must
    /// reproduce the block-diagonal form the code used to assume.
    #[test]
    fn symmetric_residuals_reproduce_the_block_diagonal_form_2484() {
        let basis = array![
            [1.0, 0.4],
            [1.0, 0.4],
            [1.0, -0.7],
            [1.0, -0.7],
            [1.0, 1.3],
            [1.0, 1.3],
        ];
        let weights = Array1::from(vec![1.0, 1.0, 0.5, 0.5, 2.0, 2.0]);
        let mean_residuals = vec![0.6, -0.6, 0.9, -0.9, 0.3, -0.3];
        let var_fitted = var_fitted(&basis);
        let (n_info, var_residuals) =
            variance_stage(&basis, &weights, &mean_residuals, &var_fitted);
        let m = system(&basis, &weights);

        let joint = assert_row_influence_matches_sandwich(
            basis.view(),
            basis.view(),
            weights.view(),
            &mean_residuals,
            &var_fitted,
            &m,
        )
        .expect("joint sandwich");
        let mean_block = standalone_sandwich(&basis, &mean_residuals, &weights, &m);
        let var_block = standalone_sandwich(&basis, &var_residuals, &weights, &n_info);

        let p = basis.ncols();
        for i in 0..p {
            for j in 0..p {
                let tol = 1.0e-9 * (1.0 + mean_block[[i, j]].abs());
                assert!(
                    (joint[[i, j]] - mean_block[[i, j]]).abs() <= tol,
                    "gam#2484: with no third moment the (mean,mean) block must reproduce the \
                     standalone sandwich; got {} vs {}",
                    joint[[i, j]],
                    mean_block[[i, j]]
                );
                let tol_v = 1.0e-9 * (1.0 + var_block[[i, j]].abs());
                assert!(
                    (joint[[p + i, p + j]] - var_block[[i, j]]).abs() <= tol_v,
                    "gam#2484: with no third moment the (var,var) block must reproduce the \
                     standalone sandwich; got {} vs {}",
                    joint[[p + i, p + j]],
                    var_block[[i, j]]
                );
                assert!(
                    joint[[i, p + j]].abs() <= 1.0e-9,
                    "gam#2484: the cross-block must vanish when the residual third moment does; \
                     got {}",
                    joint[[i, p + j]]
                );
            }
        }
    }

    /// The arm that asserts the change DOES something. Break the pairing so the
    /// residual carries a third moment, and the cross-block must be measurably
    /// non-zero — otherwise an implementation that quietly returns the old
    /// block-diagonal form passes the test above and ships.
    #[test]
    fn skewed_residuals_make_the_cross_block_nonzero_2484() {
        let basis = array![
            [1.0, 0.4],
            [1.0, 0.9],
            [1.0, -0.7],
            [1.0, 0.2],
            [1.0, 1.3],
            [1.0, -1.1],
        ];
        let weights = Array1::from(vec![1.0, 1.0, 0.5, 0.5, 2.0, 2.0]);
        // Strongly right-skewed mean residuals: one large positive, the rest
        // small negative. This is the shape the adequacy gate rejects.
        let mean_residuals = vec![-0.2, -0.3, -0.25, -0.15, 2.4, -0.35];
        let var_fitted = var_fitted(&basis);
        let (n_info, var_residuals) =
            variance_stage(&basis, &weights, &mean_residuals, &var_fitted);
        let m = system(&basis, &weights);

        let joint = assert_row_influence_matches_sandwich(
            basis.view(),
            basis.view(),
            weights.view(),
            &mean_residuals,
            &var_fitted,
            &m,
        )
        .expect("joint sandwich");
        let var_block = standalone_sandwich(&basis, &var_residuals, &weights, &n_info);

        let p = basis.ncols();
        let cross = (0..p)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| joint[[i, p + j]].abs())
            .fold(0.0_f64, f64::max);
        let scale = (0..p)
            .map(|i| joint[[i, i]].abs().max(joint[[p + i, p + i]].abs()))
            .fold(0.0_f64, f64::max);
        assert!(
            cross > 1.0e-6 * scale.max(1.0e-12),
            "gam#2484: a skewed residual must produce a NON-ZERO first-stage cross-block. \
             max|cross|={cross:e} against block scale {scale:e}. If this fails, the \
             implementation is still returning a block-diagonal V1 and the whole change is \
             inert."
        );

        // And the (var,var) block must differ from the standalone sandwich: the
        // triangular bread feeds the mean-stage uncertainty into it.
        let var_shift = (0..p)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| (joint[[p + i, p + j]] - var_block[[i, j]]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            var_shift > 1.0e-6 * scale.max(1.0e-12),
            "gam#2484: the (var,var) block must absorb the mean-stage uncertainty through the \
             lower-triangular bread; max shift {var_shift:e} is indistinguishable from the old \
             standalone block"
        );
    }
}

#[cfg(test)]
mod anchor_law_2926_tests;
pub(crate) mod axis_direction_search;
pub(crate) mod cell_moment_assembly;
#[cfg(test)]
mod closed_form_certificate_2926_tests;
#[cfg(test)]
mod conditional_law_gate_tests;
#[cfg(test)]
mod first_stage_variance_stage_tests;
#[cfg(test)]
mod empirical_intercept_solve_tests;
#[cfg(test)]
mod empirical_measure_2484_tests;
pub(crate) mod empirical_measure_sensitivity;
#[cfg(test)]
mod empirical_grid_sampling_3452_tests;
#[cfg(test)]
mod empirical_grid_fit_3452_tests;
#[cfg(test)]
mod local_law_replay_2929_tests;
#[cfg(test)]
mod residual_score_zeta_2985_tests;
#[cfg(test)]
mod normal_screen_2926_tests;
mod standard_normal_flex_fifth;
// #932 BMS flex single-source jet substrate (runtime-dimension `Jet2` + IFT
// lift + cell base-moment jets). A bare `#[cfg(test)] mod` with an allowed name
// so the build.rs ban-scanner exempts it; shared by its own FD gates and the
// `cell_moment_assembly` flex-fixture oracle gate as a private child of `bms`.
#[cfg(test)]
mod test_support;
// #932 INDEPENDENT adversarial verifier (bms-flex-verify): a high-order
// finite-difference oracle on the production compiled lowering
// `lower_bms_flex_row_order2_from_parts` of the canonical BMS FLEX program,
// plus a moving-edge Leibniz
// cross-check + a planted-corruption tripwire. Bare `#[cfg(test)] mod` with the
// allowed `*_tests` name so the build.rs ban-scanner exempts it; owned solely by
// the verifier (never edits the implementer's row_primary_hessian /
// gradient_paths / cell_moment_assembly).
pub(crate) mod custom_family_impl;
#[cfg(test)]
mod flex_verify_932_tests;
// #932 direct production-path measurement: forced 65-node empirical grid,
// warmed/cold row-op allocation counting + ns/row diagnostics for the MSI
// A/B ledger. The asserted gate is per-row allocation calls (deterministic);
// timing is eprintln-only per the SPEC ban on wall-clock correctness budgets.
#[cfg(test)]
mod flex_measure_932_tests;
#[cfg(test)]
mod third_trace_2998_tests;
// gam#2768 unit gates on the shared latent-measure decision and the conditional
// location-scale calibration it escalates to. Bare `#[cfg(test)] mod` with the
// allowed `*_tests` name so the build.rs ban-scanner exempts it.
#[cfg(test)]
mod latent_measure_2768_tests;
// gam#979: the rigid ψ axis contractions against the materialized all-beta-axes
// tensors. Bare `#[cfg(test)] mod` with the allowed `*_tests` name.
#[cfg(test)]
mod psi_axis_contractions_979_tests;
// gnomon#2359: a multistart member reuses only its own same-β stores. Bare
// `#[cfg(test)] mod` with the allowed `*_tests` name.
#[cfg(test)]
mod multistart_member_2359_tests;
// gam#3022: the rigid row kernel's per-row tensor tables. Bare
// `#[cfg(test)] mod` with the allowed `*_tests` name.
#[cfg(test)]
mod rigid_row_tensors_3022_tests;
pub(crate) mod row_primary_hessian;
mod second_correction_traces;

pub(crate) use block_specs::finite_law_in_standard_units;
pub(crate) use block_specs::fit_bernoulli_marginal_slope_terms;
pub use conditional_score_covariance::{
    ConditionalScoreCoordinate, ConditionalScoreCovariance, ScoreCovarianceField,
};
pub use gradient_paths::{
    MarginalSlopeCovariance, MarginalSlopeCovarianceShape, marginal_slope_covariance_from_scores,
    padded_deviation_seed,
};
pub use install_flex::CrossBlockIdentifiabilityWarning;
pub(crate) use install_flex::FlexCompileOutcome;
pub(crate) use residual_repair::residual_row_index;
pub use residual_repair::{
    RESIDUAL_BLOCK_NAME, ResidualBlockRuntime, ResidualRepairGeometry, ResidualRepairRefusal,
    ResidualRepairSpec,
};

// pub(crate) re-exports for internal callers:
pub(crate) use block_specs::push_deviation_aux_blockspecs;
pub use block_specs::{BmsMarginalJacobian, BmsSlopeJacobian};
pub(crate) use family::{
    BernoulliMarginalLinkMap, bernoulli_marginal_link_map,
    build_link_deviation_block_from_knots_design_seed_and_weights,
    build_score_warp_deviation_block_from_seed,
};
pub(crate) use gradient_paths::MarginalSlopeCovarianceRef;
pub(crate) use gradient_paths::standardize_latent_z_with_policy;
pub(crate) use gradient_paths::weighted_location_scale;
pub(crate) use gradient_paths::{
    empirical_intercept, signed_probit_neglog_derivatives_up_to_fourth,
    unary_derivatives_inverse_sqrt, unary_derivatives_log, unary_derivatives_log_normal_pdf,
    unary_derivatives_neglog_phi, unary_derivatives_sqrt,
};
pub(crate) use install_flex::{
    install_bms_flex_block_on_total_jacobian, install_compiled_flex_block_into_runtime,
    project_monotone_feasible_beta,
};
