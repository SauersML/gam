use crate::bms::{
    BernoulliMarginalSlopeSavedAloReplay, BernoulliMarginalSlopeSavedAloReplayInput,
    EmpiricalZGrid, LatentMeasureKind, LatentZConditionalCalibration, LatentZRankIntCalibration,
    bernoulli_marginal_link_map, empirical_intercept, replay_saved_bernoulli_marginal_slope_alo,
};
use crate::inference::model::{SavedCompiledFlexBlock, SavedLatentZNormalization};
use crate::latent_anchor::{CalibrationTail, smaller_tail_log_target, solve_log_tail_root};
use crate::marginal_slope_shared::{
    ObservedDenestedCellPartials, eval_coeff4_at,
    probit_frailty_scale as marginal_slope_probit_frailty_scale, scale_coeff4,
};
use crate::survival::lognormal_kernel::{FrailtyScale, FrailtySpec};
use gam_linalg::matrix::DesignMatrix;
use gam_math::probability::{normal_cdf, normal_pdf};
use gam_problem::types::{InverseLink, LikelihoodSpec};
use gam_runtime::resource::prediction_chunk_rows;
use gam_solve::estimate::{EstimationError, UnifiedFitResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;

pub struct PredictResult {
    pub eta: Array1<f64>,
    pub mean: Array1<f64>,
}

/// Input to prediction routines. Contains the design matrix and metadata
/// needed for point prediction plus uncertainty quantification.
pub struct PredictInput {
    /// Design matrix for the primary (mean/location) block.
    pub design: DesignMatrix,
    /// Offset vector for the primary block.
    pub offset: Array1<f64>,
    /// Optional design matrix for the noise/scale block (GAMLSS/survival).
    pub design_noise: Option<DesignMatrix>,
    /// Optional offset vector for the noise/scale block.
    pub offset_noise: Option<Array1<f64>>,
    /// Optional auxiliary scalar covariate used by specialized predictors.
    pub auxiliary_scalar: Option<Array1<f64>>,
    /// Optional auxiliary matrix used by specialized predictors.
    pub auxiliary_matrix: Option<Array2<f64>>,
}

/// Where the conditional latent calibration's conditioning span `a(C)` lives
/// inside a [`PredictInput`] for a given host family (gam#2768).
///
/// The fit regressed z on the *marginal-index* design and prediction must
/// reproduce that exact span — a different set of columns is a different map, and
/// silently applying it would evaluate a different model from the one that was
/// fitted. The two marginal-slope hosts package that span differently:
///
/// * the Bernoulli predictor's primary design **is** the marginal design;
/// * the survival predictor's primary design is the q-design
///   `[time | timewiggle | marginal]`, because q is a function of time as well as
///   of the covariates. Its marginal-index span is therefore the trailing block,
///   not the whole design.
///
/// Naming that difference here is what lets one predictor serve both without
/// either guessing. The width is cross-checked against the calibration's own
/// `basis_ncols` on every application, so a host that names the wrong block is
/// refused rather than silently mis-conditioned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentConditioningSpan {
    /// The whole primary design is the conditioning span.
    PrimaryDesign,
    /// The conditioning span is the trailing `ncols` columns of the primary
    /// design.
    PrimaryDesignTail { ncols: usize },
}

/// The maps a fit applied to its latent score before any kernel read it: the
/// saved normalisation, then either the rank-INT calibration an older model
/// replays or the conditional location-scale calibration
/// `ζ = (z − m(a))/√v(a)` (#905, gam#2926). A fit mints at most one of the two.
///
/// This is the one owner of that composition (gam#3016). The fit computes the
/// score its kernel is fitted on through it ([`Self::apply_on_span`] on the
/// training rows, [`Self::calibrate`] wherever the fit applies a calibration it
/// just estimated), the saved marginal-slope predictor reads its kernel score
/// through it, and so does `FittedModel::latent_conditional_residual`, which
/// returns ζ for new rows.
#[derive(Clone, Copy)]
pub(crate) struct FittedLatentScoreMap<'a> {
    pub(crate) normalization: &'a SavedLatentZNormalization,
    pub(crate) rank_int: Option<&'a LatentZRankIntCalibration>,
    pub(crate) conditional: Option<&'a LatentZConditionalCalibration>,
    /// Where `primary_design` carries the conditioning span `a`.
    pub(crate) span: LatentConditioningSpan,
}

/// The normalisation of a score that is already on its fitted scale.
static IDENTITY_NORMALIZATION: SavedLatentZNormalization =
    SavedLatentZNormalization { mean: 0.0, sd: 1.0 };

impl<'a> FittedLatentScoreMap<'a> {
    /// The conditional location-scale step alone, for a fit applying the
    /// calibration it just estimated to scores it has already normalised.
    pub(crate) fn conditional_only(conditional: &'a LatentZConditionalCalibration) -> Self {
        Self {
            normalization: &IDENTITY_NORMALIZATION,
            rank_int: None,
            conditional: Some(conditional),
            span: LatentConditioningSpan::PrimaryDesign,
        }
    }
}

impl FittedLatentScoreMap<'_> {
    /// The fitted latent score of each row: `z_raw` normalised, then calibrated.
    pub(crate) fn apply(
        &self,
        z_raw: &Array1<f64>,
        primary_design: &DesignMatrix,
        context: &str,
    ) -> Result<Array1<f64>, EstimationError> {
        let design = self.conditional.map(|_| primary_design.to_dense());
        let a_block = design
            .as_ref()
            .map(|design| self.conditioning_span(design.view()))
            .transpose()?;
        self.apply_on_span(z_raw, a_block, context)
    }

    /// The fitted latent score of each row with the conditioning span `a`
    /// already in hand, as the fit has it on its training rows. `a_block` is
    /// read only by a conditional calibration, which refuses its absence.
    pub(crate) fn apply_on_span(
        &self,
        z_raw: &Array1<f64>,
        a_block: Option<ArrayView2<'_, f64>>,
        context: &str,
    ) -> Result<Array1<f64>, EstimationError> {
        let normalized = self
            .normalization
            .apply(z_raw, context)
            .map_err(EstimationError::from)?;
        self.calibrate(normalized.view(), a_block)
            .map_err(EstimationError::InvalidInput)
    }

    /// The calibration steps on normalised scores: the rank-INT or the
    /// conditional location-scale map, whichever the fit minted.
    pub(crate) fn calibrate(
        &self,
        z: ArrayView1<'_, f64>,
        a_block: Option<ArrayView2<'_, f64>>,
    ) -> Result<Array1<f64>, String> {
        let z = self.rank_int_step(z);
        let Some(cal) = self.conditional else {
            return Ok(z);
        };
        let a_block = a_block.ok_or_else(|| {
            "conditional latent calibration needs its conditioning span a, and none was supplied"
                .to_string()
        })?;
        cal.apply(z.view(), a_block)
    }

    /// The columns of the dense primary design that carry `a`.
    fn conditioning_span<'d>(
        &self,
        design: ArrayView2<'d, f64>,
    ) -> Result<ArrayView2<'d, f64>, EstimationError> {
        match self.span {
            LatentConditioningSpan::PrimaryDesign => Ok(design),
            LatentConditioningSpan::PrimaryDesignTail { ncols } => {
                let width = design.ncols();
                if ncols > width {
                    return Err(EstimationError::InvalidInput(format!(
                        "conditional latent calibration names the trailing {ncols} columns of the \
                         primary design as its conditioning span, but that design has only \
                         {width} columns"
                    )));
                }
                Ok(design.slice_move(ndarray::s![.., width - ncols..]))
            }
        }
    }

    /// The rank-INT step on normalised scores, or the identity when the fit
    /// minted none.
    fn rank_int_step(&self, z: ArrayView1<'_, f64>) -> Array1<f64> {
        match self.rank_int {
            Some(cal) => z.mapv(|zi| cal.apply_at_predict(zi)),
            None => z.to_owned(),
        }
    }

    /// The conditional location-scale step, reading `a` from `primary_design` at
    /// `self.span`, or the identity when the fit minted none.
    fn conditional_step(
        &self,
        z: &Array1<f64>,
        primary_design: &DesignMatrix,
    ) -> Result<Array1<f64>, EstimationError> {
        let Some(cal) = self.conditional else {
            return Ok(z.clone());
        };
        let design = primary_design.to_dense();
        cal.apply(z.view(), self.conditioning_span(design.view())?)
            .map_err(EstimationError::InvalidInput)
    }
}

/// One prediction row's anchored marginal-slope kernel: everything the rigid
/// (standard-normal) or declared-law (empirical) intercept calibration needs
/// besides the two primaries `(q, b)` it is anchored on.
///
/// [`Self::eta`] re-solves the anchor at the supplied primaries, so a caller
/// integrating over the coefficient posterior evaluates the *anchored* model at
/// every node instead of linearising it at `θ̂`. Built by
/// [`BernoulliMarginalSlopePredictor::anchored_row_kernels`].
pub struct AnchoredRowKernel {
    z: f64,
    probit_scale: f64,
    base_link: InverseLink,
    /// `None` is the rigid standard-normal law; `Some` is the declared
    /// empirical law (global, or this row's local mixture).
    grid: Option<EmpiricalZGrid>,
}

impl AnchoredRowKernel {
    /// The calibrated latent score of this row.
    pub fn latent_z(&self) -> f64 {
        self.z
    }

    /// The probit frailty scale `s` multiplying the slope in the kernel.
    pub fn probit_scale(&self) -> f64 {
        self.probit_scale
    }

    /// The base-scale linear predictor at primaries `(q, b)`, with the anchor
    /// re-solved there: `η = c(b)·q + s·b·z` under the standard-normal law and
    /// `η = a(q, b) + s·b·z` under an empirical law, `a` being the root of
    /// `Σ wᵢ Φ(a + s·b·zᵢ) = Φ(q)`. The same formulas
    /// [`BernoulliMarginalSlopePredictor::final_eta_from_theta`] evaluates at
    /// `θ̂`. A posterior node can sit several standard deviations into the tail
    /// of `q`; the root is solved from `q` in log space on the smaller tail, so
    /// it resolves there like anywhere else (gam#2978).
    pub fn eta(&self, q: f64, b: f64) -> Result<f64, EstimationError> {
        let sb = self.probit_scale * b;
        match &self.grid {
            None => Ok((1.0 + sb * sb).sqrt() * q + sb * self.z),
            Some(grid) => {
                let marginal = bernoulli_marginal_link_map(&self.base_link, q)
                    .map_err(EstimationError::InvalidInput)?;
                let intercept = empirical_intercept(
                    marginal.q,
                    b,
                    self.probit_scale,
                    &grid.nodes,
                    &grid.weights,
                )
                .map_err(EstimationError::InvalidInput)?;
                Ok(intercept + sb * self.z)
            }
        }
    }

    /// [`Self::eta`] together with its partials `(η, ∂η/∂q, ∂η/∂b)` at the
    /// same primaries, the anchor re-solved there: `∂η/∂q = c(b)` and
    /// `∂η/∂b = s²·b·q/c(b) + s·z` under the standard-normal law, and the
    /// implicit-function derivatives `a_q = μ′(q)/F_a`, `a_b = −F_b/F_a` of the
    /// calibrated intercept (plus `s·z` on `b`) under an empirical law. These
    /// are the partials [`BernoulliMarginalSlopePredictor::predict_eta_and_time_tangent`]
    /// chains a time tangent through.
    pub fn eta_and_partials(&self, q: f64, b: f64) -> Result<(f64, f64, f64), EstimationError> {
        let scale = self.probit_scale;
        let sb = scale * b;
        match &self.grid {
            None => {
                let c = (1.0 + sb * sb).sqrt();
                Ok((
                    c * q + sb * self.z,
                    c,
                    scale * scale * b * q / c + scale * self.z,
                ))
            }
            Some(grid) => {
                let marginal = bernoulli_marginal_link_map(&self.base_link, q)
                    .map_err(EstimationError::InvalidInput)?;
                let (intercept, a_q, a_b) = empirical_intercept_and_partials(
                    marginal.q,
                    marginal.q1,
                    b,
                    scale,
                    &grid.nodes,
                    &grid.weights,
                )?;
                Ok((intercept + sb * self.z, a_q, a_b + scale * self.z))
            }
        }
    }
}

/// The empirical-law intercept `a`, the root of `Σ wᵢ Φ(a + s·b·zᵢ) = Φ(q)`,
/// with its partials `(∂a/∂η, ∂a/∂b)` in the marginal index `η` (through
/// `q′(η) = marginal_q1`) and the slope `b`. The root and its derivatives are
/// the latent anchor's log-space solve and Taylor table at the observed slope
/// `s·b`, normalized by the grid density, so a tail index keeps its exact
/// partials (gam#2978).
fn empirical_intercept_and_partials(
    q: f64,
    marginal_q1: f64,
    slope: f64,
    probit_scale: f64,
    nodes: &[f64],
    weights: &[f64],
) -> Result<(f64, f64, f64), EstimationError> {
    let observed_slope = probit_scale * slope;
    let grid = crate::latent_anchor::AnchorGridOwned::new(nodes.to_vec(), weights.to_vec());
    let derivatives = crate::latent_anchor::solve_anchor(q, observed_slope, grid.view())
        .and_then(|alpha| {
            crate::latent_anchor::AnchorTaylor::at(alpha, q, observed_slope, grid.view())
        })
        .map_err(EstimationError::InvalidInput)?
        .derivatives();
    Ok((
        derivatives.alpha,
        derivatives.a_q * marginal_q1,
        derivatives.a_b * probit_scale,
    ))
}

pub struct BernoulliMarginalSlopePredictor {
    pub beta_marginal: Array1<f64>,
    pub beta_slope: Array1<f64>,
    pub beta_score_warp: Option<Array1<f64>>,
    pub beta_link_dev: Option<Array1<f64>>,
    pub base_link: InverseLink,
    pub z_column: String,
    pub latent_z_normalization: SavedLatentZNormalization,
    pub latent_measure: LatentMeasureKind,
    pub baseline_marginal: f64,
    pub baseline_slope: f64,
    pub covariance: Option<Array2<f64>>,
    pub score_warp_runtime: Option<SavedCompiledFlexBlock>,
    pub link_deviation_runtime: Option<SavedCompiledFlexBlock>,
    pub gaussian_frailty_sd: Option<f64>,
    pub latent_z_calibration: Option<LatentZRankIntCalibration>,
    pub latent_z_conditional_calibration: Option<LatentZConditionalCalibration>,
    /// Which block of [`PredictInput::design`] carries the conditioning span the
    /// conditional calibration was fit against. Ignored when
    /// `latent_z_conditional_calibration` is `None`.
    pub latent_conditioning_span: LatentConditioningSpan,
    /// The residual genetic repair block (gam#2924): the fitted geometry that
    /// replays the joint `(z, r)` anchor, and its coefficients (block 2 of the
    /// fit). `None` is the single-score predictor unchanged. The prediction
    /// rows' residual features arrive in [`PredictInput::auxiliary_matrix`].
    pub residual_repair: Option<crate::bms::ResidualRepairGeometry>,
    pub beta_residual: Option<Array1<f64>>,
}

/// Saved marginal-slope affine row coordinates after replaying every fitted
/// latent-score transformation.
struct BernoulliMarginalSlopeSavedAloAffineState {
    marginal_eta: Array1<f64>,
    slope: Array1<f64>,
    latent_z: Array1<f64>,
}

/// Per-runtime predict-time anchor correction matrices.
///
/// Built once per top-level predict call from the marginal + slope
/// designs at the prediction rows. Each `Array2<f64>` is shaped
/// `n_predict × runtime.basis_dim` and holds `n_row(i) · M` for every
/// prediction row, where `n_row(i)` is the concatenation of the marginal
/// and slope design rows in the runtime's anchor component order.
///
/// At any `local_cubic_at` / `basis_cubic_at` / `design` call site we
/// subtract the appropriate slice of these matrices from the raw cubic
/// output to apply the cross-block residual `n_row · M` correction.
///
/// `n_anchor_rows` is the underlying `n × d` parametric anchor stack
/// (per-runtime layouts: score_warp gets `[marginal | slope]`;
/// link_dev gets `[marginal | slope | score_warp_design(z)]` when the
/// fit-time identifiability stage threaded the score-warp basis in as a
/// flex-evaluation anchor). These layouts must match the column order
/// `install_compiled_flex_block_into_runtime` used at fit time.
#[derive(Default)]
struct BmsAnchorCorrections {
    /// `[marginal | slope]` at predict rows. `Some` whenever any
    /// runtime carries an anchor residual.
    score_warp_anchor_rows: Option<Array2<f64>>,
    /// `[marginal | slope | score_warp_design(z)]` at predict rows.
    /// `Some` whenever the link-deviation runtime carries an anchor
    /// residual; the score-warp tail is included iff the saved
    /// link-deviation runtime's residual components include a
    /// `FlexEvaluation` entry.
    link_dev_anchor_rows: Option<Array2<f64>>,
    score_warp: Option<Array2<f64>>,
    link_dev: Option<Array2<f64>>,
}

impl BmsAnchorCorrections {
    fn score_warp_row(&self, row: usize) -> Option<ndarray::ArrayView1<'_, f64>> {
        self.score_warp.as_ref().map(|m| m.row(row))
    }

    fn link_dev_row(&self, row: usize) -> Option<ndarray::ArrayView1<'_, f64>> {
        self.link_dev.as_ref().map(|m| m.row(row))
    }

    fn score_warp_anchor_rows_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        self.score_warp_anchor_rows.as_ref().map(|m| m.view())
    }

    fn link_dev_anchor_rows_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        self.link_dev_anchor_rows.as_ref().map(|m| m.view())
    }
}

impl BernoulliMarginalSlopePredictor {
    /// Build the anchor correction matrices for a given predict-input batch.
    ///
    /// Returns an empty bundle (all `None`) when neither runtime carries
    /// an anchor residual — this is the fast path for fits without
    /// cross-block residualisation. When at least one runtime has a
    /// residual, materialises the marginal + slope designs at the
    /// predict rows once and computes the per-runtime correction matrices
    /// against each runtime's stored `M`.
    fn build_anchor_correction_matrices(
        &self,
        input: &PredictInput,
        design_slope: &DesignMatrix,
        z: &Array1<f64>,
    ) -> Result<BmsAnchorCorrections, EstimationError> {
        use crate::inference::model::SavedAnchorKind;
        let needs_score = self
            .score_warp_runtime
            .as_ref()
            .is_some_and(|r| r.anchor_correction.is_some());
        let needs_link = self
            .link_deviation_runtime
            .as_ref()
            .is_some_and(|r| r.anchor_correction.is_some());
        if !needs_score && !needs_link {
            return Ok(BmsAnchorCorrections::default());
        }
        // Materialise the marginal + slope designs at predict rows.
        // For large-scale predict batches the caller already chunks via
        // `prediction_chunk_rows`, so this densification is bounded per
        // chunk by `chunk_size × (p_marginal + p_slope)`.
        let marginal_dense = input
            .design
            .try_to_dense_arc(
                "bernoulli marginal-slope predict-time marginal anchor materialisation",
            )
            .map_err(EstimationError::InvalidInput)?;
        let slope_dense = design_slope
            .try_to_dense_arc(
                "bernoulli marginal-slope predict-time slope anchor materialisation",
            )
            .map_err(EstimationError::InvalidInput)?;
        let n_rows = marginal_dense.nrows();
        if slope_dense.nrows() != n_rows {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope predict anchor materialisation row mismatch: marginal {} vs slope {}",
                n_rows,
                slope_dense.nrows()
            )));
        }
        if z.len() != n_rows {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope predict anchor materialisation: z has {} entries, expected {}",
                z.len(),
                n_rows
            )));
        }
        let p_marginal = marginal_dense.ncols();
        let p_slope = slope_dense.ncols();
        let d_parametric = p_marginal + p_slope;
        let mut parametric_rows = Array2::<f64>::zeros((n_rows, d_parametric));
        parametric_rows
            .slice_mut(ndarray::s![.., 0..p_marginal])
            .assign(&marginal_dense.view());
        parametric_rows
            .slice_mut(ndarray::s![.., p_marginal..d_parametric])
            .assign(&slope_dense.view());

        // Score-warp anchor layout is `[marginal | slope]` (parametric
        // only; flex-flex anchoring goes the other direction).
        let score_warp = if needs_score {
            let runtime = self
                .score_warp_runtime
                .as_ref()
                .expect("needs_score is derived from this runtime being present");
            self.validate_runtime_anchor_layout_parametric_only(runtime, "score_warp")?;
            runtime
                .anchor_correction_matrix(parametric_rows.view())
                .map_err(EstimationError::from)?
        } else {
            None
        };

        // Link-deviation anchor layout matches the fit-time stacking in
        // `install_compiled_flex_block_into_runtime`: parametric
        // columns first, then (if a FlexEvaluation component is present)
        // the score-warp runtime's reparameterised basis at predict rows.
        let (link_dev_anchor_rows, link_dev) = if needs_link {
            let runtime = self
                .link_deviation_runtime
                .as_ref()
                .expect("needs_link is derived from this runtime being present");
            // Determine whether the saved link-dev residual carries a
            // FlexEvaluation tail and validate ordering matches the
            // fit-time invariant (all parametric components first, then
            // at most one FlexEvaluation tail).
            let mut saw_flex_tail = false;
            let mut flex_tail_ncols: usize = 0;
            for (idx, component) in runtime.anchor_components.iter().enumerate() {
                match &component.kind {
                    SavedAnchorKind::Parametric { .. } => {
                        if saw_flex_tail {
                            return Err(EstimationError::InvalidInput(format!(
                                "bernoulli marginal-slope link-deviation saved anchor components \
                                 are out of order: parametric component at index {idx} follows \
                                 a FlexEvaluation tail",
                            )));
                        }
                    }
                    SavedAnchorKind::FlexEvaluation { ncols } => {
                        if saw_flex_tail {
                            return Err(EstimationError::InvalidInput(
                                "bernoulli marginal-slope link-deviation saved anchor components \
                                 carry more than one FlexEvaluation tail; fit-time stacking emits \
                                 at most one (score-warp)"
                                    .to_string(),
                            ));
                        }
                        saw_flex_tail = true;
                        flex_tail_ncols = *ncols;
                    }
                }
            }
            let rows = if saw_flex_tail {
                let score_runtime = self.score_warp_runtime.as_ref().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "bernoulli marginal-slope link-deviation saved anchor includes a \
                         FlexEvaluation tail but the saved score-warp runtime is missing"
                            .to_string(),
                    )
                })?;
                // Evaluate the score-warp runtime at predict-row z. When
                // the score-warp itself carries an anchor residual, route
                // through `design_with_anchor_rows` so the per-row
                // subtraction is applied; otherwise the raw `design(z)`
                // is the reparameterised basis.
                let score_basis = if score_runtime.anchor_correction.is_some() {
                    score_runtime
                        .design_with_anchor_rows(z, parametric_rows.view())
                        .map_err(EstimationError::from)?
                } else {
                    score_runtime.design(z).map_err(EstimationError::from)?
                };
                if score_basis.ncols() != flex_tail_ncols {
                    return Err(EstimationError::InvalidInput(format!(
                        "bernoulli marginal-slope link-deviation FlexEvaluation tail expects \
                         {} score-warp basis columns at predict rows, got {}",
                        flex_tail_ncols,
                        score_basis.ncols()
                    )));
                }
                let mut combined = Array2::<f64>::zeros((n_rows, d_parametric + flex_tail_ncols));
                combined
                    .slice_mut(ndarray::s![.., 0..d_parametric])
                    .assign(&parametric_rows.view());
                combined
                    .slice_mut(ndarray::s![.., d_parametric..])
                    .assign(&score_basis.view());
                combined
            } else {
                parametric_rows.clone()
            };
            let corr = runtime
                .anchor_correction_matrix(rows.view())
                .map_err(EstimationError::from)?;
            (Some(rows), corr)
        } else {
            (None, None)
        };

        Ok(BmsAnchorCorrections {
            score_warp_anchor_rows: Some(parametric_rows),
            link_dev_anchor_rows,
            score_warp,
            link_dev,
        })
    }

    /// Validate that a saved deviation runtime's anchor residual contains
    /// only `Parametric` components (no `FlexEvaluation` tail). Used for
    /// the score-warp runtime, whose fit-time stacking is parametric-only.
    fn validate_runtime_anchor_layout_parametric_only(
        &self,
        runtime: &SavedCompiledFlexBlock,
        runtime_label: &str,
    ) -> Result<(), EstimationError> {
        use crate::inference::model::SavedAnchorKind;
        for (idx, component) in runtime.anchor_components.iter().enumerate() {
            match &component.kind {
                SavedAnchorKind::Parametric { .. } => {}
                SavedAnchorKind::FlexEvaluation { .. } => {
                    return Err(EstimationError::InvalidInput(format!(
                        "bernoulli marginal-slope {runtime_label} saved anchor component at \
                         index {idx} is FlexEvaluation; only Parametric components are \
                         expected for this runtime",
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn likelihood_family(&self) -> LikelihoodSpec {
        LikelihoodSpec::binomial_probit()
    }

    pub fn mean_from_eta(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        Ok(eta.mapv(normal_cdf))
    }

    pub fn mean_derivative_from_eta(
        &self,
        eta: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        Ok(eta.mapv(normal_pdf))
    }

    pub(crate) fn probit_frailty_scale(&self) -> f64 {
        marginal_slope_probit_frailty_scale(self.gaussian_frailty_sd)
    }

    /// Reconstruct the affine row coordinates consumed by the fitted
    /// likelihood, including the exact saved latent-score calibration.
    fn saved_alo_affine_state(
        &self,
        input: &PredictInput,
    ) -> Result<BernoulliMarginalSlopeSavedAloAffineState, EstimationError> {
        let latent_z_raw = input.auxiliary_scalar.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "saved marginal-slope ALO requires auxiliary z column '{}'",
                self.z_column
            ))
        })?;
        let secondary_design = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "saved marginal-slope ALO requires the fitted slope design".to_string(),
            )
        })?;
        let n = input.design.nrows();
        if latent_z_raw.len() != n
            || secondary_design.nrows() != n
            || input.offset.len() != n
            || input
                .offset_noise
                .as_ref()
                .is_some_and(|offset| offset.len() != n)
        {
            return Err(EstimationError::InvalidInput(format!(
                "saved marginal-slope ALO row mismatch: primary={n}, slope={}, z={}, primary_offset={}, slope_offset={}",
                secondary_design.nrows(),
                latent_z_raw.len(),
                input.offset.len(),
                input.offset_noise.as_ref().map_or(n, Array1::len),
            )));
        }
        if input.design.ncols() != self.beta_marginal.len()
            || secondary_design.ncols() != self.beta_slope.len()
        {
            return Err(EstimationError::InvalidInput(format!(
                "saved marginal-slope ALO coefficient mismatch: marginal design/beta={}/{}, slope design/beta={}/{}",
                input.design.ncols(),
                self.beta_marginal.len(),
                secondary_design.ncols(),
                self.beta_slope.len(),
            )));
        }

        let normalized = self
            .latent_z_normalization
            .apply(latent_z_raw, "saved marginal-slope ALO")
            .map_err(EstimationError::from)?;
        let calibrated = self.apply_latent_z_calibration(&normalized);
        let latent_z = self.apply_latent_z_conditional_calibration(&calibrated, input)?;
        let marginal_eta = input
            .design
            .dot(&self.beta_marginal)
            .mapv(|value| value + self.baseline_marginal)
            + &input.offset;
        let slope_offset = input
            .offset_noise
            .as_ref()
            .map_or_else(|| Array1::zeros(n), Clone::clone);
        let slope = secondary_design
            .dot(&self.beta_slope)
            .mapv(|value| value + self.baseline_slope)
            + &slope_offset;
        Ok(BernoulliMarginalSlopeSavedAloAffineState {
            marginal_eta,
            slope,
            latent_z,
        })
    }

    fn saved_alo_latent_measure(
        &self,
        input: &PredictInput,
        n_rows: usize,
    ) -> Result<LatentMeasureKind, EstimationError> {
        match &self.latent_measure {
            LatentMeasureKind::StandardNormal => Ok(LatentMeasureKind::StandardNormal),
            LatentMeasureKind::GlobalEmpirical { grid } => {
                Ok(LatentMeasureKind::GlobalEmpirical { grid: grid.clone() })
            }
            LatentMeasureKind::LocalEmpirical {
                feature_cols,
                input_scales,
                centers,
                grids,
                top_k,
                bandwidth,
                mixture,
                ..
            } => {
                let conditioning = self.local_conditioning_view(input).ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "saved BMS ALO with a local empirical latent measure requires the persisted conditioning matrix"
                            .to_string(),
                    )
                })?;
                let expected_dimension = centers.first().map_or(0, Vec::len);
                if conditioning.dim() != (n_rows, expected_dimension) {
                    return Err(EstimationError::InvalidInput(format!(
                        "saved BMS ALO local empirical conditioning is {}x{}; expected {n_rows}x{expected_dimension}",
                        conditioning.nrows(),
                        conditioning.ncols(),
                    )));
                }
                let mixtures = conditioning
                    .rows()
                    .into_iter()
                    .map(|row| {
                        let point = row.iter().copied().collect::<Vec<_>>();
                        Self::local_empirical_mixture_for_point(
                            &point, centers, *top_k, *bandwidth, *mixture,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(LatentMeasureKind::LocalEmpirical {
                    feature_cols: feature_cols.clone(),
                    input_scales: input_scales.clone(),
                    centers: centers.clone(),
                    grids: grids.clone(),
                    top_k: *top_k,
                    bandwidth: *bandwidth,
                    mixture: *mixture,
                    train_row_mixtures: Arc::new(mixtures),
                })
            }
        }
    }

    /// Replay the exact fitted row likelihood and observed Hessian for saved-H
    /// ALO.  This is the sole BMS saved-row authority for rigid, flexible,
    /// standard-normal, global-empirical, and local-empirical fits.
    pub fn saved_alo_replay(
        &self,
        input: &PredictInput,
        response: &Array1<f64>,
        prior_weights: &Array1<f64>,
    ) -> Result<BernoulliMarginalSlopeSavedAloReplay, EstimationError> {
        let affine = self.saved_alo_affine_state(input)?;
        let slope_design = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "saved BMS ALO requires the persisted slope design".to_string(),
            )
        })?;
        let anchor_corrections =
            self.build_anchor_correction_matrices(input, slope_design, &affine.latent_z)?;
        let latent_measure = self.saved_alo_latent_measure(input, response.len())?;
        let residual_features = match self.residual_repair.as_ref() {
            Some(geometry) => Some(self.residual_feature_view(input, geometry)?.to_owned()),
            None => None,
        };
        replay_saved_bernoulli_marginal_slope_alo(BernoulliMarginalSlopeSavedAloReplayInput {
            base_link: &self.base_link,
            marginal_design: &input.design,
            slope_design,
            marginal_beta: &self.beta_marginal,
            slope_beta: &self.beta_slope,
            score_warp_beta: self.beta_score_warp.as_ref(),
            link_deviation_beta: self.beta_link_dev.as_ref(),
            marginal_eta: &affine.marginal_eta,
            slope: &affine.slope,
            latent_z: &affine.latent_z,
            response,
            prior_weights,
            latent_measure,
            gaussian_frailty_sd: self.gaussian_frailty_sd,
            score_warp_runtime: self.score_warp_runtime.as_ref(),
            link_deviation_runtime: self.link_deviation_runtime.as_ref(),
            score_warp_anchor_rows: anchor_corrections.score_warp_anchor_rows.as_ref(),
            link_deviation_anchor_rows: anchor_corrections.link_dev_anchor_rows.as_ref(),
            residual_geometry: self.residual_repair.as_ref(),
            residual_beta: self.beta_residual.as_ref(),
            residual_features: residual_features.as_ref(),
        })
        .map_err(EstimationError::InvalidInput)
    }

    /// Apply the (optional) rank-INT latent-z calibration to a batch of
    /// normalized predict-time z values.
    ///
    /// The calibration was fit on the training z + weights as a Blom-
    /// rankit weighted rank inverse-normal transform; the calibrated
    /// sample is N(0, 1) by construction (exact, not approximate), which
    /// is why the BMS standard-normal closed-form kernel is correct on
    /// the calibrated scale. At predict time, every z that flows into a
    /// kernel evaluation site (`final_eta_and_gradient_from_theta`,
    /// `predict_eta_and_time_tangent`, and indirectly the per-row `solve_intercept_scalar`
    /// / `evaluate_prediction_calibration` / `observed_denested_cell_partials_at_z`
    /// helpers that consume per-row scalar z values from the closure-
    /// captured `z` array) must be routed through the same monotone
    /// transform. When `latent_z_calibration` is `None`, this returns
    /// the input unchanged — that case corresponds to training-time z
    /// having passed the strict normality check, so no transform was
    /// applied at fit time either.
    fn apply_latent_z_calibration(&self, z: &Array1<f64>) -> Array1<f64> {
        self.latent_score_map().rank_int_step(z.view())
    }

    /// Apply the (optional) conditional location-scale latent-z calibration
    /// (#905) to a batch of normalized predict-time z values.
    ///
    /// When `Some`, training detected a conditional `E[z|C]`/`Var(z|C)` shift
    /// and replaced its latent score by `ζ = (z − m(C))/√v(C)`. The predictor
    /// MUST apply the identical map, rebuilding the conditioning span `a(C)`
    /// from the marginal prediction design (`input.design`) — the same span the
    /// fit regressed z on. `None` ⇒ no conditional calibration was applied at
    /// fit time, so z passes through unchanged (mutually exclusive with the
    /// rank-INT calibration above).
    pub(crate) fn apply_latent_z_conditional_calibration(
        &self,
        z: &Array1<f64>,
        input: &PredictInput,
    ) -> Result<Array1<f64>, EstimationError> {
        self.latent_score_map().conditional_step(z, &input.design)
    }

    /// This predictor's fitted latent-score maps, borrowed.
    fn latent_score_map(&self) -> FittedLatentScoreMap<'_> {
        FittedLatentScoreMap {
            normalization: &self.latent_z_normalization,
            rank_int: self.latent_z_calibration.as_ref(),
            conditional: self.latent_z_conditional_calibration.as_ref(),
            span: self.latent_conditioning_span,
        }
    }

    fn rigid_intercept_from_marginal(&self, marginal_eta: f64, slope: f64) -> f64 {
        let probit_scale = self.probit_frailty_scale();
        marginal_eta * (1.0 + (probit_scale * slope).powi(2)).sqrt() / probit_scale
    }

    fn empirical_rigid_intercept_and_gradient(
        &self,
        marginal_eta: f64,
        slope: f64,
        nodes: &[f64],
        weights: &[f64],
    ) -> Result<(f64, f64, f64), EstimationError> {
        let marginal = bernoulli_marginal_link_map(&self.base_link, marginal_eta)
            .map_err(EstimationError::InvalidInput)?;
        let scale = self.probit_frailty_scale();
        empirical_intercept_and_partials(marginal.q, marginal.q1, slope, scale, nodes, weights)
    }

    fn local_empirical_mixture_for_point(
        point: &[f64],
        centers: &[Vec<f64>],
        top_k: usize,
        bandwidth: f64,
        mixture: crate::bms::LocalLawMixture,
    ) -> Result<Vec<(usize, f64)>, EstimationError> {
        // The fit computes every training row's mixture with the same function
        // (gam#2926), so a row's fitted law and its predicted law are one object.
        crate::bms::estimated_latent_law::local_empirical_mixture_for_point(
            point, centers, top_k, bandwidth, mixture,
        )
        .map_err(EstimationError::InvalidInput)
    }

    fn combine_empirical_grids(
        grids: &[EmpiricalZGrid],
        mixture: &[(usize, f64)],
    ) -> Result<EmpiricalZGrid, EstimationError> {
        // The fit combines every training row's grids with the same function
        // (gam#2926): sorted once, equal nodes coalesced, validated.
        crate::bms::combine_empirical_grids(grids, mixture).map_err(EstimationError::InvalidInput)
    }

    fn empirical_grid_for_prediction_row(
        &self,
        input: &PredictInput,
        row: usize,
    ) -> Result<Option<EmpiricalZGrid>, EstimationError> {
        match &self.latent_measure {
            LatentMeasureKind::StandardNormal => Ok(None),
            LatentMeasureKind::GlobalEmpirical { grid } => Ok(Some(grid.clone())),
            LatentMeasureKind::LocalEmpirical {
                centers,
                grids,
                top_k,
                bandwidth,
                mixture,
                ..
            } => {
                let conditioning = self.local_conditioning_view(input).ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "bernoulli marginal-slope local empirical prediction requires auxiliary conditioning matrix"
                            .to_string(),
                    )
                })?;
                if row >= conditioning.nrows() {
                    return Err(EstimationError::InvalidInput(format!(
                        "local empirical latent prediction row {row} is out of bounds for {} conditioning rows",
                        conditioning.nrows()
                    )));
                }
                let expected_dim = centers.first().map_or(0, Vec::len);
                if conditioning.ncols() != expected_dim {
                    return Err(EstimationError::InvalidInput(format!(
                        "local empirical latent prediction conditioning dimension mismatch: got {}, expected {expected_dim}",
                        conditioning.ncols()
                    )));
                }
                let point = conditioning.row(row).to_vec();
                let mixture =
                    Self::local_empirical_mixture_for_point(
                        &point, centers, *top_k, *bandwidth, *mixture,
                    )?;
                Self::combine_empirical_grids(grids, &mixture).map(Some)
            }
        }
    }

    fn transform_internal_eta_to_base_scale(
        &self,
        internal_eta: Array1<f64>,
        internal_grad: Option<Array2<f64>>,
    ) -> Result<(Array1<f64>, Option<Array2<f64>>), EstimationError> {
        Ok((internal_eta, internal_grad))
    }

    /// The residual repair features of a prediction input: the trailing block of
    /// `auxiliary_matrix`, after any local-empirical conditioning columns.
    fn residual_feature_view<'a>(
        &self,
        input: &'a PredictInput,
        geometry: &crate::bms::ResidualRepairGeometry,
    ) -> Result<ndarray::ArrayView2<'a, f64>, EstimationError> {
        let matrix = input.auxiliary_matrix.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction requires the residual columns {:?}",
                geometry.columns
            ))
        })?;
        let width = geometry.width();
        if matrix.ncols() < width {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction auxiliary matrix has {} columns; the residual \
                 repair block needs {width}",
                matrix.ncols()
            )));
        }
        Ok(matrix.slice(ndarray::s![.., matrix.ncols() - width..]))
    }

    /// The local-empirical conditioning columns of `auxiliary_matrix`: its
    /// leading block, followed by the residual repair features when a block is
    /// present.
    fn local_conditioning_view<'a>(
        &self,
        input: &'a PredictInput,
    ) -> Option<ndarray::ArrayView2<'a, f64>> {
        let matrix = input.auxiliary_matrix.as_ref()?;
        let residual_width = self
            .residual_repair
            .as_ref()
            .map_or(0, crate::bms::ResidualRepairGeometry::width);
        Some(matrix.slice(ndarray::s![.., ..matrix.ncols().saturating_sub(residual_width)]))
    }

    /// The residual block's slice of a flat coefficient vector: after the
    /// marginal and slope surfaces.
    fn residual_theta_slice<'a>(
        &self,
        theta: &'a Array1<f64>,
    ) -> Result<ArrayView1<'a, f64>, EstimationError> {
        let beta = self.beta_residual.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "bernoulli marginal-slope residual repair coefficients are missing".to_string(),
            )
        })?;
        let start = self.beta_marginal.len() + self.beta_slope.len();
        if theta.len() < start + beta.len() {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope theta length {} cannot hold the residual block at {start}..{}",
                theta.len(),
                start + beta.len()
            )));
        }
        Ok(theta.slice(ndarray::s![start..start + beta.len()]))
    }

    /// The residual genetic repair row index (gam#2924) and its gradient with
    /// respect to every coefficient: `η = c·q + s(g z + βᵀr)` with the anchor
    /// `c = √(1 + s² b̃ᵀ Σ(a) b̃)` replayed from the saved joint covariance —
    /// the pooled matrix, or the conditional model evaluated on the prediction
    /// rows' marginal design. Plug-in and posterior-mean prediction both read
    /// this one function; the coefficient-uncertainty integration of the
    /// posterior mean therefore carries `∂η/∂β` through the anchor, not only
    /// through the linear read `s·r`.
    fn residual_eta_and_gradient(
        &self,
        input: &PredictInput,
        theta: &Array1<f64>,
        need_gradient: bool,
        geometry: &crate::bms::ResidualRepairGeometry,
        z: &Array1<f64>,
        design_slope: &DesignMatrix,
        marginal_eta: &Array1<f64>,
        slope_eta: &Array1<f64>,
    ) -> Result<(Array1<f64>, Option<Array2<f64>>), EstimationError> {
        let n = z.len();
        let width = geometry.width();
        let features = self.residual_feature_view(input, geometry)?;
        if features.nrows() != n || features.ncols() != width {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope residual features are {}x{} but the prediction has {n} rows and the saved block {width} columns",
                features.nrows(),
                features.ncols()
            )));
        }
        let beta_residual = self.residual_theta_slice(theta)?;
        let beta_residual = beta_residual.to_vec();
        // The joint covariance field on the prediction rows: pooled, or the
        // conditional model on the marginal-index span exactly as at fit time.
        let a_block = input
            .design
            .try_to_dense_arc("bernoulli marginal-slope residual repair conditioning span")
            .map_err(EstimationError::InvalidInput)?;
        let field = geometry
            .covariance_field(a_block.view())
            .map_err(EstimationError::InvalidInput)?;
        let scale = self.probit_frailty_scale();
        let marginal_dim = self.beta_marginal.len();
        let slope_dim = self.beta_slope.len();
        let residual_offset = marginal_dim + slope_dim;
        let mut eta = Array1::<f64>::zeros(n);
        let mut grad = need_gradient.then(|| Array2::<f64>::zeros((n, theta.len())));
        let chunk_size = prediction_chunk_rows(theta.len(), 1, n);
        let mut start = 0usize;
        while start < n {
            let end = (start + chunk_size).min(n);
            let (mc, lc) = if need_gradient {
                (
                    Some(
                        input
                            .design
                            .try_row_chunk(start..end)
                            .map_err(|e| EstimationError::InvalidInput(e.to_string()))?,
                    ),
                    Some(
                        design_slope
                            .try_row_chunk(start..end)
                            .map_err(|e| EstimationError::InvalidInput(e.to_string()))?,
                    ),
                )
            } else {
                (None, None)
            };
            // Each row's anchor replay reads only that row, so the rows run on
            // the pool and scatter in index order: bit for bit the serial loop.
            let rows = (start..end)
                .into_par_iter()
                .map(|i| {
                    let marginal = bernoulli_marginal_link_map(&self.base_link, marginal_eta[i])
                        .map_err(EstimationError::InvalidInput)?;
                    let r = features.row(i);
                    let r = r.as_slice().ok_or_else(|| {
                        EstimationError::InvalidInput(
                            "residual feature row is not contiguous".to_string(),
                        )
                    })?;
                    let grid = self.empirical_grid_for_prediction_row(input, i)?;
                    crate::bms::residual_row_index(
                        &marginal,
                        slope_eta[i],
                        &beta_residual,
                        z[i],
                        r,
                        field.at_row(i),
                        grid.as_ref(),
                        scale,
                    )
                    .map_err(EstimationError::InvalidInput)
                })
                .collect::<Result<Vec<_>, EstimationError>>()?;
            for (li, (eta_i, d_q, d_g, d_beta)) in rows.into_iter().enumerate() {
                let i = start + li;
                eta[i] = eta_i;
                if let (Some(grad), Some(mc), Some(lc)) = (grad.as_mut(), mc.as_ref(), lc.as_ref())
                {
                    let mut row = grad.row_mut(i);
                    for j in 0..marginal_dim {
                        row[j] = d_q * mc[[li, j]];
                    }
                    for j in 0..slope_dim {
                        row[marginal_dim + j] = d_g * lc[[li, j]];
                    }
                    for (j, value) in d_beta.iter().enumerate() {
                        row[residual_offset + j] = *value;
                    }
                }
            }
            start = end;
        }
        self.transform_internal_eta_to_base_scale(eta, grad)
    }

    fn link_terms_value_d1(
        &self,
        eta0: &Array1<f64>,
        beta_link_dev: Option<&Array1<f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
    ) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        if let (Some(runtime), Some(beta)) = (&self.link_deviation_runtime, beta_link_dev) {
            // When the runtime carries a cross-block anchor residual, every
            // raw-design row needs `n_row · M` subtracted. `correction_for_row`
            // already holds the precomputed `n_row · M` for this predict row
            // (length basis_dim), so the corrected basis contribution to η
            // is `basis · beta - correction.dot(beta)` for every eta0 entry.
            // Derivative paths are unaffected (the anchor argument is a
            // different scalar than eta0).
            let basis = runtime
                .design_uncorrected(eta0)
                .map_err(EstimationError::from)?;
            let mut value = &basis.dot(beta) + eta0;
            if let Some(corr) = link_dev_correction_for_row {
                let offset = corr.dot(beta);
                for v in value.iter_mut() {
                    *v -= offset;
                }
            } else if runtime.anchor_correction.is_some() {
                return Err(EstimationError::InvalidInput(
                    "bernoulli marginal-slope link-deviation runtime has an anchor residual but \
                     no per-row correction was supplied to link_terms_value_d1"
                        .to_string(),
                ));
            }
            let d1 = runtime
                .first_derivative_design(eta0)
                .map_err(EstimationError::from)?;
            Ok((value, d1.dot(beta) + 1.0))
        } else {
            Ok((eta0.clone(), Array1::ones(eta0.len())))
        }
    }

    fn denested_partition_cells(
        &self,
        a: f64,
        b: f64,
        beta_score_warp: Option<&Array1<f64>>,
        beta_link_dev: Option<&Array1<f64>>,
        score_warp_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
    ) -> Result<Vec<crate::cubic_cell_kernel::DenestedPartitionCell>, EstimationError> {
        let score_breaks = if let Some(runtime) = self.score_warp_runtime.as_ref() {
            runtime.breakpoints().map_err(EstimationError::from)?
        } else {
            Vec::new()
        };
        let link_breaks = if let Some(runtime) = self.link_deviation_runtime.as_ref() {
            runtime.breakpoints().map_err(EstimationError::from)?
        } else {
            Vec::new()
        };
        let mut cells = crate::cubic_cell_kernel::build_denested_partition_cells_with_tails(
            a,
            b,
            &score_breaks,
            &link_breaks,
            |z| {
                if let (Some(runtime), Some(beta)) =
                    (self.score_warp_runtime.as_ref(), beta_score_warp)
                {
                    let mut span = runtime.local_cubic_at(beta.view(), z)?;
                    // `local_cubic_at`'s c0 is `Σ_j basis_c0[span][j] · beta[j]`.
                    // The cross-block residual replaces basis_c0 by
                    // basis_c0 − n_row · M, contributing a row-constant
                    // `correction.dot(beta)` to c0. Higher coefficients
                    // (c1..c3) depend on derivatives of the basis w.r.t.
                    // its own argument and are untouched.
                    if let Some(corr) = score_warp_correction_for_row {
                        span.c0 -= corr.dot(beta);
                    }
                    Ok(span)
                } else {
                    Ok(crate::cubic_cell_kernel::LocalSpanCubic {
                        left: 0.0,
                        right: 1.0,
                        c0: 0.0,
                        c1: 0.0,
                        c2: 0.0,
                        c3: 0.0,
                    })
                }
            },
            |u| {
                if let (Some(runtime), Some(beta)) =
                    (self.link_deviation_runtime.as_ref(), beta_link_dev)
                {
                    let mut span = runtime.local_cubic_at(beta.view(), u)?;
                    if let Some(corr) = link_dev_correction_for_row {
                        span.c0 -= corr.dot(beta);
                    }
                    Ok(span)
                } else {
                    Ok(crate::cubic_cell_kernel::LocalSpanCubic {
                        left: 0.0,
                        right: 1.0,
                        c0: 0.0,
                        c1: 0.0,
                        c2: 0.0,
                        c3: 0.0,
                    })
                }
            },
        )
        .map_err(EstimationError::InvalidInput)?;
        let scale = self.probit_frailty_scale();
        if scale != 1.0 {
            for partition_cell in &mut cells {
                partition_cell.cell.c0 *= scale;
                partition_cell.cell.c1 *= scale;
                partition_cell.cell.c2 *= scale;
                partition_cell.cell.c3 *= scale;
            }
        }
        Ok(cells)
    }

    /// The calibration `P(a) = Σ_cells ∫φ(z)Φ(η(z)) dz` under the standard
    /// normal latent law, read on its smaller tail exactly as the fit reads it
    /// (gam#3216, gam#3333): on the survival side each cell is evaluated with
    /// its index negated, whose value is `∫φ(z)Φ(−η(z)) dz`, and whose moments
    /// are the cell's own, so they contract with the cell's `∂c/∂a` into `P′`
    /// and `P″`.
    fn evaluate_denested_calibration_tail(
        &self,
        a: f64,
        slope: f64,
        beta_score_warp: Option<&Array1<f64>>,
        beta_link_dev: Option<&Array1<f64>>,
        score_warp_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        survival_side: bool,
    ) -> Result<CalibrationTail, EstimationError> {
        let cells = self.denested_partition_cells(
            a,
            slope,
            beta_score_warp,
            beta_link_dev,
            score_warp_correction_for_row,
            link_dev_correction_for_row,
        )?;
        let scale = self.probit_frailty_scale();
        let summands = cells.len() * crate::cubic_cell_kernel::TERMINAL_GL_ORDER;
        let mut tail = 0.0;
        let mut tail_rounding = 0.0;
        let mut density = 0.0;
        let mut density_slope = 0.0;
        for partition_cell in cells {
            let cell = partition_cell.cell;
            let (dc_da_raw, _) = crate::cubic_cell_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let (d2c_da2_raw, _, _) = crate::cubic_cell_kernel::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let d2c_da2 = scale_coeff4(d2c_da2_raw, scale);
            // Derive the moment `max_degree` from the contractions consumed
            // below, instead of hardcoding a magic constant. The second-
            // derivative contraction dominates the first-derivative one, so
            // its required degree is the binding bound. Hardcoding 7 here
            // produced 8 moments while the contraction needs 10 (#321).
            let max_degree = crate::cubic_cell_kernel::cell_second_derivative_required_max_degree(
                &dc_da, &dc_da, &d2c_da2,
            );
            let state = crate::cubic_cell_kernel::evaluate_cell_moments(
                if survival_side {
                    cell.negated()
                } else {
                    cell
                },
                max_degree,
            )
            .map_err(EstimationError::InvalidInput)?;
            tail += state.value;
            tail_rounding += state.value_rounding;
            density += crate::cubic_cell_kernel::cell_first_derivative_from_moments(
                &dc_da,
                &state.moments,
            )
            .map_err(EstimationError::InvalidInput)?;
            density_slope += crate::cubic_cell_kernel::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &d2c_da2,
                &state.moments,
            )
            .map_err(EstimationError::InvalidInput)?;
        }
        Ok(CalibrationTail {
            tail,
            density,
            density_slope: Some(density_slope),
            summands,
            tail_rounding,
        })
    }

    fn observed_denested_cell_partials_at_z(
        &self,
        z_value: f64,
        a: f64,
        b: f64,
        beta_score_warp: Option<&Array1<f64>>,
        beta_link_dev: Option<&Array1<f64>>,
        score_warp_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
    ) -> Result<ObservedDenestedCellPartials, EstimationError> {
        use crate::cubic_cell_kernel as exact;

        let zero_span = exact::LocalSpanCubic {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        };
        let u_value = a + b * z_value;
        let score_span = if let (Some(runtime), Some(beta)) =
            (self.score_warp_runtime.as_ref(), beta_score_warp)
        {
            let mut span = runtime
                .local_cubic_at(beta.view(), z_value)
                .map_err(EstimationError::from)?;
            if let Some(corr) = score_warp_correction_for_row {
                span.c0 -= corr.dot(beta);
            }
            span
        } else {
            zero_span
        };
        let link_span = if let (Some(runtime), Some(beta)) =
            (self.link_deviation_runtime.as_ref(), beta_link_dev)
        {
            let mut span = runtime
                .local_cubic_at(beta.view(), u_value)
                .map_err(EstimationError::from)?;
            if let Some(corr) = link_dev_correction_for_row {
                span.c0 -= corr.dot(beta);
            }
            span
        } else {
            zero_span
        };
        let scale = self.probit_frailty_scale();
        let coeff = scale_coeff4(
            exact::denested_cell_coefficients(score_span, link_span, a, b),
            scale,
        );
        let (dc_da_raw, dc_db_raw) =
            exact::denested_cell_coefficient_partials(score_span, link_span, a, b);
        let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) =
            exact::denested_cell_second_partials(score_span, link_span, a, b);
        let (dc_daaa, dc_daab, dc_dabb, dc_dbbb) = exact::denested_cell_third_partials(link_span);
        Ok(ObservedDenestedCellPartials {
            coeff,
            dc_da: scale_coeff4(dc_da_raw, scale),
            dc_db: scale_coeff4(dc_db_raw, scale),
            dc_daa: scale_coeff4(dc_daa_raw, scale),
            dc_dab: scale_coeff4(dc_dab_raw, scale),
            dc_dbb: scale_coeff4(dc_dbb_raw, scale),
            dc_daaa: scale_coeff4(dc_daaa, scale),
            dc_daab: scale_coeff4(dc_daab, scale),
            dc_dabb: scale_coeff4(dc_dabb, scale),
            dc_dbbb: scale_coeff4(dc_dbbb, scale),
        })
    }

    /// The calibration `P(a) = Σ_k w_k Φ(η(a, z_k))` over an empirical latent
    /// grid, read on its smaller tail `Σ_k w_k Φ(∓η_k)` (gam#3216, gam#3333).
    fn evaluate_empirical_denested_calibration_tail(
        &self,
        a: f64,
        slope: f64,
        beta_score_warp: Option<&Array1<f64>>,
        beta_link_dev: Option<&Array1<f64>>,
        grid: &EmpiricalZGrid,
        score_warp_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        survival_side: bool,
    ) -> Result<CalibrationTail, EstimationError> {
        let tail_sign = if survival_side { -1.0 } else { 1.0 };
        let mut tail = 0.0;
        let mut density = 0.0;
        let mut density_slope = 0.0;
        for (node, weight) in grid.pairs() {
            let obs = self.observed_denested_cell_partials_at_z(
                node,
                a,
                slope,
                beta_score_warp,
                beta_link_dev,
                score_warp_correction_for_row,
                link_dev_correction_for_row,
            )?;
            let eta = eval_coeff4_at(&obs.coeff, node);
            let eta_a = eval_coeff4_at(&obs.dc_da, node);
            let eta_aa = eval_coeff4_at(&obs.dc_daa, node);
            let pdf = normal_pdf(eta);
            tail += weight * normal_cdf(tail_sign * eta);
            density += weight * pdf * eta_a;
            density_slope += weight * pdf * (eta_aa - eta * eta_a * eta_a);
        }
        Ok(CalibrationTail {
            tail,
            density,
            density_slope: Some(density_slope),
            summands: grid.nodes.len(),
            tail_rounding: 0.0,
        })
    }

    fn evaluate_prediction_calibration_tail(
        &self,
        a: f64,
        slope: f64,
        beta_score_warp: Option<&Array1<f64>>,
        beta_link_dev: Option<&Array1<f64>>,
        empirical_grid: Option<&EmpiricalZGrid>,
        score_warp_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        survival_side: bool,
    ) -> Result<CalibrationTail, EstimationError> {
        if let Some(grid) = empirical_grid {
            self.evaluate_empirical_denested_calibration_tail(
                a,
                slope,
                beta_score_warp,
                beta_link_dev,
                grid,
                score_warp_correction_for_row,
                link_dev_correction_for_row,
                survival_side,
            )
        } else {
            self.evaluate_denested_calibration_tail(
                a,
                slope,
                beta_score_warp,
                beta_link_dev,
                score_warp_correction_for_row,
                link_dev_correction_for_row,
                survival_side,
            )
        }
    }

    pub fn from_unified(
        unified: &UnifiedFitResult,
        z_column: String,
        latent_z_normalization: SavedLatentZNormalization,
        latent_measure: LatentMeasureKind,
        baseline_marginal: f64,
        baseline_slope: f64,
        base_link: InverseLink,
        frailty: FrailtySpec,
        score_warp_runtime: Option<SavedCompiledFlexBlock>,
        link_deviation_runtime: Option<SavedCompiledFlexBlock>,
        latent_z_calibration: Option<crate::bms::LatentZRankIntCalibration>,
        latent_z_conditional_calibration: Option<crate::bms::LatentZConditionalCalibration>,
        latent_conditioning_span: LatentConditioningSpan,
        residual_repair: Option<crate::bms::ResidualRepairGeometry>,
    ) -> Result<Self, String> {
        let gaussian_frailty_sd = match frailty {
            FrailtySpec::None => None,
            FrailtySpec::GaussianShift {
                scale: FrailtyScale::Fixed { sigma },
            } => Some(sigma),
            FrailtySpec::GaussianShift {
                scale: FrailtyScale::Learned { .. },
            } => {
                return Err(
                    "bernoulli marginal-slope predictor requires a fixed GaussianShift sigma"
                        .to_string(),
                );
            }
            FrailtySpec::HazardMultiplier { .. } => {
                return Err(
                    "bernoulli marginal-slope predictor does not support HazardMultiplier frailty"
                        .to_string(),
                );
            }
        };
        if !matches!(
            base_link,
            InverseLink::Standard(gam_problem::types::StandardLink::Probit)
        ) {
            return Err(
                "bernoulli marginal-slope predictor requires a saved probit link".to_string(),
            );
        }
        if let Some(runtime) = score_warp_runtime.as_ref() {
            runtime.validate_exact_replay_contract().map_err(|e| {
                format!("bernoulli marginal-slope score-warp runtime is invalid: {e}")
            })?;
        }
        if let Some(runtime) = link_deviation_runtime.as_ref() {
            runtime.validate_exact_replay_contract().map_err(|e| {
                format!("bernoulli marginal-slope link-deviation runtime is invalid: {e}")
            })?;
        }
        // Cross-block anchor residuals on either runtime are now applied
        // per-row by every predict-time `local_cubic_at` / `basis_cubic_at`
        // / `design` call site via `build_anchor_correction_matrices`.
        latent_z_normalization
            .validate("bernoulli marginal-slope predictor")
            .map_err(|e| {
                format!("bernoulli marginal-slope predictor latent z normalization is invalid: {e}")
            })?;
        latent_measure
            .validate("bernoulli marginal-slope predictor latent measure")
            .map_err(|e| {
                format!("bernoulli marginal-slope predictor latent measure is invalid: {e}")
            })?;
        let blocks = &unified.blocks;
        if residual_repair.is_some()
            && (score_warp_runtime.is_some() || link_deviation_runtime.is_some())
        {
            return Err(crate::bms::ResidualRepairRefusal::FlexBlocksUnsupported.to_string());
        }
        let expected_blocks = 2
            + usize::from(residual_repair.is_some())
            + usize::from(score_warp_runtime.is_some())
            + usize::from(link_deviation_runtime.is_some());
        if blocks.len() != expected_blocks {
            return Err(format!(
                "bernoulli marginal-slope predictor requires exactly {expected_blocks} coefficient blocks under the current exact de-nested semantics, got {}",
                blocks.len()
            ));
        }
        let mut cursor = 2usize;
        let beta_residual = match residual_repair.as_ref() {
            Some(geometry) => {
                let beta = blocks
                    .get(cursor)
                    .ok_or_else(|| "missing residual repair coefficient block".to_string())?
                    .beta
                    .clone();
                if beta.len() != geometry.width() {
                    return Err(format!(
                        "bernoulli marginal-slope residual repair block has {} coefficients but \
                         the saved geometry names {} columns",
                        beta.len(),
                        geometry.width()
                    ));
                }
                cursor += 1;
                Some(beta)
            }
            None => None,
        };
        let beta_score_warp = if score_warp_runtime.is_some() {
            let beta = blocks
                .get(cursor)
                .ok_or_else(|| "missing score-warp coefficient block".to_string())?
                .beta
                .clone();
            cursor += 1;
            Some(beta)
        } else {
            None
        };
        let beta_link_dev = if link_deviation_runtime.is_some() {
            Some(
                blocks
                    .get(cursor)
                    .ok_or_else(|| "missing link-deviation coefficient block".to_string())?
                    .beta
                    .clone(),
            )
        } else {
            None
        };
        Ok(Self {
            beta_marginal: blocks[0].beta.clone(),
            beta_slope: blocks[1].beta.clone(),
            beta_score_warp,
            beta_link_dev,
            base_link,
            z_column,
            latent_z_normalization,
            latent_measure,
            baseline_marginal,
            baseline_slope,
            covariance: unified.beta_covariance().cloned(),
            score_warp_runtime,
            link_deviation_runtime,
            gaussian_frailty_sd,
            latent_z_calibration,
            latent_z_conditional_calibration,
            latent_conditioning_span,
            residual_repair,
            beta_residual,
        })
    }

    pub fn theta(&self) -> Array1<f64> {
        let total = self.beta_marginal.len()
            + self.beta_slope.len()
            + self.beta_residual.as_ref().map_or(0, |b| b.len())
            + self.beta_score_warp.as_ref().map_or(0, |b| b.len())
            + self.beta_link_dev.as_ref().map_or(0, |b| b.len());
        let mut theta = Array1::<f64>::zeros(total);
        let mut cursor = 0usize;
        theta
            .slice_mut(ndarray::s![cursor..cursor + self.beta_marginal.len()])
            .assign(&self.beta_marginal);
        cursor += self.beta_marginal.len();
        theta
            .slice_mut(ndarray::s![cursor..cursor + self.beta_slope.len()])
            .assign(&self.beta_slope);
        cursor += self.beta_slope.len();
        if let Some(beta) = self.beta_residual.as_ref() {
            theta
                .slice_mut(ndarray::s![cursor..cursor + beta.len()])
                .assign(beta);
            cursor += beta.len();
        }
        if let Some(beta) = self.beta_score_warp.as_ref() {
            theta
                .slice_mut(ndarray::s![cursor..cursor + beta.len()])
                .assign(beta);
            cursor += beta.len();
        }
        if let Some(beta) = self.beta_link_dev.as_ref() {
            theta
                .slice_mut(ndarray::s![cursor..cursor + beta.len()])
                .assign(beta);
        }
        theta
    }

    fn split_theta<'a>(
        &'a self,
        theta: &'a Array1<f64>,
    ) -> Result<
        (
            ArrayView1<'a, f64>,
            ArrayView1<'a, f64>,
            Option<ArrayView1<'a, f64>>,
            Option<ArrayView1<'a, f64>>,
        ),
        EstimationError,
    > {
        let expected = self.theta().len();
        if theta.len() != expected {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope theta length mismatch: expected {expected}, got {}",
                theta.len()
            )));
        }
        let mut cursor = 0usize;
        let marginal = theta.slice(ndarray::s![cursor..cursor + self.beta_marginal.len()]);
        cursor += self.beta_marginal.len();
        let slope = theta.slice(ndarray::s![cursor..cursor + self.beta_slope.len()]);
        cursor += self.beta_slope.len();
        if let Some(beta) = self.beta_residual.as_ref() {
            // The residual block sits between the slope surface and the flex
            // blocks; its slice is read by `residual_theta_slice`.
            cursor += beta.len();
        }
        let score_warp = self.beta_score_warp.as_ref().map(|beta| {
            let view = theta.slice(ndarray::s![cursor..cursor + beta.len()]);
            cursor += beta.len();
            view
        });
        let link_dev = self
            .beta_link_dev
            .as_ref()
            .map(|beta| theta.slice(ndarray::s![cursor..cursor + beta.len()]));
        Ok((marginal, slope, score_warp, link_dev))
    }

    /// The marginal intercept under the de-nested flexible model
    ///   η(z) = a + b z + b Δ_h(z) + Δ_w(a + b z).
    fn solve_intercept_scalar(
        &self,
        marginal_eta: f64,
        slope: f64,
        link_dev_beta: Option<&Array1<f64>>,
        score_warp_beta: Option<&Array1<f64>>,
        empirical_grid: Option<&EmpiricalZGrid>,
        warm_start_buf: &mut Array1<f64>,
        score_warp_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
        link_dev_correction_for_row: Option<ndarray::ArrayView1<'_, f64>>,
    ) -> Result<f64, EstimationError> {
        let marginal = bernoulli_marginal_link_map(&self.base_link, marginal_eta)
            .map_err(EstimationError::InvalidInput)?;
        let probit_scale = self.probit_frailty_scale();
        let a_rigid = self.rigid_intercept_from_marginal(marginal.q, slope);
        let mut intercept = a_rigid;
        if let (Some(_), Some(beta)) = (self.link_deviation_runtime.as_ref(), link_dev_beta) {
            warm_start_buf[0] = a_rigid;
            let one_pt = warm_start_buf.slice(ndarray::s![0..1]).to_owned();
            let (l_val, l_d1) =
                self.link_terms_value_d1(&one_pt, Some(beta), link_dev_correction_for_row)?;
            let ell1 = l_d1[0];
            // The affine inversion divides by ℓ₁ = 1 + w′(a), which the deviation's
            // structural monotonicity keeps positive. Where it is not positive, or
            // the quotient is not representable, the rigid seed stands.
            if ell1 > 0.0 {
                let ell0 = l_val[0] - ell1 * a_rigid;
                let observed_slope = probit_scale * ell1 * slope;
                let seed = (marginal.q * (1.0 + observed_slope * observed_slope).sqrt()
                    / probit_scale
                    - ell0)
                    / ell1;
                if seed.is_finite() {
                    intercept = seed;
                }
            }
        }

        // The saved model solves the fit's equation, `log T(a) = log Φ(∓q)` on
        // the smaller marginal tail, held to the same derived resolution, so
        // the root it reads the implicit-function gradients (`a_q`, `a_b`) at
        // is the fitted one to rounding, whatever the seed (gam#3333).
        let survival_side = marginal.q >= 0.0;
        let log_target = smaller_tail_log_target(marginal.q);
        let (root, _) = solve_log_tail_root(
            intercept,
            survival_side,
            |a| {
                self.evaluate_prediction_calibration_tail(
                    a,
                    slope,
                    score_warp_beta,
                    link_dev_beta,
                    empirical_grid,
                    score_warp_correction_for_row,
                    link_dev_correction_for_row,
                    survival_side,
                )
                .map_err(|err| err.to_string())?
                .log_residual(survival_side, log_target)
            },
            "saved bernoulli marginal-slope intercept",
            || format!("q={}, b={slope}", marginal.q),
        )
        .map_err(EstimationError::InvalidInput)?;
        Ok(root)
    }

    pub fn final_eta_and_gradient_from_theta(
        &self,
        input: &PredictInput,
        theta: &Array1<f64>,
        need_gradient: bool,
    ) -> Result<(Array1<f64>, Option<Array2<f64>>), EstimationError> {
        // P4: when training applied a rank-INT calibration to the latent
        // z (so the BMS rigid kernel could use the closed-form
        // standard-normal path), the predictor MUST apply the same
        // monotone transform to predict-time z before any kernel
        // evaluation. The transform is mathematically exact: piecewise-
        // linear interpolation on (sorted_z, weighted_cdf) followed by
        // Φ⁻¹, both strictly monotone and invertible up to the empirical
        // CDF resolution. `None` ⇒ training-time z passed the strict
        // normality check, no transform was applied, leave z unchanged.
        // #905: then replace z by ζ = (z − m(C))/√v(C) when training engaged
        // the conditional Auto gate (no-op otherwise; mutually exclusive with
        // the rank-INT calibration). Both live in `prediction_latent_z` so
        // every kernel evaluation — this one, the time tangent, and the
        // anchored row kernels — consumes the same score.
        let z = self.prediction_latent_z(input)?;
        let design_slope = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "bernoulli marginal-slope prediction requires slope design".to_string(),
            )
        })?;
        let (beta_marginal, beta_slope, beta_score_warp, beta_link_dev) =
            self.split_theta(theta)?;
        if self.score_warp_runtime.is_some() != beta_score_warp.is_some() {
            return Err(EstimationError::InvalidInput(
                "bernoulli marginal-slope saved score-warp runtime/coefficients are inconsistent"
                    .to_string(),
            ));
        }
        if self.link_deviation_runtime.is_some() != beta_link_dev.is_some() {
            return Err(EstimationError::InvalidInput(
                "bernoulli marginal-slope saved link-deviation runtime/coefficients are inconsistent"
                    .to_string(),
            ));
        }
        let n = z.len();
        if input.offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction primary offset length mismatch: rows={n}, offset={}",
                input.offset.len()
            )));
        }
        let slope_offset = input
            .offset_noise
            .as_ref()
            .map_or_else(|| Array1::zeros(n), Clone::clone);
        if slope_offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction slope offset length mismatch: rows={n}, offset_noise={}",
                slope_offset.len()
            )));
        }
        let marginal_eta = input
            .design
            .dot(&beta_marginal.to_owned())
            .mapv(|v| v + self.baseline_marginal)
            + &input.offset;
        let slope_eta = design_slope
            .dot(&beta_slope.to_owned())
            .mapv(|v| v + self.baseline_slope)
            + &slope_offset;
        if let Some(geometry) = self.residual_repair.as_ref() {
            return self.residual_eta_and_gradient(
                input,
                theta,
                need_gradient,
                geometry,
                &z,
                design_slope,
                &marginal_eta,
                &slope_eta,
            );
        }
        let flex_active =
            self.score_warp_runtime.is_some() || self.link_deviation_runtime.is_some();
        let marginal_dim = self.beta_marginal.len();
        let slope_dim = self.beta_slope.len();
        let score_warp_dim = self.beta_score_warp.as_ref().map_or(0, Array1::len);
        let link_dev_dim = self.beta_link_dev.as_ref().map_or(0, Array1::len);
        let slope_offset = marginal_dim;
        let score_warp_offset = slope_offset + slope_dim;
        let link_dev_offset = score_warp_offset + score_warp_dim;
        let chunk_size = prediction_chunk_rows(theta.len(), 1, n);
        let num_chunks = n.div_ceil(chunk_size);
        let scale = self.probit_frailty_scale();
        // Cross-block anchor corrections: when either runtime carries an
        // anchor residual, precompute the per-row correction matrices
        // (n × runtime_basis_dim) once. Each subsequent per-row evaluation
        // subtracts the corresponding row of these matrices from the raw
        // cubic-span basis output. When neither runtime has a residual,
        // the returned bundle is empty and threading is a no-op.
        let anchor_corrections =
            self.build_anchor_correction_matrices(input, design_slope, &z)?;
        let marginal_map = marginal_eta
            .iter()
            .map(|&eta| {
                bernoulli_marginal_link_map(&self.base_link, eta)
                    .map_err(EstimationError::InvalidInput)
            })
            .collect::<Result<Vec<_>, _>>()?;

        if !flex_active {
            let (final_eta_internal, marginal_scales, slope_scales) = match &self.latent_measure
            {
                LatentMeasureKind::StandardNormal => {
                    let sb_vec = slope_eta.mapv(|b| scale * b);
                    let c_vec = sb_vec.mapv(|sb| (1.0 + sb * sb).sqrt());
                    let final_eta_internal = Array1::from_iter(
                        (0..n).map(|i| c_vec[i] * marginal_eta[i] + sb_vec[i] * z[i]),
                    );
                    let marginal_scales = c_vec;
                    let slope_scales = Array1::from_iter((0..n).map(|i| {
                        marginal_eta[i] * (scale * scale) * slope_eta[i] / marginal_scales[i]
                            + scale * z[i]
                    }));
                    (final_eta_internal, marginal_scales, slope_scales)
                }
                LatentMeasureKind::GlobalEmpirical { .. } | LatentMeasureKind::LocalEmpirical { .. } => {
                    // Each row takes its own law and solves its own anchor with no
                    // warm start shared across rows, so the rows run in parallel and
                    // every value is the one the serial loop computed.
                    let rows = (0..n)
                        .into_par_iter()
                        .map(|i| {
                            let grid = self
                                .empirical_grid_for_prediction_row(input, i)?
                                .ok_or_else(|| {
                                    EstimationError::InvalidInput(
                                        "empirical latent prediction did not produce a row grid"
                                            .to_string(),
                                    )
                                })?;
                            self.empirical_rigid_intercept_and_gradient(
                                marginal_eta[i],
                                slope_eta[i],
                                &grid.nodes,
                                &grid.weights,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let final_eta = Array1::from_iter(
                        rows.iter()
                            .enumerate()
                            .map(|(i, &(intercept, _, _))| intercept + scale * slope_eta[i] * z[i]),
                    );
                    let marginal_scales = Array1::from_iter(rows.iter().map(|&(_, a_marginal, _)| a_marginal));
                    let slope_scales = Array1::from_iter(
                        rows.iter()
                            .enumerate()
                            .map(|(i, &(_, _, a_slope))| a_slope + scale * z[i]),
                    );
                    (final_eta, marginal_scales, slope_scales)
                }
            };

            if !need_gradient {
                return self.transform_internal_eta_to_base_scale(final_eta_internal, None);
            }

            // Chunk Jacobian: one pass per row fills both blocks.
            let mut grad_internal = Array2::<f64>::zeros((n, theta.len()));
            let mut start = 0usize;
            while start < n {
                let end = (start + chunk_size).min(n);
                let mc = input
                    .design
                    .try_row_chunk(start..end)
                    .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
                let lc = design_slope
                    .try_row_chunk(start..end)
                    .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;

                for li in 0..(end - start) {
                    let i = start + li;
                    let c = marginal_scales[i];
                    let g_scale = slope_scales[i];
                    let mut row = grad_internal.row_mut(i);
                    for j in 0..marginal_dim {
                        row[j] = c * mc[[li, j]];
                    }
                    for j in 0..slope_dim {
                        row[slope_offset + j] = g_scale * lc[[li, j]];
                    }
                }

                start = end;
            }
            return self
                .transform_internal_eta_to_base_scale(final_eta_internal, Some(grad_internal));
        }

        // ── Flexible path: per-row intercept solve, chunked Jacobians ──
        let score_warp_obs_design = self
            .score_warp_runtime
            .as_ref()
            .map(|runtime| {
                if runtime.anchor_correction.is_some() {
                    let anchor_rows = anchor_corrections
                        .score_warp_anchor_rows_view()
                        .ok_or_else(|| {
                            EstimationError::InvalidInput(
                                "bernoulli marginal-slope score-warp anchor residual present but \
                                 anchor_corrections bundle is missing the parametric anchor rows"
                                    .to_string(),
                            )
                        })?;
                    runtime
                        .design_with_anchor_rows(&z, anchor_rows)
                        .map_err(EstimationError::from)
                } else {
                    runtime.design(&z).map_err(EstimationError::from)
                }
            })
            .transpose()?;
        let score_dev_obs =
            if let (Some(design), Some(beta)) = (score_warp_obs_design.as_ref(), beta_score_warp) {
                design.dot(&beta.to_owned())
            } else {
                Array1::zeros(n)
            };

        // Solve intercepts and (when gradient needed) IFT scalars in chunk-parallel passes.
        // Outputs are preallocated and each parallel worker writes directly into
        // its exclusive `axis_chunks_iter_mut` slice; no per-chunk owned buffer
        // and no serial copy pass over the result chunks.
        let score_warp_beta_owned = beta_score_warp.as_ref().map(|v| v.to_owned());
        let link_dev_beta_owned = beta_link_dev.as_ref().map(|v| v.to_owned());
        let mut intercepts = Array1::<f64>::zeros(n);
        let mut a_q_vec = need_gradient.then(|| Array1::<f64>::zeros(n));
        let mut a_b_vec = need_gradient.then(|| Array1::<f64>::zeros(n));
        let mut a_h_rows = if need_gradient && score_warp_dim > 0 {
            Some(Array2::<f64>::zeros((n, score_warp_dim)))
        } else {
            None
        };
        let mut a_w_rows = if need_gradient && link_dev_dim > 0 {
            Some(Array2::<f64>::zeros((n, link_dev_dim)))
        } else {
            None
        };
        let solve_result: Result<(), EstimationError> = {
            use ndarray::Axis;
            use rayon::iter::IndexedParallelIterator;
            let intercepts_chunks: Vec<ndarray::ArrayViewMut1<f64>> = intercepts
                .axis_chunks_iter_mut(Axis(0), chunk_size)
                .collect();
            let a_q_chunks: Option<Vec<ndarray::ArrayViewMut1<f64>>> = a_q_vec
                .as_mut()
                .map(|a| a.axis_chunks_iter_mut(Axis(0), chunk_size).collect());
            let a_b_chunks: Option<Vec<ndarray::ArrayViewMut1<f64>>> = a_b_vec
                .as_mut()
                .map(|a| a.axis_chunks_iter_mut(Axis(0), chunk_size).collect());
            let a_h_chunks: Option<Vec<ndarray::ArrayViewMut2<f64>>> = a_h_rows
                .as_mut()
                .map(|a| a.axis_chunks_iter_mut(Axis(0), chunk_size).collect());
            let a_w_chunks: Option<Vec<ndarray::ArrayViewMut2<f64>>> = a_w_rows
                .as_mut()
                .map(|a| a.axis_chunks_iter_mut(Axis(0), chunk_size).collect());

            // Bundle per-chunk sinks so each parallel worker owns disjoint mutable
            // views into the shared output arrays.
            struct FlexSolveSink<'a> {
                intercepts: ndarray::ArrayViewMut1<'a, f64>,
                a_q: Option<ndarray::ArrayViewMut1<'a, f64>>,
                a_b: Option<ndarray::ArrayViewMut1<'a, f64>>,
                a_h: Option<ndarray::ArrayViewMut2<'a, f64>>,
                a_w: Option<ndarray::ArrayViewMut2<'a, f64>>,
            }
            let mut sinks: Vec<FlexSolveSink<'_>> = Vec::with_capacity(num_chunks);
            // Move each Option<Vec> into iterators so we can zip them.
            let mut intercepts_iter = intercepts_chunks.into_iter();
            let mut a_q_iter = a_q_chunks.map(|v| v.into_iter());
            let mut a_b_iter = a_b_chunks.map(|v| v.into_iter());
            let mut a_h_iter = a_h_chunks.map(|v| v.into_iter());
            let mut a_w_iter = a_w_chunks.map(|v| v.into_iter());
            for _ in 0..num_chunks {
                sinks.push(FlexSolveSink {
                    intercepts: intercepts_iter.next().expect("chunk count matches"),
                    a_q: a_q_iter
                        .as_mut()
                        .map(|it| it.next().expect("chunk count matches")),
                    a_b: a_b_iter
                        .as_mut()
                        .map(|it| it.next().expect("chunk count matches")),
                    a_h: a_h_iter
                        .as_mut()
                        .map(|it| it.next().expect("chunk count matches")),
                    a_w: a_w_iter
                        .as_mut()
                        .map(|it| it.next().expect("chunk count matches")),
                });
            }

            // Precompute the score-warp basis cubic table once when the latent
            // grid is row-constant (`GlobalEmpirical`). The per-row inner loop
            // calls `basis_cubic_at(j, node)` with `node` taken from the grid,
            // which is identical for every row in this code path, so the
            // n_rows × n_nodes × score_warp_dim table can be hoisted out of
            // the parallel chunk dispatch. Per-row work only touches the
            // basis-function-specific `c0` shift via `score_corr_row`, which
            // stays inside the row loop. Computed at the top level so no
            // OnceLock / lazy init lives inside the par closure (per the
            // OnceLock + nested rayon deadlock rule).
            let global_score_basis_table: Option<
                Vec<Vec<crate::cubic_cell_kernel::LocalSpanCubic>>,
            > = if let (LatentMeasureKind::GlobalEmpirical { grid }, Some(runtime)) =
                (&self.latent_measure, self.score_warp_runtime.as_ref())
            {
                let mut table = Vec::with_capacity(score_warp_dim);
                for j in 0..score_warp_dim {
                    let mut row = Vec::with_capacity(grid.nodes.len());
                    for &node in &grid.nodes {
                        row.push(
                            runtime
                                .basis_cubic_at(j, node)
                                .map_err(EstimationError::from)?,
                        );
                    }
                    table.push(row);
                }
                Some(table)
            } else {
                None
            };
            let global_score_basis_table = global_score_basis_table.as_ref();

            sinks
                .into_par_iter()
                .enumerate()
                .try_for_each(|(chunk_idx, mut sink)| -> Result<(), EstimationError> {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                let rows = end - start;
                // Destructure the sink into independent `&mut` references so we
                // can borrow them disjointly across iterations of the inner row
                // loop without further reborrowing through `Option::as_mut`.
                let intercepts_view = &mut sink.intercepts;
                let mut a_q = sink.a_q.as_mut();
                let mut a_b = sink.a_b.as_mut();
                let mut a_h = sink.a_h.as_mut();
                let mut a_w = sink.a_w.as_mut();
                let mut warm_start_buf = Array1::<f64>::zeros(1);
                let mut f_h_row = vec![0.0; score_warp_dim];
                let mut f_w_row = vec![0.0; link_dev_dim];

                for local_row in 0..rows {
                    let i = start + local_row;
                    let slope = slope_eta[i];
                    let q = marginal_eta[i];
                    let empirical_grid = self.empirical_grid_for_prediction_row(input, i)?;
                    let score_corr_row = anchor_corrections.score_warp_row(i);
                    let link_corr_row = anchor_corrections.link_dev_row(i);
                    intercepts_view[local_row] = self.solve_intercept_scalar(
                        q,
                        slope,
                        link_dev_beta_owned.as_ref(),
                        score_warp_beta_owned.as_ref(),
                        empirical_grid.as_ref(),
                        &mut warm_start_buf,
                        score_corr_row,
                        link_corr_row,
                    )?;

                    if !need_gradient {
                        continue;
                    }

                    let intercept = intercepts_view[local_row];
                    let m_a = self
                        .evaluate_prediction_calibration_tail(
                        intercept,
                        slope,
                        score_warp_beta_owned.as_ref(),
                        link_dev_beta_owned.as_ref(),
                        empirical_grid.as_ref(),
                        score_corr_row,
                        link_corr_row,
                        false,
                    )?
                        .density;
                    // ∂a/∂θ = −F_θ/F_a by the implicit function theorem. The
                    // calibration F is increasing in a, so F_a is positive unless
                    // every quadrature density has underflowed; the intercept then
                    // has no finite gradient and the row is refused.
                    if !(m_a > 0.0 && m_a.is_finite()) {
                        return Err(EstimationError::InvalidInput(format!(
                            "bernoulli marginal-slope prediction row {i}: the intercept \
                             calibration derivative dF/da is {m_a:e}, so the intercept has \
                             no finite gradient"
                        )));
                    }
                    a_q.as_mut().expect("a_q allocated when need_gradient")[local_row] =
                        marginal_map[i].mu1 / m_a;
                    let mut f_b = 0.0;
                    f_h_row.fill(0.0);
                    f_w_row.fill(0.0);
                    if let Some(grid) = empirical_grid.as_ref() {
                        for (node_idx, (node, weight)) in grid.pairs().enumerate() {
                            let obs = self.observed_denested_cell_partials_at_z(
                                node,
                                intercept,
                                slope,
                                score_warp_beta_owned.as_ref(),
                                link_dev_beta_owned.as_ref(),
                                score_corr_row,
                                link_corr_row,
                            )?;
                            let eta = eval_coeff4_at(&obs.coeff, node);
                            let pdf = normal_pdf(eta);
                            f_b += weight * pdf * eval_coeff4_at(&obs.dc_db, node);

                            if let Some(runtime) = self.score_warp_runtime.as_ref() {
                                for j in 0..score_warp_dim {
                                    // When the latent grid is row-constant
                                    // (`GlobalEmpirical`), the per-(j, node)
                                    // basis cubic is identical for every row
                                    // and lives in `global_score_basis_table`.
                                    // Otherwise (`LocalEmpirical`) the grid
                                    // varies per row and we fall back to a
                                    // direct `basis_cubic_at` call.
                                    let mut basis_span = if let Some(table) =
                                        global_score_basis_table
                                    {
                                        table[j][node_idx]
                                    } else {
                                        runtime
                                            .basis_cubic_at(j, node)
                                            .map_err(EstimationError::from)?
                                    };
                                    // `basis_cubic_at` returns the j-th basis
                                    // function's local cubic; the residual
                                    // subtracts `correction[j]` from the
                                    // constant term (row-constant, basis-
                                    // function-specific). Higher span
                                    // coefficients are unaffected.
                                    if let Some(corr) = score_corr_row {
                                        basis_span.c0 -= corr[j];
                                    }
                                    let coeffs = crate::cubic_cell_kernel::score_basis_cell_coefficients(
                                        basis_span,
                                        slope,
                                    );
                                    let coeffs = scale_coeff4(coeffs, scale);
                                    f_h_row[j] += weight * pdf * eval_coeff4_at(&coeffs, node);
                                }
                            }

                            if let Some(runtime) = self.link_deviation_runtime.as_ref() {
                                for j in 0..link_dev_dim {
                                    let mut basis_span = runtime
                                        .basis_cubic_at(j, intercept + slope * node)
                                        .map_err(EstimationError::from)?;
                                    if let Some(corr) = link_corr_row {
                                        basis_span.c0 -= corr[j];
                                    }
                                    let coeffs = crate::cubic_cell_kernel::link_basis_cell_coefficients(
                                        basis_span,
                                        intercept,
                                        slope,
                                    );
                                    let coeffs = scale_coeff4(coeffs, scale);
                                    f_w_row[j] += weight * pdf * eval_coeff4_at(&coeffs, node);
                                }
                            }
                        }
                    } else {
                        let cells = self.denested_partition_cells(
                            intercept,
                            slope,
                            score_warp_beta_owned.as_ref(),
                            link_dev_beta_owned.as_ref(),
                            score_corr_row,
                            link_corr_row,
                        )?;
                        for partition_cell in cells {
                            let cell = partition_cell.cell;
                            let state =
                                crate::cubic_cell_kernel::evaluate_cell_moments(
                                    cell, 9,
                                )
                                .map_err(EstimationError::InvalidInput)?;
                            let (_, dc_db_raw) = crate::cubic_cell_kernel::denested_cell_coefficient_partials(
                                partition_cell.score_span,
                                partition_cell.link_span,
                                intercept,
                                slope,
                            );
                            // `denested_partition_cells` scales the cell itself for
                            // Gaussian frailty, so every coefficient partial of
                            // F(a, theta) must carry the same probit scale as F_a.
                            let dc_db = scale_coeff4(dc_db_raw, scale);
                            f_b += crate::cubic_cell_kernel::cell_first_derivative_from_moments(
                                &dc_db,
                                &state.moments,
                            )
                            .map_err(EstimationError::InvalidInput)?;

                            let mid = 0.5 * (cell.left + cell.right);
                            if let Some(runtime) = self.score_warp_runtime.as_ref() {
                                for j in 0..score_warp_dim {
                                    let mut basis_span = runtime
                                        .basis_cubic_at(j, mid)
                                        .map_err(EstimationError::from)?;
                                    if let Some(corr) = score_corr_row {
                                        basis_span.c0 -= corr[j];
                                    }
                                    let coeffs = crate::cubic_cell_kernel::score_basis_cell_coefficients(
                                        basis_span, slope,
                                    );
                                    let coeffs = scale_coeff4(coeffs, scale);
                                    f_h_row[j] += crate::cubic_cell_kernel::cell_first_derivative_from_moments(
                                        &coeffs,
                                        &state.moments,
                                    )
                                    .map_err(EstimationError::InvalidInput)?;
                                }
                            }

                            if let Some(runtime) = self.link_deviation_runtime.as_ref() {
                                for j in 0..link_dev_dim {
                                    let mut basis_span = runtime
                                        .basis_cubic_at(j, intercept + slope * mid)
                                        .map_err(EstimationError::from)?;
                                    if let Some(corr) = link_corr_row {
                                        basis_span.c0 -= corr[j];
                                    }
                                    let coeffs = crate::cubic_cell_kernel::link_basis_cell_coefficients(
                                        basis_span,
                                        intercept,
                                        slope,
                                    );
                                    let coeffs = scale_coeff4(coeffs, scale);
                                    f_w_row[j] += crate::cubic_cell_kernel::cell_first_derivative_from_moments(
                                        &coeffs,
                                        &state.moments,
                                    )
                                    .map_err(EstimationError::InvalidInput)?;
                                }
                            }
                        }
                    }
                    if let Some(a_h_view) = a_h.as_mut() {
                        let factor = -1.0 / m_a;
                        for j in 0..score_warp_dim {
                            a_h_view[[local_row, j]] = factor * f_h_row[j];
                        }
                    }
                    if let Some(a_w_view) = a_w.as_mut() {
                        let factor = -1.0 / m_a;
                        for j in 0..link_dev_dim {
                            a_w_view[[local_row, j]] = factor * f_w_row[j];
                        }
                    }
                    a_b.as_mut().expect("a_b allocated when need_gradient")[local_row] =
                        -f_b / m_a;
                }
                Ok(())
            })
        };
        solve_result?;

        let eta_base = &intercepts + &(&slope_eta * &z);

        let mut link_c_obs: Option<Array1<f64>> = None;
        let mut link_basis_obs: Option<Array2<f64>> = None;
        let link_dev_obs = if let (Some(runtime), Some(beta_owned)) = (
            self.link_deviation_runtime.as_ref(),
            link_dev_beta_owned.as_ref(),
        ) {
            let basis = if runtime.anchor_correction.is_some() {
                let anchor_rows =
                    anchor_corrections
                        .link_dev_anchor_rows_view()
                        .ok_or_else(|| {
                            EstimationError::InvalidInput(
                            "bernoulli marginal-slope link-deviation anchor residual present but \
                             anchor_corrections bundle is missing the parametric anchor rows"
                                .to_string(),
                        )
                        })?;
                runtime
                    .design_with_anchor_rows(&eta_base, anchor_rows)
                    .map_err(EstimationError::from)?
            } else {
                runtime.design(&eta_base).map_err(EstimationError::from)?
            };
            let dev = basis.dot(beta_owned);
            if need_gradient {
                let d1 = runtime
                    .first_derivative_design(&eta_base)
                    .map_err(EstimationError::from)?;
                let mut c_obs = d1.dot(beta_owned);
                c_obs.mapv_inplace(|v| v + 1.0);
                link_c_obs = Some(c_obs);
                link_basis_obs = Some(basis);
            }
            dev
        } else {
            Array1::zeros(n)
        };
        let final_eta_internal =
            (&eta_base + &(&slope_eta * &score_dev_obs) + &link_dev_obs).mapv(|v| scale * v);

        if !need_gradient {
            return self.transform_internal_eta_to_base_scale(final_eta_internal, None);
        }

        // Both were allocated by `need_gradient.then(..)`, and the early return
        // above already handled the `!need_gradient` case.
        let allocated = "need_gradient is true past the early return, so this was allocated";
        let a_q_vec = a_q_vec.expect(allocated);
        let a_b_vec = a_b_vec.expect(allocated);

        // Emit chunk Jacobians using precomputed scalars; each worker writes
        // directly into its exclusive `axis_chunks_iter_mut` slice of the
        // preallocated `grad` output so no serial copy pass is needed.
        let mut grad = Array2::<f64>::zeros((n, theta.len()));
        {
            use ndarray::Axis;
            use rayon::iter::IndexedParallelIterator;
            let grad_result: Result<(), String> = grad
                .axis_chunks_iter_mut(Axis(0), chunk_size)
                .into_par_iter()
                .enumerate()
                .try_for_each(|(chunk_idx, mut grad_chunk)| -> Result<(), String> {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(n);
                    let mc = input
                        .design
                        .try_row_chunk(start..end)
                        .map_err(|e| e.to_string())?;
                    let lc = design_slope
                        .try_row_chunk(start..end)
                        .map_err(|e| e.to_string())?;
                    let rows = end - start;

                    for li in 0..rows {
                        let i = start + li;
                        let mut row = grad_chunk.row_mut(li);

                        let a_q = a_q_vec[i];
                        for j in 0..marginal_dim {
                            row[j] = a_q * mc[[li, j]];
                        }

                        let base_multiplier = link_c_obs.as_ref().map_or(1.0, |c| c[i]);
                        let g_scale = base_multiplier * (a_b_vec[i] + z[i]) + score_dev_obs[i];
                        for j in 0..slope_dim {
                            row[slope_offset + j] = g_scale * lc[[li, j]];
                        }

                        if let (Some(a_h_rows), Some(obs_design)) =
                            (a_h_rows.as_ref(), score_warp_obs_design.as_ref())
                        {
                            let slope = slope_eta[i];
                            for j in 0..score_warp_dim {
                                row[score_warp_offset + j] =
                                    base_multiplier * a_h_rows[[i, j]] + slope * obs_design[[i, j]];
                            }
                        }

                        if let Some(a_w_rows) = a_w_rows.as_ref() {
                            for j in 0..link_dev_dim {
                                row[link_dev_offset + j] = a_w_rows[[i, j]];
                            }
                        }

                        if let (Some(link_c), Some(link_basis)) =
                            (link_c_obs.as_ref(), link_basis_obs.as_ref())
                        {
                            let c = link_c[i];
                            for j in 0..marginal_dim {
                                row[j] *= c;
                            }
                            for j in 0..link_dev_dim {
                                row[link_dev_offset + j] =
                                    c * row[link_dev_offset + j] + link_basis[[i, j]];
                            }
                        }
                    }
                    Ok(())
                });
            grad_result.map_err(EstimationError::InvalidInput)?;
        }
        if scale != 1.0 {
            grad.mapv_inplace(|v| scale * v);
        }
        self.transform_internal_eta_to_base_scale(final_eta_internal, Some(grad))
    }

    /// Per-row final (base-scale) linear predictor for an arbitrary
    /// coefficient vector `theta` in the saved `[marginal | slope |
    /// score_warp? | link_dev?]` block order. The marginal-slope rigid
    /// kernel is applied exactly per row, so the returned η is the same
    /// object the point predictor consumes — only parameterised by an
    /// external draw instead of `self.theta()`. Used by the posterior
    /// predictive path (#1049) to map each Laplace draw to its η surface
    /// before the shared eta→bands collapse; the response scale is the
    /// probit inverse link `μ = Φ(η)`.
    pub fn final_eta_from_theta(
        &self,
        input: &PredictInput,
        theta: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let (eta, _) = self.final_eta_and_gradient_from_theta(input, theta, false)?;
        Ok(eta)
    }

    /// Length of the concatenated coefficient vector this predictor
    /// consumes (`marginal + slope + residual? + score_warp? + link_dev?`). The
    /// posterior predictive path validates each saved draw against this
    /// before mapping it through [`Self::final_eta_from_theta`].
    pub fn theta_len(&self) -> usize {
        self.beta_marginal.len()
            + self.beta_slope.len()
            + self.beta_residual.as_ref().map_or(0, Array1::len)
            + self.beta_score_warp.as_ref().map_or(0, Array1::len)
            + self.beta_link_dev.as_ref().map_or(0, Array1::len)
    }

    /// Whether a score-warp or link-deviation runtime is active. With either,
    /// the calibrated intercept depends on that runtime's whole coefficient
    /// vector, so η is not a function of the two primaries `(q, b)` alone and
    /// [`Self::anchored_row_kernels`] is unavailable.
    pub fn has_flexible_runtime(&self) -> bool {
        self.score_warp_runtime.is_some() || self.link_deviation_runtime.is_some()
    }

    /// Whether a residual repair block (gam#2924) is present. With one, η reads
    /// `βᵀr` and the anchor reads the whole residual coefficient vector through
    /// `b̃ᵀΣb̃`, so η is not a function of `(q, b)` alone and
    /// [`Self::anchored_row_kernels`] is unavailable.
    pub fn has_residual_repair(&self) -> bool {
        self.residual_repair.is_some()
    }

    /// The latent score every kernel evaluation consumes: the saved
    /// normalisation, then the rank-INT calibration or the conditional
    /// calibration, exactly as the fit applied them.
    fn prediction_latent_z(&self, input: &PredictInput) -> Result<Array1<f64>, EstimationError> {
        let z_raw = input.auxiliary_scalar.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction requires auxiliary z column '{}'",
                self.z_column
            ))
        })?;
        self.latent_score_map()
            .apply(z_raw, &input.design, "bernoulli marginal-slope prediction")
    }

    /// The two primaries the anchored kernel is a function of, at `theta`:
    /// the marginal index `q = X·β_q + q₀ + offset` and the slope
    /// `b = W·β_b + b₀ + offset_b`, one entry per prediction row. Both are
    /// affine in `theta`, which is what lets a coefficient posterior be
    /// pushed onto `(q, b)` exactly.
    pub fn anchored_primaries(
        &self,
        input: &PredictInput,
        theta: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        let (beta_marginal, beta_slope, _, _) = self.split_theta(theta)?;
        let design_slope = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "bernoulli marginal-slope prediction requires slope design".to_string(),
            )
        })?;
        let n = input.design.nrows();
        if input.offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction primary offset length mismatch: rows={n}, offset={}",
                input.offset.len()
            )));
        }
        let slope_offset = input
            .offset_noise
            .as_ref()
            .map_or_else(|| Array1::zeros(n), Clone::clone);
        if slope_offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction slope offset length mismatch: rows={n}, offset_noise={}",
                slope_offset.len()
            )));
        }
        let marginal_eta = input
            .design
            .dot(&beta_marginal.to_owned())
            .mapv(|v| v + self.baseline_marginal)
            + &input.offset;
        let slope_eta = design_slope
            .dot(&beta_slope.to_owned())
            .mapv(|v| v + self.baseline_slope)
            + &slope_offset;
        Ok((marginal_eta, slope_eta))
    }

    /// One [`AnchoredRowKernel`] per prediction row: the latent score and the
    /// declared latent law the intercept is anchored against, so a caller can
    /// re-solve the anchor at any `(q, b)` — every node of a posterior
    /// integration, not only `θ̂`. Refused while a flexible runtime is active
    /// (see [`Self::has_flexible_runtime`]).
    pub fn anchored_row_kernels(
        &self,
        input: &PredictInput,
    ) -> Result<Vec<AnchoredRowKernel>, EstimationError> {
        if self.has_flexible_runtime() {
            return Err(EstimationError::InvalidInput(
                "bernoulli marginal-slope anchored row kernels are only defined for the rigid \
                 and declared-law latent measures; a score-warp or link-deviation runtime \
                 anchors the intercept on its own coefficient vector"
                    .to_string(),
            ));
        }
        if self.has_residual_repair() {
            return Err(EstimationError::InvalidInput(
                "bernoulli marginal-slope anchored row kernels are functions of (q, b) alone; a \
                 residual repair block (gam#2924) moves the index through its coefficients and \
                 the anchor through the joint (z, r) quadratic form"
                    .to_string(),
            ));
        }
        let z = self.prediction_latent_z(input)?;
        let probit_scale = self.probit_frailty_scale();
        (0..z.len())
            .map(|row| {
                Ok(AnchoredRowKernel {
                    z: z[row],
                    probit_scale,
                    base_link: self.base_link.clone(),
                    grid: self.empirical_grid_for_prediction_row(input, row)?,
                })
            })
            .collect()
    }

    /// Per-row `(eta, eta_t)` under the exact saved-model IFT pull-back.
    ///
    /// Returns the same `eta` as `predict_plugin_response`/`predict_linear_predictor`
    /// plus the complete tangent of the internal probit index with respect to
    /// follow-up time. Both moving primary coordinates are explicit inputs:
    /// `eta_t = eta_q q_t + eta_b b_t`. Keeping value and tangent in one replay
    /// prevents survival prediction from evaluating `b(t)` while silently
    /// differentiating the time-constant-slope model.
    ///
    /// Rigid, empirical, and flexible latent laws each supply both exact
    /// partials. In the flexible path `a_q = mu_q/F_a` and `a_b = -F_b/F_a`
    /// are the two implicit derivatives of the calibrated intercept.
    pub(crate) fn predict_eta_and_time_tangent(
        &self,
        input: &PredictInput,
        q_t: &Array1<f64>,
        b_t: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        // P4: see `final_eta_and_gradient_from_theta` for the rationale.
        // The rank-INT calibration is a mathematically exact monotone
        // transform; both the rigid standard-normal kernel and the
        // implicit-function chain rule consume the calibrated z, never
        // the raw normalized z, exactly mirroring fit-time semantics.
        let z = self.prediction_latent_z(input)?;
        let design_slope = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "bernoulli marginal-slope prediction requires slope design".to_string(),
            )
        })?;
        let n = z.len();
        if q_t.len() != n || b_t.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope tangent length mismatch: rows={n}, q_t={}, b_t={}",
                q_t.len(),
                b_t.len(),
            )));
        }
        if input.offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction primary offset length mismatch: rows={n}, offset={}",
                input.offset.len()
            )));
        }
        let slope_offset = input
            .offset_noise
            .as_ref()
            .map_or_else(|| Array1::zeros(n), Clone::clone);
        if slope_offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction slope offset length mismatch: rows={n}, offset_noise={}",
                slope_offset.len()
            )));
        }
        let marginal_eta = input
            .design
            .dot(&self.beta_marginal)
            .mapv(|v| v + self.baseline_marginal)
            + &input.offset;
        let slope_eta = design_slope
            .dot(&self.beta_slope)
            .mapv(|v| v + self.baseline_slope)
            + &slope_offset;
        let scale = self.probit_frailty_scale();
        let flex_active =
            self.score_warp_runtime.is_some() || self.link_deviation_runtime.is_some();

        // Rigid path mirrors `final_eta_and_gradient_from_theta`:
        //   eta = c·q + s·b·z,
        //   eta_q = c,
        //   eta_b = s²·b·q/c + s·z.
        if !flex_active {
            match &self.latent_measure {
                LatentMeasureKind::StandardNormal => {
                    let sb = slope_eta.mapv(|x| scale * x);
                    let eta_q = sb.mapv(|s| (1.0 + s * s).sqrt());
                    let eta = &eta_q * &marginal_eta + &sb * &z;
                    let eta_b = Array1::from_iter((0..n).map(|row| {
                        scale * scale * slope_eta[row] * marginal_eta[row] / eta_q[row]
                            + scale * z[row]
                    }));
                    let eta_t = &eta_q * q_t + &eta_b * b_t;
                    return Ok((eta, eta_t));
                }
                _ => {
                    // Rows are independent, as in `final_eta_and_gradient_from_theta`:
                    // each takes its own law and solves its own anchor.
                    let rows = (0..n)
                        .into_par_iter()
                        .map(|i| {
                            let grid = self
                                .empirical_grid_for_prediction_row(input, i)?
                                .ok_or_else(|| {
                                    EstimationError::InvalidInput(
                                        "empirical latent prediction did not produce a row grid"
                                            .to_string(),
                                    )
                                })?;
                            self.empirical_rigid_intercept_and_gradient(
                                marginal_eta[i],
                                slope_eta[i],
                                &grid.nodes,
                                &grid.weights,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let eta = Array1::from_iter(
                        rows.iter()
                            .enumerate()
                            .map(|(i, &(intercept, _, _))| intercept + scale * slope_eta[i] * z[i]),
                    );
                    let eta_t = Array1::from_iter(rows.iter().enumerate().map(
                        |(i, &(_, eta_q, a_slope))| {
                            let eta_b = a_slope + scale * z[i];
                            eta_q * q_t[i] + eta_b * b_t[i]
                        },
                    ));
                    return Ok((eta, eta_t));
                }
            }
        }

        // Flex path: solve the per-row intercept, then evaluate
        //   eta = scale · (a + b·z + b·Δ_h(z) + Δ_w(a + b·z))
        //   ∂eta/∂q = scale · (1 + Δ_w'(a + b·z)) · ∂a/∂q,
        //   ∂a/∂q   = φ(q) / |F_a|         (IFT, marginal_link is probit so mu1 = φ(q))
        // Mirrors `final_eta_and_gradient_from_theta` lines 1385-1621.
        let marginal_map = marginal_eta
            .iter()
            .map(|&eta_marg| {
                bernoulli_marginal_link_map(&self.base_link, eta_marg)
                    .map_err(EstimationError::InvalidInput)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Cross-block anchor corrections (see final_eta_and_gradient_from_theta
        // for the design); precompute once before the per-row loop.
        let anchor_corrections =
            self.build_anchor_correction_matrices(input, design_slope, &z)?;
        // Per-row: solve intercept scalar, evaluate denested calibration,
        // record (intercept, a_q, a_b). The `warm_start_buf` is just per-call
        // scratch — give each rayon worker its own buffer via fold init.
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let chains: Result<Vec<(f64, f64, f64)>, EstimationError> = (0..n)
            .into_par_iter()
            .map_init(
                || Array1::<f64>::zeros(1),
                |warm_start_buf, i| {
                    let q = marginal_eta[i];
                    let slope = slope_eta[i];
                    let empirical_grid = self.empirical_grid_for_prediction_row(input, i)?;
                    let score_corr_row = anchor_corrections.score_warp_row(i);
                    let link_corr_row = anchor_corrections.link_dev_row(i);
                    let intercept = self.solve_intercept_scalar(
                        q,
                        slope,
                        self.beta_link_dev.as_ref(),
                        self.beta_score_warp.as_ref(),
                        empirical_grid.as_ref(),
                        warm_start_buf,
                        score_corr_row,
                        link_corr_row,
                    )?;
                    let m_a = self
                        .evaluate_prediction_calibration_tail(
                        intercept,
                        slope,
                        self.beta_score_warp.as_ref(),
                        self.beta_link_dev.as_ref(),
                        empirical_grid.as_ref(),
                        score_corr_row,
                        link_corr_row,
                        false,
                    )?
                        .density;
                    // ∂a/∂θ = −F_θ/F_a: a non-positive F_a leaves the intercept
                    // with no finite gradient, so the row is refused.
                    if !(m_a > 0.0 && m_a.is_finite()) {
                        return Err(EstimationError::InvalidInput(format!(
                            "bernoulli marginal-slope prediction row {i}: the intercept \
                             calibration derivative dF/da is {m_a:e}, so the intercept has \
                             no finite gradient"
                        )));
                    }
                    let mut f_b = 0.0;
                    if let Some(grid) = empirical_grid.as_ref() {
                        for (node, weight) in grid.pairs() {
                            let obs = self.observed_denested_cell_partials_at_z(
                                node,
                                intercept,
                                slope,
                                self.beta_score_warp.as_ref(),
                                self.beta_link_dev.as_ref(),
                                score_corr_row,
                                link_corr_row,
                            )?;
                            let eta = eval_coeff4_at(&obs.coeff, node);
                            f_b += weight * normal_pdf(eta) * eval_coeff4_at(&obs.dc_db, node);
                        }
                    } else {
                        for partition_cell in self.denested_partition_cells(
                            intercept,
                            slope,
                            self.beta_score_warp.as_ref(),
                            self.beta_link_dev.as_ref(),
                            score_corr_row,
                            link_corr_row,
                        )? {
                            let cell = partition_cell.cell;
                            let state = crate::cubic_cell_kernel::evaluate_cell_moments(cell, 9)
                                .map_err(EstimationError::InvalidInput)?;
                            let (_, dc_db_raw) =
                                crate::cubic_cell_kernel::denested_cell_coefficient_partials(
                                    partition_cell.score_span,
                                    partition_cell.link_span,
                                    intercept,
                                    slope,
                                );
                            let dc_db = scale_coeff4(dc_db_raw, scale);
                            f_b += crate::cubic_cell_kernel::cell_first_derivative_from_moments(
                                &dc_db,
                                &state.moments,
                            )
                            .map_err(EstimationError::InvalidInput)?;
                        }
                    }
                    Ok((
                        intercept,
                        marginal_map[i].mu1 / m_a,
                        -f_b / m_a,
                    ))
                },
            )
            .collect();
        let chains = chains?;
        let mut intercepts = Array1::<f64>::zeros(n);
        let mut a_q = Array1::<f64>::zeros(n);
        let mut a_b = Array1::<f64>::zeros(n);
        for (i, (intercept, q_chain, b_chain)) in chains.into_iter().enumerate() {
            intercepts[i] = intercept;
            a_q[i] = q_chain;
            a_b[i] = b_chain;
        }

        let score_dev_obs = if let (Some(runtime), Some(beta)) = (
            self.score_warp_runtime.as_ref(),
            self.beta_score_warp.as_ref(),
        ) {
            let design = if runtime.anchor_correction.is_some() {
                let anchor_rows = anchor_corrections
                    .score_warp_anchor_rows_view()
                    .ok_or_else(|| {
                        EstimationError::InvalidInput(
                            "bernoulli marginal-slope score-warp anchor residual present but \
                             anchor_corrections bundle is missing the parametric anchor rows"
                                .to_string(),
                        )
                    })?;
                runtime
                    .design_with_anchor_rows(&z, anchor_rows)
                    .map_err(EstimationError::from)?
            } else {
                runtime.design(&z).map_err(EstimationError::from)?
            };
            design.dot(beta)
        } else {
            Array1::zeros(n)
        };
        let eta_base = &intercepts + &(&slope_eta * &z);
        let (link_dev_obs, link_c_obs) = if let (Some(runtime), Some(beta)) = (
            self.link_deviation_runtime.as_ref(),
            self.beta_link_dev.as_ref(),
        ) {
            let basis = if runtime.anchor_correction.is_some() {
                let anchor_rows =
                    anchor_corrections
                        .link_dev_anchor_rows_view()
                        .ok_or_else(|| {
                            EstimationError::InvalidInput(
                            "bernoulli marginal-slope link-deviation anchor residual present but \
                             anchor_corrections bundle is missing the parametric anchor rows"
                                .to_string(),
                        )
                        })?;
                runtime
                    .design_with_anchor_rows(&eta_base, anchor_rows)
                    .map_err(EstimationError::from)?
            } else {
                runtime.design(&eta_base).map_err(EstimationError::from)?
            };
            let dev = basis.dot(beta);
            let d1 = runtime
                .first_derivative_design(&eta_base)
                .map_err(EstimationError::from)?;
            let mut c_obs = d1.dot(beta);
            c_obs.mapv_inplace(|v| v + 1.0);
            (dev, c_obs)
        } else {
            (Array1::zeros(n), Array1::ones(n))
        };
        let final_eta_internal =
            (&eta_base + &(&slope_eta * &score_dev_obs) + &link_dev_obs).mapv(|v| scale * v);
        let eta_q = (&link_c_obs * &a_q).mapv(|value| scale * value);
        let eta_b = Array1::from_iter((0..n).map(|row| {
            scale
                * (link_c_obs[row] * (a_b[row] + z[row]) + score_dev_obs[row])
        }));
        let eta_t = &eta_q * q_t + &eta_b * b_t;
        Ok((final_eta_internal, eta_t))
    }
}
