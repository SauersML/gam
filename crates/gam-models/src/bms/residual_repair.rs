//! Residual genetic repair (gam#2924): a shrunk block of conditionally centred
//! genetic residual features inside the anchored Bernoulli marginal-slope
//! likelihood.
//!
//! # The model
//!
//! ```text
//!   η_i = c(a_i)·q(a_i) + s·( b(a_i)·z_i + βᵀ r_i ),      r_i = φ_i − E_ref[φ | S_i, A_i]
//!   c(a) = √(1 + s²· b̃(a)ᵀ Σ(a) b̃(a)),                     b̃(a) = (b(a), β)
//!   Σ(a) = Var((z, r) | a),   s = 1/√(1+σ²) the probit frailty scale
//! ```
//!
//! The scalar score kept ONE direction of the genome; a varying slope `b(a)`
//! rescales that direction and cannot turn it into a different one. `βᵀr` reads
//! the directions the score discarded, and the population value it removes from
//! squared risk is exactly `c_rᵀ Σ_r⁺ c_r` with `c_r = E[rY]`, `Σ_r = E[rrᵀ]`
//! (`Descent.Portability.ResidualGeneticRepair.residual_repair_law`).
//!
//! # Why the anchor integrates the WHOLE drive
//!
//! The marginal interpretation `E[p | a] = Φ(q(a))` is what keeps the baseline
//! surface meaningful. Under a conditionally Gaussian joint law of `(z, r)` the
//! anchoring equation `E_{z,r}[Φ(c·q + s(b z + βᵀr)) | a] = Φ(q)` is solved by
//! `c(a) = √(1 + s²·b̃ᵀΣ(a)b̃)` — the gam#2766 identity with the residual block
//! appended to the score vector. Adding `βᵀr` WITHOUT moving the anchor would
//! leak `Var(βᵀr | a)` into the baseline: the same defect as an unconditioned
//! score. With the anchor holding along every parameter path, the baseline and
//! the predictor-shape directions are Fisher-orthogonal under the declared law
//! (`Descent.Portability.MarginalAnchor.crossInformation_baseline_shape_zero`),
//! which is what makes `β` estimable without corrupting `q`.
//!
//! # Where the block lives in the row program
//!
//! The rigid marginal-slope kernel is a `RowKernel<2>` over the primaries
//! `(q, g)`, each a row-linear predictor of its block. The anchor reads `β`
//! beyond the row-linear read `t = r_iᵀβ`: through `u = βᵀγ` and the quadratic
//! form `v = βᵀΣ_rrβ`. Those three are the only way `β` enters the row, so the
//! residual kernel (`residual_repair_kernel`) is a `RowKernel<5>` over
//! `(η_m, g, t, u, v)` whatever the width, with every primary-space channel
//! derived by the canonical jet lowering of the ONE row program
//! [`residual_row_nll`]. The coefficient-space channels are the chain rule
//! through the one quadratic map `β ↦ v`, whose constant curvature `2Σ_rr` is the
//! only term the generic pullback does not carry.
//!
//! # The centring contract
//!
//! The features are supplied already centred on the reference law. The fit does
//! not centre them — that would silently absorb a level into the baseline — it
//! CHECKS `E_w[r_k | marginal-index span] = 0` with the same robust Rao score
//! test the score's conditional gate uses (gam#2768), at the same level, and
//! refuses with a typed reason when a column fails.
//!
//! # The penalty
//!
//! One ridge block `λ_r·‖β‖²` with its own REML/LAML smoothing parameter: not a
//! fixed prior, not a grid. Under `r ⟂ Y` the marginal likelihood drives
//! `λ_r → ∞` and `β → 0`, leaving the score-only fit unchanged.

use super::conditional_score_covariance::ConditionalScoreCovariance;
use super::family::*;
use super::gradient_paths::*;
use super::*;
use gam_math::jet_scalar::{JetScalar, SymmetricQuadraticCoefficients};
use gam_math::jet_tower::Tower4;
use gam_math::nested_dual::JetField;
use crate::latent_anchor::{AnchorDensity, AnchorGridOwned, AnchorTaylor, solve_anchor};
use ndarray::Zip;

/// Name of the residual block in the parameter-block list and in the
/// identifiability audit.
pub const RESIDUAL_BLOCK_NAME: &str = "residual_repair";

/// Gauge priority of the residual block: below the parametric surfaces (a
/// shared direction is demoted out of the residual block, never out of the
/// marginal or slope surface), above the flex deviations.
pub(super) const GAUGE_PRIORITY_RESIDUAL: u8 = 110;

/// The channel a fit with this block under the conditional joint covariance
/// Σ(a) and a fired conditional latent-z calibration has no Murphy–Topel
/// correction for, so its covariance is withheld (gam#2985,
/// `CovarianceDeclined::BmsGeneratedRegressorResidualRepairChannelUnavailable`).
/// Under the pooled law the block's row, covariance and finite-law channels are
/// carried ([`super::residual_repair_kernel::ResidualDriveKernel::score_zeta_sensitivity`]).
pub(super) const RESIDUAL_REPAIR_GENERATED_REGRESSOR_CHANNEL: &str =
    "the joint (z, r) covariance escalated to the conditional Σ(a), whose regressions of r on \
     the score and innovation variances are an M-estimate fitted on the calibrated score itself, \
     so every ζ_j moves every row's anchor through Σ(a). The implicit derivative ∂Σ(a)/∂ζ_j of \
     that fit is not implemented; the pooled covariance's channel is";

/// Why a residual block was refused. Each reason names the contract that was
/// not met; none is a solver failure.
#[derive(Clone, Debug, PartialEq)]
pub enum ResidualRepairRefusal {
    /// `E_w[r_k | marginal-index span] = 0` was rejected at level `alpha`.
    ColumnNotCentred {
        column: String,
        p_value: f64,
        alpha: f64,
    },
    /// A feature value was not finite.
    ColumnNonFinite { column: String, row: usize },
    /// A feature column has no weighted variation, so it carries no direction.
    ColumnConstant { column: String },
    /// No residual columns were named.
    Empty,
    /// The feature matrix does not have one row per fitted observation.
    RowCountMismatch { rows: usize, expected: usize },
    /// The residual block is not lowered through the flexible (score-warp /
    /// link-deviation) row program.
    FlexBlocksUnsupported,
    /// A learned frailty scale is an outer axis the residual kernel does not
    /// differentiate; pass a fixed `frailty_sd` instead.
    LearnedFrailtyUnsupported,
    /// The CTN Stage-1 influence absorber widens the marginal block; the
    /// residual kernel reads the marginal design at its raw width.
    InfluenceAbsorberUnsupported,
    /// The centring test of a column could not be evaluated.
    CentringTestUnavailable { column: String, reason: String },
    /// The joint law of `(z, r)` could not be estimated on the fit rows.
    JointCovarianceUnavailable { reason: String },
}

impl std::fmt::Display for ResidualRepairRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColumnNotCentred {
                column,
                p_value,
                alpha,
            } => write!(
                f,
                "residual column '{column}' is not centred on the marginal-index span: the robust \
                 Rao score test of E_w[r | span] = 0 has p = {p_value:.3e} < {alpha:.1e}. Centre the \
                 feature on the reference law (r = φ − E_ref[φ | S, A]) before passing it; the fit \
                 will not absorb the level into the baseline"
            ),
            Self::ColumnNonFinite { column, row } => {
                write!(f, "residual column '{column}' is non-finite at row {row}")
            }
            Self::ColumnConstant { column } => write!(
                f,
                "residual column '{column}' has zero weighted variance on the fit data and carries no \
                 direction"
            ),
            Self::Empty => write!(f, "residual_columns is empty"),
            Self::RowCountMismatch { rows, expected } => write!(
                f,
                "residual feature matrix has {rows} rows but the fit has {expected} observations"
            ),
            Self::FlexBlocksUnsupported => write!(
                f,
                "residual_columns cannot be combined with linkwiggle(...) score-warp / \
                 link-deviation blocks: the residual block is lowered only through the rigid \
                 standard-normal row kernel"
            ),
            Self::LearnedFrailtyUnsupported => write!(
                f,
                "residual_columns requires a fixed frailty_sd (or none): a learned frailty scale is \
                 not differentiated by the residual row kernel"
            ),
            Self::InfluenceAbsorberUnsupported => write!(
                f,
                "residual_columns cannot be combined with an absorbed CTN Stage-1 influence block"
            ),
            Self::CentringTestUnavailable { column, reason } => write!(
                f,
                "the centring test of residual column '{column}' could not be evaluated: {reason}"
            ),
            Self::JointCovarianceUnavailable { reason } => write!(
                f,
                "the joint law of the score and the residual columns could not be estimated: \
                 {reason}"
            ),
        }
    }
}

impl std::error::Error for ResidualRepairRefusal {}

/// A refused block reaches the fit boundary under the category of its reason. A
/// contract the request or its data did not meet is `Input`. A centring test or
/// joint law that could not be evaluated carries its cause only as prose, so it
/// stays `Unclassified` rather than guessed.
impl From<ResidualRepairRefusal> for crate::fit_orchestration::FitFailure {
    fn from(refusal: ResidualRepairRefusal) -> Self {
        let category = match &refusal {
            ResidualRepairRefusal::ColumnNotCentred { .. }
            | ResidualRepairRefusal::ColumnNonFinite { .. }
            | ResidualRepairRefusal::ColumnConstant { .. }
            | ResidualRepairRefusal::Empty
            | ResidualRepairRefusal::RowCountMismatch { .. }
            | ResidualRepairRefusal::FlexBlocksUnsupported
            | ResidualRepairRefusal::LearnedFrailtyUnsupported
            | ResidualRepairRefusal::InfluenceAbsorberUnsupported => {
                gam_problem::FailureCategory::Input
            }
            ResidualRepairRefusal::CentringTestUnavailable { .. }
            | ResidualRepairRefusal::JointCovarianceUnavailable { .. } => {
                gam_problem::FailureCategory::Unclassified
            }
        };
        Self::raised(category, refusal.to_string())
    }
}

/// The residual block as the caller supplies it: named columns, already
/// centred on the reference law, one row per fitted observation.
#[derive(Clone, Debug)]
pub struct ResidualRepairSpec {
    pub columns: Vec<String>,
    /// `n × K`.
    pub features: Array2<f64>,
}

/// The fitted geometry of the residual block: what prediction needs to replay
/// the anchor, persisted with the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualRepairGeometry {
    /// Column names, in coefficient order.
    pub columns: Vec<String>,
    /// Pooled `Var((z, r))` as a dense `(K+1) × (K+1)` matrix, coordinate `0`
    /// the latent score at its declared unit variance. This is the fit's
    /// summary covariance and the object the no-escalation anchor consumes.
    pub pooled_covariance: Vec<Vec<f64>>,
    /// The gam#2766 conditional model `Σ(a)` when the pairwise Rao gate fired
    /// on some `(z, r_j)` or `(r_j, r_k)` pair. `None` ⇒ every row uses the
    /// pooled matrix.
    pub conditional_covariance: Option<ConditionalScoreCovariance>,
    /// The centring-gate p-value of each column, in coefficient order.
    pub centring_pvalues: Vec<f64>,
}

impl ResidualRepairGeometry {
    /// `K`.
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Rebuild the row-wise covariance field over `(z, r)` for a block of
    /// marginal-design rows (fit or predict).
    pub fn covariance_field(
        &self,
        a_block: ArrayView2<'_, f64>,
    ) -> Result<ScoreCovarianceField, String> {
        let dim = self.width() + 1;
        let mut dense = Array2::<f64>::zeros((dim, dim));
        if self.pooled_covariance.len() != dim {
            return Err(format!(
                "residual repair pooled covariance has {} rows, expected {dim}",
                self.pooled_covariance.len()
            ));
        }
        for (i, row) in self.pooled_covariance.iter().enumerate() {
            if row.len() != dim {
                return Err(format!(
                    "residual repair pooled covariance row {i} has {} entries, expected {dim}",
                    row.len()
                ));
            }
            for (j, &value) in row.iter().enumerate() {
                dense[[i, j]] = value;
            }
        }
        let pooled = MarginalSlopeCovariance::full(dense)?;
        match self.conditional_covariance.as_ref() {
            None => Ok(ScoreCovarianceField::pooled(pooled)),
            Some(model) => ScoreCovarianceField::conditional(pooled, model.clone(), a_block),
        }
    }
}

/// The residual block bound to the training rows: what the family's row
/// kernel reads.
pub struct ResidualBlockRuntime {
    /// `n × K` training features.
    pub features: Array2<f64>,
    /// `Var((z, r) | a_i)` at every training row; `K + 1` coordinates.
    pub field: ScoreCovarianceField,
    pub geometry: ResidualRepairGeometry,
    /// Each row's `γ(a_i)` and `2Σ_rr(a_i)` under a row-varying law, built by
    /// the first row kernel over the fit and shared by the rest.
    pub(crate) row_covariance: super::residual_repair_kernel::RowCovarianceSlot,
}

impl std::fmt::Debug for ResidualBlockRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidualBlockRuntime")
            .field("rows", &self.features.nrows())
            .field("width", &self.features.ncols())
            .field("conditional", &self.field.is_conditional())
            .finish()
    }
}

impl ResidualBlockRuntime {
    /// `K`.
    #[inline]
    pub fn width(&self) -> usize {
        self.features.ncols()
    }

    /// Validate, gate, and fit the joint covariance of `(z, r)` on the training
    /// rows.
    ///
    /// `z` is the score the row kernel will actually see (after every latent
    /// calibration), `a_block` the marginal-index span the gam#2768 gates
    /// condition on. Each column is checked for `E_w[r_k | span] = 0` including
    /// the level, with the robust Rao score statistic at
    /// [`AUTO_Z_CONDITIONAL_RAO_ALPHA`]; a failing column is a typed refusal.
    pub fn fit(
        spec: &ResidualRepairSpec,
        z: ArrayView1<'_, f64>,
        weights: ArrayView1<'_, f64>,
        a_block: ArrayView2<'_, f64>,
    ) -> Result<Self, ResidualRepairRefusal> {
        let n = z.len();
        let k = spec.features.ncols();
        if k == 0 || spec.columns.is_empty() {
            return Err(ResidualRepairRefusal::Empty);
        }
        if spec.columns.len() != k {
            return Err(ResidualRepairRefusal::RowCountMismatch {
                rows: spec.columns.len(),
                expected: k,
            });
        }
        if spec.features.nrows() != n || weights.len() != n || a_block.nrows() != n {
            return Err(ResidualRepairRefusal::RowCountMismatch {
                rows: spec.features.nrows(),
                expected: n,
            });
        }
        let total_weight = weights.iter().copied().sum::<f64>();
        for (col, name) in spec.columns.iter().enumerate() {
            let column = spec.features.column(col);
            if let Some(row) = column.iter().position(|v| !v.is_finite()) {
                return Err(ResidualRepairRefusal::ColumnNonFinite {
                    column: name.clone(),
                    row,
                });
            }
            let mean = column
                .iter()
                .zip(weights.iter())
                .map(|(&v, &w)| w * v)
                .sum::<f64>()
                / total_weight;
            let var = column
                .iter()
                .zip(weights.iter())
                .map(|(&v, &w)| w * (v - mean) * (v - mean))
                .sum::<f64>()
                / total_weight;
            let magnitude = column.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            let resolution = gam_linalg::roundoff::accumulation_growth(n) * magnitude;
            if !(var.is_finite() && var.sqrt() > resolution) {
                return Err(ResidualRepairRefusal::ColumnConstant {
                    column: name.clone(),
                });
            }
        }

        // The centring gate. `[1 | ã]` — the level is part of the hypothesis,
        // so the intercept column stays in the tested span; the marginal
        // columns are centred so the statistic's rank is the span's.
        let basis = centring_gate_basis(a_block, weights);
        let mut centring_pvalues = Vec::with_capacity(k);
        for (col, name) in spec.columns.iter().enumerate() {
            let u: Vec<f64> = spec.features.column(col).to_vec();
            let p_value = robust_conditional_score_pvalue(basis.view(), &u, weights)
                .map_err(|reason| ResidualRepairRefusal::CentringTestUnavailable {
                    column: name.clone(),
                    reason,
                })?
                .unwrap_or(1.0);
            if p_value < AUTO_Z_CONDITIONAL_RAO_ALPHA {
                return Err(ResidualRepairRefusal::ColumnNotCentred {
                    column: name.clone(),
                    p_value,
                    alpha: AUTO_Z_CONDITIONAL_RAO_ALPHA,
                });
            }
            centring_pvalues.push(p_value);
        }

        // The joint law of (z, r): the pooled second central moment, escalated
        // to Σ(a) by the gam#2766 pairwise gate on the same span, then moved
        // onto the DECLARED law of the score. Read as the modified Cholesky
        // factorisation `Σ = L·diag(d₀, d₁, …)·Lᵀ` with the score first, `L`'s
        // first column is the regression of r on z and `d₁, …` the innovation
        // variances of r given z: together, the conditional law of r given z.
        // `d₀` is Var(z | a), which the latent measure declares rather than
        // estimates — N(0, 1) here — so it is set to 1 and nothing else moves.
        // At β = 0 the anchor is then the rigid kernel's √(1 + s²g²) exactly: a
        // block with nothing to read leaves the score-only fit where it was
        // instead of moving its baseline by the sampling error of Var(z).
        let unavailable =
            |reason: String| ResidualRepairRefusal::JointCovarianceUnavailable { reason };
        let mut scores = Array2::<f64>::zeros((n, k + 1));
        scores.column_mut(0).assign(&z);
        scores.slice_mut(s![.., 1..]).assign(&spec.features);
        let weights_owned = weights.to_owned();
        let sample = marginal_slope_covariance_from_scores(scores.view(), &weights_owned)
            .map_err(unavailable)?
            .to_dense();
        let pooled_dense = declare_unit_score_variance(&sample).map_err(unavailable)?;
        let pooled = MarginalSlopeCovariance::full(pooled_dense.clone()).map_err(unavailable)?;
        let conditional = ConditionalScoreCovariance::fit(scores.view(), weights, a_block)
            .map_err(unavailable)?
            .map(|mut model| {
                // Coordinate 0 of the MCD is the score: its innovation IS
                // Var(z | a). Declare it, keep every regression and every
                // residual innovation the data estimated.
                model.coordinates[0].log_innovation = vec![0.0];
                model.coordinates[0].log_innovation_range = [0.0, 0.0];
                model
            });
        let field = match conditional.as_ref() {
            None => ScoreCovarianceField::pooled(pooled),
            Some(model) => ScoreCovarianceField::conditional(pooled, model.clone(), a_block)
                .map_err(unavailable)?,
        };
        if let Some(model) = conditional.as_ref() {
            log::debug!(
                "[BMS residual repair] joint (z, r) covariance escalated to Σ(a): {} pair(s) fired \
                 the conditional gate",
                model
                    .pair_pvalues
                    .iter()
                    .filter(|(_, _, p)| *p < AUTO_Z_CONDITIONAL_RAO_ALPHA)
                    .count()
            );
        }
        Ok(Self {
            features: spec.features.clone(),
            field,
            geometry: ResidualRepairGeometry {
                columns: spec.columns.clone(),
                pooled_covariance: pooled_dense
                    .rows()
                    .into_iter()
                    .map(|row| row.to_vec())
                    .collect(),
                conditional_covariance: conditional,
                centring_pvalues,
            },
            row_covariance: Default::default(),
        })
    }

    /// Rebind a persisted geometry to a block of rows (prediction, ALO replay).
    pub fn from_geometry(
        geometry: ResidualRepairGeometry,
        features: Array2<f64>,
        a_block: ArrayView2<'_, f64>,
    ) -> Result<Self, String> {
        if features.ncols() != geometry.width() {
            return Err(format!(
                "residual repair feature width {} does not match the saved block width {}",
                features.ncols(),
                geometry.width()
            ));
        }
        if features.nrows() != a_block.nrows() {
            return Err(format!(
                "residual repair feature rows {} do not match the conditioning rows {}",
                features.nrows(),
                a_block.nrows()
            ));
        }
        let field = geometry.covariance_field(a_block)?;
        Ok(Self {
            features,
            field,
            geometry,
            row_covariance: Default::default(),
        })
    }

    /// The ridge block spec: identity penalty on `β`, one REML/LAML coordinate.
    pub(super) fn block_spec(
        &self,
        rho: Array1<f64>,
        beta_hint: Option<Array1<f64>>,
    ) -> Result<ParameterBlockSpec, String> {
        let k = self.width();
        if rho.len() != 1 {
            return Err(format!(
                "residual repair block takes exactly one smoothing coordinate, got {}",
                rho.len()
            ));
        }
        let initial_beta = match beta_hint {
            Some(beta) if beta.len() == k => Some(beta),
            _ => Some(Array1::zeros(k)),
        };
        Ok(ParameterBlockSpec {
            name: RESIDUAL_BLOCK_NAME.to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                self.features.clone(),
            )),
            offset: Array1::zeros(self.features.nrows()),
            penalties: vec![PenaltyMatrix::Diagonal(Array1::from_elem(k, 1.0))],
            nullspace_dims: vec![0],
            initial_log_lambdas: rho,
            initial_beta,
            gauge_priority: GAUGE_PRIORITY_RESIDUAL,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        })
    }
}

/// `[1 | ã]`: the intercept plus the weighted-centred marginal columns, the span
/// the centring gate tests against.
fn centring_gate_basis(a_block: ArrayView2<'_, f64>, weights: ArrayView1<'_, f64>) -> Array2<f64> {
    let n = a_block.nrows();
    let p = a_block.ncols();
    let total_weight = weights.iter().copied().sum::<f64>();
    let mut basis = Array2::<f64>::zeros((n, p + 1));
    basis.column_mut(0).fill(1.0);
    for col in 0..p {
        let column = a_block.column(col);
        let mean = column
            .iter()
            .zip(weights.iter())
            .map(|(&v, &w)| w * v)
            .sum::<f64>()
            / total_weight;
        Zip::from(basis.column_mut(col + 1))
            .and(column)
            .for_each(|out, &v| *out = v - mean);
    }
    basis
}

/// The joint covariance of `(z, r)` on the score's declared law: the sample
/// `Σ̂ = L·diag(d₀, d₁, …)·Lᵀ` with `d₀` replaced by the declared `Var(z) = 1`.
/// Only `L`'s first column, the regression `γ = Σ̂_r0/Σ̂_00` of r on z, and the
/// Schur complement `Σ̂_rr − Σ̂_r0Σ̂_0r/Σ̂_00`, the law of r given z, survive:
///
/// ```text
///   Σ_00 = 1,   Σ_r0 = γ,   Σ_rr = Σ̂_rr − Σ̂_r0Σ̂_0r/Σ̂_00 + γγᵀ
/// ```
///
/// A Schur complement of a PSD matrix plus a rank-one PSD term, so the result
/// is PSD whenever the sample is.
pub(super) fn declare_unit_score_variance(sample: &Array2<f64>) -> Result<Array2<f64>, String> {
    let dim = sample.nrows();
    let score_variance = sample[[0, 0]];
    if !(score_variance.is_finite() && score_variance > 0.0) {
        return Err(format!("the score has weighted variance {score_variance}"));
    }
    let inv = score_variance.recip();
    // γγᵀ − Σ̂_r0Σ̂_0r/Σ̂_00 = Σ̂_r0Σ̂_0r·(1/Σ̂_00² − 1/Σ̂_00).
    let rank_one = inv * inv - inv;
    let mut declared = Array2::<f64>::zeros((dim, dim));
    declared[[0, 0]] = 1.0;
    for i in 1..dim {
        declared[[i, 0]] = sample[[i, 0]] * inv;
        declared[[0, i]] = sample[[0, i]] * inv;
        for j in 1..dim {
            declared[[i, j]] = sample[[i, j]] + sample[[i, 0]] * sample[[0, j]] * rank_one;
        }
    }
    Ok(declared)
}

/// Derivative stack of `u ↦ √(1 + u)` at `u`, given `f = √(1 + u)`.
#[inline]
fn sqrt1p_stack(f: f64) -> [f64; 5] {
    let inv = f.recip();
    let inv3 = inv * inv * inv;
    let inv5 = inv3 * inv * inv;
    let inv7 = inv5 * inv * inv;
    [
        f,
        0.5 * inv,
        -0.25 * inv3,
        0.375 * inv5,
        -0.9375 * inv7,
    ]
}

/// The anchored intercept of a finite mixture of Gaussian drives, with its
/// derivatives through order four in `(q, B)`.
///
/// Under a declared finite law `z ∼ Σ_k w_k δ_{z_k}` of the score and the
/// Gaussian law `r | z, a ∼ N(γz, Σ_{r·z})` of the residual block — `γ` and
/// `Σ_{r·z}` are the first column and the Schur complement of the declared joint
/// covariance — the genetic drive `s(g z + βᵀr)` given node `k` is `N(m·z_k, v)`
/// with `m = s(g + βᵀγ)` and `v = s²·βᵀΣ_{r·z}β`: a finite mixture of Gaussians.
/// Averaging each component (`Descent.Portability.GaussianAnchor.drive_gaussianAverage`)
/// turns the anchoring equation `E[Φ(α + d)] = Φ(q)` into
///
/// ```text
///   Σ_k w_k Φ(ã + B·z_k) = Φ(q),     α = τ·ã,   B = m/τ,   τ = √(1 + v)
/// ```
///
/// — the probit anchor on a finite declared law, whose root is unique
/// (`Descent.Portability.ProbitAnchor.exists_unique_anchor_probit`).
///
/// The value is the log-space root of the smaller tail the rigid empirical
/// kernel solves ([`solve_anchor`]); the derivative channels are the anchor's
/// Taylor table at that root ([`AnchorTaylor`]) composed with the seeded
/// `(q, B)`, normalized by the grid density so a tail index whose every node
/// density underflows keeps finite exact derivatives (gam#2978). Its first
/// order is `∂ã/∂B = −E[φ·z]/E[φ]` (`ProbitAnchor.anchor_deriv_eq_probit`).
/// The value channel is the root itself, so every path that reads the same
/// root agrees with it bit for bit.
pub(crate) fn mixture_anchor_tower(
    q: f64,
    b: f64,
    grid: &EmpiricalZGrid,
) -> Result<Tower4<2>, String> {
    let owned = AnchorGridOwned::from_grid(grid);
    let root = solve_anchor(q, b, owned.view())?;
    let taylor = AnchorTaylor::at(root, q, b, owned.view())?;
    let q_var = <Tower4<2> as JetScalar<2>>::variable(q, 0);
    let b_var = <Tower4<2> as JetScalar<2>>::variable(b, 1);
    Ok(taylor.lift(&q_var, &b_var.with_value(0.0)))
}

/// Compose a bivariate tower `f(q, B)` with the jets `Q` and `B` through order
/// four: the Taylor polynomial of `f` about `(Q.value(), B.value())` at the jets'
/// zero-valued displacements. A displacement's fifth power vanishes in every
/// truncation, so this is exact on each channel a jet carries.
fn compose_bivariate_tower<const K: usize, S: JetScalar<K>>(f: &Tower4<2>, q: &S, b: &S) -> S {
    let dq = q.with_value(0.0);
    let db = b.with_value(0.0);
    let dq2 = dq.mul(&dq);
    let dq3 = dq2.mul(&dq);
    let db2 = db.mul(&db);
    let db3 = db2.mul(&db);
    let monomials = [
        dq,
        db,
        dq2,
        dq.mul(&db),
        db2,
        dq3,
        dq2.mul(&db),
        dq.mul(&db2),
        db3,
        dq3.mul(&dq),
        dq3.mul(&db),
        dq2.mul(&db2),
        dq.mul(&db3),
        db3.mul(&db),
    ];
    // ∂^{i+j}f/∂q^i∂B^j / (i!·j!) for each monomial dq^i·dB^j above.
    let coefficients = [
        f.g[0],
        f.g[1],
        f.h[0][0] / 2.0,
        f.h[0][1],
        f.h[1][1] / 2.0,
        f.t3[0][0][0] / 6.0,
        f.t3[0][0][1] / 2.0,
        f.t3[0][1][1] / 2.0,
        f.t3[1][1][1] / 6.0,
        f.t4[0][0][0][0] / 24.0,
        f.t4[0][0][0][1] / 6.0,
        f.t4[0][0][1][1] / 4.0,
        f.t4[0][1][1][1] / 6.0,
        f.t4[1][1][1][1] / 24.0,
    ];
    S::linear_combination(&monomials, &coefficients).add_constant(f.v)
}

/// One row's data as the residual row likelihood reads it.
pub(super) struct ResidualRowState<'a> {
    pub(super) marginal: BernoulliMarginalLinkMap,
    pub(super) z: f64,
    pub(super) y: f64,
    pub(super) w: f64,
    pub(super) probit_scale: f64,
    /// `None`: the standard-normal law of the score. `Some`: its declared
    /// finite law at this row (global, or this row's local mixture).
    pub(super) grid: Option<&'a EmpiricalZGrid>,
}

/// The residual drive of a row as the likelihood reads it: the linear read
/// `t = βᵀr` and the anchor's drive moments `u = βᵀγ`, `v = βᵀΣ_rrβ` under the
/// declared joint covariance (`Σ₀₀ = 1`, `γ = Σ_r0`).
pub(super) struct ResidualDrive<S> {
    pub(super) t: S,
    pub(super) u: S,
    pub(super) v: S,
}

/// The row negative log-likelihood from the marginal predictor `η_m`, the slope
/// `g` and the residual drive: the ONE statement of the joint anchor that the
/// five-primary kernel (`residual_repair_kernel`), the finite-difference gates
/// and the plain-`f64` index [`residual_row_index`] all read.
///
/// Standard-normal score: `η = c·q + s(g z + t)`, `c = √(1 + s²(g² + 2g·u + v))`,
/// which is `√(1 + s²·b̃ᵀΣb̃)` because `Σ₀₀ = 1`. Declared finite law:
/// `η = τ·ã(q, B) + s(g z + t)` from [`mixture_anchor_tower`], with
/// `m = s(g + u)` and `τ² = 1 + s²(v − u²)`, because
/// `b̃ᵀΣb̃ = (g + βᵀγ)² + βᵀΣ_{r·z}β`. On a standard-normal grid the two branches
/// agree to quadrature tolerance.
pub(super) fn residual_row_nll<const N: usize, S: JetScalar<N>>(
    state: &ResidualRowState<'_>,
    eta_m: &S,
    g: &S,
    drive: &ResidualDrive<S>,
) -> Result<S, String> {
    residual_row_nll_perturbed(state, eta_m, g, drive, None).map(|eval| eval.nll)
}

/// First-order perturbations of what a row's likelihood reads from the
/// calibrated score ζ, as extra jet primaries (gam#2985): the row's own score,
/// and the finite law's anchor `ã(q, B)` through its value and its two slopes.
///
/// The generated-regressor correction differentiates the β-score `∇_β ℓ`, which
/// reads the anchor only through `ã`, `∂ã/∂q` and `∂ã/∂B` at the row's
/// `(q, B)`. So `ã + ε_A + ε_q·δq + ε_B·δB` carries exactly the directions in
/// which moving the law's nodes can move that score, and the mixed second
/// derivatives `∂²ℓ/∂p∂ε` are the score's node sensitivities up to the anchor's
/// own node derivatives ([`mixture_anchor_node_sensitivity`]).
pub(super) struct ScorePerturbation<S> {
    /// `δζ_i`, entering the linear read as `s·g·δζ_i`.
    pub(super) zeta: S,
    /// `(ε_A, ε_q, ε_B)`: on a declared finite law only.
    pub(super) anchor: [S; 3],
}

/// A row's negative log-likelihood, and on a declared finite law the `(q, B)`
/// at which its anchor was solved.
pub(super) struct ResidualRowEval<S> {
    pub(super) nll: S,
    pub(super) anchor_point: Option<[f64; 2]>,
}

/// [`residual_row_nll`] with the score perturbations of [`ScorePerturbation`]
/// added to what it reads; `None` is `residual_row_nll` itself.
pub(super) fn residual_row_nll_perturbed<const N: usize, S: JetScalar<N>>(
    state: &ResidualRowState<'_>,
    eta_m: &S,
    g: &S,
    drive: &ResidualDrive<S>,
    perturbation: Option<&ScorePerturbation<S>>,
) -> Result<ResidualRowEval<S>, String> {
    let s = state.probit_scale;
    let marginal = state.marginal;
    // q = Φ⁻¹(Φ(η_m)) through the supplied link stack.
    let q = eta_m.compose_unary([
        marginal.q,
        marginal.q1,
        marginal.q2,
        marginal.q3,
        marginal.q4,
    ]);
    let quad = g
        .mul(g)
        .add(&g.mul(&drive.u).scale(2.0))
        .add(&drive.v)
        .scale(s * s);
    let quad_value = quad.value();
    if !(quad_value.is_finite() && quad_value >= -f64::EPSILON * (1.0 + quad_value.abs())) {
        return Err(format!(
            "residual repair row: the anchor quadratic form b̃ᵀΣb̃ = {quad_value} is not admissible"
        ));
    }
    let mut linear = g.scale(s * state.z).add(&drive.t.scale(s));
    if let Some(perturbation) = perturbation {
        linear = linear.add(&g.mul(&perturbation.zeta).scale(s));
    }
    let mut anchor_point = None;
    let eta = match state.grid {
        None => {
            let c = quad.compose_unary(sqrt1p_stack((1.0 + quad_value.max(0.0)).sqrt()));
            q.multiply_add(&c, &linear)
        }
        Some(grid) => {
            let m = g.add(&drive.u).scale(s);
            let v = drive.v.sub(&drive.u.mul(&drive.u)).scale(s * s);
            let v_value = v.value();
            if !(v_value.is_finite() && v_value >= -f64::EPSILON * (1.0 + quad_value.abs())) {
                return Err(format!(
                    "residual repair row: the residual drive variance βᵀΣ_{{r·z}}β = {v_value} is not \
                     admissible"
                ));
            }
            let tau = v.compose_unary(sqrt1p_stack((1.0 + v_value.max(0.0)).sqrt()));
            let b_scaled = m.mul(&tau.recip());
            let tower = mixture_anchor_tower(marginal.q, b_scaled.value(), grid)?;
            let mut anchor = compose_bivariate_tower(&tower, &q, &b_scaled);
            if let Some(perturbation) = perturbation {
                let [value, slope_q, slope_b] = &perturbation.anchor;
                anchor = anchor
                    .add(value)
                    .add(&slope_q.mul(&q.with_value(0.0)))
                    .add(&slope_b.mul(&b_scaled.with_value(0.0)));
            }
            anchor_point = Some([marginal.q, b_scaled.value()]);
            anchor.multiply_add(&tau, &linear)
        }
    };
    let margin = eta.scale(2.0 * state.y - 1.0);
    let nll = margin.compose_unary(signed_probit_neglog_unary_stack(margin.value(), state.w));
    if !nll.value().is_finite() {
        return Err(format!(
            "residual repair row: non-finite log Φ at η={}, y={}, w={}",
            eta.value(),
            state.y,
            state.w
        ));
    }
    Ok(ResidualRowEval { nll, anchor_point })
}

/// The derivatives of the finite-law anchor `ã(q, B)` and of its two slopes in
/// each node `x_k` of the law, at fixed weights (gam#2985): `[∂ã/∂x_k,
/// ∂ã_q/∂x_k, ∂ã_B/∂x_k]`.
///
/// Implicit differentiation of `Σ_k w_k Φ(ã + B x_k) = Φ(q)`, with `ω` the
/// law's density weights at the root ([`AnchorDensity`]), `e_k = ã + B x_k`,
/// `x̄ = Σω x`, `ē = Σω e` and `C = Σω (x − x̄)(e − ē)`:
///
/// ```text
///   ∂ã/∂x_k   = −B ω_k
///   ∂ã_q/∂x_k = ã_q·B ω_k (e_k − ē)
///   ∂ã_B/∂x_k = −ω_k − B ω_k (C − (x_k − x̄) e_k)
/// ```
///
/// A shift of every node by `δ` moves the root by `−Bδ`, `ã_q` not at all and
/// `ã_B = −x̄` by `−δ`: the three rows sum to `−B`, `0` and `−1`.
pub(super) fn mixture_anchor_node_sensitivity(
    q: f64,
    b: f64,
    grid: &EmpiricalZGrid,
) -> Result<[Vec<f64>; 3], String> {
    let owned = AnchorGridOwned::from_grid(grid);
    let view = owned.view();
    let root = solve_anchor(q, b, view)?;
    let density = AnchorDensity::at(root, b, view)?;
    let a_q = AnchorTaylor::at(root, q, b, view)?.derivatives().a_q;
    let omega = density.weights();
    let nodes = &owned.nodes;
    let x_bar: f64 = omega.iter().zip(nodes).map(|(w, x)| w * x).sum();
    let e_bar = root + b * x_bar;
    let covariance: f64 = omega
        .iter()
        .zip(nodes)
        .map(|(w, x)| w * (x - x_bar) * (root + b * x - e_bar))
        .sum();
    let mut value = Vec::with_capacity(nodes.len());
    let mut slope_q = Vec::with_capacity(nodes.len());
    let mut slope_b = Vec::with_capacity(nodes.len());
    for (&w, &x) in omega.iter().zip(nodes) {
        let e = root + b * x;
        value.push(-b * w);
        slope_q.push(a_q * b * w * (e - e_bar));
        slope_b.push(-w - b * w * (covariance - (x - x_bar) * e));
    }
    Ok([value, slope_q, slope_b])
}

/// The row index and its first derivatives in plain `f64`: the value-only
/// path of the fit and the prediction replay share this one statement of the
/// anchor with the jet program [`residual_row_nll`].
///
/// Returns `(η, ∂η/∂q_eta, ∂η/∂g, ∂η/∂β)`.
pub(crate) fn residual_row_index(
    marginal: &BernoulliMarginalLinkMap,
    g: f64,
    beta: &[f64],
    z: f64,
    r: &[f64],
    covariance: &MarginalSlopeCovariance,
    grid: Option<&EmpiricalZGrid>,
    probit_scale: f64,
) -> Result<(f64, f64, f64, Vec<f64>), String> {
    let k = beta.len();
    if r.len() != k || covariance.dim() != k + 1 {
        return Err(format!(
            "residual row index dimension mismatch: beta={k}, r={}, covariance={}",
            r.len(),
            covariance.dim()
        ));
    }
    let s = probit_scale;
    let mut drive = Vec::with_capacity(k + 1);
    drive.push(s * g);
    drive.extend(beta.iter().map(|&b| s * b));
    let mut sigma_drive = vec![0.0; k + 1];
    covariance.multiply(&drive, &mut sigma_drive);
    let quad: f64 = drive.iter().zip(sigma_drive.iter()).map(|(a, b)| a * b).sum();
    if !(quad.is_finite() && quad >= -f64::EPSILON * (1.0 + quad.abs())) {
        return Err(format!(
            "residual row index: the anchor quadratic form b̃ᵀΣb̃ = {quad} is not admissible"
        ));
    }
    let linear = drive[0] * z + drive[1..].iter().zip(r.iter()).map(|(b, x)| b * x).sum::<f64>();
    let Some(grid) = grid else {
        let c = (1.0 + quad.max(0.0)).sqrt();
        let inv_c = c.recip();
        let eta = marginal.q * c + linear;
        let d_q = marginal.q1 * c;
        // ∂c/∂g = s·(Σ b̃)_0 / c ; ∂c/∂β_k = s·(Σ b̃)_{k+1} / c   (b̃ already carries s).
        let d_g = marginal.q * s * sigma_drive[0] * inv_c + s * z;
        let d_beta: Vec<f64> = (0..k)
            .map(|j| marginal.q * s * sigma_drive[j + 1] * inv_c + s * r[j])
            .collect();
        return Ok((eta, d_q, d_g, d_beta));
    };
    // The declared finite law: m = (Σ b̃)₀ = s(g + βᵀγ), v = b̃ᵀΣb̃ − m², τ = √(1 + v).
    let m = sigma_drive[0];
    let v = quad - m * m;
    if !(v.is_finite() && v >= -f64::EPSILON * (1.0 + quad.abs())) {
        return Err(format!(
            "residual row index: the residual drive variance βᵀΣ_{{r·z}}β = {v} is not admissible"
        ));
    }
    let tau = (1.0 + v.max(0.0)).sqrt();
    let b_scaled = m / tau;
    let tower = mixture_anchor_tower(marginal.q, b_scaled, grid)?;
    let (intercept, intercept_q, intercept_b) = (tower.v, tower.g[0], tower.g[1]);
    let eta = tau * intercept + linear;
    let d_q = tau * intercept_q * marginal.q1;
    // ∂m/∂g = s, ∂v/∂g = 0; ∂m/∂β_j = s·γ_j, ∂v/∂β_j = 2s·((Σ b̃)_{j+1} − m·γ_j).
    let d_g = s * intercept_b + s * z;
    let d_beta: Vec<f64> = (0..k)
        .map(|j| {
            let gamma = covariance.coefficient(0, j + 1);
            let d_m = s * gamma;
            let d_v = 2.0 * s * (sigma_drive[j + 1] - m * gamma);
            let d_tau = 0.5 * d_v / tau;
            let d_b = d_m / tau - m * d_tau / (tau * tau);
            d_tau * intercept + tau * intercept_b * d_b + s * r[j]
        })
        .collect();
    Ok((eta, d_q, d_g, d_beta))
}

/// A row's joint `(z, r)` anchor read through the score alone (gam#2985).
///
/// Under the fit's model of the residual given the score, with `Σ₀₀ = 1`,
/// `βᵀr | z ~ N(u z, v − u²)` for `u = βᵀγ(a)` and `v = βᵀΣ_rr(a)β`. At the row's
/// index intercept `α` (its index at `z = 0`, `r = 0`), the residual integrates out
/// in closed form:
///
/// ```text
///   E_{r|z}[Φ(α + s(g z + βᵀr))] = Φ(ã + B z),   ã = α/τ,   B = m/τ,
///   m = s(g + u),   τ = √(1 + s²(v − u²)).
/// ```
///
/// So the joint anchor under any candidate law of the score is the score-only
/// anchor with this row's intercept `ã` and slope `B`, and a latent-law
/// certificate that reads `Φ(ã + B u)` over a law's nodes evaluates the joint
/// anchor unchanged. `m` and `τ` are formed exactly as [`residual_row_index`]
/// forms them on a finite law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct JointAnchorOnScore {
    /// `B = m/τ`.
    pub(crate) slope: f64,
    /// `τ = √(1 + s²(v − u²))`.
    pub(crate) scale: f64,
}

impl JointAnchorOnScore {
    pub(crate) fn at(
        g: f64,
        beta: &[f64],
        covariance: &MarginalSlopeCovariance,
        probit_scale: f64,
    ) -> Result<Self, String> {
        let k = beta.len();
        if covariance.dim() != k + 1 {
            return Err(format!(
                "joint anchor on the score: covariance dimension {} for {k} residual coefficients",
                covariance.dim()
            ));
        }
        let s = probit_scale;
        let mut drive = Vec::with_capacity(k + 1);
        drive.push(s * g);
        drive.extend(beta.iter().map(|&b| s * b));
        let mut sigma_drive = vec![0.0; k + 1];
        covariance.multiply(&drive, &mut sigma_drive);
        let quad: f64 = drive.iter().zip(sigma_drive.iter()).map(|(a, b)| a * b).sum();
        let m = sigma_drive[0];
        let v = quad - m * m;
        if !(v.is_finite() && v >= -f64::EPSILON * (1.0 + quad.abs())) {
            return Err(format!(
                "joint anchor on the score: the residual drive variance βᵀΣ_{{r·z}}β = {v} is not \
                 admissible"
            ));
        }
        let scale = (1.0 + v.max(0.0)).sqrt();
        Ok(Self {
            slope: m / scale,
            scale,
        })
    }

    /// `ã = α/τ` for the row's index intercept `α`.
    pub(crate) fn score_intercept(&self, alpha: f64) -> f64 {
        alpha / self.scale
    }

    /// Under the candidate law `law` of the score, `(Σ_k w_k Φ(ã + B u_k) − μ, the
    /// standard deviation of Φ(ã + B U), μ)`: the anchoring residual the
    /// closed-form certificate reads, on the joint anchor.
    pub(crate) fn anchoring_residual(
        &self,
        alpha: f64,
        mu: f64,
        law: &EmpiricalZGrid,
    ) -> Result<(f64, f64, f64), String> {
        let intercept = self.score_intercept(alpha);
        let probabilities: Vec<(f64, f64)> = law
            .pairs()
            .map(|(node, weight)| (weight, normal_cdf(intercept + self.slope * node)))
            .collect();
        let mean: f64 = probabilities.iter().map(|&(w, p)| w * p).sum();
        let variance: f64 = probabilities
            .iter()
            .map(|&(w, p)| w * (p - mean) * (p - mean))
            .sum();
        if !(mean.is_finite() && variance.is_finite()) {
            return Err(format!(
                "joint anchor on the score: the anchoring residual is not finite: mean={mean}, \
                 variance={variance} at α={alpha}"
            ));
        }
        Ok((mean - mu, variance.sqrt(), mu))
    }
}

/// One training row of a fit with a residual block, as the latent-law
/// certificates read it (gam#2985): the row's index intercept `α` under the law it
/// was fitted on (its index at `z = 0`, `r = 0`), its index at the observed
/// `(z, r_i)`, `μ = Φ(q)`, and its joint anchor read on the score. `z` is the row's
/// score on the axis the certificate scores it on.
pub(super) struct ResidualCertificateRow {
    pub(super) alpha: f64,
    pub(super) observed_index: f64,
    pub(super) mu: f64,
    pub(super) anchor: JointAnchorOnScore,
}

pub(super) fn residual_certificate_row(
    family: &BernoulliMarginalSlopeFamily,
    runtime: &ResidualBlockRuntime,
    block_states: &[ParameterBlockState],
    row: usize,
    z: f64,
) -> Result<ResidualCertificateRow, String> {
    let marginal = family.marginal_link_map(block_states[0].eta[row])?;
    let g = block_states[1].eta[row];
    let beta = block_states[2].beta.as_slice().ok_or("residual beta not contiguous")?;
    let r = runtime.features.row(row);
    let r = r.as_slice().ok_or("residual feature row not contiguous")?;
    let grid = family.latent_measure.empirical_grid_for_training_row(row)?;
    let covariance = runtime.field.at_row(row);
    let s = family.probit_frailty_scale();
    let origin = vec![0.0; beta.len()];
    let (alpha, _, _, _) = residual_row_index(&marginal, g, beta, 0.0, &origin, covariance, grid.as_deref(), s)?;
    let (observed_index, _, _, _) =
        residual_row_index(&marginal, g, beta, z, r, covariance, grid.as_deref(), s)?;
    Ok(ResidualCertificateRow {
        alpha,
        observed_index,
        mu: marginal.mu,
        anchor: JointAnchorOnScore::at(g, beta, covariance, s)?,
    })
}

/// Value-only row negative log-likelihood, bit-equivalent to the jet program's
/// value channel at the same row state.
pub(super) fn residual_row_neglog_only(
    family: &BernoulliMarginalSlopeFamily,
    runtime: &ResidualBlockRuntime,
    block_states: &[ParameterBlockState],
    row: usize,
) -> Result<f64, String> {
    let marginal = family.marginal_link_map(block_states[0].eta[row])?;
    let g = block_states[1].eta[row];
    let beta = block_states[2].beta.as_slice().ok_or("residual beta not contiguous")?;
    let r = runtime.features.row(row);
    let r = r.as_slice().ok_or("residual feature row not contiguous")?;
    let grid = family.latent_measure.empirical_grid_for_training_row(row)?;
    let (eta, _, _, _) = residual_row_index(
        &marginal,
        g,
        beta,
        family.z[row],
        r,
        runtime.field.at_row(row),
        grid.as_deref(),
        family.probit_frailty_scale(),
    )?;
    let w = family.weights[row];
    let sign = 2.0 * family.y[row] - 1.0;
    let stack = signed_probit_neglog_unary_stack(sign * eta, w);
    if !stack[0].is_finite() {
        return Err(format!(
            "residual repair row {row}: non-finite log Φ at η={eta}, y={}, w={w}",
            family.y[row]
        ));
    }
    Ok(stack[0])
}

impl BernoulliMarginalSlopeFamily {
    /// Whether the family carries a residual block.
    #[inline]
    pub(super) fn residual_active(&self) -> bool {
        self.residual.is_some()
    }

    /// Index of the residual block in the parameter-block list.
    #[inline]
    pub(super) fn residual_block_index(&self) -> Option<usize> {
        self.residual.as_ref().map(|_| 2)
    }

    /// Block-diagonal working sets for the inner block-coordinate solver,
    /// from the residual kernel's generic joint assembly.
    pub(super) fn evaluate_residual_block_diagonals(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let kern = super::residual_repair_kernel::ResidualDriveKernel::new(
            self.clone(),
            block_states.to_vec(),
        )?;
        let rows = crate::row_kernel::RowSet::All;
        let cache = crate::row_kernel::build_row_kernel_cache(&kern, &rows)?;
        let log_likelihood = crate::row_kernel::row_kernel_log_likelihood(&cache, &rows);
        let gradient = crate::row_kernel::row_kernel_gradient(&kern, &cache, &rows);
        let hessian = super::residual_repair_kernel::residual_hessian_dense(&kern, &cache)?;
        let slices = super::hessian_paths::block_slices(self);
        let residual = slices
            .residual
            .clone()
            .ok_or("residual block diagonals requested without a residual range")?;
        let block = |range: std::ops::Range<usize>| BlockWorkingSet::ExactNewton {
            gradient: Self::exact_newton_score_from_objective_gradient(
                gradient.slice(s![range.clone()]).to_owned(),
            ),
            hessian: SymmetricMatrix::Dense(hessian.slice(s![range.clone(), range]).to_owned()),
        };
        Ok(FamilyEvaluation {
            log_likelihood,
            blockworking_sets: vec![
                block(slices.marginal.clone()),
                block(slices.slope.clone()),
                block(residual),
            ],
        })
    }
}

#[cfg(test)]
mod residual_repair_kernel_tests {
    //! Finite-difference gates on the residual row program: the jet-derived
    //! gradient and Hessian in primary space against central differences of
    //! the value channel, and the plain-`f64` index derivatives against the
    //! same program. These pin the ONE statement of the anchor that the fit,
    //! the value-only line search and the prediction replay all read.

    use super::super::tests_residual_repair_laws::{ResidualBlock, covariance, hermite_grid, skewed_grid};
    use super::*;
    use gam_math::jet_scalar::Order2;
    use gam_math::jet_tower::RowProgram;

    #[test]
    fn refusals_reach_the_fit_boundary_under_their_category() {
        use crate::fit_orchestration::FitFailure;
        use gam_problem::FailureCategory;
        for refusal in [
            ResidualRepairRefusal::FlexBlocksUnsupported,
            ResidualRepairRefusal::LearnedFrailtyUnsupported,
            ResidualRepairRefusal::InfluenceAbsorberUnsupported,
            ResidualRepairRefusal::ColumnConstant {
                column: "r1".to_string(),
            },
        ] {
            let text = refusal.to_string();
            let failure = FitFailure::from(refusal);
            assert_eq!(failure.category(), FailureCategory::Input, "{text}");
            assert_eq!(failure.to_string(), text);
        }
        let unevaluated = FitFailure::from(ResidualRepairRefusal::JointCovarianceUnavailable {
            reason: "not SPD".to_string(),
        });
        assert_eq!(unevaluated.category(), FailureCategory::Unclassified);
    }

    #[test]
    fn plain_index_derivatives_match_central_differences() {
        let link = InverseLink::Standard(StandardLink::Probit);
        let skewed = skewed_grid();
        for grid in [None, Some(&skewed)] {
            for (k, eta_m, g, z, s) in [(1usize, 0.3, 0.4, -0.7, 1.0), (3, -0.8, -0.5, 1.2, 0.8)] {
                let cov = covariance(k, 11 + k as u64);
                let beta: Vec<f64> = (0..k).map(|j| 0.3 - 0.2 * j as f64).collect();
                let r: Vec<f64> = (0..k).map(|j| 0.5 * (j as f64 + 1.0) - 0.9).collect();
                let marginal = bernoulli_marginal_link_map(&link, eta_m).unwrap();
                let (_, d_q, d_g, d_beta) =
                    residual_row_index(&marginal, g, &beta, z, &r, &cov, grid, s).unwrap();
                let h = 1.0e-5;
                let value = |eta_m: f64, g: f64, beta: &[f64]| {
                    let marginal = bernoulli_marginal_link_map(&link, eta_m).unwrap();
                    residual_row_index(&marginal, g, beta, z, &r, &cov, grid, s).unwrap().0
                };
                let label = if grid.is_some() { "skewed law" } else { "normal law" };
                let fd_q = (value(eta_m + h, g, &beta) - value(eta_m - h, g, &beta)) / (2.0 * h);
                let fd_g = (value(eta_m, g + h, &beta) - value(eta_m, g - h, &beta)) / (2.0 * h);
                assert!((fd_q - d_q).abs() < 1.0e-7, "{label} k={k}: ∂η/∂q {d_q} vs fd {fd_q}");
                assert!((fd_g - d_g).abs() < 1.0e-7, "{label} k={k}: ∂η/∂g {d_g} vs fd {fd_g}");
                for j in 0..k {
                    let mut plus = beta.clone();
                    let mut minus = beta.clone();
                    plus[j] += h;
                    minus[j] -= h;
                    let fd = (value(eta_m, g, &plus) - value(eta_m, g, &minus)) / (2.0 * h);
                    assert!(
                        (fd - d_beta[j]).abs() < 1.0e-7,
                        "{label} k={k}: ∂η/∂β_{j} {} vs fd {fd}",
                        d_beta[j]
                    );
                }
            }
        }
    }

    #[test]
    fn mixture_anchor_tower_matches_differences_of_the_root() {
        let link = InverseLink::Standard(StandardLink::Probit);
        let grid = skewed_grid();
        let tower_at = |q: f64, b: f64| {
            let marginal = bernoulli_marginal_link_map(&link, q).unwrap();
            mixture_anchor_tower(marginal.q, b, &grid).unwrap()
        };
        let (q, b) = (-0.7, 0.6);
        let base = tower_at(q, b);
        let marginal = bernoulli_marginal_link_map(&link, q).unwrap();
        let root = empirical_intercept(marginal.q, b, 1.0, &grid.nodes, &grid.weights).unwrap();
        assert_eq!(base.v, root, "the value channel is the root itself");
        let h = 1.0e-4;
        let shifted = |axis: usize, sign: f64| {
            if axis == 0 {
                tower_at(q + sign * h, b)
            } else {
                tower_at(q, b + sign * h)
            }
        };
        let close = |exact: f64, fd: f64, what: &str| {
            assert!(
                (exact - fd).abs() < 2.0e-6 * (1.0 + fd.abs()),
                "{what}: tower {exact} vs central difference {fd}"
            );
        };
        for a in 0..2 {
            let (plus, minus) = (shifted(a, 1.0), shifted(a, -1.0));
            close(base.g[a], (plus.v - minus.v) / (2.0 * h), &format!("g[{a}]"));
            for i in 0..2 {
                close(base.h[a][i], (plus.g[i] - minus.g[i]) / (2.0 * h), &format!("h[{a}][{i}]"));
                for j in 0..2 {
                    close(
                        base.t3[a][i][j],
                        (plus.h[i][j] - minus.h[i][j]) / (2.0 * h),
                        &format!("t3[{a}][{i}][{j}]"),
                    );
                    for l in 0..2 {
                        close(
                            base.t4[a][i][j][l],
                            (plus.t3[i][j][l] - minus.t3[i][j][l]) / (2.0 * h),
                            &format!("t4[{a}][{i}][{j}][{l}]"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mixture_anchor_is_the_gaussian_closed_form_on_a_hermite_law() {
        let link = InverseLink::Standard(StandardLink::Probit);
        let grid = hermite_grid(80);
        let cov = covariance(2, 31);
        let declared = declare_unit_score_variance(&cov.to_dense()).unwrap();
        let cov = MarginalSlopeCovariance::full(declared).unwrap();
        let (g, beta, z, r, s) = (0.55, [0.35, -0.25], -0.4, [0.9, -1.3], 0.85);
        for eta_m in [-2.5, -0.3, 1.1] {
            let marginal = bernoulli_marginal_link_map(&link, eta_m).unwrap();
            let gaussian = residual_row_index(&marginal, g, &beta, z, &r, &cov, None, s).unwrap();
            let mixture =
                residual_row_index(&marginal, g, &beta, z, &r, &cov, Some(&grid), s).unwrap();
            assert!((gaussian.0 - mixture.0).abs() < 1.0e-9, "η {} vs {}", gaussian.0, mixture.0);
            assert!((gaussian.1 - mixture.1).abs() < 1.0e-8);
            assert!((gaussian.2 - mixture.2).abs() < 1.0e-8);
            for j in 0..2 {
                assert!((gaussian.3[j] - mixture.3[j]).abs() < 1.0e-8);
            }
        }
    }

    #[test]
    fn mixture_anchor_holds_under_the_declared_joint_law() {
        // Integrate the drive's law independently: node k of the score law,
        // then r | z ~ N(γz, Σ_{r·z}) as a one-dimensional Gaussian drive.
        let link = InverseLink::Standard(StandardLink::Probit);
        let grid = skewed_grid();
        let declared = declare_unit_score_variance(&covariance(2, 47).to_dense()).unwrap();
        let cov = MarginalSlopeCovariance::full(declared.clone()).unwrap();
        let (g, beta, s) = (0.7, [0.45, -0.3], 0.9);
        let gamma = [declared[[1, 0]], declared[[2, 0]]];
        let mut schur = [[0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                schur[i][j] = declared[[i + 1, j + 1]] - gamma[i] * gamma[j];
            }
        }
        let m = s * (g + beta[0] * gamma[0] + beta[1] * gamma[1]);
        let mut v = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                v += s * s * beta[i] * schur[i][j] * beta[j];
            }
        }
        let inner = hermite_grid(64);
        for eta_m in [-1.8, 0.4] {
            let marginal = bernoulli_marginal_link_map(&link, eta_m).unwrap();
            // At z = 0, r = 0 the index is the intercept α itself.
            let (alpha, _, _, _) =
                residual_row_index(&marginal, g, &beta, 0.0, &[0.0, 0.0], &cov, Some(&grid), s)
                    .unwrap();
            let mut anchored_mean = 0.0;
            for (z, w) in grid.pairs() {
                for (u, wu) in inner.pairs() {
                    anchored_mean += w * wu * normal_cdf(alpha + m * z + v.sqrt() * u);
                }
            }
            assert!(
                (anchored_mean - marginal.mu).abs() < 1.0e-10,
                "E[Φ(α + drive)] = {anchored_mean} vs Φ(q) = {}",
                marginal.mu
            );
        }
    }

    /// gam#2985: the joint anchor read through the score alone is the joint
    /// anchor. Its residual under the law the row was anchored on is zero, and
    /// under any law it is the integral over that law of the score and the
    /// residual's conditional Gaussian, computed here directly on a second
    /// quadrature.
    #[test]
    fn joint_anchor_on_the_score_integrates_the_residual_given_the_score() {
        let link = InverseLink::Standard(StandardLink::Probit);
        let declared = declare_unit_score_variance(&covariance(2, 47).to_dense()).unwrap();
        let cov = MarginalSlopeCovariance::full(declared.clone()).unwrap();
        let (g, beta, s) = (0.7, [0.45, -0.3], 0.9);
        let gamma = [declared[[1, 0]], declared[[2, 0]]];
        let m = s * (g + beta[0] * gamma[0] + beta[1] * gamma[1]);
        let mut v = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                v += s * s * beta[i] * (declared[[i + 1, j + 1]] - gamma[i] * gamma[j]) * beta[j];
            }
        }
        let anchor = JointAnchorOnScore::at(g, &beta, &cov, s).unwrap();
        assert!((anchor.scale - (1.0 + v).sqrt()).abs() < 1.0e-14, "τ = {}", anchor.scale);
        assert!((anchor.slope - m / anchor.scale).abs() < 1.0e-14, "B = {}", anchor.slope);
        let skewed = skewed_grid();
        let normal = hermite_grid(64);
        let inner = hermite_grid(64);
        for eta_m in [-1.8, 0.4] {
            let marginal = bernoulli_marginal_link_map(&link, eta_m).unwrap();
            for (anchored_on, anchored_law, scored_on) in [
                (Some(&skewed), &skewed, &skewed),
                (Some(&skewed), &skewed, &normal),
                (None, &normal, &normal),
                (None, &normal, &skewed),
            ] {
                let (alpha, _, _, _) =
                    residual_row_index(&marginal, g, &beta, 0.0, &[0.0, 0.0], &cov, anchored_on, s)
                        .unwrap();
                let (residual, _, mu) = anchor.anchoring_residual(alpha, marginal.mu, scored_on).unwrap();
                let mut direct = 0.0;
                for (z, w) in scored_on.pairs() {
                    for (e, we) in inner.pairs() {
                        direct += w * we * normal_cdf(alpha + m * z + v.sqrt() * e);
                    }
                }
                assert!(
                    (residual - (direct - mu)).abs() < 1.0e-10,
                    "η_m = {eta_m}: residual {residual} against the direct integral {}",
                    direct - mu
                );
                if std::ptr::eq(anchored_law, scored_on) {
                    assert!(
                        residual.abs() < 1.0e-10,
                        "η_m = {eta_m}: the anchor does not hold on its own law: {residual}"
                    );
                }
            }
        }
        // With β = 0 the joint anchor is the score-only anchor: B = s·g, τ = 1.
        let score_only = JointAnchorOnScore::at(g, &[0.0, 0.0], &cov, s).unwrap();
        assert_eq!(score_only.scale, 1.0);
        assert!((score_only.slope - s * g).abs() < 1.0e-15, "B = {}", score_only.slope);
    }

    /// A standalone row program over the kernel's own [`residual_row_nll`],
    /// so the jet lowering can be gated without a family fixture.
    struct StandaloneRow<const K: usize> {
        marginal: BernoulliMarginalLinkMap,
        z: f64,
        y: f64,
        w: f64,
        s: f64,
        r: Vec<f64>,
        cov: MarginalSlopeCovariance,
        grid: Option<EmpiricalZGrid>,
        base: [f64; K],
    }

    impl<const K: usize> gam_math::jet_tower::RowProgram<K> for StandaloneRow<K> {
        fn n_rows(&self) -> usize {
            1
        }
        fn primaries(&self, row: usize) -> Result<[f64; K], String> {
            if row != 0 {
                return Err(format!("standalone row program has one row, got {row}"));
            }
            Ok(self.base)
        }
        fn eval<S: JetScalar<K>>(&self, row: usize, p: &[S; K]) -> Result<S, String> {
            if row != 0 {
                return Err(format!("standalone row program has one row, got {row}"));
            }
            let state = ResidualRowState {
                marginal: self.marginal,
                z: self.z,
                y: self.y,
                w: self.w,
                probit_scale: self.s,
                grid: self.grid.as_ref(),
            };
            // The coefficients are the primaries here: the drive is composed
            // from their jets, exactly as the coefficient-primary kernel did.
            let gamma: Vec<f64> = (0..K - 2).map(|j| self.cov.coefficient(0, j + 1)).collect();
            let beta = &p[2..];
            let drive = ResidualDrive {
                t: S::linear_combination(beta, &self.r),
                u: S::linear_combination(beta, &gamma),
                v: S::symmetric_quadratic_form(beta, &ResidualBlock(&self.cov)),
            };
            residual_row_nll(&state, &p[0], &p[1], &drive)
        }
    }

    fn gate_jet_against_fd<const K: usize>(row: &StandaloneRow<K>) {
        let (nll, grad, hess) = gam_math::jet_tower::program_row_kernel(row, 0).unwrap();
        // The value at a displaced point: the marginal link stack is expanded
        // about the displaced η, exactly as the kernel rebuilds it from the
        // block state at every evaluation.
        let link = InverseLink::Standard(StandardLink::Probit);
        let value_at = |p: [f64; K]| -> f64 {
            let displaced = StandaloneRow::<K> {
                marginal: bernoulli_marginal_link_map(&link, p[0]).unwrap(),
                z: row.z,
                y: row.y,
                w: row.w,
                s: row.s,
                r: row.r.clone(),
                cov: row.cov.clone(),
                grid: row.grid.clone(),
                base: p,
            };
            let jets: [Order2<K>; K] =
                std::array::from_fn(|a| <Order2<K> as JetScalar<K>>::constant(p[a]));
            displaced.eval(0, &jets).unwrap().value()
        };
        assert!((value_at(row.base) - nll).abs() < 1.0e-12);
        let h = 1.0e-5;
        for a in 0..K {
            let mut plus = row.base;
            let mut minus = row.base;
            plus[a] += h;
            minus[a] -= h;
            let fd = (value_at(plus) - value_at(minus)) / (2.0 * h);
            assert!(
                (fd - grad[a]).abs() < 2.0e-7 * (1.0 + fd.abs()),
                "K={K}: grad[{a}] {} vs fd {fd}",
                grad[a]
            );
            for b in 0..K {
                let mut pp = row.base;
                let mut pm = row.base;
                let mut mp = row.base;
                let mut mm = row.base;
                pp[a] += h;
                pp[b] += h;
                pm[a] += h;
                pm[b] -= h;
                mp[a] -= h;
                mp[b] += h;
                mm[a] -= h;
                mm[b] -= h;
                let fd2 = (value_at(pp) - value_at(pm) - value_at(mp) + value_at(mm)) / (4.0 * h * h);
                assert!(
                    (fd2 - hess[a][b]).abs() < 5.0e-5 * (1.0 + fd2.abs()),
                    "K={K}: hess[{a}][{b}] {} vs fd {fd2}",
                    hess[a][b]
                );
            }
        }
        // The plain-f64 index derivatives and the jet gradient agree through
        // the chain rule dℓ/dp = ℓ'(margin)·sign·∂η/∂p.
        let beta: Vec<f64> = row.base[2..].to_vec();
        let (eta, d_q, d_g, d_beta) = residual_row_index(
            &row.marginal,
            row.base[1],
            &beta,
            row.z,
            &row.r,
            &row.cov,
            row.grid.as_ref(),
            row.s,
        )
        .unwrap();
        let sign = 2.0 * row.y - 1.0;
        let stack = signed_probit_neglog_unary_stack(sign * eta, row.w);
        let outer = stack[1] * sign;
        assert!((grad[0] - outer * d_q).abs() < 1.0e-10);
        assert!((grad[1] - outer * d_g).abs() < 1.0e-10);
        for j in 0..K - 2 {
            assert!((grad[2 + j] - outer * d_beta[j]).abs() < 1.0e-10);
        }
    }

    #[test]
    fn jet_gradient_and_hessian_match_central_differences() {
        let link = InverseLink::Standard(StandardLink::Probit);
        for grid in [None, Some(skewed_grid())] {
            let row3 = StandaloneRow::<3> {
                marginal: bernoulli_marginal_link_map(&link, 0.4).unwrap(),
                z: -0.6,
                y: 1.0,
                w: 1.3,
                s: 0.9,
                r: vec![0.7],
                cov: covariance(1, 5),
                grid: grid.clone(),
                base: [0.4, 0.3, -0.5],
            };
            gate_jet_against_fd(&row3);
            let row5 = StandaloneRow::<5> {
                marginal: bernoulli_marginal_link_map(&link, -1.1).unwrap(),
                z: 1.4,
                y: 0.0,
                w: 0.8,
                s: 1.0,
                r: vec![0.2, -1.1, 0.6],
                cov: MarginalSlopeCovariance::full(
                    declare_unit_score_variance(&covariance(3, 7).to_dense()).unwrap(),
                )
                .unwrap(),
                grid,
                base: [-1.1, -0.35, 0.25, 0.4, -0.15],
            };
            gate_jet_against_fd(&row5);
        }
    }

    #[test]
    fn jet_third_and_fourth_contractions_match_differences_of_the_hessian() {
        // The outer REML/LAML derivatives read the order-3 and order-4
        // contractions; on the declared finite law they carry the mixture
        // anchor tower through its fourth order.
        let link = InverseLink::Standard(StandardLink::Probit);
        for grid in [None, Some(skewed_grid())] {
            let base = [-0.6, 0.45, 0.3, -0.2];
            let row = |p: [f64; 4]| StandaloneRow::<4> {
                marginal: bernoulli_marginal_link_map(&link, p[0]).unwrap(),
                z: 0.8,
                y: 1.0,
                w: 1.1,
                s: 0.95,
                r: vec![-0.4, 1.2],
                cov: MarginalSlopeCovariance::full(
                    declare_unit_score_variance(&covariance(2, 13).to_dense()).unwrap(),
                )
                .unwrap(),
                grid: grid.clone(),
                base: p,
            };
            let direction = [0.3, -0.7, 0.5, 0.9];
            let second = [-0.2, 0.4, 0.8, -0.6];
            let t3 = gam_math::jet_tower::program_third_contracted(&row(base), 0, &direction).unwrap();
            let t4 =
                gam_math::jet_tower::program_fourth_contracted(&row(base), 0, &direction, &second)
                    .unwrap();
            let h = 1.0e-4;
            let displaced = |scale: f64, dir: &[f64; 4], at: [f64; 4]| {
                std::array::from_fn::<f64, 4, _>(|a| at[a] + scale * dir[a])
            };
            let hessian = |p: [f64; 4]| gam_math::jet_tower::program_row_kernel(&row(p), 0).unwrap().2;
            let t3_at = |p: [f64; 4]| {
                gam_math::jet_tower::program_third_contracted(&row(p), 0, &direction).unwrap()
            };
            let (hp, hm) = (hessian(displaced(h, &direction, base)), hessian(displaced(-h, &direction, base)));
            let (tp, tm) = (t3_at(displaced(h, &second, base)), t3_at(displaced(-h, &second, base)));
            for a in 0..4 {
                for b in 0..4 {
                    let fd3 = (hp[a][b] - hm[a][b]) / (2.0 * h);
                    assert!(
                        (t3[a][b] - fd3).abs() < 1.0e-5 * (1.0 + fd3.abs()),
                        "law={}: t3[{a}][{b}] {} vs fd {fd3}",
                        grid.is_some(),
                        t3[a][b]
                    );
                    let fd4 = (tp[a][b] - tm[a][b]) / (2.0 * h);
                    assert!(
                        (t4[a][b] - fd4).abs() < 1.0e-5 * (1.0 + fd4.abs()),
                        "law={}: t4[{a}][{b}] {} vs fd {fd4}",
                        grid.is_some(),
                        t4[a][b]
                    );
                }
            }
        }
    }

    #[test]
    fn anchor_reduces_to_frailty_identity_when_residuals_are_independent_of_the_score() {
        // Σ = blockdiag(1, Σ_rr): c = √(1 + s²(g² + βᵀΣ_rrβ)), the gam#2766
        // identity with the residual variance entering like a frailty.
        let link = InverseLink::Standard(StandardLink::Probit);
        let sigma_rr = [[0.5, 0.1], [0.1, 0.3]];
        let mut sigma = Array2::<f64>::eye(3);
        for i in 0..2 {
            for j in 0..2 {
                sigma[[1 + i, 1 + j]] = sigma_rr[i][j];
            }
        }
        let cov = MarginalSlopeCovariance::full(sigma).unwrap();
        let (g, beta, s, z, r) = (0.45, [0.3, -0.2], 0.85, 0.2, [1.0, -0.5]);
        let marginal = bernoulli_marginal_link_map(&link, 0.7).unwrap();
        let (eta, _, _, _) = residual_row_index(&marginal, g, &beta, z, &r, &cov, None, s).unwrap();
        let kappa =beta[0] * beta[0] * sigma_rr[0][0]
            + 2.0 * beta[0] * beta[1] * sigma_rr[0][1]
            + beta[1] * beta[1] * sigma_rr[1][1];
        let c = (1.0 + s * s * (g * g + kappa)).sqrt();
        let expected = 0.7 * c + s * (g * z + beta[0] * r[0] + beta[1] * r[1]);
        assert!((eta - expected).abs() < 1.0e-13, "{eta} vs {expected}");
    }

    #[test]
    fn centring_gate_refuses_a_shifted_column_and_admits_a_centred_one() {
        let n = 4000usize;
        let mut state = 0x9e3779b97f4a7c15_u64;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut normal = || {
            let u1 = uniform().max(1.0e-12);
            let u2 = uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let a: Vec<f64> = (0..n).map(|_| normal()).collect();
        let z: Vec<f64> = (0..n).map(|_| normal()).collect();
        let centred: Vec<f64> = (0..n).map(|_| normal()).collect();
        let shifted: Vec<f64> = (0..n).map(|i| 0.5 * a[i] + 0.3 * normal()).collect();
        let a_block = Array2::from_shape_vec((n, 1), a.clone()).unwrap();
        let weights = Array1::from_elem(n, 1.0);
        let z = Array1::from(z);
        let ok = ResidualRepairSpec {
            columns: vec!["r".into()],
            features: Array2::from_shape_vec((n, 1), centred).unwrap(),
        };
        let runtime = ResidualBlockRuntime::fit(&ok, z.view(), weights.view(), a_block.view())
            .expect("a centred column is admitted");
        assert_eq!(runtime.width(), 1);
        assert_eq!(runtime.field.dim(), 2);
        let bad = ResidualRepairSpec {
            columns: vec!["r".into()],
            features: Array2::from_shape_vec((n, 1), shifted).unwrap(),
        };
        let err = ResidualBlockRuntime::fit(&bad, z.view(), weights.view(), a_block.view())
            .expect_err("a column with conditional mean structure is refused");
        assert!(
            matches!(err, ResidualRepairRefusal::ColumnNotCentred { .. }),
            "{err}"
        );
    }

    #[test]
    fn declared_score_variance_keeps_the_law_of_r_given_z_and_the_rigid_anchor() {
        let sample = covariance(3, 23).to_dense().mapv(|v| 1.7 * v);
        let declared = declare_unit_score_variance(&sample).unwrap();
        assert_eq!(declared[[0, 0]], 1.0);
        let schur = |m: &Array2<f64>| {
            Array2::from_shape_fn((3, 3), |(i, j)| {
                m[[i + 1, j + 1]] - m[[i + 1, 0]] * m[[0, j + 1]] / m[[0, 0]]
            })
        };
        let (s_sample, s_declared) = (schur(&sample), schur(&declared));
        for i in 0..3 {
            let gamma_sample = sample[[i + 1, 0]] / sample[[0, 0]];
            assert!((declared[[i + 1, 0]] - gamma_sample).abs() < 1.0e-14);
            for j in 0..3 {
                assert!((s_sample[[i, j]] - s_declared[[i, j]]).abs() < 1.0e-13);
            }
        }
        // β = 0: the anchor is the score-only √(1 + s²g²), whatever Var̂(z) was.
        let link = InverseLink::Standard(StandardLink::Probit);
        let marginal = bernoulli_marginal_link_map(&link, -0.4).unwrap();
        let cov = MarginalSlopeCovariance::full(declared).unwrap();
        let (g, s, z) = (0.8, 0.9, 1.3);
        let (eta, _, _, _) =
            residual_row_index(&marginal, g, &[0.0; 3], z, &[0.5, -0.2, 1.1], &cov, None, s)
                .unwrap();
        let rigid = marginal.q * (1.0 + s * s * g * g).sqrt() + s * g * z;
        assert!((eta - rigid).abs() < 1.0e-14, "{eta} vs {rigid}");
    }
}
