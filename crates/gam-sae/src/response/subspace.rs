//! The best retained response of a known MLP block, and the exact error of discarding the rest of its input
//! (#2946 R1, R2, R4, R6).
//!
//! # Entry points
//!
//! - [`KnownBlock::new`]`(readers, biases, writers, output_bias, metric, activation)` builds the block and caches
//!   `V(I)`.
//! - [`KnownBlock::retained_response`]`(frame, points)` is R1 at ambient points, output bias included.
//! - [`KnownBlock::explained_variance`], [`KnownBlock::discarded_error`] and
//!   [`KnownBlock::explained_variance_of_coordinates`] are R4 on a frame or on a coordinate set.
//! - [`KnownBlock::explained_variance_gradient`] is R6.
//! - [`KnownGatedBlock::new`] and [`KnownGatedBlock::retained_response`] are R9's gated block and its best retained
//!   response, with the SiLU quadrature band of every entry.
//! - [`covariance_rounding_band`] and [`frame_defect`] own the Cauchy–Schwarz rounding band a caller states to the
//!   pair kernels.
//!
//! # Block, law and frame
//!
//! The block is `F(z) = Σ_j u_j σ(b_j + w_jᵀ z) + c` with readers `W ∈ ℝ^{h×d}` (rows `w_jᵀ`), biases `b ∈ ℝ^h`,
//! writers `U ∈ ℝ^{p×h}` (columns `u_j`), output bias `c ∈ ℝ^p` and activation `σ`. The input law is `Z ~ N(0, I_d)`,
//! the output metric is `M ≻ 0`, and a frame `Q ∈ ℝ^{d×k}` with orthonormal columns spans the retained input
//! subspace, `P = Q Qᵀ`. The output bias moves every response and no variance.
//!
//! # R1, the best retained response
//!
//! `PZ` and `(I − P)Z` are independent, so the conditional mean given the retained input is
//!
//! ```text
//! F̄_P(Pz) = E[F(Z) | PZ = Pz] = Σ_j u_j T_{v_j⊥} σ(b_j + w_jᵀ P z) + c,   v_j⊥ = w_jᵀ (I − P) w_j,
//! ```
//!
//! with `T_v σ(t) = E σ(t + √v E)` the Gaussian smoothing of the activation.
//!
//! # R2, the error split
//!
//! For every `g`, `E‖F − g(PZ)‖²_M = E‖F − F̄_P‖²_M + E‖F̄_P − g(PZ)‖²_M`, because `F − F̄_P` is orthogonal to every
//! function of `PZ`. The discarded-input error and the fit error of a compact response are separate numbers.
//!
//! # R4, explained variance and discarded error
//!
//! Couple `Z' = PZ + (I − P)Z̃` with `Z̃` an independent copy of `Z`. Then `F̄_P(PZ) = E[F(Z') | Z]`, so
//! `E⟨F̄_P − c, F̄_P − c⟩_M = E⟨F(Z) − c, F(Z') − c⟩_M`, and `(b_j + w_jᵀZ, b_k + w_kᵀZ')` is jointly normal with
//! variances `v_j = ‖w_j‖²`, `v_k` and covariance `w_jᵀ P w_k`. Hence
//!
//! ```text
//! V(P) = E‖F̄_P − E F‖²_M = Σ_jk D_jk [K_σ(b_j, b_k; v_j, v_k, w_jᵀ P w_k) − m_j m_k],
//! ```
//!
//! with `D_jk = u_jᵀ M u_k`, `K_σ = E σ(X) σ(Y)` and `m_j = T_{v_j} σ(b_j)`. The cross terms `j ≠ k` are required.
//! R2 with `g = E F` gives `E(P) = E‖F − F̄_P‖²_M = V(I) − V(P) ≥ 0`. At `P = 0` the pair is independent and every
//! bracket vanishes, so `V(0) = 0` and `E(0) = V(I)`; `E(I) = 0`.
//!
//! # R6, the frame gradient
//!
//! Price's theorem gives `∂_r K_σ = E σ'(X) σ'(Y)`. With `B_jk = D_jk ∂_r K_σ(b_j, b_k; v_j, v_k, w_jᵀ P w_k)`, which
//! is symmetric, `dV = Σ_jk B_jk w_jᵀ (dQ Qᵀ + Q dQᵀ) w_k = 2 tr(dQᵀ Wᵀ B W Q)`. `V` depends on `Q` only through `P`,
//! so the Grassmann (horizontal) gradient is
//!
//! ```text
//! ∇_Q V = 2 (I − Q Qᵀ) Wᵀ B(P) W Q.
//! ```
//!
//! # Tiles
//!
//! The frame enters only through `R = W Q` (`h × k`) and the metric only through `D = Uᵀ M U`. `D` is read over
//! `j ≤ k` from the block's [`ReaderGram`], or, when the process-wide ledger declines that cache, streamed per tile
//! by [`fill_upper_rows`] over the same tiles, so both routes read the same words. `D` and the pair law are both
//! symmetric, so a pass evaluates only the pairs `k ≥ j` and counts each strictly upper term twice. A tile `J` of `t`
//! units starting at `s` forms `R_J R_{k≥s}ᵀ` and its weights `B_J` over `k ≥ s` (each `t × (h − s)`). It adds
//! `B_J R_{k≥s}` to its own rows of `B R`, and the strictly upper part transposed, `B_Jᵀ R_J`, to the rows `k ≥ s`.
//! No `d × d` matrix is formed, and the only resident `h × h` object is the packed half of `D`, charged to the
//! memory governor. The gradient is then `Wᵀ(B R) − Q (Qᵀ Wᵀ (B R))`.
//!
//! # R9, gated units
//!
//! A SwiGLU unit is `u_j (a_jᵀ z + c_j) s(w_jᵀ z + b_j)` with the SiLU gate `s(t) = t σ(t)`. Given `PZ = Pz`, the
//! discarded parts `A = a_jᵀ(I − P)Z` and `W = w_jᵀ(I − P)Z` are a zero-mean Gaussian pair independent of `PZ`, with
//! `Var W = v_j⊥ = w_jᵀ(I − P)w_j` and `Cov(A, W) = κ_j⊥ = a_jᵀ(I − P)w_j`, and Stein's lemma `E[A h(W)] = κ E h'(W)`
//! gives the unit's conditional mean exactly as `α_j T_{v⊥} s(t_j) + κ_j⊥ T_{v⊥} s'(t_j)`, with `α_j = a_jᵀPz + c_j`
//! and `t_j = w_jᵀPz + b_j`. SiLU has no closed-form Gaussian smoothing, so each value carries its derived quadrature
//! bound from `gaussian_gated`.
//!
//! # Rounding of the pair law
//!
//! The exact law of a pair has `|w_jᵀ P w_k| ≤ √(v_j v_k)`. The computed triple `(v̂_j, v̂_k, r̂_jk)` can leave that
//! cone only by the rounding of its formation and by a frame's orthonormality defect, and the kernel projects a
//! covariance within the stated band [`covariance_rounding_band`] onto the boundary and refuses one beyond it. A frame
//! whose measured defect reaches 1 is not a frame and is refused ([`ResponseError::FrameNotOrthonormal`]).
//!
//! # The operator's band
//!
//! `V(I)`, `V(P)` and `E(P)` are [`BandedEnergy`]. Each pair term carries a first-order bound on its distance from the
//! exact term at the exact law, summing:
//! - the kernel's own rounding;
//! - the covariance's formation error and the projection's move, through `sup|σ'|²`;
//! - the variances' rounding, through the kernel's variance partials;
//! - the means' and the metric products' errors;
//! - the term's own roundings.
//!
//! The pass adds `γ_{2h+1}` of its absolute terms, and `E = V(I) − V(P)` is their [`signed_sum`].

use super::hermite::{HermiteError, unit_tail_envelope};
use super::reader_gram::{ReaderGram, ReaderGramError, fill_upper_rows, upper_tile_rows};
use super::tiles::chaos_operations;
use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, fast_ab, fast_abt, fast_atb};
use gam_linalg::roundoff::accumulation_growth;
use gam_math::gaussian_activation::{
    GaussianActivation, GaussianActivationError, PairKernel, PreactivationPair, gaussian_hermite_coefficients,
    gaussian_smoothing_derivatives, pair_kernel, pair_kernel_variance_partials, project_covariance,
};
use gam_math::gaussian_gated::{GatedMean, GaussianGatedError, gated_conditional_mean};
use gam_math::roundoff::inflated;
use gam_runtime::resource::byte_balanced_row_chunk;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rayon::prelude::*;
use std::fmt;
use std::sync::Arc;

/// A refusal of the retained-response operator.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseError {
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        got: usize,
    },
    NonFinite {
        context: &'static str,
    },
    /// The metric must be exactly symmetric: `V` reads only its symmetric part, but the gradient formula `2 Wᵀ B W Q`
    /// holds only when `B`, and so `D = Uᵀ M U`, is symmetric.
    MetricNotSymmetric {
        row: usize,
        column: usize,
    },
    MetricNotPositiveDefinite {
        reason: String,
    },
    /// A frame of rank `k` in `ℝ^d` needs `k ≤ d` orthonormal columns.
    FrameWiderThanInput {
        rank: usize,
        input_dim: usize,
    },
    /// The frame's measured orthonormality defect [`frame_defect`] is at least 1, so it has no nearest orthonormal
    /// frame whose projector the operator could be exact for.
    FrameNotOrthonormal {
        defect: f64,
    },
    /// Retained coordinates must be strictly increasing and below the input dimension, so they name a projector.
    InvalidRetainedCoordinates {
        position: usize,
    },
    Kernel {
        context: &'static str,
        error: GaussianActivationError,
    },
    /// The reader Gram refused for a reason other than the ledger declining its footprint, which the block answers
    /// by streaming `D`.
    ReaderGram {
        error: ReaderGramError,
    },
    GatedKernel {
        context: &'static str,
        error: GaussianGatedError,
    },
}

impl fmt::Display for ResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                context,
                expected,
                got,
            } => write!(f, "{context}: expected {expected}, got {got}"),
            Self::NonFinite { context } => write!(f, "{context} holds a non-finite entry"),
            Self::MetricNotSymmetric { row, column } => write!(
                f,
                "output metric is not symmetric: entry ({row}, {column}) differs from its transpose"
            ),
            Self::MetricNotPositiveDefinite { reason } => {
                write!(f, "output metric is not positive definite: {reason}")
            }
            Self::FrameWiderThanInput { rank, input_dim } => write!(
                f,
                "a retained frame of rank {rank} cannot be orthonormal in input dimension {input_dim}"
            ),
            Self::FrameNotOrthonormal { defect } => write!(
                f,
                "retained frame is not orthonormal: its measured defect {defect} is at least 1"
            ),
            Self::InvalidRetainedCoordinates { position } => write!(
                f,
                "retained coordinate at position {position} is out of range or not strictly increasing"
            ),
            Self::Kernel { context, error } => write!(f, "{context}: {error}"),
            Self::ReaderGram { error } => write!(f, "reader Gram: {error}"),
            Self::GatedKernel { context, error } => write!(f, "{context}: {error}"),
        }
    }
}

impl std::error::Error for ResponseError {}

/// `V(P)`, `E(P)` and the horizontal frame gradient of `V` at one frame.
#[derive(Debug, Clone)]
pub struct FrameGradient {
    /// `V(P)`, the output variance the retained response explains, with its band.
    pub explained_variance: BandedEnergy,
    /// `E(P) = V(I) − V(P)`, the error of discarding the input outside the frame, with its band.
    pub discarded_error: BandedEnergy,
    /// `∇_Q V = 2 (I − Q Qᵀ) Wᵀ B(P) W Q`, a `d × k` horizontal tangent: the ascent direction of `V` and the descent
    /// direction of `E`.
    pub horizontal_gradient: Array2<f64>,
}

/// An energy together with a bound on its absolute error: the vocabulary every producer of a variance in `response/`
/// reports in, so a consumer decides "resolved from zero" against a derived band instead of a hand threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandedEnergy {
    /// The computed value.
    pub value: f64,
    /// A bound on `|computed − exact|`: the arithmetic's rounding plus the producer's accuracy contract.
    pub band: f64,
}

impl BandedEnergy {
    /// The exact zero, `V(∅)`.
    pub const ZERO: BandedEnergy = BandedEnergy {
        value: 0.0,
        band: 0.0,
    };

    /// Whether the exact value is resolved as positive: the computed value clears its band.
    pub fn resolved_positive(&self) -> bool {
        self.value > self.band
    }
}

/// `Σ added − Σ subtracted` with its band: the operands' bands plus `γ_{n−1} Σ|operand|`, the rounding of `n − 1`
/// additions of pre-formed terms (Higham, ASNA §3.1), to first order in `u`.
pub fn signed_sum(added: &[BandedEnergy], subtracted: &[BandedEnergy]) -> BandedEnergy {
    let mut value = 0.0;
    let mut absolute = 0.0;
    let mut band = 0.0;
    for term in added {
        value += term.value;
        absolute += term.value.abs();
        band += term.band;
    }
    for term in subtracted {
        value -= term.value;
        absolute += term.value.abs();
        band += term.band;
    }
    let operations = (added.len() + subtracted.len()).saturating_sub(1);
    BandedEnergy {
        value,
        band: band + accumulation_growth(operations) * absolute,
    }
}

/// How a computed covariance `r̂ = fl(Σ_a left_a right_a)` was formed, for [`covariance_rounding_band`]. Every field is
/// an absolute bound in the covariance's own units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CovarianceFormation {
    /// The number of products in `r̂`.
    pub terms: usize,
    /// `‖left‖` and `‖right‖` of the computed rows.
    pub left_norm: f64,
    pub right_norm: f64,
    /// Bounds on the Euclidean distance from each computed row to the row formed exactly from the same inputs.
    pub left_row_error: f64,
    pub right_row_error: f64,
    /// A bound on `|left_exact · right_exact − r|`, with `r` the law's covariance: `0` when the exact rows are the law's
    /// own rows, and a frame's projector gap otherwise.
    pub law_gap: f64,
    /// Bounds on `|v̂ − v|` of the two computed variances handed to the kernel.
    pub variance_error_x: f64,
    pub variance_error_y: f64,
}

/// The `covariance_rounding` of a computed pair law: a bound on how far `|r̂|` can exceed `√(v̂_x v̂_y)` when the exact
/// law satisfies `|r| ≤ √(v_x v_y)`.
///
/// `|r̂| ≤ |r| + |r̂ − r|` and `√(v_x v_y) ≤ √((v̂_x + e_x)(v̂_y + e_y))`, while
/// `|r̂ − r| ≤ γ_terms ‖left‖‖right‖ + ‖δ_left‖‖right‖ + ‖left‖‖δ_right‖ + ‖δ_left‖‖δ_right‖ + law_gap`: the
/// recursive rounding of the product sum (Higham, ASNA §3.1), the propagated row formation errors `δ`, and the gap
/// between the exactly formed rows and the law. The band is their sum.
pub fn covariance_rounding_band(formation: &CovarianceFormation, variance_x: f64, variance_y: f64) -> f64 {
    let variance_rounding = ((variance_x + formation.variance_error_x)
        * (variance_y + formation.variance_error_y))
        .sqrt()
        - (variance_x * variance_y).sqrt();
    covariance_formation_error(formation) + variance_rounding
}

/// A bound on `|r̂ − r|` for a computed covariance: `γ_terms ‖left‖‖right‖ + ‖δ_left‖‖right‖ + ‖left‖‖δ_right‖ +
/// ‖δ_left‖‖δ_right‖ + law_gap`, the part of [`covariance_rounding_band`] that moves the covariance itself.
pub fn covariance_formation_error(formation: &CovarianceFormation) -> f64 {
    let product_rounding = accumulation_growth(formation.terms) * formation.left_norm * formation.right_norm;
    let row_rounding = formation.left_row_error * formation.right_norm
        + formation.left_norm * formation.right_row_error
        + formation.left_row_error * formation.right_row_error;
    product_rounding + row_rounding + formation.law_gap
}

/// A measured bound `η ≥ ‖QᵀQ − I‖₂` on the orthonormality defect of a `d × k` frame, or infinity when no bound is
/// available.
///
/// The computed Gram `Ĝ = fl(QᵀQ)` errs entrywise by at most `γ_d ‖q_a‖‖q_b‖ ≤ γ_d (1 + η)`, so
/// `‖QᵀQ − Ĝ‖_F ≤ k γ_d (1 + η)`. The Frobenius norm of `Ĝ − I` rounds by at most `γ_{k²+1}` relative, so
/// `η ≤ n̂ (1 + γ_{k²+1}) + k γ_d (1 + η)`, which solves to the returned bound.
pub fn frame_defect(frame: ArrayView2<'_, f64>) -> f64 {
    let (input_dim, rank) = frame.dim();
    let formation = rank as f64 * accumulation_growth(input_dim);
    if !(formation < 1.0) {
        return f64::INFINITY;
    }
    let gram = fast_atb(&frame, &frame);
    let mut squares = 0.0;
    for row in 0..rank {
        for column in 0..rank {
            let entry = gram[[row, column]] - if row == column { 1.0 } else { 0.0 };
            squares += entry * entry;
        }
    }
    let measured = squares.sqrt() * (1.0 + accumulation_growth(rank * rank + 1));
    (measured + formation) / (1.0 - formation)
}

/// How a pair pass's reader coordinates were formed.
#[derive(Debug, Clone, Copy)]
enum CoordinateFormation {
    /// The coordinates are reader entries themselves: the readers, or selected reader columns.
    Copied,
    /// `R = fl(W Q)` for a frame with measured defect `frame_defect`.
    FrameProducts { frame_defect: f64 },
}

/// A known MLP block under the declared law `Z ~ N(0, I_d)`, with its output metric.
#[derive(Debug, Clone)]
pub struct KnownBlock {
    units: BlockUnits,
    total_variance: BandedEnergy,
}

/// The per-unit data every pair term reads.
#[derive(Debug, Clone)]
struct BlockUnits {
    /// `W`, `h × d`.
    readers: Array2<f64>,
    /// `b`, length `h`.
    biases: Array1<f64>,
    /// `U`, `p × h`.
    writers: Array2<f64>,
    /// `c`, length `p`.
    output_bias: Array1<f64>,
    /// `M`, `p × p`.
    metric: Array2<f64>,
    /// `M U`, `p × h`, so a tile's `D_J = U_Jᵀ (M U)`.
    metric_writers: Array2<f64>,
    /// `D = Uᵀ M U` over `j ≤ k` when the process-wide ledger admitted it; `None` streams the same rows per tile.
    /// Clones of the block share one charged cache.
    reader_gram: Option<Arc<ReaderGram>>,
    activation: GaussianActivation,
    /// `v̂_j = fl(‖w_j‖²)`.
    reader_variances: Array1<f64>,
    /// `|v̂_j − v_j| ≤ γ_d v_j ≤ γ_d v̂_j / (1 − γ_d)`: the rounding of a `d`-term sum of squares.
    reader_variance_errors: Array1<f64>,
    /// `m_j = T_{v_j} σ(b_j) = E σ(b_j + w_jᵀ Z)`.
    unit_means: Array1<f64>,
    /// A first-order bound on `|m̂_j − m_j|` ([`unit_mean`]).
    unit_mean_bands: Array1<f64>,
    /// `½ T_{v̂_j} σ''(b_j) = ∂m_j/∂v_j`, and `0` for a zero reader.
    half_curvatures: Array1<f64>,
    /// `‖u_j‖`.
    writer_norms: Array1<f64>,
    /// `γ_{2p} ‖|M|‖_∞`, so `|D̂_jk − D_jk| ≤ metric_error_scale ‖u_j‖ ‖u_k‖` ([`KnownBlock::new`]).
    metric_error_scale: f64,
    /// `sup|σ'|²`, the covariance Lipschitz constant of every pair kernel.
    slope_bound: f64,
}

impl KnownBlock {
    /// Build the block from readers `W` (`h × d`), biases `b` (`h`), writers `U` (`p × h`), the output bias `c`
    /// (`p`; zeros for a layer without one), a symmetric positive definite output metric `M` (`p × p`) and the
    /// activation. The total variance `V(I) = E‖F − E F‖²_M` is computed once here, since every `E(P)` reads it.
    pub fn new(
        readers: Array2<f64>,
        biases: Array1<f64>,
        writers: Array2<f64>,
        output_bias: Array1<f64>,
        metric: ArrayView2<'_, f64>,
        activation: GaussianActivation,
    ) -> Result<Self, ResponseError> {
        let (width, input_dim) = readers.dim();
        require_length("block biases", width, biases.len())?;
        require_length("block writer columns", width, writers.ncols())?;
        let output_dim = writers.nrows();
        require_length("block output bias", output_dim, output_bias.len())?;
        require_finite("block readers", readers.iter())?;
        require_finite("block biases", biases.iter())?;
        require_finite("block writers", writers.iter())?;
        require_finite("block output bias", output_bias.iter())?;
        require_metric(output_dim, metric)?;
        let metric_writers = fast_ab(&metric, &writers);
        // `M U` of finite factors can still overflow. Both routes to `D` read it, so it is refused here, once.
        require_finite("output metric times writers", metric_writers.iter())?;
        let reader_gram = match ReaderGram::new(writers.view(), metric_writers.view()) {
            Ok(gram) => Some(Arc::new(gram)),
            // A footprint the ledger declines, or one with no representable byte count, streams the same rows instead.
            Err(ReaderGramError::Admission { .. } | ReaderGramError::SizeOverflow { .. }) => None,
            Err(error) => return Err(ResponseError::ReaderGram { error }),
        };
        let reader_variances: Array1<f64> = readers.rows().into_iter().map(|row| row.dot(&row)).collect();
        let variance_growth = accumulation_growth(input_dim);
        let reader_variance_errors = reader_variances.mapv(|variance| variance_growth * variance / (1.0 - variance_growth));
        let mut unit_means = Array1::<f64>::zeros(width);
        let mut unit_mean_bands = Array1::<f64>::zeros(width);
        let mut half_curvatures = Array1::<f64>::zeros(width);
        for unit in 0..width {
            let mean = unit_mean(
                activation,
                biases[unit],
                reader_variances[unit],
                reader_variance_errors[unit],
            )?;
            unit_means[unit] = mean.value;
            unit_mean_bands[unit] = mean.band;
            half_curvatures[unit] = mean.half_curvature;
        }
        let writer_norms: Array1<f64> = writers
            .columns()
            .into_iter()
            .map(|column| column.dot(&column).sqrt())
            .collect();
        // `D̂ = fl(Uᵀ fl(M U))` errs by at most `2γ_p |u_j|ᵀ |M| |u_k|` to first order, and
        // `|u_j|ᵀ |M| |u_k| ≤ ‖|M|‖₂ ‖u_j‖ ‖u_k‖ ≤ ‖|M|‖_∞ ‖u_j‖ ‖u_k‖` for the symmetric nonnegative `|M|`.
        let metric_row_sum = metric
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|entry| entry.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let metric_error_scale = accumulation_growth(2 * output_dim) * metric_row_sum;
        let slope_bound = activation
            .slope_bound_squared()
            .map_err(|error| ResponseError::Kernel {
                context: "pair kernel slope bound",
                error,
            })?;
        let units = BlockUnits {
            readers,
            biases,
            writers,
            output_bias,
            metric: metric.to_owned(),
            metric_writers,
            reader_gram,
            activation,
            reader_variances,
            reader_variance_errors,
            unit_means,
            unit_mean_bands,
            half_curvatures,
            writer_norms,
            metric_error_scale,
            slope_bound,
        };
        let total_variance = units.pair_pass(units.readers.view(), CoordinateFormation::Copied, None)?;
        Ok(Self {
            units,
            total_variance,
        })
    }

    /// Input dimension `d`.
    pub fn input_dim(&self) -> usize {
        self.units.readers.ncols()
    }

    /// Unit count `h`.
    pub fn width(&self) -> usize {
        self.units.readers.nrows()
    }

    /// Output dimension `p`.
    pub fn output_dim(&self) -> usize {
        self.units.writers.nrows()
    }

    pub fn activation(&self) -> GaussianActivation {
        self.units.activation
    }

    /// Readers `W`, `h × d`.
    pub fn readers(&self) -> ArrayView2<'_, f64> {
        self.units.readers.view()
    }

    /// Biases `b`, length `h`.
    pub fn biases(&self) -> ArrayView1<'_, f64> {
        self.units.biases.view()
    }

    /// Writers `U`, `p × h`.
    pub fn writers(&self) -> ArrayView2<'_, f64> {
        self.units.writers.view()
    }

    /// The output bias `c`, length `p`.
    pub fn output_bias(&self) -> ArrayView1<'_, f64> {
        self.units.output_bias.view()
    }

    /// The output metric `M`, `p × p`.
    pub fn metric(&self) -> ArrayView2<'_, f64> {
        self.units.metric.view()
    }

    /// `M U`, `p × h`.
    pub fn metric_writers(&self) -> ArrayView2<'_, f64> {
        self.units.metric_writers.view()
    }

    /// `V(I) = E‖F − E F‖²_M`, the output variance of the whole block, with its band.
    pub fn total_variance(&self) -> BandedEnergy {
        self.total_variance
    }

    /// R1: the best retained response `F̄_P(Pz)`, output bias included, at each row `z` of `points` (`n × d`), as an
    /// `n × p` matrix comparable with the executed block's outputs.
    pub fn retained_response(
        &self,
        frame: ArrayView2<'_, f64>,
        points: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, ResponseError> {
        require_frame(self.input_dim(), frame)?;
        require_length("retained-response point columns", self.input_dim(), points.ncols())?;
        require_finite("retained-response points", points.iter())?;
        let units = &self.units;
        let coordinates = fast_ab(&units.readers, &frame);
        let discarded_variances = self.discarded_reader_variances(frame, coordinates.view());
        let rows = points.nrows();
        let mut response = Array2::<f64>::zeros((rows, self.output_dim()));
        let chunk = byte_balanced_row_chunk(self.width() + self.input_dim(), rows);
        for start in (0..rows).step_by(chunk) {
            let end = (start + chunk).min(rows);
            // Row `i` of `retained` is `zᵢᵀ Q`, so entry `(i, j)` of the product is `w_jᵀ P zᵢ`.
            let retained = fast_ab(&points.slice(s![start..end, ..]), &frame);
            let mut smoothed = fast_abt(&retained, &coordinates);
            smoothed
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .try_for_each(|mut row| {
                    for unit in 0..row.len() {
                        row[unit] = smoothing(
                            units.activation,
                            discarded_variances[unit],
                            units.biases[unit] + row[unit],
                        )?;
                    }
                    Ok::<(), ResponseError>(())
                })?;
            let mut tile_response = fast_abt(&smoothed, &units.writers);
            tile_response += &units.output_bias;
            response.slice_mut(s![start..end, ..]).assign(&tile_response);
        }
        Ok(response)
    }

    /// R4: `V(P)`, the output variance explained by the best response of the input inside `frame` (`d × k`), with its
    /// band.
    pub fn explained_variance(&self, frame: ArrayView2<'_, f64>) -> Result<BandedEnergy, ResponseError> {
        let defect = require_frame(self.input_dim(), frame)?;
        let coordinates = fast_ab(&self.units.readers, &frame);
        self.units.pair_pass(
            coordinates.view(),
            CoordinateFormation::FrameProducts {
                frame_defect: defect,
            },
            None,
        )
    }

    /// R4 on a coordinate frame: `V(P_S)` for the projector onto the input coordinates `retained`, given strictly
    /// increasing, with its band. The covariances `r_jk = Σ_{i ∈ S} W_ji W_ki` are read from the selected reader
    /// columns, so no frame is formed.
    pub fn explained_variance_of_coordinates(&self, retained: &[usize]) -> Result<BandedEnergy, ResponseError> {
        for (position, &coordinate) in retained.iter().enumerate() {
            let increasing = position == 0 || retained[position - 1] < coordinate;
            if coordinate >= self.input_dim() || !increasing {
                return Err(ResponseError::InvalidRetainedCoordinates { position });
            }
        }
        let coordinates = self.units.readers.select(Axis(1), retained);
        self.units
            .pair_pass(coordinates.view(), CoordinateFormation::Copied, None)
    }

    /// R4: `E(P) = V(I) − V(P)`, the error of discarding the input outside `frame`, with its band.
    pub fn discarded_error(&self, frame: ArrayView2<'_, f64>) -> Result<BandedEnergy, ResponseError> {
        Ok(signed_sum(&[self.total_variance], &[self.explained_variance(frame)?]))
    }

    /// R6: `V(P)`, `E(P)` and the horizontal gradient `2 (I − Q Qᵀ) Wᵀ B(P) W Q`, from one pass over the unit pairs.
    pub fn explained_variance_gradient(
        &self,
        frame: ArrayView2<'_, f64>,
    ) -> Result<FrameGradient, ResponseError> {
        let defect = require_frame(self.input_dim(), frame)?;
        let units = &self.units;
        let coordinates = fast_ab(&units.readers, &frame);
        let mut weighted_coordinates = Array2::<f64>::zeros(coordinates.dim());
        let pass = units.pair_pass(
            coordinates.view(),
            CoordinateFormation::FrameProducts {
                frame_defect: defect,
            },
            Some(&mut weighted_coordinates),
        )?;
        let ambient = fast_atb(&units.readers, &weighted_coordinates);
        let in_frame = fast_atb(&frame, &ambient);
        let horizontal_gradient = (&ambient - &fast_ab(&frame, &in_frame)) * 2.0;
        Ok(FrameGradient {
            explained_variance: pass,
            discarded_error: signed_sum(&[self.total_variance], &[pass]),
            horizontal_gradient,
        })
    }

    /// `v_j⊥ = ‖w_j − Q Qᵀ w_j‖²` for every unit, in tiles, with `coordinates = R = W Q`. It is formed from the residual
    /// itself rather than as `‖w_j‖² − ‖Qᵀ w_j‖²`, so it is a sum of squares and never negative.
    pub fn discarded_reader_variances(
        &self,
        frame: ArrayView2<'_, f64>,
        coordinates: ArrayView2<'_, f64>,
    ) -> Array1<f64> {
        let width = self.width();
        let mut variances = Array1::<f64>::zeros(width);
        let tile = byte_balanced_row_chunk(self.input_dim(), width);
        for start in (0..width).step_by(tile) {
            let end = (start + tile).min(width);
            let residual = &self.units.readers.slice(s![start..end, ..])
                - &fast_abt(&coordinates.slice(s![start..end, ..]), &frame);
            for (offset, row) in residual.axis_iter(Axis(0)).enumerate() {
                variances[start + offset] = row.dot(&row);
            }
        }
        variances
    }
}

/// Validate a frame in `ℝ^{input_dim}` and return its measured defect [`frame_defect`], refusing a defect of at
/// least 1.
fn require_frame(input_dim: usize, frame: ArrayView2<'_, f64>) -> Result<f64, ResponseError> {
    require_length("retained frame rows", input_dim, frame.nrows())?;
    if frame.ncols() > input_dim {
        return Err(ResponseError::FrameWiderThanInput {
            rank: frame.ncols(),
            input_dim,
        });
    }
    require_finite("retained frame", frame.iter())?;
    let defect = frame_defect(frame);
    if !(defect < 1.0) {
        return Err(ResponseError::FrameNotOrthonormal { defect });
    }
    Ok(defect)
}

/// Validate an output metric for `output_dim` outputs: square of that size, finite, exactly symmetric and positive
/// definite.
fn require_metric(output_dim: usize, metric: ArrayView2<'_, f64>) -> Result<(), ResponseError> {
    require_length("output metric rows", output_dim, metric.nrows())?;
    require_length("output metric columns", output_dim, metric.ncols())?;
    require_finite("output metric", metric.iter())?;
    for row in 0..output_dim {
        for column in (row + 1)..output_dim {
            if metric[[row, column]] != metric[[column, row]] {
                return Err(ResponseError::MetricNotSymmetric { row, column });
            }
        }
    }
    metric
        .cholesky(Side::Lower)
        .map_err(|error| ResponseError::MetricNotPositiveDefinite {
            reason: error.to_string(),
        })?;
    Ok(())
}

impl BlockUnits {
    /// One tiled pass over the unit pairs `j ≤ k` for reader coordinates `coordinates` (`h × c`, with covariance
    /// `r_jk = coordinates_j · coordinates_k`). It returns `Σ_jk D_jk [K_σ − m_j m_k]` with its band
    /// ([`pair_term`](Self::pair_term)), formed as
    /// `Σ_j (D_jj [K_jj − m_j²] + 2 Σ_{k>j} D_jk [K_jk − m_j m_k])` because `D` and the pair law are symmetric. When
    /// `weighted_coordinates` is given it writes `B R` into it (`h × c`, with `B_jk = D_jk ∂_r K_σ`, symmetric by
    /// construction): row `j` is `Σ_{k≥j} B_jk R_k + Σ_{i<j} B_ij R_i`. The tiles are the reader Gram's own
    /// ([`upper_tile_rows`]), so a streamed pass reads the cached rows' words. Units are summed in order and tiles merge
    /// in order, so neither the value nor `B R` depends on the thread count or on the route to `D`.
    fn pair_pass(
        &self,
        coordinates: ArrayView2<'_, f64>,
        formation: CoordinateFormation,
        mut weighted_coordinates: Option<&mut Array2<f64>>,
    ) -> Result<BandedEnergy, ResponseError> {
        let (width, terms) = coordinates.dim();
        let input_dim = self.readers.ncols();
        let coordinate_norms: Array1<f64> = coordinates
            .rows()
            .into_iter()
            .map(|row| row.dot(&row).sqrt())
            .collect();
        let reader_norm_bounds = (&self.reader_variances + &self.reader_variance_errors).mapv(f64::sqrt);
        // A frame row `fl(w_jᵀ Q)` errs per entry by at most `γ_d ‖w_j‖ ‖q_a‖`, with `‖q_a‖² ≤ 1 + η`, and its exact
        // Gram differs from the nearest orthonormal frame's projector by at most `η ‖w_j‖ ‖w_k‖`.
        let (row_error_scale, projector_defect) = match formation {
            CoordinateFormation::Copied => (0.0, 0.0),
            CoordinateFormation::FrameProducts { frame_defect } => (
                accumulation_growth(input_dim) * (terms as f64 * (1.0 + frame_defect)).sqrt(),
                frame_defect,
            ),
        };
        if let Some(out) = weighted_coordinates.as_deref_mut() {
            out.fill(0.0);
        }
        let tile = upper_tile_rows(width);
        let mut unit_sums = Vec::with_capacity(width);
        let mut streamed_rows: Vec<f64> = Vec::new();
        let mut row_starts: Vec<usize> = Vec::with_capacity(tile + 1);
        for start in (0..width).step_by(tile) {
            let end = (start + tile).min(width);
            let rows = end - start;
            // Row `j ∈ J` against the columns `k ≥ start`, at entry `k − start`.
            let covariances = fast_abt(
                &coordinates.slice(s![start..end, ..]),
                &coordinates.slice(s![start.., ..]),
            );
            // Row `j` of `D` over `k ≥ j` starts at `row_starts[j − start]` of the tile's packed rows.
            row_starts.clear();
            let mut packed_length = 0;
            for unit in start..end {
                row_starts.push(packed_length);
                packed_length += width - unit;
            }
            row_starts.push(packed_length);
            if self.reader_gram.is_none() {
                streamed_rows.resize(packed_length, 0.0);
                fill_upper_rows(
                    self.writers.view(),
                    self.metric_writers.view(),
                    start,
                    end,
                    &mut streamed_rows,
                )
                .map_err(|error| ResponseError::ReaderGram { error })?;
            }
            // `B_J` over the columns `k ≥ start`, zero below each row's diagonal.
            let mut upper_weights = Array2::<f64>::zeros((rows, width - start));
            let sums = upper_weights
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(covariances.axis_iter(Axis(0)).into_par_iter())
                .enumerate()
                .map(|(offset, (mut weight_row, covariance_row))| {
                    let unit = start + offset;
                    let metric_row = match &self.reader_gram {
                        Some(gram) => gram.upper_row(unit),
                        None => &streamed_rows[row_starts[offset]..row_starts[offset + 1]],
                    };
                    let mut row = RowSum::default();
                    let mut off_diagonal = RowSum::default();
                    for (index, &metric_product) in metric_row.iter().enumerate() {
                        let other = unit + index;
                        let formation = CovarianceFormation {
                            terms,
                            left_norm: coordinate_norms[unit],
                            right_norm: coordinate_norms[other],
                            left_row_error: row_error_scale * reader_norm_bounds[unit],
                            right_row_error: row_error_scale * reader_norm_bounds[other],
                            law_gap: projector_defect * reader_norm_bounds[unit] * reader_norm_bounds[other],
                            variance_error_x: self.reader_variance_errors[unit],
                            variance_error_y: self.reader_variance_errors[other],
                        };
                        let pair = PreactivationPair {
                            mean_x: self.biases[unit],
                            mean_y: self.biases[other],
                            variance_x: self.reader_variances[unit],
                            variance_y: self.reader_variances[other],
                            covariance: covariance_row[other - start],
                            covariance_rounding: covariance_rounding_band(
                                &formation,
                                self.reader_variances[unit],
                                self.reader_variances[other],
                            ),
                        };
                        let moments = pair_moments(self.activation, pair)?;
                        let term = self.pair_term(unit, other, metric_product, &formation, pair, &moments)?;
                        if index == 0 {
                            row = term;
                        } else {
                            off_diagonal = off_diagonal.plus(term);
                        }
                        weight_row[other - start] = metric_product * moments.covariance_derivative;
                    }
                    Ok(row.plus(off_diagonal.doubled()))
                })
                .collect::<Result<Vec<RowSum>, ResponseError>>()?;
            unit_sums.extend(sums);
            if let Some(out) = weighted_coordinates.as_deref_mut() {
                // `Σ_{k≥j} B_jk R_k` for the rows `j ∈ J`.
                let own = fast_ab(&upper_weights, &coordinates.slice(s![start.., ..]));
                out.slice_mut(s![start..end, ..]).scaled_add(1.0, &own);
                // `Σ_{j∈J, j<k} B_jk R_j` for the rows `k ≥ start`: the strictly upper part of `B_J`, transposed.
                for offset in 0..rows {
                    upper_weights[[offset, offset]] = 0.0;
                }
                let mirrored = fast_atb(&upper_weights, &coordinates.slice(s![start..end, ..]));
                out.slice_mut(s![start.., ..]).scaled_add(1.0, &mirrored);
            }
        }
        let total = unit_sums.iter().fold(RowSum::default(), |total, row| total.plus(*row));
        // Each term enters at most `2h + 1` additions: its row's off-diagonal sum, the diagonal, and the unit fold.
        Ok(BandedEnergy {
            value: total.value,
            band: total.band + accumulation_growth(2 * width + 1) * total.absolute,
        })
    }

    /// Pair `(unit, other)`'s term `D̂ [K̂ − m̂_j m̂_k]` and a first-order bound on its distance from the exact
    /// `D [K − m_j m_k]` at the exact law, which [`pair_pass`](Self::pair_pass) sums with its terms:
    ///
    /// - the kernel's own rounding `ρ` at the projected law;
    /// - the covariance's formation error and the projection's move, at most `δr + β`, through the Lipschitz constant
    ///   `sup|σ'|²` (mean value theorem, at fixed means and variances);
    /// - each variance's rounding `δv` through the kernel's variance partial `∂K/∂v`;
    /// - the means' bands, `|m_k| δm_j + |m_j| δm_k`;
    /// - the metric product's error `δD = γ_{2p} ‖|M|‖_∞ ‖u_j‖ ‖u_k‖` times `|K̂ − m̂ m̂|`;
    /// - the term's own three roundings, `γ_3 |D̂| (|K̂| + |m̂ m̂|)`.
    fn pair_term(
        &self,
        unit: usize,
        other: usize,
        metric_product: f64,
        formation: &CovarianceFormation,
        pair: PreactivationPair,
        moments: &PairKernel,
    ) -> Result<RowSum, ResponseError> {
        let mean_product = self.unit_means[unit] * self.unit_means[other];
        let centred = moments.value - mean_product;
        let term = metric_product * centred;
        let (variance_partial_x, variance_partial_y) = self.variance_partials(unit, other, pair)?;
        let law_band = moments.value_rounding
            + self.slope_bound * (covariance_formation_error(formation) + pair.covariance_rounding)
            + variance_partial_x.abs() * self.reader_variance_errors[unit]
            + variance_partial_y.abs() * self.reader_variance_errors[other]
            + self.unit_means[other].abs() * self.unit_mean_bands[unit]
            + self.unit_means[unit].abs() * self.unit_mean_bands[other];
        let band = metric_product.abs() * law_band
            + self.metric_error_scale * self.writer_norms[unit] * self.writer_norms[other] * centred.abs()
            + accumulation_growth(3) * metric_product.abs() * (moments.value.abs() + mean_product.abs());
        Ok(RowSum {
            value: term,
            absolute: term.abs(),
            band,
        })
    }

    /// `∂K/∂v_x` and `∂K/∂v_y` of a pair at its computed law. A zero reader is a constant unit whose variance is exact,
    /// so its partial is never read, and the pair factorizes as `K = σ(b_k) m_j(v_j)`: the other unit's partial is
    /// `σ(b_k) ∂m_j/∂v_j`, with no kernel call at the constant unit's kink.
    fn variance_partials(
        &self,
        unit: usize,
        other: usize,
        pair: PreactivationPair,
    ) -> Result<(f64, f64), ResponseError> {
        if pair.variance_x == 0.0 || pair.variance_y == 0.0 {
            let partial_x = self.unit_means[other] * self.half_curvatures[unit];
            let partial_y = self.unit_means[unit] * self.half_curvatures[other];
            return Ok((partial_x, partial_y));
        }
        let partials = pair_kernel_variance_partials(self.activation, pair).map_err(|error| ResponseError::Kernel {
            context: "Gaussian pair kernel variance partials",
            error,
        })?;
        Ok((partials.variance_x, partials.variance_y))
    }
}

/// One unit row's running sum in a pair pass: the value, its absolute sum and band.
#[derive(Debug, Clone, Copy, Default)]
struct RowSum {
    value: f64,
    absolute: f64,
    band: f64,
}

impl RowSum {
    fn plus(self, other: RowSum) -> RowSum {
        RowSum {
            value: self.value + other.value,
            absolute: self.absolute + other.absolute,
            band: self.band + other.band,
        }
    }

    /// The row's strictly upper pairs counted for both orders, `(j, k)` and `(k, j)`, whose laws are mirror images.
    fn doubled(self) -> RowSum {
        RowSum {
            value: 2.0 * self.value,
            absolute: 2.0 * self.absolute,
            band: 2.0 * self.band,
        }
    }
}

/// A unit's mean `m_j = E σ(b_j + s_j E)` with a first-order bound on its error, and `½ T_v σ''(b_j) = ∂m_j/∂v_j`.
#[derive(Debug, Clone, Copy)]
struct UnitMean {
    value: f64,
    band: f64,
    half_curvature: f64,
}

/// `m_j = a_{j,0}` from the coefficient owner at `s_j = fl(√v̂_j)`, with its rounding bound, plus the input error of
/// the variance it is evaluated at: `|s_j² − v_j| ≤ δv_j + γ_2 v̂_j`, which moves `m_j` by `|∂m/∂v|` times that, with
/// `∂m/∂v = ½ T_v σ''(b)` by the heat equation. A zero reader is a constant unit with an exact variance.
fn unit_mean(
    activation: GaussianActivation,
    bias: f64,
    variance: f64,
    variance_error: f64,
) -> Result<UnitMean, ResponseError> {
    let kernel_error = |error| ResponseError::Kernel {
        context: "unit mean",
        error,
    };
    let mut coefficient = [0.0];
    let mut bound = [0.0];
    gaussian_hermite_coefficients(activation, bias, variance.sqrt(), &mut coefficient, &mut bound)
        .map_err(kernel_error)?;
    if variance == 0.0 {
        return Ok(UnitMean {
            value: coefficient[0],
            band: bound[0],
            half_curvature: 0.0,
        });
    }
    let mut derivatives = [0.0; 3];
    gaussian_smoothing_derivatives(activation, bias, variance, &mut derivatives).map_err(kernel_error)?;
    let half_curvature = 0.5 * derivatives[2];
    Ok(UnitMean {
        value: coefficient[0],
        band: bound[0] + half_curvature.abs() * (variance_error + accumulation_growth(2) * variance),
        half_curvature,
    })
}

/// A known SwiGLU block `F(z) = Σ_j u_j (a_jᵀ z + c_j) s(w_jᵀ z + b_j) + c_out` with the SiLU gate `s(t) = t σ(t)`,
/// under the declared law `Z ~ N(0, I_d)` and its output metric (#2946 R9).
#[derive(Debug, Clone)]
pub struct KnownGatedBlock {
    /// `W`, `h × d`: the gate readers.
    gate_readers: Array2<f64>,
    /// `b`, length `h`.
    gate_biases: Array1<f64>,
    /// `A`, `h × d`: the up-projection readers.
    up_readers: Array2<f64>,
    /// `c`, length `h`.
    up_biases: Array1<f64>,
    /// `U`, `p × h`.
    writers: Array2<f64>,
    /// `c_out`, length `p`.
    output_bias: Array1<f64>,
    /// `M`, `p × p`.
    metric: Array2<f64>,
}

/// A response at points with a derived bound on each entry's quadrature error. Rounding is excluded.
#[derive(Debug, Clone, PartialEq)]
pub struct BandedResponse {
    /// `n × p`.
    pub values: Array2<f64>,
    /// `n × p`: entry `(i, o)` bounds the quadrature error of `values[(i, o)]` by `Σ_j |U_oj| bound_ij`, with
    /// `bound_ij` the kernel's bound on unit `j`'s conditional mean at point `i`.
    pub quadrature_band: Array2<f64>,
}

impl KnownGatedBlock {
    /// Build the block from gate readers `W` (`h × d`) and biases `b` (`h`), up-projection readers `A` (`h × d`) and
    /// biases `c` (`h`), writers `U` (`p × h`), the output bias (`p`; zeros for a layer without one) and a symmetric
    /// positive definite output metric `M` (`p × p`). A Qwen3 layer's `gate_proj`, `up_proj` and `down_proj` give `W`,
    /// `A` and `U` once the declared law is absorbed into readers and biases.
    pub fn new(
        gate_readers: Array2<f64>,
        gate_biases: Array1<f64>,
        up_readers: Array2<f64>,
        up_biases: Array1<f64>,
        writers: Array2<f64>,
        output_bias: Array1<f64>,
        metric: ArrayView2<'_, f64>,
    ) -> Result<Self, ResponseError> {
        let (width, input_dim) = gate_readers.dim();
        require_length("gated block gate biases", width, gate_biases.len())?;
        require_length("gated block up-reader rows", width, up_readers.nrows())?;
        require_length("gated block up-reader columns", input_dim, up_readers.ncols())?;
        require_length("gated block up biases", width, up_biases.len())?;
        require_length("gated block writer columns", width, writers.ncols())?;
        require_length("gated block output bias", writers.nrows(), output_bias.len())?;
        require_finite("gated block gate readers", gate_readers.iter())?;
        require_finite("gated block gate biases", gate_biases.iter())?;
        require_finite("gated block up readers", up_readers.iter())?;
        require_finite("gated block up biases", up_biases.iter())?;
        require_finite("gated block writers", writers.iter())?;
        require_finite("gated block output bias", output_bias.iter())?;
        require_metric(writers.nrows(), metric)?;
        Ok(Self {
            gate_readers,
            gate_biases,
            up_readers,
            up_biases,
            writers,
            output_bias,
            metric: metric.to_owned(),
        })
    }

    /// Input dimension `d`.
    pub fn input_dim(&self) -> usize {
        self.gate_readers.ncols()
    }

    /// Unit count `h`.
    pub fn width(&self) -> usize {
        self.gate_readers.nrows()
    }

    /// Output dimension `p`.
    pub fn output_dim(&self) -> usize {
        self.writers.nrows()
    }

    /// The output metric `M`, `p × p`.
    pub fn metric(&self) -> ArrayView2<'_, f64> {
        self.metric.view()
    }

    /// R9: the best retained response `F̄_P(Pz)`, output bias included, at each row `z` of `points` (`n × d`), with
    /// the derived quadrature band of every entry.
    pub fn retained_response(
        &self,
        frame: ArrayView2<'_, f64>,
        points: ArrayView2<'_, f64>,
    ) -> Result<BandedResponse, ResponseError> {
        // The conditional mean reads no pair law, so the frame needs only the orthonormality refusal.
        require_frame(self.input_dim(), frame)?;
        require_length("gated retained-response point columns", self.input_dim(), points.ncols())?;
        require_finite("gated retained-response points", points.iter())?;
        let gate_coordinates = fast_ab(&self.gate_readers, &frame);
        let up_coordinates = fast_ab(&self.up_readers, &frame);
        let (discarded_variances, discarded_couplings) =
            self.discarded_law(frame, gate_coordinates.view(), up_coordinates.view());
        let absolute_writers = self.writers.mapv(f64::abs);
        // The band's own sum of `h` nonnegative products is inflated by its rounding, so it bounds the exact sum.
        let band_rounding = 1.0 + accumulation_growth(self.width());
        let rows = points.nrows();
        let mut values = Array2::<f64>::zeros((rows, self.output_dim()));
        let mut quadrature_band = Array2::<f64>::zeros((rows, self.output_dim()));
        let chunk = byte_balanced_row_chunk(3 * self.width() + self.input_dim(), rows);
        for start in (0..rows).step_by(chunk) {
            let end = (start + chunk).min(rows);
            // Row `i` of `retained` is `zᵢᵀ Q`, so the products hold `w_jᵀ P zᵢ` and `a_jᵀ P zᵢ`.
            let retained = fast_ab(&points.slice(s![start..end, ..]), &frame);
            let gate_arguments = fast_abt(&retained, &gate_coordinates);
            let up_arguments = fast_abt(&retained, &up_coordinates);
            let mut unit_means = Array2::<f64>::zeros(gate_arguments.dim());
            let mut unit_bounds = Array2::<f64>::zeros(gate_arguments.dim());
            unit_means
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .zip(unit_bounds.axis_iter_mut(Axis(0)).into_par_iter())
                .zip(gate_arguments.axis_iter(Axis(0)).into_par_iter())
                .zip(up_arguments.axis_iter(Axis(0)).into_par_iter())
                .try_for_each(|(((mut mean_row, mut bound_row), gate_row), up_row)| {
                    for unit in 0..mean_row.len() {
                        let mean = gated_mean(
                            self.up_biases[unit] + up_row[unit],
                            self.gate_biases[unit] + gate_row[unit],
                            discarded_variances[unit],
                            discarded_couplings[unit],
                        )?;
                        mean_row[unit] = mean.value.value;
                        bound_row[unit] = mean.value.quadrature_bound;
                    }
                    Ok::<(), ResponseError>(())
                })?;
            let mut tile_values = fast_abt(&unit_means, &self.writers);
            tile_values += &self.output_bias;
            values.slice_mut(s![start..end, ..]).assign(&tile_values);
            quadrature_band
                .slice_mut(s![start..end, ..])
                .assign(&(fast_abt(&unit_bounds, &absolute_writers) * band_rounding));
        }
        Ok(BandedResponse {
            values,
            quadrature_band,
        })
    }

    /// `v_j⊥ = ‖w_j − Q Qᵀ w_j‖²` and `κ_j⊥ = (a_j − Q Qᵀ a_j) · (w_j − Q Qᵀ w_j) = a_jᵀ (I − P) w_j` for every unit, in
    /// tiles, from the gate and up coordinates `W Q` and `A Q`. Both are formed from the residual rows, so `v_j⊥` is a
    /// sum of squares and never negative.
    fn discarded_law(
        &self,
        frame: ArrayView2<'_, f64>,
        gate_coordinates: ArrayView2<'_, f64>,
        up_coordinates: ArrayView2<'_, f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let width = self.width();
        let mut variances = Array1::<f64>::zeros(width);
        let mut couplings = Array1::<f64>::zeros(width);
        let tile = byte_balanced_row_chunk(2 * self.input_dim(), width);
        for start in (0..width).step_by(tile) {
            let end = (start + tile).min(width);
            let gate_residual = &self.gate_readers.slice(s![start..end, ..])
                - &fast_abt(&gate_coordinates.slice(s![start..end, ..]), &frame);
            let up_residual = &self.up_readers.slice(s![start..end, ..])
                - &fast_abt(&up_coordinates.slice(s![start..end, ..]), &frame);
            for (offset, gate_row) in gate_residual.axis_iter(Axis(0)).enumerate() {
                variances[start + offset] = gate_row.dot(&gate_row);
                couplings[start + offset] = up_residual.row(offset).dot(&gate_row);
            }
        }
        (variances, couplings)
    }
}

/// The gated unit's conditional mean `E[(α + A) s(t + W)]`, with the kernel's refusal carried as a [`ResponseError`].
pub fn gated_mean(alpha: f64, t: f64, variance: f64, coupling: f64) -> Result<GatedMean, ResponseError> {
    gated_conditional_mean(alpha, t, variance, coupling).map_err(|error| ResponseError::GatedKernel {
        context: "gated conditional mean",
        error,
    })
}

/// `T_v σ(t)`, with the kernel's refusal carried as a [`ResponseError`].
pub fn smoothing(activation: GaussianActivation, variance: f64, location: f64) -> Result<f64, ResponseError> {
    let mut value = [0.0];
    gaussian_smoothing_derivatives(activation, location, variance, &mut value).map_err(|error| {
        ResponseError::Kernel {
            context: "Gaussian smoothing",
            error,
        }
    })?;
    Ok(value[0])
}

/// `K_σ` and `∂_r K_σ` for one unit pair, with the kernel's refusal carried as a [`ResponseError`]. The kernel owns
/// the projection of a rounded covariance onto its Cauchy–Schwarz interval, within the pair's stated
/// `covariance_rounding`.
pub fn pair_moments(activation: GaussianActivation, pair: PreactivationPair) -> Result<PairKernel, ResponseError> {
    pair_kernel(activation, pair).map_err(|error| ResponseError::Kernel {
        context: "Gaussian pair kernel",
        error,
    })
}

/// `Cov(σ(X), σ(Y)) = K − m_x m_y` for one unit pair, with a bound on its error, formed without the difference where
/// the difference would cancel (#4351).
///
/// # Routes
///
/// The closed form subtracts the means' product from the pair kernel. It errs by the kernel's `value_rounding`,
/// `γ_2 (|K| + |m_x m_y|)` for the product and the difference, and `|m_x| δ_y + δ_x (|m_y| + δ_y)` for the means,
/// where `δ = e_0 + γ_1 sup|σ'| ŝ` covers `m = a_0` evaluated at `ŝ = fl(√v)`. Its band therefore scales with `|K|`,
/// not with the covariance. At small correlation, or at a large mean over a small variance, every digit of
/// `K − m_x m_y` can cancel.
///
/// The chaos series never forms the means. With `s_x = √v`, `s_y = √w` and `ρ = r/(s_x s_y)`, Mehler's formula gives
/// `Cov = Σ_{n≥1} ρⁿ a_n b_n`. Here `a_n = E[σ(b + s_x E) h_n(E)]` and `b_n` likewise
/// ([`gaussian_hermite_coefficients`]), with errors `e_n` and `f_n`. Truncated at order `N` and evaluated by Horner,
/// the series errs by:
///
/// - the tail `|Σ_{n>N} ρⁿ a_n b_n| ≤ |ρ|^{N+1} √(E_x(N) E_y(N))`, by Cauchy–Schwarz on the envelopes
///   `Σ_{n>N} a_n² ≤ E(N)` of [`unit_tail_envelope`];
/// - Horner's rounding, `γ_{3N+3} Σ_n |ρ|ⁿ |a_n b_n|` (Higham, *ASNA* §5.1; see `chaos_operations`);
/// - the coefficients' errors, `Σ_n |ρ|ⁿ (e_n |b_n| + |a_n| f_n + e_n f_n)`;
/// - the rounding of the law it is evaluated at, `γ_6 |ρ| ŝ_x ŝ_y sup|σ'|²`. Forming `ρ̂ = (r/ŝ_x)/ŝ_y` errs by
///   `γ_4 |ρ|`, and Price's theorem `∂_ρ Cov = s_x s_y E[σ'(X) σ'(Y)]` turns that into `γ_4 |ρ| s_x s_y sup|σ'|²`.
///   Each `|ŝ − s| ≤ u s` moves `a_{≥1}` in `ℓ²` by at most `sup|σ'| |ŝ − s|`, because
///   `∂_s σ(b + sE) = σ'(b + sE) E`, and the other column has `‖b_{≥1}‖ ≤ s_y sup|σ'|` by the Gaussian Poincaré
///   inequality. With `|ρ|ⁿ ≤ |ρ|`, each scale adds `γ_1 |ρ| s_x s_y sup|σ'|²`.
///
/// Every term is proportional to `|ρ|`, so the series keeps its relative accuracy as `ρ → 0` and as `|K| ≫ |Cov|`.
///
/// # Order
///
/// The order is the first `N ≥ 1` whose tail falls within Horner's floor `γ_{3N+3} |ρ a_1 b_1|`. That floor bounds
/// the series' own rounding from below, so a higher order cannot shrink the band by more than the tail it removes.
/// The floor grows with `N` and the tail shrinks. The series is dropped at the first `N` whose floor plus the law's
/// rounding already reaches the closed form's band, and this also ends the scan. Once `γ_{3N+3}` is infinite the
/// floor reaches every band. The value carries whichever route's band is smaller.
///
/// # Exact zeros
///
/// A constant pre-activation (`v = 0` or `w = 0`) or an uncorrelated projected law (`r = 0`) makes `σ(X)` and
/// `σ(Y)` independent, so the covariance is exactly zero.
///
/// The band is at the law [`project_covariance`] returns. Moving the covariance onto the Cauchy–Schwarz boundary
/// moves the value by at most `sup|σ'|² β`, and that move is the caller's to carry.
pub fn pair_covariance(activation: GaussianActivation, pair: PreactivationPair) -> Result<BandedEnergy, ResponseError> {
    let kernel = pair_moments(activation, pair)?;
    let law = project_covariance(
        pair.variance_x,
        pair.variance_y,
        pair.covariance,
        pair.covariance_rounding,
    )
    .map_err(|error| ResponseError::Kernel {
        context: "pair covariance law",
        error,
    })?;
    if pair.variance_x == 0.0 || pair.variance_y == 0.0 || law.covariance == 0.0 {
        return Ok(BandedEnergy::ZERO);
    }
    let slope_bound = activation
        .slope_bound_squared()
        .map_err(|error| ResponseError::Kernel {
            context: "pair covariance slope bound",
            error,
        })?;
    let scale_x = pair.variance_x.sqrt();
    let scale_y = pair.variance_y.sqrt();
    let (lead_x, lead_errors_x) = chaos_column(activation, pair.mean_x, scale_x, 1)?;
    let (lead_y, lead_errors_y) = chaos_column(activation, pair.mean_y, scale_y, 1)?;
    let mean_move = accumulation_growth(1) * slope_bound.sqrt();
    let mean_band_x = lead_errors_x[0] + mean_move * scale_x;
    let mean_band_y = lead_errors_y[0] + mean_move * scale_y;
    let mean_product = lead_x[0] * lead_y[0];
    let closed = BandedEnergy {
        value: kernel.value - mean_product,
        band: inflated(
            kernel.value_rounding
                + accumulation_growth(2) * (kernel.value.abs() + mean_product.abs())
                + lead_x[0].abs() * mean_band_y
                + mean_band_x * (lead_y[0].abs() + mean_band_y),
            4,
        ),
    };
    let correlation = (law.covariance / scale_x / scale_y).clamp(-1.0, 1.0);
    let magnitude = correlation.abs();
    let law_band = inflated(accumulation_growth(6) * magnitude * slope_bound * scale_x * scale_y, 4);
    let anchor = magnitude * (lead_x[1] * lead_y[1]).abs();
    if !(anchor > 0.0 && law_band < closed.band) {
        return Ok(closed);
    }
    let mut order = 1;
    let mut power = magnitude * magnitude;
    let tail = loop {
        let floor = accumulation_growth(chaos_operations(order)) * anchor;
        if !(floor + law_band < closed.band) {
            return Ok(closed);
        }
        let envelopes = chaos_tail(activation, pair.mean_x, scale_x, order)?
            * chaos_tail(activation, pair.mean_y, scale_y, order)?;
        // `power` took `order + 1` products, the square root and the product with it one more each.
        let tail = inflated(power * envelopes.sqrt(), order + 3);
        if tail <= floor {
            break tail;
        }
        order += 1;
        power *= magnitude;
    };
    let (column_x, errors_x) = chaos_column(activation, pair.mean_x, scale_x, order)?;
    let (column_y, errors_y) = chaos_column(activation, pair.mean_y, scale_y, order)?;
    let mut value = 0.0;
    let mut absolute = 0.0;
    let mut coefficient_band = 0.0;
    for degree in (1..=order).rev() {
        let product = column_x[degree] * column_y[degree];
        value = product + correlation * value;
        absolute = product.abs() + magnitude * absolute;
        coefficient_band = errors_x[degree] * column_y[degree].abs()
            + column_x[degree].abs() * errors_y[degree]
            + errors_x[degree] * errors_y[degree]
            + magnitude * coefficient_band;
    }
    let operations = chaos_operations(order);
    let series = BandedEnergy {
        value: correlation * value,
        band: inflated(
            tail + accumulation_growth(operations) * magnitude * absolute + magnitude * coefficient_band + law_band,
            operations,
        ),
    };
    Ok(if series.band < closed.band { series } else { closed })
}

/// `a_0, …, a_order` of `σ(bias + scale E)` with their rounding bounds.
fn chaos_column(
    activation: GaussianActivation,
    bias: f64,
    scale: f64,
    order: usize,
) -> Result<(Vec<f64>, Vec<f64>), ResponseError> {
    let mut coefficients = vec![0.0; order + 1];
    let mut bounds = vec![0.0; order + 1];
    gaussian_hermite_coefficients(activation, bias, scale, &mut coefficients, &mut bounds).map_err(|error| {
        ResponseError::Kernel {
            context: "pair covariance Hermite coefficients",
            error,
        }
    })?;
    Ok((coefficients, bounds))
}

/// `E(order) ≥ Σ_{n>order} a_n²` for `σ(bias + scale E)`, with the expansion's refusal carried as a [`ResponseError`].
fn chaos_tail(activation: GaussianActivation, bias: f64, scale: f64, order: usize) -> Result<f64, ResponseError> {
    unit_tail_envelope(activation, bias, scale, order).map_err(|error| match error {
        HermiteError::NoClosedFormActivation { activation } => ResponseError::Kernel {
            context: "pair covariance tail envelope",
            error: GaussianActivationError::NoClosedForm { activation },
        },
        _ => ResponseError::NonFinite {
            context: "pair covariance tail envelope",
        },
    })
}

/// Refuse a length that differs from the one the block requires.
pub fn require_length(context: &'static str, expected: usize, got: usize) -> Result<(), ResponseError> {
    if expected == got {
        Ok(())
    } else {
        Err(ResponseError::DimensionMismatch {
            context,
            expected,
            got,
        })
    }
}

/// Refuse an input with a non-finite entry.
pub fn require_finite<'a>(
    context: &'static str,
    mut values: impl Iterator<Item = &'a f64>,
) -> Result<(), ResponseError> {
    if values.all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ResponseError::NonFinite { context })
    }
}

#[cfg(test)]
#[path = "subspace_tests.rs"]
mod subspace_tests;

#[cfg(test)]
#[path = "subspace_route_tests.rs"]
mod subspace_route_tests;
