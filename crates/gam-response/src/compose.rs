//! Composition across known blocks under the declared law (#2946).
//!
//! # Conditional means do not compose
//!
//! `E ReLU(Z)² = 1/2`, but `(E ReLU Z)² = 1/(2π)`. A second stage `F₂` after a block `F₁` reads the conditional LAW of
//! `Y = F₁(Z)` given the retained input, not its conditional mean.
//!
//! # The first block's law at a retained point
//!
//! With `F₁(z) = U σ(b + W z) + c₁` and `P = Q Qᵀ`, the independence of `PZ` and `(I − P)Z` gives, exactly,
//!
//! ```text
//! a = b + W Z | PZ = Pz  ~  N(μ, Σ⊥),   μ = b + W P z,   Σ⊥ = W (I − P) Wᵀ,
//! C_jl = Cov(σ(a_j), σ(a_l) | PZ = Pz) = K_σ(μ_j, μ_l; v_j⊥, v_l⊥, Σ⊥_jl) − τ_j τ_l,   τ_j = T_{v_j⊥} σ(μ_j),
//! ```
//!
//! with `K_σ` the biased pair kernel at the DISCARDED covariance. `Y | PZ` is the pushforward of that Gaussian by `U σ`:
//! known, and not Gaussian. `C_jl` vanishes with the discarded variances while `K_σ` and `τ_j τ_l` do not, so it is
//! formed centred, by the Mehler series where that is the tighter route ([`pair_covariance`]).
//!
//! # Exact compositions
//!
//! - An affine last stage `F₂(y) = A y + c`. `F₂∘F₁ = (A U) σ(b + W z) + A c₁ + c` is one block, so the retained
//!   response, `V(P)`, `E(P)` and the frame gradient of the composition are those of `(W, b, A U, A c₁ + c)`
//!   ([`compose_affine_stage`]).
//! - A quadratic readout `yᵀ A y`. `E[Yᵀ A Y | PZ] = Ȳᵀ A Ȳ + tr(A U C Uᵀ)` with `Ȳ = U τ + c₁`, so the mean composition
//!   `Ȳᵀ A Ȳ` fails by exactly `tr(A U C Uᵀ)` ([`quadratic_readout`]). For one ReLU unit at `P = 0` that is
//!   `1/2 − 1/(2π)`.
//! - Retained readers, `W (I − P) = 0`. `Y` is a function of `PZ`, so `E[F₂(F₁(Z)) | PZ] = F₂(F₁(Pz))` for every `F₂`.
//! - The residual form `F(z) = z + G(z)`. `F̄_P = Pz + Ḡ_P`, and since `E[(I − P)Z σ(b_j + w_jᵀZ)] = (I − P) w_j
//!   E σ'(b_j + w_jᵀZ)`, the energies carry a Stein cross term ([`residual_block_energies`]):
//!
//!   ```text
//!   E(P) = tr(M (I − P)) + 2 Σ_j u_jᵀ M (I − P) w_j (T_{v_j} σ)'(b_j) + E_G(P),   v_j = ‖w_j‖².
//!   ```
//!
//!   The cross term can be negative, and `E(P)` is an expected squared norm, so it never is. The error of `z + G(z)` is
//!   not the skip's error plus `G`'s.
//!
//! A nonlinear `F₂` reading discarded units needs an integral over `rank W (I − P)` dimensions of a non-smooth function,
//! which has no fixed-cost route.
//!
//! # The Gaussian closure (approximate)
//!
//! Replacing the law of `Y | PZ` by the Gaussian with its exact mean `U τ + c₁` and covariance `U C Uᵀ` gives
//!
//! ```text
//! F̂₂(Pz) = c₂ + Σ_k u₂ₖ T_{s_k²} σ₂(m_k),   m_k = b₂ₖ + w₂ₖᵀ (U τ + c₁),   s_k² = α_kᵀ C α_k,   α_k = Uᵀ w₂ₖ.
//! ```
//!
//! `m_k` and `s_k²` are the exact conditional mean and variance of the second block's pre-activation `c_k`; only its
//! shape is closed. The closure is exact for affine and quadratic stages and APPROXIMATE otherwise. Its error comes from
//! the cumulants of `c_k` beyond the second, and it is measured against the executed composition, never assumed
//! ([`GaussianClosureResponse`]).
//!
//! # The ReLU two-moment bound
//!
//! `ReLU(x) = (x + |x|)/2`, so the means cancel. `E|c| ∈ [|m|, √(m² + s²)]` for every law with mean `m` and variance
//! `s²` (Jensen, Cauchy–Schwarz), and two-point laws attain or approach both ends. Hence
//!
//! ```text
//! |E ReLU(c_k) − T_{s_k²} ReLU(m_k)| ≤ e_k = ½ max(√(m_k² + s_k²) − g_k, g_k − |m_k|),   g_k = E|ĉ_k|, ĉ_k ~ N(m_k, s_k²),
//! ```
//!
//! which is sharp over that class, and `‖E F₂ − F̂₂‖_M ≤ Σ_k ‖u₂ₖ‖_M e_k`. The bound holds on this domain only: a ReLU
//! second stage whose `m_k` and `s_k²` are exact to the kernels' accuracy. It is informative off the kink
//! (`e ≈ s²/(4|m|)` for `|m| ≫ s`) and equals the closure itself at `m = 0`. No deterministic bound is claimed for any
//! other second-stage activation.
//!
//! # The enlarged state (exact)
//!
//! In a residual stack `out = z + G₁(z) + G₂(z + G₁(z))`, with `Gᵢ(y) = Uᵢ σ(bᵢ + Wᵢ y) + cᵢ` both reading and writing
//! the stream `ℝ^d`, the second block's pre-activations are
//!
//! ```text
//! c_k = b₂ₖ + w₂ₖᵀ c₁ + w₂ₖᵀ z + Σ_j R_kj σ(a_j),   R = W₂ U₁,   a = b₁ + W₁ z.
//! ```
//!
//! Only the read units `N = {j : R_{·j} ≠ 0}` enter. On a frame `P̃` that retains their readers, `W₁,N (I − P̃) = 0`, so
//! those `a_j` are functions of `P̃Z`. Given `P̃Z`, the pre-activations of BOTH blocks are then jointly Gaussian with the
//! skip `(I − P̃)Z`. The stack is one residual block of width `h₁ + h₂` whose second-block biases move with the point,
//! `b₂ + W₂ c₁ + R τ(P̃z)`, and over its units `u` ([`residual_stack_response`]):
//!
//! ```text
//! E[out | P̃Z] = P̃z + c₁ + c₂ + U τ,
//! E[‖out − E[out | P̃Z]‖²_M | P̃Z] = tr(M (I − P̃)) + 2 Σ_u (T_{v_u⊥} σ)'(μ_u) u_uᵀ M (I − P̃) w_u + tr(M U C Uᵀ),
//! ```
//!
//! with `U = [U₁ U₂]`, `C` the units' covariance at the discarded law, and the Stein term
//! `E[(I − P̃)Z σ(a_u) | P̃Z] = (I − P̃) w_u (T_{v_u⊥} σ)'(μ_u)`. The frame that retains the read readers is
//! `Q̃ = orth[Q, W₁,Nᵀ]` ([`enlarged_frame`]).
//!
//! On any other frame the same evaluation is exact for the model `c'_k = c_k − ℓ_k`,
//! `ℓ_k = Σ_{j ∈ N} R_kj (σ(a_j) − τ_j)`, which is coupled to the stack on one probability space. The Gaussian Poincaré
//! inequality gives `‖σ(a_j) − τ_j‖_{L²} ≤ L √v_j⊥`, with `L² = sup|σ'|²`, and `σ` is `L`-Lipschitz. So at every point
//!
//! ```text
//! ‖out − out'‖_{L², M} ≤ Λ = L² Σ_k ‖u₂ₖ‖_M Σ_{j ∈ N} |R_kj| √v_j⊥.
//! ```
//!
//! The conditional mean then moves by at most `Λ` in `M`-norm, and the root conditional variance by at most `Λ`. On
//! `Q̃`, `Λ` is the rounding of the retained read readers.
//!
//! The price:
//! - `k̃ ≤ min(d, k + |N|)`. A dense second block reads every unit, and an expanding first block (`h₁ ≥ d − k`) then makes
//!   `P̃ = I`, where nothing is compressed.
//! - `V(P̃)` and `E(P̃)` have no pair-kernel closure, because the moving biases are not Gaussian in `P̃Z`. `E(P̃)` is the
//!   average of the exact conditional variance over `P̃Z`, an outer integral over `k̃` dimensions.
//! - One point costs `(h₁ + h₂)²` pair kernels.
//!
//! # Cost
//!
//! One retained point costs `h₁²` pair-kernel evaluations plus `O(h₁² q)` flops for `q` projected terms. Units are tiled,
//! so no `h₁ × h₁` matrix is resident.

use super::subspace::{
    CovarianceFormation, KnownBlock, ResponseError, covariance_rounding_band, frame_defect, pair_covariance,
    require_finite, require_length, smoothing,
};
use gam_linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb};
use gam_linalg::roundoff::accumulation_growth;
use gam_math::gaussian_activation::{GaussianActivation, PreactivationPair, gaussian_smoothing_derivatives};
use gam_math::probability::{normal_cdf, normal_pdf};
use gam_math::roundoff::inflated;
use gam_runtime::resource::byte_balanced_row_chunk;
use gam_solve::penalty_invariance::orthonormalize_columns;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rayon::prelude::*;
use std::fmt;

/// Compose an affine last stage `F₂(y) = A y + c` after a known block, with `matrix = A` (`q × p`), `offset = c` (`q`)
/// and the output metric `metric` on `ℝ^q`.
///
/// `F₂∘F₁ = (A U) σ(b + W z) + A c₁ + c` is itself one block, so every quantity of the retained-response operator is
/// exact for the composition, and the output bias drops out of `V(P)`, `E(P)` and the frame gradient.
pub fn compose_affine_stage(
    first: &KnownBlock,
    matrix: ArrayView2<'_, f64>,
    offset: ArrayView1<'_, f64>,
    metric: ArrayView2<'_, f64>,
) -> Result<KnownBlock, ResponseError> {
    require_length("affine stage columns", first.output_dim(), matrix.ncols())?;
    require_length("affine stage offset", matrix.nrows(), offset.len())?;
    require_finite("affine stage matrix", matrix.iter())?;
    require_finite("affine stage offset", offset.iter())?;
    KnownBlock::new(
        first.readers().to_owned(),
        first.biases().to_owned(),
        fast_ab(&matrix, &first.writers()),
        matrix.dot(&first.output_bias()) + &offset,
        metric,
        first.activation(),
    )
}

/// A quadratic readout `yᵀ A y` of a known block at retained points.
#[derive(Debug, Clone)]
pub struct QuadraticReadout {
    /// `E[F₁(Z)ᵀ A F₁(Z) | PZ = Pz]`, exact.
    pub exact: Array1<f64>,
    /// `F̄_{1,P}ᵀ A F̄_{1,P}`, the mean composition.
    pub mean_composition: Array1<f64>,
    /// `tr(A U C Uᵀ)`, the exact amount by which the mean composition fails.
    pub covariance_term: Array1<f64>,
}

/// The exact conditional mean of `F₁(Z)ᵀ A F₁(Z)` given the input inside `frame` (`d × k`), at each row of `points`
/// (`n × d`). `form` is `A` (`p × p`); only its symmetric part is read.
pub fn quadratic_readout(
    first: &KnownBlock,
    form: ArrayView2<'_, f64>,
    frame: ArrayView2<'_, f64>,
    points: ArrayView2<'_, f64>,
) -> Result<QuadraticReadout, ResponseError> {
    require_length("quadratic form rows", first.output_dim(), form.nrows())?;
    require_length("quadratic form columns", first.output_dim(), form.ncols())?;
    require_finite("quadratic form", form.iter())?;
    let law = DiscardedLaw::new(first, frame, points, None)?;
    let writers = first.writers();
    let mut conditional_means = fast_abt(&law.unit_means, &writers);
    conditional_means += &first.output_bias();
    let formed_means = fast_ab(&conditional_means, &form);
    let mean_composition: Array1<f64> = formed_means
        .rows()
        .into_iter()
        .zip(conditional_means.rows())
        .map(|(formed, mean)| formed.dot(&mean))
        .collect();
    // With `L = Uᵀ` and `R = (A U)ᵀ`, `Σ_k diag(Lᵀ C R)_k = Σ_jl (Uᵀ A U)_jl C_jl = tr(A U C Uᵀ)`.
    let formed_writers = fast_ab(&form, &writers);
    let covariance_term = law
        .covariance_diagonals(first.readers(), first.activation(), frame, writers.t(), formed_writers.t())?
        .sum_axis(Axis(1));
    let exact = &mean_composition + &covariance_term;
    Ok(QuadraticReadout {
        exact,
        mean_composition,
        covariance_term,
    })
}

/// `V(P)` and `E(P)` of the residual form `F(z) = z + G(z)`.
#[derive(Debug, Clone)]
pub struct ResidualEnergies {
    /// `V(P) = tr(M P) + 2 Σ_j u_jᵀ M P w_j (T_{v_j} σ)'(b_j) + V_G(P)`.
    pub explained_variance: f64,
    /// `E(P) = tr(M (I − P)) + 2 Σ_j u_jᵀ M (I − P) w_j (T_{v_j} σ)'(b_j) + E_G(P)`, an expected squared norm, so never
    /// negative.
    pub discarded_error: f64,
    /// The Stein cross term `2 Σ_j u_jᵀ M (I − P) w_j (T_{v_j} σ)'(b_j)` inside `discarded_error`, which can be negative.
    pub discarded_cross_term: f64,
}

/// The energies of the residual form `F(z) = z + G(z)` for the input inside `frame` (`d × k`). `block` is `G`, writing
/// into its own input space under the output metric `M` (`d × d`).
///
/// The skip must carry the block's declared input `z` itself. A pre-norm transformer block's skip carries the pre-norm
/// stream, not the post-norm input the declared law is placed on, so this form does not describe such a block's
/// residual output.
///
/// The discarded cross term is formed from the discarded reader parts `w_j − Q Qᵀ w_j`, tiled, rather than as a
/// difference of the full and retained sums.
pub fn residual_block_energies(
    block: &KnownBlock,
    frame: ArrayView2<'_, f64>,
) -> Result<ResidualEnergies, ResponseError> {
    require_length("residual block output dimension", block.input_dim(), block.output_dim())?;
    let explained_mlp = block.explained_variance(frame)?.value;
    let readers = block.readers();
    let metric = block.metric();
    let metric_writers = block.metric_writers();
    let activation = block.activation();
    // `(T_{v_j} σ)'(b_j) = E σ'(b_j + w_jᵀ Z)` with `v_j = ‖w_j‖²`.
    let slopes = readers
        .rows()
        .into_iter()
        .zip(block.biases().iter())
        .map(|(reader, &bias)| {
            let mut jet = [0.0; 2];
            gaussian_smoothing_derivatives(activation, bias, reader.dot(&reader), &mut jet).map_err(|error| {
                ResponseError::Kernel {
                    context: "Gaussian smoothing slope",
                    error,
                }
            })?;
            Ok(jet[1])
        })
        .collect::<Result<Array1<f64>, ResponseError>>()?;
    let total_skip = metric.diag().sum();
    let retained_skip: f64 = frame
        .columns()
        .into_iter()
        .zip(fast_ab(&metric, &frame).columns())
        .map(|(column, metric_column)| column.dot(&metric_column))
        .sum();
    let width = block.width();
    let coordinates = fast_ab(&readers, &frame);
    // Column `j` of `Qᵀ M U` dotted with row `j` of `R = W Q` is `u_jᵀ M P w_j`.
    let retained_writers = fast_atb(&frame, &metric_writers);
    let retained_cross: f64 = (0..width)
        .map(|unit| slopes[unit] * retained_writers.column(unit).dot(&coordinates.row(unit)))
        .sum();
    let tile = byte_balanced_row_chunk(block.input_dim(), width);
    let mut discarded_cross = 0.0;
    for start in (0..width).step_by(tile) {
        let end = (start + tile).min(width);
        let residual =
            &readers.slice(s![start..end, ..]) - &fast_abt(&coordinates.slice(s![start..end, ..]), &frame);
        for (offset, residual_reader) in residual.axis_iter(Axis(0)).enumerate() {
            let unit = start + offset;
            discarded_cross += slopes[unit] * residual_reader.dot(&metric_writers.column(unit));
        }
    }
    let discarded_cross_term = 2.0 * discarded_cross;
    Ok(ResidualEnergies {
        explained_variance: retained_skip + 2.0 * retained_cross + explained_mlp,
        discarded_error: (total_skip - retained_skip) + discarded_cross_term + (block.total_variance().value - explained_mlp),
        discarded_cross_term,
    })
}

/// The Gaussian closure of a second block after a first, at retained points. APPROXIMATE: see the module docs.
#[derive(Debug, Clone)]
pub struct GaussianClosureResponse {
    /// `F̂₂(Pz) = c₂ + Σ_k u₂ₖ T_{s_k²} σ₂(m_k)`, `n × p₂`.
    pub approximate_response: Array2<f64>,
    /// `m_k = E[c_k | PZ = Pz]`, the exact conditional means of the second block's pre-activations, `n × h₂`.
    pub preactivation_means: Array2<f64>,
    /// `s_k² = Var(c_k | PZ = Pz)`, their exact conditional variances, `n × h₂`.
    pub preactivation_variances: Array2<f64>,
    /// `Σ_k ‖u₂ₖ‖_M e_k` at each point, a deterministic bound on `‖E[F₂(F₁(Z)) | PZ] − F̂₂‖_M`, present only when the
    /// second block's activation is ReLU.
    pub relu_moment_class_bound: Option<Array1<f64>>,
}

/// The Gaussian closure `F̂₂` of `second ∘ first` given the input inside `frame` (`d × k`), at each row of `points`
/// (`n × d`). The second block reads the first block's output: `second.input_dim() == first.output_dim()`.
pub fn gaussian_closure_response(
    first: &KnownBlock,
    second: &KnownBlock,
    frame: ArrayView2<'_, f64>,
    points: ArrayView2<'_, f64>,
) -> Result<GaussianClosureResponse, ResponseError> {
    require_length("second block input dimension", first.output_dim(), second.input_dim())?;
    let law = DiscardedLaw::new(first, frame, points, None)?;
    let first_writers = first.writers();
    let second_readers = second.readers();
    let mut conditional_means = fast_abt(&law.unit_means, &first_writers);
    conditional_means += &first.output_bias();
    let mut preactivation_means = fast_abt(&conditional_means, &second_readers);
    preactivation_means += &second.biases();
    // `α = Uᵀ W₂ᵀ` (`h₁ × h₂`), so `diag(αᵀ C α)` holds every `s_k²`.
    let reads = fast_atb(&first_writers, &second_readers.t());
    let mut preactivation_variances =
        law.covariance_diagonals(first.readers(), first.activation(), frame, reads.view(), reads.view())?;
    // A variance is never negative, so projecting a rounded value onto `[0, ∞)` never moves it farther from the exact one.
    preactivation_variances.mapv_inplace(|variance| variance.max(0.0));
    let activation = second.activation();
    let mut smoothed = preactivation_means.clone();
    smoothed
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(preactivation_variances.axis_iter(Axis(0)).into_par_iter())
        .try_for_each(|(mut row, variances)| {
            for unit in 0..row.len() {
                row[unit] = smoothing(activation, variances[unit], row[unit])?;
            }
            Ok::<(), ResponseError>(())
        })?;
    let mut approximate_response = fast_abt(&smoothed, &second.writers());
    approximate_response += &second.output_bias();
    let relu_moment_class_bound = if matches!(activation, GaussianActivation::Relu) {
        let writer_norms = (&second.writers() * &second.metric_writers())
            .sum_axis(Axis(0))
            .mapv(f64::sqrt);
        Some(
            preactivation_means
                .rows()
                .into_iter()
                .zip(preactivation_variances.rows())
                .map(|(means, variances)| {
                    means
                        .iter()
                        .zip(variances.iter())
                        .zip(writer_norms.iter())
                        .map(|((&mean, &variance), &norm)| norm * relu_two_moment_bound(mean, variance))
                        .sum::<f64>()
                })
                .collect(),
        )
    } else {
        None
    };
    Ok(GaussianClosureResponse {
        approximate_response,
        preactivation_means,
        preactivation_variances,
        relu_moment_class_bound,
    })
}

/// `½ max(√(m² + s²) − g, g − |m|)` with `g = E|ĉ|`, `ĉ ~ N(m, s²)`: the sharp bound on `|E ReLU(c) − T_{s²} ReLU(m)|`
/// over every law of `c` with mean `m` and variance `s² = variance`.
///
/// Both branches are formed without cancellation against `|m|`: `g − |m| = 2s (φ(x) − x Φ(−x))` with `x = |m|/s`, and
/// `√(m² + s²) − |m| = s² / (√(m² + s²) + |m|)`.
pub fn relu_two_moment_bound(mean: f64, variance: f64) -> f64 {
    if variance == 0.0 {
        // `c = m` almost surely, and the closure is the executed value.
        return 0.0;
    }
    let deviation = variance.sqrt();
    let standardized = mean.abs() / deviation;
    let gaussian_excess = if standardized.is_finite() {
        2.0 * deviation * (normal_pdf(standardized) - standardized * normal_cdf(-standardized))
    } else {
        0.0
    };
    let second_moment_excess = variance / (mean.hypot(deviation) + mean.abs());
    0.5 * (second_moment_excess - gaussian_excess).max(gaussian_excess)
}

/// A refusal of the residual-stack route.
#[derive(Debug, Clone, PartialEq)]
pub enum ResidualStackError {
    Response(ResponseError),
    /// The joint law of one unit from each block needs the pair kernel `E σ₁(X) σ₂(Y)`, and the kernel owner has it
    /// only for one activation.
    MixedActivations {
        first: GaussianActivation,
        second: GaussianActivation,
    },
}

impl From<ResponseError> for ResidualStackError {
    fn from(error: ResponseError) -> Self {
        Self::Response(error)
    }
}

impl fmt::Display for ResidualStackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Response(error) => write!(f, "{error}"),
            Self::MixedActivations { first, second } => write!(
                f,
                "a residual stack of a {first:?} block and a {second:?} block needs a mixed-activation pair kernel"
            ),
        }
    }
}

impl std::error::Error for ResidualStackError {}

/// The first block's units that a second residual block reads, and a frame that retains their readers.
#[derive(Debug, Clone)]
pub struct EnlargedFrame {
    /// `Q̃`, `d × k̃`: the caller's frame and then the read units' readers, orthonormalized in that order. A column whose
    /// residual the projection arithmetic cannot resolve from zero is dropped, so `k̃ ≤ min(d, k + |N|)`.
    pub frame: Array2<f64>,
    /// `N`, increasing: the units `j` with `U₁[i, j] ≠ 0` at a coordinate `i` that some second-block reader reads.
    pub read_units: Vec<usize>,
}

/// The enlarged frame `Q̃ = orth[Q, W₁,Nᵀ]` of the residual stack `z + G₁(z) + G₂(z + G₁(z))` for the caller's `frame`
/// (`d × k`), on which [`residual_stack_response`] is exact. `first` is `G₁` and `second` is `G₂`.
///
/// `N` is decided on the factors of `R = W₂ U₁`, not on its computed entries. Unit `j` is unread exactly when every
/// product `w₂ₖᵢ u₁ᵢⱼ` has a zero factor, so a computed zero of `R` from cancellation or underflow never drops a read unit.
pub fn enlarged_frame(
    first: &KnownBlock,
    second: &KnownBlock,
    frame: ArrayView2<'_, f64>,
) -> Result<EnlargedFrame, ResidualStackError> {
    require_residual_stack(first, second)?;
    let input_dim = first.input_dim();
    require_length("retained frame rows", input_dim, frame.nrows())?;
    require_finite("retained frame", frame.iter())?;
    let second_readers = second.readers();
    let read_coordinates: Vec<usize> = (0..input_dim)
        .filter(|&coordinate| second_readers.column(coordinate).iter().any(|&entry| entry != 0.0))
        .collect();
    let first_writers = first.writers();
    let read_units: Vec<usize> = (0..first.width())
        .filter(|&unit| {
            read_coordinates
                .iter()
                .any(|&coordinate| first_writers[[coordinate, unit]] != 0.0)
        })
        .collect();
    let rank = frame.ncols();
    let first_readers = first.readers();
    let mut columns = Array2::<f64>::zeros((input_dim, rank + read_units.len()));
    columns.slice_mut(s![.., ..rank]).assign(&frame);
    for (offset, &unit) in read_units.iter().enumerate() {
        columns.column_mut(rank + offset).assign(&first_readers.row(unit));
    }
    let frame = orthonormalize_columns(&columns).unwrap_or_else(|| Array2::<f64>::zeros((input_dim, 0)));
    Ok(EnlargedFrame { frame, read_units })
}

/// The conditional law of a residual stack's output given the input inside a frame, at retained points.
#[derive(Debug, Clone)]
pub struct ResidualStackResponse {
    /// `E[out' | P̃Z = P̃z]`, `n × d`, for the model `out'` of the module docs.
    pub conditional_mean: Array2<f64>,
    /// `E[‖out' − E[out' | P̃Z]‖²_M | P̃Z = P̃z]`, length `n`.
    pub conditional_variance: Array1<f64>,
    /// `Λ ≥ ‖out − out'‖_{L², M}` at every point: the stack's conditional mean lies within `Λ` of `conditional_mean` in
    /// `M`-norm, and its root conditional variance within `Λ` of `√conditional_variance`. On an [`EnlargedFrame`] it is
    /// the rounding of the retained read readers.
    pub state_leak: f64,
}

/// The conditional mean and variance of `out = z + G₁(z) + G₂(z + G₁(z))` given the input inside `frame` (`d × k`), at
/// each row of `points` (`n × d`), with the state leak `Λ` that bounds their distance from the stack's own. They are
/// exact on an [`EnlargedFrame`].
///
/// `first` is `G₁` and `second` is `G₂`. Both read and write `ℝ^d` and share one activation, and the stack writes into
/// the second block's output space, so its metric `M` is the second block's.
pub fn residual_stack_response(
    first: &KnownBlock,
    second: &KnownBlock,
    frame: ArrayView2<'_, f64>,
    points: ArrayView2<'_, f64>,
) -> Result<ResidualStackResponse, ResidualStackError> {
    require_residual_stack(first, second)?;
    let input_dim = first.input_dim();
    let first_law = DiscardedLaw::new(first, frame, points, None)?;
    let second_readers = second.readers();
    let first_writers = first.writers();
    // `R = W₂ U₁` (`h₂ × h₁`). At point `i` the second block's biases are `b₂ + W₂ c₁ + R τᵢ`.
    let reads = fast_ab(&second_readers, &first_writers);
    let mut shifts = fast_abt(&first_law.unit_means, &reads);
    shifts += &second_readers.dot(&first.output_bias());
    let second_law = DiscardedLaw::new(second, frame, points, Some(shifts.view()))?;
    let state_leak = state_leak(first, second, &first_law, reads.view())?;
    let law = DiscardedLaw::stacked(&first_law, &second_law);
    let readers = stack_rows(first.readers(), second_readers);
    let writers = stack_columns(first_writers, second.writers());
    let metric = second.metric();
    let metric_writers = fast_ab(&metric, &writers);
    let activation = first.activation();
    let width = readers.nrows();

    let mut conditional_mean = fast_abt(&fast_ab(&points, &frame), &frame);
    conditional_mean += &fast_abt(&law.unit_means, &writers);
    conditional_mean += &(&first.output_bias() + &second.output_bias());

    let retained_skip: f64 = frame
        .columns()
        .into_iter()
        .zip(fast_ab(&metric, &frame).columns())
        .map(|(column, metric_column)| column.dot(&metric_column))
        .sum();
    let discarded_skip = metric.diag().sum() - retained_skip;
    // `u_uᵀ M (I − Q Qᵀ) w_u`, formed from the discarded reader parts in tiles.
    let mut skip_couplings = Array1::<f64>::zeros(width);
    let tile = byte_balanced_row_chunk(input_dim, width);
    for start in (0..width).step_by(tile) {
        let end = (start + tile).min(width);
        let residual =
            &readers.slice(s![start..end, ..]) - &fast_abt(&law.coordinates.slice(s![start..end, ..]), &frame);
        for (offset, residual_reader) in residual.axis_iter(Axis(0)).enumerate() {
            skip_couplings[start + offset] = residual_reader.dot(&metric_writers.column(start + offset));
        }
    }
    let stein_terms = law
        .means
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|means| {
            let mut sum = 0.0;
            for unit in 0..width {
                let mut jet = [0.0; 2];
                gaussian_smoothing_derivatives(activation, means[unit], law.discarded_variances[unit], &mut jet)
                    .map_err(|error| ResponseError::Kernel {
                        context: "Gaussian smoothing slope",
                        error,
                    })?;
                sum += jet[1] * skip_couplings[unit];
            }
            Ok(sum)
        })
        .collect::<Result<Vec<f64>, ResponseError>>()?;
    let unit_terms = law
        .covariance_diagonals(readers.view(), activation, frame, writers.t(), metric_writers.t())?
        .sum_axis(Axis(1));
    // An expected squared norm is never negative, so projecting a rounded value onto `[0, ∞)` never moves it farther
    // from the exact one.
    let conditional_variance = stein_terms
        .iter()
        .zip(unit_terms.iter())
        .map(|(&stein, &units)| (discarded_skip + 2.0 * stein + units).max(0.0))
        .collect();
    Ok(ResidualStackResponse {
        conditional_mean,
        conditional_variance,
        state_leak,
    })
}

/// Both blocks of a residual stack read and write `ℝ^d` and share one activation.
fn require_residual_stack(first: &KnownBlock, second: &KnownBlock) -> Result<(), ResidualStackError> {
    let input_dim = first.input_dim();
    require_length("first residual block output dimension", input_dim, first.output_dim())?;
    require_length("second residual block input dimension", input_dim, second.input_dim())?;
    require_length("second residual block output dimension", input_dim, second.output_dim())?;
    if first.activation() != second.activation() {
        return Err(ResidualStackError::MixedActivations {
            first: first.activation(),
            second: second.activation(),
        });
    }
    Ok(())
}

/// `Λ = L² Σ_k ‖u₂ₖ‖_M Σ_j |R_kj| √v_j⊥` of the module docs, formed as an upper bound on its exact value.
fn state_leak(
    first: &KnownBlock,
    second: &KnownBlock,
    first_law: &DiscardedLaw,
    reads: ArrayView2<'_, f64>,
) -> Result<f64, ResponseError> {
    let lipschitz_squared = first
        .activation()
        .slope_bound_squared()
        .map_err(|error| ResponseError::Kernel {
            context: "activation slope bound",
            error,
        })?;
    let input_dim = first.input_dim();
    // `|R_kj| ≤ |R̂_kj| + γ_d (|W₂| |U₁|)_kj` for the computed `d`-term product (Higham, ASNA §3.1).
    let absolute_reads = fast_ab(&second.readers().mapv(f64::abs), &first.writers().mapv(f64::abs));
    let reads_bound = reads.mapv(f64::abs) + absolute_reads * accumulation_growth(input_dim);
    // `√(v̂⊥ + e) ≥ √v⊥`, with `e` the bound on the discarded variance's error.
    let deviations = (&first_law.discarded_variances + &first_law.discarded_variance_errors).mapv(f64::sqrt);
    let writer_norms = (&second.writers() * &second.metric_writers())
        .sum_axis(Axis(0))
        .mapv(f64::sqrt);
    let leak = lipschitz_squared * writer_norms.dot(&reads_bound.dot(&deviations));
    // A sum of nonnegative terms rounds by at most `γ` of its longest chain, relative.
    Ok(inflated(leak, input_dim + first.width() + second.width()))
}

/// `[top; bottom]`.
fn stack_rows(top: ArrayView2<'_, f64>, bottom: ArrayView2<'_, f64>) -> Array2<f64> {
    let mut stacked = Array2::<f64>::zeros((top.nrows() + bottom.nrows(), top.ncols()));
    stacked.slice_mut(s![..top.nrows(), ..]).assign(&top);
    stacked.slice_mut(s![top.nrows().., ..]).assign(&bottom);
    stacked
}

/// `[left right]`.
fn stack_columns(left: ArrayView2<'_, f64>, right: ArrayView2<'_, f64>) -> Array2<f64> {
    let mut stacked = Array2::<f64>::zeros((left.nrows(), left.ncols() + right.ncols()));
    stacked.slice_mut(s![.., ..left.ncols()]).assign(&left);
    stacked.slice_mut(s![.., left.ncols()..]).assign(&right);
    stacked
}

/// A block's hidden pre-activation law at each retained point.
struct DiscardedLaw {
    /// `μ_ij = b_j + w_jᵀ P zᵢ`, plus the point's bias shift when one was given, `n × h`.
    means: Array2<f64>,
    /// `τ_ij = T_{v_j⊥} σ(μ_ij) = E[σ(a_j) | PZ = P zᵢ]`, `n × h`.
    unit_means: Array2<f64>,
    /// `v_j⊥ = ‖w_j − Q Qᵀ w_j‖²`, a sum of squares.
    discarded_variances: Array1<f64>,
    /// Bounds on `|v̂_j⊥ − w_jᵀ (I − P₀) w_j|`, with `P₀` the projector of the nearest orthonormal frame.
    discarded_variance_errors: Array1<f64>,
    /// Upper bounds on `‖w_j‖`.
    reader_norms: Array1<f64>,
    /// Bounds `δ_j` on `‖ê_j − (I − Q Qᵀ) w_j‖` for the residual row `ê_j = w_j − fl(fl(w_jᵀ Q) Qᵀ)`.
    residual_row_errors: Array1<f64>,
    /// `R = W Q`, `h × k`.
    coordinates: Array2<f64>,
    /// The measured bound `η ≥ ‖QᵀQ − I‖₂`.
    frame_defect: f64,
}

impl DiscardedLaw {
    /// The law at `points` (`n × d`), with `mean_shifts` (`n × h`, when given) added to every `μ_ij`: a bias that moves
    /// with the point.
    fn new(
        block: &KnownBlock,
        frame: ArrayView2<'_, f64>,
        points: ArrayView2<'_, f64>,
        mean_shifts: Option<ArrayView2<'_, f64>>,
    ) -> Result<Self, ResponseError> {
        let input_dim = block.input_dim();
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
        require_length("retained point columns", input_dim, points.ncols())?;
        require_finite("retained points", points.iter())?;
        let readers = block.readers();
        let coordinates = fast_ab(&readers, &frame);
        let discarded_variances = block.discarded_reader_variances(frame, coordinates.view());
        // First-order formation bounds, with `γ_n` the accumulation growth, `‖Q‖₂ ≤ √(1 + η)` and
        // `‖Q‖_F ≤ √(k (1 + η))`:
        // - `R̂_j = fl(w_jᵀ Q)` errs by at most `γ_d ‖Q‖_F ‖w_j‖`, and `Qᵀ` carries that error by `‖Q‖₂`;
        // - `fl(R̂_j Qᵀ)` rounds by at most `γ_k ‖Q‖_F ‖R̂_j‖`;
        // - the subtraction rounds by at most `γ₁ (‖w_j‖ + ‖p̂_j‖)`.
        // The exact row `(I − Q Qᵀ) w_j` differs from the nearest orthonormal frame's law by `‖Q Qᵀ − P₀‖₂ ≤ η`, so its
        // squared norm differs by at most `η² ‖w_j‖²`.
        let rank = frame.ncols();
        let product_growth = accumulation_growth(input_dim);
        let recombination_growth = accumulation_growth(rank);
        let subtraction_growth = accumulation_growth(1);
        let column_scale = (1.0 + defect).sqrt();
        let frame_norm = (rank as f64 * (1.0 + defect)).sqrt();
        let reader_norms: Array1<f64> = readers
            .rows()
            .into_iter()
            .map(|row| (row.dot(&row) / (1.0 - product_growth)).sqrt())
            .collect();
        let residual_row_errors = reader_norms.mapv(|reader_norm| {
            let coordinate_error = product_growth * frame_norm * reader_norm;
            let coordinate_norm = column_scale * reader_norm + coordinate_error;
            let recombination_error = recombination_growth * frame_norm * coordinate_norm;
            let product_norm = column_scale * coordinate_norm + recombination_error;
            column_scale * coordinate_error + recombination_error + subtraction_growth * (reader_norm + product_norm)
        });
        // `|v̂⊥ − ‖ê‖²| ≤ γ_d ‖ê‖²`, `|‖ê‖² − ‖e‖²| ≤ δ (2 ‖ê‖ + δ)` and `|‖e‖² − v⊥| ≤ η² ‖w‖²`.
        let discarded_variance_errors = Array1::from_shape_fn(block.width(), |unit| {
            let residual_norm = (discarded_variances[unit] / (1.0 - product_growth)).sqrt();
            let row_error = residual_row_errors[unit];
            product_growth * residual_norm * residual_norm
                + row_error * (2.0 * residual_norm + row_error)
                + defect * defect * reader_norms[unit] * reader_norms[unit]
        });
        // Row `i` of `points Q` is `zᵢᵀ Q`, so entry `(i, j)` of the product with `Rᵀ` is `w_jᵀ P zᵢ`.
        let mut means = fast_abt(&fast_ab(&points, &frame), &coordinates);
        means += &block.biases();
        if let Some(shifts) = mean_shifts {
            means += &shifts;
        }
        let activation = block.activation();
        let mut unit_means = means.clone();
        unit_means
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .try_for_each(|mut row| {
                for unit in 0..row.len() {
                    row[unit] = smoothing(activation, discarded_variances[unit], row[unit])?;
                }
                Ok::<(), ResponseError>(())
            })?;
        Ok(Self {
            means,
            unit_means,
            discarded_variances,
            discarded_variance_errors,
            reader_norms,
            residual_row_errors,
            coordinates,
            frame_defect: defect,
        })
    }

    /// The law of two blocks' units read through one frame at the same points, the first block's units first.
    fn stacked(first: &Self, second: &Self) -> Self {
        let entries =
            |top: &Array1<f64>, bottom: &Array1<f64>| -> Array1<f64> { top.iter().chain(bottom.iter()).copied().collect() };
        Self {
            means: stack_columns(first.means.view(), second.means.view()),
            unit_means: stack_columns(first.unit_means.view(), second.unit_means.view()),
            discarded_variances: entries(&first.discarded_variances, &second.discarded_variances),
            discarded_variance_errors: entries(&first.discarded_variance_errors, &second.discarded_variance_errors),
            reader_norms: entries(&first.reader_norms, &second.reader_norms),
            residual_row_errors: entries(&first.residual_row_errors, &second.residual_row_errors),
            coordinates: stack_rows(first.coordinates.view(), second.coordinates.view()),
            frame_defect: first.frame_defect.max(second.frame_defect),
        }
    }

    /// `diag(Lᵀ C R)` at each retained point (`n × q`), for `left = L` and `right = R` (each `h × q`), where `C` is the
    /// unit covariance of the module docs and `readers` (`h × d`) are the units this law was formed from.
    ///
    /// One tile of `t` units forms its rows of `Σ⊥` once for every point, as `(W_J − R_J Qᵀ) Wᵀ` (`t × h`); each point
    /// then holds one `h` row at a time. Points are summed tile by tile in order, so the value does not depend on the
    /// thread count.
    fn covariance_diagonals(
        &self,
        readers: ArrayView2<'_, f64>,
        activation: GaussianActivation,
        frame: ArrayView2<'_, f64>,
        left: ArrayView2<'_, f64>,
        right: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, ResponseError> {
        let input_dim = readers.ncols();
        let product_growth = accumulation_growth(input_dim);
        let (points, width) = self.means.dim();
        let terms = left.ncols();
        let mut diagonals = Array2::<f64>::zeros((points, terms));
        let tile = byte_balanced_row_chunk(input_dim + width, width);
        for start in (0..width).step_by(tile) {
            let end = (start + tile).min(width);
            let residual =
                &readers.slice(s![start..end, ..]) - &fast_abt(&self.coordinates.slice(s![start..end, ..]), &frame);
            let discarded_covariances = fast_abt(&residual, &readers);
            let residual_norms: Array1<f64> = residual
                .rows()
                .into_iter()
                .map(|row| (row.dot(&row) / (1.0 - product_growth)).sqrt())
                .collect();
            let contributions = (0..points)
                .into_par_iter()
                .map(|point| {
                    let mut contribution = Array1::<f64>::zeros(terms);
                    let mut covariance_row = Array1::<f64>::zeros(width);
                    for offset in 0..(end - start) {
                        let unit = start + offset;
                        for other in 0..width {
                            // `r̂⊥ = fl(ê_j · w_l)`: a `d`-term product of the residual row with the copied reader row.
                            let covariance_rounding = covariance_rounding_band(
                                &CovarianceFormation {
                                    terms: input_dim,
                                    left_norm: residual_norms[offset],
                                    right_norm: self.reader_norms[other],
                                    left_row_error: self.residual_row_errors[unit],
                                    right_row_error: 0.0,
                                    law_gap: self.frame_defect * self.reader_norms[unit] * self.reader_norms[other],
                                    variance_error_x: self.discarded_variance_errors[unit],
                                    variance_error_y: self.discarded_variance_errors[other],
                                },
                                self.discarded_variances[unit],
                                self.discarded_variances[other],
                            );
                            // Centred: `K − m_j m_l` would keep only `u (|K| + |m_j m_l|)` of a covariance that
                            // vanishes with the discarded variances (#4351).
                            covariance_row[other] = pair_covariance(
                                activation,
                                PreactivationPair {
                                    mean_x: self.means[[point, unit]],
                                    mean_y: self.means[[point, other]],
                                    variance_x: self.discarded_variances[unit],
                                    variance_y: self.discarded_variances[other],
                                    covariance: discarded_covariances[[offset, other]],
                                    covariance_rounding,
                                },
                            )?
                            .value;
                        }
                        contribution += &(&left.row(unit) * &right.t().dot(&covariance_row));
                    }
                    Ok(contribution)
                })
                .collect::<Result<Vec<Array1<f64>>, ResponseError>>()?;
            for (point, contribution) in contributions.iter().enumerate() {
                let mut target = diagonals.row_mut(point);
                target += contribution;
            }
        }
        Ok(diagonals)
    }
}

#[cfg(test)]
#[path = "compose_tests.rs"]
mod compose_tests;
