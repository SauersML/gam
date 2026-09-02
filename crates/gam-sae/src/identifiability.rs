//! SAE identifiability primitives and partial-supervision gauge fixing.
//!
//! # Object 4 — the Certificate ([`residual_gauge`])
//!
//! The partial-supervision solver above *removes* gauge freedom by aligning to
//! auxiliary supervision. The certificate answers the dual question: after a fit
//! has converged, **which gauge group is the model identified up to?** It does
//! so by running the same penalty-aware RRQR rank machinery the cross-block
//! identifiability audit uses
//! ([`gam_identifiability::audit::audit_identifiability`] /
//! [`gam_linalg::faer_ndarray::rrqr_with_permutation`]) — but on the
//! **symmetry generators** of the fitted model rather than on stacked design
//! columns.
//!
//! Each candidate symmetry of the SAE-manifold model (an isometry of an atom's
//! latent manifold, a rotation inside an ARD-equal eigenspace, a rotation of the
//! decoder output frame, an exchange of two topology-identical atoms) is
//! realised as a **tangent direction** `ξ` in the model's free-parameter space.
//! A generator is an *unpinned residual gauge freedom* iff the converged
//! objective is flat along it — i.e. `ξ` lies in the null space of the total
//! curvature operator `H = H_data + H_isometry` (data/likelihood curvature plus
//! the isometry-penalty curvature). It is *pinned* (broken by the data or the
//! isometry penalty) iff `ξ` has a component in `range(H)`.
//!
//! The RRQR supplies the pinning RANK via the same penalty-aware,
//! leverage-scaled rank decision the audit uses. Each generator's verdict,
//! however, keeps the curvature **magnitudes**: the relative curvature
//! fraction `‖R ξ̂‖² / σ_max(R)²` measures how much objective curvature the
//! unit generator carries, relative to the model's stiffest direction. A
//! generator is **unpinned** iff that fraction is within the calibrated
//! tolerance `max(`[`GENERATOR_FLAT_ENERGY_TOL`]`, lowering_error_scale)` —
//! genuinely flat up to numerical noise and up to the mean-frame lowering's
//! own resolution ([`FittedAtom::lowering_error`], #995). Anything larger
//! means the orbit costs objective, so the exact symmetry is broken and the
//! generator is **pinned** — including the *mixed* case (partly curved,
//! partly flat), where replicate fits do NOT differ by that group element
//! even though some flat directions remain nearby. Magnitudes (not span
//! membership) keep the statistic informative when `range(H)` is full-rank,
//! which production fits always are. The fraction and the calibration scale
//! are reported per generator so partial flatness stays visible instead of
//! being collapsed into the boolean.
//!
//! The whole computation is performed in the inner product carried by the fit's
//! [`gam_problem::RowMetric`]: the curvature root `R` is built
//! from the metric-whitened Jacobian, so the certificate's "computed in metric
//! X" line reads straight off [`gam_problem::RowMetric::provenance`]
//! ([`gam_problem::MetricProvenance`]) and cannot misreport —
//! there is only one metric object.
//!
//! # This module IS the empirical Terracini certificate (Theorem A)
//!
//! Read structurally, `residual_gauge` is doing exactly what Terracini's
//! theorem does for a join/secant variety: at a generic point of a join of
//! manifolds `M_1, …, M_K`, the tangent space to the join is the (direct) sum
//! of the individual tangent spaces `T M_1 ⊕ … ⊕ T M_K`, of dimension
//! `Σ_k dim T M_k = Σ_k (d_k + 1)` counting the scale/translation directions
//! each summand contributes. Here each atom's enumerated symmetry generators
//! (`atom_isometry_generators`, `equal_ard_rotation_generators`, the
//! cross-atom `frame_rotation_generators` / `atom_permutation_generators`)
//! are exactly a spanning set for that border-block tangent space, realised as
//! literal tangent directions `ξ` in the fitted model's free-parameter space —
//! this file's `param_dim()` coordinates are the ambient space Terracini's
//! generic-point argument is stated in. The stacked curvature root `R`
//! (`stacked_curvature_root`) is the empirical analogue of the Jacobian whose
//! rank Terracini's theorem predicts: its RRQR-pinned rank (see
//! `CurvatureReduction::from_model`) is the empirically REALISED tangent
//! dimension the fit's curvature can see, and the per-generator relative
//! curvature fraction is a Marchenko–Pastur-style edge test applied
//! direction-by-direction — a genuine signal direction has curvature energy
//! resolvably above the calibrated noise floor
//! ([`GENERATOR_FLAT_ENERGY_TOL`], widened by the #995 lowering-error scale for
//! directions the mean-frame compression cannot resolve), while a direction
//! indistinguishable from noise sits below it. When every enumerated tangent
//! direction the theory predicts is independently confirmed above that edge
//! (pinned), Theorem A's promise is realised concretely: identifiability STOPS
//! being an assumption imposed on the model class and becomes a CERTIFICATE
//! this specific fit carries, verdict-by-verdict, with the exact residual
//! (unpinned) subgroup named. This is why the certificate returns a full
//! [`ResidualGaugeReport`] rather than a boolean: Theorem A is a statement
//! about a decomposition of tangent directions, and the report is that
//! decomposition made data-legible.

pub mod frame_curvature;

pub use frame_curvature::{
    FrameColumnLayout, OutputBlockRootAccumulator, ResidualGaugeCurvature, StreamedFrameCurvature,
    StreamedLambdaMax, TriangularRootAccumulator, streamed_lambda_max,
};

use crate::chart_canonicalization::CanonicalChartTopology;
use crate::inference::layer_transport::TransportLadderReport;
use crate::inference::riesz::{RieszInput, SmoothFunctional, debias_with_dense_hessian};
use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, FaerSvd, default_rrqr_rank_alpha};
use gam_math::score_opt::{
    AffineRemlProfile, ScoreOptimumLocation, certified_exp_representative, certified_ln_positive,
};
use gam_problem::{MetricProvenance, RowMetric};
use gam_terms::inference::structure_evidence::StructureCertificate;
use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, s};

// At two or more independent ladders, each fit is pure and order-independent; fan
// them out unless this call is already running inside a rayon worker.

/// Smoothed column-2-norm of the decoder Jacobian.
///
/// Returns `(value, grad)` where `value = Σ_k √(Σ_d W[d,k]² + ε²) − ε`
/// scaled by `weight`, and `grad[d, k] = weight · W[d, k] / √(Σ_d W[d,k]² + ε²)`.
#[derive(Debug, Clone)]
pub struct MechanismSparsityJacobian {
    pub weight: f64,
    pub epsilon: f64,
}

impl MechanismSparsityJacobian {
    pub fn new(weight: f64, epsilon: f64) -> Result<Self, String> {
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "MechanismSparsityJacobian: weight must be finite and >0, got {weight}"
            ));
        }
        if !(epsilon.is_finite() && epsilon > 0.0) {
            return Err(format!(
                "MechanismSparsityJacobian: epsilon must be finite and >0, got {epsilon}"
            ));
        }
        Ok(Self { weight, epsilon })
    }

    /// Evaluate value and gradient on a (d_obs, k_latent) decoder weight matrix.
    pub fn value_and_grad(&self, w: ArrayView2<f64>) -> (f64, Array2<f64>) {
        let (d, k) = w.dim();
        let eps2 = self.epsilon * self.epsilon;
        let mut grad = Array2::<f64>::zeros((d, k));
        let mut value = 0.0;
        for col in 0..k {
            let mut sq = 0.0;
            for row in 0..d {
                sq += w[[row, col]] * w[[row, col]];
            }
            let denom = (sq + eps2).sqrt();
            value += denom - self.epsilon;
            let factor = self.weight / denom;
            for row in 0..d {
                grad[[row, col]] = factor * w[[row, col]];
            }
        }
        (self.weight * value, grad)
    }

    /// Diagonal of the Hessian wrt vec(W). Used as a Newton preconditioner.
    pub fn hessian_diag(&self, w: ArrayView2<f64>) -> Array2<f64> {
        let (d, k) = w.dim();
        let eps2 = self.epsilon * self.epsilon;
        let mut out = Array2::<f64>::zeros((d, k));
        for col in 0..k {
            let mut sq = 0.0;
            for row in 0..d {
                sq += w[[row, col]] * w[[row, col]];
            }
            let denom = (sq + eps2).sqrt();
            let inv = 1.0 / denom;
            let inv3 = inv * inv * inv;
            for row in 0..d {
                // ∂² / ∂W[d,k]² of √(||·||²+ε²) = 1/r − W[d,k]²/r³
                out[[row, col]] = self.weight * (inv - w[[row, col]] * w[[row, col]] * inv3);
            }
        }
        out
    }
}

/// iVAE-style auxiliary-conditional Gaussian log-prior on the latent block.
///
/// Stores per-row conditional means `μ` of shape `(n_rows, latent_dim)` and
/// scales `σ` of shape `(n_rows, latent_dim)`, where `(μ_{n,i}, σ_{n,i})` are
/// presumed evaluated by some external Smooth at the auxiliary `u_n`. The
/// negative log-prior contribution to the latent objective is
///
///   `½ Σ_n Σ_i [ ((t_{n,i} − μ_{n,i}) / σ_{n,i})²
///                + 2 log σ_{n,i} + log 2π ]`
///
/// scaled by `weight`. The gradient w.r.t. `t` is `(t − μ) / σ²` (times
/// `weight`); the gradient w.r.t. `μ` is its negative. Per-row scales make
/// this strictly more general than a fixed `N(0, I)`, which is recovered by
/// `μ ≡ 0`, `σ ≡ 1`.
#[derive(Debug, Clone)]
pub struct ConditionalPriorIvae {
    pub mean: Array2<f64>,
    pub scale: Array2<f64>,
    pub weight: f64,
}

impl ConditionalPriorIvae {
    pub fn new(mean: Array2<f64>, scale: Array2<f64>, weight: f64) -> Result<Self, String> {
        if mean.dim() != scale.dim() {
            return Err(format!(
                "ConditionalPriorIvae: mean shape {:?} != scale shape {:?}",
                mean.dim(),
                scale.dim()
            ));
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "ConditionalPriorIvae: weight must be finite and >0, got {weight}"
            ));
        }
        for &v in scale.iter() {
            if !(v.is_finite() && v > 0.0) {
                return Err(format!(
                    "ConditionalPriorIvae: every scale must be finite and >0, got {v}"
                ));
            }
        }
        for &v in mean.iter() {
            if !v.is_finite() {
                return Err("ConditionalPriorIvae: mean contains non-finite entry".to_string());
            }
        }

        // Khemakhem et al. (arXiv:2107.10098) Theorem 1 identifiability
        // precondition for the exponential-family conditional prior:
        // the auxiliary index `u` must yield 2k+1 distinct conditional
        // priors `p(t|u)` whose sufficient-statistic parameters
        // `(η_1(u), η_2(u)) = (μ(u)/σ(u)², −1/(2σ(u)²))` span a
        // 2k-dimensional set. For the diagonal Gaussian family this is
        // requires the BASELINE DIFFERENCES of the NATURAL parameters to span
        // `R^{2k}`. Raw `[μ, log σ]` is not an invertible LINEAR change of these
        // parameters and can have a different rank.
        //
        // This is the CLASSICAL precondition the certificate's per-generator
        // gauge-groupoid slice (see [`GeneratorFamily`]) generalises: Khemakhem's
        // rank/distinctness check is a single global yes/no gate on whether the
        // auxiliary-conditioned prior family is rich enough to break rotational
        // non-identifiability of the whole latent block at once, whereas the
        // residual-gauge certificate breaks the equivalent question into one
        // verdict per candidate symmetry direction of a much richer per-atom
        // manifold model. Both instruments answer the same underlying question
        // — "is there a source of asymmetry (auxiliary conditioning here, data
        // curvature / isometry penalty there) that pins this rotational
        // freedom?" — Khemakhem's at the level of the whole prior, this
        // certificate's at the level of Prop H's residual continuous
        // stabilizer for canonical-gauge transport.
        let (n_rows, latent_dim) = mean.dim();
        let needed_rows = 2 * latent_dim + 1;
        if n_rows < needed_rows {
            return Err(format!(
                "ConditionalPriorIvae: Khemakhem (arXiv:2107.10098) Theorem 1 \
                 precondition violated: need at least 2k+1 = {needed_rows} distinct \
                 auxiliary states for latent_dim k = {latent_dim}, got n_rows = {n_rows}"
            ));
        }
        let natural = {
            let mut s = Array2::<f64>::zeros((n_rows, 2 * latent_dim));
            for r in 0..n_rows {
                for c in 0..latent_dim {
                    let variance = scale[[r, c]] * scale[[r, c]];
                    s[[r, c]] = mean[[r, c]] / variance;
                    s[[r, latent_dim + c]] = -0.5 / variance;
                }
            }
            s
        };
        let mut differences = Array2::<f64>::zeros((n_rows - 1, 2 * latent_dim));
        for r in 1..n_rows {
            for c in 0..2 * latent_dim {
                differences[[r - 1, c]] = natural[[r, c]] - natural[[0, c]];
            }
        }
        let (_u, sv, _vt) = differences.svd(false, false).map_err(|e| {
            format!("ConditionalPriorIvae: SVD of natural-parameter differences failed: {e}")
        })?;
        let max_sv = sv.iter().cloned().fold(0.0_f64, f64::max);
        let tol = max_sv * (n_rows.max(2 * latent_dim) as f64) * f64::EPSILON;
        let numerical_rank = sv.iter().filter(|&&s| s > tol).count();
        let required = 2 * latent_dim;
        if numerical_rank < required {
            return Err(format!(
                "ConditionalPriorIvae: Khemakhem (arXiv:2107.10098) Theorem 1 \
                 precondition violated: baseline differences of Gaussian natural \
                 parameters [μ/σ² ‖ −1/(2σ²)] have \
                 numerical rank {numerical_rank} < 2·latent_dim = {required} \
                 (tolerance {tol:.3e}); the family `p(t|u)` does not span a \
                 2k-dimensional set of natural parameters"
            ));
        }

        Ok(Self {
            mean,
            scale,
            weight,
        })
    }

    /// Evaluate negative-log-prior value and gradient w.r.t. latent t.
    pub fn value_and_grad(&self, t: ArrayView2<f64>) -> (f64, Array2<f64>) {
        assert_eq!(
            t.dim(),
            self.mean.dim(),
            "ConditionalPriorIvae: t/mean shape mismatch"
        );
        let (n, d) = t.dim();
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let mut grad = Array2::<f64>::zeros((n, d));
        let mut value = 0.0;
        for row in 0..n {
            for col in 0..d {
                let mu = self.mean[[row, col]];
                let sigma = self.scale[[row, col]];
                let z = (t[[row, col]] - mu) / sigma;
                value += 0.5 * (z * z + 2.0 * sigma.ln() + log_2pi);
                grad[[row, col]] = self.weight * z / sigma;
            }
        }
        (self.weight * value, grad)
    }

    /// Evaluate value only — useful when only the loss is needed.
    pub fn value(&self, t: ArrayView2<f64>) -> f64 {
        self.value_and_grad(t).0
    }
}

/// Derive the iVAE auxiliary-conditional scale `σ(u)` from the auxiliary table.
///
/// Each auxiliary column is population-standardized across rows, then mapped as
/// `log σ_j(u) = log_amplitude * tanh(frequency_scale * (j + 1) * z_j)`.
/// Constant columns standardize to zero, so their derived scale is exactly one
/// and the downstream [`ConditionalPriorIvae::new`] rank check reports the
/// resulting non-identifiability instead of fabricating variation.
pub fn derive_ivae_aux_scale(
    aux: ArrayView2<f64>,
    log_amplitude: f64,
    frequency_scale: f64,
) -> Array2<f64> {
    let (n_rows, n_cols) = aux.dim();
    let mut out = Array2::<f64>::zeros((n_rows, n_cols));
    let n = n_rows as f64;
    for col in 0..n_cols {
        let mut mean = 0.0;
        for row in 0..n_rows {
            mean += aux[[row, col]];
        }
        mean /= n;

        let mut var = 0.0;
        for row in 0..n_rows {
            let centered = aux[[row, col]] - mean;
            var += centered * centered;
        }
        let std = (var / n).sqrt();
        let safe_std = if std > 0.0 { std } else { 1.0 };
        let freq = frequency_scale * (col + 1) as f64;
        for row in 0..n_rows {
            let z = (aux[[row, col]] - mean) / safe_std;
            let log_sigma = log_amplitude * (freq * z).tanh();
            out[[row, col]] = log_sigma.exp();
        }
    }
    out
}

/// Evaluate the fixed-hyperparameter Gaussian profile evidence used by the
/// identifiable-factor Torch interaction boundary.
///
/// This function deliberately evaluates one converged fit. It is not a model
/// selector and accepts no candidate array: sampled RSS/penalty surfaces do not
/// contain the analytic derivatives needed to certify a continuous optimum in
/// two log-weights. A zero residual makes the concentrated Gaussian likelihood
/// unbounded and is therefore an error rather than a floored finite score.
pub fn identifiable_factor_log_evidence(
    residual_sum_squares: f64,
    penalty: f64,
    n_obs: usize,
) -> Result<f64, String> {
    if n_obs == 0 {
        return Err("identifiable_factor_log_evidence: n_obs must be > 0".to_string());
    }
    if !(residual_sum_squares.is_finite() && residual_sum_squares > 0.0) {
        return Err(format!(
            "identifiable_factor_log_evidence: residual_sum_squares must be finite and \
             positive; got {residual_sum_squares}"
        ));
    }
    if !penalty.is_finite() {
        return Err(format!(
            "identifiable_factor_log_evidence: penalty must be finite; got {penalty}"
        ));
    }
    let observations = n_obs as f64;
    Ok(-0.5 * observations * (residual_sum_squares / observations).ln() - 0.5 * penalty)
}

/// Outcome of the continuous shared-λ ridge REML weight selection.
///
/// The REML criterion (σ² profiled out, no unpenalized fixed effect) for the
/// multi-response ridge map `A_λ = (G + λI)⁻¹ Tᵀaux` is
///
/// ```text
/// reml(λ) = nq · ln(S(λ)/(nq)) + q Σ_r ln(1 + γ_r/λ),
/// S(λ)    = ‖aux‖_F² − Σ_r m_r / (γ_r + λ),
/// ```
///
/// with `γ_r` the eigenvalues of `G = TᵀT` and `m_r` the per-eigenvector
/// signal energies and `q` the number of response columns. `Interior` carries a certified stationary minimiser
/// (`|d reml/d log λ| ≤ tol` with positive curvature); `FullShrinkage` is the
/// exact λ → ∞ boundary optimum (`A = 0`), which
/// the evidence prefers when the auxiliary signal does not support any
/// alignment (empirical-Bayes null recovery).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RidgeRemlWeight {
    Interior { lambda: f64, score: f64 },
    FullShrinkage { score: f64 },
}

impl RidgeRemlWeight {
    pub fn score(&self) -> f64 {
        match self {
            RidgeRemlWeight::Interior { score, .. } => *score,
            RidgeRemlWeight::FullShrinkage { score } => *score,
        }
    }
}

/// Continuous REML selection of the shared ridge weight λ.
///
/// Optimises `reml(λ)` (see [`RidgeRemlWeight`]) over ρ = ln λ with the
/// analytic first and second derivatives
///
/// ```text
/// S₁ = dS/dρ  = λ Σ m_r/(γ_r+λ)²
/// S₂ = d²S/dρ² = S₁ − 2λ² Σ m_r/(γ_r+λ)³
/// L₁ = dL/dρ  = −Σ γ_r/(γ_r+λ),   L₂ = λ Σ γ_r/(γ_r+λ)²
/// g  = nq·S₁/S + qL₁,             g′ = nq·(S₂S − S₁²)/S² + qL₂
/// ```
///
/// The analytic score/gradient oracle is searched over the complete
/// representable log-λ domain by adaptive stationary-interval isolation and
/// safeguarded derivative-root refinement. Every certified interior minimum
/// competes with the exact λ → ∞ boundary, which wins ties. Directions with
/// `γ_r = 0` are exact null directions of
/// `G` (`‖T v_r‖² = 0` forces `m_r = (T v_r)ᵀaux = 0` in exact arithmetic)
/// and drop from every sum.
///
/// Non-convergence or failure to certify stationarity is a typed error, never a
/// degraded answer.
pub fn ridge_reml_select_weight(
    eigvals: &[f64],
    signal_energy: &[f64],
    aux_norm_sq: f64,
    n_obs: usize,
    n_responses: usize,
) -> Result<RidgeRemlWeight, String> {
    if eigvals.len() != signal_energy.len() {
        return Err(format!(
            "ridge_reml_select_weight: eigvals len {} != signal_energy len {}",
            eigvals.len(),
            signal_energy.len()
        ));
    }
    if !(aux_norm_sq.is_finite() && aux_norm_sq > 0.0) {
        return Err(format!(
            "ridge_reml_select_weight: aux_norm_sq must be finite positive, got {aux_norm_sq}"
        ));
    }
    if n_obs == 0 {
        return Err("ridge_reml_select_weight: n_obs must be > 0".to_string());
    }
    if n_responses == 0 {
        return Err("ridge_reml_select_weight: n_responses must be > 0".to_string());
    }
    for (&g, &m) in eigvals.iter().zip(signal_energy) {
        if !g.is_finite() || !m.is_finite() || m < 0.0 {
            return Err(format!(
                "ridge_reml_select_weight: non-finite or invalid (γ={g}, m={m}) pair"
            ));
        }
    }
    let response_multiplicity = n_responses as f64;
    let scalar_observations = (n_obs as f64) * response_multiplicity;
    let gamma_max = eigvals.iter().cloned().fold(0.0_f64, f64::max);
    let spectral_scale = eigvals
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    // Exact-arithmetic null directions of G carry zero signal energy. Resolve
    // numerical rank at the eigensolver's dimension-scaled roundoff and reject
    // a materially indefinite "Gram" spectrum rather than silently dropping it.
    let rank_resolution = f64::EPSILON * eigvals.len().max(1) as f64 * spectral_scale;
    for (&g, &m) in eigvals.iter().zip(signal_energy) {
        if g < -rank_resolution {
            return Err(format!(
                "ridge_reml_select_weight: Gram spectrum is materially negative ({g})"
            ));
        }
        if g <= rank_resolution && m > rank_resolution * aux_norm_sq {
            return Err(format!(
                "ridge_reml_select_weight: numerical Gram-null direction carries signal \
                 energy {m}; sufficient statistics are inconsistent"
            ));
        }
    }
    let pairs: Vec<(f64, f64)> = eigvals
        .iter()
        .zip(signal_energy)
        .filter(|&(&g, _)| g > rank_resolution)
        // Work in u = λ/γ_max so every retained spectral abscissa lies in
        // (0, 1].  This keeps the score and its derivatives finite across the
        // complete representable log-domain even for ill-scaled Gram matrices.
        .map(|(&g, &m)| (g / gamma_max, m / gamma_max))
        .collect();
    let boundary_score_enclosure = certified_ln_positive(aux_norm_sq)
        .ok_or_else(|| {
            "ridge_reml_select_weight: could not enclose the full-shrinkage response log"
                .to_string()
        })?
        .sub(certified_ln_positive(scalar_observations).ok_or_else(|| {
            "ridge_reml_select_weight: could not enclose the observation-count log".to_string()
        })?)
        .scale(scalar_observations);
    let boundary_score = boundary_score_enclosure.lo
        + 0.5 * (boundary_score_enclosure.hi - boundary_score_enclosure.lo);
    if pairs.is_empty() {
        // G = 0: the ridge map is identically zero for every λ.
        return Ok(RidgeRemlWeight::FullShrinkage {
            score: boundary_score,
        });
    }

    // `AffineRemlProfile` already owns the certified scalar REML search:
    // analytic value/gradient/curvature jets plus outward-rounded derivative
    // enclosures on every interval.  Duplicate each spectral direction once
    // per response while retaining one pooled residual.  This makes its score
    // exactly
    //
    //   -1/2 { nq log(S/(nq)) + q sum_r log(1 + gamma_r/lambda) },
    //
    // i.e. minus one half of the criterion documented above.  It is not an
    // approximation and introduces no lattice of candidate weights.
    let repeated_modes = pairs.len().saturating_mul(n_responses);
    let mut gram_modes = Vec::with_capacity(repeated_modes);
    let penalty_modes = vec![1.0_f64; repeated_modes];
    let mut projected_rhs_squared = Vec::with_capacity(repeated_modes);
    for _ in 0..n_responses {
        for &(g, m) in &pairs {
            gram_modes.push(g);
            projected_rhs_squared.push(m / response_multiplicity);
        }
    }
    let response_energy = [aux_norm_sq];
    let profile = AffineRemlProfile::new(
        &gram_modes,
        &penalty_modes,
        &projected_rhs_squared,
        &response_energy,
        scalar_observations,
        repeated_modes,
        0.0,
    )
    .map_err(|error| format!("ridge_reml_select_weight: {error}"))?;

    // Search the complete finite log(λ/γ_max) domain. The exact λ=∞
    // empirical-Bayes null is compared separately below, so it is never
    // represented by an arbitrary large finite weight.
    let rho_lo = certified_ln_positive(f64::MIN_POSITIVE)
        .ok_or_else(|| {
            "ridge_reml_select_weight: could not enclose the finite-domain lower bound".to_string()
        })?
        .lo;
    let rho_hi = certified_ln_positive(f64::MAX / 2.0)
        .ok_or_else(|| {
            "ridge_reml_select_weight: could not enclose the finite-domain upper bound".to_string()
        })?
        .hi;
    let rho_tolerance = f64::EPSILON.sqrt();
    let search = profile
        .maximize_value_ordered(rho_lo, rho_hi, rho_tolerance)
        .map_err(|error| format!("ridge_reml_select_weight: {error}"))?;
    let optimum = search.optimum;
    let score = -2.0 * optimum.value;
    let finite_criterion = search.value_certificate.maximum.scale(-2.0);
    // Empirical-Bayes null recovery is the conservative decision: a finite
    // weight is admitted only when its exact-real criterion interval lies
    // strictly below the exact full-shrinkage interval. Overlap is numerical
    // non-identifiability, so the exact null wins without an epsilon heuristic.
    if finite_criterion.hi >= boundary_score_enclosure.lo {
        return Ok(RidgeRemlWeight::FullShrinkage {
            score: boundary_score,
        });
    }
    if search.value_certificate.maximum_excess > search.value_certificate.comparison_resolution {
        return Err(format!(
            "ridge_reml_select_weight: finite REML candidates are not globally ordered \
             (maximum excess {}, comparison resolution {})",
            search.value_certificate.maximum_excess, search.value_certificate.comparison_resolution
        ));
    }
    if search.location == ScoreOptimumLocation::LowerBoundary {
        return Err(
            "ridge_reml_select_weight: REML is unbounded at the λ → 0 interpolation \
             boundary; no converged Gaussian evidence fit exists"
                .to_string(),
        );
    }
    let ScoreOptimumLocation::Stationary(index) = search.location else {
        return Err(format!(
            "ridge_reml_select_weight: finite REML optimum is value-resolved but not an \
             isolated stationary point ({:?})",
            search.location
        ));
    };
    let stationary = search.stationary_points.get(index).ok_or_else(|| {
        "ridge_reml_select_weight: optimizer returned an invalid stationary index".to_string()
    })?;
    let kkt = profile
        .enclose(stationary.bracket.lo, stationary.bracket.hi)
        .map_err(|error| format!("ridge_reml_select_weight: {error}"))?;
    if !(kkt.derivative.contains_zero() && kkt.curvature.hi < 0.0) {
        return Err(format!(
            "ridge_reml_select_weight: exact-real interior maximum KKT certificate failed \
             on {:?}: {kkt:?}",
            stationary.bracket
        ));
    }
    let relative_lambda = certified_exp_representative(optimum.x).ok_or_else(|| {
        "ridge_reml_select_weight: could not construct the certified finite REML representative"
            .to_string()
    })?;
    let lambda = gamma_max * relative_lambda;
    if !(lambda.is_finite() && lambda > 0.0) {
        return Err(format!(
            "ridge_reml_select_weight: selected finite REML weight is not representable \
             after restoring the Gram scale ({gamma_max} * {relative_lambda})"
        ));
    }
    Ok(RidgeRemlWeight::Interior { lambda, score })
}

/// Column-centred thin-SVD scores: returns the leading `k` columns of
/// `U Σ` for the centred predictor matrix `X − mean(X, axis=0)`.
///
/// Used to seed `T_init` for the partial-supervision recipe when the
/// caller does not supply one. Pure-Rust path (faer SVD via the
/// `FaerSvd` bridge) so the seeding math lives in the same crate as the
/// gauge-fix solver.
pub fn thin_svd_scores(x: ArrayView2<f64>, k: usize) -> Result<Array2<f64>, String> {
    let (n, p) = x.dim();
    if k == 0 {
        return Ok(Array2::<f64>::zeros((n, 0)));
    }
    if k > n.min(p) {
        return Err(format!(
            "thin_svd_scores: requested {k} components but min(n={n}, p={p}) limits to {}",
            n.min(p)
        ));
    }
    let mut mean_row = Array1::<f64>::zeros(p);
    for row in 0..n {
        for col in 0..p {
            mean_row[col] += x[[row, col]];
        }
    }
    if n > 0 {
        let inv_n = 1.0 / (n as f64);
        for col in 0..p {
            mean_row[col] *= inv_n;
        }
    }
    let mut xc = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        for col in 0..p {
            xc[[row, col]] = x[[row, col]] - mean_row[col];
        }
    }
    let (u_opt, sigma, _vt_opt) = xc
        .svd(true, false)
        .map_err(|e| format!("thin_svd_scores: SVD failed: {e}"))?;
    let u = u_opt.ok_or_else(|| "thin_svd_scores: SVD did not return U".to_string())?;
    let mut out = Array2::<f64>::zeros((n, k));
    for row in 0..n {
        for col in 0..k {
            out[[row, col]] = u[[row, col]] * sigma[col];
        }
    }
    Ok(out)
}

/// Method for tying the supervised block to the auxiliary signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialSupervisionSupMethod {
    /// Orthogonal Procrustes: `min_{RᵀR=I} ‖T_sup R - aux‖_F²`.
    Procrustes,
    /// Affine least-squares pinned to `anchor_idx`.
    Anchor,
    /// Ridge map `A_λ = (TᵀT + λI)⁻¹ Tᵀaux` with REML-selected λ.
    SoftL2,
}

/// Free-block decorrelation rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialSupervisionFreeConstraint {
    /// QR-based projection onto the orthogonal complement of `col(T_sup)`.
    OrthogonalToSup,
    /// No projection.
    None,
}

/// Result of [`partial_supervision_solve`].
///
/// `alignment_score = 1 - ‖T_sup_aligned - aux‖_F² / ‖aux‖_F²` for every
/// method (1.0 = perfect, 0.0 = no better than the constant-zero predictor).
/// The fitted gauge map lives in the variant-specific fields:
///
/// * Procrustes → `map_r = R` (`d × d` orthogonal).
/// * Anchor    → `map_a = A` (`d × d`), `map_b` (`d`).
/// * SoftL2    → `map_a = A_λ` (`d × d`), `selected_weight = λ`.
#[derive(Debug, Clone)]
pub struct PartialSupervisionResult {
    pub t_supervised: Array2<f64>,
    pub t_free: Array2<f64>,
    pub alignment_score: f64,
    pub selected_weight: Option<f64>,
    pub map_r: Option<Array2<f64>>,
    pub map_a: Option<Array2<f64>>,
    pub map_b: Option<Array1<f64>>,
}

/// Library-level partial-supervision gauge-fix solver.
///
/// Solves the supervised-block alignment problem and applies the chosen
/// free-block decorrelation rule. Pure numerical linear algebra: SVD,
/// symmetric eigendecomposition (`Side::Lower`), and thin QR are routed
/// through the faer bridge in `gam_linalg::faer_ndarray`.
///
/// This is the single Rust source-of-math for the gauge-fix step; it is
/// language-agnostic so the CLI, R, and Julia bindings can reuse it
/// through their own marshaling layers.
///
/// Shape requirements:
/// * `t_sup` is `(N, d_sup)`; `aux` must equal that shape.
/// * `t_free` is `(N, d_free)` — `d_free` may be 0.
/// * `anchor_idx` is consulted only when `method == Anchor`; it must be
///   non-empty and every index must be `< N`.
pub fn partial_supervision_solve(
    t_sup: ArrayView2<f64>,
    aux: ArrayView2<f64>,
    t_free: ArrayView2<f64>,
    method: PartialSupervisionSupMethod,
    anchor_idx: &[usize],
    free_constraint: PartialSupervisionFreeConstraint,
) -> Result<PartialSupervisionResult, String> {
    let (n, d_sup) = t_sup.dim();
    if aux.dim() != (n, d_sup) {
        return Err(format!(
            "partial_supervision_solve: aux shape {:?} must equal t_sup shape ({}, {})",
            aux.dim(),
            n,
            d_sup
        ));
    }
    if t_free.nrows() != n {
        return Err(format!(
            "partial_supervision_solve: t_free has {} rows, expected {}",
            t_free.nrows(),
            n
        ));
    }
    let aux_norm_sq: f64 = aux.iter().map(|x| x * x).sum();
    if !(aux_norm_sq.is_finite() && aux_norm_sq > 0.0) {
        return Err(
            "partial_supervision_solve: aux has zero or non-finite Frobenius norm".to_string(),
        );
    }

    let mut t_sup_aligned = Array2::<f64>::zeros((n, d_sup));
    let mut map_r: Option<Array2<f64>> = None;
    let mut map_a: Option<Array2<f64>> = None;
    let mut map_b: Option<Array1<f64>> = None;
    let mut selected_weight: Option<f64> = None;

    match method {
        PartialSupervisionSupMethod::Procrustes => {
            // R = U Vᵀ where T_supᵀ aux = U Σ Vᵀ.
            let m = t_sup.t().dot(&aux);
            let (u_opt, _sigma, vt_opt) = m
                .svd(true, true)
                .map_err(|e| format!("partial_supervision_solve: Procrustes SVD failed: {e}"))?;
            let u = u_opt
                .ok_or_else(|| "partial_supervision_solve: SVD did not return U".to_string())?;
            let vt = vt_opt
                .ok_or_else(|| "partial_supervision_solve: SVD did not return Vᵀ".to_string())?;
            let r = u.dot(&vt);
            t_sup_aligned = t_sup.dot(&r);
            map_r = Some(r);
        }
        PartialSupervisionSupMethod::Anchor => {
            if anchor_idx.is_empty() {
                return Err(
                    "partial_supervision_solve: anchor method requires anchor_idx with at \
                     least one row"
                        .to_string(),
                );
            }
            for &idx in anchor_idx {
                if idx >= n {
                    return Err(format!(
                        "partial_supervision_solve: anchor index {idx} out of bounds (n={n})"
                    ));
                }
            }
            // Stack design [Ta | 1] of shape (m, d_sup+1); solve via SVD pseudo-inverse.
            let m_rows = anchor_idx.len();
            let mut design = Array2::<f64>::zeros((m_rows, d_sup + 1));
            let mut targets = Array2::<f64>::zeros((m_rows, d_sup));
            for (row_out, &row_in) in anchor_idx.iter().enumerate() {
                for c in 0..d_sup {
                    design[[row_out, c]] = t_sup[[row_in, c]];
                    targets[[row_out, c]] = aux[[row_in, c]];
                }
                design[[row_out, d_sup]] = 1.0;
            }
            let (u_opt, sigma, vt_opt) = design
                .svd(true, true)
                .map_err(|e| format!("partial_supervision_solve: Anchor SVD failed: {e}"))?;
            let u = u_opt
                .ok_or_else(|| "partial_supervision_solve: anchor SVD lacked U".to_string())?;
            let vt = vt_opt
                .ok_or_else(|| "partial_supervision_solve: anchor SVD lacked Vᵀ".to_string())?;
            // Tikhonov cutoff matches numpy.linalg.lstsq's default rcond policy.
            let leading = sigma.iter().cloned().fold(0.0_f64, f64::max);
            let cutoff = leading * f64::EPSILON * (m_rows.max(d_sup + 1) as f64);
            let rank = sigma.len();
            let ut_targets = u.t().dot(&targets);
            let mut scaled = Array2::<f64>::zeros((rank, d_sup));
            for r in 0..rank {
                let s = sigma[r];
                if s > cutoff {
                    let inv = 1.0 / s;
                    for c in 0..d_sup {
                        scaled[[r, c]] = inv * ut_targets[[r, c]];
                    }
                }
            }
            let coef = vt.t().dot(&scaled);
            let a = coef.slice(s![..d_sup, ..]).to_owned();
            let b_vec = coef.slice(s![d_sup, ..]).to_owned();
            for row in 0..n {
                for c in 0..d_sup {
                    let mut acc = b_vec[c];
                    for k in 0..d_sup {
                        acc += t_sup[[row, k]] * a[[k, c]];
                    }
                    t_sup_aligned[[row, c]] = acc;
                }
            }
            map_a = Some(a);
            map_b = Some(b_vec);
        }
        PartialSupervisionSupMethod::SoftL2 => {
            // Symmetric eigendecomposition of G = T_supᵀ T_sup.
            let g = t_sup.t().dot(&t_sup);
            let (eigvals, eigvecs) = g
                .eigh(Side::Lower)
                .map_err(|e| format!("partial_supervision_solve: eigh on Gram failed: {e}"))?;
            let rhs = t_sup.t().dot(&aux);
            let ut_aux = eigvecs.t().dot(&rhs);
            // Per-eigenvector signal energy m_r = ‖row_r(Vᵀ Tᵀaux)‖²; the
            // multi-response RSS at weight λ is then
            //   S(λ) = ‖aux‖_F² − Σ_r m_r/(γ_r+λ)
            // with γ_r the eigenvalues of G = TᵀT (`eigvals`).
            let m_row: Array1<f64> = Array1::from_vec(
                (0..d_sup)
                    .map(|r| (0..d_sup).map(|c| ut_aux[[r, c]] * ut_aux[[r, c]]).sum())
                    .collect(),
            );
            // Select λ by REML, never GCV. The ridge map is the linear mixed
            // model aux_j = T β_j + ε with β_j ~ N(0, σ²/λ I), ε ~ N(0, σ² I)
            // applied to each of the d columns sharing λ. The map carries no
            // unpenalized fixed effect, so REML coincides with the marginal
            // likelihood, whose profile (σ² concentrated out) criterion to
            // MINIMIZE is
            //   reml(λ) = nd·log(S(λ)/(nd)) + d·Σ_r log(1 + γ_r/λ),
            // the exact analogue of the smoothing-parameter REML used
            // everywhere else in gam.
            let selection = ridge_reml_select_weight(
                eigvals.as_slice().ok_or_else(|| {
                    "partial_supervision_solve: eigenspectrum is not contiguous".to_string()
                })?,
                m_row.as_slice().ok_or_else(|| {
                    "partial_supervision_solve: signal energies are not contiguous".to_string()
                })?,
                aux_norm_sq,
                n,
                d_sup,
            )?;
            match selection {
                RidgeRemlWeight::Interior { lambda, .. } => {
                    // Build A_λ = (G + λI)⁻¹ Tᵀaux at the certified stationary
                    // REML weight.
                    let denom: Array1<f64> = eigvals.mapv(|v| v + lambda);
                    let mut a_eig = Array2::<f64>::zeros((d_sup, d_sup));
                    for r in 0..d_sup {
                        for c in 0..d_sup {
                            a_eig[[r, c]] = ut_aux[[r, c]] / denom[r];
                        }
                    }
                    let best_a = eigvecs.dot(&a_eig);
                    t_sup_aligned = t_sup.dot(&best_a);
                    map_a = Some(best_a);
                    selected_weight = Some(lambda);
                }
                RidgeRemlWeight::FullShrinkage { .. } => {
                    // Exact empirical-Bayes null boundary: λ = ∞ and A = 0.
                    // IEEE infinity is the faithful scalar representation of that
                    // boundary in the public result; no finite proxy is substituted.
                    map_a = Some(Array2::<f64>::zeros((d_sup, d_sup)));
                    selected_weight = Some(f64::INFINITY);
                }
            }
        }
    }

    // Single source of truth for alignment_score.
    let mut sq_resid = 0.0_f64;
    for row in 0..n {
        for c in 0..d_sup {
            let r = t_sup_aligned[[row, c]] - aux[[row, c]];
            sq_resid += r * r;
        }
    }
    let alignment_score = 1.0 - sq_resid / aux_norm_sq;

    let t_free_out = match free_constraint {
        PartialSupervisionFreeConstraint::None => t_free.to_owned(),
        PartialSupervisionFreeConstraint::OrthogonalToSup => {
            if t_sup_aligned.ncols() == 0 || t_free.ncols() == 0 {
                t_free.to_owned()
            } else {
                let (u_opt, singular, _vt) = t_sup_aligned
                    .svd(true, false)
                    .map_err(|e| format!("partial_supervision_solve: SVD on T_sup failed: {e}"))?;
                let u = u_opt.ok_or_else(|| {
                    "partial_supervision_solve: SVD did not return supervised left vectors"
                        .to_string()
                })?;
                let sigma_max = singular.iter().copied().fold(0.0_f64, f64::max);
                let tol = sigma_max
                    * t_sup_aligned.nrows().max(t_sup_aligned.ncols()) as f64
                    * f64::EPSILON;
                let rank = singular.iter().filter(|&&value| value > tol).count();
                if rank == 0 {
                    return Ok(PartialSupervisionResult {
                        t_supervised: t_sup_aligned,
                        t_free: t_free.to_owned(),
                        alignment_score,
                        selected_weight,
                        map_r,
                        map_a,
                        map_b,
                    });
                }
                let q = u.slice(s![.., 0..rank]);
                let qt_free = q.t().dot(&t_free);
                let proj = q.dot(&qt_free);
                let mut out = t_free.to_owned();
                out -= &proj;
                out
            }
        }
    };

    Ok(PartialSupervisionResult {
        t_supervised: t_sup_aligned,
        t_free: t_free_out,
        alignment_score,
        selected_weight,
        map_r,
        map_a,
        map_b,
    })
}

// ============================================================================
// Object 4 — the Certificate: `residual_gauge()`
// ============================================================================

/// The latent-manifold topology of one fitted atom, as far as the certificate
/// needs it to enumerate the atom's isometry-group generators. This mirrors the
/// user-facing [`crate::manifold::SaeAtomBasisKind`] choice but
/// carries only what is required to build `Isom(M_k)` tangent directions, so the
/// certificate is decoupled from the full `SaeManifoldAtom` machinery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtomTopology {
    /// `S¹` (periodic 1-D). `Isom(S¹) = O(2)`: a single continuous rotation
    /// generator (shift of the circular coordinate) plus a reflection.
    Circle,
    /// `S²` (intrinsic sphere chart). `Isom(S²) = O(3)`: three rotation
    /// generators (so(3) basis) plus the antipodal/reflection component.
    Sphere,
    /// `RP² = S²/{u ~ -u}`. Its connected isometry group is `SO(3)` acting
    /// through the exact Killing fields on the spherical cover; conformal
    /// sphere boosts do not descend through the antipodal quotient.
    ProjectivePlane,
    /// `Tᵈ` (product of `latent_dim` circles). `Isom` contains the `d`
    /// independent circle shifts (a maximal torus of rotations).
    Torus { latent_dim: usize },
    /// Flat Klein bottle. Only translation of the first cover-circle axis is
    /// continuous; the second axis contributes a discrete deck `Z2`, not a
    /// second torus shift or a frame rotation.
    KleinBottle,
    /// A `latent_dim`-dimensional Euclidean patch / Duchon patch. Its connected
    /// isometry group `SE(d)` is generated by `d` translations and
    /// `d(d−1)/2` rotations of the latent coordinate frame.
    EuclideanPatch { latent_dim: usize },
}

impl AtomTopology {
    /// Intrinsic latent dimensionality of the atom's manifold.
    fn latent_dim(&self) -> usize {
        match self {
            AtomTopology::Circle => 1,
            AtomTopology::Sphere => 2,
            AtomTopology::ProjectivePlane => 2,
            AtomTopology::Torus { latent_dim } => *latent_dim,
            AtomTopology::KleinBottle => 2,
            AtomTopology::EuclideanPatch { latent_dim } => *latent_dim,
        }
    }
}

/// One fitted atom as the certificate sees it.
///
/// `frame` is the fitted decoder frame whose columns the isometry generators
/// rotate: an `(output_dim, latent_dim)` matrix whose column `a` is the fitted
/// image of latent axis `a` in output space (e.g. the decoder Jacobian columns
/// at the atom's centroid, or the leading decoder directions). The isometry
/// generators of `Isom(M_k)` act on these columns; the certificate lifts that
/// action to a tangent direction on the flattened decoder frame.
#[derive(Debug, Clone)]
pub struct FittedAtom {
    pub name: String,
    pub topology: AtomTopology,
    /// `(output_dim, latent_dim)` fitted decoder frame.
    pub frame: Array2<f64>,
    /// ARD prior variances (one per latent axis of this atom), used to detect
    /// equal-ARD eigenspaces inside which a rotation is unpinned by the prior.
    /// `None` ⇒ no ARD prior on this atom (every within-frame rotation is then
    /// a candidate generator, pinned-or-not decided solely by the data + the
    /// isometry penalty).
    pub ard_variances: Option<Array1<f64>>,
    /// **Lowering-error scale** (#995), in `[0, 1]`: the mass-weighted relative
    /// dispersion of the atom's per-row decoder tangents around the mean
    /// `frame` the certificate compresses them into,
    /// `Σ_n a_n Σ_ax ‖t_ax(n) − frame[:,ax]‖² / Σ_n a_n Σ_ax ‖t_ax(n)‖²`.
    ///
    /// `0` ⇒ the frame represents every row exactly (hand-built fixtures, flat
    /// decoders) and the certificate's verdicts within this atom are at full
    /// resolution. Values toward `1` ⇒ a curved decoder whose tangent field
    /// disperses strongly (e.g. a full circle, whose tangents average to ≈ 0):
    /// the mean-frame lowering then cannot distinguish gauge motion from
    /// genuine curvature, so the verdict tolerance for generators touching
    /// this atom is *calibrated up to this scale* — the certificate refuses to
    /// claim a pin it cannot resolve, the same honesty contract as the
    /// `diffeomorphism-unpinned` escalation.
    pub lowering_error: f64,
    /// #1019/#2022 stage 1: `true` when the atom's `d = 1` latent chart is pinned
    /// to its arc-length (unit-speed) canonical representative — as of #2022
    /// enforced IN-LOOP (t is arc-length at every assembled iterate), so this flag
    /// means unit-speed was maintained throughout the fit, not applied once
    /// post-fit. #1019 stage 2: `true` as well when a `d = 2` torus atom's chart
    /// was pinned post-fit to the minimum-isometry-defect flow representative, in which
    /// case the residual chart freedom is `Isom(T², flat) = U(1)² ⋊ D₄`. The
    /// certificate then records that this atom's continuous chart
    /// (reparameterization) freedom is **pinned by canonicalization** — a
    /// provenance distinct from curvature/penalty pinning
    /// ([`VerdictProvenance::PinnedByCanonicalization`]) — and that the
    /// residual chart freedom is the finite isometry group of the reference
    /// manifold for `d = 1` charts: rotation + reflection (`O(2)`) on the
    /// circle, reflection + translation on the interval.
    pub chart_canonicalized: bool,
    /// Per-atom inner-decoder-smooth byproducts harvested at fit time, the
    /// single source the post-PIRLS atom inference reports
    /// ([`AtomFunctionalReport`] #1097, [`AtomSmoothSignificance`] #1103)
    /// consume in [`dictionary_report`].
    ///
    /// The certificate path that builds `FittedSaeManifold` does so *without* a
    /// fit harness in scope, so it leaves this `None`; callers that own the
    /// fitted term attach it through [`FittedAtom::with_inner_fit`] (the term
    /// builder fills it from the live per-atom basis, decoder, assignment mass,
    /// and smoothness Gram). When `None`, both reports below are `None`: the
    /// genuine prerequisite — the post-fit inner-smooth design, penalized
    /// Hessian, and row scores — is simply not present on a bare
    /// certificate-only `FittedSaeManifold`.
    pub inner_fit: Option<AtomInnerFit>,
}

/// The fitted per-atom inner-decoder smooth, captured once at fit time so the
/// post-PIRLS atom-inference reports reuse the *same* design, penalized Hessian,
/// and per-row scores the identifiability certificate's curvature sees.
///
/// The SAE decoder reconstructs `Z_i ≈ Σ_k a_ik Φ_k(t_ik) B_k`. Holding all
/// other atoms and the assignment fixed at the fitted optimum, atom `k`'s own
/// contribution along a single output channel `j` is the Gaussian-identity
/// penalized smooth `a_ik · Φ_k(t_ik)ᵀ β_{k,j}` with roughness penalty `S_k`,
/// Gauss–Newton observation weight `w_i = a_ik²` (the assignment mass enters the
/// channel linearly, so the normal-equation weight is its square), and
/// dispersion the fitted reconstruction dispersion. That is an ordinary
/// penalized WLS smooth — exactly what [`crate::inference::riesz`],
/// [`gam_terms::inference::lawley`], and the κ-profile machinery consume. The
/// channel `j` is the atom's dominant decoder output direction (largest column
/// norm of `B_k`), i.e. the channel that carries the atom's signal.
#[derive(Debug, Clone)]
pub struct AtomInnerFit {
    /// `Φ_k` evaluated on the atom's active rows, `(n_active, M_k)`. The inner
    /// GAM smooth design. Column 0 is the constant/intercept basis column.
    pub design: Array2<f64>,
    /// `∂Φ_k/∂t` along the atom's leading latent axis on the active rows,
    /// `(n_active, M_k)`: the derivative design the average-derivative
    /// functional integrates.
    pub derivative_design: Array2<f64>,
    /// The fitted decoder coefficients for the captured output channel,
    /// `β_{k,j} ∈ ℝ^{M_k}`.
    pub beta: Array1<f64>,
    /// The atom roughness Gram `S_k`, `(M_k, M_k)`.
    pub penalty: Array2<f64>,
    /// The penalized Hessian `H = ΦᵀWΦ + S_k` at the fitted state, `(M_k, M_k)`.
    pub penalized_hessian: Array2<f64>,
    /// Per-row Gaussian-identity scores `s_i = ∂nll_i/∂β = −w_i r_i Φ_i / φ`,
    /// `(n_active, M_k)`, on the captured channel.
    pub row_scores: Array2<f64>,
    /// Per-row Gauss–Newton weights `w_i = a_ik²` on the captured channel.
    pub weights: Array1<f64>,
    /// Fitted reconstruction dispersion `φ` (Gaussian σ²).
    pub dispersion: f64,
    /// Design row at the latent peak `t_peak` (largest fitted `|g_k|`).
    pub peak_design_row: Array1<f64>,
    /// Design row at the latent mode `t_mode` (largest assignment mass).
    pub mode_design_row: Array1<f64>,
}

/// Descriptive penalty-debiased POINT summaries of one fitted atom's decoder
/// curve (#1097, narrowed under #1115). Each field is a scalar functional of the
/// atom's inner smooth `g_k(t)`, reported as a plug-in value and a one-step
/// penalty-debiased value (the regularization bias relative to the conditional
/// target is removed through the atom fit's penalized Hessian). No standard
/// error and no confidence interval are reported — by design (see below).
///
/// # Why these carry NO coverage claim (#1115)
///
/// Conditional on the fitted latent coordinates `t̂` and assignment `â`, each
/// functional is an ordinary linear functional of the penalized-WLS coefficients
/// `β` with a well-defined *conditional* population value, and one-step debiasing
/// validly removes the penalty bias for that conditional target. The point
/// estimates are therefore meaningful. A *standard error*, however, would only be
/// honest if `t̂` and `â` were fixed/known. They are not: they are **generated
/// regressors** estimated from the very activations that also form the response
/// `Z`, so `Z` enters both the design (via `t̂(Z), â(Z)`) and the response. An
/// influence-function SE built from the β-only Hessian and row scores carries no
/// `∂t̂/∂Z` / `∂â/∂Z` channel — exactly the generated-regressor correction the
/// marginal-slope family (#461 Stage 2) is *defined* by — so it omits a
/// first-order variance term and is generally anti-conservative. Rather than ship
/// an SE/CI that silently under-covers, this report exposes only the debiased
/// point summaries; a coverage-valid interval would require either freezing the
/// dictionary on a held-out split or propagating the generated-regressor
/// Jacobian, neither of which the fixed inner-fit snapshot supports.
#[derive(Debug, Clone)]
pub struct AtomFunctionalReport {
    /// `g(t_peak) − g(t_mode)`: the peak-vs-baseline contrast of the fitted
    /// decoder, penalty-debiased through the inner-fit Hessian. Point summary
    /// only (no coverage claim — see the type doc).
    pub peak_contrast: Option<AtomFunctionalEstimate>,
    /// `E_data[g(t_i)]`: the data-averaged decoder value over the atom's active
    /// rows, penalty-debiased. Point summary only.
    pub average_value: Option<AtomFunctionalEstimate>,
    /// `E_data[∂g/∂t]` along the atom's leading latent axis: how much the fitted
    /// decoder curve varies across the data distribution, **conditional on the
    /// fit**. A descriptive variation measure of the fitted curve, NOT a
    /// population "marginal slope" (the latent coordinate is itself a fitted,
    /// generated regressor). Point summary only.
    ///
    /// Despite the historical `_norm` suffix this is the **signed** mass-weighted
    /// mean derivative `E_data[∂g/∂t]` over the single leading axis, not a
    /// magnitude — it can be negative, and a value near 0 means the average slope
    /// cancels (a symmetric bump), not that the curve is flat. Use
    /// [`AtomSmoothSignificance::log_e_nonconstant`] for an honest non-constancy
    /// test; this field only describes the average local slope.
    pub decoder_variation_norm: Option<AtomFunctionalEstimate>,
}

/// One atom decoder-functional point summary: the plug-in value and the one-step
/// penalty-debiased value, with the removed penalty bias. Deliberately carries
/// NO standard error / confidence interval — the conditional-on-generated-
/// regressors variance channel is unmodelled, so any SE would under-cover
/// (#1115). Use [`AtomSmoothSignificance`] for an honest any-n-valid structure
/// test instead.
#[derive(Debug, Clone, Copy)]
pub struct AtomFunctionalEstimate {
    /// The raw plug-in functional value `θ̂ = g·β̂`.
    pub theta_plugin: f64,
    /// The one-step penalty-debiased value `θ̂ − bias`, removing the
    /// regularization bias relative to the conditional target.
    pub theta_onestep: f64,
    /// The removed penalty bias `(H⁻¹ g)·(Sβ̂)`.
    pub penalty_bias: f64,
}

/// Any-n-valid structure evidence that one atom's inner smooth `h_k(t)` is
/// genuinely non-constant (#1103): the same split-likelihood-ratio e-value the
/// atom-birth gate uses ([`gam_terms::inference::structure_evidence`]), under the
/// null H0 = "the atom's decoder curve is constant in its latent coordinate".
///
/// This replaces the earlier Lawley–Bartlett-corrected χ² test. That correction
/// was a category error here: the penalized smooth's null is effectively
/// rank ≈ n, the first-order χ² is the wrong reference entirely, and an O(1/n)
/// Bartlett factor (whose own stated size shift is ≈0.15%, flipping no admit/
/// demote decision) does not rescue it. The split-LRT e-value is finite-sample
/// valid with NO regularity conditions — exactly the instrument for "does this
/// atom earn a latent dimension".
#[derive(Debug, Clone)]
pub struct AtomSmoothSignificance {
    /// `log E` for "the atom's smooth is non-constant" (null = constant). A
    /// universal-inference split-likelihood-ratio e-value: `E_{H0}[E] ≤ 1`
    /// exactly, so `E ≥ 1/α` certifies the non-constant alternative at level α,
    /// at any data-dependent stopping time. `None` when the split is degenerate
    /// (too few active rows / a fold with no curvature column).
    pub log_e_nonconstant: Option<f64>,
}

/// The post-PIRLS inference reports for one atom, paired by atom index.
///
/// Two reports survive #1115: the descriptive penalty-debiased point summaries
/// of the fitted decoder curve ([`AtomFunctionalReport`], no coverage claim) and
/// the any-n-valid split-LRT smooth-structure e-value ([`AtomSmoothSignificance`],
/// a genuine finite-sample-valid test). The #1099 per-atom curvature *confidence
/// interval* was removed: its target (a sup-norm extrinsic-curvature BOUND read
/// off the fitted decoder) is not an estimand with a profiled criterion, and its
/// delta-method SE conditioned on the generated latent coordinates as if known.
/// The plug-in curvature point estimate itself survives — as the per-atom
/// `kappa_hat` entries of
/// [`crate::manifold::CertificateInputs::per_atom_kappa_hat`] (the
/// #1008 empirical curved-dictionary report, surfaced to Python as
/// `ManifoldSAE.curvature_report`), the single source of truth for the bound.
/// It is deliberately *not* duplicated onto this report: a descriptive geometry
/// bound is a property of the fitted decoder frames, not of the post-PIRLS
/// inner-smooth inference snapshot this type carries.
#[derive(Debug, Clone)]
pub struct AtomInferenceReport {
    pub atom_index: usize,
    pub atom_name: String,
    pub functionals: Option<AtomFunctionalReport>,
    pub smooth_significance: Option<AtomSmoothSignificance>,
}

/// The fitted SAE-manifold model the certificate consumes.
///
/// Self-contained on purpose: it carries exactly the objects the residual-gauge
/// computation needs — the atoms (with topology + fitted frames + ARD), the
/// curvature/Jacobian row-blocks that pin directions, and the one
/// [`RowMetric`] whose provenance the report reads. The flattened free-parameter
/// vector the generators live in is `vec(frame_0) ⊕ vec(frame_1) ⊕ …` in atom
/// order; `param_dim()` is its length.
pub struct FittedSaeManifold {
    pub atoms: Vec<FittedAtom>,
    /// Per-row decoder Jacobian blocks `J_n ∈ ℝ^{p × param_dim}` flattened
    /// row-major (`J_n[i, c] = jacobian_rows[n][i * param_dim + c]`), one entry
    /// per metric row. These are the directions the *data* gives cost to; the
    /// certificate whitens them through [`RowMetric`] and orthonormalizes to
    /// obtain the data part of the pinning span `range(H_data)`.
    pub jacobian_rows: Vec<Vec<f64>>,
    /// The isometry-penalty curvature root `R ∈ ℝ^{r × param_dim}` (so the
    /// penalty Hessian is `RᵀR`). Its row space is `range(H_isometry)` — the
    /// directions the isometry pin gives cost to. Empty (`0 × param_dim`) when
    /// the isometry pin is inactive, which is exactly the condition that
    /// escalates the verdict to `diffeomorphism-unpinned`.
    pub isometry_penalty_root: Array2<f64>,
    /// The single provenance-carrying per-row inner product. Read for the
    /// report's "computed in metric X" line and used to whiten the Jacobian
    /// rows so the rank decision happens in the fit's actual metric.
    pub metric: RowMetric,
}

impl FittedSaeManifold {
    /// Total flattened free-parameter dimension `Σ_k output_dim_k · latent_dim_k`
    /// (the decoder-frame coordinates the generators are tangent directions in).
    pub fn param_dim(&self) -> usize {
        self.atoms.iter().map(|a| a.frame.len()).sum()
    }

    /// Column offset of atom `k`'s flattened frame inside the joint parameter
    /// vector.
    fn atom_offset(&self, k: usize) -> usize {
        self.atoms[..k].iter().map(|a| a.frame.len()).sum()
    }
}

/// Which symmetry family a generator belongs to. Carried per-generator so the
/// report names the group the residual freedom (or pin) lives in.
///
/// # This enumeration is a slice of the gauge groupoid
///
/// The full model-class gauge groupoid has one object per fitted model and one
/// morphism per way of relabelling it (isometries of each atom, ARD-tied
/// rotations, output-frame rotations, atom exchanges, chart reparameterizations)
/// without changing what it reconstructs. `residual_gauge` cannot certify the
/// whole groupoid abstractly — it certifies, per generator, whether *this
/// specific converged fit* sits on a fixed point of that morphism (pinned) or
/// can slide along its orbit (unpinned). That per-generator pinned/unpinned
/// verdict ([`VerdictProvenance`] records which test decided it) is exactly
/// slicing the groupoid at this fit: the surviving unpinned generators are the
/// isotropy/stabilizer subgroup — the largest group under which this fit is
/// literally undetermined by the data + penalties. Two replicate fits carry
/// "the same identification" iff they slice the groupoid the same way, which
/// is exactly what [`ResidualGaugeReport::group_signature`] compares.
///
/// The surviving continuous linear stabilizer this slice exposes is the
/// uniqueness precondition (Prop H) for transporting a fit to a canonical
/// gauge: [`crate::inference::layer_transport::transport_ladder`] and the
/// chart-canonicalization machinery in [`crate::chart_canonicalization`] can
/// only pick a single canonical representative per atom when the residual
/// stabilizer is the *expected*, finite one (see [`VerdictProvenance::PinnedByCanonicalization`]
/// below) — a residual freedom the certificate reports as unpinned here is
/// precisely a direction along which "the" canonical transport is not unique,
/// so binding one fit's canonical frame to another's (transport) is only
/// honest once this slice is known. This generalises the classical
/// finite-dimensional identifiability precondition of Khemakhem et al.
/// (iVAE, arXiv:2107.10098 Theorem 1 — see [`ConditionalPriorIvae::new`]):
/// where Khemakhem's condition is a single global rank/distinctness check on
/// an auxiliary-conditional prior that either holds or fails for the whole
/// model, this groupoid slice is the same idea made *local and per-generator*
/// — instead of one yes/no precondition, each candidate symmetry direction of
/// the richer manifold-atom model gets its own verdict, so "identifiable up
/// to a named residual group" replaces "identifiable" as the deliverable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratorFamily {
    /// A generator of `Isom(M_k)` for a single atom (frame rotation/reflection
    /// realising the atom's own manifold isometry).
    IsomAtom,
    /// A rotation inside an ARD-equal eigenspace (the ARD prior cannot
    /// distinguish the two axes, so the prior does not pin this rotation).
    EqualArdRotation,
    /// A rotation of the global decoder output frame `O(output_dim)`.
    FrameRotation,
    /// An exchange of two topology-identical atoms (`Sym(F)` permutation, built
    /// as the antisymmetric transposition direction).
    AtomPermutation,
    /// The continuous chart (reparameterization) freedom `Diff(M_k)` of one
    /// `d = 1` atom (arc-length canonicalization) or `d = 2` torus atom
    /// (isometry-flow canonicalization, #1019 stage 2). Always reported
    /// **pinned** with
    /// [`VerdictProvenance::PinnedByCanonicalization`]; the verdict's
    /// description names the surviving residual group (rotation + reflection
    /// on `S¹`, reflection + translation on the interval, or `Isom(T², flat) =
    /// U(1)² ⋊ D₄` for a `d = 2` torus).
    ChartReparameterization,
}

impl GeneratorFamily {
    fn label(self) -> &'static str {
        match self {
            GeneratorFamily::IsomAtom => "Isom(M_k)",
            GeneratorFamily::EqualArdRotation => "equal-ARD rotation",
            GeneratorFamily::FrameRotation => "frame rotation O(output_dim)",
            GeneratorFamily::AtomPermutation => "Sym(F) atom permutation",
            GeneratorFamily::ChartReparameterization => "Diff(M_k) chart reparameterization",
        }
    }
}

/// How a generator's pinned/unpinned verdict was decided. Carried
/// per-generator so the report distinguishes a chart fixed **by convention**
/// (the #1019 post-fit arc-length canonicalization — an exact, image-frozen
/// representative choice) from a direction pinned **by curvature** (data or
/// the isometry penalty giving the orbit genuine objective cost).
///
/// This is the "how" half of the gauge-groupoid slice described on
/// [`GeneratorFamily`]: `CurvatureTest` is the empirical Terracini/MP-edge
/// measurement (does the fit's own curvature resolve this direction),
/// `PinnedByCanonicalization` is a *chosen* section of the groupoid (a
/// specific representative picked once, by convention, rather than measured)
/// — the distinction the report must keep honest so a canonicalization
/// choice is never mistaken for an emergent identifiability result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerdictProvenance {
    /// Decided by the relative-curvature flatness test against the stacked
    /// pinning root (data + isometry penalty, in the fit's metric) — the
    /// historical path for every enumerated generator.
    CurvatureTest,
    /// Pinned by the post-fit arc-length chart canonicalization (#1019) or the
    /// `d = 2` torus isometry-flow canonicalization (#1019 stage 2): the atom's
    /// chart is the selected representative of its `Diff(M)` orbit, so the
    /// continuous reparameterization freedom is fixed by construction — no
    /// curvature was (or needed to be) measured. Distinct from penalty-pinning
    /// on purpose: the certificate must not claim the objective resists chart
    /// motion when it is the canonicalization that removed it.
    PinnedByCanonicalization,
}

/// Noise floor for the per-generator flatness verdict: a generator is
/// certified **unpinned** iff its relative curvature fraction
/// `‖R ξ̂‖² / σ_max(R)²` (curvature along the unit generator, relative to the
/// stiffest direction of the stacked curvature root `R`) is at or below the
/// verdict tolerance `max(GENERATOR_FLAT_ENERGY_TOL, lowering_error_scale)`.
///
/// An exact residual symmetry of the converged objective has fraction 0 up to
/// roundoff; any genuinely curved component — however partial — means the
/// orbit costs objective and the exact group element is broken, so a *mixed*
/// generator (e.g. a frame rotation the anisotropic output-Fisher isometry pin
/// gives partial curvature, the #980 Theorem-2 situation) must be reported
/// pinned, never as a surviving freedom. The `lowering_error_scale` arm of the
/// tolerance is the #995 calibration: curvature attributable to the mean-frame
/// compression of a curved decoder must not be read as a pin.
pub const GENERATOR_FLAT_ENERGY_TOL: f64 = 1.0e-3;

/// One enumerated symmetry generator and the certificate's verdict on it.
#[derive(Debug, Clone)]
pub struct GeneratorVerdict {
    /// Which symmetry family this generator realises.
    pub family: GeneratorFamily,
    /// Human-readable description (which atom(s) / axes it acts on).
    pub description: String,
    /// `true` ⇒ the converged objective is flat along this generator
    /// (`ξ ∈ ker(H)`): a genuine residual gauge freedom the data + isometry
    /// penalty leave unbroken. `false` ⇒ the generator is pinned — the data or
    /// the isometry penalty gives it curvature (a pinned-energy fraction above
    /// [`GENERATOR_FLAT_ENERGY_TOL`]).
    pub unpinned: bool,
    /// `‖ξ‖₂` of the realised tangent direction (0 ⇒ the generator was
    /// structurally trivial — e.g. a rotation of a rank-deficient frame — and
    /// is reported as pinned/absent, never as a spurious freedom).
    pub generator_norm: f64,
    /// `‖R ξ̂‖² / σ_max(R)²` ∈ [0, 1]: curvature along the unit generator,
    /// relative to the stiffest direction of the stacked curvature root `R`
    /// (data + isometry penalty, in the metric). `0` ⇒ exactly flat, `1` ⇒ as
    /// stiff as the stiffest direction; strictly-interior values are the
    /// *mixed* regime — partial curvature that breaks the exact symmetry
    /// (verdict pinned when above the tolerance) while leaving nearby flat
    /// directions, kept visible here rather than collapsed into the boolean.
    /// Relative-to-σ_max (not span membership) so the statistic stays
    /// informative when the pinning span is full-rank, which production fits
    /// always are. Structurally trivial generators (zero norm) report `1.0`.
    pub pinned_energy_fraction: f64,
    /// The #995 lowering-error arm of this generator's verdict tolerance: the
    /// largest [`FittedAtom::lowering_error`] over the atoms the generator
    /// touches (its own atom for within-atom families, the exchanged pair for
    /// permutations, all atoms for global output-frame rotations). The verdict
    /// is `unpinned ⇔ pinned_energy_fraction ≤
    /// max(GENERATOR_FLAT_ENERGY_TOL, lowering_error_scale)` — curvature the
    /// mean-frame compression cannot distinguish from gauge motion is never
    /// read as a pin.
    pub lowering_error_scale: f64,
    /// How this verdict was decided: by the curvature flatness test, or
    /// pinned by the #1019 post-fit arc-length chart canonicalization
    /// (see [`VerdictProvenance`]).
    pub provenance: VerdictProvenance,
}

/// The #972 decoder-frame **inner-rotation gauge**, enumerated for the
/// certificate.
///
/// A frame-factored atom `B_k = U_k C_k` is *exactly* invariant under
/// `U_k → U_k R`, `C_k → Rᵀ C_k` for any `R ∈ O(r_k)`: the reconstruction,
/// the likelihood, the penalty — every objective term — sees only the
/// product. Unlike the latent-isometry / ARD-rotation / permutation
/// generators, this freedom is therefore **not** a candidate to be pinned by
/// data or penalty curvature (its orbit direction is identically zero in
/// function space), so running it through the pinning-span test would be a
/// category error: it would always come back "unpinned" and pollute the
/// verdict list with freedoms the parameterization already handles. The
/// honest certificate treatment is what this struct is: *enumerate* the
/// group and its dimension `Σ_k r_k(r_k−1)/2`, and record how it is fixed —
/// by the canonical orientation gauge
/// ([`crate::manifold::GrassmannFrame`]'s SVD-ordered
/// representative), which picks one point per `O(r_k)` orbit for
/// serialization/comparison stability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameInnerRotationGauge {
    /// Active frame rank `r_k` per frame-factored atom (atoms on the full-`B`
    /// path contribute no entry).
    pub per_atom_ranks: Vec<usize>,
    /// Total group dimension `Σ_k r_k (r_k − 1) / 2` (`dim O(r) = r(r−1)/2`).
    pub dim: usize,
}

/// What the certificate's reported `pinning_rank` is a rank OF — a property of
/// the MEASUREMENT, declared rather than inferred (#2757).
///
/// The pinning rank is the one thing the certificate reads off the curvature
/// that no verdict consumes: `ξᵀHξ` is every verdict's numerator and `λ_max(H)`
/// is its denominator, and both are streamable, but a rank over the whole
/// parameter space is a full-spectrum question and costs `param_dim²` scalars to
/// ask from any side. On the branch where `H` has no structure — a per-row
/// metric that couples output coordinates, at a width where a `param_dim`-square
/// object is `34 GiB` — the certificate therefore does not ask it, and says so
/// here instead of reporting a number it did not measure.
///
/// The two supports are not two qualities of the same measurement. They are
/// different measurements, and a consumer comparing pinning ranks across fits
/// must compare this first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinningRankSupport {
    /// `rank(H)` over the whole parameter space, at the rank tolerance
    /// `curvature_rank_tolerance` sets: the number of independent directions
    /// the fit's curvature resolves, out of `param_dim`. Its deficiency
    /// `param_dim − rank` is the dimension of `ker H`, i.e. of the exactly-flat
    /// directions INCLUDING the ones no enumerated generator points along.
    ParameterSpace,
    /// `rank(ΞᵀHΞ)` over the span of the enumerated generators `Ξ`, at the same
    /// tolerance taken on singular values of `RΞ`: how many of the
    /// Terracini-predicted tangent directions the curvature resolves as
    /// independent, out of `generators.len()`. This is the comparison the rank
    /// was introduced for — "a smaller pinning rank than the generator count" —
    /// and it is exact; what it is NOT is a statement about directions outside
    /// the enumerated span, so its deficiency does not bound `dim ker H`.
    GeneratorSpan,
}

impl PinningRankSupport {
    /// A stable label for summaries, certificates, and the Python surface.
    pub fn label(&self) -> &'static str {
        match self {
            Self::ParameterSpace => "parameter_space",
            Self::GeneratorSpan => "generator_span",
        }
    }
}

/// The certificate produced by [`residual_gauge`].
#[derive(Debug, Clone)]
pub struct ResidualGaugeReport {
    /// "computed in metric X" — read straight off
    /// [`RowMetric::provenance`]; the single metric object guarantees this
    /// matches the inner product the fit actually used.
    pub metric_provenance: MetricProvenance,
    /// Per-generator pinned/unpinned verdict, in enumeration order.
    pub generators: Vec<GeneratorVerdict>,
    /// Rank of the pinning span `range(H)` (data + isometry penalty) the
    /// generators were tested against, in the metric — over whichever support
    /// [`Self::pinning_rank_support`] names.
    pub pinning_rank: usize,
    /// What [`Self::pinning_rank`] is a rank of. Two certificates' pinning ranks
    /// are comparable only when this agrees (#2757).
    pub pinning_rank_support: PinningRankSupport,
    /// Number of generators certified as unpinned residual gauge freedoms.
    pub residual_gauge_dim: usize,
    /// `true` when the isometry pin is inactive (`isometry_penalty_root` has no
    /// rows): the model is then only identified up to an arbitrary
    /// diffeomorphism of the latent manifolds, and every isometry generator is
    /// reported as a residual freedom. This is the escalation flag.
    pub diffeomorphism_unpinned: bool,
    /// Under [`MetricProvenance::OutputFisher`] the `Sym(F)` permutation
    /// subgroup is expected to be *trivially pinned* — the output-Fisher metric
    /// distinguishes the atoms behaviorally so no atom-exchange can be a
    /// residual freedom. `true` ⇒ that triviality holds (every
    /// [`GeneratorFamily::AtomPermutation`] generator is pinned);
    /// `false` ⇒ a permutation survived as a residual freedom, which under
    /// OutputFisher provenance is a certificate violation the caller must
    /// surface. `None` ⇒ provenance is not `OutputFisher`, so the check does
    /// not apply.
    pub sym_f_trivial_under_output_fisher: Option<bool>,
    /// The #972 decoder-frame inner-rotation gauge `∏_k O(r_k)` — enumerated,
    /// never curvature-tested (see [`FrameInnerRotationGauge`] for why).
    /// `None` when the caller declared no frame factorization (full-`B`
    /// dictionaries, or a pre-#972 caller using [`residual_gauge`] directly);
    /// attach via [`ResidualGaugeReport::with_frame_inner_rotation`].
    pub frame_inner_rotation: Option<FrameInnerRotationGauge>,
    /// Human-readable one-line summary.
    pub summary: String,
}

impl ResidualGaugeReport {
    /// The certified residual gauge group, as a compact string naming the
    /// surviving generator families and their multiplicities. Two replicate
    /// fits are "identified up to the same group" iff this string is equal.
    ///
    /// When a frame inner-rotation gauge is enumerated it is appended with its
    /// dimension and its `[canonical-fixed]` marker — it is part of the group
    /// two replicate fits must agree on, even though it is fixed by
    /// convention rather than by curvature.
    pub fn group_signature(&self) -> String {
        let base = group_signature_of(&self.generators, self.diffeomorphism_unpinned);
        match &self.frame_inner_rotation {
            Some(gauge) if gauge.dim > 0 => format!(
                "{base} ⊕ frame-inner ∏O(r_k)×{} [dim {}, canonical-fixed]",
                gauge.per_atom_ranks.len(),
                gauge.dim
            ),
            _ => base,
        }
    }

}

/// Compact, order-independent signature of the unpinned generator families and
/// multiplicities. Two replicate fits agree on their residual gauge group iff
/// these strings are equal.
fn group_signature_of(generators: &[GeneratorVerdict], diffeomorphism_unpinned: bool) -> String {
    let mut counts: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    for g in generators {
        if g.unpinned {
            *counts.entry(g.family.label()).or_insert(0) += 1;
        }
    }
    let body = if counts.is_empty() {
        "{e} [fully pinned: rigid up to nothing]".to_string()
    } else {
        counts
            .iter()
            .map(|(name, mult)| format!("{name}×{mult}"))
            .collect::<Vec<_>>()
            .join(" ⊕ ")
    };
    if diffeomorphism_unpinned {
        // With the isometry pin inactive the residual gauge is at least the
        // manifold reparametrization (diffeomorphism) group modulo whatever the
        // data alone still pins — the surviving generators below are the
        // isometry slice of that larger freedom.
        format!("Diff(M) ⊇ {{ {body} }} [diffeomorphism-unpinned: isometry pin inactive]")
    } else {
        body
    }
}

/// Build the atom-local isometry generators for one atom as tangent directions
/// on the atom's flattened decoder frame.
///
/// An isometry of the latent manifold acts on the latent coordinate frame; we
/// lift it to the decoder output by acting on the frame columns. For a rotation
/// generator `A ∈ so(latent_dim)` (antisymmetric), the induced tangent direction
/// on `frame ∈ ℝ^{p × d}` is `frame · Aᵀ` (the first-order motion of the frame
/// columns under the one-parameter rotation `exp(tA)`), flattened row-major. For
/// the circle this is the single `so(2)` generator; for the sphere the three
/// `so(3)` generators; for the torus the `d` independent axis shifts (which on
/// the flat product manifold are translations of each circle coordinate —
/// realised as the unit tangent along each frame column).
fn atom_isometry_generators(atom: &FittedAtom) -> Vec<(Array1<f64>, String)> {
    let (p, d) = atom.frame.dim();
    // The intrinsic latent dimension of the manifold fixes `dim Isom(M_k)` (the
    // number of independent isometry generators we must enumerate). The fitted
    // decoder frame's column count `d` must realise exactly that many latent
    // axes; a frame whose column count disagrees with the topology's intrinsic
    // dimension is a structurally inconsistent atom and we refuse to fabricate
    // generators for it (returning none, so it cannot masquerade as either
    // pinned or a spurious residual freedom in the certificate).
    if d != atom.topology.latent_dim() {
        return Vec::new();
    }
    let mut out: Vec<(Array1<f64>, String)> = Vec::new();
    match &atom.topology {
        AtomTopology::Circle => {
            // The S¹ U(1) shift θ ↦ θ+ε is a *translation* of the periodic
            // coordinate, not a rotation of latent axes (so(1) = 0). Its
            // first-order action on the decoder FRAME parameter is the derivative
            // of the frame tangent along the orbit, i.e. the acceleration
            // ∂²g/∂θ² = J·(∂g/∂θ) — the 90°-rotated frame column IN THE CIRCLE'S
            // EMBEDDING 2-PLANE (the complex structure J). The mean frame stores
            // only the single tangent column ∂g/∂θ and no radial companion, so
            // that 2-plane — and hence J·frame — is not recoverable here. The
            // former emission used the tangent column `frame` ITSELF, which is the
            // amplitude/scaling direction (scaling the decoder always costs
            // reconstruction) and so read as a false "pin", biasing the circle
            // toward "identified". Rather than fabricate a J the mean frame cannot
            // supply, we emit nothing: the U(1) shift is faithfully certified by
            // the exact #998 orbit path (`exact_orbit_fields`, which realises it
            // as the chart-free coordinate motion δt = 1 and profiles out the
            // decoder compensation) and, for canonicalized d = 1 charts, recorded
            // as `PinnedByCanonicalization`. The frame path must not carry a
            // third, wrong-direction proxy for the same freedom.
        }
        AtomTopology::Sphere | AtomTopology::EuclideanPatch { .. } | AtomTopology::Torus { .. } => {
            // so(d) rotation generators: one per unordered axis pair (a < b).
            // The induced frame motion is frame · A_{ab}ᵀ, i.e. column a picks
            // up −column b and column b picks up +column a.
            for a in 0..d {
                for b in (a + 1)..d {
                    let mut g = Array1::<f64>::zeros(p * d);
                    for i in 0..p {
                        // (frame · Aᵀ)[i, a] = −frame[i, b]; [i, b] = +frame[i, a].
                        g[i * d + a] = -atom.frame[[i, b]];
                        g[i * d + b] = atom.frame[[i, a]];
                    }
                    out.push((
                        g,
                        format!(
                            "{}: {} rotation axes ({a},{b})",
                            atom.name,
                            match &atom.topology {
                                AtomTopology::Sphere => "S² so(3)",
                                AtomTopology::Torus { .. } => "Tᵈ frame",
                                _ => "patch so(d)",
                            }
                        ),
                    ));
                }
            }
            // Torus additionally carries `d` independent circle shifts (one U(1)
            // translation per axis). As in the [`AtomTopology::Circle`] arm above,
            // each shift's action on the frame parameter is the J-rotated (radial)
            // acceleration ∂²g/∂θ_a² in axis a's embedding 2-plane, which the mean
            // frame — storing only the tangent column ∂g/∂θ_a — cannot realise.
            // The former emission used the tangent column itself (the amplitude
            // direction, a false pin), so we drop it: the per-axis circle shifts
            // are certified exactly by the #998 orbit path (δt = e_a per axis) and,
            // for canonicalized torus charts, by `PinnedByCanonicalization`. The
            // so(d) latent-frame rotation generators above (realised faithfully as
            // frame·Aᵀ, a genuine rotation of latent axes) are unaffected.
        }
        // These quotient charts use exact nonlinear/group-specific orbit
        // fields below. A p×2 mean tangent frame cannot represent RP2's three
        // SO(3) generators, and treating Klein as a full torus would invent a
        // forbidden second circle shift.
        AtomTopology::ProjectivePlane | AtomTopology::KleinBottle => {}
    }
    out
}

/// Build equal-ARD rotation generators for one atom: a rotation between two
/// latent axes whose ARD variances are equal (within `rel_tol`) is not pinned by
/// the ARD prior, so it is a candidate residual gauge freedom (the data +
/// isometry penalty decide). Returns the antisymmetric frame-rotation tangent
/// for each such equal pair.
fn equal_ard_rotation_generators(atom: &FittedAtom) -> Vec<(Array1<f64>, String)> {
    let mut out: Vec<(Array1<f64>, String)> = Vec::new();
    let (p, d) = atom.frame.dim();
    let Some(ard) = atom.ard_variances.as_ref() else {
        return out;
    };
    if ard.len() != d {
        return out;
    }
    const ARD_EQUAL_REL_TOL: f64 = 1.0e-9;
    for a in 0..d {
        for b in (a + 1)..d {
            let va = ard[a];
            let vb = ard[b];
            let scale = va.abs().max(vb.abs()).max(f64::MIN_POSITIVE);
            if (va - vb).abs() <= ARD_EQUAL_REL_TOL * scale {
                let mut g = Array1::<f64>::zeros(p * d);
                for i in 0..p {
                    g[i * d + a] = -atom.frame[[i, b]];
                    g[i * d + b] = atom.frame[[i, a]];
                }
                out.push((
                    g,
                    format!("{}: equal-ARD rotation axes ({a},{b})", atom.name),
                ));
            }
        }
    }
    out
}

/// Build global decoder output-frame rotation generators: a rotation
/// `B ∈ so(output_dim)` acts on every atom's frame from the left (`B · frame`),
/// and the induced tangent on the joint parameter vector stacks `B · frame_k`
/// per atom.
///
/// # Why the full `so(output_dim)` basis is neither necessary nor feasible
///
/// Enumerating one generator per unordered output-axis pair `(oi < oj)` is
/// `p·(p−1)/2` dense param-length vectors; at production width `p = output_dim ≈
/// 1024` (with `param_dim` in the tens of thousands) that is hundreds of GB, so
/// the certificate cannot run on a real dictionary. The reduction below drops
/// exactly the generators that CANNOT affect the verdict, so it changes no
/// reported freedom while running at real width.
///
/// Let `S = span{ frame columns }` (dimension `r = rank`, `r ≤ Σ_k d_k ≪ p` for
/// the low-latent-dim atoms a dictionary actually carries), and `U` an
/// orthonormal basis of `S` in output space. Split every `so(p)` generator by
/// where its two axes sit relative to `S`:
///
/// * **both axes in `Sᗮ`** — `B · frame_k = 0` for every atom (the frame has no
///   component on either axis), so the tangent is IDENTICALLY ZERO. The
///   curvature test reports a zero-norm generator `unpinned = false`; it can
///   never be a residual freedom and never enters `residual_gauge_dim`.
/// * **one axis in `S`, one in `Sᗮ`** — the rotation tilts frame mass into an
///   ambient direction the reconstruction does not use, so it MOVES the
///   reconstruction orthogonally to itself; the data block of the curvature root
///   (`√W·J`, whose row space is the realised reconstruction motion) therefore
///   charges it a strictly positive second-order cost. It is always data-pinned,
///   so it too never enters `residual_gauge_dim`.
/// * **both axes in `S`** — a genuine reorientation of the fitted decoder
///   structure; this is the ONLY family that can be a metric-preserving residual
///   freedom (unpinned) and thus the only one the count can depend on.
///
/// So we enumerate exactly the `r·(r−1)/2` within-`S` rotations, expressed in the
/// orthonormal basis `U` and embedded per atom. This is bounded by the real
/// latent budget, and — because the dropped families are all zero-tangent or
/// data-pinned — leaves `residual_gauge_dim` (a count of UNPINNED verdicts)
/// unchanged. `output_dim` is the maximum frame row-count across atoms.
fn frame_rotation_generators(model: &FittedSaeManifold) -> Vec<(Array1<f64>, String)> {
    let mut out: Vec<(Array1<f64>, String)> = Vec::new();
    let p = model
        .atoms
        .iter()
        .map(|a| a.frame.nrows())
        .max()
        .unwrap_or(0);
    let param_dim = model.param_dim();
    if p == 0 || param_dim == 0 {
        return out;
    }

    // Stack every atom's frame columns as vectors in output space ℝ^p (rows past
    // an atom's own frame height are left zero). Column span = S.
    let n_cols: usize = model.atoms.iter().map(|a| a.frame.ncols()).sum();
    if n_cols == 0 {
        return out;
    }
    let mut cols = Array2::<f64>::zeros((p, n_cols));
    let mut col = 0usize;
    for atom in &model.atoms {
        let (ap, ad) = atom.frame.dim();
        for c in 0..ad {
            for i in 0..ap {
                cols[[i, col]] = atom.frame[[i, c]];
            }
            col += 1;
        }
    }

    // Orthonormal basis U of range(cols): left singular vectors with a non-negligible
    // singular value. Thin SVD of a p×n_cols matrix with n_cols ≪ p is cheap.
    let (u_opt, sv, _vt) = match cols.svd(true, false) {
        Ok(t) => t,
        Err(_) => return out,
    };
    let u = match u_opt {
        Some(u) => u,
        None => return out,
    };
    let smax = sv.iter().cloned().fold(0.0_f64, f64::max);
    if !(smax > 0.0) {
        return out;
    }
    let tol = smax * f64::EPSILON * (p.max(n_cols) as f64);
    let r = sv.iter().filter(|&&s| s > tol).count().min(u.ncols());
    if r < 2 {
        // Fewer than two independent frame directions ⇒ no non-trivial within-span
        // rotation exists.
        return out;
    }

    // Within-span rotation B_{ab} = u_a u_bᵀ − u_b u_aᵀ; its frame motion is
    // (B_{ab} · frame_k)[i,c] = u_a[i]·(u_bᵀ frame_k[:,c]) − u_b[i]·(u_aᵀ frame_k[:,c]).
    for a in 0..r {
        for b in (a + 1)..r {
            let mut g = Array1::<f64>::zeros(param_dim);
            for (k, atom) in model.atoms.iter().enumerate() {
                let (ap, ad) = atom.frame.dim();
                let base = model.atom_offset(k);
                for c in 0..ad {
                    // Project this frame column onto the two basis axes.
                    let mut proj_a = 0.0_f64;
                    let mut proj_b = 0.0_f64;
                    for i in 0..ap {
                        proj_a += u[[i, a]] * atom.frame[[i, c]];
                        proj_b += u[[i, b]] * atom.frame[[i, c]];
                    }
                    if proj_a == 0.0 && proj_b == 0.0 {
                        continue;
                    }
                    for i in 0..ap {
                        g[base + i * ad + c] = u[[i, a]] * proj_b - u[[i, b]] * proj_a;
                    }
                }
            }
            out.push((
                g,
                format!("output-frame rotation within-span axes ({a},{b})"),
            ));
        }
    }
    out
}

/// Build exchangeable-atom permutation generators: for every pair of atoms with
/// identical topology and matching frame shape, the transposition that swaps
/// their decoder frames is a candidate `Sym(F)` symmetry. Realised as the
/// antisymmetric "swap" tangent `(frame_b − frame_a)` placed on atom a's slot and
/// `(frame_a − frame_b)` on atom b's slot — the first-order direction of the
/// one-parameter family interpolating the swap.
/// Embed an atom-local generator (length = that atom's flattened frame length)
/// into the joint parameter vector at the atom's column offset. The per-atom
/// generator builders do not know the joint layout; the certificate does, and
/// mixing the two coordinate systems is a shape error for every model with more
/// than one atom.
fn embed_local_generator(offset: usize, local: &Array1<f64>, param_dim: usize) -> Array1<f64> {
    let mut g = Array1::<f64>::zeros(param_dim);
    g.slice_mut(s![offset..offset + local.len()]).assign(local);
    g
}

fn atom_permutation_generators(
    model: &FittedSaeManifold,
) -> Vec<(Array1<f64>, String, usize, usize)> {
    let mut out: Vec<(Array1<f64>, String, usize, usize)> = Vec::new();
    let param_dim = model.param_dim();
    for ka in 0..model.atoms.len() {
        for kb in (ka + 1)..model.atoms.len() {
            let a = &model.atoms[ka];
            let b = &model.atoms[kb];
            if a.topology != b.topology || a.frame.dim() != b.frame.dim() {
                continue;
            }
            let (ap, ad) = a.frame.dim();
            let base_a = model.atom_offset(ka);
            let base_b = model.atom_offset(kb);
            let mut g = Array1::<f64>::zeros(param_dim);
            for i in 0..ap {
                for c in 0..ad {
                    let diff = b.frame[[i, c]] - a.frame[[i, c]];
                    g[base_a + i * ad + c] = diff;
                    g[base_b + i * ad + c] = -diff;
                }
            }
            out.push((g, format!("atom-exchange {} ↔ {}", a.name, b.name), ka, kb));
        }
    }
    out
}

// ============================================================================
// #998 — the full-resolution certificate: exact gauge orbits in the model's
// own (decoder, coordinate) parameter space.
// ============================================================================

/// One atom's exact parameter-space view (#998): the raw objects the fit
/// actually optimizes, in which the model-class gauge orbits live.
///
/// The mean-frame certificate ([`FittedAtom::frame`]) is a lossy compression:
/// the true gauge orbits are **compensated** motions — the latent coordinates
/// move AND the decoder counter-rotates (e.g. `Φ(t+ε)·R(−ε)B = Φ(t)B` for the
/// harmonic circle) — whose net action on the mean frame is identically zero,
/// so no frame-space realisation can measure them (#995's calibrated tolerance
/// is the honest *floor* there). With this view the certificate realises each
/// orbit exactly: the coordinate motion field `δt` comes from the group
/// action, and the decoder compensation `δB` is **profiled out by least
/// squares** against the data motion. The leftover residual is the orbit's
/// true data cost — exactly zero when the basis family is closed under the
/// action (harmonics under shifts, linear charts under rotations), genuinely
/// positive when it is not (a Duchon patch under so(d)). Basis closure is
/// therefore a *computed* per-generator quantity, not a declared flag.
#[derive(Debug, Clone)]
pub struct AtomParameterView {
    /// Basis values `Φ`, `(n, M)`.
    pub basis_values: Array2<f64>,
    /// Basis first-derivative jet `Φ'`, `(n, M, latent_dim)`.
    pub basis_jacobian: Array3<f64>,
    /// Decoder coefficients `B`, `(M, p)`.
    pub decoder: Array2<f64>,
    /// Latent coordinates `t`, `(n, latent_dim)` — the chart the group acts on.
    pub coords: Array2<f64>,
    /// Per-row assignment mass `a_nk`, length `n`.
    pub activations: Array1<f64>,
    /// Basis second-derivative jet `Φ''`, `(n, M, latent_dim, latent_dim)`.
    /// Required only to lower an isometry [`OrbitPenaltyOperator`] for a
    /// *pin-active* fit (#998): the penalty is a function of the pullback
    /// metric `g_n = J_nᵀ W_n J_n`, and the first-order change of `g_n` under a
    /// coordinate motion `δt` differentiates `J_n = Φ'_n B` through `t`, which
    /// needs `Φ''`. `None` keeps the data-only orbit verdict (no pin), exactly
    /// as before; absence never errors.
    pub basis_second_jet: Option<Array4<f64>>,
}

/// The penalty/prior channel of the exact certificate: an operator returning
/// the penalty curvature root's image of an orbit direction `(δB, δt)`,
/// together with its stiffness scale `σ_max²`. With exact orbits the data can
/// never pin a model-class symmetry (the LS-compensated motion is a data-null
/// by construction for closed bases), so **all** pinning of such symmetries
/// flows through this channel — exactly where the #981 gauge-reduction ladder
/// says identification lives (the isometry pin does the collapsing, rungs 2
/// and 4, in whichever metric it is computed). `None` ⇒ no pin installed on
/// this atom; the orbit's verdict is then decided by the data residual alone.
pub struct OrbitPenaltyOperator {
    /// Maps an orbit direction `(δB (M, p), δt (n, latent_dim))` to the
    /// penalty curvature root's image (any length); the penalty cost along the
    /// direction is the squared norm of the image.
    pub apply: Box<dyn Fn(ArrayView2<f64>, ArrayView2<f64>) -> Array1<f64> + Send + Sync>,
    /// `σ_max²` of the penalty curvature root — the stiffness scale the
    /// orbit's penalty cost is reported relative to (the same
    /// relative-curvature convention as the frame certificate).
    pub stiffness_sq: f64,
}

/// Build the isometry-pin [`OrbitPenaltyOperator`] for one viewed atom from its
/// second jet (#998 — the orbit-space pin operator the pin-active exact path
/// needs).
///
/// The isometry penalty is `P = ½ μ Σ_n ‖g_n − g_ref‖²_F` with the pullback
/// first-fundamental-form gram `g_n = J_nᵀ J_n`, `J_n[i,c] = Σ_m Φ'_n[m,c] B[m,i]`
/// (Euclidean metric — the default isometry reference; an output-Fisher metric
/// rides the same operator once its factors are threaded, which only re-weights
/// the `i`-sum). At a converged isometric fit the residual `g_n − g_ref ≈ 0`, so
/// the penalty's curvature along an orbit direction `(δB, δt)` is the
/// Gauss-Newton term `μ Σ_n ‖δg_n‖²_F`, and the curvature-root image is
/// `√μ · {δg_n[a,b]}` — its squared norm is exactly that cost. The first-order
/// gram change
///
///   `δJ_n[i,c] = Σ_m Φ'_n[m,c] δB[m,i] + Σ_{m,e} Φ''_n[m,c,e] δt_n[e] B[m,i]`
///   `δg_n[a,b] = Σ_i ( δJ_n[i,a] J_n[i,b] + J_n[i,a] δJ_n[i,b] )`
///
/// differentiates `J_n` through `t` via the **second jet** `Φ''` — which is why
/// the pin-active path needs it and the frame path (no second jet) could not
/// supply it. A model-class symmetry that preserves the metric (e.g. a circle
/// phase shift on a closed harmonic basis) yields `δg_n = 0` → the operator
/// gives it zero cost → it stays a certified freedom even under the pin; a
/// non-isometric orbit (a Duchon/quadratic patch under rotation) yields
/// `δg_n ≠ 0` → genuine pinning. The verdict is therefore conservative: the
/// operator can only *cost* an orbit, never spuriously free one.
///
/// `weight` is the penalty strength `μ`. Returns `None` when the view carries no
/// second jet (the atom's basis exposes no analytic Hessian): with no orbit-space
/// operator the atom's verdict falls back to the data residual, never an error.
/// The stiffness `σ_max²` is `μ` times the largest unit-coordinate-motion gram
/// curvature `max_n σ_max(∂g_n/∂t)²`, so the reported relative fraction is on the
/// same convention as the frame certificate.
pub fn isometry_orbit_penalty_operator(
    view: &AtomParameterView,
    weight: f64,
) -> Option<OrbitPenaltyOperator> {
    let second = view.basis_second_jet.as_ref()?.clone();
    let (n, m) = view.basis_values.dim();
    let d = view.coords.ncols();
    let p = view.decoder.ncols();
    if second.dim() != (n, m, d, d) || view.basis_jacobian.dim() != (n, m, d) {
        return None;
    }
    if !(weight.is_finite() && weight > 0.0) {
        return None;
    }
    let sqrt_w = weight.sqrt();
    let jac = view.basis_jacobian.clone();
    let decoder = view.decoder.clone();

    // Base pullback Jacobian J_n[i,c] = Σ_m Φ'_n[m,c] B[m,i] and its per-row
    // first-fundamental gram σ_max scale (stiffness), computed once.
    let mut j_base = Array3::<f64>::zeros((n, p, d));
    for row in 0..n {
        for i in 0..p {
            for c in 0..d {
                let mut acc = 0.0;
                for mm in 0..m {
                    acc += jac[[row, mm, c]] * decoder[[mm, i]];
                }
                j_base[[row, i, c]] = acc;
            }
        }
    }

    // Stiffness: σ_max over rows of the gram derivative ∂g_n/∂t along a unit
    // coordinate motion. ∂g_n/∂t_e [a,b] = Σ_i ( H_n[i,a,e] J_n[i,b]
    // + J_n[i,a] H_n[i,b,e] ), H_n[i,c,e] = Σ_m Φ''_n[m,c,e] B[m,i]. The
    // stiffest unit δt direction's gram change drives the relative-curvature
    // denominator; we take the largest ‖∂g/∂t_e‖_F over axes e and rows as a
    // conservative (≤ true σ_max) scale, so the reported fraction never
    // under-states the pin.
    let mut max_curv_sq = 0.0_f64;
    for row in 0..n {
        // H_n[i, c, e] = Σ_m Φ''_n[m, c, e] B[m, i].
        let mut hn = vec![0.0_f64; p * d * d];
        for i in 0..p {
            for c in 0..d {
                for e in 0..d {
                    let mut acc = 0.0;
                    for mm in 0..m {
                        acc += second[[row, mm, c, e]] * decoder[[mm, i]];
                    }
                    hn[(i * d + c) * d + e] = acc;
                }
            }
        }
        for e in 0..d {
            // Build the symmetric d×d gram-derivative slab G_e (g[a,b] = g[b,a]
            // by construction).
            let mut g_e = Array2::<f64>::zeros((d, d));
            for a in 0..d {
                for b in 0..d {
                    let mut g = 0.0;
                    for i in 0..p {
                        g += hn[(i * d + a) * d + e] * j_base[[row, i, b]];
                        g += j_base[[row, i, a]] * hn[(i * d + b) * d + e];
                    }
                    g_e[[a, b]] = g;
                }
            }
            // Stiffness is the SPECTRAL norm σ_max(G_e), NOT the Frobenius norm:
            // since ‖G_e‖_F² = Σ_i σ_i² ≥ σ_max², using Frobenius OVER-states the
            // stiffness → UNDER-states the pinned-energy fraction → could drop a
            // genuinely-pinned generator below GENERATOR_FLAT_ENERGY_TOL and
            // falsely certify it UNPINNED (spurious non-identifiability — the
            // opposite of the "never under-state the pin" guarantee). G_e is
            // symmetric, so σ_max = max|eigenvalue| (see `symmetric_spectral_norm_sq`).
            max_curv_sq = max_curv_sq.max(symmetric_spectral_norm_sq(g_e.view()));
        }
    }
    let stiffness_sq = (weight * max_curv_sq).max(f64::MIN_POSITIVE);

    let apply = move |delta_b: ArrayView2<f64>, delta_t: ArrayView2<f64>| -> Array1<f64> {
        let mut image = Array1::<f64>::zeros(n * d * d);
        // δJ_n[i,c] = Σ_m Φ'_n[m,c] δB[m,i] + Σ_{m,e} Φ''_n[m,c,e] δt_n[e] B[m,i].
        let valid_b = delta_b.dim() == (m, p);
        let valid_t = delta_t.dim() == (n, d);
        if !valid_t {
            return image;
        }
        for row in 0..n {
            let mut dj = vec![0.0_f64; p * d];
            for i in 0..p {
                for c in 0..d {
                    let mut acc = 0.0;
                    if valid_b {
                        for mm in 0..m {
                            acc += jac[[row, mm, c]] * delta_b[[mm, i]];
                        }
                    }
                    for e in 0..d {
                        let dte = delta_t[[row, e]];
                        if dte == 0.0 {
                            continue;
                        }
                        for mm in 0..m {
                            acc += second[[row, mm, c, e]] * dte * decoder[[mm, i]];
                        }
                    }
                    dj[i * d + c] = acc;
                }
            }
            // δg_n[a,b] = Σ_i ( δJ[i,a] J[i,b] + J[i,a] δJ[i,b] ).
            for a in 0..d {
                for b in 0..d {
                    let mut dg = 0.0;
                    for i in 0..p {
                        dg += dj[i * d + a] * j_base[[row, i, b]];
                        dg += j_base[[row, i, a]] * dj[i * d + b];
                    }
                    image[(row * d + a) * d + b] = sqrt_w * dg;
                }
            }
        }
        image
    };

    Some(OrbitPenaltyOperator {
        apply: Box::new(apply),
        stiffness_sq,
    })
}

/// Squared spectral norm `σ_max(G)²` of a SYMMETRIC `d×d` matrix — the correct
/// stiffness scale for the isometry-orbit pin (a Frobenius `Σσ_i²` over-states it
/// and can spuriously free a pinned generator; see `isometry_orbit_penalty_operator`).
/// `σ_max = max|eigenvalue|` for a symmetric matrix. Falls back to the largest
/// column 2-norm — a valid LOWER bound on `σ_max` (`‖G e_j‖ ≤ σ_max`), so still
/// conservative in the "never under-state the pin" direction — if the symmetric
/// eig fails (e.g. a non-finite slab). For `d = 1` this is exactly `g²`
/// (Frobenius and σ_max coincide), so the `d = 1` path is unchanged.
fn symmetric_spectral_norm_sq(g: ArrayView2<'_, f64>) -> f64 {
    let d = g.nrows();
    if d == 0 {
        return 0.0;
    }
    match g.to_owned().eigh(Side::Lower) {
        Ok((evals, _)) => {
            let s = evals.iter().fold(0.0_f64, |mx, &v| mx.max(v.abs()));
            s * s
        }
        Err(_) => {
            let mut max_col_sq = 0.0_f64;
            for b in 0..g.ncols() {
                let mut col_sq = 0.0_f64;
                for a in 0..d {
                    col_sq += g[[a, b]] * g[[a, b]];
                }
                max_col_sq = max_col_sq.max(col_sq);
            }
            max_col_sq
        }
    }
}

/// Enumerate one atom's exact orbit coordinate-motion fields `δt ∈ ℝ^{n×d}`.
///
/// Circle/torus/Klein shifts and flat-patch rotations are linear in their
/// charts. `RP²` uses the exact nonlinear coordinate velocities of the ambient
/// `SO(3)` action on its spherical cover; those velocities come from the same
/// authority used by fit-time gauge deflation. The sphere's legacy chart basis
/// is not closed under ambient rotations, so sphere atoms remain on the frame
/// path. Equal-ARD rotations reuse the flat-chart rotation field for tied axis
/// pairs (the ARD prior is their pinning channel).
fn exact_orbit_fields(
    atom: &FittedAtom,
    view: &AtomParameterView,
) -> Result<Vec<(GeneratorFamily, Array2<f64>, String)>, String> {
    let n = view.coords.nrows();
    let d = view.coords.ncols();
    let mut out: Vec<(GeneratorFamily, Array2<f64>, String)> = Vec::new();
    let rotation_field = |a: usize, b: usize| -> Array2<f64> {
        let mut dt = Array2::<f64>::zeros((n, d));
        for row in 0..n {
            dt[[row, a]] = -view.coords[[row, b]];
            dt[[row, b]] = view.coords[[row, a]];
        }
        dt
    };
    match &atom.topology {
        AtomTopology::Circle => {
            out.push((
                GeneratorFamily::IsomAtom,
                Array2::<f64>::ones((n, 1)),
                format!("{}: S¹ U(1) phase shift [exact orbit]", atom.name),
            ));
        }
        AtomTopology::Torus { .. } => {
            for ax in 0..d {
                let mut dt = Array2::<f64>::zeros((n, d));
                dt.column_mut(ax).fill(1.0);
                out.push((
                    GeneratorFamily::IsomAtom,
                    dt,
                    format!("{}: Tᵈ circle shift axis {ax} [exact orbit]", atom.name),
                ));
            }
        }
        AtomTopology::KleinBottle => {
            let mut dt = Array2::<f64>::zeros((n, d));
            dt.column_mut(0).fill(1.0);
            out.push((
                GeneratorFamily::IsomAtom,
                dt,
                format!("{}: Klein S1 theta translation [exact orbit]", atom.name),
            ));
        }
        // `S²` and `RP²` share one implementation because they share one group
        // and one cover. `Isom(S²) = O(3)`, and on the ambient cover its Killing
        // fields are simply `K_a(u) = e_a × u` — LINEAR in the coordinate,
        // pole-free, and identical for the quotient, whose deck `u ~ −u` commutes
        // with every rotation.
        //
        // The sphere used to be excluded from this path entirely ("the sphere's
        // legacy chart basis is not closed under ambient rotations, so sphere
        // atoms remain on the frame path"), and `RP²` reached it only through
        // nonlinear `(lat, lon)` cover velocities that could not be evaluated at
        // the cover's own poles. Both were consequences of the chart, and both
        // go with it: the ambient basis IS closed under these generators, which
        // `ambient_sphere_basis_is_closed_under_its_own_killing_fields` asserts
        // to 1e-9. So the sphere now earns the exact-orbit certificate its own
        // quotient already had.
        AtomTopology::Sphere | AtomTopology::ProjectivePlane => {
            if d != 3 {
                return Err(format!(
                    "exact_orbit_fields({}): the spherical cover is the ambient unit vector and needs three coordinates; got {d}",
                    atom.name
                ));
            }
            let mut rotation_x = Array2::<f64>::zeros((n, 3));
            let mut rotation_y = Array2::<f64>::zeros((n, 3));
            let mut rotation_z = Array2::<f64>::zeros((n, 3));
            for row in 0..n {
                let u = [
                    view.coords[[row, 0]],
                    view.coords[[row, 1]],
                    view.coords[[row, 2]],
                ];
                // `e_x × u`, `e_y × u`, `e_z × u`.
                let generators = [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]];
                for axis in 0..3 {
                    rotation_x[[row, axis]] = generators[0][axis];
                    rotation_y[[row, axis]] = generators[1][axis];
                    rotation_z[[row, axis]] = generators[2][axis];
                }
            }
            let label = if matches!(atom.topology, AtomTopology::Sphere) {
                "S2"
            } else {
                "RP2"
            };
            for (axis, field) in [("x", rotation_x), ("y", rotation_y), ("z", rotation_z)] {
                out.push((
                    GeneratorFamily::IsomAtom,
                    field,
                    format!(
                        "{}: {label} SO(3) rotation about {axis} [exact ambient orbit]",
                        atom.name
                    ),
                ));
            }
        }
        AtomTopology::EuclideanPatch { .. } => {
            for a in 0..d {
                for b in (a + 1)..d {
                    out.push((
                        GeneratorFamily::IsomAtom,
                        rotation_field(a, b),
                        format!(
                            "{}: patch so(d) rotation axes ({a},{b}) [exact orbit]",
                            atom.name
                        ),
                    ));
                }
            }
        }
    }
    // Equal-ARD rotations between tied axes, on linearly-acting charts only.
    if !matches!(
        atom.topology,
        AtomTopology::Circle
            | AtomTopology::Sphere
            | AtomTopology::ProjectivePlane
            | AtomTopology::KleinBottle
    ) {
        if let Some(ard) = atom.ard_variances.as_ref() {
            if ard.len() == d {
                const ARD_EQUAL_REL_TOL: f64 = 1.0e-9;
                for a in 0..d {
                    for b in (a + 1)..d {
                        let scale = ard[a].abs().max(ard[b].abs()).max(f64::MIN_POSITIVE);
                        if (ard[a] - ard[b]).abs() <= ARD_EQUAL_REL_TOL * scale {
                            out.push((
                                GeneratorFamily::EqualArdRotation,
                                rotation_field(a, b),
                                format!(
                                    "{}: equal-ARD rotation axes ({a},{b}) [exact orbit]",
                                    atom.name
                                ),
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Exact-orbit verdicts for one viewed atom (#998).
///
/// For each orbit field `δt`: the uncompensated data motion is
/// `u_n = a_n · (Φ'_n B) δt_n ∈ ℝ^p`; the decoder compensation `δB` minimizing
/// `Σ_n ‖a_n Φ_n δB + u_n‖²` is profiled out through one shared SVD
/// pseudo-inverse of the activation-weighted basis `D = diag(a) Φ`; and the
/// **compensation residual fraction** `r²/‖u‖²` is the orbit's true relative
/// data cost — exactly 0 for a basis closed under the group action, genuinely
/// positive otherwise (computed closure). The penalty channel, when installed,
/// contributes `‖penalty_root(δB, δt)‖² / σ_max²` on the same
/// relative-curvature convention. The verdict needs **no lowering-error
/// calibration** (`lowering_error_scale = 0`): nothing here is compressed.
///
/// The data likelihood this measures against is the activation-reconstruction
/// objective in its own (Euclidean) inner product — which per the amended #980
/// dispatch rule is the only thing that ever whitens the likelihood unless a
/// `WhitenedStructured` noise model is installed; the output-Fisher metric
/// reaches gauge verdicts only through the penalty operator.
fn exact_orbit_verdicts(
    atom: &FittedAtom,
    view: &AtomParameterView,
    penalty: Option<&OrbitPenaltyOperator>,
) -> Result<Vec<GeneratorVerdict>, String> {
    let (n, m) = view.basis_values.dim();
    let d = view.coords.ncols();
    let p = view.decoder.ncols();
    if view.basis_jacobian.dim() != (n, m, d) {
        return Err(format!(
            "exact_orbit_verdicts({}): basis_jacobian shape {:?} must be ({n}, {m}, {d})",
            atom.name,
            view.basis_jacobian.dim()
        ));
    }
    if view.decoder.nrows() != m {
        return Err(format!(
            "exact_orbit_verdicts({}): decoder has {} rows but basis has {m} columns",
            atom.name,
            view.decoder.nrows()
        ));
    }
    if view.coords.nrows() != n || view.activations.len() != n {
        return Err(format!(
            "exact_orbit_verdicts({}): coords/activations rows must match basis rows {n}",
            atom.name
        ));
    }

    let fields = exact_orbit_fields(atom, view)?;
    if fields.is_empty() {
        return Ok(Vec::new());
    }

    // Shared compensation operator: thin SVD of D = diag(a)·Φ, computed once.
    let mut design = Array2::<f64>::zeros((n, m));
    for row in 0..n {
        let a = view.activations[row];
        for c in 0..m {
            design[[row, c]] = a * view.basis_values[[row, c]];
        }
    }
    let (u_opt, sigma, vt_opt) = design
        .svd(true, true)
        .map_err(|e| format!("exact_orbit_verdicts({}): SVD of D failed: {e}", atom.name))?;
    let u_svd =
        u_opt.ok_or_else(|| format!("exact_orbit_verdicts({}): SVD lacked U", atom.name))?;
    let vt = vt_opt.ok_or_else(|| format!("exact_orbit_verdicts({}): SVD lacked Vᵀ", atom.name))?;
    let smax = sigma.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = smax * f64::EPSILON * (n.max(m) as f64);

    let mut out: Vec<GeneratorVerdict> = Vec::with_capacity(fields.len());
    for (family, dt, description) in fields {
        // Uncompensated data motion u_n = a_n (Φ'_n B) δt_n.
        let mut u_mot = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let a = view.activations[row];
            if !(a != 0.0) {
                continue;
            }
            for ax in 0..d {
                let step = dt[[row, ax]];
                if step == 0.0 {
                    continue;
                }
                for bm in 0..m {
                    let dphi = view.basis_jacobian[[row, bm, ax]];
                    if dphi == 0.0 {
                        continue;
                    }
                    let w = a * step * dphi;
                    for j in 0..p {
                        u_mot[[row, j]] += w * view.decoder[[bm, j]];
                    }
                }
            }
        }
        let raw: f64 = u_mot.iter().map(|v| v * v).sum();
        if raw <= f64::MIN_POSITIVE {
            // The orbit does not move the fit at all (zero tangents / zero
            // mass): structurally trivial, reported pinned with zero norm,
            // mirroring the frame certificate's convention. Same
            // degenerate-tangent / RLCT-½ exclusion as the frame-space veto in
            // `residual_gauge_inner` (see that site for the full argument): a
            // `rank_eff = 0` orbit buys no genuine identifiability dimension,
            // so it must never be reported as a residual freedom.
            out.push(GeneratorVerdict {
                family,
                description,
                unpinned: false,
                generator_norm: 0.0,
                pinned_energy_fraction: 1.0,
                lowering_error_scale: 0.0,
                provenance: VerdictProvenance::CurvatureTest,
            });
            continue;
        }
        // Profile out the decoder compensation: c = Uᵀu, keep σ > cutoff.
        // Residual cost r² = ‖u‖² − ‖c_kept‖² (Pythagoras on the projection).
        let coeffs = u_svd.t().dot(&u_mot);
        let mut kept_sq = 0.0_f64;
        let mut scaled = Array2::<f64>::zeros((sigma.len(), p));
        for r in 0..sigma.len() {
            if sigma[r] > cutoff {
                let inv = 1.0 / sigma[r];
                for j in 0..p {
                    kept_sq += coeffs[[r, j]] * coeffs[[r, j]];
                    scaled[[r, j]] = -inv * coeffs[[r, j]];
                }
            }
        }
        let resid_sq = (raw - kept_sq).max(0.0);
        let data_fraction = (resid_sq / raw).clamp(0.0, 1.0);

        let penalty_fraction = match penalty {
            Some(op) if op.stiffness_sq > f64::MIN_POSITIVE => {
                let delta_b = vt.t().dot(&scaled); // δB = −V Σ⁺ Uᵀ u, (M, p)
                let image = (op.apply)(delta_b.view(), dt.view());
                // `cost = ‖image‖² = μ·Σ_n ‖δg_n‖²_F` is EXTENSIVE — it sums the
                // per-row gram-curvature over every row the orbit moves, so it
                // grows with the motion size `‖δt‖²` (≈ n for the δt = 1 circle
                // shift). `stiffness_sq = μ·max_n σ_max(∂g_n/∂t)²` is the
                // INTENSIVE per-unit-COORDINATE-motion stiffness scale. Dividing
                // the extensive cost by the intensive scale gives a ratio that
                // scales like `‖δt‖²` and clamps to 1 for any real motion,
                // overstating the pin. To land on the documented relative
                // convention — the same scale-invariant fraction `data_fraction`
                // uses — normalise the cost by the orbit's own coordinate-motion
                // energy `‖δt‖²` so the reported quantity is the penalty cost per
                // unit coordinate motion relative to the stiffest unit motion.
                let motion_sq: f64 = dt.iter().map(|v| v * v).sum();
                if motion_sq > f64::MIN_POSITIVE {
                    let cost: f64 = image.iter().map(|v| v * v).sum();
                    (cost / (op.stiffness_sq * motion_sq)).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };

        let pinned_energy_fraction = data_fraction.max(penalty_fraction);
        out.push(GeneratorVerdict {
            family,
            description,
            unpinned: pinned_energy_fraction <= GENERATOR_FLAT_ENERGY_TOL,
            generator_norm: raw.sqrt(),
            pinned_energy_fraction,
            lowering_error_scale: 0.0,
            provenance: VerdictProvenance::CurvatureTest,
        });
    }
    Ok(out)
}

/// The curvature `H = H_data + H_isometry`, reduced to exactly the three things
/// the certificate reads off it: the pinning rank, the stiffness scale
/// `σ_max(R)²`, and the quadratic form `ξᵀHξ` along a unit generator.
///
/// The variants are not implementation choices — each is a different *exact*
/// representation of the same operator, and the builder picks the one the fit's
/// structure makes available (#2757):
///
/// * [`Self::OutputBlocks`] — the pinning Jacobian is output-coordinate
///   diagonal, so with a metric that does not couple output coordinates `H` is
///   block diagonal: `p` blocks of `D × D`. Spectrum = the union of the block
///   spectra, `p·D³` flops instead of `(p·D)³`.
/// * [`Self::DualRoot`] — `H = RᵀR` with `R` having fewer rows `m` than
///   columns. `spec(RᵀR) = spec(RRᵀ) ∪ {0}^{param_dim − m}`, so the same
///   spectral decisions come from an `m × m` eigenproblem.
/// * [`Self::Gram`] — the unstructured fallback.
enum CurvatureReduction {
    Gram {
        pinning_rank: usize,
        sigma_max_sq: f64,
        gram: Array2<f64>,
    },
    /// `H` as `p` independent `D × D` output-coordinate block roots, plus the
    /// isometry pin's dense rows when one is installed.
    OutputBlockRoots {
        pinning_rank: usize,
        sigma_max_sq: f64,
        roots: Array3<f64>,
        dense_rows: Array2<f64>,
        layout: FrameColumnLayout,
    },
    /// `H = RᵀR` held as `R` (`m × param_dim`, `m ≤ param_dim`), decided on
    /// `R`'s own singular values.
    DualRoot {
        pinning_rank: usize,
        sigma_max_sq: f64,
        root: Array2<f64>,
    },
}

/// The pinning-rank tolerance, in singular values of `R`.
///
/// The same shape as [`gam_linalg::faer_ndarray::rrqr_with_permutation`]'s own
/// threshold (`α · ε · max(rows, cols) · max(scale, 1)`), so the streamed
/// reductions and the RRQR path make one decision rather than two. Written once
/// here so no reduction can drift into a tolerance of its own.
fn curvature_rank_tolerance(sigma_max: f64, root_rows: usize, param_dim: usize) -> f64 {
    default_rrqr_rank_alpha()
        * f64::EPSILON
        * (root_rows.max(param_dim).max(1) as f64)
        * sigma_max.max(1.0)
}

/// The rank decision taken where it belongs — on singular values of `R`.
///
/// `singular_values` is the multiset of *nonzero* singular values; the
/// structural zeros a rank-deficient representation omits never clear a
/// positive tolerance, so omitting them is exact rather than an approximation.
pub(crate) fn root_spectral_rank(
    singular_values: &[f64],
    root_rows: usize,
    param_dim: usize,
) -> (f64, usize) {
    let sigma_max = singular_values
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max)
        .max(0.0);
    let rank_tol = curvature_rank_tolerance(sigma_max, root_rows, param_dim);
    let pinning_rank = singular_values
        .iter()
        .filter(|&&sigma| sigma > rank_tol)
        .count();
    (sigma_max * sigma_max, pinning_rank)
}

/// The same decision when only `H = RᵀR` survives: `σ > τ ⟺ λ > τ²`, floored
/// at the eigendecomposition's own resolution.
///
/// Algebraically `λ > τ²` is identical to [`root_spectral_rank`]'s `σ > τ`.
/// Numerically it is not, and the gap is not marginal. The tolerance is
/// `τ = α·ε·N·σ_max` with `α = 100`, i.e. deliberately **100× above** the
/// backward error `≈ ε·N·σ_max` of a singular-value decomposition of `R` — that
/// margin is what makes the decision meaningful. Squaring it gives
/// `τ² = α²·ε²·N²·σ_max²` against a symmetric eigensolver's backward error
/// `≈ ε·N·σ_max²`, so the threshold lands a factor `α²·ε·N ≈ 2·10⁻¹²·N`
/// **below** the instrument's resolution: every roundoff eigenvalue clears it
/// and the reported rank is the matrix dimension whatever the true rank is.
/// Measured on the #2757 fixture, a curvature of true rank 12 in 80 parameters
/// was reported as rank 45 by exactly this route.
///
/// So a Gram-side decision must not claim resolution below `ε·dim·λ_max`, the
/// standard `|λ̃ − λ| ≤ c·ε·‖H‖₂` bound with the conservative `c = dim`. That
/// floor is derived from the instrument, not chosen: it is the smallest
/// eigenvalue this decomposition can distinguish from zero, and below it
/// "nonzero" is not a statement the Gram can make.
///
/// Every representation that retains a root avoids the whole question and uses
/// [`root_spectral_rank`]; this exists for
/// [`ResidualGaugeCurvature::DenseGram`], which by construction has no root to
/// ask — it is reached only when the root is the larger object.
fn gram_spectral_rank(spectrum: &[f64], root_rows: usize, param_dim: usize) -> (f64, usize) {
    let sigma_max_sq = spectrum.iter().cloned().fold(0.0_f64, f64::max).max(0.0);
    let rank_tol = curvature_rank_tolerance(sigma_max_sq.sqrt(), root_rows, param_dim);
    let resolution_floor = f64::EPSILON * (param_dim.max(1) as f64) * sigma_max_sq;
    let lambda_tol = (rank_tol * rank_tol).max(resolution_floor);
    let pinning_rank = spectrum
        .iter()
        .filter(|&&lambda| lambda.max(0.0) > lambda_tol)
        .count();
    (sigma_max_sq, pinning_rank)
}

impl CurvatureReduction {

    /// Reduce whichever representation the streaming builder produced.
    ///
    /// The curvature is checked against the model it will certify — in shape,
    /// in parameterization, and in finiteness — because a streamed builder and
    /// a certificate model are two objects that must have been derived from the
    /// same atom list, and nothing downstream of here can tell that they were
    /// not.
    fn from_curvature(
        curvature: ResidualGaugeCurvature,
        model: &FittedSaeManifold,
    ) -> Result<Self, String> {
        let param_dim = model.param_dim();
        if curvature.param_dim() != param_dim {
            return Err(format!(
                "residual_gauge: curvature is over {} parameters but param_dim = {param_dim}",
                curvature.param_dim()
            ));
        }
        // A matching `param_dim` is not the same as a matching PARAMETERIZATION:
        // two models with different atom shapes can agree on `Σ_k p·d_k` and
        // disagree on every `(i, l) ↦ c`, which would silently reindex the
        // certificate's generators against the curvature. A block
        // representation carries the layout it was built in, so require it to
        // be the model's own.
        if let ResidualGaugeCurvature::OutputBlockRoots { layout, .. } = &curvature {
            let expected = FrameColumnLayout::for_frames(model.atoms.iter().map(|a| &a.frame))
                .ok_or_else(|| {
                    "residual_gauge: a block-structured curvature needs one shared output \
                     dimension across the fitted frames"
                        .to_string()
                })?;
            if *layout != expected {
                return Err(
                    "residual_gauge: the curvature's frame-column layout is not the fitted \
                     model's"
                        .to_string(),
                );
            }
        }
        // A non-finite curvature is a broken fit, not a certificate with an
        // unusual spectrum. The dense path used to get this refusal for free
        // from `FaerEigh::eigh`, which validates its input; `FaerSvd::svd` does
        // not, so a NaN would otherwise flow through as `sigma > tol == false`
        // and be reported as a rank-zero, fully-unpinned model — the most
        // permissive verdict the certificate can issue, from the least
        // trustworthy input. Refuse here, once, for every representation.
        if !curvature.is_finite() {
            return Err(
                "residual_gauge: streamed curvature contains a non-finite entry; the fitted \
                 decoder Jacobian or the row metric is not finite"
                    .to_string(),
            );
        }
        let root_rows = curvature.root_rows();
        match curvature {
            ResidualGaugeCurvature::OutputBlockRoots {
                roots,
                dense_rows,
                layout,
                ..
            } => Self::from_output_block_roots(roots, dense_rows, layout, root_rows),
            ResidualGaugeCurvature::DualRoot { root, .. } => {
                Self::from_dual_root(root, root_rows, param_dim)
            }
            ResidualGaugeCurvature::DenseGram { gram, .. } => {
                Self::from_gram(gram, root_rows, param_dim)
            }
        }
    }

    /// `H` as `p` output-coordinate block ROOTS: `R = ⊕_i R_i` up to a row
    /// permutation, so `R`'s singular values are the union of the blocks', and
    /// the rank decision is taken on them directly.
    ///
    /// A block that is identically zero is skipped rather than decomposed: its
    /// `D` singular values are exact zeros, which never clear a positive rank
    /// tolerance, and a fitted dictionary whose atoms touch a minority of the
    /// output coordinates has most of its blocks in that state.
    fn from_output_block_roots(
        roots: Array3<f64>,
        dense_rows: Array2<f64>,
        layout: FrameColumnLayout,
        root_rows: usize,
    ) -> Result<Self, String> {
        let p = layout.output_dim();
        let d = layout.block_dim();
        if roots.dim() != (p, d, d) {
            return Err(format!(
                "residual_gauge: curvature block roots have shape {:?} but the frame layout is \
                 ({p}, {d}, {d})",
                roots.dim()
            ));
        }
        if dense_rows.nrows() > 0 && dense_rows.ncols() != layout.param_dim() {
            return Err(format!(
                "residual_gauge: curvature dense rows have {} columns but param_dim = {}",
                dense_rows.ncols(),
                layout.param_dim()
            ));
        }
        if layout.param_dim() == 0 || root_rows == 0 {
            return Ok(Self::OutputBlockRoots {
                pinning_rank: 0,
                sigma_max_sq: 0.0,
                roots,
                dense_rows,
                layout,
            });
        }
        if dense_rows.nrows() == 0 {
            // No pin: `H` is exactly the direct sum of the blocks, so its
            // singular values are the union of theirs and the decision needs no
            // global instrument at all.
            let mut singular_values: Vec<f64> = Vec::with_capacity(p * d);
            for i in 0..p {
                let block = roots.slice(s![i, .., ..]);
                if block.iter().all(|v| *v == 0.0) {
                    continue;
                }
                if d == 1 {
                    // A 1-by-1 factor IS its own singular value; routing it
                    // through a decomposition would only add a validation pass.
                    singular_values.push(block[[0, 0]].abs());
                    continue;
                }
                let (_u, sv, _vt) = block.to_owned().svd(false, false).map_err(|e| {
                    format!("residual_gauge: SVD of curvature block root {i} failed: {e}")
                })?;
                singular_values.extend(sv.iter().copied());
            }
            let (sigma_max_sq, pinning_rank) =
                root_spectral_rank(&singular_values, root_rows, layout.param_dim());
            return Ok(Self::OutputBlockRoots {
                pinning_rank,
                sigma_max_sq,
                roots,
                dense_rows,
                layout,
            });
        }
        // With a pin installed the curvature is block diagonal PLUS a symmetric
        // update of rank at most `D`, whose rows are spread across every output
        // coordinate. That sum's spectrum is genuinely global -- a Givens
        // elimination of such a row fills in every block -- but its eigenvalue
        // COUNT above any shift is exactly computable from Sylvester's law of
        // inertia in `O(p*D*k^2)`, and a count above a shift is all either
        // consumer needs: the rank is the count above the squared tolerance, and
        // `lambda_max` is the shift at which the count reaches zero.
        let spectrum = frame_curvature::BlockPlusRowsSpectrum::new(&roots, &dense_rows, &layout)?;
        let sigma_max_sq = spectrum.lambda_max()?;
        let rank_tol = curvature_rank_tolerance(sigma_max_sq.sqrt(), root_rows, layout.param_dim());
        let pinning_rank = spectrum.count_above(rank_tol * rank_tol)?;
        Ok(Self::OutputBlockRoots {
            pinning_rank,
            sigma_max_sq,
            roots,
            dense_rows,
            layout,
        })
    }

    /// `H = RᵀR` with `m ≤ param_dim` rows: `R` has at most `m` nonzero
    /// singular values and `H` has `param_dim − m` structural zeros, so the
    /// whole decision comes from an `m`-sized decomposition of the root itself.
    fn from_dual_root(
        root: Array2<f64>,
        root_rows: usize,
        param_dim: usize,
    ) -> Result<Self, String> {
        if root.ncols() != param_dim {
            return Err(format!(
                "residual_gauge: curvature root has {} columns but param_dim = {param_dim}",
                root.ncols()
            ));
        }
        if param_dim == 0 || root_rows == 0 || root.nrows() == 0 {
            return Ok(Self::DualRoot {
                pinning_rank: 0,
                sigma_max_sq: 0.0,
                root,
            });
        }
        let (_u, sv, _vt) = root
            .svd(false, false)
            .map_err(|e| format!("residual_gauge: SVD of the curvature root failed: {e}"))?;
        let singular_values: Vec<f64> = sv.iter().copied().collect();
        let (sigma_max_sq, pinning_rank) =
            root_spectral_rank(&singular_values, root_rows, param_dim);
        Ok(Self::DualRoot {
            pinning_rank,
            sigma_max_sq,
            root,
        })
    }

    fn from_gram(gram: Array2<f64>, root_rows: usize, param_dim: usize) -> Result<Self, String> {
        if gram.nrows() != param_dim || gram.ncols() != param_dim {
            return Err(format!(
                "residual_gauge: curvature gram has shape ({}, {}) but param_dim = {param_dim}",
                gram.nrows(),
                gram.ncols()
            ));
        }
        if param_dim == 0 || root_rows == 0 {
            return Ok(Self::Gram {
                pinning_rank: 0,
                sigma_max_sq: 0.0,
                gram,
            });
        }
        let (evals, _) = gram.eigh(Side::Lower).map_err(|e| {
            format!("residual_gauge: eigendecomposition of curvature gram failed: {e}")
        })?;
        let spectrum: Vec<f64> = evals.iter().copied().collect();
        let (sigma_max_sq, pinning_rank) = gram_spectral_rank(&spectrum, root_rows, param_dim);
        Ok(Self::Gram {
            pinning_rank,
            sigma_max_sq,
            gram,
        })
    }

    fn pinning_rank(&self) -> usize {
        match self {
            Self::Gram { pinning_rank, .. }
            | Self::OutputBlockRoots { pinning_rank, .. }
            | Self::DualRoot { pinning_rank, .. } => *pinning_rank,
        }
    }

    fn sigma_max_sq(&self) -> f64 {
        match self {
            Self::Gram { sigma_max_sq, .. }
            | Self::OutputBlockRoots { sigma_max_sq, .. }
            | Self::DualRoot { sigma_max_sq, .. } => *sigma_max_sq,
        }
    }

    fn unit_generator_energy(&self, unit: &Array1<f64>) -> f64 {
        match self {
            Self::DualRoot { root, .. } => {
                let r_xi = root.dot(unit);
                r_xi.iter().map(|c| c * c).sum::<f64>()
            }
            Self::Gram { gram, .. } => {
                let h_xi = gram.dot(unit);
                unit.dot(&h_xi).max(0.0)
            }
            // `ξᵀHξ = Σ_i ‖R_i ξ_i‖²` — the sum runs over output coordinates
            // because `H` has no entries between two of them, and each term is
            // read off that block's root rather than its Gram. `p·D²` flops
            // instead of `(p·D)²`.
            Self::OutputBlockRoots {
                roots,
                dense_rows,
                layout,
                ..
            } => {
                let d = layout.block_dim();
                let mut xi = vec![0.0_f64; d];
                let mut total = 0.0_f64;
                for i in 0..layout.output_dim() {
                    layout.gather_output(unit.view(), i, &mut xi);
                    if xi.iter().all(|v| *v == 0.0) {
                        continue;
                    }
                    for a in 0..d {
                        let mut row = 0.0_f64;
                        for b in a..d {
                            row += roots[[i, a, b]] * xi[b];
                        }
                        total += row * row;
                    }
                }
                if dense_rows.nrows() > 0 {
                    let l_xi = dense_rows.dot(unit);
                    total += l_xi.iter().map(|c| c * c).sum::<f64>();
                }
                total.max(0.0)
            }
        }
    }
}

/// How this certificate reaches the curvature.
///
/// The three arms are not three algorithms — they are three *availabilities*.
/// The certificate's arithmetic is identical downstream of
/// [`CurvatureMeasurement`]; what differs is whether `H` was handed over
/// reduced, has to be built from a hand-assembled model's retained Jacobian
/// rows, or exists only as an operator that re-streams its own root (#2757).
enum CurvatureAccess<'a> {
    /// Build `R` from the model's retained per-row Jacobian blocks. The general
    /// path, for callers that hand-build a model whose Jacobian is not
    /// frame-structured.
    /// A curvature the producer already reduced.
    Reduced(CurvatureReduction),
    /// A curvature that is never materialized.
    Streamed(&'a dyn StreamedFrameCurvature),
}

/// Exactly what the certificate reads off the curvature, and nothing else.
///
/// Writing the three reads down as one struct is what makes the streamed route
/// checkable against the stored one: the two produce this and are then
/// indistinguishable to every line below. It is also the enumeration the issue
/// asked for — "enumerate the actual consumers first" — made structural, so a
/// fourth consumer cannot be added without deciding how it is streamed.
struct CurvatureMeasurement {
    pinning_rank: usize,
    pinning_rank_support: PinningRankSupport,
    /// `σ_max(R)² = λ_max(H)`, the stiffness scale every verdict divides by.
    sigma_max_sq: f64,
    /// `‖R ξ̂‖²` per enumerated generator, aligned with the generator list.
    /// `0.0` where the generator is structurally degenerate (no direction to
    /// measure along) or where the curvature is identically flat.
    energies: Vec<f64>,
    /// A short provenance clause for the summary line: empty for a stored
    /// reduction, and for a streamed one the certified relative residual of the
    /// Krylov `λ_max` solve. A number a reader cannot see the accuracy of is a
    /// number they cannot use.
    stiffness_note: String,
}

/// One enumerated symmetry generator, normalized in place.
///
/// The certificate needs the generator's DIRECTION (to measure curvature along)
/// and its NORM (which it reports, and which vetoes a structurally trivial
/// generator), and never the unnormalized vector again — so the enumeration
/// hands over the unit and the norm rather than the raw tangent. That matters at
/// production width: a dictionary of `K` atoms over `D` frame axes enumerates
/// `D(D−1)/2` frame rotations plus `K(K−1)/2` atom exchanges, each a
/// `param_dim`-long vector, so the enumerated list is the largest object the
/// certificate holds once the curvature is streamed. Keeping the raw vectors AND
/// their units alive at once would double it for no reader.
struct EnumeratedGenerator {
    family: GeneratorFamily,
    description: String,
    /// The #995 lowering-error tolerance scale: the largest `lowering_error`
    /// over the atoms this generator touches.
    lowering_error_scale: f64,
    /// `‖ξ‖` of the generator as enumerated.
    norm: f64,
    /// `ξ/‖ξ‖`, or `None` when the generator is structurally trivial (rotation
    /// of a rank-deficient frame, a zero swap). Such a generator carries no
    /// direction, cannot be normalized, and is not a residual freedom — see the
    /// veto in [`residual_gauge_inner`], which is the degenerate-tangent
    /// exclusion Theorem A requires.
    unit: Option<Array1<f64>>,
}

impl EnumeratedGenerator {
    /// Normalize one enumerated tangent in place.
    fn from_tangent(
        family: GeneratorFamily,
        tangent: Array1<f64>,
        description: String,
        lowering_error_scale: f64,
    ) -> Self {
        let norm = tangent.iter().map(|v| v * v).sum::<f64>().sqrt();
        let unit = if norm <= f64::MIN_POSITIVE {
            None
        } else {
            let mut unit = tangent;
            unit.mapv_inplace(|v| v / norm);
            Some(unit)
        };
        Self {
            family,
            description,
            lowering_error_scale,
            norm,
            unit,
        }
    }
}

/// The stored route: the reduction already holds the whole spectrum's worth of
/// decisions, so every read is a lookup or one `p·D²` contraction.
fn measure_reduced(
    curvature: &CurvatureReduction,
    gens: &[EnumeratedGenerator],
) -> CurvatureMeasurement {
    let sigma_max_sq = curvature.sigma_max_sq();
    let energies = gens
        .iter()
        .map(|generator| match &generator.unit {
            // An identically-flat curvature makes every fraction zero without
            // asking the curvature anything, which is also the only regime in
            // which the ratio would be `0/0`.
            Some(unit) if sigma_max_sq > f64::MIN_POSITIVE => {
                curvature.unit_generator_energy(unit)
            }
            _ => 0.0,
        })
        .collect();
    CurvatureMeasurement {
        pinning_rank: curvature.pinning_rank(),
        pinning_rank_support: PinningRankSupport::ParameterSpace,
        sigma_max_sq,
        energies,
        stiffness_note: String::new(),
    }
}

/// The streamed route: `λ_max` by a certified matrix-free Krylov solve, and the
/// generator energies EXACTLY, from one pass that folds `RΞ` into a `G × G`
/// upper-triangular factor.
///
/// The factor answers both remaining reads at once and it answers them the way
/// the rest of this module insists on — on singular values rather than on their
/// squares. Its column norms are the energies (`ξ_jᵀHξ_j = Σ_a T[a,j]²`,
/// exactly, because `TᵀT = ΞᵀHΞ`), and its singular values above the shared
/// `curvature_rank_tolerance` are the generator-span pinning rank.
fn measure_streamed(
    operator: &dyn StreamedFrameCurvature,
    gens: &[EnumeratedGenerator],
) -> Result<CurvatureMeasurement, String> {
    let lambda = streamed_lambda_max(operator)?;
    let sigma_max_sq = lambda.lambda_max;
    let present: Vec<usize> = gens
        .iter()
        .enumerate()
        .filter_map(|(index, generator)| generator.unit.as_ref().map(|_| index))
        .collect();
    let directions: Vec<ArrayView1<'_, f64>> = present
        .iter()
        .map(|&index| gens[index].unit.as_ref().expect("filtered to Some").view())
        .collect();
    let mut energies = vec![0.0_f64; gens.len()];
    let mut pinning_rank = 0usize;
    if !directions.is_empty() {
        let factor = operator.project_root(&directions)?;
        if factor.nrows() != directions.len() || factor.ncols() != directions.len() {
            return Err(format!(
                "residual_gauge: streamed generator factor is {:?}, expected a square factor \
                 over the {} enumerated directions",
                factor.dim(),
                directions.len()
            ));
        }
        if factor.iter().any(|v| !v.is_finite()) {
            return Err(
                "residual_gauge: streamed generator factor contains a non-finite entry; the \
                 fitted decoder Jacobian or the row metric is not finite"
                    .to_string(),
            );
        }
        for (column, &index) in present.iter().enumerate() {
            energies[index] = factor
                .column(column)
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .max(0.0);
        }
        let (_u, sv, _vt) = factor
            .svd(false, false)
            .map_err(|e| format!("residual_gauge: SVD of the streamed generator factor: {e}"))?;
        let singular_values: Vec<f64> = sv.iter().copied().collect();
        let (_projected_sigma_max_sq, rank) =
            root_spectral_rank(&singular_values, operator.root_rows(), directions.len());
        pinning_rank = rank;
    }
    // An identically-flat curvature is reported the way the stored route reports
    // it: every fraction zero, rather than `0/0` clamped.
    if !(sigma_max_sq > f64::MIN_POSITIVE) {
        energies.iter_mut().for_each(|e| *e = 0.0);
    }
    Ok(CurvatureMeasurement {
        pinning_rank,
        pinning_rank_support: PinningRankSupport::GeneratorSpan,
        sigma_max_sq,
        energies,
        stiffness_note: format!(
            "; stiffness scale streamed (Krylov relative residual {:.1e}, tr(H) = {:.6e})",
            lambda.relative_residual, lambda.trace
        ),
    })
}

/// Exact-orbit residual-gauge certificate with a pre-reduced streamed
/// curvature `H = RᵀR`.
///
/// This is the memory-scaled entry point for callers that can stream their
/// metric-whitened Jacobian rows into the reductions the certificate consumes,
/// instead of retaining every per-row `p × param_dim` Jacobian block. The
/// curvature must cover the same rows `stacked_curvature_root` would have
/// placed in `R`; its [`ResidualGaugeCurvature::root_rows`] is that row count,
/// which sets the rank tolerance scale.
///
/// The caller passes the *structure* it was able to build, not a dense matrix:
/// a curvature that is output-coordinate block diagonal (#2757) is reduced
/// through `p` `D × D` eigenproblems rather than one `(p·D)³` one, and a
/// curvature whose root has fewer rows than columns is reduced through its dual
/// Gram. Both are exact — same spectrum, same rank decision — so the variant is
/// a cost statement, never an accuracy one.
pub fn residual_gauge_exact_from_curvature(
    model: &FittedSaeManifold,
    views: &[Option<AtomParameterView>],
    penalty_ops: &[Option<OrbitPenaltyOperator>],
    curvature: ResidualGaugeCurvature,
) -> Result<ResidualGaugeReport, String> {
    let curvature = CurvatureReduction::from_curvature(curvature, model)?;
    let exact = residual_gauge_exact_inputs(model, views, penalty_ops)?;
    residual_gauge_inner(model, Some(exact), CurvatureAccess::Reduced(curvature))
}

/// Exact-orbit residual-gauge certificate against a curvature that is **never
/// materialized** (#2757).
///
/// This is the entry point for the branch where the per-row metric couples
/// output coordinates: there `H = Σ_n J_nᵀ M_n J_n` has no structure to store, so
/// at any production row count every materialized form of it costs
/// `min(root_rows, param_dim)²` scalars and an exact full spectrum costs the
/// cube. The certificate does not need one. It needs `ξᵀHξ` along each enumerated
/// generator and `λ_max(H)`, and `operator` supplies both by re-streaming its own
/// root: the first exactly in a single pass, the second to a certified relative
/// residual by a matrix-free Krylov solve.
///
/// The one thing this route does NOT resolve is the rank of `H` over the whole
/// parameter space, which is a full-spectrum question. The report says so —
/// [`ResidualGaugeReport::pinning_rank_support`] reads
/// [`PinningRankSupport::GeneratorSpan`] and the rank is then the exact rank of
/// `ΞᵀHΞ` over the enumerated generators, which is the comparison the pinning
/// rank exists to support. Every VERDICT is identical to the one a materialized
/// curvature would have produced, to the accuracy of the certified `λ_max`.
pub fn residual_gauge_exact_from_streamed(
    model: &FittedSaeManifold,
    views: &[Option<AtomParameterView>],
    penalty_ops: &[Option<OrbitPenaltyOperator>],
    operator: &dyn StreamedFrameCurvature,
) -> Result<ResidualGaugeReport, String> {
    let param_dim = model.param_dim();
    if operator.param_dim() != param_dim {
        return Err(format!(
            "residual_gauge: streamed curvature is over {} parameters but param_dim = \
             {param_dim}",
            operator.param_dim()
        ));
    }
    let exact = residual_gauge_exact_inputs(model, views, penalty_ops)?;
    residual_gauge_inner(model, Some(exact), CurvatureAccess::Streamed(operator))
}

fn residual_gauge_exact_inputs(
    model: &FittedSaeManifold,
    views: &[Option<AtomParameterView>],
    penalty_ops: &[Option<OrbitPenaltyOperator>],
) -> Result<(Vec<bool>, Vec<GeneratorVerdict>), String> {
    if views.len() != model.atoms.len() || penalty_ops.len() != model.atoms.len() {
        return Err(format!(
            "residual_gauge_exact: views ({}) and penalty_ops ({}) must align with atoms ({})",
            views.len(),
            penalty_ops.len(),
            model.atoms.len()
        ));
    }
    let mut mask = vec![false; model.atoms.len()];
    let mut exact_verdicts: Vec<GeneratorVerdict> = Vec::new();
    for (k, (atom, view)) in model.atoms.iter().zip(views.iter()).enumerate() {
        let Some(view) = view else { continue };
        exact_verdicts.extend(exact_orbit_verdicts(atom, view, penalty_ops[k].as_ref())?);
        mask[k] = true;
    }
    Ok((mask, exact_verdicts))
}

/// The symmetry generators of a fitted model, tagged by family, embedded in the
/// joint parameter vector, and each carrying its #995 lowering-error tolerance
/// scale.
///
/// `exact_mask` names the atoms whose within-atom families are realised exactly
/// (#998); those are skipped here, because the frame-space lift of a compensated
/// orbit measures compression rather than the symmetry, and the report must not
/// carry both a lossy and an exact verdict for the same group element.
///
/// Extracted from [`residual_gauge_inner`] so the certificate and its gates
/// enumerate from ONE definition: a gate that rebuilt this list for itself would
/// be checking the certificate against a second enumeration, and the alignment
/// between a generator and its measured energy is positional.
fn enumerate_generators(
    model: &FittedSaeManifold,
    exact_mask: Option<&[bool]>,
) -> Vec<EnumeratedGenerator> {
    let param_dim = model.param_dim();
    // The per-atom builders speak the atom's LOCAL flattened-frame coordinates
    // (length `frame.len()`); the certificate's rank arithmetic runs in the
    // joint parameter vector, so each local generator is embedded at its atom's
    // offset here. (Single-atom models have local == joint, which is why only
    // multi-atom models can expose a missed embedding.)
    let scale_of = |k: usize| -> f64 { model.atoms[k].lowering_error.clamp(0.0, 1.0) };
    let global_scale = (0..model.atoms.len()).map(scale_of).fold(0.0_f64, f64::max);
    let mut gens: Vec<EnumeratedGenerator> = Vec::new();
    for (k, atom) in model.atoms.iter().enumerate() {
        // Atoms whose within-atom families are realised exactly (#998) are
        // skipped here: the frame-space lift of a compensated orbit measures
        // compression, not the symmetry, and the report must not carry both a
        // lossy and an exact verdict for the same group element.
        if exact_mask.is_some_and(|mask| mask[k]) {
            continue;
        }
        let base = model.atom_offset(k);
        for (g, desc) in atom_isometry_generators(atom) {
            gens.push(EnumeratedGenerator::from_tangent(
                GeneratorFamily::IsomAtom,
                embed_local_generator(base, &g, param_dim),
                desc,
                scale_of(k),
            ));
        }
        for (g, desc) in equal_ard_rotation_generators(atom) {
            gens.push(EnumeratedGenerator::from_tangent(
                GeneratorFamily::EqualArdRotation,
                embed_local_generator(base, &g, param_dim),
                desc,
                scale_of(k),
            ));
        }
    }
    for (g, desc) in frame_rotation_generators(model) {
        // A global output rotation moves every atom's frame at once.
        gens.push(EnumeratedGenerator::from_tangent(
            GeneratorFamily::FrameRotation,
            g,
            desc,
            global_scale,
        ));
    }
    for (g, desc, ka, kb) in atom_permutation_generators(model) {
        gens.push(EnumeratedGenerator::from_tangent(
            GeneratorFamily::AtomPermutation,
            g,
            desc,
            scale_of(ka).max(scale_of(kb)),
        ));
    }

    gens
}

fn residual_gauge_inner(
    model: &FittedSaeManifold,
    exact: Option<(Vec<bool>, Vec<GeneratorVerdict>)>,
    access: CurvatureAccess<'_>,
) -> Result<ResidualGaugeReport, String> {
    let metric_provenance = model.metric.provenance();
    let (exact_mask, exact_verdicts) = match exact {
        Some((mask, verdicts)) => (Some(mask), verdicts),
        None => (None, Vec::new()),
    };

    // 1. Enumerate generators, tagged by family (see
    // [`enumerate_generators`], which is also what the gates rebuild the same
    // list from).
    let gens = enumerate_generators(model, exact_mask.as_deref());

    // 2. The curvature, reduced to exactly what step 3 reads off it. The
    // generators are enumerated FIRST because the streamed route measures their
    // energies in one pass over the root — a pass it can only take once it knows
    // which directions to project onto — while the stored routes hold enough of
    // the spectrum to answer either order. `CurvatureMeasurement` is what both
    // produce, so nothing below this line knows which route ran.
    let measurement = match access {
        CurvatureAccess::Streamed(operator) => measure_streamed(operator, &gens)?,
        CurvatureAccess::Reduced(curvature) => measure_reduced(&curvature, &gens),
    };
    let pinning_rank = measurement.pinning_rank;
    let pinning_rank_support = measurement.pinning_rank_support;
    let sigma_max_sq = measurement.sigma_max_sq;

    // The isometry pin is inactive ⇒ diffeomorphism-unpinned escalation.
    let diffeomorphism_unpinned = model.isometry_penalty_root.nrows() == 0;

    // 3. Per-generator flatness verdict: relative curvature vs the calibrated
    // tolerance.
    let mut verdicts: Vec<GeneratorVerdict> = Vec::with_capacity(gens.len());
    for (index, generator) in gens.iter().enumerate() {
        let family = generator.family;
        let description = &generator.description;
        let lowering_error_scale = generator.lowering_error_scale;
        // A structurally trivial generator (rotation of a rank-deficient frame,
        // zero swap) carries no direction — it cannot be a residual freedom.
        // Report it pinned with zero norm rather than as a spurious gauge.
        //
        // This IS the degenerate-tangent exclusion Theorem A requires, not a
        // defensive nicety: a generator whose realised tangent is `rank_eff = 0`
        // (`‖ξ‖ = 0`, no genuine direction in the Terracini border-block
        // Jacobian) is asymptotically CHEAP under the RLCT (real log canonical
        // threshold) accounting the underlying singular-learning-theory picture
        // uses — a genuine free direction pays `½` of RLCT per dimension, but a
        // null/degenerate direction pays nothing and buys nothing, so admitting
        // it as "unpinned" would claim a free identifiability dimension the
        // model does not actually have. Vetoing it here (forced `unpinned:
        // false`, `pinned_energy_fraction: 1.0`) is therefore a VALIDITY
        // CONDITION for the certificate's rank arithmetic — Theorem A's
        // tangent-space decomposition is only meaningful once degenerate
        // summands are excluded from it — not a heuristic tie-break.
        if generator.unit.is_none() {
            verdicts.push(GeneratorVerdict {
                family,
                description: description.clone(),
                unpinned: false,
                generator_norm: 0.0,
                pinned_energy_fraction: 1.0,
                lowering_error_scale,
                provenance: VerdictProvenance::CurvatureTest,
            });
            continue;
        }
        // Relative curvature fraction ‖R ξ̂‖² / σ_max(R)² of the unit
        // generator ξ̂ = ξ/‖ξ‖. Exactly flat directions score 0 even inside a
        // full-rank span (production fits!), where the previous
        // span-membership rule degenerated to all-pinned. A MIXED generator
        // (strictly interior fraction) above the tolerance is pinned: its
        // orbit costs objective, so the exact symmetry does not survive
        // (#980 Theorem-2 arm). The tolerance is calibrated by the #995
        // lowering-error scale: curvature the mean-frame compression cannot
        // distinguish from gauge motion must not be read as a pin — the
        // certificate refuses to claim resolution it does not have.
        let pinned_energy_fraction = if sigma_max_sq <= f64::MIN_POSITIVE {
            0.0
        } else {
            (measurement.energies[index] / sigma_max_sq).clamp(0.0, 1.0)
        };
        let tolerance = GENERATOR_FLAT_ENERGY_TOL.max(lowering_error_scale);
        let unpinned = pinned_energy_fraction <= tolerance;
        verdicts.push(GeneratorVerdict {
            family,
            description: description.clone(),
            unpinned,
            generator_norm: generator.norm,
            pinned_energy_fraction,
            lowering_error_scale,
            provenance: VerdictProvenance::CurvatureTest,
        });
    }

    // Exact-orbit verdicts (#998) join the report on equal footing: the
    // group signature, residual dimension, and Sym(F) check all range over
    // the union.
    verdicts.extend(exact_verdicts);

    // #1019 / #2022 — arc-length chart canonicalization records: for every
    // canonicalized d = 1 atom the continuous chart (reparameterization) freedom
    // is pinned BY CONSTRUCTION. As of #2022 the unit-speed representative of the
    // Diff(M) orbit is enforced IN-LOOP as an exact, image-frozen retraction at
    // every chart-refresh boundary (not merely selected once post-fit), so the
    // Diff(M) chart-speed generator is removed from the optimization THROUGHOUT
    // the fit. The certificate records it pinned with the PinnedByCanonicalization
    // provenance — distinct from curvature/penalty pinning, since no objective
    // resistance is measured (the direction is re-gauged away, not resisted) —
    // and names the surviving FINITE isometry group of the reference manifold.
    // Because the speed direction is removed every refresh it never enters
    // ker(H), so the d = 1 chart-speed generator no longer needs a curvature-test
    // / deflation verdict. The group's continuous part (the circle's U(1) shift)
    // is still enumerated and curvature-tested above; this record is the
    // chart-freedom downgrade itself.
    let mut canonicalized_charts = 0usize;
    let mut canonicalized_torus_charts = 0usize;
    let mut canonicalized_patch_charts = 0usize;
    let mut canonicalized_sphere_charts = 0usize;
    for atom in &model.atoms {
        if !atom.chart_canonicalized {
            continue;
        }
        let (pinned_to, residual_group) = match &atom.topology {
            AtomTopology::Circle | AtomTopology::Torus { latent_dim: 1 } => {
                canonicalized_charts += 1;
                ("arc length", "O(2) on S¹ (rotation + reflection)")
            }
            AtomTopology::EuclideanPatch { latent_dim: 1 } => {
                canonicalized_charts += 1;
                (
                    "arc length",
                    "reflection + translation of the unit interval",
                )
            }
            // #1019 stage 2: d = 2 torus charts are pinned post-fit to the
            // minimum-isometry-defect flow representative; the surviving chart
            // freedom is the isometry group of the flat square torus.
            AtomTopology::Torus { latent_dim: 2 } => {
                canonicalized_torus_charts += 1;
                (
                    "the isometry-flow canonical chart",
                    "Isom(T², flat) = U(1)² ⋊ D₄ (axis translations + axis swap/reflections)",
                )
            }
            AtomTopology::KleinBottle => {
                canonicalized_torus_charts += 1;
                (
                    "the Klein quotient gauge chart",
                    "Isom0(K, flat) = U(1)_theta with residual deck Z2",
                )
            }
            // #1019 free-chart arm: d = 2 free/patch (Euclidean-patch) charts
            // are pinned post-fit to the flat-reference minimum-anisotropy-
            // defect flow representative; the surviving chart freedom is the
            // isometry group of the flat plane.
            AtomTopology::EuclideanPatch { latent_dim: 2 } => {
                canonicalized_patch_charts += 1;
                (
                    "the flat-reference isometry-flow canonical chart",
                    "Isom(ℝ², flat) = O(2) ⋉ ℝ² (rotation + reflection + translation)",
                )
            }
            // #1019 sphere arm: d = 2 sphere (S²) charts are pinned post-fit to
            // the round-sphere conformal-boost minimum-isometry-defect flow,
            // which breaks the conformal (Möbius) moduli down to the round
            // sphere's isometry group; the surviving chart freedom is O(3).
            AtomTopology::Sphere => {
                canonicalized_sphere_charts += 1;
                (
                    "the round-sphere conformal-boost isometry-flow canonical chart",
                    "Isom(S², round) = O(3) (rotations + reflection)",
                )
            }
            // Canonicalization only ever applies to d = 1 charts, d = 2 torus,
            // d = 2 free/patch, and d = 2 sphere charts; a flag on any other
            // topology is structurally inconsistent and must not fabricate a
            // record.
            _ => continue,
        };
        verdicts.push(GeneratorVerdict {
            family: GeneratorFamily::ChartReparameterization,
            description: format!(
                "{}: chart pinned to {pinned_to} by post-fit canonicalization; \
                 residual chart freedom = {residual_group}",
                atom.name
            ),
            unpinned: false,
            generator_norm: 0.0,
            pinned_energy_fraction: 1.0,
            lowering_error_scale: 0.0,
            provenance: VerdictProvenance::PinnedByCanonicalization,
        });
    }

    let residual_gauge_dim = verdicts.iter().filter(|v| v.unpinned).count();

    // Sym(F)-triviality under any output-Fisher provenance — same-position
    // (`OutputFisher`) or downstream-influence (`OutputFisherDownstream`, #980).
    // Both behaviorally separate the atoms (the downstream metric strictly more,
    // since it sees far-future coupling the same-position metric misses), so the
    // permutation subgroup must be trivially pinned under either.
    let sym_f_trivial_under_output_fisher = if matches!(
        metric_provenance,
        MetricProvenance::OutputFisher { .. } | MetricProvenance::OutputFisherDownstream { .. }
    ) {
        let any_perm_unpinned = verdicts
            .iter()
            .any(|v| v.family == GeneratorFamily::AtomPermutation && v.unpinned);
        Some(!any_perm_unpinned)
    } else {
        None
    };

    let summary = format!(
        "residual gauge certificate (computed in metric {metric_provenance:?}): \
         pinning rank {pinning_rank} over the {}, {residual_gauge_dim} unpinned residual gauge \
         generator(s) of {} enumerated; group = {}{}{}{}",
        pinning_rank_support.label(),
        verdicts.len(),
        group_signature_of(&verdicts, diffeomorphism_unpinned),
        match sym_f_trivial_under_output_fisher {
            Some(true) => "; Sym(F) trivially pinned under OutputFisher",
            Some(false) => "; ⚠ Sym(F) NON-trivial under OutputFisher (certificate violation)",
            None => "",
        },
        if diffeomorphism_unpinned {
            "; ⚠ isometry pin inactive"
        } else {
            ""
        },
        measurement.stiffness_note,
    );
    let summary = if canonicalized_charts > 0 {
        format!(
            "{summary}; {canonicalized_charts} chart(s) pinned to arc length by post-fit \
             canonicalization (residual chart freedom = finite isometry group)"
        )
    } else {
        summary
    };
    let summary = if canonicalized_torus_charts > 0 {
        format!(
            "{summary}; {canonicalized_torus_charts} torus chart(s) pinned to the \
             isometry-flow canonical chart by post-fit canonicalization (residual chart \
             freedom = Isom(T², flat))"
        )
    } else {
        summary
    };
    let summary = if canonicalized_patch_charts > 0 {
        format!(
            "{summary}; {canonicalized_patch_charts} free/patch chart(s) pinned to the \
             flat-reference isometry-flow canonical chart by post-fit canonicalization \
             (residual chart freedom = Isom(ℝ², flat) = O(2) ⋉ ℝ²)"
        )
    } else {
        summary
    };
    let summary = if canonicalized_sphere_charts > 0 {
        format!(
            "{summary}; {canonicalized_sphere_charts} sphere chart(s) pinned to the \
             round-sphere conformal-boost isometry-flow canonical chart by post-fit \
             canonicalization (residual chart freedom = Isom(S², round) = O(3))"
        )
    } else {
        summary
    };

    Ok(ResidualGaugeReport {
        metric_provenance,
        generators: verdicts,
        pinning_rank,
        pinning_rank_support,
        residual_gauge_dim,
        diffeomorphism_unpinned,
        sym_f_trivial_under_output_fisher,
        // The #972 inner-rotation gauge is declared by the caller (it lives in
        // the (U_k, C_k) parameterization, not in the latent-frame coordinates
        // this certificate's generators are tangent to); frame-factored
        // dictionaries attach it via `with_frame_inner_rotation`.
        frame_inner_rotation: None,
        summary,
    })
}

/// The model's two certificates, shipped together (#984 work-plan step 2):
/// the residual-gauge report says what NO data could distinguish (the
/// symmetry group the fit is identified up to — a statement about the
/// model class), the structure certificate says what THIS data
/// established (the e-BH-confirmed subset of the dictionary's structural
/// claims, FDR ≤ α, valid at the caller's stopping time — a statement
/// about the world). A claim can fail both ways, and the failure modes
/// are independent: an atom can be perfectly identified yet statistically
/// unestablished, or strongly evidenced yet gauge-ambiguous with a twin.
///
/// These two reports are the nearest thing in this file to the "one
/// statistic, two notations" pairing (Cor F): `gauge` is a curvature-based,
/// model-class statement about identification, `structure` is a
/// likelihood/evidence-ledger statement about which claims the data has
/// established, and at the shared asymptotic limit (a well-identified atom
/// with enough data to certify its structure) the two MUST cohere — a
/// perfectly-identified atom that never accumulates structural evidence, or a
/// strongly-evidenced atom that is simultaneously gauge-ambiguous with an
/// exchangeable twin, is not a conflict to silently paper over. It is a
/// finite-sample MISSPECIFICATION SIGNAL: either the isometry/ARD pin that
/// `gauge` reads is not actually active in the fit that produced `structure`'s
/// evidence, or the ledger's shard evidence is itself confounded by the
/// unresolved symmetry `gauge` is reporting. Callers that see the two reports
/// disagree on the same atom should treat that disagreement as diagnostic
/// input, not as a tie to break by preferring one report over the other.
#[derive(Debug, Clone)]
pub struct DictionaryReport {
    /// What cannot be distinguished in principle ([`residual_gauge`]).
    pub gauge: ResidualGaugeReport,
    /// What the data established
    /// ([`gam_terms::inference::structure_evidence::StructureLedger::certify`]).
    pub structure: StructureCertificate,
    /// Per-atom inter-layer transport ladders (#1096). Empty when the caller
    /// has not supplied at least one atom's canonical coordinates across two or
    /// more layers. These reports are computed in the transport module's chart
    /// convention: circle coordinates are radians on `[0, 2π)`, while SAE
    /// canonical circle charts may use an arbitrary period and are rescaled by
    /// [`dictionary_report_with_transport_ladders`] before fitting.
    pub transport_ladders: Vec<AtomTransportLadderReport>,
    /// Per-atom post-PIRLS inference reports (#1097 penalty-debiased functional
    /// POINT summaries, #1103 split-LRT smooth-structure e-value), one entry
    /// per atom in [`FittedSaeManifold::atoms`] order. The #1099 per-atom
    /// curvature CI was removed under #1115 (a curvature BOUND is not an
    /// estimand and its SE conditioned on generated regressors); the surviving
    /// plug-in curvature point estimate lives on
    /// [`crate::manifold::CertificateInputs::per_atom_kappa_hat`],
    /// not here. Each report's
    /// fields are computed when the atom carries its fit-time
    /// [`AtomInnerFit`] byproducts and the relevant numerics succeed; otherwise
    /// the field is `None` (a bare certificate-only `FittedSaeManifold` — one
    /// built by the residual-gauge path with no fit harness — leaves every
    /// `inner_fit` `None`, so both fields are `None`).
    pub atom_inference: Vec<AtomInferenceReport>,
}

/// Canonical per-layer coordinates for one atom, ready for the #1096 transport
/// ladder integration.
///
/// The caller owns extraction from the SAE fit: `layers[i]`, `coords[i]`, and
/// `topologies[i]` describe the same atom at the same layer. This type keeps
/// that extraction outside [`dictionary_report`] so the core certificate can be
/// wired without reaching into `SaeManifoldTerm`.
#[derive(Debug, Clone)]
pub struct AtomTransportLadderInput {
    /// Index into [`FittedSaeManifold::atoms`].
    pub atom_index: usize,
    /// Layer labels in ladder order.
    pub layers: Vec<usize>,
    /// One canonical coordinate vector per layer, all over the same rows.
    pub coords: Vec<Array1<f64>>,
    /// One canonical chart topology per layer.
    pub topologies: Vec<CanonicalChartTopology>,
}

/// One atom's fitted inter-layer transport ladder.
#[derive(Debug, Clone)]
pub struct AtomTransportLadderReport {
    pub atom_index: usize,
    pub atom_name: String,
    pub report: TransportLadderReport,
}

/// #1097 penalty-debiased smooth-functional POINT summaries for one atom's
/// captured inner-decoder smooth (narrowed under #1115).
///
/// All three functionals are *linear* in the atom's fitted coefficient vector
/// `β_{k,j}`, so each is one-step penalty-debiased through the SAME penalized
/// Hessian the identifiability certificate's curvature sees
/// ([`AtomInnerFit::penalized_hessian`]) by routing the functional gradient,
/// the per-row scores, and the penalty gradient `S̃_k β` through
/// [`debias_with_dense_hessian`]. Only the resulting POINT estimates (plug-in,
/// penalty-debiased, removed bias) are kept; the influence-function SE is
/// discarded because it conditions on the generated latent coordinates `t̂` /
/// assignment `â` as if known and so under-covers (see
/// [`AtomFunctionalReport`] for the full argument). A non-SPD Hessian or a
/// degenerate functional (empty design, non-finite gradient) leaves the
/// offending field `None`; the other two still report.
fn atom_functional_report(fit: &AtomInnerFit) -> AtomFunctionalReport {
    let penalty_beta = fit.penalty.dot(&fit.beta);

    // A small closed-form helper: build the Riesz input for a functional
    // gradient and penalty-debias it through the fitted penalized Hessian, then
    // KEEP ONLY the point estimates (the SE is not honest here — #1115). The
    // Riesz layer's own `EstimationError` is collapsed into `None` — a numerical
    // refusal is a missing field, not a poisoned report.
    let debias = |functional_gradient: Array1<f64>| -> Option<AtomFunctionalEstimate> {
        let input = RieszInput {
            beta: fit.beta.view(),
            functional_gradient: functional_gradient.view(),
            row_scores: fit.row_scores.view(),
            penalty_beta: penalty_beta.view(),
            leverage: None,
        };
        debias_with_dense_hessian(&input, fit.penalized_hessian.view())
            .ok()
            .map(|r| AtomFunctionalEstimate {
                theta_plugin: r.theta_plugin,
                theta_onestep: r.theta_onestep,
                penalty_bias: r.penalty_bias,
            })
    };

    // Peak-vs-mode contrast g(t_peak) − g(t_mode): the linear functional whose
    // gradient is the difference of the two design rows.
    let peak_contrast = SmoothFunctional::Contrast {
        design_row_a: fit.peak_design_row.view(),
        design_row_b: fit.mode_design_row.view(),
    }
    .gradient()
    .ok()
    .and_then(debias);

    // E_data[g(t_i)]: the mass-weighted average decoder value over active rows.
    let average_value = SmoothFunctional::AverageValue {
        value_design: fit.design.view(),
        weights: Some(fit.weights.view()),
    }
    .gradient()
    .ok()
    .and_then(debias);

    // ‖E_data[∂g/∂t]‖ along the leading latent axis: the mass-weighted average
    // of the derivative-design rows (the Gauss–Newton weights `w_i = a_ik²` are
    // the data measure over the atom's active rows). This is the conditional-
    // on-fit decoder-VARIATION norm, not a population marginal slope.
    let decoder_variation_norm = SmoothFunctional::AverageDerivative {
        derivative_design: fit.derivative_design.view(),
        weights: Some(fit.weights.view()),
    }
    .gradient()
    .ok()
    .and_then(debias);

    AtomFunctionalReport {
        peak_contrast,
        average_value,
        decoder_variation_norm,
    }
}

/// #1103 Any-n-valid structure evidence that one atom's inner smooth is
/// non-constant, via the split-likelihood-ratio e-value.
///
/// The inner decoder smooth is the Gaussian-identity penalized WLS fit
/// `a_ik · Φ_k(t)ᵀ β_{k,j}` with dispersion `φ = `[`AtomInnerFit::dispersion`],
/// working response `z_i` reconstructed from the captured per-row scores. H0 is
/// "the smooth is constant": only the intercept column 0 is free.
///
/// We compute the universal-inference e-value the atom-birth gate
/// ([`gam_terms::inference::structure_evidence::split_likelihood_log_e_value`]) uses:
///
/// * Split the active rows deterministically into an ESTIMATION fold (even
///   index) and an EVALUATION fold (odd index).
/// * On the estimation fold, fit the penalized smooth (the alternative) by
///   `β̂ = (ΦᵀWΦ + S)⁻¹ ΦᵀW z` — any fitter is admissible; zero conditions.
/// * On the evaluation fold, score the Gaussian log-likelihood under that
///   prefit alternative, and the SUPREMUM of the evaluation-fold log-likelihood
///   over the null class (the constant fit = weighted-mean response refit on the
///   eval fold — the honest constrained sup on D₀).
/// * `log E = ℓ_alt(D₀) − sup_{H0} ℓ(D₀)`, with `E_{H0}[E] ≤ 1` exactly.
///
/// The dispersion `φ` is held fixed at the fitted reconstruction dispersion in
/// both log-likelihoods so it cancels structurally and the e-value isolates the
/// mean-curvature evidence. Returns `None` when the design has no curvature
/// column (`M_k ≤ 1`), either fold is empty, or the inner Gram is not SPD.
fn atom_smooth_significance(fit: &AtomInnerFit) -> Option<AtomSmoothSignificance> {
    let m = fit.design.ncols();
    if m <= 1 || fit.beta.len() != m {
        // No curvature column: the constant null IS the full model — there is no
        // non-constant alternative to earn an e-value.
        return None;
    }
    let n = fit.design.nrows();
    if n == 0 || fit.weights.len() != n || fit.row_scores.nrows() != n {
        return None;
    }
    let phi = if fit.dispersion.is_finite() && fit.dispersion > 0.0 {
        fit.dispersion
    } else {
        return None;
    };

    // Per-row working response z_i = μ̂_i + r_i, reconstructing the scalar
    // residual r_i from the captured score projected onto the design row
    // (s_iᵀ Φ_i = −w_i r_i ‖Φ_i‖² / φ ⇒ r_i). Same reconstruction the previous
    // deviance path used; here it feeds the two folds' likelihoods.
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mu_hat = fit.design.row(i).dot(&fit.beta);
        let w_i = fit.weights[i];
        let phi_row = fit.design.row(i);
        let phi_norm_sq = phi_row.dot(&phi_row);
        let r_i = if w_i > 0.0 && phi_norm_sq > 0.0 {
            let s_dot_phi = fit.row_scores.row(i).dot(&phi_row);
            -phi * s_dot_phi / (w_i * phi_norm_sq)
        } else {
            0.0
        };
        z[i] = mu_hat + r_i;
    }

    // Deterministic estimation/evaluation split by row parity.
    let est: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let eval: Vec<usize> = (0..n).filter(|i| i % 2 == 1).collect();
    if est.is_empty() || eval.is_empty() {
        return None;
    }

    // Penalized smooth fit on the estimation fold: β̂ = (ΦᵀWΦ + S)⁻¹ ΦᵀW z.
    let mut a_gram = fit.penalty.clone();
    let mut b = Array1::<f64>::zeros(m);
    for &i in &est {
        let w_i = fit.weights[i];
        if !(w_i > 0.0) {
            continue;
        }
        let row = fit.design.row(i);
        for r in 0..m {
            let xr = row[r];
            if xr == 0.0 {
                continue;
            }
            b[r] += w_i * xr * z[i];
            for c in 0..m {
                a_gram[[r, c]] += w_i * xr * row[c];
            }
        }
    }
    let beta_alt = a_gram.cholesky(Side::Lower).ok()?.solvevec(&b);

    // Null sup on the EVALUATION fold: the weighted-mean response (the constant
    // fit's MLE on D₀, the honest constrained sup over the null class).
    let mut eval_mass = 0.0_f64;
    let mut eval_wz = 0.0_f64;
    for &i in &eval {
        let w_i = fit.weights[i];
        eval_mass += w_i;
        eval_wz += w_i * z[i];
    }
    if !(eval_mass > 0.0) {
        return None;
    }
    let null_mean = eval_wz / eval_mass;

    // Gaussian log-likelihoods on the evaluation fold at fixed dispersion φ;
    // the −½ log(2πφ) and weight-log terms are identical under both models, so
    // log E = −(½/φ) [ Σ w(z − μ_alt)² − Σ w(z − μ_null)² ].
    let mut sse_alt = 0.0_f64;
    let mut sse_null = 0.0_f64;
    for &i in &eval {
        let w_i = fit.weights[i];
        let mu_alt = fit.design.row(i).dot(&beta_alt);
        let r_alt = z[i] - mu_alt;
        let r_null = z[i] - null_mean;
        sse_alt += w_i * r_alt * r_alt;
        sse_null += w_i * r_null * r_null;
    }
    let log_lik_alt = -0.5 * sse_alt / phi;
    let log_lik_null_sup = -0.5 * sse_null / phi;
    let log_e = gam_terms::inference::structure_evidence::split_likelihood_log_e_value(
        log_lik_alt,
        log_lik_null_sup,
    )
    .ok()?;
    if !log_e.is_finite() {
        return None;
    }

    Some(AtomSmoothSignificance {
        log_e_nonconstant: Some(log_e),
    })
}

/// Assemble the post-PIRLS inference reports for every atom, reusing the
/// per-atom [`AtomInnerFit`] harvested at fit time.
///
/// * #1097 penalty-debiased functional POINT summaries and the #1103 split-LRT
///   smooth-structure e-value are computed from the captured inner-decoder
///   smooth (design, penalized Hessian, row scores, roughness Gram) — they need
///   only the fixed fitted snapshot.
/// * The #1099 per-atom curvature *confidence interval* was removed under #1115:
///   a sup-norm curvature BOUND is not an estimand with a profiled criterion,
///   and its delta-method SE conditioned on generated latent coordinates as if
///   known. The plug-in curvature point estimate survives on
///   [`crate::manifold::CertificateInputs::per_atom_kappa_hat`] (the
///   #1008 empirical curved-dictionary report), not on this report.
pub(crate) fn atom_inference_reports(model: &FittedSaeManifold) -> Vec<AtomInferenceReport> {
    model
        .atoms
        .iter()
        .enumerate()
        .map(|(atom_index, atom)| {
            let (functionals, smooth_significance) = match &atom.inner_fit {
                Some(fit) => (
                    Some(atom_functional_report(fit)),
                    atom_smooth_significance(fit),
                ),
                None => (None, None),
            };
            AtomInferenceReport {
                atom_index,
                atom_name: atom.name.clone(),
                functionals,
                smooth_significance,
            }
        })
        .collect()
}

// --- #1100: closed-loop probe runner FFI ---------------------------------
// Top-level entry points exposing the steering→structure-evidence probe loop
// (`crate::inference::probe_runner::ProbeRunner`) beside `dictionary_report`, so
// the Python driver can design and absorb interventional probes against the same
// fitted term and evidence ledger the certificate is built from.

// ----------------------------------------------------------------------------
// #1102 cross-checkpoint atom-dynamics FFI entry (new top-level block).
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    fn orbit_fixture(
        topology: AtomTopology,
        coords: Array2<f64>,
    ) -> (FittedAtom, AtomParameterView) {
        let n = coords.nrows();
        let d = coords.ncols();
        (
            FittedAtom {
                name: "quotient".to_string(),
                topology,
                frame: Array2::zeros((2, d)),
                ard_variances: None,
                lowering_error: 0.0,
                chart_canonicalized: false,
                inner_fit: None,
            },
            AtomParameterView {
                basis_values: Array2::ones((n, 1)),
                basis_jacobian: Array3::zeros((n, 1, d)),
                decoder: Array2::zeros((1, 2)),
                coords,
                activations: Array1::ones(n),
                basis_second_jet: None,
            },
        )
    }

    #[test]
    fn quotient_exact_orbits_have_the_correct_connected_gauge_dimensions() {
        let (klein, klein_view) = orbit_fixture(
            AtomTopology::KleinBottle,
            Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.4, -0.3]).unwrap(),
        );
        let klein_fields = exact_orbit_fields(&klein, &klein_view).unwrap();
        assert_eq!(klein_fields.len(), 1);
        assert!(
            klein_fields[0]
                .1
                .column(0)
                .iter()
                .all(|value| *value == 1.0)
        );
        assert!(
            klein_fields[0]
                .1
                .column(1)
                .iter()
                .all(|value| *value == 0.0)
        );

        // `RP²` on the ambient cover: the deck twin is the ANTIPODE `-u`, and
        // `K_a(-u) = e_a × (-u) = -K_a(u)`, so ALL THREE axes flip. On the
        // superseded `(lat, lon)` cover the same statement was the lopsided
        // `[-1, +1]`, because that chart's deck was `(-lat, lon + π)`.
        let u = [0.36_f64, 0.48, 0.8];
        let (rp2, rp2_view) = orbit_fixture(
            AtomTopology::ProjectivePlane,
            Array2::from_shape_vec((2, 3), vec![u[0], u[1], u[2], -u[0], -u[1], -u[2]]).unwrap(),
        );
        let rp2_fields = exact_orbit_fields(&rp2, &rp2_view).unwrap();
        assert_eq!(rp2_fields.len(), 3);
        for (_, field, _) in rp2_fields {
            for axis in 0..3 {
                assert!(
                    (field[[1, axis]] + field[[0, axis]]).abs() <= 1.0e-12,
                    "the antipodal deck must negate every ambient component of a Killing field"
                );
            }
        }

        // The SPHERE now reaches this path too, which is the point of the
        // change: it previously emitted NOTHING here and was routed to a weaker
        // frame path, so its own quotient carried a stronger certificate than it
        // did. Its three generators are the same `e_a × u`, and they must be
        // exactly tangent — a Killing field of `S²` cannot have a radial part.
        let (sphere, sphere_view) = orbit_fixture(
            AtomTopology::Sphere,
            Array2::from_shape_vec((2, 3), vec![u[0], u[1], u[2], 0.0, 0.0, 1.0]).unwrap(),
        );
        let sphere_fields = exact_orbit_fields(&sphere, &sphere_view).unwrap();
        assert_eq!(
            sphere_fields.len(),
            3,
            "Isom(S²) = O(3) has three connected generators; the sphere used to emit zero"
        );
        for (_, field, _) in &sphere_fields {
            for row in 0..2 {
                let point = if row == 0 { u } else { [0.0, 0.0, 1.0] };
                let radial: f64 = (0..3).map(|axis| field[[row, axis]] * point[axis]).sum();
                assert!(
                    radial.abs() <= 1.0e-12,
                    "a Killing field of S² is tangent, so u·K(u) = 0; got {radial:.3e}"
                );
            }
        }
    }

    /// The ambient cover has no pole, so there is nothing left to refuse. This
    /// replaces `projective_orbit_certificate_refuses_the_cover_pole`, which
    /// asserted that the `(lat, lon)` cover REJECTED `lat = π/2` because
    /// longitude has a nontrivial stabiliser there. The ambient generators are
    /// defined at every unit vector, so the former failure point is now an
    /// ordinary sample — and asserting that is the honest successor to
    /// asserting the refusal.
    #[test]
    fn ambient_cover_has_no_pole_to_refuse() {
        let (rp2, view) = orbit_fixture(
            AtomTopology::ProjectivePlane,
            Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 1.0]).unwrap(),
        );
        let fields = exact_orbit_fields(&rp2, &view)
            .expect("the former pole is an ordinary point on the ambient cover");
        assert_eq!(fields.len(), 3);
        // At the north pole `e_z × u = 0`, which is correct and not a failure:
        // the rotation about `z` fixes that point. The other two generators are
        // non-degenerate there, so the orbit is still two-dimensional.
        let magnitudes: Vec<f64> = fields
            .iter()
            .map(|(_, field, _)| (0..3).map(|a| field[[0, a]].abs()).fold(0.0, f64::max))
            .collect();
        assert_eq!(
            magnitudes.iter().filter(|m| **m > 1.0e-12).count(),
            2,
            "at a pole exactly one generator (the rotation fixing it) vanishes"
        );
    }

    /// #1097: the per-atom penalty-debiased functional point summaries must
    /// reproduce the exact linear functionals of the fitted decoder smooth
    /// (plug-in) and a finite debiased value, on a synthetic atom whose inner
    /// smooth is an analytic polynomial. No SE/CI is asserted — none is reported
    /// (#1115).
    #[test]
    fn atom_functional_report_recovers_known_functionals() {
        use ndarray::{Array1 as A1, Array2 as A2};
        // Polynomial basis Φ(t) = [1, t, t²] on a uniform active grid; the atom's
        // fitted smooth is g(t) = β·Φ(t) with a known β. We assemble a genuine
        // penalized-WLS AtomInnerFit (unit weights, identity-ish penalty) so the
        // Riesz path runs end to end.
        let n = 40usize;
        let m = 3usize;
        let beta = A1::from(vec![0.5_f64, -1.0, 2.0]);
        let mut design = A2::<f64>::zeros((n, m));
        let mut derivative_design = A2::<f64>::zeros((n, m));
        let mut weights = A1::<f64>::ones(n);
        let mut t = vec![0.0_f64; n];
        for i in 0..n {
            let ti = i as f64 / (n - 1) as f64;
            t[i] = ti;
            design[[i, 0]] = 1.0;
            design[[i, 1]] = ti;
            design[[i, 2]] = ti * ti;
            // dΦ/dt = [0, 1, 2t].
            derivative_design[[i, 0]] = 0.0;
            derivative_design[[i, 1]] = 1.0;
            derivative_design[[i, 2]] = 2.0 * ti;
            weights[i] = 1.0;
        }
        let dispersion = 1.0_f64;
        // Working response equals the fitted curve so residuals are zero → the
        // plug-in is exactly the analytic functional of β; scores are zero.
        let row_scores = A2::<f64>::zeros((n, m));
        // Penalty S = small ridge on curvature column only; penalized Hessian
        // H = ΦᵀWΦ + S.
        let mut penalty = A2::<f64>::zeros((m, m));
        penalty[[2, 2]] = 1e-3;
        let mut xtwx = A2::<f64>::zeros((m, m));
        for i in 0..n {
            for a in 0..m {
                for b in 0..m {
                    xtwx[[a, b]] += weights[i] * design[[i, a]] * design[[i, b]];
                }
            }
        }
        let penalized_hessian = &xtwx + &penalty;
        // Peak: |g| largest; mode: pick endpoints to give a known contrast.
        let mut peak_slot = 0usize;
        let mut peak_val = -1.0;
        for i in 0..n {
            let g = design.row(i).dot(&beta).abs();
            if g > peak_val {
                peak_val = g;
                peak_slot = i;
            }
        }
        let peak_design_row = design.row(peak_slot).to_owned();
        let mode_design_row = design.row(0).to_owned();

        let fit = AtomInnerFit {
            design: design.clone(),
            derivative_design: derivative_design.clone(),
            beta: beta.clone(),
            penalty,
            penalized_hessian,
            row_scores,
            weights: weights.clone(),
            dispersion,
            peak_design_row: peak_design_row.clone(),
            mode_design_row: mode_design_row.clone(),
        };

        let report = atom_functional_report(&fit);

        // Average value E_w[g] = mean_i β·Φ(t_i): exact plug-in match.
        let av = report.average_value.expect("average value");
        let expected_av: f64 = (0..n).map(|i| design.row(i).dot(&beta)).sum::<f64>() / n as f64;
        assert!(
            (av.theta_plugin - expected_av).abs() < 1e-9,
            "average value plug-in {} vs expected {}",
            av.theta_plugin,
            expected_av
        );
        // Point summary only: the debiased value is finite (no SE/CI is
        // reported by design — #1115).
        assert!(
            av.theta_onestep.is_finite(),
            "average-value debiased finite"
        );

        // Decoder-variation norm (conditional on fit): g'(t) = β1 + 2β2 t, mean
        // over the grid is β1 + 2β2 * mean(t). The functional gradient is the
        // mean derivative row; its plug-in is exactly that scalar. This is the
        // descriptive variation of the fitted curve, not a population marginal
        // slope.
        let ad = report
            .decoder_variation_norm
            .expect("decoder variation norm");
        let mean_t: f64 = t.iter().sum::<f64>() / n as f64;
        let expected_ad = beta[1] + 2.0 * beta[2] * mean_t;
        assert!(
            (ad.theta_plugin - expected_ad).abs() < 1e-9,
            "decoder variation plug-in {} vs expected {}",
            ad.theta_plugin,
            expected_ad
        );

        // Peak-vs-mode contrast g(t_peak) − g(t_mode): exact plug-in.
        let pc = report.peak_contrast.expect("peak contrast");
        let expected_pc = peak_design_row.dot(&beta) - mode_design_row.dot(&beta);
        assert!(
            (pc.theta_plugin - expected_pc).abs() < 1e-9,
            "peak contrast plug-in {} vs expected {}",
            pc.theta_plugin,
            expected_pc
        );
    }

    #[test]
    fn mechanism_sparsity_jacobian_value_matches_closed_form() {
        let w = array![[3.0_f64, 0.0], [4.0, 0.0]]; // col0 norm=5, col1 norm=0
        let pen = MechanismSparsityJacobian::new(1.0, 1.0e-8).unwrap();
        let (v, _g) = pen.value_and_grad(w.view());
        assert!((v - 5.0).abs() < 1e-6, "value {v} expected ≈5");
    }

    #[test]
    fn mechanism_sparsity_jacobian_grad_matches_finite_diff() {
        let w = array![[0.5_f64, -1.2, 0.3], [1.1, 0.4, -0.7]];
        let pen = MechanismSparsityJacobian::new(2.5, 1.0e-6).unwrap();
        let (_, g) = pen.value_and_grad(w.view());
        let h = 1.0e-5;
        for i in 0..w.nrows() {
            for j in 0..w.ncols() {
                let mut wp = w.clone();
                let mut wm = w.clone();
                wp[[i, j]] += h;
                wm[[i, j]] -= h;
                let (vp, _) = pen.value_and_grad(wp.view());
                let (vm, _) = pen.value_and_grad(wm.view());
                let fd = (vp - vm) / (2.0 * h);
                assert!(
                    (g[[i, j]] - fd).abs() < 1e-4,
                    "grad[{i},{j}] = {} vs fd {}",
                    g[[i, j]],
                    fd
                );
            }
        }
    }

    #[test]
    fn mechanism_sparsity_jacobian_rejects_bad_input() {
        assert!(MechanismSparsityJacobian::new(-1.0, 1e-6).is_err());
        assert!(MechanismSparsityJacobian::new(1.0, 0.0).is_err());
    }

    /// Build a `(n, d)` `(mean, scale)` pair whose stacked signature
    /// `[μ ‖ log σ]` has full rank `2d` (so it satisfies the Khemakhem
    /// Theorem 1 precondition baked into `ConditionalPriorIvae::new`).
    ///
    /// Each per-column function is given a distinct *frequency* (not a
    /// shared frequency with a column-dependent phase) so the resulting
    /// `2d` columns are genuinely linearly independent. `sin(ω·t + φ)`
    /// with a shared `ω` lives in the 2-dimensional span of `{sin(ω t),
    /// cos(ω t)}`, so the earlier `sin(0.7t + 0.3c)` / `cos(0.5t + 0.9c)`
    /// fixture only ever produced rank `≤ 4`, no matter how many `d`
    /// columns it built. Distinct frequencies push each column into its
    /// own subspace, so for `n ≥ 2d + 1` the SVD of `[μ ‖ log σ]` has
    /// `2d` non-trivial singular values.
    fn ivae_precondition_pair(n: usize, d: usize) -> (Array2<f64>, Array2<f64>) {
        assert!(n >= 2 * d + 1, "need at least 2d+1 rows");
        let mut mean = Array2::<f64>::zeros((n, d));
        let mut scale = Array2::<f64>::from_elem((n, d), 1.0);
        for r in 0..n {
            let t = r as f64 / (n as f64 - 1.0);
            for c in 0..d {
                let omega = (c + 1) as f64;
                mean[[r, c]] = (std::f64::consts::PI * omega * t).sin();
                scale[[r, c]] = (0.4 * (std::f64::consts::PI * omega * t).cos()).exp();
            }
        }
        (mean, scale)
    }

    #[test]
    fn conditional_prior_ivae_zero_mean_unit_scale_matches_standard_gaussian() {
        // Use varying (μ, log σ) so the identifiability precondition holds,
        // then evaluate at a `t` that matches `μ` to recover the closed-form
        // Gaussian normaliser ½·n·d·log 2π + Σ log σ.
        let n = 7;
        let d = 3;
        let (mean, scale) = ivae_precondition_pair(n, d);
        let t = mean.clone();
        let log_norm: f64 = scale.iter().map(|s| s.ln()).sum();
        let pen = ConditionalPriorIvae::new(mean, scale, 1.0).unwrap();
        let (v, g) = pen.value_and_grad(t.view());
        let expected = log_norm + 0.5 * (n * d) as f64 * (2.0 * std::f64::consts::PI).ln();
        assert!(
            (v - expected).abs() < 1e-9,
            "value {v} vs expected {expected}"
        );
        for &gv in g.iter() {
            assert!(gv.abs() < 1e-12);
        }
    }

    #[test]
    fn derive_ivae_aux_scale_matches_numpy_formula() {
        let aux = array![
            [1.0_f64, 2.0, 5.0],
            [2.5, -1.0, 5.0],
            [-0.5, 4.0, 5.0],
            [3.0, 0.5, 5.0],
        ];
        let scale = derive_ivae_aux_scale(aux.view(), 0.4, 1.0);
        let expected = array![
            [0.8694483838365188_f64, 1.2655369552163311, 1.0],
            [1.2831253020529474, 0.6734640022297076, 1.0],
            [0.6982995444784196, 1.4877547448810657, 1.0],
            [1.376498244165498, 0.7443881970243742, 1.0],
        ];
        let mut max_abs = 0.0_f64;
        for (actual, reference) in scale.iter().zip(expected.iter()) {
            max_abs = max_abs.max((actual - reference).abs());
        }
        assert!(
            max_abs < 1.0e-12,
            "Rust iVAE aux-scale derivation differs from the old NumPy formula by {max_abs}"
        );
    }

    #[test]
    fn conditional_prior_ivae_grad_matches_finite_diff() {
        let (mean, scale) = ivae_precondition_pair(5, 2);
        let mut t = mean.clone();
        for r in 0..5 {
            t[[r, 0]] += 0.4;
            t[[r, 1]] -= 0.3;
        }
        let pen = ConditionalPriorIvae::new(mean, scale, 1.7).unwrap();
        let (_, g) = pen.value_and_grad(t.view());
        let h = 1.0e-5;
        for i in 0..t.nrows() {
            for j in 0..t.ncols() {
                let mut tp = t.clone();
                let mut tm = t.clone();
                tp[[i, j]] += h;
                tm[[i, j]] -= h;
                let vp = pen.value(tp.view());
                let vm = pen.value(tm.view());
                let fd = (vp - vm) / (2.0 * h);
                assert!((g[[i, j]] - fd).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn conditional_prior_ivae_rejects_nonpositive_scale() {
        let mean = Array2::<f64>::zeros((2, 2));
        let mut scale = Array2::<f64>::ones((2, 2));
        scale[[0, 0]] = -0.1;
        assert!(ConditionalPriorIvae::new(mean, scale, 1.0).is_err());
    }

    #[test]
    fn conditional_prior_ivae_accepts_when_signature_full_rank() {
        let (mean, scale) = ivae_precondition_pair(7, 3);
        let result = ConditionalPriorIvae::new(mean, scale, 1.0);
        assert!(
            result.is_ok(),
            "full-rank signature should satisfy Khemakhem Theorem 1, got {:?}",
            result.err(),
        );
    }

    #[test]
    fn conditional_prior_ivae_rejects_trivial_constant_prior() {
        // All rows identical → unconditional N(μ, σ²), non-identifiable.
        let n = 9;
        let d = 3;
        let mean = Array2::<f64>::from_elem((n, d), 0.25);
        let scale = Array2::<f64>::from_elem((n, d), 1.5);
        let err = ConditionalPriorIvae::new(mean, scale, 1.0).unwrap_err();
        assert!(
            err.contains("trivial unconditional") && err.contains("Khemakhem"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn conditional_prior_ivae_rejects_too_few_auxiliary_states() {
        // n_rows = 4, latent_dim = 3 → need ≥ 2·3+1 = 7 rows.
        let (full_mean, full_scale) = ivae_precondition_pair(7, 3);
        let mean = full_mean.slice(s![..4, ..]).to_owned();
        let scale = full_scale.slice(s![..4, ..]).to_owned();
        let err = ConditionalPriorIvae::new(mean, scale, 1.0).unwrap_err();
        assert!(
            err.contains("2k+1") && err.contains("Khemakhem"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn conditional_prior_ivae_rejects_rank_deficient_signature() {
        // Enough rows (n = 9 ≥ 2·3+1 = 7) and rows are NOT all identical,
        // but the stacked [μ ‖ log σ] matrix lies in a strict subspace of
        // ℝ^{2d}: column 0 of μ equals column 0 of log σ, and columns 1,2
        // of both μ and σ are zero / one. So the signature has rank 1, far
        // below the required 2·3 = 6.
        let n = 9;
        let d = 3;
        let mut mean = Array2::<f64>::zeros((n, d));
        let mut scale = Array2::<f64>::from_elem((n, d), 1.0);
        for r in 0..n {
            let v = ((r as f64) * 0.5).sin();
            mean[[r, 0]] = v;
            scale[[r, 0]] = v.exp(); // log σ column 0 = v = μ column 0
        }
        let err = ConditionalPriorIvae::new(mean, scale, 1.0).unwrap_err();
        assert!(
            err.contains("numerical rank") && err.contains("Khemakhem"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn identifiable_factor_evidence_scores_one_converged_fit() {
        let score = identifiable_factor_log_evidence(4.0, 1.5, 8).unwrap();
        let expected = -4.0 * (0.5_f64).ln() - 0.75;
        assert!((score - expected).abs() < f64::EPSILON.sqrt());
    }

    #[test]
    fn identifiable_factor_evidence_rejects_unbounded_zero_residual() {
        let error = identifiable_factor_log_evidence(0.0, 1.0, 8).unwrap_err();
        assert!(error.contains("positive"));
    }

    #[test]
    fn ridge_reml_weight_matches_one_direction_stationary_solution() {
        // For one eigendirection the stationarity equation has the closed form
        // λ̂ = γ(γA-m)/(nm-γA).  Here γ=2, A=10, m=8, n=5,
        // hence λ̂=1.2.  The response multiplicity cancels analytically.
        let selected = ridge_reml_select_weight(&[2.0], &[8.0], 10.0, 5, 3).unwrap();
        match selected {
            RidgeRemlWeight::Interior { lambda, score } => {
                // The tolerance is the SEARCH'S OWN LOCATION CONTRACT, not a
                // number that happened to hold. `maximize_value_ordered`
                // certifies a stationary point's location to the requested
                // resolution in `rho = log(lambda/gamma_max)` and returns an
                // evaluated SAMPLE from that bracket — not the bracket's
                // midpoint, and not a polished root — so the reachable accuracy
                // is `|lambda - lambda_hat| <= lambda_hat * expm1(sqrt(eps))`,
                // here 1.79e-8.
                //
                // The `1.0e-9` this used to ask for is 18x inside that, and was
                // never a property of the code: measured, this search returns
                // `lambda = 1.2000000050078` from a stationary bracket 1.13e-8
                // wide, an offset of 4.17e-9 in rho against a requested
                // 1.49e-8. It is not an enclosure-quality question either —
                // `gam_math::score_opt`'s
                // `the_located_optimum_is_enclosure_independent_and_accurate_to_the_contract`
                // runs this exact profile under both the natural and the centred
                // interval extension and gets the same abscissa to the last bit.
                // A caller needing more has to polish the root against the
                // analytic stationarity condition, which this one does not.
                let tolerance = 1.2 * f64::EPSILON.sqrt().exp_m1();
                assert!(
                    (lambda - 1.2).abs() <= tolerance,
                    "lambda={lambda}, off by {} against the search's location contract \
                     {tolerance:e}",
                    (lambda - 1.2).abs()
                );
                assert!(score.is_finite());
            }
            RidgeRemlWeight::FullShrinkage { .. } => {
                panic!("the planted signal has an interior REML optimum")
            }
        }
    }

    #[test]
    fn ridge_reml_weight_recovers_exact_full_shrinkage_boundary() {
        let selected = ridge_reml_select_weight(&[2.0], &[1.0], 10.0, 5, 2).unwrap();
        assert!(matches!(selected, RidgeRemlWeight::FullShrinkage { .. }));
    }

    #[test]
    fn partial_supervision_procrustes_recovers_rotation_and_orthogonalizes_free() {
        // Construct a known orthogonal rotation Q, supervised slice = aux @ Qᵀ.
        let aux = array![
            [1.0_f64, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 2.0],
        ];
        // 90° rotation in the (0,1) plane.
        let q = array![[0.0_f64, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let t_sup = aux.dot(&q.t());
        let t_free = array![
            [1.5_f64, 0.0],
            [0.0, 1.0],
            [-1.0, 2.0],
            [0.3, -0.7],
            [2.0, 1.0],
        ];
        let result = partial_supervision_solve(
            t_sup.view(),
            aux.view(),
            t_free.view(),
            PartialSupervisionSupMethod::Procrustes,
            &[],
            PartialSupervisionFreeConstraint::OrthogonalToSup,
        )
        .expect("procrustes solve should succeed");
        // Aligned supervised block should equal aux exactly (noise-free).
        for r in 0..aux.nrows() {
            for c in 0..aux.ncols() {
                assert!(
                    (result.t_supervised[[r, c]] - aux[[r, c]]).abs() < 1.0e-10,
                    "sup[{r},{c}] = {} vs aux {}",
                    result.t_supervised[[r, c]],
                    aux[[r, c]]
                );
            }
        }
        // Cross-Gram T_freeᵀ T_sup should be near zero after orthogonalization.
        let cross = result.t_free.t().dot(&result.t_supervised);
        let frob: f64 = cross.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(frob < 1.0e-8, "cross frobenius = {frob}");
        assert!(result.alignment_score > 1.0 - 1.0e-10);
        assert!(result.map_r.is_some());
    }

    #[test]
    fn partial_supervision_anchor_pins_exact_anchors_when_full_rank() {
        let aux = array![[1.0_f64, 2.0], [-1.0, 0.5], [3.0, -2.0], [0.7, 1.2],];
        let t_sup = array![[0.5_f64, 1.0], [-0.5, 0.25], [1.5, -1.0], [0.35, 0.6],];
        let t_free = Array2::<f64>::zeros((4, 1));
        let result = partial_supervision_solve(
            t_sup.view(),
            aux.view(),
            t_free.view(),
            PartialSupervisionSupMethod::Anchor,
            &[0, 1, 2],
            PartialSupervisionFreeConstraint::None,
        )
        .expect("anchor solve should succeed");
        for &row in &[0, 1, 2] {
            for c in 0..2 {
                assert!(
                    (result.t_supervised[[row, c]] - aux[[row, c]]).abs() < 1.0e-9,
                    "anchor row {row} col {c} not pinned: {} vs {}",
                    result.t_supervised[[row, c]],
                    aux[[row, c]]
                );
            }
        }
        assert!(result.map_a.is_some() && result.map_b.is_some());
    }

    #[test]
    fn partial_supervision_softl2_selects_a_finite_weight() {
        let aux = array![
            [1.0_f64, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [0.5, -0.5],
        ];
        let t_sup = array![
            [1.0_f64, 0.1],
            [0.1, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [0.5, -0.5],
        ];
        let t_free = array![[0.5_f64], [0.5], [0.5], [0.5], [0.5]];
        let result = partial_supervision_solve(
            t_sup.view(),
            aux.view(),
            t_free.view(),
            PartialSupervisionSupMethod::SoftL2,
            &[],
            PartialSupervisionFreeConstraint::OrthogonalToSup,
        )
        .expect("soft_l2 solve should succeed");
        let lam = result.selected_weight.unwrap();
        assert!(lam.is_finite() && lam > 0.0, "lam={lam}");
        assert!(result.map_a.is_some());
    }

    #[test]
    fn partial_supervision_softl2_returns_exact_null_map_without_signal() {
        let t_sup = array![[-1.0_f64], [0.0], [1.0]];
        let aux = array![[1.0_f64], [1.0], [1.0]];
        let t_free = Array2::<f64>::zeros((3, 0));
        let result = partial_supervision_solve(
            t_sup.view(),
            aux.view(),
            t_free.view(),
            PartialSupervisionSupMethod::SoftL2,
            &[],
            PartialSupervisionFreeConstraint::None,
        )
        .expect("zero-signal soft-L2 solve should select the null boundary");
        assert_eq!(result.selected_weight, Some(f64::INFINITY));
        assert!(result.map_a.unwrap().iter().all(|&value| value == 0.0));
        assert!(result.t_supervised.iter().all(|&value| value == 0.0));
    }

    /// FINDING B (#2022 review): the isometry-orbit stiffness must be σ_max(G_e),
    /// not ‖G_e‖_F. A d=2 fixture where the two differ pins the fix; d=1 (where
    /// they coincide) must be unchanged.
    #[test]
    fn symmetric_spectral_norm_sq_uses_sigma_max_not_frobenius() {
        use ndarray::Array2;
        // Off-diagonal G = [[0,1],[1,0]]: eigenvalues ±1 ⇒ σ_max² = 1, while
        // ‖G‖_F² = 2. The Frobenius bug returns 2, doubling the stiffness →
        // halving the pinned-energy fraction → risks a false "unpinned" verdict.
        let g = Array2::<f64>::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let sigma_sq = super::symmetric_spectral_norm_sq(g.view());
        let frob_sq: f64 = g.iter().map(|&v| v * v).sum();
        assert!(
            (sigma_sq - 1.0).abs() < 1e-9,
            "σ_max² must be 1, got {sigma_sq}"
        );
        assert!(
            (frob_sq - 2.0).abs() < 1e-9,
            "fixture ‖G‖_F² is 2 (the flip)"
        );
        assert!(
            sigma_sq < frob_sq,
            "σ_max² must be strictly below ‖G‖_F² here"
        );

        // Balanced rank-2 G = [[1,0.1],[0.1,1]] ⇒ eigenvalues 1.1, 0.9 ⇒
        // σ_max² = 1.21 vs ‖G‖_F² = 2.02.
        let g2 = Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 0.1, 0.1, 1.0]).unwrap();
        let s2 = super::symmetric_spectral_norm_sq(g2.view());
        assert!((s2 - 1.21).abs() < 1e-9, "σ_max² must be 1.21, got {s2}");

        // d = 1: σ_max² == Frobenius² == g² — no regression on the scalar path.
        let g1 = Array2::<f64>::from_shape_vec((1, 1), vec![-1.3]).unwrap();
        let s1 = super::symmetric_spectral_norm_sq(g1.view());
        assert!((s1 - 1.69).abs() < 1e-9, "d=1 must be g² = 1.69, got {s1}");
    }
}
