//! Canonical Laplace evidence, IFT cascade, and topology selection.
//!
//! This module is the single canonical entry point for:
//!
//!   1. Laplace evidence `V(ρ, T) = F + (1/2) log|H| - (1/2) log|S(ρ)|+
//!      - ((dim(H)-rank(S))/2) log(2π)`
//!      evaluated at the arrow-Schur inner-loop fixed point.
//!   2. The full IFT cascade `∂u*/∂β → ∂β*/∂ρ → ∂u*/∂ρ` through the three
//!      continuous tiers `(u, β, ρ)`, per §2.2 / §2.4 / §2.6.
//!   3. The per-`ρ` evidence gradient `∂V/∂ρ` via the arrow trace formula,
//!      per §3.5 / §3.7 / §3.8.
//!   4. Discrete topology selection across `{periodic, flat, sphere, torus}`,
//!      per §4 (4.1 / 4.5 / 4.6).
//!
//! ## Crucial numerical invariants (proposal §1.7, §6.4, §6.5)
//!
//!   * Evidence log-determinants use **undamped** factors. The cached
//!     `ArrowFactorCache::htt_factors_undamped` Cholesky factors of
//!     `H_uu_i` (no `ridge_u`) are the ones that must enter
//!     `Σ_i log|H_uu_i|`. Likewise a factored Schur log-det must be of
//!     `A(0, 0) = H_ββ - Σ_i H_uβ_iᵀ H_uu_i⁻¹ H_uβ_i`, not the LM-damped
//!     surrogate. Matrix-free evidence callers must provide the matching
//!     undamped HVP so the same log-det is estimated by SLQ.
//!   * IFT solves invert `H_uu`, not `H_uu + ridge_u I` (proposal §1.7,
//!     §6.6). The evidence-side IFT predictor loop here uses the undamped
//!     `htt_factors_undamped` factors for exactly this reason.
//!   * Penalty pseudo-logdet `log|S(ρ)|+` is the prior penalty, distinct
//!     from the arrow Schur complement (proposal §3.1, §3.6). The variable
//!     names below preserve that distinction:
//!       `arrow_schur_log_det`   = `log|A|` where `A` is the arrow Schur.
//!       `penalty_log_det`       = `log|S_pen(ρ)|+` where `S_pen` is the
//!                                 prior penalty matrix pseudo-logdet.
//!
//! ## Sign discipline (proposal §3.1, §4.3)
//!
//! `V` as written is the *negative log evidence* when `F` is the
//! penalized negative log posterior. The maximizer of evidence is the
//! minimizer of `V`. For the public API we expose **negative log
//! evidence** under `laplace_evidence` and rank topologies by the
//! **minimum** of the configured per-row or per-effective-dimension
//! normalization (see `select_topology` below); equivalently the caller can
//! negate and `argmax`.

use faer::Side;
use gam_runtime::warm_start::{Fingerprint, Fingerprinter};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::arrow_schur::{ArrowFactorCache, ArrowSchurSystem};
use crate::priority_selection::{PriorityCandidate, rank_priority_candidates};
use gam_linalg::faer_ndarray::FaerEigh;
use gam_linalg::lanczos::{
    SymmetricLanczosOptions, symmetric_lanczos_eigenpairs, symmetric_lanczos_log_quadrature,
};
use gam_linalg::pairwise_reduce::{BASE_CHUNK, pairwise_sum};
use gam_linalg::triangular::cholesky_solve_vector;
use gam_math::special::bessel_i0_log_minus_abs_and_ratio;

pub const ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD: usize = 1024;
const EVIDENCE_LOGDET_SLQ_PROBES: usize = 16;
const EVIDENCE_LOGDET_LANCZOS_STEPS: usize = 32;
const EVIDENCE_HVP_SYMMETRY_REL_TOL: f64 = 1e-8;
const EVIDENCE_HVP_SYMMETRY_PROBES: usize = 4;

/// Matrix-free SPD Hessian logdet source used when the arrow Schur factor is
/// not materialized. The callback must apply the same undamped Hessian whose
/// determinant enters the Laplace evidence.
#[derive(Clone, Copy)]
pub struct EvidenceHvpLogDet<'a> {
    pub dim: usize,
    pub apply: &'a dyn Fn(&[f64]) -> Vec<f64>,
}

/// Source for the Hessian log determinant in [`laplace_evidence`].
#[derive(Clone, Copy)]
pub enum EvidenceLogDetSource<'a> {
    /// Use the exact arrow Cholesky factors, falling back to `fallback_hvp`
    /// when the Schur factor is absent on a matrix-free solve.
    FactoredArrow {
        cache: &'a ArrowFactorCache,
        fallback_hvp: Option<EvidenceHvpLogDet<'a>>,
    },
    /// Use an HVP callback directly. Dimensions at or below
    /// [`ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD`] are materialized exactly;
    /// larger operators use the same Rademacher-Lanczos SLQ constants as
    /// `FrozenAnalyticPenaltyOp`.
    Hvp(EvidenceHvpLogDet<'a>),
}

// ---------------------------------------------------------------------------
// Topology candidate enum and selection result
// ---------------------------------------------------------------------------

/// Discrete topology choice for the latent coordinate domain.
///
/// Maps directly to the set `{periodic, flat, sphere, torus}`. No additional
/// variants — unused candidate variants are deliberately not carried
/// alongside the four-way selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TopologyKind {
    /// `S¹` or periodic interval (cyclic B-spline / periodic Duchon).
    Periodic,
    /// `Rᵈ` Euclidean Duchon / Matérn / thin-plate patch.
    Flat,
    /// `S²` embedded in `R³`, spherical Wahba/Sobolev basis.
    Sphere,
    /// `S¹ × S¹` mixed-periodicity Duchon.
    Torus,
}

impl TopologyKind {
    /// Tie-break priority — smaller wins. Per §4.6: `flat < periodic <
    /// sphere < torus`.
    pub fn complexity_rank(self) -> u8 {
        match self {
            TopologyKind::Flat => 0,
            TopologyKind::Periodic => 1,
            TopologyKind::Sphere => 2,
            TopologyKind::Torus => 3,
        }
    }
}

/// One topology candidate together with the evidence ingredients it
/// produced at its own fitted optimum.
#[derive(Debug, Clone)]
pub struct TopologyCandidate {
    pub kind: TopologyKind,
    /// Negative-log-evidence `V(ρ_T*, T)` evaluated at the candidate's own
    /// fitted `(ρ_T*, β_T*, u_T*)`.
    pub negative_log_evidence: f64,
    /// Effective integrated dimension after rank/nullspace accounting. This
    /// is the dimension used for per-complexity topology normalization.
    pub effective_dim: f64,
    /// Number of response rows used to fit this topology candidate. This is
    /// the dimension used for per-observation topology normalization.
    pub n_obs: usize,
    /// `True` iff the candidate's continuous inner+outer fit converged
    /// cleanly. Failed candidates are excluded from ranking (proposal
    /// §4.4 item 7 and §6.11).
    pub converged: bool,
    /// Optional rationale string for excluded candidates (proposal
    /// §6.11): `"sphere input not on S²"`, `"torus periods missing"`, etc.
    pub exclusion_reason: Option<String>,
}

/// Outcome of [`select_topology`].
#[derive(Debug, Clone)]
pub struct SelectedTopology {
    pub winner: TopologyKind,
    /// All candidates sorted from best (lowest negative log evidence)
    /// to worst, with excluded candidates appended last.
    pub ranking: Vec<TopologyCandidate>,
    /// `True` iff the top two finite scores fall within `tie_tolerance`.
    /// Per §4.6 we still pick one — the simpler topology — but expose
    /// the tie so callers can warn.
    pub tie: bool,
}

/// Tolerance options for the topology comparator.
#[derive(Debug, Clone, Copy)]
pub struct TopologySelectOptions {
    /// Maximum `|V_a - V_b|` for which two candidates are treated as
    /// numerically tied after [`TopologyScoreScale`] normalization. Default
    /// `1e-3` per proposal §4.6 examples.
    pub tie_tolerance: f64,
    /// Score scale used for discrete topology comparison. Raw evidence is
    /// intentionally not a selector because candidates may have different
    /// row counts and basis/nullspace dimensions.
    pub score_scale: TopologyScoreScale,
}

/// Normalization applied before ranking topology candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyScoreScale {
    /// Compare negative log evidence per observation row.
    PerObservation,
    /// Compare negative log evidence per effective integrated dimension.
    PerEffectiveDim,
}

/// Convergence controls for stacking retained topology predictive densities.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StackingConfig {
    /// Exhaustion-escalation bound on solver iterations. It never selects the
    /// weights: exhausting it without the KKT certificate is an error, so an
    /// uncertified iterate can never feed model selection.
    pub max_iter: usize,
    /// Simplex-KKT residual the solution must certify before it is returned.
    /// The residual is scale-free: the KKT multiplier of the stacking problem
    /// is exactly 1, so `g_k − 1` is already a relative stationarity measure.
    pub kkt_tol: f64,
}

impl Default for StackingConfig {
    fn default() -> Self {
        Self {
            max_iter: 256,
            kkt_tol: f64::EPSILON.sqrt(),
        }
    }
}

/// Auditable global-optimality certificate for a stacking solution.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StackingCertificate {
    /// Achieved held-out mean log predictive density.
    pub mean_log_score: f64,
    /// Frank-Wolfe/KKT duality gap `max_k g_k - w·g`. Concavity makes this
    /// an upper bound on objective suboptimality.
    pub duality_gap: f64,
    /// Absolute simplex mass residual `|Σw - 1|`.
    pub simplex_residual: f64,
    /// Error in the analytic multiplier identity `w·g = 1`.
    pub multiplier_residual: f64,
    /// Largest complementary-slackness residual `w_k |g_k - w·g|`.
    pub complementarity_residual: f64,
}

impl StackingCertificate {
    pub fn residual(&self) -> f64 {
        self.duality_gap
            .max(self.simplex_residual)
            .max(self.multiplier_residual)
            .max(self.complementarity_residual)
    }
}

/// Serializable work state carried by a stacking exhaustion error and accepted
/// by [`resume_stacking_weights`]. Weights stay aligned to the original input
/// columns; no candidate is silently dropped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackingCheckpoint {
    pub weights: Array1<f64>,
    pub completed_iterations: usize,
    density_fingerprint: Fingerprint,
}

/// Typed stacking failure. In particular, exhaustion carries both its
/// certificate evidence and an exact checkpoint rather than returning weights.
#[derive(Debug, Clone)]
pub enum StackingError {
    InvalidInput {
        message: String,
    },
    NumericalFailure {
        message: String,
        certificate: Option<StackingCertificate>,
        checkpoint: Option<StackingCheckpoint>,
    },
    DidNotConverge {
        max_iterations: usize,
        tolerance: f64,
        certificate: StackingCertificate,
        checkpoint: StackingCheckpoint,
    },
}

impl std::fmt::Display for StackingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput { message } => write!(f, "invalid stacking problem: {message}"),
            Self::NumericalFailure {
                message,
                certificate,
                checkpoint,
            } => write!(
                f,
                "stacking numerical failure: {message} (certificate residual {}, checkpoint iterations {})",
                certificate.map_or(f64::NAN, |value| value.residual()),
                checkpoint
                    .as_ref()
                    .map_or(0, |value| value.completed_iterations)
            ),
            Self::DidNotConverge {
                max_iterations,
                tolerance,
                certificate,
                checkpoint,
            } => write!(
                f,
                "stacking did not certify after {max_iterations} additional iterations (total {}): KKT residual {:.6e} exceeds tolerance {:.3e}; resume from the carried weights checkpoint",
                checkpoint.completed_iterations,
                certificate.residual(),
                tolerance
            ),
        }
    }
}

impl std::error::Error for StackingError {}

/// Simplex weights for retained topology candidates plus their verified global
/// optimum certificate.
#[derive(Debug, Clone)]
pub struct StackingWeights {
    pub weights: Array1<f64>,
    pub iterations: usize,
    pub certificate: StackingCertificate,
}

impl StackingWeights {
    pub fn mean_log_score(&self) -> f64 {
        self.certificate.mean_log_score
    }
}

struct StackingProblem {
    scaled_density: Array2<f64>,
    row_log_scale: Array1<f64>,
}

impl StackingProblem {
    fn from_log_density(log_density: ArrayView2<'_, f64>) -> Result<Self, StackingError> {
        let n_obs = log_density.nrows();
        let n_cand = log_density.ncols();
        if n_cand == 0 || n_obs == 0 {
            return Err(StackingError::InvalidInput {
                message: "at least one candidate and one held-out row are required".to_string(),
            });
        }
        if let Some(((row, col), value)) = log_density
            .indexed_iter()
            .find(|(_, value)| value.is_nan() || **value == f64::INFINITY)
        {
            return Err(StackingError::InvalidInput {
                message: format!(
                    "log density at row {row}, candidate {col} is {value}; NaN and +infinity are not predictive densities"
                ),
            });
        }
        let mut scaled_density = Array2::<f64>::zeros((n_obs, n_cand));
        let mut row_log_scale = Array1::<f64>::zeros(n_obs);
        for row in 0..n_obs {
            let row_max = (0..n_cand)
                .map(|col| log_density[[row, col]])
                .fold(f64::NEG_INFINITY, f64::max);
            if !row_max.is_finite() {
                return Err(StackingError::InvalidInput {
                    message: format!(
                        "held-out row {row} has zero density under every candidate; deleting it would change the stacking target"
                    ),
                });
            }
            row_log_scale[row] = row_max;
            for col in 0..n_cand {
                let value = log_density[[row, col]];
                if value.is_finite() {
                    scaled_density[[row, col]] = (value - row_max).exp();
                }
            }
        }
        Ok(Self {
            scaled_density,
            row_log_scale,
        })
    }

    fn evaluate(
        &self,
        weights: ArrayView1<'_, f64>,
    ) -> Result<(Array1<f64>, StackingCertificate, f64), String> {
        let n = self.scaled_density.nrows();
        let k = self.scaled_density.ncols();
        let mass = weights.sum();
        if weights.len() != k
            || weights
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || !(mass.is_finite() && mass > 0.0)
        {
            return Err(
                "checkpoint weights are not a finite nonnegative simplex vector".to_string(),
            );
        }
        let mut gradient = Array1::<f64>::zeros(k);
        let mut centered_objective = 0.0_f64;
        let mut mean_log_score = 0.0_f64;
        for row in 0..n {
            let mut mixture = 0.0_f64;
            for col in 0..k {
                mixture += weights[col] * self.scaled_density[[row, col]];
            }
            if !(mixture.is_finite() && mixture > 0.0) {
                return Err(format!(
                    "candidate mixture lost held-out row {row} (scaled density {mixture})"
                ));
            }
            let log_mixture = mixture.ln();
            centered_objective += log_mixture / n as f64;
            let log_score = self.row_log_scale[row] + log_mixture;
            let count = (row + 1) as f64;
            mean_log_score = mean_log_score * ((count - 1.0) / count) + log_score / count;
            for col in 0..k {
                gradient[col] += self.scaled_density[[row, col]] / mixture / n as f64;
            }
        }
        if !centered_objective.is_finite()
            || !mean_log_score.is_finite()
            || gradient.iter().any(|value| !value.is_finite())
        {
            return Err("objective or analytic gradient became non-finite".to_string());
        }
        let multiplier = weights.dot(&gradient);
        let max_gradient = gradient.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let certificate = StackingCertificate {
            mean_log_score,
            duality_gap: (max_gradient - multiplier).max(0.0),
            simplex_residual: (mass - 1.0).abs(),
            multiplier_residual: (multiplier - 1.0).abs(),
            complementarity_residual: weights
                .iter()
                .zip(gradient.iter())
                .map(|(&weight, &gain)| weight * (gain - multiplier).abs())
                .fold(0.0_f64, f64::max),
        };
        Ok((gradient, certificate, centered_objective))
    }

    fn centered_objective(&self, weights: ArrayView1<'_, f64>) -> Option<f64> {
        let n = self.scaled_density.nrows();
        let mut objective = 0.0_f64;
        for row in 0..n {
            let mixture = self.scaled_density.row(row).dot(&weights);
            if !(mixture.is_finite() && mixture > 0.0) {
                return None;
            }
            objective += mixture.ln() / n as f64;
        }
        objective.is_finite().then_some(objective)
    }
}

/// Solve the stacking-of-predictive-distributions weight problem from a
/// per-observation held-out log-density table `log_density[i, k] = log p_k(y_i)`.
///
/// This belongs on the evidence surface rather than in a separate solver: it is
/// the topology/evidence consumer that replaces winner-take-all only when the
/// caller has retained candidate fits and per-point held-out densities.
///
/// ## Optimality certificate
///
/// The objective `f(w) = mean_i log Σ_k w_k p_ik` is concave on the simplex,
/// so first-order KKT conditions are necessary AND sufficient for the global
/// optimum. With `g_k = ∂f/∂w_k = mean_i p_ik / mix_i`, the simplex multiplier
/// is exactly `Σ_k w_k g_k = 1`, so the KKT system is `g_k ≤ 1` for every
/// candidate with `w_k · (1 − g_k) = 0` (complementary slackness). Iterates
/// use an analytic reduced-space Newton step on the current simplex face; an
/// exact concave line solve toward the most violated vertex activates a
/// candidate or globalizes a singular Newton system. The solve returns only
/// after the Frank-Wolfe duality gap and all primal/KKT residuals are verified
/// below `config.kkt_tol`. Exhaustion carries the full certificate and a
/// resumable weights checkpoint; uncertified weights never reach selection.
pub fn solve_stacking_weights(
    log_density: ArrayView2<'_, f64>,
    config: StackingConfig,
) -> Result<StackingWeights, StackingError> {
    solve_stacking_weights_impl(log_density, config, None)
}

/// Resume a previously exhausted stacking solve without redoing its accepted
/// Newton/active-set work.
pub fn resume_stacking_weights(
    log_density: ArrayView2<'_, f64>,
    config: StackingConfig,
    checkpoint: &StackingCheckpoint,
) -> Result<StackingWeights, StackingError> {
    solve_stacking_weights_impl(log_density, config, Some(checkpoint))
}

fn solve_stacking_weights_impl(
    log_density: ArrayView2<'_, f64>,
    config: StackingConfig,
    checkpoint: Option<&StackingCheckpoint>,
) -> Result<StackingWeights, StackingError> {
    if config.max_iter == 0 {
        return Err(StackingError::InvalidInput {
            message: "max_iter must be positive".to_string(),
        });
    }
    let numerical_floor = f64::EPSILON.sqrt();
    if !config.kkt_tol.is_finite() || config.kkt_tol < numerical_floor {
        return Err(StackingError::InvalidInput {
            message: format!(
                "kkt_tol must be finite and at least the floating-point resolution floor {numerical_floor:.3e}"
            ),
        });
    }
    let density_fingerprint = evidence_matrix_fingerprint("stacking-log-density-v1", log_density);
    let problem = StackingProblem::from_log_density(log_density)?;
    let k = problem.scaled_density.ncols();
    let (mut weights, completed_before) = if let Some(checkpoint) = checkpoint {
        if checkpoint.density_fingerprint != density_fingerprint {
            return Err(StackingError::InvalidInput {
                message: "checkpoint belongs to a different held-out density table".to_string(),
            });
        }
        if checkpoint.weights.len() != k {
            return Err(StackingError::InvalidInput {
                message: format!(
                    "checkpoint has {} weights but the density table has {k} candidates",
                    checkpoint.weights.len()
                ),
            });
        }
        let mut weights = checkpoint.weights.clone();
        let mass = weights.sum();
        if weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || !mass.is_finite()
            || (mass - 1.0).abs() > config.kkt_tol
        {
            return Err(StackingError::InvalidInput {
                message: "checkpoint weights must be a finite nonnegative simplex vector"
                    .to_string(),
            });
        }
        weights.mapv_inplace(|value| value / mass);
        (weights, checkpoint.completed_iterations)
    } else {
        (Array1::<f64>::from_elem(k, 1.0 / k as f64), 0)
    };

    for additional_iterations in 0..=config.max_iter {
        let completed_iterations = completed_before + additional_iterations;
        let checkpoint = StackingCheckpoint {
            weights: weights.clone(),
            completed_iterations,
            density_fingerprint,
        };
        let (gradient, certificate, objective) =
            problem.evaluate(weights.view()).map_err(|message| {
                StackingError::NumericalFailure {
                    message,
                    certificate: None,
                    checkpoint: Some(checkpoint.clone()),
                }
            })?;
        if certificate.residual() <= config.kkt_tol {
            return Ok(StackingWeights {
                weights,
                iterations: completed_iterations,
                certificate,
            });
        }
        if additional_iterations == config.max_iter {
            return Err(StackingError::DidNotConverge {
                max_iterations: config.max_iter,
                tolerance: config.kkt_tol,
                certificate,
                checkpoint,
            });
        }

        let max_gradient_col = gradient
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .expect("stacking has at least one candidate");
        let candidate = stacking_newton_step(&problem, weights.view(), gradient.view(), objective)
            .or_else(|| {
                stacking_vertex_step(&problem, weights.view(), max_gradient_col, objective)
            })
            .ok_or_else(|| StackingError::NumericalFailure {
                message: "positive KKT gap remained but neither the analytic Newton direction nor the exact vertex line solve produced a representable ascent step".to_string(),
                certificate: Some(certificate),
                checkpoint: Some(checkpoint),
            })?;
        weights = candidate;
    }
    Err(StackingError::NumericalFailure {
        message: format!(
            "stacking solver exhausted its inclusive iteration budget ({}) without producing a \
             terminal verdict",
            config.max_iter
        ),
        certificate: None,
        checkpoint: None,
    })
}

fn stacking_newton_step(
    problem: &StackingProblem,
    weights: ArrayView1<'_, f64>,
    gradient: ArrayView1<'_, f64>,
    objective: f64,
) -> Option<Array1<f64>> {
    let active: Vec<usize> = weights
        .iter()
        .enumerate()
        .filter_map(|(index, &weight)| (weight > 0.0).then_some(index))
        .collect();
    if active.len() < 2 {
        return None;
    }
    let reference_position = active
        .iter()
        .enumerate()
        .max_by(|left, right| weights[*left.1].total_cmp(&weights[*right.1]))
        .map(|(position, _)| position)?;
    let reference = active[reference_position];
    let free: Vec<usize> = active
        .iter()
        .copied()
        .filter(|&index| index != reference)
        .collect();
    let dimension = free.len();
    let n = problem.scaled_density.nrows();
    let mut information = Array2::<f64>::zeros((dimension, dimension));
    for row in 0..n {
        let mixture = problem.scaled_density.row(row).dot(&weights);
        if !(mixture.is_finite() && mixture > 0.0) {
            return None;
        }
        let reference_density = problem.scaled_density[[row, reference]];
        let contrasts: Vec<f64> = free
            .iter()
            .map(|&col| (problem.scaled_density[[row, col]] - reference_density) / mixture)
            .collect();
        for left in 0..dimension {
            for right in 0..=left {
                information[[left, right]] += contrasts[left] * contrasts[right] / n as f64;
                information[[right, left]] = information[[left, right]];
            }
        }
    }
    let reduced_gradient =
        Array1::from_iter(free.iter().map(|&col| gradient[col] - gradient[reference]));
    let (eigenvalues, eigenvectors) = information.eigh(Side::Lower).ok()?;
    let spectral_scale = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    if !(spectral_scale.is_finite() && spectral_scale > 0.0) {
        return None;
    }
    let rank_tolerance = f64::EPSILON * (dimension as f64) * spectral_scale.max(f64::MIN_POSITIVE);
    let projected = eigenvectors.t().dot(&reduced_gradient);
    let mut spectral_step = Array1::<f64>::zeros(dimension);
    for index in 0..dimension {
        if eigenvalues[index] > rank_tolerance {
            spectral_step[index] = projected[index] / eigenvalues[index];
        }
    }
    let reduced_step = eigenvectors.dot(&spectral_step);
    let ascent = reduced_gradient.dot(&reduced_step);
    if !(ascent.is_finite() && ascent > 0.0) {
        return None;
    }
    let mut direction = Array1::<f64>::zeros(weights.len());
    for (position, &col) in free.iter().enumerate() {
        direction[col] = reduced_step[position];
    }
    direction[reference] = -reduced_step.sum();
    let mut step = 1.0_f64;
    let mut boundary = None;
    for col in 0..weights.len() {
        if direction[col] < 0.0 {
            let candidate = -weights[col] / direction[col];
            if candidate < step {
                step = candidate;
                boundary = Some(col);
            }
        }
    }
    loop {
        let mut candidate = &weights + &(direction.mapv(|value| step * value));
        if let Some(col) = boundary {
            if step == -weights[col] / direction[col] {
                candidate[col] = 0.0;
            }
        }
        for value in candidate.iter_mut() {
            if *value < 0.0 && *value >= -f64::EPSILON {
                *value = 0.0;
            }
        }
        let mass = candidate.sum();
        if mass.is_finite() && mass > 0.0 {
            candidate.mapv_inplace(|value| value / mass);
            if problem
                .centered_objective(candidate.view())
                .is_some_and(|value| value > objective)
            {
                return Some(candidate);
            }
        }
        let next_step = 0.5 * step;
        if next_step == step || next_step == 0.0 {
            return None;
        }
        step = next_step;
        boundary = None;
    }
}

fn stacking_vertex_step(
    problem: &StackingProblem,
    weights: ArrayView1<'_, f64>,
    vertex: usize,
    objective: f64,
) -> Option<Array1<f64>> {
    let derivative = |step: f64| -> f64 {
        let mut value = 0.0_f64;
        let n = problem.scaled_density.nrows();
        for row in 0..n {
            let current = problem.scaled_density.row(row).dot(&weights);
            let target = problem.scaled_density[[row, vertex]];
            let mixture = (1.0 - step) * current + step * target;
            if mixture <= 0.0 {
                return f64::NEG_INFINITY;
            }
            value += (target - current) / mixture / n as f64;
        }
        value
    };
    if derivative(0.0) <= 0.0 {
        return None;
    }
    let mut step = if derivative(1.0) >= 0.0 {
        1.0
    } else {
        let mut lower = 0.0_f64;
        let mut upper = 1.0_f64;
        while upper - lower > f64::EPSILON.sqrt() {
            let middle = 0.5 * (lower + upper);
            if derivative(middle) > 0.0 {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        0.5 * (lower + upper)
    };
    loop {
        let mut candidate = weights.mapv(|weight| (1.0 - step) * weight);
        candidate[vertex] += step;
        if problem
            .centered_objective(candidate.view())
            .is_some_and(|value| value > objective)
        {
            return Some(candidate);
        }
        let next_step = 0.5 * step;
        if next_step == step || next_step == 0.0 {
            return None;
        }
        step = next_step;
    }
}

/// Combine retained candidate response-scale means with stacking weights.
pub fn stacked_predictive_mean(
    weights: &Array1<f64>,
    candidate_means: &[Array1<f64>],
) -> Result<Array1<f64>, String> {
    if candidate_means.len() != weights.len() {
        return Err(format!(
            "stacked_predictive_mean: {} weights but {} candidate mean vectors",
            weights.len(),
            candidate_means.len()
        ));
    }
    let Some(first) = candidate_means.first() else {
        return Err("stacked_predictive_mean requires at least one candidate".to_string());
    };
    let n_rows = first.len();
    if candidate_means.iter().any(|means| means.len() != n_rows) {
        return Err(
            "stacked_predictive_mean: candidate mean vectors disagree on row count".to_string(),
        );
    }
    let mut out = Array1::<f64>::zeros(n_rows);
    for (weight, means) in weights.iter().zip(candidate_means) {
        if *weight != 0.0 {
            out.scaled_add(*weight, means);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Discrete mixture rung (Object 3a / WP-C)
// ---------------------------------------------------------------------------
//
// A `k`-component full-covariance Gaussian mixture fitted by deterministic
// k-means++-style seeding (reusing `terms::basis` farthest-point k-means) plus
// EM to a tolerance. It is priced by its free-parameter count with the
// invariant BIC approximation to negative log evidence,
//
//     BIC/2 = -loglik + (P/2) log(n).
//
// BIC is intentional here. An outer product of per-observation scores is not
// an observed Hessian and need not be full-rank even when the likelihood has
// curvature (an exactly centered Gaussian mean is the simplest counterexample).
// Moreover, the covariance-floor constraint can put a component on a boundary,
// where an interior SPD Laplace expansion is mathematically invalid. Without a
// declared parameter prior and its Jacobian, a raw Hessian determinant would
// also change under reparameterization. The smooth parametric shape candidates
// use this same BIC-form score, so every shape-race corroborating score now has
// one finite, parameterization-invariant meaning.

/// Convergence + ladder controls for the discrete-mixture rung. All fields are
/// fixed (no clock randomness, no env): deterministic seeding makes the fitted
/// mixture a pure function of the data and `k`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GaussianMixtureConfig {
    /// Exhaustion-escalation bound on EM iterations. It never selects the
    /// estimator: exhausting it without the convergence certificate
    /// (monotone ascent + relative objective step below `loglik_tol`) is an
    /// error, so an uncertified mixture never enters evidence comparison.
    pub max_iter: usize,
    /// Relative mean-log-likelihood improvement tolerance for EM stopping.
    pub loglik_tol: f64,
    /// Relative max-norm tolerance for the full EM parameter-map residual.
    pub parameter_tol: f64,
    /// Lower eigenvalue constraint for every component covariance. The M-step
    /// solves this constrained likelihood problem exactly by spectral clipping;
    /// it is not an additive ridge or an unmodelled prior.
    pub covariance_floor: f64,
    /// Maximum iterations for the deterministic k-means seeding pass.
    pub kmeans_max_iter: usize,
}

impl Default for GaussianMixtureConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            loglik_tol: f64::EPSILON.sqrt(),
            parameter_tol: f64::EPSILON.sqrt(),
            covariance_floor: 1e-6,
            kmeans_max_iter: 25,
        }
    }
}

/// Residual evidence proving that the returned mixture is a fixed point of its
/// likelihood EM map rather than merely the iterate present at a work cap.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GaussianMixtureCertificate {
    /// Mean log likelihood at the exact parameter state being certified.
    pub mean_log_likelihood: f64,
    /// Signed gain produced by one further EM map application from that state.
    pub mean_log_likelihood_gain: f64,
    /// Absolute numerical uncertainty on the comparison of the two likelihoods.
    /// This includes both the final likelihood-reduction error and the
    /// scale-derived resolution floor of the composite EM map.
    pub monotonicity_uncertainty: f64,
    pub objective_residual: f64,
    pub objective_tolerance: f64,
    pub parameter_residual: f64,
    pub parameter_tolerance: f64,
}

/// Exact parameter state carried across an EM exhaustion boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianMixtureCheckpoint {
    pub weights: Array1<f64>,
    pub means: Array2<f64>,
    pub covariances: Vec<Array2<f64>>,
    pub mean_log_likelihood: f64,
    pub completed_iterations: usize,
    data_fingerprint: Fingerprint,
    covariance_floor: f64,
}

/// Typed Gaussian-mixture optimization failure. Exhaustion and a broken EM
/// monotonicity invariant both carry the last internally consistent state.
#[derive(Debug, Clone)]
pub enum GaussianMixtureError {
    InvalidInput {
        message: String,
    },
    NumericalFailure {
        message: String,
        checkpoint: Option<GaussianMixtureCheckpoint>,
    },
    MonotonicityViolation {
        previous_mean_log_likelihood: f64,
        next_mean_log_likelihood: f64,
        numerical_uncertainty: f64,
        checkpoint: GaussianMixtureCheckpoint,
    },
    DidNotConverge {
        max_iterations: usize,
        certificate: GaussianMixtureCertificate,
        checkpoint: GaussianMixtureCheckpoint,
    },
}

impl std::fmt::Display for GaussianMixtureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput { message } => write!(f, "invalid Gaussian mixture: {message}"),
            Self::NumericalFailure {
                message,
                checkpoint,
            } => write!(
                f,
                "Gaussian-mixture numerical failure: {message} (checkpoint iterations {})",
                checkpoint
                    .as_ref()
                    .map_or(0, |value| value.completed_iterations)
            ),
            Self::MonotonicityViolation {
                previous_mean_log_likelihood,
                next_mean_log_likelihood,
                numerical_uncertainty,
                checkpoint,
            } => write!(
                f,
                "Gaussian-mixture EM violated monotone ascent at iteration {}: mean log likelihood {previous_mean_log_likelihood:.12e} -> {next_mean_log_likelihood:.12e} (comparison uncertainty {numerical_uncertainty:.3e}); resume from the carried checkpoint only after diagnosing the numerical failure",
                checkpoint.completed_iterations
            ),
            Self::DidNotConverge {
                max_iterations,
                certificate,
                checkpoint,
            } => write!(
                f,
                "Gaussian-mixture EM did not certify after {max_iterations} additional iterations (total {}): signed mean-log-likelihood gain {:.6e} (numerical uncertainty {:.3e}), objective residual {:.6e}/{:.3e}, parameter-map residual {:.6e}/{:.3e}; resume from the carried checkpoint, which is not comparable evidence",
                checkpoint.completed_iterations,
                certificate.mean_log_likelihood_gain,
                certificate.monotonicity_uncertainty,
                certificate.objective_residual,
                certificate.objective_tolerance,
                certificate.parameter_residual,
                certificate.parameter_tolerance
            ),
        }
    }
}

impl std::error::Error for GaussianMixtureError {}

/// A fitted `k`-component full-covariance Gaussian mixture.
#[derive(Debug, Clone)]
pub struct GaussianMixtureFit {
    /// Mixing weights, length `k`, on the simplex.
    weights: Array1<f64>,
    /// Component means, `k × d`.
    means: Array2<f64>,
    /// Component covariances, `k` matrices of shape `d × d` (SPD).
    covariances: Vec<Array2<f64>>,
    /// Number of mixture components.
    k: usize,
    /// Data dimension.
    d: usize,
    /// Number of rows used to fit.
    n_obs: usize,
    /// Maximised total log-likelihood `Σ_i log Σ_j w_j N(y_i; μ_j, Σ_j)`.
    loglik: f64,
    /// EM iterations taken.
    iterations: usize,
    certificate: GaussianMixtureCertificate,
}

impl GaussianMixtureFit {
    pub fn weights(&self) -> ArrayView1<'_, f64> {
        self.weights.view()
    }

    pub fn means(&self) -> ArrayView2<'_, f64> {
        self.means.view()
    }

    pub fn covariances(&self) -> &[Array2<f64>] {
        &self.covariances
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn certificate(&self) -> GaussianMixtureCertificate {
        self.certificate
    }

    /// Free-parameter count `P` of a `k`-component full-covariance mixture in
    /// `d` dimensions: `(k − 1)` mixing weights on the simplex, `k·d` mean
    /// coordinates, and `k · d(d+1)/2` covariance entries. This is the exact
    /// quantity that enters the rank-aware normalizer as `dim(H) − rank(S)`.
    pub fn num_free_parameters(&self) -> usize {
        let cov_per = self.d * (self.d + 1) / 2;
        (self.k - 1) + self.k * self.d + self.k * cov_per
    }

    /// Per-observation log predictive density `log p(y_i)` under the fitted
    /// mixture, length `n`. This is the held-out-density column source for
    /// cross-class stacking when the mixture is evaluated on a held-out fold.
    pub fn per_point_log_density(&self, data: ArrayView2<'_, f64>) -> Result<Array1<f64>, String> {
        if data.ncols() != self.d {
            return Err(format!(
                "mixture log-density expects {} columns, got {}",
                self.d,
                data.ncols()
            ));
        }
        let n = data.nrows();
        let mut comp = Vec::with_capacity(self.k);
        for j in 0..self.k {
            comp.push(GaussianComponentEval::factor(
                self.means.row(j),
                &self.covariances[j],
            )?);
        }
        let mut out = Array1::<f64>::zeros(n);
        let log_w: Vec<f64> = self.weights.iter().map(|w| w.ln()).collect();
        for i in 0..n {
            let row = data.row(i);
            let mut log_terms = vec![f64::NEG_INFINITY; self.k];
            let mut max_term = f64::NEG_INFINITY;
            for j in 0..self.k {
                let lt = log_w[j] + comp[j].log_density(row);
                log_terms[j] = lt;
                if lt > max_term {
                    max_term = lt;
                }
            }
            out[i] = log_sum_exp(&log_terms, max_term);
        }
        Ok(out)
    }

    /// Schwarz BIC approximation to negative log evidence, divided by two so
    /// it shares the ordinary negative-log-likelihood scale used by the shape
    /// race. Lower is better.
    pub fn bic(&self) -> f64 {
        -self.loglik + 0.5 * self.num_free_parameters() as f64 * (self.n_obs as f64).ln()
    }
}

/// Cached per-component Gaussian evaluator: mean, precision `Σ⁻¹`, and the
/// log-normalizing constant `−½(d log 2π + log|Σ|)`.
#[derive(Debug, Clone)]
struct GaussianComponentEval {
    residual_origin: Array1<f64>,
    residual_scale: Array1<f64>,
    residual_normalized_offset: Array1<f64>,
    precision: Array2<f64>,
    log_norm: f64,
    d: usize,
}

impl GaussianComponentEval {
    fn factor(mean: ArrayView1<'_, f64>, cov: &Array2<f64>) -> Result<Self, String> {
        let d = mean.len();
        if mean.iter().any(|value| !value.is_finite()) {
            return Err("mixture component mean must be finite".to_string());
        }
        if cov.nrows() != d || cov.ncols() != d {
            return Err(format!(
                "mixture component covariance must be {d}x{d}, got {}x{}",
                cov.nrows(),
                cov.ncols()
            ));
        }
        let (evals, evecs) = cov
            .eigh(Side::Lower)
            .map_err(|e| format!("mixture component covariance eigendecomposition failed: {e}"))?;
        let mut log_det = 0.0_f64;
        let mut inv_evals = Array1::<f64>::zeros(d);
        for (idx, &ev) in evals.iter().enumerate() {
            if !ev.is_finite() || ev <= 0.0 {
                return Err(format!(
                    "mixture component covariance is not SPD: eigenvalue {idx} is {ev:.3e}"
                ));
            }
            log_det += ev.ln();
            let inverse = ev.recip();
            if !inverse.is_finite() {
                return Err(format!(
                    "mixture component precision is not representable: eigenvalue {idx} is {ev:.3e}"
                ));
            }
            inv_evals[idx] = inverse;
        }
        // Σ⁻¹ = V diag(1/λ) Vᵀ.
        let mut precision = Array2::<f64>::zeros((d, d));
        for a in 0..d {
            for b in 0..d {
                let mut acc = 0.0_f64;
                for m in 0..d {
                    acc += evecs[[a, m]] * inv_evals[m] * evecs[[b, m]];
                }
                precision[[a, b]] = acc;
            }
        }
        let log_norm = -0.5 * (d as f64 * (2.0 * std::f64::consts::PI).ln() + log_det);
        if precision.iter().any(|value| !value.is_finite()) || !log_norm.is_finite() {
            return Err(
                "mixture component factorization produced non-finite precision or log normalizer"
                    .to_string(),
            );
        }
        Ok(Self {
            residual_origin: mean.to_owned(),
            residual_scale: Array1::zeros(d),
            residual_normalized_offset: Array1::zeros(d),
            precision,
            log_norm,
            d,
        })
    }

    fn isotropic(charts: &[StableScalarMeanChart], variance: f64) -> Result<Self, String> {
        let d = charts.len();
        if d == 0 {
            return Err("isotropic Gaussian density requires positive dimension".to_string());
        }
        if !(variance.is_finite() && variance > 0.0) {
            return Err(format!(
                "isotropic Gaussian variance must be finite and positive, got {variance}"
            ));
        }
        let inverse_variance = variance.recip();
        if !inverse_variance.is_finite() {
            return Err(format!(
                "isotropic Gaussian precision is non-finite for variance {variance}"
            ));
        }
        let mut precision = Array2::<f64>::zeros((d, d));
        for axis in 0..d {
            precision[[axis, axis]] = inverse_variance;
        }
        let log_norm = -0.5 * d as f64 * ((2.0 * std::f64::consts::PI).ln() + variance.ln());
        if !log_norm.is_finite() {
            return Err("isotropic Gaussian log normalizer is non-finite".to_string());
        }
        Ok(Self {
            residual_origin: Array1::from_iter(charts.iter().map(|chart| chart.origin)),
            residual_scale: Array1::from_iter(charts.iter().map(|chart| chart.scale)),
            residual_normalized_offset: Array1::from_iter(
                charts.iter().map(|chart| chart.normalized_offset),
            ),
            precision,
            log_norm,
            d,
        })
    }

    #[inline]
    fn log_density(&self, y: ArrayView1<'_, f64>) -> f64 {
        let residual = self.residual(y);
        let pv = self.precision_times_residual(&residual);
        let mut quad = 0.0_f64;
        for c in 0..self.d {
            quad += residual[c] * pv[c];
        }
        self.log_norm - 0.5 * quad
    }

    #[inline]
    fn residual(&self, y: ArrayView1<'_, f64>) -> Vec<f64> {
        let mut residual = vec![0.0_f64; self.d];
        for axis in 0..self.d {
            residual[axis] = (-self.residual_normalized_offset[axis]).mul_add(
                self.residual_scale[axis],
                y[axis] - self.residual_origin[axis],
            );
        }
        residual
    }

    /// `Σ⁻¹ (y − μ)`.
    #[inline]
    fn precision_times_residual(&self, residual: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.d];
        for a in 0..self.d {
            let mut acc = 0.0_f64;
            for b in 0..self.d {
                acc += self.precision[[a, b]] * residual[b];
            }
            out[a] = acc;
        }
        out
    }
}

#[inline]
fn log_sum_exp(terms: &[f64], max_term: f64) -> f64 {
    if !max_term.is_finite() {
        return f64::NEG_INFINITY;
    }
    let mut acc = 0.0_f64;
    for &t in terms {
        acc += (t - max_term).exp();
    }
    max_term + acc.ln()
}

fn evidence_matrix_fingerprint(namespace: &str, values: ArrayView2<'_, f64>) -> Fingerprint {
    let mut hasher = Fingerprinter::new();
    hasher.write_str(namespace);
    hasher.write_usize(values.nrows());
    hasher.write_usize(values.ncols());
    // Hash logical row-major iteration order rather than the backing storage,
    // so an equivalent strided view resumes the same mathematical problem.
    for &value in values {
        hasher.write_f64(value);
    }
    hasher.finalize()
}

fn mixture_data_fingerprint(data: ArrayView2<'_, f64>) -> Fingerprint {
    evidence_matrix_fingerprint("gaussian-mixture-em-v1", data)
}

/// Fit a `k`-component full-covariance Gaussian mixture by deterministic
/// k-means++-style seeding (reusing the `terms::basis` farthest-point k-means,
/// a pure function of the data — no clock randomness) followed by EM to the
/// configured tolerance.
///
/// The fit is deterministic given `(data, k, config)`: the seed is the
/// farthest-point/k-means center selection, EM is a deterministic map, so
/// re-running yields the identical mixture.
pub fn fit_gaussian_mixture(
    data: ArrayView2<'_, f64>,
    k: usize,
    config: GaussianMixtureConfig,
) -> Result<GaussianMixtureFit, GaussianMixtureError> {
    validate_gaussian_mixture_problem(data, k, config)?;
    // Deterministic k-means++-style seeding via the shared basis k-means
    // (farthest-point init + Lloyd iterations).
    let means = gam_terms::basis::select_centers_by_strategy(
        data,
        &gam_terms::basis::CenterStrategy::KMeans {
            num_centers: k,
            max_iter: config.kmeans_max_iter,
        },
    )
    .map_err(|error| GaussianMixtureError::NumericalFailure {
        message: format!("deterministic k-means seeding failed: {error}"),
        checkpoint: None,
    })?;
    if means.nrows() != k || means.ncols() != data.ncols() {
        return Err(GaussianMixtureError::NumericalFailure {
            message: format!(
                "seeding returned {}x{} centers, expected {k}x{}",
                means.nrows(),
                means.ncols(),
                data.ncols()
            ),
            checkpoint: None,
        });
    }
    let global_covariance =
        constrained_data_covariance(data, config.covariance_floor).map_err(|message| {
            GaussianMixtureError::NumericalFailure {
                message,
                checkpoint: None,
            }
        })?;
    let weights = Array1::<f64>::from_elem(k, 1.0 / k as f64);
    let covariances = vec![global_covariance; k];
    let initial_e_step =
        mixture_e_step(data, &weights, &means, &covariances).map_err(|message| {
            GaussianMixtureError::NumericalFailure {
                message,
                checkpoint: None,
            }
        })?;
    let data_fingerprint = mixture_data_fingerprint(data);
    let checkpoint = GaussianMixtureCheckpoint {
        weights,
        means,
        covariances,
        mean_log_likelihood: initial_e_step.mean_log_likelihood,
        completed_iterations: 0,
        data_fingerprint,
        covariance_floor: config.covariance_floor,
    };
    run_gaussian_mixture_em(data, config, checkpoint)
}

/// Resume EM from the exact state carried by [`GaussianMixtureError`].
pub fn resume_gaussian_mixture(
    data: ArrayView2<'_, f64>,
    config: GaussianMixtureConfig,
    checkpoint: GaussianMixtureCheckpoint,
) -> Result<GaussianMixtureFit, GaussianMixtureError> {
    let k = checkpoint.weights.len();
    validate_gaussian_mixture_problem(data, k, config)?;
    validate_gaussian_mixture_checkpoint(data, config.covariance_floor, &checkpoint)?;
    run_gaussian_mixture_em(data, config, checkpoint)
}

fn validate_gaussian_mixture_problem(
    data: ArrayView2<'_, f64>,
    k: usize,
    config: GaussianMixtureConfig,
) -> Result<(), GaussianMixtureError> {
    let n = data.nrows();
    let d = data.ncols();
    if k == 0 {
        return Err(GaussianMixtureError::InvalidInput {
            message: "k must be positive".to_string(),
        });
    }
    if d == 0 {
        return Err(GaussianMixtureError::InvalidInput {
            message: "at least one data column is required".to_string(),
        });
    }
    if k > n {
        return Err(GaussianMixtureError::InvalidInput {
            message: format!("requested {k} components but data has {n} rows"),
        });
    }
    if data.iter().any(|value| !value.is_finite()) {
        return Err(GaussianMixtureError::InvalidInput {
            message: "data must be finite".to_string(),
        });
    }
    if config.max_iter == 0 || config.kmeans_max_iter == 0 {
        return Err(GaussianMixtureError::InvalidInput {
            message: "max_iter and kmeans_max_iter must be positive".to_string(),
        });
    }
    let numerical_floor = f64::EPSILON.sqrt();
    if !config.loglik_tol.is_finite()
        || config.loglik_tol < numerical_floor
        || !config.parameter_tol.is_finite()
        || config.parameter_tol < numerical_floor
        || !config.covariance_floor.is_finite()
        || config.covariance_floor <= 0.0
    {
        return Err(GaussianMixtureError::InvalidInput {
            message: format!(
                "loglik_tol and parameter_tol must be finite and >= {numerical_floor:.3e}, and covariance_floor must be finite and positive"
            ),
        });
    }
    Ok(())
}

fn validate_gaussian_mixture_checkpoint(
    data: ArrayView2<'_, f64>,
    covariance_floor: f64,
    checkpoint: &GaussianMixtureCheckpoint,
) -> Result<(), GaussianMixtureError> {
    let d = data.ncols();
    let k = checkpoint.weights.len();
    let mass = checkpoint.weights.sum();
    if k == 0
        || checkpoint.data_fingerprint != mixture_data_fingerprint(data)
        || checkpoint.covariance_floor.to_bits() != covariance_floor.to_bits()
        || checkpoint.means.dim() != (k, d)
        || checkpoint.covariances.len() != k
        || checkpoint
            .covariances
            .iter()
            .any(|covariance| covariance.dim() != (d, d))
        || checkpoint
            .weights
            .iter()
            .chain(checkpoint.means.iter())
            .chain(checkpoint.covariances.iter().flat_map(|value| value.iter()))
            .any(|value| !value.is_finite())
        || checkpoint.weights.iter().any(|value| *value <= 0.0)
        || !mass.is_finite()
        || (mass - 1.0).abs() > f64::EPSILON.sqrt()
        || !checkpoint.mean_log_likelihood.is_finite()
    {
        return Err(GaussianMixtureError::InvalidInput {
            message: "checkpoint problem identity, dimensions, interior parameters, likelihood, or simplex mass are invalid".to_string(),
        });
    }
    Ok(())
}

fn run_gaussian_mixture_em(
    data: ArrayView2<'_, f64>,
    config: GaussianMixtureConfig,
    mut checkpoint: GaussianMixtureCheckpoint,
) -> Result<GaussianMixtureFit, GaussianMixtureError> {
    validate_gaussian_mixture_checkpoint(data, config.covariance_floor, &checkpoint)?;
    let k = checkpoint.weights.len();
    let d = data.ncols();
    let data_fingerprint = mixture_data_fingerprint(data);

    // Certify the CURRENT checkpoint before accepting another EM update. The
    // inclusive bound permits exactly `max_iter` accepted updates and then one
    // final map evaluation at the resulting checkpoint. Consequently every
    // success and every exhaustion pairs its certificate with the exact same
    // parameter state; a certificate for theta_t can never be attached to
    // theta_{t+1} merely because the work boundary was reached.
    for additional_updates in 0..=config.max_iter {
        let current = mixture_e_step(
            data,
            &checkpoint.weights,
            &checkpoint.means,
            &checkpoint.covariances,
        )
        .map_err(|message| GaussianMixtureError::NumericalFailure {
            message,
            checkpoint: Some(checkpoint.clone()),
        })?;
        if (checkpoint.mean_log_likelihood - current.mean_log_likelihood).abs()
            > current.mean_log_likelihood_roundoff
        {
            return Err(GaussianMixtureError::InvalidInput {
                message: format!(
                    "checkpoint mean log likelihood {:.12e} disagrees with its parameters ({:.12e} +/- {:.3e})",
                    checkpoint.mean_log_likelihood,
                    current.mean_log_likelihood,
                    current.mean_log_likelihood_roundoff
                ),
            });
        }
        checkpoint.mean_log_likelihood = current.mean_log_likelihood;

        let (next_weights, next_means, next_covariances) = mixture_m_step(
            data,
            current.responsibilities.view(),
            config.covariance_floor,
        )
        .map_err(|message| GaussianMixtureError::NumericalFailure {
            message,
            checkpoint: Some(checkpoint.clone()),
        })?;
        let next = mixture_e_step(data, &next_weights, &next_means, &next_covariances).map_err(
            |message| GaussianMixtureError::NumericalFailure {
                message,
                checkpoint: Some(checkpoint.clone()),
            },
        )?;
        let objective_scale = current
            .mean_log_likelihood
            .abs()
            .max(next.mean_log_likelihood.abs())
            .max(1.0);
        let objective_step = next.mean_log_likelihood - current.mean_log_likelihood;
        let objective_residual = objective_step.abs() / objective_scale;
        let parameter_residual = mixture_parameter_residual(
            &checkpoint.weights,
            &checkpoint.means,
            &checkpoint.covariances,
            &next_weights,
            &next_means,
            &next_covariances,
        );
        let monotonicity_uncertainty = gaussian_mixture_monotonicity_uncertainty(
            objective_scale,
            current.mean_log_likelihood_roundoff,
            next.mean_log_likelihood_roundoff,
        );
        let certificate = GaussianMixtureCertificate {
            mean_log_likelihood: current.mean_log_likelihood,
            mean_log_likelihood_gain: objective_step,
            monotonicity_uncertainty,
            objective_residual,
            objective_tolerance: config.loglik_tol,
            parameter_residual,
            parameter_tolerance: config.parameter_tol,
        };
        if objective_step < -monotonicity_uncertainty {
            return Err(GaussianMixtureError::MonotonicityViolation {
                previous_mean_log_likelihood: current.mean_log_likelihood,
                next_mean_log_likelihood: next.mean_log_likelihood,
                numerical_uncertainty: monotonicity_uncertainty,
                checkpoint,
            });
        }
        if objective_residual <= config.loglik_tol && parameter_residual <= config.parameter_tol {
            let loglik = current.mean_log_likelihood * data.nrows() as f64;
            if !loglik.is_finite() {
                return Err(GaussianMixtureError::NumericalFailure {
                    message: "certified mean log likelihood overflows as a total likelihood"
                        .to_string(),
                    checkpoint: Some(checkpoint),
                });
            }
            return Ok(GaussianMixtureFit {
                weights: checkpoint.weights,
                means: checkpoint.means,
                covariances: checkpoint.covariances,
                k,
                d,
                n_obs: data.nrows(),
                loglik,
                iterations: checkpoint.completed_iterations,
                certificate,
            });
        }
        if additional_updates == config.max_iter {
            return Err(GaussianMixtureError::DidNotConverge {
                max_iterations: config.max_iter,
                certificate,
                checkpoint,
            });
        }
        checkpoint = GaussianMixtureCheckpoint {
            weights: next_weights,
            means: next_means,
            covariances: next_covariances,
            mean_log_likelihood: next.mean_log_likelihood,
            completed_iterations: checkpoint.completed_iterations + 1,
            data_fingerprint,
            covariance_floor: config.covariance_floor,
        };
    }
    Err(GaussianMixtureError::NumericalFailure {
        message: format!(
            "EM refinement exhausted its inclusive update budget ({}) without producing a \
             terminal verdict",
            config.max_iter
        ),
        checkpoint: Some(checkpoint),
    })
}

struct GaussianMixtureEStep {
    responsibilities: Array2<f64>,
    mean_log_likelihood: f64,
    mean_log_likelihood_roundoff: f64,
}

/// Resolution of one observed EM likelihood comparison.
///
/// `pairwise_mean_with_roundoff` bounds only the final reduction of already
/// rounded row log likelihoods. An EM comparison also traverses covariance
/// eigendecompositions, precision quadratics, log-sum-exp, the M-step, and a
/// second E-step. Treating the reduction bound as a bound for that whole map
/// is false precision and turns cancellation at a stationary point into a
/// spurious monotonicity violation. The square root of machine epsilon is the
/// numerical resolution already required of every configured EM tolerance;
/// scaling it by the observed objective magnitude makes the invariant
/// independent of data units and of user-selected stopping knobs.
fn gaussian_mixture_monotonicity_uncertainty(
    objective_scale: f64,
    current_reduction_roundoff: f64,
    next_reduction_roundoff: f64,
) -> f64 {
    let reduction_roundoff = current_reduction_roundoff + next_reduction_roundoff;
    let composite_map_resolution = f64::EPSILON.sqrt() * objective_scale;
    reduction_roundoff.max(composite_map_resolution)
}

fn pairwise_sum_max_depth(term_count: usize) -> usize {
    if term_count <= 1 {
        return 0;
    }
    let within_block = term_count.min(BASE_CHUNK) - 1;
    let blocks = term_count.div_ceil(BASE_CHUNK);
    let tree_levels = if blocks <= 1 {
        0
    } else {
        (usize::BITS - (blocks - 1).leading_zeros()) as usize
    };
    within_block.saturating_add(tree_levels)
}

fn pairwise_mean_with_roundoff(mut values: Vec<f64>) -> Result<(f64, f64), String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("mean log-likelihood terms must be nonempty and finite".to_string());
    }
    let sum = pairwise_sum(&values);
    for value in &mut values {
        *value = value.abs();
    }
    let magnitude_sum = pairwise_sum(&values);
    let unit_roundoff = 0.5 * f64::EPSILON;
    let accumulated = pairwise_sum_max_depth(values.len()) as f64 * unit_roundoff;
    let addition_bound = if accumulated < 1.0 {
        accumulated / (1.0 - accumulated) * magnitude_sum
    } else {
        f64::INFINITY
    };
    let count = values.len() as f64;
    let mean = sum / count;
    // The first term bounds the deterministic pairwise additions; the second
    // bounds the final division. This tolerance is derived from the actual
    // reduction depth and magnitudes, independently of the EM stopping knob.
    let roundoff = addition_bound / count + unit_roundoff * mean.abs();
    if !(mean.is_finite() && roundoff.is_finite()) {
        return Err("mean mixture log likelihood or its rounding bound is non-finite".to_string());
    }
    Ok((mean, roundoff))
}

fn mixture_e_step(
    data: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    means: &Array2<f64>,
    covariances: &[Array2<f64>],
) -> Result<GaussianMixtureEStep, String> {
    let n = data.nrows();
    let k = weights.len();
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err("mixture E-step requires strictly positive finite weights".to_string());
    }
    let mut components = Vec::with_capacity(k);
    for component in 0..k {
        components.push(GaussianComponentEval::factor(
            means.row(component),
            &covariances[component],
        )?);
    }
    let log_weights: Vec<f64> = weights.iter().map(|weight| weight.ln()).collect();
    let mut responsibilities = Array2::<f64>::zeros((n, k));
    let mut row_log_likelihoods = Vec::with_capacity(n);
    for row in 0..n {
        let observation = data.row(row);
        let mut log_terms = vec![f64::NEG_INFINITY; k];
        let mut max_term = f64::NEG_INFINITY;
        for component in 0..k {
            let term = log_weights[component] + components[component].log_density(observation);
            log_terms[component] = term;
            max_term = max_term.max(term);
        }
        let log_mixture = log_sum_exp(&log_terms, max_term);
        if !log_mixture.is_finite() {
            return Err(format!(
                "mixture density is non-finite at training row {row}"
            ));
        }
        row_log_likelihoods.push(log_mixture);
        for component in 0..k {
            responsibilities[[row, component]] = (log_terms[component] - log_mixture).exp();
        }
    }
    let (mean_log_likelihood, mean_log_likelihood_roundoff) =
        pairwise_mean_with_roundoff(row_log_likelihoods)?;
    Ok(GaussianMixtureEStep {
        responsibilities,
        mean_log_likelihood,
        mean_log_likelihood_roundoff,
    })
}

fn mixture_m_step(
    data: ArrayView2<'_, f64>,
    responsibilities: ArrayView2<'_, f64>,
    covariance_floor: f64,
) -> Result<(Array1<f64>, Array2<f64>, Vec<Array2<f64>>), String> {
    let n = data.nrows();
    let d = data.ncols();
    let k = responsibilities.ncols();
    let mut component_mass = Array1::<f64>::zeros(k);
    for component in 0..k {
        component_mass[component] = responsibilities.column(component).sum();
    }
    if component_mass
        .iter()
        .any(|mass| !mass.is_finite() || *mass <= 0.0)
    {
        return Err(
            "M-step reached a zero-mass component; the requested mixture order has no interior fitted density"
                .to_string(),
        );
    }
    let mut weights = component_mass.mapv(|mass| mass / n as f64);
    let total_weight = weights.sum();
    if !(total_weight.is_finite() && total_weight > 0.0) {
        return Err("M-step produced invalid mixture-weight mass".to_string());
    }
    weights.mapv_inplace(|weight| weight / total_weight);
    let mut means = Array2::<f64>::zeros((k, d));
    let mut covariances = Vec::with_capacity(k);
    for component in 0..k {
        let mass = component_mass[component];
        let mut mean = Array1::<f64>::zeros(d);
        for row in 0..n {
            let responsibility = responsibilities[[row, component]];
            for col in 0..d {
                mean[col] += responsibility * data[[row, col]];
            }
        }
        mean.mapv_inplace(|value| value / mass);
        means.row_mut(component).assign(&mean);
        let mut covariance = Array2::<f64>::zeros((d, d));
        for row in 0..n {
            let responsibility = responsibilities[[row, component]];
            for left in 0..d {
                let left_residual = data[[row, left]] - mean[left];
                for right in 0..d {
                    covariance[[left, right]] +=
                        responsibility * left_residual * (data[[row, right]] - mean[right]);
                }
            }
        }
        covariance.mapv_inplace(|value| value / mass);
        covariances.push(constrain_covariance(covariance, covariance_floor)?);
    }
    Ok((weights, means, covariances))
}

fn mixture_parameter_residual(
    previous_weights: &Array1<f64>,
    previous_means: &Array2<f64>,
    previous_covariances: &[Array2<f64>],
    next_weights: &Array1<f64>,
    next_means: &Array2<f64>,
    next_covariances: &[Array2<f64>],
) -> f64 {
    let relative_step = |previous: f64, next: f64| {
        (next - previous).abs() / previous.abs().max(next.abs()).max(1.0)
    };
    previous_weights
        .iter()
        .zip(next_weights.iter())
        .map(|(&previous, &next)| relative_step(previous, next))
        .chain(
            previous_means
                .iter()
                .zip(next_means.iter())
                .map(|(&previous, &next)| relative_step(previous, next)),
        )
        .chain(
            previous_covariances
                .iter()
                .zip(next_covariances.iter())
                .flat_map(|(previous, next)| previous.iter().zip(next.iter()))
                .map(|(&previous, &next)| relative_step(previous, next)),
        )
        .fold(0.0_f64, f64::max)
}

fn constrain_covariance(covariance: Array2<f64>, floor: f64) -> Result<Array2<f64>, String> {
    let (eigenvalues, eigenvectors) = covariance
        .eigh(Side::Lower)
        .map_err(|error| format!("covariance eigendecomposition failed: {error}"))?;
    let d = covariance.nrows();
    let mut constrained = Array2::<f64>::zeros((d, d));
    for row in 0..d {
        for col in 0..d {
            let mut value = 0.0_f64;
            for index in 0..d {
                value += eigenvectors[[row, index]]
                    * eigenvalues[index].max(floor)
                    * eigenvectors[[col, index]];
            }
            constrained[[row, col]] = value;
        }
    }
    if constrained.iter().any(|value| !value.is_finite()) {
        return Err("constrained covariance became non-finite".to_string());
    }
    Ok(constrained)
}

/// Global constrained covariance used to seed EM.
fn constrained_data_covariance(
    data: ArrayView2<'_, f64>,
    floor: f64,
) -> Result<Array2<f64>, String> {
    let n = data.nrows();
    let d = data.ncols();
    let mut mean = Array1::<f64>::zeros(d);
    for i in 0..n {
        for c in 0..d {
            mean[c] += data[[i, c]];
        }
    }
    mean.mapv_inplace(|v| v / n.max(1) as f64);
    let mut cov = Array2::<f64>::zeros((d, d));
    for i in 0..n {
        for a in 0..d {
            let da = data[[i, a]] - mean[a];
            for b in 0..d {
                cov[[a, b]] += da * (data[[i, b]] - mean[b]);
            }
        }
    }
    let inv = 1.0 / n as f64;
    cov.mapv_inplace(|v| v * inv);
    constrain_covariance(cov, floor)
}

// ---------------------------------------------------------------------------
// Ring-of-clusters candidate (#2262)
// ---------------------------------------------------------------------------
//
// A free Gaussian mixture treats the component means as unrelated points. That
// is the wrong null for a discrete cyclic concept: weekdays and months form
// tight clusters, but their component means share a low-dimensional circular
// constraint. `RingGaussianMixtureFit` models exactly that density,
//
//     x | z=j ~ N(c + r u_j, sigma^2 I_2),  ||u_j|| = 1,
//
// with free mixture weights, a shared center/radius, one angle per component,
// and a shared isotropic variance. Its `2k + 3` continuous parameters are
// priced by the same BIC-form criterion as the unconstrained mixture's `6k - 1`
// parameters in two dimensions.

/// Certified Gaussian mixture whose component centers lie on one fitted circle.
#[derive(Debug, Clone)]
pub struct RingGaussianMixtureFit {
    weights: Array1<f64>,
    center: Array1<f64>,
    radius: f64,
    directions: Array2<f64>,
    variance: f64,
    k: usize,
    n_obs: usize,
    loglik: f64,
    iterations: usize,
    certificate: GaussianMixtureCertificate,
}

impl RingGaussianMixtureFit {
    pub fn weights(&self) -> ArrayView1<'_, f64> {
        self.weights.view()
    }

    pub fn center(&self) -> ArrayView1<'_, f64> {
        self.center.view()
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    pub fn directions(&self) -> ArrayView2<'_, f64> {
        self.directions.view()
    }

    pub fn variance(&self) -> f64 {
        self.variance
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn certificate(&self) -> GaussianMixtureCertificate {
        self.certificate
    }

    /// Free parameters: `k-1` weight logits, center(2), radius(1), `k`
    /// component angles, and shared log standard deviation(1).
    pub fn num_free_parameters(&self) -> usize {
        2 * self.k + 3
    }

    pub fn per_point_log_density(&self, data: ArrayView2<'_, f64>) -> Result<Array1<f64>, String> {
        if data.ncols() != 2 {
            return Err(format!(
                "ring-of-clusters density expects two columns, got {}",
                data.ncols()
            ));
        }
        ring_mixture_log_density(
            data,
            &self.weights,
            &self.center,
            self.radius,
            &self.directions,
            self.variance,
        )
    }

    /// Schwarz BIC approximation to negative log evidence, divided by two so
    /// it is on the ordinary negative-log-likelihood scale. Lower is better.
    pub fn bic(&self) -> f64 {
        -self.loglik + 0.5 * self.num_free_parameters() as f64 * (self.n_obs as f64).ln()
    }
}

#[derive(Debug, Clone)]
struct RingMixtureState {
    weights: Array1<f64>,
    center: Array1<f64>,
    radius: f64,
    directions: Array2<f64>,
    variance: f64,
    mean_log_likelihood: f64,
    completed_iterations: usize,
}

fn ring_component_means(
    center: &Array1<f64>,
    radius: f64,
    directions: &Array2<f64>,
) -> Array2<f64> {
    let mut means = Array2::<f64>::zeros((directions.nrows(), 2));
    for component in 0..directions.nrows() {
        means[[component, 0]] = center[0] + radius * directions[[component, 0]];
        means[[component, 1]] = center[1] + radius * directions[[component, 1]];
    }
    means
}

fn ring_mixture_log_terms(
    data: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    center: &Array1<f64>,
    radius: f64,
    directions: &Array2<f64>,
    variance: f64,
) -> Result<(Array2<f64>, Vec<f64>), String> {
    if data.ncols() != 2
        || center.len() != 2
        || directions.ncols() != 2
        || directions.nrows() != weights.len()
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
        || !(radius.is_finite() && radius > 0.0)
        || !(variance.is_finite() && variance > 0.0)
    {
        return Err("invalid ring-of-clusters parameter state".to_string());
    }
    let means = ring_component_means(center, radius, directions);
    let log_normalizer = -(std::f64::consts::TAU).ln() - variance.ln();
    let mut terms = Array2::<f64>::zeros((data.nrows(), weights.len()));
    let mut row_log_likelihoods = Vec::with_capacity(data.nrows());
    for row in 0..data.nrows() {
        let mut max_term = f64::NEG_INFINITY;
        for component in 0..weights.len() {
            let dx = data[[row, 0]] - means[[component, 0]];
            let dy = data[[row, 1]] - means[[component, 1]];
            let term =
                weights[component].ln() + log_normalizer - 0.5 * (dx * dx + dy * dy) / variance;
            terms[[row, component]] = term;
            max_term = max_term.max(term);
        }
        let values = terms.row(row).to_vec();
        let log_likelihood = log_sum_exp(&values, max_term);
        if !log_likelihood.is_finite() {
            return Err(format!(
                "ring-of-clusters density is non-finite at training row {row}"
            ));
        }
        row_log_likelihoods.push(log_likelihood);
    }
    Ok((terms, row_log_likelihoods))
}

fn ring_mixture_e_step(
    data: ArrayView2<'_, f64>,
    state: &RingMixtureState,
) -> Result<(Array2<f64>, f64, f64), String> {
    let (terms, row_log_likelihoods) = ring_mixture_log_terms(
        data,
        &state.weights,
        &state.center,
        state.radius,
        &state.directions,
        state.variance,
    )?;
    let mut responsibilities = Array2::<f64>::zeros(terms.raw_dim());
    for row in 0..terms.nrows() {
        for component in 0..terms.ncols() {
            responsibilities[[row, component]] =
                (terms[[row, component]] - row_log_likelihoods[row]).exp();
        }
    }
    let (mean, roundoff) = pairwise_mean_with_roundoff(row_log_likelihoods)?;
    Ok((responsibilities, mean, roundoff))
}

fn ring_mixture_log_density(
    data: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    center: &Array1<f64>,
    radius: f64,
    directions: &Array2<f64>,
    variance: f64,
) -> Result<Array1<f64>, String> {
    let (_, row_log_likelihoods) =
        ring_mixture_log_terms(data, weights, center, radius, directions, variance)?;
    Ok(Array1::from_vec(row_log_likelihoods))
}

fn relative_parameter_step(previous: f64, next: f64) -> f64 {
    (next - previous).abs() / previous.abs().max(next.abs()).max(1.0)
}

/// Distance between two ring-mixture states in identifiable density space.
///
/// `center`, `radius`, and `directions` are only a factorization of the actual
/// component means `m_j = center + radius * direction_j`. Near a large-radius
/// (locally flat) circle those factors can move appreciably while cancelling in
/// every `m_j`; requiring the factors themselves to stop is therefore neither
/// necessary for distributional convergence nor invariant to the chosen circle
/// chart. We instead certify weights, component means in noise-standardized
/// units, and log variance. Component labels remain aligned because the E/M
/// maps preserve their responsibility-column order; no sorting or relabeling
/// occurs between `previous` and `next`.
fn ring_identifiable_parameter_residual(
    previous: &RingMixtureState,
    next: &RingMixtureState,
) -> f64 {
    let previous_means =
        ring_component_means(&previous.center, previous.radius, &previous.directions);
    let next_means = ring_component_means(&next.center, next.radius, &next.directions);
    let noise_scale = previous
        .variance
        .sqrt()
        .max(next.variance.sqrt())
        .max(f64::MIN_POSITIVE);
    let weight_residual = previous
        .weights
        .iter()
        .zip(next.weights.iter())
        .map(|(&left, &right)| (right - left).abs())
        .fold(0.0, f64::max);
    let mean_residual = previous_means
        .rows()
        .into_iter()
        .zip(next_means.rows())
        .map(|(left, right)| (right[0] - left[0]).hypot(right[1] - left[1]) / noise_scale)
        .fold(0.0, f64::max);
    let variance_residual = (next.variance / previous.variance).ln().abs();
    weight_residual.max(mean_residual).max(variance_residual)
}

fn fit_weighted_component_circle(
    component_means: &Array2<f64>,
    component_mass: &Array1<f64>,
    initial_center: &Array1<f64>,
    initial_radius: f64,
    parameter_tol: f64,
    max_iter: usize,
) -> Result<(Array1<f64>, f64, Array2<f64>), String> {
    let k = component_means.nrows();
    let total_mass = component_mass.sum();
    if component_means.ncols() != 2
        || component_mass.len() != k
        || component_mass
            .iter()
            .any(|mass| !mass.is_finite() || *mass <= 0.0)
        || !(total_mass.is_finite() && total_mass > 0.0)
    {
        return Err("ring M-step requires positive component masses and 2-D means".to_string());
    }
    let mut center = initial_center.clone();
    let mut radius = initial_radius;
    let mut directions = Array2::<f64>::zeros((k, 2));
    for _ in 0..max_iter {
        for component in 0..k {
            let dx = component_means[[component, 0]] - center[0];
            let dy = component_means[[component, 1]] - center[1];
            let norm = dx.hypot(dy);
            if !(norm.is_finite() && norm > 0.0) {
                return Err(
                    "ring M-step reached a component centroid at the circle center; its angle is unidentified"
                        .to_string(),
                );
            }
            directions[[component, 0]] = dx / norm;
            directions[[component, 1]] = dy / norm;
        }

        let mut mean_point = Array1::<f64>::zeros(2);
        let mut mean_direction = Array1::<f64>::zeros(2);
        for component in 0..k {
            let weight = component_mass[component] / total_mass;
            for axis in 0..2 {
                mean_point[axis] += weight * component_means[[component, axis]];
                mean_direction[axis] += weight * directions[[component, axis]];
            }
        }
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for component in 0..k {
            let mass = component_mass[component];
            let dux = directions[[component, 0]] - mean_direction[0];
            let duy = directions[[component, 1]] - mean_direction[1];
            numerator += mass
                * (dux * (component_means[[component, 0]] - mean_point[0])
                    + duy * (component_means[[component, 1]] - mean_point[1]));
            denominator += mass * (dux * dux + duy * duy);
        }
        if !(denominator.is_finite() && denominator > 0.0) {
            return Err(
                "ring M-step component directions are identical; radius and center are unidentified"
                    .to_string(),
            );
        }
        let mut next_radius = numerator / denominator;
        if !next_radius.is_finite() || next_radius == 0.0 {
            return Err("ring M-step produced an unidentified zero radius".to_string());
        }
        if next_radius < 0.0 {
            next_radius = -next_radius;
            directions.mapv_inplace(|value| -value);
        }
        let next_center = Array1::from_vec(vec![
            mean_point[0] - next_radius * mean_direction[0],
            mean_point[1] - next_radius * mean_direction[1],
        ]);
        let residual = center
            .iter()
            .zip(next_center.iter())
            .map(|(&left, &right)| relative_parameter_step(left, right))
            .chain(std::iter::once(relative_parameter_step(
                radius,
                next_radius,
            )))
            .fold(0.0, f64::max);
        center = next_center;
        radius = next_radius;
        if residual <= parameter_tol {
            // Recompute directions at the returned center so the stored angles
            // are the exact angular block update belonging to that center.
            for component in 0..k {
                let dx = component_means[[component, 0]] - center[0];
                let dy = component_means[[component, 1]] - center[1];
                let norm = dx.hypot(dy);
                if !(norm.is_finite() && norm > 0.0) {
                    return Err("ring M-step terminal component angle is unidentified".to_string());
                }
                directions[[component, 0]] = dx / norm;
                directions[[component, 1]] = dy / norm;
            }
            return Ok((center, radius, directions));
        }
    }
    Err(format!(
        "ring M-step did not certify its constrained center/radius fixed point after {max_iter} iterations"
    ))
}

fn ring_mixture_m_step(
    data: ArrayView2<'_, f64>,
    responsibilities: ArrayView2<'_, f64>,
    previous: &RingMixtureState,
    config: GaussianMixtureConfig,
) -> Result<RingMixtureState, String> {
    let n = data.nrows();
    let k = responsibilities.ncols();
    let mut component_mass = Array1::<f64>::zeros(k);
    let mut component_means = Array2::<f64>::zeros((k, 2));
    for component in 0..k {
        let mass = responsibilities.column(component).sum();
        if !(mass.is_finite() && mass > 0.0) {
            return Err(
                "ring M-step reached a zero-mass component; the requested order is singular"
                    .to_string(),
            );
        }
        component_mass[component] = mass;
        for row in 0..n {
            for axis in 0..2 {
                component_means[[component, axis]] +=
                    responsibilities[[row, component]] * data[[row, axis]];
            }
        }
        for axis in 0..2 {
            component_means[[component, axis]] /= mass;
        }
    }
    let mut weights = component_mass.mapv(|mass| mass / n as f64);
    let weight_sum = weights.sum();
    weights.mapv_inplace(|weight| weight / weight_sum);
    let (center, radius, directions) = fit_weighted_component_circle(
        &component_means,
        &component_mass,
        &previous.center,
        previous.radius,
        config.parameter_tol,
        config.max_iter,
    )?;
    let means = ring_component_means(&center, radius, &directions);
    let mut expected_squared_error = 0.0;
    for row in 0..n {
        for component in 0..k {
            let dx = data[[row, 0]] - means[[component, 0]];
            let dy = data[[row, 1]] - means[[component, 1]];
            expected_squared_error += responsibilities[[row, component]] * (dx * dx + dy * dy);
        }
    }
    let variance = (expected_squared_error / (2 * n) as f64).max(config.covariance_floor);
    if !variance.is_finite() {
        return Err("ring M-step produced non-finite shared variance".to_string());
    }
    Ok(RingMixtureState {
        weights,
        center,
        radius,
        directions,
        variance,
        mean_log_likelihood: f64::NAN,
        completed_iterations: previous.completed_iterations + 1,
    })
}

/// Fit a deterministic, certified `k`-component isotropic Gaussian mixture
/// whose component centers are constrained to a common circle.
pub fn fit_ring_gaussian_mixture(
    data: ArrayView2<'_, f64>,
    k: usize,
    config: GaussianMixtureConfig,
) -> Result<RingGaussianMixtureFit, String> {
    validate_gaussian_mixture_problem(data, k, config).map_err(|error| error.to_string())?;
    if data.ncols() != 2 {
        return Err(format!(
            "ring-of-clusters fitting requires exactly two columns, got {}",
            data.ncols()
        ));
    }
    if k < 3 {
        return Err(format!(
            "ring-of-clusters fitting requires at least three component centers, got {k}"
        ));
    }
    let seeded_means = gam_terms::basis::select_centers_by_strategy(
        data,
        &gam_terms::basis::CenterStrategy::KMeans {
            num_centers: k,
            max_iter: config.kmeans_max_iter,
        },
    )
    .map_err(|error| format!("ring-of-clusters deterministic seeding failed: {error}"))?;
    let component_mass = Array1::<f64>::ones(k);
    let mut initial_center = Array1::<f64>::zeros(2);
    for component in 0..k {
        initial_center[0] += seeded_means[[component, 0]] / k as f64;
        initial_center[1] += seeded_means[[component, 1]] / k as f64;
    }
    let mut initial_radius = 0.0;
    for component in 0..k {
        initial_radius += (seeded_means[[component, 0]] - initial_center[0])
            .hypot(seeded_means[[component, 1]] - initial_center[1])
            / k as f64;
    }
    if !(initial_radius.is_finite() && initial_radius > 0.0) {
        return Err("ring-of-clusters seed has an unidentified zero radius".to_string());
    }
    let (center, radius, directions) = fit_weighted_component_circle(
        &seeded_means,
        &component_mass,
        &initial_center,
        initial_radius,
        config.parameter_tol,
        config.max_iter,
    )?;
    let means = ring_component_means(&center, radius, &directions);
    let mut squared_error = 0.0;
    for row in 0..data.nrows() {
        let mut nearest = f64::INFINITY;
        for component in 0..k {
            let dx = data[[row, 0]] - means[[component, 0]];
            let dy = data[[row, 1]] - means[[component, 1]];
            nearest = nearest.min(dx * dx + dy * dy);
        }
        squared_error += nearest;
    }
    let variance = (squared_error / (2 * data.nrows()) as f64).max(config.covariance_floor);
    let mut state = RingMixtureState {
        weights: Array1::from_elem(k, 1.0 / k as f64),
        center,
        radius,
        directions,
        variance,
        mean_log_likelihood: f64::NAN,
        completed_iterations: 0,
    };
    for additional_updates in 0..=config.max_iter {
        let (responsibilities, current_mean, current_roundoff) = ring_mixture_e_step(data, &state)?;
        state.mean_log_likelihood = current_mean;
        let mut next = ring_mixture_m_step(data, responsibilities.view(), &state, config)?;
        let (_, next_mean, next_roundoff) = ring_mixture_e_step(data, &next)?;
        next.mean_log_likelihood = next_mean;
        let objective_scale = current_mean.abs().max(next_mean.abs()).max(1.0);
        let objective_step = next_mean - current_mean;
        let objective_residual = objective_step.abs() / objective_scale;
        let parameter_residual = ring_identifiable_parameter_residual(&state, &next);
        let monotonicity_uncertainty = gaussian_mixture_monotonicity_uncertainty(
            objective_scale,
            current_roundoff,
            next_roundoff,
        );
        let certificate = GaussianMixtureCertificate {
            mean_log_likelihood: current_mean,
            mean_log_likelihood_gain: objective_step,
            monotonicity_uncertainty,
            objective_residual,
            objective_tolerance: config.loglik_tol,
            parameter_residual,
            parameter_tolerance: config.parameter_tol,
        };
        if objective_step < -monotonicity_uncertainty {
            return Err(format!(
                "ring-of-clusters generalized EM violated monotone ascent at iteration {}: {current_mean:.12e} -> {next_mean:.12e} (comparison uncertainty {monotonicity_uncertainty:.3e})",
                state.completed_iterations
            ));
        }
        if objective_residual <= config.loglik_tol && parameter_residual <= config.parameter_tol {
            let loglik = current_mean * data.nrows() as f64;
            if !loglik.is_finite() {
                return Err("ring-of-clusters total log likelihood overflowed".to_string());
            }
            return Ok(RingGaussianMixtureFit {
                weights: state.weights,
                center: state.center,
                radius: state.radius,
                directions: state.directions,
                variance: state.variance,
                k,
                n_obs: data.nrows(),
                loglik,
                iterations: state.completed_iterations,
                certificate,
            });
        }
        if additional_updates == config.max_iter {
            return Err(format!(
                "ring-of-clusters generalized EM did not certify after {} iterations: objective residual {:.6e}/{:.3e}, parameter-map residual {:.6e}/{:.3e}",
                config.max_iter,
                objective_residual,
                config.loglik_tol,
                parameter_residual,
                config.parameter_tol,
            ));
        }
        state = next;
    }
    Err("ring-of-clusters generalized EM exhausted without a terminal certificate".to_string())
}

// ---------------------------------------------------------------------------
// Circular Gaussian density and structured-union candidates (#907)
// ---------------------------------------------------------------------------

/// Maximum-likelihood fit of a Gaussian-blurred circle in two dimensions.
///
/// The generative model is
///
/// `X = center + radius * U + epsilon`,
///
/// where `U` is uniform on the unit circle and
/// `epsilon ~ N(0, noise_variance * I_2)`. Integrating out `U` gives the proper
/// Cartesian density
///
/// `p(x) = exp(-(r^2 + R^2)/(2s)) I0(Rr/s) / (2 pi s)`.
///
/// Unlike a Gaussian density assigned directly to the nonnegative radius, this
/// density is normalized on the plane, remains finite at the center, and has no
/// artificial `1/r` singularity. The center is fitted jointly with `(R, s)` by
/// latent-angle EM instead of being frozen at the coordinate mean.
#[derive(Debug, Clone, Copy)]
pub struct CircularGaussianFit2d {
    center: [f64; 2],
    radius: f64,
    noise_variance: f64,
}

impl CircularGaussianFit2d {
    /// Two center coordinates, one radius, and one isotropic noise variance.
    pub const NUM_FREE_PARAMETERS: usize = 4;

    /// Construct a circular Gaussian from validated model parameters.
    pub fn from_parameters(
        center: [f64; 2],
        radius: f64,
        noise_variance: f64,
    ) -> Result<Self, String> {
        if !center.iter().all(|value| value.is_finite()) {
            return Err("circular Gaussian center must be finite".to_string());
        }
        if !(radius.is_finite() && radius >= 0.0) {
            return Err("circular Gaussian radius must be finite and nonnegative".to_string());
        }
        if !(noise_variance.is_finite() && noise_variance > 0.0) {
            return Err("circular Gaussian noise variance must be finite and positive".to_string());
        }
        Ok(Self {
            center,
            radius,
            noise_variance,
        })
    }

    /// Fit selected rows of a finite two-column coordinate matrix.
    pub fn fit(coords: ArrayView2<'_, f64>, rows: &[usize]) -> Result<Self, String> {
        if coords.ncols() != 2 {
            return Err(format!(
                "circular Gaussian requires 2-D data, got {} columns",
                coords.ncols()
            ));
        }
        if rows.is_empty() {
            return Err("circular Gaussian requires a nonempty training set".to_string());
        }
        if rows.iter().any(|&row| row >= coords.nrows()) {
            return Err("circular Gaussian row index is out of bounds".to_string());
        }
        if rows
            .iter()
            .any(|&row| !coords[[row, 0]].is_finite() || !coords[[row, 1]].is_finite())
        {
            return Err("circular Gaussian requires finite training coordinates".to_string());
        }

        // Work in a dimensionless chart relative to one observed point. This
        // preserves the low-order bits of a small translated circle and makes
        // the stopping rule and variance floor scale equivariant.
        let anchor_row = rows[0];
        let anchor = [coords[[anchor_row, 0]], coords[[anchor_row, 1]]];
        let mut scale = 0.0_f64;
        for &row in rows {
            let dx = coords[[row, 0]] - anchor[0];
            let dy = coords[[row, 1]] - anchor[1];
            if !(dx.is_finite() && dy.is_finite()) {
                return Err("circular Gaussian coordinate range exceeds f64".to_string());
            }
            scale = scale.max(dx.hypot(dy));
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err("circular Gaussian requires nonzero spatial extent".to_string());
        }

        let mut points = Vec::with_capacity(rows.len());
        let mut mean = [0.0_f64; 2];
        for &row in rows {
            let point = [
                (coords[[row, 0]] - anchor[0]) / scale,
                (coords[[row, 1]] - anchor[1]) / scale,
            ];
            points.push(point);
            mean[0] += point[0];
            mean[1] += point[1];
        }
        let count = rows.len() as f64;
        mean[0] /= count;
        mean[1] /= count;

        // Moment initialization is exact at the population level. For
        // q = ||X-E X||^2,
        //   E[q] = R^2 + 2s,  Var(q) = 4s(R^2+s),
        // hence R^4 = E[q]^2-Var(q) and s=(E[q]-R^2)/2.
        let mut squared_radii = Vec::with_capacity(rows.len());
        let mut mean_squared_radius = 0.0_f64;
        for point in &points {
            let dx = point[0] - mean[0];
            let dy = point[1] - mean[1];
            let squared_radius = dx * dx + dy * dy;
            squared_radii.push(squared_radius);
            mean_squared_radius += squared_radius;
        }
        mean_squared_radius /= count;
        let mut squared_radius_variance = 0.0_f64;
        for squared_radius in squared_radii {
            squared_radius_variance += (squared_radius - mean_squared_radius).powi(2);
        }
        squared_radius_variance /= count;

        // A noiseless observed circle is an unbounded-likelihood boundary.
        // Keep the numerical optimizer in a scale-relative interior whose
        // width is roundoff, rather than imposing a floor in data units.
        let variance_floor = (64.0 * f64::EPSILON * mean_squared_radius).max(f64::MIN_POSITIVE);
        let radius_squared = (mean_squared_radius * mean_squared_radius - squared_radius_variance)
            .max(0.0)
            .sqrt();
        let mut radius = radius_squared.sqrt();
        let mut noise_variance = (0.5 * (mean_squared_radius - radius_squared)).max(variance_floor);
        let mut center = mean;

        // Exact EM for the latent circle angle. Given current parameters, the
        // conditional mean of U is A(kappa) * (x-c)/||x-c|| with
        // A=I1/I0 and kappa=R||x-c||/s. Solving the joint quadratic M-step for
        // center and radius avoids the biased `center = sample mean` plug-in.
        const MAX_EM_ITERATIONS: usize = 4096;
        const EM_TOLERANCE: f64 = 2.0e-12;
        let mut posterior_means = vec![[0.0_f64; 2]; points.len()];
        let mut converged = false;
        for _ in 0..MAX_EM_ITERATIONS {
            let mut posterior_mean = [0.0_f64; 2];
            for (point, latent_mean) in points.iter().zip(&mut posterior_means) {
                let dx = point[0] - center[0];
                let dy = point[1] - center[1];
                let observed_radius = dx.hypot(dy);
                if observed_radius == 0.0 || radius == 0.0 {
                    *latent_mean = [0.0, 0.0];
                } else {
                    let (_, bessel_ratio) =
                        circular_gaussian_bessel_terms(radius, observed_radius, noise_variance);
                    if !(bessel_ratio.is_finite() && (0.0..=1.0).contains(&bessel_ratio)) {
                        return Err("circular Gaussian Bessel ratio left [0, 1]".to_string());
                    }
                    let multiplier = bessel_ratio / observed_radius;
                    *latent_mean = [multiplier * dx, multiplier * dy];
                }
                posterior_mean[0] += latent_mean[0];
                posterior_mean[1] += latent_mean[1];
            }
            posterior_mean[0] /= count;
            posterior_mean[1] /= count;

            let denominator =
                1.0 - posterior_mean[0] * posterior_mean[0] - posterior_mean[1] * posterior_mean[1];
            if !(denominator.is_finite() && denominator > 0.0) {
                return Err("circular Gaussian EM radius update is singular".to_string());
            }
            let mut radius_numerator = 0.0_f64;
            for (point, latent_mean) in points.iter().zip(&posterior_means) {
                radius_numerator +=
                    latent_mean[0] * (point[0] - mean[0]) + latent_mean[1] * (point[1] - mean[1]);
            }
            let next_radius = (radius_numerator / (count * denominator)).max(0.0);
            let next_center = [
                mean[0] - next_radius * posterior_mean[0],
                mean[1] - next_radius * posterior_mean[1],
            ];

            // Evaluate E||X-c-RU||^2 in an explicitly nonnegative form to
            // avoid catastrophic cancellation on a very thin ring.
            let mut residual_sum = 0.0_f64;
            for (point, latent_mean) in points.iter().zip(&posterior_means) {
                let dx = point[0] - next_center[0];
                let dy = point[1] - next_center[1];
                let ex = dx - next_radius * latent_mean[0];
                let ey = dy - next_radius * latent_mean[1];
                let latent_norm_squared =
                    latent_mean[0] * latent_mean[0] + latent_mean[1] * latent_mean[1];
                residual_sum += ex * ex
                    + ey * ey
                    + next_radius * next_radius * (1.0 - latent_norm_squared).max(0.0);
            }
            let next_noise_variance = (residual_sum / (2.0 * count)).max(variance_floor);

            let parameter_change = (next_center[0] - center[0])
                .hypot(next_center[1] - center[1])
                .max((next_radius - radius).abs())
                .max(
                    (next_noise_variance - noise_variance).abs()
                        / (next_noise_variance + noise_variance),
                );
            center = next_center;
            radius = next_radius;
            noise_variance = next_noise_variance;
            if parameter_change <= EM_TOLERANCE {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err("circular Gaussian maximum-likelihood fit did not converge".to_string());
        }

        let fitted_noise_sd = scale * noise_variance.sqrt();
        Self::from_parameters(
            [anchor[0] + scale * center[0], anchor[1] + scale * center[1]],
            scale * radius,
            fitted_noise_sd * fitted_noise_sd,
        )
        .map_err(|error| format!("circular Gaussian fit produced invalid parameters: {error}"))
    }

    /// Fitted circle center.
    pub const fn center(self) -> [f64; 2] {
        self.center
    }

    /// Fitted latent-circle radius.
    pub const fn radius(self) -> f64 {
        self.radius
    }

    /// Fitted isotropic Cartesian noise variance per coordinate.
    pub const fn noise_variance(self) -> f64 {
        self.noise_variance
    }

    /// Proper Cartesian log density at `(x, y)`.
    pub fn log_density(self, x: f64, y: f64) -> f64 {
        let observed_radius = (x - self.center[0]).hypot(y - self.center[1]);
        let (log_i0_minus_kappa, _) =
            circular_gaussian_bessel_terms(self.radius, observed_radius, self.noise_variance);
        let standardized_radial_residual =
            (observed_radius - self.radius) / self.noise_variance.sqrt();
        // Algebraically this is -(r^2+R^2)/(2s)+log I0(kappa), but this
        // rearrangement preserves log I0(kappa) ~= kappa cancellation.
        -std::f64::consts::TAU.ln()
            - self.noise_variance.ln()
            - 0.5 * standardized_radial_residual.powi(2)
            + log_i0_minus_kappa
    }

    /// Sum the fitted log density over selected rows.
    pub fn log_likelihood(
        self,
        coords: ArrayView2<'_, f64>,
        rows: &[usize],
    ) -> Result<f64, String> {
        if coords.ncols() != 2 || rows.iter().any(|&row| row >= coords.nrows()) {
            return Err(
                "circular Gaussian likelihood received invalid coordinates or rows".to_string(),
            );
        }
        let mut log_densities = Vec::with_capacity(rows.len());
        for &row in rows {
            let value = self.log_density(coords[[row, 0]], coords[[row, 1]]);
            if !value.is_finite() {
                return Err("circular Gaussian likelihood is not finite".to_string());
            }
            log_densities.push(value);
        }
        let log_likelihood = pairwise_sum(&log_densities);
        if !log_likelihood.is_finite() {
            return Err("circular Gaussian likelihood sum is not finite".to_string());
        }
        Ok(log_likelihood)
    }

    /// Fit selected rows and return both the fit and its BIC/2 (lower is
    /// better). Keeping fitting and evidence evaluation in one operation makes
    /// it impossible to label a likelihood on unrelated data as fitted BIC.
    pub fn fit_with_bic(
        coords: ArrayView2<'_, f64>,
        rows: &[usize],
    ) -> Result<(Self, f64), String> {
        let fit = Self::fit(coords, rows)?;
        let log_likelihood = fit.log_likelihood(coords, rows)?;
        let bic =
            -log_likelihood + 0.5 * Self::NUM_FREE_PARAMETERS as f64 * (rows.len() as f64).ln();
        if !bic.is_finite() {
            return Err("circular Gaussian BIC is not finite".to_string());
        }
        Ok((fit, bic))
    }
}

/// Stable Bessel terms for `kappa = radius * observed_radius / variance`.
/// The ordinary finite branch retains the shared approximation exactly. If the
/// product itself overflows, only the leading asymptotic
/// `log I0(kappa)-kappa = -log(2 pi kappa)/2 + O(1/kappa)` is representable;
/// `I1/I0` rounds to one at that scale.
fn circular_gaussian_bessel_terms(
    radius: f64,
    observed_radius: f64,
    noise_variance: f64,
) -> (f64, f64) {
    if radius == 0.0 || observed_radius == 0.0 {
        return (0.0, 0.0);
    }
    let kappa = radius * observed_radius / noise_variance;
    if kappa.is_finite() {
        return bessel_i0_log_minus_abs_and_ratio(kappa);
    }
    let log_kappa = radius.ln() + observed_radius.ln() - noise_variance.ln();
    if log_kappa <= f64::MAX.ln() {
        // The left-to-right product overflowed before a large variance divided
        // it back into range. Reconstruct the representable ratio from its log.
        return bessel_i0_log_minus_abs_and_ratio(log_kappa.exp());
    }
    (-0.5 * (std::f64::consts::TAU.ln() + log_kappa), 1.0)
}
//
// A *union* candidate is a small FIXED composite of named component structures
// joined by a hard row-responsibility split. Unlike the discrete-mixture rung
// (which is one free k-component Gaussian density), a union pins each component
// to a specific generative STRUCTURE (a circle, a line, a point cluster) and
// asks whether the data is better explained as the disjoint sum of those
// structures than by any single pure rung.
//
// The hard responsibility groups only determine which rows fit each component.
// The resulting candidate is one normalized, unlabeled soft-mixture density
// `p(y) = Σ_c π_c p_c(y)`, with `π_c = n_c / n`. It is scored on every
// training row by that same mixture density used for held-out evaluation. Its
// BIC/2 complexity price is `½(Σ_c P_c + m - 1) log(n)`: component
// parameters plus the `m - 1` free mixing weights, all on the common sample
// scale. This makes the union directly comparable to the other normalized
// parametric candidates in the topology race.

/// The fixed ladder of structured-union composites. Deterministic and closed:
/// open-ended structure search stays owned by #976's move set; these three are
/// the only composites the topology race may select.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnionStructure {
    /// Two circles (two well-separated periodic loops).
    CircleCircle,
    /// One circle plus one isolated point cluster (a loop with an outlier blob).
    CirclePointCluster,
    /// One line (anisotropic cluster) plus one isolated point cluster.
    LineCluster,
}

/// The fixed structured-union ladder, in stable order.
pub const UNION_STRUCTURE_LADDER: &[UnionStructure] = &[
    UnionStructure::CircleCircle,
    UnionStructure::CirclePointCluster,
    UnionStructure::LineCluster,
];

/// The per-component generative structure a union pins each responsibility group
/// to. `Line` is a full-covariance Gaussian, while `PointCluster` is the nested
/// isotropic Gaussian with `d + 1` parameters. The covariance constraint makes
/// line+cluster a genuine structured alternative to a generic two-component
/// full-covariance mixture instead of a duplicate candidate. `Circle` is the
/// proper Cartesian density of a uniform latent circle convolved with isotropic
/// Gaussian noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnionComponentKind {
    Circle,
    Line,
    PointCluster,
}

impl UnionStructure {
    /// Stable display name, e.g. `"union_circle+circle"`.
    pub const fn as_str(self) -> &'static str {
        match self {
            UnionStructure::CircleCircle => "union_circle+circle",
            UnionStructure::CirclePointCluster => "union_circle+cluster",
            UnionStructure::LineCluster => "union_line+cluster",
        }
    }

    /// The fixed ordered component structures of this union.
    pub const fn components(self) -> &'static [UnionComponentKind] {
        match self {
            UnionStructure::CircleCircle => {
                &[UnionComponentKind::Circle, UnionComponentKind::Circle]
            }
            UnionStructure::CirclePointCluster => {
                &[UnionComponentKind::Circle, UnionComponentKind::PointCluster]
            }
            UnionStructure::LineCluster => {
                &[UnionComponentKind::Line, UnionComponentKind::PointCluster]
            }
        }
    }

    /// Number of components (= the responsibility-split order `m`).
    pub const fn num_components(self) -> usize {
        self.components().len()
    }
}

/// One fitted component of a union: its pinned structure, the rows used to fit
/// it after the hard responsibility split, its free-parameter count, and its
/// normalized soft-mixture weight. A component has no standalone BIC inside a
/// union: the likelihood is the indivisible `log Σ_c π_c p_c(y)` scored on
/// every row.
#[derive(Debug, Clone)]
pub struct UnionComponentFit {
    pub kind: UnionComponentKind,
    pub row_count: usize,
    pub num_parameters: usize,
    pub mixing_weight: f64,
}

/// A fitted structured-union candidate: the composite kind, the per-component
/// fits, its normalized soft-mixture training likelihood, the corresponding
/// BIC-form negative-log-evidence, and the complete free-parameter count.
#[derive(Debug, Clone)]
pub struct UnionStructureFit {
    pub structure: UnionStructure,
    pub components: Vec<UnionComponentFit>,
    /// `Σ_i log(Σ_c π_c p_c(y_i))` over all training rows.
    pub log_likelihood: f64,
    /// `-log_likelihood + ½ total_parameters log(n)` (lower wins).
    pub bic: f64,
    /// `Σ_c P_c + (m - 1)`, including the free mixing weights.
    pub total_parameters: usize,
}

/// Hard responsibility split of `0..n` into `m` groups by argmax of the
/// deterministic `m`-component Gaussian-mixture responsibilities. Reuses the
/// mixture rung's seeding + EM so the split is a pure function of the data and
/// `m` (no clock). Returns one row-index vector per component.
pub fn union_responsibility_split(
    data: ArrayView2<'_, f64>,
    m: usize,
    config: GaussianMixtureConfig,
) -> Result<Vec<Vec<usize>>, String> {
    let n = data.nrows();
    if m == 0 {
        return Err("union split requires at least one component".to_string());
    }
    if m > n {
        return Err(format!(
            "union split requested {m} groups but data has {n} rows"
        ));
    }
    if m == 1 {
        return Ok(vec![(0..n).collect()]);
    }
    let fit = fit_gaussian_mixture(data, m, config).map_err(|error| error.to_string())?;
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); m];
    // Hard assignment by argmax per-component log responsibility.
    let mut comp = Vec::with_capacity(m);
    for j in 0..m {
        comp.push(GaussianComponentEval::factor(
            fit.means.row(j),
            &fit.covariances[j],
        )?);
    }
    let log_w = fit
        .weights
        .iter()
        .enumerate()
        .map(|(component, &weight)| {
            if weight.is_finite() && weight > 0.0 {
                Ok(weight.ln())
            } else {
                Err(format!(
                    "union split received invalid fitted weight {weight} for component {component}"
                ))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    for i in 0..n {
        let row = data.row(i);
        let mut best_j = 0usize;
        let mut best_lt = f64::NEG_INFINITY;
        for j in 0..m {
            let lt = log_w[j] + comp[j].log_density(row);
            if lt > best_lt {
                best_lt = lt;
                best_j = j;
            }
        }
        if !best_lt.is_finite() {
            return Err(format!(
                "union split produced no finite component score at row {i}"
            ));
        }
        groups[best_j].push(i);
    }
    Ok(groups)
}

/// Fit one structured-union candidate. The hard split identifies component
/// training groups; heterogeneous component roles are assigned by evaluating
/// every unique role permutation. Each assignment is then scored as one
/// normalized soft mixture on every training row.
///
/// Returns an error if no role assignment identifies every component. The
/// fixed-ladder caller propagates that failure for the whole declared family;
/// it never ranks a survivor subset.
pub fn fit_union_structure(
    data: ArrayView2<'_, f64>,
    structure: UnionStructure,
    config: GaussianMixtureConfig,
) -> Result<UnionStructureFit, String> {
    let fitted = fit_union_density(data, structure, config)?;
    Ok(UnionStructureFit {
        structure,
        components: fitted
            .components
            .iter()
            .map(UnionComponentDensity::summary)
            .collect(),
        log_likelihood: fitted.log_likelihood,
        bic: fitted.bic,
        total_parameters: fitted.total_parameters,
    })
}

/// Fit the whole fixed union ladder and rank in-class by normalized soft-mixture
/// BIC (lower wins). The declared ladder is one selection family: if any
/// eligible structure fails, the whole comparison fails with every per-structure
/// error rather than silently selecting from an easier survivor set.
pub fn fit_union_ladder(
    data: ArrayView2<'_, f64>,
    config: GaussianMixtureConfig,
) -> Result<Vec<UnionStructureFit>, String> {
    let mut fits = Vec::new();
    let mut errors = Vec::new();
    for &structure in UNION_STRUCTURE_LADDER {
        match fit_union_structure(data, structure, config) {
            Ok(fit) => fits.push(fit),
            Err(e) => errors.push(format!("{}: {e}", structure.as_str())),
        }
    }
    if !errors.is_empty() {
        return Err(format!(
            "union ladder comparison failed; every declared structure must fit ({})",
            errors.join("; ")
        ));
    }
    if fits.is_empty() {
        return Err("union ladder is empty".to_string());
    }
    let ranked = rank_priority_candidates(
        fits.into_iter()
            .enumerate()
            .map(|(idx, row)| {
                let score = row.bic;
                let tie = row.total_parameters; // cheaper composite wins ties
                PriorityCandidate::new(row, idx, score, tie)
            })
            .collect(),
    )
    .into_iter()
    .map(|row| row.item)
    .collect::<Vec<_>>();
    Ok(ranked)
}

fn gather_union_rows(data: ArrayView2<'_, f64>, idx: &[usize]) -> Array2<f64> {
    let d = data.ncols();
    let mut out = Array2::<f64>::zeros((idx.len(), d));
    for (r, &i) in idx.iter().enumerate() {
        for c in 0..d {
            out[[r, c]] = data[[i, c]];
        }
    }
    out
}

/// Apply the structured-union admission policy to one circle group and return
/// its complete deterministic row set.
/// The `n > P` rule belongs to this composite ladder; it is not an intrinsic
/// restriction of circular-Gaussian likelihood fitting or BIC itself.
fn union_circle_rows(group: ArrayView2<'_, f64>) -> Result<Vec<usize>, String> {
    let minimum_rows = CircularGaussianFit2d::NUM_FREE_PARAMETERS + 1;
    if group.nrows() < minimum_rows {
        return Err(format!(
            "union circle component needs at least {minimum_rows} rows, got {}",
            group.nrows()
        ));
    }
    Ok((0..group.nrows()).collect())
}

/// The shared fitted-density representation used by both in-sample BIC and
/// held-out predictive scoring. `Line` and `PointCluster` deliberately share
/// the Gaussian evaluator after fitting, but differ in covariance constraints
/// and parameter count.
#[derive(Debug, Clone)]
enum UnionDensityModel {
    Gaussian(GaussianComponentEval),
    Circle(CircularGaussianFit2d),
}

#[derive(Debug, Clone)]
struct UnionComponentDensity {
    kind: UnionComponentKind,
    row_count: usize,
    num_parameters: usize,
    mixing_weight: f64,
    log_weight: f64,
    model: UnionDensityModel,
}

impl UnionComponentDensity {
    fn summary(&self) -> UnionComponentFit {
        UnionComponentFit {
            kind: self.kind,
            row_count: self.row_count,
            num_parameters: self.num_parameters,
            mixing_weight: self.mixing_weight,
        }
    }

    fn dimension(&self) -> usize {
        match &self.model {
            UnionDensityModel::Gaussian(eval) => eval.d,
            UnionDensityModel::Circle(_) => 2,
        }
    }

    /// `log π_c + log p_c(y)` for one eval row.
    fn weighted_log_density(&self, y: ArrayView1<'_, f64>) -> f64 {
        let component_log_density = match &self.model {
            UnionDensityModel::Gaussian(eval) => eval.log_density(y),
            UnionDensityModel::Circle(fit) => fit.log_density(y[0], y[1]),
        };
        self.log_weight + component_log_density
    }
}

#[derive(Debug, Clone)]
struct FittedUnionDensity {
    components: Vec<UnionComponentDensity>,
    log_likelihood: f64,
    bic: f64,
    total_parameters: usize,
}

fn fit_union_density(
    train: ArrayView2<'_, f64>,
    structure: UnionStructure,
    config: GaussianMixtureConfig,
) -> Result<FittedUnionDensity, String> {
    let groups = union_responsibility_split(train, structure.num_components(), config)?;
    fit_union_density_from_groups(train, structure, &groups, config)
}

/// Fit and score a union for an already-established hard partition. This seam
/// makes role assignment explicitly independent of the arbitrary component
/// labels emitted by the responsibility split.
fn fit_union_density_from_groups(
    train: ArrayView2<'_, f64>,
    structure: UnionStructure,
    groups: &[Vec<usize>],
    config: GaussianMixtureConfig,
) -> Result<FittedUnionDensity, String> {
    validate_union_partition(train.nrows(), structure.num_components(), groups)?;
    let assignments = unique_union_role_assignments(structure.components());
    let mut best: Option<FittedUnionDensity> = None;
    let mut errors = Vec::new();

    for roles in assignments {
        let candidate = (|| {
            let mut components = Vec::with_capacity(groups.len());
            let n_train = train.nrows() as f64;
            for (&kind, rows) in roles.iter().zip(groups) {
                let group = gather_union_rows(train, rows);
                let mixing_weight = rows.len() as f64 / n_train;
                components.push(fit_union_component_density(
                    group.view(),
                    kind,
                    mixing_weight,
                    config,
                )?);
            }

            let component_parameters = components.iter().try_fold(0usize, |sum, component| {
                sum.checked_add(component.num_parameters)
                    .ok_or_else(|| "union component parameter count overflowed usize".to_string())
            })?;
            let mixing_parameters = components.len() - 1;
            let total_parameters = component_parameters
                .checked_add(mixing_parameters)
                .ok_or_else(|| "union total parameter count overflowed usize".to_string())?;
            let per_point = score_union_components(&components, train)?;
            let log_likelihood = pairwise_sum(
                per_point
                    .as_slice()
                    .expect("owned union score vector must be contiguous"),
            );
            if !log_likelihood.is_finite() {
                return Err("union training log likelihood is non-finite".to_string());
            }
            let bic = -log_likelihood + 0.5 * total_parameters as f64 * (train.nrows() as f64).ln();
            if !bic.is_finite() {
                return Err("union normalized soft-mixture BIC is non-finite".to_string());
            }
            Ok(FittedUnionDensity {
                components,
                log_likelihood,
                bic,
                total_parameters,
            })
        })();

        match candidate {
            Ok(candidate) => {
                let replace = match &best {
                    Some(current) => candidate.bic.total_cmp(&current.bic).is_lt(),
                    None => true,
                };
                // Unique assignments are generated in canonical order. Keeping
                // the earlier assignment on an exact score tie is deterministic.
                if replace {
                    best = Some(candidate);
                }
            }
            Err(error) => errors.push(format!("{roles:?}: {error}")),
        }
    }

    best.ok_or_else(|| {
        format!(
            "union {} has no finite role assignment ({})",
            structure.as_str(),
            errors.join("; ")
        )
    })
}

fn validate_union_partition(
    n_rows: usize,
    expected_groups: usize,
    groups: &[Vec<usize>],
) -> Result<(), String> {
    if n_rows == 0 {
        return Err("union fitting requires at least one training row".to_string());
    }
    if groups.len() != expected_groups {
        return Err(format!(
            "union partition has {} groups, expected {expected_groups}",
            groups.len()
        ));
    }
    let mut seen = vec![false; n_rows];
    for (group_index, rows) in groups.iter().enumerate() {
        if rows.is_empty() {
            return Err(format!("union partition group {group_index} is empty"));
        }
        for &row in rows {
            if row >= n_rows {
                return Err(format!(
                    "union partition group {group_index} contains out-of-range row {row} for {n_rows} rows"
                ));
            }
            if std::mem::replace(&mut seen[row], true) {
                return Err(format!("union partition contains duplicate row {row}"));
            }
        }
    }
    if let Some(missing) = seen.iter().position(|included| !included) {
        return Err(format!("union partition omits row {missing}"));
    }
    Ok(())
}

fn unique_union_role_assignments(roles: &[UnionComponentKind]) -> Vec<Vec<UnionComponentKind>> {
    fn visit(
        roles: &[UnionComponentKind],
        used: &mut [bool],
        assignment: &mut Vec<UnionComponentKind>,
        out: &mut Vec<Vec<UnionComponentKind>>,
    ) {
        if assignment.len() == roles.len() {
            out.push(assignment.clone());
            return;
        }
        let mut used_at_depth = Vec::new();
        for (index, &role) in roles.iter().enumerate() {
            if used[index] || used_at_depth.contains(&role) {
                continue;
            }
            used_at_depth.push(role);
            used[index] = true;
            assignment.push(role);
            visit(roles, used, assignment, out);
            assignment.pop();
            used[index] = false;
        }
    }

    let mut out = Vec::new();
    visit(
        roles,
        &mut vec![false; roles.len()],
        &mut Vec::with_capacity(roles.len()),
        &mut out,
    );
    out
}

fn fit_union_component_density(
    group: ArrayView2<'_, f64>,
    kind: UnionComponentKind,
    mixing_weight: f64,
    config: GaussianMixtureConfig,
) -> Result<UnionComponentDensity, String> {
    if !(mixing_weight.is_finite() && mixing_weight > 0.0 && mixing_weight <= 1.0) {
        return Err(format!(
            "union component mixing weight must be finite and in (0, 1], got {mixing_weight}"
        ));
    }
    let row_count = group.nrows();
    let (model, num_parameters) = match kind {
        UnionComponentKind::Line => {
            if group.nrows() < group.ncols() + 1 {
                return Err(format!(
                    "union line component needs >= {} rows, got {}",
                    group.ncols() + 1,
                    group.nrows()
                ));
            }
            let fit = fit_gaussian_mixture(group, 1, config).map_err(|error| error.to_string())?;
            let num_parameters = fit.num_free_parameters();
            let eval = GaussianComponentEval::factor(fit.means.row(0), &fit.covariances[0])?;
            (UnionDensityModel::Gaussian(eval), num_parameters)
        }
        UnionComponentKind::PointCluster => {
            if group.nrows() < group.ncols() + 1 {
                return Err(format!(
                    "union isotropic point component needs >= {} rows, got {}",
                    group.ncols() + 1,
                    group.nrows()
                ));
            }
            let eval = fit_isotropic_gaussian_component(group, config.covariance_floor)?;
            (
                UnionDensityModel::Gaussian(eval),
                group
                    .ncols()
                    .checked_add(1)
                    .ok_or_else(|| "union point parameter count overflowed usize".to_string())?,
            )
        }
        UnionComponentKind::Circle => {
            let rows = union_circle_rows(group)?;
            let fit = CircularGaussianFit2d::fit(group, &rows)?;
            (
                UnionDensityModel::Circle(fit),
                CircularGaussianFit2d::NUM_FREE_PARAMETERS,
            )
        }
    };
    Ok(UnionComponentDensity {
        kind,
        row_count,
        num_parameters,
        mixing_weight,
        log_weight: mixing_weight.ln(),
        model,
    })
}

#[derive(Debug, Clone, Copy)]
struct StableScalarMeanChart {
    origin: f64,
    scale: f64,
    normalized_offset: f64,
}

impl StableScalarMeanChart {
    #[inline]
    fn centered(self, value: f64) -> Result<f64, String> {
        let relative = value - self.origin;
        let centered = (-self.normalized_offset).mul_add(self.scale, relative);
        if centered.is_finite() {
            Ok(centered)
        } else {
            Err("union isotropic point residual is not representable".to_string())
        }
    }
}

/// Range-safe and translation-accurate scalar mean chart. The normalized mean
/// offset is retained separately from the rounded absolute mean so residuals
/// use `(x-origin)-offset` rather than losing a fractional offset at a large
/// common translation. FMA also lets a subnormal offset affect the correctly
/// rounded absolute mean without first rounding that offset to zero.
fn stable_scalar_mean_chart(values: ArrayView1<'_, f64>) -> Result<StableScalarMeanChart, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("stable scalar mean requires finite nonempty values".to_string());
    }
    let anchor = values[0];
    let anchor_chart_is_representable = values.iter().all(|&value| (value - anchor).is_finite());
    let origin = if anchor_chart_is_representable {
        anchor
    } else {
        0.0
    };
    let scale = values
        .iter()
        .map(|&value| (value - origin).abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Ok(StableScalarMeanChart {
            origin,
            scale: 0.0,
            normalized_offset: 0.0,
        });
    }
    let normalized = values
        .iter()
        .map(|&value| (value - origin) / scale)
        .collect::<Vec<_>>();
    let normalized_offset = pairwise_sum(&normalized) / values.len() as f64;
    let mean = normalized_offset.mul_add(scale, origin);
    if !(normalized_offset.is_finite() && mean.is_finite()) {
        return Err("union isotropic point mean is not representable".to_string());
    }
    Ok(StableScalarMeanChart {
        origin,
        scale,
        normalized_offset,
    })
}

/// Maximum-likelihood isotropic Gaussian fit in stable per-column mean charts.
fn fit_isotropic_gaussian_component(
    group: ArrayView2<'_, f64>,
    covariance_floor: f64,
) -> Result<GaussianComponentEval, String> {
    let n = group.nrows();
    let d = group.ncols();
    if n == 0 || d == 0 {
        return Err("union isotropic point component requires a non-empty matrix".to_string());
    }
    if !(covariance_floor.is_finite() && covariance_floor > 0.0) {
        return Err(format!(
            "union isotropic covariance floor must be finite and positive, got {covariance_floor}"
        ));
    }

    for row in group.rows() {
        for axis in 0..d {
            let value = row[axis];
            if !value.is_finite() {
                return Err(format!(
                    "union isotropic point data contains non-finite coordinate {value}"
                ));
            }
        }
    }

    let mut charts = Vec::with_capacity(d);
    for axis in 0..d {
        let chart = stable_scalar_mean_chart(group.column(axis))?;
        charts.push(chart);
    }

    let scalar_count = n
        .checked_mul(d)
        .ok_or_else(|| "union isotropic residual count overflowed usize".to_string())?;
    let mut residuals = Vec::with_capacity(scalar_count);
    let mut residual_scale = 0.0_f64;
    for row in group.rows() {
        for axis in 0..d {
            let residual = charts[axis].centered(row[axis])?;
            residual_scale = residual_scale.max(residual.abs());
            residuals.push(residual);
        }
    }
    let variance = if residual_scale == 0.0 {
        covariance_floor
    } else {
        for residual in &mut residuals {
            *residual = (*residual / residual_scale).powi(2);
        }
        let normalized_mean_square = pairwise_sum(&residuals) / scalar_count as f64;
        let rms = residual_scale * normalized_mean_square.sqrt();
        let unconstrained = rms * rms;
        if !unconstrained.is_finite() {
            return Err("union isotropic point variance is non-finite".to_string());
        }
        unconstrained.max(covariance_floor)
    };
    GaussianComponentEval::isotropic(&charts, variance)
}

fn score_union_components(
    components: &[UnionComponentDensity],
    eval: ArrayView2<'_, f64>,
) -> Result<Array1<f64>, String> {
    if components.is_empty() {
        return Err("union density requires at least one component".to_string());
    }
    if eval.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err("union eval coordinates must be finite".to_string());
    }
    for component in components {
        if component.dimension() != eval.ncols() {
            return Err(format!(
                "union component {:?} has dimension {}, eval has {} columns",
                component.kind,
                component.dimension(),
                eval.ncols()
            ));
        }
    }
    let mut out = Array1::<f64>::zeros(eval.nrows());
    let mut terms = vec![f64::NEG_INFINITY; components.len()];
    for i in 0..eval.nrows() {
        let row = eval.row(i);
        let mut max_term = f64::NEG_INFINITY;
        for (component_index, component) in components.iter().enumerate() {
            let term = component.weighted_log_density(row);
            terms[component_index] = term;
            if term > max_term {
                max_term = term;
            }
        }
        let value = log_sum_exp(&terms, max_term);
        if !value.is_finite() {
            return Err(format!(
                "union density produced non-finite log density at eval row {i}"
            ));
        }
        out[i] = value;
    }
    Ok(out)
}

/// Per-point held-out log predictive density of a structured-union candidate:
/// fit the component densities on `train` and score each row of `eval` as the
/// soft mixture `log Σ_c π_c p_c(y)`. This is the cross-class stacking column
/// source for a union (the analogue of [`GaussianMixtureFit::per_point_log_density`]).
pub fn union_per_point_log_density(
    train: ArrayView2<'_, f64>,
    eval: ArrayView2<'_, f64>,
    structure: UnionStructure,
    config: GaussianMixtureConfig,
) -> Result<Array1<f64>, String> {
    if train.ncols() != eval.ncols() {
        return Err(format!(
            "union held-out density: train has {} columns, eval has {}",
            train.ncols(),
            eval.ncols()
        ));
    }
    let fitted = fit_union_density(train, structure, config)?;
    score_union_components(&fitted.components, eval)
}

/// One fitted model in a REML/LAML evidence comparison.
#[derive(Clone, Debug)]
pub struct RemlCandidate {
    pub index: usize,
    pub name: String,
    /// Minimised REML/LAML cost. Lower is better. This is the model's reported
    /// evidence headline (`Model.evidence`), kept verbatim in the score table.
    pub score: f64,
    pub edf: Option<f64>,
    /// Log-likelihood at the converged mode, on the engine's
    /// constants-omitted scale (same as `gam_inference::model_comparison`).
    /// Present when the fit carries it; `None` for legacy payloads.
    pub log_lik: Option<f64>,
    /// Response-family tag (e.g. "gaussian", "gamma", "binomial"). Carried so
    /// `compare_reml_fits` can REFUSE to rank fits whose REML/LAML scores are on
    /// incomparable base measures (a cross-family comparison is meaningless;
    /// #1384). `None` for legacy payloads that did not record it — those are not
    /// guarded (back-compatible), but every current FFI candidate carries it.
    pub family: Option<String>,
    /// Number of observations the fit was trained on. Carried so
    /// `compare_reml_fits` can REFUSE to rank fits made on a different number of
    /// observations (hence different data): `−2·loglik` and the REML/LAML
    /// evidence grow with `n`, so a score difference between two fits with
    /// different `n` is not a Bayes factor — the same incomparability the family
    /// guard already rejects. `None` for payloads that do not record it (legacy /
    /// O(n) scan smoothers), which the guard treats as unconstrained.
    pub n_obs: Option<usize>,
}

impl RemlCandidate {
    /// Cost used to RANK candidates and pick the winner.
    ///
    /// The REML/LAML marginal-likelihood evidence headline (`score`) does NOT
    /// reliably Occam-penalise an added pure-noise smooth: on `y ~ s(x)` vs
    /// `y ~ s(x) + s(z)` with `z ⟂ y`, the augmented model's evidence is
    /// *lower* (apparently better) by a few nats on essentially every dataset,
    /// because the Gaussian REML Occam pair `½(log|H| − log|S|₊)` collapses
    /// toward zero for a finite-`λ̂` null term while that term still spends a
    /// few effective degrees of freedom fitting noise (issue #1362).
    ///
    /// The conditional AIC `−2ℓ + 2·edf` prices exactly those spent degrees of
    /// freedom and discriminates correctly: it penalises the noise smooth
    /// (Δ ≈ +15 nats) yet rewards a genuinely relevant smooth (Δ ≈ −650),
    /// preserving power. We therefore rank on the conditional AIC whenever both
    /// the log-likelihood and the effective degrees of freedom are available,
    /// and fall back to the raw evidence headline otherwise. The reported
    /// `score_table` still carries the unaltered evidence (`reml_score`), so
    /// `Model.evidence` / `bayes_factor_vs` stay consistent with the table.
    pub fn ranking_score(&self) -> f64 {
        match (self.log_lik, self.edf) {
            (Some(log_lik), Some(edf)) if log_lik.is_finite() && edf.is_finite() => {
                -2.0 * log_lik + 2.0 * edf
            }
            _ => self.score,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RemlComparison {
    pub ranking: Vec<RankedRow>,
    pub winner: String,
    pub evidence_summary: String,
    pub score_table: Vec<ScoreRow>,
}

#[derive(Clone, Debug)]
pub struct RankedRow {
    pub name: String,
    pub score: f64,
    /// Cost gap from the winning model on the SAME scale used to order the
    /// ranking (`ranking_score`, the Occam-penalised conditional AIC where
    /// available, issue #1362). The winner is `argmin ranking_score`, so this
    /// is `>= 0` for every row by construction — it never contradicts the
    /// declared winner (issue #1465). `score` still carries the raw REML/LAML
    /// evidence so it stays consistent with `Model.evidence`.
    pub delta: f64,
    /// Akaike evidence ratio of the winner over this row on the ranking scale.
    /// `delta` is a conditional-AIC gap (a −2·log / deviance-scale quantity), so
    /// the evidence ratio is `exp(½·delta) >= 1` (Burnham & Anderson), NOT
    /// `exp(delta)` — the latter squares the intended ratio (issues #1465, #2124).
    pub bayes_factor: f64,
    pub edf: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct ScoreRow {
    pub name: String,
    pub reml_score: f64,
    pub delta_reml: f64,
    pub bayes_factor_best_over_model: f64,
    pub effective_dof: Option<f64>,
}

/// Log Bayes factor of model `a` over model `b` from minimised REML/LAML costs.
#[inline]
pub fn log_bayes_factor(reml_score_a: f64, reml_score_b: f64) -> f64 {
    reml_score_b - reml_score_a
}

/// Compare fitted models by the single evidence ordering contract used by
/// topology ranking and seed screening: lower finite cost wins, with stable
/// original-order tie handling.
pub fn compare_reml_fits(mut candidates: Vec<RemlCandidate>) -> Result<RemlComparison, String> {
    if candidates.is_empty() {
        return Err("compare_models requires at least one fit".to_string());
    }
    // Fail-loud comparability guard (#1384): REML/LAML evidence scores are only
    // comparable across fits of the SAME response family — a Gaussian score and
    // a Gamma score live on different log-density base measures, so their
    // difference is not a Bayes factor. Ranking them anyway returns a confident
    // but meaningless winner. Refuse when two candidates carry DIFFERENT family
    // tags. Candidates with no family tag (`None`, legacy payloads) are not
    // constrained, so this never spuriously rejects an older saved model.
    {
        let mut seen_family: Option<&str> = None;
        for cand in &candidates {
            if let Some(fam) = cand.family.as_deref() {
                match seen_family {
                    None => seen_family = Some(fam),
                    Some(prev) if prev != fam => {
                        return Err(format!(
                            "compare_models: cannot compare fits of different response families                              ('{prev}' vs '{fam}'); their REML/LAML evidence scores are on                              incomparable base measures. Compare models fit to the same response                              under the same family."
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
    }
    // Fail-loud comparability guard (#1384 sibling): AIC / REML-LAML evidence are
    // only comparable across fits of the SAME response on the SAME observations.
    // `−2·loglik` (and the marginal-likelihood headline) grow with the number of
    // observations `n`, so two fits with different `n` live on incomparable
    // scales and their score gap is not a Bayes factor — comparing an n=500 and
    // an n=100 fit of the same DGP otherwise declares the n=100 model the winner
    // purely because fewer points give a less-negative total log-likelihood.
    // Refuse when two candidates carry DIFFERENT observation counts. Candidates
    // with no count (`None`, legacy / O(n) scan payloads) are unconstrained, so
    // this never spuriously rejects a fit that simply did not record `n`.
    {
        let mut seen_n: Option<usize> = None;
        for cand in &candidates {
            if let Some(n) = cand.n_obs {
                match seen_n {
                    None => seen_n = Some(n),
                    Some(prev) if prev != n => {
                        return Err(format!(
                            "compare_models: cannot compare fits made on a different number of \
                             observations (n={prev} vs n={n}); AIC / REML-LAML evidence scales \
                             with the sample size, so their score difference is not a Bayes \
                             factor. Compare models fit to the same response on the same data."
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
    }
    candidates = rank_priority_candidates(
        candidates
            .into_iter()
            .enumerate()
            .map(|(idx, row)| {
                // Rank/winner on the Occam-penalised conditional AIC where it is
                // available (issue #1362); falls back to the raw evidence score.
                let ranking = row.ranking_score();
                PriorityCandidate::new(row, idx, ranking, 0)
            })
            .collect(),
    )
    .into_iter()
    .map(|row| row.item)
    .collect();

    let winner = candidates[0].name.clone();
    // The ranking `delta` / `bayes_factor` must be measured on the SAME scale
    // that orders the table — the `ranking_score` (Occam-penalised conditional
    // AIC where available, issue #1362). `candidates[0]` is the winner =
    // `argmin ranking_score`, so its ranking score IS the minimum; every row's
    // ranking-scale gap is then `>= 0` and its Bayes factor `>= 1`, never
    // contradicting the declared winner (issue #1465). Computing these against
    // the AIC winner's *raw REML* — which is not the minimum raw REML once AIC
    // and REML disagree — produced negative deltas and Bayes factors < 1 for
    // non-winner rows.
    let best_ranking_score = candidates[0].ranking_score();
    // The raw-REML `score_table` stays on the raw evidence scale (consistent
    // with `Model.evidence` / `bayes_factor_vs`), but is referenced to the
    // genuine minimum raw REML so its best-over-model Bayes factors are also
    // coherent (`>= 1`), rather than to whichever row happens to sit at index 0.
    let best_raw_score = candidates
        .iter()
        .map(|c| c.score)
        .fold(f64::INFINITY, f64::min);
    let mut ranking = Vec::with_capacity(candidates.len());
    let mut score_table = Vec::with_capacity(candidates.len());
    for row in &candidates {
        let delta = log_bayes_factor(best_ranking_score, row.ranking_score());
        // `ranking_score` is the conditional AIC (`−2·loglik + 2·edf`), a −2·log /
        // deviance-scale cost, so `delta` is a full ΔAIC gap. The Akaike evidence
        // ratio for an AIC gap Δ is `exp(−½Δ)` (Burnham & Anderson evidence ratio),
        // hence the winner-over-row Bayes factor is `exp(½·delta)`. Reporting
        // `delta.exp()` squared the intended ratio (issue #2124). `delta` itself is
        // left on the AIC scale on purpose — only its exp() conversion is halved.
        let bayes_factor = (0.5 * delta).exp();
        let delta_reml = log_bayes_factor(best_raw_score, row.score);
        ranking.push(RankedRow {
            name: row.name.clone(),
            score: row.score,
            delta,
            bayes_factor,
            edf: row.edf,
        });
        score_table.push(ScoreRow {
            name: row.name.clone(),
            reml_score: row.score,
            delta_reml,
            bayes_factor_best_over_model: delta_reml.exp(),
            effective_dof: row.edf,
        });
    }
    // The winner is decided by `ranking_score` (the Occam-penalised conditional
    // AIC where available, issue #1362), which can disagree in sign with the raw
    // evidence Bayes factor for a noise-augmented model. Summarise the actual
    // decision margin so the headline never contradicts the chosen winner.
    let evidence_summary = if let Some(runner_up) = candidates.get(1) {
        let margin = runner_up.ranking_score() - candidates[0].ranking_score();
        // `margin` is a conditional-AIC gap (−2·log scale), so the Akaike evidence
        // ratio is `exp(−½·margin)`; `format_bayes_factor` formats `exp()` of its
        // argument, so pass the halved margin to headline `exp(½·margin)` rather
        // than the squared `exp(margin)` (issue #2124).
        format!(
            "{} wins by Bayes factor {} over {}",
            winner,
            format_bayes_factor(0.5 * margin),
            runner_up.name
        )
    } else {
        format!("{winner} (single fit; no comparison)")
    };
    Ok(RemlComparison {
        ranking,
        winner,
        evidence_summary,
        score_table,
    })
}

pub fn format_bayes_factor(log_bf: f64) -> String {
    if !log_bf.is_finite() {
        return "inf".to_string();
    }
    if log_bf.abs() >= std::f64::consts::LN_10 * 3.0 {
        return format!("1e{:+.1}", log_bf / std::f64::consts::LN_10);
    }
    format_three_significant(log_bf.exp())
}

pub fn format_three_significant(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    if !value.is_finite() {
        return format!("{value}");
    }
    let exponent = value.abs().log10().floor() as i32;
    if exponent >= 3 {
        return format!("{value:.2e}");
    }
    let decimals = (2 - exponent).max(0) as usize;
    let scale = 10f64.powi(decimals as i32);
    let rounded = (value * scale).abs().round() / scale * value.signum();
    format!("{rounded:.decimals$}")
}

impl Default for TopologySelectOptions {
    fn default() -> Self {
        Self {
            tie_tolerance: 1e-3,
            score_scale: TopologyScoreScale::PerObservation,
        }
    }
}

// ---------------------------------------------------------------------------
// Laplace evidence
// ---------------------------------------------------------------------------

/// Single canonical Laplace evidence at the inner-loop fixed point.
///
/// Returns negative log evidence:
///
/// ```text
/// V(ρ, T) = F(β*, u*; ρ, T)
///         + 0.5 log|H|
///         - 0.5 log|S_pen(ρ)|+
///         - 0.5 (dim(H) - rank(S_pen)) log(2π).
/// ```
///
/// The last term is the rank-aware Tierney-Kadane normalizer:
/// `log p(y|T) ≈ -V`, with `0.5 log|2πH⁻¹| - 0.5 log|2πS⁻¹|`.
///
/// The `H` log-determinant is computed from the arrow factorization
///
/// ```text
/// log|H| = Σ_i log|H_uu_i| + log|A|
/// ```
///
/// (proposal §3.4 / §7) using the **undamped** per-row Cholesky factors
/// `cache.htt_factors_undamped` and the **undamped** Schur factor.
///
/// `penalty_log_det` is `log|S_pen(ρ)|+` — the prior penalty
/// pseudo-logdet from `crate::reml::penalty_logdet` (proposal
/// §3.6). It must NOT be confused with the arrow Schur log-det, which
/// this function recomputes internally from `logdet_source`.
///
/// `residual_objective` is `F(β*, u*; ρ, T)` at the inner optimum. The
/// envelope theorem (proposal §3.2) makes this the only `F`-related
/// contribution.
///
/// `effective_dim` is `dim(H)` after constraints/projections and
/// `penalty_rank` is `rank(S_pen)`. Their difference is the unpenalized
/// nullspace dimension that remains in the Laplace integral.
///
/// # Errors
///
/// Returns `f64::NAN` if the exact factor path is incoherent and no HVP
/// fallback is supplied, or if the supplied dimensions are non-finite.
pub fn laplace_evidence(
    logdet_source: EvidenceLogDetSource<'_>,
    penalty_log_det: f64,
    residual_objective: f64,
    effective_dim: f64,
    penalty_rank: f64,
) -> f64 {
    if !(effective_dim.is_finite() && penalty_rank.is_finite()) {
        return f64::NAN;
    }
    let log_det_h = match evidence_hessian_log_det(logdet_source) {
        Ok(v) => v,
        Err(_) => return f64::NAN,
    };
    let null_dim = effective_dim - penalty_rank;
    if !null_dim.is_finite() || null_dim < -1e-9 {
        return f64::NAN;
    }
    residual_objective + 0.5 * log_det_h
        - 0.5 * penalty_log_det
        - 0.5 * null_dim.max(0.0) * (2.0 * std::f64::consts::PI).ln()
}

/// Compute the Hessian logdet from exact arrow factors or an HVP fallback.
pub fn evidence_hessian_log_det(source: EvidenceLogDetSource<'_>) -> Result<f64, String> {
    match source {
        EvidenceLogDetSource::FactoredArrow {
            cache,
            fallback_hvp,
        } => match arrow_log_det_from_cache(cache) {
            Some(v) => Ok(v),
            None => match fallback_hvp {
                Some(hvp) => hessian_log_det_from_hvp(hvp),
                None => {
                    Err("evidence Hessian logdet requires exact factors or HVP fallback".into())
                }
            },
        },
        EvidenceLogDetSource::Hvp(hvp) => hessian_log_det_from_hvp(hvp),
    }
}

/// Log determinant of an SPD operator supplied by HVP callback.
///
/// The dispatch boundary intentionally matches
/// `ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD` in `terms::analytic_penalties`:
/// small operators are materialized and diagonalized exactly; larger ones use
/// Rademacher stochastic Lanczos quadrature.
pub fn hessian_log_det_from_hvp(hvp: EvidenceHvpLogDet<'_>) -> Result<f64, String> {
    if hvp.dim == 0 {
        return Ok(0.0);
    }
    if hvp.dim <= ANALYTIC_LOGDET_DENSE_DIM_THRESHOLD {
        let mut dense = Array2::<f64>::zeros((hvp.dim, hvp.dim));
        let mut basis = vec![0.0_f64; hvp.dim];
        for j in 0..hvp.dim {
            basis[j] = 1.0;
            let col = (hvp.apply)(&basis);
            basis[j] = 0.0;
            if col.len() != hvp.dim || col.iter().any(|v| !v.is_finite()) {
                return Err(format!(
                    "evidence HVP logdet expected finite column of length {}, got {}",
                    hvp.dim,
                    col.len()
                ));
            }
            for i in 0..hvp.dim {
                dense[[i, j]] = col[i];
            }
        }
        validate_dense_hvp_symmetry(&dense)?;
        for i in 0..hvp.dim {
            for j in (i + 1)..hvp.dim {
                let avg = 0.5 * (dense[[i, j]] + dense[[j, i]]);
                dense[[i, j]] = avg;
                dense[[j, i]] = avg;
            }
        }
        dense_spd_log_det(&dense)
    } else {
        stochastic_hvp_log_det(hvp)
    }
}

fn dense_spd_log_det(matrix: &Array2<f64>) -> Result<f64, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "evidence dense logdet requires square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    if gam_gpu::cuda_selected().map_err(|error| error.to_string())? {
        return crate::gpu::reml_gpu::evidence_derivatives_gpu(
            crate::gpu::reml_gpu::RemlGpuInput {
                penalized_hessian: matrix.view(),
                derivative_hessians: Vec::new(),
            },
        )
        .map(|evidence| evidence.logdet_hessian);
    }
    let (evals, _) = matrix
        .eigh(Side::Lower)
        .map_err(|e| format!("evidence dense logdet eigendecomposition failed: {e}"))?;
    let mut logdet = 0.0_f64;
    for (idx, &ev) in evals.iter().enumerate() {
        if !ev.is_finite() || ev <= 0.0 {
            return Err(format!(
                "evidence dense logdet expected SPD Hessian, eigenvalue {idx} is {ev:.3e}"
            ));
        }
        logdet += ev.ln();
    }
    Ok(logdet)
}

fn validate_dense_hvp_symmetry(matrix: &Array2<f64>) -> Result<(), String> {
    let n = matrix.nrows();
    let mut norm_sq = 0.0_f64;
    for &value in matrix.iter() {
        norm_sq += value * value;
    }

    let mut skew_sq = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let skew = matrix[[i, j]] - matrix[[j, i]];
            skew_sq += 2.0 * skew * skew;
        }
    }

    let rel_skew = skew_sq.sqrt() / norm_sq.sqrt().max(1.0);
    if !rel_skew.is_finite() || rel_skew > EVIDENCE_HVP_SYMMETRY_REL_TOL {
        return Err(format!(
            "evidence HVP logdet requires symmetric operator, relative skew norm is {rel_skew:.3e}"
        ));
    }
    Ok(())
}

fn validate_hvp_randomized_symmetry(hvp: EvidenceHvpLogDet<'_>) -> Result<(), String> {
    let inv_norm = 1.0 / (hvp.dim as f64).sqrt();
    for probe in 0..EVIDENCE_HVP_SYMMETRY_PROBES.max(1) {
        let mut x = vec![0.0_f64; hvp.dim];
        let mut y = vec![0.0_f64; hvp.dim];
        rademacher_unit_probe_into_slice(&mut x, (2 * probe) as u64, inv_norm);
        rademacher_unit_probe_into_slice(&mut y, (2 * probe + 1) as u64, inv_norm);

        let hx = (hvp.apply)(&x);
        let hy = (hvp.apply)(&y);
        if hx.len() != hvp.dim || hx.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "evidence HVP symmetry check expected finite vector of length {}, got {}",
                hvp.dim,
                hx.len()
            ));
        }
        if hy.len() != hvp.dim || hy.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "evidence HVP symmetry check expected finite vector of length {}, got {}",
                hvp.dim,
                hy.len()
            ));
        }

        let lhs = dot_slice(&x, &hy);
        let rhs = dot_slice(&hx, &y);
        let scale = (norm2_slice(&hx) * norm2_slice(&y))
            .max(norm2_slice(&hy) * norm2_slice(&x))
            .max(lhs.abs())
            .max(rhs.abs())
            .max(1.0);
        let rel = (lhs - rhs).abs() / scale;
        if !rel.is_finite() || rel > EVIDENCE_HVP_SYMMETRY_REL_TOL {
            return Err(format!(
                "evidence HVP logdet requires symmetric operator, randomized symmetry probe {probe} has relative bilinear mismatch {rel:.3e}"
            ));
        }
    }
    Ok(())
}

fn stochastic_hvp_log_det(hvp: EvidenceHvpLogDet<'_>) -> Result<f64, String> {
    validate_hvp_randomized_symmetry(hvp)?;
    let probes = EVIDENCE_LOGDET_SLQ_PROBES.max(1);
    let steps = EVIDENCE_LOGDET_LANCZOS_STEPS.min(hvp.dim).max(1);
    let inv_norm = 1.0 / (hvp.dim as f64).sqrt();
    let mut estimate = 0.0_f64;
    for probe in 0..probes {
        let mut q0 = vec![0.0_f64; hvp.dim];
        rademacher_unit_probe_into_slice(&mut q0, probe as u64, inv_norm);
        let quad = lanczos_log_quadrature_hvp(hvp, q0, steps)?;
        estimate += hvp.dim as f64 * quad;
    }
    Ok(estimate / probes as f64)
}

fn lanczos_log_quadrature_hvp(
    hvp: EvidenceHvpLogDet<'_>,
    q: Vec<f64>,
    max_steps: usize,
) -> Result<f64, String> {
    let n = hvp.dim;
    let eigen = symmetric_lanczos_eigenpairs(
        n,
        &q,
        SymmetricLanczosOptions {
            max_steps,
            residual_tol: 1e-12,
            local_reorthogonalize: false,
            full_reorthogonalize: false,
        },
        |q, out| {
            let applied = (hvp.apply)(q);
            if applied.len() != n || applied.iter().any(|v| !v.is_finite()) {
                return Err(format!(
                    "evidence HVP SLQ expected finite vector of length {n}, got {}",
                    applied.len()
                ));
            }
            out.copy_from_slice(&applied);
            Ok(())
        },
    )
    .map_err(|e| format!("evidence HVP SLQ Lanczos failed: {e}"))?;
    symmetric_lanczos_log_quadrature(&eigen, "evidence HVP SLQ expected SPD Hessian")
}

#[inline]
fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut s = 0.0_f64;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn norm2_slice(a: &[f64]) -> f64 {
    dot_slice(a, a).sqrt()
}

fn rademacher_unit_probe_into_slice(z: &mut [f64], probe: u64, scale: f64) {
    let mut state = 0x6A09E667F3BCC909_u64 ^ probe.wrapping_mul(0xD1B54A32D192ED03);
    let mut bits = 0_u64;
    let mut remaining_bits = 0_u32;
    for value in z.iter_mut() {
        if remaining_bits == 0 {
            bits = splitmix64(&mut state);
            remaining_bits = 64;
        }
        *value = if bits & 1 == 0 { scale } else { -scale };
        bits >>= 1;
        remaining_bits -= 1;
    }
}

#[inline]
const fn splitmix64(state: &mut u64) -> u64 {
    gam_linalg::utils::splitmix64(state)
}

/// Authoritative factored-arrow evidence log-determinant.
///
/// This reads the stored joint value that cache construction records for the
/// operator used by selected-inverse/adjointers. It intentionally does not
/// reconstruct a determinant from row and Schur pieces for damped caches; if
/// construction did not record an exact joint log-det, evidence must refuse the
/// cache and route to an explicit matrix-free fallback.
pub fn arrow_log_det_from_cache(cache: &ArrowFactorCache) -> Option<f64> {
    if let Some(log_det) = cache.joint_hessian_log_det {
        return log_det.is_finite().then_some(log_det);
    }
    if cache.ridge_t != 0.0 || cache.ridge_beta != 0.0 {
        return None;
    }
    if cache.k > 0 && !cache.schur_factor_is_undamped {
        return None;
    }
    cache.compute_undamped_arrow_log_det()
}

// ---------------------------------------------------------------------------
// IFT cascade: ∂u*/∂β → ∂β*/∂ρ → ∂u*/∂ρ
// ---------------------------------------------------------------------------

/// Tier-1 IFT sensitivity `∂u_i*/∂β = -H_uu_i⁻¹ H_uβ_i`.
///
/// Concatenated row-major to a single `(N·d) × K` dense matrix. Each
/// row block is solved with the **undamped** Cholesky factor. Proposal
/// §2.2 / §7.
pub fn ift_du_dbeta(cache: &ArrowFactorCache) -> Array2<f64> {
    let n = cache.undamped_factor_count();
    let total_len = cache.delta_t_len();
    let k = cache.k;
    if !cache.htbeta_available() {
        return Array2::<f64>::from_elem((total_len, k), f64::NAN);
    }
    let mut out = Array2::<f64>::zeros((total_len, k));
    let mut beta_basis = Array1::<f64>::zeros(k);
    // Allocate scratch at max_d; per-row slice is ..di.
    let mut rhs = Array1::<f64>::zeros(cache.d);
    for i in 0..n {
        let di = cache.row_dims[i];
        let row_base = cache.row_offsets[i];
        let factor = cache.undamped_factor(i);
        // Solve H_uu_i Y = H_uβ_i column by column.
        for col in 0..k {
            beta_basis.fill(0.0);
            beta_basis[col] = 1.0;
            let mut rhs_i = rhs.slice_mut(ndarray::s![..di]).to_owned();
            // The Tier-2 IFT assembler is built only when the family's
            // capability surface promises cached `H_tβ` row products.
            if !cache.apply_htbeta_row(i, beta_basis.view(), &mut rhs_i) {
                // SAFETY: reaching `false` means a family declared the cache
                // available but failed to populate it — contract violation.
                return Array2::<f64>::from_elem((total_len, k), f64::NAN);
            }
            let y = cholesky_solve_vector(factor, &rhs_i);
            for c in 0..di {
                out[[row_base + c, col]] = -y[c];
            }
        }
    }
    out
}

/// Coupling components of a symmetric coefficient Hessian: the connected
/// components of the graph whose vertices are coefficient indices `0..p` and
/// whose edges are the structurally nonzero off-diagonal entries of `H` (#779).
///
/// Returns a length-`p` vector of component labels in `0..num_components`,
/// where two indices share a label iff they are connected through a chain of
/// nonzero `H[i,j]` couplings. This is the exact structural partition the
/// cone-of-influence sensitivity reuse is keyed on: a smoothing-parameter move
/// whose stationarity-gradient derivative `∂g/∂ρ` is supported only inside one
/// component can change `β = -H⁻¹ ∂g/∂ρ` only inside that same component, so
/// the sensitivity of every *other* component is provably unchanged and may be
/// reused unrecomputed (lazy/local propagation).
///
/// The nonzero test is exact (`!= 0.0`), matching the structural-coupling gate
/// used elsewhere for the joint inner Hessian: a tolerance would risk dropping a
/// genuine (small) coupling edge and silently biasing the propagated sensitivity
/// — the failure mode #779/#740 explicitly guard against. A block-diagonal `H`
/// yields the all-singletons partition (one component per block-decoupled
/// coordinate); a fully coupled `H` yields a single component (no shortcut, the
/// full joint solve is required — and is what the non-coned path performs).
pub fn coupling_components(hessian: ArrayView2<'_, f64>) -> Vec<usize> {
    let p = hessian.nrows();
    if p == 0 || hessian.ncols() != p {
        return Vec::new();
    }
    // Union-find with path compression and union by size.
    let mut parent: Vec<usize> = (0..p).collect();
    let mut size: Vec<usize> = vec![1; p];

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    for i in 0..p {
        for j in (i + 1)..p {
            // Symmetric structure: an edge exists if either triangle is nonzero,
            // so a numerically one-sided fill still couples the two indices.
            if hessian[[i, j]] != 0.0 || hessian[[j, i]] != 0.0 {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    let (small, large) = if size[ri] < size[rj] {
                        (ri, rj)
                    } else {
                        (rj, ri)
                    };
                    parent[small] = large;
                    size[large] += size[small];
                }
            }
        }
    }

    // Relabel roots to a dense `0..num_components` range, preserving
    // first-seen order so labels are deterministic.
    let mut label_of_root: Vec<Option<usize>> = vec![None; p];
    let mut next_label = 0usize;
    let mut labels = vec![0usize; p];
    for idx in 0..p {
        let root = find(&mut parent, idx);
        let label = match label_of_root[root] {
            Some(l) => l,
            None => {
                let l = next_label;
                label_of_root[root] = Some(l);
                next_label += 1;
                l
            }
        };
        labels[idx] = label;
    }
    labels
}

/// The cone of influence of a single stationarity-gradient derivative column
/// whose support (the coefficient indices where `∂g/∂ρ_k` is nonzero) lies in
/// `support`: the set of coefficient indices in the same coupling component(s)
/// as that support, given precomputed `labels` from [`coupling_components`].
///
/// `β_k = -H⁻¹ ∂g/∂ρ_k` is exactly zero outside this cone, so a confined solve
/// (or reuse of a cached zero) is exact, not an approximation. An empty support
/// (a structurally inactive `ρ_k`, e.g. a rank-0 or out-of-range penalty block)
/// yields an empty cone: the sensitivity is identically zero and no solve is
/// needed at all.
pub fn cone_of_influence(labels: &[usize], support: &[usize]) -> Vec<usize> {
    if support.is_empty() {
        return Vec::new();
    }
    let mut in_cone_labels: Vec<usize> = support
        .iter()
        .filter_map(|&idx| labels.get(idx).copied())
        .collect();
    in_cone_labels.sort_unstable();
    in_cone_labels.dedup();
    if in_cone_labels.is_empty() {
        return Vec::new();
    }
    (0..labels.len())
        .filter(|idx| in_cone_labels.binary_search(&labels[*idx]).is_ok())
        .collect()
}

/// Tier-2 IFT sensitivity `∂β*/∂ρ = -A⁻¹ ∂g_red/∂ρ` (proposal §2.4 /
/// §7).
///
/// `dg_red_drho` is the `K × R` matrix whose `a`-th column is `q_a =
/// ∂g_red/∂ρ_a`. Returns the `K × R` matrix `β_ρ`.
///
/// Returns `None` if the Schur factor is unavailable (PCG mode) or was
/// built from a damped operator, or if any solved entry is non-finite;
/// callers must not silently substitute an approximation. The solve is
/// the one sensitivity operator (#935) — this site holds no private H⁻¹
/// convention of its own.
pub fn ift_dbeta_drho(
    cache: &ArrowFactorCache,
    dg_red_drho: ArrayView2<'_, f64>,
) -> Option<Array2<f64>> {
    if !cache.schur_factor_is_undamped {
        return None;
    }
    let schur = cache.schur_factor.as_ref()?;
    if dg_red_drho.nrows() != cache.k || schur.nrows() != cache.k {
        return None;
    }
    crate::sensitivity::FitSensitivity::from_lower_triangular(schur).mode_response(dg_red_drho)
}

// ---------------------------------------------------------------------------
// ∂V/∂ρ — analytic optimized-evidence gradient via IFT mode response
// ---------------------------------------------------------------------------

/// IFT terms needed to differentiate the optimized Laplace evidence through
/// the fitted mode `(β*(ρ), u*(ρ))`.
///
/// For each hyperparameter `ρ_a`, the correction added to the direct trace is
///
/// ```text
/// F_β · β_a + F_u · u_a
/// + 0.5 (∂_β log|H| · β_a + ∂_u log|H| · u_a).
/// ```
///
/// At an exact KKT point the value-gradient pieces are zero, but they are
/// explicit here so the exported gradient matches the optimized objective
/// whenever callers carry a certified nonzero residual correction.
#[derive(Clone)]
pub struct EvidenceIftGradientTerms<'a> {
    pub dbeta_drho: ArrayView2<'a, f64>,
    pub du_drho: ArrayView2<'a, f64>,
    pub value_beta: ArrayView1<'a, f64>,
    pub value_u: ArrayView1<'a, f64>,
    pub logdet_h_beta: ArrayView1<'a, f64>,
    pub logdet_h_u: ArrayView1<'a, f64>,
}

/// Contract the IFT mode-response columns into the optimized-evidence
/// gradient correction.
pub fn evidence_ift_gradient_correction(terms: EvidenceIftGradientTerms<'_>) -> Array1<f64> {
    let k = terms.dbeta_drho.nrows();
    let nd = terms.du_drho.nrows();
    let r = terms.dbeta_drho.ncols();
    if terms.du_drho.ncols() != r
        || terms.value_beta.len() != k
        || terms.logdet_h_beta.len() != k
        || terms.value_u.len() != nd
        || terms.logdet_h_u.len() != nd
    {
        return Array1::<f64>::from_elem(r, f64::NAN);
    }

    let mut out = Array1::<f64>::zeros(r);
    for a in 0..r {
        let mut acc = 0.0_f64;
        for j in 0..k {
            let mode = terms.dbeta_drho[[j, a]];
            acc += terms.value_beta[j] * mode;
            acc += 0.5 * terms.logdet_h_beta[j] * mode;
        }
        for j in 0..nd {
            let mode = terms.du_drho[[j, a]];
            acc += terms.value_u[j] * mode;
            acc += 0.5 * terms.logdet_h_u[j] * mode;
        }
        out[a] = acc;
    }
    out
}

/// Per-`ρ` optimized-evidence gradient (proposal §3.7 / §3.8 split):
///
/// ```text
/// ∂V/∂ρ_a =
///       F_{ρ_a}                                  (value part)
///   + 0.5 tr(H⁻¹ H_{ρ_a})                        (direct Hessian)
///   + F_x · x_{ρ_a}
///   + 0.5 (∂_x log|H|) · x_{ρ_a}                 (IFT mode response)
///   - 0.5 tr(S_pen⁺ S_{pen,ρ_a})                 (penalty pseudo-logdet)
/// ```
/// where `x = (β, u)`.
///
/// The `tr(H⁻¹ H_{ρ_a})` trace is computed via the arrow structure
/// (proposal §3.5 / §3.10):
///
/// ```text
/// tr(H⁻¹ H_{ρ_a}) = Σ_i tr(H_uu_i⁻¹ ∂_{ρ_a} H_uu_i) + tr(A⁻¹ ∂_{ρ_a} A).
/// ```
///
/// `value_rho[a] = F_{ρ_a}` (envelope theorem, proposal §3.2).
/// `huu_drho[i][a]` is `∂H_uu_i/∂ρ_a` as a `d × d` matrix.
/// `hbb_drho[a]` is `∂H_ββ/∂ρ_a` as a `K × K` matrix.
/// `htbeta_drho[i][a]` is `∂H_uβ_i/∂ρ_a` as a `d × K` matrix.
/// `pen_logdet_drho[a]` is `∂_{ρ_a} log|S_pen|+`.
/// `ift_terms` carries `∂β*/∂ρ`, `∂u*/∂ρ`, and the already-contracted
/// mode derivatives of `F` and `log|H|`.
///
/// Returns the per-`ρ` gradient. Returns a NaN-filled vector when the
/// cache has no undamped Schur factor (PCG mode).
pub fn evidence_grad_rho(
    cache: &ArrowFactorCache,
    value_rho: ArrayView1<'_, f64>,
    huu_drho: &[Vec<Array2<f64>>],
    htbeta_drho: &[Vec<Array2<f64>>],
    hbb_drho: &[Array2<f64>],
    pen_logdet_drho: ArrayView1<'_, f64>,
    ift_terms: EvidenceIftGradientTerms<'_>,
) -> Array1<f64> {
    let r = value_rho.len();
    let n = cache.undamped_factor_count();
    let k = cache.k;
    let mut out = Array1::<f64>::zeros(r);
    if !cache.htbeta_available()
        || pen_logdet_drho.len() != r
        || huu_drho.len() != n
        || htbeta_drho.len() != n
        || hbb_drho.len() != r
        || huu_drho.iter().any(|row| row.len() != r)
        || htbeta_drho.iter().any(|row| row.len() != r)
        || hbb_drho.iter().any(|m| m.nrows() != k || m.ncols() != k)
        || huu_drho.iter().enumerate().any(|(i, row)| {
            let di = cache.row_dims[i];
            row.iter().any(|m| m.nrows() != di || m.ncols() != di)
        })
        || htbeta_drho.iter().enumerate().any(|(i, row)| {
            let di = cache.row_dims[i];
            row.iter().any(|m| m.nrows() != di || m.ncols() != k)
        })
    {
        out.fill(f64::NAN);
        return out;
    }
    let ift_correction = evidence_ift_gradient_correction(ift_terms);
    if ift_correction.len() != r || ift_correction.iter().any(|v| v.is_nan()) {
        out.fill(f64::NAN);
        return out;
    }

    let schur = match cache.schur_factor.as_ref() {
        Some(s) => s,
        None => {
            for a in 0..r {
                out[a] = f64::NAN;
            }
            return out;
        }
    };
    if !cache.schur_factor_is_undamped {
        for a in 0..r {
            out[a] = f64::NAN;
        }
        return out;
    }

    // Precompute Y_i = H_uu_i⁻¹ H_uβ_i (di × K). Used by both the Schur
    // derivative formula (§3.5) and the row trace `tr(H_uu_i⁻¹ ∂H_uu_i)`.
    let mut y_blocks: Vec<Array2<f64>> = Vec::with_capacity(n);
    let mut beta_basis = Array1::<f64>::zeros(k);
    // Scratch sized to max_d; per-row slice is ..di.
    let mut rhs = Array1::<f64>::zeros(cache.d);
    for i in 0..n {
        let di = cache.row_dims[i];
        let factor = cache.undamped_factor(i);
        let mut yi = Array2::<f64>::zeros((di, k));
        for col in 0..k {
            beta_basis.fill(0.0);
            beta_basis[col] = 1.0;
            let mut rhs_i = rhs.slice_mut(ndarray::s![..di]).to_owned();
            // Same H_tβ cache contract as the IFT du/dβ and du/dρ paths.
            if !cache.apply_htbeta_row(i, beta_basis.view(), &mut rhs_i) {
                // SAFETY: `false` means the family declared the cache
                // available but did not populate it — contract violation.
                out.fill(f64::NAN);
                return out;
            }
            let v = cholesky_solve_vector(factor, &rhs_i);
            for c in 0..di {
                yi[[c, col]] = v[c];
            }
        }
        y_blocks.push(yi);
    }

    // Outer-hoisted scratch reused across all (a, i) iterations.
    // Sized to max_d for trace_rhs and da_tmp; per-row slices used below.
    let mut trace_rhs = Array1::<f64>::zeros(cache.d);
    let mut da_tmp = Array2::<f64>::zeros((cache.d, k));
    let mut col_scratch = Array1::<f64>::zeros(k);
    for a in 0..r {
        // Part 1: F_{ρ_a} envelope contribution.
        let mut grad = value_rho[a];

        // Part 2a: Σ_i tr(H_uu_i⁻¹ ∂H_uu_i).
        // tr(H_uu_i⁻¹ M_i) = tr(L_iᵀ⁻¹ L_i⁻¹ M_i). Compute as the sum
        // over columns: solve L_i Lᵀ x = e_c for the c-th column of
        // M_i, then take its c-th component. Equivalently and more
        // cheaply, build (H_uu_i⁻¹ M_i) by solving column-by-column
        // and take its diagonal sum.
        let mut row_trace_acc = 0.0_f64;
        for i in 0..n {
            let di = cache.row_dims[i];
            let m_i = &huu_drho[i][a];
            assert_eq!(m_i.shape(), &[di, di]);
            for col in 0..di {
                let mut tr_rhs_i = trace_rhs.slice_mut(ndarray::s![..di]).to_owned();
                for r0 in 0..di {
                    tr_rhs_i[r0] = m_i[[r0, col]];
                }
                let v = cholesky_solve_vector(cache.undamped_factor(i), &tr_rhs_i);
                row_trace_acc += v[col];
            }
        }

        // Part 2b: tr(A⁻¹ ∂A) where (proposal §3.5)
        //     ∂A = ∂H_ββ
        //          - Σ_i (∂H_uβ_i)ᵀ Y_i
        //          - Σ_i Y_iᵀ (∂H_uβ_i)
        //          + Σ_i Y_iᵀ (∂H_uu_i) Y_i.
        // We accumulate ∂A as a dense `K × K` matrix, then evaluate
        // tr(A⁻¹ ∂A) by `Σ_j (A⁻¹ ∂A)[j, j]` via column solves of the
        // Schur Cholesky.
        let mut da = hbb_drho[a].clone();
        assert_eq!(da.shape(), &[k, k]);
        for i in 0..n {
            let di = cache.row_dims[i];
            let dhtb = &htbeta_drho[i][a]; // di × K
            let yi = &y_blocks[i]; // di × K
            // - (∂H_uβ_i)ᵀ Y_i
            for r0 in 0..k {
                for c0 in 0..k {
                    let mut acc = 0.0;
                    for cc in 0..di {
                        acc += dhtb[[cc, r0]] * yi[[cc, c0]];
                    }
                    da[[r0, c0]] -= acc;
                }
            }
            // - Y_iᵀ (∂H_uβ_i)
            for r0 in 0..k {
                for c0 in 0..k {
                    let mut acc = 0.0;
                    for cc in 0..di {
                        acc += yi[[cc, r0]] * dhtb[[cc, c0]];
                    }
                    da[[r0, c0]] -= acc;
                }
            }
            // + Y_iᵀ (∂H_uu_i) Y_i
            let dhuu = &huu_drho[i][a];
            // tmp = (∂H_uu_i) Y_i  (di × K) — use a slice of the hoisted buffer.
            let mut da_tmp_i = da_tmp.slice_mut(ndarray::s![..di, ..]).to_owned();
            for r0 in 0..di {
                for c0 in 0..k {
                    let mut acc = 0.0;
                    for cc in 0..di {
                        acc += dhuu[[r0, cc]] * yi[[cc, c0]];
                    }
                    da_tmp_i[[r0, c0]] = acc;
                }
            }
            // da += Y_iᵀ tmp
            for r0 in 0..k {
                for c0 in 0..k {
                    let mut acc = 0.0;
                    for cc in 0..di {
                        acc += yi[[cc, r0]] * da_tmp_i[[cc, c0]];
                    }
                    da[[r0, c0]] += acc;
                }
            }
        }

        // tr(A⁻¹ ∂A) via column solves.
        let mut schur_trace_acc = 0.0_f64;
        for j in 0..k {
            for r0 in 0..k {
                col_scratch[r0] = da[[r0, j]];
            }
            let v = cholesky_solve_vector(schur, &col_scratch);
            schur_trace_acc += v[j];
        }

        grad += 0.5 * (row_trace_acc + schur_trace_acc);
        grad += ift_correction[a];

        // Part 3: -0.5 ∂_{ρ_a} log|S_pen|+.
        grad -= 0.5 * pen_logdet_drho[a];

        out[a] = grad;
    }
    out
}

// ---------------------------------------------------------------------------
// Topology selection
// ---------------------------------------------------------------------------

/// Enumerate the candidate topologies, rank by normalized negative log
/// evidence, and return the winner. Failed/excluded candidates (proposal
/// §6.11) are appended at the end of `ranking` and are never the winner.
///
/// The caller fits each topology separately (proposal §4.2) and supplies
/// the resulting `TopologyCandidate` records. This function is purely
/// the discrete comparator + tie breaker.
///
/// # Tie-breaking
///
/// Per proposal §4.6: if normalized `|score_a - score_b| <= tie_tolerance`,
/// prefer the simpler topology by `TopologyKind::complexity_rank` (flat <
/// periodic < sphere < torus). The `tie` flag in the result records whether
/// such a tie occurred at the top of the ranking.
///
/// # Panics
///
/// Panics if `candidates` is empty after filtering out non-finite
/// scores. Proposal §6.11 explicitly forbids silent fallback to a
/// default topology; callers must handle the empty-candidate case
/// before invocation.
pub fn select_topology(
    candidates: &[TopologyCandidate],
    options: TopologySelectOptions,
) -> SelectedTopology {
    // Split valid and excluded.
    let mut valid: Vec<TopologyCandidate> = candidates
        .iter()
        .filter(|c| {
            c.converged
                && c.exclusion_reason.is_none()
                && c.negative_log_evidence.is_finite()
                && topology_selection_score(c, options.score_scale).is_finite()
        })
        .cloned()
        .collect();
    let mut excluded: Vec<TopologyCandidate> = candidates
        .iter()
        .filter(|c| {
            !(c.converged && c.exclusion_reason.is_none() && c.negative_log_evidence.is_finite())
                || !topology_selection_score(c, options.score_scale).is_finite()
        })
        .cloned()
        .collect();

    assert!(
        !valid.is_empty(),
        "select_topology: no finite valid candidates; proposal §6.11 forbids silent fallback"
    );

    // Sort by normalized negative log evidence (ascending = best first),
    // breaking ties by complexity_rank (smaller wins). The shared selector is
    // the single lower-is-better ordering contract used by topology ranking,
    // seed screening, and REML model comparison (#782).
    valid = rank_priority_candidates(
        valid
            .into_iter()
            .enumerate()
            .map(|(idx, row)| {
                let score = topology_selection_score(&row, options.score_scale);
                let tie_break = usize::from(row.kind.complexity_rank());
                PriorityCandidate::new(row, idx, score, tie_break)
            })
            .collect(),
    )
    .into_iter()
    .map(|row| row.item)
    .collect();

    // Detect numerical tie at the top.
    let tie = if valid.len() >= 2 {
        let top = topology_selection_score(&valid[0], options.score_scale);
        let next = topology_selection_score(&valid[1], options.score_scale);
        (next - top).abs() <= options.tie_tolerance
    } else {
        false
    };

    // If tied, prefer simpler topology among the tied prefix.
    if tie {
        let top_score = topology_selection_score(&valid[0], options.score_scale);
        // Find the tied prefix range.
        let tied_end = valid
            .iter()
            .position(|c| {
                (topology_selection_score(c, options.score_scale) - top_score).abs()
                    > options.tie_tolerance
            })
            .unwrap_or(valid.len());
        // Sort the tied prefix by complexity_rank ascending.
        valid[..tied_end].sort_by_key(|c| c.kind.complexity_rank());
    }

    let winner = valid[0].kind;
    valid.append(&mut excluded);
    SelectedTopology {
        winner,
        ranking: valid,
        tie,
    }
}

fn topology_selection_score(candidate: &TopologyCandidate, scale: TopologyScoreScale) -> f64 {
    match scale {
        TopologyScoreScale::PerObservation => {
            if candidate.n_obs == 0 {
                f64::NAN
            } else {
                candidate.negative_log_evidence / candidate.n_obs as f64
            }
        }
        TopologyScoreScale::PerEffectiveDim => {
            if !(candidate.effective_dim.is_finite() && candidate.effective_dim > 0.0) {
                f64::NAN
            } else {
                candidate.negative_log_evidence / candidate.effective_dim
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cache verification helpers
// ---------------------------------------------------------------------------

/// Verifies the `ArrowSchurSystem` dimensions match the cache. Used as
/// a debug-time precondition; never silently masks shape errors
/// (proposal §6.9 — sign and shape errors must be loud).
pub fn cache_matches_system(cache: &ArrowFactorCache, sys: &ArrowSchurSystem) -> bool {
    cache.d == sys.d
        && cache.k == sys.k
        && cache.n_rows() == sys.rows.len()
        && cache.undamped_factor_count() == sys.rows.len()
        && cache.manifold_mode_fingerprint == sys.manifold_mode_fingerprint
        && cache.row_hessian_fingerprint == sys.current_row_hessian_fingerprint()
}

// ---------------------------------------------------------------------------
// #1026 hybrid curved + linear-tail dictionary split-selection
// ---------------------------------------------------------------------------
//
// COMMON-EVIDENCE NOTE (#1202): the candidates BOTH fit the same data — the
// atom's leave-this-atom-out response residual `y_resp` (the response with every
// other atom's contribution removed). The curved candidate predicts the atom's
// actual mass-scaled contribution `a_k·γ_k`, the linear candidate the best
// mass-weighted straight line fit to `y_resp`. Because the curved family's
// `Θ = 0` member reproduces the linear prediction exactly, linear IS the nested
// `Θ = 0` sub-model on common data, so the "match-or-beat" statements below are a
// genuine data-level comparison: the curved candidate wins only when fitting the
// response residual better than its own straight projection pays for its extra
// parameters. See `crate::terms::sae::hybrid_split` for the residual assembly.
//
// The per-slot adjudication uses the SAME rank-aware Laplace evidence criterion
// the union/mixture rungs use (`−V = NLE`, lower wins), comparing the data-fit +
// complexity cost of the curved contribution against that of the straight line.
//
// ## The turning floor (Θ → 0) and the curved ceiling (Θ large)
//
// Per slot, the curved candidate fits the response residual with its actual
// mass-scaled contribution `a_k·γ_k` (data-fit `½·curved_rss`) and pays a larger
// free-parameter price `P_curved > P_linear`; the linear candidate fits the same
// residual with its best straight line (data-fit `½·linear_rss ≥ ½·curved_rss`
// whenever the curve beats its own straight projection) at a smaller price,
// charged with its genuine weighted Gram logdet `p·(log w_sum + log s_tt)`
// (#1203). Hence:
//
//   * Θ → 0 (the residual is straight): the curve and the line fit it equally, so
//     the cheaper LINEAR candidate wins — the turning floor / nested dominance. A
//     curved parameterization "buys nothing" on an already-straight residual.
//   * Θ large (a genuinely turning residual): the line's data-fit residual
//     exceeds the curved atom's extra parameter price, so CURVED wins. (Whether
//     curved wins also depends on the coordinate spread `s_tt` and amplitude, via
//     the honest logdet — a tightly-spread, mildly-curved residual can still
//     prefer the cheaper line.)
//
// The crossover is governed by the documented shatter law: a linear SAE shatters
// a feature of total turning Θ into `N(ε) ≈ Θ/(2√(2ε))` rank-1 directions at
// relative reconstruction error ε, so the curved advantage scales as `Θ/√ε`. We
// use the fitted turning Θ (`sae::chart_canonicalization::d1_atom_fitted_turning`)
// as the decision FEATURE: it both (a) sharpens the evidence comparison into a
// falsifiable per-atom prediction and (b) provides the exact-zero dominance
// guard — when an atom's fitted turning is identically zero, the curved fit has
// no curvature to price and the linear special case is selected by construction,
// independent of finite-sample evidence noise.

/// Which atom parameterization a hybrid-dictionary slot selects: a CURVED atom
/// (a `latent_dim ≥ 1` curved basis whose decoded image may turn) or its LINEAR
/// special case (the euclidean-d=1-linear atom — one straight decoder direction,
/// `γ(t) = t·b`, fitted turning `Θ = 0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HybridAtomParam {
    /// The curved atom (`latent_dim ≥ 1`), priced at its full coefficient count.
    Curved { latent_dim: usize },
    /// The linear special case: one decoder direction, zero turning.
    Linear,
}

impl HybridAtomParam {
    /// Stable display name for logs and tests.
    pub const fn as_str(self) -> &'static str {
        match self {
            HybridAtomParam::Curved { .. } => "curved",
            HybridAtomParam::Linear => "linear",
        }
    }

    /// `true` iff this is the linear special case (the linear tail).
    pub const fn is_linear(self) -> bool {
        matches!(self, HybridAtomParam::Linear)
    }
}

/// One fitted candidate parameterization for a single hybrid-dictionary atom
/// slot, scored on the COMMON rank-aware Laplace scale (`−V = NLE`, lower wins,
/// identical to the union/mixture rungs). The curved and linear candidates for
/// the SAME slot are fit on the same rows AND the same data (the atom's response
/// residual, #1202), so their NLEs are directly comparable; the structural
/// difference is the curved candidate's larger free-parameter price and whatever
/// data-fit it buys with its curvature.
#[derive(Debug, Clone, Copy)]
pub struct HybridAtomCandidate {
    pub param: HybridAtomParam,
    /// Rank-aware Laplace negative-log-evidence on the common scale (lower wins).
    pub negative_log_evidence: f64,
    /// Free-parameter count this candidate is charged for (the complexity price).
    pub num_parameters: usize,
    /// The candidate's fitted total turning `Θ = ∫κ ds` of its decoded curve, if
    /// the basis admits an analytic second jet. `Some(0.0)` for a linear atom (a
    /// straight image has no turning); `None` when the turning is honestly
    /// unavailable (no second jet / degenerate curve) — never fabricated.
    pub fitted_turning: Option<f64>,
}

impl HybridAtomCandidate {
    /// A linear special-case candidate: exact zero turning by construction.
    pub fn linear(negative_log_evidence: f64, num_parameters: usize) -> Self {
        Self {
            param: HybridAtomParam::Linear,
            negative_log_evidence,
            num_parameters,
            fitted_turning: Some(0.0),
        }
    }

    /// A curved candidate of the given latent dimension, with its fitted turning.
    pub fn curved(
        latent_dim: usize,
        negative_log_evidence: f64,
        num_parameters: usize,
        fitted_turning: Option<f64>,
    ) -> Self {
        Self {
            param: HybridAtomParam::Curved { latent_dim },
            negative_log_evidence,
            num_parameters,
            fitted_turning,
        }
    }
}

/// The evidence-selected parameterization for one hybrid-dictionary atom slot:
/// the winning candidate, plus the curved/linear NLEs that decided it (for the
/// EV-vs-Θ diagnostic and the tie-break audit trail).
#[derive(Debug, Clone, Copy)]
pub struct HybridAtomChoice {
    pub param: HybridAtomParam,
    /// The winning candidate's NLE.
    pub negative_log_evidence: f64,
    /// The winning candidate's free-parameter price.
    pub num_parameters: usize,
    /// The curved candidate's fitted turning `Θ` (the decision feature). `None`
    /// when no curved candidate offered an analytic turning.
    pub curved_turning: Option<f64>,
    /// `NLE_linear − NLE_curved`: the evidence margin the curved fit won (or lost,
    /// if negative) over the linear special case at this slot. Positive ⇒ curved
    /// bought more evidence than its parameter price; ≤ 0 ⇒ the dominance floor
    /// keeps the linear tail.
    pub curved_evidence_margin: f64,
}

/// Below this fitted turning the curved candidate is treated as straight: its
/// curvature is numerically indistinguishable from zero, so the dominance floor
/// (the linear special case is cheaper at equal likelihood) is enforced by
/// construction rather than left to finite-sample evidence noise. This is the
/// exact-zero guard from the `Θ → 0 ⇒ N(ε) → 0` limit of the shatter law, not a
/// tunable knob: it is the curvature scale below which `‖γ' ∧ γ''‖` is at the
/// floor of the Simpson quadrature for a genuinely straight image.
pub const HYBRID_LINEAR_TURNING_FLOOR: f64 = 1e-9;

/// Adjudicate the curved-vs-linear parameterization for ONE hybrid-dictionary
/// atom slot by the common rank-aware Laplace evidence criterion.
///
/// Selection rule (all on the single `NLE = −V` scale, lower wins):
///
///  1. **Dominance floor (Θ → 0).** If the curved candidate's fitted turning is
///     `Some(Θ)` with `Θ ≤ HYBRID_LINEAR_TURNING_FLOOR` and a linear candidate
///     exists, select LINEAR. A straight curved fit recovers no likelihood the
///     linear special case does not, and the linear atom is strictly cheaper, so
///     it cannot lose — we enforce that exactly instead of trusting evidence
///     noise at the floor.
///  2. **Evidence comparison.** Otherwise select the candidate with the smaller
///     `NLE`. The curved candidate wins only when its extra curvature lowers the
///     NLE by MORE than its extra parameter price — the `Θ/√ε` crossover, decided
///     here by the evidence numbers themselves, not by fiat. This is a
///     common-data comparison (both candidates fit the atom's response residual,
///     see `crate::terms::sae::hybrid_split`) in which linear is the curved
///     family's nested `Θ = 0` sub-model (#1202): the curved candidate cannot be
///     charged its extra parameters to fit the residual no better than its own
///     straight projection, and a tightly-spread, mildly-curved residual can
///     still prefer the cheaper line.
///  3. **Tie-break.** Exact NLE ties go to the cheaper (fewer-parameter)
///     candidate — i.e. linear — preserving the strict-generalization guarantee
///     that the hybrid never pays for curvature it does not need.
///
/// `candidates` must contain at most one linear and at most one curved candidate
/// for the slot; returns `None` only if `candidates` is empty.
pub fn select_hybrid_atom(candidates: &[HybridAtomCandidate]) -> Option<HybridAtomChoice> {
    if candidates.is_empty() {
        return None;
    }
    let linear = candidates.iter().find(|c| c.param.is_linear());
    let curved = candidates.iter().find(|c| !c.param.is_linear());
    let curved_turning = curved.and_then(|c| c.fitted_turning);
    let curved_evidence_margin = match (linear, curved) {
        (Some(l), Some(c)) => l.negative_log_evidence - c.negative_log_evidence,
        _ => 0.0,
    };

    // (1) Exact-zero dominance floor: a straight curved fit yields to the linear
    // special case by construction.
    if let (Some(l), Some(turning)) = (linear, curved_turning)
        && turning <= HYBRID_LINEAR_TURNING_FLOOR
    {
        return Some(HybridAtomChoice {
            param: l.param,
            negative_log_evidence: l.negative_log_evidence,
            num_parameters: l.num_parameters,
            curved_turning,
            curved_evidence_margin,
        });
    }

    // (2)+(3) Evidence argmin with the cheaper candidate winning exact ties.
    let mut best = candidates[0];
    for cand in &candidates[1..] {
        let better_evidence = cand.negative_log_evidence < best.negative_log_evidence;
        let tied = cand.negative_log_evidence == best.negative_log_evidence;
        let cheaper_on_tie = tied && cand.num_parameters < best.num_parameters;
        if better_evidence || cheaper_on_tie {
            best = *cand;
        }
    }
    Some(HybridAtomChoice {
        param: best.param,
        negative_log_evidence: best.negative_log_evidence,
        num_parameters: best.num_parameters,
        curved_turning,
        curved_evidence_margin,
    })
}

/// The evidence-selected split for a whole hybrid dictionary: the per-atom
/// curved-vs-linear choices and the dictionary-level aggregates the EV-vs-Θ
/// frontier reports against.
#[derive(Debug, Clone)]
pub struct HybridSplitSelection {
    /// One adjudicated choice per atom slot, in slot order.
    pub atoms: Vec<HybridAtomChoice>,
    /// `Σ NLE` across the selected per-atom parameterizations — the dictionary's
    /// summed rank-aware Laplace negative-log-evidence (lower wins). Because each
    /// slot picks the argmin over {curved contribution, best straight line to the
    /// response residual}, this is ≤ the sum of the per-slot LINEAR-candidate
    /// NLEs. The linear baseline is the best straight line fit to each atom's
    /// leave-this-atom-out RESPONSE residual (#1202), the curved family's nested
    /// `Θ = 0` member on common data — so this is a genuine data-level
    /// match-or-beat dominance, not a post-hoc curve-simplification one.
    pub total_negative_log_evidence: f64,
    /// `Σ P` across the selected parameterizations — the dictionary's total
    /// free-parameter price (the matched-active-budget accounting).
    pub total_parameters: usize,
    /// Count of slots that selected the curved parameterization.
    pub curved_atom_count: usize,
}

impl HybridSplitSelection {
    /// Count of slots that selected the linear special case (the linear tail).
    pub fn linear_atom_count(&self) -> usize {
        self.atoms.len() - self.curved_atom_count
    }

    /// `true` iff every slot selected linear — the pure-linear limit, reached
    /// when every feature is straight (all `Θ → 0`).
    pub fn is_pure_linear(&self) -> bool {
        self.curved_atom_count == 0 && !self.atoms.is_empty()
    }

    /// `true` iff every slot selected curved — the pure-curved limit, reached
    /// when every feature turns enough to pay for curvature.
    pub fn is_pure_curved(&self) -> bool {
        self.curved_atom_count == self.atoms.len() && !self.atoms.is_empty()
    }
}

/// Adjudicate the curved-vs-linear split across a whole hybrid dictionary by the
/// common evidence criterion. `slots[i]` holds the curved/linear candidates for
/// atom slot `i` (each scored on the same rows, on the common Laplace scale).
///
/// The result reduces EXACTLY to pure-linear when every slot's curved candidate
/// has `Θ → 0` (the turning floor fires everywhere) and to pure-curved when
/// every slot's curved candidate wins the evidence comparison. (Common-data
/// criterion, #1202 — both candidates fit the atom's response residual, with
/// linear nested as the curved family's `Θ = 0` sub-model; see the module header
/// above and `crate::terms::sae::hybrid_split`.)
///
/// Returns an error only if some slot has no candidates to adjudicate (an empty
/// dictionary slot is a caller bug, not a silent skip).
pub fn select_hybrid_split(
    slots: &[Vec<HybridAtomCandidate>],
) -> Result<HybridSplitSelection, String> {
    let mut atoms = Vec::with_capacity(slots.len());
    let mut total_nle = 0.0_f64;
    let mut total_parameters = 0usize;
    let mut curved_atom_count = 0usize;
    for (i, slot) in slots.iter().enumerate() {
        let choice = select_hybrid_atom(slot)
            .ok_or_else(|| format!("hybrid split slot {i} has no candidate parameterizations"))?;
        if !choice.negative_log_evidence.is_finite() {
            return Err(format!(
                "hybrid split slot {i} selected a non-finite evidence ({})",
                choice.negative_log_evidence
            ));
        }
        if !choice.param.is_linear() {
            curved_atom_count += 1;
        }
        total_nle += choice.negative_log_evidence;
        total_parameters += choice.num_parameters;
        atoms.push(choice);
    }
    Ok(HybridSplitSelection {
        atoms,
        total_negative_log_evidence: total_nle,
        total_parameters,
        curved_atom_count,
    })
}

// ---------------------------------------------------------------------------
// Tests
//
// These are type-level / structural tests: per the task contract we do
// not compile or run them in this session. They document the expected
// shapes and degenerate-case behavior so a future maintainer running
// `cargo test` sees the contract written down.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow_schur::ArrowFactorSlab;

    // Dense `H⁻¹` apply via explicit inverse (test-only reference solver).
    fn dense_inverse(h: &Array2<f64>) -> Array2<f64> {
        let p = h.nrows();
        let mut aug = Array2::<f64>::zeros((p, 2 * p));
        for i in 0..p {
            for j in 0..p {
                aug[[i, j]] = h[[i, j]];
            }
            aug[[i, p + i]] = 1.0;
        }
        for col in 0..p {
            let mut pivot = col;
            for row in (col + 1)..p {
                if aug[[row, col]].abs() > aug[[pivot, col]].abs() {
                    pivot = row;
                }
            }
            if pivot != col {
                for j in 0..(2 * p) {
                    aug.swap([col, j], [pivot, j]);
                }
            }
            let d = aug[[col, col]];
            for j in 0..(2 * p) {
                aug[[col, j]] /= d;
            }
            for row in 0..p {
                if row == col {
                    continue;
                }
                let f = aug[[row, col]];
                if f != 0.0 {
                    for j in 0..(2 * p) {
                        aug[[row, j]] -= f * aug[[col, j]];
                    }
                }
            }
        }
        let mut inv = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                inv[[i, j]] = aug[[i, p + j]];
            }
        }
        inv
    }

    #[test]
    fn coupling_components_block_diagonal_is_all_singletons_by_block() {
        // Two decoupled 2x2 blocks: {0,1} and {2,3}.
        let mut h = Array2::<f64>::eye(4);
        h[[0, 1]] = 0.3;
        h[[1, 0]] = 0.3;
        h[[2, 3]] = 0.7;
        h[[3, 2]] = 0.7;
        let labels = coupling_components(h.view());
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
        // Exactly two components.
        let mut uniq = labels.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 2);
    }

    #[test]
    fn coupling_components_fully_coupled_is_one_component() {
        let mut h = Array2::<f64>::eye(3);
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    h[[i, j]] = 0.1;
                }
            }
        }
        let labels = coupling_components(h.view());
        assert!(labels.iter().all(|&l| l == labels[0]));
    }

    #[test]
    fn coupling_components_transitive_chain_merges() {
        // 0-1 and 1-2 coupled (but no direct 0-2 edge) must form one component.
        let mut h = Array2::<f64>::eye(3);
        h[[0, 1]] = 0.5;
        h[[1, 0]] = 0.5;
        h[[1, 2]] = 0.5;
        h[[2, 1]] = 0.5;
        let labels = coupling_components(h.view());
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
    }

    #[test]
    fn compare_reml_fits_delta_and_bayes_factor_never_contradict_winner_gh1465() {
        // Regression for #1465: the ranking `delta` / `bayes_factor` must be
        // measured on the SAME scale that orders the table (the Occam-penalised
        // conditional AIC `ranking_score`), so every row's delta is >= 0 and its
        // Bayes factor >= 1 — the table must never claim a non-winner beats the
        // declared winner. The scenario is exactly the case the comparison
        // exists to handle: AIC and raw REML DISAGREE. `m1` is the AIC winner
        // but does NOT carry the minimum raw REML (`m2` does) — the noise
        // extra-term case from the issue.
        //
        // `ranking_score` = -2*log_lik + 2*edf; with log_lik = 0 it is `2*edf`,
        // so the AIC order is m1 < m2 < m3 while the raw-REML order has m2 lowest.
        let cand = |name: &str, score: f64, edf: f64| RemlCandidate {
            index: 0,
            name: name.to_string(),
            score,
            edf: Some(edf),
            log_lik: Some(0.0),
            family: Some("gaussian".to_string()),
            n_obs: Some(100),
        };
        // raw REML : m2 (41.605) < m1 (53.748) < m3 (120.011)
        // AIC=2*edf: m1 (100)    < m2 (102)    < m3 (130)
        let candidates = vec![
            cand("m1", 53.748, 50.0),
            cand("m2", 41.605, 51.0),
            cand("m3", 120.011, 65.0),
        ];
        let cmp = compare_reml_fits(candidates).expect("comparison");

        assert_eq!(cmp.winner, "m1", "AIC winner");
        // No ranking row may contradict the declared winner.
        for row in &cmp.ranking {
            assert!(
                row.delta >= 0.0,
                "ranking delta for {} must be >= 0, got {}",
                row.name,
                row.delta
            );
            assert!(
                row.bayes_factor >= 1.0 - 1e-12,
                "ranking bayes_factor for {} must be >= 1, got {}",
                row.name,
                row.bayes_factor
            );
        }
        let winner_row = cmp.ranking.iter().find(|r| r.name == "m1").unwrap();
        assert!(winner_row.delta.abs() < 1e-12, "winner delta == 0");
        assert!(
            (winner_row.bayes_factor - 1.0).abs() < 1e-9,
            "winner bayes_factor == 1"
        );

        // The raw-REML score table is referenced to the genuine minimum raw REML
        // (m2), so its best-over-model Bayes factors are also coherent (>= 1).
        for row in &cmp.score_table {
            assert!(
                row.delta_reml >= 0.0,
                "score-table delta_reml for {} must be >= 0, got {}",
                row.name,
                row.delta_reml
            );
            assert!(
                row.bayes_factor_best_over_model >= 1.0 - 1e-12,
                "score-table bayes_factor for {} must be >= 1, got {}",
                row.name,
                row.bayes_factor_best_over_model
            );
        }
        // m2 carries the minimum raw REML, so its raw delta is exactly 0.
        let m2 = cmp.score_table.iter().find(|r| r.name == "m2").unwrap();
        assert!(
            m2.delta_reml.abs() < 1e-12,
            "the minimum-raw-REML row has delta_reml 0"
        );
    }

    #[test]
    fn cone_of_influence_empty_support_is_empty() {
        let labels = vec![0usize, 0, 1, 1];
        assert!(cone_of_influence(&labels, &[]).is_empty());
    }

    #[test]
    fn cone_of_influence_returns_full_component() {
        let labels = vec![0usize, 0, 1, 1];
        // Support in component 0 -> cone is {0,1}.
        assert_eq!(cone_of_influence(&labels, &[0]), vec![0, 1]);
        // Support spanning both -> cone is everything.
        assert_eq!(cone_of_influence(&labels, &[1, 2]), vec![0, 1, 2, 3]);
    }

    #[test]
    fn coned_matches_full_solve_on_fully_coupled_hessian() {
        // Fully coupled SPD H: cone is the whole space, result must equal the
        // unconfined sensitivity-operator mode response bit-for-bit.
        let h = Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.8, 0.5, 0.8, 2.5])
            .unwrap();
        let inv = dense_inverse(&h);
        // Two ρ-columns, each supported on a single coefficient.
        let mut dg = Array2::<f64>::zeros((3, 2));
        dg[[0, 0]] = 1.3;
        dg[[2, 1]] = -0.7;
        let supports = vec![0..1usize, 2..3usize];

        let eye: Array2<f64> = Array2::eye(3);
        let op = crate::sensitivity::FitSensitivity::from_projected(&eye, &inv);
        let full = op.mode_response(dg.view()).unwrap();
        let coned = op
            .mode_response_coned(h.view(), dg.view(), &supports)
            .unwrap();
        for i in 0..3 {
            for a in 0..2 {
                assert!(
                    (full[[i, a]] - coned[[i, a]]).abs() < 1e-12,
                    "fully-coupled mismatch at ({i},{a}): {} vs {}",
                    full[[i, a]],
                    coned[[i, a]]
                );
            }
        }
    }

    #[test]
    fn coned_confines_to_component_on_decoupled_hessian() {
        // Block-decoupled H: blocks {0,1} and {2,3}. A column supported only in
        // block {0,1} must produce sensitivity zero in block {2,3}, and match
        // the exact solution within its own block.
        let mut h = Array2::<f64>::zeros((4, 4));
        // Block A.
        h[[0, 0]] = 4.0;
        h[[1, 1]] = 3.0;
        h[[0, 1]] = 1.0;
        h[[1, 0]] = 1.0;
        // Block B.
        h[[2, 2]] = 2.0;
        h[[3, 3]] = 5.0;
        h[[2, 3]] = 0.6;
        h[[3, 2]] = 0.6;
        let inv = dense_inverse(&h);

        let mut dg = Array2::<f64>::zeros((4, 1));
        dg[[0, 0]] = 0.9;
        dg[[1, 0]] = -0.4;
        let support_range = 0..2usize;
        let supports = std::slice::from_ref(&support_range);

        let eye: Array2<f64> = Array2::eye(4);
        let coned = crate::sensitivity::FitSensitivity::from_projected(&eye, &inv)
            .mode_response_coned(h.view(), dg.view(), supports)
            .unwrap();
        // Exact reference: -H⁻¹ q. Off-block entries are exactly zero already
        // (decoupled inverse), and the cone must preserve the in-block ones.
        let q = dg.column(0).to_owned();
        let exact = inv.dot(&q).mapv(|v| -v);
        for i in 0..4 {
            assert!(
                (coned[[i, 0]] - exact[[i]]).abs() < 1e-12,
                "decoupled mismatch at {i}: {} vs {}",
                coned[[i, 0]],
                exact[[i]]
            );
        }
        // Block B is outside the cone -> exactly zero.
        assert_eq!(coned[[2, 0]], 0.0);
        assert_eq!(coned[[3, 0]], 0.0);
    }

    #[test]
    fn coned_skips_inactive_column_with_empty_support() {
        let h = Array2::<f64>::eye(2);
        let dg = Array2::<f64>::zeros((2, 1));
        // Inactive ρ: empty support, must be skipped without solving.
        let empty_support = 0..0usize;
        let supports = std::slice::from_ref(&empty_support);
        // A NaN inverse: an empty-support column must be skipped WITHOUT
        // solving, so the operator's finite-check never sees the NaN and the
        // result is `Some(zeros)`. Were the inactive column ever solved, the
        // NaN would propagate and `mode_response_coned` would return `None`.
        let eye: Array2<f64> = Array2::eye(2);
        let nan_inv = Array2::<f64>::from_elem((2, 2), f64::NAN);
        let coned = crate::sensitivity::FitSensitivity::from_projected(&eye, &nan_inv)
            .mode_response_coned(h.view(), dg.view(), supports)
            .unwrap();
        assert_eq!(coned[[0, 0]], 0.0);
        assert_eq!(coned[[1, 0]], 0.0);
    }

    fn make_minimal_cache() -> ArrowFactorCache {
        // d = 1, k = 1, n = 1, H_uu_1 = [[2.0]] => L = [[sqrt(2)]],
        // H_uβ_1 = [[0.5]], A = 2 - 0.5 * 0.5 / 2 = 1.875.
        let l_huu = Array2::from_shape_vec((1, 1), vec![std::f64::consts::SQRT_2]).unwrap();
        let l_schur = Array2::from_shape_vec((1, 1), vec![(1.875_f64).sqrt()]).unwrap();
        let htbeta = Array2::from_shape_vec((1, 1), vec![0.5]).unwrap();
        let mut cache = ArrowFactorCache {
            htt_factors: ArrowFactorSlab::from_blocks(vec![l_huu]),
            htt_factors_undamped: crate::arrow_schur::ArrowUndampedFactors::SameAsDamped,
            schur_factor: Some(l_schur),
            schur_factor_is_undamped: true,
            beta_schur_deflation: None,
            joint_hessian_log_det: None,
            solver_mode: crate::arrow_schur::ArrowSolverMode::Direct,
            ridge_t: 0.0,
            ridge_beta: 0.0,
            htbeta: crate::arrow_schur::ArrowHtbetaCache::Dense {
                blocks: std::sync::Arc::from(vec![htbeta]),
                estimated_bytes: std::mem::size_of::<f64>(),
            },
            d: 1,
            row_dims: std::sync::Arc::from(vec![1usize]),
            row_offsets: std::sync::Arc::from(vec![0usize, 1usize]),
            k: 1,
            manifold_mode_fingerprint: 0,
            row_hessian_fingerprint: 0,
            pcg_diagnostics: crate::arrow_schur::ArrowPcgDiagnostics::default(),
            gauge_deflated_directions: 0,
            deflated_row_directions: std::sync::Arc::from(Vec::new()),
            deflation_row_spectra: std::sync::Arc::from(Vec::new()),
            beta_gauge_quotient: None,
        };
        cache.joint_hessian_log_det = cache.compute_undamped_arrow_log_det();
        cache
    }

    #[test]
    fn laplace_evidence_returns_finite_for_minimal_cache() {
        let cache = make_minimal_cache();
        // log|H| = log(2) + log(1.875). With dim(H)=2 and rank(S)=1,
        // V includes the rank-aware TK nullspace normalizer.
        let v = laplace_evidence(
            EvidenceLogDetSource::FactoredArrow {
                cache: &cache,
                fallback_hvp: None,
            },
            0.0,
            0.0,
            2.0,
            1.0,
        );
        assert!(v.is_finite());
        let expected =
            0.5 * (2.0_f64.ln() + 1.875_f64.ln()) - 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((v - expected).abs() < 1e-12);
    }

    /// #1132 bug 2: a β-profiled atom (no shared `β` block, `k == 0`) reaches
    /// `arrow_log_det_from_cache` in the dense Direct path with
    /// `schur_factor = None` — there is no reduced Schur complement to form. The
    /// joint Hessian is then block-diagonal in the latent rows, so its log-det
    /// is exactly the per-row sum with NO Schur term. Before the fix this
    /// returned `None` (the `schur_factor.as_ref()?` bail), starving the REML
    /// Laplace normaliser and erroring "arrow_log_det_from_cache returned None
    /// at ridge=0 Direct mode". Now it returns `Some(Σ_i log|H_tt^(i)|)`.
    fn k0_direct_cache_no_schur(latent_diag: f64) -> ArrowFactorCache {
        let l_huu = Array2::from_shape_vec((1, 1), vec![latent_diag.sqrt()]).unwrap();
        let mut cache = ArrowFactorCache {
            htt_factors: ArrowFactorSlab::from_blocks(vec![l_huu]),
            htt_factors_undamped: crate::arrow_schur::ArrowUndampedFactors::SameAsDamped,
            schur_factor: None,
            schur_factor_is_undamped: true,
            beta_schur_deflation: None,
            joint_hessian_log_det: None,
            solver_mode: crate::arrow_schur::ArrowSolverMode::Direct,
            ridge_t: 0.0,
            ridge_beta: 0.0,
            htbeta: crate::arrow_schur::ArrowHtbetaCache::Disabled { estimated_bytes: 0 },
            d: 1,
            row_dims: std::sync::Arc::from(vec![1usize]),
            row_offsets: std::sync::Arc::from(vec![0usize, 1usize]),
            k: 0,
            manifold_mode_fingerprint: 0,
            row_hessian_fingerprint: 0,
            pcg_diagnostics: crate::arrow_schur::ArrowPcgDiagnostics::default(),
            gauge_deflated_directions: 0,
            deflated_row_directions: std::sync::Arc::from(Vec::new()),
            deflation_row_spectra: std::sync::Arc::from(Vec::new()),
            beta_gauge_quotient: None,
        };
        cache.joint_hessian_log_det = cache.compute_undamped_arrow_log_det();
        cache
    }

    #[test]
    fn arrow_log_det_some_for_k0_direct_cache_without_schur() {
        let cache = k0_direct_cache_no_schur(3.0);
        let log_det = arrow_log_det_from_cache(&cache)
            .expect("k==0 Direct cache must yield Some(per-row sum), not None (#1132)");
        // Single latent block H_tt = [[3.0]]; no Schur term for k == 0.
        assert!(
            (log_det - 3.0_f64.ln()).abs() < 1e-12,
            "log_det = {log_det}"
        );
        // The cache's own computation must agree bit-for-bit.
        let cached = cache
            .compute_undamped_arrow_log_det()
            .expect("compute_undamped_arrow_log_det must be Some for k==0");
        assert!((cached - 3.0_f64.ln()).abs() < 1e-12, "cached = {cached}");
    }

    #[test]
    fn arrow_log_det_none_for_kpos_cache_without_schur() {
        // k > 0 but no dense Schur factor is the genuine InexactPCG case and
        // must still reject (the guard must not over-broaden to all `None`).
        let mut cache = k0_direct_cache_no_schur(3.0);
        cache.k = 1;
        cache.solver_mode = crate::arrow_schur::ArrowSolverMode::InexactPCG;
        cache.joint_hessian_log_det = None;
        assert!(arrow_log_det_from_cache(&cache).is_none());
        assert!(cache.compute_undamped_arrow_log_det().is_none());
    }

    #[test]
    fn laplace_evidence_nan_when_authoritative_logdet_missing() {
        let mut cache = make_minimal_cache();
        cache.ridge_t = 1e-3;
        cache.joint_hessian_log_det = None;
        assert!(
            laplace_evidence(
                EvidenceLogDetSource::FactoredArrow {
                    cache: &cache,
                    fallback_hvp: None,
                },
                0.0,
                0.0,
                2.0,
                1.0,
            )
            .is_nan()
        );
    }

    #[test]
    fn laplace_evidence_uses_hvp_fallback_without_authoritative_logdet() {
        let mut cache = make_minimal_cache();
        cache.schur_factor = None;
        cache.joint_hessian_log_det = None;
        let hvp = |x: &[f64]| -> Vec<f64> { vec![2.0 * x[0], 1.875 * x[1]] };
        let v = laplace_evidence(
            EvidenceLogDetSource::FactoredArrow {
                cache: &cache,
                fallback_hvp: Some(EvidenceHvpLogDet {
                    dim: 2,
                    apply: &hvp,
                }),
            },
            0.0,
            0.0,
            2.0,
            1.0,
        );
        let expected =
            0.5 * (2.0_f64.ln() + 1.875_f64.ln()) - 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((v - expected).abs() < 1e-12);
    }

    #[test]
    fn ift_du_dbeta_has_expected_shape() {
        let cache = make_minimal_cache();
        let du_db = ift_du_dbeta(&cache);
        assert_eq!(du_db.shape(), &[1, 1]);
        // ∂u/∂β = -H_uu⁻¹ H_uβ = -0.5 / 2 = -0.25.
        assert!((du_db[[0, 0]] - (-0.25)).abs() < 1e-12);
    }

    #[test]
    fn ift_dbeta_drho_returns_some_for_direct_cache() {
        let cache = make_minimal_cache();
        let q = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let out = ift_dbeta_drho(&cache, q.view()).unwrap();
        assert_eq!(out.shape(), &[1, 1]);
        // ∂β/∂ρ = -A⁻¹ · 1 = -1/1.875.
        assert!((out[[0, 0]] + 1.0 / 1.875).abs() < 1e-12);
    }

    #[test]
    fn topology_select_picks_lowest_negative_log_evidence() {
        let candidates = vec![
            TopologyCandidate {
                kind: TopologyKind::Flat,
                negative_log_evidence: 10.0,
                effective_dim: 4.0,
                n_obs: 100,
                converged: true,
                exclusion_reason: None,
            },
            TopologyCandidate {
                kind: TopologyKind::Sphere,
                negative_log_evidence: 8.0,
                effective_dim: 5.0,
                n_obs: 100,
                converged: true,
                exclusion_reason: None,
            },
            TopologyCandidate {
                kind: TopologyKind::Torus,
                negative_log_evidence: f64::NAN,
                effective_dim: 6.0,
                n_obs: 100,
                converged: false,
                exclusion_reason: Some("torus periods missing".to_string()),
            },
        ];
        let sel = select_topology(&candidates, TopologySelectOptions::default());
        assert_eq!(sel.winner, TopologyKind::Sphere);
        assert!(!sel.tie);
    }

    #[test]
    fn topology_select_tie_breaks_to_simpler() {
        let candidates = vec![
            TopologyCandidate {
                kind: TopologyKind::Sphere,
                negative_log_evidence: 5.0,
                effective_dim: 5.0,
                n_obs: 100,
                converged: true,
                exclusion_reason: None,
            },
            TopologyCandidate {
                kind: TopologyKind::Flat,
                negative_log_evidence: 5.0 + 1e-6,
                effective_dim: 4.0,
                n_obs: 100,
                converged: true,
                exclusion_reason: None,
            },
        ];
        let sel = select_topology(&candidates, TopologySelectOptions::default());
        assert_eq!(sel.winner, TopologyKind::Flat);
        assert!(sel.tie);
    }

    fn gaussian_logpdf(y: f64, mean: f64, sd: f64) -> f64 {
        let z = (y - mean) / sd;
        -0.5 * (2.0 * std::f64::consts::PI).ln() - sd.ln() - 0.5 * z * z
    }

    #[test]
    fn stacking_single_candidate_gets_full_weight() {
        let log_density = Array2::from_shape_vec((3, 1), vec![-1.0, -2.0, -0.5]).unwrap();
        let out = solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        assert!((out.weights[0] - 1.0).abs() < 1e-12);
        assert_eq!(out.weights.len(), 1);
    }

    #[test]
    fn stacking_dominant_candidate_attracts_nearly_all_weight() {
        let mut log_density = Array2::<f64>::zeros((50, 2));
        for i in 0..50 {
            log_density[[i, 0]] = -0.1;
            log_density[[i, 1]] = -5.0;
        }
        let out = solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        assert!(out.weights[0] > 0.99, "w0 = {}", out.weights[0]);
        assert!(out.weights[1] < 0.01, "w1 = {}", out.weights[1]);
    }

    #[test]
    fn stacking_complementary_candidates_share_weight() {
        // Each candidate is the better predictor on its own half of the data;
        // stacking keeps both, unlike winner-take-all.
        let n = 40;
        let mut log_density = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            if i < n / 2 {
                log_density[[i, 0]] = gaussian_logpdf(0.0, 0.0, 0.5);
                log_density[[i, 1]] = gaussian_logpdf(0.0, 1.5, 0.5);
            } else {
                log_density[[i, 0]] = gaussian_logpdf(0.0, 1.5, 0.5);
                log_density[[i, 1]] = gaussian_logpdf(0.0, 0.0, 0.5);
            }
        }
        let out = solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        assert!(
            out.weights[0] > 0.2 && out.weights[0] < 0.8,
            "w0 = {}",
            out.weights[0]
        );
        assert!((out.weights.sum() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn stacking_weights_stay_on_the_simplex() {
        let log_density = Array2::from_shape_vec(
            (3, 3),
            vec![-1.0, -2.0, -3.0, -2.5, -1.0, -2.0, -3.0, -2.0, -1.0],
        )
        .unwrap();
        let out = solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        assert!((out.weights.sum() - 1.0).abs() < 1e-9);
        assert!(out.weights.iter().all(|&w| w >= -1e-12));
    }

    #[test]
    fn stacking_solution_satisfies_the_simplex_kkt_certificate() {
        // Recompute the KKT residual from scratch at the returned weights:
        // g_k = mean_i p_ik / mix_i must satisfy g_k <= 1 (+tol) everywhere
        // and w_k * |g_k - 1| <= tol. The objective is concave, so this
        // certifies the GLOBAL optimum, not merely a stationary iterate.
        let log_density = Array2::from_shape_vec(
            (5, 2),
            vec![-0.2, -3.0, -3.0, -0.2, -0.5, -1.5, -1.5, -0.5, -0.1, -2.0],
        )
        .unwrap();
        let config = StackingConfig::default();
        let out = solve_stacking_weights(log_density.view(), config).unwrap();
        assert!(out.certificate.residual() <= config.kkt_tol);
        let n = log_density.nrows();
        for k in 0..2 {
            let mut g = 0.0_f64;
            for i in 0..n {
                let mix: f64 = (0..2)
                    .map(|c| out.weights[c] * log_density[[i, c]].exp())
                    .sum();
                g += log_density[[i, k]].exp() / mix;
            }
            g /= n as f64;
            assert!(
                g <= 1.0 + config.kkt_tol,
                "stationarity violated for candidate {k}: g = {g}"
            );
            assert!(
                out.weights[k] * (g - 1.0).abs() <= config.kkt_tol * (1.0 + 1e-6),
                "complementary slackness violated for candidate {k}: w = {}, g = {g}",
                out.weights[k]
            );
        }
    }

    #[test]
    fn stacking_exhaustion_without_certificate_is_an_error_not_weights() {
        let log_density = Array2::from_shape_vec(
            (6, 3),
            vec![
                0.0, -2.0, -4.0, -0.4, -0.1, -3.0, -2.0, 0.0, -0.3, -3.0, -1.0, 0.0, -0.2, -2.0,
                -0.5, -1.0, -0.3, -2.0,
            ],
        )
        .unwrap();
        let config = StackingConfig {
            max_iter: 1,
            ..StackingConfig::default()
        };
        let err = solve_stacking_weights(log_density.view(), config).unwrap_err();
        let checkpoint = match err {
            StackingError::DidNotConverge {
                certificate,
                checkpoint,
                ..
            } => {
                assert!(certificate.residual() > config.kkt_tol);
                assert_eq!(checkpoint.completed_iterations, 1);
                checkpoint
            }
            other => panic!("expected typed stacking exhaustion, got {other}"),
        };
        let encoded = serde_json::to_string(&checkpoint).unwrap();
        let checkpoint: StackingCheckpoint = serde_json::from_str(&encoded).unwrap();
        let mut other_density = log_density.clone();
        other_density[[0, 0]] += 0.25;
        assert!(matches!(
            resume_stacking_weights(other_density.view(), StackingConfig::default(), &checkpoint,),
            Err(StackingError::InvalidInput { .. })
        ));
        let resumed =
            resume_stacking_weights(log_density.view(), StackingConfig::default(), &checkpoint)
                .unwrap();
        let uninterrupted =
            solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        for (resumed, uninterrupted) in resumed.weights.iter().zip(uninterrupted.weights.iter()) {
            assert!((resumed - uninterrupted).abs() <= 1.0e-10);
        }
    }

    #[test]
    fn stacking_near_tied_boundary_uses_newton_not_millions_of_em_steps() {
        let log_density =
            Array2::from_shape_fn(
                (64, 2),
                |(_, candidate)| {
                    if candidate == 0 { 0.0 } else { -1.0e-6 }
                },
            );
        let out = solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        assert!(out.weights[0] >= 1.0 - StackingConfig::default().kkt_tol);
        assert!(out.iterations < 8, "iterations = {}", out.iterations);
    }

    #[test]
    fn stacking_dead_candidate_column_gets_zero_weight() {
        let log_density = Array2::from_shape_vec(
            (3, 2),
            vec![
                -1.0,
                f64::NEG_INFINITY,
                -2.0,
                f64::NEG_INFINITY,
                -0.5,
                f64::NEG_INFINITY,
            ],
        )
        .unwrap();
        let out = solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        assert_eq!(out.weights[1], 0.0);
        assert!((out.weights[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn stacking_rejects_invalid_and_unscorable_rows() {
        let log_density = Array2::from_shape_vec(
            (3, 2),
            vec![-1.0, -2.0, f64::NAN, f64::NEG_INFINITY, -2.0, -1.0],
        )
        .unwrap();
        assert!(matches!(
            solve_stacking_weights(log_density.view(), StackingConfig::default()),
            Err(StackingError::InvalidInput { .. })
        ));
        let unscorable = Array2::from_shape_vec(
            (2, 2),
            vec![-1.0, -2.0, f64::NEG_INFINITY, f64::NEG_INFINITY],
        )
        .unwrap();
        assert!(matches!(
            solve_stacking_weights(unscorable.view(), StackingConfig::default()),
            Err(StackingError::InvalidInput { .. })
        ));
    }

    fn two_cluster_mixture_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (12, 1),
            vec![
                -2.2, -2.0, -1.9, -2.1, -1.8, -2.05, 1.8, 2.0, 2.2, 1.9, 2.1, 2.05,
            ],
        )
        .unwrap()
    }

    #[test]
    fn gaussian_mixture_monotonicity_resolves_composite_map_noise_2264() {
        let objective_scale = 1.0;
        let composite_resolution = f64::EPSILON.sqrt() * objective_scale;
        let uncertainty = gaussian_mixture_monotonicity_uncertainty(objective_scale, 0.0, 0.0);
        assert_eq!(uncertainty, composite_resolution);

        let noise_scale_decrease = -0.5 * composite_resolution;
        assert!(noise_scale_decrease >= -uncertainty);
        let resolved_decrease = -2.0 * composite_resolution;
        assert!(resolved_decrease < -uncertainty);

        let larger_reduction_bound = 2.0 * composite_resolution;
        assert_eq!(
            gaussian_mixture_monotonicity_uncertainty(objective_scale, larger_reduction_bound, 0.0),
            larger_reduction_bound,
        );
    }

    #[test]
    fn gaussian_mixture_issue_scale_negative_step_is_within_computed_uncertainty_2264() {
        // Recorded issue mechanism: a signed mean-log-likelihood step of
        // -1.4e-13 was rejected even though it is unresolved at the scale of
        // the composite EM map. The admissible decrease below is derived only
        // from that map's observed objective scale and arithmetic reduction
        // bounds; the recorded step is an input to the decision, not a new
        // tolerance.
        let objective_scale = 1.0;
        let recorded_step = -1.4e-13;
        let uncertainty = gaussian_mixture_monotonicity_uncertainty(objective_scale, 0.0, 0.0);
        let certificate = GaussianMixtureCertificate {
            mean_log_likelihood: -objective_scale,
            mean_log_likelihood_gain: recorded_step,
            monotonicity_uncertainty: uncertainty,
            objective_residual: recorded_step.abs() / objective_scale,
            objective_tolerance: f64::EPSILON.sqrt(),
            parameter_residual: 0.0,
            parameter_tolerance: f64::EPSILON.sqrt(),
        };

        assert_eq!(
            certificate.monotonicity_uncertainty,
            f64::EPSILON.sqrt() * objective_scale,
            "reported uncertainty must be the computed composite-map resolution"
        );
        assert!(
            certificate.mean_log_likelihood_gain >= -certificate.monotonicity_uncertainty,
            "the recorded noise-scale decrease must not be a monotonicity violation"
        );
    }

    #[test]
    fn gaussian_mixture_below_roundoff_positive_gain_can_certify_2264() {
        // Recorded issue mechanism: a +6.6e-15 gain with a 1.5e-14 reduction
        // bound exhausted instead of certifying. A below-resolution gain is a
        // valid objective fixed point only when the independent parameter-map
        // residual also clears its configured tolerance.
        let objective_scale = 1.0;
        let recorded_gain = 6.6e-15;
        let recorded_reduction_bound = 1.5e-14;
        let objective_tolerance = f64::EPSILON.sqrt();
        let parameter_tolerance = f64::EPSILON.sqrt();
        let uncertainty = gaussian_mixture_monotonicity_uncertainty(
            objective_scale,
            recorded_reduction_bound,
            0.0,
        );
        let certificate = GaussianMixtureCertificate {
            mean_log_likelihood: -objective_scale,
            mean_log_likelihood_gain: recorded_gain,
            monotonicity_uncertainty: uncertainty,
            objective_residual: recorded_gain / objective_scale,
            objective_tolerance,
            parameter_residual: 0.5 * parameter_tolerance,
            parameter_tolerance,
        };

        assert_eq!(
            certificate.monotonicity_uncertainty,
            (f64::EPSILON.sqrt() * objective_scale).max(recorded_reduction_bound),
            "reported uncertainty must come from the composite-map and reduction bounds"
        );
        assert!(certificate.mean_log_likelihood_gain >= -certificate.monotonicity_uncertainty);
        assert!(certificate.objective_residual <= certificate.objective_tolerance);
        assert!(certificate.parameter_residual <= certificate.parameter_tolerance);
    }

    #[test]
    fn gaussian_mixture_fit_certificate_describes_the_exact_returned_iterate() {
        let data = two_cluster_mixture_data();
        let config = GaussianMixtureConfig::default();
        let fit = fit_gaussian_mixture(data.view(), 2, config).unwrap();
        let certificate = fit.certificate();
        assert!(certificate.objective_residual <= certificate.objective_tolerance);
        assert!(certificate.parameter_residual <= certificate.parameter_tolerance);

        let checkpoint = GaussianMixtureCheckpoint {
            weights: fit.weights.clone(),
            means: fit.means.clone(),
            covariances: fit.covariances.clone(),
            mean_log_likelihood: certificate.mean_log_likelihood,
            completed_iterations: fit.iterations,
            data_fingerprint: mixture_data_fingerprint(data.view()),
            covariance_floor: config.covariance_floor,
        };
        let current = mixture_e_step(
            data.view(),
            &checkpoint.weights,
            &checkpoint.means,
            &checkpoint.covariances,
        )
        .unwrap();
        let (weights, means, covariances) = mixture_m_step(
            data.view(),
            current.responsibilities.view(),
            config.covariance_floor,
        )
        .unwrap();
        let residual = mixture_parameter_residual(
            &checkpoint.weights,
            &checkpoint.means,
            &checkpoint.covariances,
            &weights,
            &means,
            &covariances,
        );
        let next = mixture_e_step(data.view(), &weights, &means, &covariances).unwrap();
        assert!(residual <= config.parameter_tol);
        assert_eq!(certificate.mean_log_likelihood, current.mean_log_likelihood);
        assert_eq!(
            certificate.mean_log_likelihood_gain,
            next.mean_log_likelihood - current.mean_log_likelihood
        );
        assert_eq!(
            certificate.monotonicity_uncertainty,
            gaussian_mixture_monotonicity_uncertainty(
                current
                    .mean_log_likelihood
                    .abs()
                    .max(next.mean_log_likelihood.abs())
                    .max(1.0),
                current.mean_log_likelihood_roundoff,
                next.mean_log_likelihood_roundoff,
            )
        );
        assert_eq!(certificate.parameter_residual, residual);
        assert!(
            (next.mean_log_likelihood - current.mean_log_likelihood).abs()
                / current
                    .mean_log_likelihood
                    .abs()
                    .max(next.mean_log_likelihood.abs())
                    .max(1.0)
                <= config.loglik_tol
        );
    }

    #[test]
    fn gaussian_mixture_exhaustion_is_typed_and_resumable() {
        let data = two_cluster_mixture_data();
        let short = GaussianMixtureConfig {
            max_iter: 1,
            ..GaussianMixtureConfig::default()
        };
        let err = fit_gaussian_mixture(data.view(), 2, short).unwrap_err();
        let checkpoint = match err {
            GaussianMixtureError::DidNotConverge {
                certificate,
                checkpoint,
                ..
            } => {
                assert!(
                    certificate.objective_residual > short.loglik_tol
                        || certificate.parameter_residual > short.parameter_tol
                );
                assert_eq!(checkpoint.completed_iterations, 1);
                let at_checkpoint = mixture_e_step(
                    data.view(),
                    &checkpoint.weights,
                    &checkpoint.means,
                    &checkpoint.covariances,
                )
                .unwrap();
                assert_eq!(
                    certificate.mean_log_likelihood, at_checkpoint.mean_log_likelihood,
                    "exhaustion evidence and checkpoint must describe one iterate"
                );
                checkpoint
            }
            other => panic!("expected typed EM exhaustion, got {other}"),
        };
        let encoded = serde_json::to_string(&checkpoint).unwrap();
        let checkpoint: GaussianMixtureCheckpoint = serde_json::from_str(&encoded).unwrap();
        let mut other_data = data.clone();
        other_data[[0, 0]] += 0.01;
        assert!(matches!(
            resume_gaussian_mixture(
                other_data.view(),
                GaussianMixtureConfig::default(),
                checkpoint.clone(),
            ),
            Err(GaussianMixtureError::InvalidInput { .. })
        ));
        let resumed =
            resume_gaussian_mixture(data.view(), GaussianMixtureConfig::default(), checkpoint)
                .unwrap();
        let uninterrupted =
            fit_gaussian_mixture(data.view(), 2, GaussianMixtureConfig::default()).unwrap();
        for (resumed, uninterrupted) in resumed.weights.iter().zip(uninterrupted.weights.iter()) {
            assert!((resumed - uninterrupted).abs() <= 1.0e-10);
        }

        assert!(resumed.bic().is_finite());
    }

    #[test]
    fn gaussian_mixture_bic_is_finite_with_an_active_covariance_floor() {
        // Component zero is exactly one-dimensional: its x coordinate never
        // changes, so the constrained MLE has one covariance eigenvalue at the
        // configured floor. The old BHHH determinant had identically-zero
        // mean-x and covariance-xy score columns and therefore rejected this
        // perfectly valid constrained predictive density as non-SPD.
        let per_cluster = 45usize;
        let mut data = Array2::<f64>::zeros((2 * per_cluster, 2));
        for sample in 0..per_cluster {
            let phase = std::f64::consts::TAU * sample as f64 / per_cluster as f64;
            data[[2 * sample, 0]] = -2.0;
            data[[2 * sample, 1]] = 0.08 * phase.sin();
            data[[2 * sample + 1, 0]] = 2.0 + 0.12 * phase.cos();
            data[[2 * sample + 1, 1]] = 0.08 * phase.sin();
        }
        let fit = fit_gaussian_mixture(data.view(), 2, GaussianMixtureConfig::default())
            .expect("the covariance floor defines a valid constrained mixture fit");
        let bic = fit.bic();
        assert!(bic.is_finite());
        assert_eq!(
            bic,
            -fit.loglik + 0.5 * fit.num_free_parameters() as f64 * (data.nrows() as f64).ln()
        );
    }

    fn seven_clusters_on_a_circle_2262() -> Array2<f64> {
        let clusters = 7usize;
        let per_cluster = 32usize;
        let mut data = Array2::<f64>::zeros((clusters * per_cluster, 2));
        for cluster in 0..clusters {
            let angle = std::f64::consts::TAU * cluster as f64 / clusters as f64;
            let (sin_angle, cos_angle) = angle.sin_cos();
            for sample in 0..per_cluster {
                let phase = std::f64::consts::TAU * sample as f64 / per_cluster as f64;
                // Vary the within-cluster radius while preserving its angular
                // symmetry. A literal constant-radius micro-circle makes the
                // Gaussian scale score identically zero and its empirical
                // Fisher singular, which is not a Gaussian-cluster fixture.
                let local_radius = 0.035 * (1.0 + 0.3 * (3.0 * phase).cos());
                let radial_noise = local_radius * phase.cos();
                let tangent_noise = local_radius * phase.sin();
                let radius = 2.0 + radial_noise;
                let row = cluster * per_cluster + sample;
                data[[row, 0]] = 0.4 + radius * cos_angle - tangent_noise * sin_angle;
                data[[row, 1]] = -0.3 + radius * sin_angle + tangent_noise * cos_angle;
            }
        }
        data
    }

    fn two_noisy_circles_for_union() -> Array2<f64> {
        let rows_per_circle = 96usize;
        let mut data = Array2::<f64>::zeros((2 * rows_per_circle, 2));
        for (circle, (center, radius)) in [([-4.0_f64, 0.3_f64], 1.2_f64), ([4.0, -0.2], 0.9)]
            .into_iter()
            .enumerate()
        {
            for sample in 0..rows_per_circle {
                let angle = std::f64::consts::TAU * sample as f64 / rows_per_circle as f64;
                let noisy_radius =
                    radius + 0.045 * (3.0 * angle).cos() + 0.018 * (5.0 * angle).sin();
                let row = circle * rows_per_circle + sample;
                data[[row, 0]] = center[0] + noisy_radius * angle.cos();
                data[[row, 1]] = center[1] + noisy_radius * angle.sin();
            }
        }
        data
    }

    #[test]
    fn circular_gaussian_density_avoids_extreme_scale_intermediate_overflow() {
        let noise_variance = f64::MAX / 2.0;
        let fit =
            CircularGaussianFit2d::from_parameters([0.0, 0.0], 1.1e154, noise_variance).unwrap();
        // Both `2πs` and `Rr` overflow if formed directly, although their log
        // and the ratio `Rr/s` are representable.
        let center_log_density = fit.log_density(0.0, 0.0);
        let off_center_log_density = fit.log_density(1.7e154, 0.0);
        assert!(center_log_density.is_finite());
        assert!(off_center_log_density.is_finite());
        let expected_center = -std::f64::consts::TAU.ln()
            - noise_variance.ln()
            - 0.5 * (fit.radius() / noise_variance.sqrt()).powi(2);
        assert_eq!(center_log_density, expected_center);
    }

    #[test]
    fn union_circles_use_the_shared_normalized_cartesian_density() {
        let data = two_noisy_circles_for_union();
        let config = GaussianMixtureConfig::default();
        let density_fit =
            fit_union_density(data.view(), UnionStructure::CircleCircle, config).unwrap();
        let union = fit_union_structure(data.view(), UnionStructure::CircleCircle, config).unwrap();
        assert_eq!(
            union.total_parameters,
            2 * CircularGaussianFit2d::NUM_FREE_PARAMETERS + 1
        );
        let component_weight_sum: f64 = union
            .components
            .iter()
            .map(|component| component.mixing_weight)
            .sum();
        assert!((component_weight_sum - 1.0).abs() <= 8.0 * f64::EPSILON);

        let mut fitted_centers = Array2::<f64>::zeros((density_fit.components.len(), 2));
        for (index, component) in density_fit.components.iter().enumerate() {
            let UnionDensityModel::Circle(fit) = &component.model else {
                panic!("circle+circle union produced a non-circle density");
            };
            let center = fit.center();
            fitted_centers[[index, 0]] = center[0];
            fitted_centers[[index, 1]] = center[1];
            let at_center = fit.log_density(center[0], center[1]);
            let expected = -std::f64::consts::TAU.ln()
                - fit.noise_variance().ln()
                - 0.5 * (fit.radius() / fit.noise_variance().sqrt()).powi(2);
            assert!(at_center.is_finite());
            assert!((at_center - expected).abs() < 1.0e-12 * (1.0 + expected.abs()));
        }

        let training_log_density = union_per_point_log_density(
            data.view(),
            data.view(),
            UnionStructure::CircleCircle,
            config,
        )
        .unwrap();
        let direct_log_likelihood = pairwise_sum(
            training_log_density
                .as_slice()
                .expect("owned score vector is contiguous"),
        );
        assert!(
            (union.log_likelihood - direct_log_likelihood).abs()
                <= 1.0e-12 * (1.0 + direct_log_likelihood.abs())
        );
        let expected_bic = -direct_log_likelihood
            + 0.5 * union.total_parameters as f64 * (data.nrows() as f64).ln();
        assert!((union.bic - expected_bic).abs() <= 1.0e-12 * (1.0 + expected_bic.abs()));

        let held_out = union_per_point_log_density(
            data.view(),
            fitted_centers.view(),
            UnionStructure::CircleCircle,
            config,
        )
        .unwrap();
        assert!(held_out.iter().all(|value| value.is_finite()));
    }

    fn circle_and_point_union_data() -> (Array2<f64>, Vec<Vec<usize>>) {
        let circle_rows = 32usize;
        let point_rows = 12usize;
        let mut data = Array2::<f64>::zeros((circle_rows + point_rows, 2));
        for row in 0..circle_rows {
            let angle = std::f64::consts::TAU * row as f64 / circle_rows as f64;
            let radius = 1.0 + 0.025 * (3.0 * angle).cos();
            data[[row, 0]] = -4.0 + radius * angle.cos();
            data[[row, 1]] = 0.2 + radius * angle.sin();
        }
        for offset in 0..point_rows {
            let phase = offset as f64;
            let row = circle_rows + offset;
            data[[row, 0]] = 4.0 + 0.055 * (1.7 * phase).cos() + 0.018 * (0.4 * phase).sin();
            data[[row, 1]] = -0.3 + 0.052 * (1.3 * phase).sin() - 0.015 * (0.9 * phase).cos();
        }
        (
            data,
            vec![
                (0..circle_rows).collect(),
                (circle_rows..circle_rows + point_rows).collect(),
            ],
        )
    }

    #[test]
    fn heterogeneous_union_role_assignment_is_group_label_invariant() {
        let (data, groups) = circle_and_point_union_data();
        let config = GaussianMixtureConfig::default();
        let forward = fit_union_density_from_groups(
            data.view(),
            UnionStructure::CirclePointCluster,
            &groups,
            config,
        )
        .unwrap();
        let reversed_groups = vec![groups[1].clone(), groups[0].clone()];
        let reversed = fit_union_density_from_groups(
            data.view(),
            UnionStructure::CirclePointCluster,
            &reversed_groups,
            config,
        )
        .unwrap();

        assert_eq!(forward.components[0].kind, UnionComponentKind::Circle);
        assert_eq!(forward.components[1].kind, UnionComponentKind::PointCluster);
        assert_eq!(
            reversed.components[0].kind,
            UnionComponentKind::PointCluster
        );
        assert_eq!(reversed.components[1].kind, UnionComponentKind::Circle);
        assert_eq!(forward.total_parameters, 4 + 3 + 1);
        assert_eq!(reversed.total_parameters, forward.total_parameters);
        assert!(
            (forward.log_likelihood - reversed.log_likelihood).abs()
                <= 1.0e-12 * (1.0 + forward.log_likelihood.abs())
        );
        assert!((forward.bic - reversed.bic).abs() <= 1.0e-12 * (1.0 + forward.bic.abs()));
    }

    #[test]
    fn point_cluster_is_isotropic_and_line_remains_full_covariance() {
        let (mut data, mut groups) = circle_and_point_union_data();
        // Replace the first group by a narrow, genuinely anisotropic line so
        // the line role is identifiable without changing the point group.
        for row in 0..groups[0].len() {
            let coordinate = (row as f64 - 15.5) / 4.0;
            data[[row, 0]] = -4.0 + coordinate;
            data[[row, 1]] = 0.2 + 0.018 * coordinate + 0.006 * (1.9 * row as f64).sin();
        }
        let fit = fit_union_density_from_groups(
            data.view(),
            UnionStructure::LineCluster,
            &groups,
            GaussianMixtureConfig::default(),
        )
        .unwrap();
        assert_eq!(fit.components[0].kind, UnionComponentKind::Line);
        assert_eq!(fit.components[0].num_parameters, 5);
        assert_eq!(fit.components[1].kind, UnionComponentKind::PointCluster);
        assert_eq!(fit.components[1].num_parameters, 3);
        assert_eq!(fit.total_parameters, 5 + 3 + 1);

        let UnionDensityModel::Gaussian(point) = &fit.components[1].model else {
            panic!("point cluster did not produce a Gaussian density");
        };
        assert_eq!(point.precision[[0, 1]], 0.0);
        assert_eq!(point.precision[[1, 0]], 0.0);
        assert_eq!(point.precision[[0, 0]], point.precision[[1, 1]]);
        let total_weight: f64 = fit
            .components
            .iter()
            .map(|component| component.mixing_weight)
            .sum();
        assert!((total_weight - 1.0).abs() <= 8.0 * f64::EPSILON);

        // Reversing group labels must reverse the selected roles, not the
        // fitted unlabeled mixture density or its common-scale score.
        groups.reverse();
        let reversed = fit_union_density_from_groups(
            data.view(),
            UnionStructure::LineCluster,
            &groups,
            GaussianMixtureConfig::default(),
        )
        .unwrap();
        assert_eq!(
            reversed.components[0].kind,
            UnionComponentKind::PointCluster
        );
        assert_eq!(reversed.components[1].kind, UnionComponentKind::Line);
        assert!((fit.bic - reversed.bic).abs() <= 1.0e-12 * (1.0 + fit.bic.abs()));
    }

    #[test]
    fn isotropic_union_density_uses_the_same_fractional_mean_chart_as_its_mle() {
        let translated = ndarray::array![[1.0e16], [1.0e16 + 2.0], [1.0e16 + 2.0]];
        let fit = fit_isotropic_gaussian_component(translated.view(), 1.0e-12).unwrap();
        let variance = fit.precision[[0, 0]].recip();
        assert!((variance - 8.0 / 9.0).abs() <= 32.0 * f64::EPSILON);

        let residuals = [-4.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
        let expected_log_norm = -0.5 * ((2.0 * std::f64::consts::PI).ln() + variance.ln());
        for (row, residual) in residuals.into_iter().enumerate() {
            let expected = expected_log_norm - 0.5 * residual * residual / variance;
            let actual = fit.log_density(translated.row(row));
            assert!(
                (actual - expected).abs() <= 32.0 * f64::EPSILON * (1.0 + expected.abs()),
                "row {row}: density chart disagrees with fitted MLE residual: actual={actual}, expected={expected}"
            );
        }

        let subnormal = f64::from_bits(1);
        let constant = ndarray::array![[subnormal], [subnormal], [subnormal]];
        let constant_fit = fit_isotropic_gaussian_component(constant.view(), 1.0).unwrap();
        assert_eq!(constant_fit.residual(constant.row(0)), vec![0.0]);
        assert_eq!(
            constant_fit.log_density(constant.row(0)),
            constant_fit.log_norm
        );
    }

    #[test]
    fn union_ladder_fails_closed_when_one_declared_structure_fails() {
        let mut data = Array2::<f64>::zeros((8, 2));
        for row in 0..5 {
            let angle = std::f64::consts::TAU * row as f64 / 5.0;
            data[[row, 0]] = -5.0 + angle.cos();
            data[[row, 1]] = angle.sin();
        }
        data[[5, 0]] = 5.00;
        data[[5, 1]] = 0.00;
        data[[6, 0]] = 5.08;
        data[[6, 1]] = 0.02;
        data[[7, 0]] = 4.97;
        data[[7, 1]] = 0.07;

        let error = fit_union_ladder(data.view(), GaussianMixtureConfig::default()).unwrap_err();
        assert!(error.contains("every declared structure must fit"));
        assert!(error.contains(UnionStructure::CircleCircle.as_str()));
        assert!(error.contains("needs at least 5 rows"));
    }

    #[test]
    fn ring_of_clusters_fit_is_stationary_and_complexity_priced_2262() {
        let data = seven_clusters_on_a_circle_2262();
        let config = GaussianMixtureConfig::default();
        let fit = fit_ring_gaussian_mixture(data.view(), 7, config).unwrap();
        let certificate = fit.certificate();
        assert!(certificate.objective_residual <= certificate.objective_tolerance);
        assert!(certificate.parameter_residual <= certificate.parameter_tolerance);
        assert_eq!(fit.num_free_parameters(), 17);
        assert!((fit.center()[0] - 0.4).abs() < 0.05);
        assert!((fit.center()[1] + 0.3).abs() < 0.05);
        assert!((fit.radius() - 2.0).abs() < 0.05);
        assert!(fit.variance().is_finite() && fit.variance() > 0.0);
        assert!(
            fit.per_point_log_density(data.view())
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(fit.bic().is_finite());

        let free = fit_gaussian_mixture(data.view(), 7, config).unwrap();
        assert_eq!(free.num_free_parameters(), 41);
        assert!(fit.num_free_parameters() < free.num_free_parameters());
    }

    #[test]
    fn ring_certificate_uses_identifiable_component_means() {
        // The points (.3, ±sqrt(.91)) lie on both unit circles centered at
        // (0, 0) and (.6, 0). Repeating one point gives three labelled
        // components. Thus center and directions move by O(1) while every
        // component mean—and therefore the represented mixture density—is
        // bit-identical.
        let y = 0.91_f64.sqrt();
        let weights = Array1::from_vec(vec![0.2, 0.3, 0.5]);
        let previous = RingMixtureState {
            weights: weights.clone(),
            center: Array1::from_vec(vec![0.0, 0.0]),
            radius: 1.0,
            directions: Array2::from_shape_vec((3, 2), vec![0.3, y, 0.3, -y, 0.3, y]).unwrap(),
            variance: 0.25,
            mean_log_likelihood: -1.0,
            completed_iterations: 10,
        };
        let next = RingMixtureState {
            weights,
            center: Array1::from_vec(vec![0.6, 0.0]),
            radius: 1.0,
            directions: Array2::from_shape_vec((3, 2), vec![-0.3, y, -0.3, -y, -0.3, y]).unwrap(),
            variance: 0.25,
            mean_log_likelihood: -1.0,
            completed_iterations: 11,
        };
        assert!(relative_parameter_step(previous.center[0], next.center[0]) > 0.5);
        assert_eq!(ring_identifiable_parameter_residual(&previous, &next), 0.0);
    }

    #[test]
    fn stacked_mean_is_weighted_combination() {
        let weights = Array1::from_vec(vec![0.25, 0.75]);
        let means = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![5.0, 6.0, 7.0]),
        ];
        let out = stacked_predictive_mean(&weights, &means).unwrap();
        assert!((out[0] - (0.25 * 1.0 + 0.75 * 5.0)).abs() < 1e-12);
        assert!((out[2] - (0.25 * 3.0 + 0.75 * 7.0)).abs() < 1e-12);
    }

    #[test]
    fn stacked_mean_rejects_shape_mismatch() {
        let weights = Array1::from_vec(vec![0.5, 0.5]);
        let means = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![3.0]),
        ];
        assert!(stacked_predictive_mean(&weights, &means).is_err());
    }

    // -----------------------------------------------------------------------
    // #1026 hybrid curved + linear-tail split-selection
    // -----------------------------------------------------------------------

    /// Build the two candidate parameterizations for one atom slot the way the
    /// fit would: the linear special case (one decoder direction, `Θ = 0`,
    /// `P_linear` params) and the curved candidate (`latent_dim` ≥ 1, more
    /// params, fitted turning `theta`). The curved candidate's likelihood is the
    /// linear likelihood MINUS `curved_loglik_gain` of NLE (curvature it captures
    /// the secant cannot), so the nesting invariant `curved_loglik ≥ linear` is
    /// honored: a straight feature has zero gain, a turning feature a positive
    /// gain that grows with Θ. The rank-aware Laplace normalizer charges the
    /// extra `½(P_curved − P_linear)·log(2π)` for the curved parameters, so the
    /// evidence comparison is the real `Θ/√ε` crossover.
    fn hybrid_slot(
        linear_nle: f64,
        p_linear: usize,
        latent_dim: usize,
        p_curved: usize,
        theta: f64,
        curved_loglik_gain: f64,
    ) -> Vec<HybridAtomCandidate> {
        let param_price =
            0.5 * (p_curved as f64 - p_linear as f64) * (2.0 * std::f64::consts::PI).ln();
        let curved_nle = linear_nle - curved_loglik_gain + param_price;
        vec![
            HybridAtomCandidate::linear(linear_nle, p_linear),
            HybridAtomCandidate::curved(latent_dim, curved_nle, p_curved, Some(theta)),
        ]
    }

    #[test]
    fn hybrid_dominance_floor_selects_linear_when_turning_is_zero() {
        // A perfectly straight curved fit (Θ = 0) gains no likelihood over its
        // linear sub-model but pays more parameters → linear must win, by
        // construction, even if finite-sample evidence noise nudged the curved
        // NLE slightly below linear.
        let slot = hybrid_slot(100.0, 2, 1, 5, 0.0, 0.0);
        let choice = select_hybrid_atom(&slot).unwrap();
        assert!(choice.param.is_linear());
        assert_eq!(choice.param, HybridAtomParam::Linear);
        // The exact-zero guard fires regardless of the evidence margin sign.
        assert!(choice.curved_turning.unwrap() <= HYBRID_LINEAR_TURNING_FLOOR);
    }

    #[test]
    fn hybrid_selects_curved_when_turning_pays_for_itself() {
        // A genuinely turning feature (Θ = 2π, a full loop): the curved fit
        // captures enough curvature that, even charged the extra-parameter price,
        // its NLE drops below the linear secant's → curved wins.
        let slot = hybrid_slot(100.0, 2, 1, 5, 2.0 * std::f64::consts::PI, 30.0);
        let choice = select_hybrid_atom(&slot).unwrap();
        assert_eq!(choice.param, HybridAtomParam::Curved { latent_dim: 1 });
        // The curved fit won a strictly positive evidence margin.
        assert!(choice.curved_evidence_margin > 0.0);
    }

    #[test]
    fn hybrid_keeps_linear_when_curvature_doesnt_pay_its_price() {
        // A barely-curved feature (small Θ): the curved fit recovers only a sliver
        // of likelihood, not enough to cover the extra-parameter price → the
        // dominance floor keeps the linear tail.
        let slot = hybrid_slot(100.0, 2, 1, 5, 0.05, 0.1);
        let choice = select_hybrid_atom(&slot).unwrap();
        assert!(choice.param.is_linear());
        assert!(choice.curved_evidence_margin <= 0.0);
    }

    #[test]
    fn hybrid_tie_breaks_to_the_cheaper_linear_atom() {
        // Exact NLE tie (above the turning floor so the evidence path decides):
        // the cheaper linear atom wins, preserving strict generalization — the
        // hybrid never pays for curvature it does not need.
        let theta = 0.5; // above the floor → evidence path, not the exact guard
        let nle = 42.0;
        let slot = vec![
            HybridAtomCandidate::linear(nle, 2),
            HybridAtomCandidate::curved(1, nle, 5, Some(theta)),
        ];
        let choice = select_hybrid_atom(&slot).unwrap();
        assert!(choice.param.is_linear());
        assert_eq!(choice.num_parameters, 2);
    }

    #[test]
    fn hybrid_split_reduces_to_pure_linear_when_all_features_are_straight() {
        // Every slot's curved candidate has Θ → 0 (flat features everywhere): the
        // dominance floor fires at every slot → the hybrid recovers the pure-
        // linear dictionary exactly. This is the `all Θ → 0` limit (3).
        let slots: Vec<Vec<HybridAtomCandidate>> = (0..6)
            .map(|i| hybrid_slot(50.0 + i as f64, 2, 1, 5, 0.0, 0.0))
            .collect();
        let split = select_hybrid_split(&slots).unwrap();
        assert!(split.is_pure_linear());
        assert_eq!(split.curved_atom_count, 0);
        assert_eq!(split.linear_atom_count(), 6);
        // Summed NLE equals the pure-linear baseline (every slot chose linear).
        let pure_linear: f64 = (0..6).map(|i| 50.0 + i as f64).sum();
        assert!((split.total_negative_log_evidence - pure_linear).abs() < 1e-12);
    }

    #[test]
    fn hybrid_split_reduces_to_pure_curved_when_every_feature_curves() {
        // Every slot's feature turns enough (Θ = 2π, large likelihood gain) that
        // curved beats linear everywhere → the pure-curved limit (3).
        let slots: Vec<Vec<HybridAtomCandidate>> = (0..5)
            .map(|i| hybrid_slot(80.0 + i as f64, 2, 1, 5, 2.0 * std::f64::consts::PI, 40.0))
            .collect();
        let split = select_hybrid_split(&slots).unwrap();
        assert!(split.is_pure_curved());
        assert_eq!(split.curved_atom_count, 5);
        assert_eq!(split.linear_atom_count(), 0);
    }

    #[test]
    fn hybrid_split_on_mixed_dictionary_picks_curved_for_circles_linear_for_directions() {
        // Mixed synthetic: slots 0..3 are CIRCLE features (high turning Θ = 2π,
        // the curved fit captures the loop), slots 3..7 are LINEAR DIRECTIONS
        // (straight, Θ = 0). The evidence split must select curved for the
        // circles and linear for the directions — and the hybrid's summed
        // evidence must be ≤ the summed per-slot LINEAR-candidate NLE (each
        // slot's best straight line fit to its response residual). This is a
        // data-level match-or-beat dominance (#1202: linear is the curved
        // family's nested Θ = 0 sub-model on common data), and holds because each
        // slot picks the argmin of its two common-data candidates.
        let mut slots: Vec<Vec<HybridAtomCandidate>> = Vec::new();
        let mut pure_linear_baseline = 0.0_f64;
        // Three circle features: a curved atom replaces ~10-30 linear secants, so
        // the curved fit buys a large likelihood gain that dwarfs its param price.
        for i in 0..3 {
            let linear_nle = 120.0 + 3.0 * i as f64;
            pure_linear_baseline += linear_nle;
            slots.push(hybrid_slot(
                linear_nle,
                2,
                1,
                5,
                2.0 * std::f64::consts::PI,
                35.0,
            ));
        }
        // Four straight linear directions: zero turning, the linear special case
        // is optimal — a curved atom buys nothing and only costs parameters.
        for i in 0..4 {
            let linear_nle = 90.0 + 2.0 * i as f64;
            pure_linear_baseline += linear_nle;
            slots.push(hybrid_slot(linear_nle, 2, 1, 5, 0.0, 0.0));
        }

        let split = select_hybrid_split(&slots).unwrap();

        // The first three (circles) chose curved; the last four (directions) chose
        // linear.
        for (idx, choice) in split.atoms.iter().enumerate() {
            if idx < 3 {
                assert_eq!(
                    choice.param,
                    HybridAtomParam::Curved { latent_dim: 1 },
                    "circle slot {idx} should select curved"
                );
            } else {
                assert!(
                    choice.param.is_linear(),
                    "direction slot {idx} should select linear"
                );
            }
        }
        assert_eq!(split.curved_atom_count, 3);
        assert_eq!(split.linear_atom_count(), 4);

        // The hybrid's summed negative-log-evidence is ≤ the summed per-slot
        // LINEAR-candidate NLE (each slot's best straight line fit to its response
        // residual): the per-slot argmin can only lower the sum. This is a
        // data-level match-or-beat dominance (#1202): linear is the curved
        // family's nested Θ = 0 sub-model on common data.
        assert!(
            split.total_negative_log_evidence <= pure_linear_baseline + 1e-9,
            "hybrid NLE {} must be <= summed linear-candidate NLE {}",
            split.total_negative_log_evidence,
            pure_linear_baseline
        );
        // And strictly better, because the curved circle slots paid off.
        assert!(split.total_negative_log_evidence < pure_linear_baseline);
    }

    #[test]
    fn hybrid_split_rejects_empty_slot() {
        let slots = vec![hybrid_slot(10.0, 2, 1, 5, 0.0, 0.0), Vec::new()];
        assert!(select_hybrid_split(&slots).is_err());
    }

    // ── #1362: compare_models must Occam-penalise a pure-noise smooth ────────
    //
    // These tests pin the ranking contract directly on `compare_reml_fits` with
    // controlled (score, edf, log_lik) inputs taken from the actual #1362
    // reproduction (Rust `reml_score` of `y ~ s(x)` vs `y ~ s(x) + s(z)` at
    // n=700). They do not need a fitted GAM or a Python wheel.

    fn cand(name: &str, score: f64, edf: f64, log_lik: f64) -> RemlCandidate {
        RemlCandidate {
            index: 0,
            name: name.to_string(),
            score,
            edf: Some(edf),
            log_lik: Some(log_lik),
            family: None,
            n_obs: None,
        }
    }

    #[test]
    fn ranking_score_is_conditional_aic_when_loglik_and_edf_present() {
        // AIC = -2ℓ + 2·edf.
        let c = cand("m", /*score (ignored)*/ 999.0, 6.748, -32.0866);
        let expected = -2.0 * -32.0866 + 2.0 * 6.748;
        assert!((c.ranking_score() - expected).abs() < 1e-9);
    }

    #[test]
    fn ranking_score_falls_back_to_evidence_without_loglik() {
        let c = RemlCandidate {
            index: 0,
            name: "m".to_string(),
            score: 151.28,
            edf: Some(6.0),
            log_lik: None,
            family: None,
            n_obs: None,
        };
        assert_eq!(c.ranking_score(), 151.28);
    }

    #[test]
    fn compare_models_rejects_pure_noise_smooth_despite_lower_evidence() {
        // Seed-3000 numbers from the #1362 Rust reproduction:
        //   small (y ~ s(x)):      reml=180.526, edf=6.748,  loglik=-32.0866
        //   big   (y ~ s(x)+s(z)): reml=177.404, edf=14.250, loglik=-32.1212
        // The big (noise-augmented) model has the LOWER (apparently better) raw
        // REML evidence, yet it spends ~7.5 extra EDF fitting noise without
        // improving the likelihood. The winner must be the SMALL model.
        let small = cand("small", 180.526, 6.748, -32.0866);
        let big = cand("big", 177.404, 14.250, -32.1212);

        // Sanity: raw evidence (the broken headline) prefers big.
        assert!(big.score < small.score);

        let cmp = compare_reml_fits(vec![small, big]).expect("compare");
        assert_eq!(
            cmp.winner, "small",
            "compare_models must Occam-penalise the pure-noise smooth and pick the smaller model"
        );
        // The score table still reports the raw evidence headline unchanged, so
        // Model.evidence / bayes_factor_vs stay consistent with the table.
        let small_row = cmp
            .score_table
            .iter()
            .find(|r| r.name == "small")
            .expect("small row");
        let big_row = cmp
            .score_table
            .iter()
            .find(|r| r.name == "big")
            .expect("big row");
        assert!((small_row.reml_score - 180.526).abs() < 1e-9);
        assert!((big_row.reml_score - 177.404).abs() < 1e-9);
    }

    #[test]
    fn ranking_bayes_factor_is_akaike_evidence_ratio_not_its_square() {
        // Issue #2124: `ranking_score` is the conditional AIC (`−2ℓ + 2·edf`), a
        // −2·log / deviance-scale cost. For an AIC gap Δ the Akaike evidence ratio
        // (Burnham & Anderson) is `exp(−½Δ)`, so the winner-over-loser
        // `bayes_factor` must be `exp(½Δ)` — NOT `exp(Δ)`, which squares it.
        //
        // Winner: AIC 0 (loglik 0, edf 0). Loser: AIC = 27.68 (loglik −13.84,
        // edf 0), matching the ΔAIC in the issue repro. Raw REML scores are set
        // distinct (100 vs 110) to lock the scoping: the raw score_table path
        // must stay `exp(Δreml)` with NO halving.
        let delta_aic = 27.68_f64;
        let winner = cand("winner", 100.0, 0.0, 0.0);
        let loser = cand("loser", 110.0, 0.0, -delta_aic / 2.0);

        let cmp = compare_reml_fits(vec![winner, loser]).expect("compare");
        assert_eq!(cmp.winner, "winner");

        let loser_row = cmp
            .ranking
            .iter()
            .find(|r| r.name == "loser")
            .expect("loser ranking row");

        // The AIC gap FIELD stays on the AIC scale, unchanged (issue #2124).
        assert!((loser_row.delta - delta_aic).abs() < 1e-9);

        // The Bayes factor is the Akaike evidence ratio exp(½·ΔAIC) = exp(13.84)
        // ≈ 1.03e6 — NOT the squared exp(27.68) ≈ 1.05e12 the bug reported.
        let expected = (0.5 * delta_aic).exp();
        assert!(
            (loser_row.bayes_factor / expected - 1.0).abs() < 1e-9,
            "ranking bayes_factor {} should be exp(½ΔAIC)={}, not exp(ΔAIC)={}",
            loser_row.bayes_factor,
            expected,
            delta_aic.exp()
        );
        // Explicit anti-regression: it must not be the squared ratio.
        assert!(loser_row.bayes_factor < delta_aic.exp() * 0.5);

        // Scoping lock (issue #2124): the RAW-REML score_table path is untouched —
        // its best-over-model Bayes factor is `exp(Δreml)` with NO halving. Raw
        // scores 100 (winner) vs 110 (loser) give Δreml = 10, so the loser's raw
        // Bayes factor is exp(10), not exp(5).
        let loser_score_row = cmp
            .score_table
            .iter()
            .find(|r| r.name == "loser")
            .expect("loser score row");
        let expected_reml_bf = 10.0_f64.exp();
        assert!(
            (loser_score_row.bayes_factor_best_over_model / expected_reml_bf - 1.0).abs() < 1e-9,
            "raw-REML bayes_factor_best_over_model must stay exp(Δreml)=exp(10), got {}",
            loser_score_row.bayes_factor_best_over_model
        );
    }

    #[test]
    fn compare_models_keeps_power_for_a_relevant_smooth() {
        // Seed-3000 relevant-z numbers from the same reproduction:
        //   small: reml=1025.067, edf≈6.75,  loglik≈-368.99 (aic≈751.5)
        //   big:   reml=199.509,  edf≈14.25, loglik≈-33.16  (aic≈94.8)
        // A genuinely relevant smooth lowers BOTH the evidence and the AIC, so
        // the bigger model must still win — a fix cannot just always pick small.
        let small = cand("small", 1025.067, 6.75, -368.985);
        let big = cand("big", 199.509, 14.25, -33.165);
        let cmp = compare_reml_fits(vec![small, big]).expect("compare");
        assert_eq!(
            cmp.winner, "big",
            "compare_models must retain power: the relevant smooth's model must win"
        );
    }

    #[test]
    fn compare_models_rejects_mismatched_observation_counts() {
        // Two same-family fits on different-sized data are not comparable by
        // AIC / evidence; the comparison must fail loud, mirroring the family
        // guard, rather than declare a sample-size-driven winner.
        let with_n = |name: &str, n: usize| RemlCandidate {
            index: 0,
            name: name.to_string(),
            score: 100.0,
            edf: Some(5.0),
            log_lik: Some(-40.0),
            family: Some("gaussian".to_string()),
            n_obs: Some(n),
        };
        let err = compare_reml_fits(vec![with_n("big", 500), with_n("small", 100)])
            .expect_err("cross-n comparison must be rejected");
        assert!(
            err.contains("number of observations") && err.contains("500") && err.contains("100"),
            "n-guard error should name the incomparable counts, got: {err}"
        );

        // Same n is comparable.
        compare_reml_fits(vec![with_n("a", 250), with_n("b", 250)])
            .expect("same-n comparison must succeed");

        // A missing count (`None`) is unconstrained: it must not block a
        // comparison against a fit that does carry one (legacy / scan payloads).
        let without_n = RemlCandidate {
            index: 0,
            name: "legacy".to_string(),
            score: 90.0,
            edf: Some(4.0),
            log_lik: Some(-35.0),
            family: Some("gaussian".to_string()),
            n_obs: None,
        };
        compare_reml_fits(vec![with_n("counted", 500), without_n])
            .expect("an unconstrained (None) count must not trip the guard");
    }
}
