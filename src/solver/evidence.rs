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
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::linalg::faer_ndarray::FaerEigh;
use crate::linalg::lanczos::{
    SymmetricLanczosOptions, symmetric_lanczos_eigenpairs, symmetric_lanczos_log_quadrature,
};
use crate::linalg::triangular::cholesky_solve_vector;
use crate::solver::arrow_schur::{ArrowFactorCache, ArrowSchurSystem};
use crate::solver::priority_selection::{PriorityCandidate, rank_priority_candidates};

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
#[derive(Debug, Clone, Copy)]
pub struct StackingConfig {
    pub max_iter: usize,
    pub weight_tol: f64,
}

impl Default for StackingConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            weight_tol: 1e-10,
        }
    }
}

/// Simplex weights for retained topology candidates plus the achieved held-out
/// mean log-score.
#[derive(Debug, Clone)]
pub struct StackingWeights {
    pub weights: Array1<f64>,
    pub mean_log_score: f64,
    pub iterations: usize,
}

/// Solve the stacking-of-predictive-distributions weight problem from a
/// per-observation held-out log-density table `log_density[i, k] = log p_k(y_i)`.
///
/// This belongs on the evidence surface rather than in a separate solver: it is
/// the topology/evidence consumer that replaces winner-take-all only when the
/// caller has retained candidate fits and per-point held-out densities.
pub fn solve_stacking_weights(
    log_density: ArrayView2<'_, f64>,
    config: StackingConfig,
) -> Result<StackingWeights, String> {
    let n_obs = log_density.nrows();
    let n_cand = log_density.ncols();
    if n_cand == 0 {
        return Err("stacking requires at least one candidate column".to_string());
    }
    if n_obs == 0 {
        return Err("stacking requires at least one held-out observation row".to_string());
    }

    let kept_cols: Vec<usize> = (0..n_cand)
        .filter(|&k| (0..n_obs).any(|i| log_density[[i, k]].is_finite()))
        .collect();
    if kept_cols.is_empty() {
        return Err("stacking found no candidate with any finite held-out density".to_string());
    }
    let rows: Vec<usize> = (0..n_obs)
        .filter(|&i| kept_cols.iter().any(|&k| log_density[[i, k]].is_finite()))
        .collect();
    if rows.is_empty() {
        return Err("stacking found no held-out row with a finite density".to_string());
    }

    let kept = kept_cols.len();
    let mut weights = Array1::<f64>::from_elem(kept, 1.0 / kept as f64);
    let mut next = Array1::<f64>::zeros(kept);
    let mut iterations = 0usize;
    for _ in 0..config.max_iter {
        iterations += 1;
        next.fill(0.0);
        let mut active_rows = 0usize;
        for &row in &rows {
            let mut row_max = f64::NEG_INFINITY;
            for (local_col, &source_col) in kept_cols.iter().enumerate() {
                let log_p = log_density[[row, source_col]];
                if log_p.is_finite() && weights[local_col] > 0.0 {
                    row_max = row_max.max(weights[local_col].ln() + log_p);
                }
            }
            if !row_max.is_finite() {
                continue;
            }
            let mut denom = 0.0_f64;
            for (local_col, &source_col) in kept_cols.iter().enumerate() {
                let log_p = log_density[[row, source_col]];
                if log_p.is_finite() && weights[local_col] > 0.0 {
                    denom += (weights[local_col].ln() + log_p - row_max).exp();
                }
            }
            if denom <= 0.0 {
                continue;
            }
            active_rows += 1;
            let log_mix = row_max + denom.ln();
            for (local_col, &source_col) in kept_cols.iter().enumerate() {
                let log_p = log_density[[row, source_col]];
                if log_p.is_finite() && weights[local_col] > 0.0 {
                    next[local_col] += (weights[local_col].ln() + log_p - log_mix).exp();
                }
            }
        }
        if active_rows == 0 {
            break;
        }
        next.mapv_inplace(|value| value / active_rows as f64);
        let total = next.sum();
        if total > 0.0 {
            next.mapv_inplace(|value| value / total);
        }
        let delta = next
            .iter()
            .zip(weights.iter())
            .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()));
        weights.assign(&next);
        if delta <= config.weight_tol {
            break;
        }
    }

    let mean_log_score = stacking_mean_log_score(log_density, &rows, &kept_cols, weights.view());
    let mut full = Array1::<f64>::zeros(n_cand);
    for (local_col, &source_col) in kept_cols.iter().enumerate() {
        full[source_col] = weights[local_col];
    }
    Ok(StackingWeights {
        weights: full,
        mean_log_score,
        iterations,
    })
}

fn stacking_mean_log_score(
    log_density: ArrayView2<'_, f64>,
    rows: &[usize],
    kept_cols: &[usize],
    weights: ArrayView1<'_, f64>,
) -> f64 {
    let mut score_sum = 0.0_f64;
    let mut counted = 0usize;
    for &row in rows {
        let mut row_max = f64::NEG_INFINITY;
        for (local_col, &source_col) in kept_cols.iter().enumerate() {
            let log_p = log_density[[row, source_col]];
            if log_p.is_finite() && weights[local_col] > 0.0 {
                row_max = row_max.max(weights[local_col].ln() + log_p);
            }
        }
        if !row_max.is_finite() {
            continue;
        }
        let mut denom = 0.0_f64;
        for (local_col, &source_col) in kept_cols.iter().enumerate() {
            let log_p = log_density[[row, source_col]];
            if log_p.is_finite() && weights[local_col] > 0.0 {
                denom += (weights[local_col].ln() + log_p - row_max).exp();
            }
        }
        if denom > 0.0 {
            score_sum += row_max + denom.ln();
            counted += 1;
        }
    }
    if counted == 0 {
        f64::NEG_INFINITY
    } else {
        score_sum / counted as f64
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
// EM to a tolerance. It is priced by its free-parameter count and scored
// through the SAME rank-aware Laplace/Tierney-Kadane normalizer as the smooth
// topology candidates: `−V = loglik − ½ log|H| + ½ P log(2π)` with the
// `−½ (dim(H) − rank(S)) log(2π)` normalizer evaluated at `dim(H) = P`,
// `rank(S) = 0` (a fully likelihood-identified, unpenalized parametric model,
// so every free parameter is unpenalized null-space). The Hessian log-det
// `log|H|` is the observed (empirical-Fisher / BHHH) information
// `H = Σ_i s_i s_iᵀ`, the exact, finite, SPD observed-information surrogate at
// the EM optimum, fed through the same `laplace_evidence` entry point used by
// the smooth rungs so the two model classes are comparable on the evidence
// scale.

/// Convergence + ladder controls for the discrete-mixture rung. All fields are
/// fixed (no clock randomness, no env): deterministic seeding makes the fitted
/// mixture a pure function of the data and `k`.
#[derive(Debug, Clone, Copy)]
pub struct GaussianMixtureConfig {
    /// Maximum EM iterations.
    pub max_iter: usize,
    /// Relative mean-log-likelihood improvement tolerance for EM stopping.
    pub loglik_tol: f64,
    /// Ridge added to each component covariance for numerical SPD safety
    /// (variance floor). A small fixed value, not a tuned knob.
    pub covariance_floor: f64,
    /// Maximum iterations for the deterministic k-means seeding pass.
    pub kmeans_max_iter: usize,
}

impl Default for GaussianMixtureConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            loglik_tol: 1e-7,
            covariance_floor: 1e-6,
            kmeans_max_iter: 25,
        }
    }
}

/// A fitted `k`-component full-covariance Gaussian mixture.
#[derive(Debug, Clone)]
pub struct GaussianMixtureFit {
    /// Mixing weights, length `k`, on the simplex.
    pub weights: Array1<f64>,
    /// Component means, `k × d`.
    pub means: Array2<f64>,
    /// Component covariances, `k` matrices of shape `d × d` (SPD).
    pub covariances: Vec<Array2<f64>>,
    /// Number of mixture components.
    pub k: usize,
    /// Data dimension.
    pub d: usize,
    /// Number of rows used to fit.
    pub n_obs: usize,
    /// Maximised total log-likelihood `Σ_i log Σ_j w_j N(y_i; μ_j, Σ_j)`.
    pub loglik: f64,
    /// EM iterations taken.
    pub iterations: usize,
}

impl GaussianMixtureFit {
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
        let mut comp = vec![GaussianComponentEval::new(self.d); self.k];
        for j in 0..self.k {
            comp[j] = GaussianComponentEval::factor(self.means.row(j), &self.covariances[j])?;
        }
        let mut out = Array1::<f64>::zeros(n);
        let log_w: Vec<f64> = self
            .weights
            .iter()
            .map(|w| w.max(f64::MIN_POSITIVE).ln())
            .collect();
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

    /// Rank-aware Laplace **negative** log evidence on the SAME scale as the
    /// smooth topology rungs. `−V = loglik − ½ log|H| + ½ P log(2π)`, realised
    /// by calling [`laplace_evidence`] with `residual_objective = −loglik`,
    /// `penalty_log_det = 0`, `penalty_rank = 0`, `effective_dim = P`, and
    /// `log|H|` the observed empirical-Fisher information at the optimum.
    pub fn laplace_negative_log_evidence(&self, data: ArrayView2<'_, f64>) -> Result<f64, String> {
        let p = self.num_free_parameters();
        let information = self.empirical_fisher_information(data)?;
        if information.nrows() != p {
            return Err(format!(
                "mixture empirical-Fisher information has dim {} but expected free-parameter count {p}",
                information.nrows()
            ));
        }
        let apply_info = |x: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0_f64; p];
            for r in 0..p {
                let mut acc = 0.0_f64;
                for c in 0..p {
                    acc += information[[r, c]] * x[c];
                }
                out[r] = acc;
            }
            out
        };
        let hvp = EvidenceHvpLogDet {
            dim: p,
            apply: &apply_info,
        };
        let v = laplace_evidence(
            EvidenceLogDetSource::Hvp(hvp),
            0.0,
            -self.loglik,
            p as f64,
            0.0,
        );
        if !v.is_finite() {
            return Err("mixture Laplace evidence is not finite".to_string());
        }
        Ok(v)
    }

    /// Observed empirical-Fisher (BHHH) information `H = Σ_i s_i s_iᵀ`, where
    /// `s_i = ∇_θ log p(y_i)` is the per-observation score in the
    /// free-parameter coordinates: softmax-logit mixing weights (`k − 1`),
    /// component means (`k·d`), and the lower-triangular covariance entries
    /// (`k · d(d+1)/2`) of each component, in that block order. This SPD matrix
    /// is the genuine observed-information surrogate evaluated at the EM
    /// optimum — its dimension is exactly `P`, which is what enters the
    /// rank-aware normalizer.
    fn empirical_fisher_information(
        &self,
        data: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        if data.ncols() != self.d {
            return Err(format!(
                "mixture information expects {} columns, got {}",
                self.d,
                data.ncols()
            ));
        }
        let n = data.nrows();
        let p = self.num_free_parameters();
        let cov_per = self.d * (self.d + 1) / 2;
        // Precompute per-component evaluators (mean, precision = Σ⁻¹).
        let mut comp = Vec::with_capacity(self.k);
        for j in 0..self.k {
            comp.push(GaussianComponentEval::factor(
                self.means.row(j),
                &self.covariances[j],
            )?);
        }
        let log_w: Vec<f64> = self
            .weights
            .iter()
            .map(|w| w.max(f64::MIN_POSITIVE).ln())
            .collect();

        let mean_base = self.k - 1;
        let cov_base = mean_base + self.k * self.d;

        let mut info = Array2::<f64>::zeros((p, p));
        let mut score = vec![0.0_f64; p];
        for i in 0..n {
            let row = data.row(i);
            // Responsibilities r_j = w_j N_j / Σ.
            let mut log_terms = vec![0.0_f64; self.k];
            let mut max_term = f64::NEG_INFINITY;
            for j in 0..self.k {
                let lt = log_w[j] + comp[j].log_density(row);
                log_terms[j] = lt;
                if lt > max_term {
                    max_term = lt;
                }
            }
            let log_mix = log_sum_exp(&log_terms, max_term);
            let resp: Vec<f64> = log_terms.iter().map(|lt| (lt - log_mix).exp()).collect();

            for s in score.iter_mut() {
                *s = 0.0;
            }
            // Softmax-logit mixing score: ∂/∂α_j log p = r_j − w_j for the free
            // logits j = 1..k-1 (component 0 is the reference / pinned logit).
            for j in 1..self.k {
                score[j - 1] = resp[j] - self.weights[j];
            }
            // Mean score: ∂/∂μ_j log p = r_j · Σ_j⁻¹ (y − μ_j).
            // Covariance score (lower-tri entries): ∂/∂Σ_j contracted through
            // the symmetric chain rule, r_j · ½ (Σ⁻¹ v vᵀ Σ⁻¹ − Σ⁻¹) with
            // off-diagonal entries doubled for the symmetric parameterization.
            for j in 0..self.k {
                let prec_v = comp[j].precision_times_residual(row); // Σ⁻¹ (y − μ_j)
                let mbo = mean_base + j * self.d;
                for c in 0..self.d {
                    score[mbo + c] = resp[j] * prec_v[c];
                }
                let cbo = cov_base + j * cov_per;
                let mut idx = 0usize;
                for a in 0..self.d {
                    for b in 0..=a {
                        let outer = prec_v[a] * prec_v[b];
                        let prec_ab = comp[j].precision[[a, b]];
                        let mut g = 0.5 * (outer - prec_ab);
                        if a != b {
                            // Off-diagonal entry appears twice in the symmetric
                            // matrix, so its free-parameter derivative doubles.
                            g *= 2.0;
                        }
                        score[cbo + idx] = resp[j] * g;
                        idx += 1;
                    }
                }
            }
            // Accumulate outer product s_i s_iᵀ.
            for r in 0..p {
                let sr = score[r];
                if sr == 0.0 {
                    continue;
                }
                for c in 0..p {
                    info[[r, c]] += sr * score[c];
                }
            }
        }
        // Symmetrize and add a unit-precision ridge `I`. This is the Hessian
        // contribution of a standard-normal prior on the (natural) parameters,
        // making the object a proper MAP observed-information `H = I_prior +
        // Σ_i s_i s_iᵀ`. It guarantees SPD (well-defined `log|H|`) for any `n`
        // and is a fixed prior, not a tuned knob — the same unit-information
        // prior the rank-aware normalizer assumes when it credits each free
        // parameter one `log(2π)` of integration volume.
        for r in 0..p {
            for c in (r + 1)..p {
                let avg = 0.5 * (info[[r, c]] + info[[c, r]]);
                info[[r, c]] = avg;
                info[[c, r]] = avg;
            }
            info[[r, r]] += 1.0;
        }
        Ok(info)
    }
}

/// Cached per-component Gaussian evaluator: mean, precision `Σ⁻¹`, and the
/// log-normalizing constant `−½(d log 2π + log|Σ|)`.
#[derive(Debug, Clone)]
struct GaussianComponentEval {
    mean: Array1<f64>,
    precision: Array2<f64>,
    log_norm: f64,
    d: usize,
}

impl GaussianComponentEval {
    fn new(d: usize) -> Self {
        Self {
            mean: Array1::zeros(d),
            precision: Array2::eye(d),
            log_norm: 0.0,
            d,
        }
    }

    fn factor(mean: ArrayView1<'_, f64>, cov: &Array2<f64>) -> Result<Self, String> {
        let d = mean.len();
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
            inv_evals[idx] = 1.0 / ev;
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
        Ok(Self {
            mean: mean.to_owned(),
            precision,
            log_norm,
            d,
        })
    }

    #[inline]
    fn log_density(&self, y: ArrayView1<'_, f64>) -> f64 {
        let pv = self.precision_times_residual(y);
        let mut quad = 0.0_f64;
        for c in 0..self.d {
            quad += (y[c] - self.mean[c]) * pv[c];
        }
        self.log_norm - 0.5 * quad
    }

    /// `Σ⁻¹ (y − μ)`.
    #[inline]
    fn precision_times_residual(&self, y: ArrayView1<'_, f64>) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.d];
        for a in 0..self.d {
            let mut acc = 0.0_f64;
            for b in 0..self.d {
                acc += self.precision[[a, b]] * (y[b] - self.mean[b]);
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
) -> Result<GaussianMixtureFit, String> {
    let n = data.nrows();
    let d = data.ncols();
    if k == 0 {
        return Err("gaussian mixture requires k >= 1".to_string());
    }
    if d == 0 {
        return Err("gaussian mixture requires at least one column".to_string());
    }
    if k > n {
        return Err(format!(
            "gaussian mixture requested {k} components but data has {n} rows"
        ));
    }
    // Deterministic k-means++-style seeding via the shared basis k-means
    // (farthest-point init + Lloyd iterations). Fixed by construction.
    let centers = crate::basis::select_centers_by_strategy(
        data,
        &crate::basis::CenterStrategy::KMeans {
            num_centers: k,
            max_iter: config.kmeans_max_iter,
        },
    )
    .map_err(|e| format!("gaussian mixture k-means seeding failed: {e}"))?;
    if centers.nrows() != k || centers.ncols() != d {
        return Err(format!(
            "gaussian mixture seeding returned {}x{} centers, expected {k}x{d}",
            centers.nrows(),
            centers.ncols()
        ));
    }

    let mut means = centers;
    // Seed covariances from the global data covariance (shared start).
    let global_cov = data_covariance(data, config.covariance_floor);
    let mut covariances = vec![global_cov; k];
    let mut weights = Array1::<f64>::from_elem(k, 1.0 / k as f64);

    let mut resp = Array2::<f64>::zeros((n, k));
    let mut prev_mean_ll = f64::NEG_INFINITY;
    let mut total_loglik = f64::NEG_INFINITY;
    let mut iterations = 0usize;

    for iter in 0..config.max_iter.max(1) {
        iterations = iter + 1;
        // E-step: responsibilities and total log-likelihood.
        let mut comp = Vec::with_capacity(k);
        for j in 0..k {
            comp.push(GaussianComponentEval::factor(
                means.row(j),
                &covariances[j],
            )?);
        }
        let log_w: Vec<f64> = weights
            .iter()
            .map(|w| w.max(f64::MIN_POSITIVE).ln())
            .collect();
        total_loglik = 0.0;
        for i in 0..n {
            let yrow = data.row(i);
            let mut log_terms = vec![0.0_f64; k];
            let mut max_term = f64::NEG_INFINITY;
            for j in 0..k {
                let lt = log_w[j] + comp[j].log_density(yrow);
                log_terms[j] = lt;
                if lt > max_term {
                    max_term = lt;
                }
            }
            let log_mix = log_sum_exp(&log_terms, max_term);
            total_loglik += log_mix;
            for j in 0..k {
                resp[[i, j]] = (log_terms[j] - log_mix).exp();
            }
        }
        let mean_ll = total_loglik / n as f64;
        if iter > 0 {
            let denom = prev_mean_ll.abs().max(1.0);
            if (mean_ll - prev_mean_ll).abs() / denom <= config.loglik_tol {
                break;
            }
        }
        prev_mean_ll = mean_ll;

        // M-step.
        let mut nk = vec![0.0_f64; k];
        for j in 0..k {
            let mut sum = 0.0_f64;
            for i in 0..n {
                sum += resp[[i, j]];
            }
            nk[j] = sum.max(f64::MIN_POSITIVE);
        }
        for j in 0..k {
            weights[j] = nk[j] / n as f64;
            // Means.
            let mut mu = Array1::<f64>::zeros(d);
            for i in 0..n {
                let r = resp[[i, j]];
                if r == 0.0 {
                    continue;
                }
                for c in 0..d {
                    mu[c] += r * data[[i, c]];
                }
            }
            mu.mapv_inplace(|v| v / nk[j]);
            for c in 0..d {
                means[[j, c]] = mu[c];
            }
            // Covariance with a fixed diagonal floor for SPD safety.
            let mut cov = Array2::<f64>::zeros((d, d));
            for i in 0..n {
                let r = resp[[i, j]];
                if r == 0.0 {
                    continue;
                }
                for a in 0..d {
                    let da = data[[i, a]] - mu[a];
                    for b in 0..d {
                        cov[[a, b]] += r * da * (data[[i, b]] - mu[b]);
                    }
                }
            }
            cov.mapv_inplace(|v| v / nk[j]);
            for a in 0..d {
                cov[[a, a]] += config.covariance_floor;
            }
            covariances[j] = cov;
        }
    }

    Ok(GaussianMixtureFit {
        weights,
        means,
        covariances,
        k,
        d,
        n_obs: n,
        loglik: total_loglik,
        iterations,
    })
}

/// Global data covariance with a fixed diagonal floor (used to seed EM).
fn data_covariance(data: ArrayView2<'_, f64>, floor: f64) -> Array2<f64> {
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
    let inv = 1.0 / (n.max(1) as f64);
    cov.mapv_inplace(|v| v * inv);
    for a in 0..d {
        cov[[a, a]] += floor;
    }
    cov
}

// ---------------------------------------------------------------------------
// Structured-union candidates (#907)
// ---------------------------------------------------------------------------
//
// A *union* candidate is a small FIXED composite of named component structures
// joined by a hard row-responsibility split. Unlike the discrete-mixture rung
// (which is one free k-component Gaussian density), a union pins each component
// to a specific generative STRUCTURE (a circle, a line, a point cluster) and
// asks whether the data is better explained as the disjoint sum of those
// structures than by any single pure rung.
//
// Each component is fit on its responsibility group as its own parametric
// generative density and scored through the SAME rank-aware Laplace /
// Tierney-Kadane normalizer used by the smooth rungs and the mixture rung:
// `−V_c = loglik_c − ½ log|H_c| + ½ P_c log(2π)` with `H_c` the observed
// empirical-Fisher (BHHH) information `I + Σ s_i s_iᵀ` at the component optimum
// (`rank(S)=0`, fully likelihood-identified). The union's evidence is the SUM
// `V = Σ_c V_c` (the components partition the rows, so their log-likelihoods
// add and their Hessians are block-diagonal — `log|H| = Σ_c log|H_c|`). The
// complexity price is the TOTAL free-parameter count across all components,
// which is exactly what the summed `+ ½ Σ_c P_c log(2π)` normalizer charges.
// A union is therefore strictly more expensive than either pure component, so
// it can only win when the structured split buys enough likelihood to pay for
// its extra parameters — the negative-control discipline of #907.

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
/// to. `Line` and `PointCluster` share the full-covariance Gaussian density
/// (a line is an anisotropic Gaussian — the covariance, not a different
/// parameterization, is what distinguishes them); `Circle` is a genuinely
/// different density on `(radius, angle)`.
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

/// One fitted component of a union: its pinned structure, the rows it owns
/// (after the hard responsibility split), its free-parameter count, and its
/// rank-aware Laplace negative-log-evidence on the common scale.
#[derive(Debug, Clone)]
pub struct UnionComponentFit {
    pub kind: UnionComponentKind,
    pub row_count: usize,
    pub num_parameters: usize,
    pub negative_log_evidence: f64,
}

/// A fitted structured-union candidate: the composite kind, the per-component
/// fits, the SUMMED rank-aware Laplace negative-log-evidence, and the TOTAL
/// free-parameter count across components (the complexity price).
#[derive(Debug, Clone)]
pub struct UnionStructureFit {
    pub structure: UnionStructure,
    pub components: Vec<UnionComponentFit>,
    /// `Σ_c V_c` — summed rank-aware Laplace negative-log-evidence (lower wins).
    pub negative_log_evidence: f64,
    /// `Σ_c P_c` — total free-parameter count across components.
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
    let fit = fit_gaussian_mixture(data, m, config)?;
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); m];
    // Hard assignment by argmax per-component log responsibility.
    let mut comp = Vec::with_capacity(m);
    for j in 0..m {
        comp.push(GaussianComponentEval::factor(
            fit.means.row(j),
            &fit.covariances[j],
        )?);
    }
    let log_w: Vec<f64> = fit
        .weights
        .iter()
        .map(|w| w.max(f64::MIN_POSITIVE).ln())
        .collect();
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
        groups[best_j].push(i);
    }
    Ok(groups)
}

/// Fit one structured-union candidate: hard-split the rows into one group per
/// component, fit each component's pinned density, and SUM the rank-aware
/// Laplace negative-log-evidence. The complexity price is the total
/// free-parameter count across components.
///
/// Returns an error if any component group is too small to identify its
/// structure (so an over-priced or non-identifiable composite simply does not
/// enter the race rather than scoring spuriously well).
pub fn fit_union_structure(
    data: ArrayView2<'_, f64>,
    structure: UnionStructure,
    config: GaussianMixtureConfig,
) -> Result<UnionStructureFit, String> {
    let comps = structure.components();
    let m = comps.len();
    let groups = union_responsibility_split(data, m, config)?;
    let mut fits = Vec::with_capacity(m);
    let mut total_nle = 0.0_f64;
    let mut total_parameters = 0usize;
    for (kind, rows) in comps.iter().zip(groups.iter()) {
        let group = gather_union_rows(data, rows);
        let (nle, p) = fit_union_component(group.view(), *kind, config)?;
        if !nle.is_finite() {
            return Err(format!(
                "union {} component {:?} produced non-finite evidence",
                structure.as_str(),
                kind
            ));
        }
        total_nle += nle;
        total_parameters += p;
        fits.push(UnionComponentFit {
            kind: *kind,
            row_count: rows.len(),
            num_parameters: p,
            negative_log_evidence: nle,
        });
    }
    Ok(UnionStructureFit {
        structure,
        components: fits,
        negative_log_evidence: total_nle,
        total_parameters,
    })
}

/// Fit the whole fixed union ladder and rank in-class by summed rank-aware
/// Laplace evidence (lower wins). Composites that fail to fit (e.g. a group too
/// small to identify a circle) are skipped. Returns the fitted ladder sorted
/// best-first.
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
    if fits.is_empty() {
        return Err(format!(
            "union ladder produced no fittable composites{}",
            if errors.is_empty() {
                String::new()
            } else {
                format!(" ({})", errors.join("; "))
            }
        ));
    }
    let ranked = rank_priority_candidates(
        fits.into_iter()
            .enumerate()
            .map(|(idx, row)| {
                let score = row.negative_log_evidence;
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

/// Fit a single union component density on its responsibility group and return
/// `(rank_aware_negative_log_evidence, free_parameter_count)`. `Line` and
/// `PointCluster` use the full-covariance Gaussian density (a single mixture
/// component); `Circle` uses the radius/angle generative density below.
fn fit_union_component(
    group: ArrayView2<'_, f64>,
    kind: UnionComponentKind,
    config: GaussianMixtureConfig,
) -> Result<(f64, usize), String> {
    match kind {
        UnionComponentKind::Line | UnionComponentKind::PointCluster => {
            // A single full-covariance Gaussian is the k=1 mixture: reuse its
            // exact rank-aware Laplace evidence so a union component is on the
            // identical scale as a mixture component.
            if group.nrows() < group.ncols() + 1 {
                return Err(format!(
                    "union gaussian component needs >= {} rows, got {}",
                    group.ncols() + 1,
                    group.nrows()
                ));
            }
            let fit = fit_gaussian_mixture(group, 1, config)?;
            let nle = fit.laplace_negative_log_evidence(group)?;
            Ok((nle, fit.num_free_parameters()))
        }
        UnionComponentKind::Circle => fit_circle_component_evidence(group, config),
    }
}

/// Rank-aware Laplace negative-log-evidence of a 2-D *circle* component: data is
/// modelled as `(r, θ)` with `r ~ N(ρ, σ_r²)` around a fitted center+radius and
/// `θ` uniform on the circle. Free parameters: center `(cx, cy)`, radius `ρ`,
/// radial variance `σ_r²` — `P = 4`. The angle is an ancillary uniform with no
/// free parameter (it carries `−log(2π r)` of density). The Hessian is the
/// observed empirical-Fisher `I + Σ s_i s_iᵀ` in `(cx, cy, ρ, log σ_r²)`
/// coordinates, fed through the SAME [`laplace_evidence`] entry point.
fn fit_circle_component_evidence(
    group: ArrayView2<'_, f64>,
    config: GaussianMixtureConfig,
) -> Result<(f64, usize), String> {
    let d = group.ncols();
    if d != 2 {
        return Err(format!(
            "union circle component requires 2-D data, got {d} columns"
        ));
    }
    let n = group.nrows();
    let p = 4usize; // cx, cy, radius, radial-variance
    if n < p + 1 {
        return Err(format!(
            "union circle component needs >= {} rows, got {n}",
            p + 1
        ));
    }
    // Center = data centroid; radius = mean distance to centroid; radial
    // variance = mean squared radial residual (floored). This is the algebraic
    // circle-fit optimum for the isotropic radial-Gaussian model and is a pure
    // function of the data.
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    for i in 0..n {
        cx += group[[i, 0]];
        cy += group[[i, 1]];
    }
    cx /= n as f64;
    cy /= n as f64;
    let mut radii = vec![0.0_f64; n];
    let mut radius = 0.0_f64;
    for i in 0..n {
        let dx = group[[i, 0]] - cx;
        let dy = group[[i, 1]] - cy;
        let r = (dx * dx + dy * dy).sqrt();
        radii[i] = r;
        radius += r;
    }
    radius /= n as f64;
    let mut var_r = 0.0_f64;
    for &r in &radii {
        let e = r - radius;
        var_r += e * e;
    }
    var_r = (var_r / n as f64).max(config.covariance_floor);
    let inv_var = 1.0 / var_r;
    // Total log-likelihood: Σ_i [ −½ log(2π σ_r²) − (r_i−ρ)²/(2σ_r²)
    //                             − log(2π r_i) ]  (radial Gaussian × uniform θ).
    let mut loglik = 0.0_f64;
    let log_2pi = (2.0 * std::f64::consts::PI).ln();
    for &r in &radii {
        let e = r - radius;
        let radial = -0.5 * (log_2pi + var_r.ln()) - 0.5 * e * e * inv_var;
        let angular = -(log_2pi + r.max(f64::MIN_POSITIVE).ln());
        loglik += radial + angular;
    }
    // Observed empirical-Fisher in (cx, cy, ρ, s) with s = log σ_r².
    // Per-row scores:
    //   ∂/∂cx log = (e/σ_r²) · (−dx/r)            (r decreases as center moves +x toward point)
    //   ∂/∂cy log = (e/σ_r²) · (−dy/r)
    //   ∂/∂ρ  log = e/σ_r²
    //   ∂/∂s  log = −½ + e²/(2σ_r²)               (s = log σ_r²)
    let mut info = Array2::<f64>::zeros((p, p));
    let mut score = [0.0_f64; 4];
    for i in 0..n {
        let dx = group[[i, 0]] - cx;
        let dy = group[[i, 1]] - cy;
        let r = radii[i].max(f64::MIN_POSITIVE);
        let e = radii[i] - radius;
        let ee = e * inv_var;
        score[0] = ee * (-dx / r);
        score[1] = ee * (-dy / r);
        score[2] = ee;
        score[3] = -0.5 + 0.5 * e * e * inv_var;
        for a in 0..p {
            let sa = score[a];
            if sa == 0.0 {
                continue;
            }
            for b in 0..p {
                info[[a, b]] += sa * score[b];
            }
        }
    }
    // Symmetrize and add the unit-information prior ridge `I` (same fixed prior
    // as the mixture path) so `log|H|` is well-defined for any `n`.
    for a in 0..p {
        for b in (a + 1)..p {
            let avg = 0.5 * (info[[a, b]] + info[[b, a]]);
            info[[a, b]] = avg;
            info[[b, a]] = avg;
        }
        info[[a, a]] += 1.0;
    }
    let apply_info = |x: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0_f64; p];
        for r in 0..p {
            let mut acc = 0.0_f64;
            for c in 0..p {
                acc += info[[r, c]] * x[c];
            }
            out[r] = acc;
        }
        out
    };
    let hvp = EvidenceHvpLogDet {
        dim: p,
        apply: &apply_info,
    };
    let v = laplace_evidence(EvidenceLogDetSource::Hvp(hvp), 0.0, -loglik, p as f64, 0.0);
    if !v.is_finite() {
        return Err("union circle component Laplace evidence is not finite".to_string());
    }
    Ok((v, p))
}

/// A fitted union component as a *predictive density* (not just an evidence
/// scalar): either a full-covariance Gaussian (`Line`/`PointCluster`) or the
/// radial-Gaussian×uniform-angle circle density. Carries the mixing weight
/// `π_c = row_count_c / n_train` so a union can be evaluated as the soft mixture
/// `Σ_c π_c p_c(y)` at held-out rows for cross-class stacking.
#[derive(Debug, Clone)]
enum UnionComponentDensity {
    Gaussian {
        log_weight: f64,
        eval: GaussianComponentEval,
    },
    Circle {
        log_weight: f64,
        center: [f64; 2],
        radius: f64,
        var_r: f64,
    },
}

impl UnionComponentDensity {
    /// `log π_c + log p_c(y)` for one eval row.
    fn weighted_log_density(&self, y: ArrayView1<'_, f64>) -> f64 {
        match self {
            UnionComponentDensity::Gaussian { log_weight, eval } => {
                log_weight + eval.log_density(y)
            }
            UnionComponentDensity::Circle {
                log_weight,
                center,
                radius,
                var_r,
            } => {
                let dx = y[0] - center[0];
                let dy = y[1] - center[1];
                let r = (dx * dx + dy * dy).sqrt();
                let log_2pi = (2.0 * std::f64::consts::PI).ln();
                let e = r - radius;
                let radial = -0.5 * (log_2pi + var_r.ln()) - 0.5 * e * e / var_r;
                let angular = -(log_2pi + r.max(f64::MIN_POSITIVE).ln());
                log_weight + radial + angular
            }
        }
    }
}

/// Fit each union component's *density* on the training rows (hard
/// responsibility split) so the composite can be evaluated as the soft mixture
/// `Σ_c π_c p_c(y)` at new rows. Mixing weights are the training row shares.
fn fit_union_component_densities(
    train: ArrayView2<'_, f64>,
    structure: UnionStructure,
    config: GaussianMixtureConfig,
) -> Result<Vec<UnionComponentDensity>, String> {
    let comps = structure.components();
    let m = comps.len();
    let groups = union_responsibility_split(train, m, config)?;
    let n_train = train.nrows().max(1) as f64;
    let mut out = Vec::with_capacity(m);
    for (kind, rows) in comps.iter().zip(groups.iter()) {
        if rows.is_empty() {
            return Err(format!(
                "union {} held-out density: empty component group",
                structure.as_str()
            ));
        }
        let log_weight = (rows.len() as f64 / n_train).max(f64::MIN_POSITIVE).ln();
        let group = gather_union_rows(train, rows);
        match kind {
            UnionComponentKind::Line | UnionComponentKind::PointCluster => {
                if group.nrows() < group.ncols() + 1 {
                    return Err(format!(
                        "union gaussian component density needs >= {} rows, got {}",
                        group.ncols() + 1,
                        group.nrows()
                    ));
                }
                let fit = fit_gaussian_mixture(group.view(), 1, config)?;
                let eval = GaussianComponentEval::factor(fit.means.row(0), &fit.covariances[0])?;
                out.push(UnionComponentDensity::Gaussian { log_weight, eval });
            }
            UnionComponentKind::Circle => {
                let d = group.ncols();
                if d != 2 {
                    return Err(format!(
                        "union circle component density requires 2-D data, got {d} columns"
                    ));
                }
                let n = group.nrows();
                if n < 5 {
                    return Err(format!(
                        "union circle component density needs >= 5 rows, got {n}"
                    ));
                }
                let mut cx = 0.0_f64;
                let mut cy = 0.0_f64;
                for i in 0..n {
                    cx += group[[i, 0]];
                    cy += group[[i, 1]];
                }
                cx /= n as f64;
                cy /= n as f64;
                let mut radius = 0.0_f64;
                let mut radii = vec![0.0_f64; n];
                for i in 0..n {
                    let dx = group[[i, 0]] - cx;
                    let dy = group[[i, 1]] - cy;
                    let r = (dx * dx + dy * dy).sqrt();
                    radii[i] = r;
                    radius += r;
                }
                radius /= n as f64;
                let mut var_r = 0.0_f64;
                for &r in &radii {
                    let e = r - radius;
                    var_r += e * e;
                }
                var_r = (var_r / n as f64).max(config.covariance_floor);
                out.push(UnionComponentDensity::Circle {
                    log_weight,
                    center: [cx, cy],
                    radius,
                    var_r,
                });
            }
        }
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
    let densities = fit_union_component_densities(train, structure, config)?;
    let mut out = Array1::<f64>::zeros(eval.nrows());
    let mut terms = vec![f64::NEG_INFINITY; densities.len()];
    for i in 0..eval.nrows() {
        let row = eval.row(i);
        let mut max_term = f64::NEG_INFINITY;
        for (c, dens) in densities.iter().enumerate() {
            let lt = dens.weighted_log_density(row);
            terms[c] = lt;
            if lt > max_term {
                max_term = lt;
            }
        }
        out[i] = log_sum_exp(&terms, max_term);
    }
    Ok(out)
}

/// One fitted model in a REML/LAML evidence comparison.
#[derive(Clone, Debug)]
pub struct RemlCandidate {
    pub index: usize,
    pub name: String,
    /// Minimised REML/LAML cost. Lower is better.
    pub score: f64,
    pub edf: Option<f64>,
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
    /// Cost gap from the winning model: `score - best_score`.
    pub delta: f64,
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
    candidates = rank_priority_candidates(
        candidates
            .into_iter()
            .enumerate()
            .map(|(idx, row)| {
                let score = row.score;
                PriorityCandidate::new(row, idx, score, 0)
            })
            .collect(),
    )
    .into_iter()
    .map(|row| row.item)
    .collect();

    let best_score = candidates[0].score;
    let winner = candidates[0].name.clone();
    let mut ranking = Vec::with_capacity(candidates.len());
    let mut score_table = Vec::with_capacity(candidates.len());
    for row in &candidates {
        let delta = log_bayes_factor(best_score, row.score);
        let bayes_factor = delta.exp();
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
            delta_reml: delta,
            bayes_factor_best_over_model: bayes_factor,
            effective_dof: row.edf,
        });
    }
    let evidence_summary = if let Some(runner_up) = candidates.get(1) {
        format!(
            "{} wins by Bayes factor {} over {}",
            winner,
            format_bayes_factor(log_bayes_factor(best_score, runner_up.score)),
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
/// pseudo-logdet from `crate::solver::reml::penalty_logdet` (proposal
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
    if crate::gpu::cuda_selected() {
        return crate::solver::gpu::reml_gpu::evidence_derivatives_gpu(
            crate::solver::gpu::reml_gpu::RemlGpuInput {
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
    crate::linalg::utils::splitmix64(state)
}

/// Sum of per-row arrow log-determinants plus the Schur log-det.
///
/// `log|H| = Σ_i log|H_uu_i| + log|A|` using the undamped Cholesky
/// factors of `H_uu_i` and the cached Schur Cholesky factor.
///
/// Returns `None` if `cache.schur_factor` is absent (InexactPCG path) or
/// if a damped/incoherent cache is supplied. [`evidence_hessian_log_det`]
/// routes such matrix-free cases to an explicit HVP fallback.
pub fn arrow_log_det_from_cache(cache: &ArrowFactorCache) -> Option<f64> {
    if cache.ridge_t != 0.0 || cache.ridge_beta != 0.0 {
        // Per proposal §6.4 / §6.5 — evidence must use the undamped
        // operator. The cache's Schur factor here was assembled under
        // ridge damping, which is a different operator. Reject loudly.
        return None;
    }
    if let Some(log_det) = cache.joint_hessian_log_det {
        return log_det.is_finite().then_some(log_det);
    }
    // A `k == 0` cache has no shared β block, so the dense Direct path forms no
    // reduced Schur complement and `schur_factor` is legitimately `None` (the
    // joint Hessian is block-diagonal in the latent rows). Its log-det is the
    // per-row sum with no Schur term. Only reject when `k > 0` and the factor
    // is absent — the InexactPCG case that never built the dense `K×K` factor.
    // (#1132 euclidean K=4: a β-profiled atom reaches here with `k == 0`.)
    let schur = match cache.schur_factor.as_ref() {
        Some(schur) => Some(schur),
        None if cache.k == 0 => None,
        None => return None,
    };

    let mut acc = 0.0_f64;
    // Per-row arrow blocks: log|H_uu_i| = 2 Σ log diag(L_i).
    for l in cache.undamped_factors_iter() {
        acc += 2.0 * log_det_from_chol_lower(l);
    }
    // Schur block: log|A| = 2 Σ log diag(L_schur). Empty for the `k == 0` case.
    if let Some(schur) = schur {
        acc += 2.0 * log_det_from_chol_lower(schur.view());
    }
    // #1038 cross-row IBP: when the cache carries an exact rank-`R` Woodbury,
    // the per-row + Schur factors above are of the NO-SELF base `H₀'`, so the
    // exact `log det H_full = log det H₀' + log det(I_R + D Uᵀ H₀'⁻¹ U)`. The
    // correction is zero (no-op) for every non-IBP cache.
    let woodbury_correction = cache.cross_row_woodbury_log_det();
    if !woodbury_correction.is_finite() {
        // A non-PD capacitance (negative determinant) is a value↔gradient
        // desync the evidence must reject loudly, not paper over.
        return None;
    }
    acc += woodbury_correction;
    Some(acc)
}

/// Twice-the-diagonal-log sum for a lower-triangular Cholesky factor.
fn log_det_from_chol_lower(l: ArrayView2<'_, f64>) -> f64 {
    let n = l.nrows();
    let mut acc = 0.0_f64;
    for i in 0..n {
        let d = l[[i, i]];
        // Guard against negative diagonal (impossible for a valid
        // Cholesky factor, but protect against caller corruption).
        if d > 0.0 {
            acc += d.ln();
        } else {
            return f64::NAN;
        }
    }
    acc
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
    if cache.ridge_t != 0.0 || cache.ridge_beta != 0.0 {
        return None;
    }
    let schur = cache.schur_factor.as_ref()?;
    if dg_red_drho.nrows() != cache.k || schur.nrows() != cache.k {
        return None;
    }
    crate::solver::sensitivity::FitSensitivity::from_lower_triangular(schur)
        .mode_response(dg_red_drho)
}

/// Tier-3 IFT sensitivity `∂u*/∂ρ` (proposal §2.6 / §7).
///
/// ```text
/// ∂u*/∂ρ_a = -H_uu⁻¹ G_{u,ρ_a} - H_uu⁻¹ H_uβ ∂β*/∂ρ_a.
/// ```
///
/// `gu_rho` is the `(N·d) × R` matrix of `G_{u,ρ_a}` columns and
/// `dbeta_drho` is the `K × R` matrix from [`ift_dbeta_drho`]. Returns
/// the `(N·d) × R` matrix `u_ρ`.
pub fn ift_du_drho(
    cache: &ArrowFactorCache,
    gu_rho: ArrayView2<'_, f64>,
    dbeta_drho: ArrayView2<'_, f64>,
) -> Array2<f64> {
    let n = cache.undamped_factor_count();
    let total_len = cache.delta_t_len();
    let k = cache.k;
    let r = dbeta_drho.ncols();
    if !cache.htbeta_available()
        || gu_rho.nrows() != total_len
        || gu_rho.ncols() != r
        || dbeta_drho.nrows() != k
    {
        return Array2::<f64>::from_elem((total_len, r), f64::NAN);
    }

    let mut out = Array2::<f64>::zeros((total_len, r));
    // Allocate scratch at max_d; per-row slice is ..di.
    let mut rhs = Array1::<f64>::zeros(cache.d);
    let mut htbeta_delta = Array1::<f64>::zeros(cache.d);
    for a in 0..r {
        // Per-row: rhs_i = G_{u_i,ρ_a} + H_uβ_i · ∂β*/∂ρ_a.
        for i in 0..n {
            let di = cache.row_dims[i];
            let row_base = cache.row_offsets[i];
            let mut htbeta_i = htbeta_delta.slice_mut(ndarray::s![..di]).to_owned();
            // Companion to the `du/dβ` assembler above; same H_tβ cache.
            if !cache.apply_htbeta_row(i, dbeta_drho.column(a), &mut htbeta_i) {
                // SAFETY: `false` here means the family declared H_tβ row
                // products available but did not populate them — contract
                // violation against the joint-evidence capability surface.
                return Array2::<f64>::from_elem((total_len, r), f64::NAN);
            }
            {
                let mut rhs_i = rhs.slice_mut(ndarray::s![..di]);
                for c in 0..di {
                    rhs_i[c] = gu_rho[[row_base + c, a]] + htbeta_i[c];
                }
            }
            let rhs_slice = rhs.slice(ndarray::s![..di]).to_owned();
            // u_ρ_i = -H_uu_i⁻¹ rhs_i, undamped factor.
            let v = cholesky_solve_vector(cache.undamped_factor(i), &rhs_slice);
            for c in 0..di {
                out[[row_base + c, a]] = -v[c];
            }
        }
    }
    out
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

/// Sanity check used by callers that require exact factor-backed evidence.
/// Proposal §6.4 — ridges must be zero on the evidence-evaluation path.
/// Matrix-free callers can instead pass an HVP fallback to
/// [`laplace_evidence`].
pub fn cache_supports_exact_evidence(cache: &ArrowFactorCache) -> bool {
    cache.ridge_t == 0.0
        && cache.ridge_beta == 0.0
        && cache.schur_factor.is_some()
        && cache.htbeta_available()
}

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
// A linear SAE atom is the EXACT special case of a curved d=1 atom: the
// euclidean-d=1-linear basis is one decoder direction (`γ(t) = t·b`, a straight
// image with zero turning). So a dictionary whose atom set INCLUDES the linear
// atom as a special case cannot lose to a pure-linear dictionary at matched
// active budget — it strictly generalizes it. The only open question is, per
// atom, whether paying for the curved parameterization buys enough likelihood
// to beat the cheaper linear special case. This module adjudicates that split
// by the SAME rank-aware Laplace evidence criterion the union/mixture rungs use
// (`−V = NLE`, lower wins), so the fit selects the curved-vs-linear split by
// evidence rather than fiat.
//
// ## The dominance floor (Θ → 0) and the curved ceiling (Θ large)
//
// The decision is structurally pinned by nesting. Because the linear fit is the
// curved family restricted to its straight (`Θ = 0`) sub-model, the curved fit's
// maximized likelihood is ALWAYS ≥ the linear fit's at the same rows. But the
// curved atom pays a strictly larger free-parameter price `P_curved > P_linear`
// (the extra basis coefficients beyond the single decoder direction), which the
// rank-aware Laplace normalizer charges. Hence:
//
//   * Θ → 0 (a straight feature): the curved fit recovers no extra likelihood
//     over its linear sub-model, so `NLE_curved ≥ NLE_linear` and LINEAR wins —
//     the dominance floor. A curved atom "buys nothing on a straight feature."
//   * Θ large (a genuinely turning feature): the curved fit captures curvature
//     the linear secant cannot, lowering `NLE_curved` below `NLE_linear` by more
//     than the parameter price, so CURVED wins.
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
/// the SAME slot are fit on the same rows, so their NLEs are directly
/// comparable; the only structural difference is the curved candidate's larger
/// free-parameter price.
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
///     `NLE`. Because the linear atom is the curved family's `Θ = 0` sub-model,
///     the curved candidate can only win when its extra curvature lowers the NLE
///     by MORE than its extra parameter price — the `Θ/√ε` crossover, decided
///     here by the evidence numbers themselves, not by fiat.
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
    /// slot picks the argmin, this is ≤ the pure-linear dictionary's summed NLE
    /// at the same slots: the hybrid match-or-beats pure-linear by construction.
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
/// has `Θ → 0` (the dominance floor fires everywhere) and to pure-curved when
/// every slot's curved candidate wins the evidence comparison — the two limits
/// the strict-generalization argument demands.
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
    use crate::solver::arrow_schur::ArrowFactorSlab;

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
        let op = crate::solver::sensitivity::FitSensitivity::from_projected(&eye, &inv);
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
        let coned = crate::solver::sensitivity::FitSensitivity::from_projected(&eye, &inv)
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
        let coned = crate::solver::sensitivity::FitSensitivity::from_projected(&eye, &nan_inv)
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
        ArrowFactorCache {
            htt_factors: ArrowFactorSlab::from_blocks(vec![l_huu]),
            htt_factors_undamped: crate::solver::arrow_schur::ArrowUndampedFactors::SameAsDamped,
            schur_factor: Some(l_schur),
            joint_hessian_log_det: None,
            solver_mode: crate::solver::arrow_schur::ArrowSolverMode::Direct,
            ridge_t: 0.0,
            ridge_beta: 0.0,
            htbeta: crate::solver::arrow_schur::ArrowHtbetaCache::Dense {
                blocks: std::sync::Arc::from(vec![htbeta]),
                estimated_bytes: std::mem::size_of::<f64>(),
            },
            d: 1,
            row_dims: std::sync::Arc::from(vec![1usize]),
            row_offsets: std::sync::Arc::from(vec![0usize, 1usize]),
            k: 1,
            manifold_mode_fingerprint: 0,
            row_hessian_fingerprint: 0,
            pcg_diagnostics: crate::solver::arrow_schur::PcgDiagnostics::default(),
            gauge_deflated_directions: 0,
            cross_row_woodbury: None,
        }
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
        ArrowFactorCache {
            htt_factors: ArrowFactorSlab::from_blocks(vec![l_huu]),
            htt_factors_undamped: crate::solver::arrow_schur::ArrowUndampedFactors::SameAsDamped,
            schur_factor: None,
            joint_hessian_log_det: None,
            solver_mode: crate::solver::arrow_schur::ArrowSolverMode::Direct,
            ridge_t: 0.0,
            ridge_beta: 0.0,
            htbeta: crate::solver::arrow_schur::ArrowHtbetaCache::Disabled { estimated_bytes: 0 },
            d: 1,
            row_dims: std::sync::Arc::from(vec![1usize]),
            row_offsets: std::sync::Arc::from(vec![0usize, 1usize]),
            k: 0,
            manifold_mode_fingerprint: 0,
            row_hessian_fingerprint: 0,
            pcg_diagnostics: crate::solver::arrow_schur::PcgDiagnostics::default(),
            gauge_deflated_directions: 0,
            cross_row_woodbury: None,
        }
    }

    #[test]
    fn arrow_log_det_some_for_k0_direct_cache_without_schur() {
        let cache = k0_direct_cache_no_schur(3.0);
        let log_det = arrow_log_det_from_cache(&cache)
            .expect("k==0 Direct cache must yield Some(per-row sum), not None (#1132)");
        // Single latent block H_tt = [[3.0]]; no Schur term for k == 0.
        assert!((log_det - 3.0_f64.ln()).abs() < 1e-12, "log_det = {log_det}");
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
        cache.solver_mode = crate::solver::arrow_schur::ArrowSolverMode::InexactPCG;
        assert!(arrow_log_det_from_cache(&cache).is_none());
        assert!(cache.compute_undamped_arrow_log_det().is_none());
    }

    #[test]
    fn laplace_evidence_nan_when_ridge_is_nonzero() {
        let mut cache = make_minimal_cache();
        cache.ridge_t = 1e-3;
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
    fn laplace_evidence_uses_hvp_fallback_without_schur_factor() {
        let mut cache = make_minimal_cache();
        cache.schur_factor = None;
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
    fn stacking_mean_log_score_is_monotone_under_more_iterations() {
        // The EM ascent is monotone in the held-out mean log-score, so allowing
        // more iterations never lowers it.
        let log_density =
            Array2::from_shape_vec((4, 2), vec![-0.2, -3.0, -3.0, -0.2, -0.5, -1.5, -1.5, -0.5])
                .unwrap();
        let mut prev = f64::NEG_INFINITY;
        for max_iter in [1usize, 2, 4, 8, 32] {
            let out = solve_stacking_weights(
                log_density.view(),
                StackingConfig {
                    max_iter,
                    weight_tol: 0.0,
                },
            )
            .unwrap();
            assert!(
                out.mean_log_score >= prev - 1e-12,
                "log-score decreased at max_iter={max_iter}: {prev} -> {}",
                out.mean_log_score
            );
            prev = out.mean_log_score;
        }
    }

    #[test]
    fn stacking_dead_candidate_column_is_rejected_and_zero_weighted() {
        let log_density = Array2::from_shape_vec(
            (3, 2),
            vec![
                -1.0,
                f64::NEG_INFINITY,
                -2.0,
                f64::NAN,
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
    fn stacking_rows_with_no_finite_density_are_dropped() {
        let log_density = Array2::from_shape_vec(
            (3, 2),
            vec![-1.0, -2.0, f64::NAN, f64::NEG_INFINITY, -2.0, -1.0],
        )
        .unwrap();
        let out = solve_stacking_weights(log_density.view(), StackingConfig::default()).unwrap();
        assert!((out.weights.sum() - 1.0).abs() < 1e-9);
        assert!(out.mean_log_score.is_finite());
    }

    #[test]
    fn stacking_all_dead_table_errors() {
        let log_density = Array2::from_elem((2, 2), f64::NEG_INFINITY);
        assert!(solve_stacking_weights(log_density.view(), StackingConfig::default()).is_err());
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
        // circles and linear for the directions — and at matched actives the
        // hybrid's summed evidence must be ≤ the pure-linear baseline (the
        // strict-generalization, match-or-beat guarantee, (4)).
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

        // EV at matched actives: the hybrid's summed negative-log-evidence is ≤
        // the pure-linear dictionary's (lower NLE = higher evidence = the curved
        // atoms strictly improved the circle slots, the linear slots are
        // unchanged). The hybrid match-or-beats pure-linear by construction.
        assert!(
            split.total_negative_log_evidence <= pure_linear_baseline + 1e-9,
            "hybrid NLE {} must be <= pure-linear baseline {}",
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
}
