//! SAE identifiability primitives and partial-supervision gauge fixing.
//!
//! # Object 4 — the Certificate ([`residual_gauge`])
//!
//! The partial-supervision solver above *removes* gauge freedom by aligning to
//! auxiliary supervision. The certificate answers the dual question: after a fit
//! has converged, **which gauge group is the model identified up to?** It does
//! so by running the same penalty-aware RRQR rank machinery the cross-block
//! identifiability audit uses
//! ([`crate::identifiability::audit::audit_identifiability`] /
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

use crate::inference::layer_transport::{ChartTopology, TransportLadderReport, transport_ladder};
use crate::inference::probe_runner::{ProbeRunner, RealizedProbe};
use crate::inference::riesz::{RieszInput, SmoothFunctional, debias_with_dense_hessian};
use gam_problem::{MetricProvenance, RowMetric};
use gam_terms::inference::structure_evidence::{StructureCertificate, StructureLedger};
use gam_linalg::faer_ndarray::{
    FaerCholesky, FaerEigh, FaerQr, FaerSvd, default_rrqr_rank_alpha, rrqr_with_permutation,
};
use crate::chart_canonicalization::CanonicalChartTopology;
use crate::manifold::SaeManifoldTerm;
use faer::Side;
use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, s};
use std::f64::consts::TAU;

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
        // equivalent (an invertible reparameterisation) to requiring that
        // the stacked signature `S = [μ(u) ‖ log σ(u)]` of shape
        // (n_rows, 2k) have rank 2k, with at least 2k+1 distinct rows.
        let (n_rows, latent_dim) = mean.dim();
        let needed_rows = 2 * latent_dim + 1;
        if n_rows < needed_rows {
            return Err(format!(
                "ConditionalPriorIvae: Khemakhem (arXiv:2107.10098) Theorem 1 \
                 precondition violated: need at least 2k+1 = {needed_rows} distinct \
                 auxiliary states for latent_dim k = {latent_dim}, got n_rows = {n_rows}"
            ));
        }
        let signature = {
            let mut s = Array2::<f64>::zeros((n_rows, 2 * latent_dim));
            for r in 0..n_rows {
                for c in 0..latent_dim {
                    s[[r, c]] = mean[[r, c]];
                    s[[r, latent_dim + c]] = scale[[r, c]].ln();
                }
            }
            s
        };
        let first = signature.row(0).to_owned();
        let all_identical = signature
            .outer_iter()
            .all(|row| row.iter().zip(first.iter()).all(|(a, b)| a == b));
        if all_identical {
            return Err(format!(
                "ConditionalPriorIvae: Khemakhem (arXiv:2107.10098) Theorem 1 \
                 precondition violated: all {n_rows} rows of the stacked auxiliary \
                 signature [μ ‖ log σ] are identical, so the conditional prior is the \
                 trivial unconditional N(μ, σ²) — provably non-identifiable (no \
                 auxiliary information)"
            ));
        }
        let (_u, sv, _vt) = signature
            .svd(false, false)
            .map_err(|e| format!("ConditionalPriorIvae: SVD of auxiliary signature failed: {e}"))?;
        let max_sv = sv.iter().cloned().fold(0.0_f64, f64::max);
        let tol = max_sv * (n_rows.max(2 * latent_dim) as f64) * f64::EPSILON;
        let numerical_rank = sv.iter().filter(|&&s| s > tol).count();
        let required = 2 * latent_dim;
        if numerical_rank < required {
            return Err(format!(
                "ConditionalPriorIvae: Khemakhem (arXiv:2107.10098) Theorem 1 \
                 precondition violated: stacked auxiliary signature [μ ‖ log σ] has \
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

/// Helper: evaluate a piecewise-linear "smooth" `f(u)` columnwise, given a
/// (k_centres, latent_dim) coefficient table and a (n_rows,) auxiliary vector
/// `u`. Used by the Python wrapper to back the iVAE per-latent (μ_i(u), σ_i(u))
/// without having to round-trip through gam's full Smooth machinery for the
/// minimal experiments. Centres are assumed evenly spaced in [u_min, u_max].
pub fn piecewise_linear_eval(
    u: ArrayView1<f64>,
    coeffs: ArrayView2<f64>,
    u_min: f64,
    u_max: f64,
) -> Array2<f64> {
    let (k, d) = coeffs.dim();
    assert!(k >= 2, "piecewise_linear_eval: need ≥2 centres");
    let n = u.len();
    let mut out = Array2::<f64>::zeros((n, d));
    let step = (u_max - u_min) / (k - 1) as f64;
    for (row, &val) in u.iter().enumerate() {
        // Clamp `pos` to the exact endpoint `(k-1)`, not `(k-1) - 1e-12`,
        // so `val = u_max` evaluates to exactly `coeffs[k-1, col]` instead
        // of `coeffs[k-1, col] + 1e-12 · (coeffs[k-2, col] − coeffs[k-1,
        // col])`. The historical `1e-12` shift was there to keep `lo + 1`
        // in range, but capping `lo` at `k − 2` achieves the same
        // structural guarantee without perturbing the endpoint value.
        let pos = ((val - u_min) / step).clamp(0.0, (k - 1) as f64);
        let lo = (pos.floor() as usize).min(k - 2);
        let hi = lo + 1;
        let frac = pos - lo as f64;
        for col in 0..d {
            out[[row, col]] = coeffs[[lo, col]] * (1.0 - frac) + coeffs[[hi, col]] * frac;
        }
    }
    out
}

/// Outcome of a 2D log-λ grid-search weight selection.
///
/// `evidence_grid[i, j]` is the Laplace-style log marginal-likelihood proxy
/// at `(lam1_grid[i], lam2_grid[j])`:
/// `evidence = −½ N log(RSS/N) − ½ (penalty)` with `RSS = rss_grid[i, j]`
/// and `penalty = penalty_grid[i, j]`.
///
/// The winner is `argmax` over the grid; ties are broken by selecting the
/// `(i, j)` with the smallest `i + j` (i.e. smallest log-weight sum on a
/// log-spaced grid), then by smallest `i`, then smallest `j` — a fully
/// deterministic, reproducible policy.
#[derive(Debug, Clone)]
pub struct WeightSearchResult {
    pub best_i: usize,
    pub best_j: usize,
    pub best_lam1: f64,
    pub best_lam2: f64,
    pub best_evidence: f64,
    pub evidence_grid: Array2<f64>,
}

/// Generic 2D log-λ weight-selection driver.
///
/// Given a precomputed `(G1, G2)` grid of residual sums-of-squares
/// `rss_grid`, a matching grid of total-penalty values `penalty_grid`, and
/// the two 1D weight grids `lam1_grid` / `lam2_grid`, computes the Laplace
/// log marginal-likelihood proxy on every cell and returns the maximising
/// cell with deterministic tie-breaking.
///
/// The primitive is intentionally agnostic to *what* the two penalty
/// weights regularise — it takes only the RSS and penalty surfaces, so it
/// can drive weight selection for any two-penalty model (identifiable
/// factor model, double-penalty smooths, IBP + sparsity, etc.).
pub fn identifiable_factor_select_weights(
    rss_grid: ArrayView2<'_, f64>,
    penalty_grid: ArrayView2<'_, f64>,
    lam1_grid: ArrayView1<'_, f64>,
    lam2_grid: ArrayView1<'_, f64>,
    n_obs: usize,
) -> Result<WeightSearchResult, String> {
    let (g1, g2) = rss_grid.dim();
    if penalty_grid.dim() != (g1, g2) {
        return Err(format!(
            "identifiable_factor_select_weights: penalty_grid shape {:?} \
             must match rss_grid shape ({}, {})",
            penalty_grid.dim(),
            g1,
            g2
        ));
    }
    if lam1_grid.len() != g1 {
        return Err(format!(
            "identifiable_factor_select_weights: lam1_grid len {} must \
             equal rss_grid rows {}",
            lam1_grid.len(),
            g1
        ));
    }
    if lam2_grid.len() != g2 {
        return Err(format!(
            "identifiable_factor_select_weights: lam2_grid len {} must \
             equal rss_grid cols {}",
            lam2_grid.len(),
            g2
        ));
    }
    if g1 == 0 || g2 == 0 {
        return Err("identifiable_factor_select_weights: grids must be non-empty".to_string());
    }
    if n_obs == 0 {
        return Err("identifiable_factor_select_weights: n_obs must be > 0".to_string());
    }
    for v in rss_grid.iter() {
        if !v.is_finite() || *v < 0.0 {
            return Err(format!(
                "identifiable_factor_select_weights: rss_grid contains non-finite or \
                 negative value {v}"
            ));
        }
    }
    for v in penalty_grid.iter() {
        if !v.is_finite() {
            return Err(format!(
                "identifiable_factor_select_weights: penalty_grid contains non-finite value {v}"
            ));
        }
    }
    for v in lam1_grid.iter().chain(lam2_grid.iter()) {
        if !v.is_finite() || *v <= 0.0 {
            return Err(format!(
                "identifiable_factor_select_weights: λ grids must contain finite positive \
                 values, got {v}"
            ));
        }
    }

    let n = n_obs as f64;
    let rss_floor = 1.0e-300_f64;
    let mut evidence_grid = Array2::<f64>::zeros((g1, g2));
    let mut best: Option<(usize, usize, f64)> = None;
    for i in 0..g1 {
        for j in 0..g2 {
            let rss = rss_grid[[i, j]];
            let pen = penalty_grid[[i, j]];
            let mean_sq = (rss / n).max(rss_floor);
            let ev = -0.5 * n * mean_sq.ln() - 0.5 * pen;
            evidence_grid[[i, j]] = ev;
            let better = match best {
                None => true,
                Some((bi, bj, bev)) => {
                    if ev > bev {
                        true
                    } else if ev == bev {
                        let cur_sum = i + j;
                        let best_sum = bi + bj;
                        if cur_sum < best_sum {
                            true
                        } else if cur_sum == best_sum && i < bi {
                            true
                        } else {
                            cur_sum == best_sum && i == bi && j < bj
                        }
                    } else {
                        false
                    }
                }
            };
            if better {
                best = Some((i, j, ev));
            }
        }
    }
    let (best_i, best_j, best_evidence) = best.ok_or_else(|| {
        "identifiable_factor_select_weights: empty search (this is a bug)".to_string()
    })?;
    Ok(WeightSearchResult {
        best_i,
        best_j,
        best_lam1: lam1_grid[best_i],
        best_lam2: lam2_grid[best_j],
        best_evidence,
        evidence_grid,
    })
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
            let lam_max = eigvals.iter().cloned().fold(0.0_f64, f64::max);
            let floor = (lam_max * 1.0e-10).max(1.0e-12);
            let top = (lam_max * 1.0e3).max(floor * 1.0e6);
            let grid_n: usize = 64;
            let log_floor = floor.ln();
            let log_top = top.ln();
            // Select λ by REML, never GCV. The ridge map is the linear mixed
            // model aux_j = T β_j + ε with β_j ~ N(0, σ²/λ I), ε ~ N(0, σ² I)
            // applied to each of the d columns sharing λ. The map carries no
            // unpenalized fixed effect, so REML coincides with the marginal
            // likelihood, whose profile (σ² concentrated out) criterion to
            // MINIMIZE is
            //   reml(λ) = n·log S(λ) + Σ_r log(1 + γ_r/λ),
            // the exact analogue of the smoothing-parameter REML used
            // everywhere else in gam.
            let mut best_score = f64::INFINITY;
            let mut best_lam = floor;
            for k in 0..grid_n {
                let frac = if grid_n == 1 {
                    0.0
                } else {
                    (k as f64) / ((grid_n - 1) as f64)
                };
                let lam = (log_floor + frac * (log_top - log_floor)).exp();
                let mut shrunk = 0.0_f64; // Σ_r m_r/(γ_r+λ)
                let mut logdet = 0.0_f64; // Σ_r log(1 + γ_r/λ)
                for r in 0..d_sup {
                    let g = eigvals[r].max(0.0);
                    shrunk += m_row[r] / (g + lam);
                    logdet += (1.0 + g / lam).ln();
                }
                let s = aux_norm_sq - shrunk;
                if !(s.is_finite() && s > 0.0) {
                    continue;
                }
                let score = (n as f64) * s.ln() + logdet;
                if score < best_score {
                    best_score = score;
                    best_lam = lam;
                }
            }
            if !best_score.is_finite() {
                return Err(
                    "partial_supervision_solve: REML grid did not find a finite-score weight"
                        .to_string(),
                );
            }
            // Build the ridge map A_λ = (G + λI)⁻¹ Tᵀaux at the REML weight.
            let denom: Array1<f64> = eigvals.mapv(|v| v + best_lam);
            let mut a_eig = Array2::<f64>::zeros((d_sup, d_sup));
            for r in 0..d_sup {
                for c in 0..d_sup {
                    a_eig[[r, c]] = ut_aux[[r, c]] / denom[r];
                }
            }
            let best_a = eigvecs.dot(&a_eig);
            t_sup_aligned = t_sup.dot(&best_a);
            map_a = Some(best_a);
            selected_weight = Some(best_lam);
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
                let qr_pair = t_sup_aligned
                    .qr()
                    .map_err(|e| format!("partial_supervision_solve: QR on T_sup failed: {e}"))?;
                let q = qr_pair.0;
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
    /// `Tᵈ` (product of `latent_dim` circles). `Isom` contains the `d`
    /// independent circle shifts (a maximal torus of rotations).
    Torus { latent_dim: usize },
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
            AtomTopology::Torus { latent_dim } => *latent_dim,
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
    /// #1019 stage 1: `true` when the atom's `d = 1` latent chart was pinned
    /// post-fit to its arc-length (unit-speed) canonical representative. #1019
    /// stage 2: `true` as well when a `d = 2` torus atom's chart was pinned
    /// post-fit to the minimum-isometry-defect flow representative, in which
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

impl FittedAtom {
    /// Attach the inner-decoder-smooth byproducts harvested at fit time. The
    /// term builder calls this so [`dictionary_report`] can produce the three
    /// post-PIRLS atom inference reports.
    pub fn with_inner_fit(mut self, inner_fit: AtomInnerFit) -> Self {
        self.inner_fit = Some(inner_fit);
        self
    }
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

impl FrameInnerRotationGauge {
    /// Enumerate the gauge from the active frame ranks.
    pub fn from_ranks(per_atom_ranks: Vec<usize>) -> Self {
        let dim = frame_inner_rotation_dim(&per_atom_ranks);
        Self {
            per_atom_ranks,
            dim,
        }
    }
}

/// `Σ_k r_k (r_k − 1) / 2` — the dimension of the #972 inner-rotation gauge
/// group `∏_k O(r_k)` over the active frame ranks. Rank-1 frames contribute
/// `0` (`O(1)` is finite, a sign — absorbed by the orientation gauge), so a
/// dictionary of single-direction atoms reports a zero-dimensional inner
/// gauge, matching the intuition that one direction has no inner rotation to
/// fix.
pub fn frame_inner_rotation_dim(ranks: &[usize]) -> usize {
    ranks.iter().map(|&r| r * r.saturating_sub(1) / 2).sum()
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
    /// generators were tested against, in the metric.
    pub pinning_rank: usize,
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

    /// Attach the #972 frame inner-rotation enumeration to the certificate
    /// (consumed by frame-factored dictionaries; `ranks` are the active frame
    /// ranks `r_k`, one per factored atom). Extends the summary so the
    /// one-line report names the enumerated-but-convention-fixed gauge.
    pub fn with_frame_inner_rotation(mut self, ranks: Vec<usize>) -> Self {
        let gauge = FrameInnerRotationGauge::from_ranks(ranks);
        if gauge.dim > 0 {
            self.summary.push_str(&format!(
                "; frame inner-rotation gauge ∏O(r_k) of dim {} enumerated \
                 (exact reparameterization, fixed by the canonical orientation gauge)",
                gauge.dim
            ));
        }
        self.frame_inner_rotation = Some(gauge);
        self
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
            // so(2): A = [[0,-1],[1,0]] on the 1 circle, but a Circle atom has a
            // single latent axis whose isometry is a *shift* of the periodic
            // coordinate. The first-order motion of the (cos,sin) frame columns
            // under a shift is the orthogonal frame column. With latent_dim == 1
            // the decoder frame's single column moves along its own
            // 90°-rotated image, which (lacking a second column) is realised as
            // the tangent that advances the periodic phase: the unit direction
            // along the frame column itself (the generator of the U(1) shift).
            if d >= 1 {
                let mut g = Array1::<f64>::zeros(p * d);
                for i in 0..p {
                    g[i * d] = atom.frame[[i, 0]];
                }
                out.push((g, format!("{}: S¹ U(1) phase shift", atom.name)));
            }
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
            // Torus additionally carries `d` independent circle shifts: the unit
            // tangent advancing each axis's periodic phase (translation of that
            // circle coordinate), realised as motion along each frame column.
            if let AtomTopology::Torus { .. } = atom.topology {
                for a in 0..d {
                    let mut g = Array1::<f64>::zeros(p * d);
                    for i in 0..p {
                        g[i * d + a] = atom.frame[[i, a]];
                    }
                    out.push((g, format!("{}: Tᵈ circle shift axis {a}", atom.name)));
                }
            }
        }
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

/// Build global decoder output-frame rotation generators `O(output_dim)`: a
/// rotation `B ∈ so(output_dim)` acts on every atom's frame from the left
/// (`B · frame`). The induced tangent on the joint parameter vector stacks
/// `B · frame_k` per atom. We enumerate the full `so(output_dim)` basis — one
/// generator per unordered output-axis pair `(oi < oj)`, count
/// `output_dim·(output_dim−1)/2` — since the per-generator rank test treats each
/// independently and we want the certificate to find every output-frame freedom,
/// not a subset. `output_dim` is taken as the maximum frame row-count across
/// atoms; an atom whose frame lacks one of the two axes contributes nothing to
/// that generator.
fn frame_rotation_generators(model: &FittedSaeManifold) -> Vec<(Array1<f64>, String)> {
    let mut out: Vec<(Array1<f64>, String)> = Vec::new();
    let p = model
        .atoms
        .iter()
        .map(|a| a.frame.nrows())
        .max()
        .unwrap_or(0);
    let param_dim = model.param_dim();
    for oi in 0..p {
        for oj in (oi + 1)..p {
            let mut g = Array1::<f64>::zeros(param_dim);
            for (k, atom) in model.atoms.iter().enumerate() {
                let (ap, ad) = atom.frame.dim();
                if oi >= ap || oj >= ap {
                    continue;
                }
                let base = model.atom_offset(k);
                // (B · frame)[oi, c] = −frame[oj, c]; [oj, c] = +frame[oi, c].
                for c in 0..ad {
                    g[base + oi * ad + c] = -atom.frame[[oj, c]];
                    g[base + oj * ad + c] = atom.frame[[oi, c]];
                }
            }
            out.push((g, format!("output-frame rotation axes ({oi},{oj})")));
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
            let mut frob = 0.0_f64;
            for a in 0..d {
                for b in 0..d {
                    let mut g = 0.0;
                    for i in 0..p {
                        g += hn[(i * d + a) * d + e] * j_base[[row, i, b]];
                        g += j_base[[row, i, a]] * hn[(i * d + b) * d + e];
                    }
                    frob += g * g;
                }
            }
            max_curv_sq = max_curv_sq.max(frob);
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

/// Enumerate one atom's exact orbit coordinate-motion fields `δt ∈ ℝ^{n×d}`.
///
/// Supported charts are the ones the group acts on **linearly** (so the
/// first-order field is exact, not a linearisation): circle/torus axis shifts
/// (`δt = e_ax`, chart-free) and flat-patch `so(d)` rotations
/// (`δt_n = A_{ab} t_n`). The sphere's `so(3)` action on an intrinsic chart is
/// nonlinear, so sphere atoms stay on the frame path (the caller must not
/// build a view for them). Equal-ARD rotations reuse the rotation field for
/// the tied axis pairs (the ARD prior is their pinning channel).
fn exact_orbit_fields(
    atom: &FittedAtom,
    view: &AtomParameterView,
) -> Vec<(GeneratorFamily, Array2<f64>, String)> {
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
        AtomTopology::Sphere => {}
    }
    // Equal-ARD rotations between tied axes, on linearly-acting charts only.
    if !matches!(atom.topology, AtomTopology::Circle | AtomTopology::Sphere) {
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
    out
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

    let fields = exact_orbit_fields(atom, view);
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
            // mirroring the frame certificate's convention.
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
                let cost: f64 = image.iter().map(|v| v * v).sum();
                (cost / op.stiffness_sq).clamp(0.0, 1.0)
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

/// The stacked curvature root `R` of the pinning operator, in the fit's
/// metric: `(m, param_dim)` with `H = H_data + H_isometry = RᵀR`.
///
/// We assemble `R = [ W^{½} J ; R_isom ]` whose row space is
/// `range(H_data) + range(H_isometry)`, where `W^{½} J` is the metric-whitened
/// decoder Jacobian (the metric whitening is the `RowMetric`'s
/// `whiten_residual_row` applied to each output residual basis vector — i.e.
/// each Jacobian row is whitened in the same inner product the likelihood
/// sums). The caller derives both faces from this one object: the pinning
/// RANK (RRQR on `Rᵀ`, the audit's leverage-scaled rank decision) and the
/// per-generator relative curvature `‖R ξ̂‖² / σ_max(R)²` — magnitudes kept,
/// not orthonormalized away, so the statistic survives a full-rank span.
fn stacked_curvature_root(model: &FittedSaeManifold) -> Result<Array2<f64>, String> {
    let param_dim = model.param_dim();
    if param_dim == 0 {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }
    let p = model.metric.p_out();
    // Metric-whitened Jacobian rows: each row's Jacobian J_n ∈ ℝ^{p × param_dim}
    // is whitened to U_nᵀ J_n ∈ ℝ^{rank × param_dim} so that the resulting rows
    // span the same directions the metric-whitened residual gives cost to. We
    // build the stacked matrix `R` with one block of whitened rows per metric
    // row, then the isometry-penalty root beneath it.
    let mut stacked_rows: Vec<Array1<f64>> = Vec::new();
    for (n, j_flat) in model.jacobian_rows.iter().enumerate() {
        if j_flat.len() != p * param_dim {
            return Err(format!(
                "stacked_curvature_root: jacobian_rows[{n}] has len {} but expected p*param_dim = {}*{} = {}",
                j_flat.len(),
                p,
                param_dim,
                p * param_dim
            ));
        }
        // Whiten each parameter column's p-vector of output sensitivities.
        // Column c of J_n is the p-vector (j_flat[i*param_dim + c])_i. Whitening
        // it through the metric row (U_nᵀ ·) maps each column to a
        // `whit_len`-vector; the resulting `whit_len × param_dim` block's rows
        // are the metric-whitened Jacobian rows whose span the data gives cost
        // to. For Euclidean provenance `whiten_residual_row` is the identity, so
        // `whit_len == p` and the block is J_n unchanged (bit-for-bit the
        // isotropic data span).
        let mut cols_whitened: Vec<Vec<f64>> = Vec::with_capacity(param_dim);
        for c in 0..param_dim {
            let mut col = vec![0.0_f64; p];
            for i in 0..p {
                col[i] = j_flat[i * param_dim + c];
            }
            cols_whitened.push(model.metric.whiten_residual_row(n, ArrayView1::from(&col)));
        }
        let whit_len = cols_whitened.first().map_or(0, |c| c.len());
        for r in 0..whit_len {
            let mut row = Array1::<f64>::zeros(param_dim);
            for (c, col) in cols_whitened.iter().enumerate() {
                row[c] = col[r];
            }
            stacked_rows.push(row);
        }
    }
    // Append isometry-penalty root rows.
    if model.isometry_penalty_root.ncols() != 0 {
        if model.isometry_penalty_root.ncols() != param_dim {
            return Err(format!(
                "stacked_curvature_root: isometry_penalty_root has {} cols but param_dim = {param_dim}",
                model.isometry_penalty_root.ncols()
            ));
        }
        for r in 0..model.isometry_penalty_root.nrows() {
            stacked_rows.push(model.isometry_penalty_root.row(r).to_owned());
        }
    }
    if stacked_rows.is_empty() {
        return Ok(Array2::<f64>::zeros((0, param_dim)));
    }
    let m = stacked_rows.len();
    let mut r_mat = Array2::<f64>::zeros((m, param_dim));
    for (i, row) in stacked_rows.iter().enumerate() {
        r_mat.row_mut(i).assign(row);
    }
    Ok(r_mat)
}

enum CurvatureReduction {
    Root {
        pinning_rank: usize,
        sigma_max_sq: f64,
        root: Array2<f64>,
    },
    Gram {
        pinning_rank: usize,
        sigma_max_sq: f64,
        gram: Array2<f64>,
    },
}

impl CurvatureReduction {
    fn from_model(model: &FittedSaeManifold) -> Result<Self, String> {
        let root = stacked_curvature_root(model)?;
        if root.nrows() == 0 {
            return Ok(Self::Root {
                pinning_rank: 0,
                sigma_max_sq: 0.0,
                root,
            });
        }
        let r_t = root.t().to_owned();
        let rrqr = rrqr_with_permutation(&r_t, default_rrqr_rank_alpha())
            .map_err(|e| format!("residual_gauge: RRQR on Rᵀ failed: {e:?}"))?;
        let (_u, sv, _vt) = root
            .svd(false, false)
            .map_err(|e| format!("residual_gauge: SVD of curvature root failed: {e}"))?;
        let smax = sv.iter().cloned().fold(0.0_f64, f64::max);
        Ok(Self::Root {
            pinning_rank: rrqr.rank,
            sigma_max_sq: smax * smax,
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
        let sigma_max_sq = evals.iter().cloned().fold(0.0_f64, f64::max).max(0.0);
        let sigma_max = sigma_max_sq.sqrt();
        let rank_tol = default_rrqr_rank_alpha()
            * f64::EPSILON
            * (root_rows.max(param_dim).max(1) as f64)
            * sigma_max.max(1.0);
        let lambda_tol = rank_tol * rank_tol;
        let pinning_rank = evals
            .iter()
            .filter(|&&lambda| lambda.max(0.0) > lambda_tol)
            .count();
        Ok(Self::Gram {
            pinning_rank,
            sigma_max_sq,
            gram,
        })
    }

    fn pinning_rank(&self) -> usize {
        match self {
            Self::Root { pinning_rank, .. } | Self::Gram { pinning_rank, .. } => *pinning_rank,
        }
    }

    fn sigma_max_sq(&self) -> f64 {
        match self {
            Self::Root { sigma_max_sq, .. } | Self::Gram { sigma_max_sq, .. } => *sigma_max_sq,
        }
    }

    fn unit_generator_energy(&self, unit: &Array1<f64>) -> f64 {
        match self {
            Self::Root { root, .. } => {
                let r_xi = root.dot(unit);
                r_xi.iter().map(|c| c * c).sum::<f64>()
            }
            Self::Gram { gram, .. } => {
                let h_xi = gram.dot(unit);
                unit.dot(&h_xi).max(0.0)
            }
        }
    }
}

/// Evaluate the identifiability rank machinery on the symmetry generators of a
/// fitted SAE-manifold model and certify which gauge group the fit is identified
/// up to.
///
/// # Method
///
/// 1. Enumerate the symmetry generators as tangent directions on the flattened
///    decoder frames: per-atom `Isom(M_k)` generators
///    ([`atom_isometry_generators`]), equal-ARD rotations
///    ([`equal_ard_rotation_generators`]), global output-frame rotations
///    ([`frame_rotation_generators`]), and exchangeable-atom permutations
///    ([`atom_permutation_generators`]).
/// 2. Build the stacked curvature root `R` of the pinning operator
///    `H = H_data + H_isometry = RᵀR` in the fit's [`RowMetric`]
///    ([`stacked_curvature_root`]); the pinning RANK is the audit's RRQR rank
///    of `R`, reported alongside.
/// 3. For each generator `ξ`, the **relative curvature fraction**
///    `‖R ξ̂‖² / σ_max(R)²` measures the curvature the converged objective has
///    along the unit generator, relative to the model's stiffest direction.
///    `ξ` is **unpinned** (a residual gauge freedom) iff that fraction is at
///    or below the calibrated tolerance
///    `max(`[`GENERATOR_FLAT_ENERGY_TOL`]`, lowering_error_scale)` — flat up
///    to numerical noise and the mean-frame lowering's own resolution
///    ([`FittedAtom::lowering_error`], #995). Any larger fraction — including
///    the *mixed* regime where `ξ` carries both a curved and a flat component
///    — means the orbit costs objective, the exact group element is broken,
///    and the generator is **pinned**. (A span-membership or rank-increase
///    test degenerates when `R` is full-rank, which production fits always
///    are: every direction is "in the span", so verdicts would collapse to
///    all-pinned regardless of magnitudes. Keeping the curvature magnitudes
///    is what lets a genuinely flat direction stay visible inside a full-rank
///    span.) The fraction and the calibration scale are reported per
///    generator so partial flatness stays visible.
///
/// # Escalations
///
/// * When the isometry pin is inactive (`isometry_penalty_root` has no rows) the
///   report sets `diffeomorphism_unpinned = true`: with no metric pin the model
///   is only identified up to an arbitrary diffeomorphism of the latent
///   manifolds, so every isometry generator is a residual freedom.
/// * Under [`MetricProvenance::OutputFisher`] the `Sym(F)` permutation subgroup
///   is checked for triviality: every atom-exchange generator must be pinned
///   (the output-Fisher metric separates the atoms behaviorally). The result is
///   carried in `sym_f_trivial_under_output_fisher`.
pub fn residual_gauge(model: &FittedSaeManifold) -> Result<ResidualGaugeReport, String> {
    residual_gauge_inner(model, None, None)
}

/// The #998 full-resolution certificate: within-atom gauge families are
/// realised as **exact orbits** in the model's own (decoder, coordinate)
/// parameter space for every atom that supplies an [`AtomParameterView`],
/// while cross-atom families (output-frame rotations, atom permutations) and
/// any unviewed atom (e.g. spheres, whose chart action is nonlinear) keep the
/// frame-space path with its #995 lowering-error calibration.
///
/// For a viewed atom the compensated orbit is a data-null **by construction**
/// when the basis family is closed under the group action — the verdict
/// carries no calibration (`lowering_error_scale = 0`), the compensation
/// residual is the computed closure, and all pinning of true model-class
/// symmetries flows through the per-atom [`OrbitPenaltyOperator`] channel
/// (the isometry pin / ARD prior — rungs 2 and 4 of the #981 ladder).
///
/// `views` and `penalty_ops` are aligned with `model.atoms`; a `None` view
/// keeps that atom entirely on the frame path. Supplying a view for an atom
/// whose pin is active without also supplying its penalty operator would
/// over-claim freedom, so callers must pass the operator (or no view) for
/// pinned atoms.
pub fn residual_gauge_exact(
    model: &FittedSaeManifold,
    views: &[Option<AtomParameterView>],
    penalty_ops: &[Option<OrbitPenaltyOperator>],
) -> Result<ResidualGaugeReport, String> {
    let exact = residual_gauge_exact_inputs(model, views, penalty_ops)?;
    residual_gauge_inner(model, Some(exact), None)
}

/// Exact-orbit residual-gauge certificate with a pre-reduced streamed curvature
/// Gram `RᵀR`.
///
/// This is the memory-scaled entry point for callers that can stream their
/// metric-whitened Jacobian rows into the reductions the certificate consumes,
/// instead of retaining every per-row `p × param_dim` Jacobian block. The Gram
/// must include the same rows [`stacked_curvature_root`] would have placed in
/// `R`; `root_rows` is that row count for the rank tolerance scale.
pub fn residual_gauge_exact_from_curvature_gram(
    model: &FittedSaeManifold,
    views: &[Option<AtomParameterView>],
    penalty_ops: &[Option<OrbitPenaltyOperator>],
    curvature_gram: Array2<f64>,
    root_rows: usize,
) -> Result<ResidualGaugeReport, String> {
    let param_dim = model.param_dim();
    let curvature = CurvatureReduction::from_gram(curvature_gram, root_rows, param_dim)?;
    let exact = residual_gauge_exact_inputs(model, views, penalty_ops)?;
    residual_gauge_inner(model, Some(exact), Some(curvature))
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
        // Sphere charts: nonlinear group action — refuse exactness, keep the
        // calibrated frame path for this atom rather than pretending.
        if matches!(atom.topology, AtomTopology::Sphere) {
            continue;
        }
        exact_verdicts.extend(exact_orbit_verdicts(atom, view, penalty_ops[k].as_ref())?);
        mask[k] = true;
    }
    Ok((mask, exact_verdicts))
}

fn residual_gauge_inner(
    model: &FittedSaeManifold,
    exact: Option<(Vec<bool>, Vec<GeneratorVerdict>)>,
    precomputed_curvature: Option<CurvatureReduction>,
) -> Result<ResidualGaugeReport, String> {
    let metric_provenance = model.metric.provenance();
    let param_dim = model.param_dim();
    let (exact_mask, exact_verdicts) = match exact {
        Some((mask, verdicts)) => (Some(mask), verdicts),
        None => (None, Vec::new()),
    };

    // 1. Enumerate generators, tagged by family. The per-atom builders speak
    // the atom's LOCAL flattened-frame coordinates (length `frame.len()`); the
    // certificate's rank arithmetic runs in the joint parameter vector, so each
    // local generator is embedded at its atom's offset here. (Single-atom
    // models have local == joint, which is why only multi-atom models can
    // expose a missed embedding.)
    // Each generator carries its #995 lowering-error tolerance scale: the
    // largest `lowering_error` over the atoms it touches.
    let scale_of = |k: usize| -> f64 { model.atoms[k].lowering_error.clamp(0.0, 1.0) };
    let global_scale = (0..model.atoms.len()).map(scale_of).fold(0.0_f64, f64::max);
    let mut gens: Vec<(GeneratorFamily, Array1<f64>, String, f64)> = Vec::new();
    for (k, atom) in model.atoms.iter().enumerate() {
        // Atoms whose within-atom families are realised exactly (#998) are
        // skipped here: the frame-space lift of a compensated orbit measures
        // compression, not the symmetry, and the report must not carry both a
        // lossy and an exact verdict for the same group element.
        if exact_mask.as_ref().is_some_and(|mask| mask[k]) {
            continue;
        }
        let base = model.atom_offset(k);
        for (g, desc) in atom_isometry_generators(atom) {
            gens.push((
                GeneratorFamily::IsomAtom,
                embed_local_generator(base, &g, param_dim),
                desc,
                scale_of(k),
            ));
        }
        for (g, desc) in equal_ard_rotation_generators(atom) {
            gens.push((
                GeneratorFamily::EqualArdRotation,
                embed_local_generator(base, &g, param_dim),
                desc,
                scale_of(k),
            ));
        }
    }
    for (g, desc) in frame_rotation_generators(model) {
        // A global output rotation moves every atom's frame at once.
        gens.push((GeneratorFamily::FrameRotation, g, desc, global_scale));
    }
    for (g, desc, ka, kb) in atom_permutation_generators(model) {
        gens.push((
            GeneratorFamily::AtomPermutation,
            g,
            desc,
            scale_of(ka).max(scale_of(kb)),
        ));
    }

    // 2. Stacked curvature root in the metric; pinning rank via the audit's
    // RRQR on Rᵀ, stiffness scale σ_max via SVD (magnitudes kept).
    let curvature = match precomputed_curvature {
        Some(curvature) => curvature,
        None => CurvatureReduction::from_model(model)?,
    };
    let pinning_rank = curvature.pinning_rank();
    let sigma_max_sq = curvature.sigma_max_sq();

    // The isometry pin is inactive ⇒ diffeomorphism-unpinned escalation.
    let diffeomorphism_unpinned = model.isometry_penalty_root.nrows() == 0;

    // 3. Per-generator flatness verdict: relative curvature vs the calibrated
    // tolerance.
    let mut verdicts: Vec<GeneratorVerdict> = Vec::with_capacity(gens.len());
    for (family, g, description, lowering_error_scale) in &gens {
        let norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        // A structurally trivial generator (rotation of a rank-deficient frame,
        // zero swap) carries no direction — it cannot be a residual freedom.
        // Report it pinned with zero norm rather than as a spurious gauge.
        if norm <= f64::MIN_POSITIVE {
            verdicts.push(GeneratorVerdict {
                family: *family,
                description: description.clone(),
                unpinned: false,
                generator_norm: 0.0,
                pinned_energy_fraction: 1.0,
                lowering_error_scale: *lowering_error_scale,
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
            let unit = g.mapv(|v| v / norm);
            (curvature.unit_generator_energy(&unit) / sigma_max_sq).clamp(0.0, 1.0)
        };
        let tolerance = GENERATOR_FLAT_ENERGY_TOL.max(*lowering_error_scale);
        let unpinned = pinned_energy_fraction <= tolerance;
        verdicts.push(GeneratorVerdict {
            family: *family,
            description: description.clone(),
            unpinned,
            generator_norm: norm,
            pinned_energy_fraction,
            lowering_error_scale: *lowering_error_scale,
            provenance: VerdictProvenance::CurvatureTest,
        });
    }

    // Exact-orbit verdicts (#998) join the report on equal footing: the
    // group signature, residual dimension, and Sym(F) check all range over
    // the union.
    verdicts.extend(exact_verdicts);

    // #1019 — post-fit arc-length chart canonicalization records: for every
    // canonicalized d = 1 atom the continuous chart (reparameterization)
    // freedom is pinned BY CONSTRUCTION (the unit-speed representative of the
    // Diff(M) orbit was selected post-fit, image-frozen), so the certificate
    // records it pinned with the PinnedByCanonicalization provenance —
    // distinct from curvature/penalty pinning, since no objective resistance
    // was measured — and names the surviving FINITE isometry group of the
    // reference manifold. The group's continuous part (the circle's U(1)
    // shift) is still enumerated and curvature-tested above; this record is
    // the chart-freedom downgrade itself.
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
         pinning rank {pinning_rank}, {residual_gauge_dim} unpinned residual gauge \
         generator(s) of {} enumerated; group = {}{}{}",
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
    );
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

/// Produce the paired certificate for a fitted model: the residual-gauge
/// report computed here plus the anytime-valid structure certificate from
/// the discovery run's evidence ledger at level `alpha`. The ledger is the
/// one the structure search absorbed its shard evidence into
/// (`structure_evidence::StructureLedger`); certifying at any
/// data-dependent stopping time is sound — that is the ledger's whole
/// design.
pub fn dictionary_report(
    model: &FittedSaeManifold,
    ledger: &StructureLedger,
    alpha: f64,
) -> Result<DictionaryReport, String> {
    Ok(DictionaryReport {
        gauge: residual_gauge(model)?,
        structure: ledger.certify(alpha),
        transport_ladders: Vec::new(),
        atom_inference: atom_inference_reports(model),
    })
}

// --- #1100: closed-loop probe runner FFI ---------------------------------
// Top-level entry points exposing the steering→structure-evidence probe loop
// (`crate::inference::probe_runner::ProbeRunner`) beside `dictionary_report`, so
// the Python driver can design and absorb interventional probes against the same
// fitted term and evidence ledger the certificate is built from.

/// Design the next interventional probe for the most contested steerable claim
/// in `ledger`, against the fitted SAE-manifold `term` read through its per-row
/// output-Fisher `metric`.
///
/// Thin top-level wrapper over [`crate::inference::probe_runner::ProbeRunner::design_next`]:
/// it selects the contested claim furthest from certification, realizes candidate
/// latent moves of its atom through `crate::inference::steering::steer_delta`,
/// and routes their doses through
/// `gam_terms::inference::structure_evidence::plan_probe_for_contested_claim` to pick
/// the most discriminating one. The returned
/// [`crate::inference::probe_runner::RealizedProbe`] carries both the experiment
/// plan and the chosen intervention's on-manifold activation delta with its
/// dosimetry and validity radius.
pub fn design_probe(
    term: &SaeManifoldTerm,
    metric: &RowMetric,
    ledger: &StructureLedger,
) -> Result<RealizedProbe, String> {
    ProbeRunner { term, metric }.design_next(ledger)
}

/// Absorb a realized probe outcome into `ledger`, banking the delivered
/// behavioral dose (`realized_nats`, the observed output-Fisher KL of the steered
/// response) as anytime-valid evidence for the probe's claim.
///
/// Thin top-level wrapper over [`crate::inference::probe_runner::ProbeRunner::absorb`].
pub fn absorb_probe(
    term: &SaeManifoldTerm,
    metric: &RowMetric,
    ledger: &mut StructureLedger,
    probe: &RealizedProbe,
    realized_nats: f64,
) {
    ProbeRunner { term, metric }.absorb(ledger, probe, realized_nats);
}

/// Produce the paired certificate plus #1096 per-atom layer-transport ladders.
///
/// This is the strict wiring seam for callers that already have canonical
/// per-layer atom coordinates. It validates atom indices, topology/coordinate
/// lengths, finite coordinates, and the circle-period convention before calling
/// [`transport_ladder`]. Single-layer inputs are refused: no transport estimand
/// exists without at least one adjacent layer pair.
pub fn dictionary_report_with_transport_ladders(
    model: &FittedSaeManifold,
    ledger: &StructureLedger,
    alpha: f64,
    ladders: &[AtomTransportLadderInput],
) -> Result<DictionaryReport, String> {
    let mut report = dictionary_report(model, ledger, alpha)?;
    report.transport_ladders = atom_transport_ladder_reports(model, ladders)?;
    Ok(report)
}

/// Fit #1096 transport ladders for the supplied atom/layer coordinate blocks.
pub fn atom_transport_ladder_reports(
    model: &FittedSaeManifold,
    ladders: &[AtomTransportLadderInput],
) -> Result<Vec<AtomTransportLadderReport>, String> {
    let mut out = Vec::with_capacity(ladders.len());
    for input in ladders {
        let atom = model.atoms.get(input.atom_index).ok_or_else(|| {
            format!(
                "atom transport ladder index {} out of range for {} fitted atoms",
                input.atom_index,
                model.atoms.len()
            )
        })?;
        let depth = input.layers.len();
        if depth < 2 {
            return Err(format!(
                "atom transport ladder for atom {} ('{}') needs at least two layers, got {depth}",
                input.atom_index, atom.name
            ));
        }
        if input.coords.len() != depth || input.topologies.len() != depth {
            return Err(format!(
                "atom transport ladder for atom {} ('{}') has {} layers, {} coordinate blocks, {} topologies",
                input.atom_index,
                atom.name,
                depth,
                input.coords.len(),
                input.topologies.len()
            ));
        }

        let mut coords = Vec::with_capacity(depth);
        let mut topologies = Vec::with_capacity(depth);
        for (layer_pos, (coord, topology)) in
            input.coords.iter().zip(input.topologies.iter()).enumerate()
        {
            coords.push(canonical_coords_for_transport(
                coord,
                topology,
                input.atom_index,
                &atom.name,
                input.layers[layer_pos],
            )?);
            topologies.push(ChartTopology::from(topology));
        }

        let report = transport_ladder(&input.layers, &coords, &topologies).map_err(|e| {
            format!(
                "atom transport ladder for atom {} ('{}') failed: {e}",
                input.atom_index, atom.name
            )
        })?;
        out.push(AtomTransportLadderReport {
            atom_index: input.atom_index,
            atom_name: atom.name.clone(),
            report,
        });
    }
    Ok(out)
}

fn canonical_coords_for_transport(
    coords: &Array1<f64>,
    topology: &CanonicalChartTopology,
    atom_index: usize,
    atom_name: &str,
    layer: usize,
) -> Result<Array1<f64>, String> {
    if coords.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "atom transport ladder for atom {atom_index} ('{atom_name}') layer {layer} has non-finite coordinates"
        ));
    }
    match topology {
        CanonicalChartTopology::Circle { period } => {
            if !(period.is_finite() && *period > 0.0) {
                return Err(format!(
                    "atom transport ladder for atom {atom_index} ('{atom_name}') layer {layer} has invalid circle period {period}"
                ));
            }
            Ok(coords.mapv(|t| (t / *period) * TAU))
        }
        CanonicalChartTopology::Interval => Ok(coords.clone()),
    }
}

// ----------------------------------------------------------------------------
// #1102 cross-checkpoint atom-dynamics FFI entry (new top-level block).
// ----------------------------------------------------------------------------

/// Run #1102 cross-checkpoint Riesz-debiased atom-trajectory dynamics for the
/// fitted dictionary's atoms.
///
/// `decoder_grid` is `[n_checkpoints, n_atoms, n_grid, ambient_dim]` and
/// `atom_names`/`checkpoint_ids`/`latent_grid` label its axes; see
/// [`crate::inference::checkpoint_dynamics`] for the estimator and the honest
/// accounting of which Riesz inputs the bare grid supports. This entry binds
/// the atom axis to the fitted model: `atom_names` must name exactly the
/// model's atoms in order, so trajectories are reported against real atoms.
pub fn atom_checkpoint_dynamics(
    model: &FittedSaeManifold,
    decoder_grid: ndarray::ArrayView4<'_, f64>,
    checkpoint_ids: &[String],
    atom_names: &[String],
    latent_grid: ArrayView1<'_, f64>,
) -> Result<Vec<crate::inference::checkpoint_dynamics::AtomTrajectory>, String> {
    if atom_names.len() != model.atoms.len() {
        return Err(format!(
            "atom_checkpoint_dynamics: {} atom names supplied for {} fitted atoms",
            atom_names.len(),
            model.atoms.len()
        ));
    }
    for (idx, (supplied, fitted)) in atom_names.iter().zip(model.atoms.iter()).enumerate() {
        if supplied != &fitted.name {
            return Err(format!(
                "atom_checkpoint_dynamics: atom {idx} name '{supplied}' does not match fitted atom '{}'",
                fitted.name
            ));
        }
    }
    crate::inference::checkpoint_dynamics::checkpoint_atom_dynamics(
        &crate::inference::checkpoint_dynamics::CheckpointDynamicsInput {
            decoder_grid,
            checkpoint_ids,
            atom_names,
            latent_grid,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

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

    #[test]
    fn frame_inner_rotation_dim_is_sum_of_so_r_dims() {
        // dim O(r) = r(r−1)/2 per factored atom; rank-1 frames contribute 0.
        assert_eq!(frame_inner_rotation_dim(&[]), 0);
        assert_eq!(frame_inner_rotation_dim(&[1]), 0);
        assert_eq!(frame_inner_rotation_dim(&[2]), 1);
        assert_eq!(frame_inner_rotation_dim(&[4]), 6);
        assert_eq!(frame_inner_rotation_dim(&[1, 4, 8]), 0 + 6 + 28);
        assert_eq!(
            FrameInnerRotationGauge::from_ranks(vec![3, 3]).dim,
            6,
            "two rank-3 frames carry 2·3 inner-rotation dims"
        );
    }

    /// The #972 inner-rotation gauge is enumerated in the certificate, never
    /// curvature-tested: attaching it must not change any generator verdict
    /// or the residual_gauge_dim, but it MUST change the group signature and
    /// the summary — two replicate frame-factored fits agree on their gauge
    /// iff they also agree on this enumerated, convention-fixed part.
    #[test]
    fn frame_inner_rotation_attaches_to_the_certificate_without_verdict_change() {
        let base = ResidualGaugeReport {
            metric_provenance: MetricProvenance::Euclidean,
            generators: Vec::new(),
            pinning_rank: 5,
            residual_gauge_dim: 0,
            diffeomorphism_unpinned: false,
            sym_f_trivial_under_output_fisher: None,
            frame_inner_rotation: None,
            summary: "base".to_string(),
        };
        let sig_before = base.group_signature();
        let report = base.with_frame_inner_rotation(vec![1, 4, 8]);
        assert_eq!(
            report.frame_inner_rotation,
            Some(FrameInnerRotationGauge {
                per_atom_ranks: vec![1, 4, 8],
                dim: 34,
            })
        );
        // Verdict-side facts untouched.
        assert_eq!(report.residual_gauge_dim, 0);
        assert!(report.generators.is_empty());
        // Signature and summary carry the enumeration.
        let sig_after = report.group_signature();
        assert_ne!(sig_before, sig_after);
        assert!(sig_after.contains("frame-inner"), "got: {sig_after}");
        assert!(sig_after.contains("dim 34"), "got: {sig_after}");
        assert!(sig_after.contains("canonical-fixed"), "got: {sig_after}");
        assert!(report.summary.contains("inner-rotation gauge"));

        // A dictionary of rank-1 atoms has a zero-dimensional inner gauge:
        // enumerated (Some), but the signature is unchanged — there is
        // nothing to fix beyond the orientation sign convention.
        let trivial = ResidualGaugeReport {
            metric_provenance: MetricProvenance::Euclidean,
            generators: Vec::new(),
            pinning_rank: 0,
            residual_gauge_dim: 0,
            diffeomorphism_unpinned: false,
            sym_f_trivial_under_output_fisher: None,
            frame_inner_rotation: None,
            summary: "base".to_string(),
        };
        let sig_trivial_before = trivial.group_signature();
        let trivial = trivial.with_frame_inner_rotation(vec![1, 1, 1]);
        assert_eq!(
            trivial.frame_inner_rotation.as_ref().map(|g| g.dim),
            Some(0)
        );
        assert_eq!(trivial.group_signature(), sig_trivial_before);
        assert_eq!(trivial.summary, "base");
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
    fn piecewise_linear_eval_endpoints_and_midpoint() {
        let coeffs = array![[0.0_f64, 10.0], [1.0, 20.0], [2.0, 30.0]];
        let u = Array1::from(vec![0.0, 0.5, 1.0]);
        let out = piecewise_linear_eval(u.view(), coeffs.view(), 0.0, 1.0);
        assert!((out[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((out[[1, 0]] - 1.0).abs() < 1e-12);
        assert!((out[[2, 0]] - 2.0).abs() < 1e-12);
        assert!((out[[1, 1]] - 20.0).abs() < 1e-12);
    }

    #[test]
    fn select_weights_picks_max_evidence() {
        let rss = array![[10.0, 9.0, 9.5], [8.0, 4.0, 5.0], [9.0, 6.0, 7.0]];
        let pen = Array2::<f64>::zeros((3, 3));
        let l1 = Array1::from(vec![0.1, 1.0, 10.0]);
        let l2 = Array1::from(vec![0.1, 1.0, 10.0]);
        let res =
            identifiable_factor_select_weights(rss.view(), pen.view(), l1.view(), l2.view(), 80)
                .unwrap();
        assert_eq!((res.best_i, res.best_j), (1, 1));
        assert!((res.best_lam1 - 1.0).abs() < 1e-12);
        assert!((res.best_lam2 - 1.0).abs() < 1e-12);
        assert!(res.best_evidence.is_finite());
    }

    #[test]
    fn select_weights_breaks_ties_by_smallest_log_weight_sum() {
        let rss = Array2::<f64>::from_elem((2, 2), 4.0);
        let pen = Array2::<f64>::from_elem((2, 2), 1.0);
        let l1 = Array1::from(vec![0.1, 10.0]);
        let l2 = Array1::from(vec![0.1, 10.0]);
        let res =
            identifiable_factor_select_weights(rss.view(), pen.view(), l1.view(), l2.view(), 8)
                .unwrap();
        assert_eq!((res.best_i, res.best_j), (0, 0));
    }

    #[test]
    fn select_weights_rejects_shape_mismatch() {
        let rss = Array2::<f64>::zeros((2, 3));
        let pen = Array2::<f64>::zeros((2, 2));
        let l1 = Array1::from(vec![1.0, 1.0]);
        let l2 = Array1::from(vec![1.0, 1.0, 1.0]);
        let err =
            identifiable_factor_select_weights(rss.view(), pen.view(), l1.view(), l2.view(), 8)
                .unwrap_err();
        assert!(err.contains("penalty_grid"));
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
}
