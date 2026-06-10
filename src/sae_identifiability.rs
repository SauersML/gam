//! SAE identifiability primitives and partial-supervision gauge fixing.
//!
//! # Object 4 — the Certificate ([`residual_gauge`])
//!
//! The partial-supervision solver above *removes* gauge freedom by aligning to
//! auxiliary supervision. The certificate answers the dual question: after a fit
//! has converged, **which gauge group is the model identified up to?** It does
//! so by running the same penalty-aware RRQR rank machinery the cross-block
//! identifiability audit uses
//! ([`crate::solver::identifiability_audit::audit_identifiability`] /
//! [`crate::linalg::faer_ndarray::rrqr_with_permutation`]) — but on the
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
//! The RRQR decides exactly that "is this column in the span of the
//! higher-priority columns" question: we stack `[ P | G ]` with `P` an
//! orthonormal basis for `range(H)` (high gauge priority) and `G` the generator
//! columns (lower priority); a generator demoted past the rank threshold lies in
//! `span(P)` and is **pinned**, a generator that survives the pivot carries an
//! orthogonal-complement component and is an **unpinned** residual gauge freedom.
//!
//! The whole computation is performed in the inner product carried by the fit's
//! [`crate::inference::row_metric::RowMetric`]: the curvature span `P` is built
//! from the metric-whitened Jacobian, so the certificate's "computed in metric
//! X" line reads straight off [`crate::inference::row_metric::RowMetric::provenance`]
//! ([`crate::inference::row_metric::MetricProvenance`]) and cannot misreport —
//! there is only one metric object.

use crate::inference::row_metric::{MetricProvenance, RowMetric};
use crate::inference::structure_evidence::{StructureCertificate, StructureLedger};
use crate::linalg::faer_ndarray::{
    FaerEigh, FaerQr, FaerSvd, default_rrqr_rank_alpha, rrqr_with_permutation,
};
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

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
/// through the faer bridge in `crate::linalg::faer_ndarray`.
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
/// user-facing [`crate::terms::sae_manifold::SaeAtomBasisKind`] choice but
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
}

impl GeneratorFamily {
    fn label(self) -> &'static str {
        match self {
            GeneratorFamily::IsomAtom => "Isom(M_k)",
            GeneratorFamily::EqualArdRotation => "equal-ARD rotation",
            GeneratorFamily::FrameRotation => "frame rotation O(output_dim)",
            GeneratorFamily::AtomPermutation => "Sym(F) atom permutation",
        }
    }
}

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
    /// the isometry penalty gives it curvature (`ξ` demoted into `span(P)` by
    /// the RRQR).
    pub unpinned: bool,
    /// `‖ξ‖₂` of the realised tangent direction (0 ⇒ the generator was
    /// structurally trivial — e.g. a rotation of a rank-deficient frame — and
    /// is reported as pinned/absent, never as a spurious freedom).
    pub generator_norm: f64,
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
    /// Human-readable one-line summary.
    pub summary: String,
}

impl ResidualGaugeReport {
    /// The certified residual gauge group, as a compact string naming the
    /// surviving generator families and their multiplicities. Two replicate
    /// fits are "identified up to the same group" iff this string is equal.
    pub fn group_signature(&self) -> String {
        group_signature_of(&self.generators, self.diffeomorphism_unpinned)
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

fn atom_permutation_generators(model: &FittedSaeManifold) -> Vec<(Array1<f64>, String)> {
    let mut out: Vec<(Array1<f64>, String)> = Vec::new();
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
            out.push((g, format!("atom-exchange {} ↔ {}", a.name, b.name)));
        }
    }
    out
}

/// Orthonormal basis for the pinning span `range(H)` in the fit's metric.
///
/// `H = H_data + H_isometry`. We assemble the stacked root
/// `R = [ W^{½} J ; R_isom ]` whose row space is `range(H_data) + range(H_isometry)`
/// (since `H = RᵀR`), where `W^{½} J` is the metric-whitened decoder Jacobian
/// (the metric whitening is the `RowMetric`'s `whiten_residual_row` applied to
/// each output residual basis vector — i.e. each Jacobian row is whitened in the
/// same inner product the likelihood sums). An orthonormal basis for that row
/// space — equivalently for `range(H)` — is the kept-column space of an RRQR on
/// `Rᵀ`; we return it as `(param_dim, pinning_rank)`.
fn pinning_span_basis(model: &FittedSaeManifold) -> Result<Array2<f64>, String> {
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
                "pinning_span_basis: jacobian_rows[{n}] has len {} but expected p*param_dim = {}*{} = {}",
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
                "pinning_span_basis: isometry_penalty_root has {} cols but param_dim = {param_dim}",
                model.isometry_penalty_root.ncols()
            ));
        }
        for r in 0..model.isometry_penalty_root.nrows() {
            stacked_rows.push(model.isometry_penalty_root.row(r).to_owned());
        }
    }
    if stacked_rows.is_empty() {
        return Ok(Array2::<f64>::zeros((param_dim, 0)));
    }
    let m = stacked_rows.len();
    let mut r_mat = Array2::<f64>::zeros((m, param_dim));
    for (i, row) in stacked_rows.iter().enumerate() {
        r_mat.row_mut(i).assign(row);
    }
    // Orthonormal basis for row-space of R = range(H): RRQR on Rᵀ
    // (param_dim × m) reveals the rank AND names which `rank` columns of `Rᵀ`
    // are linearly independent (the leading entries of `column_permutation`).
    // This is the same penalty-aware, leverage-scaled rank decision the
    // identifiability audit uses — here applied to the curvature root.
    let r_t = r_mat.t().to_owned();
    let rrqr = rrqr_with_permutation(&r_t, default_rrqr_rank_alpha())
        .map_err(|e| format!("pinning_span_basis: RRQR on Rᵀ failed: {e:?}"))?;
    let rank = rrqr.rank;
    if rank == 0 {
        return Ok(Array2::<f64>::zeros((param_dim, 0)));
    }
    // Gather exactly the `rank` independent columns of `Rᵀ` named by the RRQR
    // pivot, then thin-QR that full-rank `(param_dim, rank)` block to get an
    // orthonormal basis of range(R) = range(H). Using the pivoted subset (not
    // the leading `rank` raw columns) is essential: the raw leading columns may
    // be rank-deficient, in which case a plain QR of them would NOT span
    // range(H).
    let mut r_t_indep = Array2::<f64>::zeros((param_dim, rank));
    for (out_c, &src_c) in rrqr.column_permutation[..rank].iter().enumerate() {
        r_t_indep.column_mut(out_c).assign(&r_t.column(src_c));
    }
    // The thin Q is (param_dim, rank); its columns are an orthonormal basis of
    // range(H). Guard the column count in case the bridge returns a wider Q.
    let q = r_t_indep
        .qr()
        .map_err(|e| format!("pinning_span_basis: thin QR of pivoted subset failed: {e:?}"))?
        .0;
    let kept = q.slice(s![.., ..rank.min(q.ncols())]).to_owned();
    Ok(kept)
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
/// 2. Build an orthonormal basis `P` for the pinning span `range(H)` =
///    `range(H_data) + range(H_isometry)` in the fit's [`RowMetric`]
///    ([`pinning_span_basis`]).
/// 3. For each generator `ξ`, stack `[ P | ξ ]` and run
///    [`rrqr_with_permutation`]: `ξ` is **pinned** iff the rank does not
///    increase (i.e. `ξ ∈ span(P)`, the generator demoted past the rank
///    threshold), **unpinned** (a residual gauge freedom) iff the rank
///    increases (its orthogonal-complement component survives). This is the same
///    penalty-aware, leverage-scaled rank decision
///    [`crate::solver::identifiability_audit::audit_identifiability`] makes on
///    design columns, applied to generators.
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
    let metric_provenance = model.metric.provenance();
    let param_dim = model.param_dim();

    // 1. Enumerate generators, tagged by family. The per-atom builders speak
    // the atom's LOCAL flattened-frame coordinates (length `frame.len()`); the
    // certificate's rank arithmetic runs in the joint parameter vector, so each
    // local generator is embedded at its atom's offset here. (Single-atom
    // models have local == joint, which is why only multi-atom models can
    // expose a missed embedding.)
    let mut gens: Vec<(GeneratorFamily, Array1<f64>, String)> = Vec::new();
    for (k, atom) in model.atoms.iter().enumerate() {
        let base = model.atom_offset(k);
        for (g, desc) in atom_isometry_generators(atom) {
            gens.push((
                GeneratorFamily::IsomAtom,
                embed_local_generator(base, &g, param_dim),
                desc,
            ));
        }
        for (g, desc) in equal_ard_rotation_generators(atom) {
            gens.push((
                GeneratorFamily::EqualArdRotation,
                embed_local_generator(base, &g, param_dim),
                desc,
            ));
        }
    }
    for (g, desc) in frame_rotation_generators(model) {
        gens.push((GeneratorFamily::FrameRotation, g, desc));
    }
    for (g, desc) in atom_permutation_generators(model) {
        gens.push((GeneratorFamily::AtomPermutation, g, desc));
    }

    // 2. Pinning span basis in the metric.
    let p_basis = pinning_span_basis(model)?;
    let pinning_rank = p_basis.ncols();

    // The isometry pin is inactive ⇒ diffeomorphism-unpinned escalation.
    let diffeomorphism_unpinned = model.isometry_penalty_root.nrows() == 0;

    // 3. Per-generator RRQR rank test against `[ P | ξ ]`.
    let mut verdicts: Vec<GeneratorVerdict> = Vec::with_capacity(gens.len());
    for (family, g, description) in &gens {
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
            });
            continue;
        }
        // Stack [ P | ξ ] (param_dim × (pinning_rank + 1)) and rank-test.
        let mut stacked = Array2::<f64>::zeros((param_dim, pinning_rank + 1));
        if pinning_rank > 0 {
            stacked.slice_mut(s![.., ..pinning_rank]).assign(&p_basis);
        }
        stacked.column_mut(pinning_rank).assign(g);
        let rrqr = rrqr_with_permutation(&stacked, default_rrqr_rank_alpha())
            .map_err(|e| format!("residual_gauge: RRQR on [P|ξ] failed: {e:?}"))?;
        // The generator is the trailing column. It is UNPINNED iff the stacked
        // rank strictly exceeds the pinning rank — i.e. ξ added a new direction
        // outside span(P). Because P is orthonormal of full column rank, that is
        // exactly `rank == pinning_rank + 1`.
        let unpinned = rrqr.rank > pinning_rank;
        verdicts.push(GeneratorVerdict {
            family: *family,
            description: description.clone(),
            unpinned,
            generator_norm: norm,
        });
    }

    let residual_gauge_dim = verdicts.iter().filter(|v| v.unpinned).count();

    // Sym(F)-triviality under OutputFisher provenance.
    let sym_f_trivial_under_output_fisher =
        if matches!(metric_provenance, MetricProvenance::OutputFisher { .. }) {
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

    Ok(ResidualGaugeReport {
        metric_provenance,
        generators: verdicts,
        pinning_rank,
        residual_gauge_dim,
        diffeomorphism_unpinned,
        sym_f_trivial_under_output_fisher,
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
    /// ([`crate::inference::structure_evidence::StructureLedger::certify`]).
    pub structure: StructureCertificate,
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

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
