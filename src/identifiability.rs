//! Identifiability primitives — frontier 2026 unification.
//!
//! Two analytic primitives that compose with the SAE-manifold infrastructure,
//! formalising the supervised-block plus free-discovery-block gauge-fix recipe
//! (a small number of supervised axes alongside additional free axes that
//! unsupervisedly align with downstream semantics):
//!
//! * `MechanismSparsityJacobian` — column-2-norm penalty on the decoder
//!   Jacobian: `Σ_k ||J_dec[:, k]||_2`. For an affine decoder `x = W t + b`,
//!   `J_dec = W` so this reduces to `Σ_k ||W[:, k]||_2`. Implements the
//!   mechanism-sparsity-identifiability theorem of Lachapelle (2401.04890).
//!   The penalty is smoothed by an `epsilon` to keep gradients defined at the
//!   origin: `ψ_k(W) = √(||W[:, k]||² + ε²) − ε`.
//!
//! * `ConditionalPriorIvae` — auxiliary-conditional Gaussian log-prior on the
//!   latent (Khemakhem iVAE, 2107.10098). Given per-row mean `μ_i(u_n)` and
//!   scale `σ_i(u_n)` evaluated at auxiliaries `u_n`, evaluates
//!   `−log p(t_n | u_n) = ½ Σ_{n,i} [ ((t_{n,i} − μ_{n,i}) / σ_{n,i})²
//!                                    + 2 log σ_{n,i} + log 2π ]`
//!   and returns its analytic gradient w.r.t. `t`.
//!
//! Both primitives are pure compute helpers — they take dense arrays and
//! return `(value, grad)` so they can be summed into any external optimiser
//! (PyTorch, JAX, the existing gam solver via a custom term, etc.) without
//! plumbing through the full dispatch registry. They are intentionally
//! standalone to keep the patch surface tight while parallel agents are
//! editing `analytic_penalties.rs`.

use crate::linalg::faer_ndarray::{FaerEigh, FaerQr, FaerSvd};
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
        let pos = ((val - u_min) / step).clamp(0.0, (k - 1) as f64 - 1e-12);
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(k - 1);
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
pub fn thin_svd_scores(
    x: ArrayView2<f64>,
    k: usize,
) -> Result<Array2<f64>, String> {
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

// ---------------------------------------------------------------------------
// Partial-supervision gauge-fix solver
// ---------------------------------------------------------------------------

/// Method for tying the supervised block to the auxiliary signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialSupervisionSupMethod {
    /// Orthogonal Procrustes: `min_{RᵀR=I} ‖T_sup R - aux‖_F²`.
    Procrustes,
    /// Affine least-squares pinned to `anchor_idx`.
    Anchor,
    /// Ridge map `A_λ = (TᵀT + λI)⁻¹ Tᵀaux` with GCV-selected λ.
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
            let lam_max = eigvals.iter().cloned().fold(0.0_f64, f64::max);
            let floor = (lam_max * 1.0e-10).max(1.0e-12);
            let top = (lam_max * 1.0e3).max(floor * 1.0e6);
            let grid_n: usize = 64;
            let log_floor = floor.ln();
            let log_top = top.ln();
            let mut best_score = f64::INFINITY;
            let mut best_lam = floor;
            let mut best_a = Array2::<f64>::zeros((d_sup, d_sup));
            for k in 0..grid_n {
                let frac = if grid_n == 1 {
                    0.0
                } else {
                    (k as f64) / ((grid_n - 1) as f64)
                };
                let lam = (log_floor + frac * (log_top - log_floor)).exp();
                let denom: Array1<f64> = eigvals.mapv(|v| v + lam);
                let mut a_eig = Array2::<f64>::zeros((d_sup, d_sup));
                for r in 0..d_sup {
                    for c in 0..d_sup {
                        a_eig[[r, c]] = ut_aux[[r, c]] / denom[r];
                    }
                }
                let a_lam = eigvecs.dot(&a_eig);
                let fitted = t_sup.dot(&a_lam);
                let resid = &fitted - &aux;
                let rss: f64 = resid.iter().map(|x| x * x).sum();
                let trace_h: f64 = (0..d_sup).map(|r| eigvals[r] / denom[r]).sum();
                let gcv_denom = (n as f64) - trace_h;
                if gcv_denom <= 0.0 || !rss.is_finite() {
                    continue;
                }
                let score = rss / (gcv_denom * gcv_denom);
                if score < best_score {
                    best_score = score;
                    best_lam = lam;
                    best_a = a_lam;
                }
            }
            if !best_score.is_finite() {
                return Err(
                    "partial_supervision_solve: GCV grid did not find a finite-score weight"
                        .to_string(),
                );
            }
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

    #[test]
    fn conditional_prior_ivae_zero_mean_unit_scale_matches_standard_gaussian() {
        let n = 4;
        let d = 3;
        let mean = Array2::<f64>::zeros((n, d));
        let scale = Array2::<f64>::ones((n, d));
        let pen = ConditionalPriorIvae::new(mean, scale, 1.0).unwrap();
        let t = Array2::<f64>::from_elem((n, d), 0.5);
        let (v, g) = pen.value_and_grad(t.view());
        // Expected ½ Σ t² + (n*d/2) log 2π
        let expected_quad: f64 = 0.5 * t.iter().map(|x| x * x).sum::<f64>();
        let expected = expected_quad + 0.5 * (n * d) as f64 * (2.0 * std::f64::consts::PI).ln();
        assert!((v - expected).abs() < 1e-9);
        for &gv in g.iter() {
            assert!((gv - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn conditional_prior_ivae_grad_matches_finite_diff() {
        let mean = array![[0.1_f64, -0.2], [0.3, 0.0]];
        let scale = array![[0.5_f64, 2.0], [1.0, 0.7]];
        let t = array![[0.4_f64, -0.1], [1.2, 0.6]];
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
        let res = identifiable_factor_select_weights(
            rss.view(), pen.view(), l1.view(), l2.view(), 80,
        )
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
        let res = identifiable_factor_select_weights(
            rss.view(), pen.view(), l1.view(), l2.view(), 8,
        )
        .unwrap();
        assert_eq!((res.best_i, res.best_j), (0, 0));
    }

    #[test]
    fn select_weights_rejects_shape_mismatch() {
        let rss = Array2::<f64>::zeros((2, 3));
        let pen = Array2::<f64>::zeros((2, 2));
        let l1 = Array1::from(vec![1.0, 1.0]);
        let l2 = Array1::from(vec![1.0, 1.0, 1.0]);
        let err = identifiable_factor_select_weights(
            rss.view(), pen.view(), l1.view(), l2.view(), 8,
        )
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
        let aux = array![
            [1.0_f64, 2.0],
            [-1.0, 0.5],
            [3.0, -2.0],
            [0.7, 1.2],
        ];
        let t_sup = array![
            [0.5_f64, 1.0],
            [-0.5, 0.25],
            [1.5, -1.0],
            [0.35, 0.6],
        ];
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
