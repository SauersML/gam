//! Canonical Laplace evidence, IFT cascade, and topology selection.
//!
//! This module is the single canonical entry point for:
//!
//!   1. Laplace evidence `V(ρ, T) = F + (1/2) log|H| - (1/2) log|S(ρ)|+
//!      - ((dim(H)-rank(S))/2) log(2π)`
//!      evaluated at the arrow-Schur inner-loop fixed point, per
//!      `proposals/arrow_schur_evidence.md` §3 (3.1 and the formula sheet
//!      in §7).
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
//!     §6.6). `predict_delta_t_from_delta_beta` and
//!     `predict_delta_t_from_delta_gt` already use the undamped factors.
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
use crate::solver::arrow_schur::{ArrowFactorCache, ArrowSchurSystem};

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
/// Maps directly to the requested set `{periodic, flat, sphere, torus}`
/// from `proposals/arrow_schur_evidence.md` §4.1. No additional variants
/// — the proposal §6.12 explicitly forbids carrying unused candidate
/// variants alongside the four-way selector.
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
                None => Err("evidence Hessian logdet requires exact factors or HVP fallback".into()),
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
    mut q: Vec<f64>,
    max_steps: usize,
) -> Result<f64, String> {
    let n = hvp.dim;
    let mut q_prev = vec![0.0_f64; n];
    let mut alphas = Vec::<f64>::with_capacity(max_steps);
    let mut betas = Vec::<f64>::with_capacity(max_steps.saturating_sub(1));
    let mut beta_prev = 0.0_f64;
    let tol = 1e-12_f64;

    for step in 0..max_steps {
        let applied = (hvp.apply)(&q);
        if applied.len() != n || applied.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "evidence HVP SLQ expected finite vector of length {n}, got {}",
                applied.len()
            ));
        }
        let mut w = applied;
        if step > 0 {
            for i in 0..n {
                w[i] -= beta_prev * q_prev[i];
            }
        }
        let alpha = dot_slice(&q, &w);
        if !alpha.is_finite() {
            return Err("evidence HVP SLQ produced non-finite alpha".to_string());
        }
        for i in 0..n {
            w[i] -= alpha * q[i];
        }
        let beta = norm2_slice(&w);
        alphas.push(alpha);
        if step + 1 == max_steps || beta <= tol {
            break;
        }
        if !beta.is_finite() {
            return Err("evidence HVP SLQ produced non-finite beta".to_string());
        }
        betas.push(beta);
        q_prev = q;
        q = w;
        for v in q.iter_mut() {
            *v /= beta;
        }
        beta_prev = beta;
    }

    let k = alphas.len();
    let mut tri = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        tri[[i, i]] = alphas[i];
        if i + 1 < k {
            tri[[i, i + 1]] = betas[i];
            tri[[i + 1, i]] = betas[i];
        }
    }
    let (evals, evecs) = tri
        .eigh(Side::Lower)
        .map_err(|e| format!("evidence HVP SLQ eigendecomposition failed: {e}"))?;
    let mut quad = 0.0_f64;
    for j in 0..k {
        let theta = evals[j];
        if !theta.is_finite() || theta <= 0.0 {
            return Err(format!(
                "evidence HVP SLQ expected SPD Hessian, Lanczos Ritz value {j} is {theta:.3e}"
            ));
        }
        let weight = evecs[[0, j]] * evecs[[0, j]];
        quad += weight * theta.ln();
    }
    Ok(quad)
}

#[inline]
fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
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
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
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
    let schur = cache.schur_factor.as_ref()?;

    let mut acc = 0.0_f64;
    // Per-row arrow blocks: log|H_uu_i| = 2 Σ log diag(L_i).
    for l in cache.undamped_factors_iter() {
        acc += 2.0 * log_det_from_chol_lower(l);
    }
    // Schur block: log|A| = 2 Σ log diag(L_schur).
    acc += 2.0 * log_det_from_chol_lower(schur);
    Some(acc)
}

/// Twice-the-diagonal-log sum for a lower-triangular Cholesky factor.
fn log_det_from_chol_lower(l: &Array2<f64>) -> f64 {
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
    let d = cache.d;
    let k = cache.k;
    if !cache.htbeta_available() {
        return Array2::<f64>::from_elem((n * d, k), f64::NAN);
    }
    let mut out = Array2::<f64>::zeros((n * d, k));
    let mut beta_basis = Array1::<f64>::zeros(k);
    let mut rhs = Array1::<f64>::zeros(d);
    for i in 0..n {
        let factor = cache.undamped_factor(i);
        // Solve H_uu_i Y = H_uβ_i column by column.
        for col in 0..k {
            beta_basis.fill(0.0);
            beta_basis[col] = 1.0;
            // The Tier-2 IFT assembler is built only when the family's
            // capability surface promises cached `H_tβ` row products.
            if !cache.apply_htbeta_row(i, beta_basis.view(), &mut rhs) {
                // SAFETY: reaching `false` means a family declared the cache
                // available but failed to populate it — contract violation.
                return Array2::<f64>::from_elem((n * d, k), f64::NAN);
            }
            let y = chol_lower_solve_vector(factor, &rhs);
            for c in 0..d {
                out[[i * d + c, col]] = -y[c];
            }
        }
    }
    out
}

/// Tier-2 IFT sensitivity `∂β*/∂ρ = -A⁻¹ ∂g_red/∂ρ` (proposal §2.4 /
/// §7).
///
/// `dg_red_drho` is the `K × R` matrix whose `a`-th column is `q_a =
/// ∂g_red/∂ρ_a`. Returns the `K × R` matrix `β_ρ`.
///
/// Returns `None` if the Schur factor is unavailable (PCG mode) or was
/// built from a damped operator; callers must not silently substitute an
/// approximation.
pub fn ift_dbeta_drho(
    cache: &ArrowFactorCache,
    dg_red_drho: ArrayView2<'_, f64>,
) -> Option<Array2<f64>> {
    if cache.ridge_t != 0.0 || cache.ridge_beta != 0.0 {
        return None;
    }
    let schur = cache.schur_factor.as_ref()?;
    let k = cache.k;
    let r = dg_red_drho.ncols();
    if dg_red_drho.nrows() != k {
        return None;
    }
    let mut out = Array2::<f64>::zeros((k, r));
    let mut rhs = Array1::<f64>::zeros(k);
    for a in 0..r {
        for row in 0..k {
            rhs[row] = dg_red_drho[[row, a]];
        }
        let x = chol_lower_solve_vector(schur, &rhs);
        for row in 0..k {
            out[[row, a]] = -x[row];
        }
    }
    Some(out)
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
    let d = cache.d;
    let k = cache.k;
    let r = dbeta_drho.ncols();
    if !cache.htbeta_available()
        || gu_rho.nrows() != n * d
        || gu_rho.ncols() != r
        || dbeta_drho.nrows() != k
    {
        return Array2::<f64>::from_elem((n * d, r), f64::NAN);
    }

    let mut out = Array2::<f64>::zeros((n * d, r));
    let mut rhs = Array1::<f64>::zeros(d);
    let mut htbeta_delta = Array1::<f64>::zeros(d);
    for a in 0..r {
        // Per-row: rhs_i = G_{u_i,ρ_a} + H_uβ_i · ∂β*/∂ρ_a.
        for i in 0..n {
            // Companion to the `du/dβ` assembler above; same H_tβ cache.
            if !cache.apply_htbeta_row(i, dbeta_drho.column(a), &mut htbeta_delta) {
                // SAFETY: `false` here means the family declared H_tβ row
                // products available but did not populate them — contract
                // violation against the joint-evidence capability surface.
                return Array2::<f64>::from_elem((n * d, r), f64::NAN);
            }
            for c in 0..d {
                rhs[c] = gu_rho[[i * d + c, a]] + htbeta_delta[c];
            }
            // u_ρ_i = -H_uu_i⁻¹ rhs_i, undamped factor.
            let v = chol_lower_solve_vector(cache.undamped_factor(i), &rhs);
            for c in 0..d {
                out[[i * d + c, a]] = -v[c];
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
    let d = cache.d;
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
        || huu_drho
            .iter()
            .any(|row| row.iter().any(|m| m.nrows() != d || m.ncols() != d))
        || htbeta_drho
            .iter()
            .any(|row| row.iter().any(|m| m.nrows() != d || m.ncols() != k))
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

    // Precompute Y_i = H_uu_i⁻¹ H_uβ_i (d × K). Used by both the Schur
    // derivative formula (§3.5) and the row trace `tr(H_uu_i⁻¹ ∂H_uu_i)`.
    let mut y_blocks: Vec<Array2<f64>> = Vec::with_capacity(n);
    let mut beta_basis = Array1::<f64>::zeros(k);
    let mut rhs = Array1::<f64>::zeros(d);
    for i in 0..n {
        let factor = cache.undamped_factor(i);
        let mut yi = Array2::<f64>::zeros((d, k));
        for col in 0..k {
            beta_basis.fill(0.0);
            beta_basis[col] = 1.0;
            // Same H_tβ cache contract as the IFT du/dβ and du/dρ paths.
            if !cache.apply_htbeta_row(i, beta_basis.view(), &mut rhs) {
                // SAFETY: `false` means the family declared the cache
                // available but did not populate it — contract violation.
                out.fill(f64::NAN);
                return out;
            }
            let v = chol_lower_solve_vector(factor, &rhs);
            for c in 0..d {
                yi[[c, col]] = v[c];
            }
        }
        y_blocks.push(yi);
    }

    // Outer-hoisted scratch reused across all (a, i) iterations.
    let mut trace_rhs = Array1::<f64>::zeros(d);
    let mut da_tmp = Array2::<f64>::zeros((d, k));
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
            let m_i = &huu_drho[i][a];
            debug_assert_eq!(m_i.shape(), &[d, d]);
            for col in 0..d {
                for r0 in 0..d {
                    trace_rhs[r0] = m_i[[r0, col]];
                }
                let v = chol_lower_solve_vector(cache.undamped_factor(i), &trace_rhs);
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
        debug_assert_eq!(da.shape(), &[k, k]);
        for i in 0..n {
            let dhtb = &htbeta_drho[i][a]; // d × K
            let yi = &y_blocks[i]; // d × K
            // - (∂H_uβ_i)ᵀ Y_i
            for r0 in 0..k {
                for c0 in 0..k {
                    let mut acc = 0.0;
                    for cc in 0..d {
                        acc += dhtb[[cc, r0]] * yi[[cc, c0]];
                    }
                    da[[r0, c0]] -= acc;
                }
            }
            // - Y_iᵀ (∂H_uβ_i)
            for r0 in 0..k {
                for c0 in 0..k {
                    let mut acc = 0.0;
                    for cc in 0..d {
                        acc += yi[[cc, r0]] * dhtb[[cc, c0]];
                    }
                    da[[r0, c0]] -= acc;
                }
            }
            // + Y_iᵀ (∂H_uu_i) Y_i
            let dhuu = &huu_drho[i][a];
            // tmp = (∂H_uu_i) Y_i  (d × K) — reuse hoisted da_tmp buffer.
            for r0 in 0..d {
                for c0 in 0..k {
                    let mut acc = 0.0;
                    for cc in 0..d {
                        acc += dhuu[[r0, cc]] * yi[[cc, c0]];
                    }
                    da_tmp[[r0, c0]] = acc;
                }
            }
            // da += Y_iᵀ tmp
            for r0 in 0..k {
                for c0 in 0..k {
                    let mut acc = 0.0;
                    for cc in 0..d {
                        acc += yi[[cc, r0]] * da_tmp[[cc, c0]];
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
            let v = chol_lower_solve_vector(schur, &col_scratch);
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
            c.converged && c.exclusion_reason.is_none() && c.negative_log_evidence.is_finite()
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
    // breaking ties by complexity_rank (smaller wins).
    valid.sort_by(|a, b| {
        topology_selection_score(a, options.score_scale)
            .partial_cmp(&topology_selection_score(b, options.score_scale))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.kind.complexity_rank().cmp(&b.kind.complexity_rank()))
    });

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
// Local linear-algebra utilities (copy of arrow_schur's private helpers).
//
// arrow_schur.rs keeps `cholesky_lower` / `chol_solve_vector` private to
// its module. We re-implement the solve here so this module does not
// modify arrow_schur.rs's public surface. The Cholesky decomposition
// itself is owned by the cache producer; we only need the solve.
// ---------------------------------------------------------------------------

fn chol_lower_solve_vector(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = l.nrows();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for kk in 0..i {
            sum -= l[[i, kk]] * y[kk];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for kk in (i + 1)..n {
            sum -= l[[kk, i]] * x[kk];
        }
        x[i] = sum / l[[i, i]];
    }
    x
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

    fn make_minimal_cache() -> ArrowFactorCache {
        // d = 1, k = 1, n = 1, H_uu_1 = [[2.0]] => L = [[sqrt(2)]],
        // H_uβ_1 = [[0.5]], A = 2 - 0.5 * 0.5 / 2 = 1.875.
        let l_huu = Array2::from_shape_vec((1, 1), vec![std::f64::consts::SQRT_2]).unwrap();
        let l_schur = Array2::from_shape_vec((1, 1), vec![(1.875_f64).sqrt()]).unwrap();
        let htbeta = Array2::from_shape_vec((1, 1), vec![0.5]).unwrap();
        ArrowFactorCache {
            htt_factors: std::sync::Arc::from(vec![l_huu]),
            htt_factors_undamped: crate::solver::arrow_schur::ArrowUndampedFactors::SameAsDamped,
            schur_factor: Some(l_schur),
            solver_mode: crate::solver::arrow_schur::ArrowSolverMode::Direct,
            ridge_t: 0.0,
            ridge_beta: 0.0,
            htbeta: crate::solver::arrow_schur::ArrowHtbetaCache::Dense {
                blocks: std::sync::Arc::from(vec![htbeta]),
                estimated_bytes: std::mem::size_of::<f64>(),
            },
            d: 1,
            k: 1,
            manifold_mode_fingerprint: 0,
            row_hessian_fingerprint: 0,
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
        let expected = 0.5 * (2.0_f64.ln() + 1.875_f64.ln())
            - 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((v - expected).abs() < 1e-12);
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
        let expected = 0.5 * (2.0_f64.ln() + 1.875_f64.ln())
            - 0.5 * (2.0 * std::f64::consts::PI).ln();
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
}
