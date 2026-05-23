//! Canonical Laplace evidence, IFT cascade, and topology selection.
//!
//! This module is the single canonical entry point for:
//!
//!   1. Laplace evidence `V(ρ, T) = F + (1/2) log|H| - (1/2) log|S(ρ)|+`
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
//!     `Σ_i log|H_uu_i|`. Likewise the Schur log-det must be of
//!     `A(0, 0) = H_ββ - Σ_i H_uβ_iᵀ H_uu_i⁻¹ H_uβ_i`, not the LM-damped
//!     surrogate.
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
//! **minimum** of that scalar (see `select_topology` below); equivalently
//! the caller can negate and `argmax`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::solver::arrow_schur::{ArrowFactorCache, ArrowSchurSystem};

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
    /// numerically tied. Default `1e-3` per proposal §4.6 examples.
    pub tie_tolerance: f64,
}

impl Default for TopologySelectOptions {
    fn default() -> Self {
        Self {
            tie_tolerance: 1e-3,
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
/// V(ρ, T) = F(β*, u*; ρ, T) + 0.5 log|H| - 0.5 log|S_pen(ρ)|+.
/// ```
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
/// this function recomputes internally from `cache`.
///
/// `residual_objective` is `F(β*, u*; ρ, T)` at the inner optimum. The
/// envelope theorem (proposal §3.2) makes this the only `F`-related
/// contribution.
///
/// # Errors
///
/// Returns `f64::NAN` if the cache lacks the undamped Schur factor
/// (`cache.schur_factor` was built under ridge damping or via PCG; see
/// proposal §6.5 — PCG does not give exact log-determinants).
pub fn laplace_evidence(
    cache: &ArrowFactorCache,
    penalty_log_det: f64,
    residual_objective: f64,
) -> f64 {
    let log_det_h = match arrow_log_det_from_cache(cache) {
        Some(v) => v,
        None => return f64::NAN,
    };
    residual_objective + 0.5 * log_det_h - 0.5 * penalty_log_det
}

/// Sum of per-row arrow log-determinants plus the Schur log-det.
///
/// `log|H| = Σ_i log|H_uu_i| + log|A|` using the undamped Cholesky
/// factors of `H_uu_i` and the cached Schur Cholesky factor.
///
/// Returns `None` if `cache.schur_factor` is absent (InexactPCG path —
/// proposal §6.5 says this must error rather than silently approximate)
/// or if a damped/incoherent cache is supplied. Callers that need a
/// stochastic estimator should request it explicitly, not through this
/// function.
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
    for l in cache.htt_factors_undamped.iter() {
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
    let n = cache.htt_factors_undamped.len();
    let d = cache.d;
    let k = cache.k;
    let mut out = Array2::<f64>::zeros((n * d, k));
    for i in 0..n {
        let factor = &cache.htt_factors_undamped[i];
        let htb = &cache.htbeta[i];
        // Solve H_uu_i Y = H_uβ_i column by column.
        for col in 0..k {
            let mut rhs = Array1::<f64>::zeros(d);
            for c in 0..d {
                rhs[c] = htb[[c, col]];
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
/// Returns `None` if the Schur factor is unavailable (PCG mode);
/// callers must not silently substitute an approximation.
pub fn ift_dbeta_drho(
    cache: &ArrowFactorCache,
    dg_red_drho: ArrayView2<'_, f64>,
) -> Option<Array2<f64>> {
    let schur = cache.schur_factor.as_ref()?;
    let k = cache.k;
    let r = dg_red_drho.ncols();
    debug_assert_eq!(dg_red_drho.nrows(), k);
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
    let n = cache.htt_factors_undamped.len();
    let d = cache.d;
    let k = cache.k;
    let r = dbeta_drho.ncols();
    debug_assert_eq!(gu_rho.shape(), &[n * d, r]);
    debug_assert_eq!(dbeta_drho.nrows(), k);

    let mut out = Array2::<f64>::zeros((n * d, r));
    let mut rhs = Array1::<f64>::zeros(d);
    for a in 0..r {
        // Per-row: rhs_i = G_{u_i,ρ_a} + H_uβ_i · ∂β*/∂ρ_a.
        for i in 0..n {
            for c in 0..d {
                let mut acc = gu_rho[[i * d + c, a]];
                for j in 0..k {
                    acc += cache.htbeta[i][[c, j]] * dbeta_drho[[j, a]];
                }
                rhs[c] = acc;
            }
            // u_ρ_i = -H_uu_i⁻¹ rhs_i, undamped factor.
            let v = chol_lower_solve_vector(&cache.htt_factors_undamped[i], &rhs);
            for c in 0..d {
                out[[i * d + c, a]] = -v[c];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// ∂V/∂ρ — analytic gradient via arrow trace formula (frozen-Hessian)
// ---------------------------------------------------------------------------

/// Per-`ρ` evidence gradient (frozen-Hessian Laplace mode per proposal
/// §3.7 / §3.8 split):
///
/// ```text
/// ∂V/∂ρ_a =
///       F_{ρ_a}                                  (value part)
///   + 0.5 tr(H⁻¹ H_{ρ_a})                        (direct Hessian)
///   - 0.5 tr(S_pen⁺ S_{pen,ρ_a})                 (penalty pseudo-logdet)
/// ```
///
/// The `tr(H⁻¹ H_{ρ_a})` trace is computed via the arrow structure
/// (proposal §3.5 / §3.10):
///
/// ```text
/// tr(H⁻¹ H_{ρ_a}) = Σ_i tr(H_uu_i⁻¹ ∂_{ρ_a} H_uu_i) + tr(A⁻¹ ∂_{ρ_a} A).
/// ```
///
/// The exact ExactLaplace mode (proposal §3.8 grad_h_ift_beta /
/// grad_h_ift_u parts) requires third-derivative callbacks. Per
/// proposal §3.8 and §6.6, that exact extension is left to a separate
/// signature; this function implements only the FrozenHessian branch
/// and returns the named split.
///
/// `value_rho[a] = F_{ρ_a}` (envelope theorem, proposal §3.2).
/// `huu_drho[i][a]` is `∂H_uu_i/∂ρ_a` as a `d × d` matrix.
/// `hbb_drho[a]` is `∂H_ββ/∂ρ_a` as a `K × K` matrix.
/// `htbeta_drho[i][a]` is `∂H_uβ_i/∂ρ_a` as a `d × K` matrix.
/// `pen_logdet_drho[a]` is `∂_{ρ_a} log|S_pen|+`.
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
) -> Array1<f64> {
    let r = value_rho.len();
    let n = cache.htt_factors_undamped.len();
    let d = cache.d;
    let k = cache.k;
    let mut out = Array1::<f64>::zeros(r);

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
    for i in 0..n {
        let factor = &cache.htt_factors_undamped[i];
        let mut yi = Array2::<f64>::zeros((d, k));
        let mut rhs = Array1::<f64>::zeros(d);
        for col in 0..k {
            for c in 0..d {
                rhs[c] = cache.htbeta[i][[c, col]];
            }
            let v = chol_lower_solve_vector(factor, &rhs);
            for c in 0..d {
                yi[[c, col]] = v[c];
            }
        }
        y_blocks.push(yi);
    }

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
            let mut rhs = Array1::<f64>::zeros(d);
            for col in 0..d {
                for r0 in 0..d {
                    rhs[r0] = m_i[[r0, col]];
                }
                let v = chol_lower_solve_vector(&cache.htt_factors_undamped[i], &rhs);
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
            // tmp = (∂H_uu_i) Y_i  (d × K)
            let mut tmp = Array2::<f64>::zeros((d, k));
            for r0 in 0..d {
                for c0 in 0..k {
                    let mut acc = 0.0;
                    for cc in 0..d {
                        acc += dhuu[[r0, cc]] * yi[[cc, c0]];
                    }
                    tmp[[r0, c0]] = acc;
                }
            }
            // da += Y_iᵀ tmp
            for r0 in 0..k {
                for c0 in 0..k {
                    let mut acc = 0.0;
                    for cc in 0..d {
                        acc += yi[[cc, r0]] * tmp[[cc, c0]];
                    }
                    da[[r0, c0]] += acc;
                }
            }
        }

        // tr(A⁻¹ ∂A) via column solves.
        let mut schur_trace_acc = 0.0_f64;
        let mut col = Array1::<f64>::zeros(k);
        for j in 0..k {
            for r0 in 0..k {
                col[r0] = da[[r0, j]];
            }
            let v = chol_lower_solve_vector(schur, &col);
            schur_trace_acc += v[j];
        }

        grad += 0.5 * (row_trace_acc + schur_trace_acc);

        // Part 3: -0.5 ∂_{ρ_a} log|S_pen|+.
        grad -= 0.5 * pen_logdet_drho[a];

        out[a] = grad;
    }
    out
}

// ---------------------------------------------------------------------------
// Topology selection
// ---------------------------------------------------------------------------

/// Enumerate the candidate topologies, rank by negative log evidence,
/// and return the winner. Failed/excluded candidates (proposal §6.11)
/// are appended at the end of `ranking` and are never the winner.
///
/// The caller fits each topology separately (proposal §4.2) and supplies
/// the resulting `TopologyCandidate` records. This function is purely
/// the discrete comparator + tie breaker.
///
/// # Tie-breaking
///
/// Per proposal §4.6: if `|V_a - V_b| <= tie_tolerance`, prefer the
/// simpler topology by `TopologyKind::complexity_rank` (flat < periodic
/// < sphere < torus). The `tie` flag in the result records whether such
/// a tie occurred at the top of the ranking.
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
        })
        .cloned()
        .collect();
    let mut excluded: Vec<TopologyCandidate> = candidates
        .iter()
        .filter(|c| {
            !(c.converged && c.exclusion_reason.is_none() && c.negative_log_evidence.is_finite())
        })
        .cloned()
        .collect();

    assert!(
        !valid.is_empty(),
        "select_topology: no finite valid candidates; proposal §6.11 forbids silent fallback"
    );

    // Sort by negative log evidence (ascending = best first), breaking
    // ties by complexity_rank (smaller wins).
    valid.sort_by(|a, b| {
        a.negative_log_evidence
            .partial_cmp(&b.negative_log_evidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.kind.complexity_rank().cmp(&b.kind.complexity_rank()))
    });

    // Detect numerical tie at the top.
    let tie = if valid.len() >= 2 {
        let top = valid[0].negative_log_evidence;
        let next = valid[1].negative_log_evidence;
        (next - top).abs() <= options.tie_tolerance
    } else {
        false
    };

    // If tied, prefer simpler topology among the tied prefix.
    if tie {
        let top_score = valid[0].negative_log_evidence;
        // Find the tied prefix range.
        let tied_end = valid
            .iter()
            .position(|c| (c.negative_log_evidence - top_score).abs() > options.tie_tolerance)
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

// ---------------------------------------------------------------------------
// Cache verification helpers
// ---------------------------------------------------------------------------

/// Sanity check used by callers before passing a cache to
/// [`laplace_evidence`] or [`evidence_grad_rho`]. Proposal §6.4 — ridges
/// must be zero on the evidence-evaluation path. Proposal §6.5 — PCG
/// caches lack the Schur factor and cannot supply exact logdets.
pub fn cache_supports_exact_evidence(cache: &ArrowFactorCache) -> bool {
    cache.ridge_t == 0.0 && cache.ridge_beta == 0.0 && cache.schur_factor.is_some()
}

/// Verifies the `ArrowSchurSystem` dimensions match the cache. Used as
/// a debug-time precondition; never silently masks shape errors
/// (proposal §6.9 — sign and shape errors must be loud).
pub fn cache_matches_system(cache: &ArrowFactorCache, sys: &ArrowSchurSystem) -> bool {
    cache.d == sys.d
        && cache.k == sys.k
        && cache.htbeta.len() == sys.rows.len()
        && cache.htt_factors_undamped.len() == sys.rows.len()
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
        let htbeta = vec![Array2::from_shape_vec((1, 1), vec![0.5]).unwrap()];
        ArrowFactorCache {
            htt_factors: vec![l_huu.clone()],
            htt_factors_undamped: vec![l_huu],
            schur_factor: Some(l_schur),
            solver_mode: crate::solver::arrow_schur::ArrowSolverMode::Direct,
            ridge_t: 0.0,
            ridge_beta: 0.0,
            htbeta,
            d: 1,
            k: 1,
        }
    }

    #[test]
    fn laplace_evidence_returns_finite_for_minimal_cache() {
        let cache = make_minimal_cache();
        // log|H| = log(2) + log(1.875), V = F + 0.5 log|H| - 0.5 log|S_pen|+.
        let v = laplace_evidence(&cache, 0.0, 0.0);
        assert!(v.is_finite());
        let expected = 0.5 * (2.0_f64.ln() + 1.875_f64.ln());
        assert!((v - expected).abs() < 1e-12);
    }

    #[test]
    fn laplace_evidence_nan_when_ridge_is_nonzero() {
        let mut cache = make_minimal_cache();
        cache.ridge_t = 1e-3;
        assert!(laplace_evidence(&cache, 0.0, 0.0).is_nan());
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
                converged: true,
                exclusion_reason: None,
            },
            TopologyCandidate {
                kind: TopologyKind::Sphere,
                negative_log_evidence: 8.0,
                converged: true,
                exclusion_reason: None,
            },
            TopologyCandidate {
                kind: TopologyKind::Torus,
                negative_log_evidence: f64::NAN,
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
                converged: true,
                exclusion_reason: None,
            },
            TopologyCandidate {
                kind: TopologyKind::Flat,
                negative_log_evidence: 5.0 + 1e-6,
                converged: true,
                exclusion_reason: None,
            },
        ];
        let sel = select_topology(&candidates, TopologySelectOptions::default());
        assert_eq!(sel.winner, TopologyKind::Flat);
        assert!(sel.tie);
    }
}
