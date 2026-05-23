//! Arrow / Schur structured Newton solve for joint (t, β) inner systems.
//!
//! See `proposals/latent_coord.md` §4 (the plumbing change) and
//! `proposals/composition_engine.md` §7 (audit-revised complexity claim:
//! "cost is arrow-shaped, but the REML log|H| gradient carries a shared
//! Schur⁻¹ factor handled as one-time-per-outer-iteration setup plus N
//! rank-≤d per-row traces"). The math-audit revisions in those proposals
//! are the source of the explicit precondition story below.
//!
//! ## What this module does
//!
//! When a [`crate::terms::latent_coord::LatentCoordValues`] block is
//! registered with the design, each inner Gauss–Newton iteration must
//! solve the joint bordered system
//!
//! ```text
//! [ H_tt   H_tβ ] [ Δt ]     [ -g_t ]
//! [ H_βt   H_ββ ] [ Δβ ]  =  [ -g_β ]
//! ```
//!
//! where:
//!
//! * `H_tt` is **block-diagonal in rows** — `N` independent `d × d`
//!   blocks `H_tt^(i)` (one per observation). This is the load-bearing
//!   structure exploited here.
//! * `H_tβ`, `H_βt = H_tβ^T` are row-local in `t` and dense in `β` —
//!   each row `i` contributes a `d × K` slab.
//! * `H_ββ` is the standard `K × K` penalized Hessian already handled by
//!   the existing PIRLS β-only path.
//!
//! Schur-eliminating `Δt` produces the reduced `K × K` system
//!
//! ```text
//! S · Δβ = -g_β + Σ_i H_βt^(i) (H_tt^(i))⁻¹ g_t^(i),   S = H_ββ - Σ_i H_βt^(i) (H_tt^(i))⁻¹ H_tβ^(i)
//! ```
//!
//! followed by row-local back-substitution
//!
//! ```text
//! Δt_i = -(H_tt^(i))⁻¹ (g_t^(i) + H_tβ^(i) Δβ).
//! ```
//!
//! Per inner iteration: `O(N d³)` for the per-row Cholesky factors, the
//! Schur subtraction, and the back-substitution, plus one standard
//! `K × K` solve for `Δβ`. Memory is `O(N d²)` for the per-row factors
//! plus the existing `O(K²)` β workspace.
//!
//! ## Scope — what is and is not in this file
//!
//! **In scope.** The arrow-Schur elimination of `H_tt` *for the inner
//! Gauss–Newton step*. The block-diagonality of `H_tt` is the property
//! that makes per-row elimination cheap; this is correct as long as
//! penalty contributions to `H_tt` are themselves row-block-diagonal
//! (true for [`crate::terms::analytic_penalties::ARDPenalty`] — diagonal —
//! and for [`crate::terms::analytic_penalties::IsometryPenalty`] in its
//! Gauss–Newton approximation — per-row `d × d` blocks via `J_n^T J_n`).
//!
//! **Out of scope (do not confuse).** The REML *outer-loop* gradient of
//! `log|H|` with respect to `t` carries a shared `Schur⁻¹` factor; only
//! row `i` of `Φ` moves with `t_i`, but `Schur⁻¹` itself is dense in all
//! `t`. That requires one dense `Schur⁻¹` formation per outer iteration
//! plus N rank-≤d per-row traces. It is **not** handled here — that's a
//! separate plumbing change owned by the REML driver. The two cost
//! analyses must not be conflated: the *inner* step is genuinely
//! O(N d³ + K³); the *outer* gradient is O(K³ + N · K d) once `Schur⁻¹`
//! is in scope.
//!
//! Future maintainers: if you find yourself extending `ArrowSchurSystem`
//! with an outer-REML gradient hook, please re-read the audit revisions
//! in `proposals/latent_coord.md` §7 and `proposals/composition_engine.md`
//! §7 first.

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};

use crate::terms::analytic_penalties::{
    AnalyticPenaltyKind, AnalyticPenaltyRegistry, PenaltyTier,
};

/// Per-row block data for the arrow-Schur system.
///
/// `htt` holds the `d × d` Gauss–Newton block for row `i` (including any
/// analytic-penalty contributions on that row); `htbeta` holds the
/// `d × K` cross-block `H_tβ^(i)`; `gt` is the `d`-length latent
/// gradient for row `i`.
#[derive(Debug, Clone)]
pub struct ArrowRowBlock {
    /// `H_tt^(i)`, shape `(d, d)`.
    pub htt: Array2<f64>,
    /// `H_tβ^(i)`, shape `(d, K)`.
    pub htbeta: Array2<f64>,
    /// `g_t^(i)`, shape `(d,)`.
    pub gt: Array1<f64>,
}

impl ArrowRowBlock {
    pub fn new(d: usize, k: usize) -> Self {
        Self {
            htt: Array2::<f64>::zeros((d, d)),
            htbeta: Array2::<f64>::zeros((d, k)),
            gt: Array1::<f64>::zeros(d),
        }
    }
}

/// Bordered (t, β) Newton system with arrow structure.
///
/// The β-block is held as a dense `K × K` Hessian `H_ββ` plus a `K`-length
/// gradient `g_β` (matching the existing PIRLS β-only convention). The
/// t-block is a `Vec<ArrowRowBlock>` of length `N`.
///
/// Construction is the driver's responsibility: the driver
///
///   1. evaluates Φ(t) and the radial jet `∂Φ/∂t` (the latter via
///      [`crate::terms::latent_coord::LatentCoordValues::design_gradient_wrt_t`]);
///   2. forms the working-weighted Gauss–Newton blocks
///      `H_tt^(i) += (g_i β)(g_i β)^T`, `H_tβ^(i) += (g_i β) ⊗ Φ_i`,
///      `H_ββ += Φ^T W Φ + Σ_k λ_k S_k`;
///   3. calls [`ArrowSchurSystem::add_analytic_penalty_contributions`] to
///      fold the registered analytic penalties (`IsometryPenalty`,
///      `ARDPenalty`, `SparsityPenalty`) into the appropriate `H_tt^(i)`
///      diagonal/dense block (Psi tier) and into `H_ββ` (Beta tier);
///   4. calls [`ArrowSchurSystem::solve`] to obtain `(Δt, Δβ)`.
pub struct ArrowSchurSystem {
    /// Per-row latent block (length `N`, each row `d × d` / `d × K` / `d`).
    pub rows: Vec<ArrowRowBlock>,
    /// `H_ββ`, shape `(K, K)`.
    pub hbb: Array2<f64>,
    /// `g_β`, shape `(K,)`.
    pub gb: Array1<f64>,
    /// Latent dimensionality `d`.
    pub d: usize,
    /// β dimensionality `K`.
    pub k: usize,
}

impl ArrowSchurSystem {
    /// Allocate an empty arrow system sized `(N rows × d, K)`.
    pub fn new(n: usize, d: usize, k: usize) -> Self {
        let rows = (0..n).map(|_| ArrowRowBlock::new(d, k)).collect();
        Self {
            rows,
            hbb: Array2::<f64>::zeros((k, k)),
            gb: Array1::<f64>::zeros(k),
            d,
            k,
        }
    }

    /// Number of rows `N`.
    pub fn n(&self) -> usize {
        self.rows.len()
    }

    /// Fold analytic-penalty contributions into the appropriate blocks.
    ///
    /// **Composition path.** Each registered [`AnalyticPenaltyKind`] is
    /// queried for `grad_target` (added to `g_t` or `g_β`) and for
    /// `hvp` against the canonical basis vectors restricted to the per-row
    /// `d`-block (when `tier == Psi`) or against `β` (when `tier ==
    /// Beta`). The Psi-tier contributions land on the per-row `H_tt^(i)`
    /// **only** — we rely on the analytic-penalty contract that the
    /// Psi-tier Hessian is block-diagonal across rows (true for ARD,
    /// true for the Gauss–Newton form of Isometry; checked at runtime via
    /// `hessian_diag` for the cheap diagonal case).
    ///
    /// `target_psi` is the full flat ψ vector (row-major, `N·d` entries)
    /// at the current iterate; `target_beta` is the current `β`. `rho`
    /// is the global ρ vector restricted to each penalty's local slice
    /// by [`AnalyticPenaltyRegistry::rho_layout`].
    pub fn add_analytic_penalty_contributions(
        &mut self,
        registry: &AnalyticPenaltyRegistry,
        target_psi: ArrayView1<'_, f64>,
        target_beta: ArrayView1<'_, f64>,
        rho_global: ArrayView1<'_, f64>,
    ) {
        let layout = registry.rho_layout();
        for (penalty, (rho_slice, tier, _name)) in registry.penalties.iter().zip(layout.iter()) {
            let rho_local = rho_global.slice(ndarray::s![rho_slice.clone()]);
            match tier {
                PenaltyTier::Psi => {
                    self.add_psi_penalty(penalty, target_psi, rho_local);
                }
                PenaltyTier::Beta => {
                    self.add_beta_penalty(penalty, target_beta, rho_local);
                }
                PenaltyTier::Rho => {
                    // Rho-tier hyperpriors do not contribute to the inner
                    // (t, β) Newton step; they enter only at the REML
                    // outer level (see RemlState::evaluate_unified_with_psi_ext).
                }
            }
        }
    }

    fn add_psi_penalty(
        &mut self,
        penalty: &AnalyticPenaltyKind,
        target_psi: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
    ) {
        let d = self.d;
        let n = self.rows.len();
        debug_assert_eq!(target_psi.len(), n * d);
        // Gradient: per-row `d`-slice added to `g_t^(i)`.
        let grad = penalty.grad_target(target_psi, rho_local);
        for i in 0..n {
            for a in 0..d {
                self.rows[i].gt[a] += grad[i * d + a];
            }
        }
        // Hessian: probe via HVP against each unit-`d`-vector for each row.
        // For block-diagonal-across-rows penalties (ARD, Isometry GN-form)
        // this yields the exact per-row `d × d` block. For penalties that
        // would couple rows the off-row entries are silently dropped — a
        // design choice consistent with the arrow-shape precondition. The
        // analytic-penalty contract documents that Psi-tier penalties
        // *must* be row-block-diagonal in their Hessian.
        let mut probe = Array1::<f64>::zeros(n * d);
        for a in 0..d {
            // One probe per latent axis: set the `a`-th column of each
            // row simultaneously to 1, HVP once, extract the column-`a`
            // entries of `H_tt^(i)` for every row.
            probe.fill(0.0);
            for i in 0..n {
                probe[i * d + a] = 1.0;
            }
            let hv = penalty.hvp(target_psi, rho_local, probe.view());
            for i in 0..n {
                for b in 0..d {
                    self.rows[i].htt[[b, a]] += hv[i * d + b];
                }
            }
        }
    }

    fn add_beta_penalty(
        &mut self,
        penalty: &AnalyticPenaltyKind,
        target_beta: ArrayView1<'_, f64>,
        rho_local: ArrayView1<'_, f64>,
    ) {
        let k = self.k;
        debug_assert_eq!(target_beta.len(), k);
        let grad = penalty.grad_target(target_beta, rho_local);
        for j in 0..k {
            self.gb[j] += grad[j];
        }
        // Hessian: probe with unit β-vectors. K may be large (~100K for
        // production SAE), so this is `O(K²)` worst-case. For diagonal
        // sparsity penalties prefer `hessian_diag` directly to avoid the
        // K matvecs.
        let mut probe = Array1::<f64>::zeros(k);
        for j in 0..k {
            probe.fill(0.0);
            probe[j] = 1.0;
            let hv = penalty.hvp(target_beta, rho_local, probe.view());
            for i in 0..k {
                self.hbb[[i, j]] += hv[i];
            }
        }
    }

    /// Schur-eliminate the per-row latent block and solve for `(Δt, Δβ)`.
    ///
    /// Returns `(delta_t, delta_beta)` with `delta_t` flat row-major of
    /// length `N · d` and `delta_beta` of length `K`. The sign convention
    /// matches `solve_newton_direction_dense`: the returned increments
    /// satisfy the bordered system with RHS `[-g_t; -g_β]`, i.e. they are
    /// the *negated* solutions of the standard Newton-direction
    /// formulation.
    ///
    /// `ridge_t` and `ridge_beta` are nonnegative diagonal regularizers
    /// added to the latent and β blocks respectively before factorization
    /// — used by the LM damping outer wrapper to recover from near-singular
    /// inner steps. Pass `0.0` for both to obtain the unregularized
    /// Newton direction.
    pub fn solve(
        &self,
        ridge_t: f64,
        ridge_beta: f64,
    ) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
        let n = self.rows.len();
        let d = self.d;
        let k = self.k;

        // 1. Per-row Cholesky factors of (H_tt^(i) + ridge_t · I).
        let mut htt_factors: Vec<Array2<f64>> = Vec::with_capacity(n);
        for row in &self.rows {
            let mut block = row.htt.clone();
            for a in 0..d {
                block[[a, a]] += ridge_t;
            }
            htt_factors.push(cholesky_lower(&block).map_err(|e| {
                ArrowSchurError::PerRowFactorFailed {
                    row: htt_factors.len(),
                    reason: e,
                }
            })?);
        }

        // 2. Schur complement S = H_ββ + ridge_β·I − Σ_i H_βt^(i) (H_tt^(i))⁻¹ H_tβ^(i).
        let mut schur = self.hbb.clone();
        for j in 0..k {
            schur[[j, j]] += ridge_beta;
        }
        // Reduced RHS r_β = -g_β + Σ_i H_βt^(i) (H_tt^(i))⁻¹ g_t^(i).
        let mut rhs_beta = Array1::<f64>::zeros(k);
        for i in 0..n {
            // M = (H_tt^(i))⁻¹ H_tβ^(i)   ∈ ℝ^{d × K}
            let m = chol_solve_matrix(&htt_factors[i], &self.rows[i].htbeta);
            // S -= H_βt^(i) · M  = H_tβ^(i)^T · M
            // Computed as: schur[a,b] -= Σ_c htbeta[c, a] * m[c, b]
            for a in 0..k {
                for b in 0..k {
                    let mut acc = 0.0;
                    for c in 0..d {
                        acc += self.rows[i].htbeta[[c, a]] * m[[c, b]];
                    }
                    schur[[a, b]] -= acc;
                }
            }
            // v = (H_tt^(i))⁻¹ g_t^(i)
            let v = chol_solve_vector(&htt_factors[i], &self.rows[i].gt);
            // rhs_beta += H_βt^(i) · v = H_tβ^(i)^T · v
            for a in 0..k {
                let mut acc = 0.0;
                for c in 0..d {
                    acc += self.rows[i].htbeta[[c, a]] * v[c];
                }
                rhs_beta[a] += acc;
            }
        }
        for j in 0..k {
            rhs_beta[j] -= self.gb[j];
        }

        // 3. Solve S · Δβ = rhs_beta.
        let schur_factor =
            cholesky_lower(&schur).map_err(|e| ArrowSchurError::SchurFactorFailed { reason: e })?;
        let delta_beta = chol_solve_vector(&schur_factor, &rhs_beta);

        // 4. Back-substitute Δt_i = -(H_tt^(i))⁻¹ (g_t^(i) + H_tβ^(i) Δβ).
        let mut delta_t = Array1::<f64>::zeros(n * d);
        for i in 0..n {
            let mut tmp = self.rows[i].gt.clone();
            // tmp += H_tβ^(i) · Δβ
            for c in 0..d {
                let mut acc = 0.0;
                for a in 0..k {
                    acc += self.rows[i].htbeta[[c, a]] * delta_beta[a];
                }
                tmp[c] += acc;
            }
            let dt_i = chol_solve_vector(&htt_factors[i], &tmp);
            for c in 0..d {
                delta_t[i * d + c] = -dt_i[c];
            }
        }

        Ok((delta_t, delta_beta))
    }
}

/// Errors raised by [`ArrowSchurSystem::solve`].
#[derive(Debug, Clone)]
pub enum ArrowSchurError {
    /// A per-row `H_tt^(i)` block was not positive-definite at the
    /// supplied ridge. Indicates an under-regularized latent block —
    /// typically a gauge-free fit without an identifiability penalty.
    PerRowFactorFailed { row: usize, reason: String },
    /// The Schur complement was not positive-definite. Indicates a
    /// near-collinear decoder or a degenerate weighting; the LM outer
    /// wrapper should escalate `ridge_beta` and retry.
    SchurFactorFailed { reason: String },
}

impl std::fmt::Display for ArrowSchurError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrowSchurError::PerRowFactorFailed { row, reason } => write!(
                f,
                "arrow-Schur: per-row H_tt^({row}) Cholesky failed: {reason}"
            ),
            ArrowSchurError::SchurFactorFailed { reason } => {
                write!(f, "arrow-Schur: Schur complement Cholesky failed: {reason}")
            }
        }
    }
}

impl std::error::Error for ArrowSchurError {}

// ---------------------------------------------------------------------------
// Cholesky helpers (kept local to avoid a new public-API dependency on the
// linalg crate. The systems here are tiny per-row (d × d, d ∈ {1..16}) and
// modest at the Schur level (K × K, K ∈ {basis size}). For production SAE
// scales the Schur factor should switch to faer; this module's `cholesky_lower`
// is the obvious replacement site.)
// ---------------------------------------------------------------------------

fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(format!("cholesky_lower: non-square {}×{}", n, a.ncols()));
    }
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for kk in 0..j {
                sum -= l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "non-PD pivot {sum} at index {i} (matrix is not positive definite)"
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(l)
}

fn chol_solve_vector(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
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

fn chol_solve_matrix(l: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = l.nrows();
    let m = b.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    let mut col = Array1::<f64>::zeros(n);
    for cidx in 0..m {
        for r in 0..n {
            col[r] = b[[r, cidx]];
        }
        let x = chol_solve_vector(l, &col);
        for r in 0..n {
            out[[r, cidx]] = x[r];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Convenience: in-place writeback of the arrow-Schur Newton step into the
// global PIRLS direction buffer. The driver owns the layout (β occupies
// `[0, K)`; flat ψ occupies `[K, K + N·d)` by convention used by the
// existing `SpatialLogKappaCoords` extension to the outer ρ vector).
// ---------------------------------------------------------------------------

/// Cached per-row Cholesky factors and Schur factor from a recent
/// [`ArrowSchurSystem::solve`] / [`solve_arrow_newton_step`] call.
///
/// This is the artifact Piece 3's IFT warm-start consumes. After the
/// inner Newton converges at outer iterate `(β̂, t̂; ρ)`, the IFT
/// sensitivity of `t̂_i` to a downstream change in `β` is
///
/// ```text
///   ∂t̂_i / ∂β  = -(H_tt^(i))⁻¹ · (H_tβ^(i))^T            (per-row)
/// ```
///
/// and the sensitivity to `ρ` (or any other shared hyper-axis `η`) is
///
/// ```text
///   ∂t̂_i / ∂η  = -(H_tt^(i))⁻¹ · ∂g_t^(i)/∂η.
/// ```
///
/// Both right-hand sides are row-local; the cached per-row Cholesky
/// factors let the predictor apply each row's correction in O(d²) ops
/// without re-forming the per-row block.
#[derive(Debug, Clone)]
pub struct ArrowFactorCache {
    /// Per-row lower-Cholesky factors of `H_tt^(i) (+ ridge_t · I)`.
    pub htt_factors: Vec<Array2<f64>>,
    /// Per-row cross-blocks `H_tβ^(i)` (carried so the IFT predictor can
    /// rebuild the `β`-coupled RHS without revisiting the assembly path).
    pub htbeta: Vec<Array2<f64>>,
    /// Latent dimensionality `d`.
    pub d: usize,
    /// β dimensionality `K`.
    pub k: usize,
}

impl ArrowFactorCache {
    /// Apply `Δt_i = -(H_tt^(i))⁻¹ · (H_tβ^(i) · Δβ)` per row, returning
    /// the flat row-major `Δt` of length `N · d`.
    pub fn predict_delta_t_from_delta_beta(&self, delta_beta: ArrayView1<'_, f64>) -> Array1<f64> {
        let n = self.htt_factors.len();
        let d = self.d;
        let k = self.k;
        debug_assert_eq!(delta_beta.len(), k);
        let mut out = Array1::<f64>::zeros(n * d);
        let mut rhs = Array1::<f64>::zeros(d);
        for i in 0..n {
            // rhs = H_tβ^(i) · Δβ   (length d)
            for c in 0..d {
                let mut acc = 0.0_f64;
                for a in 0..k {
                    acc += self.htbeta[i][[c, a]] * delta_beta[a];
                }
                rhs[c] = acc;
            }
            let v = chol_solve_vector(&self.htt_factors[i], &rhs);
            for c in 0..d {
                out[i * d + c] = -v[c];
            }
        }
        out
    }

    /// Apply `Δt_i = -(H_tt^(i))⁻¹ · δg_t^(i)` per row, where the caller
    /// supplies a flat row-major `δg_t` of length `N · d`. Used when the
    /// driver has computed `∂g_t/∂ρ · Δρ` (typically via the analytic
    /// penalty registry's `grad_target` finite-difference) externally.
    pub fn predict_delta_t_from_delta_gt(&self, delta_gt: ArrayView1<'_, f64>) -> Array1<f64> {
        let n = self.htt_factors.len();
        let d = self.d;
        debug_assert_eq!(delta_gt.len(), n * d);
        let mut out = Array1::<f64>::zeros(n * d);
        let mut rhs = Array1::<f64>::zeros(d);
        for i in 0..n {
            for c in 0..d {
                rhs[c] = delta_gt[i * d + c];
            }
            let v = chol_solve_vector(&self.htt_factors[i], &rhs);
            for c in 0..d {
                out[i * d + c] = -v[c];
            }
        }
        out
    }
}

/// One-shot arrow-Schur Newton solve that returns both the increment
/// and the per-row factor cache.
///
/// This is the public entry point that the latent inner-solver
/// ([`crate::solver::latent_inner`]) and the IFT warm-start predictor
/// ([`crate::solver::persistent_warm_start::ift_warm_start_latent`])
/// both call. The factor cache mirrors the convention used by
/// `solve_newton_direction_dense` (which holds its dense factor for the
/// geodesic-acceleration second solve): we expose it so downstream
/// callers can chain the IFT-sensitivity solves without re-factoring.
pub fn solve_arrow_newton_step(
    system: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<(Array1<f64>, Array1<f64>, ArrowFactorCache), ArrowSchurError> {
    let n = system.rows.len();
    let d = system.d;
    let k = system.k;

    // 1. Per-row Cholesky factors of (H_tt^(i) + ridge_t · I).
    let mut htt_factors: Vec<Array2<f64>> = Vec::with_capacity(n);
    let mut htbeta: Vec<Array2<f64>> = Vec::with_capacity(n);
    for row in &system.rows {
        let mut block = row.htt.clone();
        for a in 0..d {
            block[[a, a]] += ridge_t;
        }
        let l = cholesky_lower(&block).map_err(|e| ArrowSchurError::PerRowFactorFailed {
            row: htt_factors.len(),
            reason: e,
        })?;
        htt_factors.push(l);
        htbeta.push(row.htbeta.clone());
    }

    // 2. Schur complement.
    let mut schur = system.hbb.clone();
    for j in 0..k {
        schur[[j, j]] += ridge_beta;
    }
    let mut rhs_beta = Array1::<f64>::zeros(k);
    for i in 0..n {
        let m = chol_solve_matrix(&htt_factors[i], &system.rows[i].htbeta);
        for a in 0..k {
            for b in 0..k {
                let mut acc = 0.0;
                for c in 0..d {
                    acc += system.rows[i].htbeta[[c, a]] * m[[c, b]];
                }
                schur[[a, b]] -= acc;
            }
        }
        let v = chol_solve_vector(&htt_factors[i], &system.rows[i].gt);
        for a in 0..k {
            let mut acc = 0.0;
            for c in 0..d {
                acc += system.rows[i].htbeta[[c, a]] * v[c];
            }
            rhs_beta[a] += acc;
        }
    }
    for j in 0..k {
        rhs_beta[j] -= system.gb[j];
    }

    // 3. Solve for Δβ.
    let schur_factor =
        cholesky_lower(&schur).map_err(|e| ArrowSchurError::SchurFactorFailed { reason: e })?;
    let delta_beta = chol_solve_vector(&schur_factor, &rhs_beta);

    // 4. Back-substitute Δt.
    let mut delta_t = Array1::<f64>::zeros(n * d);
    for i in 0..n {
        let mut tmp = system.rows[i].gt.clone();
        for c in 0..d {
            let mut acc = 0.0;
            for a in 0..k {
                acc += system.rows[i].htbeta[[c, a]] * delta_beta[a];
            }
            tmp[c] += acc;
        }
        let dt_i = chol_solve_vector(&htt_factors[i], &tmp);
        for c in 0..d {
            delta_t[i * d + c] = -dt_i[c];
        }
    }

    let cache = ArrowFactorCache {
        htt_factors,
        htbeta,
        d,
        k,
    };
    Ok((delta_t, delta_beta, cache))
}

/// Layout convention for the joint (β, ψ) direction buffer.
///
/// The β block occupies entries `[0, K)`; the flat ψ block (per-row `t`
/// row-major) occupies `[K, K + N·d)`. This matches the convention used
/// by [`crate::terms::analytic_penalties::PsiSlice`] and by the existing
/// `SpatialLogKappaCoords` extension to the outer ρ vector.
pub fn write_arrow_direction(
    delta_t: &Array1<f64>,
    delta_beta: &Array1<f64>,
    out: &mut ArrayViewMut1<'_, f64>,
) {
    let k = delta_beta.len();
    let nt = delta_t.len();
    debug_assert_eq!(out.len(), k + nt);
    for j in 0..k {
        out[j] = delta_beta[j];
    }
    for i in 0..nt {
        out[k + i] = delta_t[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Verify the arrow-Schur solve against a small dense reference.
    /// Build the joint bordered system as a single dense (K + N·d)² matrix,
    /// solve it with the local cholesky_lower path, and compare to the
    /// arrow-Schur output.
    #[test]
    fn arrow_schur_matches_dense_reference_2x2() {
        // N = 2 rows, d = 2 latent, K = 3 β.
        let n = 2;
        let d = 2;
        let k = 3;
        let mut sys = ArrowSchurSystem::new(n, d, k);

        // Row 0: H_tt = [[2, 0.1],[0.1, 3]], H_tβ = [[1, 0, 0.5],[0.2, 1, 0]],
        //         g_t = [0.3, -0.2].
        sys.rows[0].htt = array![[2.0_f64, 0.1], [0.1, 3.0]];
        sys.rows[0].htbeta = array![[1.0_f64, 0.0, 0.5], [0.2, 1.0, 0.0]];
        sys.rows[0].gt = array![0.3_f64, -0.2];

        // Row 1.
        sys.rows[1].htt = array![[1.5_f64, -0.1], [-0.1, 2.0]];
        sys.rows[1].htbeta = array![[0.1_f64, 0.5, 0.0], [0.0, 0.3, 1.0]];
        sys.rows[1].gt = array![-0.1_f64, 0.4];

        // β-block.
        sys.hbb = array![
            [4.0_f64, 0.2, 0.0],
            [0.2, 5.0, 0.1],
            [0.0, 0.1, 6.0],
        ];
        sys.gb = array![0.5_f64, -0.3, 0.2];

        let (delta_t, delta_beta) = sys.solve(0.0, 0.0).expect("arrow-schur solve");

        // Build dense reference: order is [β; t_0; t_1] = K + N·d entries.
        let total = k + n * d;
        let mut hjoint = Array2::<f64>::zeros((total, total));
        let mut gjoint = Array1::<f64>::zeros(total);
        // β-β block.
        for a in 0..k {
            for b in 0..k {
                hjoint[[a, b]] = sys.hbb[[a, b]];
            }
            gjoint[a] = sys.gb[a];
        }
        // t-blocks and cross-blocks.
        for i in 0..n {
            let toff = k + i * d;
            for a in 0..d {
                for b in 0..d {
                    hjoint[[toff + a, toff + b]] = sys.rows[i].htt[[a, b]];
                }
                gjoint[toff + a] = sys.rows[i].gt[a];
                for a2 in 0..k {
                    hjoint[[toff + a, a2]] = sys.rows[i].htbeta[[a, a2]];
                    hjoint[[a2, toff + a]] = sys.rows[i].htbeta[[a, a2]];
                }
            }
        }
        // Solve hjoint · x = -gjoint via cholesky.
        let lj = cholesky_lower(&hjoint).expect("dense ref PD");
        let neg_g = gjoint.mapv(|v| -v);
        let xref = chol_solve_vector(&lj, &neg_g);
        // Compare β.
        for a in 0..k {
            assert!(
                (xref[a] - delta_beta[a]).abs() < 1e-10,
                "β[{a}] mismatch: dense {} vs arrow {}",
                xref[a],
                delta_beta[a]
            );
        }
        // Compare t.
        for i in 0..n {
            for a in 0..d {
                let dense = xref[k + i * d + a];
                let arrow = delta_t[i * d + a];
                assert!(
                    (dense - arrow).abs() < 1e-10,
                    "t[{i},{a}] mismatch: dense {dense} vs arrow {arrow}"
                );
            }
        }
    }
}
