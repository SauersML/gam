use super::*;

pub(crate) const TK_BLOCK_SIZE: usize = 128;

/// Upper bound on the parallel row-chunk length for the TK accumulation, so a
/// large `n / (4·threads)` split does not produce chunks so coarse that load
/// balancing across rayon workers suffers. Pairs with the [`TK_BLOCK_SIZE`]
/// lower bound. The `4×` oversubscription on the thread count keeps each worker
/// fed with several chunks for work stealing.
pub(crate) const TK_CHUNK_MAX_ROWS: usize = 2048;

pub(crate) const TK_CHUNK_OVERSUBSCRIBE: usize = 4;

pub(crate) const TK_MAX_OBSERVATIONS: usize = 20_000;

pub(crate) const TK_MAX_COEFFICIENTS: usize = 2_000;

pub(crate) const ADAPTIVE_KKT_ETA: f64 = 0.1;

pub(crate) const ADAPTIVE_KKT_FLOOR_REML_DIVISOR: f64 = 100.0;

pub(crate) const TK_MAX_DENSE_WORK: usize = 5_000_000;

// `n * p` catches bam-shaped tall designs while avoiding small-n wide problems.
pub(crate) const LARGE_N_EFS_THRESHOLD: f64 = 1.0e8;

pub(crate) const EFS_SINGLE_LOOP_PIRLS_SWEEPS: usize = 2;

pub(crate) const EFS_SINGLE_LOOP_PIRLS_CAP_SENTINEL: usize = usize::MAX / 4;

// Bail after 3 consecutive iterations whose surrogate/partial-inner drift is >= 10%.
pub(crate) const EFS_SINGLE_LOOP_BIAS_THRESHOLD: f64 = 0.10;

pub(crate) const EFS_SINGLE_LOOP_BIAS_CONSECUTIVE_LIMIT: usize = 3;

pub(crate) const HGB_INNER_FLOOR: f64 = 1e-12;

pub(crate) const HGB_LINEAR_FLOOR: f64 = 1e-12;

pub(crate) const HGB_TRACE_FLOOR: f64 = 1e-12;

pub(crate) const HGB_HISTORY_CAP: usize = 10;

pub(crate) const HGB_WARMUP_ITERS_MIN: usize = 3;

pub(crate) const HGB_WARMUP_ITERS_MAX: usize = 10;

pub(crate) const HGB_TARGET_FRACTION: f64 = 0.1;

// Treat log-lambda secant steps as local only up to a 1.0 log-scale step.
pub(crate) const HGB_SECANT_DRHO_MAX_SQUARED: f64 = 1.0;

pub(crate) const HGB_MIN_PAIRS_FOR_SENSITIVITY: usize = 3;

pub(crate) const HGB_REGRESSION_RIDGE: f64 = 1e-6;

pub(crate) const HGB_SENS_STABILITY_RATIO: f64 = 1.5;

pub(crate) const S_INNER_INIT: f64 = 1.0;

pub(crate) const S_LINEAR_INIT: f64 = 1.0;

pub(crate) const S_TRACE_INIT: f64 = 1.0;

pub(crate) const HGB_SENS_FLOOR: f64 = 1e-6;

pub(crate) const IFT_QUALITY_HISTORY_CAP: usize = 5;

/// Clamp bound on a linear predictor `eta` so `exp(eta)` cannot overflow f64
/// (`exp` overflows near `709`). Mirrors the canonical PIRLS `ETA_CLAMP`; kept
/// as a local const because that one is private to the `pirls` module. Used to
/// detect out-of-range η rows when materializing the logit fifth-derivative
/// channel (an out-of-range row contributes zero rather than a garbage jet).
pub(crate) const ETA_OVERFLOW_CLAMP: f64 = 700.0;

/// Rolling-quality bands and step-cap adjustment factors for the IFT step-cap
/// controller (`record_ift_prediction_quality`). `quality` is the relative
/// prediction residual averaged over the last [`IFT_QUALITY_HISTORY_CAP`]
/// predictions; below `GROW` the linearization is reliably excellent and the cap
/// is loosened, above `SHRINK` it is tightened, in between it is held. A rolling
/// quality at or above `FLAT_FALLBACK` flips the predictor to flat warm-start.
pub(crate) const IFT_QUALITY_GROW_BAND: f64 = 1e-3;

pub(crate) const IFT_QUALITY_SHRINK_BAND: f64 = 1e-1;

pub(crate) const IFT_QUALITY_FLAT_FALLBACK_BAND: f64 = 0.5;

pub(crate) const IFT_STEP_CAP_GROW_FACTOR: f64 = 1.5;

pub(crate) const IFT_STEP_CAP_SHRINK_FACTOR: f64 = 0.5;

// KKT residual acceptance tolerances for the active-set inner solver.
// Primal/dual/complementarity are checked at 1e-7 (matches the inner
// barrier-stopping tolerance used in PIRLS); stationarity uses a looser
// 5e-6 because the gradient is scaled by penalised Hessian curvature
// that can carry an extra ~order of magnitude of roundoff at convergence.
pub(crate) const KKT_TOL_PRIMAL: f64 = 1e-7;

pub(crate) const KKT_TOL_DUAL: f64 = 1e-7;

pub(crate) const KKT_TOL_COMP: f64 = 1e-7;

pub(crate) const KKT_TOL_STAT: f64 = 5e-6;

// Slack threshold below which a linear-inequality constraint Aβ ≥ b is
// considered active when extracting the constraint-free tangent basis.
// Chosen ~3 orders of magnitude above f64 roundoff on the dot product so
// constraints that just-touch within IRLS roundoff are correctly flagged.
pub(crate) const ACTIVE_CONSTRAINT_SLACK_TOL: f64 = 1e-8;

// Norm threshold for accepting a Gram–Schmidt residual as a basis
// direction when orthonormalising active-row vectors / null-space
// directions. One order of magnitude below ACTIVE_CONSTRAINT_SLACK_TOL
// because we are comparing squared-norm residuals after subtraction.
pub(crate) const ORTHONORM_DROP_TOL: f64 = 1e-10;

#[derive(Debug, Clone)]
pub(crate) struct AloStabilizationEval {
    pub(crate) cost: f64,
    pub(crate) gradient: Option<Array1<f64>>,
    pub(crate) k_hat: Option<f64>,
    pub(crate) max_leverage: f64,
    pub(crate) min_denominator: f64,
}

// --- ALO-stabilization constants (conservative stabilization choices) ---
//
// Every constant below is a deliberately conservative gate or weight whose only
// job is to keep the REML objective from being driven by a handful of
// near-singular leave-one-out denominators on a high-leverage Gaussian design.
// None of them are tunable (the repo bans CLI flags / env vars); they are fixed
// at values that leave well-conditioned fits bit-identical and only activate on
// genuine instability.
//
// Below this sample size the leave-one-out leverage estimate is too noisy to
// distinguish a genuinely influential point from sampling jitter, so the
// stabilization stays off entirely.
pub(crate) const ALO_STABILIZATION_MIN_N: usize = 20;

// Effective-dof fraction (edf / n) above which the design is treated as
// over-parameterized / near-interpolating and the ALO stabilization is
// suppressed.
//
// The stabilizer exists to robustify the REML criterion against a *handful* of
// genuinely influential observations on an *identified* design. On a
// near-saturated basis (edf approaching n — e.g. a tensor-product `te()` smooth
// whose marginal-product column count rivals n at small n), leverage is high
// for essentially *every* row purely from basis geometry, not from outliers.
// There the augmentation — whose mechanism is to pull λ upward until each row's
// LOO denominator clears the leverage barrier — can never satisfy its own gate
// (no finite λ drives an over-parameterized basis's leverage below 0.80 for all
// rows), so it adds a near-flat, ill-conditioned ridge to the outer surface.
// The outer optimizer then crawls that ridge to its iteration cap (the
// `[ALO-STABILIZED-REML]` "cost decreasing ~1e-4 per step over thousands of
// evals" / `min_denom≈0.043` signature of #813 / #821), re-running PIRLS every
// step. RKHS smooths (`duchon`/`matern`) regularize edf well below n and never
// reach this regime, which is exactly why `te()` was pathological while they
// were not on identical data. The 0.70 cut leaves genuinely identified,
// moderately-fit designs (where a few isolated rows carry the high leverage)
// fully stabilized while excluding the basis-saturation artifact.
pub(crate) const ALO_EDF_FRACTION_SATURATION: f64 = 0.70;

// Fraction of rows that may clear the leverage activation threshold before the
// high leverage is judged pervasive (a basis-geometry artifact) rather than
// concentrated in a few influential observations. Genuine influential-point
// stabilization touches a small minority of rows; a near-interpolating
// tensor-product basis trips a large fraction. Above this fraction the
// stabilizer is suppressed (see the pervasiveness guard in
// `alo_stabilization_eval`). 0.25 is well above the handful of rows a real
// outlier cluster produces yet far below the pervasive activation a saturated
// `te()` basis exhibits.
pub(crate) const ALO_PERVASIVE_LEVERAGE_FRACTION: f64 = 0.25;

// Suppress ALO when every ALO-triggering row is already high-leverage in the
// exact pure-parametric subdesign. Those directions are unpenalized, so no
// smoothness-parameter move can clear the leverage barrier (#862).
pub(crate) const ALO_PARAMETRIC_LEVERAGE_SHARE: f64 = 0.75;

// Activation gate on the leave-one-out denominator (1 - h). 0.20 means we only
// engage once some observation's LOO predictor is amplified by >5×; below that
// the correction is negligible and we preserve the unstabilized objective.
pub(crate) const ALO_DENOM_INSTABILITY_THRESHOLD: f64 = 0.20;

// Activation gate on raw leverage. 0.80 is the standard "very high leverage"
// rule-of-thumb cut (well above the 2p/n and 3p/n flags); only points past it
// can trip the stabilizer.
pub(crate) const ALO_MAX_LEVERAGE_THRESHOLD: f64 = 0.80;

// Weight on the smooth leverage barrier 0.5·τ·Σ(h - 0.80)₊². τ = 0.5 keeps the
// barrier a soft nudge that grows quadratically past the threshold rather than
// a hard wall, so the augmented objective stays smooth and differentiable.
pub(crate) const ALO_TAU: f64 = 0.5;

// Weight on the PSIS-reweighted Gaussian ALO deviance term. γ = 0.5 matches τ
// so the leverage barrier and the predictive-deviance term enter on equal
// conservative footing; neither dominates.
pub(crate) const ALO_GAMMA: f64 = 0.5;

// Saturation cap (in units of φ) on each observation's standardized squared
// leave-one-out deviance contribution w_i·(y_i − η̃_i)²/φ. PSIS bounds the
// *variance* of the importance weights but NOT the per-observation squared LOO
// residual, which for an isolated near-unit-leverage point is (y_i − η̂_i)/(1 −
// h_i): as 1 − h_i → 0 it explodes, dominating Σ D_ALO and dragging the
// selected λ upward (global over-smoothing) just to suppress a residual that is
// driven by basis geometry, not by model misfit — the model with that isolated
// point removed has no support there and its LOO prediction is intrinsically
// hopeless no matter how λ is chosen. We therefore pass each contribution
// through the smooth saturator g(d) = cap·tanh(d/cap): for a well-fit point d ≈
// 1 ≪ cap so g(d) ≈ d (the criterion is the ordinary LOO deviance), while a
// geometry-driven d ≫ cap saturates to ≈ cap so no single hopeless point can
// dominate the λ selection. cap = 9 is a robust 3-σ² cutoff on a standardized
// squared residual — well above the ~6.63 chi-square(1) 99th percentile, so
// every genuine bulk point is left untouched and only the pathological isolated
// rows are bounded. This bounds the *influence* of high-leverage points on the
// criterion (the stated design goal) without globally inflating λ.
pub(crate) const ALO_DEVIANCE_SATURATION: f64 = 9.0;

// Cap on n·p work for the analytic first-order gradient. Above this the dense
// H⁻¹Xᵀ solve is too expensive to justify per outer evaluation, so the
// stabilizer falls back to value-only augmentation (still bit-preserving the
// gate-off path).
pub(crate) const ALO_GRADIENT_MAX_WORK: usize = 4_000_000;

/// Shared factorization of the stabilized penalized Hessian, computed once on
/// the value path and threaded into the ALO ρ-gradient so the gradient never
/// re-materializes dense `X` or re-factorizes the same matrix (#862). The
/// inverse itself lives behind the one sensitivity operator (#935), so this
/// site holds no private H⁻¹ convention.
pub(crate) struct AloFactoredHessian<'a> {
    /// Dense transformed design `X` (n × p).
    pub(crate) x: &'a Array2<f64>,
    /// The fit's sensitivity operator over the stabilized penalized Hessian.
    pub(crate) sensitivity: &'a crate::sensitivity::FitSensitivity<'a>,
    /// `H⁻¹Xᵀ` (p × n), the column-solve the gradient reuses per observation.
    pub(crate) h_inv_xt: &'a Array2<f64>,
}

pub(crate) fn alo_leverage_barrier(h: f64) -> f64 {
    let excess = (h - ALO_MAX_LEVERAGE_THRESHOLD).max(0.0);
    excess * excess
}

pub(crate) fn alo_leverage_barrier_derivative(h: f64) -> f64 {
    if h > ALO_MAX_LEVERAGE_THRESHOLD {
        2.0 * (h - ALO_MAX_LEVERAGE_THRESHOLD)
    } else {
        0.0
    }
}

/// Raw standardized leave-one-out deviance contribution
/// d = w·(y − η̃)²/φ for one observation, before saturation.
pub(crate) fn gaussian_alo_raw_deviance(y: f64, eta_loo: f64, prior_weight: f64, phi: f64) -> f64 {
    let residual = y - eta_loo;
    prior_weight * residual * residual / phi.max(f64::MIN_POSITIVE)
}

/// Saturated per-observation Gaussian ALO deviance contribution
/// g(d) = cap·tanh(d/cap) with d the raw standardized squared LOO residual.
/// `g(d) ≈ d` for d ≪ cap and `g(d) → cap` for d ≫ cap, so an isolated
/// near-unit-leverage point whose LOO residual explodes from basis geometry
/// (not model misfit) contributes a bounded amount to the λ-selection
/// criterion instead of dragging λ up via global over-smoothing.
pub(crate) fn gaussian_alo_deviance(y: f64, eta_loo: f64, prior_weight: f64, phi: f64) -> f64 {
    let raw = gaussian_alo_raw_deviance(y, eta_loo, prior_weight, phi);
    ALO_DEVIANCE_SATURATION * (raw / ALO_DEVIANCE_SATURATION).tanh()
}

/// Saturator derivative g'(d) = 1 − tanh²(d/cap) evaluated at the raw
/// standardized squared LOO residual `raw`. Used to chain-rule the analytic
/// ρ-gradient of the saturated deviance term: ∂g(d_i)/∂η̃_i = g'(d_i)·∂d_i/∂η̃_i.
pub(crate) fn gaussian_alo_deviance_saturation_factor(raw: f64) -> f64 {
    let t = (raw / ALO_DEVIANCE_SATURATION).tanh();
    1.0 - t * t
}

pub(crate) fn transformed_penalty_matvec(
    penalty: &gam_terms::construction::CanonicalPenalty,
    beta: &Array1<f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(beta.len());
    let beta_block = beta.slice(ndarray::s![penalty.col_range.clone()]);
    let centered = &beta_block - &penalty.prior_mean;
    let local = penalty.local.dot(&centered);
    out.slice_mut(ndarray::s![penalty.col_range.clone()])
        .assign(&local);
    out
}

impl EvalShared {
    /// Canonical penalty scores `S_k β̂` at this bundle's inner mode
    /// `β̂ = pirls_result.beta_transformed`, computed once per inner solution
    /// and shared by every assemble call on the same bundle (exact hoist —
    /// see the field doc on `penalty_scores_at_mode`).
    ///
    /// `canonical_penalties` must be the owning `RemlState`'s fixed
    /// `canonical_penalties` slice; on a cache hit the stored length is
    /// checked against it so a frame mismatch fails loudly instead of
    /// silently feeding stale scores.
    pub(crate) fn canonical_penalty_scores_at_mode(
        &self,
        canonical_penalties: &[gam_terms::construction::CanonicalPenalty],
    ) -> Result<Arc<Vec<Array1<f64>>>, EstimationError> {
        if let Some(scores) = self.penalty_scores_at_mode.get() {
            if scores.len() != canonical_penalties.len() {
                return Err(EstimationError::LayoutError(format!(
                    "shared penalty-score cache mismatch: cached {} score vectors, \
                     requested {} canonical penalties",
                    scores.len(),
                    canonical_penalties.len()
                )));
            }
            return Ok(Arc::clone(scores));
        }
        let beta_hat = self.pirls_result.beta_transformed.as_ref();
        let scores = Arc::new(
            canonical_penalties
                .iter()
                .map(|pen| transformed_penalty_matvec(pen, beta_hat))
                .collect::<Vec<_>>(),
        );
        match self.penalty_scores_at_mode.set(Arc::clone(&scores)) {
            Ok(()) => Ok(scores),
            // A concurrent caller initialized the cell first; both vectors
            // were built from identical inputs (same β̂, same penalties) —
            // return the canonical winner so every consumer holds literally
            // the same allocation.
            Err(_) => Ok(Arc::clone(
                self.penalty_scores_at_mode
                    .get()
                    .expect("OnceLock set raced, so it is initialized"),
            )),
        }
    }
}

pub(crate) static OUTER_IFT_RESIDUAL_ENERGY: OnceLock<Mutex<HashMap<Vec<u64>, (f64, u64)>>> =
    OnceLock::new();

pub(crate) static OUTER_IFT_RESIDUAL_ENERGY_ITER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn outer_ift_residual_energy_cache() -> &'static Mutex<HashMap<Vec<u64>, (f64, u64)>> {
    OUTER_IFT_RESIDUAL_ENERGY.get_or_init(|| Mutex::new(HashMap::new()))
}

// `pub` (paired with `current_outer_iter` below) so the IFT design-cache
// memo-invalidation regression guard re-homed into gam-models by #1601 can drive
// the outer-iteration counter through the canonical `outer_eval` module path.
// Test-harness-only writer; the production caller is the in-crate outer loop.
pub fn record_current_outer_iter_for_ift(iter: u64) {
    OUTER_IFT_RESIDUAL_ENERGY_ITER.store(iter, Ordering::Relaxed);
}

pub fn current_outer_iter() -> u64 {
    OUTER_IFT_RESIDUAL_ENERGY_ITER.load(Ordering::Relaxed)
}

pub(crate) fn clear_outer_ift_residual_energy_for_fit() {
    if let Some(cache) = OUTER_IFT_RESIDUAL_ENERGY.get()
        && let Ok(mut cache) = cache.lock()
    {
        cache.clear();
    }
    OUTER_IFT_RESIDUAL_ENERGY_ITER.store(0, Ordering::Relaxed);
}

pub(crate) fn store_ift_residual_energy_for_outer_theta(theta: &Array1<f64>, energy: Option<f64>) {
    let Some(key) = super::rho_key::sanitized_rhokey(theta) else {
        return;
    };
    if let Ok(mut cache) = outer_ift_residual_energy_cache().lock() {
        if let Some(energy) = energy.filter(|energy| energy.is_finite() && *energy >= 0.0) {
            cache.insert(key, (energy, current_outer_iter()));
        } else {
            cache.remove(&key);
        }
    }
}

pub(crate) struct PenaltySubspace {
    pub(crate) evals: Array1<f64>,
    pub(crate) rank: usize,
}

pub(crate) struct HyperGradHistoryEntry {
    pub(crate) rho: Array1<f64>,
    pub(crate) g_outer: Array1<f64>,
    pub(crate) e_inner: f64,
    pub(crate) e_linear: f64,
    pub(crate) sigma_sq: f64,
    pub(crate) k: usize,
}

/// Online controller that allocates a target hypergradient-MSE budget across
/// the inner-PIRLS, linear-solve, and trace-probe channels using sensitivity
/// estimates derived from the observed `(Δρ, Δg_outer, E_inner, E_linear)`
/// history.
///
/// # Math
///
/// The mean-square error of the outer gradient `∇V` is decomposed by
/// channel as
///
/// ```text
///   MSE(∇V) ≈ s_inner²  · 2·E_inner
///           + s_linear² · 2·E_linear
///           + s_trace² / k,
/// ```
///
/// where `E_inner = ½ rᵀH⁻¹r` is the Newton-decrement energy of the inner
/// residual, `E_linear` is the analogous energy for the linear-solve
/// residual on the IFT mode-response system, and `k` is the trace-probe
/// count. The sensitivities `s_inner`, `s_linear`, `s_trace` are
/// re-estimated each outer iteration from finite-difference pairs of
/// `(Δρ, Δg_outer)` against per-channel energy proxies, then used to pick
/// per-channel tolerances/probe counts that hit `target_mse` while
/// respecting per-channel floors.
///
/// # Warmup
///
/// During warmup the history has too few stable finite-difference sensitivity
/// estimates to identify sensitivities reliably; the controller falls back to
/// the per-channel default tolerances and only switches to budgeted allocation
/// once enough stable estimates accumulate, or after the hard warmup cap.
pub(crate) struct HyperGradientBudget {
    /// Target MSE on `∇V` that the budgeted allocation aims to achieve.
    pub(crate) target_mse: f64,
    /// Minimum inner-channel energy `E_inner` the controller will request,
    /// guarding against unreachable inner tolerances.
    pub(crate) inner_floor: f64,
    /// Minimum linear-solve-channel energy `E_linear` the controller will
    /// request on the IFT mode-response system.
    pub(crate) linear_floor: f64,
    /// Minimum reciprocal trace-probe count (i.e., max `1/k`) the
    /// controller will request, capping the trace-probe budget.
    pub(crate) trace_floor: f64,
    /// Current sensitivity of `MSE(∇V)` with respect to inner-channel
    /// energy `E_inner`; re-estimated each iteration from history.
    pub(crate) s_inner: f64,
    /// Current sensitivity of `MSE(∇V)` with respect to linear-solve
    /// energy `E_linear` on the mode-response system.
    pub(crate) s_linear: f64,
    /// Current sensitivity of `MSE(∇V)` with respect to trace-probe
    /// variance (per-probe contribution).
    pub(crate) s_trace: f64,
    /// Bounded history of `(ρ, g_outer, E_inner, E_linear, σ², k)`
    /// observations used to fit `s_*` via finite differences.
    pub(crate) history: VecDeque<HyperGradHistoryEntry>,
    /// Recent successful sensitivity estimates used to decide when HGB warmup
    /// can safely engage.
    pub(crate) sensitivity_history: VecDeque<[f64; 3]>,
    pub(crate) warmup_engaged: bool,
}

impl HyperGradientBudget {
    pub(crate) fn new() -> Self {
        Self {
            target_mse: 0.0,
            inner_floor: HGB_INNER_FLOOR,
            linear_floor: HGB_LINEAR_FLOOR,
            trace_floor: HGB_TRACE_FLOOR,
            s_inner: S_INNER_INIT,
            s_linear: S_LINEAR_INIT,
            s_trace: S_TRACE_INIT,
            history: VecDeque::with_capacity(HGB_HISTORY_CAP),
            sensitivity_history: VecDeque::with_capacity(HGB_WARMUP_ITERS_MIN),
            warmup_engaged: false,
        }
    }

    pub(crate) fn push(&mut self, entry: HyperGradHistoryEntry) {
        self.history.push_back(entry);
        while self.history.len() > HGB_HISTORY_CAP {
            self.history.pop_front();
        }
    }

    pub(crate) fn previous_gradient_norm(&self) -> f64 {
        self.history
            .iter()
            .rev()
            .nth(1)
            .or_else(|| self.history.back())
            .map(|entry| l2_norm(&entry.g_outer))
            .filter(|norm| norm.is_finite())
            .unwrap_or(0.0)
    }

    pub(crate) fn reestimate_sensitivities(&mut self) -> Option<[f64; 3]> {
        let pairs = self.secant_gradient_pairs();
        if pairs.len() < HGB_MIN_PAIRS_FOR_SENSITIVITY {
            log::info!(
                "[HGB] small-sample fallback to defaults: pairs={}, threshold={}",
                pairs.len(),
                HGB_MIN_PAIRS_FOR_SENSITIVITY
            );
            return None;
        }
        let Some(s_inner) = self.estimate_energy_sensitivity(&pairs, |entry| entry.e_inner) else {
            return None;
        };
        let Some(s_linear) = self.estimate_energy_sensitivity(&pairs, |entry| entry.e_linear)
        else {
            return None;
        };
        let Some(s_trace) = self.estimate_trace_sensitivity() else {
            return None;
        };
        let sensitivities = [
            s_inner.max(HGB_SENS_FLOOR),
            s_linear.max(HGB_SENS_FLOOR),
            s_trace.max(HGB_SENS_FLOOR),
        ];
        self.s_inner = sensitivities[0];
        self.s_linear = sensitivities[1];
        self.s_trace = sensitivities[2];
        self.sensitivity_history.push_back(sensitivities);
        while self.sensitivity_history.len() > HGB_WARMUP_ITERS_MIN {
            self.sensitivity_history.pop_front();
        }
        Some(sensitivities)
    }

    pub(crate) fn estimate_energy_sensitivity<F>(
        &self,
        pairs: &[(Array1<f64>, Array1<f64>, usize)],
        energy: F,
    ) -> Option<f64>
    where
        F: Fn(&HyperGradHistoryEntry) -> f64,
    {
        let mut estimates = Vec::new();
        for i in 0..pairs.len() {
            let (drho_i, dg_i, left_idx) = &pairs[i];
            let rho_dim = drho_i.len();
            let grad_dim = dg_i.len();
            if rho_dim == 0 || grad_dim == 0 {
                continue;
            }
            let mut xtx = Array2::<f64>::zeros((rho_dim, rho_dim));
            for d in 0..rho_dim {
                xtx[[d, d]] = HGB_REGRESSION_RIDGE;
            }
            let mut xty = Array2::<f64>::zeros((rho_dim, grad_dim));
            let mut fit_pairs = 0usize;
            for (j, (drho_j, dg_j, _)) in pairs.iter().enumerate() {
                if i == j || drho_j.len() != rho_dim || dg_j.len() != grad_dim {
                    continue;
                }
                if drho_j.iter().any(|v| !v.is_finite()) || dg_j.iter().any(|v| !v.is_finite()) {
                    continue;
                }
                for row in 0..rho_dim {
                    for col in 0..rho_dim {
                        xtx[[row, col]] += drho_j[row] * drho_j[col];
                    }
                    for grad in 0..grad_dim {
                        xty[[row, grad]] += drho_j[row] * dg_j[grad];
                    }
                }
                fit_pairs += 1;
            }
            if fit_pairs == 0 {
                continue;
            }
            let Ok(chol) = xtx.cholesky(Side::Lower) else {
                continue;
            };
            chol.solve_mat_in_place(&mut xty);
            let mut predicted = Array1::<f64>::zeros(grad_dim);
            for grad in 0..grad_dim {
                let mut value = 0.0;
                for rho in 0..rho_dim {
                    value += drho_i[rho] * xty[[rho, grad]];
                }
                predicted[grad] = value;
            }
            let residual = dg_i - &predicted;
            let e0 = energy(&self.history[*left_idx]);
            let e1 = energy(&self.history[*left_idx + 1]);
            let denom_energy = e0.max(e1).max(1e-300);
            if !denom_energy.is_finite() || denom_energy < 0.0 {
                continue;
            }
            let estimate = l2_norm(&residual) / (2.0 * denom_energy).sqrt();
            if estimate.is_finite() && estimate > 0.0 {
                estimates.push(estimate);
            }
        }
        mean_positive(&estimates)
    }

    pub(crate) fn secant_gradient_pairs(&self) -> Vec<(Array1<f64>, Array1<f64>, usize)> {
        let entries: Vec<_> = self.history.iter().collect();
        let mut pairs = Vec::new();
        for i in 0..entries.len().saturating_sub(1) {
            let a = entries[i];
            let b = entries[i + 1];
            if a.rho.len() != b.rho.len() || a.g_outer.len() != b.g_outer.len() {
                continue;
            }
            let drho = &b.rho - &a.rho;
            let dg = &b.g_outer - &a.g_outer;
            let drho_norm_squared = drho.dot(&drho);
            if drho.iter().all(|v| v.is_finite())
                && dg.iter().all(|v| v.is_finite())
                && drho_norm_squared > 0.0
                && drho_norm_squared <= HGB_SECANT_DRHO_MAX_SQUARED
            {
                pairs.push((drho, dg, i));
            }
        }
        pairs
    }

    pub(crate) fn estimate_trace_sensitivity(&self) -> Option<f64> {
        let last_k = self.history.back()?.k;
        if last_k == 0 {
            return None;
        }
        let fixed: Vec<&HyperGradHistoryEntry> = self
            .history
            .iter()
            .rev()
            .take_while(|entry| entry.k == last_k)
            .collect();
        if fixed.len() < HGB_WARMUP_ITERS_MIN {
            return None;
        }
        if fixed
            .iter()
            .any(|entry| !entry.sigma_sq.is_finite() || entry.sigma_sq < 0.0)
        {
            return None;
        }
        let dim = fixed[0].g_outer.len();
        if dim == 0 || fixed.iter().any(|entry| entry.g_outer.len() != dim) {
            return None;
        }
        let mut means = Array1::<f64>::zeros(dim);
        for entry in fixed.iter() {
            means += &entry.g_outer;
        }
        means /= fixed.len() as f64;
        let mut variance_sum = 0.0;
        for entry in fixed.iter() {
            let diff = &entry.g_outer - &means;
            variance_sum += diff.dot(&diff);
        }
        let denom = ((fixed.len() - 1) * dim) as f64;
        let std = (variance_sum / denom).max(0.0).sqrt();
        (std.is_finite() && std > 0.0).then_some(std)
    }

    pub(crate) fn allocate_with_sensitivities(
        &self,
        s_inner: f64,
        s_linear: f64,
        s_trace: f64,
    ) -> (f64, f64, f64, [bool; 3]) {
        let floors = self.inner_floor + self.linear_floor + self.trace_floor;
        let usable = self.target_mse - floors;
        if usable <= 0.0 || !usable.is_finite() {
            log::warn!(
                "[HGB] target_mse below mandatory floors; target_mse={:.3e} floors={:.3e}",
                self.target_mse,
                floors
            );
            return (
                self.inner_floor,
                self.linear_floor,
                self.trace_floor,
                [true, true, true],
            );
        }
        // Sensitivity-weighted water filling: split the budget by s^2, then
        // add mandatory floors. This monotone rule is deliberately simple and
        // keeps every channel funded even when one sensitivity dominates.
        let wi = s_inner * s_inner;
        let wl = s_linear * s_linear;
        let wt = s_trace * s_trace;
        let sum = (wi + wl + wt).max(HGB_SENS_FLOOR * HGB_SENS_FLOOR);
        (
            self.inner_floor + usable * wi / sum,
            self.linear_floor + usable * wl / sum,
            self.trace_floor + usable * wt / sum,
            [false, false, false],
        )
    }

    pub(crate) fn sensitivities_stable(&self) -> bool {
        if self.sensitivity_history.len() < HGB_WARMUP_ITERS_MIN {
            return false;
        }
        for channel in 0..3 {
            let mut min_recent = f64::INFINITY;
            let mut max_recent: f64 = 0.0;
            for sensitivities in self.sensitivity_history.iter() {
                let value = sensitivities[channel];
                if !value.is_finite() || value <= 0.0 {
                    return false;
                }
                min_recent = min_recent.min(value);
                max_recent = max_recent.max(value);
            }
            if max_recent / min_recent >= HGB_SENS_STABILITY_RATIO {
                return false;
            }
        }
        true
    }
}

pub(crate) struct HyperGradientRuntimeState {
    pub(crate) budget: HyperGradientBudget,
    pub(crate) adaptive_kkt_override: Option<f64>,
    pub(crate) trace_state: Arc<Mutex<super::reml_outer_engine::StochasticTraceState>>,
}

impl HyperGradientRuntimeState {
    pub(crate) fn new() -> Self {
        Self {
            budget: HyperGradientBudget::new(),
            adaptive_kkt_override: None,
            trace_state: Arc::new(Mutex::new(
                super::reml_outer_engine::StochasticTraceState::default(),
            )),
        }
    }
}

pub(crate) static HYPERGRADIENT_BUDGETS: OnceLock<
    Mutex<HashMap<usize, HyperGradientRuntimeState>>,
> = OnceLock::new();

pub(crate) fn hypergradient_budgets() -> &'static Mutex<HashMap<usize, HyperGradientRuntimeState>> {
    HYPERGRADIENT_BUDGETS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Default)]
pub(crate) struct IftQualityRuntimeState {
    pub(crate) quality_history: Vec<f64>,
    pub(crate) next_step_cap: Option<f64>,
    pub(crate) fallback_next_flat: bool,
}

pub(crate) static IFT_QUALITY_STATES: OnceLock<Mutex<HashMap<usize, IftQualityRuntimeState>>> =
    OnceLock::new();

pub(crate) fn ift_quality_states() -> &'static Mutex<HashMap<usize, IftQualityRuntimeState>> {
    IFT_QUALITY_STATES.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone)]
pub(crate) struct IftModeResponseRuntimeCache {
    pub(crate) rho: Array1<f64>,
    pub(crate) rho_mode_response_cols: Option<Array2<f64>>,
    pub(crate) ext_mode_response_cols: Option<Array2<f64>>,
}

pub(crate) static IFT_MODE_RESPONSE_CACHES: OnceLock<
    Mutex<HashMap<usize, IftModeResponseRuntimeCache>>,
> = OnceLock::new();

pub(crate) fn ift_mode_response_caches()
-> &'static Mutex<HashMap<usize, IftModeResponseRuntimeCache>> {
    IFT_MODE_RESPONSE_CACHES.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone)]
pub(crate) struct IftJointModeResponseRuntimeCache {
    pub(crate) theta: Array1<f64>,
    pub(crate) rho_dim: usize,
    pub(crate) beta_original: Array1<f64>,
    pub(crate) mode_response_cols: Array2<f64>,
    pub(crate) active_constraints: bool,
}

pub(crate) static IFT_JOINT_MODE_RESPONSE_CACHES: OnceLock<
    Mutex<HashMap<usize, IftJointModeResponseRuntimeCache>>,
> = OnceLock::new();

pub(crate) fn ift_joint_mode_response_caches()
-> &'static Mutex<HashMap<usize, IftJointModeResponseRuntimeCache>> {
    IFT_JOINT_MODE_RESPONSE_CACHES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn joint_ift_cache_matches_theta(
    cache: &IftJointModeResponseRuntimeCache,
    theta: &Array1<f64>,
    new_rho: &Array1<f64>,
) -> bool {
    if cache.theta.len() <= cache.rho_dim
        || theta.len() != cache.theta.len()
        || new_rho.len() != cache.rho_dim
    {
        return false;
    }
    for i in 0..cache.rho_dim {
        if theta[i].to_bits() != new_rho[i].to_bits() {
            return false;
        }
    }
    for i in cache.rho_dim..theta.len() {
        if theta[i].to_bits() != cache.theta[i].to_bits() {
            return false;
        }
    }
    true
}

thread_local! {
    pub(crate) static IFT_LATEST_OUTER_THETA: std::cell::RefCell<Option<Array1<f64>>> =
        const { std::cell::RefCell::new(None) };

    pub(crate) static IFT_LATEST_OUTER_RHO_UPPER_BOUNDS: std::cell::RefCell<Option<Array1<f64>>> =
        const { std::cell::RefCell::new(None) };
}

pub(crate) fn record_current_outer_theta_for_ift(theta: &Array1<f64>) {
    let value = if theta.is_empty() || theta.iter().any(|v| !v.is_finite()) {
        None
    } else {
        Some(theta.clone())
    };
    IFT_LATEST_OUTER_THETA.with(|slot| *slot.borrow_mut() = value);
}

pub(crate) fn record_current_outer_rho_upper_bounds_for_ift(upper: &Array1<f64>) {
    let value = if upper.is_empty() || upper.iter().any(|v| !v.is_finite()) {
        None
    } else {
        Some(upper.clone())
    };
    IFT_LATEST_OUTER_RHO_UPPER_BOUNDS.with(|slot| *slot.borrow_mut() = value);
}

pub(crate) fn latest_outer_rho_upper_bounds_for_ift() -> Option<Array1<f64>> {
    IFT_LATEST_OUTER_RHO_UPPER_BOUNDS.with(|slot| slot.borrow().clone())
}

pub(crate) fn latest_outer_theta_for_ift() -> Option<Array1<f64>> {
    IFT_LATEST_OUTER_THETA.with(|slot| slot.borrow().clone())
}

pub(crate) fn l2_norm(values: &Array1<f64>) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub(crate) fn mean_positive(values: &[f64]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &value in values {
        if value.is_finite() && value > 0.0 {
            sum += value;
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f64)
}

#[derive(Default)]
pub(crate) struct EfsSingleLoopBiasGuardState {
    pub(crate) owner: usize,
    pub(crate) consecutive: usize,
}

// `LazyLock` (not `OnceLock` lazy init) so the init closure never parks
// callers on the OS condvar. The init body here is trivial — just a default-
// constructed `Mutex` — but the call site sits in a module that elsewhere
// dispatches rayon parallel iterators, and the codebase-level lint
// (see `tests/once_lock_get_or_init_not_inside_parallel_regions.rs`)
// forbids the lazy `OnceLock` accessor in any rayon-adjacent file.
// `LazyLock`'s initializer runs at first deref under its own dedicated
// synchronization that does not interact with rayon's worker pool.
pub(crate) static EFS_SINGLE_LOOP_BIAS_GUARD: LazyLock<Mutex<EfsSingleLoopBiasGuardState>> =
    LazyLock::new(|| Mutex::new(EfsSingleLoopBiasGuardState::default()));

#[inline]
pub(crate) fn compute_gradient_for_tk(mode: super::reml_outer_engine::EvalMode) -> bool {
    mode != super::reml_outer_engine::EvalMode::ValueOnly
}

#[inline]
pub(crate) fn efs_single_loop_encoded_cap() -> usize {
    EFS_SINGLE_LOOP_PIRLS_CAP_SENTINEL + EFS_SINGLE_LOOP_PIRLS_SWEEPS
}

#[inline]
pub(crate) fn decode_efs_single_loop_cap(raw_cap: usize) -> Option<usize> {
    // `.then_some` evaluates its argument eagerly, so the subtraction must be
    // guarded by `.then(|| ...)` to avoid usize underflow when raw_cap <
    // SENTINEL (the common path for non-EFS-single-loop iterates).
    (raw_cap >= EFS_SINGLE_LOOP_PIRLS_CAP_SENTINEL)
        .then(|| raw_cap - EFS_SINGLE_LOOP_PIRLS_CAP_SENTINEL)
        .filter(|cap| *cap > 0)
}

/// Apply the screening residual penalty to a cost.
///
/// Under multi-start seed screening (a ranking pass over candidate ρ
/// vectors), the inner P-IRLS is intentionally capped at a few iterations.
/// Partial modes that do not certify stationarity are accepted for ranking
/// in `execute_pirls_if_needed`; this helper turns the partial cost into a
/// finite ranking score
///
/// ```text
/// C_screen(s) = C_approx(s) + ½ · r_g(s)² ,
/// ```
///
/// where r_g = ‖g‖ / (1 + ‖score‖ + ‖Sβ‖ + ridge·‖β‖) is the scale-invariant
/// relative gradient residual. Using r_g rather than the absolute ‖g‖ keeps
/// the penalty meaningful at large-scale n: the absolute score grows as O(√n),
/// so an absolute residual term would swamp the actual REML cost differences
/// across seeds and reduce the screen to a √n-scaled tie-break. r_g is
/// dimensionless and bounded above by 1 for any well-defined PIRLS state, so
/// the penalty stays comparable to the cost differences that actually
/// distinguish good seeds from bad. The penalty vanishes at the true inner
/// mode (r_g → 0), so converged screening fits incur no penalty.
///
/// In the standard two-loop driver, partial fits never reach this helper:
/// `execute_pirls_if_needed` surfaces `MaxIterationsReached` and
/// `LmStepSearchExhausted` as `EstimationError::PirlsDidNotConverge`, so those
/// REML evaluations always operate on certified inner modes and this helper is
/// a strict no-op for them. The EFS single-loop strategy is the exception: it
/// intentionally ACCEPTS a partial (`is_failed_max_iterations`) inner state at
/// large n (the bam / Wood 2015 amortization tradeoff; see
/// `execute_pirls_if_needed`'s `in_efs_single_loop` branch), so an uncertified
/// inner mode can flow into cost assembly with the barrier active.
///
/// SINGLE SOURCE OF TRUTH (objective↔gradient consistency): this helper is the
/// one and only place the outer objective VALUE gains the `+0.5·r_g²` barrier.
/// Every outer-cost emission in the REML evaluator MUST route through it so the
/// `eval_cost` line-search value (`compute_cost`), the value+gradient/Hessian
/// path (`compute_outer_eval_with_order`), and the EFS step value
/// (`assemble_and_evaluate_efs`) report the IDENTICAL objective. The barrier
/// carries no analytic ρ/ψ-gradient and vanishes at every converged point, so
/// the analytic gradient is exact wherever the barrier is inactive; the only
/// requirement is that the reported VALUE never drifts from the gradient's
/// objective. Omitting the wrap on any one path reintroduces the
/// objective↔gradient desync that stalls the EFS iso-κ optimizer at large n
/// with a nonzero `final_grad_norm` (#1122). The complete caller set is:
///   * `compute_cost` (dense + sparse) — the `eval_cost` value
///   * `compute_outer_eval_with_order` (value-only early return + main path)
///   * `assemble_and_evaluate_efs` — the EFS step value
/// Add the wrap to any future outer-cost emission as well.
#[inline]
pub(crate) fn screening_residual_penalty(cost: f64, pr: &PirlsResult) -> f64 {
    crate::objective_base::failed_inner_residual_barrier_cost(
        cost,
        pr.status.is_failed_max_iterations(),
        pr.relative_gradient_norm(),
    )
}

pub(crate) fn hash_array_view(hasher: &mut Fingerprinter, values: ndarray::ArrayView1<'_, f64>) {
    hasher.write_usize(values.len());
    for &value in values {
        hasher.write_f64(value);
    }
}

pub(crate) fn hash_array2(hasher: &mut Fingerprinter, values: &Array2<f64>) {
    hasher.write_usize(values.nrows());
    hasher.write_usize(values.ncols());
    for &value in values {
        hasher.write_f64(value);
    }
}

pub(crate) fn hash_aux_prior_strength(
    hasher: &mut Fingerprinter,
    strength: gam_terms::latent::AuxPriorStrength,
) {
    use gam_terms::latent::AuxPriorStrength;
    match strength {
        AuxPriorStrength::Auto => hasher.write_str("auto"),
        AuxPriorStrength::Fixed(value) => {
            hasher.write_str("fixed");
            hasher.write_f64(value);
        }
    }
}

pub(in crate::estimate) fn latent_id_mode_cache_fingerprint(
    id_mode: &gam_terms::latent::LatentIdMode,
) -> u64 {
    use gam_terms::latent::{AuxPriorFamily, LatentIdMode};
    let mut hasher = Fingerprinter::new();
    hasher.write_str("latent-id-mode-cache-v1");
    match id_mode {
        LatentIdMode::AuxPrior {
            u,
            family,
            strength,
        } => {
            hasher.write_str("aux-prior");
            hash_array2(&mut hasher, u);
            match family {
                AuxPriorFamily::Ridge => hasher.write_str("ridge"),
                AuxPriorFamily::Linear => hasher.write_str("linear"),
            }
            hash_aux_prior_strength(&mut hasher, *strength);
        }
        LatentIdMode::AuxPriorDimSelection {
            u,
            family,
            strength,
            ..
        } => {
            hasher.write_str("aux-prior-dim-selection");
            hash_array2(&mut hasher, u);
            match family {
                AuxPriorFamily::Ridge => hasher.write_str("ridge"),
                AuxPriorFamily::Linear => hasher.write_str("linear"),
            }
            hash_aux_prior_strength(&mut hasher, *strength);
        }
        LatentIdMode::DimSelection { .. } => hasher.write_str("dim-selection"),
        LatentIdMode::IsometryToReference {
            reference,
            strength,
        } => {
            hasher.write_str("isometry-to-reference");
            hash_array2(&mut hasher, reference);
            hash_aux_prior_strength(&mut hasher, *strength);
        }
        LatentIdMode::AuxOutcome { head, .. } => {
            use gam_terms::decoders::behavioral_head::AuxOutcomeFamily;
            hasher.write_str("aux-outcome");
            match head.family() {
                AuxOutcomeFamily::Binomial => hasher.write_str("binomial"),
                AuxOutcomeFamily::Multinomial { n_classes } => {
                    hasher.write_str("multinomial");
                    hasher.write_usize(n_classes);
                }
            }
            hasher.write_usize(head.n_obs());
            hasher.write_f64(head.effective_labeled_count());
        }
        LatentIdMode::None => hasher.write_str("none"),
    }
    hasher.finish_u64()
}

pub(crate) fn hash_array3(hasher: &mut Fingerprinter, values: &ndarray::Array3<f64>) {
    let (a, b, c) = values.dim();
    hasher.write_usize(a);
    hasher.write_usize(b);
    hasher.write_usize(c);
    for &value in values {
        hasher.write_f64(value);
    }
}

pub(crate) fn hash_psi_slice(
    hasher: &mut Fingerprinter,
    target: &gam_terms::analytic_penalties::PsiSlice,
) {
    hasher.write_usize(target.range.start);
    hasher.write_usize(target.range.end);
    match target.latent_dim {
        Some(latent_dim) => {
            hasher.write_bool(true);
            hasher.write_usize(latent_dim);
        }
        None => hasher.write_bool(false),
    }
}

pub(crate) fn hash_scalar_weight_schedule(
    hasher: &mut Fingerprinter,
    schedule: &gam_terms::analytic_penalties::ScalarWeightSchedule,
) {
    use gam_problem::schedule::ScheduleKind;

    hasher.write_f64(schedule.w_start);
    hasher.write_f64(schedule.w_end);
    match &schedule.kind {
        ScheduleKind::Geometric { rate } => {
            hasher.write_str("geometric");
            hasher.write_f64(*rate);
        }
        ScheduleKind::Linear { steps } => {
            hasher.write_str("linear");
            hasher.write_usize(*steps);
        }
        ScheduleKind::ReciprocalIter => hasher.write_str("reciprocal-iter"),
    }
    hasher.write_usize(schedule.iter_count);
}

pub(crate) fn hash_weight_schedule_option(
    hasher: &mut Fingerprinter,
    schedule: &Option<gam_terms::analytic_penalties::ScalarWeightSchedule>,
) {
    match schedule {
        Some(schedule) => {
            hasher.write_bool(true);
            hash_scalar_weight_schedule(hasher, schedule);
        }
        None => hasher.write_bool(false),
    }
}

pub(crate) fn hash_gumbel_temperature_schedule(
    hasher: &mut Fingerprinter,
    schedule: &gam_problem::schedule::GumbelTemperatureSchedule,
) {
    use gam_problem::schedule::ScheduleKind;

    hasher.write_f64(schedule.tau_start);
    hasher.write_f64(schedule.tau_min);
    match &schedule.decay {
        ScheduleKind::Geometric { rate } => {
            hasher.write_str("geometric");
            hasher.write_f64(*rate);
        }
        ScheduleKind::Linear { steps } => {
            hasher.write_str("linear");
            hasher.write_usize(*steps);
        }
        ScheduleKind::ReciprocalIter => hasher.write_str("reciprocal-iter"),
    }
    hasher.write_usize(schedule.iter_count);
}

pub(crate) fn hash_gumbel_schedule_option(
    hasher: &mut Fingerprinter,
    schedule: &Option<gam_problem::schedule::GumbelTemperatureSchedule>,
) {
    match schedule {
        Some(schedule) => {
            hasher.write_bool(true);
            hash_gumbel_temperature_schedule(hasher, schedule);
        }
        None => hasher.write_bool(false),
    }
}

pub(crate) fn hash_isometry_reference(
    hasher: &mut Fingerprinter,
    reference: &gam_terms::analytic_penalties::IsometryReference,
) {
    use gam_terms::analytic_penalties::IsometryReference;

    match reference {
        IsometryReference::Euclidean => hasher.write_str("euclidean"),
        IsometryReference::UserSupplied(values) => {
            hasher.write_str("user-supplied");
            hash_array2(hasher, values.as_ref());
        }
    }
}

pub(crate) fn hash_weight_field(
    hasher: &mut Fingerprinter,
    field: &gam_terms::analytic_penalties::WeightField,
) {
    use gam_terms::analytic_penalties::WeightField;

    match field {
        WeightField::Identity => hasher.write_str("identity"),
        WeightField::Factored { u, rank, p_out } => {
            hasher.write_str("factored");
            hash_array2(hasher, u.as_ref());
            hasher.write_usize(*rank);
            hasher.write_usize(*p_out);
        }
    }
}

pub(crate) fn hash_sparsity_kind(
    hasher: &mut Fingerprinter,
    kind: gam_terms::analytic_penalties::SparsityKind,
) {
    use gam_terms::analytic_penalties::SparsityKind;

    match kind {
        SparsityKind::SmoothedL1 { eps } => {
            hasher.write_str("smoothed-l1");
            hasher.write_f64(eps);
        }
        SparsityKind::Hoyer => hasher.write_str("hoyer"),
        SparsityKind::Log { delta } => {
            hasher.write_str("log");
            hasher.write_f64(delta);
        }
    }
}

pub(crate) fn hash_difference_op_kind(
    hasher: &mut Fingerprinter,
    kind: &gam_terms::analytic_penalties::DifferenceOpKind,
) {
    use gam_terms::analytic_penalties::DifferenceOpKind;

    match kind {
        DifferenceOpKind::ForwardDiff1D => hasher.write_str("forward-diff-1d"),
        DifferenceOpKind::GraphEdges(edges) => {
            hasher.write_str("graph-edges");
            hasher.write_usize(edges.len());
            for &(from, to) in edges {
                hasher.write_usize(from);
                hasher.write_usize(to);
            }
        }
    }
}

pub(crate) fn hash_groups(hasher: &mut Fingerprinter, groups: &[Vec<usize>]) {
    hasher.write_usize(groups.len());
    for group in groups {
        hasher.write_usize(group.len());
        for &axis in group {
            hasher.write_usize(axis);
        }
    }
}

pub(crate) fn hash_analytic_penalty_kind(
    hasher: &mut Fingerprinter,
    penalty: &gam_terms::analytic_penalties::AnalyticPenaltyKind,
) {
    use gam_terms::analytic_penalties::{AnalyticPenaltyKind, PenaltyConcavity};

    hasher.write_str(penalty.name());
    hasher.write_str(&format!("{:?}", penalty.tier()));
    hasher.write_usize(penalty.rho_count());
    match penalty {
        AnalyticPenaltyKind::Isometry(p) => {
            hasher.write_str("isometry");
            hash_psi_slice(hasher, &p.target);
            hash_isometry_reference(hasher, &p.reference);
            hasher.write_usize(p.rho_index);
            hasher.write_usize(p.p_out);
            hash_weight_field(hasher, &p.weight);
            hasher.write_f64(p.scalar_weight);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
            // The `jacobian_cache` / `jacobian_second_cache` /
            // `third_decoder_derivative` slots are interior-mutable
            // (`RwLock<Option<Arc<…>>>`), lazily populated, and θ-DEPENDENT:
            // the SAE/IFT driver calls `refresh_caches` each outer step so the
            // cached J / H / K reflect the Jacobian at the *current* outer θ.
            // They are NOT part of this penalty's identity — they are a pure
            // (recomputable) function of the basis + θ, and the basis identity
            // is already captured exactly by `duchon_radial_source` (below) for
            // the Duchon path and by the hashed design matrix / latent
            // fingerprint for the SAE path. Hashing the live cache snapshot made
            // the persistent warm-start key non-reproducible across otherwise
            // identical fits: a cold fit opens its session with the slots empty
            // (`None`), while a repeat fit sees them populated from the prior
            // run's converged θ, so the key drifted and the `skip-outer-
            // validation` warm hit was lost (issue #1048). The stored payload is
            // the converged (ρ, β) — equivalence to recomputing is unaffected by
            // dropping these derived snapshots from the key, so we deliberately
            // do NOT hash them.
            match p.duchon_radial_source.as_ref() {
                Some(source) => {
                    hasher.write_bool(true);
                    hash_array2(hasher, source.centers.as_ref());
                    hash_array2(hasher, source.radial_coefficients.as_ref());
                    match source.length_scale {
                        Some(length_scale) => {
                            hasher.write_bool(true);
                            hasher.write_f64(length_scale);
                        }
                        None => hasher.write_bool(false),
                    }
                    hasher.write_str(&format!("{:?}", source.nullspace_order));
                }
                None => hasher.write_bool(false),
            }
        }
        AnalyticPenaltyKind::Sparsity(p) => {
            hasher.write_str("sparsity");
            hasher.write_str(&format!("{:?}", p.target_tier));
            hash_sparsity_kind(hasher, p.kind);
            hasher.write_f64(p.weight);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
            hasher.write_usize(p.strength_rho_index);
            match p.eps_rho_index {
                Some(idx) => {
                    hasher.write_bool(true);
                    hasher.write_usize(idx);
                }
                None => hasher.write_bool(false),
            }
        }
        AnalyticPenaltyKind::SoftmaxAssignmentSparsity(p) => {
            hasher.write_str("softmax-assignment-sparsity");
            hasher.write_usize(p.k_atoms);
            hasher.write_f64(p.temperature);
            hasher.write_f64(p.weight);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::IBPAssignment(p) => {
            hasher.write_str("ibp-assignment");
            hasher.write_usize(p.k_max);
            hasher.write_f64(p.alpha);
            hasher.write_f64(p.tau);
            hash_gumbel_schedule_option(hasher, &p.temperature_schedule);
            hasher.write_bool(p.learnable_alpha);
            hasher.write_f64(p.weight);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::Ard(p) => {
            hasher.write_str("ard");
            hash_psi_slice(hasher, &p.target);
            hasher.write_usize(p.latent_dim);
            hasher.write_f64(p.weight);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
            hasher.write_usize(p.rho_indices.len());
            for &idx in &p.rho_indices {
                hasher.write_usize(idx);
            }
            hasher.write_f64(p.n_eff);
        }
        AnalyticPenaltyKind::TopKActivation(p) => {
            hasher.write_str("topk-activation");
            hash_psi_slice(hasher, &p.target);
            hasher.write_usize(p.k);
            hasher.write_usize(p.latent_dim);
            hasher.write_f64(p.weight);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::JumpReLU(p) => {
            hasher.write_str("jumprelu");
            hash_psi_slice(hasher, &p.target);
            hasher.write_usize(p.latent_dim);
            hash_array_view(hasher, p.thresholds.view());
            hasher.write_f64(p.weight);
            hasher.write_f64(p.smoothing_eps);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::TotalVariation(p) => {
            hasher.write_str("total-variation");
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hash_difference_op_kind(hasher, &p.difference_op);
            hasher.write_f64(p.smoothing_eps);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::NuclearNorm(p) => {
            hasher.write_str("nuclear-norm");
            hash_psi_slice(hasher, &p.target);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_f64(p.smoothing_eps);
            match p.max_rank {
                Some(max_rank) => {
                    hasher.write_bool(true);
                    hasher.write_usize(max_rank);
                }
                None => hasher.write_bool(false),
            }
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::BlockSparsity(p) => {
            hasher.write_str("block-sparsity");
            hash_psi_slice(hasher, &p.target);
            hash_groups(hasher, &p.groups);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_f64(p.smoothing_eps);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::MechanismSparsity(p) => {
            hasher.write_str("mechanism-sparsity");
            hash_psi_slice(hasher, &p.target);
            hash_groups(hasher, &p.feature_groups);
            hasher.write_f64(p.weight);
            hasher.write_f64(p.smoothing_eps);
            hasher.write_f64(p.n_eff);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            match &p.weight_schedule {
                Some(schedule) => {
                    hasher.write_bool(true);
                    hash_scalar_weight_schedule(hasher, schedule.as_ref());
                }
                None => hasher.write_bool(false),
            }
        }
        AnalyticPenaltyKind::RowPrecisionPrior(p) => {
            hasher.write_str("row-precision-prior");
            hash_array3(hasher, &p.lambda_per_row);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_psi_slice(hasher, &p.target);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::IvaeRidgeMeanGauge(p) => {
            hasher.write_str("ivae-ridge-mean-gauge");
            hash_array2(hasher, &p.aux);
            hash_array2(hasher, &p.ridge_inv);
            hasher.write_f64(p.ridge_eps);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_psi_slice(hasher, &p.target);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::ParametricRowPrecisionPrior(p) => {
            hasher.write_str("parametric-row-precision-prior");
            hash_array2(hasher, &p.aux);
            hash_array_view(hasher, p.log_alpha.view());
            hash_array_view(hasher, p.raw_beta.view());
            hash_array2(hasher, &p.mu);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_bool(p.learnable_weight);
            hash_psi_slice(hasher, &p.target);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::ScadMcp(p) => {
            hasher.write_str("scad-mcp");
            hash_psi_slice(hasher, &p.target);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_f64(p.gamma);
            hasher.write_f64(p.smoothing_eps);
            match p.variant {
                PenaltyConcavity::Mcp => hasher.write_str("mcp"),
                PenaltyConcavity::Scad => hasher.write_str("scad"),
            }
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::BlockOrthogonality(p) => {
            hasher.write_str("block-orthogonality");
            hash_psi_slice(hasher, &p.target);
            hash_groups(hasher, &p.groups);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::DecoderIncoherence(p) => {
            hasher.write_str("decoder-incoherence");
            hash_psi_slice(hasher, &p.target);
            hasher.write_usize(p.block_sizes.len());
            for &m in &p.block_sizes {
                hasher.write_usize(m);
            }
            hasher.write_usize(p.p_out);
            hasher.write_usize(p.k_atoms);
            hasher.write_usize(p.pairs.len());
            for &(j, k, w) in &p.pairs {
                hasher.write_usize(j);
                hasher.write_usize(k);
                hasher.write_f64(w);
            }
            hasher.write_f64(p.weight);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::Orthogonality(p) => {
            hasher.write_str("orthogonality");
            hash_psi_slice(hasher, &p.target);
            hasher.write_usize(p.latent_dim);
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::NestedPrefix(p) => {
            hasher.write_str("nested-prefix");
            hash_psi_slice(hasher, &p.target);
            hasher.write_str(&format!("{:?}", p.target_tier));
            hasher.write_usize(p.prefix_sizes.len());
            for &m in &p.prefix_sizes {
                hasher.write_usize(m);
            }
            hasher.write_usize(p.shell_weights.len());
            for &w in &p.shell_weights {
                hasher.write_f64(w);
            }
            hasher.write_f64(p.eps);
            hasher.write_usize(p.rho_indices.len());
            for &idx in &p.rho_indices {
                hasher.write_usize(idx);
            }
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::Monotonicity(p) => {
            hasher.write_str("monotonicity");
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            hasher.write_f64(p.direction);
            hasher.write_f64(p.smoothing_eps);
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::SheafConsistency(p) => {
            hasher.write_str("sheaf-consistency");
            hasher.write_f64(p.weight());
            let dims = p.stalk_dims();
            hasher.write_usize(dims.len());
            for &d in dims {
                hasher.write_usize(d);
            }
        }
        AnalyticPenaltyKind::HarmonicRoughness(p) => {
            hasher.write_str("harmonic-roughness");
            hasher.write_f64(p.weight);
            hasher.write_usize(p.n_eff);
            // Per-period diagonal weights are the penalty's identity: the
            // resolved operator tiles them across the `n_eff` rows, so two
            // penalties with equal (weight, n_eff, rho_index) but different
            // row_weights are distinct and must not share a warm-start key.
            hasher.write_usize(p.row_weights.len());
            for &w in &p.row_weights {
                hasher.write_f64(w);
            }
            hasher.write_bool(p.learnable_weight);
            hasher.write_usize(p.rho_index);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
    }
}

pub(crate) fn analytic_penalty_registry_fingerprint(
    registry: &gam_terms::analytic_penalties::AnalyticPenaltyRegistry,
) -> u64 {
    let mut hasher = Fingerprinter::new();
    hasher.write_str("analytic-penalty-registry-v1");
    hasher.write_usize(registry.penalties.len());
    for penalty in &registry.penalties {
        hash_analytic_penalty_kind(&mut hasher, penalty);
    }
    hasher.finish_u64()
}

pub(crate) fn hash_design_matrix(
    hasher: &mut Fingerprinter,
    design: &DesignMatrix,
) -> Result<(), String> {
    // Stream the design through fixed-byte row blocks so a large-scale design
    // is never fully materialized just to fingerprint it. Target ~8 MiB of
    // working set per chunk, with a row-count floor of 1 (always make progress)
    // and a ceiling so a very narrow design does not request an unbounded chunk.
    const HASH_CHUNK_TARGET_BYTES: usize = 8 * 1024 * 1024;
    const HASH_CHUNK_MIN_ROWS: usize = 1;
    const HASH_CHUNK_MAX_ROWS: usize = 4096;
    let n = design.nrows();
    let p = design.ncols();
    hasher.write_usize(n);
    hasher.write_usize(p);
    let bytes_per_row = p.saturating_mul(std::mem::size_of::<f64>()).max(1);
    let chunk_rows =
        (HASH_CHUNK_TARGET_BYTES / bytes_per_row).clamp(HASH_CHUNK_MIN_ROWS, HASH_CHUNK_MAX_ROWS);
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let chunk = design
            .try_row_chunk(start..end)
            .map_err(|e| format!("persistent warm-start design hash failed: {e}"))?;
        hash_array2(hasher, &chunk);
    }
    Ok(())
}

pub(crate) fn hash_canonical_penalties(
    hasher: &mut Fingerprinter,
    penalties: &[gam_terms::construction::CanonicalPenalty],
) {
    hasher.write_usize(penalties.len());
    for penalty in penalties {
        hasher.write_usize(penalty.col_range.start);
        hasher.write_usize(penalty.col_range.end);
        hasher.write_usize(penalty.total_dim);
        hasher.write_usize(penalty.nullity);
        hash_array2(hasher, &penalty.root);
        hash_array2(hasher, &penalty.local);
        hash_array_view(hasher, penalty.prior_mean.view());
        hasher.write_usize(penalty.positive_eigenvalues.len());
        for &value in &penalty.positive_eigenvalues {
            hasher.write_f64(value);
        }
        hasher.write_bool(penalty.op.is_some());
    }
}

pub(crate) fn finite_positive_from_bits(bits: u64) -> Option<f64> {
    if bits == 0 {
        return None;
    }
    let value = f64::from_bits(bits);
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

pub(crate) fn finite_nonnegative_from_bits(bits: u64) -> Option<f64> {
    let value = f64::from_bits(bits);
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

pub(crate) fn finite_nonnegative_bits_or_no_signal(value: Option<f64>) -> u64 {
    value
        .filter(|v| v.is_finite() && *v >= 0.0)
        .map(f64::to_bits)
        .unwrap_or(IFT_RESIDUAL_NO_SIGNAL_BITS)
}

#[derive(Clone)]
pub(crate) struct TkCorrectionTerms {
    pub(crate) value: f64,
    pub(crate) gradient: Option<Array1<f64>>,
    pub(crate) hessian: Option<Array2<f64>>,
}

pub(crate) struct TkSharedIntermediates {
    pub(crate) h_diag: Array1<f64>,
    pub(crate) x_m: Array1<f64>,
    pub(crate) y: Array1<f64>,
    pub(crate) active_blocks: Vec<TkActiveBlock>,
}

pub(crate) struct TkActiveBlock {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) entries: Vec<(usize, f64)>,
}

/// Family-dependent derivative context shared by all assembly builders.
///
/// Both `build_dense_derivative_context` and `build_sparse_derivative_context`
/// return this, eliminating the tuple-order mismatch that previously existed
/// between the two paths.
pub(crate) struct DerivativeContext {
    pub(crate) deriv_provider: Box<dyn super::reml_outer_engine::HessianDerivativeProvider>,
    pub(crate) dispersion: super::reml_outer_engine::DispersionHandling,
    pub(crate) log_likelihood: f64,
    pub(crate) firth_op: Option<std::sync::Arc<super::FirthDenseOperator>>,
    pub(crate) barrier_config: Option<super::reml_outer_engine::BarrierConfig>,
}

/// Project a `GlmLikelihoodSpec` onto a `LikelihoodSpec` for pattern matching
/// on the `(response, link)` form used elsewhere in the codebase.
#[inline]
pub(crate) fn reml_spec(likelihood: &GlmLikelihoodSpec) -> LikelihoodSpec {
    likelihood.spec.clone()
}

#[inline]
pub(crate) fn reml_is_gaussian_identity(likelihood: &GlmLikelihoodSpec) -> bool {
    reml_spec(likelihood).is_gaussian_identity()
}

/// Inverse link of a Binomial family for which a Fisher-weight jet exists, i.e.
/// the links the link-general Jeffreys term can regularize. This includes
/// standard `{Logit, Probit, CLogLog}` and stateful links whose fourth/fifth
/// inverse-link derivatives are available, including mixture LogLog/Cauchit
/// components. Returns `None` for any other response or link.
#[inline]
pub(crate) fn reml_jeffreys_supported_link(likelihood: &GlmLikelihoodSpec) -> Option<InverseLink> {
    let spec = reml_spec(likelihood);
    if !matches!(spec.response, ResponseFamily::Binomial) {
        return None;
    }
    if spec.link.has_fisher_weight_jet() {
        Some(spec.link.clone())
    } else {
        None
    }
}

/// Resolve whether the Jeffreys/Firth term should be assembled on the REML path
/// and, if so, the inverse link to evaluate the Fisher weight with.
///
/// The Jeffreys term is assembled iff the caller requested Firth bias reduction
/// (`firth_bias_reduction`) on a Binomial inverse link that exposes a
/// Fisher-weight jet. This MUST agree with the inner P-IRLS Firth activation in
/// `loop_driver.rs`: the outer analytic derivatives (`H`, `u`, IFT) and the
/// converged inner mode have to be derivatives of the SAME penalized objective.
/// Arming the outer term while the inner mode is non-Firth (or vice-versa)
/// desyncs the two by exactly the Jeffreys score/curvature contribution and
/// breaks the τ-τ Hessian-vs-FD and stationarity-cancellation identities
/// (#825). Unsupported links return `None` instead of pretending they are Logit.
#[inline]
pub(crate) fn reml_robust_jeffreys_link(config: &RemlConfig) -> Option<InverseLink> {
    if !config.firth_bias_reduction {
        return None;
    }
    reml_jeffreys_supported_link(&config.likelihood)
}

/// `upper`/`tail_prob` calibrating the firth-general default barrier on an unset
/// smoothing coordinate. The tail statement `P(d > upper) = tail_prob` on the
/// marginal-SD distance scale `d = exp(−ρ/2)` calibrates the exponential rate
/// `θ = −ln(tail_prob)/upper`. We use `upper = 10`, `tail_prob = 0.01`
/// ⇒ `θ = −ln(0.01)/10 ≈ 0.4605`.
pub(crate) const FIRTH_DEFAULT_PC_UPPER: f64 = 10.0;

pub(crate) const FIRTH_DEFAULT_PC_TAIL_PROB: f64 = 0.01;

/// Weakly-informative DEFAULT outer ρ prior used by the firth-general policy on
/// any smoothing coordinate the caller left unset (`RhoPrior::Flat`).
///
/// The *value* returned here is a [`RhoPrior::PenalizedComplexity`] so that
/// downstream consumers that round-trip the resolved prior (serialization, the
/// joint-HMC refinement at `effective_rho_prior().into_owned()`) see a
/// well-defined prior family. Its *outer-objective contribution*, however, is NOT
/// the plain PC term: the REML/LAML runtime evaluates firth-default coordinates
/// through the SELF-GATED, one-sided barrier
/// [`crate::rho_prior_eval::firth_default_barrier_terms`], which is byte-identically
/// flat (cost/grad/hess = 0) on the identified side `ρ ≥ −2 ln(upper)` and only a
/// convex wall against the `λ → 0` / `ρ → −∞` under-smoothing degeneracy below
/// it. This restores STRICT zero-downside (a clean / well-conditioned fit is
/// byte-identical to plain REML, mirroring the Jeffreys conditioning gate)
/// instead of the plain PC term's persistent `+1/2` Occam pull, which would
/// shift every identified `λ` by an `O(1/n)` amount on every fit.
#[inline]
pub(crate) fn firth_default_pc_prior() -> RhoPrior {
    RhoPrior::PenalizedComplexity {
        upper: FIRTH_DEFAULT_PC_UPPER,
        tail_prob: FIRTH_DEFAULT_PC_TAIL_PROB,
    }
}

/// A Gamma prior on the physical precision `λ` with shape `1` and rate `0` has
/// density proportional to a constant in `λ`. Under the deterministic
/// MAP-in-`λ` REML convention used for [`RhoPrior::GammaPrecision`], its
/// negative-log contribution is exactly zero (cost/gradient/Hessian all vanish),
/// so it is semantically the same "unset/flat" coordinate as [`RhoPrior::Flat`].
///
/// Keep this equivalence explicit at the prior-policy boundary. Otherwise an
/// `Independent([GammaPrecision { shape: 1, rate: 0 }])` is treated as an
/// explicitly configured prior while `Flat` is treated as an unset coordinate,
/// sending mathematically identical fits through different default-prior and
/// seed-selection branches.
#[inline]
pub(crate) fn is_unset_flat_rho_prior(prior: &RhoPrior) -> bool {
    match prior {
        RhoPrior::Flat => true,
        RhoPrior::GammaPrecision { shape, rate } => *shape == 1.0 && *rate == 0.0,
        _ => false,
    }
}

/// Per-coordinate `true` where the firth-general default barrier (rather than an
/// explicitly-configured prior) governs that smoothing coordinate. A coordinate
/// is a firth default exactly when the caller left it mathematically flat:
/// `Flat`, or the equivalent `GammaPrecision { shape: 1, rate: 0 }`, either as a
/// whole prior or as holes in an `Independent` prior. Returned per-`ρ`-coordinate
/// so the runtime can override just those coordinates' objective contribution
/// with the self-gated barrier.
pub(crate) fn firth_default_coord_mask(configured: &RhoPrior, len: usize) -> Vec<bool> {
    match configured {
        RhoPrior::Flat => vec![true; len],
        RhoPrior::Independent(priors) if priors.len() == len => {
            priors.iter().map(is_unset_flat_rho_prior).collect()
        }
        _ => vec![false; len],
    }
}

/// Resolve the *effective* outer ρ prior under the (unconditional) firth-general
/// default policy.
///
/// An *unset* prior (the `Flat` sentinel or equivalent Gamma(1, 0) flat prior —
/// whole-prior, or any such coordinate of an `Independent`) is filled with the weakly-informative
/// [`firth_default_pc_prior`]; any explicitly-configured prior is honored
/// unchanged. Pulled out as a free function so the decision is unit-testable
/// without constructing a full `RemlState`.
pub(crate) fn resolve_effective_rho_prior(configured: &RhoPrior) -> std::borrow::Cow<'_, RhoPrior> {
    match configured {
        // Whole prior unset → fill every coordinate with the weak PC default.
        RhoPrior::Flat => std::borrow::Cow::Owned(firth_default_pc_prior()),
        // Gamma(1, 0) is exactly flat in the deterministic MAP-in-λ convention,
        // so route it through the same unset path as `Flat`.
        RhoPrior::GammaPrecision { shape, rate } if *shape == 1.0 && *rate == 0.0 => {
            std::borrow::Cow::Owned(firth_default_pc_prior())
        }
        // Per-coordinate priors: only the `Flat` (unset) coordinates inherit the
        // PC default; explicitly configured coordinates are preserved.
        RhoPrior::Independent(priors) if priors.iter().any(is_unset_flat_rho_prior) => {
            let filled = priors
                .iter()
                .map(|p| match p {
                    p if is_unset_flat_rho_prior(p) => firth_default_pc_prior(),
                    other => other.clone(),
                })
                .collect();
            std::borrow::Cow::Owned(RhoPrior::Independent(filled))
        }
        // Any explicitly configured prior is honored as-is.
        other => std::borrow::Cow::Borrowed(other),
    }
}

#[inline]
pub(crate) fn reml_fixed_glm_dispersion(likelihood: &GlmLikelihoodSpec) -> f64 {
    let spec = reml_spec(likelihood);
    match (&spec.response, &spec.link) {
        // Beta carries phi inside the response variant under the LikelihoodSpec form.
        (ResponseFamily::Beta { phi }, _) => *phi,
        // REML uses unit scale for NB; overdispersion is encoded by theta in the
        // response variant. IRLS consumes theta through the variance chain, not
        // as a separate phi.
        (ResponseFamily::NegativeBinomial { .. }, _) => 1.0,
        // Tweedie's variance power lives on the response variant; the scale phi
        // comes from the dispersion slot.
        (ResponseFamily::Tweedie { .. }, _) => likelihood.fixed_phi().unwrap_or(1.0),
        // All other (response, link) combinations share the dispersion-slot phi
        // (defaulting to 1.0 when absent).
        (
            ResponseFamily::Gaussian
            | ResponseFamily::Binomial
            | ResponseFamily::Poisson
            | ResponseFamily::Gamma,
            _,
        ) => likelihood.fixed_phi().unwrap_or(1.0),
        // RoystonParmar is survival-specific and not produced by
        // `reml_spec` from any `LikelihoodSpec` GLM family combination.
        (ResponseFamily::RoystonParmar, _) => likelihood.fixed_phi().unwrap_or(1.0),
    }
}

/// Minimum importance-sampling effective-sample fraction below which the #784
/// block-local sampled marginalization is declined (the Monte-Carlo estimate
/// would be noisier than the Laplace error it corrects). Auto-derived constant,
/// not a tunable flag.
pub(crate) const MIN_IMPORTANCE_ESS_FRACTION: f64 = 0.10;

/// Block-local non-Gaussian-remainder target for the adaptive Laplace-to-
/// sampling fallback (issue #784).
///
/// Implements the engine-local HMC I/O block target for the standard-GAM
/// GLM inner loop. The fallback sampler asks this target, for each whitened
/// block displacement `t` (coordinates in the curvature-heavy H-eigenvector
/// subspace `V_b`), for the non-Gaussian remainder
///
///   ΔF(t) = F(β̂ + δ) − F(β̂) − ½ δᵀ H δ,   δ = V_b t,
///   F(β)  = −ℓ(β) + ½ βᵀ S(ρ) β.
///
/// Using the mode condition `S β̂ = ∇ℓ(β̂)` and `∇²ℓ = −Xᵀ W X`, the penalty's
/// (exactly quadratic) curvature cancels and the remainder reduces to a
/// family-uniform expression in the deviance plus the explicit penalty score:
///
///   ΔF(t) = (1/2φ)[D(μ(η̂ + Xδ)) − D(μ(η̂))]   (= −[ℓ(β̂+δ) − ℓ(β̂)])
///           + (S β̂)·δ                          (penalty-score channel)
///           − ½ Σ_i W_i (Xδ)_i².               (likelihood-curvature subtraction)
///
/// `D` is the family deviance (`calculate_deviance`), so this works uniformly
/// across every GLM family/link without per-family score code. The only place
/// ρ appears *explicitly* (with δ held fixed in coefficient space) is the
/// penalty-score term, giving the exact explicit ρ-gradient
///   ∂ΔF/∂ρ_k = λ_k (S_k β̂)·δ.
/// The implicit β̂(ρ) channel is the same envelope term the surrounding
/// Laplace/LAML evaluator already accounts for at the mode.
pub(crate) struct Gam784BlockTarget<'t> {
    /// `X_t` (transformed-basis dense design, matching `h_total`/`solve_c_array`).
    pub(crate) x_transformed: &'t Array2<f64>,
    /// Block eigenvectors `V_b` (columns), shape `p × m`.
    pub(crate) block_vecs: Array2<f64>,
    /// Block curvatures `λ_r` (the `H_total` eigenvalues), length `m`.
    pub(crate) block_lambdas: Array1<f64>,
    /// Mode linear predictor η̂ = X_t β̂.
    pub(crate) eta_hat: Array1<f64>,
    /// Per-row observed weights `W_i` (the likelihood Hessian diagonal).
    pub(crate) weights_obs: Array1<f64>,
    /// Response y and prior weights for the deviance.
    pub(crate) y: Array1<f64>,
    pub(crate) prior_weights: Array1<f64>,
    /// Family/link spec for the deviance and the inverse link.
    pub(crate) likelihood: GlmLikelihoodSpec,
    pub(crate) inverse_link: InverseLink,
    /// Dispersion φ used to scale the deviance into a log-likelihood.
    pub(crate) phi: f64,
    /// Penalty scores `S_k β̂` per canonical penalty (unscaled by λ_k).
    /// Shared from the eval bundle's once-per-inner-solution cache
    /// (`EvalShared::canonical_penalty_scores_at_mode`).
    pub(crate) penalty_scores: Arc<Vec<Array1<f64>>>,
    /// `λ_k = e^{ρ_k}` per canonical penalty, aligned with `penalty_scores`.
    pub(crate) lambdas: Vec<f64>,
    /// Deviance at the base mode.
    pub(crate) base_deviance: f64,
}

impl Gam784BlockTarget<'_> {
    /// Map a whitened block displacement `t` to the coefficient displacement
    /// `δ = V_b t` and the per-row score `s = X_t δ`.
    pub(crate) fn displacement(&self, t: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let delta = self.block_vecs.dot(t);
        let s = gam_linalg::faer_ndarray::fast_av(self.x_transformed, &delta);
        (delta, s)
    }

    /// Per-row score `∂(D(η)/2φ)/∂η` at the given linear predictor.
    ///
    /// Mirrors `calculate_deviance`'s per-family μ-floors so this is the
    /// exact η-derivative of the SAME deviance the value channel sums:
    /// `dD_i/dμ = −2·w_i·(y_i − μ_i)/V(μ_i) · s_fam` (the exponential-family
    /// unit-deviance identity, exact for Binomial/Poisson/Gamma/NB/Tweedie/
    /// Gaussian — the families `block_local_sampled_correction` admits),
    /// `s_fam` being the internal dispersion division the Gaussian and
    /// Tweedie branches of `calculate_deviance` apply. The trailing `/(2φ)`
    /// matches the excess definition `[D_disp − D_base]/(2φ)`.
    ///
    /// Rows whose inverse-link jet is infeasible return 0: the value channel
    /// scores such draws as `ΔF = ∞` (zero importance weight), so their score
    /// is never consumed.
    pub(crate) fn neg_score_at(&self, eta: &Array1<f64>) -> Array1<f64> {
        let spec_response = reml_spec(&self.likelihood).response.clone();
        let family = pirls::weight_family_for_glm_likelihood(&self.likelihood);
        let fam_scale = match &spec_response {
            ResponseFamily::Gaussian | ResponseFamily::Tweedie { .. } => {
                1.0 / self.likelihood.fixed_phi().unwrap_or(1.0)
            }
            _ => 1.0,
        };
        // Same floors as `calculate_deviance`: binomial clamps μ to
        // [1e-12, 1−1e-12]; the remaining families floor μ at 1e-10.
        const BINOMIAL_MU_EPS: f64 = 1e-12;
        const MU_FLOOR: f64 = 1e-10;
        let is_binomial = matches!(spec_response, ResponseFamily::Binomial);
        let mut out = Array1::<f64>::zeros(eta.len());
        for i in 0..eta.len() {
            let jet = match crate::mixture_link::inverse_link_jet_for_inverse_link(
                &self.inverse_link,
                eta[i],
            ) {
                Ok(jet) => jet,
                Err(_) => continue,
            };
            let mu_c = if is_binomial {
                jet.mu.clamp(BINOMIAL_MU_EPS, 1.0 - BINOMIAL_MU_EPS)
            } else {
                jet.mu.max(MU_FLOOR)
            };
            let v = pirls::variance_jet_for_weight_family(family, mu_c).v;
            if !(v.is_finite() && v > 0.0) {
                continue;
            }
            let d_dev_d_mu = -2.0 * self.prior_weights[i] * (self.y[i] - mu_c) / v * fam_scale;
            out[i] = d_dev_d_mu * jet.d1 / (2.0 * self.phi);
        }
        out
    }
}

impl BlockExcessTarget for Gam784BlockTarget<'_> {
    fn block_dim(&self) -> usize {
        self.block_lambdas.len()
    }

    fn rho_dim(&self) -> usize {
        self.lambdas.len()
    }

    fn block_curvatures(&self) -> &Array1<f64> {
        &self.block_lambdas
    }

    fn excess(&self, t: &Array1<f64>) -> f64 {
        let (delta, s) = self.displacement(t);
        // Displaced mean μ(η̂ + s) via the inverse-link jet (family-uniform).
        let mut mu_disp = Array1::<f64>::zeros(self.eta_hat.len());
        for i in 0..self.eta_hat.len() {
            let eta_i = self.eta_hat[i] + s[i];
            match crate::mixture_link::inverse_link_jet_for_inverse_link(&self.inverse_link, eta_i)
            {
                Ok(jet) => mu_disp[i] = jet.mu,
                Err(_) => return f64::INFINITY,
            }
        }
        let dev_disp = crate::pirls::calculate_deviance(
            self.y.view(),
            &mu_disp,
            &self.likelihood,
            self.prior_weights.view(),
        );
        if !dev_disp.is_finite() {
            return f64::INFINITY;
        }
        // −[ℓ(β̂+δ) − ℓ(β̂)] = (1/2φ)[D_disp − D_base].
        let neg_loglik_diff = (dev_disp - self.base_deviance) / (2.0 * self.phi);
        // Penalty-score channel (S β̂)·δ = Σ_k λ_k (S_k β̂)·δ.
        let mut penalty_term = 0.0_f64;
        for (score, &lam) in self.penalty_scores.iter().zip(self.lambdas.iter()) {
            penalty_term += lam * score.dot(&delta);
        }
        // Likelihood-curvature subtraction ½ Σ_i W_i s_i².
        let mut curv = 0.0_f64;
        for i in 0..s.len() {
            curv += self.weights_obs[i] * s[i] * s[i];
        }
        neg_loglik_diff + penalty_term - 0.5 * curv
    }

    fn excess_rho_gradient(&self, t: &Array1<f64>) -> Array1<f64> {
        // Only the coefficient displacement `δ = V_b t` (O(pm)) is needed here;
        // the per-row score `s = X_t δ` (the O(np) design matvec) that
        // `displacement` also computes is unused, so skip it.
        let delta = self.block_vecs.dot(t);
        let mut grad = Array1::<f64>::zeros(self.lambdas.len());
        for (k, (score, &lam)) in self
            .penalty_scores
            .iter()
            .zip(self.lambdas.iter())
            .enumerate()
        {
            // ∂ΔF/∂ρ_k = λ_k (S_k β̂)·δ (the only explicit ρ-appearance).
            grad[k] = lam * score.dot(&delta);
        }
        grad
    }

    fn displaced_neg_score(&self, t: &Array1<f64>) -> Array1<f64> {
        let (_delta, s) = self.displacement(t);
        self.neg_score_at(&(&self.eta_hat + &s))
    }

    fn base_neg_score(&self) -> Array1<f64> {
        self.neg_score_at(&self.eta_hat)
    }

    /// Fused excess + displaced score sharing ONE design matvec `s = X_t δ` and
    /// ONE inverse-link jet sweep at `η̂ + s`. The jet yields both `μ` (the
    /// deviance/excess channel) and `d1 = dμ/dη` (the score channel), so the
    /// per-draw O(n·p) matvec and O(n) jet evaluation — previously paid twice
    /// (once in `excess`, once in `displaced_neg_score`) — are paid once. The
    /// summed value is bit-identical to the separate calls: `excess` reads
    /// `jet.mu` and `displaced_neg_score` reads the SAME jet's `d1`/`mu` floors,
    /// which this method reproduces exactly (#784, #1082).
    fn excess_with_displaced_neg_score(&self, t: &Array1<f64>) -> (f64, Option<Array1<f64>>) {
        let (delta, s) = self.displacement(t);
        let n = self.eta_hat.len();

        // Family constants mirrored from `neg_score_at` / `calculate_deviance`.
        let spec_response = reml_spec(&self.likelihood).response.clone();
        let family = pirls::weight_family_for_glm_likelihood(&self.likelihood);
        let fam_scale = match &spec_response {
            ResponseFamily::Gaussian | ResponseFamily::Tweedie { .. } => {
                1.0 / self.likelihood.fixed_phi().unwrap_or(1.0)
            }
            _ => 1.0,
        };
        const BINOMIAL_MU_EPS: f64 = 1e-12;
        const MU_FLOOR: f64 = 1e-10;
        let is_binomial = matches!(spec_response, ResponseFamily::Binomial);

        // One jet sweep: collect μ (unclamped, for the deviance — matching
        // `excess`) and the per-row score (clamped μ, for `displaced_neg_score`).
        let mut mu_disp = Array1::<f64>::zeros(n);
        let mut ngs = Array1::<f64>::zeros(n);
        for i in 0..n {
            let eta_i = self.eta_hat[i] + s[i];
            let jet = match crate::mixture_link::inverse_link_jet_for_inverse_link(
                &self.inverse_link,
                eta_i,
            ) {
                Ok(jet) => jet,
                // `excess` returns +∞ on an infeasible inverse-link jet.
                Err(_) => return (f64::INFINITY, None),
            };
            mu_disp[i] = jet.mu;
            let mu_c = if is_binomial {
                jet.mu.clamp(BINOMIAL_MU_EPS, 1.0 - BINOMIAL_MU_EPS)
            } else {
                jet.mu.max(MU_FLOOR)
            };
            let v = pirls::variance_jet_for_weight_family(family, mu_c).v;
            if v.is_finite() && v > 0.0 {
                let d_dev_d_mu = -2.0 * self.prior_weights[i] * (self.y[i] - mu_c) / v * fam_scale;
                ngs[i] = d_dev_d_mu * jet.d1 / (2.0 * self.phi);
            }
        }

        let dev_disp = crate::pirls::calculate_deviance(
            self.y.view(),
            &mu_disp,
            &self.likelihood,
            self.prior_weights.view(),
        );
        if !dev_disp.is_finite() {
            return (f64::INFINITY, None);
        }
        let neg_loglik_diff = (dev_disp - self.base_deviance) / (2.0 * self.phi);
        let mut penalty_term = 0.0_f64;
        for (score, &lam) in self.penalty_scores.iter().zip(self.lambdas.iter()) {
            penalty_term += lam * score.dot(&delta);
        }
        let mut curv = 0.0_f64;
        for i in 0..s.len() {
            curv += self.weights_obs[i] * s[i] * s[i];
        }
        let excess = neg_loglik_diff + penalty_term - 0.5 * curv;
        if excess.is_finite() {
            (excess, Some(ngs))
        } else {
            (excess, None)
        }
    }

    /// Batched excess + displaced score over all importance draws (#784/#1082
    /// hot path). The per-draw cost of [`Self::excess_with_displaced_neg_score`]
    /// is dominated by the design matvec `s = X_t · δ` (O(n·p)), repeated
    /// `n_draws` times (up to 4096). Those matvecs share the SAME design `X_t`
    /// and the SAME block frame `V_b`, so they batch into two dense matrix–matrix
    /// products (BLAS-3) instead of `n_draws` matrix–vector products (BLAS-2):
    ///
    /// ```text
    ///   Δ = V_b · T            (p × n_draws)   T = draws (m × n_draws)
    ///   S = X_t · Δ            (n × n_draws)   one big GEMM, the win
    /// ```
    ///
    /// Column `s` of `S` is exactly `fast_av(X_t, V_b · t_s)` — the same vector
    /// the serial path forms — and everything downstream (the inverse-link jet
    /// sweep, deviance, penalty-score and curvature terms) is then computed
    /// per-column with byte-for-byte the same arithmetic as the serial
    /// `excess_with_displaced_neg_score`. Only the matvec→GEMM reassociation can
    /// perturb `S` (faer reduces the inner `p`-sum the same way per output
    /// element regardless of the RHS column count, so this is at the level of
    /// floating-point reassociation, not a different estimator).
    fn excess_with_displaced_neg_score_batch(
        &self,
        draws: &Array2<f64>,
    ) -> Vec<(f64, Option<Array1<f64>>)> {
        let m = self.block_lambdas.len();
        let n = self.eta_hat.len();
        let n_draws = draws.ncols();
        assert_eq!(
            draws.nrows(),
            m,
            "posterior displacement draw rows must match smoothing block count"
        );

        // δ-columns: Δ = V_b · T  (p × n_draws). Cheap (O(p·m·n_draws)) and kept
        // identical to the serial `block_vecs.dot(t)` per column.
        let delta_all = gam_linalg::faer_ndarray::fast_ab(&self.block_vecs, draws);
        // s-columns: S = X_t · Δ  (n × n_draws). THE batched matvec — one GEMM
        // replacing `n_draws` separate `fast_av(x_transformed, δ_s)` calls.
        let s_all = gam_linalg::faer_ndarray::fast_ab(self.x_transformed, &delta_all);

        // Family constants mirrored from `excess_with_displaced_neg_score`.
        let spec_response = reml_spec(&self.likelihood).response.clone();
        let family = pirls::weight_family_for_glm_likelihood(&self.likelihood);
        let fam_scale = match &spec_response {
            ResponseFamily::Gaussian | ResponseFamily::Tweedie { .. } => {
                1.0 / self.likelihood.fixed_phi().unwrap_or(1.0)
            }
            _ => 1.0,
        };
        const BINOMIAL_MU_EPS: f64 = 1e-12;
        const MU_FLOOR: f64 = 1e-10;
        let is_binomial = matches!(spec_response, ResponseFamily::Binomial);

        let mut out = Vec::with_capacity(n_draws);
        let mut mu_disp = Array1::<f64>::zeros(n);
        let mut ngs = Array1::<f64>::zeros(n);
        let mut delta = Array1::<f64>::zeros(self.block_vecs.nrows());
        'draw: for sidx in 0..n_draws {
            let s_col = s_all.column(sidx);
            ngs.fill(0.0);
            // One jet sweep at η̂ + s, identical to the serial fused path.
            for i in 0..n {
                let eta_i = self.eta_hat[i] + s_col[i];
                let jet = match crate::mixture_link::inverse_link_jet_for_inverse_link(
                    &self.inverse_link,
                    eta_i,
                ) {
                    Ok(jet) => jet,
                    Err(_) => {
                        out.push((f64::INFINITY, None));
                        continue 'draw;
                    }
                };
                mu_disp[i] = jet.mu;
                let mu_c = if is_binomial {
                    jet.mu.clamp(BINOMIAL_MU_EPS, 1.0 - BINOMIAL_MU_EPS)
                } else {
                    jet.mu.max(MU_FLOOR)
                };
                let v = pirls::variance_jet_for_weight_family(family, mu_c).v;
                if v.is_finite() && v > 0.0 {
                    let d_dev_d_mu =
                        -2.0 * self.prior_weights[i] * (self.y[i] - mu_c) / v * fam_scale;
                    ngs[i] = d_dev_d_mu * jet.d1 / (2.0 * self.phi);
                }
            }

            let dev_disp = crate::pirls::calculate_deviance(
                self.y.view(),
                &mu_disp,
                &self.likelihood,
                self.prior_weights.view(),
            );
            if !dev_disp.is_finite() {
                out.push((f64::INFINITY, None));
                continue;
            }
            let neg_loglik_diff = (dev_disp - self.base_deviance) / (2.0 * self.phi);
            delta.assign(&delta_all.column(sidx));
            let mut penalty_term = 0.0_f64;
            for (score, &lam) in self.penalty_scores.iter().zip(self.lambdas.iter()) {
                penalty_term += lam * score.dot(&delta);
            }
            let mut curv = 0.0_f64;
            for i in 0..n {
                curv += self.weights_obs[i] * s_col[i] * s_col[i];
            }
            let excess = neg_loglik_diff + penalty_term - 0.5 * curv;
            if excess.is_finite() {
                out.push((excess, Some(ngs.clone())));
            } else {
                out.push((excess, None));
            }
        }
        out
    }
}
