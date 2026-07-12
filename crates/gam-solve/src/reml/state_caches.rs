use super::*;

pub(crate) const TK_BLOCK_SIZE: usize = 128;

pub(crate) const TK_MAX_OBSERVATIONS: usize = 20_000;

pub(crate) const TK_MAX_COEFFICIENTS: usize = 2_000;

pub(crate) const ADAPTIVE_KKT_ETA: f64 = 0.1;

pub(crate) const ADAPTIVE_KKT_FLOOR_REML_DIVISOR: f64 = 100.0;

pub(crate) const TK_MAX_DENSE_WORK: usize = 5_000_000;

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

#[inline]
pub(crate) fn compute_gradient_for_tk(mode: super::reml_outer_engine::EvalMode) -> bool {
    mode != super::reml_outer_engine::EvalMode::ValueOnly
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
            hasher.write_bool(p.learns_smoothing());
        }
        AnalyticPenaltyKind::SoftmaxAssignmentSparsity(p) => {
            hasher.write_str("softmax-assignment-sparsity");
            hasher.write_usize(p.k_atoms);
            hasher.write_f64(p.temperature);
            hasher.write_f64(p.weight);
            hash_weight_schedule_option(hasher, &p.weight_schedule);
        }
        AnalyticPenaltyKind::OrderedBetaBernoulli(p) => {
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
        AnalyticPenaltyKind::SmoothThreshold(p) => {
            hasher.write_str("smooth_threshold");
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
pub(crate) fn reml_fixed_glm_dispersion(
    likelihood: &GlmLikelihoodSpec,
) -> Result<f64, EstimationError> {
    use gam_problem::ResolvedLikelihoodScale as Scale;
    let resolved = likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let phi = match resolved {
        // These likelihoods carry their complete scale in the family geometry:
        // NB through theta, Beta through its precision-dependent likelihood and
        // Hessian. Treating Beta precision as EDM dispersion double-scales EFS.
        Scale::Unit | Scale::NegativeBinomial { .. } | Scale::BetaPrecision { .. } => 1.0,
        Scale::FixedGaussian { phi } | Scale::Tweedie { phi, .. } => phi.value(),
        Scale::Gamma { .. } => resolved
            .gamma_phi()
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
        Scale::ProfiledGaussian => {
            return Err(EstimationError::InvalidInput(
                "profiled Gaussian has no fixed REML dispersion".to_string(),
            ));
        }
        Scale::Unspecified => {
            return Err(EstimationError::InvalidInput(
                "family has no scalar GLM dispersion".to_string(),
            ));
        }
    };
    if phi.is_finite() && phi > 0.0 {
        Ok(phi)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "{} REML dispersion must be finite and positive; got {phi}",
            likelihood.spec.response.name()
        )))
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
///   ΔF(t) = [D(η̂ + Xδ) − D(η̂)]/(2φ)          (= −[ℓ(β̂+δ) − ℓ(β̂)])
///           + (S β̂)·δ                          (penalty-score channel)
///           − ½ Σ_i W_i (Xδ)_i².               (likelihood-curvature subtraction)
///
/// `D/2` and its η-score come from one fallible family row oracle; `φ = 1` for
/// families whose reported deviance already carries the likelihood scale
/// (including fixed-scale Gaussian and Beta), and is the EDM dispersion for
/// unscaled Gamma/Tweedie deviance. The only place
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
    /// `ln|W_i|` (`-inf` for exact zero), certified once at target creation.
    pub(crate) weights_obs_log_abs: Array1<f64>,
    /// Response y and prior weights for the deviance.
    pub(crate) y: Array1<f64>,
    pub(crate) prior_weights: Array1<f64>,
    /// Family/link spec for the deviance and the inverse link.
    pub(crate) likelihood: GlmLikelihoodSpec,
    pub(crate) inverse_link: InverseLink,
    /// Divisor converting reported half-deviance to negative log-likelihood.
    pub(crate) phi: f64,
    /// Penalty scores `S_k β̂` per canonical penalty (unscaled by λ_k).
    /// Shared from the eval bundle's once-per-inner-solution cache
    /// (`EvalShared::canonical_penalty_scores_at_mode`).
    pub(crate) penalty_scores: Arc<Vec<Array1<f64>>>,
    /// `λ_k = e^{ρ_k}` per canonical penalty, aligned with `penalty_scores`.
    pub(crate) lambdas: Vec<f64>,
    /// Certified `D(eta_hat)/(2 phi)` on the exact row surface.
    pub(crate) base_scaled_half_deviance: f64,
    /// Its per-row eta gradient, cached once for the sampler moment channels.
    pub(crate) base_neg_score_at_mode: Array1<f64>,
}

impl Gam784BlockTarget<'_> {
    /// Map a whitened block displacement `t` to the coefficient displacement
    /// `δ = V_b t` and the per-row score `s = X_t δ`.
    pub(crate) fn displacement(&self, t: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let delta = self.block_vecs.dot(t);
        let s = gam_linalg::faer_ndarray::fast_av(self.x_transformed, &delta);
        (delta, s)
    }

    /// Evaluate `D(eta)/(2 phi)` and its full per-row eta gradient atomically.
    /// The row oracle owns both channels, including canonical logit tails and
    /// log-coordinate Bregman algebra for every log-link family.  An invalid
    /// row therefore invalidates the entire draw; no zero-score surrogate is
    /// ever paired with a different objective value.
    pub(crate) fn likelihood_surface_at(
        &self,
        eta: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>), EstimationError> {
        if !(self.phi.is_finite() && self.phi > 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "#784 likelihood scale must be finite and positive; got {}",
                self.phi
            )));
        }
        let rows = crate::pirls::deviance_eta_rows_with_log_measure_scale(
            self.y.view(),
            eta,
            &self.likelihood,
            &self.inverse_link,
            self.prior_weights.view(),
            -self.phi.ln(),
        )?;
        let half_values: Vec<f64> = rows.iter().map(|row| row.half_deviance).collect();
        let half_deviance =
            crate::pirls::stable_finite_signed_sum(&half_values, "#784 scaled half-deviance")?;
        let mut score = Array1::<f64>::zeros(rows.len());
        for (i, row) in rows.into_iter().enumerate() {
            let value = row.eta_score;
            if !value.is_finite() {
                return Err(EstimationError::PirlsRowGeometryUnrepresentable {
                    row: i,
                    quantity: "scaled deviance eta score",
                    eta: eta[i],
                    value,
                });
            }
            score[i] = value;
        }
        Ok((half_deviance, score))
    }

    pub(crate) fn neg_score_at(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        self.likelihood_surface_at(eta).map(|(_, score)| score)
    }

    /// `sum_i W_i s_i^2` on an exponent-scaled signed surface.  Squaring `s_i`
    /// before multiplying by a tiny `W_i` can overflow even when the weighted
    /// term is finite; scaling every term by the largest log magnitude avoids
    /// that false refusal.  One deterministic Neumaier pass preserves signed
    /// observed-curvature cancellation.
    pub(crate) fn observed_quadratic(
        &self,
        s: ndarray::ArrayView1<'_, f64>,
    ) -> Result<f64, EstimationError> {
        if s.len() != self.weights_obs.len() {
            return Err(EstimationError::InvalidInput(format!(
                "#784 observed quadratic length mismatch: scores={}, weights={}",
                s.len(),
                self.weights_obs.len()
            )));
        }
        let mut max_log = f64::NEG_INFINITY;
        for i in 0..s.len() {
            if !s[i].is_finite() {
                return Err(EstimationError::PirlsRowGeometryUnrepresentable {
                    row: i,
                    quantity: "#784 displacement score",
                    eta: self.eta_hat[i],
                    value: s[i],
                });
            }
            let log_term = if self.weights_obs[i] == 0.0 || s[i] == 0.0 {
                f64::NEG_INFINITY
            } else {
                self.weights_obs_log_abs[i] + 2.0 * s[i].abs().ln()
            };
            max_log = max_log.max(log_term);
        }
        if max_log == f64::NEG_INFINITY {
            return Ok(0.0);
        }
        let mut sum = 0.0_f64;
        let mut compensation = 0.0_f64;
        for i in 0..s.len() {
            if self.weights_obs[i] == 0.0 || s[i] == 0.0 {
                continue;
            }
            let log_term = self.weights_obs_log_abs[i] + 2.0 * s[i].abs().ln();
            let term = self.weights_obs[i].signum() * (log_term - max_log).exp();
            let next = sum + term;
            compensation += if sum.abs() >= term.abs() {
                (sum - next) + term
            } else {
                (term - next) + sum
            };
            sum = next;
        }
        let normalized = sum + compensation;
        if normalized == 0.0 {
            return Ok(0.0);
        }
        let value = normalized.signum() * (max_log + normalized.abs().ln()).exp();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(EstimationError::InvalidInput(
                "#784 signed observed quadratic is outside f64 range".to_string(),
            ))
        }
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
        let eta_disp = &self.eta_hat + &s;
        let Ok((scaled_half_deviance, _score)) = self.likelihood_surface_at(&eta_disp) else {
            return f64::INFINITY;
        };
        // −[ℓ(β̂+δ) − ℓ(β̂)] = [D_disp − D_base]/(2φ).
        let neg_loglik_diff = scaled_half_deviance - self.base_scaled_half_deviance;
        // Penalty-score channel (S β̂)·δ = Σ_k λ_k (S_k β̂)·δ.
        let mut penalty_term = 0.0_f64;
        for (score, &lam) in self.penalty_scores.iter().zip(self.lambdas.iter()) {
            penalty_term += lam * score.dot(&delta);
        }
        // Likelihood-curvature subtraction ½ Σ_i W_i s_i².
        let Ok(curv) = self.observed_quadratic(s.view()) else {
            return f64::INFINITY;
        };
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

    fn displaced_neg_score(&self, t: &Array1<f64>) -> Result<Array1<f64>, String> {
        let (_delta, s) = self.displacement(t);
        self.neg_score_at(&(&self.eta_hat + &s))
            .map_err(|error| error.to_string())
    }

    fn base_neg_score(&self) -> Result<Array1<f64>, String> {
        Ok(self.base_neg_score_at_mode.clone())
    }

    /// Fused excess + displaced score sharing one design matvec `s = X_t δ`
    /// and one atomic row-oracle sweep at `η̂ + s`. Each row's value and score
    /// are evaluated together on the same unprojected surface (#784, #1082).
    fn excess_with_displaced_neg_score(&self, t: &Array1<f64>) -> (f64, Option<Array1<f64>>) {
        let (delta, s) = self.displacement(t);
        let eta_disp = &self.eta_hat + &s;
        let Ok((scaled_half_deviance, ngs)) = self.likelihood_surface_at(&eta_disp) else {
            return (f64::INFINITY, None);
        };
        let neg_loglik_diff = scaled_half_deviance - self.base_scaled_half_deviance;
        let mut penalty_term = 0.0_f64;
        for (score, &lam) in self.penalty_scores.iter().zip(self.lambdas.iter()) {
            penalty_term += lam * score.dot(&delta);
        }
        let Ok(curv) = self.observed_quadratic(s.view()) else {
            return (f64::INFINITY, None);
        };
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

        let mut out = Vec::with_capacity(n_draws);
        let mut delta = Array1::<f64>::zeros(self.block_vecs.nrows());
        for sidx in 0..n_draws {
            let s_col = s_all.column(sidx);
            let mut eta_disp = self.eta_hat.clone();
            for i in 0..n {
                eta_disp[i] += s_col[i];
            }
            let Ok((scaled_half_deviance, ngs)) = self.likelihood_surface_at(&eta_disp) else {
                out.push((f64::INFINITY, None));
                continue;
            };
            let neg_loglik_diff = scaled_half_deviance - self.base_scaled_half_deviance;
            delta.assign(&delta_all.column(sidx));
            let mut penalty_term = 0.0_f64;
            for (score, &lam) in self.penalty_scores.iter().zip(self.lambdas.iter()) {
                penalty_term += lam * score.dot(&delta);
            }
            let Ok(curv) = self.observed_quadratic(s_col) else {
                out.push((f64::INFINITY, None));
                continue;
            };
            let excess = neg_loglik_diff + penalty_term - 0.5 * curv;
            if excess.is_finite() {
                out.push((excess, Some(ngs)));
            } else {
                out.push((excess, None));
            }
        }
        out
    }
}

#[cfg(test)]
mod exact_deviance_state_cache_tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn observed_quadratic_scales_before_squaring_and_preserves_sign() {
        let x = Array2::<f64>::zeros((2, 1));
        let weights_obs = array![1.0e-320, -1.0e-320];
        let weights_obs_log_abs = weights_obs.mapv(|weight| weight.abs().ln());
        let target = Gam784BlockTarget {
            x_transformed: &x,
            block_vecs: Array2::zeros((1, 1)),
            block_lambdas: array![1.0],
            eta_hat: array![0.0, 0.0],
            weights_obs,
            weights_obs_log_abs,
            y: array![0.0, 0.0],
            prior_weights: array![1.0, 1.0],
            likelihood: GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                ResponseFamily::Poisson,
                InverseLink::Standard(StandardLink::Log),
            )),
            inverse_link: InverseLink::Standard(StandardLink::Log),
            phi: 1.0,
            penalty_scores: Arc::new(Vec::new()),
            lambdas: Vec::new(),
            base_scaled_half_deviance: 0.0,
            base_neg_score_at_mode: array![0.0, 0.0],
        };
        let s = array![1.0e200, 5.0e199];
        let observed = target
            .observed_quadratic(s.view())
            .expect("weighted quadratic");
        let first = (target.weights_obs[0].ln() + 2.0 * s[0].ln()).exp();
        let second = (target.weights_obs[1].abs().ln() + 2.0 * s[1].ln()).exp();
        let expected = first - second;
        approx::assert_relative_eq!(observed, expected, max_relative = 2.0e-14);
    }
}
