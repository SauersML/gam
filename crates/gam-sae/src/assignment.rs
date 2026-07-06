//! Assignment gates and sparsity-prior helpers for the SAE manifold term.
//! Mechanically split from `sae_manifold.rs`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::SaeManifoldRho;
use gam_solve::evidence::{HybridAtomCandidate, HybridAtomChoice, select_hybrid_atom};
use gam_terms::analytic_penalties::{
    AnalyticPenalty, IBPAssignmentPenalty, IbpHessianDiagThirdChannels,
    SoftmaxAssignmentSparsityPenalty, resolve_learnable_weight,
};
use gam_terms::latent::{LatentCoordValues, LatentIdMode, LatentManifold};

/// Shared per-atom row support measure.
///
/// The weights are the fitted assignment masses `w_i = a_{ik}` for one atom:
/// non-negative, unnormalised, and on the same scale as the reconstruction gate.
/// Diagnostics should read atom occupancy through this object instead of
/// re-deriving hard owner sets or local soft-mass sums. Three sizes are exposed
/// because they answer different questions:
///
/// * [`Self::mass`] is the soft occupancy `Σ_i w_i`.
/// * [`Self::fisher_n`] is the reconstruction-information count `Σ_i w_i²`,
///   matching the rank-charge Gram `Φᵀdiag(w²)Φ`.
/// * [`Self::ess`] is the scale-invariant Kish effective support
///   `(Σ_i w_i)² / Σ_i w_i²`, the number of equally weighted rows represented by
///   the support distribution.
#[derive(Clone, Debug)]
pub struct SupportMeasure {
    atom_idx: usize,
    weights: Array1<f64>,
    mass: f64,
    fisher_n: f64,
}

impl SupportMeasure {
    #[must_use = "support construction error must be handled"]
    pub fn from_assignment(assignment: &SaeAssignment, atom_idx: usize) -> Result<Self, String> {
        let assignments = assignment.assignments();
        Self::from_assignment_matrix(assignments.view(), atom_idx)
    }

    #[must_use = "support construction error must be handled"]
    pub fn from_assignment_matrix(
        assignments: ArrayView2<'_, f64>,
        atom_idx: usize,
    ) -> Result<Self, String> {
        let (_n, k) = assignments.dim();
        if atom_idx >= k {
            return Err(format!(
                "SupportMeasure::from_assignment_matrix: atom {atom_idx} out of range K={k}"
            ));
        }
        let weights = assignments.column(atom_idx).to_owned();
        Self::from_weights(atom_idx, weights)
    }

    #[must_use = "support construction error must be handled"]
    pub fn from_argmax_owners(
        owners: &[usize],
        atom_idx: usize,
        k_atoms: usize,
    ) -> Result<Self, String> {
        if atom_idx >= k_atoms {
            return Err(format!(
                "SupportMeasure::from_argmax_owners: atom {atom_idx} out of range K={k_atoms}"
            ));
        }
        let mut weights = Array1::<f64>::zeros(owners.len());
        for (row, &owner) in owners.iter().enumerate() {
            if owner >= k_atoms {
                return Err(format!(
                    "SupportMeasure::from_argmax_owners: row {row} owner {owner} out of range K={k_atoms}"
                ));
            }
            if owner == atom_idx {
                weights[row] = 1.0;
            }
        }
        Self::from_weights(atom_idx, weights)
    }

    #[must_use = "support construction error must be handled"]
    pub fn from_weights(atom_idx: usize, weights: Array1<f64>) -> Result<Self, String> {
        let mut mass = 0.0_f64;
        let mut fisher_n = 0.0_f64;
        for (row, &w) in weights.iter().enumerate() {
            if !(w.is_finite() && w >= 0.0) {
                return Err(format!(
                    "SupportMeasure::from_weights: row {row} has invalid support weight {w}"
                ));
            }
            mass += w;
            fisher_n += w * w;
        }
        Ok(Self {
            atom_idx,
            weights,
            mass,
            fisher_n,
        })
    }

    pub fn atom_idx(&self) -> usize {
        self.atom_idx
    }

    pub fn weights(&self) -> ArrayView1<'_, f64> {
        self.weights.view()
    }

    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    pub fn mass(&self) -> f64 {
        self.mass
    }

    pub fn fisher_n(&self) -> f64 {
        self.fisher_n
    }

    pub fn ess(&self) -> f64 {
        if self.fisher_n > 0.0 {
            (self.mass * self.mass) / self.fisher_n
        } else {
            0.0
        }
    }

    pub fn weight(&self, row: usize) -> f64 {
        self.weights[row]
    }

    pub fn positive_rows(&self) -> Vec<usize> {
        self.weights
            .iter()
            .enumerate()
            .filter_map(|(row, &w)| if w > 0.0 { Some(row) } else { None })
            .collect()
    }
}

/// #976 Layer-1 guard: cap on one accepted iteration's assignment-logit
/// update, in units of the gate temperature τ (the gate's natural length
/// scale — every assignment mode reads logits through `σ(·/τ)` /
/// `softmax(·/τ)`). A 4τ move spans the gate's whole soft range, so healthy
/// convergence is never throttled, but no single inner iteration can carry a
/// gate from contention to numerically-zero support: a collapse takes
/// multiple accepted iterations, which guarantees the per-iteration
/// active-mass guard observes the decay before it completes. The clamp is
/// applied where the step is realised; when it binds, the realised objective
/// is evaluated on the clamped state, so the Armijo comparison stays
/// value-consistent (the unclamped quadratic model is merely conservative,
/// and step halvings shrink the trial below the cap).
pub(crate) const SAE_ASSIGNMENT_LOGIT_STEP_CAP_TAUS: f64 = 4.0;

/// #976 Layer-1 guard: re-seed budget per atom per joint fit. One second
/// chance from a fresh basin; a second breach means the collapse is (locally)
/// the objective's verdict at the current hyperparameters, which is recorded
/// as a terminal collapse event and left for the structure-search death move
/// to adjudicate — re-seeding in a loop would fight the optimizer.
pub(crate) const SAE_ATOM_COLLAPSE_RESEED_BUDGET: usize = 1;

/// #976 Layer-1 guard (decoder arm): an atom whose decoder block Frobenius norm
/// has fallen to this fraction of the dictionary's MEDIAN decoder norm carries
/// no material reconstruction signal — it has degenerated to (near-)zero output
/// and decodes the same nothing as every other collapsed atom. This is the
/// real-data K>1 failure that the gate-mass floor cannot see: the assignment
/// gates can stay spread across rows (mass guard satisfied) while the decoders
/// all collapse to ~0, giving EV≈0 and a rank-deficient per-row coordinate
/// Hessian on every row (the 0→K·n evidence-deflation jump). The statistic is a
/// RATIO to the dictionary median so it is scale-free and never fires for a
/// uniformly-small but well-conditioned decoder; only an atom that has fallen
/// far behind its peers is caught. By construction this is a no-op for K=1
/// (a single atom has no peer to fall behind, and the median equals its own
/// norm), so the K=1 path is byte-for-byte unchanged.
pub(crate) const SAE_ATOM_DECODER_NORM_COLLAPSE_RATIO: f64 = 1.0e-3;

/// #976 / #1117 K>1 robustness: bounded DICTIONARY-level multi-start budget for
/// the simultaneous co-collapse arm (the EV-floor branch of
/// [`crate::manifold::SaeManifoldTerm::enforce_decoder_norm_guard`]).
/// Distinct from the per-atom [`SAE_ATOM_COLLAPSE_RESEED_BUDGET`] (= 1): that
/// budget governs reseeding ONE atom's gate logits against an optimizer that
/// keeps killing it, where a loop would fight the optimizer. A co-collapse
/// reseed is categorically different — it is a full-dictionary multi-start that
/// re-diversifies ALL atoms onto distinct principal directions of a FRESHLY
/// recomputed residual, so successive attempts explore genuinely different
/// basins. A single such reseed empirically cannot always break a K≥3 three-way
/// basin (identical (K, seed) flips EV≈0.40 ↔ 0.00), so this arm gets a small
/// bounded budget of independent multi-starts. S1 (guard surgery): it is consumed
/// ONLY at iteration > 0 when the whole dictionary's reconstruction EV is at or
/// below the SIGNAL-FREE null floor (`absolute_degeneracy_ev_floor` = `q / n`, the
/// classical null-`R²`) AND the reconstruction OUTPUT has co-vanished (output
/// energy at or below the same null level). Both hold only in a genuine #853/#976
/// co-collapse; a healthy fit (real OLMo K=1 ~0.22, K=2 ~0.40) and a
/// merely-uncompetitive present-decoder fit keep output energy and never consume
/// the budget — the former `0.5 × dense PCA ceiling` bar that tripped on those has
/// been retired.
pub(crate) const SAE_DICTIONARY_COCOLLAPSE_RESEED_BUDGET: usize = 3;

/// Machine-precision support cutoff for the smooth JumpReLU assignment prior,
/// in units of the gate temperature below the hard threshold. The forward gate
/// remains hard-zero at and below `threshold`, but the prior value/gradient and
/// compact Newton layout keep every logit with `(logit - threshold)/tau > -36`.
/// At the excluded edge `sigma(-36) ~= 2e-16`, so dropped value/gradient/Hessian
/// terms are below f64 noise instead of creating an algorithmic discontinuity.
pub(crate) const JUMPRELU_OPTIMIZATION_LOGIT_CUTOFF: f64 = -36.0;

/// Shared support predicate for JumpReLU optimization inclusion. This is
/// strictly weaker than the hard forward gate `logit > threshold`, which still
/// governs data-fit reconstruction and its logit JVP.
#[inline]
pub(crate) fn jumprelu_in_optimization_band(logit: f64, threshold: f64, temperature: f64) -> bool {
    (logit - threshold) / temperature > JUMPRELU_OPTIMIZATION_LOGIT_CUTOFF
}

/// Assignment prior/relaxation used by [`SaeAssignment`].
#[derive(Debug, Clone, Copy)]
pub enum AssignmentMode {
    /// Row-wise simplex assignment with entropy sparsity.
    Softmax { temperature: f64, sparsity: f64 },
    /// Deterministic concrete relaxation of a truncated IBP active set: each
    /// atom's gate is the INDEPENDENT prior-shrunk activation
    /// `σ(logit/temperature) · π_k`, with `π_k = (α/(α+1))^{k+1}` the
    /// stick-breaking prior mean (see [`ibp_map_row`]). These are per-atom gates,
    /// NOT mixture/simplex responsibilities: they are computed independently per
    /// column and do NOT sum to 1 across atoms (there is no row normalization).
    /// Each `a_k ∈ [0, π_k] ⊂ [0, 1)` is the relaxed "atom k is active in this
    /// row" indicator of a truncated IBP, not a share of a unit reconstruction
    /// budget.
    IBPMap {
        temperature: f64,
        alpha: f64,
        learnable_alpha: bool,
    },
    /// Hard-thresholded bounded gate: each atom is off (gate = 0) when its logit
    /// is at or below `threshold`, and on with a threshold-centered shifted
    /// sigmoid `σ((logit − threshold) / temperature) ∈ [0.5, 1)` above it.
    ///
    /// #1777 RENAMED from `JumpReLU` (an inaccurate name): this is NOT the
    /// literature JumpReLU activation `z·1[z>θ]`, which carries the thresholded
    /// MAGNITUDE `z`. This mode is a thresholded-logistic GATE (a hard-sigmoid
    /// gate): it carries no magnitude at all — its output is a bounded `[0, 1)`
    /// indicator. `ThresholdGate` names it for what it is. It is a member of the
    /// gate family (softmax simplex / IBP sigmoid / this hard gate); reconstruction
    /// magnitude lives entirely in the decoder curve `g_k(t) = φ(t)ᵀ B_k`. The
    /// discontinuity at `threshold` (0 → 0.5) is the intended "jump".
    ///
    /// BACK-COMPAT: the constructor [`Self::threshold_gate`] is the primary
    /// spelling; [`Self::jumprelu`] is retained as a deprecated alias, and the FFI
    /// string parser accepts both `"threshold_gate"` and the legacy `"jumprelu"`.
    ThresholdGate { temperature: f64, threshold: f64 },
}

/// Caller intent for assignment-mode admission. `Default` is the production
/// route: it never selects IBP-MAP implicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentModeRequest {
    Default,
    Softmax,
    ThresholdGate,
    IbpMap,
}

/// Scale-aware assignment admission result.
#[derive(Debug, Clone, Copy)]
pub struct AssignmentModeAdmission {
    pub mode: AssignmentMode,
    /// Train-time active-set cap to thread into `SaeManifoldTerm::set_softmax_active_cap`.
    pub top_k: Option<usize>,
}

/// #1033 — the fixed-form predictor that produces the ρ-invariant FROZEN routing
/// (amortized routing). Both forms are NO-learned-net deterministic functions of
/// the current dictionary; they differ in how faithfully they track the
/// dictionary as it evolves across outer iterates. Kept as alternatives so the
/// accuracy gate can pick whichever passes the fit-quality bar (the cheap
/// `Snapshot` if it suffices, the `ChartGeometry` distill otherwise).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingPredictor {
    /// Snapshot the current (converged) logits as the frozen routing — the
    /// cheapest fixed-form distill, exact at the dictionary it is taken from.
    /// Goes stale as the dictionary moves (needs a refresh to track), so it is the
    /// MVP/baseline form.
    Snapshot,
    /// Re-derive the per-(row, atom) routing logit from the atom's encode-chart
    /// geometry against the CURRENT dictionary: encode each row to its predicted
    /// coord `t̂`, reconstruct the amplitude-1 image `γ_k(t̂) = Bᵀφ(t̂)`, and map
    /// the reconstruction ALIGNMENT to a logit. This tracks the dictionary
    /// (a moved decoder changes `γ_k(t̂)` and hence the routing) without re-running
    /// the free-logit inner solve, so it is the default-readiness form when the
    /// snapshot proves too stale.
    ChartGeometry,
}

impl AssignmentMode {
    #[must_use]
    pub fn softmax(temperature: f64) -> Self {
        Self::Softmax {
            temperature,
            sparsity: 1.0,
        }
    }

    #[must_use]
    pub fn ibp_map(temperature: f64, alpha: f64, learnable_alpha: bool) -> Self {
        Self::IBPMap {
            temperature,
            alpha,
            learnable_alpha,
        }
    }

    /// #1777 — construct the hard-sigmoid [`Self::ThresholdGate`] (the accurate
    /// name for what was `jumprelu`). Primary spelling; [`Self::jumprelu`] is a
    /// deprecated alias kept for back-compat.
    #[must_use]
    pub fn threshold_gate(temperature: f64, threshold: f64) -> Self {
        Self::ThresholdGate {
            temperature,
            threshold,
        }
    }

    /// Back-compat alias for [`Self::threshold_gate`] (#1777): the mode is a
    /// hard-sigmoid gate, not the literature JumpReLU magnitude activation. Retained
    /// (NOT `#[deprecated]`, since the workspace denies warnings and many callers
    /// and the legacy `"jumprelu"` FFI token still use it) so existing code keeps
    /// compiling; new code should prefer `threshold_gate`.
    #[must_use]
    pub fn jumprelu(temperature: f64, threshold: f64) -> Self {
        Self::threshold_gate(temperature, threshold)
    }

    pub fn temperature(&self) -> f64 {
        match *self {
            AssignmentMode::Softmax { temperature, .. }
            | AssignmentMode::IBPMap { temperature, .. }
            | AssignmentMode::ThresholdGate { temperature, .. } => temperature,
        }
    }

    pub(crate) fn set_temperature(&mut self, new_temperature: f64) -> Result<(), String> {
        if !(new_temperature.is_finite() && new_temperature > 0.0) {
            return Err(format!(
                "AssignmentMode: temperature must be finite and positive; got {new_temperature}"
            ));
        }
        match self {
            AssignmentMode::Softmax { temperature, .. }
            | AssignmentMode::IBPMap { temperature, .. }
            | AssignmentMode::ThresholdGate { temperature, .. } => {
                *temperature = new_temperature;
            }
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        let temperature = self.temperature();
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(format!(
                "AssignmentMode: temperature must be finite and positive; got {temperature}"
            ));
        }
        match *self {
            AssignmentMode::Softmax { sparsity, .. } => {
                if !(sparsity.is_finite() && sparsity > 0.0) {
                    return Err(format!(
                        "AssignmentMode::Softmax: sparsity must be finite and positive; got {sparsity}"
                    ));
                }
            }
            AssignmentMode::IBPMap { alpha, .. } => {
                if !(alpha.is_finite() && alpha > 0.0) {
                    return Err(format!(
                        "AssignmentMode::IBPMap: alpha must be finite and positive; got {alpha}"
                    ));
                }
            }
            AssignmentMode::ThresholdGate { threshold, .. } => {
                if !threshold.is_finite() {
                    return Err(format!(
                        "AssignmentMode::ThresholdGate: threshold must be finite; got {threshold}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Resolve the effective truncated-IBP concentration `α` for this mode.
    ///
    /// `per_fit_override` is the #1777 PER-FIT override (from
    /// [`SaeAssignment::ibp_alpha_override`]) and is the SOURCE OF TRUTH when set.
    /// It falls back to the deprecated process-global [`ibp_alpha_override`] atomic
    /// only when the per-fit field is unset, then to the mode's own `α` /
    /// learnable schedule — so nothing breaks, but concurrent fits are isolatable
    /// via the per-fit field.
    pub(crate) fn resolved_ibp_alpha(
        &self,
        rho: &SaeManifoldRho,
        per_fit_override: Option<f64>,
    ) -> Option<f64> {
        match *self {
            AssignmentMode::IBPMap {
                alpha,
                learnable_alpha,
                ..
            } => Some(
                if let Some(over) = per_fit_override.or_else(ibp_alpha_override) {
                    // #1777 — the per-fit override (else the deprecated process-global
                    // one) flattens the ordered geometric prior π_k = (α/(α+1))^{k+1}
                    // so all K atoms can contribute to the reconstruction (the
                    // production α=1 gives a (0.5)^{k+1} schedule that structurally
                    // caps atoms 4..K → effective-K≈3). Forces the fixed value,
                    // bypassing the learnable schedule.
                    over
                } else if learnable_alpha {
                    resolve_learnable_weight(alpha, rho.log_lambda_sparse)
                } else {
                    alpha
                },
            ),
            _ => None,
        }
    }
}

/// Default large-K active cap derived from the data-per-atom ratio. When each
/// atom has fewer rows-per-atom than there are atoms (`N/K < K`), the dense
/// all-K routing model is the wrong scale; cap each row to the number of rows
/// available per atom. Otherwise the dense softmax path is admitted.
pub fn default_top_k_for_large_dictionary(n_obs: usize, k_atoms: usize) -> Option<usize> {
    if n_obs == 0 || k_atoms <= 1 {
        return None;
    }
    if n_obs >= k_atoms.saturating_mul(k_atoms) {
        return None;
    }
    let cap = n_obs.div_ceil(k_atoms).clamp(1, k_atoms - 1);
    Some(cap)
}

/// Admit the assignment mode for a fit size. The default route is softmax, with
/// a top-k cap at large K. IBP-MAP is a research-mode opt-in and is refused once
/// the large-K top-k admission engages.
pub fn admit_assignment_mode_for_size(
    request: AssignmentModeRequest,
    n_obs: usize,
    k_atoms: usize,
    temperature: f64,
    alpha: f64,
    learnable_alpha: bool,
    threshold: f64,
) -> Result<AssignmentModeAdmission, String> {
    if n_obs == 0 {
        return Err("admit_assignment_mode_for_size: n_obs must be positive".to_string());
    }
    if k_atoms == 0 {
        return Err("admit_assignment_mode_for_size: k_atoms must be positive".to_string());
    }
    let large_k_top = default_top_k_for_large_dictionary(n_obs, k_atoms);
    let admission = match request {
        AssignmentModeRequest::Default | AssignmentModeRequest::Softmax => AssignmentModeAdmission {
            mode: AssignmentMode::softmax(temperature),
            top_k: large_k_top,
        },
        AssignmentModeRequest::ThresholdGate => AssignmentModeAdmission {
            mode: AssignmentMode::threshold_gate(temperature, threshold),
            top_k: None,
        },
        AssignmentModeRequest::IbpMap => {
            if let Some(top_k) = large_k_top {
                return Err(format!(
                    "admit_assignment_mode_for_size: IBP-MAP is admitted only for explicit small-N research fits; N={n_obs}, K={k_atoms} requires top_k={top_k}"
                ));
            }
            AssignmentModeAdmission {
                mode: AssignmentMode::ibp_map(temperature, alpha, learnable_alpha),
                top_k: None,
            }
        }
    };
    admission.mode.validate()?;
    Ok(admission)
}

// #1026 — process-global IBP-α override (NaN sentinel = "unset → use the
// AssignmentMode's compiled α"). Lets ONE wheel sweep the prior-flattening axis
// from Python (`sae_set_ibp_alpha`) without recompiling the gam crate.
//
// CONCURRENCY WARNING: this is a PROCESS-GLOBAL atomic, not per-fit config. It is
// read by `ibp_alpha_override` from every IBP assignment evaluation in the
// process, so setting it affects ALL in-flight fits, not just the caller's. It is
// therefore UNSAFE to use across concurrent / parallel in-process fits — one
// fit's sweep value leaks into another's gates. It is safe only for serial,
// whole-process sweeps (the single-wheel FFI sweep driver it exists for). This
// should be migrated to per-fit configuration (threaded through `SaeManifoldRho`
// / the AssignmentMode) before any concurrent multi-fit use; that refactor is
// cross-cutting (FFI + term plumbing) and deliberately out of scope here.
static IBP_ALPHA_OVERRIDE_BITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0x7ff8_0000_0000_0000);

pub(crate) fn ibp_alpha_override() -> Option<f64> {
    let v = f64::from_bits(IBP_ALPHA_OVERRIDE_BITS.load(std::sync::atomic::Ordering::Relaxed));
    if v.is_finite() && v > 0.0 {
        Some(v)
    } else {
        None
    }
}

/// Set (or, with a non-finite/non-positive value, clear) the process-global
/// IBP-α override. Called from the gamfit Python FFI sweep driver.
///
/// PROCESS-GLOBAL / NOT CONCURRENCY-SAFE: this mutates one process-wide atomic
/// read by every IBP assignment in the process. Calling it while any other fit is
/// running leaks the override into that fit's gates. Use only for serial,
/// whole-process sweeps; do not use across concurrent in-process fits. See the
/// `IBP_ALPHA_OVERRIDE_BITS` note on migrating this to per-fit config.
pub fn set_ibp_alpha_override(alpha: f64) {
    IBP_ALPHA_OVERRIDE_BITS.store(alpha.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// Per-row latent assignment state.
///
/// The stored assignment parameter is `logits`; non-negative assignments are
/// derived by row-wise softmax, independent IBP-MAP sigmoid active indicators,
/// or JumpReLU gates. Softmax logits are canonicalized to the reference chart
/// `logits[K - 1] = 0`, so the row-local Newton coordinates contain only the
/// first `K - 1` logits (`0` coordinates for `K = 1`). Gate-style modes keep
/// all `K` logits as identifiable scalar parameters. `coords[k]` holds
/// `t_{.,k}` for atom `k`.
#[derive(Debug, Clone)]
pub struct SaeAssignment {
    pub logits: Array2<f64>,
    pub coords: Vec<LatentCoordValues>,
    pub mode: AssignmentMode,
    /// #1026 — per-atom UNGATED flag (length `K`, default all-`false`). An
    /// ungated atom is the dense linear/background tier: its per-row gate is
    /// fixed at `a_k ≡ 1` (it contributes `γ_k(t_k)` to EVERY row, unweighted),
    /// it is excluded from the other atoms' gate (for the column-separable
    /// IBP / JumpReLU modes the remaining atoms are computed independently, so
    /// they are unaffected), and its logit is NOT a free parameter — its
    /// logit-JVP, sparsity-prior gradient/curvature, and softmax majorizer
    /// contributions are all zero, leaving its logit slot an inert
    /// (ridge-regularized) null direction in the per-row Newton block. This lets
    /// the linear tier carry FULL-RANK reconstructible variance
    /// (`fitted = γ_ungated(x) + Σ_{gated} a_k·γ_k(x)`) so a linear SAE can reach
    /// the rank-(K·d) PCA ceiling, while the gated curved atoms still add sparse
    /// structure on the residual (#1026 routing-bound finding).
    pub ungated: Vec<bool>,
    /// #1033 — AMORTIZED / FROZEN routing. When `Some`, this `(n, K)` matrix is a
    /// ρ-INVARIANT predicted routing (the amortized `x → logits` map distilled
    /// from the frozen dictionary): the gates are computed from THESE logits
    /// instead of the free `self.logits`, and the logits are NOT optimized by the
    /// inner Newton (their gradient/curvature/prior contributions are zeroed,
    /// exactly as for [`Self::ungated`]). This is the generalization of an ungated
    /// atom from "pin the gate at 1" to "pin the gate at the predicted value": it
    /// makes the per-row routing a fixed function of `x` + the frozen dictionary,
    /// so the outer ρ-search reuses ONE routing instead of re-solving per-row
    /// gates every outer eval — the n-independent-outer-loop lever (#1033). `None`
    /// is the historical free-logit path (bit-identical).
    pub frozen_logits: Option<Array2<f64>>,
    /// #1777 PER-FIT IBP-α override — the source of truth for the truncated-IBP
    /// concentration `α` when set, replacing the process-global
    /// [`set_ibp_alpha_override`] atomic. `Some(α)` forces the fixed value
    /// (bypassing the learnable schedule), scoped to THIS assignment/fit so
    /// concurrent in-process fits are isolated. `None` ⇒ fall back to the
    /// deprecated process-global override, then to the `AssignmentMode`'s own `α` /
    /// learnable schedule (bit-identical to the historical path when neither
    /// override is set). Read via [`Self::resolved_ibp_alpha`]; set from the FFI
    /// through the term's `set_fit_config`.
    pub ibp_alpha_override: Option<f64>,
}

impl SaeAssignment {
    #[must_use = "build error must be handled"]
    pub fn new(
        logits: Array2<f64>,
        coords: Vec<LatentCoordValues>,
        temperature: f64,
    ) -> Result<Self, String> {
        Self::with_mode(logits, coords, AssignmentMode::softmax(temperature))
    }

    #[must_use = "build error must be handled"]
    pub fn with_mode(
        mut logits: Array2<f64>,
        coords: Vec<LatentCoordValues>,
        mode: AssignmentMode,
    ) -> Result<Self, String> {
        mode.validate()?;
        let n = logits.nrows();
        let k = logits.ncols();
        if coords.len() != k {
            return Err(format!(
                "SaeAssignment::new: coords length {} must equal K={k}",
                coords.len()
            ));
        }
        for (atom, coord) in coords.iter().enumerate() {
            if coord.n_obs() != n {
                return Err(format!(
                    "SaeAssignment::new: coord atom {atom} has n_obs={} but logits has {n}",
                    coord.n_obs()
                ));
            }
        }
        for row in 0..n {
            validate_finite_logits(logits.row(row), row)?;
        }
        if matches!(mode, AssignmentMode::Softmax { .. }) {
            canonicalize_softmax_logits(&mut logits);
        }
        Ok(Self {
            logits,
            coords,
            mode,
            ungated: vec![false; k],
            frozen_logits: None,
            ibp_alpha_override: None,
        })
    }

    /// #1033 — install a ρ-INVARIANT FROZEN routing (the amortized predicted
    /// logits; see [`SaeAssignment::frozen_logits`]). `predicted` must be
    /// `(n, K)`. With routing frozen, the gates are computed from `predicted` and
    /// the logits are excluded from the inner Newton (their gradient/curvature are
    /// inert, like an ungated atom's). Passing `None` restores the free-logit
    /// path.
    #[must_use = "build error must be handled"]
    pub fn with_frozen_routing(mut self, predicted: Option<Array2<f64>>) -> Result<Self, String> {
        if let Some(ref p) = predicted {
            if p.dim() != (self.n_obs(), self.k_atoms()) {
                return Err(format!(
                    "SaeAssignment::with_frozen_routing: predicted shape {:?} must be ({}, {})",
                    p.dim(),
                    self.n_obs(),
                    self.k_atoms()
                ));
            }
            if matches!(self.mode, AssignmentMode::Softmax { .. }) {
                return Err(
                    "SaeAssignment::with_frozen_routing: frozen routing under Softmax is rejected \
                     — the coupled simplex's entropy majorizer is assembled over the logits, which \
                     a frozen (non-optimized) routing would leave inconsistent; this separable-mode \
                     contract supports IBP-MAP and JumpReLU, whose per-atom gates have no \
                     simplex-coupled curvature to skip"
                        .to_string(),
                );
            }
            for row in 0..p.nrows() {
                validate_finite_logits(p.row(row), row)?;
            }
        }
        self.frozen_logits = predicted;
        Ok(self)
    }

    /// Whether the per-row routing is FROZEN (amortized) rather than free-logit.
    pub fn routing_is_frozen(&self) -> bool {
        self.frozen_logits.is_some()
    }

    /// The active routing logits for `row`: the frozen/predicted logits when
    /// routing is frozen (#1033), else the free `self.logits`. This is the SINGLE
    /// source the gate value reads, so freezing routing changes every gate
    /// consistently.
    pub(crate) fn routing_logits_row(&self, row: usize) -> ArrayView1<'_, f64> {
        match self.frozen_logits {
            Some(ref f) => f.row(row),
            None => self.logits.row(row),
        }
    }

    /// Whether atom `k`'s logit is held fixed (not a free Newton parameter): true
    /// for an ungated atom (#1026, gate pinned at 1) OR when routing is frozen
    /// (#1033, gate pinned at the predicted value). Both share the same inert
    /// treatment — zero logit-JVP, zero sparsity-prior gradient/curvature, zero
    /// softmax majorizer — so the logit slot never moves.
    pub(crate) fn logit_is_fixed(&self, k: usize) -> bool {
        self.routing_is_frozen() || self.ungated.get(k).copied().unwrap_or(false)
    }

    /// Per-atom mask (length `K`) of [`Self::logit_is_fixed`] — the logit slots
    /// that are NOT free Newton parameters (ungated #1026 and/or frozen-routing
    /// #1033). Precompute once per assembly and pass to the logit-JVP fillers so
    /// the data-fit Jacobian zeroes those rows. Under frozen routing every entry
    /// is `true`; with only ungated atoms it equals `ungated`; otherwise all
    /// `false` (the historical free-logit path).
    pub(crate) fn fixed_logit_mask(&self) -> Vec<bool> {
        if self.routing_is_frozen() {
            vec![true; self.k_atoms()]
        } else {
            self.ungated.clone()
        }
    }

    /// #1033 — install the simplest faithful AMORTIZED routing predictor: a
    /// fixed-form DISTILL of the current dictionary's routing, namely the current
    /// (converged) logits SNAPSHOTTED as the ρ-invariant frozen routing. This is
    /// the `x → logits` map "evaluated once at the frozen dictionary" — the
    /// routing the dictionary already expresses — held fixed so the outer ρ-search
    /// reuses it instead of re-optimizing the gates at every ρ. (A richer
    /// predictor that recomputes logits from `x` via the encode-atlas chart
    /// geometry is a later refinement; snapshotting the converged routing is the
    /// exact fixed-point it would target at the frozen dictionary.) Rejected for
    /// Softmax for the same simplex-coupling reason as [`Self::with_frozen_routing`].
    #[must_use = "build error must be handled"]
    pub fn freeze_routing_from_current_logits(self) -> Result<Self, String> {
        let snapshot = self.logits.clone();
        self.with_frozen_routing(Some(snapshot))
    }

    /// #1033 — in-place variant of [`Self::freeze_routing_from_current_logits`]
    /// for callers holding `&mut SaeAssignment` (e.g. inside a `SaeManifoldTerm`),
    /// where moving the assignment out is awkward. Same contract: snapshot the
    /// current logits as the ρ-invariant frozen routing; reject Softmax.
    pub fn freeze_routing_in_place(&mut self) -> Result<(), String> {
        if matches!(self.mode, AssignmentMode::Softmax { .. }) {
            return Err(
                "SaeAssignment::freeze_routing_in_place: frozen routing under Softmax is rejected \
                 (coupled-simplex entropy-majorizer); use IBP-MAP or JumpReLU"
                    .to_string(),
            );
        }
        let snapshot = self.logits.clone();
        for row in 0..snapshot.nrows() {
            validate_finite_logits(snapshot.row(row), row)?;
        }
        self.frozen_logits = Some(snapshot);
        Ok(())
    }

    /// #1033 — install an explicit predicted routing in place (the
    /// [`RoutingPredictor::ChartGeometry`] output), `&mut self` variant of
    /// [`Self::with_frozen_routing`]. `predicted` must be `(n, K)`; rejects Softmax
    /// (separable-mode contract) and non-finite predictions.
    pub fn set_frozen_routing_in_place(&mut self, predicted: Array2<f64>) -> Result<(), String> {
        if predicted.dim() != (self.n_obs(), self.k_atoms()) {
            return Err(format!(
                "SaeAssignment::set_frozen_routing_in_place: predicted shape {:?} must be ({}, {})",
                predicted.dim(),
                self.n_obs(),
                self.k_atoms()
            ));
        }
        if matches!(self.mode, AssignmentMode::Softmax { .. }) {
            return Err(
                "SaeAssignment::set_frozen_routing_in_place: frozen routing under Softmax is \
                 rejected (coupled-simplex entropy-majorizer); use IBP-MAP or JumpReLU"
                    .to_string(),
            );
        }
        for row in 0..predicted.nrows() {
            validate_finite_logits(predicted.row(row), row)?;
        }
        self.frozen_logits = Some(predicted);
        Ok(())
    }

    /// #1033 — lift the frozen routing, restoring the free-logit search path.
    pub fn thaw_routing(&mut self) {
        self.frozen_logits = None;
    }

    /// #1026 — designate which atoms are UNGATED (the dense linear/background
    /// tier; see [`SaeAssignment::ungated`]). `flags` must have length `K`.
    ///
    /// Ungating is defined for the COLUMN-SEPARABLE gate modes (IBP-MAP and
    /// JumpReLU): each atom's gate is an independent per-atom function of its own
    /// logit, so pinning one atom to `a_k ≡ 1` leaves every other atom's gate
    /// exactly as computed. Softmax is a coupled simplex (`Σ_k a_k = 1` over all
    /// `K`), so a unit gate for one atom is only well defined relative to a
    /// gated-subset renormalization that must also be reflected in the logit-JVP
    /// and the entropy majorizer; this constructor's contract is restricted to
    /// the separable modes, and an ungated atom under Softmax is REJECTED here so
    /// the inner solve never runs on a value/gradient-mismatched gate. Callers
    /// wanting a dense background tier under Softmax route it as an IBP-MAP or
    /// JumpReLU atom.
    #[must_use = "build error must be handled"]
    pub fn with_ungated(mut self, flags: Vec<bool>) -> Result<Self, String> {
        if flags.len() != self.k_atoms() {
            return Err(format!(
                "SaeAssignment::with_ungated: flags length {} must equal K={}",
                flags.len(),
                self.k_atoms()
            ));
        }
        if matches!(self.mode, AssignmentMode::Softmax { .. }) && flags.iter().any(|&u| u) {
            return Err(
                "SaeAssignment::with_ungated: an ungated atom under Softmax routing is \
                 rejected — the coupled simplex requires a gated-subset renormalization \
                 reflected in the logit-JVP and entropy majorizer, which this separable-mode \
                 contract does not perform; route a dense background tier as IBP-MAP or JumpReLU"
                    .to_string(),
            );
        }
        self.ungated = flags;
        Ok(self)
    }

    /// Whether any atom is ungated (the #1026 background tier is engaged).
    pub fn has_ungated(&self) -> bool {
        self.ungated.iter().any(|&u| u)
    }

    pub fn n_obs(&self) -> usize {
        self.logits.nrows()
    }

    pub fn k_atoms(&self) -> usize {
        self.logits.ncols()
    }

    pub fn total_coord_dim(&self) -> usize {
        self.coords.iter().map(|c| c.latent_dim()).sum()
    }

    pub fn assignment_coord_dim(&self) -> usize {
        match self.mode {
            AssignmentMode::Softmax { .. } => self.k_atoms().saturating_sub(1),
            AssignmentMode::IBPMap { .. } | AssignmentMode::ThresholdGate { .. } => self.k_atoms(),
        }
    }

    pub fn row_block_dim(&self) -> usize {
        self.assignment_coord_dim() + self.total_coord_dim()
    }

    pub fn coord_offsets(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.k_atoms());
        let mut cursor = self.assignment_coord_dim();
        for coord in &self.coords {
            out.push(cursor);
            cursor += coord.latent_dim();
        }
        out
    }

    pub fn assignments(&self) -> Array2<f64> {
        let n = self.n_obs();
        let k = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let a = self.assignments_row(row);
            for atom in 0..k {
                out[[row, atom]] = a[atom];
            }
        }
        out
    }

    pub fn assignments_row(&self, row: usize) -> Array1<f64> {
        self.try_assignments_row(row)
            .expect("assignment logits must be finite")
    }

    pub fn try_assignments_row(&self, row: usize) -> Result<Array1<f64>, String> {
        self.try_assignments_row_with_alpha(row, None)
    }

    /// #1777 — the effective truncated-IBP `α` for this assignment at `rho`,
    /// honoring the PER-FIT [`Self::ibp_alpha_override`] first (source of truth),
    /// then the deprecated process-global override, then the mode's own schedule.
    /// The single seam every gate/jet/prior site reads so the per-fit override is
    /// applied consistently. `None` for non-IBP modes.
    pub(crate) fn resolved_ibp_alpha(&self, rho: &SaeManifoldRho) -> Option<f64> {
        self.mode.resolved_ibp_alpha(rho, self.ibp_alpha_override)
    }

    /// Whether the truncated-IBP concentration α is a FREE outer parameter that
    /// varies with ρ (`rho.log_lambda_sparse`). α is learnable ONLY when the mode
    /// requests it AND no override (per-fit #1777 or the deprecated process-global
    /// one) pins it: an override forces the fixed value and bypasses the learnable
    /// schedule (see [`AssignmentMode::resolved_ibp_alpha`]), so α's ρ-derivatives
    /// are then identically zero and every prior / log-det / IFT term must treat α
    /// as a constant to stay consistent with the forward gate. `false` for non-IBP
    /// modes. (#Bug6)
    pub(crate) fn effective_alpha_is_learnable(&self) -> bool {
        match self.mode {
            AssignmentMode::IBPMap { learnable_alpha, .. } => {
                learnable_alpha
                    && self.ibp_alpha_override.is_none()
                    && ibp_alpha_override().is_none()
            }
            _ => false,
        }
    }

    /// #1777 — install (or clear, with `None`) the PER-FIT IBP-α override on this
    /// assignment. Source of truth used by [`Self::resolved_ibp_alpha`]; the FFI
    /// reaches it through the term's `set_fit_config`.
    pub fn set_ibp_alpha_override(&mut self, alpha: Option<f64>) {
        self.ibp_alpha_override = alpha;
    }

    pub(crate) fn try_assignments_row_for_rho(
        &self,
        row: usize,
        rho: &SaeManifoldRho,
    ) -> Result<Array1<f64>, String> {
        self.try_assignments_row_with_alpha(row, self.resolved_ibp_alpha(rho))
    }

    fn try_assignments_row_with_alpha(
        &self,
        row: usize,
        resolved_ibp_alpha: Option<f64>,
    ) -> Result<Array1<f64>, String> {
        // #1033 — read the ACTIVE routing logits: the ρ-invariant frozen/predicted
        // logits when routing is frozen, else the free `self.logits`. This single
        // source makes the gate value ρ-invariant under frozen routing (the
        // amortized-routing lever) and bit-identical to the historical path when
        // not frozen.
        let routing = self.routing_logits_row(row);
        validate_finite_logits(routing, row)?;
        // Only Softmax collapses to a fixed assignment at K==1: its
        // assignment_coord_dim is K-1 = 0, so there is no free logit. IBPMap and
        // JumpReLU keep a free per-atom gate logit even at K==1
        // (assignment_coord_dim = K = 1), so they must fall through to their real
        // row functions or the logit would move the prior but not the gate.
        if self.k_atoms() == 1 && matches!(self.mode, AssignmentMode::Softmax { .. }) {
            return Ok(Array1::from_vec(vec![1.0]));
        }
        let mut row_gates = match self.mode {
            AssignmentMode::Softmax { temperature, .. } => softmax_row(routing, temperature),
            AssignmentMode::IBPMap {
                temperature, alpha, ..
            } => ibp_map_row(routing, temperature, resolved_ibp_alpha.unwrap_or(alpha)),
            AssignmentMode::ThresholdGate {
                temperature,
                threshold,
            } => jumprelu_row(routing, temperature, threshold),
        };
        // #1026 — ungated (background-tier) atoms have a fixed unit gate. For the
        // column-separable IBP / JumpReLU modes the other atoms' gates are
        // computed independently above, so overwriting the ungated entries to 1.0
        // leaves the gated atoms exactly as they were; the ungated atom then
        // contributes `γ_k(t_k)` unweighted to every row. (Softmax + ungated is
        // rejected at `with_ungated`, so no simplex renormalization is needed
        // here.)
        if self.has_ungated() {
            for (k, gate) in row_gates.iter_mut().enumerate() {
                if self.ungated[k] {
                    *gate = 1.0;
                }
            }
        }
        Ok(row_gates)
    }

    /// #1557 — fill-into-caller-buffer twin of [`Self::try_assignments_row_for_rho`].
    ///
    /// Writes the EXACT SAME per-atom assignment row into `out` (length
    /// `k_atoms()`) instead of allocating a fresh `Array1`. Bit-identical to the
    /// allocating path; intended for the hot per-row loops that immediately
    /// consume the row, reusing a single scratch buffer across rows.
    pub(crate) fn try_assignments_row_for_rho_into(
        &self,
        row: usize,
        rho: &SaeManifoldRho,
        out: &mut [f64],
    ) -> Result<(), String> {
        self.try_assignments_row_with_alpha_into(row, self.resolved_ibp_alpha(rho), out)
    }

    /// #1557 — fill-into-caller-buffer twin of [`Self::try_assignments_row_with_alpha`].
    ///
    /// `out` must have length `k_atoms()`; it is fully overwritten with the same
    /// values the allocating variant would return. Every branch (early-return
    /// K==1 Softmax, the per-mode row math, the #1026 ungated overwrite) mirrors
    /// the allocating path exactly so the two are bit-identical.
    pub(crate) fn try_assignments_row_with_alpha_into(
        &self,
        row: usize,
        resolved_ibp_alpha: Option<f64>,
        out: &mut [f64],
    ) -> Result<(), String> {
        // `out` is sized `k_atoms()` by every caller; the per-mode helpers below
        // fully overwrite indices `0..k_atoms()`.
        let routing = self.routing_logits_row(row);
        validate_finite_logits(routing, row)?;
        // Mirror the allocating early-return: only Softmax collapses to a fixed
        // unit assignment at K==1.
        if self.k_atoms() == 1 && matches!(self.mode, AssignmentMode::Softmax { .. }) {
            out[0] = 1.0;
            return Ok(());
        }
        match self.mode {
            AssignmentMode::Softmax { temperature, .. } => {
                softmax_row_into(routing, temperature, out)
            }
            AssignmentMode::IBPMap {
                temperature, alpha, ..
            } => ibp_map_row_into(
                routing,
                temperature,
                resolved_ibp_alpha.unwrap_or(alpha),
                out,
            ),
            AssignmentMode::ThresholdGate {
                temperature,
                threshold,
            } => jumprelu_row_into(routing, temperature, threshold, out),
        };
        // #1026 — ungated (background-tier) atoms have a fixed unit gate, exactly
        // as in the allocating path.
        if self.has_ungated() {
            for (k, gate) in out.iter_mut().enumerate() {
                if self.ungated[k] {
                    *gate = 1.0;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn persist_resolved_ibp_alpha(&mut self, rho: &SaeManifoldRho) -> bool {
        let AssignmentMode::IBPMap {
            temperature,
            alpha,
            learnable_alpha: true,
        } = self.mode
        else {
            return false;
        };
        let resolved_alpha = resolve_learnable_weight(alpha, rho.log_lambda_sparse);
        self.mode = AssignmentMode::IBPMap {
            temperature,
            alpha: resolved_alpha,
            learnable_alpha: false,
        };
        true
    }

    pub(crate) fn assignments_for_rho(&self, rho: &SaeManifoldRho) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let k = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let a = self.try_assignments_row_for_rho(row, rho)?;
            for atom in 0..k {
                out[[row, atom]] = a[atom];
            }
        }
        Ok(out)
    }

    /// Flatten extension coordinates in row-major SAE layout:
    /// `(assignment chart_i, t_i0[0..d_0], ..., t_iK[0..d_K])` for every row.
    /// Softmax contributes the first `K - 1` reference logits and omits the
    /// fixed reference logit; gate-style assignment modes contribute all `K`
    /// logits.
    pub fn flatten_ext_coords(&self) -> Array1<f64> {
        let n = self.n_obs();
        let q = self.row_block_dim();
        let k = self.k_atoms();
        let assignment_dim = self.assignment_coord_dim();
        let offsets = self.coord_offsets();
        let mut out = Array1::<f64>::zeros(n * q);
        for row in 0..n {
            let base = row * q;
            for atom in 0..assignment_dim {
                out[base + atom] = self.logits[[row, atom]];
            }
            for atom in 0..k {
                let d = self.coords[atom].latent_dim();
                let t_row = self.coords[atom].row(row);
                for axis in 0..d {
                    out[base + offsets[atom] + axis] = t_row[axis];
                }
            }
        }
        out
    }

    #[must_use = "build error must be handled"]
    pub fn from_blocks_with_mode(
        logits: Array2<f64>,
        coord_blocks: Vec<Array2<f64>>,
        mode: AssignmentMode,
    ) -> Result<Self, String> {
        let coords = coord_blocks
            .iter()
            .map(|c| LatentCoordValues::from_matrix(c.view(), LatentIdMode::None))
            .collect();
        Self::with_mode(logits, coords, mode)
    }

    #[must_use = "build error must be handled"]
    pub fn from_blocks_with_mode_and_manifolds(
        logits: Array2<f64>,
        coord_blocks: Vec<Array2<f64>>,
        manifolds: Vec<LatentManifold>,
        mode: AssignmentMode,
    ) -> Result<Self, String> {
        if coord_blocks.len() != manifolds.len() {
            return Err(format!(
                "SaeAssignment::from_blocks_with_mode_and_manifolds: coord block length {} != manifold length {}",
                coord_blocks.len(),
                manifolds.len()
            ));
        }
        let coords = coord_blocks
            .iter()
            .zip(manifolds)
            .map(|(c, manifold)| {
                LatentCoordValues::from_matrix_with_manifold(c.view(), LatentIdMode::None, manifold)
            })
            .collect();
        Self::with_mode(logits, coords, mode)
    }
}

pub(crate) fn neutral_gate_weights(mode: AssignmentMode, k_atoms: usize) -> Array1<f64> {
    match mode {
        AssignmentMode::Softmax { .. } => Array1::from_elem(k_atoms, 1.0 / (k_atoms.max(1) as f64)),
        AssignmentMode::IBPMap {
            temperature, alpha, ..
        } => ibp_map_row(Array1::<f64>::zeros(k_atoms).view(), temperature, alpha),
        AssignmentMode::ThresholdGate { .. } => Array1::from_elem(k_atoms, 0.5),
    }
}

pub(crate) fn softmax_row(logits: ArrayView1<'_, f64>, temperature: f64) -> Array1<f64> {
    let k = logits.len();
    let inv_tau = 1.0 / temperature;
    let mut max_logit = f64::NEG_INFINITY;
    for &v in logits.iter() {
        max_logit = max_logit.max(v);
    }
    let mut out = Array1::<f64>::zeros(k);
    let mut sum = 0.0;
    for i in 0..k {
        let v = ((logits[i] - max_logit) * inv_tau).exp();
        out[i] = v;
        sum += v;
    }
    assert!(sum.is_finite() && sum > 0.0);
    for v in out.iter_mut() {
        *v /= sum;
    }
    out
}

pub(crate) fn validate_finite_logits(
    logits: ArrayView1<'_, f64>,
    row: usize,
) -> Result<(), String> {
    for (col, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!(
                "SaeAssignment: non-finite assignment logit at row {row}, atom {col}: {v}"
            ));
        }
    }
    Ok(())
}

pub(crate) fn canonicalize_softmax_logits(logits: &mut Array2<f64>) {
    let k = logits.ncols();
    if k == 0 {
        return;
    }
    if k == 1 {
        logits.fill(0.0);
        return;
    }
    for row in 0..logits.nrows() {
        let reference = logits[[row, k - 1]];
        for col in 0..k - 1 {
            logits[[row, col]] -= reference;
        }
        logits[[row, k - 1]] = 0.0;
    }
}

/// Truncated Indian-Buffet-Process stick-breaking prior *means*
/// `π_k = E[∏_{j=0}^{k} v_j] = (α/(α+1))^{k+1}` for k = 0, .., K-1, with sticks
/// `v_j ~ Beta(α, 1)` so `E[v_j] = α/(α+1)`. EVERY atom (including the first,
/// `π_0 = α/(α+1)`) carries the consistent Beta(α, 1) shrinkage: there is no
/// special-cased always-on base atom, so `α` behaves as a genuine IBP
/// concentration — larger `α` ⇒ heavier mass / slower decay, `α → 0` ⇒ all mass
/// collapses onto nothing, matching the stick-breaking limit. This is the
/// deterministic MAP / mean-field form of the IBP prior (the closed form the
/// analytic Newton / Hessian / Woodbury machinery differentiates); no sticks are
/// *sampled* here, the per-atom weight is the exact expectation of the
/// stick-breaking product. (#614: previously `π_0 = 1` left the first atom
/// unshrunk, which is the prior mean of NO stick at all and broke α's role as a
/// concentration; the consistent product mean restores genuine IBP semantics.)
pub(crate) fn ordered_geometric_shrinkage_prior(k_atoms: usize, alpha: f64) -> Array1<f64> {
    // Accumulate the geometric schedule `π_k = ratio^(k+1)` in LOG space so the
    // prior stays a finite *soft* weight even for large `K`. The naive product
    // `acc *= ratio` underflows to exact `0.0` once `ratio^(k+1) < f64::MIN_POSITIVE`
    // (e.g. `(0.1/1.1)^320`), which would turn the soft shrinkage prior into a
    // HARD mask: such atoms would receive zero assignment AND zero logit
    // gradient (the gradient is multiplied by `π_k`), so they could never
    // reactivate. Working in log-space and flooring the exponentiated weight at
    // the smallest positive normal keeps every atom's gradient path alive while
    // preserving the geometric ordering.
    let mut out = Array1::<f64>::zeros(k_atoms);
    let log_ratio = (alpha / (alpha + 1.0)).ln();
    for k in 0..k_atoms {
        // π_k = (α/(α+1))^{k+1}: the product of (k+1) i.i.d. Beta(α,1) stick
        // means, so atom 0 is also shrunk by one stick (E[v_0] = α/(α+1)).
        let log_pi = ((k + 1) as f64) * log_ratio;
        out[k] = log_pi.exp().max(f64::MIN_POSITIVE);
    }
    out
}

/// #1784 — K-aware default IBP concentration.
///
/// The ordered stick-breaking prior mean `π_k = (α/(α+1))^{k+1}` decays
/// GEOMETRICALLY in the atom INDEX, so a fixed small concentration (the
/// historical default `α = 1`, i.e. the `(0.5)^{k+1}` schedule) collapses to a
/// near-hard mask past atom ~3: a K-atom dictionary can then only ever place
/// mass on its first handful of atoms. That is exactly why the manifold SAE
/// UNDERFITS a linear dictionary of equal K on real activations, and why its
/// late atoms carry zero mass and leave the per-row joint Hessian rank-deficient
/// (the K = 128 `RemlConvergenceError`).
///
/// For a K-atom dictionary to actually USE all K atoms the IBP concentration must
/// scale with K. Choosing `α` so the LAST atom retains prior mass
/// `π_{K-1} = (α/(α+1))^K ≈ e^{-1}` spans the whole dictionary while keeping the
/// prior monotone (still an honest ordered stick-breaking prior — no atom is
/// structurally masked). Solving `(α/(α+1))^K = e^{-1}` gives
/// `α = 1/(exp(1/K) − 1) ≈ K − 1/2`. Floored at `1.0` so `K = 1` keeps the
/// historical `α = 1`.
pub fn default_ibp_concentration_for_k_atoms(k_atoms: usize) -> f64 {
    let k = k_atoms.max(1) as f64;
    // π_{K-1} = (α/(α+1))^K = e^{-1}  ⇒  α = 1/(e^{1/K} − 1).
    let alpha = 1.0 / ((1.0 / k).exp() - 1.0);
    alpha.max(1.0)
}

/// IBP-MAP row activations: per-atom sigmoid likelihood times the truncated
/// stick-breaking prior mean `π_k = (α/(α+1))^{k+1}`. With tied logits the prior
/// dominates and yields strictly decreasing activations in atom index, with the
/// first atom already shrunk by one Beta(α,1) stick mean (no unshrunk base atom).
pub fn ibp_map_row(logits: ArrayView1<'_, f64>, temperature: f64, alpha: f64) -> Array1<f64> {
    let prior = ordered_geometric_shrinkage_prior(logits.len(), alpha);
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        out[i] = gam_linalg::utils::stable_logistic(logits[i] / temperature) * prior[i];
    }
    out
}

/// IBP-MAP activations together with the diagonal Jacobian `∂z_k/∂l_k`,
/// shared with the torch autograd `Function` so the Python IBP-Gumbel path
/// applies the same stick-breaking prior mean `π_k = (α/(α+1))^{k+1}` and
/// temperature scaling as the Rust closed form. With `z_k = σ(l_k/τ)·π_k` the
/// per-atom derivative is
/// `σ(l_k/τ)(1 − σ(l_k/τ))·π_k / τ`; the map is diagonal in `k`, so the
/// Jacobian is returned as the per-atom diagonal vector.
#[must_use]
pub fn ibp_map_row_value_grad(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
    alpha: f64,
) -> (Array1<f64>, Array1<f64>) {
    let prior = ordered_geometric_shrinkage_prior(logits.len(), alpha);
    let inv_tau = 1.0 / temperature;
    let mut value = Array1::<f64>::zeros(logits.len());
    let mut grad = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        let sig = gam_linalg::utils::stable_logistic(logits[i] * inv_tau);
        value[i] = sig * prior[i];
        grad[i] = sig * (1.0 - sig) * inv_tau * prior[i];
    }
    (value, grad)
}

pub fn jumprelu_row(logits: ArrayView1<'_, f64>, temperature: f64, threshold: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        // Hard gate: strictly zero below threshold (the intended "jump"). Above
        // threshold the surrogate is centered at the threshold so the gate is
        // most informative exactly at the boundary it switches on:
        // σ((l−θ)/τ) ∈ [0.5, 1). Magnitude lives in the decoder, so the gate
        // stays bounded in [0, 1] by design.
        if logits[i] > threshold {
            out[i] = gam_linalg::utils::stable_logistic((logits[i] - threshold) / temperature);
        }
    }
    out
}

/// Bounded threshold-gate activations together with the straight-through
/// derivative `∂a_k/∂l_k` of the smooth surrogate, shared with the torch
/// autograd `Function` so the torch `jumprelu` lane applies the SAME bounded
/// gate as the closed-form fit (`jumprelu_row`): `a_k = σ((l_k − θ_k)/τ)` for
/// `l_k > θ_k` and exactly `0` otherwise — magnitude lives in the decoder, the
/// gate stays in `[0, 1)`. The returned derivative is the smooth surrogate's
/// `σ'((l_k − θ_k)/τ)/τ` evaluated on BOTH sides of the jump (a straight-through
/// estimator: the hard forward has zero derivative below threshold, which would
/// permanently kill gradient flow to gated-off atoms). `∂a_k/∂θ_k` is the
/// negation of the returned logit derivative; callers negate. `thresholds` is
/// per-atom (the torch lane learns one threshold per atom); the scalar
/// closed-form threshold is the constant-vector special case and the value
/// arithmetic matches `jumprelu_row` exactly there.
#[must_use]
pub fn jumprelu_row_value_grad(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
    thresholds: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>) {
    assert_eq!(
        logits.len(),
        thresholds.len(),
        "jumprelu_row_value_grad: logits/thresholds length mismatch"
    );
    let inv_tau = 1.0 / temperature;
    let mut value = Array1::<f64>::zeros(logits.len());
    let mut grad = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        let sig = gam_linalg::utils::stable_logistic((logits[i] - thresholds[i]) * inv_tau);
        if logits[i] > thresholds[i] {
            value[i] = sig;
        }
        grad[i] = sig * (1.0 - sig) * inv_tau;
    }
    (value, grad)
}

/// Batched bounded threshold-gate value+grad over an `(N, K)` logit matrix,
/// sharing the EXACT per-atom arithmetic of [`jumprelu_row_value_grad`] (same
/// `stable_logistic`, same `(l − θ) * inv_tau` order, same hard-jump gate) so a
/// single batched call is bit-identical to invoking the row kernel row-by-row.
///
/// `thresholds` is per-atom (length `K`, broadcast across the `N` rows). Returns
/// `(value, grad)`, each `(N, K)`:
///   * `value[i, k] = σ((l − θ)/τ) · 1[l > θ]` — the bounded `[0, 1)` gate,
///   * `grad[i, k]  = σ·(1 − σ)/τ` — the straight-through diagonal derivative
///     `∂a/∂l`, alive on BOTH sides of the jump (`∂a/∂θ = −∂a/∂l`).
///
/// This is the single source of truth for `gamfit.torch`'s bounded jumprelu
/// gate: the torch autograd `Function` crosses the FFI boundary ONCE with the
/// whole matrix instead of once per row.
#[must_use]
pub fn jumprelu_batch_value_grad(
    logits: ArrayView2<'_, f64>,
    temperature: f64,
    thresholds: ArrayView1<'_, f64>,
) -> (Array2<f64>, Array2<f64>) {
    let (n, k) = logits.dim();
    assert_eq!(
        k,
        thresholds.len(),
        "jumprelu_batch_value_grad: logits columns {k} != thresholds length {}",
        thresholds.len()
    );
    let inv_tau = 1.0 / temperature;
    let mut value = Array2::<f64>::zeros((n, k));
    let mut grad = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            let sig =
                gam_linalg::utils::stable_logistic((logits[[i, j]] - thresholds[j]) * inv_tau);
            if logits[[i, j]] > thresholds[j] {
                value[[i, j]] = sig;
            }
            grad[[i, j]] = sig * (1.0 - sig) * inv_tau;
        }
    }
    (value, grad)
}

#[cfg(test)]
mod jumprelu_batch_tests {
    use super::*;

    #[test]
    fn jumprelu_batch_matches_row_kernel_bit_for_bit() {
        // Deterministic (N, K) logit matrix with per-atom thresholds spanning
        // both sides of the jump.
        let n = 5usize;
        let k = 7usize;
        let temperature = 0.41_f64;
        let logits = Array2::from_shape_fn((n, k), |(i, j)| {
            ((i as f64) * 0.37 - (j as f64) * 0.19 + 0.11).sin()
        });
        let thresholds = Array1::from_shape_fn(k, |j| 0.2 - 0.05 * j as f64);

        let (value, grad) =
            jumprelu_batch_value_grad(logits.view(), temperature, thresholds.view());
        assert_eq!(value.dim(), (n, k));
        assert_eq!(grad.dim(), (n, k));

        // The batch kernel must reproduce the row kernel EXACTLY (same ops, same
        // order) — bit-for-bit, not merely within a tolerance.
        for i in 0..n {
            let (rv, rg) = jumprelu_row_value_grad(logits.row(i), temperature, thresholds.view());
            for j in 0..k {
                assert_eq!(value[[i, j]], rv[j], "value mismatch at row {i} atom {j}");
                assert_eq!(grad[[i, j]], rg[j], "grad mismatch at row {i} atom {j}");
            }
        }
    }
}

// #1557 — fill-into-caller-buffer variants of the three per-mode row functions.
// These compute the EXACT SAME values as `softmax_row` / `ibp_map_row` /
// `jumprelu_row` (same arithmetic, same order of operations) but write into a
// caller-provided `&mut [f64]` slice instead of heap-allocating a fresh
// `Array1<f64>` per call. The hot per-row loops (loss eval, arrow/Schur row
// loops) call these with a reused scratch buffer, eliminating millions of tiny
// K-sized allocations while staying bit-identical to the allocating path.
// `out` must have length `logits.len()`; the slice is fully overwritten.

pub(crate) fn softmax_row_into(logits: ArrayView1<'_, f64>, temperature: f64, out: &mut [f64]) {
    let k = logits.len();
    let inv_tau = 1.0 / temperature;
    let mut max_logit = f64::NEG_INFINITY;
    for &v in logits.iter() {
        max_logit = max_logit.max(v);
    }
    let mut sum = 0.0;
    for i in 0..k {
        let v = ((logits[i] - max_logit) * inv_tau).exp();
        out[i] = v;
        sum += v;
    }
    assert!(sum.is_finite() && sum > 0.0);
    for v in out.iter_mut() {
        *v /= sum;
    }
}

pub(crate) fn ibp_map_row_into(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
    alpha: f64,
    out: &mut [f64],
) {
    let prior = ordered_geometric_shrinkage_prior(logits.len(), alpha);
    for i in 0..logits.len() {
        out[i] = gam_linalg::utils::stable_logistic(logits[i] / temperature) * prior[i];
    }
}

pub(crate) fn jumprelu_row_into(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
    threshold: f64,
    out: &mut [f64],
) {
    for i in 0..logits.len() {
        // Match `jumprelu_row`: strictly zero below threshold, sigmoid surrogate
        // above. The buffer is fully overwritten (no read of prior contents).
        if logits[i] > threshold {
            out[i] = gam_linalg::utils::stable_logistic((logits[i] - threshold) / temperature);
        } else {
            out[i] = 0.0;
        }
    }
}

pub(crate) struct ActiveAtomLogitJvp<'a> {
    pub(crate) mode: AssignmentMode,
    pub(crate) k: usize,
    pub(crate) logit_k: f64,
    pub(crate) a_k: f64,
    pub(crate) decoded_k: ArrayView1<'a, f64>,
    pub(crate) fitted: ArrayView1<'a, f64>,
    pub(crate) ibp_prior: Option<&'a [f64]>,
    pub(crate) compact_index: usize,
    /// #1026 — when `true`, atom `k` is the ungated background tier: its gate is
    /// the constant `1`, so its logit-JVP `da_k/dl_k` is identically zero (the
    /// compact row is left untouched / zero).
    pub(crate) ungated: bool,
}

/// Fill the single compact logit-JVP row for active atom `k`, using the
/// per-mode assignment sensitivity `da_k/dl_k` contracted into the decoded /
/// fitted-corrected output direction. This is the active-set analogue of
/// [`fill_assignment_logit_jvp_rows`]: it reproduces that function's diagonal
/// logit row exactly for the atom `k`, but writes into a compact position of a
/// heterogeneous-`q` row block instead of the dense full-`K` Jacobian. `fitted`
/// is the row's *active-set* reconstruction so the softmax cross term
/// `(decoded_k − fitted)` is consistent with the curvature the compact block
/// carries.
pub(crate) fn fill_active_atom_logit_jvp(
    input: ActiveAtomLogitJvp<'_>,
    jac_compact: &mut Array2<f64>,
) {
    let ActiveAtomLogitJvp {
        mode,
        k,
        logit_k,
        a_k,
        decoded_k,
        fitted,
        ibp_prior,
        compact_index,
        ungated,
    } = input;
    let p = fitted.len();
    // #1026 — an ungated atom's gate is constant, so its logit-JVP is zero; leave
    // its compact row untouched (the buffer row is pre-zeroed by the caller).
    if ungated {
        return;
    }
    match mode {
        AssignmentMode::Softmax { temperature, .. } => {
            // da_k/dl_k contracted: a_k (decoded_k − fitted) / τ.
            let inv_tau = 1.0 / temperature;
            for out_col in 0..p {
                jac_compact[[compact_index, out_col]] =
                    a_k * (decoded_k[out_col] - fitted[out_col]) * inv_tau;
            }
        }
        AssignmentMode::IBPMap { temperature, .. } => {
            // z_k = σ(l_k/τ)·π_k ⇒ dz_k/dl_k = a_k(π_k − a_k)/(π_k τ) · π_k form
            // (matches `fill_assignment_logit_jvp_rows`).
            let inv_tau = 1.0 / temperature;
            let prior =
                ibp_prior.expect("fill_active_atom_logit_jvp: IBPMap requires precomputed prior");
            let pi_k = prior[k];
            let sig = if pi_k > 0.0 { a_k / pi_k } else { 0.0 };
            let dz = sig * (1.0 - sig) * inv_tau * pi_k;
            for out_col in 0..p {
                jac_compact[[compact_index, out_col]] = dz * decoded_k[out_col];
            }
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            // The data-fit Jacobian follows the hard forward gate. Below the
            // threshold the reconstruction contribution is exactly zero, so the
            // data-fit logit derivative must also be zero. Band-only atoms stay
            // in the compact row for prior terms, not phantom reconstruction
            // slope.
            if logit_k <= threshold {
                return;
            }
            let inv_tau = 1.0 / temperature;
            let activation = gam_linalg::utils::stable_logistic((logit_k - threshold) * inv_tau);
            let da = activation * (1.0 - activation) * inv_tau;
            for out_col in 0..p {
                jac_compact[[compact_index, out_col]] = da * decoded_k[out_col];
            }
        }
    }
}

pub(crate) fn fill_assignment_logit_jvp_rows(
    mode: AssignmentMode,
    logits: ArrayView1<'_, f64>,
    assignments: ArrayView1<'_, f64>,
    decoded: ArrayView2<'_, f64>,
    fitted: ArrayView1<'_, f64>,
    ibp_prior: Option<&[f64]>,
    // #1026 — per-atom ungated flags (length `K`). An ungated atom's gate is
    // constant, so its logit-JVP row is identically zero (skipped below). Empty
    // ⇒ no atom is ungated (the historical path, bit-identical).
    ungated: &[bool],
    local_jac: &mut Array2<f64>,
) {
    let is_ungated = |k: usize| ungated.get(k).copied().unwrap_or(false);
    match mode {
        AssignmentMode::Softmax { temperature, .. } => {
            if assignments.len() == 1 {
                return;
            }
            // da_k/dl_j = a_k (1[k=j] - a_j) / tau, contracted against
            // the assignment-weighted fitted row. The dense row layout uses
            // the reference-logit chart, so only columns `0..K-1` are free;
            // the final reference logit is fixed at zero and has no row.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() - 1 {
                if is_ungated(logit_col) {
                    continue;
                }
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = assignments[logit_col]
                        * (decoded[[logit_col, out_col]] - fitted[out_col])
                        * inv_tau;
                }
            }
        }
        AssignmentMode::IBPMap { temperature, .. } => {
            // Truncated-IBP concrete relaxation: z_k = σ(l_k/τ) · π_k where
            // π_k is the stick-breaking prior. Thus
            // dz_k/dl_k = σ(l/τ)(1-σ(l/τ))/τ · π_k = a_k(π_k - a_k)/(π_k τ).
            let inv_tau = 1.0 / temperature;
            let prior = ibp_prior
                .expect("fill_assignment_logit_jvp_rows: IBPMap requires precomputed prior");
            for logit_col in 0..assignments.len() {
                if is_ungated(logit_col) {
                    continue;
                }
                let pi_k = prior[logit_col];
                let a_k = assignments[logit_col];
                let sig = if pi_k > 0.0 { a_k / pi_k } else { 0.0 };
                let dz = sig * (1.0 - sig) * inv_tau * pi_k;
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = dz * decoded[[logit_col, out_col]];
                }
            }
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            // Data-fit sensitivity follows the hard forward gate: rows at or
            // below the threshold have zero reconstruction value and therefore
            // zero data-fit logit derivative. The wider machine-precision prior
            // support is a compact-layout/prior rule, not a data-fit STE.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() {
                if is_ungated(logit_col) || logits[logit_col] <= threshold {
                    continue;
                }
                let activation =
                    gam_linalg::utils::stable_logistic((logits[logit_col] - threshold) * inv_tau);
                let da = activation * (1.0 - activation) * inv_tau;
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = da * decoded[[logit_col, out_col]];
                }
            }
        }
    }
}

pub(crate) fn flat_logits(logits: ArrayView2<'_, f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for row in 0..logits.nrows() {
        let start = row * logits.ncols();
        for col in 0..logits.ncols() {
            out[start + col] = logits[[row, col]];
        }
    }
    out
}

/// Build the IBP sparsity penalty used by every assignment-prior term at `rho`,
/// honoring #Bug6 (α is FIXED to the forward-gate value whenever an override
/// pins it — `effective_alpha_is_learnable`, `resolved_ibp_alpha`) and #Bug4
/// (ungated atoms are inert columns excluded from value/gradient/curvature).
/// Returns `(penalty, rho_view)`; the fixed-α branch uses the `lambda_sparse`
/// weight convention with an empty `rho_view`.
fn ibp_prior_penalty(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    base_alpha: f64,
    temperature: f64,
) -> (IBPAssignmentPenalty, Array1<f64>) {
    let learnable = assignment.effective_alpha_is_learnable();
    let alpha_eff = if learnable {
        base_alpha
    } else {
        assignment.resolved_ibp_alpha(rho).unwrap_or(base_alpha)
    };
    let mut penalty =
        IBPAssignmentPenalty::new(assignment.k_atoms(), alpha_eff, temperature, learnable);
    // #Bug4: ungated atoms have a pinned unit gate and a held-constant logit — they
    // are inert columns excluded from the sparsity energy and all its derivatives.
    if assignment.has_ungated() {
        penalty.fixed_columns = Some(assignment.ungated.clone());
    }
    let rho_view = if learnable {
        Array1::from_vec(vec![rho.log_lambda_sparse])
    } else {
        penalty.weight = rho.lambda_sparse();
        Array1::zeros(0)
    };
    (penalty, rho_view)
}

pub(crate) fn assignment_prior_value(assignment: &SaeAssignment, rho: &SaeManifoldRho) -> f64 {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)
            .expect("assignment logits must be finite");
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return 0.0;
    }
    // #Bug4: under FROZEN routing every logit is inert (the gates come from the
    // ρ-invariant frozen predictor, not `self.logits`), so the whole assignment
    // sparsity prior is a constant with zero gradient/curvature — score it as 0 to
    // match the derivative-side treatment. (Softmax rejects frozen routing.)
    if assignment.routing_is_frozen() {
        return 0.0;
    }
    match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            penalty.value(target.view(), rho_view.view())
        }
        AssignmentMode::IBPMap {
            temperature, alpha, ..
        } => {
            let (penalty, rho_view) = ibp_prior_penalty(assignment, rho, alpha, temperature);
            penalty.value(target.view(), rho_view.view())
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            // Sparsity penalty uses the same threshold-centered surrogate and
            // machine-precision support as its gradient/Hessian. Data-fit
            // reconstruction remains hard-gated by `jumprelu_row`.
            let sparsity_strength = rho.lambda_sparse();
            let k = assignment.k_atoms();
            let mut acc = 0.0;
            for (idx, &logit) in target.iter().enumerate() {
                // #Bug4: skip ungated (inert) atoms' logits.
                if assignment.logit_is_fixed(idx % k) {
                    continue;
                }
                if jumprelu_in_optimization_band(logit, threshold, temperature) {
                    acc += gam_linalg::utils::stable_logistic((logit - threshold) / temperature);
                }
            }
            sparsity_strength * acc
        }
    }
}

pub(crate) fn assignment_prior_log_strength_derivative(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> f64 {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)
            .expect("assignment logits must be finite");
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return 0.0;
    }
    // #Bug4: frozen routing ⇒ inert prior ⇒ zero ρ-derivative.
    if assignment.routing_is_frozen() {
        return 0.0;
    }
    match assignment.mode {
        AssignmentMode::Softmax { .. } | AssignmentMode::ThresholdGate { .. } => {
            assignment_prior_value(assignment, rho)
        }
        AssignmentMode::IBPMap {
            temperature, alpha, ..
        } => {
            // #Bug6: `ibp_prior_penalty` picks the effective-α learnability (an
            // override forces the fixed-α value branch) and the #Bug4 ungated mask.
            let (penalty, rho_view) = ibp_prior_penalty(assignment, rho, alpha, temperature);
            if penalty.learnable_alpha {
                penalty.grad_rho(target.view(), rho_view.view())[0]
            } else {
                penalty.value(target.view(), rho_view.view())
            }
        }
    }
}

pub(crate) fn assignment_prior_log_strength_hdiag(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<Array1<f64>, String> {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return Ok(Array1::<f64>::zeros(target.len()));
    }
    // #Bug4: frozen routing ⇒ inert prior ⇒ zero curvature everywhere.
    if assignment.routing_is_frozen() {
        return Ok(Array1::<f64>::zeros(target.len()));
    }
    match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            let mut d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| {
                    "softmax assignment log-strength hessian diag unavailable".to_string()
                })?;
            // #Bug4: the softmax array method is not internally column-masked, so
            // zero any fixed-logit (ungated) column's curvature diagonal to match
            // `assignment_prior_grad_hdiag`'s post-hoc masking.
            mask_fixed_logit_entries(assignment, &mut d);
            Ok(d)
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            let sparsity_strength = rho.lambda_sparse();
            let inv_tau = 1.0 / temperature;
            let inv_tau2 = inv_tau * inv_tau;
            let k = assignment.k_atoms();
            let mut d = Array1::<f64>::zeros(target.len());
            for idx in 0..target.len() {
                // #Bug4: ungated (inert) atoms carry no curvature.
                if assignment.logit_is_fixed(idx % k) {
                    continue;
                }
                let logit = target[idx];
                if !jumprelu_in_optimization_band(logit, threshold, temperature) {
                    continue;
                }
                let activation = gam_linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                d[idx] = sparsity_strength * slope * (1.0 - 2.0 * activation) * inv_tau2;
            }
            Ok(d)
        }
        AssignmentMode::IBPMap {
            temperature, alpha, ..
        } => {
            let (penalty, rho_view) = ibp_prior_penalty(assignment, rho, alpha, temperature);
            let mut d = if penalty.learnable_alpha {
                penalty.hessian_diag_log_alpha_derivative(target.view(), rho_view.view())
            } else {
                penalty
                    .hessian_diag(target.view(), rho_view.view())
                    .ok_or_else(|| {
                        "IBP assignment log-strength hessian diag unavailable".to_string()
                    })?
            };
            // #Bug4: zero the curvature diagonal of ungated (inert) columns so the
            // log-det ρ-trace never charges them (the array methods are not
            // internally column-masked).
            mask_fixed_logit_entries(assignment, &mut d);
            Ok(d)
        }
    }
}

/// Zero the entries of a flat `(n·K)` per-(row, atom) array whose atom is a FIXED
/// (ungated / frozen) logit, so an inert atom contributes nothing to the term.
/// (#Bug4) No-op when nothing is fixed.
fn mask_fixed_logit_entries(assignment: &SaeAssignment, arr: &mut Array1<f64>) {
    if !(assignment.has_ungated() || assignment.routing_is_frozen()) {
        return;
    }
    let k = assignment.k_atoms();
    for idx in 0..arr.len() {
        if assignment.logit_is_fixed(idx % k) {
            arr[idx] = 0.0;
        }
    }
}

pub(crate) fn assignment_prior_log_strength_target_mixed(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<Array1<f64>, String> {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return Ok(Array1::<f64>::zeros(target.len()));
    }
    // #Bug4: frozen routing ⇒ inert prior ⇒ zero mixed derivative.
    if assignment.routing_is_frozen() {
        return Ok(Array1::<f64>::zeros(target.len()));
    }
    // #Bug6: the α-target mixed derivative only exists when α is EFFECTIVELY
    // learnable (mode-learnable AND not pinned by an override); otherwise α is a
    // constant and there is no log-α channel, so fall through to the grad_hdiag
    // (fixed-α) path.
    match assignment.mode {
        AssignmentMode::IBPMap {
            temperature, alpha, ..
        } if assignment.effective_alpha_is_learnable() => {
            let (penalty, rho_view) = ibp_prior_penalty(assignment, rho, alpha, temperature);
            let mut d = penalty.log_alpha_target_mixed_derivative(target.view(), rho_view.view());
            // #Bug4: inert columns carry no mixed derivative.
            mask_fixed_logit_entries(assignment, &mut d);
            Ok(d)
        }
        _ => Ok(assignment_prior_grad_hdiag(assignment, rho)?.0),
    }
}

pub(crate) fn assignment_prior_grad_hdiag(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    let mut grad = Array1::<f64>::zeros(target.len());
    let mut diag = Array1::<f64>::zeros(target.len());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return Ok((grad, diag));
    }
    let (sparsity_grad, sparsity_diag) = match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            let g = penalty.grad_target(target.view(), rho_view.view());
            let d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| "softmax assignment hessian diag unavailable".to_string())?;
            (g, d)
        }
        AssignmentMode::IBPMap {
            temperature, alpha, ..
        } => {
            // Scale the IBP assignment-sparsity prior by `lambda_sparse` in the
            // fixed-α branch (Softmax folds it into the penalty's rho coordinate;
            // JumpReLU multiplies `sparsity_strength`). #Bug6: `ibp_prior_penalty`
            // picks the EFFECTIVE-α learnability — an override pins α so the prior
            // uses the fixed-α weight convention and the resolved (override) α,
            // matching the forward gate — and installs the #Bug4 ungated mask. The
            // per-atom fixed-logit columns are additionally zeroed post-hoc below,
            // so the array (grad/hessian) methods need no internal column mask.
            let (penalty, rho_view) = ibp_prior_penalty(assignment, rho, alpha, temperature);
            let g = penalty.grad_target(target.view(), rho_view.view());
            let d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| "IBP assignment hessian diag unavailable".to_string())?;
            (g, d)
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            // Gradient and exact diagonal Hessian of the sparsity value's
            // threshold-centered surrogate σ((l−θ)/τ), using the same
            // machine-precision support as the value path. Data-fit JVP support
            // is narrower and follows the hard forward gate.
            let sparsity_strength = rho.lambda_sparse();
            let inv_tau = 1.0 / temperature;
            let inv_tau2 = inv_tau * inv_tau;
            let mut g = Array1::<f64>::zeros(target.len());
            let mut d = Array1::<f64>::zeros(target.len());
            for idx in 0..target.len() {
                let logit = target[idx];
                if !jumprelu_in_optimization_band(logit, threshold, temperature) {
                    continue;
                }
                let activation = gam_linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                g[idx] = sparsity_strength * slope * inv_tau;
                d[idx] = sparsity_strength * slope * (1.0 - 2.0 * activation) * inv_tau2;
            }
            (g, d)
        }
    };
    grad += &sparsity_grad;
    diag += &sparsity_diag;
    // #1026/#1033 — a FIXED logit (an ungated atom's, or every atom's under
    // frozen routing) is not a free parameter, so it carries NO sparsity-prior
    // gradient or curvature. Zero its flat columns (`flat_logits` is row-major
    // `row*K + atom`) so the assembled `gt` and `htt` logit slots stay zero —
    // matching the zero logit-JVP. The column-separable IBP / JumpReLU priors are
    // per-atom, so zeroing one atom's columns leaves the others' prior intact;
    // under frozen routing ALL atoms' logit columns are zeroed (the whole routing
    // is a fixed predicted function, not optimized).
    if assignment.has_ungated() || assignment.routing_is_frozen() {
        let k = assignment.k_atoms();
        for idx in 0..grad.len() {
            if assignment.logit_is_fixed(idx % k) {
                grad[idx] = 0.0;
                diag[idx] = 0.0;
            }
        }
    }
    Ok((grad, diag))
}

/// Build the exact IBP `hessian_diag` logit third-derivative channels (#1006)
/// for the SAE log-det adjoint Γ, using the SAME penalty configuration —
/// `alpha`/`tau`/`learnable_alpha` and the `lambda_sparse` weight convention —
/// that [`assignment_prior_grad_hdiag`] assembles into `htt`. Returns `None`
/// for non-IBP assignment modes (no cross-row empirical-π coupling to correct).
pub(crate) fn ibp_assignment_third_channels(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    majorize: bool,
) -> Result<Option<IbpHessianDiagThirdChannels>, String> {
    let AssignmentMode::IBPMap {
        temperature, alpha, ..
    } = assignment.mode
    else {
        return Ok(None);
    };
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    // #Bug6: build with the EFFECTIVE-α learnability and weight convention that
    // `assignment_prior_grad_hdiag` uses, so an α override differentiates the same
    // fixed-α operator. Fixed-logit columns are zeroed post-hoc below (the channel
    // arrays are not internally column-masked).
    let (penalty, rho_view) = ibp_prior_penalty(assignment, rho, alpha, temperature);
    let mut channels =
        penalty.hessian_diag_logit_third_channels(target.view(), rho_view.view(), majorize);
    // #1026/#1033 — zero the log-det third-derivative channels of FIXED-logit
    // atoms (ungated, or all atoms under frozen routing) so the #1006 θ-adjoint
    // differentiates the SAME (fixed-logit-zeroed) `htt` that
    // `assignment_prior_grad_hdiag` assembled. `k_max` columns, row-major `N·K`
    // for the per-(row,atom) arrays and length-`K` for the per-column ones.
    if assignment.has_ungated() || assignment.routing_is_frozen() {
        let k = channels.k_max;
        for idx in 0..channels.z_jac.len() {
            if assignment.logit_is_fixed(idx % k) {
                channels.z_jac[idx] = 0.0;
                channels.local_logit_third[idx] = 0.0;
                channels.m_channel[idx] = 0.0;
                channels.logit_curvature[idx] = 0.0;
            }
        }
        for atom in 0..k {
            if assignment.logit_is_fixed(atom) {
                channels.cross_row_d[atom] = 0.0;
                channels.cross_row_dd[atom] = 0.0;
            }
        }
    }
    Ok(Some(channels))
}

/// #1026 hybrid curved + linear-tail adjudication for one SAE atom slot.
///
/// A hybrid dictionary lets each atom slot be either a CURVED atom (its fitted
/// `latent_dim ≥ 1` manifold chart, whose decoded image may turn) or its LINEAR
/// special case (the euclidean-d=1-linear atom — one straight decoder direction,
/// `γ(t) = t·b`, zero turning). The two are nested: the linear atom is exactly
/// the curved family restricted to its straight sub-model, so a hybrid slot
/// cannot lose to pure-linear at matched actives — it strictly generalizes it.
///
/// This is the single call the SAE fitter makes per atom to choose the split by
/// EVIDENCE rather than fiat. It packages the atom's two already-fitted
/// candidates — each scored on the COMMON rank-aware Laplace scale (`−V = NLE`,
/// lower wins, identical to the union/mixture rungs) on the same rows — and
/// routes them through [`select_hybrid_atom`]. The curved candidate's fitted
/// turning `Θ` (from
/// [`crate::chart_canonicalization::d1_atom_fitted_turning`]) enters
/// as the decision feature: a `Θ → 0` atom yields to the cheaper linear tail by
/// construction (the dominance floor — a curved atom buys nothing on a straight
/// feature), a high-`Θ` atom takes the curved parameterization when its
/// curvature lowers the NLE by more than its extra-parameter price (the `Θ/√ε`
/// crossover).
///
/// `manifold` is the atom's fitted chart manifold; a non-curveable (already
/// Euclidean-flat) chart can only present the linear candidate, which this
/// helper enforces by ignoring any curved candidate offered for a flat chart —
/// a flat chart has no curvature to price, so the linear special case is its
/// only honest parameterization. Curveable charts present both candidates.
///
/// # Wiring into the fitter (the one call into `sae_manifold.rs`)
///
/// The post-fit pass in `sae_manifold.rs` already computes each d=1 atom's
/// fitted turning `Θ` (the read-only EV-vs-Θ diagnostic). To make the split
/// load-bearing, that pass supplies, per atom, the curved-candidate NLE +
/// parameter count + `Θ` and the linear-candidate NLE + parameter count (both
/// fitted on the atom's rows), and calls this helper; the returned
/// [`HybridAtomChoice`] tells the fitter which parameterization to keep for that
/// slot. The fitting of the two candidates lives in `sae_manifold.rs` (the
/// manifold-chart fitter); the SELECTION/scoring lives here.
pub fn select_hybrid_atom_parameterization(
    manifold: &LatentManifold,
    curved: Option<HybridAtomCandidate>,
    linear: HybridAtomCandidate,
) -> HybridAtomChoice {
    // A flat (Euclidean) chart has no curvature to price: its only honest
    // parameterization is the linear special case, so any curved candidate
    // offered for it is dropped before the evidence comparison. Curveable charts
    // (Circle / Sphere / Torus / curved products) present both candidates.
    let curved = if manifold.is_euclidean() {
        None
    } else {
        curved
    };
    let candidates: Vec<HybridAtomCandidate> = match curved {
        Some(c) => vec![linear, c],
        None => vec![linear],
    };
    // `candidates` is never empty (it always contains the linear candidate), so
    // the selector always returns a choice.
    select_hybrid_atom(&candidates).expect("hybrid atom slot always has the linear candidate")
}

#[cfg(test)]
mod ibp_prior_614_tests {
    // #614: `ibp_stick_breaking_prior` used to compute `π_k = (α/(α+1))^k` with
    // `π_0 = 1`, i.e. an UNSHRUNK first atom — the prior mean of no stick at all,
    // which broke α's role as an IBP concentration parameter. The consistent
    // truncated-IBP stick-breaking prior mean is `π_k = (α/(α+1))^{k+1}`, the
    // expectation of the product of (k+1) i.i.d. Beta(α,1) stick means, so EVERY
    // atom (including the first) carries one stick of shrinkage. This test pins
    // that contract so the regression cannot silently return.
    use super::*;

    fn ratio(alpha: f64) -> f64 {
        alpha / (alpha + 1.0)
    }

    #[test]
    fn first_atom_is_shrunk_not_unity() {
        // The #614 defect: π_0 must equal the single-stick mean α/(α+1), NOT 1.0.
        for &alpha in &[0.1_f64, 0.5, 1.0, 2.0, 5.0] {
            let prior = ordered_geometric_shrinkage_prior(8, alpha);
            let r = ratio(alpha);
            assert!(
                (prior[0] - r).abs() < 1e-12,
                "π_0 must be the single-stick mean α/(α+1)={r} (was the unshrunk 1.0 in #614); got {}",
                prior[0]
            );
            assert!(
                prior[0] < 1.0,
                "first atom must be shrunk (π_0<1) for alpha={alpha}; got {}",
                prior[0]
            );
        }
    }

    #[test]
    fn prior_is_consistent_geometric_product_mean() {
        // π_k = (α/(α+1))^{k+1} exactly, and every successive ratio equals α/(α+1).
        for &alpha in &[0.3_f64, 1.0, 4.0] {
            let k = 12;
            let prior = ordered_geometric_shrinkage_prior(k, alpha);
            let r = ratio(alpha);
            for j in 0..k {
                let expected = r.powi((j + 1) as i32);
                assert!(
                    (prior[j] - expected).abs() < 1e-12 * expected.max(1.0),
                    "alpha={alpha} π_{j}: expected {expected}, got {}",
                    prior[j]
                );
            }
            // Strictly decreasing (ordered shrinkage), no plateau at the head.
            for j in 1..k {
                assert!(
                    prior[j] < prior[j - 1],
                    "alpha={alpha}: prior must strictly decrease at index {j}"
                );
            }
        }
    }

    #[test]
    fn alpha_behaves_as_concentration() {
        // Larger α => heavier mass / slower decay: π_0 increases toward 1 and the
        // tail (e.g. π_4) carries more mass. This is the IBP-concentration role
        // the #614 fix restored.
        let lo = ordered_geometric_shrinkage_prior(8, 0.5);
        let hi = ordered_geometric_shrinkage_prior(8, 5.0);
        assert!(
            hi[0] > lo[0],
            "larger alpha must raise π_0 (concentration): {} vs {}",
            hi[0],
            lo[0]
        );
        assert!(
            hi[4] > lo[4],
            "larger alpha must put more mass in the tail: {} vs {}",
            hi[4],
            lo[4]
        );
    }
}

#[cfg(test)]
mod hybrid_split_tests {
    use super::*;
    use gam_solve::evidence::HybridAtomParam;

    #[test]
    fn flat_chart_drops_curved_candidate_and_keeps_linear() {
        // A Euclidean chart has no curvature: even if a curved candidate with a
        // lower NLE is offered, the helper drops it (a flat chart cannot honestly
        // present a curved parameterization).
        let linear = HybridAtomCandidate::linear(100.0, 2);
        let curved = HybridAtomCandidate::curved(1, 1.0, 5, Some(2.0));
        let choice =
            select_hybrid_atom_parameterization(&LatentManifold::Euclidean, Some(curved), linear);
        assert!(choice.param.is_linear());
    }

    #[test]
    fn curveable_chart_selects_curved_when_turning_pays() {
        // A Circle chart presents both candidates; a turning feature whose curved
        // fit beats the linear secant on evidence selects curved.
        let linear = HybridAtomCandidate::linear(100.0, 2);
        let curved = HybridAtomCandidate::curved(1, 70.0, 5, Some(2.0 * std::f64::consts::PI));
        let choice = select_hybrid_atom_parameterization(
            &LatentManifold::Circle {
                period: 2.0 * std::f64::consts::PI,
            },
            Some(curved),
            linear,
        );
        assert_eq!(choice.param, HybridAtomParam::Curved { latent_dim: 1 });
    }

    #[test]
    fn curveable_chart_falls_back_to_linear_when_no_curved_candidate() {
        let linear = HybridAtomCandidate::linear(33.0, 2);
        let choice = select_hybrid_atom_parameterization(
            &LatentManifold::Circle {
                period: 2.0 * std::f64::consts::PI,
            },
            None,
            linear,
        );
        assert!(choice.param.is_linear());
        assert_eq!(choice.num_parameters, 2);
    }
}

#[cfg(test)]
mod frozen_routing_1033_tests {
    //! #1033 — the FROZEN (amortized) routing mechanism: once installed, the
    //! per-row gate is a ρ-invariant function of the FROZEN predicted logits and
    //! is DECOUPLED from any subsequent update to the free `self.logits` (the
    //! inner-fit logit drift the outer ρ-search would otherwise re-incur every
    //! eval). These are deterministic mechanism invariants — no inner fit — so
    //! they pin the load-bearing freeze properties without the cluster.
    use super::*;

    fn ibp_assignment(n: usize, k: usize) -> SaeAssignment {
        let logits = Array2::from_shape_fn((n, k), |(i, kk)| {
            0.3 + 0.05 * (i as f64) - 0.1 * (kk as f64)
        });
        let coords: Vec<Array2<f64>> = (0..k)
            .map(|_| Array2::from_shape_fn((n, 1), |(i, _)| (i as f64) * 0.1))
            .collect();
        // learnable_alpha = false: alpha is ρ-independent, isolating the routing.
        SaeAssignment::from_blocks_with_mode(
            logits,
            coords,
            AssignmentMode::ibp_map(0.5, 1.0, false),
        )
        .unwrap()
    }

    #[test]
    fn frozen_routing_decouples_gates_from_logit_updates_1033() {
        let (n, k) = (6usize, 3usize);
        let mut a = ibp_assignment(n, k)
            .freeze_routing_from_current_logits()
            .unwrap();
        assert!(a.routing_is_frozen());
        // Gates BEFORE mutating the free logits.
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k]);
        let before: Vec<Array1<f64>> = (0..n)
            .map(|r| a.try_assignments_row_for_rho(r, &rho).unwrap())
            .collect();
        // Simulate an inner-fit logit update (what the ρ-search would otherwise do
        // every eval): perturb every free logit substantially.
        a.logits.mapv_inplace(|v| v + 5.0);
        let after: Vec<Array1<f64>> = (0..n)
            .map(|r| a.try_assignments_row_for_rho(r, &rho).unwrap())
            .collect();
        // FROZEN routing reads the snapshot, so the gates are UNCHANGED by the
        // free-logit perturbation — the routing is decoupled from inner-fit drift.
        for r in 0..n {
            for kk in 0..k {
                assert_eq!(
                    before[r][kk], after[r][kk],
                    "row {r} atom {kk}: frozen-routing gate must be UNCHANGED by a free-logit \
                     update (decoupled from inner-fit drift); {} vs {}",
                    before[r][kk], after[r][kk]
                );
            }
        }
    }

    #[test]
    fn frozen_routing_gates_are_rho_invariant_1033() {
        let (n, k) = (5usize, 2usize);
        let a = ibp_assignment(n, k)
            .freeze_routing_from_current_logits()
            .unwrap();
        // Two different ρ (different sparse + smooth strengths). With frozen routing
        // and learnable_alpha=false, the gate value must be identical at both ρ.
        let rho_a = SaeManifoldRho::new(
            (1e-3_f64).ln(),
            (1e-2_f64).ln(),
            vec![Array1::<f64>::zeros(1); k],
        );
        let rho_b = SaeManifoldRho::new(
            (1e3_f64).ln(),
            (1e1_f64).ln(),
            vec![Array1::<f64>::zeros(1); k],
        );
        for r in 0..n {
            let ga = a.try_assignments_row_for_rho(r, &rho_a).unwrap();
            let gb = a.try_assignments_row_for_rho(r, &rho_b).unwrap();
            for kk in 0..k {
                assert_eq!(
                    ga[kk], gb[kk],
                    "row {r} atom {kk}: frozen-routing gate must be ρ-INVARIANT (the n-independence \
                     lever); {} at ρ_a vs {} at ρ_b",
                    ga[kk], gb[kk]
                );
            }
        }
    }

    #[test]
    fn frozen_routing_fixes_all_logits_and_thaw_restores_free_path_1033() {
        let (n, k) = (4usize, 3usize);
        let mut a = ibp_assignment(n, k)
            .freeze_routing_from_current_logits()
            .unwrap();
        // Under frozen routing EVERY logit is fixed (not a free Newton coord).
        let mask = a.fixed_logit_mask();
        assert_eq!(mask.len(), k);
        assert!(
            mask.iter().all(|&f| f),
            "frozen routing must fix ALL logits"
        );
        for kk in 0..k {
            assert!(
                a.logit_is_fixed(kk),
                "atom {kk} logit must be fixed under frozen routing"
            );
        }
        // Thawing restores the free-logit path (no fixed logits, no ungated).
        a.thaw_routing();
        assert!(!a.routing_is_frozen());
        assert!(
            a.fixed_logit_mask().iter().all(|&f| !f),
            "thaw must restore the free-logit path"
        );
    }

    #[test]
    fn frozen_routing_rejects_softmax_1033() {
        let (n, k) = (4usize, 3usize);
        let logits = Array2::from_shape_fn((n, k), |(i, kk)| 0.1 * (i as f64) - 0.05 * (kk as f64));
        let coords: Vec<Array2<f64>> = (0..k)
            .map(|_| Array2::from_shape_fn((n, 1), |(i, _)| (i as f64) * 0.1))
            .collect();
        let a = SaeAssignment::from_blocks_with_mode(logits, coords, AssignmentMode::softmax(1.0))
            .unwrap();
        // Softmax + frozen routing is rejected (the coupled-simplex entropy
        // majorizer would be inconsistent with a frozen, non-optimized routing).
        assert!(
            a.freeze_routing_from_current_logits().is_err(),
            "frozen routing under Softmax must be rejected (simplex entropy-majorizer coupling)"
        );
    }
}

#[cfg(test)]
mod support_measure_tests {
    use super::*;

    #[test]
    fn support_measure_matches_hard_and_diffuse_semantics() {
        let hard_weights = Array1::from_vec(vec![1.0, 1.0, 0.0, 1.0, 0.0]);
        let hard = SupportMeasure::from_weights(0, hard_weights).unwrap();
        assert_eq!(hard.mass(), 3.0);
        assert_eq!(hard.fisher_n(), 3.0);
        assert_eq!(hard.ess(), 3.0);
        assert_eq!(hard.positive_rows(), vec![0usize, 1, 3]);
        let from_owners = SupportMeasure::from_argmax_owners(&[0, 0, 1, 0, 1], 0, 2).unwrap();
        assert_eq!(from_owners.mass(), hard.mass());
        assert_eq!(from_owners.fisher_n(), hard.fisher_n());
        assert_eq!(from_owners.ess(), hard.ess());
        assert_eq!(from_owners.positive_rows(), hard.positive_rows());

        let diffuse_weights = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        let diffuse = SupportMeasure::from_weights(1, diffuse_weights).unwrap();
        assert_eq!(diffuse.mass(), 2.0);
        assert_eq!(diffuse.fisher_n(), 1.0);
        assert_eq!(diffuse.ess(), 4.0);
    }

    #[test]
    fn support_measure_reads_assignment_column() {
        let assignments =
            Array2::from_shape_vec((3, 2), vec![0.8, 0.2, 0.4, 0.6, 0.0, 1.0]).unwrap();
        let support = SupportMeasure::from_assignment_matrix(assignments.view(), 1).unwrap();
        assert!((support.mass() - 1.8).abs() < 1e-12);
        assert!((support.fisher_n() - 1.4).abs() < 1e-12);
        assert!((support.ess() - (1.8_f64 * 1.8 / 1.4)).abs() < 1e-12);
        assert_eq!(support.positive_rows(), vec![0usize, 1, 2]);
    }
}

#[cfg(test)]
mod fill_into_buffer_1557_tests {
    //! #1557 — the fill-into-caller-buffer variant
    //! [`SaeAssignment::try_assignments_row_for_rho_into`] must produce
    //! BIT-IDENTICAL output to the allocating
    //! [`SaeAssignment::try_assignments_row_for_rho`] across every assignment
    //! mode (Softmax, IBPMap, JumpReLU), the #1026 ungated case, and the K==1
    //! edge. Exact `==` on f64 — not an approximate tolerance — because the
    //! `_into` path is a pure allocation-elision refactor and any numeric drift
    //! is a regression.
    use super::*;

    fn build(n: usize, k: usize, mode: AssignmentMode) -> SaeAssignment {
        // Deterministic, asymmetric logits/coords so every atom takes a distinct
        // value (no accidental ties masking an index bug).
        let logits = Array2::from_shape_fn((n, k), |(i, kk)| {
            0.37 + 0.11 * (i as f64) - 0.23 * (kk as f64)
        });
        let coords: Vec<Array2<f64>> = (0..k)
            .map(|_| Array2::from_shape_fn((n, 1), |(i, _)| 0.1 + 0.05 * (i as f64)))
            .collect();
        SaeAssignment::from_blocks_with_mode(logits, coords, mode).unwrap()
    }

    fn rho(k: usize) -> SaeManifoldRho {
        SaeManifoldRho::new(
            (1e-2_f64).ln(),
            (1e-1_f64).ln(),
            vec![Array1::<f64>::zeros(1); k],
        )
    }

    fn assert_into_matches_alloc(a: &SaeAssignment) {
        let n = a.n_obs();
        let k = a.k_atoms();
        let rho = rho(k);
        let mut scratch = vec![f64::NAN; k];
        for row in 0..n {
            let allocated = a.try_assignments_row_for_rho(row, &rho).unwrap();
            // Pre-fill with NaN so a partial write (e.g. a JumpReLU below-threshold
            // entry left untouched) is caught as a mismatch, not silently passed.
            for s in scratch.iter_mut() {
                *s = f64::NAN;
            }
            a.try_assignments_row_for_rho_into(row, &rho, &mut scratch)
                .unwrap();
            assert_eq!(allocated.len(), k);
            for kk in 0..k {
                assert_eq!(
                    allocated[kk], scratch[kk],
                    "row {row} atom {kk}: _into must be BIT-IDENTICAL to the allocating \
                     try_assignments_row_for_rho; got {} vs {}",
                    allocated[kk], scratch[kk]
                );
            }
        }
    }

    #[test]
    fn softmax_into_is_bit_identical() {
        assert_into_matches_alloc(&build(7, 4, AssignmentMode::softmax(0.8)));
    }

    #[test]
    fn ibp_map_into_is_bit_identical() {
        // Both learnable and fixed alpha exercise the resolved-alpha branch.
        assert_into_matches_alloc(&build(7, 5, AssignmentMode::ibp_map(0.6, 1.3, false)));
        assert_into_matches_alloc(&build(7, 5, AssignmentMode::ibp_map(0.6, 1.3, true)));
    }

    #[test]
    fn jumprelu_into_is_bit_identical() {
        // Threshold chosen so SOME atoms fall below it (the untouched-entry path)
        // and some clear it (the sigmoid path) — both branches are exercised.
        assert_into_matches_alloc(&build(7, 5, AssignmentMode::jumprelu(0.9, 0.2)));
    }

    #[test]
    fn ungated_into_is_bit_identical() {
        // #1026 ungated overwrite under a gate-style mode (IBP/JumpReLU allow it).
        let a = build(6, 4, AssignmentMode::ibp_map(0.6, 1.1, false))
            .with_ungated(vec![false, true, false, true])
            .unwrap();
        assert_into_matches_alloc(&a);
        let j = build(6, 4, AssignmentMode::jumprelu(0.9, 0.15))
            .with_ungated(vec![true, false, true, false])
            .unwrap();
        assert_into_matches_alloc(&j);
    }

    #[test]
    fn k_equals_one_into_is_bit_identical() {
        // Softmax K==1 hits the fixed-unit early return; IBP/JumpReLU K==1 keep a
        // free per-atom gate and fall through to the real row functions.
        assert_into_matches_alloc(&build(5, 1, AssignmentMode::softmax(1.0)));
        assert_into_matches_alloc(&build(5, 1, AssignmentMode::ibp_map(0.7, 1.0, false)));
        assert_into_matches_alloc(&build(5, 1, AssignmentMode::jumprelu(0.8, 0.1)));
    }
}
