//! Assignment gates and sparsity-prior helpers for the SAE manifold term.
//! Mechanically split from `sae_manifold.rs`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::SaeManifoldRho;
use gam_solve::evidence::{HybridAtomCandidate, HybridAtomChoice, select_hybrid_atom};
use gam_terms::analytic_penalties::{
    AnalyticPenalty, OrderedBetaBernoulliPenalty, OrderedBetaBernoulliHessianDiagThirdChannels,
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

/// Assignment prior/relaxation used by [`SaeAssignment`].
#[derive(Debug, Clone, Copy)]
pub enum AssignmentMode {
    /// Row-wise simplex assignment with entropy sparsity.
    Softmax { temperature: f64, sparsity: f64 },
    /// Deterministic concrete posterior means for a truncated ordered Beta--Bernoulli active set:
    /// `a_k = σ(logit_k/temperature)`. These are independent Bernoulli gates,
    /// not mixture/simplex responsibilities. The ordered stick-breaking mean
    /// `π_k = (α/(α+1))^{k+1}` is scored once by the ordered Beta--Bernoulli prior; it is not
    /// multiplied into the final reconstructed function.
    OrderedBetaBernoulli {
        temperature: f64,
        alpha: f64,
        learnable_alpha: bool,
    },
    /// Smooth threshold-centered logistic gate
    /// `a_k = σ((logit_k − threshold) / temperature)`. Magnitude lives in the
    /// decoder curve `g_k(t) = φ(t)ᵀB_k`; this gate supplies a bounded posterior
    /// activation in `(0, 1)`. Its derivative is exact on both sides of the
    /// threshold, so fitted values, data-fit Jacobians, priors, and Hessians are
    /// derivatives of one smooth objective.
    ThresholdGate { temperature: f64, threshold: f64 },
    /// Hard top-`k` support gate: the `k` atoms with the LARGEST routing logits
    /// in a row carry gate 1, every other atom carries gate 0 (ties broken
    /// toward the lower atom index, so the support is deterministic).
    ///
    /// Sparsity is BY CONSTRUCTION, not by penalty: there is no sparsity term
    /// in the objective, no gate logit in the inner system
    /// (`assignment_coord_dim() == 0` — at K = 32,000 this deletes 32k
    /// coordinates from the inner Newton), and no sparsity coordinate in the
    /// outer ρ search. At fixed support size `|S| = k` this is the
    /// constrained-MAP member of the same spike-slab generative family the ordered Beta--Bernoulli
    /// gate lives in, with the ℓ2,0 constraint standing in for the
    /// stick-breaking prior. The gate is per-row independent (couples rows
    /// through NOTHING), so fits stream chunk-invariantly at any K, and it is
    /// exchangeable across atom index — the ordered-ordered Beta--Bernoulli concentration
    /// pathology (#1784) cannot arise.
    TopK { k: usize },
}

/// Caller intent for assignment-mode admission. `Default` is the production
/// route: it never selects ordered Beta--Bernoulli-MAP implicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentModeRequest {
    Default,
    Softmax,
    ThresholdGate,
    OrderedBetaBernoulli,
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
    pub fn ordered_beta_bernoulli(temperature: f64, alpha: f64, learnable_alpha: bool) -> Self {
        Self::OrderedBetaBernoulli {
            temperature,
            alpha,
            learnable_alpha,
        }
    }

    /// #1777 — construct the hard-sigmoid [`Self::ThresholdGate`].
    #[must_use]
    pub fn threshold_gate(temperature: f64, threshold: f64) -> Self {
        Self::ThresholdGate {
            temperature,
            threshold,
        }
    }

    /// Construct the hard top-`k` support gate ([`Self::TopK`]): sparsity by
    /// construction, zero gate coordinates in the inner system, per-row
    /// independent. `k` is clamped to at least 1 by the fit-time validator.
    #[must_use]
    pub fn top_k_support(k: usize) -> Self {
        Self::TopK { k }
    }

    pub fn temperature(&self) -> f64 {
        match *self {
            AssignmentMode::Softmax { temperature, .. }
            | AssignmentMode::OrderedBetaBernoulli { temperature, .. }
            | AssignmentMode::ThresholdGate { temperature, .. } => temperature,
            // The hard support gate has no relaxation, hence no temperature; the
            // unit value keeps generic temperature-logging paths well-defined.
            AssignmentMode::TopK { .. } => 1.0,
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
            | AssignmentMode::OrderedBetaBernoulli { temperature, .. }
            | AssignmentMode::ThresholdGate { temperature, .. } => {
                *temperature = new_temperature;
            }
            // No relaxation to anneal: the hard support is temperature-free, so
            // annealing schedules pass through as a no-op.
            AssignmentMode::TopK { .. } => {}
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
            AssignmentMode::OrderedBetaBernoulli { alpha, .. } => {
                if !(alpha.is_finite() && alpha > 0.0) {
                    return Err(format!(
                        "AssignmentMode::OrderedBetaBernoulli: alpha must be finite and positive; got {alpha}"
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
            AssignmentMode::TopK { k } => {
                if k == 0 {
                    return Err(
                        "AssignmentMode::TopK: support size k must be at least 1".to_string()
                    );
                }
            }
        }
        Ok(())
    }

    /// Resolve the effective truncated-ordered Beta--Bernoulli concentration `α` for this mode.
    ///
    /// `per_fit_override` is the #1777 PER-FIT override (from
    /// [`SaeAssignment::ordered_beta_bernoulli_alpha_override`]) and is the source of truth when set.
    /// Otherwise the mode's canonical fixed `α` or learnable schedule is used.
    pub(crate) fn resolved_ordered_beta_bernoulli_alpha(
        &self,
        rho: &SaeManifoldRho,
        per_fit_override: Option<f64>,
    ) -> Option<f64> {
        match *self {
            AssignmentMode::OrderedBetaBernoulli {
                alpha,
                learnable_alpha,
                ..
            } => Some(if let Some(over) = per_fit_override {
                // #1777 — the per-fit override flattens the ordered geometric
                // prior π_k = (α/(α+1))^{k+1}
                // so all K atoms can contribute to the reconstruction (the
                // production α=1 gives a (0.5)^{k+1} schedule that structurally
                // caps atoms 4..K → effective-K≈3). Forces the fixed value,
                // bypassing the learnable schedule.
                over
            } else if learnable_alpha {
                resolve_learnable_weight(alpha, rho.log_lambda_sparse)
            } else {
                alpha
            }),
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
/// a top-k cap at large K. ordered Beta--Bernoulli-MAP is a research-mode opt-in and is refused once
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
        AssignmentModeRequest::Default | AssignmentModeRequest::Softmax => {
            AssignmentModeAdmission {
                mode: AssignmentMode::softmax(temperature),
                top_k: large_k_top,
            }
        }
        AssignmentModeRequest::ThresholdGate => AssignmentModeAdmission {
            mode: AssignmentMode::threshold_gate(temperature, threshold),
            top_k: None,
        },
        AssignmentModeRequest::OrderedBetaBernoulli => {
            // #F2 — re-admit ordered Beta--Bernoulli-MAP at large K, with the same rows-per-atom
            // `top_k` used as the ACTIVE-SET COMPUTE CAP (the softmax lane's
            // large-K cap), instead of refusing the request. The occupancy-driven
            // empirical-Bayes α M-step (#F1) now un-caps the effective atom count
            // that the fixed geometric schedule used to pin at ~3, so ordered Beta--Bernoulli-MAP is a
            // usable large-K lane once its per-row work is bounded by `top_k`.
            // Small fits keep `top_k = None` (dense ordered Beta--Bernoulli-MAP), unchanged.
            AssignmentModeAdmission {
                mode: AssignmentMode::ordered_beta_bernoulli(temperature, alpha, learnable_alpha),
                top_k: large_k_top,
            }
        }
    };
    admission.mode.validate()?;
    Ok(admission)
}

/// Per-row latent assignment state — the DENSE-CERTIFICATION / debug-and-research
/// lane state only (#985 / E1), NOT the production route.
///
/// This is the dense `N×K` routing representation. The production SAE path is the
/// sparse-code lane ([`crate::sparse_dict`]), whose per-row state is fixed-width
/// `(indices, codes)` and never materializes an `N×K` assignment; large-K public
/// fits are routed there by the front door ([`crate::front_door::admit_sae_fit`] /
/// [`crate::front_door::admit_dense_certification`], #14). The dense manifold
/// engine that owns this type is reached only for the small-`K` certification lane
/// (`K ≤ P`) and for overcomplete research fits at small `N`. A source-guard test
/// (`sparse_lane_constructs_no_dense_assignment`) locks the invariant that the
/// sparse lane constructs zero `SaeAssignment`s; `#[doc(hidden)]` keeps this dense
/// state off the public API surface to match the demotion.
///
/// The stored assignment parameter is `logits`; non-negative assignments are
/// derived by row-wise softmax, independent ordered Beta--Bernoulli-MAP sigmoid active indicators,
/// or JumpReLU gates. Softmax logits are canonicalized to the reference chart
/// `logits[K - 1] = 0`, so the row-local Newton coordinates contain only the
/// first `K - 1` logits (`0` coordinates for `K = 1`). Gate-style modes keep
/// all `K` logits as identifiable scalar parameters. `coords[k]` holds
/// `t_{.,k}` for atom `k`.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct SaeAssignment {
    pub logits: Array2<f64>,
    pub coords: Vec<LatentCoordValues>,
    pub mode: AssignmentMode,
    /// #1026 — per-atom UNGATED flag (length `K`, default all-`false`). An
    /// ungated atom is the dense linear/background tier: its per-row gate is
    /// fixed at `a_k ≡ 1` (it contributes `γ_k(t_k)` to EVERY row, unweighted),
    /// it is excluded from the other atoms' gate (for the column-separable
    /// ordered Beta--Bernoulli / JumpReLU modes the remaining atoms are computed independently, so
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
    /// #1777 PER-FIT ordered Beta--Bernoulli-α override. `Some(α)` forces a fixed value and bypasses
    /// the learnable schedule for this assignment/fit. `None` uses the
    /// [`AssignmentMode`]'s canonical fixed `α` or learnable schedule. Read via
    /// [`Self::resolved_ordered_beta_bernoulli_alpha`]; set from the FFI through the term's
    /// `set_fit_config`.
    pub ordered_beta_bernoulli_alpha_override: Option<f64>,
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
            ordered_beta_bernoulli_alpha_override: None,
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
                     contract supports ordered Beta--Bernoulli-MAP and JumpReLU, whose per-atom gates have no \
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
        // TopK: NO logit is ever a free Newton parameter — the support is a
        // deterministic function of the routing logits (assignment_coord_dim
        // is 0), so every gate rides as a constant, exactly like frozen routing.
        matches!(self.mode, AssignmentMode::TopK { .. })
            || self.routing_is_frozen()
            || self.ungated.get(k).copied().unwrap_or(false)
    }

    /// Per-atom mask (length `K`) of [`Self::logit_is_fixed`] — the logit slots
    /// that are NOT free Newton parameters (ungated #1026 and/or frozen-routing
    /// #1033). Precompute once per assembly and pass to the logit-JVP fillers so
    /// the data-fit Jacobian zeroes those rows. Under frozen routing every entry
    /// is `true`; with only ungated atoms it equals `ungated`; otherwise all
    /// `false` (the historical free-logit path).
    pub(crate) fn fixed_logit_mask(&self) -> Vec<bool> {
        if matches!(self.mode, AssignmentMode::TopK { .. }) || self.routing_is_frozen() {
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
                 (coupled-simplex entropy-majorizer); use ordered Beta--Bernoulli-MAP or JumpReLU"
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
                 rejected (coupled-simplex entropy-majorizer); use ordered Beta--Bernoulli-MAP or JumpReLU"
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
    /// Ungating is defined for the COLUMN-SEPARABLE gate modes (ordered Beta--Bernoulli-MAP and
    /// JumpReLU): each atom's gate is an independent per-atom function of its own
    /// logit, so pinning one atom to `a_k ≡ 1` leaves every other atom's gate
    /// exactly as computed. Softmax is a coupled simplex (`Σ_k a_k = 1` over all
    /// `K`), so a unit gate for one atom is only well defined relative to a
    /// gated-subset renormalization that must also be reflected in the logit-JVP
    /// and the entropy majorizer; this constructor's contract is restricted to
    /// the separable modes, and an ungated atom under Softmax is REJECTED here so
    /// the inner solve never runs on a value/gradient-mismatched gate. Callers
    /// wanting a dense background tier under Softmax route it as an ordered Beta--Bernoulli-MAP or
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
                 contract does not perform; route a dense background tier as ordered Beta--Bernoulli-MAP or JumpReLU"
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
            AssignmentMode::OrderedBetaBernoulli { .. } | AssignmentMode::ThresholdGate { .. } => self.k_atoms(),
            // Sparsity by construction: the support is a deterministic function
            // of the routing logits, so there are NO free gate coordinates in
            // the inner system.
            AssignmentMode::TopK { .. } => 0,
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
        self.try_assignments_row_inner(row)
    }

    /// #1777 — the effective truncated-ordered Beta--Bernoulli `α` for this assignment at `rho`,
    /// honoring the PER-FIT [`Self::ordered_beta_bernoulli_alpha_override`] before the mode's
    /// canonical value or learnable schedule. The single seam every
    /// gate/jet/prior site reads so the per-fit override is applied consistently.
    /// `None` for non-ordered Beta--Bernoulli modes.
    pub(crate) fn resolved_ordered_beta_bernoulli_alpha(&self, rho: &SaeManifoldRho) -> Option<f64> {
        self.mode.resolved_ordered_beta_bernoulli_alpha(rho, self.ordered_beta_bernoulli_alpha_override)
    }

    /// Whether the truncated-ordered Beta--Bernoulli concentration α is a FREE outer parameter that
    /// varies with ρ (`rho.log_lambda_sparse`). α is learnable ONLY when the mode
    /// requests it AND no per-fit override pins it: an override forces the fixed
    /// value and bypasses the learnable
    /// schedule (see [`AssignmentMode::resolved_ordered_beta_bernoulli_alpha`]), so α's ρ-derivatives
    /// are then identically zero and every prior / log-det / IFT term must treat α
    /// as a constant to stay consistent with the forward gate. `false` for non-ordered Beta--Bernoulli
    /// modes. (#Bug6)
    pub(crate) fn effective_alpha_is_learnable(&self) -> bool {
        match self.mode {
            AssignmentMode::OrderedBetaBernoulli {
                learnable_alpha, ..
            } => learnable_alpha && self.ordered_beta_bernoulli_alpha_override.is_none(),
            _ => false,
        }
    }

    /// #1777 — install (or clear, with `None`) the PER-FIT ordered Beta--Bernoulli-α override on this
    /// assignment. Source of truth used by [`Self::resolved_ordered_beta_bernoulli_alpha`]; the FFI
    /// reaches it through the term's `set_fit_config`.
    pub fn set_ordered_beta_bernoulli_alpha_override(&mut self, alpha: Option<f64>) {
        self.ordered_beta_bernoulli_alpha_override = alpha;
    }

    /// #F1 — the Fellner–Schall-analog empirical-Bayes M-step for the
    /// `log_lambda_sparse` slot: the additive step `Δθ = ln α_EB* − ln α_current`
    /// that moves the ordered-ordered Beta--Bernoulli concentration to the occupancy-driven marginal
    /// stationary point (`log_lambda_sparse += Δθ` IS the multiplicative α update,
    /// since the resolved α is `α_base · exp(log_lambda_sparse)`).
    ///
    /// Returns `Some(Δθ)` ONLY when the effective α is a FREE learnable parameter
    /// (ordered Beta--Bernoulli-MAP, `learnable_alpha`, no per-fit override) —
    /// exactly when [`Self::effective_alpha_is_learnable`] holds, so the α
    /// ρ-derivatives are non-zero and the marginal M-step is the coherent update.
    /// `None` for every other sparsity prior (softmax entropy, gated L1) or a
    /// pinned α, whose non-quadratic prior has no closed-form fixed point and
    /// keeps the historical zero step. This is the ONE place the large-K /
    /// streaming `λ_sparse`-frozen bug is fixed: occupancy `M_k = Σ_i a_{ik}` is
    /// accumulated per-row from the FITTED gates at `rho` (O(N·K) time, O(K)
    /// memory — no dense `N×K` materialisation), so it is valid in the streaming
    /// regime where the value-lane gradient is identically zero.
    ///
    /// The returned step is trust-region bounded (`|Δθ| ≤ 2`) so a single outer
    /// iterate cannot overshoot the α axis; the fixed point is reached over
    /// successive Fellner–Schall iterates, each accepted through the REML cost
    /// lane like every other coordinate's step.
    pub(crate) fn ordered_beta_bernoulli_eb_log_alpha_step(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Option<f64>, String> {
        if !self.effective_alpha_is_learnable() {
            return Ok(None);
        }
        let resolved = self.resolved_ordered_beta_bernoulli_alpha(rho);
        let Some(alpha_current) = resolved else {
            return Ok(None);
        };
        if !(alpha_current.is_finite() && alpha_current > 0.0) {
            return Ok(None);
        }
        let k = self.k_atoms();
        let n = self.n_obs();
        if k == 0 || n == 0 {
            return Ok(None);
        }
        // Soft occupancy `M_k = Σ_i a_{ik}` from the fitted gates at `rho`,
        // accumulated row-by-row into a K-buffer (streaming-safe).
        let mut occupancy = vec![0.0_f64; k];
        let mut buf = vec![0.0_f64; k];
        for row in 0..n {
            self.try_assignments_row_into(row, &mut buf)?;
            for (acc, &g) in occupancy.iter_mut().zip(buf.iter()) {
                *acc += g;
            }
        }
        let alpha_star = ordered_beta_bernoulli_eb_geometric_alpha_fixed_point(&occupancy, n as f64, alpha_current);
        if !(alpha_star.is_finite() && alpha_star > 0.0) {
            return Ok(None);
        }
        const LOG_ALPHA_STEP_CAP: f64 = 2.0;
        let step =
            (alpha_star.ln() - alpha_current.ln()).clamp(-LOG_ALPHA_STEP_CAP, LOG_ALPHA_STEP_CAP);
        Ok(Some(step))
    }

    /// Post-#1033 the row gates are ρ-INVARIANT (frozen/predicted or free
    /// routing logits never read ρ), so the assignment APIs take no ρ — the
    /// signatures state the invariance instead of threading a dead parameter.
    /// (A previous "wiring contract" rejected ρ whose per-atom width differed
    /// from `k_atoms()`, but K legitimately moves mid-fit — births, deaths,
    /// compaction, topology-race candidates — while ρ updates lag, so that
    /// contract vetoed valid states and broke seed validation fleet-wide;
    /// bisected to 6297a7e9f.)
    fn try_assignments_row_inner(&self, row: usize) -> Result<Array1<f64>, String> {
        // #1033 — read the ACTIVE routing logits: the ρ-invariant frozen/predicted
        // logits when routing is frozen, else the free `self.logits`. This single
        // source makes the gate value ρ-invariant under frozen routing (the
        // amortized-routing lever) and bit-identical to the historical path when
        // not frozen.
        let routing = self.routing_logits_row(row);
        validate_finite_logits(routing, row)?;
        // Only Softmax collapses to a fixed assignment at K==1: its
        // assignment_coord_dim is K-1 = 0, so there is no free logit. OrderedBetaBernoulli and
        // JumpReLU keep a free per-atom gate logit even at K==1
        // (assignment_coord_dim = K = 1), so they must fall through to their real
        // row functions or the logit would move the prior but not the gate.
        if self.k_atoms() == 1 && matches!(self.mode, AssignmentMode::Softmax { .. }) {
            return Ok(Array1::from_vec(vec![1.0]));
        }
        let mut row_gates = match self.mode {
            AssignmentMode::Softmax { temperature, .. } => softmax_row(routing, temperature),
            AssignmentMode::OrderedBetaBernoulli { temperature, .. } => ordered_beta_bernoulli_row(routing, temperature),
            AssignmentMode::ThresholdGate {
                temperature,
                threshold,
            } => threshold_gate_row(routing, temperature, threshold),
            AssignmentMode::TopK { k } => topk_row(routing, k),
        };
        // #1026 — ungated (background-tier) atoms have a fixed unit gate. For the
        // column-separable ordered Beta--Bernoulli / JumpReLU modes the other atoms' gates are
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

    /// #1557 — fill-into-caller-buffer twin of [`Self::try_assignments_row`].
    ///
    /// `out` must have length `k_atoms()`; it is fully overwritten with the same
    /// values the allocating variant would return. Every branch (early-return
    /// K==1 Softmax, the per-mode row math, the #1026 ungated overwrite) mirrors
    /// the allocating path exactly so the two are bit-identical.
    pub(crate) fn try_assignments_row_into(
        &self,
        row: usize,
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
            AssignmentMode::OrderedBetaBernoulli { temperature, .. } => {
                ordered_beta_bernoulli_row_into(routing, temperature, out)
            }
            AssignmentMode::ThresholdGate {
                temperature,
                threshold,
            } => threshold_gate_row_into(routing, temperature, threshold, out),
            AssignmentMode::TopK { k } => topk_row_into(routing, k, out),
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

    pub(crate) fn persist_resolved_ordered_beta_bernoulli_alpha(&mut self, rho: &SaeManifoldRho) -> bool {
        let AssignmentMode::OrderedBetaBernoulli {
            temperature,
            alpha,
            learnable_alpha: true,
        } = self.mode
        else {
            return false;
        };
        let resolved_alpha = resolve_learnable_weight(alpha, rho.log_lambda_sparse);
        self.mode = AssignmentMode::OrderedBetaBernoulli {
            temperature,
            alpha: resolved_alpha,
            learnable_alpha: false,
        };
        true
    }

    pub(crate) fn try_assignments(&self) -> Result<Array2<f64>, String> {
        let n = self.n_obs();
        let k = self.k_atoms();
        let mut out = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let a = self.try_assignments_row(row)?;
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
        AssignmentMode::OrderedBetaBernoulli { temperature, .. } => {
            ordered_beta_bernoulli_row(Array1::<f64>::zeros(k_atoms).view(), temperature)
        }
        AssignmentMode::ThresholdGate { .. } => Array1::from_elem(k_atoms, 0.5),
        // At all-equal (zero) logits the deterministic tie-break admits the
        // FIRST k atoms — the neutral support under index-stable ordering.
        AssignmentMode::TopK { k } => topk_row(Array1::<f64>::zeros(k_atoms).view(), k),
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
/// special-cased always-on base atom, so `α` behaves as a genuine ordered Beta--Bernoulli
/// concentration — larger `α` ⇒ heavier mass / slower decay, `α → 0` ⇒ all mass
/// collapses onto nothing, matching the stick-breaking limit. This is the
/// deterministic mean-field form of the ordered Beta--Bernoulli prior (the closed form the
/// analytic Newton / Hessian / Woodbury machinery differentiates); no sticks are
/// *sampled* here, the per-atom weight is the exact expectation of the
/// stick-breaking product. (#614: previously `π_0 = 1` left the first atom
/// unshrunk, which is the prior mean of NO stick at all and broke α's role as a
/// concentration; the consistent product mean restores genuine ordered Beta--Bernoulli semantics.)
/// Ordered prior-mean *schedule* for the truncated-ordered Beta--Bernoulli assignment prior. Both
/// forms produce a strictly positive, ordered (decreasing) prior mean
/// `μ_k ∈ (0, 1]` consumed by the Beta--Bernoulli penalty.
///
/// * [`Self::Geometric`] — the historical stick-breaking mean
///   `μ_k = (α/(α+1))^{k+1}`, which decays GEOMETRICALLY in the atom index and
///   at `α = 1` assigns overwhelming prior shrinkage to late atoms.
/// * [`Self::PowerLaw`] — a heavier (near-Zipfian) polynomial tail
///   `μ_k = c/(k + k0)^s`, whose sub-exponential decay keeps late atoms
///   un-masked at large `K` where the geometric schedule has already collapsed
///   to numerical zero (#F2). This is the correct tail for near-Zipf feature
///   frequencies. Both schedules share the SAME occupancy-driven empirical-Bayes
///   fixed point (the Beta–Bernoulli marginal is schedule-agnostic; only the
///   `a_k(θ)` map and its derivatives differ — see [`ordered_beta_bernoulli_eb_marginal_score`]).
#[derive(Debug, Clone, Copy)]
pub enum OrderedPriorSchedule {
    /// Stick-breaking mean `μ_k = (α/(α+1))^{k+1}`, `α > 0`.
    Geometric { alpha: f64 },
    /// Power-law (Zipf-like) mean `μ_k = c/(k + k0)^s`, `c > 0`, `s > 0`,
    /// `k0 > 0`.
    PowerLaw { c: f64, s: f64, k0: f64 },
}

/// Ordered per-atom prior means `μ_k` for the requested [`OrderedPriorSchedule`].
///
/// Both branches accumulate in LOG space and floor the exponentiated weight at
/// the smallest positive normal so the soft shrinkage prior never becomes a HARD
/// mask: exact zero would make the Beta prior undefined. The geometric branch
/// is the historical `π_k = ratio^(k+1)`
/// computation, unchanged; the power-law branch additionally clamps `μ_k ≤ 1`
/// (a raw `c/(k0)^s` can exceed 1 for the first atoms).
pub fn ordered_prior_means(k_atoms: usize, schedule: OrderedPriorSchedule) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(k_atoms);
    match schedule {
        OrderedPriorSchedule::Geometric { alpha } => {
            let log_ratio = (alpha / (alpha + 1.0)).ln();
            for k in 0..k_atoms {
                // π_k = (α/(α+1))^{k+1}: the product of (k+1) i.i.d. Beta(α,1)
                // stick means, so atom 0 is also shrunk by one stick.
                let log_pi = ((k + 1) as f64) * log_ratio;
                out[k] = log_pi.exp().max(f64::MIN_POSITIVE);
            }
        }
        OrderedPriorSchedule::PowerLaw { c, s, k0 } => {
            // μ_k = c/(k + k0)^s. Clamp into (0, 1]: the smallest positive normal
            // floor keeps every atom's gradient path alive (as in the geometric
            // branch), and the unit ceiling keeps `μ_k` a valid prior mean.
            for k in 0..k_atoms {
                let log_pi = c.ln() - s * ((k as f64) + k0).ln();
                out[k] = log_pi.exp().clamp(f64::MIN_POSITIVE, 1.0);
            }
        }
    }
    out
}

/// #1784 — K-aware default ordered Beta--Bernoulli concentration.
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
/// For a K-atom dictionary to actually USE all K atoms the ordered Beta--Bernoulli concentration must
/// scale with K. Choosing `α` so the LAST atom retains prior mass
/// `π_{K-1} = (α/(α+1))^K ≈ e^{-1}` spans the whole dictionary while keeping the
/// prior monotone (still an honest ordered stick-breaking prior — no atom is
/// structurally masked). Solving `(α/(α+1))^K = e^{-1}` gives
/// `α = 1/(exp(1/K) − 1) ≈ K − 1/2`. Floored at `1.0` so `K = 1` keeps the
/// historical `α = 1`.
pub fn default_ordered_beta_bernoulli_concentration_for_k_atoms(k_atoms: usize) -> f64 {
    let k = k_atoms.max(1) as f64;
    // π_{K-1} = (α/(α+1))^K = e^{-1}  ⇒  α = 1/(e^{1/K} − 1).
    let alpha = 1.0 / ((1.0 / k).exp() - 1.0);
    alpha.max(1.0)
}

/// Trigamma `ψ'(x)` (Abramowitz & Stegun recurrence + asymptotic series),
/// mirroring the `gam-solve` PIRLS implementation so the empirical-Bayes M-step
/// curvature is computed to the same accuracy the rest of the workspace uses.
#[inline]
fn trigamma(mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut acc = 0.0;
    while x < 8.0 {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    acc + inv + 0.5 * inv2 + inv2 * inv / 6.0 - inv2 * inv2 * inv / 30.0
        + inv2 * inv2 * inv2 * inv / 42.0
        - inv2 * inv2 * inv2 * inv2 * inv / 30.0
}

// ---------------------------------------------------------------------------
// #F1/#F2 — occupancy-driven empirical-Bayes fixed point for the ordered ordered Beta--Bernoulli
// assignment prior (the Fellner–Schall-analog M-step for `log_lambda_sparse`).
//
// COHERENT MODEL. Each atom's ordered prior mass is `π_k ~ Beta(a_k, 1)` with
// mean `μ_k = a_k/(a_k + 1)`, i.e. `a_k = μ_k/(1 − μ_k)` (the pseudo-count that
// the shrinkage schedule `μ_k(θ)` — geometric α or power-law (c,s,k0) — pins).
// The per-atom activation indicators are `z_{ik} ~ Bernoulli(π_k)`, so with
// occupancy `M_k = Σ_i z_{ik}` out of `N` rows the marginal (integrating out the
// conjugate `π_k`) is Beta–Bernoulli:
//   log P(M_k | a_k, N) = logΓ(M_k + a_k) − logΓ(N + a_k + 1) + ln a_k + const.
// Its `a_k`-score and curvature are elementary digamma/trigamma expressions
// (`g`, `g'` below). The schedule concentration is then the stationary point of
// `Σ_k log P(M_k | a_k(θ), N)`, found by a guarded Newton root-find on the score
// in `θ = ln α` — analytic first AND second derivatives, NO grid, NO finite
// differences (SPEC). This M-step MOVES `log_lambda_sparse` exactly in the
// large-K / streaming regime where the value-lane gradient is identically zero,
// and the occupancy feedback un-caps the effective atom count that a fixed
// `α = 1` geometric schedule structurally pins at ~3.
// ---------------------------------------------------------------------------

/// Per-atom Beta–Bernoulli marginal score `g(a) = ψ(M+a) − ψ(N+a+1) + 1/a` and
/// its derivative `g'(a) = ψ'(M+a) − ψ'(N+a+1) − 1/a²` in the pseudo-count `a`.
/// This is the schedule-INDEPENDENT core: every schedule reaches the M-step
/// through the same `(g, g')`, differing only in the `a_k(θ)` map it feeds.
#[inline]
fn ordered_beta_bernoulli_eb_atom_score_deriv(m_k: f64, n_obs: f64, a: f64) -> (f64, f64) {
    let g = statrs::function::gamma::digamma(m_k + a)
        - statrs::function::gamma::digamma(n_obs + a + 1.0)
        + 1.0 / a;
    let gp = trigamma(m_k + a) - trigamma(n_obs + a + 1.0) - 1.0 / (a * a);
    (g, gp)
}

/// Schedule-agnostic empirical-Bayes marginal score `Σ_k g(a_k)·(da_k/dθ)`.
///
/// The Beta–Bernoulli marginal is identical for the geometric and power-law
/// schedules; only the `a_k(θ)` map and its `θ`-derivative change (#F2). A
/// schedule supplies its own `a` (pseudo-counts) and `da_dtheta` arrays and this
/// core assembles the total score. Exposed so a power-law fit reaches the SAME
/// fixed point as the geometric one.
pub fn ordered_beta_bernoulli_eb_marginal_score(occupancy: &[f64], n_obs: f64, a: &[f64], da_dtheta: &[f64]) -> f64 {
    let mut s = 0.0;
    for k in 0..occupancy.len() {
        let (g, _) = ordered_beta_bernoulli_eb_atom_score_deriv(occupancy[k].clamp(0.0, n_obs), n_obs, a[k]);
        s += g * da_dtheta[k];
    }
    s
}

/// Total empirical-Bayes marginal score `S(θ)` and curvature `H(θ)` for the
/// ordered GEOMETRIC ordered Beta--Bernoulli schedule, parameterised in `θ = ln α` (so the additive
/// engine step `log_lambda_sparse += Δθ` IS the multiplicative α update).
///
/// Uses `ρ = σ(θ) = α/(α+1)` (`dρ/dθ = ρ(1−ρ)`, `d²ρ/dθ² = ρ(1−ρ)(1−2ρ)`),
/// `μ_k = ρ^{k+1}`, and `a_k = μ_k/(1−μ_k)`, all differentiated analytically.
/// `occupancy[k] = M_k`, `n_obs = N`. Pure closed form (digamma/trigamma).
pub fn ordered_beta_bernoulli_eb_alpha_score_hess(occupancy: &[f64], n_obs: f64, alpha: f64) -> (f64, f64) {
    let rho = alpha / (alpha + 1.0);
    let one_m_rho = 1.0 - rho;
    let mut s = 0.0;
    let mut h = 0.0;
    for (k, &m_raw) in occupancy.iter().enumerate() {
        let u = (k + 1) as f64;
        let m_k = m_raw.clamp(0.0, n_obs);
        // μ_k = ρ^u and its θ-derivatives (clamp below 1 to keep a_k finite).
        let mu = rho.powf(u).clamp(f64::MIN_POSITIVE, 1.0 - 1.0e-12);
        let dmu = u * mu * one_m_rho; // dμ/dθ
        let d2mu = u * mu * one_m_rho * (u * one_m_rho - rho); // d²μ/dθ²
        // a_k = μ/(1−μ) and its θ-derivatives.
        let om = 1.0 - mu;
        let a = mu / om;
        let da = dmu / (om * om);
        let d2a = 2.0 * dmu * dmu / (om * om * om) + d2mu / (om * om);
        let (g, gp) = ordered_beta_bernoulli_eb_atom_score_deriv(m_k, n_obs, a);
        s += g * da;
        h += gp * da * da + g * d2a;
    }
    (s, h)
}

/// Occupancy-driven empirical-Bayes concentration `α*` for the ordered geometric
/// ordered Beta--Bernoulli prior: the stationary point of the Beta–Bernoulli marginal, found by a
/// GUARDED Newton root-find on `S(θ) = 0` in `θ = ln α` with analytic score and
/// curvature (no grid, no finite differences — SPEC). Guarding: a Newton step
/// where the marginal is locally concave (`H < 0`), otherwise a trust-region
/// gradient step; `θ` is clamped to a wide finite band and the per-iterate step
/// is bounded, so a monotone (data-wants-the-boundary) marginal converges to the
/// clamp instead of diverging. Returns a finite `α* > 0`.
pub fn ordered_beta_bernoulli_eb_geometric_alpha_fixed_point(occupancy: &[f64], n_obs: f64, alpha_seed: f64) -> f64 {
    const THETA_LO: f64 = -12.0; // α ≈ 6e-6
    const THETA_HI: f64 = 16.0; // α ≈ 8.9e6
    const NEWTON_MAX_ITERS: usize = 100;
    const STEP_TR: f64 = 1.0; // trust region on |Δθ| per iterate
    const TOL: f64 = 1.0e-10;
    if !(n_obs > 0.0) || occupancy.is_empty() {
        return alpha_seed;
    }
    let seed = if alpha_seed.is_finite() && alpha_seed > 0.0 {
        alpha_seed
    } else {
        1.0
    };
    let mut theta = seed.ln().clamp(THETA_LO, THETA_HI);
    for _ in 0..NEWTON_MAX_ITERS {
        let (s, h) = ordered_beta_bernoulli_eb_alpha_score_hess(occupancy, n_obs, theta.exp());
        if !s.is_finite() || s.abs() < TOL {
            break;
        }
        let mut step = if h < -1.0e-12 {
            -s / h
        } else {
            s.signum() * STEP_TR
        };
        if !step.is_finite() {
            break;
        }
        step = step.clamp(-STEP_TR, STEP_TR);
        let new_theta = (theta + step).clamp(THETA_LO, THETA_HI);
        let converged = (new_theta - theta).abs() < TOL;
        theta = new_theta;
        if converged {
            break;
        }
    }
    theta.exp()
}

/// Posterior-mean Bernoulli activations for the ordered Beta--Bernoulli assignment model.
///
/// Ordered shrinkage belongs to the Beta--Bernoulli prior scored by
/// [`OrderedBetaBernoulliPenalty`], not as a second multiplicative factor on the final
/// reconstruction. Multiplying by the prior mean capped atom `k` at `mu_k < 1`
/// even when its posterior inclusion probability was one, double-counted the
/// prior, and made the fitted function depend on atom index. The concrete
/// posterior mean is therefore simply `sigmoid(logit_k / temperature)`.
pub fn ordered_beta_bernoulli_row(logits: ArrayView1<'_, f64>, temperature: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        out[i] = gam_linalg::utils::stable_logistic(logits[i] / temperature);
    }
    out
}

/// ordered Beta--Bernoulli-MAP activations together with the diagonal Jacobian `∂z_k/∂l_k`,
/// shared with the torch autograd `Function` so the Python ordered Beta--Bernoulli-Gumbel path
/// applies the same posterior-mean gate and temperature scaling as the Rust
/// closed form. With `z_k = σ(l_k/τ)` the per-atom derivative is
/// `σ(l_k/τ)(1 − σ(l_k/τ)) / τ`; the map is diagonal in `k`, so the
/// Jacobian is returned as the per-atom diagonal vector.
#[must_use]
pub fn ordered_beta_bernoulli_row_value_grad(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
) -> (Array1<f64>, Array1<f64>) {
    let inv_tau = 1.0 / temperature;
    let mut value = Array1::<f64>::zeros(logits.len());
    let mut grad = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        let sig = gam_linalg::utils::stable_logistic(logits[i] * inv_tau);
        value[i] = sig;
        grad[i] = sig * (1.0 - sig) * inv_tau;
    }
    (value, grad)
}

/// Batched ordered Beta--Bernoulli-MAP value and diagonal logit Jacobian over an `(N, K)` logit
/// matrix. This shares the per-element arithmetic of
/// [`ordered_beta_bernoulli_row_value_grad`] while crossing the Rust/Python boundary once for
/// the whole batch.
#[must_use]
pub fn ordered_beta_bernoulli_batch_value_grad(
    logits: ArrayView2<'_, f64>,
    temperature: f64,
) -> (Array2<f64>, Array2<f64>) {
    let (n, k) = logits.dim();
    let inv_tau = 1.0 / temperature;
    let mut value = Array2::<f64>::zeros((n, k));
    let mut grad = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            let sig = gam_linalg::utils::stable_logistic(logits[[i, j]] * inv_tau);
            value[[i, j]] = sig;
            grad[[i, j]] = sig * (1.0 - sig) * inv_tau;
        }
    }
    (value, grad)
}

pub fn threshold_gate_row(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
    threshold: f64,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        out[i] = gam_linalg::utils::stable_logistic((logits[i] - threshold) / temperature);
    }
    out
}

/// Fitted-encoder gate activations `a = gate(logits)` for a full `(N, K)` logit
/// matrix under the named assignment family.
///
/// This is the SINGLE SOURCE OF TRUTH for "turn a fitted model's routing logits
/// into per-atom activations": the closed-form fitter (via [`softmax_row`] /
/// [`ordered_beta_bernoulli_row`] / [`threshold_gate_row`]) and the post-hoc distilled-encoder path
/// both read it, so a distilled encoder's activation is bit-identical to the
/// model it distills. Formerly the Python `gamfit.distill` module re-derived
/// this math and drifted (issue #2011: Python is a thin wrapper, no shadow math).
///
/// `kind` is the canonical assignment token (`"softmax"`, `"ordered_beta_bernoulli"`,
/// or `"threshold_gate"`). `threshold` is read only for the gate family.
/// Non-finite logits and unsupported kinds are surfaced as errors.
pub fn activation_matrix_from_logits(
    logits: ArrayView2<'_, f64>,
    kind: &str,
    temperature: f64,
    threshold: f64,
) -> Result<Array2<f64>, String> {
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(format!(
            "activation_matrix_from_logits: temperature must be finite and positive; got {temperature}"
        ));
    }
    let (n_rows, k_atoms) = logits.dim();
    let mut out = Array2::<f64>::zeros((n_rows, k_atoms));
    for row in 0..n_rows {
        let row_logits = logits.row(row);
        validate_finite_logits(row_logits, row)?;
        let activation = match kind {
            "softmax" => softmax_row(row_logits, temperature),
            "ordered_beta_bernoulli" => ordered_beta_bernoulli_row(row_logits, temperature),
            "threshold_gate" => threshold_gate_row(row_logits, temperature, threshold),
            other => {
                return Err(format!(
                    "activation_matrix_from_logits: unsupported assignment kind {other:?} \
                     (expected 'softmax', 'ordered_beta_bernoulli', or 'threshold_gate')"
                ));
            }
        };
        out.row_mut(row).assign(&activation);
    }
    Ok(out)
}

/// Hard top-`k` support row (the [`AssignmentMode::TopK`] gate): 1.0 for the
/// `k` largest routing logits in the row, 0.0 elsewhere. Ties break toward the
/// LOWER atom index so the support is deterministic. `k ≥ len` degenerates to
/// the all-active row. Logits are validated finite upstream.
pub fn topk_row(logits: ArrayView1<'_, f64>, k: usize) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    topk_row_into(
        logits,
        k,
        out.as_slice_mut()
            .expect("freshly allocated 1-D array is contiguous"),
    );
    out
}

/// Fill-into-caller-buffer twin of [`topk_row`] — bit-identical values, no
/// allocation beyond the O(K) index scratch. Average O(K) via quickselect.
pub(crate) fn topk_row_into(logits: ArrayView1<'_, f64>, k: usize, out: &mut [f64]) {
    let n = logits.len();
    if k >= n {
        out[..n].fill(1.0);
        return;
    }
    out[..n].fill(0.0);
    let mut idx: Vec<usize> = (0..n).collect();
    // Larger logit first; equal logits fall back to index order so the
    // boundary atom is deterministic across runs and chunkings.
    idx.select_nth_unstable_by(k, |&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    for &i in &idx[..k] {
        out[i] = 1.0;
    }
}

/// Smooth bounded threshold-gate activations and their exact diagonal Jacobian.
/// With `a_k = σ((l_k − θ_k)/τ)`, `∂a_k/∂l_k = a_k(1-a_k)/τ` and
/// `∂a_k/∂θ_k = -∂a_k/∂l_k` on the entire real line.
#[must_use]
pub fn threshold_gate_row_value_grad(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
    thresholds: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>) {
    assert_eq!(
        logits.len(),
        thresholds.len(),
        "threshold_gate_row_value_grad: logits/thresholds length mismatch"
    );
    let inv_tau = 1.0 / temperature;
    let mut value = Array1::<f64>::zeros(logits.len());
    let mut grad = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        let sig = gam_linalg::utils::stable_logistic((logits[i] - thresholds[i]) * inv_tau);
        value[i] = sig;
        grad[i] = sig * (1.0 - sig) * inv_tau;
    }
    (value, grad)
}

/// Batched bounded threshold-gate value+grad over an `(N, K)` logit matrix,
/// sharing the exact per-atom arithmetic of [`threshold_gate_row_value_grad`] so a
/// single batched call is bit-identical to invoking the row kernel row-by-row.
///
/// `thresholds` is per-atom (length `K`, broadcast across the `N` rows). Returns
/// `(value, grad)`, each `(N, K)`:
///   * `value[i, k] = σ((l − θ)/τ)` — the bounded `(0, 1)` gate,
///   * `grad[i, k]  = σ·(1 − σ)/τ` — its exact diagonal derivative.
///
/// This is the single source of truth for `gamfit.torch`'s bounded jumprelu
/// gate: the torch autograd `Function` crosses the FFI boundary ONCE with the
/// whole matrix instead of once per row.
#[must_use]
pub fn threshold_gate_batch_value_grad(
    logits: ArrayView2<'_, f64>,
    temperature: f64,
    thresholds: ArrayView1<'_, f64>,
) -> (Array2<f64>, Array2<f64>) {
    let (n, k) = logits.dim();
    assert_eq!(
        k,
        thresholds.len(),
        "threshold_gate_batch_value_grad: logits columns {k} != thresholds length {}",
        thresholds.len()
    );
    let inv_tau = 1.0 / temperature;
    let mut value = Array2::<f64>::zeros((n, k));
    let mut grad = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            let sig =
                gam_linalg::utils::stable_logistic((logits[[i, j]] - thresholds[j]) * inv_tau);
            value[[i, j]] = sig;
            grad[[i, j]] = sig * (1.0 - sig) * inv_tau;
        }
    }
    (value, grad)
}

/// Exact numerical inverse of the softplus link `softplus(x) = log(1 + eˣ)`
/// (the forward direction is [`gam_linalg::utils::stable_softplus`], used by
/// [`topk_activation_row_value_grad`] below). This is the single source of
/// truth for the softplus⁻¹ reparameterization the SAE penalty FFI uses to map
/// a positive scale hyperparameter `β > 0` back to its raw pre-softplus
/// coordinate `raw = softplus⁻¹(β)` (the `raw_beta` of the parametric
/// row-precision / aux-conditional priors). Moved out of the pyffi shim
/// (`geometry_ffi::inverse_softplus_scalar`) so no numeric policy lives in the
/// FFI layer.
///
/// Domain / stability contract (preserved exactly from the shim):
///   * `value ≤ 0` or `NaN` → `NaN` (softplus is strictly positive, so its
///     inverse is undefined off the positive reals);
///   * `value > 30` uses the overflow-safe identity
///     `softplus⁻¹(v) = v + log1p(−e^{−v})` (`eᵛ` would overflow);
///   * otherwise the direct `log(e^v − 1) = ln(expm1(v))`.
#[must_use]
pub fn inverse_softplus(value: f64) -> f64 {
    if value <= 0.0 || value.is_nan() {
        f64::NAN
    } else if value > 30.0 {
        value + (-(-value).exp()).ln_1p()
    } else {
        value.exp_m1().ln()
    }
}

/// Top-k SAE activation value+grad: the per-atom **independent**, strictly
/// non-negative activation `a_k = τ · softplus(l_k / τ)` and its diagonal logit
/// derivative `∂a_k/∂l_k = σ(l_k / τ)`.
///
/// This is the smooth, temperature-annealed activation the `softmax_topk` SAE
/// gate scores atoms with (the hard top-k *selection* and its masked gradient
/// stay on the torch tape — see `gamfit.torch`'s `_topk_gate`). `τ → 0` anneals
/// it to a plain ReLU. The activation is computed independently per atom (no
/// row-wise softmax competition), which is what lets an atom sit near zero when
/// its feature is absent and rise on its own when present (issue #583).
///
/// Because `a = τ·softplus(l/τ)`, the chain rule gives
/// `da/dl = τ · softplus'(l/τ) · (1/τ) = softplus'(l/τ) = σ(l/τ)`, so the
/// temperature cancels out of the derivative. This is the single source of
/// truth shared with `gamfit.torch`'s top-k activation so the torch lane's
/// forward/backward match the Rust-defined family exactly (parity-pinned).
#[must_use]
pub fn topk_activation_row_value_grad(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
) -> (Array1<f64>, Array1<f64>) {
    let inv_tau = 1.0 / temperature;
    let mut value = Array1::<f64>::zeros(logits.len());
    let mut grad = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        let scaled = logits[i] * inv_tau;
        value[i] = temperature * gam_linalg::utils::stable_softplus(scaled);
        grad[i] = gam_linalg::utils::stable_logistic(scaled);
    }
    (value, grad)
}

/// Batched sibling of [`topk_activation_row_value_grad`] over an `(N, K)` logit
/// matrix, sharing the EXACT per-atom arithmetic (same `stable_softplus`,
/// `stable_logistic`, same `l * inv_tau` order) so a single batched call is
/// bit-identical to invoking the row kernel row-by-row.
///
/// Returns `(value, grad)`, each `(N, K)`:
///   * `value[i, k] = τ · softplus(l / τ)` — the non-negative activation,
///   * `grad[i, k]  = σ(l / τ)` — the diagonal derivative `∂a/∂l`.
#[must_use]
pub fn topk_activation_batch_value_grad(
    logits: ArrayView2<'_, f64>,
    temperature: f64,
) -> (Array2<f64>, Array2<f64>) {
    let (n, k) = logits.dim();
    let inv_tau = 1.0 / temperature;
    let mut value = Array2::<f64>::zeros((n, k));
    let mut grad = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            let scaled = logits[[i, j]] * inv_tau;
            value[[i, j]] = temperature * gam_linalg::utils::stable_softplus(scaled);
            grad[[i, j]] = gam_linalg::utils::stable_logistic(scaled);
        }
    }
    (value, grad)
}

#[cfg(test)]
mod ordered_beta_bernoulli_batch_tests {
    use super::*;

    #[test]
    fn ordered_beta_bernoulli_batch_matches_row_kernel_bit_for_bit() {
        let n = 5usize;
        let k = 7usize;
        let temperature = 0.41_f64;
        let logits = Array2::from_shape_fn((n, k), |(i, j)| {
            ((i as f64) * 0.37 - (j as f64) * 0.19 + 0.11).sin()
        });

        let (value, grad) = ordered_beta_bernoulli_batch_value_grad(logits.view(), temperature);
        assert_eq!(value.dim(), (n, k));
        assert_eq!(grad.dim(), (n, k));

        for i in 0..n {
            let (rv, rg) = ordered_beta_bernoulli_row_value_grad(logits.row(i), temperature);
            for j in 0..k {
                assert_eq!(value[[i, j]], rv[j], "value mismatch at row {i} atom {j}");
                assert_eq!(grad[[i, j]], rg[j], "grad mismatch at row {i} atom {j}");
            }
        }
    }
}

#[cfg(test)]
mod topk_activation_tests {
    use super::*;

    #[test]
    fn topk_activation_batch_matches_row_kernel_bit_for_bit() {
        let n = 5usize;
        let k = 7usize;
        let temperature = 0.41_f64;
        let logits = Array2::from_shape_fn((n, k), |(i, j)| {
            ((i as f64) * 0.37 - (j as f64) * 0.19 + 0.11).sin()
        });

        let (value, grad) = topk_activation_batch_value_grad(logits.view(), temperature);
        assert_eq!(value.dim(), (n, k));
        assert_eq!(grad.dim(), (n, k));

        for i in 0..n {
            let (rv, rg) = topk_activation_row_value_grad(logits.row(i), temperature);
            for j in 0..k {
                assert_eq!(value[[i, j]], rv[j], "value mismatch at row {i} atom {j}");
                assert_eq!(grad[[i, j]], rg[j], "grad mismatch at row {i} atom {j}");
            }
        }
    }

    #[test]
    fn topk_activation_is_nonnegative_and_grad_is_logistic() {
        // Independent per-atom activation: strictly non-negative value, and the
        // derivative equals σ(l/τ) exactly (temperature cancels in the chain).
        let temperature = 0.7_f64;
        let logits = Array1::from(vec![-4.0_f64, -0.5, 0.0, 0.5, 4.0]);
        let (value, grad) = topk_activation_row_value_grad(logits.view(), temperature);
        for (&v, &g) in value.iter().zip(grad.iter()) {
            assert!(v >= 0.0, "activation must be non-negative, got {v}");
            assert!(
                (0.0..=1.0).contains(&g),
                "grad must be a logistic in [0,1], got {g}"
            );
        }
        // Spot-check the closed form at l = 0: τ·softplus(0) = τ·ln 2, σ(0) = 0.5.
        assert!((value[2] - temperature * 2.0_f64.ln()).abs() < 1e-15);
        assert!((grad[2] - 0.5).abs() < 1e-15);
    }
}

#[cfg(test)]
mod topk_support_gate_tests {
    // Contract tests for the [`AssignmentMode::TopK`] hard-support gate: the
    // support is EXACTLY the k largest routing logits (deterministic lower-index
    // tie-break), L0 is exactly k, the fill-into twin is bit-identical, and the
    // all-equal neutral support is the first k atoms.
    use super::*;

    #[test]
    fn topk_row_selects_exact_support_and_l0_is_k() {
        let logits = Array1::from(vec![0.3_f64, 0.9, 0.9, -1.0, 0.5]);
        let g = topk_row(logits.view(), 3);
        assert_eq!(g.to_vec(), vec![0.0, 1.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            g.iter().filter(|&&v| v == 1.0).count(),
            3,
            "L0 must equal k exactly"
        );
        assert!(
            g.iter().all(|&v| v == 0.0 || v == 1.0),
            "gates are hard {{0,1}}"
        );
    }

    #[test]
    fn topk_boundary_tie_breaks_toward_lower_index() {
        let logits = Array1::from(vec![1.0_f64, 0.5, 0.5, 0.1]);
        let g = topk_row(logits.view(), 2);
        assert_eq!(
            g.to_vec(),
            vec![1.0, 1.0, 0.0, 0.0],
            "the tied boundary atom with the LOWER index wins deterministically"
        );
    }

    #[test]
    fn topk_row_into_is_bit_identical_and_k_ge_n_is_all_active() {
        let logits = Array1::from(vec![-0.2_f64, 3.0, 0.7, 0.7, -5.0, 2.2]);
        for k in [1usize, 2, 4, 6, 9] {
            let alloc = topk_row(logits.view(), k);
            let mut buf = vec![f64::NAN; logits.len()];
            topk_row_into(logits.view(), k, &mut buf);
            assert_eq!(
                alloc.to_vec(),
                buf,
                "into-twin must be bit-identical at k={k}"
            );
        }
        let all = topk_row(logits.view(), 99);
        assert!(
            all.iter().all(|&v| v == 1.0),
            "k >= n degenerates to all-active"
        );
    }

    #[test]
    fn topk_neutral_support_is_first_k_atoms() {
        let w = neutral_gate_weights(AssignmentMode::top_k_support(3), 6);
        assert_eq!(w.to_vec(), vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn topk_mode_carries_no_temperature_or_prior_knobs() {
        let mode = AssignmentMode::top_k_support(4);
        mode.validate().expect("k >= 1 validates");
        assert!(
            AssignmentMode::top_k_support(0).validate().is_err(),
            "k = 0 must be rejected"
        );
    }
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
// These compute the EXACT SAME values as `softmax_row` / `ordered_beta_bernoulli_row` /
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

pub(crate) fn ordered_beta_bernoulli_row_into(logits: ArrayView1<'_, f64>, temperature: f64, out: &mut [f64]) {
    for i in 0..logits.len() {
        out[i] = gam_linalg::utils::stable_logistic(logits[i] / temperature);
    }
}

pub(crate) fn threshold_gate_row_into(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
    threshold: f64,
    out: &mut [f64],
) {
    for i in 0..logits.len() {
        out[i] = gam_linalg::utils::stable_logistic((logits[i] - threshold) / temperature);
    }
}

pub(crate) struct ActiveAtomLogitJvp<'a> {
    pub(crate) mode: AssignmentMode,
    pub(crate) logit_k: f64,
    pub(crate) a_k: f64,
    pub(crate) decoded_k: ArrayView1<'a, f64>,
    pub(crate) fitted: ArrayView1<'a, f64>,
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
        logit_k,
        a_k,
        decoded_k,
        fitted,
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
        AssignmentMode::OrderedBetaBernoulli { temperature, .. } => {
            // Posterior-mean Bernoulli gate `z_k = σ(l_k/τ)`.
            let inv_tau = 1.0 / temperature;
            let dz = a_k * (1.0 - a_k) * inv_tau;
            for out_col in 0..p {
                jac_compact[[compact_index, out_col]] = dz * decoded_k[out_col];
            }
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            // Exact derivative of the smooth threshold-centered logistic gate.
            let inv_tau = 1.0 / temperature;
            let activation = gam_linalg::utils::stable_logistic((logit_k - threshold) * inv_tau);
            let da = activation * (1.0 - activation) * inv_tau;
            for out_col in 0..p {
                jac_compact[[compact_index, out_col]] = da * decoded_k[out_col];
            }
        }
        // Constant {0, 1} gate: zero logit-JVP (no free logit exists; TopK rows
        // carry no logit slots in the compact layout, so this is unreachable —
        // kept as the explicit zero for exhaustiveness).
        AssignmentMode::TopK { .. } => {}
    }
}

pub(crate) fn fill_assignment_logit_jvp_rows(
    mode: AssignmentMode,
    logits: ArrayView1<'_, f64>,
    assignments: ArrayView1<'_, f64>,
    decoded: ArrayView2<'_, f64>,
    fitted: ArrayView1<'_, f64>,
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
        AssignmentMode::OrderedBetaBernoulli { temperature, .. } => {
            // Posterior-mean Bernoulli gate `z_k = σ(l_k/τ)`; ordered
            // stick-breaking shrinkage is scored once, in the ordered Beta--Bernoulli prior.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() {
                if is_ungated(logit_col) {
                    continue;
                }
                let a_k = assignments[logit_col];
                let dz = a_k * (1.0 - a_k) * inv_tau;
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = dz * decoded[[logit_col, out_col]];
                }
            }
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            // Exact derivative of the smooth threshold-centered logistic gate.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() {
                if is_ungated(logit_col) {
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
        // Constant {0, 1} gates: zero data-fit logit derivative everywhere (no
        // logit is a free parameter — the caller's fixed-logit mask already
        // skips every column; this arm keeps the JVP identically zero).
        AssignmentMode::TopK { .. } => {}
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

/// Build the ordered Beta--Bernoulli sparsity penalty used by every assignment-prior term at `rho`,
/// honoring #Bug6 (α is FIXED to the forward-gate value whenever an override
/// pins it — `effective_alpha_is_learnable`, `resolved_ordered_beta_bernoulli_alpha`) and #Bug4
/// (ungated atoms are inert columns excluded from value/gradient/curvature).
/// Returns `(penalty, rho_view)`; the fixed-α branch uses the `lambda_sparse`
/// weight convention with an empty `rho_view`.
fn ordered_beta_bernoulli_prior_penalty(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    base_alpha: f64,
    temperature: f64,
    row_weights: Option<&[f64]>,
) -> (OrderedBetaBernoulliPenalty, Array1<f64>) {
    let learnable = assignment.effective_alpha_is_learnable();
    let alpha_eff = if learnable {
        base_alpha
    } else {
        assignment.resolved_ordered_beta_bernoulli_alpha(rho).unwrap_or(base_alpha)
    };
    // #991 design-honesty weights: the ordered Beta--Bernoulli prior is not row-separable (the
    // plug-in π̂ couples rows through the column active mass), so the weights are
    // installed ON the penalty — its value/grad/hessian/hvp/ρ- and third
    // channels all fold them identically (weighted mass `M_k = Σ w_i z_ik` and
    // carrier `u = w·J`), keeping every channel the exact derivative of one
    // weighted energy. `None` ⇒ bit-for-bit the historical unweighted path.
    let mut penalty =
        OrderedBetaBernoulliPenalty::new(assignment.k_atoms(), alpha_eff, temperature, learnable)
            .with_row_weights(row_weights);
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

pub fn assignment_prior_value(assignment: &SaeAssignment, rho: &SaeManifoldRho) -> f64 {
    assignment_prior_value_weighted(assignment, rho, None)
}

/// As [`assignment_prior_value`], but with #991 design-honesty per-row weights:
/// row `i`'s per-row prior contribution is scaled by `w_i` (mean-1). This is the
/// per-row latent prior's analog of the `√w_i`-weighted data likelihood and the
/// `w_i`-weighted `ard_value` — each retained row of a design-honest subsample
/// stands in for `w_i` population rows, so its routing prior carries `w_i` too.
/// `None` ⇒ the unweighted path, bit-for-bit. Softmax/JumpReLU are row-separable
/// and fully weighted here; the ordered Beta--Bernoulli prior lives in the un-owned `ibp.rs` penalty
/// and is left unweighted (self-consistent) until that penalty gains a per-row
/// weight hook — see the module note on `assignment_prior_grad_hdiag`.
pub(crate) fn assignment_prior_value_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> f64 {
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
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature)
                .with_row_weights(row_weights);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            penalty.value(target.view(), rho_view.view())
        }
        AssignmentMode::OrderedBetaBernoulli {
            temperature, alpha, ..
        } => {
            let (penalty, rho_view) =
                ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights);
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
                // #991 — this row stands in for `w_i` population rows.
                let w_row = row_weights.map_or(1.0, |w| w[idx / k]);
                acc += w_row
                    * gam_linalg::utils::stable_logistic((logit - threshold) / temperature);
            }
            sparsity_strength * acc
        }
        // Sparsity by construction: the fixed-|S| support IS the sparsity — there
        // is no penalty term, so the prior contributes exactly zero.
        AssignmentMode::TopK { .. } => 0.0,
    }
}

pub fn assignment_prior_log_strength_derivative(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> f64 {
    assignment_prior_log_strength_derivative_weighted(assignment, rho, None)
}

/// #991-weighted [`assignment_prior_log_strength_derivative`]. Softmax/JumpReLU
/// route through the `w_i`-weighted value; ordered Beta--Bernoulli stays unweighted (un-owned
/// `ibp.rs`).
pub(crate) fn assignment_prior_log_strength_derivative_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
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
            assignment_prior_value_weighted(assignment, rho, row_weights)
        }
        AssignmentMode::OrderedBetaBernoulli {
            temperature, alpha, ..
        } => {
            // #Bug6: `ordered_beta_bernoulli_prior_penalty` picks the effective-α learnability (an
            // override forces the fixed-α value branch) and the #Bug4 ungated mask.
            let (penalty, rho_view) =
                ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights);
            if penalty.learnable_alpha {
                penalty.grad_rho(target.view(), rho_view.view())[0]
            } else {
                penalty.value(target.view(), rho_view.view())
            }
        }
        // No prior term ⇒ no ρ-derivative (sparsity lives in the fixed support).
        AssignmentMode::TopK { .. } => 0.0,
    }
}

pub fn assignment_prior_log_strength_hdiag(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<Array1<f64>, String> {
    assignment_prior_log_strength_hdiag_weighted(assignment, rho, None)
}

/// #991-weighted [`assignment_prior_log_strength_hdiag`]. Softmax/JumpReLU carry
/// `w_i` per row; ordered Beta--Bernoulli stays unweighted (un-owned `ibp.rs`).
pub(crate) fn assignment_prior_log_strength_hdiag_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
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
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature)
                .with_row_weights(row_weights);
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
                let activation = gam_linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                // #991 — row `idx / k`'s design weight.
                let w_row = row_weights.map_or(1.0, |w| w[idx / k]);
                d[idx] = w_row * sparsity_strength * slope * (1.0 - 2.0 * activation) * inv_tau2;
            }
            Ok(d)
        }
        AssignmentMode::OrderedBetaBernoulli {
            temperature, alpha, ..
        } => {
            let (penalty, rho_view) =
                ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights);
            let mut d = if penalty.learnable_alpha {
                penalty.hessian_diag_log_alpha_derivative(target.view(), rho_view.view())
            } else {
                penalty
                    .hessian_diag(target.view(), rho_view.view())
                    .ok_or_else(|| {
                        "ordered Beta--Bernoulli assignment log-strength hessian diag unavailable".to_string()
                    })?
            };
            // #Bug4: zero the curvature diagonal of ungated (inert) columns so the
            // log-det ρ-trace never charges them (the array methods are not
            // internally column-masked).
            mask_fixed_logit_entries(assignment, &mut d);
            Ok(d)
        }
        // No prior term ⇒ zero curvature everywhere (mirrors the frozen-routing
        // early return; the support carries no free logits at all).
        AssignmentMode::TopK { .. } => Ok(Array1::<f64>::zeros(target.len())),
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

pub fn assignment_prior_log_strength_target_mixed(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<Array1<f64>, String> {
    assignment_prior_log_strength_target_mixed_weighted(assignment, rho, None)
}

/// #991-weighted [`assignment_prior_log_strength_target_mixed`]. The fixed-α
/// fall-through reuses the `w_i`-weighted gradient; the learnable-α ordered Beta--Bernoulli branch
/// lives in the un-owned `ibp.rs` penalty and stays unweighted.
pub(crate) fn assignment_prior_log_strength_target_mixed_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
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
        AssignmentMode::OrderedBetaBernoulli {
            temperature, alpha, ..
        } if assignment.effective_alpha_is_learnable() => {
            let (penalty, rho_view) =
                ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights);
            let mut d = penalty.log_alpha_target_mixed_derivative(target.view(), rho_view.view());
            // #Bug4: inert columns carry no mixed derivative.
            mask_fixed_logit_entries(assignment, &mut d);
            Ok(d)
        }
        _ => Ok(assignment_prior_grad_hdiag_weighted(assignment, rho, row_weights)?.0),
    }
}

pub fn assignment_prior_grad_hdiag(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    assignment_prior_grad_hdiag_weighted(assignment, rho, None)
}

/// #991-weighted [`assignment_prior_grad_hdiag`] — the per-(row, atom) logit
/// gradient and Hessian diagonal of the assignment prior, each row scaled by its
/// design weight `w_i`. Softmax/JumpReLU are fully weighted (row-separable); the
/// ordered Beta--Bernoulli prior is in the un-owned `ibp.rs` penalty and is left unweighted, so a
/// design-honesty subsample under ordered Beta--Bernoulli routing keeps its current (self-consistent)
/// prior strength — closing that gap needs a per-row weight hook on
/// `OrderedBetaBernoulliPenalty::{value, grad_target, hessian_diag, grad_rho,
/// hessian_diag_logit_third_channels}`.
///
/// The assembly (`construction_arrow_schur_assembly`) consumes THIS gradient
/// unchanged for `gt`; the softmax curvature written to `htt` is the per-row
/// Gershgorin/`row_psd_majorizer` block, which its call sites weight by folding
/// `w_row` into the `scale` they pass — so the softmax gradient and curvature
/// both carry `w_i` without any double application.
pub(crate) fn assignment_prior_grad_hdiag_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
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
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature)
                .with_row_weights(row_weights);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            let g = penalty.grad_target(target.view(), rho_view.view());
            let d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| "softmax assignment hessian diag unavailable".to_string())?;
            (g, d)
        }
        AssignmentMode::OrderedBetaBernoulli {
            temperature, alpha, ..
        } => {
            // Scale the ordered Beta--Bernoulli assignment-sparsity prior by `lambda_sparse` in the
            // fixed-α branch (Softmax folds it into the penalty's rho coordinate;
            // JumpReLU multiplies `sparsity_strength`). #Bug6: `ordered_beta_bernoulli_prior_penalty`
            // picks the EFFECTIVE-α learnability — an override pins α so the prior
            // uses the fixed-α weight convention and the resolved (override) α,
            // matching the forward gate — and installs the #Bug4 ungated mask. The
            // per-atom fixed-logit columns are additionally zeroed post-hoc below,
            // so the array (grad/hessian) methods need no internal column mask.
            let (penalty, rho_view) =
                ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights);
            let g = penalty.grad_target(target.view(), rho_view.view());
            let d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| "ordered Beta--Bernoulli assignment hessian diag unavailable".to_string())?;
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
            let k = assignment.k_atoms();
            let mut g = Array1::<f64>::zeros(target.len());
            let mut d = Array1::<f64>::zeros(target.len());
            for idx in 0..target.len() {
                let logit = target[idx];
                let activation = gam_linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                // #991 — row `idx / k`'s design weight scales this row's prior
                // gradient AND curvature identically (both linear in strength).
                let w_row = row_weights.map_or(1.0, |w| w[idx / k]);
                g[idx] = w_row * sparsity_strength * slope * inv_tau;
                d[idx] = w_row * sparsity_strength * slope * (1.0 - 2.0 * activation) * inv_tau2;
            }
            (g, d)
        }
        // No sparsity prior and no free logits: zero gradient and curvature by
        // construction (every column is also masked as fixed below).
        AssignmentMode::TopK { .. } => (
            Array1::<f64>::zeros(target.len()),
            Array1::<f64>::zeros(target.len()),
        ),
    };
    grad += &sparsity_grad;
    diag += &sparsity_diag;
    // #1026/#1033 — a FIXED logit (an ungated atom's, or every atom's under
    // frozen routing) is not a free parameter, so it carries NO sparsity-prior
    // gradient or curvature. Zero its flat columns (`flat_logits` is row-major
    // `row*K + atom`) so the assembled `gt` and `htt` logit slots stay zero —
    // matching the zero logit-JVP. The column-separable ordered Beta--Bernoulli / JumpReLU priors are
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

/// Build the exact ordered Beta--Bernoulli `hessian_diag` logit third-derivative channels (#1006)
/// for the SAE log-det adjoint Γ, using the SAME penalty configuration —
/// `alpha`/`tau`/`learnable_alpha` and the `lambda_sparse` weight convention —
/// that [`assignment_prior_grad_hdiag`] assembles into `htt`. Returns `None`
/// for non-ordered Beta--Bernoulli assignment modes (no cross-row empirical-π coupling to correct).
pub fn ordered_beta_bernoulli_assignment_third_channels(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    majorize: bool,
) -> Result<Option<OrderedBetaBernoulliHessianDiagThirdChannels>, String> {
    ordered_beta_bernoulli_assignment_third_channels_weighted(assignment, rho, majorize, None)
}

/// As [`ordered_beta_bernoulli_assignment_third_channels`], with the #991 design-honesty per-row
/// weights the assembled `htt` carried (the channels must differentiate the
/// SAME weighted operator; `z_jac` then carries the weighted carrier `u = w·J`
/// and `logit_curvature` its slot derivative `w·c`).
pub(crate) fn ordered_beta_bernoulli_assignment_third_channels_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    majorize: bool,
    row_weights: Option<&[f64]>,
) -> Result<Option<OrderedBetaBernoulliHessianDiagThirdChannels>, String> {
    let AssignmentMode::OrderedBetaBernoulli {
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
    let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights);
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
mod ordered_beta_bernoulli_prior_614_tests {
    // #614: `ibp_stick_breaking_prior` used to compute `π_k = (α/(α+1))^k` with
    // `π_0 = 1`, i.e. an UNSHRUNK first atom — the prior mean of no stick at all,
    // which broke α's role as an ordered Beta--Bernoulli concentration parameter. The consistent
    // truncated-ordered Beta--Bernoulli stick-breaking prior mean is `π_k = (α/(α+1))^{k+1}`, the
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
            let prior = ordered_prior_means(8, OrderedPriorSchedule::Geometric { alpha });
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
            let prior = ordered_prior_means(k, OrderedPriorSchedule::Geometric { alpha });
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
        // tail (e.g. π_4) carries more mass. This is the ordered Beta--Bernoulli-concentration role
        // the #614 fix restored.
        let lo = ordered_prior_means(8, OrderedPriorSchedule::Geometric { alpha: 0.5 });
        let hi = ordered_prior_means(8, OrderedPriorSchedule::Geometric { alpha: 5.0 });
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

    fn ordered_beta_bernoulli_assignment(n: usize, k: usize) -> SaeAssignment {
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
            AssignmentMode::ordered_beta_bernoulli(0.5, 1.0, false),
        )
        .unwrap()
    }

    #[test]
    fn frozen_routing_decouples_gates_from_logit_updates_1033() {
        let (n, k) = (6usize, 3usize);
        let mut a = ordered_beta_bernoulli_assignment(n, k)
            .freeze_routing_from_current_logits()
            .unwrap();
        assert!(a.routing_is_frozen());
        // Gates BEFORE mutating the free logits.
        let before: Vec<Array1<f64>> = (0..n).map(|r| a.try_assignments_row(r).unwrap()).collect();
        // Simulate an inner-fit logit update (what the ρ-search would otherwise do
        // every eval): perturb every free logit substantially.
        a.logits.mapv_inplace(|v| v + 5.0);
        let after: Vec<Array1<f64>> = (0..n).map(|r| a.try_assignments_row(r).unwrap()).collect();
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
        let a = ordered_beta_bernoulli_assignment(n, k)
            .freeze_routing_from_current_logits()
            .unwrap();
        // The ρ-invariance is now STRUCTURAL: the assignment APIs take no ρ
        // (the signature is the proof). What remains observable is purity —
        // repeated reads of a frozen row must be identical.
        for r in 0..n {
            let ga = a.try_assignments_row(r).unwrap();
            let gb = a.try_assignments_row(r).unwrap();
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
        let mut a = ordered_beta_bernoulli_assignment(n, k)
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
    //! [`SaeAssignment::try_assignments_row_into`] must produce
    //! BIT-IDENTICAL output to the allocating
    //! [`SaeAssignment::try_assignments_row`] across every assignment
    //! mode (Softmax, OrderedBetaBernoulli, JumpReLU), the #1026 ungated case, and the K==1
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

    fn assert_into_matches_alloc(a: &SaeAssignment) {
        let n = a.n_obs();
        let k = a.k_atoms();
        let mut scratch = vec![f64::NAN; k];
        for row in 0..n {
            let allocated = a.try_assignments_row(row).unwrap();
            // Pre-fill with NaN so a partial write (e.g. a JumpReLU below-threshold
            // entry left untouched) is caught as a mismatch, not silently passed.
            for s in scratch.iter_mut() {
                *s = f64::NAN;
            }
            a.try_assignments_row_into(row, &mut scratch).unwrap();
            assert_eq!(allocated.len(), k);
            for kk in 0..k {
                assert_eq!(
                    allocated[kk], scratch[kk],
                    "row {row} atom {kk}: _into must be BIT-IDENTICAL to the allocating \
                     try_assignments_row; got {} vs {}",
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
    fn ordered_beta_bernoulli_into_is_bit_identical() {
        // Both learnable and fixed alpha exercise the resolved-alpha branch.
        assert_into_matches_alloc(&build(7, 5, AssignmentMode::ordered_beta_bernoulli(0.6, 1.3, false)));
        assert_into_matches_alloc(&build(7, 5, AssignmentMode::ordered_beta_bernoulli(0.6, 1.3, true)));
    }

    #[test]
    fn jumprelu_into_is_bit_identical() {
        // Threshold chosen so SOME atoms fall below it (the untouched-entry path)
        // and some clear it (the sigmoid path) — both branches are exercised.
        assert_into_matches_alloc(&build(7, 5, AssignmentMode::threshold_gate(0.9, 0.2)));
    }

    #[test]
    fn ungated_into_is_bit_identical() {
        // #1026 ungated overwrite under a gate-style mode (ordered Beta--Bernoulli/JumpReLU allow it).
        let a = build(6, 4, AssignmentMode::ordered_beta_bernoulli(0.6, 1.1, false))
            .with_ungated(vec![false, true, false, true])
            .unwrap();
        assert_into_matches_alloc(&a);
        let j = build(6, 4, AssignmentMode::threshold_gate(0.9, 0.15))
            .with_ungated(vec![true, false, true, false])
            .unwrap();
        assert_into_matches_alloc(&j);
    }

    #[test]
    fn k_equals_one_into_is_bit_identical() {
        // Softmax K==1 hits the fixed-unit early return; ordered Beta--Bernoulli/JumpReLU K==1 keep a
        // free per-atom gate and fall through to the real row functions.
        assert_into_matches_alloc(&build(5, 1, AssignmentMode::softmax(1.0)));
        assert_into_matches_alloc(&build(5, 1, AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false)));
        assert_into_matches_alloc(&build(5, 1, AssignmentMode::threshold_gate(0.8, 0.1)));
    }
}

#[cfg(test)]
mod ordered_beta_bernoulli_eb_alpha_f1_tests {
    //! #F1/#F2 — empirical-Bayes concentration M-step + power-law schedule.
    use super::*;

    // a_k(θ) for the geometric schedule: α = e^θ, ρ = α/(α+1), μ = ρ^{k+1},
    // a = μ/(1−μ). Independent re-derivation used by the brute-force checks.
    fn geometric_a(theta: f64, k: usize) -> f64 {
        let alpha = theta.exp();
        let rho = alpha / (alpha + 1.0);
        let mu = rho.powf((k + 1) as f64);
        mu / (1.0 - mu)
    }

    // Total Beta–Bernoulli marginal L(θ) (θ-dependent terms only), computed
    // straight from ln_gamma — the "brute digamma eval" reference the analytic
    // score/curvature are checked against.
    fn brute_marginal(occupancy: &[f64], n_obs: f64, theta: f64) -> f64 {
        let mut l = 0.0;
        for (k, &m) in occupancy.iter().enumerate() {
            let a = geometric_a(theta, k);
            l += statrs::function::gamma::ln_gamma(m + a)
                - statrs::function::gamma::ln_gamma(n_obs + a + 1.0)
                + a.ln();
        }
        l
    }

    #[test]
    fn geometric_delegate_is_bit_identical() {
        // The refactor must leave the geometric prior byte-for-byte unchanged.
        for &alpha in &[0.3_f64, 1.0, 4.5, 37.0] {
            for &k in &[1usize, 3, 8, 64] {
                let old = ordered_prior_means(k, OrderedPriorSchedule::Geometric { alpha });
                let via = ordered_prior_means(k, OrderedPriorSchedule::Geometric { alpha });
                for j in 0..k {
                    assert_eq!(
                        old[j], via[j],
                        "geometric prior drift at alpha={alpha}, k={j}"
                    );
                }
            }
        }
    }

    #[test]
    fn power_law_schedule_round_trips() {
        // μ_k = c/(k+k0)^s, positive, decreasing, clamped ≤ 1.
        let (c, s, k0) = (0.9_f64, 1.2_f64, 1.0_f64);
        let k = 32usize;
        let mu = ordered_prior_means(k, OrderedPriorSchedule::PowerLaw { c, s, k0 });
        for j in 0..k {
            let expected = (c / ((j as f64) + k0).powf(s)).clamp(f64::MIN_POSITIVE, 1.0);
            assert!(
                (mu[j] - expected).abs() <= 1e-12 * expected.max(1.0),
                "power-law mismatch at k={j}: {} vs {expected}",
                mu[j]
            );
            assert!(mu[j] > 0.0 && mu[j] <= 1.0, "μ_{j}={} out of (0,1]", mu[j]);
            if j > 0 {
                assert!(mu[j] <= mu[j - 1], "power-law not decreasing at k={j}");
            }
        }
    }

    #[test]
    fn eb_alpha_score_matches_brute_digamma() {
        // Analytic S(θ), H(θ) must match a central finite difference of the
        // independently-computed Beta–Bernoulli marginal (score = dL/dθ,
        // curvature = dS/dθ). FD lives ONLY in the test; production is closed form.
        let n_obs = 500.0_f64;
        let occupancy = vec![300.0_f64, 120.0, 60.0, 30.0, 12.0, 5.0, 2.0, 1.0];
        let h = 1e-6_f64;
        for &alpha in &[0.4_f64, 1.0, 3.0, 12.0] {
            let theta = alpha.ln();
            let (s, hess) = ordered_beta_bernoulli_eb_alpha_score_hess(&occupancy, n_obs, alpha);

            let l_plus = brute_marginal(&occupancy, n_obs, theta + h);
            let l_minus = brute_marginal(&occupancy, n_obs, theta - h);
            let s_fd = (l_plus - l_minus) / (2.0 * h);
            assert!(
                (s - s_fd).abs() <= 1e-4 * (1.0 + s_fd.abs()),
                "score mismatch at alpha={alpha}: analytic {s} vs FD {s_fd}"
            );

            let (s_plus, _) = ordered_beta_bernoulli_eb_alpha_score_hess(&occupancy, n_obs, (theta + h).exp());
            let (s_minus, _) = ordered_beta_bernoulli_eb_alpha_score_hess(&occupancy, n_obs, (theta - h).exp());
            let h_fd = (s_plus - s_minus) / (2.0 * h);
            assert!(
                (hess - h_fd).abs() <= 1e-3 * (1.0 + h_fd.abs()),
                "curvature mismatch at alpha={alpha}: analytic {hess} vs FD {h_fd}"
            );
        }
    }

    #[test]
    fn eb_marginal_core_is_schedule_agnostic() {
        // The schedule-agnostic score core, fed the geometric a_k and da_k/dθ,
        // reproduces the score component of the geometric-specialised routine.
        let n_obs = 400.0_f64;
        let occupancy = vec![200.0_f64, 90.0, 40.0, 18.0, 7.0, 3.0];
        let alpha = 2.5_f64;
        let theta = alpha.ln();
        let dh = 1e-6_f64;
        let (a, da): (Vec<f64>, Vec<f64>) = (0..occupancy.len())
            .map(|k| {
                let a0 = geometric_a(theta, k);
                let da = (geometric_a(theta + dh, k) - geometric_a(theta - dh, k)) / (2.0 * dh);
                (a0, da)
            })
            .unzip();
        let core = ordered_beta_bernoulli_eb_marginal_score(&occupancy, n_obs, &a, &da);
        let (s, _) = ordered_beta_bernoulli_eb_alpha_score_hess(&occupancy, n_obs, alpha);
        assert!(
            (core - s).abs() <= 1e-5 * (1.0 + s.abs()),
            "schedule-agnostic core {core} != geometric score {s}"
        );
    }

    #[test]
    fn eb_fixed_point_is_stationary_and_moves() {
        let n_obs = 1000.0_f64;
        // Flat (near-uniform) occupancy is grossly inconsistent with the steep
        // α=1 geometric decay, so the EB fixed point must RAISE α far above the
        // seed — the movement the frozen-λ_sparse bug prevented.
        let occupancy = vec![500.0_f64; 8];
        let alpha_star = ordered_beta_bernoulli_eb_geometric_alpha_fixed_point(&occupancy, n_obs, 1.0);
        assert!(
            alpha_star > 5.0,
            "flat occupancy must raise α; got {alpha_star}"
        );
        // Stationarity: at α* the analytic score is ~0 (interior) or α* railed to
        // a clamp (monotone) — here it is interior.
        let (s, _) = ordered_beta_bernoulli_eb_alpha_score_hess(&occupancy, n_obs, alpha_star);
        assert!(
            s.abs() < 1e-4,
            "score not stationary at α*={alpha_star}: S={s}"
        );

        // Conversely, occupancy that already matches a small-α steep decay pulls
        // α back down toward that value.
        let steep: Vec<f64> = (0..8).map(|k| n_obs * 0.5_f64.powi(k as i32 + 1)).collect();
        let alpha_low = ordered_beta_bernoulli_eb_geometric_alpha_fixed_point(&steep, n_obs, 20.0);
        assert!(
            alpha_low < 5.0,
            "steep occupancy must lower α; got {alpha_low}"
        );
    }

    fn learnable_ibp(logits: Array2<f64>, alpha_base: f64) -> SaeAssignment {
        let (n, k) = logits.dim();
        let coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
        SaeAssignment::from_blocks_with_mode(
            logits,
            coords,
            AssignmentMode::ordered_beta_bernoulli(1.0, alpha_base, true),
        )
        .unwrap()
    }

    #[test]
    fn eb_log_alpha_step_moves_for_learnable_ibp_only() {
        let n = 300usize;
        let k = 8usize;
        // Logits rising in k lift σ(l_k) toward 1, flattening occupancy relative
        // to the α=1 geometric decay, so the EB step must be a POSITIVE Δ (raise α).
        let logits = Array2::from_shape_fn((n, k), |(_, kk)| 2.0 * kk as f64);
        let assign = learnable_ibp(logits.clone(), 1.0);
        // effective α = alpha_base·exp(log_lambda_sparse) = 1·exp(0) = 1.
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k]);
        let step = assign
            .ordered_beta_bernoulli_eb_log_alpha_step(&rho)
            .unwrap()
            .expect("learnable ordered Beta--Bernoulli must yield an EB step");
        assert!(step.is_finite(), "step must be finite, got {step}");
        assert!(
            step > 1e-3,
            "flattened occupancy must MOVE λ_sparse up (raise α); got Δ={step}"
        );

        // Softmax has no closed-form M-step → None (historical zero step).
        let sm_coords: Vec<Array2<f64>> = (0..k).map(|_| Array2::<f64>::zeros((n, 1))).collect();
        let softmax = SaeAssignment::from_blocks_with_mode(
            logits.clone(),
            sm_coords,
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        assert!(
            softmax.ordered_beta_bernoulli_eb_log_alpha_step(&rho).unwrap().is_none(),
            "softmax sparsity prior has no EB α M-step"
        );

        // A pinned (overridden) α is not a free parameter → None.
        let mut pinned = learnable_ibp(logits, 1.0);
        pinned.set_ordered_beta_bernoulli_alpha_override(Some(3.0));
        assert!(
            pinned.ordered_beta_bernoulli_eb_log_alpha_step(&rho).unwrap().is_none(),
            "override-pinned α must not take the EB M-step"
        );
    }
}
