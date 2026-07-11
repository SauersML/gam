//! Assignment gates and sparsity-prior helpers for the SAE manifold term.
//! Mechanically split from `sae_manifold.rs`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::SaeManifoldRho;
use gam_solve::evidence::{HybridAtomCandidate, HybridAtomChoice, select_hybrid_atom};
use gam_terms::analytic_penalties::{
    AnalyticPenalty, OrderedBetaBernoulliHessianDiagThirdChannels, OrderedBetaBernoulliPenalty,
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
    /// Deterministic sigmoid relaxation for an ordered independent
    /// Beta--Bernoulli active set:
    /// `a_k = σ(logit_k/temperature)`. These are independent Bernoulli gates,
    /// not mixture/simplex responsibilities. The ordered geometric mean schedule
    /// `π_k = (α/(α+1))^{k+1}` is scored once by the ordered Beta--Bernoulli prior; it is not
    /// multiplied into the final reconstructed function.
    OrderedBetaBernoulli {
        temperature: f64,
        alpha: f64,
        learnable_alpha: bool,
    },
    /// Smooth threshold-centered logistic gate
    /// `a_k = σ((logit_k − threshold) / temperature)`. Magnitude lives in the
    /// decoder curve `g_k(t) = φ(t)ᵀB_k`; this gate supplies a bounded
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
    /// outer ρ search. This is deterministic fixed-cardinality support, not a
    /// probabilistic prior or a MAP approximation to one. The gate is per-row
    /// independent (couples rows through NOTHING), so fits stream
    /// chunk-invariantly at any K, and it is exchangeable across atom index.
    TopK { k: usize },
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

    /// Construct the smooth threshold-centered logistic [`Self::ThresholdGate`].
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

    /// Resolve the effective ordered independent Beta--Bernoulli concentration `α` for this mode.
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
/// derived by row-wise softmax, independent ordered Beta--Bernoulli sigmoid active indicators,
/// or threshold gate gates. Softmax logits are canonicalized to the reference chart
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
    /// ordered Beta--Bernoulli / threshold gate modes the remaining atoms are computed independently, so
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
                     contract supports ordered Beta--Bernoulli and threshold gate, whose per-atom gates have no \
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
                 (coupled-simplex entropy-majorizer); use ordered Beta--Bernoulli or threshold gate"
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
                 rejected (coupled-simplex entropy-majorizer); use ordered Beta--Bernoulli or threshold gate"
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
    /// Ungating is defined for the COLUMN-SEPARABLE gate modes (ordered Beta--Bernoulli and
    /// threshold gate): each atom's gate is an independent per-atom function of its own
    /// logit, so pinning one atom to `a_k ≡ 1` leaves every other atom's gate
    /// exactly as computed. Softmax is a coupled simplex (`Σ_k a_k = 1` over all
    /// `K`), so a unit gate for one atom is only well defined relative to a
    /// gated-subset renormalization that must also be reflected in the logit-JVP
    /// and the entropy majorizer; this constructor's contract is restricted to
    /// the separable modes, and an ungated atom under Softmax is REJECTED here so
    /// the inner solve never runs on a value/gradient-mismatched gate. Callers
    /// wanting a dense background tier under Softmax route it as an ordered Beta--Bernoulli or
    /// threshold gate atom.
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
                 contract does not perform; route a dense background tier as ordered Beta--Bernoulli or threshold gate"
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
            AssignmentMode::OrderedBetaBernoulli { .. } | AssignmentMode::ThresholdGate { .. } => {
                self.k_atoms()
            }
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

    /// #1777 — the effective ordered independent Beta--Bernoulli `α` for this assignment at `rho`,
    /// honoring the PER-FIT [`Self::ordered_beta_bernoulli_alpha_override`] before the mode's
    /// canonical value or learnable schedule. The single seam every
    /// gate/jet/prior site reads so the per-fit override is applied consistently.
    /// `None` for non-ordered Beta--Bernoulli modes.
    pub(crate) fn resolved_ordered_beta_bernoulli_alpha(
        &self,
        rho: &SaeManifoldRho,
    ) -> Option<f64> {
        self.mode
            .resolved_ordered_beta_bernoulli_alpha(rho, self.ordered_beta_bernoulli_alpha_override)
    }

    /// Whether the ordered independent Beta--Bernoulli concentration α is a FREE outer parameter that
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
        // threshold gate keep a free per-atom gate logit even at K==1
        // (assignment_coord_dim = K = 1), so they must fall through to their real
        // row functions or the logit would move the prior but not the gate.
        if self.k_atoms() == 1 && matches!(self.mode, AssignmentMode::Softmax { .. }) {
            return Ok(Array1::from_vec(vec![1.0]));
        }
        let mut row_gates = match self.mode {
            AssignmentMode::Softmax { temperature, .. } => softmax_row(routing, temperature),
            AssignmentMode::OrderedBetaBernoulli { temperature, .. } => {
                ordered_beta_bernoulli_row(routing, temperature)
            }
            AssignmentMode::ThresholdGate {
                temperature,
                threshold,
            } => threshold_gate_row(routing, temperature, threshold),
            AssignmentMode::TopK { k } => topk_row(routing, k),
        };
        // #1026 — ungated (background-tier) atoms have a fixed unit gate. For the
        // column-separable ordered Beta--Bernoulli / threshold gate modes the other atoms' gates are
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

    pub(crate) fn persist_resolved_ordered_beta_bernoulli_alpha(
        &mut self,
        rho: &SaeManifoldRho,
    ) -> bool {
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

/// #1784 — K-aware default ordered Beta--Bernoulli concentration.
///
/// The independent-Beta prior-mean schedule `μ_k = (α/(α+1))^{k+1}` decays
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
/// prior monotone (no atom is structurally masked). Solving
/// `(α/(α+1))^K = e^{-1}` gives
/// `α = 1/(exp(1/K) − 1) ≈ K − 1/2`. Floored at `1.0` so `K = 1` keeps the
/// historical `α = 1`.
pub fn default_ordered_beta_bernoulli_concentration_for_k_atoms(k_atoms: usize) -> f64 {
    let k = k_atoms.max(1) as f64;
    // π_{K-1} = (α/(α+1))^K = e^{-1}  ⇒  α = 1/(e^{1/K} − 1).
    let alpha = 1.0 / ((1.0 / k).exp() - 1.0);
    alpha.max(1.0)
}

/// Sigmoid activations for the ordered Beta--Bernoulli assignment model.
///
/// Ordered shrinkage belongs to the Beta--Bernoulli prior scored by
/// [`OrderedBetaBernoulliPenalty`], not as a second multiplicative factor on the final
/// reconstruction. Multiplying by the prior mean capped atom `k` at `mu_k < 1`
/// even when its learned gate approached one, double-counted the prior and made
/// the fitted function depend on atom index. The reconstruction gate is simply
/// `sigmoid(logit_k / temperature)`.
pub fn ordered_beta_bernoulli_row(logits: ArrayView1<'_, f64>, temperature: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        out[i] = gam_linalg::utils::stable_logistic(logits[i] / temperature);
    }
    out
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

/// Exact numerical inverse of the softplus link `softplus(x) = log(1 + eˣ)`
/// (the forward direction is [`gam_linalg::utils::stable_softplus`], used by
/// the penalty implementations). This is the single source of truth for the
/// softplus⁻¹ reparameterization the SAE penalty FFI uses to map
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

// #1557 — fill-into-caller-buffer variants of the three per-mode row functions.
// These compute the EXACT SAME values as `softmax_row` / `ordered_beta_bernoulli_row` /
// `threshold_gate_row` (same arithmetic, same order of operations) but write into a
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

pub(crate) fn ordered_beta_bernoulli_row_into(
    logits: ArrayView1<'_, f64>,
    temperature: f64,
    out: &mut [f64],
) {
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
            // Posterior-mean Bernoulli gate `z_k = σ(l_k/τ)`; independent-Beta
            // shrinkage is scored once, in the ordered Beta--Bernoulli prior.
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
        assignment
            .resolved_ordered_beta_bernoulli_alpha(rho)
            .unwrap_or(base_alpha)
    };
    // #991 design-honesty weights: the ordered Beta--Bernoulli prior is not row-separable (the
    // exact integrated scalar couples rows through the column active mass), so the weights are
    // installed ON the penalty — its value/grad/hessian/hvp/ρ- and third
    // channels all fold them identically (weighted mass `M_k = Σ w_i z_ik` and
    // active-mass Jacobian `u = w·J`), keeping every channel the exact derivative of one
    // weighted energy. `None` gives the unit-weight operator.
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

/// Apply the exact ordered Beta--Bernoulli logit Hessian minus the diagonal
/// PSD majorizer installed in the Newton/Laplace operator.
///
/// The exact integrated marginal contributes a dense-within-column Hessian:
/// a negative rank-one active-mass term plus a row-local concrete-Jacobian
/// diagonal. The assembled operator keeps only the positive part of that
/// diagonal, because zero is a PSD Loewner majorizer of the negative rank-one
/// term. The stationarity IFT must nevertheless invert the exact scalar
/// Hessian, so `A - B` is applied here analytically and matrix-free. No dense
/// `N K × N K` matrix or persistent low-rank carrier is constructed.
pub(crate) fn ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
    direction: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let AssignmentMode::OrderedBetaBernoulli {
        temperature, alpha, ..
    } = assignment.mode
    else {
        return Err(
            "ordered Beta--Bernoulli exact-Hessian correction requires ordered assignment mode"
                .to_string(),
        );
    };
    let target = flat_logits(assignment.logits.view());
    if direction.len() != target.len() {
        return Err(format!(
            "ordered Beta--Bernoulli exact-Hessian direction has length {}; expected {}",
            direction.len(),
            target.len()
        ));
    }
    if !direction.iter().all(|value| value.is_finite()) {
        return Err("ordered Beta--Bernoulli exact-Hessian direction must be finite".to_string());
    }
    if assignment.routing_is_frozen() {
        return Ok(Array1::<f64>::zeros(target.len()));
    }
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }

    let (penalty, rho_view) =
        ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights);
    let mut delta = penalty.hvp(target.view(), rho_view.view(), direction);
    let channels = penalty.psd_majorizer_logit_third_channels(target.view(), rho_view.view());
    for index in 0..delta.len() {
        delta[index] -= channels.diagonal_term[index].max(0.0) * direction[index];
    }
    Ok(delta)
}

#[cfg(test)]
mod ordered_beta_bernoulli_exact_hessian_tests {
    use super::*;

    #[test]
    fn exact_hessian_minus_majorizer_hvp_matches_gradient_fd_and_keeps_cross_row_term() {
        let n = 4usize;
        let k = 2usize;
        let logits =
            Array2::from_shape_vec((n, k), vec![0.2, -0.3, 0.7, -0.1, 0.4, 0.5, -0.2, 0.6])
                .unwrap();
        let coords = vec![Array2::<f64>::zeros((n, 1)); k];
        let assignment = SaeAssignment::from_blocks_with_mode(
            logits,
            coords,
            AssignmentMode::ordered_beta_bernoulli(0.8, 1.7, false),
        )
        .unwrap();
        let rho = SaeManifoldRho::new(1.3_f64.ln(), 0.0, vec![Array1::zeros(1); k]);
        // Excite one logit only. The exact integrated marginal must still
        // produce nonzero output on other rows of the same atom column.
        let mut direction = Array1::<f64>::zeros(n * k);
        direction[0] = 0.7;
        let analytic = ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
            &assignment,
            &rho,
            None,
            direction.view(),
        )
        .unwrap();

        let (penalty, rho_view) =
            ordered_beta_bernoulli_prior_penalty(&assignment, &rho, 1.7, 0.8, None);
        let target = flat_logits(assignment.logits.view());
        let step = 1.0e-6;
        let plus = &target + &(step * &direction);
        let minus = &target - &(step * &direction);
        let gradient_plus = penalty.grad_target(plus.view(), rho_view.view());
        let gradient_minus = penalty.grad_target(minus.view(), rho_view.view());
        let channels = penalty.psd_majorizer_logit_third_channels(target.view(), rho_view.view());
        for index in 0..analytic.len() {
            let exact_fd = (gradient_plus[index] - gradient_minus[index]) / (2.0 * step);
            let expected = exact_fd - channels.diagonal_term[index].max(0.0) * direction[index];
            assert!(
                (analytic[index] - expected).abs() <= 2.0e-7,
                "index {index}: analytic A-B={} expected={} exact_fd={exact_fd}",
                analytic[index],
                expected,
            );
        }
        assert!(
            analytic[2].abs() > 1.0e-6 && analytic[4].abs() > 1.0e-6,
            "a one-row direction must produce the exact cross-row rank-one action: {analytic:?}"
        );
    }
}

pub fn assignment_prior_value(assignment: &SaeAssignment, rho: &SaeManifoldRho) -> f64 {
    assignment_prior_value_weighted(assignment, rho, None)
}

/// As [`assignment_prior_value`], but with #991 design-honesty per-row weights:
/// row `i`'s per-row prior contribution is scaled by `w_i` (mean-1). This is the
/// per-row latent prior's analog of the `√w_i`-weighted data likelihood and the
/// `w_i`-weighted `ard_value` — each retained row of a design-honest subsample
/// stands in for `w_i` population rows, so its routing prior carries `w_i` too.
/// `None` gives the unit-weight path. Softmax/threshold gate are row-separable;
/// ordered Beta--Bernoulli instead forms weighted active mass and effective row
/// count inside its integrated scalar. Every derivative uses that same measure.
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
            let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
                assignment,
                rho,
                alpha,
                temperature,
                row_weights,
            );
            penalty.value(target.view(), rho_view.view())
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            // Sparsity penalty and reconstruction use the same smooth
            // threshold-centered logistic gate as the gradient and Hessian.
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
                acc +=
                    w_row * gam_linalg::utils::stable_logistic((logit - threshold) / temperature);
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

/// #991-weighted [`assignment_prior_log_strength_derivative`]. Every assignment
/// mode differentiates the same weighted scalar used by its value path.
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
            let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
                assignment,
                rho,
                alpha,
                temperature,
                row_weights,
            );
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

/// #991-weighted [`assignment_prior_log_strength_hdiag`]. Every assignment mode
/// differentiates the same weighted scalar used by its value path.
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
            let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
                assignment,
                rho,
                alpha,
                temperature,
                row_weights,
            );
            let mut d = if penalty.learnable_alpha {
                penalty.hessian_diag_log_alpha_derivative(target.view(), rho_view.view())
            } else {
                penalty
                    .hessian_diag(target.view(), rho_view.view())
                    .ok_or_else(|| {
                        "ordered Beta--Bernoulli assignment log-strength hessian diag unavailable"
                            .to_string()
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
/// uses the same weighted active mass as the value, gradient, and Hessian.
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
            let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
                assignment,
                rho,
                alpha,
                temperature,
                row_weights,
            );
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
/// design weight `w_i`. Softmax, threshold gate, and ordered Beta--Bernoulli
/// modes all use the same row weights in value, gradient, curvature, and outer
/// concentration derivatives.
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
            // threshold gate multiplies `sparsity_strength`). #Bug6: `ordered_beta_bernoulli_prior_penalty`
            // picks the EFFECTIVE-α learnability — an override pins α so the prior
            // uses the fixed-α weight convention and the resolved (override) α,
            // matching the forward gate — and installs the #Bug4 ungated mask. The
            // per-atom fixed-logit columns are additionally zeroed post-hoc below,
            // so the array (grad/hessian) methods need no internal column mask.
            let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
                assignment,
                rho,
                alpha,
                temperature,
                row_weights,
            );
            let g = penalty.grad_target(target.view(), rho_view.view());
            let d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| {
                    "ordered Beta--Bernoulli assignment hessian diag unavailable".to_string()
                })?;
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
    // matching the zero logit-JVP. The column-separable ordered Beta--Bernoulli / threshold gate priors are
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

/// Build exact derivatives of the ordered Beta--Bernoulli PSD curvature
/// majorizer for the SAE log-det adjoint Γ, using the same penalty configuration —
/// `alpha`/`tau`/`learnable_alpha` and the `lambda_sparse` weight convention —
/// that [`assignment_prior_grad_hdiag`] assembles into `htt`. Returns `None`
/// for other assignment modes.
pub fn ordered_beta_bernoulli_psd_majorizer_third_channels(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
) -> Result<Option<OrderedBetaBernoulliHessianDiagThirdChannels>, String> {
    ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(assignment, rho, None)
}

/// As [`ordered_beta_bernoulli_psd_majorizer_third_channels`], with the #991 design-honesty per-row
/// weights the assembled `htt` carried (the channels must differentiate the
/// same weighted operator; `z_jac` carries the weighted active-mass derivative
/// `u = w·J`).
pub(crate) fn ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
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
    let (penalty, rho_view) =
        ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights);
    let mut channels = penalty.psd_majorizer_logit_third_channels(target.view(), rho_view.view());
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
                channels.diagonal_term[idx] = 0.0;
            }
        }
        for atom in 0..k {
            if assignment.logit_is_fixed(atom) {
                channels.mass_hessian_coefficient[atom] = 0.0;
                channels.mass_hessian_log_alpha_derivative[atom] = 0.0;
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
    //! mode (Softmax, OrderedBetaBernoulli, threshold gate), the #1026 ungated case, and the K==1
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
            // Pre-fill with NaN so a partial write (e.g. a threshold gate below-threshold
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
        assert_into_matches_alloc(&build(
            7,
            5,
            AssignmentMode::ordered_beta_bernoulli(0.6, 1.3, false),
        ));
        assert_into_matches_alloc(&build(
            7,
            5,
            AssignmentMode::ordered_beta_bernoulli(0.6, 1.3, true),
        ));
    }

    #[test]
    fn threshold_gate_into_is_bit_identical() {
        // Threshold chosen so SOME atoms fall below it (the untouched-entry path)
        // and some clear it (the sigmoid path) — both branches are exercised.
        assert_into_matches_alloc(&build(7, 5, AssignmentMode::threshold_gate(0.9, 0.2)));
    }

    #[test]
    fn ungated_into_is_bit_identical() {
        // #1026 ungated overwrite under a gate-style mode (ordered Beta--Bernoulli/threshold gate allow it).
        let a = build(
            6,
            4,
            AssignmentMode::ordered_beta_bernoulli(0.6, 1.1, false),
        )
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
        // Softmax K==1 hits the fixed-unit early return; ordered Beta--Bernoulli/threshold gate K==1 keep a
        // free per-atom gate and fall through to the real row functions.
        assert_into_matches_alloc(&build(5, 1, AssignmentMode::softmax(1.0)));
        assert_into_matches_alloc(&build(
            5,
            1,
            AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
        ));
        assert_into_matches_alloc(&build(5, 1, AssignmentMode::threshold_gate(0.8, 0.1)));
    }
}
