//! Assignment gates and sparsity-prior helpers for the SAE manifold term.
//! Mechanically split from `sae_manifold.rs`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::solver::evidence::{HybridAtomCandidate, HybridAtomChoice, select_hybrid_atom};
use crate::terms::analytic_penalties::{
    AnalyticPenalty, IBPAssignmentPenalty, IbpHessianDiagThirdChannels,
    SoftmaxAssignmentSparsityPenalty, resolve_learnable_weight,
};
use crate::terms::latent::{LatentCoordValues, LatentIdMode, LatentManifold};
use crate::terms::sae::manifold::SaeManifoldRho;

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

/// #976 Layer-1 guard: per-atom active-mass floor. The collapse statistic is
/// the atom's MAXIMUM assignment mass over rows, not its mean: a legitimately
/// sparse atom has a small mean but high mass on its own rows, while only an
/// atom with no material support anywhere — the #853 failure — has a small
/// max. An atom whose max mass falls below this floor is re-seeded (once) or
/// recorded as terminally collapsed; never a silent death, never a fit error.
pub(crate) const SAE_ATOM_ACTIVE_MASS_FLOOR: f64 = 1.0e-3;

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

/// #976 Layer-1 guard (simultaneous-collapse arm): the reconstruction
/// explained-variance below which a K>=2 dictionary is judged to have
/// CO-collapsed — every atom degenerate together, so the median-relative
/// [`SAE_ATOM_DECODER_NORM_COLLAPSE_RATIO`] test sees no atom "behind" its peers
/// and stays silent. This is the real-data K>=2 failure: atoms can split enough
/// signal to avoid a relative-norm breach while still under-recovering the
/// long-tailed target. The floor therefore lives above the committed
/// OLMo-mixed-layer held-out acceptance bar (0.5 × rank-8 PCA EV ≈ 0.27515),
/// so a partial co-collapse is re-diversified instead of being accepted as a
/// merely-difficult fit. K=1 returns before this guard, so the single-atom OLMo
/// path is unchanged. When tripped, the guard reseeds the dictionary onto
/// distinct residual PCs to break the shared basin.
pub(crate) const SAE_DICTIONARY_COLLAPSE_EV_FLOOR: f64 = 0.28;

/// #976 / #1117 K>1 robustness: bounded DICTIONARY-level multi-start budget for
/// the simultaneous co-collapse arm (the EV-floor branch of
/// [`crate::terms::sae::manifold::SaeManifoldTerm::enforce_decoder_norm_guard`]).
/// Distinct from the per-atom [`SAE_ATOM_COLLAPSE_RESEED_BUDGET`] (= 1): that
/// budget governs reseeding ONE atom's gate logits against an optimizer that
/// keeps killing it, where a loop would fight the optimizer. A co-collapse
/// reseed is categorically different — it is a full-dictionary multi-start that
/// re-diversifies ALL atoms onto distinct principal directions of a FRESHLY
/// recomputed residual, so successive attempts explore genuinely different
/// basins. A single such reseed empirically cannot always break a K≥3 three-way
/// basin (identical (K, seed) flips EV≈0.40 ↔ 0.00), so this arm gets a small
/// bounded budget of independent multi-starts. It is consumed ONLY when the
/// whole dictionary explains < [`SAE_DICTIONARY_COLLAPSE_EV_FLOOR`] of the
/// variance — a no-op for any healthy fit (real OLMo K=1 ~0.22, K=2 ~0.40).
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
    /// Deterministic concrete relaxation of a truncated IBP active set.
    IBPMap {
        temperature: f64,
        alpha: f64,
        learnable_alpha: bool,
    },
    /// Hard-thresholded bounded gate: each atom is off (gate = 0) when its logit
    /// is at or below `threshold`, and on with a threshold-centered shifted
    /// sigmoid `σ((logit − threshold) / temperature) ∈ [0.5, 1)` above it. This
    /// is NOT literal JumpReLU `z·1[z>θ]` — the gate carries no magnitude; it is
    /// a member of the gate family (softmax simplex / IBP sigmoid / this hard
    /// gate) and stays bounded in [0, 1]. Reconstruction magnitude lives entirely
    /// in the decoder curve `g_k(t) = φ(t)ᵀ B_k`. The discontinuity at `threshold`
    /// (0 → 0.5) is the intended "jump".
    JumpReLU { temperature: f64, threshold: f64 },
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

    #[must_use]
    pub fn jumprelu(temperature: f64, threshold: f64) -> Self {
        Self::JumpReLU {
            temperature,
            threshold,
        }
    }

    pub fn temperature(&self) -> f64 {
        match *self {
            AssignmentMode::Softmax { temperature, .. }
            | AssignmentMode::IBPMap { temperature, .. }
            | AssignmentMode::JumpReLU { temperature, .. } => temperature,
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
            | AssignmentMode::JumpReLU { temperature, .. } => {
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
            AssignmentMode::JumpReLU { threshold, .. } => {
                if !threshold.is_finite() {
                    return Err(format!(
                        "AssignmentMode::JumpReLU: threshold must be finite; got {threshold}"
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn resolved_ibp_alpha(&self, rho: &SaeManifoldRho) -> Option<f64> {
        match *self {
            AssignmentMode::IBPMap {
                alpha,
                learnable_alpha,
                ..
            } => Some(if learnable_alpha {
                resolve_learnable_weight(alpha, rho.log_lambda_sparse)
            } else {
                alpha
            }),
            _ => None,
        }
    }
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
        })
    }

    /// #1033 — install a ρ-INVARIANT FROZEN routing (the amortized predicted
    /// logits; see [`SaeAssignment::frozen_logits`]). `predicted` must be
    /// `(n, K)`. With routing frozen, the gates are computed from `predicted` and
    /// the logits are excluded from the inner Newton (their gradient/curvature are
    /// inert, like an ungated atom's). Passing `None` restores the free-logit
    /// path.
    #[must_use = "build error must be handled"]
    pub fn with_frozen_routing(
        mut self,
        predicted: Option<Array2<f64>>,
    ) -> Result<Self, String> {
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
            AssignmentMode::IBPMap { .. } | AssignmentMode::JumpReLU { .. } => self.k_atoms(),
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

    pub(crate) fn try_assignments_row_for_rho(
        &self,
        row: usize,
        rho: &SaeManifoldRho,
    ) -> Result<Array1<f64>, String> {
        self.try_assignments_row_with_alpha(row, self.mode.resolved_ibp_alpha(rho))
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
            AssignmentMode::JumpReLU {
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

pub(crate) fn sae_sigmoid_derivatives_from_value(
    value: f64,
    inv_tau: f64,
    scale: f64,
) -> (f64, f64, f64) {
    let sig = if scale > 0.0 { value / scale } else { 0.0 };
    let dz = scale * sig * (1.0 - sig) * inv_tau;
    let d2z = scale * sig * (1.0 - sig) * (1.0 - 2.0 * sig) * inv_tau * inv_tau;
    (value, dz, d2z)
}

pub(crate) fn neutral_gate_weights(mode: AssignmentMode, k_atoms: usize) -> Array1<f64> {
    match mode {
        AssignmentMode::Softmax { .. } => Array1::from_elem(k_atoms, 1.0 / (k_atoms.max(1) as f64)),
        AssignmentMode::IBPMap {
            temperature, alpha, ..
        } => ibp_map_row(Array1::<f64>::zeros(k_atoms).view(), temperature, alpha),
        AssignmentMode::JumpReLU { .. } => Array1::from_elem(k_atoms, 0.5),
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

/// IBP-MAP row activations: per-atom sigmoid likelihood times the truncated
/// stick-breaking prior mean `π_k = (α/(α+1))^{k+1}`. With tied logits the prior
/// dominates and yields strictly decreasing activations in atom index, with the
/// first atom already shrunk by one Beta(α,1) stick mean (no unshrunk base atom).
pub fn ibp_map_row(logits: ArrayView1<'_, f64>, temperature: f64, alpha: f64) -> Array1<f64> {
    let prior = ordered_geometric_shrinkage_prior(logits.len(), alpha);
    let mut out = Array1::<f64>::zeros(logits.len());
    for i in 0..logits.len() {
        out[i] = crate::linalg::utils::stable_logistic(logits[i] / temperature) * prior[i];
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
        let sig = crate::linalg::utils::stable_logistic(logits[i] * inv_tau);
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
            out[i] = crate::linalg::utils::stable_logistic((logits[i] - threshold) / temperature);
        }
    }
    out
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
        AssignmentMode::JumpReLU {
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
            let activation = crate::linalg::utils::stable_logistic((logit_k - threshold) * inv_tau);
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
        AssignmentMode::JumpReLU {
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
                let activation = crate::linalg::utils::stable_logistic(
                    (logits[logit_col] - threshold) * inv_tau,
                );
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

pub(crate) fn assignment_prior_value(assignment: &SaeAssignment, rho: &SaeManifoldRho) -> f64 {
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)
            .expect("assignment logits must be finite");
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
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
            temperature,
            alpha,
            learnable_alpha,
        } => {
            let mut penalty = IBPAssignmentPenalty::new(
                assignment.k_atoms(),
                alpha,
                temperature,
                learnable_alpha,
            );
            let rho_view = if learnable_alpha {
                Array1::from_vec(vec![rho.log_lambda_sparse])
            } else {
                // Keep the fixed-alpha value path on the same weighting branch as
                // assignment_prior_grad_hdiag; that gradient path owns the
                // lambda_sparse convention for IBP assignment sparsity.
                penalty.weight = rho.lambda_sparse();
                Array1::zeros(0)
            };
            penalty.value(target.view(), rho_view.view())
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            // Sparsity penalty uses the same threshold-centered surrogate and
            // machine-precision support as its gradient/Hessian. Data-fit
            // reconstruction remains hard-gated by `jumprelu_row`.
            let sparsity_strength = rho.lambda_sparse();
            let mut acc = 0.0;
            for &logit in target.iter() {
                if jumprelu_in_optimization_band(logit, threshold, temperature) {
                    acc += crate::linalg::utils::stable_logistic((logit - threshold) / temperature);
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
    match assignment.mode {
        AssignmentMode::Softmax { .. } | AssignmentMode::JumpReLU { .. } => {
            assignment_prior_value(assignment, rho)
        }
        AssignmentMode::IBPMap {
            temperature,
            alpha,
            learnable_alpha,
        } => {
            let mut penalty = IBPAssignmentPenalty::new(
                assignment.k_atoms(),
                alpha,
                temperature,
                learnable_alpha,
            );
            if learnable_alpha {
                let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse]);
                penalty.grad_rho(target.view(), rho_view.view())[0]
            } else {
                penalty.weight = rho.lambda_sparse();
                penalty.value(target.view(), Array1::<f64>::zeros(0).view())
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
    match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| {
                    "softmax assignment log-strength hessian diag unavailable".to_string()
                })
        }
        AssignmentMode::JumpReLU {
            temperature,
            threshold,
        } => {
            let sparsity_strength = rho.lambda_sparse();
            let inv_tau = 1.0 / temperature;
            let inv_tau2 = inv_tau * inv_tau;
            let mut d = Array1::<f64>::zeros(target.len());
            for idx in 0..target.len() {
                let logit = target[idx];
                if !jumprelu_in_optimization_band(logit, threshold, temperature) {
                    continue;
                }
                let activation =
                    crate::linalg::utils::stable_logistic((logit - threshold) * inv_tau);
                let slope = activation * (1.0 - activation);
                d[idx] = sparsity_strength * slope * (1.0 - 2.0 * activation) * inv_tau2;
            }
            Ok(d)
        }
        AssignmentMode::IBPMap {
            temperature,
            alpha,
            learnable_alpha,
        } => {
            let mut penalty = IBPAssignmentPenalty::new(
                assignment.k_atoms(),
                alpha,
                temperature,
                learnable_alpha,
            );
            if learnable_alpha {
                let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse]);
                Ok(penalty.hessian_diag_log_alpha_derivative(target.view(), rho_view.view()))
            } else {
                penalty.weight = rho.lambda_sparse();
                penalty
                    .hessian_diag(target.view(), Array1::<f64>::zeros(0).view())
                    .ok_or_else(|| {
                        "IBP assignment log-strength hessian diag unavailable".to_string()
                    })
            }
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
    match assignment.mode {
        AssignmentMode::IBPMap {
            temperature,
            alpha,
            learnable_alpha: true,
        } => {
            let penalty = IBPAssignmentPenalty::new(assignment.k_atoms(), alpha, temperature, true);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse]);
            Ok(penalty.log_alpha_target_mixed_derivative(target.view(), rho_view.view()))
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
            temperature,
            alpha,
            learnable_alpha,
        } => {
            // Scale the IBP assignment-sparsity prior by `lambda_sparse`, exactly
            // like the Softmax and JumpReLU branches do (Softmax folds it into the
            // penalty's rho coordinate, JumpReLU multiplies `sparsity_strength`).
            // Previously the IBP penalty used its hardcoded `weight = 1.0` and the
            // `rho.log_lambda_sparse` coordinate never reached it (the rho_view was
            // empty for the common `learnable_alpha = false` config), so the prior
            // ran at full strength with no way to dial it down — and its
            // Beta-Bernoulli BCE energy `−mass·ln π_k − (n−mass)·ln(1−π_k)` toward
            // the self-referential empirical active fraction `π_k` has its global
            // minimum at the all-off gate, so at full weight it over-shrank the
            // assignment off both atoms even with a truth-seeded decoder (#853).
            // Routing `lambda_sparse` into the penalty weight makes the prior a
            // genuine, user-controllable lever balanced against the data fit.
            let mut penalty = IBPAssignmentPenalty::new(
                assignment.k_atoms(),
                alpha,
                temperature,
                learnable_alpha,
            );
            // When `alpha` is learnable, `log_lambda_sparse` already modulates
            // it through `resolved_alpha(rho)`, so the weight stays 1.0 to avoid
            // double-counting that coordinate. Only when `alpha` is fixed (so the
            // sparse coordinate would otherwise be ignored entirely) does
            // `lambda_sparse` become the prior's weight lever.
            let rho_view = if learnable_alpha {
                Array1::from_vec(vec![rho.log_lambda_sparse])
            } else {
                penalty.weight = rho.lambda_sparse();
                Array1::zeros(0)
            };
            let g = penalty.grad_target(target.view(), rho_view.view());
            let d = penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| "IBP assignment hessian diag unavailable".to_string())?;
            (g, d)
        }
        AssignmentMode::JumpReLU {
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
                let activation =
                    crate::linalg::utils::stable_logistic((logit - threshold) * inv_tau);
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
) -> Result<Option<IbpHessianDiagThirdChannels>, String> {
    let AssignmentMode::IBPMap {
        temperature,
        alpha,
        learnable_alpha,
    } = assignment.mode
    else {
        return Ok(None);
    };
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    let mut penalty =
        IBPAssignmentPenalty::new(assignment.k_atoms(), alpha, temperature, learnable_alpha);
    // Mirror assignment_prior_grad_hdiag exactly: when alpha is learnable the
    // sparse coordinate already modulates it through resolved_alpha(rho), so the
    // weight stays 1.0; otherwise lambda_sparse becomes the prior's weight lever.
    let rho_view = if learnable_alpha {
        Array1::from_vec(vec![rho.log_lambda_sparse])
    } else {
        penalty.weight = rho.lambda_sparse();
        Array1::zeros(0)
    };
    let mut channels =
        penalty.hessian_diag_logit_third_channels(target.view(), rho_view.view());
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
/// [`crate::terms::sae::chart_canonicalization::d1_atom_fitted_turning`]) enters
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
    use crate::solver::evidence::HybridAtomParam;

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
        let logits = Array2::from_shape_fn((n, k), |(i, kk)| 0.3 + 0.05 * (i as f64) - 0.1 * (kk as f64));
        let coords: Vec<Array2<f64>> =
            (0..k).map(|_| Array2::from_shape_fn((n, 1), |(i, _)| (i as f64) * 0.1)).collect();
        // learnable_alpha = false: alpha is ρ-independent, isolating the routing.
        SaeAssignment::from_blocks_with_mode(logits, coords, AssignmentMode::ibp_map(0.5, 1.0, false))
            .unwrap()
    }

    #[test]
    fn frozen_routing_decouples_gates_from_logit_updates_1033() {
        let (n, k) = (6usize, 3usize);
        let mut a = ibp_assignment(n, k).freeze_routing_from_current_logits().unwrap();
        assert!(a.routing_is_frozen());
        // Gates BEFORE mutating the free logits.
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k]);
        let before: Vec<Array1<f64>> =
            (0..n).map(|r| a.try_assignments_row_for_rho(r, &rho).unwrap()).collect();
        // Simulate an inner-fit logit update (what the ρ-search would otherwise do
        // every eval): perturb every free logit substantially.
        a.logits.mapv_inplace(|v| v + 5.0);
        let after: Vec<Array1<f64>> =
            (0..n).map(|r| a.try_assignments_row_for_rho(r, &rho).unwrap()).collect();
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
        let a = ibp_assignment(n, k).freeze_routing_from_current_logits().unwrap();
        // Two different ρ (different sparse + smooth strengths). With frozen routing
        // and learnable_alpha=false, the gate value must be identical at both ρ.
        let rho_a = SaeManifoldRho::new((1e-3_f64).ln(), (1e-2_f64).ln(), vec![Array1::<f64>::zeros(1); k]);
        let rho_b = SaeManifoldRho::new((1e3_f64).ln(), (1e1_f64).ln(), vec![Array1::<f64>::zeros(1); k]);
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
        let mut a = ibp_assignment(n, k).freeze_routing_from_current_logits().unwrap();
        // Under frozen routing EVERY logit is fixed (not a free Newton coord).
        let mask = a.fixed_logit_mask();
        assert_eq!(mask.len(), k);
        assert!(mask.iter().all(|&f| f), "frozen routing must fix ALL logits");
        for kk in 0..k {
            assert!(a.logit_is_fixed(kk), "atom {kk} logit must be fixed under frozen routing");
        }
        // Thawing restores the free-logit path (no fixed logits, no ungated).
        a.thaw_routing();
        assert!(!a.routing_is_frozen());
        assert!(a.fixed_logit_mask().iter().all(|&f| !f), "thaw must restore the free-logit path");
    }

    #[test]
    fn frozen_routing_rejects_softmax_1033() {
        let (n, k) = (4usize, 3usize);
        let logits = Array2::from_shape_fn((n, k), |(i, kk)| 0.1 * (i as f64) - 0.05 * (kk as f64));
        let coords: Vec<Array2<f64>> =
            (0..k).map(|_| Array2::from_shape_fn((n, 1), |(i, _)| (i as f64) * 0.1)).collect();
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
