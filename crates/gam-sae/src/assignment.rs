//! Assignment gates and sparsity-prior helpers for the SAE manifold term.
//! Mechanically split from `sae_manifold.rs`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::manifold::{AssignmentStrengthLayout, SaeManifoldRho};
use gam_terms::analytic_penalties::{
    AnalyticPenalty, OrderedBetaBernoulliHessianDiagThirdChannels,
    OrderedBetaBernoulliLogitAdjointData, OrderedBetaBernoulliPenalty,
    SoftmaxAssignmentSparsityPenalty, resolve_learnable_weight, softmax_entropy_log_partition,
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
    pub(crate) fn from_assignment(assignment: &SaeAssignment, atom_idx: usize) -> Result<Self, String> {
        let assignments = assignment.assignments();
        Self::from_assignment_matrix(assignments.view(), atom_idx)
    }

    #[must_use = "support construction error must be handled"]
    pub(crate) fn from_assignment_matrix(
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
    pub(crate) fn from_weights(atom_idx: usize, weights: Array1<f64>) -> Result<Self, String> {
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
/// the simultaneous co-collapse arm of
/// [`crate::manifold::SaeManifoldTerm::enforce_decoder_norm_guard`]).
/// Distinct from the per-atom [`SAE_ATOM_COLLAPSE_RESEED_BUDGET`] (= 1): that
/// budget governs reseeding ONE atom's gate logits against an optimizer that
/// keeps killing it, where a loop would fight the optimizer. A co-collapse
/// reseed is categorically different — it is a full-dictionary multi-start that
/// re-diversifies ALL atoms onto distinct principal directions of a FRESHLY
/// recomputed residual, so successive attempts explore genuinely different
/// basins. A single such reseed empirically cannot always break a K≥3 three-way
/// basin (identical (K, seed) flips EV≈0.40 ↔ 0.00), so this arm gets a small
/// bounded budget of independent multi-starts. It is consumed only after
/// iteration zero when the same-state certificate proves that all gated decoder
/// signals disappeared at floating-point resolution or #2362 proves structural
/// union-output-span collapse. Training EV is telemetry, so a healthy or merely
/// uncompetitive live-decoder fit never consumes this budget.
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

impl AssignmentMode {
    /// The family's stable snake_case name, for refusal text and provenance.
    /// Matching on the enum here means a family added later cannot be reported
    /// as an anonymous "non-softmax prior" by any message that uses this.
    #[must_use]
    pub fn family_label(&self) -> &'static str {
        match self {
            Self::Softmax { .. } => "softmax",
            Self::OrderedBetaBernoulli { .. } => "ordered_beta_bernoulli",
            Self::ThresholdGate { .. } => "threshold_gate",
            Self::TopK { .. } => "topk",
        }
    }

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
}

/// The ordered Beta--Bernoulli prior at one `rho` is the complete prior `P(concentration)`
/// at weight one, with its partition; see
/// [`SaeAssignment::ordered_beta_bernoulli_prior_parameters`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OrderedBetaBernoulliPriorParameters {
    pub(crate) concentration: f64,
    /// Whether `rho.log_lambda_sparse` is the outer coordinate `log(concentration / α_mode)`.
    /// Otherwise the concentration is fixed and no coordinate enters the prior.
    pub(crate) concentration_is_learnable: bool,
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
#[derive(Debug, Clone)]
pub struct SaeAssignment {
    pub logits: Array2<f64>,
    pub coords: Vec<LatentCoordValues>,
    pub mode: AssignmentMode,
    /// #1033 — AMORTIZED / FROZEN routing. When `Some`, this `(n, K)` matrix is a
    /// ρ-INVARIANT predicted routing (the amortized `x → logits` map distilled
    /// from the frozen dictionary): the gates are computed from THESE logits
    /// instead of the free `self.logits`, and the logits are NOT optimized by the
    /// inner Newton (their gradient/curvature/prior contributions are zeroed). It
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
    pub(crate) fn with_mode(
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
            frozen_logits: None,
        })
    }

    /// Whether the per-row routing is FROZEN (amortized) rather than free-logit.
    pub(crate) fn routing_is_frozen(&self) -> bool {
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

    /// Whether the routing logits are held fixed rather than free Newton parameters. Under
    /// TopK the support is a deterministic function of the routing logits
    /// (`assignment_coord_dim` is 0), and under frozen routing (#1033) the gates come from
    /// the predicted logits. Either way every logit is inert together — zero logit-JVP, zero
    /// sparsity-prior gradient and curvature, zero softmax majorizer — so no slot ever moves.
    pub(crate) fn logits_are_fixed(&self) -> bool {
        matches!(self.mode, AssignmentMode::TopK { .. }) || self.routing_is_frozen()
    }

    pub fn n_obs(&self) -> usize {
        self.logits.nrows()
    }

    pub fn k_atoms(&self) -> usize {
        self.logits.ncols()
    }

    pub(crate) fn total_coord_dim(&self) -> usize {
        self.coords.iter().map(|c| c.latent_dim()).sum()
    }

    pub(crate) fn assignment_coord_dim(&self) -> usize {
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

    pub(crate) fn coord_offsets(&self) -> Vec<usize> {
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

    pub(crate) fn assignments_row(&self, row: usize) -> Array1<f64> {
        self.try_assignments_row(row)
            .expect("assignment logits must be finite")
    }

    pub fn try_assignments_row(&self, row: usize) -> Result<Array1<f64>, String> {
        self.try_assignments_row_inner(row)
    }

    /// Whether the ordered independent Beta--Bernoulli concentration α is a FREE outer parameter that
    /// varies with ρ (`rho.log_lambda_sparse`), exactly when the mode requests it. A fixed α has
    /// identically zero ρ-derivatives, and every prior / log-det / IFT term treats it as a
    /// constant to stay consistent with the forward gate. `false` for non-ordered
    /// Beta--Bernoulli modes. (#Bug6)
    pub(crate) fn effective_alpha_is_learnable(&self) -> bool {
        matches!(
            self.mode,
            AssignmentMode::OrderedBetaBernoulli {
                learnable_alpha: true,
                ..
            }
        )
    }

    /// The ordered Beta--Bernoulli prior this assignment scores at `rho`: the complete
    /// prior `P(concentration)` at weight one.
    ///
    /// While the concentration is effectively learnable ([`Self::effective_alpha_is_learnable`])
    /// `rho.log_lambda_sparse` is `log(concentration / α_mode)`. Otherwise the concentration is
    /// fixed at the mode's `α`, the stored coordinate is
    /// an inert placeholder, and the rho layout carries no sparse coordinate. There is no
    /// tempering strength: `λ·L` with `λ ≠ 1` has a partition over the relaxed gates that does
    /// not reduce to the one-dimensional rate integral, and sparsity is already tuned through
    /// the concentration (#2933 F45). `None` for other modes.
    pub(crate) fn ordered_beta_bernoulli_prior_parameters(
        &self,
        rho: &SaeManifoldRho,
    ) -> Result<Option<OrderedBetaBernoulliPriorParameters>, String> {
        let AssignmentMode::OrderedBetaBernoulli { alpha, .. } = self.mode else {
            return Ok(None);
        };
        let parameters = if self.effective_alpha_is_learnable() {
            OrderedBetaBernoulliPriorParameters {
                concentration: resolve_learnable_weight(alpha, rho.log_lambda_sparse).map_err(
                    |error| format!("ordered Beta--Bernoulli learnable concentration: {error}"),
                )?,
                concentration_is_learnable: true,
            }
        } else {
            OrderedBetaBernoulliPriorParameters {
                concentration: alpha,
                concentration_is_learnable: false,
            }
        };
        Ok(Some(parameters))
    }

    pub(crate) fn validate_rho_domain(&self, rho: &SaeManifoldRho) -> Result<(), String> {
        self.validate_rho_layout(rho)?;
        rho.validate_log_strength_domain()?;
        if let AssignmentMode::OrderedBetaBernoulli { alpha, .. } = self.mode
            && self.effective_alpha_is_learnable()
        {
            resolve_learnable_weight(alpha, rho.log_lambda_sparse).map_err(|error| {
                format!("ordered Beta--Bernoulli learnable concentration: {error}")
            })?;
        }
        Ok(())
    }

    /// Refuse a rho whose ordered Beta--Bernoulli layout contradicts this assignment
    /// (#2933 F45). A layout bound for a learnable concentration carries a sparse coordinate
    /// that a fixed-concentration assignment does not have (and the reverse), so the flat length
    /// the rho was built with does not describe this prior. Rebinding here would change that
    /// length mid-objective, so the contradiction is an error. An unbound rho
    /// ([`AssignmentStrengthLayout::PenaltyWeight`], the constructors' tag) is not an ordered
    /// Beta--Bernoulli layout and is admitted: its coordinate is the concentration offset while
    /// the concentration is learnable, and no fixed-concentration channel reads it.
    pub(crate) fn validate_rho_layout(&self, rho: &SaeManifoldRho) -> Result<(), String> {
        let bound = rho.assignment_strength_layout;
        let expected = match self.mode {
            AssignmentMode::OrderedBetaBernoulli { .. } if self.effective_alpha_is_learnable() => {
                AssignmentStrengthLayout::ConcentrationOffset
            }
            AssignmentMode::OrderedBetaBernoulli { .. } => {
                AssignmentStrengthLayout::FixedConcentration
            }
            _ => {
                return if matches!(
                    bound,
                    AssignmentStrengthLayout::ConcentrationOffset
                        | AssignmentStrengthLayout::FixedConcentration
                ) {
                    Err(format!(
                        "rho layout {bound:?} was bound for an ordered Beta--Bernoulli prior, \
                         but the assignment is {} (#2933 F45)",
                        self.mode.family_label()
                    ))
                } else {
                    Ok(())
                };
            }
        };
        if matches!(
            bound,
            AssignmentStrengthLayout::ConcentrationOffset
                | AssignmentStrengthLayout::FixedConcentration
        ) && bound != expected
        {
            return Err(format!(
                "rho layout {bound:?} contradicts the ordered Beta--Bernoulli concentration, \
                 which is {} (#2933 F45)",
                if self.effective_alpha_is_learnable() {
                    "learnable"
                } else {
                    "fixed"
                },
            ));
        }
        Ok(())
    }

    pub(crate) fn learnable_alpha_rho_domain(&self) -> Result<Option<(f64, f64)>, String> {
        let AssignmentMode::OrderedBetaBernoulli { alpha, .. } = self.mode else {
            return Ok(None);
        };
        if !self.effective_alpha_is_learnable() {
            return Ok(None);
        }
        gam_terms::analytic_penalties::learnable_weight_coordinate_domain(alpha)
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
        let row_gates = match self.mode {
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
        Ok(row_gates)
    }

    /// #1557 — fill-into-caller-buffer twin of [`Self::try_assignments_row`].
    ///
    /// `out` must have length `k_atoms()`; it is fully overwritten with the same
    /// values the allocating variant would return. Every branch (early-return
    /// K==1 Softmax and the per-mode row math) mirrors the allocating path exactly
    /// so the two are bit-identical.
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
        Ok(())
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
    pub(crate) fn flatten_ext_coords(&self) -> Array1<f64> {
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
    local_jac: &mut Array2<f64>,
) {
    match mode {
        AssignmentMode::Softmax { temperature, .. } => {
            if assignments.len() == 1 {
                return;
            }
            // da_k/dl_j = a_k (1[k=j] - a_j) / tau, contracted against
            // the assignment-weighted fitted row. The dense row layout uses
            // the reference-logit chart, so only columns `0..K-1` are free;
            // the final reference logit is fixed at zero and has no row.
            // `fitted` is `Σ_k a_k γ_k` over the whole simplex the gates enter.
            let inv_tau = 1.0 / temperature;
            for logit_col in 0..assignments.len() - 1 {
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
                let activation =
                    gam_linalg::utils::stable_logistic((logits[logit_col] - threshold) * inv_tau);
                let da = activation * (1.0 - activation) * inv_tau;
                for out_col in 0..fitted.len() {
                    local_jac[[logit_col, out_col]] = da * decoded[[logit_col, out_col]];
                }
            }
        }
        // Constant {0, 1} gates: zero data-fit logit derivative everywhere (no
        // logit is a free parameter; callers skip fixed logits, and this arm keeps
        // the JVP identically zero).
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
/// pins it — see [`SaeAssignment::ordered_beta_bernoulli_prior_parameters`]).
/// Returns `(penalty, rho_view)` at weight one; the fixed-α branch has an empty `rho_view`.
fn ordered_beta_bernoulli_prior_penalty(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    base_alpha: f64,
    temperature: f64,
    row_weights: Option<&[f64]>,
) -> Result<(OrderedBetaBernoulliPenalty, Array1<f64>), String> {
    let parameters = assignment
        .ordered_beta_bernoulli_prior_parameters(rho)?
        .ok_or_else(|| {
            "ordered Beta--Bernoulli prior requires an ordered Beta--Bernoulli assignment"
                .to_string()
        })?;
    let learnable = parameters.concentration_is_learnable;
    // A learnable penalty re-resolves `base · exp(ρ)` from its own rho view, so it keeps
    // the base; a fixed penalty carries the resolved concentration directly.
    let alpha_eff = if learnable {
        base_alpha
    } else {
        parameters.concentration
    };
    // #991 design-honesty weights: the ordered Beta--Bernoulli prior is not row-separable (the
    // exact integrated scalar couples rows through the column active mass), so the weights are
    // installed ON the penalty — its value/grad/hessian/hvp/ρ- and third
    // channels all fold them identically (weighted mass `M_k = Σ w_i z_ik` and
    // active-mass Jacobian `u = w·J`), keeping every channel the exact derivative of one
    // weighted energy. `None` gives the unit-weight operator.
    let penalty =
        OrderedBetaBernoulliPenalty::new(assignment.k_atoms(), alpha_eff, temperature, learnable)
            .with_row_weights(row_weights);
    let rho_view = if learnable {
        Array1::from_vec(vec![rho.log_lambda_sparse])
    } else {
        Array1::zeros(0)
    };
    Ok((penalty, rho_view))
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
/// #2330 Patch D — the ordered-Beta--Bernoulli prior structural data the exact-A
/// θ-adjoint contracts for `∂ΔC_obb/∂ℓ` (see
/// `OrderedBetaBernoulliPenalty::logit_theta_adjoint_data`). `None` when the mode
/// is not ordered-Beta--Bernoulli or routing is frozen (the prior curvature is
/// then ρ/θ-inert). The cache-layout contraction lives in gam-sae.
pub(crate) fn ordered_beta_bernoulli_logit_adjoint_data_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<Option<OrderedBetaBernoulliLogitAdjointData>, String> {
    assignment.validate_rho_domain(rho)?;
    let AssignmentMode::OrderedBetaBernoulli {
        temperature, alpha, ..
    } = assignment.mode
    else {
        return Ok(None);
    };
    if assignment.routing_is_frozen() {
        return Ok(None);
    }
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let (penalty, rho_view) =
        ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights)?;
    let target = flat_logits(assignment.logits.view());
    Ok(Some(
        penalty.logit_theta_adjoint_data(target.view(), rho_view.view()),
    ))
}

pub(crate) fn ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
    direction: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    assignment.validate_rho_domain(rho)?;
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
        ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights)?;
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
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords,
            vec![LatentManifold::Euclidean; k],
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
            ordered_beta_bernoulli_prior_penalty(&assignment, &rho, 1.7, 0.8, None).unwrap();
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

/// The assignment prior's value with #991 design-honesty per-row weights:
/// row `i`'s per-row prior contribution is scaled by `w_i` (mean-1). This is the
/// per-row latent prior's analog of the `√w_i`-weighted data likelihood and the
/// `w_i`-weighted `ard_value` — each retained row of a design-honest subsample
/// stands in for `w_i` population rows, so its routing prior carries `w_i` too.
/// `None` gives the unit-weight path. Softmax/threshold gate are row-separable;
/// ordered Beta--Bernoulli instead forms weighted active mass and effective row
/// count inside its integrated scalar. Every derivative uses that same measure.
///
/// The ThresholdGate value is the normalized negative log density `λz + log Z(λ)` per free
/// gate ([`ThresholdGateLogPartition`]), and the softmax value is `λH(a) + ln Z_K(λ)` per row
/// ([`softmax_entropy_partition_weighted`]), so each strength derivative carries `∂_ρ log Z`.
pub(crate) fn assignment_prior_value_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<f64, String> {
    assignment.validate_rho_domain(rho)?;
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return Ok(0.0);
    }
    // #Bug4: under FROZEN routing every logit is inert (the gates come from the
    // ρ-invariant frozen predictor, not `self.logits`), so the whole assignment
    // sparsity prior is a constant with zero gradient/curvature — score it as 0 to
    // match the derivative-side treatment. That holds for a softmax row too: its
    // gates over frozen logits are constant in the free logits, so its entropy
    // prior is inert in exactly the same way.
    if assignment.routing_is_frozen() {
        return Ok(0.0);
    }
    Ok(match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature)
                .with_row_weights(row_weights);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            // #2933 F45 — `exp(−λH)` is a density over the simplex only with its partition.
            penalty.value(target.view(), rho_view.view())
                + softmax_entropy_partition_weighted(assignment, rho_view[0].exp(), row_weights)?
                    .0
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
            )?;
            // #2933 F45 — the prior `exp(−L)` is a density over the relaxed gates only with its
            // partition `C(a_k, N)`. Both branches score it at weight one; with a fixed
            // concentration the partition is a constant, and nothing optimized enters it.
            penalty.value(target.view(), rho_view.view())
                + penalty.log_partition(target.view(), rho_view.view())?.0
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            // Sparsity penalty and reconstruction use the same smooth
            // threshold-centered logistic gate as the gradient and Hessian.
            let sparsity_strength = rho.lambda_sparse()?;
            let gates = threshold_gate_free_gates(
                assignment,
                target.view(),
                threshold,
                temperature,
                row_weights,
            );
            sparsity_strength * gates.weighted_activation
                + gates.weight * ThresholdGateLogPartition::eval(sparsity_strength).value()
        }
        // Sparsity by construction: the fixed-|S| support IS the sparsity — there
        // is no penalty term, so the prior contributes exactly zero.
        AssignmentMode::TopK { .. } => 0.0,
    })
}

/// #991-weighted derivative of the assignment prior in its log strength. Every
/// assignment mode differentiates the same weighted scalar used by its value path.
pub(crate) fn assignment_prior_log_strength_derivative_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<f64, String> {
    assignment.validate_rho_domain(rho)?;
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let target = flat_logits(assignment.logits.view());
    if matches!(assignment.mode, AssignmentMode::Softmax { .. }) && assignment.k_atoms() == 1 {
        return Ok(0.0);
    }
    // #Bug4: frozen routing ⇒ inert prior ⇒ zero ρ-derivative.
    if assignment.routing_is_frozen() {
        return Ok(0.0);
    }
    Ok(match assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            // `∂_ρ[λ·Σ w·H + Σ w·ln Z_K(λ)]`: the energy is degree one in `λ = e^ρ`, the
            // partition is not.
            let penalty = SoftmaxAssignmentSparsityPenalty::new(assignment.k_atoms(), temperature)
                .with_row_weights(row_weights);
            let rho_view = Array1::from_vec(vec![rho.log_lambda_sparse + sparsity.ln()]);
            penalty.value(target.view(), rho_view.view())
                + softmax_entropy_partition_weighted(assignment, rho_view[0].exp(), row_weights)?
                    .1
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            // `∂_ρ[λ·Σ w·z + Σ w·log Z(λ)]`: the energy is degree one in `λ = e^ρ`, the
            // partition is not.
            let sparsity_strength = rho.lambda_sparse()?;
            let gates = threshold_gate_free_gates(
                assignment,
                target.view(),
                threshold,
                temperature,
                row_weights,
            );
            sparsity_strength * gates.weighted_activation
                + gates.weight
                    * ThresholdGateLogPartition::eval(sparsity_strength).log_strength_derivative()
        }
        AssignmentMode::OrderedBetaBernoulli {
            temperature, alpha, ..
        } => {
            // #Bug6: `ordered_beta_bernoulli_prior_penalty` picks the effective-α learnability (an
            // override forces the fixed-α value branch).
            let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
                assignment,
                rho,
                alpha,
                temperature,
                row_weights,
            )?;
            if penalty.learnable_alpha {
                penalty.grad_rho(target.view(), rho_view.view())[0]
                    + penalty.log_partition(target.view(), rho_view.view())?.1[0]
            } else {
                // A fixed concentration leaves `log_lambda_sparse` out of the prior value, so
                // the derivative is an exact zero, like the absent coordinate it would carry.
                0.0
            }
        }
        // No prior term ⇒ no ρ-derivative (sparsity lives in the fixed support).
        AssignmentMode::TopK { .. } => 0.0,
    })
}

/// #991-weighted log-strength Hessian diagonal of the assignment prior. Every
/// assignment mode differentiates the same weighted scalar used by its value path.
pub(crate) fn assignment_prior_log_strength_hdiag_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<Array1<f64>, String> {
    assignment.validate_rho_domain(rho)?;
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
            penalty
                .hessian_diag(target.view(), rho_view.view())
                .ok_or_else(|| {
                    "softmax assignment log-strength hessian diag unavailable".to_string()
                })
        }
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => {
            let sparsity_strength = rho.lambda_sparse()?;
            let inv_tau = 1.0 / temperature;
            let k = assignment.k_atoms();
            let mut d = Array1::<f64>::zeros(target.len());
            for idx in 0..target.len() {
                // #991 — row `idx / k`'s design weight.
                let w_row = row_weights.map_or(1.0, |w| w[idx / k]);
                // #2520 — `∂/∂ρ_sparse` of the curvature `B` actually carries.
                // `smooth_psd_clamp` is homogeneous of degree 1 in its
                // prefactor and the prefactor carries `λ_sparse`, so that
                // derivative IS the majorizer; reading the shared seam is what
                // keeps this channel exact without a second derivation.
                d[idx] = ThresholdGateLogitCurvature::eval(
                    w_row * sparsity_strength,
                    target[idx],
                    threshold,
                    inv_tau,
                )
                .psd_majorizer_hess();
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
            )?;
            if !penalty.learnable_alpha {
                // No coordinate enters a fixed-concentration prior, so its curvature does not
                // move with `log_lambda_sparse`.
                return Ok(Array1::<f64>::zeros(target.len()));
            }
            Ok(penalty.hessian_diag_log_alpha_derivative(target.view(), rho_view.view()))
        }
        // No prior term ⇒ zero curvature everywhere (mirrors the frozen-routing
        // early return; the support carries no free logits at all).
        AssignmentMode::TopK { .. } => Ok(Array1::<f64>::zeros(target.len())),
    }
}

/// The free gates of a ThresholdGate assignment: `Σ w_row·z` over every free logit with
/// `z = σ((ℓ − θ)/τ)`, and `Σ w_row` over the same logits.
struct ThresholdGateFreeGates {
    weighted_activation: f64,
    weight: f64,
}

fn threshold_gate_free_gates(
    assignment: &SaeAssignment,
    target: ArrayView1<'_, f64>,
    threshold: f64,
    temperature: f64,
    row_weights: Option<&[f64]>,
) -> ThresholdGateFreeGates {
    let k = assignment.k_atoms();
    let mut gates = ThresholdGateFreeGates {
        weighted_activation: 0.0,
        weight: 0.0,
    };
    for (idx, &logit) in target.iter().enumerate() {
        // #991 — this row stands in for `w_i` population rows.
        let w_row = row_weights.map_or(1.0, |w| w[idx / k]);
        gates.weighted_activation +=
            w_row * gam_linalg::utils::stable_logistic((logit - threshold) / temperature);
        gates.weight += w_row;
    }
    gates
}

/// The log partition function of the ThresholdGate prior on one gate, and its derivative in
/// the log strength `ρ = ln λ`.
///
/// #2933 F45. The energy `λz` on `z ∈ (0, 1)` is a density only after dividing by
/// `Z(λ) = ∫₀¹ e^{−λz} dz = (1 − e^{−λ})/λ`, so each free gate's negative log prior is
/// `λz + log Z(λ)`, and with the change of variables [`GateLogitJacobian`] it integrates to one
/// over the logit for every `λ`, `θ` and `τ`. Without `log Z` the no-data mass is `Z(λ) < 1`
/// and the strength derivative lacks `λ·∂_λ log Z = λ/(e^λ − 1) − 1 = −λ·E[z]`, which is the
/// term that lets the criterion balance the fitted mean gate against the prior mean instead of
/// always lowering `λ`. A design-weighted row carries `w·log Z(λ)` beside its `w·λz`, as the
/// Jacobian carries `w·J`: it stands in for `w` population rows, each a normalized density.
///
/// `Z(λ) = exprel(−λ)`, evaluated by [`gam_math::special::log_exprel`]. The derivative is
/// `−(e^λ − 1 − λ)/(e^λ − 1)` through [`gam_math::special::expm1_minus_x`] up to `λ = 1/2`,
/// which never subtracts two terms near one as `λ → 0` (where it is `−λ/2`); above that it is
/// `λe^{−λ}/(1 − e^{−λ}) − 1`, which never forms `e^λ`. Both forms are accurate at the branch
/// point, the same one `gam_math` uses for its small-argument series.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ThresholdGateLogPartition {
    value: f64,
    log_strength_derivative: f64,
}

impl ThresholdGateLogPartition {
    pub(crate) fn eval(strength: f64) -> Self {
        let log_strength_derivative = if strength <= 0.5 {
            -gam_math::special::expm1_minus_x(strength) / strength.exp_m1()
        } else {
            strength * (-strength).exp() / -(-strength).exp_m1() - 1.0
        };
        Self {
            value: gam_math::special::log_exprel(-strength),
            log_strength_derivative,
        }
    }

    /// `log Z(λ) = ln[(1 − e^{−λ})/λ]`.
    pub(crate) fn value(self) -> f64 {
        self.value
    }

    /// `∂ log Z/∂ ln λ = λ/(e^λ − 1) − 1`.
    pub(crate) fn log_strength_derivative(self) -> f64 {
        self.log_strength_derivative
    }
}

/// `(Σ_i w_i·ln Z_K(λ), Σ_i w_i·∂ ln Z_K/∂ ln λ)`: the softmax entropy prior's partition per unit
/// row weight ([`softmax_entropy_log_partition`], #2933 F45).
///
/// The normalizer integrates the simplex of all `K` gates, which is what the chart over the
/// `K − 1` free logits of a row covers. A row whose logits are held beyond the reference atom
/// integrates a lower-dimensional slice with a different normalizer, so it is refused rather than
/// priced with this one.
fn softmax_entropy_partition_weighted(
    assignment: &SaeAssignment,
    strength: f64,
    row_weights: Option<&[f64]>,
) -> Result<(f64, f64), String> {
    let k = assignment.k_atoms();
    match simplex_gate_frame(assignment) {
        Some((free, _)) if free.len() + 1 == k => {}
        _ => {
            return Err(format!(
                "softmax entropy partition: the prior is normalized over the simplex of all K={k} \
                 gates, but this assignment holds logits beyond the reference atom (#2933 F45)"
            ));
        }
    }
    let partition = softmax_entropy_log_partition(k, strength)?;
    let rows = row_weights.map_or(assignment.n_obs() as f64, |weights| weights.iter().sum());
    Ok((
        rows * partition.value,
        rows * partition.log_strength_derivative,
    ))
}

/// The ThresholdGate sparsity prior's curvature at ONE logit, split into the
/// PSD majorizer the Newton/Schur factor declares and the non-positive
/// remainder that restores the exact signed curvature.
///
/// #2520. The exact second derivative of `λ·σ((ℓ−θ)/τ)` is
/// `λ·s·(1 − 2a)/τ²` with `a = σ((ℓ−θ)/τ)` and `s = a(1−a) ≥ 0`, which is
/// NEGATIVE for every logit above the threshold — the sigmoid penalty is
/// concave there. Written into `B` verbatim it made the per-row `H_tt` block
/// indefinite on exactly the atoms the gate had switched ON, and the
/// factorization then spectrally deflated those directions to unit stiffness:
/// #1419's pathology, on the one prior family that never received #1419's
/// treatment.
///
/// The split is [`SaeManifoldAtom`]'s periodic-ARD pair
/// (`psd_majorizer_hess` / `negative_hessian_remainder`) applied verbatim, and
/// it introduces NO new constant: the signed factor `1 − 2a ∈ (−1, 1)` is
/// dimensionless and lives on the same scale as the ARD axis's `cos κt ∈
/// [−1, 1]`, so #2339's derived softplus temperature transfers with its
/// derivation intact.
///
/// Homogeneity is load-bearing exactly as it is for ARD:
/// [`gam_linalg::utils::smooth_psd_clamp`] is degree-1 in its prefactor, and
/// the prefactor carries `λ_sparse`, so
/// `∂/∂ρ_sparse[majorizer] == majorizer` and the log-strength ρ-channel
/// ([`assignment_prior_log_strength_hdiag_weighted`]) stays exact by reading
/// the same seam rather than by a separate derivation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ThresholdGateLogitCurvature {
    activation: f64,
    exact: f64,
    majorized: f64,
    exact_logit_derivative: f64,
    majorized_logit_derivative: f64,
}

impl ThresholdGateLogitCurvature {
    /// `strength` is the design-weighted prior strength `w_row·λ_sparse`; the
    /// caller keeps that convention so value, gradient and curvature share one
    /// weighting (#991).
    pub(crate) fn eval(strength: f64, logit: f64, threshold: f64, inv_tau: f64) -> Self {
        let activation = gam_linalg::utils::stable_logistic((logit - threshold) * inv_tau);
        let slope = activation * (1.0 - activation);
        // Non-negative magnitude, and the dimensionless signed factor it
        // multiplies. The clamp acts on the second and scales with the first.
        let magnitude = strength * slope * inv_tau * inv_tau;
        let signed = 1.0 - 2.0 * activation;
        // The logit derivative of BOTH curvatures, from the SAME evaluation, so
        // no theta-adjoint can differentiate a function no route installs.
        //
        // Writing `C` for the dimensionless clamp
        // ([`gam_linalg::utils::smooth_psd_clamp`] at unit prefactor) and using
        // `da/dl = s/tau`, `ds/dl = s(1-2a)/tau`, hence
        // `d(magnitude)/dl = strength*s*(1-2a)/tau^3` and `d(signed)/dl = -2s/tau`:
        //
        //   d(exact)/dl     = strength*s*(1-6a+6a^2)/tau^3        (`C` = identity)
        //   d(majorized)/dl = strength*s/tau^3 * [(1-2a)*C(1-2a) - 2s*C'(1-2a)]
        //
        // The two coincide wherever the clamp is inactive (`C(x) = x`, `C' = 1`
        // gives `(1-2a)^2 - 2s = 1-6a+6a^2` exactly); the majorized one is `0`
        // on the deep concave half where `B` carries a hard `0`; and at the seam
        // (`a = 1/2`) it is HALF the exact value -- the C-1 average of the hard
        // clamp's two one-sided slopes, which is what #2339's smoothing is for.
        let signed_clamp = gam_linalg::utils::smooth_psd_clamp(1.0, signed);
        let signed_clamp_slope = gam_linalg::utils::smooth_psd_clamp_slope(signed);
        let third_scale = strength * slope * inv_tau * inv_tau * inv_tau;
        Self {
            activation,
            exact: magnitude * signed,
            majorized: gam_linalg::utils::smooth_psd_clamp(magnitude, signed),
            exact_logit_derivative: third_scale
                * (1.0 - 6.0 * activation + 6.0 * activation * activation),
            majorized_logit_derivative: third_scale
                * (signed * signed_clamp - 2.0 * slope * signed_clamp_slope),
        }
    }

    /// `a = σ((ℓ−θ)/τ)`, shared with the gradient so both read one evaluation.
    pub(crate) fn activation(self) -> f64 {
        self.activation
    }

    /// Positive-semidefinite curvature written into `B`.
    pub(crate) fn psd_majorizer_hess(self) -> f64 {
        self.majorized
    }

    /// Signed correction with `psd_majorizer_hess + negative_hessian_remainder
    /// == exact` bit-for-bit, so `A = B + ΔC` is unchanged as an operator.
    /// Non-positive, because `softplus_{τ₀}(c) ≥ max(c, 0) ≥ c`.
    pub(crate) fn negative_hessian_remainder(self) -> f64 {
        self.exact - self.majorized
    }

    /// Logit derivative of [`Self::psd_majorizer_hess`] -- the theta-adjoint of
    /// the curvature `B` actually installs, and the ThresholdGate twin of
    /// `SaeManifoldTerm::ard_majorized_hessian_derivative`. It must exist
    /// because #2520 replaced the raw signed curvature in `B` with the clamp,
    /// which left the pre-#2520 third derivative `P'''` differentiating an
    /// operator no route installs any more.
    pub(crate) fn majorized_hess_logit_derivative(self) -> f64 {
        self.majorized_logit_derivative
    }

    /// Logit derivative of the UNCLAMPED signed curvature that `A = B + dC`
    /// carries: `P'''(l) = (lambda/tau^3)*s*(1-6a+6a^2)` (#1415).
    pub(crate) fn exact_hess_logit_derivative(self) -> f64 {
        self.exact_logit_derivative
    }

}

/// The logit Jacobian of a sigmoid gate's prior density (#2080).
///
/// The ordered Beta--Bernoulli and ThresholdGate priors are energies on the gate
/// `z = σ((ℓ − θ)/τ)`, while the inner solve and the quasi-Laplace criterion integrate over
/// the logit `ℓ`. Changing variables, `p(ℓ) = p(z)·|dz/dℓ| = p(z)·z(1 − z)/τ`, so the
/// penalized objective carries `−ln[z(1 − z)/τ]` per free gate. That leaves the mass of the
/// energy unchanged, so it does not normalize it (#2933 F45). The ThresholdGate energy
/// carries its partition ([`ThresholdGateLogPartition`]), and so does ordered Beta--Bernoulli
/// with a learnable concentration (`OrderedBetaBernoulliPenalty::log_partition`); with a fixed
/// concentration its energy is `λ_sparse·L`, whose normalizer is not computed, and it stays an
/// unnormalized energy. Without the Jacobian the prior in
/// `ℓ` is improper: along a saturated logit the data slope and the prior slope both
/// decay like `e^{−|ℓ|/τ}`, the objective has no finite mode, and the evidence curvature
/// `λ_ℓ ∝ e^{−|ℓ|/τ}` drifts through the exact-A rank floor. On the #2080 wide-p fixture
/// 121–144 of 424 exact-A directions, every one a gate logit at `15 ≤ |ℓ/τ| < 50`,
/// crossed that floor between neighbouring ρ and moved the criterion by `½·|ln floor|`
/// per crossing (pool jobs 598388, 603989).
///
/// With `x = (ℓ − θ)/τ`: `J = |x| + 2·ln(1 + e^{−|x|}) + ln τ`, `J′ = (2z − 1)/τ`,
/// `J″ = 2z(1 − z)/τ²` and `J‴ = 2z(1 − z)(1 − 2z)/τ³`. `J″ ≥ 0` is the exact curvature,
/// so `B` carries it and `ΔC` gains no remainder. Nothing here depends on ρ, so no ρ
/// channel changes. Each gate carries its row's design weight (#991).
#[derive(Clone, Copy, Debug)]
pub(crate) struct GateLogitJacobian {
    value: f64,
    gradient: f64,
    curvature: f64,
    third: f64,
}

impl GateLogitJacobian {
    pub(crate) fn eval(weight: f64, logit: f64, threshold: f64, temperature: f64) -> Self {
        let inv_tau = 1.0 / temperature;
        let x = (logit - threshold) * inv_tau;
        // `z(1 − z) = e^{−|x|}/(1 + e^{−|x|})²` and `2z − 1 = tanh(x/2)`, both free of the
        // cancellation `1 − z` suffers where a gate saturates on: at `x = 25` that loses five
        // digits, and past `x ≈ 37` it rounds the slope to exactly zero.
        let tail = (-x.abs()).exp();
        let slope = tail / ((1.0 + tail) * (1.0 + tail));
        let centre = (0.5 * x).tanh();
        Self {
            value: weight * (x.abs() + 2.0 * tail.ln_1p() + temperature.ln()),
            gradient: weight * centre * inv_tau,
            curvature: weight * 2.0 * slope * inv_tau * inv_tau,
            third: -weight * 2.0 * slope * centre * inv_tau * inv_tau * inv_tau,
        }
    }

    /// `w·(−ln[z(1 − z)/τ])`.
    pub(crate) fn value(self) -> f64 {
        self.value
    }

    /// `w·(2z − 1)/τ`.
    pub(crate) fn gradient(self) -> f64 {
        self.gradient
    }

    /// `w·2z(1 − z)/τ²`: exact and non-negative.
    pub(crate) fn curvature(self) -> f64 {
        self.curvature
    }

    /// `w·2z(1 − z)(1 − 2z)/τ³`, the logit derivative of [`Self::curvature`].
    pub(crate) fn third(self) -> f64 {
        self.third
    }
}

/// `(θ, τ)` for a mode whose gates are per-logit sigmoids `z = σ((ℓ − θ)/τ)`, where
/// [`GateLogitJacobian`] applies. Softmax gates share one simplex per row, and TopK has
/// no free logits.
pub(crate) fn sigmoid_gate_frame(mode: &AssignmentMode) -> Option<(f64, f64)> {
    match *mode {
        AssignmentMode::OrderedBetaBernoulli { temperature, .. } => Some((0.0, temperature)),
        AssignmentMode::ThresholdGate {
            temperature,
            threshold,
        } => Some((threshold, temperature)),
        AssignmentMode::Softmax { .. } | AssignmentMode::TopK { .. } => None,
    }
}

/// One free gate's [`GateLogitJacobian`], or `None` for a fixed logit or a mode without
/// per-logit sigmoid gates, which carry no gate prior and so no change of variables.
fn gate_logit_jacobian_at(
    assignment: &SaeAssignment,
    row_weights: Option<&[f64]>,
    row: usize,
    atom: usize,
) -> Option<GateLogitJacobian> {
    let (threshold, temperature) = sigmoid_gate_frame(&assignment.mode)?;
    if assignment.logits_are_fixed() {
        return None;
    }
    let weight = row_weights.map_or(1.0, |w| w[row]);
    Some(GateLogitJacobian::eval(
        weight,
        assignment.logits[[row, atom]],
        threshold,
        temperature,
    ))
}

/// The free logits of a softmax assignment and its temperature, or `None` when the mode is
/// not softmax or no logit is free.
///
/// A softmax row's prior is the entropy energy `λ·H(z)` on the simplex, while the inner solve
/// and the quasi-Laplace criterion integrate over its free logits. That energy is not
/// normalized: its simplex partition `∫_Δ e^{−λH(z)} dz` depends on `λ` and is not computed,
/// and the change of variables below leaves the mass unchanged (#2933 F45). The chart holds the reference logit `K − 1` at
/// zero, and frozen routing holds every logit, leaving no free set.
/// For the free set `F`, with `R = 1 − Σ_{i∈F} z_i` the mass on the held atoms, the change of
/// variables has `|det ∂z_F/∂ℓ_F| = Π_{i∈F} z_i · R / τ^{|F|}` (#2080), so each row carries
/// `J = −Σ_{i∈F} ln z_i − ln R + |F|·ln τ`. With `c = |F| + 1`, `∂J/∂ℓ_j = (c·z_j − 1)/τ` and
/// `∂²J/∂ℓ_i∂ℓ_j = c·z_i(δ_ij − z_j)/τ²`, which is exact and PSD. Without it a minority atom's
/// probability runs to zero with no finite mode, the softmax twin of the saturated sigmoid
/// gate [`GateLogitJacobian`] fixes. Nothing here depends on ρ.
fn simplex_gate_frame(assignment: &SaeAssignment) -> Option<(Vec<usize>, f64)> {
    let AssignmentMode::Softmax { temperature, .. } = assignment.mode else {
        return None;
    };
    if assignment.logits_are_fixed() {
        return None;
    }
    let free: Vec<usize> = (0..assignment.k_atoms().saturating_sub(1)).collect();
    (!free.is_empty()).then_some((free, temperature))
}

/// `ln Σ_{atom ∈ atoms} e^{ℓ_atom/τ}`, through gam-math's max-shifted compensated log-sum-exp so no
/// exponent overflows or underflows to an infinite logarithm.
fn log_sum_exp_scaled(
    logits: ArrayView1<'_, f64>,
    atoms: impl Iterator<Item = usize>,
    inv_tau: f64,
) -> f64 {
    let scaled: Vec<f64> = atoms.map(|atom| logits[atom] * inv_tau).collect();
    gam_math::probability::positive_log_sum_exp(&scaled)
}

/// One softmax row's `J = −Σ_{i∈F} ln z_i − ln R + |F|·ln τ` (see [`simplex_gate_frame`]), from
/// log-sum-exps so a minority atom's `ln z_i` and the held mass `ln R` stay finite.
fn simplex_gate_logit_jacobian_value(
    logits: ArrayView1<'_, f64>,
    free: &[usize],
    temperature: f64,
) -> f64 {
    let inv_tau = temperature.recip();
    let k = logits.len();
    let all = log_sum_exp_scaled(logits, 0..k, inv_tau);
    let held = log_sum_exp_scaled(logits, (0..k).filter(|atom| !free.contains(atom)), inv_tau);
    let free_log_mass: f64 = free.iter().map(|&atom| logits[atom] * inv_tau - all).sum();
    -free_log_mass - (held - all) + free.len() as f64 * temperature.ln()
}

/// `z_i(1 − z_i)` as `z_i·Σ_{k≠i} z_k`, free of the cancellation at a dominant atom.
fn simplex_spread(z: &Array1<f64>, i: usize) -> f64 {
    let others: f64 = (0..z.len()).filter(|&atom| atom != i).map(|atom| z[atom]).sum();
    z[i] * others
}

/// The gate prior's change-of-variables term in logit coordinates, summed over every free
/// sigmoid gate ([`GateLogitJacobian`]) or every softmax row ([`simplex_gate_frame`]).
pub(crate) fn gate_logit_jacobian_value_weighted(
    assignment: &SaeAssignment,
    row_weights: Option<&[f64]>,
) -> f64 {
    let mut total = 0.0_f64;
    if let Some((free, temperature)) = simplex_gate_frame(assignment) {
        for row in 0..assignment.n_obs() {
            let weight = row_weights.map_or(1.0, |w| w[row]);
            total += weight
                * simplex_gate_logit_jacobian_value(assignment.logits.row(row), &free, temperature);
        }
        return total;
    }
    let k = assignment.k_atoms();
    for row in 0..assignment.n_obs() {
        for atom in 0..k {
            if let Some(jacobian) = gate_logit_jacobian_at(assignment, row_weights, row, atom) {
                total += jacobian.value();
            }
        }
    }
    total
}

/// Gradient and curvature diagonal of [`gate_logit_jacobian_value_weighted`] per flat
/// `(row·K + atom)` logit, the layout of [`assignment_prior_grad_hdiag_weighted`]. A softmax
/// row's full curvature block is [`simplex_gate_logit_jacobian_row_block`].
pub(crate) fn gate_logit_jacobian_grad_hdiag_weighted(
    assignment: &SaeAssignment,
    row_weights: Option<&[f64]>,
) -> (Array1<f64>, Array1<f64>) {
    let k = assignment.k_atoms();
    let n = assignment.n_obs();
    let mut grad = Array1::<f64>::zeros(n * k);
    let mut curvature = Array1::<f64>::zeros(n * k);
    if let Some((free, temperature)) = simplex_gate_frame(assignment) {
        let inv_tau = temperature.recip();
        let count = (free.len() + 1) as f64;
        for row in 0..n {
            let weight = row_weights.map_or(1.0, |w| w[row]);
            let z = softmax_row(assignment.logits.row(row), temperature);
            for &atom in &free {
                grad[row * k + atom] = weight * (count * z[atom] - 1.0) * inv_tau;
                curvature[row * k + atom] =
                    weight * count * simplex_spread(&z, atom) * inv_tau * inv_tau;
            }
        }
        return (grad, curvature);
    }
    for row in 0..n {
        for atom in 0..k {
            if let Some(jacobian) = gate_logit_jacobian_at(assignment, row_weights, row, atom) {
                grad[row * k + atom] = jacobian.gradient();
                curvature[row * k + atom] = jacobian.curvature();
            }
        }
    }
    (grad, curvature)
}

/// One softmax row's Jacobian curvature `c·z_i(δ_ij − z_j)/τ²` over the chart's `K − 1` logit
/// slots, zero on held slots, or `None` when the assignment has no free softmax logit.
pub(crate) fn simplex_gate_logit_jacobian_row_block(
    assignment: &SaeAssignment,
    row_weights: Option<&[f64]>,
    row: usize,
) -> Option<Array2<f64>> {
    let (free, temperature) = simplex_gate_frame(assignment)?;
    let k = assignment.k_atoms();
    let inv_tau = temperature.recip();
    let scale = row_weights.map_or(1.0, |w| w[row]) * (free.len() + 1) as f64 * inv_tau * inv_tau;
    let z = softmax_row(assignment.logits.row(row), temperature);
    let mut block = Array2::<f64>::zeros((k - 1, k - 1));
    for &i in &free {
        block[[i, i]] = scale * simplex_spread(&z, i);
        for &j in &free {
            if j != i {
                block[[i, j]] = -scale * z[i] * z[j];
            }
        }
    }
    Some(block)
}

/// `c = |F| + 1` for a softmax assignment's free logits (see [`simplex_gate_frame`]), or `None`
/// when there is none.
pub(crate) fn simplex_gate_free_count(assignment: &SaeAssignment) -> Option<f64> {
    simplex_gate_frame(assignment).map(|(free, _)| (free.len() + 1) as f64)
}

/// `∂³J/∂ℓ_i∂ℓ_j∂ℓ_w = (c/τ³)·[z_i(δ_iw − z_w)δ_ij − z_i(δ_iw − z_w)z_j − z_i z_j(δ_jw − z_w)]`, the
/// logit derivative of one softmax row's Jacobian curvature, for the θ-adjoints. `z` is the
/// row's full softmax vector, `count` is [`simplex_gate_free_count`], and the caller masks held
/// logits.
pub(crate) fn simplex_gate_logit_jacobian_third(
    z: &[f64],
    i: usize,
    j: usize,
    w: usize,
    count: f64,
    inv_tau: f64,
) -> f64 {
    let delta = |a: usize, b: usize| if a == b { 1.0 } else { 0.0 };
    let dz_i = z[i] * (delta(i, w) - z[w]);
    let dz_j = z[j] * (delta(j, w) - z[w]);
    count * inv_tau * inv_tau * inv_tau * (dz_i * delta(i, j) - dz_i * z[j] - z[i] * dz_j)
}

/// The logit derivative of one gate's [`GateLogitJacobian`] curvature, for the θ-adjoints
/// that differentiate the logit diagonal of `B` or `A`.
pub(crate) fn gate_logit_jacobian_third_weighted(
    assignment: &SaeAssignment,
    row_weights: Option<&[f64]>,
    row: usize,
    atom: usize,
) -> f64 {
    gate_logit_jacobian_at(assignment, row_weights, row, atom).map_or(0.0, GateLogitJacobian::third)
}

/// The ΔC channel of the ThresholdGate prior: the non-positive remainder
/// `exact − majorizer` per flat `(row·K + atom)` logit, masked identically to
/// the curvature [`assignment_prior_grad_hdiag_weighted`] writes into `B`.
///
/// #2520. Reads the same [`ThresholdGateLogitCurvature`] seam and the same
/// [`mask_fixed_logit_entries`] rule as the majorizer, so `A = B + ΔC` cannot
/// drift by construction. Every non-ThresholdGate mode returns zeros: their
/// majorizers are exact, or their remainder travels its own channel (ordered
/// Beta--Bernoulli's rank-one HVP).
pub(crate) fn threshold_gate_negative_hessian_remainder_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<Array1<f64>, String> {
    assignment.validate_rho_domain(rho)?;
    let target = flat_logits(assignment.logits.view());
    let mut remainder = Array1::<f64>::zeros(target.len());
    let AssignmentMode::ThresholdGate {
        temperature,
        threshold,
    } = assignment.mode
    else {
        return Ok(remainder);
    };
    for row in 0..assignment.n_obs() {
        validate_finite_logits(assignment.logits.row(row), row)?;
    }
    let sparsity_strength = rho.lambda_sparse()?;
    let inv_tau = 1.0 / temperature;
    let k = assignment.k_atoms();
    for idx in 0..target.len() {
        let w_row = row_weights.map_or(1.0, |w| w[idx / k]);
        let curvature = ThresholdGateLogitCurvature::eval(
            w_row * sparsity_strength,
            target[idx],
            threshold,
            inv_tau,
        );
        remainder[idx] = curvature.negative_hessian_remainder();
    }
    mask_fixed_logit_entries(assignment, &mut remainder);
    Ok(remainder)
}

/// Zero a flat `(n·K)` per-(row, atom) array when no logit is a free parameter (TopK or
/// frozen routing), so an inert routing contributes nothing to the term (#Bug4).
fn mask_fixed_logit_entries(assignment: &SaeAssignment, arr: &mut Array1<f64>) {
    if assignment.logits_are_fixed() {
        arr.fill(0.0);
    }
}

/// #991-weighted log-strength/target mixed derivative of the assignment prior. The
/// degree-one strength families reuse the `w_i`-weighted gradient; the learnable-α ordered
/// Beta--Bernoulli branch uses the same weighted active mass as the value, gradient, and
/// Hessian, and a fixed-α ordered Beta--Bernoulli prior has no strength to mix with.
pub(crate) fn assignment_prior_log_strength_target_mixed_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<Array1<f64>, String> {
    assignment.validate_rho_domain(rho)?;
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
    // learnable (mode-learnable AND not pinned by an override). Otherwise α is a
    // constant, the prior sits at weight one (#2933 F45), and no coordinate enters it.
    match assignment.mode {
        AssignmentMode::OrderedBetaBernoulli { .. }
            if !assignment.effective_alpha_is_learnable() =>
        {
            Ok(Array1::<f64>::zeros(target.len()))
        }
        AssignmentMode::OrderedBetaBernoulli {
            temperature, alpha, ..
        } if assignment.effective_alpha_is_learnable() => {
            let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
                assignment,
                rho,
                alpha,
                temperature,
                row_weights,
            )?;
            let mut d = penalty.log_alpha_target_mixed_derivative(target.view(), rho_view.view());
            // #Bug4: inert columns carry no mixed derivative.
            mask_fixed_logit_entries(assignment, &mut d);
            Ok(d)
        }
        _ => Ok(assignment_prior_grad_hdiag_weighted(assignment, rho, row_weights)?.0),
    }
}

/// #991-weighted per-(row, atom) logit
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
    assignment.validate_rho_domain(rho)?;
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
            // matching the forward gate. Fixed logits are zeroed post-hoc below.
            let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
                assignment,
                rho,
                alpha,
                temperature,
                row_weights,
            )?;
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
            //
            // The `d` returned here is the curvature the arrow assembly writes
            // into `block.htt` VERBATIM.
            //
            // HISTORY — this block used to say `d` was the EXACT signed curvature
            // and that it was the one assignment/coordinate prior in the SAE inner
            // system that was not PSD-majorized first. `2956f601c` (#2520) ended
            // that: `d` is now `psd_majorizer_hess`, and the concave half travels
            // `threshold_gate_negative_hessian_remainder_weighted` into `ΔC`, so
            // `A = B + ΔC` is still the exact signed operator while `B` — the thing
            // that gets factored, and whose ½log|B| the criterion prices — is PSD
            // like `λ_k·S_k ⊗ I`, periodic ARD's `α·softplus_{τ₀}(cos κt)`,
            // softmax's Gershgorin radius, and
            // `ordered_beta_bernoulli_psd_majorized_hdiag`.
            //
            // A MEASUREMENT IN THIS COMMENT WAS FALSIFIED, and it is kept here
            // because the #2500 gates were written against it and still assert it:
            // "Measured on `threshold_gate_tiny_fixture(straddle = true)`: all ten
            // rows deflate exactly one direction each, and each one is the
            // negative-curvature logit." NO LONGER TRUE — run 30503262222 measures
            // `deflated_direction_count == 0` on BOTH arms of that fixture, which
            // is what `threshold_gate_sparse_operator_is_the_installed_exact_a_\
            // derivative_2500`, `..._is_not_the_raw_prior_on_deflated_rows_2500`
            // and `deflation_map_applies_to_every_row_local_curvature_\
            // coordinate_2500` report when they fail. The `1 − 2a < 0` indefinite
            // `H_tt` those rows deflated is no longer what is factored: `d` is the
            // majorizer. What is NOT established is the mechanism — note the
            // majorizer is not merely non-negative but EXACTLY ZERO above the
            // threshold (`|1 − 2a| ≫ τ₀ ≈ 1.44e-8` makes the softplus term
            // underflow), so a vanishing diagonal entry there would if anything
            // deflate MORE. Do not re-derive that story from this comment; measure.
            //
            // What #2520 did NOT do: teach
            // `materialize_ard_concave_clamp_diagonal` about this remainder, so a
            // mode whose only indefiniteness is the gate's own concave half is
            // still REFUSED rather than priced at its basin curvature under
            // #2336's E-attributability rule. That is a separate change — it moves
            // the `½log|B|` criterion of every ThresholdGate fit and wants its own
            // pre-registered A/B.
            let sparsity_strength = rho.lambda_sparse()?;
            let inv_tau = 1.0 / temperature;
            let k = assignment.k_atoms();
            let mut g = Array1::<f64>::zeros(target.len());
            let mut d = Array1::<f64>::zeros(target.len());
            for idx in 0..target.len() {
                // #991 — row `idx / k`'s design weight scales this row's prior
                // gradient AND curvature identically (both linear in strength).
                let w_row = row_weights.map_or(1.0, |w| w[idx / k]);
                let strength = w_row * sparsity_strength;
                let curvature = ThresholdGateLogitCurvature::eval(
                    strength,
                    target[idx],
                    threshold,
                    inv_tau,
                );
                let activation = curvature.activation();
                g[idx] = strength * activation * (1.0 - activation) * inv_tau;
                // #2520 — the PSD majorizer, not the exact signed curvature.
                // `ΔC` restores the concave half through
                // `threshold_gate_negative_hessian_remainder_weighted`, so the
                // EXACT operator `A = B + ΔC` is unchanged while `B` — the
                // thing that gets factored, and whose ½log|B| the criterion
                // prices — is positive semidefinite like every other
                // assignment/coordinate prior in the crate.
                d[idx] = curvature.psd_majorizer_hess();
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
    // #1033 — under TopK or frozen routing no logit is a free parameter, so none carries a
    // sparsity-prior gradient or curvature: the assembled `gt` and `htt` logit slots stay
    // zero, matching the zero logit-JVP.
    if assignment.logits_are_fixed() {
        grad.fill(0.0);
        diag.fill(0.0);
    }
    Ok((grad, diag))
}

/// Per-row summands `q_ik = w_i z_ik` of the ordered Beta--Bernoulli prior's
/// weighted active mass `M_k = Σ_i q_ik`, row-major `N·K`, from the same penalty
/// configuration as [`ordered_beta_bernoulli_psd_majorizer_third_channels_weighted`].
/// Returns `None` for other assignment modes.
pub(crate) fn ordered_beta_bernoulli_weighted_active_mass_rows(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<Option<Array1<f64>>, String> {
    assignment.validate_rho_domain(rho)?;
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
    let (penalty, rho_view) =
        ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights)?;
    penalty.validate_rho(rho_view.view())?;
    Ok(Some(penalty.weighted_active_mass_rows(target.view())))
}

/// Build exact derivatives of the ordered Beta--Bernoulli PSD curvature
/// majorizer for the SAE log-det adjoint Γ, using the same penalty configuration —
/// `alpha`/`tau`/`learnable_alpha` and the `lambda_sparse` weight convention —
/// that [`assignment_prior_grad_hdiag_weighted`] assembles into `htt`. Returns
/// `None` for other assignment modes. Every row carries the #991 design-honesty
/// weight the assembled `htt` carried (the channels must differentiate the
/// same weighted operator; `z_jac` carries the weighted active-mass derivative
/// `u = w·J`).
pub(crate) fn ordered_beta_bernoulli_psd_majorizer_third_channels_weighted(
    assignment: &SaeAssignment,
    rho: &SaeManifoldRho,
    row_weights: Option<&[f64]>,
) -> Result<Option<OrderedBetaBernoulliHessianDiagThirdChannels>, String> {
    assignment.validate_rho_domain(rho)?;
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
    // `assignment_prior_grad_hdiag_weighted` uses, so an α override differentiates the same
    // fixed-α operator. Fixed-logit columns are zeroed post-hoc below (the channel
    // arrays are not internally column-masked).
    let (penalty, rho_view) =
        ordered_beta_bernoulli_prior_penalty(assignment, rho, alpha, temperature, row_weights)?;
    let mut channels = penalty.psd_majorizer_logit_third_channels(target.view(), rho_view.view());
    // #1033 — under TopK or frozen routing every logit is fixed, so the #1006 θ-adjoint
    // differentiates the same zeroed `htt` that `assignment_prior_grad_hdiag_weighted`
    // assembled.
    if assignment.logits_are_fixed() {
        channels.z_jac.fill(0.0);
        channels.local_logit_third.fill(0.0);
        channels.m_channel.fill(0.0);
        channels.diagonal_term.fill(0.0);
        channels.mass_hessian_coefficient.fill(0.0);
        channels.mass_hessian_log_alpha_derivative.fill(0.0);
    }
    Ok(Some(channels))
}

#[cfg(test)]
mod support_measure_tests {
    use super::*;

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
mod ordered_alpha_domain_tests {
    use super::*;
    use gam_problem::{LOG_STRENGTH_MAX, LOG_STRENGTH_MIN};

    fn ordered_assignment(alpha: f64) -> SaeAssignment {
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((3, 2)),
            vec![Array2::<f64>::zeros((3, 1)); 2],
            vec![LatentManifold::Euclidean; 2],
            AssignmentMode::ordered_beta_bernoulli(0.8, alpha, true),
        )
        .unwrap()
    }

    #[test]
    fn learnable_ordered_alpha_tightens_sparse_rho_face_without_saturation() {
        let alpha = 1.7_f64;
        let assignment = ordered_assignment(alpha);
        let (lower, upper) = assignment
            .learnable_alpha_rho_domain()
            .unwrap()
            .expect("learnable ordered alpha owns the sparse rho coordinate");
        assert!(upper < LOG_STRENGTH_MAX);

        let legal = SaeManifoldRho::new(upper, 0.0, vec![Array1::zeros(1); 2])
            .for_assignment(&assignment);
        assignment
            .validate_rho_domain(&legal)
            .expect("closed effective-alpha upper face is legal");
        let invalid = SaeManifoldRho::new(upper + 1.0e-6, 0.0, vec![Array1::zeros(1); 2])
            .for_assignment(&assignment);
        assert!(assignment.validate_rho_domain(&invalid).is_err());

        assert_eq!(
            lower,
            LOG_STRENGTH_MIN - alpha.ln(),
            "lower face must be shifted by the base concentration too"
        );
    }
}

#[cfg(test)]
mod fill_into_buffer_1557_tests {
    //! #1557 — the fill-into-caller-buffer variant
    //! [`SaeAssignment::try_assignments_row_into`] must produce
    //! BIT-IDENTICAL output to the allocating
    //! [`SaeAssignment::try_assignments_row`] across every assignment
    //! mode (Softmax, OrderedBetaBernoulli, threshold gate) and the K==1
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
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords,
            vec![LatentManifold::Euclidean; k],
            mode,
        )
        .unwrap()
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
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords,
            vec![LatentManifold::Euclidean; k],
            AssignmentMode::ordered_beta_bernoulli(0.5, 1.0, false),
        )
        .unwrap()
    }

    /// Freeze the free logits as the routing, as the amortized fit driver does.
    fn frozen(mut assignment: SaeAssignment) -> SaeAssignment {
        assignment.frozen_logits = Some(assignment.logits.clone());
        assignment
    }

    #[test]
    fn frozen_routing_decouples_gates_from_logit_updates_1033() {
        let (n, k) = (6usize, 3usize);
        let mut a = frozen(ordered_beta_bernoulli_assignment(n, k));
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
        let a = frozen(ordered_beta_bernoulli_assignment(n, k));
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
        let mut a = frozen(ordered_beta_bernoulli_assignment(n, k));
        // Under frozen routing EVERY logit is fixed (not a free Newton coord).
        assert!(a.logits_are_fixed(), "frozen routing must fix ALL logits");
        // Thawing restores the free-logit path.
        a.frozen_logits = None;
        assert!(!a.routing_is_frozen());
        assert!(!a.logits_are_fixed(), "thaw must restore the free-logit path");
    }
}

#[cfg(test)]
mod gate_logit_jvp_2933_tests {
    //! #2933 F04 — the production logit JVP of every gate family equals a central difference of
    //! the production forward seam, and under frozen routing, where every logit is held, the
    //! forward gates do not move with the free logits, so the zero JVP its callers install is
    //! the forward slope.
    use super::*;

    /// Decoder rows `γ_k` (K ≤ 3, p = 2), distinct per atom so no contraction cancels.
    const DECODED: [[f64; 2]; 3] = [[2.0, -0.7], [3.0, 1.1], [-1.3, 0.4]];

    fn modes() -> [AssignmentMode; 3] {
        [
            AssignmentMode::softmax(0.7),
            AssignmentMode::ordered_beta_bernoulli(0.8, 1.3, false),
            AssignmentMode::threshold_gate(0.9, 0.2),
        ]
    }

    fn one_row_assignment(mode: AssignmentMode, logits: &[f64]) -> SaeAssignment {
        let k = logits.len();
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::from_shape_vec((1, k), logits.to_vec()).unwrap(),
            vec![Array2::<f64>::zeros((1, 1)); k],
            vec![LatentManifold::Euclidean; k],
            mode,
        )
        .unwrap()
    }

    /// `Σ_k a_k γ_k`, read through the production forward seam.
    fn fitted_row(assignment: &SaeAssignment, decoded: &Array2<f64>) -> Array1<f64> {
        let gates = assignment
            .try_assignments_row(0)
            .expect("the forward seam evaluates");
        decoded.t().dot(&gates)
    }

    #[test]
    fn gate_logit_jvp_matches_forward_difference_for_every_gate_family_2933() {
        let base_logits = [0.4_f64, -0.6, 0.9];
        let step = 1.0e-5;
        let mut checked = 0usize;
        for mode in modes() {
            for k in [2usize, 3] {
                let decoded = Array2::from_shape_fn((k, 2), |(atom, out)| DECODED[atom][out]);
                let label = format!("{} K={k}", mode.family_label());
                let assignment = one_row_assignment(mode, &base_logits[..k]);
                assert!(
                    !assignment.logits_are_fixed(),
                    "{label}: free routing holds no logit"
                );
                let gates = assignment
                    .try_assignments_row(0)
                    .expect("free routing evaluates");
                let fitted = decoded.t().dot(&gates);
                let mut jac = Array2::<f64>::zeros((assignment.row_block_dim(), 2));
                fill_assignment_logit_jvp_rows(
                    assignment.mode,
                    assignment.logits.row(0),
                    gates.view(),
                    decoded.view(),
                    fitted.view(),
                    &mut jac,
                );
                let mut live_slope = 0.0_f64;
                for slot in 0..assignment.assignment_coord_dim() {
                    let mut plus = assignment.clone();
                    let mut minus = assignment.clone();
                    plus.logits[[0, slot]] += step;
                    minus.logits[[0, slot]] -= step;
                    let difference = (fitted_row(&plus, &decoded)
                        - fitted_row(&minus, &decoded))
                        / (2.0 * step);
                    for out in 0..2 {
                        assert!(
                            (jac[[slot, out]] - difference[out]).abs()
                                <= 1.0e-8 + 1.0e-6 * difference[out].abs(),
                            "{label}: logit slot {slot}, output {out}: JVP {} vs forward \
                             difference {}",
                            jac[[slot, out]],
                            difference[out]
                        );
                        live_slope = live_slope.max(difference[out].abs());
                    }
                }
                assert!(
                    live_slope > 5.0e-2,
                    "{label}: no live logit slope ({live_slope:e})"
                );
                if matches!(mode, AssignmentMode::Softmax { .. }) {
                    assert!(
                        (gates.sum() - 1.0).abs() <= 1.0e-12,
                        "{label}: softmax row mass {}",
                        gates.sum()
                    );
                }
                for atom in 0..k {
                    assert!(
                        gates[atom] > 0.0 && gates[atom] < 1.0,
                        "{label}: gate {atom} = {}",
                        gates[atom]
                    );
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 6, "every gate family at K = 2 and K = 3");
    }

    #[test]
    fn frozen_routing_gates_do_not_move_with_the_free_logits_2933() {
        let decoded = Array2::from_shape_fn((3, 2), |(atom, out)| DECODED[atom][out]);
        for mode in modes() {
            let label = mode.family_label();
            let thawed = one_row_assignment(mode, &[0.4, -0.6, 0.9]);
            let mut frozen = thawed.clone();
            frozen.frozen_logits = Some(thawed.logits.clone());
            assert!(frozen.logits_are_fixed(), "{label}: frozen routing holds every logit");
            let frozen_fitted = fitted_row(&frozen, &decoded);
            let thawed_fitted = fitted_row(&thawed, &decoded);
            for slot in 0..thawed.assignment_coord_dim() {
                let mut moved = frozen.clone();
                moved.logits[[0, slot]] += 0.3;
                assert_eq!(
                    fitted_row(&moved, &decoded),
                    frozen_fitted,
                    "{label}: a frozen-routing gate moved with free logit {slot}"
                );
                let mut thawed_moved = thawed.clone();
                thawed_moved.logits[[0, slot]] += 0.3;
                let response = (fitted_row(&thawed_moved, &decoded) - &thawed_fitted)
                    .mapv(f64::abs)
                    .fold(0.0_f64, |a, &b| a.max(b));
                assert!(
                    response > 1.0e-2,
                    "{label}: the thawed gates must respond to free logit {slot} (control); \
                     response {response:e}"
                );
            }
        }
    }
}

#[cfg(test)]
mod threshold_gate_partition_2933_tests {
    //! #2933 F45 — the ThresholdGate energy `λz` on a gate `z ∈ (0, 1)` is a density only with
    //! its partition `Z(λ) = (1 − e^{−λ})/λ`. With the logit change of variables, the production
    //! prior value must integrate to one over the logit for every strength, threshold and
    //! temperature, and its log-strength derivative must have zero mean under that prior: the
    //! score identity `E[∂_ρ(−log p)] = 0` holds for a normalized density and fails for an
    //! energy whose normalizer depends on `ρ`. A derivative check against the same unnormalized
    //! scalar cannot see the difference, so these integrate instead.
    use super::*;
    use ndarray::array;

    /// Strengths from the small-`λ` series regime through deep saturation.
    const STRENGTHS: [f64; 5] = [1.0e-11, 0.05, 2.0, 30.0, 150.0];
    /// `(θ, τ)` gate frames.
    const FRAMES: [(f64, f64); 3] = [(0.0, 1.0), (0.7, 0.35), (-1.2, 2.5)];

    fn gate_assignment(logits: Array2<f64>, threshold: f64, temperature: f64) -> SaeAssignment {
        let (n, k) = logits.dim();
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![Array2::<f64>::zeros((n, 1)); k],
            vec![LatentManifold::Euclidean; k],
            AssignmentMode::threshold_gate(temperature, threshold),
        )
        .expect("one logit column, coordinate block and manifold per atom")
    }

    fn strength_rho(assignment: &SaeAssignment, strength: f64) -> SaeManifoldRho {
        SaeManifoldRho::new(strength.ln(), 0.0, vec![Array1::zeros(1); assignment.k_atoms()])
            .for_assignment(assignment)
    }

    /// The production negative log prior in logit coordinates: prior value plus Jacobian.
    fn negative_log_prior(assignment: &SaeAssignment, rho: &SaeManifoldRho) -> f64 {
        assignment_prior_value_weighted(assignment, rho, None).expect("admitted prior value")
            + gate_logit_jacobian_value_weighted(assignment, None)
    }

    /// `Σ h·f(x)` over `x = (ℓ − θ)/τ ∈ [−60, 60]` at `h = 1/16`. Each integrand below is
    /// `e^{−λσ(x)}σ(x)σ(−x)/Z` times a factor bounded by `1 + λ`, with `1/Z ≤ 1 + λ`. It is
    /// analytic in `|Im x| < π`, and on `|Im x| ≤ π/2` `Re σ ≥ 0` keeps `|e^{−λσ}| ≤ 1`, so the
    /// trapezoid error is below `(1 + λ)²·e^{−2π(π/2)·16}` and the dropped tails below
    /// `2(1 + λ)²e^{−60}`: both far under the bars, which are set by rounding in the sum.
    fn trapezoid(mut f: impl FnMut(f64) -> f64) -> f64 {
        let h = 1.0 / 16.0;
        (-960_i32..=960).map(|i| h * f(f64::from(i) * h)).sum()
    }

    /// With no data, prior plus Jacobian is a normalized density over the logit. Before the
    /// partition its mass was `Z(λ)`: `0.9754` at `λ = 0.05`, `0.4323` at `λ = 2`, `1/30` and
    /// `1/150` above.
    #[test]
    fn threshold_gate_prior_integrates_to_one_over_its_logit_2933() {
        for (threshold, temperature) in FRAMES {
            let mut assignment = gate_assignment(Array2::zeros((1, 1)), threshold, temperature);
            for strength in STRENGTHS {
                let rho = strength_rho(&assignment, strength);
                let mass = temperature
                    * trapezoid(|x| {
                        assignment.logits[[0, 0]] = threshold + temperature * x;
                        (-negative_log_prior(&assignment, &rho)).exp()
                    });
                assert!(
                    (mass - 1.0).abs() <= 1.0e-11,
                    "ThresholdGate prior at λ={strength}, θ={threshold}, τ={temperature} has \
                     no-data mass {mass:.15e}, not one"
                );
            }
        }
    }

    /// The log-strength derivative has zero mean under the prior at every strength and frame,
    /// and its partition part is `λ·∂_λ log Z = λ/(e^λ − 1) − 1`: `2·(−0.3434823572503343)` at
    /// `λ = 2` (audit §12 check 35), and `−λ/2 + λ²/12` near zero, where the difference form
    /// `λ/expm1(λ) − 1` would carry an absolute error of `ε`, i.e. a relative error of `2ε/λ`.
    #[test]
    fn threshold_gate_log_strength_derivative_has_zero_prior_mean_2933() {
        for (threshold, temperature) in FRAMES {
            let mut assignment = gate_assignment(Array2::zeros((1, 1)), threshold, temperature);
            for strength in STRENGTHS {
                let rho = strength_rho(&assignment, strength);
                let mean = temperature
                    * trapezoid(|x| {
                        assignment.logits[[0, 0]] = threshold + temperature * x;
                        let derivative =
                            assignment_prior_log_strength_derivative_weighted(&assignment, &rho, None)
                                .expect("admitted log-strength derivative");
                        derivative * (-negative_log_prior(&assignment, &rho)).exp()
                    });
                assert!(
                    mean.abs() <= 1.0e-11 * (1.0 + strength),
                    "ThresholdGate log-strength derivative at λ={strength}, θ={threshold}, \
                     τ={temperature} has prior mean {mean:.15e}, not zero"
                );
            }
        }
        let assignment = gate_assignment(array![[0.4]], 0.0, 1.0);
        let gate = 1.0 / (1.0 + (-0.4_f64).exp());
        let partition_slope = |strength: f64| {
            let rho = strength_rho(&assignment, strength);
            let derivative =
                assignment_prior_log_strength_derivative_weighted(&assignment, &rho, None)
                    .expect("admitted log-strength derivative");
            (derivative - strength * gate) / strength
        };
        let at_two = partition_slope(2.0);
        assert!(
            (at_two + 0.3434823572503343).abs() <= 1.0e-13,
            "∂_λ log Z(2) = {at_two:.16e}, expected −0.3434823572503343"
        );
        let small = 1.0e-11;
        let at_small = partition_slope(small);
        assert!(
            (at_small + 0.5 - small / 12.0).abs() <= 1.0e-9,
            "∂_λ log Z({small:e}) = {at_small:.16e}, expected −1/2 + λ/12"
        );
    }

    /// Under row weights the value carries `Σ w·log Z(λ)` over every free gate, and the
    /// log-strength derivative is a central difference of that value.
    #[test]
    fn threshold_gate_partition_follows_free_gates_and_row_weights_2933() {
        let logits = array![[1.3, -0.4], [-2.2, 0.8], [0.1, 4.0]];
        let weights = [0.5, 1.0, 2.0];
        let (threshold, temperature) = (0.3_f64, 0.6_f64);
        let assignment = gate_assignment(logits.clone(), threshold, temperature);
        let strength = 1.7_f64;
        let rho = strength_rho(&assignment, strength);
        let activation: f64 = (0..3)
            .flat_map(|row| (0..2).map(move |atom| (row, atom)))
            .map(|(row, atom)| {
                weights[row] / (1.0 + (-(logits[[row, atom]] - threshold) / temperature).exp())
            })
            .sum();
        let free_weight: f64 = 2.0 * weights.iter().sum::<f64>();
        let log_partition = (-(-strength).exp_m1() / strength).ln();
        let value = assignment_prior_value_weighted(&assignment, &rho, Some(&weights))
            .expect("admitted prior value");
        let expected = strength * activation + free_weight * log_partition;
        assert!(
            (value - expected).abs() <= 1.0e-13 * (1.0 + expected.abs()),
            "weighted ThresholdGate prior value {value:.15e}, expected λ·Σw·z + Σw·log Z = \
             {expected:.15e}"
        );
        let h = 1.0e-4;
        let shifted = |delta: f64| {
            let mut moved = rho.clone();
            moved.log_lambda_sparse += delta;
            assignment_prior_value_weighted(&assignment, &moved, Some(&weights))
                .expect("admitted prior value")
        };
        let difference = (shifted(h) - shifted(-h)) / (2.0 * h);
        let derivative =
            assignment_prior_log_strength_derivative_weighted(&assignment, &rho, Some(&weights))
                .expect("admitted log-strength derivative");
        assert!(
            (derivative - difference).abs() <= 1.0e-7 * (1.0 + derivative.abs()),
            "log-strength derivative {derivative:.12e} vs central difference {difference:.12e}"
        );
    }
}

#[cfg(test)]
mod ordered_beta_bernoulli_partition_2933_tests {
    //! #2933 F45 — with a learnable concentration the ordered Beta--Bernoulli prior `exp(−L)` is
    //! a density over the relaxed gates only with its partition `C(a, N)`. Through the logit
    //! change of variables the production prior value must integrate to one over the logits,
    //! and its log-concentration derivative must have zero mean under that prior. Before the
    //! partition the no-data mass of one gate was `C(α, 1)`: `0.1977`, `0.4263` and `0.3267` at
    //! `α = 0.1, 1, 10`.
    use super::*;

    fn learnable_assignment(logits: Array2<f64>, temperature: f64) -> SaeAssignment {
        let (n, k) = logits.dim();
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![Array2::<f64>::zeros((n, 1)); k],
            vec![LatentManifold::Euclidean; k],
            AssignmentMode::ordered_beta_bernoulli(temperature, 1.0, true),
        )
        .expect("one logit column, coordinate block and manifold per atom")
    }

    /// `ρ = ln α` against the unit base concentration.
    fn concentration_rho(assignment: &SaeAssignment, alpha: f64) -> SaeManifoldRho {
        SaeManifoldRho::new(alpha.ln(), 0.0, vec![Array1::zeros(1); assignment.k_atoms()])
            .for_assignment(assignment)
    }

    fn negative_log_prior(assignment: &SaeAssignment, rho: &SaeManifoldRho) -> f64 {
        assignment_prior_value_weighted(assignment, rho, None).expect("admitted prior value")
            + gate_logit_jacobian_value_weighted(assignment, None)
    }

    /// `Σ h·f(x)` over `x = ℓ/τ ∈ [−range, range]`. The integrand `q_a(σ(x))σ(x)σ(−x)/C` is
    /// analytic in `|Im x| < π/2`, where `Re σ ∈ [0, 1]` keeps both Gamma arguments of `q_a` in
    /// the right half-plane, and it decays like `e^{−|x|}/C`: at `h = 1/16`, `range = 60` (and
    /// `h = 1/8`, `range = 40` per axis in two dimensions) the rule's error and the dropped tails
    /// are far below the bars, which are set by rounding.
    fn trapezoid(range: f64, h: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
        let count = (range / h).round() as i64;
        (-count..=count).map(|i| h * f(i as f64 * h)).sum()
    }

    #[test]
    fn learnable_ordered_beta_bernoulli_prior_integrates_to_one_over_its_logit_2933() {
        for temperature in [0.5_f64, 1.7] {
            let mut assignment = learnable_assignment(Array2::zeros((1, 1)), temperature);
            for alpha in [0.1_f64, 1.0, 10.0] {
                let rho = concentration_rho(&assignment, alpha);
                let mass = temperature
                    * trapezoid(60.0, 1.0 / 16.0, |x| {
                        assignment.logits[[0, 0]] = temperature * x;
                        (-negative_log_prior(&assignment, &rho)).exp()
                    });
                assert!(
                    (mass - 1.0).abs() <= 1.0e-10,
                    "learnable ordered Beta--Bernoulli prior at α={alpha}, τ={temperature} has \
                     no-data mass {mass:.15e}, not one"
                );
            }
        }
    }

    #[test]
    fn learnable_log_concentration_derivative_has_zero_prior_mean_2933() {
        for temperature in [0.5_f64, 1.7] {
            let mut assignment = learnable_assignment(Array2::zeros((1, 1)), temperature);
            for alpha in [0.1_f64, 1.0, 10.0] {
                let rho = concentration_rho(&assignment, alpha);
                let mean = temperature
                    * trapezoid(60.0, 1.0 / 16.0, |x| {
                        assignment.logits[[0, 0]] = temperature * x;
                        let derivative =
                            assignment_prior_log_strength_derivative_weighted(&assignment, &rho, None)
                                .expect("admitted log-concentration derivative");
                        derivative * (-negative_log_prior(&assignment, &rho)).exp()
                    });
                assert!(
                    mean.abs() <= 1.0e-10 * (1.0 + alpha),
                    "log-concentration derivative at α={alpha}, τ={temperature} has prior mean \
                     {mean:.15e}, not zero"
                );
            }
        }
    }

    /// Two rows share one column rate, so the prior does not factor over rows. The normalizer
    /// the production value adds beyond the energy must not depend on the logits, and the
    /// energy, that normalizer and the Jacobian must integrate to one over both logits.
    #[test]
    fn two_row_learnable_prior_integrates_to_one_over_its_logits_2933() {
        let temperature = 0.8_f64;
        for alpha in [0.5_f64, 1.7, 6.0] {
            let mut assignment = learnable_assignment(Array2::zeros((2, 1)), temperature);
            let rho = concentration_rho(&assignment, alpha);
            let (penalty, rho_view) =
                ordered_beta_bernoulli_prior_penalty(&assignment, &rho, 1.0, temperature, None)
                    .expect("learnable ordered Beta--Bernoulli penalty");
            let energy = |assignment: &SaeAssignment| {
                penalty.value(flat_logits(assignment.logits.view()).view(), rho_view.view())
            };
            let normalizer = |assignment: &SaeAssignment| {
                assignment_prior_value_weighted(assignment, &rho, None)
                    .expect("admitted prior value")
                    - energy(assignment)
            };
            let log_partition = normalizer(&assignment);
            let mut moved = assignment.clone();
            moved.logits[[0, 0]] = 1.3;
            moved.logits[[1, 0]] = -2.1;
            assert!(
                (normalizer(&moved) - log_partition).abs() <= 1.0e-12 * (1.0 + log_partition.abs()),
                "the prior normalizer must not depend on the logits: {} vs {log_partition}",
                normalizer(&moved)
            );
            let h = 1.0 / 8.0;
            let mut mass = 0.0;
            let count = (40.0 / h) as i64;
            for i in -count..=count {
                assignment.logits[[0, 0]] = temperature * i as f64 * h;
                for j in -count..=count {
                    assignment.logits[[1, 0]] = temperature * j as f64 * h;
                    let jacobian = gate_logit_jacobian_value_weighted(&assignment, None);
                    mass += h * h * (-(energy(&assignment) + jacobian + log_partition)).exp();
                }
            }
            mass *= temperature * temperature;
            assert!(
                (mass - 1.0).abs() <= 1.0e-10,
                "two-row learnable ordered Beta--Bernoulli prior at α={alpha} has no-data mass \
                 {mass:.15e}, not one"
            );
        }
    }
}

#[cfg(test)]
mod ordered_beta_bernoulli_partition_columns_2933_tests {
    //! #2933 F45 — the learnable ordered Beta--Bernoulli normalizer is `Σ_k log C(a_k, N)` over
    //! every column at the effective row count `N = Σ w_i`, and its log-concentration
    //! derivative is a central difference of the value.
    use super::*;
    use gam_terms::analytic_penalties::ordered_beta_bernoulli_log_partition;
    use ndarray::array;

    #[test]
    fn learnable_partition_follows_free_columns_and_effective_rows_2933() {
        let logits = array![[1.1, -0.3, 0.6], [-1.8, 2.0, -0.2], [0.4, 0.9, -2.5]];
        let weights = [0.5, 1.0, 2.0];
        let temperature = 0.7_f64;
        let alpha = 1.3_f64;
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![Array2::<f64>::zeros((3, 1)); 3],
            vec![LatentManifold::Euclidean; 3],
            AssignmentMode::ordered_beta_bernoulli(temperature, 1.0, true),
        )
        .expect("one logit column, coordinate block and manifold per atom");
        let rho = SaeManifoldRho::new(alpha.ln(), 0.0, vec![Array1::zeros(1); 3])
            .for_assignment(&assignment);
        let (penalty, rho_view) = ordered_beta_bernoulli_prior_penalty(
            &assignment,
            &rho,
            1.0,
            temperature,
            Some(&weights),
        )
        .expect("learnable ordered Beta--Bernoulli penalty");
        let target = flat_logits(assignment.logits.view());
        let energy = penalty.value(target.view(), rho_view.view());
        let value = assignment_prior_value_weighted(&assignment, &rho, Some(&weights))
            .expect("admitted prior value");
        let rows: f64 = weights.iter().sum();
        let expected: f64 = (0..3usize)
            .map(|k| {
                let mean = (alpha / (alpha + 1.0)).powi(k as i32 + 1);
                ordered_beta_bernoulli_log_partition(mean / (1.0 - mean), rows)
                    .expect("a, N > 0")
                    .value
            })
            .sum();
        assert!(
            (value - energy - expected).abs() <= 1.0e-12 * (1.0 + expected.abs()),
            "normalizer {} vs Σ over the columns of log C(a_k, {rows}) = {expected}",
            value - energy
        );
        let h = 1.0e-4;
        let shifted = |delta: f64| {
            let mut moved = rho.clone();
            moved.log_lambda_sparse += delta;
            assignment_prior_value_weighted(&assignment, &moved, Some(&weights))
                .expect("admitted prior value")
        };
        let difference = (shifted(h) - shifted(-h)) / (2.0 * h);
        let derivative =
            assignment_prior_log_strength_derivative_weighted(&assignment, &rho, Some(&weights))
                .expect("admitted log-concentration derivative");
        assert!(
            (derivative - difference).abs() <= 1.0e-7 * (1.0 + derivative.abs()),
            "log-concentration derivative {derivative:.12e} vs central difference \
             {difference:.12e}"
        );
    }
}

#[cfg(test)]
mod ordered_beta_bernoulli_fixed_concentration_2933_tests {
    //! #2933 F45 — with its concentration fixed, by the mode or by a per-fit override, the
    //! ordered Beta--Bernoulli prior is the complete prior `exp(−L)` at weight one plus its
    //! constant partition `Σ_k log C(a_k, N)`. `rho.log_lambda_sparse` is then a placeholder
    //! that must not enter the value: a tempering `λ·L` has a partition over the relaxed gates
    //! that the one-dimensional rate integral does not give, and before this change the branch
    //! scored `λ·L` with no normalizer at all. Every expectation is built from an
    //! [`OrderedBetaBernoulliPenalty`] at the stated concentration and
    //! [`ordered_beta_bernoulli_log_partition`], so no assertion reads the resolution it checks.
    use super::*;
    use gam_terms::analytic_penalties::ordered_beta_bernoulli_log_partition;
    use ndarray::array;

    const FIXED_ALPHA: f64 = 0.6;

    /// A fixed-concentration ordered Beta--Bernoulli assignment.
    fn fixed_assignment(logits: Array2<f64>, temperature: f64, alpha: f64) -> SaeAssignment {
        let (n, k) = logits.dim();
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![Array2::<f64>::zeros((n, 1)); k],
            vec![LatentManifold::Euclidean; k],
            AssignmentMode::ordered_beta_bernoulli(temperature, alpha, false),
        )
        .expect("one logit column, coordinate block and manifold per atom")
    }

    fn placeholder_rho(k: usize, log_lambda_sparse: f64) -> SaeManifoldRho {
        SaeManifoldRho::new(log_lambda_sparse, 0.0, vec![Array1::zeros(1); k])
    }

    fn trapezoid(range: f64, h: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
        let count = (range / h).round() as i64;
        (-count..=count).map(|i| h * f(i as f64 * h)).sum()
    }

    /// No data: the production prior value plus the logit Jacobian integrates to one over the
    /// logit for every concentration, temperature and placeholder. Before the change the mass
    /// was `λ`-dependent and `C(α, 1) ≠ 1` even at `λ = 1`.
    #[test]
    fn fixed_concentration_prior_integrates_to_one_for_every_placeholder_2933() {
        for temperature in [0.5_f64, 1.7] {
            for alpha in [0.1_f64, 1.0, 10.0] {
                for log_lambda_sparse in [0.3_f64.ln(), 0.0, 3.0_f64.ln()] {
                    let mut assignment = fixed_assignment(Array2::zeros((1, 1)), temperature, alpha);
                    let rho = placeholder_rho(1, log_lambda_sparse);
                    let mass = temperature
                        * trapezoid(60.0, 1.0 / 16.0, |x| {
                            assignment.logits[[0, 0]] = temperature * x;
                            let negative_log_prior =
                                assignment_prior_value_weighted(&assignment, &rho, None)
                                    .expect("admitted prior value")
                                    + gate_logit_jacobian_value_weighted(&assignment, None);
                            (-negative_log_prior).exp()
                        });
                    assert!(
                        (mass - 1.0).abs() <= 1.0e-10,
                        "fixed ordered Beta--Bernoulli prior at α={alpha}, τ={temperature}, \
                         log_lambda_sparse={log_lambda_sparse:.4} has no-data mass {mass:.15e}, \
                         not one"
                    );
                }
            }
        }
    }

    /// With data and row weights, the value is the weight-one energy plus every
    /// column's partition at `N = Σ w_i`, whatever the placeholder holds; its central
    /// difference in `log_lambda_sparse` is exactly zero, and so is the derivative channel.
    #[test]
    fn fixed_concentration_prior_is_complete_and_reads_no_strength_2933() {
        let logits = array![[1.1, -0.3, 0.6], [-1.8, 2.0, -0.2], [0.4, 0.9, -2.5]];
        let weights = [0.5, 1.0, 2.0];
        let temperature = 0.7_f64;
        let alpha = FIXED_ALPHA;
        let target = Array1::from_iter(logits.iter().copied());
        let rows: f64 = weights.iter().sum();
        let energy_penalty = OrderedBetaBernoulliPenalty::new(3, alpha, temperature, false)
            .with_row_weights(Some(&weights));
        let energy = energy_penalty.value(target.view(), Array1::<f64>::zeros(0).view());
        let partition: f64 = (0..3usize)
            .map(|k| {
                let mean = (alpha / (alpha + 1.0)).powi(k as i32 + 1);
                ordered_beta_bernoulli_log_partition(mean / (1.0 - mean), rows)
                    .expect("a, N > 0")
                    .value
            })
            .sum();
        let expected = energy + partition;
        let assignment = fixed_assignment(logits.clone(), temperature, alpha);
        for log_lambda_sparse in [2.0_f64.ln(), -1.5] {
            let rho = placeholder_rho(3, log_lambda_sparse);
            let value = assignment_prior_value_weighted(&assignment, &rho, Some(&weights))
                .expect("admitted prior value");
            assert!(
                (value - expected).abs() <= 1.0e-12 * (1.0 + expected.abs()),
                "log_lambda_sparse={log_lambda_sparse}: prior value \
                 {value:.15e}, but the weight-one energy plus Σ log C(a_k, {rows}) is \
                 {expected:.15e}"
            );
            let h = 1.0e-3;
            let shifted = |delta: f64| {
                let mut moved = rho.clone();
                moved.log_lambda_sparse += delta;
                assignment_prior_value_weighted(&assignment, &moved, Some(&weights))
                    .expect("admitted prior value")
            };
            assert_eq!(
                shifted(h),
                shifted(-h),
                "the fixed-concentration prior must not read \
                 log_lambda_sparse"
            );
            assert_eq!(
                assignment_prior_log_strength_derivative_weighted(
                    &assignment,
                    &rho,
                    Some(&weights)
                )
                .expect("admitted derivative"),
                0.0,
                "the log-strength derivative of a prior that does \
                 not read the coordinate is zero"
            );
        }
    }
}

#[cfg(test)]
mod softmax_entropy_partition_2933_tests {
    //! #2933 F45 — the softmax entropy energy `λ·H(a)` on the simplex is a density only with
    //! its partition `Z_K(λ) = ∫_Δ exp(−λ·H(a)) da`. Through the chart over the `K − 1` free
    //! logits the production prior value must integrate to one for every strength and
    //! temperature, and its log-strength derivative must have zero mean under that prior. Before
    //! the partition the no-data mass was `Z_K(λ)` itself. Every reference is a quadrature over
    //! the logits or the gate, never the production normalizer.
    use super::*;
    use ndarray::array;

    fn softmax_assignment(logits: Array2<f64>, temperature: f64) -> SaeAssignment {
        let (n, k) = logits.dim();
        SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![Array2::<f64>::zeros((n, 1)); k],
            vec![LatentManifold::Euclidean; k],
            AssignmentMode::softmax(temperature),
        )
        .expect("one logit column, coordinate block and manifold per atom")
    }

    fn strength_rho(k: usize, strength: f64) -> SaeManifoldRho {
        SaeManifoldRho::new(strength.ln(), 0.0, vec![Array1::zeros(1); k])
    }

    fn negative_log_prior(assignment: &SaeAssignment, rho: &SaeManifoldRho) -> f64 {
        assignment_prior_value_weighted(assignment, rho, None).expect("admitted prior value")
            + gate_logit_jacobian_value_weighted(assignment, None)
    }

    /// `Σ h·f(i·h)` over `i·h ∈ [−range, range]`: the integrands decay like `e^{−|x|}` in the
    /// scaled logit and are analytic in a strip, so the rule's error and the tails are far below
    /// the bars at these steps.
    fn trapezoid(range: f64, h: f64, mut f: impl FnMut(f64) -> f64) -> f64 {
        let count = (range / h).round() as i64;
        (-count..=count).map(|i| h * f(i as f64 * h)).sum()
    }

    #[test]
    fn softmax_prior_integrates_to_one_over_its_free_logits_2933() {
        for temperature in [0.6_f64, 1.4] {
            for strength in [0.05_f64, 2.0, 9.0] {
                let rho = strength_rho(2, strength);
                let mut two = softmax_assignment(Array2::zeros((1, 2)), temperature);
                let mass_two = temperature
                    * trapezoid(60.0, 1.0 / 16.0, |x| {
                        two.logits[[0, 0]] = temperature * x;
                        (-negative_log_prior(&two, &rho)).exp()
                    });
                assert!(
                    (mass_two - 1.0).abs() <= 1.0e-8,
                    "K=2 softmax prior at λ={strength}, τ={temperature} has no-data mass \
                     {mass_two:.15e}, not one"
                );
                let rho = strength_rho(3, strength);
                let mut three = softmax_assignment(Array2::zeros((1, 3)), temperature);
                let mass_three = temperature
                    * temperature
                    * trapezoid(40.0, 1.0 / 8.0, |x| {
                        trapezoid(40.0, 1.0 / 8.0, |y| {
                            three.logits[[0, 0]] = temperature * x;
                            three.logits[[0, 1]] = temperature * y;
                            (-negative_log_prior(&three, &rho)).exp()
                        })
                    });
                assert!(
                    (mass_three - 1.0).abs() <= 1.0e-8,
                    "K=3 softmax prior at λ={strength}, τ={temperature} has no-data mass \
                     {mass_three:.15e}, not one"
                );
            }
        }
    }

    #[test]
    fn softmax_log_strength_derivative_has_zero_prior_mean_2933() {
        for temperature in [0.6_f64, 1.4] {
            for strength in [0.05_f64, 2.0, 9.0] {
                let rho = strength_rho(2, strength);
                let mut assignment = softmax_assignment(Array2::zeros((1, 2)), temperature);
                let mean = temperature
                    * trapezoid(60.0, 1.0 / 16.0, |x| {
                        assignment.logits[[0, 0]] = temperature * x;
                        let derivative =
                            assignment_prior_log_strength_derivative_weighted(&assignment, &rho, None)
                                .expect("admitted log-strength derivative");
                        derivative * (-negative_log_prior(&assignment, &rho)).exp()
                    });
                assert!(
                    mean.abs() <= 1.0e-8 * (1.0 + strength),
                    "softmax log-strength derivative at λ={strength}, τ={temperature} has prior \
                     mean {mean:.15e}, not zero"
                );
            }
        }
    }

    /// Rows carry the normalized negative log density at their design weight:
    /// `λ·Σ w_i H(a_i) + Σ w_i·ln Z_K(λ)`, with `ln Z_2` from a quadrature over the gate.
    #[test]
    fn weighted_softmax_prior_carries_one_partition_per_unit_weight_2933() {
        let strength = 3.0_f64;
        let temperature = 0.8_f64;
        let logits = array![[1.1, 0.0], [-2.0, 0.0], [0.3, 0.0]];
        let weights = [0.5, 1.0, 1.5];
        let assignment = softmax_assignment(logits.clone(), temperature);
        let rho = strength_rho(2, strength);
        let entropy = |row: usize| {
            let a = 1.0 / (1.0 + (-logits[[row, 0]] / temperature).exp());
            -(a * a.ln() + (1.0 - a) * (1.0 - a).ln())
        };
        let energy: f64 = (0..3).map(|row| weights[row] * strength * entropy(row)).sum();
        // `ln ∫₀¹ exp(λ[a ln a + (1 − a) ln(1 − a)]) da`, through `a = σ(x)`, `da = σ(1 − σ) dx`.
        let log_partition = trapezoid(60.0, 1.0 / 16.0, |x| {
            let a = 1.0 / (1.0 + (-x).exp());
            let b = 1.0 / (1.0 + x.exp());
            (strength * (a * a.ln() + b * b.ln())).exp() * a * b
        })
        .ln();
        let expected = energy + weights.iter().sum::<f64>() * log_partition;
        let value = assignment_prior_value_weighted(&assignment, &rho, Some(&weights))
            .expect("admitted prior value");
        assert!(
            (value - expected).abs() <= 1.0e-9 * (1.0 + expected.abs()),
            "weighted softmax prior {value:.15e}, expected λ·Σw·H + Σw·ln Z_2 = {expected:.15e}"
        );
    }
}
