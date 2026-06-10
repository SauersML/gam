//! The profiled criterion calculus: LAML as a sum of self-differentiating
//! atoms over one sensitivity operator (#931 + #935, consuming #932; carries
//! #740 and #934 for free).
//!
//! # Why this module exists
//!
//! The single most recurring structural bug class in this codebase is
//! objective↔gradient desync: a criterion term's VALUE and its analytic
//! DERIVATIVES computed by separate code that drifts (#752, #748, #808,
//! #901). The #901 campaign proved the class is TWO-layer deep:
//!
//! 1. the determinant term's value used one object (range(Sλ)-projected
//!    logdet) while its trace kernel meant another — fixed by
//!    `intrinsic_hessian_pseudo_logdet_parts` (value and kernel are now one
//!    eigendecomposition); and yet
//! 2. the FD drivers STILL fail with byte-identical blow-ups, because the
//!    DRIFT matrices fed to that kernel (`Ḣ_j = ∂H/∂θ_j + D_βH[β̇_j]`) are
//!    assembled by a third code path that disagrees with the cost's actual
//!    θ-dependence. Fixing the object did not fix the chain rule, because
//!    the chain rule is hand-distributed across thousands of lines.
//!
//! Auditing finds instances; only architecture kills the class. The cure
//! already exists in miniature: `penalty_logdet.rs` emits log|S|₊ and ALL
//! its ρ/ψ/cross derivatives as projections of one factorization, and that
//! term has never desynced since. This module is the design for applying
//! that cure to the WHOLE criterion — including the part `penalty_logdet.rs`
//! never had to face: terms coupled through the inner optimum β̂(θ).
//!
//! # The three-layer design
//!
//! ```text
//!   layer 3   CriterionSum = fold of CriterionAtom emissions      (#931)
//!   layer 2   Sensitivity  = ONE factored H⁺; β̇, ALO, influence,
//!             deletion, θ-HVP are four contractions of it          (#935)
//!   layer 1   row jets     = each family's scalar log-likelihood
//!             written once, derivative towers derived mechanically (#932)
//! ```
//!
//! ## The coupling discipline (the part no issue sketch resolved)
//!
//! Atoms are NOT independent functions of θ: every term touching the inner
//! state moves through β̂(θ) as well. The naive "sum of atoms, each emitting
//! dA/dθ" just relocates the desync into each atom's private chain rule.
//! The discipline here is different — **atoms emit FROZEN partials only**:
//!
//! - `frozen_d1(dir)`  = ∂A/∂θ[dir] at FIXED inner state (β̂, W, H frozen);
//! - `beta_channel()`  = the atom's exact ∂A/∂β̂ data (a gradient vector
//!   and, for second order, the bilinear forms it needs);
//!
//! and the CALCULUS — not the atom — assembles the profiled total
//! derivative through the one sensitivity operator:
//!
//! ```text
//!   D_θ A [dir] = ∂_θ A [dir]  +  ⟨ ∂_β A , β̇(dir) ⟩ ,
//!   β̇(dir)     = −H⁺ · F_{βθ}[dir]          (computed ONCE per direction,
//!                                             shared by every atom)
//! ```
//!
//! Consequences, each of which is a past bug made impossible:
//!
//! - **One β̇ per direction.** Today `D_βH[v]` drifts are built per consumer
//!   (#901 layer 2). Here `ThetaDirection` carries the induced `β̇`, `Ẇ`,
//!   and `Ḣ_total` once; an atom cannot see a different chain than its
//!   neighbors because it never computes one.
//! - **The envelope theorem is a theorem, not a convention.** The inner
//!   objective's β-channel is the KKT residual r itself: ⟨r, β̇⟩ vanishes at
//!   exact stationarity and produces the −½rᵀH⁺r noise-floor correction
//!   (and its gradient) mechanically when r ≠ 0. No site can "forget the
//!   IFT correction" — the calculus applies it to whoever declares a
//!   β-channel. Equally, no site can WRONGLY claim the envelope absorbs a
//!   non-stationary functional: the #784 sampled correction's comment
//!   ("the implicit β̂(ρ) channel is the same envelope term the evaluator
//!   already accounts for") was exactly that error — Δ_b is not stationary
//!   in β̂, so its β-channel is nonzero and the calculus would have charged
//!   it `⟨g_d, β̇⟩` automatically.
//! - **Which inverse "H⁻¹" means is decided once.** The 0dc469bd projected
//!   pseudo-inverse convention, the #901 spectral-threshold matching, and
//!   the Smooth/HardPseudo floor semantics live inside `Sensitivity`; the
//!   five existing dialects (`ift_dbeta_drho_from_solver`, ALO, the #461
//!   influence Jacobian, the unified.rs correction traces, the biobank
//!   marginal-slope route) become contractions of it and their private
//!   factorizations are DELETED (no parallel layers).
//!
//! ## Directional-first derivatives (#740 falls out)
//!
//! Atoms expose `frozen_d1(dir)` / `frozen_d2(dir_i, dir_j)`, never "the
//! gradient vector". The expensive moving data (β̇, Ẇ, Ḣ) is attached to the
//! DIRECTION object and computed once per direction — so the outer gradient
//! costs K direction-builds (not K per-atom chains), the outer Hessian is a
//! θ-HVP per direction pair with no O(K²) dense pair assembly, and a
//! matrix-free trust-region Newton consumes the same channel. Exact only —
//! an approximate directional channel would bias REML (standing #740 rule).
//!
//! ## Self-certification (#934 falls out)
//!
//! Because every atom is a named object emitting value + derivatives from
//! one internal state, `CriterionSum::certify` can FD-audit EACH ATOM
//! SEPARATELY at the optimum for ~2 extra evaluations per atom, naming the
//! desyncing term in the error. The #901 hunt — weeks of triangulating
//! which of {object, kernel, drift, splice} disagreed — becomes
//! `certificate: atom "hessian_logdet" frozen_d1 mismatch on ψ[0]`.
//! Atoms must also declare their smoothness stratum (rank set, active
//! eigenvalue gaps, gate states) so the certifier refuses to FD across a
//! genuine non-differentiability instead of reporting it as a bug: rank
//! changes of the pseudo-logdet, eigenvalue crossings in the #784 frame
//! channel, and trust-gate flips are strata boundaries, not desyncs.
//!
//! # Migration law
//!
//! One term per pass: port the term into an atom, FD-verify the atom in
//! isolation (its own `certify`), delete the old value+gradient code in the
//! SAME commit. No compat shims, no parallel evaluation layers, no
//! "fallback to legacy path" flags. `penalty_logdet.rs` is already the
//! first atom in everything but the trait impl; the landed
//! `intrinsic_hessian_pseudo_logdet_parts` (#901) is the second — its
//! (value, spectral kernel) pair is precisely a `frozen` emission and its
//! `PenaltySubspaceTrace` is the contraction state. The #784 moment seam
//! specified on `block_sampled_marginal_correction` is the third, and the
//! hardest test of the abstraction: a SAMPLED atom whose frozen channel is
//! the explicit penalty score, whose direction channel is one rank-≤3m
//! trace against the shared `Ḣ`, and whose β-channel is the moment vector
//! g_d — it fits the same trait with no special cases, which is the
//! evidence the abstraction is the right one.

use ndarray::{Array1, Array2};
use std::sync::Arc;

use super::unified::PenaltySubspaceTrace;

/// An outer-coordinate direction with its induced inner motion, built ONCE
/// per direction by the calculus and shared by every atom.
///
/// This object is the cure for #901-layer-2: today each gradient consumer
/// assembles its own `Ḣ` (penalty drift + cubic IFT correction) and its own
/// `β̇`, and they disagree. Here the direction owns them; atoms borrow.
///
/// Channels are filled lazily by [`Sensitivity`] (β̇ needs the factored
/// solve; Ḣ_total needs β̇) so a value-only evaluation pays nothing.
pub struct ThetaDirection {
    /// Coordinate index in the packed θ = (ρ‖ψ) layout, with the unit
    /// direction implied; general directions carry a dense `dir` instead.
    pub index: Option<usize>,
    /// Dense direction in θ-space (None ⇒ unit vector at `index`).
    pub dir: Option<Array1<f64>>,
    /// Explicit penalty drift `∂Sλ/∂θ[dir]` (ρ: λ_k S_k; ψ: λ-weighted
    /// basis κ-derivatives from the penalty atom's own factorization).
    pub s_dot: Option<Arc<Array2<f64>>>,
    /// Frozen-state Hessian drift `∂H/∂θ[dir]` at fixed β̂ (for GLMs this is
    /// `s_dot` plus the explicit design-motion term `X_θᵀWX + XᵀWX_θ` when
    /// the design itself moves with ψ).
    pub h_dot_frozen: Option<Arc<Array2<f64>>>,
    /// Induced mode motion `β̇ = −H⁺ F_{βθ}[dir]`, the ONE chain-rule vector
    /// every atom's profiled derivative contracts against.
    pub beta_dot: Option<Arc<Array1<f64>>>,
    /// Total Hessian drift `Ḣ = h_dot_frozen + D_βH[β̇]` (the cubic
    /// correction `Xᵀdiag(c ⊙ Xβ̇)X` applied to the SAME β̇ above). This is
    /// the matrix the logdet trace, the #784 Q_b/Q_c trace, and the θ-HVP
    /// all consume — one construction, no per-consumer reassembly.
    pub h_dot_total: Option<Arc<Array2<f64>>>,
}

/// The one sensitivity operator (#935): a factored, convention-complete
/// `H⁺` built once at the inner optimum.
///
/// Owns the ONLY answer to "which inverse": the spectral pseudo-inverse
/// whose kept set matches the criterion's pseudo-logdet threshold exactly
/// (#901 — value, trace kernel, IFT energy correction, and every solve here
/// share one eigendecomposition and one threshold), or the sparse Cholesky
/// /  Takahashi form at scale. Every consumer below is a CONTRACTION of
/// this object; none holds its own factorization:
///
/// - `beta_dot(dir)`       — dβ̂/dθ for the REML gradient (IFT);
/// - `alo_leverages()`     — t = case-weight perturbations (ALO; absorbs
///                           `AloFactoredHessian`);
/// - `influence(J)`        — t = stage-1 nuisance (#461 absorber);
/// - `case_deletion(i)`    — exact Cook's/dfbeta diagnostics;
/// - `hvp(dir)`            — outer-Hessian θ-HVP (#740): directional trace
///                           + β̈ channel, no K² pair assembly;
/// - `energy(r)`           — −½ rᵀH⁺r noise-floor cost correction with the
///                           same kept-set masking as the logdet value.
///
/// `kernel` doubles as the logdet atom's trace kernel: `tr(H⁺ Ḣ)` IS the
/// pseudo-logdet derivative on the constant-rank stratum, so the gradient
/// of the determinant term and the IFT solves cannot use different
/// inverses — they are fields of the same struct.
pub struct Sensitivity {
    /// Spectral form (U_kept, diag σ_kept) of the penalized Hessian at the
    /// optimum — the same object `intrinsic_hessian_pseudo_logdet_parts`
    /// emits; `kernel.h_proj_inverse = diag(1/σ)` exactly.
    pub kernel: Arc<PenaltySubspaceTrace>,
    /// Pseudo-logdet of the SAME kept set — pinning value and solve to one
    /// threshold decision (the #748/#752/#901 invariant, structural here).
    pub logdet: f64,
    /// Smoothness-stratum fingerprint: kept-rank plus the smallest kept
    /// eigengap. `certify` refuses FD probes that cross a stratum boundary
    /// (rank change or near-degenerate frame) instead of flagging them.
    pub stratum: StratumFingerprint,
}

/// Where the criterion is — and is not — differentiable.
///
/// Pseudo-logdets, eigenframe channels (#784 Q_c), and gate splices are C¹
/// only on constant-rank / gap-bounded strata. Atoms DECLARE their stratum
/// instead of letting consumers discover non-differentiability as
/// "mysterious FD noise"; the certifier and the line search both read it.
pub struct StratumFingerprint {
    /// Number of kept (above-threshold) eigenvalues.
    pub kept_rank: usize,
    /// Smallest relative gap |σ_r − σ_q|/σ_max over the pairs a frame
    /// derivative would divide by; ~0 ⇒ frame channels must decline.
    pub min_relative_eigengap: f64,
}

/// One importance-weighted β-channel: the atom's exact ∂A/∂β̂ together with
/// the bilinear data second-order assembly needs. The calculus contracts
/// `grad_beta` with the direction's shared β̇ — atoms never see H⁺.
pub struct BetaChannel {
    /// ∂A/∂β̂ as a dense p-vector (e.g. the KKT residual r for the inner
    /// objective; the moment vector g_d for the #784 sampled atom).
    pub grad_beta: Array1<f64>,
}

/// A criterion term that owns its factorization and emits value and FROZEN
/// derivatives from that single internal state.
///
/// # Contract (the desync killers)
///
/// 1. `value()` and every `frozen_d*` MUST be projections of one internal
///    decomposition. If a derivative needs a second factorization, the term
///    is two atoms.
/// 2. `frozen_d1` is the partial at FIXED inner state. Atoms MUST NOT chain
///    through β̂ themselves — declare a [`BetaChannel`] and let the calculus
///    contract it with the shared β̇. (An atom with no inner-state
///    dependence — e.g. log|S|₊ — returns `None`.)
/// 3. Non-smooth machinery (rank thresholds, eigenframes, trust gates,
///    sampled splices) MUST be reflected in `stratum()` so the certifier
///    and the outer line search can distinguish strata boundaries from
///    bugs.
/// 4. Deleting the atom's legacy value+gradient code lands in the SAME
///    commit that ports it. No parallel layers.
pub trait CriterionAtom {
    /// Stable name, used by the certificate to indict a desyncing term.
    fn name(&self) -> &'static str;
    /// The term's value at the current (θ, β̂) state.
    fn value(&self) -> f64;
    /// Frozen partial ∂A/∂θ[dir] at fixed inner state.
    fn frozen_d1(&self, dir: &ThetaDirection) -> f64;
    /// Frozen second partial (directional). Default for atoms migrated
    /// gradient-first is to be ported in a later pass — second order keeps
    /// flowing through the existing assembly until then; there is NO
    /// approximate fallback inside the calculus.
    fn frozen_d2(&self, dir_i: &ThetaDirection, dir_j: &ThetaDirection) -> Option<f64>;
    /// The atom's exact ∂A/∂β̂ data, if it depends on the inner state.
    fn beta_channel(&self) -> Option<BetaChannel>;
    /// The smoothness stratum this atom's emissions are valid on.
    fn stratum(&self) -> Option<StratumFingerprint>;
}

/// The criterion as a fold over atoms, with the profiled chain rule applied
/// in exactly one place.
///
/// ```text
///   V(θ)        = Σ_a value(a)
///   DV[dir]     = Σ_a frozen_d1(a, dir) + ⟨ Σ_a grad_beta(a), β̇(dir) ⟩
/// ```
///
/// Note the β-channels SUM BEFORE the contraction: one solve-product per
/// direction for the whole criterion, not per atom — the chain rule is a
/// linear functional and the calculus exploits that; hand-distributed code
/// never could.
pub struct CriterionSum {
    pub atoms: Vec<Box<dyn CriterionAtom + Send + Sync>>,
}

impl CriterionSum {
    pub fn value(&self) -> f64 {
        self.atoms.iter().map(|a| a.value()).sum()
    }

    /// Profiled total directional derivative — THE chain rule, applied once.
    pub fn d1(&self, dir: &ThetaDirection) -> f64 {
        let frozen: f64 = self.atoms.iter().map(|a| a.frozen_d1(dir)).sum();
        let beta_dot = dir
            .beta_dot
            .as_ref()
            .expect("calculus must fill beta_dot before profiled d1");
        let mut chained = 0.0;
        for atom in &self.atoms {
            if let Some(channel) = atom.beta_channel() {
                chained += channel.grad_beta.dot(beta_dot.as_ref());
            }
        }
        frozen + chained
    }

    /// First-order optimality certificate (#934): per-atom FD audit at the
    /// optimum, ~2 evaluations per atom, naming the desyncing term. Probes
    /// that would cross a declared stratum boundary (rank change /
    /// collapsed eigengap within ±h) are refused, not reported as bugs.
    ///
    /// This is the structural replacement for the multi-week #901-style
    /// triangulation: the certificate output IS the diagnosis.
    pub fn certify(&self, _dir: &ThetaDirection, _h: f64) -> Vec<CertificateEntry> {
        // Sketch: for each atom, re-evaluate value() at θ±h·dir holding the
        // OTHER atoms' state fixed (each atom owns its factorization, so
        // per-atom re-evaluation is local), compare the centered difference
        // against frozen_d1 + its β-chain share, and emit a named verdict.
        // Refuse when stratum().kept_rank differs across ±h or the minimal
        // eigengap is below the frame-derivative floor.
        Vec::new()
    }
}

/// Per-atom certificate verdict — the unit of blame.
pub struct CertificateEntry {
    pub atom: &'static str,
    pub analytic: f64,
    pub finite_difference: f64,
    pub relative_error: f64,
    /// True when the probe was refused because ±h crosses a declared
    /// stratum boundary; `relative_error` is meaningless in that case.
    pub stratum_refusal: bool,
}

// ───────────────────────────────────────────────────────────────────────────
// Worked atom sketches — the three migration anchors.
// ───────────────────────────────────────────────────────────────────────────

/// Atom 1 (landed math, #901): the Hessian determinant term
/// `½ log|H_pen|₊` over `range(H_pen)`.
///
/// Internal state = the ONE spectral decomposition that
/// `intrinsic_hessian_pseudo_logdet_parts` already produces; the same
/// object IS the sensitivity kernel, so the determinant gradient and every
/// IFT solve share an inverse by construction.
///
/// - `value`      = Σ_{σ>thr} log σ (already the production value);
/// - `frozen_d1`  = ½ tr(H⁺ · Ḣ_frozen[dir]) via the spectral kernel —
///   exact on the constant-rank stratum for ANY drift, moving-subspace ψ
///   included (first-order eigenvector motion cancels);
/// - `beta_channel` = NONE — by design. The β̂-motion of H enters through
///   the direction's `h_dot_total` (the calculus adds `D_βH[β̇]` into the
///   SHARED drift before atoms see it), not through a per-atom chain. This
///   single decision removes the #901-layer-2 failure mode: there is no
///   second place a cubic correction can be (mis)assembled.
pub struct HessianLogdetAtom {
    pub sensitivity: Arc<Sensitivity>,
}

impl CriterionAtom for HessianLogdetAtom {
    fn name(&self) -> &'static str {
        "hessian_logdet"
    }
    fn value(&self) -> f64 {
        0.5 * self.sensitivity.logdet
    }
    fn frozen_d1(&self, dir: &ThetaDirection) -> f64 {
        let h_dot = dir
            .h_dot_total
            .as_ref()
            .expect("calculus fills h_dot_total before logdet d1");
        0.5 * self.sensitivity.kernel.trace_projected_logdet(h_dot)
    }
    fn frozen_d2(&self, _dir_i: &ThetaDirection, _dir_j: &ThetaDirection) -> Option<f64> {
        // −½ tr(H⁺ Ḣ_i H⁺ Ḣ_j) + ½ tr(H⁺ Ḧ_ij) via the same kernel
        // (trace_projected_logdet_cross_reduced); ported in the
        // second-order pass.
        None
    }
    fn beta_channel(&self) -> Option<BetaChannel> {
        None
    }
    fn stratum(&self) -> Option<StratumFingerprint> {
        Some(StratumFingerprint {
            kept_rank: self.sensitivity.stratum.kept_rank,
            min_relative_eigengap: self.sensitivity.stratum.min_relative_eigengap,
        })
    }
}

/// Atom 3 (the abstraction's hardest test, #784): the block-local sampled
/// marginal correction `−Δ_b`.
///
/// A SAMPLED atom: value from importance draws, derivatives from the
/// importance-weighted moments specified on
/// `block_sampled_marginal_correction` — and it fits the same trait with
/// no special cases:
///
/// - `frozen_d1`  = explicit penalty-score channel
///   PLUS `tr(Ḣ[dir] · (Q_b + Q_c))` — the draw-rescale and frame-rotation
///   channels collapsed into one rank-≤3m trace against the SHARED drift
///   (so this atom and the logdet atom cannot disagree about what
///   direction `dir` means: they trace the same matrix);
/// - `beta_channel` = the moment vector g_d = E_p[∂ΔF/∂β̂] — the calculus
///   charges ⟨g_d, β̇⟩ automatically, which is precisely the term the
///   current splice's "the envelope handles it" comment wrongly waves away;
/// - `stratum`    = the block eigenframe's minimal gap (Q_c divides by
///   λ_r − λ_q) and the trust-gate state: near-degenerate frames and gate
///   flips are declared boundaries, so the certifier refuses rather than
///   misdiagnoses, and the splice declines rather than clamps.
pub struct SampledBlockAtom {
    /// −Δ_b (cost-side sign already applied).
    pub value: f64,
    /// Explicit-channel gradient per packed θ coordinate (ρ entries only;
    /// ψ explicit channel is zero — its motion enters via Q_b/Q_c and g_d).
    pub explicit: Array1<f64>,
    /// `Q_b + Q_c`, symmetric rank ≤ 3m, built once from the sampler
    /// moments (M_r, R_r) and the block eigenpairs.
    pub q_bc: Arc<Array2<f64>>,
    /// Mode-motion moment `g_d = E_p[∂ΔF/∂β̂]`.
    pub g_d: Array1<f64>,
    pub stratum: StratumFingerprint,
}

impl CriterionAtom for SampledBlockAtom {
    fn name(&self) -> &'static str {
        "sampled_block_marginal"
    }
    fn value(&self) -> f64 {
        self.value
    }
    fn frozen_d1(&self, dir: &ThetaDirection) -> f64 {
        let explicit = match dir.index {
            Some(idx) if idx < self.explicit.len() => self.explicit[idx],
            _ => 0.0,
        };
        let h_dot = dir
            .h_dot_total
            .as_ref()
            .expect("calculus fills h_dot_total before sampled-block d1");
        // tr(Ḣ · Q_bc): same drift the logdet atom traces — shared meaning
        // of the direction is structural, not aspirational.
        let mut trace = 0.0;
        for i in 0..h_dot.nrows() {
            for j in 0..h_dot.ncols() {
                trace += h_dot[[i, j]] * self.q_bc[[j, i]];
            }
        }
        explicit + trace
    }
    fn frozen_d2(&self, _dir_i: &ThetaDirection, _dir_j: &ThetaDirection) -> Option<f64> {
        None
    }
    fn beta_channel(&self) -> Option<BetaChannel> {
        Some(BetaChannel {
            grad_beta: self.g_d.clone(),
        })
    }
    fn stratum(&self) -> Option<StratumFingerprint> {
        Some(StratumFingerprint {
            kept_rank: self.stratum.kept_rank,
            min_relative_eigengap: self.stratum.min_relative_eigengap,
        })
    }
}

// Atom 2 in the migration order is `penalty_logdet.rs` itself — it already
// satisfies the contract (one factorization → value + ρ/ψ/cross
// derivatives) and needs only the trait impl plus the deletion of its
// remaining call-site special-casing. The penalty quadratic
// `½ λ_k (β−μ_k)ᵀ S_k (β−μ_k)` is the simplest β-channel atom (frozen_d1 =
// the explicit ½λ_k quadratic; beta_channel = Sλ(β̂−μ) = the KKT residual's
// penalty half) and should be ported alongside the inner-objective atom so
// the envelope/noise-floor correction emerges from the calculus on day one.
