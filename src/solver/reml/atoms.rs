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
//!   influence Jacobian, the unified.rs correction traces, the large-scale
//!   marginal-slope route) become contractions of it and their private
//!   factorizations are DELETED (no parallel layers).
//!
//! ## Directional-first derivatives (#740 falls out)
//!
//! Atoms expose `frozen_d1(dir)` — and, once the second-order pass lands,
//! a directional `frozen_d2(dir_i, dir_j)` capability — never "the gradient
//! vector". The expensive moving data (β̇, Ẇ, Ḣ) is attached to the
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

use super::reml_outer_engine::PenaltySubspaceTrace;

/// An outer-coordinate direction with its induced inner motion, built ONCE
/// per direction by the calculus and shared by every atom.
///
/// This object is the cure for #901-layer-2: today each gradient consumer
/// assembles its own `Ḣ` (penalty drift + cubic IFT correction) and its own
/// `β̇`, and they disagree. Here the direction owns them; atoms borrow.
///
/// Channels are filled lazily by [`Sensitivity`] (β̇ needs the factored
/// solve; Ḣ_total needs β̇) so a value-only evaluation pays nothing.
///
/// Only the channels the LANDED first-order calculus reads live here:
/// `index` (unit θ-coordinate), `beta_dot` (the shared β̇), and
/// `h_dot_total` (the total drift Ḣ every atom traces). The further channels
/// the design names — a dense `dir` for general (non-unit) directions, and
/// the staged `s_dot` (∂Sλ/∂θ) / `h_dot_frozen` (∂H/∂θ at fixed β̂) inputs
/// from which the [`Sensitivity`] operator assembles `h_dot_total =
/// h_dot_frozen + D_βH[β̇]` — re-land as fields together with that operator
/// (#935), the code that fills AND reads them. Carrying them now would be
/// unread design surface (the same no-stub discipline this module applies to
/// its second-order and certify passes).
pub struct ThetaDirection {
    /// Coordinate index in the packed θ = (ρ‖ψ) layout, with the unit
    /// direction implied. (A dense general-direction channel re-lands with
    /// the #935 calculus that consumes it.)
    pub index: Option<usize>,
    /// Induced mode motion `β̇ = −H⁺ F_{βθ}[dir]`, the ONE chain-rule vector
    /// every atom's profiled derivative contracts against.
    pub beta_dot: Option<Arc<Array1<f64>>>,
    /// Total Hessian drift `Ḣ = ∂H/∂θ[dir]|_{β̂} + D_βH[β̇]` (the frozen
    /// penalty/design drift plus the cubic correction `Xᵀdiag(c ⊙ Xβ̇)X`
    /// applied to the SAME β̇ above), assembled once by [`Sensitivity`]. This
    /// is the matrix the logdet trace, the #784 Q_b/Q_c trace, and the θ-HVP
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
    // NOTE: the directional SECOND derivative (#740) is intentionally NOT a
    // method on this trait yet. Until an atom actually computes it from its
    // own factorization, an `-> Option<f64>` whose body is `None` is a stub —
    // banned, and precisely the desync-by-placeholder this module exists to
    // forbid (a half-emitted second order is the #901-layer-2 failure all
    // over again). It lands as a capability, not a stub: its own
    // `SecondOrderAtom` impl carrying a REAL `−½ tr(H⁺ Ḣ_i H⁺ Ḣ_j) +
    // ½ tr(H⁺ Ḧ_ij)` body, in the second-order pass. Second order keeps
    // flowing through the existing assembly until then, with NO approximate
    // fallback inside the calculus.
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

    // First-order optimality certificate (#934): a per-atom FD audit at the
    // optimum (~2 evaluations per atom) that names the desyncing term and
    // refuses probes crossing a declared stratum boundary (rank change /
    // collapsed eigengap within ±h). It lands with a REAL body — the per-atom
    // re-evaluation closure it requires does not exist yet — in the #934 pass,
    // not as a `Vec::new()` placeholder (a stub certifier that always
    // certifies is worse than none). The "## Self-certification" design above
    // is the spec; see the module Migration law for the port-with-real-body
    // discipline.
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
//
// ── Migration ledger ──────────────────────────────────────────────────────
//
// LANDED (atom 2, the single-factorization half): one original-frame
// `PenaltyPseudologdet` per evaluation point, shared through
// `EvalShared::penalty_pseudologdet_original` by the ρ-side criterion
// value/derivatives (eval.rs / runtime.rs) and the original-basis τ
// gradient-and-pair builders (hyper.rs). The three per-builder duplicate
// eigendecompositions are deleted; the ridge/positive-eigenspace threshold
// of `log|Sλ|₊` is decided exactly once. The transformed-frame pair
// callbacks factorize the canonical-TRANSFORMED (possibly constraint-
// projected) penalties — a different matrix, not a duplicate.
//
// LANDED (#934 sibling): `CriterionCertificate` FD-audits the value path
// against the analytic gradient at every returned optimum, so each further
// port inherits an end-to-end desync alarm even before its per-atom
// `certify` body exists.
//
// DELIBERATELY NOT FORCED: the penalty-quadratic VALUE stays
// `pirls_result.stable_penalty_term` (computed in the stable reparameterized
// basis) rather than being rewritten onto the gradient's per-coordinate
// `shifted_quadratic`. The two formulas are mathematically one atom, but the
// stable-basis evaluation exists because `βᵀSλβ` cancels catastrophically in
// the original basis at large λ — unifying the source text would trade a
// certified value/gradient pair for worse numerics. That pair is covered by
// the #934 certificate until the quadratic atom can own a stable-basis
// emission for BOTH channels.
//
// LANDED (pass 2, the ThetaDirection shared-drift pass — the β̇ kernel
// half): `ThetaModeResponseKernel` in unified.rs is now the ONE place the
// IFT mode-response kernel selection lives (lifted constrained
// `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S` under active inequality
// constraints; full `H⁻¹` otherwise, projection on the trace side only).
// Converted to contractions of it: the gradient solve stack in
// `reml_laml_evaluate`, the ρ- and ext-coordinate standalone fallbacks in
// `compute_outer_hessian` (which now share ONE lazily-built kernel per
// Hessian evaluation instead of two independent Schur factorizations), and
// the standalone fallback in `build_outer_hessian_operator` (which also
// stops building the constrained kernel on the production precomputed
// path, where it was unused work). The four hand-copied selection rules —
// each carrying a comment warning the others to "mirror the selection
// exactly" — are deleted; gradient, dense Hessian, and operator Hessian
// structurally cannot pick different inverses for the same evaluation
// point (the dβ̂/dθ half of #901-layer-2's per-consumer drift). Per-atom
// certify body: `certify_tangency` audits every constrained emission
// against the defining invariant `A_act·v = 0` on the `[CERTIFICATE]`
// stream (#934 pattern; the unconstrained arm is covered end-to-end by
// `CriterionCertificate`'s FD audit at every optimum). Bit-identity pin:
// `theta_mode_response_kernel_matches_preport_assembly_bitwise` reproduces
// the pre-port per-site assemblies inline and asserts bitwise-equal
// emissions in both regimes plus the masked-ρ and subspace-without-
// constraints edges.
//
// DELIBERATELY NOT PORTED (pass 2 non-duplicates): `compute_adjoint_z_c`
// keeps its bare `K_S.apply_pseudo_inverse` route under a penalty subspace
// WITHOUT active constraints — that is the TRACE-side adjoint (z_c must
// contract against the same kernel as the leverage h^{G,proj}; see its
// comment block), a different convention from the IFT mode response, not a
// missed copy of the selection rule. The per-site solve SHAPES
// (`respond_one` single-RHS vs `respond_stack` batched) also stay
// distinct on purpose: GEMV-per-column and blocked GEMM sum in different
// orders, so collapsing them would break bit-identity with the pre-port
// assemblies. The remaining ThetaDirection channels — caching β̇/Ḣ_total
// on the direction object and replacing the per-consumer Ḣ assemblies in
// the trace branches — land with the #935 Sensitivity operator pass that
// fills AND reads them.

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Per-atom isolation check the Migration law demands: build the landed
    /// #901 `HessianLogdetAtom` anchor from a hand-chosen spectral kernel and
    /// confirm its `value` / `frozen_d1` emissions are exactly the closed-form
    /// `½ log|H_pen|₊` and `½ tr(H_pen⁺ Ḣ)`, then confirm the `CriterionSum`
    /// fold assembles the profiled total derivative `Σ frozen_d1 + ⟨Σ ∂_βA,
    /// β̇⟩` from those emissions plus one shared β̇ contraction.
    ///
    /// `H_pen⁺` is taken diagonal in the identity basis (`u_s = I`,
    /// `h_proj_inverse = diag(1/σ)`) so every quantity is verifiable by hand:
    /// `tr(H⁺ A) = Σ_a A_aa / σ_a`. This is the same spectral object
    /// `intrinsic_hessian_pseudo_logdet_parts` emits, so the test pins the
    /// atom's contract against the production kernel, not a re-derivation.
    #[test]
    pub(crate) fn hessian_logdet_atom_emits_closed_form_value_and_directional_derivative() {
        // σ = (2, 4) ⇒ H⁺ = diag(1/2, 1/4), log|H|₊ = ln 2 + ln 4 = ln 8.
        let kernel = Arc::new(PenaltySubspaceTrace {
            u_s: array![[1.0, 0.0], [0.0, 1.0]],
            h_proj_inverse: array![[0.5, 0.0], [0.0, 0.25]],
        });
        let stratum = StratumFingerprint {
            kept_rank: 2,
            // smallest relative gap (4 − 2)/4 = 0.5 — well clear of any frame
            // floor, so this evaluation lives on a single constant-rank stratum.
            min_relative_eigengap: 0.5,
        };
        let sensitivity = Arc::new(Sensitivity {
            kernel: kernel.clone(),
            logdet: 8.0_f64.ln(),
            stratum: StratumFingerprint {
                kept_rank: stratum.kept_rank,
                min_relative_eigengap: stratum.min_relative_eigengap,
            },
        });
        let hess = HessianLogdetAtom {
            sensitivity: sensitivity.clone(),
        };

        // value = ½ log|H|₊.
        assert_eq!(hess.name(), "hessian_logdet");
        assert!((hess.value() - 0.5 * 8.0_f64.ln()).abs() < 1e-12);
        assert!(
            hess.beta_channel().is_none(),
            "logdet atom has no β-channel"
        );
        assert_eq!(hess.stratum().expect("declared stratum").kept_rank, 2);

        // Shared drift Ḣ = [[1, 0.3], [0.3, 1]]. frozen_d1 = ½ tr(H⁺ Ḣ)
        //   = ½ (1/2 · 1 + 1/4 · 1) = ½ · 0.75 = 0.375.
        let h_dot = Arc::new(array![[1.0, 0.3], [0.3, 1.0]]);
        let dir = ThetaDirection {
            index: Some(0),
            beta_dot: Some(Arc::new(array![0.5, 0.5])),
            h_dot_total: Some(h_dot.clone()),
        };
        assert!((hess.frozen_d1(&dir) - 0.375).abs() < 1e-12);

        // Sampled atom (#784): frozen_d1 = explicit[idx] + tr(Ḣ · Q_bc);
        // β-channel = g_d. With explicit[0] = 0.2 and symmetric
        // Q_bc = [[0.5, 0.1], [0.1, 0.3]]: tr(Ḣ Q_bc) = 1·0.5 + 0.3·0.1 +
        // 0.3·0.1 + 1·0.3 = 0.86, so frozen_d1 = 1.06.
        let sampled = SampledBlockAtom {
            value: -0.4,
            explicit: array![0.2, -0.1],
            q_bc: Arc::new(array![[0.5, 0.1], [0.1, 0.3]]),
            g_d: array![1.0, -2.0],
            stratum,
        };
        assert!((sampled.value() - (-0.4)).abs() < 1e-12);
        assert!((sampled.frozen_d1(&dir) - 1.06).abs() < 1e-12);
        assert!(
            (sampled
                .beta_channel()
                .expect("sampled atom declares a β-channel")
                .grad_beta
                .dot(&array![0.5, 0.5])
                - (-0.5))
                .abs()
                < 1e-12
        );

        // CriterionSum fold: value sums, and the profiled d1 adds ONE shared
        // β̇ contraction of the SUMMED β-channels (here only the sampled atom
        // contributes g_d). β̇ = [0.5, 0.5] ⇒ ⟨g_d, β̇⟩ = 0.5 − 1.0 = −0.5.
        //   value = ½ ln 8 − 0.4
        //   d1    = 0.375 + 1.06 + (−0.5) = 0.935
        let sum = CriterionSum {
            atoms: vec![Box::new(hess), Box::new(sampled)],
        };
        assert!((sum.value() - (0.5 * 8.0_f64.ln() - 0.4)).abs() < 1e-12);
        assert!((sum.d1(&dir) - 0.935).abs() < 1e-12);
    }
}
