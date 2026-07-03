use super::*;

pub(crate) const SAE_MANIFOLD_ARMIJO_C1: f64 = 1.0e-4;

pub(crate) const SAE_MANIFOLD_MAX_LINESEARCH_HALVINGS: usize = 12;

/// Relative Cholesky-pivot floor for the analytic SAE outer-rho gradient.
///
/// The evidence value can still be honest below this threshold because it only
/// sums `log(diag(L))`. The analytic gradient is different: selected-inverse
/// traces and `ArrowFactorCache::full_inverse_apply` divide by those pivots.
/// Once `min_pivot / max_pivot` is below this floor, the gradient lane must
/// either identify a closed-form gauge orbit and stiffen only that quotient
/// direction, or reject the trial rho as numerically singular.
pub(crate) const SAE_OUTER_GRADIENT_PIVOT_RATIO_FLOOR: f64 = 1.0e-12;

pub(crate) const SAE_OUTER_GRADIENT_GAUGE_RAYLEIGH_FACTOR: f64 = 1.0e-8;

/// Relative spectral cutoff below which a penalised decoder β-curvature
/// eigenvalue (`G_k + λ_smooth·S_k`) is treated as a genuine flat direction of
/// the joint inner Hessian — the rank-deficient-decoder null quotiented out of
/// the inner convergence measure and deflated in the outer gradient (#1051).
/// Matches the `1e-9` relative rank cutoff used across the codebase.
pub(crate) const SAE_DECODER_BETA_NULL_RELATIVE_FLOOR: f64 = 1.0e-9;

/// Nominal curvature-homotopy `η` step (#1007): the tracker covers `η ∈ [0, 1]`
/// in this many equal predictor-corrector waypoints when the branch is clean.
/// Five waypoints is a few corrector solves — far cheaper than the multi-seed
/// cascade it replaces — and the step is halved adaptively when the arrow-factor
/// min pivot shrinks, so a near-bifurcation stretch is resolved at finer
/// granularity without a separate knob.
pub(crate) const CURVATURE_WALK_INITIAL_ETA_STEP: f64 = 0.2;

/// Smallest curvature-homotopy `η` step (#1007). A pivot collapse (or corrector
/// failure) that persists at this step is a DETECTED branch bifurcation, not a
/// step-size artifact: the walk records it and defers to the seed cascade.
pub(crate) const CURVATURE_WALK_MIN_ETA_STEP: f64 = 1.0 / 256.0;

/// Hard ceiling on accepted corrector solves in one curvature-homotopy walk
/// (#1007). Bounds the walk's cost under repeated halving; reaching it is a
/// structural-termination signal (the branch is not cleanly trackable) that
/// defers to the cascade, never a spin.
pub(crate) const CURVATURE_WALK_MAX_CORRECTORS: usize = 32;

/// Joint Newton iteration budget for the curvature-walk degenerate-basin
/// recovery (#1117). When the walk lands on a sub-arrival-floor (see the
/// `arrival_floor` in `run_curvature_homotopy_entry_at_rho`: the achievable
/// linear ceiling `anchor_ev` minus one atom's share) reconstruction, the
/// recovery runs a REAL joint fit from the pristine
/// (circle-aware) seed with at least this many inner iterations, independent of
/// the outer objective's possibly-frozen `inner_max_iter = 0` — otherwise the
/// recovery (and the fallback cascade) would re-freeze at the cold seed and
/// never escape the linear basin. Matches the cold-start budget that recovers
/// EV ≈ 0.94 on the K = 1 periodic circle.
pub(crate) const CURVATURE_WALK_RECOVERY_INNER_ITERS: usize = 25;

/// Relative floor on the Newton directional decrease, expressed as a tiny
/// multiple of `‖g‖·‖Δ‖`. A predicted decrease below this is at the level of
/// f64 round-off in the quadratic model and is treated as no progress (the step
/// is rejected). Scaling by the gradient/step norms makes the floor invariant
/// to the problem's overall magnitude.
pub(crate) const SAE_MANIFOLD_DIRECTIONAL_DECREASE_REL_FLOOR: f64 = 1.0e-14;

/// Row count at or above which the fused SAE reconstruction data-fit
/// (`loss_scaled`) fans its per-row decode + residual reduction out over
/// rayon. Below this the single-threaded fused pass is cheaper than the
/// fan-out; matched in spirit to the arrow-Schur `SCHUR_MATVEC_PARALLEL_ROW_MIN`
/// gate so short batches inside an outer fan-out stay sequential (#1017).
pub(crate) const SAE_LOSS_PARALLEL_ROW_MIN: usize = 64;

/// Relative tolerance on the undamped Newton step norm (scaled by the iterate
/// scale) for accepting inner-solve convergence.
pub(crate) const SAE_MANIFOLD_INNER_STEP_REL_TOL: f64 = 1.0e-4;

/// Relative tolerance on the KKT gradient norm (scaled by the iterate scale) for
/// accepting inner-solve convergence.
pub(crate) const SAE_MANIFOLD_INNER_GRAD_REL_TOL: f64 = 1.0e-5;

/// Relative per-refine-round penalised-objective decrease below which the inner
/// solve is treated as having reached its numerical fixed point (#1051). On an
/// ill-conditioned penalised bilinear fit the KKT gradient and undamped step
/// stay above tolerance while the objective stops moving; this `√εmach`-scale
/// floor recognises that stalled iterate as the converged inner optimum instead
/// of grinding the refine budget to the `1e12` infeasible sentinel.
pub(crate) const SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL: f64 = 1.0e-8;

/// Fraction of the total since-entry objective reduction below which a refine
/// round's contribution is treated as cosmetic flat-valley crawl (#1051), so the
/// inner solve is accepted as numerically converged. At `1e-4` the inner fit has
/// captured ≥ 99.99% of the achievable penalised-objective reduction before the
/// criterion is ranked — far past the point where further crawl can change the
/// Laplace evidence, yet strict enough that a materially-improving fit refines on.
pub(crate) const SAE_MANIFOLD_INNER_OBJECTIVE_STALL_FRACTION: f64 = 1.0e-4;

/// Minimum completed refine rounds before the objective-stagnation fixed point
/// may be accepted (#1051). Enough rounds to establish a meaningful
/// total-improvement baseline for the fraction test, but far below the full
/// refine budget — terminating the ill-conditioned crawl early is the goal.
pub(crate) const SAE_MANIFOLD_INNER_OBJECTIVE_STALL_MIN_ROUNDS: usize = 3;

/// Above this full-`B` β width, dense beta-penalty curvature is never
/// materialized when Grassmann frames are engaged; exact curvature is probed
/// directly in the factored coordinate space instead.
pub(crate) const SAE_DENSE_BETA_PENALTY_PROBE_MAX_DIM: usize = 4096;

/// Relative spectral cutoff for counting the numerical rank / nullity of a
/// symmetric penalty Gram: eigenvalues at or below `cutoff · λ_max` are treated
/// as zero. Shared by [`SaeManifoldTerm::symmetric_rank`] and
/// [`smooth_penalty_nullity`] so the two stay in lockstep.
pub(crate) const SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF: f64 = 1.0e-9;

/// Floor on the Levenberg-Marquardt ridge added to a per-row Hessian before
/// Cholesky, so the first attempt is always strictly positive even when the
/// caller passes a zero base ridge.
pub(crate) const SAE_MANIFOLD_ROW_RIDGE_FLOOR: f64 = 1.0e-12;

/// Multiplicative factor by which the LM ridge is escalated after a failed
/// Cholesky factorisation of a per-row Hessian.
pub(crate) const SAE_MANIFOLD_ROW_RIDGE_GROWTH: f64 = 10.0;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SaeBetaPenaltyAssembly {
    pub(crate) dense_written: bool,
    pub(crate) deferred_factored: bool,
}

impl SaeBetaPenaltyAssembly {
    pub(crate) fn record_curvature(&mut self, dense_beta_curvature: bool) {
        if dense_beta_curvature {
            self.dense_written = true;
        } else {
            self.deferred_factored = true;
        }
    }
}

/// ABSOLUTE "worse than a constant predictor" explained-variance floor (#1023),
/// used by the curved-ARRIVAL quality gate and the inner-fit incumbent-restore
/// gate as the point below which a CONVERGED fit is degenerate whatever the data's
/// achievable ceiling. `0.10` is a deliberately conservative "must explain at least
/// a tenth of the variance or it is structurally degenerate" floor for those
/// post-convergence quality checks; it is NOT a tuned operating point.
///
/// S1 (guard surgery) — this is NO LONGER used by the collapse DETECTOR: the
/// co-collapse verdict and reseed arm now key on the signal-free null floor
/// ([`super::outer_objective::absolute_degeneracy_ev_floor`] = `q / n`), not on any
/// fraction of a dense PCA ceiling. This constant survives only where a fixed
/// "worse than the mean" reference is genuinely wanted (arrival / incumbent gates).
pub(crate) const SAE_FIT_DATA_COLLAPSE_EV_FLOOR: f64 = 0.10;

/// #1189/#1217/#1610 — the finite BFGS INFEASIBILITY WALL substituted for a
/// non-finite / collapsed REML criterion so the outer line search REJECTS the
/// step and backtracks toward the feasible basin instead of aborting the whole
/// outer solve (the `opt` BFGS treats a non-finite probe as "line search failed"
/// and gives up at the current iterate — observed on real OLMo K=2, stall at
/// iter 2 with |g|≈65; see [`super::outer_objective`]). It is deliberately a
/// FINITE wall, not `∞`: an `∞` probe is non-differentiable and un-backtrackable,
/// whereas a large finite value presents as an ordinary infeasible step. The
/// magnitude is NOT a tuned operating point — it is a sentinel chosen only to
/// DOMINATE any realizable REML criterion for a non-collapsed SAE fit (which is
/// `O(N·p·log σ²)` — tens to thousands in magnitude) by many orders, so a
/// collapsed/infeasible probe always loses the line-search comparison. `1e12`
/// clears that with headroom while staying far below `f64::MAX` (the sum path
/// clamps to `2·wall` so a near-`f64::MAX` finite base can never overflow to
/// `∞`). The non-collapsed, finite-cost path is byte-for-byte unchanged.
pub(crate) const SAE_FIT_DATA_COLLAPSE_COST: f64 = 1.0e12;

/// #1026 — reconstruction EV-improvement / EV-degradation tolerance for the
/// inner-fit reconstruction INCUMBENT. A new inner iterate is recorded as the
/// best-reconstruction state only when its explained variance improves by more
/// than this; the pre-degradation incumbent is restored only when EV degrades by
/// more than this ([`super::fit_drivers`], [`super::outer_objective`]). EV is a
/// dimensionless quantity in `[0,1]`, so this is a scale-invariant "0.1% of
/// variance" negligibility tolerance (the SAME dimensionless negligibility point
/// as [`crate::hybrid_split`]'s collapse gate): it debounces round-off-level EV
/// wobble between iterates from a material EV move, without a corpus-tuned
/// magnitude. Where it instead gates an OBJECTIVE (not an EV) it is applied
/// scale-relative as `TOL·(1 + |objective|)` (`fit_drivers.rs` accept floor), so
/// the same constant reads as a relative tolerance on that scale-bearing
/// quantity.
pub(crate) const SAE_FINAL_EV_DEGRADATION_TOL: f64 = 1.0e-3;

/// #1026 — per-row coordinate re-projection grid resolution for the final
/// reconstruction polish (matches the resolution the curvature-homotopy arrival
/// polish uses in `outer_objective.rs`). The projection snaps each row's latent
/// coordinate to the nearest point of a `resolution`-point decode grid, so a
/// finer grid is a tighter chart re-fit; `256` is the established cost/precision
/// point for the chart re-projection (a circle/sphere/Euclidean grid sweep).
pub(crate) const SAE_FINAL_DECODER_POLISH_PROJECTION_RESOLUTION: usize = 256;

pub(crate) const SAE_SEED_DISPERSION_FLOOR: f64 = 1.0e-12;

/// #1026/#1610 decoder-repulsion conditioner strength as a DIMENSIONLESS ratio of
/// the primary separation-barrier strength. The collinearity-gated cross-decoder
/// repulsion injects POSITIVE curvature in the inter-atom co-collapse direction.
/// It is a conditioner/separator, not a primary objective term: it is identically
/// zero (value, gradient, curvature) unless two atom decoders exceed
/// [`SAE_DECODER_REPULSION_COLLINEARITY_GATE`], so it is a strict no-op for
/// well-separated atoms and for `K = 1`. Without it the SAE joint inner Newton
/// solve has NO inter-atom repulsion (data-fit is independent per atom given the
/// soft assignment), so at `K >= 4` on real residual geometry two atoms drift
/// onto the same decoder direction, the per-row `H_tt` block goes near-singular,
/// the reduced β-Schur over-subtracts and goes indefinite, and the inner solve
/// never converges (#1026).
///
/// #1610 — the previous value was an absolute `1e-3` whose magnitude was NOT
/// derived: it was four orders below the separation barrier (`10.0`) with no
/// stated relationship, and it weighted the un-normalized cross-Gram energy
/// `‖B_jB_kᵀ‖²_F = c_jk²·‖B_j‖²_F·‖B_k‖²_F`, so the repulsion's force on the
/// collinearity `c_jk²` scaled with the decoder energy `‖B_j‖²_F·‖B_k‖²_F` while
/// the separation barrier's force on the SAME `c_jk²` was scale-free. The `1e-3`
/// was therefore implicitly absorbing the decoder-energy units at one assumed
/// corpus scale — exactly the unprincipled, non-generalizing hand-pick #1610
/// flags. The fix (see [`super::penalties::SaeManifoldTerm::refresh_decoder_repulsion_gate`])
/// normalizes the per-pair weight by `‖B_j‖²_F·‖B_k‖²_F`, so the repulsion now
/// penalizes the SAME dimensionless collinearity `c_jk² ∈ [0,1]` the separation
/// barrier does. With both terms acting on the dimensionless `c_jk²`, the
/// repulsion strength is a pure dimensionless ratio of the barrier strength:
///
///   `μ_rep = SAE_DECODER_REPULSION_BARRIER_RATIO · μ_sep`.
///
/// The repulsion is a SUBDOMINANT conditioner — the separation barrier is the
/// primary anti-collapse cure (it DIVERGES at `c²→1`; the repulsion is a finite
/// PSD quadratic). The ratio `1e-4` keeps the repulsion four orders below the
/// barrier BY DESIGN (so it conditions the indefinite directions without fighting
/// the barrier's restoring force), and at the canonical unit decoder scale
/// (`‖B_k‖²_F ≈ 1`) the effective per-pair weight `μ_sep·1e-4 = 10·1e-4 = 1e-3`
/// reduces EXACTLY to the historical absolute constant, so unit-scale fits are
/// byte-unchanged. Unlike the old absolute `1e-3` this engages identically at any
/// corpus scale.
pub(crate) const SAE_DECODER_REPULSION_BARRIER_RATIO: f64 = 1.0e-4;

// The derived decoder-repulsion strength is a dimensionless fraction
// [`SAE_DECODER_REPULSION_BARRIER_RATIO`] of the data-derived (possibly
// runtime-overridden) separation-barrier strength μ_C; it is computed on the
// term itself (it needs the dictionary's overcompleteness, not a global
// constant) — see [`super::penalties::SaeManifoldTerm::decoder_repulsion_strength`].

/// #1026 normalized collinearity score
/// `s_jk = ‖B_jB_kᵀ‖²_F / (‖B_j‖²_F·‖B_k‖²_F)` at/above which the decoder
/// repulsion engages. Below this the smoothstep gate is exactly 0 (zero penalty
/// / gradient / curvature), so healthy fits whose atoms point in distinct
/// directions never activate the term. `s_jk` is scale-free in each decoder's
/// magnitude (it measures ANGLE, not norm) so it cannot fight the data-fit's
/// choice of decoder scale.
pub(crate) const SAE_DECODER_REPULSION_COLLINEARITY_GATE: f64 = 0.5;

/// Relative-mass floor defining when two atoms genuinely CO-FIRE on a row, used
/// by the anti-collapse co-activation scan (`barrier_coactive_pairs`). An atom
/// whose assignment mass on a row is below this fraction of that row's peak mass
/// is treated as NOT firing there: it contributes `≤ floor·peak` reconstruction
/// mass, so the co-activation it would register and the anti-collapse repulsion /
/// separation it would receive are negligible.
///
/// For structurally sparse assignments (JumpReLU hard gate, IBP-MAP) the surviving
/// active atoms sit far above this floor and the hard zeros are excluded anyway,
/// so the co-active support is unchanged. It is load-bearing for SOFTMAX, whose
/// normalization gives EVERY atom a tiny but strictly nonzero tail mass: a plain
/// `a ≠ 0` test would mark all `K` atoms co-active on every row and collapse the
/// scan to the dense `O(N·K²)` all-pairs cost (minutes at `K = 10⁴`). Matches the
/// `1e-3` relative cutoff the compact row layout uses (`from_dense_weights`), so
/// the penalty support and the assembled Newton support stay consistent.
pub(crate) const SAE_COACTIVE_RELATIVE_MASS_FLOOR: f64 = 1.0e-3;

// ── #1026 / #1522 interior-point COLLAPSE-PREVENTION barriers ────────────────
//
// The collinearity-gated `SAE_DECODER_REPULSION_*` term above is a SEPARATOR for
// already-large, already-collinear decoders; it is a strict no-op at the
// principal failure state (a decoder with norm → 0) because a zero decoder has
// no direction to be collinear with. The two barriers below are the actual
// anti-collapse core: they are interior-point log-barriers that DIVERGE at the
// collapse boundary, so the inner Newton can never reach it.

// #1026/#1522 — RUNTIME separation-barrier-strength override. Read through the
// accessor below so a SINGLE compiled wheel can sweep μ_sep from Python
// (`set_sae_barrier_overrides`) without recompiling. A quiet-NaN sentinel means
// "unset → derive μ_C from the problem" (the data-derived overcompleteness
// ratio), so 0.0 remains a legitimate swept value (barrier disabled). The amplitude
// (keep-alive) barrier was removed: an over-complete dictionary's surplus
// features SHOULD die, and a dead atom's decoder block is parked into a
// well-conditioned state by the inner per-row Tikhonov ridge — forcing the
// norm away from zero only over-inflated healthy atoms and fought that natural
// death. So there is no amplitude strength or active-atom gate to override.
static SAE_SEP_STRENGTH_OVERRIDE_BITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0x7ff8_0000_0000_0000);
/// #1026/#1522/#1610 — read the process-global separation-barrier-strength
/// override, or `None` when unset. `None` ⇒ the evidence-derived per-pair μ_jk
/// (the data-fit inseparability strength from
/// [`super::penalties::SaeManifoldTerm::barrier_pair_strength_with_gates`]) is
/// used. A quiet-NaN sentinel means "unset → derive from the problem"; `0.0`
/// stays a legitimate swept value (barrier disabled).
pub(crate) fn sae_separation_barrier_override() -> Option<f64> {
    let v =
        f64::from_bits(SAE_SEP_STRENGTH_OVERRIDE_BITS.load(std::sync::atomic::Ordering::Relaxed));
    if v.is_nan() { None } else { Some(v) }
}

/// Set the process-global SAE separation-barrier strength override (one wheel,
/// many configs). `sep_strength` is NaN to clear the override back to the
/// evidence-derived per-pair μ_jk (`γ_jk / (1 - γ_jk)`, the data-fit
/// inseparability strength — see
/// [`super::penalties::SaeManifoldTerm::barrier_pair_strength_with_gates`]); there
/// is no compiled-constant default any more.
/// The amplitude (keep-alive) barrier and its active-atom gate were removed
/// (surplus features are allowed to die into a ridge-parked state), so this
/// takes only the separation strength. Called from the gamfit Python FFI.
///
/// CONCURRENCY: this is a PROCESS-GLOBAL atomic, so it is NOT safe to use across
/// concurrent in-process fits — a parallel candidate/rung/layer sweep that sets
/// different strengths in different threads will leak the override between fits
/// (last writer wins for all readers). It is intended for a single-fit-at-a-time
/// CLI/notebook sweep where the process runs one configuration at a time. The
/// proper fix is to migrate this to a per-fit configuration field threaded
/// through the solver rather than a global; that refactor is out of scope here.
pub fn set_sae_barrier_overrides(sep_strength: f64) {
    SAE_SEP_STRENGTH_OVERRIDE_BITS
        .store(sep_strength.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

// #1026/#1522/#1610 — the SEPARATION barrier strength is NO LONGER a hand-picked
// absolute constant (it was `10.0`, matched to no problem scale), nor the
// overcompleteness rank ratio `Σ min(M_k,p)/min(n,p)` (a geometry heuristic).
// It is now EVIDENCE-DERIVED PER PAIR from the reconstruction objective: the
// data-fit inseparability `γ_jk` (largest canonical correlation of the two atoms'
// coactivation-weighted chart designs — the quantity that governs whether the
// joint inner Laplace/REML Hessian stays PD) sets `μ_jk = γ_jk/(1-γ_jk)`. See the
// full derivation on
// [`super::penalties::SaeManifoldTerm::barrier_pair_strength_with_gates`] (the
// penalty form `P_sep = Σ μ_jk q_jk [-log(1-c²+ε)]` on the normalized decoder
// shapes is documented on `separation_barrier_value`). The runtime override
// (`set_sae_barrier_overrides`) still takes precedence over the derived value.

/// #1026/#1522 SEPARATION barrier softening `ε` in `log(1 - c_jk² + ε)`. Bounds
/// the barrier (and its PSD majorizer) at the exact-alignment limit `c_jk² = 1`.
pub(crate) const SAE_SEPARATION_BARRIER_EPS: f64 = 1.0e-6;

/// #1625 — normalized collinearity `c_jk² = ‖U_jU_kᵀ‖²_F ∈ [0,1]` at/above which
/// the SEPARATION barrier engages, the analog of
/// [`SAE_DECODER_REPULSION_COLLINEARITY_GATE`] for the primary anti-collapse
/// barrier. A C1 smoothstep ramps the barrier from exactly 0 at this threshold to
/// full strength (and the interior-point divergence) as `c_jk² → 1`, so the
/// barrier is a genuine COLLAPSE-prevention force — active only once two coactive
/// atoms are materially aligned — rather than a global orthogonality prior that
/// taxes every distinct-but-correlated pair.
///
/// WHY a gate at all (the ungated `−log(1−c²+ε)` was the #1625 stall): the
/// interior-point shape has force `∂P/∂c² = 1/(1−c²+ε) ≈ 1` even at MODERATE
/// collinearity, so two genuinely-distinct atoms (e.g. `c² = 0.36`, a 53° angle —
/// nowhere near the `c² → 1` collapse) feel an O(1) separating force that, on a
/// well-specified fit whose data residual is near zero, DOMINATES the objective and
/// drags the decoders off the data optimum. The inner (t, β) Newton then has to
/// chase a barrier-shifted optimum it converges to only slowly, and
/// `reml_criterion`'s undamped-PD inner solve never reaches KKT stationarity —
/// the #1625 "inner solve did not converge" refusal. Gating the barrier to the
/// near-collapse regime (`c² ≳ 0.5`) leaves well-separated dictionaries at their
/// data optimum (so they converge) while preserving the divergent restoring force
/// exactly where collapse happens (`c² → 1`).
///
/// Chosen equal to the repulsion gate (`0.5`): the two anti-collapse terms then
/// engage on the same near-collinear pair set, the barrier as the divergent
/// interior-point core and the repulsion as its subdominant conditioner. Pairs
/// below `0.5` (≥ 45° apart) are not collapsing and need no anti-collapse force.
pub(crate) const SAE_SEPARATION_BARRIER_COLLINEARITY_GATE: f64 = 0.5;

/// #1026/#1522/#1610 RELATIVE decoder-norm floor below which an atom is treated
/// as inactive / shape-undefined for the SEPARATION barrier (its `U_k` is
/// ill-conditioned). The separation barrier abstains for a pair until the
/// decoder norm is lifted above the floor.
///
/// #1610 — this is a RELATIVE fraction of the live dictionary's largest decoder
/// energy (`max_k ‖B_k‖_F`), NOT an absolute magnitude. The previous absolute
/// `1e-6` floor was NOT scale-invariant: the SAE decoders inherit the scale of
/// the activations being modeled, so under a global rescaling of the corpus by a
/// factor `s` every decoder norm scales by `s` while the absolute floor stayed
/// fixed — on a corpus whose natural decoder scale is below `1e-6` the barrier
/// spuriously abstained on EVERY pair (collapse prevention silently disabled),
/// and on a corpus scaled far above it the floor was inert. Keying the floor to
/// `max_k ‖B_k‖²_F` makes the abstain set scale-free: it depends only on the
/// RATIO of an atom's energy to the dictionary's, so collapse prevention engages
/// identically regardless of the corpus scale. At the canonical unit decoder
/// scale (`max_k ‖B_k‖²_F ≈ 1`) it reduces to the historical `1e-6` floor.
/// Consumed via [`super::penalties::barrier_norm_floor_sq`], the single source
/// for both the barrier value and its gradient/curvature, so the line-search
/// value never desyncs from the step.
pub(crate) const SAE_BARRIER_ACTIVE_NORM_REL_FLOOR: f64 = 1.0e-6;

/// Full SAE-manifold term.
#[derive(Debug)]
pub struct SaeManifoldTerm {
    pub atoms: Vec<SaeManifoldAtom>,
    pub assignment: SaeAssignment,
    pub(crate) temperature_schedule: Option<GumbelTemperatureSchedule>,
    /// Active-set row layout from the most recent `assemble_arrow_schur` call.
    /// `None` for dense modes (Softmax / IBPMap) or when not yet assembled.
    pub(crate) last_row_layout: Option<SaeRowLayout>,
    /// The single provenance-carrying per-row inner product (Object 2). The
    /// reconstruction likelihood whitens residuals through it and the isometry
    /// gauge's [`gam_terms::analytic_penalties::WeightField`] is constructed
    /// from the same object, so a likelihood-metric ≠ gauge-metric state is
    /// unrepresentable. `None` ⇒ Euclidean / isotropic (magic-by-default: the
    /// metric is selected by whether per-row Fisher factors were installed, not
    /// by a flag), which is bit-for-bit the historical isotropic `φ̂` path.
    pub(crate) row_metric: Option<gam_problem::RowMetric>,
    /// #976 Layer-1 guard ledger for the most recent joint fit: every
    /// active-mass breach with the action taken (re-seed / terminal). Cleared
    /// at the start of each `run_joint_fit_arrow_schur`; read post-fit via
    /// [`SaeManifoldTerm::collapse_events`] and carried onto the
    /// structure-search [`gam_solve::structure_search::SearchLedger`].
    pub(crate) collapse_events: Vec<CollapseEvent>,
    /// Per-row **design honesty weights** (#991): Horvitz–Thompson inclusion
    /// corrections from a designed corpus subsample
    /// ([`gam_solve::row_sampling_measure::RowSamplingMeasure::designed_subsample`] /
    /// [`crate::corpus::designed_target`]), self-normalized to
    /// mean `1.0` over the term's rows so dispersion, dof, and the
    /// data-vs-penalty balance stay consistent at the fitted sample size while
    /// the design's selection bias is removed (oversampled loud rows are
    /// downweighted back).
    ///
    /// The weights enter the objective as a per-row scalar metric `w_i · I_p`
    /// on the reconstruction channel ONLY, realized as a `√w_i` scaling of the
    /// per-row residual, latent Jacobian, and β basis load at their single
    /// construction sites in the assembly — so the data-fit value, the t-block
    /// Gauss-Newton, the β gradient/Gram, and the cross blocks all carry
    /// exactly one factor of `w_i` and cannot desync (the same discipline as
    /// the #974 whitening seam). Per-row latent priors (assignment prior, ARD
    /// coordinate prior) are deliberately NOT weighted: included rows' latent
    /// states are genuine model components of the subsampled model,
    /// conditional on inclusion; the HT correction applies to the row
    /// *evidence* about shared structure (decoder β, ρ), not to the latent
    /// priors. `None` ⇒ the exact unweighted path, bit-for-bit.
    pub(crate) row_loss_weights: Option<Vec<f64>>,
    /// #972 / #977 T1: whether the MOST RECENT `assemble_arrow_schur` built the
    /// β-tier in the *factored* Grassmann-coordinate layout (border width
    /// [`Self::factored_border_dim`], the per-atom `C_k` blocks) rather than the
    /// full-`B` layout ([`Self::beta_dim`]). When `true`, the `delta_beta` the
    /// arrow solver returns is a `ΔC` (factored coordinates) that
    /// [`Self::apply_newton_step_impl`] must LIFT through each active frame
    /// (`ΔB_k = ΔC_k U_kᵀ`) before applying it to the decoder. `false` ⇒ the
    /// historical full-`B` path, where `delta_beta` is `ΔB` directly. Set in
    /// lock-step with the assembled system so the step interpretation cannot
    /// drift from the layout the system was built in.
    pub(crate) last_frames_active: bool,
    /// #1033 test seam: force the large-`n` assembly fold to use this row chunk
    /// width instead of the streaming-plan's `chunk_size`. `None` ⇒ production
    /// behavior (the admission plan picks the window). Tests set a tiny value to
    /// drive the multi-chunk fold path on a small problem and assert it is
    /// bit-identical to the single-pass (`chunk_size == n`) fold. Never set in
    /// production; it only re-partitions the fold, never changes per-row math.
    pub(crate) assembly_chunk_override: Option<usize>,
    /// #1407: when set, `assemble_arrow_schur` emits ONLY the per-row block-
    /// diagonal `htt`/`gt` (the data-fit Gauss-Newton + assignment/ARD prior
    /// curvature and gradient), skipping the entire β decoder tier — the β Gram
    /// `G`, the `gb` gradient, the matrix-free `H_tβ` Kronecker operator, the
    /// dense `H_ββ`, and all β-tier analytic penalties. The fixed-decoder encode
    /// (`run_fixed_decoder_arrow_schur`) freezes the decoder and its per-row step
    /// reads only `htt`/`gt` (`fixed_decoder_step_from_rows`), so the entire
    /// `K`-dependent decoder assembly it otherwise computes-and-discards is
    /// avoided. A transient mode set in lock-step by that driver, not persisted.
    pub(crate) fixed_decoder_assembly: bool,
    /// #1408/#1409 — optional hard per-row active-atom cap for Softmax mode,
    /// threaded from the fit/encode `top_k` argument. When set (and `< K`), the
    /// Softmax assignment engages the COMPACT top-`k` row layout
    /// ([`SaeRowLayout::from_dense_weights`]) inside the optimization itself, so
    /// the Newton assembly/solve and the criterion only ever touch each row's
    /// top-`k` softmax atoms instead of all `K`. The full-`K` softmax
    /// normalization is still used to FORM each row's assignment vector `a`
    /// (the gate map); only the dropped tail logits — carrying negligible
    /// `O(a)` reconstruction mass and `O(a²)` curvature — are excluded from the
    /// per-row block. This folds `top_k` into the optimization rather than
    /// applying it as a post-fit projection (the #1409 defect), and makes the
    /// FFI's after-the-fact top-`k` projection a no-op at the optimum. `None`
    /// (the default) keeps the budget-driven engagement only.
    pub(crate) softmax_active_cap: Option<usize>,
    /// Reusable dense β-tier workspace for analytic penalty assembly. SAE
    /// immediately lowers the dense block into a `BetaPenaltyOp`, so the returned
    /// `ArrowSchurSystem` does not need to keep owning the allocation.
    pub(crate) border_hbb_workspace: Array2<f64>,
    /// Fitted Gaussian reconstruction dispersion used only by the empirical
    /// incoherence/curvature certificate-input report. `None` for synthetic terms
    /// or legacy internal callers that have not computed post-fit dispersion.
    pub(crate) certificate_dispersion: Option<f64>,
    /// Outcome of the most recent curvature-homotopy entry walk (#1007), or
    /// `None` when no walk has run (the seed cascade entry, or any consumer that
    /// never invokes the tracker). Recorded on the fit payload so the bifurcation
    /// / collapse outcome is observable — never a silent fallback. Cleared by
    /// the objective's `reset` so each seed's walk reports only its own run.
    pub(crate) curvature_walk_report: Option<CurvatureWalkReport>,
    /// Deflated row-gauge direction count established by the first undamped
    /// evidence factorization in the current optimization. A later change means
    /// the quotient dimension changed mid-solve, which is a structural event and
    /// must not be hidden inside the Laplace normalizer.
    pub(crate) expected_evidence_gauge_deflated_directions: Option<usize>,
    /// #1037 re-anchor counter: how many times the quotient (gauge-deflation)
    /// dimension has been re-anchored within the current optimization. A
    /// legitimate quotient-dimension change (an atom born / reseeded /
    /// rank-reduced) re-anchors the comparison once; an unbounded churn that
    /// never settles is the genuine pathology the guard must still catch. Reset
    /// to `0` alongside `expected_evidence_gauge_deflated_directions`.
    pub(crate) evidence_gauge_deflation_reanchors: usize,
    /// #1217 oscillation detector: the sign of the most recent change in the
    /// gauge-deflation count (`+1` when it last increased, `−1` when it last
    /// decreased, `0` before the first change). The deflation count is a
    /// per-ROW-summed count of near-null evidence directions, so on real K≥2
    /// data it drifts smoothly (and monotonically) as the conditioning improves
    /// across the ρ-walk — a benign O(N) quantity, NOT a discrete dictionary
    /// event. The genuine pathology the #1037 guard must catch is an
    /// OSCILLATING count (repeated direction reversals that never settle), so
    /// the re-anchor budget is charged only on a direction REVERSAL, not on
    /// every monotone drift step. Reset to `0` alongside the re-anchor counter.
    pub(crate) evidence_gauge_deflation_last_delta_sign: i8,
    /// #976 / #1117 K>1 robustness: how many full-dictionary co-collapse
    /// multi-starts the decoder-norm guard has already spent in the current
    /// optimization. Bounded by [`SAE_DICTIONARY_COCOLLAPSE_RESEED_BUDGET`];
    /// reset to `0` alongside [`Self::evidence_gauge_deflation_reanchors`] at the
    /// start of each outer optimization. Distinct from the per-atom reseed
    /// ledger in [`Self::collapse_events`] because a co-collapse reseed is a
    /// whole-dictionary multi-start, not a per-atom second chance.
    pub(crate) dictionary_cocollapse_reseeds: usize,
    /// #1026 co-collapse multi-start incumbent: the best (highest reconstruction
    /// EV) dictionary state seen across the full-dictionary co-collapse reseeds in
    /// the current optimization, with that EV. A blind reseed sequence can land in
    /// a strictly WORSE basin than the seed it replaced (real OLMo K=4: EV 0.127 →
    /// −1.01 across reseeds), so the multi-start must RETAIN the best basin and
    /// restore it when the reseed budget is spent, instead of keeping the last
    /// (often catastrophic) attempt. `None` until the first co-collapse reseed;
    /// reset to `None` alongside [`Self::dictionary_cocollapse_reseeds`] at the
    /// start of each outer optimization.
    ///
    /// #2081 — the middle `Option<f64>` is the incumbent basin's aggregate
    /// coordinate-uniformity score ([`SaeManifoldTerm::coordinate_uniformity_aggregate`]),
    /// carried alongside the EV so the multi-start can break (near-)equal-EV ties
    /// on coordinate fidelity ([`prefer_candidate_basin`]).
    pub(crate) best_cocollapse_incumbent: Option<(f64, Option<f64>, SaeManifoldMutableState)>,
    /// #1026 decoder-repulsion gate, frozen per assembly (lagged-diffusivity
    /// discipline, exactly like [`SaeManifoldAtom::smooth_penalty`]): the
    /// symmetric `(K, K)` matrix of collinearity gate weights
    /// `gate(s_jk)·decoder_repulsion_strength()/(‖B_j‖²_F·‖B_k‖²_F)` computed
    /// from the decoder state at assembly entry (#1610: energy-normalized, so the
    /// realized penalty is a function of the dimensionless collinearity `c_jk²`
    /// alone). Both the assembly (gradient into `gb`, PSD
    /// curvature into `hbb`) and the line-search value path
    /// ([`Self::penalized_objective_total`]) read THIS frozen gate, so the
    /// repulsion's value, gradient, and curvature stay mutually consistent
    /// across a Newton step even as the trial decoders move (the cross-Gram
    /// quadratic moves with the trial; only the gate weight is frozen). `None`
    /// when no repulsion is active (`K < 2`, or every pair below the gate
    /// threshold — the strict no-op case). Refreshed at the same chokepoint as
    /// the smoothness Gram; not part of the persisted term identity (Clone
    /// starts `None`).
    /// SPARSE near-collinear pair gate (#1026): the list of `(j, k, w)` with
    /// `j < k` whose frozen repulsion weight `w > 0`. Only near-collinear pairs
    /// (above the collinearity gate) ever carry a nonzero weight, so this list is
    /// tiny even at large `K` — never the dense `K×K` matrix (8 GiB at K=32768).
    pub(crate) decoder_repulsion_gate: Option<Vec<(usize, usize, f64)>>,
    /// #1625 — the SEPARATION barrier's frozen normalized-coactivation weights
    /// `q_jk`, the analog of [`Self::decoder_repulsion_gate`] for the #1522
    /// collapse-prevention barrier. The barrier energy
    /// `P_sep = μ_C·Σ_{j<k} −q_jk·log(1−c_jk²+ε)` is weighted by the normalized
    /// coactivation `q_jk = (Σ_i a_ij a_ik)/√(Σa_ij²·Σa_ik²)`, which is a function
    /// of the assignment masses `a_ik` (hence the logits). The barrier's GRADIENT
    /// assembly ([`Self::add_sae_separation_barrier`]) treats `q_jk` as a constant
    /// multiplicative weight (it differentiates only the decoder shape `c_jk²`), so
    /// for value/gradient consistency the VALUE path
    /// ([`Self::separation_barrier_value`], read by the line-search
    /// [`Self::penalized_objective_total`]) MUST read the SAME `q_jk` the gradient
    /// used — not recompute it from the trial logits the line search moves.
    /// Freezing it here at assembly entry (lagged-diffusivity, exactly like the
    /// smoothness Gram and the repulsion gate) makes the barrier a pure function of
    /// the decoder shapes within a Newton step, so it exerts NO phantom force on
    /// the routing and the inner solve reaches true KKT stationarity. `None` when
    /// no pair co-fires (`K < 2`, or a fully-disjoint routing — the strict no-op);
    /// callers fall back to the live coactivation in that case. Transient: not part
    /// of the persisted term identity (Clone starts `None`, rebuilt next assembly).
    /// #1610 — per-assembly FROZEN separation-barrier pairs
    /// `(j, k, q_jk, μ_jk)`: the co-firing atom indices, their normalized
    /// coactivation weight `q_jk`, and the EVIDENCE-DERIVED per-pair barrier
    /// strength `μ_jk` (the data-fit inseparability strength, see
    /// [`super::penalties::SaeManifoldTerm::barrier_pair_strength`]). Both the
    /// coactivation and the strength are functions of the frozen design (chart
    /// basis + routing), so freezing them here keeps the barrier value, gradient,
    /// and curvature reading the SAME weights across an inner line search.
    pub(crate) barrier_coactivation_gate: Option<Vec<(usize, usize, f64, f64)>>,
    /// #1026: the load-bearing curved-vs-linear hybrid-split verdict, computed
    /// once in [`Self::canonicalize_charts_post_fit`] after the joint fit
    /// converges. Each eligible `d = 1` atom's fitted curved image is adjudicated
    /// against its straight (linear special-case) sub-model on the common
    /// rank-aware Laplace evidence scale. `None` until the post-fit pass runs (or
    /// when no atom is eligible). Surfaced in the Python model output so a user
    /// sees which atoms genuinely earn their curvature and which collapse to the
    /// linear tail. Read via [`Self::hybrid_split_report`].
    pub(crate) hybrid_split_report: Option<crate::hybrid_split::SaeHybridSplitReport>,
    /// Per-atom inner-decoder-smooth byproducts harvested post-fit (#1097 /
    /// #1103), one entry per atom in [`Self::atoms`] order. Each is the fixed
    /// fitted snapshot the residual-gauge certificate's three post-PIRLS atom
    /// inference reports consume
    /// ([`crate::identifiability::AtomInnerFit`]). `None` until
    /// [`Self::set_atom_inner_fits`] runs (it needs the reconstruction target
    /// `Z`, available only at the post-fit harness seam where the dispersion is
    /// also profiled); a per-atom `None` means that atom had no active rows or a
    /// degenerate inner design. Read by [`Self::to_residual_gauge_model`], which
    /// attaches each onto its [`crate::identifiability::FittedAtom`].
    pub(crate) atom_inner_fits: Option<Vec<Option<crate::identifiability::AtomInnerFit>>>,
    /// #1228 — the trained dictionary's hybrid-collapsed linear images, attached
    /// to an OOS term so held-out reconstruction decodes verdict-linear `d = 1`
    /// slots by the SAME straight sub-model the training reconstruction used.
    /// `None` on a fitted term (its collapse policy lives in
    /// [`Self::hybrid_split_report`]); `Some` only on the fresh OOS term built in
    /// `sae_manifold_predict_oos`, set from the serialized payload via
    /// [`Self::set_hybrid_linear_images`]. Consulted by
    /// [`Self::hybrid_linear_image_map`] so train and OOS share one collapse map.
    pub(crate) oos_linear_images: Option<Vec<crate::hybrid_split::AtomLinearImage>>,
    /// #1777 PER-FIT separation-barrier strength override `μ_C` — the source of
    /// truth for the barrier strength when set, replacing the process-global
    /// [`set_sae_barrier_overrides`] atomic. `Some(μ_C)` forces the absolute
    /// strength (bypassing the #1610 evidence-derived per-pair reciprocal-margin
    /// strengths), scoped to THIS term/fit so concurrent in-process fits are
    /// isolated; `0.0` disables the barrier. `None` ⇒ fall back to the deprecated
    /// process-global override, then the evidence-derived strength (bit-identical
    /// to the historical path when neither override is set). Read via
    /// [`super::penalties::SaeManifoldTerm::separation_barrier_strength`]; set from
    /// the FFI through [`SaeManifoldTerm::set_fit_config`]. Carried across clones
    /// (persisted configuration, like the assignment mode).
    pub(crate) separation_barrier_strength_override: Option<f64>,
    /// #2022 — persisted per-fit opt-in for the SCALE-gauge quotient (default
    /// false ⇒ bit-for-bit historical path). When true, the decoder is confined
    /// to the unit-Frobenius sphere with its magnitude in the explicit per-atom
    /// log-amplitude (seed/step/refit peel + sphere retract). Set from the FFI
    /// via the typed `quotient_scale` kwarg — no env lever. Carried across clones
    /// like the other per-fit config.
    pub(crate) quotient_scale: bool,
    /// #1939 — persisted per-fit opt-in for the cone-atom RECOVERY retraction
    /// (default false ⇒ bit-for-bit historical path). When true, at each accepted
    /// OUTER-iterate boundary any atom whose decoder has COLLAPSED relative to its
    /// dictionary peers (`‖B_k‖ < ratio·median`) is retracted onto the unit sphere
    /// and its amplitude re-solved, re-homing a co-vanished born decoder without
    /// touching healthy atoms. Distinct from `quotient_scale`: this does ONLY the
    /// stable breach-gated boundary retraction and NEVER the #2022 per-Newton fold
    /// (fit_drivers.rs:3227), so it cannot detonate a healthy fit. Carried across
    /// clones like the other per-fit config.
    pub(crate) cone_atom_recovery: bool,
    /// #5/(B) — persisted per-fit opt-in (default false ⇒ bit-for-bit historical
    /// path) for the RANK-CHARGE evidence criterion. When true, the Laplace
    /// complexity's per-atom COORDINATE-block term ½log|H_tt| — which mis-prices
    /// with the `log(a²‖B‖²)` scale (over-charging real atoms, rewarding
    /// a²‖B‖²→0) — is replaced by the honest BIC ½·d_eff·log n on the atom's
    /// realised decoder RANK: d_eff_k = rank_eff_k · basis_edf_k, rank_eff_k =
    /// Σ_i s_i²/(s_i²+R) over the decoder singular values with a FIXED dictionary-
    /// relative floor R (so a vanishing atom → rank→0 → charge 0 → neutral, the
    /// co-collapse fix). Rotation-invariant; it does NOT distinguish a clean
    /// circle from a blend (both rank-2) — that is the producer's job. Carried
    /// across clones like the other per-fit config.
    pub(crate) rank_charge_evidence: bool,
    /// #2023 — persisted per-fit opt-in for the dead-atom DATA-ROW reseed
    /// (default false). Set from the FFI via the typed `data_row_reseed` kwarg —
    /// no env lever. Carried across clones.
    pub(crate) data_row_reseed: bool,
    /// SAC — whether the #976 Layer-1 collapse-guard stack (active-mass /
    /// decoder-norm re-seed, co-collapse reseed-all) is armed on this term's
    /// inner joint fits. Default `true` (bit-for-bit historical path). The
    /// Sequential Atom Composition driver ([`super::stagewise`]) fits one atom at
    /// a time and drives this to `false` on the K=1 path: a single atom never
    /// trips the guards (there is no dictionary peer to collapse against), so the
    /// guards are pure no-ops there, and disarming them makes the per-atom /
    /// backfitting refits provably reseed-free — a reseed mid-refit would break
    /// the block-coordinate monotonicity the composition rests on. Carried across
    /// clones like the other per-fit config so a cloned candidate keeps the lane.
    pub(crate) guards_enabled: bool,
    /// Rung-2 behavioral data block: when `Some`, this term's output is the
    /// AUGMENTED stack `[activation | √λ_y · behavior-tangent]` and each atom's
    /// decoder is the widened `[B_k | C_k]`. The block records the sphere-tangent
    /// chart, the unscaled nats-unit behavior target, the activation/behavior
    /// output split, and the REML-selected relative weight `λ_y`. `None` ⇒ the
    /// ordinary single-block (activation-only) term, bit-for-bit unchanged. It is
    /// pure descriptor state (which output columns are behavior, and how to decode
    /// them back to distributions); the joint fit itself needs no special path,
    /// because the augmented output shares `t` and `a` by construction. Carried
    /// across clones so a cloned candidate keeps its behavioral identity.
    pub(crate) behavior: Option<crate::manifold::BehaviorBlock>,
}

/// #1777 — PER-FIT configuration overrides the FFI sets on a term to isolate a
/// fit from the deprecated process-global atomics
/// ([`set_sae_barrier_overrides`] / [`crate::assignment::set_ibp_alpha_override`]).
///
/// This is the additive, back-compatible config surface the Python/FFI layer
/// consumes: build it, then apply it with [`SaeManifoldTerm::set_fit_config`]. Any
/// `None` field leaves the corresponding axis on its historical fallback
/// (process-global override, then the compiled/data-derived default), so an
/// all-`None` config is a strict no-op. Both fields are the SOURCE OF TRUTH for
/// their axis when `Some`, so two terms carrying different configs produce
/// correspondingly-different barrier/α WITHOUT touching any global — concurrent
/// fits are isolatable.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SaeFitConfig {
    /// Per-fit separation-barrier strength `μ_C`. `Some` bypasses both the global
    /// override and the #1610 evidence-derived per-pair strengths (`0.0` = barrier off).
    pub separation_barrier_strength_override: Option<f64>,
    /// Per-fit truncated-IBP concentration `α`. `Some` bypasses both the global
    /// override and the mode's own `α` / learnable schedule.
    pub ibp_alpha_override: Option<f64>,
}

impl Clone for SaeManifoldTerm {
    fn clone(&self) -> Self {
        Self {
            atoms: self.atoms.clone(),
            assignment: self.assignment.clone(),
            temperature_schedule: self.temperature_schedule.clone(),
            last_row_layout: self.last_row_layout.clone(),
            row_metric: self.row_metric.clone(),
            collapse_events: self.collapse_events.clone(),
            row_loss_weights: self.row_loss_weights.clone(),
            last_frames_active: self.last_frames_active,
            assembly_chunk_override: self.assembly_chunk_override,
            fixed_decoder_assembly: false,
            // Persisted configuration (like the assignment mode), carried across
            // clones so a cloned term optimizes the same compact top-`k` problem.
            softmax_active_cap: self.softmax_active_cap,
            border_hbb_workspace: Array2::<f64>::zeros((0, 0)),
            certificate_dispersion: self.certificate_dispersion,
            curvature_walk_report: self.curvature_walk_report.clone(),
            expected_evidence_gauge_deflated_directions: self
                .expected_evidence_gauge_deflated_directions,
            evidence_gauge_deflation_reanchors: self.evidence_gauge_deflation_reanchors,
            evidence_gauge_deflation_last_delta_sign: self.evidence_gauge_deflation_last_delta_sign,
            dictionary_cocollapse_reseeds: self.dictionary_cocollapse_reseeds,
            // Transient in-fit multi-start incumbent — not part of the persisted
            // term identity (like `border_hbb_workspace`); a fresh clone starts
            // with no incumbent and rebuilds it if it re-enters co-collapse.
            best_cocollapse_incumbent: None,
            // Transient per-assembly frozen gate — rebuilt at the next assembly.
            decoder_repulsion_gate: None,
            // #1625 — transient per-assembly frozen barrier coactivation; rebuilt
            // at the next assembly, exactly like the repulsion gate above.
            barrier_coactivation_gate: None,
            hybrid_split_report: self.hybrid_split_report.clone(),
            atom_inner_fits: self.atom_inner_fits.clone(),
            oos_linear_images: self.oos_linear_images.clone(),
            // #1777 — persisted per-fit config, carried across clones like the
            // assignment mode so a cloned term keeps the same barrier override.
            separation_barrier_strength_override: self.separation_barrier_strength_override,
            quotient_scale: self.quotient_scale,
            cone_atom_recovery: self.cone_atom_recovery,
            rank_charge_evidence: self.rank_charge_evidence,
            data_row_reseed: self.data_row_reseed,
            guards_enabled: self.guards_enabled,
            // Rung-2 behavioral identity is persisted configuration (like the
            // assignment mode / barrier override), carried across clones so a
            // cloned candidate fits the same augmented two-block problem.
            behavior: self.behavior.clone(),
        }
    }
}

/// Snapshot of exactly the mutable term state that an `apply_newton_step` +
/// `loss` line-search trial perturbs: per-atom decoder coefficients, the
/// `refresh_basis`-rebuilt basis evaluations (`basis_values`, `basis_jacobian`),
/// and the live intrinsic smoothness Gram read by the objective, plus the
/// assignment logits and latent coordinates.
///
/// Static fields (atom names, basis kinds, basis-evaluator `Arc`s, assignment
/// mode, temperature schedule) are *not* snapshotted: they are invariant across
/// an inner Newton line search, so the previous `self.clone()` per halving
/// re-copied them needlessly. Cloning only the line-search state keeps the
/// `O(N·M·d)` `basis_jacobian` copy off the per-halving hot path (one snapshot
/// before the search, one restore per rejected trial) instead of firing it on
/// every Armijo backtrack.
///
/// The canonical `smooth_penalty_raw` / `smooth_penalty_order` are static, but
/// the live intrinsic roughness Gram `smooth_penalty` is mutable state: it is
/// refreshed by assembly from the current decoder and basis Jacobian, and the
/// line-search objective reads it directly. Restoring it with the decoder and
/// basis caches keeps every rejected trial's baseline and nonlinear objective
/// on the same lagged-diffusivity quadratic.
#[derive(Debug)]
pub(crate) struct SaeManifoldMutableState {
    /// Per-atom `(basis_values, basis_jacobian, decoder_coefficients, smooth_penalty)`.
    pub(crate) atoms: Vec<(Array2<f64>, Array3<f64>, Array2<f64>, Array2<f64>)>,
    pub(crate) logits: Array2<f64>,
    pub(crate) coords: Vec<LatentCoordValues>,
    pub(crate) last_row_layout: Option<SaeRowLayout>,
}
