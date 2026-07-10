use super::*;

/// Armijo sufficient-decrease constant — sourced from the shared optimizer
/// constants so the workspace has exactly one `c₁`.
pub(crate) const SAE_MANIFOLD_ARMIJO_C1: f64 = opt::constants::ARMIJO_C1;

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
/// magnitude. This tolerance is intentionally EV-only: penalized-objective
/// comparisons use [`SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL`] so the much
/// coarser reporting-scale EV band cannot override a genuine objective descent.
pub(crate) const SAE_FINAL_EV_DEGRADATION_TOL: f64 = 1.0e-3;

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
// [`SAE_DECODER_REPULSION_BARRIER_RATIO`] of the data-derived (or explicitly
// per-fit) separation-barrier strength μ_C; it is computed on the
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

// The SEPARATION barrier has NO strength scalar `μ_C` at all: it is the SAE decoder
// Jeffreys prior `−½ log det F` (see [`super::penalties::BarrierComponent`]), whose
// exponent `½` is fixed by the prior (`π(B) ∝ √det F`) and is the exact
// reparametrization-invariant counter-term to the Laplace evidence's `+½ log(volume)`
// collapse reward — so a per-pair strength is neither present nor needed. The
// historical `μ_jk = γ_jk/(1−γ_jk)` (data-fit inseparability, see
// [`super::penalties::SaeManifoldTerm::barrier_pair_strength_with_gates`]) survives
// ONLY as the per-dictionary strength of the subdominant decoder-repulsion CONDITIONER
// (`decoder_repulsion_strength = ratio · μ_C`), not of the barrier. An explicit
// per-fit strength can scale that conditioner through [`SaeFitConfig`].

/// SEPARATION barrier softening `ε`: the floor on the eigenvalues of the Jeffreys
/// Fisher `F = Q ∘ O` (see [`super::penalties::BarrierComponent`]) in
/// `−½ log det F`. Bounds the barrier (and its PSD curvature majorizer) at exact
/// collapse (`det F → 0`) at a finite `−½ log ε` — the multi-atom analog of the
/// historical pairwise softening `1 − c_jk² + ε`.
pub(crate) const SAE_SEPARATION_BARRIER_EPS: f64 = 1.0e-6;

// The SEPARATION barrier no longer has a hard collinearity gate. Its predecessor,
// the pairwise `−μ q w(c²) log(1−c²+ε)`, needed a C1 smoothstep `w(c²)` (exactly 0
// below `c² ≈ 0.5`) to suppress the ungated `−log(1−c²+ε)`'s O(1) force at moderate
// collinearity (the #1625 stall: an O(1) force at `c² = 0.36` drags a healthy fit
// off the data optimum). The Jeffreys `−½ log det F` needs no such gate: its force
// on an edge is `q²o/(1−q²o²)·∂o/∂B`, which vanishes as `O(o)` for separated atoms
// (so it cannot dominate a near-zero data residual) and diverges only as
// `det F → 0`. The soft interior-point structure IS the gate — with the Jeffreys
// exponent `½` fixed and no tuned threshold `s0`.

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
    /// #1890 — multi-chart semantic atoms.  The numerical storage remains one
    /// decoder/routing block per local chart; each entry records which blocks
    /// form one atlas plus the fitted unit-speed transition cocycle.  Existing
    /// gates then factor exactly into atlas activation × partition-of-unity, so
    /// registration preserves the fitted image while retaining pole and Möbius
    /// seams that a destructive single-chart fusion would erase.
    pub(crate) chart_atlases: Vec<ManifoldChartAtlas>,
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
    /// Exact recurrence ledger for the OBJECTIVE-keyed in-call incumbent
    /// restore. This is convergence evidence only: the saved state is compared
    /// for identity on the next call and is never installed across ρ values or
    /// after the outer certificate.
    pub(crate) best_fit_incumbent: Option<SaeFitIncumbent>,
    /// Bounded high-EV structural-collapse reseeds spent by the frame-coherence
    /// guard in the current optimization. This is separate from total decoder
    /// co-collapse: these atoms still carry decoder norm and EV, but duplicate an
    /// occupied output frame.
    pub(crate) structural_cocollapse_reseeds: usize,
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
    /// Per-assembly FROZEN separation-barrier support: the co-firing pairs
    /// `(j, k, q_jk)` — the entries of the Jeffreys Fisher `F = Q ∘ O` (see
    /// [`super::penalties::BarrierComponent`]) — TOGETHER with the per-atom
    /// effective sample sizes `N_eff,k = Σ_{i∈J_k} a_ik²` accumulated over the
    /// SAME truncated active support as `q`'s denominators. Both are functions of
    /// the routing masses (hence the logits the inner Newton solve moves), so
    /// freezing them here keeps the barrier value, gradient, and curvature reading
    /// the SAME `Q`, the SAME per-atom effective sample sizes, and therefore the
    /// SAME data-derived softening `ε_C` across an inner line search. The decoder
    /// overlaps `O` are NOT frozen — they are the LIVE shapes
    /// the barrier separates
    /// ([`super::penalties::SaeManifoldTerm::decoder_gram_cosine_sq`]), moving with
    /// the trial decoders. The Jeffreys exponent `½` is fixed, so there is no
    /// per-pair strength `μ_jk` to freeze (that under-derived scalar is gone).
    pub(crate) barrier_coactivation_gate: Option<BarrierCoactivationGate>,
    /// #1801 — STREAMING gate-freeze flag. The collapse-prevention gates
    /// ([`Self::decoder_repulsion_gate`], [`Self::barrier_coactivation_gate`]) are
    /// GLOBAL dictionary properties: their per-pair strength `μ_jk` inverts the
    /// coactivation-weighted design Grams `G_j` (`M_j × M_j`), which are
    /// near-singular on a small minibatch, so recomputing them per streaming chunk
    /// makes `μ_jk = γ/(1−γ)` blow up as `γ→1` and the fit depend on `chunk_size`.
    /// The streaming fit driver instead FREEZES these gates ONCE per outer
    /// iteration from the full resident routing (exactly like the decoder frames)
    /// and carries them onto every materialized chunk. When this flag is `true` the
    /// per-chunk assembly SKIPS its own gate refresh and uses the carried global
    /// gate, so the reduced β-Newton step and line-search objective are
    /// chunk-size invariant. Default `false` (the dense/full-batch path refreshes
    /// per assembly, bit-for-bit unchanged). Transient (Clone starts `false`).
    pub(crate) streaming_gates_frozen: bool,
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
    /// #1777 PER-FIT decoder-repulsion CONDITIONER strength override `μ_C`. The
    /// primary
    /// separation barrier is now the parameter-free Jeffreys prior `−½ log det F`
    /// and ignores this; the override scales only the subdominant repulsion
    /// conditioner. `Some(μ_C)` forces the absolute strength (bypassing the #1610
    /// evidence-derived per-pair reciprocal-margin strengths), scoped to THIS
    /// term/fit so concurrent in-process fits are isolated; `0.0` disables the
    /// conditioner. `None` uses the evidence-derived strength. Read via
    /// [`super::penalties::SaeManifoldTerm::separation_barrier_strength`]; set from
    /// the FFI through [`SaeManifoldTerm::set_fit_config`]. Carried across clones
    /// (persisted configuration, like the assignment mode).
    pub(crate) separation_barrier_strength_override: Option<f64>,
    /// #1939 — persisted per-fit opt-in for the cone-atom RECOVERY retraction
    /// (default false ⇒ bit-for-bit historical path). When true, at each accepted
    /// OUTER-iterate boundary any atom whose decoder has COLLAPSED relative to its
    /// dictionary peers (`‖B_k‖ < ratio·median`) is retracted onto the unit sphere
    /// and its amplitude re-solved, re-homing a co-vanished born decoder without
    /// touching healthy atoms. This does only the stable breach-gated boundary
    /// retraction. Carried across clones like the other per-fit config.
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
    /// Theorem K — persisted per-fit opt-in (default false ⇒ bit-for-bit historical
    /// path) for the WBIC SOFT rank-charge ledger. Only has effect when
    /// `rank_charge_evidence` is also on: inside that branch the per-atom coefficient
    /// of the occupancy log-scale `ln N_eff,k` becomes the finite-n WBIC learning
    /// coefficient `λ_k = ½·rank_soft_k·basis_edf_k` (the β=1/log n_eff likelihood-tempered, prior-untempered count on
    /// the occupancy-corrected reconstruction spectrum) instead of the hard limit
    /// `½·d_eff,k`. The two coincide away from the Marchenko–Pastur edge (soft→hard)
    /// and the soft one is strictly smaller near it — the honest Watanabe correction
    /// for near-singular curved atoms. The rank_eff==0 categorical veto (a validity
    /// condition, not a small charge) is unchanged. Carried across clones like the
    /// other per-fit config so a cloned candidate (e.g. a stagewise per-birth clone)
    /// keeps the same charge currency — the field, NOT a thread-local, is the source
    /// of truth so it propagates correctly across rayon worker threads.
    pub(crate) soft_rank_charge: bool,
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
    /// Crosscoder — stacked-column layout of a multi-block fit: the anchor width
    /// `p_x`, the per-layer output-block widths/labels, and the fitted per-block
    /// relevance `log λ_ℓ`. `Some` after a [`SaeManifoldTerm::run_multiblock_reml_fit`]
    /// (or an explicit [`SaeManifoldTerm::set_crosscoder_layout`]); it is the single
    /// owner of the offset bookkeeping so [`SaeManifoldTerm::layer_decoder`] returns
    /// an honest per-layer decoder `B_k^(ℓ)` without any caller re-slicing by hand.
    /// Pure descriptor state (which columns are which layer, and each layer's
    /// weight) — the joint fit never reads it, so `None` is the ordinary path,
    /// bit-for-bit unchanged. Carried across clones like the behavior block.
    pub(crate) crosscoder_layout: Option<crate::manifold::CrosscoderLayout>,
    /// #2023 C4 — the manifold-tier analogue of [`crate::tiered::Tier0Mean`]: the
    /// single shared mean μ (length `p`) that carries the global DC for THIS term.
    /// `None` (the default) ⇒ the historical, bit-for-bit path (no Tier-0; every
    /// atom is free to smear a piece of the mean through its constant basis column).
    /// `Some(μ)` ⇒ the atoms are fit against the de-meaned target `Z − μ` and the
    /// reconstruction adds μ back ([`SaeManifoldTerm::try_fitted_with_rho`]).
    ///
    /// Moving the global DC out of the K per-atom intercepts into ONE Tier-0 mean
    /// structurally kills the co-collapse-to-mean zombie class (#10 / #1893): on
    /// de-meaned data the "every atom decodes the mean" state reconstructs zero, so
    /// it earns zero explained variance and is pruned by the rank charge (its
    /// realised rank → 0), instead of being rewarded and PC-reseeded. A pure
    /// DC-constant decoder is then EV-invisible BY CONSTRUCTION — the mean it would
    /// chase is already gone. Carried across clones like the other persisted config.
    pub(crate) tier0_mean: Option<Array1<f64>>,
    /// #1939 — the boundary amplitude solve's converged empirical-Bayes prior
    /// state: the MacKay/Fellner–Schall evidence precision `α` on the active
    /// log-amplitudes, the per-atom SCAD selection coefficients `λ_k` (β-space),
    /// and the noise scale `σ̂²` that converts the solve's noise-whitened prior
    /// energy into data-fit units. Persisted so the FIT-LEVEL referee
    /// ([`SaeManifoldTerm::penalized_objective_total`], hence
    /// `prefer_candidate_state` and the Armijo line search) prices the SAME
    /// amplitude prior the solve minimised — without it the keep-best incumbent
    /// ranks states on the bare data-fit and silently vetoes the prior's
    /// accepted shrinkage (the #2230 objective-desync class, relocated to the
    /// amplitude channel). `None` until the first amplitude solve (bit-for-bit
    /// historical objective); transient across clones and stale-guarded on `K`
    /// (a mismatched `λ` length is skipped, then re-installed by the next
    /// solve).
    pub(crate) amplitude_prior: Option<AmplitudePriorState>,
}

/// #1939 — converged empirical-Bayes amplitude-prior state persisted by
/// [`SaeManifoldTerm::optimize_log_amplitudes_closed_form`]; see the field doc
/// on [`SaeManifoldTerm::amplitude_prior`].
#[derive(Debug, Clone)]
pub(crate) struct AmplitudePriorState {
    /// Evidence (ARD) precision on the ACTIVE log-amplitudes, `s_k ~ N(0, 1/α)`.
    pub(crate) alpha: f64,
    /// Per-atom SCAD coefficient `λ_k` in amplitude (β) space.
    pub(crate) scad_lambda: Vec<f64>,
    /// LS residual noise scale `σ̂²`: multiplying the noise-whitened prior
    /// energy by `σ̂²` expresses it in the same units as the weighted data-fit.
    pub(crate) sigma2: f64,
}

/// Per-fit SAE configuration consumed by the Python/FFI layer. Build it, then
/// apply it with [`SaeManifoldTerm::set_fit_config`]. A `None` field selects the
/// corresponding canonical data-derived or assignment-mode default. Both fields
/// are the source of truth for their axis when `Some`, so concurrent fits remain
/// fully isolated.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SaeFitConfig {
    /// Per-fit separation-barrier strength `μ_C`. `Some` bypasses the #1610
    /// evidence-derived per-pair strengths (`0.0` = conditioner off).
    pub separation_barrier_strength_override: Option<f64>,
    /// Per-fit truncated-IBP concentration `α`. `Some` bypasses the mode's own
    /// `α` / learnable schedule.
    pub ibp_alpha_override: Option<f64>,
}

impl Clone for SaeManifoldTerm {
    fn clone(&self) -> Self {
        Self {
            atoms: self.atoms.clone(),
            assignment: self.assignment.clone(),
            chart_atlases: self.chart_atlases.clone(),
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
            best_fit_incumbent: None,
            structural_cocollapse_reseeds: self.structural_cocollapse_reseeds,
            // Transient per-assembly frozen gate — rebuilt at the next assembly.
            decoder_repulsion_gate: None,
            // #1625 — transient per-assembly frozen barrier coactivation; rebuilt
            // at the next assembly, exactly like the repulsion gate above.
            barrier_coactivation_gate: None,
            // #1801 — transient streaming gate-freeze flag; a fresh clone refreshes
            // its gates per assembly like the dense path until a streaming fit
            // re-arms it.
            streaming_gates_frozen: false,
            hybrid_split_report: self.hybrid_split_report.clone(),
            atom_inner_fits: self.atom_inner_fits.clone(),
            oos_linear_images: self.oos_linear_images.clone(),
            // #1777 — persisted per-fit config, carried across clones like the
            // assignment mode so a cloned term keeps the same barrier override.
            separation_barrier_strength_override: self.separation_barrier_strength_override,
            cone_atom_recovery: self.cone_atom_recovery,
            rank_charge_evidence: self.rank_charge_evidence,
            soft_rank_charge: self.soft_rank_charge,
            data_row_reseed: self.data_row_reseed,
            guards_enabled: self.guards_enabled,
            // Rung-2 behavioral identity is persisted configuration (like the
            // assignment mode / barrier override), carried across clones so a
            // cloned candidate fits the same augmented two-block problem.
            behavior: self.behavior.clone(),
            // Crosscoder stacked-column layout is persisted descriptor state (like
            // the behavior block), carried across clones so a cloned candidate
            // reports the same per-layer decoders.
            crosscoder_layout: self.crosscoder_layout.clone(),
            // #2023 C4 — persisted Tier-0 shared mean, carried across clones like
            // the assignment mode so a cloned candidate de-means identically.
            tier0_mean: self.tier0_mean.clone(),
            // #1939 — transient like the incumbents: a clone (stagewise birth /
            // candidate) may change K, so it starts with no amplitude prior and
            // the next boundary solve re-installs one sized to its dictionary.
            amplitude_prior: None,
        }
    }
}

/// Snapshot of exactly the mutable term state that an `apply_newton_step` +
/// `loss` line-search trial perturbs, stored DIFFERENTIALLY: the CHEAP driving
/// state — decoder coefficients, the live intrinsic smoothness Gram, and the
/// per-atom basis-determining handles (`basis_evaluator`, `basis_second_jet`,
/// `homotopy_eta`) — plus the assignment logits, latent coordinates, and row
/// layout.
///
/// The expensive `basis_values` (`N×M`) and `basis_jacobian` (`N×M×d`) arrays are
/// NOT stored: they are pure deterministic functions of `(coords, basis_evaluator,
/// homotopy_eta)` via [`SaeManifoldAtom::refresh_basis`]. At the production curved
/// tier (K=512, N=96k, M=7, d=1) copying them was ~5.5 GB of `memcpy` per snapshot
/// — taken once per Newton line search, per incumbent-banking improvement, per
/// basin-bundle member, per polish round — the dominant memory-bandwidth wall.
/// [`SaeManifoldTerm::restore_mutable_state`] reassigns the cheap state, restores
/// the basis-determining handles, and rebuilds `basis_values`/`basis_jacobian`
/// in place via `refresh_basis_from_current_coords` (reusing the atoms' existing
/// buffers through `SaeBasisEvaluator::evaluate_into`). Restores happen only on
/// REJECTED trials / incumbent rollbacks, which are rare relative to snapshots, so
/// trading a snapshot-time `O(K·N·M·(1+d))` copy for a restore-time single basis
/// evaluation is a large net win — and the previous restore ALSO deep-copied the
/// basis via `.assign()`, so restore gets cheaper too.
///
/// The basis-determining handles are captured because `basis_values` is a function
/// of the evaluator and `η`, not of the coordinates alone: `canonicalize_atom_
/// affine_gauge` swaps `basis_evaluator`/`basis_second_jet`, and the curvature-
/// homotopy walk moves `homotopy_eta`. Restoring all three makes the rebuilt basis
/// bit-identical to the snapshotted state in every case (the affine-gauge reject
/// and the η=0 base-topology anchor restore both re-derive the exact pre-change
/// basis), where the previous direct `basis_values` restore left a mixed evaluator/
/// η state. The `Arc`/`f64` handles are pointer-cheap to clone.
///
/// Static fields (atom names, basis kinds, assignment mode, temperature schedule)
/// are still not snapshotted: they are invariant across a Newton line search.
///
/// The canonical `smooth_penalty_raw` / `smooth_penalty_order` are static, but
/// the live intrinsic roughness Gram `smooth_penalty` is mutable state: it is
/// refreshed by assembly from the current decoder and basis Jacobian, and the
/// line-search objective reads it directly. Restoring it with the decoder keeps
/// every rejected trial's baseline and nonlinear objective on the same
/// lagged-diffusivity quadratic.
#[derive(Debug)]
pub(crate) struct SaeManifoldMutableState {
    /// Per-atom differential state
    /// `(decoder_coefficients, smooth_penalty, basis_evaluator, basis_second_jet,
    /// homotopy_eta)`. `basis_values`/`basis_jacobian` are rebuilt on restore.
    pub(crate) atoms: Vec<SaeManifoldAtomSnapshot>,
    pub(crate) logits: Array2<f64>,
    pub(crate) coords: Vec<LatentCoordValues>,
    pub(crate) last_row_layout: Option<SaeRowLayout>,
}

/// Recurrent objective-incumbent identity and its outer-stationarity evidence.
///
/// `consecutive_inner_restores` counts successful joint-fit calls that ended by
/// restoring exactly the same objective-keyed in-call incumbent (checked on the
/// snapshot's decoder, gates, coordinates, evaluator handles, and homotopy
/// dial). Any different terminal state resets the count. The EFS bridge consumes
/// this as a typed termination certificate instead of inferring flatness from a
/// workload-tuned relative-cost threshold.
#[derive(Debug)]
pub(crate) struct SaeFitIncumbent {
    pub(crate) state: SaeManifoldMutableState,
    pub(crate) consecutive_inner_restores: usize,
}

/// Per-atom differential snapshot — see [`SaeManifoldMutableState`].
#[derive(Debug)]
pub(crate) struct SaeManifoldAtomSnapshot {
    pub(crate) decoder_coefficients: Array2<f64>,
    /// Explicit physical log-amplitude `s_k`. A snapshot that banked only the
    /// decoder would mix the banked `B_k` with a different amplitude after a
    /// boundary recovery solve, producing a physically different model from the
    /// one whose objective was banked.
    pub(crate) log_amplitude: f64,
    pub(crate) smooth_penalty: Array2<f64>,
    pub(crate) basis_evaluator: Option<Arc<dyn SaeBasisEvaluator>>,
    pub(crate) basis_second_jet: Option<Arc<dyn SaeBasisSecondJet>>,
    pub(crate) homotopy_eta: f64,
}

#[cfg(test)]
mod soft_rank_charge_flag_tests {
    use crate::manifold::{
        AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
        SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldTerm,
    };
    use gam_terms::latent::LatentManifold;
    use ndarray::Array2;
    use std::sync::Arc;

    /// A minimal unfitted K=1 periodic term — enough to exercise the manual
    /// `Clone for SaeManifoldTerm` field copy (no joint fit needed).
    fn tiny_term() -> SaeManifoldTerm {
        let n = 8usize;
        let p = 4usize;
        let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
        let coords = Array2::<f64>::from_shape_fn((n, 1), |(r, _)| r as f64 / n as f64);
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let mut decoder = Array2::<f64>::zeros((3, p));
        decoder[[1, 0]] = 1.0;
        decoder[[2, 1]] = 1.0;
        let atom = SaeManifoldAtom::new(
            "circle".to_string(),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap();
        let logits = Array2::<f64>::from_elem((n, 1), 1.0);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![coords],
            vec![LatentManifold::Circle { period: 1.0 }],
            AssignmentMode::ibp_map(0.7, 1.0, false),
        )
        .unwrap();
        SaeManifoldTerm::new(vec![atom], assignment).unwrap()
    }

    /// The Theorem K soft-rank-charge opt-in must default OFF (bit-identical hard
    /// path) and, being persisted per-fit config, must PROPAGATE across the manual
    /// `Clone` impl. Stagewise clones the term once per birth and parallel folds
    /// hand clones to rayon workers, so a flag that failed to clone would silently
    /// drop the soft charge in exactly those paths (the reason the earlier
    /// thread-local gate was replaced by this field).
    #[test]
    fn soft_rank_charge_defaults_false_and_clones() {
        let mut term = tiny_term();
        assert!(
            !term.soft_rank_charge(),
            "soft_rank_charge must default OFF (bit-identical historical hard path)"
        );
        assert!(
            !term.clone().soft_rank_charge(),
            "a false flag must clone as false"
        );

        term.set_soft_rank_charge(true);
        assert!(term.soft_rank_charge());
        assert!(
            term.clone().soft_rank_charge(),
            "soft_rank_charge=true must propagate across the manual Clone (stagewise \
             per-birth clone / rayon worker), NOT silently reset"
        );

        term.set_soft_rank_charge(false);
        assert!(!term.soft_rank_charge());
        assert!(!term.clone().soft_rank_charge());
    }
}
