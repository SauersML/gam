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

/// ABSOLUTE FALLBACK explained-variance floor for the reconstruction-collapse
/// guard (#1023), used ONLY when the data-derived bar is un-computable.
///
/// #1522 retired this as the primary collapse threshold: the live bar is
/// [`SAE_COLLAPSE_PCA_EV_FRACTION`]`·pca_ev_ceiling(target, K)` (see
/// `collapse_ev_bar`), which tracks the data's own achievable EV instead of a
/// corpus-tuned constant. This number is reached only on a degenerate target
/// (constant columns / SVD failure → non-finite ceiling), where any positive
/// floor is arbitrary; `0.10` is a deliberately conservative "the fitted matrix
/// must explain at least a tenth of the variance or it is a structural collapse"
/// fallback so the guard still keys on *something* finite in that corner. It is
/// NOT a tuned operating point — a fit on real data is judged against the PCA
/// ceiling, never this constant.
pub(crate) const SAE_FIT_DATA_COLLAPSE_EV_FLOOR: f64 = 0.10;

/// #1522 — fraction of the rank-`K` PCA / Eckart-Young EV ceiling below which a
/// fit counts as a structural co-collapse. The collapse bar is
/// `SAE_COLLAPSE_PCA_EV_FRACTION · pca_ev_ceiling(target, dictionary_rank)`: the
/// ceiling is the BEST EV any rank-`K` linear dictionary could reach on THIS
/// centered target, so the bar is data-derived (scales with what the data
/// actually admits) rather than an absolute corpus-tuned number. `0.5` encodes
/// the decision "a fit that explains less than HALF of the linearly-achievable
/// variance has structurally collapsed, whatever its absolute EV" — a
/// dimensionless ratio with that single, explicit meaning, not a magnitude tuned
/// to any one corpus. It is the sole source for the ratio used at every
/// collapse-guard site.
pub(crate) const SAE_COLLAPSE_PCA_EV_FRACTION: f64 = 0.5;

pub(crate) const SAE_FIT_DATA_COLLAPSE_COST: f64 = 1.0e12;

pub(crate) const SAE_FINAL_EV_DEGRADATION_TOL: f64 = 1.0e-3;

/// #1026 — per-row coordinate re-projection grid resolution for the final
/// reconstruction polish (matches the resolution the curvature-homotopy arrival
/// polish uses in `outer_objective.rs`). The projection snaps each row's latent
/// coordinate to the nearest point of a `resolution`-point decode grid, so a
/// finer grid is a tighter chart re-fit; `256` is the established cost/precision
/// point for the chart re-projection (a circle/sphere/Euclidean grid sweep).
pub(crate) const SAE_FINAL_DECODER_POLISH_PROJECTION_RESOLUTION: usize = 256;

pub(crate) const SAE_SEED_DISPERSION_FLOOR: f64 = 1.0e-12;

/// #1026 decoder-repulsion conditioner strength. Small fixed weight on the
/// collinearity-gated cross-decoder repulsion that injects POSITIVE curvature in
/// the inter-atom co-collapse direction. It is a conditioner/separator, not a
/// primary objective term: it is identically zero (value, gradient, curvature)
/// unless two atom decoders exceed
/// [`SAE_DECODER_REPULSION_COLLINEARITY_GATE`], so it is a strict no-op for
/// well-separated atoms and for `K = 1`. Without it the SAE joint inner Newton
/// solve has NO inter-atom repulsion (data-fit is independent per atom given the
/// soft assignment), so at `K >= 4` on real residual geometry two atoms drift
/// onto the same decoder direction, the per-row `H_tt` block goes near-singular,
/// the reduced β-Schur over-subtracts and goes indefinite, and the inner solve
/// never converges (#1026).
pub(crate) const SAE_DECODER_REPULSION_STRENGTH: f64 = 1.0e-3;

/// #1026 normalized collinearity score
/// `s_jk = ‖B_jB_kᵀ‖²_F / (‖B_j‖²_F·‖B_k‖²_F)` at/above which the decoder
/// repulsion engages. Below this the smoothstep gate is exactly 0 (zero penalty
/// / gradient / curvature), so healthy fits whose atoms point in distinct
/// directions never activate the term. `s_jk` is scale-free in each decoder's
/// magnitude (it measures ANGLE, not norm) so it cannot fight the data-fit's
/// choice of decoder scale.
pub(crate) const SAE_DECODER_REPULSION_COLLINEARITY_GATE: f64 = 0.5;

// ── #1026 / #1522 interior-point COLLAPSE-PREVENTION barriers ────────────────
//
// The collinearity-gated `SAE_DECODER_REPULSION_*` term above is a SEPARATOR for
// already-large, already-collinear decoders; it is a strict no-op at the
// principal failure state (a decoder with norm → 0) because a zero decoder has
// no direction to be collinear with. The two barriers below are the actual
// anti-collapse core: they are interior-point log-barriers that DIVERGE at the
// collapse boundary, so the inner Newton can never reach it.

/// #1026/#1522 AMPLITUDE barrier strength `μ_s`. Penalty
/// `P_amp = -μ_s · Σ_{k active} log(s_k² + ε)` with `s_k² = ‖B_k‖²_F`. Its
/// decoder gradient `∂P/∂B_k = -2μ_s/(s_k²+ε)·B_k` has magnitude `∝ 1/‖B_k‖`
/// near zero, a genuine OUTWARD restoring force (unlike the angle penalty, whose
/// force vanishes with the norm). Fixed strength — there is no μ-continuation
/// schedule; the only per-fit scaling is the minibatch chunk fraction `n_chunk/N`
/// applied at the call site (`penalty_scale`). The value is empirically tuned to
/// be commensurate with the unit-scale reconstruction objective and is overridable
/// at runtime via [`set_sae_barrier_overrides`] to sweep the response surface.
///
/// #1026/#1522 RETUNE (2026-06-24, real OLMo-L25 EV-vs-K ladder): the original
/// `100` is NOT commensurate — at a healthy decoder `s_k²~O(1)` the outward force
/// `2μ_s/(s_k²+ε)·B_k ≈ 200·B_k` dwarfs the unit-scale reconstruction gradient,
/// pinning the outer fit in a barrier-dominated flat valley (a healthy K=1 fit
/// stalled 3494 s with no recorded EV at `μ_s=100`). At `μ_s=1` the force is
/// `~2·B_k`, the K=1 fit is fast, and the curved held-out EV climbs cleanly with K
/// (0.2132 → 0.3256 → 0.4891 for K=1,2,4 on OLMo L25). So `1` IS the
/// data-commensurate strength; `100` was itself an over-strong magic constant.
pub(crate) const SAE_AMPLITUDE_BARRIER_STRENGTH: f64 = 1.0;

// #1026/#1522 — RUNTIME barrier-tuning overrides. The strengths and the
// active-atom GATE MODE are read through the accessors below instead of the
// raw consts so a SINGLE compiled wheel can sweep the entire
// (μ_amp × μ_sep × gate) response surface from Python (`set_sae_barrier_overrides`),
// rather than recompiling the (one-rustc-invocation) gam crate per constant.
// A quiet-NaN sentinel means "unset → use the compiled default const", so 0.0
// remains a legitimate swept value (barrier fully disabled). With no override
// set, the accessor returns exactly the const, so behaviour is unchanged.
static SAE_AMP_STRENGTH_OVERRIDE_BITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0x7ff8_0000_0000_0000);
static SAE_SEP_STRENGTH_OVERRIDE_BITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0x7ff8_0000_0000_0000);
/// Active-atom gate mode for the amplitude barrier: 0 = decoder-norm (the
/// correct default — hold up every live decoder), 1 = legacy assignment-energy
/// gate (drains during co-collapse; kept only to A/B the regression), 2 =
/// unconditional (every dictionary slot). Sweepable to attribute the OLMo
/// co-collapse to amplitude vs collinear (separation) failure in one build.
static SAE_BARRIER_GATE_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub(crate) fn sae_amplitude_barrier_strength() -> f64 {
    let v =
        f64::from_bits(SAE_AMP_STRENGTH_OVERRIDE_BITS.load(std::sync::atomic::Ordering::Relaxed));
    if v.is_nan() {
        SAE_AMPLITUDE_BARRIER_STRENGTH
    } else {
        v
    }
}

pub(crate) fn sae_separation_barrier_strength() -> f64 {
    let v =
        f64::from_bits(SAE_SEP_STRENGTH_OVERRIDE_BITS.load(std::sync::atomic::Ordering::Relaxed));
    if v.is_nan() {
        SAE_SEPARATION_BARRIER_STRENGTH
    } else {
        v
    }
}

pub(crate) fn sae_barrier_gate_mode() -> u8 {
    SAE_BARRIER_GATE_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Set the process-global SAE-barrier tuning overrides (one wheel, many configs).
/// `amp_strength`/`sep_strength` are NaN to clear an override back to the compiled
/// default; `gate_mode` selects the amplitude active-atom gate (see
/// [`SAE_BARRIER_GATE_MODE`]). Called from the gamfit Python FFI sweep driver.
pub fn set_sae_barrier_overrides(amp_strength: f64, sep_strength: f64, gate_mode: u8) {
    SAE_AMP_STRENGTH_OVERRIDE_BITS
        .store(amp_strength.to_bits(), std::sync::atomic::Ordering::Relaxed);
    SAE_SEP_STRENGTH_OVERRIDE_BITS
        .store(sep_strength.to_bits(), std::sync::atomic::Ordering::Relaxed);
    SAE_BARRIER_GATE_MODE.store(gate_mode, std::sync::atomic::Ordering::Relaxed);
}

/// #1026/#1522 AMPLITUDE barrier softening `ε` (added inside `log(s_k²+ε)` and in
/// every denominator). Keeps the barrier finite and the PSD majorizer bounded at
/// an exactly-zero decoder, while remaining negligible against a healthy
/// decoder's `‖B_k‖²_F` (which is O(1) on unit-scale data).
pub(crate) const SAE_AMPLITUDE_BARRIER_EPS: f64 = 1.0e-8;

/// #1026/#1522 SEPARATION barrier strength `μ_C`. Penalty
/// `P_sep = -μ_C · Σ_{j<k} q_jk · log(1 - c_jk² + ε)` on the NORMALIZED decoder
/// shapes `U_k = B_k/‖B_k‖`, `c_jk² = ‖U_jU_kᵀ‖²_F` (squared principal-angle
/// cosine ∈ [0,1]), weighted by the normalized coactivation
/// `q_jk = (Σ_i a_ij a_ik)/sqrt(Σa_ij²·Σa_ik²) ∈ [0,1]`. Force on `c_jk`
/// `∂P/∂c_jk = 2μ_C q_jk c_jk/(1-c_jk²+ε)` DIVERGES as atoms align (`c_jk→1`) and
/// is exactly 0 when `c_jk = 0` — and, unlike the threshold repulsion, it does
/// NOT switch off at small amplitude (it sees only the SHAPE `U_k`).
///
/// #1026/#1522 RETUNE (2026-06-24): paired down to `10` alongside the amplitude
/// retune (`SAE_AMPLITUDE_BARRIER_STRENGTH` 100→1) — this is the
/// `{μ_amp=1, μ_sep=10}` config that produced the clean OLMo L25 EV-vs-K climb.
/// The separation force already DIVERGES near alignment (`1/(1-c²+ε)`), so a
/// large flat-region prefactor is unnecessary and only adds outer-objective
/// stiffness away from collapse.
pub(crate) const SAE_SEPARATION_BARRIER_STRENGTH: f64 = 10.0;

/// #1026/#1522 SEPARATION barrier softening `ε` in `log(1 - c_jk² + ε)`. Bounds
/// the barrier (and its PSD majorizer) at the exact-alignment limit `c_jk² = 1`.
pub(crate) const SAE_SEPARATION_BARRIER_EPS: f64 = 1.0e-6;

/// #1026/#1522 decoder-norm floor below which an atom is treated as inactive /
/// shape-undefined for the SEPARATION barrier (its `U_k` is ill-conditioned). The
/// AMPLITUDE barrier is what restores such an atom; the separation barrier simply
/// abstains for the pair until the amplitude barrier has lifted the norm above
/// this floor.
pub(crate) const SAE_BARRIER_ACTIVE_NORM_FLOOR: f64 = 1.0e-6;

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
    /// gauge's [`crate::terms::analytic_penalties::WeightField`] is constructed
    /// from the same object, so a likelihood-metric ≠ gauge-metric state is
    /// unrepresentable. `None` ⇒ Euclidean / isotropic (magic-by-default: the
    /// metric is selected by whether per-row Fisher factors were installed, not
    /// by a flag), which is bit-for-bit the historical isotropic `φ̂` path.
    pub(crate) row_metric: Option<crate::inference::row_metric::RowMetric>,
    /// #976 Layer-1 guard ledger for the most recent joint fit: every
    /// active-mass breach with the action taken (re-seed / terminal). Cleared
    /// at the start of each `run_joint_fit_arrow_schur`; read post-fit via
    /// [`SaeManifoldTerm::collapse_events`] and carried onto the
    /// structure-search [`crate::solver::structure_search::SearchLedger`].
    pub(crate) collapse_events: Vec<CollapseEvent>,
    /// Per-row **design honesty weights** (#991): Horvitz–Thompson inclusion
    /// corrections from a designed corpus subsample
    /// ([`crate::inference::row_measure::RowSamplingMeasure::designed_subsample`] /
    /// [`crate::terms::sae::corpus::designed_target`]), self-normalized to
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
    pub(crate) best_cocollapse_incumbent: Option<(f64, SaeManifoldMutableState)>,
    /// #1026 decoder-repulsion gate, frozen per assembly (lagged-diffusivity
    /// discipline, exactly like [`SaeManifoldAtom::smooth_penalty`]): the
    /// symmetric `(K, K)` matrix of collinearity gate weights
    /// `gate(s_jk)·SAE_DECODER_REPULSION_STRENGTH` computed from the decoder
    /// state at assembly entry. Both the assembly (gradient into `gb`, PSD
    /// curvature into `hbb`) and the line-search value path
    /// ([`Self::penalized_objective_total`]) read THIS frozen gate, so the
    /// repulsion's value, gradient, and curvature stay mutually consistent
    /// across a Newton step even as the trial decoders move (the cross-Gram
    /// quadratic moves with the trial; only the gate weight is frozen). `None`
    /// when no repulsion is active (`K < 2`, or every pair below the gate
    /// threshold — the strict no-op case). Refreshed at the same chokepoint as
    /// the smoothness Gram; not part of the persisted term identity (Clone
    /// starts `None`).
    pub(crate) decoder_repulsion_gate: Option<Array2<f64>>,
    /// #1026: the load-bearing curved-vs-linear hybrid-split verdict, computed
    /// once in [`Self::canonicalize_charts_post_fit`] after the joint fit
    /// converges. Each eligible `d = 1` atom's fitted curved image is adjudicated
    /// against its straight (linear special-case) sub-model on the common
    /// rank-aware Laplace evidence scale. `None` until the post-fit pass runs (or
    /// when no atom is eligible). Surfaced in the Python model output so a user
    /// sees which atoms genuinely earn their curvature and which collapse to the
    /// linear tail. Read via [`Self::hybrid_split_report`].
    pub(crate) hybrid_split_report: Option<crate::terms::sae::hybrid_split::SaeHybridSplitReport>,
    /// Per-atom inner-decoder-smooth byproducts harvested post-fit (#1097 /
    /// #1103), one entry per atom in [`Self::atoms`] order. Each is the fixed
    /// fitted snapshot the residual-gauge certificate's three post-PIRLS atom
    /// inference reports consume
    /// ([`crate::terms::sae::identifiability::AtomInnerFit`]). `None` until
    /// [`Self::set_atom_inner_fits`] runs (it needs the reconstruction target
    /// `Z`, available only at the post-fit harness seam where the dispersion is
    /// also profiled); a per-atom `None` means that atom had no active rows or a
    /// degenerate inner design. Read by [`Self::to_residual_gauge_model`], which
    /// attaches each onto its [`crate::terms::sae::identifiability::FittedAtom`].
    pub(crate) atom_inner_fits:
        Option<Vec<Option<crate::terms::sae::identifiability::AtomInnerFit>>>,
    /// #1228 — the trained dictionary's hybrid-collapsed linear images, attached
    /// to an OOS term so held-out reconstruction decodes verdict-linear `d = 1`
    /// slots by the SAME straight sub-model the training reconstruction used.
    /// `None` on a fitted term (its collapse policy lives in
    /// [`Self::hybrid_split_report`]); `Some` only on the fresh OOS term built in
    /// `sae_manifold_predict_oos`, set from the serialized payload via
    /// [`Self::set_hybrid_linear_images`]. Consulted by
    /// [`Self::hybrid_linear_image_map`] so train and OOS share one collapse map.
    pub(crate) oos_linear_images: Option<Vec<crate::terms::sae::hybrid_split::AtomLinearImage>>,
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
            hybrid_split_report: self.hybrid_split_report.clone(),
            atom_inner_fits: self.atom_inner_fits.clone(),
            oos_linear_images: self.oos_linear_images.clone(),
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
