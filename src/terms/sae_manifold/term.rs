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

/// Largest decoder (`β`) block dimension for which the outer-gradient
/// conditioning path may additionally probe the β coordinate basis for a
/// near-null subspace of the joint Hessian (issue #1051, #1095).
///
/// The closed-form gauge orbit ([`SaeManifoldTerm::dense_step_gauge_vectors`])
/// only covers the *chart* reparametrisation freedom (constant + linear
/// coordinate fields). It does NOT cover a **rank-deficient decoder design** —
/// e.g. a euclidean-1D atom fit to a straight line in a `p = 2` ambient leaves
/// the decoder column space rank-1, so one decoder direction is unidentified by
/// the data and the joint Hessian acquires a near-null direction that lives in
/// the β block, not the gauge orbit. That direction is exactly a Faddeev-Popov
/// gauge of the *same* kind (a flat direction of the evidence quotient), so it
/// is deflated identically — but only after the β basis is admitted as a
/// deflation candidate. The dense `k×k` Rayleigh eigendecomposition that
/// resolves it is `O(k³)`, so it is gated to moderate β blocks; large-`p`
/// LLM-scale fits keep the pure gauge-orbit path untouched (they reach low
/// decoder rank through the Grassmann frame, which reduces the border width
/// from `M·p` to `M·r` where `r ≪ p`, so `k ≤ M·r` is always small). PCA-
/// reduced fits (p ≈ 32–128) with the Grassmann frame active can have
/// `k = M·r` up to ~512 (e.g. m=8 basis fns, p=32, r=8 → k=64, but for
/// m=16 → k=128, m=32 → k=256); 512 covers all typical small-atom PCA cases
/// while keeping the O(k³) cost ≈ 0.13B ops — negligible next to the solve.
pub(crate) const SAE_OUTER_GRADIENT_BETA_NULL_PROBE_MAX_DIM: usize = 512;

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

/// Maximum number of LM ridge-escalation attempts before declaring the per-row
/// Hessian unfactorable.
const SAE_MANIFOLD_ROW_RIDGE_MAX_ATTEMPTS: usize = 12;

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

/// Final fitted-data explained-variance floor for the reconstruction-collapse
/// guard (#1023). This is deliberately an effectively-zero threshold: ordinary
/// under-fitting is a model-quality issue, but returning a K>=1 active SAE whose
/// fitted matrix is indistinguishable from the column mean is a structural
/// collapse and must enter the #976 CollapseEvent ledger.
pub(crate) const SAE_FIT_DATA_COLLAPSE_EV_FLOOR: f64 = 0.10;

pub(crate) const SAE_FIT_DATA_COLLAPSE_COST: f64 = 1.0e12;

pub(crate) const SAE_PRISTINE_SEED_EV_RETAIN_FLOOR: f64 = 0.95;

pub(crate) const SAE_FINAL_EV_DEGRADATION_TOL: f64 = 1.0e-3;

pub(crate) const SAE_SEED_DISPERSION_FLOOR: f64 = 1.0e-12;
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
    /// ([`crate::inference::row_measure::RowMeasure::designed_subsample`] /
    /// [`crate::terms::sae_corpus::designed_target`]), self-normalized to
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
    /// ([`crate::sae_identifiability::AtomInnerFit`]). `None` until
    /// [`Self::set_atom_inner_fits`] runs (it needs the reconstruction target
    /// `Z`, available only at the post-fit harness seam where the dispersion is
    /// also profiled); a per-atom `None` means that atom had no active rows or a
    /// degenerate inner design. Read by [`Self::to_residual_gauge_model`], which
    /// attaches each onto its [`crate::sae_identifiability::FittedAtom`].
    pub(crate) atom_inner_fits: Option<Vec<Option<crate::sae_identifiability::AtomInnerFit>>>,
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
            border_hbb_workspace: Array2::<f64>::zeros((0, 0)),
            certificate_dispersion: self.certificate_dispersion,
            curvature_walk_report: self.curvature_walk_report.clone(),
            expected_evidence_gauge_deflated_directions: self
                .expected_evidence_gauge_deflated_directions,
            hybrid_split_report: self.hybrid_split_report.clone(),
            atom_inner_fits: self.atom_inner_fits.clone(),
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
struct SaeManifoldMutableState {
    /// Per-atom `(basis_values, basis_jacobian, decoder_coefficients, smooth_penalty)`.
    atoms: Vec<(Array2<f64>, Array3<f64>, Array2<f64>, Array2<f64>)>,
    logits: Array2<f64>,
    coords: Vec<LatentCoordValues>,
    last_row_layout: Option<SaeRowLayout>,
}
