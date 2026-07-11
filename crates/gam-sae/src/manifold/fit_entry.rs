//! The single, python-free SAE-manifold fit ENTRY (#2236 Increment 1).
//!
//! This module owns the fit ORCHESTRATION that historically lived inside
//! `gam-pyffi`'s `sae_manifold_fit_inner`: constructing the
//! [`SaeManifoldOuterObjective`], running the outer ρ cascade
//! ([`OuterProblem`]) or the fixed-ρ inner solve, the #2021 structured-residual
//! outer alternation (including its Λ nursery→promotion births), the #977/#997
//! evidence-guarded structure search, every post-fit diagnostic
//! (shape-uncertainty bands, trust/fit reports, coordinate fidelity, …), and
//! the coherent fitted-model diagnostics. A binding only needs to assemble
//! the incoming arrays into a configured [`SaeManifoldTerm`] and typed
//! [`SaeFitRequest`], execute [`run_sae_manifold_fit`] on its worker thread, and
//! marshal the returned [`SaeFitReport`].
//!
//! The analytic-penalty registry is the only seam needed to keep this library
//! entry free of python and of the crate that sits above `gam-sae` in the
//! dependency graph:
//!
//! The registry (built by `gam-models`, which depends on `gam-sae`) is passed in
//! PRE-BUILT and cloned at each of the three objective-construction sites —
//! identical to the binding rebuilding it from the same `latent_payload` +
//! descriptor JSON each time. Every numerical fit policy, including the #2071
//! Beta-null residual-promotion threshold, is derived inside this entry.
//!
//! Interruptibility is preserved by the caller: the whole entry runs on the
//! binding's GIL-released worker thread and shares the `cancel` flag, which each
//! inner objective polls and bails its next outer eval on.

use ndarray::{Array1, Array2};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use gam_math::probability::beta_quantile;
use gam_problem::{EstimationError, MetricProvenance};
use gam_solve::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
use gam_solve::rho_optimizer::{OuterProblem, OuterResult};
use gam_solve::seeding::SeedConfig;
use gam_solve::structure_search::{MoveBudget, StructureMove};
use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
use gam_terms::inference::structure_evidence::StructureLedger;

use crate::structure_harvest;

use super::{
    AssignmentMode, SaeManifoldFitDiagnostics, SaeManifoldLoss, SaeManifoldOuterObjective,
    SaeManifoldRho, SaeManifoldTerm, SaeOuterTermination, SaeShapeUncertainty, SaeTrustDiagnostics,
};

/// Hard cap on evidence-certified #2021 whitened-residual refit passes.
pub const STRUCTURED_RESIDUAL_PASSES_MAX: usize = 4;
/// Canonical structured-residual alternation budget. The iid-only A/B mode was
/// removed; every production fit starts with this evidence-refined budget.
pub const STRUCTURED_RESIDUAL_PASSES_DEFAULT: usize = 2;

/// Absolute precision floor on the RELATIVE post-dictionary residual energy
/// `‖Z − Ẑ‖²_F / ‖Z‖²_F` below which the structured-residual pass is skipped and
/// the fit degrades to the already-certified pass-0 iid model.
///
/// A dictionary that explains the target to within this bound leaves only the
/// fit's own numerical-convergence noise as "residual": there is genuinely no
/// structured covariance to whiten. Fitting a residual-covariance model on that
/// noise is DEGENERATE — the idiosyncratic diagonal `D` collapses toward its
/// floor (`residual_factor` floors it at `1e-6 · mean_var`, still ~6 orders
/// below a genuine noise scale on near-noiseless data),
/// the whitening metric `1/D` becomes near-singular, and the whitened-residual
/// penalized quasi-Laplace criterion the outer ρ-optimizer then descends is ill-conditioned with no interior
/// stationary point. The outer correctly refuses to certify a non-stationary
/// optimum, so a fit that SHOULD succeed (its iid pass-0 already certified)
/// instead fails. Skipping the structured pass when there is nothing to model is
/// the correct behavior, not a workaround.
///
/// DERIVED: the value `1e-10` on the relative *energy* corresponds to a residual
/// RMS of `1e-5` relative to the target RMS — an order of magnitude below the
/// inner SAE solve's own convergence scale (`SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL`
/// `= 1e-8`), so it triggers only on numerically-exact reconstructions while
/// leaving every genuinely-structured residual (relative energy `≥ ~1e-8`, i.e. a
/// fit that leaves `≥ 1e-4` RMS unexplained) to run the pass unchanged.
pub(crate) const STRUCTURED_RESIDUAL_MIN_REL_ENERGY: f64 = 1.0e-10;

/// #2071 residual-promotion alignment threshold under the random-direction
/// null. Rank one has no informative angle, so its threshold is exactly one.
/// Keeping this derivation in `gam-sae` makes the typed fit entry self-sufficient
/// for Rust, CLI, and binding callers alike.
fn promotion_alignment_threshold(factor_rank: usize) -> f64 {
    if factor_rank <= 1 {
        return 1.0;
    }
    let rank = factor_rank as f64;
    beta_quantile(0.95, 0.5, (rank - 1.0) / 2.0)
        .sqrt()
        .clamp(0.0, 1.0)
}

/// One #2021 structured-residual outer-alternation pass's diagnostic record. The
/// binding serializes a `&[StructuredResidualPassDiagnostic]` into the payload;
/// producing it here keeps the alternation (and its accounting) python-free.
#[derive(Clone, Debug)]
pub struct StructuredResidualPassDiagnostic {
    pub pass: usize,
    pub gamma: f64,
    pub factor_rank: usize,
    pub log_evidence: f64,
    pub factor_energy: f64,
    pub diagonal_mean: f64,
    pub dispersion_before: f64,
    pub dispersion_after: f64,
    pub log_lambda_smooth_before: Vec<f64>,
    pub log_lambda_smooth_after: Vec<f64>,
}

/// The python-facing label for a [`MetricProvenance`] (#980). Centralized so a
/// new provenance variant is labelled in exactly one place; shared by the fit
/// entry and every binding site that surfaces `metric_provenance`.
pub fn metric_provenance_label(provenance: MetricProvenance) -> &'static str {
    match provenance {
        MetricProvenance::Euclidean => "Euclidean",
        MetricProvenance::OutputFisher { .. } => "OutputFisher",
        MetricProvenance::OutputFisherDownstream { .. } => "OutputFisherDownstream",
        MetricProvenance::BehavioralFisher { .. } => "BehavioralFisher",
        MetricProvenance::WhitenedStructured { .. } => "WhitenedStructured",
    }
}

/// Fit the whitened residual-covariance model on the current fitted residuals of
/// `term` against `target`, or `Ok(None)` when there is nothing to mine (fewer
/// than two output channels). Errors propagate a genuine fit breakdown (#2070/
/// #2021) rather than degrading silently to prior-pass geometry.
fn sae_structured_residual_model(
    term: &SaeManifoldTerm,
    target: ndarray::ArrayView2<'_, f64>,
) -> Result<Option<StructuredResidualModel>, String> {
    let fitted = term.try_fitted_target_aware(target, None)?;
    let (n, p) = fitted.dim();
    // Need >= 2 output channels for an off-diagonal factor subspace.
    if n == 0 || p <= 1 {
        return Ok(None);
    }
    if target.dim() != (n, p) {
        return Err(format!(
            "sae_structured_residual_model: target must be ({n}, {p}); got {:?}",
            target.dim()
        ));
    }
    // R = target − fitted (post-dictionary residual). Bind `fitted` first so the
    // owned temporary outlives the in-place subtraction.
    let mut residuals = target.to_owned();
    residuals -= &fitted;
    // Degeneracy guard: when the dictionary already explains the target to within
    // numerical precision, the residual is pure convergence noise with no
    // structured covariance to model. Fitting a residual-factor model on it
    // collapses the idiosyncratic diagonal `D → 0`, the whitening `1/D` goes
    // near-singular, and the whitened-residual penalized quasi-Laplace criterion the outer optimizer descends
    // has no interior stationary point (a fit that SHOULD certify then refuses).
    // Degrade to the pass-0 iid fit (which already certified) instead. Scale-free:
    // the floor is on the residual energy RELATIVE to the target energy. See
    // `STRUCTURED_RESIDUAL_MIN_REL_ENERGY`.
    let target_energy: f64 = target.iter().map(|v| v * v).sum();
    let residual_energy: f64 = residuals.iter().map(|v| v * v).sum();
    if residual_energy <= STRUCTURED_RESIDUAL_MIN_REL_ENERGY * target_energy {
        return Ok(None);
    }
    // Activity = per-row total assignment mass (mirrors structure_harvest.rs and
    // the fit tail's own assignment read).
    let assignments = term.assignment.assignments();
    let activity: ndarray::Array1<f64> = (0..n).map(|r| assignments.row(r).sum()).collect();
    // Let the evidence ladder pick the rank up to p-1 (`fit` re-caps to p-1 and
    // scores r = 0..=cap, keeping the penalized-evidence maximizer).
    let max_factor_rank = p.saturating_sub(1);
    match StructuredResidualModel::fit(ResidualFactorInput {
        residuals: residuals.view(),
        activity: activity.view(),
        max_factor_rank,
    }) {
        Ok(m) => Ok(Some(m)),
        // Propagate a genuine fit failure instead of swallowing it (#2070/#2021).
        // The only benign "nothing to mine" case — fewer than two output channels
        // — is already handled by the early `Ok(None)` above, and the evidence
        // ladder always scores at least rank 0, so every error reaching here is a
        // real breakdown (non-finite residuals/activity, a dimension mismatch, or
        // an inner-alternation numerical failure). Accepting-on-any-error would
        // silently degrade to prior-pass geometry and hide the failure; surface it.
        Err(e) => Err(format!(
            "sae_structured_residual_model: structured residual-covariance fit failed: {e}"
        )),
    }
}

/// Everything the payload-dict build needs from a completed SAE-manifold fit. The
/// binding reads these fields directly (no python object lives here), re-deriving
/// per-atom vectors (`atom_basis`, `atom_dim`, `k_atoms`) from `term` on its side.
pub struct SaeFitReport {
    pub term: SaeManifoldTerm,
    pub rho: SaeManifoldRho,
    /// Penalized loss of the fitted model.
    pub loss: SaeManifoldLoss,
    /// Terminal custom penalized quasi-Laplace criterion at the outer stationary
    /// state, including its PSD/Gauss--Newton factor and rank charges and preceding any optional
    /// image-frozen post-fit chart canonicalization. It is not normalized
    /// LAML, REML, or model evidence, and it is not
    /// `-loss.total()`.
    pub penalized_quasi_laplace_criterion: f64,
    pub assignments: Array2<f64>,
    pub fitted: Array2<f64>,
    pub active_mask: Vec<bool>,
    pub reconstruction_r2: f64,
    pub outer_termination: SaeOuterTermination,
    pub shape_uncertainty: SaeShapeUncertainty,
    pub metric_provenance: &'static str,
    pub structured_residual_diagnostics: Vec<StructuredResidualPassDiagnostic>,
    pub trust_diagnostics: SaeTrustDiagnostics,
    pub fit_diagnostics: SaeManifoldFitDiagnostics,
    /// Serialized per-round structure-search ledger (#997) as a JSON string;
    /// `None` when the search did not run (skipped by K ceiling or
    /// `run_structure_search == false`).
    pub structure_search_json: Option<String>,
    /// The anytime-valid structure certificate (#1058/#984), serialized JSON.
    pub structure_certificate_json: String,
    /// The reported `log_alpha` (ordered Beta--Bernoulli concentration or the caller's α fallback).
    pub reported_log_alpha: f64,
}

/// Optimization phase that owns an SAE wall-survival checkpoint and convergence
/// verdict. Structured phases include the configured pass count because their
/// residual-metric damping `γ = pass / (total_passes + 1)` depends on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaeFitStage {
    Primary,
    StructuredResidual {
        /// One-based pass number.
        pass: usize,
        total_passes: usize,
    },
}

impl SaeFitStage {
    fn checkpoint_tag(self) -> String {
        match self {
            Self::Primary => "primary".to_string(),
            Self::StructuredResidual { pass, total_passes } => {
                format!("structured-residual-{pass}-of-{total_passes}")
            }
        }
    }
}

impl std::fmt::Display for SaeFitStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Primary => f.write_str("primary"),
            Self::StructuredResidual { pass, total_passes } => {
                write!(f, "structured-residual pass {pass}/{total_passes}")
            }
        }
    }
}

/// Typed failure from [`run_sae_manifold_fit`]. A non-converged outer run keeps
/// the complete [`OuterResult`] as machine-readable evidence; it is never
/// flattened into a message or converted into a fit.
#[derive(Debug)]
pub enum SaeFitError {
    Fit(String),
    OuterRun {
        stage: SaeFitStage,
        source: EstimationError,
    },
    OuterDidNotConverge {
        stage: SaeFitStage,
        result: Box<OuterResult>,
    },
}

impl From<String> for SaeFitError {
    fn from(message: String) -> Self {
        Self::Fit(message)
    }
}

impl std::fmt::Display for SaeFitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fit(message) => f.write_str(message),
            Self::OuterRun { stage, source } => {
                write!(f, "SAE manifold {stage} outer search failed: {source}")
            }
            Self::OuterDidNotConverge { stage, result } => {
                let grad = result
                    .final_grad_norm
                    .map(|value| format!("{value:.6e}"))
                    .unwrap_or_else(|| "unmeasured".to_string());
                write!(
                    f,
                    "SAE manifold {stage} outer search stopped without a stationarity \
                     certificate (iterations={}, final_value={:.6e}, final_grad_norm={}, \
                     plan={}, stop_reason={:?}, rho_checkpoint={:?}); refusing to mint a fit",
                    result.iterations,
                    result.final_value,
                    grad,
                    result.plan_used,
                    result.operator_stop_reason,
                    result.rho,
                )
            }
        }
    }
}

impl std::error::Error for SaeFitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OuterRun { source, .. } => Some(source),
            Self::Fit(_) | Self::OuterDidNotConverge { .. } => None,
        }
    }
}

/// Give each fit phase its own checkpoint address. The target/K fingerprint
/// still verifies the payload; the phase tag prevents a structured-metric state
/// from being installed into the primary Euclidean objective, and includes the
/// total pass count because it determines the structured damping schedule.
pub(crate) fn scope_outer_checkpoint_to_stage(
    objective: &mut SaeManifoldOuterObjective,
    stage: SaeFitStage,
) {
    let mut path =
        super::checkpoint::SaeFitCheckpoint::default_store_path(&objective.checkpoint_fingerprint);
    path.set_file_name(format!(
        "{}.{}.json",
        objective.checkpoint_fingerprint.content_hash,
        stage.checkpoint_tag(),
    ));
    objective.checkpoint_path = path;
}

/// Ownership gate for fit-producing outer phases. The objective is returned
/// only with a converged [`OuterResult`]; otherwise it is dropped without
/// checkpoint cleanup and the complete verdict is retained in a typed error.
pub(crate) fn certify_outer_stage(
    objective: SaeManifoldOuterObjective,
    stage: SaeFitStage,
    run_result: Result<OuterResult, EstimationError>,
) -> Result<SaeManifoldOuterObjective, SaeFitError> {
    match run_result {
        Ok(result) if result.converged => {
            let mut objective = objective;
            match objective.certify_outer_result(&result) {
                Ok(()) => Ok(objective),
                Err(_) => Err(SaeFitError::OuterDidNotConverge {
                    stage,
                    result: Box::new(result),
                }),
            }
        }
        Ok(result) => Err(SaeFitError::OuterDidNotConverge {
            stage,
            result: Box::new(result),
        }),
        Err(source) => Err(SaeFitError::OuterRun { stage, source }),
    }
}

/// Fully typed request for the single SAE-manifold fit entry.
///
/// Seed construction is deliberately outside this type: callers build and
/// validate the [`SaeManifoldTerm`] once, then hand ownership of the complete
/// per-fit state to the engine.  The request owns every orchestration choice so
/// bindings do not need a parallel fit driver or process-global configuration.
pub struct SaeFitRequest {
    pub base_term: SaeManifoldTerm,
    pub target: Array2<f64>,
    pub registry: AnalyticPenaltyRegistry,
    pub initial_rho: SaeManifoldRho,
    pub max_iter: usize,
    pub learning_rate: f64,
    pub ridge_ext_coord: f64,
    pub ridge_beta: f64,
    pub alpha: f64,
    pub isometry_pin_active: bool,
    pub metric_provenance: &'static str,
    pub promote_from_residual: bool,
    pub run_structure_search: bool,
    pub run_outer_rho_search: bool,
    /// Number of structured-residual whitening passes (each installs a NEW
    /// row-metric likelihood and re-runs the entire outer ρ search). `None` =
    /// the historical default ([`STRUCTURED_RESIDUAL_PASSES_DEFAULT`] = 2);
    /// `Some(0)` = the UNBUNDLED direct path: seed → single certified fit on
    /// the iid likelihood. Together with `promote_from_residual = false` and
    /// `run_structure_search = false` this is the stage-5 "which path did my
    /// fit take" answer: exactly one.
    pub structured_residual_passes: Option<usize>,
    pub cancel: Option<Arc<AtomicBool>>,
}

/// Run the SAE-manifold fit end-to-end from a fully-constructed, fully-configured
/// seed `base_term` and its seed ρ. This is the python-free single source the
/// binding, the CLI, and Rust library users all call. `base_term` must already
/// carry every per-fit switch the binding installs (fit config, temperature
/// schedule, softmax active cap, row metric, row loss
/// weights, and the cold routing seed refinement) — this entry owns the fit and
/// everything after it, not the seed construction.
///
/// * `registry` is the pre-built analytic-penalty registry; it is cloned at each
///   objective-construction site (three at most: pass 0, each structured pass, and
///   the post-search joint shape recompute).
/// * `cancel`, when present, is polled by every inner objective; the caller sets
///   it on interrupt so the abandoned worker's next outer eval bails.
pub fn run_sae_manifold_fit(mut request: SaeFitRequest) -> Result<SaeFitReport, SaeFitError> {
    // #2023 Increment 5 — Tier-0 shared-mean peel as the ONE entry's NATIVE
    // preprocessing (the "seed policy" tier of the tiered schedule, folded into the
    // single fit rather than a separate surface). The shared column mean μ is the
    // global DC that a raw activation target carries; left in the target it is the
    // co-collapse-to-mean magnet (a constant "zombie" atom loads it and survives
    // selection, #2082/#1893). Peeling it once here makes that class EV-invisible by
    // construction on the primary path — the exact guarantee the C4 tier-0 tests
    // prove for a hand-built term, now wired into production.
    //
    // Mean ownership is exactly one stage (the DOUBLE-SUBTRACTION HAZARD): when the
    // caller has ALREADY installed a Tier-0 mean on the seed term (already-centered
    // upstream data-prep, e.g. the COMPOSE `tier0.json` mean), that stage owns μ and
    // the reconstruction add-back is already wired — run verbatim, do not peel again.
    // Otherwise compute μ from THIS target and run the WHOLE fit on `Z − μ` (so every
    // internal decoder LSQ, cold-start, structured-residual pass, and EV sees the
    // de-meaned target — no stage double-counts μ), then attach μ to the fitted
    // artifact. Every reconstruction path already adds μ back
    // (`add_tier0_mean_inplace`), so the returned term is self-contained; the
    // returned reconstruction arrays are lifted back to raw-target space here.
    // Reconstruction R² is mean-invariant (both RSS and the centered TSS remove μ),
    // so it is identical either way.
    if request.base_term.tier0_mean().is_some() {
        return run_sae_manifold_fit_on_target(request);
    }
    let Some(mu) = request.target.mean_axis(ndarray::Axis(0)) else {
        // Empty target (N = 0): nothing to peel; the inner entry validates shapes.
        return run_sae_manifold_fit_on_target(request);
    };
    for mut row in request.target.rows_mut() {
        row -= &mu;
    }
    // Tier-0 INPUT STANDARDIZATION — the conditioning half of the peel. There is
    // no column equilibration anywhere else in the fit path, so a raw activation
    // target's column-norm spread (measured ~1.3e4, joint Hessian κ ≈ 1e8 on
    // #2015) directly sets the linear contraction rate of the majorized inner
    // solver — the driver of the "~1e3 iterations then refusal" wall. Fit on
    // `(Z − μ)/σ` with σ_c the per-column RMS of the centered target; the term
    // stores σ next to μ and every reconstruction lifts back exactly
    // (`μ + σ ⊙ x̂` in `add_tier0_mean_inplace`), so the model is self-contained
    // in raw units and reconstruction is exact by construction — only the
    // optimization geometry (and the equal-column-weight penalty pricing, the
    // intended modeling change) differs.
    //
    // Gates: (a) a column whose centered RMS is below `√ε · max σ` is
    // numerically empty — standardizing it would amplify representation noise,
    // so it keeps unit scale (the scalar-type-derived floor, no tuning);
    // (b) behavior / crosscoder fits are excluded: their targets carry the
    // `√λ_y`-scaled block-encoding whose column magnitudes ARE the model (the
    // λ_y Jacobian identity), not conditioning noise.
    let standardizable = request.base_term.behavior.is_none()
        && request.base_term.crosscoder_layout.is_none()
        && request.target.nrows() > 0;
    let sigma = if standardizable {
        let n = request.target.nrows() as f64;
        let mut sigma = Array1::<f64>::zeros(request.target.ncols());
        for (col_idx, col) in request.target.columns().into_iter().enumerate() {
            sigma[col_idx] = (col.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
        }
        let sigma_max = sigma.iter().cloned().fold(0.0_f64, f64::max);
        if sigma_max.is_finite() && sigma_max > 0.0 {
            let floor = sigma_max * f64::EPSILON.sqrt();
            for s in sigma.iter_mut() {
                if !(*s > floor) {
                    *s = 1.0;
                }
            }
            for mut row in request.target.rows_mut() {
                row /= &sigma;
            }
            // The standardization is a CHANGE OF COORDINATES on the output
            // space, so it must map EVERY fit input into the internal frame —
            // the target AND the seed state. The seed was constructed by the
            // caller in raw units; leaving its decoder raw would hand the fit
            // a warm start mis-scaled by up to the per-column RMS ratio
            // (x̂_int must satisfy σ ⊙ x̂_int ≈ x_raw ⇒ B_int[:,c] =
            // B_raw[:,c]/σ_c). Latent coordinates and gate logits are
            // unit-free and untouched; a cold all-zero decoder is a no-op.
            for atom in &mut request.base_term.atoms {
                for (col_idx, s) in sigma.iter().enumerate() {
                    for coeff in atom.decoder_coefficients.column_mut(col_idx).iter_mut() {
                        *coeff /= *s;
                    }
                }
            }
            Some(sigma)
        } else {
            None
        }
    } else {
        None
    };
    let mut report = run_sae_manifold_fit_on_target(request)?;
    report
        .term
        .set_tier0_mean(mu.clone())
        .map_err(SaeFitError::Fit)?;
    if let Some(sigma) = sigma.as_ref() {
        report
            .term
            .set_tier0_scale(sigma.clone())
            .map_err(SaeFitError::Fit)?;
    }
    // Lift the reported reconstructions back to raw-target space (the fit produced
    // them against `(Z − μ)/σ`); assignment masses carry no mean/scale and are
    // untouched.
    lift_tier0_rows(&mut report.fitted, &mu, sigma.as_ref());
    Ok(report)
}

/// Lift an `N×p` reconstruction produced against the standardized de-meaned
/// target back to raw-target space: `x̂ ← μ + σ ⊙ x̂`. Mirrors
/// [`SaeManifoldTerm::add_tier0_mean_inplace`] for the report's standalone
/// reconstruction arrays.
fn lift_tier0_rows(recon: &mut Array2<f64>, mu: &Array1<f64>, sigma: Option<&Array1<f64>>) {
    for mut row in recon.rows_mut() {
        if let Some(sigma) = sigma {
            row *= sigma;
        }
        row += mu;
    }
}

/// The SAE-manifold fit body, run against a target whose Tier-0 shared mean has
/// already been peeled by [`run_sae_manifold_fit`] (or that the caller centered
/// and whose mean the seed term already owns). Every reconstruction/EV inside is
/// therefore in the de-meaned frame; the wrapper owns the μ add-back.
fn run_sae_manifold_fit_on_target(request: SaeFitRequest) -> Result<SaeFitReport, SaeFitError> {
    let SaeFitRequest {
        base_term,
        target: z,
        registry,
        initial_rho: init_rho,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        alpha,
        isometry_pin_active,
        metric_provenance: metric_provenance_initial,
        promote_from_residual,
        run_structure_search,
        run_outer_rho_search,
        structured_residual_passes,
        cancel,
    } = request;
    let (n_obs, p_out) = z.dim();
    let mut metric_provenance: &'static str = metric_provenance_initial;

    // The seed ρ vector the outer engine optimizes; its length is the objective's
    // declared `n_params`.
    let init_rho = init_rho.for_assignment(base_term.assignment.mode);
    let init_rho_flat = init_rho.to_flat();
    let n_params = init_rho_flat.len();

    // #2138 — the whole entry runs on the binding's GIL-released worker thread, so
    // interruptibility is the shared `cancel` flag rather than a per-fit thread.
    // Each objective polls it and bails its next outer eval when the caller sets
    // it on interrupt. Absent ⇒ a fresh, never-set flag (no cancellation).
    let cancel_flag = cancel.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

    // Route every problem size through the full-batch objective on the owned
    // `target`: the inner Arrow-Schur fit materializes the `(N × M_total)`
    // basis, `(N × M_total × d)` jacobian, and `(N × K)` logit buffers in full,
    // so the outer-cascade entry point owns the full target verbatim.
    let mut objective = SaeManifoldOuterObjective::new(
        base_term,
        z.clone(),
        Some(registry.clone()),
        init_rho,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
    );
    scope_outer_checkpoint_to_stage(&mut objective, SaeFitStage::Primary);
    // #1026 — "normal SAE" entry: a single seed (the PCA decoder-projection
    // seed already installed on the term) with NO ρ-multistart. seed_budget=1 +
    // max_seeds=1 collapses the cascade to the single initial ρ.
    objective.set_cancel_flag(Arc::clone(&cancel_flag));
    let mut objective = if run_outer_rho_search {
        // #2241 — do not tune convergence to a workload's observed criterion
        // creep and do not return merely because an iteration budget expired.
        // SAE's Fellner–Schall lane carries a typed recurrent-incumbent
        // certificate: two consecutive inner solves restoring the same banked
        // model terminate the fixed-point walk directly. Gradient-based plans
        // retain the shared solver's stationarity and cost-stall tests.
        // SPEC wall-survival resume: if a checkpoint for this exact data
        // fingerprint exists (a prior job died at its wall mid-search), install
        // the banked incumbent as the warm start and open the ρ search at the
        // banked coordinate. The resumed run must still CONVERGE on its own —
        // a checkpoint never mints a fit, it only saves the work.
        let search_init_rho = match objective.try_resume_from_checkpoint(n_params) {
            Some(banked) => ndarray::Array1::from(banked),
            None => init_rho_flat.clone(),
        };
        let problem = OuterProblem::new(n_params)
            .with_initial_rho(search_init_rho)
            .with_seed_config(SeedConfig {
                max_seeds: 1,
                seed_budget: 1,
                ..Default::default()
            });
        let run_result = problem.run(&mut objective, "SAE manifold");
        certify_outer_stage(objective, SaeFitStage::Primary, run_result)?
    } else {
        objective.fit_at_fixed_rho(init_rho_flat.view())?;
        objective
    };
    // Posterior shape uncertainty: per-atom φ-scaled decoder covariance and
    // ambient bands, read off the converged joint-Hessian Schur factor at the
    // settled ρ. Computed before `into_fitted` consumes the objective; reflects
    // the fitted (smooth) decoder shape, independent of any top-k assignment
    // gate applied below.
    let mut shape_uncertainty = objective.decoder_shape_uncertainty()?;
    // A converged fit is being minted: the wall-survival checkpoint has served
    // its purpose (it must not warm-start a FUTURE fresh fit — that is
    // `persistent_warm_start`'s job, with its own TTL/eviction discipline).
    objective.remove_checkpoint();
    let fitted_result = objective.into_fitted().map_err(SaeFitError::Fit)?;
    let mut finalization_invalidated_shape_uncertainty =
        fitted_result.invalidates_pre_final_shape_uncertainty();
    // #2235 — the outer termination verdict + ledger, surfaced on the payload.
    // `mut`: each structured-residual pass below re-runs the outer search, and
    // the payload must report the termination of the fit actually returned
    // (the final pass), not pass 0's.
    let mut outer_termination = fitted_result.termination;
    let mut term = fitted_result.term;
    let mut rho = fitted_result.rho;
    let mut loss = fitted_result.loss;
    let mut penalized_quasi_laplace_criterion = fitted_result.penalized_quasi_laplace_criterion;

    // #2021 (EXPERIMENT) — structured-residual OUTER ALTERNATION.
    // Pass 0 above is the iid fit (unchanged, bit-for-bit). When the caller's
    // Run the canonical structured pass budget when no explicit metric was
    // installed at pass 0 (a WP-D `OutputFisher` gauge lives in the SAME slot
    // and must not be clobbered), run N extra passes: fit the whitened
    // residual-covariance model on the current fitted residuals, materialize the
    // Σ-DAMPED per-row metric, install it — `loss_scaled` and
    // `assemble_arrow_schur` auto-route on `metric.whitens_likelihood()` (the
    // #974 seam, so no construction.rs change is needed) — and refit
    // warm-started from the settled ρ. The returned provenance / shape bands /
    // loss are refreshed from the final pass. A `None` model (no factor
    // subspace, or a degenerate residual fit) stops the alternation early,
    // degrading to the pass-0 iid fit.
    //
    // Covariance-domain damping (residual-fix's `row_metric_damped`):
    // Σ_t = (1−γ)·Σ_prev + γ·Σ̂_t, with Σ_prev = the previous pass's fitted model
    // (or, on the first structured pass, the MEASURED iid anchor φ̂·I —
    // `isotropic_dispersion`, #2243 cap #2: a unit-I anchor assumed unit noise,
    // so near-noiseless factors were whitened ~1/φ̂ too coarsely and the
    // unit-dispersion penalized quasi-Laplace criterion over-penalized them; anchoring at the
    // measured scale prices the smoothing penalty against the real
    // dispersion). A small, increasing γ schedule
    // γ_p = (p+1)/(N+1) ∈ (0,1) trusts the new estimate more each pass while
    // damping the early jump off the iid fit (γ is never 0 or 1, so every pass
    // builds a genuine WhitenedStructured blend).
    let structured_passes =
        structured_residual_passes.unwrap_or(STRUCTURED_RESIDUAL_PASSES_DEFAULT);
    let mut structured_residual_diagnostics: Vec<StructuredResidualPassDiagnostic> = Vec::new();
    if structured_passes > 0 && metric_provenance == "Euclidean" {
        let mut prev_model: Option<StructuredResidualModel> = None;
        // #2021 Λ nursery→promotion (evidence-gated). Accumulate residual-factor
        // directions that PERSIST across passes (producer
        // `StructuredResidualModel::promotion_candidates`: energy above the
        // idiosyncratic-noise floor AND |cos|-alignment with the previous pass's
        // Λ) and, once a lineage matures, promote it to a born atom so the NEXT
        // pass refits with the discovered structure. A lineage that skips a pass
        // loses its dwell; at most one birth per pass, and only when a later pass
        // remains to refit the born atom, so K grows ≤ the pass budget and no
        // born atom is left unrefit inside the alternation.
        //
        // #2239 evidence-driven pass extension: a live nursery lineage is itself
        // the certificate that residual structure persists. When the planned
        // budget would expire with lineages still maturing (or a matured lineage
        // still owed its post-birth refit), the alternation grants itself one
        // more pass, hard-capped at `STRUCTURED_RESIDUAL_PASSES_MAX`. Compute
        // grows only while the certificate keeps firing; on structureless data
        // the nursery stays empty and the planned budget is exact.
        //
        // PROMOTION_ENERGY_FLOOR_MULT — DERIVED (identity). The energy gate is
        // "above the idiosyncratic-noise floor"; the floor is already the
        // data-estimated detection threshold, so the canonical multiplier is 1.0.
        const PROMOTION_ENERGY_FLOOR_MULT: f64 = 1.0;
        // PROMOTION_NURSERY_MIN_PASSES — DERIVED (minimal persistence). Two is the
        // smallest dwell at which a direction has been re-observed across a refit,
        // i.e. the minimal count that distinguishes a repeated structural signal
        // from a one-pass artifact.
        const PROMOTION_NURSERY_MIN_PASSES: usize = 2;
        // The #2071 per-pass alignment threshold `align_min(r)` is the
        // Beta-quantile of the random-alignment null keyed to the residual factor
        // rank `r`. It is derived here and used identically by the producer-side
        // candidate gate and the nursery lineage-dedup below.
        //
        // `promote_from_residual` is the typed caller flag (default TRUE, #2239:
        // magic-by-default — the evidence certificate above, not the flag, is the
        // real gate). `false` pins the historical whitening-without-growth path.
        let mut nursery: Vec<(Array1<f64>, usize)> = Vec::new();
        let mut total_passes = structured_passes;
        let mut pass = 0usize;
        while pass < total_passes {
            let Some(model) = sae_structured_residual_model(&term, z.view())? else {
                break;
            };
            let gamma = (pass as f64 + 1.0) / (total_passes as f64 + 1.0);
            let metric = model.row_metric_damped(n_obs, gamma, prev_model.as_ref())?;
            let installed_label = metric_provenance_label(metric.provenance());
            let factor_energy = model.factor().iter().map(|v| v * v).sum::<f64>();
            let diagonal_mean = model.diagonal().iter().copied().sum::<f64>() / p_out as f64;
            let dispersion_before = shape_uncertainty.dispersion;
            let log_lambda_smooth_before = rho.log_lambda_smooth.clone();
            term.set_row_metric(metric)?;
            // Clone the pre-built registry (cheap) and warm-start ρ from the
            // settled fit — identical to the binding rebuilding it from the same
            // `latent_payload` + descriptor JSON each pass.
            let warm_flat = rho.to_flat();
            let mut objective = SaeManifoldOuterObjective::new(
                term,
                z.clone(),
                Some(registry.clone()),
                rho,
                max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            );
            let stage = SaeFitStage::StructuredResidual {
                pass: pass + 1,
                total_passes,
            };
            scope_outer_checkpoint_to_stage(&mut objective, stage);
            // #2021 — a promotion (below) grows K, enlarging ρ; size the outer
            // problem from the CURRENT warm vector, not the pass-0 `n_params`
            // (identical to `n_params` when no birth has occurred).
            // Resume only the checkpoint for this exact structured phase. Earlier
            // phases have already been deterministically rebuilt on this run;
            // their distinct files cannot leak a differently-whitened state here.
            let search_init_rho = match objective.try_resume_from_checkpoint(warm_flat.len()) {
                Some(banked) => ndarray::Array1::from(banked),
                None => warm_flat,
            };
            // #2138 — same shared cancel flag; each pass's objective polls it.
            objective.set_cancel_flag(Arc::clone(&cancel_flag));
            // Honor the same typed outer-search contract as pass 0. Structured
            // residual alternation still runs when ρ search is disabled; only
            // the hyperparameter walk is fixed at the warm ρ. The former
            // unconditional `OuterProblem` silently re-enabled an outer search
            // after a fixed-ρ primary fit.
            let mut objective = if run_outer_rho_search {
                let problem = OuterProblem::new(search_init_rho.len())
                    .with_initial_rho(search_init_rho)
                    .with_seed_config(SeedConfig {
                        max_seeds: 1,
                        seed_budget: 1,
                        ..Default::default()
                    });
                // SPEC 20 — possession of the objective below is itself the
                // convergence certificate: `certify_outer_stage` returns it only for
                // `OuterResult.converged`, and drops it without removing its
                // phase-scoped checkpoint for every typed failure.
                let run_result = problem.run(&mut objective, "SAE manifold (structured)");
                certify_outer_stage(objective, stage, run_result)?
            } else {
                objective.fit_at_fixed_rho(search_init_rho.view())?;
                objective
            };
            // Refresh shape bands + fitted state from the FINAL pass objective
            // (decoder_shape_uncertainty must be read before `into_fitted`).
            shape_uncertainty = objective.decoder_shape_uncertainty()?;
            objective.remove_checkpoint();
            let fitted_result = objective.into_fitted().map_err(SaeFitError::Fit)?;
            finalization_invalidated_shape_uncertainty =
                fitted_result.invalidates_pre_final_shape_uncertainty();
            // #2235 — the returned fit is this pass's; report its termination.
            outer_termination = fitted_result.termination;
            term = fitted_result.term;
            rho = fitted_result.rho;
            loss = fitted_result.loss;
            penalized_quasi_laplace_criterion = fitted_result.penalized_quasi_laplace_criterion;
            structured_residual_diagnostics.push(StructuredResidualPassDiagnostic {
                pass: pass + 1,
                gamma,
                factor_rank: model.factor_rank(),
                log_evidence: model.log_evidence(),
                factor_energy,
                diagonal_mean,
                dispersion_before,
                dispersion_after: shape_uncertainty.dispersion,
                log_lambda_smooth_before,
                log_lambda_smooth_after: rho.log_lambda_smooth.clone(),
            });
            // Report the geometry actually used by the returned fit.
            metric_provenance = installed_label;
            // #2021 promotion: fold this pass's persisted factor directions into
            // the nursery, then promote (birth) at most one matured lineage so the
            // NEXT pass refits with it. Runs only when the opt-in lever is set
            // (default off) AND from pass 1 on (needs a `prev`). Gating via a
            // `None` prev keeps the block un-indented and inert when off.
            let prev_for_promotion = if promote_from_residual {
                prev_model.as_ref()
            } else {
                None
            };
            if let Some(prev) = prev_for_promotion {
                // Per-pass derived alignment threshold from the current residual
                // factor rank (#2071); used identically by the producer-side
                // candidate gate and the nursery lineage-dedup below.
                let align_min = promotion_alignment_threshold(model.factor_rank());
                let cands = model.promotion_candidates(
                    Some(prev),
                    align_min,
                    PROMOTION_ENERGY_FLOOR_MULT,
                )?;
                let mut seen = vec![false; nursery.len()];
                for cand in &cands {
                    let hit = nursery
                        .iter()
                        .position(|(d, _)| cand.direction.dot(d).abs() >= align_min);
                    match hit {
                        Some(i) => {
                            nursery[i].0 = cand.direction.clone();
                            nursery[i].1 += 1;
                            seen[i] = true;
                        }
                        None => {
                            nursery.push((cand.direction.clone(), 1));
                            seen.push(true);
                        }
                    }
                }
                // A lineage that did not recur this pass loses its dwell.
                let mut keep = seen.into_iter();
                nursery.retain(|_| keep.next().unwrap_or(false));
                // #2239 evidence-driven extension: if the budget is about to
                // expire while lineages are still live (maturing, or matured and
                // owed the post-birth refit), grant one more pass, capped at
                // `STRUCTURED_RESIDUAL_PASSES_MAX`. An empty nursery never
                // extends, so structureless data keeps the planned budget exact.
                if !nursery.is_empty()
                    && pass + 1 == total_passes
                    && total_passes < STRUCTURED_RESIDUAL_PASSES_MAX
                {
                    total_passes += 1;
                }
                // Promote at most one matured lineage, and only if a later pass
                // remains to refit the born atom. Collect the direction BEFORE
                // mutating `term` to avoid overlapping borrows.
                let matured = if pass + 1 < total_passes {
                    nursery
                        .iter()
                        .find(|(_, count)| *count >= PROMOTION_NURSERY_MIN_PASSES)
                        .map(|(dir, _)| dir.clone())
                } else {
                    None
                };
                if let Some(dir) = matured {
                    // Born-atom decoder: the unit direction on atom-0's constant
                    // (row-0) basis row, shape (m, p) per `born_atom`'s contract.
                    let m = term.atoms[0].basis_size();
                    let mut decoder = Array2::<f64>::zeros((m, p_out));
                    for out in 0..p_out {
                        decoder[[0, out]] = dir[out];
                    }
                    let (grown_term, grown_rho) = structure_harvest::apply_structure_move(
                        &term,
                        &rho,
                        &StructureMove::Birth { candidate: 0 },
                        std::slice::from_ref(&decoder),
                    )?;
                    term = grown_term;
                    rho = grown_rho;
                    // Drop the promoted lineage so it is not re-promoted; the next
                    // pass rebuilds the objective from the grown `term`/`rho` and
                    // `warm_flat.len()` picks up the enlarged ρ automatically.
                    nursery.retain(|(d, _)| d.dot(&dir).abs() < align_min);
                }
            }
            // Carry this pass's model forward as the next pass's damping anchor.
            prev_model = Some(model);
            pass += 1;
        }
    }
    {
        let assignments = term.assignment.assignments();
        let fitted = term.try_fitted_target_aware(z.view(), Some(&rho))?;
        term.record_fit_data_collapse_if_needed(
            z.view(),
            fitted.view(),
            assignments.view(),
            max_iter,
        )?;
    }

    // #977 / #997 — evidence-guarded structure search around the production fit:
    // the genuine dictionary learner. Harvest deaths (diverged ARD ∪ terminal
    // collapse), fusions (co-activation), fission audits (absorption asymmetry),
    // and BIRTHS (whitened residual-factor subspace), then run the e-gated move
    // engine over a held-out estimation/evaluation row split. So K is DISCOVERED
    // from the data rather than pinned at the user's input K; the SearchLedger
    // (+ the joint fit's collapse events) is serialized onto the payload as the
    // honesty surface — never a silent restructure.
    let mut structure_ledger = StructureLedger::new();
    // #1230 — whether structure search actually changed the model (a landed
    // birth/fission/fusion or a demoted death). When it did, the pre-search
    // joint-Hessian shape bands assembled above are stale and must be recomputed
    // from the final post-search per-atom inner fits (below).
    let mut structure_changed = false;
    let structure_search_json = 'structure: {
        if !run_structure_search {
            break 'structure None;
        }
        // #1026 — structure search is a post-fit DISCOVERY pass: each round refits
        // the full dictionary over ALL N rows. Scale rounds down with K and SKIP
        // entirely past a ceiling so a fixed-K performance run returns the fitted
        // dictionary without paying the search.
        let structure_max_rounds = {
            let k_now = term.k_atoms().max(1);
            if k_now <= 2 {
                3
            } else if k_now <= 8 {
                2
            } else if k_now <= 64 {
                1
            } else {
                0
            }
        };
        if structure_max_rounds == 0 {
            break 'structure None;
        }
        // Per-round harvest breadth derived from the fitted K (magic-by-default):
        // propose at most a handful of each move kind, scaled gently with the
        // dictionary size, with a small fixed floor so even a K=1 fit can grow.
        let k_now = term.k_atoms().max(1);
        let births_per_round = (k_now + 1).min(4);
        let fissions_per_round = k_now.min(4);
        let harvest_params = structure_harvest::HarvestParams {
            max_fusions: 4,
            max_fissions: fissions_per_round,
            max_births: births_per_round,
        };
        // The per-candidate scoring refit is capped well below the outer fit's
        // `max_iter`: a structural move yields a WARM child, so only the touched
        // atom must re-equilibrate before the held-out evidence gate can rank the
        // candidate. Each round's accepted winner is re-refit at the full
        // `max_iter` before adoption (#1026, verified move-equivalent).
        const STRUCTURE_SCORING_INNER_MAX_ITER: usize = 8;
        let refit_params = structure_harvest::ProductionRefitParams {
            inner_max_iter: max_iter,
            scoring_inner_max_iter: STRUCTURE_SCORING_INNER_MAX_ITER.min(max_iter),
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        };
        // Moves that may LAND this round (accepted births/fissions/fusions +
        // demoted deaths); remaining proposals are recorded `Deferred` and
        // replayed next round. Magic-by-default — a function of the fitted K.
        let max_moves = k_now + births_per_round + fissions_per_round;
        let budget = MoveBudget {
            max_moves,
            alpha: 0.05,
        };
        let config = structure_harvest::RoundDriverConfig {
            n_shards: 4,
            budget,
            max_rounds: structure_max_rounds,
            harvest_params,
            // Curl/flatten structure moves stay off in the production path until
            // the killer-demo gate graduates them (INTEGRATION_PLAN §8).
            curl: None,
        };
        match structure_harvest::run_production_structure_search(
            term,
            rho,
            z.view(),
            config,
            refit_params,
            &mut structure_ledger,
        ) {
            Ok(result) => {
                structure_changed = result.structure_changed();
                term = result.term;
                rho = result.rho;
                structure_harvest::rounds_to_json(&result.rounds).ok()
            }
            Err(e) => {
                // Structure search is a post-fit audit pass; a failure must not
                // silently corrupt the fit — surface it loudly.
                return Err(SaeFitError::Fit(format!(
                    "structure search around SAE fit failed: {e}"
                )));
            }
        }
    };

    // Clear any per-row estimation mask the structure-search refit left on the
    // adopted term so the returned `fitted` / dispersion / diagnostics are
    // computed over ALL rows (the mask is an internal split device, not a
    // property of the returned fit).
    term.clear_row_loss_weights();

    // #977 — VARIABLE-K boundary. `term.k_atoms()` is the source of truth from
    // this point on; the input (seed) K is stale the moment a birth lands.
    let k_atoms = term.k_atoms();

    term.set_certificate_dispersion(shape_uncertainty.dispersion)?;

    // #1097 / #1103 — harvest each atom's fixed inner-decoder-smooth snapshot at
    // the settled state, so the diagnostics report can produce per-atom
    // Riesz-debiased functionals and the split-LRT smooth-structure e-value.
    term.set_atom_inner_fits(z.view(), shape_uncertainty.dispersion)?;

    // #977 / #1230 — recompute the joint-Hessian shape bands when structure
    // search changed the model OR a finalization fallback fired: the pre-search
    // bands are stale. Rebuild the JOINT inverse-Hessian bands from the FINAL
    // term + ρ for EVERY atom (seed and born); on a non-PD post-search Hessian
    // fall back to the per-atom Laplace completion below.
    if structure_changed || finalization_invalidated_shape_uncertainty {
        let joint_registry = registry.clone();
        // Snapshot the fitted term: the optional joint recompute mutates `term`
        // while re-solving, so a recoverable refusal must not leave the actual
        // fitted model perturbed. Restore it before degrading to per-atom bands.
        let saved_term_for_shape_recompute = term.clone();
        match term.recompute_joint_shape_uncertainty(
            z.view(),
            &rho,
            Some(&joint_registry),
            max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        ) {
            Ok(joint) => {
                shape_uncertainty = joint;
                // The certificate dispersion was seeded from the (now stale)
                // pre-search φ̂; refresh it to the joint recompute's final value.
                term.set_certificate_dispersion(shape_uncertainty.dispersion)?;
            }
            Err(e) => {
                term = saved_term_for_shape_recompute;
                // The joint factor could not be reformed at the final state. Fall
                // back to the per-atom Laplace completion: invalidate the stale
                // joint bands so `complete_born_atom_shape_bands` refills each from
                // its OWN penalized inner Hessian.
                log::warn!(
                    "[shape-uncertainty] joint band recompute after structure/finalization \
                     change failed ({e}); falling back to per-atom Laplace bands"
                );
                shape_uncertainty.invalidate_bands_for_recompute();
            }
        }
    }
    // Backstop: fill any atom the joint factor left unidentified (all-NaN) — a
    // structure-search-born atom the pre-search Schur never covered, or a
    // degenerate joint block — from its own inner Hessian. A no-op after a
    // successful joint recompute.
    term.complete_born_atom_shape_bands(&mut shape_uncertainty)?;

    // Additive post-fit diagnostics (#980): the two-score per-atom lens and the
    // residual-gauge certificate. Per-atom ARD variances (∝ exp(−log_precision))
    // are threaded in when native ARD was enabled, else `None` per atom.
    let ard_variances: Vec<Option<Array1<f64>>> = rho
        .log_ard
        .iter()
        .map(|log_prec| {
            if log_prec.is_empty() {
                None
            } else {
                Some(log_prec.mapv(|lp| (-lp).exp()))
            }
        })
        .collect();
    let assignments = term.assignment.assignments();
    let fitted = term.try_fitted_target_aware(z.view(), Some(&rho))?;
    term.record_fit_data_collapse_if_needed(z.view(), fitted.view(), assignments.view(), max_iter)?;
    let trust_diagnostics = term.trust_diagnostics_report(assignments.view())?;
    // Assignment-support diagnostics read the exact assignments used by the
    // reconstruction and objective.
    let fit_diagnostics = term.fit_diagnostics_report(
        Some(&ard_variances),
        isometry_pin_active,
        Some(shape_uncertainty.dispersion),
        fitted.view(),
        Some(assignments.view()),
    )?;

    let active_mask: Vec<bool> = (0..k_atoms)
        .map(|atom_idx| assignments.column(atom_idx).sum() > 1.0e-8)
        .collect();
    let mut means = vec![0.0_f64; p_out];
    for row in 0..n_obs {
        for out_col in 0..p_out {
            means[out_col] += z[[row, out_col]];
        }
    }
    if n_obs > 0 {
        let inv_n = 1.0 / n_obs as f64;
        for mean in means.iter_mut() {
            *mean *= inv_n;
        }
    }
    let mut rss = 0.0_f64;
    let mut tss = 0.0_f64;
    for row in 0..n_obs {
        for out_col in 0..p_out {
            let residual = z[[row, out_col]] - fitted[[row, out_col]];
            let centered = z[[row, out_col]] - means[out_col];
            rss += residual * residual;
            tss += centered * centered;
        }
    }
    let reconstruction_r2 = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

    let reported_log_alpha = match term.assignment.mode {
        AssignmentMode::OrderedBetaBernoulli { alpha, .. } => alpha.ln(),
        _ => alpha.ln(),
    };

    // Anytime-valid structure certificate (#1058 / #984): the e-BH certificate
    // over the ledger's per-claim e-processes at the search FDR level α = 0.05.
    let structure_certificate = structure_ledger.certify(0.05);
    let structure_certificate_json =
        serde_json::to_string(&structure_certificate).map_err(|e| e.to_string())?;

    Ok(SaeFitReport {
        term,
        rho,
        loss,
        penalized_quasi_laplace_criterion,
        assignments,
        fitted,
        active_mask,
        reconstruction_r2,
        outer_termination,
        shape_uncertainty,
        metric_provenance,
        structured_residual_diagnostics,
        trust_diagnostics,
        fit_diagnostics,
        structure_search_json,
        structure_certificate_json,
        reported_log_alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::promotion_alignment_threshold;

    #[test]
    fn promotion_alignment_threshold_is_core_owned_and_rank_aware() {
        assert_eq!(promotion_alignment_threshold(0), 1.0);
        assert_eq!(promotion_alignment_threshold(1), 1.0);

        let rank_two = promotion_alignment_threshold(2);
        let rank_four = promotion_alignment_threshold(4);
        assert!(rank_two.is_finite() && (0.0..=1.0).contains(&rank_two));
        assert!(rank_four.is_finite() && (0.0..=1.0).contains(&rank_four));
        assert!(rank_four < rank_two);
    }
}
