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
use gam_problem::topology_certificates::CertificateLedger;
use gam_problem::{EstimationError, MetricProvenance};
use gam_solve::inference::residual_factor::{ResidualFactorInput, StructuredResidualModel};
use gam_solve::rho_optimizer::{OuterProblem, OuterResult, audit_stationary_point};
use gam_solve::seeding::SeedConfig;
use gam_solve::structure_search::{MoveBudget, StructureMove};
use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
use gam_terms::inference::structure_evidence::StructureLedger;

use crate::structure_harvest;
use crate::tiered::Tier0Mean;

use super::{
    AmortizedEncoderConsistency, AssignmentMode, CoordinateFidelityCertificate,
    SaeManifoldFitDiagnostics, SaeManifoldLoss, SaeManifoldOuterObjective, SaeManifoldRho,
    SaeManifoldTerm, SaeOuterTermination, SaeShapeUncertainty,
    SaeTrustDiagnostics, TopologyPersistenceCertificate, VanishedAtoms,
};

/// Hard cap on evidence-certified #2021 whitened-residual refit passes.
pub const STRUCTURED_RESIDUAL_PASSES_MAX: usize = 4;

fn validate_structured_residual_passes(passes: usize) -> Result<(), SaeFitError> {
    if passes > STRUCTURED_RESIDUAL_PASSES_MAX {
        return Err(SaeFitError::InvalidRequest(format!(
            "structured_residual_passes={passes} exceeds the hard maximum {STRUCTURED_RESIDUAL_PASSES_MAX}"
        )));
    }
    Ok(())
}

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
    /// Consistency of the fitted native encoder with the converged latent solve.
    pub amortized_encoder_consistency: AmortizedEncoderConsistency,
    /// Unified conservative certificate ledger assembled from this fit's reports.
    pub certificate_ledger: CertificateLedger,
    /// Serialized per-round structure-search ledger (#997) as a JSON string;
    /// `None` when the search did not run (skipped by K ceiling or
    /// `run_structure_search == false`).
    pub structure_search_json: Option<String>,
    /// The anytime-valid structure certificate (#1058/#984), serialized JSON;
    /// absent when no genuine structure search ran.
    pub structure_certificate_json: Option<String>,
    /// The reported `log_alpha` (ordered Beta--Bernoulli concentration or the caller's α fallback).
    pub reported_log_alpha: f64,
}

/// Exact inner-KKT measurements at one caller-installed external state. No
/// optimizer is invoked to form these values; they are read directly from the
/// analytic joint system assembled at the supplied `(term, rho)`.
#[derive(Clone, Debug, PartialEq)]
pub struct SaeInstalledInnerKktAudit {
    pub raw_gradient_norm: f64,
    pub quotient_gradient_norm: f64,
    pub stationarity_bound: f64,
}

impl SaeInstalledInnerKktAudit {
    pub fn certifies(&self) -> bool {
        SaeManifoldTerm::quasi_laplace_kkt_stationary(
            self.raw_gradient_norm,
            self.quotient_gradient_norm,
            self.stationarity_bound,
        )
    }
}

/// Typed diagnostic for an externally supplied state that was evaluated but
/// refused as a fit. It intentionally carries no term, fitted payload, shape
/// uncertainty, certificate ledger, or structure evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct SaeExternalEvaluationReport {
    pub inner: SaeInstalledInnerKktAudit,
    pub outer_raw_gradient_norm: Option<f64>,
    pub outer_projected_gradient_norm: Option<f64>,
    pub outer_stationarity_bound: Option<f64>,
    pub optimization_iterations: usize,
    pub reason: String,
}

/// External state certification either mints the ordinary converged-fit report
/// or returns a non-fit diagnostic. The rejected variant cannot be confused
/// with [`SaeFitOutcome`] and cannot reach inference/structure marshalling.
pub enum SaeExternalCertificationOutcome {
    Certified(SaeFitReport),
    NonStationary(SaeExternalEvaluationReport),
}

fn installed_inner_kkt_audit(
    term: &mut SaeManifoldTerm,
    target: ndarray::ArrayView2<'_, f64>,
    rho: &SaeManifoldRho,
    registry: &AnalyticPenaltyRegistry,
) -> Result<SaeInstalledInnerKktAudit, SaeFitError> {
    let system = term
        .assemble_arrow_schur(target, rho, Some(registry))
        .map_err(SaeFitError::Fit)?;
    let raw_gradient_norm_sq = SaeManifoldTerm::system_grad_norm_sq(&system);
    let raw_gradient_norm = raw_gradient_norm_sq.sqrt();
    let lambda_smooth = rho.lambda_smooth_vec().map_err(SaeFitError::Fit)?;
    let quotient_gradient_norm = term.quotient_gradient_norm_from_system(
        &system,
        raw_gradient_norm_sq,
        &lambda_smooth,
    );
    Ok(SaeInstalledInnerKktAudit {
        raw_gradient_norm,
        quotient_gradient_norm,
        stationarity_bound: super::SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale(),
    })
}

fn external_nonstationary_report(
    inner: SaeInstalledInnerKktAudit,
    outer: Option<&OuterResult>,
    reason: String,
) -> SaeExternalCertificationOutcome {
    let stationarity = outer
        .and_then(|result| result.criterion_certificate.as_ref())
        .map(|certificate| &certificate.stationarity);
    SaeExternalCertificationOutcome::NonStationary(SaeExternalEvaluationReport {
        inner,
        outer_raw_gradient_norm: stationarity.map(|certificate| certificate.raw_norm()),
        outer_projected_gradient_norm: stationarity
            .map(|certificate| certificate.projected_norm()),
        outer_stationarity_bound: stationarity.map(|certificate| certificate.bound()),
        optimization_iterations: outer.map_or(0, |result| result.iterations),
        reason,
    })
}

/// Exact intercept-only result when the committed fixed-`K` terminal state has
/// zero realised decoder rank for every atom.  This is not a non-converged
/// manifold fit and therefore carries no manifold rho, shape bands, or outer
/// termination fiction.  Tier-0 is closed form; the vanished set records the
/// structural boundary that selected it.
pub struct SaeNullFitReport {
    pub tier0: Tier0Mean,
    pub fitted: Array2<f64>,
    pub residual_sum_squares: f64,
    pub reconstruction_r2: f64,
    pub metric_provenance: &'static str,
    pub vanished_atoms: VanishedAtoms,
}

/// A native SAE fit either has at least one certified manifold atom or is the
/// exact Tier-0 null.  Keeping these variants distinct prevents a `K=0` payload
/// from masquerading as a converged `SaeManifoldTerm`, whose constructors and
/// inference reports require at least one atom.
pub enum SaeFitOutcome {
    Manifold(SaeFitReport),
    Null(SaeNullFitReport),
}

impl SaeFitOutcome {
    pub fn manifold_or_error(self) -> Result<SaeFitReport, String> {
        match self {
            Self::Manifold(report) => Ok(report),
            Self::Null(report) => Err(format!(
                "fit selected the exact Tier-0 null after {} atom(s) vanished",
                report.vanished_atoms.len()
            )),
        }
    }
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
    InvalidRequest(String),
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
            Self::InvalidRequest(message) | Self::Fit(message) => f.write_str(message),
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
            Self::InvalidRequest(_) | Self::Fit(_) | Self::OuterDidNotConverge { .. } => None,
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

enum SaeStageFit {
    Certified(SaeManifoldOuterObjective),
    Null(SaeNullFitReport),
}

fn exact_null_report(
    state: super::SaeVanishedStageState,
    target: &Array2<f64>,
    metric_provenance: &'static str,
) -> SaeNullFitReport {
    let p = target.ncols();
    let mean = state
        .term
        .tier0_mean()
        .cloned()
        .unwrap_or_else(|| Array1::<f64>::zeros(p));
    let fitted = Array2::from_shape_fn(target.dim(), |(_, col)| mean[col]);
    let target_mean = target
        .mean_axis(ndarray::Axis(0))
        .unwrap_or_else(|| Array1::<f64>::zeros(p));
    let mut residual_sum_squares = 0.0_f64;
    let mut total_sum_squares = 0.0_f64;
    for row in 0..target.nrows() {
        for col in 0..p {
            let residual = target[[row, col]] - fitted[[row, col]];
            let centered = target[[row, col]] - target_mean[col];
            residual_sum_squares += residual * residual;
            total_sum_squares += centered * centered;
        }
    }
    let reconstruction_r2 = if total_sum_squares > 0.0 {
        1.0 - residual_sum_squares / total_sum_squares
    } else {
        0.0
    };
    SaeNullFitReport {
        tier0: Tier0Mean { mean },
        fitted,
        residual_sum_squares,
        reconstruction_r2,
        metric_provenance,
        vanished_atoms: state.atoms,
    }
}

enum SaeBoundaryDisposition {
    Restart {
        term: SaeManifoldTerm,
        rho: SaeManifoldRho,
    },
    Null(SaeNullFitReport),
}

fn vanished_disposition(
    mut state: super::SaeVanishedStageState,
    target: &Array2<f64>,
    metric_provenance: &'static str,
) -> Result<SaeBoundaryDisposition, SaeFitError> {
    if state.atoms.len() == state.term.k_atoms() {
        return Ok(SaeBoundaryDisposition::Null(exact_null_report(
            state,
            target,
            metric_provenance,
        )));
    }
    let remove = state.atoms.as_btree_set();
    structure_harvest::remove_atoms(&mut state.term, &mut state.rho, &remove)
        .map_err(SaeFitError::Fit)?;
    Ok(SaeBoundaryDisposition::Restart {
        term: state.term,
        rho: state.rho,
    })
}

fn fit_outer_stage_to_boundary(
    mut term: SaeManifoldTerm,
    target: &Array2<f64>,
    registry: &AnalyticPenaltyRegistry,
    mut rho: SaeManifoldRho,
    max_iter: usize,
    learning_rate: f64,
    ridge_ext_coord: f64,
    ridge_beta: f64,
    run_outer_rho_search: bool,
    stage: SaeFitStage,
    cancel_flag: &Arc<AtomicBool>,
    metric_provenance: &'static str,
) -> Result<SaeStageFit, SaeFitError> {
    loop {
        let rho_flat = rho.to_flat();
        let mut objective = SaeManifoldOuterObjective::new(
            term,
            target.clone(),
            Some(registry.clone()),
            rho,
            max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        );
        scope_outer_checkpoint_to_stage(&mut objective, stage);
        objective.set_cancel_flag(Arc::clone(cancel_flag));

        let boundary = if run_outer_rho_search {
            let search_init_rho = match objective.try_resume_from_checkpoint(rho_flat.len())? {
                Some(banked) => ndarray::Array1::from(banked),
                None => rho_flat,
            };
            let problem = OuterProblem::new(search_init_rho.len())
                .with_initial_rho(search_init_rho)
                .with_seed_config(SeedConfig {
                    max_seeds: 1,
                    seed_budget: 1,
                    ..Default::default()
                });
            match problem.run(&mut objective, "SAE manifold") {
                Ok(result) if result.converged => {
                    return certify_outer_stage(objective, stage, Ok(result))
                        .map(SaeStageFit::Certified);
                }
                Ok(result) => {
                    let terminal_rho = Array1::from(result.rho.clone());
                    match objective.vanished_stage_state_at(terminal_rho.view()) {
                        Ok(Some(state)) => Some(state),
                        Ok(None) => {
                            return Err(SaeFitError::OuterDidNotConverge {
                                stage,
                                result: Box::new(result),
                            });
                        }
                        Err(error) => return Err(SaeFitError::Fit(error)),
                    }
                }
                Err(source) => {
                    let terminal_rho = objective.current_rho_flat();
                    match objective.vanished_stage_state_at(terminal_rho.view()) {
                        Ok(Some(state)) => Some(state),
                        Ok(None) => {
                            return Err(SaeFitError::OuterRun { stage, source });
                        }
                        Err(error) => return Err(SaeFitError::Fit(error)),
                    }
                }
            }
        } else {
            match objective.fit_at_fixed_rho(rho_flat.view()) {
                Ok(()) => return Ok(SaeStageFit::Certified(objective)),
                Err(original) => match objective.vanished_stage_state_at(rho_flat.view()) {
                    Ok(Some(state)) => Some(state),
                    Ok(None) => return Err(SaeFitError::Fit(original)),
                    Err(error) => return Err(SaeFitError::Fit(error)),
                },
            }
        };

        let state = boundary.expect("each non-returning branch installs a boundary state");
        objective.remove_checkpoint();
        match vanished_disposition(state, target, metric_provenance)? {
            SaeBoundaryDisposition::Restart {
                term: reduced_term,
                rho: reduced_rho,
            } => {
                term = reduced_term;
                rho = reduced_rho;
            }
            SaeBoundaryDisposition::Null(report) => return Ok(SaeStageFit::Null(report)),
        }
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
    /// Explicit number of structured-residual whitening passes. Each pass
    /// installs a new row-metric likelihood and re-runs the full outer search;
    /// zero is the direct seed → single certified fit path.
    pub structured_residual_passes: usize,
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
pub fn run_sae_manifold_fit(mut request: SaeFitRequest) -> Result<SaeFitOutcome, SaeFitError> {
    validate_structured_residual_passes(request.structured_residual_passes)?;
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
    let tier0_residual_sum_squares = request
        .target
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
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
    let mut outcome = run_sae_manifold_fit_on_target(request)?;
    match &mut outcome {
        SaeFitOutcome::Manifold(report) => {
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
            lift_tier0_rows(&mut report.fitted, &mu, sigma.as_ref());
        }
        SaeFitOutcome::Null(report) => {
            report.tier0 = Tier0Mean { mean: mu.clone() };
            lift_tier0_rows(&mut report.fitted, &mu, sigma.as_ref());
            report.residual_sum_squares = tier0_residual_sum_squares;
            report.reconstruction_r2 = 0.0;
        }
    }
    Ok(outcome)
}

#[cfg(test)]
mod structured_pass_request_tests {
    use super::*;

    #[test]
    fn explicit_structured_pass_count_above_hard_cap_is_rejected_2267() {
        assert!(validate_structured_residual_passes(0).is_ok());
        assert!(validate_structured_residual_passes(STRUCTURED_RESIDUAL_PASSES_MAX).is_ok());
        assert!(matches!(
            validate_structured_residual_passes(STRUCTURED_RESIDUAL_PASSES_MAX + 1),
            Err(SaeFitError::InvalidRequest(_))
        ));
    }
}

#[cfg(test)]
mod vanished_stage_tests {
    use super::*;
    use crate::manifold::{AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeManifoldAtom};
    use gam_terms::latent::LatentManifold;
    use ndarray::Array3;

    fn fixed_boundary_term(k: usize, live_first: bool) -> (SaeManifoldTerm, SaeManifoldRho) {
        let n = 8usize;
        let p = 2usize;
        let mut atoms = Vec::with_capacity(k);
        for atom in 0..k {
            let mut decoder = Array2::<f64>::zeros((1, p));
            if atom == 0 && live_first {
                decoder[[0, 0]] = 1.0;
            }
            atoms.push(
                SaeManifoldAtom::new_with_provided_function_gram(
                    format!("atom{atom}"),
                    SaeAtomBasisKind::EuclideanPatch,
                    1,
                    Array2::<f64>::ones((n, 1)),
                    Array3::<f64>::zeros((n, 1, 1)),
                    decoder,
                    Array2::<f64>::eye(1),
                )
                .unwrap(),
            );
        }
        let mut logits = Array2::<f64>::zeros((n, k));
        if k > 1 {
            logits.column_mut(1).fill(-40.0);
        }
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![Array2::<f64>::zeros((n, 1)); k],
            vec![LatentManifold::Euclidean; k],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); k]);
        (term, rho)
    }

    #[test]
    fn committed_k2_boundary_compacts_and_fixed_rho_restart_certifies_k1() {
        let (term, rho) = fixed_boundary_term(2, true);
        let mut target = Array2::<f64>::zeros((8, 2));
        target.column_mut(0).fill(1.0);
        let registry = AnalyticPenaltyRegistry::new();
        let cancel = Arc::new(AtomicBool::new(false));
        let stage = fit_outer_stage_to_boundary(
            term,
            &target,
            &registry,
            rho,
            0,
            1.0,
            1.0e-6,
            1.0e-6,
            false,
            SaeFitStage::Primary,
            &cancel,
            "Euclidean",
        )
        .expect("proper vanished subset must restart on the compacted stratum");
        let SaeStageFit::Certified(objective) = stage else {
            panic!("one live atom must not collapse to the Tier-0 null");
        };
        let fitted = objective
            .into_fitted()
            .expect("reduced fixed-rho state must carry an inner certificate");
        assert_eq!(fitted.term.k_atoms(), 1);
        assert_eq!(fitted.rho.log_lambda_smooth.len(), 1);
        assert_eq!(fitted.rho.log_ard.len(), 1);
        assert!(fitted.penalized_quasi_laplace_criterion.is_finite());
    }

    #[test]
    fn committed_k1_boundary_returns_exact_tier0_null_not_manifold_fit() {
        let (term, rho) = fixed_boundary_term(1, false);
        let target = Array2::<f64>::ones((8, 2));
        let registry = AnalyticPenaltyRegistry::new();
        let cancel = Arc::new(AtomicBool::new(false));
        let stage = fit_outer_stage_to_boundary(
            term,
            &target,
            &registry,
            rho,
            0,
            1.0,
            1.0e-6,
            1.0e-6,
            false,
            SaeFitStage::Primary,
            &cancel,
            "Euclidean",
        )
        .expect("all-vanished state must be an exact structural result");
        let SaeStageFit::Null(report) = stage else {
            panic!("K=1 vanished boundary must not mint a manifold fit");
        };
        assert_eq!(report.vanished_atoms.iter().collect::<Vec<_>>(), vec![0]);
        assert_eq!(report.tier0.mean, Array1::<f64>::zeros(2));
        assert!(report.residual_sum_squares.is_finite());
        assert_eq!(report.fitted, Array2::<f64>::zeros((8, 2)));
    }
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
fn run_sae_manifold_fit_on_target(request: SaeFitRequest) -> Result<SaeFitOutcome, SaeFitError> {
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
    base_term
        .assignment
        .validate_rho_domain(&init_rho)
        .map_err(SaeFitError::Fit)?;
    // #2138 — the whole entry runs on the binding's GIL-released worker thread, so
    // interruptibility is the shared `cancel` flag rather than a per-fit thread.
    // Each objective polls it and bails its next outer eval when the caller sets
    // it on interrupt. Absent ⇒ a fresh, never-set flag (no cancellation).
    let cancel_flag = cancel.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

    let mut objective = match fit_outer_stage_to_boundary(
        base_term,
        &z,
        &registry,
        init_rho,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        run_outer_rho_search,
        SaeFitStage::Primary,
        &cancel_flag,
        metric_provenance,
    )? {
        SaeStageFit::Certified(objective) => objective,
        SaeStageFit::Null(report) => return Ok(SaeFitOutcome::Null(report)),
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
    let structured_passes = structured_residual_passes;
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
        // `promote_from_residual` is an explicit typed stage switch. Evidence
        // gates candidates inside the stage; a direct fit never enters it.
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
            let stage = SaeFitStage::StructuredResidual {
                pass: pass + 1,
                total_passes,
            };
            let mut objective = match fit_outer_stage_to_boundary(
                term,
                &z,
                &registry,
                rho,
                max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                run_outer_rho_search,
                stage,
                &cancel_flag,
                installed_label,
            )? {
                SaeStageFit::Certified(objective) => objective,
                SaeStageFit::Null(report) => return Ok(SaeFitOutcome::Null(report)),
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
        // Structure search is a convergent greedy coordinate search. Each round
        // proposes the strongest birth, fission, and fusion direction, permits
        // one certified structural move, refits it to convergence, and repeats
        // until a round applies no move. This keeps candidate memory bounded
        // independently of K and p without a size-dependent skip or round cap.
        let harvest_params = structure_harvest::HarvestParams {
            max_fusions: 1,
            max_fissions: 1,
            max_births: 1,
        };
        let refit_params = structure_harvest::ProductionRefitParams {
            inner_max_iter: max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        };
        let budget = MoveBudget {
            max_moves: 1,
            alpha: 0.05,
        };
        // The evaluation half is streamed one row per shard. The shard count is
        // therefore derived from the sample size rather than an optimization
        // knob, while memory remains O(N) for the row-index partition.
        let n_shards = n_obs.saturating_sub(n_obs / 2).max(1);
        let config = structure_harvest::RoundDriverConfig {
            n_shards,
            budget,
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
                Some(structure_harvest::rounds_to_json(&result.rounds)?)
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
    // term + ρ for EVERY atom (seed and born). Failure to reform the final
    // joint covariance is an inference failure, not a reason to substitute a
    // different per-atom covariance model.
    if structure_changed || finalization_invalidated_shape_uncertainty {
        let joint_registry = registry.clone();
        shape_uncertainty = term.recompute_joint_shape_uncertainty(
            z.view(),
            &rho,
            Some(&joint_registry),
            max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        // The certificate dispersion was seeded from the (now stale) φ̂;
        // refresh it to the final joint recompute's value.
        term.set_certificate_dispersion(shape_uncertainty.dispersion)?;
    }
    if shape_uncertainty.atoms.len() != term.k_atoms() {
        return Err(SaeFitError::Fit(
            "final joint shape uncertainty does not match the fitted atom count".to_string(),
        ));
    }
    for (atom_idx, uncertainty) in shape_uncertainty.atoms.iter().enumerate() {
        match (
            &uncertainty.band_coords,
            &uncertainty.band_mean,
            &uncertainty.band_sd,
        ) {
            (None, None, None) => {
                if uncertainty.decoder_covariance.is_some() || uncertainty.band_sd_robust.is_some()
                {
                    return Err(SaeFitError::Fit(format!(
                        "atom {atom_idx} has a partial unavailable shape-uncertainty payload"
                    )));
                }
            }
            (Some(coords), Some(mean), Some(sd)) => {
                if coords.nrows() != mean.nrows()
                    || mean.dim() != sd.dim()
                    || coords
                        .iter()
                        .chain(mean.iter())
                        .chain(sd.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(SaeFitError::Fit(format!(
                        "atom {atom_idx} has inconsistent or non-finite joint shape uncertainty"
                    )));
                }
                if let Some(covariance) = &uncertainty.decoder_covariance
                    && covariance.iter().any(|value| !value.is_finite())
                {
                    return Err(SaeFitError::Fit(format!(
                        "atom {atom_idx} has non-finite decoder covariance"
                    )));
                }
                if let Some(robust) = &uncertainty.band_sd_robust
                    && (robust.dim() != sd.dim() || robust.iter().any(|value| !value.is_finite()))
                {
                    return Err(SaeFitError::Fit(format!(
                        "atom {atom_idx} has inconsistent robust shape uncertainty"
                    )));
                }
            }
            _ => {
                return Err(SaeFitError::Fit(format!(
                    "atom {atom_idx} has a partial joint shape-uncertainty band"
                )));
            }
        }
    }

    // Additive post-fit diagnostics (#980): the two-score per-atom lens and the
    // residual-gauge certificate. Per-atom ARD variances (∝ exp(−log_precision))
    // are threaded in when native ARD was enabled, else `None` per atom.
    term.assignment
        .validate_rho_domain(&rho)
        .map_err(SaeFitError::Fit)?;
    let ard_variances: Vec<Option<Array1<f64>>> = term
        .validated_ard_precisions(&rho)
        .map_err(SaeFitError::Fit)?
        .iter()
        .map(|precision| {
            if precision.is_empty() {
                None
            } else {
                Some(precision.mapv(|alpha| alpha.recip()))
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
    let amortized_encoder_consistency = term.amortized_encoder_consistency(z.view(), &rho)?;
    let mut certificate_ledger = CertificateLedger::new();
    certificate_ledger.record(&fit_diagnostics.residual_gauge);
    certificate_ledger.record(&CoordinateFidelityCertificate::new(
        &fit_diagnostics.coordinate_fidelity,
    ));
    certificate_ledger.record(&TopologyPersistenceCertificate::new(
        &fit_diagnostics.topology_persistence,
    ));
    if let Some(report) = &fit_diagnostics.incoherence_report {
        certificate_ledger.record(report);
    }

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

    // A structure certificate is evidence about a structure search, not a
    // generic stationary-fit badge. An absent/skipped search therefore carries
    // no empty-ledger certificate.
    let structure_certificate_json = structure_search_json
        .as_ref()
        .map(|_| {
            structure_ledger
                .certify(0.05)
                .map_err(|error| error.to_string())
                .and_then(|certificate| {
                    serde_json::to_string(&certificate).map_err(|error| error.to_string())
                })
        })
        .transpose()?;

    Ok(SaeFitOutcome::Manifold(SaeFitReport {
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
        amortized_encoder_consistency,
        certificate_ledger,
        structure_search_json,
        structure_certificate_json,
        reported_log_alpha,
    }))
}

/// Fully typed request for the EVALUATION-ONLY certification entry (#2266):
/// diagnostics + certificates for an externally-trained (torch-lane) fit,
/// WITHOUT running any closed-form solve. `base_term` must already carry the
/// external decoder / coordinates / gate logits exactly as a fit seed would
/// (mirrors [`SaeFitRequest::base_term`]); `initial_rho` is installed as the
/// certified ρ verbatim — the only transform applied is
/// [`SaeManifoldRho::for_assignment`], which binds the flat-layout tag to the
/// term's assignment family and touches no numeric value.
///
/// There are deliberately no pipeline flags beyond `run_structure_search`: no
/// `promote_from_residual`, no `run_outer_rho_search`, no
/// `structured_residual_passes` — this entry never runs an outer search or an
/// inner solve, so those switches have nothing to govern.
pub struct SaeCertifyRequest {
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
    /// #977/#997 evidence-guarded structure search around the installed
    /// state. This is explicit opt-in: evaluation-only certification preserves
    /// the dictionary the external trainer supplied unless asked to search.
    pub run_structure_search: bool,
}

/// Zero-optimization certification entry (#2263). Installs an externally
/// trained SAE-manifold state verbatim, measures its exact inner KKT residual,
/// then applies the shared analytic outer-criterion certificate at the supplied
/// rho. A failed audit returns [`SaeExternalEvaluationReport`], never a fit.
/// Only a state that independently passes both authorities enters the native
/// post-fit diagnostics and optional structure-evidence pipeline.
///
/// KEEP IN SYNC WITH `run_sae_manifold_fit_on_target`'s postlude (#2266): the
/// shared post-fit pipeline is duplicated here rather than extracted into a
/// common helper, because the source function is under concurrent edit
/// elsewhere in this workspace and an extraction risked colliding with that
/// churn. A change to one postlude must be mirrored in the other.
pub fn run_sae_manifold_certify(
    request: SaeCertifyRequest,
) -> Result<SaeExternalCertificationOutcome, SaeFitError> {
    let SaeCertifyRequest {
        base_term,
        target: z,
        registry,
        initial_rho,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        alpha,
        isometry_pin_active,
        metric_provenance,
        run_structure_search,
    } = request;
    let (n_obs, p_out) = z.dim();
    let mut term = base_term;
    // Bind the flat assignment-strength layout tag to the term's assignment
    // family; this changes no numeric value, so `rho` is otherwise installed
    // verbatim from the caller.
    let rho = initial_rho.for_assignment(term.assignment.mode);
    term.assignment
        .validate_rho_domain(&rho)
        .map_err(SaeFitError::Fit)?;

    let inner_audit = installed_inner_kkt_audit(&mut term, z.view(), &rho, &registry)?;
    if !inner_audit.certifies() {
        return Ok(external_nonstationary_report(
            inner_audit.clone(),
            None,
            format!(
                "installed external state failed inner KKT stationarity: raw={:.6e}, quotient={:.6e}, bound={:.6e}",
                inner_audit.raw_gradient_norm,
                inner_audit.quotient_gradient_norm,
                inner_audit.stationarity_bound,
            ),
        ));
    }

    // Construct the ordinary native outer objective in frozen installed-state
    // mode. The audit evaluates the exact supplied point once; it runs neither
    // an inner update nor an outer optimization loop.
    let rho_flat = rho.to_flat();
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        Some(registry.clone()),
        rho,
        0,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
    )
    .for_installed_state_audit();
    let outer_result = match audit_stationary_point(
        &mut objective,
        rho_flat,
        "SAE external installed-state audit",
    ) {
        Ok(result) => result,
        Err(rejection) => {
            return Ok(external_nonstationary_report(
                inner_audit,
                Some(&rejection.result),
                rejection.source.to_string(),
            ));
        }
    };
    objective
        .certify_installed_state_audit(&outer_result)
        .map_err(SaeFitError::Fit)?;
    let mut shape_uncertainty = objective.decoder_shape_uncertainty()?;
    let fitted_result = objective.into_fitted().map_err(SaeFitError::Fit)?;
    let mut term = fitted_result.term;
    let mut rho = fitted_result.rho;
    let penalized_quasi_laplace_criterion = fitted_result.penalized_quasi_laplace_criterion;
    let outer_termination = fitted_result.termination;

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

    term.clear_row_loss_weights();
    term.set_certificate_dispersion(shape_uncertainty.dispersion)?;
    term.set_atom_inner_fits(z.view(), shape_uncertainty.dispersion)?;

    // #977 / #997 — the same evidence-guarded structure search
    // `run_sae_manifold_fit_on_target` runs around a native fit, applied here
    // around the installed external state.
    let mut structure_ledger = StructureLedger::new();
    let mut structure_changed = false;
    let structure_search_json = 'structure: {
        if !run_structure_search {
            break 'structure None;
        }
        let harvest_params = structure_harvest::HarvestParams {
            max_fusions: 1,
            max_fissions: 1,
            max_births: 1,
        };
        let refit_params = structure_harvest::ProductionRefitParams {
            inner_max_iter: max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        };
        let budget = MoveBudget {
            max_moves: 1,
            alpha: 0.05,
        };
        let n_shards = n_obs.saturating_sub(n_obs / 2).max(1);
        let config = structure_harvest::RoundDriverConfig {
            n_shards,
            budget,
            harvest_params,
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
                Some(structure_harvest::rounds_to_json(&result.rounds)?)
            }
            Err(e) => {
                return Err(SaeFitError::Fit(format!(
                    "structure search around SAE certify entry failed: {e}"
                )));
            }
        }
    };

    // The structure search may have changed K or refit the dictionary; the
    // shape bands / certificate dispersion / atom inner fits formed above are
    // then stale and must be rebuilt from the FINAL term + ρ, exactly as
    // `run_sae_manifold_fit_on_target` does when its own `structure_changed`.
    if structure_changed {
        shape_uncertainty = term.recompute_joint_shape_uncertainty(
            z.view(),
            &rho,
            Some(&registry),
            max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )?;
        term.set_certificate_dispersion(shape_uncertainty.dispersion)?;
        term.set_atom_inner_fits(z.view(), shape_uncertainty.dispersion)?;
    }

    let k_atoms = term.k_atoms();
    if shape_uncertainty.atoms.len() != k_atoms {
        return Err(SaeFitError::Fit(
            "final joint shape uncertainty does not match the certified atom count".to_string(),
        ));
    }
    for (atom_idx, uncertainty) in shape_uncertainty.atoms.iter().enumerate() {
        match (
            &uncertainty.band_coords,
            &uncertainty.band_mean,
            &uncertainty.band_sd,
        ) {
            (None, None, None) => {
                if uncertainty.decoder_covariance.is_some() || uncertainty.band_sd_robust.is_some()
                {
                    return Err(SaeFitError::Fit(format!(
                        "atom {atom_idx} has a partial unavailable shape-uncertainty payload"
                    )));
                }
            }
            (Some(coords), Some(mean), Some(sd)) => {
                if coords.nrows() != mean.nrows()
                    || mean.dim() != sd.dim()
                    || coords
                        .iter()
                        .chain(mean.iter())
                        .chain(sd.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(SaeFitError::Fit(format!(
                        "atom {atom_idx} has inconsistent or non-finite joint shape uncertainty"
                    )));
                }
                if let Some(covariance) = &uncertainty.decoder_covariance
                    && covariance.iter().any(|value| !value.is_finite())
                {
                    return Err(SaeFitError::Fit(format!(
                        "atom {atom_idx} has non-finite decoder covariance"
                    )));
                }
                if let Some(robust) = &uncertainty.band_sd_robust
                    && (robust.dim() != sd.dim() || robust.iter().any(|value| !value.is_finite()))
                {
                    return Err(SaeFitError::Fit(format!(
                        "atom {atom_idx} has inconsistent robust shape uncertainty"
                    )));
                }
            }
            _ => {
                return Err(SaeFitError::Fit(format!(
                    "atom {atom_idx} has a partial joint shape-uncertainty band"
                )));
            }
        }
    }

    term.assignment
        .validate_rho_domain(&rho)
        .map_err(SaeFitError::Fit)?;
    let ard_variances: Vec<Option<Array1<f64>>> = term
        .validated_ard_precisions(&rho)
        .map_err(SaeFitError::Fit)?
        .iter()
        .map(|precision| {
            if precision.is_empty() {
                None
            } else {
                Some(precision.mapv(|alpha| alpha.recip()))
            }
        })
        .collect();
    let assignments = term.assignment.assignments();
    let fitted = term.try_fitted_target_aware(z.view(), Some(&rho))?;
    term.record_fit_data_collapse_if_needed(z.view(), fitted.view(), assignments.view(), max_iter)?;
    let trust_diagnostics = term.trust_diagnostics_report(assignments.view())?;
    let fit_diagnostics = term.fit_diagnostics_report(
        Some(&ard_variances),
        isometry_pin_active,
        Some(shape_uncertainty.dispersion),
        fitted.view(),
        Some(assignments.view()),
    )?;
    let amortized_encoder_consistency = term.amortized_encoder_consistency(z.view(), &rho)?;
    let mut certificate_ledger = CertificateLedger::new();
    certificate_ledger.record(&fit_diagnostics.residual_gauge);
    certificate_ledger.record(&CoordinateFidelityCertificate::new(
        &fit_diagnostics.coordinate_fidelity,
    ));
    certificate_ledger.record(&TopologyPersistenceCertificate::new(
        &fit_diagnostics.topology_persistence,
    ));
    if let Some(report) = &fit_diagnostics.incoherence_report {
        certificate_ledger.record(report);
    }

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

    // A structure certificate exists only when this entry genuinely ran the
    // structure search. Stationarity alone certifies the installed fit state,
    // not unsearched dictionary alternatives.
    let structure_certificate_json = structure_search_json
        .as_ref()
        .map(|_| {
            structure_ledger
                .certify(0.05)
                .map_err(|error| error.to_string())
                .and_then(|certificate| {
                    serde_json::to_string(&certificate).map_err(|error| error.to_string())
                })
        })
        .transpose()?;
    let loss = term.loss(z.view(), &rho)?;

    Ok(SaeExternalCertificationOutcome::Certified(SaeFitReport {
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
        structured_residual_diagnostics: Vec::new(),
        trust_diagnostics,
        fit_diagnostics,
        amortized_encoder_consistency,
        certificate_ledger,
        structure_search_json,
        structure_certificate_json,
        reported_log_alpha,
    }))
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
