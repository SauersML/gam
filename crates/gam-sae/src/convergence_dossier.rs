//! M6 — the convergence dossier: a single grep-able JSON that answers "did this
//! fit converge, and why should I believe it?" in one file.
//!
//! The signals that certify a settled SAE-manifold fit are each produced in a
//! different place and, historically, land in different logs / payload corners:
//! the outer probe-budget telemetry ([`OuterProbeTelemetry`]), the amortized
//! warm-start tally ([`AmortizedWarmStartTelemetry`]), the curvature-homotopy
//! walk report ([`CurvatureWalkReport`]), the finite-difference optimality
//! certificate ([`CriterionCertificate`]), the active-mass collapse ledger
//! ([`CollapseEvent`]s), and the seed/pristine/canonicalization guard verdicts
//! carried on [`SaeIntoFittedResult`]. For the scale campaign we want one artifact
//! that collects them, so a run's convergence provenance is a single JSON to grep,
//! not a scavenger hunt across telemetry surfaces.
//!
//! This module does NOT recompute any of those signals — it is pure aggregation.
//! [`SaeManifoldOuterObjective::into_fitted_with_dossier`] snapshots the telemetry
//! the objective already accumulated, evaluates the optimality certificate (the
//! same four value-path probes as the standalone certificate), consumes the
//! objective through the unmodified [`SaeManifoldOuterObjective::into_fitted`], and
//! reads the settled term's collapse ledger. Structure-search signals that a plain
//! outer REML fit does not produce (the `SearchLedger` rounds, an EFS λ-trajectory)
//! are attachable after the fact via [`ConvergenceDossier::with_search_ledger_json`]
//! and [`ConvergenceDossier::with_efs_trajectory`], so a driver that ran structure
//! search can fold those in without this module reaching into a subsystem it does
//! not own.

use serde::Serialize;

use crate::certificates::CriterionCertificate;
use crate::manifold::{
    AmortizedWarmStartTelemetry, CollapseEvent, CurvatureWalkReport, OuterProbeTelemetry,
    SaeIntoFittedResult, SaeManifoldOuterObjective,
};

/// The tolerance the dossier's headline `optimality.passes` verdict uses for the
/// FD/analytic directional agreement at the settled optimum. A relative gap above
/// this (or a non-well-posed curvature) is the loud desync/rail flag.
pub const CONVERGENCE_DOSSIER_AGREEMENT_TOL: f64 = 1e-3;

/// Outer probe-budget telemetry (mirror of [`OuterProbeTelemetry`], serialized).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct OuterProbeDossier {
    pub criterion_calls: usize,
    pub fd_probe_calls: usize,
    pub infeasible_non_pd_per_row: usize,
    pub infeasible_cross_row: usize,
    pub infeasible_schur: usize,
    pub infeasible_inner_not_converged: usize,
    pub infeasible_total: usize,
    pub wall_cost_value_probes: usize,
    /// A nonzero count is a REGRESSION: a probe lane that mutated the committed
    /// basin (the #629/#630/#2080 stateful-objective corruption).
    pub mutating_value_probes: usize,
}

impl From<OuterProbeTelemetry> for OuterProbeDossier {
    fn from(t: OuterProbeTelemetry) -> Self {
        Self {
            criterion_calls: t.criterion_calls,
            fd_probe_calls: t.fd_probe_calls,
            infeasible_non_pd_per_row: t.infeasible_non_pd_per_row,
            infeasible_cross_row: t.infeasible_cross_row,
            infeasible_schur: t.infeasible_schur,
            infeasible_inner_not_converged: t.infeasible_inner_not_converged,
            infeasible_total: t.infeasible_total(),
            wall_cost_value_probes: t.wall_cost_value_probes,
            mutating_value_probes: t.mutating_value_probes,
        }
    }
}

/// Amortized warm-start tally (mirror of [`AmortizedWarmStartTelemetry`]).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct WarmStartDossier {
    pub attempts: usize,
    pub warm_started_evals: usize,
    pub cold_fallback_evals: usize,
    pub failed_evals: usize,
    pub total_rows_warm_started: usize,
    /// `warm_started_evals > 0` — the verifiable "uses amortized warm-start" fact.
    pub used_amortized_warm_start: bool,
}

impl From<AmortizedWarmStartTelemetry> for WarmStartDossier {
    fn from(t: AmortizedWarmStartTelemetry) -> Self {
        Self {
            attempts: t.attempts,
            warm_started_evals: t.warm_started_evals,
            cold_fallback_evals: t.cold_fallback_evals,
            failed_evals: t.failed_evals,
            total_rows_warm_started: t.total_rows_warm_started,
            used_amortized_warm_start: t.warm_started_evals > 0,
        }
    }
}

/// Curvature-homotopy walk report (mirror of [`CurvatureWalkReport`]).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct CurvatureWalkDossier {
    /// Whether the walk reached `η = 1` on the certified optimal branch.
    pub arrived: bool,
    pub anchor_residual_norm_sq: f64,
    /// Whether a branch bifurcation (pivot collapse) was detected on the walk.
    pub bifurcated: bool,
    pub eta_steps: usize,
    pub step_halvings: usize,
    pub collapse_events: usize,
    pub reseeds: usize,
    /// A certified walk from the global anchor: arrived, no bifurcation, no
    /// reseeds, no arrival-state collapse events.
    pub clean_walk: bool,
}

impl From<&CurvatureWalkReport> for CurvatureWalkDossier {
    fn from(r: &CurvatureWalkReport) -> Self {
        let bifurcated = r.bifurcation.is_some();
        Self {
            arrived: r.arrived,
            anchor_residual_norm_sq: r.anchor_residual_norm_sq,
            bifurcated,
            eta_steps: r.eta_steps,
            step_halvings: r.step_halvings,
            collapse_events: r.collapse_events,
            reseeds: r.reseeds,
            clean_walk: r.arrived && !bifurcated && r.reseeds == 0 && r.collapse_events == 0,
        }
    }
}

/// First-order optimality certificate (mirror of [`CriterionCertificate`]) plus
/// its derived verdicts.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct OptimalityDossier {
    pub grad_norm: f64,
    pub fd_directional: f64,
    pub analytic_directional: f64,
    pub fd_error_bar: f64,
    pub step: f64,
    pub well_posed: bool,
    /// `|FD − analytic|` normalized to the larger directional magnitude, floored
    /// by the FD error bar.
    pub agreement_rel: f64,
    /// `well_posed && agreement_rel <= CONVERGENCE_DOSSIER_AGREEMENT_TOL`.
    pub passes: bool,
    pub agreement_tol: f64,
}

impl From<&CriterionCertificate> for OptimalityDossier {
    fn from(c: &CriterionCertificate) -> Self {
        Self {
            grad_norm: c.grad_norm,
            fd_directional: c.fd_directional,
            analytic_directional: c.analytic_directional,
            fd_error_bar: c.fd_error_bar,
            step: c.step,
            well_posed: c.well_posed,
            agreement_rel: c.agreement_rel(),
            passes: c.passes(CONVERGENCE_DOSSIER_AGREEMENT_TOL),
            agreement_tol: CONVERGENCE_DOSSIER_AGREEMENT_TOL,
        }
    }
}

/// One active-mass collapse event (mirror of [`CollapseEvent`]).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct CollapseEventDossier {
    pub iteration: usize,
    pub atom: usize,
    pub max_active_mass: f64,
    pub floor: f64,
    /// `"reseeded"` (one second chance from a fresh basin) or `"terminal"`
    /// (re-seed budget exhausted; the collapse is the objective's verdict).
    pub action: &'static str,
}

impl From<&CollapseEvent> for CollapseEventDossier {
    fn from(e: &CollapseEvent) -> Self {
        Self {
            iteration: e.iteration,
            atom: e.atom,
            max_active_mass: e.max_active_mass,
            floor: e.floor,
            action: match e.action {
                crate::manifold::CollapseAction::Reseeded => "reseeded",
                crate::manifold::CollapseAction::Terminal => "terminal",
            },
        }
    }
}

/// The post-fit guard verdicts carried on [`SaeIntoFittedResult`]: which basin
/// the returned state came from and whether chart canonicalization moved it.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct GuardVerdictDossier {
    pub used_seed_basin_fallback: bool,
    pub used_pristine_seed_fallback: bool,
    pub charts_canonicalized: bool,
    /// True when NONE of the post-fit guards fired — the outer walk's own settled
    /// state was returned verbatim (the clean convergence signature).
    pub returned_outer_state_verbatim: bool,
}

impl From<&SaeIntoFittedResult> for GuardVerdictDossier {
    fn from(f: &SaeIntoFittedResult) -> Self {
        Self {
            used_seed_basin_fallback: f.used_seed_basin_fallback,
            used_pristine_seed_fallback: f.used_pristine_seed_fallback,
            charts_canonicalized: f.charts_canonicalized,
            returned_outer_state_verbatim: !f.used_seed_basin_fallback
                && !f.used_pristine_seed_fallback
                && !f.charts_canonicalized,
        }
    }
}

/// The M6 convergence dossier: one JSON collecting every already-produced
/// convergence signal for a settled fit.
#[derive(Clone, Debug, Serialize)]
pub struct ConvergenceDossier {
    /// Schema tag so downstream greppers can pin the layout.
    pub schema: &'static str,
    pub outer_probe: OuterProbeDossier,
    pub warm_start: WarmStartDossier,
    /// `None` when no curvature-homotopy walk ran (a base-topology-only fit).
    pub curvature_walk: Option<CurvatureWalkDossier>,
    /// `None` when the optimality certificate could not be evaluated (a degenerate
    /// value path at ρ̂); the reason string is preserved in `optimality_error`.
    pub optimality: Option<OptimalityDossier>,
    pub optimality_error: Option<String>,
    pub collapse_ledger: Vec<CollapseEventDossier>,
    pub guard_verdict: GuardVerdictDossier,
    /// Whether the outer ρ-search ran on frozen (amortized) routing.
    pub routing_frozen: bool,
    /// Structure-search rounds JSON (`structure_harvest::rounds_to_json`), attached
    /// by a driver that ran structure search; `None` for a plain outer REML fit.
    pub search_ledger_json: Option<serde_json::Value>,
    /// An EFS λ-trajectory (per-iterate outer-coordinate flat values), attached by
    /// a driver that recorded one; `None` otherwise.
    pub efs_trajectory: Option<Vec<Vec<f64>>>,
    /// The single headline: did this fit converge with nothing to worry about?
    /// True iff the optimality certificate passed, the outer state was returned
    /// verbatim (no guard rescue), and there is no residual mutating-probe or
    /// terminal-collapse regression.
    pub converged_clean: bool,
}

impl ConvergenceDossier {
    /// Attach the structure-search rounds JSON produced by
    /// [`crate::structure_harvest::rounds_to_json`] (a `SearchLedger` slice).
    #[must_use]
    pub fn with_search_ledger_json(mut self, rounds_json: &str) -> Self {
        self.search_ledger_json = serde_json::from_str(rounds_json).ok();
        self
    }

    /// Attach an EFS λ-trajectory (one flat outer-coordinate vector per iterate).
    #[must_use]
    pub fn with_efs_trajectory(mut self, trajectory: Vec<Vec<f64>>) -> Self {
        self.efs_trajectory = Some(trajectory);
        self
    }

    /// Serialize to a pretty JSON string — the one grep-able artifact.
    pub fn to_json_pretty(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("ConvergenceDossier::to_json_pretty: {e}"))
    }

    /// Serialize to a compact JSON string.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("ConvergenceDossier::to_json: {e}"))
    }

    /// Assemble from the collected signals. Pure aggregation; recomputes nothing.
    fn assemble(
        outer_probe: OuterProbeDossier,
        warm_start: WarmStartDossier,
        curvature_walk: Option<CurvatureWalkDossier>,
        optimality: Option<OptimalityDossier>,
        optimality_error: Option<String>,
        collapse_ledger: Vec<CollapseEventDossier>,
        guard_verdict: GuardVerdictDossier,
        routing_frozen: bool,
    ) -> Self {
        let no_terminal_collapse = !collapse_ledger.iter().any(|e| e.action == "terminal");
        let converged_clean = optimality.as_ref().is_some_and(|o| o.passes)
            && guard_verdict.returned_outer_state_verbatim
            && outer_probe.mutating_value_probes == 0
            && no_terminal_collapse;
        Self {
            schema: "gam-sae/convergence-dossier/v1",
            outer_probe,
            warm_start,
            curvature_walk,
            optimality,
            optimality_error,
            collapse_ledger,
            guard_verdict,
            routing_frozen,
            search_ledger_json: None,
            efs_trajectory: None,
            converged_clean,
        }
    }
}

impl SaeManifoldOuterObjective {
    /// M6 — consume the settled objective, returning the fitted result AND the
    /// convergence dossier aggregating every already-produced convergence signal.
    ///
    /// This is the additive companion to [`Self::into_fitted`]: it snapshots the
    /// probe/warm-start telemetry and evaluates the optimality certificate (the
    /// same four value-path probes as [`Self::optimality_certificate`], taken on a
    /// cold clone of the pristine baseline term) BEFORE consuming the objective,
    /// then folds in the settled term's collapse ledger and the post-fit guard
    /// verdicts. The existing [`Self::into_fitted`] path is untouched — callers
    /// that do not want the dossier keep calling it directly.
    pub fn into_fitted_with_dossier(mut self) -> (SaeIntoFittedResult, ConvergenceDossier) {
        let outer_probe: OuterProbeDossier = self.probe_telemetry().into();
        let warm_start: WarmStartDossier = self.warm_start_telemetry().into();
        let curvature_walk: Option<CurvatureWalkDossier> =
            self.curvature_walk_report().map(CurvatureWalkDossier::from);
        let routing_frozen = self.routing_is_frozen();

        // The optimality certificate takes `&mut self` and must be evaluated
        // before `into_fitted` consumes the objective (it probes a cold clone of
        // the pristine baseline term, so it does not corrupt the settled state).
        let (optimality, optimality_error) = match self.optimality_certificate() {
            Ok(cert) => (Some(OptimalityDossier::from(&cert)), None),
            Err(err) => (None, Some(err)),
        };

        let fitted = self.into_fitted();
        let guard_verdict = GuardVerdictDossier::from(&fitted);
        let collapse_ledger: Vec<CollapseEventDossier> = fitted
            .term
            .collapse_events()
            .iter()
            .map(CollapseEventDossier::from)
            .collect();

        let dossier = ConvergenceDossier::assemble(
            outer_probe,
            warm_start,
            curvature_walk,
            optimality,
            optimality_error,
            collapse_ledger,
            guard_verdict,
            routing_frozen,
        );
        (fitted, dossier)
    }
}
