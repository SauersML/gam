//! Event histories: marked counting processes with smooth covariate and time
//! effects, a per-subject latent state made of unit-variance
//! Ornstein–Uhlenbeck atoms with evidence-selected loadings and rates that
//! enters as the individual deviation from a population log-intensity (the
//! loadings' Gaussian mixing is cancelled in the intensity, so the smooth
//! surfaces are population-average rates), marginalisation of the latent
//! chain by adaptive Gauss-Hermite–Lagrange filtering on grids centred at
//! each node's posterior mean, the exact gradient of that computed
//! likelihood from forward-mode duals through the same filter, and the
//! Hessian and its directional derivatives from Louis' identity accumulated
//! in coefficient space by one forward sweep. Forecasts and the predictive
//! PIT are expectations under the filtered state, with every probability a
//! chronological integral of a killed process.
//!
//! What is exact and what is certified: the latent integral at the nodes is
//! resolved by Gauss-Hermite quadrature, the latent path is sampled at the
//! nodes of a mesh, and the fit refines both — the order and the mesh —
//! until the fitted coefficients are stationary under refinement to a
//! stated fraction of their posterior standard deviation. That certificate,
//! not the word "exact", is what the fit carries.
//!
//! Survival with a slow frailty, competing risks (terminal marks), once-only
//! marks, recurrent events and history-conditioned prediction are all one
//! family here: they differ only in the rows and the mark kinds. The rank of
//! the latent covariance is grown from zero by the evidence: each atom is
//! proposed by the covariance score of the residuals, its loadings get the
//! Gaussian prior whose precision maximises the marginal likelihood under
//! the exact one-dimensional marginal of the score's quartic evidence model,
//! and it enters exactly when that prior places the loading's posterior
//! mode away from zero. The reported latent object is the posterior-mean
//! covariance `C(Δ)` with its eigenmodes and their uncertainty; the smoothed
//! latent state of every subject is exposed with its covariance.

mod chain;
mod cohort;
mod covariance;
mod family;
mod forecast;
mod formula;
mod marginal;
mod preserve;
mod scalar;

pub use cohort::{
    CohortNodes, CovariateSegment, Event, EventHistoryCohort, EventHistoryError, MarkKind,
    SubjectHistory, SubjectNodes, design_rows, expand_nodes, quadrature_order_for_degree,
};
pub use covariance::{
    DirectionEvidence, DirectionProfile, RidgeProfile, effective_rank, eigenmodes,
    empirical_bayes_ridge, factor_covariance, quartic_moments, temporal_covariance,
};
pub use family::{
    EventHistoryFamily, EventHistoryFit, EventHistorySpec, JointEvaluation, QuadratureCertificate,
    RankStart, RankStep, ReferenceTables, RefinementCheck, RiskSetCentring, fit_event_history,
    fit_event_history_formula, fit_event_history_formulas, latent_block_spec, mark_block_spec,
    seeded_one, seeded_two,
};
pub use forecast::{
    Forecast, ForecastRequest, FutureSegment, HistoryForecastRequest, PopulationForecastRequest,
    SmoothedLatentState, SpellPit, forecast, forecast_history, kolmogorov_smirnov_uniform,
    latent_state, pit_uniform_distance, population_forecast, predictive_pit, training_eta,
};
pub use formula::{TIME_COLUMN, covariate_spec_from_formula, node_dataset};
pub use marginal::transition_score_polynomials;
pub use preserve::{ReferenceGrid, ReferenceStrata};

#[cfg(test)]
mod tests;
