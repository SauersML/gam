//! Marked event histories with shared Gaussian OU latent factors.
//!
//! Reference centring is differentiated through the population evolution.
//! Its time discretisation is checked at fixed coefficients, and the final
//! coefficients and centring snapshot are exported together. Forecasts have
//! explicit reference-horizon and entry-conditioning contracts.
//!
//! Latent integration uses product Gauss-Hermite grids. It is an approximation
//! with limited rank capacity; streaming backward rows avoids a quadratic
//! state-memory allocation. Filtered residuals propose factor rates, while
//! likelihood derivatives check loading curvature. Sampled directional
//! profiles and Laplace rank comparisons remain approximations.

mod chain;
mod cohort;
mod covariance;
mod family;
mod forecast;
mod formula;
pub mod joint;
mod marginal;
mod preserve;
mod scalar;
mod static_state;

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
    baseline_log_rates,
};
pub use formula::{TIME_COLUMN, covariate_spec_from_formula, node_dataset};
pub use marginal::transition_score_polynomials;
pub use preserve::{ReferenceGrid, ReferenceStrata};

#[cfg(test)]
mod tests;
