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

#[cfg(test)]
mod test_support;

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
    CohortNodes, CovariateCells, CovariateSegment, CovariateValue, Event, EventHistoryCohort,
    EventHistoryError, MarkKind, SubjectHistory, SubjectNodes, code_covariate_value,
    mark_index_of, observed_mark_vocabulary, resolve_mark_vocabulary,
};
pub use covariance::{effective_rank, temporal_covariance};
pub use family::{
    DecisionIntegral, EventHistoryFamily, EventHistoryFit, QuadratureCertificate, RankStep,
    RefinementCheck, RiskSetCentring, UnresolvedGrowth, fit_event_history_formulas,
};
pub use forecast::{
    Forecast, ForecastRequest, FutureSegment, HistoryForecastRequest, PopulationForecastRequest,
    SmoothedLatentState, SpellPit, forecast, forecast_history, latent_state, pit_uniform_distance,
    population_forecast, predictive_pit, baseline_log_rates,
};
pub use preserve::{ReferenceGrid, ReferenceStrata};

#[cfg(test)]
mod tests;
