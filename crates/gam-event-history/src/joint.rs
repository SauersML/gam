//! The joint latent-signature event model (#2961).
//!
//! One continuous-time latent state with positive signature contributions,
//! genetics-dependent trajectories, learned disease-associated transitions and
//! marginally anchored posterior-predictive risks (`docs/latent-signatures.md`).
//! Its components land only as parts of one Rust-owned production path: fit a
//! cohort, condition on a new history, forecast, save and reload. gam-cli,
//! gam-pyffi and gamfit call that path.
//!
//! Rank zero is exact. Without signatures every mark's intensity is a constant
//! rate and the reference normaliser is identically one, so the rates have a
//! closed-form Gamma posterior under the empirical-Bayes exponential prior.
//! Forecasts average that posterior; they never insert a fitted rate.
//!
//! Every planned submodule has a slot below. Its owner replaces the slot line
//! with the module declaration when the module lands with its production caller.

// model.rs (jls-serve): production entry, conditioning, forecasts, saved artifact
mod model;
pub use model::{ConditionedJointModel, JointEventModel, JointForecast, fit_joint_event_model};

// law.rs (jls-law after slice 0): specification, history, complete-path density
mod law;

// jumps.rs (jls-jumps): learned disease-associated state jumps
// [unlanded: jumps.rs]

// constant_rate_inference.rs (jls-serve): exact rank-zero rate posterior and its forecasts
mod constant_rate_inference;

// reference_rank_zero.rs (jls-serve): exact reference law without signatures
// [unlanded: reference_rank_zero.rs]

// decoder.rs (jls-law): positive normalized signature decoder
// [unlanded: decoder.rs]

// emission.rs (jls-law): measurement observation channels
mod emission;

// category_prior.rs (jls-law): simplex measure on binary and ordinal baselines
// [unlanded: category_prior.rs]

// precision.rs (jls-posterior): block-tridiagonal state precision with a genetic border
// [unlanded: precision.rs]

// posterior.rs (jls-posterior): structured state-space posterior
// [unlanded: posterior.rs]

// transport.rs (jls-posterior): fixed OU innovation coordinates
// [unlanded: transport.rs]

// reference.rs (jls-reference): reference evolution with jumps and mortality
// [unlanded: reference.rs]

// reference_value.rs (jls-reference): reference moment values M_d
// [unlanded: reference_value.rs]

// reference_sensitivity.rs (jls-reference): analytic reference sensitivities
// [unlanded: reference_sensitivity.rs]

// reference_functional.rs (jls-reference): reference functionals
// [unlanded: reference_functional.rs]

// resolution.rs (jls-reference): independent reference replication and refinement
// [unlanded: resolution.rs]

// function_prior.rs (jls-priors): normalized final-function priors
// [unlanded: function_prior.rs]

// function_prior_tests.rs (jls-priors): function-prior test oracles
// [unlanded: function_prior_tests.rs]

// decoder_prior.rs (jls-priors): decoder weight prior
// [unlanded: decoder_prior.rs]

// structural_prior.rs (jls-priors): structural priors with chart Jacobians
// [unlanded: structural_prior.rs]

// structure.rs (jls-priors): zero-effect and reduced-rank structure models
// [unlanded: structure.rs]

// score.rs (jls-fit): analytic complete-path coefficient scores
// [unlanded: score.rs]

// cohort.rs (jls-fit): cohort integrals under shared reference strata
// [unlanded: cohort.rs]

// cohort_resolution.rs (jls-fit): cohort integration resolution
// [unlanded: cohort_resolution.rs]

// integration.rs (jls-fit): importance integration with analytic scores
// [unlanded: integration.rs]

// coefficient_inference.rs (jls-fit): coefficient posterior inference
// [unlanded: coefficient_inference.rs]

// coefficient_integral.rs (jls-fit): coefficient integrals
// [unlanded: coefficient_integral.rs]

// coefficient_pilot.rs (jls-fit): pilot coefficient integration
// [unlanded: coefficient_pilot.rs]

// coefficient_proposal.rs (jls-fit): guided coefficient proposals
// [unlanded: coefficient_proposal.rs]

// strength_fit.rs (jls-fit): empirical-Bayes strengths through opt
// [unlanded: strength_fit.rs]

// coefficient_prediction.rs (jls-predict): coefficient-integrated predictive densities
// [unlanded: coefficient_prediction.rs]

// conditional_prediction.rs (jls-predict): continuation densities of earlier histories
// [unlanded: conditional_prediction.rs]

// forecast.rs (jls-predict): posterior-predictive absolute risks with signatures
// [unlanded: forecast.rs]

// tests.rs (jls-verify): acceptance fixtures A1-A6
// [unlanded: tests.rs]
