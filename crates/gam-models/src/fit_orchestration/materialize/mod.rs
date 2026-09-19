use super::*;

mod columns;
mod family;
mod latent;
mod location_scale;
mod marginal_slope;
mod secondary;
mod standard;
mod survival;
mod survival_time;
mod terms;
mod transformation;
mod validation;

pub use columns::{
    expand_automatic_fit_formula, fit_required_columns, formula_columns,
    resolve_fit_weight_column, resolve_offset_column,
    resolve_weight_column,
};
pub(crate) use columns::resolve_continuous_column;
pub(crate) use family::code_two_level_label_response;
pub use family::{
    FamilyNuisanceOverrides, is_multinomial_family_name, resolve_family, response_column_kind,
    scalar_family_from_name,
};
pub use survival_time::{PreparedSurvivalTimeStack, prepare_survival_time_stack};
pub use validation::is_binary_response;

pub(crate) use location_scale::materialize_location_scale;
pub(crate) use marginal_slope::materialize_bernoulli_marginal_slope;
pub(crate) use standard::materialize_standard;
pub(crate) use survival::materialize_survival;
pub use terms::gate_duchon_operator_penalties_for_family;
pub(crate) use terms::{
    build_termspec_with_geometry_and_overrides, prune_unidentified_linear_terms_for_marginal_slope,
};
pub(crate) use transformation::materialize_transformation_normal;
pub(crate) use validation::{
    reject_marginal_slope_controls_for_transformation_normal,
    reject_survival_only_config_for_nonsurvival, reject_survival_only_terms_for_nonsurvival,
    requests_bernoulli_marginal_slope,
};

use latent::*;
use secondary::*;
use validation::*;

/// The custom-family solver options a materialized request takes from the
/// caller's configuration, resolved once.
///
/// A request builder that spread `..BlockwiseFitOptions::default()` had to
/// restate every caller field it meant to keep, and one that forgot a field
/// dropped it silently. The latent survival and latent binary requests never
/// read `FitConfig::compute_covariance`, so the default `false` withheld the
/// conditional covariance of every latent fit, including fits whose truncated
/// cone moments were available (#2677 B0). `None` computes the covariance,
/// which the default posterior-mean prediction reads (SPEC rule 3).
///
/// Location-scale and the binomial link-wiggle refit do not take their options
/// from here: their model is incomplete without the joint posterior, so their
/// fit drivers force covariance at the final fit and keep the pilots cheap.
fn blockwise_fit_options(config: &FitConfig) -> BlockwiseFitOptions {
    with_caller_warm_start(
        BlockwiseFitOptions {
            compute_covariance: config.compute_covariance.unwrap_or(true),
            persistent_warm_start_store: config.persistent_warm_start_store.clone(),
            ..BlockwiseFitOptions::default()
        },
        config,
    )
}

/// A `warm_start_from` point on a custom-family request (gam#3002). Every
/// custom-family request built from a `FitConfig` passes through here, so the
/// point reaches the outer driver on every such route.
fn with_caller_warm_start(
    mut options: BlockwiseFitOptions,
    config: &FitConfig,
) -> BlockwiseFitOptions {
    options.warm_start = config.warm_start.clone();
    options
}

#[cfg(test)]
mod tests;
