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

pub use columns::{resolve_offset_column, resolve_weight_column};
pub use family::resolve_family;
pub use survival_time::{PreparedSurvivalTimeStack, prepare_survival_time_stack};
pub use validation::is_binary_response;

pub(crate) use family::response_column_kind;
pub(crate) use location_scale::materialize_location_scale;
pub(crate) use marginal_slope::materialize_bernoulli_marginal_slope;
pub(crate) use standard::materialize_standard;
pub(crate) use survival::materialize_survival;
pub(crate) use terms::{
    build_termspec_with_geometry_and_overrides, gate_duchon_operator_penalties_for_family,
    prune_unidentified_linear_terms_for_marginal_slope,
};
pub(crate) use transformation::materialize_transformation_normal;
pub(crate) use validation::{
    reject_marginal_slope_controls_for_transformation_normal,
    reject_survival_likelihood_for_nonsurvival, reject_survival_only_terms_for_nonsurvival,
};

use latent::*;
use secondary::*;
use terms::*;
use validation::*;

#[cfg(test)]
mod tests;
