// Split from the original oversized module; keep included in order.
include!("workflow/imports.rs");

/// Shared analytic-penalty descriptor parser. Both this in-process workflow
/// pipeline and the Python FFI (`gam-pyffi`) build their analytic-penalty
/// registries through [`descriptors::build_analytic_penalty_registry_from_descriptors`],
/// so the descriptor schema, defaults, shape checks, and error messages are
/// identical for every caller.
pub mod descriptors;

include!("workflow/fit_request.rs");
include!("workflow/materialize.rs");
