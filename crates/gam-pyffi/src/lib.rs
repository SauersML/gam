// Split from the original oversized module; keep included in order.
include!("ffi_prelude.rs");

mod benchmark_scores;

mod competing_risks_decode;

mod inference_instruments;

mod manifold_pyclasses;

mod python_literal;

mod sklearn_metadata;

mod summary_render;

mod survival_surface_io;

include!("model_ffi.rs");
include!("latent_basis_and_sae_ffi.rs");
include!("reml_latent_fit_ffi.rs");
include!("geometry_ffi.rs");
include!("manifold_and_posterior_ffi.rs");
