//! Front-end-agnostic decoder / transcoder primitives.
//!
//! These are general latent-decode primitives callable from the `gam` Rust
//! library, the CLI, and PyTorch via `gam-pyffi`. They are not specific to any
//! one SAE construction:
//!
//! * [`interchange_decoder`] — per-feature scalar-gate decoder with a masked
//!   interchange-swap variant (Distributed Alignment Search).
//! * [`skip_transcoder`] — closed-form Gaussian REML/Laplace score of a
//!   trained skip-transcoder.

pub mod interchange_decoder;
pub mod skip_transcoder;
