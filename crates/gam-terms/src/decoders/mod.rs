//! Front-end-agnostic decoder / transcoder primitives.
//!
//! These are general latent-decode primitives callable from the `gam` Rust
//! library, the CLI, and PyTorch via `gam-pyffi`. They are not specific to any
//! one SAE construction:
//!
//! * [`interchange_decoder`] — per-feature scalar-gate decoder with a masked
//!   interchange-swap variant (Distributed Alignment Search).

pub mod behavioral_head;
pub mod gated_decoder;
pub mod interchange_decoder;
