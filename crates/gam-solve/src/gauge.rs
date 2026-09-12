// One Gauge object (#933) — moved DOWN into the neutral `gam-problem`
// crate (#1521 contract inversion). The whole `Gauge` section type, its
// builders, and the `assemble_block_triangular_t` assembler now live in
// `gam_problem::gauge` so the `terms`/`basis` and `identifiability`
// layers stop reaching UP into `solver` for it. This module re-exports
// the neutral type so every existing `crate::gauge::Gauge`
// (and `crate::gauge::assemble_block_triangular_t`) path keeps
// resolving unchanged.

pub use gam_problem::gauge::{Gauge, assemble_block_triangular_t};
