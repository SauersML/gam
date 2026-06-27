//! `gam-model-kernels` — the shared BASE kernel layer carved out of
//! `gam-models` under #1521. These are the family-agnostic infrastructure
//! modules that every concrete family (survival / bms / gamlss / multinomial /
//! custom) consumes: the cubic-cell / cell-moment kernels, the design-scaling,
//! link, and root-solving primitives.
//!
//! The crate sits strictly BELOW `gam-models`: it depends only on the lower
//! tiers (`gam-linalg`, `gam-math`, `gam-problem`, `gam-runtime`, `gam-solve`)
//! and on its own sibling modules — never up into a concrete family. The
//! modules are re-exported at the `gam-models` crate root so the relocated
//! families' `crate::cubic_cell_kernel::*` / `crate::scale_design::*` paths
//! resolve unchanged.

// `impl_reason_error_boilerplate!` derive, shared with `gam-models`. Brought
// into crate-wide textual scope here so the relocated modules' unqualified
// `impl_reason_error_boilerplate! { .. }` call sites resolve unchanged.
#[macro_use]
mod macros;

pub mod cell_moment_family;
pub mod cubic_cell_kernel;
pub mod inverse_link;
pub mod monotone_root;
pub mod penalized_projection;
pub mod scale_design;
pub mod sigma_link;
