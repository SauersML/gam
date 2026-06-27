//! #1521: the quadrature implementation was carved into `gam-solve`. The root
//! `gam` crate re-exports it so `gam::quadrature::*` (and `gam::inference::quadrature`)
//! resolve to the single canonical `gam_solve::quadrature` types — eliminating the
//! stale monolith duplicate of `QuadratureContext` that diverged from the carved copy.
pub use gam_solve::quadrature::*;
