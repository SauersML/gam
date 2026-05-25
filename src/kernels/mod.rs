//! Reusable numerical kernels exposed across the gam engine and its
//! Python bindings.
//!
//! These are stand-alone, well-tested primitives that do not depend on
//! the GAM solver state. They live here (rather than under
//! [`crate::geometry`] or [`crate::terms`]) because they are general
//! numerical building blocks — entropic optimal transport, Sinkhorn
//! barycenter, cost-matrix helpers — that other modules may need but
//! that are also useful to expose to Python end-users directly.

pub mod sinkhorn_barycenter;
