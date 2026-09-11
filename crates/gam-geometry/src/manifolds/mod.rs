//! Concrete Riemannian manifold implementations.
//!
//! Each submodule provides one manifold family (its embedding, projection,
//! exponential/logarithm maps, distance, and curvature). The core trait and
//! the [`ManifoldSpec`](crate::manifold::ManifoldSpec) builder live one level
//! up in [`crate::manifold`]; these are the structs it instantiates.
//!
//! Primary manifold types are re-exported at the crate root so callers use one
//! canonical import surface.

pub mod aitchison_ilr;
pub mod circle;
/// Axioms every [`RiemannianManifold`](crate::manifold::RiemannianManifold)
/// implementation must satisfy, checked against one shared inventory keyed off
/// [`ManifoldSpec`](crate::manifold::ManifoldSpec).
#[cfg(test)]
mod conformance_tests;
pub mod constant_curvature;
/// gam#2687: what the κ box's half-margin to the antipodal fold buys, measured
/// against a cancellation-free closed form rather than asserted.
#[cfg(test)]
mod constant_curvature_antipodal_resolution_tests;
/// Independent-oracle checks for the κ-stereographic family, which carries a
/// continuous parameter and so is not a
/// [`ManifoldSpec`](crate::manifold::ManifoldSpec) variant.
#[cfg(test)]
mod constant_curvature_conformance_tests;
pub mod euclidean;
pub mod grassmann;
pub mod lie_so;
pub mod poincare;
pub mod product;
pub mod simplex;
pub mod spd;
pub mod sphere;
pub mod stiefel;
pub mod torus;

pub use circle::CircleManifold;
pub use constant_curvature::{
    ConstantCurvature, constant_curvature_dirichlet_penalty,
    constant_curvature_dirichlet_penalty_kappa_derivative, distance_kappa_jet, exp_map_kappa_jet,
    log_map_kappa_jet,
};
pub use euclidean::EuclideanManifold;
pub use grassmann::GrassmannManifold;
pub use product::ProductManifold;
pub use spd::{SpdManifold, spd_frechet_mean};
pub use sphere::SphereManifold;
pub use stiefel::StiefelManifold;
pub use torus::TorusManifold;
