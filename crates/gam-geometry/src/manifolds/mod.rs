//! Concrete Riemannian manifold implementations.
//!
//! Each submodule provides one manifold family (its embedding, projection,
//! exponential/logarithm maps, distance, and curvature). The core trait and
//! the [`ManifoldSpec`](crate::manifold::ManifoldSpec) builder live one level
//! up in [`crate::manifold`]; these are the structs it instantiates.
//!
//! For backwards compatibility every submodule and its primary public items are
//! re-exported at the crate root, so `gam_geometry::sphere::SphereManifold` and
//! `gam_geometry::SphereManifold` both continue to resolve.

pub mod circle;
pub mod constant_curvature;
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
    ConstantCurvature, distance_kappa_jet, exp_map_kappa_jet, log_map_kappa_jet,
};
pub use euclidean::EuclideanManifold;
pub use grassmann::GrassmannManifold;
pub use product::ProductManifold;
pub use spd::{SpdManifold, spd_frechet_mean};
pub use sphere::SphereManifold;
pub use stiefel::StiefelManifold;
pub use torus::TorusManifold;
