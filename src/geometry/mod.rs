pub mod circle;
pub mod closure_family;
pub mod constant_curvature;
pub mod curvature_estimand;
pub mod euclidean;
pub mod grassmann;
pub mod integrator;
pub mod latent_seed;
pub mod lie_so;
pub mod manifold;
pub mod optimizer;
pub mod poincare;
pub mod product;
pub mod response_geometry;
pub mod simplex;
pub mod sinkhorn_barycenter;
pub mod spd;
pub mod sphere;
pub mod stiefel;
pub mod torus;

pub use circle::CircleManifold;
pub use closure_family::{
    ClosureFamily, ClosureProfileCi, boundary_conductance, conductance_penalty_jet,
    profile_ci_from_grid,
};
pub use constant_curvature::{
    ConstantCurvature, distance_kappa_jet, exp_map_kappa_jet, log_map_kappa_jet,
};
pub use curvature_estimand::{
    CurvatureVerdict, DesignCoordKappaJet, FlatnessTest, KappaProfileCi,
    design_coord_kappa_derivative, flatness_lr_test, profile_ci_walk, wald_half_width,
};
pub use euclidean::EuclideanManifold;
pub use grassmann::GrassmannManifold;
pub use integrator::GeodesicIntegrator;
pub use latent_seed::laplacian_eigenmap_coords;
pub use manifold::{GeometryError, GeometryResult, ManifoldSpec, RiemannianManifold};
pub use optimizer::{RiemannianLBFGS, RiemannianObjective, RiemannianTrustRegion};
pub use product::ProductManifold;
pub use response_geometry::{
    ResponseCurvatureFit, ResponseManifold, fit_response_curvature, response_curvature_criterion,
    response_exp_map, response_frechet_mean, response_log_map,
};
pub use spd::{SpdManifold, spd_frechet_mean};
pub use sphere::SphereManifold;
pub use stiefel::StiefelManifold;
pub use torus::TorusManifold;

use ndarray::{Array1, ArrayView1};

/// Validate and normalize per-row weights for a manifold barycenter computation.
///
/// With `None`, returns uniform weights `1/n`. With `Some(w)`, requires `w.len() == n`
/// and every entry finite, non-negative, with positive total, then returns `w` divided
/// by its total so the weights sum to one.
pub(crate) fn normalize_weights(
    n: usize,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, String> {
    match weights {
        None => Ok(Array1::from_elem(n, 1.0 / n as f64)),
        Some(w) => {
            if w.len() != n {
                return Err("weights length must match the number of rows".to_string());
            }
            let mut total = 0.0_f64;
            for value in w.iter() {
                if !value.is_finite() || *value < 0.0 {
                    return Err(
                        "weights must be finite, non-negative, and have positive total".to_string(),
                    );
                }
                total += *value;
            }
            if total <= 0.0 {
                return Err(
                    "weights must be finite, non-negative, and have positive total".to_string(),
                );
            }
            Ok(w.mapv(|v| v / total))
        }
    }
}
