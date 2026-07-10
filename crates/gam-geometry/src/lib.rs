pub mod closure_family;
pub mod curvature_estimand;
pub mod integrator;
pub mod latent_seed;
pub mod manifold;
pub mod manifolds;
pub mod optimizer;
pub mod response_geometry;
pub mod sae_routing;
pub mod sinkhorn_barycenter;

// Re-export each manifold submodule at the crate root so the historical paths
// (`gam_geometry::sphere::SphereManifold`, …) keep resolving after the
// `manifolds/` regrouping.
pub use manifolds::{
    circle, constant_curvature, euclidean, grassmann, lie_so, poincare, product, simplex, spd,
    sphere, stiefel, torus,
};

pub use closure_family::{
    ClosureFamily, ClosureProfileCi, boundary_conductance, conductance_penalty_jet,
    profile_ci_from_grid,
};
pub use curvature_estimand::{
    CurvatureVerdict, DesignCoordKappaJet, FlatnessTest, KappaProfileCi,
    design_coord_kappa_derivative, flatness_lr_test, profile_ci_walk, wald_half_width,
};
pub use integrator::GeodesicIntegrator;
pub use latent_seed::laplacian_eigenmap_coords;
pub use manifold::{GeometryError, GeometryResult, ManifoldSpec, RiemannianManifold};
pub use manifolds::{
    CircleManifold, ConstantCurvature, EuclideanManifold, GrassmannManifold, ProductManifold,
    SpdManifold, SphereManifold, StiefelManifold, TorusManifold, distance_kappa_jet,
    exp_map_kappa_jet, log_map_kappa_jet, spd_frechet_mean,
};
pub use optimizer::{RiemannianLBFGS, RiemannianObjective, RiemannianTrustRegion};
pub use response_geometry::{
    ResponseCurvatureFit, ResponseGeometryError, ResponseManifold, fit_response_curvature,
    response_curvature_criterion, response_exp_map, response_frechet_mean, response_log_map,
    response_projection_residual,
};

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
