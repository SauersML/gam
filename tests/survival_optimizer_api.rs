use gam::estimate::{FitOptions, fit_gam};
use gam::types::LikelihoodFamily;
use ndarray::array;

#[test]
fn fit_gam_rejects_royston_parmar_and_points_to_survival_api() {
    let x = array![[1.0], [1.0]];
    let y = array![0.0, 1.0];
    let w = array![1.0, 1.0];
    let offset = array![0.0, 0.0];
    let s_list = Vec::new();
    let opts = FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        max_iter: 60,
        tol: 1e-6,
        nullspace_dims: Vec::new(),
        linear_constraints: None,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };

    let err: gam::estimate::EstimationError = match fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s_list,
        LikelihoodFamily::RoystonParmar,
        &opts,
    ) {
        Ok(_) => panic!("RoystonParmar should be rejected by fit_gam external-design path"),
        Err(err) => err,
    };

    let msg = err.to_string();
    assert!(
        msg.contains("fit_gam external design path does not support RoystonParmar"),
        "unexpected error message: {msg}"
    );
    assert!(
        msg.contains("use survival training APIs"),
        "error should direct callers to survival-specific APIs: {msg}"
    );
}
