use gam::estimate::{FitOptions, fit_gamwith_heuristic_lambdas};
use gam::terms::smooth::BlockwisePenalty;
use gam::types::LikelihoodSpec;
use ndarray::array;

#[test]
fn extreme_rho_bounds_keep_reml_finite() {
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![1.1, 1.9, 3.1, 4.2, 5.1];
    let w = array![1.0, 1.0, 1.0, 1.0, 1.0];
    let offset = array![0.0, 0.0, 0.0, 0.0, 0.0];
    let s_list: Vec<BlockwisePenalty> = vec![];
    let fam = LikelihoodSpec::gaussian_identity();
    let opts = FitOptions::default();

    for rho in [-20.0_f64, 20.0_f64] {
        let lambdas = [rho.exp()];
        let fit = fit_gamwith_heuristic_lambdas(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            &s_list,
            Some(&lambdas),
            fam.clone(),
            &opts,
        )
        .expect("fit should complete at extreme rho");
        assert!(
            fit.reml_score.is_finite(),
            "REML score must be finite at rho={rho}"
        );
    }
}
