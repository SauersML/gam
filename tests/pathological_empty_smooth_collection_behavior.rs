use gam::estimate::{
    FitOptions, PenaltySpec, fit_gam, fit_gam_with_penalty_specs, fit_gamwith_heuristic_lambdas,
};
use gam::types::LikelihoodSpec;
use ndarray::array;

#[test]
fn empty_smooth_collection_is_handled_or_rejected_cleanly() {
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![1.0, 2.0, 3.0, 4.0];
    let w = array![1.0, 1.0, 1.0, 1.0];
    let offset = array![0.0, 0.0, 0.0, 0.0];
    let fam = LikelihoodSpec::gaussian_identity();
    let opts = FitOptions::default();

    let r1 = fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &[],
        fam.clone(),
        &opts,
    );
    let r2 = fit_gamwith_heuristic_lambdas(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &[],
        None,
        fam.clone(),
        &opts,
    );
    let r3 = fit_gam_with_penalty_specs(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        Vec::<PenaltySpec>::new(),
        vec![],
        fam,
        &opts,
    );
    for r in [r1, r2, r3] {
        if let Ok(fit) = r {
            assert_eq!(
                fit.lambdas.len(),
                0,
                "pure parametric fit should carry zero smooth lambdas"
            );
        }
    }
}
