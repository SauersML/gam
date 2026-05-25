use gam::estimate::{
    FitOptions, PenaltySpec, fit_gam, fit_gam_with_penalty_specs, fit_gamwith_heuristic_lambdas,
};
use gam::terms::smooth::BlockwisePenalty;
use gam::types::LikelihoodSpec;
use ndarray::array;

#[test]
fn single_row_input_errors_gracefully() {
    let x = array![[1.0, 2.0]];
    let y = array![1.0];
    let w = array![1.0];
    let offset = array![0.0];
    let s_list: Vec<BlockwisePenalty> = vec![];
    let fam = LikelihoodSpec::gaussian_identity();
    let opts = FitOptions::default();

    assert!(
        fit_gam(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            &s_list,
            fam.clone(),
            &opts
        )
        .is_err()
    );
    assert!(
        fit_gamwith_heuristic_lambdas(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            &s_list,
            None,
            fam.clone(),
            &opts
        )
        .is_err()
    );
    assert!(
        fit_gam_with_penalty_specs(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            Vec::<PenaltySpec>::new(),
            vec![],
            fam,
            &opts
        )
        .is_err()
    );
}
