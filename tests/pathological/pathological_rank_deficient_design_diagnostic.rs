use gam::estimate::{
    FitOptions, PenaltySpec, fit_gam, fit_gam_with_penalty_specs, fit_gamwith_heuristic_lambdas,
};
use gam::terms::smooth::BlockwisePenalty;
use gam::types::LikelihoodSpec;
use ndarray::array;

#[test]
fn rank_deficient_design_reports_diagnostic_or_fits() {
    let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
    let y = array![1.0, 2.0, 3.0, 4.0];
    let w = array![1.0, 1.0, 1.0, 1.0];
    let offset = array![0.0, 0.0, 0.0, 0.0];
    let s_list: Vec<BlockwisePenalty> = vec![];
    let fam = LikelihoodSpec::gaussian_identity();
    let opts = FitOptions::default();

    for out in [
        fit_gam(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            &s_list,
            fam.clone(),
            &opts,
        ),
        fit_gamwith_heuristic_lambdas(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            &s_list,
            None,
            fam.clone(),
            &opts,
        ),
        fit_gam_with_penalty_specs(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            Vec::<PenaltySpec>::new(),
            vec![],
            fam.clone(),
            &opts,
        ),
    ] {
        if let Err(e) = out {
            let m = e.to_string().to_lowercase();
            assert!(
                m.contains("rank")
                    || m.contains("singular")
                    || m.contains("collinear")
                    || m.contains("deficien"),
                "unexpected error message: {m}"
            );
        }
    }
}
