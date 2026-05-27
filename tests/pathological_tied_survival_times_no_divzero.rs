use gam::estimate::{
    FitOptions, PenaltySpec, fit_gam, fit_gam_with_penalty_specs, fit_gamwith_heuristic_lambdas,
};
use gam::terms::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::array;

#[test]
fn tied_survival_times_do_not_trigger_division_by_zero() {
    let x = array![[1.0], [1.0], [1.0], [1.0]];
    let y = array![5.0, 5.0, 5.0, 5.0];
    let w = array![1.0, 1.0, 1.0, 1.0];
    let offset = array![0.0, 0.0, 0.0, 0.0];
    let s_list: Vec<BlockwisePenalty> = vec![];
    let fam = LikelihoodSpec::new(
        ResponseFamily::RoystonParmar,
        InverseLink::Standard(StandardLink::Identity),
    );
    let opts = FitOptions::default();

    for r in [
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
        if let Err(e) = r {
            let msg = e.to_string().to_lowercase();
            assert!(!msg.contains("division by zero") && !msg.contains("divide by zero"));
        }
    }
}
