use gam::estimate::{
    FitOptions, PenaltySpec, fit_gam, fit_gam_with_penalty_specs, fit_gamwith_heuristic_lambdas,
};
use gam::terms::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array2, array};

fn opts() -> FitOptions {
    FitOptions::default()
}
fn fam() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

#[test]
fn fit_family_rejects_empty_rows_and_columns_without_panic() {
    let x = Array2::<f64>::zeros((0, 0));
    let y = array![];
    let w = array![];
    let offset = array![];
    let s_list: Vec<BlockwisePenalty> = vec![];

    assert!(
        fit_gam(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            &s_list,
            fam(),
            &opts()
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
            fam(),
            &opts()
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
            fam(),
            &opts()
        )
        .is_err()
    );
}
