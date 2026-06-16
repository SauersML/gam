use gam::families::survival::assemble_competing_risks_cif_from_endpoints;
use gam::families::survival::construction::{
    SurvivalBaselineConfig, SurvivalBaselineTarget, SurvivalLikelihoodMode,
    build_survival_time_offsets_for_likelihood,
};
use ndarray::{Array2, array};

#[test]
fn baseline_cumulative_hazard_is_exactly_zero_at_time_zero() {
    let age = array![0.0];
    let cfg = SurvivalBaselineConfig {
        target: SurvivalBaselineTarget::Weibull,
        scale: Some(2.0),
        shape: Some(1.7),
        rate: None,
        makeham: None,
    };
    let (_entry, exit, _derivative) = build_survival_time_offsets_for_likelihood(
        &age,
        &age,
        &cfg,
        SurvivalLikelihoodMode::Transformation,
        None,
    )
    .expect("baseline offsets should evaluate at zero time");

    let cumulative = exit[0].exp();
    assert!(
        cumulative == 0.0,
        "Baseline cumulative hazard must be exactly zero at t=0, but got positive value {cumulative:e}"
    );
}

#[test]
fn competing_risks_cif_plus_survival_is_one_at_every_time() {
    let times = array![0.0, 0.2, 0.7, 1.4, 2.5];
    let h1 = array![0.0, 0.08, 0.22, 0.45, 0.9];
    let h2 = array![0.0, 0.04, 0.12, 0.31, 0.6];
    let endpoints = vec![
        h1.clone().insert_axis(ndarray::Axis(0)),
        h2.clone().insert_axis(ndarray::Axis(0)),
    ];

    let assembled = assemble_competing_risks_cif_from_endpoints(times.view(), &endpoints)
        .expect("CIF assembly should succeed for monotone cumulative hazards");

    for t in 0..times.len() {
        let fsum = assembled
            .cif
            .iter()
            .map(|cif: &Array2<f64>| cif[[0, t]])
            .sum::<f64>();
        let s = assembled.overall_survival[[0, t]];
        assert!(
            fsum + s == 1.0,
            "Competing-risks identity must hold exactly: F1+F2+...+S=1, but got {} at t index {}",
            fsum + s,
            t
        );
    }
}

#[test]
fn competing_risks_cif_is_bounded_between_zero_and_one() {
    let times = array![0.0, 0.5, 1.0, 2.0, 4.0];
    let h1 = array![0.0, 0.4, 1.1, 2.0, 4.2];
    let h2 = array![0.0, 0.3, 0.9, 1.7, 3.6];
    let endpoints = vec![
        h1.insert_axis(ndarray::Axis(0)),
        h2.insert_axis(ndarray::Axis(0)),
    ];

    let assembled = assemble_competing_risks_cif_from_endpoints(times.view(), &endpoints)
        .expect("CIF assembly should succeed for monotone cumulative hazards");

    for (k, cif) in assembled.cif.iter().enumerate() {
        for t in 0..times.len() {
            let v = cif[[0, t]];
            assert!(
                (0.0..=1.0).contains(&v),
                "Cause-specific CIF must stay in [0,1], but cause {} at t index {} was {}",
                k + 1,
                t,
                v
            );
        }
    }
}
