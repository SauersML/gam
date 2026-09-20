#![cfg(test)]
//! #3001: a Gaussian location-scale fit with a link wiggle publishes its fitted mean as
//! two block states, `μ = η_μ + η_wiggle`, and its σ as `σ = s·b + exp(η_σ)` with the
//! fit's own σ floor `b`. The saved model evaluates each of them by one function, so on
//! its own fit's design it must predict exactly that μ and σ.
//!
//! The fixture is gnomon's 48-row calibration table with two principal components
//! (`calibrate::estimate::tests::gaussian_fixture`). Its request is built the way gnomon
//! builds it: the same Duchon score smooth, linear sex term and joint PC smooth in both
//! channels, and the default link wiggle added to the request (the formula grammar keeps
//! `linkwiggle()` for binomial means). The fit is saved to disk and loaded, and the
//! loaded model predicts from the fit's own mean and noise designs. Every channel's
//! agreement is printed before anything is asserted: the location η, the wiggle's
//! share, μ and σ.
//!
//! The same model then predicts from the design its frozen spec REBUILDS on these rows,
//! which must be the fit's design bit for bit: the replay applies the term-local chart,
//! the joint-null rotation `Q` and the collection chart `T` as the three products the
//! fit applied, in the fit's order, and subtracts the row-space correction through the
//! one function the fit forms it with. Composing `Q·T` into one frozen chart moved μ by
//! up to 2 ulp in 11 of the 48 rows, which is the other half of #3001.

use crate::FittedModelPredictExt;
use crate::test_support::init_parallelism;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::{
    FitConfig, FitRequest, FitResult, LinkWiggleConfig, fit_model, materialize,
};
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::payload_for_gaussian_location_scale;
use gam_models::inference::predict_input::build_predict_input_for_model;
use gam_models::inference::predict_io::PredictInput;
use gam_problem::BlockRole;
use gam_terms::smooth::{build_term_collection_design, freeze_term_collection_from_design};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

const ROWS: usize = 48;

/// Both channels' terms, as gnomon's `score_smooth` and `context_formula` write them.
const TERMS: &str =
    "s(score, bs=duchon, centers=4) + sex + s(PC1, PC2, bs=duchon, centers=4)";

fn gnomon_gaussian_table() -> gam_data::EncodedDataset {
    let headers = ["y", "score", "sex", "PC1", "PC2"].map(String::from).to_vec();
    let records = (0..ROWS)
        .map(|index| {
            let score = (index as f64 - 24.0) / 12.0;
            let sex = (index % 2) as f64;
            let pc = |component: usize| ((index * (3 + 2 * component)) as f64 * 0.37).sin();
            let y = 2.0 + 0.7 * score + 0.3 * sex + 0.4 * pc(0) * pc(1)
                + 0.2 * (index as f64 * 1.7).sin();
            csv::StringRecord::from(vec![
                format!("{y:.17e}"),
                format!("{score:.17e}"),
                format!("{sex}"),
                format!("{:.17e}", pc(0)),
                format!("{:.17e}", pc(1)),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode the 48-row table")
}

/// How far `predicted` lies from `fitted`: the rows that differ in any bit, and the
/// largest gap, absolute and in units of the fitted value's ulp.
fn agreement(label: &str, predicted: &Array1<f64>, fitted: &Array1<f64>) -> usize {
    assert_eq!(predicted.len(), fitted.len(), "{label}: row counts differ");
    let mut differing = 0usize;
    let mut largest = (0.0f64, 0.0f64, 0usize);
    for (row, (&p, &f)) in predicted.iter().zip(fitted.iter()).enumerate() {
        if p.to_bits() != f.to_bits() {
            differing += 1;
        }
        let gap = (p - f).abs();
        let ulp = f64::from_bits(f.abs().to_bits() + 1) - f.abs();
        if gap > largest.0 {
            largest = (gap, gap / ulp, row);
        }
    }
    println!(
        "[3001] {label}: {differing}/{} rows differ; largest gap {:.3e} = {:.2} ulp at row {}",
        predicted.len(),
        largest.0,
        largest.1,
        largest.2
    );
    differing
}

/// The entries of `rebuilt` whose bits differ from `fitted`'s.
fn entries_differing(label: &str, rebuilt: &Array2<f64>, fitted: &Array2<f64>) -> usize {
    assert_eq!(rebuilt.dim(), fitted.dim(), "{label}: design shapes differ");
    let differing = rebuilt
        .iter()
        .zip(fitted.iter())
        .filter(|(r, f)| r.to_bits() != f.to_bits())
        .count();
    println!(
        "[3001] {label}: {differing}/{} design entries differ",
        fitted.len()
    );
    differing
}

/// The design half of #3001 without the fit: each channel's term collection is built on
/// the 48 rows, frozen, and rebuilt from the frozen spec on the same rows, and the rebuild
/// must be the built design bit for bit. The control rebuilds from the chart the freeze
/// used to write, `Q·T` composed into one product, and must NOT be, or this test could not
/// tell the two replays apart.
#[test]
fn the_frozen_term_collection_rebuilds_its_design_bit_for_bit_3001() {
    init_parallelism();
    let data = gnomon_gaussian_table();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some(TERMS.to_string()),
        ..FitConfig::default()
    };
    let materialized =
        materialize(&format!("y ~ {TERMS}"), &data, &config).expect("materialize the request");
    let FitRequest::GaussianLocationScale(request) = materialized.request else {
        panic!("expected a Gaussian location-scale request");
    };
    let mut composed_differing = 0usize;
    for (channel, spec) in [
        ("location", &request.spec.meanspec),
        ("log-σ", &request.spec.log_sigmaspec),
    ] {
        let built = build_term_collection_design(request.data, spec).expect("build the design");
        assert!(
            built.smooth.terms.iter().any(|term| term
                .collection_gauge
                .as_ref()
                .is_some_and(|gauge| gauge.joint_null_rotation.is_some())),
            "premise: a {channel} smooth takes a collection gauge on a joint-null rotation"
        );
        let frozen = freeze_term_collection_from_design(spec, &built).expect("freeze the spec");
        let rebuilt =
            build_term_collection_design(request.data, &frozen).expect("rebuild the design");
        let fitted_dense = built.design.to_dense();
        let differing = entries_differing(
            &format!("{channel} (rebuilt design)"),
            &rebuilt.design.to_dense(),
            &fitted_dense,
        );
        assert_eq!(
            differing, 0,
            "the {channel} design the frozen spec rebuilds must be the built design, bit for bit"
        );

        let mut composed = frozen.clone();
        for term in &mut composed.smooth_terms {
            if let (Some(rotation), Some(chart)) = (
                term.joint_null_rotation.take(),
                term.frozen_parametric_residualization.as_mut(),
            ) {
                chart.coefficient_transform = rotation.rotation.dot(&chart.coefficient_transform);
            }
        }
        let composed_rebuilt =
            build_term_collection_design(request.data, &composed).expect("rebuild composed");
        composed_differing += entries_differing(
            &format!("{channel} (composed Q·T control)"),
            &composed_rebuilt.design.to_dense(),
            &fitted_dense,
        );
    }
    assert!(
        composed_differing > 0,
        "the control replay `B·(Q·T)` must differ from the fit's `(B·Q)·T` somewhere, or \
         this test cannot tell the two apart"
    );
}

#[test]
fn a_saved_gaussian_location_scale_wiggle_model_predicts_its_own_fitted_mean_and_scale_3001() {
    init_parallelism();
    let data = gnomon_gaussian_table();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some(TERMS.to_string()),
        ..FitConfig::default()
    };
    let materialized =
        materialize(&format!("y ~ {TERMS}"), &data, &config).expect("materialize the request");
    let FitRequest::GaussianLocationScale(mut request) = materialized.request else {
        panic!("expected a Gaussian location-scale request");
    };
    let wiggle = gam_spec::WigglePenaltyConfig::cubic_triple_operator_default();
    request.wiggle = Some(LinkWiggleConfig {
        degree: wiggle.degree,
        num_internal_knots: wiggle.num_internal_knots,
        penalty_orders: wiggle.penalty_orders,
        double_penalty: wiggle.double_penalty,
    });
    let FitResult::GaussianLocationScale(fitted) =
        fit_model(FitRequest::GaussianLocationScale(request)).expect("the 48-row table fits")
    else {
        panic!("expected a Gaussian location-scale fit");
    };
    assert!(
        fitted.wiggle_knots.is_some(),
        "the request's link wiggle must be fitted"
    );
    let formula = format!("y ~ {TERMS} + linkwiggle()");
    let states = fitted.fit.fit.block_states.clone();
    assert_eq!(states.len(), 3, "location, log-σ and link-wiggle block states");
    let scaled_floor = fitted.response_scale * fitted.sigma_floor;
    let fitted_mean = &states[0].eta + &states[2].eta;
    let fitted_sigma = states[1]
        .eta
        .mapv(|eta| gam_model_kernels::sigma_link::logb_sigma_from_eta_scalar(scaled_floor, eta));
    let zero = Array1::<f64>::zeros(ROWS);
    // The fit's own designs, the ones its block states were published from.
    let fit_input = PredictInput {
        design: fitted.fit.mean_design.design.clone(),
        offset: zero.clone(),
        design_noise: Some(fitted.fit.noise_design.design.clone()),
        offset_noise: Some(zero.clone()),
        auxiliary_scalar: None,
        auxiliary_matrix: None,
    };

    let payload = payload_for_gaussian_location_scale(formula, &data, &config, fitted)
        .expect("assemble the saved model");
    let path = std::env::temp_dir().join(format!("gam-3001-{}.json", std::process::id()));
    FittedModel::from_payload(payload)
        .save_to_path(&path)
        .expect("save the model");
    let model = FittedModel::load_from_path(&path).expect("load the model");
    std::fs::remove_file(&path).expect("remove the saved model");
    let predictor = model
        .predictor()
        .expect("a saved Gaussian location-scale model predicts");
    let saved = model.fit_result.as_ref().expect("the saved fit");
    let beta_location = &saved
        .block_by_role(BlockRole::Location)
        .expect("a location block")
        .beta;
    let beta_noise = &saved
        .block_by_role(BlockRole::Scale)
        .expect("a scale block")
        .beta;

    let predicted_location = fit_input.design.dot(beta_location) + &fit_input.offset;
    agreement("location η (fit design)", &predicted_location, &states[0].eta);
    let predicted_noise = fit_input
        .design_noise
        .as_ref()
        .expect("a noise design")
        .dot(beta_noise)
        + fit_input.offset_noise.as_ref().expect("a noise offset");
    agreement("log-σ η (fit design)", &predicted_noise, &states[1].eta);
    let mean = predictor
        .predict_plugin_response(&fit_input)
        .expect("predict μ")
        .mean;
    let sigma = predictor
        .predict_noise_scale(&fit_input)
        .expect("predict σ")
        .expect("a location-scale model has a scale");
    let wiggle_share = model
        .saved_link_wiggle()
        .expect("read the saved link wiggle")
        .expect("the saved model carries its link wiggle")
        .contribution(&predicted_location)
        .expect("the saved link wiggle evaluates on the training rows");
    agreement("wiggle share (fit design)", &wiggle_share, &states[2].eta);
    let mean_differing = agreement("μ (fit design)", &mean, &fitted_mean);
    let sigma_differing = agreement("σ (fit design)", &sigma, &fitted_sigma);

    // The other half of #3001: the design the frozen spec rebuilds.
    let col_map: HashMap<String, usize> = data
        .headers
        .iter()
        .enumerate()
        .map(|(column, header)| (header.clone(), column))
        .collect();
    let rebuilt_input = build_predict_input_for_model(
        &model,
        data.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &zero,
        &zero,
        false,
    )
    .expect("the training rows' predict input");
    let rebuilt_mean = predictor
        .predict_plugin_response(&rebuilt_input)
        .expect("predict μ on the rebuilt design")
        .mean;
    let rebuilt_sigma = predictor
        .predict_noise_scale(&rebuilt_input)
        .expect("predict σ on the rebuilt design")
        .expect("a location-scale model has a scale");
    let rebuilt_mean_differing = agreement("μ (rebuilt design)", &rebuilt_mean, &fitted_mean);
    let rebuilt_sigma_differing =
        agreement("σ (rebuilt design)", &rebuilt_sigma, &fitted_sigma);

    assert_eq!(
        mean_differing, 0,
        "on its fit's own design the saved model's μ must be the fit's own μ, bit for bit"
    );
    assert_eq!(
        sigma_differing, 0,
        "on its fit's own design the saved model's σ must be the fit's own σ, bit for bit"
    );
    assert_eq!(
        rebuilt_mean_differing, 0,
        "on the design its frozen spec rebuilds at the training rows the saved model's μ \
         must be the fit's own μ, bit for bit"
    );
    assert_eq!(
        rebuilt_sigma_differing, 0,
        "on the design its frozen spec rebuilds at the training rows the saved model's σ \
         must be the fit's own σ, bit for bit"
    );
}
