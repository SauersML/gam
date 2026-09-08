use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_models::inference::model::{FittedModel, FittedModelPayload};
use gam_models::inference::model_payload_builders::{
    StandardPayloadInputs, assemble_standard_payload,
};
use gam_predict::input::build_predict_input_for_model;
use gam_predict::{FittedModelPredictExt, PosteriorMeanOptions};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

fn fixture() -> EncodedDataset {
    let headers = ["y", "x0", "x1", "x2", "x3", "x4"]
        .into_iter()
        .map(String::from)
        .collect();
    let records = (0..256)
        .map(|row| {
            let t = -2.75 + 5.5 * row as f64 / 255.0;
            let x = [
                t,
                (1.3 * t).sin(),
                (0.7 * t).cos(),
                t * t - 2.5,
                (2.1 * t + 0.2).sin(),
            ];
            let eta = 0.15 + 0.65 * x[0] - 0.45 * x[1] + 0.30 * x[2] - 0.08 * x[3] + 0.22 * x[4];
            let probability = 1.0 / (1.0 + (-eta - 0.35 * eta.tanh()).exp());
            let uniform = ((row + 1) as f64 * 0.618_033_988_749_894_9).fract();
            let mut record = vec![usize::from(uniform < probability).to_string()];
            record.extend(x.map(|value| value.to_string()));
            record.into()
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode flexible binomial fixture")
}

fn check_saved_response_scale(link: &str) {
    let dataset = fixture();
    let formula = format!("y ~ x0 + x1 + x2 + x3 + x4 + link(type=flexible({link}))");
    let config = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(result) = fit_from_formula(&formula, &dataset, &config)
        .expect("fit identifiable flexible binomial model")
    else {
        panic!("flexible binomial must produce a standard fit");
    };
    assert_eq!(result.design.design.ncols(), 6);
    assert_eq!(
        result
            .fit
            .geometry
            .as_ref()
            .expect("joint coefficient geometry")
            .coefficient_gauge
            .raw_total(),
        17,
        "exercise the six mean plus eleven raw warp coefficients",
    );
    let payload = assemble_standard_payload(StandardPayloadInputs {
        formula,
        dataset: &dataset,
        fit_config: &config,
        result,
    })
    .expect("assemble flexible binomial payload");
    let model = FittedModel::from_payload(payload.clone());
    let encoded = serde_json::to_vec(&payload).expect("serialize fitted payload");
    let decoded: FittedModelPayload =
        serde_json::from_slice(&encoded).expect("deserialize fitted payload");
    let reloaded = FittedModel::from_payload(decoded);

    // Midpoints are held-out rows. Rebuild their design from the saved model,
    // so these assertions exercise the deployed response map and warp layout.
    let held_out = Array2::from_shape_fn((255, 5), |(row, column)| {
        0.5 * (dataset.values[[row, column + 1]] + dataset.values[[row + 1, column + 1]])
    });
    let columns: HashMap<String, usize> = (0..5).map(|j| (format!("x{j}"), j)).collect();
    let offset = Array1::zeros(held_out.nrows());
    let predict = |saved: &FittedModel| {
        let input = build_predict_input_for_model(
            saved,
            held_out.view(),
            &columns,
            Some(&dataset.headers),
            &offset,
            &offset,
            false,
        )
        .expect("rebuild held-out design from saved model");
        let predictor = saved
            .predictor()
            .expect("construct saved binomial predictor");
        let plugin = predictor
            .predict_plugin_response(&input)
            .expect("plugin prediction");
        let posterior = predictor
            .predict_posterior_mean(
                &input,
                saved
                    .payload()
                    .fit_result
                    .as_ref()
                    .expect("saved canonical fit"),
                &PosteriorMeanOptions::point_only(),
            )
            .expect("posterior mean prediction");
        for mean in [&plugin.mean, &posterior.mean] {
            assert!(
                mean.iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            );
            assert!(
                mean.iter()
                    .zip(plugin.eta.iter())
                    .any(|(mu, eta)| (mu - eta).abs() > 0.1),
                "{link}: response predictions must differ from the linear predictor"
            );
        }
        (plugin.eta, plugin.mean, posterior.mean)
    };
    let before = predict(&model);
    let after = predict(&reloaded);
    assert_eq!(
        before, after,
        "saved response map must round-trip without numerical drift"
    );
}

#[test]
fn flexible_logit_saved_model_predicts_probabilities_2748() {
    check_saved_response_scale("logit");
}

#[test]
fn flexible_probit_saved_model_predicts_probabilities_2748() {
    check_saved_response_scale("probit");
}
