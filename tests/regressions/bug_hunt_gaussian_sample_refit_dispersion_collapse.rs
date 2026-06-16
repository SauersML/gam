//! Regression for issue #521: sampling a saved Gaussian identity model must
//! condition on the saved training fit, not refit/re-estimate dispersion on the
//! rows supplied to `sample_saved_model`.
//!
//! The first six rows below lie exactly on the linear plane. The full training
//! set has real residual noise. The historical standard sampling path rebuilt
//! `X,y` from the caller's data, refit REML/PIRLS on those six rows, estimated
//! φ≈0, and returned a coefficient posterior thousands of times too tight for
//! the very same saved model. A saved-model posterior is conditioned on the
//! training fit: passing fewer prediction rows cannot make the coefficient
//! posterior tighter.

use gam::hmc::NutsConfig;
use gam::inference::model::{
    FittedFamily, FittedModel, FittedModelPayload, MODEL_PAYLOAD_VERSION, ModelKind,
};
use gam::sample::sample_saved_model;
use gam::smooth::{build_term_collection_design, freeze_term_collection_from_design};
use gam::types::{LikelihoodSpec, StandardLink};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array2, s};
use std::io::Write;

fn gaussian_training_csv(n: usize) -> String {
    let mut csv = String::from("x1,x2,y\n");
    for i in 0..n {
        let t = i as f64;
        let x1 = (t / 17.0).sin() + 0.003 * t;
        let x2 = (t / 29.0).cos() - 0.002 * t;
        let noise = if i < 6 {
            0.0
        } else {
            0.08 * (t * 1.618_033_988_75).sin() + 0.03 * (t * 0.37).cos()
        };
        let y = 1.25 + 2.0 * x1 - 1.5 * x2 + noise;
        csv.push_str(&format!("{x1:.17e},{x2:.17e},{y:.17e}\n"));
    }
    csv
}

fn write_temp_csv(contents: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "gam_issue_521_gaussian_sample_{}.csv",
        std::process::id()
    ));
    let mut file = std::fs::File::create(&path).expect("create issue #521 synthetic csv");
    file.write_all(contents.as_bytes())
        .expect("write issue #521 synthetic csv");
    path
}

fn saved_standard_gaussian_model(
    formula: &str,
    ds: &gam::inference::data::EncodedDataset,
) -> (FittedModel, usize) {
    let cfg = FitConfig::default();
    let result = fit_from_formula(formula, ds, &cfg).expect("fit full Gaussian model");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian identity fit");
    };

    let design = build_term_collection_design(ds.values.view(), &fit.resolvedspec)
        .expect("rebuild full training design");
    let p = design.design.ncols();
    let frozenspec = freeze_term_collection_from_design(&fit.resolvedspec, &design)
        .expect("freeze training term collection");

    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        formula.to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::gaussian_identity(),
            link: Some(StandardLink::Identity),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "gaussian".to_string(),
    );
    payload.fit_result = Some(fit.fit.clone());
    payload.unified = Some(fit.fit);
    payload.data_schema = Some(ds.schema.clone());
    payload.resolved_termspec = Some(frozenspec);
    payload.set_training_feature_metadata(ds.headers.clone(), ds.feature_ranges());

    (FittedModel::from_payload(payload), p)
}

#[test]
fn gaussian_saved_model_subset_sampling_does_not_tighten_posterior() {
    init_parallelism();

    let csv = gaussian_training_csv(600);
    let csv_path = write_temp_csv(&csv);
    let ds = load_csvwith_inferred_schema(&csv_path).expect("load issue #521 synthetic csv");
    std::fs::remove_file(&csv_path).expect("remove issue #521 synthetic csv");
    let col = ds.column_map();

    let (model, p) = saved_standard_gaussian_model("y ~ x1 + x2", &ds);
    let cfg = NutsConfig {
        n_samples: 256,
        nwarmup: 64,
        n_chains: 2,
        seed: 521,
        ..NutsConfig::for_dimension(p)
    };

    let all_rows = sample_saved_model(
        &model,
        ds.values.view(),
        &col,
        model.training_headers.as_ref(),
        &cfg,
    )
    .expect("sample saved Gaussian model on all rows");
    let subset_values: Array2<f64> = ds.values.slice(s![0..6, ..]).to_owned();
    let subset_rows = sample_saved_model(
        &model,
        subset_values.view(),
        &col,
        model.training_headers.as_ref(),
        &cfg,
    )
    .expect("sample saved Gaussian model on six-row subset");

    assert!(
        all_rows.converged,
        "full-data saved-model sampler converged"
    );
    assert!(
        subset_rows.converged,
        "subset saved-model sampler converged"
    );
    assert_eq!(all_rows.posterior_std.len(), p);
    assert_eq!(subset_rows.posterior_std.len(), p);

    for j in 0..p {
        let full_std = all_rows.posterior_std[j];
        let subset_std = subset_rows.posterior_std[j];
        assert!(
            subset_std + 1e-12 >= full_std,
            "coefficient {j}: subset posterior std {subset_std:.6e} tightened below full-data std {full_std:.6e}"
        );
    }
}
