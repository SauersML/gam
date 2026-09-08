//! #2714 repro, R-free: the latent hazard-multiplier frailty fit on the
//! Veterans' lung-cancer trial must CONVERGE.
//!
//! The issue's witness is
//! `quality_vs_survival_coxph_frailty_hazard_multiplier::gam_hazard_multiplier_frailty_matches_coxph_frailty_on_real_data`,
//! whose accuracy assertions are graded against R `survival::coxph`. But the
//! reported failure happens strictly BEFORE R is reached — the fit itself
//! returns
//!
//! ```text
//! custom-family inner solve did not converge after 28 cycle(s)
//! [joint-Newton terminal cycle 29: stationarity_residual=1.741513e-2
//!  (tol=3.585252e-10) ... rejects [model,likelihood,objective,feasibility]=[0,0,2,0]]
//! ```
//!
//! so the defect is reproducible with no reference tool installed, and grading
//! it needs no R. This file carries exactly that half: the same rows, the same
//! `FitConfig`, and the single assertion that the fit comes back at all.
//!
//! Keeping it separate from the quality fixture is deliberate. A convergence
//! defect that can only be observed through a test which also needs R, a
//! held-out concordance and a match-or-beat comparison is a defect most CI
//! lanes cannot see; this one fails on the mechanism and nothing else.

use csv::StringRecord;
use gam_data::{encode_recordswith_inferred_schema, load_csvwith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_models::survival::lognormal_kernel::{FrailtyScale, FrailtySpec, HazardLoading};
use std::path::Path;

const VETERAN_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench/datasets/veteran_lung.csv"
);

#[test]
fn latent_hazard_multiplier_frailty_converges_on_veteran_2714() {
    // The quality binary this fixture was split out of installs no logging
    // backend, so everything the solver reports about WHY it refused is
    // discarded before it reaches the failure message. Install one here: a
    // convergence defect whose only evidence is the terminal string costs a
    // full fit per question asked of it.
    gam_runtime::test_support::install_diagnostic_logger();
    super::initialize_cpu_fitting();

    let raw = load_csvwith_inferred_schema(Path::new(VETERAN_CSV)).expect("load veteran_lung.csv");
    let rcol = raw.column_map();
    let (r_time, r_status, r_karno, r_celltype) = (
        rcol["time"],
        rcol["status"],
        rcol["karno"],
        rcol["celltype"],
    );
    let celltype_levels = &raw.schema.columns[r_celltype].levels;
    let n = raw.values.nrows();

    let time: Vec<f64> = raw.values.column(r_time).to_vec();
    let status: Vec<f64> = raw.values.column(r_status).to_vec();
    let karno: Vec<f64> = raw.values.column(r_karno).to_vec();
    let celltype_label: Vec<String> = raw
        .values
        .column(r_celltype)
        .iter()
        .map(|&code| celltype_levels[code as usize].clone())
        .collect();

    // The witness's deterministic split: every 4th row held out, fit on the rest.
    let train_rows: Vec<usize> = (0..n).filter(|i| i % 4 != 0).collect();
    let headers = vec![
        "time".to_string(),
        "status".to_string(),
        "karno".to_string(),
        "celltype".to_string(),
    ];
    let train_records: Vec<StringRecord> = train_rows
        .iter()
        .map(|&i| {
            StringRecord::from(vec![
                time[i].to_string(),
                status[i].to_string(),
                karno[i].to_string(),
                celltype_label[i].clone(),
            ])
        })
        .collect();
    let train_ds = encode_recordswith_inferred_schema(headers, train_records)
        .expect("encode veteran train survival data");

    let cfg = FitConfig {
        survival_likelihood: Some("latent".to_string()),
        baseline_target: "weibull".to_string(),
        time_basis: "ispline".to_string(),
        frailty: FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Learned { initial_sigma: 0.5 },
            loading: HazardLoading::Full,
        },
        ..FitConfig::default()
    };

    let result =
        fit_from_formula("Surv(time, status) ~ karno", &train_ds, &cfg).unwrap_or_else(|error| {
            panic!(
                "#2714: the latent hazard-multiplier frailty fit must converge on the \
                 veteran training rows, and it returned an error instead:\n{error}\n\n\
                 The reported failure is an inner joint-Newton stall whose steps are \
                 rejected by the OBJECTIVE while the model and the likelihood accept \
                 them, i.e. the objective the accept test evaluates and the gradient \
                 the step is built from are not on one approximation surface."
            )
        });
    let FitResult::LatentSurvival(fit) = result else {
        panic!("expected a LatentSurvival fit result for survival_likelihood=latent");
    };

    // Whatever else it does, a converged frailty fit must report a finite,
    // strictly positive learned scale — a collapsed or saturated sigma would
    // make the convergence assertion above vacuous.
    assert!(
        fit.latent_sd.is_finite() && fit.latent_sd > 0.0,
        "#2714: the learned frailty scale must be finite and positive, got {}",
        fit.latent_sd
    );
    eprintln!(
        "[2714] veteran latent frailty fit converged: latent_sd={}",
        fit.latent_sd
    );
}
