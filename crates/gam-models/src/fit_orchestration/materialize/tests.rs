use super::*;
use gam_data::load_dataset_projected;
use gam_data::{ColumnKindTag, DataSchema, SchemaColumn};
use gam_terms::basis::{
    DuchonNullspaceOrder, center_strategy_is_auto, minimum_duchon_power_for_operator_penalties,
    starting_num_centers,
};
use gam_terms::inference::formula_dsl::{
    default_linkwiggle_formulaspec, parse_linkwiggle_formulaspec,
};
use gam_terms::smooth::SmoothBasisSpec;
use ndarray::Array2;
use std::fs;
use tempfile::tempdir;

fn load_survival_dataset() -> gam_data::EncodedDataset {
    let td = tempdir().expect("tempdir");
    let data_path = td.path().join("survival.csv");
    fs::write(
        &data_path,
        "entry,exit,event,x,z\n0.0,1.0,1,0.2,-0.4\n0.3,1.6,0,-0.1,0.6\n",
    )
    .expect("write survival csv");
    load_dataset_projected(
        &data_path,
        &[
            "entry".to_string(),
            "exit".to_string(),
            "event".to_string(),
            "x".to_string(),
            "z".to_string(),
        ],
    )
    .expect("load survival dataset")
}

#[test]
fn competing_risks_baseline_seed_replicates_to_match_cause_specific_beta_length() {
    // Regression for #378's downstream break: the cause-specific assembly in
    // `fit_cause_specific_survival_transformation_custom` requires exactly
    // `p * cause_count` initial coefficients (it slices `cause * p..(cause +
    // 1) * p` per cause). The pooled baseline working model returns a
    // length-`p` seed, so without per-cause replication every `cause_count >
    // 1` fit aborts with a `SchemaMismatch` length mismatch. This pins that
    // the replication helper produces the exact length the assembly checks
    // for, and seeds each cause from the same pooled baseline.
    let pooled = Array1::from_vec(vec![-1.5_f64, 0.8, 0.0]);
    let p = pooled.len();

    for cause_count in [1usize, 2, 3] {
        let flat = replicate_pooled_baseline_seed_per_cause(pooled.view(), cause_count);
        // The exact invariant the cause-specific length guard enforces.
        assert_eq!(
            flat.len(),
            p * cause_count,
            "replicated seed must satisfy the `p * cause_count` length contract"
        );
        // Every per-cause slice must equal the shared pooled baseline seed.
        for cause in 0..cause_count {
            let slice = flat.slice(s![cause * p..(cause + 1) * p]);
            assert_eq!(
                slice.to_owned(),
                pooled,
                "cause {cause} block must be seeded from the pooled baseline"
            );
        }
    }
}

#[test]
fn survival_marginal_slope_materialize_rejects_z_column_in_main_formula() {
    let data = load_survival_dataset();
    let mut config = FitConfig::default();
    config.survival_likelihood = Some("marginal-slope".to_string());
    config.logslope_formula = Some("1".to_string());
    config.z_column = Some("z".to_string());

    let err = materialize("Surv(entry, exit, event) ~ x + z", &data, &config)
        .err()
        .expect("main formula should reject z-column reuse");

    assert!(
        err.to_string()
            .contains("survival marginal-slope reserves z column 'z'")
    );
    assert!(err.to_string().contains("main formula"));
}

#[test]
fn survival_marginal_slope_materialize_rejects_z_column_in_logslope_formula() {
    let data = load_survival_dataset();
    let mut config = FitConfig::default();
    config.survival_likelihood = Some("marginal-slope".to_string());
    config.logslope_formula = Some("1 + z".to_string());
    config.z_column = Some("z".to_string());

    let err = materialize("Surv(entry, exit, event) ~ x", &data, &config)
        .err()
        .expect("logslope formula should reject z-column reuse");

    assert!(
        err.to_string()
            .contains("survival marginal-slope reserves z column 'z'")
    );
    assert!(err.to_string().contains("logslope_formula"));
}

#[test]
fn survival_marginal_slope_materialize_rejects_z_column_when_logslope_defaults_to_main_spec() {
    let data = load_survival_dataset();
    let mut config = FitConfig::default();
    config.survival_likelihood = Some("marginal-slope".to_string());
    config.z_column = Some("z".to_string());

    let err = materialize("Surv(entry, exit, event) ~ x + z", &data, &config)
        .err()
        .expect("defaulted logslope spec should still reject z-column reuse");

    assert!(
        err.to_string()
            .contains("survival marginal-slope reserves z column 'z'")
    );
    assert!(err.to_string().contains("main formula"));
}

/// Regression for #1790: a left-truncated `Surv(entry, exit, event)` fit under
/// the DEFAULT `transformation` (Royston-Parmar) likelihood must center its
/// baseline time basis at the robust interior median-exit anchor, NOT the
/// earliest entry age.
///
/// Anchoring at the earliest entry under genuine left truncation
/// (`entry > 0`) leaves the centered baseline linear-trend column
/// `X(exit) − X(anchor)` large and one-signed across all rows — the unpenalized
/// polynomial null space of the time penalty — which inflates the time-block
/// seed score by orders of magnitude and rails the transformation-survival
/// smoothing selection into a degenerate, covariate-flat baseline (predicted
/// cumulative hazard ~10³× too large, survival collapsing to 0, covariate
/// dependence erased). The marginal-slope path already anchors at the median
/// exit for exactly this reason (#751); the fix extends that robust anchor to
/// every time-basis-carrying likelihood whenever the data is left-truncated.
///
/// This asserts the resolved `time_anchor` equals the median exit (a robust
/// interior time) rather than the earliest entry age. Before the fix it was the
/// earliest entry (0.5); after, the median exit (3.0).
#[test]
fn survival_transformation_left_truncated_uses_median_exit_anchor() {
    let td = tempdir().expect("tempdir");
    let data_path = td.path().join("left_truncated.csv");
    // Constant entry = 0.5 (genuine left truncation), five spread exits with an
    // odd count so the median exit is exactly the middle value 3.0.
    fs::write(
        &data_path,
        "entry,exit,event,x\n\
         0.5,1.0,1,-0.8\n\
         0.5,2.0,0,0.4\n\
         0.5,3.0,1,-0.2\n\
         0.5,4.0,1,0.7\n\
         0.5,5.0,0,0.1\n",
    )
    .expect("write left-truncated csv");
    let data = load_dataset_projected(
        &data_path,
        &[
            "entry".to_string(),
            "exit".to_string(),
            "event".to_string(),
            "x".to_string(),
        ],
    )
    .expect("load left-truncated dataset");

    // `FitConfig::default()` leaves `survival_likelihood` unset (`None`), and
    // every frontend resolves the one canonical default `"transformation"`
    // (Royston-Parmar) at the `resolved_survival_likelihood` seam (#2301).
    // Request it explicitly here so the test pins the mode independently of
    // that seam.
    let mut config = FitConfig::default();
    config.survival_likelihood = Some("transformation".to_string());

    let materialized = materialize("Surv(entry, exit, event) ~ x", &data, &config)
        .expect("left-truncated transformation survival should materialize");
    let FitRequest::SurvivalTransformation(request) = materialized.request else {
        panic!("expected a survival transformation request under the default likelihood");
    };

    let anchor = request.spec.time_anchor;
    // Median of {1,2,3,4,5} exits.
    let median_exit = 3.0_f64;
    let earliest_entry = 0.5_f64;
    assert!(
        (anchor - median_exit).abs() < 1e-9,
        "left-truncated transformation fit must center at the robust median-exit \
         anchor ({median_exit}), got {anchor}; the earliest-entry anchor \
         ({earliest_entry}) is the #1790 defect that rails the smoothing selection"
    );
    assert!(
        (anchor - earliest_entry).abs() > 1e-6,
        "anchor must not fall back to the earliest entry age under left truncation"
    );
}

#[test]
fn survival_marginal_slope_matern_logslope_penalties_keep_surface_width() {
    let n = 24usize;
    let mut values = Array2::<f64>::zeros((n, 8));
    for i in 0..n {
        let u = i as f64 / (n - 1) as f64;
        values[[i, 0]] = 0.0;
        values[[i, 1]] = 0.25 + 8.0 * u;
        values[[i, 2]] = if i % 3 == 0 { 1.0 } else { 0.0 };
        values[[i, 3]] = ((i * 17 % 23) as f64 - 11.0) / 7.0;
        values[[i, 4]] = (2.0 * std::f64::consts::PI * u).sin();
        values[[i, 5]] = (2.0 * std::f64::consts::PI * u).cos();
        values[[i, 6]] = 2.0 * u - 1.0;
        values[[i, 7]] = if i % 2 == 0 { 0.0 } else { 1.0 };
    }
    let data = Dataset {
        headers: vec![
            "t0".to_string(),
            "t1".to_string(),
            "event".to_string(),
            "z".to_string(),
            "PC1".to_string(),
            "PC2".to_string(),
            "PC3".to_string(),
            "sex".to_string(),
        ],
        values,
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "t0".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "t1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "sex".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Binary,
        ],
    };
    for (case, formula) in [
        (
            "with parametric sex term",
            "Surv(t0, t1, event) ~ matern(PC1, PC2, PC3, centers=6) + sex",
        ),
        (
            "without parametric sex term",
            "Surv(t0, t1, event) ~ matern(PC1, PC2, PC3, centers=6)",
        ),
    ] {
        let config = FitConfig {
            survival_likelihood: Some("marginal-slope".to_string()),
            logslope_formula: Some("matern(PC1, PC2, PC3, centers=6)".to_string()),
            z_column: Some("z".to_string()),
            ..FitConfig::default()
        };

        let materialized = materialize(formula, &data, &config).unwrap_or_else(|err| {
            panic!(
                "survival marginal-slope materialization should keep block-local penalties \
                     {case}: {err}"
            )
        });
        let FitRequest::SurvivalMarginalSlope(request) = materialized.request else {
            panic!("expected survival marginal-slope request for {case}");
        };
        let specs = vec![
            request.spec.marginalspec.clone(),
            request.spec.logslopespec.clone(),
        ];
        let (designs, frozen_specs) =
            crate::fit_orchestration::drivers::build_term_collection_designs_and_freeze_joint(
                data.values.view(),
                &specs,
            )
            .unwrap_or_else(|err| {
                panic!("joint freeze should preserve per-block penalty geometry {case}: {err}")
            });
        let (rebuilt, _) =
            crate::fit_orchestration::drivers::build_term_collection_designs_and_freeze_joint(
                data.values.view(),
                &frozen_specs,
            )
            .unwrap_or_else(|err| {
                panic!("frozen rebuild should preserve per-block penalty geometry {case}: {err}")
            });

        for (label, design) in [
            ("raw marginal", &designs[0]),
            ("raw logslope", &designs[1]),
            ("frozen marginal", &rebuilt[0]),
            ("frozen logslope", &rebuilt[1]),
        ] {
            let width = design.design.ncols();
            assert!(
                width > 2,
                "{case}: {label} design should be surface-width, not sex/intercept-width; \
                     width={width}"
            );
            for (idx, penalty) in design.penalties_as_penalty_matrix().iter().enumerate() {
                assert_eq!(
                    penalty.shape(),
                    (width, width),
                    "{case}: {label} penalty {idx} must be block-local at the surface width"
                );
            }
        }
    }
}

fn workflow_test_dataset() -> Dataset {
    Dataset {
        headers: vec![
            "age_entry".to_string(),
            "age_exit".to_string(),
            "event".to_string(),
            "bmi".to_string(),
            "z".to_string(),
        ],
        values: Array2::from_shape_vec(
            (4, 5),
            vec![
                40.0, 43.0, 1.0, 22.0, -1.0, 41.0, 46.0, 0.0, 24.0, -0.2, 42.0, 47.0, 1.0, 27.0,
                0.3, 44.0, 49.0, 0.0, 29.0, 1.2,
            ],
        )
        .expect("workflow test data shape"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "age_entry".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "age_exit".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "bmi".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    }
}

/// #1590 end-to-end: a cause-specific competing-risks Weibull fit must reach
/// convergence rather than aborting in `canonicalize_for_identifiability` with
/// "post-T rank invariant violated". This is the exact public-API repro from
/// the issue (`Surv(entry, exit, event) ~ age`, `event ∈ {0, 1, 2}`,
/// `survival_likelihood = "weibull"`), driven straight through the orchestration
/// entry so it exercises the real cause-specific block construction
/// (`fit_cause_specific_survival_transformation_custom`) and the channel-aware
/// identifiability audit on the genuine `x_exit` time-basis geometry.
#[test]
fn competing_risks_weibull_fit_is_reachable_1590() {
    let n = 320usize;
    // Deterministic synthetic competing-risks data (LCG, no external RNG dep) of
    // the same shape as the issue repro: two cause-specific exponential hazards
    // depending on a centered age covariate plus independent censoring.
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut unif = || {
        // SplitMix64 → (0, 1).
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut values = Array2::<f64>::zeros((n, 4)); // entry, exit, event, age
    for i in 0..n {
        let age = 40.0 + 35.0 * unif();
        let x = (age - 55.0) / 10.0;
        // Cause-specific exponential event times via inverse-CDF (-ln(u)/rate)
        // plus independent exponential censoring, matching the issue repro.
        let rate1 = (-3.0 + 0.25 * x).exp();
        let rate2 = (-3.2 - 0.20 * x).exp();
        let t1 = -unif().ln() / rate1;
        let t2 = -unif().ln() / rate2;
        let c = -unif().ln() * 22.0;
        let exit = t1.min(t2).min(c) + 0.1;
        let event = if t1 < t2 && t1 < c {
            1.0
        } else if t2 < t1 && t2 < c {
            2.0
        } else {
            0.0
        };
        values[[i, 0]] = 0.0;
        values[[i, 1]] = exit;
        values[[i, 2]] = event;
        values[[i, 3]] = age;
    }

    let data = Dataset {
        headers: vec![
            "entry".to_string(),
            "exit".to_string(),
            "event".to_string(),
            "age".to_string(),
        ],
        values,
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "entry".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "exit".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "age".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    };
    let config = FitConfig {
        survival_likelihood: Some("weibull".to_string()),
        ..FitConfig::default()
    };

    let result = crate::fit_orchestration::entry::fit_from_formula(
        "Surv(entry, exit, event) ~ age",
        &data,
        &config,
    );
    let fit_result = match result {
        Ok(r) => r,
        Err(e) => {
            let err = e.to_string();
            assert!(
                !err.contains("rank invariant violated"),
                "competing-risks Weibull fit must not abort on the post-T rank invariant (#1590); \
                 got: {err}"
            );
            assert!(
                !err.contains("beta length mismatch"),
                "competing-risks Weibull fit must not abort on a reduced-width/raw-width beta \
                 mismatch (#1590); got: {err}"
            );
            panic!("competing-risks Weibull fit failed (#1590): {err}");
        }
    };

    // The fit must not merely complete — it must actually estimate. Recover the
    // per-cause coefficient blocks and verify the cause-specific structure was
    // learned, not left at the pooled seed (the pre-fix failure mode kept every
    // coefficient pinned at its initial value behind a singular dead-column
    // Hessian).
    let FitResult::SurvivalTransformation(surv) = fit_result else {
        panic!("competing-risks Weibull fit must return a SurvivalTransformation result (#1590)");
    };
    assert_eq!(
        surv.fit.blocks.len(),
        2,
        "two competing causes must yield two coefficient blocks"
    );
    // Layout per cause: [β0 = dead anchor-centered time constant, β1 = Weibull
    // shape (slope on log t), β2 = covariate intercept (baseline level),
    // β3 = age]. The data-generating cause-specific log-rates are
    // +0.25·(age−55)/10 for cause 1 and −0.20·(age−55)/10 for cause 2, i.e. the
    // raw-age coefficient is +0.025 for cause 1 and −0.020 for cause 2.
    let beta1 = &surv.fit.blocks[0].beta;
    let beta2 = &surv.fit.blocks[1].beta;
    assert_eq!(
        beta1.len(),
        4,
        "cause 1 must keep raw width 4 (no reduction)"
    );
    assert_eq!(
        beta2.len(),
        4,
        "cause 2 must keep raw width 4 (no reduction)"
    );
    // The dead centered-constant coefficient is pinned to ~0 by the stabilization
    // ridge rather than left at its arbitrary unidentified seed.
    assert!(
        beta1[0].abs() < 1e-3 && beta2[0].abs() < 1e-3,
        "dead anchor-centered time constant β0 must be pinned to ~0, got {} and {}",
        beta1[0],
        beta2[0]
    );
    // Shape recovered near 1 (exponential cause-specific hazards).
    for (c, b) in [beta1, beta2].iter().enumerate() {
        assert!(
            b[1] > 0.5 && b[1] < 1.6,
            "cause {} Weibull shape β1 must be ~1 for exponential data, got {}",
            c + 1,
            b[1]
        );
    }
    // The qualitative cause-specific effect must be recovered: cause 1's hazard
    // RISES with age, cause 2's FALLS — opposite-signed age coefficients.
    assert!(
        beta1[3] > 0.0,
        "cause 1 age effect must be positive (hazard rises with age), got {}",
        beta1[3]
    );
    assert!(
        beta2[3] < 0.0,
        "cause 2 age effect must be negative (hazard falls with age), got {}",
        beta2[3]
    );
    // And the two causes must be genuinely DISTINCT fits, not a degenerate copy.
    assert!(
        (beta1[3] - beta2[3]).abs() > 0.01,
        "cause-specific age effects must differ (distinct fits), got {} vs {}",
        beta1[3],
        beta2[3]
    );
}

/// #1561 incidental bug: the Gaussian location-scale joint fit must not abort
/// (panic or hard-error) when the scale smooth is requested at a larger basis
/// size (`bs='tp', k>=20`). The owner's #1561 investigation reported a
/// joint-Newton crash there (`phantom_multiplier_with_well_conditioned_H`,
/// carrying-block μ) — a KKT-refusal robustness failure that is independent of
/// the (research-grade) scale-block λ-selection metric. A valid model spec
/// must always either fit or return a catchable error, never panic. This
/// reproduction fits the #1561 heteroscedastic-sinusoid fixture with the scale
/// formula at k=25 and asserts the call returns (Ok or Err) without panicking
/// and without bubbling the KKT cert-refusal diagnosis as a user-facing abort.
#[test]
fn issue_1561_locscale_large_scale_basis_does_not_crash_joint_newton() {
    let n = 200usize;
    let two_pi = 2.0 * std::f64::consts::PI;

    // Same seed-42 LCG fixture as the gating metric test, reproduced tool-free.
    let mut state: u64 = 42;
    let mut next_unit = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x: Vec<f64> = (0..n).map(|_| next_unit()).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut z: Vec<f64> = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_unit().max(1e-300);
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (two_pi * u2).cos());
        if z.len() < n {
            z.push(r * (two_pi * u2).sin());
        }
    }
    let mu_true = |t: f64| (two_pi * t).sin();
    let sigma_true = |t: f64| 0.1 + 0.2 * (two_pi * t).sin();
    let y: Vec<f64> = (0..n)
        .map(|i| mu_true(x[i]) + sigma_true(x[i]) * z[i])
        .collect();

    let td = tempdir().expect("tempdir");
    let data_path = td.path().join("locscale.csv");
    let mut csv = String::from("x,y\n");
    for i in 0..n {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    fs::write(&data_path, csv).expect("write locscale csv");
    let data = load_dataset_projected(&data_path, &["x".to_string(), "y".to_string()])
        .expect("load locscale dataset");

    // Sweep the scale-basis size across and above the k>=20 boundary the owner
    // reported as the joint-Newton crash region. Each must fit (Ok) without
    // panicking and without bubbling the KKT cert-refusal as a user abort.
    for k in [20usize, 25, 30] {
        let config = FitConfig {
            family: Some("gaussian".to_string()),
            noise_formula: Some(format!("1 + s(x, bs='tp', k={k})")),
            ..FitConfig::default()
        };

        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::fit_orchestration::entry::fit_from_formula("y ~ s(x, bs='tp')", &data, &config)
        }));
        let result = caught.unwrap_or_else(|_| {
            panic!(
                "#1561: location-scale fit with a k={k} scale smooth PANICKED inside the \
                 joint-Newton solver; a valid model spec must fit or return a catchable error, \
                 never unwind"
            )
        });
        match result {
            Ok(_) => {}
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    !msg.contains("phantom_multiplier_with_well_conditioned_H"),
                    "#1561: location-scale fit with a k={k} scale smooth bubbled a KKT \
                     cert-refusal (phantom_multiplier_with_well_conditioned_H) as a user-facing \
                     abort; the joint solver must recover (rho-anneal/seed-retry) on a \
                     well-conditioned penalized Hessian instead of refusing. Got: {msg}"
                );
                // Any other error is still a fit-quality/spec issue, not the
                // robustness crash this guard targets — surface it so the guard
                // stays honest about what it does and does not cover.
                panic!(
                    "#1561: location-scale fit with a k={k} scale smooth returned an unexpected \
                     error (not the targeted KKT crash): {msg}"
                );
            }
        }
    }
}

#[test]
fn issue_1561_secondary_smooth_retains_null_recovery_default() {
    let mut data = workflow_test_dataset();
    data.values = Array2::from_shape_fn((12, 5), |(row, col)| {
        let z = -1.0 + 2.0 * row as f64 / 11.0;
        [
            40.0 + row as f64,
            43.0 + row as f64,
            (row % 2) as f64,
            24.0 + z,
            z,
        ][col]
    });
    for (noise_formula, expected_double_penalty) in [
        ("1 + s(z, bs='tp')", true),
        ("1 + s(z, bs='tp', double_penalty=false)", false),
    ] {
        let materialized = materialize(
            "bmi ~ 1",
            &data,
            &FitConfig {
                family: Some("gaussian".to_string()),
                noise_formula: Some(noise_formula.to_string()),
                ..FitConfig::default()
            },
        )
        .expect("materialize Gaussian location-scale formula");
        let FitRequest::GaussianLocationScale(request) = materialized.request else {
            panic!("noise formula must materialize Gaussian location-scale");
        };
        let basis = &request.spec.log_sigmaspec.smooth_terms[0].basis;
        let SmoothBasisSpec::ThinPlate { spec, .. } = basis else {
            panic!("bs='tp' scale formula must resolve a thin-plate basis");
        };
        assert_eq!(
            spec.double_penalty, expected_double_penalty,
            "secondary materialization must preserve the ordinary null-recovery default and the explicit opt-out for `{noise_formula}`"
        );
    }
}

#[test]
fn issue_789_transformation_normal_rejects_marginal_slope_controls_before_dispatch() {
    let data = workflow_test_dataset();
    let config = FitConfig {
        transformation_normal: true,
        family: Some("bernoulli-marginal-slope".to_string()),
        logslope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };

    let err = materialize("event ~ bmi", &data, &config)
        .err()
        .expect("transformation_normal must not steal marginal-slope fits");

    assert!(
        err.to_string()
            .contains("transformation_normal cannot be combined with marginal-slope")
    );
}

#[test]
fn bernoulli_marginal_slope_ctn_stage1_recipe_only_dispatches_to_bms_issue_2139() {
    let data = workflow_test_dataset();
    let recipe = CtnStage1Recipe::new(
        "z",
        "bmi",
        TransformationNormalConfig::default(),
        None,
        None,
    )
    .expect("valid CTN Stage-1 recipe");
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        ctn_stage1: Some(recipe),
        ..FitConfig::default()
    };

    let err = materialize("event ~ bmi", &data, &config)
        .err()
        .expect("recipe-only BMS request should reach BMS validation");
    let msg = err.to_string();
    assert!(
        msg.contains("Bernoulli marginal-slope requires logslope_formula"),
        "recipe-only BMS request should fail with BMS-specific validation, got: {msg}"
    );
    assert!(
        !msg.contains("unknown family"),
        "family=bernoulli-marginal-slope with ctn_stage1 must not fall through to standard-family dispatch: {msg}"
    );
}

#[test]
fn family_transformation_normal_routes_to_ctn_materializer() {
    let data = workflow_test_dataset();
    let config = FitConfig {
        family: Some("transformation-normal".to_string()),
        ..FitConfig::default()
    };

    let mat = materialize("bmi ~ s(age_entry, k=4)", &data, &config)
        .expect("family='transformation-normal' must materialize as CTN");

    assert!(
        matches!(mat.request, FitRequest::TransformationNormal(_)),
        "family='transformation-normal' must not silently fall through to a standard Gaussian GAM"
    );
}

#[test]
fn family_transformation_normal_uses_ctn_conflict_validation() {
    let data = workflow_test_dataset();
    let config = FitConfig {
        family: Some("transformation_normal".to_string()),
        noise_formula: Some("~ 1".to_string()),
        ..FitConfig::default()
    };

    let err = materialize("bmi ~ s(age_entry, k=4)", &data, &config)
        .err()
        .expect("family='transformation-normal' must reject CTN-incompatible controls");

    assert!(
        err.to_string()
            .contains("transformation_normal cannot be combined with noise_formula"),
        "unexpected error: {err}"
    );
}

#[test]
fn survival_marginal_slope_rejects_zero_event_data_before_fit() {
    let mut data = workflow_test_dataset();
    data.values.column_mut(2).fill(0.0);
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        logslope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };

    let err = materialize("Surv(age_entry, age_exit, event) ~ bmi", &data, &config)
        .err()
        .expect("zero-event survival marginal-slope data must fail before optimization");

    assert!(err.to_string().contains("at least one target event"));
}

/// #2276: the fittability gate must weigh events. `workflow_test_dataset` has
/// events at rows 0 and 2; a weight column that is zero on exactly those rows
/// leaves an empty *weighted* event score (every kernel drops `weight <= 0`
/// rows), so the fit would spin on a flat landscape. The gate must reject it up
/// front — the raw event-code count alone was weight-blind.
#[test]
fn survival_all_events_zero_weighted_rejected_before_fit_issue_2276() {
    let data = workflow_dataset_with_weight([0.0, 1.0, 0.0, 1.0]);
    let config = FitConfig {
        weight_column: Some("w".to_string()),
        ..FitConfig::default()
    };

    let err = materialize("Surv(age_entry, age_exit, event) ~ bmi", &data, &config)
        .err()
        .expect("all-events-zero-weighted survival data must fail before optimization");
    let msg = err.to_string();
    assert!(
        msg.contains("at least one target event with positive weight"),
        "unexpected error: {msg}"
    );
}

/// #2276 control: a single event row with positive weight yields a positive
/// weighted event score, so the fittability gate must NOT trip (the fit may
/// still fail downstream for unrelated reasons, but never on the gate).
#[test]
fn survival_one_positive_weight_event_passes_fittability_gate_issue_2276() {
    let data = workflow_dataset_with_weight([1.0, 0.0, 0.0, 0.0]);
    let config = FitConfig {
        weight_column: Some("w".to_string()),
        ..FitConfig::default()
    };

    if let Err(err) = materialize("Surv(age_entry, age_exit, event) ~ bmi", &data, &config) {
        assert!(
            !err.to_string().contains("target event"),
            "a positive-weight event must not trip the fittability gate: {err}"
        );
    }
}

/// #2276 hardening: per-cause identifiability for competing risks. Codes
/// `{0, 1, 2}` with cause 1 carrying a positive-weight event but cause 2's only
/// event zero-weighted: the TOTAL weighted event mass is positive (cause 1), so
/// the total gate passes, yet cause 2's cause-specific hazard block is
/// unidentifiable and must be rejected before the fit.
#[test]
fn competing_risks_zero_weight_cause_rejected_before_fit_issue_2276() {
    let data = competing_risks_weighted_dataset([1.0, 2.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0]);
    let config = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        weight_column: Some("w".to_string()),
        ..FitConfig::default()
    };

    let err = materialize("Surv(age_entry, age_exit, event) ~ bmi", &data, &config)
        .err()
        .expect("a competing-risks cause with only zero-weight events must fail before fit");
    let msg = err.to_string();
    assert!(
        msg.contains("cause 2 of 2") && msg.contains("positive weight"),
        "unexpected error: {msg}"
    );
}

/// #2276 hardening control: when EVERY modeled cause carries a positive-weight
/// event, the per-cause gate must not trip (the fit may still fail downstream
/// for unrelated reasons, but never with the unidentifiable-cause message).
#[test]
fn competing_risks_all_causes_weighted_passes_per_cause_gate_issue_2276() {
    let data = competing_risks_weighted_dataset([1.0, 2.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]);
    let config = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        weight_column: Some("w".to_string()),
        ..FitConfig::default()
    };

    if let Err(err) = materialize("Surv(age_entry, age_exit, event) ~ bmi", &data, &config) {
        assert!(
            !err.to_string().contains("unidentifiable"),
            "all causes carrying a positive-weight event must not trip the per-cause gate: {err}"
        );
    }
}

/// Two-cause competing-risks dataset (`event` codes `{0, 1, 2}`) with a weight
/// column `w`, parallel to `codes`/`weights`.
fn competing_risks_weighted_dataset(codes: [f64; 4], weights: [f64; 4]) -> Dataset {
    let continuous = |name: &str| SchemaColumn {
        name: name.to_string(),
        kind: ColumnKindTag::Continuous,
        levels: vec![],
    };
    let entry = [40.0, 41.0, 42.0, 44.0];
    let exit = [43.0, 46.0, 47.0, 49.0];
    let bmi = [22.0, 24.0, 27.0, 29.0];
    let mut values = Array2::<f64>::zeros((4, 5));
    for i in 0..4 {
        values[[i, 0]] = entry[i];
        values[[i, 1]] = exit[i];
        values[[i, 2]] = codes[i];
        values[[i, 3]] = bmi[i];
        values[[i, 4]] = weights[i];
    }
    Dataset {
        headers: vec![
            "age_entry".to_string(),
            "age_exit".to_string(),
            "event".to_string(),
            "bmi".to_string(),
            "w".to_string(),
        ],
        values,
        schema: DataSchema {
            columns: vec![
                continuous("age_entry"),
                continuous("age_exit"),
                continuous("event"),
                continuous("bmi"),
                continuous("w"),
            ],
        },
        column_kinds: vec![ColumnKindTag::Continuous; 5],
    }
}

/// #2277: a bracketed interval-censored row (`event >= 1`) whose right boundary
/// equals its left boundary is a zero-width interval; the kernel term
/// `log[S(L) − S(R)] = log 0 = −∞` would poison the whole fit. Materialization
/// must reject it and name the offending row instead.
#[test]
fn surv_interval_rejects_degenerate_zero_width_bracket_issue_2277() {
    let data = surv_interval_degenerate_bracket_dataset();
    let config = FitConfig {
        survival_likelihood: Some("latent".to_string()),
        ..FitConfig::default()
    };

    let err = materialize("SurvInterval(left, right, event) ~ bmi", &data, &config)
        .err()
        .expect("a zero-width interval bracket (R == L) must be rejected at materialization");
    let msg = err.to_string();
    assert!(
        msg.contains("requires a finite R > L"),
        "unexpected error: {msg}"
    );
    assert!(
        msg.contains("row 1"),
        "the error must name the offending 1-based row: {msg}"
    );
}

/// `workflow_test_dataset` extended with a continuous weight column `w`.
fn workflow_dataset_with_weight(weights: [f64; 4]) -> Dataset {
    let mut data = workflow_test_dataset();
    let base_cols = data.values.ncols();
    let n = data.values.nrows();
    let mut values = Array2::<f64>::zeros((n, base_cols + 1));
    for i in 0..n {
        for j in 0..base_cols {
            values[[i, j]] = data.values[[i, j]];
        }
        values[[i, base_cols]] = weights[i];
    }
    data.headers.push("w".to_string());
    data.schema.columns.push(SchemaColumn {
        name: "w".to_string(),
        kind: ColumnKindTag::Continuous,
        levels: vec![],
    });
    data.column_kinds.push(ColumnKindTag::Continuous);
    data.values = values;
    data
}

/// Interval-censored dataset whose first bracketed row has `R == L` (a degenerate
/// zero-width interval). The remaining rows are well-formed so only the
/// degenerate row 0 (reported 1-based as "row 1") can trip the validator.
fn surv_interval_degenerate_bracket_dataset() -> Dataset {
    let continuous = |name: &str| SchemaColumn {
        name: name.to_string(),
        kind: ColumnKindTag::Continuous,
        levels: vec![],
    };
    Dataset {
        headers: vec![
            "left".to_string(),
            "right".to_string(),
            "event".to_string(),
            "bmi".to_string(),
        ],
        values: Array2::from_shape_vec(
            (3, 4),
            vec![
                5.0, 5.0, 1.0, 22.0, // L == R == 5 : degenerate bracket, event = 1
                6.0, 9.0, 1.0, 24.0, // valid bracket
                7.0, 10.0, 0.0, 27.0, // right-censored beyond last inspection
            ],
        )
        .expect("interval bracket test data shape"),
        schema: DataSchema {
            columns: vec![
                continuous("left"),
                continuous("right"),
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                continuous("bmi"),
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
        ],
    }
}

fn duchon_workflow_dataset() -> Dataset {
    let n = 72usize;
    let mut values = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let t = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        values[[i, 0]] = 0.5 * t.sin() + 0.15 * (3.0 * t).cos();
        values[[i, 1]] = t.cos();
        values[[i, 2]] = t.sin();
    }
    Dataset {
        headers: vec!["y".to_string(), "ct".to_string(), "st".to_string()],
        values,
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "ct".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "st".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    }
}

fn univariate_radial_workflow_dataset() -> Dataset {
    let n = 30usize;
    let mut values = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let x = i as f64 / (n - 1) as f64;
        values[[i, 0]] = (4.0 * std::f64::consts::PI * x).sin();
        values[[i, 1]] = x;
    }
    Dataset {
        headers: vec!["y".to_string(), "x".to_string()],
        values,
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "x".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![ColumnKindTag::Continuous, ColumnKindTag::Continuous],
    }
}

fn planned_radial_centers(basis: &SmoothBasisSpec) -> (usize, bool) {
    match basis {
        SmoothBasisSpec::Duchon {
            feature_cols, spec, ..
        } => (
            spec.center_strategy.planned_num_centers(feature_cols.len()),
            center_strategy_is_auto(&spec.center_strategy),
        ),
        other => panic!("expected Duchon basis, got {other:?}"),
    }
}

#[test]
fn adaptive_univariate_duchon_start_preserves_formula_floor_and_applies_growth_1867() {
    let data = univariate_radial_workflow_dataset();

    let label = "Duchon";
    let formula = "y ~ duchon(x)";
    let raw = materialize(formula, &data, &FitConfig::default())
        .unwrap_or_else(|error| panic!("raw {label} materialization failed: {error}"));
    let FitRequest::Standard(raw_request) = raw.request else {
        panic!("expected standard {label} request");
    };
    let (raw_centers, raw_is_auto) =
        planned_radial_centers(&raw_request.spec.smooth_terms[0].basis);
    assert!(
        raw_is_auto,
        "implicit {label} centers must retain Auto provenance"
    );
    assert!(
        raw_centers > starting_num_centers(data.values.nrows(), 1),
        "{label} formula default must retain its derived univariate resolution floor"
    );

    let initial_config = FitConfig {
        spatial_center_counts: Some(Vec::new()),
        ..FitConfig::default()
    };
    let initial = materialize(formula, &data, &initial_config)
        .unwrap_or_else(|error| panic!("initial adaptive {label} materialization failed: {error}"));
    let FitRequest::Standard(initial_request) = initial.request else {
        panic!("expected standard adaptive {label} request");
    };
    let (initial_centers, initial_is_auto) =
        planned_radial_centers(&initial_request.spec.smooth_terms[0].basis);
    assert_eq!(
        initial_centers, raw_centers,
        "an absent adaptive proposal must preserve the canonical 1-D {label} formula resolution"
    );
    assert!(
        initial_is_auto,
        "adaptive {label} centers must retain Auto provenance"
    );

    let proposed_centers = raw_centers.saturating_mul(2).min(data.values.nrows());
    assert!(
        proposed_centers > raw_centers,
        "test data must leave room for a genuine {label} growth proposal"
    );
    let growth_config = FitConfig {
        spatial_center_counts: Some(vec![Some(proposed_centers)]),
        ..FitConfig::default()
    };
    let grown = materialize(formula, &data, &growth_config)
        .unwrap_or_else(|error| panic!("grown adaptive {label} materialization failed: {error}"));
    let FitRequest::Standard(grown_request) = grown.request else {
        panic!("expected grown standard {label} request");
    };
    let (grown_centers, grown_is_auto) =
        planned_radial_centers(&grown_request.spec.smooth_terms[0].basis);
    assert_eq!(
        grown_centers, proposed_centers,
        "an explicit adaptive {label} growth proposal must remain authoritative"
    );
    assert!(
        grown_is_auto,
        "grown {label} centers must retain Auto provenance"
    );
}

#[test]
fn matern_is_excluded_from_generic_adaptive_center_growth() {
    let data = univariate_radial_workflow_dataset();
    let formula = "y ~ matern(x)";
    let raw = materialize(formula, &data, &FitConfig::default()).expect("raw Matérn request");
    let FitRequest::Standard(raw_request) = raw.request else {
        panic!("expected standard Matérn request");
    };
    let SmoothBasisSpec::Matern { spec: raw_spec, .. } = &raw_request.spec.smooth_terms[0].basis
    else {
        panic!("expected Matérn basis");
    };
    let raw_centers = raw_spec.center_strategy.planned_num_centers(1);

    let adaptive = materialize(
        formula,
        &data,
        &FitConfig {
            spatial_center_counts: Some(vec![Some(raw_centers.saturating_mul(2))]),
            ..FitConfig::default()
        },
    )
    .expect("Matérn request with generic adaptive proposal");
    let FitRequest::Standard(adaptive_request) = adaptive.request else {
        panic!("expected standard Matérn request");
    };
    let SmoothBasisSpec::Matern {
        spec: adaptive_spec,
        ..
    } = &adaptive_request.spec.smooth_terms[0].basis
    else {
        panic!("expected Matérn basis");
    };
    assert_eq!(
        adaptive_spec.center_strategy.planned_num_centers(1),
        raw_centers,
        "generic EDF saturation proposals must not rewrite Matérn center topology"
    );
}

#[test]
fn adaptive_spatial_start_is_activated_only_by_its_orchestrator() {
    let data = duchon_workflow_dataset();

    let raw = materialize("y ~ duchon(ct, st)", &data, &FitConfig::default())
        .expect("raw Duchon materialization");
    let FitRequest::Standard(raw_request) = raw.request else {
        panic!("expected standard request");
    };
    let SmoothBasisSpec::Duchon { spec: raw_spec, .. } = &raw_request.spec.smooth_terms[0].basis
    else {
        panic!("expected Duchon smooth");
    };
    let raw_centers = raw_spec.center_strategy.planned_num_centers(2);

    let adaptive_config = FitConfig {
        spatial_center_counts: Some(Vec::new()),
        ..FitConfig::default()
    };
    let adaptive = materialize("y ~ duchon(ct, st)", &data, &adaptive_config)
        .expect("adaptive-start Duchon materialization");
    let FitRequest::Standard(adaptive_request) = adaptive.request else {
        panic!("expected standard request");
    };
    let SmoothBasisSpec::Duchon {
        spec: adaptive_spec,
        ..
    } = &adaptive_request.spec.smooth_terms[0].basis
    else {
        panic!("expected Duchon smooth");
    };
    let adaptive_centers = adaptive_spec.center_strategy.planned_num_centers(2);
    assert_eq!(
        adaptive_centers,
        starting_num_centers(data.values.nrows(), 2)
    );
    assert!(
        raw_centers > adaptive_centers,
        "raw materialization must retain the ordinary basis because it has no grow-loop owner"
    );
    assert!(center_strategy_is_auto(&adaptive_spec.center_strategy));

    let explicit = materialize("y ~ duchon(ct, st, centers=12)", &data, &adaptive_config)
        .expect("explicit-center Duchon materialization");
    let FitRequest::Standard(explicit_request) = explicit.request else {
        panic!("expected standard request");
    };
    let SmoothBasisSpec::Duchon {
        spec: explicit_spec,
        ..
    } = &explicit_request.spec.smooth_terms[0].basis
    else {
        panic!("expected Duchon smooth");
    };
    assert_eq!(explicit_spec.center_strategy.planned_num_centers(2), 12);
    assert!(!center_strategy_is_auto(&explicit_spec.center_strategy));
}

#[test]
fn materialize_standard_keeps_adaptive_regularization_off_by_default_for_duchon() {
    let data = duchon_workflow_dataset();
    let materialized = materialize(
        "y ~ duchon(ct, st, centers=12)",
        &data,
        &FitConfig::default(),
    )
    .expect("Duchon standard materialization should succeed");
    let FitRequest::Standard(request) = materialized.request else {
        panic!("expected standard request");
    };
    assert!(request.options.adaptive_regularization.is_none());
}

#[test]
fn materialize_standard_honors_adaptive_regularization_enable() {
    let data = duchon_workflow_dataset();
    let config = FitConfig {
        adaptive_regularization: Some(true),
        ..FitConfig::default()
    };
    let materialized = materialize("y ~ duchon(ct, st, centers=12)", &data, &config)
        .expect("Duchon materialization should allow enabling adaptive regularization");
    let FitRequest::Standard(request) = materialized.request else {
        panic!("expected standard request");
    };
    let opts = request
        .options
        .adaptive_regularization
        .expect("Duchon should enable adaptive regularization when requested");
    assert!(opts.enabled);
}

#[test]
fn materialize_standard_honors_adaptive_regularization_disable() {
    let data = duchon_workflow_dataset();
    let config = FitConfig {
        adaptive_regularization: Some(false),
        ..FitConfig::default()
    };
    let materialized = materialize("y ~ duchon(ct, st, centers=12)", &data, &config)
        .expect("Duchon materialization should allow disabling adaptive regularization");
    let FitRequest::Standard(request) = materialized.request else {
        panic!("expected standard request");
    };
    assert!(request.options.adaptive_regularization.is_none());
}

#[test]
fn issue_2094_sas_and_beta_logistic_links_enable_optimize_sas_on_formula_path() {
    // #2094: the learnable `sas` (sinh-arcsinh) and `beta-logistic` links carry
    // a shape pair `(epsilon, log_delta)` that the standard fit only estimates
    // when `optimize_sas=true` AND a `sas_link` spec is threaded into the fit
    // options. Before the fix, `materialize_standard` left both at their
    // defaults (`sas_link=None`, `optimize_sas=false`), so the outer optimizer
    // set `sas_dim=0`, the shape stayed frozen at its init, and `link(type=sas)`
    // silently collapsed to plain probit / `link(type=beta-logistic)` to plain
    // logit on the formula/Python path — while the `gam` CLI fit them correctly.
    // This mirrors the CLI's `sas_linkspec` / `optimize_sas` wiring in
    // run_fit.rs and is the SAS analogue of the mixture-link freeze fixed in
    // #1598. This test fails before the fix (both assertions false) and passes
    // after.
    let data = workflow_test_dataset();
    for (formula, label) in [
        ("event ~ bmi + link(type=sas)", "sas"),
        ("event ~ bmi + link(type=beta-logistic)", "beta-logistic"),
    ] {
        let materialized = materialize(formula, &data, &FitConfig::default())
            .unwrap_or_else(|e| panic!("{label}: materialize failed: {e}"));
        let FitRequest::Standard(request) = materialized.request else {
            panic!("{label}: expected a standard fit request");
        };
        assert!(
            request.options.sas_link.is_some(),
            "{label}: sas_link must be populated so the standard path can rebuild \
             and fit the learnable link state (#2094)"
        );
        assert!(
            request.options.optimize_sas,
            "{label}: optimize_sas must be true or the SAS shape stays frozen at \
             its init and the link degrades to its plain base link (#2094)"
        );
    }
}

#[test]
fn materialize_standard_duchon_defaults_to_pure_scale_free_basis() {
    let data = duchon_workflow_dataset();
    let materialized = materialize(
        "y ~ duchon(ct, st, centers=12)",
        &data,
        &FitConfig::default(),
    )
    .expect("Duchon materialization should succeed");
    let FitRequest::Standard(request) = materialized.request else {
        panic!("expected standard request");
    };
    let SmoothBasisSpec::Duchon { spec, .. } = &request.spec.smooth_terms[0].basis else {
        panic!("expected Duchon smooth");
    };
    assert_eq!(spec.length_scale, None);
    assert!(matches!(spec.nullspace_order, DuchonNullspaceOrder::Linear));
    assert_eq!(spec.power, 0.5);
}

#[test]
fn materialize_standard_duchon_length_scale_opts_into_hybrid_basis() {
    let data = duchon_workflow_dataset();
    let materialized = materialize(
        "y ~ duchon(ct, st, centers=12, length_scale=1.0)",
        &data,
        &FitConfig::default(),
    )
    .expect("hybrid Duchon materialization should succeed");
    let FitRequest::Standard(request) = materialized.request else {
        panic!("expected standard request");
    };
    let SmoothBasisSpec::Duchon { spec, .. } = &request.spec.smooth_terms[0].basis else {
        panic!("expected Duchon smooth");
    };
    assert_eq!(spec.length_scale, Some(1.0));
    assert_eq!(spec.nullspace_order, DuchonNullspaceOrder::Linear);
    // The hybrid Matérn-blended kernel requires an INTEGER power. The cubic
    // structural default's fractional s=(d-1)/2 = 0.5 (d=2) is resolved at the
    // request layer to the smallest admissible integer (here s=0, the d=2
    // thin-plate order) rather than carried in as 0.5 and silently truncated
    // to 0 by the basis builder (#750). The pure path above still keeps 0.5.
    assert_eq!(spec.power, 0.0);
}

#[test]
fn workflow_survival_marginal_slope_routes_logslope_linkwiggle_into_score_warp_only() {
    let data = workflow_test_dataset();
    // #384: the score-warp / link-deviation runtime is structurally cubic, so
    // only `degree=3` is realizable on these blocks; non-cubic degrees are
    // rejected up front (see
    // `linkwiggle_noncubic_degree_is_rejected_at_the_routing_boundary_issue_384`).
    // This test exercises the orthogonal routing/metadata contract: the
    // logslope_formula linkwiggle lands on `score_warp` and the main-formula
    // linkwiggle on `link_dev`, with knots/penalty orders carried through. The
    // two blocks are distinguished here by `internal_knots` (9 vs 7) and
    // `penalty_order` (1 vs 2,3), not by an unrealizable degree.
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        logslope_formula: Some(
            "1 + linkwiggle(degree=3, internal_knots=7, penalty_order=\"2,3\")".to_string(),
        ),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };
    let materialized = materialize(
            "Surv(age_entry, age_exit, event) ~ s(bmi) + linkwiggle(degree=3, internal_knots=9, penalty_order=\"1\")",
            &data,
            &config,
        )
        .expect("workflow materialization should succeed");

    let MaterializedModel {
        request,
        inference_notes,
    } = materialized;
    let FitRequest::SurvivalMarginalSlope(request) = request else {
        panic!("expected survival marginal-slope request");
    };

    let link_dev = request.spec.link_dev.expect("main-formula link-dev");
    let score_warp = request.spec.score_warp.expect("logslope score-warp");
    assert_eq!(link_dev.degree, 3);
    assert_eq!(link_dev.num_internal_knots, 9);
    assert_eq!(link_dev.penalty_order, 1);
    assert_eq!(link_dev.penalty_orders, vec![1]);
    assert_eq!(score_warp.degree, 3);
    assert_eq!(score_warp.num_internal_knots, 7);
    assert_eq!(score_warp.penalty_order, 3);
    assert_eq!(score_warp.penalty_orders, vec![2, 3]);
    assert!(
        inference_notes
            .iter()
            .any(|note| note.contains("link-deviation block")),
        "workflow notes should mention main-formula linkwiggle routing"
    );
    assert!(
        inference_notes
            .iter()
            .any(|note| note.contains("score-warp block")),
        "workflow notes should mention logslope_formula linkwiggle routing"
    );
}

#[test]
fn materialize_routes_bernoulli_marginal_slope_when_logslope_and_z_are_set() {
    let data = workflow_test_dataset();
    let config = FitConfig {
        logslope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };
    let materialized = materialize("event ~ bmi", &data, &config)
        .expect("Bernoulli marginal-slope materialization should succeed");
    assert!(matches!(
        materialized.request,
        FitRequest::BernoulliMarginalSlope(_)
    ));
}

#[test]
fn materialize_bernoulli_marginal_slope_prunes_redundant_scalar_term() {
    let data = Dataset {
        headers: vec![
            "event".to_string(),
            "x".to_string(),
            "constant_spline_col".to_string(),
            "prs_z".to_string(),
            "PC1".to_string(),
            "PC2".to_string(),
            "PC3".to_string(),
        ],
        values: Array2::from_shape_vec(
            (6, 7),
            vec![
                0.0, -2.0, 1.0, -1.2, -1.0, 0.2, 0.7, 1.0, -1.0, 1.0, -0.4, -0.4, -0.3, 0.5, 0.0,
                0.0, 1.0, 0.1, 0.1, 0.4, -0.2, 1.0, 1.0, 1.0, 0.5, 0.7, -0.6, 0.3, 0.0, 2.0, 1.0,
                1.1, 1.2, 0.9, 0.0, 1.0, 3.0, 1.0, 1.7, 1.6, -0.8, -0.4,
            ],
        )
        .expect("BMS redundant scalar test data shape"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "x".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "constant_spline_col".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "prs_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    };
    let config = FitConfig {
        logslope_formula: Some("matern(PC1, PC2, PC3, centers=3)".to_string()),
        z_column: Some("prs_z".to_string()),
        ..FitConfig::default()
    };
    let materialized = materialize(
        "event ~ matern(PC1, PC2, PC3, centers=3) + x + constant_spline_col",
        &data,
        &config,
    )
    .expect("BMS materialization should prune the redundant scalar term");
    let MaterializedModel {
        request,
        inference_notes,
    } = materialized;
    let FitRequest::BernoulliMarginalSlope(request) = request else {
        panic!("expected Bernoulli marginal-slope request");
    };
    let kept: Vec<&str> = request
        .spec
        .marginalspec
        .linear_terms
        .iter()
        .map(|term| term.name.as_str())
        .collect();
    assert_eq!(kept, vec!["x"]);
    assert_eq!(request.spec.marginalspec.smooth_terms.len(), 1);
    assert_eq!(request.spec.logslopespec.smooth_terms.len(), 1);
    assert!(
        inference_notes
            .iter()
            .any(|note| note.contains("constant_spline_col")),
        "materialization should report the removed redundant scalar term; notes={inference_notes:?}"
    );
}

#[test]
fn materialize_bernoulli_marginal_slope_prunes_binary_outcome_style_scalar_alias() {
    let data = Dataset {
        headers: vec![
            "event".to_string(),
            "sex".to_string(),
            "entry_age_z".to_string(),
            "current_age_ns_1".to_string(),
            "current_age_ns_2".to_string(),
            "current_age_ns_3".to_string(),
            "current_age_ns_4".to_string(),
            "prs_z".to_string(),
            "PC1".to_string(),
            "PC2".to_string(),
            "PC3".to_string(),
        ],
        values: Array2::from_shape_vec(
            (8, 11),
            vec![
                0.0, 0.0, -1.4, 1.0, -0.6, 0.36, -0.216, -1.3, -1.0, 0.2, 0.7, 1.0, 1.0, -0.9, 1.0,
                -0.2, 0.04, -0.008, -0.8, -0.5, -0.3, 0.5, 0.0, 0.0, -0.5, 1.0, 0.1, 0.01, 0.001,
                -0.2, 0.1, 0.4, -0.2, 1.0, 1.0, -0.1, 1.0, 0.4, 0.16, 0.064, 0.3, 0.7, -0.6, 0.3,
                0.0, 0.0, 0.3, 1.0, 0.7, 0.49, 0.343, 0.8, 1.2, 0.9, 0.0, 1.0, 1.0, 0.7, 1.0, 1.0,
                1.0, 1.0, 1.2, 1.6, -0.8, -0.4, 0.0, 0.0, 1.1, 1.0, 1.3, 1.69, 2.197, 1.6, -1.4,
                0.8, -0.9, 1.0, 1.0, 1.5, 1.0, 1.6, 2.56, 4.096, 2.0, 0.3, -1.1, 0.6,
            ],
        )
        .expect("binary-outcome-style BMS scalar-alias test data shape"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "sex".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "entry_age_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "current_age_ns_4".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "prs_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC1".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC2".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "PC3".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Binary,
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    };
    let config = FitConfig {
        logslope_formula: Some("matern(PC1, PC2, PC3, centers=3)".to_string()),
        z_column: Some("prs_z".to_string()),
        ..FitConfig::default()
    };
    let materialized = materialize(
            "event ~ matern(PC1, PC2, PC3, centers=3) + sex + entry_age_z + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4",
            &data,
            &config,
        )
        .expect("BMS materialization should prune the local-column-3 scalar alias");
    let FitRequest::BernoulliMarginalSlope(request) = materialized.request else {
        panic!("expected Bernoulli marginal-slope request");
    };
    let kept: Vec<&str> = request
        .spec
        .marginalspec
        .linear_terms
        .iter()
        .map(|term| term.name.as_str())
        .collect();
    assert_eq!(
        kept,
        vec![
            "sex",
            "entry_age_z",
            "current_age_ns_2",
            "current_age_ns_3",
            "current_age_ns_4"
        ]
    );
    assert_eq!(request.spec.marginalspec.smooth_terms.len(), 1);
    assert_eq!(request.spec.logslopespec.smooth_terms.len(), 1);
    assert!(
        materialized
            .inference_notes
            .iter()
            .any(|note| note.contains("current_age_ns_1")),
        "materialization should report the removed binary-outcome-style scalar alias; notes={:?}",
        materialized.inference_notes
    );
}

#[test]
fn materialize_bernoulli_marginal_slope_rejects_constrained_redundant_scalar_term() {
    let data = Dataset {
        headers: vec![
            "event".to_string(),
            "x".to_string(),
            "constant_spline_col".to_string(),
            "prs_z".to_string(),
        ],
        values: Array2::from_shape_vec(
            (6, 4),
            vec![
                0.0, -2.0, 1.0, -1.2, 1.0, -1.0, 1.0, -0.4, 0.0, 0.0, 1.0, 0.1, 1.0, 1.0, 1.0, 0.5,
                0.0, 2.0, 1.0, 1.1, 1.0, 3.0, 1.0, 1.7,
            ],
        )
        .expect("BMS constrained redundant scalar test data shape"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "x".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "constant_spline_col".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "prs_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    };
    let config = FitConfig {
        logslope_formula: Some("1".to_string()),
        z_column: Some("prs_z".to_string()),
        ..FitConfig::default()
    };
    let err = match materialize(
        "event ~ x + linear(constant_spline_col, min=0.0)",
        &data,
        &config,
    ) {
        Ok(_) => panic!("constrained duplicate scalar term must be rejected, not pruned"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("constrained linear term 'constant_spline_col' is redundant"),
        "error should explain that the constrained duplicate scalar cannot be pruned: {msg}"
    );
}

#[test]
fn bernoulli_marginal_slope_prune_rejects_penalized_redundant_scalar_term() {
    let data = Dataset {
        headers: vec!["event".to_string(), "constant_spline_col".to_string()],
        values: Array2::from_shape_vec((4, 2), vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
            .expect("BMS penalized redundant scalar test data shape"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "constant_spline_col".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![ColumnKindTag::Binary, ColumnKindTag::Continuous],
    };
    let mut spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "constant_spline_col".to_string(),
            feature_col: 1,
            feature_cols: vec![1],
            categorical_levels: vec![],
            double_penalty: true,
            coefficient_geometry: gam_terms::smooth::LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: vec![],
        smooth_terms: vec![],
    };
    let mut notes = Vec::new();
    let err = prune_unidentified_linear_terms_for_marginal_slope(
        &mut spec,
        &data,
        "test BMS formula",
        &mut notes,
    )
    .err()
    .expect("explicitly penalized duplicate scalar term must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("explicitly penalized linear term 'constant_spline_col' is redundant"),
        "error should reject ridge-identification of duplicate scalar directions: {msg}"
    );
    assert_eq!(spec.linear_terms.len(), 1);
    assert!(notes.is_empty());
}

#[test]
fn materialize_bernoulli_marginal_slope_names_constant_z_column() {
    let data = Dataset {
        headers: vec!["event".to_string(), "bmi".to_string(), "prs_z".to_string()],
        values: Array2::from_shape_vec(
            (4, 3),
            vec![
                0.0, 22.0, -0.58, 1.0, 24.0, -0.58, 0.0, 27.0, -0.58, 1.0, 29.0, -0.58,
            ],
        )
        .expect("constant z test data shape"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "bmi".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "prs_z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ],
    };
    let config = FitConfig {
        logslope_formula: Some("1".to_string()),
        z_column: Some("prs_z".to_string()),
        ..FitConfig::default()
    };

    let err = match materialize("event ~ bmi", &data, &config) {
        Ok(_) => panic!("constant z_column should be rejected before BMS integration"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("z_column 'prs_z' has zero weighted variance"),
        "error should name the constant z_column and diagnose weighted variance: {msg}"
    );
    assert!(
        msg.contains("all 4 values ~= -0.580000"),
        "error should summarize the observed constant value: {msg}"
    );
    assert!(
        msg.contains("weighted_sd=0.000000e0") && msg.contains("n=4"),
        "error should report weighted_sd and n: {msg}"
    );
    assert!(
            msg.contains(
                "bernoulli-marginal-slope cannot identify a covariate-varying slope from a constant score"
            ),
            "error should explain why the input is invalid: {msg}"
        );
    assert!(
        !msg.contains("requires z with positive finite weighted standard deviation"),
        "workflow should surface the input-style message instead of the generic BMS normalization error: {msg}"
    );
}

#[test]
fn linkwiggle_noncubic_degree_is_rejected_at_the_routing_boundary_issue_384() {
    // #384: the score-warp / link-deviation block is realized by a structurally
    // *cubic* I-spline runtime, so only `degree == 3` is realizable. The shared
    // parser stays general (it also feeds the arbitrary-degree `timewiggle` /
    // location-scale monotone basis), so a non-cubic `linkwiggle(degree=k)`
    // parses fine — the cubic-only contract must be enforced UP FRONT at the
    // marginal-slope routing boundary, not deep inside the fit where it
    // surfaced as a cryptic "structural deviation runtime is cubic; degree must
    // be 3" IntegrationError after expensive setup.
    use crate::fit_orchestration::route_marginal_slope_deviation_blocks;

    for deg in [1usize, 2, 4, 10] {
        let mut options = std::collections::BTreeMap::new();
        options.insert("degree".to_string(), deg.to_string());
        options.insert("internal_knots".to_string(), "3".to_string());
        let raw = format!("linkwiggle(degree={deg}, internal_knots=3)");
        let spec = parse_linkwiggle_formulaspec(&options, &raw)
            .expect("non-cubic wiggle degree must still parse at the shared layer");
        assert_eq!(
            spec.degree, deg,
            "parser must carry the degree through verbatim"
        );

        // logslope_formula = linkwiggle(...) is the score-warp route the Python
        // marginal-slope path uses.
        let err = route_marginal_slope_deviation_blocks(None, Some(&spec))
            .err()
            .expect("non-cubic linkwiggle must be rejected before any fit");
        assert!(
            err.contains("degree must be 3"),
            "rejection must name the cubic-only contract, got: {err}"
        );
        assert!(
            err.contains("score-warp"),
            "rejection must identify the score-warp / link-deviation block, got: {err}"
        );

        // The main-formula link-deviation route is gated identically.
        let err_main = route_marginal_slope_deviation_blocks(Some(&spec), None)
            .err()
            .expect("non-cubic link-deviation must be rejected before any fit");
        assert!(err_main.contains("degree must be 3"));
    }

    // The realizable cubic degree routes successfully (no false rejection).
    let mut cubic_opts = std::collections::BTreeMap::new();
    cubic_opts.insert("degree".to_string(), "3".to_string());
    cubic_opts.insert("internal_knots".to_string(), "3".to_string());
    let cubic = parse_linkwiggle_formulaspec(&cubic_opts, "linkwiggle(degree=3, internal_knots=3)")
        .expect("cubic linkwiggle parses");
    let routing = route_marginal_slope_deviation_blocks(None, Some(&cubic))
        .expect("cubic degree must route without error");
    assert!(routing.score_warp.is_some());
    assert_eq!(routing.score_warp.unwrap().degree, 3);
}

#[test]
fn linkwiggle_defaults_are_consistent_across_formula_and_runtime() {
    let parsed = parse_linkwiggle_formulaspec(&Default::default(), "linkwiggle()")
        .expect("default linkwiggle should parse");
    let formula_default = default_linkwiggle_formulaspec();
    let runtime_default = DeviationBlockConfig::default();
    assert_eq!(parsed.degree, formula_default.degree);
    assert_eq!(
        parsed.num_internal_knots,
        formula_default.num_internal_knots
    );
    assert_eq!(parsed.penalty_orders, formula_default.penalty_orders);
    assert_eq!(parsed.double_penalty, formula_default.double_penalty);
    assert_eq!(runtime_default.degree, formula_default.degree);
    assert_eq!(
        runtime_default.num_internal_knots,
        formula_default.num_internal_knots
    );
    assert_eq!(
        runtime_default.penalty_orders,
        formula_default.penalty_orders
    );
    assert_eq!(
        runtime_default.double_penalty,
        formula_default.double_penalty
    );
}

#[test]
fn survival_marginal_slope_accepts_explicit_probit_link() {
    let data = workflow_test_dataset();
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        logslope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };
    let ok = materialize(
        "Surv(age_entry, age_exit, event) ~ bmi + link(type=probit)",
        &data,
        &config,
    );
    assert!(ok.is_ok(), "explicit probit should be accepted");

    let err = match materialize(
        "Surv(age_entry, age_exit, event) ~ bmi + link(type=logit)",
        &data,
        &config,
    ) {
        Ok(_) => panic!("non-probit link should be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("only link(type=probit)"));
}

#[test]
fn high_dimensional_duchon_default_power_is_admissible() {
    let dim = 16;
    let power = minimum_duchon_power_for_operator_penalties(dim, DuchonNullspaceOrder::Zero, 2);
    assert!(2 * (1 + power) > dim + 2);
}

#[test]
fn survival_location_scale_wiggle_rejects_unsupported_inverse_link() {
    let data = workflow_test_dataset();
    let materialized = materialize(
            "Surv(age_entry, age_exit, event) ~ bmi + linkwiggle(degree=4, internal_knots=3, penalty_order=\"1\")",
            &data,
            &FitConfig::default(),
        )
        .expect("workflow materialization should succeed");

    let MaterializedModel { request, .. } = materialized;
    let FitRequest::SurvivalLocationScale(mut request) = request else {
        panic!("expected survival location-scale request");
    };
    request.spec.inverse_link = InverseLink::Sas(
        state_from_sasspec(SasLinkSpec {
            initial_epsilon: 0.1,
            initial_log_delta: 0.0,
        })
        .expect("valid SAS state"),
    );
    request.optimize_inverse_link = false;

    let err = match fit_survival_location_scale_model(request) {
        Ok(_) => panic!("survival link wiggle should reject unsupported inverse links"),
        Err(e) => e,
    };

    assert!(err.contains("survival link wiggle"));
    assert!(err.contains("does not support"));
}

// #371: survival-only / binomial-only DSL controls must be *rejected* in a
// non-survival main formula, not parsed-and-silently-dropped. The bug was
// that `parsed.timewiggle` / `parsed.survivalspec` are consumed only by
// `materialize_survival`, and an explicit `linkwiggle(...)` is wired into
// the fit only on the binomial arm, so a Gaussian formula carrying any of
// these accepted the term and then ignored it — the user got an ordinary
// GAM while believing they had configured a time-varying / wiggled model.

#[test]
fn timewiggle_rejected_in_nonsurvival_main_formula() {
    // `bmi` is a continuous response -> Gaussian standard path, no Surv(...).
    let data = workflow_test_dataset();
    let err = materialize(
        "bmi ~ z + timewiggle(internal_knots=4)",
        &data,
        &FitConfig::default(),
    )
    .err()
    .expect("timewiggle in a non-survival formula must be rejected, not silently ignored");
    let msg = err.to_string();
    assert!(
        msg.contains("timewiggle(...)") && msg.contains("survival"),
        "error should explain timewiggle is survival-only, got: {msg}"
    );
}

#[test]
fn survmodel_rejected_in_nonsurvival_main_formula() {
    let data = workflow_test_dataset();
    let err = materialize(
        "bmi ~ z + survmodel(spec=net)",
        &data,
        &FitConfig::default(),
    )
    .err()
    .expect("survmodel in a non-survival formula must be rejected, not silently ignored");
    let msg = err.to_string();
    assert!(
        msg.contains("survmodel(...)") && msg.contains("survival"),
        "error should explain survmodel is survival-only, got: {msg}"
    );
}

#[test]
fn linkwiggle_rejected_for_nonbinomial_response() {
    // `bmi` is continuous -> Gaussian; an explicit `linkwiggle(...)` corrects
    // a binomial link and would otherwise be dropped on the floor here.
    let data = workflow_test_dataset();
    let err = materialize(
        "bmi ~ z + linkwiggle(internal_knots=4)",
        &data,
        &FitConfig::default(),
    )
    .err()
    .expect("linkwiggle on a non-binomial response must be rejected, not silently ignored");
    let msg = err.to_string();
    assert!(
        msg.contains("linkwiggle(...)") && msg.contains("binomial"),
        "error should explain linkwiggle is binomial-only, got: {msg}"
    );
}

#[test]
fn flexible_link_rejected_for_nonbinomial_standard_response() {
    let data = workflow_test_dataset();
    let mut config = FitConfig::default();
    config.family = Some("poisson".to_string());
    config.link = Some("flexible(log)".to_string());

    let err = materialize("bmi ~ z", &data, &config)
        .err()
        .expect("flexible(log) on a Poisson response must be rejected, not silently ignored");
    let msg = err.to_string();
    assert!(
        msg.contains("flexible(...)") && msg.contains("non-binomial"),
        "error should explain flexible links are binomial-only, got: {msg}"
    );
}

#[test]
fn formula_flexible_link_rejected_for_nonbinomial_standard_response() {
    let data = workflow_test_dataset();
    let mut config = FitConfig::default();
    config.family = Some("poisson".to_string());

    let err = materialize("bmi ~ z + link(type=flexible(log))", &data, &config)
        .err()
        .expect("formula flexible(log) on a Poisson response must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("flexible(...)") && msg.contains("non-binomial"),
        "error should explain flexible links are binomial-only, got: {msg}"
    );
}

#[test]
fn flexible_link_flag_rejected_for_nonbinomial_standard_response() {
    let data = workflow_test_dataset();
    let mut config = FitConfig::default();
    config.family = Some("gaussian".to_string());
    config.flexible_link = true;

    let err = materialize("bmi ~ z", &data, &config)
        .err()
        .expect("flexible_link=True on a Gaussian response must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("flexible(...)") && msg.contains("non-binomial"),
        "error should explain flexible links are binomial-only, got: {msg}"
    );
}

#[test]
fn flexible_link_rejected_for_nonbinomial_location_scale_response() {
    let data = workflow_test_dataset();
    let mut config = FitConfig::default();
    config.link = Some("flexible(identity)".to_string());
    config.noise_formula = Some("1".to_string());

    let err = materialize("bmi ~ z", &data, &config)
        .err()
        .expect("flexible(identity) on a Gaussian location-scale response must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("flexible(...)") && msg.contains("non-binomial"),
        "error should explain flexible links are binomial-only, got: {msg}"
    );
}

#[test]
fn timewiggle_still_accepted_in_survival_formula() {
    // Guard must not regress the legitimate survival path: a Surv(...)
    // response still consumes timewiggle(...) without hitting the
    // non-survival rejection. We assert it does not error with the
    // non-survival "only supported in the main survival formula" message.
    let data = load_survival_dataset();
    let result = materialize(
        "Surv(entry, exit, event) ~ x + timewiggle(internal_knots=2)",
        &data,
        &FitConfig::default(),
    );
    if let Err(err) = result {
        let msg = err.to_string();
        assert!(
            !(msg.contains("timewiggle(...)") && msg.contains("meaningless")),
            "survival timewiggle wrongly rejected by the non-survival guard: {msg}"
        );
    }
}

// ---- #430 location-scale wiggle-pilot unification: parity tests ---------
//
// The Gaussian and binomial location-scale model entry points are now thin
// adapters over the single `fit_location_scale_with_optional_wiggle` engine.
// The tests below pin that the unified engine reproduces, coefficient for
// coefficient, the exact per-family reference sequence it replaced — both
// with and without a wiggle config — so the deslop cannot silently change
// any fitted result. The reference replays the *old* hand-rolled flow
// (pilot fit → select link-wiggle basis from the pilot → refit with that
// basis → extract `beta_link_wiggle` from block 2) directly against the
// family functions, with no shared code path with the engine other than
// those leaf family functions.

fn gaussian_location_scale_dataset() -> Dataset {
    // A mildly heteroscedastic, monotone-in-x signal with enough rows for a
    // stable mean+scale fit and a small wiggle basis.
    let n = 48usize;
    let mut records: Vec<csv::StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let x = -2.0 + 4.0 * (i as f64) / ((n - 1) as f64);
        // Deterministic, smooth response; the σ-model is intercept-only so
        // the test stays small while still exercising both blocks.
        let y = 0.7 * x + 0.3 * (1.3 * x).sin();
        records.push(csv::StringRecord::from(vec![
            format!("{y:.17e}"),
            format!("{x:.17e}"),
        ]));
    }
    gam_data::encode_recordswith_inferred_schema(vec!["y".to_string(), "x".to_string()], records)
        .expect("encode gaussian location-scale dataset")
}

fn binomial_location_scale_dataset() -> Dataset {
    // Balanced 0/1 response with a clear monotone gradient in x so the
    // threshold/log-σ blocks are well posed.
    let n = 60usize;
    let mut records: Vec<csv::StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let x = -2.0 + 4.0 * (i as f64) / ((n - 1) as f64);
        let y = if i % 2 == 0 { 1.0 } else { 0.0 };
        records.push(csv::StringRecord::from(vec![
            format!("{y:.17e}"),
            format!("{x:.17e}"),
        ]));
    }
    gam_data::encode_recordswith_inferred_schema(vec!["y".to_string(), "x".to_string()], records)
        .expect("encode binomial location-scale dataset")
}

fn small_wiggle_cfg() -> LinkWiggleConfig {
    LinkWiggleConfig {
        degree: 3,
        num_internal_knots: 3,
        penalty_orders: vec![2],
        double_penalty: false,
    }
}

fn assert_block_states_match(label: &str, lhs: &UnifiedFitResult, rhs: &UnifiedFitResult) {
    assert_eq!(
        lhs.block_states.len(),
        rhs.block_states.len(),
        "{label}: block count mismatch (engine {} vs reference {})",
        lhs.block_states.len(),
        rhs.block_states.len()
    );
    for (i, (a, b)) in lhs
        .block_states
        .iter()
        .zip(rhs.block_states.iter())
        .enumerate()
    {
        assert_eq!(
            a.beta.len(),
            b.beta.len(),
            "{label}: block {i} coefficient length mismatch"
        );
        for (j, (&av, &bv)) in a.beta.iter().zip(b.beta.iter()).enumerate() {
            // The engine and reference share the same leaf family functions
            // and feed them identical inputs, so the fitted coefficients
            // must agree to full numerical precision — this is a refactor,
            // not an approximation. A loose tolerance here would let a real
            // orchestration bug slip through, so the bound stays at the
            // bit-noise floor of an exact replay.
            assert!(
                (av - bv).abs() <= 1e-12 * (1.0 + bv.abs()),
                "{label}: block {i} coef {j} diverged: engine {av:.17e} vs reference {bv:.17e}"
            );
        }
    }
}

fn assert_beta_link_wiggle_match(
    label: &str,
    engine: &Option<Vec<f64>>,
    reference: &Option<Vec<f64>>,
) {
    match (engine, reference) {
        (Some(e), Some(r)) => {
            assert_eq!(
                e.len(),
                r.len(),
                "{label}: beta_link_wiggle length mismatch (engine {} vs reference {})",
                e.len(),
                r.len()
            );
            for (j, (&ev, &rv)) in e.iter().zip(r.iter()).enumerate() {
                // Same exact-replay floor as the block-state comparison: the
                // engine reads block 2 off the very fit the reference refit
                // produced, so any divergence beyond bit noise is a bug.
                assert!(
                    (ev - rv).abs() <= 1e-12 * (1.0 + rv.abs()),
                    "{label}: beta_link_wiggle coef {j} diverged: \
                         engine {ev:.17e} vs reference {rv:.17e}"
                );
            }
        }
        (None, None) => {}
        (e, r) => panic!(
            "{label}: beta_link_wiggle presence mismatch (engine is_some={}, reference is_some={})",
            e.is_some(),
            r.is_some()
        ),
    }
}

/// Standardize a Gaussian location-scale spec by the same response factor the
/// engine applies internally (`fit_gaussian_location_scale_model`): fit on
/// `y / s` and `mean_offset / s` so the fixed log-σ soft floor is scale-relative
/// (#884). Returns the factor `s` used (1.0 ⇒ no standardization needed).
///
/// This lets the reference flow exercise the *identical* model contract as the
/// engine — both standardize, fit the longhand pilot/refit terms, then rescale
/// back to raw units — so the engine-vs-reference equivalence stays an honest
/// orchestration check rather than comparing two different σ-floor models.
fn standardize_gaussian_spec_like_engine(spec: &mut GaussianLocationScaleTermSpec) -> f64 {
    let s = gaussian_response_sample_std(spec.y.view()).max(1e-6);
    if s != 1.0 {
        spec.y.mapv_inplace(|v| v / s);
        spec.mean_offset.mapv_inplace(|v| v / s);
    }
    s
}

/// Reference Gaussian no-wiggle fit through the longhand terms path, wrapped in
/// the same standardize→fit→rescale envelope the engine wrapper applies.
fn reference_gaussian_no_wiggle(
    data: ArrayView2<'_, f64>,
    mut spec: GaussianLocationScaleTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> GaussianLocationScaleFitResult {
    let s = standardize_gaussian_spec_like_engine(&mut spec);
    let fit = fit_gaussian_location_scale_terms(data, spec, options, kappa_options)
        .expect("reference gaussian no-wiggle terms fit");
    let mut result = GaussianLocationScaleFitResult {
        fit,
        wiggle_knots: None,
        wiggle_degree: None,
        beta_link_wiggle: None,
        response_scale: 1.0,
    };
    rescale_gaussian_location_scale_to_raw(&mut result, s);
    result
}

/// Reference Gaussian wiggle fit (pilot → basis selection → refit → assemble)
/// through the longhand terms path, under the engine's standardization envelope.
fn reference_gaussian_wiggle(
    data: ArrayView2<'_, f64>,
    mut spec: GaussianLocationScaleTermSpec,
    wiggle_cfg: &LinkWiggleConfig,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> GaussianLocationScaleFitResult {
    let s = standardize_gaussian_spec_like_engine(&mut spec);
    let ref_pilot = fit_gaussian_location_scale_terms(data, spec.clone(), options, kappa_options)
        .expect("reference gaussian pilot");
    let ref_basis = select_gaussian_location_scale_link_wiggle_basis_from_pilot(
        &ref_pilot,
        &WiggleBlockConfig {
            degree: wiggle_cfg.degree,
            num_internal_knots: wiggle_cfg.num_internal_knots,
            penalty_order: 2,
            double_penalty: wiggle_cfg.double_penalty,
        },
        &wiggle_cfg.penalty_orders,
    )
    .expect("reference gaussian wiggle basis selection");
    let ref_solved = fit_gaussian_location_scale_terms_with_selected_wiggle(
        data,
        spec,
        ref_basis,
        options,
        kappa_options,
    )
    .expect("reference gaussian wiggle refit");

    let beta_link_wiggle = ref_solved
        .fit
        .fit
        .block_states
        .get(2)
        .map(|b| b.beta.to_vec());
    let mut result = GaussianLocationScaleFitResult {
        fit: ref_solved.fit,
        wiggle_knots: Some(ref_solved.wiggle_knots),
        wiggle_degree: Some(ref_solved.wiggle_degree),
        beta_link_wiggle,
        response_scale: 1.0,
    };
    rescale_gaussian_location_scale_to_raw(&mut result, s);
    result
}

/// #2386: predict on a saved Gaussian location-scale fit revalidates through
/// `UnifiedFitResult::try_from_parts`, which requires the inference-block
/// covariance copies to be **bitwise equal** to their top-level twins. The
/// raw-units remap therefore must move every copy through the identical
/// congruence — a copy left in standardized units made every location-scale
/// predict refuse with "inference corrected covariance must match top-level
/// covariance_corrected" once #2346 began publishing the corrected matrix.
#[test]
fn gaussian_location_scale_raw_remap_keeps_inference_covariance_copies_bitwise_equal_2386() {
    // A response in the hundreds so the standardization factor `s` is far from
    // 1 and the raw remap is a strong, unmistakable congruence (D ≠ I).
    let n = 48usize;
    let mut records: Vec<csv::StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let x = -2.0 + 4.0 * (i as f64) / ((n - 1) as f64);
        let y = 140.0 * (0.7 * x + 0.3 * (1.3 * x).sin());
        records.push(csv::StringRecord::from(vec![
            format!("{y:.17e}"),
            format!("{x:.17e}"),
        ]));
    }
    let data =
        gam_data::encode_recordswith_inferred_schema(vec!["y".to_string(), "x".to_string()], records)
            .expect("encode scaled gaussian location-scale dataset");
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1".to_string()),
        ..FitConfig::default()
    };
    let materialized =
        materialize("y ~ x", &data, &config).expect("gaussian location-scale materialization");
    let FitRequest::GaussianLocationScale(request) = materialized.request else {
        panic!("expected a Gaussian location-scale request");
    };
    let GaussianLocationScaleFitRequest {
        data: req_data,
        spec,
        options,
        kappa_options,
        ..
    } = request;

    // Fit in *standardized* units (the engine's internal state before the raw
    // remap) so the remap under test is applied exactly once, by this test.
    let mut spec = spec;
    let s = standardize_gaussian_spec_like_engine(&mut spec);
    assert!(
        (s - 1.0).abs() > 10.0,
        "fixture response scale must make the remap non-trivial, got s={s}"
    );
    let fit = fit_gaussian_location_scale_terms(req_data, spec, &options, &kappa_options)
        .expect("standardized gaussian location-scale terms fit");
    let mut result = GaussianLocationScaleFitResult {
        fit,
        wiggle_knots: None,
        wiggle_degree: None,
        beta_link_wiggle: None,
        response_scale: 1.0,
    };

    // Install the #2346-shaped covariance state: the corrected matrix mirrored
    // bitwise at the inference level and the top level, exactly as the
    // custom-family assembly publishes it, plus a conditional-copy pair.
    let p = result.fit.fit.beta.len();
    assert!(p > 0, "joint coefficient vector must be non-empty");
    let mut corrected = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            corrected[[i, j]] = if i == j {
                1.5 + i as f64
            } else {
                0.25 / (1.0 + (i as f64 - j as f64).abs())
            };
        }
    }
    let conditional = corrected.mapv(|v| 0.5 * v);
    result.fit.fit.covariance_conditional = Some(conditional.clone());
    result.fit.fit.covariance_corrected = Some(corrected.clone());
    {
        let inference = result
            .fit
            .fit
            .inference
            .as_mut()
            .expect("terms fit must carry an inference block");
        inference.beta_covariance = Some(conditional.clone().into());
        inference.beta_covariance_corrected = Some(corrected.clone());
        inference.beta_standard_errors_corrected =
            Some(corrected.diag().mapv(|v| v.max(0.0).sqrt()));
        inference.smoothing_correction = Some(&corrected - &conditional);
        inference.smoothing_correction_method = Some(
            gam_solve::model_types::SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                active_rank: 1,
                rho_dimension: 1,
            },
        );
    }

    rescale_gaussian_location_scale_to_raw(&mut result, s);

    let fit = &result.fit.fit;
    let top_conditional = fit
        .covariance_conditional
        .as_ref()
        .expect("top-level conditional covariance survives the remap");
    let top_corrected = fit
        .covariance_corrected
        .as_ref()
        .expect("top-level corrected covariance survives the remap");
    let inference = fit.inference.as_ref().expect("inference block survives");

    // The remap must actually have moved the matrices (non-identity D)...
    assert!(
        (top_corrected[[0, 0]] - corrected[[0, 0]]).abs() > 1e-12,
        "remap with s={s} must rescale the corrected covariance"
    );
    // ...and every inference copy must land bitwise on its top-level twin,
    // which is exactly what the predict-time revalidation requires.
    assert_eq!(
        inference
            .beta_covariance
            .as_ref()
            .expect("inference conditional copy survives")
            .as_array(),
        top_conditional,
        "inference conditional covariance must ride the raw remap bitwise (#2386)"
    );
    assert_eq!(
        inference
            .beta_covariance_corrected
            .as_ref()
            .expect("inference corrected copy survives"),
        top_corrected,
        "inference corrected covariance must ride the raw remap bitwise (#2386)"
    );
    // The corrected decomposition Vp = Vb + C must keep holding in raw units:
    // both sides ride the same congruence, so their difference is the remapped
    // correction matrix.
    let correction = inference
        .smoothing_correction
        .as_ref()
        .expect("correction matrix survives");
    let recomposed = top_conditional + correction;
    for i in 0..p {
        for j in 0..p {
            let expected = top_corrected[[i, j]];
            assert!(
                (recomposed[[i, j]] - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
                "Vp = Vb + C must survive the remap at ({i},{j}): {} vs {expected}",
                recomposed[[i, j]]
            );
        }
    }
    // Corrected SEs are the remapped per-coordinate scale of the corrected
    // diagonal: se_raw_i = f_i * se_i with f_i > 0, so se_raw_i^2 must equal
    // the corrected diagonal exactly up to float regrouping.
    let se = inference
        .beta_standard_errors_corrected
        .as_ref()
        .expect("corrected SEs survive");
    assert_eq!(se.len(), p);
    for i in 0..p {
        let expected = top_corrected[[i, i]].max(0.0).sqrt();
        assert!(
            (se[i] - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
            "corrected SE {i} must track the remapped corrected diagonal: {} vs {expected}",
            se[i]
        );
    }
}

#[test]
fn gaussian_location_scale_engine_matches_reference_flow() {
    let data = gaussian_location_scale_dataset();
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1".to_string()),
        ..FitConfig::default()
    };
    let materialized =
        materialize("y ~ x", &data, &config).expect("gaussian location-scale materialization");
    let FitRequest::GaussianLocationScale(request) = materialized.request else {
        panic!("expected a Gaussian location-scale request");
    };
    let GaussianLocationScaleFitRequest {
        data: req_data,
        spec,
        options,
        kappa_options,
        ..
    } = request;

    // --- no-wiggle parity ------------------------------------------------
    let engine_plain = fit_gaussian_location_scale_model(GaussianLocationScaleFitRequest {
        data: req_data,
        spec: spec.clone(),
        wiggle: None,
        options: options.clone(),
        kappa_options: kappa_options.clone(),
    })
    .expect("engine gaussian no-wiggle fit");
    let reference_plain =
        reference_gaussian_no_wiggle(req_data, spec.clone(), &options, &kappa_options);
    assert_block_states_match(
        "gaussian/no-wiggle",
        &engine_plain.fit.fit,
        &reference_plain.fit.fit,
    );
    assert!(engine_plain.wiggle_knots.is_none());
    assert!(engine_plain.wiggle_degree.is_none());
    assert!(engine_plain.beta_link_wiggle.is_none());

    // --- wiggle parity ---------------------------------------------------
    let wiggle_cfg = small_wiggle_cfg();
    let engine_wiggle = fit_gaussian_location_scale_model(GaussianLocationScaleFitRequest {
        data: req_data,
        spec: spec.clone(),
        wiggle: Some(wiggle_cfg.clone()),
        options: options.clone(),
        kappa_options: kappa_options.clone(),
    })
    .expect("engine gaussian wiggle fit");

    // Reference: the exact pre-unification hand-rolled sequence, wrapped in the
    // same standardize→fit→rescale envelope the engine applies (#884), so the
    // two paths compare the same σ-floor model rather than diverging on the
    // raw-vs-scale-relative floor.
    let ref_solved = reference_gaussian_wiggle(
        req_data,
        spec.clone(),
        &wiggle_cfg,
        &options,
        &kappa_options,
    );

    assert_block_states_match(
        "gaussian/wiggle",
        &engine_wiggle.fit.fit,
        &ref_solved.fit.fit,
    );
    assert_eq!(
        engine_wiggle.wiggle_degree, ref_solved.wiggle_degree,
        "gaussian wiggle degree must match the reference refit"
    );
    let engine_knots = engine_wiggle
        .wiggle_knots
        .as_ref()
        .expect("engine gaussian wiggle knots present");
    let ref_knots = ref_solved
        .wiggle_knots
        .as_ref()
        .expect("reference gaussian wiggle knots present");
    assert_eq!(
        engine_knots.len(),
        ref_knots.len(),
        "gaussian wiggle knot count must match the reference refit"
    );
    for (k, (&ek, &rk)) in engine_knots.iter().zip(ref_knots.iter()).enumerate() {
        assert!(
            (ek - rk).abs() <= 1e-12 * (1.0 + rk.abs()),
            "gaussian wiggle knot {k} diverged: engine {ek:.17e} vs reference {rk:.17e}"
        );
    }
    // `beta_link_wiggle` is block 2 of the refit; the engine must extract it
    // exactly as the reference would read it off the same fit.
    let ref_beta_link_wiggle = ref_solved.beta_link_wiggle.clone();
    assert_beta_link_wiggle_match(
        "gaussian",
        &engine_wiggle.beta_link_wiggle,
        &ref_beta_link_wiggle,
    );
    assert!(
        engine_wiggle.beta_link_wiggle.is_some(),
        "a wiggle refit must populate beta_link_wiggle (block 2 present)"
    );
}

#[test]
fn binomial_location_scale_engine_matches_reference_flow() {
    let data = binomial_location_scale_dataset();
    let config = FitConfig {
        family: Some("binomial".to_string()),
        noise_formula: Some("1".to_string()),
        ..FitConfig::default()
    };
    let materialized =
        materialize("y ~ x", &data, &config).expect("binomial location-scale materialization");
    let FitRequest::BinomialLocationScale(request) = materialized.request else {
        panic!("expected a binomial location-scale request");
    };
    let BinomialLocationScaleFitRequest {
        data: req_data,
        spec,
        options,
        kappa_options,
        ..
    } = request;

    // --- no-wiggle parity ------------------------------------------------
    let engine_plain = fit_binomial_location_scale_model(BinomialLocationScaleFitRequest {
        data: req_data,
        spec: spec.clone(),
        wiggle: None,
        options: options.clone(),
        kappa_options: kappa_options.clone(),
    })
    .expect("engine binomial no-wiggle fit");
    let reference_plain =
        fit_binomial_location_scale_terms(req_data, spec.clone(), &options, &kappa_options)
            .expect("reference binomial no-wiggle fit");
    assert_block_states_match(
        "binomial/no-wiggle",
        &engine_plain.fit.fit,
        &reference_plain.fit,
    );
    assert!(engine_plain.wiggle_knots.is_none());
    assert!(engine_plain.wiggle_degree.is_none());
    assert!(engine_plain.beta_link_wiggle.is_none());

    // --- wiggle parity ---------------------------------------------------
    let wiggle_cfg = small_wiggle_cfg();
    let engine_wiggle = fit_binomial_location_scale_model(BinomialLocationScaleFitRequest {
        data: req_data,
        spec: spec.clone(),
        wiggle: Some(wiggle_cfg.clone()),
        options: options.clone(),
        kappa_options: kappa_options.clone(),
    })
    .expect("engine binomial wiggle fit");

    // Reference: the exact pre-unification hand-rolled sequence, including
    // the binomial-only link compatibility guard.
    require_inverse_link_supports_joint_wiggle(
        &spec.link_kind,
        "binomial location-scale link wiggle",
    )
    .expect("logit link supports joint wiggle");
    let ref_pilot =
        fit_binomial_location_scale_terms(req_data, spec.clone(), &options, &kappa_options)
            .expect("reference binomial pilot");
    let ref_basis = select_binomial_location_scale_link_wiggle_basis_from_pilot(
        &ref_pilot,
        &WiggleBlockConfig {
            degree: wiggle_cfg.degree,
            num_internal_knots: wiggle_cfg.num_internal_knots,
            penalty_order: 2,
            double_penalty: wiggle_cfg.double_penalty,
        },
        &wiggle_cfg.penalty_orders,
    )
    .expect("reference binomial wiggle basis selection");
    let ref_solved = fit_binomial_location_scale_terms_with_selected_wiggle(
        req_data,
        spec.clone(),
        ref_basis,
        &options,
        &kappa_options,
    )
    .expect("reference binomial wiggle refit");

    assert_block_states_match(
        "binomial/wiggle",
        &engine_wiggle.fit.fit,
        &ref_solved.fit.fit,
    );
    assert_eq!(
        engine_wiggle.wiggle_degree,
        Some(ref_solved.wiggle_degree),
        "binomial wiggle degree must match the reference refit"
    );
    let engine_knots = engine_wiggle
        .wiggle_knots
        .as_ref()
        .expect("engine binomial wiggle knots present");
    assert_eq!(
        engine_knots.len(),
        ref_solved.wiggle_knots.len(),
        "binomial wiggle knot count must match the reference refit"
    );
    for (k, (&ek, &rk)) in engine_knots
        .iter()
        .zip(ref_solved.wiggle_knots.iter())
        .enumerate()
    {
        assert!(
            (ek - rk).abs() <= 1e-12 * (1.0 + rk.abs()),
            "binomial wiggle knot {k} diverged: engine {ek:.17e} vs reference {rk:.17e}"
        );
    }
    let ref_beta_link_wiggle = ref_solved
        .fit
        .fit
        .block_states
        .get(2)
        .map(|b| b.beta.to_vec());
    assert_beta_link_wiggle_match(
        "binomial",
        &engine_wiggle.beta_link_wiggle,
        &ref_beta_link_wiggle,
    );
    assert!(
        engine_wiggle.beta_link_wiggle.is_some(),
        "a wiggle refit must populate beta_link_wiggle (block 2 present)"
    );
}

#[test]
fn resolve_family_accepts_mgcv_parenthesized_family_link_syntax() {
    // mgcv writes GLM families in R as `family(link)` — `binomial(logit)`,
    // `gaussian(identity)`, `Binomial(Probit)`. Three tests in-repo pass
    // `family: Some("binomial(logit)".to_string())` straight through to the
    // resolver (`sphere_logit_predict_finite_at_pole`, `sphere_binomial_*`),
    // and would otherwise be rejected as `unknown family`.
    use crate::fit_orchestration::resolve_family;
    use gam_problem::{
        InverseLink, LinkFunction, ResponseColumnKind, ResponseFamily, StandardLink,
    };
    let y = ndarray::array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    for raw in [
        "binomial(logit)",
        "Binomial(Logit)",
        "binomial(LOGIT)",
        "binomial( logit )",
        "binomial_logit",
        "binomial-logit",
    ] {
        let spec = resolve_family(
            Some(raw),
            None,
            None,
            y.view(),
            ResponseColumnKind::Numeric,
            "y",
        )
        .unwrap_or_else(|err| panic!("resolve_family({raw:?}) failed: {err}"));
        assert!(
            matches!(spec.response, ResponseFamily::Binomial),
            "{raw}: expected Binomial response"
        );
        assert_eq!(
            spec.link.link_function(),
            LinkFunction::Logit,
            "{raw}: expected logit link"
        );
    }
    let probit = resolve_family(
        Some("binomial(probit)"),
        None,
        None,
        y.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect("binomial(probit) resolves");
    assert_eq!(probit.link.link_function(), LinkFunction::Probit);
    let cloglog = resolve_family(
        Some("Binomial(CLogLog)"),
        None,
        None,
        y.view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect("binomial(cloglog) resolves");
    assert_eq!(cloglog.link.link_function(), LinkFunction::CLogLog);
    let nb = resolve_family(
        Some("negative_binomial(log)"),
        None,
        None,
        ndarray::array![0.0, 1.0, 2.0, 3.0].view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect("negative_binomial(log) resolves");
    assert!(matches!(
        nb.response,
        ResponseFamily::NegativeBinomial { .. }
    ));
    assert!(matches!(nb.link, InverseLink::Standard(StandardLink::Log)));
}

/// A strictly-increasing 1-D Gaussian dataset — the #1191 reproduce shape
/// (`y = sqrt(x) + small noise`) on which `s(x, shape=monotone_increasing)`
/// must fit. Deterministic (no RNG) so the parity assertion is exact.
fn monotone_parity_dataset() -> Dataset {
    let n = 60usize;
    let mut flat = Vec::with_capacity(n * 2);
    for i in 0..n {
        let x = (i as f64 + 0.5) / n as f64; // (0,1), strictly increasing
        // Deterministic tiny wiggle so the data is not perfectly smooth but
        // is unambiguously increasing; keeps the monotone constraint feasible.
        let y = x.sqrt() + 0.01 * ((7 * i) % 5) as f64 / 5.0;
        flat.push(x);
        flat.push(y);
    }
    Dataset {
        headers: vec!["x".to_string(), "y".to_string()],
        values: Array2::from_shape_vec((n, 2), flat).expect("monotone parity data shape"),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "x".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "y".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![ColumnKindTag::Continuous, ColumnKindTag::Continuous],
    }
}

/// #1196 structural-parity guard. The `gam` CLI (`run_fit`) and the
/// formula/Python (`materialize_standard`) entry points must build the SAME
/// outer-REML `FitOptions` policy for the same model. Both now route through
/// `canonical_standard_fit_options`; this test reconstructs the CLI-side call
/// with the CLI's request-specific inputs and asserts the resulting options are
/// byte-for-byte identical (Debug form, since `FitOptions` is not `PartialEq`)
/// to the options `materialize` actually puts on the `StandardFitRequest`.
/// Before #1196 the CLI used `tol: 1e-6` / `skip_rho_posterior_inference:
/// false` while the formula path used `1e-10`/`true`, so this would have
/// diverged — the exact class of defect #1191 exposed.
#[test]
fn issue_1196_cli_and_formula_standard_fit_options_match() {
    let data = monotone_parity_dataset();
    let config = FitConfig::default();
    let formula = "y ~ s(x, shape=monotone_increasing)";

    let materialized =
        materialize(formula, &data, &config).expect("formula path materializes the monotone fit");
    let FitRequest::Standard(request) = materialized.request else {
        panic!("expected a standard request for a Gaussian shape-constrained smooth");
    };

    // Reconstruct the CLI's call: the CLI passes only request-specific inputs
    // (here: no mixture/SAS link, Firth off, no adaptive regularization), the
    // same set `run_fit` feeds for an ordinary Gaussian smooth.
    let cli_options = crate::fit_orchestration::canonical_standard_fit_options(
        &config,
        crate::fit_orchestration::StandardFitOptionsInputs {
            firth_bias_reduction: config.firth,
            ..Default::default()
        },
    );

    assert_eq!(
        format!("{:#?}", request.options),
        format!("{cli_options:#?}"),
        "CLI and formula entry points must build identical standard FitOptions (#1196)"
    );

    // The policy fields that diverged pre-#1196 are now the single-sourced
    // canonical values for BOTH paths.
    assert!(
        request.options.skip_rho_posterior_inference,
        "canonical formula/CLI policy skips the live-rho posterior path"
    );
    assert_eq!(
        request.options.tol, 1e-10,
        "canonical outer-REML tolerance is the gam#893 value, not the stale CLI 1e-6"
    );
}

/// #1191 regression, structural form: the shape-constrained smooth that the
/// CLI fit but `gamfit.fit` rejected must now fit through the SHARED driver
/// (`materialize` + `fit_model`) that the Python path uses — no "no candidate
/// seeds passed outer startup validation" ALO-NaN rejection. Because the CLI
/// and Python now share this exact driver, a pass here is a pass for both.
#[test]
fn issue_1191_shape_constrained_monotone_fits_through_shared_driver() {
    let data = monotone_parity_dataset();
    let config = FitConfig::default();
    let formula = "y ~ s(x, shape=monotone_increasing)";

    let materialized = materialize(formula, &data, &config)
        .expect("monotone shape-constrained smooth materializes");
    let result = fit_model(materialized.request)
        .expect("monotone shape-constrained smooth fits through the shared driver (#1191)");
    let FitResult::Standard(standard) = result else {
        panic!("expected a standard fit result");
    };
    // A genuine converged fit, not a degenerate seed-rejection escape.
    let beta = standard
        .fit
        .block_by_role(gam_problem::BlockRole::Mean)
        .expect("fitted mean block")
        .beta
        .clone();
    assert!(
        beta.iter().all(|b| b.is_finite()),
        "fitted coefficients must be finite (no ALO-NaN seed rejection)"
    );
    let fitted = standard.design.design.to_dense().dot(&beta);
    for row in 1..fitted.len() {
        assert!(
            fitted[row] + 1e-10 >= fitted[row - 1],
            "monotone-increasing fit decreased between sorted rows {} and {}: {} -> {}",
            row - 1,
            row,
            fitted[row - 1],
            fitted[row]
        );
    }
}

/// Regression for #1767: a non-default `survival_likelihood` on a non-survival
/// response (no `Surv(...)` wrapper) used to be silently discarded, degrading
/// the requested survival model to an ordinary Gaussian GAM. It must now error.
fn nonsurvival_gaussian_dataset() -> Dataset {
    // A simple smooth signal with enough rows for a stable `s(x)` fit. The
    // response column is named `time` to mirror the issue's
    // `time ~ s(x)` formula (a bare column, *not* `Surv(time, event)`).
    let n = 48usize;
    let mut records: Vec<csv::StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let x = -2.0 + 4.0 * (i as f64) / ((n - 1) as f64);
        let time = 0.7 * x + 0.3 * (1.3 * x).sin();
        records.push(csv::StringRecord::from(vec![
            format!("{time:.17e}"),
            format!("{x:.17e}"),
        ]));
    }
    gam_data::encode_recordswith_inferred_schema(vec!["time".to_string(), "x".to_string()], records)
        .expect("encode non-survival gaussian dataset")
}

#[test]
fn survival_likelihood_rejected_on_nonsurvival_response() {
    let data = nonsurvival_gaussian_dataset();
    let mut config = FitConfig::default();
    // Explicitly request a survival likelihood mode *without* a Surv(...) LHS.
    config.survival_likelihood = Some("weibull".to_string());

    let err = materialize("time ~ s(x)", &data, &config)
        .err()
        .expect("a non-default survival_likelihood on a non-survival response must error (#1767)");

    let msg = err.to_string();
    assert!(
        msg.contains("survival_likelihood"),
        "error must name the offending knob, got: {msg}"
    );
    assert!(
        msg.contains("Surv(...)"),
        "error must point the user at the Surv(...) wrapper, got: {msg}"
    );
}

#[test]
fn default_survival_likelihood_allowed_on_nonsurvival_response() {
    // Positive control: the default survival_likelihood is now `None` (unset) —
    // there is no library-side string default (#2301). `None` is unambiguously
    // "unset" and must NOT be rejected on a non-survival response, so the guard
    // isn't over-broad. The single canonical default (`"transformation"`) is
    // resolved only at the `Surv(...)` seam, which a non-survival fit never hits.
    let data = nonsurvival_gaussian_dataset();
    let config = FitConfig::default();
    assert_eq!(config.survival_likelihood, None);
    assert_eq!(config.resolved_survival_likelihood(), "transformation");

    materialize("time ~ s(x)", &data, &config)
        .expect("default survival_likelihood must still materialize an ordinary GAM (#1767)");
}
