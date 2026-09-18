//! gam#2718 / gam#2484 — what a BMS fit does when its latent-z adequacy gate
//! rejects StandardNormal.
//!
//! The answer changed once, and the change is the point of this file. gam#2718
//! established the CONTRACT — mint the point estimates, and if the covariance
//! cannot be corrected, withhold it and DECLARE the withholding rather than
//! publish an uncorrected (too narrow) matrix that is indistinguishable on the
//! wire from a corrected one. gam#2484 then removed the reason the reachable
//! fixture was withholding at all: the cross-row channel through the empirical
//! grid has a closed form, so the ordinary rigid `global-empirical` fit now
//! PUBLISHES a corrected covariance. Both halves are asserted here, on the same
//! fixtures, so neither can be lost to the other.
//!
//! Before this contract the seam returned `Err` from
//! `fit_bernoulli_marginal_slope_terms`, so a conditional location-scale
//! calibration whose calibrated residual failed the standard-normal adequacy
//! gate destroyed the whole fit — point estimates included — three stages after
//! the decision that caused it. The state is a legitimate point-estimation
//! result: the measure-selection warning has always said so. It was
//! unreachable in practice because both production marginal-slope entry points
//! set `compute_covariance = true` unconditionally.
//!
//! Three arms, because no one of them is satisfiable alone:
//!
//! * **corrected** — the PRS/PC-confounded fixture routes through the
//!   conditional calibration, fails the gate (measured: |skew| x16.5,
//!   |excess kurtosis| x10.1, KS x6.40), selects `global-empirical`, and must
//!   now come back with finite coefficients AND a published covariance AND no
//!   declaration. This is gam#2484's acceptance.
//! * **still withheld** — the withholding contract itself, asserted on the
//!   payload rather than on a fit. Its end-to-end witness moved when gam#2484
//!   landed, and the obvious replacement (the same fixture with a
//!   `linkwiggle(...)` score-warp block) was built and MEASURED: it does not
//!   complete in 90 minutes at n=96, against 22 for the whole file before. So
//!   WHICH shapes withhold is unit-tested on the classifier with no fit in the
//!   loop, and THAT a withholding survives the wire is asserted here.
//! * **negative** — the rank-reduced `centers=60` fixture reaches
//!   `StandardNormal` via rank-INT and must declare NOTHING. A declaration
//!   there would mean the gate fires on fits that have a valid covariance,
//!   which is the failure mode that makes the whole channel meaningless.
//!
//! Publishing the UNCORRECTED covariance is not, and never becomes, the
//! alternative to withholding: it omits the first-stage generated-regressor
//! uncertainty, so the intervals come out too narrow. What gam#2484 publishes
//! is the CORRECTED one.

use gam::data::EncodedDataset;
use gam::estimate::CovarianceDeclined;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::{FitConfig, FitResult, fit_from_formula};
use ndarray::Array2;

fn binary_outcome_shape_dataset() -> EncodedDataset {
    let n = 96usize;
    let headers = vec![
        "event",
        "sex",
        "entry_age_z",
        "current_age_ns_1",
        "current_age_ns_2",
        "current_age_ns_3",
        "current_age_ns_4",
        "prs_z",
        "PC1",
        "PC2",
        "PC3",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    let mut values = Vec::<f64>::with_capacity(n * headers.len());
    for i in 0..n {
        let sex = if i % 2 == 0 { 0.0 } else { 1.0 };
        let entry_age_z = (i as f64 - 47.5) / 18.0;
        let t = ((i % 19) as f64 - 9.0) / 9.0;
        let current_age_ns_1 = 1.0;
        let current_age_ns_2 = t;
        let current_age_ns_3 = t * t;
        let current_age_ns_4 = t * t * t;
        let prs_z = (((i * 37) % 101) as f64 - 50.0) / 22.0;
        let pc1 = ((i as f64) * 0.17).sin();
        let pc2 = ((i as f64) * 0.23).cos();
        let pc3 = ((i as f64) * 0.31).sin() * ((i as f64) * 0.07).cos();
        let eta = -0.15 + 0.25 * sex + 0.18 * entry_age_z + 0.16 * t + 0.10 * prs_z + 0.08 * pc1;
        let deterministic_noise = (((i * 13) % 17) as f64 - 8.0) / 11.0;
        let event = if eta + deterministic_noise > 0.0 {
            1.0
        } else {
            0.0
        };
        values.extend_from_slice(&[
            event,
            sex,
            entry_age_z,
            current_age_ns_1,
            current_age_ns_2,
            current_age_ns_3,
            current_age_ns_4,
            prs_z,
            pc1,
            pc2,
            pc3,
        ]);
    }
    EncodedDataset {
        headers,
        values: Array2::from_shape_vec((n, 11), values)
            .expect("binary-outcome-shape BMS data shape"),
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
    }
}

fn duplicate_pc_binary_outcome_shape_dataset() -> EncodedDataset {
    let n = 160usize;
    let headers = vec![
        "event",
        "sex",
        "entry_age_z",
        "current_age_ns_1",
        "current_age_ns_2",
        "current_age_ns_3",
        "current_age_ns_4",
        "prs_z",
        "PC1",
        "PC2",
        "PC3",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    let pc_cloud = [
        [-1.0, -0.5, 0.25],
        [-0.25, 0.75, -0.5],
        [0.5, -0.75, 0.75],
        [1.0, 0.5, -0.25],
    ];
    let mut values = Vec::<f64>::with_capacity(n * headers.len());
    for i in 0..n {
        let sex = if i % 2 == 0 { 0.0 } else { 1.0 };
        let entry_age_z = (i as f64 - 79.5) / 30.0;
        let t = ((i % 29) as f64 - 14.0) / 14.0;
        let prs_z = (((i * 17) % 131) as f64 - 65.0) / 31.0;
        let pc = pc_cloud[i % pc_cloud.len()];
        let eta = -0.05 + 0.12 * sex + 0.08 * entry_age_z + 0.06 * t + 0.05 * prs_z + 0.04 * pc[0];
        let deterministic_noise = (((i * 11) % 23) as f64 - 11.0) / 10.0;
        let event = if eta + deterministic_noise > 0.0 {
            1.0
        } else {
            0.0
        };
        values.extend_from_slice(&[
            event,
            sex,
            entry_age_z,
            t,
            t * t,
            t * t * t,
            t * t * t * t,
            prs_z,
            pc[0],
            pc[1],
            pc[2],
        ]);
    }
    EncodedDataset {
        headers,
        values: Array2::from_shape_vec((n, 11), values)
            .expect("duplicate-PC binary-outcome-shape BMS data shape"),
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
    }
}

const MARGINAL_FORMULA_CENTERS6: &str = "event ~ matern(PC1, PC2, PC3, centers=6, length_scale=1.0) + sex + entry_age_z + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4";
const SLOPE_FORMULA_CENTERS6: &str = "matern(PC1, PC2, PC3, centers=6, length_scale=1.0)";
const MARGINAL_FORMULA_CENTERS60: &str = "event ~ matern(PC1, PC2, PC3, centers=60, length_scale=1.0) + sex + entry_age_z + current_age_ns_1 + current_age_ns_2 + current_age_ns_3 + current_age_ns_4";
const SLOPE_FORMULA_CENTERS60: &str = "matern(PC1, PC2, PC3, centers=60, length_scale=1.0)";

/// The witness fixture: `prs_z` is made an exact affine function of PC1/PC2, so
/// the conditional `E[z|C]`/`Var(z|C)` Rao gate fires and the calibrated
/// residual then fails the pooled standard-normal adequacy gate. Byte-identical
/// to the confound injection in
/// `tests/identifiability/misc/binary_outcome_bms_identifiability.rs`.
fn prs_pc_confounded_dataset() -> EncodedDataset {
    let mut data = binary_outcome_shape_dataset();
    let prs_idx = data
        .headers
        .iter()
        .position(|h| h == "prs_z")
        .expect("prs_z column");
    let pc1_idx = data
        .headers
        .iter()
        .position(|h| h == "PC1")
        .expect("PC1 column");
    let pc2_idx = data
        .headers
        .iter()
        .position(|h| h == "PC2")
        .expect("PC2 column");
    for row in 0..data.values.nrows() {
        data.values[[row, prs_idx]] =
            data.values[[row, pc1_idx]] + 0.1 * data.values[[row, pc2_idx]];
    }
    data
}

fn fit_bms(
    data: &EncodedDataset,
    marginal: &str,
    slope: &str,
    label: &str,
    latent_measure: Option<&str>,
) -> gam::families::bms::BernoulliMarginalSlopeFitResult {
    // A declared latent law names the marginal-slope family it is declared for.
    let cfg = FitConfig {
        family: latent_measure.map(|_| "bernoulli-marginal-slope".to_string()),
        slope_formula: Some(slope.to_string()),
        z_column: Some("prs_z".to_string()),
        latent_measure: latent_measure.map(str::to_string),
        ..FitConfig::default()
    };
    match fit_from_formula(marginal, data, &cfg) {
        Ok(FitResult::BernoulliMarginalSlope(out)) => out,
        Ok(_) => panic!("{label}: fit returned the wrong family variant"),
        Err(err) => panic!(
            "{label}: the fit must be MINTED. A latent measure the Murphy-Topel \
             correction cannot handle withholds the covariance; it does not \
             destroy the point estimates (gam#2718). Got: {err}"
        ),
    }
}

#[test]
fn bms_publishes_the_corrected_covariance_on_a_global_empirical_measure_2484() {
    gam::init_parallelism();
    let data = prs_pc_confounded_dataset();
    // gam#2926: the calibrated pair this correction is about is minted only by
    // the declared conditional location-scale law.
    let out = fit_bms(
        &data,
        MARGINAL_FORMULA_CENTERS6,
        SLOPE_FORMULA_CENTERS6,
        "prs/pc-confounded BMS fit",
        Some("conditional-location-scale"),
    );

    // 1. The point estimates are published. This never stopped being true, and
    //    it is what gam#2718 moved the refusal to protect.
    assert!(
        out.fit.beta.iter().all(|coef| coef.is_finite()),
        "gam#2718: the fit must publish finite coefficients, got {:?}",
        out.fit.beta
    );
    assert!(
        !out.fit.beta.is_empty(),
        "gam#2718: the published coefficient vector must be non-empty"
    );
    assert!(
        out.fit.log_lambdas.iter().all(|rho| rho.is_finite()),
        "gam#2718: finite smoothing parameters, got {:?}",
        out.fit.log_lambdas
    );

    // 2. Nothing is declared, because nothing was withheld. This is the
    //    assertion that inverts: before gam#2484 the same fixture reached here
    //    with a typed `CovarianceDeclined` and no covariance at all.
    assert!(
        out.fit.artifacts.covariance_declined.is_none(),
        "gam#2484: the rigid global-empirical measure has a cross-row channel now, so this fit \
         must be CORRECTED rather than declined; got {:?}",
        out.fit.artifacts.covariance_declined
    );

    // 3. The covariance is published, and it is a covariance: symmetric with a
    //    strictly positive diagonal. The Murphy-Topel term is a congruence of
    //    the PSD first-stage covariance, so adding it cannot break either.
    let covariance = out
        .fit
        .covariance_conditional
        .as_ref()
        .expect("gam#2484: the conditional covariance must be published");
    let p = covariance.nrows();
    assert_eq!(covariance.ncols(), p);
    for i in 0..p {
        assert!(
            covariance[[i, i]] > 0.0 && covariance[[i, i]].is_finite(),
            "gam#2484: diagonal {i} of the corrected covariance is {}",
            covariance[[i, i]]
        );
        for j in 0..p {
            let asymmetry = (covariance[[i, j]] - covariance[[j, i]]).abs();
            let scale = 1.0 + covariance[[i, i]].abs().max(covariance[[j, j]].abs());
            assert!(
                asymmetry <= 1.0e-9 * scale,
                "gam#2484: the corrected covariance must stay symmetric; ({i},{j}) differs by \
                 {asymmetry:.3e}"
            );
        }
    }

    // 4. And the derived surfaces are populated, not just the matrix. A
    //    consumer reads standard errors, not the covariance.
    let ses = out
        .fit
        .beta_standard_errors()
        .expect("gam#2484: standard errors must be published alongside the covariance");
    assert!(
        ses.iter().all(|se| se.is_finite() && *se >= 0.0),
        "gam#2484: published standard errors must be finite and non-negative, got {ses:?}"
    );

    // 5. gam#2943: the published standard errors are the published covariance's,
    //    for both pairs. The correction used to reach only the top-level matrices,
    //    so the standard errors kept the uncorrected diagonal and every load of the
    //    saved model refused the fit.
    for (label, standard_errors, covariance) in [
        ("conditional", out.fit.beta_standard_errors(), out.fit.beta_covariance()),
        (
            "corrected",
            out.fit.beta_standard_errors_corrected(),
            out.fit.beta_covariance_corrected(),
        ),
    ] {
        if let Some(standard_errors) = standard_errors {
            let covariance = covariance.unwrap_or_else(|| {
                panic!("gam#2943: {label} standard errors are published without their covariance")
            });
            for (i, (se, diagonal)) in standard_errors.iter().zip(covariance.diag().iter()).enumerate() {
                assert!(
                    (se * se - diagonal).abs() <= 1.0e-12 * diagonal.abs().max(1.0),
                    "gam#2943: {label} standard error {i} squares to {:.17e} against the covariance \
                     diagonal {diagonal:.17e}",
                    se * se
                );
            }
        }
    }
}

#[test]
fn bms_standard_normal_latent_measure_declares_nothing_2718() {
    // Negative control. Without it, an implementation that sets the
    // declaration unconditionally passes the positive arm above — and a
    // declaration on a fit that HAS a valid covariance is worse than no
    // channel at all, because it teaches consumers to ignore the field.
    gam::init_parallelism();
    let data = duplicate_pc_binary_outcome_shape_dataset();
    let out = fit_bms(
        &data,
        MARGINAL_FORMULA_CENTERS60,
        SLOPE_FORMULA_CENTERS60,
        "rank-reduced centers=60 BMS fit",
        None,
    );

    assert!(
        out.fit.beta.iter().all(|coef| coef.is_finite()),
        "control fixture must fit: got {:?}",
        out.fit.beta
    );
    assert!(
        out.fit.artifacts.covariance_declined.is_none(),
        "gam#2718: a fit that did NOT hit the non-StandardNormal seam must declare nothing; \
         got {:?}",
        out.fit.artifacts.covariance_declined
    );
}

#[test]
fn a_withheld_covariance_names_the_missing_channel_and_survives_the_wire_2718() {
    // gam#2718's contract, at the level it actually lives at.
    //
    // The end-to-end witness for withholding MOVED when gam#2484 landed: the
    // PRS/PC fixture above now publishes. The obvious replacement — the same
    // fixture with a `linkwiggle(...)` score-warp block — was built and
    // measured, and it does NOT complete: 90 minutes on n=96 without finishing
    // the four-arm file, against 22 minutes for the whole file before. A fit
    // nobody can run is not a regression gate, so the withholding contract is
    // asserted where it is cheap and exhaustive instead:
    //
    //   * WHICH shapes withhold, and why, is a pure decision over the measure,
    //     the build record and the flex flag. All four arms — closed-form,
    //     correctable, score-warp, local-empirical, non-differentiable data —
    //     are unit-tested directly in
    //     `crates/gam-models/src/bms/empirical_measure_2484_tests.rs`
    //     (`the_shapes_with_no_channel_are_still_withheld_and_say_which_2484`),
    //     with no fit in the loop.
    //   * That a withholding SURVIVES THE WIRE is this file's half, and it is a
    //     property of the payload, not of any particular fit. It is asserted
    //     here on the declaration itself.
    //
    // The pairing gam#2718 cared about — H ships, so the declaration must ship
    // beside it — is preserved by the arm below, which checks both on a real
    // fit.
    let declined = CovarianceDeclined::BmsGeneratedRegressorLatentMeasureNotStandardNormal {
        latent_measure: "global-empirical".to_string(),
        unavailable_channel: "a score-warp / link-deviation block evaluates a basis AT the \
                              latent score"
            .to_string(),
    };

    let explanation = declined.explain();
    assert!(
        explanation.contains("gam#2484"),
        "gam#2718: the explanation must point at the issue that owns the channel, got: \
         {explanation}"
    );
    assert!(
        explanation.contains("score-warp"),
        "gam#2484: the explanation must name the CHANNEL that is missing, not just the measure — \
         the measure alone no longer determines whether a correction exists. Got: {explanation}"
    );

    let mut artifacts = gam::estimate::FitArtifacts::default();
    artifacts.covariance_declined = Some(declined.clone());
    let encoded = serde_json::to_string(&artifacts).expect("gam#2718: artifacts must serialize");
    let decoded: gam::estimate::FitArtifacts =
        serde_json::from_str(&encoded).expect("gam#2718: artifacts must deserialize");
    assert_eq!(
        decoded.covariance_declined,
        Some(declined),
        "gam#2718: the declination must survive serialization. If this field ever becomes \
         `skip_serializing`, or a persistence route reassembles the fit without carrying \
         `artifacts`, a consumer loading the saved model gets H and phi with NO warning — which \
         is the silent consumption this channel exists to prevent."
    );

    // And an OLD payload — written before gam#2484 split the reason from the
    // channel — must still load, with the channel simply empty. A hard
    // deserialization failure there would lock consumers out of models they
    // could previously read.
    let legacy = r#"{"covariance_declined":{"reason":"bms-generated-regressor-latent-measure-not-standard-normal","latent_measure":"global-empirical"}}"#;
    let loaded: gam::estimate::FitArtifacts = serde_json::from_str(legacy)
        .expect("gam#2484: a pre-channel payload must still deserialize");
    match loaded.covariance_declined {
        Some(CovarianceDeclined::BmsGeneratedRegressorLatentMeasureNotStandardNormal {
            latent_measure,
            unavailable_channel,
        }) => {
            assert_eq!(latent_measure, "global-empirical");
            assert!(
                unavailable_channel.is_empty(),
                "gam#2484: an old payload carries no channel, so the field defaults empty rather \
                 than inventing one; got {unavailable_channel:?}"
            );
        }
        other => panic!("gam#2484: the legacy payload decoded to {other:?}"),
    }
}

#[test]
fn a_published_fit_ships_the_curvature_a_declination_would_be_about_2718() {
    // The other half of the pairing. gam#2718's worry was that a consumer
    // loading a saved fit gets the penalized Hessian `H` and the dispersion
    // `phi`, from which `Vb_naive = phi * H^-1` is one Cholesky away — so a
    // withheld covariance MUST travel with a declaration or the consumer never
    // receives one.
    //
    // That worry is only load-bearing because H really does ship. Asserted here
    // on the fixture gam#2484 made publish: H is present, the covariance is
    // present, and nothing is declared. If a future change ever puts this
    // fixture back into withholding, the arm above is what must then carry the
    // declaration across the wire.
    gam::init_parallelism();
    let data = prs_pc_confounded_dataset();
    let out = fit_bms(
        &data,
        MARGINAL_FORMULA_CENTERS6,
        SLOPE_FORMULA_CENTERS6,
        "prs/pc-confounded BMS fit (persistence arm)",
        Some("conditional-location-scale"),
    );

    assert!(
        out.fit.penalized_hessian().is_some(),
        "gam#2718: a BMS fit publishes the penalized Hessian — EDF accounting, posterior \
         whitening and prediction all read it. This is why a WITHHELD covariance has to ship a \
         declaration beside it."
    );
    assert!(
        out.fit.covariance_conditional.is_some(),
        "gam#2484: this fixture is corrected, so it publishes a covariance"
    );
    assert!(
        out.fit.artifacts.covariance_declined.is_none(),
        "gam#2484: and therefore declares nothing; got {:?}",
        out.fit.artifacts.covariance_declined
    );

    // The artifacts still round-trip, corrected or not — the wire is the same
    // wire either way.
    let encoded =
        serde_json::to_string(&out.fit.artifacts).expect("gam#2718: fit artifacts must serialize");
    let decoded: gam::estimate::FitArtifacts =
        serde_json::from_str(&encoded).expect("gam#2718: fit artifacts must deserialize");
    assert_eq!(
        decoded.covariance_declined,
        out.fit.artifacts.covariance_declined
    );
}
