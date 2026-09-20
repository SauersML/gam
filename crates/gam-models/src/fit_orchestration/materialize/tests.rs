#![cfg(test)]

use super::*;
use gam_data::load_dataset_projected;
use gam_data::{ColumnKindTag, DataSchema, SchemaColumn};
use gam_terms::basis::{DuchonNullspaceOrder, center_strategy_is_auto, default_num_centers, starting_num_centers};
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
    config.slope_formula = Some("1".to_string());
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
fn survival_marginal_slope_materialize_rejects_z_column_in_slope_formula() {
    let data = load_survival_dataset();
    let mut config = FitConfig::default();
    config.survival_likelihood = Some("marginal-slope".to_string());
    config.slope_formula = Some("1 + z".to_string());
    config.z_column = Some("z".to_string());

    let err = materialize("Surv(entry, exit, event) ~ x", &data, &config)
        .err()
        .expect("slope formula should reject z-column reuse");

    assert!(
        err.to_string()
            .contains("survival marginal-slope reserves z column 'z'")
    );
    assert!(err.to_string().contains("slope_formula"));
}

#[test]
fn survival_marginal_slope_materialize_rejects_z_column_when_slope_defaults_to_main_spec() {
    let data = load_survival_dataset();
    let mut config = FitConfig::default();
    config.survival_likelihood = Some("marginal-slope".to_string());
    config.z_column = Some("z".to_string());

    let err = materialize("Surv(entry, exit, event) ~ x + z", &data, &config)
        .err()
        .expect("defaulted slope spec should still reject z-column reuse");

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

/// #2470 round trip: the survival time basis a payload persists must be the one
/// the fit was materialized with, never a second derivation from the same
/// `FitConfig`.
///
/// `materialize_survival` switches the time anchor to the robust interior
/// median exit whenever the data is left-truncated (#751/#1790), for EVERY
/// time-basis-carrying likelihood. The save path used to re-derive the anchor
/// with `resolve_survival_time_anchor_value`, which unconditionally takes the
/// earliest entry — so a left-truncated location-scale model persisted an
/// anchor its own fit never centered at, and `survival::predict` (which
/// re-centers the design at the persisted anchor for `LocationScale`) then
/// evaluated the basis in a different affine frame than the coefficients were
/// fitted in.
///
/// `MaterializedModel::survival_time_basis` carries the realised snapshot so
/// the two cannot disagree. Here the anchor must be the median exit (5.0), not
/// the earliest entry (0.5).
#[test]
fn materialized_survival_time_basis_carries_the_left_truncated_anchor_2470() {
    let td = tempdir().expect("tempdir");
    let data_path = td.path().join("left_truncated_locscale.csv");
    // Constant entry = 0.5 (genuine left truncation); nine spread exits so the
    // median exit is exactly the middle value 5.0 and differs sharply from the
    // earliest entry.
    fs::write(
        &data_path,
        "entry,exit,event,x\n\
         0.5,1.0,1,-0.8\n\
         0.5,2.0,0,0.4\n\
         0.5,3.0,1,-0.2\n\
         0.5,4.0,1,0.7\n\
         0.5,5.0,0,0.1\n\
         0.5,6.0,1,-0.5\n\
         0.5,7.0,1,0.9\n\
         0.5,8.0,0,-0.3\n\
         0.5,9.0,1,0.2\n",
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

    // A `linkwiggle(...)` term routes the survival formula to the
    // location-scale request — the variant whose save path carried the defect.
    let materialized = materialize(
        "Surv(entry, exit, event) ~ x + linkwiggle(degree=2, internal_knots=1)",
        &data,
        &FitConfig::default(),
    )
    .expect("left-truncated survival location-scale should materialize");
    assert!(
        matches!(
            materialized.request,
            FitRequest::SurvivalLocationScale(_)
        ),
        "linkwiggle(...) must route to the survival location-scale request"
    );

    let carried = materialized
        .survival_time_basis
        .expect("a survival materialization must carry its realised time basis");
    let median_exit = 5.0_f64;
    let earliest_entry = 0.5_f64;
    assert!(
        (carried.anchor - median_exit).abs() < 1e-9,
        "the carried time basis must record the robust median-exit anchor \
         ({median_exit}) the fit centered at, got {}",
        carried.anchor
    );
    assert!(
        (carried.anchor - earliest_entry).abs() > 1e-6,
        "the carried anchor must not be the earliest entry ({earliest_entry}) — that \
         is the re-derived value the save path used to persist"
    );
}

/// A left-truncated `Surv(...)` fixture whose earliest entry (0.5) and median
/// exit (3.0) are far apart, so which anchor a path chose is unambiguous.
fn left_truncated_survival_dataset() -> gam_data::EncodedDataset {
    let td = tempdir().expect("tempdir");
    let data_path = td.path().join("left_truncated_anchor_2631.csv");
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
    load_dataset_projected(
        &data_path,
        &[
            "entry".to_string(),
            "exit".to_string(),
            "event".to_string(),
            "x".to_string(),
        ],
    )
    .expect("load left-truncated dataset")
}

/// #2631: an explicit `survival_time_anchor` must reach the fit through
/// `FitConfig`, for EVERY likelihood mode the materializer can build.
///
/// This is the half of the divergence that was invisible from the engine side.
/// The anchor override used to exist only as the CLI's `--survival-time-anchor`,
/// read by the CLI's own copy of the anchor rule — so on the CLI's *default*
/// (transformation / Weibull) route, which delegates to `fit_from_formula`, the
/// flag was parsed, validated, and then dropped on the floor. There was no
/// `FitConfig` field for it to arrive in.
///
/// The override is honored verbatim even where the default would have been the
/// robust median exit (3.0 here): naming the anchor is overriding the
/// conditioning heuristic on purpose.
#[test]
fn explicit_survival_time_anchor_reaches_the_materialized_fit_2631() {
    let data = left_truncated_survival_dataset();
    const EXPLICIT: f64 = 1.25;
    // `latent` / `latent-binary` are covered at the rule level instead
    // (`resolve_survival_time_anchor_for_mode` is exercised across all six modes
    // in `survival::construction`): their materialization RUNS a baseline
    // optimization, so a mode-coverage assertion here would be gated on that
    // solve converging (open: #2538, #2600) rather than on the anchor.
    for mode in ["transformation", "weibull", "location-scale"] {
        let mut config = FitConfig::default();
        config.survival_likelihood = Some(mode.to_string());
        config.survival_time_anchor = Some(EXPLICIT);
        let materialized = materialize("Surv(entry, exit, event) ~ x", &data, &config)
            .unwrap_or_else(|error| panic!("{mode} should materialize: {error}"));
        let carried = materialized
            .survival_time_basis
            .unwrap_or_else(|| panic!("{mode} must carry its realised time basis"));
        assert!(
            (carried.anchor - EXPLICIT).abs() < 1e-12,
            "{mode} must center at the explicit anchor {EXPLICIT}, got {}; \
             the left-truncated default (median exit 3.0) must not win over an \
             explicit request",
            carried.anchor
        );
    }
}

/// #2631: the anchor is a survival-only knob, so like `survival_likelihood` it is
/// refused rather than silently dropped on a non-survival response. The CLI has
/// always refused `--survival-time-anchor` without a `Surv(...)` response; now
/// that the knob is model configuration reachable from every front end, the
/// engine must refuse it identically or the two surfaces disagree again — this
/// time about which configurations are legal.
#[test]
fn survival_time_anchor_rejected_on_nonsurvival_response_2631() {
    let data = nonsurvival_gaussian_dataset();
    let mut config = FitConfig::default();
    config.survival_time_anchor = Some(2.5);

    let err = materialize("time ~ s(x)", &data, &config)
        .err()
        .expect("survival_time_anchor on a non-survival response must error");
    let msg = err.to_string();
    assert!(
        msg.contains("survival_time_anchor"),
        "error must name the offending knob, got: {msg}"
    );
    assert!(
        msg.contains("Surv(...)"),
        "error must point the user at the Surv(...) wrapper, got: {msg}"
    );
}

/// An absent `warm_start_from` leaves every custom-family request without a cache
/// session, so the default fit is unchanged.
#[test]
fn an_absent_warm_start_attaches_no_cache_session() {
    let absent = blockwise_fit_options(&FitConfig::default());
    assert!(absent.cache_session.is_none() && absent.warm_start.is_none());
}

/// The carrier is survival-only: a standard fit has no survival time basis to
/// record, and must not fabricate one.
#[test]
fn materialized_standard_fit_carries_no_survival_time_basis_2470() {
    let data = workflow_test_dataset();
    let materialized = materialize("bmi ~ z", &data, &FitConfig::default())
        .expect("standard formula should materialize");
    assert!(
        materialized.survival_time_basis.is_none(),
        "a non-survival materialization must not carry a survival time basis"
    );
}

/// Speed F6: a saved standard GAM carries only O(p²) state and term metadata,
/// never per-row training data, so its size does not grow with the training
/// rows. The same formula is fit at two training sizes; the longest array
/// anywhere in the serialized payload must be the same length at both, and the
/// exact full-conformal field must hold only the p × p frozen penalty. Before
/// the fix the payload persisted the training design and response for the
/// conformal set and the final PIRLS working weights and response (twice, once
/// under `unified` and once under `fit_result`), each of length n.
#[test]
fn saved_standard_payload_carries_no_per_row_training_data() {
    use crate::inference::model_payload_builders::fit_formula_to_payload;

    fn longest_array(value: &serde_json::Value) -> usize {
        match value {
            serde_json::Value::Array(items) => items
                .iter()
                .map(longest_array)
                .max()
                .unwrap_or(0)
                .max(items.len()),
            serde_json::Value::Object(fields) => {
                fields.values().map(longest_array).max().unwrap_or(0)
            }
            _ => 0,
        }
    }

    let td = tempdir().expect("tempdir");
    let formula = "y ~ s(x1, k=6) + s(x2, k=6)".to_string();
    let fit_at = |n: u32| {
        let data_path = td.path().join(format!("rows_{n}.csv"));
        let mut csv = String::from("y,x1,x2\n");
        for i in 0..n {
            let t = f64::from(i) / f64::from(n);
            let x1 = t * 10.0;
            let x2 = f64::from((i * 7) % 40) / 4.0;
            let y = (x1 * 0.7).sin() * 2.0 + (x2 * 0.4).cos() + 0.3 * (f64::from(i) * 1.7).sin();
            csv.push_str(&format!("{y:.6},{x1:.6},{x2:.6}\n"));
        }
        fs::write(&data_path, csv).expect("write training csv");
        let data = load_dataset_projected(
            &data_path,
            &["y".to_string(), "x1".to_string(), "x2".to_string()],
        )
        .expect("load training dataset");
        let payload = fit_formula_to_payload(formula.clone(), &data, &FitConfig::default())
            .expect("gaussian fit should materialize and fit");
        serde_json::to_value(&payload).expect("payload serializes")
    };
    let small_rows = 400u32;
    let large_rows = 4 * small_rows;
    let small = fit_at(small_rows);
    let large = fit_at(large_rows);

    // Two smooths, so the single-smooth spline-scan path is not taken and the
    // payload is the standard one that carries the conformal penalty.
    let conformal = large
        .get("full_conformal")
        .and_then(serde_json::Value::as_object)
        .expect("an eligible gaussian fit persists its exact full-conformal penalty");
    // The field holds the p × p frozen penalty and the fit's smoothing-parameter
    // count (a scalar, gam#3296), and nothing else: no labeled rows.
    assert_eq!(
        conformal.keys().collect::<Vec<_>>(),
        vec!["penalty_count", "s_lambda"],
        "the exact full-conformal field must persist only the frozen penalty and its count"
    );
    assert_eq!(
        conformal.get("penalty_count").and_then(serde_json::Value::as_u64),
        Some(4),
        "the conformal penalty count is the fit's four smoothing parameters \
         (two smooths, each carrying a wiggliness and a null-space penalty under \
         the default double penalty)"
    );
    assert!(
        longest_array(&large) < small_rows as usize,
        "a saved standard payload must hold no array as long as the training rows \
         (longest array {} at n={large_rows})",
        longest_array(&large)
    );
    assert_eq!(
        longest_array(&small),
        longest_array(&large),
        "the longest array in the saved payload must not depend on the training rows"
    );

    // A v28 payload of the same fit carried the per-row data this version no
    // longer writes: the conformal training `x` and `y`, and the working PIRLS
    // geometry under every fit geometry. It must still load.
    fn insert_working_geometry(value: &mut serde_json::Value, rows: usize) -> usize {
        match value {
            serde_json::Value::Object(fields) => {
                let mut inserted = 0;
                if fields.contains_key("coefficient_gauge") && fields.contains_key("penalized_hessian")
                {
                    fields.insert(
                        "working".to_string(),
                        serde_json::json!({
                            "weights": vec![1.0; rows],
                            "working_response": vec![0.5; rows],
                        }),
                    );
                    inserted += 1;
                }
                inserted + fields.values_mut().map(|v| insert_working_geometry(v, rows)).sum::<usize>()
            }
            serde_json::Value::Array(items) => {
                items.iter_mut().map(|v| insert_working_geometry(v, rows)).sum()
            }
            _ => 0,
        }
    }
    let rows = large_rows as usize;
    let mut legacy = large.clone();
    legacy["version"] = serde_json::json!(28);
    let p = conformal["s_lambda"]["dim"][0].as_u64().expect("s_lambda dim") as usize;
    legacy["full_conformal"]["x"] = serde_json::to_value(Array2::<f64>::zeros((rows, p))).expect("x");
    legacy["full_conformal"]["y"] = serde_json::to_value(ndarray::Array1::<f64>::zeros(rows)).expect("y");
    assert!(
        insert_working_geometry(&mut legacy, rows) > 0,
        "the fit payload must carry a fit geometry to plant the v28 working rows in"
    );
    let loaded: crate::inference::model::FittedModelPayload =
        serde_json::from_value(legacy).expect("a v28 standard payload with training rows must load");
    let penalty = loaded
        .full_conformal
        .as_ref()
        .expect("the v28 conformal field loads as the frozen penalty");
    assert_eq!(penalty.p(), p);
    crate::inference::model::FittedModel::from_payload(loaded)
        .validate_for_persistence()
        .expect("a loaded v28 standard payload passes the saved-model gate");
}

#[test]
fn survival_marginal_slope_matern_slope_penalties_keep_surface_width() {
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
            slope_formula: Some("matern(PC1, PC2, PC3, centers=6)".to_string()),
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
            request.spec.slopespec.clone(),
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
            ("raw slope", &designs[1]),
            ("frozen marginal", &rebuilt[0]),
            ("frozen slope", &rebuilt[1]),
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
/// convergence rather than aborting in `canonicalize_for_identifiability_with_operating_scalars` with
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
    // Layout per cause: [β0 = Weibull shape (slope on log t), β1 = covariate
    // intercept (baseline level), β2 = age]. The linear time basis is the single
    // column `log t`: its former constant column was exactly confounded with the
    // covariate intercept and is no longer built (#2301), so there is no dead
    // coefficient left to pin. The data-generating cause-specific log-rates are
    // +0.25·(age−55)/10 for cause 1 and −0.20·(age−55)/10 for cause 2, i.e. the
    // raw-age coefficient is +0.025 for cause 1 and −0.020 for cause 2.
    let beta1 = &surv.fit.blocks[0].beta;
    let beta2 = &surv.fit.blocks[1].beta;
    assert_eq!(
        beta1.len(),
        3,
        "cause 1 must carry [shape, intercept, age] (#1590, #2301)"
    );
    assert_eq!(
        beta2.len(),
        3,
        "cause 2 must carry [shape, intercept, age] (#1590, #2301)"
    );
    // Shape recovered near 1 (exponential cause-specific hazards).
    for (c, b) in [beta1, beta2].iter().enumerate() {
        assert!(
            b[0] > 0.5 && b[0] < 1.6,
            "cause {} Weibull shape β0 must be ~1 for exponential data, got {}",
            c + 1,
            b[0]
        );
    }
    // The qualitative cause-specific effect must be recovered: cause 1's hazard
    // RISES with age, cause 2's FALLS — opposite-signed age coefficients.
    assert!(
        beta1[2] > 0.0,
        "cause 1 age effect must be positive (hazard rises with age), got {}",
        beta1[2]
    );
    assert!(
        beta2[2] < 0.0,
        "cause 2 age effect must be negative (hazard falls with age), got {}",
        beta2[2]
    );
    // And the two causes must be genuinely DISTINCT fits, not a degenerate copy.
    assert!(
        (beta1[2] - beta2[2]).abs() > 0.01,
        "cause-specific age effects must differ (distinct fits), got {} vs {}",
        beta1[2],
        beta2[2]
    );
}

/// #1561 incidental bug: the Gaussian location-scale joint fit must not abort
/// (panic or hard-error) when the scale smooth is requested at a larger basis
/// size (`bs='tps', k>=20`). The owner's #1561 investigation reported a
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
            noise_formula: Some(format!("1 + s(x, bs='tps', k={k})")),
            ..FitConfig::default()
        };

        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::fit_orchestration::entry::fit_from_formula("y ~ s(x, bs='tps')", &data, &config)
        }));
        let result = caught.unwrap_or_else(|payload| {
            // The payload IS the evidence: without it this reports only that
            // something unwound, which is the least useful half of the finding.
            let detail = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("<panic payload was not a string>");
            panic!(
                "#1561: location-scale fit with a k={k} scale smooth PANICKED inside the \
                 joint-Newton solver; a valid model spec must fit or return a catchable error, \
                 never unwind: {detail}"
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
        ("1 + s(z, bs='tps')", true),
        ("1 + s(z, bs='tps', double_penalty=false)", false),
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
            panic!("bs='tps' scale formula must resolve a thin-plate basis");
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
        slope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };

    let err = materialize("event ~ bmi", &data, &config).err();
    assert!(
        matches!(
            err,
            Some(WorkflowError::TransformationNormalConflict {
                conflict: TransformationNormalConflict::MarginalSlopeControls,
            })
        ),
        "transformation_normal must not steal marginal-slope fits: the refusal must be the typed \
         marginal-slope conflict, got {:?}",
        err.map(|error| error.to_string())
    );
}

#[test]
fn ctn_composition_requires_the_complete_fitted_model_service() {
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
        .expect("a composed fit cannot be reduced to one materialized block");
    let msg = err.to_string();
    assert!(
        msg.contains("CTN composition requires fit_from_formula"),
        "materialization must direct CTN callers to the complete model service: {msg}"
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
        // An auxiliary formula is its right-hand side alone; "~ 1" is refused by the
        // formula service before any model is selected, so it cannot reach the conflict.
        noise_formula: Some("1".to_string()),
        ..FitConfig::default()
    };

    let err = materialize("bmi ~ s(age_entry, k=4)", &data, &config).err();
    assert!(
        matches!(
            err,
            Some(WorkflowError::TransformationNormalConflict {
                conflict: TransformationNormalConflict::NoiseFormula,
            })
        ),
        "family='transformation-normal' must refuse a noise_formula as the typed CTN conflict, \
         got {:?}",
        err.map(|error| error.to_string())
    );
}

#[test]
fn location_scale_refuses_an_active_frailty_it_cannot_realize() {
    use crate::survival::lognormal_kernel::{FrailtyScale, FrailtySpec, HazardLoading};
    let data = workflow_test_dataset();
    let frailties = [
        FrailtySpec::GaussianShift {
            scale: FrailtyScale::Fixed { sigma: 0.5 },
        },
        FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Fixed { sigma: 0.5 },
            loading: HazardLoading::Full,
        },
    ];
    for (formula, family) in [("bmi ~ age_entry", None), ("event ~ bmi", Some("binomial"))] {
        for frailty in frailties.clone() {
            let config = FitConfig {
                family: family.map(str::to_string),
                noise_formula: Some("1".to_string()),
                frailty,
                ..FitConfig::default()
            };
            let err = materialize(formula, &data, &config).err();
            assert!(
                matches!(&err, Some(WorkflowError::InvalidConfig { reason })
                    if reason.contains("frailty is not supported for location-scale")),
                "{formula}: a location-scale fit must refuse a frailty it would drop, got {:?}",
                err.map(|error| error.to_string())
            );
        }
    }
    // Without a frailty the same requests still materialize.
    for (formula, family) in [("bmi ~ age_entry", None), ("event ~ bmi", Some("binomial"))] {
        let config = FitConfig {
            family: family.map(str::to_string),
            noise_formula: Some("1".to_string()),
            ..FitConfig::default()
        };
        materialize(formula, &data, &config)
            .unwrap_or_else(|error| panic!("{formula}: location-scale without frailty: {error}"));
    }
}

#[test]
fn survival_marginal_slope_rejects_zero_event_data_before_fit() {
    let mut data = workflow_test_dataset();
    data.values.column_mut(2).fill(0.0);
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        slope_formula: Some("1".to_string()),
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

/// An explicit `survival_likelihood='transformation'` is the user's choice, not
/// the unset default: `linkwiggle(...)` must not silently swap it for the
/// location-scale model. The existing linkwiggle refusal fires instead.
#[test]
fn explicit_transformation_likelihood_with_linkwiggle_is_refused_not_swapped() {
    let data = competing_risks_weighted_dataset([1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]);
    let config = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        ..FitConfig::default()
    };
    let err = materialize(
        "Surv(age_entry, age_exit, event) ~ bmi + linkwiggle(degree=2, internal_knots=1)",
        &data,
        &config,
    )
    .err()
    .expect("an explicit transformation likelihood must not be promoted by linkwiggle");
    let msg = err.to_string();
    assert!(
        msg.contains("linkwiggle(...) is not defined for survival_likelihood='transformation'"),
        "unexpected error: {msg}"
    );
}

/// Same contract for `noise_formula`: an explicit transformation likelihood has
/// no log-sigma predictor, so the noise formula is refused.
#[test]
fn explicit_transformation_likelihood_with_noise_formula_is_refused_not_swapped() {
    let data = competing_risks_weighted_dataset([1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]);
    let config = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        noise_formula: Some("bmi".to_string()),
        ..FitConfig::default()
    };
    let err = materialize("Surv(age_entry, age_exit, event) ~ bmi", &data, &config)
        .err()
        .expect("an explicit transformation likelihood must not be promoted by noise_formula");
    let msg = err.to_string();
    assert!(
        msg.contains("noise_formula requires the survival location-scale likelihood"),
        "unexpected error: {msg}"
    );
}

/// Control: with the likelihood unset, a noise formula still selects the
/// location-scale model, so the transformation refusal never fires.
#[test]
fn unset_likelihood_with_noise_formula_still_selects_location_scale() {
    let data = competing_risks_weighted_dataset([1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]);
    let config = FitConfig {
        noise_formula: Some("bmi".to_string()),
        ..FitConfig::default()
    };
    if let Err(err) = materialize("Surv(age_entry, age_exit, event) ~ bmi", &data, &config) {
        assert!(
            !err.to_string().contains("noise_formula requires"),
            "an unset likelihood must be promoted to location-scale: {err}"
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
    duchon_workflow_dataset_with_rows(72)
}

fn duchon_workflow_dataset_with_rows(n: usize) -> Dataset {
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
        raw_centers >= starting_num_centers(data.values.nrows(), 1, 2),
        "{label} formula default must retain its derived univariate resolution floor"
    );

    let initial_config = FitConfig {
        adaptive_resolution: Some(Vec::new()),
        ..FitConfig::default()
    };
    let initial = materialize(formula, &data, &initial_config)
        .unwrap_or_else(|error| panic!("initial adaptive {label} materialization failed: {error}"));
    let FitRequest::Standard(initial_request) = initial.request else {
        panic!("expected standard adaptive {label} request");
    };
    let (initial_centers, initial_is_auto) =
        planned_radial_centers(&initial_request.spec.smooth_terms[0].basis);
    // #3149: the orchestrated request starts at the pilot (here the rate count
    // `starting_num_centers`, above the pilot `s(x)` floor), and the raw
    // request, which nothing grows, at the provisioned default above it.
    assert_eq!(
        initial_centers,
        starting_num_centers(data.values.nrows(), 1, 2),
        "an absent adaptive proposal must start the 1-D {label} at its pilot"
    );
    assert!(
        raw_centers > initial_centers,
        "the provisioned 1-D {label} default ({raw_centers}) sits above the pilot ({initial_centers})"
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
        adaptive_resolution: Some(vec![Some(
            gam_terms::smooth::AdaptiveResolution::Centers(proposed_centers),
        )]),
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
            adaptive_resolution: Some(vec![Some(
                gam_terms::smooth::AdaptiveResolution::Centers(raw_centers.saturating_mul(2)),
            )]),
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
        adaptive_resolution: Some(Vec::new()),
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
        starting_num_centers(data.values.nrows(), 2, 3)
    );
    // #3149: only the orchestrated request, whose loop grows the basis,
    // starts at the rate-derived pilot `starting_num_centers(n, d, nullspace)`.
    // The raw request has no loop, so it keeps the provisioned low-rank
    // default (#1757): the generic spatial count held to `10 · 3^(d - 1)` =
    // 30 centers in 2-D.
    assert_eq!(
        raw_centers,
        default_num_centers(data.values.nrows(), 2).min(30),
        "the raw 2-D Duchon default is the provisioned low-rank default"
    );
    assert!(
        raw_centers > adaptive_centers,
        "the pilot start ({adaptive_centers}) sits below the provisioned default ({raw_centers})"
    );
    assert!(
        adaptive_centers <= default_num_centers(data.values.nrows(), 2),
        "the pilot start must never exceed the validated production ceiling"
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

    // Second arm, at an n where the `n / COND_N_DIVISOR` conditioning cap in
    // `default_num_centers` no longer binds: the raw request is at the 30-center
    // provisioned cap, and the rate pilot is STRICTLY below the production
    // ceiling, so the orchestrator's grow loop has something to escalate.
    let wide = duchon_workflow_dataset_with_rows(200);
    let wide_rows = wide.values.nrows();
    let low_rank_representer_rank = starting_num_centers(wide_rows, 2, 3);
    let wide_raw = materialize("y ~ duchon(ct, st)", &wide, &FitConfig::default())
        .expect("raw Duchon materialization at 200 rows");
    let FitRequest::Standard(wide_raw_request) = wide_raw.request else {
        panic!("expected standard request");
    };
    let SmoothBasisSpec::Duchon {
        spec: wide_raw_spec,
        ..
    } = &wide_raw_request.spec.smooth_terms[0].basis
    else {
        panic!("expected Duchon smooth");
    };
    assert_eq!(
        wide_raw_spec.center_strategy.planned_num_centers(2),
        30,
        "the raw 2-D Duchon default is the provisioned low-rank cap (#1757)"
    );
    assert!(
        default_num_centers(wide_rows, 2) > low_rank_representer_rank,
        "the grow-loop ceiling must strictly exceed the low-rank start at {wide_rows} rows,          or an orchestrated 2-D Duchon has nothing to escalate: ceiling={}, start={}",
        default_num_centers(wide_rows, 2),
        low_rank_representer_rank
    );
}

/// Two continuous coordinates and an unbalanced two-level factor: `a` on
/// `n_a` rows, `b` on `n_b`.
fn by_level_radial_workflow_dataset(n_a: usize, n_b: usize) -> Dataset {
    let n = n_a + n_b;
    let mut values = Array2::<f64>::zeros((n, 4));
    for i in 0..n {
        let x = i as f64 / (n - 1) as f64;
        let z = ((i * 37) % n) as f64 / (n - 1) as f64;
        values[[i, 0]] = (3.0 * x).sin() + z;
        values[[i, 1]] = x;
        values[[i, 2]] = z;
        values[[i, 3]] = if i < n_a { 0.0 } else { 1.0 };
    }
    let continuous = |name: &str| SchemaColumn {
        name: name.to_string(),
        kind: ColumnKindTag::Continuous,
        levels: vec![],
    };
    Dataset {
        headers: vec!["y".to_string(), "x".to_string(), "z".to_string(), "g".to_string()],
        values,
        schema: DataSchema {
            columns: vec![
                continuous("y"),
                continuous("x"),
                continuous("z"),
                SchemaColumn {
                    name: "g".to_string(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["a".to_string(), "b".to_string()],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Categorical,
        ],
    }
}

/// #3149/#2993: a factor-by level's block is identified on its own level's
/// rows, so the orchestrated start of an auto-sized radial smooth in each
/// level is the rate pilot at that level's row count, not at the pooled rows
/// the other levels contribute.
#[test]
fn adaptive_spatial_start_of_a_by_level_smooth_counts_its_own_level_rows() {
    let (n_a, n_b) = (120, 60);
    let data = by_level_radial_workflow_dataset(n_a, n_b);
    let config = FitConfig {
        adaptive_resolution: Some(Vec::new()),
        ..FitConfig::default()
    };
    let mat = materialize("y ~ s(x, z, by=g)", &data, &config)
        .expect("adaptive by-level thin-plate materialization");
    let FitRequest::Standard(request) = mat.request else {
        panic!("expected standard request");
    };
    let mut starts = Vec::new();
    for term in &request.spec.smooth_terms {
        let SmoothBasisSpec::ByVariable { inner, .. } = &term.basis else {
            continue;
        };
        let SmoothBasisSpec::ThinPlate { spec, .. } = inner.as_ref() else {
            panic!("expected a thin-plate level block, got {inner:?}");
        };
        assert!(center_strategy_is_auto(&spec.center_strategy));
        starts.push(spec.center_strategy.planned_num_centers(2));
    }
    assert_eq!(
        starts,
        vec![
            starting_num_centers(n_a, 2, 3),
            starting_num_centers(n_b, 2, 3)
        ],
        "each level starts from its own {n_a} / {n_b} rows, not the pooled {}",
        n_a + n_b
    );
    assert_ne!(
        starting_num_centers(n_b, 2, 3),
        starting_num_centers(n_a + n_b, 2, 3),
        "the fixture must separate a level's rows from the pooled rows"
    );
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
fn workflow_survival_marginal_slope_routes_slope_linkwiggle_into_score_warp_only() {
    let data = workflow_test_dataset();
    // #384: the score-warp / link-deviation runtime is structurally cubic, so
    // only `degree=3` is realizable on these blocks; non-cubic degrees are
    // rejected up front (see
    // `linkwiggle_noncubic_degree_is_rejected_at_the_routing_boundary_issue_384`).
    // This test exercises the orthogonal routing/metadata contract: the
    // slope_formula linkwiggle lands on `score_warp` and the main-formula
    // linkwiggle on `link_dev`, with knots/penalty orders carried through. The
    // two blocks are distinguished here by `internal_knots` (9 vs 7) and
    // `penalty_order` (1 vs 2,3), not by an unrealizable degree.
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        slope_formula: Some(
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
        ..
    } = materialized;
    let FitRequest::SurvivalMarginalSlope(request) = request else {
        panic!("expected survival marginal-slope request");
    };

    let link_dev = request.spec.link_dev.expect("main-formula link-dev");
    let score_warp = request.spec.score_warp.expect("slope score-warp");
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
        "workflow notes should mention slope_formula linkwiggle routing"
    );
}

#[test]
fn materialize_routes_bernoulli_marginal_slope_when_slope_and_z_are_set() {
    let data = workflow_test_dataset();
    let config = FitConfig {
        slope_formula: Some("1".to_string()),
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
        slope_formula: Some("matern(PC1, PC2, PC3, centers=3)".to_string()),
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
        unidentified_scalar_terms,
        ..
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
    assert_eq!(request.spec.slopespec.smooth_terms.len(), 1);
    assert!(
        inference_notes
            .iter()
            .any(|note| note.contains("constant_spline_col")),
        "materialization should report the removed redundant scalar term; notes={inference_notes:?}"
    );
    // #2627: the removal is published as a typed record naming the formula, the
    // term and the residual that decided it, not only as note text.
    assert_eq!(
        unidentified_scalar_terms
            .iter()
            .map(|removed| (removed.formula.as_str(), removed.term.as_str()))
            .collect::<Vec<_>>(),
        vec![("bernoulli marginal-slope marginal formula", "constant_spline_col")],
        "the removed scalar term must be published as a typed record"
    );
    assert!(
        unidentified_scalar_terms
            .iter()
            .all(|removed| removed.residual_norm <= removed.tolerance),
        "each record carries the residual that fell inside its rank tolerance: {unidentified_scalar_terms:?}"
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
        slope_formula: Some("matern(PC1, PC2, PC3, centers=3)".to_string()),
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
    assert_eq!(request.spec.slopespec.smooth_terms.len(), 1);
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
        slope_formula: Some("1".to_string()),
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
fn bernoulli_marginal_slope_prune_drops_penalized_redundant_scalar_term() {
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
        level: Default::default(),
    };
    let mut notes = crate::fit_orchestration::FitNotes::default();
    let removed = prune_unidentified_linear_terms_for_marginal_slope(
        &mut spec,
        &data,
        "test BMS formula",
        &mut notes,
    )
    .expect("a ridge-carrying duplicate scalar term is pruned like an unpenalized one");
    assert_eq!(
        removed
            .iter()
            .map(|record| (record.formula.as_str(), record.term.as_str()))
            .collect::<Vec<_>>(),
        vec![("test BMS formula", "constant_spline_col")],
        "the prune returns the typed record of what it removed"
    );
    assert!(
        spec.linear_terms.is_empty(),
        "the duplicate scalar direction must be pruned, not left for its ridge to identify: {:?}",
        spec.linear_terms
            .iter()
            .map(|term| term.name.as_str())
            .collect::<Vec<_>>()
    );
    assert!(
        notes.iter().any(|note| note.contains("constant_spline_col")),
        "materialization should report the pruned ridge-carrying term; notes={notes:?}"
    );
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
        slope_formula: Some("1".to_string()),
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

        // slope_formula = linkwiggle(...) is the score-warp route the Python
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
        slope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };
    if let Err(err) = materialize(
        "Surv(age_entry, age_exit, event) ~ bmi + link(type=probit)",
        &data,
        &config,
    ) {
        panic!("explicit probit must be accepted, got {err:?}");
    }

    let err = match materialize(
        "Surv(age_entry, age_exit, event) ~ bmi + link(type=logit)",
        &data,
        &config,
    ) {
        Ok(_) => panic!("non-probit link should be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(
            err,
            WorkflowError::MarginalSlopeLink {
                context: "survival marginal-slope",
                refusal: MarginalSlopeLinkRefusal::NonProbit,
            }
        ),
        "a non-probit survival marginal-slope link must be the typed probit-only refusal, got {err:?}"
    );
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

    // Through the fit boundary: the refusal keeps its category (#2937).
    let err = match fit_model(FitRequest::SurvivalLocationScale(request)) {
        Ok(_) => panic!("survival link wiggle should reject unsupported inverse links"),
        Err(e) => e,
    };
    assert_eq!(err.failure_category(), gam_problem::FailureCategory::Input, "{err}");
    assert_eq!(err.variant_name(), "FitFailure::Input", "{err}");
    let err = err.to_string();

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
    // Replicated Bernoulli observations from a nonlinear monotone probability
    // curve. Repetition keeps every local empirical probability strictly inside
    // (0, 1), while the cubic warp gives the link-wiggle arm real shape to
    // estimate instead of placing its nonnegative coefficients on the cone
    // vertex. The former alternating fixture was globally balanced but had no
    // gradient in x at all; its population optimum was the zero-wiggle boundary,
    // where the cone-truncated quadratic approximation is genuinely improper.
    let support_points = 15usize;
    let replicates = 8usize;
    let n = support_points * replicates;
    let mut records: Vec<csv::StringRecord> = Vec::with_capacity(n);
    for group in 0..support_points {
        let x = -2.0 + 4.0 * (group as f64) / ((support_points - 1) as f64);
        let warped_logit = 0.75 * x + 0.16 * x.powi(3);
        let probability = 1.0 / (1.0 + (-warped_logit).exp());
        let successes = ((replicates as f64 * probability).round() as usize)
            .clamp(1, replicates - 1);
        for replicate in 0..replicates {
            let y = if replicate < successes { 1.0 } else { 0.0 };
            records.push(csv::StringRecord::from(vec![
                format!("{y:.17e}"),
                format!("{x:.17e}"),
            ]));
        }
    }
    gam_data::encode_recordswith_inferred_schema(vec!["y".to_string(), "x".to_string()], records)
        .expect("encode binomial location-scale dataset")
}

fn small_wiggle_cfg() -> LinkWiggleConfig {
    LinkWiggleConfig {
        degree: 3,
        num_internal_knots: 3,
        // Keep this orchestration-parity fixture on the full-rank anchored
        // order-one function metric. An order-two penalty has an affine null
        // space, making posterior propriety an unrelated fixture precondition.
        penalty_orders: vec![1],
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
    let raw_offsets = GaussianLocationScaleRawOffsets::of(&spec);
    let s = standardize_gaussian_spec_like_engine(&mut spec);
    let sigma_floor =
        crate::sigma_link::gaussian_resolution_sigma_floor(spec.y.view(), spec.weights.view())
            .expect("gaussian location-scale resolution σ floor");
    let fit = fit_gaussian_location_scale_terms(data, spec, options, kappa_options)
        .expect("reference gaussian no-wiggle terms fit");
    let mut result = GaussianLocationScaleFitResult {
        fit,
        wiggle_knots: None,
        wiggle_degree: None,
        beta_link_wiggle: None,
        response_scale: 1.0,
        sigma_floor,
    };
    rescale_gaussian_location_scale_to_raw(&mut result, s, &raw_offsets)
        .expect("gaussian location-scale raw remap");
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
    let raw_offsets = GaussianLocationScaleRawOffsets::of(&spec);
    let s = standardize_gaussian_spec_like_engine(&mut spec);
    let sigma_floor =
        crate::sigma_link::gaussian_resolution_sigma_floor(spec.y.view(), spec.weights.view())
            .expect("gaussian location-scale resolution σ floor");
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
        sigma_floor,
    };
    rescale_gaussian_location_scale_to_raw(&mut result, s, &raw_offsets)
        .expect("gaussian location-scale raw remap");
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
    let raw_offsets = GaussianLocationScaleRawOffsets::of(&spec);
    let s = standardize_gaussian_spec_like_engine(&mut spec);
    let sigma_floor =
        crate::sigma_link::gaussian_resolution_sigma_floor(spec.y.view(), spec.weights.view())
            .expect("gaussian location-scale resolution σ floor");
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
        sigma_floor,
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
        inference.smoothing_correction = Some(&corrected - &conditional);
        inference.smoothing_correction_method = Some(
            gam_solve::model_types::SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                active_rank: 1,
                rho_dimension: 1,
            },
        );
    }

    rescale_gaussian_location_scale_to_raw(&mut result, s, &raw_offsets)
        .expect("gaussian location-scale raw remap");

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
    // Corrected SEs are derived from the one remapped corrected covariance, so
    // se_raw_i^2 must equal the corrected diagonal exactly up to float
    // regrouping.
    let se = fit
        .beta_standard_errors_corrected()
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

/// #1561: the raw remap carries the change of units on the precision side in one
/// of two representations of one saved state. It either rescales the precision
/// (the gauge's active coordinates are the saved coordinates) or composes the unit
/// map into the gauge and leaves the precision as solved (a reduced active frame).
/// Forced onto one identity-gauge fit, the two must agree on every quantity a
/// consumer reads, or one representation is wrong.
#[test]
fn gaussian_location_scale_raw_remap_representations_agree_1561() {
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
    let mut spec = spec;
    let raw_offsets = GaussianLocationScaleRawOffsets::of(&spec);
    let s = standardize_gaussian_spec_like_engine(&mut spec);
    let sigma_floor =
        crate::sigma_link::gaussian_resolution_sigma_floor(spec.y.view(), spec.weights.view())
            .expect("gaussian location-scale resolution σ floor");
    assert!(
        (s - 1.0).abs() > 10.0,
        "fixture response scale must make the remap non-trivial, got s={s}"
    );
    // The production workflow (`fit_location_scale_with_optional_wiggle`) fits with
    // `compute_covariance = true`, so the remap always meets a published covariance.
    let mut options = options;
    options.compute_covariance = true;
    let fit = fit_gaussian_location_scale_terms(req_data, spec, &options, &kappa_options)
        .expect("standardized gaussian location-scale terms fit");
    assert!(
        fit.fit
            .geometry
            .as_ref()
            .expect("terms fit carries saved geometry")
            .coefficient_gauge
            .is_identity(),
        "the equivalence fixture must be an identity-gauge fit"
    );
    let wrap = |fit| GaussianLocationScaleFitResult {
        fit,
        wiggle_knots: None,
        wiggle_degree: None,
        beta_link_wiggle: None,
        response_scale: 1.0,
        sigma_floor,
    };
    let mut rescaled = wrap(fit.clone());
    let mut composed = wrap(fit);
    rescale_gaussian_location_scale_to_raw_with_units(
        &mut rescaled,
        s,
        ActiveFrameUnits::RescalePrecision,
        &raw_offsets,
    )
    .expect("rescaled representation");
    rescale_gaussian_location_scale_to_raw_with_units(
        &mut composed,
        s,
        ActiveFrameUnits::ComposeIntoGauge,
        &raw_offsets,
    )
    .expect("composed representation");
    let (a, b) = (&rescaled.fit.fit, &composed.fit.fit);

    // Raw coefficients, raw covariance and SE bands: what predictions and
    // intervals read. The conditional covariance and SEs must be published, so
    // their agreement is not `None == None`.
    assert_eq!(a.blocks.len(), b.blocks.len());
    for (block_a, block_b) in a.blocks.iter().zip(&b.blocks) {
        assert_eq!(
            block_a.beta, block_b.beta,
            "raw coefficients must not depend on the representation"
        );
    }
    let covariance_a = a
        .covariance_conditional
        .as_ref()
        .expect("the terms fit publishes a conditional covariance");
    assert_eq!(Some(covariance_a), b.covariance_conditional.as_ref());
    assert_eq!(a.covariance_corrected, b.covariance_corrected);
    assert!(
        a.beta_standard_errors().is_some(),
        "the terms fit publishes conditional standard errors"
    );
    assert_eq!(a.beta_standard_errors(), b.beta_standard_errors());
    assert_eq!(
        a.beta_standard_errors_corrected(),
        b.beta_standard_errors_corrected()
    );

    // Predictions on the fitted rows: each channel's linear predictor
    // `affine_offset + X·β` and its conditional SE band `sqrt(diag(X·V·Xᵀ))`.
    let covariance_b = b
        .covariance_conditional
        .as_ref()
        .expect("composed conditional covariance");
    let mut start = 0usize;
    for (block_a, block_b) in a.blocks.iter().zip(&b.blocks) {
        let width = block_a.beta.len();
        let (design_a, design_b) = if matches!(block_a.role, gam_problem::BlockRole::Scale) {
            (&rescaled.fit.noise_design, &composed.fit.noise_design)
        } else {
            (&rescaled.fit.mean_design, &composed.fit.mean_design)
        };
        let x_a = design_a.design.to_dense();
        let x_b = design_b.design.to_dense();
        assert_eq!(x_a.ncols(), width, "block design width must match its coefficients");
        let eta_a = &design_a.affine_offset + &x_a.dot(&block_a.beta);
        let eta_b = &design_b.affine_offset + &x_b.dot(&block_b.beta);
        assert_eq!(
            eta_a, eta_b,
            "fitted linear predictors must not depend on the representation"
        );
        let block = ndarray::s![start..start + width, start..start + width];
        let band_a = x_a
            .dot(&covariance_a.slice(block))
            .dot(&x_a.t())
            .diag()
            .mapv(f64::sqrt);
        let band_b = x_b
            .dot(&covariance_b.slice(block))
            .dot(&x_b.t())
            .diag()
            .mapv(f64::sqrt);
        assert!(
            band_a.iter().all(|se| se.is_finite() && *se > 0.0),
            "each fitted row carries a positive finite SE band"
        );
        assert_eq!(
            band_a, band_b,
            "SE bands must not depend on the representation"
        );
        start += width;
    }
    assert_eq!(start, covariance_a.nrows(), "the blocks tile the joint covariance");

    // The two representations: the rescaled precision on an identity gauge, and
    // the as-solved precision on the composed section β_raw = D·β_internal + a.
    let geom_a = a.geometry.as_ref().expect("rescaled geometry");
    let geom_b = b.geometry.as_ref().expect("composed geometry");
    let inf_a = a.inference.as_ref().expect("rescaled inference block");
    let inf_b = b.inference.as_ref().expect("composed inference block");
    assert!(geom_a.coefficient_gauge.is_identity());
    assert!(!geom_b.coefficient_gauge.is_identity());
    assert!(
        geom_a.penalized_hessian == inf_a.penalized_hessian
            && geom_b.penalized_hessian == inf_b.penalized_hessian,
        "each representation keeps its geometry and inference precision copies equal"
    );
    let lift = &geom_b.coefficient_gauge.t_full;
    let p = lift.nrows();
    assert_eq!(
        lift.dim(),
        (p, p),
        "an identity gauge composed with the unit map stays square"
    );

    // The raw precision pulled back through the composed gauge is the rescaled
    // precision.
    let h_a = &geom_a.penalized_hessian.0;
    let h_theta = &geom_b.penalized_hessian.0;
    let inverse_lift: Vec<f64> = (0..p).map(|i| 1.0 / lift[[i, i]]).collect();
    let scale = h_a.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())).max(1.0);
    for i in 0..p {
        for j in 0..p {
            let pulled_back = inverse_lift[i] * h_theta[[i, j]] * inverse_lift[j];
            assert!(
                (pulled_back - h_a[[i, j]]).abs() <= 1e-12 * scale,
                "raw precision [{i},{j}]: pulled back {pulled_back:e} against rescaled {:e}",
                h_a[[i, j]]
            );
        }
    }

    // #2623: raw directions N restrict the saved precision as NᵀH_raw N. Through the
    // gauge, C = T⁻¹N restricts H_θ as CᵀH_θC. Both representations must price the
    // same restricted curvature.
    let directions =
        Array2::from_shape_fn((p, 2), |(i, k)| (0.37 * (i + 1) as f64 + 1.3 * k as f64).sin());
    let restricted_a = directions.t().dot(&h_a.dot(&directions));
    let pulled = Array2::from_shape_fn((p, 2), |(i, k)| inverse_lift[i] * directions[[i, k]]);
    let restricted_b = pulled.t().dot(&h_theta.dot(&pulled));
    let restricted_scale = restricted_a
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
        .max(1.0);
    for r in 0..2 {
        for c in 0..2 {
            assert!(
                (restricted_a[[r, c]] - restricted_b[[r, c]]).abs() <= 1e-12 * restricted_scale,
                "restricted curvature [{r},{c}]: rescaled {:e} against composed {:e}",
                restricted_a[[r, c]],
                restricted_b[[r, c]]
            );
        }
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
    let engine_plain_covariance = engine_plain
        .fit
        .fit
        .beta_covariance()
        .expect("engine gaussian no-wiggle fit must retain joint posterior covariance");
    assert_eq!(
        engine_plain_covariance.dim(),
        (
            engine_plain.fit.fit.beta_flat().len(),
            engine_plain.fit.fit.beta_flat().len(),
        ),
        "engine gaussian no-wiggle covariance must cover every saved coefficient"
    );

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

    // The facts the assertions below decide on, printed before any of them can
    // fail: the arming evidence the lifecycle published, and the typed decline
    // with its certificate, or its absence (#2627, #979).
    let arming_evidence = format!("{:?}", engine_wiggle.fit.fit.artifacts.jeffreys_arming_evidence);
    let decline_summary = engine_wiggle
        .fit
        .fit
        .posterior_moment_decline()
        .map_or_else(|| "no typed moment decline".to_string(), |decline| decline.summary());
    eprintln!(
        "[2627-CONE] gaussian wiggle: arming evidence {arming_evidence}; covariance published {}; {decline_summary}",
        engine_wiggle.fit.fit.beta_covariance().is_some()
    );
    // #2635: the ambient Hessian is indefinite, so the fit keeps its converged mode
    // under a typed moment decline and refuses to relabel that mode as a posterior
    // mean. Jeffreys ruling (b) (#979): the decline's exact cone certificate must
    // decide properness, and a published fit that did not arm cannot carry a
    // proved-improper cone, because that proof is arming evidence. The fixture
    // prints which objective the published fit is the mode of, and why, before it
    // asserts.
    assert!(
        engine_wiggle.fit.fit.beta_covariance().is_none(),
        "the indefinite ambient precision has no Gaussian covariance to report"
    );
    let decline = engine_wiggle
        .fit
        .fit
        .posterior_moment_decline()
        .expect("the indefinite-precision fit must retain a typed moment decline");
    let arming = &engine_wiggle.fit.fit.artifacts.jeffreys_arming_evidence;
    println!(
        "[#979 engine] jeffreys_arming_evidence={arming:?} verdict={:?} decline={}",
        decline.properness.is_proper(),
        decline.summary()
    );
    assert!(
        decline.properness.is_proper().is_some(),
        "the exact cone certificate must survive assembly with a decided verdict: {}",
        decline.summary()
    );
    assert!(
        arming.is_some() || decline.properness.is_proper() == Some(true),
        "an unarmed fit whose cone posterior is proved improper must have armed: {}",
        decline.summary()
    );
    let refusal = engine_wiggle
        .fit
        .fit
        .require_posterior_mean("Gaussian location-scale prediction")
        .expect_err("prediction must not substitute the mode for the posterior mean")
        .to_string();
    assert!(
        refusal.contains("posterior-mean") && refusal.contains("PROPER"),
        "the refusal must name the missing estimand and certified law: {refusal}"
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
    let engine_plain_covariance = engine_plain
        .fit
        .fit
        .beta_covariance()
        .expect("engine binomial no-wiggle fit must retain joint posterior covariance");
    assert_eq!(
        engine_plain_covariance.dim(),
        (
            engine_plain.fit.fit.beta_flat().len(),
            engine_plain.fit.fit.beta_flat().len(),
        ),
        "engine binomial no-wiggle covariance must cover every saved coefficient"
    );

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
fn resolve_family_accepts_parenthesized_family_link_syntax() {
    // A family may carry its link as `family(link)` — `binomial(logit)`,
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
        Some("negative-binomial(log)"),
        None,
        None,
        ndarray::array![0.0, 1.0, 2.0, 3.0].view(),
        ResponseColumnKind::Numeric,
        "y",
    )
    .expect("negative-binomial(log) resolves");
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

/// gam#2894 bar 4: the Gaussian location-scale link-wiggle family on an ACTIVE monotonicity face.
///
/// The response is linear for `x < 0` and convex beyond, so the monotone wiggle keeps some of its
/// nonnegative coefficients at the cone boundary and grows others. The refit therefore lands on a
/// proper face, where the criterion prices `½·log|Zᵀ M Z|` with its kernel on the face tangent.
/// Within one active set its analytic ρ-gradient must be the derivative of its value. The seed
/// probe checks this at the seed and at the returned optimum, against central differences of the
/// criterion. A stencil whose points change the pinned set differences two criteria, so that
/// coordinate is reported and not graded.
#[test]
fn gaussian_location_scale_wiggle_face_criterion_gradient_matches_central_differences_2894() {
    use gam_solve::estimate::outer_eval_capture::{
        OuterSeedOrder, OuterSeedProbe, observe_next_outer_seed,
    };
    use std::cell::RefCell;
    use std::rc::Rc;

    let n = 72usize;
    let mut records: Vec<csv::StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let x = -2.0 + 4.0 * (i as f64) / ((n - 1) as f64);
        let y = 0.7 * x + 0.5 * x.max(0.0).powi(2) + 0.05 * (7.3 * x).sin();
        records.push(csv::StringRecord::from(vec![
            format!("{y:.17e}"),
            format!("{x:.17e}"),
        ]));
    }
    let data =
        gam_data::encode_recordswith_inferred_schema(vec!["y".to_string(), "x".to_string()], records)
            .expect("encode the face fixture");
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
        mut spec,
        options,
        kappa_options,
        ..
    } = request;
    standardize_gaussian_spec_like_engine(&mut spec);
    let wiggle_cfg = small_wiggle_cfg();
    let pilot = fit_gaussian_location_scale_terms(req_data, spec.clone(), &options, &kappa_options)
        .expect("face fixture pilot");
    let refit = |spec: GaussianLocationScaleTermSpec| {
        let basis = select_gaussian_location_scale_link_wiggle_basis_from_pilot(
            &pilot,
            &WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            },
            &wiggle_cfg.penalty_orders,
        )
        .expect("face fixture wiggle basis selection");
        fit_gaussian_location_scale_terms_with_selected_wiggle(
            req_data,
            spec,
            basis,
            &options,
            &kappa_options,
        )
    };
    let solved = refit(spec.clone()).expect("face fixture wiggle refit");
    let wiggle = solved.fit.fit.block_states[2].beta.clone();
    let wiggle_width = wiggle.len();
    let bound = 1.0e-10;
    let pinned_at_optimum = wiggle.iter().filter(|value| value.abs() <= bound).count();
    assert!(
        pinned_at_optimum >= 1 && pinned_at_optimum < wiggle_width,
        "the fixture must land on a proper face: {pinned_at_optimum} of {wiggle_width} wiggle \
         coefficients pinned ({wiggle:?})"
    );
    let optimum = solved.fit.fit.log_lambdas.clone();

    type Grade = (&'static str, usize, f64, f64, f64, bool);
    let captured: Rc<RefCell<Option<Result<Vec<Grade>, String>>>> = Rc::new(RefCell::new(None));
    let sink = Rc::clone(&captured);
    observe_next_outer_seed(
        0,
        Box::new(
            move |probe: &mut dyn OuterSeedProbe| -> Result<(), gam_solve::estimate::EstimationError> {
                let outcome = (|| -> Result<Vec<Grade>, String> {
                    let layout = probe.layout().clone();
                    if optimum.len() != layout.seed.len() {
                        return Err(format!(
                            "the returned optimum has {} coordinates, the seed {}",
                            optimum.len(),
                            layout.seed.len()
                        ));
                    }
                    let pinned = |beta: &ndarray::Array1<f64>| -> Vec<usize> {
                        let start = beta.len() - wiggle_width;
                        (0..wiggle_width).filter(|&k| beta[start + k].abs() <= bound).collect()
                    };
                    let clamp = |theta: ndarray::Array1<f64>| {
                        ndarray::Array1::from_shape_fn(theta.len(), |i| {
                            theta[i].clamp(layout.lower[i], layout.upper[i])
                        })
                    };
                    let mut grades = Vec::new();
                    for (point, theta) in
                        [("seed", layout.seed.clone()), ("optimum", clamp(optimum.clone()))]
                    {
                        let at = probe
                            .evaluate(&theta, OuterSeedOrder::ValueAndGradient)
                            .map_err(|error| format!("{point}: {error}"))?;
                        let gradient = at
                            .gradient
                            .ok_or_else(|| format!("{point}: no analytic gradient"))?;
                        let (beta, _) = at
                            .selected_mode
                            .ok_or_else(|| {
                                format!(
                                    "{point}: no selected mode (criterion components published: {})",
                                    at.criterion_components.is_some()
                                )
                            })?;
                        let face = pinned(&beta);
                        for j in 0..theta.len() {
                            let room = (theta[j] - layout.lower[j])
                                .min(layout.upper[j] - theta[j])
                                .max(0.0);
                            let mut step = (1.0e-2 * (1.0 + theta[j].abs())).min(0.5 * room);
                            if !(step > 0.0) {
                                continue;
                            }
                            let mut estimates: Vec<(f64, f64, bool)> = Vec::new();
                            for _ in 0..6 {
                                let mut same_face = true;
                                let mut values = [0.0_f64; 2];
                                for (slot, sign) in [(0usize, 1.0_f64), (1, -1.0)] {
                                    let mut displaced = theta.clone();
                                    displaced[j] += sign * step;
                                    let evaluation = probe
                                        .evaluate(&displaced, OuterSeedOrder::Value)
                                        .map_err(|error| format!("{point}: {error}"))?;
                                    let (displaced_beta, _) = evaluation
                                        .selected_mode
                                        .ok_or_else(|| format!("{point}: no displaced mode"))?;
                                    same_face &= pinned(&displaced_beta) == face;
                                    values[slot] = evaluation.cost;
                                }
                                estimates.push((step, (values[0] - values[1]) / (2.0 * step), same_face));
                                step *= 0.5;
                            }
                            let (index, settle) = (1..estimates.len())
                                .map(|i| (i, (estimates[i].1 - estimates[i - 1].1).abs()))
                                .min_by(|left, right| left.1.total_cmp(&right.1))
                                .expect("the ladder has six rungs");
                            let on_face =
                                !face.is_empty() && estimates[index].2 && estimates[index - 1].2;
                            grades.push((point, j, gradient[j], estimates[index].1, settle, on_face));
                        }
                    }
                    Ok(grades)
                })();
                *sink.borrow_mut() = Some(outcome);
                Ok(())
            },
        ),
    );
    let refit_with_probe = refit(spec);
    let grades = captured
        .borrow_mut()
        .take()
        .unwrap_or_else(|| {
            panic!(
                "the outer runner lent no seed probe: {:?}",
                refit_with_probe.err()
            )
        })
        .unwrap_or_else(|reason| panic!("the seed probe refused: {reason}"));
    let mut graded_on_face = 0usize;
    for (point, coordinate, analytic, difference, settle, on_face) in &grades {
        eprintln!(
            "[2894-WIGGLE-FACE] point={point} coordinate={coordinate} analytic={analytic:.9e} \
             difference={difference:.9e} settle={settle:e} on_face={on_face}"
        );
        let scale = analytic.abs().max(1.0);
        if !on_face || *settle > 1.0e-4 * scale {
            continue;
        }
        graded_on_face += 1;
        assert!(
            (analytic - difference).abs() <= 1.0e-4 * scale + 10.0 * settle,
            "{point} coordinate {coordinate}: analytic {analytic:e} vs central difference {difference:e} on an active face"
        );
    }
    assert!(
        graded_on_face >= 1,
        "no coordinate was graded on a stable active face: {grades:?}"
    );
}

#[test]
fn marginal_slope_base_link_accepts_only_probit() {
    let parsed = gam_terms::inference::formula_dsl::parse_formula("y ~ x + link(type=probit)")
        .expect("main formula");
    let (resolved, _) = super::marginal_slope::resolve_marginal_slope_link(
        parsed.linkspec.as_ref(),
        None,
        false,
        "bernoulli marginal-slope",
    )
    .expect("explicit probit base link");
    assert_eq!(resolved, InverseLink::Standard(StandardLink::Probit));

    for formula in [
        "y ~ x + link(type=logit)",
        "y ~ x + link(type=sas, sas_init=\"0.1,-0.2\")",
        "y ~ x + link(type=beta-logistic, beta_logistic_init=\"0.3,0.7\")",
        "y ~ x + link(type=blended(logit,probit,cloglog), rho=\"0.4,-0.1\")",
        "y ~ x + link(type=flexible(logit))",
        "y ~ x + link(type=log)",
    ] {
        let parsed =
            gam_terms::inference::formula_dsl::parse_formula(formula).expect("main formula");
        let err = super::marginal_slope::resolve_marginal_slope_link(
            parsed.linkspec.as_ref(),
            None,
            false,
            "bernoulli marginal-slope",
        )
        .expect_err("non-probit marginal-slope link should be rejected");
        assert!(
            matches!(
                err,
                WorkflowError::MarginalSlopeLink {
                    context: "bernoulli marginal-slope",
                    refusal: MarginalSlopeLinkRefusal::NonProbit,
                }
            ),
            "unexpected error for {formula}: {err:?}"
        );
    }
}

/// gam#2999: the request's `link` and `flexible_link` arguments (gamfit's `link=` and
/// `flexible_link=`) are read, never dropped. A non-probit link named there is refused
/// by the argument's name, and a flexible probit link, in either place, comes back as
/// a flexible choice for the default link deviation.
#[test]
fn marginal_slope_link_arguments_are_honoured_or_refused_by_name_2999() {
    use gam_terms::inference::formula_dsl::LinkMode;
    let resolve = |formula: &str, link: Option<&str>, flexible_link: bool| {
        let parsed =
            gam_terms::inference::formula_dsl::parse_formula(formula).expect("main formula");
        super::marginal_slope::resolve_marginal_slope_link(
            parsed.linkspec.as_ref(),
            link,
            flexible_link,
            "bernoulli marginal-slope",
        )
    };
    let mode = |formula: &str, link: Option<&str>, flexible_link: bool| {
        let (base, choice) = resolve(formula, link, flexible_link)
            .unwrap_or_else(|err| panic!("{formula} link={link:?} flexible_link={flexible_link}: {err}"));
        assert_eq!(base, InverseLink::Standard(StandardLink::Probit));
        choice.map(|choice| {
            assert_eq!(choice.link, LinkFunction::Probit);
            matches!(choice.mode, LinkMode::Flexible)
        })
    };
    assert_eq!(mode("y ~ x", None, false), None);
    assert_eq!(mode("y ~ x", Some("probit"), false), Some(false));
    assert_eq!(mode("y ~ x + link(type=probit)", Some("probit"), false), Some(false));
    for (formula, link, flexible_link) in [
        ("y ~ x", None, true),
        ("y ~ x", Some("probit"), true),
        ("y ~ x", Some("flexible(probit)"), false),
        ("y ~ x + link(type=flexible(probit))", None, false),
        ("y ~ x + link(type=probit)", None, true),
        ("y ~ x + link(type=probit)", Some("flexible(probit)"), false),
    ] {
        assert_eq!(
            mode(formula, link, flexible_link),
            Some(true),
            "{formula} link={link:?} flexible_link={flexible_link} must ask for the link deviation"
        );
    }
    for (formula, link, flexible_link) in [
        ("y ~ x", "logit", false),
        ("y ~ x", "cloglog", false),
        ("y ~ x", "sas", false),
        ("y ~ x", "cauchit", false),
        ("y ~ x", "flexible(logit)", false),
        ("y ~ x", "logit", true),
        ("y ~ x + link(type=probit)", "logit", false),
    ] {
        let err = resolve(formula, Some(link), flexible_link)
            .expect_err("a non-probit link argument must be refused");
        assert!(
            matches!(
                &err,
                WorkflowError::MarginalSlopeLink {
                    context: "bernoulli marginal-slope",
                    refusal: MarginalSlopeLinkRefusal::NonProbitArgument { link: named },
                } if named == link
            ),
            "{formula} link={link}: {err:?}"
        );
        assert!(
            err.to_string().contains(&format!("the link argument names '{link}'")),
            "the refusal must name the argument and its value: {err}"
        );
    }
}

/// gam#2999: through materialization, `flexible_link=True` and `link="flexible(probit)"`
/// give the Bernoulli marginal-slope fit the link deviation the formula's `linkwiggle()`
/// gives, an explicit `linkwiggle(...)` still wins, and `link="logit"` is refused.
#[test]
fn bernoulli_marginal_slope_reads_the_link_arguments_2999() {
    let data = workflow_test_dataset();
    let link_dev = |formula: &str, link: Option<&str>, flexible_link: bool| {
        let config = FitConfig {
            slope_formula: Some("1".to_string()),
            z_column: Some("z".to_string()),
            link: link.map(str::to_string),
            flexible_link,
            ..FitConfig::default()
        };
        let materialized = materialize(formula, &data, &config).unwrap_or_else(|err| {
            panic!("{formula} link={link:?} flexible_link={flexible_link}: {err}")
        });
        let FitRequest::BernoulliMarginalSlope(request) = materialized.request else {
            panic!("expected a Bernoulli marginal-slope request");
        };
        request.spec.link_dev.map(|config| format!("{config:?}"))
    };
    let formula_default = link_dev("event ~ bmi + linkwiggle()", None, false)
        .expect("linkwiggle() builds the link deviation");
    assert_eq!(link_dev("event ~ bmi", None, false), None);
    assert_eq!(link_dev("event ~ bmi", Some("probit"), false), None);
    for (link, flexible_link) in [(None, true), (Some("flexible(probit)"), false)] {
        assert_eq!(
            link_dev("event ~ bmi", link, flexible_link).as_deref(),
            Some(formula_default.as_str()),
            "link={link:?} flexible_link={flexible_link} must build linkwiggle()'s block"
        );
    }
    let explicit = "event ~ bmi + linkwiggle(degree=3, internal_knots=9, penalty_order=\"1\")";
    assert_eq!(
        link_dev(explicit, None, true),
        link_dev(explicit, None, false),
        "an explicit linkwiggle(...) wins over flexible_link"
    );
    let config = FitConfig {
        slope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let err = match materialize("event ~ bmi", &data, &config) {
        Ok(_) => panic!("link=\"logit\" must be refused"),
        Err(err) => err,
    };
    assert!(
        matches!(
            &err,
            WorkflowError::MarginalSlopeLink {
                context: "bernoulli marginal-slope",
                refusal: MarginalSlopeLinkRefusal::NonProbitArgument { link },
            } if link == "logit"
        ),
        "link=\"logit\" must be the named link-argument refusal, got {err:?}"
    );
}

/// gam#2999: survival marginal-slope reads the link arguments too. Its main-formula
/// `linkwiggle()` is its link deviation, so a flexible link builds that block, and
/// `link="logit"` is refused by the argument's name.
#[test]
fn survival_marginal_slope_reads_the_link_arguments_2999() {
    let data = workflow_test_dataset();
    let link_dev = |formula: &str, link: Option<&str>, flexible_link: bool| {
        let config = FitConfig {
            survival_likelihood: Some("marginal-slope".to_string()),
            slope_formula: Some("1".to_string()),
            z_column: Some("z".to_string()),
            link: link.map(str::to_string),
            flexible_link,
            ..FitConfig::default()
        };
        let materialized = materialize(formula, &data, &config).unwrap_or_else(|err| {
            panic!("{formula} link={link:?} flexible_link={flexible_link}: {err}")
        });
        let FitRequest::SurvivalMarginalSlope(request) = materialized.request else {
            panic!("expected a survival marginal-slope request");
        };
        request.spec.link_dev.map(|config| format!("{config:?}"))
    };
    let formula = "Surv(age_entry, age_exit, event) ~ bmi";
    let formula_default = link_dev(&format!("{formula} + linkwiggle()"), None, false)
        .expect("linkwiggle() builds the link deviation");
    assert_eq!(link_dev(formula, None, false), None);
    assert_eq!(link_dev(formula, Some("probit"), false), None);
    for (link, flexible_link) in [(None, true), (Some("flexible(probit)"), false)] {
        assert_eq!(
            link_dev(formula, link, flexible_link).as_deref(),
            Some(formula_default.as_str()),
            "link={link:?} flexible_link={flexible_link} must build linkwiggle()'s block"
        );
    }

    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        slope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let err = match materialize("Surv(age_entry, age_exit, event) ~ bmi", &data, &config) {
        Ok(_) => panic!("link=\"logit\" must be refused on survival marginal-slope"),
        Err(err) => err,
    };
    assert!(
        matches!(
            &err,
            WorkflowError::MarginalSlopeLink {
                context: "survival marginal-slope",
                refusal: MarginalSlopeLinkRefusal::NonProbitArgument { link },
            } if link == "logit"
        ),
        "link=\"logit\" must be the named link-argument refusal, got {err:?}"
    );
}

/// #2677 B0: every materialized request whose custom-family solver options come
/// from `blockwise_fit_options` computes the conditional covariance unless the
/// caller declines it.
///
/// The latent survival and latent binary requests used to spread
/// `BlockwiseFitOptions::default()`, whose `compute_covariance` is `false`, and
/// never read `FitConfig::compute_covariance`. `compute_joint_posterior`
/// publishes the conditional covariance only under that flag, so a latent fit
/// whose cone moments were available (probe 1201712: moment status
/// `Available`) published none. Every builder the resolver serves is covered,
/// so a builder that bypasses it shows here. Materializing these requests runs
/// no fit (#2714 moved the latent baseline chart into the fit), so this reads
/// the request itself.
#[test]
fn materialized_requests_carry_the_callers_covariance_request_2677() {
    use crate::fit_orchestration::request::FitRequest;
    use crate::survival::lognormal_kernel::{FrailtyScale, FrailtySpec, HazardLoading};

    let carried = |label: &str, request: &FitRequest<'_>| match request {
        FitRequest::SurvivalMarginalSlope(request) => request.options.compute_covariance,
        FitRequest::LatentSurvival(request) => request.options.compute_covariance,
        FitRequest::LatentBinary(request) => request.options.compute_covariance,
        FitRequest::BernoulliMarginalSlope(request) => request.options.compute_covariance,
        FitRequest::TransformationNormal(request) => request.options.compute_covariance,
        _ => panic!("{label} must materialize its own custom-family request"),
    };
    let requests = [(None, true), (Some(false), false), (Some(true), true)];

    let workflow = workflow_test_dataset();
    for (label, formula, family, slope_formula, z_column) in [
        ("bernoulli marginal-slope", "event ~ bmi", None, Some("1"), Some("z")),
        ("transformation-normal", "bmi ~ s(age_entry, k=4)", Some("transformation-normal"), None, None),
    ] {
        for (requested, expected) in requests {
            let config = FitConfig {
                family: family.map(str::to_string),
                slope_formula: slope_formula.map(str::to_string),
                z_column: z_column.map(str::to_string),
                compute_covariance: requested,
                ..FitConfig::default()
            };
            let materialized = materialize(formula, &workflow, &config)
                .unwrap_or_else(|error| panic!("{label} should materialize: {error}"));
            assert_eq!(
                carried(label, &materialized.request),
                expected,
                "#2677 B0: {label} with compute_covariance={requested:?} must carry \
                 compute_covariance={expected} to the fit"
            );
        }
    }

    let td = tempdir().expect("tempdir");
    let data_path = td.path().join("survival_covariance_request_2677.csv");
    fs::write(
        &data_path,
        "entry,exit,event,x,z\n\
         0.0,0.4,1,-0.9,0.3\n\
         0.0,0.7,0,-0.6,-1.1\n\
         0.0,0.9,1,-0.3,0.8\n\
         0.0,1.2,1,-0.1,-0.4\n\
         0.0,1.5,0,0.2,1.3\n\
         0.0,1.8,1,0.4,-0.7\n\
         0.0,2.2,0,0.6,0.1\n\
         0.0,2.6,1,0.8,-1.5\n\
         0.0,3.1,1,0.9,0.6\n\
         0.0,3.7,0,-0.4,-0.2\n\
         0.0,4.2,1,0.1,1.0\n\
         0.0,4.8,0,-0.7,-0.9\n",
    )
    .expect("write survival covariance request csv");
    let data = load_dataset_projected(
        &data_path,
        &[
            "entry".to_string(),
            "exit".to_string(),
            "event".to_string(),
            "x".to_string(),
            "z".to_string(),
        ],
    )
    .expect("load survival covariance request dataset");

    for mode in ["marginal-slope", "latent", "latent-binary"] {
        for (requested, expected) in requests {
            let config = if mode == "marginal-slope" {
                FitConfig {
                    survival_likelihood: Some(mode.to_string()),
                    z_column: Some("z".to_string()),
                    compute_covariance: requested,
                    ..FitConfig::default()
                }
            } else {
                FitConfig {
                    survival_likelihood: Some(mode.to_string()),
                    baseline_target: "weibull".to_string(),
                    frailty: FrailtySpec::HazardMultiplier {
                        scale: FrailtyScale::Fixed { sigma: 0.5 },
                        loading: HazardLoading::Full,
                    },
                    compute_covariance: requested,
                    ..FitConfig::default()
                }
            };
            let materialized = materialize("Surv(entry, exit, event) ~ x", &data, &config)
                .unwrap_or_else(|error| panic!("{mode} should materialize: {error}"));
            assert_eq!(
                carried(mode, &materialized.request),
                expected,
                "#2677 B0: survival {mode} with compute_covariance={requested:?} must carry \
                 compute_covariance={expected} to the fit"
            );
        }
    }
}

/// #2937: a survival marginal-slope, latent or latent-binary fit refused by its
/// own input validation raises that category through `fit_model`. All three
/// routes handed back text, which `fit_model` recorded as
/// `FitFailure::Unclassified`, so Python raised the bare `FitError`.
#[test]
fn survival_marginal_slope_and_latent_refusals_raise_their_category_2937() {
    use crate::fit_orchestration::request::FitRequest;
    use crate::survival::lognormal_kernel::{FrailtyScale, FrailtySpec, HazardLoading};

    let td = tempdir().expect("tempdir");
    let data_path = td.path().join("survival_refusal_category_2937.csv");
    fs::write(
        &data_path,
        "entry,exit,event,x,z\n\
         0.0,0.4,1,-0.9,0.3\n\
         0.0,0.7,0,-0.6,-1.1\n\
         0.0,0.9,1,-0.3,0.8\n\
         0.0,1.2,1,-0.1,-0.4\n\
         0.0,1.5,0,0.2,1.3\n\
         0.0,1.8,1,0.4,-0.7\n\
         0.0,2.2,0,0.6,0.1\n\
         0.0,2.6,1,0.8,-1.5\n\
         0.0,3.1,1,0.9,0.6\n\
         0.0,3.7,0,-0.4,-0.2\n\
         0.0,4.2,1,0.1,1.0\n\
         0.0,4.8,0,-0.7,-0.9\n",
    )
    .expect("write survival refusal category csv");
    let data = load_dataset_projected(
        &data_path,
        &[
            "entry".to_string(),
            "exit".to_string(),
            "event".to_string(),
            "x".to_string(),
            "z".to_string(),
        ],
    )
    .expect("load survival refusal category dataset");

    for mode in ["marginal-slope", "latent", "latent-binary"] {
        let config = if mode == "marginal-slope" {
            FitConfig {
                survival_likelihood: Some(mode.to_string()),
                z_column: Some("z".to_string()),
                ..FitConfig::default()
            }
        } else {
            FitConfig {
                survival_likelihood: Some(mode.to_string()),
                baseline_target: "weibull".to_string(),
                frailty: FrailtySpec::HazardMultiplier {
                    scale: FrailtyScale::Fixed { sigma: 0.5 },
                    loading: HazardLoading::Full,
                },
                ..FitConfig::default()
            }
        };
        let mut request = materialize("Surv(entry, exit, event) ~ x", &data, &config)
            .unwrap_or_else(|error| panic!("{mode} should materialize: {error}"))
            .request;
        // A negative prior weight, which each route's own validator refuses.
        match &mut request {
            FitRequest::SurvivalMarginalSlope(request) => request.spec.weights[0] = -1.0,
            FitRequest::LatentSurvival(request) => request.spec.weights[0] = -1.0,
            FitRequest::LatentBinary(request) => request.spec.weights[0] = -1.0,
            _ => panic!("{mode} must materialize its own request"),
        }
        let err = match fit_model(request) {
            Ok(_) => panic!("#2937: {mode} must refuse a negative prior weight"),
            Err(err) => err,
        };
        assert_eq!(
            err.failure_category(),
            gam_problem::FailureCategory::Input,
            "{mode}: {err}"
        );
        assert_eq!(err.variant_name(), "FitFailure::Input", "{mode}: {err}");
    }
}

/// PKG-10: one predicate names the multinomial-logit family for the CLI, the
/// Python `fit_table` entry and the latent fitters, and the scalar resolver
/// refuses exactly those names.
#[test]
fn multinomial_family_names_are_one_predicate() {
    for name in [
        "multinomial",
        "Multinomial_Logit",
        "categorical",
        "categorical-logit",
        "SOFTMAX",
    ] {
        assert!(is_multinomial_family_name(name), "{name}");
        assert!(
            scalar_family_from_name(name, FamilyNuisanceOverrides::default()).is_err(),
            "{name}"
        );
    }
    for name in ["binomial", "gaussian", "poisson", "ordinal", "auto"] {
        assert!(!is_multinomial_family_name(name), "{name}");
    }
}

/// gam#3014: on the standard, location-scale and survival paths every link spelling
/// is read. `flexible_link` flexes the formula's `link(...)` as it flexes a `link`
/// argument, a `flexible(...)` in either place makes the choice flexible, and a `link`
/// argument naming a different base link from the formula's is refused by name.
#[test]
fn every_link_spelling_is_read_and_a_disagreeing_link_argument_is_refused_3014() {
    use gam_terms::inference::formula_dsl::LinkMode;
    let resolve = |formula: &str, link: Option<&str>, flexible_link: bool| {
        let parsed =
            gam_terms::inference::formula_dsl::parse_formula(formula).expect("main formula");
        super::validation::resolve_link_spellings(parsed.linkspec.as_ref(), link, flexible_link)
    };
    let choice = |formula: &str, link: Option<&str>, flexible_link: bool| {
        resolve(formula, link, flexible_link)
            .unwrap_or_else(|err| {
                panic!("{formula} link={link:?} flexible_link={flexible_link}: {err}")
            })
            .map(|choice| (choice.link, matches!(choice.mode, LinkMode::Flexible)))
    };
    assert_eq!(choice("y ~ x", None, false), None);
    assert_eq!(choice("y ~ x", None, true), Some((LinkFunction::Probit, true)));
    for (formula, link, flexible_link, expected) in [
        ("y ~ x + link(type=probit)", None, false, (LinkFunction::Probit, false)),
        ("y ~ x + link(type=probit)", None, true, (LinkFunction::Probit, true)),
        ("y ~ x + link(type=logit)", None, true, (LinkFunction::Logit, true)),
        ("y ~ x + link(type=probit)", Some("probit"), false, (LinkFunction::Probit, false)),
        ("y ~ x + link(type=probit)", Some("probit"), true, (LinkFunction::Probit, true)),
        (
            "y ~ x + link(type=probit)",
            Some("flexible(probit)"),
            false,
            (LinkFunction::Probit, true),
        ),
        (
            "y ~ x + link(type=flexible(probit))",
            Some("probit"),
            false,
            (LinkFunction::Probit, true),
        ),
        ("y ~ x", Some("logit"), false, (LinkFunction::Logit, false)),
        ("y ~ x", Some("logit"), true, (LinkFunction::Logit, true)),
    ] {
        assert_eq!(
            choice(formula, link, flexible_link),
            Some(expected),
            "{formula} link={link:?} flexible_link={flexible_link}"
        );
    }
    for (formula, link) in [
        ("y ~ x + link(type=probit)", "logit"),
        ("y ~ x + link(type=probit)", "flexible(logit)"),
        ("y ~ x + link(type=flexible(probit))", "cloglog"),
        ("y ~ x + link(type=logit)", "blended(logit,probit)"),
    ] {
        let err = resolve(formula, Some(link), false)
            .expect_err("a link argument that disagrees with the formula must be refused");
        let message = err.to_string();
        assert!(
            matches!(err, WorkflowError::InvalidConfig { .. })
                && message.contains("link(type=")
                && message.contains(&format!("link=\"{link}\"")),
            "{formula} link={link}: {message}"
        );
    }
    // A flexible request of a link the joint wiggle cannot flex is still refused,
    // now also when the link is named in the formula.
    assert!(resolve("y ~ x + link(type=sas)", None, true).is_err());
}
