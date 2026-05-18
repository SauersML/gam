use ndarray::Array2;

use csv::StringRecord;
use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::{ParsedTerm, parse_formula};
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::resource::ResourcePolicy;
use gam::smooth::{ByVarKind, FactorSmoothFlavour, SmoothBasisSpec};
use gam::terms::term_builder::build_termspec;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

#[test]
fn factor_smooth_aliases_and_by_options_parse() {
    let cases = [
        ("y ~ s(x, fac, bs=\"fs\")", Some("fs"), vec!["x", "fac"]),
        ("y ~ fs(x, fac)", Some("fs"), vec!["x", "fac"]),
        ("y ~ s(x, fac, bs=fs, k=10)", Some("fs"), vec!["x", "fac"]),
        ("y ~ s(x, fac, bs=\"sz\")", Some("sz"), vec!["x", "fac"]),
        ("y ~ sz(x, fac)", Some("sz"), vec!["x", "fac"]),
        ("y ~ s(x, by=fac)", None, vec!["x"]),
        ("y ~ s(x, by=group, k=8)", None, vec!["x"]),
    ];
    for (formula, bs, expected_vars) in cases {
        let parsed = parse_formula(formula).unwrap_or_else(|e| panic!("{formula}: {e}"));
        let ParsedTerm::Smooth { vars, options, .. } = &parsed.terms[0] else {
            panic!("{formula} did not parse to a smooth");
        };
        assert_eq!(
            vars,
            &expected_vars
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        );
        if let Some(bs) = bs {
            assert_eq!(options.get("bs").map(String::as_str), Some(bs));
        } else {
            assert!(options.contains_key("by"));
        }
    }
}

fn tiny_dataset() -> EncodedDataset {
    let rows = vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.2, 0.0, 2.0],
        [0.0, 0.4, 1.0, 1.0],
        [1.0, 0.6, 1.0, 2.0],
        [0.0, 0.8, 0.0, 1.0],
        [1.0, 1.0, 1.0, 2.0],
    ];
    EncodedDataset {
        headers: vec!["y".into(), "x".into(), "fac".into(), "z".into()],
        values: Array2::from_shape_vec((rows.len(), 4), rows.into_iter().flatten().collect())
            .unwrap(),
        schema: DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "y".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "x".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "fac".into(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["a".into(), "b".into()],
                },
                SchemaColumn {
                    name: "z".into(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Categorical,
            ColumnKindTag::Continuous,
        ],
    }
}

#[test]
fn factor_smooth_forms_route_to_new_termspec_variants() {
    let ds = tiny_dataset();
    let cmap = ds.column_map();
    let mut notes = Vec::new();

    let parsed = parse_formula("y ~ s(x, by=fac) + fac").unwrap();
    let spec = build_termspec(
        &parsed.terms,
        &ds,
        &cmap,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .unwrap();
    match &spec.smooth_terms[0].basis {
        SmoothBasisSpec::BySmooth {
            by_kind: ByVarKind::Factor { frozen_levels, .. },
            ..
        } => {
            assert_eq!(frozen_levels.as_ref().unwrap().len(), 2);
        }
        other => panic!("expected factor by smooth, got {other:?}"),
    }

    let parsed = parse_formula("y ~ s(x, by=z)").unwrap();
    let spec = build_termspec(
        &parsed.terms,
        &ds,
        &cmap,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .unwrap();
    assert!(matches!(
        spec.smooth_terms[0].basis,
        SmoothBasisSpec::BySmooth {
            by_kind: ByVarKind::Numeric { .. },
            ..
        }
    ));

    let parsed = parse_formula("y ~ fs(x, fac, k=5)").unwrap();
    let spec = build_termspec(
        &parsed.terms,
        &ds,
        &cmap,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .unwrap();
    assert!(matches!(
        spec.smooth_terms[0].basis,
        SmoothBasisSpec::FactorSmooth {
            spec: gam::smooth::FactorSmoothSpec {
                flavour: FactorSmoothFlavour::Fs { .. },
                ..
            }
        }
    ));

    let parsed = parse_formula("y ~ sz(fac, x, k=5)").unwrap();
    let spec = build_termspec(
        &parsed.terms,
        &ds,
        &cmap,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .unwrap();
    assert!(matches!(
        spec.smooth_terms[0].basis,
        SmoothBasisSpec::FactorSmooth {
            spec: gam::smooth::FactorSmoothSpec {
                flavour: FactorSmoothFlavour::Sz,
                ..
            }
        }
    ));

    let parsed = parse_formula("y ~ s(fac, x, bs=re, k=5)").unwrap();
    let spec = build_termspec(
        &parsed.terms,
        &ds,
        &cmap,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .unwrap();
    assert!(matches!(
        spec.smooth_terms[0].basis,
        SmoothBasisSpec::FactorSmooth {
            spec: gam::smooth::FactorSmoothSpec {
                flavour: FactorSmoothFlavour::Re,
                ..
            }
        }
    ));
}

fn factor_by_dataset(n: usize) -> EncodedDataset {
    // n rows split evenly across two levels of `fac` with `y` driven by a
    // level-dependent quadratic in `x` plus mild noise. Enough rows to fit
    // a factor-by smooth + main effect through PIRLS/REML.
    let headers: Vec<String> = ["y", "x", "fac", "z"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let t = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
            let fac = i % 2;
            let z = 0.4 + 0.5 * (i as f64 / (n - 1) as f64);
            let y = 0.5 + 0.3 * t + 0.2 * t * t - 0.4 * (fac as f64) * t
                + 0.05 * ((7 * i + 3) as f64 / 11.0).sin();
            StringRecord::from(vec![
                y.to_string(),
                t.to_string(),
                fac.to_string(),
                z.to_string(),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode rows")
}

fn fit_succeeds(formula: &str, ds: &EncodedDataset) -> usize {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, ds, &cfg)
        .unwrap_or_else(|e| panic!("{formula} failed to fit: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("{formula}: expected standard fit");
    };
    assert!(
        fit.fit.beta.iter().all(|v: &f64| v.is_finite()),
        "{formula}: beta has non-finite entries"
    );
    assert!(
        fit.fit.deviance.is_finite(),
        "{formula}: deviance is non-finite"
    );
    fit.fit.beta.len()
}

#[test]
fn factor_smooth_variants_fit_end_to_end_through_reml() {
    let ds = factor_by_dataset(200);
    // Numeric by= smooth — basis multiplied by the numeric covariate.
    assert!(fit_succeeds("y ~ s(x, by=z, k=6)", &ds) > 1);
    // Factor by= smooth alone (per-level smooth, no main effect).
    assert!(fit_succeeds("y ~ s(x, by=fac, k=4)", &ds) > 1);
    // Factor by= smooth combined with a random-effect main effect — the
    // canonical mgcv idiom for identifiable per-group trajectories.
    assert!(fit_succeeds("y ~ s(x, by=fac, k=4) + fac", &ds) > 1);
    // Hierarchical/partial-pooling factor smooth (bs="fs").
    assert!(fit_succeeds("y ~ fs(x, fac, k=4)", &ds) > 1);
    // Sum-to-zero deviations from a population smooth.
    assert!(fit_succeeds("y ~ s(x, k=4) + sz(fac, x, k=4)", &ds) > 1);
}

#[test]
fn new_factor_smooth_terms_build_designs() {
    let ds = tiny_dataset();
    let cmap = ds.column_map();
    for formula in [
        "y ~ s(x, by=fac) + fac",
        "y ~ s(x, by=z)",
        "y ~ fs(x, fac, k=5)",
        "y ~ sz(fac, x, k=5)",
        "y ~ s(fac, x, bs=re, k=5) + group(fac)",
    ] {
        let parsed = parse_formula(formula).unwrap();
        let mut notes = Vec::new();
        let spec = build_termspec(
            &parsed.terms,
            &ds,
            &cmap,
            &mut notes,
            &ResourcePolicy::default_library(),
        )
        .unwrap_or_else(|e| panic!("{formula}: {e}"));
        let design = gam::smooth::build_term_collection_design(ds.values.view(), &spec)
            .unwrap_or_else(|e| panic!("{formula} design build failed: {e}"));
        assert_eq!(design.design.nrows(), ds.values.nrows(), "{formula}");
        assert!(design.design.ncols() > 0, "{formula}");
    }
}
