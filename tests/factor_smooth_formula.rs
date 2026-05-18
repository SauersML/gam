use ndarray::{Array1, Array2};

use gam::estimate::FitOptions;
use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::{ParsedTerm, parse_formula};
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::resource::ResourcePolicy;
use gam::smooth::{ByVarKind, FactorSmoothFlavour, SmoothBasisSpec};
use gam::terms::term_builder::build_termspec;
use gam::types::LikelihoodFamily;

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

fn medium_factor_dataset() -> (EncodedDataset, Array1<f64>) {
    // 60 rows with one continuous `x`, a numeric `z` covariate, and a
    // categorical `fac`. `y` is driven by a smooth in x modulated by z, with
    // a level-specific offset. Enough rows to fit a by-smooth through REML
    // without rank collapse.
    let n = 60usize;
    let mut values = Array2::<f64>::zeros((n, 4));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
        let fac = if i % 2 == 0 { 0.0 } else { 1.0 };
        let z = 0.4 + 0.5 * (i as f64 / (n - 1) as f64);
        values[[i, 0]] = 0.5 + 0.3 * t + 0.2 * t * t - 0.4 * fac * t; // y placeholder
        values[[i, 1]] = t;
        values[[i, 2]] = fac;
        values[[i, 3]] = z;
        y[i] = values[[i, 0]];
    }
    (
        EncodedDataset {
            headers: vec!["y".into(), "x".into(), "fac".into(), "z".into()],
            values,
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
        },
        y,
    )
}

#[test]
fn factor_by_smooth_fits_end_to_end_through_reml() {
    let (ds, y) = medium_factor_dataset();
    let cmap = ds.column_map();
    let mut notes = Vec::new();
    let parsed = parse_formula("y ~ s(x, by=fac, k=4) + fac").unwrap();
    let spec = build_termspec(
        &parsed.terms,
        &ds,
        &cmap,
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .expect("term spec should build for s(x, by=fac) + fac");

    let weights = Array1::<f64>::ones(y.len());
    let offset = Array1::<f64>::zeros(y.len());
    let options = FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        max_iter: 60,
        tol: 1e-6,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let fitted = gam::smooth::fit_term_collection_forspec(
        ds.values.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        LikelihoodFamily::GaussianIdentity,
        &options,
    )
    .expect("factor-by smooth fit should succeed via the engine");
    assert!(fitted.fit.beta.iter().all(|v: &f64| v.is_finite()));
    assert!(fitted.fit.beta.len() >= 2);
    assert!(fitted.fit.deviance.is_finite());
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
