use ndarray::Array2;

use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::{ParsedTerm, parse_formula};
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::resource::ResourcePolicy;
use gam::smooth::{ByVarKind, FactorSmoothFlavour, SmoothBasisSpec};
use gam::terms::term_builder::build_termspec;

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
