use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::{ParsedTerm, parse_formula};
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::resource::ResourcePolicy;
use gam::smooth::{ByVarKind, FactorSmoothFlavour, SmoothBasisSpec, build_term_collection_design};
use gam::term_builder::build_termspec;
use ndarray::Array2;

fn ds() -> EncodedDataset {
    let n = 24;
    let mut values = Array2::<f64>::zeros((n, 5));
    for i in 0..n {
        values[[i, 0]] = i as f64 / (n as f64 - 1.0); // x
        values[[i, 1]] = (i % 3) as f64; // fac
        values[[i, 2]] = if i % 2 == 0 { 0.5 } else { 1.5 }; // z
        values[[i, 3]] = (i % 3) as f64; // ord_fac
        values[[i, 4]] = (n - i) as f64 / n as f64; // x2
    }
    let headers = vec!["x", "fac", "z", "ord_fac", "x2"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let kinds = vec![
        ColumnKindTag::Continuous,
        ColumnKindTag::Categorical,
        ColumnKindTag::Continuous,
        ColumnKindTag::Categorical,
        ColumnKindTag::Continuous,
    ];
    EncodedDataset {
        headers: headers.clone(),
        values,
        schema: DataSchema {
            columns: headers
                .iter()
                .zip(kinds.iter())
                .map(|(name, kind)| SchemaColumn {
                    name: name.clone(),
                    kind: *kind,
                    levels: vec![],
                })
                .collect(),
        },
        column_kinds: kinds,
    }
}

#[test]
fn factor_smooth_aliases_parse_to_options() {
    for (formula, bs) in [
        ("y ~ s(x, fac, bs=\"fs\")", "fs"),
        ("y ~ fs(x, fac)", "fs"),
        ("y ~ s(x, fac, bs=fs, k=10)", "fs"),
        ("y ~ s(x, fac, bs=\"sz\")", "sz"),
        ("y ~ sz(x, fac)", "sz"),
    ] {
        let parsed = parse_formula(formula).unwrap_or_else(|e| panic!("{formula}: {e}"));
        let ParsedTerm::Smooth { options, .. } = &parsed.terms[0] else {
            panic!("expected smooth")
        };
        assert_eq!(options.get("bs").map(String::as_str), Some(bs));
    }
    for formula in ["y ~ s(x, by=fac)", "y ~ s(x, by=z, k=8)"] {
        let parsed = parse_formula(formula).unwrap_or_else(|e| panic!("{formula}: {e}"));
        let ParsedTerm::Smooth { options, .. } = &parsed.terms[0] else {
            panic!("expected smooth")
        };
        assert!(options.contains_key("by"));
    }
}

#[test]
fn by_fs_sz_and_random_slope_build_termspec() {
    let data = ds();
    let col_map = data.column_map();
    let policy = ResourcePolicy::default_library();
    for formula in [
        "y ~ s(x, by=z)",
        "y ~ s(x, by=fac) + fac",
        "y ~ s(x) + s(x, by=ord_fac)",
        "y ~ s(x, fac, bs=\"fs\")",
        "y ~ s(x, fac, bs=\"fs\", m=1, k=10)",
        "y ~ fs(x, fac)",
        "y ~ s(fac, x, bs=\"sz\") + s(x)",
        "y ~ sz(fac, x)",
        "y ~ s(fac, x, bs=\"re\") + group(fac)",
        "y ~ te(x, x2, by=fac)",
        "y ~ s(x, by=fac, id=1)",
        "y ~ s(x) + s(x2, fac, bs=\"fs\", k=5) + s(z, k=20)",
    ] {
        let parsed = parse_formula(formula).unwrap_or_else(|e| panic!("parse {formula}: {e}"));
        let mut notes = Vec::new();
        let spec = build_termspec(&parsed.terms, &data, &col_map, &mut notes, &policy)
            .unwrap_or_else(|e| panic!("build {formula}: {e}"));
        let design = build_term_collection_design(data.values.view(), &spec)
            .unwrap_or_else(|e| panic!("design {formula}: {e}"));
        assert_eq!(design.design.nrows(), data.values.nrows(), "{formula}");
    }
}

#[test]
fn termspec_routes_new_constructs_to_new_variants() {
    let data = ds();
    let col_map = data.column_map();
    let policy = ResourcePolicy::default_library();

    let parsed = parse_formula("y ~ s(x, by=z)").unwrap();
    let spec = build_termspec(&parsed.terms, &data, &col_map, &mut vec![], &policy).unwrap();
    assert!(matches!(
        spec.smooth_terms[0].basis,
        SmoothBasisSpec::BySmooth {
            by_kind: ByVarKind::Numeric { .. },
            ..
        }
    ));

    let parsed = parse_formula("y ~ s(x, by=fac)").unwrap();
    let spec = build_termspec(&parsed.terms, &data, &col_map, &mut vec![], &policy).unwrap();
    assert!(matches!(
        spec.smooth_terms[0].basis,
        SmoothBasisSpec::BySmooth {
            by_kind: ByVarKind::Factor { .. },
            ..
        }
    ));

    let parsed = parse_formula("y ~ fs(x, fac)").unwrap();
    let spec = build_termspec(&parsed.terms, &data, &col_map, &mut vec![], &policy).unwrap();
    assert!(
        matches!(spec.smooth_terms[0].basis, SmoothBasisSpec::FactorSmooth { ref spec } if matches!(spec.flavour, FactorSmoothFlavour::Fs { .. }))
    );
}
