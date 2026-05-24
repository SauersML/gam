use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::parse_formula;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::resource::ResourcePolicy;
use gam::smooth::{BySmoothKind, SmoothBasisSpec};
use gam::terms::term_builder::build_termspec;
use ndarray::array;

fn dataset() -> EncodedDataset {
    EncodedDataset {
        headers: vec!["y".into(), "x".into(), "group".into(), "treated".into()],
        values: array![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.2, 1.0, 1.0],
            [0.0, 0.4, 0.0, 0.0],
            [1.0, 0.6, 1.0, 1.0],
            [0.0, 0.8, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
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
                    name: "group".into(),
                    kind: ColumnKindTag::Categorical,
                    levels: vec!["A".into(), "B".into()],
                },
                SchemaColumn {
                    name: "treated".into(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
            ],
        },
        column_kinds: vec![
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
            ColumnKindTag::Categorical,
            ColumnKindTag::Binary,
        ],
    }
}

#[test]
fn unordered_by_factor_expands_to_level_smooths_and_fixed_main_effect() {
    let parsed = parse_formula("y ~ s(x, by=group, k=5)").expect("parse");
    let ds = dataset();
    let mut notes = Vec::new();
    let spec = build_termspec(
        &parsed.terms,
        &ds,
        &ds.column_map(),
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .expect("termspec");
    assert_eq!(spec.smooth_terms.len(), 2);
    assert!(
        spec.random_effect_terms
            .iter()
            .any(|term| term.name == "group" && !term.penalized && term.drop_first_level)
    );
    assert!(spec.smooth_terms.iter().all(|term| matches!(
        term.basis,
        SmoothBasisSpec::ByVariable {
            kind: BySmoothKind::Level { .. },
            ..
        }
    )));
}

#[test]
fn binary_by_and_sz_parse_to_specialized_basis_specs() {
    let ds = dataset();
    let mut notes = Vec::new();
    let parsed_binary = parse_formula("y ~ s(x, by=treated, k=5)").expect("parse binary");
    let binary = build_termspec(
        &parsed_binary.terms,
        &ds,
        &ds.column_map(),
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .expect("binary termspec");
    assert!(matches!(
        binary.smooth_terms[0].basis,
        SmoothBasisSpec::ByVariable {
            kind: BySmoothKind::Numeric,
            ..
        }
    ));

    let parsed_sz = parse_formula("y ~ s(x) + s(group, x, bs=sz, k=5)").expect("parse sz");
    let sz = build_termspec(
        &parsed_sz.terms,
        &ds,
        &ds.column_map(),
        &mut notes,
        &ResourcePolicy::default_library(),
    )
    .expect("sz termspec");
    assert!(sz.smooth_terms.iter().any(|term| matches!(term.basis, SmoothBasisSpec::FactorSumToZero { ref levels, .. } if levels.len() == 2)));
}
