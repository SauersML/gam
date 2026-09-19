use gam::inference::data::EncodedDataset;
use gam::inference::formula_dsl::parse_formula;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::smooth::{BySmoothKind, FactorSmoothFlavour, SmoothBasisSpec};
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
fn unordered_by_factor_expands_to_level_smooths_and_random_main_effect() {
    let parsed = parse_formula("y ~ s(x, by=group, k=5)").expect("parse");
    let ds = dataset();
    let mut notes = Vec::new();
    let spec = build_termspec(
        &parsed.terms,
        &ds,
        &ds.column_map(),
        &mut notes,
    )
    .expect("termspec");
    assert_eq!(spec.smooth_terms.len(), 2);
    // The auto-added factor main effect is the penalized full-level random block a
    // bare `+ group` lowers to (35c8b53864, SPEC rules 12 and 14), so REML can
    // shrink every level offset to the null.
    assert!(
        spec.random_effect_terms
            .iter()
            .any(|term| term.name == "group" && !term.carries_level),
        "`s(x, by=group)` must add `group` as a penalized full-level random block; got {:?}",
        spec.random_effect_terms
            .iter()
            .map(|term| (&term.name, term.carries_level))
            .collect::<Vec<_>>()
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
    )
    .expect("sz termspec");
    // `bs="sz"` types as `FactorSmooth { flavour: Sz }`, NOT the legacy
    // `FactorSumToZero` envelope this previously asserted. That envelope is the
    // mis-typed representation #1981 reintroduced and #1403 removed again
    // (owner-confirmed in #1887, and documented at the construction site in
    // term_builder.rs): re-wrapping `sz(fac, x)` into it broke the
    // refit/predict freeze path's `(FactorSmooth, …)` metadata match. The
    // zero-sum geometry is still the construction underneath — it is reused
    // internally — so what identifies the term is the flavour plus the factor it
    // is keyed to. Levels are not carried on the spec at PARSE time
    // (`group_frozen_levels` is populated at freeze), so the 2-level factor is
    // pinned by its column instead: `group` is index 2 of the fixture headers.
    assert!(
        sz.smooth_terms.iter().any(|term| matches!(
            term.basis,
            SmoothBasisSpec::FactorSmooth { ref spec }
                if matches!(spec.flavour, FactorSmoothFlavour::Sz) && spec.group_col == 2
        )),
        "`bs=sz` must parse to FactorSmooth {{ flavour: Sz }} keyed to `group`; got {:?}",
        sz.smooth_terms
            .iter()
            .map(|term| std::mem::discriminant(&term.basis))
            .collect::<Vec<_>>()
    );
}
